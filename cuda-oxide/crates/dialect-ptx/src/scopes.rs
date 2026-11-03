/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Fail-closed normalization plans for callable-local lexical scopes.
//!
//! Flattening keeps declarations and instructions at their original source
//! positions. Nested register bindings are first made callable-unique, then
//! only anonymous braces whose contents are understood are blanked in place.
//! This prevents neighboring tokens from fusing and preserves line structure.

use crate::registers::{RegisterAlphaError, RegisterAlphaPlan};
use crate::version::{PtxVersionError, validate_ptx_version};
use ptx_parse::{
    AppliedEdits, Document, EditError, EditScript, ScopeId, StatementId, StatementKind,
};
use std::collections::HashMap;
use std::fmt;
use std::ops::Range;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScopeFlattenPlan {
    register_alpha: RegisterAlphaPlan,
    scopes: Vec<FlattenedScope>,
}

impl ScopeFlattenPlan {
    pub fn analyze(document: &Document<'_>) -> Result<Self, ScopeFlattenError> {
        // Flattening removes lexical structure; gate it behind the same PTX
        // version ceiling as the CFG so an unaudited ISA never gets rewritten.
        validate_ptx_version(document).map_err(ScopeFlattenError::Version)?;
        let register_alpha =
            RegisterAlphaPlan::analyze(document).map_err(ScopeFlattenError::RegisterAlpha)?;
        let mut callable_by_scope = HashMap::new();
        for callable in document.callables() {
            let Some(scope_id) = callable.definition_scope() else {
                continue;
            };
            let scope = document
                .scope(scope_id)
                .expect("a callable definition scope comes from this document");
            if scope.open_span().is_none() || scope.close_span().is_none() {
                return Err(ScopeFlattenError::MissingDelimiter {
                    callable: callable.statement(),
                    scope: scope_id,
                });
            }
            callable_by_scope.insert(scope_id, callable.statement());
        }
        let mut owner_by_scope = vec![None; document.scopes().len()];
        for scope in document.scopes().iter().skip(1) {
            owner_by_scope[scope.id().index()] =
                callable_by_scope.get(&scope.id()).copied().or_else(|| {
                    scope
                        .parent()
                        .and_then(|parent| owner_by_scope[parent.index()])
                });
        }

        let mut scopes = Vec::new();
        for scope in document.scopes().iter().skip(1) {
            let Some(callable) = owner_by_scope[scope.id().index()] else {
                continue;
            };
            if callable_by_scope.contains_key(&scope.id()) {
                continue;
            }
            if let Some(header) = scope.header() {
                return Err(ScopeFlattenError::HeaderOwnedScope {
                    callable,
                    scope: scope.id(),
                    header,
                });
            }
            let Some(open_span) = scope.open_span() else {
                return Err(ScopeFlattenError::MissingDelimiter {
                    callable,
                    scope: scope.id(),
                });
            };
            let Some(close_span) = scope.close_span() else {
                return Err(ScopeFlattenError::MissingDelimiter {
                    callable,
                    scope: scope.id(),
                });
            };
            for statement in document.statements_in_scope(scope.id()) {
                match statement.kind() {
                    StatementKind::Instruction | StatementKind::Label => {}
                    StatementKind::Directive => {
                        let name = document
                            .directive_for_statement(statement.id())
                            .map(|directive| directive.name())
                            .unwrap_or("");
                        if !matches!(
                            directive_scope_effect(name),
                            DirectiveScopeEffect::RegisterBinding
                                | DirectiveScopeEffect::ScopeNeutral
                        ) {
                            return Err(ScopeFlattenError::UnsupportedDirective {
                                callable,
                                scope: scope.id(),
                                statement: statement.id(),
                                name: name.to_string(),
                            });
                        }
                    }
                    kind => {
                        return Err(ScopeFlattenError::UnsupportedStatement {
                            callable,
                            scope: scope.id(),
                            statement: statement.id(),
                            kind,
                        });
                    }
                }
            }
            scopes.push(FlattenedScope {
                callable,
                scope: scope.id(),
                open_span,
                close_span,
            });
        }
        Ok(Self {
            register_alpha,
            scopes,
        })
    }

    pub fn register_alpha(&self) -> &RegisterAlphaPlan {
        &self.register_alpha
    }

    pub fn scopes(&self) -> &[FlattenedScope] {
        &self.scopes
    }

    pub fn edit_script(&self) -> Result<EditScript, EditError> {
        let mut edits = EditScript::new();
        self.register_alpha.add_edits(&mut edits)?;
        for scope in &self.scopes {
            let open = scope.open_span();
            let close = scope.close_span();
            edits.replace(open.clone(), " ".repeat(open.len()))?;
            edits.replace(close.clone(), " ".repeat(close.len()))?;
        }
        Ok(edits)
    }

    pub fn apply(&self, source: &str) -> Result<String, EditError> {
        self.edit_script()?.apply(source)
    }

    pub fn apply_with_map(&self, source: &str) -> Result<AppliedEdits, EditError> {
        self.edit_script()?.apply_with_map(source)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DirectiveScopeEffect {
    RegisterBinding,
    ScopeNeutral,
    ScopeSensitive,
    Unknown,
}

fn directive_scope_effect(name: &str) -> DirectiveScopeEffect {
    match name {
        ".reg" => DirectiveScopeEffect::RegisterBinding,
        // .loc only sets debug source-location state for the instructions
        // that lexically follow it; braces never bound its effect, and
        // flattening leaves lexical order unchanged.
        ".loc" => DirectiveScopeEffect::ScopeNeutral,
        // Declarations and indexed-dispatch tables have lexical ownership.
        // .pragma is scope-sensitive too: pragma strings are tool-defined
        // with placement-dependent meanings (module, entry-function, or
        // statement level), and the flattener cannot know the semantics of
        // every string ptxas accepts. Deleting the braces around a
        // block-scoped pragma such as `.pragma "nounroll";` can widen its
        // effect to the whole callable, so refuse to flatten instead of
        // changing optimization scope.
        ".branchtargets" | ".const" | ".global" | ".local" | ".param" | ".pragma" | ".shared" => {
            DirectiveScopeEffect::ScopeSensitive
        }
        _ => DirectiveScopeEffect::Unknown,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlattenedScope {
    callable: StatementId,
    scope: ScopeId,
    open_span: Range<usize>,
    close_span: Range<usize>,
}

impl FlattenedScope {
    pub fn callable(&self) -> StatementId {
        self.callable
    }

    pub fn scope(&self) -> ScopeId {
        self.scope
    }

    pub fn open_span(&self) -> Range<usize> {
        self.open_span.clone()
    }

    pub fn close_span(&self) -> Range<usize> {
        self.close_span.clone()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScopeFlattenError {
    Version(PtxVersionError),
    RegisterAlpha(RegisterAlphaError),
    HeaderOwnedScope {
        callable: StatementId,
        scope: ScopeId,
        header: StatementId,
    },
    MissingDelimiter {
        callable: StatementId,
        scope: ScopeId,
    },
    UnsupportedDirective {
        callable: StatementId,
        scope: ScopeId,
        statement: StatementId,
        name: String,
    },
    UnsupportedStatement {
        callable: StatementId,
        scope: ScopeId,
        statement: StatementId,
        kind: StatementKind,
    },
}

impl fmt::Display for ScopeFlattenError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Version(error) => error.fmt(formatter),
            Self::RegisterAlpha(error) => error.fmt(formatter),
            Self::HeaderOwnedScope {
                callable,
                scope,
                header,
            } => write!(
                formatter,
                "PTX callable statement {} scope {} has non-anonymous header statement {}",
                callable.index(),
                scope.index(),
                header.index()
            ),
            Self::MissingDelimiter { callable, scope } => write!(
                formatter,
                "PTX callable statement {} scope {} is not closed by explicit delimiters",
                callable.index(),
                scope.index()
            ),
            Self::UnsupportedDirective {
                callable,
                scope,
                statement,
                name,
            } => write!(
                formatter,
                "PTX callable statement {} scope {} contains unsupported directive {name:?} at statement {}",
                callable.index(),
                scope.index(),
                statement.index()
            ),
            Self::UnsupportedStatement {
                callable,
                scope,
                statement,
                kind,
            } => write!(
                formatter,
                "PTX callable statement {} scope {} contains unsupported {kind:?} statement {}",
                callable.index(),
                scope.index(),
                statement.index()
            ),
        }
    }
}

impl std::error::Error for ScopeFlattenError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_renames_then_removes_only_nested_braces() {
        let source = "\
.version 9.3
.entry kernel() {
    .reg .b32 x;
    {
        .reg .b32 x;
        mov.u32 x, 1;
        {
            .reg .b32 y;
            add.u32 y, x, 1;
        }
    }
    mov.u32 x, 2;
    ret;
}
";
        let document = Document::parse(source).unwrap();
        let before_statements = document.statements().len();
        let before_instructions = document.instructions().len();
        let plan = ScopeFlattenPlan::analyze(&document).unwrap();
        assert_eq!(plan.scopes().len(), 2);
        assert_eq!(plan.register_alpha().callables()[0].renames().len(), 2);

        let rewritten = plan.apply(source).unwrap();
        assert_eq!(rewritten.lines().count(), source.lines().count());
        let reparsed = Document::parse(&rewritten).unwrap();
        assert_eq!(reparsed.scopes().len(), 2);
        assert_eq!(reparsed.statements().len(), before_statements);
        assert_eq!(reparsed.instructions().len(), before_instructions);
        assert!(rewritten.contains(".reg .b32 __oxide_s2_x;"));
        assert!(rewritten.contains("add.u32 __oxide_s3_y, __oxide_s2_x, 1;"));
        assert!(rewritten.contains("mov.u32 x, 2;"));
    }

    #[test]
    fn preserves_explicitly_scope_neutral_directives() {
        let source = "\
.version 9.3
.entry kernel() {
    {
        .loc 1 2 3;
        ret;
    }
}
";
        let document = Document::parse(source).unwrap();
        let rewritten = ScopeFlattenPlan::analyze(&document)
            .unwrap()
            .apply(source)
            .unwrap();
        assert_eq!(rewritten.matches('{').count(), 1);
        assert!(rewritten.contains(".loc 1 2 3;"));
    }

    #[test]
    fn rejects_pragma_directives_in_nested_scopes() {
        // A block-scoped `.pragma "nounroll";` covers only its block's loops;
        // flattening would widen it to the whole callable, so refuse.
        let source = "\
.version 9.3
.entry kernel() {
    {
        .pragma \"nounroll\";
        ret;
    }
}
";
        let document = Document::parse(source).unwrap();
        assert!(matches!(
            ScopeFlattenPlan::analyze(&document),
            Err(ScopeFlattenError::UnsupportedDirective { name, .. }) if name == ".pragma"
        ));
    }

    #[test]
    fn rejects_unknown_directives_in_nested_scopes() {
        let source = ".version 9.3\n.entry kernel() { { .future_scope x; ret; } }";
        let document = Document::parse(source).unwrap();
        assert!(matches!(
            ScopeFlattenPlan::analyze(&document),
            Err(ScopeFlattenError::UnsupportedDirective { name, .. }) if name == ".future_scope"
        ));
    }

    #[test]
    fn gates_flatten_plans_behind_the_ptx_version_ceiling() {
        let body = ".entry kernel() { { .reg .b32 x; mov.u32 x, 1; } ret; }";
        let gated = format!(".version 9.4\n{body}");
        let gated = Document::parse(&gated).unwrap();
        assert!(matches!(
            ScopeFlattenPlan::analyze(&gated),
            Err(ScopeFlattenError::Version(
                PtxVersionError::Unsupported { .. }
            ))
        ));
        let supported = format!(".version 9.3\n{body}");
        let supported = Document::parse(&supported).unwrap();
        let plan = ScopeFlattenPlan::analyze(&supported).unwrap();
        assert_eq!(plan.scopes().len(), 1);
    }

    #[test]
    fn rejects_unclosed_nested_scopes() {
        let source = ".version 9.3\n.entry kernel() { { ret; }";
        let document = Document::parse(source).unwrap();
        assert!(matches!(
            ScopeFlattenPlan::analyze(&document),
            Err(ScopeFlattenError::MissingDelimiter { .. })
        ));
    }
}
