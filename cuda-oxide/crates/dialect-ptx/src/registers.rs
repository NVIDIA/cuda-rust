/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lexical register resolution and alpha-renaming plans for surface PTX.
//!
//! This analysis does not move declarations or remove lexical scopes. It
//! produces a separately auditable rename plan which makes every nested
//! register binding callable-unique, a prerequisite for later scope flattening.

use crate::version::{PtxVersionError, validate_ptx_version};
use ptx_parse::{
    Document, EditError, EditScript, RegisterDeclarationError, ScopeId, StatementId, Token,
    TokenKind,
};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ops::Range;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegisterAlphaPlan {
    callables: Vec<CallableRegisterAlphaPlan>,
}

impl RegisterAlphaPlan {
    pub fn analyze(document: &Document<'_>) -> Result<Self, RegisterAlphaError> {
        // Renaming rewrites live register uses; a PTX version newer than the
        // audited semantics ceiling could bind or spell registers in ways
        // this resolver does not model, so rewrites gate exactly like the CFG.
        validate_ptx_version(document).map_err(RegisterAlphaError::Version)?;
        let parsed_declarations = document
            .register_declarations()
            .collect::<Result<Vec<_>, _>>()
            .map_err(RegisterAlphaError::MalformedDeclaration)?;
        let callable_scopes = document
            .callables()
            .iter()
            .filter_map(|callable| {
                Some((
                    callable.statement(),
                    callable.definition_scope()?,
                    callable.body_span()?,
                ))
            })
            .collect::<Vec<_>>();
        let mut callable_roots = vec![None; document.scopes().len()];
        for (index, (_, scope, _)) in callable_scopes.iter().enumerate() {
            callable_roots[scope.index()] = Some(index);
        }
        let mut callable_by_scope = vec![None; document.scopes().len()];
        for scope in document.scopes().iter().skip(1) {
            callable_by_scope[scope.id().index()] =
                callable_roots[scope.id().index()].or_else(|| {
                    scope
                        .parent()
                        .and_then(|parent| callable_by_scope[parent.index()])
                });
        }
        let mut declarations_by_callable = vec![Vec::new(); callable_scopes.len()];
        for declaration in parsed_declarations {
            if let Some(callable) = callable_by_scope[declaration.scope().index()] {
                declarations_by_callable[callable].push(declaration);
            }
        }

        let mut callables = Vec::with_capacity(callable_scopes.len());
        for (index, (callable, scope, body)) in callable_scopes.into_iter().enumerate() {
            callables.push(analyze_callable(
                document,
                callable,
                scope,
                body,
                &declarations_by_callable[index],
            )?);
        }
        Ok(Self { callables })
    }

    pub fn callables(&self) -> &[CallableRegisterAlphaPlan] {
        &self.callables
    }

    pub fn for_callable(&self, callable: StatementId) -> Option<&CallableRegisterAlphaPlan> {
        self.callables.iter().find(|plan| plan.callable == callable)
    }

    pub fn edit_script(&self) -> Result<EditScript, EditError> {
        let mut edits = EditScript::new();
        self.add_edits(&mut edits)?;
        Ok(edits)
    }

    /// Append this plan's source-preserving renames to a larger transaction.
    pub fn add_edits(&self, edits: &mut EditScript) -> Result<(), EditError> {
        for callable in &self.callables {
            for rename in &callable.renames {
                edits.replace(rename.declaration_span(), rename.new_name())?;
                for usage in rename.uses() {
                    edits.replace(usage.span(), usage.new_name())?;
                }
            }
        }
        Ok(())
    }

    pub fn apply(&self, source: &str) -> Result<String, EditError> {
        self.edit_script()?.apply(source)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallableRegisterAlphaPlan {
    callable: StatementId,
    scope: ScopeId,
    renames: Vec<RegisterRename>,
}

impl CallableRegisterAlphaPlan {
    pub fn callable(&self) -> StatementId {
        self.callable
    }

    pub fn scope(&self) -> ScopeId {
        self.scope
    }

    pub fn renames(&self) -> &[RegisterRename] {
        &self.renames
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegisterRename {
    declaration: StatementId,
    scope: ScopeId,
    declaration_span: Range<usize>,
    old_name: String,
    new_name: String,
    bank_size: Option<u32>,
    uses: Vec<RegisterUseRename>,
}

impl RegisterRename {
    pub fn declaration(&self) -> StatementId {
        self.declaration
    }

    pub fn scope(&self) -> ScopeId {
        self.scope
    }

    pub fn declaration_span(&self) -> Range<usize> {
        self.declaration_span.clone()
    }

    pub fn old_name(&self) -> &str {
        &self.old_name
    }

    pub fn new_name(&self) -> &str {
        &self.new_name
    }

    pub fn bank_size(&self) -> Option<u32> {
        self.bank_size
    }

    pub fn uses(&self) -> &[RegisterUseRename] {
        &self.uses
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegisterUseRename {
    statement: StatementId,
    span: Range<usize>,
    old_name: String,
    new_name: String,
}

impl RegisterUseRename {
    pub fn statement(&self) -> StatementId {
        self.statement
    }

    pub fn span(&self) -> Range<usize> {
        self.span.clone()
    }

    pub fn old_name(&self) -> &str {
        &self.old_name
    }

    pub fn new_name(&self) -> &str {
        &self.new_name
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegisterAlphaError {
    Version(PtxVersionError),
    MalformedDeclaration(RegisterDeclarationError),
    DuplicateBinding {
        callable: StatementId,
        scope: ScopeId,
        first: StatementId,
        second: StatementId,
        name: String,
    },
    /// A binding scheduled for renaming is also spelled with a vector-element
    /// suffix (`v.x`). The suffixed spelling lexes as one word, so a rename of
    /// `v` would leave `v.x` behind, silently rebinding it to any same-named
    /// outer register.
    VectorElementUse {
        callable: StatementId,
        scope: ScopeId,
        name: String,
        token: String,
        span: Range<usize>,
    },
    /// A generated rename target spells a label of this callable or a
    /// callable name of this module. The rewritten register would capture
    /// branch or call targets, producing PTX that ptxas rejects or, worse,
    /// assembles with wrong semantics.
    RenameTargetCollision {
        callable: StatementId,
        scope: ScopeId,
        old_name: String,
        new_name: String,
        symbol: String,
        kind: RenameCollisionKind,
    },
}

/// The namespace a colliding rename target belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenameCollisionKind {
    /// A label defined in the callable body.
    Label,
    /// A callable declared or defined in the module.
    Callable,
}

impl fmt::Display for RegisterAlphaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Version(error) => error.fmt(formatter),
            Self::MalformedDeclaration(error) => error.fmt(formatter),
            Self::DuplicateBinding {
                callable,
                scope,
                first,
                second,
                name,
            } => write!(
                formatter,
                "PTX callable statement {} scope {} has overlapping register binding {name:?} in statements {} and {}",
                callable.index(),
                scope.index(),
                first.index(),
                second.index()
            ),
            Self::VectorElementUse {
                callable,
                scope,
                name,
                token,
                span,
            } => write!(
                formatter,
                "PTX callable statement {} cannot rename register {name:?} in scope {}: \
                 vector-element token {token:?} at bytes {}..{} lexes as one word and would \
                 not be rewritten",
                callable.index(),
                scope.index(),
                span.start,
                span.end
            ),
            Self::RenameTargetCollision {
                callable,
                scope,
                old_name,
                new_name,
                symbol,
                kind,
            } => write!(
                formatter,
                "PTX callable statement {} cannot rename register {old_name:?} in scope {} \
                 to {new_name:?}: it collides with {} {symbol:?}",
                callable.index(),
                scope.index(),
                match kind {
                    RenameCollisionKind::Label => "label",
                    RenameCollisionKind::Callable => "callable",
                }
            ),
        }
    }
}

impl std::error::Error for RegisterAlphaError {}

#[derive(Clone, Debug)]
struct BindingDefinition {
    declaration: StatementId,
    scope: ScopeId,
    declaration_span: Range<usize>,
    declaration_offset: usize,
    name: String,
    bank_size: Option<u32>,
    uses: Vec<ResolvedUse>,
}

#[derive(Clone, Debug)]
struct ResolvedUse {
    statement: StatementId,
    span: Range<usize>,
    name: String,
}

fn analyze_callable(
    document: &Document<'_>,
    callable: StatementId,
    callable_scope: ScopeId,
    body: Range<usize>,
    declarations: &[ptx_parse::RegisterDeclaration<'_>],
) -> Result<CallableRegisterAlphaPlan, RegisterAlphaError> {
    let mut definitions = Vec::new();
    let mut definitions_by_scope = HashMap::<ScopeId, Vec<usize>>::new();
    for declaration in declarations {
        for binding in declaration.bindings() {
            let definition = BindingDefinition {
                declaration: declaration.statement(),
                scope: declaration.scope(),
                declaration_span: binding.name_span(),
                declaration_offset: binding.name_span().start,
                name: binding.name().to_string(),
                bank_size: binding.bank_size(),
                uses: Vec::new(),
            };
            if let Some(previous) =
                definitions_by_scope
                    .get(&definition.scope)
                    .and_then(|indices| {
                        indices
                            .iter()
                            .copied()
                            .find(|index| bindings_overlap(&definitions[*index], &definition))
                    })
            {
                return Err(RegisterAlphaError::DuplicateBinding {
                    callable,
                    scope: definition.scope,
                    first: definitions[previous].declaration,
                    second: definition.declaration,
                    name: definition.name,
                });
            }
            let index = definitions.len();
            definitions_by_scope
                .entry(definition.scope)
                .or_default()
                .push(index);
            definitions.push(definition);
        }
    }

    for instruction in document.instructions_in(body.clone()) {
        if let Some(predicate) = instruction.predicate() {
            resolve_use_span(
                document,
                instruction.statement(),
                instruction.scope(),
                predicate.register_span(),
                &definitions_by_scope,
                &mut definitions,
            );
        }
        for operand in instruction.operand_spans() {
            for token in tokens_in(document.tokens(), operand)
                .filter(|token| token.kind() == TokenKind::Word)
            {
                resolve_use_span(
                    document,
                    instruction.statement(),
                    instruction.scope(),
                    token.span(),
                    &definitions_by_scope,
                    &mut definitions,
                );
            }
        }
    }

    let occupied = tokens_in(document.tokens(), body.clone())
        .filter(|token| token.kind() == TokenKind::Word)
        .map(|token| token.text(document.source()).to_string())
        .collect::<HashSet<_>>();
    let nested = definitions
        .iter()
        .enumerate()
        .filter(|(_, definition)| definition.scope != callable_scope)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();

    // '.' is a word byte, so a vector-element use such as `v.x` lexes as one
    // token and never resolves to the binding `v`. Renaming the binding would
    // leave the element use behind, silently rebinding it to any same-named
    // outer register. Fail closed before planning any rename.
    let element_uses = tokens_in(document.tokens(), body.clone())
        .filter(|token| token.kind() == TokenKind::Word)
        .filter_map(|token| {
            let text = token.text(document.source());
            let (base, suffix) = text.split_once('.')?;
            (!base.is_empty() && !suffix.is_empty()).then(|| (base, text, token.span()))
        })
        .collect::<Vec<_>>();
    for index in &nested {
        let definition = &definitions[*index];
        if let Some((_, token, span)) = element_uses
            .iter()
            .find(|(base, _, _)| binding_matches(definition, base))
        {
            return Err(RegisterAlphaError::VectorElementUse {
                callable,
                scope: definition.scope,
                name: definition.name.clone(),
                token: (*token).to_string(),
                span: span.clone(),
            });
        }
    }

    // Rename targets must not spell a label of this callable or a callable
    // name of this module. The occupied-token check avoids everything spelled
    // in the body, but a module callable that is never referenced in this
    // body, or a label equal to a bank base name, would slip through and
    // capture branch or call targets in the rewritten PTX.
    let labels = document
        .labels()
        .iter()
        .filter(|label| {
            let span = label.span();
            span.start >= body.start && span.end <= body.end
        })
        .map(|label| label.name())
        .collect::<Vec<_>>();
    let callable_names = document
        .callables()
        .iter()
        .map(|callable| callable.name())
        .collect::<Vec<_>>();

    let mut generated = Vec::<BindingName>::new();
    let mut renames = Vec::with_capacity(nested.len());
    for index in nested {
        let definition = &definitions[index];
        let base = generated_name(&definition.name, definition.scope);
        let mut candidate = base.clone();
        let mut discriminator = 0usize;
        while !binding_name_available(&candidate, definition.bank_size, &occupied, &generated) {
            discriminator += 1;
            candidate = format!("{base}_{discriminator}");
        }
        for (symbols, kind) in [
            (&labels, RenameCollisionKind::Label),
            (&callable_names, RenameCollisionKind::Callable),
        ] {
            if let Some(symbol) = colliding_symbol(&candidate, definition.bank_size, symbols) {
                return Err(RegisterAlphaError::RenameTargetCollision {
                    callable,
                    scope: definition.scope,
                    old_name: definition.name.clone(),
                    new_name: candidate,
                    symbol: symbol.to_string(),
                    kind,
                });
            }
        }
        generated.push(BindingName {
            name: candidate.clone(),
            bank_size: definition.bank_size,
        });
        let uses = definition
            .uses
            .iter()
            .map(|usage| RegisterUseRename {
                statement: usage.statement,
                span: usage.span.clone(),
                old_name: usage.name.clone(),
                new_name: renamed_use(&definition.name, &candidate, &usage.name),
            })
            .collect();
        renames.push(RegisterRename {
            declaration: definition.declaration,
            scope: definition.scope,
            declaration_span: definition.declaration_span.clone(),
            old_name: definition.name.clone(),
            new_name: candidate,
            bank_size: definition.bank_size,
            uses,
        });
    }
    Ok(CallableRegisterAlphaPlan {
        callable,
        scope: callable_scope,
        renames,
    })
}

fn resolve_use_span(
    document: &Document<'_>,
    statement: StatementId,
    mut scope: ScopeId,
    span: Range<usize>,
    definitions_by_scope: &HashMap<ScopeId, Vec<usize>>,
    definitions: &mut [BindingDefinition],
) {
    let name = &document.source()[span.clone()];
    loop {
        if let Some(definition) = definitions_by_scope.get(&scope).and_then(|indices| {
            indices.iter().copied().find(|index| {
                definitions[*index].declaration_offset < span.start
                    && binding_matches(&definitions[*index], name)
            })
        }) {
            definitions[definition].uses.push(ResolvedUse {
                statement,
                span,
                name: name.to_string(),
            });
            return;
        }
        let Some(parent) = document.scope(scope).and_then(|scope| scope.parent()) else {
            return;
        };
        scope = parent;
    }
}

fn tokens_in(tokens: &[Token], span: Range<usize>) -> impl Iterator<Item = &Token> {
    let start = tokens.partition_point(|token| token.span().end <= span.start);
    tokens[start..]
        .iter()
        .take_while(move |token| token.span().start < span.end)
        .filter(move |token| {
            let token_span = token.span();
            token_span.start >= span.start && token_span.end <= span.end
        })
}

fn binding_matches(definition: &BindingDefinition, name: &str) -> bool {
    match definition.bank_size {
        None => definition.name == name,
        Some(size) => bank_index(&definition.name, name).is_some_and(|index| index < size),
    }
}

fn bindings_overlap(first: &BindingDefinition, second: &BindingDefinition) -> bool {
    binding_names_overlap(&first.name, first.bank_size, &second.name, second.bank_size)
}

fn binding_names_overlap(
    first: &str,
    first_size: Option<u32>,
    second: &str,
    second_size: Option<u32>,
) -> bool {
    match (first_size, second_size) {
        (None, None) => first == second,
        (Some(size), None) => bank_index(first, second).is_some_and(|index| index < size),
        (None, Some(size)) => bank_index(second, first).is_some_and(|index| index < size),
        (Some(_), Some(_)) => {
            first == second
                || numeric_suffix(first, second).is_some()
                || numeric_suffix(second, first).is_some()
        }
    }
}

fn bank_index(base: &str, name: &str) -> Option<u32> {
    let suffix = name.strip_prefix(base)?;
    let index = suffix.parse::<u32>().ok()?;
    (suffix == index.to_string()).then_some(index)
}

fn numeric_suffix(base: &str, name: &str) -> Option<()> {
    let suffix = name.strip_prefix(base)?;
    (!suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())).then_some(())
}

fn generated_name(old_name: &str, scope: ScopeId) -> String {
    if let Some(name) = old_name.strip_prefix('%') {
        format!("%__oxide_s{}_{}", scope.index(), name)
    } else {
        format!("__oxide_s{}_{}", scope.index(), old_name)
    }
}

struct BindingName {
    name: String,
    bank_size: Option<u32>,
}

fn binding_name_available(
    name: &str,
    bank_size: Option<u32>,
    occupied: &HashSet<String>,
    generated: &[BindingName],
) -> bool {
    if occupied.iter().any(|occupied| match bank_size {
        None => occupied == name,
        Some(size) => bank_index(name, occupied).is_some_and(|index| index < size),
    }) {
        return false;
    }
    generated
        .iter()
        .all(|binding| !binding_names_overlap(name, bank_size, &binding.name, binding.bank_size))
}

/// Find a symbol which a rename to `candidate` would capture: the exact name,
/// or for a bank binding any name the bank expands to (`candidate` + index).
fn colliding_symbol<'symbols>(
    candidate: &str,
    bank_size: Option<u32>,
    symbols: &[&'symbols str],
) -> Option<&'symbols str> {
    symbols.iter().copied().find(|symbol| {
        *symbol == candidate
            || bank_size
                .is_some_and(|size| bank_index(candidate, symbol).is_some_and(|index| index < size))
    })
}

fn renamed_use(old_base: &str, new_base: &str, old_use: &str) -> String {
    let suffix = old_use
        .strip_prefix(old_base)
        .expect("a resolved register use matches its binding");
    format!("{new_base}{suffix}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans_nested_scalar_bank_shadowing_and_collision_free_edits() {
        let source = "\
.version 9.3
.entry kernel() {
    .reg .pred %p<2>;
    .reg .b32 x, __oxide_s2_x;
    mov.u32 x, 0;
    {
        .reg .pred %p0;
        .reg .b32 x, y;
        .reg .b32 %r<2>;
        @%p0 mov.u32 x, y;
        add.u32 %r0, %r1, x;
        {
            .reg .b32 x;
            mov.u32 x, 7;
        }
        add.u32 x, x, 1;
    }
    @%p0 mov.u32 x, 2;
    ret;
}
";
        let document = Document::parse(source).unwrap();
        let plan = RegisterAlphaPlan::analyze(&document).unwrap();
        let callable = &plan.callables()[0];
        assert_eq!(callable.renames().len(), 5);
        assert!(
            callable
                .renames()
                .iter()
                .all(|rename| rename.scope() != callable.scope())
        );
        let outer_x = callable
            .renames()
            .iter()
            .find(|rename| rename.old_name() == "x" && rename.uses().len() == 4)
            .unwrap();
        assert_eq!(outer_x.new_name(), "__oxide_s2_x_1");
        let bank = callable
            .renames()
            .iter()
            .find(|rename| rename.bank_size() == Some(2))
            .unwrap();
        assert_eq!(
            bank.uses()
                .iter()
                .map(RegisterUseRename::old_name)
                .collect::<Vec<_>>(),
            ["%r0", "%r1"]
        );
        assert_eq!(
            bank.uses()
                .iter()
                .map(RegisterUseRename::new_name)
                .collect::<Vec<_>>(),
            ["%__oxide_s2_r0", "%__oxide_s2_r1"]
        );

        let rewritten = plan.apply(source).unwrap();
        assert!(rewritten.contains(".reg .b32 __oxide_s2_x_1, __oxide_s2_y;"));
        assert!(rewritten.contains("@%__oxide_s2_p0 mov.u32 __oxide_s2_x_1, __oxide_s2_y;"));
        assert!(rewritten.contains(".reg .b32 %__oxide_s2_r<2>;"));
        assert!(rewritten.contains("@%p0 mov.u32 x, 2;"));
        Document::parse(&rewritten).unwrap();
    }

    #[test]
    fn rejects_overlapping_bindings_in_one_scope() {
        let source = ".version 9.3\n.entry kernel() { .reg .b32 %r<2>; .reg .b32 %r1; ret; }";
        let document = Document::parse(source).unwrap();
        assert!(matches!(
            RegisterAlphaPlan::analyze(&document),
            Err(RegisterAlphaError::DuplicateBinding { .. })
        ));
    }

    #[test]
    fn rejects_vector_element_uses_of_renamed_bindings() {
        let source = "\
.version 9.3
.entry kernel() {
    .reg .v4 .f32 v;
    {
        .reg .v4 .f32 v;
        mov.f32 v.x, 0f00000000;
        ret;
    }
}
";
        let document = Document::parse(source).unwrap();
        let error = RegisterAlphaPlan::analyze(&document).unwrap_err();
        let RegisterAlphaError::VectorElementUse {
            name, token, span, ..
        } = &error
        else {
            panic!("expected a vector-element rejection, got {error}");
        };
        assert_eq!(name, "v");
        assert_eq!(token, "v.x");
        assert_eq!(&source[span.clone()], "v.x");
    }

    #[test]
    fn plans_nested_vector_bindings_without_element_uses() {
        let source = "\
.version 9.3
.entry kernel() {
    .reg .v4 .f32 v;
    {
        .reg .v4 .f32 v;
        mov.v4.f32 v, {0f00000000, 0f00000000, 0f00000000, 0f00000000};
        ret;
    }
}
";
        let document = Document::parse(source).unwrap();
        let plan = RegisterAlphaPlan::analyze(&document).unwrap();
        let renames = plan.callables()[0].renames();
        assert_eq!(renames.len(), 1);
        assert_eq!(renames[0].old_name(), "v");
        assert_eq!(renames[0].new_name(), "__oxide_s2_v");
        let rewritten = plan.apply(source).unwrap();
        assert!(rewritten.contains("mov.v4.f32 __oxide_s2_v, {"));
        Document::parse(&rewritten).unwrap();
    }

    #[test]
    fn rejects_rename_targets_that_capture_module_callable_names() {
        let source = "\
.version 9.3
.extern .func __oxide_s2_x ();
.entry kernel() {
    .reg .b32 x;
    {
        .reg .b32 x;
        mov.u32 x, 1;
    }
    ret;
}
";
        let document = Document::parse(source).unwrap();
        let error = RegisterAlphaPlan::analyze(&document).unwrap_err();
        let RegisterAlphaError::RenameTargetCollision {
            old_name,
            new_name,
            symbol,
            kind,
            ..
        } = &error
        else {
            panic!("expected a rename-target collision, got {error}");
        };
        assert_eq!(old_name, "x");
        assert_eq!(new_name, "__oxide_s2_x");
        assert_eq!(symbol, "__oxide_s2_x");
        assert_eq!(*kind, RenameCollisionKind::Callable);

        let harmless = source.replace("__oxide_s2_x", "helper");
        let document = Document::parse(&harmless).unwrap();
        let plan = RegisterAlphaPlan::analyze(&document).unwrap();
        assert_eq!(plan.callables()[0].renames()[0].new_name(), "__oxide_s2_x");
    }

    #[test]
    fn rejects_bank_rename_targets_that_capture_labels() {
        let source = "\
.version 9.3
.entry kernel() {
    {
        .reg .b32 r<2>;
        mov.u32 r0, 1;
    }
    bra __oxide_s2_r;
__oxide_s2_r:
    ret;
}
";
        let document = Document::parse(source).unwrap();
        let error = RegisterAlphaPlan::analyze(&document).unwrap_err();
        let RegisterAlphaError::RenameTargetCollision {
            new_name,
            symbol,
            kind,
            ..
        } = &error
        else {
            panic!("expected a rename-target collision, got {error}");
        };
        assert_eq!(new_name, "__oxide_s2_r");
        assert_eq!(symbol, "__oxide_s2_r");
        assert_eq!(*kind, RenameCollisionKind::Label);

        let harmless = source.replace("__oxide_s2_r", "Done");
        let document = Document::parse(&harmless).unwrap();
        let plan = RegisterAlphaPlan::analyze(&document).unwrap();
        assert_eq!(plan.callables()[0].renames()[0].new_name(), "__oxide_s2_r");
    }

    #[test]
    fn gates_rename_plans_behind_the_ptx_version_ceiling() {
        let body = ".entry kernel() { { .reg .b32 x; mov.u32 x, 1; } ret; }";
        let gated = format!(".version 9.4\n{body}");
        let gated = Document::parse(&gated).unwrap();
        assert!(matches!(
            RegisterAlphaPlan::analyze(&gated),
            Err(RegisterAlphaError::Version(
                PtxVersionError::Unsupported { .. }
            ))
        ));
        let missing = Document::parse(body).unwrap();
        assert!(matches!(
            RegisterAlphaPlan::analyze(&missing),
            Err(RegisterAlphaError::Version(PtxVersionError::Missing))
        ));
        let supported = format!(".version 9.3\n{body}");
        let supported = Document::parse(&supported).unwrap();
        let plan = RegisterAlphaPlan::analyze(&supported).unwrap();
        assert_eq!(plan.callables()[0].renames().len(), 1);
    }

    #[test]
    fn forwards_malformed_declaration_errors() {
        let source = ".version 9.3\n.entry kernel() { .reg .b32 %r<0>; ret; }";
        let document = Document::parse(source).unwrap();
        assert!(matches!(
            RegisterAlphaPlan::analyze(&document),
            Err(RegisterAlphaError::MalformedDeclaration(_))
        ));
    }
}
