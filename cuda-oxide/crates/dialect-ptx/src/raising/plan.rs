/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Context-independent planning and transactional native-CFG materialization.

mod materialize;

use super::NativeCfgProjection;
use crate::attributes::{CallableKindAttr, PredicateAttr, TerminatorKindAttr};
use crate::cfg::{BlockId, CfgError, ControlFlow, EdgeKind, ExitKind};
use crate::scopes::{ScopeFlattenError, ScopeFlattenPlan};
use pliron::context::Context;
use ptx_parse::{
    Callable, CallableKind, Document, EditError, EditMap, LabelId, ParseError, ScopeId,
    StatementId, StatementKind,
};
use std::fmt;
use std::ops::Range;

/// A fully checked, context-independent native CFG construction plan.
///
/// Analysis performs all fallible parsing, normalization, CFG recovery, and
/// statement placement. [`Self::materialize`] only allocates the already
/// proven operation graph, so a failed analysis cannot partially mutate a
/// caller's Pliron IR.
pub struct NativeCfgPlan {
    normalized_source: String,
    edit_map: EditMap,
    items: Vec<ModuleItemPlan>,
    block_count: usize,
}

impl NativeCfgPlan {
    pub fn analyze(source: &str) -> Result<Self, RaiseError> {
        let surface = Document::parse(source).map_err(RaiseError::Parse)?;
        let flatten = ScopeFlattenPlan::analyze(&surface).map_err(RaiseError::ScopeFlatten)?;
        let normalized = flatten.apply_with_map(source).map_err(RaiseError::Edit)?;
        let normalized_source = normalized.text;
        let document = Document::parse(&normalized_source).map_err(RaiseError::Parse)?;
        let control_flow = ControlFlow::analyze(&document).map_err(RaiseError::ControlFlow)?;
        validate_root_scopes(&document)?;

        let mut items = Vec::new();
        let mut block_count = 0;
        for statement in document.statements_in_scope(ScopeId::ROOT) {
            let item = match statement.kind() {
                StatementKind::Directive => {
                    let directive = document.directive_for_statement(statement.id()).ok_or(
                        RaiseError::UnsupportedStatement {
                            statement: statement.id(),
                            kind: statement.kind(),
                        },
                    )?;
                    ModuleItemPlan::Directive {
                        statement: statement.id(),
                        span: statement.span(),
                        labels: document
                            .labels_for_statement(statement.id())
                            .map(|label| LabelPlan {
                                id: label.id(),
                                name: label.name_span(),
                            })
                            .collect(),
                        name: directive.name_span(),
                        arguments: directive.arguments_span(),
                    }
                }
                StatementKind::CallableHeader => {
                    let callable = document.callable_for_statement(statement.id()).ok_or(
                        RaiseError::UnsupportedStatement {
                            statement: statement.id(),
                            kind: statement.kind(),
                        },
                    )?;
                    if callable.definition_scope().is_some() {
                        let recovered = control_flow.for_callable(statement.id()).ok_or(
                            RaiseError::MissingCallableControlFlow {
                                statement: statement.id(),
                            },
                        )?;
                        let plan = plan_callable(&document, callable, recovered)?;
                        block_count += plan.blocks.len();
                        ModuleItemPlan::Definition(plan)
                    } else {
                        ModuleItemPlan::Declaration(CallableHeaderPlan::new(
                            callable,
                            trim_header_span(document.source(), statement.span()),
                            statement.span(),
                        ))
                    }
                }
                StatementKind::Preprocessor => ModuleItemPlan::Raw {
                    statement: statement.id(),
                    span: statement.span(),
                },
                kind => {
                    return Err(RaiseError::UnsupportedStatement {
                        statement: statement.id(),
                        kind,
                    });
                }
            };
            items.push(item);
        }
        Ok(Self {
            normalized_source,
            edit_map: normalized.map,
            items,
            block_count,
        })
    }

    pub fn normalized_source(&self) -> &str {
        &self.normalized_source
    }

    pub fn edit_map(&self) -> &EditMap {
        &self.edit_map
    }

    pub fn callable_count(&self) -> usize {
        self.items
            .iter()
            .filter(|item| matches!(item, ModuleItemPlan::Definition(_)))
            .count()
    }

    pub fn block_count(&self) -> usize {
        self.block_count
    }

    pub fn materialize(self, ctx: &mut Context) -> NativeCfgProjection {
        materialize::materialize(self, ctx)
    }
}

#[derive(Debug)]
pub enum RaiseError {
    Parse(ParseError),
    ScopeFlatten(ScopeFlattenError),
    Edit(EditError),
    ControlFlow(CfgError),
    UnsupportedRootScope {
        scope: ScopeId,
        header: Option<StatementId>,
    },
    UnsupportedStatement {
        statement: StatementId,
        kind: StatementKind,
    },
    MissingCallableControlFlow {
        statement: StatementId,
    },
    MissingInstructionBlock {
        callable: StatementId,
        statement: StatementId,
    },
    TrailingStatement {
        callable: StatementId,
        statement: StatementId,
    },
    InvalidTerminatorOperands {
        statement: StatementId,
        head: String,
    },
}

impl fmt::Display for RaiseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(error) => error.fmt(formatter),
            Self::ScopeFlatten(error) => error.fmt(formatter),
            Self::Edit(error) => error.fmt(formatter),
            Self::ControlFlow(error) => error.fmt(formatter),
            Self::UnsupportedRootScope { scope, header } => write!(
                formatter,
                "PTX native CFG raising does not support root scope {} with header {header:?}",
                scope.index()
            ),
            Self::UnsupportedStatement { statement, kind } => write!(
                formatter,
                "PTX native CFG raising does not support {kind:?} statement {}",
                statement.index()
            ),
            Self::MissingCallableControlFlow { statement } => write!(
                formatter,
                "PTX callable statement {} has no recovered control flow",
                statement.index()
            ),
            Self::MissingInstructionBlock {
                callable,
                statement,
            } => write!(
                formatter,
                "PTX callable statement {} instruction statement {} has no recovered block",
                callable.index(),
                statement.index()
            ),
            Self::TrailingStatement {
                callable,
                statement,
            } => write!(
                formatter,
                "PTX callable statement {} has statement {} after its final instruction",
                callable.index(),
                statement.index()
            ),
            Self::InvalidTerminatorOperands { statement, head } => write!(
                formatter,
                "PTX terminator {head:?} at statement {} has unsupported operands",
                statement.index()
            ),
        }
    }
}

impl std::error::Error for RaiseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Parse(error) => Some(error),
            Self::ScopeFlatten(error) => Some(error),
            Self::Edit(error) => Some(error),
            Self::ControlFlow(error) => Some(error),
            Self::UnsupportedRootScope { .. }
            | Self::UnsupportedStatement { .. }
            | Self::MissingCallableControlFlow { .. }
            | Self::MissingInstructionBlock { .. }
            | Self::TrailingStatement { .. }
            | Self::InvalidTerminatorOperands { .. } => None,
        }
    }
}

enum ModuleItemPlan {
    Directive {
        statement: StatementId,
        span: Range<usize>,
        labels: Vec<LabelPlan>,
        name: Range<usize>,
        arguments: Range<usize>,
    },
    Raw {
        statement: StatementId,
        span: Range<usize>,
    },
    Declaration(CallableHeaderPlan),
    Definition(CallablePlan),
}

struct CallableHeaderPlan {
    statement: StatementId,
    span: Range<usize>,
    name: Range<usize>,
    kind: CallableKindAttr,
    is_extern: bool,
    header: Range<usize>,
}

impl CallableHeaderPlan {
    fn new(callable: &Callable<'_>, header: Range<usize>, span: Range<usize>) -> Self {
        Self {
            statement: callable.statement(),
            span,
            name: callable.name_span(),
            kind: callable_kind(callable.kind()),
            is_extern: callable.is_extern(),
            header,
        }
    }
}

struct CallablePlan {
    header: CallableHeaderPlan,
    blocks: Vec<BlockPlan>,
}

struct LabelPlan {
    id: LabelId,
    name: Range<usize>,
}

struct BlockPlan {
    id: BlockId,
    nodes: Vec<NodePlan>,
    terminator: TerminatorPlan,
}

enum NodePlan {
    Label {
        label: LabelId,
        span: Range<usize>,
        name: Range<usize>,
    },
    Directive {
        statement: StatementId,
        span: Range<usize>,
        labels: Vec<LabelPlan>,
        name: Range<usize>,
        arguments: Range<usize>,
    },
    BranchTargets {
        statement: StatementId,
        span: Range<usize>,
        label: LabelId,
        name: Range<usize>,
    },
    Instruction {
        statement: StatementId,
        span: Range<usize>,
        predicate: Option<PredicateAttr>,
        head: Range<usize>,
        operands: Vec<Range<usize>>,
    },
}

struct TerminatorPlan {
    source: Option<(StatementId, Range<usize>)>,
    kind: TerminatorKindAttr,
    predicate: Option<PredicateAttr>,
    head: Range<usize>,
    operands: Vec<Range<usize>>,
    target_table: Option<Range<usize>>,
    has_fallthrough: bool,
    successors: Vec<usize>,
}

fn validate_root_scopes(document: &Document<'_>) -> Result<(), RaiseError> {
    let callable_scopes = document
        .callables()
        .iter()
        .filter_map(|callable| callable.definition_scope())
        .collect::<std::collections::HashSet<_>>();
    for scope in document
        .scopes()
        .iter()
        .filter(|scope| scope.parent() == Some(ScopeId::ROOT))
    {
        if !callable_scopes.contains(&scope.id()) {
            return Err(RaiseError::UnsupportedRootScope {
                scope: scope.id(),
                header: scope.header(),
            });
        }
    }
    Ok(())
}

fn plan_callable(
    document: &Document<'_>,
    callable: &Callable<'_>,
    recovered: &crate::cfg::CallableControlFlow,
) -> Result<CallablePlan, RaiseError> {
    let scope = callable
        .definition_scope()
        .expect("a recovered callable is a definition");
    let mut instruction_blocks = vec![None; document.statements().len()];
    for block in recovered.blocks() {
        for statement in block.instructions() {
            instruction_blocks[statement.index()] = Some(block.id().index());
        }
    }
    let mut statements_by_block = vec![Vec::new(); recovered.blocks().len()];
    let mut pending = Vec::new();
    for statement in document.statements_in_scope(scope) {
        if statement.kind() == StatementKind::Instruction {
            let block = instruction_blocks[statement.id().index()].ok_or(
                RaiseError::MissingInstructionBlock {
                    callable: callable.statement(),
                    statement: statement.id(),
                },
            )?;
            statements_by_block[block].append(&mut pending);
            statements_by_block[block].push(statement.id());
        } else {
            pending.push(statement.id());
        }
    }
    if let Some(statement) = pending.first().copied() {
        return Err(RaiseError::TrailingStatement {
            callable: callable.statement(),
            statement,
        });
    }

    let mut blocks = Vec::with_capacity(recovered.blocks().len());
    for block in recovered.blocks() {
        let actual_kind = actual_terminator_kind(block);
        let final_instruction = block.instructions().last().copied();
        let mut nodes = Vec::new();
        let mut terminator = None;
        for statement_id in &statements_by_block[block.id().index()] {
            let statement = document
                .statement(*statement_id)
                .expect("planned statement belongs to the document");
            match statement.kind() {
                StatementKind::Label => plan_labels(document, *statement_id, &mut nodes),
                StatementKind::Directive => {
                    let directive = document.directive_for_statement(*statement_id).ok_or(
                        RaiseError::UnsupportedStatement {
                            statement: *statement_id,
                            kind: statement.kind(),
                        },
                    )?;
                    if directive.name() == ".branchtargets" {
                        let mut table_labels = directive.labels();
                        let Some(_name) = table_labels.next() else {
                            return Err(RaiseError::UnsupportedStatement {
                                statement: *statement_id,
                                kind: statement.kind(),
                            });
                        };
                        if table_labels.next().is_some() {
                            return Err(RaiseError::UnsupportedStatement {
                                statement: *statement_id,
                                kind: statement.kind(),
                            });
                        }
                        nodes.push(NodePlan::BranchTargets {
                            statement: *statement_id,
                            span: statement.span(),
                            label: document
                                .labels_for_statement(*statement_id)
                                .next()
                                .expect("validated branch table label")
                                .id(),
                            name: document
                                .labels_for_statement(*statement_id)
                                .next()
                                .expect("validated branch table label")
                                .name_span(),
                        });
                    } else {
                        nodes.push(NodePlan::Directive {
                            statement: *statement_id,
                            span: statement.span(),
                            labels: document
                                .labels_for_statement(*statement_id)
                                .map(|label| LabelPlan {
                                    id: label.id(),
                                    name: label.name_span(),
                                })
                                .collect(),
                            name: directive.name_span(),
                            arguments: directive.arguments_span(),
                        });
                    }
                }
                StatementKind::Instruction => {
                    plan_labels(document, *statement_id, &mut nodes);
                    let instruction = document.instruction_for_statement(*statement_id).ok_or(
                        RaiseError::UnsupportedStatement {
                            statement: *statement_id,
                            kind: statement.kind(),
                        },
                    )?;
                    let predicate = instruction.predicate().map(PredicateAttr::from);
                    if Some(*statement_id) == final_instruction
                        && let Some(kind) = actual_kind
                    {
                        let source_operands: Vec<_> = instruction.operand_spans().collect();
                        let (operands, target_table) = match kind {
                            TerminatorKindAttr::Branch if source_operands.len() == 1 => {
                                (Vec::new(), None)
                            }
                            TerminatorKindAttr::IndexedBranch if source_operands.len() == 2 => (
                                vec![source_operands[0].clone()],
                                Some(source_operands[1].clone()),
                            ),
                            TerminatorKindAttr::Return
                            | TerminatorKindAttr::ThreadExit
                            | TerminatorKindAttr::Trap
                                if source_operands.is_empty() =>
                            {
                                (Vec::new(), None)
                            }
                            _ => {
                                return Err(RaiseError::InvalidTerminatorOperands {
                                    statement: *statement_id,
                                    head: instruction.head().to_string(),
                                });
                            }
                        };
                        terminator = Some(TerminatorPlan {
                            source: Some((*statement_id, statement.span())),
                            kind,
                            predicate,
                            head: instruction.head_span(),
                            operands,
                            target_table,
                            has_fallthrough: block
                                .successors()
                                .iter()
                                .any(|edge| edge.kind() == EdgeKind::Fallthrough),
                            successors: ordered_successors(block),
                        });
                    } else {
                        nodes.push(NodePlan::Instruction {
                            statement: *statement_id,
                            span: statement.span(),
                            predicate,
                            head: instruction.head_span(),
                            operands: instruction.operand_spans().collect(),
                        });
                    }
                }
                kind => {
                    return Err(RaiseError::UnsupportedStatement {
                        statement: *statement_id,
                        kind,
                    });
                }
            }
        }
        let terminator = terminator.unwrap_or_else(|| TerminatorPlan {
            source: None,
            kind: TerminatorKindAttr::Fallthrough,
            predicate: None,
            head: 0..0,
            operands: Vec::new(),
            target_table: None,
            has_fallthrough: false,
            successors: ordered_successors(block),
        });
        blocks.push(BlockPlan {
            id: block.id(),
            nodes,
            terminator,
        });
    }
    let header = CallableHeaderPlan::new(
        callable,
        trim_span(
            document.source(),
            callable
                .definition_header_span()
                .expect("a recovered callable has a closed header"),
        ),
        document
            .statement(callable.statement())
            .expect("callable statement belongs to the document")
            .span(),
    );
    Ok(CallablePlan { header, blocks })
}

fn actual_terminator_kind(block: &crate::cfg::BasicBlock) -> Option<TerminatorKindAttr> {
    if let Some(exit) = block.exit() {
        return Some(match exit {
            ExitKind::Return => TerminatorKindAttr::Return,
            ExitKind::Thread => TerminatorKindAttr::ThreadExit,
            ExitKind::Trap => TerminatorKindAttr::Trap,
        });
    }
    if block
        .successors()
        .iter()
        .any(|edge| edge.kind() == EdgeKind::IndexedBranch)
    {
        return Some(TerminatorKindAttr::IndexedBranch);
    }
    block
        .successors()
        .iter()
        .any(|edge| edge.kind() == EdgeKind::Branch)
        .then_some(TerminatorKindAttr::Branch)
}

fn ordered_successors(block: &crate::cfg::BasicBlock) -> Vec<usize> {
    let mut successors = block
        .successors()
        .iter()
        .filter(|edge| edge.kind() == EdgeKind::Fallthrough)
        .map(|edge| edge.block().index())
        .collect::<Vec<_>>();
    if block.indexed_targets().is_empty() {
        successors.extend(
            block
                .successors()
                .iter()
                .filter(|edge| edge.kind() == EdgeKind::Branch)
                .map(|edge| edge.block().index()),
        );
    } else {
        successors.extend(block.indexed_targets().iter().map(|block| block.index()));
    }
    successors
}

fn plan_labels(document: &Document<'_>, statement: StatementId, nodes: &mut Vec<NodePlan>) {
    for label in document.labels_for_statement(statement) {
        nodes.push(NodePlan::Label {
            label: label.id(),
            span: label.span(),
            name: label.name_span(),
        });
    }
}

fn callable_kind(kind: CallableKind) -> CallableKindAttr {
    match kind {
        CallableKind::Entry => CallableKindAttr::Entry,
        CallableKind::Function => CallableKindAttr::Function,
    }
}

fn trim_span(source: &str, mut span: Range<usize>) -> Range<usize> {
    while span.start < span.end && source.as_bytes()[span.start].is_ascii_whitespace() {
        span.start += 1;
    }
    while span.start < span.end && source.as_bytes()[span.end - 1].is_ascii_whitespace() {
        span.end -= 1;
    }
    span
}

fn trim_header_span(source: &str, span: Range<usize>) -> Range<usize> {
    let mut span = trim_span(source, span);
    while span.start < span.end && matches!(source.as_bytes()[span.end - 1], b';' | b'{') {
        span.end -= 1;
        span = trim_span(source, span);
    }
    span
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::SourceNode;
    use crate::{emit_canonical_module, register};
    use pliron::common_traits::Verify;
    use pliron::op::Op;

    #[test]
    fn plans_before_materializing_loops_predicates_and_indexed_targets() {
        let source = "\
.version 9.3
.target sm_120a
.visible .entry kernel() {
    .reg .pred %p;
    .reg .b32 %r<2>;
    {
        .reg .b32 %r0;
        mov.u32 %r0, 0;
    }
targets: .branchtargets Done, L0, Done;
L0:
    @%p bra Done;
    @%p brx.idx %r0, targets;
    bra L0;
Done:
    ret;
}
";
        let plan = NativeCfgPlan::analyze(source).unwrap();
        assert_eq!(plan.callable_count(), 1);
        assert_eq!(plan.block_count(), 5);
        assert!(!plan.normalized_source().contains(".reg .b32 %r0;"));

        let mut ctx = Context::new();
        register(&mut ctx);
        let raised = plan.materialize(&mut ctx);
        raised
            .module()
            .get_operation()
            .deref(&ctx)
            .verify(&ctx)
            .unwrap();
        assert_eq!(raised.blocks().len(), 5);
        let emitted = emit_canonical_module(&ctx, &raised.module()).unwrap();
        assert!(emitted.contains("targets: .branchtargets Done, L0, Done;"));
        let reparsed = Document::parse(&emitted).unwrap();
        let cfg = ControlFlow::analyze(&reparsed).unwrap();
        assert_eq!(cfg.callables()[0].blocks().len(), 5);
    }

    #[test]
    fn retains_original_and_normalized_lineage_across_alpha_renaming() {
        let source = ".version 9.3\n.entry kernel() { { .reg .b32 x; mov.u32 x, 1; } ret; }";
        let original = Document::parse(source).unwrap();
        let statement = original
            .instructions()
            .iter()
            .find(|instruction| instruction.head() == "mov.u32")
            .unwrap()
            .statement();
        let original_span = original.statement(statement).unwrap().span();
        let mut ctx = Context::new();
        register(&mut ctx);
        let raised = NativeCfgPlan::analyze(source)
            .unwrap()
            .materialize(&mut ctx);
        let operation = raised
            .operation_for_source(SourceNode::Statement { statement })
            .unwrap();
        let node = raised
            .nodes()
            .iter()
            .find(|node| node.operation() == operation)
            .unwrap();
        assert_eq!(node.original_source_span(), Some(original_span));
        assert_ne!(node.original_source_span(), node.normalized_source_span());
    }

    #[test]
    fn analysis_failure_does_not_require_or_mutate_a_context() {
        let source = ".version 9.3\n.entry kernel() { { .future_scope x; ret; } }";
        assert!(matches!(
            NativeCfgPlan::analyze(source),
            Err(RaiseError::ScopeFlatten(
                ScopeFlattenError::UnsupportedDirective { .. }
            ))
        ));
    }
}
