/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Conservative intraprocedural control-flow recovery over surface PTX.
//!
//! The analysis models the control-flow forms defined through PTX ISA 9.3:
//! direct `bra`, indexed `brx.idx` via `.branchtargets`, predicated
//! fallthrough, `ret`, `exit`, and terminating `trap`. Calls fall through
//! within the caller. A newer PTX version or an unclosed target relation is a
//! hard error so clients never receive a guessed graph.
//!
//! This graph is a read-only recovery result over [`ptx_parse::Document`]. It
//! deliberately does not claim that the projected Pliron operation tree has
//! native basic-block structure. Turning this proof into native CFG is a
//! separate, fallible normalization step.

use crate::version::{PtxVersionError, validate_ptx_version};
use ptx_parse::{
    Document, Instruction, LabelId, ScopeId, StatementId, StatementKind, split_top_level,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum EdgeKind {
    Fallthrough,
    Branch,
    IndexedBranch,
}

/// A control-flow edge that leaves the callable rather than targeting another
/// recovered basic block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExitKind {
    /// Return from a `.func`, or terminate the calling thread from an `.entry`.
    Return,
    /// Terminate the current thread through `exit`.
    Thread,
    /// Abort execution through the PTX debug `trap` instruction.
    Trap,
}

/// Stable index of a block within one [`CallableControlFlow`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockId(usize);

impl BlockId {
    pub fn index(self) -> usize {
        self.0
    }
}

/// An adjacent block and the edge kind connecting it.
///
/// In [`BasicBlock::successors`] `block` is the target. In
/// [`BasicBlock::predecessors`] it is the source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Edge {
    block: BlockId,
    kind: EdgeKind,
}

impl Edge {
    pub fn block(self) -> BlockId {
        self.block
    }

    pub fn kind(self) -> EdgeKind {
        self.kind
    }
}

/// A maximal contiguous run of block instructions in one lexical PTX scope.
///
/// CFG blocks and lexical scopes are orthogonal: one recovered block can
/// contain several segments, including a return to an outer scope after a
/// nested scope closes. The range indexes [`BasicBlock::instructions`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScopeSegment {
    scope: ScopeId,
    instruction_range: Range<usize>,
}

impl ScopeSegment {
    pub fn scope(&self) -> ScopeId {
        self.scope
    }

    pub fn instruction_range(&self) -> Range<usize> {
        self.instruction_range.clone()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BasicBlock {
    id: BlockId,
    labels: Vec<LabelId>,
    instructions: Vec<StatementId>,
    scope_segments: Vec<ScopeSegment>,
    indexed_targets: Vec<BlockId>,
    successors: Vec<Edge>,
    predecessors: Vec<Edge>,
    exit: Option<ExitKind>,
}

impl BasicBlock {
    pub fn id(&self) -> BlockId {
        self.id
    }

    /// Source labels which designate this block, in source order.
    ///
    /// Several PTX labels may alias the same instruction and therefore the
    /// same recovered block.
    pub fn labels(&self) -> &[LabelId] {
        &self.labels
    }

    /// Authoritative syntax statements for the instructions in this block.
    pub fn instructions(&self) -> &[StatementId] {
        &self.instructions
    }

    pub fn scope_segments(&self) -> &[ScopeSegment] {
        &self.scope_segments
    }

    pub fn successors(&self) -> &[Edge] {
        &self.successors
    }

    /// Ordered `brx.idx` table slots, including duplicate destinations.
    ///
    /// [`Self::successors`] remains the deduplicated graph adjacency relation;
    /// this sequence is the authoritative indexed-dispatch relation.
    pub fn indexed_targets(&self) -> &[BlockId] {
        &self.indexed_targets
    }

    pub fn predecessors(&self) -> &[Edge] {
        &self.predecessors
    }

    /// The callable-exit edge produced by this block's final instruction.
    ///
    /// A predicated exit also has a normal fallthrough successor.
    pub fn exit(&self) -> Option<ExitKind> {
        self.exit
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallableControlFlow {
    callable: StatementId,
    scope: ScopeId,
    name: String,
    blocks: Vec<BasicBlock>,
}

impl CallableControlFlow {
    pub fn callable(&self) -> StatementId {
        self.callable
    }

    pub fn scope(&self) -> ScopeId {
        self.scope
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn blocks(&self) -> &[BasicBlock] {
        &self.blocks
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ControlFlow {
    callables: Vec<CallableControlFlow>,
    by_callable: HashMap<StatementId, usize>,
}

impl ControlFlow {
    pub fn analyze(document: &Document<'_>) -> Result<Self, CfgError> {
        validate_ptx_version(document).map_err(CfgError::Version)?;
        let mut callables = Vec::new();
        for callable in document.callables() {
            let body_span = match (callable.body_span(), callable.definition_scope()) {
                (Some(body), Some(_)) => body,
                (None, Some(_)) => {
                    return Err(CfgError::UnclosedCallable {
                        callable: callable.name().to_string(),
                    });
                }
                (None, None) => continue,
                (Some(_), None) => unreachable!("a callable body has a lexical scope"),
            };
            callables.push(analyze_callable(
                document,
                callable.statement(),
                callable
                    .definition_scope()
                    .expect("a callable with a body has a definition scope"),
                callable.name(),
                body_span.start,
                body_span.end,
            )?);
        }
        let by_callable = callables
            .iter()
            .enumerate()
            .map(|(index, callable)| (callable.callable, index))
            .collect();
        Ok(Self {
            callables,
            by_callable,
        })
    }

    pub fn callables(&self) -> &[CallableControlFlow] {
        &self.callables
    }

    pub fn for_callable(&self, callable: StatementId) -> Option<&CallableControlFlow> {
        self.by_callable
            .get(&callable)
            .map(|index| &self.callables[*index])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CfgError {
    Version(PtxVersionError),
    EmptyCallable {
        callable: String,
    },
    UnclosedCallable {
        callable: String,
    },
    DuplicateLabel {
        callable: String,
        label: String,
    },
    BranchWithoutTarget {
        callable: String,
        instruction: StatementId,
    },
    UnknownBranchTarget {
        callable: String,
        instruction: StatementId,
        target: String,
    },
    IndexedBranchWithoutTable {
        callable: String,
        instruction: StatementId,
    },
    UnknownBranchTable {
        callable: String,
        instruction: StatementId,
        table: String,
    },
    MalformedBranchTable {
        callable: String,
        table: String,
    },
    DuplicateBranchTable {
        callable: String,
        table: String,
    },
    BranchTableOutsideCallableScope {
        callable: String,
        table: String,
    },
    BranchTableAfterUse {
        callable: String,
        instruction: StatementId,
        table: String,
    },
    OpenFallthrough {
        callable: String,
        instruction: StatementId,
    },
}

impl fmt::Display for CfgError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Version(error) => error.fmt(formatter),
            Self::EmptyCallable { callable } => {
                write!(formatter, "PTX callable {callable:?} has no instructions")
            }
            Self::UnclosedCallable { callable } => {
                write!(formatter, "PTX callable {callable:?} has an unclosed body")
            }
            Self::DuplicateLabel { callable, label } => write!(
                formatter,
                "PTX callable {callable:?} defines label {label:?} more than once"
            ),
            Self::BranchWithoutTarget {
                callable,
                instruction,
            } => write!(
                formatter,
                "PTX branch statement {} in {callable:?} has no target",
                instruction.index()
            ),
            Self::UnknownBranchTarget {
                callable,
                instruction,
                target,
            } => write!(
                formatter,
                "PTX branch statement {} in {callable:?} targets unknown label {target:?}",
                instruction.index()
            ),
            Self::IndexedBranchWithoutTable {
                callable,
                instruction,
            } => write!(
                formatter,
                "PTX brx.idx statement {} in {callable:?} has no target table",
                instruction.index()
            ),
            Self::UnknownBranchTable {
                callable,
                instruction,
                table,
            } => write!(
                formatter,
                "PTX brx.idx statement {} in {callable:?} uses unknown table {table:?}",
                instruction.index()
            ),
            Self::MalformedBranchTable { callable, table } => write!(
                formatter,
                "PTX .branchtargets table {table:?} in {callable:?} is malformed"
            ),
            Self::DuplicateBranchTable { callable, table } => write!(
                formatter,
                "PTX callable {callable:?} defines .branchtargets table {table:?} more than once"
            ),
            Self::BranchTableOutsideCallableScope { callable, table } => write!(
                formatter,
                "PTX .branchtargets table {table:?} in {callable:?} is not declared at callable scope"
            ),
            Self::BranchTableAfterUse {
                callable,
                instruction,
                table,
            } => write!(
                formatter,
                "PTX brx.idx statement {} in {callable:?} uses .branchtargets table {table:?} before its declaration",
                instruction.index()
            ),
            Self::OpenFallthrough {
                callable,
                instruction,
            } => write!(
                formatter,
                "PTX callable {callable:?} falls through past final instruction statement {}",
                instruction.index()
            ),
        }
    }
}

impl std::error::Error for CfgError {}

fn analyze_callable(
    document: &Document<'_>,
    callable: StatementId,
    callable_scope: ScopeId,
    callable_name: &str,
    body_start: usize,
    body_end: usize,
) -> Result<CallableControlFlow, CfgError> {
    let instructions: Vec<_> = document.instructions_in(body_start..body_end).collect();
    if instructions.is_empty() {
        return Err(CfgError::EmptyCallable {
            callable: callable_name.to_string(),
        });
    }

    let instruction_by_statement: HashMap<StatementId, usize> = instructions
        .iter()
        .enumerate()
        .map(|(index, instruction)| (instruction.statement(), index))
        .collect();
    #[derive(Clone, Copy)]
    struct LabelBinding {
        instruction: usize,
        label: LabelId,
    }

    let mut labels = BTreeMap::<String, LabelBinding>::new();
    for label in document.labels_in(body_start..body_end) {
        if document
            .statement(label.statement())
            .is_some_and(|statement| statement.kind() == StatementKind::Directive)
        {
            continue;
        }
        let label_span = label.span();
        let Some(instruction) = instructions
            .iter()
            .position(|instruction| instruction.span().end > label_span.start)
        else {
            continue;
        };
        let binding = LabelBinding {
            instruction,
            label: label.id(),
        };
        if labels.insert(label.name().to_string(), binding).is_some() {
            return Err(CfgError::DuplicateLabel {
                callable: callable_name.to_string(),
                label: label.name().to_string(),
            });
        }
    }

    struct BranchTable {
        declaration_start: usize,
        targets: Vec<String>,
    }

    let mut branch_tables = BTreeMap::<String, BranchTable>::new();
    for directive in document
        .directives_in(body_start..body_end)
        .filter(|directive| directive.name() == ".branchtargets")
    {
        let Some(table) = directive.labels().last() else {
            continue;
        };
        if directive.scope() != callable_scope {
            return Err(CfgError::BranchTableOutsideCallableScope {
                callable: callable_name.to_string(),
                table: table.to_string(),
            });
        }
        let arguments = directive.arguments().trim_end().trim_end_matches(';');
        let Some(targets) = split_top_level(arguments) else {
            return Err(CfgError::MalformedBranchTable {
                callable: callable_name.to_string(),
                table: table.to_string(),
            });
        };
        if targets.is_empty() {
            return Err(CfgError::MalformedBranchTable {
                callable: callable_name.to_string(),
                table: table.to_string(),
            });
        }
        let branch_table = BranchTable {
            declaration_start: directive.span().start,
            targets: targets.into_iter().map(str::to_string).collect(),
        };
        if branch_tables
            .insert(table.to_string(), branch_table)
            .is_some()
        {
            return Err(CfgError::DuplicateBranchTable {
                callable: callable_name.to_string(),
                table: table.to_string(),
            });
        }
    }

    let mut leaders = BTreeSet::from([0usize]);
    for binding in labels.values() {
        leaders.insert(binding.instruction);
    }
    for (position, instruction) in instructions.iter().enumerate() {
        if terminator_kind(instruction).is_some() && position + 1 < instructions.len() {
            leaders.insert(position + 1);
        }
    }
    let leaders: Vec<usize> = leaders.into_iter().collect();
    let mut blocks: Vec<BasicBlock> = leaders
        .iter()
        .copied()
        .enumerate()
        .map(|(ordinal, start)| {
            let end = leaders
                .get(ordinal + 1)
                .copied()
                .unwrap_or(instructions.len());
            let instructions = instructions[start..end]
                .iter()
                .map(|instruction| instruction.statement())
                .collect::<Vec<_>>();
            let scope_segments = recover_scope_segments(document, &instructions);
            BasicBlock {
                id: BlockId(ordinal),
                labels: Vec::new(),
                instructions,
                scope_segments,
                indexed_targets: Vec::new(),
                successors: Vec::new(),
                predecessors: Vec::new(),
                exit: None,
            }
        })
        .collect();
    let block_for_position: Vec<usize> = (0..instructions.len())
        .map(|position| leaders.partition_point(|leader| *leader <= position) - 1)
        .collect();
    let label_blocks: BTreeMap<&str, usize> = labels
        .iter()
        .map(|(label, binding)| (label.as_str(), block_for_position[binding.instruction]))
        .collect();
    for binding in labels.values() {
        let block = block_for_position[binding.instruction];
        blocks[block].labels.push(binding.label);
    }
    for block in &mut blocks {
        block.labels.sort_by_key(|label| {
            document
                .label(*label)
                .expect("CFG label belongs to the document")
                .span()
                .start
        });
    }

    for block_index in 0..blocks.len() {
        let instruction_statement = *blocks[block_index]
            .instructions
            .last()
            .expect("CFG blocks are non-empty");
        let instruction_index = instruction_by_statement[&instruction_statement];
        let instruction = instructions[instruction_index];
        let mut successors = BTreeSet::new();
        match terminator_kind(instruction) {
            Some(TerminatorKind::Branch) => {
                let target =
                    instruction
                        .operands()
                        .next()
                        .ok_or_else(|| CfgError::BranchWithoutTarget {
                            callable: callable_name.to_string(),
                            instruction: instruction_statement,
                        })?;
                let target_block = label_blocks.get(target).copied().ok_or_else(|| {
                    CfgError::UnknownBranchTarget {
                        callable: callable_name.to_string(),
                        instruction: instruction_statement,
                        target: target.to_string(),
                    }
                })?;
                successors.insert(Edge {
                    block: BlockId(target_block),
                    kind: EdgeKind::Branch,
                });
                if instruction.predicate().is_some() {
                    add_fallthrough(
                        &mut successors,
                        block_index,
                        blocks.len(),
                        callable_name,
                        instruction_statement,
                    )?;
                }
            }
            Some(TerminatorKind::IndexedBranch) => {
                let operands: Vec<&str> = instruction.operands().collect();
                let table = operands.get(1).copied().ok_or_else(|| {
                    CfgError::IndexedBranchWithoutTable {
                        callable: callable_name.to_string(),
                        instruction: instruction_statement,
                    }
                })?;
                let branch_table =
                    branch_tables
                        .get(table)
                        .ok_or_else(|| CfgError::UnknownBranchTable {
                            callable: callable_name.to_string(),
                            instruction: instruction_statement,
                            table: table.to_string(),
                        })?;
                if branch_table.declaration_start >= instruction.span().start {
                    return Err(CfgError::BranchTableAfterUse {
                        callable: callable_name.to_string(),
                        instruction: instruction_statement,
                        table: table.to_string(),
                    });
                }
                for target in &branch_table.targets {
                    let target_block =
                        label_blocks.get(target.as_str()).copied().ok_or_else(|| {
                            CfgError::UnknownBranchTarget {
                                callable: callable_name.to_string(),
                                instruction: instruction_statement,
                                target: target.clone(),
                            }
                        })?;
                    successors.insert(Edge {
                        block: BlockId(target_block),
                        kind: EdgeKind::IndexedBranch,
                    });
                    blocks[block_index]
                        .indexed_targets
                        .push(BlockId(target_block));
                }
                if instruction.predicate().is_some() {
                    add_fallthrough(
                        &mut successors,
                        block_index,
                        blocks.len(),
                        callable_name,
                        instruction_statement,
                    )?;
                }
            }
            Some(kind @ (TerminatorKind::Return | TerminatorKind::Exit | TerminatorKind::Trap)) => {
                blocks[block_index].exit = Some(match kind {
                    TerminatorKind::Return => ExitKind::Return,
                    TerminatorKind::Exit => ExitKind::Thread,
                    TerminatorKind::Trap => ExitKind::Trap,
                    TerminatorKind::Branch | TerminatorKind::IndexedBranch => {
                        unreachable!("matched an exiting terminator")
                    }
                });
                if instruction.predicate().is_some() {
                    add_fallthrough(
                        &mut successors,
                        block_index,
                        blocks.len(),
                        callable_name,
                        instruction_statement,
                    )?;
                }
            }
            None => add_fallthrough(
                &mut successors,
                block_index,
                blocks.len(),
                callable_name,
                instruction_statement,
            )?,
        }
        blocks[block_index].successors = successors.into_iter().collect();
    }

    for source in 0..blocks.len() {
        let successors = blocks[source].successors.clone();
        for edge in successors {
            blocks[edge.block.index()].predecessors.push(Edge {
                block: BlockId(source),
                kind: edge.kind,
            });
        }
    }
    for block in &mut blocks {
        block.predecessors.sort();
        block.predecessors.dedup();
    }

    Ok(CallableControlFlow {
        callable,
        scope: callable_scope,
        name: callable_name.to_string(),
        blocks,
    })
}

fn recover_scope_segments(
    document: &Document<'_>,
    instructions: &[StatementId],
) -> Vec<ScopeSegment> {
    let mut segments = Vec::new();
    let mut start = 0;
    while start < instructions.len() {
        let scope = document
            .statement(instructions[start])
            .expect("CFG instruction belongs to the document")
            .scope();
        let mut end = start + 1;
        while end < instructions.len()
            && document
                .statement(instructions[end])
                .expect("CFG instruction belongs to the document")
                .scope()
                == scope
        {
            end += 1;
        }
        segments.push(ScopeSegment {
            scope,
            instruction_range: start..end,
        });
        start = end;
    }
    segments
}

fn add_fallthrough(
    successors: &mut BTreeSet<Edge>,
    block: usize,
    block_count: usize,
    callable: &str,
    instruction: StatementId,
) -> Result<(), CfgError> {
    if block + 1 >= block_count {
        return Err(CfgError::OpenFallthrough {
            callable: callable.to_string(),
            instruction,
        });
    }
    successors.insert(Edge {
        block: BlockId(block + 1),
        kind: EdgeKind::Fallthrough,
    });
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TerminatorKind {
    Branch,
    IndexedBranch,
    Return,
    Exit,
    Trap,
}

fn terminator_kind(instruction: &Instruction<'_>) -> Option<TerminatorKind> {
    let mut parts = instruction.head().split('.');
    match (parts.next(), parts.next()) {
        (Some("bra"), _) => Some(TerminatorKind::Branch),
        (Some("brx"), Some("idx")) => Some(TerminatorKind::IndexedBranch),
        (Some("ret"), _) => Some(TerminatorKind::Return),
        (Some("exit"), _) => Some(TerminatorKind::Exit),
        (Some("trap"), _) => Some(TerminatorKind::Trap),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn analyze(source: &str) -> Result<ControlFlow, CfgError> {
        let document = Document::parse(source).unwrap();
        ControlFlow::analyze(&document)
    }

    #[test]
    fn recovers_direct_branches_predicates_and_loops() {
        let cfg = analyze(
            "\
.version 8.9
.target sm_120a
.visible .entry kernel() {
L0:
    @%p0 bra L1;
    add.u32 %r0, %r0, 1;
    bra L0;
L1:
    ret;
}
",
        )
        .unwrap();
        let callable = &cfg.callables()[0];
        assert_eq!(cfg.for_callable(callable.callable()), Some(callable));
        assert_ne!(callable.scope(), ScopeId::ROOT);
        let blocks = callable.blocks();
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].id().index(), 0);
        assert!(!blocks[0].instructions().is_empty());
        assert_eq!(
            blocks[0].successors(),
            [
                Edge {
                    block: BlockId(1),
                    kind: EdgeKind::Fallthrough,
                },
                Edge {
                    block: BlockId(2),
                    kind: EdgeKind::Branch,
                },
            ]
        );
        assert_eq!(
            blocks[1].successors(),
            [Edge {
                block: BlockId(0),
                kind: EdgeKind::Branch,
            }]
        );
        assert!(blocks[2].successors().is_empty());
        assert_eq!(blocks[2].exit(), Some(ExitKind::Return));
    }

    #[test]
    fn resolves_indexed_branch_target_tables() {
        let cfg = analyze(
            "\
.version 9.0
.target sm_120a
.visible .entry kernel() {
targets: .branchtargets L1, L0, L1;
    @%p0 brx.idx %r0, targets;
    bra Done;
L0:
    mov.u32 %r1, 0;
    bra Done;
L1:
    mov.u32 %r1, 1;
Done:
    ret;
}
",
        )
        .unwrap();
        let blocks = cfg.callables()[0].blocks();
        assert_eq!(blocks.len(), 5);
        assert_eq!(
            blocks[0].successors(),
            [
                Edge {
                    block: BlockId(1),
                    kind: EdgeKind::Fallthrough,
                },
                Edge {
                    block: BlockId(2),
                    kind: EdgeKind::IndexedBranch,
                },
                Edge {
                    block: BlockId(3),
                    kind: EdgeKind::IndexedBranch,
                },
            ]
        );
        assert_eq!(
            blocks[0].indexed_targets(),
            [BlockId(3), BlockId(2), BlockId(3)]
        );
    }

    #[test]
    fn rejects_newer_isa_and_unclosed_targets() {
        assert!(matches!(
            analyze(".version 9.4\n.entry kernel() { ret; }"),
            Err(CfgError::Version(PtxVersionError::Unsupported { .. }))
        ));
        assert!(matches!(
            analyze(".version 9.0\n.entry kernel() { bra Missing; }"),
            Err(CfgError::UnknownBranchTarget { .. })
        ));
        assert!(matches!(
            analyze(".version 9.0\n.entry kernel() { add.u32 %r0, %r0, 1; }"),
            Err(CfgError::OpenFallthrough { .. })
        ));
        assert!(matches!(
            analyze(".version 9.0\n.entry kernel() { ret;"),
            Err(CfgError::UnclosedCallable { .. })
        ));
    }

    #[test]
    fn preserves_alias_labels_nested_scopes_unreachable_blocks_and_exit_kinds() {
        let source = "\
.version 9.3
.target sm_120a
.visible .entry kernel() {
    {
L0: Alias: @%p0 ret;
    }
    trap;
Dead:
    @%p1 exit;
    ret;
}
";
        let document = Document::parse(source).unwrap();
        let cfg = ControlFlow::analyze(&document).unwrap();
        let blocks = cfg.callables()[0].blocks();
        assert_eq!(blocks.len(), 4);

        let label_names = |block: &BasicBlock| {
            block
                .labels()
                .iter()
                .map(|label| document.label(*label).unwrap().name())
                .collect::<Vec<_>>()
        };
        assert_eq!(label_names(&blocks[0]), ["L0", "Alias"]);
        assert_eq!(blocks[0].exit(), Some(ExitKind::Return));
        assert_eq!(
            blocks[0].successors(),
            [Edge {
                block: BlockId(1),
                kind: EdgeKind::Fallthrough,
            }]
        );
        assert_eq!(blocks[1].exit(), Some(ExitKind::Trap));
        assert!(blocks[1].successors().is_empty());
        assert_eq!(label_names(&blocks[2]), ["Dead"]);
        assert_eq!(blocks[2].exit(), Some(ExitKind::Thread));
        assert_eq!(
            blocks[2].successors(),
            [Edge {
                block: BlockId(3),
                kind: EdgeKind::Fallthrough,
            }]
        );
        assert_eq!(blocks[3].exit(), Some(ExitKind::Return));
    }

    #[test]
    fn requires_branch_target_tables_at_callable_scope_before_use() {
        assert!(matches!(
            analyze(
                ".version 9.3\n.entry kernel() {\n@%p0 brx.idx %r0, targets;\ntargets: .branchtargets L0;\nL0: ret;\n}\n"
            ),
            Err(CfgError::BranchTableAfterUse { .. })
        ));
        assert!(matches!(
            analyze(
                ".version 9.3\n.entry kernel() {\n{ targets: .branchtargets L0; }\n@%p0 brx.idx %r0, targets;\nL0: ret;\n}\n"
            ),
            Err(CfgError::BranchTableOutsideCallableScope { .. })
        ));
        assert!(matches!(
            analyze(
                ".version 9.3\n.entry kernel() {\ntargets: .branchtargets L0;\ntargets: .branchtargets L0;\n@%p0 brx.idx %r0, targets;\nL0: ret;\n}\n"
            ),
            Err(CfgError::DuplicateBranchTable { .. })
        ));
    }

    #[test]
    fn preserves_lexical_scope_segments_inside_one_cfg_block() {
        let source = "\
.version 9.3
.entry kernel() {
    mov.u32 %r0, 0;
    {
        add.u32 %r0, %r0, 1;
    }
    sub.u32 %r0, %r0, 1;
    ret;
}
";
        let document = Document::parse(source).unwrap();
        let cfg = ControlFlow::analyze(&document).unwrap();
        let block = &cfg.callables()[0].blocks()[0];
        assert_eq!(block.instructions().len(), 4);
        assert_eq!(block.scope_segments().len(), 3);
        assert_eq!(block.scope_segments()[0].instruction_range(), 0..1);
        assert_eq!(block.scope_segments()[1].instruction_range(), 1..2);
        assert_eq!(block.scope_segments()[2].instruction_range(), 2..4);
        assert_eq!(
            block.scope_segments()[0].scope(),
            block.scope_segments()[2].scope()
        );
        assert_ne!(
            block.scope_segments()[0].scope(),
            block.scope_segments()[1].scope()
        );
    }
}
