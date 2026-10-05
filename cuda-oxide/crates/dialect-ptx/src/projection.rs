/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::attributes::{CallableKindAttr, PredicateAttr};
use crate::cfg::{
    BasicBlock as RecoveredBasicBlock, BlockId, CallableControlFlow, CfgError, ControlFlow, Edge,
    ExitKind, ScopeSegment,
};
use crate::ops::{
    PtxCallableOp, PtxDirectiveOp, PtxInstructionOp, PtxLabelOp, PtxModuleOp, PtxRawOp, PtxScopeOp,
};
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::op::Op;
use pliron::operation::Operation;
use ptx_parse::{Document, LabelId, ParseError, ScopeId, StatementId, StatementKind};
use std::collections::HashMap;
use std::ops::Range;

/// The authoritative syntax node from which a projected entity was built.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SourceNode {
    Label { label: LabelId },
    Statement { statement: StatementId },
    Scope { scope: ScopeId },
}

impl SourceNode {
    pub fn statement(self) -> Option<StatementId> {
        match self {
            Self::Label { .. } => None,
            Self::Statement { statement } => Some(statement),
            Self::Scope { .. } => None,
        }
    }

    pub fn scope(self) -> Option<ScopeId> {
        match self {
            Self::Label { .. } => None,
            Self::Statement { .. } => None,
            Self::Scope { scope } => Some(scope),
        }
    }

    pub fn label(self) -> Option<LabelId> {
        match self {
            Self::Label { label } => Some(label),
            Self::Statement { .. } | Self::Scope { .. } => None,
        }
    }
}

/// One Pliron operation and its immutable source lineage.
#[derive(Clone, Debug)]
pub struct ProjectedNode {
    operation: Ptr<Operation>,
    source_node: SourceNode,
    source_span: Range<usize>,
}

impl ProjectedNode {
    pub fn operation(&self) -> Ptr<Operation> {
        self.operation
    }

    pub fn source_node(&self) -> SourceNode {
        self.source_node
    }

    pub fn source_span(&self) -> Range<usize> {
        self.source_span.clone()
    }
}

/// One Pliron basic block and the lexical scope it represents.
#[derive(Clone, Debug)]
pub struct ProjectedBlock {
    block: Ptr<BasicBlock>,
    source_scope: ScopeId,
    source_span: Range<usize>,
}

impl ProjectedBlock {
    pub fn block(&self) -> Ptr<BasicBlock> {
        self.block
    }

    pub fn source_scope(&self) -> ScopeId {
        self.source_scope
    }

    pub fn source_span(&self) -> Range<usize> {
        self.source_span.clone()
    }
}

/// A lossless syntax document paired with a structured, independently
/// emittable Pliron PTX module.
pub struct Projection<'source> {
    document: Document<'source>,
    module: PtxModuleOp,
    nodes: Vec<ProjectedNode>,
    nodes_by_operation: HashMap<Ptr<Operation>, usize>,
    nodes_by_source: HashMap<SourceNode, usize>,
    blocks: Vec<ProjectedBlock>,
    blocks_by_pointer: HashMap<Ptr<BasicBlock>, usize>,
    blocks_by_source: HashMap<ScopeId, usize>,
}

impl<'source> Projection<'source> {
    /// Parse PTX and project its structural statements and lexical scopes.
    ///
    /// The caller must register this dialect in `ctx` before parsing, matching
    /// the lifecycle of CUDA Oxide's other Pliron dialects.
    pub fn parse(ctx: &mut Context, source: &'source str) -> Result<Self, ParseError> {
        let document = Document::parse(source)?;
        Ok(Self::from_document(ctx, document))
    }

    /// Project an already-parsed document. Source text remains authoritative
    /// for lossless edits; the produced operation tree is authoritative for
    /// structured analysis, construction, and canonical emission.
    pub fn from_document(ctx: &mut Context, document: Document<'source>) -> Self {
        let module = PtxModuleOp::build(ctx);
        let root_block = module.body(ctx);
        let (nodes, blocks, source_aliases) = {
            let mut projector = Projector::new(ctx, &document);
            projector.record_block(root_block, ScopeId::ROOT);
            projector.project_scope(ScopeId::ROOT, root_block);
            (projector.nodes, projector.blocks, projector.source_aliases)
        };
        let nodes_by_operation: HashMap<Ptr<Operation>, usize> = nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.operation, index))
            .collect();
        let mut nodes_by_source: HashMap<SourceNode, usize> = nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.source_node, index))
            .collect();
        for (source, operation) in source_aliases {
            let index = *nodes_by_operation
                .get(&operation)
                .expect("source alias operation is projected");
            nodes_by_source.insert(source, index);
        }
        let blocks_by_pointer = blocks
            .iter()
            .enumerate()
            .map(|(index, block)| (block.block, index))
            .collect();
        let blocks_by_source = blocks
            .iter()
            .enumerate()
            .map(|(index, block)| (block.source_scope, index))
            .collect();

        Self {
            document,
            module,
            nodes,
            nodes_by_operation,
            nodes_by_source,
            blocks,
            blocks_by_pointer,
            blocks_by_source,
        }
    }

    pub fn document(&self) -> &Document<'source> {
        &self.document
    }

    pub fn module(&self) -> PtxModuleOp {
        self.module
    }

    pub fn nodes(&self) -> &[ProjectedNode] {
        &self.nodes
    }

    pub fn blocks(&self) -> &[ProjectedBlock] {
        &self.blocks
    }

    pub fn source_node(&self, operation: Ptr<Operation>) -> Option<SourceNode> {
        self.nodes_by_operation
            .get(&operation)
            .map(|index| self.nodes[*index].source_node)
    }

    /// Resolve source lineage back to the projected operation.
    pub fn operation_for_source(&self, source: SourceNode) -> Option<Ptr<Operation>> {
        self.nodes_by_source
            .get(&source)
            .map(|index| self.nodes[*index].operation)
    }

    pub fn operation_for_statement(&self, statement: StatementId) -> Option<Ptr<Operation>> {
        self.operation_for_source(SourceNode::Statement { statement })
    }

    pub fn operation_for_label(&self, label: LabelId) -> Option<Ptr<Operation>> {
        self.operation_for_source(SourceNode::Label { label })
    }

    pub fn operation_for_scope(&self, scope: ScopeId) -> Option<Ptr<Operation>> {
        self.operation_for_source(SourceNode::Scope { scope })
    }

    pub fn source_scope(&self, block: Ptr<BasicBlock>) -> Option<ScopeId> {
        self.blocks_by_pointer
            .get(&block)
            .map(|index| self.blocks[*index].source_scope)
    }

    /// Resolve one lexical source scope to its projected Pliron block.
    pub fn block_for_source_scope(&self, scope: ScopeId) -> Option<Ptr<BasicBlock>> {
        self.blocks_by_source
            .get(&scope)
            .map(|index| self.blocks[*index].block)
    }

    /// Recover conservative intraprocedural CFGs from the authoritative
    /// lossless syntax. The returned graph retains `StatementId`/`ScopeId`
    /// lineage and fails closed when PTX control-flow semantics are uncertain.
    pub fn control_flow(&self) -> Result<ControlFlow, CfgError> {
        ControlFlow::analyze(&self.document)
    }

    /// Join recovered surface CFG with projected Pliron operations without
    /// claiming that those operations already inhabit native CFG blocks.
    pub fn projected_control_flow(&self) -> Result<ProjectedControlFlow<'_, 'source>, CfgError> {
        let recovered = self.control_flow()?;
        let mut blocks_by_source = HashMap::new();
        for (callable_index, callable) in recovered.callables().iter().enumerate() {
            for block in callable.blocks() {
                for label in block.labels() {
                    blocks_by_source.insert(
                        SourceNode::Label { label: *label },
                        (callable_index, block.id()),
                    );
                }
                for statement in block.instructions() {
                    blocks_by_source.insert(
                        SourceNode::Statement {
                            statement: *statement,
                        },
                        (callable_index, block.id()),
                    );
                }
            }
        }
        Ok(ProjectedControlFlow {
            projection: self,
            recovered,
            blocks_by_source,
        })
    }
}

/// A read-only join between recovered surface CFG and projected operations.
///
/// The recovered graph remains authoritative for block boundaries and edges.
/// This view only resolves its source lineage to the corresponding Pliron ops;
/// it does not materialize Pliron basic blocks or successors.
pub struct ProjectedControlFlow<'projection, 'source> {
    projection: &'projection Projection<'source>,
    recovered: ControlFlow,
    blocks_by_source: HashMap<SourceNode, (usize, BlockId)>,
}

impl<'projection, 'source> ProjectedControlFlow<'projection, 'source> {
    pub fn recovered(&self) -> &ControlFlow {
        &self.recovered
    }

    pub fn callables(
        &self,
    ) -> impl ExactSizeIterator<Item = ProjectedCallableControlFlow<'_, 'source>> + '_ {
        self.recovered
            .callables()
            .iter()
            .map(|callable| ProjectedCallableControlFlow {
                projection: self.projection,
                recovered: callable,
            })
    }

    pub fn for_callable_operation(
        &self,
        operation: Ptr<Operation>,
    ) -> Option<ProjectedCallableControlFlow<'_, 'source>> {
        let statement = self.projection.source_node(operation)?.statement()?;
        self.for_callable_statement(statement)
    }

    pub fn for_callable_statement(
        &self,
        statement: StatementId,
    ) -> Option<ProjectedCallableControlFlow<'_, 'source>> {
        self.recovered
            .for_callable(statement)
            .map(|callable| ProjectedCallableControlFlow {
                projection: self.projection,
                recovered: callable,
            })
    }

    /// Find the recovered CFG block containing one projected instruction or
    /// label operation.
    pub fn block_for_operation(
        &self,
        operation: Ptr<Operation>,
    ) -> Option<ProjectedCfgBlock<'_, 'source>> {
        self.block_for_source(self.projection.source_node(operation)?)
    }

    pub fn block_for_statement(
        &self,
        statement: StatementId,
    ) -> Option<ProjectedCfgBlock<'_, 'source>> {
        self.block_for_source(SourceNode::Statement { statement })
    }

    pub fn block_for_label(&self, label: LabelId) -> Option<ProjectedCfgBlock<'_, 'source>> {
        self.block_for_source(SourceNode::Label { label })
    }

    fn block_for_source(&self, source: SourceNode) -> Option<ProjectedCfgBlock<'_, 'source>> {
        let (callable, block) = self.blocks_by_source.get(&source).copied()?;
        let recovered = self
            .recovered
            .callables()
            .get(callable)?
            .blocks()
            .get(block.index())?;
        Some(ProjectedCfgBlock {
            projection: self.projection,
            recovered,
        })
    }
}

/// One recovered callable paired with its projected `ptx.callable` operation.
#[derive(Clone, Copy)]
pub struct ProjectedCallableControlFlow<'projection, 'source> {
    projection: &'projection Projection<'source>,
    recovered: &'projection CallableControlFlow,
}

impl ProjectedCallableControlFlow<'_, '_> {
    pub fn recovered(&self) -> &CallableControlFlow {
        self.recovered
    }

    pub fn operation(&self) -> Ptr<Operation> {
        self.projection
            .operation_for_statement(self.recovered.callable())
            .expect("a recovered callable belongs to its projection")
    }

    pub fn blocks(&self) -> impl ExactSizeIterator<Item = ProjectedCfgBlock<'_, '_>> + '_ {
        self.recovered
            .blocks()
            .iter()
            .map(|block| ProjectedCfgBlock {
                projection: self.projection,
                recovered: block,
            })
    }
}

/// One recovered CFG block with operation-level label and instruction views.
#[derive(Clone, Copy)]
pub struct ProjectedCfgBlock<'projection, 'source> {
    projection: &'projection Projection<'source>,
    recovered: &'projection RecoveredBasicBlock,
}

impl ProjectedCfgBlock<'_, '_> {
    pub fn recovered(&self) -> &RecoveredBasicBlock {
        self.recovered
    }

    pub fn id(&self) -> BlockId {
        self.recovered.id()
    }

    pub fn labels(&self) -> impl Iterator<Item = (LabelId, Ptr<Operation>)> + '_ {
        self.recovered.labels().iter().copied().map(|label| {
            let operation = self
                .projection
                .operation_for_label(label)
                .expect("a recovered label belongs to its projection");
            (label, operation)
        })
    }

    pub fn instructions(&self) -> impl Iterator<Item = (StatementId, Ptr<Operation>)> + '_ {
        self.recovered
            .instructions()
            .iter()
            .copied()
            .map(|statement| {
                let operation = self
                    .projection
                    .operation_for_statement(statement)
                    .expect("a recovered instruction belongs to its projection");
                (statement, operation)
            })
    }

    pub fn scope_segments(
        &self,
    ) -> impl ExactSizeIterator<Item = ProjectedCfgScopeSegment<'_, '_>> + '_ {
        self.recovered
            .scope_segments()
            .iter()
            .map(|segment| ProjectedCfgScopeSegment {
                projection: self.projection,
                block: self.recovered,
                recovered: segment,
            })
    }

    pub fn successors(&self) -> &[Edge] {
        self.recovered.successors()
    }

    pub fn predecessors(&self) -> &[Edge] {
        self.recovered.predecessors()
    }

    pub fn exit(&self) -> Option<ExitKind> {
        self.recovered.exit()
    }
}

/// One lexical scope segment within a recovered CFG block.
#[derive(Clone, Copy)]
pub struct ProjectedCfgScopeSegment<'projection, 'source> {
    projection: &'projection Projection<'source>,
    block: &'projection RecoveredBasicBlock,
    recovered: &'projection ScopeSegment,
}

impl ProjectedCfgScopeSegment<'_, '_> {
    pub fn recovered(&self) -> &ScopeSegment {
        self.recovered
    }

    pub fn scope(&self) -> ScopeId {
        self.recovered.scope()
    }

    /// The existing lexical Pliron block for this source scope.
    pub fn lexical_block(&self) -> Ptr<BasicBlock> {
        self.projection
            .block_for_source_scope(self.scope())
            .expect("a recovered scope belongs to its projection")
    }

    pub fn instructions(&self) -> impl Iterator<Item = (StatementId, Ptr<Operation>)> + '_ {
        self.block.instructions()[self.recovered.instruction_range()]
            .iter()
            .copied()
            .map(|statement| {
                let operation = self
                    .projection
                    .operation_for_statement(statement)
                    .expect("a recovered instruction belongs to its projection");
                (statement, operation)
            })
    }
}

struct Projector<'ctx, 'document, 'source> {
    ctx: &'ctx mut Context,
    document: &'document Document<'source>,
    scopes_by_parent: Vec<Vec<ScopeId>>,
    nodes: Vec<ProjectedNode>,
    blocks: Vec<ProjectedBlock>,
    source_aliases: Vec<(SourceNode, Ptr<Operation>)>,
}

impl<'ctx, 'document, 'source> Projector<'ctx, 'document, 'source> {
    fn new(ctx: &'ctx mut Context, document: &'document Document<'source>) -> Self {
        let mut scopes_by_parent = vec![Vec::new(); document.scopes().len()];
        for scope in document.scopes().iter().skip(1) {
            if let Some(parent) = scope.parent() {
                scopes_by_parent[parent.index()].push(scope.id());
            }
        }
        Self {
            ctx,
            document,
            scopes_by_parent,
            nodes: Vec::new(),
            blocks: Vec::new(),
            source_aliases: Vec::new(),
        }
    }

    fn record_block(&mut self, block: Ptr<BasicBlock>, scope: ScopeId) {
        let source_span = self
            .document
            .scope(scope)
            .expect("projected scope belongs to the document")
            .body_span();
        self.blocks.push(ProjectedBlock {
            block,
            source_scope: scope,
            source_span,
        });
    }

    fn record_operation(
        &mut self,
        operation: Ptr<Operation>,
        source_node: SourceNode,
        source_span: Range<usize>,
        destination: Ptr<BasicBlock>,
    ) {
        operation.insert_at_back(destination, self.ctx);
        self.nodes.push(ProjectedNode {
            operation,
            source_node,
            source_span,
        });
    }

    fn project_scope(&mut self, scope: ScopeId, destination: Ptr<BasicBlock>) {
        #[derive(Clone, Copy)]
        enum Event {
            Statement(StatementId),
            AnonymousScope(ScopeId),
        }

        let child_scopes = self.scopes_by_parent[scope.index()].clone();
        let scopes_by_header: HashMap<StatementId, ScopeId> = child_scopes
            .iter()
            .filter_map(|scope| {
                self.document
                    .scope(*scope)
                    .and_then(|scope_node| scope_node.header().map(|header| (header, *scope)))
            })
            .collect();
        let mut events: Vec<(usize, Event)> = self
            .document
            .statements_in_scope(scope)
            .map(|statement| statement.id())
            .map(|statement| {
                let start = self
                    .document
                    .statement(statement)
                    .expect("indexed statement belongs to the document")
                    .span()
                    .start;
                (start, Event::Statement(statement))
            })
            .collect();
        events.extend(child_scopes.into_iter().filter_map(|child| {
            let child = self.document.scope(child)?;
            if child.header().is_some() {
                return None;
            }
            Some((child.open_span()?.start, Event::AnonymousScope(child.id())))
        }));
        events.sort_by_key(|(start, _)| *start);

        for (_, event) in events {
            match event {
                Event::Statement(statement) => {
                    if let Some(child_scope) = scopes_by_header.get(&statement).copied() {
                        self.project_header_scope(statement, child_scope, destination);
                    } else {
                        self.project_statement(statement, destination);
                    }
                }
                Event::AnonymousScope(child_scope) => {
                    self.project_lexical_scope(child_scope, "", destination);
                }
            }
        }
    }

    fn project_header_scope(
        &mut self,
        statement: StatementId,
        scope: ScopeId,
        destination: Ptr<BasicBlock>,
    ) {
        self.project_labels(statement, destination);
        if let Some(callable) = self.document.callable_for_statement(statement) {
            let kind = CallableKindAttr::from(callable.kind());
            let statement_node = self
                .document
                .statement(statement)
                .expect("callable statement belongs to the document");
            let header = trim_header(statement_node.text(self.document.source()));
            let operation = PtxCallableOp::build_definition(
                self.ctx,
                callable.name(),
                kind,
                callable.is_extern(),
                header,
            );
            let body = operation
                .entry_block(self.ctx)
                .expect("a definition has an entry block");
            self.record_operation(
                operation.get_operation(),
                SourceNode::Statement { statement },
                statement_node.span(),
                destination,
            );
            self.record_block(body, scope);
            self.project_scope(scope, body);
            return;
        }

        let header = self
            .document
            .statement(statement)
            .expect("scope header belongs to the document")
            .text(self.document.source());
        self.project_lexical_scope(scope, trim_header(header), destination);
    }

    fn project_lexical_scope(
        &mut self,
        scope: ScopeId,
        header: &str,
        destination: Ptr<BasicBlock>,
    ) {
        let source_span = self
            .document
            .scope(scope)
            .expect("projected scope belongs to the document")
            .span();
        let operation = PtxScopeOp::build(self.ctx, header);
        let body = operation.body(self.ctx);
        self.record_operation(
            operation.get_operation(),
            SourceNode::Scope { scope },
            source_span,
            destination,
        );
        self.record_block(body, scope);
        self.project_scope(scope, body);
    }

    fn project_statement(&mut self, statement: StatementId, destination: Ptr<BasicBlock>) {
        let statement_node = self
            .document
            .statement(statement)
            .expect("indexed statement belongs to the document");
        let source_span = statement_node.span();
        let directive_labels = if statement_node.kind() == StatementKind::Directive {
            self.document
                .labels_for_statement(statement)
                .cloned()
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let projected_labels = if directive_labels.is_empty() {
            self.project_labels(statement, destination)
        } else {
            false
        };
        let operation = match statement_node.kind() {
            StatementKind::Directive => {
                self.document
                    .directive_for_statement(statement)
                    .map(|directive| {
                        PtxDirectiveOp::build_labeled(
                            self.ctx,
                            directive_labels.iter().map(|label| label.name()),
                            directive.name(),
                            directive.arguments(),
                        )
                        .get_operation()
                    })
            }
            StatementKind::Instruction => {
                self.document
                    .instruction_for_statement(statement)
                    .map(|instruction| {
                        let predicate = instruction.predicate().map(PredicateAttr::from);
                        PtxInstructionOp::build(
                            self.ctx,
                            predicate,
                            instruction.head(),
                            instruction.operands(),
                        )
                        .get_operation()
                    })
            }
            StatementKind::CallableHeader => {
                self.document
                    .callable_for_statement(statement)
                    .map(|callable| {
                        let kind = CallableKindAttr::from(callable.kind());
                        PtxCallableOp::build_declaration(
                            self.ctx,
                            callable.name(),
                            kind,
                            callable.is_extern(),
                            trim_header(statement_node.text(self.document.source())),
                        )
                        .get_operation()
                    })
            }
            StatementKind::Label if projected_labels => return,
            StatementKind::Label | StatementKind::Preprocessor | StatementKind::Unknown => None,
        }
        .unwrap_or_else(|| {
            PtxRawOp::build(self.ctx, statement_node.text(self.document.source())).get_operation()
        });
        self.record_operation(
            operation,
            SourceNode::Statement { statement },
            source_span,
            destination,
        );
        self.source_aliases.extend(
            directive_labels
                .into_iter()
                .map(|label| (SourceNode::Label { label: label.id() }, operation)),
        );
    }

    fn project_labels(&mut self, statement: StatementId, destination: Ptr<BasicBlock>) -> bool {
        let labels: Vec<_> = self
            .document
            .labels_for_statement(statement)
            .cloned()
            .collect();
        for label in &labels {
            let operation = PtxLabelOp::build(self.ctx, label.name()).get_operation();
            self.record_operation(
                operation,
                SourceNode::Label { label: label.id() },
                label.span(),
                destination,
            );
        }
        !labels.is_empty()
    }
}

fn trim_header(text: &str) -> &str {
    text.trim().trim_end_matches([';', '{']).trim_end()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{PtxCallableOp, PtxDirectiveOp, PtxInstructionOp, PtxLabelOp, PtxScopeOp};
    use pliron::common_traits::Verify;
    use pliron::context::Context;
    use pliron::linked_list::ContainsLinkedList;
    use pliron::op::Op;

    #[test]
    fn projects_module_and_callable_structure_without_source_attributes() {
        let source = "\
.version 8.9
.target sm_120a
.visible .entry kernel() {
    .reg .pred %p<2>;
L0:
    @%p0 future.op.u32 {%r1, %r2}, [%rd3];
    {
        mov.u32 %r1, 7;
    }
    ret;
}
";
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        let projection = Projection::parse(&mut ctx, source).unwrap();
        assert_eq!(projection.document().source(), source);

        let module_ops: Vec<_> = projection
            .module()
            .body(&ctx)
            .deref(&ctx)
            .iter(&ctx)
            .collect();
        assert_eq!(module_ops.len(), 3);
        assert!(Operation::is_op::<PtxDirectiveOp>(module_ops[0], &ctx));
        assert!(Operation::is_op::<PtxDirectiveOp>(module_ops[1], &ctx));
        let callable = Operation::get_op::<PtxCallableOp>(module_ops[2], &ctx).unwrap();
        assert!(callable.is_definition(&ctx));

        let callable_ops: Vec<_> = callable
            .entry_block(&ctx)
            .unwrap()
            .deref(&ctx)
            .iter(&ctx)
            .collect();
        assert!(Operation::is_op::<PtxDirectiveOp>(callable_ops[0], &ctx));
        assert!(Operation::is_op::<PtxLabelOp>(callable_ops[1], &ctx));
        assert!(Operation::is_op::<PtxInstructionOp>(callable_ops[2], &ctx));
        assert!(Operation::is_op::<PtxScopeOp>(callable_ops[3], &ctx));
        assert!(Operation::is_op::<PtxInstructionOp>(callable_ops[4], &ctx));

        assert_eq!(projection.blocks().len(), 3);
        for node in projection.nodes() {
            assert_eq!(
                projection.source_node(node.operation()),
                Some(node.source_node())
            );
            assert_eq!(
                projection.operation_for_source(node.source_node()),
                Some(node.operation())
            );
            if let SourceNode::Label { label } = node.source_node() {
                assert_eq!(
                    projection.document().label(label).unwrap().span(),
                    node.source_span()
                );
            }
        }
        for block in projection.blocks() {
            assert_eq!(
                projection.source_scope(block.block()),
                Some(block.source_scope())
            );
            assert_eq!(
                projection.block_for_source_scope(block.source_scope()),
                Some(block.block())
            );
        }
        projection
            .module()
            .get_operation()
            .deref(&ctx)
            .verify(&ctx)
            .unwrap();
    }

    #[test]
    fn projects_predicates_into_typed_attributes() {
        let source = "\
.visible .entry kernel() {
    @%p0 add.u32 %r0, %r0, 1;
    @!%p2 bra L0;
L0:
    ret;
}
";
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        let projection = Projection::parse(&mut ctx, source).unwrap();
        let callable_op = projection
            .module()
            .body(&ctx)
            .deref(&ctx)
            .iter(&ctx)
            .next()
            .unwrap();
        let callable = Operation::get_op::<PtxCallableOp>(callable_op, &ctx).unwrap();
        let predicates: Vec<_> = callable
            .entry_block(&ctx)
            .unwrap()
            .deref(&ctx)
            .iter(&ctx)
            .filter_map(|operation| Operation::get_op::<PtxInstructionOp>(operation, &ctx))
            .map(|instruction| instruction.predicate(&ctx))
            .collect();
        assert_eq!(
            predicates,
            vec![
                Some(PredicateAttr::new("%p0", false)),
                Some(PredicateAttr::new("%p2", true)),
                None
            ]
        );
    }

    #[test]
    fn projects_declarations_without_inventing_body_regions() {
        let source = ".extern .func helper(.param .b32 x);\n";
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        let projection = Projection::parse(&mut ctx, source).unwrap();
        let operation = projection
            .module()
            .body(&ctx)
            .deref(&ctx)
            .iter(&ctx)
            .next()
            .unwrap();
        let callable = Operation::get_op::<PtxCallableOp>(operation, &ctx).unwrap();
        assert!(!callable.is_definition(&ctx));
        assert!(callable.is_external(&ctx));
    }

    #[test]
    fn keeps_labeled_directives_structural_and_on_one_statement() {
        let source = ".version 9.3\n.entry kernel() { targets: .branchtargets L0; L0: ret; }\n";
        let document = Document::parse(source).unwrap();
        let table = document
            .directives()
            .iter()
            .find(|directive| directive.name() == ".branchtargets")
            .unwrap();
        let label = document
            .labels_for_statement(table.statement())
            .next()
            .unwrap()
            .id();
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        let projection = Projection::from_document(&mut ctx, document);
        let operation = projection.operation_for_label(label).unwrap();
        assert!(Operation::get_op::<PtxDirectiveOp>(operation, &ctx).is_some());
        let emitted = crate::emit_canonical_module(&ctx, &projection.module()).unwrap();
        assert!(emitted.contains("targets: .branchtargets L0;"));
    }

    #[test]
    fn joins_recovered_cfg_to_projected_operations_without_materializing_blocks() {
        let source = "\
.version 9.3
.target sm_120a
.visible .entry kernel() {
L0: Alias: @%p0 bra Done;
    {
        add.u32 %r0, %r0, 1;
    }
Done:
    ret;
}
";
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        let projection = Projection::parse(&mut ctx, source).unwrap();
        let cfg = projection.projected_control_flow().unwrap();
        let callable = cfg.callables().next().unwrap();
        let callable_operation = callable.operation();
        assert!(Operation::is_op::<PtxCallableOp>(callable_operation, &ctx));
        assert_eq!(
            cfg.for_callable_operation(callable_operation)
                .unwrap()
                .recovered()
                .callable(),
            callable.recovered().callable()
        );

        let blocks = callable.blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 3);
        let labels = blocks[0].labels().collect::<Vec<_>>();
        assert_eq!(labels.len(), 2);
        assert_eq!(
            labels
                .iter()
                .map(|(_, operation)| {
                    Operation::get_op::<PtxLabelOp>(*operation, &ctx)
                        .unwrap()
                        .name(&ctx)
                })
                .collect::<Vec<_>>(),
            ["L0", "Alias"]
        );
        assert_eq!(
            cfg.block_for_operation(labels[0].1).unwrap().id(),
            blocks[0].id()
        );
        for block in &blocks {
            for (statement, operation) in block.instructions() {
                assert_eq!(
                    projection.source_node(operation),
                    Some(SourceNode::Statement { statement })
                );
                assert!(Operation::is_op::<PtxInstructionOp>(operation, &ctx));
                assert_eq!(cfg.block_for_operation(operation).unwrap().id(), block.id());
            }
            for segment in block.scope_segments() {
                assert_eq!(
                    projection.source_scope(segment.lexical_block()),
                    Some(segment.scope())
                );
                for (statement, _) in segment.instructions() {
                    assert_eq!(
                        projection.document().statement(statement).unwrap().scope(),
                        segment.scope()
                    );
                }
            }
        }

        // Projection stays in its original lexical form: callable body plus
        // nested scope, not three newly-materialized Pliron CFG blocks.
        let projected_callable =
            Operation::get_op::<PtxCallableOp>(callable_operation, &ctx).unwrap();
        let region = projected_callable.region(&ctx).unwrap();
        assert_eq!(region.deref(&ctx).iter(&ctx).count(), 1);
    }
}
