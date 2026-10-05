/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::{CallablePlan, ModuleItemPlan, NativeCfgPlan, NodePlan};
use crate::ops::{
    PtxBranchTargetsOp, PtxCallableOp, PtxDirectiveOp, PtxInstructionOp, PtxLabelOp, PtxModuleOp,
    PtxRawOp, PtxTerminatorOp, PtxTerminatorSpec,
};
use crate::projection::SourceNode;
use crate::raising::{NativeCfgProjection, RaisedBlock, RaisedNode};
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::op::Op;
use pliron::operation::Operation;
use ptx_parse::EditMap;
use std::collections::HashMap;
use std::ops::Range;

pub(super) fn materialize(plan: NativeCfgPlan, ctx: &mut Context) -> NativeCfgProjection {
    let module = PtxModuleOp::build(ctx);
    let destination = module.body(ctx);
    let mut nodes = Vec::new();
    let mut source_aliases = Vec::new();
    let mut blocks = Vec::with_capacity(plan.block_count);
    for item in plan.items {
        match item {
            ModuleItemPlan::Directive {
                statement,
                span,
                labels,
                name,
                arguments,
            } => {
                let operation = PtxDirectiveOp::build_labeled(
                    ctx,
                    labels
                        .iter()
                        .map(|label| text(&plan.normalized_source, &label.name)),
                    text(&plan.normalized_source, &name),
                    text(&plan.normalized_source, &arguments),
                )
                .get_operation();
                insert_node(
                    ctx,
                    operation,
                    destination,
                    Some(SourceNode::Statement { statement }),
                    Some(span),
                    &plan.edit_map,
                    &mut nodes,
                );
                source_aliases.extend(
                    labels
                        .into_iter()
                        .map(|label| (SourceNode::Label { label: label.id }, operation)),
                );
            }
            ModuleItemPlan::Raw { statement, span } => {
                let operation =
                    PtxRawOp::build(ctx, text(&plan.normalized_source, &span)).get_operation();
                insert_node(
                    ctx,
                    operation,
                    destination,
                    Some(SourceNode::Statement { statement }),
                    Some(span),
                    &plan.edit_map,
                    &mut nodes,
                );
            }
            ModuleItemPlan::Declaration(header) => {
                let operation = PtxCallableOp::build_declaration(
                    ctx,
                    text(&plan.normalized_source, &header.name),
                    header.kind,
                    header.is_extern,
                    text(&plan.normalized_source, &header.header),
                )
                .get_operation();
                insert_node(
                    ctx,
                    operation,
                    destination,
                    Some(SourceNode::Statement {
                        statement: header.statement,
                    }),
                    Some(header.span),
                    &plan.edit_map,
                    &mut nodes,
                );
            }
            ModuleItemPlan::Definition(callable) => {
                let mut sinks = MaterializationSinks {
                    nodes: &mut nodes,
                    blocks: &mut blocks,
                    source_aliases: &mut source_aliases,
                };
                materialize_callable(
                    ctx,
                    callable,
                    destination,
                    &plan.normalized_source,
                    &plan.edit_map,
                    &mut sinks,
                );
            }
        }
    }
    let nodes_by_operation: HashMap<Ptr<Operation>, usize> = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.operation, index))
        .collect();
    let mut nodes_by_source: HashMap<SourceNode, usize> = nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| Some((node.source_node?, index)))
        .collect();
    for (source, operation) in source_aliases {
        let index = *nodes_by_operation
            .get(&operation)
            .expect("source alias operation is raised");
        nodes_by_source.insert(source, index);
    }
    NativeCfgProjection {
        normalized_source: plan.normalized_source,
        edit_map: plan.edit_map,
        module,
        nodes,
        nodes_by_operation,
        nodes_by_source,
        blocks,
    }
}

struct MaterializationSinks<'output> {
    nodes: &'output mut Vec<RaisedNode>,
    blocks: &'output mut Vec<RaisedBlock>,
    source_aliases: &'output mut Vec<(SourceNode, Ptr<Operation>)>,
}

fn materialize_callable(
    ctx: &mut Context,
    callable: CallablePlan,
    destination: Ptr<BasicBlock>,
    source: &str,
    edit_map: &EditMap,
    sinks: &mut MaterializationSinks<'_>,
) {
    let operation = PtxCallableOp::build_cfg_definition(
        ctx,
        text(source, &callable.header.name),
        callable.header.kind,
        callable.header.is_extern,
        text(source, &callable.header.header),
    );
    insert_node(
        ctx,
        operation.get_operation(),
        destination,
        Some(SourceNode::Statement {
            statement: callable.header.statement,
        }),
        Some(callable.header.span),
        edit_map,
        sinks.nodes,
    );
    let cfg_body = operation
        .cfg_body(ctx)
        .expect("a CFG definition has a CFG body form");
    let block_ptrs = callable
        .blocks
        .iter()
        .map(|_| cfg_body.append_block(ctx))
        .collect::<Vec<_>>();
    for (block_plan, block) in callable.blocks.into_iter().zip(block_ptrs.iter().copied()) {
        sinks.blocks.push(RaisedBlock {
            block,
            callable: callable.header.statement,
            source_block: block_plan.id,
        });
        for node in block_plan.nodes {
            materialize_node(ctx, node, block, source, edit_map, sinks);
        }
        let successors = block_plan
            .terminator
            .successors
            .iter()
            .map(|index| block_ptrs[*index])
            .collect::<Vec<_>>();
        let terminator = PtxTerminatorOp::build(
            ctx,
            PtxTerminatorSpec {
                kind: block_plan.terminator.kind,
                predicate: block_plan.terminator.predicate,
                head: text(source, &block_plan.terminator.head),
                operands: block_plan
                    .terminator
                    .operands
                    .iter()
                    .map(|operand| text(source, operand))
                    .collect(),
                target_table: block_plan
                    .terminator
                    .target_table
                    .as_ref()
                    .map_or("", |table| text(source, table)),
                has_fallthrough: block_plan.terminator.has_fallthrough,
            },
            successors,
        )
        .get_operation();
        let (source_node, source_span) = block_plan
            .terminator
            .source
            .map_or((None, None), |(statement, span)| {
                (Some(SourceNode::Statement { statement }), Some(span))
            });
        insert_node(
            ctx,
            terminator,
            block,
            source_node,
            source_span,
            edit_map,
            sinks.nodes,
        );
    }
}

fn materialize_node(
    ctx: &mut Context,
    node: NodePlan,
    destination: Ptr<BasicBlock>,
    source_text: &str,
    edit_map: &EditMap,
    sinks: &mut MaterializationSinks<'_>,
) {
    let (operation, source, span, aliases) = match node {
        NodePlan::Label { label, span, name } => (
            PtxLabelOp::build(ctx, text(source_text, &name)).get_operation(),
            SourceNode::Label { label },
            span,
            Vec::new(),
        ),
        NodePlan::Directive {
            statement,
            span,
            labels,
            name,
            arguments,
        } => (
            PtxDirectiveOp::build_labeled(
                ctx,
                labels.iter().map(|label| text(source_text, &label.name)),
                text(source_text, &name),
                text(source_text, &arguments),
            )
            .get_operation(),
            SourceNode::Statement { statement },
            span,
            labels
                .into_iter()
                .map(|label| SourceNode::Label { label: label.id })
                .collect(),
        ),
        NodePlan::BranchTargets {
            statement,
            span,
            label,
            name,
        } => (
            PtxBranchTargetsOp::build(ctx, text(source_text, &name)).get_operation(),
            SourceNode::Statement { statement },
            span,
            vec![SourceNode::Label { label }],
        ),
        NodePlan::Instruction {
            statement,
            span,
            predicate,
            head,
            operands,
        } => (
            PtxInstructionOp::build(
                ctx,
                predicate,
                text(source_text, &head),
                operands.iter().map(|operand| text(source_text, operand)),
            )
            .get_operation(),
            SourceNode::Statement { statement },
            span,
            Vec::new(),
        ),
    };
    insert_node(
        ctx,
        operation,
        destination,
        Some(source),
        Some(span),
        edit_map,
        sinks.nodes,
    );
    sinks
        .source_aliases
        .extend(aliases.into_iter().map(|alias| (alias, operation)));
}

fn insert_node(
    ctx: &Context,
    operation: Ptr<Operation>,
    destination: Ptr<BasicBlock>,
    source_node: Option<SourceNode>,
    source_span: Option<Range<usize>>,
    edit_map: &EditMap,
    nodes: &mut Vec<RaisedNode>,
) {
    operation.insert_at_back(destination, ctx);
    let original_source_span = source_span
        .as_ref()
        .and_then(|span| edit_map.output_range_to_original(span.clone()));
    nodes.push(RaisedNode {
        operation,
        source_node,
        original_source_span,
        normalized_source_span: source_span,
    });
}

fn text<'source>(source: &'source str, span: &Range<usize>) -> &'source str {
    &source[span.clone()]
}
