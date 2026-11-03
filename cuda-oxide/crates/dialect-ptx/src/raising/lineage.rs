/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::cfg::BlockId;
use crate::ops::PtxModuleOp;
use crate::projection::SourceNode;
use pliron::basic_block::BasicBlock;
use pliron::context::Ptr;
use pliron::operation::Operation;
use ptx_parse::{EditMap, StatementId};
use std::collections::HashMap;
use std::ops::Range;

/// A native CFG together with its original and normalized source lineage.
pub struct NativeCfgProjection {
    pub(super) normalized_source: String,
    pub(super) edit_map: EditMap,
    pub(super) module: PtxModuleOp,
    pub(super) nodes: Vec<RaisedNode>,
    pub(super) nodes_by_operation: HashMap<Ptr<Operation>, usize>,
    pub(super) nodes_by_source: HashMap<SourceNode, usize>,
    pub(super) blocks: Vec<RaisedBlock>,
}

impl NativeCfgProjection {
    pub fn normalized_source(&self) -> &str {
        &self.normalized_source
    }

    pub fn edit_map(&self) -> &EditMap {
        &self.edit_map
    }

    pub fn module(&self) -> PtxModuleOp {
        self.module
    }

    pub fn nodes(&self) -> &[RaisedNode] {
        &self.nodes
    }

    pub fn blocks(&self) -> &[RaisedBlock] {
        &self.blocks
    }

    pub fn source_node(&self, operation: Ptr<Operation>) -> Option<SourceNode> {
        self.nodes_by_operation
            .get(&operation)
            .and_then(|index| self.nodes[*index].source_node)
    }

    pub fn operation_for_source(&self, source: SourceNode) -> Option<Ptr<Operation>> {
        self.nodes_by_source
            .get(&source)
            .map(|index| self.nodes[*index].operation)
    }
}

#[derive(Clone, Debug)]
pub struct RaisedNode {
    pub(super) operation: Ptr<Operation>,
    pub(super) source_node: Option<SourceNode>,
    pub(super) original_source_span: Option<Range<usize>>,
    pub(super) normalized_source_span: Option<Range<usize>>,
}

impl RaisedNode {
    pub fn operation(&self) -> Ptr<Operation> {
        self.operation
    }

    pub fn source_node(&self) -> Option<SourceNode> {
        self.source_node
    }

    pub fn source_span(&self) -> Option<Range<usize>> {
        self.original_source_span.clone()
    }

    pub fn original_source_span(&self) -> Option<Range<usize>> {
        self.original_source_span.clone()
    }

    pub fn normalized_source_span(&self) -> Option<Range<usize>> {
        self.normalized_source_span.clone()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RaisedBlock {
    pub(super) block: Ptr<BasicBlock>,
    pub(super) callable: StatementId,
    pub(super) source_block: BlockId,
}

impl RaisedBlock {
    pub fn block(self) -> Ptr<BasicBlock> {
        self.block
    }

    pub fn callable(self) -> StatementId {
        self.callable
    }

    pub fn source_block(self) -> BlockId {
        self.source_block
    }
}
