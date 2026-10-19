/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SSA value handles.
//!
//! Values are lightweight index-based handles into the module's value table.
//! No lifetime parameters — values can be stored and passed freely.

use super::types::Type;

/// Index-based handle to an SSA value in the IR.
///
/// Values are produced by operations (results) or by block arguments.
/// They are identified by a dense index that the bytecode writer maps
/// to sequential value numbering per-function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Value(pub(crate) u32);

impl Value {
    pub fn index(self) -> u32 {
        self.0
    }
}

/// Metadata stored for each value in the module's value arena.
#[derive(Debug, Clone)]
pub struct ValueData {
    pub ty: Type,
    /// Which operation produced this value (result), or which block
    /// owns it (block argument).
    pub producer: ValueProducer,
}

/// Where a value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueProducer {
    /// Result `result_index` of operation `OpId`.
    OpResult { op: OpId, result_index: u32 },
    /// Argument `arg_index` of block `BlockId`.
    BlockArg { block: BlockId, arg_index: u32 },
}

/// Index-based handle to an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpId(pub(crate) u32);

/// Index-based handle to a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub(crate) u32);

/// Index-based handle to a region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegionId(pub(crate) u32);
