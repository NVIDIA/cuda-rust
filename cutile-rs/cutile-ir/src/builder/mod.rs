/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Builder API for constructing Tile IR programs.
//!
//! Provides [`OpBuilder`] for constructing individual operations and
//! convenience helpers for common patterns (functions, blocks, regions).
//!
//! # Example
//!
//! ```ignore
//! use cutile_ir::builder::OpBuilder;
//! use cutile_ir::ir::*;
//! use cutile_ir::bytecode::Opcode;
//!
//! let mut module = Module::new("my_module");
//!
//! // Build a function body block.
//! let arg = module.alloc_value(
//!     Type::Scalar(ScalarType::F32),
//!     ValueProducer::BlockArg { block: BlockId(0), arg_index: 0 },
//! );
//! let block_id = module.alloc_block(Block {
//!     args: vec![(arg, Type::Scalar(ScalarType::F32))],
//!     ops: vec![],
//! });
//! let region_id = module.alloc_region(Region { blocks: vec![block_id] });
//!
//! // Build an entry operation.
//! let (entry_id, _results) = OpBuilder::new(Opcode::Entry, Location::Unknown)
//!     .attr("sym_name", Attribute::String("my_kernel".into()))
//!     .region(region_id)
//!     .build(&mut module);
//!
//! module.functions.push(entry_id);
//! ```

use crate::bytecode::Opcode;
use crate::ir::{
    Attribute, Block, BlockId, Location, Module, OpId, Operation, Region, RegionId, Type, Value,
    ValueProducer,
};

/// Builds a single [`Operation`] and registers it in a [`Module`].
///
/// Mirrors the `OperationBuilder` pattern from melior, but without any
/// lifetime parameters — all references are index-based.
pub struct OpBuilder {
    opcode: Opcode,
    location: Location,
    operands: Vec<Value>,
    result_types: Vec<Type>,
    attributes: Vec<(String, Attribute)>,
    regions: Vec<RegionId>,
}

impl OpBuilder {
    pub fn new(opcode: Opcode, location: Location) -> Self {
        Self {
            opcode,
            location,
            operands: Vec::new(),
            result_types: Vec::new(),
            attributes: Vec::new(),
            regions: Vec::new(),
        }
    }

    /// Add an SSA operand.
    pub fn operand(mut self, value: Value) -> Self {
        self.operands.push(value);
        self
    }

    /// Add multiple SSA operands.
    pub fn operands(mut self, values: impl IntoIterator<Item = Value>) -> Self {
        self.operands.extend(values);
        self
    }

    /// Declare a result type. One [`Value`] will be created per result type.
    pub fn result(mut self, ty: Type) -> Self {
        self.result_types.push(ty);
        self
    }

    /// Declare multiple result types.
    pub fn results(mut self, types: impl IntoIterator<Item = Type>) -> Self {
        self.result_types.extend(types);
        self
    }

    /// Add a named attribute.
    pub fn attr(mut self, name: impl Into<String>, value: Attribute) -> Self {
        self.attributes.push((name.into(), value));
        self
    }

    /// Add multiple named attributes.
    pub fn attrs(mut self, attrs: impl IntoIterator<Item = (String, Attribute)>) -> Self {
        self.attributes.extend(attrs);
        self
    }

    /// Attach a region to this operation.
    pub fn region(mut self, region: RegionId) -> Self {
        self.regions.push(region);
        self
    }

    /// Attach multiple regions.
    pub fn regions(mut self, regions: impl IntoIterator<Item = RegionId>) -> Self {
        self.regions.extend(regions);
        self
    }

    /// Consume the builder, allocate the operation and its result values
    /// in `module`, and return the operation ID plus result values.
    pub fn build(self, module: &mut Module) -> (OpId, Vec<Value>) {
        let op = Operation {
            opcode: self.opcode,
            operands: self.operands,
            result_types: self.result_types,
            attributes: self.attributes,
            regions: self.regions,
            location: self.location,
        };
        let op_id = module.alloc_op(op);

        // Allocate result values.
        let result_types: Vec<Type> = module.op(op_id).result_types.clone();
        let results: Vec<Value> = result_types
            .into_iter()
            .enumerate()
            .map(|(i, ty)| {
                module.alloc_value(
                    ty,
                    ValueProducer::OpResult {
                        op: op_id,
                        result_index: i as u32,
                    },
                )
            })
            .collect();

        (op_id, results)
    }
}

// ---------------------------------------------------------------------------
// Convenience helpers for common patterns
// ---------------------------------------------------------------------------

/// Create a block with typed arguments and register it in the module.
///
/// Returns the block ID and the values for each block argument.
pub fn build_block(module: &mut Module, arg_types: &[Type]) -> (BlockId, Vec<Value>) {
    // Pre-calculate the block ID (it'll be the next slot).
    let block_id = BlockId(module.blocks.len() as u32);

    let args: Vec<(Value, Type)> = arg_types
        .iter()
        .enumerate()
        .map(|(i, ty): (usize, &Type)| {
            let v = module.alloc_value(
                ty.clone(),
                ValueProducer::BlockArg {
                    block: block_id,
                    arg_index: i as u32,
                },
            );
            (v, ty.clone())
        })
        .collect();

    let id = module.alloc_block(Block {
        args,
        ops: Vec::new(),
    });
    debug_assert_eq!(id, block_id);

    let arg_values: Vec<Value> = module.block(id).args.iter().map(|(v, _)| *v).collect();
    (id, arg_values)
}

/// Create a region containing a single block and register both in the module.
///
/// Returns `(region_id, block_id, block_arg_values)`.
pub fn build_single_block_region(
    module: &mut Module,
    arg_types: &[Type],
) -> (RegionId, BlockId, Vec<Value>) {
    let (block_id, args) = build_block(module, arg_types);
    let region_id = module.alloc_region(Region {
        blocks: vec![block_id],
    });
    (region_id, block_id, args)
}

/// Append an operation to a block (push its ID onto the block's op list).
pub fn append_op(module: &mut Module, block: BlockId, op: OpId) {
    module.block_mut(block).ops.push(op);
}
