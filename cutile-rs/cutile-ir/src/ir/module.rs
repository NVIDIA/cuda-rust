/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Top-level module — the root IR container that owns all operations, values,
//! blocks, and regions via arena storage.

use std::collections::HashSet;

use super::attr::DenseElements;
use super::block::Block;
use super::location::Location;
use super::op::Operation;
use super::region::Region;
use super::types::Type;
use super::value::{BlockId, OpId, RegionId, Value, ValueData, ValueProducer};

use crate::Error;

/// A module-level global variable with static initialization.
///
/// Globals are allocated in GPU global memory at module load time.
/// Access them from kernels via `GetGlobal` + pointer load/store.
#[derive(Debug, Clone)]
pub struct Global {
    pub sym_name: String,
    pub value: DenseElements,
    pub alignment: u64,
    pub constant: bool,
    pub symbol_visibility: SymbolVisibility,
}

/// Visibility of a module global symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SymbolVisibility {
    Public = 0,
    Private = 1,
}

/// The root container for a Tile IR program.
///
/// A `Module` owns all IR objects via flat `Vec` arenas. References between
/// objects use index-based handles ([`Value`], [`OpId`], [`BlockId`],
/// [`RegionId`]) — no Rust lifetimes required.
///
/// Typical usage:
/// 1. Create a `Module`.
/// 2. Use the builder API ([`crate::builder`]) to populate it.
/// 3. Pass it to [`crate::bytecode::write_bytecode`] for serialization.
#[derive(Debug, Clone)]
pub struct Module {
    pub name: String,
    /// Optional producer identity, carried in its own bytecode section (13.3+).
    pub producer: Option<String>,

    // ------ Arenas ------
    pub(crate) values: Vec<ValueData>,
    pub(crate) operations: Vec<Operation>,
    pub(crate) blocks: Vec<Block>,
    pub(crate) regions: Vec<Region>,

    /// The top-level entry functions in this module.
    pub functions: Vec<OpId>,

    /// Module-level global variables.
    pub globals: Vec<Global>,
}

impl Module {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            producer: None,
            values: Vec::new(),
            operations: Vec::new(),
            blocks: Vec::new(),
            regions: Vec::new(),
            functions: Vec::new(),
            globals: Vec::new(),
        }
    }

    // ------ Value arena ------

    pub fn alloc_value(&mut self, ty: Type, producer: ValueProducer) -> Value {
        let id = Value(self.values.len() as u32);
        self.values.push(ValueData { ty, producer });
        id
    }

    /// Number of values allocated so far. Usable as a dominance watermark:
    /// a value with `index() < num_values()` taken at some program point was
    /// allocated before that point.
    pub fn num_values(&self) -> usize {
        self.values.len()
    }

    pub fn value_data(&self, v: Value) -> &ValueData {
        &self.values[v.0 as usize]
    }

    pub fn value_type(&self, v: Value) -> &Type {
        &self.values[v.0 as usize].ty
    }

    // ------ Operation arena ------

    pub fn alloc_op(&mut self, op: Operation) -> OpId {
        let id = OpId(self.operations.len() as u32);
        self.operations.push(op);
        id
    }

    pub fn op(&self, id: OpId) -> &Operation {
        &self.operations[id.0 as usize]
    }

    pub fn op_mut(&mut self, id: OpId) -> &mut Operation {
        &mut self.operations[id.0 as usize]
    }

    // ------ Block arena ------

    pub fn alloc_block(&mut self, block: Block) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        self.blocks.push(block);
        id
    }

    pub fn block(&self, id: BlockId) -> &Block {
        &self.blocks[id.0 as usize]
    }

    pub fn block_mut(&mut self, id: BlockId) -> &mut Block {
        &mut self.blocks[id.0 as usize]
    }

    // ------ Region arena ------

    pub fn alloc_region(&mut self, region: Region) -> RegionId {
        let id = RegionId(self.regions.len() as u32);
        self.regions.push(region);
        id
    }

    pub fn region(&self, id: RegionId) -> &Region {
        &self.regions[id.0 as usize]
    }

    pub fn region_mut(&mut self, id: RegionId) -> &mut Region {
        &mut self.regions[id.0 as usize]
    }

    // ------ Location ------

    pub fn op_location(&self, id: OpId) -> &Location {
        &self.operations[id.0 as usize].location
    }

    // ------ Verification ------

    /// Verify that every operand in the module references a value that
    /// dominates its use point.
    ///
    /// In structured control flow (no CFG edges), dominance is simple:
    /// - Block arguments dominate all operations in the block.
    /// - An operation's results dominate later operations in the same block.
    /// - All values visible in a parent scope dominate operations in nested
    ///   regions (for, if, loop, reduce, scan bodies).
    ///
    /// Call this after building IR and before bytecode emission to catch
    /// value reference ordering bugs with clear error messages.
    pub fn verify_dominance(&self) -> std::result::Result<(), Error> {
        let mut errors = Vec::new();
        for &func_id in &self.functions {
            let func_op = self.op(func_id);
            let func_name = func_op
                .attributes
                .iter()
                .find(|(k, _)| k == "sym_name")
                .map(|(_, v)| format!("{v:?}"))
                .unwrap_or_else(|| format!("op#{}", func_id.0));

            // Function-level ops have no operands to check; start with their regions.
            let visible = HashSet::new();
            for &region_id in &func_op.regions {
                self.verify_region(region_id, &visible, &func_name, &mut errors);
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(Error::Verification(errors.join("\n")))
        }
    }

    fn verify_region(
        &self,
        region_id: RegionId,
        parent_visible: &HashSet<Value>,
        func_name: &str,
        errors: &mut Vec<String>,
    ) {
        let region = self.region(region_id);
        for &block_id in &region.blocks {
            self.verify_block(block_id, parent_visible, func_name, errors);
        }
    }

    fn verify_block(
        &self,
        block_id: BlockId,
        parent_visible: &HashSet<Value>,
        func_name: &str,
        errors: &mut Vec<String>,
    ) {
        let block = self.block(block_id);

        // Start with parent scope values + this block's arguments.
        let mut visible = parent_visible.clone();
        for &(arg_val, _) in &block.args {
            visible.insert(arg_val);
        }

        for &op_id in &block.ops {
            let op = self.op(op_id);

            // Check that every operand is visible at this point.
            for (i, &operand) in op.operands.iter().enumerate() {
                if !visible.contains(&operand) {
                    errors.push(format!(
                        "in {func_name}: operand {i} of {:?} (op#{}) references \
                         value %{} which does not dominate this use",
                        op.opcode, op_id.0, operand.0,
                    ));
                }
            }

            // Add this op's results to the visible set.
            for (ri, rt) in op.result_types.iter().enumerate() {
                // Find the Value allocated for this result.
                // Results are allocated immediately after the op, so search
                // for the matching ValueProducer.
                if let Some(val) = self.find_op_result(op_id, ri as u32) {
                    visible.insert(val);
                } else {
                    // Result value not found — likely a builder bug, but don't
                    // crash dominance checking for it.
                    let _ = rt;
                }
            }

            // Recurse into regions attached to this op.
            for &region_id in &op.regions {
                self.verify_region(region_id, &visible, func_name, errors);
            }
        }
    }

    /// Find the `Value` handle for result `result_index` of operation `op_id`.
    fn find_op_result(&self, op_id: OpId, result_index: u32) -> Option<Value> {
        // Linear scan — the value arena is small per-function and this runs
        // once for verification, not on the hot path.
        for (i, vd) in self.values.iter().enumerate() {
            if vd.producer
                == (ValueProducer::OpResult {
                    op: op_id,
                    result_index,
                })
            {
                return Some(Value(i as u32));
            }
        }
        None
    }

    // ------ Bytecode value-index verification ------

    /// Verify that the bytecode writer's value numbering will produce valid
    /// operand indices that the C++ reader (tileiras) can look up.
    ///
    /// This simulates the writer's sequential value numbering with
    /// block-scoped rollback, and checks:
    /// 1. Every operand is in the value map when referenced
    /// 2. Every op result has a matching Value in the arena
    /// 3. `next_idx` advances by exactly `result_types.len()` per op
    ///    (matching the reader, which always allocates that many values)
    pub fn verify_bytecode_indices(&self) -> std::result::Result<(), Error> {
        let mut errors = Vec::new();
        for &func_id in &self.functions {
            let func_op = self.op(func_id);
            let func_name = func_op
                .attributes
                .iter()
                .find(|(k, _)| k == "sym_name")
                .map(|(_, v)| format!("{v:?}"))
                .unwrap_or_else(|| format!("op#{}", func_id.0));

            let mut value_map: std::collections::HashMap<Value, u64> =
                std::collections::HashMap::new();
            let mut next_idx: u64 = 0;

            // Register function argument values from the entry block.
            if let Some(&region_id) = func_op.regions.first() {
                let region = self.region(region_id);
                if let Some(&entry_block) = region.blocks.first() {
                    let block = self.block(entry_block);
                    for (val, _) in &block.args {
                        value_map.insert(*val, next_idx);
                        next_idx += 1;
                    }
                    // Walk ops in the entry block (not via write_block —
                    // the function body writes entry block ops directly).
                    for &op_id in &block.ops {
                        self.verify_bc_operation(
                            op_id,
                            &mut value_map,
                            &mut next_idx,
                            &func_name,
                            &mut errors,
                        );
                    }
                }
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(Error::Verification(errors.join("\n")))
        }
    }

    fn verify_bc_operation(
        &self,
        op_id: OpId,
        value_map: &mut std::collections::HashMap<Value, u64>,
        next_idx: &mut u64,
        func_name: &str,
        errors: &mut Vec<String>,
    ) {
        let op = self.op(op_id);

        // Check operands: every operand must already be in the value map.
        for (i, &operand) in op.operands.iter().enumerate() {
            if !value_map.contains_key(&operand) {
                errors.push(format!(
                    "in {func_name}: bytecode operand {i} of {:?} (op#{}) \
                     references value %{} which is not in the bytecode \
                     value map (next_idx={}, map_size={})",
                    op.opcode,
                    op_id.0,
                    operand.0,
                    next_idx,
                    value_map.len(),
                ));
            }
        }

        // Check result count: for fixed-result ops, the reader uses a
        // hard-coded count. If our IR has a different number of result types,
        // the reader's value tracking will diverge from the writer's.
        if let Some(expected) = op.opcode.fixed_result_count() {
            if op.result_types.len() != expected {
                errors.push(format!(
                    "in {func_name}: {:?} (op#{}) has {} result types but the \
                     bytecode reader expects exactly {expected} (fixed count)",
                    op.opcode,
                    op_id.0,
                    op.result_types.len(),
                ));
            }
        }

        // Register results: the C++ reader always allocates result_types.len()
        // values, so the writer must too. Check that each result has a Value
        // in the arena.
        for i in 0..op.result_types.len() {
            match self.find_op_result(op_id, i as u32) {
                Some(v) => {
                    value_map.insert(v, *next_idx);
                    *next_idx += 1;
                }
                None => {
                    errors.push(format!(
                        "in {func_name}: {:?} (op#{}) declares {} result types \
                         but result {i} has no Value in the arena — the bytecode \
                         reader will allocate a value the writer never registered",
                        op.opcode,
                        op_id.0,
                        op.result_types.len(),
                    ));
                    // Still increment to stay in sync with what the reader does.
                    *next_idx += 1;
                }
            }
        }

        // Recurse into regions (nested blocks with scoped rollback).
        for &region_id in &op.regions {
            let region = self.region(region_id);
            for &block_id in &region.blocks {
                self.verify_bc_block(block_id, value_map, next_idx, func_name, errors);
            }
        }
    }

    fn verify_bc_block(
        &self,
        block_id: BlockId,
        value_map: &mut std::collections::HashMap<Value, u64>,
        next_idx: &mut u64,
        func_name: &str,
        errors: &mut Vec<String>,
    ) {
        let saved_next_idx = *next_idx;
        let block = self.block(block_id);

        // Block arguments.
        for (val, _) in &block.args {
            value_map.insert(*val, *next_idx);
            *next_idx += 1;
        }

        // Operations.
        for &op_id in &block.ops {
            self.verify_bc_operation(op_id, value_map, next_idx, func_name, errors);
        }

        // Rollback: remove block-scoped values.
        value_map.retain(|_, v| *v < saved_next_idx);
        *next_idx = saved_next_idx;
    }
}
