/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Structural inspection of `cutile_ir::Module`.
//!
//! `ir_text()` cannot answer questions about which SSA value an operand uses:
//! the printer renders several operations as `<structural-op>` and does not
//! identify an operand's producer. These helpers inspect `cutile_ir::Module`
//! directly, allowing tests to assert the producer and type of a specific
//! operand.

use cutile_ir::bytecode::Opcode;
use cutile_ir::ir::{OpId, Type, Value, ValueProducer};
use cutile_ir::Module;

/// A borrowed module, queried structurally. `Copy`, so the iterators below
/// borrow the module rather than the inspection.
#[derive(Clone, Copy)]
pub struct ModuleInspection<'a> {
    module: &'a Module,
}

/// Grants [`Module`] the [`inspect`](ModuleInspectionExt::inspect) method.
/// Import it at the call site for `.inspect()` to resolve.
pub trait ModuleInspectionExt {
    /// Borrows this module for structural queries.
    fn inspect(&self) -> ModuleInspection<'_>;
}

impl ModuleInspectionExt for Module {
    fn inspect(&self) -> ModuleInspection<'_> {
        ModuleInspection { module: self }
    }
}

impl<'a> ModuleInspection<'a> {
    /// Immediate child operations of `op`, in region/block order.
    fn child_ops(self, op: OpId) -> impl Iterator<Item = OpId> + 'a {
        let module = self.module;
        module
            .op(op)
            .regions
            .iter()
            .copied()
            .flat_map(move |region| module.region(region).blocks.iter().copied())
            .flat_map(move |block| module.block(block).ops.iter().copied())
    }

    /// `op` and all its descendants, in preorder.
    fn nested_ops(self, op: OpId) -> Box<dyn Iterator<Item = OpId> + 'a> {
        Box::new(
            std::iter::once(op).chain(
                self.child_ops(op)
                    .flat_map(move |child| self.nested_ops(child)),
            ),
        )
    }

    /// Every operation in the module, nested regions included.
    ///
    /// A bounds check or a loop puts operations inside a nested region, so a
    /// flat walk over a function's own blocks misses them.
    fn ops_in_order(self) -> impl Iterator<Item = OpId> + 'a {
        self.module
            .functions
            .iter()
            .copied()
            .flat_map(move |function| self.nested_ops(function))
    }

    /// Operations with opcode `opcode`, in order.
    pub fn ops_with_opcode(self, opcode: Opcode) -> Vec<OpId> {
        let module = self.module;
        self.ops_in_order()
            .filter(move |&op| module.op(op).opcode == opcode)
            .collect()
    }

    /// `(producing opcode, value type)` of `value`; `None` for a block
    /// argument.
    ///
    /// Producer *and* type. SSA identity on its own proves nothing about a
    /// token slot: a handler that returned its view instead of its token
    /// handed back a value that was also reachable from the view, so two
    /// consumers of "the same value" can agree while both hold the wrong kind
    /// of thing.
    pub fn producer_and_type(self, value: Value) -> (Option<Opcode>, Type) {
        let producer = match self.module.value_data(value).producer {
            ValueProducer::OpResult { op, .. } => Some(self.module.op(op).opcode),
            ValueProducer::BlockArg { .. } => None,
        };
        (producer, self.module.value_type(value).clone())
    }
}
