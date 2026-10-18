// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-mlir-interop contributors

//! MLIR translation for `builtin` dialect ops.

use std::fmt;

use pliron::{
    builtin::{
        op_interfaces::{OneRegionInterface, SymbolOpInterface},
        ops::{ConstantOp, FuncOp, ModuleOp},
    },
    context::Context,
    derive::op_interface_impl,
    linked_list::ContainsLinkedList,
    op::Op,
    printable::State,
    result::Result,
};

use crate::{
    ToMlirOp,
    printers::{GenericOp, mlir_string_literal},
};

/// `builtin.module`.
///
/// pliron's module is always named, MLIR's `sym_name` is optional.
#[op_interface_impl]
impl ToMlirOp for ModuleOp {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        GenericOp::new(self.get_operation(), "builtin.module")
            .prop_raw(
                "sym_name",
                mlir_string_literal(self.get_symbol_name(ctx).as_ref()),
            )
            .print(ctx, state, f)
    }
}

/// `func.func`. MLIR's builtin dialect has no function op; `builtin.func` is
/// modelled on `func.func`.
#[op_interface_impl]
impl ToMlirOp for FuncOp {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        let func_type = self
            .get_attr_builtin_func_type(ctx)
            .expect("FuncOp is missing its function type attribute");
        // A `func.func` with an empty body is a declaration, and MLIR does not
        // let a declaration keep the default (public) visibility.
        let declaration = self.get_region(ctx).deref(ctx).get_head().is_none();
        GenericOp::new(self.get_operation(), "func.func")
            .prop_attr("function_type", &*func_type)
            .prop_raw(
                "sym_name",
                mlir_string_literal(self.get_symbol_name(ctx).as_ref()),
            )
            .prop_raw_opt("sym_visibility", declaration.then_some("\"private\""))
            .min_regions(1)
            .print(ctx, state, f)
    }
}

/// `arith.constant`. MLIR's builtin dialect has no constant op.
#[op_interface_impl]
impl ToMlirOp for ConstantOp {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        let value = self.get_value(ctx);
        GenericOp::new(self.get_operation(), "arith.constant")
            .prop_attr("value", &*value)
            .print(ctx, state, f)
    }
}
