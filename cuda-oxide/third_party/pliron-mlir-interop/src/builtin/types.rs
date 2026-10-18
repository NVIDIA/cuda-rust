// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-mlir-interop contributors

//! MLIR translation for `builtin` dialect types.

use std::fmt;

use pliron::{
    builtin::type_interfaces::FunctionTypeInterface,
    builtin::types::{
        BF16Type, FP16Type, FP32Type, FP64Type, FunctionType, IntegerType, Signedness,
    },
    context::Context,
    derive::type_interface_impl,
    printable::State,
    result::Result,
};

use crate::{ToMlirType, printers::print_type_list};

/// `i32`, `si32` or `ui32`, matching MLIR's
/// [IntegerType](https://mlir.llvm.org/docs/Dialects/Builtin/#integertype).
#[type_interface_impl]
impl ToMlirType for IntegerType {
    fn to_mlir(&self, _ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        match self.signedness() {
            Signedness::Signed => write!(f, "si{}", self.width())?,
            Signedness::Unsigned => write!(f, "ui{}", self.width())?,
            Signedness::Signless => write!(f, "i{}", self.width())?,
        }
        Ok(())
    }
}

/// `(arg, ...) -> (res, ...)`, matching MLIR's
/// [FunctionType](https://mlir.llvm.org/docs/Dialects/Builtin/#functiontype).
#[type_interface_impl]
impl ToMlirType for FunctionType {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        write!(f, "(")?;
        print_type_list(ctx, self.arg_types(), state, f)?;
        write!(f, ") -> (")?;
        print_type_list(ctx, self.res_types(), state, f)?;
        write!(f, ")")?;
        Ok(())
    }
}

/// `f16`.
#[type_interface_impl]
impl ToMlirType for FP16Type {
    fn to_mlir(&self, _ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        write!(f, "f16")?;
        Ok(())
    }
}

/// `bf16`.
#[type_interface_impl]
impl ToMlirType for BF16Type {
    fn to_mlir(&self, _ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        write!(f, "bf16")?;
        Ok(())
    }
}

/// `f32`.
#[type_interface_impl]
impl ToMlirType for FP32Type {
    fn to_mlir(&self, _ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        write!(f, "f32")?;
        Ok(())
    }
}

/// `f64`.
#[type_interface_impl]
impl ToMlirType for FP64Type {
    fn to_mlir(&self, _ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        write!(f, "f64")?;
        Ok(())
    }
}
