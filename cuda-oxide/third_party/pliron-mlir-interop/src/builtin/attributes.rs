// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-mlir-interop contributors

//! MLIR translation for `builtin` dialect attributes.

use std::fmt;

use pliron::{
    builtin::attributes::{
        BoolAttr, DictAttr, FPBFloatAttr, FPDoubleAttr, FPHalfAttr, FPSingleAttr, IdentifierAttr,
        IntegerAttr, OperandSegmentSizesAttr, StringAttr, TypeAttr, UnitAttr, VecAttr,
    },
    builtin::types::Signedness,
    context::Context,
    derive::attr_interface_impl,
    identifier::Identifier,
    printable::State,
    result::Result,
    r#type::Typed,
    utils::apfloat::Float,
};

use crate::{
    ToMlirAttr,
    printers::{mlir_dense_array, mlir_hex_float, mlir_string_literal, print_attr, print_type},
};

/// `@name`, i.e. MLIR's
/// [FlatSymbolRefAttr](https://mlir.llvm.org/docs/Dialects/Builtin/#flatsymbolrefattr).
///
/// pliron uses this attribute wherever MLIR names a symbol (a callee, a global).
#[attr_interface_impl]
impl ToMlirAttr for IdentifierAttr {
    fn to_mlir(&self, _ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        let name: &Identifier = self.as_ref();
        write!(f, "@{name}")?;
        Ok(())
    }
}

/// A quoted MLIR string literal.
#[attr_interface_impl]
impl ToMlirAttr for StringAttr {
    fn to_mlir(&self, _ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        write!(f, "{}", mlir_string_literal(self.as_str()))?;
        Ok(())
    }
}

/// `true` or `false`.
#[attr_interface_impl]
impl ToMlirAttr for BoolAttr {
    fn to_mlir(&self, _ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        write!(f, "{}", bool::from(self.clone()))?;
        Ok(())
    }
}

/// `<value> : <type>`, MLIR's
/// [IntegerAttr](https://mlir.llvm.org/docs/Dialects/Builtin/#integerattr).
#[attr_interface_impl]
impl ToMlirAttr for IntegerAttr {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        let ty = IntegerAttr::get_type(self);
        let signed = ty.deref(ctx).signedness() == Signedness::Signed;
        write!(f, "{} : ", self.value().to_string_decimal(signed))?;
        print_type(ctx, ty.into(), state, f)
    }
}

/// `0xHHHH : f16`.
#[attr_interface_impl]
impl ToMlirAttr for FPHalfAttr {
    fn to_mlir(&self, _ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        write!(f, "{} : f16", mlir_hex_float(self.0.to_bits(), 16))?;
        Ok(())
    }
}

/// `0xHHHH : bf16`.
#[attr_interface_impl]
impl ToMlirAttr for FPBFloatAttr {
    fn to_mlir(&self, _ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        write!(f, "{} : bf16", mlir_hex_float(self.0.to_bits(), 16))?;
        Ok(())
    }
}

/// `0xHHHHHHHH : f32`.
#[attr_interface_impl]
impl ToMlirAttr for FPSingleAttr {
    fn to_mlir(&self, _ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        write!(f, "{} : f32", mlir_hex_float(self.0.to_bits(), 32))?;
        Ok(())
    }
}

/// `0xHHHHHHHHHHHHHHHH : f64`.
#[attr_interface_impl]
impl ToMlirAttr for FPDoubleAttr {
    fn to_mlir(&self, _ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        write!(f, "{} : f64", mlir_hex_float(self.0.to_bits(), 64))?;
        Ok(())
    }
}

/// `{key = value, ...}`, MLIR's
/// [DictionaryAttr](https://mlir.llvm.org/docs/Dialects/Builtin/#dictionaryattr).
#[attr_interface_impl]
impl ToMlirAttr for DictAttr {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        write!(f, "{{")?;
        for (i, (key, val)) in self.0.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{key} = ")?;
            print_attr(ctx, &**val, state, f)?;
        }
        write!(f, "}}")?;
        Ok(())
    }
}

/// `[value, ...]`, MLIR's
/// [ArrayAttr](https://mlir.llvm.org/docs/Dialects/Builtin/#arrayattr).
#[attr_interface_impl]
impl ToMlirAttr for VecAttr {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        write!(f, "[")?;
        for (i, val) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            print_attr(ctx, &**val, state, f)?;
        }
        write!(f, "]")?;
        Ok(())
    }
}

/// `unit`.
#[attr_interface_impl]
impl ToMlirAttr for UnitAttr {
    fn to_mlir(&self, _ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        write!(f, "unit")?;
        Ok(())
    }
}

/// The wrapped type, MLIR's
/// [TypeAttr](https://mlir.llvm.org/docs/Dialects/Builtin/#typeattr).
#[attr_interface_impl]
impl ToMlirAttr for TypeAttr {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        print_type(ctx, self.get_type(ctx), state, f)
    }
}

/// `array<i32: ...>`, matching the `DenseI32ArrayAttr` MLIR's
/// `AttrSizedOperandSegments` uses for `operandSegmentSizes`.
#[attr_interface_impl]
impl ToMlirAttr for OperandSegmentSizesAttr {
    fn to_mlir(&self, _ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        write!(f, "{}", mlir_dense_array("i32", &self.0))?;
        Ok(())
    }
}
