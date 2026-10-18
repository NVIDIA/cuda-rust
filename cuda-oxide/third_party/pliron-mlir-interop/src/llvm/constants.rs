// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-mlir-interop contributors

//! MLIR translation for LLVM constant values.
//!
//! - Scalar, array, vector of scalars translate to a single MLIR attribute.
//! - Others (struct / symbol address etc) do not have an MLIR attribute.
//!   [lower_for_mlir](crate::llvm::lowering::lower_for_mlir) replaces them
//!   with operations that build them.

use std::fmt;

use pliron::{
    attribute::{Attribute, attr_cast},
    builtin::{
        attr_interfaces::TypedAttrInterface,
        attributes::{FPBFloatAttr, FPDoubleAttr, FPHalfAttr, FPSingleAttr, IntegerAttr},
        types::{BF16Type, FP16Type, FP32Type, FP64Type, IntegerType, Signedness},
    },
    context::Context,
    input_err_noloc,
    printable::State,
    result::Result,
    r#type::TypeHandle,
    utils::apfloat::Float,
};
use pliron_llvm::{
    attributes::{AggregateAttr, BytesAttr, SplatAttr},
    types::{ArrayType, VectorType},
};

use crate::{
    Error,
    printers::{mlir_hex_float, print_type},
};

/// The type of the constant `value`.
pub(crate) fn constant_type(ctx: &Context, value: &dyn Attribute) -> Result<TypeHandle> {
    match attr_cast::<dyn TypedAttrInterface>(value) {
        Some(typed) => Ok(typed.get_type(ctx)),
        None => input_err_noloc!(Error::Untranslatable(format!(
            "constant `{}` has no type",
            value.get_attr_id()
        ))),
    }
}

/// The MLIR literal of the scalar constant `value`, without its type.
///
/// Floats are in hexadecimal, so that their exact bits stay the same.
fn scalar_literal(ctx: &Context, value: &dyn Attribute) -> Option<String> {
    if let Some(int) = value.downcast_ref::<IntegerAttr>() {
        let signed = int.get_type().deref(ctx).signedness() == Signedness::Signed;
        Some(int.value().to_string_decimal(signed))
    } else if let Some(fp) = value.downcast_ref::<FPHalfAttr>() {
        Some(mlir_hex_float(fp.0.to_bits(), 16))
    } else if let Some(fp) = value.downcast_ref::<FPBFloatAttr>() {
        Some(mlir_hex_float(fp.0.to_bits(), 16))
    } else if let Some(fp) = value.downcast_ref::<FPSingleAttr>() {
        Some(mlir_hex_float(fp.0.to_bits(), 32))
    } else {
        value
            .downcast_ref::<FPDoubleAttr>()
            .map(|fp| mlir_hex_float(fp.0.to_bits(), 64))
    }
}

/// True if `ty` is an integer or a floating point type.
fn is_scalar_type(ctx: &Context, ty: TypeHandle) -> bool {
    let ty = ty.deref(ctx);
    ty.is::<IntegerType>()
        || ty.is::<FP16Type>()
        || ty.is::<BF16Type>()
        || ty.is::<FP32Type>()
        || ty.is::<FP64Type>()
}

/// The (nested) element list of a `dense<...>` attribute for the array or
/// vector constant `value`. `None` if an element is not a scalar or an
/// array of scalars.
fn dense_literal(ctx: &Context, value: &dyn Attribute) -> Option<String> {
    let elems: Vec<String> = if let Some(bytes) = value.downcast_ref::<BytesAttr>() {
        // MLIR's `i8` elements have a sign, so reinterpret each byte.
        bytes
            .as_ref()
            .iter()
            .map(|b| (*b as i8).to_string())
            .collect()
    } else {
        value
            .downcast_ref::<AggregateAttr>()?
            .elements()
            .iter()
            .map(|elem| {
                let elem: &dyn Attribute = &**elem;
                scalar_literal(ctx, elem).or_else(|| dense_literal(ctx, elem))
            })
            .collect::<Option<_>>()?
    };
    Some(format!("[{}]", elems.join(", ")))
}

/// The shape of a `dense<...>` attribute:
/// - `tensor<AxBxE>` for an array of arrays of scalars
/// - `vector<NxE>` for a vector of scalars.
struct DenseShape {
    dims: Vec<u64>,
    scalar: TypeHandle,
    /// For a vector: `Some(true)` if it is scalable. `None` for a tensor.
    vector: Option<bool>,
}

impl DenseShape {
    /// The `dense<...>` shape of `ty`. `None` if `ty` is not an array or vector of scalars.
    fn of(ctx: &Context, ty: TypeHandle) -> Option<Self> {
        let ty_ref = ty.deref(ctx);
        if let Some(vector) = ty_ref.downcast_ref::<VectorType>() {
            let scalar = vector.elem_type();
            return is_scalar_type(ctx, scalar).then(|| DenseShape {
                dims: vec![vector.num_elements().into()],
                scalar,
                vector: Some(vector.is_scalable()),
            });
        }
        let array = ty_ref.downcast_ref::<ArrayType>()?;
        let elem = array.elem_type();
        if is_scalar_type(ctx, elem) {
            return Some(DenseShape {
                dims: vec![array.size()],
                scalar: elem,
                vector: None,
            });
        }
        let mut shape = DenseShape::of(ctx, elem)?;
        if shape.vector.is_some() {
            // MLIR's dense array constants do not nest vectors in arrays.
            return None;
        }
        shape.dims.insert(0, array.size());
        Some(shape)
    }

    /// Print as `tensor<AxBxE>` or `vector<NxE>` (`vector<[N]xE>` if scalable).
    fn print(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        let dims: Vec<String> = self.dims.iter().map(u64::to_string).collect();
        match self.vector {
            Some(true) => write!(f, "vector<[{}]x", dims.join("x"))?,
            Some(false) => write!(f, "vector<{}x", dims.join("x"))?,
            None => write!(f, "tensor<{}x", dims.join("x"))?,
        }
        print_type(ctx, self.scalar, state, f)?;
        write!(f, ">")?;
        Ok(())
    }
}

/// Get the literal and the shape of the `dense<...>` attribute
/// (if it is one) for the constant `value`.
fn dense(ctx: &Context, value: &dyn Attribute) -> Option<(String, DenseShape)> {
    let shape = DenseShape::of(ctx, constant_type(ctx, value).ok()?)?;
    let literal = match value.downcast_ref::<SplatAttr>() {
        Some(splat) => scalar_literal(ctx, splat.element())?,
        None => dense_literal(ctx, value)?,
    };
    // MLIR cannot infer a shape from `[]`, but accepts `dense<>` for any empty shape.
    let literal = if shape.dims.contains(&0) {
        String::new()
    } else {
        literal
    };
    Some((literal, shape))
}

/// Print the constant `value` as a `dense<...>` MLIR attribute, if it has one.
///
/// Returns `false`, and prints nothing, if it does not.
pub fn print_dense(
    ctx: &Context,
    value: &dyn Attribute,
    state: &State,
    f: &mut fmt::Formatter<'_>,
) -> Result<bool> {
    let Some((literal, shape)) = dense(ctx, value) else {
        return Ok(false);
    };
    write!(f, "dense<{literal}> : ")?;
    shape.print(ctx, state, f)?;
    Ok(true)
}

/// True if MLIR can hold the constant `value` as an attribute.
pub fn is_constant_attr(ctx: &Context, value: &dyn Attribute) -> bool {
    scalar_literal(ctx, value).is_some() || dense(ctx, value).is_some()
}
