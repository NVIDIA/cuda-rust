/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! MIR constant operations.
//!
//! This module defines constant value operations for the MIR dialect.

use pliron::{
    attribute::attr_cast,
    builtin::{
        attr_interfaces::TypedAttrInterface,
        attributes::{FPDoubleAttr, FPSingleAttr, IntegerAttr},
        op_interfaces::{NOpdsInterface, NResultsInterface, OneResultInterface},
    },
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::{TypeHandle, Typed},
    verify_err,
};
use pliron_derive::pliron_op;

use crate::attributes::MirFP16Attr;

// ============================================================================
// MirConstantOp
// ============================================================================

/// MIR constant operation.
///
/// Creates a constant integer value.
///
/// # Attributes
///
/// ```text
/// | Name    | Type        | Description           |
/// |---------|-------------|-----------------------|
/// | `value` | IntegerAttr | The integer constant  |
/// ```
///
/// # Results
///
/// ```text
/// | Name  | Type                         |
/// |-------|------------------------------|
/// | `res` | Same type as value attribute |
/// ```
///
/// # Verification
///
/// - Must have `value` attribute.
/// - Result type must match the value attribute type.
#[pliron_op(
    name = "mir.constant",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>, OneResultInterface],
    attributes = (value: IntegerAttr)
)]
pub struct MirConstantOp;

impl MirConstantOp {
    /// Create a new MirConstantOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirConstantOp { op }
    }
}

impl Verify for MirConstantOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        // Check result type matches value attribute type
        let val_attr = match self.get_attr_value(ctx) {
            Some(attr) => attr,
            None => return verify_err!(op.loc(), "MirConstantOp missing value attribute"),
        };

        let typed_attr = match attr_cast::<dyn TypedAttrInterface>(&*val_attr) {
            Some(t) => t,
            None => return verify_err!(op.loc(), "Value attribute is not typed"),
        };
        let val_ty = typed_attr.get_type(ctx);

        let res = op.get_result(0);
        let res_ty = res.get_type(ctx);

        if val_ty != res_ty {
            return verify_err!(
                op.loc(),
                "MirConstantOp result type must match constant value type"
            );
        }
        Ok(())
    }
}

// ============================================================================
// MirFloatConstantOp
// ============================================================================

/// MIR float constant operation.
///
/// Creates a floating-point constant value.
///
/// # Attributes
///
/// ```text
/// | Name             | Type         | Description                        |
/// |------------------|--------------|-----------------------------------|
/// | `float_value_f16`| MirFP16Attr  | The f16 floating-point constant   |
/// | `float_value`    | FPSingleAttr | The f32 floating-point constant   |
/// | `float_value_f64`| FPDoubleAttr | The f64 floating-point constant   |
/// ```
///
/// Only one float attribute should be set.
///
/// # Results
///
/// ```text
/// | Name  | Type                               |
/// |-------|------------------------------------|
/// | `res` | Same type as float_value attribute |
/// ```
///
/// # Verification
///
/// - Must have exactly one float attribute.
/// - Result type must match the value attribute type.
#[pliron_op(
    name = "mir.float_constant",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>, OneResultInterface],
    attributes = (float_value_f16: MirFP16Attr, float_value: FPSingleAttr, float_value_f64: FPDoubleAttr)
)]
pub struct MirFloatConstantOp;

impl MirFloatConstantOp {
    /// Create a new MirFloatConstantOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirFloatConstantOp { op }
    }
}

impl Verify for MirFloatConstantOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        let attrs = [
            self.get_attr_float_value_f16(ctx)
                .map(|attr| Typed::get_type(&*attr, ctx)),
            self.get_attr_float_value(ctx)
                .map(|attr| attr.get_type(ctx)),
            self.get_attr_float_value_f64(ctx)
                .map(|attr| attr.get_type(ctx)),
        ];
        let set_count = attrs.iter().filter(|attr| attr.is_some()).count();
        if set_count != 1 {
            return verify_err!(
                op.loc(),
                "MirFloatConstantOp must have exactly one float value attribute"
            );
        };
        let val_ty = attrs.into_iter().flatten().next().unwrap();

        let res = op.get_result(0);
        let res_ty = res.get_type(ctx);

        if val_ty != res_ty {
            return verify_err!(
                op.loc(),
                "MirFloatConstantOp result type must match constant value type"
            );
        }
        Ok(())
    }
}

// ============================================================================
// MirUndefOp
// ============================================================================

/// MIR undefined value.
///
/// Represents an uninitialized value of a given type. Used by the `mem2reg`
/// pass as the default reaching definition when a load is not dominated by
/// any store to its alloca slot.
///
/// # Results
///
/// ```text
/// | Name  | Type     |
/// |-------|----------|
/// | `res` | Any type |
/// ```
///
/// # Verification
///
/// Structural only (no operands, exactly one result). The generated
/// `verifier = "succ"` impl always succeeds.
#[pliron_op(
    name = "mir.undef",
    format,
    interfaces = [NOpdsInterface<0>, OneResultInterface, NResultsInterface<1>],
    verifier = "succ"
)]
pub struct MirUndefOp;

impl MirUndefOp {
    /// Create a new `MirUndefOp` producing a single result of `result_ty`.
    pub fn new(ctx: &mut Context, result_ty: TypeHandle) -> Self {
        MirUndefOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result_ty],
                vec![],
                vec![],
                0,
            ),
        }
    }
}

/// Register constant operations into the given context.
pub fn register(ctx: &mut Context) {
    MirConstantOp::register(ctx);
    MirFloatConstantOp::register(ctx);
    MirUndefOp::register(ctx);
}
