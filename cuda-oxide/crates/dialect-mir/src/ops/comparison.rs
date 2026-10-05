/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! MIR comparison operations.
//!
//! This module defines relational and equality comparison operations for the MIR dialect.

use pliron::{
    builtin::{
        op_interfaces::{NOpdsInterface, NResultsInterface, OneResultInterface},
        types::IntegerType,
    },
    common_traits::Verify,
    context::Context,
    location::Located,
    op::Op,
    printable::Printable,
    result::Error,
    r#type::Typed,
    verify_err,
};
use pliron_derive::pliron_op;

use crate::types::MirEnumType;

// ============================================================================
// Relational Comparisons
// ============================================================================

/// MIR less than comparison (result is bool).
///
/// # Verification
///
/// - Must have exactly 2 operands.
/// - Both operands must have the same type.
/// - Result must be `i1`.
#[pliron_op(
    name = "mir.lt",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface]
)]
pub struct MirLtOp;

impl Verify for MirLtOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let lhs = op.get_operand(0);
        let rhs = op.get_operand(1);
        let res = op.get_result(0);

        let lhs_ty = lhs.get_type(ctx);
        let rhs_ty = rhs.get_type(ctx);
        let res_ty = res.get_type(ctx);
        let res_ty_obj = res_ty.deref(ctx);

        if lhs_ty != rhs_ty {
            return verify_err!(
                op.loc(),
                "MirLtOp operands must be of the same type. LHS: {}, RHS: {}",
                lhs_ty.disp(ctx),
                rhs_ty.disp(ctx)
            );
        }

        if let Some(int_ty) = res_ty_obj.downcast_ref::<IntegerType>() {
            if int_ty.width() != 1 {
                return verify_err!(op.loc(), "MirLtOp result must be i1");
            }
        } else {
            return verify_err!(op.loc(), "MirLtOp result must be integer type (i1)");
        }

        Ok(())
    }
}

/// MIR less than or equal comparison (result is bool).
///
/// # Verification
///
/// - Must have exactly 2 operands.
/// - Both operands must have the same type.
/// - Result must be `i1`.
#[pliron_op(
    name = "mir.le",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface]
)]
pub struct MirLeOp;

impl Verify for MirLeOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let lhs = op.get_operand(0);
        let rhs = op.get_operand(1);
        let res = op.get_result(0);

        let lhs_ty = lhs.get_type(ctx);
        let rhs_ty = rhs.get_type(ctx);
        let res_ty = res.get_type(ctx);
        let res_ty_obj = res_ty.deref(ctx);

        if lhs_ty != rhs_ty {
            return verify_err!(op.loc(), "MirLeOp operands must be of the same type");
        }
        if let Some(int_ty) = res_ty_obj.downcast_ref::<IntegerType>() {
            if int_ty.width() != 1 {
                return verify_err!(op.loc(), "MirLeOp result must be i1");
            }
        } else {
            return verify_err!(op.loc(), "MirLeOp result must be integer type (i1)");
        }
        Ok(())
    }
}

/// MIR greater than comparison (result is bool).
///
/// # Verification
///
/// - Must have exactly 2 operands.
/// - Both operands must have the same type.
/// - Result must be `i1`.
#[pliron_op(
    name = "mir.gt",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface]
)]
pub struct MirGtOp;

impl Verify for MirGtOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let lhs = op.get_operand(0);
        let rhs = op.get_operand(1);
        let res = op.get_result(0);

        let lhs_ty = lhs.get_type(ctx);
        let rhs_ty = rhs.get_type(ctx);
        let res_ty = res.get_type(ctx);
        let res_ty_obj = res_ty.deref(ctx);

        if lhs_ty != rhs_ty {
            return verify_err!(op.loc(), "MirGtOp operands must be of the same type");
        }
        if let Some(int_ty) = res_ty_obj.downcast_ref::<IntegerType>() {
            if int_ty.width() != 1 {
                return verify_err!(op.loc(), "MirGtOp result must be i1");
            }
        } else {
            return verify_err!(op.loc(), "MirGtOp result must be integer type (i1)");
        }
        Ok(())
    }
}

/// MIR greater than or equal comparison (result is bool).
///
/// # Verification
///
/// - Must have exactly 2 operands.
/// - Both operands must have the same type.
/// - Result must be `i1`.
#[pliron_op(
    name = "mir.ge",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface]
)]
pub struct MirGeOp;

impl Verify for MirGeOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let lhs = op.get_operand(0);
        let rhs = op.get_operand(1);
        let res = op.get_result(0);

        let lhs_ty = lhs.get_type(ctx);
        let rhs_ty = rhs.get_type(ctx);
        let res_ty = res.get_type(ctx);
        let res_ty_obj = res_ty.deref(ctx);

        if lhs_ty != rhs_ty {
            return verify_err!(op.loc(), "MirGeOp operands must be of the same type");
        }
        if let Some(int_ty) = res_ty_obj.downcast_ref::<IntegerType>() {
            if int_ty.width() != 1 {
                return verify_err!(op.loc(), "MirGeOp result must be i1");
            }
        } else {
            return verify_err!(op.loc(), "MirGeOp result must be integer type (i1)");
        }
        Ok(())
    }
}

/// MIR three-way comparison (`Ord::cmp`, result is `core::cmp::Ordering`).
///
/// # Verification
///
/// - Must have exactly 2 operands.
/// - Both operands must have the same type.
/// - Operands must be integers (covers `iN`/`uN` plus `bool` and `char`,
///   which the type translator models as `i1` and `ui32`). rustc never
///   emits `BinOp::Cmp` for floats (they are not `Ord`), and the lowering
///   has no float total-order support, so float operands are rejected
///   here instead of reaching the lowering.
/// - Result must be a fieldless 3-variant enum (the `Ordering` shape).
#[pliron_op(
    name = "mir.cmp",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface]
)]
pub struct MirCmpOp;

impl Verify for MirCmpOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let lhs = op.get_operand(0);
        let rhs = op.get_operand(1);
        let res = op.get_result(0);

        let lhs_ty = lhs.get_type(ctx);
        let rhs_ty = rhs.get_type(ctx);
        if lhs_ty != rhs_ty {
            return verify_err!(op.loc(), "MirCmpOp operands must be of the same type");
        }
        if !lhs_ty.deref(ctx).is::<IntegerType>() {
            return verify_err!(
                op.loc(),
                "MirCmpOp operands must be integers (bool/char included); \
                 floats have no BinOp::Cmp in rustc"
            );
        }

        let res_ty = res.get_type(ctx);
        let res_ty_obj = res_ty.deref(ctx);
        let Some(enum_ty) = res_ty_obj.downcast_ref::<MirEnumType>() else {
            return verify_err!(op.loc(), "MirCmpOp result must be an enum type");
        };
        if enum_ty.variant_count() != 3 {
            return verify_err!(op.loc(), "MirCmpOp result enum must have three variants");
        }
        if enum_ty.variant_field_counts.iter().any(|&c| c != 0) {
            return verify_err!(
                op.loc(),
                "MirCmpOp result enum variants must be fieldless (Ordering shape)"
            );
        }

        Ok(())
    }
}

// ============================================================================
// Equality Comparisons
// ============================================================================

/// MIR equality comparison (result is bool).
///
/// # Verification
///
/// - Must have exactly 2 operands.
/// - Both operands must have the same type.
/// - Result must be `i1`.
#[pliron_op(
    name = "mir.eq",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface]
)]
pub struct MirEqOp;

impl Verify for MirEqOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let lhs = op.get_operand(0);
        let rhs = op.get_operand(1);
        let res = op.get_result(0);

        let lhs_ty = lhs.get_type(ctx);
        let rhs_ty = rhs.get_type(ctx);
        let res_ty = res.get_type(ctx);
        let res_ty_obj = res_ty.deref(ctx);

        if lhs_ty != rhs_ty {
            return verify_err!(op.loc(), "MirEqOp operands must be of the same type");
        }
        if let Some(int_ty) = res_ty_obj.downcast_ref::<IntegerType>() {
            if int_ty.width() != 1 {
                return verify_err!(op.loc(), "MirEqOp result must be i1");
            }
        } else {
            return verify_err!(op.loc(), "MirEqOp result must be integer type (i1)");
        }
        Ok(())
    }
}

/// MIR inequality comparison (result is bool).
///
/// # Verification
///
/// - Must have exactly 2 operands.
/// - Both operands must have the same type.
/// - Result must be `i1`.
#[pliron_op(
    name = "mir.ne",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface]
)]
pub struct MirNeOp;

impl Verify for MirNeOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let lhs = op.get_operand(0);
        let rhs = op.get_operand(1);
        let res = op.get_result(0);

        let lhs_ty = lhs.get_type(ctx);
        let rhs_ty = rhs.get_type(ctx);
        let res_ty = res.get_type(ctx);
        let res_ty_obj = res_ty.deref(ctx);

        if lhs_ty != rhs_ty {
            return verify_err!(op.loc(), "MirNeOp operands must be of the same type");
        }
        if let Some(int_ty) = res_ty_obj.downcast_ref::<IntegerType>() {
            if int_ty.width() != 1 {
                return verify_err!(op.loc(), "MirNeOp result must be i1");
            }
        } else {
            return verify_err!(op.loc(), "MirNeOp result must be integer type (i1)");
        }
        Ok(())
    }
}

/// Register comparison operations into the given context.
pub fn register(ctx: &mut Context) {
    MirLtOp::register(ctx);
    MirLeOp::register(ctx);
    MirGtOp::register(ctx);
    MirGeOp::register(ctx);
    MirCmpOp::register(ctx);
    MirEqOp::register(ctx);
    MirNeOp::register(ctx);
}
