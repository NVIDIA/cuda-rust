/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! MIR arithmetic operations.
//!
//! This module defines arithmetic, bitwise, and shift operations for the MIR dialect.

use pliron::{
    builtin::{
        op_interfaces::{NOpdsInterface, NResultsInterface, OneOpdInterface, OneResultInterface},
        types::IntegerType,
    },
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::Typed,
    verify_err,
};
use pliron_derive::pliron_op;

use crate::types::MirTupleType;

// ============================================================================
// Binary Arithmetic Operations
// ============================================================================

/// MIR add operation.
///
/// Integer or floating-point addition.
///
/// # Verification
///
/// - Both operands must have the same type.
/// - Result type must match operand types.
#[pliron_op(
    name = "mir.add",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface]
)]
pub struct MirAddOp;

impl Verify for MirAddOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let lhs = op.get_operand(0);
        let rhs = op.get_operand(1);
        let res = op.get_result(0);

        let lhs_ty = lhs.get_type(ctx);
        let rhs_ty = rhs.get_type(ctx);
        let res_ty = res.get_type(ctx);

        if lhs_ty != rhs_ty {
            return verify_err!(op.loc(), "MirAddOp operands must be of the same type");
        }
        if lhs_ty != res_ty {
            return verify_err!(op.loc(), "MirAddOp result type must match operand types");
        }
        Ok(())
    }
}

/// MIR sub operation.
///
/// Integer or floating-point subtraction.
///
/// # Verification
///
/// - Both operands must have the same type.
/// - Result type must match operand types.
#[pliron_op(
    name = "mir.sub",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface]
)]
pub struct MirSubOp;

impl Verify for MirSubOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let lhs = op.get_operand(0);
        let rhs = op.get_operand(1);
        let res = op.get_result(0);

        let lhs_ty = lhs.get_type(ctx);
        let rhs_ty = rhs.get_type(ctx);
        let res_ty = res.get_type(ctx);

        if lhs_ty != rhs_ty {
            return verify_err!(op.loc(), "MirSubOp operands must be of the same type");
        }
        if lhs_ty != res_ty {
            return verify_err!(op.loc(), "MirSubOp result type must match operand types");
        }
        Ok(())
    }
}

/// MIR mul operation.
///
/// Integer or floating-point multiplication.
///
/// # Verification
///
/// - Both operands must have the same type.
/// - Result type must match operand types.
#[pliron_op(
    name = "mir.mul",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface]
)]
pub struct MirMulOp;

impl Verify for MirMulOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let lhs = op.get_operand(0);
        let rhs = op.get_operand(1);
        let res = op.get_result(0);

        let lhs_ty = lhs.get_type(ctx);
        let rhs_ty = rhs.get_type(ctx);
        let res_ty = res.get_type(ctx);

        if lhs_ty != rhs_ty {
            return verify_err!(op.loc(), "MirMulOp operands must be of the same type");
        }
        if lhs_ty != res_ty {
            return verify_err!(op.loc(), "MirMulOp result type must match operand types");
        }
        Ok(())
    }
}

/// MIR div operation.
///
/// Integer or floating-point division.
///
/// # Verification
///
/// - Both operands must have the same type.
/// - Result type must match operand types.
#[pliron_op(
    name = "mir.div",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface]
)]
pub struct MirDivOp;

impl Verify for MirDivOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let lhs = op.get_operand(0);
        let rhs = op.get_operand(1);
        let res = op.get_result(0);

        let lhs_ty = lhs.get_type(ctx);
        let rhs_ty = rhs.get_type(ctx);
        let res_ty = res.get_type(ctx);

        if lhs_ty != rhs_ty {
            return verify_err!(op.loc(), "MirDivOp operands must be of the same type");
        }
        if lhs_ty != res_ty {
            return verify_err!(op.loc(), "MirDivOp result type must match operand types");
        }
        Ok(())
    }
}

/// MIR rem operation.
///
/// Integer or floating-point remainder.
///
/// # Verification
///
/// - Both operands must have the same type.
/// - Result type must match operand types.
#[pliron_op(
    name = "mir.rem",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface]
)]
pub struct MirRemOp;

impl Verify for MirRemOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let lhs = op.get_operand(0);
        let rhs = op.get_operand(1);
        let res = op.get_result(0);

        let lhs_ty = lhs.get_type(ctx);
        let rhs_ty = rhs.get_type(ctx);
        let res_ty = res.get_type(ctx);

        if lhs_ty != rhs_ty {
            return verify_err!(op.loc(), "MirRemOp operands must be of the same type");
        }
        if lhs_ty != res_ty {
            return verify_err!(op.loc(), "MirRemOp result type must match operand types");
        }
        Ok(())
    }
}

// ============================================================================
// Checked Arithmetic Operations
// ============================================================================

/// MIR checked add operation.
///
/// Integer addition with overflow flag.
///
/// # Operands
///
/// ```text
/// | Name  | Type        |
/// |-------|-------------|
/// | `lhs` | IntegerType |
/// | `rhs` | IntegerType |
/// ```
///
/// # Results
///
/// ```text
/// | Name  | Type                       |
/// |-------|----------------------------|
/// | `res` | MirTupleType of `(T, i1)`  |
/// ```
///
/// # Verification
///
/// - Operands must be same integer type.
/// - Result must be `MirTupleType` of `[T, i1]` where T is operand type.
#[pliron_op(
    name = "mir.checked_add",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface]
)]
pub struct MirCheckedAddOp;

impl MirCheckedAddOp {
    /// Create a new MirCheckedAddOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirCheckedAddOp { op }
    }
}

impl Verify for MirCheckedAddOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        let lhs = op.get_operand(0);
        let rhs = op.get_operand(1);
        let lhs_ty = lhs.get_type(ctx);
        let rhs_ty = rhs.get_type(ctx);

        if lhs_ty != rhs_ty {
            return verify_err!(
                op.loc(),
                "MirCheckedAddOp operands must be of the same type"
            );
        }
        // Ensure operands are integers
        if lhs_ty.deref(ctx).downcast_ref::<IntegerType>().is_none() {
            return verify_err!(op.loc(), "MirCheckedAddOp operands must be integers");
        }

        let res = op.get_result(0);
        let res_ty = res.get_type(ctx);
        let res_ty_obj = res_ty.deref(ctx);

        // Result must be a Tuple<(T, i1)>
        if let Some(tuple_ty) = res_ty_obj.downcast_ref::<MirTupleType>() {
            let subtypes = tuple_ty.get_types();
            if subtypes.len() != 2 {
                return verify_err!(
                    op.loc(),
                    "MirCheckedAddOp result tuple must have 2 elements"
                );
            }
            if subtypes[0] != lhs_ty {
                return verify_err!(
                    op.loc(),
                    "MirCheckedAddOp result tuple 0 must match operand type"
                );
            }

            let bool_ty = subtypes[1];
            let bool_ty_obj = bool_ty.deref(ctx);
            if let Some(int_ty) = bool_ty_obj.downcast_ref::<IntegerType>() {
                if int_ty.width() != 1 {
                    return verify_err!(op.loc(), "MirCheckedAddOp result tuple 1 must be i1");
                }
            } else {
                return verify_err!(
                    op.loc(),
                    "MirCheckedAddOp result tuple 1 must be integer type (i1)"
                );
            }
        } else {
            return verify_err!(op.loc(), "MirCheckedAddOp result must be a MirTupleType");
        }

        Ok(())
    }
}

/// MIR checked mul operation.
///
/// Signed or unsigned integer multiplication with overflow checking.
/// Returns a tuple of `(result, overflow_flag)`.
///
/// # Syntax
///
/// ```text
/// %res = mir.checked_mul %lhs, %rhs : (type, i1)
/// ```
///
/// # Semantics
///
/// Performs integer multiplication with overflow detection:
/// - First element: multiplication result (wrapping on overflow)
/// - Second element: boolean flag indicating overflow occurred
///
/// # Example
///
/// ```text
/// %0, %1 = mir.checked_mul %a, %b : (i32, i1)
/// // %0 = a * b (wrapping)
/// // %1 = true if overflow occurred
/// ```
///
/// # Table
///
/// ```text
/// | Name  | Type                       |
/// |-------|----------------------------|
/// | `lhs` | Integer type T             |
/// | `rhs` | Same as lhs                |
/// | `res` | MirTupleType of `(T, i1)`  |
/// ```
///
/// # Verification
///
/// - Operands must be same integer type.
/// - Result must be `MirTupleType` of `[T, i1]` where T is operand type.
#[pliron_op(
    name = "mir.checked_mul",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface]
)]
pub struct MirCheckedMulOp;

impl MirCheckedMulOp {
    /// Create a new MirCheckedMulOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirCheckedMulOp { op }
    }
}

impl Verify for MirCheckedMulOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        let lhs = op.get_operand(0);
        let rhs = op.get_operand(1);
        let lhs_ty = lhs.get_type(ctx);
        let rhs_ty = rhs.get_type(ctx);

        if lhs_ty != rhs_ty {
            return verify_err!(
                op.loc(),
                "MirCheckedMulOp operands must be of the same type"
            );
        }
        // Ensure operands are integers
        if lhs_ty.deref(ctx).downcast_ref::<IntegerType>().is_none() {
            return verify_err!(op.loc(), "MirCheckedMulOp operands must be integers");
        }

        let res = op.get_result(0);
        let res_ty = res.get_type(ctx);
        let res_ty_obj = res_ty.deref(ctx);

        // Result must be a Tuple<(T, i1)>
        if let Some(tuple_ty) = res_ty_obj.downcast_ref::<MirTupleType>() {
            let subtypes = tuple_ty.get_types();
            if subtypes.len() != 2 {
                return verify_err!(
                    op.loc(),
                    "MirCheckedMulOp result tuple must have 2 elements"
                );
            }
            if subtypes[0] != lhs_ty {
                return verify_err!(
                    op.loc(),
                    "MirCheckedMulOp result tuple 0 must match operand type"
                );
            }

            let bool_ty = subtypes[1];
            let bool_ty_obj = bool_ty.deref(ctx);
            if let Some(int_ty) = bool_ty_obj.downcast_ref::<IntegerType>() {
                if int_ty.width() != 1 {
                    return verify_err!(op.loc(), "MirCheckedMulOp result tuple 1 must be i1");
                }
            } else {
                return verify_err!(
                    op.loc(),
                    "MirCheckedMulOp result tuple 1 must be integer type (i1)"
                );
            }
        } else {
            return verify_err!(op.loc(), "MirCheckedMulOp result must be a MirTupleType");
        }

        Ok(())
    }
}

/// Checked subtraction: produces (result, overflow_flag) tuple
#[pliron_op(
    name = "mir.checked_sub",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface]
)]
pub struct MirCheckedSubOp;

impl MirCheckedSubOp {
    /// Create a new MirCheckedSubOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirCheckedSubOp { op }
    }
}

impl Verify for MirCheckedSubOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        let lhs = op.get_operand(0);
        let rhs = op.get_operand(1);
        let lhs_ty = lhs.get_type(ctx);
        let rhs_ty = rhs.get_type(ctx);

        if lhs_ty != rhs_ty {
            return verify_err!(
                op.loc(),
                "MirCheckedSubOp operands must be of the same type"
            );
        }
        // Ensure operands are integers
        if lhs_ty.deref(ctx).downcast_ref::<IntegerType>().is_none() {
            return verify_err!(op.loc(), "MirCheckedSubOp operands must be integers");
        }

        let res = op.get_result(0);
        let res_ty = res.get_type(ctx);
        let res_ty_obj = res_ty.deref(ctx);

        // Result must be a Tuple<(T, i1)>
        if let Some(tuple_ty) = res_ty_obj.downcast_ref::<MirTupleType>() {
            let subtypes = tuple_ty.get_types();
            if subtypes.len() != 2 {
                return verify_err!(
                    op.loc(),
                    "MirCheckedSubOp result tuple must have 2 elements"
                );
            }
            if subtypes[0] != lhs_ty {
                return verify_err!(
                    op.loc(),
                    "MirCheckedSubOp result tuple 0 must match operand type"
                );
            }

            let bool_ty = subtypes[1];
            let bool_ty_obj = bool_ty.deref(ctx);
            if let Some(int_ty) = bool_ty_obj.downcast_ref::<IntegerType>() {
                if int_ty.width() != 1 {
                    return verify_err!(op.loc(), "MirCheckedSubOp result tuple 1 must be i1");
                }
            } else {
                return verify_err!(
                    op.loc(),
                    "MirCheckedSubOp result tuple 1 must be integer type (i1)"
                );
            }
        } else {
            return verify_err!(op.loc(), "MirCheckedSubOp result must be a MirTupleType");
        }

        Ok(())
    }
}

// ============================================================================
// Unary Operations
// ============================================================================

/// MIR neg operation.
///
/// Integer or floating-point negation.
///
/// # Verification
///
/// - Result type must match operand type.
#[pliron_op(
    name = "mir.neg",
    format,
    interfaces = [NOpdsInterface<1>, OneOpdInterface, NResultsInterface<1>, OneResultInterface]
)]
pub struct MirNegOp;

impl Verify for MirNegOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let operand = op.get_operand(0);
        let res = op.get_result(0);

        let operand_ty = operand.get_type(ctx);
        let res_ty = res.get_type(ctx);

        if operand_ty != res_ty {
            return verify_err!(op.loc(), "MirNegOp result type must match operand type");
        }
        Ok(())
    }
}

/// MIR not operation.
///
/// Boolean/bitwise negation.
///
/// # Verification
///
/// - Must have exactly 1 operand and 1 result.
/// - Operand and result types must match.
#[pliron_op(
    name = "mir.not",
    format,
    interfaces = [NOpdsInterface<1>, OneOpdInterface, NResultsInterface<1>, OneResultInterface]
)]
pub struct MirNotOp;

impl Verify for MirNotOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let operand = op.get_operand(0);
        let res = op.get_result(0);

        let operand_ty = operand.get_type(ctx);
        let res_ty = res.get_type(ctx);

        if operand_ty != res_ty {
            return verify_err!(op.loc(), "MirNotOp result type must match operand type");
        }
        Ok(())
    }
}

// ============================================================================
// Shift Operations
// ============================================================================

/// MIR shr (shift right) operation.
///
/// Arithmetic or logical shift right depending on signedness.
///
/// # Verification
///
/// - Result type must match first operand type.
#[pliron_op(
    name = "mir.shr",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface]
)]
pub struct MirShrOp;

impl Verify for MirShrOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let lhs = op.get_operand(0);
        let res = op.get_result(0);

        let lhs_ty = lhs.get_type(ctx);
        let res_ty = res.get_type(ctx);

        if lhs_ty != res_ty {
            return verify_err!(op.loc(), "MirShrOp result type must match operand type");
        }
        Ok(())
    }
}

/// MIR shl (shift left) operation.
///
/// Logical shift left.
///
/// # Verification
///
/// - Result type must match first operand type.
#[pliron_op(
    name = "mir.shl",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface]
)]
pub struct MirShlOp;

impl Verify for MirShlOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let lhs = op.get_operand(0);
        let res = op.get_result(0);

        let lhs_ty = lhs.get_type(ctx);
        let res_ty = res.get_type(ctx);

        if lhs_ty != res_ty {
            return verify_err!(op.loc(), "MirShlOp result type must match operand type");
        }
        Ok(())
    }
}

// ============================================================================
// Bitwise Operations
// ============================================================================

/// MIR bitand (bitwise AND) operation.
///
/// # Verification
///
/// - Both operands must have the same type.
/// - Result type must match operand types.
#[pliron_op(
    name = "mir.bitand",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface]
)]
pub struct MirBitAndOp;

impl Verify for MirBitAndOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let lhs = op.get_operand(0);
        let rhs = op.get_operand(1);
        let res = op.get_result(0);

        let lhs_ty = lhs.get_type(ctx);
        let rhs_ty = rhs.get_type(ctx);
        let res_ty = res.get_type(ctx);

        if lhs_ty != rhs_ty {
            return verify_err!(op.loc(), "MirBitAndOp operands must be of the same type");
        }
        if lhs_ty != res_ty {
            return verify_err!(op.loc(), "MirBitAndOp result type must match operand types");
        }
        Ok(())
    }
}

/// MIR bitor (bitwise OR) operation.
///
/// # Verification
///
/// - Both operands must have the same type.
/// - Result type must match operand types.
#[pliron_op(
    name = "mir.bitor",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface]
)]
pub struct MirBitOrOp;

impl Verify for MirBitOrOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let lhs = op.get_operand(0);
        let rhs = op.get_operand(1);
        let res = op.get_result(0);

        let lhs_ty = lhs.get_type(ctx);
        let rhs_ty = rhs.get_type(ctx);
        let res_ty = res.get_type(ctx);

        if lhs_ty != rhs_ty {
            return verify_err!(op.loc(), "MirBitOrOp operands must be of the same type");
        }
        if lhs_ty != res_ty {
            return verify_err!(op.loc(), "MirBitOrOp result type must match operand types");
        }
        Ok(())
    }
}

/// MIR bitxor (bitwise XOR) operation.
///
/// # Verification
///
/// - Both operands must have the same type.
/// - Result type must match operand types.
#[pliron_op(
    name = "mir.bitxor",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface]
)]
pub struct MirBitXorOp;

impl Verify for MirBitXorOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let lhs = op.get_operand(0);
        let rhs = op.get_operand(1);
        let res = op.get_result(0);

        let lhs_ty = lhs.get_type(ctx);
        let rhs_ty = rhs.get_type(ctx);
        let res_ty = res.get_type(ctx);

        if lhs_ty != rhs_ty {
            return verify_err!(op.loc(), "MirBitXorOp operands must be of the same type");
        }
        if lhs_ty != res_ty {
            return verify_err!(op.loc(), "MirBitXorOp result type must match operand types");
        }
        Ok(())
    }
}

/// Register arithmetic operations into the given context.
pub fn register(ctx: &mut Context) {
    MirAddOp::register(ctx);
    MirSubOp::register(ctx);
    MirMulOp::register(ctx);
    MirDivOp::register(ctx);
    MirRemOp::register(ctx);
    MirCheckedAddOp::register(ctx);
    MirCheckedMulOp::register(ctx);
    MirCheckedSubOp::register(ctx);
    MirNegOp::register(ctx);
    MirNotOp::register(ctx);
    MirShrOp::register(ctx);
    MirShlOp::register(ctx);
    MirBitAndOp::register(ctx);
    MirBitOrOp::register(ctx);
    MirBitXorOp::register(ctx);
}
