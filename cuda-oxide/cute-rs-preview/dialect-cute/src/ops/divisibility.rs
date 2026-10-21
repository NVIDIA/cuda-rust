/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SSA identity operations preserving explicit integer divisibility promises.

use pliron::{
    builtin::{
        op_interfaces::{NOpdsInterface, NResultsInterface, OneOpdInterface, OneResultInterface},
        types::IntegerType,
    },
    common_traits::Verify,
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
    result::Error,
    r#type::Typed,
    value::Value,
    verify_err,
};
use pliron_derive::pliron_op;

use crate::attributes::CuteDivisibilityAttr;

/// Preserve the promise `value % divisor == 0` in SSA form.
///
/// ```text
/// Rust                       cute IR                       lowering
/// ─────                      ───────                       ────────
/// y = assume_div::<16>(x) -> y = cute.assume_div x {16} -> y uses x
///                                  │
///                                  └─ direct SSA consumers read the fact
/// ```
///
/// The operation does not change the runtime value. Its result exists so the
/// fact follows that exact value through ordinary SSA uses; a detached
/// zero-result annotation would require guessing which later value it meant.
#[pliron_op(
    name = "cute.assume_div",
    format,
    interfaces = [
        NOpdsInterface<1>,
        NResultsInterface<1>,
        OneOpdInterface,
        OneResultInterface
    ],
    attributes = (divisor: CuteDivisibilityAttr)
)]
pub struct CuteAssumeDivOp;

impl CuteAssumeDivOp {
    /// Build a detached identity operation carrying the caller's divisibility promise.
    pub fn new(ctx: &mut Context, value: Value, divisor: u64) -> Self {
        let result_ty = value.get_type(ctx);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_ty],
            vec![value],
            vec![],
            0,
        );
        let assume = Self { op };
        assume.set_attr_divisor(ctx, CuteDivisibilityAttr(divisor));
        assume
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Return the promised positive integer divisor, if the attribute is present.
    pub fn divisor(&self, ctx: &Context) -> Option<u64> {
        self.get_attr_divisor(ctx).map(|attr| attr.0)
    }
}

impl Verify for CuteAssumeDivOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        if op.get_num_operands() != 1 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "cute.assume_div needs exactly 1 operand and 1 result, got {} and {}",
                op.get_num_operands(),
                op.get_num_results()
            );
        }
        let Some(divisor) = self.divisor(ctx) else {
            return verify_err!(op.loc(), "cute.assume_div must have a divisor attribute");
        };
        if divisor == 0 {
            return verify_err!(
                op.loc(),
                "cute.assume_div divisor must be greater than zero"
            );
        }

        let input_ty = op.get_operand(0).get_type(ctx);
        if input_ty.deref(ctx).downcast_ref::<IntegerType>().is_none() {
            return verify_err!(op.loc(), "cute.assume_div input must be an integer");
        }
        if op.get_result(0).get_type(ctx) != input_ty {
            return verify_err!(
                op.loc(),
                "cute.assume_div result must have the same type as its input"
            );
        }
        Ok(())
    }
}
