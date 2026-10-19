/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Constant folding for the pure `dialect-mir` integer ops.
//!
//! pliron ships a generic constant-propagation pass (`sccp`) plus a CFG
//! simplifier (`simplify_cfg`). Both are dialect-agnostic: they only act on ops
//! that opt in via an interface. This module implements those interfaces for the
//! `dialect-mir` integer ops so the folding happens in **our** middle-end, before
//! we export textual LLVM IR. That makes the fold independent of `opt -O2` and
//! of whatever the NVVM backend optimises. By the time we export, the constants
//! are already in the IR we hand off.
//!
//! Two interfaces:
//!
//! - [`ConstFoldInterface`] on every pure value-producing op (arithmetic,
//!   bitwise, shift, comparison) and on `mir.constant` itself (the lattice seed).
//!   `check_fold` computes the result on [`APInt`] when all operands are known
//!   constants; `fold_in_place` materialises that constant. We delegate
//!   `fold_in_place` to the framework's [`fold_with_materialization`], which
//!   builds a `builtin.constant`; `mir-lower` lowers that to `llvm.constant`.
//!   (`sccp` materialises constant block arguments the same way, so a
//!   `builtin.constant` can appear from either path: both are lowered.)
//! - [`BranchOpFoldInterface`] on `mir.cond_br`, so `simplify_cfg` can collapse a
//!   conditional branch whose condition folded to a constant into an
//!   unconditional `mir.goto` (the `match`-on-a-constant-index collapse).
//!
//! Rust-semantics notes (we are folding Rust code, not LLVM IR):
//! - `mir.add`/`sub`/`mul` are the **wrapping** ops (Rust's overflow checks are
//!   separate `mir.checked_*` ops), so we fold with wrapping [`APInt`] arithmetic
//!   and never produce poison.
//! - Division/remainder by zero is a Rust panic, so we **do not** fold it (return
//!   "not constant" and let the runtime path stand). Signed `INT_MIN / -1` is
//!   likewise left unfolded.
//! - Signedness is carried on the operand's `IntegerType` (Rust `ui32`/`si32`),
//!   so `mir.shr`/`div`/`rem` and the ordering comparisons pick the
//!   signed/unsigned operation from the operand type.
//!
//! For example: `5u32 / 2` folds to `2`, but `5u32 / 0` stays a runtime
//! `mir.div` (Rust would panic), and a shift amount `>=` the bit width stays
//! unfolded too.
//!
//! [`fold_with_materialization`]: pliron::opts::constants::ConstFoldInterface::fold_with_materialization

use std::num::NonZero;

use pliron::attribute::AttrObj;
use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::IntegerAttr;
use pliron::builtin::op_interfaces::BranchOpInterface;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::derive::op_interface_impl;
use pliron::irbuild::{IRStatus, rewriter::Rewriter};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::opts::constants::{BranchOpFoldInterface, ConstFoldInterface};
use pliron::r#type::{Typed, TypedHandle};
use pliron::utils::apint::APInt;

use crate::ops::{
    MirAddOp, MirBitAndOp, MirBitOrOp, MirBitXorOp, MirCondBranchOp, MirConstantOp, MirDivOp,
    MirEqOp, MirGeOp, MirGotoOp, MirGtOp, MirLeOp, MirLtOp, MirMulOp, MirNeOp, MirRemOp, MirShlOp,
    MirShrOp, MirSubOp,
};

// ---------------------------------------------------------------------------
// Helpers (ported from pliron-llvm's interface_impls.rs, adapted for mir types).
// ---------------------------------------------------------------------------

/// If `operand_attrs` is `[Some(IntegerAttr), Some(IntegerAttr)]`, return the two
/// integer attributes; otherwise `None` (an operand is non-constant).
fn int_bin_operands(operand_attrs: &[Option<AttrObj>]) -> Option<(IntegerAttr, IntegerAttr)> {
    let [Some(lhs), Some(rhs)] = operand_attrs else {
        return None;
    };
    let lhs_int = lhs
        .downcast_ref::<IntegerAttr>()
        .expect("invalid operand type: typecheck before optimizing");
    let rhs_int = rhs
        .downcast_ref::<IntegerAttr>()
        .expect("invalid operand type: typecheck before optimizing");
    Some((lhs_int.clone(), rhs_int.clone()))
}

/// Fold a binary integer op whose result has the same type as its left operand
/// (true for every `dialect-mir` binary op here), computing the value with
/// `combine`. Returns a one-element result vector, or `[None]` if not foldable.
fn fold_bin(
    operand_attrs: &[Option<AttrObj>],
    combine: impl Fn(&APInt, &APInt) -> APInt,
) -> Vec<Option<AttrObj>> {
    let Some((lhs, rhs)) = int_bin_operands(operand_attrs) else {
        return vec![None];
    };
    let res = IntegerAttr::new(lhs.get_type(), combine(&lhs.value(), &rhs.value()));
    vec![Some(Box::new(res) as AttrObj)]
}

/// `true` if `attr`'s integer type is signed (so signed div/rem/shr/compare).
fn is_signed(ctx: &Context, attr: &IntegerAttr) -> bool {
    attr.get_type().deref(ctx).signedness() == Signedness::Signed
}

/// `true` if signed-dividing/remaindering `lhs` by `rhs` is UB (Rust panic /
/// not representable), so it must not be folded: division by zero, or the signed
/// overflow `INT_MIN / -1`.
fn is_signed_div_ub(lhs: &APInt, rhs: &APInt) -> bool {
    let bw = NonZero::new(rhs.bw()).expect("operand has zero bitwidth");
    // `-1` is the all-ones pattern, i.e. the unsigned max.
    rhs.is_zero() || (*lhs == APInt::imin(bw) && *rhs == APInt::umax(bw))
}

/// Build a boolean result (`i1`) attribute matching `i1_ty`.
fn bool_attr(value: bool, i1_ty: TypedHandle<IntegerType>) -> AttrObj {
    let bit = APInt::from_u64(value as u64, NonZero::new(1).expect("nonzero width"));
    Box::new(IntegerAttr::new(i1_ty, bit)) as AttrObj
}

/// Fold an ordering/equality comparison whose result is `i1`. `signed_cmp` is
/// used when the operands are signed, `unsigned_cmp` otherwise. `op` is the
/// comparison op (its result type supplies the `i1` to build).
fn fold_compare(
    ctx: &Context,
    op: Ptr<Operation>,
    operand_attrs: &[Option<AttrObj>],
    signed_cmp: fn(i128, i128) -> bool,
    unsigned_cmp: fn(u128, u128) -> bool,
) -> Vec<Option<AttrObj>> {
    let Some((lhs, rhs)) = int_bin_operands(operand_attrs) else {
        return vec![None];
    };
    let res = if is_signed(ctx, &lhs) {
        signed_cmp(lhs.value().to_i128(), rhs.value().to_i128())
    } else {
        unsigned_cmp(lhs.value().to_u128(), rhs.value().to_u128())
    };
    let res_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let Ok(i1) = TypedHandle::<IntegerType>::from_handle(res_ty, ctx) else {
        return vec![None];
    };
    vec![Some(bool_attr(res, i1))]
}

// ---------------------------------------------------------------------------
// Arithmetic / bitwise / left-shift: result depends only on the two constants.
// `mir.add`/`sub`/`mul` are wrapping (Rust release semantics).
// ---------------------------------------------------------------------------

/// Implement `ConstFoldInterface` for a binary op whose fold is `fold_bin` with
/// the given `APInt` combiner; `fold_in_place` always materialises the constant.
macro_rules! const_fold_bin {
    ($op:ty, $combine:expr) => {
        #[op_interface_impl]
        impl ConstFoldInterface for $op {
            fn check_fold(&self, _ctx: &Context, ops: &[Option<AttrObj>]) -> Vec<Option<AttrObj>> {
                fold_bin(ops, $combine)
            }
            fn fold_in_place(
                &self,
                ctx: &mut Context,
                ops: &[Option<AttrObj>],
                rw: &mut dyn Rewriter,
            ) -> IRStatus {
                self.fold_with_materialization(ctx, ops, rw)
            }
        }
    };
}

const_fold_bin!(MirAddOp, APInt::add);
const_fold_bin!(MirSubOp, APInt::sub);
const_fold_bin!(MirMulOp, APInt::mul);
const_fold_bin!(MirBitAndOp, APInt::and);
const_fold_bin!(MirBitOrOp, APInt::or);
const_fold_bin!(MirBitXorOp, APInt::xor);

// ---------------------------------------------------------------------------
// Shifts: left-shift is sign-agnostic; right-shift picks arithmetic vs logical
// from the shifted operand's signedness. A shift amount >= the bit width is a
// Rust panic in debug (and unspecified otherwise), so we do not fold it.
// ---------------------------------------------------------------------------

/// The shift amount re-expressed at the shifted value's width, or `None` when
/// it is `>= width` (a Rust shift overflow, not folded).
///
/// Rust allows the amount to have any integer type (`u32 << usize`), and the
/// lowering widens or narrows it to the value's width. `APInt`'s shifts assert
/// equal widths, so the fold has to do the same. The range check reads the
/// amount at its own width first, so narrowing never drops a set bit.
fn shift_amount(value: &APInt, amount: &APInt) -> Option<APInt> {
    let width = NonZero::new(value.bw())?;
    let amt = amount.to_u128();
    if amt >= value.bw() as u128 {
        return None;
    }
    Some(APInt::from_u128(amt, width))
}

#[op_interface_impl]
impl ConstFoldInterface for MirShlOp {
    fn check_fold(&self, _ctx: &Context, ops: &[Option<AttrObj>]) -> Vec<Option<AttrObj>> {
        let Some((lhs, rhs)) = int_bin_operands(ops) else {
            return vec![None];
        };
        let Some(amount) = shift_amount(&lhs.value(), &rhs.value()) else {
            return vec![None];
        };
        let res = IntegerAttr::new(lhs.get_type(), lhs.value().shl(&amount));
        vec![Some(Box::new(res) as AttrObj)]
    }
    fn fold_in_place(
        &self,
        ctx: &mut Context,
        ops: &[Option<AttrObj>],
        rw: &mut dyn Rewriter,
    ) -> IRStatus {
        self.fold_with_materialization(ctx, ops, rw)
    }
}

#[op_interface_impl]
impl ConstFoldInterface for MirShrOp {
    fn check_fold(&self, ctx: &Context, ops: &[Option<AttrObj>]) -> Vec<Option<AttrObj>> {
        let Some((lhs, rhs)) = int_bin_operands(ops) else {
            return vec![None];
        };
        let Some(amount) = shift_amount(&lhs.value(), &rhs.value()) else {
            return vec![None];
        };
        let res = if is_signed(ctx, &lhs) {
            lhs.value().ashr(&amount)
        } else {
            lhs.value().lshr(&amount)
        };
        vec![Some(
            Box::new(IntegerAttr::new(lhs.get_type(), res)) as AttrObj
        )]
    }
    fn fold_in_place(
        &self,
        ctx: &mut Context,
        ops: &[Option<AttrObj>],
        rw: &mut dyn Rewriter,
    ) -> IRStatus {
        self.fold_with_materialization(ctx, ops, rw)
    }
}

// ---------------------------------------------------------------------------
// Division / remainder: signedness from the operand type; never fold a Rust
// panic (divide/remainder by zero, or signed INT_MIN / -1).
// ---------------------------------------------------------------------------

#[op_interface_impl]
impl ConstFoldInterface for MirDivOp {
    fn check_fold(&self, ctx: &Context, ops: &[Option<AttrObj>]) -> Vec<Option<AttrObj>> {
        let Some((lhs, rhs)) = int_bin_operands(ops) else {
            return vec![None];
        };
        let (l, r) = (lhs.value(), rhs.value());
        let res = if is_signed(ctx, &lhs) {
            if is_signed_div_ub(&l, &r) {
                return vec![None];
            }
            l.sdiv(&r)
        } else {
            if r.is_zero() {
                return vec![None];
            }
            l.udiv(&r)
        };
        vec![Some(
            Box::new(IntegerAttr::new(lhs.get_type(), res)) as AttrObj
        )]
    }
    fn fold_in_place(
        &self,
        ctx: &mut Context,
        ops: &[Option<AttrObj>],
        rw: &mut dyn Rewriter,
    ) -> IRStatus {
        self.fold_with_materialization(ctx, ops, rw)
    }
}

#[op_interface_impl]
impl ConstFoldInterface for MirRemOp {
    fn check_fold(&self, ctx: &Context, ops: &[Option<AttrObj>]) -> Vec<Option<AttrObj>> {
        let Some((lhs, rhs)) = int_bin_operands(ops) else {
            return vec![None];
        };
        let (l, r) = (lhs.value(), rhs.value());
        let res = if is_signed(ctx, &lhs) {
            if is_signed_div_ub(&l, &r) {
                return vec![None];
            }
            l.srem(&r)
        } else {
            if r.is_zero() {
                return vec![None];
            }
            l.urem(&r)
        };
        vec![Some(
            Box::new(IntegerAttr::new(lhs.get_type(), res)) as AttrObj
        )]
    }
    fn fold_in_place(
        &self,
        ctx: &mut Context,
        ops: &[Option<AttrObj>],
        rw: &mut dyn Rewriter,
    ) -> IRStatus {
        self.fold_with_materialization(ctx, ops, rw)
    }
}

// ---------------------------------------------------------------------------
// Comparisons: result is `i1`. Equality is sign-agnostic; ordering picks the
// signed/unsigned comparison from the operand type.
// ---------------------------------------------------------------------------

/// Implement `ConstFoldInterface` for a comparison op, using `$s` on signed
/// operands and `$u` on unsigned ones (the same expression for eq/ne).
macro_rules! const_fold_cmp {
    ($op:ty, $s:expr, $u:expr) => {
        #[op_interface_impl]
        impl ConstFoldInterface for $op {
            fn check_fold(&self, ctx: &Context, ops: &[Option<AttrObj>]) -> Vec<Option<AttrObj>> {
                fold_compare(ctx, self.get_operation(), ops, $s, $u)
            }
            fn fold_in_place(
                &self,
                ctx: &mut Context,
                ops: &[Option<AttrObj>],
                rw: &mut dyn Rewriter,
            ) -> IRStatus {
                self.fold_with_materialization(ctx, ops, rw)
            }
        }
    };
}

const_fold_cmp!(MirLtOp, |a, b| a < b, |a, b| a < b);
const_fold_cmp!(MirLeOp, |a, b| a <= b, |a, b| a <= b);
const_fold_cmp!(MirGtOp, |a, b| a > b, |a, b| a > b);
const_fold_cmp!(MirGeOp, |a, b| a >= b, |a, b| a >= b);
const_fold_cmp!(MirEqOp, |a, b| a == b, |a, b| a == b);
const_fold_cmp!(MirNeOp, |a, b| a != b, |a, b| a != b);

// ---------------------------------------------------------------------------
// `mir.constant`: the lattice seed. It reports its own value as the constant and
// has nothing to rewrite.
// ---------------------------------------------------------------------------

#[op_interface_impl]
impl ConstFoldInterface for MirConstantOp {
    fn check_fold(&self, ctx: &Context, _ops: &[Option<AttrObj>]) -> Vec<Option<AttrObj>> {
        match self.get_attr_value(ctx) {
            Some(attr) => vec![Some(Box::new(attr.clone()) as AttrObj)],
            None => vec![None],
        }
    }
    fn fold_in_place(
        &self,
        _ctx: &mut Context,
        _ops: &[Option<AttrObj>],
        _rw: &mut dyn Rewriter,
    ) -> IRStatus {
        IRStatus::Unchanged
    }
}

// ---------------------------------------------------------------------------
// `mir.cond_br`: branch folding. When the condition is a known constant, only
// one successor is feasible; `simplify_cfg` then rewrites the branch into an
// unconditional `mir.goto` to that successor (the match/if collapse).
// Successor 0 is the true edge, successor 1 the false edge.
// ---------------------------------------------------------------------------

impl MirCondBranchOp {
    /// The successor indices still reachable given `operands` (the condition
    /// attribute, if constant). All successors when the condition is unknown.
    fn possible_successor_indices(
        &self,
        ctx: &Context,
        operands: &[Option<AttrObj>],
    ) -> Vec<usize> {
        let Some(cond_attr) = operands.first().and_then(|o| o.as_ref()) else {
            let n = self.get_operation().deref(ctx).successors().count();
            return (0..n).collect();
        };
        let cond_int = cond_attr
            .downcast_ref::<IntegerAttr>()
            .expect("cond_br condition must be an IntegerAttr");
        // Successor 0 = true edge, successor 1 = false edge.
        let taken = if cond_int.value().is_zero() { 1 } else { 0 };
        vec![taken]
    }
}

#[op_interface_impl]
impl BranchOpFoldInterface for MirCondBranchOp {
    fn check_fold(&self, ctx: &Context, operands: &[Option<AttrObj>]) -> Vec<Ptr<BasicBlock>> {
        let successors: Vec<Ptr<BasicBlock>> =
            self.get_operation().deref(ctx).successors().collect();
        self.possible_successor_indices(ctx, operands)
            .iter()
            .map(|i| successors[*i])
            .collect()
    }

    fn fold_in_place(
        &self,
        ctx: &mut Context,
        ops: &[Option<AttrObj>],
        rewriter: &mut dyn Rewriter,
    ) -> IRStatus {
        let indices = self.possible_successor_indices(ctx, ops);
        if indices.len() != 1 {
            return IRStatus::Unchanged;
        }
        let taken = indices[0];
        let successors: Vec<Ptr<BasicBlock>> =
            self.get_operation().deref(ctx).successors().collect();
        let target = successors[taken];
        // The taken edge's block-argument operands carry over to the goto.
        let args = self.successor_operands(ctx, taken);
        let new_op = Operation::new(
            ctx,
            MirGotoOp::get_concrete_op_info(),
            vec![],
            args,
            vec![target],
            0,
        );
        let old_op = self.get_operation();
        rewriter.insert_operation(ctx, new_op);
        rewriter.replace_operation(ctx, old_op, new_op);
        IRStatus::Changed
    }
}
