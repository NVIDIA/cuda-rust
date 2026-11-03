/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Unit tests for the `dialect-mir` constant-folding interfaces (see
//! `src/const_fold.rs`).
//!
//! `ConstFoldInterface::check_fold` takes the operands' known constant values
//! as a parameter and returns the folded result value(s), so we can exercise
//! each op's fold rule directly: build the op, hand `check_fold` two integer
//! attributes, and check the result. `BranchOpFoldInterface::check_fold` is
//! tested the same way (constant condition -> which successor stays feasible).
//!
//! These assert the *fold rules* in isolation; `unroll_smoke` and the
//! mir-transforms unroll-pass test cover the end-to-end `sccp` path.

use std::num::NonZero;

use dialect_mir::ops::{
    MirAddOp, MirBitAndOp, MirBitOrOp, MirBitXorOp, MirCondBranchOp, MirConstantOp, MirDivOp,
    MirEqOp, MirFuncOp, MirGeOp, MirGtOp, MirLeOp, MirLtOp, MirMulOp, MirNeOp, MirRemOp, MirShlOp,
    MirShrOp, MirSubOp,
};
use pliron::attribute::AttrObj;
use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::{IntegerAttr, TypeAttr};
use pliron::builtin::op_interfaces::OperandSegmentInterface;
use pliron::builtin::types::{FunctionType, IntegerType, Signedness};
use pliron::common_traits::Verify;
use pliron::context::{Context, Ptr};
use pliron::op::{Op, op_cast};
use pliron::operation::Operation;
use pliron::opts::constants::{BranchOpFoldInterface, ConstFoldInterface};
use pliron::region::Region;
use pliron::r#type::TypedHandle;
use pliron::value::Value;

/// A fresh context with the `mir` dialect registered.
fn ctx() -> Context {
    let mut c = Context::new();
    dialect_mir::register(&mut c);
    c
}

/// `fn foo()` with one entry block; returns `(region, entry)`.
fn func_with_entry(ctx: &mut Context) -> (Ptr<Region>, Ptr<BasicBlock>) {
    let func_ty = FunctionType::get(ctx, vec![], vec![]);
    let op = Operation::new(
        ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let func = MirFuncOp::new(ctx, op, TypeAttr::new(func_ty.into()));
    let region = func.get_operation().deref(ctx).get_region(0);
    let entry = BasicBlock::new(ctx, None, vec![]);
    entry.insert_at_front(region, ctx);
    (region, entry)
}

fn int_ty(ctx: &mut Context, width: u32, sign: Signedness) -> TypedHandle<IntegerType> {
    IntegerType::get(ctx, width, sign)
}

/// An `IntegerAttr` of `ty` holding `v`, as an `AttrObj` (what `check_fold` takes).
fn iattr(ctx: &Context, ty: TypedHandle<IntegerType>, v: impl Into<i128>) -> AttrObj {
    let width = ty.deref(ctx).width() as usize;
    IntegerAttr::new(
        ty,
        pliron::utils::apint::APInt::from_i128(v.into(), NonZero::new(width).unwrap()),
    )
    .into()
}

/// Append a constant with the same type and value as an operand attribute.
fn constant_operand(ctx: &mut Context, blk: Ptr<BasicBlock>, attr: &AttrObj) -> Value {
    let attr = attr.downcast_ref::<IntegerAttr>().unwrap().clone();
    attr.verify(ctx).unwrap();
    let op = Operation::new(
        ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![attr.get_type().into()],
        vec![],
        vec![],
        0,
    );
    MirConstantOp::new(op).set_attr_value(ctx, attr);
    op.insert_at_back(blk, ctx);
    op.deref(ctx).get_result(0)
}

/// A placeholder for tests that pass their constant operands directly to the fold.
fn placeholder(ctx: &mut Context, blk: Ptr<BasicBlock>) -> Value {
    let ty = int_ty(ctx, 32, Signedness::Signless);
    let attr = iattr(ctx, ty, 0);
    constant_operand(ctx, blk, &attr)
}

/// Build a two-operand op of `$opty` (result type `$res_ty`), then fold it with
/// operand attrs `$a`,`$b`. Returns the folded result as `Option<i128>` (`None`
/// if it refused to fold). The attrs are bound first so their immutable borrow
/// of `ctx` is released before the op-building mutable borrow.
macro_rules! fold_bin {
    ($ctx:expr, $blk:expr, $opty:ty, $res_ty:expr, $a:expr, $b:expr) => {{
        let a_attr = $a;
        let b_attr = $b;
        let res_ty: pliron::r#type::TypeHandle = $res_ty.into();
        let lv = constant_operand($ctx, $blk, &a_attr);
        let rv = constant_operand($ctx, $blk, &b_attr);
        let op = Operation::new(
            $ctx,
            <$opty>::get_concrete_op_info(),
            vec![res_ty],
            vec![lv, rv],
            vec![],
            0,
        );
        op.insert_at_back($blk, $ctx);
        let op_dyn = Operation::get_op_dyn(op, $ctx);
        op_dyn.verify($ctx).unwrap();
        let fold = op_cast::<dyn ConstFoldInterface>(op_dyn.as_ref())
            .expect("op implements ConstFoldInterface");
        let out = fold.check_fold($ctx, &[Some(a_attr), Some(b_attr)]);
        assert_eq!(out.len(), 1, "a binary op has exactly one result");
        match &out[0] {
            Some(attr) => {
                let attr = attr.downcast_ref::<IntegerAttr>().unwrap();
                attr.verify($ctx).unwrap();
                let folded_ty: pliron::r#type::TypeHandle = attr.get_type().into();
                assert_eq!(folded_ty, res_ty, "fold must preserve the result type");
                Some(attr.value().to_i128())
            }
            None => None,
        }
    }};
}

/// Like [`fold_bin`], but reads the `i1` result of a comparison as a `bool`
/// (the raw bit, not a sign-extended `to_i128`: a 1-bit `1` is `-1` signed).
macro_rules! fold_cmp {
    ($ctx:expr, $blk:expr, $opty:ty, $a:expr, $b:expr) => {{
        let a_attr = $a;
        let b_attr = $b;
        let i1t = int_ty($ctx, 1, Signedness::Signless);
        let lv = placeholder($ctx, $blk);
        let rv = placeholder($ctx, $blk);
        let op = Operation::new(
            $ctx,
            <$opty>::get_concrete_op_info(),
            vec![i1t.into()],
            vec![lv, rv],
            vec![],
            0,
        );
        op.insert_at_back($blk, $ctx);
        let op_dyn = Operation::get_op_dyn(op, $ctx);
        let fold = op_cast::<dyn ConstFoldInterface>(op_dyn.as_ref())
            .expect("op implements ConstFoldInterface");
        let out = fold.check_fold($ctx, &[Some(a_attr), Some(b_attr)]);
        assert_eq!(out.len(), 1, "a comparison has exactly one result");
        out[0].as_ref().map(|attr| {
            !attr
                .downcast_ref::<IntegerAttr>()
                .unwrap()
                .value()
                .is_zero()
        })
    }};
}

#[test]
fn arithmetic_and_bitwise_fold() {
    let mut ctx = ctx();
    let (_r, b) = func_with_entry(&mut ctx);
    let i32t = int_ty(&mut ctx, 32, Signedness::Signless);
    let (a, c) = (12, 4);

    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirAddOp,
            i32t,
            iattr(&ctx, i32t, a),
            iattr(&ctx, i32t, c)
        ),
        Some(16)
    );
    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirSubOp,
            i32t,
            iattr(&ctx, i32t, a),
            iattr(&ctx, i32t, c)
        ),
        Some(8)
    );
    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirMulOp,
            i32t,
            iattr(&ctx, i32t, a),
            iattr(&ctx, i32t, c)
        ),
        Some(48)
    );
    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirBitAndOp,
            i32t,
            iattr(&ctx, i32t, a),
            iattr(&ctx, i32t, c)
        ),
        Some(4)
    );
    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirBitOrOp,
            i32t,
            iattr(&ctx, i32t, a),
            iattr(&ctx, i32t, c)
        ),
        Some(12)
    );
    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirBitXorOp,
            i32t,
            iattr(&ctx, i32t, a),
            iattr(&ctx, i32t, c)
        ),
        Some(8)
    );
}

#[test]
fn shifts_fold_and_respect_signedness() {
    let mut ctx = ctx();
    let (_r, b) = func_with_entry(&mut ctx);
    let signless = int_ty(&mut ctx, 32, Signedness::Signless);
    let unsigned = int_ty(&mut ctx, 32, Signedness::Unsigned);
    let signed = int_ty(&mut ctx, 32, Signedness::Signed);

    // 1 << 4 == 16
    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirShlOp,
            signless,
            iattr(&ctx, signless, 1),
            iattr(&ctx, signless, 4)
        ),
        Some(16)
    );

    // logical shift: 16u32 >> 2 == 4
    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirShrOp,
            unsigned,
            iattr(&ctx, unsigned, 16),
            iattr(&ctx, unsigned, 2)
        ),
        Some(4)
    );

    // arithmetic shift: (-16i32) >> 2 == -4 (sign bit copied in)
    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirShrOp,
            signed,
            iattr(&ctx, signed, -16),
            iattr(&ctx, signed, 2)
        ),
        Some(-4)
    );
}

/// Rust lets the shift amount have a different width from the shifted value
/// (`u32 << usize`), and the lowering accepts that. After `#[unroll]` turns a
/// `usize` loop counter into a constant, SCCP hands the fold exactly that pair.
#[test]
fn shifts_fold_with_a_shift_amount_of_a_different_width() {
    let mut ctx = ctx();
    let (_r, b) = func_with_entry(&mut ctx);
    let u32t = int_ty(&mut ctx, 32, Signedness::Unsigned);
    let u64t = int_ty(&mut ctx, 64, Signedness::Unsigned);
    let i32t = int_ty(&mut ctx, 32, Signedness::Signed);
    let u8t = int_ty(&mut ctx, 8, Signedness::Unsigned);

    // wider amount: 1u32 << 4usize == 16, result keeps the value's type
    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirShlOp,
            u32t,
            iattr(&ctx, u32t, 1),
            iattr(&ctx, u64t, 4)
        ),
        Some(16)
    );

    // wider amount, logical: 16u32 >> 2usize == 4
    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirShrOp,
            u32t,
            iattr(&ctx, u32t, 16),
            iattr(&ctx, u64t, 2)
        ),
        Some(4)
    );

    // wider amount, arithmetic: (-16i32) >> 2usize == -4
    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirShrOp,
            i32t,
            iattr(&ctx, i32t, -16),
            iattr(&ctx, u64t, 2)
        ),
        Some(-4)
    );

    // narrower amount: 1u64 << 40u8 == 2^40
    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirShlOp,
            u64t,
            iattr(&ctx, u64t, 1),
            iattr(&ctx, u8t, 40)
        ),
        Some(1i128 << 40)
    );

    // the range check still reads the amount at its own width: 2^32 must not
    // be truncated to 0 and folded
    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirShlOp,
            u32t,
            iattr(&ctx, u32t, 1),
            iattr(&ctx, u64t, 1i64 << 32)
        ),
        None
    );
    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirShrOp,
            u32t,
            iattr(&ctx, u32t, 1),
            iattr(&ctx, u64t, 32)
        ),
        None
    );
}

#[test]
fn shifts_preserve_128_bit_values_and_reject_invalid_counts() {
    let mut ctx = ctx();
    let (_r, b) = func_with_entry(&mut ctx);
    let u8t = int_ty(&mut ctx, 8, Signedness::Unsigned);
    let i8t = int_ty(&mut ctx, 8, Signedness::Signed);
    let u32t = int_ty(&mut ctx, 32, Signedness::Unsigned);
    let u128t = int_ty(&mut ctx, 128, Signedness::Unsigned);
    let i128t = int_ty(&mut ctx, 128, Signedness::Signed);

    // Narrow counts cover zero and width - 1. Right-shift signedness comes
    // from the value, even when the count has the opposite signedness.
    for amount in [0u32, 127] {
        assert_eq!(
            fold_bin!(
                &mut ctx,
                b,
                MirShlOp,
                u128t,
                iattr(&ctx, u128t, 1),
                iattr(&ctx, i8t, amount)
            ),
            Some(1i128 << amount)
        );
        assert_eq!(
            fold_bin!(
                &mut ctx,
                b,
                MirShrOp,
                u128t,
                iattr(&ctx, u128t, i128::MIN),
                iattr(&ctx, i8t, amount)
            ),
            Some(((1u128 << 127) >> amount) as i128)
        );
        assert_eq!(
            fold_bin!(
                &mut ctx,
                b,
                MirShrOp,
                i128t,
                iattr(&ctx, i128t, -16),
                iattr(&ctx, u8t, amount)
            ),
            Some(-16i128 >> amount)
        );
    }

    // Check the count before narrowing: bits above 64 must survive the guard.
    for amount in [32, 33, 128, 1i128 << 64, 1i128 << 96, i128::MIN, -1] {
        assert_eq!(
            fold_bin!(
                &mut ctx,
                b,
                MirShlOp,
                u32t,
                iattr(&ctx, u32t, 1),
                iattr(&ctx, u128t, amount)
            ),
            None
        );
        assert_eq!(
            fold_bin!(
                &mut ctx,
                b,
                MirShrOp,
                u32t,
                iattr(&ctx, u32t, -1),
                iattr(&ctx, u128t, amount)
            ),
            None
        );
    }

    // Negative narrow counts must not become valid shifts of a wider value.
    for amount in [-1, -128] {
        assert_eq!(
            fold_bin!(
                &mut ctx,
                b,
                MirShlOp,
                u128t,
                iattr(&ctx, u128t, 1),
                iattr(&ctx, i8t, amount)
            ),
            None
        );
        assert_eq!(
            fold_bin!(
                &mut ctx,
                b,
                MirShrOp,
                i128t,
                iattr(&ctx, i128t, -16),
                iattr(&ctx, i8t, amount)
            ),
            None
        );
    }
}

#[test]
fn div_rem_fold_and_refuse_div_by_zero() {
    let mut ctx = ctx();
    let (_r, b) = func_with_entry(&mut ctx);
    let unsigned = int_ty(&mut ctx, 32, Signedness::Unsigned);
    let signed = int_ty(&mut ctx, 32, Signedness::Signed);

    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirDivOp,
            unsigned,
            iattr(&ctx, unsigned, 17),
            iattr(&ctx, unsigned, 5)
        ),
        Some(3)
    );
    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirRemOp,
            unsigned,
            iattr(&ctx, unsigned, 17),
            iattr(&ctx, unsigned, 5)
        ),
        Some(2)
    );

    // signed division truncates toward zero: -17 / 5 == -3, -17 % 5 == -2
    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirDivOp,
            signed,
            iattr(&ctx, signed, -17),
            iattr(&ctx, signed, 5)
        ),
        Some(-3)
    );
    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirRemOp,
            signed,
            iattr(&ctx, signed, -17),
            iattr(&ctx, signed, 5)
        ),
        Some(-2)
    );

    // division by zero is a Rust panic, never folded.
    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirDivOp,
            unsigned,
            iattr(&ctx, unsigned, 5),
            iattr(&ctx, unsigned, 0)
        ),
        None
    );
    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirRemOp,
            unsigned,
            iattr(&ctx, unsigned, 5),
            iattr(&ctx, unsigned, 0)
        ),
        None
    );
}

#[test]
fn comparisons_fold_to_i1_and_respect_signedness() {
    let mut ctx = ctx();
    let (_r, b) = func_with_entry(&mut ctx);
    let signed = int_ty(&mut ctx, 32, Signedness::Signed);
    let unsigned = int_ty(&mut ctx, 32, Signedness::Unsigned);

    // equality / inequality / ordering
    assert_eq!(
        fold_cmp!(
            &mut ctx,
            b,
            MirEqOp,
            iattr(&ctx, signed, 5),
            iattr(&ctx, signed, 5)
        ),
        Some(true)
    );
    assert_eq!(
        fold_cmp!(
            &mut ctx,
            b,
            MirNeOp,
            iattr(&ctx, signed, 5),
            iattr(&ctx, signed, 6)
        ),
        Some(true)
    );
    assert_eq!(
        fold_cmp!(
            &mut ctx,
            b,
            MirLeOp,
            iattr(&ctx, signed, 5),
            iattr(&ctx, signed, 5)
        ),
        Some(true)
    );
    assert_eq!(
        fold_cmp!(
            &mut ctx,
            b,
            MirGeOp,
            iattr(&ctx, signed, 5),
            iattr(&ctx, signed, 6)
        ),
        Some(false)
    );
    assert_eq!(
        fold_cmp!(
            &mut ctx,
            b,
            MirGtOp,
            iattr(&ctx, signed, 7),
            iattr(&ctx, signed, 6)
        ),
        Some(true)
    );

    // `-1 < 1` is true signed, but false unsigned (where -1 is the max value).
    assert_eq!(
        fold_cmp!(
            &mut ctx,
            b,
            MirLtOp,
            iattr(&ctx, signed, -1),
            iattr(&ctx, signed, 1)
        ),
        Some(true)
    );
    assert_eq!(
        fold_cmp!(
            &mut ctx,
            b,
            MirLtOp,
            iattr(&ctx, unsigned, -1),
            iattr(&ctx, unsigned, 1)
        ),
        Some(false)
    );
}

#[test]
fn cond_br_folds_to_the_taken_successor() {
    let mut ctx = ctx();
    let (region, entry) = func_with_entry(&mut ctx);
    let i1t = int_ty(&mut ctx, 1, Signedness::Signless);

    let t = BasicBlock::new(&mut ctx, None, vec![]);
    let f = BasicBlock::new(&mut ctx, None, vec![]);
    t.insert_at_back(region, &ctx);
    f.insert_at_back(region, &ctx);

    let cond = placeholder(&mut ctx, entry);
    let (flat, segs) = MirCondBranchOp::compute_segment_sizes(vec![vec![cond], vec![], vec![]]);
    let op = Operation::new(
        &mut ctx,
        MirCondBranchOp::get_concrete_op_info(),
        vec![],
        flat,
        vec![t, f],
        0,
    );
    Operation::get_op::<MirCondBranchOp>(op, &ctx)
        .unwrap()
        .set_operand_segment_sizes(&ctx, segs);
    op.insert_at_back(entry, &ctx);

    let op_dyn = Operation::get_op_dyn(op, &ctx);
    let branch = op_cast::<dyn BranchOpFoldInterface>(op_dyn.as_ref())
        .expect("cond_br implements BranchOpFoldInterface");

    // condition true -> only the true successor (index 0) stays feasible.
    let true_succs = branch.check_fold(&ctx, &[Some(iattr(&ctx, i1t, 1))]);
    assert_eq!(true_succs, vec![t]);

    // condition false -> only the false successor (index 1).
    let false_succs = branch.check_fold(&ctx, &[Some(iattr(&ctx, i1t, 0))]);
    assert_eq!(false_succs, vec![f]);
}
