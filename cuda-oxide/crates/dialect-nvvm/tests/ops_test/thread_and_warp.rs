/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_mir::types::{MirPointerKind, MirPtrType};
use dialect_nvvm::ops::{
    ActiveMaskOp, BarWarpSyncOp, Barrier0Op, ClcQueryIsCanceledOp, Dp2aS32Op, Dp2aU32Op, Dp4aS32Op,
    Dp4aU32Op, ElectSyncOp, FmaBf16x2Op, MatchAllSyncI32Op, MatchAllSyncI64Op, MatchAnySyncI32Op,
    MatchAnySyncI64Op, ReadPtxSregDynamicSmemSizeOp, ReadPtxSregGridIdOp, ReadPtxSregLaneIdOp,
    ReadPtxSregLanemaskEqOp, ReadPtxSregLanemaskGeOp, ReadPtxSregLanemaskGtOp,
    ReadPtxSregLanemaskLeOp, ReadPtxSregLanemaskLtOp, ReadPtxSregNsmIdOp, ReadPtxSregNwarpIdOp,
    ReadPtxSregSmIdOp, ReadPtxSregTidXOp, ReadPtxSregTotalSmemSizeOp, ReadPtxSregWarpIdOp,
    ReduxSyncAddOp, ReduxSyncAndOp, ReduxSyncFmaxAbsNanOp, ReduxSyncFmaxAbsOp, ReduxSyncFmaxNanOp,
    ReduxSyncFmaxOp, ReduxSyncFminAbsNanOp, ReduxSyncFminAbsOp, ReduxSyncFminNanOp,
    ReduxSyncFminOp, ReduxSyncMaxOp, ReduxSyncMinOp, ReduxSyncOrOp, ReduxSyncUmaxOp,
    ReduxSyncUminOp, ReduxSyncXorOp, ShflSyncBflyI64Op, ShflSyncDownI64Op, ShflSyncIdxI64Op,
    ShflSyncUpI64Op, ThreadfenceBlockOp, ThreadfenceOp, ThreadfenceSystemOp, VoteSyncAllOp,
    VoteSyncAnyOp, VoteSyncBallotOp, VoteSyncUniOp,
};

use pliron::{
    basic_block::BasicBlock,
    builtin::types::{FP32Type, IntegerType, Signedness},
    common_traits::Verify,
    context::Context,
    op::{Op, verify_op},
    operation::Operation,
};

#[test]
fn test_thread_register_ops_verify_i32_results() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);

    let tid_x = Operation::new(
        &mut ctx,
        ReadPtxSregTidXOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );
    assert!(ReadPtxSregTidXOp::new(tid_x).verify(&ctx).is_ok());

    let lane_id = Operation::new(
        &mut ctx,
        ReadPtxSregLaneIdOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );
    assert!(ReadPtxSregLaneIdOp::new(lane_id).verify(&ctx).is_ok());
}

#[test]
fn test_thread_register_ops_reject_non_i32_results() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let op = Operation::new(
        &mut ctx,
        ReadPtxSregTidXOp::get_concrete_op_info(),
        vec![i64_ty.into()],
        vec![],
        vec![],
        0,
    );

    assert!(ReadPtxSregTidXOp::new(op).verify(&ctx).is_err());
}

#[test]
fn test_lanemask_ops_verify_i32_results() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);

    // Each lane-position mask is a zero-operand, single-i32-result sreg read.
    let lt = Operation::new(
        &mut ctx,
        ReadPtxSregLanemaskLtOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );
    assert!(ReadPtxSregLanemaskLtOp::new(lt).verify(&ctx).is_ok());

    let le = Operation::new(
        &mut ctx,
        ReadPtxSregLanemaskLeOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );
    assert!(ReadPtxSregLanemaskLeOp::new(le).verify(&ctx).is_ok());

    let eq = Operation::new(
        &mut ctx,
        ReadPtxSregLanemaskEqOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );
    assert!(ReadPtxSregLanemaskEqOp::new(eq).verify(&ctx).is_ok());

    let ge = Operation::new(
        &mut ctx,
        ReadPtxSregLanemaskGeOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );
    assert!(ReadPtxSregLanemaskGeOp::new(ge).verify(&ctx).is_ok());

    let gt = Operation::new(
        &mut ctx,
        ReadPtxSregLanemaskGtOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );
    assert!(ReadPtxSregLanemaskGtOp::new(gt).verify(&ctx).is_ok());
}

#[test]
fn test_lanemask_op_rejects_non_i32_result() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    // A 64-bit result must fail the shared lane-position mask verifier.
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let op = Operation::new(
        &mut ctx,
        ReadPtxSregLanemaskLtOp::get_concrete_op_info(),
        vec![i64_ty.into()],
        vec![],
        vec![],
        0,
    );
    assert!(ReadPtxSregLanemaskLtOp::new(op).verify(&ctx).is_err());
}

#[test]
fn test_generated_vote_sync_family_requires_exact_mask_predicate_and_result_types() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![i32_ty.into(), i1_ty.into(), i64_ty.into()],
    );
    let mask = block.deref(&ctx).get_argument(0);
    let predicate = block.deref(&ctx).get_argument(1);
    let wide_mask = block.deref(&ctx).get_argument(2);

    macro_rules! check_vote {
        ($op:ty, $result_ty:expr, $wrong_result_ty:expr) => {{
            let valid = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![$result_ty.into()],
                vec![mask, predicate],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(valid), &ctx).is_ok());

            for operands in [vec![], vec![mask], vec![mask, predicate, predicate]] {
                let wrong_arity = Operation::new(
                    &mut ctx,
                    <$op>::get_concrete_op_info(),
                    vec![$result_ty.into()],
                    operands,
                    vec![],
                    0,
                );
                assert!(verify_op(&<$op>::new(wrong_arity), &ctx).is_err());
            }

            for results in [vec![], vec![$result_ty.into(), $result_ty.into()]] {
                let wrong_arity = Operation::new(
                    &mut ctx,
                    <$op>::get_concrete_op_info(),
                    results,
                    vec![mask, predicate],
                    vec![],
                    0,
                );
                assert!(verify_op(&<$op>::new(wrong_arity), &ctx).is_err());
            }

            let wrong_mask = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![$result_ty.into()],
                vec![wide_mask, predicate],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(wrong_mask), &ctx).is_err());

            let wrong_predicate = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![$result_ty.into()],
                vec![mask, mask],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(wrong_predicate), &ctx).is_err());

            let wrong_result = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![$wrong_result_ty.into()],
                vec![mask, predicate],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(wrong_result), &ctx).is_err());
        }};
    }

    check_vote!(VoteSyncAllOp, i1_ty, u32_ty);
    check_vote!(VoteSyncAnyOp, i1_ty, u32_ty);
    check_vote!(VoteSyncBallotOp, u32_ty, i1_ty);
    check_vote!(VoteSyncUniOp, i1_ty, u32_ty);
}

#[test]
fn test_generated_active_mask_requires_no_operands_and_i32_result() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let unexpected_operand = block.deref(&ctx).get_argument(0);

    let valid = Operation::new(
        &mut ctx,
        ActiveMaskOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );
    assert!(verify_op(&ActiveMaskOp::new(valid), &ctx).is_ok());

    let wrong_operand_count = Operation::new(
        &mut ctx,
        ActiveMaskOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![unexpected_operand],
        vec![],
        0,
    );
    assert!(verify_op(&ActiveMaskOp::new(wrong_operand_count), &ctx).is_err());

    for results in [vec![], vec![i32_ty.into(), i32_ty.into()]] {
        let wrong_result_count = Operation::new(
            &mut ctx,
            ActiveMaskOp::get_concrete_op_info(),
            results,
            vec![],
            vec![],
            0,
        );
        assert!(verify_op(&ActiveMaskOp::new(wrong_result_count), &ctx).is_err());
    }

    let wrong_result_width = Operation::new(
        &mut ctx,
        ActiveMaskOp::get_concrete_op_info(),
        vec![i64_ty.into()],
        vec![],
        vec![],
        0,
    );
    assert!(verify_op(&ActiveMaskOp::new(wrong_result_width), &ctx).is_err());
}

#[test]
fn test_generated_match_family_requires_exact_mask_value_and_result_widths() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![i32_ty.into(), i32_ty.into(), i64_ty.into()],
    );
    let mask = block.deref(&ctx).get_argument(0);
    let value32 = block.deref(&ctx).get_argument(1);
    let value64 = block.deref(&ctx).get_argument(2);

    macro_rules! check_match {
        ($op:ty, $value:expr, $wrong_value:expr) => {{
            let valid = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![i32_ty.into()],
                vec![mask, $value],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(valid), &ctx).is_ok());

            for operands in [vec![], vec![mask], vec![mask, $value, $value]] {
                let wrong_operand_count = Operation::new(
                    &mut ctx,
                    <$op>::get_concrete_op_info(),
                    vec![i32_ty.into()],
                    operands,
                    vec![],
                    0,
                );
                assert!(verify_op(&<$op>::new(wrong_operand_count), &ctx).is_err());
            }

            for results in [vec![], vec![i32_ty.into(), i32_ty.into()]] {
                let wrong_result_count = Operation::new(
                    &mut ctx,
                    <$op>::get_concrete_op_info(),
                    results,
                    vec![mask, $value],
                    vec![],
                    0,
                );
                assert!(verify_op(&<$op>::new(wrong_result_count), &ctx).is_err());
            }

            let wrong_mask_width = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![i32_ty.into()],
                vec![value64, $value],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(wrong_mask_width), &ctx).is_err());

            let wrong_value_width = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![i32_ty.into()],
                vec![mask, $wrong_value],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(wrong_value_width), &ctx).is_err());

            let wrong_result_width = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![i64_ty.into()],
                vec![mask, $value],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(wrong_result_width), &ctx).is_err());
        }};
    }

    check_match!(MatchAnySyncI32Op, value32, value64);
    check_match!(MatchAllSyncI32Op, value32, value64);
    check_match!(MatchAnySyncI64Op, value64, value32);
    check_match!(MatchAllSyncI64Op, value64, value32);
}

#[test]
fn test_special_register_ops_verify_authoritative_widths() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);

    macro_rules! check_width {
        ($op:ty, $good:expr, $bad:expr) => {{
            let good = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![$good.into()],
                vec![],
                vec![],
                0,
            );
            assert!(
                verify_op(&<$op>::new(good), &ctx).is_ok(),
                "{} must accept its PTX register width",
                stringify!($op)
            );

            let bad = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![$bad.into()],
                vec![],
                vec![],
                0,
            );
            assert!(
                verify_op(&<$op>::new(bad), &ctx).is_err(),
                "{} must reject the other integer width",
                stringify!($op)
            );
        }};
    }

    check_width!(ReadPtxSregWarpIdOp, i32_ty, i64_ty);
    check_width!(ReadPtxSregNwarpIdOp, i32_ty, i64_ty);
    check_width!(ReadPtxSregSmIdOp, i32_ty, i64_ty);
    check_width!(ReadPtxSregNsmIdOp, i32_ty, i64_ty);
    check_width!(ReadPtxSregDynamicSmemSizeOp, i32_ty, i64_ty);
    check_width!(ReadPtxSregTotalSmemSizeOp, i32_ty, i64_ty);
    check_width!(ReadPtxSregGridIdOp, i64_ty, i32_ty);
}

#[test]
fn test_sync_ops_construct_and_verify() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let barrier = Operation::new(
        &mut ctx,
        Barrier0Op::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    assert!(Barrier0Op::new(barrier).verify(&ctx).is_ok());

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let unexpected_operand = block.deref(&ctx).get_argument(0);
    let bad_operand = Operation::new(
        &mut ctx,
        Barrier0Op::get_concrete_op_info(),
        vec![],
        vec![unexpected_operand],
        vec![],
        0,
    );
    assert!(verify_op(&Barrier0Op::new(bad_operand), &ctx).is_err());

    let bad_result = Operation::new(
        &mut ctx,
        Barrier0Op::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );
    assert!(verify_op(&Barrier0Op::new(bad_result), &ctx).is_err());

    let block_fence = Operation::new(
        &mut ctx,
        ThreadfenceBlockOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    assert!(ThreadfenceBlockOp::new(block_fence).verify(&ctx).is_ok());

    let device_fence = Operation::new(
        &mut ctx,
        ThreadfenceOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    assert!(ThreadfenceOp::new(device_fence).verify(&ctx).is_ok());

    let system_fence = Operation::new(
        &mut ctx,
        ThreadfenceSystemOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    assert!(ThreadfenceSystemOp::new(system_fence).verify(&ctx).is_ok());
}

#[test]
fn test_bf16x2_fma_constructs_and_verifies_three_operands() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);

    let a = Operation::new(
        &mut ctx,
        ReadPtxSregTidXOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );
    let b = Operation::new(
        &mut ctx,
        ReadPtxSregTidXOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );
    let c = Operation::new(
        &mut ctx,
        ReadPtxSregTidXOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );

    let operands = vec![
        a.deref(&ctx).get_result(0),
        b.deref(&ctx).get_result(0),
        c.deref(&ctx).get_result(0),
    ];

    let fma = Operation::new(
        &mut ctx,
        FmaBf16x2Op::get_concrete_op_info(),
        vec![u32_ty.into()],
        operands,
        vec![],
        0,
    );

    assert!(FmaBf16x2Op::new(fma).verify(&ctx).is_ok());
}

#[test]
fn test_generated_dot_products_require_three_i32_operands_and_one_i32_result() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![i32_ty.into(), i32_ty.into(), i32_ty.into(), i64_ty.into()],
    );
    let a = block.deref(&ctx).get_argument(0);
    let b = block.deref(&ctx).get_argument(1);
    let c = block.deref(&ctx).get_argument(2);
    let wide = block.deref(&ctx).get_argument(3);

    macro_rules! check_variant {
        ($op:ty) => {{
            let valid = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![i32_ty.into()],
                vec![a, b, c],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(valid), &ctx).is_ok());

            let wrong_width = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![i32_ty.into()],
                vec![a, b, wide],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(wrong_width), &ctx).is_err());

            for (results, operands) in [
                (vec![], vec![a, b, c]),
                (vec![i32_ty.into(), i32_ty.into()], vec![a, b, c]),
                (vec![i32_ty.into()], vec![a, b]),
                (vec![i32_ty.into()], vec![a, b, c, c]),
            ] {
                let wrong_count = Operation::new(
                    &mut ctx,
                    <$op>::get_concrete_op_info(),
                    results,
                    operands,
                    vec![],
                    0,
                );
                assert!(verify_op(&<$op>::new(wrong_count), &ctx).is_err());
            }
        }};
    }

    check_variant!(Dp4aS32Op);
    check_variant!(Dp4aU32Op);
    check_variant!(Dp2aS32Op);
    check_variant!(Dp2aU32Op);
}

#[test]
fn test_redux_sync_add_construct_and_verify() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);

    // A block supplies the two operands [mask, value].
    let block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into(), i32_ty.into()]);
    let mask = block.deref(&ctx).get_argument(0);
    let value = block.deref(&ctx).get_argument(1);

    // Valid: 2 operands, 1 result (matches NOpdsInterface<2>/NResultsInterface<1>).
    let op = Operation::new(
        &mut ctx,
        ReduxSyncAddOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![mask, value],
        vec![],
        0,
    );
    assert!(verify_op(&ReduxSyncAddOp::new(op), &ctx).is_ok());

    // Invalid: wrong operand count (1 instead of 2) must fail verification.
    let bad_opnds = Operation::new(
        &mut ctx,
        ReduxSyncAddOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![mask],
        vec![],
        0,
    );
    assert!(verify_op(&ReduxSyncAddOp::new(bad_opnds), &ctx).is_err());

    // Invalid: wrong result count (0 instead of 1) must fail verification.
    let bad_results = Operation::new(
        &mut ctx,
        ReduxSyncAddOp::get_concrete_op_info(),
        vec![],
        vec![mask, value],
        vec![],
        0,
    );
    assert!(verify_op(&ReduxSyncAddOp::new(bad_results), &ctx).is_err());
}

#[test]
fn test_redux_sync_integer_family_construct_and_verify() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into(), i32_ty.into()]);
    let mask = block.deref(&ctx).get_argument(0);
    let value = block.deref(&ctx).get_argument(1);

    // Every integer-family variant has the same 2-operand/1-result shape. A
    // valid build of each must verify; a wrong operand count must not. The
    // `new` wrapper is invoked so each concrete op type is exercised.
    macro_rules! check_variant {
        ($op:ty) => {{
            let good = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![i32_ty.into()],
                vec![mask, value],
                vec![],
                0,
            );
            assert!(
                verify_op(&<$op>::new(good), &ctx).is_ok(),
                "{} should verify with [mask, value] -> i32",
                stringify!($op)
            );

            let bad = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![i32_ty.into()],
                vec![mask],
                vec![],
                0,
            );
            assert!(
                verify_op(&<$op>::new(bad), &ctx).is_err(),
                "{} must reject a single operand",
                stringify!($op)
            );
        }};
    }

    check_variant!(ReduxSyncUminOp);
    check_variant!(ReduxSyncMinOp);
    check_variant!(ReduxSyncUmaxOp);
    check_variant!(ReduxSyncMaxOp);
    check_variant!(ReduxSyncAndOp);
    check_variant!(ReduxSyncOrOp);
    check_variant!(ReduxSyncXorOp);
}

#[test]
fn test_redux_sync_f32_family_construct_and_verify() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let f32_ty = FP32Type::get(&ctx);
    let block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into(), f32_ty.into()]);
    let mask = block.deref(&ctx).get_argument(0);
    let value = block.deref(&ctx).get_argument(1);

    // The good case goes through the generated `build()` so its f32 result
    // construction is exercised; the bad case's i32 value and result must
    // fail the `is_f32` verifier path.
    macro_rules! check_variant {
        ($op:ty) => {{
            let good = <$op>::build(&mut ctx, mask, value);
            assert!(
                verify_op(&<$op>::new(good), &ctx).is_ok(),
                "{} should verify with [mask, value] -> f32",
                stringify!($op)
            );

            let bad = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![i32_ty.into()],
                vec![mask, mask],
                vec![],
                0,
            );
            assert!(
                verify_op(&<$op>::new(bad), &ctx).is_err(),
                "{} must reject an i32 value and result",
                stringify!($op)
            );
        }};
    }

    check_variant!(ReduxSyncFminOp);
    check_variant!(ReduxSyncFminNanOp);
    check_variant!(ReduxSyncFminAbsOp);
    check_variant!(ReduxSyncFminAbsNanOp);
    check_variant!(ReduxSyncFmaxOp);
    check_variant!(ReduxSyncFmaxNanOp);
    check_variant!(ReduxSyncFmaxAbsOp);
    check_variant!(ReduxSyncFmaxAbsNanOp);
}

#[test]
fn test_bar_warp_sync_construct_and_verify() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into(), i64_ty.into()]);
    let mask = block.deref(&ctx).get_argument(0);
    let wrong_mask = block.deref(&ctx).get_argument(1);

    let valid = BarWarpSyncOp::build(&mut ctx, mask);
    assert!(verify_op(&BarWarpSyncOp::new(valid), &ctx).is_ok());

    let wrong_type = BarWarpSyncOp::build(&mut ctx, wrong_mask);
    assert!(verify_op(&BarWarpSyncOp::new(wrong_type), &ctx).is_err());

    let wrong_arity = Operation::new(
        &mut ctx,
        BarWarpSyncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    assert!(verify_op(&BarWarpSyncOp::new(wrong_arity), &ctx).is_err());
}

#[test]
fn test_elect_sync_construct_and_verify() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);

    // A block supplies the single `mask` operand.
    let block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let mask = block.deref(&ctx).get_argument(0);

    // Valid: 1 operand [mask], 2 results [leader (i32), is_elected (i1)]
    // (matches NOpdsInterface<1>/NResultsInterface<2>).
    let op = Operation::new(
        &mut ctx,
        ElectSyncOp::get_concrete_op_info(),
        vec![i32_ty.into(), i1_ty.into()],
        vec![mask],
        vec![],
        0,
    );
    assert!(verify_op(&ElectSyncOp::new(op), &ctx).is_ok());

    // Invalid: wrong operand count (0 instead of 1) must fail verification.
    let bad_opnds = Operation::new(
        &mut ctx,
        ElectSyncOp::get_concrete_op_info(),
        vec![i32_ty.into(), i1_ty.into()],
        vec![],
        vec![],
        0,
    );
    assert!(verify_op(&ElectSyncOp::new(bad_opnds), &ctx).is_err());

    // Invalid: wrong result count (1 instead of 2) must fail verification.
    let bad_results = Operation::new(
        &mut ctx,
        ElectSyncOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![mask],
        vec![],
        0,
    );
    assert!(verify_op(&ElectSyncOp::new(bad_results), &ctx).is_err());

    // Invalid: a correctly-sized result list still cannot use a pointer as
    // either result and thereby manufacture a MIR pointer kind.
    let pointer_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, i32_ty.into(), true, MirPointerKind::UniqueRef);
    let bad_result_type = Operation::new(
        &mut ctx,
        ElectSyncOp::get_concrete_op_info(),
        vec![i32_ty.into(), pointer_ty.into()],
        vec![mask],
        vec![],
        0,
    );
    assert!(verify_op(&ElectSyncOp::new(bad_result_type), &ctx).is_err());
}

#[test]
fn test_clc_query_rejects_pointer_results() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let block = BasicBlock::new(&mut ctx, None, vec![u64_ty.into(), u64_ty.into()]);
    let operands = vec![
        block.deref(&ctx).get_argument(0),
        block.deref(&ctx).get_argument(1),
    ];

    let valid = Operation::new(
        &mut ctx,
        ClcQueryIsCanceledOp::get_concrete_op_info(),
        vec![u32_ty.into()],
        operands.clone(),
        vec![],
        0,
    );
    assert!(verify_op(&ClcQueryIsCanceledOp::new(valid), &ctx).is_ok());

    let pointer_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, u32_ty.into(), true, MirPointerKind::UniqueRef);
    let bad = Operation::new(
        &mut ctx,
        ClcQueryIsCanceledOp::get_concrete_op_info(),
        vec![pointer_ty.into()],
        operands,
        vec![],
        0,
    );
    assert!(verify_op(&ClcQueryIsCanceledOp::new(bad), &ctx).is_err());
}

#[test]
fn test_shfl_sync_i64_construct_and_verify() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);

    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![i32_ty.into(), i64_ty.into(), i32_ty.into()],
    );
    let mask = block.deref(&ctx).get_argument(0);
    let value = block.deref(&ctx).get_argument(1);
    let lane = block.deref(&ctx).get_argument(2);

    macro_rules! check_mode {
        ($op:ty) => {{
            let valid = <$op>::build(&mut ctx, mask, value, lane);
            assert!(verify_op(&<$op>::new(valid), &ctx).is_ok());

            for (operands, result_ty) in [
                (vec![mask, value], i64_ty.into()),
                (vec![value, value, lane], i64_ty.into()),
                (vec![mask, mask, lane], i64_ty.into()),
                (vec![mask, value, value], i64_ty.into()),
                (vec![mask, value, lane], i32_ty.into()),
            ] {
                let invalid = Operation::new(
                    &mut ctx,
                    <$op>::get_concrete_op_info(),
                    vec![result_ty],
                    operands,
                    vec![],
                    0,
                );
                assert!(verify_op(&<$op>::new(invalid), &ctx).is_err());
            }

            let no_result = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![],
                vec![mask, value, lane],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(no_result), &ctx).is_err());
        }};
    }

    check_mode!(ShflSyncIdxI64Op);
    check_mode!(ShflSyncBflyI64Op);
    check_mode!(ShflSyncDownI64Op);
    check_mode!(ShflSyncUpI64Op);
}
