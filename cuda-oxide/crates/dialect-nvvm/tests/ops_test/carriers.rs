/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_mir::types::MirPtrType;
use dialect_nvvm::ops::{
    AssertFailOp, CvtaGenericToSharedOffsetOp, VprintfOp, WgmmaMakeSmemDescOp, WgmmaMaxPendingAttr,
    WgmmaMmaGroupM64N64K16F32Bf16Op, WgmmaMmaGroupValuesM64N64K8F32Tf32Op,
    WgmmaMmaGroupValuesM64N64K16F32Bf16Op, WgmmaMmaGroupValuesM64N64K16F32F16Op,
    WgmmaMmaGroupValuesM64N128K16F32Bf16Op, WgmmaMmaLoopPipelineValuesM64N64K16F32Bf16Op,
    WgmmaMmaLoopValuesM64N64K16F32Bf16Op, WgmmaMmaLoopValuesM64N64K16F32F16Op,
    WgmmaMmaM64N64K8F32Tf32Op, WgmmaMmaM64N64K16F32Bf16Op, WgmmaMmaM64N64K16F32F16Op,
    WgmmaMmaM64N128K16F32Bf16Op, WgmmaMmaPipelineValuesM64N64K16F32Bf16Op,
};

use pliron::{
    basic_block::BasicBlock,
    builtin::types::{FP32Type, IntegerType, Signedness},
    common_traits::Verify,
    context::Context,
    op::Op,
    operation::Operation,
};

#[test]
fn handwritten_ffi_and_wgmma_carriers_verify_exact_shapes() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let u8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let f32_ty = FP32Type::get(&ctx);
    let pointer_ty = MirPtrType::get_generic(&mut ctx, u8_ty.into(), false);
    let accumulator_pointer_ty = MirPtrType::get_generic(&mut ctx, u8_ty.into(), true);
    let global_pointer_ty = MirPtrType::get_global(&mut ctx, u8_ty.into(), false);
    let mutable_global_pointer_ty = MirPtrType::get_global(&mut ctx, u8_ty.into(), true);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            pointer_ty.into(),
            accumulator_pointer_ty.into(),
            global_pointer_ty.into(),
            mutable_global_pointer_ty.into(),
            u32_ty.into(),
            u64_ty.into(),
            f32_ty.into(),
        ],
    );
    let pointer = block.deref(&ctx).get_argument(0);
    let accumulator_pointer = block.deref(&ctx).get_argument(1);
    let global_pointer = block.deref(&ctx).get_argument(2);
    let mutable_global_pointer = block.deref(&ctx).get_argument(3);
    let u32_value = block.deref(&ctx).get_argument(4);
    let u64_value = block.deref(&ctx).get_argument(5);
    let f32_value = block.deref(&ctx).get_argument(6);

    let vprintf = VprintfOp::build(&mut ctx, pointer, pointer);
    assert!(VprintfOp::new(vprintf).verify(&ctx).is_ok());

    let i64_signless_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let assertfail = AssertFailOp::build(&mut ctx, pointer, pointer, u32_value, pointer, u64_value);
    assert!(AssertFailOp::new(assertfail).verify(&ctx).is_ok());
    for (operands, results) in [
        // message must be a MIR pointer
        (
            vec![u32_value, pointer, u32_value, pointer, u64_value],
            vec![],
        ),
        // line must be a 32-bit integer
        (
            vec![pointer, pointer, u64_value, pointer, u64_value],
            vec![],
        ),
        // char size must be a 64-bit integer
        (
            vec![pointer, pointer, u32_value, pointer, u32_value],
            vec![],
        ),
        // wrong operand count
        (vec![pointer, pointer, u32_value], vec![]),
        // must have no results
        (
            vec![pointer, pointer, u32_value, pointer, u64_value],
            vec![i64_signless_ty.into()],
        ),
    ] {
        let invalid = Operation::new(
            &mut ctx,
            AssertFailOp::get_concrete_op_info(),
            results,
            operands,
            vec![],
            0,
        );
        assert!(AssertFailOp::new(invalid).verify(&ctx).is_err());
    }
    let bad_vprintf = Operation::new(
        &mut ctx,
        VprintfOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![pointer, u32_value],
        vec![],
        0,
    );
    assert!(VprintfOp::new(bad_vprintf).verify(&ctx).is_err());

    let descriptor = Operation::new(
        &mut ctx,
        WgmmaMakeSmemDescOp::get_concrete_op_info(),
        vec![u64_ty.into()],
        vec![pointer],
        vec![],
        0,
    );
    assert!(WgmmaMakeSmemDescOp::new(descriptor).verify(&ctx).is_ok());
    for (operands, results) in [
        (vec![u32_value], vec![u64_ty.into()]),
        (vec![global_pointer], vec![u64_ty.into()]),
        (vec![pointer], vec![u32_ty.into()]),
        (vec![], vec![u64_ty.into()]),
    ] {
        let invalid = Operation::new(
            &mut ctx,
            WgmmaMakeSmemDescOp::get_concrete_op_info(),
            results,
            operands,
            vec![],
            0,
        );
        assert!(WgmmaMakeSmemDescOp::new(invalid).verify(&ctx).is_err());
    }

    let cvta = Operation::new(
        &mut ctx,
        CvtaGenericToSharedOffsetOp::get_concrete_op_info(),
        vec![u64_ty.into()],
        vec![pointer],
        vec![],
        0,
    );
    assert!(CvtaGenericToSharedOffsetOp::new(cvta).verify(&ctx).is_ok());
    let cvta_u32 = Operation::new(
        &mut ctx,
        CvtaGenericToSharedOffsetOp::get_concrete_op_info(),
        vec![u32_ty.into()],
        vec![pointer],
        vec![],
        0,
    );
    assert!(
        CvtaGenericToSharedOffsetOp::new(cvta_u32)
            .verify(&ctx)
            .is_ok()
    );
    for (operands, results) in [
        // Operand must be a MIR pointer.
        (vec![u32_value], vec![u64_ty.into()]),
        // Operand must point to generic or shared memory.
        (vec![global_pointer], vec![u64_ty.into()]),
        // Result must be an unsigned u32 or u64.
        (vec![pointer], vec![i32_ty.into()]),
        // Exact arity is required.
        (vec![], vec![u64_ty.into()]),
    ] {
        let invalid = Operation::new(
            &mut ctx,
            CvtaGenericToSharedOffsetOp::get_concrete_op_info(),
            results,
            operands,
            vec![],
            0,
        );
        assert!(
            CvtaGenericToSharedOffsetOp::new(invalid)
                .verify(&ctx)
                .is_err()
        );
    }

    let mma = Operation::new(
        &mut ctx,
        WgmmaMmaM64N64K16F32Bf16Op::get_concrete_op_info(),
        vec![],
        vec![accumulator_pointer, u64_value, u64_value],
        vec![],
        0,
    );
    assert!(WgmmaMmaM64N64K16F32Bf16Op::new(mma).verify(&ctx).is_ok());
    for (operands, results) in [
        // Accumulator must be mutable.
        (vec![pointer, u64_value, u64_value], vec![]),
        // Accumulator must use generic address space.
        (vec![mutable_global_pointer, u64_value, u64_value], vec![]),
        // Accumulator must be a MIR pointer.
        (vec![u32_value, u64_value, u64_value], vec![]),
        // Descriptors must be u64.
        (vec![accumulator_pointer, u32_value, u64_value], vec![]),
        // Exact arity is required.
        (vec![accumulator_pointer, u64_value], vec![]),
        // No results are permitted.
        (
            vec![accumulator_pointer, u64_value, u64_value],
            vec![u32_ty.into()],
        ),
    ] {
        let invalid = Operation::new(
            &mut ctx,
            WgmmaMmaM64N64K16F32Bf16Op::get_concrete_op_info(),
            results,
            operands,
            vec![],
            0,
        );
        assert!(
            WgmmaMmaM64N64K16F32Bf16Op::new(invalid)
                .verify(&ctx)
                .is_err()
        );
    }

    let group = WgmmaMmaGroupM64N64K16F32Bf16Op::build(
        &mut ctx,
        accumulator_pointer,
        vec![u64_value, u64_value, u64_value, u64_value],
    );
    assert!(
        WgmmaMmaGroupM64N64K16F32Bf16Op::new(group)
            .verify(&ctx)
            .is_ok()
    );
    for (operands, results) in [
        // Missing descriptor.
        (vec![accumulator_pointer, u64_value], vec![]),
        // Incomplete descriptor pair.
        (
            vec![accumulator_pointer, u64_value, u64_value, u64_value],
            vec![],
        ),
        // Accumulator must be mutable.
        (vec![pointer, u64_value, u64_value], vec![]),
        // Accumulator must use generic address space.
        (vec![mutable_global_pointer, u64_value, u64_value], vec![]),
        // Accumulator must be a MIR pointer.
        (vec![u32_value, u64_value, u64_value], vec![]),
        // Descriptors must be u64.
        (vec![accumulator_pointer, u64_value, u32_value], vec![]),
        // No results are permitted.
        (
            vec![accumulator_pointer, u64_value, u64_value],
            vec![u32_ty.into()],
        ),
    ] {
        let invalid = Operation::new(
            &mut ctx,
            WgmmaMmaGroupM64N64K16F32Bf16Op::get_concrete_op_info(),
            results,
            operands,
            vec![],
            0,
        );
        assert!(
            WgmmaMmaGroupM64N64K16F32Bf16Op::new(invalid)
                .verify(&ctx)
                .is_err()
        );
    }

    let value_group = WgmmaMmaGroupValuesM64N64K16F32Bf16Op::build(
        &mut ctx,
        vec![f32_value; 32],
        vec![u64_value, u64_value, u64_value, u64_value],
    );
    {
        let value_group_ref = value_group.deref(&ctx);
        assert_eq!(value_group_ref.get_num_operands(), 36);
        assert_eq!(value_group_ref.get_num_results(), 32);
    }
    assert!(
        WgmmaMmaGroupValuesM64N64K16F32Bf16Op::new(value_group)
            .verify(&ctx)
            .is_ok()
    );

    let too_few_accumulators = WgmmaMmaGroupValuesM64N64K16F32Bf16Op::build(
        &mut ctx,
        vec![f32_value; 31],
        vec![u64_value, u64_value],
    );
    assert!(
        WgmmaMmaGroupValuesM64N64K16F32Bf16Op::new(too_few_accumulators)
            .verify(&ctx)
            .is_err()
    );

    let missing_descriptors =
        WgmmaMmaGroupValuesM64N64K16F32Bf16Op::build(&mut ctx, vec![f32_value; 32], vec![]);
    assert!(
        WgmmaMmaGroupValuesM64N64K16F32Bf16Op::new(missing_descriptors)
            .verify(&ctx)
            .is_err()
    );

    let incomplete_descriptor_pair = WgmmaMmaGroupValuesM64N64K16F32Bf16Op::build(
        &mut ctx,
        vec![f32_value; 32],
        vec![u64_value, u64_value, u64_value],
    );
    assert!(
        WgmmaMmaGroupValuesM64N64K16F32Bf16Op::new(incomplete_descriptor_pair)
            .verify(&ctx)
            .is_err()
    );

    let mut wrong_accumulator_operands = vec![f32_value; 32];
    wrong_accumulator_operands[0] = u32_value;
    wrong_accumulator_operands.extend([u64_value, u64_value]);
    let wrong_accumulator = Operation::new(
        &mut ctx,
        WgmmaMmaGroupValuesM64N64K16F32Bf16Op::get_concrete_op_info(),
        vec![f32_ty.into(); 32],
        wrong_accumulator_operands,
        vec![],
        0,
    );
    assert!(
        WgmmaMmaGroupValuesM64N64K16F32Bf16Op::new(wrong_accumulator)
            .verify(&ctx)
            .is_err()
    );

    let mut wrong_descriptor_operands = vec![f32_value; 32];
    wrong_descriptor_operands.extend([u32_value, u64_value]);
    let wrong_descriptor = Operation::new(
        &mut ctx,
        WgmmaMmaGroupValuesM64N64K16F32Bf16Op::get_concrete_op_info(),
        vec![f32_ty.into(); 32],
        wrong_descriptor_operands,
        vec![],
        0,
    );
    assert!(
        WgmmaMmaGroupValuesM64N64K16F32Bf16Op::new(wrong_descriptor)
            .verify(&ctx)
            .is_err()
    );

    let mut valid_value_operands = vec![f32_value; 32];
    valid_value_operands.extend([u64_value, u64_value]);

    let wrong_result_count = Operation::new(
        &mut ctx,
        WgmmaMmaGroupValuesM64N64K16F32Bf16Op::get_concrete_op_info(),
        vec![f32_ty.into(); 31],
        valid_value_operands.clone(),
        vec![],
        0,
    );
    assert!(
        WgmmaMmaGroupValuesM64N64K16F32Bf16Op::new(wrong_result_count)
            .verify(&ctx)
            .is_err()
    );

    let mut wrong_result_types = vec![f32_ty.into(); 32];
    wrong_result_types[0] = u32_ty.into();
    let wrong_result_type = Operation::new(
        &mut ctx,
        WgmmaMmaGroupValuesM64N64K16F32Bf16Op::get_concrete_op_info(),
        wrong_result_types,
        valid_value_operands,
        vec![],
        0,
    );
    assert!(
        WgmmaMmaGroupValuesM64N64K16F32Bf16Op::new(wrong_result_type)
            .verify(&ctx)
            .is_err()
    );

    let wide_mma = Operation::new(
        &mut ctx,
        WgmmaMmaM64N128K16F32Bf16Op::get_concrete_op_info(),
        vec![],
        vec![accumulator_pointer, u64_value, u64_value],
        vec![],
        0,
    );
    assert!(
        WgmmaMmaM64N128K16F32Bf16Op::new(wide_mma)
            .verify(&ctx)
            .is_ok()
    );

    let wide_value_group = WgmmaMmaGroupValuesM64N128K16F32Bf16Op::build(
        &mut ctx,
        vec![f32_value; 64],
        vec![u64_value, u64_value],
    );
    {
        let group_ref = wide_value_group.deref(&ctx);
        assert_eq!(group_ref.get_num_operands(), 66);
        assert_eq!(group_ref.get_num_results(), 64);
    }
    assert!(
        WgmmaMmaGroupValuesM64N128K16F32Bf16Op::new(wide_value_group)
            .verify(&ctx)
            .is_ok()
    );

    let wide_too_few = WgmmaMmaGroupValuesM64N128K16F32Bf16Op::build(
        &mut ctx,
        vec![f32_value; 63],
        vec![u64_value, u64_value],
    );
    assert!(
        WgmmaMmaGroupValuesM64N128K16F32Bf16Op::new(wide_too_few)
            .verify(&ctx)
            .is_err()
    );

    let wide_incomplete_pair = WgmmaMmaGroupValuesM64N128K16F32Bf16Op::build(
        &mut ctx,
        vec![f32_value; 64],
        vec![u64_value, u64_value, u64_value],
    );
    assert!(
        WgmmaMmaGroupValuesM64N128K16F32Bf16Op::new(wide_incomplete_pair)
            .verify(&ctx)
            .is_err()
    );

    let f16_mma = Operation::new(
        &mut ctx,
        WgmmaMmaM64N64K16F32F16Op::get_concrete_op_info(),
        vec![],
        vec![accumulator_pointer, u64_value, u64_value],
        vec![],
        0,
    );
    assert!(WgmmaMmaM64N64K16F32F16Op::new(f16_mma).verify(&ctx).is_ok());
    for operands in [
        vec![pointer, u64_value, u64_value],
        vec![mutable_global_pointer, u64_value, u64_value],
        vec![accumulator_pointer, u32_value, u64_value],
    ] {
        let invalid = Operation::new(
            &mut ctx,
            WgmmaMmaM64N64K16F32F16Op::get_concrete_op_info(),
            vec![],
            operands,
            vec![],
            0,
        );
        assert!(
            WgmmaMmaM64N64K16F32F16Op::new(invalid)
                .verify(&ctx)
                .is_err()
        );
    }

    let f16_value_group = WgmmaMmaGroupValuesM64N64K16F32F16Op::build(
        &mut ctx,
        vec![f32_value; 32],
        vec![u64_value, u64_value],
    );
    assert!(
        WgmmaMmaGroupValuesM64N64K16F32F16Op::new(f16_value_group)
            .verify(&ctx)
            .is_ok()
    );

    let f16_too_few_accumulators = WgmmaMmaGroupValuesM64N64K16F32F16Op::build(
        &mut ctx,
        vec![f32_value; 31],
        vec![u64_value, u64_value],
    );
    assert!(
        WgmmaMmaGroupValuesM64N64K16F32F16Op::new(f16_too_few_accumulators)
            .verify(&ctx)
            .is_err()
    );

    let f16_incomplete_descriptor_pair = WgmmaMmaGroupValuesM64N64K16F32F16Op::build(
        &mut ctx,
        vec![f32_value; 32],
        vec![u64_value, u64_value, u64_value],
    );
    assert!(
        WgmmaMmaGroupValuesM64N64K16F32F16Op::new(f16_incomplete_descriptor_pair)
            .verify(&ctx)
            .is_err()
    );

    let mut f16_wrong_accumulator_operands = vec![f32_value; 32];
    f16_wrong_accumulator_operands[0] = u32_value;
    f16_wrong_accumulator_operands.extend([u64_value, u64_value]);
    let f16_wrong_accumulator = Operation::new(
        &mut ctx,
        WgmmaMmaGroupValuesM64N64K16F32F16Op::get_concrete_op_info(),
        vec![f32_ty.into(); 32],
        f16_wrong_accumulator_operands,
        vec![],
        0,
    );
    assert!(
        WgmmaMmaGroupValuesM64N64K16F32F16Op::new(f16_wrong_accumulator)
            .verify(&ctx)
            .is_err()
    );

    let tf32_mma = Operation::new(
        &mut ctx,
        WgmmaMmaM64N64K8F32Tf32Op::get_concrete_op_info(),
        vec![],
        vec![accumulator_pointer, u64_value, u64_value],
        vec![],
        0,
    );
    assert!(
        WgmmaMmaM64N64K8F32Tf32Op::new(tf32_mma)
            .verify(&ctx)
            .is_ok()
    );
    for operands in [
        vec![pointer, u64_value, u64_value],
        vec![mutable_global_pointer, u64_value, u64_value],
        vec![accumulator_pointer, u32_value, u64_value],
    ] {
        let invalid = Operation::new(
            &mut ctx,
            WgmmaMmaM64N64K8F32Tf32Op::get_concrete_op_info(),
            vec![],
            operands,
            vec![],
            0,
        );
        assert!(
            WgmmaMmaM64N64K8F32Tf32Op::new(invalid)
                .verify(&ctx)
                .is_err()
        );
    }

    let tf32_value_group = WgmmaMmaGroupValuesM64N64K8F32Tf32Op::build(
        &mut ctx,
        vec![f32_value; 32],
        vec![u64_value, u64_value],
    );
    assert!(
        WgmmaMmaGroupValuesM64N64K8F32Tf32Op::new(tf32_value_group)
            .verify(&ctx)
            .is_ok()
    );

    let tf32_too_few_accumulators = WgmmaMmaGroupValuesM64N64K8F32Tf32Op::build(
        &mut ctx,
        vec![f32_value; 31],
        vec![u64_value, u64_value],
    );
    assert!(
        WgmmaMmaGroupValuesM64N64K8F32Tf32Op::new(tf32_too_few_accumulators)
            .verify(&ctx)
            .is_err()
    );

    let tf32_incomplete_descriptor_pair = WgmmaMmaGroupValuesM64N64K8F32Tf32Op::build(
        &mut ctx,
        vec![f32_value; 32],
        vec![u64_value, u64_value, u64_value],
    );
    assert!(
        WgmmaMmaGroupValuesM64N64K8F32Tf32Op::new(tf32_incomplete_descriptor_pair)
            .verify(&ctx)
            .is_err()
    );

    let mut tf32_wrong_accumulator_operands = vec![f32_value; 32];
    tf32_wrong_accumulator_operands[0] = u32_value;
    tf32_wrong_accumulator_operands.extend([u64_value, u64_value]);
    let tf32_wrong_accumulator = Operation::new(
        &mut ctx,
        WgmmaMmaGroupValuesM64N64K8F32Tf32Op::get_concrete_op_info(),
        vec![f32_ty.into(); 32],
        tf32_wrong_accumulator_operands,
        vec![],
        0,
    );
    assert!(
        WgmmaMmaGroupValuesM64N64K8F32Tf32Op::new(tf32_wrong_accumulator)
            .verify(&ctx)
            .is_err()
    );

    let loop_group = WgmmaMmaLoopValuesM64N64K16F32Bf16Op::build(
        &mut ctx,
        vec![f32_value; 32],
        u64_value,
        u64_value,
        u64_value,
        u64_value,
        u64_value,
    );
    {
        let loop_group_ref = loop_group.deref(&ctx);
        assert_eq!(loop_group_ref.get_num_operands(), 37);
        assert_eq!(loop_group_ref.get_num_results(), 32);
    }
    assert!(
        WgmmaMmaLoopValuesM64N64K16F32Bf16Op::new(loop_group)
            .verify(&ctx)
            .is_ok()
    );

    let f16_loop_group = WgmmaMmaLoopValuesM64N64K16F32F16Op::build(
        &mut ctx,
        vec![f32_value; 32],
        u64_value,
        u64_value,
        u64_value,
        u64_value,
        u64_value,
    );
    {
        let loop_group_ref = f16_loop_group.deref(&ctx);
        assert_eq!(loop_group_ref.get_num_operands(), 37);
        assert_eq!(loop_group_ref.get_num_results(), 32);
    }
    assert!(
        WgmmaMmaLoopValuesM64N64K16F32F16Op::new(f16_loop_group)
            .verify(&ctx)
            .is_ok()
    );

    let mut f16_wrong_loop_control_operands = vec![f32_value; 32];
    f16_wrong_loop_control_operands.extend([u64_value, u64_value, u32_value, u64_value, u64_value]);
    let f16_wrong_loop_control = Operation::new(
        &mut ctx,
        WgmmaMmaLoopValuesM64N64K16F32F16Op::get_concrete_op_info(),
        vec![f32_ty.into(); 32],
        f16_wrong_loop_control_operands,
        vec![],
        0,
    );
    assert!(
        WgmmaMmaLoopValuesM64N64K16F32F16Op::new(f16_wrong_loop_control)
            .verify(&ctx)
            .is_err()
    );

    let too_few_loop_accumulators = WgmmaMmaLoopValuesM64N64K16F32Bf16Op::build(
        &mut ctx,
        vec![f32_value; 31],
        u64_value,
        u64_value,
        u64_value,
        u64_value,
        u64_value,
    );
    assert!(
        WgmmaMmaLoopValuesM64N64K16F32Bf16Op::new(too_few_loop_accumulators)
            .verify(&ctx)
            .is_err()
    );

    let mut wrong_loop_control_operands = vec![f32_value; 32];
    wrong_loop_control_operands.extend([u64_value, u64_value, u32_value, u64_value, u64_value]);
    let wrong_loop_control = Operation::new(
        &mut ctx,
        WgmmaMmaLoopValuesM64N64K16F32Bf16Op::get_concrete_op_info(),
        vec![f32_ty.into(); 32],
        wrong_loop_control_operands,
        vec![],
        0,
    );
    assert!(
        WgmmaMmaLoopValuesM64N64K16F32Bf16Op::new(wrong_loop_control)
            .verify(&ctx)
            .is_err()
    );

    let mut valid_loop_operands = vec![f32_value; 32];
    valid_loop_operands.extend([u64_value; 5]);

    let wrong_loop_result_count = Operation::new(
        &mut ctx,
        WgmmaMmaLoopValuesM64N64K16F32Bf16Op::get_concrete_op_info(),
        vec![f32_ty.into(); 31],
        valid_loop_operands.clone(),
        vec![],
        0,
    );
    assert!(
        WgmmaMmaLoopValuesM64N64K16F32Bf16Op::new(wrong_loop_result_count)
            .verify(&ctx)
            .is_err()
    );

    let mut wrong_loop_result_types = vec![f32_ty.into(); 32];
    wrong_loop_result_types[0] = u32_ty.into();
    let wrong_loop_result_type = Operation::new(
        &mut ctx,
        WgmmaMmaLoopValuesM64N64K16F32Bf16Op::get_concrete_op_info(),
        wrong_loop_result_types,
        valid_loop_operands,
        vec![],
        0,
    );
    assert!(
        WgmmaMmaLoopValuesM64N64K16F32Bf16Op::new(wrong_loop_result_type)
            .verify(&ctx)
            .is_err()
    );

    let counted_pipeline_group = WgmmaMmaLoopPipelineValuesM64N64K16F32Bf16Op::build(
        &mut ctx,
        vec![f32_value; 64],
        vec![u64_value; 4],
        vec![u64_value; 4],
        u64_value,
        1,
    );
    {
        let counted_pipeline_ref = counted_pipeline_group.deref(&ctx);
        assert_eq!(counted_pipeline_ref.get_num_operands(), 73);
        assert_eq!(counted_pipeline_ref.get_num_results(), 64);
    }
    let counted_pipeline =
        WgmmaMmaLoopPipelineValuesM64N64K16F32Bf16Op::new(counted_pipeline_group);
    assert_eq!(counted_pipeline.max_pending_groups(&ctx), Some(1));
    assert!(counted_pipeline.verify(&ctx).is_ok());

    let three_slot_counted_pipeline = WgmmaMmaLoopPipelineValuesM64N64K16F32Bf16Op::build(
        &mut ctx,
        vec![f32_value; 96],
        vec![u64_value; 6],
        vec![u64_value; 6],
        u64_value,
        2,
    );
    {
        let counted_pipeline_ref = three_slot_counted_pipeline.deref(&ctx);
        assert_eq!(counted_pipeline_ref.get_num_operands(), 109);
        assert_eq!(counted_pipeline_ref.get_num_results(), 96);
    }
    let three_slot_counted_pipeline =
        WgmmaMmaLoopPipelineValuesM64N64K16F32Bf16Op::new(three_slot_counted_pipeline);
    assert_eq!(
        three_slot_counted_pipeline.max_pending_groups(&ctx),
        Some(2)
    );
    assert!(three_slot_counted_pipeline.verify(&ctx).is_ok());

    let too_many_accumulators_for_wait_one = WgmmaMmaLoopPipelineValuesM64N64K16F32Bf16Op::build(
        &mut ctx,
        vec![f32_value; 96],
        vec![u64_value; 4],
        vec![u64_value; 4],
        u64_value,
        1,
    );
    assert!(
        WgmmaMmaLoopPipelineValuesM64N64K16F32Bf16Op::new(too_many_accumulators_for_wait_one)
            .verify(&ctx)
            .is_err()
    );

    let too_few_accumulators_for_wait_two = WgmmaMmaLoopPipelineValuesM64N64K16F32Bf16Op::build(
        &mut ctx,
        vec![f32_value; 64],
        vec![u64_value; 6],
        vec![u64_value; 6],
        u64_value,
        2,
    );
    assert!(
        WgmmaMmaLoopPipelineValuesM64N64K16F32Bf16Op::new(too_few_accumulators_for_wait_two)
            .verify(&ctx)
            .is_err()
    );

    let unsupported_wait_three = WgmmaMmaLoopPipelineValuesM64N64K16F32Bf16Op::build(
        &mut ctx,
        vec![f32_value; 128],
        vec![u64_value; 8],
        vec![u64_value; 8],
        u64_value,
        3,
    );
    assert!(
        WgmmaMmaLoopPipelineValuesM64N64K16F32Bf16Op::new(unsupported_wait_three)
            .verify(&ctx)
            .is_err()
    );

    let wrong_counted_pipeline_control = WgmmaMmaLoopPipelineValuesM64N64K16F32Bf16Op::build(
        &mut ctx,
        vec![f32_value; 64],
        vec![u64_value; 4],
        vec![u64_value, u64_value, u32_value, u64_value],
        u64_value,
        1,
    );
    assert!(
        WgmmaMmaLoopPipelineValuesM64N64K16F32Bf16Op::new(wrong_counted_pipeline_control)
            .verify(&ctx)
            .is_err()
    );

    let mut valid_counted_pipeline_operands = vec![f32_value; 64];
    valid_counted_pipeline_operands.extend([u64_value; 9]);
    let wrong_counted_pipeline_result_count = Operation::new(
        &mut ctx,
        WgmmaMmaLoopPipelineValuesM64N64K16F32Bf16Op::get_concrete_op_info(),
        vec![f32_ty.into(); 63],
        valid_counted_pipeline_operands.clone(),
        vec![],
        0,
    );
    let wrong_counted_pipeline_result_count =
        WgmmaMmaLoopPipelineValuesM64N64K16F32Bf16Op::new(wrong_counted_pipeline_result_count);
    wrong_counted_pipeline_result_count
        .set_attr_counted_max_pending_groups(&ctx, WgmmaMaxPendingAttr(1));
    assert!(wrong_counted_pipeline_result_count.verify(&ctx).is_err());

    let mut wrong_counted_pipeline_result_types = vec![f32_ty.into(); 64];
    wrong_counted_pipeline_result_types[0] = u32_ty.into();
    let wrong_counted_pipeline_result_type = Operation::new(
        &mut ctx,
        WgmmaMmaLoopPipelineValuesM64N64K16F32Bf16Op::get_concrete_op_info(),
        wrong_counted_pipeline_result_types,
        valid_counted_pipeline_operands,
        vec![],
        0,
    );
    let wrong_counted_pipeline_result_type =
        WgmmaMmaLoopPipelineValuesM64N64K16F32Bf16Op::new(wrong_counted_pipeline_result_type);
    wrong_counted_pipeline_result_type
        .set_attr_counted_max_pending_groups(&ctx, WgmmaMaxPendingAttr(1));
    assert!(wrong_counted_pipeline_result_type.verify(&ctx).is_err());

    let pipeline_group = WgmmaMmaPipelineValuesM64N64K16F32Bf16Op::build(
        &mut ctx,
        vec![f32_value; 64],
        vec![u64_value; 8],
        1,
    );
    {
        let pipeline_ref = pipeline_group.deref(&ctx);
        assert_eq!(pipeline_ref.get_num_operands(), 72);
        assert_eq!(pipeline_ref.get_num_results(), 64);
    }
    let pipeline = WgmmaMmaPipelineValuesM64N64K16F32Bf16Op::new(pipeline_group);
    assert_eq!(pipeline.max_pending_groups(&ctx), Some(1));
    assert!(pipeline.verify(&ctx).is_ok());

    let too_few_pipeline_slots = WgmmaMmaPipelineValuesM64N64K16F32Bf16Op::build(
        &mut ctx,
        vec![f32_value; 32],
        vec![u64_value; 4],
        1,
    );
    assert!(
        WgmmaMmaPipelineValuesM64N64K16F32Bf16Op::new(too_few_pipeline_slots)
            .verify(&ctx)
            .is_err()
    );

    assert!(WgmmaMaxPendingAttr(1).verify(&ctx).is_ok());
    assert!(WgmmaMaxPendingAttr(7).verify(&ctx).is_ok());
    assert!(WgmmaMaxPendingAttr(0).verify(&ctx).is_err());
    assert!(WgmmaMaxPendingAttr(8).verify(&ctx).is_err());

    let zero_pending_pipeline = WgmmaMmaPipelineValuesM64N64K16F32Bf16Op::build(
        &mut ctx,
        vec![f32_value; 32],
        vec![u64_value; 2],
        0,
    );
    let zero_pending_error = WgmmaMmaPipelineValuesM64N64K16F32Bf16Op::new(zero_pending_pipeline)
        .verify(&ctx)
        .unwrap_err();
    assert!(
        zero_pending_error
            .err
            .to_string()
            .contains("nvvm.wgmma_max_pending_groups must be in 1..=7, got 0"),
        "unexpected zero-pending error: {}",
        zero_pending_error.err
    );

    let excess_pending_pipeline = WgmmaMmaPipelineValuesM64N64K16F32Bf16Op::build(
        &mut ctx,
        vec![f32_value; 32],
        vec![u64_value; 2],
        8,
    );
    let excess_pending_error =
        WgmmaMmaPipelineValuesM64N64K16F32Bf16Op::new(excess_pending_pipeline)
            .verify(&ctx)
            .unwrap_err();
    assert!(
        excess_pending_error
            .err
            .to_string()
            .contains("nvvm.wgmma_max_pending_groups must be in 1..=7, got 8"),
        "unexpected excess-pending error: {}",
        excess_pending_error.err
    );

    let mut missing_attr_operands = vec![f32_value; 64];
    missing_attr_operands.extend([u64_value; 8]);
    let missing_attr_pipeline = Operation::new(
        &mut ctx,
        WgmmaMmaPipelineValuesM64N64K16F32Bf16Op::get_concrete_op_info(),
        vec![f32_ty.into(); 64],
        missing_attr_operands,
        vec![],
        0,
    );
    let missing_attr_error = WgmmaMmaPipelineValuesM64N64K16F32Bf16Op::new(missing_attr_pipeline)
        .verify(&ctx)
        .unwrap_err();
    assert!(
        missing_attr_error
            .err
            .to_string()
            .contains("requires an nvvm.wgmma_max_pending_groups attribute"),
        "unexpected missing-attribute error: {}",
        missing_attr_error.err
    );
}
