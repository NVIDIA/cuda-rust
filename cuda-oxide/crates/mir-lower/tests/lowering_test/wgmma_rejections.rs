/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_mir::ops as mir;
use dialect_nvvm::ops as nvvm;
use pliron::context::Context;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;

use crate::common::{append_return, build_test_kernel, make_test_ctx};
use crate::wgmma_lowering::{
    append_mir_unsigned_constant, append_pointer_wgmma_mma, append_pointer_wgmma_mma_f16,
    append_pointer_wgmma_mma_m64n128, append_pointer_wgmma_mma_tf32,
    append_wgmma_wait_group_constant, build_pointer_form_wgmma_counted_pipeline_case,
    build_wgmma_canonical_pointer_test_kernel, build_wgmma_pointer_test_kernel,
};

fn assert_wgmma_lowering_rejected(
    ctx: &mut Context,
    module_ptr: pliron::context::Ptr<Operation>,
    expected_diagnostic: &str,
) {
    let error = mir_lower::lower_mir_to_llvm(ctx, module_ptr)
        .expect_err("invalid deferred WGMMA sequence must fail closed")
        .to_string();

    assert!(
        error.contains(expected_diagnostic),
        "expected diagnostic containing `{expected_diagnostic}`, got:\n{error}"
    );
}

// ---------------------------------------------------------------------------
// mma.sync m16n8k16 f16 intrinsic lowering test
// ---------------------------------------------------------------------------

#[test]
fn test_pointer_form_wgmma_counted_pipeline_rejects_wait_two_with_two_slots() {
    let mut ctx = make_test_ctx();
    let module_ptr = build_pointer_form_wgmma_counted_pipeline_case(&mut ctx, 2, &[2, 2], false);

    assert!(
        mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).is_err(),
        "wait_group<2> must require three counted-pipeline accumulator slots"
    );
}

#[test]
fn test_pointer_form_wgmma_counted_pipeline_rejects_mixed_partial_waits() {
    let mut ctx = make_test_ctx();
    let module_ptr = build_pointer_form_wgmma_counted_pipeline_case(&mut ctx, 3, &[2, 1, 2], false);

    assert!(
        mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).is_err(),
        "all counted-pipeline stages must use the same partial-wait depth"
    );
}

#[test]
fn test_pointer_form_wgmma_counted_pipeline_rejects_reused_accumulator_slot() {
    let mut ctx = make_test_ctx();
    let module_ptr = build_pointer_form_wgmma_counted_pipeline_case(&mut ctx, 3, &[2, 2, 2], true);

    assert!(
        mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).is_err(),
        "counted-pipeline accumulator slots must be pairwise distinct"
    );
}

#[test]
fn test_tf32_wgmma_counted_k_loop_remains_unsupported() -> Result<(), anyhow::Error> {
    use dialect_mir::types::{MirArrayType, MirPtrType};
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::op_interfaces::OperandSegmentInterface;
    use pliron::builtin::types::{FP32Type, IntegerType, Signedness};

    const TRIP_COUNT: u64 = 4;
    const DESC_A_STEP: u64 = 16;
    const DESC_B_STEP: u64 = 32;

    let mut ctx = make_test_ctx();
    let f32_ty = FP32Type::get(&ctx);
    let row_ty = MirArrayType::get(&mut ctx, f32_ty.into(), 8);
    let accumulator_ty = MirArrayType::get(&mut ctx, row_ty.into(), 4);
    let accumulator_ptr_ty = MirPtrType::get_generic(&mut ctx, accumulator_ty.into(), true);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);

    let (module_ptr, preheader) = build_test_kernel(
        &mut ctx,
        vec![accumulator_ptr_ty.into(), u64_ty.into(), u64_ty.into()],
    );
    let accumulator = preheader.deref(&ctx).get_argument(0);
    let desc_a_base = preheader.deref(&ctx).get_argument(1);
    let desc_b_base = preheader.deref(&ctx).get_argument(2);

    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    let function = module_block.deref(&ctx).iter(&ctx).next().unwrap();
    let function_region = function.deref(&ctx).get_region(0);

    let header = BasicBlock::new(
        &mut ctx,
        None,
        vec![u32_ty.into(), u64_ty.into(), u64_ty.into()],
    );
    header.insert_at_back(function_region, &ctx);
    let latch = BasicBlock::new(&mut ctx, None, vec![]);
    latch.insert_at_back(function_region, &ctx);
    let exit = BasicBlock::new(&mut ctx, None, vec![]);
    exit.insert_at_back(function_region, &ctx);

    // preheader: fence; i0 = 0; goto header(i0, desc_a_base, desc_b_base)
    nvvm::WgmmaFenceSyncAlignedOp::build(&mut ctx).insert_at_back(preheader, &ctx);
    let i0 = append_mir_unsigned_constant(&mut ctx, preheader, u32_ty, 0);
    Operation::new(
        &mut ctx,
        mir::MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![i0, desc_a_base, desc_b_base],
        vec![header],
        0,
    )
    .insert_at_back(preheader, &ctx);

    // header(i, desc_a, desc_b): if !(i < 4) exit else latch.
    let i = header.deref(&ctx).get_argument(0);
    let desc_a = header.deref(&ctx).get_argument(1);
    let desc_b = header.deref(&ctx).get_argument(2);
    let bound = append_mir_unsigned_constant(&mut ctx, header, u32_ty, TRIP_COUNT);
    let lt = Operation::new(
        &mut ctx,
        mir::MirLtOp::get_concrete_op_info(),
        vec![i1_ty.into()],
        vec![i, bound],
        vec![],
        0,
    );
    lt.insert_at_back(header, &ctx);
    let lt_value = lt.deref(&ctx).get_result(0);
    let not_lt = Operation::new(
        &mut ctx,
        mir::MirNotOp::get_concrete_op_info(),
        vec![i1_ty.into()],
        vec![lt_value],
        vec![],
        0,
    );
    not_lt.insert_at_back(header, &ctx);
    let not_lt_value = not_lt.deref(&ctx).get_result(0);
    let (branch_operands, segment_sizes) =
        mir::MirCondBranchOp::compute_segment_sizes(vec![vec![not_lt_value], vec![], vec![]]);
    let branch = Operation::new(
        &mut ctx,
        mir::MirCondBranchOp::get_concrete_op_info(),
        vec![],
        branch_operands,
        vec![exit, latch],
        0,
    );
    Operation::get_op::<mir::MirCondBranchOp>(branch, &ctx)
        .expect("MirCondBranchOp")
        .set_operand_segment_sizes(&ctx, segment_sizes);
    branch.insert_at_back(header, &ctx);

    // latch: one WGMMA per K iteration and affine descriptor recurrences.
    append_pointer_wgmma_mma_tf32(&mut ctx, latch, accumulator, desc_a, desc_b);

    let one = append_mir_unsigned_constant(&mut ctx, latch, u32_ty, 1);
    let i_next = Operation::new(
        &mut ctx,
        mir::MirAddOp::get_concrete_op_info(),
        vec![u32_ty.into()],
        vec![i, one],
        vec![],
        0,
    );
    i_next.insert_at_back(latch, &ctx);
    let i_next = i_next.deref(&ctx).get_result(0);

    let desc_a_step = append_mir_unsigned_constant(&mut ctx, latch, u64_ty, DESC_A_STEP);
    let desc_a_next = Operation::new(
        &mut ctx,
        mir::MirAddOp::get_concrete_op_info(),
        vec![u64_ty.into()],
        vec![desc_a, desc_a_step],
        vec![],
        0,
    );
    desc_a_next.insert_at_back(latch, &ctx);
    let desc_a_next = desc_a_next.deref(&ctx).get_result(0);

    let desc_b_step = append_mir_unsigned_constant(&mut ctx, latch, u64_ty, DESC_B_STEP);
    let desc_b_next = Operation::new(
        &mut ctx,
        mir::MirAddOp::get_concrete_op_info(),
        vec![u64_ty.into()],
        vec![desc_b, desc_b_step],
        vec![],
        0,
    );
    desc_b_next.insert_at_back(latch, &ctx);
    let desc_b_next = desc_b_next.deref(&ctx).get_result(0);

    Operation::new(
        &mut ctx,
        mir::MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![i_next, desc_a_next, desc_b_next],
        vec![header],
        0,
    )
    .insert_at_back(latch, &ctx);

    // exit: the only place where the asynchronous lifetime may become visible.
    nvvm::WgmmaCommitGroupSyncAlignedOp::build(&mut ctx).insert_at_back(exit, &ctx);
    append_wgmma_wait_group_constant(&mut ctx, exit, 0);
    append_return(&mut ctx, exit);

    assert_wgmma_lowering_rejected(
        &mut ctx,
        module_ptr,
        "WGMMA MMA reached lowering without deferred accumulator fusion",
    );
    Ok(())
}

#[test]
fn test_m64n128_wgmma_noncanonical_accumulator_has_no_pointer_fallback() -> Result<(), anyhow::Error>
{
    let mut ctx = make_test_ctx();
    let (module_ptr, entry, accumulators, desc_a, desc_b, _) =
        build_wgmma_pointer_test_kernel(&mut ctx, 1, vec![]);

    nvvm::WgmmaFenceSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_pointer_wgmma_mma_m64n128(&mut ctx, entry, accumulators[0], desc_a, desc_b);
    nvvm::WgmmaCommitGroupSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_wgmma_wait_group_constant(&mut ctx, entry, 0);
    append_return(&mut ctx, entry);

    assert_wgmma_lowering_rejected(
        &mut ctx,
        module_ptr,
        "WGMMA linear full-drain lowering for this variant requires a canonical [[f32; 8]; 8] accumulator",
    );
    Ok(())
}

#[test]
fn test_f16_wgmma_noncanonical_accumulator_has_no_pointer_fallback() -> Result<(), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let (module_ptr, entry, accumulators, desc_a, desc_b, _) =
        build_wgmma_pointer_test_kernel(&mut ctx, 1, vec![]);

    nvvm::WgmmaFenceSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_pointer_wgmma_mma_f16(&mut ctx, entry, accumulators[0], desc_a, desc_b);
    nvvm::WgmmaCommitGroupSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_wgmma_wait_group_constant(&mut ctx, entry, 0);
    append_return(&mut ctx, entry);

    assert_wgmma_lowering_rejected(
        &mut ctx,
        module_ptr,
        "WGMMA linear full-drain lowering for this variant requires a canonical [[f32; 8]; 4] accumulator",
    );
    Ok(())
}

#[test]
fn test_tf32_wgmma_noncanonical_accumulator_has_no_pointer_fallback() -> Result<(), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let (module_ptr, entry, accumulators, desc_a, desc_b, _) =
        build_wgmma_pointer_test_kernel(&mut ctx, 1, vec![]);

    nvvm::WgmmaFenceSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_pointer_wgmma_mma_tf32(&mut ctx, entry, accumulators[0], desc_a, desc_b);
    nvvm::WgmmaCommitGroupSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_wgmma_wait_group_constant(&mut ctx, entry, 0);
    append_return(&mut ctx, entry);

    assert_wgmma_lowering_rejected(
        &mut ctx,
        module_ptr,
        "WGMMA linear full-drain lowering for this variant requires a canonical [[f32; 8]; 4] accumulator",
    );
    Ok(())
}

#[test]
fn test_linear_wgmma_full_drain_rejects_mixed_bf16_and_f16() -> Result<(), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let (module_ptr, entry, accumulators, descriptors) =
        build_wgmma_canonical_pointer_test_kernel(&mut ctx, 1, 4);
    let accumulator = accumulators[0];

    nvvm::WgmmaFenceSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_pointer_wgmma_mma(&mut ctx, entry, accumulator, descriptors[0], descriptors[1]);
    append_pointer_wgmma_mma_f16(&mut ctx, entry, accumulator, descriptors[2], descriptors[3]);
    nvvm::WgmmaCommitGroupSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_wgmma_wait_group_constant(&mut ctx, entry, 0);
    append_return(&mut ctx, entry);

    assert_wgmma_lowering_rejected(
        &mut ctx,
        module_ptr,
        "one linear WGMMA full-drain region cannot mix MMA variants or shapes",
    );
    Ok(())
}

#[test]
fn test_linear_wgmma_full_drain_rejects_mixed_bf16_and_tf32() -> Result<(), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let (module_ptr, entry, accumulators, descriptors) =
        build_wgmma_canonical_pointer_test_kernel(&mut ctx, 1, 4);
    let accumulator = accumulators[0];

    nvvm::WgmmaFenceSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_pointer_wgmma_mma(&mut ctx, entry, accumulator, descriptors[0], descriptors[1]);
    append_pointer_wgmma_mma_tf32(&mut ctx, entry, accumulator, descriptors[2], descriptors[3]);
    nvvm::WgmmaCommitGroupSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_wgmma_wait_group_constant(&mut ctx, entry, 0);
    append_return(&mut ctx, entry);

    assert_wgmma_lowering_rejected(
        &mut ctx,
        module_ptr,
        "one linear WGMMA full-drain region cannot mix MMA variants or shapes",
    );
    Ok(())
}

#[test]
fn test_linear_wgmma_full_drain_rejects_mixed_f16_and_tf32() -> Result<(), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let (module_ptr, entry, accumulators, descriptors) =
        build_wgmma_canonical_pointer_test_kernel(&mut ctx, 1, 4);
    let accumulator = accumulators[0];

    nvvm::WgmmaFenceSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_pointer_wgmma_mma_f16(&mut ctx, entry, accumulator, descriptors[0], descriptors[1]);
    append_pointer_wgmma_mma_tf32(&mut ctx, entry, accumulator, descriptors[2], descriptors[3]);
    nvvm::WgmmaCommitGroupSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_wgmma_wait_group_constant(&mut ctx, entry, 0);
    append_return(&mut ctx, entry);

    assert_wgmma_lowering_rejected(
        &mut ctx,
        module_ptr,
        "one linear WGMMA full-drain region cannot mix MMA variants or shapes",
    );
    Ok(())
}

#[test]
fn test_f16_wgmma_partial_wait_remains_unsupported() -> Result<(), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let (module_ptr, entry, accumulators, descriptors) =
        build_wgmma_canonical_pointer_test_kernel(&mut ctx, 1, 2);

    nvvm::WgmmaFenceSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_pointer_wgmma_mma_f16(
        &mut ctx,
        entry,
        accumulators[0],
        descriptors[0],
        descriptors[1],
    );
    nvvm::WgmmaCommitGroupSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_wgmma_wait_group_constant(&mut ctx, entry, 1);
    append_wgmma_wait_group_constant(&mut ctx, entry, 0);
    append_return(&mut ctx, entry);

    assert_wgmma_lowering_rejected(
        &mut ctx,
        module_ptr,
        "WGMMA deferred accumulator lowering requires wait_group<0>",
    );
    Ok(())
}

#[test]
fn test_tf32_wgmma_partial_wait_remains_unsupported() -> Result<(), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let (module_ptr, entry, accumulators, descriptors) =
        build_wgmma_canonical_pointer_test_kernel(&mut ctx, 1, 2);

    nvvm::WgmmaFenceSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_pointer_wgmma_mma_tf32(
        &mut ctx,
        entry,
        accumulators[0],
        descriptors[0],
        descriptors[1],
    );
    nvvm::WgmmaCommitGroupSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_wgmma_wait_group_constant(&mut ctx, entry, 1);
    append_wgmma_wait_group_constant(&mut ctx, entry, 0);
    append_return(&mut ctx, entry);

    assert_wgmma_lowering_rejected(
        &mut ctx,
        module_ptr,
        "WGMMA deferred accumulator lowering requires wait_group<0>",
    );
    Ok(())
}

#[test]
fn test_pointer_form_wgmma_without_fence_is_rejected() -> Result<(), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let (module_ptr, entry, accumulators, desc_a, desc_b, _) =
        build_wgmma_pointer_test_kernel(&mut ctx, 1, vec![]);

    append_pointer_wgmma_mma(&mut ctx, entry, accumulators[0], desc_a, desc_b);
    nvvm::WgmmaCommitGroupSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_wgmma_wait_group_constant(&mut ctx, entry, 0);
    append_return(&mut ctx, entry);

    assert_wgmma_lowering_rejected(
        &mut ctx,
        module_ptr,
        "WGMMA MMA reached lowering without deferred accumulator fusion",
    );
    Ok(())
}

#[test]
fn test_deferred_wgmma_without_commit_is_rejected() -> Result<(), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let (module_ptr, entry, accumulators, desc_a, desc_b, _) =
        build_wgmma_pointer_test_kernel(&mut ctx, 1, vec![]);

    nvvm::WgmmaFenceSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_pointer_wgmma_mma(&mut ctx, entry, accumulators[0], desc_a, desc_b);
    append_wgmma_wait_group_constant(&mut ctx, entry, 0);
    append_return(&mut ctx, entry);

    assert_wgmma_lowering_rejected(
        &mut ctx,
        module_ptr,
        "WGMMA wait_group requires a preceding commit_group",
    );
    Ok(())
}

#[test]
fn test_wgmma_partial_wait_without_final_wait_zero_is_rejected() -> Result<(), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let (module_ptr, entry, accumulators, desc_a, desc_b, _) =
        build_wgmma_pointer_test_kernel(&mut ctx, 1, vec![]);

    nvvm::WgmmaFenceSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_pointer_wgmma_mma(&mut ctx, entry, accumulators[0], desc_a, desc_b);
    nvvm::WgmmaCommitGroupSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_wgmma_wait_group_constant(&mut ctx, entry, 1);
    append_return(&mut ctx, entry);

    assert_wgmma_lowering_rejected(
        &mut ctx,
        module_ptr,
        "WGMMA partial-wait pipeline requires a final wait_group<0>",
    );
    Ok(())
}

#[test]
fn test_wgmma_partial_wait_pipeline_rejects_unsafe_accumulator_reuse() -> Result<(), anyhow::Error>
{
    let mut ctx = make_test_ctx();
    let (module_ptr, entry, accumulators, descriptors) =
        build_wgmma_canonical_pointer_test_kernel(&mut ctx, 1, 4);

    nvvm::WgmmaFenceSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_pointer_wgmma_mma(
        &mut ctx,
        entry,
        accumulators[0],
        descriptors[0],
        descriptors[1],
    );
    nvvm::WgmmaCommitGroupSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_pointer_wgmma_mma(
        &mut ctx,
        entry,
        accumulators[0],
        descriptors[2],
        descriptors[3],
    );
    nvvm::WgmmaCommitGroupSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_wgmma_wait_group_constant(&mut ctx, entry, 1);
    append_wgmma_wait_group_constant(&mut ctx, entry, 0);
    append_return(&mut ctx, entry);

    assert_wgmma_lowering_rejected(
        &mut ctx,
        module_ptr,
        "WGMMA partial-wait pipeline requires max_pending_groups + 1 distinct accumulator slots",
    );
    Ok(())
}

#[test]
fn test_wgmma_wait_group_eight_is_rejected() -> Result<(), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let (module_ptr, entry, accumulators, desc_a, desc_b, _) =
        build_wgmma_pointer_test_kernel(&mut ctx, 1, vec![]);

    nvvm::WgmmaFenceSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_pointer_wgmma_mma(&mut ctx, entry, accumulators[0], desc_a, desc_b);
    nvvm::WgmmaCommitGroupSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_wgmma_wait_group_constant(&mut ctx, entry, 8);
    append_return(&mut ctx, entry);

    assert_wgmma_lowering_rejected(
        &mut ctx,
        module_ptr,
        "WGMMA wait_group<N> immediate must be in 0..=7",
    );
    Ok(())
}

#[test]
fn test_deferred_wgmma_two_commits_are_rejected() -> Result<(), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let (module_ptr, entry, accumulators, desc_a, desc_b, _) =
        build_wgmma_pointer_test_kernel(&mut ctx, 1, vec![]);

    nvvm::WgmmaFenceSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_pointer_wgmma_mma(&mut ctx, entry, accumulators[0], desc_a, desc_b);
    nvvm::WgmmaCommitGroupSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    nvvm::WgmmaCommitGroupSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_wgmma_wait_group_constant(&mut ctx, entry, 0);
    append_return(&mut ctx, entry);

    assert_wgmma_lowering_rejected(
        &mut ctx,
        module_ptr,
        "WGMMA deferred accumulator region supports exactly one commit_group",
    );
    Ok(())
}

#[test]
fn test_deferred_wgmma_multiple_accumulators_are_rejected() -> Result<(), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let (module_ptr, entry, accumulators, desc_a, desc_b, _) =
        build_wgmma_pointer_test_kernel(&mut ctx, 2, vec![]);

    nvvm::WgmmaFenceSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_pointer_wgmma_mma(&mut ctx, entry, accumulators[0], desc_a, desc_b);
    append_pointer_wgmma_mma(&mut ctx, entry, accumulators[1], desc_a, desc_b);
    nvvm::WgmmaCommitGroupSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_wgmma_wait_group_constant(&mut ctx, entry, 0);
    append_return(&mut ctx, entry);

    assert_wgmma_lowering_rejected(
        &mut ctx,
        module_ptr,
        "WGMMA deferred accumulator region uses more than one accumulator",
    );
    Ok(())
}

#[test]
fn test_deferred_wgmma_mma_after_commit_is_rejected() -> Result<(), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let (module_ptr, entry, accumulators, desc_a, desc_b, _) =
        build_wgmma_pointer_test_kernel(&mut ctx, 1, vec![]);

    nvvm::WgmmaFenceSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_pointer_wgmma_mma(&mut ctx, entry, accumulators[0], desc_a, desc_b);
    nvvm::WgmmaCommitGroupSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_pointer_wgmma_mma(&mut ctx, entry, accumulators[0], desc_a, desc_b);
    append_wgmma_wait_group_constant(&mut ctx, entry, 0);
    append_return(&mut ctx, entry);

    assert_wgmma_lowering_rejected(
        &mut ctx,
        module_ptr,
        "WGMMA MMA cannot appear after commit_group in a deferred accumulator region",
    );
    Ok(())
}

#[test]
fn test_deferred_wgmma_branch_is_rejected() -> Result<(), anyhow::Error> {
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::op_interfaces::OperandSegmentInterface;
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let bool_ty = IntegerType::get(&ctx, 1, Signedness::Signless);
    let (module_ptr, entry, accumulators, desc_a, desc_b, trailing) =
        build_wgmma_pointer_test_kernel(&mut ctx, 1, vec![bool_ty.into()]);
    let condition = trailing[0];

    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    let function = module_block.deref(&ctx).iter(&ctx).next().unwrap();
    let function_region = function.deref(&ctx).get_region(0);

    let then_block = BasicBlock::new(&mut ctx, None, vec![]);
    then_block.insert_at_back(function_region, &ctx);
    let else_block = BasicBlock::new(&mut ctx, None, vec![]);
    else_block.insert_at_back(function_region, &ctx);

    nvvm::WgmmaFenceSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_pointer_wgmma_mma(&mut ctx, entry, accumulators[0], desc_a, desc_b);

    let (flat_operands, segment_sizes) =
        mir::MirCondBranchOp::compute_segment_sizes(vec![vec![condition], vec![], vec![]]);
    let branch = Operation::new(
        &mut ctx,
        mir::MirCondBranchOp::get_concrete_op_info(),
        vec![],
        flat_operands,
        vec![then_block, else_block],
        0,
    );
    Operation::get_op::<mir::MirCondBranchOp>(branch, &ctx)
        .expect("MirCondBranchOp")
        .set_operand_segment_sizes(&ctx, segment_sizes);
    branch.insert_at_back(entry, &ctx);

    append_return(&mut ctx, then_block);
    append_return(&mut ctx, else_block);

    assert_wgmma_lowering_rejected(
        &mut ctx,
        module_ptr,
        "unsupported operation inside WGMMA deferred accumulator region",
    );
    Ok(())
}

#[test]
fn test_deferred_wgmma_join_is_rejected() -> Result<(), anyhow::Error> {
    use pliron::basic_block::BasicBlock;

    let mut ctx = make_test_ctx();
    let (module_ptr, entry, accumulators, desc_a, desc_b, _) =
        build_wgmma_pointer_test_kernel(&mut ctx, 1, vec![]);

    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    let function = module_block.deref(&ctx).iter(&ctx).next().unwrap();
    let function_region = function.deref(&ctx).get_region(0);

    let second_predecessor = BasicBlock::new(&mut ctx, None, vec![]);
    second_predecessor.insert_at_back(function_region, &ctx);
    let join = BasicBlock::new(&mut ctx, None, vec![]);
    join.insert_at_back(function_region, &ctx);

    nvvm::WgmmaFenceSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_pointer_wgmma_mma(&mut ctx, entry, accumulators[0], desc_a, desc_b);

    Operation::new(
        &mut ctx,
        mir::MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![join],
        0,
    )
    .insert_at_back(entry, &ctx);

    Operation::new(
        &mut ctx,
        mir::MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![join],
        0,
    )
    .insert_at_back(second_predecessor, &ctx);

    nvvm::WgmmaCommitGroupSyncAlignedOp::build(&mut ctx).insert_at_back(join, &ctx);
    append_wgmma_wait_group_constant(&mut ctx, join, 0);
    append_return(&mut ctx, join);

    assert_wgmma_lowering_rejected(
        &mut ctx,
        module_ptr,
        "WGMMA deferred accumulator region cannot cross a control-flow join",
    );
    Ok(())
}

#[test]
fn test_deferred_wgmma_intervening_operation_is_rejected() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let (module_ptr, entry, accumulators, desc_a, desc_b, _) =
        build_wgmma_pointer_test_kernel(&mut ctx, 1, vec![]);

    nvvm::WgmmaFenceSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_pointer_wgmma_mma(&mut ctx, entry, accumulators[0], desc_a, desc_b);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    Operation::new(
        &mut ctx,
        nvvm::ReadPtxSregTidXOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    )
    .insert_at_back(entry, &ctx);

    nvvm::WgmmaCommitGroupSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_wgmma_wait_group_constant(&mut ctx, entry, 0);
    append_return(&mut ctx, entry);

    assert_wgmma_lowering_rejected(
        &mut ctx,
        module_ptr,
        "unsupported operation inside WGMMA deferred accumulator region",
    );
    Ok(())
}

#[test]
fn test_deferred_wgmma_nested_fence_is_rejected() -> Result<(), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let (module_ptr, entry, accumulators, desc_a, desc_b, _) =
        build_wgmma_pointer_test_kernel(&mut ctx, 1, vec![]);

    nvvm::WgmmaFenceSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_pointer_wgmma_mma(&mut ctx, entry, accumulators[0], desc_a, desc_b);
    nvvm::WgmmaFenceSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    nvvm::WgmmaCommitGroupSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_wgmma_wait_group_constant(&mut ctx, entry, 0);
    append_return(&mut ctx, entry);

    assert_wgmma_lowering_rejected(
        &mut ctx,
        module_ptr,
        "nested WGMMA fences are not supported in one deferred accumulator region",
    );
    Ok(())
}
