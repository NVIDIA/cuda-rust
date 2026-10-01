/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tensor Memory Accelerator (TMA) intrinsics.
//!
//! Handles asynchronous bulk data movement between global and shared memory.

use super::super::helpers::{emit_goto, set_generated_intrinsic_marker};
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_mir::ops::MirConstantOp;
use dialect_nvvm::ops::{
    CpAsyncBulkCommitGroupOp, CpAsyncBulkTensorG2sTile1dOp,
    CpAsyncBulkTensorG2sTile2dMulticastCg2Op, CpAsyncBulkTensorG2sTile2dMulticastOp,
    CpAsyncBulkTensorG2sTile2dOp, CpAsyncBulkTensorG2sTile3dOp, CpAsyncBulkTensorG2sTile4dOp,
    CpAsyncBulkTensorG2sTile5dOp, CpAsyncBulkTensorS2gTile1dOp, CpAsyncBulkTensorS2gTile2dOp,
    CpAsyncBulkTensorS2gTile3dOp, CpAsyncBulkTensorS2gTile4dOp, CpAsyncBulkTensorS2gTile5dOp,
    CpAsyncBulkWaitGroupOp, CpAsyncBulkWaitGroupReadOp,
};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;
/// Emits `cp_async_bulk_tensor_Nd_g2s`: Async tensor copy global → shared via TMA.
///
/// Initiates an asynchronous bulk tensor copy from global memory to shared memory
/// using the Tensor Memory Accelerator (TMA) hardware unit. The copy is tracked by
/// an mbarrier for completion notification.
///
/// # Arguments
///
/// - `args[0]`: `*mut u8` - Destination in shared memory
/// - `args[1]`: `*const TmaDescriptor` - Tensor map descriptor
/// - `args[2..2+dims]`: `i32` - Coordinates for each dimension
/// - `args[last]`: `*mut u64` - Barrier for completion tracking
///
/// # Supported Dimensions
///
/// 1D through 5D variants are supported (`dims` parameter).
///
/// # TMA Hardware
///
/// TMA is a dedicated hardware unit for efficient bulk data movement.
/// It handles address calculation, bounds checking, and cache management.
pub fn emit_tma_g2s(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    dims: usize,
    marker: &str,
) -> TranslationResult<Ptr<Operation>> {
    // Expected args: dst, tensor_map, coord0, [coord1, ...], barrier
    let expected_args = 3 + dims; // dst + tensor_map + coords + barrier
    if args.len() != expected_args {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "cp_async_bulk_tensor_{}d_g2s expects {} arguments, got {}",
                dims,
                expected_args,
                args.len()
            ))
        );
    }

    // Translate all arguments
    let mut operands = Vec::new();
    let mut last_op = prev_op;

    // arg[0]: dst (shared memory pointer)
    let (dst, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    operands.push(dst);
    last_op = last_op_after;

    // arg[last]: barrier (shared memory pointer) - we need this early for LLVM intrinsic order
    let barrier_idx = expected_args - 1;
    let (barrier, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[barrier_idx],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    operands.push(barrier);
    last_op = last_op_after;

    // arg[1]: tensor_map (generic pointer)
    let (tensor_map, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    operands.push(tensor_map);
    last_op = last_op_after;

    // args[2..2+dims]: coordinates
    for i in 0..dims {
        let (coord, last_op_after) = rvalue::translate_operand(
            ctx,
            body,
            &args[2 + i],
            value_map,
            block_ptr,
            last_op,
            loc.clone(),
        )?;
        operands.push(coord);
        last_op = last_op_after;
    }

    // Add default cta_mask (i16 = 0) and cache_hint (i64 = 0)
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let i16_type = IntegerType::get(ctx, 16, Signedness::Signed);
    let i64_type = IntegerType::get(ctx, 64, Signedness::Unsigned);

    // Create constant for cta_mask = 0
    let cta_mask_apint = APInt::from_i64(0, NonZeroUsize::new(16).unwrap());
    let cta_mask_attr = pliron::builtin::attributes::IntegerAttr::new(i16_type, cta_mask_apint);

    let cta_mask_raw_op = Operation::new(
        ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![i16_type.to_handle()],
        vec![],
        vec![],
        0,
    );
    cta_mask_raw_op.deref_mut(ctx).set_loc(loc.clone());
    let cta_mask_const = MirConstantOp::new(cta_mask_raw_op);
    cta_mask_const.set_attr_value(ctx, cta_mask_attr);

    if let Some(prev) = last_op {
        cta_mask_const.get_operation().insert_after(ctx, prev);
    } else {
        cta_mask_const
            .get_operation()
            .insert_at_front(block_ptr, ctx);
    }
    let cta_mask = cta_mask_const.get_operation().deref(ctx).get_result(0);
    operands.push(cta_mask);

    // Create constant for cache_hint = 0
    let cache_hint_apint = APInt::from_i64(0, NonZeroUsize::new(64).unwrap());
    let cache_hint_attr = pliron::builtin::attributes::IntegerAttr::new(i64_type, cache_hint_apint);

    let cache_hint_raw_op = Operation::new(
        ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![i64_type.to_handle()],
        vec![],
        vec![],
        0,
    );
    cache_hint_raw_op.deref_mut(ctx).set_loc(loc.clone());
    let cache_hint_const = MirConstantOp::new(cache_hint_raw_op);
    cache_hint_const.set_attr_value(ctx, cache_hint_attr);
    cache_hint_const
        .get_operation()
        .insert_after(ctx, cta_mask_const.get_operation());

    let cache_hint = cache_hint_const.get_operation().deref(ctx).get_result(0);
    operands.push(cache_hint);

    // Select the appropriate NVVM op based on dimensions
    let op_id = match dims {
        1 => CpAsyncBulkTensorG2sTile1dOp::get_concrete_op_info(),
        2 => CpAsyncBulkTensorG2sTile2dOp::get_concrete_op_info(),
        3 => CpAsyncBulkTensorG2sTile3dOp::get_concrete_op_info(),
        4 => CpAsyncBulkTensorG2sTile4dOp::get_concrete_op_info(),
        5 => CpAsyncBulkTensorG2sTile5dOp::get_concrete_op_info(),
        _ => {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "TMA G2S with {} dimensions not supported",
                    dims
                ))
            );
        }
    };

    // Create the TMA copy operation (void return)
    let tma_op = Operation::new(
        ctx,
        op_id,
        vec![],   // No results
        operands, // dst, barrier, tensor_map, coords..., cta_mask, cache_hint
        vec![],
        0,
    );
    tma_op.deref_mut(ctx).set_loc(loc.clone());
    set_generated_intrinsic_marker(ctx, tma_op, marker);
    tma_op.insert_after(ctx, cache_hint_const.get_operation());

    // Emit goto to target block
    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, tma_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("TMA G2S call without target block".to_string())
        )
    }
}

/// Emit cp_async_bulk_tensor_Nd_s2g: Async tensor copy shared → global via TMA.
///
/// Args for 2D:
/// - `args[0]`: *const u8 (source in shared memory)
/// - `args[1]`: *const TmaDescriptor (tensor map)
/// - `args[2]`: i32 (coord0)
/// - `args[3]`: i32 (coord1)
///
/// Returns: void
pub fn emit_tma_s2g(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    dims: usize,
    marker: &str,
) -> TranslationResult<Ptr<Operation>> {
    // Expected args: src, tensor_map, coords...
    let expected_args = 2 + dims;
    if args.len() != expected_args {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "cp_async_bulk_tensor_{}d_s2g expects {} arguments, got {}",
                dims,
                expected_args,
                args.len()
            ))
        );
    }

    // Translate all arguments
    let mut operands = Vec::new();
    let mut last_op = prev_op;

    // arg[0]: src (shared memory pointer)
    let (src, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    operands.push(src);
    last_op = last_op_after;

    // arg[1]: tensor_map (generic pointer)
    let (tensor_map, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    operands.push(tensor_map);
    last_op = last_op_after;

    // args[2..]: coordinates
    for i in 0..dims {
        let (coord, last_op_after) = rvalue::translate_operand(
            ctx,
            body,
            &args[2 + i],
            value_map,
            block_ptr,
            last_op,
            loc.clone(),
        )?;
        operands.push(coord);
        last_op = last_op_after;
    }

    // Select the appropriate NVVM op based on dimensions
    let op_id = match dims {
        1 => CpAsyncBulkTensorS2gTile1dOp::get_concrete_op_info(),
        2 => CpAsyncBulkTensorS2gTile2dOp::get_concrete_op_info(),
        3 => CpAsyncBulkTensorS2gTile3dOp::get_concrete_op_info(),
        4 => CpAsyncBulkTensorS2gTile4dOp::get_concrete_op_info(),
        5 => CpAsyncBulkTensorS2gTile5dOp::get_concrete_op_info(),
        _ => {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "TMA S2G with {} dimensions not supported",
                    dims
                ))
            );
        }
    };

    // Create the TMA copy operation (void return)
    let tma_op = Operation::new(
        ctx,
        op_id,
        vec![],   // No results
        operands, // src, tensor_map, coords...
        vec![],
        0,
    );
    tma_op.deref_mut(ctx).set_loc(loc.clone());
    set_generated_intrinsic_marker(ctx, tma_op, marker);

    if let Some(prev) = last_op {
        tma_op.insert_after(ctx, prev);
    } else {
        tma_op.insert_at_front(block_ptr, ctx);
    }

    // Emit goto to target block
    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, tma_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("TMA S2G call without target block".to_string())
        )
    }
}

/// Emits `cp_async_bulk_tensor_2d_g2s_multicast`: TMA 2D copy with multicast.
///
/// Same LLVM intrinsic as the non-multicast G2S, but the user supplies the
/// `cta_mask` (which CTAs in the cluster receive the tile) and the lowering
/// sets `use_cta_mask = true`.
///
/// # Arguments
///
/// - `args[0]`: `*mut u8` - Destination in shared memory
/// - `args[1]`: `*const TmaDescriptor` - Tensor map descriptor
/// - `args[2]`: `i32` - coord0
/// - `args[3]`: `i32` - coord1
/// - `args[4]`: `*mut u64` - Barrier
/// - `args[5]`: `u16` - CTA multicast mask
pub fn emit_tma_g2s_multicast(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    marker: &str,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 6 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "cp_async_bulk_tensor_2d_g2s_multicast expects 6 arguments, got {}",
                args.len()
            ))
        );
    }

    let mut operands = Vec::new();
    let mut last_op = prev_op;

    // arg[0]: dst (shared memory pointer)
    let (dst, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    operands.push(dst);
    last_op = last_op_after;

    // arg[4]: barrier — reordered to match LLVM intrinsic (dst, barrier, tensor_map, ...)
    let (barrier, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[4],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    operands.push(barrier);
    last_op = last_op_after;

    // arg[1]: tensor_map
    let (tensor_map, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    operands.push(tensor_map);
    last_op = last_op_after;

    // arg[2]: coord0
    let (coord0, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    operands.push(coord0);
    last_op = last_op_after;

    // arg[3]: coord1
    let (coord1, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[3],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    operands.push(coord1);
    last_op = last_op_after;

    // arg[5]: cta_mask (u16) — user-supplied, NOT defaulted to 0
    let (cta_mask, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[5],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    operands.push(cta_mask);
    last_op = last_op_after;

    // cache_hint = 0 (i64) — still defaulted
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let i64_type = IntegerType::get(ctx, 64, Signedness::Unsigned);
    let cache_hint_apint = APInt::from_i64(0, NonZeroUsize::new(64).unwrap());
    let cache_hint_attr = pliron::builtin::attributes::IntegerAttr::new(i64_type, cache_hint_apint);

    let cache_hint_raw_op = Operation::new(
        ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![i64_type.to_handle()],
        vec![],
        vec![],
        0,
    );
    cache_hint_raw_op.deref_mut(ctx).set_loc(loc.clone());
    let cache_hint_const = MirConstantOp::new(cache_hint_raw_op);
    cache_hint_const.set_attr_value(ctx, cache_hint_attr);

    if let Some(prev) = last_op {
        cache_hint_const.get_operation().insert_after(ctx, prev);
    } else {
        cache_hint_const
            .get_operation()
            .insert_at_front(block_ptr, ctx);
    }

    let cache_hint = cache_hint_const.get_operation().deref(ctx).get_result(0);
    operands.push(cache_hint);

    // Create the multicast TMA op
    let tma_op = Operation::new(
        ctx,
        CpAsyncBulkTensorG2sTile2dMulticastOp::get_concrete_op_info(),
        vec![],
        operands,
        vec![],
        0,
    );
    tma_op.deref_mut(ctx).set_loc(loc.clone());
    set_generated_intrinsic_marker(ctx, tma_op, marker);
    tma_op.insert_after(ctx, cache_hint_const.get_operation());

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, tma_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("TMA G2S multicast call without target block".to_string(),)
        )
    }
}

/// Emits `cp_async_bulk_tensor_2d_g2s_multicast_cg2`: TMA 2D copy with
/// multicast and `cta_group::2` (TPC pair).
///
/// Same operand layout as [`emit_tma_g2s_multicast`] but creates a
/// `CpAsyncBulkTensorG2sTile2dMulticastCg2Op` which is lowered to inline PTX
/// with the `cta_group::2` qualifier.
///
/// # Arguments
///
/// - `args[0]`: `*mut u8` - Destination in shared memory
/// - `args[1]`: `*const TmaDescriptor` - Tensor map descriptor
/// - `args[2]`: `i32` - coord0
/// - `args[3]`: `i32` - coord1
/// - `args[4]`: `*mut u64` - Barrier
/// - `args[5]`: `u16` - CTA multicast mask
pub fn emit_tma_g2s_multicast_cg2(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    marker: &str,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 6 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "cp_async_bulk_tensor_2d_g2s_multicast_cg2 expects 6 arguments, got {}",
                args.len()
            ))
        );
    }

    let mut operands = Vec::new();
    let mut last_op = prev_op;

    // arg[0]: dst (shared memory pointer)
    let (dst, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    operands.push(dst);
    last_op = last_op_after;

    // arg[4]: barrier — reordered to match LLVM intrinsic pattern (dst, barrier, tensor_map, ...)
    let (barrier, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[4],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    operands.push(barrier);
    last_op = last_op_after;

    // arg[1]: tensor_map
    let (tensor_map, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    operands.push(tensor_map);
    last_op = last_op_after;

    // arg[2]: coord0
    let (coord0, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    operands.push(coord0);
    last_op = last_op_after;

    // arg[3]: coord1
    let (coord1, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[3],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    operands.push(coord1);
    last_op = last_op_after;

    // arg[5]: cta_mask (u16)
    let (cta_mask, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[5],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    operands.push(cta_mask);
    last_op = last_op_after;

    // cache_hint = 0 (i64)
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let i64_type = IntegerType::get(ctx, 64, Signedness::Unsigned);
    let cache_hint_apint = APInt::from_i64(0, NonZeroUsize::new(64).unwrap());
    let cache_hint_attr = pliron::builtin::attributes::IntegerAttr::new(i64_type, cache_hint_apint);

    let cache_hint_raw_op = Operation::new(
        ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![i64_type.to_handle()],
        vec![],
        vec![],
        0,
    );
    cache_hint_raw_op.deref_mut(ctx).set_loc(loc.clone());
    let cache_hint_const = MirConstantOp::new(cache_hint_raw_op);
    cache_hint_const.set_attr_value(ctx, cache_hint_attr);

    if let Some(prev) = last_op {
        cache_hint_const.get_operation().insert_after(ctx, prev);
    } else {
        cache_hint_const
            .get_operation()
            .insert_at_front(block_ptr, ctx);
    }

    let cache_hint = cache_hint_const.get_operation().deref(ctx).get_result(0);
    operands.push(cache_hint);

    // Create the cg2 multicast TMA op
    let tma_op = Operation::new(
        ctx,
        CpAsyncBulkTensorG2sTile2dMulticastCg2Op::get_concrete_op_info(),
        vec![],
        operands,
        vec![],
        0,
    );
    tma_op.deref_mut(ctx).set_loc(loc.clone());
    set_generated_intrinsic_marker(ctx, tma_op, marker);
    tma_op.insert_after(ctx, cache_hint_const.get_operation());

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, tma_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "TMA G2S multicast cg2 call without target block".to_string(),
            )
        )
    }
}

/// Emit cp_async_bulk_commit_group: Commit pending async bulk operations.
///
/// Args: none
/// Returns: void
pub fn emit_tma_commit_group(
    ctx: &mut Context,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    marker: &str,
) -> TranslationResult<Ptr<Operation>> {
    if !args.is_empty() {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "cp_async_bulk_commit_group expects 0 arguments, got {}",
                args.len()
            ))
        );
    }

    // Create the commit_group operation (void return, no operands)
    let commit_op = Operation::new(
        ctx,
        CpAsyncBulkCommitGroupOp::get_concrete_op_info(),
        vec![], // No results
        vec![], // No operands
        vec![],
        0,
    );
    commit_op.deref_mut(ctx).set_loc(loc.clone());
    set_generated_intrinsic_marker(ctx, commit_op, marker);

    if let Some(prev) = prev_op {
        commit_op.insert_after(ctx, prev);
    } else {
        commit_op.insert_at_front(block_ptr, ctx);
    }

    // Emit goto to target block
    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, commit_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("TMA commit_group call without target block".to_string())
        )
    }
}

/// Emit cp_async_bulk_wait_group: Wait for async bulk operation groups.
///
/// Args:
/// - `args[0]`: u32 (max pending groups, 0 = wait for all)
///
/// Returns: void
pub fn emit_tma_wait_group(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    read_variant: bool,
    marker: &str,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 1 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "cp_async_bulk_wait_group expects 1 argument, got {}",
                args.len()
            ))
        );
    }

    // Get the count argument
    let (count, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // Select the appropriate op
    let op_info = if read_variant {
        CpAsyncBulkWaitGroupReadOp::get_concrete_op_info()
    } else {
        CpAsyncBulkWaitGroupOp::get_concrete_op_info()
    };

    // Create the wait_group operation (void return)
    let wait_op = Operation::new(
        ctx,
        op_info,
        vec![],      // No results
        vec![count], // Operand: count
        vec![],
        0,
    );
    wait_op.deref_mut(ctx).set_loc(loc.clone());
    set_generated_intrinsic_marker(ctx, wait_op, marker);

    if let Some(prev) = last_op {
        wait_op.insert_after(ctx, prev);
    } else {
        wait_op.insert_at_front(block_ptr, ctx);
    }

    // Emit goto to target block
    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, wait_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("TMA wait_group call without target block".to_string())
        )
    }
}
