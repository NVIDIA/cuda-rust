/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Hopper WGMMA (Warpgroup Matrix Multiply-Accumulate) intrinsics.
//!
//! Handles Hopper `sm_90a` asynchronous warpgroup matrix operations.

use super::super::helpers::{self, emit_goto};
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::{
    WgmmaMakeSmemDescOp, WgmmaMmaM64N64K8F32Tf32Op, WgmmaMmaM64N64K16F32Bf16Op,
    WgmmaMmaM64N64K16F32F16Op, WgmmaMmaM64N128K16F32Bf16Op,
};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::value::Value;
use rustc_public::mir;

const CUSTOM_DESCRIPTOR_UNSUPPORTED: &str = "custom WGMMA descriptor encoding is not yet supported";
const MMA_UNSUPPORTED: &str =
    "this WGMMA MMA variant is not yet supported by the importer and lowering pipeline";

fn unsupported_diagnostic(path: &str) -> Option<&'static str> {
    match path {
        "cuda_device::wgmma::make_smem_desc_custom" => Some(CUSTOM_DESCRIPTOR_UNSUPPORTED),
        "cuda_device::wgmma::wgmma_mma_m64n64k16_f32_tf32" => Some(MMA_UNSUPPORTED),
        _ => None,
    }
}

/// Reject public WGMMA entries that do not have a sound lowering yet.
pub(crate) fn reject_unsupported(path: &str, loc: Location) -> TranslationResult<()> {
    let Some(diagnostic) = unsupported_diagnostic(path) else {
        return Ok(());
    };
    input_err!(loc, TranslationErr::unsupported(diagnostic))
}

/// Emit make_smem_desc: Create SMEM descriptor for WGMMA.
pub fn emit_wgmma_make_smem_desc(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 1 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "make_smem_desc expects 1 argument, got {}",
                args.len()
            ))
        );
    }

    let (ptr_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let (prepared_destination, last_op) = helpers::prepare_destination_write(
        ctx,
        body,
        destination,
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    let u64_ty = IntegerType::get(ctx, 64, Signedness::Unsigned);
    let desc_op = Operation::new(
        ctx,
        WgmmaMakeSmemDescOp::get_concrete_op_info(),
        vec![u64_ty.into()],
        vec![ptr_val],
        vec![],
        0,
    );
    desc_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        desc_op.insert_after(ctx, prev);
    } else {
        desc_op.insert_at_front(block_ptr, ctx);
    }

    let result_value = desc_op.deref(ctx).get_result(0);
    helpers::emit_prepared_result_and_goto(
        ctx,
        prepared_destination,
        result_value,
        target,
        block_ptr,
        desc_op,
        value_map,
        block_map,
        loc,
        "make_smem_desc call without target block",
    )
}

#[allow(clippy::too_many_arguments)]
fn emit_wgmma_mma_pointer_form<F>(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    build_op: F,
    intrinsic_name: &'static str,
) -> TranslationResult<Ptr<Operation>>
where
    F: FnOnce(&mut Context, Vec<Value>) -> Ptr<Operation>,
{
    if args.len() != 3 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "{intrinsic_name} expects 3 arguments (acc_ptr, desc_a, desc_b), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;
    let (acc_ptr, next) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = next;
    let (desc_a, next) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = next;
    let (desc_b, next) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = next;

    let mma_op = build_op(ctx, vec![acc_ptr, desc_a, desc_b]);
    mma_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        mma_op.insert_after(ctx, prev);
    } else {
        mma_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        Ok(emit_goto(ctx, *target_idx, mma_op, block_map, loc))
    } else {
        input_err!(
            loc,
            TranslationErr::unsupported(format!("{intrinsic_name} call without target block"))
        )
    }
}

/// Emit BF16 m64n64k16 WGMMA pointer form.
///
/// `mir-lower` later selects a proven-safe enclosing region: a linear full
/// drain, a static partial-wait pipeline, or the canonical counted K-loop.
/// The selected region is fused so LLVM cannot observe an accumulator while a
/// WGMMA group that references it is still pending.
#[allow(clippy::too_many_arguments)]
pub fn emit_wgmma_mma_m64n64k16_f32_bf16(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    emit_wgmma_mma_pointer_form(
        ctx,
        body,
        args,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
        |ctx, operands| {
            Operation::new(
                ctx,
                WgmmaMmaM64N64K16F32Bf16Op::get_concrete_op_info(),
                vec![],
                operands,
                vec![],
                0,
            )
        },
        "wgmma_mma_m64n64k16_f32_bf16",
    )
}

/// Emit BF16 m64n128k16 WGMMA pointer form.
///
/// `mir-lower` accepts this variant only in a canonical linear full-drain
/// region with a `[[f32; 8]; 8]` accumulator and a final `wait_group<0>`.
#[allow(clippy::too_many_arguments)]
pub fn emit_wgmma_mma_m64n128k16_f32_bf16(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    emit_wgmma_mma_pointer_form(
        ctx,
        body,
        args,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
        |ctx, operands| {
            Operation::new(
                ctx,
                WgmmaMmaM64N128K16F32Bf16Op::get_concrete_op_info(),
                vec![],
                operands,
                vec![],
                0,
            )
        },
        "wgmma_mma_m64n128k16_f32_bf16",
    )
}

/// Emit F16 m64n64k16 WGMMA pointer form.
///
/// `mir-lower` accepts this variant only in a canonical linear full-drain
/// region ending in `wait_group<0>`. Counted loops and partial-wait pipelines
/// remain BF16-only.
#[allow(clippy::too_many_arguments)]
pub fn emit_wgmma_mma_m64n64k16_f32_f16(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    emit_wgmma_mma_pointer_form(
        ctx,
        body,
        args,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
        |ctx, operands| {
            Operation::new(
                ctx,
                WgmmaMmaM64N64K16F32F16Op::get_concrete_op_info(),
                vec![],
                operands,
                vec![],
                0,
            )
        },
        "wgmma_mma_m64n64k16_f32_f16",
    )
}

/// Emit TF32 m64n64k8 WGMMA pointer form.
///
/// `mir-lower` accepts this variant only in a canonical linear full-drain
/// region ending in `wait_group<0>`. Counted loops, partial-wait pipelines,
/// and pointer fallback remain BF16-only.
pub fn emit_wgmma_mma_m64n64k8_f32_tf32(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 3 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "wgmma_mma_m64n64k8_f32_tf32 expects 3 arguments (acc_ptr, desc_a, desc_b), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;
    let (acc_ptr, next) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = next;
    let (desc_a, next) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = next;
    let (desc_b, next) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = next;

    let mma_op = Operation::new(
        ctx,
        WgmmaMmaM64N64K8F32Tf32Op::get_concrete_op_info(),
        vec![],
        vec![acc_ptr, desc_a, desc_b],
        vec![],
        0,
    );
    mma_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        mma_op.insert_after(ctx, prev);
    } else {
        mma_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        Ok(emit_goto(ctx, *target_idx, mma_op, block_map, loc))
    } else {
        input_err!(
            loc,
            TranslationErr::unsupported(
                "wgmma_mma_m64n64k8_f32_tf32 call without target block".to_string()
            )
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{CUSTOM_DESCRIPTOR_UNSUPPORTED, MMA_UNSUPPORTED, unsupported_diagnostic};

    #[test]
    fn unsupported_wgmma_paths_are_exact() {
        assert_eq!(
            unsupported_diagnostic("cuda_device::wgmma::make_smem_desc_custom"),
            Some(CUSTOM_DESCRIPTOR_UNSUPPORTED)
        );
        for path in [
            "cuda_device::wgmma::wgmma_mma_m64n64k16_f32_bf16",
            "cuda_device::wgmma::wgmma_mma_m64n128k16_f32_bf16",
            "cuda_device::wgmma::wgmma_mma_m64n64k16_f32_f16",
            "cuda_device::wgmma::wgmma_mma_m64n64k8_f32_tf32",
        ] {
            assert_eq!(unsupported_diagnostic(path), None);
        }
        assert_eq!(
            unsupported_diagnostic("cuda_device::wgmma::wgmma_mma_m64n64k16_f32_tf32"),
            Some(MMA_UNSUPPORTED)
        );

        for path in [
            "cuda_device::wgmma::make_smem_desc",
            "cuda_device::wgmma::wgmma_fence",
            "cuda_device::wgmma::wgmma_mma_m64n64k16_f32_bf16_extra",
            "other_crate::wgmma::wgmma_mma_m64n64k16_f32_bf16",
        ] {
            assert_eq!(unsupported_diagnostic(path), None);
        }
    }
}
