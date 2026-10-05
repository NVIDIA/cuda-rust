/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level intrinsic conversion: shuffle and vote operations.
//!
//! # Shuffle Operations
//!
//! | Operation          | LLVM Intrinsic                 | Description       |
//! |--------------------|--------------------------------|-------------------|
//! | `ShflSyncIdxI32`   | `llvm.nvvm.shfl.sync.idx.i32`  | Indexed shuffle   |
//! | `ShflSyncBflyI32`  | `llvm.nvvm.shfl.sync.bfly.i32` | Butterfly shuffle |
//! | `ShflSyncDownI32`  | `llvm.nvvm.shfl.sync.down.i32` | Down shuffle      |
//! | `ShflSyncUpI32`    | `llvm.nvvm.shfl.sync.up.i32`   | Up shuffle        |
//!
//! # Vote Operations
//!
//! | Operation        | LLVM Intrinsic               | Description           |
//! |------------------|------------------------------|-----------------------|
//! | `VoteSyncAll`    | `llvm.nvvm.vote.all.sync`    | All lanes true        |
//! | `VoteSyncAny`    | `llvm.nvvm.vote.any.sync`    | Any lane true         |
//! | `VoteSyncBallot` | `llvm.nvvm.vote.ballot.sync` | Bitmask of predicates |
//!
//! # Match Operations (sm_70+)
//!
//! | Operation         | LLVM Intrinsic                    | Description                  |
//! |-------------------|-----------------------------------|------------------------------|
//! | `MatchAnySyncI32` | `llvm.nvvm.match.any.sync.i32`    | Mask of equal-value lanes    |
//! | `MatchAnySyncI64` | `llvm.nvvm.match.any.sync.i64`    | 64-bit variant               |
//! | `MatchAllSyncI32` | `llvm.nvvm.match.all.sync.i32p`   | Full mask iff all agree      |
//! | `MatchAllSyncI64` | `llvm.nvvm.match.all.sync.i64p`   | 64-bit variant               |

use crate::convert::intrinsics::common::*;
use crate::convert::types::convert_type;
use llvm_export::types as llvm_types;
use pliron::builtin::types::{FP32Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::Typed;

/// Convert i32 shuffle operation to LLVM intrinsic call.
///
/// Operand layout: `[mask, value, lane_or_delta]`. The mask reaches us
/// already type-converted by the framework (any `u32`/`i32` carrier
/// works); we forward it straight to the intrinsic. For full-warp ops
/// the mask is just `0xFFFFFFFF` baked in by the caller.
pub(crate) fn convert_shuffle_i32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    intrinsic_name: &str,
    clamp: i32,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 3 {
        return pliron::input_err_noloc!(
            "Warp shuffle i32 requires 3 operands [mask, value, lane_or_delta]"
        );
    }
    let (mask, val, lane_or_delta) = (operands[0], operands[1], operands[2]);

    let clamp_val = create_i32_const(ctx, rewriter, clamp);

    let func_ty = llvm_types::FuncType::get(
        ctx,
        i32_ty.into(),
        vec![i32_ty.into(), i32_ty.into(), i32_ty.into(), i32_ty.into()],
        false,
    );

    let call_op = call_intrinsic(
        ctx,
        rewriter,
        op,
        intrinsic_name,
        func_ty,
        vec![mask, val, lane_or_delta, clamp_val],
    )?;
    rewriter.replace_operation(ctx, op, call_op);
    Ok(())
}

/// Convert f32 shuffle operation to LLVM intrinsic call.
///
/// Operand layout: `[mask, value, lane_or_delta]`. See `convert_shuffle_i32`
/// for the mask forwarding rationale.
pub(crate) fn convert_shuffle_f32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    intrinsic_name: &str,
    clamp: i32,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let f32_ty = FP32Type::get(ctx);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 3 {
        return pliron::input_err_noloc!(
            "Warp shuffle f32 requires 3 operands [mask, value, lane_or_delta]"
        );
    }
    let (mask, val, lane_or_delta) = (operands[0], operands[1], operands[2]);

    let clamp_val = create_i32_const(ctx, rewriter, clamp);

    let func_ty = llvm_types::FuncType::get(
        ctx,
        f32_ty.into(),
        vec![i32_ty.into(), f32_ty.into(), i32_ty.into(), i32_ty.into()],
        false,
    );

    let call_op = call_intrinsic(
        ctx,
        rewriter,
        op,
        intrinsic_name,
        func_ty,
        vec![mask, val, lane_or_delta, clamp_val],
    )?;
    rewriter.replace_operation(ctx, op, call_op);
    Ok(())
}

/// Convert a 64-bit shuffle op to convergent inline PTX.
///
/// PTX `shfl.sync` only moves 32-bit registers (no `.b64` form, no
/// `@llvm.nvvm.shfl.sync.*.i64` intrinsic), so a 64-bit shuffle is two 32-bit
/// shuffles. We emit a single inline-PTX block that unpacks the value into
/// `{lo, hi}` halves with `mov.b64`, runs `shfl.sync.<mode>.b32` on each half
/// with the shared lane and membermask operands, then repacks the result.
/// Keeping both halves in one compiler-visible convergent block prevents code
/// motion between them. Hardware still executes two sequential `b32` shuffles.
///
/// The shfl `c` (clamp/segmentation) operand is baked into the template per
/// mode: `31` for idx/bfly/down and `0` for up — exactly the value the 32-bit
/// intrinsic path passes (see [`convert_shuffle_i32`]).
///
/// Operand layout: `[mask, value, lane_or_delta]`. Inline-asm operand order is
/// `$0`=result, `$1`=value (i64, `l`), `$2`=lane/delta (i32, `r`),
/// `$3`=membermask (i32, `r`).
pub(crate) fn convert_shuffle_i64(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    mode: &str,
    clamp: i32,
) -> Result<()> {
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 3 {
        return pliron::input_err_noloc!(
            "Warp shuffle i64 requires 3 operands [mask, value, lane_or_delta]"
        );
    }
    let (mask, val, lane_or_delta) = (operands[0], operands[1], operands[2]);

    let asm_template = format!(
        "{{ .reg .b32 lo; .reg .b32 hi; mov.b64 {{lo, hi}}, $1; \
         shfl.sync.{mode}.b32 lo, lo, $2, {clamp}, $3; \
         shfl.sync.{mode}.b32 hi, hi, $2, {clamp}, $3; \
         mov.b64 $0, {{lo, hi}}; }}"
    );
    let asm_op = inline_asm_convergent(
        ctx,
        rewriter,
        op,
        i64_ty.into(),
        vec![val, lane_or_delta, mask],
        &asm_template,
        "=l,l,r,r",
    );
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}

/// Convert vote operation to LLVM intrinsic call.
///
/// Operand layout: `[mask, predicate]`. See `convert_shuffle_i32` for
/// the mask forwarding rationale.
pub(crate) fn convert_vote(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    intrinsic_name: &str,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!("Warp vote requires 2 operands [mask, predicate]");
    }
    let (mask, predicate) = (operands[0], operands[1]);

    let result_ty: pliron::r#type::TypeHandle = if intrinsic_name.contains("ballot") {
        i32_ty.into()
    } else {
        i1_ty.into()
    };

    let func_ty =
        llvm_types::FuncType::get(ctx, result_ty, vec![i32_ty.into(), i1_ty.into()], false);
    let call_op = call_intrinsic(
        ctx,
        rewriter,
        op,
        intrinsic_name,
        func_ty,
        vec![mask, predicate],
    )?;
    rewriter.replace_operation(ctx, op, call_op);
    Ok(())
}

/// Convert a `match.any.sync` op to its LLVM intrinsic call.
///
/// Operand layout: `[mask, value]`. Result is i32 (bitmask of equal-value lanes).
/// The `value_ty` is i32 or i64 to pick `@llvm.nvvm.match.any.sync.{i32,i64}`.
pub(crate) fn convert_match_any(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    intrinsic_name: &str,
    value_ty: pliron::r#type::TypeHandle,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!("match.any.sync requires 2 operands [mask, value]");
    }
    let (mask, value) = (operands[0], operands[1]);

    let func_ty =
        llvm_types::FuncType::get(ctx, i32_ty.into(), vec![i32_ty.into(), value_ty], false);

    let call_op = call_intrinsic(
        ctx,
        rewriter,
        op,
        intrinsic_name,
        func_ty,
        vec![mask, value],
    )?;
    rewriter.replace_operation(ctx, op, call_op);
    Ok(())
}

/// Convert a `redux.sync.add` op to its LLVM intrinsic call.
///
/// Op operand layout is `[mask, value]` (matching the other `*_sync`
/// collectives), but the LLVM intrinsic signature is `(src, membermask)`, so
/// we forward the operands flipped as `[value, mask]`. The value and result
/// types are carried by the dialect record.
///
/// The result type is converted to its LLVM form: the op's `ui32`/`si32`
/// result would otherwise declare the intrinsic as returning a signed or
/// unsigned integer while its operands are already signless, and a device
/// function returning the reduction directly fails verification (#811).
pub(crate) fn convert_redux(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    intrinsic_name: &str,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!("redux requires 2 operands [mask, value]");
    }
    let (mask, value) = (operands[0], operands[1]);

    let value_ty = value.get_type(ctx);
    let mir_result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let result_ty =
        convert_type(ctx, mir_result_ty).map_err(|e| pliron::input_error_noloc!("{}", e))?;
    let func_ty = llvm_types::FuncType::get(ctx, result_ty, vec![value_ty, i32_ty.into()], false);

    // LLVM intrinsic wants (src, membermask): flip to [value, mask].
    let call_op = call_intrinsic(
        ctx,
        rewriter,
        op,
        intrinsic_name,
        func_ty,
        vec![value, mask],
    )?;
    rewriter.replace_operation(ctx, op, call_op);
    Ok(())
}

/// Convert an `activemask` op to its LLVM intrinsic call.
///
/// Lowers to `call i32 @llvm.nvvm.activemask()`. The op has no operands.
pub(crate) fn convert_active_mask(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let func_ty = llvm_types::FuncType::get(ctx, i32_ty.into(), vec![], false);

    let call_op = call_intrinsic(ctx, rewriter, op, "llvm_nvvm_activemask", func_ty, vec![])?;
    rewriter.replace_operation(ctx, op, call_op);
    Ok(())
}

/// Convert a `bar.warp.sync` op to its LLVM intrinsic call.
///
/// Lowers to `call void @llvm.nvvm.bar.warp.sync(i32 mask)`. The op has one
/// operand (the participation mask) and no result.
pub(crate) fn convert_bar_warp_sync(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let void_ty = llvm_types::VoidType::get(ctx);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 {
        return pliron::input_err_noloc!("bar.warp.sync requires 1 operand [mask]");
    }
    let mask = operands[0];

    let func_ty = llvm_types::FuncType::get(ctx, void_ty.into(), vec![i32_ty.into()], false);
    call_intrinsic(
        ctx,
        rewriter,
        op,
        "llvm_nvvm_bar_warp_sync",
        func_ty,
        vec![mask],
    )?;
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert a `match.all.sync` op to its LLVM intrinsic call.
///
/// The LLVM intrinsic signature is `{i32, i1} @llvm.nvvm.match.all.sync.*p(i32 mask, T value)`:
/// field 0 is the matching mask, field 1 is the all-match predicate. We expose
/// only the mask (callers can recover the predicate as `result != 0`); the
/// extracted i1 is dead and gets removed by LLVM DCE.
pub(crate) fn convert_match_all(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    intrinsic_name: &str,
    value_ty: pliron::r#type::TypeHandle,
) -> Result<()> {
    use llvm_export::ops::ExtractValueOp;

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!("match.all.sync requires 2 operands [mask, value]");
    }
    let (mask, value) = (operands[0], operands[1]);

    let struct_ty = llvm_types::StructType::get_unnamed(
        ctx,
        (
            vec![i32_ty.into(), i1_ty.into()],
            llvm_types::StructLayout::Unpacked,
        ),
    );
    let func_ty =
        llvm_types::FuncType::get(ctx, struct_ty.into(), vec![i32_ty.into(), value_ty], false);

    let call_op = call_intrinsic(
        ctx,
        rewriter,
        op,
        intrinsic_name,
        func_ty,
        vec![mask, value],
    )?;
    let struct_result = call_op.deref(ctx).get_result(0);

    let extract_op = ExtractValueOp::new(ctx, struct_result, vec![0])
        .map_err(|e| pliron::input_error_noloc!("match.all.sync extractvalue: {}", e))?;
    rewriter.insert_operation(ctx, extract_op.get_operation());
    let mask_result = extract_op.get_operation().deref(ctx).get_result(0);

    rewriter.replace_operation_with_values(ctx, op, vec![mask_result]);
    Ok(())
}

/// Convert `elect.sync` through LLVM's typed NVVM intrinsic.
pub(crate) fn convert_elect_sync_typed(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    intrinsic_name: &str,
) -> Result<()> {
    use llvm_export::ops::ExtractValueOp;

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 {
        return pliron::input_err_noloc!("elect.sync requires 1 operand [mask]");
    }

    let struct_ty = llvm_types::StructType::get_unnamed(
        ctx,
        (
            vec![i32_ty.into(), i1_ty.into()],
            llvm_types::StructLayout::Unpacked,
        ),
    );
    let func_ty = llvm_types::FuncType::get(ctx, struct_ty.into(), vec![i32_ty.into()], false);
    let call = call_intrinsic(
        ctx,
        rewriter,
        op,
        intrinsic_name,
        func_ty,
        vec![operands[0]],
    )?;
    let result = call.deref(ctx).get_result(0);
    let leader = ExtractValueOp::new(ctx, result, vec![0])
        .map_err(|error| pliron::input_error_noloc!("elect.sync extractvalue: {}", error))?;
    rewriter.insert_operation(ctx, leader.get_operation());
    let elected = ExtractValueOp::new(ctx, result, vec![1])
        .map_err(|error| pliron::input_error_noloc!("elect.sync extractvalue: {}", error))?;
    rewriter.insert_operation(ctx, elected.get_operation());
    let leader = leader.get_operation().deref(ctx).get_result(0);
    let elected = elected.get_operation().deref(ctx).get_result(0);
    rewriter.replace_operation_with_values(ctx, op, vec![leader, elected]);
    Ok(())
}

/// Convert `elect.sync` to convergent inline PTX.
///
/// PTX `elect.sync d|p, membermask;` writes the leader lane id into `d` and the
/// per-lane "I am the leader" predicate into `p`. Inline asm can't yield a
/// `.pred` directly, so we `selp.b32` it into a 0/1 register and truncate to i1.
/// The op has two results — leader (i32) and is_elected (i1) — bound to the two
/// asm outputs. The single operand (the membermask) is the asm input; either
/// result may be unused at the call site and is then removed by LLVM DCE.
pub(crate) fn convert_elect_sync_inline(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    use llvm_export::ops::ExtractValueOp;

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 {
        return pliron::input_err_noloc!("elect.sync requires 1 operand [mask]");
    }
    let mask = operands[0];

    // Two register outputs: $0 = leader lane id, $1 = predicate materialized as
    // 0/1; $2 = membermask input. The `.pred p` is scoped to the asm block.
    let asm_template = "{ .reg .pred p; elect.sync $0|p, $2; selp.b32 $1, 1, 0, p; }";
    let struct_ty = llvm_types::StructType::get_unnamed(
        ctx,
        (
            vec![i32_ty.into(), i32_ty.into()],
            llvm_types::StructLayout::Unpacked,
        ),
    );
    let asm_op = inline_asm_convergent(
        ctx,
        rewriter,
        op,
        struct_ty.into(),
        vec![mask],
        asm_template,
        "=r,=r,r",
    );
    let struct_result = asm_op.deref(ctx).get_result(0);

    // Field 0 → leader lane id (result 0). Field 1 → predicate as 0/1 i32,
    // truncated to the i1 is_elected result (result 1).
    let leader = {
        let extract_op = ExtractValueOp::new(ctx, struct_result, vec![0])
            .map_err(|e| pliron::input_error_noloc!("elect.sync extractvalue: {}", e))?;
        rewriter.insert_operation(ctx, extract_op.get_operation());
        extract_op.get_operation().deref(ctx).get_result(0)
    };
    let elected_i32 = {
        let extract_op = ExtractValueOp::new(ctx, struct_result, vec![1])
            .map_err(|e| pliron::input_error_noloc!("elect.sync extractvalue: {}", e))?;
        rewriter.insert_operation(ctx, extract_op.get_operation());
        extract_op.get_operation().deref(ctx).get_result(0)
    };
    let is_elected = trunc_to_i1(ctx, rewriter, elected_i32);

    rewriter.replace_operation_with_values(ctx, op, vec![leader, is_elected]);
    Ok(())
}
