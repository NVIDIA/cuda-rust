/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Emit the semantic SM120 output-tile protocol.
//!
//! The operations annotate ordinary carriers. The full tile is one pointer,
//! a warp slice is that pointer plus two indices, and both the accumulator and
//! TMA-half structs keep their existing MIR representation.

use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue::translate_operand;
use crate::translator::values::ValueMap;
use dialect_cute::attributes::{
    CuteCountedCtaBarrierAttr, CuteEpilogueHalfAttr, CuteEpilogueSyncPhaseAttr,
    CuteTmaStorePipelineAttr,
};
use dialect_cute::epilogue_ops::{
    CuteEpilogueHalfOp, CuteEpilogueSmemOverlayOp, CuteEpilogueStoreFragmentOp, CuteEpilogueSyncOp,
    CuteEpilogueWarpSliceOp, CuteTmaStoreAcquireOp, CuteTmaStoreCommitOp, CuteTmaStoreTailOp,
};
use dialect_mir::types::MirFP16Type;
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::TypeHandle;
use pliron::value::Value;
use rustc_public::mir;

use super::emit::recover_cta_shared_pointer;
use super::epilogue::{
    is_accumulator, is_epilogue_tile, is_epilogue_tile_receiver, is_epilogue_warp_slice_receiver,
    one_unsigned_const, tile_type,
};
use super::smem_mma_emit::{
    aggregate_field, build_struct_prefix, destination_rust_type, destination_type, finish,
    finish_value, insert, operand_rust_type, pointer_cast, struct_field_type,
};

fn invalid(what: &str, error: impl core::fmt::Display) -> pliron::result::Error {
    pliron::input_error_noloc!(TranslationErr::unsupported(format!(
        "invalid {what}: {error}"
    )))
}

fn require_type_fact(result: Result<bool, String>, what: &str) -> TranslationResult<()> {
    match result {
        Ok(true) => Ok(()),
        Ok(false) => Err(pliron::input_error_noloc!(TranslationErr::unsupported(
            format!("{what} has the wrong Rust carrier type")
        ))),
        Err(error) => Err(invalid(what, error)),
    }
}

fn translated_field_type(
    ctx: &mut Context,
    body: &mir::Body,
    destination: &mir::Place,
    field: usize,
    what: &str,
) -> TranslationResult<(TypeHandle, TypeHandle)> {
    let outer = destination_type(ctx, body, destination, what)?;
    let field = struct_field_type(ctx, outer, field, what)?;
    Ok((outer, field))
}

#[allow(clippy::too_many_arguments)]
fn epilogue_base(
    ctx: &mut Context,
    body: &mir::Body,
    operand: &mir::Operand,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    loc: &Location,
    what: &str,
) -> TranslationResult<(Value, Ptr<Operation>)> {
    require_type_fact(
        is_epilogue_tile_receiver(&operand_rust_type(body, operand, what)?),
        what,
    )?;
    let (tile, after) = translate_operand(ctx, body, operand, value_map, block, prev, loc.clone())?;
    let (base, after) = aggregate_field(ctx, tile, 0, what, block, after.or(prev), loc)?;
    let storage: TypeHandle = MirFP16Type::get(ctx).into();
    recover_cta_shared_pointer(ctx, base, storage, what, block, after, loc)
}

#[allow(clippy::too_many_arguments)]
fn warp_slice_carriers(
    ctx: &mut Context,
    body: &mir::Body,
    operand: &mir::Operand,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    loc: &Location,
    what: &str,
) -> TranslationResult<(Value, Value, Value, Ptr<Operation>)> {
    require_type_fact(
        is_epilogue_warp_slice_receiver(&operand_rust_type(body, operand, what)?),
        what,
    )?;
    let (slice, after) =
        translate_operand(ctx, body, operand, value_map, block, prev, loc.clone())?;
    let (base, after) = aggregate_field(ctx, slice, 0, what, block, after.or(prev), loc)?;
    let (warp, after) = aggregate_field(ctx, slice, 1, what, block, Some(after), loc)?;
    let (lane, after) = aggregate_field(ctx, slice, 2, what, block, Some(after), loc)?;
    let storage: TypeHandle = MirFP16Type::get(ctx).into();
    let (base, after) = recover_cta_shared_pointer(ctx, base, storage, what, block, after, loc)?;
    Ok((base, warp, lane, after))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_overlay(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 1 {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "epilogue_smem_overlay expects one base pointer".to_owned()
            )
        );
    }
    require_type_fact(
        is_epilogue_tile(&destination_rust_type(
            body,
            destination,
            "epilogue overlay",
        )?),
        "epilogue overlay result",
    )?;
    let (base, after) =
        translate_operand(ctx, body, &args[0], value_map, block, prev, loc.clone())?;
    let storage: TypeHandle = MirFP16Type::get(ctx).into();
    let anchor = after.or(prev).ok_or_else(|| {
        pliron::input_error_noloc!(TranslationErr::unsupported(
            "epilogue overlay base produced no insertion anchor".to_owned()
        ))
    })?;
    let (base, anchor) = recover_cta_shared_pointer(
        ctx,
        base,
        storage,
        "epilogue overlay base",
        block,
        anchor,
        &loc,
    )?;
    let tile = tile_type(ctx);
    let operation = CuteEpilogueSmemOverlayOp::new(ctx, base, tile);
    let op = operation.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, Some(anchor));

    let (outer, base_ty) = translated_field_type(ctx, body, destination, 0, "epilogue overlay")?;
    let (base, anchor) = pointer_cast(ctx, operation.base(ctx), base_ty, block, op, &loc)?;
    let (result, anchor) = build_struct_prefix(ctx, outer, &[base], block, anchor, &loc)?;
    finish_value(
        ctx,
        anchor,
        result,
        destination,
        target,
        block,
        value_map,
        block_map,
        loc,
        "epilogue overlay",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_warp_slice(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 3 {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "epilogue_warp_slice expects tile, warp, and lane".to_owned()
            )
        );
    }
    let (base, after) = epilogue_base(
        ctx,
        body,
        &args[0],
        block,
        prev,
        value_map,
        &loc,
        "epilogue warp-slice tile",
    )?;
    let (warp, warp_after) = translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block,
        Some(after),
        loc.clone(),
    )?;
    let anchor = warp_after.unwrap_or(after);
    let (lane, lane_after) = translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block,
        Some(anchor),
        loc.clone(),
    )?;
    let anchor = lane_after.unwrap_or(anchor);
    let tile = tile_type(ctx);
    let operation = CuteEpilogueWarpSliceOp::new(ctx, base, warp, lane, tile);
    let op = operation.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, Some(anchor));

    let outer = destination_type(ctx, body, destination, "epilogue warp slice")?;
    let base_ty = struct_field_type(ctx, outer, 0, "epilogue warp slice")?;
    let (base, anchor) = pointer_cast(ctx, operation.base(ctx), base_ty, block, op, &loc)?;
    let (result, anchor) = build_struct_prefix(
        ctx,
        outer,
        &[base, operation.warp_id(ctx), operation.lane(ctx)],
        block,
        anchor,
        &loc,
    )?;
    finish_value(
        ctx,
        anchor,
        result,
        destination,
        target,
        block,
        value_map,
        block_map,
        loc,
        "epilogue warp slice",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_store_fragment(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 2 {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "epilogue_store_fragment expects slice and accumulator".to_owned()
            )
        );
    }
    require_type_fact(
        is_accumulator(&operand_rust_type(body, &args[1], "epilogue accumulator")?),
        "epilogue accumulator",
    )?;
    let (base, warp, lane, after) = warp_slice_carriers(
        ctx,
        body,
        &args[0],
        block,
        prev,
        value_map,
        &loc,
        "epilogue warp slice",
    )?;
    let (accumulator, accumulator_after) = translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block,
        Some(after),
        loc.clone(),
    )?;
    let after = accumulator_after.unwrap_or(after);
    let tile = tile_type(ctx);
    let operation = CuteEpilogueStoreFragmentOp::new(ctx, base, warp, lane, accumulator, tile);
    let op = operation.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, Some(after));
    finish(ctx, op, target, block_map, loc, "epilogue store fragment")
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_sync(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    phase: CuteEpilogueSyncPhaseAttr,
    target: &Option<usize>,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 1 {
        return input_err!(
            loc,
            TranslationErr::unsupported("epilogue sync expects one tile".to_owned())
        );
    }
    let (base, after) = epilogue_base(
        ctx,
        body,
        &args[0],
        block,
        prev,
        value_map,
        &loc,
        "epilogue sync tile",
    )?;
    let barrier = CuteCountedCtaBarrierAttr::new(2, 0, 8, 9, 32);
    let tile = tile_type(ctx);
    let operation = CuteEpilogueSyncOp::new(ctx, base, tile, barrier, phase);
    let op = operation.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, Some(after));
    finish(ctx, op, target, block_map, loc, "epilogue sync")
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_half(
    ctx: &mut Context,
    body: &mir::Body,
    func: &mir::Operand,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 1 {
        return input_err!(
            loc,
            TranslationErr::unsupported("epilogue_half expects one tile".to_owned())
        );
    }
    let half = one_unsigned_const(func, "epilogue half")
        .map_err(|error| invalid("epilogue half", error))?;
    let half = u32::try_from(half).map_err(|_| {
        pliron::input_error_noloc!(TranslationErr::unsupported(
            "epilogue half index does not fit u32".to_owned()
        ))
    })?;
    let (base, after) = epilogue_base(
        ctx,
        body,
        &args[0],
        block,
        prev,
        value_map,
        &loc,
        "epilogue half tile",
    )?;
    let tile = tile_type(ctx);
    let operation = CuteEpilogueHalfOp::new(ctx, base, tile, CuteEpilogueHalfAttr(half));
    let op = operation.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, Some(after));

    let outer = destination_type(ctx, body, destination, "epilogue half")?;
    let base_ty = struct_field_type(ctx, outer, 0, "epilogue half")?;
    let (base, anchor) = pointer_cast(ctx, operation.half_base(ctx), base_ty, block, op, &loc)?;
    let (result, anchor) = build_struct_prefix(
        ctx,
        outer,
        &[base, operation.capacity(ctx)],
        block,
        anchor,
        &loc,
    )?;
    finish_value(
        ctx,
        anchor,
        result,
        destination,
        target,
        block,
        value_map,
        block_map,
        loc,
        "epilogue half",
    )
}

#[derive(Clone, Copy)]
pub(crate) enum StorePipelineEffect {
    Acquire,
    Commit,
    Tail,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_store_pipeline_effect(
    ctx: &mut Context,
    func: &mir::Operand,
    args: &[mir::Operand],
    effect: StorePipelineEffect,
    target: &Option<usize>,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() > 1 {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "TMA store-pipeline operation expects zero or one ZST receiver".to_owned()
            )
        );
    }
    let stages = one_unsigned_const(func, "TMA store pipeline")
        .map_err(|error| invalid("TMA store pipeline", error))?;
    if stages != 1 {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!("v0 needs one TMA store stage, got {stages}"))
        );
    }
    let pipeline = CuteTmaStorePipelineAttr::new(1);
    let op = match effect {
        StorePipelineEffect::Acquire => CuteTmaStoreAcquireOp::new(ctx, pipeline).get_operation(),
        StorePipelineEffect::Commit => CuteTmaStoreCommitOp::new(ctx, pipeline).get_operation(),
        StorePipelineEffect::Tail => CuteTmaStoreTailOp::new(ctx, pipeline).get_operation(),
    };
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, prev);
    finish(
        ctx,
        op,
        target,
        block_map,
        loc,
        "TMA store-pipeline operation",
    )
}
