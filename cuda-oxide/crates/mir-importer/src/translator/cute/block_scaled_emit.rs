/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Emit the high-level block-scaled GEMV chain.
//!
//! Pointer arithmetic and packed-number conversion stay out of the importer:
//!
//! ```text
//! slices -> scaled view -> row -> K64 tile -> fragment -> dot
//! ```

use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue::translate_operand;
use crate::translator::terminator::helpers::emit_goto;
use crate::translator::values::ValueMap;
use dialect_cute::attributes::CuteTensorRoleAttr;
use dialect_cute::gemv_ops::{
    CuteDotOp, CuteScaledViewKTileOp, CuteScaledViewLoadOp, CuteScaledViewMakeOp,
    CuteScaledViewRowOp, CuteTensorMake2DOp,
};
use dialect_mir::attributes::FieldIndexAttr;
use dialect_mir::ops::MirExtractFieldOp;
use dialect_mir::types::{MirPtrType, MirSliceType};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;
use rustc_public::mir;

use super::block_scaled::{BlockScaledStage, decode_block_scaled};

fn insert(
    ctx: &mut Context,
    op: Ptr<Operation>,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
) {
    if let Some(prev) = prev {
        op.insert_after(ctx, prev);
    } else {
        op.insert_at_front(block, ctx);
    }
}

fn finish(
    ctx: &mut Context,
    effect: Ptr<Operation>,
    target: &Option<usize>,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    what: &str,
) -> TranslationResult<Ptr<Operation>> {
    let Some(target) = target else {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!("{what} call without target is not supported"))
        );
    };
    Ok(emit_goto(ctx, *target, effect, block_map, loc))
}

#[allow(clippy::too_many_arguments)]
fn finish_value(
    ctx: &mut Context,
    producer: Ptr<Operation>,
    value: Value,
    destination: &mir::Place,
    target: &Option<usize>,
    block: Ptr<BasicBlock>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    what: &str,
) -> TranslationResult<Ptr<Operation>> {
    if !destination.projection.is_empty() {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "{what} destination with projections is not supported"
            ))
        );
    }
    let after_store = value_map
        .store_local(ctx, destination.local, value, block, Some(producer))
        .unwrap_or(producer);
    finish(ctx, after_store, target, block_map, loc, what)
}

fn extract_field(
    ctx: &mut Context,
    aggregate: Value,
    index: u32,
    result_ty: TypeHandle,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    loc: &Location,
) -> (Value, Ptr<Operation>) {
    let op = Operation::new(
        ctx,
        MirExtractFieldOp::get_concrete_op_info(),
        vec![result_ty],
        vec![aggregate],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());
    MirExtractFieldOp::new(op).set_attr_index(ctx, FieldIndexAttr(index));
    insert(ctx, op, block, prev);
    (op.deref(ctx).get_result(0), op)
}

fn split_u8_slice(
    ctx: &mut Context,
    slice: Value,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    loc: &Location,
    what: &str,
) -> TranslationResult<(Value, Value, Ptr<Operation>)> {
    let u8_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
    let element_type = {
        let slice_ty = slice.get_type(ctx);
        let slice_ty = slice_ty.deref(ctx);
        let Some(slice_ty) = slice_ty.downcast_ref::<MirSliceType>() else {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(format!("{what} must be a slice"))
            );
        };
        slice_ty.element_type()
    };
    if element_type != u8_ty {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!("{what} must contain u8 storage"))
        );
    }

    let data_ty: TypeHandle = MirPtrType::get_generic(ctx, u8_ty, false).into();
    let (data, data_op) = extract_field(ctx, slice, 0, data_ty, block, prev, loc);
    let len_ty: TypeHandle = crate::translator::types::get_usize_type(ctx).into();
    let (len, len_op) = extract_field(ctx, slice, 1, len_ty, block, Some(data_op), loc);
    Ok((data, len, len_op))
}

fn destination_facts(
    body: &mir::Body,
    destination: &mir::Place,
    expected_stage: BlockScaledStage,
    loc: &Location,
    what: &str,
) -> TranslationResult<CuteTensorRoleAttr> {
    let ty = destination.ty(body.locals()).map_err(|error| {
        pliron::input_error_noloc!(TranslationErr::unsupported(format!(
            "cannot read {what} destination type: {error:?}"
        )))
    })?;
    let decoded = decode_block_scaled(&ty)
        .map_err(|error| {
            pliron::input_error_noloc!(TranslationErr::unsupported(format!(
                "invalid {what} destination: {error}"
            )))
        })?
        .ok_or_else(|| {
            pliron::input_error_noloc!(TranslationErr::unsupported(format!(
                "{what} destination is not a block-scaled view"
            )))
        })?;
    if decoded.stage != expected_stage {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!("{what} destination has the wrong view stage"))
        );
    }
    Ok(decoded.role)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_make(
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
    if args.len() != 4 {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "block_scaled_make expects value/scale slices, rows, and K".to_string()
            )
        );
    }
    let role = destination_facts(
        body,
        destination,
        BlockScaledStage::Full,
        &loc,
        "block_scaled_make",
    )?;

    let (values_slice, after) =
        translate_operand(ctx, body, &args[0], value_map, block, prev, loc.clone())?;
    let (values_data, values_len, after) =
        split_u8_slice(ctx, values_slice, block, after, &loc, "block-scaled values")?;
    let (scales_slice, after) = translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block,
        Some(after),
        loc.clone(),
    )?;
    let (scales_data, scales_len, after) =
        split_u8_slice(ctx, scales_slice, block, after, &loc, "block-scaled scales")?;
    let (rows, after) = translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block,
        Some(after),
        loc.clone(),
    )?;
    let (k, after) = translate_operand(ctx, body, &args[3], value_map, block, after, loc.clone())?;

    let values = CuteTensorMake2DOp::new_e2m1(ctx, values_data, values_len, rows, k, role, 1);
    let values_op = values.get_operation();
    values_op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, values_op, block, after.or(prev));

    let scales = CuteTensorMake2DOp::new_ue8m0(ctx, scales_data, scales_len, rows, k, role, 1, 16);
    let scales_op = scales.get_operation();
    scales_op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, scales_op, block, Some(values_op));

    let values_view = values_op.deref(ctx).get_result(0);
    let scales_view = scales_op.deref(ctx).get_result(0);
    let scaled = CuteScaledViewMakeOp::new(ctx, values_view, scales_view);
    let op = scaled.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, Some(scales_op));
    let result = op.deref(ctx).get_result(0);
    finish_value(
        ctx,
        op,
        result,
        destination,
        target,
        block,
        value_map,
        block_map,
        loc,
        "block_scaled_make",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_row(
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
                "block_scaled_thread_row expects a view, batch, and row".to_string()
            )
        );
    }
    destination_facts(
        body,
        destination,
        BlockScaledStage::Row,
        &loc,
        "block_scaled_thread_row",
    )?;
    let (scaled, after) =
        translate_operand(ctx, body, &args[0], value_map, block, prev, loc.clone())?;
    let (batch, after) =
        translate_operand(ctx, body, &args[1], value_map, block, after, loc.clone())?;
    let (row, after) =
        translate_operand(ctx, body, &args[2], value_map, block, after, loc.clone())?;
    let row_op = CuteScaledViewRowOp::new(ctx, scaled, batch, row);
    let op = row_op.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, after.or(prev));
    let result = op.deref(ctx).get_result(0);
    finish_value(
        ctx,
        op,
        result,
        destination,
        target,
        block,
        value_map,
        block_map,
        loc,
        "block_scaled_thread_row",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_k_tile(
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
    if args.len() != 2 {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "block_scaled_k_tile expects a row and tile index".to_string()
            )
        );
    }
    destination_facts(
        body,
        destination,
        BlockScaledStage::KTile64,
        &loc,
        "block_scaled_k_tile",
    )?;
    let (row, after) = translate_operand(ctx, body, &args[0], value_map, block, prev, loc.clone())?;
    let (tile, after) =
        translate_operand(ctx, body, &args[1], value_map, block, after, loc.clone())?;
    let tile_op = CuteScaledViewKTileOp::new(ctx, row, tile);
    let op = tile_op.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, after.or(prev));
    let result = op.deref(ctx).get_result(0);
    finish_value(
        ctx,
        op,
        result,
        destination,
        target,
        block,
        value_map,
        block_map,
        loc,
        "block_scaled_k_tile",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_load(
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
            TranslationErr::unsupported("block_scaled_load_k64 expects one K tile".to_string())
        );
    }
    destination_facts(
        body,
        destination,
        BlockScaledStage::Fragment64,
        &loc,
        "block_scaled_load_k64",
    )?;
    let (tile, after) =
        translate_operand(ctx, body, &args[0], value_map, block, prev, loc.clone())?;
    // The safe slice constructor knows only byte alignment. These stronger
    // selected-address promises belong to the unsafe Rust load boundary.
    let load = CuteScaledViewLoadOp::new(ctx, tile, 16, 4);
    let op = load.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, after.or(prev));
    let result = op.deref(ctx).get_result(0);
    finish_value(
        ctx,
        op,
        result,
        destination,
        target,
        block,
        value_map,
        block_map,
        loc,
        "block_scaled_load_k64",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_dot(
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
                "block_scaled_dot_k64 expects matrix/vector fragments and acc".to_string()
            )
        );
    }
    let (matrix, after) =
        translate_operand(ctx, body, &args[0], value_map, block, prev, loc.clone())?;
    let (vector, after) =
        translate_operand(ctx, body, &args[1], value_map, block, after, loc.clone())?;
    let (acc, after) =
        translate_operand(ctx, body, &args[2], value_map, block, after, loc.clone())?;
    let dot = CuteDotOp::new(ctx, matrix, vector, acc);
    let op = dot.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, after.or(prev));
    let result = op.deref(ctx).get_result(0);
    finish_value(
        ctx,
        op,
        result,
        destination,
        target,
        block,
        value_map,
        block_map,
        loc,
        "block_scaled_dot_k64",
    )
}
