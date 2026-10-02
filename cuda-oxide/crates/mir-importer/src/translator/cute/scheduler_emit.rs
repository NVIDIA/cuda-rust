/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Emit the five high-level persistent-scheduler operations.
//!
//! The Rust scheduler stays an ordinary `{ current, stride }` value. Only the
//! selected `WorkTile` is a short-lived compiler handle:
//!
//! ```text
//! new_1d -> has_work -> current_tile -> coordinates -> advance
//! ```

use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue::translate_operand;
use crate::translator::terminator::helpers::emit_goto;
use crate::translator::types::translate_type;
use crate::translator::values::ValueMap;
use dialect_cute::attributes::CuteTileGridAttr;
use dialect_cute::scheduler_ops::{
    CuteSchedulerAdvanceOp, CuteSchedulerCurrentOp, CuteSchedulerHasWorkOp, CuteSchedulerNew1dOp,
    CuteWorkTileCoordinatesOp,
};
use dialect_mir::attributes::FieldIndexAttr;
use dialect_mir::ops::{MirFieldAddrOp, MirInsertFieldOp, MirLoadOp, MirStoreOp, MirUndefOp};
use dialect_mir::types::{MirPtrType, MirStructType};
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::printable::Printable;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;
use rustc_public::mir;

use super::scheduler::{
    decode_scheduler, decode_scheduler_receiver, decode_work_tile, decode_work_tile_receiver,
};

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

fn destination_type(
    ctx: &mut Context,
    body: &mir::Body,
    destination: &mir::Place,
    _loc: &Location,
    what: &str,
) -> TranslationResult<TypeHandle> {
    let ty = destination.ty(body.locals()).map_err(|error| {
        pliron::input_error_noloc!(TranslationErr::unsupported(format!(
            "cannot read {what} destination type: {error:?}"
        )))
    })?;
    translate_type(ctx, &ty).map_err(|error| {
        pliron::input_error_noloc!(TranslationErr::unsupported(format!(
            "cannot translate {what} destination type: {}",
            error.disp(ctx)
        )))
    })
}

fn build_aggregate(
    ctx: &mut Context,
    aggregate_type: TypeHandle,
    fields: &[Value],
    block: Ptr<BasicBlock>,
    prev: Ptr<Operation>,
    loc: &Location,
) -> (Value, Ptr<Operation>) {
    let undef = MirUndefOp::new(ctx, aggregate_type).get_operation();
    undef.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, undef, block, Some(prev));
    let mut aggregate = undef.deref(ctx).get_result(0);
    let mut anchor = undef;

    for (index, field) in fields.iter().copied().enumerate() {
        let op = Operation::new(
            ctx,
            MirInsertFieldOp::get_concrete_op_info(),
            vec![aggregate_type],
            vec![aggregate, field],
            vec![],
            0,
        );
        op.deref_mut(ctx).set_loc(loc.clone());
        MirInsertFieldOp::new(op).set_attr_insert_index(ctx, FieldIndexAttr(index as u32));
        insert(ctx, op, block, Some(anchor));
        aggregate = op.deref(ctx).get_result(0);
        anchor = op;
    }
    (aggregate, anchor)
}

fn receiver_grid(
    body: &mir::Body,
    operand: &mir::Operand,
    work_tile: bool,
    _loc: &Location,
    what: &str,
) -> TranslationResult<CuteTileGridAttr> {
    let ty = operand.ty(body.locals()).map_err(|error| {
        pliron::input_error_noloc!(TranslationErr::unsupported(format!(
            "cannot read {what} receiver type: {error:?}"
        )))
    })?;
    let decoded = if work_tile {
        decode_work_tile_receiver(&ty)
    } else {
        decode_scheduler_receiver(&ty)
    };
    decoded
        .map_err(|error| {
            pliron::input_error_noloc!(TranslationErr::unsupported(format!(
                "invalid {what} receiver: {error}"
            )))
        })?
        .ok_or_else(|| {
            pliron::input_error_noloc!(TranslationErr::unsupported(format!(
                "{what} receiver has the wrong cute-rs type"
            )))
        })
}

fn scheduler_field_addr(
    ctx: &mut Context,
    scheduler: Value,
    index: usize,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    loc: &Location,
    what: &str,
) -> TranslationResult<(Value, TypeHandle, Ptr<Operation>)> {
    let field = {
        let pointer_type = scheduler.get_type(ctx);
        let pointer_type = pointer_type.deref(ctx);
        let Some(pointer) = pointer_type.downcast_ref::<MirPtrType>() else {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(format!("{what} receiver must be a pointer"))
            );
        };
        let pointee = pointer.pointee.deref(ctx);
        let Some(scheduler_type) = pointee.downcast_ref::<MirStructType>() else {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(format!("{what} receiver must point to a scheduler"))
            );
        };
        let Some(field) = scheduler_type.field_types.get(index).copied() else {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(format!("{what} scheduler has no field {index}"))
            );
        };
        (field, pointer.is_mutable, pointer.address_space)
    };
    let pointer_type: TypeHandle = MirPtrType::get(ctx, field.0, field.1, field.2).into();
    // Go through the one construction door so `aggregate_ty` is stamped;
    // the verifier rejects a `mir.field_addr` without it.
    let op = MirFieldAddrOp::build(ctx, scheduler, pointer_type, index as u32)?;
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, prev);
    Ok((op.deref(ctx).get_result(0), field.0, op))
}

fn load_field(
    ctx: &mut Context,
    pointer: Value,
    field_type: TypeHandle,
    block: Ptr<BasicBlock>,
    prev: Ptr<Operation>,
    loc: &Location,
) -> (Value, Ptr<Operation>) {
    let op = Operation::new(
        ctx,
        MirLoadOp::get_concrete_op_info(),
        vec![field_type],
        vec![pointer],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());
    MirLoadOp::new(op).set_volatile(ctx, false);
    insert(ctx, op, block, Some(prev));
    (op.deref(ctx).get_result(0), op)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_new_1d(
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
    if !args.is_empty() {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "scheduler_new_1d expects no runtime arguments".to_string()
            )
        );
    }
    let rust_type = destination.ty(body.locals()).map_err(|error| {
        pliron::input_error_noloc!(TranslationErr::unsupported(format!(
            "cannot read scheduler_new_1d result: {error:?}"
        )))
    })?;
    let grid = decode_scheduler(&rust_type)
        .map_err(|error| {
            pliron::input_error_noloc!(TranslationErr::unsupported(format!(
                "invalid scheduler_new_1d result: {error}"
            )))
        })?
        .ok_or_else(|| {
            pliron::input_error_noloc!(TranslationErr::unsupported(
                "scheduler_new_1d must return StaticPersistentTileScheduler".to_string()
            ))
        })?;

    let operation = CuteSchedulerNew1dOp::new(ctx, grid);
    let op = operation.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, prev);
    let result_type = destination_type(ctx, body, destination, &loc, "scheduler_new_1d")?;
    let (result, bundle) = build_aggregate(
        ctx,
        result_type,
        &[operation.current(ctx), operation.stride(ctx)],
        block,
        op,
        &loc,
    );
    finish_value(
        ctx,
        bundle,
        result,
        destination,
        target,
        block,
        value_map,
        block_map,
        loc,
        "scheduler_new_1d",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_has_work(
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
            TranslationErr::unsupported("scheduler_has_work expects one receiver".to_string())
        );
    }
    let grid = receiver_grid(body, &args[0], false, &loc, "scheduler_has_work")?;
    let (scheduler, after) =
        translate_operand(ctx, body, &args[0], value_map, block, prev, loc.clone())?;
    let (current_pointer, current_type, after) = scheduler_field_addr(
        ctx,
        scheduler,
        0,
        block,
        after.or(prev),
        &loc,
        "scheduler_has_work",
    )?;
    let (current, after) = load_field(ctx, current_pointer, current_type, block, after, &loc);
    let operation = CuteSchedulerHasWorkOp::new(ctx, current, grid);
    let op = operation.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, Some(after));
    finish_value(
        ctx,
        op,
        operation.has_work(ctx),
        destination,
        target,
        block,
        value_map,
        block_map,
        loc,
        "scheduler_has_work",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_current_tile(
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
            TranslationErr::unsupported("scheduler_current_tile expects one receiver".to_string())
        );
    }
    let grid = receiver_grid(body, &args[0], false, &loc, "scheduler_current_tile")?;
    let result_type = destination.ty(body.locals()).map_err(|error| {
        pliron::input_error_noloc!(TranslationErr::unsupported(format!(
            "cannot read scheduler_current_tile result: {error:?}"
        )))
    })?;
    if decode_work_tile(&result_type).map_err(|error| {
        pliron::input_error_noloc!(TranslationErr::unsupported(format!(
            "invalid scheduler_current_tile result: {error}"
        )))
    })? != Some(grid)
    {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "scheduler_current_tile result must use the receiver's tile grid".to_string()
            )
        );
    }
    let (scheduler, after) =
        translate_operand(ctx, body, &args[0], value_map, block, prev, loc.clone())?;
    let (current_pointer, current_type, after) = scheduler_field_addr(
        ctx,
        scheduler,
        0,
        block,
        after.or(prev),
        &loc,
        "scheduler_current_tile",
    )?;
    let (current, after) = load_field(ctx, current_pointer, current_type, block, after, &loc);
    let operation = CuteSchedulerCurrentOp::new(ctx, current, grid);
    let op = operation.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, Some(after));
    finish_value(
        ctx,
        op,
        operation.work_tile(ctx),
        destination,
        target,
        block,
        value_map,
        block_map,
        loc,
        "scheduler_current_tile",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_coordinates(
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
            TranslationErr::unsupported("work_tile_coordinates expects one tile".to_string())
        );
    }
    let _grid = receiver_grid(body, &args[0], true, &loc, "work_tile_coordinates")?;
    let (tile, after) =
        translate_operand(ctx, body, &args[0], value_map, block, prev, loc.clone())?;
    let operation = CuteWorkTileCoordinatesOp::new(ctx, tile);
    let op = operation.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, after.or(prev));
    let result_type = destination_type(ctx, body, destination, &loc, "work_tile_coordinates")?;
    let (result, bundle) = build_aggregate(
        ctx,
        result_type,
        &[
            operation.linear(ctx),
            operation.m_tile(ctx),
            operation.n_tile(ctx),
            operation.batch(ctx),
        ],
        block,
        op,
        &loc,
    );
    finish_value(
        ctx,
        bundle,
        result,
        destination,
        target,
        block,
        value_map,
        block_map,
        loc,
        "work_tile_coordinates",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_advance(
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
    if args.len() != 1 {
        return input_err!(
            loc,
            TranslationErr::unsupported("scheduler_advance expects one receiver".to_string())
        );
    }
    let grid = receiver_grid(body, &args[0], false, &loc, "scheduler_advance")?;
    let (scheduler, after) =
        translate_operand(ctx, body, &args[0], value_map, block, prev, loc.clone())?;
    let (current_pointer, current_type, after) = scheduler_field_addr(
        ctx,
        scheduler,
        0,
        block,
        after.or(prev),
        &loc,
        "scheduler_advance",
    )?;
    let (current, after) = load_field(ctx, current_pointer, current_type, block, after, &loc);
    let (stride_pointer, stride_type, after) = scheduler_field_addr(
        ctx,
        scheduler,
        1,
        block,
        Some(after),
        &loc,
        "scheduler_advance",
    )?;
    let (stride, after) = load_field(ctx, stride_pointer, stride_type, block, after, &loc);
    let operation = CuteSchedulerAdvanceOp::new(ctx, current, stride, grid);
    let op = operation.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, Some(after));

    let store = Operation::new(
        ctx,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![current_pointer, operation.next(ctx)],
        vec![],
        0,
    );
    store.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, store, block, Some(op));
    finish(ctx, store, target, block_map, loc, "scheduler_advance")
}
