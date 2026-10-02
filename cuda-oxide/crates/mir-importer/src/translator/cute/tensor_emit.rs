/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Emit the high-level elementwise tensor chain.
//!
//! Each recognized Rust call becomes one operation. Pointer arithmetic stays
//! out of the importer; the selected backend lowers it from the preserved view.

use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue::translate_operand;
use crate::translator::terminator::helpers::emit_goto;
use crate::translator::types::translate_type;
use crate::translator::values::ValueMap;
use dialect_cute::attributes::CuteTensorAccessAttr;
use dialect_cute::tensor_ops::{
    CuteTensorBaseOp, CuteTensorIsFullOp, CuteTensorLoadIntoOp, CuteTensorMakeOp,
    CuteTensorSliceOp, CuteTensorStoreElementAbsOp, CuteTensorStoreFromOp,
    CuteTensorZippedDivideOp,
};
use dialect_cute::types::CuteTensorViewType;
use dialect_mir::attributes::{FieldIndexAttr, MirCastKindAttr};
use dialect_mir::ops::{MirCastOp, MirExtractFieldOp, MirLoadOp};
use dialect_mir::types::{MirDisjointSliceType, MirPtrType, MirSliceType};
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;
use rustc_public::mir;

use super::tensor::{decode_register_tile, decode_tensor_view};

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

fn load_pointer_pointee(
    ctx: &mut Context,
    pointer: Value,
    expected: fn(&dyn pliron::r#type::Type) -> bool,
    what: &str,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    loc: &Location,
) -> TranslationResult<(Value, TypeHandle, Ptr<Operation>)> {
    let pointee = {
        let pointer_ty = pointer.get_type(ctx);
        let pointer_ty = pointer_ty.deref(ctx);
        let Some(pointer) = pointer_ty.downcast_ref::<MirPtrType>() else {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(format!("{what} must be a MIR pointer"))
            );
        };
        let pointee_ty = pointer.pointee.deref(ctx);
        if !expected(&*pointee_ty) {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(format!("{what} has the wrong pointee type"))
            );
        }
        pointer.pointee
    };
    let load = Operation::new(
        ctx,
        MirLoadOp::get_concrete_op_info(),
        vec![pointee],
        vec![pointer],
        vec![],
        0,
    );
    load.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, load, block, prev);
    Ok((load.deref(ctx).get_result(0), pointee, load))
}

fn load_tensor_if_referenced(
    ctx: &mut Context,
    value: Value,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    loc: &Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    if value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<CuteTensorViewType>()
        .is_some()
    {
        return Ok((value, prev));
    }
    let (loaded, _, op) = load_pointer_pointee(
        ctx,
        value,
        |ty| ty.downcast_ref::<CuteTensorViewType>().is_some(),
        "tensor receiver",
        block,
        prev,
        loc,
    )?;
    Ok((loaded, Some(op)))
}

fn cast_carrier_to_element(
    ctx: &mut Context,
    pointer: Value,
    element: TypeHandle,
    mutable: bool,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    loc: &Location,
) -> TranslationResult<(Value, Ptr<Operation>)> {
    let (address_space, source_mutable) = {
        let pointer_ty = pointer.get_type(ctx);
        let pointer_ty = pointer_ty.deref(ctx);
        let Some(pointer) = pointer_ty.downcast_ref::<MirPtrType>() else {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported("register tile carrier must be a pointer".to_string())
            );
        };
        (pointer.address_space, pointer.is_mutable)
    };
    if mutable && !source_mutable {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported("load destination carrier must be mutable".to_string())
        );
    }
    let result_ty: TypeHandle = MirPtrType::get(ctx, element, source_mutable, address_space).into();
    let cast = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![result_ty],
        vec![pointer],
        vec![],
        0,
    );
    cast.deref_mut(ctx).set_loc(loc.clone());
    MirCastOp::new(cast).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
    insert(ctx, cast, block, prev);
    Ok((cast.deref(ctx).get_result(0), cast))
}

fn view_transfer_facts(
    ctx: &Context,
    tensor: Value,
    loc: &Location,
) -> TranslationResult<(TypeHandle, u64, u64)> {
    let view = tensor.get_type(ctx);
    let view = view.deref(ctx);
    let Some(view) = view.downcast_ref::<CuteTensorViewType>() else {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported("tensor transfer operand is not a tensor view".to_string())
        );
    };
    let Some(tile) = view.selected_tile_size() else {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported("tensor transfer needs a selected tile".to_string())
        );
    };
    let Some(storage_bytes) = view.storage_bytes(ctx) else {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported("tensor storage width is not supported".to_string())
        );
    };
    let alignment = tile.checked_mul(storage_bytes).ok_or_else(|| {
        pliron::input_error_noloc!(TranslationErr::unsupported(
            "tensor transfer byte width overflows u64".to_string()
        ))
    })?;
    Ok((view.storage, tile, alignment))
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
    expected_access: CuteTensorAccessAttr,
) -> TranslationResult<Ptr<Operation>> {
    let destination_ty = destination.ty(body.locals()).map_err(|error| {
        pliron::input_error_noloc!(TranslationErr::unsupported(format!(
            "cannot read tensor_make destination type: {error:?}"
        )))
    })?;
    let decoded = decode_tensor_view(&destination_ty)
        .map_err(|error| {
            pliron::input_error_noloc!(TranslationErr::unsupported(format!(
                "invalid tensor_make destination: {error}"
            )))
        })?
        .ok_or_else(|| {
            pliron::input_error_noloc!(TranslationErr::unsupported(
                "tensor_make destination is not a v0 tensor view".to_string()
            ))
        })?;
    if decoded.access != expected_access {
        return input_err!(
            loc,
            TranslationErr::unsupported("tensor_make access does not match its API".to_string())
        );
    }
    let logical = translate_type(ctx, &decoded.logical)?;
    let storage = translate_type(ctx, &decoded.storage)?;
    let alignment = decoded
        .storage
        .layout()
        .map_err(|error| {
            pliron::input_error_noloc!(TranslationErr::unsupported(format!(
                "cannot read tensor storage alignment: {error:?}"
            )))
        })?
        .shape()
        .abi_align;

    let (data, len, after_inputs) = if expected_access == CuteTensorAccessAttr::ReadOnly {
        if args.len() != 1 {
            return input_err!(
                loc,
                TranslationErr::unsupported("read tensor_make expects one slice".to_string())
            );
        }
        let (slice, after) =
            translate_operand(ctx, body, &args[0], value_map, block, prev, loc.clone())?;
        if slice
            .get_type(ctx)
            .deref(ctx)
            .downcast_ref::<MirSliceType>()
            .is_none()
        {
            return input_err!(
                loc,
                TranslationErr::unsupported("read tensor_make input is not a slice".to_string())
            );
        }
        let data_ty: TypeHandle = MirPtrType::get_generic(ctx, storage, false).into();
        let (data, data_op) = extract_field(ctx, slice, 0, data_ty, block, after, &loc);
        let len_ty: TypeHandle = crate::translator::types::get_usize_type(ctx).into();
        let (len, len_op) = extract_field(ctx, slice, 1, len_ty, block, Some(data_op), &loc);
        (data, len, Some(len_op))
    } else if args.len() == 2 {
        // Stable free boundary: `(raw pointer, length)`.
        let (data, after) =
            translate_operand(ctx, body, &args[0], value_map, block, prev, loc.clone())?;
        let (len, after) =
            translate_operand(ctx, body, &args[1], value_map, block, after, loc.clone())?;
        (data, len, after)
    } else if args.len() == 1 {
        // Ergonomic method boundary: `&mut DisjointSlice<T>`.
        let (receiver, after) =
            translate_operand(ctx, body, &args[0], value_map, block, prev, loc.clone())?;
        let (slice, pointee, load) = load_pointer_pointee(
            ctx,
            receiver,
            |ty| ty.downcast_ref::<MirDisjointSliceType>().is_some(),
            "TensorMut::from_disjoint_slice receiver",
            block,
            after,
            &loc,
        )?;
        let element = pointee
            .deref(ctx)
            .downcast_ref::<MirDisjointSliceType>()
            .expect("checked above")
            .element_type();
        if element != storage {
            return input_err!(
                loc,
                TranslationErr::unsupported(
                    "DisjointSlice element does not match tensor storage".to_string()
                )
            );
        }
        let data_ty: TypeHandle = MirPtrType::get_generic(ctx, storage, true).into();
        let (data, data_op) = extract_field(ctx, slice, 0, data_ty, block, Some(load), &loc);
        let len_ty: TypeHandle = crate::translator::types::get_usize_type(ctx).into();
        let (len, len_op) = extract_field(ctx, slice, 1, len_ty, block, Some(data_op), &loc);
        (data, len, Some(len_op))
    } else {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "writable tensor_make expects a DisjointSlice or raw pointer and length"
                    .to_string()
            )
        );
    };

    let make = CuteTensorMakeOp::new(ctx, data, len, logical, storage, expected_access, alignment);
    let op = make.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, after_inputs.or(prev));
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
        "tensor_make",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_zipped_divide(
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
            TranslationErr::unsupported("tensor zipped_divide expects one view".to_string())
        );
    }
    let destination_ty = destination.ty(body.locals()).map_err(|error| {
        pliron::input_error_noloc!(TranslationErr::unsupported(format!(
            "cannot read zipped_divide destination type: {error:?}"
        )))
    })?;
    let result = decode_tensor_view(&destination_ty)
        .map_err(|error| {
            pliron::input_error_noloc!(TranslationErr::unsupported(format!(
                "invalid zipped_divide destination: {error}"
            )))
        })?
        .ok_or_else(|| {
            pliron::input_error_noloc!(TranslationErr::unsupported(
                "zipped_divide destination is not a v0 tensor view".to_string()
            ))
        })?;
    let tile_size = result.layout.tile_size().ok_or_else(|| {
        pliron::input_error_noloc!(TranslationErr::unsupported(
            "zipped_divide result does not carry a tile width".to_string()
        ))
    })?;
    let (tensor, after) =
        translate_operand(ctx, body, &args[0], value_map, block, prev, loc.clone())?;
    let divide = CuteTensorZippedDivideOp::new(ctx, tensor, tile_size);
    let op = divide.get_operation();
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
        "tensor_zipped_divide",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_slice(
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
            TranslationErr::unsupported("tensor slice expects a view and tile index".to_string())
        );
    }
    let (tensor, after) =
        translate_operand(ctx, body, &args[0], value_map, block, prev, loc.clone())?;
    let (tile_index, after) =
        translate_operand(ctx, body, &args[1], value_map, block, after, loc.clone())?;
    let slice = CuteTensorSliceOp::new(ctx, tensor, tile_index);
    let op = slice.get_operation();
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
        "tensor_slice",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_query(
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
    is_full: bool,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 1 {
        return input_err!(
            loc,
            TranslationErr::unsupported("tensor query expects one selected tile".to_string())
        );
    }
    let (tensor, after) =
        translate_operand(ctx, body, &args[0], value_map, block, prev, loc.clone())?;
    let op = if is_full {
        CuteTensorIsFullOp::new(ctx, tensor).get_operation()
    } else {
        CuteTensorBaseOp::new(ctx, tensor).get_operation()
    };
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
        if is_full {
            "tensor_is_full"
        } else {
            "tensor_base"
        },
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
    if args.len() != 1 || !destination.projection.is_empty() {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "tensor load expects one view and a local destination".to_string()
            )
        );
    }
    let (tensor, after) =
        translate_operand(ctx, body, &args[0], value_map, block, prev, loc.clone())?;
    let (storage, tile_size, alignment) = view_transfer_facts(ctx, tensor, &loc)?;
    let destination_ty = destination.ty(body.locals()).map_err(|error| {
        pliron::input_error_noloc!(TranslationErr::unsupported(format!(
            "cannot read tensor load destination type: {error:?}"
        )))
    })?;
    let register = decode_register_tile(&destination_ty)
        .map_err(|error| {
            pliron::input_error_noloc!(TranslationErr::unsupported(format!(
                "invalid tensor load destination: {error}"
            )))
        })?
        .ok_or_else(|| {
            pliron::input_error_noloc!(TranslationErr::unsupported(
                "tensor load destination is not RegisterTile".to_string()
            ))
        })?;
    if register.size != tile_size || translate_type(ctx, &register.element)? != storage {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "tensor load tile shape does not match RegisterTile".to_string()
            )
        );
    }
    let Some(slot) = value_map.get_slot(destination.local) else {
        return input_err!(
            loc,
            TranslationErr::unsupported("tensor load destination has no local slot".to_string())
        );
    };
    let (destination, cast) =
        cast_carrier_to_element(ctx, slot, storage, true, block, after, &loc)?;
    let load = CuteTensorLoadIntoOp::new(ctx, tensor, destination, alignment);
    let op = load.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, Some(cast));
    finish(ctx, op, target, block_map, loc, "tensor_load_into")
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_store(
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
            TranslationErr::unsupported("tensor store expects a view and RegisterTile".to_string())
        );
    }
    let (tensor, after) =
        translate_operand(ctx, body, &args[0], value_map, block, prev, loc.clone())?;
    let (storage, tile_size, alignment) = view_transfer_facts(ctx, tensor, &loc)?;
    let value_ty = args[1].ty(body.locals()).map_err(|error| {
        pliron::input_error_noloc!(TranslationErr::unsupported(format!(
            "cannot read tensor store source type: {error:?}"
        )))
    })?;
    let register = decode_register_tile(&value_ty)
        .map_err(|error| {
            pliron::input_error_noloc!(TranslationErr::unsupported(format!(
                "invalid tensor store source: {error}"
            )))
        })?
        .ok_or_else(|| {
            pliron::input_error_noloc!(TranslationErr::unsupported(
                "tensor store source is not RegisterTile".to_string()
            ))
        })?;
    if register.size != tile_size || translate_type(ctx, &register.element)? != storage {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "tensor store tile shape does not match RegisterTile".to_string()
            )
        );
    }
    let source = match &args[1] {
        mir::Operand::Copy(place) | mir::Operand::Move(place) if place.projection.is_empty() => {
            value_map.get_slot(place.local)
        }
        _ => None,
    }
    .ok_or_else(|| {
        pliron::input_error_noloc!(TranslationErr::unsupported(
            "tensor store RegisterTile must live in a local slot".to_string()
        ))
    })?;
    let (source, cast) = cast_carrier_to_element(ctx, source, storage, false, block, after, &loc)?;
    let store = CuteTensorStoreFromOp::new(ctx, source, tensor, alignment);
    let op = store.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, Some(cast));
    finish(ctx, op, target, block_map, loc, "tensor_store_from")
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_store_element_abs(
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
    if args.len() != 3 {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "tensor tail store expects a view, absolute index, and value".to_string()
            )
        );
    }
    let (receiver, after) =
        translate_operand(ctx, body, &args[0], value_map, block, prev, loc.clone())?;
    let (tensor, after) = load_tensor_if_referenced(ctx, receiver, block, after, &loc)?;
    let (index, after) =
        translate_operand(ctx, body, &args[1], value_map, block, after, loc.clone())?;
    let (value, after) =
        translate_operand(ctx, body, &args[2], value_map, block, after, loc.clone())?;
    let store = CuteTensorStoreElementAbsOp::new(ctx, tensor, index, value);
    let op = store.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, after.or(prev));
    finish(ctx, op, target, block_map, loc, "tensor_store_element_abs")
}
