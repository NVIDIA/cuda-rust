/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Emit the semantic shared-memory MXFP4 MMA chain.
//!
//! Every result keeps the ordinary Rust carrier it had before recognition:
//! pointers and coordinates stay scalars, fragments stay register aggregates,
//! and the accumulator remains the loop-carried 64-f32 value.

use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue::translate_operand;
use crate::translator::terminator::helpers::emit_goto;
use crate::translator::types::translate_type;
use crate::translator::values::ValueMap;
use dialect_cute::attributes::{
    CuteMmaCarrierKindAttr, CuteTensorAccessAttr, CuteTensorAddressSpaceAttr, CuteTensorFormatAttr,
    CuteTensorLayoutAttr, CuteTiledMmaPlanAttr,
};
use dialect_cute::smem_mma_ops::{
    CuteFragmentFillOp, CuteFragmentSliceKOp, CuteMmaLoadAOp, CuteMmaLoadScalesOp,
    CuteMmaPartitionBOp, CuteSmemTensorOverlayOp, CuteTiledGemmOp, CuteTiledMmaSliceOp,
};
use dialect_cute::types::{CuteSmemTensorType, CuteTensorViewType};
use dialect_mir::attributes::{FieldIndexAttr, MirCastKindAttr};
use dialect_mir::ops::{
    MirCastOp, MirExtractFieldOp, MirFloatConstantOp, MirInsertFieldOp, MirStoreOp, MirUndefOp,
};
use dialect_mir::types::{MirFP16Type, MirPtrType, MirStructType};
use pliron::basic_block::BasicBlock;
use pliron::builtin::{
    attributes::FPSingleAttr,
    types::{FP32Type, IntegerType, Signedness},
};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::printable::Printable;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;
use rustc_public::mir;

use super::emit::{load_through, recover_cta_shared_pointer, struct_field_addr};
use super::smem_mma::{
    SharedTensorKind, SharedTensorRust, TiledMmaRust, canonical_data_placement, decode_b_tile,
    decode_shared_tensor, decode_shared_tensor_receiver, decode_tiled_mma,
    decode_tiled_mma_receiver, is_accumulator, is_scale_k64, is_scale_stage,
};

pub(super) fn insert(
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

pub(super) fn finish(
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
pub(super) fn finish_value(
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
    let after = value_map
        .store_local(ctx, destination.local, value, block, Some(producer))
        .unwrap_or(producer);
    finish(ctx, after, target, block_map, loc, what)
}

pub(super) fn destination_rust_type(
    body: &mir::Body,
    destination: &mir::Place,
    what: &str,
) -> TranslationResult<rustc_public::ty::Ty> {
    destination.ty(body.locals()).map_err(|error| {
        pliron::input_error_noloc!(TranslationErr::unsupported(format!(
            "cannot read {what} destination type: {error:?}"
        )))
    })
}

pub(super) fn destination_type(
    ctx: &mut Context,
    body: &mir::Body,
    destination: &mir::Place,
    what: &str,
) -> TranslationResult<TypeHandle> {
    let ty = destination_rust_type(body, destination, what)?;
    translate_type(ctx, &ty).map_err(|error| {
        pliron::input_error_noloc!(TranslationErr::unsupported(format!(
            "cannot translate {what} destination type: {}",
            error.disp(ctx)
        )))
    })
}

pub(super) fn operand_rust_type(
    body: &mir::Body,
    operand: &mir::Operand,
    what: &str,
) -> TranslationResult<rustc_public::ty::Ty> {
    operand.ty(body.locals()).map_err(|error| {
        pliron::input_error_noloc!(TranslationErr::unsupported(format!(
            "cannot read {what} operand type: {error:?}"
        )))
    })
}

pub(super) fn aggregate_field(
    ctx: &mut Context,
    aggregate: Value,
    index: usize,
    what: &str,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    loc: &Location,
) -> TranslationResult<(Value, Ptr<Operation>)> {
    let aggregate_ty = aggregate.get_type(ctx);
    if aggregate_ty
        .deref(ctx)
        .downcast_ref::<MirPtrType>()
        .is_some()
    {
        let (pointer, field_ty, addr) =
            struct_field_addr(ctx, aggregate, index, what, block, prev, loc)?;
        return Ok(load_through(ctx, pointer, field_ty, block, Some(addr), loc));
    }
    let field_ty = {
        let aggregate_ty = aggregate_ty.deref(ctx);
        let Some(structure) = aggregate_ty.downcast_ref::<MirStructType>() else {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(format!("{what} must be a Rust struct carrier"))
            );
        };
        let Some(field_ty) = structure.field_types.get(index).copied() else {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(format!("{what} has no field {index}"))
            );
        };
        field_ty
    };
    let op = Operation::new(
        ctx,
        MirExtractFieldOp::get_concrete_op_info(),
        vec![field_ty],
        vec![aggregate],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());
    MirExtractFieldOp::new(op).set_attr_index(ctx, FieldIndexAttr(index as u32));
    insert(ctx, op, block, prev);
    Ok((op.deref(ctx).get_result(0), op))
}

pub(super) fn build_struct_prefix(
    ctx: &mut Context,
    ty: TypeHandle,
    fields: &[Value],
    block: Ptr<BasicBlock>,
    prev: Ptr<Operation>,
    loc: &Location,
) -> TranslationResult<(Value, Ptr<Operation>)> {
    let field_types = {
        let ty_ref = ty.deref(ctx);
        let Some(structure) = ty_ref.downcast_ref::<MirStructType>() else {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported("compiler result must be a Rust struct".to_string())
            );
        };
        structure.field_types.clone()
    };
    if fields.len() > field_types.len() {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported("compiler result has too few carrier fields".to_string())
        );
    }
    let undef = MirUndefOp::new(ctx, ty).get_operation();
    undef.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, undef, block, Some(prev));
    let mut value = undef.deref(ctx).get_result(0);
    let mut anchor = undef;
    for (index, field) in fields.iter().copied().enumerate() {
        if field.get_type(ctx) != field_types[index] {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "compiler result field {index} has the wrong carrier type"
                ))
            );
        }
        let op = Operation::new(
            ctx,
            MirInsertFieldOp::get_concrete_op_info(),
            vec![ty],
            vec![value, field],
            vec![],
            0,
        );
        op.deref_mut(ctx).set_loc(loc.clone());
        MirInsertFieldOp::new(op).set_attr_insert_index(ctx, FieldIndexAttr(index as u32));
        insert(ctx, op, block, Some(anchor));
        value = op.deref(ctx).get_result(0);
        anchor = op;
    }
    Ok((value, anchor))
}

pub(super) fn pointer_cast(
    ctx: &mut Context,
    value: Value,
    target: TypeHandle,
    block: Ptr<BasicBlock>,
    prev: Ptr<Operation>,
    loc: &Location,
) -> TranslationResult<(Value, Ptr<Operation>)> {
    if value.get_type(ctx) == target {
        return Ok((value, prev));
    }
    if value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<MirPtrType>()
        .is_none()
        || target.deref(ctx).downcast_ref::<MirPtrType>().is_none()
    {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported("compiler carrier cast must be pointer-to-pointer")
        );
    }
    let cast = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![target],
        vec![value],
        vec![],
        0,
    );
    cast.deref_mut(ctx).set_loc(loc.clone());
    MirCastOp::new(cast).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
    // This cast adapts a compiler-produced carrier to the exact Rust type
    // declared on the cute-rs stub. When that type names a concrete raw
    // pointer kind (e.g. `*mut f16` = RawMut), the kind transition needs an
    // explicit authority. `RawAddress` is the same authority the importer
    // stamps on `DynamicSharedArray`'s result: the stub hands out a raw
    // pointer to compiler-managed storage, exactly a `&raw` boundary.
    if dialect_mir::types::type_contains_concrete_pointer_kind(ctx, target) {
        MirCastOp::new(cast).set_pointer_kind_authority(
            ctx,
            dialect_mir::attributes::MirPointerKindAuthorityAttr::RawAddress,
        );
    }
    insert(ctx, cast, block, Some(prev));
    Ok((cast.deref(ctx).get_result(0), cast))
}

fn plan(facts: &TiledMmaRust) -> CuteTiledMmaPlanAttr {
    CuteTiledMmaPlanAttr::mxf4_128x128x128(facts.placement.clone())
}

fn fixed_plan() -> CuteTiledMmaPlanAttr {
    CuteTiledMmaPlanAttr::mxf4_128x128x128(canonical_data_placement())
}

fn smem_view_type(ctx: &mut Context, facts: &SharedTensorRust) -> (TypeHandle, TypeHandle) {
    let logical: TypeHandle = FP32Type::get(ctx).into();
    let (storage, alignment, format, layout): (TypeHandle, u64, _, _) = match facts.kind {
        SharedTensorKind::Data => (
            MirFP16Type::get(ctx).into(),
            16,
            CuteTensorFormatAttr::E2M1,
            CuteTensorLayoutAttr::KMajor,
        ),
        SharedTensorKind::Scale => (
            IntegerType::get(ctx, 32, Signedness::Unsigned).into(),
            4,
            CuteTensorFormatAttr::UE8M0,
            CuteTensorLayoutAttr::BlockScaleKMajor(32),
        ),
    };
    let tensor: TypeHandle = CuteTensorViewType::get_with_facts(
        ctx,
        logical,
        storage,
        CuteTensorAddressSpaceAttr::Smem,
        CuteTensorAccessAttr::ReadOnly,
        alignment,
        format,
        facts.role,
        layout,
    )
    .into();
    (
        CuteSmemTensorType::get(ctx, tensor, facts.placement.clone()).into(),
        storage,
    )
}

fn decode_shared_operand(
    body: &mir::Body,
    operand: &mir::Operand,
    what: &str,
) -> TranslationResult<SharedTensorRust> {
    let ty = operand_rust_type(body, operand, what)?;
    decode_shared_tensor_receiver(&ty)
        .map_err(|error| {
            pliron::input_error_noloc!(TranslationErr::unsupported(format!(
                "invalid {what}: {error}"
            )))
        })?
        .ok_or_else(|| {
            pliron::input_error_noloc!(TranslationErr::unsupported(format!(
                "{what} must be a shared MMA tensor"
            )))
        })
}

fn decode_mma_operand(
    body: &mir::Body,
    operand: &mir::Operand,
    what: &str,
) -> TranslationResult<TiledMmaRust> {
    let ty = operand_rust_type(body, operand, what)?;
    decode_tiled_mma_receiver(&ty)
        .map_err(|error| {
            pliron::input_error_noloc!(TranslationErr::unsupported(format!(
                "invalid {what}: {error}"
            )))
        })?
        .ok_or_else(|| {
            pliron::input_error_noloc!(TranslationErr::unsupported(format!(
                "{what} must be Mxfp4TiledMma"
            )))
        })
}

#[allow(clippy::too_many_arguments)]
fn shared_carriers(
    ctx: &mut Context,
    body: &mir::Body,
    operand: &mir::Operand,
    facts: &SharedTensorRust,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    loc: &Location,
    what: &str,
) -> TranslationResult<(Value, Value, Ptr<Operation>)> {
    let (shared, after) =
        translate_operand(ctx, body, operand, value_map, block, prev, loc.clone())?;
    let (carrier, after) = aggregate_field(ctx, shared, 0, what, block, after.or(prev), loc)?;
    let (base, after) = aggregate_field(ctx, carrier, 0, what, block, Some(after), loc)?;
    let (capacity, after) = aggregate_field(ctx, carrier, 1, what, block, Some(after), loc)?;
    let (_, storage) = smem_view_type(ctx, facts);
    let (base, after) = recover_cta_shared_pointer(ctx, base, storage, what, block, after, loc)?;
    Ok((base, capacity, after))
}

#[allow(clippy::too_many_arguments)]
fn mma_lane(
    ctx: &mut Context,
    body: &mir::Body,
    operand: &mir::Operand,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    loc: &Location,
) -> TranslationResult<(Value, Ptr<Operation>)> {
    let (mma, after) = translate_operand(ctx, body, operand, value_map, block, prev, loc.clone())?;
    aggregate_field(ctx, mma, 0, "tiled MMA slice", block, after.or(prev), loc)
}

fn emit_zero_f32(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    loc: &Location,
) -> (Value, Ptr<Operation>) {
    let ty: TypeHandle = FP32Type::get(ctx).into();
    let op = Operation::new(
        ctx,
        MirFloatConstantOp::get_concrete_op_info(),
        vec![ty],
        vec![],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());
    MirFloatConstantOp::new(op).set_attr_float_value(ctx, FPSingleAttr::from(0.0_f32));
    insert(ctx, op, block, prev);
    (op.deref(ctx).get_result(0), op)
}

pub(super) fn struct_field_type(
    ctx: &Context,
    ty: TypeHandle,
    index: usize,
    what: &str,
) -> TranslationResult<TypeHandle> {
    ty.deref(ctx)
        .downcast_ref::<MirStructType>()
        .and_then(|structure| structure.field_types.get(index).copied())
        .ok_or_else(|| {
            pliron::input_error_noloc!(TranslationErr::unsupported(format!(
                "{what} has no Rust carrier field {index}"
            )))
        })
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
    if args.len() != 2 {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "shared_tensor_overlay expects base and capacity".to_string()
            )
        );
    }
    let rust_ty = destination_rust_type(body, destination, "shared_tensor_overlay")?;
    let facts = decode_shared_tensor(&rust_ty)
        .map_err(|error| {
            pliron::input_error_noloc!(TranslationErr::unsupported(format!(
                "invalid shared_tensor_overlay result: {error}"
            )))
        })?
        .ok_or_else(|| {
            pliron::input_error_noloc!(TranslationErr::unsupported(
                "shared_tensor_overlay must return SharedTensor".to_string()
            ))
        })?;
    let (base, after) =
        translate_operand(ctx, body, &args[0], value_map, block, prev, loc.clone())?;
    let (capacity, after) = translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block,
        after.or(prev),
        loc.clone(),
    )?;
    let (view, storage) = smem_view_type(ctx, &facts);
    let anchor = after.or(prev).ok_or_else(|| {
        pliron::input_error_noloc!(TranslationErr::unsupported(
            "shared_tensor_overlay arguments produced no insertion anchor".to_string()
        ))
    })?;
    let (base, anchor) = recover_cta_shared_pointer(
        ctx,
        base,
        storage,
        "shared_tensor_overlay base",
        block,
        anchor,
        &loc,
    )?;
    let operation = CuteSmemTensorOverlayOp::new(ctx, base, capacity, view);
    let op = operation.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, Some(anchor));

    let outer_ty = destination_type(ctx, body, destination, "shared_tensor_overlay")?;
    let carrier_ty = struct_field_type(ctx, outer_ty, 0, "SharedTensor")?;
    let base_ty = struct_field_type(ctx, carrier_ty, 0, "SharedTensor carrier")?;
    let (base, anchor) = pointer_cast(ctx, operation.base(ctx), base_ty, block, op, &loc)?;
    let (carrier, anchor) = build_struct_prefix(
        ctx,
        carrier_ty,
        &[base, operation.capacity(ctx)],
        block,
        anchor,
        &loc,
    )?;
    let (result, anchor) = build_struct_prefix(ctx, outer_ty, &[carrier], block, anchor, &loc)?;
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
        "shared_tensor_overlay",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_mma_slice(
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
            TranslationErr::unsupported("tiled_mma_slice expects one lane".to_string())
        );
    }
    let rust_ty = destination_rust_type(body, destination, "tiled_mma_slice")?;
    let facts = decode_tiled_mma(&rust_ty)
        .map_err(|error| {
            pliron::input_error_noloc!(TranslationErr::unsupported(format!(
                "invalid tiled_mma_slice result: {error}"
            )))
        })?
        .ok_or_else(|| {
            pliron::input_error_noloc!(TranslationErr::unsupported(
                "tiled_mma_slice must return Mxfp4TiledMma".to_string()
            ))
        })?;
    let (lane, after) =
        translate_operand(ctx, body, &args[0], value_map, block, prev, loc.clone())?;
    let operation = CuteTiledMmaSliceOp::new(ctx, lane, plan(&facts));
    let op = operation.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, after.or(prev));
    let result_ty = destination_type(ctx, body, destination, "tiled_mma_slice")?;
    let (result, anchor) = build_struct_prefix(
        ctx,
        result_ty,
        &[operation.sliced_lane(ctx)],
        block,
        op,
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
        "tiled_mma_slice",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_fragment_fill(
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
    if args.len() > 1 {
        return input_err!(
            loc,
            TranslationErr::unsupported("fragment_fill expects zero or one fill value")
        );
    }
    if !is_accumulator(&destination_rust_type(body, destination, "fragment_fill")?).map_err(
        |error| {
            pliron::input_error_noloc!(TranslationErr::unsupported(format!(
                "invalid fragment_fill result: {error}"
            )))
        },
    )? {
        return input_err!(
            loc,
            TranslationErr::unsupported("fragment_fill must return Mxf4AccumulatorTile2x8")
        );
    }
    let (fill, anchor) = if let Some(fill) = args.first() {
        let (value, after) =
            translate_operand(ctx, body, fill, value_map, block, prev, loc.clone())?;
        let anchor = after.or(prev).ok_or_else(|| {
            pliron::input_error_noloc!(TranslationErr::unsupported(
                "fragment_fill value produced no insertion anchor"
            ))
        })?;
        (value, anchor)
    } else {
        emit_zero_f32(ctx, block, prev, &loc)
    };
    let result_ty = destination_type(ctx, body, destination, "fragment_fill")?;
    let operation = CuteFragmentFillOp::new(
        ctx,
        fill,
        result_ty,
        fixed_plan(),
        CuteMmaCarrierKindAttr::Accumulator,
    );
    let op = operation.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, Some(anchor));
    finish_value(
        ctx,
        op,
        operation.fragment(ctx),
        destination,
        target,
        block,
        value_map,
        block_map,
        loc,
        "fragment_fill",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_load_scales(
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
    if args.len() != 5 {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "mma_load_scales expects MMA, A/B scales, warp-M, and warp-N"
            )
        );
    }
    let mma = decode_mma_operand(body, &args[0], "mma_load_scales MMA")?;
    let scale_a = decode_shared_operand(body, &args[1], "mma_load_scales A scales")?;
    let scale_b = decode_shared_operand(body, &args[2], "mma_load_scales B scales")?;
    if scale_a.kind != SharedTensorKind::Scale
        || scale_b.kind != SharedTensorKind::Scale
        || scale_a.role != dialect_cute::attributes::CuteTensorRoleAttr::Mkl
        || scale_b.role != dialect_cute::attributes::CuteTensorRoleAttr::Nkl
        || !is_scale_stage(&destination_rust_type(
            body,
            destination,
            "mma_load_scales",
        )?)
        .map_err(|error| {
            pliron::input_error_noloc!(TranslationErr::unsupported(format!(
                "invalid mma_load_scales result: {error}"
            )))
        })?
    {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "mma_load_scales needs M/N scale atoms and Mxf4ScaleTile128 output"
            )
        );
    }

    let (lane, after) = mma_lane(ctx, body, &args[0], block, prev, value_map, &loc)?;
    let (a_base, a_capacity, after) = shared_carriers(
        ctx,
        body,
        &args[1],
        &scale_a,
        block,
        Some(after),
        value_map,
        &loc,
        "mma_load_scales A scales",
    )?;
    let (b_base, b_capacity, after) = shared_carriers(
        ctx,
        body,
        &args[2],
        &scale_b,
        block,
        Some(after),
        value_map,
        &loc,
        "mma_load_scales B scales",
    )?;
    let (warp_m, after) = translate_operand(
        ctx,
        body,
        &args[3],
        value_map,
        block,
        Some(after),
        loc.clone(),
    )?;
    let (warp_n, after) =
        translate_operand(ctx, body, &args[4], value_map, block, after, loc.clone())?;
    let (a_view, _) = smem_view_type(ctx, &scale_a);
    let (b_view, _) = smem_view_type(ctx, &scale_b);
    let result_ty = destination_type(ctx, body, destination, "mma_load_scales")?;
    let operation = CuteMmaLoadScalesOp::new(
        ctx,
        lane,
        a_base,
        a_capacity,
        b_base,
        b_capacity,
        warp_m,
        warp_n,
        a_view,
        b_view,
        result_ty,
        plan(&mma),
    );
    let op = operation.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, after);
    finish_value(
        ctx,
        op,
        operation.scales(ctx),
        destination,
        target,
        block,
        value_map,
        block_map,
        loc,
        "mma_load_scales",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_slice_k(
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
    if args.len() != 2
        || !is_scale_stage(&operand_rust_type(
            body,
            &args[0],
            "fragment_slice_k input",
        )?)
        .unwrap_or(false)
        || !is_scale_k64(&destination_rust_type(
            body,
            destination,
            "fragment_slice_k",
        )?)
        .unwrap_or(false)
    {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "fragment_slice_k expects Mxf4ScaleTile128, K-half, and Mxf4ScalePairs128 output"
            )
        );
    }
    let (scales, after) =
        translate_operand(ctx, body, &args[0], value_map, block, prev, loc.clone())?;
    let (half, after) = translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block,
        after.or(prev),
        loc.clone(),
    )?;
    let result_ty = destination_type(ctx, body, destination, "fragment_slice_k")?;
    let operation = CuteFragmentSliceKOp::new(ctx, scales, half, result_ty, fixed_plan());
    let op = operation.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, after);
    finish_value(
        ctx,
        op,
        operation.scales(ctx),
        destination,
        target,
        block,
        value_map,
        block_map,
        loc,
        "fragment_slice_k",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_load_a(
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
            TranslationErr::unsupported("mma_load_a expects MMA, shared A, warp-M, and K-half")
        );
    }
    let mma = decode_mma_operand(body, &args[0], "mma_load_a MMA")?;
    let a = decode_shared_operand(body, &args[1], "mma_load_a shared A")?;
    if a.kind != SharedTensorKind::Data
        || a.role != dialect_cute::attributes::CuteTensorRoleAttr::Mkl
        || a.placement != mma.placement
    {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "mma_load_a needs an M-role data tile with the MMA shared layout"
            )
        );
    }
    let (lane, after) = mma_lane(ctx, body, &args[0], block, prev, value_map, &loc)?;
    let (base, capacity, after) = shared_carriers(
        ctx,
        body,
        &args[1],
        &a,
        block,
        Some(after),
        value_map,
        &loc,
        "mma_load_a shared A",
    )?;
    let (warp_m, after) = translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block,
        Some(after),
        loc.clone(),
    )?;
    let (half, after) =
        translate_operand(ctx, body, &args[3], value_map, block, after, loc.clone())?;
    let (view, _) = smem_view_type(ctx, &a);
    let result_ty = destination_type(ctx, body, destination, "mma_load_a")?;
    let operation = CuteMmaLoadAOp::new(
        ctx,
        lane,
        base,
        capacity,
        warp_m,
        half,
        view,
        result_ty,
        plan(&mma),
    );
    let op = operation.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, after);
    finish_value(
        ctx,
        op,
        operation.fragment(ctx),
        destination,
        target,
        block,
        value_map,
        block_map,
        loc,
        "mma_load_a",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_partition_b(
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
                "mma_partition_b expects MMA, shared B, warp-N, and K-half"
            )
        );
    }
    let mma = decode_mma_operand(body, &args[0], "mma_partition_b MMA")?;
    let b = decode_shared_operand(body, &args[1], "mma_partition_b shared B")?;
    let result_facts = decode_b_tile(&destination_rust_type(
        body,
        destination,
        "mma_partition_b",
    )?)
    .map_err(|error| {
        pliron::input_error_noloc!(TranslationErr::unsupported(format!(
            "invalid mma_partition_b result: {error}"
        )))
    })?
    .ok_or_else(|| {
        pliron::input_error_noloc!(TranslationErr::unsupported(
            "mma_partition_b must return Mxf4BTileK64"
        ))
    })?;
    if b.kind != SharedTensorKind::Data
        || b.role != dialect_cute::attributes::CuteTensorRoleAttr::Nkl
        || b.placement != mma.placement
        || result_facts != mma
    {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "mma_partition_b needs one N-role tile and matching MMA/B result layouts"
            )
        );
    }
    let (_lane, after) = mma_lane(ctx, body, &args[0], block, prev, value_map, &loc)?;
    let (base, capacity, after) = shared_carriers(
        ctx,
        body,
        &args[1],
        &b,
        block,
        Some(after),
        value_map,
        &loc,
        "mma_partition_b shared B",
    )?;
    let (warp_n, after) = translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block,
        Some(after),
        loc.clone(),
    )?;
    let (half, after) =
        translate_operand(ctx, body, &args[3], value_map, block, after, loc.clone())?;
    let (view, _) = smem_view_type(ctx, &b);
    let operation = CuteMmaPartitionBOp::new(ctx, base, capacity, warp_n, half, view, plan(&mma));
    let op = operation.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, after);

    let result_ty = destination_type(ctx, body, destination, "mma_partition_b")?;
    let base_ty = struct_field_type(ctx, result_ty, 0, "Mxf4BTileK64")?;
    let (base, anchor) = pointer_cast(ctx, operation.base(ctx), base_ty, block, op, &loc)?;
    let (result, anchor) = build_struct_prefix(
        ctx,
        result_ty,
        &[
            base,
            operation.capacity(ctx),
            operation.warp_n(ctx),
            operation.k_half(ctx),
        ],
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
        "mma_partition_b",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_tiled_gemm(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    _destination: &mir::Place,
    target: &Option<usize>,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 5 {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "tiled_gemm expects accumulator, MMA, A, lazy B, and scales"
            )
        );
    }
    let acc_ty = operand_rust_type(body, &args[0], "tiled_gemm accumulator")?;
    let mma = decode_mma_operand(body, &args[1], "tiled_gemm MMA")?;
    let b_facts = decode_b_tile(&operand_rust_type(body, &args[3], "tiled_gemm lazy B")?)
        .map_err(|error| {
            pliron::input_error_noloc!(TranslationErr::unsupported(format!(
                "invalid tiled_gemm lazy B: {error}"
            )))
        })?
        .ok_or_else(|| {
            pliron::input_error_noloc!(TranslationErr::unsupported("tiled_gemm needs Mxf4BTileK64"))
        })?;
    if !is_accumulator(&acc_ty).unwrap_or(false)
        || !is_scale_k64(&operand_rust_type(body, &args[4], "tiled_gemm scales")?).unwrap_or(false)
        || b_facts != mma
    {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "tiled_gemm accumulator, scales, or lazy-B layout is invalid"
            )
        );
    }

    let (acc_pointer, after) =
        translate_operand(ctx, body, &args[0], value_map, block, prev, loc.clone())?;
    let acc_pointee = {
        let ty = acc_pointer.get_type(ctx);
        let ty = ty.deref(ctx);
        let Some(pointer) = ty.downcast_ref::<MirPtrType>() else {
            return input_err!(
                loc,
                TranslationErr::unsupported(
                    "tiled_gemm accumulator receiver must be a mutable reference"
                )
            );
        };
        if !pointer.is_mutable {
            return input_err!(
                loc,
                TranslationErr::unsupported("tiled_gemm accumulator receiver must be mutable")
            );
        }
        pointer.pointee
    };
    let (accumulator, after) =
        load_through(ctx, acc_pointer, acc_pointee, block, after.or(prev), &loc);
    let (lane, after) = mma_lane(ctx, body, &args[1], block, Some(after), value_map, &loc)?;
    let (a, after) = translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block,
        Some(after),
        loc.clone(),
    )?;
    let (b, after) = translate_operand(ctx, body, &args[3], value_map, block, after, loc.clone())?;
    let (b_base, after) = aggregate_field(ctx, b, 0, "tiled_gemm lazy B", block, after, &loc)?;
    let (b_capacity, after) =
        aggregate_field(ctx, b, 1, "tiled_gemm lazy B", block, Some(after), &loc)?;
    let (b_warp_n, after) =
        aggregate_field(ctx, b, 2, "tiled_gemm lazy B", block, Some(after), &loc)?;
    let (b_half, after) =
        aggregate_field(ctx, b, 3, "tiled_gemm lazy B", block, Some(after), &loc)?;
    let b_shared = SharedTensorRust {
        kind: SharedTensorKind::Data,
        role: dialect_cute::attributes::CuteTensorRoleAttr::Nkl,
        placement: mma.placement.clone(),
    };
    let (b_view, b_storage) = smem_view_type(ctx, &b_shared);
    let (b_base, after) = recover_cta_shared_pointer(
        ctx,
        b_base,
        b_storage,
        "tiled_gemm lazy B base",
        block,
        after,
        &loc,
    )?;
    let (scales, after) = translate_operand(
        ctx,
        body,
        &args[4],
        value_map,
        block,
        Some(after),
        loc.clone(),
    )?;
    let operation = CuteTiledGemmOp::new(
        ctx,
        lane,
        a,
        b_base,
        b_capacity,
        b_warp_n,
        b_half,
        scales,
        accumulator,
        b_view,
        plan(&mma),
    );
    let op = operation.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, after);
    let store = Operation::new(
        ctx,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![acc_pointer, operation.accumulator(ctx)],
        vec![],
        0,
    );
    store.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, store, block, Some(op));
    finish(ctx, store, target, block_map, loc, "tiled_gemm")
}
