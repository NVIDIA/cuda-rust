/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! dialect-cute op emission.
//!
//! One function per recognized `cute-rs` API entry point. Per-thread tile
//! calls become one zero-result `cute.copy`; `assume` becomes one SSA
//! fact. Cooperative boundaries become typed semantic operations for the
//! selected backend continuation. Nothing here rebuilds a layout from scalar
//! arithmetic.

use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue::translate_operand;
use crate::translator::terminator::helpers::emit_goto;
use crate::translator::types::translate_type;
use crate::translator::values::ValueMap;
use dialect_cute::gemm_tma_ops::{CuteTmaCopy2dOp, CuteTmaGmemViewOp, CuteTmaSmemViewOp};
use dialect_cute::layout::{ComposedLayout, Layout};
use dialect_cute::ops::{CuteAssumeDivOp, CuteCopyG2SOp, CuteCopyOp};
use dialect_mir::attributes::{FieldIndexAttr, MirCastKindAttr};
use dialect_mir::ops::{MirCastOp, MirExtractFieldOp, MirFieldAddrOp, MirLoadOp, MirPtrOffsetOp};
use dialect_mir::types::{MirPtrType, MirSliceType, MirStructType, address_space};
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::Location;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;
use rustc_public::mir;

use super::layout::{assume_divisor, tile_elem_and_count};
use super::smem_mma::emit::aggregate_field;
use super::static_config::{
    decode_copy_g2s_callee, decode_copy_tma_callee, decode_copy_tma_s2g_callee,
    decode_load_matrix_callee,
};

/// Take the address of field `index` of a by-reference struct argument.
///
/// The result pointer keeps the carrier's mutability and address space; the
/// field's own type comes from the translated struct, so this helper never
/// guesses an offset.
pub(super) fn struct_field_addr(
    ctx: &mut Context,
    struct_ptr: Value,
    index: usize,
    what: &str,
    block_ptr: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    loc: &Location,
) -> TranslationResult<(Value, TypeHandle, Ptr<Operation>)> {
    let looked_up = {
        let ty = struct_ptr.get_type(ctx);
        let ty_ref = ty.deref(ctx);
        ty_ref.downcast_ref::<MirPtrType>().and_then(|pointer| {
            let pointee = pointer.pointee;
            let pointee_ref = pointee.deref(ctx);
            pointee_ref
                .downcast_ref::<MirStructType>()
                .and_then(|strct| strct.field_types.get(index).copied())
                .map(|field_ty| (field_ty, pointer.is_mutable, pointer.address_space))
        })
    };
    let Some((field_ty, is_mutable, address_space)) = looked_up else {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "{what} must be a reference to a cute-rs carrier struct with field {index}"
            ))
        );
    };
    let field_ptr_ty: TypeHandle = MirPtrType::get(ctx, field_ty, is_mutable, address_space).into();
    // Go through the one construction door so `aggregate_ty` is stamped;
    // the verifier rejects a `mir.field_addr` without it.
    let op = MirFieldAddrOp::build(ctx, struct_ptr, field_ptr_ty, index as u32)?;
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block_ptr, prev);
    Ok((op.deref(ctx).get_result(0), field_ty, op))
}

/// Load one value through a typed pointer.
pub(super) fn load_through(
    ctx: &mut Context,
    pointer: Value,
    pointee: TypeHandle,
    block_ptr: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    loc: &Location,
) -> (Value, Ptr<Operation>) {
    let op = Operation::new(
        ctx,
        MirLoadOp::get_concrete_op_info(),
        vec![pointee],
        vec![pointer],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());
    MirLoadOp::new(op).set_volatile(ctx, false);
    insert(ctx, op, block_ptr, prev);
    (op.deref(ctx).get_result(0), op)
}

/// Recover the CTA-shared address-space proof at a recognized unsafe API
/// boundary.
///
/// Rust raw-pointer fields do not carry CUDA address spaces. Consequently,
/// constructing `SmemTile { base, .. }` normalizes an AS3 pointer to the
/// generic AS0 type of its `base: *mut T` field. That normalization is sound
/// for an ordinary Rust aggregate, but `copy_tma_2d` has a stronger contract:
/// its destination must be CTA-local shared memory. The unsafe call is the
/// precise place where the caller asserts that contract, so recover AS3 here
/// instead of teaching the type translator that every generic pointer is
/// shared.
///
/// Already-shared pointers pass through unchanged. Any future carrier that
/// reaches this boundary with a concrete non-shared address space fails
/// closed rather than being silently reinterpreted.
pub(super) fn recover_cta_shared_pointer(
    ctx: &mut Context,
    pointer: Value,
    expected_pointee: TypeHandle,
    what: &str,
    block_ptr: Ptr<BasicBlock>,
    prev: Ptr<Operation>,
    loc: &Location,
) -> TranslationResult<(Value, Ptr<Operation>)> {
    let pointer_info = {
        let ty = pointer.get_type(ctx);
        let ty_ref = ty.deref(ctx);
        ty_ref
            .downcast_ref::<MirPtrType>()
            .map(|ptr| (ptr.pointee, ptr.is_mutable, ptr.address_space))
    };
    let Some((pointee, is_mutable, source_address_space)) = pointer_info else {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!("{what} must be a MIR pointer"))
        );
    };
    if pointee != expected_pointee || !is_mutable {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "{what} must be a mutable pointer to the decoded TMA element type"
            ))
        );
    }
    if source_address_space == address_space::SHARED {
        return Ok((pointer, prev));
    }
    if source_address_space != address_space::GENERIC {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "{what} must be generic or CTA-shared memory, found address space {source_address_space}"
            ))
        );
    }

    let shared_ty: TypeHandle =
        MirPtrType::get(ctx, expected_pointee, true, address_space::SHARED).into();
    let cast = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![shared_ty],
        vec![pointer],
        vec![],
        0,
    );
    cast.deref_mut(ctx).set_loc(loc.clone());
    MirCastOp::new(cast).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
    insert(ctx, cast, block_ptr, Some(prev));
    Ok((cast.deref(ctx).get_result(0), cast))
}

/// Load field `index` of a by-reference carrier struct. When the field is
/// itself a single-field struct (`LeadingDim<D>` is `repr(transparent)` but
/// still translates as a struct), the load reads through that wrapper.
fn struct_field_load(
    ctx: &mut Context,
    struct_ptr: Value,
    index: usize,
    what: &str,
    block_ptr: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    loc: &Location,
) -> TranslationResult<(Value, Ptr<Operation>)> {
    let (mut pointer, mut pointee, mut prev_op) =
        struct_field_addr(ctx, struct_ptr, index, what, block_ptr, prev, loc)?;
    let inner_field = {
        let pointee_ref = pointee.deref(ctx);
        pointee_ref
            .downcast_ref::<MirStructType>()
            .and_then(|wrapper| (wrapper.field_types.len() == 1).then(|| wrapper.field_types[0]))
    };
    if inner_field.is_some() {
        let (inner_ptr, inner_ty, inner_op) =
            struct_field_addr(ctx, pointer, 0, what, block_ptr, Some(prev_op), loc)?;
        pointer = inner_ptr;
        pointee = inner_ty;
        prev_op = inner_op;
    }
    let (value, op) = load_through(ctx, pointer, pointee, block_ptr, Some(prev_op), loc);
    Ok((value, op))
}

/// Extract field `index` from a by-value aggregate (the tile-coordinate
/// tuple), producing a `u64`-typed value.
fn tuple_field(
    ctx: &mut Context,
    aggregate: Value,
    index: usize,
    result_ty: TypeHandle,
    block_ptr: Ptr<BasicBlock>,
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
    MirExtractFieldOp::new(op).set_attr_index(ctx, FieldIndexAttr(index as u32));
    insert(ctx, op, block_ptr, prev);
    (op.deref(ctx).get_result(0), op)
}

/// Keep one recognized TMA transfer as three directly connected semantic
/// operations. The view values never enter a Rust local or function ABI:
/// they exist only between this call boundary and the selected backend.
#[allow(clippy::too_many_arguments)]
fn emit_tma_copy_2d_views(
    ctx: &mut Context,
    descriptor: Value,
    smem_base: Value,
    smem_capacity: Value,
    tile_row: Value,
    tile_column: Value,
    barrier: Value,
    element: TypeHandle,
    smem_layout: &ComposedLayout,
    smem_alignment_bytes: u64,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    loc: &Location,
) -> Ptr<Operation> {
    let source = CuteTmaGmemViewOp::new(ctx, descriptor, element, smem_layout.clone());
    let source_op = source.get_operation();
    source_op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, source_op, block, prev);
    let source = source_op.deref(ctx).get_result(0);

    let destination = CuteTmaSmemViewOp::new(
        ctx,
        smem_base,
        smem_capacity,
        element,
        smem_layout.clone(),
        smem_alignment_bytes,
    );
    let destination_op = destination.get_operation();
    destination_op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, destination_op, block, Some(source_op));
    let destination = destination_op.deref(ctx).get_result(0);

    let copy = CuteTmaCopy2dOp::new(ctx, source, destination, tile_row, tile_column, barrier);
    let copy_op = copy.get_operation();
    copy_op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, copy_op, block, Some(destination_op));
    copy_op
}

/// `cute_rs::atom::cooperative::copy_g2s::<Atom, ThrL, ValL, TileL, SmemL, T>`
///
/// The importer's whole job is flattening: read the validated static
/// configuration from the substs, read the runtime facts out of the carrier
/// structs, and emit ONE `cute.copy_g2s` whose nine operands carry every
/// runtime value in a fixed order:
///
/// ```text
/// &GmemMatrix ──field loads──► base, rows, cols, leading_dim
/// (r, c)      ──extracts────► tile_row, tile_column
/// &SmemTile   ──field loads──► smem base, capacity
/// tidx        ──as is───────► thread index
/// ```
///
/// Address math, predication, and target instruction selection happen later
/// in the selected backend, which re-reads the same typed attributes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_copy_g2s(
    ctx: &mut Context,
    body: &mir::Body,
    func: &mir::Operand,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 4 {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "copy_g2s expects 4 runtime arguments, got {}",
                args.len()
            ))
        );
    }
    let config = match decode_copy_g2s_callee(func) {
        Ok(config) => config,
        Err(error) => {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "copy_g2s static configuration is invalid: {error}"
                ))
            );
        }
    };
    let elem = translate_type(ctx, &config.element_type)?;

    let (gmem_ref, prev) = translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    let prev = prev.or(prev_op);
    let (coord_val, prev) =
        translate_operand(ctx, body, &args[1], value_map, block_ptr, prev, loc.clone())?;
    let (smem_ref, prev) =
        translate_operand(ctx, body, &args[2], value_map, block_ptr, prev, loc.clone())?;
    let (tidx, prev) =
        translate_operand(ctx, body, &args[3], value_map, block_ptr, prev, loc.clone())?;

    let (gmem_base, prev) =
        struct_field_load(ctx, gmem_ref, 0, "copy_g2s source", block_ptr, prev, &loc)?;
    let prev = Some(prev);
    let (rows, prev) =
        struct_field_load(ctx, gmem_ref, 1, "copy_g2s source", block_ptr, prev, &loc)?;
    let prev = Some(prev);
    let (cols, prev) =
        struct_field_load(ctx, gmem_ref, 2, "copy_g2s source", block_ptr, prev, &loc)?;
    let prev = Some(prev);
    let (leading_dim, prev) =
        struct_field_load(ctx, gmem_ref, 3, "copy_g2s source", block_ptr, prev, &loc)?;
    let prev = Some(prev);

    let coordinate_ty = rows.get_type(ctx);
    let (tile_row, prev) = tuple_field(ctx, coord_val, 0, coordinate_ty, block_ptr, prev, &loc);
    let (tile_col, prev) = tuple_field(
        ctx,
        coord_val,
        1,
        coordinate_ty,
        block_ptr,
        Some(prev),
        &loc,
    );

    let (smem_base, prev) = struct_field_load(
        ctx,
        smem_ref,
        0,
        "copy_g2s destination",
        block_ptr,
        Some(prev),
        &loc,
    )?;
    let prev = Some(prev);
    let (capacity, prev) = struct_field_load(
        ctx,
        smem_ref,
        1,
        "copy_g2s destination",
        block_ptr,
        prev,
        &loc,
    )?;

    let copy = CuteCopyG2SOp::new(
        ctx,
        [
            gmem_base,
            rows,
            cols,
            leading_dim,
            tile_row,
            tile_col,
            smem_base,
            capacity,
            tidx,
        ],
        config.atom.bytes,
        config.thread_layout.clone(),
        config.value_layout.clone(),
        config.tile_layout.clone(),
        config.smem_layout.clone(),
        config.leading_dim_divisor,
        elem,
    );
    let copy_op = copy.get_operation();
    copy_op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, copy_op, block_ptr, Some(prev));

    let Some(target_idx) = target else {
        return input_err!(
            loc,
            TranslationErr::unsupported("copy_g2s call without target not supported".to_string())
        );
    };
    Ok(emit_goto(ctx, *target_idx, copy_op, block_map, loc))
}

/// `cute_rs::nvgpu::warp::load_matrix_{a,b}::<SmemL>(src, warp_tile, lane) -> Frag`
///
/// Flattening only, like `copy_g2s`: the shared base comes out of the
/// carrier, the tuple splits into two coordinates, and the destination
/// local's slot receives the fragment. One zero-result `cute.ldmatrix`
/// carries the layout facts; the selected backend computes per-lane addresses.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_load_matrix(
    ctx: &mut Context,
    body: &mir::Body,
    func: &mir::Operand,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    role: dialect_cute::attributes::CuteMatrixRoleAttr,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 3 {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "load_matrix expects 3 runtime arguments, got {}",
                args.len()
            ))
        );
    }
    let config = match decode_load_matrix_callee(func) {
        Ok(config) => config,
        Err(error) => {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "load_matrix static configuration is invalid: {error}"
                ))
            );
        }
    };
    let (smem_ref, prev) = translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    let prev = prev.or(prev_op);
    let (coord_val, prev) =
        translate_operand(ctx, body, &args[1], value_map, block_ptr, prev, loc.clone())?;
    let (lane, prev) =
        translate_operand(ctx, body, &args[2], value_map, block_ptr, prev, loc.clone())?;

    let (smem_base, prev) = struct_field_load(
        ctx,
        smem_ref,
        0,
        "load_matrix source",
        block_ptr,
        prev,
        &loc,
    )?;
    let elem: TypeHandle = smem_base
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<MirPtrType>()
        .map(|ptr| ptr.pointee)
        .unwrap_or_else(|| dialect_mir::types::MirFP16Type::get(ctx).into());
    let prev = Some(prev);

    let coordinate_ty: TypeHandle = pliron::builtin::types::IntegerType::get(
        ctx,
        64,
        pliron::builtin::types::Signedness::Unsigned,
    )
    .into();
    let (warp_tile_r, prev) = tuple_field(ctx, coord_val, 0, coordinate_ty, block_ptr, prev, &loc);
    let (warp_tile_c, prev) = tuple_field(
        ctx,
        coord_val,
        1,
        coordinate_ty,
        block_ptr,
        Some(prev),
        &loc,
    );

    if !destination.projection.is_empty() {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "load_matrix destination with projections not supported".to_string()
            )
        );
    }
    let Some(slot) = value_map.get_slot(destination.local) else {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "load_matrix destination local has no memory slot".to_string()
            )
        );
    };

    let load = dialect_cute::ops::CuteLdmatrixOp::new(
        ctx,
        [smem_base, warp_tile_r, warp_tile_c, lane, slot],
        role,
        config.smem_layout.clone(),
        elem,
    );
    let load_op = load.get_operation();
    load_op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, load_op, block_ptr, Some(prev));

    let Some(target_idx) = target else {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "load_matrix call without target not supported".to_string()
            )
        );
    };
    Ok(emit_goto(ctx, *target_idx, load_op, block_map, loc))
}

/// Regression tests for exact source identity and semantic type preservation.
#[cfg(test)]
// Keep these address-space and TMA-view tests next to the helper boundary they
// exercise; the remaining emitters below intentionally follow frontend call order.
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use dialect_cute::types::CuteTmaViewType;
    use dialect_mir::ops::MirUndefOp;
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::common_traits::Verify;

    /// Create a typed pointer block argument for importer boundary tests.
    fn pointer_value(
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        pointee: TypeHandle,
        pointer_address_space: u32,
    ) -> (Value, Ptr<Operation>) {
        let pointer_ty: TypeHandle =
            MirPtrType::get(ctx, pointee, true, pointer_address_space).into();
        let undef = MirUndefOp::new(ctx, pointer_ty).get_operation();
        insert(ctx, undef, block, None);
        (undef.deref(ctx).get_result(0), undef)
    }

    /// Read the address space from a pointer-valued test operand.
    fn pointer_address_space(ctx: &Context, value: Value) -> u32 {
        value
            .get_type(ctx)
            .deref(ctx)
            .downcast_ref::<MirPtrType>()
            .unwrap()
            .address_space
    }

    /// Insert an undefined carrier after the test's current operation.
    fn undef_after(
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        ty: TypeHandle,
        prev: Option<Ptr<Operation>>,
    ) -> (Value, Ptr<Operation>) {
        let undef = MirUndefOp::new(ctx, ty).get_operation();
        insert(ctx, undef, block, prev);
        (undef.deref(ctx).get_result(0), undef)
    }

    /// Check that global stores retain the raw mutable pointer authority.
    #[test]
    fn global_store_destination_keeps_raw_mut_pointer_authority() {
        use dialect_mir::{attributes::MirPointerKindAuthorityAttr, types::MirPointerKind};
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        let element: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let generic: TypeHandle = crate::translator::facts::mint_ptr_type(
            &mut ctx,
            element,
            address_space::GENERIC,
            crate::translator::facts::pointer_origin_of_raw_mutability(mir::Mutability::Mut),
        )
        .into();
        let block = BasicBlock::new(&mut ctx, None, vec![generic]);
        let value = block.deref(&ctx).arguments().next().unwrap();
        let pointer = generic
            .deref(&ctx)
            .downcast_ref::<MirPtrType>()
            .unwrap()
            .clone();
        let global = global_store_pointer_type(&mut ctx, &pointer);
        let op = Operation::new(
            &mut ctx,
            MirCastOp::get_concrete_op_info(),
            vec![global],
            vec![value],
            vec![],
            0,
        );
        let cast = MirCastOp::new(op);
        cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
        cast.set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::RawAddress);
        assert_eq!(
            global
                .deref(&ctx)
                .downcast_ref::<MirPtrType>()
                .unwrap()
                .kind,
            MirPointerKind::RawMut
        );
        assert!(cast.verify(&ctx).is_ok());
    }

    /// Check that TMA accepts only generic or CTA-shared pointer provenance.
    #[test]
    fn tma_boundary_recovers_only_generic_or_existing_cta_shared_pointers() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        let block = BasicBlock::new(&mut ctx, None, vec![]);
        let elem: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();

        let (generic, generic_anchor) =
            pointer_value(&mut ctx, block, elem, address_space::GENERIC);
        let (recovered, cast) = recover_cta_shared_pointer(
            &mut ctx,
            generic,
            elem,
            "test destination",
            block,
            generic_anchor,
            &Location::Unknown,
        )
        .unwrap();
        assert_eq!(
            pointer_address_space(&ctx, recovered),
            address_space::SHARED
        );
        assert!(Operation::get_opid(cast, &ctx) == MirCastOp::get_opid_static());

        let (shared, shared_anchor) = pointer_value(&mut ctx, block, elem, address_space::SHARED);
        let (unchanged, unchanged_anchor) = recover_cta_shared_pointer(
            &mut ctx,
            shared,
            elem,
            "test destination",
            block,
            shared_anchor,
            &Location::Unknown,
        )
        .unwrap();
        assert_eq!(unchanged, shared);
        assert_eq!(unchanged_anchor, shared_anchor);

        let (global, global_anchor) = pointer_value(&mut ctx, block, elem, address_space::GLOBAL);
        let error = recover_cta_shared_pointer(
            &mut ctx,
            global,
            elem,
            "test destination",
            block,
            global_anchor,
            &Location::Unknown,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("generic or CTA-shared memory"), "{error}");
    }

    /// Check that TMA import connects the global source and shared destination views.
    #[test]
    fn tma_importer_connects_direct_global_and_shared_views() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        dialect_cute::register(&mut ctx);
        let block = BasicBlock::new(&mut ctx, None, vec![]);
        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let descriptor_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, u8_ty, false).into();
        let smem_ty: TypeHandle =
            MirPtrType::get(&mut ctx, u8_ty, true, address_space::SHARED).into();
        let barrier_ty: TypeHandle =
            MirPtrType::get(&mut ctx, u64_ty, true, address_space::SHARED).into();

        let (descriptor, prev) = undef_after(&mut ctx, block, descriptor_ty, None);
        let (smem_base, prev) = undef_after(&mut ctx, block, smem_ty, Some(prev));
        let (capacity, prev) = undef_after(&mut ctx, block, u64_ty, Some(prev));
        let (tile_row, prev) = undef_after(&mut ctx, block, u64_ty, Some(prev));
        let (tile_column, prev) = undef_after(&mut ctx, block, u64_ty, Some(prev));
        let (barrier, prev) = undef_after(&mut ctx, block, barrier_ty, Some(prev));
        let layout = ComposedLayout::new(
            dialect_cute::layout::Swizzle::new(2, 4, 3),
            0,
            "(128,64):(64,1)".parse().unwrap(),
            dialect_cute::layout::OffsetUnit::Elements,
        )
        .unwrap();

        let copy_op = emit_tma_copy_2d_views(
            &mut ctx,
            descriptor,
            smem_base,
            capacity,
            tile_row,
            tile_column,
            barrier,
            u8_ty,
            &layout,
            512,
            block,
            Some(prev),
            &Location::Unknown,
        );
        let copy = CuteTmaCopy2dOp::wrap(copy_op);
        assert!(copy.verify(&ctx).is_ok());

        let source_op = copy.source(&ctx).defining_op().unwrap();
        let destination_op = copy.destination(&ctx).defining_op().unwrap();
        assert!(Operation::get_opid(source_op, &ctx) == CuteTmaGmemViewOp::get_opid_static());
        assert!(Operation::get_opid(destination_op, &ctx) == CuteTmaSmemViewOp::get_opid_static());
        assert_eq!(
            CuteTmaGmemViewOp::wrap(source_op).descriptor(&ctx),
            descriptor
        );
        let destination = CuteTmaSmemViewOp::wrap(destination_op);
        assert_eq!(destination.base(&ctx), smem_base);
        assert_eq!(destination.capacity(&ctx), capacity);

        let destination_type = copy.destination(&ctx).get_type(&ctx);
        let destination_type = destination_type.deref(&ctx);
        let destination_view = destination_type.downcast_ref::<CuteTmaViewType>().unwrap();
        assert_eq!(destination_view.tensor_view(&ctx).unwrap().alignment.0, 512);
    }
}

/// Insert `op` after `prev` (or at block front when the call is the block's
/// first operation).
fn insert(
    ctx: &mut Context,
    op: Ptr<Operation>,
    block_ptr: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
) {
    if let Some(prev) = prev {
        op.insert_after(ctx, prev);
    } else {
        op.insert_at_front(block_ptr, ctx);
    }
}

/// Extract the data pointer (field 0 of the `{ptr, len}` fat pointer) from a
/// slice value, mirroring `apply_deref_projection`'s slice arm.
fn slice_data_ptr(
    ctx: &mut Context,
    slice_val: Value,
    elem: TypeHandle,
    block_ptr: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    loc: &Location,
) -> (Value, Ptr<Operation>) {
    let ptr_ty: TypeHandle = MirPtrType::get_generic(ctx, elem, false).into();
    let op = Operation::new(
        ctx,
        MirExtractFieldOp::get_concrete_op_info(),
        vec![ptr_ty],
        vec![slice_val],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());
    MirExtractFieldOp::new(op).set_attr_index(ctx, FieldIndexAttr(0));
    insert(ctx, op, block_ptr, prev);
    (op.deref(ctx).get_result(0), op)
}

/// `base + idx` (element-scaled, like GEP).
fn ptr_offset(
    ctx: &mut Context,
    base: Value,
    idx: Value,
    block_ptr: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    loc: &Location,
) -> (Value, Ptr<Operation>) {
    let res_ty = base.get_type(ctx);
    let op = Operation::new(
        ctx,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![res_ty],
        vec![base, idx],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block_ptr, prev);
    (op.deref(ctx).get_result(0), op)
}

/// Reinterpret a pointer's pointee as `new_pointee` (PtrToPtr), keeping its
/// mutability and address space.
fn cast_pointee(
    ctx: &mut Context,
    ptr: Value,
    new_pointee: TypeHandle,
    block_ptr: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    loc: &Location,
) -> TranslationResult<(Value, Ptr<Operation>)> {
    let (is_mut, addrspace) = {
        let ty = ptr.get_type(ctx);
        let ty_ref = ty.deref(ctx);
        let Some(p) = ty_ref.downcast_ref::<MirPtrType>() else {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(
                    "cute-rs tile operand expected to be a pointer".to_string()
                )
            );
        };
        (p.is_mutable, p.address_space)
    };
    let res_ty: TypeHandle = MirPtrType::get(ctx, new_pointee, is_mut, addrspace).into();
    let op = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![res_ty],
        vec![ptr],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());
    MirCastOp::new(op).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
    insert(ctx, op, block_ptr, prev);
    Ok((op.deref(ctx).get_result(0), op))
}

/// Emit the `cute.copy` plus the mandatory goto epilogue.
#[allow(clippy::too_many_arguments)]
fn finish_with_copy(
    ctx: &mut Context,
    src: Value,
    dst: Value,
    n: u64,
    elem: TypeHandle,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    what: &str,
) -> TranslationResult<Ptr<Operation>> {
    let copy = CuteCopyOp::new(ctx, src, dst, Layout::contiguous(n as i64), elem);
    let copy_op = copy.get_operation();
    copy_op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, copy_op, block_ptr, prev);

    let Some(target_idx) = target else {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!("{what} call without target not supported"))
        );
    };
    Ok(emit_goto(ctx, *target_idx, copy_op, block_map, loc))
}

/// `cute_rs::atom::vector::load_tile::<T, N>(src: &[T], idx: usize) -> [T; N]`
///
/// The returned tile is written directly into the destination local's slot
/// (memory-to-memory `cute.copy`), so there is no SSA result to record.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_load_tile(
    ctx: &mut Context,
    body: &mir::Body,
    func: &mir::Operand,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 2 {
        return input_err!(
            loc,
            TranslationErr::unsupported("load_tile expects exactly 2 arguments".to_string())
        );
    }
    let (elem, n) = tile_elem_and_count(ctx, func, &loc)?;

    let (slice_val, prev) = translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    {
        let ty = slice_val.get_type(ctx);
        if ty.deref(ctx).downcast_ref::<MirSliceType>().is_none() {
            return input_err!(
                loc,
                TranslationErr::unsupported("load_tile source is not a slice value".to_string())
            );
        }
    }
    let (idx_val, prev) =
        translate_operand(ctx, body, &args[1], value_map, block_ptr, prev, loc.clone())?;

    let (data_ptr, prev) = slice_data_ptr(ctx, slice_val, elem, block_ptr, prev, &loc);
    let (src, prev) = ptr_offset(ctx, data_ptr, idx_val, block_ptr, Some(prev), &loc);

    // The destination local's alloca slot IS the tile destination.
    if !destination.projection.is_empty() {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "load_tile destination with projections not supported".to_string()
            )
        );
    }
    let Some(slot) = value_map.get_slot(destination.local) else {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "load_tile destination local has no memory slot".to_string()
            )
        );
    };
    let (dst, prev) = cast_pointee(ctx, slot, elem, block_ptr, Some(prev), &loc)?;

    finish_with_copy(
        ctx,
        src,
        dst,
        n,
        elem,
        target,
        block_ptr,
        Some(prev),
        block_map,
        loc,
        "load_tile",
    )
}

/// `cute_rs::atom::vector::assume::<D>(x: usize) -> usize`
///
/// The result equals the input at runtime, but the SSA edge carries `D`:
///
/// ```text
/// x ──> cute.assume_div {divisor = D} ──> y ──> later address use
///                    M3 reads this fact ──┘
/// ```
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_assume_div(
    ctx: &mut Context,
    body: &mir::Body,
    func: &mir::Operand,
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
            loc,
            TranslationErr::unsupported("assume expects exactly 1 argument".to_string())
        );
    }
    let divisor = assume_divisor(func, &loc)?;
    let (val, prev) = translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    let assume = CuteAssumeDivOp::new(ctx, val, divisor);
    let assume_op = assume.get_operation();
    assume_op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, assume_op, block_ptr, prev.or(prev_op));
    let assumed = assume_op.deref(ctx).get_result(0);
    if !destination.projection.is_empty() {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "assume destination with projections not supported".to_string()
            )
        );
    }
    let goto_prev = value_map
        .store_local(ctx, destination.local, assumed, block_ptr, Some(assume_op))
        .unwrap_or(assume_op);
    let Some(target_idx) = target else {
        return input_err!(
            loc,
            TranslationErr::unsupported("assume call without target not supported".to_string())
        );
    };
    Ok(emit_goto(ctx, *target_idx, goto_prev, block_map, loc))
}

/// `cute_rs::numeric::__compiler::cvt_f32_to_bf16(value: f32) -> u16`
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_cvt_f32_to_bf16(
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
    if args.len() != 1 || !destination.projection.is_empty() {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "cvt_f32_to_bf16 expects 1 argument and direct destination".to_string()
            )
        );
    }
    let (val, prev) = translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    let op = dialect_cute::tensor_ops::CuteConvertF32ToBf16Op::new(ctx, val).get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block_ptr, prev.or(prev_op));
    let res = op.deref(ctx).get_result(0);
    let goto_prev = value_map
        .store_local(ctx, destination.local, res, block_ptr, Some(op))
        .unwrap_or(op);
    let Some(target_idx) = target else {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "cvt_f32_to_bf16 call without target not supported".to_string()
            )
        );
    };
    Ok(emit_goto(ctx, *target_idx, goto_prev, block_map, loc))
}

/// `cute_rs::numeric::__compiler::cvt_f32_to_e4m3(value: f32) -> u8`
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_cvt_f32_to_e4m3(
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
    if args.len() != 1 || !destination.projection.is_empty() {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "cvt_f32_to_e4m3 expects 1 argument and direct destination".to_string()
            )
        );
    }
    let (val, prev) = translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    let op = dialect_cute::tensor_ops::CuteConvertF32ToE4m3Op::new(ctx, val).get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block_ptr, prev.or(prev_op));
    let res = op.deref(ctx).get_result(0);
    let goto_prev = value_map
        .store_local(ctx, destination.local, res, block_ptr, Some(op))
        .unwrap_or(op);
    let Some(target_idx) = target else {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "cvt_f32_to_e4m3 call without target not supported".to_string()
            )
        );
    };
    Ok(emit_goto(ctx, *target_idx, goto_prev, block_map, loc))
}

/// `cute_rs::numeric::__compiler::cvt_e4m3_to_f32(bits: u8) -> f32`
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_cvt_e4m3_to_f32(
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
    if args.len() != 1 || !destination.projection.is_empty() {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "cvt_e4m3_to_f32 expects 1 argument and direct destination".to_string()
            )
        );
    }
    let (val, prev) = translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    let op = dialect_cute::tensor_ops::CuteConvertE4m3ToF32Op::new(ctx, val).get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block_ptr, prev.or(prev_op));
    let res = op.deref(ctx).get_result(0);
    let goto_prev = value_map
        .store_local(ctx, destination.local, res, block_ptr, Some(op))
        .unwrap_or(op);
    let Some(target_idx) = target else {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "cvt_e4m3_to_f32 call without target not supported".to_string()
            )
        );
    };
    Ok(emit_goto(ctx, *target_idx, goto_prev, block_map, loc))
}

/// `cute_rs::numeric::__compiler::cvt_f32x8_to_e2m1x8(v0..v7: f32) -> u32`
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_cvt_f32x8_to_e2m1x8(
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
    if args.len() != 8 || !destination.projection.is_empty() {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "cvt_f32x8_to_e2m1x8 expects 8 arguments and direct destination".to_string()
            )
        );
    }
    let mut vals = Vec::with_capacity(8);
    let mut cur_prev = prev_op;
    for arg in args {
        let (val, prev) =
            translate_operand(ctx, body, arg, value_map, block_ptr, cur_prev, loc.clone())?;
        cur_prev = prev.or(cur_prev);
        vals.push(val);
    }
    let op = dialect_cute::tensor_ops::CuteConvertF32x8ToE2m1x8Op::new(ctx, vals).get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block_ptr, cur_prev);
    let res = op.deref(ctx).get_result(0);
    let goto_prev = value_map
        .store_local(ctx, destination.local, res, block_ptr, Some(op))
        .unwrap_or(op);
    let Some(target_idx) = target else {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "cvt_f32x8_to_e2m1x8 call without target not supported".to_string()
            )
        );
    };
    Ok(emit_goto(ctx, *target_idx, goto_prev, block_map, loc))
}

/// Extract the scalar leaf type of `elem_ty` (unwrapping 1-field `#[repr(transparent)]` wrapper structs
/// such as `BFloat16(u16)`, `Float8E4M3FN(u8)`, and `Float4E2M1FNx8(u32)`).
fn scalar_leaf_type(ctx: &Context, elem_ty: TypeHandle) -> (TypeHandle, bool) {
    let ty_ref = elem_ty.deref(ctx);
    if let Some(strct) = ty_ref.downcast_ref::<MirStructType>()
        && strct.field_types.len() == 1
    {
        return (strct.field_types[0], true);
    }
    (elem_ty, false)
}

/// Unwrap a scalar `Value` if its type is a 1-field wrapper struct (`BFloat16`, `Float8E4M3FN`, `Float4E2M1FNx8`).
fn unwrap_scalar_value(
    ctx: &mut Context,
    val: Value,
    block_ptr: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    loc: &Location,
) -> (Value, TypeHandle, Option<Ptr<Operation>>) {
    let ty = val.get_type(ctx);
    let (leaf_ty, is_wrapper) = scalar_leaf_type(ctx, ty);
    if !is_wrapper {
        return (val, ty, prev);
    }
    let op = Operation::new(
        ctx,
        MirExtractFieldOp::get_concrete_op_info(),
        vec![leaf_ty],
        vec![val],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());
    MirExtractFieldOp::new(op).set_attr_index(ctx, FieldIndexAttr(0));
    insert(ctx, op, block_ptr, prev);
    (op.deref(ctx).get_result(0), leaf_ty, Some(op))
}

/// Read all `N` scalar elements from a `&RegisterTile<T, N>` reference operand using constant-index
/// `MirFieldAddrOp` + `MirArrayElementAddrOp` + `MirLoadOp` chains so `forward_constant_projected_aggregates`
/// eliminates intermediate tile slots in MIR.
fn load_register_tile_scalars(
    ctx: &mut Context,
    tile_ptr: Value,
    block_ptr: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    loc: &Location,
) -> TranslationResult<(Vec<Value>, TypeHandle, Option<Ptr<Operation>>)> {
    use core::num::NonZeroUsize;
    use dialect_mir::ops::{MirArrayElementAddrOp, MirConstantOp};
    use dialect_mir::types::MirArrayType;
    use pliron::builtin::attributes::IntegerAttr;
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::utils::apint::APInt;

    let (arr_ptr, arr_ty, arr_op) =
        struct_field_addr(ctx, tile_ptr, 0, "RegisterTile", block_ptr, prev, loc)?;
    let (elem_ty, n, is_mutable, address_space) = {
        let ptr_ty = arr_ptr.get_type(ctx);
        let ptr_ref = ptr_ty.deref(ctx);
        let Some(ptr) = ptr_ref.downcast_ref::<MirPtrType>() else {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported("RegisterTile field 0 pointer invalid".to_string())
            );
        };
        let arr_ref = arr_ty.deref(ctx);
        let Some(arr) = arr_ref.downcast_ref::<MirArrayType>() else {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported("RegisterTile field 0 is not MirArrayType".to_string())
            );
        };
        (
            arr.element_type(),
            arr.size as usize,
            ptr.is_mutable,
            ptr.address_space,
        )
    };
    let (leaf_ty, is_wrapper) = scalar_leaf_type(ctx, elem_ty);
    let elem_ptr_ty: TypeHandle = MirPtrType::get(ctx, elem_ty, is_mutable, address_space).into();
    let u64_ty = IntegerType::get(ctx, 64, Signedness::Unsigned);

    let mut cur_prev = Some(arr_op);
    let mut vals = Vec::with_capacity(n);
    for i in 0..n {
        let idx_op = Operation::new(
            ctx,
            MirConstantOp::get_concrete_op_info(),
            vec![u64_ty.into()],
            vec![],
            vec![],
            0,
        );
        idx_op.deref_mut(ctx).set_loc(loc.clone());
        let apint = APInt::from_u64(i as u64, NonZeroUsize::new(64).unwrap());
        MirConstantOp::new(idx_op).set_attr_value(ctx, IntegerAttr::new(u64_ty, apint));
        insert(ctx, idx_op, block_ptr, cur_prev);
        let idx_val = idx_op.deref(ctx).get_result(0);

        let elem_addr_op = Operation::new(
            ctx,
            MirArrayElementAddrOp::get_concrete_op_info(),
            vec![elem_ptr_ty],
            vec![arr_ptr, idx_val],
            vec![],
            0,
        );
        elem_addr_op.deref_mut(ctx).set_loc(loc.clone());
        insert(ctx, elem_addr_op, block_ptr, Some(idx_op));
        let elem_ptr = elem_addr_op.deref(ctx).get_result(0);

        let (leaf_ptr, after_addr) = if is_wrapper {
            let (lp, _, lp_op) = struct_field_addr(
                ctx,
                elem_ptr,
                0,
                "RegisterTile element wrapper",
                block_ptr,
                Some(elem_addr_op),
                loc,
            )?;
            (lp, lp_op)
        } else {
            (elem_ptr, elem_addr_op)
        };
        let (val_i, load_op) =
            load_through(ctx, leaf_ptr, leaf_ty, block_ptr, Some(after_addr), loc);
        vals.push(val_i);
        cur_prev = Some(load_op);
    }
    Ok((vals, leaf_ty, cur_prev))
}

/// Package `M` scalar result values into `MirConstructStructOp(RegisterTile, [MirConstructArrayOp(...)])`
/// and store into `destination.local`.
#[allow(clippy::too_many_arguments)]
fn store_register_tile_scalars(
    ctx: &mut Context,
    body: &mir::Body,
    destination: &mir::Place,
    scalars: Vec<Value>,
    block_ptr: Ptr<BasicBlock>,
    prev: Ptr<Operation>,
    value_map: &mut ValueMap,
    loc: &Location,
) -> TranslationResult<Ptr<Operation>> {
    use dialect_mir::ops::{MirConstructArrayOp, MirConstructStructOp};
    use dialect_mir::types::MirArrayType;

    let dest_rust_ty = destination.ty(body.locals()).map_err(|e| {
        pliron::input_error_noloc!(TranslationErr::unsupported(format!(
            "cannot query RegisterTile destination type: {e:?}"
        )))
    })?;
    let dest_ty = translate_type(ctx, &dest_rust_ty)?;
    let arr_ty = {
        let ty_ref = dest_ty.deref(ctx);
        let Some(strct) = ty_ref.downcast_ref::<MirStructType>() else {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(
                    "RegisterTile destination is not MirStructType".to_string()
                )
            );
        };
        strct.field_types[0]
    };
    let elem_ty = {
        let arr_ref = arr_ty.deref(ctx);
        let Some(arr) = arr_ref.downcast_ref::<MirArrayType>() else {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported("RegisterTile field 0 is not MirArrayType".to_string())
            );
        };
        arr.element_type()
    };
    let (_, is_wrapper) = scalar_leaf_type(ctx, elem_ty);
    let mut cur_prev = prev;
    let mut elem_vals = Vec::with_capacity(scalars.len());
    if is_wrapper {
        for s in scalars {
            let wrap_op = Operation::new(
                ctx,
                MirConstructStructOp::get_concrete_op_info(),
                vec![elem_ty],
                vec![s],
                vec![],
                0,
            );
            wrap_op.deref_mut(ctx).set_loc(loc.clone());
            insert(ctx, wrap_op, block_ptr, Some(cur_prev));
            elem_vals.push(wrap_op.deref(ctx).get_result(0));
            cur_prev = wrap_op;
        }
    } else {
        elem_vals = scalars;
    }
    let arr_op = Operation::new(
        ctx,
        MirConstructArrayOp::get_concrete_op_info(),
        vec![arr_ty],
        elem_vals,
        vec![],
        0,
    );
    arr_op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, arr_op, block_ptr, Some(cur_prev));
    let arr_val = arr_op.deref(ctx).get_result(0);

    let tile_op = Operation::new(
        ctx,
        MirConstructStructOp::get_concrete_op_info(),
        vec![dest_ty],
        vec![arr_val],
        vec![],
        0,
    );
    tile_op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, tile_op, block_ptr, Some(arr_op));
    let tile_val = tile_op.deref(ctx).get_result(0);
    Ok(value_map
        .store_local(ctx, destination.local, tile_val, block_ptr, Some(tile_op))
        .unwrap_or(tile_op))
}

/// Extract all `Const` generic arguments of `func` as `u64` values in declaration order.
fn callee_const_u64s(func: &mir::Operand, loc: &Location) -> TranslationResult<Vec<u64>> {
    let mir::Operand::Constant(const_op) = func else {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported("RegisterTile hook called indirectly".to_string())
        );
    };
    let rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::FnDef(_, substs)) =
        const_op.const_.ty().kind()
    else {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported("RegisterTile hook is not a FnDef".to_string())
        );
    };
    let mut out = Vec::new();
    for arg in &substs.0 {
        if let rustc_public::ty::GenericArgKind::Const(c) = arg {
            out.push(super::layout::const_u64(c, "RegisterTile", loc)?);
        }
    }
    Ok(out)
}

/// Emit `dialect-cute` vector ops (`CuteVector*Op`) for `cute_rs::tensor::__compiler::register_tile_*`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_register_tile_vector_op(
    ctx: &mut Context,
    body: &mir::Body,
    func: &mir::Operand,
    cute_fn: super::registry::CuteFn,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    use super::registry::CuteFn;
    use dialect_cute::tensor_ops::{
        CuteVectorBinaryOp, CuteVectorBitcastOp, CuteVectorBroadcastOp, CuteVectorCastOp,
        CuteVectorConcatOp, CuteVectorReduceOp, CuteVectorScalarOp, CuteVectorSliceOp,
        CuteVectorUnaryOp,
    };

    if !destination.projection.is_empty() {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "register_tile vector op destination with projections not supported".to_string()
            )
        );
    }
    let consts = callee_const_u64s(func, &loc)?;
    let mut cur_prev = prev_op;

    let goto_prev = match cute_fn {
        CuteFn::RegisterTileBroadcast => {
            let n = consts[0] as usize;
            let (raw_val, prev) = translate_operand(
                ctx,
                body,
                &args[0],
                value_map,
                block_ptr,
                cur_prev,
                loc.clone(),
            )?;
            let (val, lane_ty, prev) =
                unwrap_scalar_value(ctx, raw_val, block_ptr, prev.or(cur_prev), &loc);
            let op = CuteVectorBroadcastOp::new(ctx, val, lane_ty, n).get_operation();
            op.deref_mut(ctx).set_loc(loc.clone());
            insert(ctx, op, block_ptr, prev);
            let results: Vec<Value> = (0..n).map(|i| op.deref(ctx).get_result(i)).collect();
            store_register_tile_scalars(
                ctx,
                body,
                destination,
                results,
                block_ptr,
                op,
                value_map,
                &loc,
            )?
        }
        CuteFn::RegisterTileBinary => {
            let n = consts[0] as usize;
            let kind = consts[1];
            let (lhs_ptr, prev) = translate_operand(
                ctx,
                body,
                &args[0],
                value_map,
                block_ptr,
                cur_prev,
                loc.clone(),
            )?;
            cur_prev = prev.or(cur_prev);
            let (rhs_ptr, prev) = translate_operand(
                ctx,
                body,
                &args[1],
                value_map,
                block_ptr,
                cur_prev,
                loc.clone(),
            )?;
            cur_prev = prev.or(cur_prev);
            let (mut operands, lane_ty, prev) =
                load_register_tile_scalars(ctx, lhs_ptr, block_ptr, cur_prev, &loc)?;
            let (rhs_scalars, _, prev) =
                load_register_tile_scalars(ctx, rhs_ptr, block_ptr, prev, &loc)?;
            operands.extend(rhs_scalars);
            let op = CuteVectorBinaryOp::new(ctx, operands, lane_ty, n, kind).get_operation();
            op.deref_mut(ctx).set_loc(loc.clone());
            insert(ctx, op, block_ptr, prev);
            let results: Vec<Value> = (0..n).map(|i| op.deref(ctx).get_result(i)).collect();
            store_register_tile_scalars(
                ctx,
                body,
                destination,
                results,
                block_ptr,
                op,
                value_map,
                &loc,
            )?
        }
        CuteFn::RegisterTileScalar => {
            let n = consts[0] as usize;
            let kind = consts[1];
            let (lhs_ptr, prev) = translate_operand(
                ctx,
                body,
                &args[0],
                value_map,
                block_ptr,
                cur_prev,
                loc.clone(),
            )?;
            cur_prev = prev.or(cur_prev);
            let (raw_rhs, prev) = translate_operand(
                ctx,
                body,
                &args[1],
                value_map,
                block_ptr,
                cur_prev,
                loc.clone(),
            )?;
            cur_prev = prev.or(cur_prev);
            let (rhs, _, prev) = unwrap_scalar_value(ctx, raw_rhs, block_ptr, cur_prev, &loc);
            let (mut operands, lane_ty, prev) =
                load_register_tile_scalars(ctx, lhs_ptr, block_ptr, prev, &loc)?;
            operands.push(rhs);
            let op = CuteVectorScalarOp::new(ctx, operands, lane_ty, n, kind).get_operation();
            op.deref_mut(ctx).set_loc(loc.clone());
            insert(ctx, op, block_ptr, prev);
            let results: Vec<Value> = (0..n).map(|i| op.deref(ctx).get_result(i)).collect();
            store_register_tile_scalars(
                ctx,
                body,
                destination,
                results,
                block_ptr,
                op,
                value_map,
                &loc,
            )?
        }
        CuteFn::RegisterTileUnary => {
            let n = consts[0] as usize;
            let kind = consts[1];
            let (src_ptr, prev) = translate_operand(
                ctx,
                body,
                &args[0],
                value_map,
                block_ptr,
                cur_prev,
                loc.clone(),
            )?;
            cur_prev = prev.or(cur_prev);
            let (operands, lane_ty, prev) =
                load_register_tile_scalars(ctx, src_ptr, block_ptr, cur_prev, &loc)?;
            let op = CuteVectorUnaryOp::new(ctx, operands, lane_ty, n, kind).get_operation();
            op.deref_mut(ctx).set_loc(loc.clone());
            insert(ctx, op, block_ptr, prev);
            let results: Vec<Value> = (0..n).map(|i| op.deref(ctx).get_result(i)).collect();
            store_register_tile_scalars(
                ctx,
                body,
                destination,
                results,
                block_ptr,
                op,
                value_map,
                &loc,
            )?
        }
        CuteFn::RegisterTileReduce => {
            let kind = consts[1];
            let (src_ptr, prev) = translate_operand(
                ctx,
                body,
                &args[0],
                value_map,
                block_ptr,
                cur_prev,
                loc.clone(),
            )?;
            cur_prev = prev.or(cur_prev);
            let (init_val, prev) = translate_operand(
                ctx,
                body,
                &args[1],
                value_map,
                block_ptr,
                cur_prev,
                loc.clone(),
            )?;
            cur_prev = prev.or(cur_prev);
            let (mut operands, _, prev) =
                load_register_tile_scalars(ctx, src_ptr, block_ptr, cur_prev, &loc)?;
            operands.push(init_val);
            let op = CuteVectorReduceOp::new(ctx, operands, kind).get_operation();
            op.deref_mut(ctx).set_loc(loc.clone());
            insert(ctx, op, block_ptr, prev);
            let res = op.deref(ctx).get_result(0);
            value_map
                .store_local(ctx, destination.local, res, block_ptr, Some(op))
                .unwrap_or(op)
        }
        CuteFn::RegisterTileCast => {
            let n = consts[0] as usize;
            let kind = consts[1];
            let (src_ptr, prev) = translate_operand(
                ctx,
                body,
                &args[0],
                value_map,
                block_ptr,
                cur_prev,
                loc.clone(),
            )?;
            cur_prev = prev.or(cur_prev);
            let (operands, _, prev) =
                load_register_tile_scalars(ctx, src_ptr, block_ptr, cur_prev, &loc)?;
            let dest_rust_ty = destination.ty(body.locals()).unwrap();
            let dest_ty = translate_type(ctx, &dest_rust_ty)?;
            let arr_ty = dest_ty
                .deref(ctx)
                .downcast_ref::<MirStructType>()
                .unwrap()
                .field_types[0];
            let elem_ty = arr_ty
                .deref(ctx)
                .downcast_ref::<dialect_mir::types::MirArrayType>()
                .unwrap()
                .element_type();
            let (dst_lane_ty, _) = scalar_leaf_type(ctx, elem_ty);
            let op = CuteVectorCastOp::new(ctx, operands, dst_lane_ty, n, kind).get_operation();
            op.deref_mut(ctx).set_loc(loc.clone());
            insert(ctx, op, block_ptr, prev);
            let results: Vec<Value> = (0..n).map(|i| op.deref(ctx).get_result(i)).collect();
            store_register_tile_scalars(
                ctx,
                body,
                destination,
                results,
                block_ptr,
                op,
                value_map,
                &loc,
            )?
        }
        CuteFn::RegisterTileBitcast => {
            let m = consts[1] as usize;
            let (src_ptr, prev) = translate_operand(
                ctx,
                body,
                &args[0],
                value_map,
                block_ptr,
                cur_prev,
                loc.clone(),
            )?;
            cur_prev = prev.or(cur_prev);
            let (operands, _, prev) =
                load_register_tile_scalars(ctx, src_ptr, block_ptr, cur_prev, &loc)?;
            let dest_rust_ty = destination.ty(body.locals()).unwrap();
            let dest_ty = translate_type(ctx, &dest_rust_ty)?;
            let arr_ty = dest_ty
                .deref(ctx)
                .downcast_ref::<MirStructType>()
                .unwrap()
                .field_types[0];
            let elem_ty = arr_ty
                .deref(ctx)
                .downcast_ref::<dialect_mir::types::MirArrayType>()
                .unwrap()
                .element_type();
            let (dst_lane_ty, _) = scalar_leaf_type(ctx, elem_ty);
            let op = CuteVectorBitcastOp::new(ctx, operands, dst_lane_ty, m).get_operation();
            op.deref_mut(ctx).set_loc(loc.clone());
            insert(ctx, op, block_ptr, prev);
            let results: Vec<Value> = (0..m).map(|i| op.deref(ctx).get_result(i)).collect();
            store_register_tile_scalars(
                ctx,
                body,
                destination,
                results,
                block_ptr,
                op,
                value_map,
                &loc,
            )?
        }
        CuteFn::RegisterTileSlice => {
            let offset = consts[1];
            let m = consts[2] as usize;
            let (src_ptr, prev) = translate_operand(
                ctx,
                body,
                &args[0],
                value_map,
                block_ptr,
                cur_prev,
                loc.clone(),
            )?;
            cur_prev = prev.or(cur_prev);
            let (operands, lane_ty, prev) =
                load_register_tile_scalars(ctx, src_ptr, block_ptr, cur_prev, &loc)?;
            let op = CuteVectorSliceOp::new(ctx, operands, lane_ty, m, offset).get_operation();
            op.deref_mut(ctx).set_loc(loc.clone());
            insert(ctx, op, block_ptr, prev);
            let results: Vec<Value> = (0..m).map(|i| op.deref(ctx).get_result(i)).collect();
            store_register_tile_scalars(
                ctx,
                body,
                destination,
                results,
                block_ptr,
                op,
                value_map,
                &loc,
            )?
        }
        CuteFn::RegisterTileConcat => {
            let m = consts[1] as usize;
            let (lo_ptr, prev) = translate_operand(
                ctx,
                body,
                &args[0],
                value_map,
                block_ptr,
                cur_prev,
                loc.clone(),
            )?;
            cur_prev = prev.or(cur_prev);
            let (hi_ptr, prev) = translate_operand(
                ctx,
                body,
                &args[1],
                value_map,
                block_ptr,
                cur_prev,
                loc.clone(),
            )?;
            cur_prev = prev.or(cur_prev);
            let (mut operands, lane_ty, prev) =
                load_register_tile_scalars(ctx, lo_ptr, block_ptr, cur_prev, &loc)?;
            let (hi_scalars, _, prev) =
                load_register_tile_scalars(ctx, hi_ptr, block_ptr, prev, &loc)?;
            operands.extend(hi_scalars);
            let op = CuteVectorConcatOp::new(ctx, operands, lane_ty, m).get_operation();
            op.deref_mut(ctx).set_loc(loc.clone());
            insert(ctx, op, block_ptr, prev);
            let results: Vec<Value> = (0..m).map(|i| op.deref(ctx).get_result(i)).collect();
            store_register_tile_scalars(
                ctx,
                body,
                destination,
                results,
                block_ptr,
                op,
                value_map,
                &loc,
            )?
        }
        _ => unreachable!(),
    };

    let Some(target_idx) = target else {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "register_tile vector op call without target not supported".to_string()
            )
        );
    };
    Ok(emit_goto(ctx, *target_idx, goto_prev, block_map, loc))
}

/// `cute_rs::atom::vector::store_tile::<T, N>(dst: *mut T, idx: usize, vals: &[T; N])`
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_store_tile(
    ctx: &mut Context,
    body: &mir::Body,
    func: &mir::Operand,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    // The unit return value is never materialized; the copy is the effect.
    let _ = destination;
    if args.len() != 3 {
        return input_err!(
            loc,
            TranslationErr::unsupported("store_tile expects exactly 3 arguments".to_string())
        );
    }
    let (elem, n) = tile_elem_and_count(ctx, func, &loc)?;

    let (dst_base, prev) = translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    let (idx_val, prev) =
        translate_operand(ctx, body, &args[1], value_map, block_ptr, prev, loc.clone())?;
    let (vals_ref, prev) =
        translate_operand(ctx, body, &args[2], value_map, block_ptr, prev, loc.clone())?;

    let (dst, prev) = ptr_offset(ctx, dst_base, idx_val, block_ptr, prev, &loc);
    let (src, prev) = cast_pointee(ctx, vals_ref, elem, block_ptr, Some(prev), &loc)?;

    finish_with_copy(
        ctx,
        src,
        dst,
        n,
        elem,
        target,
        block_ptr,
        Some(prev),
        block_map,
        loc,
        "store_tile",
    )
}

/// `cute_rs::nvgpu::cpasync::copy_tma_2d::<T, SmemL>(desc, (r, c), &mut SmemTile, bar)`
///
/// The descriptor becomes a global transport view; the selected shared base
/// and exact remaining capacity become a shared transport view. One
/// zero-result semantic copy connects those views to the coordinate and
/// completion barrier. The temporary view values stay in direct SSA and do
/// not change the Rust ABI.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_copy_tma(
    ctx: &mut Context,
    body: &mir::Body,
    func: &mir::Operand,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 4 {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "copy_tma_2d expects 4 runtime arguments, got {}",
                args.len()
            ))
        );
    }
    let config = match decode_copy_tma_callee(func) {
        Ok(config) => config,
        Err(error) => {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "copy_tma_2d static configuration is invalid: {error}"
                ))
            );
        }
    };
    let elem = translate_type(ctx, &config.element_type)?;

    let (tensor_map, prev) = translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    let prev = prev.or(prev_op);
    let (coord_val, prev) =
        translate_operand(ctx, body, &args[1], value_map, block_ptr, prev, loc.clone())?;
    let (smem_ref, prev) =
        translate_operand(ctx, body, &args[2], value_map, block_ptr, prev, loc.clone())?;
    let (barrier, prev) =
        translate_operand(ctx, body, &args[3], value_map, block_ptr, prev, loc.clone())?;

    let (smem_base, prev) = struct_field_load(
        ctx,
        smem_ref,
        0,
        "copy_tma_2d destination",
        block_ptr,
        prev,
        &loc,
    )?;
    let (smem_capacity, prev) = struct_field_load(
        ctx,
        smem_ref,
        1,
        "copy_tma_2d destination capacity",
        block_ptr,
        Some(prev),
        &loc,
    )?;
    let (smem_base, prev) = recover_cta_shared_pointer(
        ctx,
        smem_base,
        elem,
        "copy_tma_2d destination",
        block_ptr,
        prev,
        &loc,
    )?;
    let prev = Some(prev);

    let coordinate_ty: TypeHandle = pliron::builtin::types::IntegerType::get(
        ctx,
        64,
        pliron::builtin::types::Signedness::Unsigned,
    )
    .into();
    let (tile_r, prev) = tuple_field(ctx, coord_val, 0, coordinate_ty, block_ptr, prev, &loc);
    let (tile_c, prev) = tuple_field(
        ctx,
        coord_val,
        1,
        coordinate_ty,
        block_ptr,
        Some(prev),
        &loc,
    );

    let copy_op = emit_tma_copy_2d_views(
        ctx,
        tensor_map,
        smem_base,
        smem_capacity,
        tile_r,
        tile_c,
        barrier,
        elem,
        &config.smem_layout,
        config.smem_alignment_bytes,
        block_ptr,
        Some(prev),
        &loc,
    );

    let Some(target_idx) = target else {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "copy_tma_2d call without target not supported".to_string()
            )
        );
    };
    Ok(emit_goto(ctx, *target_idx, copy_op, block_map, loc))
}

/// `cute_rs::nvgpu::cpasync::copy_tma_s2g_2d::<T, SmemL>(desc, (r, c), SmemTile)`
///
/// The recognized boundary recovers the CTA-shared source proof from the
/// carrier, splits the tile coordinate, and creates two short-lived TMA views
/// feeding one semantic store. The store-pipeline acquire/commit/tail calls
/// are recognized separately, so the complete source protocol stays visible.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_copy_tma_s2g(
    ctx: &mut Context,
    body: &mir::Body,
    func: &mir::Operand,
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
            loc,
            TranslationErr::unsupported(format!(
                "copy_tma_s2g_2d expects 3 runtime arguments, got {}",
                args.len()
            ))
        );
    }
    let config = match decode_copy_tma_s2g_callee(func) {
        Ok(config) => config,
        Err(error) => {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "copy_tma_s2g_2d static configuration is invalid: {error}"
                ))
            );
        }
    };
    let elem = translate_type(ctx, &config.element_type)?;

    let (tensor_map, prev) = translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    let prev = prev.or(prev_op);
    let (coord_val, prev) =
        translate_operand(ctx, body, &args[1], value_map, block_ptr, prev, loc.clone())?;
    let (smem_tile, prev) =
        translate_operand(ctx, body, &args[2], value_map, block_ptr, prev, loc.clone())?;

    let (smem_base, prev) = aggregate_field(
        ctx,
        smem_tile,
        0,
        "copy_tma_s2g_2d source",
        block_ptr,
        prev,
        &loc,
    )?;
    let (smem_capacity, prev) = aggregate_field(
        ctx,
        smem_tile,
        1,
        "copy_tma_s2g_2d source",
        block_ptr,
        Some(prev),
        &loc,
    )?;
    let (smem_base, prev) = recover_cta_shared_pointer(
        ctx,
        smem_base,
        elem,
        "copy_tma_s2g_2d source",
        block_ptr,
        prev,
        &loc,
    )?;
    let prev = Some(prev);

    let coordinate_ty: TypeHandle = pliron::builtin::types::IntegerType::get(
        ctx,
        64,
        pliron::builtin::types::Signedness::Unsigned,
    )
    .into();
    let (tile_r, prev) = tuple_field(ctx, coord_val, 0, coordinate_ty, block_ptr, prev, &loc);
    let (tile_c, prev) = tuple_field(
        ctx,
        coord_val,
        1,
        coordinate_ty,
        block_ptr,
        Some(prev),
        &loc,
    );

    let destination =
        CuteTmaGmemViewOp::new_destination(ctx, tensor_map, elem, config.smem_layout.clone());
    let destination_op = destination.get_operation();
    destination_op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, destination_op, block_ptr, Some(prev));
    let destination = destination_op.deref(ctx).get_result(0);
    let source = CuteTmaSmemViewOp::new(
        ctx,
        smem_base,
        smem_capacity,
        elem,
        config.smem_layout.clone(),
        config.smem_alignment_bytes,
    );
    let source_op = source.get_operation();
    source_op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, source_op, block_ptr, Some(destination_op));
    let source = source_op.deref(ctx).get_result(0);
    let store = dialect_cute::epilogue_ops::CuteTmaStore2dSemanticOp::new(
        ctx,
        source,
        destination,
        tile_r,
        tile_c,
    );
    let store_op = store.get_operation();
    store_op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, store_op, block_ptr, Some(source_op));

    let Some(target_idx) = target else {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "copy_tma_s2g_2d call without target not supported".to_string()
            )
        );
    };
    Ok(emit_goto(ctx, *target_idx, store_op, block_map, loc))
}

/// Preserve a typed cache-aware register-to-global copy until backend selection.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_store_global_no_allocate(
    ctx: &mut Context,
    body: &mir::Body,
    func: &mir::Operand,
    args: &[mir::Operand],
    target: &Option<usize>,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    use dialect_cute::copy_ops::CuteStoreGlobalNoAllocateOp;
    use dialect_mir::attributes::MirPointerKindAuthorityAttr;
    use pliron::common_traits::Verify;
    if args.len() != 2 {
        return input_err!(
            loc,
            TranslationErr::unsupported("global store expects a destination and register array")
        );
    }
    let (element, n) = tile_elem_and_count(ctx, func, &loc)?;
    let (dst, prev) = translate_operand(ctx, body, &args[0], value_map, block, prev, loc.clone())?;
    let (src, prev) = translate_operand(ctx, body, &args[1], value_map, block, prev, loc.clone())?;
    let (src, source_cast) = cast_pointee(ctx, src, element, block, prev, &loc)?;
    let pointer = dst
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<MirPtrType>()
        .cloned();
    let Some(pointer) = pointer else {
        return input_err!(
            loc,
            TranslationErr::unsupported("global store destination must be a pointer")
        );
    };
    if !pointer.is_mutable
        || pointer.pointee != element
        || !matches!(
            pointer.address_space,
            address_space::GLOBAL | address_space::GENERIC
        )
    {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "global store destination must be a writable global pointer of the element type"
            )
        );
    }
    let global = global_store_pointer_type(ctx, &pointer);
    let (dst, after) = if dst.get_type(ctx) == global {
        (dst, source_cast)
    } else {
        let op = Operation::new(
            ctx,
            MirCastOp::get_concrete_op_info(),
            vec![global],
            vec![dst],
            vec![],
            0,
        );
        op.deref_mut(ctx).set_loc(loc.clone());
        let cast = MirCastOp::new(op);
        cast.set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
        if dialect_mir::types::type_contains_concrete_pointer_kind(ctx, global) {
            cast.set_pointer_kind_authority(ctx, MirPointerKindAuthorityAttr::RawAddress);
        }
        insert(ctx, op, block, Some(source_cast));
        (op.deref(ctx).get_result(0), op)
    };
    let store = CuteStoreGlobalNoAllocateOp::new(ctx, dst, src, n);
    store
        .verify(ctx)
        .map_err(|e| pliron::input_error_noloc!(TranslationErr::unsupported(e.to_string())))?;
    let op = store.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, Some(after));
    let Some(target) = target else {
        return input_err!(
            loc,
            TranslationErr::unsupported("global store requires a continuation")
        );
    };
    Ok(emit_goto(ctx, *target, op, block_map, loc))
}

/// Translate a global-store destination while preserving raw mutable-pointer authority.
fn global_store_pointer_type(ctx: &mut Context, pointer: &MirPtrType) -> TypeHandle {
    crate::translator::facts::mint_ptr_type(
        ctx,
        pointer.pointee,
        address_space::GLOBAL,
        crate::translator::facts::pointer_origin_of_ptr_carrier(pointer),
    )
    .into()
}
