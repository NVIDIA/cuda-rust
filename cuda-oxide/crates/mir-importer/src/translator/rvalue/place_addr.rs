/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! [`translate_place_address`] and the stateful slot projection loop.

use super::place_read::translate_place;
use super::pointee::{
    PointeeKind, emit_indexed_element_addr, erased_slice_data_pointer_type, is_empty_tuple_type,
    mir_ptr_pointee, pointer_pointee_kind, projected_pointer_type, slice_like_element_type,
};
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::facts;
use crate::translator::types;
use crate::translator::values::ValueMap;
use dialect_mir::attributes::MirCastKindAttr;
use dialect_mir::ops::{
    MirCastOp, MirConstantOp, MirConstructArrayOp, MirExtractFieldOp, MirLoadOp, MirPtrOffsetOp,
    MirSubOp,
};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::printable::Printable;
use pliron::r#type::{TypeHandle, Typed};
use pliron::utils::apint::APInt;
use pliron::value::Value;
use pliron::{input_err, input_error_noloc};
use rustc_public::mir;
use rustc_public_bridge::IndexedVal;
use std::num::NonZeroUsize;

/// Compute the in-memory address of `place` by walking its FULL projection
/// list starting from `place.local`'s alloca slot.
///
/// Single entry point for `Rvalue::Ref` / `Rvalue::AddressOf` address
/// materialisation: `&(*ptr)` loads the pointer, `&(*ptr).field` adds a
/// field address, `&x.arr[i]` adds an element address, and arbitrary
/// combinations compose.
///
/// Returns `Ok(None)` when the local has no slot (ZST / ghost locals) or
/// when the projection chain contains an element
/// [`translate_place_addr_from_slot`] cannot lower. The caller decides
/// whether a value-copy fallback is sound (shared borrows: reads through a
/// copy are fine) or the construct must be rejected (mutable borrows / raw
/// mut pointers: writes through a copy are silently lost).
///
/// Also used by statement translation to compute the destination address
/// of projected assignments (indexed `(*ptr)[i]` writes and 3+ element
/// projection chains), where the same "walk the chain, then act through
/// the address" logic applies with a store instead of a borrow.
#[allow(clippy::too_many_arguments)]
pub(crate) fn translate_place_address(
    ctx: &mut Context,
    body: &mir::Body,
    value_map: &ValueMap,
    place: &mir::Place,
    is_mutable: bool,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<Option<(Value, Option<Ptr<Operation>>)>> {
    let Some(slot) = value_map.get_slot(place.local) else {
        return Ok(None);
    };
    translate_place_addr_from_slot(
        ctx,
        body,
        value_map,
        slot,
        &place.projection,
        is_mutable,
        block_ptr,
        prev_op,
        loc,
    )
}

/// Whether an enum payload's SEMANTIC type needs canonical-storage coercion
/// when its bytes live inside enum storage.
///
/// Mirrors what mir-lower's `enum_payload_storage_type` rewrites: `bool`
/// leaves are stored as canonical `i8` bytes and shared-memory pointer
/// leaves are stored as CUDA generic pointers, recursively through
/// struct/tuple/array nesting. An address of such a payload cannot carry
/// that coercion (the address escapes, and loads and stores through it are
/// typed with the SEMANTIC type), so the address walker uses this predicate
/// to punt SHARED borrows back to the sound value-copy fallback.
///
/// Layering: this predicate is allowed to be conservative. A nested enum
/// payload, for example, is treated as needing coercion whenever any of its
/// own payload fields does, without proving the leaf survives into the
/// nested enum's converted storage. Over-punting only costs a copy for a
/// shared borrow and stays sound. mir-lower's canonical-storage gate on
/// `mir.field_addr` remains the fail-closed authority, so a miss here still
/// errors loudly instead of miscompiling.
pub(crate) fn enum_payload_needs_storage_coercion_pub(ctx: &Context, ty: TypeHandle) -> bool {
    enum_payload_needs_storage_coercion(ctx, ty)
}

fn enum_payload_needs_storage_coercion(ctx: &Context, ty: TypeHandle) -> bool {
    // Bool leaf: semantic i1, canonical i8 byte in enum storage.
    if let Some(integer) = ty.deref(ctx).downcast_ref::<IntegerType>() {
        return integer.width() == 1;
    }
    // Pointer leaf: shared-memory pointers are stored as generic pointers
    // because their physical width is target-mode dependent.
    if let Some(pointer) = ty
        .deref(ctx)
        .downcast_ref::<dialect_mir::types::MirPtrType>()
    {
        return pointer.address_space() == dialect_mir::types::address_space::SHARED;
    }
    // Aggregates: recurse through every leaf position the storage rewrite
    // visits. Collect the children first so the type `Ref` is dropped
    // before recursing.
    let children: Vec<TypeHandle> = {
        let ty_ref = ty.deref(ctx);
        if let Some(tuple) = ty_ref.downcast_ref::<dialect_mir::types::MirTupleType>() {
            tuple.types.clone()
        } else if let Some(struct_ty) = ty_ref.downcast_ref::<dialect_mir::types::MirStructType>() {
            struct_ty.field_types.clone()
        } else if let Some(array) = ty_ref.downcast_ref::<dialect_mir::types::MirArrayType>() {
            vec![array.element_ty]
        } else if let Some(enum_ty) = ty_ref.downcast_ref::<dialect_mir::types::MirEnumType>() {
            enum_ty.all_field_types.clone()
        } else {
            return false;
        }
    };
    children
        .into_iter()
        .any(|child| enum_payload_needs_storage_coercion(ctx, child))
}

/// Extract the runtime element count from a slice-shaped fat value.
///
/// CUDA Oxide models slices as `(data_ptr, len)`, with the length in field 1.
/// The extraction must use the fat value itself: inferring a length from the
/// data pointer's pointee type is wrong for slices whose elements are arrays.
pub(super) fn emit_slice_len_extract(
    ctx: &mut Context,
    slice_value: Value,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> (Value, Ptr<Operation>) {
    let usize_ty = types::get_usize_type(ctx);
    let op = Operation::new(
        ctx,
        MirExtractFieldOp::get_concrete_op_info(),
        vec![usize_ty.to_handle()],
        vec![slice_value],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc);
    MirExtractFieldOp::new(op).set_attr_index(ctx, dialect_mir::attributes::FieldIndexAttr(1));
    match prev_op {
        Some(prev) => op.insert_after(ctx, prev),
        None => op.insert_at_front(block_ptr, ctx),
    }
    (op.deref(ctx).get_result(0), op)
}

/// Materialize the zero-based index for
/// `ConstantIndex { offset, from_end: true }` on a runtime-length slice.
///
/// rustc defines the offset as 1-based from the end, so the index is
/// `slice_len - offset`. The MIR pattern-length test dominates this place,
/// therefore the subtraction cannot underflow on an executed path.
fn emit_from_end_slice_index(
    ctx: &mut Context,
    slice_len: Value,
    offset: u64,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Ptr<Operation>)> {
    let usize_ty = types::get_usize_type(ctx);
    let usize_handle = usize_ty.to_handle();
    if slice_len.get_type(ctx) != usize_handle {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "from-end ConstantIndex expected slice length of type {}, got {}",
                usize_handle.disp(ctx),
                slice_len.get_type(ctx).disp(ctx)
            ))
        );
    }

    let width = NonZeroUsize::new(
        usize::try_from(usize_ty.deref(ctx).width()).expect("usize width must fit usize"),
    )
    .expect("usize integer width must be non-zero");
    let offset_attr =
        pliron::builtin::attributes::IntegerAttr::new(usize_ty, APInt::from_u64(offset, width));
    let offset_op = Operation::new(
        ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![usize_handle],
        vec![],
        vec![],
        0,
    );
    offset_op.deref_mut(ctx).set_loc(loc.clone());
    MirConstantOp::new(offset_op).set_attr_value(ctx, offset_attr);
    match prev_op {
        Some(prev) => offset_op.insert_after(ctx, prev),
        None => offset_op.insert_at_front(block_ptr, ctx),
    }
    let offset_value = offset_op.deref(ctx).get_result(0);

    let sub_op = Operation::new(
        ctx,
        MirSubOp::get_concrete_op_info(),
        vec![usize_handle],
        vec![slice_len, offset_value],
        vec![],
        0,
    );
    sub_op.deref_mut(ctx).set_loc(loc);
    sub_op.insert_after(ctx, offset_op);

    Ok((sub_op.deref(ctx).get_result(0), sub_op))
}

/// Emit a `usize` constant and insert it after `prev_op`.
fn emit_usize_constant(
    ctx: &mut Context,
    value: u64,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> (Value, Ptr<Operation>) {
    use pliron::builtin::attributes::IntegerAttr;

    let usize_ty = types::get_usize_type(ctx);
    let attr = IntegerAttr::new(
        IntegerType::get(ctx, 64, Signedness::Unsigned),
        APInt::from_u64(value, NonZeroUsize::new(64).unwrap()),
    );
    let op = Operation::new(
        ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![usize_ty.to_handle()],
        vec![],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc);
    MirConstantOp::new(op).set_attr_value(ctx, attr);
    match prev_op {
        Some(prev) => op.insert_after(ctx, prev),
        None => op.insert_at_front(block_ptr, ctx),
    }
    (op.deref(ctx).get_result(0), op)
}

/// Compute the address of a sized array subslice.
///
/// For `Subslice { from, to, from_end: false }`, rustc codegen takes the
/// address of element `from` and re-types that address as `[T; to - from]`.
#[allow(clippy::too_many_arguments)]
fn emit_array_subslice_address(
    ctx: &mut Context,
    array_ptr: Value,
    from: u64,
    to: u64,
    is_mutable: bool,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Ptr<Operation>)> {
    use dialect_mir::ops::MirArrayElementAddrOp;

    let (element_ty, array_len) = {
        let ptr_ty = array_ptr.get_type(ctx);
        let ptr_ty = ptr_ty.deref(ctx);
        let Some(ptr_ty) = ptr_ty.downcast_ref::<dialect_mir::types::MirPtrType>() else {
            return input_err!(
                loc,
                TranslationErr::unsupported("Subslice base is not a pointer".to_string())
            );
        };
        let array_ty = ptr_ty.pointee.deref(ctx);
        let Some(array_ty) = array_ty.downcast_ref::<dialect_mir::types::MirArrayType>() else {
            return input_err!(
                loc,
                TranslationErr::unsupported(
                    "array Subslice base pointer does not point to MirArrayType".to_string()
                )
            );
        };
        (array_ty.element_type(), array_ty.size())
    };

    let Some(projected_len) = to.checked_sub(from) else {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "array Subslice has inverted bounds: from={from}, to={to}"
            ))
        );
    };
    if to > array_len {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "array Subslice end {to} exceeds array length {array_len}"
            ))
        );
    }

    let (from_value, from_op) = emit_usize_constant(ctx, from, block_ptr, prev_op, loc.clone());
    let elem_ptr_ty = projected_pointer_type(
        ctx,
        array_ptr.get_type(ctx),
        element_ty,
        /* legacy requested mutability, ignored */ is_mutable,
    )
    .expect("array subslice base must be a MirPtrType");
    let addr_op = Operation::new(
        ctx,
        MirArrayElementAddrOp::get_concrete_op_info(),
        vec![elem_ptr_ty],
        vec![array_ptr, from_value],
        vec![],
        0,
    );
    addr_op.deref_mut(ctx).set_loc(loc.clone());
    addr_op.insert_after(ctx, from_op);
    let elem_ptr = addr_op.deref(ctx).get_result(0);

    let projected_array_ty: TypeHandle =
        dialect_mir::types::MirArrayType::get(ctx, element_ty, projected_len).into();
    let projected_ptr_ty = projected_pointer_type(
        ctx,
        elem_ptr.get_type(ctx),
        projected_array_ty,
        /* legacy requested mutability, ignored */ is_mutable,
    )
    .expect("array subslice element address must be a MirPtrType");
    let cast_op = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![projected_ptr_ty],
        vec![elem_ptr],
        vec![],
        0,
    );
    cast_op.deref_mut(ctx).set_loc(loc);
    MirCastOp::new(cast_op).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
    cast_op.insert_after(ctx, addr_op);

    Ok((cast_op.deref(ctx).get_result(0), cast_op))
}

/// Build the fat value for a slice subslice from the original fat slice.
///
/// rustc's semantics for `from_end=true` are
/// `slice[from..slice.len() - to]`: advance the data pointer by `from` and
/// replace the metadata with `len - (from + to)`.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_slice_subslice_value(
    ctx: &mut Context,
    slice_value: Value,
    element_ty: TypeHandle,
    is_mutable: bool,
    from: u64,
    to: u64,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Ptr<Operation>)> {
    // Subslice is a projection of an existing fat pointer, not a new borrow.
    // Preserve the source pointer category; callers that explicitly form a
    // reborrow/raw address normalize the terminal result to their new Rust
    // type afterwards. A non-slice carrier keeps the caller's mutability with
    // Erased provenance.
    let origin = slice_value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<dialect_mir::types::MirSliceType>()
        .map(facts::pointer_origin_of_slice_carrier);

    use dialect_mir::ops::MirConstructSliceOp;

    let Some(trim) = from.checked_add(to) else {
        return input_err!(
            loc,
            TranslationErr::unsupported("Subslice bounds overflow usize metadata".to_string())
        );
    };

    let data_ptr_ty: TypeHandle = match origin {
        Some(origin) => facts::mint_generic_ptr_type(ctx, element_ty, origin).into(),
        None => dialect_mir::types::MirPtrType::get_generic(ctx, element_ty, is_mutable).into(),
    };
    let extract_ptr = Operation::new(
        ctx,
        MirExtractFieldOp::get_concrete_op_info(),
        vec![data_ptr_ty],
        vec![slice_value],
        vec![],
        0,
    );
    extract_ptr.deref_mut(ctx).set_loc(loc.clone());
    MirExtractFieldOp::new(extract_ptr)
        .set_attr_index(ctx, dialect_mir::attributes::FieldIndexAttr(0));
    match prev_op {
        Some(prev) => extract_ptr.insert_after(ctx, prev),
        None => extract_ptr.insert_at_front(block_ptr, ctx),
    }
    let data_ptr = extract_ptr.deref(ctx).get_result(0);

    let usize_ty = types::get_usize_type(ctx);
    let extract_len = Operation::new(
        ctx,
        MirExtractFieldOp::get_concrete_op_info(),
        vec![usize_ty.to_handle()],
        vec![slice_value],
        vec![],
        0,
    );
    extract_len.deref_mut(ctx).set_loc(loc.clone());
    MirExtractFieldOp::new(extract_len)
        .set_attr_index(ctx, dialect_mir::attributes::FieldIndexAttr(1));
    extract_len.insert_after(ctx, extract_ptr);
    let len = extract_len.deref(ctx).get_result(0);

    let (from_value, from_op) =
        emit_usize_constant(ctx, from, block_ptr, Some(extract_len), loc.clone());
    let offset_op = Operation::new(
        ctx,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![data_ptr_ty],
        vec![data_ptr, from_value],
        vec![],
        0,
    );
    offset_op.deref_mut(ctx).set_loc(loc.clone());
    offset_op.insert_after(ctx, from_op);
    let new_data = offset_op.deref(ctx).get_result(0);

    let (trim_value, trim_op) =
        emit_usize_constant(ctx, trim, block_ptr, Some(offset_op), loc.clone());
    let sub_op = Operation::new(
        ctx,
        MirSubOp::get_concrete_op_info(),
        vec![usize_ty.to_handle()],
        vec![len, trim_value],
        vec![],
        0,
    );
    sub_op.deref_mut(ctx).set_loc(loc.clone());
    sub_op.insert_after(ctx, trim_op);
    let new_len = sub_op.deref(ctx).get_result(0);

    let slice_ty = match origin {
        Some(origin) => facts::mint_slice_type(ctx, element_ty, origin),
        None => dialect_mir::types::MirSliceType::get_with_mutability(ctx, element_ty, is_mutable),
    };
    let construct = Operation::new(
        ctx,
        MirConstructSliceOp::get_concrete_op_info(),
        vec![slice_ty.into()],
        vec![new_data, new_len],
        vec![],
        0,
    );
    construct.deref_mut(ctx).set_loc(loc);
    construct.insert_after(ctx, sub_op);

    Ok((construct.deref(ctx).get_result(0), construct))
}

/// Materialize a sized array subslice from an SSA array value.
pub(super) fn emit_array_subslice_value(
    ctx: &mut Context,
    array_value: Value,
    from: u64,
    to: u64,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    let (element_ty, array_len) = {
        let array_ty = array_value.get_type(ctx);
        let array_ty = array_ty.deref(ctx);
        let Some(array_ty) = array_ty.downcast_ref::<dialect_mir::types::MirArrayType>() else {
            return input_err!(
                loc,
                TranslationErr::unsupported("Subslice value base is not MirArrayType".to_string())
            );
        };
        (array_ty.element_type(), array_ty.size())
    };

    let Some(projected_len) = to.checked_sub(from) else {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "array Subslice has inverted bounds: from={from}, to={to}"
            ))
        );
    };
    if to > array_len {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "array Subslice end {to} exceeds array length {array_len}"
            ))
        );
    }

    let mut elements = Vec::with_capacity(projected_len as usize);
    let mut current_prev = prev_op;
    for index in from..to {
        let field_index = u32::try_from(index).map_err(|_| {
            input_error_noloc!(TranslationErr::unsupported(format!(
                "array Subslice index {index} exceeds dialect field-index range"
            )))
        })?;
        let extract = Operation::new(
            ctx,
            MirExtractFieldOp::get_concrete_op_info(),
            vec![element_ty],
            vec![array_value],
            vec![],
            0,
        );
        extract.deref_mut(ctx).set_loc(loc.clone());
        MirExtractFieldOp::new(extract)
            .set_attr_index(ctx, dialect_mir::attributes::FieldIndexAttr(field_index));
        match current_prev {
            Some(prev) => extract.insert_after(ctx, prev),
            None => extract.insert_at_front(block_ptr, ctx),
        }
        elements.push(extract.deref(ctx).get_result(0));
        current_prev = Some(extract);
    }

    let projected_array_ty: TypeHandle =
        dialect_mir::types::MirArrayType::get(ctx, element_ty, projected_len).into();
    let construct = Operation::new(
        ctx,
        MirConstructArrayOp::get_concrete_op_info(),
        vec![projected_array_ty],
        elements,
        vec![],
        0,
    );
    construct.deref_mut(ctx).set_loc(loc);
    match current_prev {
        Some(prev) => construct.insert_after(ctx, prev),
        None => construct.insert_at_front(block_ptr, ctx),
    }

    Ok((construct.deref(ctx).get_result(0), Some(construct)))
}

/// Compute the in-memory address of `place` starting from its alloca `slot`.
///
/// Walks the projection chain and emits the correct pliron ops for each
/// element:
///
/// - `Field(idx, _)`   → [`MirFieldAddrOp`]
/// - `ConstantIndex {offset, from_end, ..}` → an element address. Forward
///   indexes materialize `offset` directly; from-end indexes on slices extract
///   the fat-pointer length and materialize `len - offset` at runtime.
/// - `Index(local)`    → `load_local(local)` + [`MirArrayElementAddrOp`]
///   (array pointee) or `load_local(local)` + [`MirPtrOffsetOp`] (slice data pointer)
/// - `Deref`           → `MirLoadOp` of the pointer (the loaded pointer IS
///   the pointee's address); subsequent projections apply to the pointee.
///   ZST pointees skip the load (SharedArray exception). Fat (slice-shaped)
///   pointees scalarize to a (data ptr, len) pair: a mid-chain fat deref
///   loads the whole fat value and extracts the thin data pointer (field 0)
///   so the walk continues against the ORIGINAL elements, while a trailing
///   fat deref (`&*s` reborrow) is just a load of the fat value.
///
/// `Downcast` records the variant for the `Field` immediately after it (the
/// pair addresses an enum payload through the flattened `all_field_types`
/// index); rustc guarantees that pairing, and any other continuation punts.
/// A SHARED borrow of a payload whose enum storage is canonical rather than
/// semantic (see [`enum_payload_needs_storage_coercion`]) also punts, so the
/// caller's value-copy fallback handles the read soundly instead of handing
/// out an address that cannot honor the storage coercion.
/// `Subslice` is handled for both sized arrays and slice fat pointers. Array
/// subslices keep an in-memory address; slice subslices advance the data
/// pointer and rebuild metadata as `len - (from + to)`, and can continue into
/// element Index/forward-ConstantIndex projections. From-end `ConstantIndex`
/// is accepted only when the immediately preceding fat-slice deref supplied
/// runtime length metadata; other bases (including a from-end index after a
/// Subslice) punt instead of guessing a length.
///
/// Returns `Ok(Some((addr, last_op)))` on success, `Ok(None)` if the
/// projection chain contains an element this helper doesn't know how to
/// turn into an address (the caller decides whether a value fallback is
/// sound or the construct must be rejected), or `Err` if something
/// structurally invalid happens (wrong pointee kind, unsupported type).
///
/// `is_mutable` governs the mutability of intermediate pointer types; the
/// final result pointer also carries this mutability.
fn translate_place_addr_from_slot(
    ctx: &mut Context,
    body: &mir::Body,
    value_map: &ValueMap,
    slot: Value,
    projection: &[mir::ProjectionElem],
    is_mutable: bool,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<Option<(Value, Option<Ptr<Operation>>)>> {
    use dialect_mir::ops::MirConstantOp;

    let mut current = slot;
    let mut current_prev_op = prev_op;
    let mut current_is_slice_data = false;
    // A fat-slice Deref may lower the immediately following Subslice while its
    // length metadata is still available. The next loop iteration then skips
    // that already-consumed projection and preserves slice-data provenance for
    // a following Index/ConstantIndex.
    let mut consumed_slice_subslice = false;
    // A fat DST Deref can consume the immediately following slice-tail Field
    // while rebuilding the `(data_ptr, len)` value needed for an indexed
    // element address. The next loop iteration skips that already-lowered
    // Field and preserves slice-data provenance for Index/ConstantIndex.
    let mut consumed_slice_tail_field = false;
    let mut current_slice_len: Option<Value> = None;
    // A fat reference to a slice-tailed struct carries the runtime length of
    // its final slice field. Preserve that metadata while walking nested
    // slice-tailed struct fields until the actual `[T]` field is reached.
    let mut carried_slice_tail_len: Option<Value> = None;
    // Set by a `Downcast` and consumed by the `Field` that follows it, which
    // is the only projection pair that can name an enum payload.
    let mut pending_variant: Option<usize> = None;

    for (proj_idx, elem) in projection.iter().enumerate() {
        // The slice-data provenance bit only describes the pointer produced by
        // the immediately-preceding `Deref` of a fat slice (index it by
        // element, not as a pointer to one array object). Capture it for this
        // iteration and clear it up front, so the invariant stays local: it is
        // true only when the previous step handed us a slice DATA pointer, and
        // no later projection arm can accidentally leak it forward.
        let entered_as_slice_data = current_is_slice_data;
        let entered_slice_len = current_slice_len.take();
        current_is_slice_data = false;

        // The preceding fat-DST Deref already lowered this slice-tail Field
        // so it could preserve the metadata before continuing to an element
        // index. Skip the syntactic Field here and carry the normalized
        // slice-data state into the following Index/ConstantIndex.
        if consumed_slice_tail_field {
            if !matches!(elem, mir::ProjectionElem::Field(_, _)) {
                return Ok(None);
            }
            consumed_slice_tail_field = false;
            current_is_slice_data = entered_as_slice_data;
            current_slice_len = entered_slice_len;
            continue;
        }

        // A `Downcast` names a variant only for the `Field` IMMEDIATELY
        // after it (rustc's MIR validator enforces the pairing). Any other
        // continuation is not a shape valid MIR produces; punt rather than
        // let a stale variant leak into a later `Field` arm.
        if pending_variant.is_some() && !matches!(elem, mir::ProjectionElem::Field(_, _)) {
            return Ok(None);
        }

        match elem {
            // `*place` -- the place walked so far holds a pointer; the
            // address of the dereferenced place is that pointer VALUE, so a
            // single `mir.load` of `current` yields it. Subsequent
            // projections then apply to the pointee.
            mir::ProjectionElem::Deref => {
                // Type of the place being dereferenced (= pointee of the
                // `current` address).
                let Some(place_ty) = mir_ptr_pointee(ctx, current.get_type(ctx)) else {
                    // `current` is not a pointer-typed address; punt to the
                    // caller.
                    return Ok(None);
                };
                let pointee_is_zst_tuple = is_empty_tuple_type(ctx, place_ty);
                let pointee_is_thin_ptr =
                    place_ty.deref(ctx).is::<dialect_mir::types::MirPtrType>();
                // Slice-shaped (fat) pointees carry their element type.
                let fat_elem_ty = slice_like_element_type(ctx, place_ty);

                if pointee_is_zst_tuple {
                    // ZST-pointee no-load exception (mirrors the Deref
                    // handling in `translate_place`, where it covers
                    // SharedArray): a pointer to a ZST *is* the runtime
                    // representation of the ZST place, so the deref adds no
                    // indirection. Keep `current` unchanged instead of
                    // emitting a meaningless load.
                    continue;
                }

                let is_last = proj_idx + 1 == projection.len();
                if let Some(elem_ty) = fat_elem_ty {
                    // Fat values (`&[T]`, `DisjointSlice<T>`, fat references
                    // to slice-tailed structs) are a (data pointer, element
                    // count) pair; dereferencing THROUGH them with a single
                    // `mir.load` would treat the pair as a thin address, a
                    // silent miscompile, so we never do that. What we CAN do:
                    //
                    // - Trailing `&*s` reborrow (the deref is the last
                    //   projection): the borrow result IS the fat value,
                    //   which lives whole in the slot, so the plain load
                    //   below is exactly right.
                    //
                    // - When the next projection is one we understand,
                    //   continue the walk by hand: load the fat PAIR,
                    //   extract its data pointer (field 0), and process the
                    //   following projection against that pointer. The data
                    //   pointer addresses the ORIGINAL elements, so both
                    //   shared and mutable borrows stay sound. This covers
                    //   field access through a fat reference to a
                    //   slice-tailed struct (the `(*iter).alive.start`
                    //   accesses inside `core::array::IntoIter::next`,
                    //   issue #138) and element access through a slice
                    //   reference (`(*slice)[i]`, including the inlined
                    //   body of `slice::get_mut`, issue #58).
                    //
                    // - Anything else keeps the loud failure (mutable) or
                    //   the value-copy fallback (shared).
                    if is_last {
                        // Fall through to the load below.
                    } else {
                        // Load the fat (ptr, len) pair from the slot.
                        let fat_load = Operation::new(
                            ctx,
                            MirLoadOp::get_concrete_op_info(),
                            vec![place_ty],
                            vec![current],
                            vec![],
                            0,
                        );
                        fat_load.deref_mut(ctx).set_loc(loc.clone());
                        match current_prev_op {
                            Some(p) => fat_load.insert_after(ctx, p),
                            None => fat_load.insert_at_front(block_ptr, ctx),
                        }
                        let fat_val = fat_load.deref(ctx).get_result(0);

                        if let mir::ProjectionElem::Subslice { from, to, from_end } =
                            &projection[proj_idx + 1]
                        {
                            // Slice Subslice needs both halves of the fat value,
                            // so lower it here before Deref would discard length
                            // metadata. Optimized MIR can continue with an
                            // Index/ConstantIndex, e.g. `(*s)[1:-1][0 of 1]`.
                            if !*from_end {
                                return Ok(None);
                            }
                            let (subslice, last_op) = emit_slice_subslice_value(
                                ctx,
                                fat_val,
                                elem_ty,
                                is_mutable,
                                *from,
                                *to,
                                block_ptr,
                                Some(fat_load),
                                loc.clone(),
                            )?;

                            if proj_idx + 2 == projection.len() {
                                return Ok(Some((subslice, Some(last_op))));
                            }

                            match &projection[proj_idx + 2] {
                                mir::ProjectionElem::Index(_)
                                | mir::ProjectionElem::ConstantIndex {
                                    from_end: false, ..
                                } => {
                                    let Some(data_ptr_ty) =
                                        erased_slice_data_pointer_type(ctx, subslice, elem_ty)
                                    else {
                                        return Ok(None);
                                    };
                                    let extract_ptr = Operation::new(
                                        ctx,
                                        MirExtractFieldOp::get_concrete_op_info(),
                                        vec![data_ptr_ty],
                                        vec![subslice],
                                        vec![],
                                        0,
                                    );
                                    extract_ptr.deref_mut(ctx).set_loc(loc.clone());
                                    MirExtractFieldOp::new(extract_ptr).set_attr_index(
                                        ctx,
                                        dialect_mir::attributes::FieldIndexAttr(0),
                                    );
                                    extract_ptr.insert_after(ctx, last_op);
                                    current = extract_ptr.deref(ctx).get_result(0);
                                    current_prev_op = Some(extract_ptr);
                                    current_is_slice_data = true;
                                    consumed_slice_subslice = true;
                                    continue;
                                }
                                _ => return Ok(None),
                            }
                        }

                        // Extract the data pointer (field 0 of the pair).
                        // Its pointee is the slice's element type: the
                        // struct itself for a fat struct reference, or the
                        // element for an ordinary `&[T]` / `DisjointSlice`.
                        let Some(data_ptr_ty) =
                            erased_slice_data_pointer_type(ctx, fat_val, elem_ty)
                        else {
                            return Ok(None);
                        };
                        let extract_ptr = Operation::new(
                            ctx,
                            MirExtractFieldOp::get_concrete_op_info(),
                            vec![data_ptr_ty],
                            vec![fat_val],
                            vec![],
                            0,
                        );
                        extract_ptr.deref_mut(ctx).set_loc(loc.clone());
                        MirExtractFieldOp::new(extract_ptr)
                            .set_attr_index(ctx, dialect_mir::attributes::FieldIndexAttr(0));
                        extract_ptr.insert_after(ctx, fat_load);
                        let data_ptr = extract_ptr.deref(ctx).get_result(0);
                        current_prev_op = Some(extract_ptr);

                        // Borrow or index of the struct's unsized slice tail,
                        // e.g. `&(*iter).data` or `&(*p).tail[k]`. No thin
                        // pointer can represent the whole tail: rebuild the
                        // `(tail pointer, len)` pair while the fat reference's
                        // metadata is still available.
                        if let mir::ProjectionElem::Field(field_idx, field_rust_ty) =
                            &projection[proj_idx + 1]
                            && let rustc_public::ty::TyKind::RigidTy(
                                rustc_public::ty::RigidTy::Slice(tail_elem_rust_ty),
                            ) = field_rust_ty.kind()
                        {
                            let tail_continuation = projection.get(proj_idx + 2);
                            if !matches!(
                                tail_continuation,
                                None | Some(&mir::ProjectionElem::Index(_))
                                    | Some(&mir::ProjectionElem::ConstantIndex { .. })
                            ) {
                                return Ok(None);
                            }

                            let tail_elem_ty = types::translate_type(ctx, &tail_elem_rust_ty)?;

                            // Address of the first tail element. The struct
                            // model stores the tail field with the ELEMENT
                            // type (see `translate_type`'s ADT arm), so the
                            // field-addr result is a pointer to the element
                            // and the dialect verifier agrees.
                            let tail_ptr_ty = projected_pointer_type(
                                ctx,
                                data_ptr.get_type(ctx),
                                tail_elem_ty,
                                /* legacy requested mutability, ignored */ is_mutable,
                            )
                            .expect("fat-pointer data extraction must yield a MirPtrType");
                            let tail_addr = dialect_mir::ops::MirFieldAddrOp::build(
                                ctx,
                                data_ptr,
                                tail_ptr_ty,
                                *field_idx as u32,
                            )?;
                            tail_addr.deref_mut(ctx).set_loc(loc.clone());
                            tail_addr.insert_after(ctx, extract_ptr);
                            let tail_ptr = tail_addr.deref(ctx).get_result(0);

                            // The element count (field 1 of the fat pair).
                            let (len_val, extract_len) = emit_slice_len_extract(
                                ctx,
                                fat_val,
                                block_ptr,
                                Some(tail_addr),
                                loc.clone(),
                            );

                            // Erased kind on purpose: this fat value is a projection-internal
                            // reconstruction of the DST tail. A Rust kind is established only
                            // when the value crosses an `Rvalue::Ref`/`AddressOf` boundary,
                            // which retypes it to the declared kind.
                            let tail_is_mutable = tail_ptr_ty
                                .deref(ctx)
                                .downcast_ref::<dialect_mir::types::MirPtrType>()
                                .expect("slice-tail projection must produce a MirPtrType")
                                .is_mutable;
                            let slice_ty = dialect_mir::types::MirSliceType::get_with_mutability(
                                ctx,
                                tail_elem_ty,
                                tail_is_mutable,
                            );
                            use dialect_mir::ops::MirConstructSliceOp;
                            let construct = Operation::new(
                                ctx,
                                MirConstructSliceOp::get_concrete_op_info(),
                                vec![slice_ty.into()],
                                vec![tail_ptr, len_val],
                                vec![],
                                0,
                            );
                            construct.deref_mut(ctx).set_loc(loc.clone());
                            construct.insert_after(ctx, extract_len);
                            let tail_slice = construct.deref(ctx).get_result(0);

                            // Whole-tail borrow/reborrow: return the fat value.
                            if tail_continuation.is_none() {
                                return Ok(Some((tail_slice, Some(construct))));
                            }

                            // Element borrow/write: normalize the rebuilt tail
                            // to its data pointer, then let the existing
                            // Index/ConstantIndex arms perform the element
                            // offset and return the real address.
                            let Some(indexed_data_ptr_ty) =
                                erased_slice_data_pointer_type(ctx, tail_slice, tail_elem_ty)
                            else {
                                return Ok(None);
                            };
                            let indexed_extract = Operation::new(
                                ctx,
                                MirExtractFieldOp::get_concrete_op_info(),
                                vec![indexed_data_ptr_ty],
                                vec![tail_slice],
                                vec![],
                                0,
                            );
                            indexed_extract.deref_mut(ctx).set_loc(loc.clone());
                            MirExtractFieldOp::new(indexed_extract)
                                .set_attr_index(ctx, dialect_mir::attributes::FieldIndexAttr(0));
                            indexed_extract.insert_after(ctx, construct);

                            current = indexed_extract.deref(ctx).get_result(0);
                            current_prev_op = Some(indexed_extract);
                            current_is_slice_data = true;
                            current_slice_len = if matches!(
                                tail_continuation,
                                Some(&mir::ProjectionElem::ConstantIndex { from_end: true, .. })
                            ) {
                                Some(len_val)
                            } else {
                                None
                            };
                            consumed_slice_tail_field = true;
                            continue;
                        }

                        match &projection[proj_idx + 1] {
                            // Sized field access: hand the data pointer to
                            // the field arm below. If the field is itself a
                            // slice-tailed DST, preserve the fat reference's
                            // metadata until the walk reaches the actual `[T]`
                            // tail field.
                            mir::ProjectionElem::Field(_, field_rust_ty) => {
                                if types::slice_tail_element_ty(field_rust_ty).is_some() {
                                    let (len, len_op) = emit_slice_len_extract(
                                        ctx,
                                        fat_val,
                                        block_ptr,
                                        current_prev_op,
                                        loc.clone(),
                                    );
                                    current_prev_op = Some(len_op);
                                    carried_slice_tail_len = Some(len);
                                }
                                current = data_ptr;
                                continue;
                            }
                            // Element access through a slice data pointer is
                            // pointer arithmetic over the slice element type.
                            // That remains true when the element type is
                            // itself an array (`&[[T; N]][i]`), where a
                            // type-only check would otherwise mistake the
                            // data pointer for a pointer to one array object
                            // and index inside row 0.
                            mir::ProjectionElem::Index(_) => {
                                current = data_ptr;
                                current_is_slice_data = true;
                                continue;
                            }
                            mir::ProjectionElem::ConstantIndex { from_end, .. } => {
                                current = data_ptr;
                                current_is_slice_data = true;
                                if *from_end {
                                    let (len, len_op) = emit_slice_len_extract(
                                        ctx,
                                        fat_val,
                                        block_ptr,
                                        current_prev_op,
                                        loc.clone(),
                                    );
                                    current_prev_op = Some(len_op);
                                    current_slice_len = Some(len);
                                }
                                continue;
                            }
                            // Unknown continuation: keep the conservative
                            // behaviour (loud failure for mutable borrows,
                            // value-copy fallback for shared ones).
                            _ => {
                                if is_mutable {
                                    return input_err!(
                                        loc,
                                        TranslationErr::unsupported(format!(
                                            "cannot compute a mutable in-memory address through \
                                             fat-pointer deref (projection {:?})",
                                            projection
                                        ))
                                    );
                                }
                                return Ok(None);
                            }
                        }
                    }
                } else if !pointee_is_thin_ptr {
                    // Deref of a non-pointer-typed place (a type the
                    // importer models by value); punt to the caller.
                    return Ok(None);
                }

                let load_op = Operation::new(
                    ctx,
                    MirLoadOp::get_concrete_op_info(),
                    vec![place_ty],
                    vec![current],
                    vec![],
                    0,
                );
                load_op.deref_mut(ctx).set_loc(loc.clone());
                match current_prev_op {
                    Some(p) => load_op.insert_after(ctx, p),
                    None => load_op.insert_at_front(block_ptr, ctx),
                }
                current = load_op.deref(ctx).get_result(0);
                current_prev_op = Some(load_op);
            }

            mir::ProjectionElem::Field(field_idx, field_ty) => {
                let tail_elem_rust_ty = match field_ty.kind() {
                    rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Slice(elem)) => {
                        Some(elem)
                    }
                    _ => None,
                };

                if let Some(tail_len) = carried_slice_tail_len.take() {
                    if let Some(tail_elem_rust_ty) = tail_elem_rust_ty {
                        // Element-address traversal through an unsized tail is
                        // intentionally left to #881. For #880 the address path
                        // only needs to materialize the whole nested slice tail.
                        if proj_idx + 1 != projection.len() || pending_variant.is_some() {
                            return Ok(None);
                        }

                        let tail_elem_ty = types::translate_type(ctx, &tail_elem_rust_ty)?;
                        let Some(tail_ptr_ty) = projected_pointer_type(
                            ctx,
                            current.get_type(ctx),
                            tail_elem_ty,
                            is_mutable,
                        ) else {
                            return Ok(None);
                        };

                        // The physical struct stores an unsized slice tail as
                        // its element type, so the field address must be `*T`,
                        // not `*[T]`.
                        let tail_addr = dialect_mir::ops::MirFieldAddrOp::build(
                            ctx,
                            current,
                            tail_ptr_ty,
                            *field_idx as u32,
                        )?;
                        tail_addr.deref_mut(ctx).set_loc(loc.clone());
                        match current_prev_op {
                            Some(previous) => tail_addr.insert_after(ctx, previous),
                            None => tail_addr.insert_at_front(block_ptr, ctx),
                        }
                        let tail_ptr = tail_addr.deref(ctx).get_result(0);

                        // Erased kind on purpose: this fat value is a projection-internal
                        // reconstruction of the DST tail. A Rust kind is established only
                        // when the value crosses an `Rvalue::Ref`/`AddressOf` boundary,
                        // which retypes it to the declared kind.
                        let tail_is_mutable = tail_ptr_ty
                            .deref(ctx)
                            .downcast_ref::<dialect_mir::types::MirPtrType>()
                            .expect("slice-tail projection must produce a MirPtrType")
                            .is_mutable;
                        let slice_ty = dialect_mir::types::MirSliceType::get_with_mutability(
                            ctx,
                            tail_elem_ty,
                            tail_is_mutable,
                        );
                        use dialect_mir::ops::MirConstructSliceOp;
                        let construct = Operation::new(
                            ctx,
                            MirConstructSliceOp::get_concrete_op_info(),
                            vec![slice_ty.into()],
                            vec![tail_ptr, tail_len],
                            vec![],
                            0,
                        );
                        construct.deref_mut(ctx).set_loc(loc.clone());
                        construct.insert_after(ctx, tail_addr);
                        return Ok(Some((construct.deref(ctx).get_result(0), Some(construct))));
                    }

                    // Metadata remains relevant only while descending through
                    // another slice-tailed DST field. A sized field projects
                    // away from the unsized tail, so discard it there.
                    if types::slice_tail_element_ty(field_ty).is_some() {
                        carried_slice_tail_len = Some(tail_len);
                    }
                }

                let field_type = types::translate_type(ctx, field_ty)?;

                // After a `Downcast`, the field belongs to that variant, and an
                // enum names its payload fields by position in the flattened
                // `all_field_types`. Translate the per-variant index into that
                // flat one; a non-enum pointee keeps the index as written.
                let pointee = current
                    .get_type(ctx)
                    .deref(ctx)
                    .downcast_ref::<dialect_mir::types::MirPtrType>()
                    .map(|ptr| ptr.pointee);
                let pointee_is_enum = pointee.is_some_and(|pointee| {
                    pointee.deref(ctx).is::<dialect_mir::types::MirEnumType>()
                });
                let flat_field_index = match pending_variant.take() {
                    Some(variant) => {
                        let flat = pointee.and_then(|pointee| {
                            pointee
                                .deref(ctx)
                                .downcast_ref::<dialect_mir::types::MirEnumType>()
                                .and_then(|enum_ty| enum_ty.flat_field_index(variant, *field_idx))
                        });
                        match flat {
                            Some(flat) => {
                                // A payload whose bytes use canonical storage
                                // that differs from its semantic type (bool
                                // leaves are i8 bytes, shared-memory pointer
                                // leaves are generic pointers) has no honest
                                // raw address: reads and writes through one
                                // are typed with the SEMANTIC type. For a
                                // SHARED borrow the value-copy fallback is
                                // sound and matches what the importer did
                                // before payload addressing existed, so punt.
                                // Mutable borrows and assignment stores keep
                                // the address path, where mir-lower's
                                // canonical-storage gate stays the loud,
                                // fail-closed authority (so a conservative
                                // miss here errors instead of miscompiling).
                                if !is_mutable
                                    && enum_payload_needs_storage_coercion(ctx, field_type)
                                {
                                    return Ok(None);
                                }
                                flat as u32
                            }
                            // A downcast over something this walker cannot
                            // resolve to an enum payload position. Punt rather
                            // than address the wrong bytes.
                            None => return Ok(None),
                        }
                    }
                    // Valid MIR never applies `Field` to an enum place without
                    // a `Downcast` naming the variant first (rustc's own place
                    // typing has no answer for it). `MirFieldAddrOp` reads an
                    // enum-pointee index as a FLATTENED (variant, field)
                    // position, so passing this raw per-variant index through
                    // could silently address another variant's payload. Only an
                    // importer bug or invalid MIR reaches here; fail loudly.
                    None if pointee_is_enum => {
                        return input_err!(
                            loc,
                            TranslationErr::unsupported(format!(
                                "Field projection on an enum place without a preceding \
                                 Downcast (projection {:?})",
                                projection
                            ))
                        );
                    }
                    None => *field_idx as u32,
                };

                // Field address computation must remain in the address space of the
                // aggregate pointer. LLVM GEP cannot change address spaces.
                let Some(result_ptr_ty) =
                    projected_pointer_type(ctx, current.get_type(ctx), field_type, is_mutable)
                else {
                    return Ok(None);
                };

                let op = dialect_mir::ops::MirFieldAddrOp::build(
                    ctx,
                    current,
                    result_ptr_ty,
                    flat_field_index,
                )?;
                op.deref_mut(ctx).set_loc(loc.clone());

                match current_prev_op {
                    Some(previous) => op.insert_after(ctx, previous),
                    None => op.insert_at_front(block_ptr, ctx),
                }

                current = op.deref(ctx).get_result(0);
                current_prev_op = Some(op);
            }

            mir::ProjectionElem::ConstantIndex {
                offset,
                min_length: _,
                from_end,
            } => {
                let (mut pointee_kind, addr_space) = match pointer_pointee_kind(ctx, current) {
                    Some(kind) => kind,
                    None => return Ok(None),
                };
                if entered_as_slice_data {
                    pointee_kind = PointeeKind::Direct;
                }

                let index_val = if *from_end {
                    // Never derive this from the pointee type. For
                    // `&[[T; N]]`, the pointee array length N is the row
                    // width, while this index selects a row from the outer
                    // runtime-length slice.
                    if !entered_as_slice_data {
                        return Ok(None);
                    }
                    let Some(slice_len) = entered_slice_len else {
                        return Ok(None);
                    };
                    let (index, sub_op) = emit_from_end_slice_index(
                        ctx,
                        slice_len,
                        *offset,
                        block_ptr,
                        current_prev_op,
                        loc.clone(),
                    )?;
                    current_prev_op = Some(sub_op);
                    index
                } else {
                    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signed);
                    let index_apint =
                        APInt::from_i64(*offset as i64, NonZeroUsize::new(64).unwrap());
                    let const_attr =
                        pliron::builtin::attributes::IntegerAttr::new(i64_ty, index_apint);
                    let const_op_ptr = Operation::new(
                        ctx,
                        MirConstantOp::get_concrete_op_info(),
                        vec![i64_ty.into()],
                        vec![],
                        vec![],
                        0,
                    );
                    const_op_ptr.deref_mut(ctx).set_loc(loc.clone());
                    MirConstantOp::new(const_op_ptr).set_attr_value(ctx, const_attr);
                    match current_prev_op {
                        Some(p) => const_op_ptr.insert_after(ctx, p),
                        None => const_op_ptr.insert_at_front(block_ptr, ctx),
                    }
                    current_prev_op = Some(const_op_ptr);
                    const_op_ptr.deref(ctx).get_result(0)
                };

                let (addr_op, next_current) = emit_indexed_element_addr(
                    ctx,
                    current,
                    index_val,
                    pointee_kind,
                    addr_space,
                    is_mutable,
                    block_ptr,
                    current_prev_op,
                    loc.clone(),
                );
                current = next_current;
                current_prev_op = Some(addr_op);
            }

            // Runtime `arr[i]` indexing. Without this arm, a place like
            // `&(*ptr).field[i]` would silently drop the `Index` projection
            // and return a pointer to the array's first slot, miscompiling
            // every load through the reference into a load of element 0.
            mir::ProjectionElem::Index(index_local) => {
                let (mut pointee_kind, addr_space) = match pointer_pointee_kind(ctx, current) {
                    Some(kind) => kind,
                    None => return Ok(None),
                };
                if entered_as_slice_data {
                    pointee_kind = PointeeKind::Direct;
                }

                let index_place = mir::Place {
                    local: *index_local,
                    projection: vec![],
                };
                let (index_val, next_prev_op) = translate_place(
                    ctx,
                    body,
                    &index_place,
                    value_map,
                    block_ptr,
                    current_prev_op,
                    loc.clone(),
                )?;
                current_prev_op = next_prev_op;

                let (addr_op, next_current) = emit_indexed_element_addr(
                    ctx,
                    current,
                    index_val,
                    pointee_kind,
                    addr_space,
                    is_mutable,
                    block_ptr,
                    current_prev_op,
                    loc.clone(),
                );
                current = next_current;
                current_prev_op = Some(addr_op);
            }

            mir::ProjectionElem::Subslice { from, to, from_end } => {
                if *from_end && consumed_slice_subslice {
                    consumed_slice_subslice = false;
                    current_is_slice_data = true;
                    continue;
                }
                // A slice Subslice must be lowered while the fat-pointer
                // metadata is available in the preceding Deref arm.
                if *from_end {
                    return Ok(None);
                }
                let (subslice_ptr, last_op) = emit_array_subslice_address(
                    ctx,
                    current,
                    *from,
                    *to,
                    is_mutable,
                    block_ptr,
                    current_prev_op,
                    loc.clone(),
                )?;
                current = subslice_ptr;
                current_prev_op = Some(last_op);
            }

            // Enum-variant downcast (`(x as Variant).field`). The downcast
            // itself moves no address: a payload shares the enum's storage, so
            // the variant only decides which field the next `Field` names.
            // Record it and let that arm resolve the flattened payload
            // position; lowering maps it to a slot or a byte offset through
            // the enum slot map.
            mir::ProjectionElem::Downcast(variant_idx) => {
                pending_variant = Some(variant_idx.to_index());
            }

            // Remaining projection kinds (OpaqueCast, Subtype) aren't lowered
            // to addresses here yet. Punt to the caller,
            // which decides between a value fallback (shared borrows) and a
            // hard error (mutable borrows).
            _ => return Ok(None),
        }
    }

    // A chain that ENDS on a `Downcast` never occurs in valid MIR (the
    // validator requires a `Field` after it). Punt rather than hand back the
    // enum's own address as if it were the variant's payload place.
    if pending_variant.is_some() || carried_slice_tail_len.is_some() {
        return Ok(None);
    }

    Ok(Some((current, current_prev_op)))
}

#[cfg(test)]
// Tests build kinded fixture types directly; production code mints via facts::PointerOrigin.
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;
    use dialect_mir::types::{MirPtrType, MirStructType};
    use pliron::builtin::types::FP32Type;

    /// The shared-borrow punt predicate must flag exactly the payload shapes
    /// whose enum storage differs from their semantic type: bool leaves and
    /// shared-memory pointer leaves, at any nesting depth. Canonical scalars
    /// and generic pointers must stay on the address path so shared reads of
    /// ordinary payloads keep compiling without a copy.
    #[test]
    fn payload_storage_coercion_predicate_flags_bool_and_shared_pointer_leaves() {
        use dialect_mir::types::{MirArrayType, MirTupleType};

        let mut ctx = Context::new();
        crate::translator::register_dialects(&mut ctx);

        let bool_ty: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        let shared_ptr: TypeHandle = MirPtrType::get_shared(&mut ctx, u32_ty, false).into();
        let generic_ptr: TypeHandle = MirPtrType::get_generic(&mut ctx, u32_ty, false).into();

        // Leaves.
        assert!(enum_payload_needs_storage_coercion(&ctx, bool_ty));
        assert!(enum_payload_needs_storage_coercion(&ctx, shared_ptr));
        assert!(!enum_payload_needs_storage_coercion(&ctx, u32_ty));
        assert!(!enum_payload_needs_storage_coercion(&ctx, f32_ty));
        assert!(!enum_payload_needs_storage_coercion(&ctx, generic_ptr));

        // Nesting: one flagged leaf taints the aggregate, and a clean
        // aggregate stays clean.
        let mixed_tuple: TypeHandle = MirTupleType::get(&mut ctx, vec![u32_ty, bool_ty]).into();
        let clean_tuple: TypeHandle = MirTupleType::get(&mut ctx, vec![u32_ty, f32_ty]).into();
        assert!(enum_payload_needs_storage_coercion(&ctx, mixed_tuple));
        assert!(!enum_payload_needs_storage_coercion(&ctx, clean_tuple));

        let bool_struct: TypeHandle = MirStructType::get(
            &mut ctx,
            "HasBool".into(),
            vec!["a".into(), "b".into()],
            vec![u32_ty, bool_ty],
        )
        .into();
        assert!(enum_payload_needs_storage_coercion(&ctx, bool_struct));

        let bool_array: TypeHandle = MirArrayType::get(&mut ctx, bool_ty, 4).into();
        let f32_array: TypeHandle = MirArrayType::get(&mut ctx, f32_ty, 4).into();
        assert!(enum_payload_needs_storage_coercion(&ctx, bool_array));
        assert!(!enum_payload_needs_storage_coercion(&ctx, f32_array));

        // Deep nesting: struct-of-tuple-of-shared-pointer.
        let inner: TypeHandle = MirTupleType::get(&mut ctx, vec![f32_ty, shared_ptr]).into();
        let deep: TypeHandle =
            MirStructType::get(&mut ctx, "Deep".into(), vec!["inner".into()], vec![inner]).into();
        assert!(enum_payload_needs_storage_coercion(&ctx, deep));
    }
}
