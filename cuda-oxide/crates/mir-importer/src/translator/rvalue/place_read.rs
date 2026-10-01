/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! [`translate_place`]: read-strategy classification and `apply_*`
//! projection helpers.

use super::const_bytes::translate_zero_sized_constant_value;
use super::const_enum::create_ghost_enum_default;
use super::place_addr::translate_place_address;
use super::place_iter::translate_place_iterative;
use super::pointee::{
    PointeeKind, indexed_element_ptr_type, is_empty_tuple_type, mir_ptr_pointee,
    pointer_type_pointee_kind, projected_pointer_type, rust_ty_is_slice, slice_like_element_type,
    tuple_has_over_aligned_zst_field,
};
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::facts;
use crate::translator::types;
use crate::translator::values::ValueMap;
use dialect_mir::ops::{MirExtractFieldOp, MirLoadOp, MirUndefOp};
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
use pliron::{input_err, input_error, input_error_noloc};
use rustc_public::mir;
use rustc_public::mir::ProjectionElem;
use rustc_public_bridge::IndexedVal;
use std::num::NonZeroUsize;

/// Translate MIR [`Place`](mir::Place) reads to pliron IR SSA [`Value`]s.
///
/// Reads for `Copy(place)` and `Move(place)` first ask a side-effect-free
/// classifier whether the place has a real final load address. Addressable
/// reads use the same
/// `translate_place_address` walker as refs, raw addresses, and writes, then
/// emit one `mir.load` at the end.
///
/// Places that are not representable as one final load address use the explicit
/// value fallback below. The fallback handles value-only projections such as
/// enum payload extraction, tuple field extraction, ZST reads, and no-slot
/// locals.
///
/// A bare slot-backed local is the trivial addressable case: it reads by
/// loading the local's alloca slot once. Projected reads compose address
/// operations for `field`, `index`, and `deref` when the whole projection
/// chain stays addressable.
///
/// # Value fallback and ghost locals
///
/// A local may have no backing slot in `value_map` if rustc optimised away its
/// assignment, or if the local is ZST and has no runtime footprint.
///
/// When such a local is still *used* within a block (e.g. `discriminant(_6)`)
/// and happens to be an enum, we synthesise a variant-0 default via
/// `create_ghost_enum_default`. Non-enum ghost locals currently produce an
/// error -- extend this match if new patterns appear in future toolchains.
///
/// This is the SSA equivalent of rustc's codegen reading an uninitialized
/// alloca, which produces LLVM `undef`.
///
/// # Returns
///
/// `(value, last_inserted_op)` -- the pliron IR value for the place and the last
/// operation inserted into the block (for op-ordering bookkeeping).
pub fn translate_place(
    ctx: &mut Context,
    body: &mir::Body,
    place: &mir::Place,
    value_map: &ValueMap,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    match classify_place_read_strategy(ctx, place, value_map)? {
        PlaceReadStrategy::Address => {
            if let Some((value, last_op)) = translate_place_load_from_address(
                ctx,
                body,
                place,
                value_map,
                block_ptr,
                prev_op,
                loc.clone(),
            )? {
                return Ok((value, last_op));
            }

            input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "place read {:?} was classified as addressable but did not lower to a \
                     final load address",
                    place.projection
                ))
            )
        }
        PlaceReadStrategy::ValueFallback => {
            translate_place_value_fallback(ctx, body, place, value_map, block_ptr, prev_op, loc)
        }
    }
}

// ============================================================================
// Place Read Strategy and Address Path
// ============================================================================

enum PlaceReadStrategy {
    /// Read the place by walking to its in-memory address and loading once.
    Address,
    /// Read the place through value projection because it is not representable
    /// as one final load address.
    ValueFallback,
}

/// Choose how to lower MIR place reads.
///
/// This deliberately does not emit IR. The address walker may create several
/// operations before discovering an unsupported projection, so read lowering
/// must not call it speculatively and then fall back to value projections.
///
/// Conservative fallback is part of the design. Enum payload addressing,
/// unsupported slice forms, ZST results, and computed/no-slot values remain on
/// the value path until they have dedicated address-lowering support.
fn classify_place_read_strategy(
    ctx: &mut Context,
    place: &mir::Place,
    value_map: &ValueMap,
) -> TranslationResult<PlaceReadStrategy> {
    let Some(slot) = value_map.get_slot(place.local) else {
        return Ok(PlaceReadStrategy::ValueFallback);
    };

    if place.projection.is_empty() {
        let Some(final_pointee) = mir_ptr_pointee(ctx, slot.get_type(ctx)) else {
            return Ok(PlaceReadStrategy::ValueFallback);
        };
        if types::is_zst_type(ctx, final_pointee) {
            return Ok(PlaceReadStrategy::ValueFallback);
        }
        return Ok(PlaceReadStrategy::Address);
    };

    let mut current_ptr_ty = slot.get_type(ctx);
    let mut current_is_slice_data = false;

    for (proj_idx, elem) in place.projection.iter().enumerate() {
        let entered_as_slice_data = current_is_slice_data;
        current_is_slice_data = false;

        match elem {
            mir::ProjectionElem::Deref => {
                let Some(place_ty) = mir_ptr_pointee(ctx, current_ptr_ty) else {
                    return Ok(PlaceReadStrategy::ValueFallback);
                };

                if is_empty_tuple_type(ctx, place_ty) {
                    continue;
                }

                if let Some(elem_ty) = slice_like_element_type(ctx, place_ty) {
                    let is_last = proj_idx + 1 == place.projection.len();
                    if is_last {
                        // The address walker returns a loaded fat value for a
                        // trailing slice-shaped deref. This helper handles only
                        // paths whose final result is an address to load from.
                        return Ok(PlaceReadStrategy::ValueFallback);
                    }

                    match &place.projection[proj_idx + 1] {
                        mir::ProjectionElem::Field(_, field_rust_ty) => {
                            if rust_ty_is_slice(field_rust_ty)
                                || types::slice_tail_element_ty(field_rust_ty).is_some()
                            {
                                // A direct slice tail or a nested slice-tailed DST needs
                                // fat-pointer metadata that the address read path does not carry
                                // across fields. Keep it on the value path.
                                return Ok(PlaceReadStrategy::ValueFallback);
                            }

                            current_ptr_ty = dialect_mir::types::MirPtrType::get_generic(
                                ctx, elem_ty, /* is_mutable */ false,
                            )
                            .into();
                        }
                        mir::ProjectionElem::Index(_)
                        | mir::ProjectionElem::ConstantIndex { .. } => {
                            current_ptr_ty = dialect_mir::types::MirPtrType::get_generic(
                                ctx, elem_ty, /* is_mutable */ false,
                            )
                            .into();
                            current_is_slice_data = true;
                        }
                        _ => return Ok(PlaceReadStrategy::ValueFallback),
                    }
                } else if place_ty.deref(ctx).is::<dialect_mir::types::MirPtrType>() {
                    current_ptr_ty = place_ty;
                } else {
                    return Ok(PlaceReadStrategy::ValueFallback);
                }
            }

            mir::ProjectionElem::Field(_, field_ty) => {
                let Some(pointee) = mir_ptr_pointee(ctx, current_ptr_ty) else {
                    return Ok(PlaceReadStrategy::ValueFallback);
                };
                let is_struct_or_tuple =
                    pointee.deref(ctx).is::<dialect_mir::types::MirStructType>()
                        || pointee.deref(ctx).is::<dialect_mir::types::MirTupleType>();
                if !is_struct_or_tuple {
                    // `mir.field_addr` verifies struct, tuple, union and enum
                    // pointees; anything else stays on the value path, where
                    // `mir.extract_field` supports it instead.
                    return Ok(PlaceReadStrategy::ValueFallback);
                }
                if tuple_has_over_aligned_zst_field(ctx, pointee) {
                    // Code-shape guard: the address path's final load states
                    // natural alignment only, while a zero-byte
                    // `repr(align(N))` field raises the tuple's ABI alignment
                    // without appearing in its LLVM storage type. Keep such
                    // tuples on the value path, which moves the whole
                    // aggregate at its recorded alignment. Gate shape from
                    // PR #715 (vyncint), with the byte-size gap closed.
                    return Ok(PlaceReadStrategy::ValueFallback);
                }

                let field_type = types::translate_type(ctx, field_ty)?;
                let Some(projected_ptr_ty) = projected_pointer_type(
                    ctx,
                    current_ptr_ty,
                    field_type,
                    /* is_mutable */ false,
                ) else {
                    return Ok(PlaceReadStrategy::ValueFallback);
                };

                current_ptr_ty = projected_ptr_ty;
            }

            mir::ProjectionElem::Index(_) => {
                let Some((mut pointee_kind, addr_space)) =
                    pointer_type_pointee_kind(ctx, current_ptr_ty)
                else {
                    return Ok(PlaceReadStrategy::ValueFallback);
                };
                if entered_as_slice_data {
                    pointee_kind = PointeeKind::Direct;
                }
                current_ptr_ty = indexed_element_ptr_type(
                    ctx,
                    current_ptr_ty,
                    pointee_kind,
                    addr_space,
                    /* is_mutable */ false,
                );
            }

            mir::ProjectionElem::ConstantIndex { from_end, .. } => {
                // rustc emits from-end ConstantIndex only for runtime-length
                // slices. Require the immediately preceding fat-slice deref;
                // arrays never reach stable MIR with `from_end=true`.
                if *from_end && !entered_as_slice_data {
                    return Ok(PlaceReadStrategy::ValueFallback);
                }
                let Some((mut pointee_kind, addr_space)) =
                    pointer_type_pointee_kind(ctx, current_ptr_ty)
                else {
                    return Ok(PlaceReadStrategy::ValueFallback);
                };
                if entered_as_slice_data {
                    pointee_kind = PointeeKind::Direct;
                }
                current_ptr_ty = indexed_element_ptr_type(
                    ctx,
                    current_ptr_ty,
                    pointee_kind,
                    addr_space,
                    /* is_mutable */ false,
                );
            }

            mir::ProjectionElem::Subslice { from, to, from_end } => {
                // PlaceElem::Subslice uses from_end=false for arrays and
                // from_end=true for slices. A slice subslice is unsized and
                // therefore is not a final load address; references/raw
                // pointers to it are handled by the address walker directly.
                if *from_end {
                    return Ok(PlaceReadStrategy::ValueFallback);
                }

                let (element_ty, array_len, addr_space) = {
                    let ptr_ty = current_ptr_ty.deref(ctx);
                    let Some(ptr_ty) = ptr_ty.downcast_ref::<dialect_mir::types::MirPtrType>()
                    else {
                        return Ok(PlaceReadStrategy::ValueFallback);
                    };
                    let array_ty = ptr_ty.pointee.deref(ctx);
                    let Some(array_ty) =
                        array_ty.downcast_ref::<dialect_mir::types::MirArrayType>()
                    else {
                        return Ok(PlaceReadStrategy::ValueFallback);
                    };
                    (
                        array_ty.element_type(),
                        array_ty.size(),
                        ptr_ty.address_space,
                    )
                };
                let Some(projected_len) = to.checked_sub(*from) else {
                    return Ok(PlaceReadStrategy::ValueFallback);
                };
                if *to > array_len {
                    return Ok(PlaceReadStrategy::ValueFallback);
                }

                let projected_array_ty: TypeHandle =
                    dialect_mir::types::MirArrayType::get(ctx, element_ty, projected_len).into();
                current_ptr_ty = dialect_mir::types::MirPtrType::get(
                    ctx,
                    projected_array_ty,
                    /* is_mutable */ false,
                    addr_space,
                )
                .into();
            }

            // Enum payload addressing is a value projection, not a final load
            // address. Other unknown projections stay on the conservative path.
            mir::ProjectionElem::Downcast(_) => return Ok(PlaceReadStrategy::ValueFallback),
            _ => return Ok(PlaceReadStrategy::ValueFallback),
        }
    }

    let Some(final_pointee) = mir_ptr_pointee(ctx, current_ptr_ty) else {
        return Ok(PlaceReadStrategy::ValueFallback);
    };
    if types::is_zst_type(ctx, final_pointee) {
        return Ok(PlaceReadStrategy::ValueFallback);
    }

    Ok(PlaceReadStrategy::Address)
}

/// Lower an addressable place read by computing its in-memory address, then
/// emitting one final `mir.load` from that address.
///
/// Returning `None` means the address walker did not produce a final load
/// address. `translate_place` treats that as a checker/walker divergence when
/// the classifier selected the address path.
#[allow(clippy::too_many_arguments)]
fn translate_place_load_from_address(
    ctx: &mut Context,
    body: &mir::Body,
    place: &mir::Place,
    value_map: &ValueMap,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<Option<(Value, Option<Ptr<Operation>>)>> {
    let Some((addr, addr_prev_op)) = translate_place_address(
        ctx,
        body,
        value_map,
        place,
        /* is_mutable */ false,
        block_ptr,
        prev_op,
        loc.clone(),
    )?
    else {
        return Ok(None);
    };

    let Some(pointee) = mir_ptr_pointee(ctx, addr.get_type(ctx)) else {
        return Ok(None);
    };
    if types::is_zst_type(ctx, pointee) {
        return Ok(None);
    }

    let load_op = Operation::new(
        ctx,
        MirLoadOp::get_concrete_op_info(),
        vec![pointee],
        vec![addr],
        vec![],
        0,
    );
    load_op.deref_mut(ctx).set_loc(loc);
    match addr_prev_op.or(prev_op) {
        Some(prev) => load_op.insert_after(ctx, prev),
        None => load_op.insert_at_front(block_ptr, ctx),
    }

    let value = load_op.deref(ctx).get_result(0);
    Ok(Some((value, Some(load_op))))
}

// ============================================================================
// Place Read Value Fallback
// ============================================================================

/// Explicit value-projection fallback for place reads that are not addressable.
///
/// Handles value-only reads, including enum downcast/payload extraction, tuple
/// field extraction, no-slot ghost locals, and ZST synthesis.
fn translate_place_value_fallback(
    ctx: &mut Context,
    body: &mir::Body,
    place: &mir::Place,
    value_map: &ValueMap,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    if place.projection.is_empty() {
        let local = place.local;
        // Alloca + load/store model: emit `mir.load slot`. Every non-ZST local
        // has a slot allocated in the entry block, so the loaded value is the
        // local's current contents. `mem2reg` promotes these loads back into
        // SSA form when the slot's address doesn't escape.
        if let Some((load_op, val)) = value_map.load_local(ctx, local, block_ptr, prev_op) {
            return Ok((val, Some(load_op)));
        }
        // ZST or unsupported local -- synthesise a value for it so callers
        // can uniformly consume a `Value`. An enum gets its variant-0 default
        // (ghost-enum), a struct/tuple ZST gets an empty aggregate. Loads of
        // these are otherwise meaningless.
        let local_decl = &body.locals()[local];
        let ty_ptr = types::translate_type(ctx, &local_decl.ty)?;
        if ty_ptr.deref(ctx).is::<dialect_mir::types::MirEnumType>() {
            let op = create_ghost_enum_default(ctx, ty_ptr, loc.clone());
            match prev_op {
                Some(p) => op.insert_after(ctx, p),
                None => op.insert_at_front(block_ptr, ctx),
            }
            let val = op.deref(ctx).get_result(0);
            return Ok((val, Some(op)));
        }
        if types::is_zst_type(ctx, ty_ptr) {
            return translate_zero_sized_constant_value(ctx, ty_ptr, block_ptr, prev_op, loc);
        }
        input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Local {} has no alloca slot and is not a ZST",
                Into::<usize>::into(local)
            ))
        )
    } else {
        // Handle projections (place.field, place[index], etc.)
        // For now, handle tuple field projections (_3.0, _3.1, etc.)
        if place.projection.len() == 1 {
            // Check if this is a tuple field projection
            match &place.projection[0] {
                ProjectionElem::Deref => {
                    // Dereference: *ptr
                    // The base value must be a pointer
                    let base_place = mir::Place {
                        local: place.local,
                        projection: vec![],
                    };
                    let (base_value, prev_op_after_base) = translate_place(
                        ctx,
                        body,
                        &base_place,
                        value_map,
                        block_ptr,
                        prev_op,
                        loc.clone(),
                    )?;

                    // Get the result type from the pointer's element type
                    let base_ty = base_value.get_type(ctx);

                    // Extract pointee info while holding the borrow, then release before fallback
                    let pointee_info: Option<(pliron::r#type::TypeHandle, bool)> = {
                        let base_ty_ref = base_ty.deref(ctx);
                        base_ty_ref
                            .downcast_ref::<dialect_mir::types::MirPtrType>()
                            .map(|ptr_ty| {
                                let pointee = ptr_ty.pointee;
                                let pointee_ref = pointee.deref(ctx);

                                // Check if pointee is a ZST (empty tuple) - this happens for SharedArray
                                // which is a zero-sized type. For ZSTs, dereferencing just returns the
                                // same pointer (there's nothing to load).
                                let is_empty_tuple = pointee_ref
                                    .downcast_ref::<dialect_mir::types::MirTupleType>()
                                    .is_some_and(|tt| tt.get_types().is_empty());

                                (pointee, is_empty_tuple)
                            })
                    };

                    let (res_ty, is_zst) = pointee_info.unwrap_or_else(|| {
                        // Fallback: assume i32 if we can't determine the type
                        (types::get_i32_type(ctx).to_handle(), false)
                    });

                    // For ZST pointees (like SharedArray), don't create a load op.
                    // Instead, just return the pointer itself - dereferencing a pointer
                    // to a ZST and taking a reference back gives the same pointer.
                    // NOTE: We still load from shared memory pointers (addrspace:3) -
                    // the ZST check only applies to SharedArray itself, not to data
                    // stored in shared memory.
                    if is_zst {
                        return Ok((base_value, prev_op_after_base));
                    }

                    let op = Operation::new(
                        ctx,
                        MirLoadOp::get_concrete_op_info(),
                        vec![res_ty],
                        vec![base_value],
                        vec![],
                        0,
                    );
                    op.deref_mut(ctx).set_loc(loc);

                    let load_op = MirLoadOp::new(op);

                    if let Some(prev) = prev_op_after_base {
                        load_op.get_operation().insert_after(ctx, prev);
                    } else {
                        load_op.get_operation().insert_at_front(block_ptr, ctx);
                    }

                    let loaded_val = load_op.get_operation().deref(ctx).get_result(0);

                    Ok((loaded_val, Some(load_op.get_operation())))
                }
                ProjectionElem::Field(field_idx, ty) => {
                    // Get the base value (the tuple/struct).
                    //
                    // In the alloca model the recursive call may emit a
                    // `mir.load <slot>` into the block to materialise the
                    // aggregate value; we must anchor our `mir.extract_field`
                    // **after** that load, otherwise the extract ends up
                    // before the load (and subsequent ops keep pushing the
                    // load past the block's terminator).
                    let base_place = mir::Place {
                        local: place.local,
                        projection: vec![],
                    };
                    let (base_value, prev_op_after_base) = translate_place(
                        ctx,
                        body,
                        &base_place,
                        value_map,
                        block_ptr,
                        prev_op,
                        loc.clone(),
                    )?;
                    let anchor = prev_op_after_base.or(prev_op);

                    let field_type = types::translate_type(ctx, ty)?;

                    let op = Operation::new(
                        ctx,
                        MirExtractFieldOp::get_concrete_op_info(),
                        vec![field_type],
                        vec![base_value],
                        vec![],
                        0,
                    );
                    op.deref_mut(ctx).set_loc(loc);

                    let extract_op = MirExtractFieldOp::new(op);
                    extract_op.set_attr_index(
                        ctx,
                        dialect_mir::attributes::FieldIndexAttr(*field_idx as u32),
                    );

                    if let Some(prev) = anchor {
                        extract_op.get_operation().insert_after(ctx, prev);
                    } else {
                        extract_op.get_operation().insert_at_front(block_ptr, ctx);
                    }

                    let field_value = extract_op.get_operation().deref(ctx).get_result(0);
                    Ok((field_value, Some(extract_op.get_operation())))
                }
                ProjectionElem::Downcast(_variant_idx) => {
                    // Downcast by itself is a no-op - it just narrows the type.
                    // The actual field extraction happens with the following Field projection.
                    // For now, just return the base value unchanged.
                    let base_place = mir::Place {
                        local: place.local,
                        projection: vec![],
                    };
                    translate_place(ctx, body, &base_place, value_map, block_ptr, prev_op, loc)
                }
                ProjectionElem::Index(index_local) => {
                    // Array indexing with a runtime index: array[index]
                    //
                    // Alloca model: `array` is backed by a stack slot whose
                    // pointee is `MirArrayType`, so we compute the element
                    // address from that slot directly (no MirRefOp needed)
                    // and load the element.

                    let mut current_prev = prev_op;

                    let Some(arr_ptr) = value_map.get_slot(place.local) else {
                        return input_err!(
                            loc,
                            TranslationErr::unsupported(format!(
                                "Array local {} has no alloca slot; cannot index",
                                Into::<usize>::into(place.local)
                            ))
                        );
                    };

                    // Get the index value
                    let index_place = mir::Place {
                        local: *index_local,
                        projection: vec![],
                    };
                    let (index_value, prev_op_after_index) = translate_place(
                        ctx,
                        body,
                        &index_place,
                        value_map,
                        block_ptr,
                        current_prev,
                        loc.clone(),
                    )?;
                    current_prev = prev_op_after_index;

                    // Get element type from pointer type
                    let arr_ptr_ty = arr_ptr.get_type(ctx);
                    let element_ty = {
                        let arr_ptr_ty_ref = arr_ptr_ty.deref(ctx);
                        let mir_ptr_ty = arr_ptr_ty_ref
                            .downcast_ref::<dialect_mir::types::MirPtrType>()
                            .expect("Memory array pointer should be MirPtrType");
                        let array_ty = mir_ptr_ty.pointee;
                        let array_ty_ref = array_ty.deref(ctx);
                        array_ty_ref
                            .downcast_ref::<dialect_mir::types::MirArrayType>()
                            .expect("Pointee should be MirArrayType")
                            .element_type()
                    };

                    // Projection keeps the slot address's machine mutability,
                    // address space, and pointer kind (or explicitly erases
                    // only the kind). A read does not require manufacturing
                    // an immutable address type.
                    let elem_ptr_ty = projected_pointer_type(
                        ctx, arr_ptr_ty, element_ty,
                        /* legacy requested mutability, ignored */ false,
                    )
                    .expect("array slot must be a MirPtrType");

                    // Create MirArrayElementAddrOp to get element pointer
                    use dialect_mir::ops::MirArrayElementAddrOp;
                    let addr_op = Operation::new(
                        ctx,
                        MirArrayElementAddrOp::get_concrete_op_info(),
                        vec![elem_ptr_ty],
                        vec![arr_ptr, index_value],
                        vec![],
                        0,
                    );
                    addr_op.deref_mut(ctx).set_loc(loc.clone());

                    if let Some(prev) = current_prev {
                        addr_op.insert_after(ctx, prev);
                    } else {
                        addr_op.insert_at_front(block_ptr, ctx);
                    }
                    current_prev = Some(addr_op);

                    let elem_ptr = addr_op.deref(ctx).get_result(0);

                    // Load the element value
                    use dialect_mir::ops::MirLoadOp;
                    let load_op = Operation::new(
                        ctx,
                        MirLoadOp::get_concrete_op_info(),
                        vec![element_ty],
                        vec![elem_ptr],
                        vec![],
                        0,
                    );
                    load_op.deref_mut(ctx).set_loc(loc);

                    if let Some(prev) = current_prev {
                        load_op.insert_after(ctx, prev);
                    } else {
                        load_op.insert_at_front(block_ptr, ctx);
                    }

                    let result = load_op.deref(ctx).get_result(0);
                    Ok((result, Some(load_op)))
                }
                ProjectionElem::ConstantIndex {
                    offset,
                    min_length: _,
                    from_end,
                } => {
                    // Array indexing with a compile-time constant index.
                    //
                    // Alloca model: the array local already has a `*mut [T; N]`
                    // slot, so compute the element address via
                    // `MirConstantOp` + `MirArrayElementAddrOp` and load.
                    // `mem2reg` collapses the resulting load-after-store pairs
                    // back into SSA extracts for promotable arrays.

                    let index = if *from_end {
                        return input_err!(
                            loc,
                            TranslationErr::unsupported(
                                "ConstantIndex with from_end=true not yet supported"
                            )
                        );
                    } else {
                        *offset as usize
                    };

                    // Load the current array value if we don't have a slot (ZST/edge case)
                    // so that we fall back to the SSA extract-field behaviour.
                    let Some(arr_ptr) = value_map.get_slot(place.local) else {
                        // ZST / no-slot fallback: materialise the whole
                        // aggregate and extract. Anchor the extract after
                        // whatever the base-place materialiser inserted.
                        let base_place = mir::Place {
                            local: place.local,
                            projection: vec![],
                        };
                        let (array_value, prev_op_after_base) = translate_place(
                            ctx,
                            body,
                            &base_place,
                            value_map,
                            block_ptr,
                            prev_op,
                            loc.clone(),
                        )?;
                        let anchor = prev_op_after_base.or(prev_op);

                        let array_ty = array_value.get_type(ctx);
                        let element_ty = {
                            let array_ty_ref = array_ty.deref(ctx);
                            if let Some(arr_ty) =
                                array_ty_ref.downcast_ref::<dialect_mir::types::MirArrayType>()
                            {
                                arr_ty.element_type()
                            } else {
                                return input_err!(
                                    loc,
                                    TranslationErr::unsupported(format!(
                                        "ConstantIndex projection on non-array type: {}",
                                        array_ty.disp(ctx)
                                    ))
                                );
                            }
                        };

                        let op = Operation::new(
                            ctx,
                            MirExtractFieldOp::get_concrete_op_info(),
                            vec![element_ty],
                            vec![array_value],
                            vec![],
                            0,
                        );
                        op.deref_mut(ctx).set_loc(loc);

                        let extract_op = MirExtractFieldOp::new(op);
                        extract_op.set_attr_index(
                            ctx,
                            dialect_mir::attributes::FieldIndexAttr(index as u32),
                        );

                        if let Some(prev) = anchor {
                            extract_op.get_operation().insert_after(ctx, prev);
                        } else {
                            extract_op.get_operation().insert_at_front(block_ptr, ctx);
                        }

                        let result = extract_op.get_operation().deref(ctx).get_result(0);
                        return Ok((result, Some(extract_op.get_operation())));
                    };

                    // Slot-backed path: GEP + load from the slot.
                    let mut current_prev = prev_op;

                    let (element_ty, arr_ptr_ty) = {
                        let arr_ptr_ty = arr_ptr.get_type(ctx);
                        let arr_ptr_ty_ref = arr_ptr_ty.deref(ctx);
                        let mir_ptr_ty = arr_ptr_ty_ref
                            .downcast_ref::<dialect_mir::types::MirPtrType>()
                            .ok_or_else(|| {
                                input_error!(
                                    loc.clone(),
                                    TranslationErr::unsupported(format!(
                                        "ConstantIndex base slot is not a pointer: {}",
                                        arr_ptr_ty.disp(ctx)
                                    ))
                                )
                            })?;
                        let array_ty_ref = mir_ptr_ty.pointee.deref(ctx);
                        let elem_ty = array_ty_ref
                            .downcast_ref::<dialect_mir::types::MirArrayType>()
                            .ok_or_else(|| {
                                input_error_noloc!(TranslationErr::unsupported(
                                    "ConstantIndex base slot pointee is not MirArrayType"
                                ))
                            })?
                            .element_type();
                        (elem_ty, arr_ptr_ty)
                    };

                    use dialect_mir::ops::MirConstantOp;
                    use pliron::builtin::attributes::IntegerAttr;

                    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signed);
                    let index_apint = APInt::from_i64(index as i64, NonZeroUsize::new(64).unwrap());
                    let index_attr = IntegerAttr::new(i64_ty, index_apint);

                    let const_op_ptr = Operation::new(
                        ctx,
                        MirConstantOp::get_concrete_op_info(),
                        vec![i64_ty.into()],
                        vec![],
                        vec![],
                        0,
                    );
                    const_op_ptr.deref_mut(ctx).set_loc(loc.clone());
                    MirConstantOp::new(const_op_ptr).set_attr_value(ctx, index_attr);
                    if let Some(prev) = current_prev {
                        const_op_ptr.insert_after(ctx, prev);
                    } else {
                        const_op_ptr.insert_at_front(block_ptr, ctx);
                    }
                    current_prev = Some(const_op_ptr);
                    let index_value = const_op_ptr.deref(ctx).get_result(0);

                    let elem_ptr_ty = projected_pointer_type(
                        ctx, arr_ptr_ty, element_ty,
                        /* legacy requested mutability, ignored */ false,
                    )
                    .expect("array slot must be a MirPtrType");

                    use dialect_mir::ops::MirArrayElementAddrOp;
                    let addr_op = Operation::new(
                        ctx,
                        MirArrayElementAddrOp::get_concrete_op_info(),
                        vec![elem_ptr_ty],
                        vec![arr_ptr, index_value],
                        vec![],
                        0,
                    );
                    addr_op.deref_mut(ctx).set_loc(loc.clone());
                    if let Some(prev) = current_prev {
                        addr_op.insert_after(ctx, prev);
                    } else {
                        addr_op.insert_at_front(block_ptr, ctx);
                    }
                    current_prev = Some(addr_op);
                    let elem_ptr = addr_op.deref(ctx).get_result(0);

                    let load_op = Operation::new(
                        ctx,
                        MirLoadOp::get_concrete_op_info(),
                        vec![element_ty],
                        vec![elem_ptr],
                        vec![],
                        0,
                    );
                    load_op.deref_mut(ctx).set_loc(loc);
                    if let Some(prev) = current_prev {
                        load_op.insert_after(ctx, prev);
                    } else {
                        load_op.insert_at_front(block_ptr, ctx);
                    }

                    let result = load_op.deref(ctx).get_result(0);
                    Ok((result, Some(load_op)))
                }
                _ => input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Projection element {:?} not yet implemented",
                        place.projection[0]
                    ))
                ),
            }
        } else {
            // Multi-level projections (2+): use iterative processing.
            // The iterative path handles Deref on slices (extracts data pointer),
            // Index/ConstantIndex on both arrays and pointers, Field, Downcast, etc.
            translate_place_iterative(ctx, body, place, value_map, block_ptr, prev_op, loc)
        }
    }
}

// ============================================================================
// Iterative Projection Helpers
// ============================================================================
// These functions support the iterative processing of MIR projections.
// Each projection element is handled independently, allowing arbitrary depth.

/// Apply a Deref projection: load from pointer.
pub(super) fn apply_deref_projection(
    ctx: &mut Context,
    ptr_value: Value,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    let ptr_ty = ptr_value.get_type(ctx);

    enum DerefKind {
        Ptr {
            pointee: pliron::r#type::TypeHandle,
            is_zst: bool,
        },
        Slice {
            element_ty: pliron::r#type::TypeHandle,
        },
    }

    let deref_kind = {
        let ptr_ty_ref = ptr_ty.deref(ctx);
        if let Some(mir_ptr_ty) = ptr_ty_ref.downcast_ref::<dialect_mir::types::MirPtrType>() {
            let pointee = mir_ptr_ty.pointee;
            let is_zst = pointee
                .deref(ctx)
                .downcast_ref::<dialect_mir::types::MirTupleType>()
                .is_some_and(|tt| tt.get_types().is_empty());
            Some(DerefKind::Ptr { pointee, is_zst })
        } else {
            ptr_ty_ref
                .downcast_ref::<dialect_mir::types::MirSliceType>()
                .map(|slice_ty| DerefKind::Slice {
                    element_ty: slice_ty.element_type(),
                })
        }
    };

    let deref_kind = deref_kind.ok_or_else(|| {
        let ty_dbg = format!("{:?}", ptr_ty.deref(ctx));
        input_error_noloc!(TranslationErr::unsupported(format!(
            "Deref projection on unsupported type in apply_deref_projection.\n\
             \n  pliron type: {}\n\
             \n  display    : {}\n\
             \n\
             \nDeref currently handles MirPtrType (thin pointer load) and MirSliceType\n\
             (fat pointer → extract data pointer). The type above matched neither.\n\
             A new handler may need to be added.",
            ty_dbg,
            ptr_ty.disp(ctx)
        )))
    })?;

    match deref_kind {
        DerefKind::Ptr { pointee, is_zst } => {
            if is_zst {
                return Ok((ptr_value, prev_op));
            }

            let op = Operation::new(
                ctx,
                MirLoadOp::get_concrete_op_info(),
                vec![pointee],
                vec![ptr_value],
                vec![],
                0,
            );
            op.deref_mut(ctx).set_loc(loc);
            let load_op = MirLoadOp::new(op);

            if let Some(prev) = prev_op {
                load_op.get_operation().insert_after(ctx, prev);
            } else {
                load_op.get_operation().insert_at_front(block_ptr, ctx);
            }

            Ok((
                load_op.get_operation().deref(ctx).get_result(0),
                Some(load_op.get_operation()),
            ))
        }

        DerefKind::Slice { element_ty } => {
            // Slices are unsized — we can't load `[T]` into an SSA value.
            // Extract the data pointer (field 0 of the fat pointer {ptr, len}).
            // Subsequent Index/ConstantIndex projections will do ptr arithmetic + load.
            let origin = ptr_value
                .get_type(ctx)
                .deref(ctx)
                .downcast_ref::<dialect_mir::types::MirSliceType>()
                .map(facts::pointer_origin_of_slice_carrier);
            let ptr_ty: TypeHandle = match origin {
                Some(origin) => facts::mint_generic_ptr_type(ctx, element_ty, origin).into(),
                None => dialect_mir::types::MirPtrType::get_generic(ctx, element_ty, false).into(),
            };

            let extract_op = Operation::new(
                ctx,
                MirExtractFieldOp::get_concrete_op_info(),
                vec![ptr_ty],
                vec![ptr_value],
                vec![],
                0,
            );
            extract_op.deref_mut(ctx).set_loc(loc);

            let extract = MirExtractFieldOp::new(extract_op);
            extract.set_attr_index(ctx, dialect_mir::attributes::FieldIndexAttr(0));

            if let Some(prev) = prev_op {
                extract.get_operation().insert_after(ctx, prev);
            } else {
                extract.get_operation().insert_at_front(block_ptr, ctx);
            }

            Ok((
                extract.get_operation().deref(ctx).get_result(0),
                Some(extract.get_operation()),
            ))
        }
    }
}

/// Apply a Field projection against a POINTER to the aggregate: compute the
/// field's address with `mir.field_addr` and load the field value.
///
/// Used when the projection walk holds an address rather than an aggregate
/// value, which happens after dereferencing a fat pointer (the unsized
/// pointee cannot be loaded whole, so the deref hands back the data
/// pointer; see `apply_deref_projection`).
pub(super) fn apply_field_addr_and_load(
    ctx: &mut Context,
    aggregate_ptr: Value,
    field_idx: mir::FieldIdx,
    field_ty: &rustc_public::ty::Ty,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    let field_type = types::translate_type(ctx, field_ty)?;
    let field_ptr_ty = projected_pointer_type(
        ctx,
        aggregate_ptr.get_type(ctx),
        field_type,
        /* legacy requested mutability, ignored */ false,
    )
    .ok_or_else(|| {
        input_error!(
            loc.clone(),
            TranslationErr::unsupported("field-address base is not a MIR pointer".to_string())
        )
    })?;

    let addr_op = dialect_mir::ops::MirFieldAddrOp::build(
        ctx,
        aggregate_ptr,
        field_ptr_ty,
        field_idx as u32,
    )?;
    addr_op.deref_mut(ctx).set_loc(loc.clone());
    match prev_op {
        Some(p) => addr_op.insert_after(ctx, p),
        None => addr_op.insert_at_front(block_ptr, ctx),
    }
    let field_ptr = addr_op.deref(ctx).get_result(0);

    let load_op = Operation::new(
        ctx,
        MirLoadOp::get_concrete_op_info(),
        vec![field_type],
        vec![field_ptr],
        vec![],
        0,
    );
    load_op.deref_mut(ctx).set_loc(loc);
    load_op.insert_after(ctx, addr_op);

    Ok((load_op.deref(ctx).get_result(0), Some(load_op)))
}

/// Apply a Field projection: extract field from struct/tuple.
pub(super) fn apply_field_projection(
    ctx: &mut Context,
    aggregate_value: Value,
    field_idx: mir::FieldIdx,
    field_ty: &rustc_public::ty::Ty,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    let field_type = types::translate_type(ctx, field_ty)?;

    let op = Operation::new(
        ctx,
        MirExtractFieldOp::get_concrete_op_info(),
        vec![field_type],
        vec![aggregate_value],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());

    let extract_op = MirExtractFieldOp::new(op);
    extract_op.set_attr_index(
        ctx,
        dialect_mir::attributes::FieldIndexAttr(field_idx as u32),
    );

    if let Some(prev) = prev_op {
        extract_op.get_operation().insert_after(ctx, prev);
    } else {
        extract_op.get_operation().insert_at_front(block_ptr, ctx);
    }

    let field_value = extract_op.get_operation().deref(ctx).get_result(0);

    Ok((field_value, Some(extract_op.get_operation())))
}

/// Apply a Field projection on an enum variant (after Downcast).
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_enum_field_projection_pub(
    ctx: &mut Context,
    enum_value: Value,
    enum_rust_ty: &rustc_public::ty::Ty,
    variant_idx: rustc_public::ty::VariantIdx,
    field_idx: mir::FieldIdx,
    field_ty: &rustc_public::ty::Ty,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    apply_enum_field_projection(
        ctx,
        enum_value,
        enum_rust_ty,
        variant_idx,
        field_idx,
        field_ty,
        block_ptr,
        prev_op,
        loc,
    )
}

pub(super) fn apply_enum_field_projection(
    ctx: &mut Context,
    enum_value: Value,
    enum_rust_ty: &rustc_public::ty::Ty,
    variant_idx: rustc_public::ty::VariantIdx,
    field_idx: mir::FieldIdx,
    field_ty: &rustc_public::ty::Ty,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    use dialect_mir::ops::MirEnumPayloadOp;

    let field_type = types::translate_type(ctx, field_ty)?;

    // Get the variant index
    // NOTE: variant_idx IS the index (0, 1, 2, ...), NOT the discriminant!
    // We just need to validate it's an ADT type, then use the index directly.
    let variant_idx_val: usize = match enum_rust_ty.kind() {
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Adt(_adt_def, _)) => {
            variant_idx.to_index()
        }
        _ => {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "Downcast on non-ADT type: {:?}",
                    enum_rust_ty
                ))
            );
        }
    };

    // A value inhabiting this variant cannot exist, so the read sits on a
    // dynamically dead path that rustc nevertheless keeps in MIR (e.g. the
    // `ControlFlow::Break(NeverShortCircuitResidual)` arm inside
    // `array::try_from_fn`). `mir.enum_payload` refuses uninhabited
    // variants by verification, so keep the dead path representable with a
    // typed undef instead — the same treatment `[T; 0]` extraction gets.
    let variant_is_uninhabited = {
        let enum_ty = enum_value.get_type(ctx);
        enum_ty
            .deref(ctx)
            .downcast_ref::<dialect_mir::types::MirEnumType>()
            .and_then(|enum_ty| enum_ty.variant_is_inhabited(variant_idx_val))
            .is_some_and(|inhabited| !inhabited)
    };
    if variant_is_uninhabited {
        let undef = MirUndefOp::new(ctx, field_type).get_operation();
        undef.deref_mut(ctx).set_loc(loc);
        match prev_op {
            Some(prev) => undef.insert_after(ctx, prev),
            None => undef.insert_at_front(block_ptr, ctx),
        }
        return Ok((undef.deref(ctx).get_result(0), Some(undef)));
    }

    let op = Operation::new(
        ctx,
        MirEnumPayloadOp::get_concrete_op_info(),
        vec![field_type],
        vec![enum_value],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());

    let payload_op = MirEnumPayloadOp::new(op);

    payload_op.set_attr_payload_variant_index(
        ctx,
        dialect_mir::attributes::VariantIndexAttr(variant_idx_val as u32),
    );
    payload_op.set_attr_payload_field_index(
        ctx,
        dialect_mir::attributes::FieldIndexAttr(field_idx as u32),
    );

    if let Some(prev) = prev_op {
        payload_op.get_operation().insert_after(ctx, prev);
    } else {
        payload_op.get_operation().insert_at_front(block_ptr, ctx);
    }

    let payload_value = payload_op.get_operation().deref(ctx).get_result(0);

    Ok((payload_value, Some(payload_op.get_operation())))
}
