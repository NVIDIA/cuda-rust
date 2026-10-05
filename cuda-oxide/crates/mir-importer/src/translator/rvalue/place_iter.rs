/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! [`translate_place_iterative`]: the iterative place value walker.

use super::const_bytes::translate_zero_sized_constant_value;
use super::const_enum::create_ghost_enum_default;
use super::place_addr::{
    emit_array_subslice_value, emit_slice_len_extract, emit_slice_subslice_value,
};
use super::place_read::{
    apply_deref_projection, apply_enum_field_projection, apply_field_addr_and_load,
    apply_field_projection, translate_place,
};
use super::pointee::{normalize_slice_value_to_data_ptr, projected_pointer_type, rust_ty_is_slice};
use super::static_global::get_static_pointer_info;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::types;
use crate::translator::values::ValueMap;
use dialect_mir::ops::{MirExtractFieldOp, MirLoadOp, MirPtrOffsetOp};
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
use pliron::{input_err, input_error};
use rustc_public::mir;
use rustc_public::mir::ProjectionElem;
use std::num::NonZeroUsize;

/// Translate a MIR Place using iterative projection processing.
/// This handles arbitrary depth projections by processing each element in sequence.
pub fn translate_place_iterative(
    ctx: &mut Context,
    body: &mir::Body,
    place: &mir::Place,
    value_map: &ValueMap,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    // Start with the base local's current value. In the alloca model every
    // non-ZST local has a stack slot, so we emit `mir.load` once here and
    // then layer projections on top of the loaded SSA value; `mem2reg` folds
    // the load back into a direct SSA use when the slot is promotable. ZST /
    // unsupported locals fall back to the same ghost-enum / empty-aggregate
    // synthesis as [`translate_place`].
    let local = place.local;
    let (mut current_value, mut current_prev_op) = match value_map
        .load_local(ctx, local, block_ptr, prev_op)
    {
        Some((load_op, val)) => (val, Some(load_op)),
        None => {
            let local_decl = &body.locals()[local];
            let ty_ptr = types::translate_type(ctx, &local_decl.ty)?;
            if ty_ptr.deref(ctx).is::<dialect_mir::types::MirEnumType>() {
                let synth_op = create_ghost_enum_default(ctx, ty_ptr, loc.clone());
                match prev_op {
                    Some(p) => synth_op.insert_after(ctx, p),
                    None => synth_op.insert_at_front(block_ptr, ctx),
                }
                let val = synth_op.deref(ctx).get_result(0);
                (val, Some(synth_op))
            } else if types::is_zst_type(ctx, ty_ptr) {
                translate_zero_sized_constant_value(ctx, ty_ptr, block_ptr, prev_op, loc.clone())?
            } else {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Local {} has no alloca slot and is not a ZST",
                        Into::<usize>::into(local)
                    ))
                );
            }
        }
    };

    // Track the Rust type of `current_value` alongside the pliron value.
    // Each iteration below advances it through rustc_public's own projection
    // typing (`ProjectionElem::ty`) AFTER the arm has processed the element,
    // so every arm observes the type *before* its own projection applies and
    // the next iteration sees the narrowed type. `Downcast` deliberately
    // leaves the type unchanged (still the enum ADT), which is exactly what
    // `apply_enum_field_projection` expects when the following `Field` fires.
    //
    // This single fold is the only place `current_rust_ty` is updated;
    // individual arms must not update it themselves. Per-arm updates were
    // the cause of issue #131: only `Field` advanced the type, so chains
    // like `[Index, Downcast, Field]` (from `match xs[i]` over an array of
    // enums) handed the stale Array type to the Downcast/Field handler,
    // which bailed with "Downcast on non-ADT type: Array". The same
    // staleness affected `Deref` and `ConstantIndex`.
    let mut current_rust_ty = body.locals()[local].ty;

    // Track pending downcast (Downcast is a no-op, but we need variant info for Field on enums)
    // Type inferred from ProjectionElem::Downcast pattern
    let mut pending_downcast = None;

    // When Deref of a slice is immediately followed by Subslice, keep the fat
    // value intact for one iteration so Subslice can update both data and len.
    // The ordinary Deref helper intentionally drops len and returns only the
    // data pointer because Index/ConstantIndex do not need metadata.
    let mut preserved_slice_deref_mutability: Option<bool> = None;

    // A fat reference to a slice-tailed struct carries the runtime length of
    // that tail in field 1. Keep the metadata alongside the projected address
    // until the walk reaches the actual slice field. This mirrors rustc's
    // `llextra` model and lets metadata survive through nested slice-tailed
    // struct fields such as `outer.inner.tail[k]`.
    let mut carried_slice_tail_len: Option<Value> = None;

    // Process each projection element iteratively
    for (proj_idx, projection) in place.projection.iter().enumerate() {
        match projection {
            ProjectionElem::Deref => {
                let next_is_slice_subslice = matches!(
                    place.projection.get(proj_idx + 1),
                    Some(ProjectionElem::Subslice { from_end: true, .. })
                );
                let current_is_fat_slice = current_value
                    .get_type(ctx)
                    .deref(ctx)
                    .is::<dialect_mir::types::MirSliceType>();
                let pointer_info = get_static_pointer_info(&current_rust_ty);
                let slice_deref_mutability = pointer_info.as_ref().and_then(|(pointee, origin)| {
                    rust_ty_is_slice(pointee).then_some(origin.is_mutable())
                });
                let current_is_slice_tail_ref = pointer_info
                    .as_ref()
                    .is_some_and(|(pointee, _)| types::slice_tail_element_ty(pointee).is_some());

                if next_is_slice_subslice
                    && current_is_fat_slice
                    && slice_deref_mutability.is_some()
                {
                    // Preserve the fat pair. The following Subslice consumes
                    // both data and len and advances `current_rust_ty` normally.
                    preserved_slice_deref_mutability = slice_deref_mutability;
                    carried_slice_tail_len = None;
                } else {
                    // `&S<[T]>` uses the same fat-pair carrier as a slice, but
                    // field 0 points at the struct prefix while field 1 is the
                    // trailing slice length. Extract that metadata before Deref
                    // reduces the value to the struct data pointer. Unlike the
                    // old one-step handoff, keep it alive across any nested
                    // slice-tailed struct fields until the real `[T]` field is
                    // reached.
                    carried_slice_tail_len = if current_is_fat_slice && current_is_slice_tail_ref {
                        let (len, len_op) = emit_slice_len_extract(
                            ctx,
                            current_value,
                            block_ptr,
                            current_prev_op,
                            loc.clone(),
                        );
                        current_prev_op = Some(len_op);
                        Some(len)
                    } else {
                        None
                    };

                    (current_value, current_prev_op) = apply_deref_projection(
                        ctx,
                        current_value,
                        block_ptr,
                        current_prev_op,
                        loc.clone(),
                    )?;
                    preserved_slice_deref_mutability = None;
                }
                pending_downcast = None;
            }

            ProjectionElem::Field(field_idx, field_ty) => {
                // Check if this is a field access on an enum (preceded by Downcast).
                if let Some(variant_idx) = pending_downcast.take() {
                    if carried_slice_tail_len.is_some() {
                        return input_err!(
                            loc,
                            TranslationErr::unsupported(
                                "slice-tail metadata cannot cross an enum Downcast/Field projection"
                                    .to_string()
                            )
                        );
                    }

                    // Enum variant field access - use MirEnumPayloadOp.
                    (current_value, current_prev_op) = apply_enum_field_projection(
                        ctx,
                        current_value,
                        &current_rust_ty,
                        variant_idx,
                        *field_idx,
                        field_ty,
                        block_ptr,
                        current_prev_op,
                        loc.clone(),
                    )?;
                } else {
                    let tail_elem_rust_ty = match field_ty.kind() {
                        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Slice(
                            elem,
                        )) => Some(elem),
                        _ => None,
                    };
                    let field_is_slice_tailed_adt =
                        types::slice_tail_element_ty(field_ty).is_some();

                    if let Some(tail_elem_rust_ty) = tail_elem_rust_ty
                        && let Some(tail_len) = carried_slice_tail_len.take()
                    {
                        let tail_elem_ty = types::translate_type(ctx, &tail_elem_rust_ty)?;

                        // The struct model stores an unsized `[T]` tail as the
                        // element type `T`, because the elements live inline after
                        // the sized prefix. Address the field as `*T`, not `*[T]`.
                        use dialect_mir::ops::MirConstructSliceOp;
                        let tail_ptr_ty = projected_pointer_type(
                            ctx,
                            current_value.get_type(ctx),
                            tail_elem_ty,
                            /* legacy requested mutability, ignored */ false,
                        )
                        .expect("slice-tail field base must be a MirPtrType");
                        let tail_addr = dialect_mir::ops::MirFieldAddrOp::build(
                            ctx,
                            current_value,
                            tail_ptr_ty,
                            *field_idx as u32,
                        )?;
                        tail_addr.deref_mut(ctx).set_loc(loc.clone());
                        match current_prev_op {
                            Some(prev) => tail_addr.insert_after(ctx, prev),
                            None => tail_addr.insert_at_front(block_ptr, ctx),
                        }
                        let tail_ptr = tail_addr.deref(ctx).get_result(0);

                        // Reconstruct the semantic `[T]` value from the inline tail
                        // address plus the metadata carried from the outer fat reference,
                        // with an intentionally Erased pointer kind: a Rust kind is
                        // established only at the `Rvalue::Ref`/`AddressOf` boundary.
                        // A following Index or ConstantIndex can scalarize this
                        // MirSliceType to field 0 and reuse the existing pointer-offset
                        // + load path.
                        let tail_is_mutable = tail_ptr_ty
                            .deref(ctx)
                            .downcast_ref::<dialect_mir::types::MirPtrType>()
                            .expect("slice-tail projection must produce a MirPtrType")
                            .is_mutable;
                        let tail_slice_ty = dialect_mir::types::MirSliceType::get_with_mutability(
                            ctx,
                            tail_elem_ty,
                            tail_is_mutable,
                        );
                        let construct_tail = Operation::new(
                            ctx,
                            MirConstructSliceOp::get_concrete_op_info(),
                            vec![tail_slice_ty.into()],
                            vec![tail_ptr, tail_len],
                            vec![],
                            0,
                        );
                        construct_tail.deref_mut(ctx).set_loc(loc.clone());
                        construct_tail.insert_after(ctx, tail_addr);
                        current_value = construct_tail.deref(ctx).get_result(0);
                        current_prev_op = Some(construct_tail);
                    } else if carried_slice_tail_len.is_some() && field_is_slice_tailed_adt {
                        // The selected field is itself an unsized struct whose final
                        // field eventually contains the same slice tail. Such a field
                        // cannot be loaded as an SSA aggregate. Advance only the
                        // address and keep the metadata riding alongside it for the
                        // next projection step.
                        if !current_value
                            .get_type(ctx)
                            .deref(ctx)
                            .is::<dialect_mir::types::MirPtrType>()
                        {
                            return input_err!(
                                loc,
                                TranslationErr::unsupported(format!(
                                    "slice-tail metadata reached nested DST field {:?}, but the current value is not an address",
                                    field_ty
                                ))
                            );
                        }

                        let field_type = types::translate_type(ctx, field_ty)?;
                        let field_ptr_ty = projected_pointer_type(
                            ctx,
                            current_value.get_type(ctx),
                            field_type,
                            /* legacy requested mutability, ignored */ false,
                        )
                        .expect("nested DST field base must be a MirPtrType");
                        let field_addr = dialect_mir::ops::MirFieldAddrOp::build(
                            ctx,
                            current_value,
                            field_ptr_ty,
                            *field_idx as u32,
                        )?;
                        field_addr.deref_mut(ctx).set_loc(loc.clone());
                        match current_prev_op {
                            Some(prev) => field_addr.insert_after(ctx, prev),
                            None => field_addr.insert_at_front(block_ptr, ctx),
                        }
                        current_value = field_addr.deref(ctx).get_result(0);
                        current_prev_op = Some(field_addr);
                    } else {
                        // A sized field does not inherit DST metadata. Once the
                        // projection leaves the unsized-tail path, drop the
                        // metadata and lower the field normally.
                        carried_slice_tail_len = None;

                        let current_is_ptr = current_value
                            .get_type(ctx)
                            .deref(ctx)
                            .is::<dialect_mir::types::MirPtrType>();
                        if current_is_ptr {
                            // `current_value` is an ADDRESS, not an aggregate
                            // value. This happens after dereferencing a fat
                            // pointer: `apply_deref_projection` cannot load an
                            // unsized pointee, so it hands back the data
                            // pointer instead (e.g. reading
                            // `(*iter).alive.start` through the fat
                            // `&mut PolymorphicIter<[MaybeUninit<T>]>` inside
                            // `core::array::IntoIter::next`, issue #138).
                            // Compute the field's address and load the field.
                            (current_value, current_prev_op) = apply_field_addr_and_load(
                                ctx,
                                current_value,
                                *field_idx,
                                field_ty,
                                block_ptr,
                                current_prev_op,
                                loc.clone(),
                            )?;
                        } else {
                            // Regular struct/tuple field access.
                            (current_value, current_prev_op) = apply_field_projection(
                                ctx,
                                current_value,
                                *field_idx,
                                field_ty,
                                block_ptr,
                                current_prev_op,
                                loc.clone(),
                            )?;
                        }
                    }
                }
            }

            ProjectionElem::Downcast(variant_idx) => {
                if carried_slice_tail_len.is_some() {
                    return input_err!(
                        loc,
                        TranslationErr::unsupported(
                            "slice-tail metadata reached Downcast before the unsized slice field"
                                .to_string()
                        )
                    );
                }
                // Downcast is a no-op - it just narrows the type for the next Field access
                // Store the variant index for use by the next Field projection
                pending_downcast = Some(*variant_idx);
                // Don't change current_value
            }

            ProjectionElem::Index(index_local) => {
                if carried_slice_tail_len.is_some() {
                    return input_err!(
                        loc,
                        TranslationErr::unsupported(
                            "slice-tail metadata reached Index before the unsized slice field"
                                .to_string()
                        )
                    );
                }
                let index_place = mir::Place {
                    local: *index_local,
                    projection: vec![],
                };
                let (index_value, next_prev_op) = translate_place(
                    ctx,
                    body,
                    &index_place,
                    value_map,
                    block_ptr,
                    current_prev_op,
                    loc.clone(),
                )?;
                current_prev_op = next_prev_op;

                // A projected unsized slice tail is represented as a fat
                // `MirSliceType` value. Runtime Index only needs the data
                // pointer, so normalize it before reusing the pointer path.
                (current_value, current_prev_op) = normalize_slice_value_to_data_ptr(
                    ctx,
                    current_value,
                    block_ptr,
                    current_prev_op,
                    loc.clone(),
                );

                // Determine indexable kind upfront so we drop the immutable borrow
                // before creating operations (which need &mut ctx).
                enum IndexableKind {
                    Array {
                        element_ty: TypeHandle,
                    },
                    Ptr {
                        element_ty: TypeHandle,
                        ptr_ty: TypeHandle,
                    },
                }

                let cur_ty = current_value.get_type(ctx);
                let kind = {
                    let cur_ty_ref = cur_ty.deref(ctx);
                    if let Some(arr_ty) =
                        cur_ty_ref.downcast_ref::<dialect_mir::types::MirArrayType>()
                    {
                        Ok(IndexableKind::Array {
                            element_ty: arr_ty.element_type(),
                        })
                    } else if let Some(ptr_ty) =
                        cur_ty_ref.downcast_ref::<dialect_mir::types::MirPtrType>()
                    {
                        Ok(IndexableKind::Ptr {
                            element_ty: ptr_ty.pointee,
                            ptr_ty: cur_ty,
                        })
                    } else {
                        let ty_dbg = format!("{:?}", cur_ty_ref);
                        Err(ty_dbg)
                    }
                };

                match kind {
                    Ok(IndexableKind::Array { element_ty }) => {
                        use dialect_mir::ops::MirExtractArrayElementOp;
                        let op = Operation::new(
                            ctx,
                            MirExtractArrayElementOp::get_concrete_op_info(),
                            vec![element_ty],
                            vec![current_value, index_value],
                            vec![],
                            0,
                        );
                        op.deref_mut(ctx).set_loc(loc.clone());

                        if let Some(prev) = current_prev_op {
                            op.insert_after(ctx, prev);
                        } else {
                            op.insert_at_front(block_ptr, ctx);
                        }

                        current_value = op.deref(ctx).get_result(0);
                        current_prev_op = Some(op);
                    }
                    Ok(IndexableKind::Ptr { element_ty, ptr_ty }) => {
                        let offset_op = Operation::new(
                            ctx,
                            MirPtrOffsetOp::get_concrete_op_info(),
                            vec![ptr_ty],
                            vec![current_value, index_value],
                            vec![],
                            0,
                        );
                        offset_op.deref_mut(ctx).set_loc(loc.clone());
                        if let Some(prev) = current_prev_op {
                            offset_op.insert_after(ctx, prev);
                        } else {
                            offset_op.insert_at_front(block_ptr, ctx);
                        }
                        current_prev_op = Some(offset_op);
                        let offset_ptr = offset_op.deref(ctx).get_result(0);

                        let load_op = Operation::new(
                            ctx,
                            MirLoadOp::get_concrete_op_info(),
                            vec![element_ty],
                            vec![offset_ptr],
                            vec![],
                            0,
                        );
                        load_op.deref_mut(ctx).set_loc(loc.clone());
                        let load = MirLoadOp::new(load_op);
                        if let Some(prev) = current_prev_op {
                            load.get_operation().insert_after(ctx, prev);
                        } else {
                            load.get_operation().insert_at_front(block_ptr, ctx);
                        }

                        current_value = load.get_operation().deref(ctx).get_result(0);
                        current_prev_op = Some(load.get_operation());
                    }
                    Err(ty_dbg) => {
                        return input_err!(
                            loc,
                            TranslationErr::unsupported(format!(
                                "Index projection on unsupported type.\n\
                                 \n  pliron type: {}\n\
                                 \n  display    : {}\n\
                                 \n\
                                 \nIndex handles MirArrayType (extract_array_element) and MirPtrType\n\
                                 (pointer offset + load, e.g. after Deref on a slice). The type above\n\
                                 matched neither. A new handler may need to be added.",
                                ty_dbg,
                                cur_ty.disp(ctx)
                            ))
                        );
                    }
                }
                pending_downcast = None;
            }

            ProjectionElem::ConstantIndex {
                offset,
                min_length: _,
                from_end,
            } => {
                if carried_slice_tail_len.is_some() {
                    return input_err!(
                        loc,
                        TranslationErr::unsupported(
                            "slice-tail metadata reached ConstantIndex before the unsized slice field"
                                .to_string()
                        )
                    );
                }
                if *from_end {
                    return input_err!(
                        loc,
                        TranslationErr::unsupported(
                            "ConstantIndex with from_end=true not yet supported"
                        )
                    );
                }
                let index = *offset as usize;

                // A projected unsized slice tail is represented as a fat
                // `MirSliceType` value. ConstantIndex only needs the data
                // pointer, so normalize it before reusing the pointer path.
                (current_value, current_prev_op) = normalize_slice_value_to_data_ptr(
                    ctx,
                    current_value,
                    block_ptr,
                    current_prev_op,
                    loc.clone(),
                );

                // Determine indexable kind upfront so we drop the immutable borrow
                // before creating operations (which need &mut ctx).
                enum ConstIndexKind {
                    Array {
                        element_ty: TypeHandle,
                    },
                    Ptr {
                        element_ty: TypeHandle,
                        ptr_ty: TypeHandle,
                    },
                }

                let cur_ty = current_value.get_type(ctx);
                let kind = {
                    let cur_ty_ref = cur_ty.deref(ctx);
                    if let Some(arr_ty) =
                        cur_ty_ref.downcast_ref::<dialect_mir::types::MirArrayType>()
                    {
                        Ok(ConstIndexKind::Array {
                            element_ty: arr_ty.element_type(),
                        })
                    } else if let Some(ptr_ty) =
                        cur_ty_ref.downcast_ref::<dialect_mir::types::MirPtrType>()
                    {
                        Ok(ConstIndexKind::Ptr {
                            element_ty: ptr_ty.pointee,
                            ptr_ty: cur_ty,
                        })
                    } else {
                        let ty_dbg = format!("{:?}", cur_ty_ref);
                        Err(ty_dbg)
                    }
                };

                match kind {
                    Ok(ConstIndexKind::Array { element_ty }) => {
                        let op = Operation::new(
                            ctx,
                            MirExtractFieldOp::get_concrete_op_info(),
                            vec![element_ty],
                            vec![current_value],
                            vec![],
                            0,
                        );
                        op.deref_mut(ctx).set_loc(loc.clone());
                        let extract_op = MirExtractFieldOp::new(op);
                        extract_op.set_attr_index(
                            ctx,
                            dialect_mir::attributes::FieldIndexAttr(index as u32),
                        );

                        if let Some(prev) = current_prev_op {
                            extract_op.get_operation().insert_after(ctx, prev);
                        } else {
                            extract_op.get_operation().insert_at_front(block_ptr, ctx);
                        }

                        current_value = extract_op.get_operation().deref(ctx).get_result(0);
                        current_prev_op = Some(extract_op.get_operation());
                    }
                    Ok(ConstIndexKind::Ptr { element_ty, ptr_ty }) => {
                        // Create constant index value
                        let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
                        let apint = APInt::from_u32(index as u32, NonZeroUsize::new(32).unwrap());
                        let index_attr =
                            pliron::builtin::attributes::IntegerAttr::new(i32_ty, apint);
                        use dialect_mir::ops::MirConstantOp;
                        let const_op = Operation::new(
                            ctx,
                            MirConstantOp::get_concrete_op_info(),
                            vec![i32_ty.into()],
                            vec![],
                            vec![],
                            0,
                        );
                        const_op.deref_mut(ctx).set_loc(loc.clone());
                        let const_mir = MirConstantOp::new(const_op);
                        const_mir.set_attr_value(ctx, index_attr);
                        if let Some(prev) = current_prev_op {
                            const_mir.get_operation().insert_after(ctx, prev);
                        } else {
                            const_mir.get_operation().insert_at_front(block_ptr, ctx);
                        }
                        current_prev_op = Some(const_mir.get_operation());
                        let index_value = const_mir.get_operation().deref(ctx).get_result(0);

                        // Pointer offset
                        let offset_op = Operation::new(
                            ctx,
                            MirPtrOffsetOp::get_concrete_op_info(),
                            vec![ptr_ty],
                            vec![current_value, index_value],
                            vec![],
                            0,
                        );
                        offset_op.deref_mut(ctx).set_loc(loc.clone());
                        if let Some(prev) = current_prev_op {
                            offset_op.insert_after(ctx, prev);
                        } else {
                            offset_op.insert_at_front(block_ptr, ctx);
                        }
                        current_prev_op = Some(offset_op);
                        let offset_ptr = offset_op.deref(ctx).get_result(0);

                        // Load element
                        let load_op = Operation::new(
                            ctx,
                            MirLoadOp::get_concrete_op_info(),
                            vec![element_ty],
                            vec![offset_ptr],
                            vec![],
                            0,
                        );
                        load_op.deref_mut(ctx).set_loc(loc.clone());
                        let load = MirLoadOp::new(load_op);
                        if let Some(prev) = current_prev_op {
                            load.get_operation().insert_after(ctx, prev);
                        } else {
                            load.get_operation().insert_at_front(block_ptr, ctx);
                        }

                        current_value = load.get_operation().deref(ctx).get_result(0);
                        current_prev_op = Some(load.get_operation());
                    }
                    Err(ty_dbg) => {
                        return input_err!(
                            loc,
                            TranslationErr::unsupported(format!(
                                "ConstantIndex projection on unsupported type.\n\
                                 \n  pliron type: {}\n\
                                 \n  display    : {}\n\
                                 \n  index      : {}\n\
                                 \n\
                                 \nConstantIndex handles MirArrayType (extractvalue) and MirPtrType\n\
                                 (pointer offset + load, e.g. after Deref on a slice). The type above\n\
                                 matched neither. A new handler may need to be added.",
                                ty_dbg,
                                cur_ty.disp(ctx),
                                index
                            ))
                        );
                    }
                }
                pending_downcast = None;
            }

            ProjectionElem::Subslice { from, to, from_end } => {
                if carried_slice_tail_len.is_some() {
                    return input_err!(
                        loc,
                        TranslationErr::unsupported(
                            "slice-tail metadata reached Subslice before the unsized slice field"
                                .to_string()
                        )
                    );
                }
                if *from_end {
                    let Some(is_mutable) = preserved_slice_deref_mutability.take() else {
                        return input_err!(
                            loc,
                            TranslationErr::unsupported(
                                "slice Subslice reached iterative lowering without preserved fat-pointer metadata"
                                    .to_string()
                            )
                        );
                    };
                    let element_ty = {
                        let current_ty = current_value.get_type(ctx);
                        let current_ty = current_ty.deref(ctx);
                        let Some(slice_ty) =
                            current_ty.downcast_ref::<dialect_mir::types::MirSliceType>()
                        else {
                            return input_err!(
                                loc,
                                TranslationErr::unsupported(
                                    "slice Subslice preserved value is not MirSliceType"
                                        .to_string()
                                )
                            );
                        };
                        slice_ty.element_type()
                    };

                    let (subslice, last_op) = emit_slice_subslice_value(
                        ctx,
                        current_value,
                        element_ty,
                        is_mutable,
                        *from,
                        *to,
                        block_ptr,
                        current_prev_op,
                        loc.clone(),
                    )?;
                    current_value = subslice;
                    current_prev_op = Some(last_op);

                    // A subsequent element projection needs only the adjusted
                    // data pointer. Keep a terminal Subslice as a fat value so
                    // PtrMetadata/borrow users can observe the rebuilt length.
                    if matches!(
                        place.projection.get(proj_idx + 1),
                        Some(ProjectionElem::Index(_))
                            | Some(ProjectionElem::ConstantIndex {
                                from_end: false,
                                ..
                            })
                    ) {
                        let data_ptr_ty: TypeHandle = dialect_mir::types::MirPtrType::get_generic(
                            ctx, element_ty, is_mutable,
                        )
                        .into();
                        let extract_ptr = Operation::new(
                            ctx,
                            MirExtractFieldOp::get_concrete_op_info(),
                            vec![data_ptr_ty],
                            vec![current_value],
                            vec![],
                            0,
                        );
                        extract_ptr.deref_mut(ctx).set_loc(loc.clone());
                        MirExtractFieldOp::new(extract_ptr)
                            .set_attr_index(ctx, dialect_mir::attributes::FieldIndexAttr(0));
                        extract_ptr.insert_after(ctx, last_op);
                        current_value = extract_ptr.deref(ctx).get_result(0);
                        current_prev_op = Some(extract_ptr);
                    }
                } else {
                    (current_value, current_prev_op) = emit_array_subslice_value(
                        ctx,
                        current_value,
                        *from,
                        *to,
                        block_ptr,
                        current_prev_op,
                        loc.clone(),
                    )?;
                    preserved_slice_deref_mutability = None;
                }
                pending_downcast = None;
            }

            // Unsupported projection types
            other => {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Projection element {:?} not yet implemented in iterative mode",
                        other
                    ))
                );
            }
        }

        // Advance the running Rust type with rustc_public's own projection
        // typing (single source of truth; see the comment on
        // `current_rust_ty` above). For well-formed MIR this never fails;
        // if it does, surface the projection element and the type it was
        // applied to so the bail-out is actionable.
        current_rust_ty = projection.ty(current_rust_ty).map_err(|e| {
            input_error!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "Failed to type projection {:?} applied to {:?}: {:?}",
                    projection, current_rust_ty, e
                ))
            )
        })?;
    }

    if carried_slice_tail_len.is_some() {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "place projection ended on a slice-tailed DST before its metadata was attached to the slice field"
                    .to_string()
            )
        );
    }

    Ok((current_value, current_prev_op))
}
