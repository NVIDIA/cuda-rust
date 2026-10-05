/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! [`translate_rvalue`]: the main rvalue dispatch.

use super::aggregate::translate_aggregate_rvalue;
use super::coerce::cast_to_declared_rust_pointer_type_if_needed;
use super::const_bytes::translate_zero_sized_constant_value;
use super::const_enum::create_ghost_enum_default;
use super::fn_ptr::{translate_closure_fn_pointer, translate_reify_fn_pointer};
use super::operand::translate_operand;
use super::place_addr::translate_place_address;
use super::place_read::translate_place;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::facts;
use crate::translator::types;
use crate::translator::values::ValueMap;
use dialect_mir::attributes::{MirCastKindAttr, MirPointerKindAuthorityAttr};
use dialect_mir::ops::{
    MirAddOp, MirBitAndOp, MirBitOrOp, MirBitXorOp, MirCastOp, MirCheckedAddOp, MirCheckedMulOp,
    MirCheckedSubOp, MirCmpOp, MirConstructArrayOp, MirDivOp, MirEqOp, MirExtractFieldOp, MirGeOp,
    MirGtOp, MirLeOp, MirLtOp, MirMulOp, MirNeOp, MirNegOp, MirNotOp, MirPtrOffsetOp, MirRefOp,
    MirRemOp, MirShlOp, MirShrOp, MirSubOp, MirUndefOp,
};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{FP32Type, FP64Type, IntegerType};
use pliron::context::{Context, Ptr};
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::Typed;
use pliron::value::Value;
use pliron::{input_err, input_error, input_error_noloc};
use rustc_public::mir;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CheckedBinaryOpKind {
    Add,
    Sub,
    Mul,
}

fn classify_checked_binary_op(bin_op: &mir::BinOp) -> Result<CheckedBinaryOpKind, String> {
    match bin_op {
        mir::BinOp::Add => Ok(CheckedBinaryOpKind::Add),
        mir::BinOp::Sub => Ok(CheckedBinaryOpKind::Sub),
        mir::BinOp::Mul => Ok(CheckedBinaryOpKind::Mul),
        _ => Err(format!("CheckedBinaryOp {:?} not yet implemented", bin_op)),
    }
}

/// Translates a MIR rvalue to pliron IR operation(s).
///
/// # Returns
///
/// Tuple of `(Option<op>, result_value, last_inserted)`:
/// - `op`: The main operation (None for `Rvalue::Use`)
/// - `result_value`: The SSA value produced
/// - `last_inserted`: Last inserted helper op (for operation ordering)
///
/// The operation is created but **not inserted** - caller must insert it.
pub fn translate_rvalue(
    ctx: &mut Context,
    body: &mir::Body,
    rvalue: &mir::Rvalue,
    value_map: &mut ValueMap,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Option<Ptr<Operation>>, Value, Option<Ptr<Operation>>)> {
    match rvalue {
        mir::Rvalue::BinaryOp(bin_op, left, right) => {
            let (left_val, prev_op_after_left) =
                translate_operand(ctx, body, left, value_map, block_ptr, prev_op, loc.clone())?;
            let (right_val, prev_op_after_right) = translate_operand(
                ctx,
                body,
                right,
                value_map,
                block_ptr,
                prev_op_after_left,
                loc.clone(),
            )?;

            // Check if this is a comparison operation that may need type coercion
            let is_comparison = matches!(
                bin_op,
                mir::BinOp::Eq
                    | mir::BinOp::Ne
                    | mir::BinOp::Lt
                    | mir::BinOp::Le
                    | mir::BinOp::Gt
                    | mir::BinOp::Ge
            );

            // For comparison operations, handle type mismatches by casting the right operand
            // to match the left operand's type. This commonly occurs when comparing enum
            // discriminants (u8) against isize constants in Rust's MIR.
            let (final_right_val, final_prev_op) = if is_comparison {
                let left_type = left_val.get_type(ctx);
                let right_type = right_val.get_type(ctx);

                if left_type != right_type {
                    // Insert a cast operation to coerce right to match left's type
                    let cast_op = Operation::new(
                        ctx,
                        MirCastOp::get_concrete_op_info(),
                        vec![left_type],
                        vec![right_val],
                        vec![],
                        0,
                    );
                    cast_op.deref_mut(ctx).set_loc(loc.clone());
                    let coercion_kind = {
                        let l = left_type.deref(ctx);
                        let r = right_type.deref(ctx);
                        if l.downcast_ref::<IntegerType>().is_some()
                            && r.downcast_ref::<IntegerType>().is_some()
                        {
                            MirCastKindAttr::IntToInt
                        } else if l.downcast_ref::<FP32Type>().is_some()
                            || l.downcast_ref::<FP64Type>().is_some()
                        {
                            if r.downcast_ref::<FP32Type>().is_some()
                                || r.downcast_ref::<FP64Type>().is_some()
                            {
                                MirCastKindAttr::FloatToFloat
                            } else {
                                MirCastKindAttr::Transmute
                            }
                        } else if l.downcast_ref::<dialect_mir::types::MirPtrType>().is_some()
                            && r.downcast_ref::<dialect_mir::types::MirPtrType>().is_some()
                        {
                            MirCastKindAttr::PtrToPtr
                        } else {
                            MirCastKindAttr::Transmute
                        }
                    };
                    MirCastOp::new(cast_op).set_attr_cast_kind(ctx, coercion_kind);

                    // Insert the cast op after the right operand was processed
                    if let Some(prev) = prev_op_after_right {
                        cast_op.insert_after(ctx, prev);
                    } else {
                        cast_op.insert_at_front(block_ptr, ctx);
                    }

                    let casted_right = cast_op.deref(ctx).get_result(0);
                    (casted_right, Some(cast_op))
                } else {
                    (right_val, prev_op_after_right)
                }
            } else {
                (right_val, prev_op_after_right)
            };

            // Determine result type and operation
            // Comparison operations return bool (i1), arithmetic ops return operand type
            let (op_id, result_type) = match bin_op {
                // Arithmetic operations - return same type as operands
                // Unchecked variants are identical - overflow check is elided at MIR level
                mir::BinOp::Add | mir::BinOp::AddUnchecked => {
                    (MirAddOp::get_concrete_op_info(), left_val.get_type(ctx))
                }
                mir::BinOp::Sub | mir::BinOp::SubUnchecked => {
                    (MirSubOp::get_concrete_op_info(), left_val.get_type(ctx))
                }
                mir::BinOp::Mul | mir::BinOp::MulUnchecked => {
                    (MirMulOp::get_concrete_op_info(), left_val.get_type(ctx))
                }
                mir::BinOp::Div => (MirDivOp::get_concrete_op_info(), left_val.get_type(ctx)),
                mir::BinOp::Rem => (MirRemOp::get_concrete_op_info(), left_val.get_type(ctx)),

                // Comparison operations - return bool (i1)
                mir::BinOp::Lt => (
                    MirLtOp::get_concrete_op_info(),
                    types::get_bool_type(ctx).to_handle(),
                ),
                mir::BinOp::Le => (
                    MirLeOp::get_concrete_op_info(),
                    types::get_bool_type(ctx).to_handle(),
                ),
                mir::BinOp::Gt => (
                    MirGtOp::get_concrete_op_info(),
                    types::get_bool_type(ctx).to_handle(),
                ),
                mir::BinOp::Ge => (
                    MirGeOp::get_concrete_op_info(),
                    types::get_bool_type(ctx).to_handle(),
                ),
                mir::BinOp::Eq => (
                    MirEqOp::get_concrete_op_info(),
                    types::get_bool_type(ctx).to_handle(),
                ),
                mir::BinOp::Ne => (
                    MirNeOp::get_concrete_op_info(),
                    types::get_bool_type(ctx).to_handle(),
                ),
                // Three-way comparison (`Ord::cmp`) - returns
                // `core::cmp::Ordering`. rustc's `BinOp::ty` knows the
                // result type of every binop (including `Cmp`, for which it
                // returns the `Ordering` enum), so derive it locally from
                // the operand types instead of threading the assignment
                // destination type through every translate_rvalue caller.
                mir::BinOp::Cmp => {
                    let left_ty = left.ty(body.locals()).map_err(|e| {
                        pliron::input_error!(
                            loc.clone(),
                            TranslationErr::unsupported(format!(
                                "Failed to resolve BinOp::Cmp lhs type: {:?}",
                                e
                            ))
                        )
                    })?;
                    let right_ty = right.ty(body.locals()).map_err(|e| {
                        pliron::input_error!(
                            loc.clone(),
                            TranslationErr::unsupported(format!(
                                "Failed to resolve BinOp::Cmp rhs type: {:?}",
                                e
                            ))
                        )
                    })?;
                    let ordering_ty = bin_op.ty(left_ty, right_ty);
                    (
                        MirCmpOp::get_concrete_op_info(),
                        types::translate_type(ctx, &ordering_ty)?,
                    )
                }

                // Pointer offset - ptr.add(n) returns ptr + n * sizeof(element)
                mir::BinOp::Offset => (
                    MirPtrOffsetOp::get_concrete_op_info(),
                    left_val.get_type(ctx), // Result is same pointer type
                ),

                // Shift operations - result is same as left operand type
                // Unchecked variants are identical - overflow check is elided at MIR level
                mir::BinOp::Shr | mir::BinOp::ShrUnchecked => {
                    (MirShrOp::get_concrete_op_info(), left_val.get_type(ctx))
                }
                mir::BinOp::Shl | mir::BinOp::ShlUnchecked => {
                    (MirShlOp::get_concrete_op_info(), left_val.get_type(ctx))
                }

                // Bitwise operations - result is same as operand type
                mir::BinOp::BitAnd => (MirBitAndOp::get_concrete_op_info(), left_val.get_type(ctx)),
                mir::BinOp::BitOr => (MirBitOrOp::get_concrete_op_info(), left_val.get_type(ctx)),
                mir::BinOp::BitXor => (MirBitXorOp::get_concrete_op_info(), left_val.get_type(ctx)),
            };

            let op = Operation::new(
                ctx,
                op_id,
                vec![result_type],
                vec![left_val, final_right_val],
                vec![],
                0,
            );
            op.deref_mut(ctx).set_loc(loc);

            let result = op.deref(ctx).get_result(0);

            Ok((Some(op), result, final_prev_op))
        }
        mir::Rvalue::UnaryOp(un_op, operand) => {
            match un_op {
                mir::UnOp::PtrMetadata => {
                    // PtrMetadata extracts the length from a slice (fat pointer)
                    // For a slice &[T], this is field 1 (field 0 is the pointer, field 1 is length)
                    let (operand_val, prev_op_after_operand) = translate_operand(
                        ctx,
                        body,
                        operand,
                        value_map,
                        block_ptr,
                        prev_op,
                        loc.clone(),
                    )?;

                    // Result type is usize (the length)
                    let result_type = types::get_usize_type(ctx);

                    // Create an extract field operation to get field 1 (length)
                    let op = Operation::new(
                        ctx,
                        MirExtractFieldOp::get_concrete_op_info(),
                        vec![result_type.to_handle()],
                        vec![operand_val],
                        vec![],
                        0,
                    );
                    op.deref_mut(ctx).set_loc(loc.clone());

                    let extract_op = MirExtractFieldOp::new(op);
                    extract_op.set_attr_index(ctx, dialect_mir::attributes::FieldIndexAttr(1));

                    let result = extract_op.get_operation().deref(ctx).get_result(0);

                    Ok((
                        Some(extract_op.get_operation()),
                        result,
                        prev_op_after_operand,
                    ))
                }
                mir::UnOp::Not | mir::UnOp::Neg => {
                    let (operand_val, prev_op_after_operand) = translate_operand(
                        ctx,
                        body,
                        operand,
                        value_map,
                        block_ptr,
                        prev_op,
                        loc.clone(),
                    )?;
                    let result_type = operand_val.get_type(ctx);

                    let op_id = match un_op {
                        mir::UnOp::Not => MirNotOp::get_concrete_op_info(),
                        mir::UnOp::Neg => MirNegOp::get_concrete_op_info(),
                        _ => unreachable!(),
                    };

                    let op =
                        Operation::new(ctx, op_id, vec![result_type], vec![operand_val], vec![], 0);
                    op.deref_mut(ctx).set_loc(loc);

                    let result = op.deref(ctx).get_result(0);

                    Ok((Some(op), result, prev_op_after_operand))
                }
            }
        }
        mir::Rvalue::Cast(kind, operand, ty) => {
            // `let f: fn(u32) -> u32 = inc;` compiles to a ReifyFnPointer
            // cast. It is not a value conversion: the fn item `inc` is
            // zero-sized, so there is nothing to convert. What the program
            // needs is some address-like value identifying the function.
            // Real code addresses do not exist on the device (the function
            // may not even be compiled), so we make a stable stand-in: a
            // hash of the function's mangled name, cast int -> ptr. With
            // that, `f == f` is true and two different functions compare
            // unequal (Rust permits, but does not promise, distinct fn
            // addresses, so a hash stand-in is within contract). CALLING
            // through the pointer is still unsupported and fails loudly at
            // the call site. Handled before translate_operand because the
            // zero-sized fn-item operand itself never becomes a value.
            if let mir::CastKind::PointerCoercion(mir::PointerCoercion::ReifyFnPointer(_)) = kind {
                return translate_reify_fn_pointer(ctx, body, operand, ty, block_ptr, prev_op, loc);
            }
            if let mir::CastKind::PointerCoercion(mir::PointerCoercion::ClosureFnPointer(_)) = kind
            {
                return translate_closure_fn_pointer(
                    ctx, body, operand, ty, block_ptr, prev_op, loc,
                );
            }

            let (operand_val, prev_op_after_operand) = translate_operand(
                ctx,
                body,
                operand,
                value_map,
                block_ptr,
                prev_op,
                loc.clone(),
            )?;

            let result_type = types::translate_type(ctx, ty)?;

            let op = Operation::new(
                ctx,
                MirCastOp::get_concrete_op_info(),
                vec![result_type],
                vec![operand_val],
                vec![],
                0,
            );
            op.deref_mut(ctx).set_loc(loc);

            let cast_kind_attr = match kind {
                mir::CastKind::IntToInt => MirCastKindAttr::IntToInt,
                mir::CastKind::IntToFloat => MirCastKindAttr::IntToFloat,
                mir::CastKind::FloatToInt => MirCastKindAttr::FloatToInt,
                mir::CastKind::FloatToFloat => MirCastKindAttr::FloatToFloat,
                mir::CastKind::PtrToPtr => MirCastKindAttr::PtrToPtr,
                mir::CastKind::FnPtrToPtr => MirCastKindAttr::FnPtrToPtr,
                mir::CastKind::PointerExposeAddress => MirCastKindAttr::PointerExposeAddress,
                mir::CastKind::PointerWithExposedProvenance => {
                    MirCastKindAttr::PointerWithExposedProvenance
                }
                mir::CastKind::Transmute => MirCastKindAttr::Transmute,
                // Elaborated `box` derefs turn the inner pointer into a raw
                // pointer with this cast. Upstream documents it as "almost
                // equivalent to a regular transmute except that if the input
                // would not be valid as `Box<T>`, the cast is UB. Backends
                // that do not care about UB detection can treat this like a
                // regular transmute", and rustc_codegen_ssa lowers it in the
                // same match arm as `Transmute` (mir/rvalue.rs). We follow
                // codegen_ssa: a plain same-size bit reinterpretation.
                mir::CastKind::BoxDerefTransmute => MirCastKindAttr::Transmute,
                mir::CastKind::PointerCoercion(coercion) => match coercion {
                    mir::PointerCoercion::Unsize => MirCastKindAttr::PointerCoercionUnsize,
                    mir::PointerCoercion::MutToConstPointer => {
                        MirCastKindAttr::PointerCoercionMutToConst
                    }
                    mir::PointerCoercion::ArrayToPointer => {
                        MirCastKindAttr::PointerCoercionArrayToPointer
                    }
                    mir::PointerCoercion::ReifyFnPointer(_) => {
                        MirCastKindAttr::PointerCoercionReifyFnPointer
                    }
                    mir::PointerCoercion::UnsafeFnPointer => {
                        MirCastKindAttr::PointerCoercionUnsafeFnPointer
                    }
                    mir::PointerCoercion::ClosureFnPointer(_safety) => {
                        MirCastKindAttr::PointerCoercionClosureFnPointer
                    }
                },
                mir::CastKind::Subtype => MirCastKindAttr::Subtype,
            };
            let cast_op = MirCastOp::new(op);
            cast_op.set_attr_cast_kind(ctx, cast_kind_attr.clone());
            if dialect_mir::types::type_contains_concrete_pointer_kind(ctx, result_type)
                || (cast_kind_attr == MirCastKindAttr::Transmute
                    && !dialect_mir::types::pointer_kinds_in_type(ctx, result_type).is_empty())
            {
                // This operation comes directly from a rustc MIR CastKind, so
                // any concrete result category, including one nested in an
                // aggregate transmute, is an explicit Rust semantic boundary
                // rather than a generic representation adjustment.
                cast_op.set_pointer_kind_authority(ctx, MirPointerKindAuthorityAttr::RustCast);
            }

            let result = op.deref(ctx).get_result(0);

            Ok((Some(op), result, prev_op_after_operand))
        }
        mir::Rvalue::CheckedBinaryOp(bin_op, left, right) => {
            let checked_kind = classify_checked_binary_op(bin_op).map_err(|message| {
                input_error!(loc.clone(), TranslationErr::unsupported(message))
            })?;

            let (left_val, prev_op_after_left) =
                translate_operand(ctx, body, left, value_map, block_ptr, prev_op, loc.clone())?;
            let (right_val, prev_op_after_right) = translate_operand(
                ctx,
                body,
                right,
                value_map,
                block_ptr,
                prev_op_after_left,
                loc.clone(),
            )?;

            // The result type is the MIR-level `(T, bool)` tuple.
            // Translate it from the rvalue's rustc type so it is the
            // same uniqued, layout-carrying tuple type the rest of
            // the body (locals, places) uses.
            let rust_tuple_ty = rvalue.ty(body.locals()).map_err(|e| {
                input_error_noloc!(TranslationErr::unsupported(format!(
                    "Failed to query checked-arithmetic result type: {:?}",
                    e
                )))
            })?;
            let result_type = types::translate_type(ctx, &rust_tuple_ty)?;

            let op_id = match checked_kind {
                CheckedBinaryOpKind::Add => MirCheckedAddOp::get_concrete_op_info(),
                CheckedBinaryOpKind::Sub => MirCheckedSubOp::get_concrete_op_info(),
                CheckedBinaryOpKind::Mul => MirCheckedMulOp::get_concrete_op_info(),
            };
            let op = Operation::new(
                ctx,
                op_id,
                vec![result_type],
                vec![left_val, right_val],
                vec![],
                0,
            );
            op.deref_mut(ctx).set_loc(loc);

            let result = op.deref(ctx).get_result(0);
            Ok((Some(op), result, prev_op_after_right))
        }
        mir::Rvalue::Use(operand, _) => {
            // Use just copies/moves a value - no operation needed, just pass through
            // The operand translation may insert field extraction operations
            let (val, last_inserted) =
                translate_operand(ctx, body, operand, value_map, block_ptr, prev_op, loc)?;

            // Return None for operation - Use doesn't create an operation
            // Any field extractions are already inserted and tracked in last_inserted
            Ok((None, val, last_inserted))
        }
        mir::Rvalue::Ref(_region, borrow_kind, place) => {
            // Ref creates a reference to a place: &place or &mut place.
            //
            // Strategy:
            //
            // 1. `&local` / `&mut local` -- return the local's alloca slot
            //    pointer directly (ZST locals get a synthesised pointer).
            // 2. Any projected place -- compute the real in-memory address
            //    by walking the FULL projection list from the base local's
            //    slot via `translate_place_address`: `&(*ptr)` loads the
            //    pointer, `&(*ptr).field` adds a `mir.field_addr`,
            //    `&x.arr[i]` adds a `mir.array_element_addr`, and arbitrary
            //    combinations compose. Borrows produced this way ALIAS the
            //    original storage, which is what Rust requires: e.g.
            //    `Enumerate::next` takes `&mut (*_1).0` and `Iter::next`
            //    must advance the ORIGINAL Iter in place -- a `mir.ref` of
            //    an extracted field VALUE would mutate a copy and loop
            //    forever.
            // 3. Only when no address can be computed (slot-less computed
            //    value, or a projection the walker cannot lower, e.g.
            //    Downcast) do we fall back to materialising the VALUE and
            //    wrapping it in `mir.ref` (fresh slot + store of a COPY).
            //    That is sound for shared borrows (reads through a copy)
            //    and a silent miscompile for mutable ones (writes land in
            //    the copy), so mutable borrows hard-error instead.

            // Case 1: bare local reference `&local` / `&mut local`.
            //
            // Alloca + load/store model: every non-ZST MIR local is backed by
            // a stack slot emitted at the top of the entry block. Taking the
            // address of the local therefore just returns that slot pointer --
            // no extra allocation is needed. `mem2reg` folds this back into
            // SSA when the borrow doesn't escape.
            //
            // Slots are always allocated mutable because the importer writes
            // locals through them. They are compiler storage, not `&mut T`.
            // The normalization below retypes the physical slot address to the
            // exact Rust reference kind without treating slot mutability as
            // evidence of uniqueness.
            let origin = facts::pointer_origin_of_borrow(*borrow_kind);
            let is_mutable = origin.is_mutable();
            if place.projection.is_empty() {
                if let Some(slot) = value_map.get_slot(place.local) {
                    // The slot is compiler storage (`Erased`) and is always
                    // mutable. Taking a Rust borrow is the semantic boundary
                    // where that physical address acquires the exact `&T` /
                    // `&mut T` type recorded by rustc.
                    let rust_result_type = rvalue.ty(body.locals()).map_err(|error| {
                        input_error_noloc!(TranslationErr::unsupported(format!(
                            "failed to determine reference rvalue type: {error:?}"
                        )))
                    })?;
                    let expected_ptr_type = types::translate_type(ctx, &rust_result_type)?;
                    let (result, last_inserted) = cast_to_declared_rust_pointer_type_if_needed(
                        ctx,
                        slot,
                        expected_ptr_type,
                        block_ptr,
                        prev_op,
                        loc.clone(),
                        MirPointerKindAuthorityAttr::Reborrow,
                    );
                    return Ok((None, result, last_inserted));
                }
                // ZST local (no slot). Synthesise a pointer-to-ZST via
                // MirRefOp as a fallback so callers still get a well-typed
                // pointer value.
                let local_decl = &body.locals()[place.local];
                let ty_ptr = crate::translator::types::translate_type(ctx, &local_decl.ty)?;
                let (zst_val, last_inserted) =
                    if ty_ptr.deref(ctx).is::<dialect_mir::types::MirEnumType>() {
                        let op = create_ghost_enum_default(ctx, ty_ptr, loc.clone());
                        match prev_op {
                            Some(p) => op.insert_after(ctx, p),
                            None => op.insert_at_front(block_ptr, ctx),
                        }
                        (op.deref(ctx).get_result(0), Some(op))
                    } else {
                        translate_zero_sized_constant_value(
                            ctx,
                            ty_ptr,
                            block_ptr,
                            prev_op,
                            loc.clone(),
                        )?
                    };
                let ptr_ty = facts::mint_generic_ptr_type(ctx, ty_ptr, origin);
                let ref_op = Operation::new(
                    ctx,
                    MirRefOp::get_concrete_op_info(),
                    vec![ptr_ty.into()],
                    vec![zst_val],
                    vec![],
                    0,
                );
                ref_op.deref_mut(ctx).set_loc(loc);
                let mir_ref = MirRefOp::new(ref_op);
                mir_ref.set_mutable(ctx, is_mutable);
                mir_ref.set_pointer_kind_authority(ctx, MirPointerKindAuthorityAttr::Reborrow);
                match last_inserted {
                    Some(p) => ref_op.insert_after(ctx, p),
                    None => ref_op.insert_at_front(block_ptr, ctx),
                }
                let result_val = ref_op.deref(ctx).get_result(0);
                return Ok((None, result_val, Some(ref_op)));
            }

            // Case 2: unified address path -- walk the full projection list
            // (`Deref`, `Field`, `Index`, `ConstantIndex`) from the base
            // local's alloca slot. This is the "correct-refs" path: the
            // resulting pointer addresses the ORIGINAL storage, so writes
            // through the borrow mutate the borrowed place.
            if let Some((result_val, last_inserted)) = translate_place_address(
                ctx,
                body,
                value_map,
                place,
                is_mutable,
                block_ptr,
                prev_op,
                loc.clone(),
            )? {
                // Address arithmetic remains in the physical address space, but the
                // resulting Rust reference must have its exact translated Rust type.
                let rust_result_type = rvalue.ty(body.locals()).map_err(|error| {
                    input_error_noloc!(TranslationErr::unsupported(format!(
                        "failed to determine reference rvalue type: {error:?}"
                    )))
                })?;
                let expected_ptr_type = types::translate_type(ctx, &rust_result_type)?;

                let (result_val, last_inserted) = cast_to_declared_rust_pointer_type_if_needed(
                    ctx,
                    result_val,
                    expected_ptr_type,
                    block_ptr,
                    last_inserted,
                    loc.clone(),
                    MirPointerKindAuthorityAttr::Reborrow,
                );

                return Ok((None, result_val, last_inserted));
            }

            // No address could be computed. The only remaining strategy is
            // the value-copy fallback below, which is a silent miscompile
            // for mutable borrows: writes through the borrow would land in
            // the copy and the original place would never change. Refuse
            // loudly instead of emitting wrong code.
            if is_mutable {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Rvalue::Ref: cannot compute an in-memory address for the mutable \
                         borrow of place {:?} (projection {:?}); the value-copy fallback \
                         would silently discard writes through the borrow",
                        place, place.projection
                    ))
                );
            }

            // Case 3: shared-borrow fallback -- reference to a computed
            // value that has no backing slot (e.g. the result of an rvalue
            // expression) or whose projection the address walker cannot
            // lower (e.g. enum Downcast, issues #131/#146). Emit `mir.ref`
            // which allocates a fresh slot, stores a COPY of the value, and
            // returns the pointer. Sound for shared borrows only (reads);
            // mutable borrows were rejected above.
            let (val, last_inserted) =
                translate_place(ctx, body, place, value_map, block_ptr, prev_op, loc.clone())?;

            let val_ty = val.get_type(ctx);
            let ptr_ty = facts::mint_generic_ptr_type(ctx, val_ty, origin);

            let ref_op = Operation::new(
                ctx,
                MirRefOp::get_concrete_op_info(),
                vec![ptr_ty.into()],
                vec![val],
                vec![],
                0,
            );
            ref_op.deref_mut(ctx).set_loc(loc);
            let mir_ref = MirRefOp::new(ref_op);
            mir_ref.set_mutable(ctx, is_mutable);
            mir_ref.set_pointer_kind_authority(ctx, MirPointerKindAuthorityAttr::Reborrow);

            let result_val = ref_op.deref(ctx).get_result(0);
            Ok((Some(ref_op), result_val, last_inserted))
        }
        mir::Rvalue::AddressOf(mutability, place) => {
            // AddressOf creates a raw pointer to a place: `&raw const place`
            // / `&raw mut place` (also `core::ptr::addr_of!`). Raw pointers
            // have the same aliasing requirement as references: the pointer
            // must address the ORIGINAL place, so this routes through the
            // same unified address walker as `Rvalue::Ref` (which also gives
            // raw pointers the runtime-Index / ConstantIndex handling).
            let origin = facts::pointer_origin_of_raw_ptr(*mutability);
            let is_mutable = origin.is_mutable();

            // Bare local: the alloca slot is the physical address, but the
            // result must carry the exact raw-pointer kind rather than the
            // slot's compiler-only `Erased` provenance.
            if place.projection.is_empty()
                && let Some(slot) = value_map.get_slot(place.local)
            {
                let rust_result_type = rvalue.ty(body.locals()).map_err(|error| {
                    input_error_noloc!(TranslationErr::unsupported(format!(
                        "failed to determine address-of rvalue type: {error:?}"
                    )))
                })?;
                let expected_ptr_type = types::translate_type(ctx, &rust_result_type)?;
                let (result, last_inserted) = cast_to_declared_rust_pointer_type_if_needed(
                    ctx,
                    slot,
                    expected_ptr_type,
                    block_ptr,
                    prev_op,
                    loc.clone(),
                    MirPointerKindAuthorityAttr::RawAddress,
                );
                return Ok((None, result, last_inserted));
            }

            // Unified address path: full projection walk from the slot
            // (`&raw (*ptr)` loads the pointer, `&raw (*ptr).field[i]`
            // composes field + element addresses, ...).
            if let Some((result_val, last_inserted)) = translate_place_address(
                ctx,
                body,
                value_map,
                place,
                is_mutable,
                block_ptr,
                prev_op,
                loc.clone(),
            )? {
                // Preserve physical address spaces while walking the place, then restore
                // the exact raw-pointer type required by the Rust rvalue.
                let rust_result_type = rvalue.ty(body.locals()).map_err(|error| {
                    input_error_noloc!(TranslationErr::unsupported(format!(
                        "failed to determine address-of rvalue type: {error:?}"
                    )))
                })?;
                let expected_ptr_type = types::translate_type(ctx, &rust_result_type)?;

                let (result_val, last_inserted) = cast_to_declared_rust_pointer_type_if_needed(
                    ctx,
                    result_val,
                    expected_ptr_type,
                    block_ptr,
                    last_inserted,
                    loc.clone(),
                    MirPointerKindAuthorityAttr::RawAddress,
                );

                return Ok((None, result_val, last_inserted));
            }

            // No address could be computed. The value-copy fallback below
            // returns a pointer to a COPY, so writes through a `&raw mut`
            // would be silently lost -- refuse loudly. Exception: a bare
            // slot-less local is a ZST (no bytes), so a copy cannot lose
            // writes; let it use the fallback for both mutabilities, the
            // same way `Rvalue::Ref` synthesises ZST borrows.
            if is_mutable && !place.projection.is_empty() {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Rvalue::AddressOf: cannot compute an in-memory address for \
                         `&raw mut` of place {:?} (projection {:?}); the value-copy \
                         fallback would silently discard writes through the pointer",
                        place, place.projection
                    ))
                );
            }

            // Shared (or bare-ZST) fallback: translate to a value and
            // materialize an address of a copy.
            let (val, last_inserted) =
                translate_place(ctx, body, place, value_map, block_ptr, prev_op, loc.clone())?;

            let val_ty = val.get_type(ctx);
            let ptr_ty = facts::mint_generic_ptr_type(ctx, val_ty, origin);

            use dialect_mir::ops::MirRefOp;
            let ref_op = Operation::new(
                ctx,
                MirRefOp::get_concrete_op_info(),
                vec![ptr_ty.into()],
                vec![val],
                vec![],
                0,
            );
            ref_op.deref_mut(ctx).set_loc(loc);

            let mir_ref_op = MirRefOp::new(ref_op);
            mir_ref_op.set_mutable(ctx, is_mutable);
            mir_ref_op.set_pointer_kind_authority(ctx, MirPointerKindAuthorityAttr::RawAddress);

            let result_val = ref_op.deref(ctx).get_result(0);

            Ok((Some(ref_op), result_val, last_inserted))
        }
        mir::Rvalue::Aggregate(aggregate_kind, operands) => translate_aggregate_rvalue(
            ctx,
            body,
            rvalue,
            aggregate_kind,
            operands,
            value_map,
            block_ptr,
            prev_op,
            loc,
        ),
        mir::Rvalue::Discriminant(place) => {
            // Get the discriminant (tag) from an enum value.
            //
            // Two discriminant types can be in play:
            //   - `native_tag_ty`: the logical result type produced by our
            //     enum operation. For Direct layouts this is the physical tag
            //     type; for Niche/Single layouts the operation decodes or
            //     materializes the logical discriminant directly.
            //   - `mir_discr_ty`: the type stable-MIR declares for the
            //     `Rvalue::Discriminant(place)` value itself, via
            //     `Ty::kind().discriminant_ty()`. This is what rustc uses
            //     to type the destination local (often `i64`).
            //
            // When the two types disagree (normally a narrow Direct tag versus
            // stable MIR's wider declared type), widen via `mir.cast IntToInt`
            // so the rvalue matches what stable MIR promised. Without this,
            // storing the result into its destination slot would fail
            // verification.
            use dialect_mir::ops::MirGetDiscriminantOp;
            use dialect_mir::types::MirEnumType;
            use pliron::builtin::types::IntegerType;
            use pliron::printable::Printable;

            let (enum_val, prev_op_after) =
                translate_place(ctx, body, place, value_map, block_ptr, prev_op, loc.clone())?;

            let enum_ty = enum_val.get_type(ctx);
            let (native_tag_ty, enum_is_uninhabited) = {
                let enum_ty_obj = enum_ty.deref(ctx);
                if let Some(enum_type) = enum_ty_obj.downcast_ref::<MirEnumType>() {
                    let uninhabited = !enum_type
                        .variant_inhabited
                        .iter()
                        .any(|inhabited| *inhabited != 0);
                    (enum_type.discriminant_type(), uninhabited)
                } else {
                    return input_err!(
                        loc,
                        TranslationErr::unsupported(format!(
                            "Discriminant on non-enum type: {}",
                            enum_ty.disp(ctx)
                        ))
                    );
                }
            };

            // No value of an uninhabited enum can exist, so this read sits
            // on a dynamically dead path rustc keeps in MIR (e.g. matching
            // the residual `ControlFlow<Infallible, NeverShortCircuitResidual>`
            // inside `array::try_from_fn`). `mir.get_discriminant` refuses
            // uninhabited enums by verification, so keep the dead path
            // representable with a typed undef of the declared discriminant
            // type instead.
            if enum_is_uninhabited {
                let declared_discr_ty = place
                    .ty(body.locals())
                    .ok()
                    .and_then(|place_ty| place_ty.kind().discriminant_ty());
                let undef_ty = match declared_discr_ty {
                    Some(ty) => crate::translator::types::translate_type(ctx, &ty)?,
                    None => native_tag_ty,
                };
                let undef = MirUndefOp::new(ctx, undef_ty).get_operation();
                undef.deref_mut(ctx).set_loc(loc);
                let result = undef.deref(ctx).get_result(0);
                return Ok((Some(undef), result, prev_op_after));
            }

            let get_disc_op = Operation::new(
                ctx,
                MirGetDiscriminantOp::get_concrete_op_info(),
                vec![native_tag_ty],
                vec![enum_val],
                vec![],
                0,
            );
            get_disc_op.deref_mut(ctx).set_loc(loc.clone());
            let native_result = get_disc_op.deref(ctx).get_result(0);

            // Ask stable-MIR what the declared discriminant type of this
            // place is. For well-formed MIR on an enum place this should
            // always succeed; if we can't compute it, fall back to the
            // native tag (no cast). In the fallback path we preserve the
            // original contract: the caller inserts `get_disc_op`.
            let place_ty = place.ty(body.locals()).map_err(|e| {
                input_error!(
                    loc.clone(),
                    TranslationErr::unsupported(format!(
                        "Failed to resolve place type for Discriminant: {:?}",
                        e
                    ))
                )
            })?;
            let declared_discr_ty = place_ty.kind().discriminant_ty();

            let mir_discr_ty = match declared_discr_ty {
                Some(ty) => crate::translator::types::translate_type(ctx, &ty)?,
                None => {
                    return Ok((Some(get_disc_op), native_result, prev_op_after));
                }
            };

            // Only widen when both sides are integers and differ. Anything
            // else is either already matched or a dialect-level mismatch
            // that deserves its own verifier error upstream.
            let needs_cast = mir_discr_ty != native_tag_ty && {
                let src = native_tag_ty.deref(ctx);
                let dst = mir_discr_ty.deref(ctx);
                src.is::<IntegerType>() && dst.is::<IntegerType>()
            };

            if !needs_cast {
                return Ok((Some(get_disc_op), native_result, prev_op_after));
            }

            // Widening path: we emit two ops (get_discriminant + cast) and
            // the caller only inserts a single "main" op. Insert both here
            // and return `None` as the main op so the caller does not try
            // to re-insert.
            if let Some(prev) = prev_op_after {
                get_disc_op.insert_after(ctx, prev);
            } else {
                get_disc_op.insert_at_front(block_ptr, ctx);
            }

            let cast_op = Operation::new(
                ctx,
                MirCastOp::get_concrete_op_info(),
                vec![mir_discr_ty],
                vec![native_result],
                vec![],
                0,
            );
            cast_op.deref_mut(ctx).set_loc(loc);
            MirCastOp::new(cast_op).set_attr_cast_kind(ctx, MirCastKindAttr::IntToInt);
            cast_op.insert_after(ctx, get_disc_op);

            let result = cast_op.deref(ctx).get_result(0);
            Ok((None, result, Some(cast_op)))
        }
        mir::Rvalue::Repeat(operand, count) => {
            // Array repeat: [value; N] -> mir.construct_array with N copies of value
            //
            // MIR: _1 = Repeat(Constant(0.0f32), 16)
            // Produces: [0.0, 0.0, 0.0, ...] (16 elements)

            // Extract the count from TyConst
            let array_size = count.eval_target_usize().map_err(|e| {
                input_error!(
                    loc.clone(),
                    TranslationErr::unsupported(format!(
                        "Failed to evaluate Repeat count: {:?}",
                        e
                    ))
                )
            })?;

            // Translate the operand to get the element value
            let (element_val, prev_op_after_operand) = translate_operand(
                ctx,
                body,
                operand,
                value_map,
                block_ptr,
                prev_op,
                loc.clone(),
            )?;

            // Get the element type from the value
            let element_type = element_val.get_type(ctx);

            // Create element values by repeating the single value
            let element_values: Vec<Value> =
                std::iter::repeat_n(element_val, array_size as usize).collect();

            // Create the array type
            let array_ty = dialect_mir::types::MirArrayType::get(ctx, element_type, array_size);

            // Create mir.construct_array operation
            let op = Operation::new(
                ctx,
                MirConstructArrayOp::get_concrete_op_info(),
                vec![array_ty.into()],
                element_values,
                vec![],
                0,
            );
            op.deref_mut(ctx).set_loc(loc);

            let result = op.deref(ctx).get_result(0);

            Ok((Some(op), result, prev_op_after_operand))
        }
        mir::Rvalue::CopyForDeref(place) => {
            // CopyForDeref is semantically a place read. Any load or projection
            // operations emitted by translate_place are already inserted and tracked.
            //
            // Note: on the pinned toolchain the MIR optimization pipeline
            // (copy-prop/GVN) eliminates every CopyForDeref before the
            // optimized MIR reaches this importer; nested-deref kernels
            // (`**rr`, including through raw pointers) were probed and none
            // produce one here today. This arm is defensive coverage so a
            // future toolchain bump or lower mir-opt-level cannot regress
            // valid nested-deref kernels into an "unsupported construct"
            // failure.
            let (value, last_inserted) =
                translate_place(ctx, body, place, value_map, block_ptr, prev_op, loc)?;

            Ok((None, value, last_inserted))
        }
        mir::Rvalue::Reborrow(target_ty, _mutability, place) => {
            // `Rvalue::Reborrow` (rust-lang/rust#159103, `feature(reborrow)`,
            // nightly-2026-08-28+): reborrowing a user ADT that implements
            // the `Reborrow` marker trait. Semantically this is a bitwise
            // copy of `place`:
            //
            // - `Mutability::Mut` (Reborrow): the target type IS the source
            //   type, so this is exactly a place read, like `Rvalue::Use` of
            //   `Operand::Copy(place)`.
            // - `Mutability::Not` (CoerceShared): the target is the
            //   `CoerceShared` target ADT, which the trait's coherence rules
            //   force to have the identical memory layout to the source
            //   (rustc_middle mir/syntax.rs doc on `Rvalue::Reborrow`).
            //
            // rustc_codegen_ssa lowers both cases as
            // `codegen_operand(bx, &Operand::Copy(place))` (mir/rvalue.rs:
            // "Exclusive reborrowing is always equal to a memcpy ... the
            // coherence check places such restrictions on the CoerceShared
            // trait as to guarantee that it is [too]"); the const
            // interpreter uses `copy_op` / `copy_op_allow_transmute`
            // (interpret/step.rs). We mirror that: read the place, and when
            // the translated dialect types diverge (a CoerceShared reborrow
            // into a distinct same-layout ADT), reuse the same
            // `MirCastKindAttr::Transmute` path a `CastKind::Transmute`
            // would take.
            //
            // The optimizer (GVN/copy-prop) usually folds these into plain
            // copies at `-C opt-level>0`, but they reach the importer intact
            // at `-Zmir-opt-level=0`, which the reborrow example scopes to its
            // own package via profile-rustflags (verified on the pinned
            // nightly).
            let (value, last_inserted) =
                translate_place(ctx, body, place, value_map, block_ptr, prev_op, loc.clone())?;

            let target_type = types::translate_type(ctx, target_ty)?;
            if value.get_type(ctx) == target_type {
                return Ok((None, value, last_inserted));
            }

            let op = Operation::new(
                ctx,
                MirCastOp::get_concrete_op_info(),
                vec![target_type],
                vec![value],
                vec![],
                0,
            );
            op.deref_mut(ctx).set_loc(loc);

            let cast_op = MirCastOp::new(op);
            cast_op.set_attr_cast_kind(ctx, MirCastKindAttr::Transmute);
            // Same authority rule as the `Rvalue::Cast` transmute path: this
            // reinterpretation comes directly from rustc MIR, so any concrete
            // pointer category it carries is an explicit Rust semantic
            // boundary.
            if dialect_mir::types::type_contains_concrete_pointer_kind(ctx, target_type)
                || !dialect_mir::types::pointer_kinds_in_type(ctx, target_type).is_empty()
            {
                cast_op.set_pointer_kind_authority(ctx, MirPointerKindAuthorityAttr::RustCast);
            }

            let result = op.deref(ctx).get_result(0);
            Ok((Some(op), result, last_inserted))
        }
        mir::Rvalue::Len(place) => input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Rvalue::Len for place {:?} not yet implemented",
                place
            ))
        ),
        mir::Rvalue::ThreadLocalRef(item) => input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Rvalue::ThreadLocalRef {:?} is not supported in device code",
                item
            ))
        ),
    }
}

#[cfg(test)]
mod checked_binary_op_tests {
    use super::*;

    #[test]
    fn checked_binary_op_accepts_supported_operators() {
        assert_eq!(
            classify_checked_binary_op(&mir::BinOp::Add),
            Ok(CheckedBinaryOpKind::Add)
        );
        assert_eq!(
            classify_checked_binary_op(&mir::BinOp::Sub),
            Ok(CheckedBinaryOpKind::Sub)
        );
        assert_eq!(
            classify_checked_binary_op(&mir::BinOp::Mul),
            Ok(CheckedBinaryOpKind::Mul)
        );
    }

    #[test]
    fn checked_binary_op_rejects_unsupported_operator() {
        let error = classify_checked_binary_op(&mir::BinOp::Div)
            .expect_err("Div must remain unsupported as CheckedBinaryOp");

        assert_eq!(error, "CheckedBinaryOp Div not yet implemented");
    }
}
