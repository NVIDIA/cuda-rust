/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Statement translation: MIR statements → `dialect-mir` operations.
//!
//! Handles MIR statements like assignments, storage markers, and projections.
//!
//! # Supported Statements
//!
//! | Statement Kind      | Translation                                          |
//! |---------------------|------------------------------------------------------|
//! | `Assign(_l, rv)`    | Rvalue → ops; result stored into `_l`'s alloca slot  |
//! | `*ptr = val`        | `mir.store`                                          |
//! | `s.field = val`     | `mir.field_addr` + `mir.store` through the slot      |
//! | `SetDiscriminant`   | `mir.set_discriminant` (enum tag write)              |
//! | `StorageLive`       | `mir.storage_live` (lifetime marker)                 |
//! | `StorageDead`       | `mir.storage_dead` (lifetime marker)                 |
//! | `Nop`               | Skipped                                              |
//!
//! # Projections
//!
//! 1- and 2-level projections have dedicated arms:
//! - `*ptr` → Store through pointer
//! - `s.field` → Field-address from the slot, then `mir.store`
//! - `(*ptr).field` → Load pointer, compute field address, store
//! - `s.outer.inner` → Chained field-address from the slot, then store
//! - `(*ptr)[i]` → Element address from the unified place-address walker
//!   (handles both `&mut [T; N]` and fat `&mut [T]` bases), then store
//!
//! Deeper chains (e.g. `(*iter).alive.start` from the `for x in arr`
//! loop machinery) are handled generically: the full projection list is
//! walked to a destination address with the same place-address walker
//! that `Rvalue::Ref` uses, then a single `mir.store` writes through it.

use super::facts;
use super::types;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::location::span_to_location;
use crate::translator::payload_store;
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_mir::ops::{
    MirMemcpyOp, MirSetDiscriminantOp, MirStorageDeadOp, MirStorageLiveOp, MirStoreOp,
};
use dialect_mir::types::MirEnumType;
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::printable::Printable;
use pliron::r#type::Typed;
use pliron::utils::apint::APInt;
use pliron::value::Value;
use pliron::{input_err, input_error};
use rustc_public::mir;
use rustc_public_bridge::IndexedVal;
use std::num::NonZeroUsize;

/// Translates a MIR statement to one or more `dialect-mir` operations.
///
/// # Returns
///
/// The last inserted operation (for chaining), or `prev_op` if no ops were created.
/// For `Rvalue::Use`, no operation is created - just updates `value_map`.
pub fn translate_statement(
    ctx: &mut Context,
    body: &mir::Body,
    stmt: &mir::Statement,
    value_map: &mut ValueMap,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
) -> TranslationResult<Option<Ptr<Operation>>> {
    let loc = span_to_location(ctx, stmt.source_info.span);

    match &stmt.kind {
        mir::StatementKind::Assign(place, rvalue) => {
            // Fast paths: array initializers assigned to an addressable local.
            // Store each element directly into the alloca storage instead of
            // building an SSA aggregate (insertvalue chain) and then storing it.
            if value_map.get_slot(place.local).is_some() {
                match rvalue {
                    mir::Rvalue::Aggregate(mir::AggregateKind::Array(_), operands) => {
                        return translate_array_agg_into_alloca(
                            ctx, body, place, operands, value_map, block_ptr, prev_op, loc,
                        );
                    }
                    // A fully-constant array: copy it in from an immutable
                    // device global rather than storing it element by element in
                    // every thread. Falls through when the constant is not a
                    // shape that can be reduced to a byte image.
                    mir::Rvalue::Use(mir::Operand::Constant(constant), _) => {
                        if let Some(last) = rvalue::translate_array_constant_into_alloca(
                            ctx,
                            body,
                            place,
                            constant,
                            value_map,
                            block_ptr,
                            prev_op,
                            loc.clone(),
                        )? {
                            return Ok(Some(last));
                        }
                    }
                    mir::Rvalue::Repeat(operand, count) => {
                        let n = count.eval_target_usize().map_err(|e| {
                            input_error!(
                                loc.clone(),
                                TranslationErr::unsupported(format!(
                                    "Failed to evaluate Repeat count: {:?}",
                                    e
                                ))
                            )
                        })? as usize;
                        return translate_repeat_into_alloca(
                            ctx, body, place, operand, n, value_map, block_ptr, prev_op, loc,
                        );
                    }
                    _ => {}
                }
            }

            // Translate the Rvalue to get the value being assigned
            let (rvalue_op_opt, result_value, last_inserted) = rvalue::translate_rvalue(
                ctx,
                body,
                rvalue,
                value_map,
                block_ptr,
                prev_op,
                loc.clone(),
            )?;

            // Map the result to the place (local variable)
            if place.projection.is_empty() {
                // Simple local assignment: write the rvalue into the local's
                // stack slot (`mir.store local_slot, value`). ZST locals
                // (no slot) are silently skipped -- nothing to materialise.
                let local = place.local;

                // Insert the rvalue operation if it's not None
                // For Rvalue::Use, rvalue_op_opt is None (no operation to insert)
                // For other Rvalues (like CheckedAdd), we need to insert the operation
                let current_prev = if let Some(rvalue_op) = rvalue_op_opt {
                    if let Some(prev) = last_inserted {
                        rvalue_op.insert_after(ctx, prev);
                    } else if let Some(prev) = prev_op {
                        rvalue_op.insert_after(ctx, prev);
                    } else {
                        rvalue_op.insert_at_front(block_ptr, ctx);
                    }
                    Some(rvalue_op)
                } else {
                    // For Rvalue::Use, return the last inserted operation (field extraction if any)
                    // If last_inserted is None, we return prev_op
                    last_inserted.or(prev_op)
                };

                let store_op =
                    value_map.store_local(ctx, local, result_value, block_ptr, current_prev);
                Ok(store_op.or(current_prev))
            } else if place.projection.len() == 1 {
                match &place.projection[0] {
                    mir::ProjectionElem::Deref => {
                        // *ptr = value (Store)
                        // Translate the pointer (base)
                        let base_place = mir::Place {
                            local: place.local,
                            projection: vec![],
                        };

                        // Determine current_prev based on rvalue insertion
                        let mut current_prev = prev_op;
                        if let Some(rvalue_op) = rvalue_op_opt {
                            if let Some(prev) = last_inserted {
                                rvalue_op.insert_after(ctx, prev);
                                current_prev = Some(rvalue_op);
                            } else if let Some(prev) = prev_op {
                                rvalue_op.insert_after(ctx, prev);
                                current_prev = Some(rvalue_op);
                            } else {
                                rvalue_op.insert_at_front(block_ptr, ctx);
                                current_prev = Some(rvalue_op);
                            }
                        } else if let Some(prev) = last_inserted {
                            current_prev = Some(prev);
                        }

                        // Re-translate place with updated prev_op to ensure ordering
                        let (ptr_val, prev_op_after_ptr) = rvalue::translate_place(
                            ctx,
                            body,
                            &base_place,
                            value_map,
                            block_ptr,
                            current_prev,
                            loc.clone(),
                        )?;

                        // Create Store Op
                        let store_op = Operation::new(
                            ctx,
                            MirStoreOp::get_concrete_op_info(),
                            vec![],                      // No results
                            vec![ptr_val, result_value], // ptr, value
                            vec![],
                            0,
                        );
                        store_op.deref_mut(ctx).set_loc(loc);

                        if let Some(prev) = prev_op_after_ptr {
                            store_op.insert_after(ctx, prev);
                        } else {
                            // This implies block was empty and both rvalue and place didn't insert ops?
                            // Or they inserted at front.
                            store_op.insert_at_front(block_ptr, ctx);
                        }

                        Ok(Some(store_op))
                    }
                    mir::ProjectionElem::Field(field_idx, field_ty) => {
                        // struct.field = value (field assignment)
                        //
                        // Alloca model: compute the field's address from the
                        // local's slot via [`MirFieldAddrOp`] and store
                        // directly. This keeps the write addressable by
                        // `mem2reg` and avoids rebuilding the whole aggregate
                        // on every field update.

                        let mut current_prev = prev_op;
                        if let Some(rvalue_op) = rvalue_op_opt {
                            if let Some(prev) = last_inserted {
                                rvalue_op.insert_after(ctx, prev);
                                current_prev = Some(rvalue_op);
                            } else if let Some(prev) = prev_op {
                                rvalue_op.insert_after(ctx, prev);
                                current_prev = Some(rvalue_op);
                            } else {
                                rvalue_op.insert_at_front(block_ptr, ctx);
                                current_prev = Some(rvalue_op);
                            }
                        } else if let Some(prev) = last_inserted {
                            current_prev = Some(prev);
                        }

                        let local = place.local;
                        let Some(slot) = value_map.get_slot(local) else {
                            return input_err!(
                                loc,
                                TranslationErr::unsupported(format!(
                                    "Local {:?} has no alloca slot for field assignment",
                                    local
                                ))
                            );
                        };
                        reject_raw_field_index_on_enum_pointee(ctx, slot, &place.projection, &loc)?;

                        let field_type = types::translate_type(ctx, field_ty)?;
                        let slot_mutable = pointer_is_mutable(ctx, slot);
                        let field_ptr_ty = dialect_mir::types::MirPtrType::get(
                            ctx,
                            field_type,
                            slot_mutable,
                            pointer_address_space(ctx, slot),
                        )
                        .into();

                        use dialect_mir::ops::MirFieldAddrOp;
                        let field_addr_op =
                            MirFieldAddrOp::build(ctx, slot, field_ptr_ty, *field_idx as u32)?;
                        field_addr_op.deref_mut(ctx).set_loc(loc.clone());
                        if let Some(prev) = current_prev {
                            field_addr_op.insert_after(ctx, prev);
                        } else {
                            field_addr_op.insert_at_front(block_ptr, ctx);
                        }
                        let field_ptr = field_addr_op.deref(ctx).get_result(0);

                        let store_op = Operation::new(
                            ctx,
                            MirStoreOp::get_concrete_op_info(),
                            vec![],
                            vec![field_ptr, result_value],
                            vec![],
                            0,
                        );
                        store_op.deref_mut(ctx).set_loc(loc);
                        store_op.insert_after(ctx, field_addr_op);
                        Ok(Some(store_op))
                    }
                    mir::ProjectionElem::ConstantIndex {
                        offset,
                        min_length: _,
                        from_end,
                    } => {
                        // arr[const_idx] = value.
                        //
                        // Alloca model: locate the element via
                        // `MirConstantOp` + `MirArrayElementAddrOp` from the
                        // local's slot and emit `mir.store`.

                        if *from_end {
                            return input_err!(
                                loc,
                                TranslationErr::unsupported(
                                    "ConstantIndex with from_end=true not yet supported for writes"
                                )
                            );
                        }

                        let mut current_prev = prev_op;
                        if let Some(rvalue_op) = rvalue_op_opt {
                            if let Some(prev) = last_inserted {
                                rvalue_op.insert_after(ctx, prev);
                                current_prev = Some(rvalue_op);
                            } else if let Some(prev) = prev_op {
                                rvalue_op.insert_after(ctx, prev);
                                current_prev = Some(rvalue_op);
                            } else {
                                rvalue_op.insert_at_front(block_ptr, ctx);
                                current_prev = Some(rvalue_op);
                            }
                        } else if let Some(prev) = last_inserted {
                            current_prev = Some(prev);
                        }

                        let local = place.local;
                        let index = *offset as usize;
                        let Some(arr_ptr) = value_map.get_slot(local) else {
                            return input_err!(
                                loc,
                                TranslationErr::unsupported(format!(
                                    "Local {:?} has no alloca slot for array element assignment",
                                    local
                                ))
                            );
                        };

                        let (element_ty, address_space) =
                            slot_array_element_ty(ctx, arr_ptr, &loc)?;

                        use dialect_mir::ops::MirConstantOp;
                        use pliron::builtin::attributes::IntegerAttr;

                        let i64_ty = IntegerType::get(ctx, 64, Signedness::Signed);
                        let index_apint =
                            APInt::from_i64(index as i64, NonZeroUsize::new(64).unwrap());
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

                        let store_op = emit_array_element_store(
                            ctx,
                            arr_ptr,
                            index_value,
                            result_value,
                            element_ty,
                            address_space,
                            block_ptr,
                            current_prev,
                            loc,
                        );
                        Ok(Some(store_op))
                    }
                    mir::ProjectionElem::Index(index_local) => {
                        // arr[i] = value with runtime index.
                        //
                        // Alloca model: fetch the index (via `load_local`
                        // through translate_place), GEP from the array's
                        // slot, and `mir.store` the value.

                        let mut current_prev = prev_op;
                        if let Some(rvalue_op) = rvalue_op_opt {
                            if let Some(prev) = last_inserted {
                                rvalue_op.insert_after(ctx, prev);
                                current_prev = Some(rvalue_op);
                            } else if let Some(prev) = prev_op {
                                rvalue_op.insert_after(ctx, prev);
                                current_prev = Some(rvalue_op);
                            } else {
                                rvalue_op.insert_at_front(block_ptr, ctx);
                                current_prev = Some(rvalue_op);
                            }
                        } else if let Some(prev) = last_inserted {
                            current_prev = Some(prev);
                        }

                        let local = place.local;
                        let Some(arr_ptr) = value_map.get_slot(local) else {
                            return input_err!(
                                loc,
                                TranslationErr::unsupported(format!(
                                    "Local {:?} has no alloca slot for runtime index write",
                                    local
                                ))
                            );
                        };

                        let index_place = mir::Place {
                            local: *index_local,
                            projection: vec![],
                        };
                        let (index_value, prev_op_after_index) = rvalue::translate_place(
                            ctx,
                            body,
                            &index_place,
                            value_map,
                            block_ptr,
                            current_prev,
                            loc.clone(),
                        )?;
                        current_prev = prev_op_after_index;

                        let (element_ty, address_space) =
                            slot_array_element_ty(ctx, arr_ptr, &loc)?;

                        let store_op = emit_array_element_store(
                            ctx,
                            arr_ptr,
                            index_value,
                            result_value,
                            element_ty,
                            address_space,
                            block_ptr,
                            current_prev,
                            loc,
                        );
                        Ok(Some(store_op))
                    }
                    _ => input_err!(
                        loc,
                        TranslationErr::unsupported(
                            "Assignments to projections other than Deref, Field, ConstantIndex, and Index not yet implemented"
                        )
                    ),
                }
            } else if place.projection.len() == 2 {
                // Handle 2-level projections
                match (&place.projection[0], &place.projection[1]) {
                    (
                        mir::ProjectionElem::Deref,
                        mir::ProjectionElem::Field(field_idx, field_ty),
                    ) => {
                        // `(*ptr).field = value`.
                        //
                        // Alloca model: compute the field's address with
                        // `MirFieldAddrOp` applied to the pointer directly
                        // and store the new value with `MirStoreOp`.

                        let mut current_prev = prev_op;
                        if let Some(rvalue_op) = rvalue_op_opt {
                            if let Some(prev) = last_inserted {
                                rvalue_op.insert_after(ctx, prev);
                                current_prev = Some(rvalue_op);
                            } else if let Some(prev) = prev_op {
                                rvalue_op.insert_after(ctx, prev);
                                current_prev = Some(rvalue_op);
                            } else {
                                rvalue_op.insert_at_front(block_ptr, ctx);
                                current_prev = Some(rvalue_op);
                            }
                        } else if let Some(prev) = last_inserted {
                            current_prev = Some(prev);
                        }

                        let base_place = mir::Place {
                            local: place.local,
                            projection: vec![],
                        };
                        let (ptr_val, prev_op_after_ptr) = rvalue::translate_place(
                            ctx,
                            body,
                            &base_place,
                            value_map,
                            block_ptr,
                            current_prev,
                            loc.clone(),
                        )?;
                        current_prev = prev_op_after_ptr.or(current_prev);
                        reject_raw_field_index_on_enum_pointee(
                            ctx,
                            ptr_val,
                            &place.projection,
                            &loc,
                        )?;

                        let ptr_mutable = pointer_is_mutable(ctx, ptr_val);
                        let ptr_addr_space = pointer_address_space(ctx, ptr_val);

                        let field_type = types::translate_type(ctx, field_ty)?;
                        let field_ptr_ty = dialect_mir::types::MirPtrType::get(
                            ctx,
                            field_type,
                            ptr_mutable,
                            ptr_addr_space,
                        )
                        .into();

                        use dialect_mir::ops::MirFieldAddrOp;
                        let addr_op =
                            MirFieldAddrOp::build(ctx, ptr_val, field_ptr_ty, *field_idx as u32)?;
                        addr_op.deref_mut(ctx).set_loc(loc.clone());
                        if let Some(prev) = current_prev {
                            addr_op.insert_after(ctx, prev);
                        } else {
                            addr_op.insert_at_front(block_ptr, ctx);
                        }
                        let field_ptr = addr_op.deref(ctx).get_result(0);

                        let store_op = Operation::new(
                            ctx,
                            MirStoreOp::get_concrete_op_info(),
                            vec![],
                            vec![field_ptr, result_value],
                            vec![],
                            0,
                        );
                        store_op.deref_mut(ctx).set_loc(loc);
                        store_op.insert_after(ctx, addr_op);

                        Ok(Some(store_op))
                    }
                    (
                        mir::ProjectionElem::Field(outer_field_idx, outer_field_ty),
                        mir::ProjectionElem::Field(inner_field_idx, inner_field_ty),
                    ) => {
                        // `_local.outer.inner = value`.
                        //
                        // Alloca model: compose two `MirFieldAddrOp`s from the
                        // local's slot to reach the inner field's address,
                        // then store directly. `mem2reg` folds the chained
                        // addresses back into scalar field updates.

                        let mut current_prev = prev_op;
                        if let Some(rvalue_op) = rvalue_op_opt {
                            if let Some(prev) = last_inserted {
                                rvalue_op.insert_after(ctx, prev);
                                current_prev = Some(rvalue_op);
                            } else if let Some(prev) = prev_op {
                                rvalue_op.insert_after(ctx, prev);
                                current_prev = Some(rvalue_op);
                            } else {
                                rvalue_op.insert_at_front(block_ptr, ctx);
                                current_prev = Some(rvalue_op);
                            }
                        } else if let Some(prev) = last_inserted {
                            current_prev = Some(prev);
                        }

                        let Some(slot) = value_map.get_slot(place.local) else {
                            return input_err!(
                                loc,
                                TranslationErr::unsupported(format!(
                                    "Local {} has no alloca slot for nested field assignment",
                                    Into::<usize>::into(place.local)
                                ))
                            );
                        };
                        reject_raw_field_index_on_enum_pointee(ctx, slot, &place.projection, &loc)?;
                        let slot_mutable = pointer_is_mutable(ctx, slot);
                        let slot_addr_space = pointer_address_space(ctx, slot);

                        let outer_field_type = types::translate_type(ctx, outer_field_ty)?;
                        let outer_ptr_ty = dialect_mir::types::MirPtrType::get(
                            ctx,
                            outer_field_type,
                            slot_mutable,
                            slot_addr_space,
                        )
                        .into();

                        use dialect_mir::ops::MirFieldAddrOp;
                        let outer_addr_op = MirFieldAddrOp::build(
                            ctx,
                            slot,
                            outer_ptr_ty,
                            *outer_field_idx as u32,
                        )?;
                        outer_addr_op.deref_mut(ctx).set_loc(loc.clone());
                        if let Some(prev) = current_prev {
                            outer_addr_op.insert_after(ctx, prev);
                        } else {
                            outer_addr_op.insert_at_front(block_ptr, ctx);
                        }
                        current_prev = Some(outer_addr_op);
                        let outer_ptr = outer_addr_op.deref(ctx).get_result(0);
                        reject_raw_field_index_on_enum_pointee(
                            ctx,
                            outer_ptr,
                            &place.projection,
                            &loc,
                        )?;

                        let inner_field_type = types::translate_type(ctx, inner_field_ty)?;
                        let inner_ptr_ty = dialect_mir::types::MirPtrType::get(
                            ctx,
                            inner_field_type,
                            slot_mutable,
                            slot_addr_space,
                        )
                        .into();
                        let inner_addr_op = MirFieldAddrOp::build(
                            ctx,
                            outer_ptr,
                            inner_ptr_ty,
                            *inner_field_idx as u32,
                        )?;
                        inner_addr_op.deref_mut(ctx).set_loc(loc.clone());
                        if let Some(prev) = current_prev {
                            inner_addr_op.insert_after(ctx, prev);
                        } else {
                            inner_addr_op.insert_at_front(block_ptr, ctx);
                        }
                        let inner_ptr = inner_addr_op.deref(ctx).get_result(0);

                        let store_op = Operation::new(
                            ctx,
                            MirStoreOp::get_concrete_op_info(),
                            vec![],
                            vec![inner_ptr, result_value],
                            vec![],
                            0,
                        );
                        store_op.deref_mut(ctx).set_loc(loc);
                        store_op.insert_after(ctx, inner_addr_op);

                        Ok(Some(store_op))
                    }
                    (
                        mir::ProjectionElem::Deref,
                        mir::ProjectionElem::Index(_) | mir::ProjectionElem::ConstantIndex { .. },
                    ) => {
                        // `(*ptr)[i] = value`, e.g. `a[i] = v` where `a` is
                        // `&mut [T; N]` (thin pointer to an array) or
                        // `&mut [T]` (fat slice pointer). The shared
                        // walk-and-store path loads the pointer for the
                        // `Deref` (extracting the thin data pointer when the
                        // pointee is slice-shaped) and applies the index, so
                        // the store writes to the ORIGINAL storage.
                        store_through_place_address(
                            ctx,
                            body,
                            value_map,
                            place,
                            result_value,
                            rvalue_op_opt,
                            last_inserted,
                            prev_op,
                            block_ptr,
                            loc,
                        )
                    }
                    (
                        mir::ProjectionElem::Index(_) | mir::ProjectionElem::ConstantIndex { .. },
                        mir::ProjectionElem::Index(_) | mir::ProjectionElem::ConstantIndex { .. },
                    ) => {
                        // Nested array element assignment with any mix of
                        // runtime and constant indexes: `_local[i][j]`,
                        // `_local[CONST][j]`, `_local[i][CONST]`, or
                        // `_local[CONST][CONST]` (the last two arise when GVN
                        // or user code fixes one level). The shared
                        // walk-and-store path already handles chained indexes
                        // of either kind, so delegate to it instead of
                        // re-deriving the address here. That keeps this
                        // 2-level arm from drifting from the (Deref, Index)
                        // arm above and the N-projection fallback below,
                        // which use the same helper. The store target of an
                        // assignment is always a mutable place, so the
                        // helper's mutable-address request is correct here.
                        // `ConstantIndex { from_end: true, .. }` is accepted
                        // by the walker only when it follows a fat-slice deref,
                        // which supplies the runtime length metadata.
                        store_through_place_address(
                            ctx,
                            body,
                            value_map,
                            place,
                            result_value,
                            rvalue_op_opt,
                            last_inserted,
                            prev_op,
                            block_ptr,
                            loc,
                        )
                    }
                    (
                        mir::ProjectionElem::Field(_, _),
                        mir::ProjectionElem::ConstantIndex { .. } | mir::ProjectionElem::Index(_),
                    ) => {
                        // `_local.field[const]` or `_local.field[i]`: step into a
                        // struct field, then index into the resulting array. The
                        // walk-and-store helper resolves the full address chain.
                        store_through_place_address(
                            ctx,
                            body,
                            value_map,
                            place,
                            result_value,
                            rvalue_op_opt,
                            last_inserted,
                            prev_op,
                            block_ptr,
                            loc,
                        )
                    }
                    (
                        mir::ProjectionElem::Index(_) | mir::ProjectionElem::ConstantIndex { .. },
                        mir::ProjectionElem::Field(_, _),
                    ) => {
                        // `_local[i].field` or `_local[const].field`: index into
                        // an array/slice element, then step into a struct field.
                        // The walk-and-store helper resolves the full address
                        // chain, mirroring the (Field, Index) and (Index, Index)
                        // arms above. The store target of an assignment is always
                        // a mutable place, so the helper's mutable-address request
                        // is correct here.
                        store_through_place_address(
                            ctx,
                            body,
                            value_map,
                            place,
                            result_value,
                            rvalue_op_opt,
                            last_inserted,
                            prev_op,
                            block_ptr,
                            loc,
                        )
                    }
                    (mir::ProjectionElem::Downcast(_), mir::ProjectionElem::Field(_, _)) => {
                        // `(_local as Variant).field = value`, which is how a
                        // write through `&mut` to an enum payload arrives once
                        // the borrow is inlined away. The payload shares the
                        // enum's storage, so the walk-and-store path resolves
                        // the flattened payload position and the store lands in
                        // the enum itself rather than in a copy.
                        store_through_place_address(
                            ctx,
                            body,
                            value_map,
                            place,
                            result_value,
                            rvalue_op_opt,
                            last_inserted,
                            prev_op,
                            block_ptr,
                            loc,
                        )
                    }
                    _ => input_err!(
                        loc,
                        TranslationErr::unsupported(format!(
                            "2-level projection {:?} -> {:?} not yet implemented for assignment",
                            place.projection[0], place.projection[1]
                        ))
                    ),
                }
            } else {
                // 3+ projections, e.g. `(*iter).alive.start = value` from the
                // inlined `IndexRange::next_unchecked` inside
                // `core::array::IntoIter`'s `next` (the `for x in arr` loop
                // machinery, issue #138). Instead of enumerating every
                // combination by hand like the 1- and 2-level arms above, walk
                // the full projection chain to a destination address with the
                // same walker that `Rvalue::Ref` uses, then store through it.
                store_through_place_address(
                    ctx,
                    body,
                    value_map,
                    place,
                    result_value,
                    rvalue_op_opt,
                    last_inserted,
                    prev_op,
                    block_ptr,
                    loc,
                )
            }
        }
        mir::StatementKind::StorageLive(_local) => {
            // StorageLive marker
            let op = Operation::new(
                ctx,
                MirStorageLiveOp::get_concrete_op_info(),
                vec![],
                vec![],
                vec![],
                0,
            );
            op.deref_mut(ctx).set_loc(loc);

            if let Some(prev) = prev_op {
                op.insert_after(ctx, prev);
            } else {
                op.insert_at_front(block_ptr, ctx);
            }
            Ok(Some(op))
        }
        mir::StatementKind::StorageDead(_local) => {
            // StorageDead marker
            let op = Operation::new(
                ctx,
                MirStorageDeadOp::get_concrete_op_info(),
                vec![],
                vec![],
                vec![],
                0,
            );
            op.deref_mut(ctx).set_loc(loc);

            if let Some(prev) = prev_op {
                op.insert_after(ctx, prev);
            } else {
                op.insert_at_front(block_ptr, ctx);
            }
            Ok(Some(op))
        }
        mir::StatementKind::Nop => {
            // No-op statement, skip
            Ok(prev_op)
        }

        // Retag note: `StatementKind::Retag` no longer exists on the pinned
        // nightly; retagging is now carried as the `WithRetag` flag on
        // `Rvalue::Use`. Retags refine rustc's dynamic alias/provenance model
        // (Stacked/Tree Borrows) but have no runtime effect, so the importer
        // ignores the flag, exactly as it used to skip the statement. Any
        // future LLVM alias metadata must be based on an explicit audited
        // policy, never inferred from skipped retags or `MirPtrType::is_mutable`.

        // Other codegen-irrelevant borrow-check / type-system / coverage hints
        // have no runtime effect and are intentionally skipped.
        mir::StatementKind::FakeRead(..)
        | mir::StatementKind::PlaceMention(..)
        | mir::StatementKind::AscribeUserType { .. }
        | mir::StatementKind::Coverage(..)
        | mir::StatementKind::ConstEvalCounter => Ok(prev_op),

        // `Assume` is an optimisation hint with no observable effect; safe to skip.
        mir::StatementKind::Intrinsic(mir::NonDivergingIntrinsic::Assume(_)) => Ok(prev_op),

        mir::StatementKind::Intrinsic(mir::NonDivergingIntrinsic::CopyNonOverlapping(copy)) => {
            let (dst, last_op) = rvalue::translate_operand(
                ctx,
                body,
                &copy.dst,
                value_map,
                block_ptr,
                prev_op,
                loc.clone(),
            )?;
            let (src, last_op) = rvalue::translate_operand(
                ctx,
                body,
                &copy.src,
                value_map,
                block_ptr,
                last_op,
                loc.clone(),
            )?;
            let (count, last_op) = rvalue::translate_operand(
                ctx,
                body,
                &copy.count,
                value_map,
                block_ptr,
                last_op,
                loc.clone(),
            )?;

            // The typed builder stamps the elem_type fact from dst; lowering
            // scales the byte count from that attribute, not operand history.
            let memcpy_op = MirMemcpyOp::build(ctx, dst, src, count)?;
            memcpy_op.deref_mut(ctx).set_loc(loc);
            if let Some(prev) = last_op {
                memcpy_op.insert_after(ctx, prev);
            } else {
                memcpy_op.insert_at_front(block_ptr, ctx);
            }
            Ok(Some(memcpy_op))
        }

        // Statements with observable runtime effect that are not yet lowered.
        // Returning a hard error here converts what was previously a silent
        // miscompile (the catch-all `Ok(prev_op)`) into a clear build failure.
        // `SetDiscriminant` mutates an enum's discriminant in place.
        mir::StatementKind::SetDiscriminant {
            place,
            variant_index,
        } => {
            let place_ty = place.ty(body.locals()).map_err(|e| {
                input_error!(
                    loc.clone(),
                    TranslationErr::unsupported(format!(
                        "Failed to resolve place type for SetDiscriminant: {:?}",
                        e
                    ))
                )
            })?;
            let variant_idx = variant_index.to_index();
            let enum_mir_ty = types::translate_type(ctx, &place_ty)?;
            let (layout_kind, single_variant) = {
                let enum_ty_obj = enum_mir_ty.deref(ctx);
                let Some(enum_ty) = enum_ty_obj.downcast_ref::<MirEnumType>() else {
                    return input_err!(
                        loc,
                        TranslationErr::unsupported(format!(
                            "SetDiscriminant place type is not a supported enum: {}",
                            enum_mir_ty.disp(ctx)
                        ))
                    );
                };
                if variant_idx >= enum_ty.variant_count() {
                    return input_err!(
                        loc,
                        TranslationErr::unsupported(format!(
                            "SetDiscriminant variant index {} out of bounds for enum '{}'",
                            variant_idx,
                            enum_ty.name()
                        ))
                    );
                }
                if enum_ty.variant_is_inhabited(variant_idx) != Some(true) {
                    return input_err!(
                        loc,
                        TranslationErr::unsupported(format!(
                            "SetDiscriminant cannot select uninhabited variant {}",
                            variant_idx
                        ))
                    );
                }
                (enum_ty.layout_kind, enum_ty.single_variant as usize)
            };

            match layout_kind {
                dialect_mir::types::EnumLayoutKind::Single if variant_idx == single_variant => {
                    return Ok(prev_op);
                }
                dialect_mir::types::EnumLayoutKind::Direct
                | dialect_mir::types::EnumLayoutKind::Niche => {}
                other => {
                    return input_err!(
                        loc,
                        TranslationErr::unsupported(format!(
                            "SetDiscriminant cannot lower enum layout kind {:?}",
                            other
                        ))
                    );
                }
            }

            // Get the address of the enum place.
            let (enum_ptr, addr_prev) = match rvalue::translate_place_address(
                ctx,
                body,
                value_map,
                place,
                /* is_mutable */ true,
                block_ptr,
                prev_op,
                loc.clone(),
            )? {
                Some(pair) => pair,
                None => {
                    return input_err!(
                        loc,
                        TranslationErr::unsupported(
                            "SetDiscriminant place has no addressable slot".to_string()
                        )
                    );
                }
            };

            // The pointer type is determined by the place address walker;
            // MirSetDiscriminantOp's verifier will catch any type mismatch.
            let set_op = Operation::new(
                ctx,
                MirSetDiscriminantOp::get_concrete_op_info(),
                vec![],
                vec![enum_ptr],
                vec![],
                0,
            );
            set_op.deref_mut(ctx).set_loc(loc.clone());

            MirSetDiscriminantOp::new(set_op).set_attr_set_discriminant_variant_index(
                ctx,
                dialect_mir::attributes::VariantIndexAttr(variant_idx as u32),
            );
            // Stamp the enum type fact: lowering derives the tag/niche layout
            // from this attribute, never from operand type history.
            MirSetDiscriminantOp::new(set_op).set_attr_set_discriminant_enum_ty(
                ctx,
                pliron::builtin::attributes::TypeAttr::new(enum_mir_ty),
            );

            let insert_after = addr_prev.or(prev_op);
            if let Some(prev) = insert_after {
                set_op.insert_after(ctx, prev);
            } else {
                set_op.insert_at_front(block_ptr, ctx);
            }

            Ok(Some(set_op))
        }
    }
}

/// Shared walk-and-store path for projected assignments: insert the pending
/// rvalue op (if any), walk the FULL projection chain of `place` to a
/// mutable destination address with the same place-address walker that
/// `Rvalue::Ref` uses (`rvalue::translate_place_address`), then write
/// `result_value` through it with a single `mir.store`.
///
/// Used by the dedicated `(*ptr)[i] = value` arm (thin `&mut [T; N]` and
/// fat `&mut [T]` bases, issue #58) and by the generic fallback for 3+
/// projection chains such as `(*iter).alive.start = value` from the
/// `for x in arr` loop machinery (issue #138).
///
/// A punt from the walker is reported as an unsupported construct: the
/// destination is written through, so falling back to a value copy would
/// silently lose the write.
#[allow(clippy::too_many_arguments)]
fn store_through_place_address(
    ctx: &mut Context,
    body: &mir::Body,
    value_map: &ValueMap,
    place: &mir::Place,
    result_value: Value,
    rvalue_op_opt: Option<Ptr<Operation>>,
    last_inserted: Option<Ptr<Operation>>,
    prev_op: Option<Ptr<Operation>>,
    block_ptr: Ptr<BasicBlock>,
    loc: Location,
) -> TranslationResult<Option<Ptr<Operation>>> {
    let mut current_prev = prev_op;
    if let Some(rvalue_op) = rvalue_op_opt {
        if let Some(prev) = last_inserted {
            rvalue_op.insert_after(ctx, prev);
        } else if let Some(prev) = prev_op {
            rvalue_op.insert_after(ctx, prev);
        } else {
            rvalue_op.insert_at_front(block_ptr, ctx);
        }
        current_prev = Some(rvalue_op);
    } else if let Some(prev) = last_inserted {
        current_prev = Some(prev);
    }

    // A payload whose bytes use canonical storage has no address to write
    // through: bool payloads occupy a full byte and shared-memory pointers
    // are stored generic, while a store through an escaped address carries
    // the semantic type. Rebuild the enum around the new payload instead,
    // which coerces on the way in exactly as a whole-enum assignment does.
    if let Some(payload_store) = payload_store::classify(ctx, body, place)?
        && let Some(result) = payload_store::rebuild_and_store(
            ctx,
            body,
            value_map,
            &payload_store,
            result_value,
            block_ptr,
            current_prev,
            loc.clone(),
        )?
    {
        return Ok(result);
    }

    // The destination is written through, so request a mutable address.
    let walked = rvalue::translate_place_address(
        ctx,
        body,
        value_map,
        place,
        /* is_mutable */ true,
        block_ptr,
        current_prev,
        loc.clone(),
    )?;
    let Some((dest_addr, addr_prev)) = walked else {
        // The walker punted (a projection it cannot turn into an address,
        // or the local has no slot). Reject loudly instead of copying.
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "cannot compute the destination address for the assignment \
                 (projections {:?})",
                place.projection
            ))
        );
    };
    let current_prev = addr_prev.or(current_prev);

    let store_op = Operation::new(
        ctx,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![dest_addr, result_value],
        vec![],
        0,
    );
    store_op.deref_mut(ctx).set_loc(loc);
    match current_prev {
        Some(prev) => store_op.insert_after(ctx, prev),
        None => store_op.insert_at_front(block_ptr, ctx),
    }
    Ok(Some(store_op))
}

/// Extract the element type and address space from a pointer that points
/// to an array.
///
/// Used by the statement-level element write helpers. Returns a structured
/// error when the pointer's pointee isn't a [`MirArrayType`](dialect_mir::types::MirArrayType), which signals
/// a structural mismatch (most likely the wrong MIR projection reaching
/// this path).
pub(crate) fn slot_array_element_ty(
    ctx: &pliron::context::Context,
    arr_ptr: Value,
    loc: &Location,
) -> TranslationResult<(pliron::r#type::TypeHandle, u32)> {
    let arr_ptr_ty = arr_ptr.get_type(ctx);
    let arr_ptr_ty_ref = arr_ptr_ty.deref(ctx);
    let mir_ptr_ty = arr_ptr_ty_ref
        .downcast_ref::<dialect_mir::types::MirPtrType>()
        .ok_or_else(|| {
            pliron::input_error!(
                loc.clone(),
                TranslationErr::unsupported("Array-index slot is not a MirPtrType")
            )
        })?;
    let address_space = mir_ptr_ty.address_space;
    let pointee_ref = mir_ptr_ty.pointee.deref(ctx);
    let element_ty = pointee_ref
        .downcast_ref::<dialect_mir::types::MirArrayType>()
        .ok_or_else(|| {
            pliron::input_error!(
                loc.clone(),
                TranslationErr::unsupported("Array-index slot pointee is not MirArrayType",)
            )
        })?
        .element_type();
    Ok((element_ty, address_space))
}

/// Emit `mir.array_element_addr` + `mir.store` to assign `value` into
/// `array_ptr[index]`, returning the `mir.store` op.
///
/// The caller owns positioning (`prev_op`): we chain the address op after
/// it, then chain the store after the address op.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_array_element_store(
    ctx: &mut pliron::context::Context,
    array_ptr: Value,
    index: Value,
    value: Value,
    element_ty: pliron::r#type::TypeHandle,
    _address_space: u32,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> Ptr<Operation> {
    // Address projections retain the base pointer's provenance, mutability,
    // and address space. The result is not a new Rust borrow and therefore has
    // no authority to erase and later recover a different pointer category.
    let (base_origin, base_address_space) = {
        let base_ty = array_ptr.get_type(ctx);
        let base_ty = base_ty.deref(ctx);
        let base_ty = base_ty
            .downcast_ref::<dialect_mir::types::MirPtrType>()
            .expect("array element store base must be a MirPtrType");
        (
            facts::pointer_origin_of_ptr_carrier(base_ty),
            base_ty.address_space,
        )
    };
    let elem_ptr_ty = facts::mint_ptr_type(ctx, element_ty, base_address_space, base_origin).into();

    use dialect_mir::ops::MirArrayElementAddrOp;
    let addr_op = Operation::new(
        ctx,
        MirArrayElementAddrOp::get_concrete_op_info(),
        vec![elem_ptr_ty],
        vec![array_ptr, index],
        vec![],
        0,
    );
    addr_op.deref_mut(ctx).set_loc(loc.clone());
    match prev_op {
        Some(prev) => addr_op.insert_after(ctx, prev),
        None => addr_op.insert_at_front(block_ptr, ctx),
    }
    let elem_ptr = addr_op.deref(ctx).get_result(0);

    let store_op = Operation::new(
        ctx,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![elem_ptr, value],
        vec![],
        0,
    );
    store_op.deref_mut(ctx).set_loc(loc);
    store_op.insert_after(ctx, addr_op);
    store_op
}

/// Return `true` if the pointer value's type is a mutable [`MirPtrType`](dialect_mir::types::MirPtrType).
///
/// Slots emitted by the entry-block alloca loop are always mutable, but
/// callers of the statement module sometimes thread pointers coming from
/// other sources (loads, field-addr ops, ...), which may be immutable.
/// Derived addresses inherit the base pointer's mutability to keep pliron
/// type checking consistent. This bit is not an aliasing or uniqueness proof;
/// source-level reference kind is tracked separately by `MirPointerKind`.
pub(crate) fn pointer_is_mutable(ctx: &pliron::context::Context, ptr: Value) -> bool {
    let ty = ptr.get_type(ctx);
    let ty_ref = ty.deref(ctx);
    ty_ref
        .downcast_ref::<dialect_mir::types::MirPtrType>()
        .is_some_and(|p| p.is_mutable)
}

/// Return the address space of a pointer value. Defaults to 0 (the generic
/// address space) if the value is not a [`MirPtrType`](dialect_mir::types::MirPtrType).
pub(crate) fn pointer_address_space(ctx: &pliron::context::Context, ptr: Value) -> u32 {
    let ty = ptr.get_type(ctx);
    let ty_ref = ty.deref(ctx);
    ty_ref
        .downcast_ref::<dialect_mir::types::MirPtrType>()
        .map(|p| p.address_space)
        .unwrap_or(0)
}

/// Fail closed if a raw per-variant `Field` index is about to be applied to
/// an enum pointee by one of the fast assignment paths.
///
/// Valid MIR never applies `Field` to an enum place without a `Downcast`
/// naming the variant first, and every Downcast-bearing assignment routes
/// through the walk-and-store path, so the 1- and 2-level fast paths can
/// only meet an enum pointee through an importer bug or invalid MIR.
/// `MirFieldAddrOp` reads an enum-pointee index as a FLATTENED
/// (variant, field) position, so letting a raw per-variant index through
/// could silently address another variant's payload. This mirrors the
/// enum-pointee guard in the address walker's `Field` arm: an equally loud
/// failure at the only layer that can still tell the two index spaces apart.
pub(crate) fn reject_raw_field_index_on_enum_pointee(
    ctx: &Context,
    base_ptr: Value,
    projection: &[mir::ProjectionElem],
    loc: &Location,
) -> TranslationResult<()> {
    let pointee = base_ptr
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<dialect_mir::types::MirPtrType>()
        .map(|ptr| ptr.pointee);
    let pointee_is_enum = pointee.is_some_and(|pointee| pointee.deref(ctx).is::<MirEnumType>());
    if pointee_is_enum {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "Field assignment on an enum place without a preceding Downcast \
                 (projection {:?}); a raw per-variant field index would be misread \
                 as a flattened (variant, field) position",
                projection
            ))
        );
    }
    Ok(())
}

/// Assign an array aggregate element-by-element into addressable storage.
///
/// Handles `Rvalue::Repeat(operand, N)` assigned to an addressable local.
///
/// Translates the operand once and stores N copies directly into the alloca
/// via ConstantIndex projections. Avoids building a full SSA aggregate and the
/// resulting `insertvalue` chain in LLVM IR.
fn translate_repeat_into_alloca(
    ctx: &mut Context,
    body: &mir::Body,
    dest_place: &mir::Place,
    operand: &mir::Operand,
    count: usize,
    value_map: &mut ValueMap,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<Option<Ptr<Operation>>> {
    let (elem_val, after_operand) = rvalue::translate_operand(
        ctx,
        body,
        operand,
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    let mut current_prev = after_operand.or(prev_op);

    for i in 0..count {
        let elem_place = mir::Place {
            local: dest_place.local,
            projection: dest_place
                .projection
                .iter()
                .cloned()
                .chain(std::iter::once(mir::ProjectionElem::ConstantIndex {
                    offset: i as u64,
                    min_length: count as u64,
                    from_end: false,
                }))
                .collect(),
        };

        let after_store = store_through_place_address(
            ctx,
            body,
            value_map,
            &elem_place,
            elem_val,
            None,
            current_prev,
            current_prev,
            block_ptr,
            loc.clone(),
        )?;
        current_prev = after_store.or(current_prev);
    }

    Ok(current_prev)
}

/// When the destination has an alloca slot and the RHS is
/// `Rvalue::Aggregate(Array, operands)`, building a full SSA aggregate
/// (`MirConstructArrayOp`) followed by a single large store produces a chain
/// of `insertvalue` instructions in LLVM IR. This helper skips the aggregate
/// entirely and stores each element directly to its computed address.
fn translate_array_agg_into_alloca(
    ctx: &mut Context,
    body: &mir::Body,
    dest_place: &mir::Place,
    operands: &[mir::Operand],
    value_map: &mut ValueMap,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<Option<Ptr<Operation>>> {
    let mut current_prev = prev_op;

    for (i, operand) in operands.iter().enumerate() {
        let (elem_val, after_operand) = rvalue::translate_operand(
            ctx,
            body,
            operand,
            value_map,
            block_ptr,
            current_prev,
            loc.clone(),
        )?;
        current_prev = after_operand.or(current_prev);

        // Extend the destination place with a ConstantIndex for this element.
        let elem_place = mir::Place {
            local: dest_place.local,
            projection: dest_place
                .projection
                .iter()
                .cloned()
                .chain(std::iter::once(mir::ProjectionElem::ConstantIndex {
                    offset: i as u64,
                    min_length: operands.len() as u64,
                    from_end: false,
                }))
                .collect(),
        };

        let after_store = store_through_place_address(
            ctx,
            body,
            value_map,
            &elem_place,
            elem_val,
            None,
            current_prev,
            current_prev,
            block_ptr,
            loc.clone(),
        )?;
        current_prev = after_store.or(current_prev);
    }

    Ok(current_prev)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dialect_mir::types::{EnumVariant, MirEnumType, MirPtrType, MirStructType};
    use pliron::r#type::TypeHandle;

    /// The fast-path guard must refuse a raw `Field` index over an enum
    /// pointee (MirFieldAddrOp would misread it as a flattened
    /// (variant, field) position) and accept every non-enum pointee shape
    /// the fast paths legitimately handle.
    #[test]
    fn raw_field_index_guard_rejects_enum_pointees_only() {
        let mut ctx = Context::new();
        crate::translator::register_dialects(&mut ctx);

        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let struct_ty: TypeHandle =
            MirStructType::get(&mut ctx, "Plain".into(), vec!["a".into()], vec![u32_ty]).into();
        let enum_ty: TypeHandle = MirEnumType::get(
            &mut ctx,
            "Guarded".into(),
            u32_ty,
            vec![0, 1],
            vec![
                EnumVariant::new("A".into(), vec![u32_ty]),
                EnumVariant::unit("B".into()),
            ],
        )
        .into();
        let struct_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, struct_ty, true).into();
        let enum_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, enum_ty, true).into();

        let block = BasicBlock::new(&mut ctx, None, vec![struct_ptr_ty, enum_ptr_ty]);
        let struct_ptr = block.deref(&ctx).get_argument(0);
        let enum_ptr = block.deref(&ctx).get_argument(1);

        assert!(
            reject_raw_field_index_on_enum_pointee(&ctx, struct_ptr, &[], &Location::Unknown)
                .is_ok(),
            "a struct pointee is the fast paths' ordinary case and must pass"
        );

        let err = reject_raw_field_index_on_enum_pointee(&ctx, enum_ptr, &[], &Location::Unknown)
            .expect_err("an enum pointee must be refused loudly");
        assert!(
            format!("{err:?}").contains("without a preceding Downcast"),
            "unexpected error: {err:?}"
        );
    }
}
