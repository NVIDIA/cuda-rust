/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::common::{anyhow_to_pliron, byte_offset_gep, emit_integer_constant, spill_enum_value};
use super::enum_layout::{
    canonicalize_bool_value_bytes, coerce_enum_payload_storage, emit_carrier_constant,
    enum_carrier_bits_for_variant,
};
use crate::convert::enum_payload_storage::enum_payload_storage_type;
use crate::convert::types::{EnumSlotMap, build_enum_slot_map, convert_type, is_zero_sized_type};
use dialect_mir::ops::{MirConstructEnumOp, MirEnumPayloadOp, MirSetDiscriminantOp};
use dialect_mir::types::{EnumCarrierKind, EnumLayoutKind, MirEnumType};
use llvm_export::attributes::{ICmpPredicateAttr, IntegerOverflowFlagsAttr};
use llvm_export::op_interfaces::{
    CastOpInterface, CastOpWithNNegInterface, IntBinArithOpWithOverflowFlag,
};
use llvm_export::ops as llvm;
use llvm_export::types as llvm_types;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;

/// Convert `mir.construct_enum` (e.g. `E::A(x)`) to LLVM operations.
///
/// Builds the enum value slot by slot, taking every index from
/// [`build_enum_slot_map`] (indexes are never computed by hand here):
///
/// 1. Put the variant's declared discriminant VALUE into the tag slot.
/// 2. `insertvalue` each payload field that owns a struct slot.
/// 3. If some field has no slot (its bytes are shared with a
///    different-typed field of another variant), finish through memory:
///    copy the value to a stack slot, store that field at its byte
///    position, and load the completed enum back.
pub(crate) fn convert_construct_enum(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let (result_ty, operands, variant_index) = {
        let mir_op = op.deref(ctx);
        let result_ty = mir_op.get_result(0).get_type(ctx);
        let operands: Vec<_> = mir_op.operands().collect();

        let enum_op = MirConstructEnumOp::new(op);
        let variant_index = enum_op
            .get_attr_construct_enum_variant_index(ctx)
            .map(|attr| attr.0 as usize)
            .unwrap_or(0);

        (result_ty, operands, variant_index)
    };

    let enum_ty: MirEnumType = {
        let ty_ref = result_ty.deref(ctx);
        match ty_ref.downcast_ref::<MirEnumType>() {
            Some(e) => e.clone(),
            None => {
                return pliron::input_err_noloc!(
                    "MirConstructEnumOp result type must be MirEnumType"
                );
            }
        }
    };

    // Build the value as the SAME struct type the type converter
    // produces everywhere else (block args, loads, allocas, ...). Taking
    // both the type and the indices from one slot map is what keeps them
    // in agreement. Filler slots are simply never written.
    let slot_map = build_enum_slot_map(ctx, result_ty).map_err(anyhow_to_pliron)?;
    let llvm_struct_ty = slot_map.llvm_struct_ty;

    let undef_op = llvm::UndefOp::new(ctx, llvm_struct_ty);
    rewriter.insert_operation(ctx, undef_op.get_operation());
    let mut current_struct = undef_op.get_operation().deref(ctx).get_result(0);
    let mut last_op = undef_op.get_operation();

    if let Some(bits) = enum_carrier_bits_for_variant(&enum_ty, variant_index)
        .map_err(|error| pliron::input_error_noloc!("MirConstructEnumOp: {error}"))?
    {
        let carrier_slot = slot_map.carrier_slot.ok_or_else(|| {
            pliron::input_error_noloc!("MirConstructEnumOp requires a physical carrier slot")
        })?;
        let carrier_ty = slot_map.carrier_llvm_ty.ok_or_else(|| {
            pliron::input_error_noloc!("MirConstructEnumOp requires a physical carrier type")
        })?;
        let carrier = emit_carrier_constant(ctx, rewriter, &enum_ty, carrier_ty, bits)?;
        let insert = llvm::InsertValueOp::new(ctx, current_struct, carrier, vec![carrier_slot]);
        rewriter.insert_operation(ctx, insert.get_operation());
        current_struct = insert.get_operation().deref(ctx).get_result(0);
        last_op = insert.get_operation();
    }

    let field_base: usize = enum_ty
        .variant_field_counts
        .iter()
        .take(variant_index)
        .map(|&c| c as usize)
        .sum();

    // Insert every payload field that owns a struct slot; remember the
    // slotless ones for the memory pass below.
    let storage_field_tys: Vec<TypeHandle> = {
        let storage_type = llvm_struct_ty.deref(ctx);
        let storage_struct = storage_type
            .downcast_ref::<llvm_types::StructType>()
            .ok_or_else(|| {
                pliron::input_error_noloc!(
                    "MirConstructEnumOp physical storage is not an LLVM struct"
                )
            })?;
        storage_struct.fields().collect()
    };
    let mut deferred: Vec<(usize, Value)> = Vec::new();
    for (i, operand) in operands.into_iter().enumerate() {
        let flat = field_base + i;
        let Some(slot) = slot_map.field_slots.get(flat) else {
            return pliron::input_err_noloc!(
                "MirConstructEnumOp field {} of variant {} is out of range for the enum's {} fields",
                i,
                variant_index,
                slot_map.field_slots.len()
            );
        };
        // A decomposed aggregate field: one insertvalue per leaf, extracted
        // from the operand and coerced to its slot's storage type. No spill.
        if let Some(leaves) = slot_map.field_leaf_slots.get(flat).and_then(|l| l.as_ref()) {
            for (path, leaf_slot) in leaves {
                let extract_op = llvm::ExtractValueOp::new(ctx, operand, path.clone())?;
                rewriter.insert_operation(ctx, extract_op.get_operation());
                let leaf = extract_op.get_operation().deref(ctx).get_result(0);
                let storage_slot = *leaf_slot as usize;
                let Some(storage_ty) = storage_field_tys.get(storage_slot).copied() else {
                    return pliron::input_err_noloc!(
                        "MirConstructEnumOp physical field slot {} is out of range",
                        leaf_slot
                    );
                };
                let stored_leaf = coerce_enum_payload_storage(ctx, rewriter, leaf, storage_ty)?;
                let insert_op =
                    llvm::InsertValueOp::new(ctx, current_struct, stored_leaf, vec![*leaf_slot]);
                rewriter.insert_operation(ctx, insert_op.get_operation());
                current_struct = insert_op.get_operation().deref(ctx).get_result(0);
                last_op = insert_op.get_operation();
            }
            continue;
        }
        match slot {
            Some(slot) => {
                let storage_slot = *slot as usize;
                let Some(storage_ty) = storage_field_tys.get(storage_slot).copied() else {
                    return pliron::input_err_noloc!(
                        "MirConstructEnumOp physical field slot {} is out of range",
                        slot
                    );
                };
                let stored_operand =
                    coerce_enum_payload_storage(ctx, rewriter, operand, storage_ty)?;
                let insert_op =
                    llvm::InsertValueOp::new(ctx, current_struct, stored_operand, vec![*slot]);
                rewriter.insert_operation(ctx, insert_op.get_operation());
                current_struct = insert_op.get_operation().deref(ctx).get_result(0);
                last_op = insert_op.get_operation();
            }
            None => {
                // Zero-sized fields own no bytes; nothing to write.
                if is_zero_sized_type(ctx, slot_map.field_llvm_types[flat]) {
                    continue;
                }
                deferred.push((flat, operand));
            }
        }
    }

    if deferred.is_empty() {
        rewriter.replace_operation(ctx, op, last_op);
        return Ok(());
    }

    // Slotless fields: copy the half-built value to the stack, write
    // each remaining payload at its byte position, and load the finished
    // enum back as the result.
    let abi_align = enum_ty.abi_align();
    let slot_ptr = spill_enum_value(ctx, rewriter, current_struct, llvm_struct_ty, abi_align);
    for (flat, operand) in deferred {
        let field_ptr = byte_offset_gep(ctx, rewriter, slot_ptr, slot_map.field_offsets[flat]);
        // `bool` is an LLVM i1 as a value but occupies one full byte in
        // Rust memory. Enum storage claims the byte-faithful twin of every
        // bool-bearing payload (scalar i8 byte, or an aggregate with each
        // i1 leaf widened to i8), so canonicalize the stored value to that
        // twin: every physical bool byte becomes an unambiguous 0 or 1.
        let semantic_ty = slot_map.field_llvm_types[flat];
        let storage_ty = enum_payload_storage_type(ctx, semantic_ty).map_err(anyhow_to_pliron)?;
        let canonical_operand = canonicalize_bool_value_bytes(ctx, rewriter, operand)?;
        let stored_operand =
            coerce_enum_payload_storage(ctx, rewriter, canonical_operand, storage_ty)?;
        let store_op = llvm::StoreOp::new(ctx, stored_operand, field_ptr);
        rewriter.insert_operation(ctx, store_op.get_operation());
    }
    let load_op = llvm::LoadOp::new(ctx, slot_ptr, llvm_struct_ty);
    rewriter.insert_operation(ctx, load_op.get_operation());
    if abi_align > 0 {
        llvm_export::ops::set_op_alignment(ctx, load_op.get_operation(), abi_align as u32);
    }
    rewriter.replace_operation(ctx, op, load_op.get_operation());

    Ok(())
}

/// Get the slot map for an enum operand.
///
/// By the time an op is converted, its operand's type has already been
/// rewritten to the LLVM struct, so we look up the ORIGINAL `MirEnumType`
/// the framework recorded for it and rebuild the map from that. Also
/// returns the enum's rustc alignment, which spill slots need.
fn enum_slot_map_of_operand(
    ctx: &mut Context,
    operands_info: &OperandsInfo,
    enum_val: Value,
) -> Result<(EnumSlotMap, u64)> {
    // Clone the type data out so the `Ref` borrow of `ctx` ends before
    // re-interning (types are hash-consed: registering an equal instance
    // returns the existing pointer).
    let enum_ty: MirEnumType = {
        match operands_info.lookup_most_recent_of_type::<MirEnumType>(ctx, enum_val) {
            Some(r) => r.clone(),
            None => {
                return pliron::input_err_noloc!("Expected MirEnumType for enum value access");
            }
        }
    };
    let abi_align = enum_ty.abi_align();
    let mir_ty: TypeHandle = pliron::r#type::Type::instantiate(enum_ty, ctx).into();
    let map = build_enum_slot_map(ctx, mir_ty).map_err(anyhow_to_pliron)?;
    Ok((map, abi_align))
}

/// Convert `mir.set_discriminant` to the one physical carrier write rustc's
/// layout requires. Direct layouts write the declared discriminant; Niche
/// layouts write wrapping `niche_start + range_offset`. Selecting the
/// untagged Niche variant or an inhabited Single variant is a no-op.
///
/// The enum layout comes from the op's own `set_discriminant_enum_ty`
/// attribute, stamped at build time. Operand type history is not usable
/// here: a kind-only `mir.cast` lowers to a plain value forwarding, history
/// does not follow that edge, and a stale hit would write the tag at the
/// wrong offset.
pub(crate) fn convert_set_discriminant(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let enum_ptr = op.deref(ctx).get_operand(0);
    let target = MirSetDiscriminantOp::new(op)
        .get_attr_set_discriminant_variant_index(ctx)
        .map(|attr| attr.0 as usize)
        .ok_or_else(|| {
            pliron::input_error_noloc!(
                "MirSetDiscriminantOp missing set_discriminant_variant_index"
            )
        })?;

    let enum_ty: MirEnumType = {
        let stamped_ty = MirSetDiscriminantOp::new(op)
            .get_attr_set_discriminant_enum_ty(ctx)
            .map(|attr| attr.get_type(ctx))
            .ok_or_else(|| {
                pliron::input_error_noloc!(
                    "mir.set_discriminant missing enum type attribute; \
                     discriminant write has no fact to derive from"
                )
            })?;
        match stamped_ty.deref(ctx).downcast_ref::<MirEnumType>() {
            Some(et) => et.clone(),
            None => {
                return pliron::input_err_noloc!(
                    "mir.set_discriminant enum type attribute must be an enum type"
                );
            }
        }
    };

    let Some(bits) = enum_carrier_bits_for_variant(&enum_ty, target)
        .map_err(|error| pliron::input_error_noloc!("MirSetDiscriminantOp: {error}"))?
    else {
        rewriter.erase_operation(ctx, op);
        return Ok(());
    };

    let tag_offset = enum_ty.tag_offset();
    let mir_ty: TypeHandle = pliron::r#type::Type::instantiate(enum_ty.clone(), ctx).into();
    let slot_map = build_enum_slot_map(ctx, mir_ty).map_err(anyhow_to_pliron)?;
    let carrier_ty = slot_map.carrier_llvm_ty.ok_or_else(|| {
        pliron::input_error_noloc!("MirSetDiscriminantOp physical write has no carrier type")
    })?;
    let carrier = emit_carrier_constant(ctx, rewriter, &enum_ty, carrier_ty, bits)?;
    let carrier_ptr = byte_offset_gep(ctx, rewriter, enum_ptr, tag_offset);
    let store_op = llvm::StoreOp::new(ctx, carrier, carrier_ptr);
    rewriter.insert_operation(ctx, store_op.get_operation());

    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert `mir.get_discriminant` (reading which variant is alive) to
/// `llvm.extractvalue`.
///
/// Direct layouts read the tag from the slot map's carrier slot. Niche
/// layouts decode rustc's wrapping carrier range; Single layouts materialize
/// their one logical discriminant as a constant. No slot number is assumed.
pub(crate) fn convert_get_discriminant(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let enum_val = match op.deref(ctx).operands().next() {
        Some(v) => v,
        None => return pliron::input_err_noloc!("MirGetDiscriminantOp requires an operand"),
    };

    let enum_ty: MirEnumType = operands_info
        .lookup_most_recent_of_type::<MirEnumType>(ctx, enum_val)
        .map(|ty| ty.clone())
        .ok_or_else(|| pliron::input_error_noloc!("Expected MirEnumType for discriminant read"))?;
    let mir_ty: TypeHandle = pliron::r#type::Type::instantiate(enum_ty.clone(), ctx).into();
    let slot_map = build_enum_slot_map(ctx, mir_ty).map_err(anyhow_to_pliron)?;
    let logical_ty = convert_type(ctx, enum_ty.discriminant_ty).map_err(anyhow_to_pliron)?;
    let logical_width = logical_ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .map(IntegerType::width)
        .ok_or_else(|| {
            pliron::input_error_noloc!("MirGetDiscriminantOp logical result must be integer")
        })?;

    let result = match enum_ty.layout_kind {
        EnumLayoutKind::Direct => {
            let slot = slot_map.carrier_slot.ok_or_else(|| {
                pliron::input_error_noloc!("Direct enum has no physical carrier slot")
            })?;
            let extract = llvm::ExtractValueOp::new(ctx, enum_val, vec![slot])?;
            rewriter.insert_operation(ctx, extract.get_operation());
            extract.get_operation().deref(ctx).get_result(0)
        }
        EnumLayoutKind::Single => {
            if enum_ty.variant_is_inhabited(enum_ty.single_variant as usize) != Some(true) {
                return pliron::input_err_noloc!(
                    "Cannot read discriminant of an uninhabited single-variant enum"
                );
            }
            let value = *enum_ty
                .variant_discriminants
                .get(enum_ty.single_variant as usize)
                .ok_or_else(|| {
                    pliron::input_error_noloc!("Single enum has no declared discriminant")
                })?;
            emit_integer_constant(ctx, rewriter, logical_width, u128::from(value))
        }
        EnumLayoutKind::Niche => {
            let slot = slot_map.carrier_slot.ok_or_else(|| {
                pliron::input_error_noloc!("Niche enum has no physical carrier slot")
            })?;
            let extract = llvm::ExtractValueOp::new(ctx, enum_val, vec![slot])?;
            rewriter.insert_operation(ctx, extract.get_operation());
            let carrier = extract.get_operation().deref(ctx).get_result(0);
            let carrier_int_ty: TypeHandle =
                IntegerType::get(ctx, enum_ty.carrier_width, Signedness::Signless).into();
            let carrier_int = if enum_ty.carrier_kind == EnumCarrierKind::Pointer {
                let cast = llvm::PtrToIntOp::new(ctx, carrier, carrier_int_ty);
                rewriter.insert_operation(ctx, cast.get_operation());
                cast.get_operation().deref(ctx).get_result(0)
            } else {
                carrier
            };

            let niche_start =
                emit_integer_constant(ctx, rewriter, enum_ty.carrier_width, enum_ty.niche_start());
            let relative = llvm::SubOp::new_with_overflow_flag(
                ctx,
                carrier_int,
                niche_start,
                IntegerOverflowFlagsAttr::default(),
            )
            .get_operation();
            rewriter.insert_operation(ctx, relative);
            let relative_val = relative.deref(ctx).get_result(0);
            let max = emit_integer_constant(
                ctx,
                rewriter,
                enum_ty.carrier_width,
                u128::from(enum_ty.niche_variant_end - enum_ty.niche_variant_start),
            );
            let in_range =
                llvm::ICmpOp::new(ctx, ICmpPredicateAttr::ULE, relative_val, max).get_operation();
            rewriter.insert_operation(ctx, in_range);
            // The range test is carrier-width wrapping arithmetic, exactly
            // like rustc. Variant indices themselves live at the logical
            // discriminant width: the start index may be larger than the
            // carrier can represent (e.g. variants 298..=299 in an i8 niche).
            let logical_relative = match enum_ty.carrier_width.cmp(&logical_width) {
                std::cmp::Ordering::Equal => relative_val,
                std::cmp::Ordering::Greater => {
                    let cast = llvm::TruncOp::new(ctx, relative_val, logical_ty).get_operation();
                    rewriter.insert_operation(ctx, cast);
                    cast.deref(ctx).get_result(0)
                }
                std::cmp::Ordering::Less => {
                    // LLVM's zext op requires its explicit `nneg` flag even
                    // when the flag is false. Niche decoding is ordinary
                    // unsigned extension, so it must never claim nneg.
                    let cast = llvm::ZExtOp::new_with_nneg(ctx, relative_val, logical_ty, false)
                        .get_operation();
                    rewriter.insert_operation(ctx, cast);
                    cast.deref(ctx).get_result(0)
                }
            };
            let niche_base = emit_integer_constant(
                ctx,
                rewriter,
                logical_width,
                u128::from(enum_ty.niche_variant_start),
            );
            let niche_variant = llvm::AddOp::new_with_overflow_flag(
                ctx,
                logical_relative,
                niche_base,
                IntegerOverflowFlagsAttr::default(),
            )
            .get_operation();
            rewriter.insert_operation(ctx, niche_variant);
            let untagged = emit_integer_constant(
                ctx,
                rewriter,
                logical_width,
                u128::from(enum_ty.untagged_variant),
            );
            let in_range_value = in_range.deref(ctx).get_result(0);
            let niche_variant_value = niche_variant.deref(ctx).get_result(0);
            let select = llvm::SelectOp::new(ctx, in_range_value, niche_variant_value, untagged)
                .get_operation();
            rewriter.insert_operation(ctx, select);
            select.deref(ctx).get_result(0)
        }
        EnumLayoutKind::Empty => {
            return pliron::input_err_noloc!("Cannot read discriminant of uninhabited enum");
        }
        _ => {
            return pliron::input_err_noloc!(
                "Cannot read discriminant of enum with unknown physical layout"
            );
        }
    };

    rewriter.replace_operation_with_values(ctx, op, vec![result]);

    Ok(())
}

/// The LLVM type at an `extractvalue`/`insertvalue` index `path` into `ty`.
fn llvm_type_at_path(ctx: &Context, mut ty: TypeHandle, path: &[u32]) -> Option<TypeHandle> {
    for &index in path {
        ty = ty
            .deref(ctx)
            .downcast_ref::<llvm_types::StructType>()?
            .field_type(index as usize);
    }
    Some(ty)
}

/// Convert `mir.enum_payload` (reading a variant's field, e.g. the `x`
/// in `E::A(x) => x`) to a payload-field read.
///
/// Four cases, decided by the [`EnumSlotMap`]:
///
/// - The field owns a struct slot: a plain `llvm.extractvalue`.
/// - The field was decomposed into leaf slots: rebuild the aggregate in
///   SSA, one `llvm.extractvalue` per leaf slot and one `llvm.insertvalue`
///   per leaf path, coercing each leaf back to its semantic type. No spill.
/// - The field has no slot (its bytes are shared with a different-typed
///   field of another variant): go through memory. Copy the enum to a
///   stack slot, point at the field's byte position, and load it with
///   its own type. Same trick as
///   [`convert_extract_array_element`](super::array_extract::convert_extract_array_element),
///   and
///   it avoids LLVM `bitcast` entirely.
/// - The field is zero-sized: there is nothing to read; produce `undef`.
pub(crate) fn convert_enum_payload(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let enum_val = match op.deref(ctx).operands().next() {
        Some(v) => v,
        None => return pliron::input_err_noloc!("MirEnumPayloadOp requires an operand"),
    };

    let payload_op = MirEnumPayloadOp::new(op);
    let variant_index = payload_op
        .get_attr_payload_variant_index(ctx)
        .map(|attr| attr.0 as usize)
        .unwrap_or(0);
    let field_index = payload_op
        .get_attr_payload_field_index(ctx)
        .map(|attr| attr.0 as usize)
        .unwrap_or(0);

    let variant_field_counts = {
        match operands_info.lookup_most_recent_of_type::<MirEnumType>(ctx, enum_val) {
            Some(r) => r.variant_field_counts.clone(),
            None => {
                return pliron::input_err_noloc!(
                    "Expected MirEnumType for enum payload extraction"
                );
            }
        }
    };
    let (slot_map, abi_align) = enum_slot_map_of_operand(ctx, operands_info, enum_val)?;

    let field_base: usize = variant_field_counts
        .iter()
        .take(variant_index)
        .map(|&c| c as usize)
        .sum();
    let flat = field_base + field_index;
    let Some(slot) = slot_map.field_slots.get(flat).copied() else {
        return pliron::input_err_noloc!(
            "MirEnumPayloadOp field {} of variant {} is out of range for the enum's {} fields",
            field_index,
            variant_index,
            slot_map.field_slots.len()
        );
    };

    // A decomposed aggregate field: rebuild the whole value leaf by leaf
    // from the enum's SSA slots, with no round-trip through memory.
    if let Some(leaves) = slot_map.field_leaf_slots.get(flat).and_then(|l| l.as_ref()) {
        let semantic_ty = slot_map.field_llvm_types[flat];
        let undef_op = llvm::UndefOp::new(ctx, semantic_ty);
        rewriter.insert_operation(ctx, undef_op.get_operation());
        let mut current = undef_op.get_operation().deref(ctx).get_result(0);
        for (path, leaf_slot) in leaves {
            let extract_op = llvm::ExtractValueOp::new(ctx, enum_val, vec![*leaf_slot])?;
            rewriter.insert_operation(ctx, extract_op.get_operation());
            let stored_leaf = extract_op.get_operation().deref(ctx).get_result(0);
            let Some(semantic_leaf_ty) = llvm_type_at_path(ctx, semantic_ty, path) else {
                return pliron::input_err_noloc!(
                    "MirEnumPayloadOp leaf path {:?} does not resolve inside the field type",
                    path
                );
            };
            let semantic_leaf =
                coerce_enum_payload_storage(ctx, rewriter, stored_leaf, semantic_leaf_ty)?;
            let insert_op = llvm::InsertValueOp::new(ctx, current, semantic_leaf, path.clone());
            rewriter.insert_operation(ctx, insert_op.get_operation());
            current = insert_op.get_operation().deref(ctx).get_result(0);
        }
        rewriter.replace_operation_with_values(ctx, op, vec![current]);
        return Ok(());
    }

    match slot {
        Some(slot) => {
            let extract_op = llvm::ExtractValueOp::new(ctx, enum_val, vec![slot])?;
            rewriter.insert_operation(ctx, extract_op.get_operation());
            let extracted = extract_op.get_operation().deref(ctx).get_result(0);
            let semantic_value = coerce_enum_payload_storage(
                ctx,
                rewriter,
                extracted,
                slot_map.field_llvm_types[flat],
            )?;
            rewriter.replace_operation_with_values(ctx, op, vec![semantic_value]);
        }
        None if is_zero_sized_type(ctx, slot_map.field_llvm_types[flat]) => {
            let undef_op = llvm::UndefOp::new(ctx, slot_map.field_llvm_types[flat]);
            rewriter.insert_operation(ctx, undef_op.get_operation());
            rewriter.replace_operation(ctx, op, undef_op.get_operation());
        }
        None => {
            let slot_ptr =
                spill_enum_value(ctx, rewriter, enum_val, slot_map.llvm_struct_ty, abi_align);
            let field_ptr = byte_offset_gep(ctx, rewriter, slot_ptr, slot_map.field_offsets[flat]);
            let semantic_ty = slot_map.field_llvm_types[flat];
            let storage_ty =
                enum_payload_storage_type(ctx, semantic_ty).map_err(anyhow_to_pliron)?;
            let load_op = llvm::LoadOp::new(ctx, field_ptr, storage_ty);
            rewriter.insert_operation(ctx, load_op.get_operation());
            let stored = load_op.get_operation().deref(ctx).get_result(0);
            let semantic = coerce_enum_payload_storage(ctx, rewriter, stored, semantic_ty)?;
            rewriter.replace_operation_with_values(ctx, op, vec![semantic]);
        }
    }

    Ok(())
}

#[cfg(test)]
// Tests build kinded fixture types directly; production minting lives in mir-importer's facts.rs.
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;
    use crate::convert::ops::test_util::*;
    use dialect_mir::attributes::{FieldIndexAttr, MirCastKindAttr, VariantIndexAttr};
    use dialect_mir::ops as mir;
    use dialect_mir::types::{
        EnumEncoding, EnumVariant, MirArrayType, MirPtrType, MirStructType, MirTupleType,
    };
    use llvm_export::types as llvm_types;
    use pliron::builtin::attributes::IntegerAttr;
    use pliron::common_traits::Verify;

    use super::super::test_support::*;

    /// Enum construction must store the declared discriminant value, not the
    /// variant index. This locks the `Ordering::Less = -1` style case as the
    /// i8 bit-pattern `255`.
    #[test]
    fn construct_enum_uses_declared_discriminant_not_variant_index() {
        let mut ctx = make_ctx();

        let discr_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Signed).into();
        let enum_ty: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "OrderingLike".to_string(),
            discr_ty,
            vec![255, 0, 1],
            vec![
                EnumVariant::unit("Less".to_string()),
                EnumVariant::unit("Equal".to_string()),
                EnumVariant::unit("Greater".to_string()),
            ],
            0,
            1,
            1,
        )
        .into();

        let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);

        let op = Operation::new(
            &mut ctx,
            MirConstructEnumOp::get_concrete_op_info(),
            vec![enum_ty],
            vec![],
            vec![],
            0,
        );
        MirConstructEnumOp::new(op)
            .set_attr_construct_enum_variant_index(&ctx, VariantIndexAttr(0));
        op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        let inserts = find_all::<llvm::InsertValueOp>(&ctx, &body);

        assert_eq!(
            insert_indices(&ctx, &inserts),
            vec![vec![0]],
            "unit enum construction should insert only the discriminant tag"
        );
        let tag_insert = &inserts[0];
        assert!(
            tag_insert.verify(&ctx).is_ok(),
            "the enum tag insertion must satisfy LLVM dialect verification"
        );
        let tag = tag_insert.get_operation().deref(&ctx).get_operand(1);
        let tag_def = tag
            .defining_op()
            .expect("the inserted enum tag must have a defining operation");
        let tag_constant = Operation::get_op::<llvm::ConstantOp>(tag_def, &ctx)
            .expect("the inserted enum tag must be defined by llvm.constant");
        let tag_attr = tag_constant.get_value(&ctx);
        let tag_integer = (&*tag_attr as &dyn pliron::attribute::Attribute)
            .downcast_ref::<IntegerAttr>()
            .expect("the inserted enum tag must be an integer constant");
        assert_eq!(tag_integer.value().bw(), 8, "the enum tag must be 8-bit");
        assert_eq!(
            tag_integer.value().to_u64(),
            255,
            "Less must lower to its declared i8 bit-pattern 255, not variant index 0"
        );
    }

    /// `Option<(usize, &T)>` with the niche on the tuple's pointer: the
    /// aggregate payload is decomposed into leaf slots (the `usize` leaf at
    /// byte 0, the pointer leaf reusing the carrier at byte 8), so construct
    /// and payload read rebuild the value entirely in SSA.
    ///
    /// This test used to pin the spill design ("the payload moves as one
    /// unit through memory, never field by field"): construction and
    /// extraction each spilled the enum and moved the whole tuple through
    /// the slot. The leaf-slot rule (enum-typed-payload round) replaced that
    /// with typed leaf traffic on purpose: byte-wise spills of mixed-type
    /// filler were what SROA could not promote, leaving local-memory
    /// traffic in `opt -O2` output. Both leaves still round-trip — now as
    /// `extractvalue`/`insertvalue` pairs, with zero memory ops.
    #[test]
    fn nested_pointer_niche_tuple_construct_extract_and_discriminant_lower() {
        let mut ctx = make_ctx();
        let logical: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signed).into();
        let index: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let pointee: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let pointer: TypeHandle = MirPtrType::get_generic(&mut ctx, pointee, false).into();
        let tuple_ty: TypeHandle = MirTupleType::get(&mut ctx, vec![index, pointer]).into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "Option".into(),
            logical,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout("Some".into(), vec![tuple_ty], vec![0], vec![16]),
            ],
            EnumEncoding {
                tag_offset: 8,
                total_size: 16,
                abi_align: 8,
                layout_kind: EnumLayoutKind::Niche,
                carrier_kind: EnumCarrierKind::Pointer,
                carrier_width: 64,
                untagged_variant: 1,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into();
        let slot_map = build_enum_slot_map(&mut ctx, enum_ty).unwrap();
        assert_eq!(slot_map.carrier_slot, Some(1));
        assert_eq!(slot_map.field_slots, vec![None]);

        let (module, block) = build_kernel(&mut ctx, vec![index, pointer], vec![]);
        let index_value = block.deref(&ctx).get_argument(0);
        let pointer_value = block.deref(&ctx).get_argument(1);
        let tuple = Operation::new(
            &mut ctx,
            mir::MirConstructTupleOp::get_concrete_op_info(),
            vec![tuple_ty],
            vec![index_value, pointer_value],
            vec![],
            0,
        );
        tuple.insert_at_back(block, &ctx);
        let tuple_value = tuple.deref(&ctx).get_result(0);

        let construct = Operation::new(
            &mut ctx,
            mir::MirConstructEnumOp::get_concrete_op_info(),
            vec![enum_ty],
            vec![tuple_value],
            vec![],
            0,
        );
        mir::MirConstructEnumOp::new(construct)
            .set_attr_construct_enum_variant_index(&ctx, VariantIndexAttr(1));
        construct.insert_at_back(block, &ctx);
        let enum_value = construct.deref(&ctx).get_result(0);

        let payload = Operation::new(
            &mut ctx,
            mir::MirEnumPayloadOp::get_concrete_op_info(),
            vec![tuple_ty],
            vec![enum_value],
            vec![],
            0,
        );
        mir::MirEnumPayloadOp::new(payload)
            .set_attr_payload_variant_index(&ctx, VariantIndexAttr(1));
        mir::MirEnumPayloadOp::new(payload).set_attr_payload_field_index(&ctx, FieldIndexAttr(0));
        payload.insert_at_back(block, &ctx);

        let discriminant = Operation::new(
            &mut ctx,
            mir::MirGetDiscriminantOp::get_concrete_op_info(),
            vec![logical],
            vec![enum_value],
            vec![],
            0,
        );
        discriminant.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module).expect("lowering failed");
        let body = kernel_blocks(&ctx, module);
        assert_eq!(
            count_ops::<llvm::IntToPtrOp>(&ctx, &body),
            0,
            "constructing the untagged Some payload must not recreate its pointer from bits"
        );
        assert_eq!(
            count_ops::<llvm::PtrToIntOp>(&ctx, &body),
            1,
            "reading the pointer niche should inspect the carrier exactly once"
        );
        assert_eq!(
            count_ops::<llvm::AllocaOp>(&ctx, &body),
            0,
            "leaf slots keep construction and extraction out of memory entirely"
        );
        assert_eq!(
            count_ops::<llvm::StoreOp>(&ctx, &body),
            0,
            "the decomposed payload must not spill through a stack slot"
        );
        assert_eq!(
            count_ops::<llvm::LoadOp>(&ctx, &body),
            0,
            "the decomposed payload must be rebuilt from SSA slots, not loads"
        );
        // Both leaves must actually move: the tuple's own construction,
        // the enum construction, and the payload rebuild each insert both
        // leaves, and the discriminant read extracts the carrier once.
        // (The enum's physical storage here is `{i64, ptr}` -- the same
        // interned type as the lowered payload tuple, since the carrier
        // claims the pointer at byte 8 -- so the counts, not the aggregate
        // types, carry the assertion.)
        let mut inserted: Vec<Vec<u32>> =
            insert_indices(&ctx, &find_all::<llvm::InsertValueOp>(&ctx, &body));
        inserted.sort();
        assert_eq!(
            inserted,
            vec![vec![0], vec![0], vec![0], vec![1], vec![1], vec![1]],
            "the tuple build, the enum build, and the payload rebuild each insert both leaves"
        );
        let mut extracted: Vec<Vec<u32>> = find_all::<llvm::ExtractValueOp>(&ctx, &body)
            .iter()
            .map(|extract| extract.indices(&ctx))
            .collect();
        extracted.sort();
        assert_eq!(
            extracted,
            vec![vec![0], vec![0], vec![1], vec![1], vec![1]],
            "each leaf is read from the operand tuple, from the enum slots, plus the carrier"
        );
    }

    /// The `Option<RopePair>` shape ({&mut T, usize, usize}, niche on the
    /// pointer): one fixture shared by the SSA and niche tests below.
    fn rope_pair_enum(ctx: &mut Context) -> (TypeHandle, TypeHandle, TypeHandle) {
        let logical: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
        let word: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
        let pointee: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
        let pointer: TypeHandle = MirPtrType::get_generic(ctx, pointee, false).into();
        let payload: TypeHandle = MirStructType::get_with_full_layout(
            ctx,
            "RopePair".into(),
            vec!["rope".into(), "base".into(), "cache".into()],
            vec![pointer, word, word],
            vec![0, 1, 2],
            vec![0, 8, 16],
            24,
            8,
        )
        .into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            ctx,
            "MaybeRopePair".into(),
            logical,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout("Some".into(), vec![payload], vec![0], vec![24]),
            ],
            EnumEncoding {
                tag_offset: 0,
                total_size: 24,
                abi_align: 8,
                layout_kind: EnumLayoutKind::Niche,
                carrier_kind: EnumCarrierKind::Pointer,
                carrier_width: 64,
                untagged_variant: 1,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into();
        (enum_ty, payload, pointer)
    }

    /// The `Option<(f32, f32, &mut f32)>` shape: floats below the pointer
    /// that carries the niche.
    fn float_pair_enum(ctx: &mut Context) -> (TypeHandle, TypeHandle, TypeHandle, TypeHandle) {
        use pliron::builtin::types::FP32Type;
        let logical: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
        let float: TypeHandle = FP32Type::get(ctx).into();
        let pointee: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
        let pointer: TypeHandle = MirPtrType::get_generic(ctx, pointee, false).into();
        let payload: TypeHandle = MirTupleType::get_with_layout(
            ctx,
            vec![float, float, pointer],
            vec![0, 1, 2],
            vec![0, 4, 8],
            16,
            8,
        )
        .into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            ctx,
            "MaybeFloatPair".into(),
            logical,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout("Some".into(), vec![payload], vec![0], vec![16]),
            ],
            EnumEncoding {
                tag_offset: 8,
                total_size: 16,
                abi_align: 8,
                layout_kind: EnumLayoutKind::Niche,
                carrier_kind: EnumCarrierKind::Pointer,
                carrier_width: 64,
                untagged_variant: 1,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into();
        (enum_ty, payload, pointer, float)
    }

    /// Construct-then-payload-read of a decomposed aggregate payload stays
    /// in SSA: no `alloca`, no `store`, no `load`. The spill the old lowering
    /// needed here (byte stores into filler, one wide load back) is exactly
    /// what SROA cannot promote, and what survived `opt -O2` as local-memory
    /// traffic in the rms_norm/rope kernels. The tagged no-niche pair is the
    /// regression guard: it was already whole-slot SSA before the leaf rule.
    #[test]
    fn aggregate_payload_construct_and_read_stay_in_ssa() {
        // Option<RopePair>: {&mut T, usize, usize} built from its fields.
        {
            let mut ctx = make_ctx();
            let (enum_ty, payload, pointer) = rope_pair_enum(&mut ctx);
            let word: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
            let (module, block) = build_kernel(&mut ctx, vec![pointer, word, word], vec![]);
            let rope = block.deref(&ctx).get_argument(0);
            let base = block.deref(&ctx).get_argument(1);
            let cache = block.deref(&ctx).get_argument(2);
            let struct_op = Operation::new(
                &mut ctx,
                mir::MirConstructStructOp::get_concrete_op_info(),
                vec![payload],
                vec![rope, base, cache],
                vec![],
                0,
            );
            struct_op.insert_at_back(block, &ctx);
            let struct_value = struct_op.deref(&ctx).get_result(0);
            let construct = Operation::new(
                &mut ctx,
                mir::MirConstructEnumOp::get_concrete_op_info(),
                vec![enum_ty],
                vec![struct_value],
                vec![],
                0,
            );
            mir::MirConstructEnumOp::new(construct)
                .set_attr_construct_enum_variant_index(&ctx, VariantIndexAttr(1));
            construct.insert_at_back(block, &ctx);
            let enum_value = construct.deref(&ctx).get_result(0);
            let payload_read = Operation::new(
                &mut ctx,
                mir::MirEnumPayloadOp::get_concrete_op_info(),
                vec![payload],
                vec![enum_value],
                vec![],
                0,
            );
            mir::MirEnumPayloadOp::new(payload_read)
                .set_attr_payload_variant_index(&ctx, VariantIndexAttr(1));
            mir::MirEnumPayloadOp::new(payload_read)
                .set_attr_payload_field_index(&ctx, FieldIndexAttr(0));
            payload_read.insert_at_back(block, &ctx);
            append_mir_return(&mut ctx, block, vec![]);

            crate::lower_mir_to_llvm(&mut ctx, module).expect("lowering failed");
            let body = kernel_blocks(&ctx, module);
            assert_eq!(
                count_ops::<llvm::AllocaOp>(&ctx, &body),
                0,
                "the RopePair payload must not spill through a stack slot"
            );
            assert_eq!(count_ops::<llvm::StoreOp>(&ctx, &body), 0);
            assert_eq!(count_ops::<llvm::LoadOp>(&ctx, &body), 0);
        }
        // Option<(f32, f32, &mut f32)>: tuple built from its elements.
        {
            let mut ctx = make_ctx();
            let (enum_ty, payload, pointer, float) = float_pair_enum(&mut ctx);
            let (module, block) = build_kernel(&mut ctx, vec![float, float, pointer], vec![]);
            let first = block.deref(&ctx).get_argument(0);
            let second = block.deref(&ctx).get_argument(1);
            let cell = block.deref(&ctx).get_argument(2);
            let tuple_op = Operation::new(
                &mut ctx,
                mir::MirConstructTupleOp::get_concrete_op_info(),
                vec![payload],
                vec![first, second, cell],
                vec![],
                0,
            );
            tuple_op.insert_at_back(block, &ctx);
            let tuple_value = tuple_op.deref(&ctx).get_result(0);
            let construct = Operation::new(
                &mut ctx,
                mir::MirConstructEnumOp::get_concrete_op_info(),
                vec![enum_ty],
                vec![tuple_value],
                vec![],
                0,
            );
            mir::MirConstructEnumOp::new(construct)
                .set_attr_construct_enum_variant_index(&ctx, VariantIndexAttr(1));
            construct.insert_at_back(block, &ctx);
            let enum_value = construct.deref(&ctx).get_result(0);
            let payload_read = Operation::new(
                &mut ctx,
                mir::MirEnumPayloadOp::get_concrete_op_info(),
                vec![payload],
                vec![enum_value],
                vec![],
                0,
            );
            mir::MirEnumPayloadOp::new(payload_read)
                .set_attr_payload_variant_index(&ctx, VariantIndexAttr(1));
            mir::MirEnumPayloadOp::new(payload_read)
                .set_attr_payload_field_index(&ctx, FieldIndexAttr(0));
            payload_read.insert_at_back(block, &ctx);
            append_mir_return(&mut ctx, block, vec![]);

            crate::lower_mir_to_llvm(&mut ctx, module).expect("lowering failed");
            let body = kernel_blocks(&ctx, module);
            assert_eq!(
                count_ops::<llvm::AllocaOp>(&ctx, &body),
                0,
                "the float-pair payload must not spill through a stack slot"
            );
            assert_eq!(count_ops::<llvm::StoreOp>(&ctx, &body), 0);
            assert_eq!(count_ops::<llvm::LoadOp>(&ctx, &body), 0);
        }
        // Option<(u32, u32)>: tagged, no niche, whole-slot payload — the
        // regression guard that passes with and without the leaf rule.
        {
            let mut ctx = make_ctx();
            let logical: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
            let word: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
            let payload: TypeHandle = MirTupleType::get_with_layout(
                &mut ctx,
                vec![word, word],
                vec![0, 1],
                vec![0, 4],
                8,
                4,
            )
            .into();
            let enum_ty: TypeHandle = MirEnumType::get_with_layout(
                &mut ctx,
                "MaybeWords".into(),
                logical,
                vec![0, 1],
                vec![
                    EnumVariant::unit("None".into()),
                    EnumVariant::new_with_layout("Some".into(), vec![payload], vec![4], vec![8]),
                ],
                0,
                12,
                4,
            )
            .into();
            let (module, block) = build_kernel(&mut ctx, vec![word, word], vec![]);
            let first = block.deref(&ctx).get_argument(0);
            let second = block.deref(&ctx).get_argument(1);
            let tuple_op = Operation::new(
                &mut ctx,
                mir::MirConstructTupleOp::get_concrete_op_info(),
                vec![payload],
                vec![first, second],
                vec![],
                0,
            );
            tuple_op.insert_at_back(block, &ctx);
            let tuple_value = tuple_op.deref(&ctx).get_result(0);
            let construct = Operation::new(
                &mut ctx,
                mir::MirConstructEnumOp::get_concrete_op_info(),
                vec![enum_ty],
                vec![tuple_value],
                vec![],
                0,
            );
            mir::MirConstructEnumOp::new(construct)
                .set_attr_construct_enum_variant_index(&ctx, VariantIndexAttr(1));
            construct.insert_at_back(block, &ctx);
            let enum_value = construct.deref(&ctx).get_result(0);
            let payload_read = Operation::new(
                &mut ctx,
                mir::MirEnumPayloadOp::get_concrete_op_info(),
                vec![payload],
                vec![enum_value],
                vec![],
                0,
            );
            mir::MirEnumPayloadOp::new(payload_read)
                .set_attr_payload_variant_index(&ctx, VariantIndexAttr(1));
            mir::MirEnumPayloadOp::new(payload_read)
                .set_attr_payload_field_index(&ctx, FieldIndexAttr(0));
            payload_read.insert_at_back(block, &ctx);
            append_mir_return(&mut ctx, block, vec![]);

            crate::lower_mir_to_llvm(&mut ctx, module).expect("lowering failed");
            let body = kernel_blocks(&ctx, module);
            assert_eq!(count_ops::<llvm::AllocaOp>(&ctx, &body), 0);
            assert_eq!(count_ops::<llvm::StoreOp>(&ctx, &body), 0);
            assert_eq!(count_ops::<llvm::LoadOp>(&ctx, &body), 0);
        }
    }

    /// The leaf rule must not disturb the niche encoding. `None` still
    /// writes a null pointer into the carrier slot and the discriminant read
    /// still inspects exactly that slot; `Some` still writes the payload's
    /// pointer there — now as the leaf insert instead of a memory spill.
    #[test]
    fn pointer_niche_leaf_slots_keep_null_none_and_payload_pointer_some() {
        let check = |carrier_slot: u32, pointer_path: Vec<u32>, is_struct_payload: bool| {
            // None: the carrier write is IntToPtr(constant 0), and the
            // discriminant reads the carrier slot once.
            {
                let mut ctx = make_ctx();
                let (enum_ty, payload) = if is_struct_payload {
                    let (enum_ty, payload, _) = rope_pair_enum(&mut ctx);
                    (enum_ty, payload)
                } else {
                    let (enum_ty, payload, _, _) = float_pair_enum(&mut ctx);
                    (enum_ty, payload)
                };
                let _ = &payload;
                let logical: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
                let (module, block) = build_kernel(&mut ctx, vec![], vec![logical]);
                let construct = Operation::new(
                    &mut ctx,
                    mir::MirConstructEnumOp::get_concrete_op_info(),
                    vec![enum_ty],
                    vec![],
                    vec![],
                    0,
                );
                mir::MirConstructEnumOp::new(construct)
                    .set_attr_construct_enum_variant_index(&ctx, VariantIndexAttr(0));
                construct.insert_at_back(block, &ctx);
                let enum_value = construct.deref(&ctx).get_result(0);
                let get = Operation::new(
                    &mut ctx,
                    mir::MirGetDiscriminantOp::get_concrete_op_info(),
                    vec![logical],
                    vec![enum_value],
                    vec![],
                    0,
                );
                get.insert_at_back(block, &ctx);
                let discriminant = get.deref(&ctx).get_result(0);
                append_mir_return(&mut ctx, block, vec![discriminant]);

                crate::lower_mir_to_llvm(&mut ctx, module).expect("lowering failed");
                let body = kernel_blocks(&ctx, module);
                let carrier_inserts: Vec<_> = find_all::<llvm::InsertValueOp>(&ctx, &body)
                    .into_iter()
                    .filter(|insert| insert.indices(&ctx) == vec![carrier_slot])
                    .collect();
                assert_eq!(
                    carrier_inserts.len(),
                    1,
                    "constructing None writes the carrier slot exactly once"
                );
                let value = carrier_inserts[0]
                    .get_operation()
                    .deref(&ctx)
                    .get_operand(1);
                let int_to_ptr = Operation::get_op::<llvm::IntToPtrOp>(
                    value.defining_op().expect("carrier value must be defined"),
                    &ctx,
                )
                .expect("None's carrier write is a null pointer, not a payload");
                let bits = int_to_ptr.get_operation().deref(&ctx).get_operand(0);
                let bits_constant = Operation::get_op::<llvm::ConstantOp>(
                    bits.defining_op().expect("null bits must be defined"),
                    &ctx,
                )
                .expect("None's niche bits must be a constant");
                let attr = bits_constant.get_value(&ctx);
                let integer = (&*attr as &dyn pliron::attribute::Attribute)
                    .downcast_ref::<IntegerAttr>()
                    .expect("niche bits must be an integer");
                assert_eq!(
                    integer.value().to_u64(),
                    0,
                    "None must encode as the null pointer niche"
                );
                let ptr_to_ints = find_all::<llvm::PtrToIntOp>(&ctx, &body);
                assert_eq!(
                    ptr_to_ints.len(),
                    1,
                    "the discriminant reads the carrier once"
                );
                let carrier_read = ptr_to_ints[0].get_operation().deref(&ctx).get_operand(0);
                let extract = Operation::get_op::<llvm::ExtractValueOp>(
                    carrier_read
                        .defining_op()
                        .expect("carrier read must be defined"),
                    &ctx,
                )
                .expect("the discriminant must read the carrier slot");
                assert_eq!(extract.indices(&ctx), vec![carrier_slot]);
            }
            // Some: the carrier slot receives the payload's pointer leaf.
            {
                let mut ctx = make_ctx();
                let (enum_ty, payload, operands, pointer) = if is_struct_payload {
                    let (enum_ty, payload, pointer) = rope_pair_enum(&mut ctx);
                    let word: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
                    (enum_ty, payload, vec![pointer, word, word], pointer)
                } else {
                    let (enum_ty, payload, pointer, float) = float_pair_enum(&mut ctx);
                    (enum_ty, payload, vec![float, float, pointer], pointer)
                };
                let _ = pointer;
                let (module, block) = build_kernel(&mut ctx, operands.clone(), vec![]);
                let values: Vec<_> = (0..operands.len())
                    .map(|index| block.deref(&ctx).get_argument(index))
                    .collect();
                let build_info = if is_struct_payload {
                    mir::MirConstructStructOp::get_concrete_op_info()
                } else {
                    mir::MirConstructTupleOp::get_concrete_op_info()
                };
                let payload_op =
                    Operation::new(&mut ctx, build_info, vec![payload], values, vec![], 0);
                payload_op.insert_at_back(block, &ctx);
                let payload_value = payload_op.deref(&ctx).get_result(0);
                let construct = Operation::new(
                    &mut ctx,
                    mir::MirConstructEnumOp::get_concrete_op_info(),
                    vec![enum_ty],
                    vec![payload_value],
                    vec![],
                    0,
                );
                mir::MirConstructEnumOp::new(construct)
                    .set_attr_construct_enum_variant_index(&ctx, VariantIndexAttr(1));
                construct.insert_at_back(block, &ctx);
                append_mir_return(&mut ctx, block, vec![]);

                crate::lower_mir_to_llvm(&mut ctx, module).expect("lowering failed");
                let body = kernel_blocks(&ctx, module);
                // Inserts at the carrier slot come from two builds here: the
                // payload's own construction (same index inside the payload
                // struct) and the enum's. Exactly one of them must receive
                // the payload's pointer leaf as an SSA extract — never a
                // constant or a load.
                let carrier_inserts: Vec<_> = find_all::<llvm::InsertValueOp>(&ctx, &body)
                    .into_iter()
                    .filter(|insert| insert.indices(&ctx) == vec![carrier_slot])
                    .collect();
                assert!(
                    !carrier_inserts.is_empty(),
                    "constructing Some must write the carrier slot"
                );
                let mut leaf_into_carrier = 0;
                for insert in &carrier_inserts {
                    let value = insert.get_operation().deref(&ctx).get_operand(1);
                    let Some(def) = value.defining_op() else {
                        continue;
                    };
                    let Some(extract) = Operation::get_op::<llvm::ExtractValueOp>(def, &ctx) else {
                        continue;
                    };
                    assert_eq!(
                        extract.indices(&ctx),
                        pointer_path,
                        "the carrier receives the payload's pointer leaf, not a constant or a load"
                    );
                    leaf_into_carrier += 1;
                }
                assert_eq!(
                    leaf_into_carrier, 1,
                    "exactly one carrier write carries the payload pointer leaf"
                );
            }
        };
        // RopePair: carrier at slot 0, pointer leaf at payload path [0].
        check(0, vec![0], true);
        // Float pair: carrier at slot 2, pointer leaf at payload path [2].
        check(2, vec![2], false);
    }

    /// SetDiscriminant must use the slot map instead of assuming that the tag
    /// is field zero, and its GEP must retain the source pointer's GPU address
    /// space. This shape puts an i64 payload first and an i8 tag above the
    /// u32 range, proving the byte GEP does not truncate offsets to u32.
    #[test]
    fn set_discriminant_uses_tag_slot_and_preserves_shared_address_space() {
        use llvm_export::ops::GepIndex;
        use llvm_export::types::{PointerType, address_space};

        let mut ctx = make_ctx();
        let discr_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let payload_a: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let payload_b: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let tag_offset = u64::from(u32::MAX) + 1;
        let enum_ty: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "TagAfterPayload".to_string(),
            discr_ty,
            vec![3, 7],
            vec![
                EnumVariant::new_with_layout("A".to_string(), vec![payload_a], vec![0], vec![8]),
                EnumVariant::new_with_layout("B".to_string(), vec![payload_b], vec![0], vec![8]),
            ],
            tag_offset,
            tag_offset + 8,
            8,
        )
        .into();
        let ptr_ty: TypeHandle = MirPtrType::get_shared(&mut ctx, enum_ty, true).into();

        let (module_ptr, block) = build_kernel(&mut ctx, vec![ptr_ty], vec![]);
        let enum_ptr = block.deref(&ctx).get_argument(0);
        let set = Operation::new(
            &mut ctx,
            mir::MirSetDiscriminantOp::get_concrete_op_info(),
            vec![],
            vec![enum_ptr],
            vec![],
            0,
        );
        let set_op = mir::MirSetDiscriminantOp::new(set);
        set_op.set_attr_set_discriminant_variant_index(&ctx, VariantIndexAttr(1));
        set_op.set_attr_set_discriminant_enum_ty(
            &ctx,
            pliron::builtin::attributes::TypeAttr::new(enum_ty),
        );
        set.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        assert_eq!(count_ops::<mir::MirSetDiscriminantOp>(&ctx, &body), 0);
        assert_eq!(count_ops::<llvm::GetElementPtrOp>(&ctx, &body), 1);
        assert_eq!(count_ops::<llvm::StoreOp>(&ctx, &body), 1);

        let gep = find_first::<llvm::GetElementPtrOp>(&ctx, &body).unwrap();
        let indices = gep.indices(&ctx);
        assert!(matches!(indices.as_slice(), [GepIndex::Value(_)]));
        let GepIndex::Value(offset) = indices[0] else {
            unreachable!()
        };
        let offset_def = offset.defining_op().expect("byte offset must be constant");
        let offset_constant = Operation::get_op::<llvm::ConstantOp>(offset_def, &ctx)
            .expect("byte offset must be an LLVM constant");
        let offset_attr = offset_constant.get_value(&ctx);
        assert_eq!(
            (&*offset_attr as &dyn pliron::attribute::Attribute)
                .downcast_ref::<IntegerAttr>()
                .expect("byte offset must be integer")
                .value()
                .to_u64(),
            tag_offset,
            "the write must use rustc's absolute carrier byte offset"
        );
        let gep_result_ty = gep.get_operation().deref(&ctx).get_result(0).get_type(&ctx);
        assert_eq!(
            gep_result_ty
                .deref(&ctx)
                .downcast_ref::<PointerType>()
                .expect("GEP result must be a pointer")
                .address_space(),
            address_space::SHARED,
            "tag GEP must preserve shared address space"
        );

        let store = find_first::<llvm::StoreOp>(&ctx, &body).unwrap();
        let stored_ty = store.get_operand_value(&ctx).get_type(&ctx);
        assert_eq!(
            stored_ty
                .deref(&ctx)
                .downcast_ref::<IntegerType>()
                .expect("stored tag must be an integer")
                .width(),
            8
        );
        assert_eq!(
            store.get_operand_address(&ctx),
            gep.get_operation().deref(&ctx).get_result(0),
            "the store must use the tag GEP result"
        );
    }

    #[test]
    fn set_discriminant_niche_writes_only_tagged_variant_carrier() {
        for (target, expected_stores) in [(0, 1), (1, 0)] {
            let mut ctx = make_ctx();
            let enum_ty = unit_niche_enum(
                &mut ctx,
                (EnumCarrierKind::Integer, 8, 0),
                0,
                0..=0,
                1,
                vec![1, 1],
            );
            let ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, enum_ty, true).into();
            let (module, block) = build_kernel(&mut ctx, vec![ptr_ty], vec![]);
            let ptr = block.deref(&ctx).get_argument(0);
            let set = Operation::new(
                &mut ctx,
                mir::MirSetDiscriminantOp::get_concrete_op_info(),
                vec![],
                vec![ptr],
                vec![],
                0,
            );
            let set_op = mir::MirSetDiscriminantOp::new(set);
            set_op.set_attr_set_discriminant_variant_index(&ctx, VariantIndexAttr(target));
            set_op.set_attr_set_discriminant_enum_ty(
                &ctx,
                pliron::builtin::attributes::TypeAttr::new(enum_ty),
            );
            set.insert_at_back(block, &ctx);
            append_mir_return(&mut ctx, block, vec![]);

            crate::lower_mir_to_llvm(&mut ctx, module).expect("lowering failed");
            let body = kernel_blocks(&ctx, module);
            assert_eq!(count_ops::<llvm::StoreOp>(&ctx, &body), expected_stores);
            assert_eq!(count_ops::<mir::MirSetDiscriminantOp>(&ctx, &body), 0);
        }
    }

    #[test]
    fn shared_pointer_niche_carrier_rejects_target_dependent_width() {
        let mut ctx = make_ctx();
        let enum_ty = unit_niche_enum(
            &mut ctx,
            (EnumCarrierKind::Pointer, 64, 3),
            0,
            0..=0,
            1,
            vec![1, 1],
        );
        let error = build_enum_slot_map(&mut ctx, enum_ty)
            .err()
            .expect("shared pointer carrier must reject");
        assert!(
            error.to_string().contains("target-mode dependent"),
            "{error}"
        );
    }

    #[test]
    fn shared_pointer_niche_payload_round_trips_through_generic_storage() {
        let mut ctx = make_ctx();
        let logical: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signed).into();
        let pointee: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let shared: TypeHandle = MirPtrType::get_shared(&mut ctx, pointee, true).into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "OptionShared".into(),
            logical,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout("Some".into(), vec![shared], vec![0], vec![8]),
            ],
            EnumEncoding {
                tag_offset: 0,
                total_size: 8,
                abi_align: 8,
                layout_kind: EnumLayoutKind::Niche,
                carrier_kind: EnumCarrierKind::Pointer,
                carrier_width: 64,
                carrier_address_space: llvm_types::address_space::GENERIC,
                niche_start: 0,
                niche_variant_start: 0,
                niche_variant_end: 0,
                untagged_variant: 1,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into();

        let (module, block) = build_kernel(&mut ctx, vec![shared], vec![shared]);
        let pointer = block.deref(&ctx).get_argument(0);
        let construct = Operation::new(
            &mut ctx,
            MirConstructEnumOp::get_concrete_op_info(),
            vec![enum_ty],
            vec![pointer],
            vec![],
            0,
        );
        MirConstructEnumOp::new(construct)
            .set_attr_construct_enum_variant_index(&ctx, VariantIndexAttr(1));
        construct.insert_at_back(block, &ctx);
        let option = construct.deref(&ctx).get_result(0);

        let discriminant = Operation::new(
            &mut ctx,
            mir::MirGetDiscriminantOp::get_concrete_op_info(),
            vec![logical],
            vec![option],
            vec![],
            0,
        );
        discriminant.insert_at_back(block, &ctx);

        let payload = Operation::new(
            &mut ctx,
            MirEnumPayloadOp::get_concrete_op_info(),
            vec![shared],
            vec![option],
            vec![],
            0,
        );
        let payload_op = MirEnumPayloadOp::new(payload);
        payload_op.set_attr_payload_variant_index(&ctx, VariantIndexAttr(1));
        payload_op.set_attr_payload_field_index(&ctx, FieldIndexAttr(0));
        payload.insert_at_back(block, &ctx);
        let result = payload.deref(&ctx).get_result(0);
        append_mir_return(&mut ctx, block, vec![result]);

        crate::lower_mir_to_llvm(&mut ctx, module).expect("lowering failed");
        let body = kernel_blocks(&ctx, module);
        assert_eq!(
            count_ops::<llvm::AddrSpaceCastOp>(&ctx, &body),
            2,
            "construction must genericize the pointer and extraction must restore shared space"
        );
        assert_eq!(
            count_ops::<llvm::PtrToIntOp>(&ctx, &body),
            1,
            "niche discrimination must inspect the generic pointer carrier"
        );
        assert_eq!(count_ops::<MirConstructEnumOp>(&ctx, &body), 0);
        assert_eq!(count_ops::<MirEnumPayloadOp>(&ctx, &body), 0);
    }

    #[test]
    fn nested_shared_pointer_payload_round_trips_through_recursive_storage() {
        let mut ctx = make_ctx();
        let logical: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let pointee: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let shared: TypeHandle = MirPtrType::get_shared(&mut ctx, pointee, true).into();
        let inner: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "SharedPointerInner".into(),
            vec!["pointer".into(), "cookie".into()],
            vec![shared, logical],
            vec![0, 1],
            vec![0, 8],
            16,
            8,
        )
        .into();
        let outer: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "SharedPointerOuter".into(),
            vec!["inner".into(), "guard".into()],
            vec![inner, logical],
            vec![0, 1],
            vec![0, 16],
            24,
            8,
        )
        .into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "NestedSharedPointer".into(),
            logical,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout("Some".into(), vec![outer], vec![8], vec![24]),
            ],
            EnumEncoding {
                tag_offset: 0,
                total_size: 32,
                abi_align: 8,
                layout_kind: EnumLayoutKind::Direct,
                carrier_kind: EnumCarrierKind::Integer,
                carrier_width: 32,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into();

        let (module, block) = build_kernel(&mut ctx, vec![outer], vec![outer]);
        let wrapper = block.deref(&ctx).get_argument(0);
        let construct = Operation::new(
            &mut ctx,
            MirConstructEnumOp::get_concrete_op_info(),
            vec![enum_ty],
            vec![wrapper],
            vec![],
            0,
        );
        MirConstructEnumOp::new(construct)
            .set_attr_construct_enum_variant_index(&ctx, VariantIndexAttr(1));
        construct.insert_at_back(block, &ctx);
        let option = construct.deref(&ctx).get_result(0);

        let payload = Operation::new(
            &mut ctx,
            MirEnumPayloadOp::get_concrete_op_info(),
            vec![outer],
            vec![option],
            vec![],
            0,
        );
        let payload_op = MirEnumPayloadOp::new(payload);
        payload_op.set_attr_payload_variant_index(&ctx, VariantIndexAttr(1));
        payload_op.set_attr_payload_field_index(&ctx, FieldIndexAttr(0));
        payload.insert_at_back(block, &ctx);
        let result = payload.deref(&ctx).get_result(0);
        append_mir_return(&mut ctx, block, vec![result]);

        crate::lower_mir_to_llvm(&mut ctx, module).expect("lowering failed");
        let body = kernel_blocks(&ctx, module);
        assert_eq!(
            count_ops::<llvm::AddrSpaceCastOp>(&ctx, &body),
            2,
            "construction and extraction must cast the nested shared pointer leaf"
        );
        assert_eq!(count_ops::<MirConstructEnumOp>(&ctx, &body), 0);
        assert_eq!(count_ops::<MirEnumPayloadOp>(&ctx, &body), 0);
    }

    #[test]
    fn bounded_shared_pointer_array_niche_payload_round_trips_through_recursive_storage() {
        let mut ctx = make_ctx();
        let logical: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signed).into();
        let pointee: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let shared: TypeHandle = MirPtrType::get_shared(&mut ctx, pointee, true).into();
        let pointers: TypeHandle = MirArrayType::get(&mut ctx, shared, 2).into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "OptionSharedPointerArray".into(),
            logical,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout("Some".into(), vec![pointers], vec![0], vec![16]),
            ],
            EnumEncoding {
                tag_offset: 0,
                total_size: 16,
                abi_align: 8,
                layout_kind: EnumLayoutKind::Niche,
                carrier_kind: EnumCarrierKind::Pointer,
                carrier_width: 64,
                carrier_address_space: llvm_types::address_space::GENERIC,
                niche_start: 0,
                niche_variant_start: 0,
                niche_variant_end: 0,
                untagged_variant: 1,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into();

        let (module, block) = build_kernel(&mut ctx, vec![pointers], vec![pointers]);
        let array = block.deref(&ctx).get_argument(0);
        let construct = Operation::new(
            &mut ctx,
            MirConstructEnumOp::get_concrete_op_info(),
            vec![enum_ty],
            vec![array],
            vec![],
            0,
        );
        MirConstructEnumOp::new(construct)
            .set_attr_construct_enum_variant_index(&ctx, VariantIndexAttr(1));
        construct.insert_at_back(block, &ctx);
        let option = construct.deref(&ctx).get_result(0);

        let discriminant = Operation::new(
            &mut ctx,
            mir::MirGetDiscriminantOp::get_concrete_op_info(),
            vec![logical],
            vec![option],
            vec![],
            0,
        );
        discriminant.insert_at_back(block, &ctx);

        let payload = Operation::new(
            &mut ctx,
            MirEnumPayloadOp::get_concrete_op_info(),
            vec![pointers],
            vec![option],
            vec![],
            0,
        );
        let payload_op = MirEnumPayloadOp::new(payload);
        payload_op.set_attr_payload_variant_index(&ctx, VariantIndexAttr(1));
        payload_op.set_attr_payload_field_index(&ctx, FieldIndexAttr(0));
        payload.insert_at_back(block, &ctx);
        let result = payload.deref(&ctx).get_result(0);
        append_mir_return(&mut ctx, block, vec![result]);

        crate::lower_mir_to_llvm(&mut ctx, module).expect("lowering failed");
        let body = kernel_blocks(&ctx, module);
        assert_eq!(
            count_ops::<llvm::AddrSpaceCastOp>(&ctx, &body),
            4,
            "construction and extraction must cast both shared-pointer array elements"
        );
        assert_eq!(
            count_ops::<llvm::PtrToIntOp>(&ctx, &body),
            1,
            "niche discrimination must inspect the generic first-pointer carrier"
        );
        assert_eq!(count_ops::<MirConstructEnumOp>(&ctx, &body), 0);
        assert_eq!(count_ops::<MirEnumPayloadOp>(&ctx, &body), 0);
    }

    #[test]
    fn option_bool_uses_i8_carrier_and_spills_i1_payload() {
        let mut ctx = make_ctx();
        let logical: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let bool_ty: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "OptionBool".into(),
            logical,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout("Some".into(), vec![bool_ty], vec![0], vec![1]),
            ],
            EnumEncoding {
                tag_offset: 0,
                total_size: 1,
                abi_align: 1,
                layout_kind: EnumLayoutKind::Niche,
                carrier_kind: EnumCarrierKind::Integer,
                carrier_width: 8,
                niche_start: 2,
                untagged_variant: 1,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into();
        let slot_map = build_enum_slot_map(&mut ctx, enum_ty).unwrap();
        assert_eq!(
            slot_map
                .carrier_llvm_ty
                .unwrap()
                .deref(&ctx)
                .downcast_ref::<IntegerType>()
                .unwrap()
                .width(),
            8,
            "Option<bool>'s memory carrier is i8, not bool's semantic i1"
        );
        assert_eq!(slot_map.field_slots, vec![None]);

        let (module, block) = build_kernel(&mut ctx, vec![bool_ty], vec![enum_ty]);
        let payload = block.deref(&ctx).get_argument(0);
        let construct = Operation::new(
            &mut ctx,
            mir::MirConstructEnumOp::get_concrete_op_info(),
            vec![enum_ty],
            vec![payload],
            vec![],
            0,
        );
        mir::MirConstructEnumOp::new(construct)
            .set_attr_construct_enum_variant_index(&ctx, VariantIndexAttr(1));
        construct.insert_at_back(block, &ctx);
        let result = construct.deref(&ctx).get_result(0);
        append_mir_return(&mut ctx, block, vec![result]);

        crate::lower_mir_to_llvm(&mut ctx, module).expect("lowering failed");
        let body = kernel_blocks(&ctx, module);
        assert_eq!(count_ops::<llvm::InsertValueOp>(&ctx, &body), 0);
        assert!(find_all::<llvm::ZExtOp>(&ctx, &body).iter().any(|zext| {
            zext.get_operation()
                .deref(&ctx)
                .get_operand(0)
                .get_type(&ctx)
                .deref(&ctx)
                .downcast_ref::<IntegerType>()
                .is_some_and(|integer| integer.width() == 1)
                && zext
                    .get_operation()
                    .deref(&ctx)
                    .get_result(0)
                    .get_type(&ctx)
                    .deref(&ctx)
                    .downcast_ref::<IntegerType>()
                    .is_some_and(|integer| integer.width() == 8)
        }));
        assert!(find_all::<llvm::StoreOp>(&ctx, &body).iter().any(|store| {
            store
                .get_operand_value(&ctx)
                .get_type(&ctx)
                .deref(&ctx)
                .downcast_ref::<IntegerType>()
                .is_some_and(|integer| integer.width() == 8)
        }));
    }

    #[test]
    fn direct_bool_uses_i8_storage_and_spills_i1_payload() {
        let mut ctx = make_ctx();
        let tag: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let bool_ty: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let enum_ty: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "DirectBool".into(),
            tag,
            vec![0, 1],
            vec![
                EnumVariant::new_with_layout("A".into(), vec![bool_ty], vec![4], vec![1]),
                EnumVariant::unit("B".into()),
            ],
            0,
            8,
            4,
        )
        .into();
        let slot_map = build_enum_slot_map(&mut ctx, enum_ty).unwrap();
        assert_eq!(slot_map.field_slots, vec![None]);
        let storage_fields = slot_map
            .llvm_struct_ty
            .deref(&ctx)
            .downcast_ref::<llvm_types::StructType>()
            .expect("enum storage must be an LLVM struct")
            .fields()
            .collect::<Vec<_>>();
        assert_eq!(
            storage_fields[1]
                .deref(&ctx)
                .downcast_ref::<IntegerType>()
                .map(IntegerType::width),
            Some(8),
            "the standalone Rust bool byte must use physical i8 storage"
        );

        let (module, block) = build_kernel(&mut ctx, vec![bool_ty], vec![enum_ty]);
        let payload = block.deref(&ctx).get_argument(0);
        let construct = Operation::new(
            &mut ctx,
            mir::MirConstructEnumOp::get_concrete_op_info(),
            vec![enum_ty],
            vec![payload],
            vec![],
            0,
        );
        mir::MirConstructEnumOp::new(construct)
            .set_attr_construct_enum_variant_index(&ctx, VariantIndexAttr(0));
        construct.insert_at_back(block, &ctx);
        let result = construct.deref(&ctx).get_result(0);
        append_mir_return(&mut ctx, block, vec![result]);

        crate::lower_mir_to_llvm(&mut ctx, module).expect("lowering failed");
        let body = kernel_blocks(&ctx, module);
        assert_eq!(
            count_ops::<llvm::InsertValueOp>(&ctx, &body),
            1,
            "only the direct tag should be inserted as an SSA struct field"
        );
        assert!(find_all::<llvm::ZExtOp>(&ctx, &body).iter().any(|zext| {
            zext.get_operation()
                .deref(&ctx)
                .get_operand(0)
                .get_type(&ctx)
                .deref(&ctx)
                .downcast_ref::<IntegerType>()
                .is_some_and(|integer| integer.width() == 1)
                && zext
                    .get_operation()
                    .deref(&ctx)
                    .get_result(0)
                    .get_type(&ctx)
                    .deref(&ctx)
                    .downcast_ref::<IntegerType>()
                    .is_some_and(|integer| integer.width() == 8)
        }));
        assert!(find_all::<llvm::StoreOp>(&ctx, &body).iter().any(|store| {
            store
                .get_operand_value(&ctx)
                .get_type(&ctx)
                .deref(&ctx)
                .downcast_ref::<IntegerType>()
                .is_some_and(|integer| integer.width() == 8)
        }));
    }

    #[test]
    fn later_field_niche_set_writes_exact_carrier_offset() {
        use llvm_export::ops::GepIndex;

        let mut ctx = make_ctx();
        let logical: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let wrapper: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "Wrapper".into(),
            vec!["pad".into(), "nz".into()],
            vec![u32_ty, u32_ty],
            vec![0, 1],
            vec![0, 4],
            8,
            4,
        )
        .into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "MaybeWrapper".into(),
            logical,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout("Some".into(), vec![wrapper], vec![0], vec![8]),
            ],
            EnumEncoding {
                tag_offset: 4,
                total_size: 8,
                abi_align: 4,
                layout_kind: EnumLayoutKind::Niche,
                carrier_kind: EnumCarrierKind::Integer,
                carrier_width: 32,
                untagged_variant: 1,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into();
        let ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, enum_ty, true).into();
        let (module, block) = build_kernel(&mut ctx, vec![ptr_ty], vec![]);
        let ptr = block.deref(&ctx).get_argument(0);
        let set = Operation::new(
            &mut ctx,
            mir::MirSetDiscriminantOp::get_concrete_op_info(),
            vec![],
            vec![ptr],
            vec![],
            0,
        );
        let set_op = mir::MirSetDiscriminantOp::new(set);
        set_op.set_attr_set_discriminant_variant_index(&ctx, VariantIndexAttr(0));
        set_op.set_attr_set_discriminant_enum_ty(
            &ctx,
            pliron::builtin::attributes::TypeAttr::new(enum_ty),
        );
        set.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module).expect("lowering failed");
        let body = kernel_blocks(&ctx, module);
        let gep = find_first::<llvm::GetElementPtrOp>(&ctx, &body).unwrap();
        let indices = gep.indices(&ctx);
        let [GepIndex::Value(offset)] = indices.as_slice() else {
            panic!("carrier access must use a byte-offset SSA value");
        };
        let constant =
            Operation::get_op::<llvm::ConstantOp>(offset.defining_op().unwrap(), &ctx).unwrap();
        let attr = constant.get_value(&ctx);
        assert_eq!(
            (&*attr as &dyn pliron::attribute::Attribute)
                .downcast_ref::<IntegerAttr>()
                .unwrap()
                .value()
                .to_u64(),
            4
        );
    }

    #[test]
    fn get_niche_discriminant_adds_large_range_start_at_logical_width() {
        let mut ctx = make_ctx();
        let logical_ty: TypeHandle = IntegerType::get(&ctx, 16, Signedness::Unsigned).into();
        let carrier_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let mut variants = (0..300)
            .map(|index| EnumVariant::unit(format!("V{index}")))
            .collect::<Vec<_>>();
        // A valid niche layout stores the carrier inside the untagged
        // variant's payload. Keep the large logical-variant test realistic
        // instead of relying on a carrier with no backing field.
        variants[0] = EnumVariant::new_with_layout("V0".into(), vec![carrier_ty], vec![0], vec![1]);
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "LargeNiche".into(),
            logical_ty,
            (0..300).collect(),
            variants,
            EnumEncoding {
                total_size: 1,
                abi_align: 1,
                layout_kind: EnumLayoutKind::Niche,
                carrier_kind: EnumCarrierKind::Integer,
                carrier_width: 8,
                niche_variant_start: 298,
                niche_variant_end: 299,
                untagged_variant: 0,
                variant_inhabited: {
                    let mut inhabited = vec![0; 300];
                    inhabited[0] = 1;
                    inhabited[298] = 1;
                    inhabited[299] = 1;
                    inhabited
                },
                ..EnumEncoding::default()
            },
        )
        .into();
        let (module, block) = build_kernel(&mut ctx, vec![enum_ty], vec![logical_ty]);
        let value = block.deref(&ctx).get_argument(0);
        let get = Operation::new(
            &mut ctx,
            mir::MirGetDiscriminantOp::get_concrete_op_info(),
            vec![logical_ty],
            vec![value],
            vec![],
            0,
        );
        get.insert_at_back(block, &ctx);
        let result = get.deref(&ctx).get_result(0);
        append_mir_return(&mut ctx, block, vec![result]);

        crate::lower_mir_to_llvm(&mut ctx, module).expect("lowering failed");
        let body = kernel_blocks(&ctx, module);
        assert_eq!(count_ops::<llvm::ZExtOp>(&ctx, &body), 1);
        let add = find_first::<llvm::AddOp>(&ctx, &body).expect("logical variant add");
        for operand in add.get_operation().deref(&ctx).operands() {
            assert_eq!(
                operand
                    .get_type(&ctx)
                    .deref(&ctx)
                    .downcast_ref::<IntegerType>()
                    .unwrap()
                    .width(),
                16,
                "variant-index arithmetic must occur at logical width"
            );
        }
    }

    #[test]
    fn direct_negative_discriminant_read_remains_signed_for_widening() {
        let mut ctx = make_ctx();
        let i8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Signed).into();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signed).into();
        let enum_ty: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "Negative".into(),
            i8_ty,
            vec![255, 0],
            vec![EnumVariant::unit("N".into()), EnumVariant::unit("Z".into())],
            0,
            1,
            1,
        )
        .into();
        let (module, block) = build_kernel(&mut ctx, vec![], vec![i32_ty]);
        let construct = Operation::new(
            &mut ctx,
            mir::MirConstructEnumOp::get_concrete_op_info(),
            vec![enum_ty],
            vec![],
            vec![],
            0,
        );
        mir::MirConstructEnumOp::new(construct)
            .set_attr_construct_enum_variant_index(&ctx, VariantIndexAttr(0));
        construct.insert_at_back(block, &ctx);
        let enum_value = construct.deref(&ctx).get_result(0);
        let get = Operation::new(
            &mut ctx,
            mir::MirGetDiscriminantOp::get_concrete_op_info(),
            vec![i8_ty],
            vec![enum_value],
            vec![],
            0,
        );
        get.insert_at_back(block, &ctx);
        let discriminant = get.deref(&ctx).get_result(0);
        let cast = Operation::new(
            &mut ctx,
            mir::MirCastOp::get_concrete_op_info(),
            vec![i32_ty],
            vec![discriminant],
            vec![],
            0,
        );
        mir::MirCastOp::new(cast).set_attr_cast_kind(&ctx, MirCastKindAttr::IntToInt);
        cast.insert_at_back(block, &ctx);
        let widened = cast.deref(&ctx).get_result(0);
        append_mir_return(&mut ctx, block, vec![widened]);

        crate::lower_mir_to_llvm(&mut ctx, module).expect("lowering failed");
        let body = kernel_blocks(&ctx, module);
        assert_eq!(count_ops::<llvm::SExtOp>(&ctx, &body), 1);
        assert_eq!(count_ops::<llvm::ZExtOp>(&ctx, &body), 0);
    }

    #[test]
    fn single_layout_preserves_large_and_negative_declared_discriminants() {
        for (width, signedness, bits, widened_width, expects_sext) in [
            (16, Signedness::Unsigned, 1_000, 16, false),
            (8, Signedness::Signed, 251, 32, true),
        ] {
            let mut ctx = make_ctx();
            let logical: TypeHandle = IntegerType::get(&ctx, width, signedness).into();
            let destination: TypeHandle = IntegerType::get(&ctx, widened_width, signedness).into();
            let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
                &mut ctx,
                "Single".into(),
                logical,
                vec![bits],
                vec![EnumVariant::unit("Only".into())],
                EnumEncoding {
                    tag_offset: 0,
                    total_size: 0,
                    abi_align: 1,
                    layout_kind: EnumLayoutKind::Single,
                    variant_inhabited: vec![1],
                    ..EnumEncoding::default()
                },
            )
            .into();
            let (module, block) = build_kernel(&mut ctx, vec![], vec![destination]);
            let construct = Operation::new(
                &mut ctx,
                mir::MirConstructEnumOp::get_concrete_op_info(),
                vec![enum_ty],
                vec![],
                vec![],
                0,
            );
            mir::MirConstructEnumOp::new(construct)
                .set_attr_construct_enum_variant_index(&ctx, VariantIndexAttr(0));
            construct.insert_at_back(block, &ctx);
            let enum_value = construct.deref(&ctx).get_result(0);
            let get = Operation::new(
                &mut ctx,
                mir::MirGetDiscriminantOp::get_concrete_op_info(),
                vec![logical],
                vec![enum_value],
                vec![],
                0,
            );
            get.insert_at_back(block, &ctx);
            let discr = get.deref(&ctx).get_result(0);
            let returned = if widened_width == width {
                discr
            } else {
                let cast = Operation::new(
                    &mut ctx,
                    mir::MirCastOp::get_concrete_op_info(),
                    vec![destination],
                    vec![discr],
                    vec![],
                    0,
                );
                mir::MirCastOp::new(cast).set_attr_cast_kind(&ctx, MirCastKindAttr::IntToInt);
                cast.insert_at_back(block, &ctx);
                cast.deref(&ctx).get_result(0)
            };
            append_mir_return(&mut ctx, block, vec![returned]);

            crate::lower_mir_to_llvm(&mut ctx, module).expect("lowering failed");
            let body = kernel_blocks(&ctx, module);
            let found_bits = find_all::<llvm::ConstantOp>(&ctx, &body)
                .iter()
                .any(|constant| {
                    let attr = constant.get_value(&ctx);
                    (&*attr as &dyn pliron::attribute::Attribute)
                        .downcast_ref::<IntegerAttr>()
                        .is_some_and(|value| value.value().to_u64() == bits)
                });
            assert!(
                found_bits,
                "single discriminant {bits} must not be truncated"
            );
            assert_eq!(count_ops::<llvm::SExtOp>(&ctx, &body) == 1, expects_sext);
        }
    }
}
