/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::common::emit_integer_constant;
use crate::convert::enum_payload_storage::coerce_enum_payload_value;
use crate::convert::types::{llvm_byte_faithful_twin, llvm_type_contains_i1};
use dialect_mir::types::{EnumCarrierKind, EnumLayoutKind, MirEnumType};
use llvm_export::op_interfaces::{CastOpInterface, CastOpWithNNegInterface};
use llvm_export::ops as llvm;
use llvm_export::types as llvm_types;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::Context;
use pliron::irbuild::dialect_conversion::DialectConversionRewriter;
use pliron::irbuild::inserter::Inserter;
use pliron::op::Op;
use pliron::result::Result;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;

/// The physical carrier write required to select one source variant.
/// `None` is a real semantic result for the untagged niche variant and for
/// rustc's single inhabited variant; it must never be replaced by a guessed
/// memory write.
pub(super) fn enum_carrier_bits_for_variant(
    enum_ty: &MirEnumType,
    variant: usize,
) -> std::result::Result<Option<u128>, String> {
    if variant >= enum_ty.variant_count() || enum_ty.variant_is_inhabited(variant) != Some(true) {
        return Err(format!(
            "cannot select uninhabited or missing variant {} of '{}'",
            variant,
            enum_ty.name()
        ));
    }

    match enum_ty.layout_kind {
        EnumLayoutKind::Direct => enum_ty
            .variant_discriminants
            .get(variant)
            .copied()
            .map(|bits| Some(u128::from(bits)))
            .ok_or_else(|| format!("variant {} has no declared discriminant", variant)),
        EnumLayoutKind::Niche => {
            // This check deliberately comes before the encoded range check:
            // rustc permits the untagged variant index to lie inside that
            // range. Its range position is a dead niche value; selecting the
            // actual untagged variant remains a no-op.
            if variant == enum_ty.untagged_variant as usize {
                return Ok(None);
            }
            if !(enum_ty.niche_variant_start as usize..=enum_ty.niche_variant_end as usize)
                .contains(&variant)
            {
                return Err(format!(
                    "inhabited variant {} is not representable by niche layout of '{}'",
                    variant,
                    enum_ty.name()
                ));
            }
            let offset = (variant as u128) - u128::from(enum_ty.niche_variant_start);
            let mut bits = enum_ty.niche_start().wrapping_add(offset);
            if enum_ty.carrier_width < 128 {
                bits &= (1u128 << enum_ty.carrier_width) - 1;
            }
            Ok(Some(bits))
        }
        EnumLayoutKind::Single if variant == enum_ty.single_variant as usize => Ok(None),
        EnumLayoutKind::Single => Err(format!(
            "variant {} is not the single inhabited variant of '{}'",
            variant,
            enum_ty.name()
        )),
        EnumLayoutKind::Empty => Err(format!("enum '{}' is uninhabited", enum_ty.name())),
        EnumLayoutKind::Unknown => Err(format!(
            "enum '{}' has unknown physical layout",
            enum_ty.name()
        )),
    }
}

pub(super) fn emit_carrier_constant(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    enum_ty: &MirEnumType,
    carrier_ty: TypeHandle,
    bits: u128,
) -> Result<Value> {
    let integer = emit_integer_constant(ctx, rewriter, enum_ty.carrier_width, bits);
    match enum_ty.carrier_kind {
        EnumCarrierKind::Integer => Ok(integer),
        EnumCarrierKind::Pointer => {
            let cast = llvm::IntToPtrOp::new(ctx, integer, carrier_ty);
            rewriter.insert_operation(ctx, cast.get_operation());
            Ok(cast.get_operation().deref(ctx).get_result(0))
        }
        _ => pliron::input_err_noloc!("enum carrier constant requested without a carrier"),
    }
}

/// Convert a value to its byte-faithful storage twin: every `i1` leaf is
/// zero-extended to its canonical `i8` memory byte, recursively through
/// structs and arrays. Values without `i1` storage pass through unchanged.
///
/// This is the value-level half of [`llvm_byte_faithful_twin`]: the enum
/// slot map claims twin-typed storage for bool-bearing payloads, and this
/// produces the twin-typed value the store into that storage needs.
pub(super) fn canonicalize_bool_value_bytes(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    value: Value,
) -> Result<Value> {
    let ty = value.get_type(ctx);
    if !llvm_type_contains_i1(ctx, ty) {
        return Ok(value);
    }
    let is_scalar_i1 = ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| integer.width() == 1);
    if is_scalar_i1 {
        let byte_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Signless).into();
        let zext = llvm::ZExtOp::new_with_nneg(ctx, value, byte_ty, false);
        rewriter.insert_operation(ctx, zext.get_operation());
        return Ok(zext.get_operation().deref(ctx).get_result(0));
    }
    let Some(twin) = llvm_byte_faithful_twin(ctx, ty) else {
        return pliron::input_err_noloc!(
            "enum construction: bool storage in this value's shape cannot be canonicalized"
        );
    };
    let element_count = {
        let ty_ref = ty.deref(ctx);
        if let Some(struct_ty) = ty_ref.downcast_ref::<llvm_types::StructType>() {
            struct_ty.fields().count() as u64
        } else if let Some(array_ty) = ty_ref.downcast_ref::<llvm_types::ArrayType>() {
            array_ty.size()
        } else {
            return pliron::input_err_noloc!(
                "enum construction: unexpected container for bool storage canonicalization"
            );
        }
    };
    let undef_op = llvm::UndefOp::new(ctx, twin);
    rewriter.insert_operation(ctx, undef_op.get_operation());
    let mut current = undef_op.get_operation().deref(ctx).get_result(0);
    for index in 0..element_count {
        let extract_op = llvm::ExtractValueOp::new(ctx, value, vec![index as u32])?;
        rewriter.insert_operation(ctx, extract_op.get_operation());
        let element = extract_op.get_operation().deref(ctx).get_result(0);
        let converted = canonicalize_bool_value_bytes(ctx, rewriter, element)?;
        let insert_op = llvm::InsertValueOp::new(ctx, current, converted, vec![index as u32]);
        rewriter.insert_operation(ctx, insert_op.get_operation());
        current = insert_op.get_operation().deref(ctx).get_result(0);
    }
    Ok(current)
}

/// Adapt a semantic enum payload value to or from its physical storage type.
///
/// Shared pointer leaves are converted through CUDA generic space recursively
/// through struct/tuple payloads. Bool leaves are canonicalized separately on
/// construction and narrowed recursively on extraction.
pub(super) fn coerce_enum_payload_storage(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    value: Value,
    target_ty: TypeHandle,
) -> Result<Value> {
    coerce_enum_payload_value(ctx, rewriter, value, target_ty)
}

#[cfg(test)]
// Tests build kinded fixture types directly; production minting lives in mir-importer's facts.rs.
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;
    use crate::convert::ops::test_util::*;

    use super::super::test_support::*;

    #[test]
    fn niche_encoding_handles_untagged_inside_range_and_u128_wrap() {
        let mut ctx = make_ctx();
        let inside = unit_niche_enum(
            &mut ctx,
            (EnumCarrierKind::Integer, 8, 0),
            42,
            0..=1,
            1,
            vec![1, 1],
        );
        let inside_ref = inside.deref(&ctx);
        let inside_enum = inside_ref.downcast_ref::<MirEnumType>().unwrap();
        assert_eq!(enum_carrier_bits_for_variant(inside_enum, 0), Ok(Some(42)));
        assert_eq!(
            enum_carrier_bits_for_variant(inside_enum, 1),
            Ok(None),
            "the untagged variant is a no-op even when its index is in the niche range"
        );
        drop(inside_ref);

        let wrapping = unit_niche_enum(
            &mut ctx,
            (EnumCarrierKind::Integer, 128, 0),
            u128::MAX,
            0..=1,
            0,
            vec![1, 1],
        );
        let wrapping_ref = wrapping.deref(&ctx);
        let wrapping_enum = wrapping_ref.downcast_ref::<MirEnumType>().unwrap();
        assert_eq!(
            enum_carrier_bits_for_variant(wrapping_enum, 1),
            Ok(Some(0)),
            "niche arithmetic must wrap across the full u128 carrier"
        );
    }
}
