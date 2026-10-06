/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Enum constant and niche decoding.

use super::coerce::cast_enum_fields_to_expected_types;
use super::const_alloc::{
    alloc_slice_bytes_zeroing_uninit, audit_aggregate_relocations,
    relocation_offsets_overlapping_range, translate_constant_value_from_alloc,
};
use super::const_bytes::{rust_type_layout_size, translate_constant_value_from_bytes};
use super::promoted::constant_allocation;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::layout::{enum_tag_offset, enum_variant_field_offsets};
use dialect_mir::ops::MirConstructEnumOp;
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::TypeHandle;
use pliron::value::Value;
use pliron::{input_err, input_error, input_error_noloc};
use rustc_public::CrateDefType;
use rustc_public::mir;
use rustc_public_bridge::IndexedVal;

/// Translate an enum constant by reconstructing both its active variant and any
/// payload operands from the constant's allocation.
///
/// Pointer-bearing enums must retain the outer allocation: the bytes stored in a
/// rustc relocation slot are only an addend, while the provenance map identifies
/// the target allocation. Following the first relocation here would replace the
/// enum's storage image with its pointee and lose both the tag and other fields.
pub(super) fn translate_enum_constant(
    ctx: &mut Context,
    constant: &mir::ConstOperand,
    rust_ty: &rustc_public::ty::Ty,
    const_ty_ptr: TypeHandle,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    if let Some(allocation) = constant_allocation(constant) {
        return translate_enum_constant_from_alloc(
            ctx,
            allocation,
            0,
            rust_ty,
            const_ty_ptr,
            block_ptr,
            prev_op,
            loc,
        );
    }

    let enum_size = rust_type_layout_size(*rust_ty, loc.clone())?;
    if enum_size != 0 {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Enum constant of {enum_size} byte(s) must be backed by an allocation, found {:?}",
                constant.const_.kind()
            ))
        );
    }

    translate_enum_constant_from_bytes(ctx, rust_ty, const_ty_ptr, &[], block_ptr, prev_op, loc)
}

/// Translate an enum constant while retaining rustc's provenance map.
pub(super) fn translate_enum_constant_from_alloc(
    ctx: &mut Context,
    alloc: &rustc_public::ty::Allocation,
    base_offset: usize,
    rust_ty: &rustc_public::ty::Ty,
    const_ty_ptr: TypeHandle,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    let enum_size = rust_type_layout_size(*rust_ty, loc.clone())?;
    let enum_bytes =
        alloc_slice_bytes_zeroing_uninit(alloc, base_offset, enum_size, "Enum constant", &loc)?;

    translate_enum_constant_from_storage(
        ctx,
        rust_ty,
        const_ty_ptr,
        &enum_bytes,
        Some((alloc, base_offset)),
        block_ptr,
        prev_op,
        loc,
    )
}

/// Translate an enum value from raw bytes plus the Rust type/layout metadata.
pub(super) fn translate_enum_constant_from_bytes(
    ctx: &mut Context,
    rust_ty: &rustc_public::ty::Ty,
    const_ty_ptr: TypeHandle,
    enum_bytes: &[u8],
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    translate_enum_constant_from_storage(
        ctx,
        rust_ty,
        const_ty_ptr,
        enum_bytes,
        None,
        block_ptr,
        prev_op,
        loc,
    )
}

/// Shared enum decoder. When `allocation` is present, direct thin-pointer
/// fields are reconstructed from rustc relocations and niche selection can
/// distinguish a relocated pointer from the all-zero placeholder bytes.
#[allow(clippy::too_many_arguments)]
fn translate_enum_constant_from_storage(
    ctx: &mut Context,
    rust_ty: &rustc_public::ty::Ty,
    const_ty_ptr: TypeHandle,
    enum_bytes: &[u8],
    allocation: Option<(&rustc_public::ty::Allocation, usize)>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    let enum_variant = {
        let ty_obj = const_ty_ptr.deref(ctx);
        let enum_ty = ty_obj
            .downcast_ref::<dialect_mir::types::MirEnumType>()
            .ok_or_else(|| {
                input_error_noloc!(TranslationErr::unsupported(
                    "translate_enum_constant_from_storage called on non-enum type"
                ))
            })?;

        let variant_index = match allocation {
            Some((alloc, base_offset)) => {
                enum_variant_index_from_alloc(rust_ty, enum_bytes, alloc, base_offset, loc.clone())?
            }
            None => enum_variant_index_from_bytes(rust_ty, enum_bytes, loc.clone())?,
        };
        let variant = enum_ty.get_variant(variant_index).ok_or_else(|| {
            input_error_noloc!(TranslationErr::unsupported(format!(
                "Enum constant resolved to variant index {} outside translated MIR enum '{}'",
                variant_index,
                enum_ty.name()
            )))
        })?;
        (variant_index, variant)
    };
    let variant_index = enum_variant.0;
    let variant = enum_variant.1;

    let mut field_values = Vec::with_capacity(variant.field_types.len());
    let mut field_ranges = Vec::with_capacity(variant.field_types.len());
    let mut current_prev_op = prev_op;

    if !variant.field_types.is_empty() {
        use rustc_public::ty::{RigidTy, TyKind};

        let layout = rust_ty.layout().map_err(|e| {
            input_error_noloc!(TranslationErr::unsupported(format!(
                "Failed to query layout for enum constant: {:?}",
                e
            )))
        })?;
        let field_offsets =
            enum_variant_field_offsets(&layout.shape(), variant_index, loc.clone())?;

        let (adt_def, substs) = match rust_ty.kind() {
            TyKind::RigidTy(RigidTy::Adt(adt_def, substs)) => (adt_def, substs),
            other => {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Expected ADT Rust type for enum constant, got {:?}",
                        other
                    ))
                );
            }
        };
        let rust_variant = &adt_def.variants()[variant_index];

        for (field_idx, field_ty_ptr) in variant.field_types.iter().copied().enumerate() {
            let rust_field_ty = rust_variant.fields()[field_idx].ty_with_args(&substs);
            let field_layout = rust_field_ty.layout().map_err(|e| {
                input_error_noloc!(TranslationErr::unsupported(format!(
                    "Failed to query layout for enum field {} of variant '{}': {:?}",
                    field_idx,
                    rust_variant.name(),
                    e
                )))
            })?;
            let field_offset = *field_offsets.get(field_idx).ok_or_else(|| {
                input_error_noloc!(TranslationErr::unsupported(format!(
                    "Missing layout offset for enum field {} of variant '{}'",
                    field_idx,
                    rust_variant.name()
                )))
            })?;
            let field_size = field_layout.shape().size.bytes() as usize;
            let field_end = field_offset.checked_add(field_size).ok_or_else(|| {
                input_error_noloc!(TranslationErr::unsupported(format!(
                    "Enum field {} of variant '{}' overflowed offset computation",
                    field_idx,
                    rust_variant.name()
                )))
            })?;

            if field_end > enum_bytes.len() {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Enum constant for variant '{}' has {} bytes, but field {} needs [{}..{})",
                        rust_variant.name(),
                        enum_bytes.len(),
                        field_idx,
                        field_offset,
                        field_end
                    ))
                );
            }

            let (field_val, new_prev_op) = match allocation {
                Some((alloc, base_offset)) => {
                    let absolute_field_offset =
                        base_offset.checked_add(field_offset).ok_or_else(|| {
                            input_error_noloc!(TranslationErr::unsupported(format!(
                                "Enum field {} of variant '{}' overflowed absolute offset computation",
                                field_idx,
                                rust_variant.name()
                            )))
                        })?;
                    field_ranges.push((absolute_field_offset, field_size));
                    translate_constant_value_from_alloc(
                        ctx,
                        alloc,
                        absolute_field_offset,
                        &rust_field_ty,
                        field_ty_ptr,
                        block_ptr,
                        current_prev_op,
                        loc.clone(),
                    )?
                }
                None => {
                    let field_bytes = &enum_bytes[field_offset..field_end];
                    translate_constant_value_from_bytes(
                        ctx,
                        &rust_field_ty,
                        field_ty_ptr,
                        field_bytes,
                        block_ptr,
                        current_prev_op,
                        loc.clone(),
                    )?
                }
            };
            field_values.push(field_val);
            current_prev_op = new_prev_op;
        }

        let (casted_field_values, prev_after_casts) = cast_enum_fields_to_expected_types(
            ctx,
            field_values,
            const_ty_ptr,
            variant_index,
            block_ptr,
            current_prev_op,
            loc.clone(),
        );
        field_values = casted_field_values;
        current_prev_op = prev_after_casts;
    }

    if let Some((alloc, base_offset)) = allocation {
        let enum_size = rust_type_layout_size(*rust_ty, loc.clone())?;
        audit_aggregate_relocations(alloc, base_offset, enum_size, &field_ranges, "Enum", &loc)?;
    }

    let op = Operation::new(
        ctx,
        MirConstructEnumOp::get_concrete_op_info(),
        vec![const_ty_ptr],
        field_values,
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());

    let enum_op = MirConstructEnumOp::new(op);
    enum_op.set_attr_construct_enum_variant_index(
        ctx,
        dialect_mir::attributes::VariantIndexAttr(variant_index as u32),
    );

    if let Some(prev) = current_prev_op {
        enum_op.get_operation().insert_after(ctx, prev);
    } else {
        enum_op.get_operation().insert_at_front(block_ptr, ctx);
    }

    let val = enum_op.get_operation().deref(ctx).get_result(0);

    Ok((val, Some(enum_op.get_operation())))
}

/// Determine the active enum variant from layout metadata plus raw bytes.
fn decode_niche_variant_index(
    tag_value: u128,
    carrier_mask: u128,
    niche_start: u128,
    niche_variant_start: usize,
    niche_variant_end: usize,
    untagged_variant: usize,
) -> usize {
    let relative = tag_value.wrapping_sub(niche_start) & carrier_mask;
    let span = (niche_variant_end - niche_variant_start) as u128;

    // Compare at the full physical carrier width. Converting `relative` to
    // host usize before this check can turn 2^64 into zero on a 64-bit host
    // and select the wrong variant for an i128 carrier.
    if relative <= span {
        niche_variant_start + relative as usize
    } else {
        untagged_variant
    }
}

fn enum_variant_index_from_bytes(
    rust_ty: &rustc_public::ty::Ty,
    enum_bytes: &[u8],
    loc: Location,
) -> TranslationResult<usize> {
    enum_variant_index_from_storage(rust_ty, enum_bytes, None, loc)
}

fn enum_variant_index_from_alloc(
    rust_ty: &rustc_public::ty::Ty,
    enum_bytes: &[u8],
    alloc: &rustc_public::ty::Allocation,
    base_offset: usize,
    loc: Location,
) -> TranslationResult<usize> {
    enum_variant_index_from_storage(rust_ty, enum_bytes, Some((alloc, base_offset)), loc)
}

/// Determine the active enum variant from layout metadata, raw bytes, and
/// optionally the provenance map of the enclosing allocation.
///
/// A niche-encoded pointer enum such as `Option<&T>` stores its carrier in the
/// pointer word itself. rustc leaves addend bytes in that word and records the
/// target separately as a relocation, so all-zero bytes do not mean the niche
/// variant when a full-width relocation covers the carrier.
fn enum_variant_index_from_storage(
    rust_ty: &rustc_public::ty::Ty,
    enum_bytes: &[u8],
    allocation: Option<(&rustc_public::ty::Allocation, usize)>,
    loc: Location,
) -> TranslationResult<usize> {
    let layout = rust_ty.layout().map_err(|e| {
        input_error_noloc!(TranslationErr::unsupported(format!(
            "Failed to query enum layout: {:?}",
            e
        )))
    })?;
    let shape = layout.shape();

    match &shape.variants {
        rustc_public::abi::VariantsShape::Single { index } => Ok(index.to_index()),
        rustc_public::abi::VariantsShape::Empty => input_err!(
            loc,
            TranslationErr::unsupported("Cannot materialize a constant for an uninhabited enum")
        ),
        rustc_public::abi::VariantsShape::Multiple {
            tag,
            tag_encoding,
            tag_field,
            ..
        } => {
            let primitive = match tag {
                rustc_public::abi::Scalar::Initialized { value, .. }
                | rustc_public::abi::Scalar::Union { value } => *value,
            };
            let scalar_size = primitive.size(&rustc_public::target::MachineInfo::target());
            let mask = scalar_size.unsigned_int_max().ok_or_else(|| {
                input_error_noloc!(TranslationErr::unsupported(format!(
                    "Enum tag width {} exceeds 128 bits",
                    scalar_size.bits()
                )))
            })?;

            if let Some((alloc, base_offset)) = allocation {
                let (relative_tag_offset, tag_size) =
                    enum_tag_byte_range(&shape.fields, *tag_field, *tag, loc.clone())?;
                let absolute_tag_offset =
                    base_offset
                        .checked_add(relative_tag_offset)
                        .ok_or_else(|| {
                            input_error_noloc!(TranslationErr::unsupported(
                                "Enum tag absolute offset overflowed"
                            ))
                        })?;
                let absolute_tag_end =
                    absolute_tag_offset.checked_add(tag_size).ok_or_else(|| {
                        input_error_noloc!(TranslationErr::unsupported(
                            "Enum tag absolute range overflowed"
                        ))
                    })?;
                let pointer_width =
                    rustc_public::target::MachineInfo::target_pointer_width().bytes();
                let overlaps = relocation_offsets_overlapping_range(
                    &alloc.provenance.ptrs,
                    absolute_tag_offset,
                    absolute_tag_end,
                    pointer_width,
                );

                if !overlaps.is_empty() {
                    if let rustc_public::abi::TagEncoding::Niche {
                        untagged_variant, ..
                    } = tag_encoding
                        && tag_size == pointer_width
                        && overlaps.len() == 1
                        && overlaps[0] == absolute_tag_offset
                    {
                        return Ok(untagged_variant.to_index());
                    }

                    return input_err!(
                        loc,
                        TranslationErr::unsupported(format!(
                            "Enum tag bytes [{absolute_tag_offset}..{absolute_tag_end}) overlap \
                             pointer relocation(s) at {overlaps:?}; only one full-width \
                             relocation exactly covering a niche pointer carrier is supported"
                        ))
                    );
                }
            }

            let tag_value =
                read_enum_tag_value(enum_bytes, &shape.fields, *tag_field, *tag, loc.clone())?;

            match tag_encoding {
                rustc_public::abi::TagEncoding::Direct => {
                    // The tag bytes hold a declared discriminant VALUE
                    // truncated to the PHYSICAL tag width; the caller wants
                    // a variant INDEX. `discriminant_for_variant().val` is
                    // at the declared discriminant type's width (isize for
                    // default-repr enums), so the comparison must mask both
                    // sides to the tag width (`Neg::N = -5` is byte 0xFB in
                    // an i8 tag but 0xFFFF_FFFF_FFFF_FFFB as isize). A tag
                    // that matches no declared discriminant means we
                    // misread the constant; falling back to
                    // "value == index" would silently conflate the two
                    // semantics (the issue #146 bug class).
                    discriminant_to_variant_index(rust_ty, tag_value, mask).ok_or_else(|| {
                        input_error!(
                            loc.clone(),
                            TranslationErr::unsupported(format!(
                                "Enum constant tag value {} matches no declared discriminant",
                                tag_value
                            ))
                        )
                    })
                }
                rustc_public::abi::TagEncoding::Niche {
                    untagged_variant,
                    niche_variants,
                    niche_start,
                } => {
                    let niche_start_idx = niche_variants.start().to_index();
                    let niche_end_idx = niche_variants.end().to_index();
                    Ok(decode_niche_variant_index(
                        tag_value,
                        mask,
                        *niche_start,
                        niche_start_idx,
                        niche_end_idx,
                        untagged_variant.to_index(),
                    ))
                }
            }
        }
    }
}

/// Return the byte range occupied by an enum's direct tag or niche carrier.
fn enum_tag_byte_range(
    fields: &rustc_public::abi::FieldsShape,
    tag_field: usize,
    tag: rustc_public::abi::Scalar,
    loc: Location,
) -> TranslationResult<(usize, usize)> {
    let primitive = match tag {
        rustc_public::abi::Scalar::Initialized { value, .. }
        | rustc_public::abi::Scalar::Union { value } => value,
    };
    let byte_size = primitive
        .size(&rustc_public::target::MachineInfo::target())
        .bytes();
    let offset = enum_tag_offset(fields, tag_field, loc)?;
    Ok((offset, byte_size))
}

/// Read an enum tag scalar from raw bytes using the stable layout metadata.
fn read_enum_tag_value(
    enum_bytes: &[u8],
    fields: &rustc_public::abi::FieldsShape,
    tag_field: usize,
    tag: rustc_public::abi::Scalar,
    loc: Location,
) -> TranslationResult<u128> {
    let (offset, byte_size) = enum_tag_byte_range(fields, tag_field, tag, loc.clone())?;

    let end = offset.checked_add(byte_size).ok_or_else(|| {
        input_error_noloc!(TranslationErr::unsupported(format!(
            "Enum tag overflowed offset computation: offset={}, size={}",
            offset, byte_size
        )))
    })?;
    if end > enum_bytes.len() {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Enum tag needs bytes [{}..{}), but constant only has {} bytes",
                offset,
                end,
                enum_bytes.len()
            ))
        );
    }

    Ok(read_uint_from_bytes(&enum_bytes[offset..end]))
}

/// Decode an integer from raw bytes using the current target endianness.
pub(super) fn read_uint_from_bytes(bytes: &[u8]) -> u128 {
    match rustc_public::target::MachineInfo::target_endianness() {
        rustc_public::target::Endian::Little => {
            bytes.iter().enumerate().fold(0u128, |acc, (idx, byte)| {
                acc | ((*byte as u128) << (idx * 8))
            })
        }
        rustc_public::target::Endian::Big => bytes
            .iter()
            .fold(0u128, |acc, byte| (acc << 8) | (*byte as u128)),
    }
}

/// Convert a discriminant value to a variant index.
///
/// For enums with explicit discriminants (e.g., `enum { A = 0, B = 2, C = 6 }`),
/// the discriminant value differs from the variant index:
/// - Variant index: position in the enum (0, 1, 2, ...)
/// - Discriminant: the explicit or implicit value assigned to each variant
///
/// `tag_value` is the raw tag read from memory, i.e. the discriminant
/// truncated to the PHYSICAL tag width, while `discriminant_for_variant`
/// reports values at the declared discriminant type's width (isize for
/// default-repr enums). `mask` is the tag width's unsigned max; both
/// sides are masked to it so negative discriminants compare correctly
/// (`-5` is `0xFB` in an i8 tag but `0xFFFF_FFFF_FFFF_FFFB` as isize).
///
/// This function iterates through variants to find which one has the given discriminant.
fn discriminant_to_variant_index(
    rust_ty: &rustc_public::ty::Ty,
    tag_value: u128,
    mask: u128,
) -> Option<usize> {
    use rustc_public::ty::{RigidTy, TyKind};

    match rust_ty.kind() {
        TyKind::RigidTy(RigidTy::Adt(adt_def, _)) => {
            for (idx, _variant_def) in adt_def.variants().iter().enumerate() {
                let variant_idx = rustc_public::ty::VariantIdx::to_val(idx);
                let discr = adt_def.discriminant_for_variant(variant_idx);
                if discr.val & mask == tag_value & mask {
                    return Some(idx);
                }
            }
            None
        }
        _ => None,
    }
}

/// Create a placeholder `MirConstructEnumOp` for a ghost local.
///
/// Ghost locals are MIR locals that are referenced but never assigned — e.g.
/// rustc optimised away their definition. When translation encounters one we
/// synthesise a variant-0 enum value with no fields -- the moral equivalent
/// of LLVM `undef` for an enum.
///
/// Typical trigger: `Option<Infallible>` which is always `None` (variant 0,
/// no payload) after MIR optimisations.
///
/// The returned operation is **not** inserted into any block; the caller must
/// link it via `insert_after` / `insert_at_front`.
pub(super) fn create_ghost_enum_default(
    ctx: &mut Context,
    ty_ptr: pliron::r#type::TypeHandle,
    loc: Location,
) -> Ptr<Operation> {
    use dialect_mir::ops::MirConstructEnumOp;
    let op = Operation::new(
        ctx,
        MirConstructEnumOp::get_concrete_op_info(),
        vec![ty_ptr],
        vec![],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc);
    MirConstructEnumOp::new(op)
        .set_attr_construct_enum_variant_index(ctx, dialect_mir::attributes::VariantIndexAttr(0));
    op
}

#[cfg(test)]
mod enum_niche_decode_tests {
    use super::decode_niche_variant_index;

    #[test]
    fn i128_relative_value_is_checked_before_usize_conversion() {
        assert_eq!(
            decode_niche_variant_index(1u128 << 64, u128::MAX, 0, 0, 1, 2),
            2,
            "2^64 must not truncate to relative variant zero on a 64-bit host"
        );
    }

    #[test]
    fn niche_decode_wraps_at_the_carrier_width() {
        assert_eq!(
            decode_niche_variant_index(0, u8::MAX.into(), u8::MAX.into(), 3, 4, 1),
            4,
            "u8 carrier value 0 is one step after niche_start 255"
        );
    }
}
