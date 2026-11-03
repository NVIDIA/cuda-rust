// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared readers over rustc's aggregate layout metadata.
//!
//! Both the type importer (`translator/types.rs`, which records tag and
//! field byte offsets on `MirEnumType`) and constant decoding
//! (`translator/rvalue/const_enum.rs` and `const_alloc.rs`, which slice
//! payload bytes out of constant allocations) need the same offset
//! lookups. Keeping them here means the
//! two consumers cannot drift apart on how an offset is derived.

use pliron::location::Location;
use pliron::{input_err, input_error, input_error_noloc};
use rustc_public_bridge::IndexedVal;

use crate::error::{TranslationErr, TranslationResult};

/// Byte offsets of an aggregate's fields, in declaration order.
///
/// A constant's bytes are the memory image of its type, so a field's bytes start
/// where rustc's layout puts them. Advancing a cursor by each field's size instead
/// assumes the fields are packed in declaration order, which holds for neither a
/// type rustc pads nor one it reorders: in `(u8, u64)` the `u64` sits at offset 8,
/// rather than at offset 1.
///
/// `FieldsShape::Primitive` is rejected here. A struct or tuple carrying fields
/// is laid out as `Arbitrary`, so a primitive shape means the type was not the
/// aggregate the caller took it for. The enum reader above accepts it because a
/// single-variant enum legitimately reaches that shape with no fields to place.
pub(crate) fn aggregate_field_offsets(
    rust_ty: &rustc_public::ty::Ty,
    what: &str,
    loc: &Location,
) -> TranslationResult<Vec<usize>> {
    let layout = rust_ty.layout().map_err(|e| {
        input_error!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "Failed to query layout for {what} constant: {e:?}"
            ))
        )
    })?;

    match &layout.shape().fields {
        rustc_public::abi::FieldsShape::Arbitrary { offsets } => {
            Ok(offsets.iter().map(|offset| offset.bytes()).collect())
        }
        other => input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "{what} constant has an unsupported field shape: {other:?}"
            ))
        ),
    }
}

/// Return the byte offsets for the fields of one active enum variant.
pub(crate) fn enum_variant_field_offsets(
    layout: &rustc_public::abi::LayoutShape,
    variant_index: usize,
    loc: Location,
) -> TranslationResult<Vec<usize>> {
    match &layout.variants {
        rustc_public::abi::VariantsShape::Single { index } => {
            if index.to_index() != variant_index {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Enum layout single-variant index {} disagrees with requested variant {}",
                        index.to_index(),
                        variant_index
                    ))
                );
            }

            match &layout.fields {
                rustc_public::abi::FieldsShape::Primitive => Ok(vec![]),
                rustc_public::abi::FieldsShape::Arbitrary { offsets } => {
                    Ok(offsets.iter().map(|offset| offset.bytes()).collect())
                }
                other => input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Single-variant enum fields use unsupported shape {:?}",
                        other
                    ))
                ),
            }
        }
        rustc_public::abi::VariantsShape::Multiple { variants, .. } => variants
            .get(variant_index)
            .map(|variant| {
                variant
                    .offsets
                    .iter()
                    .map(|offset| offset.bytes())
                    .collect()
            })
            .ok_or_else(|| {
                input_error_noloc!(TranslationErr::unsupported(format!(
                    "Missing layout info for enum variant {}",
                    variant_index
                )))
            }),
        rustc_public::abi::VariantsShape::Empty => Ok(vec![]),
    }
}

/// Return the byte offset of the enum's discriminant tag within the object,
/// given the enum layout's own field shape and the tag's field index.
pub(crate) fn enum_tag_offset(
    fields: &rustc_public::abi::FieldsShape,
    tag_field: usize,
    loc: Location,
) -> TranslationResult<usize> {
    match fields {
        rustc_public::abi::FieldsShape::Primitive => {
            if tag_field == 0 {
                Ok(0)
            } else {
                input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Enum tag field {} out of bounds for primitive layout",
                        tag_field
                    ))
                )
            }
        }
        rustc_public::abi::FieldsShape::Arbitrary { offsets } => offsets
            .get(tag_field)
            .map(|offset| offset.bytes())
            .ok_or_else(|| {
                input_error_noloc!(TranslationErr::unsupported(format!(
                    "Enum tag field {} out of bounds for {} layout fields",
                    tag_field,
                    offsets.len()
                )))
            }),
        other => input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Enum tag extraction does not support field shape {:?}",
                other
            ))
        ),
    }
}
