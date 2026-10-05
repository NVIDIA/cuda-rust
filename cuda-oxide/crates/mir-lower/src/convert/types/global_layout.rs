/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Layout validators for initialized (and relocated) device globals.

use dialect_mir::types::{MirEnumType, MirStructType, MirTupleType};
use llvm_export::types as llvm_types;
use pliron::context::Context;
use pliron::r#type::TypeHandle;

use super::layout::mir_stored_size;
use super::{
    StructLayoutInfo, build_struct_slot_map, convert_type, enum_unmodeled_in_memory,
    llvm_type_size_align,
};

/// Prove that an initialized Rust global can be accessed through the LLVM
/// semantic type produced by this lowering pipeline.
///
/// The initializer itself is emitted as exact bytes. That is only half of the
/// contract: later field GEPs and typed loads still use `mir_ty`. If that type
/// places a field at a different byte offset, an exact initializer can still be
/// read incorrectly (or even past the end of the object). Reject every shape
/// for which we cannot prove that the two views agree.
pub(crate) fn validate_initialized_global_layout(
    ctx: &mut Context,
    mir_ty: TypeHandle,
    initializer_size: u64,
    initializer_align: u64,
) -> Result<(), anyhow::Error> {
    if initializer_align == 0 || !initializer_align.is_power_of_two() {
        return Err(anyhow::anyhow!(
            "initialized global has invalid rustc allocation alignment {}",
            initializer_align
        ));
    }

    validate_initialized_global_type(ctx, mir_ty, &mut Vec::new())?;

    let llvm_ty = convert_type(ctx, mir_ty)?;
    let (llvm_size, llvm_align) = llvm_type_size_align(ctx, llvm_ty)
        .ok_or_else(|| anyhow::anyhow!("initialized global has unsupported LLVM size/alignment"))?;
    if llvm_size != initializer_size || llvm_align > initializer_align {
        return Err(anyhow::anyhow!(
            "initialized global type is not byte-compatible with rustc's allocation: the lowered LLVM value has size/alignment {}/{}, but the initializer has size/alignment {}/{}",
            llvm_size,
            llvm_align,
            initializer_size,
            initializer_align
        ));
    }

    Ok(())
}

/// Validate an initialized global that carries pointer relocations.
///
/// Ordinary initialized globals keep their existing conservative semantic
/// layout validator. Relocated globals use a separate segmented physical
/// carrier, so a top-level `repr(packed)` struct may use that relocation path
/// as long as rustc's field ranges are explicit, non-overlapping, and in-bounds.
/// One direct nested struct may use the same relaxation when the top-level
/// struct itself has an ordinary, non-divergent LLVM layout and the child's
/// LLVM packed representation exactly reproduces rustc's offsets and size. A
/// packed top-level struct may not stack the relaxation on a packed child, and
/// deeper packed nesting remains on the ordinary fail-closed path.
pub(crate) fn validate_relocated_initialized_global_layout(
    ctx: &mut Context,
    mir_ty: TypeHandle,
    initializer_size: u64,
    initializer_align: u64,
) -> Result<(), anyhow::Error> {
    match validate_initialized_global_layout(ctx, mir_ty, initializer_size, initializer_align) {
        Ok(()) => Ok(()),
        Err(original_error) => {
            if initializer_align == 0 || !initializer_align.is_power_of_two() {
                return Err(original_error);
            }

            let struct_ty = {
                let ty_ref = mir_ty.deref(ctx);
                ty_ref.downcast_ref::<MirStructType>().cloned()
            };
            let Some(struct_ty) = struct_ty else {
                return Err(original_error);
            };
            if !struct_ty.has_explicit_layout()
                || struct_ty.total_size() != initializer_size
                || struct_ty.abi_align != initializer_align
            {
                return Err(original_error);
            }

            validate_relocated_struct_field_ranges(ctx, &struct_ty)?;

            let top_level_layout = StructLayoutInfo::of_struct(&struct_ty);
            let top_level_map = build_struct_slot_map(ctx, &top_level_layout)?;
            let top_level_is_ordinary = !top_level_map.layout_diverges
                && top_level_map.by_value_layout_faithful
                && top_level_map
                    .llvm_struct_ty
                    .deref(ctx)
                    .downcast_ref::<llvm_types::StructType>()
                    .is_some_and(|ty| ty.layout() == llvm_types::StructLayout::Unpacked);

            let mut visited = vec![mir_ty];
            for field_ty in struct_ty.field_types.iter().copied() {
                if top_level_is_ordinary {
                    validate_relocated_initialized_global_child(ctx, field_ty, &mut visited)?;
                } else {
                    validate_initialized_global_type(ctx, field_ty, &mut visited)?;
                }
            }
            Ok(())
        }
    }
}

/// Prove that rustc's explicit field ranges for a relocated top-level struct
/// are non-overlapping and fit within the evaluated allocation.
fn validate_relocated_struct_field_ranges(
    ctx: &mut Context,
    struct_ty: &MirStructType,
) -> Result<(), anyhow::Error> {
    let layout = StructLayoutInfo::of_struct(struct_ty);

    // Relocated globals relax only natural LLVM placement. Preserve the
    // ordinary struct-layout metadata invariants before accepting rustc's
    // explicit byte ranges as the physical initializer contract.
    build_struct_slot_map(ctx, &layout)?;

    let mut ranges = Vec::with_capacity(layout.field_types.len());
    for (decl_index, field_ty) in layout.field_types.iter().copied().enumerate() {
        let field_size = if let Some(size) = mir_stored_size(ctx, field_ty) {
            size
        } else {
            let llvm_ty = convert_type(ctx, field_ty)?;
            llvm_type_size_align(ctx, llvm_ty)
                .map(|(size, _)| size)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "relocated initialized struct `{}` field {} has unsupported size",
                        struct_ty.name(),
                        decl_index
                    )
                })?
        };
        if field_size == 0 {
            continue;
        }

        let start = layout.field_offsets[decl_index];
        let end = start.checked_add(field_size).ok_or_else(|| {
            anyhow::anyhow!(
                "relocated initialized struct `{}` field {} range overflows",
                struct_ty.name(),
                decl_index
            )
        })?;
        if end > layout.total_size {
            return Err(anyhow::anyhow!(
                "relocated initialized struct `{}` field {} occupies bytes {}..{}, but the allocation is only {} bytes",
                struct_ty.name(),
                decl_index,
                start,
                end,
                layout.total_size
            ));
        }
        ranges.push((start, end, decl_index));
    }

    ranges.sort_by_key(|(start, _, _)| *start);
    for pair in ranges.windows(2) {
        let (_, previous_end, previous_index) = pair[0];
        let (next_start, _, next_index) = pair[1];
        if next_start < previous_end {
            return Err(anyhow::anyhow!(
                "relocated initialized struct `{}` fields {} and {} overlap in rustc's byte layout",
                struct_ty.name(),
                previous_index,
                next_index
            ));
        }
    }

    Ok(())
}

/// Validate one direct child of an ordinary relocated initialized global.
///
/// The ordinary initialized-global validator remains the default. Only when it
/// rejects a direct `MirStructType` do we consider the relocation-specific
/// relaxation, and only if `build_struct_slot_map` proves that the divergent
/// rustc layout is represented exactly by an LLVM packed struct. Its own field
/// ranges must still be explicit, non-overlapping, and in-bounds. Children of
/// that packed struct go back through the ordinary validator, deliberately
/// limiting this exception to one nesting level.
fn validate_relocated_initialized_global_child(
    ctx: &mut Context,
    mir_ty: TypeHandle,
    visited: &mut Vec<TypeHandle>,
) -> Result<(), anyhow::Error> {
    if visited.contains(&mir_ty) {
        return Ok(());
    }

    let mut ordinary_visited = visited.clone();
    let original_error = match validate_initialized_global_type(ctx, mir_ty, &mut ordinary_visited)
    {
        Ok(()) => {
            *visited = ordinary_visited;
            return Ok(());
        }
        Err(error) => error,
    };

    let struct_ty = {
        let ty_ref = mir_ty.deref(ctx);
        ty_ref.downcast_ref::<MirStructType>().cloned()
    };
    let Some(struct_ty) = struct_ty else {
        return Err(original_error);
    };
    if !struct_ty.has_explicit_layout() {
        return Err(original_error);
    }

    let layout = StructLayoutInfo::of_struct(&struct_ty);
    let map = build_struct_slot_map(ctx, &layout)?;
    let is_exact_packed = map.layout_diverges
        && map.by_value_layout_faithful
        && map
            .llvm_struct_ty
            .deref(ctx)
            .downcast_ref::<llvm_types::StructType>()
            .is_some_and(|ty| ty.layout() == llvm_types::StructLayout::Packed);
    if !is_exact_packed {
        return Err(original_error);
    }

    validate_relocated_struct_field_ranges(ctx, &struct_ty)?;

    visited.push(mir_ty);
    for field_ty in struct_ty.field_types.iter().copied() {
        validate_initialized_global_type(ctx, field_ty, visited)?;
    }
    Ok(())
}

fn validate_initialized_global_type(
    ctx: &mut Context,
    mir_ty: TypeHandle,
    visited: &mut Vec<TypeHandle>,
) -> Result<(), anyhow::Error> {
    if visited.contains(&mir_ty) {
        return Ok(());
    }
    visited.push(mir_ty);

    if let Some(name) = enum_unmodeled_in_memory(ctx, mir_ty) {
        return Err(anyhow::anyhow!(
            "initialized global contains enum `{}` with unknown physical layout; refusing to guess a byte representation",
            name
        ));
    }

    enum Kind {
        Struct(MirStructType),
        Tuple(MirTupleType),
        Enum(MirEnumType),
        Array { element_ty: TypeHandle, size: u64 },
        Leaf,
    }

    let kind = {
        let ty_ref = mir_ty.deref(ctx);
        if let Some(struct_ty) = ty_ref.downcast_ref::<MirStructType>() {
            Kind::Struct(struct_ty.clone())
        } else if let Some(tuple_ty) = ty_ref.downcast_ref::<MirTupleType>() {
            Kind::Tuple(tuple_ty.clone())
        } else if let Some(enum_ty) = ty_ref.downcast_ref::<MirEnumType>() {
            Kind::Enum(enum_ty.clone())
        } else if let Some(array_ty) = ty_ref.downcast_ref::<dialect_mir::types::MirArrayType>() {
            Kind::Array {
                element_ty: array_ty.element_ty,
                size: array_ty.size,
            }
        } else {
            Kind::Leaf
        }
    };

    match kind {
        Kind::Struct(struct_ty) => {
            validate_initialized_struct_layout(ctx, mir_ty, &struct_ty)?;
            for field_ty in struct_ty.field_types {
                validate_initialized_global_type(ctx, field_ty, visited)?;
            }
        }
        Kind::Tuple(tuple_ty) => {
            if !tuple_ty.get_types().is_empty() {
                // Tuples carry rustc's field offsets exactly like structs;
                // prove the lowered aggregate reproduces them byte-for-byte.
                let layout = StructLayoutInfo::of_tuple(&tuple_ty);
                validate_initialized_aggregate_layout(
                    ctx,
                    mir_ty,
                    "tuple",
                    "tuple",
                    &layout,
                    tuple_ty.abi_align(),
                    tuple_ty.has_explicit_layout(),
                )?;
            }
            for field_ty in tuple_ty.get_types().iter().copied() {
                validate_initialized_global_type(ctx, field_ty, visited)?;
            }
        }
        Kind::Enum(enum_ty) => {
            if enum_ty.total_size() > 0 {
                let llvm_ty = convert_type(ctx, mir_ty)?;
                let (llvm_size, llvm_align) =
                    llvm_type_size_align(ctx, llvm_ty).ok_or_else(|| {
                        anyhow::anyhow!(
                            "initialized enum `{}` has unsupported LLVM size/alignment",
                            enum_ty.name()
                        )
                    })?;
                if llvm_size != enum_ty.total_size() || llvm_align > enum_ty.abi_align() {
                    return Err(anyhow::anyhow!(
                        "initialized enum `{}` is not byte-compatible with rustc's layout: the lowered LLVM value has size/alignment {}/{}, but rustc requires {}/{}",
                        enum_ty.name(),
                        llvm_size,
                        llvm_align,
                        enum_ty.total_size(),
                        enum_ty.abi_align()
                    ));
                }
            }
            for field_ty in enum_ty.all_field_types {
                validate_initialized_global_type(ctx, field_ty, visited)?;
            }
        }
        Kind::Array { element_ty, size } => {
            // A zero-length array contributes no element bytes. Its outer LLVM
            // size/alignment is still checked by
            // `validate_initialized_global_layout`, but recursively rejecting
            // an element layout would incorrectly reject valid promoted
            // `&mut [Packed; 0]` constants whose physical allocation is empty.
            if size != 0 {
                validate_initialized_global_type(ctx, element_ty, visited)?;
            }
        }
        Kind::Leaf => {}
    }

    Ok(())
}

fn validate_initialized_struct_layout(
    ctx: &mut Context,
    mir_ty: TypeHandle,
    struct_ty: &MirStructType,
) -> Result<(), anyhow::Error> {
    let has_layout = struct_ty.has_explicit_layout();
    let layout = StructLayoutInfo::of_struct(struct_ty);
    let name = struct_ty.name().to_string();
    validate_initialized_aggregate_layout(
        ctx,
        mir_ty,
        "struct",
        &name,
        &layout,
        struct_ty.abi_align,
        has_layout,
    )
}

/// Shared byte-layout validation for initialized-global structs and tuples.
///
/// Proves that the lowered LLVM aggregate places every field at exactly the
/// byte offset rustc chose and matches rustc's total size, so a constant
/// initializer written slot-by-slot reproduces the host bytes.
#[allow(clippy::too_many_arguments)]
fn validate_initialized_aggregate_layout(
    ctx: &mut Context,
    mir_ty: TypeHandle,
    kind_noun: &str,
    name: &str,
    layout: &StructLayoutInfo,
    abi_align: u64,
    has_explicit_layout: bool,
) -> Result<(), anyhow::Error> {
    if layout.total_size == 0 {
        let llvm_ty = convert_type(ctx, mir_ty)?;
        let (llvm_size, _) = llvm_type_size_align(ctx, llvm_ty).ok_or_else(|| {
            anyhow::anyhow!(
                "initialized {} `{}` has unsupported LLVM size/alignment",
                kind_noun,
                name
            )
        })?;
        if llvm_size == 0 {
            return Ok(());
        }
        return Err(anyhow::anyhow!(
            "initialized {} `{}` has no stored size but lowers to {} bytes",
            kind_noun,
            name,
            llvm_size
        ));
    }
    if !has_explicit_layout {
        return Err(anyhow::anyhow!(
            "initialized {} `{}` has no rustc field-offset metadata",
            kind_noun,
            name
        ));
    }

    let slots = build_struct_slot_map(ctx, layout)?;
    let llvm_fields: Vec<_> = slots
        .llvm_struct_ty
        .deref(ctx)
        .downcast_ref::<llvm_types::StructType>()
        .expect("aggregate slot map must produce an LLVM struct")
        .fields()
        .collect();

    let mut slot_offsets = Vec::with_capacity(llvm_fields.len());
    let mut current_offset = 0u64;
    for llvm_field in &llvm_fields {
        let (field_size, field_align) =
            llvm_type_size_align(ctx, *llvm_field).ok_or_else(|| {
                anyhow::anyhow!(
                    "initialized {} `{}` field has unsupported LLVM size/alignment",
                    kind_noun,
                    name
                )
            })?;
        current_offset = current_offset.div_ceil(field_align.max(1)) * field_align.max(1);
        slot_offsets.push(current_offset);
        current_offset += field_size;
    }

    for (decl_index, slot) in slots.decl_to_llvm.iter().enumerate() {
        let Some(slot) = slot else {
            continue;
        };
        let actual_offset = slot_offsets[*slot as usize];
        let expected_offset = layout.field_offsets[decl_index];
        if actual_offset != expected_offset {
            return Err(anyhow::anyhow!(
                "initialized {} `{}` field {} lowers at byte {}, but rustc placed it at byte {}; packed and overlapping field layouts are not yet supported",
                kind_noun,
                name,
                decl_index,
                actual_offset,
                expected_offset
            ));
        }
    }

    let (llvm_size, llvm_align) =
        llvm_type_size_align(ctx, slots.llvm_struct_ty).ok_or_else(|| {
            anyhow::anyhow!(
                "initialized {} `{}` has unsupported LLVM size/alignment",
                kind_noun,
                name
            )
        })?;
    if llvm_size != layout.total_size || llvm_align > abi_align {
        return Err(anyhow::anyhow!(
            "initialized {} `{}` lowers to size/alignment {}/{}, but rustc requires {}/{}",
            kind_noun,
            name,
            llvm_size,
            llvm_align,
            layout.total_size,
            abi_align
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{make_ctx, mir_uint, mir_zst};
    use super::*;
    use dialect_mir::types::{EnumVariant, MirArrayType, MirPtrType, MirUnionType};

    #[test]
    fn initialized_global_layout_accepts_explicit_overalignment() {
        let mut ctx = make_ctx();
        let zst = mir_zst(&mut ctx);
        validate_initialized_global_layout(&mut ctx, zst, 0, 1).unwrap();

        let byte = mir_uint(&mut ctx, 8);
        let over_aligned: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "OverAligned".into(),
            vec!["byte".into()],
            vec![byte],
            vec![0],
            vec![0],
            16,
            16,
        )
        .into();

        validate_initialized_global_layout(&mut ctx, over_aligned, 16, 16).unwrap();
    }

    #[test]
    fn initialized_global_layout_rejects_packed_and_nested_packed_structs() {
        let mut ctx = make_ctx();
        let byte = mir_uint(&mut ctx, 8);
        let word = mir_uint(&mut ctx, 32);
        let packed: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "Packed".into(),
            vec!["tag".into(), "word".into()],
            vec![byte, word],
            vec![0, 1],
            vec![0, 1],
            5,
            1,
        )
        .into();

        let empty_packed: TypeHandle = MirArrayType::get(&mut ctx, packed, 0).into();
        validate_initialized_global_layout(&mut ctx, empty_packed, 0, 1)
            .expect("[Packed; 0] has no element bytes whose field layout can diverge");

        let err = validate_initialized_global_layout(&mut ctx, packed, 5, 1).unwrap_err();
        assert!(err.to_string().contains("field 1 lowers at byte 4"));

        // Nesting must not hide the incompatible packed representation.
        let wide = mir_uint(&mut ctx, 64);
        let outer: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "Outer".into(),
            vec!["packed".into(), "wide".into()],
            vec![packed, wide],
            vec![0, 1],
            vec![0, 8],
            16,
            8,
        )
        .into();
        let err = validate_initialized_global_layout(&mut ctx, outer, 16, 8).unwrap_err();
        assert!(err.to_string().contains("lowers at byte"));
    }

    #[test]
    fn relocated_initialized_global_layout_accepts_top_level_packed_struct() {
        let mut ctx = make_ctx();
        let byte = mir_uint(&mut ctx, 8);
        let target = mir_uint(&mut ctx, 32);
        let pointer: TypeHandle = MirPtrType::get_global(&mut ctx, target, false).into();
        let packed: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "PackedRelocation".into(),
            vec!["tag".into(), "ptr".into()],
            vec![byte, pointer],
            vec![0, 1],
            vec![0, 1],
            9,
            1,
        )
        .into();

        validate_relocated_initialized_global_layout(&mut ctx, packed, 9, 1)
            .expect("relocated top-level packed struct must be accepted");
        assert!(validate_initialized_global_layout(&mut ctx, packed, 9, 1).is_err());
    }

    #[test]
    fn relocated_initialized_global_layout_accepts_thin_pointer_union() {
        let mut ctx = make_ctx();
        let u32_ty = mir_uint(&mut ctx, 32);
        let u8_ty = mir_uint(&mut ctx, 8);
        let word_ptr: TypeHandle = MirPtrType::get_generic(&mut ctx, u32_ty, false).into();
        let byte_ptr: TypeHandle = MirPtrType::get_generic(&mut ctx, u8_ty, false).into();
        let union_ty: TypeHandle = MirUnionType::get(
            &mut ctx,
            "RelocatedPointerUnion".into(),
            vec!["word".into(), "byte".into()],
            vec![word_ptr, byte_ptr],
            8,
            8,
        )
        .into();

        validate_relocated_initialized_global_layout(&mut ctx, union_ty, 8, 8)
            .expect("one pointer-word union must be valid relocated global storage");
    }

    #[test]
    fn relocated_initialized_global_layout_rejects_malformed_memory_order() {
        let mut ctx = make_ctx();
        let byte = mir_uint(&mut ctx, 8);
        let target = mir_uint(&mut ctx, 32);
        let pointer: TypeHandle = MirPtrType::get_global(&mut ctx, target, false).into();
        let malformed: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "MalformedPackedRelocation".into(),
            vec!["tag".into(), "ptr".into()],
            vec![byte, pointer],
            vec![0, 0],
            vec![0, 1],
            9,
            1,
        )
        .into();

        let error = validate_relocated_initialized_global_layout(&mut ctx, malformed, 9, 1)
            .expect_err("malformed memory order must remain unsupported");
        assert!(error.to_string().contains("not a permutation"), "{error}");
    }

    #[test]
    fn relocated_initialized_global_layout_accepts_one_direct_nested_packed_struct() {
        let mut ctx = make_ctx();
        let byte = mir_uint(&mut ctx, 8);
        let word = mir_uint(&mut ctx, 32);
        let pointer: TypeHandle = MirPtrType::get_global(&mut ctx, word, false).into();
        let nested_packed: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "NestedPackedRelocation".into(),
            vec!["tag".into(), "ptr".into()],
            vec![byte, pointer],
            vec![0, 1],
            vec![0, 1],
            9,
            1,
        )
        .into();
        let outer: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "OuterPackedRelocation".into(),
            vec!["head".into(), "nested".into()],
            vec![word, nested_packed],
            vec![0, 1],
            vec![0, 4],
            16,
            4,
        )
        .into();

        validate_relocated_initialized_global_layout(&mut ctx, outer, 16, 4)
            .expect("one direct nested packed relocation carrier must be accepted");
        assert!(validate_initialized_global_layout(&mut ctx, outer, 16, 4).is_err());
    }

    #[test]
    fn relocated_initialized_global_layout_rejects_packed_root_with_packed_child() {
        let mut ctx = make_ctx();
        let byte = mir_uint(&mut ctx, 8);
        let word = mir_uint(&mut ctx, 32);
        let target = mir_uint(&mut ctx, 32);
        let pointer: TypeHandle = MirPtrType::get_global(&mut ctx, target, false).into();
        let nested_packed: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "NestedPackedRelocation".into(),
            vec!["tag".into(), "ptr".into()],
            vec![byte, pointer],
            vec![0, 1],
            vec![0, 1],
            9,
            1,
        )
        .into();
        let packed_root: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "PackedRootRelocation".into(),
            vec!["tag".into(), "word".into(), "nested".into()],
            vec![byte, word, nested_packed],
            vec![0, 1, 2],
            vec![0, 1, 5],
            14,
            1,
        )
        .into();

        let error = validate_relocated_initialized_global_layout(&mut ctx, packed_root, 14, 1)
            .expect_err("a packed top-level struct must not stack the nested packed relaxation");
        assert!(error.to_string().contains("lowers at byte"), "{error}");
    }

    #[test]
    fn relocated_initialized_global_layout_rejects_deeper_packed_nesting() {
        let mut ctx = make_ctx();
        let byte = mir_uint(&mut ctx, 8);
        let word = mir_uint(&mut ctx, 32);
        let target = mir_uint(&mut ctx, 32);
        let pointer: TypeHandle = MirPtrType::get_global(&mut ctx, target, false).into();
        let inner_packed: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "InnerPackedRelocation".into(),
            vec!["tag".into(), "ptr".into()],
            vec![byte, pointer],
            vec![0, 1],
            vec![0, 1],
            9,
            1,
        )
        .into();
        let middle_packed: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "MiddlePackedRelocation".into(),
            vec!["tag".into(), "word".into(), "inner".into()],
            vec![byte, word, inner_packed],
            vec![0, 1, 2],
            vec![0, 1, 5],
            14,
            1,
        )
        .into();
        let outer: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "OuterDeepPackedRelocation".into(),
            vec!["head".into(), "middle".into()],
            vec![word, middle_packed],
            vec![0, 1],
            vec![0, 4],
            20,
            4,
        )
        .into();

        let error = validate_relocated_initialized_global_layout(&mut ctx, outer, 20, 4)
            .expect_err("packed relocation relaxation must stop after one nesting level");
        assert!(error.to_string().contains("lowers at byte"), "{error}");
    }

    #[test]
    fn relocated_initialized_global_layout_rejects_overlapping_struct_fields() {
        let mut ctx = make_ctx();
        let word = mir_uint(&mut ctx, 32);
        let overlapping: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "OverlappingRelocation".into(),
            vec!["left".into(), "right".into()],
            vec![word, word],
            vec![0, 1],
            vec![0, 0],
            4,
            1,
        )
        .into();

        let error = validate_relocated_initialized_global_layout(&mut ctx, overlapping, 4, 1)
            .expect_err("overlapping field ranges must remain unsupported");
        assert!(error.to_string().contains("overlap"), "{error}");
    }

    #[test]
    fn initialized_global_layout_rejects_old_union_and_tuple_models() {
        let mut ctx = make_ctx();
        let word = mir_uint(&mut ctx, 32);
        let union_as_struct: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "UnionBeforeSharedStorageLowering".into(),
            vec!["left".into(), "right".into()],
            vec![word, word],
            vec![0, 1],
            vec![0, 0],
            4,
            4,
        )
        .into();
        let err = validate_initialized_global_layout(&mut ctx, union_as_struct, 4, 4).unwrap_err();
        assert!(err.to_string().contains("field 1 lowers at byte 4"));

        // A tuple without recorded rustc layout cannot prove its bytes.
        let byte = mir_uint(&mut ctx, 8);
        let wide = mir_uint(&mut ctx, 64);
        let tuple: TypeHandle = MirTupleType::get(&mut ctx, vec![byte, wide]).into();
        let err = validate_initialized_global_layout(&mut ctx, tuple, 16, 8).unwrap_err();
        assert!(
            err.to_string()
                .contains("no stored size but lowers to 16 bytes"),
            "{err}"
        );

        // The same tuple carrying rustc's real (reordered) layout validates:
        // memory order is (wide @ 0, byte @ 8), total size 16.
        let laid_out_tuple: TypeHandle = MirTupleType::get_with_layout(
            &mut ctx,
            vec![byte, wide],
            vec![1, 0],
            vec![8, 0],
            16,
            8,
        )
        .into();
        validate_initialized_global_layout(&mut ctx, laid_out_tuple, 16, 8)
            .expect("tuples with recorded rustc offsets are provable initialized-global storage");
    }

    #[test]
    fn initialized_global_layout_rejects_enum_with_unknown_physical_layout() {
        let mut ctx = make_ctx();
        let discr = mir_uint(&mut ctx, 8);
        let payload = mir_uint(&mut ctx, 32);
        let niche: TypeHandle = MirEnumType::get(
            &mut ctx,
            "OptionNonZero".into(),
            discr,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new("Some".into(), vec![payload]),
            ],
        )
        .into();

        let err = validate_initialized_global_layout(&mut ctx, niche, 4, 4).unwrap_err();
        assert!(err.to_string().contains("unknown physical layout"));
    }
}
