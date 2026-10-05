/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The `EnumSlotMap` placement engine and enum type conversion
//! (`build_enum_slot_map`, `convert_enum_to_llvm`, unmodeled-enum probes).

use dialect_mir::types::{
    EnumCarrierKind, EnumLayoutKind, MirDisjointSliceType, MirEnumType, MirSliceType,
    MirStructType, MirTupleType, MirUnionType,
};
use llvm_export::types as llvm_types;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::Context;
use pliron::r#type::TypeHandle;

use super::layout::{make_enum_filler_type, mir_type_contains_i1};
use super::pointer_storage::{
    PointerOverlapRejection, analyze_pointer_overlap, llvm_type_contains_pointer,
};
use super::{
    convert_type, is_zero_sized_type, llvm_type_contains_i1, llvm_type_is_byte_faithful,
    llvm_type_size_align, natural_struct_layout,
};
use crate::convert::enum_payload_storage::{
    MAX_ENUM_PAYLOAD_ARRAY_REWRITE_LEAVES, enum_payload_storage_type,
};

/// The LLVM struct for an enum, plus a map saying where the tag and each
/// payload field ended up.
///
/// The struct type and the indices into it are produced by one walk in
/// [`build_enum_slot_map`], so they can never disagree. (Computing them
/// separately is how the issue #128 class of bug happened for structs.)
pub(crate) struct EnumSlotMap {
    /// The final LLVM struct type, including any `[N x i8]` filler slots.
    pub llvm_struct_ty: TypeHandle,
    /// Which struct slot holds rustc's physical tag/niche carrier. `None`
    /// for `Single` and `Empty` layouts, which have no tag in memory.
    pub carrier_slot: Option<u32>,
    /// Converted physical carrier type (integer or pointer), when present.
    pub carrier_llvm_ty: Option<TypeHandle>,
    /// Which struct slot holds each payload field, in the flattened
    /// order of `MirEnumType::all_field_types`. `None` means the field
    /// has no slot of its own: it is zero-sized, or its bytes are shared
    /// with a different-typed field of another variant. Such fields are
    /// read and written through memory at `field_offsets` instead.
    pub field_slots: Vec<Option<u32>>,
    /// Byte position of each payload field inside the enum.
    pub field_offsets: Vec<u64>,
    /// Converted LLVM type of each payload field.
    pub field_llvm_types: Vec<TypeHandle>,
}

/// Build the LLVM struct for an enum, placing everything at the byte
/// positions rustc chose.
///
/// Why this matters: the host (CPU) lays out enum values with rustc's
/// layout. If the device used different byte positions, every enum
/// passed to a kernel would be read wrong. So the device struct is built
/// to have the same bytes, position for position.
///
/// The wrinkle is that enum variants SHARE bytes (only one variant is
/// alive at a time), and an LLVM struct cannot say "these two fields
/// overlap". The slot map resolves each field one of three ways:
///
/// ```text
/// #[repr(u32)] enum E { A(u32), B(f32), C }
/// rustc: 8 bytes, tag at byte 0, A's u32 and B's f32 both at byte 4
///
/// LLVM struct: { i32, i32 }
///                 |     |
///        tag_slot=0     A's payload: own slot (nothing else typed i32
///                       wanted byte 4 first... B did, see below)
///
/// - own slot:   the field's bytes collide with nothing already placed.
/// - shared slot: another variant already placed the SAME type at the
///                SAME position; both map to that slot. (If B were
///                B(u32), A and B would simply share slot 1.)
/// - no slot:    the bytes are taken by a different type (B's f32 vs
///                A's u32 here). The field is still at byte 4, just not
///                nameable as a struct field; reads and writes go
///                through memory: spill the value to a stack slot, then
///                use a byte-precise pointer. No slot, but no lie.
/// ```
///
/// Gaps between placed fields, and the tail, are covered with explicit
/// `[N x i8]` filler so the struct's size is exactly rustc's no matter
/// what LLVM's own layout rules would have done.
///
/// Direct and Niche carriers are claimed first, so a semantic field with a
/// different SSA type cannot redefine the same physical bytes. Single and
/// Empty layouts simply have no carrier. Unknown layouts are rejected.
///
/// If the finished struct's size does not come out equal to rustc's,
/// something is deeply wrong and lowering would miscompile, so that is a
/// hard error rather than a debug assertion.
pub(crate) fn build_enum_slot_map(
    ctx: &mut Context,
    ty: TypeHandle,
) -> Result<EnumSlotMap, anyhow::Error> {
    let (
        name,
        discriminant_ty,
        all_field_types,
        all_field_offsets,
        all_field_sizes,
        variant_field_counts,
        variant_inhabited,
        tag_offset,
        total_size,
        abi_align,
        layout_kind,
        carrier_kind,
        carrier_width,
        carrier_address_space,
    ) = {
        let ty_ref = ty.deref(ctx);
        let enum_ty = ty_ref
            .downcast_ref::<MirEnumType>()
            .ok_or_else(|| anyhow::anyhow!("build_enum_slot_map: expected MirEnumType"))?;
        (
            enum_ty.name().to_string(),
            enum_ty.discriminant_ty,
            enum_ty.all_field_types.clone(),
            enum_ty.all_field_offsets.clone(),
            enum_ty.all_field_sizes.clone(),
            enum_ty.variant_field_counts.clone(),
            enum_ty.variant_inhabited.clone(),
            enum_ty.tag_offset(),
            enum_ty.total_size(),
            enum_ty.abi_align(),
            enum_ty.layout_kind,
            enum_ty.carrier_kind,
            enum_ty.carrier_width,
            enum_ty.carrier_address_space,
        )
    };

    if layout_kind == EnumLayoutKind::Unknown {
        return Err(anyhow::anyhow!(
            "enum slot map: `{}` has unknown physical layout; refusing to guess its bytes",
            name
        ));
    }
    if carrier_kind == EnumCarrierKind::Pointer
        && carrier_address_space == llvm_types::address_space::SHARED
    {
        return Err(anyhow::anyhow!(
            "enum slot map: `{}` uses a shared-memory pointer carrier whose size is target-mode dependent (64-bit PTX/legacy, 32-bit modern NVVM); refusing target-agnostic enum lowering",
            name
        ));
    }
    if carrier_kind == EnumCarrierKind::Integer && !carrier_width.is_multiple_of(8) {
        return Err(anyhow::anyhow!(
            "enum slot map: `{}` integer carrier width {} is not a whole number of bytes; refusing physical storage with unspecified upper byte bits",
            name,
            carrier_width
        ));
    }

    let carrier_ty: Option<TypeHandle> = match carrier_kind {
        EnumCarrierKind::None => None,
        EnumCarrierKind::Integer if layout_kind == EnumLayoutKind::Direct => {
            let converted = convert_type(ctx, discriminant_ty)?;
            let width = converted
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .map(IntegerType::width);
            if width != Some(carrier_width) {
                return Err(anyhow::anyhow!(
                    "enum slot map: `{}` direct carrier does not match its declared discriminant type",
                    name
                ));
            }
            Some(converted)
        }
        EnumCarrierKind::Integer => {
            Some(IntegerType::get(ctx, carrier_width, Signedness::Signless).into())
        }
        EnumCarrierKind::Pointer => {
            Some(llvm_types::PointerType::get(ctx, carrier_address_space).into())
        }
    };
    let mut field_llvm_types = Vec::with_capacity(all_field_types.len());
    for &field_ty in &all_field_types {
        field_llvm_types.push(convert_type(ctx, field_ty)?);
    }

    if all_field_offsets.len() != all_field_types.len()
        || all_field_sizes.len() != all_field_types.len()
    {
        return Err(anyhow::anyhow!(
            "enum slot map: `{}` has {} field offsets for {} fields",
            name,
            all_field_offsets.len().min(all_field_sizes.len()),
            all_field_types.len()
        ));
    }

    // Phase 1: decide who gets a struct slot. The physical carrier goes
    // first so a semantic field can never claim its bytes using a different
    // type (e.g. `bool` is i1 semantically but i8 in Option<bool> storage).
    // claims: (byte position, byte size, converted type), no two overlap.
    let mut claims: Vec<(u64, u64, TypeHandle)> = Vec::new();
    let carrier_claim = if let Some(carrier_ty) = carrier_ty {
        let (carrier_size, carrier_align) =
            llvm_type_size_align(ctx, carrier_ty).ok_or_else(|| {
                anyhow::anyhow!(
                    "enum slot map: `{}` carrier has unsupported LLVM layout",
                    name
                )
            })?;
        let expected_size = u64::from(carrier_width).div_ceil(8);
        if carrier_size != expected_size
            || tag_offset % carrier_align.max(1) != 0
            || tag_offset
                .checked_add(carrier_size)
                .is_none_or(|end| end > total_size)
        {
            return Err(anyhow::anyhow!(
                "enum slot map: `{}` carrier (size {}, align {}) cannot sit at byte {} of {}",
                name,
                carrier_size,
                carrier_align,
                tag_offset,
                total_size
            ));
        }
        claims.push((tag_offset, carrier_size, carrier_ty));
        Some(0usize)
    } else {
        None
    };

    let mut claim_of_field: Vec<Option<usize>> = vec![None; field_llvm_types.len()];
    let mut field_is_inhabited = Vec::with_capacity(field_llvm_types.len());
    for (variant, count) in variant_field_counts.iter().enumerate() {
        field_is_inhabited.extend(std::iter::repeat_n(
            variant_inhabited.get(variant).copied().unwrap_or(0) != 0,
            *count as usize,
        ));
    }
    let mut order: Vec<usize> = (0..field_llvm_types.len()).collect();
    // At one byte position, prefer a representation that preserves all of
    // its stored bits. This makes the result independent of source-variant
    // order and prevents an i1/bool view from claiming storage that a later
    // i8 view needs to preserve. A scalar i1 always uses an i8 storage claim:
    // construction explicitly zero-extends that scalar below.
    order.sort_by_key(|&i| {
        (
            all_field_offsets[i],
            !llvm_type_is_byte_faithful(ctx, field_llvm_types[i]),
            i,
        )
    });
    for flat in order {
        if !field_is_inhabited.get(flat).copied().unwrap_or(false) {
            continue;
        }
        let llvm_ty = field_llvm_types[flat];
        // Enum payload storage uses one target-stable physical view. Shared
        // pointer leaves become generic pointers recursively through LLVM
        // structs (MIR structs/tuples) and bounded arrays, while bool leaves
        // become canonical i8 bytes. Oversized shared-pointer arrays and
        // pointer vectors fail closed here.
        let storage_ty = enum_payload_storage_type(ctx, llvm_ty).map_err(|error| {
            anyhow::anyhow!("enum slot map: `{}` field {}: {error}", name, flat)
        })?;
        let (size, align) = llvm_type_size_align(ctx, storage_ty).ok_or_else(|| {
            anyhow::anyhow!(
                "enum slot map: `{}` field {} has unsupported LLVM size/alignment",
                name,
                flat
            )
        })?;
        if size != all_field_sizes[flat] {
            return Err(anyhow::anyhow!(
                "enum slot map: `{}` field {} lowers to {} bytes but rustc says {}",
                name,
                flat,
                size,
                all_field_sizes[flat]
            ));
        }
        if size == 0 || is_zero_sized_type(ctx, llvm_ty) {
            // ZSTs own no bytes and no slot.
            continue;
        }
        let offset = all_field_offsets[flat];
        if offset.checked_add(size).is_none_or(|end| end > total_size) {
            return Err(anyhow::anyhow!(
                "enum slot map: `{}` field {} (size {}) at byte {} exceeds total size {}",
                name,
                flat,
                size,
                offset,
                total_size
            ));
        }
        if !offset.is_multiple_of(align.max(1)) {
            return Err(anyhow::anyhow!(
                "enum slot map: `{}` field {} requires alignment {} but rustc offset {} is not aligned; packed enum payload access is not yet supported",
                name,
                flat,
                align,
                offset
            ));
        }

        let is_scalar_i1 = llvm_ty
            .deref(ctx)
            .downcast_ref::<IntegerType>()
            .is_some_and(|integer| integer.width() == 1);
        // A bool the MIR type promises but the LLVM lowering hides (a union
        // whose storage is a raw byte blob) cannot be canonicalized here:
        // its bytes were written by the union's own stores, not by this
        // enum's construction.
        if !is_scalar_i1
            && mir_type_contains_i1(ctx, all_field_types[flat])
            && !llvm_type_contains_i1(ctx, llvm_ty)
        {
            return Err(anyhow::anyhow!(
                "enum slot map: `{}` field {} contains bool storage hidden behind a union; cuda-oxide cannot prove those bytes are canonical",
                name,
                flat
            ));
        }

        // Rust bool is a semantic i1 but occupies a complete byte in memory.
        // Never make i1 the struct's physical storage type: give a standalone
        // bool an i8 claim, or reuse the exact i8 carrier/field claim already
        // covering that byte. Construction leaves the bool slotless and the
        // deferred store below explicitly zero-extends i1 -> i8; extraction
        // loads i1 from that byte.
        if is_scalar_i1 {
            let colliding_claims = claims
                .iter()
                .filter(|&&(o, s, _)| offset < o + s && o < offset + size)
                .collect::<Vec<_>>();
            if colliding_claims.is_empty() {
                let byte_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Signless).into();
                claims.push((offset, 1, byte_ty));
                continue;
            }
            let has_exact_i8_storage = colliding_claims.iter().all(|&&(o, s, claim_ty)| {
                o == offset
                    && s == 1
                    && claim_ty
                        .deref(ctx)
                        .downcast_ref::<IntegerType>()
                        .is_some_and(|integer| integer.width() == 8)
            });
            if has_exact_i8_storage {
                continue;
            }
            return Err(anyhow::anyhow!(
                "enum slot map: `{}` scalar bool field {} overlaps storage other than one exact i8 byte; refusing to expose non-canonical bool bits",
                name,
                flat
            ));
        }

        // Aggregates containing bool claim their byte-faithful twin (every
        // i1 leaf becomes its canonical i8 memory byte) and stay slotless:
        // construction canonicalizes the value and writes it at its byte
        // offset; extraction re-loads the original type from the canonical
        // bytes, exactly like the scalar-bool path above.
        // Bool-bearing aggregates remain slotless so their canonical byte view
        // continues through the established spill path. Pointer-only nested
        // aggregates may own a slot because construction/extraction now rebuild
        // them recursively at the semantic/physical boundary.
        let field_gets_slot = !llvm_type_contains_i1(ctx, llvm_ty);

        // Another variant already placed the same storage type at the same
        // position? Then both fields can simply use that claim: variants
        // share bytes, and here they even agree on the type.
        if let Some(ci) = claims
            .iter()
            .position(|&(o, _, t)| o == offset && t == storage_ty)
        {
            if field_gets_slot {
                claim_of_field[flat] = Some(ci);
            }
            continue;
        }
        // The bytes are taken by a different type. Pointer-free values can
        // use the memory fallback below. Pointer-bearing values may do so only
        // when a physical-layout walk proves that every pointer leaf exactly
        // matches an existing pointer slot. This is the common
        // `Option<(usize, &T)>` shape: the niche carrier is the tuple's pointer
        // field. A pointer/non-pointer overlap, address-space mismatch, or
        // additional pointer without a pointer slot still fails closed.
        let colliding_claims = claims
            .iter()
            .filter(|&&(o, s, _)| offset < o + s && o < offset + size)
            .collect::<Vec<_>>();
        if !colliding_claims.is_empty() {
            let has_pointer_overlap = llvm_type_contains_pointer(ctx, storage_ty)
                || colliding_claims
                    .iter()
                    .any(|&&(_, _, claim_ty)| llvm_type_contains_pointer(ctx, claim_ty));
            // Pointer-bearing overlaps must back every pointer leaf with a real
            // `ptr` slot so the memory round-trip preserves provenance. The
            // niche carrier backs the leaf that coincides with it; any further
            // pointer leaf (the extra slice pointer in `split_at_mut_checked`'s
            // `Option<(&mut [T], &mut [T])>`) gets its own fresh `ptr` slot.
            // `ProvenanceLoss` means a pointer punned against non-pointer
            // bits, which stays fail-closed; `OverArrayLeafBound` reports the
            // same bounded rewrite contract as the payload storage gate.
            let extra_pointer_claims = if has_pointer_overlap {
                match analyze_pointer_overlap(ctx, offset, size, storage_ty, &colliding_claims) {
                    Ok(extra) => extra,
                    Err(PointerOverlapRejection::OverArrayLeafBound { required }) => {
                        return Err(anyhow::anyhow!(
                            "enum slot map: `{}` field {} overlaps pointer storage whose arrays are not supported above the bounded rewrite limit; rewrite requires {required} pointer conversions, supported bound is {MAX_ENUM_PAYLOAD_ARRAY_REWRITE_LEAVES}",
                            name,
                            flat
                        ));
                    }
                    Err(PointerOverlapRejection::ProvenanceLoss) => {
                        return Err(anyhow::anyhow!(
                            "enum slot map: `{}` has overlapping pointer and non-identical storage at byte {}; refusing to erase LLVM pointer provenance",
                            name,
                            offset
                        ));
                    }
                }
            } else {
                Vec::new()
            };
            let incoming_is_byte_faithful = llvm_type_is_byte_faithful(ctx, storage_ty);
            let claims_are_byte_faithful = colliding_claims
                .iter()
                .all(|&&(_, _, claim_ty)| llvm_type_is_byte_faithful(ctx, claim_ty));
            if !incoming_is_byte_faithful || !claims_are_byte_faithful {
                return Err(anyhow::anyhow!(
                    "enum slot map: `{}` field {} overlaps non-identical storage but its lowered type is not byte-faithful (for example, it may contain implicit padding); refusing a type-punned store",
                    name,
                    flat
                ));
            }
            // Back each extra pointer leaf with its own `ptr` slot (the field
            // itself stays slotless and round-trips through memory). Added after
            // the byte-faithfulness gate so `colliding_claims`'s borrow of
            // `claims` has ended before we push.
            for leaf in extra_pointer_claims {
                let ptr_ty = llvm_types::PointerType::get(ctx, leaf.address_space).into();
                claims.push((leaf.offset, leaf.size, ptr_ty));
            }
            continue;
        }
        claims.push((offset, size, storage_ty));
        if field_gets_slot {
            claim_of_field[flat] = Some(claims.len() - 1);
        }
    }

    // Phase 2: lay the slots down in byte order, filling every gap (and
    // the tail) so the struct's size is exactly rustc's. One integer per gap
    // where that is layout-neutral, else [N x i8]; see `make_enum_filler_type`.
    let mut emit_order: Vec<usize> = (0..claims.len()).collect();
    emit_order.sort_by_key(|&ci| claims[ci].0);
    let mut llvm_fields: Vec<TypeHandle> = Vec::new();
    let mut slot_of_claim: Vec<u32> = vec![0; claims.len()];
    let mut current_offset: u64 = 0;
    for &ci in &emit_order {
        let (offset, size, llvm_ty) = claims[ci];
        if current_offset < offset {
            let filler =
                make_enum_filler_type(ctx, current_offset, offset - current_offset, abi_align);
            llvm_fields.push(filler);
            current_offset = offset;
        }
        slot_of_claim[ci] = llvm_fields.len() as u32;
        llvm_fields.push(llvm_ty);
        current_offset += size;
    }
    if current_offset < total_size {
        let filler =
            make_enum_filler_type(ctx, current_offset, total_size - current_offset, abi_align);
        llvm_fields.push(filler);
    }

    // Sanity: the struct we just built must be exactly rustc's size.
    // Arrays of enums step by this size, so a mismatch means every
    // element after the first is read from the wrong place. That is a
    // guaranteed miscompile, hence a hard error, not a debug check.
    let (_end, natural_size, natural_align) = natural_struct_layout(ctx, &llvm_fields)
        .ok_or_else(|| anyhow::anyhow!("enum slot map: `{}` has unsupported LLVM layout", name))?;
    if natural_size != total_size {
        return Err(anyhow::anyhow!(
            "enum slot map: `{}` lowered to {} bytes but rustc says {}",
            name,
            natural_size,
            total_size
        ));
    }
    let required_align = abi_align.max(1);
    if natural_align > required_align {
        return Err(anyhow::anyhow!(
            "enum slot map: `{}` lowered with alignment {} but rustc requires {}; explicit over-aligned enum storage is not yet supported",
            name,
            natural_align,
            abi_align
        ));
    }
    if natural_align < required_align {
        // The byte claims alone can under-align the storage, e.g. when the
        // only claim is an i8 niche carrier inside a 4-aligned enum. Raise
        // the struct's alignment with a zero-length anchor field, the same
        // mechanism union storage uses; it occupies no bytes, so every slot
        // index simply shifts by one.
        let anchor_int = IntegerType::get(ctx, (required_align * 8) as u32, Signedness::Signless);
        let anchor: TypeHandle = llvm_types::ArrayType::get(ctx, anchor_int.into(), 0).into();
        llvm_fields.insert(0, anchor);
        for slot in &mut slot_of_claim {
            *slot += 1;
        }
    }

    let field_slots = claim_of_field
        .into_iter()
        .map(|c| c.map(|ci| slot_of_claim[ci]))
        .collect();
    Ok(EnumSlotMap {
        llvm_struct_ty: llvm_types::StructType::get_unnamed(
            ctx,
            (llvm_fields, llvm_types::StructLayout::Unpacked),
        )
        .into(),
        carrier_slot: carrier_claim.map(|claim| slot_of_claim[claim]),
        carrier_llvm_ty: carrier_ty,
        field_slots,
        field_offsets: all_field_offsets,
        field_llvm_types,
    })
}

/// Convert a `MirEnumType` to its LLVM struct representation.
///
/// Thin wrapper over [`build_enum_slot_map`], which explains the layout.
/// Any op that needs an index into the converted enum must take it from
/// the slot map, never compute it by hand.
pub(crate) fn convert_enum_to_llvm(
    ctx: &mut Context,
    ty: TypeHandle,
) -> Result<TypeHandle, anyhow::Error> {
    Ok(build_enum_slot_map(ctx, ty)?.llvm_struct_ty)
}

/// Return an enum name only when the dialect lacks rustc's physical layout.
/// All importer-produced Direct, Niche, Single, and Empty layouts are
/// byte-faithful; legacy `Unknown` values are rejected everywhere rather than
/// receiving a guessed internal representation.
pub(crate) fn enum_unmodeled_in_memory(ctx: &Context, ty: TypeHandle) -> Option<String> {
    let ty_ref = ty.deref(ctx);
    let enum_ty = ty_ref.downcast_ref::<MirEnumType>()?;
    (enum_ty.layout_kind == EnumLayoutKind::Unknown).then(|| enum_ty.name().to_string())
}

/// Search a kernel parameter's type for an enum the host and device
/// would disagree about (see [`enum_unmodeled_in_memory`]).
///
/// The search looks everywhere host data can hide: behind pointers,
/// inside slices and arrays, in struct/tuple fields, and in other enums'
/// payloads. It returns the first offending enum's name.
///
/// Kernel signatures are checked early for a focused diagnostic. Lowering also
/// rejects Unknown layouts for locals, globals, and physical operations, so no
/// guessed representation can escape through an internal-only path.
///
/// `visited` breaks cycles through recursive types (`TypeHandle` is
/// interned, so equality is identity).
pub(crate) fn find_unmodeled_enum_in_abi(
    ctx: &mut Context,
    ty: TypeHandle,
    visited: &mut Vec<TypeHandle>,
) -> Result<Option<String>, anyhow::Error> {
    if visited.contains(&ty) {
        return Ok(None);
    }
    visited.push(ty);

    if let Some(name) = enum_unmodeled_in_memory(ctx, ty) {
        return Ok(Some(name));
    }

    let children: Vec<TypeHandle> = {
        let ty_ref = ty.deref(ctx);
        if let Some(p) = ty_ref.downcast_ref::<dialect_mir::types::MirPtrType>() {
            vec![p.pointee]
        } else if let Some(s) = ty_ref.downcast_ref::<MirSliceType>() {
            vec![s.element_ty]
        } else if let Some(s) = ty_ref.downcast_ref::<MirDisjointSliceType>() {
            let mut nested = vec![s.element_ty];
            nested.extend_from_slice(&s.space_tys);
            nested
        } else if let Some(a) = ty_ref.downcast_ref::<dialect_mir::types::MirArrayType>() {
            vec![a.element_ty]
        } else if let Some(s) = ty_ref.downcast_ref::<MirStructType>() {
            s.field_types.clone()
        } else if let Some(u) = ty_ref.downcast_ref::<MirUnionType>() {
            u.field_types.clone()
        } else if let Some(t) = ty_ref.downcast_ref::<MirTupleType>() {
            t.get_types().to_vec()
        } else if let Some(e) = ty_ref.downcast_ref::<MirEnumType>() {
            e.all_field_types.clone()
        } else {
            vec![]
        }
    };

    for child in children {
        if let Some(name) = find_unmodeled_enum_in_abi(ctx, child, visited)? {
            return Ok(Some(name));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::super::llvm_type_contains_pointer_in_address_space;
    use super::super::test_support::{llvm_int, make_ctx, mir_uint, struct_fields};
    use super::*;
    use dialect_mir::types::{EnumEncoding, EnumVariant, MirArrayType, MirPtrType};
    use pliron::builtin::types::FP32Type;

    #[test]
    fn enum_array_payload_keeps_exact_size_alignment_and_stride() {
        let mut ctx = make_ctx();
        let tag = mir_uint(&mut ctx, 8);
        let element = mir_uint(&mut ctx, 16);
        let payload: TypeHandle = MirArrayType::get(&mut ctx, element, 3).into();
        let enum_ty: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "ArrayPayload".into(),
            tag,
            vec![0, 1],
            vec![
                EnumVariant::unit("Empty".into()),
                EnumVariant::new_with_layout("Data".into(), vec![payload], vec![2], vec![6]),
            ],
            0,
            8,
            2,
        )
        .into();
        let map = build_enum_slot_map(&mut ctx, enum_ty).unwrap();
        assert_eq!(llvm_type_size_align(&ctx, map.llvm_struct_ty), Some((8, 2)));

        let array: TypeHandle = MirArrayType::get(&mut ctx, enum_ty, 5).into();
        let lowered = convert_type(&mut ctx, array).unwrap();
        assert_eq!(llvm_type_size_align(&ctx, lowered), Some((40, 2)));
    }

    #[test]
    fn enum_slot_map_rejects_partial_byte_integer_carriers() {
        let mut ctx = make_ctx();
        let partial_discriminant = mir_uint(&mut ctx, 7);
        let direct: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "PartialDirect".into(),
            partial_discriminant,
            vec![0, 1],
            vec![EnumVariant::unit("A".into()), EnumVariant::unit("B".into())],
            EnumEncoding {
                tag_offset: 0,
                total_size: 1,
                abi_align: 1,
                layout_kind: EnumLayoutKind::Direct,
                carrier_kind: EnumCarrierKind::Integer,
                carrier_width: 7,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into();
        let error = build_enum_slot_map(&mut ctx, direct)
            .err()
            .expect("partial-byte Direct carrier must fail closed");
        assert!(
            error.to_string().contains("whole number of bytes"),
            "{error}"
        );

        let logical = mir_uint(&mut ctx, 8);
        let payload = mir_uint(&mut ctx, 8);
        let niche: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "PartialNiche".into(),
            logical,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout("Some".into(), vec![payload], vec![0], vec![1]),
            ],
            EnumEncoding {
                tag_offset: 0,
                total_size: 1,
                abi_align: 1,
                layout_kind: EnumLayoutKind::Niche,
                carrier_kind: EnumCarrierKind::Integer,
                carrier_width: 7,
                untagged_variant: 1,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into();
        let error = build_enum_slot_map(&mut ctx, niche)
            .err()
            .expect("partial-byte Niche carrier must fail closed");
        assert!(
            error.to_string().contains("whole number of bytes"),
            "{error}"
        );
    }

    #[test]
    fn enum_slot_map_rejects_misaligned_inhabited_payload() {
        let mut ctx = make_ctx();
        let tag = mir_uint(&mut ctx, 8);
        let word = mir_uint(&mut ctx, 32);
        let enum_ty: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "PackedPayload".into(),
            tag,
            vec![0, 1],
            vec![
                EnumVariant::unit("Empty".into()),
                EnumVariant::new_with_layout("Data".into(), vec![word], vec![1], vec![4]),
            ],
            0,
            8,
            4,
        )
        .into();
        let error = build_enum_slot_map(&mut ctx, enum_ty)
            .err()
            .expect("misaligned payload must be rejected");
        assert!(error.to_string().contains("offset 1 is not aligned"));
    }

    #[test]
    fn enum_slot_map_uses_i8_storage_for_nonoverlapping_scalar_bool() {
        let mut ctx = make_ctx();
        let tag = mir_uint(&mut ctx, 32);
        let boolean: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let enum_ty: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "DirectBool".into(),
            tag,
            vec![0, 1],
            vec![
                EnumVariant::new_with_layout("A".into(), vec![boolean], vec![4], vec![1]),
                EnumVariant::unit("B".into()),
            ],
            0,
            8,
            4,
        )
        .into();

        let map = build_enum_slot_map(&mut ctx, enum_ty).unwrap();
        assert_eq!(map.field_slots, vec![None]);
        let fields = struct_fields(&ctx, map.llvm_struct_ty);
        assert_eq!(fields.len(), 3, "tag, canonical bool byte, tail pad");
        assert_eq!(
            fields[1]
                .deref(&ctx)
                .downcast_ref::<IntegerType>()
                .map(IntegerType::width),
            Some(8),
            "Rust bool storage must be an i8 byte, never an LLVM i1 slot"
        );
    }

    #[test]
    fn enum_slot_map_allows_nonoverlapping_aggregate_padding_without_bool() {
        let mut ctx = make_ctx();
        let tag = mir_uint(&mut ctx, 32);
        let byte = mir_uint(&mut ctx, 8);
        let word = mir_uint(&mut ctx, 32);
        let padded: TypeHandle = MirTupleType::get(&mut ctx, vec![byte, word]).into();
        let lowered_padded = convert_type(&mut ctx, padded).unwrap();
        assert!(
            !llvm_type_is_byte_faithful(&ctx, lowered_padded),
            "the LLVM i8/i32 tuple has harmless implicit padding"
        );
        let enum_ty: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "PaddedPayload".into(),
            tag,
            vec![0, 1],
            vec![
                EnumVariant::new_with_layout("Data".into(), vec![padded], vec![4], vec![8]),
                EnumVariant::unit("Empty".into()),
            ],
            0,
            12,
            4,
        )
        .into();

        let map = build_enum_slot_map(&mut ctx, enum_ty).unwrap();
        assert_eq!(map.field_slots, vec![Some(1)]);
        assert_eq!(
            llvm_type_size_align(&ctx, map.llvm_struct_ty),
            Some((12, 4))
        );
    }

    #[test]
    fn enum_slot_map_rejects_nonoverlapping_nested_bool_storage() {
        let mut ctx = make_ctx();
        let tag = mir_uint(&mut ctx, 32);
        let boolean: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let wrapper: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "BoolWrapper".into(),
            vec!["value".into()],
            vec![boolean],
            vec![0],
            vec![0],
            1,
            1,
        )
        .into();
        let enum_ty: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "DirectBoolWrapper".into(),
            tag,
            vec![0, 1],
            vec![
                EnumVariant::new_with_layout("A".into(), vec![wrapper], vec![4], vec![1]),
                EnumVariant::unit("B".into()),
            ],
            0,
            8,
            4,
        )
        .into();

        // The wrapper claims its byte-faithful twin ({i8}) and stays
        // slotless: construction canonicalizes the bool byte, extraction
        // re-loads the original {i1} shape from the canonical byte.
        let map = build_enum_slot_map(&mut ctx, enum_ty)
            .expect("nested bool storage canonicalizes through its byte-faithful twin");
        assert_eq!(map.field_slots, vec![None]);
        assert_eq!(llvm_type_size_align(&ctx, map.llvm_struct_ty), Some((8, 4)));
        let struct_fields: Vec<_> = map
            .llvm_struct_ty
            .deref(&ctx)
            .downcast_ref::<llvm_types::StructType>()
            .unwrap()
            .fields()
            .collect();
        assert!(
            struct_fields
                .iter()
                .all(|field| !llvm_type_contains_i1(&ctx, *field)),
            "enum storage must never contain physical i1"
        );
    }

    #[test]
    fn enum_slot_map_rejects_bool_hidden_by_union_byte_storage() {
        let mut ctx = make_ctx();
        let tag = mir_uint(&mut ctx, 32);
        let byte = mir_uint(&mut ctx, 8);
        let boolean: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let union: TypeHandle = MirUnionType::get(
            &mut ctx,
            "BoolOrByte".into(),
            vec!["flag".into(), "byte".into()],
            vec![boolean, byte],
            1,
            1,
        )
        .into();
        let lowered_union = convert_type(&mut ctx, union).unwrap();
        assert!(
            !llvm_type_contains_i1(&ctx, lowered_union),
            "the union's raw-byte carrier intentionally hides its semantic bool"
        );
        let enum_ty: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "DirectUnionBool".into(),
            tag,
            vec![0, 1],
            vec![
                EnumVariant::new_with_layout("A".into(), vec![union], vec![4], vec![1]),
                EnumVariant::unit("B".into()),
            ],
            0,
            8,
            4,
        )
        .into();

        let error = build_enum_slot_map(&mut ctx, enum_ty)
            .err()
            .expect("a bool hidden by union byte storage must still fail closed");
        assert!(
            error.to_string().contains("hidden behind a union"),
            "{error}"
        );
    }

    #[test]
    fn enum_slot_map_does_not_descend_through_pointer_to_bool() {
        let mut ctx = make_ctx();
        let tag = mir_uint(&mut ctx, 32);
        let boolean: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let pointer: TypeHandle = MirPtrType::get_generic(&mut ctx, boolean, false).into();
        let enum_ty: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "PointerToBool".into(),
            tag,
            vec![0, 1],
            vec![
                EnumVariant::new_with_layout("A".into(), vec![pointer], vec![8], vec![8]),
                EnumVariant::unit("B".into()),
            ],
            0,
            16,
            8,
        )
        .into();

        let map = build_enum_slot_map(&mut ctx, enum_ty).unwrap();
        assert!(map.field_slots[0].is_some());
    }

    #[test]
    fn enum_slot_map_rejects_pointer_nonpointer_overlap_in_either_variant_order() {
        for pointer_first in [false, true] {
            let mut ctx = make_ctx();
            let discr = mir_uint(&mut ctx, 32);
            let u8_ty = mir_uint(&mut ctx, 8);
            let u64_ty = mir_uint(&mut ctx, 64);
            let pointer: TypeHandle = MirPtrType::get_generic(&mut ctx, u8_ty, false).into();
            let (first_name, first_ty, second_name, second_ty) = if pointer_first {
                ("Ptr", pointer, "Bits", u64_ty)
            } else {
                ("Bits", u64_ty, "Ptr", pointer)
            };
            let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
                &mut ctx,
                "PointerOrBits".into(),
                discr,
                vec![0, 1],
                vec![
                    EnumVariant::new_with_layout(
                        first_name.into(),
                        vec![first_ty],
                        vec![8],
                        vec![8],
                    ),
                    EnumVariant::new_with_layout(
                        second_name.into(),
                        vec![second_ty],
                        vec![8],
                        vec![8],
                    ),
                ],
                EnumEncoding {
                    tag_offset: 0,
                    total_size: 16,
                    abi_align: 8,
                    layout_kind: EnumLayoutKind::Direct,
                    carrier_kind: EnumCarrierKind::Integer,
                    carrier_width: 32,
                    variant_inhabited: vec![1, 1],
                    ..EnumEncoding::default()
                },
            )
            .into();
            let error = build_enum_slot_map(&mut ctx, enum_ty)
                .err()
                .expect("pointer overlap must reject");
            assert!(
                error.to_string().contains("pointer provenance"),
                "pointer_first={pointer_first}: {error}"
            );
        }
    }

    #[test]
    fn enum_slot_map_canonicalizes_overlapping_bool_wrapper_storage() {
        let mut ctx = make_ctx();
        let logical = mir_uint(&mut ctx, 8);
        let boolean: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let wrapper: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "BoolWrapper".into(),
            vec!["value".into()],
            vec![boolean],
            vec![0],
            vec![0],
            1,
            1,
        )
        .into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "MaybeBoolWrapper".into(),
            logical,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout("Some".into(), vec![wrapper], vec![0], vec![1]),
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

        // Option<Wrapper(bool)>: the wrapper's byte-faithful twin ({i8})
        // shares the single canonical byte with the i8 niche carrier. The
        // wrapper stays slotless; construction zero-extends its bool.
        let map = build_enum_slot_map(&mut ctx, enum_ty)
            .expect("a bool wrapper shares its canonical byte with the i8 carrier");
        assert_eq!(map.field_slots, vec![None]);
        assert_eq!(llvm_type_size_align(&ctx, map.llvm_struct_ty), Some((1, 1)));
        assert!(
            !llvm_type_contains_i1(&ctx, map.llvm_struct_ty),
            "enum storage must never contain physical i1"
        );
    }

    #[test]
    fn enum_slot_map_canonicalizes_bool_wrapper_overlap_in_either_variant_order() {
        for wrapper_first in [false, true] {
            let mut ctx = make_ctx();
            let tag = mir_uint(&mut ctx, 32);
            let byte = mir_uint(&mut ctx, 8);
            let boolean: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
            let wrapper: TypeHandle = MirStructType::get_with_full_layout(
                &mut ctx,
                "BoolWrapper".into(),
                vec!["value".into()],
                vec![boolean],
                vec![0],
                vec![0],
                1,
                1,
            )
            .into();
            let (first, second) = if wrapper_first {
                (wrapper, byte)
            } else {
                (byte, wrapper)
            };
            let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
                &mut ctx,
                "BoolWrapperOrByte".into(),
                tag,
                vec![0, 1],
                vec![
                    EnumVariant::new_with_layout("First".into(), vec![first], vec![4], vec![1]),
                    EnumVariant::new_with_layout("Second".into(), vec![second], vec![4], vec![1]),
                ],
                EnumEncoding {
                    tag_offset: 0,
                    total_size: 8,
                    abi_align: 4,
                    layout_kind: EnumLayoutKind::Direct,
                    carrier_kind: EnumCarrierKind::Integer,
                    carrier_width: 32,
                    variant_inhabited: vec![1, 1],
                    ..EnumEncoding::default()
                },
            )
            .into();

            // The wrapper's canonical twin ({i8}) and the plain u8 variant
            // share one byte-faithful byte, independent of declaration
            // order; the wrapper stays slotless either way.
            let map = build_enum_slot_map(&mut ctx, enum_ty)
                .expect("canonical bool bytes may share storage with a u8 variant");
            assert_eq!(llvm_type_size_align(&ctx, map.llvm_struct_ty), Some((8, 4)));
            assert!(
                !llvm_type_contains_i1(&ctx, map.llvm_struct_ty),
                "wrapper_first={wrapper_first}: enum storage must never contain physical i1"
            );
            let wrapper_flat = if wrapper_first { 0 } else { 1 };
            assert_eq!(map.field_slots[wrapper_flat], None);
        }
    }

    #[test]
    fn enum_slot_map_backs_pointer_beside_integer_niche_carrier() {
        let mut ctx = make_ctx();
        let logical = mir_uint(&mut ctx, 8);
        let u32_ty = mir_uint(&mut ctx, 32);
        let pointer: TypeHandle = MirPtrType::get_generic(&mut ctx, u32_ty, false).into();
        let payload: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "PointerThenNiche".into(),
            vec!["pointer".into(), "niche".into()],
            vec![pointer, u32_ty],
            vec![0, 1],
            vec![0, 8],
            16,
            8,
        )
        .into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "MaybePointerThenNiche".into(),
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
                carrier_kind: EnumCarrierKind::Integer,
                carrier_width: 32,
                untagged_variant: 1,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into();
        // The pointer at byte 0 never shares bytes with the integer niche
        // carrier at byte 8, so it is representable: it gets its own `ptr` slot
        // and the carrier stays an integer slot. `{ptr@0, i32@8, i32 filler}`
        // -- the trailing 4 bytes are 4-aligned inside an 8-aligned enum, so
        // they lower to one `i32` rather than `[4 x i8]`.
        let map = build_enum_slot_map(&mut ctx, enum_ty).unwrap();
        assert_eq!(map.field_slots, vec![None]);
        let lowered_pointer = convert_type(&mut ctx, pointer).unwrap();
        let carrier: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        assert_eq!(
            struct_fields(&ctx, map.llvm_struct_ty),
            vec![lowered_pointer, carrier, llvm_int(&mut ctx, 32)]
        );
        assert_eq!(
            llvm_type_size_align(&ctx, map.llvm_struct_ty),
            Some((16, 8))
        );
    }

    #[test]
    fn enum_slot_map_keeps_plain_pointer_niche_in_one_pointer_slot() {
        let mut ctx = make_ctx();
        let logical = mir_uint(&mut ctx, 8);
        let u32_ty = mir_uint(&mut ctx, 32);
        let pointer: TypeHandle = MirPtrType::get_generic(&mut ctx, u32_ty, false).into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "MaybePointer".into(),
            logical,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout("Some".into(), vec![pointer], vec![0], vec![8]),
            ],
            EnumEncoding {
                tag_offset: 0,
                total_size: 8,
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
        let map = build_enum_slot_map(&mut ctx, enum_ty).unwrap();
        assert_eq!(map.carrier_slot, Some(0));
        assert_eq!(map.field_slots, vec![Some(0)]);
        assert!(
            struct_fields(&ctx, map.llvm_struct_ty)[0]
                .deref(&ctx)
                .is::<llvm_types::PointerType>()
        );
    }

    #[test]
    fn enum_slot_map_accepts_pointer_first_and_pointer_second_aggregate_niches() {
        for pointer_first in [true, false] {
            let mut ctx = make_ctx();
            let logical = mir_uint(&mut ctx, 64);
            let index = mir_uint(&mut ctx, 64);
            let pointee = mir_uint(&mut ctx, 32);
            let pointer: TypeHandle = MirPtrType::get_generic(&mut ctx, pointee, false).into();
            let payload_types = if pointer_first {
                vec![pointer, index]
            } else {
                vec![index, pointer]
            };
            let payload: TypeHandle = MirTupleType::get(&mut ctx, payload_types).into();
            let tag_offset = if pointer_first { 0 } else { 8 };
            let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
                &mut ctx,
                "MaybeIndexedRef".into(),
                logical,
                vec![0, 1],
                vec![
                    EnumVariant::unit("None".into()),
                    EnumVariant::new_with_layout("Some".into(), vec![payload], vec![0], vec![16]),
                ],
                EnumEncoding {
                    tag_offset,
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

            let map = build_enum_slot_map(&mut ctx, enum_ty).unwrap();
            assert_eq!(map.field_slots, vec![None]);
            assert_eq!(map.carrier_slot, Some(if pointer_first { 0 } else { 1 }));
            assert_eq!(
                llvm_type_size_align(&ctx, map.llvm_struct_ty),
                Some((16, 8))
            );
            let lowered_pointer = convert_type(&mut ctx, pointer).unwrap();
            // The 8 bytes the pointer slot does not claim are 8-aligned inside
            // an 8-aligned enum, so they lower to one `i64`, not `[8 x i8]`.
            let filler = llvm_int(&mut ctx, 64);
            let expected = if pointer_first {
                vec![lowered_pointer, filler]
            } else {
                vec![filler, lowered_pointer]
            };
            assert_eq!(struct_fields(&ctx, map.llvm_struct_ty), expected);
        }
    }

    #[test]
    fn enum_slot_map_backs_each_aggregate_pointer_leaf_with_its_own_slot() {
        let mut ctx = make_ctx();
        let logical = mir_uint(&mut ctx, 64);
        let pointee = mir_uint(&mut ctx, 32);
        let pointer: TypeHandle = MirPtrType::get_generic(&mut ctx, pointee, false).into();
        let payload: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "TwoRefs".into(),
            vec!["first".into(), "second".into()],
            vec![pointer, pointer],
            vec![0, 1],
            vec![0, 8],
            16,
            8,
        )
        .into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "MaybeTwoRefs".into(),
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

        // Both pointer leaves are representable without erasing provenance: the
        // carrier backs the leaf at byte 8, and the leaf at byte 0 gets its own
        // fresh `ptr` slot. The payload stays slotless and round-trips through
        // `{ptr@0, ptr@8}`.
        let map = build_enum_slot_map(&mut ctx, enum_ty).unwrap();
        assert_eq!(map.field_slots, vec![None]);
        let lowered_pointer = convert_type(&mut ctx, pointer).unwrap();
        assert_eq!(
            struct_fields(&ctx, map.llvm_struct_ty),
            vec![lowered_pointer, lowered_pointer]
        );
        assert_eq!(
            llvm_type_size_align(&ctx, map.llvm_struct_ty),
            Some((16, 8))
        );
    }

    #[test]
    fn enum_slot_map_backs_split_at_mut_slice_pair() {
        // Models `Option<(&mut [u32], &mut [u32])>`, the `split_at_mut_checked`
        // return type: two fat slice pointers `{ptr, len}` back to back. The
        // None niche lives in the first data pointer; both data pointers must
        // keep their own `ptr` slot so provenance survives the memory
        // round-trip, with the two `len` fields as raw byte storage.
        let mut ctx = make_ctx();
        let logical = mir_uint(&mut ctx, 64);
        let len = mir_uint(&mut ctx, 64);
        let pointee = mir_uint(&mut ctx, 32);
        let pointer: TypeHandle = MirPtrType::get_generic(&mut ctx, pointee, false).into();
        let payload: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "SlicePair".into(),
            vec![
                "a_ptr".into(),
                "a_len".into(),
                "b_ptr".into(),
                "b_len".into(),
            ],
            vec![pointer, len, pointer, len],
            vec![0, 1, 2, 3],
            vec![0, 8, 16, 24],
            32,
            8,
        )
        .into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "MaybeSlicePair".into(),
            logical,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout("Some".into(), vec![payload], vec![0], vec![32]),
            ],
            EnumEncoding {
                tag_offset: 0,
                total_size: 32,
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

        let map = build_enum_slot_map(&mut ctx, enum_ty).unwrap();
        assert_eq!(map.field_slots, vec![None]);
        let lowered_pointer = convert_type(&mut ctx, pointer).unwrap();
        // Each slice's length sits in the 8 bytes after its pointer, 8-aligned
        // inside an 8-aligned enum, so both lower to one `i64`. This is the
        // `split_at_mut_checked` shape, and the whole point of the widening:
        // `{ptr, i64, ptr, i64}` moves two lengths as two values, where
        // `{ptr, [8 x i8], ptr, [8 x i8]}` moved them as sixteen separate bytes.
        let filler = llvm_int(&mut ctx, 64);
        assert_eq!(
            struct_fields(&ctx, map.llvm_struct_ty),
            vec![lowered_pointer, filler, lowered_pointer, filler]
        );
        assert_eq!(
            llvm_type_size_align(&ctx, map.llvm_struct_ty),
            Some((32, 8))
        );
    }

    #[test]
    fn enum_slot_map_three_field_tuple_follows_recorded_offsets() {
        // rustc lays out `(u32, f32, &T)` with the pointer first in memory:
        // ptr @ 0, u32 @ 8, f32 @ 12, size 16. With those offsets recorded on
        // the tuple, the pointer leaf coincides exactly with the niche
        // carrier at byte 0 and the payload is accepted.
        let mut ctx = make_ctx();
        let logical = mir_uint(&mut ctx, 64);
        let word = mir_uint(&mut ctx, 32);
        let float: TypeHandle = FP32Type::get(&ctx).into();
        let pointee = mir_uint(&mut ctx, 32);
        let pointer: TypeHandle = MirPtrType::get_generic(&mut ctx, pointee, false).into();
        let payload: TypeHandle = MirTupleType::get_with_layout(
            &mut ctx,
            vec![word, float, pointer],
            vec![2, 0, 1],
            vec![8, 12, 0],
            16,
            8,
        )
        .into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "MaybeMixedTuple".into(),
            logical,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout("Some".into(), vec![payload], vec![0], vec![16]),
            ],
            EnumEncoding {
                tag_offset: 0,
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

        let map = build_enum_slot_map(&mut ctx, enum_ty)
            .expect("recorded tuple offsets prove the pointer word matches the carrier");
        assert_eq!(map.carrier_slot, Some(0));
        assert_eq!(
            llvm_type_size_align(&ctx, map.llvm_struct_ty),
            Some((16, 8))
        );

        // The same tuple with offsets that put the u32 over the pointer
        // carrier (ptr @ 8 instead) must still fail closed: integer bytes
        // may not alias pointer storage.
        let mismatched_payload: TypeHandle = MirTupleType::get_with_layout(
            &mut ctx,
            vec![word, float, pointer],
            vec![0, 1, 2],
            vec![0, 4, 8],
            16,
            8,
        )
        .into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "MaybeMixedTupleShifted".into(),
            logical,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout(
                    "Some".into(),
                    vec![mismatched_payload],
                    vec![0],
                    vec![16],
                ),
            ],
            EnumEncoding {
                tag_offset: 0,
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
        let error = build_enum_slot_map(&mut ctx, enum_ty)
            .err()
            .expect("a non-pointer word over the pointer carrier must fail closed");
        assert!(error.to_string().contains("pointer provenance"), "{error}");
    }

    #[test]
    fn enum_slot_map_rejects_shifted_nested_pointer_carrier() {
        let mut ctx = make_ctx();
        let logical = mir_uint(&mut ctx, 64);
        let index = mir_uint(&mut ctx, 64);
        let pointee = mir_uint(&mut ctx, 32);
        let pointer: TypeHandle = MirPtrType::get_generic(&mut ctx, pointee, false).into();
        let payload: TypeHandle = MirTupleType::get(&mut ctx, vec![index, pointer]).into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "ShiftedPointerCarrier".into(),
            logical,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout("Some".into(), vec![payload], vec![0], vec![16]),
            ],
            EnumEncoding {
                tag_offset: 0,
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

        let error = build_enum_slot_map(&mut ctx, enum_ty)
            .err()
            .expect("pointer leaves at different offsets must not be conflated");
        assert!(error.to_string().contains("pointer provenance"), "{error}");
    }

    #[test]
    fn enum_slot_map_rejects_nested_carrier_address_space_mismatch() {
        let mut ctx = make_ctx();
        let logical = mir_uint(&mut ctx, 64);
        let index = mir_uint(&mut ctx, 64);
        let pointee = mir_uint(&mut ctx, 32);
        let generic_pointer: TypeHandle = MirPtrType::get_generic(&mut ctx, pointee, false).into();
        let payload: TypeHandle = MirTupleType::get(&mut ctx, vec![generic_pointer, index]).into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "MismatchedPointerSpace".into(),
            logical,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout("Some".into(), vec![payload], vec![0], vec![16]),
            ],
            EnumEncoding {
                tag_offset: 0,
                total_size: 16,
                abi_align: 8,
                layout_kind: EnumLayoutKind::Niche,
                carrier_kind: EnumCarrierKind::Pointer,
                carrier_width: 64,
                carrier_address_space: llvm_types::address_space::GLOBAL,
                untagged_variant: 1,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into();

        let error = build_enum_slot_map(&mut ctx, enum_ty)
            .err()
            .expect("equal byte ranges in different address spaces must not alias");
        assert!(error.to_string().contains("pointer provenance"), "{error}");
    }

    #[test]
    fn enum_slot_map_genericizes_nonoverlapping_shared_pointer_payload() {
        let mut ctx = make_ctx();
        let discr = mir_uint(&mut ctx, 32);
        let u32_ty = mir_uint(&mut ctx, 32);
        let shared: TypeHandle = MirPtrType::get_shared(&mut ctx, u32_ty, false).into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "SharedPointerPayload".into(),
            discr,
            vec![0, 1],
            vec![
                EnumVariant::unit("Unit".into()),
                EnumVariant::new_with_layout("Ptr".into(), vec![shared], vec![8], vec![8]),
            ],
            EnumEncoding {
                tag_offset: 0,
                total_size: 16,
                abi_align: 8,
                layout_kind: EnumLayoutKind::Direct,
                carrier_kind: EnumCarrierKind::Integer,
                carrier_width: 32,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into();
        let map = build_enum_slot_map(&mut ctx, enum_ty)
            .expect("a direct shared-pointer payload should use generic physical storage");
        let field_slot = map.field_slots[0].expect("shared pointer field should own a slot");
        let stored_ty = map
            .llvm_struct_ty
            .deref(&ctx)
            .downcast_ref::<llvm_types::StructType>()
            .map(|struct_ty| struct_ty.field_type(field_slot as usize))
            .expect("field slot must exist");
        assert_eq!(
            stored_ty
                .deref(&ctx)
                .downcast_ref::<llvm_types::PointerType>()
                .expect("shared pointer storage must remain a pointer")
                .address_space(),
            llvm_types::address_space::GENERIC,
            "enum storage must use a target-stable generic pointer"
        );
        assert_eq!(
            map.field_llvm_types[0]
                .deref(&ctx)
                .downcast_ref::<llvm_types::PointerType>()
                .expect("semantic field must remain a pointer")
                .address_space(),
            llvm_types::address_space::SHARED,
            "payload operations must retain the semantic shared address space"
        );
    }

    #[test]
    fn enum_slot_map_genericizes_nested_shared_pointer_payload() {
        let mut ctx = make_ctx();
        let discr = mir_uint(&mut ctx, 32);
        let u32_ty = mir_uint(&mut ctx, 32);
        let shared: TypeHandle = MirPtrType::get_shared(&mut ctx, u32_ty, false).into();
        let wrapper: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "SharedPointerWrapper".into(),
            vec!["pointer".into()],
            vec![shared],
            vec![0],
            vec![0],
            8,
            8,
        )
        .into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "NestedSharedPointerPayload".into(),
            discr,
            vec![0, 1],
            vec![
                EnumVariant::unit("Unit".into()),
                EnumVariant::new_with_layout("Ptr".into(), vec![wrapper], vec![8], vec![8]),
            ],
            EnumEncoding {
                tag_offset: 0,
                total_size: 16,
                abi_align: 8,
                layout_kind: EnumLayoutKind::Direct,
                carrier_kind: EnumCarrierKind::Integer,
                carrier_width: 32,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into();

        let map = build_enum_slot_map(&mut ctx, enum_ty)
            .expect("a struct-nested shared pointer should use generic physical storage");
        let field_slot = map.field_slots[0].expect("nested payload should own a slot");
        let stored_ty = map
            .llvm_struct_ty
            .deref(&ctx)
            .downcast_ref::<llvm_types::StructType>()
            .map(|struct_ty| struct_ty.field_type(field_slot as usize))
            .expect("field slot must exist");
        assert!(
            llvm_type_contains_pointer_in_address_space(
                &ctx,
                stored_ty,
                llvm_types::address_space::GENERIC
            ),
            "physical nested payload must contain a generic pointer"
        );
        assert!(
            !llvm_type_contains_pointer_in_address_space(
                &ctx,
                stored_ty,
                llvm_types::address_space::SHARED
            ),
            "physical nested payload must not retain the target-dependent AS3 pointer"
        );
        assert!(
            llvm_type_contains_pointer_in_address_space(
                &ctx,
                map.field_llvm_types[0],
                llvm_types::address_space::SHARED
            ),
            "semantic payload type must retain shared address-space semantics"
        );
    }

    #[test]
    fn enum_slot_map_genericizes_bounded_shared_pointer_array_payload() {
        let mut ctx = make_ctx();
        let discr = mir_uint(&mut ctx, 32);
        let u32_ty = mir_uint(&mut ctx, 32);
        let shared: TypeHandle = MirPtrType::get_shared(&mut ctx, u32_ty, false).into();
        let pointers: TypeHandle = MirArrayType::get(&mut ctx, shared, 2).into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "SharedPointerArrayPayload".into(),
            discr,
            vec![0, 1],
            vec![
                EnumVariant::unit("Unit".into()),
                EnumVariant::new_with_layout("Pointers".into(), vec![pointers], vec![8], vec![16]),
            ],
            EnumEncoding {
                tag_offset: 0,
                total_size: 24,
                abi_align: 8,
                layout_kind: EnumLayoutKind::Direct,
                carrier_kind: EnumCarrierKind::Integer,
                carrier_width: 32,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into();

        let map = build_enum_slot_map(&mut ctx, enum_ty)
            .expect("a bounded shared-pointer array should use generic physical storage");
        let field_slot = map.field_slots[0].expect("bounded pointer array should own a slot");
        let stored_ty = map
            .llvm_struct_ty
            .deref(&ctx)
            .downcast_ref::<llvm_types::StructType>()
            .map(|struct_ty| struct_ty.field_type(field_slot as usize))
            .expect("field slot must exist");
        assert!(
            llvm_type_contains_pointer_in_address_space(
                &ctx,
                stored_ty,
                llvm_types::address_space::GENERIC
            ),
            "physical array payload must contain generic pointers"
        );
        assert!(
            !llvm_type_contains_pointer_in_address_space(
                &ctx,
                stored_ty,
                llvm_types::address_space::SHARED
            ),
            "physical array payload must not retain target-dependent AS3 pointers"
        );
        assert!(
            llvm_type_contains_pointer_in_address_space(
                &ctx,
                map.field_llvm_types[0],
                llvm_types::address_space::SHARED
            ),
            "semantic array payload must retain shared address-space semantics"
        );
    }

    #[test]
    fn enum_slot_map_rejects_oversized_shared_pointer_array_payload() {
        let mut ctx = make_ctx();
        let discr = mir_uint(&mut ctx, 32);
        let u32_ty = mir_uint(&mut ctx, 32);
        let shared: TypeHandle = MirPtrType::get_shared(&mut ctx, u32_ty, false).into();
        let count = MAX_ENUM_PAYLOAD_ARRAY_REWRITE_LEAVES + 1;
        let pointers: TypeHandle = MirArrayType::get(&mut ctx, shared, count).into();
        let payload_size = count * 8;
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "OversizedSharedPointerArrayPayload".into(),
            discr,
            vec![0, 1],
            vec![
                EnumVariant::unit("Unit".into()),
                EnumVariant::new_with_layout(
                    "Pointers".into(),
                    vec![pointers],
                    vec![8],
                    vec![payload_size],
                ),
            ],
            EnumEncoding {
                tag_offset: 0,
                total_size: 8 + payload_size,
                abi_align: 8,
                layout_kind: EnumLayoutKind::Direct,
                carrier_kind: EnumCarrierKind::Integer,
                carrier_width: 32,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into();

        let error = build_enum_slot_map(&mut ctx, enum_ty)
            .err()
            .expect("an oversized shared-pointer array must remain fail-closed");
        assert!(
            error
                .to_string()
                .contains("arrays containing shared-memory pointers are not supported"),
            "{error}"
        );
        assert!(
            error.to_string().contains(&format!(
                "supported bound is {MAX_ENUM_PAYLOAD_ARRAY_REWRITE_LEAVES}"
            )),
            "{error}"
        );
    }

    /// A struct payload holding two shared-pointer arrays whose leaves exceed
    /// the bound only in total. The slot map is target-mode agnostic (it is
    /// built once, before the exporter picks the legacy 64-bit or modern
    /// p3:32 data layout), so this one gate covers both output modes.
    fn struct_of_shared_pointer_arrays_over_total_bound(ctx: &mut Context) -> TypeHandle {
        let u32_ty = mir_uint(ctx, 32);
        let shared: TypeHandle = MirPtrType::get_shared(ctx, u32_ty, false).into();
        let first: TypeHandle = MirArrayType::get(ctx, shared, 9).into();
        let second: TypeHandle = MirArrayType::get(ctx, shared, 8).into();
        MirStructType::get_with_full_layout(
            ctx,
            "SharedPointerArrayPair".into(),
            vec!["first".into(), "second".into()],
            vec![first, second],
            vec![],
            vec![0, 72],
            136,
            8,
        )
        .into()
    }

    #[test]
    fn enum_slot_map_rejects_struct_of_shared_pointer_arrays_over_total_bound() {
        // Each array alone (9 and 8 leaves) is within the bound; their total
        // of 17 is not. The payload-root gate must reject the direct layout
        // with the same bound diagnostic as a single oversized array.
        let mut ctx = make_ctx();
        let discr = mir_uint(&mut ctx, 32);
        let payload = struct_of_shared_pointer_arrays_over_total_bound(&mut ctx);
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "StructOfSharedPointerArraysPayload".into(),
            discr,
            vec![0, 1],
            vec![
                EnumVariant::unit("Unit".into()),
                EnumVariant::new_with_layout("Pointers".into(), vec![payload], vec![8], vec![136]),
            ],
            EnumEncoding {
                tag_offset: 0,
                total_size: 144,
                abi_align: 8,
                layout_kind: EnumLayoutKind::Direct,
                carrier_kind: EnumCarrierKind::Integer,
                carrier_width: 32,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into();

        let error = build_enum_slot_map(&mut ctx, enum_ty)
            .err()
            .expect("two in-bound arrays exceeding the bound in total must fail closed");
        assert!(
            error
                .to_string()
                .contains("arrays containing shared-memory pointers are not supported"),
            "{error}"
        );
        assert!(
            error.to_string().contains(&format!(
                "rewrite requires 17 pointer conversions, supported bound is {MAX_ENUM_PAYLOAD_ARRAY_REWRITE_LEAVES}"
            )),
            "{error}"
        );
    }

    #[test]
    fn enum_slot_map_rejects_struct_of_shared_pointer_arrays_in_niche_layout() {
        // The same over-bound payload in a niche layout must report the same
        // bound diagnostic as the direct layout, not a pointer-provenance
        // error from the overlap walk: the payload-root gate runs first in
        // both layouts.
        let mut ctx = make_ctx();
        let logical = mir_uint(&mut ctx, 64);
        let payload = struct_of_shared_pointer_arrays_over_total_bound(&mut ctx);
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "NicheStructOfSharedPointerArrays".into(),
            logical,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout("Some".into(), vec![payload], vec![0], vec![136]),
            ],
            EnumEncoding {
                tag_offset: 0,
                total_size: 136,
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

        let error = build_enum_slot_map(&mut ctx, enum_ty)
            .err()
            .expect("the payload-root bound must also gate niche layouts");
        assert!(
            error.to_string().contains(&format!(
                "rewrite requires 17 pointer conversions, supported bound is {MAX_ENUM_PAYLOAD_ARRAY_REWRITE_LEAVES}"
            )),
            "{error}"
        );
        assert!(
            !error.to_string().contains("pointer provenance"),
            "the bound diagnostic must not be masked by the provenance error: {error}"
        );
    }

    #[test]
    fn enum_slot_map_reports_bound_for_oversized_pointer_array_over_niche_carrier() {
        // Generic pointers pass the shared-pointer storage gate untouched, so
        // an oversized generic-pointer array reaches the overlap walk. Its
        // exhausted expansion budget must surface the bound diagnostic, not
        // the unrelated provenance error.
        let mut ctx = make_ctx();
        let logical = mir_uint(&mut ctx, 64);
        let pointee = mir_uint(&mut ctx, 32);
        let pointer: TypeHandle = MirPtrType::get_generic(&mut ctx, pointee, false).into();
        let count = MAX_ENUM_PAYLOAD_ARRAY_REWRITE_LEAVES + 1;
        let pointers: TypeHandle = MirArrayType::get(&mut ctx, pointer, count).into();
        let payload_size = count * 8;
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "NicheOversizedGenericPointerArray".into(),
            logical,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout(
                    "Some".into(),
                    vec![pointers],
                    vec![0],
                    vec![payload_size],
                ),
            ],
            EnumEncoding {
                tag_offset: 0,
                total_size: payload_size,
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

        let error = build_enum_slot_map(&mut ctx, enum_ty)
            .err()
            .expect("an oversized pointer-array expansion over a niche carrier must fail closed");
        assert!(
            error.to_string().contains(&format!(
                "rewrite requires {count} pointer conversions, supported bound is {MAX_ENUM_PAYLOAD_ARRAY_REWRITE_LEAVES}"
            )),
            "{error}"
        );
        assert!(
            !error.to_string().contains("pointer provenance"),
            "the bound diagnostic must not be masked by the provenance error: {error}"
        );
    }

    #[test]
    fn enum_slot_map_rejects_shared_pointer_vector_payload() {
        let mut ctx = make_ctx();
        let tag = mir_uint(&mut ctx, 32);
        let shared_pointer: TypeHandle =
            llvm_types::PointerType::get(&ctx, llvm_types::address_space::SHARED).into();
        let shared_vector: TypeHandle =
            llvm_types::VectorType::get(&ctx, shared_pointer, 2, llvm_types::VectorTypeKind::Fixed)
                .into();
        let enum_ty: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "SharedPointerVector".into(),
            tag,
            vec![0, 1],
            vec![
                EnumVariant::new_with_layout(
                    "Data".into(),
                    vec![shared_vector],
                    vec![16],
                    vec![16],
                ),
                EnumVariant::unit("Empty".into()),
            ],
            0,
            32,
            16,
        )
        .into();

        let error = build_enum_slot_map(&mut ctx, enum_ty)
            .err()
            .expect("a vector of shared pointers must fail closed");
        assert!(
            error.to_string().contains("shared-memory pointer"),
            "{error}"
        );
    }
}
