/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Struct slot mapping: `StructLayoutInfo`, `StructSlotMap`, and
//! `build_struct_slot_map` (single source of truth, issue #128).

use dialect_mir::types::{MirStructType, MirTupleType};
use llvm_export::types as llvm_types;
use pliron::context::Context;
use pliron::r#type::TypeHandle;

use super::layout::{make_padding_type, mir_stored_size, natural_layout_walk};
use super::{convert_type, is_zero_sized_type, llvm_type_size_align};

// =============================================================================
// Struct Slot Mapping (single source of truth, issue #128)
// =============================================================================

/// Declaration-order layout facts for one MIR aggregate, in the exact form
/// [`build_struct_slot_map`] consumes.
///
/// Extracting this owned carrier first (and dropping the `Ref` returned by
/// `Ptr::deref`) keeps the borrow checker happy: the slot-map build needs
/// `&mut Context` for type interning.
pub(crate) struct StructLayoutInfo {
    /// Field types in declaration order.
    pub field_types: Vec<TypeHandle>,
    /// Memory order: `mem_to_decl[mem_idx] = decl_idx`. Always full length
    /// (identity when rustc did not reorder).
    pub mem_to_decl: Vec<usize>,
    /// Byte offset of each field in declaration order; empty when rustc
    /// layout is unknown.
    pub field_offsets: Vec<u64>,
    /// Total size in bytes including trailing padding; 0 when unknown.
    pub total_size: u64,
}

impl StructLayoutInfo {
    /// Layout facts of a `MirStructType`.
    pub(crate) fn of_struct(s: &MirStructType) -> Self {
        StructLayoutInfo {
            field_types: s.field_types.clone(),
            mem_to_decl: s.memory_order(),
            field_offsets: s.field_offsets().to_vec(),
            total_size: s.total_size(),
        }
    }

    /// Layout facts of a `MirTupleType`.
    ///
    /// Tuples translated from a rustc type carry rustc's exact layout
    /// (offsets, memory order, size), which is consumed here identically to
    /// structs, so reordered tuples like `(u32, &T)` lower byte-correctly.
    /// Only synthetic layout-less tuples (the unit tuple, hand-built test
    /// types) fall back to LLVM natural layout.
    pub(crate) fn of_tuple(t: &MirTupleType) -> Self {
        StructLayoutInfo {
            field_types: t.get_types().to_vec(),
            mem_to_decl: t.memory_order(),
            field_offsets: t.field_offsets().to_vec(),
            total_size: t.total_size(),
        }
    }
}

/// One lowered LLVM struct plus the value-level slot mapping into it.
///
/// [`build_struct_slot_map`] produces the struct type and the index map in
/// the same walk, so every op that indexes into the struct (`insertvalue`,
/// `extractvalue`, GEP, call-boundary flatten/reconstruct) shares the type
/// converter's view of where each field landed. Computing the indices
/// separately is how the issue #128 class of bug (indices that ignore the
/// `[N x i8]` padding slots) happened.
pub(crate) struct StructSlotMap {
    /// The final LLVM struct type, including any `[N x i8]` padding slots.
    pub llvm_struct_ty: TypeHandle,
    /// `decl_to_llvm[decl_idx]` = LLVM slot of that declaration-order field;
    /// `None` when the field is zero-sized and was stripped.
    pub decl_to_llvm: Vec<Option<u32>>,
    /// Converted LLVM type of each declaration-order field (ZSTs included).
    pub field_llvm_types: Vec<TypeHandle>,
    /// Natural (non-packed) LLVM byte offset of every slot of
    /// `llvm_struct_ty`, padding slots included; `None` when some field type
    /// cannot be sized ([`llvm_type_size_align`] declined).
    pub natural_slot_offsets: Option<Vec<u64>>,
    /// True when rustc's recorded byte layout cannot be represented by a
    /// naturally laid-out LLVM struct: some field's natural slot offset differs
    /// from rustc's byte offset, or the natural struct size differs from
    /// rustc's total size. `repr(packed)` is the canonical case. Address-path
    /// consumers retain this signal and the natural offsets so the #859
    /// byte-GEP fallback stays stable for packed field projections.
    pub layout_diverges: bool,
    /// True when the selected LLVM struct representation reproduces rustc's
    /// recorded field offsets and total size for by-value movement. Natural
    /// layouts are faithful when they do not diverge; divergent layouts are
    /// faithful only when a sequential LLVM packed struct can express them.
    /// Overlapping/union-like legacy struct models therefore remain false.
    pub by_value_layout_faithful: bool,
}

/// Whether a `MirStructType` value lowers to a byte-faithful LLVM struct:
/// either its natural LLVM layout already matches rustc's offsets and total
/// size, or the packed-with-explicit-padding representation reproduces them
/// exactly.
///
/// Public only as a coupling oracle for mir-importer: its constant-promotion
/// gate must never admit a struct layout whose converted storage falls back
/// to a divergent natural layout, because a promoted constant's byte image
/// would then disagree with every typed read through the converted struct.
/// mir-importer asserts agreement against this function in its tests.
pub fn struct_value_lowering_is_byte_faithful(
    ctx: &mut Context,
    struct_ty: TypeHandle,
) -> Result<bool, anyhow::Error> {
    let layout = {
        let ty_ref = struct_ty.deref(ctx);
        let mir_struct = ty_ref
            .downcast_ref::<MirStructType>()
            .ok_or_else(|| anyhow::anyhow!("expected a MirStructType"))?;
        StructLayoutInfo::of_struct(mir_struct)
    };
    Ok(build_struct_slot_map(ctx, &layout)?.by_value_layout_faithful)
}

/// Lower a struct/tuple layout to its LLVM struct type and slot map.
///
/// When rustc layout is present (`field_offsets` non-empty and
/// `total_size > 0`), fields are placed at their exact byte offsets with
/// explicit `[N x i8]` padding slots in between, plus a trailing pad up to
/// `total_size`. This makes the layout independent of LLVM's datalayout
/// and so ABI-identical to what rustc computed on the host. For
/// `struct Extreme { a: u8, b: i128 }` where rustc puts `b` at offset 0
/// and `a` at offset 16 with total size 32, we build:
///
/// ```text
/// { i128, i8, [15 x i8] }   ; slots:  b = 0, a = 1, pad = 2
/// ```
///
/// Without rustc layout, fields are emitted in memory order with no
/// padding. On both paths zero-sized fields (e.g. `PhantomData`) are
/// stripped, because NVPTX rejects empty types; stripped fields get
/// `None` in `decl_to_llvm`.
///
/// Malformed layout metadata (a `mem_to_decl` that is not a permutation,
/// or an offsets vector of the wrong length) is rejected loudly: guessing
/// here would scramble every downstream field access.
pub(crate) fn build_struct_slot_map(
    ctx: &mut Context,
    layout: &StructLayoutInfo,
) -> Result<StructSlotMap, anyhow::Error> {
    let num_fields = layout.field_types.len();

    if layout.mem_to_decl.len() != num_fields {
        return Err(anyhow::anyhow!(
            "struct slot map: memory order has {} entries but the struct has {} fields",
            layout.mem_to_decl.len(),
            num_fields
        ));
    }
    let mut seen = vec![false; num_fields];
    for &decl_idx in &layout.mem_to_decl {
        if decl_idx >= num_fields || seen[decl_idx] {
            return Err(anyhow::anyhow!(
                "struct slot map: memory order {:?} is not a permutation of 0..{}",
                layout.mem_to_decl,
                num_fields
            ));
        }
        seen[decl_idx] = true;
    }
    let has_explicit_layout = !layout.field_offsets.is_empty() && layout.total_size > 0;
    if has_explicit_layout && layout.field_offsets.len() != num_fields {
        return Err(anyhow::anyhow!(
            "struct slot map: {} field offsets for {} fields",
            layout.field_offsets.len(),
            num_fields
        ));
    }

    // Convert every field up front, in declaration order.
    let mut field_llvm_types = Vec::with_capacity(num_fields);
    for &field_ty in &layout.field_types {
        field_llvm_types.push(convert_type(ctx, field_ty)?);
    }

    let mut llvm_fields: Vec<TypeHandle> = Vec::new();
    let mut decl_to_llvm: Vec<Option<u32>> = vec![None; num_fields];
    let mut current_offset: u64 = 0;

    // Place fields in memory order.
    for &decl_idx in &layout.mem_to_decl {
        let llvm_ty = field_llvm_types[decl_idx];

        // ZST fields are stripped: no slot, no offset advance (rustc gives
        // them size 0).
        if is_zero_sized_type(ctx, llvm_ty) {
            continue;
        }

        if has_explicit_layout {
            // Insert padding if needed to reach the rustc field offset.
            let target_offset = layout.field_offsets[decl_idx];
            if current_offset < target_offset {
                let padding_ty = make_padding_type(ctx, target_offset - current_offset);
                llvm_fields.push(padding_ty);
                current_offset = target_offset;
            }
        }

        decl_to_llvm[decl_idx] = Some(llvm_fields.len() as u32);
        llvm_fields.push(llvm_ty);

        if has_explicit_layout {
            // Offset advance is exact or an error, never guessed. Prefer
            // rustc's stored size for the field: nested aggregates carry
            // interior/trailing padding the converted type cannot always
            // reproduce, and a wrong advance here either forces interior
            // padding where rustc has none or overshoots the next field's
            // offset. Scalars and other rustc-silent types fall back to the
            // LLVM natural size.
            let advance = match mir_stored_size(ctx, layout.field_types[decl_idx]) {
                Some(size) => size,
                None => llvm_type_size_align(ctx, llvm_ty)
                    .map(|(size, _)| size)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "struct slot map: field {decl_idx} lowers to `{}`, which has no \
                             exact size; refusing to guess the next field offset",
                            llvm_ty.deref(ctx).disp(ctx)
                        )
                    })?,
            };
            current_offset += advance;
        }
    }

    // Add trailing padding to reach total_size.
    if has_explicit_layout && current_offset < layout.total_size {
        let padding_ty = make_padding_type(ctx, layout.total_size - current_offset);
        llvm_fields.push(padding_ty);
    }

    // The explicit `[N x i8]` gaps above can only ADD bytes; they cannot make
    // LLVM place a slot earlier than its natural alignment allows. So when
    // rustc's layout is tighter than natural (repr(packed)), the struct built
    // here is a lie at the byte level, and every consumer needs to know.
    // Unsizable field types leave no verdict: consumers that need the
    // comparison (field_addr with a recorded rustc offset) fail closed at
    // their own sites.
    let walk = natural_layout_walk(ctx, &llvm_fields);
    let natural_slot_offsets = walk.as_ref().map(|(offsets, _, _)| offsets.clone());
    let mut layout_diverges = false;
    if has_explicit_layout && let Some((offsets, end, align)) = &walk {
        for (decl_idx, slot) in decl_to_llvm.iter().enumerate() {
            if let Some(slot) = slot
                && offsets[*slot as usize] != layout.field_offsets[decl_idx]
            {
                layout_diverges = true;
            }
        }
        let natural_size = end.div_ceil(*align) * *align;
        if natural_size != layout.total_size {
            layout_diverges = true;
        }
    }

    // A packed LLVM struct is sequential with alignment 1. That faithfully
    // represents repr(packed) only when the sequential packed offsets and
    // final byte count exactly match rustc's metadata. A divergent layout can
    // also be an old union-like/overlapping model; selecting Packed for such a
    // shape would merely turn one incorrect sequential layout into another.
    let packed_walk = if layout_diverges && has_explicit_layout {
        let mut offsets = Vec::with_capacity(llvm_fields.len());
        let mut end = 0u64;
        let mut sizeable = true;
        for &field in &llvm_fields {
            offsets.push(end);
            let Some((field_size, _)) = llvm_type_size_align(ctx, field) else {
                sizeable = false;
                break;
            };
            let Some(next) = end.checked_add(field_size) else {
                sizeable = false;
                break;
            };
            end = next;
        }
        sizeable.then_some((offsets, end))
    } else {
        None
    };
    let packed_representable = packed_walk.as_ref().is_some_and(|(offsets, end)| {
        *end == layout.total_size
            && decl_to_llvm
                .iter()
                .enumerate()
                .all(|(decl_idx, slot)| match slot {
                    Some(slot) => offsets[*slot as usize] == layout.field_offsets[decl_idx],
                    None => true,
                })
    });

    let struct_layout = if packed_representable {
        llvm_types::StructLayout::Packed
    } else {
        llvm_types::StructLayout::Unpacked
    };
    let llvm_struct_ty: TypeHandle =
        llvm_types::StructType::get_unnamed(ctx, (llvm_fields, struct_layout)).into();
    let by_value_layout_faithful = !layout_diverges || packed_representable;

    Ok(StructSlotMap {
        llvm_struct_ty,
        decl_to_llvm,
        field_llvm_types,
        natural_slot_offsets,
        layout_diverges,
        by_value_layout_faithful,
    })
}

#[cfg(test)]
mod tests {
    //! Hardware-free unit tests for [`build_struct_slot_map`]: the slot map
    //! and the LLVM struct type are produced by the same walk, so these
    //! tests pin down both for the layout shapes from issue #128.

    use super::super::test_support::{llvm_int, make_ctx, mir_uint, mir_zst, pad, struct_fields};
    use super::*;
    use dialect_mir::types::{EnumVariant, MirEnumType};

    #[test]
    fn slot_map_reorder_only() {
        let mut ctx = make_ctx();
        // struct { a: u8, b: u64 }, memory order [b, a], no rustc offsets.
        let a = mir_uint(&mut ctx, 8);
        let b = mir_uint(&mut ctx, 64);
        let layout = StructLayoutInfo {
            field_types: vec![a, b],
            mem_to_decl: vec![1, 0],
            field_offsets: vec![],
            total_size: 0,
        };
        let map = build_struct_slot_map(&mut ctx, &layout).unwrap();

        assert_eq!(map.decl_to_llvm, vec![Some(1), Some(0)]);
        let i8s = llvm_int(&mut ctx, 8);
        let i64s = llvm_int(&mut ctx, 64);
        assert_eq!(struct_fields(&ctx, map.llvm_struct_ty), vec![i64s, i8s]);
    }

    #[test]
    fn slot_map_padding_only() {
        let mut ctx = make_ctx();
        // struct { a: u8 @ 0, b: u64 @ 8 }, declaration order == memory
        // order, size 16: lowers to { i8, [7 x i8], i64 }. The pad consumes
        // slot 1, so b lands at slot 2 (the issue #128 sites used 1).
        let a = mir_uint(&mut ctx, 8);
        let b = mir_uint(&mut ctx, 64);
        let layout = StructLayoutInfo {
            field_types: vec![a, b],
            mem_to_decl: vec![0, 1],
            field_offsets: vec![0, 8],
            total_size: 16,
        };
        let map = build_struct_slot_map(&mut ctx, &layout).unwrap();

        assert_eq!(map.decl_to_llvm, vec![Some(0), Some(2)]);
        let i8s = llvm_int(&mut ctx, 8);
        let i64s = llvm_int(&mut ctx, 64);
        let pad7 = pad(&mut ctx, 7);
        assert_eq!(
            struct_fields(&ctx, map.llvm_struct_ty),
            vec![i8s, pad7, i64s]
        );
    }

    #[test]
    fn slot_map_uses_packed_layout_when_natural_offsets_diverge() {
        let mut ctx = make_ctx();
        let tag = mir_uint(&mut ctx, 8);
        let value = mir_uint(&mut ctx, 32);
        let layout = StructLayoutInfo {
            field_types: vec![tag, value],
            mem_to_decl: vec![0, 1],
            field_offsets: vec![0, 1],
            total_size: 5,
        };

        let map = build_struct_slot_map(&mut ctx, &layout).unwrap();

        assert!(map.layout_diverges);
        assert_eq!(map.decl_to_llvm, vec![Some(0), Some(1)]);
        let llvm_struct_ty_ref = map.llvm_struct_ty.deref(&ctx);
        let struct_ty = llvm_struct_ty_ref
            .downcast_ref::<llvm_types::StructType>()
            .expect("slot map must produce an LLVM struct");
        assert_eq!(struct_ty.layout(), llvm_types::StructLayout::Packed);
        assert_eq!(llvm_type_size_align(&ctx, map.llvm_struct_ty), Some((5, 1)));
    }

    #[test]
    fn slot_map_packed_two_keeps_explicit_padding_slot() {
        let mut ctx = make_ctx();
        let tag = mir_uint(&mut ctx, 8);
        let value = mir_uint(&mut ctx, 32);
        let layout = StructLayoutInfo {
            field_types: vec![tag, value],
            mem_to_decl: vec![0, 1],
            field_offsets: vec![0, 2],
            total_size: 6,
        };

        let map = build_struct_slot_map(&mut ctx, &layout).unwrap();

        assert!(map.layout_diverges);
        assert_eq!(map.decl_to_llvm, vec![Some(0), Some(2)]);
        let i8s = llvm_int(&mut ctx, 8);
        let i32s = llvm_int(&mut ctx, 32);
        let pad1 = pad(&mut ctx, 1);
        assert_eq!(
            struct_fields(&ctx, map.llvm_struct_ty),
            vec![i8s, pad1, i32s]
        );
        let llvm_struct_ty_ref = map.llvm_struct_ty.deref(&ctx);
        let struct_ty = llvm_struct_ty_ref
            .downcast_ref::<llvm_types::StructType>()
            .expect("slot map must produce an LLVM struct");
        assert_eq!(struct_ty.layout(), llvm_types::StructLayout::Packed);
        assert_eq!(llvm_type_size_align(&ctx, map.llvm_struct_ty), Some((6, 1)));
    }

    #[test]
    fn slot_map_reorder_plus_padding() {
        let mut ctx = make_ctx();
        // struct { a: u8 @ 8, b: u64 @ 0 }, memory order [b, a], size 16:
        // lowers to { i64, i8, [7 x i8] } with a trailing pad.
        let a = mir_uint(&mut ctx, 8);
        let b = mir_uint(&mut ctx, 64);
        let layout = StructLayoutInfo {
            field_types: vec![a, b],
            mem_to_decl: vec![1, 0],
            field_offsets: vec![8, 0],
            total_size: 16,
        };
        let map = build_struct_slot_map(&mut ctx, &layout).unwrap();

        assert_eq!(map.decl_to_llvm, vec![Some(1), Some(0)]);
        let i8s = llvm_int(&mut ctx, 8);
        let i64s = llvm_int(&mut ctx, 64);
        let pad7 = pad(&mut ctx, 7);
        assert_eq!(
            struct_fields(&ctx, map.llvm_struct_ty),
            vec![i64s, i8s, pad7]
        );
    }

    #[test]
    fn slot_map_zst_interleaving() {
        let mut ctx = make_ctx();
        // struct { a: u32 @ 0, z: PhantomData @ 4, b: u32 @ 4 }, size 8.
        // The ZST is stripped (no slot, no pad split): { i32, i32 }.
        let a = mir_uint(&mut ctx, 32);
        let z = mir_zst(&mut ctx);
        let b = mir_uint(&mut ctx, 32);
        let layout = StructLayoutInfo {
            field_types: vec![a, z, b],
            mem_to_decl: vec![0, 1, 2],
            field_offsets: vec![0, 4, 4],
            total_size: 8,
        };
        let map = build_struct_slot_map(&mut ctx, &layout).unwrap();

        assert_eq!(map.decl_to_llvm, vec![Some(0), None, Some(1)]);
        let i32s = llvm_int(&mut ctx, 32);
        assert_eq!(struct_fields(&ctx, map.llvm_struct_ty), vec![i32s, i32s]);
    }

    #[test]
    fn slot_map_issue128_arena_shape() {
        let mut ctx = make_ctx();
        // The exact shape from issue #128 (examples/struct_field_layout):
        //
        //   enum Layout { Aos, Soa, AoSoA(u32) }          // -> { i8, i32 }
        //   struct Arena { layout: Layout, cap: u32, stride: u32, big: u64 }
        //
        // rustc layout: layout @ 0 (8 bytes), big @ 8, cap @ 16,
        // stride @ 20, size 24. The enum now carries its own explicit
        // internal padding, so the OUTER struct needs no extra padding slot:
        //
        //   { { i8, [3 x i8], i32 }, i64, i32, i32 }
        //     layout=0                  big=1 cap=2 stride=3
        let discr = mir_uint(&mut ctx, 8);
        let payload = mir_uint(&mut ctx, 32);
        let layout_enum: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "Layout".into(),
            discr,
            vec![0, 1, 2],
            vec![
                EnumVariant::unit("Aos".into()),
                EnumVariant::unit("Soa".into()),
                EnumVariant::new_with_layout("AoSoA".into(), vec![payload], vec![4], vec![4]),
            ],
            0,
            8,
            4,
        )
        .into();
        let cap = mir_uint(&mut ctx, 32);
        let stride = mir_uint(&mut ctx, 32);
        let big = mir_uint(&mut ctx, 64);

        let layout = StructLayoutInfo {
            field_types: vec![layout_enum, cap, stride, big],
            mem_to_decl: vec![0, 3, 1, 2],
            field_offsets: vec![0, 16, 20, 8],
            total_size: 24,
        };
        let map = build_struct_slot_map(&mut ctx, &layout).unwrap();

        assert_eq!(
            map.decl_to_llvm,
            vec![Some(0), Some(2), Some(3), Some(1)],
            "the enum's internal pad must not create an outer struct slot"
        );

        let i8s = llvm_int(&mut ctx, 8);
        let i32s = llvm_int(&mut ctx, 32);
        let i64s = llvm_int(&mut ctx, 64);
        let enum_pad3 = pad(&mut ctx, 3);
        let enum_llvm: TypeHandle = llvm_types::StructType::get_unnamed(
            &ctx,
            (
                vec![i8s, enum_pad3, i32s],
                llvm_types::StructLayout::Unpacked,
            ),
        )
        .into();
        assert_eq!(
            struct_fields(&ctx, map.llvm_struct_ty),
            vec![enum_llvm, i64s, i32s, i32s]
        );
    }

    #[test]
    fn slot_map_nested_struct_uses_stored_size() {
        let mut ctx = make_ctx();
        // Inner struct whose stored rustc size (16) exceeds the sum of its
        // converted LLVM field sizes (i8 + i64 = 9, no offsets stored).
        // The outer walk must advance by the stored 16, reaching the next
        // field's offset exactly: NO interior pad before it.
        let x = mir_uint(&mut ctx, 8);
        let y = mir_uint(&mut ctx, 64);
        let inner: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "Inner".into(),
            vec!["x".into(), "y".into()],
            vec![x, y],
            vec![],
            vec![],
            16,
            0,
        )
        .into();
        let c = mir_uint(&mut ctx, 8);

        let layout = StructLayoutInfo {
            field_types: vec![inner, c],
            mem_to_decl: vec![0, 1],
            field_offsets: vec![0, 16],
            total_size: 24,
        };
        let map = build_struct_slot_map(&mut ctx, &layout).unwrap();

        // inner = slot 0, c = slot 1 (adjacent), trailing [7 x i8] pad.
        assert_eq!(map.decl_to_llvm, vec![Some(0), Some(1)]);
        let fields = struct_fields(&ctx, map.llvm_struct_ty);
        assert_eq!(fields.len(), 3, "exactly one (trailing) pad slot");
        let pad7 = pad(&mut ctx, 7);
        assert_eq!(fields[2], pad7);
    }

    #[test]
    fn slot_map_rejects_malformed_memory_order() {
        let mut ctx = make_ctx();
        let a = mir_uint(&mut ctx, 8);
        let b = mir_uint(&mut ctx, 64);

        // Not a permutation: decl index 0 appears twice.
        let dup = StructLayoutInfo {
            field_types: vec![a, b],
            mem_to_decl: vec![0, 0],
            field_offsets: vec![],
            total_size: 0,
        };
        assert!(build_struct_slot_map(&mut ctx, &dup).is_err());

        // Wrong length.
        let short = StructLayoutInfo {
            field_types: vec![a, b],
            mem_to_decl: vec![0],
            field_offsets: vec![],
            total_size: 0,
        };
        assert!(build_struct_slot_map(&mut ctx, &short).is_err());

        // Offsets vector length mismatch (with explicit layout engaged).
        let bad_offsets = StructLayoutInfo {
            field_types: vec![a, b],
            mem_to_decl: vec![0, 1],
            field_offsets: vec![0],
            total_size: 16,
        };
        assert!(build_struct_slot_map(&mut ctx, &bad_offsets).is_err());
    }
}
