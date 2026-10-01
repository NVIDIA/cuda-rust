/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::disallowed_methods)]

use super::*;

// =========================================================================
// Enum layout: converted width per shape + divergent-enum rejection
// =========================================================================

use dialect_mir::types::{EnumVariant, MirEnumType};

/// Build a Direct-tag `MirEnumType` the way the importer does:
/// unsigned tag of `tag_bits`, plus rustc's `total_size`/`abi_align`.
fn make_enum_ty(
    ctx: &mut Context,
    name: &str,
    tag_bits: u32,
    variants: Vec<EnumVariant>,
    total_size: u64,
    abi_align: u64,
) -> TypeHandle {
    let tag_ty: TypeHandle = IntegerType::get(ctx, tag_bits, Signedness::Unsigned).into();
    // Sequential 0..n discriminants: these layout tests only exercise
    // size/width, not value mapping.
    let discriminants: Vec<u64> = (0..variants.len() as u64).collect();
    MirEnumType::get_with_layout(
        ctx,
        name.to_string(),
        tag_ty,
        discriminants,
        variants,
        0, // tag at byte 0, like every shape these tests exercise
        total_size,
        abi_align,
    )
    .into()
}

fn unit_variants(n: usize) -> Vec<EnumVariant> {
    (0..n).map(|i| EnumVariant::unit(format!("V{i}"))).collect()
}

/// Converted enum allocation size must equal rustc's `total_size` for
/// every memory-faithful tag shape: that size is what GEP strides by.
#[test]
fn enum_conversion_strides_by_rustc_size() {
    use crate::convert::types::llvm_type_size_align;

    let mut ctx = make_ctx();

    // #[repr(u32)] fieldless (issue #118 shape): {i32}, 4 bytes.
    let repr_u32 = make_enum_ty(&mut ctx, "ReprU32", 32, unit_variants(4), 4, 4);
    let conv = convert_type(&mut ctx, repr_u32).unwrap();
    assert_eq!(
        llvm_type_size_align(&ctx, conv),
        Some((4, 4)),
        "repr(u32) tag"
    );

    // #[repr(usize)] fieldless: {i64}, 8 bytes.
    let repr_usize = make_enum_ty(&mut ctx, "ReprUsize", 64, unit_variants(4), 8, 8);
    let conv = convert_type(&mut ctx, repr_usize).unwrap();
    assert_eq!(
        llvm_type_size_align(&ctx, conv),
        Some((8, 8)),
        "repr(usize) tag"
    );

    // repr(align(8)) with an i8 tag: the byte claims alone would give
    // alignment 1, so the slot map raises the storage alignment with a
    // zero-length anchor field. Size and alignment must both match
    // rustc, or arrays and by-value ABI uses would be unsound.
    let padded = make_enum_ty(&mut ctx, "Padded", 8, unit_variants(2), 8, 8);
    let conv = convert_type(&mut ctx, padded).unwrap();
    assert_eq!(
        llvm_type_size_align(&ctx, conv),
        Some((8, 8)),
        "repr(align(8)) i8 tag"
    );

    // u8 tag + i64 payload, rustc size 16: the slot map places the
    // payload at its rustc byte offset 8 behind an explicit
    // [7 x i8] filler, making the layout datalayout-independent.
    let i64_payload: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
    let payload = make_enum_ty(
        &mut ctx,
        "OnePayload",
        8,
        vec![
            EnumVariant::new_with_layout("A".to_string(), vec![i64_payload], vec![8], vec![8]),
            EnumVariant::unit("B".to_string()),
        ],
        16,
        8,
    );
    let conv = convert_type(&mut ctx, payload).unwrap();
    let (size, _align) = llvm_type_size_align(&ctx, conv).unwrap();
    assert_eq!(size, 16, "natural layout matches rustc size, no pad");
    let conv_ref = conv.deref(&ctx);
    let struct_ty = conv_ref
        .downcast_ref::<llvm_export::types::StructType>()
        .expect("converted enum is a struct");
    assert_eq!(
        struct_ty.fields().count(),
        3,
        "{{tag, [7 x i8] filler, payload}}: explicit filler to byte 8"
    );
}

/// Multi-payload enum: variants overlap in Rust, and identical
/// (offset, converted type) payloads share one typed slot, so the
/// converted struct is byte-identical to rustc's layout AND every
/// access stays pure SSA (no spill).
#[test]
fn multi_payload_enum_shares_payload_slot() {
    use crate::convert::types::{build_enum_slot_map, llvm_type_size_align};

    let mut ctx = make_ctx();
    let e = make_multi_payload_enum_ty(&mut ctx);
    let map = build_enum_slot_map(&mut ctx, e).unwrap();
    assert_eq!(map.carrier_slot, Some(0));
    assert_eq!(
        map.field_slots,
        vec![Some(1), Some(1)],
        "A.0 and B.0 overlap at byte 4 with the same type: one shared slot"
    );
    assert_eq!(
        llvm_type_size_align(&ctx, map.llvm_struct_ty),
        Some((8, 4)),
        "byte-identical to rustc's 8-byte layout, not the 12-byte concat"
    );
}

/// rustc may place the tag AFTER payload bytes; the slot map must
/// follow the recorded tag_offset, never assume slot 0.
#[test]
fn enum_slot_map_tag_not_at_zero() {
    use crate::convert::types::{build_enum_slot_map, llvm_type_size_align};

    let mut ctx = make_ctx();
    let u64_a: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
    let u64_b: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
    let tag_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
    // enum F { A(u64), B(u64) }: payloads share byte 0, tag at byte 8.
    let ty: TypeHandle = MirEnumType::get_with_layout(
        &mut ctx,
        "TagAtEight".to_string(),
        tag_ty,
        vec![0, 1],
        vec![
            EnumVariant::new_with_layout("A".to_string(), vec![u64_a], vec![0], vec![8]),
            EnumVariant::new_with_layout("B".to_string(), vec![u64_b], vec![0], vec![8]),
        ],
        8,
        16,
        8,
    )
    .into();
    let map = build_enum_slot_map(&mut ctx, ty).unwrap();
    assert_eq!(
        map.field_slots,
        vec![Some(0), Some(0)],
        "payloads share the first slot"
    );
    assert_eq!(
        map.carrier_slot,
        Some(1),
        "tag claims its own slot at byte 8"
    );
    let (size, _align) = llvm_type_size_align(&ctx, map.llvm_struct_ty).unwrap();
    assert_eq!(size, 16, "{{ i64, i8, [7 x i8] }}");
}

/// Multi-payload enum whose variants overlap in Rust: both payloads use
/// bytes 4..8 after the direct `u32` tag, so the complete value is eight
/// bytes. Mimics `#[repr(u32)] enum E { A(u32), B(u32) }`.
fn make_multi_payload_enum_ty(ctx: &mut Context) -> TypeHandle {
    let i32_a: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
    let i32_b: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
    make_enum_ty(
        ctx,
        "MultiPayload",
        32,
        vec![
            EnumVariant::new_with_layout("A".to_string(), vec![i32_a], vec![4], vec![4]),
            EnumVariant::new_with_layout("B".to_string(), vec![i32_b], vec![4], vec![4]),
        ],
        8,
        4,
    )
}

/// Device-local GEP + load must use the same exact eight-byte rustc layout
/// as every other enum path. This protects pointer stride in issue #131's
/// in-kernel `[E; 4]` arrays; the sibling kernel-parameter test proves the
/// same representation is also valid at the host/device boundary.
#[test]
fn device_local_multi_payload_enum_gep_and_load_lower() {
    let mut ctx = make_ctx();
    let enum_ty = make_multi_payload_enum_ty(&mut ctx);
    let i64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signless).into();
    let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, enum_ty, true);

    let (module_ptr, block) = build_kernel(&mut ctx, vec![mir_ptr_ty.into(), i64_ty], vec![]);
    let ptr_val = block.deref(&ctx).get_argument(0);
    let off_val = block.deref(&ctx).get_argument(1);

    let off_op = Operation::new(
        &mut ctx,
        mir::MirPtrOffsetOp::get_concrete_op_info(),
        vec![mir_ptr_ty.into()],
        vec![ptr_val, off_val],
        vec![],
        0,
    );
    off_op.insert_at_back(block, &ctx);
    let elem_ptr = off_op.deref(&ctx).get_result(0);

    let load_op = Operation::new(
        &mut ctx,
        mir::MirLoadOp::get_concrete_op_info(),
        vec![enum_ty],
        vec![elem_ptr],
        vec![],
        0,
    );
    load_op.insert_at_back(block, &ctx);
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr)
        .expect("device-local rustc-layout enum GEP + load must lower");
}

/// A kernel parameter may carry this enum because the device slot map is
/// byte-identical to rustc's overlapped host layout.
#[test]
fn kernel_param_accepts_multi_payload_enum() {
    use pliron::builtin::attributes::StringAttr;

    let mut ctx = make_ctx();
    let enum_ty = make_multi_payload_enum_ty(&mut ctx);
    let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, enum_ty, false);

    let (module_ptr, block) = build_kernel(&mut ctx, vec![mir_ptr_ty.into()], vec![]);
    append_mir_return(&mut ctx, block, vec![]);

    // Mark the function as a GPU kernel the way the importer does.
    {
        let module_block = module_ptr
            .deref(&ctx)
            .get_region(0)
            .deref(&ctx)
            .iter(&ctx)
            .next()
            .unwrap();
        let func_op = module_block.deref(&ctx).iter(&ctx).next().unwrap();
        let kernel_attr = StringAttr::new("true".to_string());
        let key: pliron::identifier::Identifier = "gpu_kernel".try_into().unwrap();
        func_op.deref_mut(&ctx).attributes.set(key, kernel_attr);
    }

    // The slot map lowers MultiPayload byte-identically to rustc's
    // layout ({ i32, i32 }, 8 bytes), so the kernel ABI accepts it.
    crate::lower_mir_to_llvm(&mut ctx, module_ptr)
        .expect("multi-payload enum kernel param must lower");
}

/// A legacy enum without physical rustc metadata must be rejected at the
/// kernel boundary. Importer-produced niche layouts are now known and are
/// accepted; this fixture deliberately constructs `Unknown` metadata.
#[test]
fn kernel_param_rejects_enum_with_unknown_layout() {
    use pliron::builtin::attributes::StringAttr;

    let mut ctx = make_ctx();
    // MirEnumType::get with no layout is the legacy Unknown form.
    let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
    let pointee = MirPtrType::get_generic(&mut ctx, i32_ty, false);
    let tag_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
    let niched: TypeHandle = MirEnumType::get(
        &mut ctx,
        "Option".to_string(),
        tag_ty,
        vec![0, 1],
        vec![
            EnumVariant::unit("None".to_string()),
            EnumVariant::new("Some".to_string(), vec![pointee.into()]),
        ],
    )
    .into();
    let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, niched, false);

    let (module_ptr, block) = build_kernel(&mut ctx, vec![mir_ptr_ty.into()], vec![]);
    append_mir_return(&mut ctx, block, vec![]);

    {
        let module_block = module_ptr
            .deref(&ctx)
            .get_region(0)
            .deref(&ctx)
            .iter(&ctx)
            .next()
            .unwrap();
        let func_op = module_block.deref(&ctx).iter(&ctx).next().unwrap();
        let kernel_attr = StringAttr::new("true".to_string());
        let key: pliron::identifier::Identifier = "gpu_kernel".try_into().unwrap();
        func_op.deref_mut(&ctx).attributes.set(key, kernel_attr);
    }

    let err = crate::lower_mir_to_llvm(&mut ctx, module_ptr)
        .expect_err("unknown-layout enum kernel param must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("Option") && msg.contains("kernel boundary"),
        "error must name the enum and the kernel boundary, got: {msg}"
    );
    assert!(msg.contains("unknown physical rustc layout"));
}
