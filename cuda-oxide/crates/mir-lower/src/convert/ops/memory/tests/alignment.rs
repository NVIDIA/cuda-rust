/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::disallowed_methods)]

use super::*;

/// Lower `mir.load (mir.field_addr %p, field_index)` for a struct of
/// signless integer fields with the given layout and report the alignment
/// stamped on the resulting `llvm.load`. `None` means no stamp survived
/// and the exporter's natural-alignment default applies.
fn lowered_field_load_alignment(
    field_bit_widths: Vec<u32>,
    field_offsets: Vec<u64>,
    total_size: u64,
    abi_align: u64,
    field_index: u32,
) -> Option<u32> {
    let mut ctx = make_ctx();
    let field_types: Vec<TypeHandle> = field_bit_widths
        .iter()
        .map(|w| IntegerType::get(&ctx, *w, Signedness::Signless).into())
        .collect();
    let field_names = (0..field_types.len()).map(|i| format!("f{i}")).collect();
    let struct_ty: TypeHandle = MirStructType::get_with_full_layout(
        &mut ctx,
        "FieldLoadAlign".into(),
        field_names,
        field_types.clone(),
        vec![],
        field_offsets,
        total_size,
        abi_align,
    )
    .into();
    let struct_ptr_ty = MirPtrType::get_generic(&mut ctx, struct_ty, false);
    let field_ty = field_types[field_index as usize];
    let field_ptr_ty = MirPtrType::get_generic(&mut ctx, field_ty, false);

    let (module_ptr, block) = build_kernel(&mut ctx, vec![struct_ptr_ty.into()], vec![]);
    let struct_ptr_val = block.deref(&ctx).get_argument(0);

    let field_addr_op =
        mir::MirFieldAddrOp::build(&mut ctx, struct_ptr_val, field_ptr_ty.into(), field_index)
            .expect("field_addr build");
    field_addr_op.insert_at_back(block, &ctx);
    let field_ptr_val = field_addr_op.deref(&ctx).get_result(0);

    let load_op = Operation::new(
        &mut ctx,
        mir::MirLoadOp::get_concrete_op_info(),
        vec![field_ty],
        vec![field_ptr_val],
        vec![],
        0,
    );
    load_op.insert_at_back(block, &ctx);
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

    let body = kernel_blocks(&ctx, module_ptr);
    let load = find_first::<llvm::LoadOp>(&ctx, &body).expect("expected one llvm.load");
    llvm_export::ops::op_alignment(&ctx, load.get_operation())
}

/// Field 0 of an over-aligned struct sits at the aggregate's own
/// alignment, which the field's scalar result type cannot state on its
/// own. This is what lets LoadStoreVectorizer fuse the adjacent pair.
#[test]
fn convert_load_inherits_overaligned_field_alignment_at_offset_zero() {
    // #[repr(C, align(8))] struct { a: i32, b: i32 }
    assert_eq!(
        lowered_field_load_alignment(vec![32, 32], vec![0, 4], 8, 8, 0),
        Some(8)
    );
}

/// A field at a nonzero offset proves `gcd(abi_align, offset)`: an i32 at
/// offset 8 of an align-16 struct proves 8, beating its natural 4.
#[test]
fn convert_load_narrows_field_alignment_to_gcd_of_align_and_offset() {
    // #[repr(C, align(16))] struct { a: i64, b: i32 }
    assert_eq!(
        lowered_field_load_alignment(vec![64, 32], vec![0, 8], 16, 16, 1),
        Some(8)
    );
}

/// Whole-value loads of pointer-free packed structs use the packed LLVM
/// representation, preserving rustc's byte size and field offsets.
#[test]
fn packed_struct_whole_value_load_uses_packed_layout() {
    let mut ctx = make_ctx();
    let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
    let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
    let packed_ty: TypeHandle = MirStructType::get_with_full_layout(
        &mut ctx,
        "Packed".into(),
        vec!["tag".into(), "value".into()],
        vec![u8_ty, u32_ty],
        vec![0, 1],
        vec![0, 1],
        5,
        1,
    )
    .into();
    let ptr_ty = MirPtrType::get_generic(&mut ctx, packed_ty, false);

    let (module_ptr, block) = build_kernel(&mut ctx, vec![ptr_ty.into()], vec![]);
    let ptr_val = block.deref(&ctx).get_argument(0);

    let load_op = Operation::new(
        &mut ctx,
        mir::MirLoadOp::get_concrete_op_info(),
        vec![packed_ty],
        vec![ptr_val],
        vec![],
        0,
    );
    load_op.insert_at_back(block, &ctx);
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr)
        .expect("whole-value load of a pointer-free packed struct must lower");

    let body = kernel_blocks(&ctx, module_ptr);
    let load = find_first::<llvm::LoadOp>(&ctx, &body).expect("expected packed llvm.load");
    let result_ty = load
        .get_operation()
        .deref(&ctx)
        .get_result(0)
        .get_type(&ctx);
    let result_ty_ref = result_ty.deref(&ctx);
    let struct_ty = result_ty_ref
        .downcast_ref::<StructType>()
        .expect("packed load result must be an LLVM struct");
    assert_eq!(struct_ty.layout(), StructLayout::Packed);
    assert_eq!(
        crate::convert::types::llvm_type_size_align(&ctx, result_ty),
        Some((5, 1))
    );
    assert_eq!(
        llvm_export::ops::op_alignment(&ctx, load.get_operation()),
        Some(1)
    );
}

#[test]
fn packed_struct_whole_value_load_with_shared_pointer_fails_closed() {
    let mut ctx = make_ctx();
    let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
    let pointee: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
    let shared_ty: TypeHandle = MirPtrType::get_shared(&mut ctx, pointee, false).into();
    let packed_ty: TypeHandle = MirStructType::get_with_full_layout(
        &mut ctx,
        "PackedShared".into(),
        vec!["tag".into(), "ptr".into()],
        vec![u8_ty, shared_ty],
        vec![0, 1],
        vec![0, 1],
        9,
        1,
    )
    .into();
    let ptr_ty = MirPtrType::get_generic(&mut ctx, packed_ty, false);

    let (module_ptr, block) = build_kernel(&mut ctx, vec![ptr_ty.into()], vec![]);
    let ptr_val = block.deref(&ctx).get_argument(0);
    let load_op = Operation::new(
        &mut ctx,
        mir::MirLoadOp::get_concrete_op_info(),
        vec![packed_ty],
        vec![ptr_val],
        vec![],
        0,
    );
    load_op.insert_at_back(block, &ctx);
    append_mir_return(&mut ctx, block, vec![]);

    let err = crate::lower_mir_to_llvm(&mut ctx, module_ptr)
        .expect_err("packed whole-value load containing AS3 must remain fail-closed");
    assert!(
        format!("{err:?}").contains("target-mode dependent"),
        "the refusal must identify the target-dependent packed AS3 image: {err:?}"
    );
}

/// Whole-value stores use the same packed representation as construction
/// and loads, while preserving the MIR aggregate's proved ABI alignment.
#[test]
fn packed_struct_whole_value_store_uses_packed_layout() {
    let mut ctx = make_ctx();
    let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
    let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
    let packed_ty: TypeHandle = MirStructType::get_with_full_layout(
        &mut ctx,
        "Packed".into(),
        vec!["tag".into(), "value".into()],
        vec![u8_ty, u32_ty],
        vec![0, 1],
        vec![0, 1],
        5,
        1,
    )
    .into();
    let ptr_ty = MirPtrType::get_generic(&mut ctx, packed_ty, false);

    let (module_ptr, block) = build_kernel(&mut ctx, vec![ptr_ty.into(), packed_ty], vec![]);
    let ptr_val = block.deref(&ctx).get_argument(0);
    let val = block.deref(&ctx).get_argument(1);

    let store_op = Operation::new(
        &mut ctx,
        mir::MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![ptr_val, val],
        vec![],
        0,
    );
    store_op.insert_at_back(block, &ctx);
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr)
        .expect("whole-value store of a pointer-free packed struct must lower");

    let body = kernel_blocks(&ctx, module_ptr);
    let store = find_first::<llvm::StoreOp>(&ctx, &body).expect("expected packed llvm.store");
    let value_ty = store
        .get_operation()
        .deref(&ctx)
        .get_operand(0)
        .get_type(&ctx);
    let value_ty_ref = value_ty.deref(&ctx);
    let struct_ty = value_ty_ref
        .downcast_ref::<StructType>()
        .expect("packed store value must be an LLVM struct");
    assert_eq!(struct_ty.layout(), StructLayout::Packed);
    assert_eq!(
        crate::convert::types::llvm_type_size_align(&ctx, value_ty),
        Some((5, 1))
    );
    assert_eq!(
        llvm_export::ops::op_alignment(&ctx, store.get_operation()),
        Some(1)
    );
}

/// A naturally aligned inner struct sitting at a packed byte offset: the
/// field address only proves align 1, and the load must claim that over
/// the inner type's recorded abi alignment. Claiming the abi would stamp
/// `align 4` on a 1-aligned address, which llc may honor with a wider
/// access than the bytes allow.
#[test]
fn convert_load_claims_address_alignment_over_abi_at_packed_offsets() {
    let mut ctx = make_ctx();
    let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
    let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
    let inner_ty: TypeHandle = MirStructType::get_with_full_layout(
        &mut ctx,
        "Inner".into(),
        vec!["v".into()],
        vec![u32_ty],
        vec![0],
        vec![0],
        4,
        4,
    )
    .into();
    let outer_ty: TypeHandle = MirStructType::get_with_full_layout(
        &mut ctx,
        "PackedOuter".into(),
        vec!["tag".into(), "inner".into()],
        vec![u8_ty, inner_ty],
        vec![0, 1],
        vec![0, 1],
        5,
        1,
    )
    .into();
    let outer_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, outer_ty, false).into();
    let inner_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, inner_ty, false).into();

    let (module_ptr, block) = build_kernel(&mut ctx, vec![outer_ptr_ty], vec![]);
    let base = block.deref(&ctx).get_argument(0);

    let field_addr_op =
        mir::MirFieldAddrOp::build(&mut ctx, base, inner_ptr_ty, 1).expect("field_addr build");
    field_addr_op.insert_at_back(block, &ctx);
    let field_ptr_val = field_addr_op.deref(&ctx).get_result(0);

    let load_op = Operation::new(
        &mut ctx,
        mir::MirLoadOp::get_concrete_op_info(),
        vec![inner_ty],
        vec![field_ptr_val],
        vec![],
        0,
    );
    load_op.insert_at_back(block, &ctx);
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

    let body = kernel_blocks(&ctx, module_ptr);
    let load = find_first::<llvm::LoadOp>(&ctx, &body).expect("expected one llvm.load");
    assert_eq!(
        llvm_export::ops::op_alignment(&ctx, load.get_operation()),
        Some(1),
        "the load must claim the address's proved alignment, not the inner abi"
    );
}

/// A struct with no extra alignment proves nothing beyond the scalar's
/// natural alignment: the stamp equals the exporter's default 4, so the
/// emitted access is unchanged.
#[test]
fn convert_load_keeps_natural_alignment_without_overalignment() {
    // struct { a: i32, b: i32 } with rustc's natural abi_align 4
    assert_eq!(
        lowered_field_load_alignment(vec![32, 32], vec![0, 4], 8, 4, 1),
        Some(4)
    );
}

/// dialect-mir only verifier-enforces power-of-two alignment for unions
/// and enums. A malformed hand-built struct layout must decline the stamp
/// rather than emit a non-power-of-two `align N` that llc rejects.
#[test]
fn convert_load_declines_non_power_of_two_field_alignment() {
    assert_eq!(
        lowered_field_load_alignment(vec![32], vec![0], 12, 12, 0),
        None
    );
}

/// Lower `mir.store %v, (mir.field_addr %p, field_index)` for a struct of
/// signless integer fields with the given layout and report the alignment
/// stamped on the resulting `llvm.store`. `None` means no stamp survived
/// and the exporter's natural-alignment default applies.
fn lowered_field_store_alignment(
    field_bit_widths: Vec<u32>,
    field_offsets: Vec<u64>,
    total_size: u64,
    abi_align: u64,
    field_index: u32,
) -> Option<u32> {
    let mut ctx = make_ctx();
    let field_types: Vec<TypeHandle> = field_bit_widths
        .iter()
        .map(|w| IntegerType::get(&ctx, *w, Signedness::Signless).into())
        .collect();
    let field_names = (0..field_types.len()).map(|i| format!("f{i}")).collect();
    let struct_ty: TypeHandle = MirStructType::get_with_full_layout(
        &mut ctx,
        "FieldStoreAlign".into(),
        field_names,
        field_types.clone(),
        vec![],
        field_offsets,
        total_size,
        abi_align,
    )
    .into();
    let struct_ptr_ty = MirPtrType::get_generic(&mut ctx, struct_ty, true);
    let field_ty = field_types[field_index as usize];
    let field_ptr_ty = MirPtrType::get_generic(&mut ctx, field_ty, true);

    // The stored value arrives as a kernel argument of the field's own
    // scalar type, so `value_abi_align` reports nothing about it and the
    // address's stamp is the only alignment left -- the case this covers.
    let (module_ptr, block) = build_kernel(&mut ctx, vec![struct_ptr_ty.into(), field_ty], vec![]);
    let struct_ptr_val = block.deref(&ctx).get_argument(0);
    let val = block.deref(&ctx).get_argument(1);

    let field_addr_op =
        mir::MirFieldAddrOp::build(&mut ctx, struct_ptr_val, field_ptr_ty.into(), field_index)
            .expect("field_addr build");
    field_addr_op.insert_at_back(block, &ctx);
    let field_ptr_val = field_addr_op.deref(&ctx).get_result(0);

    let store_op = Operation::new(
        &mut ctx,
        mir::MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![field_ptr_val, val],
        vec![],
        0,
    );
    store_op.insert_at_back(block, &ctx);
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

    let body = kernel_blocks(&ctx, module_ptr);
    let store = find_first::<llvm::StoreOp>(&ctx, &body).expect("expected one llvm.store");
    llvm_export::ops::op_alignment(&ctx, store.get_operation())
}

/// Field 0 of an over-aligned struct sits at the aggregate's own alignment,
/// which the stored scalar's type cannot state. This is what lets
/// LoadStoreVectorizer fuse the adjacent pair into one wide store.
#[test]
fn convert_store_inherits_overaligned_field_alignment_at_offset_zero() {
    // #[repr(C, align(8))] struct { a: i32, b: i32 }
    assert_eq!(
        lowered_field_store_alignment(vec![32, 32], vec![0, 4], 8, 8, 0),
        Some(8)
    );
}

/// A field at a nonzero offset proves `gcd(abi_align, offset)`: an i32 at
/// offset 8 of an align-16 struct proves 8, beating its natural 4.
#[test]
fn convert_store_narrows_field_alignment_to_gcd_of_align_and_offset() {
    // #[repr(C, align(16))] struct { a: i64, b: i32 }
    assert_eq!(
        lowered_field_store_alignment(vec![64, 32], vec![0, 8], 16, 16, 1),
        Some(8)
    );
}

/// A struct with no extra alignment proves nothing beyond the scalar's
/// natural alignment, so the emitted store is unchanged. Widening here
/// would claim an alignment the source never guaranteed.
#[test]
fn convert_store_keeps_natural_alignment_without_overalignment() {
    // struct { a: i32, b: i32 } with rustc's natural abi_align 4
    assert_eq!(
        lowered_field_store_alignment(vec![32, 32], vec![0, 4], 8, 4, 1),
        Some(4)
    );
}

/// A malformed hand-built layout must decline the stamp rather than emit a
/// non-power-of-two `align N` that llc rejects. Same guard the load path
/// has, and it matters more here: an over-aligned store instruction on an
/// under-aligned address is undefined, not merely slow.
#[test]
fn convert_store_declines_non_power_of_two_field_alignment() {
    assert_eq!(
        lowered_field_store_alignment(vec![32], vec![0], 12, 12, 0),
        None
    );
}

/// Lower `load (&arr[index])` where `arr` is the array field of an
/// over-aligned struct, and report the alignment the load ends up with.
///
/// `index` of `Some(i)` builds a constant index, `None` a runtime one.
fn lowered_element_load_alignment(
    element_bits: u32,
    element_count: u64,
    struct_abi_align: u64,
    index: Option<u64>,
) -> Option<u32> {
    use pliron::builtin::attributes::IntegerAttr;
    use std::num::NonZeroUsize;

    let mut ctx = make_ctx();
    let element_ty: TypeHandle = IntegerType::get(&ctx, element_bits, Signedness::Signless).into();
    let array_ty: TypeHandle =
        dialect_mir::types::MirArrayType::get(&mut ctx, element_ty, element_count).into();
    let elem_bytes = u64::from(element_bits) / 8;
    let struct_ty: TypeHandle = MirStructType::get_with_full_layout(
        &mut ctx,
        "ElementLoadAlign".into(),
        vec!["lanes".into()],
        vec![array_ty],
        vec![],
        vec![0],
        elem_bytes * element_count,
        struct_abi_align,
    )
    .into();
    let struct_ptr_ty = MirPtrType::get_generic(&mut ctx, struct_ty, false);
    let array_ptr_ty = MirPtrType::get_generic(&mut ctx, array_ty, false);
    let element_ptr_ty = MirPtrType::get_generic(&mut ctx, element_ty, false);
    let i64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signed).into();

    let (module_ptr, block) = build_kernel(&mut ctx, vec![struct_ptr_ty.into(), i64_ty], vec![]);
    let struct_ptr_val = block.deref(&ctx).get_argument(0);

    // &s.lanes -- carries the struct's alignment onto the array address.
    let field_addr_op =
        mir::MirFieldAddrOp::build(&mut ctx, struct_ptr_val, array_ptr_ty.into(), 0)
            .expect("field_addr build");
    field_addr_op.insert_at_back(block, &ctx);
    let array_ptr_val = field_addr_op.deref(&ctx).get_result(0);

    let index_val = match index {
        Some(i) => {
            let constant = Operation::new(
                &mut ctx,
                mir::MirConstantOp::get_concrete_op_info(),
                vec![i64_ty],
                vec![],
                vec![],
                0,
            );
            mir::MirConstantOp::new(constant).set_attr_value(
                &ctx,
                IntegerAttr::new(
                    IntegerType::get(&ctx, 64, Signedness::Signed),
                    APInt::from_u64(i, NonZeroUsize::new(64).unwrap()),
                ),
            );
            constant.insert_at_back(block, &ctx);
            constant.deref(&ctx).get_result(0)
        }
        None => block.deref(&ctx).get_argument(1),
    };

    let elem_addr_op = Operation::new(
        &mut ctx,
        mir::MirArrayElementAddrOp::get_concrete_op_info(),
        vec![element_ptr_ty.into()],
        vec![array_ptr_val, index_val],
        vec![],
        0,
    );
    elem_addr_op.insert_at_back(block, &ctx);
    let elem_ptr_val = elem_addr_op.deref(&ctx).get_result(0);

    let load_op = Operation::new(
        &mut ctx,
        mir::MirLoadOp::get_concrete_op_info(),
        vec![element_ty],
        vec![elem_ptr_val],
        vec![],
        0,
    );
    load_op.insert_at_back(block, &ctx);
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

    let body = kernel_blocks(&ctx, module_ptr);
    let load = find_first::<llvm::LoadOp>(&ctx, &body).expect("expected one llvm.load");
    llvm_export::ops::op_alignment(&ctx, load.get_operation())
}

/// Element 0 inherits the whole alignment the base address proved, which is
/// what lets the adjacent pair fuse into one wide load.
#[test]
fn convert_load_inherits_base_alignment_at_element_zero() {
    // &(#[repr(C, align(8))] struct { lanes: [i32; 2] }).lanes[0]
    assert_eq!(lowered_element_load_alignment(32, 2, 8, Some(0)), Some(8));
}

/// A nonzero constant index proves `gcd(base, i * stride)`: element 1 of an
/// align-8 `[i32; 2]` sits at byte 4, so it proves 4, not 8.
#[test]
fn convert_load_narrows_element_alignment_to_gcd_with_offset() {
    assert_eq!(lowered_element_load_alignment(32, 2, 8, Some(1)), Some(4));
}

/// A runtime index can land on any element, so only what every stride
/// preserves may be claimed -- `gcd(base, stride)`, never the base itself.
#[test]
fn convert_load_claims_only_stride_alignment_for_a_runtime_index() {
    assert_eq!(lowered_element_load_alignment(32, 2, 8, None), Some(4));
}

/// A base with no extra alignment proves nothing beyond the element's own
/// natural alignment, so the emitted access is unchanged.
#[test]
fn convert_load_keeps_natural_element_alignment_without_overalignment() {
    assert_eq!(lowered_element_load_alignment(32, 2, 4, Some(0)), Some(4));
}

/// Like [`lowered_element_load_alignment`], but for an array whose element
/// is an aggregate built by `element_ty_of` (which also reports the
/// element's stored size in bytes). The claim on the element address must
/// then come from the element's *exact* stride — rustc's stored size,
/// padding included — not from any LLVM-level approximation.
///
/// With `load_first_scalar` the element must itself be an array and the
/// access becomes `s.lanes[index][0]`, mirroring the nested-read chain
/// where the outer stamp is inherited by the inner index-0 address and
/// ends up on a scalar load that LoadStoreVectorizer trusts. Without it,
/// the element itself is loaded.
///
/// Reports `(element address stamp, final load alignment)`.
fn lowered_aggregate_element_alignments(
    element_ty_of: impl FnOnce(&mut Context) -> (TypeHandle, u64),
    element_count: u64,
    struct_abi_align: u64,
    index: Option<u64>,
    load_first_scalar: bool,
) -> (Option<u32>, Option<u32>) {
    use pliron::builtin::attributes::IntegerAttr;
    use std::num::NonZeroUsize;

    let mut ctx = make_ctx();
    let (element_ty, elem_bytes) = element_ty_of(&mut ctx);
    let inner_scalar_ty = load_first_scalar.then(|| {
        let element_ref = element_ty.deref(&ctx);
        element_ref
            .downcast_ref::<MirArrayType>()
            .expect("load_first_scalar needs an array element")
            .element_type()
    });
    let array_ty: TypeHandle = MirArrayType::get(&mut ctx, element_ty, element_count).into();
    let struct_ty: TypeHandle = MirStructType::get_with_full_layout(
        &mut ctx,
        "AggregateElementAlign".into(),
        vec!["lanes".into()],
        vec![array_ty],
        vec![],
        vec![0],
        elem_bytes * element_count,
        struct_abi_align,
    )
    .into();
    let struct_ptr_ty = MirPtrType::get_generic(&mut ctx, struct_ty, false);
    let array_ptr_ty = MirPtrType::get_generic(&mut ctx, array_ty, false);
    let element_ptr_ty = MirPtrType::get_generic(&mut ctx, element_ty, false);
    let i64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signed).into();

    let (module_ptr, block) = build_kernel(&mut ctx, vec![struct_ptr_ty.into(), i64_ty], vec![]);
    let struct_ptr_val = block.deref(&ctx).get_argument(0);

    // &s.lanes -- carries the struct's alignment onto the array address.
    let field_addr_op =
        mir::MirFieldAddrOp::build(&mut ctx, struct_ptr_val, array_ptr_ty.into(), 0)
            .expect("field_addr build");
    field_addr_op.insert_at_back(block, &ctx);
    let array_ptr_val = field_addr_op.deref(&ctx).get_result(0);

    let constant_index = |ctx: &mut Context, i: u64| {
        let constant = Operation::new(
            ctx,
            mir::MirConstantOp::get_concrete_op_info(),
            vec![i64_ty],
            vec![],
            vec![],
            0,
        );
        mir::MirConstantOp::new(constant).set_attr_value(
            ctx,
            IntegerAttr::new(
                IntegerType::get(ctx, 64, Signedness::Signed),
                APInt::from_u64(i, NonZeroUsize::new(64).unwrap()),
            ),
        );
        constant.insert_at_back(block, ctx);
        constant.deref(ctx).get_result(0)
    };

    let index_val = match index {
        Some(i) => constant_index(&mut ctx, i),
        None => block.deref(&ctx).get_argument(1),
    };

    let elem_addr_op = Operation::new(
        &mut ctx,
        mir::MirArrayElementAddrOp::get_concrete_op_info(),
        vec![element_ptr_ty.into()],
        vec![array_ptr_val, index_val],
        vec![],
        0,
    );
    elem_addr_op.insert_at_back(block, &ctx);
    let elem_ptr_val = elem_addr_op.deref(&ctx).get_result(0);

    let (loaded_ty, loaded_ptr_val) = match inner_scalar_ty {
        Some(scalar_ty) => {
            let zero_val = constant_index(&mut ctx, 0);
            let scalar_ptr_ty = MirPtrType::get_generic(&mut ctx, scalar_ty, false);
            let inner_addr_op = Operation::new(
                &mut ctx,
                mir::MirArrayElementAddrOp::get_concrete_op_info(),
                vec![scalar_ptr_ty.into()],
                vec![elem_ptr_val, zero_val],
                vec![],
                0,
            );
            inner_addr_op.insert_at_back(block, &ctx);
            (scalar_ty, inner_addr_op.deref(&ctx).get_result(0))
        }
        None => (element_ty, elem_ptr_val),
    };

    let load_op = Operation::new(
        &mut ctx,
        mir::MirLoadOp::get_concrete_op_info(),
        vec![loaded_ty],
        vec![loaded_ptr_val],
        vec![],
        0,
    );
    load_op.insert_at_back(block, &ctx);
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

    let body = kernel_blocks(&ctx, module_ptr);
    // GEP order follows source order: field address, then the element
    // address under test (then the inner index-0 address when nested).
    let geps = find_all::<llvm::GetElementPtrOp>(&ctx, &body);
    assert_eq!(geps.len(), if load_first_scalar { 3 } else { 2 });
    let element_gep_align = llvm_export::ops::address_alignment(&ctx, geps[1].get_operation());
    let load = find_first::<llvm::LoadOp>(&ctx, &body).expect("expected one llvm.load");
    let load_align = llvm_export::ops::op_alignment(&ctx, load.get_operation());
    (element_gep_align, load_align)
}

/// `[f32; 3]` element under an align-8 base: stride is 12, so a runtime
/// index proves `gcd(8, 12) = 4` — and the inner index-0 scalar read
/// inherits exactly that. Guards against sizing the element through an
/// LLVM-level approximation, whose guessed stride of 8 would stamp
/// align 8 onto addresses that are only 4-aligned (a miscompile once
/// LoadStoreVectorizer trusts it).
#[test]
fn convert_load_claims_exact_aggregate_stride_for_nested_array_elements() {
    use pliron::builtin::types::FP32Type;
    // &(#[repr(C, align(8))] struct { lanes: [[f32; 3]; 4] }).lanes[i][0]
    let nested_f32x3 = |ctx: &mut Context| {
        let f32_ty: TypeHandle = FP32Type::get(ctx).into();
        (MirArrayType::get(ctx, f32_ty, 3).into(), 12)
    };
    assert_eq!(
        lowered_aggregate_element_alignments(nested_f32x3, 4, 8, None, true),
        (Some(4), Some(4))
    );
}

/// A constant index into the same nested array uses the exact byte
/// offset: element 1 sits at byte 12 (`gcd(8, 12) = 4`), element 2 at
/// byte 24 (`gcd(8, 24) = 8`, the full base alignment again).
#[test]
fn convert_load_narrows_nested_element_alignment_by_exact_byte_offset() {
    use pliron::builtin::types::FP32Type;
    let nested_f32x3 = |ctx: &mut Context| {
        let f32_ty: TypeHandle = FP32Type::get(ctx).into();
        (MirArrayType::get(ctx, f32_ty, 3).into(), 12)
    };
    assert_eq!(
        lowered_aggregate_element_alignments(nested_f32x3, 4, 8, Some(1), true),
        (Some(4), Some(4))
    );
    assert_eq!(
        lowered_aggregate_element_alignments(nested_f32x3, 4, 8, Some(2), true),
        (Some(8), Some(8))
    );
}

/// A tuple element's stride comes from rustc's recorded `total_size`
/// (trailing padding included): `(f32, f32, f32)` stores 12 bytes, so an
/// align-8 base proves only 4 on a runtime element address.
#[test]
fn convert_array_element_addr_takes_tuple_stride_from_recorded_layout() {
    use pliron::builtin::types::FP32Type;
    let f32x3_tuple = |ctx: &mut Context| {
        let f32_ty: TypeHandle = FP32Type::get(ctx).into();
        let tuple_ty: TypeHandle =
            MirTupleType::get_with_layout(ctx, vec![f32_ty; 3], vec![], vec![0, 4, 8], 12, 4)
                .into();
        (tuple_ty, 12)
    };
    let (element_gep_align, _load_align) =
        lowered_aggregate_element_alignments(f32x3_tuple, 4, 8, None, false);
    assert_eq!(element_gep_align, Some(4));
}

/// `f16` arrives from the importer as `MirFP16Type`, not the converted
/// LLVM `half`, and its stride is exactly 2: an align-8 base proves 2 on
/// a runtime element address and `gcd(8, 4) = 4` at element 2. Guards
/// the arm the importer actually exercises — the old sizing guessed 8
/// for this type too, the same over-claim as the aggregate cases.
#[test]
fn convert_load_claims_exact_f16_element_stride() {
    let f16_scalar = |ctx: &mut Context| {
        let f16_ty: TypeHandle = dialect_mir::types::MirFP16Type::get(ctx).into();
        (f16_ty, 2)
    };
    assert_eq!(
        lowered_aggregate_element_alignments(f16_scalar, 4, 8, None, false),
        (Some(2), Some(2))
    );
    assert_eq!(
        lowered_aggregate_element_alignments(f16_scalar, 4, 8, Some(2), false),
        (Some(4), Some(4))
    );
}

/// An element whose stored size is unknown (a struct built without rustc
/// layout) must not have its stride guessed: the element address claims
/// nothing and the load keeps the previous, weaker-but-sound behaviour.
#[test]
fn convert_array_element_addr_declines_unknown_element_stride() {
    use pliron::builtin::types::FP32Type;
    let opaque_struct = |ctx: &mut Context| {
        let f32_ty: TypeHandle = FP32Type::get(ctx).into();
        let struct_ty: TypeHandle =
            MirStructType::get(ctx, "OpaqueElement".into(), vec!["x".into()], vec![f32_ty]).into();
        (struct_ty, 4)
    };
    assert_eq!(
        lowered_aggregate_element_alignments(opaque_struct, 4, 8, None, false),
        (None, None)
    );
}
