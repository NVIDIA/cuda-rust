/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_mir::{
    attributes::{FieldIndexAttr, MirCastKindAttr, MirPointerKindAuthorityAttr},
    ops::{
        MirCastOp, MirConstructDisjointSliceOp, MirConstructSliceOp, MirExtractFieldOp,
        MirFieldAddrOp, MirStoreOp,
    },
    types::{
        MirDisjointSliceType, MirPointerKind, MirPtrType, MirSliceType, MirTupleType, MirUnionType,
    },
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::TypeAttr,
        types::{FP32Type, IntegerType, Signedness},
    },
    common_traits::Verify,
    context::Context,
    op::Op,
    operation::Operation,
};

#[test]
fn test_mir_extract_field_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let tuple_ty = MirTupleType::get(&mut ctx, vec![i32_ty.into(), i32_ty.into()]);

    let block = BasicBlock::new(&mut ctx, None, vec![tuple_ty.into()]);
    let tuple_val = block.deref(&ctx).get_argument(0);

    let op = Operation::new(
        &mut ctx,
        MirExtractFieldOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![tuple_val],
        vec![],
        0,
    );
    let extract_op = MirExtractFieldOp::new(op);
    extract_op.set_attr_index(&ctx, dialect_mir::attributes::FieldIndexAttr(0));
    assert!(extract_op.verify(&ctx).is_ok(), "Valid Tuple Extract");

    let op_oob = Operation::new(
        &mut ctx,
        MirExtractFieldOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![tuple_val],
        vec![],
        0,
    );
    let extract_op_oob = MirExtractFieldOp::new(op_oob);
    extract_op_oob.set_attr_index(&ctx, dialect_mir::attributes::FieldIndexAttr(2));
    assert!(extract_op_oob.verify(&ctx).is_err(), "OOB Index");

    let union_ty = MirUnionType::get(
        &mut ctx,
        "Bits".into(),
        vec!["word".into(), "alias".into()],
        vec![i32_ty.into(), i32_ty.into()],
        4,
        4,
    );
    let union_block = BasicBlock::new(&mut ctx, None, vec![union_ty.into()]);
    let union_val = union_block.deref(&ctx).get_argument(0);
    let union_extract = Operation::new(
        &mut ctx,
        MirExtractFieldOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![union_val],
        vec![],
        0,
    );
    let union_extract = MirExtractFieldOp::new(union_extract);
    union_extract.set_attr_index(&ctx, dialect_mir::attributes::FieldIndexAttr(1));
    assert!(union_extract.verify(&ctx).is_ok(), "Valid union extract");
}

#[test]
fn test_mir_construct_disjoint_slice_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let f32_ty = FP32Type::get(&ctx);
    let usize_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let width_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let f32_ptr_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, f32_ty.into(), true, MirPointerKind::RawMut);
    let plain_ty = MirDisjointSliceType::get(&mut ctx, f32_ty.into());
    let width_ty_handle: pliron::r#type::TypeHandle = width_ty.into();
    let row_width_ty =
        MirDisjointSliceType::get_with_space(&mut ctx, f32_ty.into(), vec![width_ty_handle]);

    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![f32_ptr_ty.into(), usize_ty.into(), width_ty.into()],
    );
    let ptr_val = block.deref(&ctx).get_argument(0);
    let len_val = block.deref(&ctx).get_argument(1);
    let width_val = block.deref(&ctx).get_argument(2);

    // Valid: an index space with no runtime layout takes two operands.
    let op = Operation::new(
        &mut ctx,
        MirConstructDisjointSliceOp::get_concrete_op_info(),
        vec![plain_ty.into()],
        vec![ptr_val, len_val],
        vec![],
        0,
    );
    assert!(
        MirConstructDisjointSliceOp::new(op).verify(&ctx).is_ok(),
        "Valid space-free disjoint slice construction"
    );

    // Valid: a runtime row width takes a third operand.
    let op_width = Operation::new(
        &mut ctx,
        MirConstructDisjointSliceOp::get_concrete_op_info(),
        vec![row_width_ty.into()],
        vec![ptr_val, len_val, width_val],
        vec![],
        0,
    );
    assert!(
        MirConstructDisjointSliceOp::new(op_width)
            .verify(&ctx)
            .is_ok(),
        "Valid row-width disjoint slice construction"
    );

    let erased_ptr_ty = MirPtrType::get_generic(&mut ctx, f32_ty.into(), true);
    let erased_block = BasicBlock::new(&mut ctx, None, vec![erased_ptr_ty.into()]);
    let erased_ptr = erased_block.deref(&ctx).get_argument(0);
    let op_erased_data = Operation::new(
        &mut ctx,
        MirConstructDisjointSliceOp::get_concrete_op_info(),
        vec![plain_ty.into()],
        vec![erased_ptr, len_val],
        vec![],
        0,
    );
    assert!(
        MirConstructDisjointSliceOp::new(op_erased_data)
            .verify(&ctx)
            .is_err(),
        "DisjointSlice's fixed RawMut field cannot be reconstructed from Erased"
    );

    // Invalid: the row width is missing, so the slice would carry whatever
    // slot 2 held.
    let op_missing_width = Operation::new(
        &mut ctx,
        MirConstructDisjointSliceOp::get_concrete_op_info(),
        vec![row_width_ty.into()],
        vec![ptr_val, len_val],
        vec![],
        0,
    );
    assert!(
        MirConstructDisjointSliceOp::new(op_missing_width)
            .verify(&ctx)
            .is_err(),
        "Missing index-space operand"
    );

    // Invalid: a space-free slice given a third operand.
    let op_extra = Operation::new(
        &mut ctx,
        MirConstructDisjointSliceOp::get_concrete_op_info(),
        vec![plain_ty.into()],
        vec![ptr_val, len_val, width_val],
        vec![],
        0,
    );
    assert!(
        MirConstructDisjointSliceOp::new(op_extra)
            .verify(&ctx)
            .is_err(),
        "Index-space operand for a space-free slice"
    );

    // Invalid: the row width operand has the wrong width, which would write a
    // 64-bit value into the 32-bit row width slot.
    let op_wrong_width_ty = Operation::new(
        &mut ctx,
        MirConstructDisjointSliceOp::get_concrete_op_info(),
        vec![row_width_ty.into()],
        vec![ptr_val, len_val, len_val],
        vec![],
        0,
    );
    assert!(
        MirConstructDisjointSliceOp::new(op_wrong_width_ty)
            .verify(&ctx)
            .is_err(),
        "Index-space operand type mismatch"
    );

    // Invalid: result is a plain slice, which `mir.construct_slice` owns.
    let plain_slice_ty = MirSliceType::get(&mut ctx, f32_ty.into());
    let op_bad_res = Operation::new(
        &mut ctx,
        MirConstructDisjointSliceOp::get_concrete_op_info(),
        vec![plain_slice_ty.into()],
        vec![ptr_val, len_val],
        vec![],
        0,
    );
    assert!(
        MirConstructDisjointSliceOp::new(op_bad_res)
            .verify(&ctx)
            .is_err(),
        "Result must be a disjoint slice type"
    );
}

#[test]
fn test_mir_construct_slice_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let u8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let usize_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let u8_ptr_ty = MirPtrType::get_generic(&mut ctx, u8_ty.into(), false);
    let u8_slice_ty = MirSliceType::get(&mut ctx, u8_ty.into());
    let i32_slice_ty = MirSliceType::get(&mut ctx, i32_ty.into());

    let block = BasicBlock::new(&mut ctx, None, vec![u8_ptr_ty.into(), usize_ty.into()]);
    let ptr_val = block.deref(&ctx).get_argument(0);
    let len_val = block.deref(&ctx).get_argument(1);

    // Valid: (ptr to u8, usize len) -> slice of u8
    let op = Operation::new(
        &mut ctx,
        MirConstructSliceOp::get_concrete_op_info(),
        vec![u8_slice_ty.into()],
        vec![ptr_val, len_val],
        vec![],
        0,
    );
    assert!(
        MirConstructSliceOp::new(op).verify(&ctx).is_ok(),
        "Valid slice construction"
    );

    // Invalid: data pointer pointee does not match slice element type
    let op_bad_elem = Operation::new(
        &mut ctx,
        MirConstructSliceOp::get_concrete_op_info(),
        vec![i32_slice_ty.into()],
        vec![ptr_val, len_val],
        vec![],
        0,
    );
    assert!(
        MirConstructSliceOp::new(op_bad_elem).verify(&ctx).is_err(),
        "Pointee/element mismatch"
    );

    // Invalid: operands swapped (length where the pointer should be)
    let op_swapped = Operation::new(
        &mut ctx,
        MirConstructSliceOp::get_concrete_op_info(),
        vec![u8_slice_ty.into()],
        vec![len_val, ptr_val],
        vec![],
        0,
    );
    assert!(
        MirConstructSliceOp::new(op_swapped).verify(&ctx).is_err(),
        "Swapped operands"
    );

    // Invalid: result is not a slice type
    let op_bad_res = Operation::new(
        &mut ctx,
        MirConstructSliceOp::get_concrete_op_info(),
        vec![u8_ptr_ty.into()],
        vec![ptr_val, len_val],
        vec![],
        0,
    );
    assert!(
        MirConstructSliceOp::new(op_bad_res).verify(&ctx).is_err(),
        "Non-slice result type"
    );
}

#[test]
fn test_slice_carrier_cannot_launder_pointer_kind() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let u8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let usize_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let raw_mut_ptr =
        MirPtrType::get_generic_with_kind(&mut ctx, u8_ty.into(), true, MirPointerKind::RawMut);
    let erased_ptr = MirPtrType::get_generic(&mut ctx, u8_ty.into(), true);
    let erased_const_ptr = MirPtrType::get_generic(&mut ctx, u8_ty.into(), false);
    let global_raw_mut_ptr = MirPtrType::get_with_kind(
        &mut ctx,
        u8_ty.into(),
        true,
        dialect_mir::types::address_space::GLOBAL,
        MirPointerKind::RawMut,
    );
    let unique_ptr =
        MirPtrType::get_generic_with_kind(&mut ctx, u8_ty.into(), true, MirPointerKind::UniqueRef);
    let raw_mut_slice = MirSliceType::get_with_kind(&mut ctx, u8_ty.into(), MirPointerKind::RawMut);
    let unique_slice =
        MirSliceType::get_with_kind(&mut ctx, u8_ty.into(), MirPointerKind::UniqueRef);
    let erased_slice = MirSliceType::get(&mut ctx, u8_ty.into());
    let erased_mut_slice = MirSliceType::get_with_mutability(&mut ctx, u8_ty.into(), true);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            raw_mut_ptr.into(),
            erased_ptr.into(),
            erased_const_ptr.into(),
            global_raw_mut_ptr.into(),
            usize_ty.into(),
            raw_mut_slice.into(),
            erased_slice.into(),
            erased_mut_slice.into(),
        ],
    );
    let raw_mut = block.deref(&ctx).get_argument(0);
    let erased = block.deref(&ctx).get_argument(1);
    let erased_const = block.deref(&ctx).get_argument(2);
    let global_raw_mut = block.deref(&ctx).get_argument(3);
    let len = block.deref(&ctx).get_argument(4);
    let raw_slice = block.deref(&ctx).get_argument(5);
    let erased_slice_value = block.deref(&ctx).get_argument(6);
    let erased_mut_slice_value = block.deref(&ctx).get_argument(7);

    let construct = |ctx: &mut Context, data, result_ty| {
        Operation::new(
            ctx,
            MirConstructSliceOp::get_concrete_op_info(),
            vec![result_ty],
            vec![data, len],
            vec![],
            0,
        )
    };
    assert!(
        MirConstructSliceOp::new(construct(&mut ctx, raw_mut, raw_mut_slice.into()))
            .verify(&ctx)
            .is_ok()
    );
    assert!(
        MirConstructSliceOp::new(construct(&mut ctx, raw_mut, unique_slice.into()))
            .verify(&ctx)
            .is_err(),
        "construct_slice must not turn RawMut into UniqueRef"
    );
    assert!(
        MirConstructSliceOp::new(construct(&mut ctx, erased, raw_mut_slice.into()))
            .verify(&ctx)
            .is_err(),
        "construct_slice must not recover RawMut from Erased"
    );
    assert!(
        MirConstructSliceOp::new(construct(&mut ctx, global_raw_mut, raw_mut_slice.into()))
            .verify(&ctx)
            .is_err(),
        "ordinary slice carriers always use generic address space"
    );
    assert!(
        MirConstructSliceOp::new(construct(&mut ctx, erased_const, erased_slice.into()))
            .verify(&ctx)
            .is_ok(),
        "an immutable Erased data pointer constructs an immutable Erased slice"
    );

    let extract = |ctx: &mut Context, result_ty| {
        let op = Operation::new(
            ctx,
            MirExtractFieldOp::get_concrete_op_info(),
            vec![result_ty],
            vec![raw_slice],
            vec![],
            0,
        );
        MirExtractFieldOp::new(op).set_attr_index(ctx, FieldIndexAttr(0));
        op
    };
    assert!(
        MirExtractFieldOp::new(extract(&mut ctx, raw_mut_ptr.into()))
            .verify(&ctx)
            .is_ok()
    );
    assert!(
        MirExtractFieldOp::new(extract(&mut ctx, erased_ptr.into()))
            .verify(&ctx)
            .is_ok(),
        "slice extraction may deliberately erase a concrete kind"
    );
    assert!(
        MirExtractFieldOp::new(extract(&mut ctx, unique_ptr.into()))
            .verify(&ctx)
            .is_err(),
        "slice extraction must not change RawMut into UniqueRef"
    );
    assert!(
        MirExtractFieldOp::new(extract(&mut ctx, global_raw_mut_ptr.into()))
            .verify(&ctx)
            .is_err(),
        "ordinary slice extraction cannot invent a non-generic address space"
    );
    assert!(
        MirExtractFieldOp::new(extract(&mut ctx, erased_const_ptr.into()))
            .verify(&ctx)
            .is_err(),
        "erasing a concrete slice kind must preserve machine mutability"
    );

    let extract_erased = |ctx: &mut Context, result_ty| {
        let op = Operation::new(
            ctx,
            MirExtractFieldOp::get_concrete_op_info(),
            vec![result_ty],
            vec![erased_slice_value],
            vec![],
            0,
        );
        MirExtractFieldOp::new(op).set_attr_index(ctx, FieldIndexAttr(0));
        op
    };
    assert!(
        MirExtractFieldOp::new(extract_erased(&mut ctx, erased_const_ptr.into()))
            .verify(&ctx)
            .is_ok(),
        "an immutable Erased slice extracts an immutable Erased data pointer"
    );
    assert!(
        MirExtractFieldOp::new(extract_erased(&mut ctx, erased_ptr.into()))
            .verify(&ctx)
            .is_err(),
        "an immutable Erased slice cannot manufacture writable data-pointer evidence"
    );

    let extract_erased_mut = |ctx: &mut Context, result_ty| {
        let op = Operation::new(
            ctx,
            MirExtractFieldOp::get_concrete_op_info(),
            vec![result_ty],
            vec![erased_mut_slice_value],
            vec![],
            0,
        );
        MirExtractFieldOp::new(op).set_attr_index(ctx, FieldIndexAttr(0));
        op
    };
    assert!(
        MirExtractFieldOp::new(extract_erased_mut(&mut ctx, erased_ptr.into()))
            .verify(&ctx)
            .is_ok(),
        "a mutable Erased slice retains writable data-pointer evidence"
    );

    let immutable_slice_reborrow = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![unique_slice.into()],
        vec![erased_slice_value],
        vec![],
        0,
    );
    let immutable_slice_reborrow = MirCastOp::new(immutable_slice_reborrow);
    immutable_slice_reborrow.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    immutable_slice_reborrow
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::Reborrow);
    assert!(
        immutable_slice_reborrow.verify(&ctx).is_err(),
        "an immutable Erased slice cannot be reborrowed as UniqueRef"
    );

    let mutable_slice_reborrow = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![unique_slice.into()],
        vec![erased_mut_slice_value],
        vec![],
        0,
    );
    let mutable_slice_reborrow = MirCastOp::new(mutable_slice_reborrow);
    mutable_slice_reborrow.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    mutable_slice_reborrow
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::Reborrow);
    assert!(
        mutable_slice_reborrow.verify(&ctx).is_ok(),
        "a mutable Erased slice may establish UniqueRef at a real reborrow boundary"
    );
}

#[test]
fn test_mir_field_addr_tuple_pointee_verify() {
    // `(u8, u32)` laid out the way rustc actually places it: the u32 field
    // first in memory for alignment, so declaration index 0 (`u8`) lands at
    // byte offset 4 and declaration index 1 (`u32`) lands at byte offset 0.
    // `field_addr`'s `field_index` attribute is a DECLARATION index (it names
    // `.0`/`.1` as written), so this test only passes if the op resolves the
    // field's type through `MirTupleType::get_types()` (declaration order)
    // rather than assuming identity with memory order.
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let u8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);

    let tuple_ty = MirTupleType::get_with_layout(
        &mut ctx,
        vec![u8_ty.into(), u32_ty.into()],
        vec![1, 0],
        vec![4, 0],
        8,
        4,
    );

    let tuple_ptr_ty = MirPtrType::get_generic(&mut ctx, tuple_ty.into(), false);
    let blk = BasicBlock::new(&mut ctx, None, vec![tuple_ptr_ty.into()]);
    let tuple_ptr = blk.deref(&ctx).get_argument(0);

    let u8_ptr_ty = MirPtrType::get_generic(&mut ctx, u8_ty.into(), false);
    let op_field0 = Operation::new(
        &mut ctx,
        MirFieldAddrOp::get_concrete_op_info(),
        vec![u8_ptr_ty.into()],
        vec![tuple_ptr],
        vec![],
        0,
    );
    let field0 = MirFieldAddrOp::new(op_field0);
    field0.set_attr_field_index(&ctx, FieldIndexAttr(0));
    field0.set_attr_aggregate_ty(&ctx, TypeAttr::new(tuple_ty.into()));
    assert!(
        field0.verify(&ctx).is_ok(),
        "tuple field 0 (u8) address accepted"
    );

    let u32_ptr_ty = MirPtrType::get_generic(&mut ctx, u32_ty.into(), false);
    let op_field1 = Operation::new(
        &mut ctx,
        MirFieldAddrOp::get_concrete_op_info(),
        vec![u32_ptr_ty.into()],
        vec![tuple_ptr],
        vec![],
        0,
    );
    let field1 = MirFieldAddrOp::new(op_field1);
    field1.set_attr_field_index(&ctx, FieldIndexAttr(1));
    field1.set_attr_aggregate_ty(&ctx, TypeAttr::new(tuple_ty.into()));
    assert!(
        field1.verify(&ctx).is_ok(),
        "tuple field 1 (u32) address accepted"
    );

    // Result pointee type must match the DECLARED field type, not whatever
    // sits at that byte offset: pointing field 0's result at u32 (field 1's
    // type) must be rejected even though both are in-bounds indices.
    let op_wrong_result_ty = Operation::new(
        &mut ctx,
        MirFieldAddrOp::get_concrete_op_info(),
        vec![u32_ptr_ty.into()],
        vec![tuple_ptr],
        vec![],
        0,
    );
    let wrong_result_ty = MirFieldAddrOp::new(op_wrong_result_ty);
    wrong_result_ty.set_attr_field_index(&ctx, FieldIndexAttr(0));
    wrong_result_ty.set_attr_aggregate_ty(&ctx, TypeAttr::new(tuple_ty.into()));
    assert!(
        wrong_result_ty.verify(&ctx).is_err(),
        "result pointee type mismatch rejected"
    );

    let op_out_of_bounds = Operation::new(
        &mut ctx,
        MirFieldAddrOp::get_concrete_op_info(),
        vec![u8_ptr_ty.into()],
        vec![tuple_ptr],
        vec![],
        0,
    );
    let out_of_bounds = MirFieldAddrOp::new(op_out_of_bounds);
    out_of_bounds.set_attr_field_index(&ctx, FieldIndexAttr(2));
    out_of_bounds.set_attr_aggregate_ty(&ctx, TypeAttr::new(tuple_ty.into()));
    assert!(
        out_of_bounds.verify(&ctx).is_err(),
        "out-of-bounds tuple field index rejected"
    );
}

#[test]
fn test_mir_field_addr_tuple_pointee_store_verify() {
    // The WRITE side of the tuple-pointee unlock: `t.1 = x` / `arr[i].1 = x`
    // lower to `mir.field_addr` + `mir.store` through the field's address, so
    // a tuple-pointee field address used as a store destination must pass
    // verification too. Same reordered `(u8, u32)` layout as above (the u32
    // field first in memory), so the store type-checks against the DECLARED
    // field type, not whatever occupies that memory slot.
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let u8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);

    let tuple_ty = MirTupleType::get_with_layout(
        &mut ctx,
        vec![u8_ty.into(), u32_ty.into()],
        vec![1, 0],
        vec![4, 0],
        8,
        4,
    );

    let tuple_ptr_ty = MirPtrType::get_generic(&mut ctx, tuple_ty.into(), false);
    let blk = BasicBlock::new(
        &mut ctx,
        None,
        vec![tuple_ptr_ty.into(), u32_ty.into(), u8_ty.into()],
    );
    let tuple_ptr = blk.deref(&ctx).get_argument(0);
    let u32_val = blk.deref(&ctx).get_argument(1);
    let u8_val = blk.deref(&ctx).get_argument(2);

    // `.1 = x`: address declaration field 1 (u32, memory slot 0) and store a
    // u32 through it.
    let u32_ptr_ty = MirPtrType::get_generic(&mut ctx, u32_ty.into(), false);
    let op_field1 = Operation::new(
        &mut ctx,
        MirFieldAddrOp::get_concrete_op_info(),
        vec![u32_ptr_ty.into()],
        vec![tuple_ptr],
        vec![],
        0,
    );
    let field1 = MirFieldAddrOp::new(op_field1);
    field1.set_attr_field_index(&ctx, FieldIndexAttr(1));
    field1.set_attr_aggregate_ty(&ctx, TypeAttr::new(tuple_ty.into()));
    assert!(
        field1.verify(&ctx).is_ok(),
        "tuple field 1 (u32) address accepted as a store destination"
    );
    let field1_ptr = op_field1.deref(&ctx).get_result(0);

    let op_store = Operation::new(
        &mut ctx,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![field1_ptr, u32_val],
        vec![],
        0,
    );
    assert!(
        MirStoreOp::new(op_store).verify(&ctx).is_ok(),
        "store through a tuple field address verifies"
    );

    // The stored value must match the DECLARED field type (`u32` for `.1`),
    // not the type of the field sharing the tuple: a u8 store through the
    // `.1` pointer is a type mismatch.
    let op_store_wrong_ty = Operation::new(
        &mut ctx,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![field1_ptr, u8_val],
        vec![],
        0,
    );
    assert!(
        MirStoreOp::new(op_store_wrong_ty).verify(&ctx).is_err(),
        "store of a mismatched value type through a tuple field address rejected"
    );
}
