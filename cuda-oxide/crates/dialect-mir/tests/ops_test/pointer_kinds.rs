/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_mir::{
    attributes::{MirCastKindAttr, MirPointerKindAuthorityAttr},
    ops::{MirAllocaOp, MirCastOp, MirGlobalAllocOp, MirPtrOffsetOp, MirRefOp, MirSharedAllocOp},
    types::{
        MirArrayType, MirDisjointSliceType, MirPointerKind, MirPtrType, MirSliceType,
        MirStructType, MirTupleType,
    },
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{IntegerAttr, StringAttr, TypeAttr},
        types::{FP32Type, IntegerType, Signedness},
    },
    common_traits::Verify,
    context::Context,
    op::Op,
    operation::Operation,
    r#type::TypeHandle,
    utils::apint::APInt,
};
use std::num::NonZeroUsize;

#[test]
fn test_mir_pointer_kind_distinguishes_references_from_raw_pointers() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let pointee: pliron::r#type::TypeHandle = i32_ty.into();

    let shared_ref =
        MirPtrType::get_generic_with_kind(&mut ctx, pointee, false, MirPointerKind::SharedRef);
    let raw_const =
        MirPtrType::get_generic_with_kind(&mut ctx, pointee, false, MirPointerKind::RawConst);
    let unique_ref =
        MirPtrType::get_generic_with_kind(&mut ctx, pointee, true, MirPointerKind::UniqueRef);
    let raw_mut =
        MirPtrType::get_generic_with_kind(&mut ctx, pointee, true, MirPointerKind::RawMut);

    assert_ne!(
        shared_ref, raw_const,
        "&T must remain distinct from *const T"
    );
    assert_ne!(
        unique_ref, raw_mut,
        "&mut T must remain distinct from *mut T"
    );
    assert_ne!(
        shared_ref, unique_ref,
        "&T must remain distinct from &mut T"
    );
    assert_ne!(
        raw_const, raw_mut,
        "*const T must remain distinct from *mut T"
    );

    assert_eq!(
        shared_ref.deref(&ctx).pointer_kind(),
        MirPointerKind::SharedRef
    );
    assert_eq!(
        unique_ref.deref(&ctx).pointer_kind(),
        MirPointerKind::UniqueRef
    );
    assert_eq!(
        raw_const.deref(&ctx).pointer_kind(),
        MirPointerKind::RawConst
    );
    assert_eq!(raw_mut.deref(&ctx).pointer_kind(), MirPointerKind::RawMut);
}

#[test]
fn test_mir_slice_preserves_pointer_kind() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let u8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let element: pliron::r#type::TypeHandle = u8_ty.into();

    let shared = MirSliceType::get_with_kind(&mut ctx, element, MirPointerKind::SharedRef);
    let raw = MirSliceType::get_with_kind(&mut ctx, element, MirPointerKind::RawConst);

    assert_ne!(shared, raw, "&[T] must remain distinct from *const [T]");
    assert_eq!(shared.deref(&ctx).pointer_kind(), MirPointerKind::SharedRef);
    assert_eq!(raw.deref(&ctx).pointer_kind(), MirPointerKind::RawConst);
}

#[test]
fn test_mir_pointer_kind_mutability_consistency_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let invalid_shared = MirPtrType {
        pointee: i32_ty.into(),
        is_mutable: true,
        address_space: 0,
        kind: MirPointerKind::SharedRef,
    };
    let invalid_raw_mut = MirPtrType {
        pointee: i32_ty.into(),
        is_mutable: false,
        address_space: 0,
        kind: MirPointerKind::RawMut,
    };

    assert!(invalid_shared.verify(&ctx).is_err());
    assert!(invalid_raw_mut.verify(&ctx).is_err());

    let invalid_shared_slice = MirSliceType {
        element_ty: i32_ty.into(),
        is_mutable: true,
        kind: MirPointerKind::SharedRef,
    };
    let valid_erased_mut_slice = MirSliceType {
        element_ty: i32_ty.into(),
        is_mutable: true,
        kind: MirPointerKind::Erased,
    };
    assert!(invalid_shared_slice.verify(&ctx).is_err());
    assert!(valid_erased_mut_slice.verify(&ctx).is_ok());
}

#[test]
fn test_alloca_cannot_claim_a_rust_pointer_kind() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let erased = MirPtrType::get_generic(&mut ctx, i32_ty.into(), true);
    let erased_alloca = Operation::new(
        &mut ctx,
        MirAllocaOp::get_concrete_op_info(),
        vec![erased.into()],
        vec![],
        vec![],
        0,
    );
    assert!(MirAllocaOp::new(erased_alloca).verify(&ctx).is_ok());

    let immutable_erased = MirPtrType::get_generic(&mut ctx, i32_ty.into(), false);
    let immutable_alloca = Operation::new(
        &mut ctx,
        MirAllocaOp::get_concrete_op_info(),
        vec![immutable_erased.into()],
        vec![],
        vec![],
        0,
    );
    assert!(
        MirAllocaOp::new(immutable_alloca).verify(&ctx).is_err(),
        "an alloca cannot masquerade as the immutable canonical function-pointer carrier"
    );

    let shared_erased = MirPtrType::get_shared(&mut ctx, i32_ty.into(), true);
    let shared_alloca = Operation::new(
        &mut ctx,
        MirAllocaOp::get_concrete_op_info(),
        vec![shared_erased.into()],
        vec![],
        vec![],
        0,
    );
    assert!(
        MirAllocaOp::new(shared_alloca).verify(&ctx).is_err(),
        "a stack allocation must remain in generic address space"
    );

    let unique =
        MirPtrType::get_generic_with_kind(&mut ctx, i32_ty.into(), true, MirPointerKind::UniqueRef);
    let unique_alloca = Operation::new(
        &mut ctx,
        MirAllocaOp::get_concrete_op_info(),
        vec![unique.into()],
        vec![],
        vec![],
        0,
    );
    assert!(
        MirAllocaOp::new(unique_alloca).verify(&ctx).is_err(),
        "compiler storage must not manufacture UniqueRef"
    );
}

#[test]
fn test_shared_alloc_result_pointee_must_match_element_type() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Signless).into();
    let i64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signless).into();
    let usize_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let mismatched_result = MirPtrType::get_shared(&mut ctx, i64_ty, true);
    let op = Operation::new(
        &mut ctx,
        MirSharedAllocOp::get_concrete_op_info(),
        vec![mismatched_result.into()],
        vec![],
        vec![],
        0,
    );
    let alloc = MirSharedAllocOp::new(op);
    alloc.set_attr_elem_type(&ctx, TypeAttr::new(i8_ty));
    alloc.set_attr_size(
        &ctx,
        IntegerAttr::new(usize_ty, APInt::from_u64(1, NonZeroUsize::new(64).unwrap())),
    );
    alloc.set_attr_alloc_key(&ctx, StringAttr::new("mismatched-shared".to_string()));

    let error = alloc
        .verify(&ctx)
        .expect_err("shared storage cannot claim an unrelated result pointee type");
    assert!(
        error
            .to_string()
            .contains("pointee type must match elem_type"),
        "{error}"
    );
}

#[test]
fn test_pointer_kind_laundering_requires_explicit_authority() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let usize_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let raw_mut_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, i32_ty.into(), true, MirPointerKind::RawMut);
    let erased_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), true);
    let erased_read_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), false);
    let unique_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, i32_ty.into(), true, MirPointerKind::UniqueRef);
    let shared_ty = MirPtrType::get_generic_with_kind(
        &mut ctx,
        i32_ty.into(),
        false,
        MirPointerKind::SharedRef,
    );
    let raw_const_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, i32_ty.into(), false, MirPointerKind::RawConst);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            raw_mut_ty.into(),
            erased_ty.into(),
            usize_ty.into(),
            shared_ty.into(),
            raw_const_ty.into(),
            erased_read_ty.into(),
        ],
    );
    let raw_mut = block.deref(&ctx).get_argument(0);
    let erased = block.deref(&ctx).get_argument(1);
    let offset = block.deref(&ctx).get_argument(2);
    let shared = block.deref(&ctx).get_argument(3);
    let raw_const = block.deref(&ctx).get_argument(4);
    let erased_read = block.deref(&ctx).get_argument(5);

    // The concrete laundering example: pointer arithmetic is not a Rust
    // reborrow and cannot manufacture `&mut T` from `*mut T`.
    let raw_offset_to_unique = Operation::new(
        &mut ctx,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![unique_ty.into()],
        vec![raw_mut, offset],
        vec![],
        0,
    );
    assert!(
        MirPtrOffsetOp::new(raw_offset_to_unique)
            .verify(&ctx)
            .is_err(),
        "ptr_offset must not invent UniqueRef from RawMut"
    );

    let erased_offset_to_unique = Operation::new(
        &mut ctx,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![unique_ty.into()],
        vec![erased, offset],
        vec![],
        0,
    );
    assert!(
        MirPtrOffsetOp::new(erased_offset_to_unique)
            .verify(&ctx)
            .is_err(),
        "ptr_offset must not recover UniqueRef from Erased"
    );

    let preserving_offset = Operation::new(
        &mut ctx,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![raw_mut_ty.into()],
        vec![raw_mut, offset],
        vec![],
        0,
    );
    assert!(MirPtrOffsetOp::new(preserving_offset).verify(&ctx).is_ok());

    let erasing_offset = Operation::new(
        &mut ctx,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![erased_ty.into()],
        vec![raw_mut, offset],
        vec![],
        0,
    );
    assert!(MirPtrOffsetOp::new(erasing_offset).verify(&ctx).is_ok());

    let mutability_launder = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![erased_ty.into()],
        vec![erased_read],
        vec![],
        0,
    );
    let mutability_launder_cast = MirCastOp::new(mutability_launder);
    mutability_launder_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    assert!(
        mutability_launder_cast.verify(&ctx).is_err(),
        "an unmarked cast cannot manufacture writable Erased evidence"
    );

    let reborrow = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![unique_ty.into()],
        vec![raw_mut],
        vec![],
        0,
    );
    let reborrow_cast = MirCastOp::new(reborrow);
    reborrow_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    assert!(
        reborrow_cast.verify(&ctx).is_err(),
        "an unmarked cast must not manufacture UniqueRef"
    );
    reborrow_cast.set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::Reborrow);
    assert!(
        reborrow_cast.verify(&ctx).is_ok(),
        "a rustc-declared reborrow is the explicit authority"
    );

    for (target, authority) in [
        (unique_ty.into(), MirPointerKindAuthorityAttr::Reborrow),
        (raw_mut_ty.into(), MirPointerKindAuthorityAttr::RawAddress),
    ] {
        let mutable_storage_boundary = Operation::new(
            &mut ctx,
            MirCastOp::get_concrete_op_info(),
            vec![target],
            vec![erased],
            vec![],
            0,
        );
        let mutable_storage_boundary_cast = MirCastOp::new(mutable_storage_boundary);
        mutable_storage_boundary_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
        mutable_storage_boundary_cast.set_pointer_kind_authority(&mut ctx, authority);
        assert!(
            mutable_storage_boundary_cast.verify(&ctx).is_ok(),
            "mutable compiler storage is a valid source for a mutable Rust boundary"
        );
    }

    let raw_to_shared_static = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![shared_ty.into()],
        vec![raw_const],
        vec![],
        0,
    );
    let raw_to_shared_static_cast = MirCastOp::new(raw_to_shared_static);
    raw_to_shared_static_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    raw_to_shared_static_cast
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::StaticAddress);
    assert!(
        raw_to_shared_static_cast.verify(&ctx).is_err(),
        "StaticAddress may establish a typed static value only from Erased storage, not relabel an arbitrary raw pointer as SharedRef"
    );

    let raw_to_erased = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![erased_read_ty.into()],
        vec![raw_const],
        vec![],
        0,
    );
    let raw_to_erased_cast = MirCastOp::new(raw_to_erased);
    raw_to_erased_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    assert!(raw_to_erased_cast.verify(&ctx).is_ok());
    let erased_from_raw = raw_to_erased.deref(&ctx).get_result(0);
    let laundered_static = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![shared_ty.into()],
        vec![erased_from_raw],
        vec![],
        0,
    );
    let laundered_static_cast = MirCastOp::new(laundered_static);
    laundered_static_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    laundered_static_cast
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::StaticAddress);
    assert!(
        laundered_static_cast.verify(&ctx).is_err(),
        "erasing RawConst must not make it valid StaticAddress storage"
    );

    let raw_mut_to_erased = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![erased_ty.into()],
        vec![raw_mut],
        vec![],
        0,
    );
    let raw_mut_to_erased_cast = MirCastOp::new(raw_mut_to_erased);
    raw_mut_to_erased_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    assert!(raw_mut_to_erased_cast.verify(&ctx).is_ok());
    let erased_from_raw_mut = raw_mut_to_erased.deref(&ctx).get_result(0);
    let laundered_abi = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![unique_ty.into()],
        vec![erased_from_raw_mut],
        vec![],
        0,
    );
    let laundered_abi_cast = MirCastOp::new(laundered_abi);
    laundered_abi_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    laundered_abi_cast
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::AbiBoundary);
    assert!(
        laundered_abi_cast.verify(&ctx).is_err(),
        "erasing RawMut must not let AbiBoundary manufacture UniqueRef"
    );

    let global_erased_ty = MirPtrType::get_with_kind(
        &mut ctx,
        i32_ty.into(),
        false,
        dialect_mir::types::address_space::GLOBAL,
        MirPointerKind::Erased,
    );
    let global_op = Operation::new(
        &mut ctx,
        MirGlobalAllocOp::get_concrete_op_info(),
        vec![global_erased_ty.into()],
        vec![],
        vec![],
        0,
    );
    let global = MirGlobalAllocOp::new(global_op);
    global.set_attr_global_type(&ctx, TypeAttr::new(i32_ty.into()));
    global.set_attr_global_key(&ctx, StringAttr::new("lineage-global".to_string()));
    assert!(global.verify(&ctx).is_ok());
    let global_storage = global_op.deref(&ctx).get_result(0);
    let typed_global = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![shared_ty.into()],
        vec![global_storage],
        vec![],
        0,
    );
    let typed_global_cast = MirCastOp::new(typed_global);
    typed_global_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    typed_global_cast
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::StaticAddress);
    assert!(
        typed_global_cast.verify(&ctx).is_ok(),
        "StaticAddress accepts a verified global-allocation root"
    );

    let shared_erased_ty = MirPtrType::get_shared(&mut ctx, i32_ty.into(), true);
    let shared_op = Operation::new(
        &mut ctx,
        MirSharedAllocOp::get_concrete_op_info(),
        vec![shared_erased_ty.into()],
        vec![],
        vec![],
        0,
    );
    let shared_alloc = MirSharedAllocOp::new(shared_op);
    shared_alloc.set_attr_elem_type(&ctx, TypeAttr::new(i32_ty.into()));
    shared_alloc.set_attr_size(
        &ctx,
        IntegerAttr::new(usize_ty, APInt::from_u64(1, NonZeroUsize::new(64).unwrap())),
    );
    shared_alloc.set_attr_alloc_key(&ctx, StringAttr::new("lineage-shared".to_string()));
    assert!(shared_alloc.verify(&ctx).is_ok());
    let shared_storage = shared_op.deref(&ctx).get_result(0);
    let typed_shared = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![raw_mut_ty.into()],
        vec![shared_storage],
        vec![],
        0,
    );
    let typed_shared_cast = MirCastOp::new(typed_shared);
    typed_shared_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    typed_shared_cast
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::StaticAddress);
    assert!(
        typed_shared_cast.verify(&ctx).is_ok(),
        "StaticAddress accepts a verified shared-allocation root"
    );

    let alloca_op = Operation::new(
        &mut ctx,
        MirAllocaOp::get_concrete_op_info(),
        vec![erased_ty.into()],
        vec![],
        vec![],
        0,
    );
    let alloca = MirAllocaOp::new(alloca_op);
    assert!(alloca.verify(&ctx).is_ok());
    let compiler_storage = alloca_op.deref(&ctx).get_result(0);
    let typed_abi = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![raw_mut_ty.into()],
        vec![compiler_storage],
        vec![],
        0,
    );
    let typed_abi_cast = MirCastOp::new(typed_abi);
    typed_abi_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    typed_abi_cast.set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::AbiBoundary);
    assert!(
        typed_abi_cast.verify(&ctx).is_ok(),
        "AbiBoundary accepts verified compiler-owned alloca storage"
    );

    for (target, authority) in [
        (unique_ty.into(), MirPointerKindAuthorityAttr::Reborrow),
        (raw_mut_ty.into(), MirPointerKindAuthorityAttr::RawAddress),
        (
            raw_mut_ty.into(),
            MirPointerKindAuthorityAttr::StaticAddress,
        ),
        (raw_mut_ty.into(), MirPointerKindAuthorityAttr::AbiBoundary),
    ] {
        let immutable_storage_boundary = Operation::new(
            &mut ctx,
            MirCastOp::get_concrete_op_info(),
            vec![target],
            vec![erased_read],
            vec![],
            0,
        );
        let immutable_storage_boundary_cast = MirCastOp::new(immutable_storage_boundary);
        immutable_storage_boundary_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
        immutable_storage_boundary_cast.set_pointer_kind_authority(&mut ctx, authority);
        assert!(
            immutable_storage_boundary_cast.verify(&ctx).is_err(),
            "an immutable Erased thin pointer cannot establish a mutable Rust pointer kind"
        );
    }

    let wrong_authority = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![unique_ty.into()],
        vec![erased],
        vec![],
        0,
    );
    let wrong_authority_cast = MirCastOp::new(wrong_authority);
    wrong_authority_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    wrong_authority_cast
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::RawAddress);
    assert!(
        wrong_authority_cast.verify(&ctx).is_err(),
        "RawAddress authority cannot establish a reference kind"
    );

    let inline_asm_authority = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![unique_ty.into()],
        vec![erased],
        vec![],
        0,
    );
    let inline_asm_authority_cast = MirCastOp::new(inline_asm_authority);
    inline_asm_authority_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    inline_asm_authority_cast
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::InlineAsm);
    assert!(
        inline_asm_authority_cast.verify(&ctx).is_err(),
        "InlineAsm is a producer authority and must never authorize a cast"
    );

    let integer_reborrow = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![unique_ty.into()],
        vec![offset],
        vec![],
        0,
    );
    let integer_reborrow_cast = MirCastOp::new(integer_reborrow);
    integer_reborrow_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    integer_reborrow_cast
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::Reborrow);
    assert!(
        integer_reborrow_cast.verify(&ctx).is_err(),
        "Reborrow authority cannot reinterpret an integer as a Rust reference"
    );
    integer_reborrow_cast
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::RustCast);
    assert!(
        integer_reborrow_cast.verify(&ctx).is_err(),
        "RustCast authority cannot make an integer a PtrToPtr operand"
    );
    integer_reborrow_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::FnPtrToPtr);
    assert!(
        integer_reborrow_cast.verify(&ctx).is_err(),
        "FnPtrToPtr also requires a real pointer carrier"
    );

    let f32_ty = FP32Type::get(&ctx);
    let wrong_pointee_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, f32_ty.into(), true, MirPointerKind::RawMut);
    let wrong_pointee_block = BasicBlock::new(&mut ctx, None, vec![wrong_pointee_ty.into()]);
    let wrong_pointee = wrong_pointee_block.deref(&ctx).get_argument(0);
    let wrong_pointee_reborrow = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![unique_ty.into()],
        vec![wrong_pointee],
        vec![],
        0,
    );
    let wrong_pointee_reborrow_cast = MirCastOp::new(wrong_pointee_reborrow);
    wrong_pointee_reborrow_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    wrong_pointee_reborrow_cast
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::Reborrow);
    assert!(
        wrong_pointee_reborrow_cast.verify(&ctx).is_err(),
        "Reborrow authority must retain the pointee type"
    );

    for immutable_source in [shared, raw_const] {
        let invalid_unique = Operation::new(
            &mut ctx,
            MirCastOp::get_concrete_op_info(),
            vec![unique_ty.into()],
            vec![immutable_source],
            vec![],
            0,
        );
        let invalid_unique_cast = MirCastOp::new(invalid_unique);
        invalid_unique_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
        invalid_unique_cast
            .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::Reborrow);
        assert!(
            invalid_unique_cast.verify(&ctx).is_err(),
            "an immutable source cannot be relabelled as UniqueRef by Reborrow authority"
        );

        let invalid_raw_mut = Operation::new(
            &mut ctx,
            MirCastOp::get_concrete_op_info(),
            vec![raw_mut_ty.into()],
            vec![immutable_source],
            vec![],
            0,
        );
        let invalid_raw_mut_cast = MirCastOp::new(invalid_raw_mut);
        invalid_raw_mut_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
        invalid_raw_mut_cast
            .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::RawAddress);
        assert!(
            invalid_raw_mut_cast.verify(&ctx).is_err(),
            "an immutable source cannot be relabelled as RawMut by RawAddress authority"
        );
    }
}

#[test]
fn test_promoted_empty_mutable_reference_is_a_narrow_static_exception() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let empty_array_ty = MirArrayType::get(&mut ctx, i32_ty.into(), 0);
    let nonempty_array_ty = MirArrayType::get(&mut ctx, i32_ty.into(), 1);

    let empty_storage_ty = MirPtrType::get_with_kind(
        &mut ctx,
        empty_array_ty.into(),
        false,
        dialect_mir::types::address_space::GLOBAL,
        MirPointerKind::Erased,
    );
    let empty_unique_ty = MirPtrType::get_generic_with_kind(
        &mut ctx,
        empty_array_ty.into(),
        true,
        MirPointerKind::UniqueRef,
    );
    let empty_global_op = Operation::new(
        &mut ctx,
        MirGlobalAllocOp::get_concrete_op_info(),
        vec![empty_storage_ty.into()],
        vec![],
        vec![],
        0,
    );
    let empty_global = MirGlobalAllocOp::new(empty_global_op);
    empty_global.set_attr_global_type(&ctx, TypeAttr::new(empty_array_ty.into()));
    empty_global.set_attr_global_key(
        &ctx,
        StringAttr::new("promoted-empty-mutable-reference".to_string()),
    );
    empty_global.set_alignment_value(&mut ctx, 4);
    empty_global_op.deref_mut(&ctx).attributes.set(
        "global_initializer_hex".try_into().unwrap(),
        StringAttr::new(String::new()),
    );
    empty_global.mark_immutable(&mut ctx);
    assert!(empty_global.verify(&ctx).is_ok());

    let empty_global_storage = empty_global_op.deref(&ctx).get_result(0);
    let empty_static_borrow_op = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![empty_unique_ty.into()],
        vec![empty_global_storage],
        vec![],
        0,
    );
    let empty_static_borrow = MirCastOp::new(empty_static_borrow_op);
    empty_static_borrow.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    empty_static_borrow
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::StaticAddress);

    empty_global.set_alignment_value(&mut ctx, 1);
    assert!(
        empty_static_borrow.verify(&ctx).is_err(),
        "even [i32; 0] must retain i32's natural four-byte alignment"
    );
    empty_global.set_alignment_value(&mut ctx, 4);
    assert!(
        empty_static_borrow.verify(&ctx).is_ok(),
        "an immutable promoted [T; 0] global may back rustc's vacuous &mut []"
    );

    empty_global_op.deref_mut(&ctx).attributes.set(
        "global_initializer_hex".try_into().unwrap(),
        StringAttr::new("00".to_string()),
    );
    assert!(
        empty_static_borrow.verify(&ctx).is_err(),
        "the exception requires an actually empty initializer, not merely the attribute"
    );
    empty_global_op.deref_mut(&ctx).attributes.set(
        "global_initializer_hex".try_into().unwrap(),
        StringAttr::new(String::new()),
    );
    empty_global_op.deref_mut(&ctx).attributes.set(
        "global_initializer_relocations".try_into().unwrap(),
        StringAttr::new("unexpected-relocation".to_string()),
    );
    assert!(
        empty_static_borrow.verify(&ctx).is_err(),
        "zero-byte promoted storage cannot carry a relocation"
    );

    let mutable_empty_global_op = Operation::new(
        &mut ctx,
        MirGlobalAllocOp::get_concrete_op_info(),
        vec![empty_storage_ty.into()],
        vec![],
        vec![],
        0,
    );
    let mutable_empty_global = MirGlobalAllocOp::new(mutable_empty_global_op);
    mutable_empty_global.set_attr_global_type(&ctx, TypeAttr::new(empty_array_ty.into()));
    mutable_empty_global.set_attr_global_key(
        &ctx,
        StringAttr::new("non-promoted-empty-global".to_string()),
    );
    assert!(mutable_empty_global.verify(&ctx).is_ok());
    let mutable_empty_storage = mutable_empty_global_op.deref(&ctx).get_result(0);
    let mutable_empty_borrow_op = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![empty_unique_ty.into()],
        vec![mutable_empty_storage],
        vec![],
        0,
    );
    let mutable_empty_borrow = MirCastOp::new(mutable_empty_borrow_op);
    mutable_empty_borrow.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    mutable_empty_borrow
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::StaticAddress);
    assert!(
        mutable_empty_borrow.verify(&ctx).is_err(),
        "the [T; 0] exception requires compiler-promoted immutable storage"
    );

    let aligned_element_ty: TypeHandle = MirStructType::get_with_full_layout(
        &mut ctx,
        "Align16".to_string(),
        vec!["value".to_string()],
        vec![i32_ty.into()],
        vec![],
        vec![0],
        16,
        16,
    )
    .into();
    let aligned_empty_array_ty = MirArrayType::get(&mut ctx, aligned_element_ty, 0);
    let underaligned_storage_ty = MirPtrType::get_with_kind(
        &mut ctx,
        aligned_empty_array_ty.into(),
        false,
        dialect_mir::types::address_space::GLOBAL,
        MirPointerKind::Erased,
    );
    let aligned_empty_unique_ty = MirPtrType::get_generic_with_kind(
        &mut ctx,
        aligned_empty_array_ty.into(),
        true,
        MirPointerKind::UniqueRef,
    );
    let underaligned_global_op = Operation::new(
        &mut ctx,
        MirGlobalAllocOp::get_concrete_op_info(),
        vec![underaligned_storage_ty.into()],
        vec![],
        vec![],
        0,
    );
    let underaligned_global = MirGlobalAllocOp::new(underaligned_global_op);
    underaligned_global.set_attr_global_type(&ctx, TypeAttr::new(aligned_empty_array_ty.into()));
    underaligned_global.set_attr_global_key(
        &ctx,
        StringAttr::new("underaligned-empty-reference".to_string()),
    );
    underaligned_global.set_alignment_value(&mut ctx, 1);
    underaligned_global_op.deref_mut(&ctx).attributes.set(
        "global_initializer_hex".try_into().unwrap(),
        StringAttr::new(String::new()),
    );
    underaligned_global.mark_immutable(&mut ctx);
    assert!(underaligned_global.verify(&ctx).is_ok());
    let underaligned_storage = underaligned_global_op.deref(&ctx).get_result(0);
    let underaligned_borrow_op = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![aligned_empty_unique_ty.into()],
        vec![underaligned_storage],
        vec![],
        0,
    );
    let underaligned_borrow = MirCastOp::new(underaligned_borrow_op);
    underaligned_borrow.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    underaligned_borrow
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::StaticAddress);
    assert!(
        underaligned_borrow.verify(&ctx).is_err(),
        "promoted &mut [Align16; 0] requires at least 16-byte global alignment"
    );

    let nonempty_storage_ty = MirPtrType::get_with_kind(
        &mut ctx,
        nonempty_array_ty.into(),
        false,
        dialect_mir::types::address_space::GLOBAL,
        MirPointerKind::Erased,
    );
    let nonempty_unique_ty = MirPtrType::get_generic_with_kind(
        &mut ctx,
        nonempty_array_ty.into(),
        true,
        MirPointerKind::UniqueRef,
    );
    let nonempty_global_op = Operation::new(
        &mut ctx,
        MirGlobalAllocOp::get_concrete_op_info(),
        vec![nonempty_storage_ty.into()],
        vec![],
        vec![],
        0,
    );
    let nonempty_global = MirGlobalAllocOp::new(nonempty_global_op);
    nonempty_global.set_attr_global_type(&ctx, TypeAttr::new(nonempty_array_ty.into()));
    nonempty_global.set_attr_global_key(
        &ctx,
        StringAttr::new("promoted-nonempty-mutable-reference".to_string()),
    );
    nonempty_global.mark_immutable(&mut ctx);
    assert!(nonempty_global.verify(&ctx).is_ok());
    let nonempty_global_storage = nonempty_global_op.deref(&ctx).get_result(0);
    let nonempty_static_borrow_op = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![nonempty_unique_ty.into()],
        vec![nonempty_global_storage],
        vec![],
        0,
    );
    let nonempty_static_borrow = MirCastOp::new(nonempty_static_borrow_op);
    nonempty_static_borrow.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    nonempty_static_borrow
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::StaticAddress);
    assert!(
        nonempty_static_borrow.verify(&ctx).is_err(),
        "StaticAddress must never manufacture UniqueRef for non-empty promoted storage"
    );

    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![empty_storage_ty.into(), empty_unique_ty.into()],
    );
    let erased_block_argument = block.deref(&ctx).get_argument(0);
    let block_argument_borrow_op = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![empty_unique_ty.into()],
        vec![erased_block_argument],
        vec![],
        0,
    );
    let block_argument_borrow = MirCastOp::new(block_argument_borrow_op);
    block_argument_borrow.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    block_argument_borrow
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::StaticAddress);
    assert!(
        block_argument_borrow.verify(&ctx).is_err(),
        "an Erased [T; 0] block argument has no proven promoted-global lineage"
    );

    let raw_empty_ty = MirPtrType::get_generic_with_kind(
        &mut ctx,
        empty_array_ty.into(),
        false,
        MirPointerKind::RawConst,
    );
    let raw_block = BasicBlock::new(&mut ctx, None, vec![raw_empty_ty.into()]);
    let raw_empty = raw_block.deref(&ctx).get_argument(0);
    let erase_raw_op = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![empty_storage_ty.into()],
        vec![raw_empty],
        vec![],
        0,
    );
    let erase_raw = MirCastOp::new(erase_raw_op);
    erase_raw.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    assert!(erase_raw.verify(&ctx).is_ok());
    let erased_from_raw = erase_raw_op.deref(&ctx).get_result(0);
    let laundered_borrow_op = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![empty_unique_ty.into()],
        vec![erased_from_raw],
        vec![],
        0,
    );
    let laundered_borrow = MirCastOp::new(laundered_borrow_op);
    laundered_borrow.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    laundered_borrow
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::StaticAddress);
    assert!(
        laundered_borrow.verify(&ctx).is_err(),
        "RawConst -> Erased must not launder a zero-length pointer into UniqueRef"
    );

    let byte_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let byte_storage_ty = MirPtrType::get_with_kind(
        &mut ctx,
        byte_ty.into(),
        false,
        dialect_mir::types::address_space::GLOBAL,
        MirPointerKind::Erased,
    );
    let byte_global_op = Operation::new(
        &mut ctx,
        MirGlobalAllocOp::get_concrete_op_info(),
        vec![byte_storage_ty.into()],
        vec![],
        vec![],
        0,
    );
    let byte_global = MirGlobalAllocOp::new(byte_global_op);
    byte_global.set_attr_global_type(&ctx, TypeAttr::new(byte_ty.into()));
    byte_global.set_attr_global_key(
        &ctx,
        StringAttr::new("misaligned-empty-reference-root".to_string()),
    );
    byte_global.mark_immutable(&mut ctx);
    assert!(byte_global.verify(&ctx).is_ok());
    let byte_storage = byte_global_op.deref(&ctx).get_result(0);
    let retype_root_op = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![empty_storage_ty.into()],
        vec![byte_storage],
        vec![],
        0,
    );
    let retype_root = MirCastOp::new(retype_root_op);
    retype_root.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    assert!(retype_root.verify(&ctx).is_ok());
    let retyped_storage = retype_root_op.deref(&ctx).get_result(0);
    let misaligned_borrow_op = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![empty_unique_ty.into()],
        vec![retyped_storage],
        vec![],
        0,
    );
    let misaligned_borrow = MirCastOp::new(misaligned_borrow_op);
    misaligned_borrow.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    misaligned_borrow
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::StaticAddress);
    assert!(
        misaligned_borrow.verify(&ctx).is_err(),
        "an immutable byte global cannot be retyped into an aligned &mut [i32; 0] capability"
    );
}

fn promoted_empty_unique_ref_verifies(
    ctx: &mut Context,
    element_ty: TypeHandle,
    alignment: u64,
) -> bool {
    let empty_array_ty: TypeHandle = MirArrayType::get(ctx, element_ty, 0).into();
    let storage_ty = MirPtrType::get_with_kind(
        ctx,
        empty_array_ty,
        false,
        dialect_mir::types::address_space::GLOBAL,
        MirPointerKind::Erased,
    );
    let unique_ty =
        MirPtrType::get_generic_with_kind(ctx, empty_array_ty, true, MirPointerKind::UniqueRef);
    let global_op = Operation::new(
        ctx,
        MirGlobalAllocOp::get_concrete_op_info(),
        vec![storage_ty.into()],
        vec![],
        vec![],
        0,
    );
    let global = MirGlobalAllocOp::new(global_op);
    global.set_attr_global_type(ctx, TypeAttr::new(empty_array_ty));
    global.set_attr_global_key(ctx, StringAttr::new("promoted-empty-shape".to_string()));
    global.set_alignment_value(ctx, alignment);
    global_op.deref_mut(ctx).attributes.set(
        "global_initializer_hex".try_into().unwrap(),
        StringAttr::new(String::new()),
    );
    global.mark_immutable(ctx);

    let storage = global_op.deref(ctx).get_result(0);
    let cast_op = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![unique_ty.into()],
        vec![storage],
        vec![],
        0,
    );
    let cast = MirCastOp::new(cast_op);
    cast.set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
    cast.set_pointer_kind_authority(ctx, MirPointerKindAuthorityAttr::StaticAddress);
    cast.verify(ctx).is_ok()
}

#[test]
fn test_promoted_empty_alignment_covers_supported_fat_and_unit_elements() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let byte: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
    let word: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
    let unit: TypeHandle = MirTupleType::get(&mut ctx, vec![]).into();
    let slice: TypeHandle =
        MirSliceType::get_with_kind(&mut ctx, byte, MirPointerKind::SharedRef).into();
    let disjoint: TypeHandle = MirDisjointSliceType::get(&mut ctx, byte).into();

    assert!(promoted_empty_unique_ref_verifies(&mut ctx, unit, 1));
    assert!(promoted_empty_unique_ref_verifies(&mut ctx, slice, 8));
    assert!(promoted_empty_unique_ref_verifies(&mut ctx, disjoint, 8));
    assert!(promoted_empty_unique_ref_verifies(&mut ctx, word, 8));
    assert!(
        !promoted_empty_unique_ref_verifies(&mut ctx, word, 12),
        "a non-power-of-two numeric value is not a valid alignment guarantee"
    );
    assert!(
        !promoted_empty_unique_ref_verifies(&mut ctx, slice, 4),
        "a stored slice value still requires pointer-word alignment"
    );
}

#[test]
fn test_mir_ref_requires_exact_pointer_kind_authority() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let shared_ty = MirPtrType::get_generic_with_kind(
        &mut ctx,
        i32_ty.into(),
        false,
        MirPointerKind::SharedRef,
    );
    let unique_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, i32_ty.into(), true, MirPointerKind::UniqueRef);
    let raw_const_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, i32_ty.into(), false, MirPointerKind::RawConst);
    let raw_mut_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, i32_ty.into(), true, MirPointerKind::RawMut);
    let erased_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), false);
    let global_shared_ty = MirPtrType::get_with_kind(
        &mut ctx,
        i32_ty.into(),
        false,
        dialect_mir::types::address_space::GLOBAL,
        MirPointerKind::SharedRef,
    );
    let block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let value = block.deref(&ctx).get_argument(0);

    let build =
        |ctx: &mut Context, result_ty, mutable, authority: Option<MirPointerKindAuthorityAttr>| {
            let op = Operation::new(
                ctx,
                MirRefOp::get_concrete_op_info(),
                vec![result_ty],
                vec![value],
                vec![],
                0,
            );
            let reference = MirRefOp::new(op);
            reference.set_mutable(ctx, mutable);
            if let Some(authority) = authority {
                reference.set_pointer_kind_authority(ctx, authority);
            }
            op
        };

    for (result_ty, mutable, authority) in [
        (
            shared_ty.into(),
            false,
            MirPointerKindAuthorityAttr::Reborrow,
        ),
        (
            unique_ty.into(),
            true,
            MirPointerKindAuthorityAttr::Reborrow,
        ),
        (
            raw_const_ty.into(),
            false,
            MirPointerKindAuthorityAttr::RawAddress,
        ),
        (
            raw_mut_ty.into(),
            true,
            MirPointerKindAuthorityAttr::RawAddress,
        ),
        (
            shared_ty.into(),
            false,
            MirPointerKindAuthorityAttr::StaticAddress,
        ),
    ] {
        assert!(
            MirRefOp::new(build(&mut ctx, result_ty, mutable, Some(authority)))
                .verify(&ctx)
                .is_ok()
        );
    }

    assert!(
        MirRefOp::new(build(&mut ctx, shared_ty.into(), false, None))
            .verify(&ctx)
            .is_err(),
        "mir.ref must visibly identify its Rust semantic origin"
    );
    assert!(
        MirRefOp::new(build(&mut ctx, erased_ty.into(), false, None))
            .verify(&ctx)
            .is_err(),
        "mir.ref is an explicit pointer-creation boundary, not generic Erased storage"
    );
    assert!(
        MirRefOp::new(build(
            &mut ctx,
            shared_ty.into(),
            false,
            Some(MirPointerKindAuthorityAttr::RawAddress),
        ))
        .verify(&ctx)
        .is_err(),
        "RawAddress cannot manufacture SharedRef"
    );
    assert!(
        MirRefOp::new(build(
            &mut ctx,
            unique_ty.into(),
            true,
            Some(MirPointerKindAuthorityAttr::StaticAddress),
        ))
        .verify(&ctx)
        .is_err(),
        "constant/static materialization cannot manufacture uniqueness"
    );
    assert!(
        MirRefOp::new(build(
            &mut ctx,
            unique_ty.into(),
            false,
            Some(MirPointerKindAuthorityAttr::Reborrow),
        ))
        .verify(&ctx)
        .is_err(),
        "reference kind must agree with the operation's mutability"
    );
    assert!(
        MirRefOp::new(build(
            &mut ctx,
            global_shared_ty.into(),
            false,
            Some(MirPointerKindAuthorityAttr::Reborrow),
        ))
        .verify(&ctx)
        .is_err(),
        "mir.ref materializes generic stack storage"
    );
}
