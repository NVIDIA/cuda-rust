/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_mir::{
    attributes::{FieldIndexAttr, MirCastKindAttr, MirPointerKindAuthorityAttr},
    ops::{MirArrayElementAddrOp, MirCastOp, MirExtractFieldOp, MirFieldAddrOp, MirPtrOffsetOp},
    types::{MirArrayType, MirPointerKind, MirPtrType, MirSliceType, MirStructType, MirTupleType},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::TypeAttr,
        types::{IntegerType, Signedness},
    },
    common_traits::Verify,
    context::Context,
    op::Op,
    operation::Operation,
};

#[test]
fn test_aggregate_cast_cannot_hide_pointer_kind_laundering() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let raw_mut_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, i32_ty.into(), true, MirPointerKind::RawMut);
    let unique_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, i32_ty.into(), true, MirPointerKind::UniqueRef);
    let raw_tuple_ty = MirTupleType::get(&mut ctx, vec![raw_mut_ty.into()]);
    let unique_tuple_ty = MirTupleType::get(&mut ctx, vec![unique_ty.into()]);
    let block = BasicBlock::new(&mut ctx, None, vec![raw_tuple_ty.into()]);
    let raw_tuple = block.deref(&ctx).get_argument(0);

    let build = |ctx: &mut Context, authority: Option<MirPointerKindAuthorityAttr>| {
        let op = Operation::new(
            ctx,
            MirCastOp::get_concrete_op_info(),
            vec![unique_tuple_ty.into()],
            vec![raw_tuple],
            vec![],
            0,
        );
        let cast = MirCastOp::new(op);
        cast.set_attr_cast_kind(ctx, MirCastKindAttr::Transmute);
        if let Some(authority) = authority {
            cast.set_pointer_kind_authority(ctx, authority);
        }
        op
    };

    assert!(
        MirCastOp::new(build(&mut ctx, None)).verify(&ctx).is_err(),
        "nested RawMut -> UniqueRef laundering must be rejected"
    );
    assert!(
        MirCastOp::new(build(&mut ctx, Some(MirPointerKindAuthorityAttr::Reborrow),))
            .verify(&ctx)
            .is_err(),
        "Rvalue::Ref cannot authorize a nested aggregate transition"
    );
    assert!(
        MirCastOp::new(build(&mut ctx, Some(MirPointerKindAuthorityAttr::RustCast),))
            .verify(&ctx)
            .is_ok(),
        "an explicit rustc aggregate transmute is visible and authorized"
    );
}

#[test]
fn test_aggregate_reinterpretation_requires_rust_cast_authority() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let shared_ty = MirPtrType::get_generic_with_kind(
        &mut ctx,
        i64_ty.into(),
        false,
        MirPointerKind::SharedRef,
    );
    // Both tuple declarations list the same pointer at field 0, but its byte
    // offset changes from 0 to 8. Pairing by declaration index alone would
    // therefore mistake an integer's old bytes for a preserved reference.
    let source_ty = MirTupleType::get_with_layout(
        &mut ctx,
        vec![shared_ty.into(), i64_ty.into()],
        vec![0, 1],
        vec![0, 8],
        16,
        8,
    );
    let target_ty = MirTupleType::get_with_layout(
        &mut ctx,
        vec![shared_ty.into(), i64_ty.into()],
        vec![1, 0],
        vec![8, 0],
        16,
        8,
    );
    let block = BasicBlock::new(&mut ctx, None, vec![source_ty.into()]);
    let source = block.deref(&ctx).get_argument(0);

    let build = |ctx: &mut Context, authority: Option<MirPointerKindAuthorityAttr>| {
        let op = Operation::new(
            ctx,
            MirCastOp::get_concrete_op_info(),
            vec![target_ty.into()],
            vec![source],
            vec![],
            0,
        );
        let cast = MirCastOp::new(op);
        cast.set_attr_cast_kind(ctx, MirCastKindAttr::Transmute);
        if let Some(authority) = authority {
            cast.set_pointer_kind_authority(ctx, authority);
        }
        op
    };

    assert!(
        MirCastOp::new(build(&mut ctx, None)).verify(&ctx).is_err(),
        "layout-changing aggregate casts must not infer pointer preservation by field index"
    );
    assert!(
        MirCastOp::new(build(&mut ctx, Some(MirPointerKindAuthorityAttr::RustCast),))
            .verify(&ctx)
            .is_ok(),
        "an explicit rustc transmute makes the representation reinterpretation auditable"
    );
}

#[test]
fn test_generic_aggregate_cast_cannot_invent_erased_pointer_carriers() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let word_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let erased_mut = MirPtrType::get_generic(&mut ctx, word_ty.into(), true);
    let source_ty = MirTupleType::get_with_layout(
        &mut ctx,
        vec![erased_mut.into(), word_ty.into()],
        vec![0, 1],
        vec![0, 8],
        16,
        8,
    );
    let extra_pointer_ty = MirTupleType::get_with_layout(
        &mut ctx,
        vec![erased_mut.into(), erased_mut.into()],
        vec![0, 1],
        vec![0, 8],
        16,
        8,
    );
    let moved_pointer_ty = MirTupleType::get_with_layout(
        &mut ctx,
        vec![erased_mut.into(), word_ty.into()],
        vec![1, 0],
        vec![8, 0],
        16,
        8,
    );
    let one_pointer = MirArrayType::get(&mut ctx, erased_mut.into(), 1);
    let two_pointers = MirArrayType::get(&mut ctx, erased_mut.into(), 2);

    let source_block = BasicBlock::new(&mut ctx, None, vec![source_ty.into()]);
    let source = source_block.deref(&ctx).get_argument(0);
    let array_block = BasicBlock::new(&mut ctx, None, vec![one_pointer.into()]);
    let array_source = array_block.deref(&ctx).get_argument(0);
    let build = |ctx: &mut Context, source, target| {
        let op = Operation::new(
            ctx,
            MirCastOp::get_concrete_op_info(),
            vec![target],
            vec![source],
            vec![],
            0,
        );
        MirCastOp::new(op).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
        op
    };

    assert!(
        MirCastOp::new(build(&mut ctx, source, extra_pointer_ty.into()))
            .verify(&ctx)
            .is_err(),
        "PtrToPtr cannot reinterpret an integer field as writable Erased pointer evidence"
    );
    assert!(
        MirCastOp::new(build(&mut ctx, source, moved_pointer_ty.into()))
            .verify(&ctx)
            .is_err(),
        "field-index equality cannot hide an Erased pointer moving onto integer bytes"
    );
    assert!(
        MirCastOp::new(build(&mut ctx, array_source, two_pointers.into()))
            .verify(&ctx)
            .is_err(),
        "homogeneous array traversal must retain pointer-carrier cardinality"
    );
}

#[test]
fn test_rust_cast_authority_obeys_cast_kind_semantics() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let u8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let usize_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let array_ty = MirArrayType::get(&mut ctx, u8_ty.into(), 4);
    let raw_mut =
        MirPtrType::get_generic_with_kind(&mut ctx, u8_ty.into(), true, MirPointerKind::RawMut);
    let raw_const =
        MirPtrType::get_generic_with_kind(&mut ctx, u8_ty.into(), false, MirPointerKind::RawConst);
    let unique =
        MirPtrType::get_generic_with_kind(&mut ctx, u8_ty.into(), true, MirPointerKind::UniqueRef);
    let fn_target = MirStructType::get_with_full_layout(
        &mut ctx,
        "FnPtrTarget".into(),
        vec![],
        vec![],
        vec![],
        vec![],
        0,
        0,
    );
    let fn_target_ty: pliron::r#type::TypeHandle = fn_target.into();
    let fn_carrier = MirPtrType::get_generic(&mut ctx, fn_target_ty, false);
    let raw_mut_array =
        MirPtrType::get_generic_with_kind(&mut ctx, array_ty.into(), true, MirPointerKind::RawMut);
    let raw_const_array = MirPtrType::get_generic_with_kind(
        &mut ctx,
        array_ty.into(),
        false,
        MirPointerKind::RawConst,
    );
    let erased_array = MirPtrType::get_generic(&mut ctx, array_ty.into(), false);
    let erased_slice = MirSliceType::get(&mut ctx, u8_ty.into());
    let erased_mut_slice = MirSliceType::get_with_mutability(&mut ctx, u8_ty.into(), true);
    let raw_const_slice =
        MirSliceType::get_with_kind(&mut ctx, u8_ty.into(), MirPointerKind::RawConst);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            raw_mut.into(),
            raw_const.into(),
            fn_carrier.into(),
            raw_mut_array.into(),
            raw_const_array.into(),
            erased_array.into(),
            usize_ty.into(),
        ],
    );
    let raw_mut_value = block.deref(&ctx).get_argument(0);
    let raw_const_value = block.deref(&ctx).get_argument(1);
    let fn_value = block.deref(&ctx).get_argument(2);
    let raw_mut_array_value = block.deref(&ctx).get_argument(3);
    let raw_const_array_value = block.deref(&ctx).get_argument(4);
    let erased_array_value = block.deref(&ctx).get_argument(5);
    let integer_value = block.deref(&ctx).get_argument(6);

    let build = |ctx: &mut Context,
                 source,
                 target,
                 kind,
                 authority: Option<MirPointerKindAuthorityAttr>| {
        let op = Operation::new(
            ctx,
            MirCastOp::get_concrete_op_info(),
            vec![target],
            vec![source],
            vec![],
            0,
        );
        let cast = MirCastOp::new(op);
        cast.set_attr_cast_kind(ctx, kind);
        if let Some(authority) = authority {
            cast.set_pointer_kind_authority(ctx, authority);
        }
        op
    };
    let rust_cast = Some(MirPointerKindAuthorityAttr::RustCast);

    assert!(
        MirCastOp::new(build(
            &mut ctx,
            raw_mut_value,
            unique.into(),
            MirCastKindAttr::PtrToPtr,
            rust_cast.clone(),
        ))
        .verify(&ctx)
        .is_err(),
        "PtrToPtr cannot use RustCast to invent a reference"
    );
    assert!(
        MirCastOp::new(build(
            &mut ctx,
            raw_mut_value,
            raw_const.into(),
            MirCastKindAttr::PtrToPtr,
            rust_cast.clone(),
        ))
        .verify(&ctx)
        .is_ok(),
        "PtrToPtr may perform an explicit raw-to-raw cast"
    );
    assert!(
        MirCastOp::new(build(
            &mut ctx,
            fn_value,
            raw_const.into(),
            MirCastKindAttr::FnPtrToPtr,
            rust_cast.clone(),
        ))
        .verify(&ctx)
        .is_ok(),
        "FnPtrToPtr may expose an opaque function pointer as a raw pointer"
    );
    assert!(
        MirCastOp::new(build(
            &mut ctx,
            erased_array_value,
            raw_const.into(),
            MirCastKindAttr::FnPtrToPtr,
            rust_cast.clone(),
        ))
        .verify(&ctx)
        .is_err(),
        "FnPtrToPtr cannot relabel arbitrary Erased storage as a function pointer"
    );
    assert!(
        MirCastOp::new(build(
            &mut ctx,
            fn_value,
            unique.into(),
            MirCastKindAttr::FnPtrToPtr,
            rust_cast.clone(),
        ))
        .verify(&ctx)
        .is_err()
    );
    assert!(
        MirCastOp::new(build(
            &mut ctx,
            raw_mut_value,
            raw_const.into(),
            MirCastKindAttr::PointerCoercionMutToConst,
            rust_cast.clone(),
        ))
        .verify(&ctx)
        .is_ok()
    );
    assert!(
        MirCastOp::new(build(
            &mut ctx,
            raw_const_value,
            raw_mut.into(),
            MirCastKindAttr::PointerCoercionMutToConst,
            rust_cast.clone(),
        ))
        .verify(&ctx)
        .is_err()
    );
    assert!(
        MirCastOp::new(build(
            &mut ctx,
            raw_mut_value,
            raw_mut.into(),
            MirCastKindAttr::PointerCoercionMutToConst,
            None,
        ))
        .verify(&ctx)
        .is_err(),
        "MutToConst cannot masquerade as a same-category pointer cast"
    );
    assert!(
        MirCastOp::new(build(
            &mut ctx,
            raw_const_array_value,
            raw_const_slice.into(),
            MirCastKindAttr::PointerCoercionUnsize,
            rust_cast.clone(),
        ))
        .verify(&ctx)
        .is_ok(),
        "Unsize may change thin/fat shape while preserving its carrier"
    );
    assert!(
        MirCastOp::new(build(
            &mut ctx,
            erased_array_value,
            erased_slice.into(),
            MirCastKindAttr::PointerCoercionUnsize,
            None,
        ))
        .verify(&ctx)
        .is_ok(),
        "an all-Erased unsize still preserves read-only carrier state"
    );
    assert!(
        MirCastOp::new(build(
            &mut ctx,
            erased_array_value,
            erased_mut_slice.into(),
            MirCastKindAttr::PointerCoercionUnsize,
            None,
        ))
        .verify(&ctx)
        .is_err(),
        "Unsize cannot turn an immutable Erased thin pointer into writable fat evidence"
    );

    let prefix_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let sized_tail = MirStructType::get_with_full_layout(
        &mut ctx,
        "TailCarrier".into(),
        vec!["prefix".into(), "tail".into()],
        vec![prefix_ty.into(), array_ty.into()],
        vec![0, 1],
        vec![0, 8],
        16,
        8,
    );
    let shifted_unsized_tail = MirStructType::get_with_full_layout(
        &mut ctx,
        "TailCarrier".into(),
        vec!["prefix".into(), "tail".into()],
        vec![prefix_ty.into(), u8_ty.into()],
        vec![0, 1],
        vec![0, 16],
        24,
        8,
    );
    let sized_tail_ptr = MirPtrType::get_generic_with_kind(
        &mut ctx,
        sized_tail.into(),
        false,
        MirPointerKind::RawConst,
    );
    let shifted_tail_slice = MirSliceType::get_with_kind(
        &mut ctx,
        shifted_unsized_tail.into(),
        MirPointerKind::RawConst,
    );
    let tail_block = BasicBlock::new(&mut ctx, None, vec![sized_tail_ptr.into()]);
    let tail_value = tail_block.deref(&ctx).get_argument(0);
    assert!(
        MirCastOp::new(build(
            &mut ctx,
            tail_value,
            shifted_tail_slice.into(),
            MirCastKindAttr::PointerCoercionUnsize,
            rust_cast.clone(),
        ))
        .verify(&ctx)
        .is_err(),
        "Unsize cannot move the trailing data behind a changed struct-field offset"
    );

    assert!(
        MirCastOp::new(build(
            &mut ctx,
            raw_const_array_value,
            raw_mut.into(),
            MirCastKindAttr::PointerCoercionArrayToPointer,
            rust_cast.clone(),
        ))
        .verify(&ctx)
        .is_err(),
        "ArrayToPointer cannot strengthen const raw storage to mutable"
    );
    assert!(
        MirCastOp::new(build(
            &mut ctx,
            raw_mut_value,
            raw_const.into(),
            MirCastKindAttr::PointerCoercionArrayToPointer,
            rust_cast.clone(),
        ))
        .verify(&ctx)
        .is_err(),
        "ArrayToPointer requires a raw pointer to an array, not an arbitrary pointer"
    );
    assert!(
        MirCastOp::new(build(
            &mut ctx,
            raw_mut_value,
            raw_mut.into(),
            MirCastKindAttr::PointerCoercionArrayToPointer,
            None,
        ))
        .verify(&ctx)
        .is_err(),
        "an unmarked same-kind ArrayToPointer still requires an array source"
    );
    assert!(
        MirCastOp::new(build(
            &mut ctx,
            raw_mut_array_value,
            raw_const.into(),
            MirCastKindAttr::PointerCoercionArrayToPointer,
            rust_cast.clone(),
        ))
        .verify(&ctx)
        .is_ok()
    );
    assert!(
        MirCastOp::new(build(
            &mut ctx,
            integer_value,
            unique.into(),
            MirCastKindAttr::Transmute,
            rust_cast,
        ))
        .verify(&ctx)
        .is_ok(),
        "only an explicit Transmute may establish an arbitrary pointer category"
    );

    let word_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let source_wrapper = MirStructType::get_with_full_layout(
        &mut ctx,
        "CarrierWrapper".into(),
        vec!["pointer".into(), "word".into()],
        vec![unique.into(), word_ty.into()],
        vec![0, 1],
        vec![0, 8],
        16,
        8,
    );
    let moved_wrapper = MirStructType::get_with_full_layout(
        &mut ctx,
        "CarrierWrapper".into(),
        vec!["word".into(), "pointer".into()],
        vec![word_ty.into(), unique.into()],
        vec![0, 1],
        vec![0, 8],
        16,
        8,
    );
    let wrapper_block = BasicBlock::new(&mut ctx, None, vec![source_wrapper.into()]);
    let wrapper_value = wrapper_block.deref(&ctx).get_argument(0);
    for kind in [
        MirCastKindAttr::PointerCoercionUnsize,
        MirCastKindAttr::Subtype,
    ] {
        assert!(
            MirCastOp::new(build(
                &mut ctx,
                wrapper_value,
                moved_wrapper.into(),
                kind,
                Some(MirPointerKindAuthorityAttr::RustCast),
            ))
            .verify(&ctx)
            .is_err(),
            "Unsize/Subtype cannot bless a pointer carrier moved onto integer bytes"
        );
    }
}

#[test]
fn test_pointer_projections_preserve_or_erase_kind_only() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let usize_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let tuple_ty = MirTupleType::get(&mut ctx, vec![i32_ty.into()]);
    let array_ty = MirArrayType::get(&mut ctx, i32_ty.into(), 4);
    let raw_tuple_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, tuple_ty.into(), true, MirPointerKind::RawMut);
    let raw_array_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, array_ty.into(), true, MirPointerKind::RawMut);
    let raw_field_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, i32_ty.into(), true, MirPointerKind::RawMut);
    let erased_field_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), true);
    let erased_read_field_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), false);
    let unique_field_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, i32_ty.into(), true, MirPointerKind::UniqueRef);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![raw_tuple_ty.into(), raw_array_ty.into(), usize_ty.into()],
    );
    let tuple = block.deref(&ctx).get_argument(0);
    let array = block.deref(&ctx).get_argument(1);
    let index = block.deref(&ctx).get_argument(2);

    let field = |ctx: &mut Context, result_ty| {
        let op = Operation::new(
            ctx,
            MirFieldAddrOp::get_concrete_op_info(),
            vec![result_ty],
            vec![tuple],
            vec![],
            0,
        );
        MirFieldAddrOp::new(op).set_attr_field_index(ctx, FieldIndexAttr(0));
        MirFieldAddrOp::new(op).set_attr_aggregate_ty(ctx, TypeAttr::new(tuple_ty.into()));
        op
    };
    assert!(
        MirFieldAddrOp::new(field(&mut ctx, raw_field_ty.into()))
            .verify(&ctx)
            .is_ok()
    );
    assert!(
        MirFieldAddrOp::new(field(&mut ctx, erased_field_ty.into()))
            .verify(&ctx)
            .is_ok()
    );
    assert!(
        MirFieldAddrOp::new(field(&mut ctx, unique_field_ty.into()))
            .verify(&ctx)
            .is_err()
    );
    assert!(
        MirFieldAddrOp::new(field(&mut ctx, erased_read_field_ty.into()))
            .verify(&ctx)
            .is_err(),
        "field projection cannot flip an Erased address from writable to read-only"
    );

    let array_launder = Operation::new(
        &mut ctx,
        MirArrayElementAddrOp::get_concrete_op_info(),
        vec![unique_field_ty.into()],
        vec![array, index],
        vec![],
        0,
    );
    assert!(
        MirArrayElementAddrOp::new(array_launder)
            .verify(&ctx)
            .is_err()
    );
    let array_erase = Operation::new(
        &mut ctx,
        MirArrayElementAddrOp::get_concrete_op_info(),
        vec![erased_field_ty.into()],
        vec![array, index],
        vec![],
        0,
    );
    assert!(
        MirArrayElementAddrOp::new(array_erase).verify(&ctx).is_ok(),
        "projection may erase kind while preserving machine mutability"
    );
    let array_mutability_flip = Operation::new(
        &mut ctx,
        MirArrayElementAddrOp::get_concrete_op_info(),
        vec![erased_read_field_ty.into()],
        vec![array, index],
        vec![],
        0,
    );
    assert!(
        MirArrayElementAddrOp::new(array_mutability_flip)
            .verify(&ctx)
            .is_err(),
        "array projection cannot change machine mutability"
    );
}

#[test]
fn test_pointer_with_exposed_provenance_only_creates_raw_or_erased_pointer() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let usize_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let fn_target = MirStructType::get_with_full_layout(
        &mut ctx,
        "FnPtrTarget".into(),
        vec![],
        vec![],
        vec![],
        vec![],
        0,
        0,
    );
    let fn_target_ty: pliron::r#type::TypeHandle = fn_target.into();
    let raw_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, i32_ty.into(), true, MirPointerKind::RawMut);
    let opaque_fn_ty = MirPtrType::get_generic(&mut ctx, fn_target_ty, false);
    let arbitrary_erased_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), false);
    let writable_erased_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), true);
    let unique_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, i32_ty.into(), true, MirPointerKind::UniqueRef);
    let block = BasicBlock::new(&mut ctx, None, vec![usize_ty.into()]);
    let address = block.deref(&ctx).get_argument(0);

    let build = |ctx: &mut Context, result_ty| {
        let op = Operation::new(
            ctx,
            MirCastOp::get_concrete_op_info(),
            vec![result_ty],
            vec![address],
            vec![],
            0,
        );
        let cast = MirCastOp::new(op);
        cast.set_attr_cast_kind(ctx, MirCastKindAttr::PointerWithExposedProvenance);
        cast.set_pointer_kind_authority(ctx, MirPointerKindAuthorityAttr::RustCast);
        op
    };

    assert!(
        MirCastOp::new(build(&mut ctx, raw_ty.into()))
            .verify(&ctx)
            .is_ok()
    );
    assert!(
        MirCastOp::new(build(&mut ctx, unique_ty.into()))
            .verify(&ctx)
            .is_err(),
        "integer provenance cannot directly materialize a Rust reference"
    );

    let build_unmarked = |ctx: &mut Context, result_ty| {
        let op = Operation::new(
            ctx,
            MirCastOp::get_concrete_op_info(),
            vec![result_ty],
            vec![address],
            vec![],
            0,
        );
        MirCastOp::new(op).set_attr_cast_kind(ctx, MirCastKindAttr::PointerWithExposedProvenance);
        op
    };
    assert!(
        MirCastOp::new(build_unmarked(&mut ctx, opaque_fn_ty.into()))
            .verify(&ctx)
            .is_ok(),
        "function-pointer tokens may materialize only the canonical immutable Erased carrier"
    );
    assert!(
        MirCastOp::new(build_unmarked(&mut ctx, arbitrary_erased_ty.into()))
            .verify(&ctx)
            .is_err(),
        "an integer cannot manufacture arbitrary Erased pointer evidence"
    );
    assert!(
        MirCastOp::new(build_unmarked(&mut ctx, writable_erased_ty.into()))
            .verify(&ctx)
            .is_err(),
        "an integer cannot manufacture writable Erased evidence and then reborrow it as UniqueRef"
    );

    let fn_block = BasicBlock::new(&mut ctx, None, vec![opaque_fn_ty.into()]);
    let opaque_fn = fn_block.deref(&ctx).get_argument(0);
    let disguised_data_pointer = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![arbitrary_erased_ty.into()],
        vec![opaque_fn],
        vec![],
        0,
    );
    MirCastOp::new(disguised_data_pointer)
        .set_attr_cast_kind(&ctx, MirCastKindAttr::PointerCoercionArrayToPointer);
    assert!(
        MirCastOp::new(disguised_data_pointer).verify(&ctx).is_err(),
        "an unrelated coercion cannot turn the opaque function token into Erased data storage"
    );

    let shared_fn_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, fn_target_ty, false, MirPointerKind::SharedRef);
    let fake_reborrow = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![shared_fn_ty.into()],
        vec![opaque_fn],
        vec![],
        0,
    );
    let fake_reborrow = MirCastOp::new(fake_reborrow);
    fake_reborrow.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    fake_reborrow.set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::Reborrow);
    assert!(
        fake_reborrow.verify(&ctx).is_err(),
        "an opaque function-pointer value is not compiler storage that may be reborrowed"
    );

    // Keeping only kind+mutability while recursively pairing aggregate fields
    // is insufficient: the canonical function token is also identified by
    // its pointee. Otherwise a tuple cast can disguise data storage as a
    // function pointer (or vice versa), after which individually legal casts
    // can manufacture a reference.
    let data_tuple_ty = MirTupleType::get(&mut ctx, vec![arbitrary_erased_ty.into()]);
    let fn_tuple_ty = MirTupleType::get(&mut ctx, vec![opaque_fn_ty.into()]);
    let nested_block = BasicBlock::new(
        &mut ctx,
        None,
        vec![data_tuple_ty.into(), fn_tuple_ty.into()],
    );
    let data_tuple = nested_block.deref(&ctx).get_argument(0);
    let fn_tuple = nested_block.deref(&ctx).get_argument(1);
    let nested_cast = |ctx: &mut Context, source, target| {
        let op = Operation::new(
            ctx,
            MirCastOp::get_concrete_op_info(),
            vec![target],
            vec![source],
            vec![],
            0,
        );
        MirCastOp::new(op).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
        op
    };
    let data_to_fn = nested_cast(&mut ctx, data_tuple, fn_tuple_ty.into());
    assert!(
        MirCastOp::new(data_to_fn).verify(&ctx).is_err(),
        "an aggregate cast cannot manufacture a nested canonical function token"
    );
    assert!(
        MirCastOp::new(nested_cast(&mut ctx, fn_tuple, data_tuple_ty.into(),))
            .verify(&ctx)
            .is_err(),
        "an aggregate cast cannot disguise a nested function token as Erased data"
    );

    let forged_fn_tuple = data_to_fn.deref(&ctx).get_result(0);
    let extract_forged_fn = Operation::new(
        &mut ctx,
        MirExtractFieldOp::get_concrete_op_info(),
        vec![opaque_fn_ty.into()],
        vec![forged_fn_tuple],
        vec![],
        0,
    );
    let extract_forged_fn = MirExtractFieldOp::new(extract_forged_fn);
    extract_forged_fn.set_attr_index(&ctx, FieldIndexAttr(0));
    assert!(extract_forged_fn.verify(&ctx).is_ok());

    let forged_fn = extract_forged_fn.get_operation().deref(&ctx).get_result(0);
    let expose_forged_fn = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![raw_ty.into()],
        vec![forged_fn],
        vec![],
        0,
    );
    let expose_forged_fn = MirCastOp::new(expose_forged_fn);
    expose_forged_fn.set_attr_cast_kind(&ctx, MirCastKindAttr::FnPtrToPtr);
    expose_forged_fn.set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::RustCast);
    assert!(expose_forged_fn.verify(&ctx).is_ok());

    let forged_raw = expose_forged_fn.get_operation().deref(&ctx).get_result(0);
    let reborrow_forged_fn = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![unique_ty.into()],
        vec![forged_raw],
        vec![],
        0,
    );
    let reborrow_forged_fn = MirCastOp::new(reborrow_forged_fn);
    reborrow_forged_fn.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    reborrow_forged_fn.set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::Reborrow);
    assert!(
        reborrow_forged_fn.verify(&ctx).is_ok(),
        "the laundering chain must be rejected at its aggregate reinterpretation"
    );

    let shared_marker_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, fn_target_ty, false, MirPointerKind::SharedRef);
    let marker_tuple_ty = MirTupleType::get(&mut ctx, vec![fn_target_ty]);
    let shared_marker_tuple_ty = MirPtrType::get_generic_with_kind(
        &mut ctx,
        marker_tuple_ty.into(),
        false,
        MirPointerKind::SharedRef,
    );
    let marker_array_ty = MirArrayType::get(&mut ctx, fn_target_ty, 1);
    let shared_marker_array_ty = MirPtrType::get_generic_with_kind(
        &mut ctx,
        marker_array_ty.into(),
        false,
        MirPointerKind::SharedRef,
    );
    let shared_marker_slice_ty =
        MirSliceType::get_with_kind(&mut ctx, fn_target_ty, MirPointerKind::SharedRef);
    let projection_block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            shared_marker_ty.into(),
            shared_marker_tuple_ty.into(),
            shared_marker_array_ty.into(),
            shared_marker_slice_ty.into(),
            usize_ty.into(),
        ],
    );
    let marker_address = projection_block.deref(&ctx).get_argument(0);
    let marker_tuple_address = projection_block.deref(&ctx).get_argument(1);
    let marker_array_address = projection_block.deref(&ctx).get_argument(2);
    let marker_slice = projection_block.deref(&ctx).get_argument(3);
    let zero = projection_block.deref(&ctx).get_argument(4);

    let offset_to_token = Operation::new(
        &mut ctx,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![opaque_fn_ty.into()],
        vec![marker_address, zero],
        vec![],
        0,
    );
    assert!(
        MirPtrOffsetOp::new(offset_to_token).verify(&ctx).is_err(),
        "pointer arithmetic produces a data address, never a function token"
    );

    let field_address_to_token = Operation::new(
        &mut ctx,
        MirFieldAddrOp::get_concrete_op_info(),
        vec![opaque_fn_ty.into()],
        vec![marker_tuple_address],
        vec![],
        0,
    );
    let field_address_to_token = MirFieldAddrOp::new(field_address_to_token);
    field_address_to_token.set_attr_field_index(&ctx, FieldIndexAttr(0));
    assert!(
        field_address_to_token.verify(&ctx).is_err(),
        "a field address cannot masquerade as a function token"
    );

    let array_address_to_token = Operation::new(
        &mut ctx,
        MirArrayElementAddrOp::get_concrete_op_info(),
        vec![opaque_fn_ty.into()],
        vec![marker_array_address, zero],
        vec![],
        0,
    );
    assert!(
        MirArrayElementAddrOp::new(array_address_to_token)
            .verify(&ctx)
            .is_err(),
        "an array element address cannot masquerade as a function token"
    );

    let slice_data_to_token = Operation::new(
        &mut ctx,
        MirExtractFieldOp::get_concrete_op_info(),
        vec![opaque_fn_ty.into()],
        vec![marker_slice],
        vec![],
        0,
    );
    let slice_data_to_token = MirExtractFieldOp::new(slice_data_to_token);
    slice_data_to_token.set_attr_index(&ctx, FieldIndexAttr(0));
    assert!(
        slice_data_to_token.verify(&ctx).is_err(),
        "a slice data address cannot masquerade as a function token"
    );

    // A ClosureFnPointer cast must not extract captured reference bits as a
    // function pointer, expose them as RawMut, and then reborrow them as
    // UniqueRef. Only a genuinely non-capturing, zero-sized closure may enter
    // the opaque function-pointer path.
    let captured_ref = MirPtrType::get_generic_with_kind(
        &mut ctx,
        i32_ty.into(),
        false,
        MirPointerKind::SharedRef,
    );
    let captured_closure = MirStructType::get_with_full_layout(
        &mut ctx,
        "CapturedClosure".into(),
        vec!["capture_0".into()],
        vec![captured_ref.into()],
        vec![0],
        vec![0],
        8,
        8,
    );
    let empty_closure = MirStructType::get_with_full_layout(
        &mut ctx,
        "NonCapturingClosure".into(),
        vec![],
        vec![],
        vec![],
        vec![],
        0,
        1,
    );
    let closure_block = BasicBlock::new(
        &mut ctx,
        None,
        vec![captured_closure.into(), empty_closure.into()],
    );
    let captured_value = closure_block.deref(&ctx).get_argument(0);
    let empty_value = closure_block.deref(&ctx).get_argument(1);
    let closure_cast = |ctx: &mut Context, source| {
        let op = Operation::new(
            ctx,
            MirCastOp::get_concrete_op_info(),
            vec![opaque_fn_ty.into()],
            vec![source],
            vec![],
            0,
        );
        MirCastOp::new(op)
            .set_attr_cast_kind(ctx, MirCastKindAttr::PointerCoercionClosureFnPointer);
        op
    };
    let captured_to_fn = closure_cast(&mut ctx, captured_value);
    assert!(
        MirCastOp::new(captured_to_fn).verify(&ctx).is_err(),
        "a closure carrying SharedRef bytes cannot become an opaque function pointer"
    );
    assert!(
        MirCastOp::new(closure_cast(&mut ctx, empty_value))
            .verify(&ctx)
            .is_err(),
        "the importer materializes a closure function token directly; the legacy cast is not lowerable"
    );

    let captured_fn_value = captured_to_fn.deref(&ctx).get_result(0);
    let exposed_capture = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![raw_ty.into()],
        vec![captured_fn_value],
        vec![],
        0,
    );
    let exposed_capture = MirCastOp::new(exposed_capture);
    exposed_capture.set_attr_cast_kind(&ctx, MirCastKindAttr::FnPtrToPtr);
    exposed_capture.set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::RustCast);
    assert!(exposed_capture.verify(&ctx).is_ok());

    let exposed_capture_value = exposed_capture.get_operation().deref(&ctx).get_result(0);
    let unique_capture = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![unique_ty.into()],
        vec![exposed_capture_value],
        vec![],
        0,
    );
    let unique_capture = MirCastOp::new(unique_capture);
    unique_capture.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    unique_capture.set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::Reborrow);
    assert!(
        unique_capture.verify(&ctx).is_ok(),
        "the chain must be stopped at ClosureFnPointer before later individually legal boundaries"
    );
}
