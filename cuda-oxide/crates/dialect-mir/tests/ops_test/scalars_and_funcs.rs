/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_mir::{
    attributes::{MirCastKindAttr, ReferenceParamValidityAttr},
    ops::{
        MirAddOp, MirAssignOp, MirCallOp, MirCastOp, MirCheckedAddOp, MirCmpOp, MirConstantOp,
        MirDivOp, MirEqOp, MirFuncOp, MirGeOp, MirGtOp, MirLeOp, MirLtOp, MirMulOp, MirNeOp,
        MirNegOp, MirNotOp, MirRemOp, MirSubOp,
    },
    types::{EnumVariant, MirEnumType, MirPointerKind, MirPtrType, MirSliceType, MirTupleType},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{IntegerAttr, StringAttr, TypeAttr},
        types::{FP32Type, FunctionType, IntegerType, Signedness},
    },
    common_traits::Verify,
    context::Context,
    op::Op,
    operation::Operation,
    utils::apint::APInt,
};
use std::num::NonZeroUsize;

#[test]
fn test_mir_arithmetic_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into(), i32_ty.into()]);
    let lhs = block.deref(&ctx).get_argument(0);

    let check_bin_op = |opid: (
        fn(pliron::context::Ptr<pliron::operation::Operation>) -> pliron::op::OpObj,
        std::any::TypeId,
    ),
                        name: &str| {
        let mut context = Context::new();
        dialect_mir::register(&mut context);
        let ty = IntegerType::get(&context, 32, Signedness::Signed);
        let blk = BasicBlock::new(&mut context, None, vec![ty.into(), ty.into()]);
        let l = blk.deref(&context).get_argument(0);
        let r = blk.deref(&context).get_argument(1);

        let op = Operation::new(&mut context, opid, vec![ty.into()], vec![l, r], vec![], 0);
        assert!(op.verify(&context).is_ok(), "Valid {}", name);

        let f32_t = FP32Type::get(&context);
        let blk2 = BasicBlock::new(&mut context, None, vec![f32_t.into()]);
        let f32_val = blk2.deref(&context).get_argument(0);

        let op_bad = Operation::new(
            &mut context,
            opid,
            vec![ty.into()],
            vec![l, f32_val],
            vec![],
            0,
        );
        assert!(op_bad.verify(&context).is_err(), "Type mismatch {}", name);
    };

    check_bin_op(MirAddOp::get_concrete_op_info(), "Add");
    check_bin_op(MirSubOp::get_concrete_op_info(), "Sub");
    check_bin_op(MirMulOp::get_concrete_op_info(), "Mul");
    check_bin_op(MirDivOp::get_concrete_op_info(), "Div");
    check_bin_op(MirRemOp::get_concrete_op_info(), "Rem");

    let op_neg = Operation::new(
        &mut ctx,
        MirNegOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![lhs],
        vec![],
        0,
    );
    assert!(op_neg.verify(&ctx).is_ok(), "Valid Neg");

    let f32_ty = FP32Type::get(&ctx);
    let op_neg_bad = Operation::new(
        &mut ctx,
        MirNegOp::get_concrete_op_info(),
        vec![f32_ty.into()],
        vec![lhs],
        vec![],
        0,
    );
    assert!(op_neg_bad.verify(&ctx).is_err(), "Neg type mismatch");

    let op_not = Operation::new(
        &mut ctx,
        MirNotOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![lhs],
        vec![],
        0,
    );
    assert!(op_not.verify(&ctx).is_ok(), "Valid Not");
}

#[test]
fn test_mir_misc_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signed);
    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);

    // 1. MirConstantOp
    let i32_signless = IntegerType::get(&ctx, 32, Signedness::Signless);
    let width = NonZeroUsize::new(32).unwrap();
    let apint = APInt::from_u32(42, width);
    let int_attr = IntegerAttr::new(i32_signless, apint);

    let const_op_ptr = Operation::new(
        &mut ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![i32_signless.into()],
        vec![],
        vec![],
        0,
    );
    let const_op = MirConstantOp::new(const_op_ptr);
    const_op.set_attr_value(&ctx, int_attr);
    assert!(const_op.verify(&ctx).is_ok(), "Valid Constant");

    // Mismatch type
    let i64_signless = IntegerType::get(&ctx, 64, Signedness::Signless);
    let i64_width = NonZeroUsize::new(64).unwrap();
    let i64_attr = IntegerAttr::new(i64_signless, APInt::from_u64(42, i64_width));
    const_op.set_attr_value(&ctx, i64_attr);
    assert!(const_op.verify(&ctx).is_err(), "Constant type mismatch");

    // 2. MirCastOp
    let block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let arg = block.deref(&ctx).get_argument(0);

    let cast_op = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![i64_ty.into()],
        vec![arg],
        vec![],
        0,
    );
    MirCastOp::new(cast_op).set_attr_cast_kind(&ctx, MirCastKindAttr::IntToInt);
    assert!(MirCastOp::new(cast_op).verify(&ctx).is_ok(), "Valid Cast");

    // 3. MirCheckedAddOp
    let tuple_ty = MirTupleType::get(&mut ctx, vec![i32_ty.into(), i1_ty.into()]);
    let block2 = BasicBlock::new(&mut ctx, None, vec![i32_ty.into(), i32_ty.into()]);
    let lhs = block2.deref(&ctx).get_argument(0);
    let rhs = block2.deref(&ctx).get_argument(1);

    let checked_add = Operation::new(
        &mut ctx,
        MirCheckedAddOp::get_concrete_op_info(),
        vec![tuple_ty.into()],
        vec![lhs, rhs],
        vec![],
        0,
    );
    assert!(
        MirCheckedAddOp::new(checked_add).verify(&ctx).is_ok(),
        "Valid CheckedAdd"
    );

    // Invalid result type (not tuple)
    let checked_add_bad = Operation::new(
        &mut ctx,
        MirCheckedAddOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![lhs, rhs],
        vec![],
        0,
    );
    assert!(
        MirCheckedAddOp::new(checked_add_bad).verify(&ctx).is_err(),
        "CheckedAdd bad result"
    );
}

#[test]
fn test_mir_comparison_verify() {
    let check_cmp = |opid: (
        fn(pliron::context::Ptr<pliron::operation::Operation>) -> pliron::op::OpObj,
        std::any::TypeId,
    ),
                     name: &str| {
        let mut context = Context::new();
        dialect_mir::register(&mut context);
        let ty = IntegerType::get(&context, 32, Signedness::Signed);
        let res_ty = IntegerType::get(&context, 1, Signedness::Signless);
        let blk = BasicBlock::new(&mut context, None, vec![ty.into(), ty.into()]);
        let l = blk.deref(&context).get_argument(0);
        let r = blk.deref(&context).get_argument(1);

        let op = Operation::new(
            &mut context,
            opid,
            vec![res_ty.into()],
            vec![l, r],
            vec![],
            0,
        );
        assert!(op.verify(&context).is_ok(), "Valid {}", name);

        // Invalid operand types
        let f32_ty = FP32Type::get(&context);
        let blk2 = BasicBlock::new(&mut context, None, vec![f32_ty.into()]);
        let f32_val = blk2.deref(&context).get_argument(0);
        let op_bad = Operation::new(
            &mut context,
            opid,
            vec![res_ty.into()],
            vec![l, f32_val],
            vec![],
            0,
        );
        assert!(op_bad.verify(&context).is_err(), "Type mismatch {}", name);

        // Invalid result type
        let op_bad_res = Operation::new(
            &mut context,
            opid,
            vec![ty.into()], // i32 result instead of i1
            vec![l, r],
            vec![],
            0,
        );
        assert!(
            op_bad_res.verify(&context).is_err(),
            "Result type mismatch {}",
            name
        );
    };

    check_cmp(MirEqOp::get_concrete_op_info(), "Eq");
    check_cmp(MirNeOp::get_concrete_op_info(), "Ne");
    check_cmp(MirLtOp::get_concrete_op_info(), "Lt");
    check_cmp(MirLeOp::get_concrete_op_info(), "Le");
    check_cmp(MirGtOp::get_concrete_op_info(), "Gt");
    check_cmp(MirGeOp::get_concrete_op_info(), "Ge");

    let mut context = Context::new();
    dialect_mir::register(&mut context);
    let i8_ty = IntegerType::get(&context, 8, Signedness::Signed);
    let i32_ty = IntegerType::get(&context, 32, Signedness::Signed);
    let unit = |name: &str| EnumVariant::unit(name.to_string());
    let ordering_ty = MirEnumType::get(
        &mut context,
        "Ordering".to_string(),
        i8_ty.into(),
        vec![255, 0, 1],
        vec![unit("Less"), unit("Equal"), unit("Greater")],
    );
    let blk = BasicBlock::new(&mut context, None, vec![i32_ty.into(), i32_ty.into()]);
    let lhs = blk.deref(&context).get_argument(0);
    let rhs = blk.deref(&context).get_argument(1);
    let two_variant_ty = MirEnumType::get(
        &mut context,
        "Two".to_string(),
        i8_ty.into(),
        vec![0, 1],
        vec![unit("A"), unit("B")],
    );
    // Payload variants disqualify the Ordering shape.
    let payload_ty = MirEnumType::get(
        &mut context,
        "ThreeWithPayload".to_string(),
        i8_ty.into(),
        vec![0, 1, 2],
        vec![
            unit("A"),
            EnumVariant::new("B".to_string(), vec![i32_ty.into()]),
            unit("C"),
        ],
    );
    let mut check_cmp_result = |result_ty, valid| {
        let op = Operation::new(
            &mut context,
            MirCmpOp::get_concrete_op_info(),
            vec![result_ty],
            vec![lhs, rhs],
            vec![],
            0,
        );
        assert_eq!(op.verify(&context).is_ok(), valid);
    };
    check_cmp_result(ordering_ty.into(), true);
    check_cmp_result(i32_ty.into(), false);
    check_cmp_result(two_variant_ty.into(), false);
    check_cmp_result(payload_ty.into(), false);

    // Float operands are rejected: rustc never emits BinOp::Cmp on floats.
    let f32_ty = FP32Type::get(&context);
    let fblk = BasicBlock::new(&mut context, None, vec![f32_ty.into(), f32_ty.into()]);
    let flhs = fblk.deref(&context).get_argument(0);
    let frhs = fblk.deref(&context).get_argument(1);
    let float_cmp = Operation::new(
        &mut context,
        MirCmpOp::get_concrete_op_info(),
        vec![ordering_ty.into()],
        vec![flhs, frhs],
        vec![],
        0,
    );
    assert!(
        float_cmp.verify(&context).is_err(),
        "float mir.cmp must be rejected"
    );
}

#[test]
fn test_mir_func_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let func_ty = FunctionType::get(&ctx, vec![i32_ty.into()], vec![]);
    let func_ty_attr = TypeAttr::new(func_ty.into());

    // Valid Function
    let op_ptr = Operation::new(
        &mut ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let mir_func = MirFuncOp::new(&mut ctx, op_ptr, func_ty_attr.clone());

    // Add entry block with correct argument
    let region = mir_func.get_operation().deref(&ctx).get_region(0);
    let entry_block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    entry_block.insert_at_front(region, &ctx);

    assert!(mir_func.verify(&ctx).is_ok(), "Valid MirFuncOp");

    // Invalid: Argument count mismatch
    let op_ptr2 = Operation::new(
        &mut ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let mir_func2 = MirFuncOp::new(&mut ctx, op_ptr2, func_ty_attr.clone());
    let region2 = mir_func2.get_operation().deref(&ctx).get_region(0);
    // Block with 0 args
    let entry_block2 = BasicBlock::new(&mut ctx, None, vec![]);
    entry_block2.insert_at_front(region2, &ctx);

    assert!(
        mir_func2.verify(&ctx).is_err(),
        "MirFuncOp argument count mismatch"
    );

    // Invalid: Argument type mismatch
    let op_ptr3 = Operation::new(
        &mut ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let mir_func3 = MirFuncOp::new(&mut ctx, op_ptr3, func_ty_attr);
    let region3 = mir_func3.get_operation().deref(&ctx).get_region(0);
    let f32_ty = FP32Type::get(&ctx);
    let entry_block3 = BasicBlock::new(&mut ctx, None, vec![f32_ty.into()]);
    entry_block3.insert_at_front(region3, &ctx);

    assert!(
        mir_func3.verify(&ctx).is_err(),
        "MirFuncOp argument type mismatch"
    );
}

#[test]
fn test_mir_assign_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let val = block.deref(&ctx).get_argument(0);

    let op = Operation::new(
        &mut ctx,
        MirAssignOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![val],
        vec![],
        0,
    );
    assert!(
        MirAssignOp::new(op).verify(&ctx).is_ok(),
        "Valid MirAssignOp"
    );

    let f32_ty = FP32Type::get(&ctx);
    let op_bad = Operation::new(
        &mut ctx,
        MirAssignOp::get_concrete_op_info(),
        vec![f32_ty.into()],
        vec![val],
        vec![],
        0,
    );
    assert!(
        MirAssignOp::new(op_bad).verify(&ctx).is_err(),
        "MirAssignOp type mismatch"
    );
}

#[test]
fn test_mir_call_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let op = Operation::new(
        &mut ctx,
        MirCallOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    let call_op = MirCallOp::new(op);

    // Missing attribute
    assert!(call_op.verify(&ctx).is_err(), "MirCallOp missing attribute");

    // With attribute
    let name = StringAttr::new("my_func".to_string());
    call_op.set_attr_callee(&ctx, name);
    assert!(call_op.verify(&ctx).is_ok(), "Valid MirCallOp");
}

#[test]
fn test_mir_func_reference_param_validity_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let f32_ty = FP32Type::get(&ctx);
    let shared_ref = MirPtrType::get_generic_with_kind(
        &mut ctx,
        f32_ty.into(),
        false,
        MirPointerKind::SharedRef,
    );
    let shared_slice = MirSliceType::get_with_mutability_and_kind(
        &mut ctx,
        f32_ty.into(),
        false,
        MirPointerKind::SharedRef,
    );
    let raw_ptr =
        MirPtrType::get_generic_with_kind(&mut ctx, f32_ty.into(), false, MirPointerKind::RawConst);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);

    let make_func = |ctx: &mut Context, kernel: bool| {
        let func_ty = FunctionType::get(
            ctx,
            vec![
                shared_ref.into(),
                shared_slice.into(),
                raw_ptr.into(),
                i32_ty.into(),
            ],
            vec![],
        );
        let op = Operation::new(
            ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let func = MirFuncOp::new(ctx, op, TypeAttr::new(func_ty.into()));
        if kernel {
            func.get_operation().deref_mut(ctx).attributes.set(
                "gpu_kernel".try_into().unwrap(),
                StringAttr::new("true".to_string()),
            );
        }
        let region = func.get_operation().deref(ctx).get_region(0);
        BasicBlock::new(
            ctx,
            None,
            vec![
                shared_ref.into(),
                shared_slice.into(),
                raw_ptr.into(),
                i32_ty.into(),
            ],
        )
        .insert_at_front(region, ctx);
        func
    };

    let valid = make_func(&mut ctx, true);
    valid.set_reference_param_validity(&mut ctx, 0, ReferenceParamValidityAttr(4));
    valid.set_reference_param_validity(&mut ctx, 1, ReferenceParamValidityAttr(4));
    assert!(
        valid.verify(&ctx).is_ok(),
        "Rust reference facts on kernel source arguments are valid"
    );

    let non_kernel = make_func(&mut ctx, false);
    non_kernel.set_reference_param_validity(&mut ctx, 0, ReferenceParamValidityAttr(4));
    assert!(
        non_kernel.verify(&ctx).is_err(),
        "reference validity is kernel-entry-only"
    );

    let raw = make_func(&mut ctx, true);
    raw.set_reference_param_validity(&mut ctx, 2, ReferenceParamValidityAttr(4));
    assert!(
        raw.verify(&ctx).is_err(),
        "raw pointers cannot carry Rust-reference validity"
    );

    let scalar = make_func(&mut ctx, true);
    scalar.set_reference_param_validity(&mut ctx, 3, ReferenceParamValidityAttr(4));
    assert!(
        scalar.verify(&ctx).is_err(),
        "non-pointer parameters cannot carry Rust-reference validity"
    );

    let bad_alignment = make_func(&mut ctx, true);
    bad_alignment.set_reference_param_validity(&mut ctx, 0, ReferenceParamValidityAttr(3));
    assert!(
        bad_alignment.verify(&ctx).is_err(),
        "alignment must be a non-zero power of two"
    );

    let out_of_range = make_func(&mut ctx, true);
    out_of_range.set_reference_param_validity(&mut ctx, 9, ReferenceParamValidityAttr(4));
    assert!(
        out_of_range.verify(&ctx).is_err(),
        "source argument index must be in range"
    );
}
