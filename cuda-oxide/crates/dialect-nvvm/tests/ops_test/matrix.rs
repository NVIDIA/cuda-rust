/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_mir::types::{MirPtrType, address_space};
use dialect_nvvm::ops::{
    LdmatrixElementAttr, LdmatrixLayoutAttr, LdmatrixMultiplicityAttr, LdmatrixOp,
    LdmatrixShapeAttr, LdmatrixStateSpaceAttr, LdmatrixX1Op, LdmatrixX1TransOp, LdmatrixX2Op,
    LdmatrixX2TransOp, LdmatrixX4Op, LdmatrixX4TransOp, MmaM8N8K4F64Op, MmaM16N8K8F32Tf32Op,
    MmaM16N8K16F32Bf16Op, MmaM16N8K16F32F16Op, MmaM16N8K32S32S8Op, MovmatrixTransB16Op,
    RegisterMmaAccumulatorAttr, RegisterMmaElementAttr, RegisterMmaLayoutAttr, RegisterMmaOp,
    RegisterMmaOperationAttr, RegisterMmaOverflowAttr, RegisterMmaShapeAttr,
    SparseMmaAccumulatorAttr, SparseMmaElementAttr, SparseMmaLayoutAttr, SparseMmaMetadataAttr,
    SparseMmaOp, SparseMmaOverflowAttr, SparseMmaSelectorAttr, SparseMmaShapeAttr,
    StmatrixM8n8X4Op,
};

use pliron::{
    basic_block::BasicBlock,
    builtin::types::{FP32Type, FP64Type, IntegerType, Signedness},
    common_traits::Verify,
    context::Context,
    op::{Op, verify_op},
    operation::Operation,
    r#type::Typed,
};

#[test]
fn test_mma_m8n8k4_f64_requires_four_f64_operands_and_two_f64_results() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let f64_ty = FP64Type::get(&ctx);
    let f32_ty = FP32Type::get(&ctx);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            f64_ty.into(),
            f64_ty.into(),
            f64_ty.into(),
            f64_ty.into(),
            f32_ty.into(),
        ],
    );
    let f64_operands = (0..4)
        .map(|index| block.deref(&ctx).get_argument(index))
        .collect::<Vec<_>>();
    let f32_value = block.deref(&ctx).get_argument(4);

    let valid = Operation::new(
        &mut ctx,
        MmaM8N8K4F64Op::get_concrete_op_info(),
        vec![f64_ty.into(), f64_ty.into()],
        f64_operands.clone(),
        vec![],
        0,
    );
    assert!(verify_op(&MmaM8N8K4F64Op::new(valid), &ctx).is_ok());

    let mut bad_operands = f64_operands.clone();
    bad_operands[2] = f32_value;
    let invalid_operand = Operation::new(
        &mut ctx,
        MmaM8N8K4F64Op::get_concrete_op_info(),
        vec![f64_ty.into(), f64_ty.into()],
        bad_operands,
        vec![],
        0,
    );
    assert!(verify_op(&MmaM8N8K4F64Op::new(invalid_operand), &ctx).is_err());

    let invalid_result = Operation::new(
        &mut ctx,
        MmaM8N8K4F64Op::get_concrete_op_info(),
        vec![f64_ty.into(), f32_ty.into()],
        f64_operands,
        vec![],
        0,
    );
    assert!(verify_op(&MmaM8N8K4F64Op::new(invalid_result), &ctx).is_err());
}

#[test]
fn test_movmatrix_requires_one_i32_operand_and_result() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let f32_ty = FP32Type::get(&ctx);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![i32_ty.into(), i64_ty.into(), f32_ty.into()],
    );
    let i32_value = block.deref(&ctx).get_argument(0);
    let i64_value = block.deref(&ctx).get_argument(1);
    let f32_value = block.deref(&ctx).get_argument(2);

    let valid = Operation::new(
        &mut ctx,
        MovmatrixTransB16Op::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![i32_value],
        vec![],
        0,
    );
    assert!(verify_op(&MovmatrixTransB16Op::new(valid), &ctx).is_ok());

    for (operand, result_type) in [
        (i64_value, i32_ty.into()),
        (f32_value, i32_ty.into()),
        (i32_value, i64_ty.into()),
        (i32_value, f32_ty.into()),
    ] {
        let invalid = Operation::new(
            &mut ctx,
            MovmatrixTransB16Op::get_concrete_op_info(),
            vec![result_type],
            vec![operand],
            vec![],
            0,
        );
        assert!(
            verify_op(&MovmatrixTransB16Op::new(invalid), &ctx).is_err(),
            "movmatrix must reject non-i32 carriers"
        );
    }
}

fn make_ldmatrix_x2(
    ctx: &mut Context,
    pointer: pliron::value::Value,
    result_types: Vec<pliron::r#type::TypeHandle>,
) -> LdmatrixOp {
    let operation = Operation::new(
        ctx,
        LdmatrixOp::get_concrete_op_info(),
        result_types,
        vec![pointer],
        vec![],
        0,
    );
    let ldmatrix = LdmatrixOp::new(operation);
    ldmatrix.set_attr_nvvm_ldmatrix_shape(ctx, LdmatrixShapeAttr::M8n8);
    ldmatrix.set_attr_nvvm_ldmatrix_multiplicity(ctx, LdmatrixMultiplicityAttr::X2);
    ldmatrix.set_attr_nvvm_ldmatrix_layout(ctx, LdmatrixLayoutAttr::Normal);
    ldmatrix.set_attr_nvvm_ldmatrix_element(ctx, LdmatrixElementAttr::B16);
    ldmatrix.set_attr_nvvm_ldmatrix_state_space(ctx, LdmatrixStateSpaceAttr::Shared);
    ldmatrix
}

#[test]
fn test_ldmatrix_accepts_native_u32_or_compatible_shared_pointers() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let u16_ty = IntegerType::get(&ctx, 16, Signedness::Unsigned);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let pointer_types = [
        MirPtrType::get_generic(&mut ctx, u32_ty.into(), false),
        MirPtrType::get_shared(&mut ctx, u32_ty.into(), false),
        MirPtrType::get_global(&mut ctx, u32_ty.into(), false),
        MirPtrType::get_constant(&mut ctx, u32_ty.into(), false),
        MirPtrType::get(&mut ctx, u32_ty.into(), false, address_space::LOCAL),
        MirPtrType::get_generic(&mut ctx, u16_ty.into(), false),
    ];
    let block = BasicBlock::new(
        &mut ctx,
        None,
        pointer_types
            .iter()
            .map(|pointer| (*pointer).into())
            .collect(),
    );

    for index in 0..pointer_types.len() {
        let pointer = block.deref(&ctx).get_argument(index);
        let operation = make_ldmatrix_x2(&mut ctx, pointer, vec![u32_ty.into(), u32_ty.into()]);
        let verified = verify_op(&operation, &ctx);
        if index < 2 {
            assert!(verified.is_ok(), "pointer case {index} should be accepted");
        } else {
            assert!(verified.is_err(), "pointer case {index} should be rejected");
        }
    }

    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let scalar_block = BasicBlock::new(&mut ctx, None, vec![u32_ty.into(), u64_ty.into()]);
    let native_u32 = scalar_block.deref(&ctx).get_argument(0);
    let wide_u64 = scalar_block.deref(&ctx).get_argument(1);
    assert!(
        verify_op(
            &make_ldmatrix_x2(&mut ctx, native_u32, vec![u32_ty.into(), u32_ty.into()]),
            &ctx,
        )
        .is_ok()
    );
    assert!(
        verify_op(
            &make_ldmatrix_x2(&mut ctx, wide_u64, vec![u32_ty.into(), u32_ty.into()]),
            &ctx,
        )
        .is_err()
    );
}

#[test]
fn generated_ldmatrix_verifier_rejects_zero_or_two_operands_without_panicking() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let pointer_ty = MirPtrType::get_shared(&mut ctx, u32_ty.into(), false);
    let block = BasicBlock::new(&mut ctx, None, vec![pointer_ty.into(), pointer_ty.into()]);
    let pointer0 = block.deref(&ctx).get_argument(0);
    let pointer1 = block.deref(&ctx).get_argument(1);

    let zero = Operation::new(
        &mut ctx,
        LdmatrixOp::get_concrete_op_info(),
        vec![u32_ty.into(); 4],
        vec![],
        vec![],
        0,
    );
    assert!(LdmatrixOp::new(zero).verify(&ctx).is_err());

    let two = Operation::new(
        &mut ctx,
        LdmatrixOp::get_concrete_op_info(),
        vec![u32_ty.into(); 4],
        vec![pointer0, pointer1],
        vec![],
        0,
    );
    assert!(LdmatrixOp::new(two).verify(&ctx).is_err());

    let valid = LdmatrixOp::build(
        &mut ctx,
        pointer0,
        LdmatrixShapeAttr::M8n8,
        LdmatrixMultiplicityAttr::X4,
        LdmatrixLayoutAttr::Normal,
        LdmatrixElementAttr::B16,
        LdmatrixStateSpaceAttr::Shared,
    );
    assert!(LdmatrixOp::new(valid).verify(&ctx).is_ok());
}

#[test]
fn blackwell_ldmatrix_verifier_accepts_only_reviewed_shapes() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);
    let u8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let shared_u8 = MirPtrType::get_shared(&mut ctx, u8_ty.into(), false);
    let generic_u8 = MirPtrType::get_generic(&mut ctx, u8_ty.into(), false);
    let global_u8 = MirPtrType::get_global(&mut ctx, u8_ty.into(), false);
    let shared_u32 = MirPtrType::get_shared(&mut ctx, u32_ty.into(), false);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            shared_u8.into(),
            generic_u8.into(),
            global_u8.into(),
            shared_u32.into(),
        ],
    );
    let shared_pointer = block.deref(&ctx).get_argument(0);

    for pointer_index in [0, 1] {
        let pointer = block.deref(&ctx).get_argument(pointer_index);
        for (shape, multiplicity, layout, element, result_count) in [
            (
                LdmatrixShapeAttr::M16n16,
                LdmatrixMultiplicityAttr::X1,
                LdmatrixLayoutAttr::Transposed,
                LdmatrixElementAttr::B8,
                2,
            ),
            (
                LdmatrixShapeAttr::M16n16,
                LdmatrixMultiplicityAttr::X1,
                LdmatrixLayoutAttr::Transposed,
                LdmatrixElementAttr::B8x16B4x16P64,
                2,
            ),
            (
                LdmatrixShapeAttr::M16n16,
                LdmatrixMultiplicityAttr::X1,
                LdmatrixLayoutAttr::Transposed,
                LdmatrixElementAttr::B8x16B6x16P32,
                2,
            ),
            (
                LdmatrixShapeAttr::M16n16,
                LdmatrixMultiplicityAttr::X2,
                LdmatrixLayoutAttr::Transposed,
                LdmatrixElementAttr::B8,
                4,
            ),
            (
                LdmatrixShapeAttr::M16n16,
                LdmatrixMultiplicityAttr::X2,
                LdmatrixLayoutAttr::Transposed,
                LdmatrixElementAttr::B8x16B4x16P64,
                4,
            ),
            (
                LdmatrixShapeAttr::M16n16,
                LdmatrixMultiplicityAttr::X2,
                LdmatrixLayoutAttr::Transposed,
                LdmatrixElementAttr::B8x16B6x16P32,
                4,
            ),
            (
                LdmatrixShapeAttr::M8n16,
                LdmatrixMultiplicityAttr::X1,
                LdmatrixLayoutAttr::Normal,
                LdmatrixElementAttr::B8x16B4x16P64,
                1,
            ),
            (
                LdmatrixShapeAttr::M8n16,
                LdmatrixMultiplicityAttr::X1,
                LdmatrixLayoutAttr::Normal,
                LdmatrixElementAttr::B8x16B6x16P32,
                1,
            ),
            (
                LdmatrixShapeAttr::M8n16,
                LdmatrixMultiplicityAttr::X2,
                LdmatrixLayoutAttr::Normal,
                LdmatrixElementAttr::B8x16B4x16P64,
                2,
            ),
            (
                LdmatrixShapeAttr::M8n16,
                LdmatrixMultiplicityAttr::X2,
                LdmatrixLayoutAttr::Normal,
                LdmatrixElementAttr::B8x16B6x16P32,
                2,
            ),
            (
                LdmatrixShapeAttr::M8n16,
                LdmatrixMultiplicityAttr::X4,
                LdmatrixLayoutAttr::Normal,
                LdmatrixElementAttr::B8x16B4x16P64,
                4,
            ),
            (
                LdmatrixShapeAttr::M8n16,
                LdmatrixMultiplicityAttr::X4,
                LdmatrixLayoutAttr::Normal,
                LdmatrixElementAttr::B8x16B6x16P32,
                4,
            ),
        ] {
            let op = LdmatrixOp::build(
                &mut ctx,
                pointer,
                shape,
                multiplicity,
                layout,
                element,
                LdmatrixStateSpaceAttr::Shared,
            );
            assert_eq!(op.deref(&ctx).get_num_results(), result_count);
            assert!(LdmatrixOp::new(op).verify(&ctx).is_ok());
        }
    }

    for (shape, multiplicity, layout, element) in [
        (
            LdmatrixShapeAttr::M16n16,
            LdmatrixMultiplicityAttr::X4,
            LdmatrixLayoutAttr::Transposed,
            LdmatrixElementAttr::B8,
        ),
        (
            LdmatrixShapeAttr::M16n16,
            LdmatrixMultiplicityAttr::X1,
            LdmatrixLayoutAttr::Normal,
            LdmatrixElementAttr::B8,
        ),
        (
            LdmatrixShapeAttr::M8n16,
            LdmatrixMultiplicityAttr::X1,
            LdmatrixLayoutAttr::Transposed,
            LdmatrixElementAttr::B8x16B4x16P64,
        ),
        (
            LdmatrixShapeAttr::M8n16,
            LdmatrixMultiplicityAttr::X1,
            LdmatrixLayoutAttr::Normal,
            LdmatrixElementAttr::B8,
        ),
    ] {
        let op = LdmatrixOp::build(
            &mut ctx,
            shared_pointer,
            shape,
            multiplicity,
            layout,
            element,
            LdmatrixStateSpaceAttr::Shared,
        );
        assert!(LdmatrixOp::new(op).verify(&ctx).is_err());
    }

    for pointer_index in [2, 3] {
        let pointer = block.deref(&ctx).get_argument(pointer_index);
        let op = LdmatrixOp::build(
            &mut ctx,
            pointer,
            LdmatrixShapeAttr::M16n16,
            LdmatrixMultiplicityAttr::X1,
            LdmatrixLayoutAttr::Transposed,
            LdmatrixElementAttr::B8,
            LdmatrixStateSpaceAttr::Shared,
        );
        assert!(LdmatrixOp::new(op).verify(&ctx).is_err());
    }
}

#[test]
fn classic_ldmatrix_compatibility_ops_keep_names_and_register_shapes() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let pointer_ty = MirPtrType::get_shared(&mut ctx, u32_ty.into(), false);
    let block = BasicBlock::new(&mut ctx, None, vec![pointer_ty.into()]);
    let pointer = block.deref(&ctx).get_argument(0);

    macro_rules! check_compat {
        ($op:ty, $name:literal, $results:literal) => {{
            assert_eq!(<$op>::get_opid_static().to_string(), $name);
            let valid = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![u32_ty.into(); $results],
                vec![pointer],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(valid), &ctx).is_ok());

            let wrong_shape = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![u32_ty.into(); $results + 1],
                vec![pointer],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(wrong_shape), &ctx).is_err());
        }};
    }

    check_compat!(LdmatrixX1Op, "nvvm.ldmatrix_x1", 1);
    check_compat!(LdmatrixX1TransOp, "nvvm.ldmatrix_x1_trans", 1);
    check_compat!(LdmatrixX2Op, "nvvm.ldmatrix_x2", 2);
    check_compat!(LdmatrixX2TransOp, "nvvm.ldmatrix_x2_trans", 2);
    check_compat!(LdmatrixX4Op, "nvvm.ldmatrix_x4", 4);
    check_compat!(LdmatrixX4TransOp, "nvvm.ldmatrix_x4_trans", 4);
}

#[test]
fn test_mma_m16n8k16_bf16_verifies_exact_register_signature() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let f32_ty = FP32Type::get(&ctx);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![f32_ty.into(), i32_ty.into(), i64_ty.into()],
    );
    let f32_value = block.deref(&ctx).get_argument(0);
    let i32_value = block.deref(&ctx).get_argument(1);
    let i64_value = block.deref(&ctx).get_argument(2);

    let valid_operands = (0..4)
        .map(|_| f32_value)
        .chain((0..6).map(|_| i32_value))
        .collect();
    let valid = Operation::new(
        &mut ctx,
        MmaM16N8K16F32Bf16Op::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        valid_operands,
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K16F32Bf16Op::new(valid), &ctx).is_ok());

    let bad_c_operands = (0..4)
        .map(|index| if index == 0 { i32_value } else { f32_value })
        .chain((0..6).map(|_| i32_value))
        .collect();
    let bad_c = Operation::new(
        &mut ctx,
        MmaM16N8K16F32Bf16Op::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        bad_c_operands,
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K16F32Bf16Op::new(bad_c), &ctx).is_err());

    let bad_a_operands = (0..4)
        .map(|_| f32_value)
        .chain((0..6).map(|index| if index == 0 { i64_value } else { i32_value }))
        .collect();
    let bad_a = Operation::new(
        &mut ctx,
        MmaM16N8K16F32Bf16Op::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        bad_a_operands,
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K16F32Bf16Op::new(bad_a), &ctx).is_err());

    let bad_result = Operation::new(
        &mut ctx,
        MmaM16N8K16F32Bf16Op::get_concrete_op_info(),
        vec![f32_ty.into(), f32_ty.into(), f32_ty.into(), i32_ty.into()],
        (0..4)
            .map(|_| f32_value)
            .chain((0..6).map(|_| i32_value))
            .collect(),
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K16F32Bf16Op::new(bad_result), &ctx).is_err());

    let bad_arity = Operation::new(
        &mut ctx,
        MmaM16N8K16F32Bf16Op::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        (0..4)
            .map(|_| f32_value)
            .chain((0..5).map(|_| i32_value))
            .collect(),
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K16F32Bf16Op::new(bad_arity), &ctx).is_err());
}

#[test]
fn test_mma_m16n8k16_f16_verifies_exact_register_signature() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let f32_ty = FP32Type::get(&ctx);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![f32_ty.into(), i32_ty.into(), i64_ty.into()],
    );
    let f32_value = block.deref(&ctx).get_argument(0);
    let i32_value = block.deref(&ctx).get_argument(1);
    let i64_value = block.deref(&ctx).get_argument(2);

    let operands = || {
        (0..4)
            .map(|_| f32_value)
            .chain((0..6).map(|_| i32_value))
            .collect()
    };
    let valid = Operation::new(
        &mut ctx,
        MmaM16N8K16F32F16Op::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        operands(),
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K16F32F16Op::new(valid), &ctx).is_ok());

    let bad_c_operands = (0..4)
        .map(|index| if index == 0 { i32_value } else { f32_value })
        .chain((0..6).map(|_| i32_value))
        .collect();
    let bad_c = Operation::new(
        &mut ctx,
        MmaM16N8K16F32F16Op::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        bad_c_operands,
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K16F32F16Op::new(bad_c), &ctx).is_err());

    let bad_packed_operands = (0..4)
        .map(|_| f32_value)
        .chain((0..6).map(|index| if index == 0 { i64_value } else { i32_value }))
        .collect();
    let bad_packed = Operation::new(
        &mut ctx,
        MmaM16N8K16F32F16Op::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        bad_packed_operands,
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K16F32F16Op::new(bad_packed), &ctx).is_err());

    let bad_result_type = Operation::new(
        &mut ctx,
        MmaM16N8K16F32F16Op::get_concrete_op_info(),
        vec![f32_ty.into(), f32_ty.into(), f32_ty.into(), i32_ty.into()],
        operands(),
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K16F32F16Op::new(bad_result_type), &ctx).is_err());

    let bad_operand_arity = Operation::new(
        &mut ctx,
        MmaM16N8K16F32F16Op::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        (0..4)
            .map(|_| f32_value)
            .chain((0..5).map(|_| i32_value))
            .collect(),
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K16F32F16Op::new(bad_operand_arity), &ctx).is_err());

    let bad_result_arity = Operation::new(
        &mut ctx,
        MmaM16N8K16F32F16Op::get_concrete_op_info(),
        vec![f32_ty.into(); 3],
        operands(),
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K16F32F16Op::new(bad_result_arity), &ctx).is_err());
}

#[test]
fn test_mma_m16n8k8_tf32_verifies_exact_register_signature() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let f32_ty = FP32Type::get(&ctx);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![f32_ty.into(), i32_ty.into(), i64_ty.into()],
    );
    let f32_value = block.deref(&ctx).get_argument(0);
    let i32_value = block.deref(&ctx).get_argument(1);
    let i64_value = block.deref(&ctx).get_argument(2);

    let operands = || {
        (0..4)
            .map(|_| f32_value)
            .chain((0..6).map(|_| i32_value))
            .collect()
    };
    let valid = Operation::new(
        &mut ctx,
        MmaM16N8K8F32Tf32Op::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        operands(),
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K8F32Tf32Op::new(valid), &ctx).is_ok());

    let bad_c_operands = (0..4)
        .map(|index| if index == 0 { i32_value } else { f32_value })
        .chain((0..6).map(|_| i32_value))
        .collect();
    let bad_c = Operation::new(
        &mut ctx,
        MmaM16N8K8F32Tf32Op::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        bad_c_operands,
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K8F32Tf32Op::new(bad_c), &ctx).is_err());

    let bad_packed_operands = (0..4)
        .map(|_| f32_value)
        .chain((0..6).map(|index| if index == 0 { i64_value } else { i32_value }))
        .collect();
    let bad_packed = Operation::new(
        &mut ctx,
        MmaM16N8K8F32Tf32Op::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        bad_packed_operands,
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K8F32Tf32Op::new(bad_packed), &ctx).is_err());

    let bad_result_type = Operation::new(
        &mut ctx,
        MmaM16N8K8F32Tf32Op::get_concrete_op_info(),
        vec![f32_ty.into(), f32_ty.into(), f32_ty.into(), i32_ty.into()],
        operands(),
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K8F32Tf32Op::new(bad_result_type), &ctx).is_err());

    let bad_operand_arity = Operation::new(
        &mut ctx,
        MmaM16N8K8F32Tf32Op::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        (0..4)
            .map(|_| f32_value)
            .chain((0..5).map(|_| i32_value))
            .collect(),
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K8F32Tf32Op::new(bad_operand_arity), &ctx).is_err());

    let bad_result_arity = Operation::new(
        &mut ctx,
        MmaM16N8K8F32Tf32Op::get_concrete_op_info(),
        vec![f32_ty.into(); 3],
        operands(),
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K8F32Tf32Op::new(bad_result_arity), &ctx).is_err());
}

#[test]
fn test_mma_m16n8k32_s8_verifies_exact_register_signature() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let f32_ty = FP32Type::get(&ctx);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![i32_ty.into(), i64_ty.into(), f32_ty.into()],
    );
    let i32_value = block.deref(&ctx).get_argument(0);
    let i64_value = block.deref(&ctx).get_argument(1);
    let f32_value = block.deref(&ctx).get_argument(2);
    let valid_operands = vec![i32_value; 10];

    let valid = Operation::new(
        &mut ctx,
        MmaM16N8K32S32S8Op::get_concrete_op_info(),
        vec![i32_ty.into(); 4],
        valid_operands.clone(),
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K32S32S8Op::new(valid), &ctx).is_ok());

    for bad_value in [i64_value, f32_value] {
        let mut bad_operands = valid_operands.clone();
        bad_operands[4] = bad_value;
        let invalid = Operation::new(
            &mut ctx,
            MmaM16N8K32S32S8Op::get_concrete_op_info(),
            vec![i32_ty.into(); 4],
            bad_operands,
            vec![],
            0,
        );
        assert!(
            verify_op(&MmaM16N8K32S32S8Op::new(invalid), &ctx).is_err(),
            "MMA must reject non-i32 register operands"
        );
    }

    for bad_results in [
        vec![i32_ty.into(), i32_ty.into(), i32_ty.into(), i64_ty.into()],
        vec![i32_ty.into(), i32_ty.into(), i32_ty.into(), f32_ty.into()],
        vec![i32_ty.into(); 3],
    ] {
        let invalid = Operation::new(
            &mut ctx,
            MmaM16N8K32S32S8Op::get_concrete_op_info(),
            bad_results,
            valid_operands.clone(),
            vec![],
            0,
        );
        assert!(
            verify_op(&MmaM16N8K32S32S8Op::new(invalid), &ctx).is_err(),
            "MMA must reject the wrong result register signature"
        );
    }

    let invalid_arity = Operation::new(
        &mut ctx,
        MmaM16N8K32S32S8Op::get_concrete_op_info(),
        vec![i32_ty.into(); 4],
        vec![i32_value; 9],
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K32S32S8Op::new(invalid_arity), &ctx).is_err());
}

#[test]
fn generated_register_mma_verifier_rejects_crossed_variants_and_carriers() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    macro_rules! set_variant {
        ($op:expr, $shape:expr, $acc:expr, $a:expr, $b:expr, $overflow:expr) => {{
            $op.set_attr_nvvm_register_mma_shape(&ctx, $shape);
            $op.set_attr_nvvm_register_mma_accumulator(&ctx, $acc);
            $op.set_attr_nvvm_register_mma_a_element(&ctx, $a);
            $op.set_attr_nvvm_register_mma_b_element(&ctx, $b);
            $op.set_attr_nvvm_register_mma_a_layout(&ctx, RegisterMmaLayoutAttr::Row);
            $op.set_attr_nvvm_register_mma_b_layout(&ctx, RegisterMmaLayoutAttr::Col);
            $op.set_attr_nvvm_register_mma_overflow(&ctx, $overflow);
        }};
    }

    let f32_ty = FP32Type::get(&ctx);
    let f64_ty = FP64Type::get(&ctx);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let signless_i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![f32_ty.into(), f64_ty.into(), i32_ty.into(), u32_ty.into()],
    );
    let f32_value = block.deref(&ctx).get_argument(0);
    let f64_value = block.deref(&ctx).get_argument(1);
    let i32_value = block.deref(&ctx).get_argument(2);
    let u32_value = block.deref(&ctx).get_argument(3);

    let bf16_operation = Operation::new(
        &mut ctx,
        RegisterMmaOp::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        [vec![f32_value; 4], vec![u32_value; 6]].concat(),
        vec![],
        0,
    );
    let bf16 = RegisterMmaOp::new(bf16_operation);
    set_variant!(
        bf16,
        RegisterMmaShapeAttr::M16n8k16,
        RegisterMmaAccumulatorAttr::F32,
        RegisterMmaElementAttr::Bf16,
        RegisterMmaElementAttr::Bf16,
        RegisterMmaOverflowAttr::NotApplicable
    );
    assert!(bf16.get_attr_nvvm_register_mma_operation(&ctx).is_none());
    assert!(verify_op(&bf16, &ctx).is_ok());
    bf16.set_attr_nvvm_register_mma_b_element(&ctx, RegisterMmaElementAttr::F16);
    assert!(verify_op(&bf16, &ctx).is_err());

    let f64_operation = Operation::new(
        &mut ctx,
        RegisterMmaOp::get_concrete_op_info(),
        vec![f64_ty.into(); 2],
        vec![f64_value; 4],
        vec![],
        0,
    );
    let f64_mma = RegisterMmaOp::new(f64_operation);
    set_variant!(
        f64_mma,
        RegisterMmaShapeAttr::M8n8k4,
        RegisterMmaAccumulatorAttr::F64,
        RegisterMmaElementAttr::F64,
        RegisterMmaElementAttr::F64,
        RegisterMmaOverflowAttr::NotApplicable
    );
    assert!(verify_op(&f64_mma, &ctx).is_ok());

    let int_operands = [vec![i32_value; 4], vec![u32_value; 6]].concat();
    let int_operation = Operation::new(
        &mut ctx,
        RegisterMmaOp::get_concrete_op_info(),
        vec![i32_ty.into(); 4],
        int_operands.clone(),
        vec![],
        0,
    );
    let int_mma = RegisterMmaOp::new(int_operation);
    set_variant!(
        int_mma,
        RegisterMmaShapeAttr::M16n8k32,
        RegisterMmaAccumulatorAttr::S32,
        RegisterMmaElementAttr::S8,
        RegisterMmaElementAttr::S8,
        RegisterMmaOverflowAttr::Wrapping
    );
    assert!(verify_op(&int_mma, &ctx).is_ok());

    let wrong_signedness = Operation::new(
        &mut ctx,
        RegisterMmaOp::get_concrete_op_info(),
        vec![signless_i32_ty.into(); 4],
        int_operands,
        vec![],
        0,
    );
    let wrong_signedness = RegisterMmaOp::new(wrong_signedness);
    set_variant!(
        wrong_signedness,
        RegisterMmaShapeAttr::M16n8k32,
        RegisterMmaAccumulatorAttr::S32,
        RegisterMmaElementAttr::S8,
        RegisterMmaElementAttr::S8,
        RegisterMmaOverflowAttr::Wrapping
    );
    assert!(verify_op(&wrong_signedness, &ctx).is_err());

    let missing_attributes = Operation::new(
        &mut ctx,
        RegisterMmaOp::get_concrete_op_info(),
        vec![f64_ty.into(); 2],
        vec![f64_value; 4],
        vec![],
        0,
    );
    assert!(verify_op(&RegisterMmaOp::new(missing_attributes), &ctx).is_err());
}

#[test]
fn generated_register_mma_verifies_dense_integer_families() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let signless_i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![i32_ty.into(), u32_ty.into(), signless_i32_ty.into()],
    );
    let i32_value = block.deref(&ctx).get_argument(0);
    let u32_value = block.deref(&ctx).get_argument(1);

    macro_rules! int_mma {
        ($shape:expr, $a:expr, $b:expr, $overflow:expr, $operands:expr, $results:expr) => {{
            let operation = Operation::new(
                &mut ctx,
                RegisterMmaOp::get_concrete_op_info(),
                $results,
                $operands,
                vec![],
                0,
            );
            let mma = RegisterMmaOp::new(operation);
            mma.set_attr_nvvm_register_mma_shape(&ctx, $shape);
            mma.set_attr_nvvm_register_mma_operation(&ctx, RegisterMmaOperationAttr::Multiply);
            mma.set_attr_nvvm_register_mma_accumulator(&ctx, RegisterMmaAccumulatorAttr::S32);
            mma.set_attr_nvvm_register_mma_a_element(&ctx, $a);
            mma.set_attr_nvvm_register_mma_b_element(&ctx, $b);
            mma.set_attr_nvvm_register_mma_a_layout(&ctx, RegisterMmaLayoutAttr::Row);
            mma.set_attr_nvvm_register_mma_b_layout(&ctx, RegisterMmaLayoutAttr::Col);
            mma.set_attr_nvvm_register_mma_overflow(&ctx, $overflow);
            mma
        }};
    }

    let mut accepted = 0;
    for (shape, accumulator_count, operand_count, result_count) in [
        (RegisterMmaShapeAttr::M8n8k16, 2, 4, 2),
        (RegisterMmaShapeAttr::M16n8k16, 4, 7, 4),
        (RegisterMmaShapeAttr::M16n8k32, 4, 10, 4),
    ] {
        for (a_element, b_element) in [
            (RegisterMmaElementAttr::S8, RegisterMmaElementAttr::S8),
            (RegisterMmaElementAttr::S8, RegisterMmaElementAttr::U8),
            (RegisterMmaElementAttr::U8, RegisterMmaElementAttr::S8),
            (RegisterMmaElementAttr::U8, RegisterMmaElementAttr::U8),
        ] {
            for overflow in [
                RegisterMmaOverflowAttr::Wrapping,
                RegisterMmaOverflowAttr::Satfinite,
            ] {
                let operands = [
                    vec![i32_value; accumulator_count],
                    vec![u32_value; operand_count - accumulator_count],
                ]
                .concat();
                let mma = int_mma!(
                    shape.clone(),
                    a_element.clone(),
                    b_element.clone(),
                    overflow.clone(),
                    operands,
                    vec![i32_ty.into(); result_count]
                );
                assert_eq!(
                    mma.get_operation().deref(&ctx).get_num_operands(),
                    operand_count
                );
                assert_eq!(
                    mma.get_operation().deref(&ctx).get_num_results(),
                    result_count
                );
                assert!(
                    verify_op(&mma, &ctx).is_ok(),
                    "rejected {shape:?} {a_element:?}x{b_element:?} {overflow:?}"
                );
                accepted += 1;
            }
        }
    }
    assert_eq!(accepted, 24);

    let mut int4_accepted = 0;
    for (shape, accumulator_count, a_count, b_count, result_count) in [
        (RegisterMmaShapeAttr::M8n8k32, 2, 1, 1, 2),
        (RegisterMmaShapeAttr::M16n8k32, 4, 2, 1, 4),
        (RegisterMmaShapeAttr::M16n8k64, 4, 4, 2, 4),
    ] {
        for (a_element, b_element) in [
            (RegisterMmaElementAttr::S4, RegisterMmaElementAttr::S4),
            (RegisterMmaElementAttr::S4, RegisterMmaElementAttr::U4),
            (RegisterMmaElementAttr::U4, RegisterMmaElementAttr::S4),
            (RegisterMmaElementAttr::U4, RegisterMmaElementAttr::U4),
        ] {
            for overflow in [
                RegisterMmaOverflowAttr::Wrapping,
                RegisterMmaOverflowAttr::Satfinite,
            ] {
                let operands = [
                    vec![i32_value; accumulator_count],
                    vec![u32_value; a_count],
                    vec![u32_value; b_count],
                ]
                .concat();
                let expected_operand_types = [
                    vec![i32_ty.into(); accumulator_count],
                    vec![u32_ty.into(); a_count],
                    vec![u32_ty.into(); b_count],
                ]
                .concat();
                let mma = int_mma!(
                    shape.clone(),
                    a_element.clone(),
                    b_element.clone(),
                    overflow.clone(),
                    operands,
                    vec![i32_ty.into(); result_count]
                );
                let operation = mma.get_operation().deref(&ctx);
                assert_eq!(
                    operation.get_num_operands(),
                    accumulator_count + a_count + b_count
                );
                assert_eq!(operation.get_num_results(), result_count);
                assert_eq!(
                    operation
                        .operands()
                        .map(|operand| operand.get_type(&ctx))
                        .collect::<Vec<_>>(),
                    expected_operand_types
                );
                assert_eq!(
                    (0..operation.get_num_results())
                        .map(|index| operation.get_result(index).get_type(&ctx))
                        .collect::<Vec<_>>(),
                    vec![i32_ty.into(); result_count]
                );
                assert_eq!(
                    mma.get_attr_nvvm_register_mma_shape(&ctx).as_deref(),
                    Some(&shape)
                );
                assert_eq!(
                    mma.get_attr_nvvm_register_mma_accumulator(&ctx).as_deref(),
                    Some(&RegisterMmaAccumulatorAttr::S32)
                );
                assert_eq!(
                    mma.get_attr_nvvm_register_mma_a_element(&ctx).as_deref(),
                    Some(&a_element)
                );
                assert_eq!(
                    mma.get_attr_nvvm_register_mma_b_element(&ctx).as_deref(),
                    Some(&b_element)
                );
                assert_eq!(
                    mma.get_attr_nvvm_register_mma_a_layout(&ctx).as_deref(),
                    Some(&RegisterMmaLayoutAttr::Row)
                );
                assert_eq!(
                    mma.get_attr_nvvm_register_mma_b_layout(&ctx).as_deref(),
                    Some(&RegisterMmaLayoutAttr::Col)
                );
                assert_eq!(
                    mma.get_attr_nvvm_register_mma_overflow(&ctx).as_deref(),
                    Some(&overflow)
                );
                assert!(
                    verify_op(&mma, &ctx).is_ok(),
                    "rejected {shape:?} {a_element:?}x{b_element:?} {overflow:?}"
                );
                int4_accepted += 1;
            }
        }
    }
    assert_eq!(int4_accepted, 24);

    for (shape, accumulator_count, operand_count, result_count) in [
        (RegisterMmaShapeAttr::M8n8k32, 2, 4, 2),
        (RegisterMmaShapeAttr::M16n8k32, 4, 7, 4),
        (RegisterMmaShapeAttr::M16n8k64, 4, 10, 4),
    ] {
        for wrong_operand_count in [operand_count - 1, operand_count + 1] {
            let mma = int_mma!(
                shape.clone(),
                RegisterMmaElementAttr::S4,
                RegisterMmaElementAttr::U4,
                RegisterMmaOverflowAttr::Wrapping,
                [
                    vec![i32_value; accumulator_count],
                    vec![u32_value; wrong_operand_count - accumulator_count],
                ]
                .concat(),
                vec![i32_ty.into(); result_count]
            );
            assert!(verify_op(&mma, &ctx).is_err());
        }

        for wrong_result_count in [result_count - 1, result_count + 1] {
            let mma = int_mma!(
                shape.clone(),
                RegisterMmaElementAttr::U4,
                RegisterMmaElementAttr::S4,
                RegisterMmaOverflowAttr::Satfinite,
                [
                    vec![i32_value; accumulator_count],
                    vec![u32_value; operand_count - accumulator_count],
                ]
                .concat(),
                vec![i32_ty.into(); wrong_result_count]
            );
            assert!(verify_op(&mma, &ctx).is_err());
        }
    }

    let int4_on_int8_shape = int_mma!(
        RegisterMmaShapeAttr::M8n8k16,
        RegisterMmaElementAttr::S4,
        RegisterMmaElementAttr::U4,
        RegisterMmaOverflowAttr::Wrapping,
        [vec![i32_value; 2], vec![u32_value; 2]].concat(),
        vec![i32_ty.into(); 2]
    );
    assert!(verify_op(&int4_on_int8_shape, &ctx).is_err());

    let int8_on_int4_shape = int_mma!(
        RegisterMmaShapeAttr::M8n8k32,
        RegisterMmaElementAttr::S8,
        RegisterMmaElementAttr::U8,
        RegisterMmaOverflowAttr::Wrapping,
        [vec![i32_value; 2], vec![u32_value; 2]].concat(),
        vec![i32_ty.into(); 2]
    );
    assert!(verify_op(&int8_on_int4_shape, &ctx).is_err());

    let crossed_integer_width = int_mma!(
        RegisterMmaShapeAttr::M16n8k32,
        RegisterMmaElementAttr::S4,
        RegisterMmaElementAttr::U8,
        RegisterMmaOverflowAttr::Satfinite,
        [vec![i32_value; 4], vec![u32_value; 3]].concat(),
        vec![i32_ty.into(); 4]
    );
    assert!(verify_op(&crossed_integer_width, &ctx).is_err());

    let m16k32_int4_with_int8_carriers = int_mma!(
        RegisterMmaShapeAttr::M16n8k32,
        RegisterMmaElementAttr::U4,
        RegisterMmaElementAttr::S4,
        RegisterMmaOverflowAttr::Wrapping,
        [vec![i32_value; 4], vec![u32_value; 6]].concat(),
        vec![i32_ty.into(); 4]
    );
    assert!(verify_op(&m16k32_int4_with_int8_carriers, &ctx).is_err());

    let m16k32_int8_with_int4_carriers = int_mma!(
        RegisterMmaShapeAttr::M16n8k32,
        RegisterMmaElementAttr::S8,
        RegisterMmaElementAttr::U8,
        RegisterMmaOverflowAttr::Satfinite,
        [vec![i32_value; 4], vec![u32_value; 3]].concat(),
        vec![i32_ty.into(); 4]
    );
    assert!(verify_op(&m16k32_int8_with_int4_carriers, &ctx).is_err());

    let m16k64_int4_with_k32_carriers = int_mma!(
        RegisterMmaShapeAttr::M16n8k64,
        RegisterMmaElementAttr::S4,
        RegisterMmaElementAttr::U4,
        RegisterMmaOverflowAttr::Wrapping,
        [vec![i32_value; 4], vec![u32_value; 3]].concat(),
        vec![i32_ty.into(); 4]
    );
    assert!(verify_op(&m16k64_int4_with_k32_carriers, &ctx).is_err());

    let m16k32_int4_with_k64_carriers = int_mma!(
        RegisterMmaShapeAttr::M16n8k32,
        RegisterMmaElementAttr::U4,
        RegisterMmaElementAttr::S4,
        RegisterMmaOverflowAttr::Satfinite,
        [vec![i32_value; 4], vec![u32_value; 6]].concat(),
        vec![i32_ty.into(); 4]
    );
    assert!(verify_op(&m16k32_int4_with_k64_carriers, &ctx).is_err());

    let int8_on_m16k64_shape = int_mma!(
        RegisterMmaShapeAttr::M16n8k64,
        RegisterMmaElementAttr::S8,
        RegisterMmaElementAttr::U8,
        RegisterMmaOverflowAttr::Wrapping,
        [vec![i32_value; 4], vec![u32_value; 6]].concat(),
        vec![i32_ty.into(); 4]
    );
    assert!(verify_op(&int8_on_m16k64_shape, &ctx).is_err());

    for (shape, accumulator_count, operand_count, result_count) in [
        (RegisterMmaShapeAttr::M8n8k16, 2, 4, 2),
        (RegisterMmaShapeAttr::M16n8k16, 4, 7, 4),
        (RegisterMmaShapeAttr::M16n8k32, 4, 10, 4),
    ] {
        for wrong_count in [operand_count - 1, operand_count + 1] {
            let operands = [
                vec![i32_value; accumulator_count],
                vec![u32_value; wrong_count - accumulator_count],
            ]
            .concat();
            let mma = int_mma!(
                shape.clone(),
                RegisterMmaElementAttr::S8,
                RegisterMmaElementAttr::U8,
                RegisterMmaOverflowAttr::Wrapping,
                operands,
                vec![i32_ty.into(); result_count]
            );
            assert!(verify_op(&mma, &ctx).is_err());
        }
    }

    let m8_wrong_result_signedness = int_mma!(
        RegisterMmaShapeAttr::M8n8k16,
        RegisterMmaElementAttr::S8,
        RegisterMmaElementAttr::U8,
        RegisterMmaOverflowAttr::Wrapping,
        [vec![i32_value; 2], vec![u32_value; 2]].concat(),
        vec![signless_i32_ty.into(); 2]
    );
    assert!(verify_op(&m8_wrong_result_signedness, &ctx).is_err());

    let m8_wrong_accumulator_signedness = int_mma!(
        RegisterMmaShapeAttr::M8n8k16,
        RegisterMmaElementAttr::S8,
        RegisterMmaElementAttr::U8,
        RegisterMmaOverflowAttr::Wrapping,
        vec![u32_value; 4],
        vec![i32_ty.into(); 2]
    );
    assert!(verify_op(&m8_wrong_accumulator_signedness, &ctx).is_err());

    let m8_wrong_fragment_signedness = int_mma!(
        RegisterMmaShapeAttr::M8n8k16,
        RegisterMmaElementAttr::U8,
        RegisterMmaElementAttr::S8,
        RegisterMmaOverflowAttr::Satfinite,
        vec![i32_value; 4],
        vec![i32_ty.into(); 2]
    );
    assert!(verify_op(&m8_wrong_fragment_signedness, &ctx).is_err());

    let m8_crossed_element = int_mma!(
        RegisterMmaShapeAttr::M8n8k16,
        RegisterMmaElementAttr::S8,
        RegisterMmaElementAttr::F16,
        RegisterMmaOverflowAttr::Wrapping,
        [vec![i32_value; 2], vec![u32_value; 2]].concat(),
        vec![i32_ty.into(); 2]
    );
    assert!(verify_op(&m8_crossed_element, &ctx).is_err());

    let m8_crossed_overflow = int_mma!(
        RegisterMmaShapeAttr::M8n8k16,
        RegisterMmaElementAttr::U8,
        RegisterMmaElementAttr::U8,
        RegisterMmaOverflowAttr::NotApplicable,
        [vec![i32_value; 2], vec![u32_value; 2]].concat(),
        vec![i32_ty.into(); 2]
    );
    assert!(verify_op(&m8_crossed_overflow, &ctx).is_err());

    let m8_crossed_carrier_shape = int_mma!(
        RegisterMmaShapeAttr::M8n8k16,
        RegisterMmaElementAttr::S8,
        RegisterMmaElementAttr::S8,
        RegisterMmaOverflowAttr::Wrapping,
        [vec![i32_value; 4], vec![u32_value; 3]].concat(),
        vec![i32_ty.into(); 4]
    );
    assert!(verify_op(&m8_crossed_carrier_shape, &ctx).is_err());

    let m8_crossed_shape = int_mma!(
        RegisterMmaShapeAttr::M8n8k4,
        RegisterMmaElementAttr::S8,
        RegisterMmaElementAttr::S8,
        RegisterMmaOverflowAttr::Wrapping,
        [vec![i32_value; 2], vec![u32_value; 2]].concat(),
        vec![i32_ty.into(); 2]
    );
    assert!(verify_op(&m8_crossed_shape, &ctx).is_err());

    let wrong_result_signedness = int_mma!(
        RegisterMmaShapeAttr::M16n8k16,
        RegisterMmaElementAttr::S8,
        RegisterMmaElementAttr::U8,
        RegisterMmaOverflowAttr::Wrapping,
        [vec![i32_value; 4], vec![u32_value; 3]].concat(),
        vec![signless_i32_ty.into(); 4]
    );
    assert!(verify_op(&wrong_result_signedness, &ctx).is_err());

    let wrong_accumulator_signedness = int_mma!(
        RegisterMmaShapeAttr::M16n8k16,
        RegisterMmaElementAttr::S8,
        RegisterMmaElementAttr::U8,
        RegisterMmaOverflowAttr::Wrapping,
        vec![u32_value; 7],
        vec![i32_ty.into(); 4]
    );
    assert!(verify_op(&wrong_accumulator_signedness, &ctx).is_err());

    let wrong_fragment_signedness = int_mma!(
        RegisterMmaShapeAttr::M16n8k32,
        RegisterMmaElementAttr::U8,
        RegisterMmaElementAttr::S8,
        RegisterMmaOverflowAttr::Satfinite,
        vec![i32_value; 10],
        vec![i32_ty.into(); 4]
    );
    assert!(verify_op(&wrong_fragment_signedness, &ctx).is_err());

    let crossed_element = int_mma!(
        RegisterMmaShapeAttr::M16n8k16,
        RegisterMmaElementAttr::S8,
        RegisterMmaElementAttr::F16,
        RegisterMmaOverflowAttr::Wrapping,
        [vec![i32_value; 4], vec![u32_value; 3]].concat(),
        vec![i32_ty.into(); 4]
    );
    assert!(verify_op(&crossed_element, &ctx).is_err());

    let crossed_overflow = int_mma!(
        RegisterMmaShapeAttr::M16n8k32,
        RegisterMmaElementAttr::U8,
        RegisterMmaElementAttr::U8,
        RegisterMmaOverflowAttr::NotApplicable,
        [vec![i32_value; 4], vec![u32_value; 6]].concat(),
        vec![i32_ty.into(); 4]
    );
    assert!(verify_op(&crossed_overflow, &ctx).is_err());

    let crossed_shape = int_mma!(
        RegisterMmaShapeAttr::M16n8k8,
        RegisterMmaElementAttr::S8,
        RegisterMmaElementAttr::S8,
        RegisterMmaOverflowAttr::Satfinite,
        [vec![i32_value; 4], vec![u32_value; 3]].concat(),
        vec![i32_ty.into(); 4]
    );
    assert!(verify_op(&crossed_shape, &ctx).is_err());
}

#[test]
fn generated_register_mma_verifies_dense_b1_families() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    fn b1_mma(
        ctx: &mut Context,
        shape: RegisterMmaShapeAttr,
        operation: Option<RegisterMmaOperationAttr>,
    ) -> RegisterMmaOp {
        let (accumulator_count, a_count, b_count, result_count) = match shape {
            RegisterMmaShapeAttr::M8n8k128 => (2, 1, 1, 2),
            RegisterMmaShapeAttr::M16n8k128 => (4, 2, 1, 4),
            RegisterMmaShapeAttr::M16n8k256 => (4, 4, 2, 4),
            _ => panic!("unsupported B1 MMA shape"),
        };
        let i32_ty = IntegerType::get(ctx, 32, Signedness::Signed);
        let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
        let argument_types = (0..accumulator_count)
            .map(|_| i32_ty.into())
            .chain((0..a_count + b_count).map(|_| u32_ty.into()))
            .collect();
        let block = BasicBlock::new(ctx, None, argument_types);
        let operands = (0..accumulator_count + a_count + b_count)
            .map(|index| block.deref(ctx).get_argument(index))
            .collect();
        let op = Operation::new(
            ctx,
            RegisterMmaOp::get_concrete_op_info(),
            vec![i32_ty.into(); result_count],
            operands,
            vec![],
            0,
        );
        let mma = RegisterMmaOp::new(op);
        mma.set_attr_nvvm_register_mma_shape(ctx, shape);
        if let Some(operation) = operation {
            mma.set_attr_nvvm_register_mma_operation(ctx, operation);
        }
        mma.set_attr_nvvm_register_mma_accumulator(ctx, RegisterMmaAccumulatorAttr::S32);
        mma.set_attr_nvvm_register_mma_a_element(ctx, RegisterMmaElementAttr::B1);
        mma.set_attr_nvvm_register_mma_b_element(ctx, RegisterMmaElementAttr::B1);
        mma.set_attr_nvvm_register_mma_a_layout(ctx, RegisterMmaLayoutAttr::Row);
        mma.set_attr_nvvm_register_mma_b_layout(ctx, RegisterMmaLayoutAttr::Col);
        mma.set_attr_nvvm_register_mma_overflow(ctx, RegisterMmaOverflowAttr::Wrapping);
        mma
    }

    let mut accepted = 0;
    for shape in [
        RegisterMmaShapeAttr::M8n8k128,
        RegisterMmaShapeAttr::M16n8k128,
        RegisterMmaShapeAttr::M16n8k256,
    ] {
        for operation in [
            RegisterMmaOperationAttr::XorPopc,
            RegisterMmaOperationAttr::AndPopc,
        ] {
            let mma = b1_mma(&mut ctx, shape.clone(), Some(operation));
            assert!(verify_op(&mma, &ctx).is_ok(), "rejected {shape:?}");
            accepted += 1;
        }
    }
    assert_eq!(accepted, 6);

    let multiply = b1_mma(
        &mut ctx,
        RegisterMmaShapeAttr::M8n8k128,
        Some(RegisterMmaOperationAttr::Multiply),
    );
    assert!(verify_op(&multiply, &ctx).is_err());

    let wrong_shape = b1_mma(
        &mut ctx,
        RegisterMmaShapeAttr::M16n8k128,
        Some(RegisterMmaOperationAttr::XorPopc),
    );
    wrong_shape.set_attr_nvvm_register_mma_shape(&ctx, RegisterMmaShapeAttr::M16n8k64);
    assert!(verify_op(&wrong_shape, &ctx).is_err());

    let missing_operation = b1_mma(&mut ctx, RegisterMmaShapeAttr::M8n8k128, None);
    assert!(verify_op(&missing_operation, &ctx).is_err());
}

#[test]
fn generated_sparse_mma_verifies_all_int8_variants_and_metadata_modes() {
    use dialect_mir::ops::MirConstantOp;
    use pliron::builtin::{attributes::IntegerAttr, ops::ConstantOp};
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let signless_i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![i32_ty.into(), u32_ty.into(), signless_i32_ty.into()],
    );
    let i32_value = block.deref(&ctx).get_argument(0);
    let u32_value = block.deref(&ctx).get_argument(1);
    let signless_i32_value = block.deref(&ctx).get_argument(2);

    let integer = |value| {
        IntegerAttr::new(
            u32_ty,
            APInt::from_u32(value, NonZeroUsize::new(32).unwrap()),
        )
    };
    let builtin_zero = ConstantOp::new(&mut ctx, Box::new(integer(0)));
    let builtin_zero = builtin_zero.get_operation().deref(&ctx).get_result(0);
    let builtin_two = ConstantOp::new(&mut ctx, Box::new(integer(2)));
    let builtin_two = builtin_two.get_operation().deref(&ctx).get_result(0);
    let mir_one = Operation::new(
        &mut ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![u32_ty.into()],
        vec![],
        vec![],
        0,
    );
    MirConstantOp::new(mir_one).set_attr_value(&ctx, integer(1));
    let mir_one = mir_one.deref(&ctx).get_result(0);

    macro_rules! sparse_mma {
        ($operands:expr, $results:expr, $a:expr, $b:expr, $overflow:expr, $metadata:expr) => {{
            let operation = Operation::new(
                &mut ctx,
                SparseMmaOp::get_concrete_op_info(),
                $results,
                $operands,
                vec![],
                0,
            );
            let mma = SparseMmaOp::new(operation);
            mma.set_attr_nvvm_sparse_mma_shape(&ctx, SparseMmaShapeAttr::M16n8k32);
            mma.set_attr_nvvm_sparse_mma_accumulator(&ctx, SparseMmaAccumulatorAttr::S32);
            mma.set_attr_nvvm_sparse_mma_a_element(&ctx, $a);
            mma.set_attr_nvvm_sparse_mma_b_element(&ctx, $b);
            mma.set_attr_nvvm_sparse_mma_a_layout(&ctx, SparseMmaLayoutAttr::Row);
            mma.set_attr_nvvm_sparse_mma_b_layout(&ctx, SparseMmaLayoutAttr::Col);
            mma.set_attr_nvvm_sparse_mma_overflow(&ctx, $overflow);
            mma.set_attr_nvvm_sparse_mma_metadata(&ctx, $metadata);
            mma.set_attr_nvvm_sparse_mma_selector(&ctx, SparseMmaSelectorAttr::ImmediateZeroOrOne);
            mma
        }};
    }

    let variants = [
        (
            SparseMmaElementAttr::S8,
            SparseMmaElementAttr::S8,
            SparseMmaOverflowAttr::Wrapping,
        ),
        (
            SparseMmaElementAttr::S8,
            SparseMmaElementAttr::U8,
            SparseMmaOverflowAttr::Wrapping,
        ),
        (
            SparseMmaElementAttr::U8,
            SparseMmaElementAttr::U8,
            SparseMmaOverflowAttr::Wrapping,
        ),
        (
            SparseMmaElementAttr::U8,
            SparseMmaElementAttr::S8,
            SparseMmaOverflowAttr::Wrapping,
        ),
        (
            SparseMmaElementAttr::S8,
            SparseMmaElementAttr::S8,
            SparseMmaOverflowAttr::Satfinite,
        ),
        (
            SparseMmaElementAttr::S8,
            SparseMmaElementAttr::U8,
            SparseMmaOverflowAttr::Satfinite,
        ),
        (
            SparseMmaElementAttr::U8,
            SparseMmaElementAttr::U8,
            SparseMmaOverflowAttr::Satfinite,
        ),
        (
            SparseMmaElementAttr::U8,
            SparseMmaElementAttr::S8,
            SparseMmaOverflowAttr::Satfinite,
        ),
    ];
    for metadata in [
        SparseMmaMetadataAttr::Standard,
        SparseMmaMetadataAttr::Ordered,
    ] {
        for (index, (a_element, b_element, overflow)) in variants.iter().enumerate() {
            let selector = if index % 2 == 0 {
                builtin_zero
            } else {
                mir_one
            };
            let operands = [vec![i32_value; 4], vec![u32_value; 5], vec![selector]].concat();
            let mma = sparse_mma!(
                operands,
                vec![i32_ty.into(); 4],
                a_element.clone(),
                b_element.clone(),
                overflow.clone(),
                metadata.clone()
            );
            assert_eq!(
                mma.get_attr_nvvm_sparse_mma_a_element(&ctx).as_deref(),
                Some(a_element)
            );
            assert_eq!(
                mma.get_attr_nvvm_sparse_mma_b_element(&ctx).as_deref(),
                Some(b_element)
            );
            assert_eq!(
                mma.get_attr_nvvm_sparse_mma_overflow(&ctx).as_deref(),
                Some(overflow)
            );
            assert_eq!(
                mma.get_attr_nvvm_sparse_mma_metadata(&ctx).as_deref(),
                Some(&metadata)
            );
            assert!(
                verify_op(&mma, &ctx).is_ok(),
                "rejected sparse {metadata:?} {a_element:?}x{b_element:?} {overflow:?}"
            );
        }
    }

    for metadata in [
        SparseMmaMetadataAttr::Standard,
        SparseMmaMetadataAttr::Ordered,
    ] {
        for selector in [u32_value, builtin_two] {
            let invalid = sparse_mma!(
                [vec![i32_value; 4], vec![u32_value; 5], vec![selector],].concat(),
                vec![i32_ty.into(); 4],
                SparseMmaElementAttr::S8,
                SparseMmaElementAttr::U8,
                SparseMmaOverflowAttr::Wrapping,
                metadata.clone()
            );
            assert!(verify_op(&invalid, &ctx).is_err());
        }
    }

    let wrong_accumulator_type = sparse_mma!(
        [
            vec![signless_i32_value; 4],
            vec![u32_value; 5],
            vec![builtin_zero],
        ]
        .concat(),
        vec![i32_ty.into(); 4],
        SparseMmaElementAttr::U8,
        SparseMmaElementAttr::S8,
        SparseMmaOverflowAttr::Satfinite,
        SparseMmaMetadataAttr::Standard
    );
    assert!(verify_op(&wrong_accumulator_type, &ctx).is_err());

    let wrong_count = sparse_mma!(
        [vec![i32_value; 4], vec![u32_value; 4], vec![builtin_zero],].concat(),
        vec![i32_ty.into(); 4],
        SparseMmaElementAttr::S8,
        SparseMmaElementAttr::S8,
        SparseMmaOverflowAttr::Wrapping,
        SparseMmaMetadataAttr::Standard
    );
    assert!(verify_op(&wrong_count, &ctx).is_err());

    let wrong_results = sparse_mma!(
        [vec![i32_value; 4], vec![u32_value; 5], vec![mir_one],].concat(),
        vec![signless_i32_ty.into(); 4],
        SparseMmaElementAttr::U8,
        SparseMmaElementAttr::U8,
        SparseMmaOverflowAttr::Wrapping,
        SparseMmaMetadataAttr::Standard
    );
    assert!(verify_op(&wrong_results, &ctx).is_err());

    let wrong_layout = sparse_mma!(
        [vec![i32_value; 4], vec![u32_value; 5], vec![builtin_zero],].concat(),
        vec![i32_ty.into(); 4],
        SparseMmaElementAttr::S8,
        SparseMmaElementAttr::U8,
        SparseMmaOverflowAttr::Satfinite,
        SparseMmaMetadataAttr::Standard
    );
    wrong_layout.set_attr_nvvm_sparse_mma_b_layout(&ctx, SparseMmaLayoutAttr::Row);
    assert!(verify_op(&wrong_layout, &ctx).is_err());
}

#[test]
fn generated_sparse_mma_m16n8k64_verifies_selector_and_carriers() {
    use pliron::builtin::{attributes::IntegerAttr, ops::ConstantOp};
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let signless_i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![i32_ty.into(), u32_ty.into(), signless_i32_ty.into()],
    );
    let i32_value = block.deref(&ctx).get_argument(0);
    let u32_value = block.deref(&ctx).get_argument(1);
    let signless_i32_value = block.deref(&ctx).get_argument(2);
    let integer = |value| {
        IntegerAttr::new(
            u32_ty,
            APInt::from_u32(value, NonZeroUsize::new(32).unwrap()),
        )
    };
    let zero = ConstantOp::new(&mut ctx, Box::new(integer(0)))
        .get_operation()
        .deref(&ctx)
        .get_result(0);
    let one = ConstantOp::new(&mut ctx, Box::new(integer(1)))
        .get_operation()
        .deref(&ctx)
        .get_result(0);
    let two = ConstantOp::new(&mut ctx, Box::new(integer(2)))
        .get_operation()
        .deref(&ctx)
        .get_result(0);

    macro_rules! k64_mma {
        ($operands:expr, $metadata:expr) => {{
            let operation = Operation::new(
                &mut ctx,
                SparseMmaOp::get_concrete_op_info(),
                vec![i32_ty.into(); 4],
                $operands,
                vec![],
                0,
            );
            let mma = SparseMmaOp::new(operation);
            mma.set_attr_nvvm_sparse_mma_shape(&ctx, SparseMmaShapeAttr::M16n8k64);
            mma.set_attr_nvvm_sparse_mma_accumulator(&ctx, SparseMmaAccumulatorAttr::S32);
            mma.set_attr_nvvm_sparse_mma_a_element(&ctx, SparseMmaElementAttr::S8);
            mma.set_attr_nvvm_sparse_mma_b_element(&ctx, SparseMmaElementAttr::U8);
            mma.set_attr_nvvm_sparse_mma_a_layout(&ctx, SparseMmaLayoutAttr::Row);
            mma.set_attr_nvvm_sparse_mma_b_layout(&ctx, SparseMmaLayoutAttr::Col);
            mma.set_attr_nvvm_sparse_mma_overflow(&ctx, SparseMmaOverflowAttr::Wrapping);
            mma.set_attr_nvvm_sparse_mma_metadata(&ctx, $metadata);
            mma.set_attr_nvvm_sparse_mma_selector(&ctx, SparseMmaSelectorAttr::ImmediateZero);
            mma
        }};
    }

    let operands = |selector| [vec![i32_value; 4], vec![u32_value; 9], vec![selector]].concat();
    assert!(
        verify_op(
            &k64_mma!(operands(zero), SparseMmaMetadataAttr::Standard),
            &ctx
        )
        .is_ok()
    );
    assert!(
        verify_op(
            &k64_mma!(operands(zero), SparseMmaMetadataAttr::Ordered),
            &ctx
        )
        .is_ok()
    );
    assert!(
        verify_op(
            &k64_mma!(operands(one), SparseMmaMetadataAttr::Standard),
            &ctx
        )
        .is_err()
    );
    assert!(
        verify_op(
            &k64_mma!(operands(one), SparseMmaMetadataAttr::Ordered),
            &ctx
        )
        .is_err()
    );
    assert!(
        verify_op(
            &k64_mma!(operands(u32_value), SparseMmaMetadataAttr::Standard),
            &ctx
        )
        .is_err()
    );
    assert!(
        verify_op(
            &k64_mma!(operands(u32_value), SparseMmaMetadataAttr::Ordered),
            &ctx
        )
        .is_err()
    );

    let wrong_count = [vec![i32_value; 4], vec![u32_value; 8], vec![zero]].concat();
    assert!(
        verify_op(
            &k64_mma!(wrong_count, SparseMmaMetadataAttr::Standard),
            &ctx
        )
        .is_err()
    );

    let mut wrong_type = operands(zero);
    wrong_type[4] = signless_i32_value;
    assert!(verify_op(&k64_mma!(wrong_type, SparseMmaMetadataAttr::Standard), &ctx).is_err());

    macro_rules! int4_mma {
        ($operands:expr, $a:expr, $b:expr, $overflow:expr, $metadata:expr) => {{
            let operation = Operation::new(
                &mut ctx,
                SparseMmaOp::get_concrete_op_info(),
                vec![i32_ty.into(); 4],
                $operands,
                vec![],
                0,
            );
            let mma = SparseMmaOp::new(operation);
            mma.set_attr_nvvm_sparse_mma_shape(&ctx, SparseMmaShapeAttr::M16n8k64);
            mma.set_attr_nvvm_sparse_mma_accumulator(&ctx, SparseMmaAccumulatorAttr::S32);
            mma.set_attr_nvvm_sparse_mma_a_element(&ctx, $a);
            mma.set_attr_nvvm_sparse_mma_b_element(&ctx, $b);
            mma.set_attr_nvvm_sparse_mma_a_layout(&ctx, SparseMmaLayoutAttr::Row);
            mma.set_attr_nvvm_sparse_mma_b_layout(&ctx, SparseMmaLayoutAttr::Col);
            mma.set_attr_nvvm_sparse_mma_overflow(&ctx, $overflow);
            mma.set_attr_nvvm_sparse_mma_metadata(&ctx, $metadata);
            mma.set_attr_nvvm_sparse_mma_selector(&ctx, SparseMmaSelectorAttr::ImmediateZeroOrOne);
            mma
        }};
    }

    let int4_variants = [
        (
            SparseMmaElementAttr::S4,
            SparseMmaElementAttr::S4,
            SparseMmaOverflowAttr::Wrapping,
        ),
        (
            SparseMmaElementAttr::S4,
            SparseMmaElementAttr::U4,
            SparseMmaOverflowAttr::Wrapping,
        ),
        (
            SparseMmaElementAttr::U4,
            SparseMmaElementAttr::U4,
            SparseMmaOverflowAttr::Wrapping,
        ),
        (
            SparseMmaElementAttr::U4,
            SparseMmaElementAttr::S4,
            SparseMmaOverflowAttr::Wrapping,
        ),
        (
            SparseMmaElementAttr::S4,
            SparseMmaElementAttr::S4,
            SparseMmaOverflowAttr::Satfinite,
        ),
        (
            SparseMmaElementAttr::S4,
            SparseMmaElementAttr::U4,
            SparseMmaOverflowAttr::Satfinite,
        ),
        (
            SparseMmaElementAttr::U4,
            SparseMmaElementAttr::U4,
            SparseMmaOverflowAttr::Satfinite,
        ),
        (
            SparseMmaElementAttr::U4,
            SparseMmaElementAttr::S4,
            SparseMmaOverflowAttr::Satfinite,
        ),
    ];
    let int4_operands =
        |selector| [vec![i32_value; 4], vec![u32_value; 5], vec![selector]].concat();
    for metadata in [
        SparseMmaMetadataAttr::Standard,
        SparseMmaMetadataAttr::Ordered,
    ] {
        for (index, (a, b, overflow)) in int4_variants.iter().enumerate() {
            let selector = if index % 2 == 0 { zero } else { one };
            assert!(
                verify_op(
                    &int4_mma!(
                        int4_operands(selector),
                        a.clone(),
                        b.clone(),
                        overflow.clone(),
                        metadata.clone()
                    ),
                    &ctx,
                )
                .is_ok()
            );
        }
    }

    assert!(
        verify_op(
            &int4_mma!(
                int4_operands(zero),
                SparseMmaElementAttr::S4,
                SparseMmaElementAttr::U8,
                SparseMmaOverflowAttr::Wrapping,
                SparseMmaMetadataAttr::Standard
            ),
            &ctx,
        )
        .is_err()
    );
    for metadata in [
        SparseMmaMetadataAttr::Standard,
        SparseMmaMetadataAttr::Ordered,
    ] {
        for selector in [two, u32_value] {
            assert!(
                verify_op(
                    &int4_mma!(
                        int4_operands(selector),
                        SparseMmaElementAttr::S4,
                        SparseMmaElementAttr::U4,
                        SparseMmaOverflowAttr::Wrapping,
                        metadata.clone()
                    ),
                    &ctx,
                )
                .is_err()
            );
        }
    }
}

#[test]
fn generated_sparse_mma_m16n8k128_int4_verifies_metadata_selector_and_widths() {
    use pliron::builtin::{attributes::IntegerAttr, ops::ConstantOp};
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into(), u32_ty.into()]);
    let i32_value = block.deref(&ctx).get_argument(0);
    let u32_value = block.deref(&ctx).get_argument(1);
    let integer = |value| {
        IntegerAttr::new(
            u32_ty,
            APInt::from_u32(value, NonZeroUsize::new(32).unwrap()),
        )
    };
    let zero = ConstantOp::new(&mut ctx, Box::new(integer(0)))
        .get_operation()
        .deref(&ctx)
        .get_result(0);
    let one = ConstantOp::new(&mut ctx, Box::new(integer(1)))
        .get_operation()
        .deref(&ctx)
        .get_result(0);

    macro_rules! k128_mma {
        ($operands:expr, $a:expr, $b:expr, $overflow:expr, $metadata:expr) => {{
            let operation = Operation::new(
                &mut ctx,
                SparseMmaOp::get_concrete_op_info(),
                vec![i32_ty.into(); 4],
                $operands,
                vec![],
                0,
            );
            let mma = SparseMmaOp::new(operation);
            mma.set_attr_nvvm_sparse_mma_shape(&ctx, SparseMmaShapeAttr::M16n8k128);
            mma.set_attr_nvvm_sparse_mma_accumulator(&ctx, SparseMmaAccumulatorAttr::S32);
            mma.set_attr_nvvm_sparse_mma_a_element(&ctx, $a);
            mma.set_attr_nvvm_sparse_mma_b_element(&ctx, $b);
            mma.set_attr_nvvm_sparse_mma_a_layout(&ctx, SparseMmaLayoutAttr::Row);
            mma.set_attr_nvvm_sparse_mma_b_layout(&ctx, SparseMmaLayoutAttr::Col);
            mma.set_attr_nvvm_sparse_mma_overflow(&ctx, $overflow);
            mma.set_attr_nvvm_sparse_mma_metadata(&ctx, $metadata);
            mma.set_attr_nvvm_sparse_mma_selector(&ctx, SparseMmaSelectorAttr::ImmediateZero);
            mma
        }};
    }

    let variants = [
        (
            SparseMmaElementAttr::S4,
            SparseMmaElementAttr::S4,
            SparseMmaOverflowAttr::Wrapping,
        ),
        (
            SparseMmaElementAttr::S4,
            SparseMmaElementAttr::U4,
            SparseMmaOverflowAttr::Wrapping,
        ),
        (
            SparseMmaElementAttr::U4,
            SparseMmaElementAttr::U4,
            SparseMmaOverflowAttr::Wrapping,
        ),
        (
            SparseMmaElementAttr::U4,
            SparseMmaElementAttr::S4,
            SparseMmaOverflowAttr::Wrapping,
        ),
        (
            SparseMmaElementAttr::S4,
            SparseMmaElementAttr::S4,
            SparseMmaOverflowAttr::Satfinite,
        ),
        (
            SparseMmaElementAttr::S4,
            SparseMmaElementAttr::U4,
            SparseMmaOverflowAttr::Satfinite,
        ),
        (
            SparseMmaElementAttr::U4,
            SparseMmaElementAttr::U4,
            SparseMmaOverflowAttr::Satfinite,
        ),
        (
            SparseMmaElementAttr::U4,
            SparseMmaElementAttr::S4,
            SparseMmaOverflowAttr::Satfinite,
        ),
    ];
    let operands = |selector| [vec![i32_value; 4], vec![u32_value; 9], vec![selector]].concat();
    for metadata in [
        SparseMmaMetadataAttr::Standard,
        SparseMmaMetadataAttr::Ordered,
    ] {
        for (a, b, overflow) in &variants {
            assert!(
                verify_op(
                    &k128_mma!(
                        operands(zero),
                        a.clone(),
                        b.clone(),
                        overflow.clone(),
                        metadata.clone()
                    ),
                    &ctx,
                )
                .is_ok()
            );
        }
    }

    for metadata in [
        SparseMmaMetadataAttr::Standard,
        SparseMmaMetadataAttr::Ordered,
    ] {
        for selector in [one, u32_value] {
            assert!(
                verify_op(
                    &k128_mma!(
                        operands(selector),
                        SparseMmaElementAttr::S4,
                        SparseMmaElementAttr::U4,
                        SparseMmaOverflowAttr::Wrapping,
                        metadata.clone()
                    ),
                    &ctx,
                )
                .is_err()
            );
        }
    }
    assert!(
        verify_op(
            &k128_mma!(
                operands(zero),
                SparseMmaElementAttr::S4,
                SparseMmaElementAttr::U8,
                SparseMmaOverflowAttr::Wrapping,
                SparseMmaMetadataAttr::Standard
            ),
            &ctx,
        )
        .is_err()
    );
    assert!(
        verify_op(
            &k128_mma!(
                [vec![i32_value; 4], vec![u32_value; 8], vec![zero]].concat(),
                SparseMmaElementAttr::S4,
                SparseMmaElementAttr::S4,
                SparseMmaOverflowAttr::Wrapping,
                SparseMmaMetadataAttr::Standard
            ),
            &ctx,
        )
        .is_err()
    );
}

#[test]
fn generated_sparse_mma_f8f6f4_verifies_all_formats_and_closed_shape() {
    use pliron::builtin::{attributes::IntegerAttr, ops::ConstantOp};
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let f32_ty = FP32Type::get(&ctx);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let block = BasicBlock::new(&mut ctx, None, vec![f32_ty.into(), u32_ty.into()]);
    let f32_value = block.deref(&ctx).get_argument(0);
    let u32_value = block.deref(&ctx).get_argument(1);
    let integer = |value| {
        IntegerAttr::new(
            u32_ty,
            APInt::from_u32(value, NonZeroUsize::new(32).unwrap()),
        )
    };
    let zero = ConstantOp::new(&mut ctx, Box::new(integer(0)))
        .get_operation()
        .deref(&ctx)
        .get_result(0);
    let one = ConstantOp::new(&mut ctx, Box::new(integer(1)))
        .get_operation()
        .deref(&ctx)
        .get_result(0);

    macro_rules! f8f6f4_mma {
        ($operands:expr, $results:expr, $a:expr, $b:expr, $overflow:expr, $metadata:expr) => {{
            let operation = Operation::new(
                &mut ctx,
                SparseMmaOp::get_concrete_op_info(),
                $results,
                $operands,
                vec![],
                0,
            );
            let mma = SparseMmaOp::new(operation);
            mma.set_attr_nvvm_sparse_mma_shape(&ctx, SparseMmaShapeAttr::M16n8k64);
            mma.set_attr_nvvm_sparse_mma_accumulator(&ctx, SparseMmaAccumulatorAttr::F32);
            mma.set_attr_nvvm_sparse_mma_a_element(&ctx, $a);
            mma.set_attr_nvvm_sparse_mma_b_element(&ctx, $b);
            mma.set_attr_nvvm_sparse_mma_a_layout(&ctx, SparseMmaLayoutAttr::Row);
            mma.set_attr_nvvm_sparse_mma_b_layout(&ctx, SparseMmaLayoutAttr::Col);
            mma.set_attr_nvvm_sparse_mma_overflow(&ctx, $overflow);
            mma.set_attr_nvvm_sparse_mma_metadata(&ctx, $metadata);
            mma.set_attr_nvvm_sparse_mma_selector(&ctx, SparseMmaSelectorAttr::ImmediateZero);
            mma
        }};
    }

    let elements = [
        SparseMmaElementAttr::E2m1,
        SparseMmaElementAttr::E2m3,
        SparseMmaElementAttr::E3m2,
        SparseMmaElementAttr::E4m3,
        SparseMmaElementAttr::E5m2,
    ];
    let operands = |selector| [vec![f32_value; 4], vec![u32_value; 9], vec![selector]].concat();
    for a in &elements {
        for b in &elements {
            let mma = f8f6f4_mma!(
                operands(zero),
                vec![f32_ty.into(); 4],
                a.clone(),
                b.clone(),
                SparseMmaOverflowAttr::NotApplicable,
                SparseMmaMetadataAttr::Ordered
            );
            assert!(verify_op(&mma, &ctx).is_ok(), "rejected {a:?}x{b:?}");
        }
    }

    for invalid in [
        f8f6f4_mma!(
            operands(one),
            vec![f32_ty.into(); 4],
            SparseMmaElementAttr::E2m1,
            SparseMmaElementAttr::E2m1,
            SparseMmaOverflowAttr::NotApplicable,
            SparseMmaMetadataAttr::Ordered
        ),
        f8f6f4_mma!(
            operands(zero),
            vec![f32_ty.into(); 4],
            SparseMmaElementAttr::E2m1,
            SparseMmaElementAttr::E2m1,
            SparseMmaOverflowAttr::Wrapping,
            SparseMmaMetadataAttr::Ordered
        ),
        f8f6f4_mma!(
            operands(zero),
            vec![f32_ty.into(); 4],
            SparseMmaElementAttr::E2m1,
            SparseMmaElementAttr::E2m1,
            SparseMmaOverflowAttr::NotApplicable,
            SparseMmaMetadataAttr::Standard
        ),
        f8f6f4_mma!(
            operands(zero),
            vec![u32_ty.into(); 4],
            SparseMmaElementAttr::E2m1,
            SparseMmaElementAttr::E2m1,
            SparseMmaOverflowAttr::NotApplicable,
            SparseMmaMetadataAttr::Ordered
        ),
    ] {
        assert!(verify_op(&invalid, &ctx).is_err());
    }
}

#[test]
fn test_matrix_memory_ops_verify_pointer_and_packed_register_types() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let f32_ty = FP32Type::get(&ctx);
    let ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), true);
    let load_ptr_ty = MirPtrType::get_generic(&mut ctx, u32_ty.into(), true);

    let load_block = BasicBlock::new(&mut ctx, None, vec![load_ptr_ty.into()]);
    let load_pointer = load_block.deref(&ctx).get_argument(0);
    let load = make_ldmatrix_x2(&mut ctx, load_pointer, vec![u32_ty.into(), u32_ty.into()]);
    assert!(load.verify(&ctx).is_ok());

    let bad_load_pointer_block = BasicBlock::new(&mut ctx, None, vec![i64_ty.into()]);
    let bad_pointer = bad_load_pointer_block.deref(&ctx).get_argument(0);
    let bad_load_pointer =
        make_ldmatrix_x2(&mut ctx, bad_pointer, vec![u32_ty.into(), u32_ty.into()]);
    assert!(bad_load_pointer.verify(&ctx).is_err());

    let bad_load_result =
        make_ldmatrix_x2(&mut ctx, load_pointer, vec![u32_ty.into(), f32_ty.into()]);
    assert!(bad_load_result.verify(&ctx).is_err());

    let store_block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            ptr_ty.into(),
            i32_ty.into(),
            i32_ty.into(),
            i32_ty.into(),
            i32_ty.into(),
        ],
    );
    let store_operands = (0..5)
        .map(|index| store_block.deref(&ctx).get_argument(index))
        .collect();
    let store = Operation::new(
        &mut ctx,
        StmatrixM8n8X4Op::get_concrete_op_info(),
        vec![],
        store_operands,
        vec![],
        0,
    );
    assert!(StmatrixM8n8X4Op::new(store).verify(&ctx).is_ok());

    let bad_store_block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            ptr_ty.into(),
            f32_ty.into(),
            i32_ty.into(),
            i32_ty.into(),
            i32_ty.into(),
        ],
    );
    let bad_store_operands = (0..5)
        .map(|index| bad_store_block.deref(&ctx).get_argument(index))
        .collect();
    let bad_store = Operation::new(
        &mut ctx,
        StmatrixM8n8X4Op::get_concrete_op_info(),
        vec![],
        bad_store_operands,
        vec![],
        0,
    );
    assert!(StmatrixM8n8X4Op::new(bad_store).verify(&ctx).is_err());
}

#[test]
fn generated_plain_sparse_fp8_standard_metadata_is_a_verified_combination() {
    use pliron::builtin::{attributes::IntegerAttr, ops::ConstantOp};
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let f32_ty = FP32Type::get(&ctx);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let block = BasicBlock::new(&mut ctx, None, vec![f32_ty.into(), u32_ty.into()]);
    let f32_value = block.deref(&ctx).get_argument(0);
    let u32_value = block.deref(&ctx).get_argument(1);
    let selector_for = |ctx: &mut Context, value| {
        let attribute = IntegerAttr::new(
            u32_ty,
            APInt::from_u32(value, NonZeroUsize::new(32).unwrap()),
        );
        ConstantOp::new(ctx, Box::new(attribute))
            .get_operation()
            .deref(ctx)
            .get_result(0)
    };
    let zero = selector_for(&mut ctx, 0);
    let one = selector_for(&mut ctx, 1);

    // The reviewed plain SM89 forms: standard sparsity metadata with an F32
    // accumulator over the e4m3/e5m2 multiplicand pair.
    let plain_mma =
        |ctx: &mut Context, selector, a: SparseMmaElementAttr, b: SparseMmaElementAttr| {
            let operands = [vec![f32_value; 4], vec![u32_value; 9], vec![selector]].concat();
            let operation = Operation::new(
                ctx,
                SparseMmaOp::get_concrete_op_info(),
                vec![f32_ty.into(); 4],
                operands,
                vec![],
                0,
            );
            let mma = SparseMmaOp::new(operation);
            mma.set_attr_nvvm_sparse_mma_shape(ctx, SparseMmaShapeAttr::M16n8k64);
            mma.set_attr_nvvm_sparse_mma_accumulator(ctx, SparseMmaAccumulatorAttr::F32);
            mma.set_attr_nvvm_sparse_mma_a_element(ctx, a);
            mma.set_attr_nvvm_sparse_mma_b_element(ctx, b);
            mma.set_attr_nvvm_sparse_mma_a_layout(ctx, SparseMmaLayoutAttr::Row);
            mma.set_attr_nvvm_sparse_mma_b_layout(ctx, SparseMmaLayoutAttr::Col);
            mma.set_attr_nvvm_sparse_mma_overflow(ctx, SparseMmaOverflowAttr::NotApplicable);
            mma.set_attr_nvvm_sparse_mma_metadata(ctx, SparseMmaMetadataAttr::Standard);
            mma.set_attr_nvvm_sparse_mma_selector(ctx, SparseMmaSelectorAttr::ImmediateZero);
            mma
        };

    for a in [SparseMmaElementAttr::E4m3, SparseMmaElementAttr::E5m2] {
        for b in [SparseMmaElementAttr::E4m3, SparseMmaElementAttr::E5m2] {
            let mma = plain_mma(&mut ctx, zero, a.clone(), b.clone());
            assert!(verify_op(&mma, &ctx).is_ok(), "rejected {a:?}x{b:?}");
        }
    }

    // The plain forms are selector-immediate-zero only.
    let nonzero_selector = plain_mma(
        &mut ctx,
        one,
        SparseMmaElementAttr::E4m3,
        SparseMmaElementAttr::E4m3,
    );
    assert!(verify_op(&nonzero_selector, &ctx).is_err());

    // Block-scale-only formats stay rejected under standard metadata.
    let block_scale_only = plain_mma(
        &mut ctx,
        zero,
        SparseMmaElementAttr::E2m1,
        SparseMmaElementAttr::E2m1,
    );
    assert!(verify_op(&block_scale_only, &ctx).is_err());
}
