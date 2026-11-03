/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_mir::types::{MirPtrType, address_space};
use dialect_nvvm::ops::{
    AtomicOrdering, AtomicRmwKind, AtomicScope, NvvmAtomAddBf16x2Op, NvvmAtomAddF16x2Op,
    NvvmAtomicCmpxchgOp, NvvmAtomicLoadOp, NvvmAtomicRmwOp, NvvmAtomicStoreOp, PackedAtomicAddOp,
    PackedAtomicAtomicityAttr, PackedAtomicFormatAttr, PackedAtomicOrderingAttr,
    PackedAtomicRoundingAttr, PackedAtomicScopeAttr, PackedAtomicStateSpaceAttr,
    PackedAtomicSubnormalAttr,
};

use pliron::{
    basic_block::BasicBlock,
    builtin::types::{FP32Type, IntegerType, Signedness},
    common_traits::Verify,
    context::Context,
    op::{Op, verify_op},
    operation::Operation,
};

#[test]
fn test_packed_atomic_add_verifies_exact_raw_u32_shape() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let signless_i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let generic_u32_ptr = MirPtrType::get_generic(&mut ctx, u32_ty.into(), true);
    let global_u32_ptr = MirPtrType::get_global(&mut ctx, u32_ty.into(), true);
    let immutable_u32_ptr = MirPtrType::get_generic(&mut ctx, u32_ty.into(), false);
    let shared_u32_ptr = MirPtrType::get_shared(&mut ctx, u32_ty.into(), true);
    let generic_u64_ptr = MirPtrType::get_generic(&mut ctx, u64_ty.into(), true);

    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            generic_u32_ptr.into(),
            global_u32_ptr.into(),
            immutable_u32_ptr.into(),
            shared_u32_ptr.into(),
            generic_u64_ptr.into(),
            u32_ty.into(),
            signless_i32_ty.into(),
        ],
    );
    let generic_ptr = block.deref(&ctx).get_argument(0);
    let global_ptr = block.deref(&ctx).get_argument(1);
    let immutable_ptr = block.deref(&ctx).get_argument(2);
    let shared_ptr = block.deref(&ctx).get_argument(3);
    let wrong_pointee_ptr = block.deref(&ctx).get_argument(4);
    let addend = block.deref(&ctx).get_argument(5);
    let signless_addend = block.deref(&ctx).get_argument(6);

    macro_rules! check_variant {
        ($op:ty) => {{
            for pointer in [generic_ptr, global_ptr] {
                let valid = Operation::new(
                    &mut ctx,
                    <$op>::get_concrete_op_info(),
                    vec![u32_ty.into()],
                    vec![pointer, addend],
                    vec![],
                    0,
                );
                assert!(verify_op(&<$op>::new(valid), &ctx).is_ok());
            }

            for pointer in [immutable_ptr, shared_ptr, wrong_pointee_ptr] {
                let invalid = Operation::new(
                    &mut ctx,
                    <$op>::get_concrete_op_info(),
                    vec![u32_ty.into()],
                    vec![pointer, addend],
                    vec![],
                    0,
                );
                assert!(verify_op(&<$op>::new(invalid), &ctx).is_err());
            }

            let bad_address = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![u32_ty.into()],
                vec![addend, addend],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(bad_address), &ctx).is_err());

            let bad_addend = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![u32_ty.into()],
                vec![generic_ptr, signless_addend],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(bad_addend), &ctx).is_err());

            let bad_result = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![signless_i32_ty.into()],
                vec![generic_ptr, addend],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(bad_result), &ctx).is_err());

            let bad_counts = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![],
                vec![generic_ptr],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(bad_counts), &ctx).is_err());
        }};
    }

    check_variant!(NvvmAtomAddF16x2Op);
    check_variant!(NvvmAtomAddBf16x2Op);
}

#[test]
fn test_generated_packed_atomic_add_requires_closed_attributes_and_raw_u32_shape() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let signless_i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let signed_i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let global_u32_ptr = MirPtrType::get_global(&mut ctx, u32_ty.into(), true);
    let immutable_u32_ptr = MirPtrType::get_global(&mut ctx, u32_ty.into(), false);
    let shared_u32_ptr = MirPtrType::get_shared(&mut ctx, u32_ty.into(), true);
    let local_u32_ptr = MirPtrType::get(&mut ctx, u32_ty.into(), true, address_space::LOCAL);
    let constant_u32_ptr = MirPtrType::get_constant(&mut ctx, u32_ty.into(), true);
    let global_u64_ptr = MirPtrType::get_global(&mut ctx, u64_ty.into(), true);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            global_u32_ptr.into(),
            immutable_u32_ptr.into(),
            shared_u32_ptr.into(),
            local_u32_ptr.into(),
            constant_u32_ptr.into(),
            global_u64_ptr.into(),
            u32_ty.into(),
            signless_i32_ty.into(),
        ],
    );
    let global = block.deref(&ctx).get_argument(0);
    let immutable = block.deref(&ctx).get_argument(1);
    let shared = block.deref(&ctx).get_argument(2);
    let local = block.deref(&ctx).get_argument(3);
    let constant = block.deref(&ctx).get_argument(4);
    let wrong_pointee = block.deref(&ctx).get_argument(5);
    let addend = block.deref(&ctx).get_argument(6);
    let signless = block.deref(&ctx).get_argument(7);

    for format in [
        PackedAtomicFormatAttr::F16x2,
        PackedAtomicFormatAttr::Bf16x2,
    ] {
        let valid = PackedAtomicAddOp::build(&mut ctx, global, addend, format);
        assert!(verify_op(&PackedAtomicAddOp::new(valid), &ctx).is_ok());
    }

    for pointer in [immutable, shared, local, constant, wrong_pointee, addend] {
        let invalid =
            PackedAtomicAddOp::build(&mut ctx, pointer, addend, PackedAtomicFormatAttr::F16x2);
        assert!(verify_op(&PackedAtomicAddOp::new(invalid), &ctx).is_err());
    }
    let bad_addend = Operation::new(
        &mut ctx,
        PackedAtomicAddOp::get_concrete_op_info(),
        vec![u32_ty.into()],
        vec![global, signless],
        vec![],
        0,
    );
    assert!(verify_op(&PackedAtomicAddOp::new(bad_addend), &ctx).is_err());

    fn set_closed_attributes(ctx: &Context, op: pliron::context::Ptr<Operation>) {
        let packed = PackedAtomicAddOp::new(op);
        packed.set_attr_nvvm_packed_atomic_format(ctx, PackedAtomicFormatAttr::F16x2);
        packed.set_attr_nvvm_packed_atomic_state_space(ctx, PackedAtomicStateSpaceAttr::Global);
        packed.set_attr_nvvm_packed_atomic_ordering(ctx, PackedAtomicOrderingAttr::Relaxed);
        packed.set_attr_nvvm_packed_atomic_scope(ctx, PackedAtomicScopeAttr::Gpu);
        packed.set_attr_nvvm_packed_atomic_rounding(ctx, PackedAtomicRoundingAttr::Rn);
        packed.set_attr_nvvm_packed_atomic_subnormal(ctx, PackedAtomicSubnormalAttr::NoFtz);
        packed.set_attr_nvvm_packed_atomic_atomicity(ctx, PackedAtomicAtomicityAttr::PerElement);
    }

    for result_ty in [signless_i32_ty.into(), signed_i32_ty.into(), u64_ty.into()] {
        let bad_result = Operation::new(
            &mut ctx,
            PackedAtomicAddOp::get_concrete_op_info(),
            vec![result_ty],
            vec![global, addend],
            vec![],
            0,
        );
        set_closed_attributes(&ctx, bad_result);
        assert!(verify_op(&PackedAtomicAddOp::new(bad_result), &ctx).is_err());
    }

    for (results, operands) in [
        (vec![], vec![global, addend]),
        (vec![u32_ty.into(), u32_ty.into()], vec![global, addend]),
        (vec![u32_ty.into()], vec![global]),
        (vec![u32_ty.into()], vec![global, addend, addend]),
    ] {
        let bad_counts = Operation::new(
            &mut ctx,
            PackedAtomicAddOp::get_concrete_op_info(),
            results,
            operands,
            vec![],
            0,
        );
        set_closed_attributes(&ctx, bad_counts);
        assert!(verify_op(&PackedAtomicAddOp::new(bad_counts), &ctx).is_err());
    }

    // A structurally correct operation without the closed semantic attributes
    // must fail instead of inheriting implicit defaults.
    let missing_attributes = Operation::new(
        &mut ctx,
        PackedAtomicAddOp::get_concrete_op_info(),
        vec![u32_ty.into()],
        vec![global, addend],
        vec![],
        0,
    );
    assert!(verify_op(&PackedAtomicAddOp::new(missing_attributes), &ctx).is_err());
}

#[test]
fn handwritten_atomic_carriers_reject_malformed_ir() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let f32_ty = FP32Type::get(&ctx);
    let generic_ptr = MirPtrType::get_generic(&mut ctx, u32_ty.into(), false);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            generic_ptr.into(),
            u32_ty.into(),
            i32_ty.into(),
            f32_ty.into(),
        ],
    );
    let pointer = block.deref(&ctx).get_argument(0);
    let u32_value = block.deref(&ctx).get_argument(1);
    let i32_value = block.deref(&ctx).get_argument(2);
    let f32_value = block.deref(&ctx).get_argument(3);

    assert!(
        NvvmAtomicLoadOp::build(
            &mut ctx,
            pointer,
            u32_ty.into(),
            AtomicOrdering::Acquire,
            AtomicScope::Device,
        )
        .verify(&ctx)
        .is_ok()
    );
    assert!(
        NvvmAtomicStoreOp::build(
            &mut ctx,
            u32_value,
            pointer,
            AtomicOrdering::Release,
            AtomicScope::Device,
        )
        .verify(&ctx)
        .is_ok()
    );
    assert!(
        NvvmAtomicRmwOp::build(
            &mut ctx,
            pointer,
            u32_value,
            u32_ty.into(),
            AtomicRmwKind::Add,
            AtomicOrdering::AcqRel,
            AtomicScope::Device,
        )
        .verify(&ctx)
        .is_ok()
    );
    assert!(
        NvvmAtomicRmwOp::build(
            &mut ctx,
            pointer,
            f32_value,
            f32_ty.into(),
            AtomicRmwKind::FAdd,
            AtomicOrdering::Relaxed,
            AtomicScope::Device,
        )
        .verify(&ctx)
        .is_ok()
    );
    assert!(
        NvvmAtomicCmpxchgOp::build(
            &mut ctx,
            pointer,
            u32_value,
            u32_value,
            u32_ty.into(),
            AtomicOrdering::Relaxed,
            AtomicOrdering::Acquire,
            AtomicScope::System,
        )
        .verify(&ctx)
        .is_ok()
    );

    for invalid in [
        NvvmAtomicLoadOp::build(
            &mut ctx,
            pointer,
            u32_ty.into(),
            AtomicOrdering::Release,
            AtomicScope::Device,
        ),
        NvvmAtomicLoadOp::build(
            &mut ctx,
            u32_value,
            u32_ty.into(),
            AtomicOrdering::Relaxed,
            AtomicScope::Device,
        ),
    ] {
        assert!(invalid.verify(&ctx).is_err());
    }
    assert!(
        NvvmAtomicStoreOp::build(
            &mut ctx,
            u32_value,
            pointer,
            AtomicOrdering::Acquire,
            AtomicScope::Device,
        )
        .verify(&ctx)
        .is_err()
    );
    assert!(
        NvvmAtomicRmwOp::build(
            &mut ctx,
            pointer,
            u32_value,
            u64_ty.into(),
            AtomicRmwKind::Add,
            AtomicOrdering::Relaxed,
            AtomicScope::Device,
        )
        .verify(&ctx)
        .is_err()
    );
    assert!(
        NvvmAtomicRmwOp::build(
            &mut ctx,
            pointer,
            u32_value,
            u32_ty.into(),
            AtomicRmwKind::FAdd,
            AtomicOrdering::Relaxed,
            AtomicScope::Device,
        )
        .verify(&ctx)
        .is_err()
    );
    assert!(
        NvvmAtomicRmwOp::build(
            &mut ctx,
            pointer,
            i32_value,
            i32_ty.into(),
            AtomicRmwKind::UMin,
            AtomicOrdering::Relaxed,
            AtomicScope::Device,
        )
        .verify(&ctx)
        .is_err()
    );
    assert!(
        NvvmAtomicCmpxchgOp::build(
            &mut ctx,
            pointer,
            f32_value,
            f32_value,
            f32_ty.into(),
            AtomicOrdering::SeqCst,
            AtomicOrdering::Relaxed,
            AtomicScope::Device,
        )
        .verify(&ctx)
        .is_err()
    );
    assert!(
        NvvmAtomicCmpxchgOp::build(
            &mut ctx,
            pointer,
            u32_value,
            i32_value,
            u32_ty.into(),
            AtomicOrdering::SeqCst,
            AtomicOrdering::Relaxed,
            AtomicScope::Device,
        )
        .verify(&ctx)
        .is_err()
    );
    assert!(
        NvvmAtomicCmpxchgOp::build(
            &mut ctx,
            pointer,
            u32_value,
            u32_value,
            u32_ty.into(),
            AtomicOrdering::SeqCst,
            AtomicOrdering::Release,
            AtomicScope::Device,
        )
        .verify(&ctx)
        .is_err()
    );

    let missing_attributes = Operation::new(
        &mut ctx,
        NvvmAtomicRmwOp::get_concrete_op_info(),
        vec![u32_ty.into()],
        vec![pointer, u32_value],
        vec![],
        0,
    );
    assert!(
        NvvmAtomicRmwOp::new(missing_attributes)
            .verify(&ctx)
            .is_err()
    );
    let bad_count = Operation::new(
        &mut ctx,
        NvvmAtomicCmpxchgOp::get_concrete_op_info(),
        vec![],
        vec![pointer, u32_value],
        vec![],
        0,
    );
    assert!(NvvmAtomicCmpxchgOp::new(bad_count).verify(&ctx).is_err());
}

#[test]
fn pointer_atomic_values_accept_only_pointer_valid_operations() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);

    // The atomic value itself is a pointer to u64.
    let value_ptr_ty: pliron::r#type::TypeHandle =
        MirPtrType::get_generic(&mut ctx, u64_ty.into(), true).into();

    // The atomic storage therefore points to a pointer value.
    let address_ty: pliron::r#type::TypeHandle =
        MirPtrType::get_generic(&mut ctx, value_ptr_ty, true).into();

    let block = BasicBlock::new(&mut ctx, None, vec![address_ty, value_ptr_ty]);
    let address = block.deref(&ctx).get_argument(0);
    let pointer_value = block.deref(&ctx).get_argument(1);

    assert!(
        NvvmAtomicLoadOp::build(
            &mut ctx,
            address,
            value_ptr_ty,
            AtomicOrdering::Relaxed,
            AtomicScope::System,
        )
        .verify(&ctx)
        .is_ok()
    );

    assert!(
        NvvmAtomicStoreOp::build(
            &mut ctx,
            pointer_value,
            address,
            AtomicOrdering::Relaxed,
            AtomicScope::System,
        )
        .verify(&ctx)
        .is_ok()
    );

    assert!(
        NvvmAtomicRmwOp::build(
            &mut ctx,
            address,
            pointer_value,
            value_ptr_ty,
            AtomicRmwKind::Xchg,
            AtomicOrdering::Relaxed,
            AtomicScope::System,
        )
        .verify(&ctx)
        .is_ok()
    );

    assert!(
        NvvmAtomicCmpxchgOp::build(
            &mut ctx,
            address,
            pointer_value,
            pointer_value,
            value_ptr_ty,
            AtomicOrdering::Relaxed,
            AtomicOrdering::Relaxed,
            AtomicScope::System,
        )
        .verify(&ctx)
        .is_ok()
    );

    // Pointer exchange is valid, but integer/float arithmetic RMWs are not.
    assert!(
        NvvmAtomicRmwOp::build(
            &mut ctx,
            address,
            pointer_value,
            value_ptr_ty,
            AtomicRmwKind::Add,
            AtomicOrdering::Relaxed,
            AtomicScope::System,
        )
        .verify(&ctx)
        .is_err()
    );

    assert!(
        NvvmAtomicRmwOp::build(
            &mut ctx,
            address,
            pointer_value,
            value_ptr_ty,
            AtomicRmwKind::FAdd,
            AtomicOrdering::Relaxed,
            AtomicScope::System,
        )
        .verify(&ctx)
        .is_err()
    );
}

#[test]
fn pointer_atomic_values_reject_shared_address_space() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);

    let shared_value_ptr_ty: pliron::r#type::TypeHandle =
        MirPtrType::get_shared(&mut ctx, u64_ty.into(), true).into();

    let address_ty: pliron::r#type::TypeHandle =
        MirPtrType::get_generic(&mut ctx, shared_value_ptr_ty, true).into();

    let block = BasicBlock::new(&mut ctx, None, vec![address_ty]);
    let address = block.deref(&ctx).get_argument(0);

    assert!(
        NvvmAtomicLoadOp::build(
            &mut ctx,
            address,
            shared_value_ptr_ty,
            AtomicOrdering::Relaxed,
            AtomicScope::System,
        )
        .verify(&ctx)
        .is_err(),
        "shared address-space pointer values must remain unsupported",
    );
}

#[test]
fn atomic_cmpxchg_accepts_exactly_llvm_ordering_pairs() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let pointer_ty = MirPtrType::get_generic(&mut ctx, u32_ty.into(), false);
    let block = BasicBlock::new(&mut ctx, None, vec![pointer_ty.into(), u32_ty.into()]);
    let pointer = block.deref(&ctx).get_argument(0);
    let value = block.deref(&ctx).get_argument(1);

    let orderings = [
        AtomicOrdering::Relaxed,
        AtomicOrdering::Acquire,
        AtomicOrdering::Release,
        AtomicOrdering::AcqRel,
        AtomicOrdering::SeqCst,
    ];
    for success in &orderings {
        for failure in &orderings {
            let expected = matches!(
                failure,
                AtomicOrdering::Relaxed | AtomicOrdering::Acquire | AtomicOrdering::SeqCst
            );
            let actual = NvvmAtomicCmpxchgOp::build(
                &mut ctx,
                pointer,
                value,
                value,
                u32_ty.into(),
                success.clone(),
                failure.clone(),
                AtomicScope::Device,
            )
            .verify(&ctx)
            .is_ok();
            assert_eq!(
                actual, expected,
                "unexpected cmpxchg ordering result for success={success:?}, failure={failure:?}"
            );
        }
    }
}
