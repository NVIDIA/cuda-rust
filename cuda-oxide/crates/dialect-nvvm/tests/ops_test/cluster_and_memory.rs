/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_mir::types::{MirPtrType, address_space};
use dialect_nvvm::ops::{
    ClusterBarrierModeAttr, ClusterBarrierOp, CpAsyncCa4Op, CpAsyncCaZfill4Op,
    CpAsyncMbarrierArriveNoIncOp, CpAsyncMbarrierArriveNoIncSharedOp, CpAsyncMbarrierArriveOp,
    CpAsyncMbarrierArriveSharedOp, CpAsyncWaitGroupOp, MbarrierArriveNoCompleteSharedOp,
    MbarrierArriveSharedOp, MbarrierInitSharedOp, MbarrierInvalSharedOp, MbarrierTestWaitSharedOp,
    ReadPtxSregClusterIdxOp, ReadPtxSregNclusterIdOp, ScalarArithmeticFormatAttr,
    ScalarArithmeticOp, ScalarArithmeticOperationAttr, ScalarArithmeticRoundingAttr,
    ScalarArithmeticSaturationAttr, ScalarArithmeticSubnormalAttr, ScalarConversionOp,
    ScalarConversionRoundingAttr, ScalarConversionSaturationAttr, Tcgen05AllocOp,
    Tcgen05CommitMulticastCg2Op, Tcgen05Ld16x32bx2X1RawOp, Tcgen05Ld16x256bPureOp, Tcgen05MmaF16Op,
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
fn cluster_grid_compatibility_ops_keep_names_and_i32_shape() {
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::common_traits::Verify;
    use pliron::context::Context;
    use pliron::op::Op;
    use pliron::operation::Operation;

    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);
    assert_eq!(
        ReadPtxSregClusterIdxOp::get_opid_static().to_string(),
        "nvvm.read_ptx_sreg_cluster_idx"
    );
    assert_eq!(
        ReadPtxSregNclusterIdOp::get_opid_static().to_string(),
        "nvvm.read_ptx_sreg_nclusterid"
    );

    let i32_type = IntegerType::get(&ctx, 32, Signedness::Signless);
    for op_info in [
        ReadPtxSregClusterIdxOp::get_concrete_op_info(),
        ReadPtxSregNclusterIdOp::get_concrete_op_info(),
    ] {
        let op = Operation::new(&mut ctx, op_info, vec![i32_type.into()], vec![], vec![], 0);
        assert!(op.deref(&ctx).verify(&ctx).is_ok());
    }
}

#[test]
fn mapa_shared_cluster_requires_and_preserves_raw_pointer_kind() {
    use dialect_mir::types::{MirPointerKind, MirPtrType, address_space};
    use dialect_nvvm::ops::MapaSharedClusterOp;
    use pliron::{
        basic_block::BasicBlock,
        builtin::types::{IntegerType, Signedness},
        common_traits::Verify,
        context::Context,
        op::Op,
        operation::Operation,
    };

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let raw_const_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, i32_ty.into(), false, MirPointerKind::RawConst);
    let raw_mut_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, i32_ty.into(), true, MirPointerKind::RawMut);
    let erased_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), false);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            raw_const_ty.into(),
            raw_mut_ty.into(),
            erased_ty.into(),
            i32_ty.into(),
        ],
    );
    let raw_const = block.deref(&ctx).get_argument(0);
    let raw_mut = block.deref(&ctx).get_argument(1);
    let erased = block.deref(&ctx).get_argument(2);
    let rank = block.deref(&ctx).get_argument(3);

    for (source, expected_kind) in [
        (raw_const, MirPointerKind::RawConst),
        (raw_mut, MirPointerKind::RawMut),
    ] {
        let operation = MapaSharedClusterOp::build(&mut ctx, source, rank);
        let result_ty = operation.deref(&ctx).get_result(0).get_type(&ctx);
        let result_ty = result_ty.deref(&ctx);
        let result_ptr = result_ty
            .downcast_ref::<MirPtrType>()
            .expect("mapa result must be a MIR pointer");
        assert_eq!(result_ptr.pointer_kind(), expected_kind);
        assert_eq!(result_ptr.address_space(), address_space::CLUSTER_SHARED);
        assert!(MapaSharedClusterOp::new(operation).verify(&ctx).is_ok());
    }

    let erased_result = MirPtrType::get_with_kind(
        &mut ctx,
        i32_ty.into(),
        true,
        address_space::CLUSTER_SHARED,
        MirPointerKind::Erased,
    );
    let erasing = Operation::new(
        &mut ctx,
        MapaSharedClusterOp::get_concrete_op_info(),
        vec![erased_result.into()],
        vec![raw_mut, rank],
        vec![],
        0,
    );
    assert!(
        MapaSharedClusterOp::new(erasing).verify(&ctx).is_err(),
        "cluster address mapping must preserve the source raw-pointer kind"
    );

    let erased_source = MapaSharedClusterOp::build(&mut ctx, erased, rank);
    assert!(
        MapaSharedClusterOp::new(erased_source)
            .verify(&ctx)
            .is_err(),
        "an Erased data/function carrier cannot enter the raw-pointer intrinsic ABI"
    );

    let invented_unique = MirPtrType::get_with_kind(
        &mut ctx,
        i32_ty.into(),
        true,
        address_space::CLUSTER_SHARED,
        MirPointerKind::UniqueRef,
    );
    let laundering = Operation::new(
        &mut ctx,
        MapaSharedClusterOp::get_concrete_op_info(),
        vec![invented_unique.into()],
        vec![raw_mut, rank],
        vec![],
        0,
    );
    assert!(
        MapaSharedClusterOp::new(laundering).verify(&ctx).is_err(),
        "cluster address mapping must not turn RawMut into UniqueRef"
    );

    let invented_shared = MirPtrType::get_with_kind(
        &mut ctx,
        i32_ty.into(),
        false,
        address_space::CLUSTER_SHARED,
        MirPointerKind::SharedRef,
    );
    let laundering = Operation::new(
        &mut ctx,
        MapaSharedClusterOp::get_concrete_op_info(),
        vec![invented_shared.into()],
        vec![raw_const, rank],
        vec![],
        0,
    );
    assert!(
        MapaSharedClusterOp::new(laundering).verify(&ctx).is_err(),
        "cluster address mapping must not turn RawConst into SharedRef"
    );
}

#[test]
fn generated_cluster_barrier_requires_one_closed_mode_and_no_values() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    for mode in [
        ClusterBarrierModeAttr::Arrive,
        ClusterBarrierModeAttr::ArriveAligned,
        ClusterBarrierModeAttr::ArriveRelaxed,
        ClusterBarrierModeAttr::ArriveRelaxedAligned,
        ClusterBarrierModeAttr::Wait,
        ClusterBarrierModeAttr::WaitAligned,
    ] {
        let op = ClusterBarrierOp::build(&mut ctx, mode);
        assert!(verify_op(&ClusterBarrierOp::new(op), &ctx).is_ok());
    }

    let missing_mode = Operation::new(
        &mut ctx,
        ClusterBarrierOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    assert!(verify_op(&ClusterBarrierOp::new(missing_mode), &ctx).is_err());

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let wrong_shape = Operation::new(
        &mut ctx,
        ClusterBarrierOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );
    ClusterBarrierOp::new(wrong_shape)
        .set_attr_nvvm_cluster_barrier_mode(&ctx, ClusterBarrierModeAttr::Wait);
    assert!(verify_op(&ClusterBarrierOp::new(wrong_shape), &ctx).is_err());
}

#[test]
fn generated_scalar_conversion_accepts_only_reviewed_f32_to_i32_variants() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let f32_ty = FP32Type::get(&ctx);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let block = BasicBlock::new(&mut ctx, None, vec![f32_ty.into(), i32_ty.into()]);
    let f32_value = block.deref(&ctx).get_argument(0);
    let i32_value = block.deref(&ctx).get_argument(1);

    for (rounding, saturation) in [
        (
            ScalarConversionRoundingAttr::NearestAway,
            ScalarConversionSaturationAttr::None,
        ),
        (
            ScalarConversionRoundingAttr::NearestAway,
            ScalarConversionSaturationAttr::Satfinite,
        ),
        (
            ScalarConversionRoundingAttr::NearestEven,
            ScalarConversionSaturationAttr::None,
        ),
        (
            ScalarConversionRoundingAttr::NearestEven,
            ScalarConversionSaturationAttr::Relu,
        ),
        (
            ScalarConversionRoundingAttr::NearestEven,
            ScalarConversionSaturationAttr::Satfinite,
        ),
        (
            ScalarConversionRoundingAttr::NearestEven,
            ScalarConversionSaturationAttr::ReluSatfinite,
        ),
        (
            ScalarConversionRoundingAttr::TowardZero,
            ScalarConversionSaturationAttr::None,
        ),
        (
            ScalarConversionRoundingAttr::TowardZero,
            ScalarConversionSaturationAttr::Relu,
        ),
        (
            ScalarConversionRoundingAttr::TowardZero,
            ScalarConversionSaturationAttr::Satfinite,
        ),
        (
            ScalarConversionRoundingAttr::TowardZero,
            ScalarConversionSaturationAttr::ReluSatfinite,
        ),
    ] {
        let op = ScalarConversionOp::build(&mut ctx, f32_value, rounding, saturation);
        assert!(verify_op(&ScalarConversionOp::new(op), &ctx).is_ok());
    }

    let invalid_variant = ScalarConversionOp::build(
        &mut ctx,
        f32_value,
        ScalarConversionRoundingAttr::NearestAway,
        ScalarConversionSaturationAttr::Relu,
    );
    assert!(verify_op(&ScalarConversionOp::new(invalid_variant), &ctx).is_err());

    let wrong_operand = Operation::new(
        &mut ctx,
        ScalarConversionOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![i32_value],
        vec![],
        0,
    );
    let wrong_operand = ScalarConversionOp::new(wrong_operand);
    wrong_operand
        .set_attr_nvvm_scalar_conversion_rounding(&ctx, ScalarConversionRoundingAttr::NearestEven);
    wrong_operand
        .set_attr_nvvm_scalar_conversion_saturation(&ctx, ScalarConversionSaturationAttr::None);
    assert!(verify_op(&wrong_operand, &ctx).is_err());

    let wrong_result = Operation::new(
        &mut ctx,
        ScalarConversionOp::get_concrete_op_info(),
        vec![i64_ty.into()],
        vec![f32_value],
        vec![],
        0,
    );
    let wrong_result = ScalarConversionOp::new(wrong_result);
    wrong_result
        .set_attr_nvvm_scalar_conversion_rounding(&ctx, ScalarConversionRoundingAttr::TowardZero);
    wrong_result
        .set_attr_nvvm_scalar_conversion_saturation(&ctx, ScalarConversionSaturationAttr::None);
    assert!(verify_op(&wrong_result, &ctx).is_err());
}

#[test]
fn generated_scalar_arithmetic_accepts_only_admitted_shapes_and_types() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let f32_ty = FP32Type::get(&ctx);
    let f64_ty = FP64Type::get(&ctx);
    let block = BasicBlock::new(&mut ctx, None, vec![f32_ty.into(), f64_ty.into()]);
    let f32_value = block.deref(&ctx).get_argument(0);
    let f64_value = block.deref(&ctx).get_argument(1);

    let valid_f32 = ScalarArithmeticOp::build(
        &mut ctx,
        vec![f32_value, f32_value],
        ScalarArithmeticFormatAttr::F32,
        ScalarArithmeticOperationAttr::Mul,
        ScalarArithmeticRoundingAttr::Rn,
        ScalarArithmeticSubnormalAttr::Preserve,
        ScalarArithmeticSaturationAttr::None,
    );
    assert!(verify_op(&ScalarArithmeticOp::new(valid_f32), &ctx).is_ok());

    let valid_f64 = ScalarArithmeticOp::build(
        &mut ctx,
        vec![f64_value, f64_value, f64_value],
        ScalarArithmeticFormatAttr::F64,
        ScalarArithmeticOperationAttr::Fma,
        ScalarArithmeticRoundingAttr::Rz,
        ScalarArithmeticSubnormalAttr::Preserve,
        ScalarArithmeticSaturationAttr::None,
    );
    assert!(verify_op(&ScalarArithmeticOp::new(valid_f64), &ctx).is_ok());

    let valid_add = ScalarArithmeticOp::build(
        &mut ctx,
        vec![f32_value, f32_value],
        ScalarArithmeticFormatAttr::F32,
        ScalarArithmeticOperationAttr::Add,
        ScalarArithmeticRoundingAttr::Rp,
        ScalarArithmeticSubnormalAttr::Ftz,
        ScalarArithmeticSaturationAttr::Sat,
    );
    assert!(verify_op(&ScalarArithmeticOp::new(valid_add), &ctx).is_ok());

    let f64_ftz = ScalarArithmeticOp::build(
        &mut ctx,
        vec![f64_value, f64_value],
        ScalarArithmeticFormatAttr::F64,
        ScalarArithmeticOperationAttr::Mul,
        ScalarArithmeticRoundingAttr::Rn,
        ScalarArithmeticSubnormalAttr::Ftz,
        ScalarArithmeticSaturationAttr::None,
    );
    assert!(verify_op(&ScalarArithmeticOp::new(f64_ftz), &ctx).is_err());

    let wrong_arity = ScalarArithmeticOp::build(
        &mut ctx,
        vec![f32_value, f32_value],
        ScalarArithmeticFormatAttr::F32,
        ScalarArithmeticOperationAttr::Fma,
        ScalarArithmeticRoundingAttr::Rn,
        ScalarArithmeticSubnormalAttr::Preserve,
        ScalarArithmeticSaturationAttr::None,
    );
    assert!(verify_op(&ScalarArithmeticOp::new(wrong_arity), &ctx).is_err());

    let wrong_type = ScalarArithmeticOp::build(
        &mut ctx,
        vec![f32_value, f64_value],
        ScalarArithmeticFormatAttr::F32,
        ScalarArithmeticOperationAttr::Mul,
        ScalarArithmeticRoundingAttr::Rn,
        ScalarArithmeticSubnormalAttr::Preserve,
        ScalarArithmeticSaturationAttr::None,
    );
    assert!(verify_op(&ScalarArithmeticOp::new(wrong_type), &ctx).is_err());

    let missing_attrs = Operation::new(
        &mut ctx,
        ScalarArithmeticOp::get_concrete_op_info(),
        vec![f32_ty.into()],
        vec![f32_value, f32_value],
        vec![],
        0,
    );
    assert!(verify_op(&ScalarArithmeticOp::new(missing_attrs), &ctx).is_err());
}

#[test]
fn test_generated_cp_async_accepts_pointer_shapes_and_both_constant_kinds() {
    use dialect_mir::ops::MirConstantOp;
    use pliron::builtin::{attributes::IntegerAttr, ops::ConstantOp};
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let u8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let dst_ty = MirPtrType::get_generic(&mut ctx, u8_ty.into(), false);
    let src_ty = MirPtrType::get(&mut ctx, u8_ty.into(), true, address_space::GLOBAL);
    let wrong_dst_ty = MirPtrType::get(&mut ctx, u8_ty.into(), true, address_space::GLOBAL);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            dst_ty.into(),
            src_ty.into(),
            wrong_dst_ty.into(),
            u32_ty.into(),
        ],
    );
    let dst = block.deref(&ctx).get_argument(0);
    let src = block.deref(&ctx).get_argument(1);
    let wrong_dst = block.deref(&ctx).get_argument(2);
    let dynamic = block.deref(&ctx).get_argument(3);

    let copy = CpAsyncCa4Op::build(&mut ctx, dst, src);
    assert!(verify_op(&CpAsyncCa4Op::new(copy), &ctx).is_ok());
    let zfill = CpAsyncCaZfill4Op::build(&mut ctx, dst, src, dynamic);
    assert!(verify_op(&CpAsyncCaZfill4Op::new(zfill), &ctx).is_ok());
    let wrong_space = CpAsyncCa4Op::build(&mut ctx, wrong_dst, src);
    assert!(verify_op(&CpAsyncCa4Op::new(wrong_space), &ctx).is_err());

    let value = IntegerAttr::new(u32_ty, APInt::from_u32(0, NonZeroUsize::new(32).unwrap()));
    let builtin = ConstantOp::new(&mut ctx, Box::new(value.clone()));
    let builtin_value = builtin.get_operation().deref(&ctx).get_result(0);
    let builtin_wait = CpAsyncWaitGroupOp::build(&mut ctx, builtin_value);
    assert!(verify_op(&CpAsyncWaitGroupOp::new(builtin_wait), &ctx).is_ok());

    let mir_constant = Operation::new(
        &mut ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![u32_ty.into()],
        vec![],
        vec![],
        0,
    );
    MirConstantOp::new(mir_constant).set_attr_value(&ctx, value);
    let mir_value = mir_constant.deref(&ctx).get_result(0);
    let mir_wait = CpAsyncWaitGroupOp::build(&mut ctx, mir_value);
    assert!(verify_op(&CpAsyncWaitGroupOp::new(mir_wait), &ctx).is_ok());

    let dynamic_wait = CpAsyncWaitGroupOp::build(&mut ctx, dynamic);
    assert!(verify_op(&CpAsyncWaitGroupOp::new(dynamic_wait), &ctx).is_err());
}

#[test]
fn generated_tcgen05_verifies_carriers_and_half_split_constants() {
    use dialect_mir::ops::MirConstantOp;
    use pliron::builtin::{attributes::IntegerAttr, ops::ConstantOp};
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);
    let i16_ty = IntegerType::get(&ctx, 16, Signedness::Signless);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let f32_ty = FP32Type::get(&ctx);
    let pointer_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), true);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            pointer_ty.into(),
            i1_ty.into(),
            i16_ty.into(),
            i32_ty.into(),
            i64_ty.into(),
            f32_ty.into(),
        ],
    );
    let pointer = block.deref(&ctx).get_argument(0);
    let predicate = block.deref(&ctx).get_argument(1);
    let mask = block.deref(&ctx).get_argument(2);
    let address = block.deref(&ctx).get_argument(3);
    let dynamic_offset = block.deref(&ctx).get_argument(4);

    let offset_attr = IntegerAttr::new(i64_ty, APInt::from_i64(16, NonZeroUsize::new(64).unwrap()));
    let builtin_offset = ConstantOp::new(&mut ctx, Box::new(offset_attr.clone()))
        .get_operation()
        .deref(&ctx)
        .get_result(0);
    let mir_constant = Operation::new(
        &mut ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![i64_ty.into()],
        vec![],
        vec![],
        0,
    );
    MirConstantOp::new(mir_constant).set_attr_value(&ctx, offset_attr);
    let mir_offset = mir_constant.deref(&ctx).get_result(0);

    for offset in [builtin_offset, mir_offset] {
        let load = Operation::new(
            &mut ctx,
            Tcgen05Ld16x32bx2X1RawOp::get_concrete_op_info(),
            vec![i32_ty.into()],
            vec![address, offset],
            vec![],
            0,
        );
        assert!(verify_op(&Tcgen05Ld16x32bx2X1RawOp::new(load), &ctx).is_ok());
    }

    let dynamic = Operation::new(
        &mut ctx,
        Tcgen05Ld16x32bx2X1RawOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![address, dynamic_offset],
        vec![],
        0,
    );
    assert!(verify_op(&Tcgen05Ld16x32bx2X1RawOp::new(dynamic), &ctx).is_err());

    let wrong_offset_type = Operation::new(
        &mut ctx,
        Tcgen05Ld16x32bx2X1RawOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![address, address],
        vec![],
        0,
    );
    assert!(verify_op(&Tcgen05Ld16x32bx2X1RawOp::new(wrong_offset_type), &ctx).is_err());

    let wrong_result_type = Operation::new(
        &mut ctx,
        Tcgen05Ld16x32bx2X1RawOp::get_concrete_op_info(),
        vec![i64_ty.into()],
        vec![address, builtin_offset],
        vec![],
        0,
    );
    assert!(verify_op(&Tcgen05Ld16x32bx2X1RawOp::new(wrong_result_type), &ctx).is_err());

    let alloc = Operation::new(
        &mut ctx,
        Tcgen05AllocOp::get_concrete_op_info(),
        vec![],
        vec![pointer, address],
        vec![],
        0,
    );
    assert!(verify_op(&Tcgen05AllocOp::new(alloc), &ctx).is_ok());

    let multicast = Operation::new(
        &mut ctx,
        Tcgen05CommitMulticastCg2Op::get_concrete_op_info(),
        vec![],
        vec![pointer, mask],
        vec![],
        0,
    );
    assert!(verify_op(&Tcgen05CommitMulticastCg2Op::new(multicast), &ctx).is_ok());

    let mma = Operation::new(
        &mut ctx,
        Tcgen05MmaF16Op::get_concrete_op_info(),
        vec![],
        vec![address, dynamic_offset, dynamic_offset, address, predicate],
        vec![],
        0,
    );
    assert!(verify_op(&Tcgen05MmaF16Op::new(mma), &ctx).is_ok());

    let pure_load = Operation::new(
        &mut ctx,
        Tcgen05Ld16x256bPureOp::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        vec![address],
        vec![],
        0,
    );
    assert!(verify_op(&Tcgen05Ld16x256bPureOp::new(pure_load), &ctx).is_ok());
}

#[test]
fn generated_cp_async_mbarrier_requires_mutable_generic_or_shared_u64() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let generic_u64 = MirPtrType::get_generic(&mut ctx, u64_ty.into(), true);
    let shared_u64 = MirPtrType::get_shared(&mut ctx, u64_ty.into(), true);
    let global_u64 = MirPtrType::get_global(&mut ctx, u64_ty.into(), true);
    let immutable_u64 = MirPtrType::get_generic(&mut ctx, u64_ty.into(), false);
    let generic_u32 = MirPtrType::get_generic(&mut ctx, u32_ty.into(), true);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            generic_u64.into(),
            shared_u64.into(),
            global_u64.into(),
            immutable_u64.into(),
            generic_u32.into(),
            u64_ty.into(),
        ],
    );
    let generic = block.deref(&ctx).get_argument(0);
    let shared = block.deref(&ctx).get_argument(1);
    let global = block.deref(&ctx).get_argument(2);
    let immutable = block.deref(&ctx).get_argument(3);
    let wrong_pointee = block.deref(&ctx).get_argument(4);
    let scalar = block.deref(&ctx).get_argument(5);

    macro_rules! check_bridge {
        ($op:ty) => {{
            for barrier in [generic, shared] {
                let valid = <$op>::build(&mut ctx, barrier);
                assert!(verify_op(&<$op>::new(valid), &ctx).is_ok());
            }
            for barrier in [global, immutable, wrong_pointee, scalar] {
                let invalid = <$op>::build(&mut ctx, barrier);
                assert!(verify_op(&<$op>::new(invalid), &ctx).is_err());
            }
            let wrong_shape = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![u64_ty.into()],
                vec![generic],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(wrong_shape), &ctx).is_err());
        }};
    }

    check_bridge!(CpAsyncMbarrierArriveOp);
    check_bridge!(CpAsyncMbarrierArriveSharedOp);
    check_bridge!(CpAsyncMbarrierArriveNoIncOp);
    check_bridge!(CpAsyncMbarrierArriveNoIncSharedOp);
}

#[test]
fn generated_mbarrier_builders_and_verifiers_are_closed_over_their_shapes() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let u1_ty = IntegerType::get(&ctx, 1, Signedness::Unsigned);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let generic_ptr = MirPtrType::get_generic(&mut ctx, u64_ty.into(), false);
    let shared_ptr = MirPtrType::get_shared(&mut ctx, u64_ty.into(), false);
    let global_ptr = MirPtrType::get_global(&mut ctx, u64_ty.into(), false);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            generic_ptr.into(),
            shared_ptr.into(),
            global_ptr.into(),
            i32_ty.into(),
            u32_ty.into(),
            u64_ty.into(),
        ],
    );
    let generic = block.deref(&ctx).get_argument(0);
    let shared = block.deref(&ctx).get_argument(1);
    let global = block.deref(&ctx).get_argument(2);
    let signless_i32 = block.deref(&ctx).get_argument(3);
    let count = block.deref(&ctx).get_argument(4);
    let token = block.deref(&ctx).get_argument(5);

    for barrier in [generic, shared] {
        let init = MbarrierInitSharedOp::build(&mut ctx, barrier, count);
        assert!(MbarrierInitSharedOp::new(init).verify(&ctx).is_ok());
        let arrive = MbarrierArriveSharedOp::build(&mut ctx, barrier);
        assert!(MbarrierArriveSharedOp::new(arrive).verify(&ctx).is_ok());
        let arrive_no_complete = MbarrierArriveNoCompleteSharedOp::build(&mut ctx, barrier, count);
        assert!(
            MbarrierArriveNoCompleteSharedOp::new(arrive_no_complete)
                .verify(&ctx)
                .is_ok()
        );
        let test_wait = MbarrierTestWaitSharedOp::build(&mut ctx, barrier, token);
        assert!(
            MbarrierTestWaitSharedOp::new(test_wait)
                .verify(&ctx)
                .is_ok()
        );
        let inval = MbarrierInvalSharedOp::build(&mut ctx, barrier);
        assert!(MbarrierInvalSharedOp::new(inval).verify(&ctx).is_ok());
    }

    for barrier in [token, global] {
        let inval = MbarrierInvalSharedOp::build(&mut ctx, barrier);
        assert!(MbarrierInvalSharedOp::new(inval).verify(&ctx).is_err());
    }

    let bad_count = MbarrierInitSharedOp::build(&mut ctx, shared, signless_i32);
    assert!(MbarrierInitSharedOp::new(bad_count).verify(&ctx).is_err());
    let bad_arrive_result = Operation::new(
        &mut ctx,
        MbarrierArriveSharedOp::get_concrete_op_info(),
        vec![u32_ty.into()],
        vec![shared],
        vec![],
        0,
    );
    assert!(
        MbarrierArriveSharedOp::new(bad_arrive_result)
            .verify(&ctx)
            .is_err()
    );
    let bad_no_complete_count =
        MbarrierArriveNoCompleteSharedOp::build(&mut ctx, shared, signless_i32);
    assert!(
        MbarrierArriveNoCompleteSharedOp::new(bad_no_complete_count)
            .verify(&ctx)
            .is_err()
    );
    let bad_no_complete_result = Operation::new(
        &mut ctx,
        MbarrierArriveNoCompleteSharedOp::get_concrete_op_info(),
        vec![u32_ty.into()],
        vec![shared, count],
        vec![],
        0,
    );
    assert!(
        MbarrierArriveNoCompleteSharedOp::new(bad_no_complete_result)
            .verify(&ctx)
            .is_err()
    );
    let bad_token = MbarrierTestWaitSharedOp::build(&mut ctx, shared, count);
    assert!(
        MbarrierTestWaitSharedOp::new(bad_token)
            .verify(&ctx)
            .is_err()
    );
    let bad_predicate = Operation::new(
        &mut ctx,
        MbarrierTestWaitSharedOp::get_concrete_op_info(),
        vec![u1_ty.into()],
        vec![shared, token],
        vec![],
        0,
    );
    assert!(
        MbarrierTestWaitSharedOp::new(bad_predicate)
            .verify(&ctx)
            .is_err()
    );

    let missing_operands = Operation::new(
        &mut ctx,
        MbarrierInitSharedOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    assert!(
        MbarrierInitSharedOp::new(missing_operands)
            .verify(&ctx)
            .is_err()
    );
}
