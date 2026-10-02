/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compiler-only regressions for the TCGen05 and cluster/TMA translation boundary.
//! This fixture is never launched; collective participation is the example's
//! runtime contract, which requires separate testing on an SM100 device.

use super::{
    CUTLASS_COMPILER_LIBRARY_SHA256, CUTLASS_PRECOMPILED_PIPELINE, extract_kernels_binary,
};
use crate::cutlass_compiler::CutlassCompilerLibrary;
use cuda_oxide_mlir_export::{CutlassFullCuteMlir22, MlirConsumerProfile};
use dialect_mir::{
    attributes::MirCastKindAttr,
    ops::{MirCastOp, MirConstantOp, MirFuncOp, MirReturnOp, MirStoreOp},
    types::{MirPointerKind, MirPtrType, address_space},
};
use dialect_nvvm::ops::{
    BarrierCtaSyncCountOp, ClusterBarrierModeAttr, ClusterBarrierOp, CpAsyncBulkCommitGroupOp,
    CpAsyncBulkTensorG2sTile2dMulticastCg2Op, CpAsyncBulkTensorS2gTile2dOp,
    CpAsyncBulkWaitGroupReadOp, CvtaGenericToSharedOffsetOp, ElectSyncOp,
    FenceMbarrierInitReleaseClusterOp, FenceProxyAsyncSharedCtaOp, MapaSharedClusterOp,
    MbarrierArriveClusterOp, MbarrierArriveExpectTxClusterOp, MbarrierInitSharedOp,
    MbarrierTryWaitParityClusterOp, PrefetchTensorMapOp, Tcgen05AllocCg2Op,
    Tcgen05CommitMulticastCg2Op, Tcgen05DeallocCg2Op, Tcgen05FenceAfterThreadSyncOp,
    Tcgen05FenceBeforeThreadSyncOp, Tcgen05Ld32x32bX32RawOp, Tcgen05LoadWaitOp,
    Tcgen05MmaCollectorAAttr, Tcgen05MmaCtaGroupAttr, Tcgen05MmaFormAttr, Tcgen05MmaKindAttr,
    Tcgen05MmaOp, Tcgen05RelinquishAllocPermitCg2Op,
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{IntegerAttr, StringAttr, TypeAttr},
        op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
        ops::ModuleOp,
        types::{FunctionType, IntegerType, Signedness},
    },
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
    r#type::{TypeHandle, TypedHandle},
    utils::apint::APInt,
    value::Value,
};
use std::num::NonZeroUsize;

fn append<O: Op>(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    operands: Vec<Value>,
    results: Vec<TypeHandle>,
    marker: Option<&str>,
) -> Ptr<Operation> {
    let op = Operation::new(ctx, O::get_concrete_op_info(), results, operands, vec![], 0);
    if let Some(marker) = marker {
        op.deref_mut(ctx).attributes.set(
            "cuda_oxide_intrinsic_marker".try_into().unwrap(),
            StringAttr::new(marker.into()),
        );
    }
    op.insert_at_back(block, ctx);
    op
}

/// Run explicitly after installing the fingerprint-pinned CUTLASS compiler:
///
/// ```text
/// CUDA_OXIDE_TEST_CUTLASS_COMPILER=/absolute/path/libCutlassCompiler.so \
///   cargo test -p cuda-oxide-codegen cutlass_tcgen05_pack_compiles -- --ignored
/// ```
#[test]
#[ignore = "requires the installed official CUTLASS 4.7 compiler; does not require a GPU"]
fn cutlass_tcgen05_pack_compiles() {
    let compiler_path = std::env::var_os("CUDA_OXIDE_TEST_CUTLASS_COMPILER")
        .expect("set CUDA_OXIDE_TEST_CUTLASS_COMPILER to the pinned compiler library");
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    dialect_cute::register(&mut ctx);
    let module = ModuleOp::new(&mut ctx, "tcgen05_smoke".try_into().unwrap());
    let i1: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Unsigned).into();
    let i16: TypeHandle = IntegerType::get(&ctx, 16, Signedness::Unsigned).into();
    let i32: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
    let i64: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
    let pointer: TypeHandle = MirPtrType::get(&mut ctx, i32, true, address_space::GENERIC).into();
    let arguments = vec![pointer, i32, i64, i64, i32, i32, i1, i16, pointer];
    let function_type = FunctionType::get(&ctx, arguments.clone(), vec![]);
    let function_op = Operation::new(
        &mut ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let function = MirFuncOp::new(&mut ctx, function_op, TypeAttr::new(function_type.into()));
    function.set_symbol_name(&mut ctx, "tcgen05_probe".try_into().unwrap());
    function_op.deref_mut(&ctx).attributes.set(
        "gpu_kernel".try_into().unwrap(),
        StringAttr::new("true".into()),
    );
    module.append_operation(&mut ctx, function_op, 0);
    let block = BasicBlock::new(&mut ctx, None, arguments);
    block.insert_at_back(function_op.deref(&ctx).get_region(0), &ctx);
    let a = block.deref(&ctx).arguments().collect::<Vec<_>>();
    append::<CvtaGenericToSharedOffsetOp>(&mut ctx, block, vec![a[0]], vec![i64], None);
    append::<Tcgen05AllocCg2Op>(&mut ctx, block, vec![a[0], a[5]], vec![], Some("v1:i0359"));
    append::<Tcgen05RelinquishAllocPermitCg2Op>(&mut ctx, block, vec![], vec![], Some("v1:i0361"));
    for collector in [
        Tcgen05MmaCollectorAAttr::Fill,
        Tcgen05MmaCollectorAAttr::LastUse,
    ] {
        let op = append::<Tcgen05MmaOp>(
            &mut ctx,
            block,
            vec![a[1], a[2], a[3], a[4], a[6]],
            vec![],
            Some("v1:i0763"),
        );
        let mma = Tcgen05MmaOp::new(op);
        mma.set_attr_nvvm_tcgen05_mma_form(&ctx, Tcgen05MmaFormAttr::Shared);
        mma.set_attr_nvvm_tcgen05_mma_kind(&ctx, Tcgen05MmaKindAttr::F16);
        mma.set_attr_nvvm_tcgen05_mma_cta_group(&ctx, Tcgen05MmaCtaGroupAttr::Cg2);
        mma.set_attr_nvvm_tcgen05_mma_collector_a(&ctx, collector);
    }
    append::<Tcgen05CommitMulticastCg2Op>(
        &mut ctx,
        block,
        vec![a[0], a[7]],
        vec![],
        Some("v1:i0365"),
    );
    let load = append::<Tcgen05Ld32x32bX32RawOp>(
        &mut ctx,
        block,
        vec![a[1]],
        vec![i32; 32],
        Some("v1:i0664"),
    );
    append::<Tcgen05LoadWaitOp>(&mut ctx, block, vec![], vec![], Some("v1:i0357"));
    let last_register = load.deref(&ctx).get_result(31);
    append::<MirStoreOp>(&mut ctx, block, vec![a[8], last_register], vec![], None);
    append::<Tcgen05FenceBeforeThreadSyncOp>(&mut ctx, block, vec![], vec![], Some("v1:i0346"));
    append::<Tcgen05FenceAfterThreadSyncOp>(&mut ctx, block, vec![], vec![], Some("v1:i0347"));
    append::<Tcgen05DeallocCg2Op>(&mut ctx, block, vec![a[1], a[5]], vec![], Some("v1:i0360"));
    append::<MirReturnOp>(&mut ctx, block, vec![], vec![], None);

    let profile = CutlassFullCuteMlir22::new("sm_100a").unwrap();
    let target = profile.translate_module(&ctx, &module).unwrap();
    let mlir = pliron_mlir_export::render_module(&target);
    let library =
        CutlassCompilerLibrary::load(compiler_path, CUTLASS_COMPILER_LIBRARY_SHA256).unwrap();
    let object = library
        .compile_precompiled_mlir_to_object(&mlir, "sm_100a", CUTLASS_PRECOMPILED_PIPELINE)
        .unwrap_or_else(|error| {
            panic!("TCGen05 translated MLIR failed compilation: {error}\n{mlir}")
        });
    let image = extract_kernels_binary(&object.elf).unwrap();
    assert!(
        image.len() > 64,
        "compiler did not produce a nonempty device image"
    );
}

/// Compile the actual exported cluster/TMA mappings against the pinned consumer.
/// In particular, CUTLASS 4.7's two G2S intrinsic paths require different pointer
/// address spaces; MLIR parsing alone cannot catch the LLVM intrinsic ABI error.
#[test]
#[ignore = "requires the installed official CUTLASS 4.7 compiler; does not require a GPU"]
#[allow(clippy::disallowed_methods)] // Synthetic Rust raw pointer provenance for mapa's source contract.
fn cutlass_cluster_tma_pack_compiles() {
    let compiler_path = std::env::var_os("CUDA_OXIDE_TEST_CUTLASS_COMPILER")
        .expect("set CUDA_OXIDE_TEST_CUTLASS_COMPILER to the pinned compiler library");
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    dialect_cute::register(&mut ctx);
    let module = ModuleOp::new(&mut ctx, "cluster_tma_smoke".try_into().unwrap());
    let i1: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
    let u16 = IntegerType::get(&ctx, 16, Signedness::Unsigned);
    let u32 = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let u64 = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let pointer: TypeHandle = MirPtrType::get_with_kind(
        &mut ctx,
        u32.into(),
        true,
        address_space::GENERIC,
        MirPointerKind::RawMut,
    )
    .into();
    let predicate_pointer: TypeHandle = MirPtrType::get_with_kind(
        &mut ctx,
        i1,
        true,
        address_space::GENERIC,
        MirPointerKind::RawMut,
    )
    .into();
    // Shared source, tensor descriptor, local barrier, lane result, wait result.
    let arguments = vec![pointer, pointer, pointer, pointer, predicate_pointer];
    let function_type = FunctionType::get(&ctx, arguments.clone(), vec![]);
    let function_op = Operation::new(
        &mut ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let function = MirFuncOp::new(&mut ctx, function_op, TypeAttr::new(function_type.into()));
    function.set_symbol_name(&mut ctx, "cluster_tma_probe".try_into().unwrap());
    function_op.deref_mut(&ctx).attributes.set(
        "gpu_kernel".try_into().unwrap(),
        StringAttr::new("true".into()),
    );
    module.append_operation(&mut ctx, function_op, 0);
    let block = BasicBlock::new(&mut ctx, None, arguments);
    block.insert_at_back(function_op.deref(&ctx).get_region(0), &ctx);
    let a = block.deref(&ctx).arguments().collect::<Vec<_>>();

    let zero = integer_constant(&mut ctx, block, u32, 0, 32);
    let one = integer_constant(&mut ctx, block, u32, 1, 32);
    let threads = integer_constant(&mut ctx, block, u32, 128, 32);
    let bytes = integer_constant(&mut ctx, block, u32, 77824, 32);
    let warp_mask = integer_constant(&mut ctx, block, u32, u32::MAX.into(), 32);
    let multicast = integer_constant(&mut ctx, block, u16, 3, 16);
    let cache_hint = integer_constant(&mut ctx, block, u64, 0, 64);

    append::<PrefetchTensorMapOp>(&mut ctx, block, vec![a[1]], vec![], Some("v1:i0887"));
    append::<MbarrierInitSharedOp>(&mut ctx, block, vec![a[2], one], vec![], Some("v1:i0097"));
    append::<FenceMbarrierInitReleaseClusterOp>(&mut ctx, block, vec![], vec![], Some("v1:i0313"));
    for (mode, marker) in [
        (ClusterBarrierModeAttr::ArriveRelaxed, "v1:i0279"),
        (ClusterBarrierModeAttr::Wait, "v1:i0281"),
    ] {
        let op = append::<ClusterBarrierOp>(&mut ctx, block, vec![], vec![], Some(marker));
        ClusterBarrierOp::new(op).set_attr_nvvm_cluster_barrier_mode(&ctx, mode);
    }
    append::<MbarrierArriveExpectTxClusterOp>(
        &mut ctx,
        block,
        vec![a[2], bytes],
        vec![u64.into()],
        Some("v1:i0307"),
    );
    let mapped = MapaSharedClusterOp::build(&mut ctx, a[2], zero);
    mapped.deref_mut(&ctx).attributes.set(
        "cuda_oxide_intrinsic_marker".try_into().unwrap(),
        StringAttr::new("v1:i0320".into()),
    );
    mapped.insert_at_back(block, &ctx);
    let remote_barrier = mapped.deref(&ctx).get_result(0);
    // Native semantic order: dst, barrier, descriptor, x, y, mask, cache hint.
    append::<CpAsyncBulkTensorG2sTile2dMulticastCg2Op>(
        &mut ctx,
        block,
        vec![
            a[0],
            remote_barrier,
            a[1],
            zero,
            zero,
            multicast,
            cache_hint,
        ],
        vec![],
        Some("v1:i0331"),
    );
    let waited = append::<MbarrierTryWaitParityClusterOp>(
        &mut ctx,
        block,
        vec![a[2], zero],
        vec![i1],
        Some("v1:i0311"),
    );
    let ready = waited.deref(&ctx).get_result(0);
    append::<MirStoreOp>(&mut ctx, block, vec![a[4], ready], vec![], None);
    let election = append::<ElectSyncOp>(
        &mut ctx,
        block,
        vec![warp_mask],
        vec![u32.into(), i1],
        Some("v1:i0367"),
    );
    let lane = election.deref(&ctx).get_result(0);
    append::<MirStoreOp>(&mut ctx, block, vec![a[3], lane], vec![], None);
    append::<BarrierCtaSyncCountOp>(
        &mut ctx,
        block,
        vec![one, threads],
        vec![],
        Some("v1:i0883"),
    );
    append::<FenceProxyAsyncSharedCtaOp>(&mut ctx, block, vec![], vec![], Some("v1:i0312"));
    append::<CpAsyncBulkTensorS2gTile2dOp>(
        &mut ctx,
        block,
        vec![a[0], a[1], zero, zero],
        vec![],
        Some("v1:i0336"),
    );
    append::<CpAsyncBulkCommitGroupOp>(&mut ctx, block, vec![], vec![], Some("v1:i0340"));
    append::<CpAsyncBulkWaitGroupReadOp>(&mut ctx, block, vec![zero], vec![], Some("v1:i0342"));
    let remote_address = append::<MirCastOp>(
        &mut ctx,
        block,
        vec![remote_barrier],
        vec![u64.into()],
        None,
    );
    MirCastOp::new(remote_address).set_attr_cast_kind(&ctx, MirCastKindAttr::PointerExposeAddress);
    let address = remote_address.deref(&ctx).get_result(0);
    append::<MbarrierArriveClusterOp>(&mut ctx, block, vec![address], vec![], Some("v1:i0308"));
    append::<MirReturnOp>(&mut ctx, block, vec![], vec![], None);

    let profile = CutlassFullCuteMlir22::new("sm_100a").unwrap();
    let target = profile.translate_module(&ctx, &module).unwrap();
    let mlir = pliron_mlir_export::render_module(&target);
    let library =
        CutlassCompilerLibrary::load(compiler_path, CUTLASS_COMPILER_LIBRARY_SHA256).unwrap();
    let object = library
        .compile_precompiled_mlir_to_object(&mlir, "sm_100a", CUTLASS_PRECOMPILED_PIPELINE)
        .unwrap_or_else(|error| {
            panic!("Cluster/TMA translated MLIR failed compilation: {error}\n{mlir}")
        });
    assert!(
        extract_kernels_binary(&object.elf).unwrap().len() > 64,
        "compiler did not produce a nonempty device image"
    );
}

fn integer_constant(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    ty: TypedHandle<IntegerType>,
    value: u64,
    bits: usize,
) -> Value {
    let op = append::<MirConstantOp>(ctx, block, vec![], vec![ty.into()], None);
    MirConstantOp::new(op).set_attr_value(
        ctx,
        IntegerAttr::new(ty, APInt::from_u64(value, NonZeroUsize::new(bits).unwrap())),
    );
    op.deref(ctx).get_result(0)
}
