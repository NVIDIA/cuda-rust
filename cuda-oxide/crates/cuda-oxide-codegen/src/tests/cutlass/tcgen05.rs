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
    FenceAcquireSharedClusterClusterOp, FenceMbarrierInitReleaseClusterOp,
    FenceProxyAsyncSharedCtaOp, MapaSharedClusterOp, MbarrierArriveClusterOp,
    MbarrierArriveClusterRelaxedOp, MbarrierArriveExpectTxClusterOp, MbarrierInitSharedOp,
    MbarrierTryWaitParityClusterOp, MbarrierTryWaitParityClusterRelaxedOp, PrefetchTensorMapOp,
    Tcgen05AllocCg2Op, Tcgen05CommitMulticastCg2Op, Tcgen05DeallocCg2Op,
    Tcgen05FenceAfterThreadSyncOp, Tcgen05FenceBeforeThreadSyncOp, Tcgen05Ld32x32bX32RawOp,
    Tcgen05LoadWaitOp, Tcgen05MmaCollectorAAttr, Tcgen05MmaCtaGroupAttr, Tcgen05MmaFormAttr,
    Tcgen05MmaKindAttr, Tcgen05MmaOp, Tcgen05RelinquishAllocPermitCg2Op,
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
    let mlir = cuda_oxide_mlir_export::render_module(&target).unwrap();
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

#[test]
#[ignore = "requires the installed official CUTLASS 4.7 compiler; does not require a GPU"]
fn cutlass_attention_tmem_pack_compiles() {
    // Group mode must be consistent throughout a kernel. Probe separate modules.
    for group in [1, 2] {
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
        let pointer: TypeHandle =
            MirPtrType::get(&mut ctx, i32, true, address_space::GENERIC).into();
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
        if group == 1 {
            append::<dialect_nvvm::ops::Tcgen05AllocOp>(
                &mut ctx,
                block,
                vec![a[0], a[5]],
                vec![],
                Some("v1:i0343"),
            );
            append::<dialect_nvvm::ops::Tcgen05RelinquishAllocPermitOp>(
                &mut ctx,
                block,
                vec![],
                vec![],
                Some("v1:i0345"),
            );
        } else {
            append::<Tcgen05AllocCg2Op>(
                &mut ctx,
                block,
                vec![a[0], a[5]],
                vec![],
                Some("v1:i0359"),
            );
            append::<Tcgen05RelinquishAllocPermitCg2Op>(
                &mut ctx,
                block,
                vec![],
                vec![],
                Some("v1:i0361"),
            );
        }
        for tensor in [false, true] {
            let op = append::<Tcgen05MmaOp>(
                &mut ctx,
                block,
                vec![a[1], if tensor { a[1] } else { a[2] }, a[3], a[4], a[6]],
                vec![],
                Some(if tensor { "v1:i0764" } else { "v1:i0763" }),
            );
            let mma = Tcgen05MmaOp::new(op);
            mma.set_attr_nvvm_tcgen05_mma_form(
                &ctx,
                if tensor {
                    Tcgen05MmaFormAttr::Tensor
                } else {
                    Tcgen05MmaFormAttr::Shared
                },
            );
            mma.set_attr_nvvm_tcgen05_mma_kind(&ctx, Tcgen05MmaKindAttr::F16);
            mma.set_attr_nvvm_tcgen05_mma_cta_group(
                &ctx,
                if group == 1 {
                    Tcgen05MmaCtaGroupAttr::Cg1
                } else {
                    Tcgen05MmaCtaGroupAttr::Cg2
                },
            );
            mma.set_attr_nvvm_tcgen05_mma_collector_a(&ctx, Tcgen05MmaCollectorAAttr::Discard);
        }
        if group == 1 {
            append::<dialect_nvvm::ops::Tcgen05CommitOp>(
                &mut ctx,
                block,
                vec![a[0]],
                vec![],
                Some("v1:i0348"),
            );
        } else {
            append::<dialect_nvvm::ops::Tcgen05CommitCg2Op>(
                &mut ctx,
                block,
                vec![a[0]],
                vec![],
                Some("v1:i0363"),
            );
        }
        macro_rules! memory_pair {
            ($load:ty,$store:ty,$width:literal,$load_marker:literal,$store_marker:literal) => {{
                let ld = append::<$load>(
                    &mut ctx,
                    block,
                    vec![a[1]],
                    vec![i32; $width],
                    Some($load_marker),
                );
                append::<Tcgen05LoadWaitOp>(&mut ctx, block, vec![], vec![], Some("v1:i0357"));
                let regs = ld.deref(&ctx).results().collect::<Vec<_>>();
                // Retained scalar uses exercise extraction independently of the
                // native load/store vector round trip.
                for &value in &regs {
                    let store =
                        append::<MirStoreOp>(&mut ctx, block, vec![a[8], value], vec![], None);
                    MirStoreOp::new(store).set_volatile(&mut ctx, true);
                }
                append::<$store>(
                    &mut ctx,
                    block,
                    std::iter::once(a[1]).chain(regs).collect(),
                    vec![],
                    Some($store_marker),
                );
            }};
        }
        use dialect_nvvm::ops::*;
        memory_pair!(
            Tcgen05Ld32x32bX1RawOp,
            Tcgen05St32x32bX1RawOp,
            1,
            "v1:i0654",
            "v1:i0712"
        );
        memory_pair!(
            Tcgen05Ld32x32bX1Pack16Op,
            Tcgen05St32x32bX1Unpack16Op,
            1,
            "v1:i0655",
            "v1:i0713"
        );
        memory_pair!(
            Tcgen05Ld32x32bX2RawOp,
            Tcgen05St32x32bX2RawOp,
            2,
            "v1:i0656",
            "v1:i0714"
        );
        memory_pair!(
            Tcgen05Ld32x32bX2Pack16Op,
            Tcgen05St32x32bX2Unpack16Op,
            2,
            "v1:i0657",
            "v1:i0715"
        );
        memory_pair!(
            Tcgen05Ld32x32bX4RawOp,
            Tcgen05St32x32bX4RawOp,
            4,
            "v1:i0658",
            "v1:i0716"
        );
        memory_pair!(
            Tcgen05Ld32x32bX4Pack16Op,
            Tcgen05St32x32bX4Unpack16Op,
            4,
            "v1:i0659",
            "v1:i0717"
        );
        memory_pair!(
            Tcgen05Ld32x32bX8RawOp,
            Tcgen05St32x32bX8RawOp,
            8,
            "v1:i0660",
            "v1:i0718"
        );
        memory_pair!(
            Tcgen05Ld32x32bX8Pack16Op,
            Tcgen05St32x32bX8Unpack16Op,
            8,
            "v1:i0661",
            "v1:i0719"
        );
        memory_pair!(
            Tcgen05Ld32x32bX16RawOp,
            Tcgen05St32x32bX16RawOp,
            16,
            "v1:i0662",
            "v1:i0720"
        );
        memory_pair!(
            Tcgen05Ld32x32bX16Pack16Op,
            Tcgen05St32x32bX16Unpack16Op,
            16,
            "v1:i0663",
            "v1:i0721"
        );
        memory_pair!(
            Tcgen05Ld32x32bX32RawOp,
            Tcgen05St32x32bX32RawOp,
            32,
            "v1:i0664",
            "v1:i0722"
        );
        memory_pair!(
            Tcgen05Ld32x32bX32Pack16Op,
            Tcgen05St32x32bX32Unpack16Op,
            32,
            "v1:i0665",
            "v1:i0723"
        );
        memory_pair!(
            Tcgen05Ld32x32bX64RawOp,
            Tcgen05St32x32bX64RawOp,
            64,
            "v1:i0666",
            "v1:i0724"
        );
        memory_pair!(
            Tcgen05Ld32x32bX64Pack16Op,
            Tcgen05St32x32bX64Unpack16Op,
            64,
            "v1:i0667",
            "v1:i0725"
        );
        memory_pair!(
            Tcgen05Ld32x32bX128RawOp,
            Tcgen05St32x32bX128RawOp,
            128,
            "v1:i0668",
            "v1:i0726"
        );
        memory_pair!(
            Tcgen05Ld32x32bX128Pack16Op,
            Tcgen05St32x32bX128Unpack16Op,
            128,
            "v1:i0669",
            "v1:i0727"
        );
        append::<dialect_nvvm::ops::Tcgen05StoreWaitOp>(
            &mut ctx,
            block,
            vec![],
            vec![],
            Some("v1:i0358"),
        );
        if group == 1 {
            append::<dialect_nvvm::ops::Tcgen05DeallocOp>(
                &mut ctx,
                block,
                vec![a[1], a[5]],
                vec![],
                Some("v1:i0344"),
            );
        } else {
            append::<Tcgen05DeallocCg2Op>(
                &mut ctx,
                block,
                vec![a[1], a[5]],
                vec![],
                Some("v1:i0360"),
            );
        }
        append::<MirReturnOp>(&mut ctx, block, vec![], vec![], None);
        let profile = CutlassFullCuteMlir22::new("sm_100a").unwrap();
        let target = profile.translate_module(&ctx, &module).unwrap();
        let mlir = cuda_oxide_mlir_export::render_module(&target).unwrap();
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
    append::<MbarrierArriveClusterRelaxedOp>(
        &mut ctx,
        block,
        vec![address],
        vec![],
        Some("v1:i1040"),
    );
    let relaxed_wait = append::<MbarrierTryWaitParityClusterRelaxedOp>(
        &mut ctx,
        block,
        vec![a[2], zero],
        vec![i1],
        Some("v1:i1041"),
    );
    let ready = relaxed_wait.deref(&ctx).get_result(0);
    let store = append::<MirStoreOp>(&mut ctx, block, vec![a[4], ready], vec![], None);
    MirStoreOp::new(store).set_volatile(&mut ctx, true);
    append::<FenceAcquireSharedClusterClusterOp>(&mut ctx, block, vec![], vec![], Some("v1:i1042"));
    append::<MirReturnOp>(&mut ctx, block, vec![], vec![], None);

    let profile = CutlassFullCuteMlir22::new("sm_100a").unwrap();
    let target = profile.translate_module(&ctx, &module).unwrap();
    let mlir = cuda_oxide_mlir_export::render_module(&target).unwrap();
    let library =
        CutlassCompilerLibrary::load(compiler_path, CUTLASS_COMPILER_LIBRARY_SHA256).unwrap();
    let dump = tempfile::tempdir().unwrap();
    let ptx_prefix = dump.path().join("cluster_barriers");
    let pipeline = CUTLASS_PRECOMPILED_PIPELINE.replace(
        "cubin-format=bin",
        &format!("cubin-format=bin dump-ptx-path={}", ptx_prefix.display()),
    );
    let object = library
        .compile_precompiled_mlir_to_object(&mlir, "sm_100a", &pipeline)
        .unwrap_or_else(|error| {
            panic!("Cluster/TMA translated MLIR failed compilation: {error}\n{mlir}")
        });
    let ptx = std::fs::read_to_string(ptx_prefix.with_extension("sm_100a.ptx")).unwrap();
    let prefetch_count = ptx
        .lines()
        .filter(|line| line.split_whitespace().next() == Some("prefetch.tensormap"))
        .count();
    assert_eq!(prefetch_count, 1, "{ptx}");
    assert!(mlir.contains("\"nvvm.prefetch\""), "{mlir}");
    assert!(
        ptx.contains("mbarrier.arrive.release.cluster.shared::cluster.b64"),
        "{ptx}"
    );
    assert!(
        ptx.contains("mbarrier.arrive.relaxed.cluster.shared::cluster.b64"),
        "{ptx}"
    );
    assert!(
        ptx.contains("mbarrier.try_wait.parity.relaxed.cluster"),
        "{ptx}"
    );
    assert!(
        ptx.contains("fence.acquire.sync_restrict::shared::cluster.cluster;"),
        "{ptx}"
    );
    assert!(!mlir.contains("inline_ptx"), "{mlir}");
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

mod cutlass_attention_semantic_pack_compiles_module {
    use cuda_oxide_mlir_export::{CutlassFullCuteMlir22, MlirConsumerProfile};
    use dialect_cute::sm100_attention_ops::*;
    use dialect_mir::{
        ops::{MirFuncOp, MirReturnOp},
        types::{MirFP16Type, MirPtrType},
    };
    use pliron::{
        basic_block::BasicBlock,
        builtin::{
            attributes::TypeAttr,
            op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
            ops::ModuleOp,
            types::{FunctionType, IntegerType, Signedness},
        },
        context::Context,
        op::Op,
        operation::Operation,
        r#type::TypeHandle,
    };

    #[test]
    #[ignore = "requires the pinned CUTLASS compiler; no GPU"]
    fn cutlass_attention_semantic_pack_compiles() {
        for group in [1, 2] {
            let mut ctx = Context::new();
            dialect_mir::register(&mut ctx);
            dialect_nvvm::register(&mut ctx);
            dialect_cute::register(&mut ctx);
            let module = ModuleOp::new(&mut ctx, "attention".try_into().unwrap());
            let f16 = MirFP16Type::get(&ctx).into();
            let u32: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
            let u64: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
            let b: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Unsigned).into();
            let shared: TypeHandle = MirPtrType::get(&mut ctx, f16, true, 3).into();
            let desc: TypeHandle = MirPtrType::get(&mut ctx, u64, false, 0).into();
            let barrier: TypeHandle = MirPtrType::get(&mut ctx, u64, true, 3).into();
            let args = vec![u32, shared, shared, b, u32, desc, barrier, u32, u32];
            let ty = FunctionType::get(&ctx, args.clone(), vec![]);
            let f = Operation::new(
                &mut ctx,
                MirFuncOp::get_concrete_op_info(),
                vec![],
                vec![],
                vec![],
                1,
            );
            MirFuncOp::new(&mut ctx, f, TypeAttr::new(ty.into()))
                .set_symbol_name(&mut ctx, "attention_probe".try_into().unwrap());
            module.append_operation(&mut ctx, f, 0);
            let block = BasicBlock::new(&mut ctx, None, args);
            block.insert_at_back(f.deref(&ctx).get_region(0), &ctx);
            let a = block.deref(&ctx).arguments().collect::<Vec<_>>();
            let plan = CuteSm100MmaAttr {
                m: 128,
                n: 128,
                k: 128,
                cta_group: group,
                a_mn: false,
                b_mn: false,
                a_layout: shared_layout(128 / group, 128, false).unwrap(),
                b_layout: shared_layout(128 / group, 128, false).unwrap(),
            };
            CuteSm100MmaSsOp::new(
                &mut ctx,
                vec![a[0], a[1], a[2], a[3], a[7], a[8]],
                plan.clone(),
            )
            .get_operation()
            .insert_at_back(block, &ctx);
            if group == 1 {
                let mut pv = plan.clone();
                pv.b_mn = true;
                pv.b_layout = shared_layout(128, 128, true).unwrap();
                CuteSm100MmaTsOp::new(&mut ctx, vec![a[0], a[4], a[2], a[3], a[8]], pv)
                    .get_operation()
                    .insert_at_back(block, &ctx);
            }
            let copy = CuteSm100TmaAttr {
                layout: plan.a_layout,
                cta_group: group,
            };
            CuteSm100TmaLoadOp::new(
                &mut ctx,
                vec![a[5], a[1], a[4], a[4], a[4], a[6], a[4]],
                copy.clone(),
            )
            .get_operation()
            .insert_at_back(block, &ctx);
            CuteSm100TmaStoreOp::new(&mut ctx, vec![a[5], a[1], a[4], a[4], a[4]], copy)
                .get_operation()
                .insert_at_back(block, &ctx);
            Operation::new(
                &mut ctx,
                MirReturnOp::get_concrete_op_info(),
                vec![],
                vec![],
                vec![],
                0,
            )
            .insert_at_back(block, &ctx);
            f.deref_mut(&ctx).attributes.set(
                "gpu_kernel".try_into().unwrap(),
                pliron::builtin::attributes::StringAttr::new("true".into()),
            );
            let target = CutlassFullCuteMlir22::new("sm_100a")
                .unwrap()
                .translate_module(&ctx, &module)
                .unwrap();
            let text = cuda_oxide_mlir_export::render_module(&target).unwrap();
            let library = super::CutlassCompilerLibrary::load(
                std::env::var_os("CUDA_OXIDE_TEST_CUTLASS_COMPILER").expect("pinned compiler path"),
                super::CUTLASS_COMPILER_LIBRARY_SHA256,
            )
            .unwrap();
            let dump = tempfile::tempdir().unwrap();
            let prefix = dump.path().join("attention_descriptors");
            let pipeline = super::CUTLASS_PRECOMPILED_PIPELINE.replace(
                "cubin-format=bin",
                &format!("cubin-format=bin dump-ptx-path={}", prefix.display()),
            );
            let object = library
                .compile_precompiled_mlir_to_object(&text, "sm_100a", &pipeline)
                .unwrap_or_else(|e| panic!("attention compiler probe: {e}\n{text}"));
            assert!(super::extract_kernels_binary(&object.elf).unwrap().len() > 64);
            let ptx = std::fs::read_to_string(prefix.with_extension("sm_100a.ptx")).unwrap();
            let copies = ptx
                .lines()
                .filter(|line| {
                    line.contains("cp.async.bulk.tensor")
                        && line.contains(".mbarrier::complete_tx::bytes")
                })
                .collect::<Vec<_>>();
            assert_eq!(copies.len(), 1, "{ptx}");
            assert!(!copies[0].contains(".multicast"), "{ptx}");
            assert_eq!(copies[0].contains(".cta_group::2"), group == 2, "{ptx}");

            assert_eq!(
                text.matches("\"nvvm.tcgen05.mma\"").count(),
                if group == 1 { 16 } else { 8 }
            );
            assert!(text.contains("#cute_nvgpu.major<k>"), "{text}");
            assert!(text.contains("num_cta = "), "{text}");
            assert!(text.contains("1, 1, 1, 3,"), "{text}");
            if group == 1 {
                assert!(text.contains("#cute_nvgpu.major<mn>"), "{text}");
            }
        }
    }
}

mod cutlass_attention_numeric_pack_compiles_module {
    /*
     * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
     * SPDX-License-Identifier: Apache-2.0
     */

    use cuda_oxide_mlir_export::{CutlassFullCuteMlir22, MlirConsumerProfile};
    use dialect_mir::ops::{MirFuncOp, MirReturnOp};
    use dialect_nvvm::ops::*;
    use pliron::{
        basic_block::BasicBlock,
        builtin::{
            attributes::{StringAttr, TypeAttr},
            op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
            ops::ModuleOp,
            types::{FP32Type, FunctionType, IntegerType, Signedness},
        },
        context::{Context, Ptr},
        op::Op,
        operation::Operation,
        r#type::{TypeHandle, Typed},
    };

    fn mark(ctx: &mut Context, op: Ptr<Operation>, block: Ptr<BasicBlock>, marker: &str) {
        op.deref_mut(ctx).attributes.set(
            "cuda_oxide_intrinsic_marker".try_into().unwrap(),
            StringAttr::new(marker.into()),
        );
        op.insert_at_back(block, ctx);
        if op.deref(ctx).get_num_results() == 1 {
            let value = op.deref(ctx).get_result(0);
            let args = block.deref(ctx).arguments().collect::<Vec<_>>();
            let ptr = args
                .iter()
                .rev()
                .find(|arg| {
                    arg.get_type(ctx)
                        .deref(ctx)
                        .downcast_ref::<dialect_mir::types::MirPtrType>()
                        .is_some_and(|p| p.pointee == value.get_type(ctx))
                })
                .copied()
                .expect("output pointer");
            let store = Operation::new(
                ctx,
                dialect_mir::ops::MirStoreOp::get_concrete_op_info(),
                vec![],
                vec![ptr, value],
                vec![],
                0,
            );
            dialect_mir::ops::MirStoreOp::new(store).set_volatile(ctx, true);
            store.insert_at_back(block, ctx);
        }
    }

    #[test]
    #[ignore = "requires the pinned CUTLASS compiler; no GPU"]
    fn cutlass_attention_numeric_pack_compiles() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        dialect_nvvm::register(&mut ctx);
        dialect_cute::register(&mut ctx);
        let module = ModuleOp::new(&mut ctx, "numeric".try_into().unwrap());
        let f32: TypeHandle = FP32Type::get(&ctx).into();
        let i32: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let pred = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let shared = dialect_mir::types::MirPtrType::get(&mut ctx, i32, true, 3).into();
        let global = dialect_mir::types::MirPtrType::get(&mut ctx, i32, false, 1).into();
        let u64 = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let bar = dialect_mir::types::MirPtrType::get(&mut ctx, u64, true, 3).into();
        let f_out = dialect_mir::types::MirPtrType::get(&mut ctx, f32, true, 1).into();
        let u_out = dialect_mir::types::MirPtrType::get(&mut ctx, i32, true, 1).into();
        let boolean = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let b_out = dialect_mir::types::MirPtrType::get(&mut ctx, boolean, true, 1).into();
        let packed_out = dialect_mir::types::MirPtrType::get(&mut ctx, u64, true, 1).into();
        let args = vec![
            f32, f32, f32, i32, i32, pred, shared, global, bar, f_out, u_out, b_out, u64, u64, u64,
            packed_out,
        ];
        let fty = FunctionType::get(&ctx, args.clone(), vec![]);
        let op = Operation::new(
            &mut ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let f = MirFuncOp::new(&mut ctx, op, TypeAttr::new(fty.into()));
        for (key, value) in [("maxntid", 128), ("minctasm", 1)] {
            let integer = pliron::builtin::attributes::IntegerAttr::new(
                IntegerType::get(&ctx, 32, Signedness::Unsigned),
                pliron::utils::apint::APInt::from_u32(
                    value,
                    std::num::NonZeroUsize::new(32).unwrap(),
                ),
            );
            op.deref_mut(&ctx)
                .attributes
                .set(key.try_into().unwrap(), integer);
        }

        f.set_symbol_name(&mut ctx, "numeric_probe".try_into().unwrap());
        module.append_operation(&mut ctx, op, 0);
        let block = BasicBlock::new(&mut ctx, None, args);
        block.insert_at_back(op.deref(&ctx).get_region(0), &ctx);
        let a = block.deref(&ctx).arguments().collect::<Vec<_>>();
        // Distinct dynamic packed words retain both FP32 lanes through native arithmetic.
        let add = AddF32x2Op::build(&mut ctx, a[12], a[13]);
        mark(&mut ctx, add, block, "v1:i0995");
        let mul = MulF32x2Op::build(&mut ctx, a[12], a[13]);
        mark(&mut ctx, mul, block, "v1:i0999");
        let fma = FmaF32x2Op::build(&mut ctx, a[12], a[13], a[14]);
        mark(&mut ctx, fma, block, "v1:i1001");
        let shuffle = ShflSyncDownF32Op::build(&mut ctx, a[3], a[0], a[4]);
        mark(&mut ctx, shuffle, block, "v1:i0056");
        let shuffle = ShflSyncBflyI32Op::build(&mut ctx, a[3], a[4], a[4]);
        mark(&mut ctx, shuffle, block, "v1:i0051");
        // Ordinary Rust selection and saturation survive the same consumer boundary.
        use dialect_mir::{
            ops::{MirCallOp, MirStoreOp},
            rust_intrinsics::{CALLEE_SATURATING_SUB, CALLEE_SELECT_UNPREDICTABLE},
        };
        let maximum = Operation::new(
            &mut ctx,
            MirCallOp::get_concrete_op_info(),
            vec![f32],
            vec![a[0], a[1]],
            vec![],
            0,
        );
        MirCallOp::new(maximum).set_attr_callee(
            &ctx,
            StringAttr::new(dialect_mir::rust_intrinsics::CALLEE_MAXNUM_NSZ_F32.into()),
        );
        maximum.insert_at_back(block, &ctx);
        let max_value = maximum.deref(&ctx).get_result(0);
        let max_store = Operation::new(
            &mut ctx,
            MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![a[9], max_value],
            vec![],
            0,
        );
        MirStoreOp::new(max_store).set_volatile(&mut ctx, true);
        max_store.insert_at_back(block, &ctx);
        let select = Operation::new(
            &mut ctx,
            MirCallOp::get_concrete_op_info(),
            vec![i32],
            vec![a[5], a[3], a[4]],
            vec![],
            0,
        );
        MirCallOp::new(select)
            .set_attr_callee(&ctx, StringAttr::new(CALLEE_SELECT_UNPREDICTABLE.into()));
        select.insert_at_back(block, &ctx);
        let selected = select.deref(&ctx).get_result(0);
        let sub = Operation::new(
            &mut ctx,
            MirCallOp::get_concrete_op_info(),
            vec![i32],
            vec![selected, a[4]],
            vec![],
            0,
        );
        MirCallOp::new(sub).set_attr_callee(&ctx, StringAttr::new(CALLEE_SATURATING_SUB.into()));
        sub.insert_at_back(block, &ctx);
        let value = sub.deref(&ctx).get_result(0);
        let store = Operation::new(
            &mut ctx,
            MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![a[10], value],
            vec![],
            0,
        );
        MirStoreOp::new(store).set_volatile(&mut ctx, true);
        store.insert_at_back(block, &ctx);

        for (kind, ftz, marker) in [
            (ScalarMathOperationAttr::Ex2, false, "v1:i0819"),
            (ScalarMathOperationAttr::Lg2, false, "v1:i0786"),
            (ScalarMathOperationAttr::Rcp, true, "v1:i0788"),
        ] {
            let op = ScalarMathOp::build(
                &mut ctx,
                a[0],
                ScalarMathFormatAttr::F32,
                kind,
                ScalarMathPrecisionAttr::Approx,
                if ftz {
                    ScalarMathSubnormalAttr::Ftz
                } else {
                    ScalarMathSubnormalAttr::Preserve
                },
            );
            mark(&mut ctx, op, block, marker);
        }
        let op = ScalarArithmeticOp::build(
            &mut ctx,
            a[..3].to_vec(),
            ScalarArithmeticFormatAttr::F32,
            ScalarArithmeticOperationAttr::Fma,
            ScalarArithmeticRoundingAttr::Rn,
            ScalarArithmeticSubnormalAttr::Preserve,
            ScalarArithmeticSaturationAttr::None,
        );
        mark(&mut ctx, op, block, "v1:i0418");
        let op = CvtF16x2F32Op::build(&mut ctx, a[0], a[1]);
        mark(&mut ctx, op, block, "v1:i0081");
        let op = ReduxSyncFmaxOp::build(&mut ctx, a[3], a[0]);
        mark(&mut ctx, op, block, "v1:i1007");
        let op = SetMaxNRegIncOp::build(&mut ctx, 192);
        mark(&mut ctx, op, block, "v1:i0915");
        let op = CpAsyncCa4Op::build(&mut ctx, a[6], a[7]);
        mark(&mut ctx, op, block, "v1:i0086");
        let op = CpAsyncMbarrierArriveNoIncOp::build(&mut ctx, a[8]);
        mark(&mut ctx, op, block, "v1:i0103");
        let op = VoteSyncAllOp::build(&mut ctx, a[3], a[5]);
        mark(&mut ctx, op, block, "v1:i0040");
        let op = BarWarpSyncOp::build(&mut ctx, a[3]);
        mark(&mut ctx, op, block, "v1:i0049");
        let ret = Operation::new(
            &mut ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        ret.insert_at_back(block, &ctx);
        block
            .deref(&ctx)
            .get_parent_region()
            .unwrap()
            .deref(&ctx)
            .get_parent_op()
            .deref_mut(&ctx)
            .attributes
            .set(
                "gpu_kernel".try_into().unwrap(),
                StringAttr::new("true".into()),
            );
        let result = CutlassFullCuteMlir22::new("sm_100a")
            .unwrap()
            .translate_module(&ctx, &module)
            .unwrap();
        let text = cuda_oxide_mlir_export::render_module(&result).unwrap();
        let library = super::CutlassCompilerLibrary::load(
            std::env::var_os("CUDA_OXIDE_TEST_CUTLASS_COMPILER").expect("pinned compiler path"),
            super::CUTLASS_COMPILER_LIBRARY_SHA256,
        )
        .unwrap();
        let dump = tempfile::tempdir().unwrap();
        let ptx_prefix = dump.path().join("attention_numeric");
        let pipeline = super::CUTLASS_PRECOMPILED_PIPELINE.replace(
            "cubin-format=bin",
            &format!("cubin-format=bin dump-ptx-path={}", ptx_prefix.display()),
        );
        let object = library
            .compile_precompiled_mlir_to_object(&text, "sm_100a", &pipeline)
            .unwrap_or_else(|e| panic!("attention compiler probe: {e}\n{text}"));
        assert!(super::extract_kernels_binary(&object.elf).unwrap().len() > 64);
        let ptx = std::fs::read_to_string(ptx_prefix.with_extension("sm_100a.ptx")).unwrap();
        assert!(ptx.contains(".maxntid 128, 1, 1"), "{ptx}");
        assert!(ptx.contains(".minnctapersm 1"), "{ptx}");
        assert!(ptx.contains("setmaxnreg.inc.sync.aligned.u32"), "{ptx}");
        assert!(
            ptx.lines()
                .any(|line| line.split_whitespace().next() == Some("max.f32")),
            "{ptx}"
        );
        assert!(
            !ptx.contains("max.NaN.f32") && !ptx.contains("max.ftz.f32"),
            "{ptx}"
        );

        for instruction in ["add.rn.f32x2", "mul.rn.f32x2", "fma.rn.f32x2"] {
            assert!(
                ptx.contains(instruction),
                "packed instruction lost: {instruction}\n{ptx}"
            );
        }

        for name in [
            "llvm.nvvm.ex2.approx.f32",
            "llvm.nvvm.lg2.approx.f",
            "llvm.nvvm.rcp.approx.ftz.f",
            "llvm.nvvm.fma.rn.f",
            "nvvm.redux.sync",
            "nvvm.cp.async.shared.global",
            "nvvm.cp.async.mbarrier.arrive",
            "nvvm.vote.sync",
            "nvvm.bar.warp.sync",
            "llvm.nvvm.ff2f16x2.rn",
            "nvvm.add.packed.f32x2",
            "nvvm.mul.packed.f32x2",
            "nvvm.fma.packed.f32x2",
        ] {
            assert!(text.contains(name), "{text}");
        }
        assert!(text.contains("192 : i32"), "{text}");
        assert!(!text.contains("cuda_oxide_intrinsic_marker"));
    }
}

/// The source-level CuTe-RS hint reuses the catalog-owned full-warp shuffle.
/// Compare that existing native route with the pinned CuTe operation itself.
#[test]
#[ignore = "requires the pinned CUTLASS compiler; no GPU"]
fn cutlass_warp_uniform_matches_cute_native_hint() {
    use dialect_mir::ops::MirShrOp;
    use dialect_nvvm::ops::{ReadPtxSregTidXOp, ShflSyncIdxI32Op};
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    dialect_cute::register(&mut ctx);
    let module = ModuleOp::new(&mut ctx, "warp_uniform".try_into().unwrap());
    let word = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let pointer: TypeHandle = MirPtrType::get(&mut ctx, word.into(), true, 1).into();
    let ty = FunctionType::get(&ctx, vec![pointer], vec![]);
    let func = Operation::new(
        &mut ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    MirFuncOp::new(&mut ctx, func, TypeAttr::new(ty.into()))
        .set_symbol_name(&mut ctx, "warp_uniform_probe".try_into().unwrap());
    func.deref_mut(&ctx).attributes.set(
        "gpu_kernel".try_into().unwrap(),
        StringAttr::new("true".into()),
    );
    module.append_operation(&mut ctx, func, 0);
    let block = BasicBlock::new(&mut ctx, None, vec![pointer]);
    block.insert_at_back(func.deref(&ctx).get_region(0), &ctx);
    let output = block.deref(&ctx).arguments().next().unwrap();
    let tid =
        append::<ReadPtxSregTidXOp>(&mut ctx, block, vec![], vec![word.into()], Some("v1:i0001"));
    let tid = tid.deref(&ctx).get_result(0);
    let five = integer_constant(&mut ctx, block, word, 5, 32);
    let warp = append::<MirShrOp>(&mut ctx, block, vec![tid, five], vec![word.into()], None);
    let warp = warp.deref(&ctx).get_result(0);
    let full_mask = integer_constant(&mut ctx, block, word, u32::MAX as u64, 32);
    let zero = integer_constant(&mut ctx, block, word, 0, 32);
    let uniform = append::<ShflSyncIdxI32Op>(
        &mut ctx,
        block,
        vec![full_mask, warp, zero],
        vec![word.into()],
        Some("v1:i0050"),
    );
    let value = uniform.deref(&ctx).get_result(0);
    let store = append::<MirStoreOp>(&mut ctx, block, vec![output, value], vec![], None);
    MirStoreOp::new(store).set_volatile(&mut ctx, true);
    append::<MirReturnOp>(&mut ctx, block, vec![], vec![], None);
    let target = CutlassFullCuteMlir22::new("sm_100a")
        .unwrap()
        .translate_module(&ctx, &module)
        .unwrap();
    let native = cuda_oxide_mlir_export::render_module(&target).unwrap();
    assert!(native.contains("llvm.nvvm.shfl.sync.idx.i32"), "{native}");
    let cute = r#"module {
      gpu.module @kernels {
        llvm.func @warp_uniform_probe(%out: !llvm.ptr<1>) attributes {nvvm.kernel} {
          %tid = nvvm.read.ptx.sreg.tid.x : i32
          %five = arith.constant 5 : i32
          %warp = arith.shrui %tid, %five : i32
          %uniform = "cute_nvgpu.arch.make_warp_uniform"(%warp) : (i32) -> i32
          llvm.store %uniform, %out {volatile_} : i32, !llvm.ptr<1>
          llvm.return
        }
      }
    }"#;
    let library = CutlassCompilerLibrary::load(
        std::env::var_os("CUDA_OXIDE_TEST_CUTLASS_COMPILER").expect("pinned compiler path"),
        CUTLASS_COMPILER_LIBRARY_SHA256,
    )
    .unwrap();
    let dump = tempfile::tempdir().unwrap();
    for (name, source) in [("native", native.as_str()), ("cute", cute)] {
        let prefix = dump.path().join(name);
        let pipeline = CUTLASS_PRECOMPILED_PIPELINE.replace(
            "cubin-format=bin",
            &format!("cubin-format=bin dump-ptx-path={}", prefix.display()),
        );
        let object = library
            .compile_precompiled_mlir_to_object(source, "sm_100a", &pipeline)
            .unwrap_or_else(|e| panic!("{name} uniform probe: {e}\n{source}"));
        assert!(extract_kernels_binary(&object.elf).unwrap().len() > 64);
        let ptx = std::fs::read_to_string(prefix.with_extension("sm_100a.ptx")).unwrap();
        let shuffle = ptx
            .lines()
            .filter(|line| line.contains("shfl.sync"))
            .collect::<Vec<_>>();
        assert_eq!(shuffle.len(), 1, "{name}: {ptx}");
        assert!(shuffle[0].contains("shfl.sync.idx.b32"), "{name}: {ptx}");
        assert!(
            shuffle[0].trim_end().ends_with(", 0, 31, -1;"),
            "{name}: {ptx}"
        );
    }
}

/// A semantic register tile must remain one cache-aware 256-bit global store.
#[test]
#[ignore = "requires the pinned CUTLASS compiler; no GPU"]
fn cutlass_global_store_no_allocate_is_one_256_bit_transaction() {
    assert!(CutlassFullCuteMlir22::new("sm_90a").is_err());
    use dialect_cute::copy_ops::CuteStoreGlobalNoAllocateOp;
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    dialect_cute::register(&mut ctx);
    let module = ModuleOp::new(&mut ctx, "wide_store".try_into().unwrap());
    let word: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
    let output: TypeHandle = MirPtrType::get(&mut ctx, word, true, address_space::GLOBAL).into();
    let source: TypeHandle = MirPtrType::get(&mut ctx, word, false, address_space::GENERIC).into();
    let ty = FunctionType::get(&ctx, vec![output, source], vec![]);
    let func = Operation::new(
        &mut ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    MirFuncOp::new(&mut ctx, func, TypeAttr::new(ty.into()))
        .set_symbol_name(&mut ctx, "wide_store_probe".try_into().unwrap());
    func.deref_mut(&ctx).attributes.set(
        "gpu_kernel".try_into().unwrap(),
        StringAttr::new("true".into()),
    );
    module.append_operation(&mut ctx, func, 0);
    let block = BasicBlock::new(&mut ctx, None, vec![output, source]);
    block.insert_at_back(func.deref(&ctx).get_region(0), &ctx);
    let a: Vec<_> = block.deref(&ctx).arguments().collect();
    let store = CuteStoreGlobalNoAllocateOp::new(&mut ctx, a[0], a[1], 8);
    store.get_operation().insert_at_back(block, &ctx);
    append::<MirReturnOp>(&mut ctx, block, vec![], vec![], None);
    let target = CutlassFullCuteMlir22::new("sm_100a")
        .unwrap()
        .translate_module(&ctx, &module)
        .unwrap();
    let text = cuda_oxide_mlir_export::render_module(&target).unwrap();
    assert!(
        text.contains("atom.r2g<i32 copy_bits = 256 l1_cache_evict_priority = no_allocate>"),
        "{text}"
    );
    assert!(text.contains("gmem, align<32>"), "{text}");
    assert!(text.contains("rmem, align<32>"), "{text}");
    let compiler = CutlassCompilerLibrary::load(
        std::env::var_os("CUDA_OXIDE_TEST_CUTLASS_COMPILER").expect("pinned compiler path"),
        CUTLASS_COMPILER_LIBRARY_SHA256,
    )
    .unwrap();
    let dump = tempfile::tempdir().unwrap();
    let prefix = dump.path().join("wide-store");
    let pipeline = CUTLASS_PRECOMPILED_PIPELINE.replace(
        "cubin-format=bin",
        &format!("cubin-format=bin dump-ptx-path={}", prefix.display()),
    );
    let object = compiler
        .compile_precompiled_mlir_to_object(&text, "sm_100a", &pipeline)
        .unwrap_or_else(|e| panic!("{e}\n{text}"));
    assert!(extract_kernels_binary(&object.elf).unwrap().len() > 64);
    let ptx = std::fs::read_to_string(prefix.with_extension("sm_100a.ptx")).unwrap();
    let stores: Vec<_> = ptx
        .lines()
        .filter(|line| line.contains("st.global"))
        .collect();
    assert_eq!(stores.len(), 1, "{ptx}");
    assert!(
        stores[0].contains("L1::no_allocate")
            && (stores[0].contains(".v8.b32") || stores[0].contains(".v4.b64")),
        "{ptx}"
    );
}
