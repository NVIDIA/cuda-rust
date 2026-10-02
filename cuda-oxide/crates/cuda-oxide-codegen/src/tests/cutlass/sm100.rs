/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile the actual semantic SM100 mapping family through the pinned consumer.
//! This fixture is never launched: warp/cluster participation is tested by the
//! example's GPU smoke tests. In particular, successful MLIR parsing alone does
//! not validate CUTLASS 4.7's integer-register ABI for native TMEM loads.

use super::{
    CUTLASS_COMPILER_LIBRARY_SHA256, CUTLASS_PRECOMPILED_PIPELINE, extract_kernels_binary,
};
use crate::cutlass_compiler::CutlassCompilerLibrary;
use cuda_oxide_mlir_export::{CutlassFullCuteMlir22, MlirConsumerProfile};
use dialect_cute::{
    attributes::{CuteComposedLayoutAttr, CuteTmaStorePipelineAttr},
    layout::{ComposedLayout, OffsetUnit, Swizzle},
    sm100_ops::*,
};
use dialect_mir::{
    ops::{MirFuncOp, MirReturnOp},
    types::{MirFP16Type, MirPtrType, address_space},
};
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
    r#type::TypeHandle,
};

fn shared_layout(rows: u32, columns: u32, bits: u32) -> CuteComposedLayoutAttr {
    CuteComposedLayoutAttr(
        ComposedLayout::new(
            Swizzle::new(bits, 3, 3),
            0,
            format!("({rows},{columns}):({columns},1)").parse().unwrap(),
            OffsetUnit::Elements,
        )
        .unwrap(),
    )
}
fn append<O: Op>(ctx: &Context, block: Ptr<BasicBlock>, op: O) {
    op.get_operation().insert_at_back(block, ctx);
}

#[test]
#[ignore = "requires the installed official CUTLASS 4.7 compiler; does not require a GPU"]
fn cutlass_sm100_semantic_pack_compiles() {
    let path = std::env::var_os("CUDA_OXIDE_TEST_CUTLASS_COMPILER")
        .expect("set CUDA_OXIDE_TEST_CUTLASS_COMPILER to the pinned compiler library");
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    dialect_cute::register(&mut ctx);
    let u32: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
    let u64: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
    let i1: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Unsigned).into();
    let f32: TypeHandle = FP32Type::get(&ctx).into();
    let f16: TypeHandle = MirFP16Type::get(&ctx).into();
    let halfptr: TypeHandle = MirPtrType::get(&mut ctx, f16, true, address_space::SHARED).into();
    let barptr: TypeHandle = MirPtrType::get(&mut ctx, u64, true, address_space::SHARED).into();
    let tokenptr: TypeHandle = MirPtrType::get(&mut ctx, u32, true, address_space::SHARED).into();
    let descptr: TypeHandle = MirPtrType::get(&mut ctx, u64, false, address_space::GENERIC).into();
    // TMEM, A/B/C shared tiles, predicate, full/empty barriers, stage/phase,
    // bias, allocation token, descriptor, and cluster rank.
    let args = vec![
        u32, halfptr, halfptr, halfptr, i1, barptr, barptr, u32, u32, f32, tokenptr, descptr, u32,
    ];
    let module = ModuleOp::new(&mut ctx, "sm100_semantic_smoke".try_into().unwrap());
    let ft = FunctionType::get(&ctx, args.clone(), vec![]);
    let func = Operation::new(
        &mut ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let function = MirFuncOp::new(&mut ctx, func, TypeAttr::new(ft.into()));
    function.set_symbol_name(&mut ctx, "sm100_semantic_probe".try_into().unwrap());
    func.deref_mut(&ctx).attributes.set(
        "gpu_kernel".try_into().unwrap(),
        StringAttr::new("true".into()),
    );
    module.append_operation(&mut ctx, func, 0);
    let block = BasicBlock::new(&mut ctx, None, args);
    block.insert_at_back(func.deref(&ctx).get_region(0), &ctx);
    let a = block.deref(&ctx).arguments().collect::<Vec<_>>();
    let tmem = CuteSm100TmemPlanAttr {
        columns: 512,
        cta_group: 2,
    };
    let mma = CuteSm100MmaPlanAttr {
        m: 256,
        n0: 192,
        n1: 160,
        k: 64,
        tmem_columns: 512,
        cta_group: 2,
        collector: CuteSm100CollectorAttr::AFillLastUse,
        a_layout: shared_layout(128, 64, 3),
        b0_layout: shared_layout(96, 64, 3),
        b1_layout: shared_layout(80, 64, 3),
    };
    let pipeline = CuteSm100PipelinePlanAttr {
        stages: 5,
        transaction_bytes: 77_824,
        consumer_arrivals: 1,
        cta_group: 2,
        leader_rank: 0,
        multicast_mask: 3,
    };
    let accumulator = CuteSm100PipelinePlanAttr {
        stages: 1,
        transaction_bytes: 0,
        consumer_arrivals: 8,
        cta_group: 2,
        leader_rank: 0,
        multicast_mask: 3,
    };
    let epilogue = CuteSm100EpiloguePlanAttr {
        layout: shared_layout(128, 32, 2),
        rows: 128,
        columns: 32,
    };
    let store = CuteTmaStorePipelineAttr { stages: 2 };
    macro_rules! emit {($ty:ty,[$($operand:expr),*],$plan:expr)=>{{
        let op=<$ty>::new(&mut ctx,vec![$($operand),*],$plan);append(&ctx,block,op);
    }};}
    emit!(CuteSm100TmemAllocOp, [a[10]], tmem);
    emit!(CuteSm100PipelineInitOp, [a[5], a[6]], pipeline);
    emit!(CuteSm100AccumulatorInitOp, [a[6], a[5]], accumulator);
    emit!(CuteSm100PipelineAcquireOp, [a[6], a[7], a[8]], pipeline);
    emit!(CuteSm100PipelineExpectOp, [a[5], a[7]], pipeline);
    for (ptr, layout) in [
        (a[1], mma.a_layout.clone()),
        (a[2], mma.b0_layout.clone()),
        (a[3], mma.b1_layout.clone()),
    ] {
        let copy = CuteSm100ClusterCopyPlanAttr {
            layout,
            cta_group: 2,
            leader_rank: 0,
        };
        emit!(
            CuteSm100ClusterTmaLoadOp,
            [a[11], ptr, a[7], a[7], a[5], a[12]],
            copy
        );
    }
    emit!(CuteSm100PipelineWaitOp, [a[5], a[7], a[8]], pipeline);
    emit!(CuteSm100AccumulatorAcquireOp, [a[6], a[8]], accumulator);
    emit!(CuteSm100TiledMmaOp, [a[0], a[1], a[2], a[3], a[4]], mma);
    emit!(CuteSm100PipelineReleaseOp, [a[6], a[7]], pipeline);
    emit!(CuteSm100AccumulatorCommitOp, [a[5]], accumulator);
    emit!(CuteSm100AccumulatorWaitOp, [a[5], a[8]], accumulator);
    emit!(
        CuteSm100TmemEpilogueOp,
        [a[0], a[1], a[7], a[9]],
        epilogue.clone()
    );
    emit!(CuteSm100TmaStoreOp, [a[11], a[1], a[7], a[7]], epilogue);
    emit!(CuteSm100StoreCommitOp, [], store);
    emit!(CuteSm100StoreAcquireOp, [], store);
    emit!(CuteSm100StoreTailOp, [], store);
    emit!(CuteSm100AccumulatorReleaseOp, [a[6]], accumulator);
    emit!(CuteSm100TmemDeallocOp, [a[0]], tmem);
    let ret = Operation::new(
        &mut ctx,
        MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    ret.insert_at_back(block, &ctx);
    let profile = CutlassFullCuteMlir22::new("sm_100a").unwrap();
    let target = profile.translate_module(&ctx, &module).unwrap();
    let mlir = pliron_mlir_export::render_module(&target);
    assert!(mlir.contains("cute_nvgpu.arch.copy.SM100.tmem_load"));
    assert!(mlir.contains("vector<32xi32>"));
    assert_eq!(mlir.matches("cute_nvgpu.make_umma_smem_desc").count(), 3);
    let library = CutlassCompilerLibrary::load(path, CUTLASS_COMPILER_LIBRARY_SHA256).unwrap();
    let object = library
        .compile_precompiled_mlir_to_object(&mlir, "sm_100a", CUTLASS_PRECOMPILED_PIPELINE)
        .unwrap_or_else(|error| panic!("semantic SM100 MLIR failed compilation: {error}\n{mlir}"));
    let image = extract_kernels_binary(&object.elf).unwrap();
    assert!(image.len() > 64);
}
