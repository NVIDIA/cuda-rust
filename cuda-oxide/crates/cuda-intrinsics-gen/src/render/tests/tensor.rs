/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;
use crate::model::{CatalogTargetAlternative, TargetSelectorBinding};

use crate::model::{
    CatalogHardwareAlternative, CatalogHardwareTarget, CatalogTargetContract, IntrinsicSource,
    Tcgen05Adapter, Tcgen05MmaBUsage, Tcgen05MmaForm, Tcgen05MmaKind, TmaAdapter,
};
use crate::render::common::{generated_hardware_target, hardware_target_label, llvm};
use crate::render::families::{cluster_barriers, tcgen05_mma_inline_asm};
use std::path::Path;

#[test]
fn paired_target_matrix_renders_selectors_and_ptx_floors() {
    let target = CatalogHardwareTarget::TargetMatrix {
        contracts: vec![CatalogTargetContract {
            selectors: vec![TargetSelectorBinding {
                name: "kind".into(),
                value: "i8".into(),
            }],
            alternatives: vec![
                CatalogTargetAlternative {
                    minimum_ptx: "8.8".parse().unwrap(),
                    hardware: CatalogHardwareAlternative::ExactArchitecture { sm: 100 },
                },
                CatalogTargetAlternative {
                    minimum_ptx: "9.0".parse().unwrap(),
                    hardware: CatalogHardwareAlternative::ExactArchitecture { sm: 110 },
                },
            ],
        }],
    };

    assert_eq!(
        hardware_target_label(&target),
        "sm_100a at PTX 8.8 or sm_110a at PTX 9.0 for kind=i8"
    );
    assert_eq!(
        generated_hardware_target(&target),
        "GeneratedHardwareTarget::TargetMatrix { contracts: &[GeneratedTargetContract { selectors: &[GeneratedTargetSelectorBinding { name: \"kind\", value: \"i8\" }], alternatives: &[GeneratedTargetAlternative { minimum_ptx: GeneratedPtxVersion::from_encoded(88), hardware: GeneratedHardwareAlternative::ExactArchitecture(100) }, GeneratedTargetAlternative { minimum_ptx: GeneratedPtxVersion::from_encoded(90), hardware: GeneratedHardwareAlternative::ExactArchitecture(110) }] }] }"
    );
}

#[test]
fn tcgen05_mma_inline_asm_closes_selectors_and_operand_order() {
    for (kind, spelling) in [
        (Tcgen05MmaKind::F16, "kind::f16"),
        (Tcgen05MmaKind::Tf32, "kind::tf32"),
        (Tcgen05MmaKind::F8f6f4, "kind::f8f6f4"),
        (Tcgen05MmaKind::I8, "kind::i8"),
    ] {
        let (template, _) = tcgen05_mma_inline_asm(
            Tcgen05MmaForm::WsTensor,
            kind,
            1,
            None,
            Some(0),
            Some(Tcgen05MmaBUsage::Discard),
        );
        assert!(template.contains(spelling));
    }
    for (usage, spelling) in [
        (Tcgen05MmaBUsage::Discard, "collector::b0::discard"),
        (Tcgen05MmaBUsage::LastUse, "collector::b0::lastuse"),
        (Tcgen05MmaBUsage::Fill, "collector::b0::fill"),
        (Tcgen05MmaBUsage::Use, "collector::b0::use"),
    ] {
        let (template, _) = tcgen05_mma_inline_asm(
            Tcgen05MmaForm::WsTensor,
            Tcgen05MmaKind::F16,
            1,
            None,
            Some(0),
            Some(usage),
        );
        assert!(template.contains(spelling));
    }

    for (form, operands, constraints) in [
        (
            Tcgen05MmaForm::WsTensor,
            "[$0], [$1], $2, $3, %enable_pred;",
            "r,r,l,r,r,~{memory}",
        ),
        (
            Tcgen05MmaForm::WsSpTensor,
            "[$0], [$1], $2, [$5], $3, %enable_pred;",
            "r,r,l,r,r,r,~{memory}",
        ),
        (
            Tcgen05MmaForm::WsTensorZeroColMask,
            "[$0], [$1], $2, $3, %enable_pred, $5;",
            "r,r,l,r,r,l,~{memory}",
        ),
        (
            Tcgen05MmaForm::WsSpTensorZeroColMask,
            "[$0], [$1], $2, [$5], $3, %enable_pred, $6;",
            "r,r,l,r,r,r,l,~{memory}",
        ),
    ] {
        let (template, actual_constraints) = tcgen05_mma_inline_asm(
            form,
            Tcgen05MmaKind::F16,
            1,
            None,
            Some(3),
            Some(Tcgen05MmaBUsage::LastUse),
        );
        assert!(template.contains(operands));
        assert_eq!(actual_constraints, constraints);
    }
}

#[test]
fn wgmma_controls_render_closed_compatibility_and_backend_routes() {
    let catalog = catalog_with_wgmma_controls();
    validate_renderable(&catalog).unwrap();
    assert_eq!(wgmma_controls(&catalog).count(), 3);

    let compatibility = render_compat_wgmma_control(&catalog, "test-hash");
    assert!(compatibility.contains("pub fn wgmma_fence()"));
    assert!(compatibility.contains("pub fn wgmma_commit_group()"));
    assert!(compatibility.contains("pub fn wgmma_wait_group<const N: u32>()"));
    assert!(compatibility.contains("__wgmma_wait_group(N as u64);"));
    assert!(compatibility.contains("pub(crate) fn __wgmma_wait_group(_max_pending: u64)"));

    let raw = render_raw_abi(&catalog, "test-hash").unwrap();
    assert!(raw.contains("pub unsafe fn i0317()"));
    assert!(raw.contains("pub unsafe fn i0318()"));
    assert!(raw.contains("pub unsafe fn i0319(_arg0: u64)"));
    let raw_mod = render_raw_mod(&catalog, "test-hash");
    assert!(
        raw_mod.contains("pub use crate::__cuda_oxide_intrinsic_abi_v1::i0317 as wgmma_fence;")
    );
    assert!(
        raw_mod
            .contains("pub use crate::__cuda_oxide_intrinsic_abi_v1::i0319 as wgmma_wait_group;")
    );

    let dialect = render_dialect_wgmma_control(&catalog, "test-hash");
    for op in [
        "WgmmaFenceSyncAlignedOp",
        "WgmmaCommitGroupSyncAlignedOp",
        "WgmmaWaitGroupSyncAlignedOp",
    ] {
        assert!(dialect.contains(&format!("pub struct {op};")));
        assert!(dialect.contains(&format!("{op}::register(ctx);")));
    }
    assert!(dialect.contains("operand must be i64"));

    let importer = render_importer(&catalog, "test-hash");
    assert!(importer.contains("cuda_device::wgmma::wgmma_fence"));
    assert!(importer.contains("cuda_device::wgmma::wgmma_commit_group"));
    assert!(importer.contains("cuda_device::wgmma::__wgmma_wait_group"));
    assert!(importer.contains("wgmma_wait_group requires a compile-time constant"));
    assert!(importer.contains("WgmmaWaitGroupSyncAlignedOp::build(ctx, max_pending)"));

    let lowering = render_lowering(&catalog, "test-hash");
    assert!(lowering.contains("llvm_nvvm_wgmma_fence_sync_aligned"));
    assert!(lowering.contains("llvm__nvvm_dwgmma_dcommit_ugroup_dsync_daligned"));
    assert!(lowering.contains("llvm__nvvm_dwgmma_dwait_ugroup_dsync_daligned"));
    assert!(lowering.contains("IntrinsicBackend::LlvmNvptx"));
    assert!(lowering.contains("IntrinsicBackend::LibNvvm"));
    assert!(lowering.contains("wgmma.fence.sync.aligned;"));
    assert!(lowering.contains("wgmma.commit_group.sync.aligned;"));
    assert!(lowering.contains("wgmma.wait_group.sync.aligned $0;"));
    assert!(lowering.contains("\"n,~{memory}\""));

    let targets = render_targets(&catalog, "test-hash");
    assert!(
        targets.contains("pub enum GeneratedWgmmaControlMode { Fence, CommitGroup, WaitGroup }")
    );
    assert!(targets.contains("GeneratedIntrinsicVariant::WgmmaControl"));
    assert!(targets.contains("WgmmaFenceSyncAlignedOp>(operation, ctx).is_some()"));

    for record in wgmma_controls(&catalog) {
        let probe = render_probe(&catalog, record, "test-hash");
        assert!(probe.contains(&format!("@{}", llvm(record).symbol)));
        assert!(!probe.contains("asm sideeffect"));
        assert!(probe.contains("attributes #0 = { convergent }"));
    }

    let outputs = all_outputs(&catalog, "{}\n".into(), "test-hash").unwrap();
    assert!(outputs.contains_key(&PathBuf::from(
        "crates/cuda-device/src/generated/wgmma_control.rs"
    )));
    assert!(outputs.contains_key(&PathBuf::from(
        "crates/dialect-nvvm/src/ops/generated/wgmma_control.rs"
    )));
    let reference = render_reference(&catalog, "test-hash");
    assert!(reference.contains("## WGMMA-control contracts"));
}

#[test]
fn cluster_composites_use_generated_private_sreg_leaves() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::resolve(&repo_root).unwrap();
    validate_renderable(&catalog).unwrap();

    let leaves = render_compat_cluster_sreg(&catalog, "test-hash");
    assert_eq!(leaves.matches("pub(crate) fn __cluster_").count(), 6);
    for name in [
        "__cluster_idxX",
        "__cluster_idxY",
        "__cluster_idxZ",
        "__cluster_grid_dimX",
        "__cluster_grid_dimY",
        "__cluster_grid_dimZ",
    ] {
        assert!(leaves.contains(&format!("pub(crate) fn {name}() -> u32")));
    }
    assert!(!leaves.contains("pub fn cluster_ctaidX"));
    assert!(!leaves.contains("pub fn cluster_nctaidX"));

    let importer = render_importer(&catalog, "test-hash");
    assert!(importer.contains("cuda_device::cluster::__cluster_idxX"));
    assert!(importer.contains("cuda_device::cluster::__cluster_grid_dimZ"));

    let outputs = all_outputs(&catalog, "{}\n".into(), "test-hash").unwrap();
    assert!(outputs.contains_key(&PathBuf::from(
        "crates/cuda-device/src/generated/cluster_sreg.rs"
    )));
}

#[test]
fn cluster_barrier_rendering_owns_compatibility_sync_and_both_backends() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::resolve(&repo_root).unwrap();
    assert_eq!(cluster_barriers(&catalog).count(), 6);

    let dialect = render_dialect_cluster_barrier(&catalog, "test-hash");
    assert_eq!(dialect.matches("pub struct ClusterBarrierOp").count(), 1);
    assert_eq!(dialect.matches("pub struct ClusterSyncOp").count(), 1);
    assert!(dialect.contains("name = \"nvvm.cluster_sync\""));
    assert!(dialect.contains("ClusterSyncOp::register(ctx);"));

    let importer = render_importer(&catalog, "test-hash");
    assert!(importer.contains("\"cuda_device::cluster::cluster_sync\" =>"));
    assert!(importer.contains("ClusterBarrierModeAttr::ArriveAligned"));
    assert!(importer.contains("ClusterBarrierModeAttr::WaitAligned"));
    assert!(!importer.contains("ClusterSyncOp"));

    let lowering = render_lowering(&catalog, "test-hash");
    assert_eq!(
        lowering
            .matches("impl MirToLlvmConversion for ClusterSyncOp")
            .count(),
        1
    );
    assert!(lowering.contains("llvm_nvvm_barrier_cluster_arrive_aligned"));
    assert!(lowering.contains("llvm_nvvm_barrier_cluster_wait_aligned"));
    assert!(lowering.contains("barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;"));

    let reference = render_reference(&catalog, "test-hash");
    assert!(reference.contains("## Cluster-barrier contracts"));
    assert!(
        reference.contains(
            "`cuda_device::cluster::cluster_sync` is the generated compatibility operation"
        )
    );

    let targets = render_targets(&catalog, "test-hash");
    assert!(targets.contains(
            "GeneratedIntrinsicVariant::ClusterBarrier { mode: GeneratedClusterBarrierMode::ArriveRelaxedAligned }"
        ));
    assert!(targets.contains("Operation::get_op::<ClusterBarrierOp>"));
    assert!(targets.contains("ClusterBarrierModeAttr::WaitAligned"));
}

#[test]
fn cluster_memory_rendering_preserves_cluster_shared_addrspace_and_composite_load() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::resolve(&repo_root).unwrap();
    validate_renderable(&catalog).unwrap();
    let records = cluster_memory(&catalog).collect::<Vec<_>>();
    assert_eq!(records.len(), 2);

    let compatibility = render_compat_cluster_memory(&catalog, "test-hash");
    for signature in [
        "pub unsafe fn map_shared_rank<T>(local_ptr: *const T, target_rank: u32) -> *const T",
        "pub unsafe fn map_shared_rank_mut<T>(local_ptr: *mut T, target_rank: u32) -> *mut T",
        "pub unsafe fn dsmem_read_u32(local_ptr: *const u32, target_rank: u32) -> u32",
    ] {
        assert!(compatibility.contains(signature), "missing {signature}");
    }
    assert!(compatibility.contains("cluster-shared pointer in address space 7"));

    let dialect_mod = render_dialect_mod(&catalog, "test-hash");
    assert!(dialect_mod.contains("mod cluster_memory;"));
    assert!(dialect_mod.contains("pub use cluster_memory::*;"));
    assert!(dialect_mod.contains("cluster_memory::register(ctx);"));

    let dialect = render_dialect_cluster_memory(&catalog, "test-hash");
    assert!(dialect.contains("pub struct MapaSharedClusterOp"));
    assert!(dialect.contains("pub struct DsmemReadU32Op"));
    assert!(dialect.contains("source_ptr.pointer_kind()"));
    assert!(dialect.contains("address_space::CLUSTER_SHARED"));
    assert!(dialect.contains("MirPointerKind::RawConst | MirPointerKind::RawMut"));
    assert!(dialect.contains("result_ptr.pointer_kind() != source_ptr.pointer_kind()"));
    assert!(dialect.contains(
            "nvvm.mapa_shared_cluster requires a raw source pointer and must preserve its pointee, mutability, and raw kind while returning addrspace(7)"
        ));
    assert!(dialect.contains("nvvm.dsmem_read_u32 result must be u32"));
    assert!(dialect.contains("MapaSharedClusterOp::register(ctx);"));
    assert!(dialect.contains("DsmemReadU32Op::register(ctx);"));

    let importer = render_importer(&catalog, "test-hash");
    for path in [
        "cuda_device::cluster::map_shared_rank",
        "cuda_device::cluster::map_shared_rank_mut",
        "cuda_device::cluster::dsmem_read_u32",
    ] {
        assert!(importer.contains(path));
    }
    assert!(importer.contains("MapaSharedClusterOp::build(ctx, source, rank)"));
    assert!(importer.contains("DsmemReadU32Op::build(ctx, source, rank)"));

    let lowering = render_lowering(&catalog, "test-hash");
    assert!(lowering.contains("cast_to_shared_addrspace"));
    assert!(lowering.contains("llvm_ops::IntToPtrOp::new"));
    assert!(lowering.contains("llvm_types::PointerType::get(ctx, 7)"));
    assert!(lowering.contains("mapa.shared::cluster.u64 $0, $1, $2;"));
    assert!(lowering.contains(
            "{ .reg .u64 %mapped; mapa.shared::cluster.u64 %mapped, $1, $2; ld.shared::cluster.u32 $0, [%mapped]; }"
        ));
    assert!(lowering.contains("\"=l,l,r\""));
    assert!(lowering.contains("\"=r,l,r,~{memory}\""));

    let map = records
        .iter()
        .find(|record| record.id == "map_shared_rank")
        .unwrap();
    assert_eq!(map.llvm.as_ref().unwrap().results, ["shared_cluster_ptr"]);
    let map_probe = render_probe(&catalog, map, "test-hash");
    assert!(map_probe.contains("define ptr addrspace(7)"));
    assert!(map_probe.contains("inttoptr i64"));
    assert!(map_probe.contains("asm sideeffect"));
    assert!(!map_probe.contains("~{memory}"));
    assert!(map_probe.contains("attributes #0 = { convergent }"));

    let read = records
        .iter()
        .find(|record| record.id == "dsmem_read_u32")
        .unwrap();
    assert!(read.llvm.is_none());
    let read_probe = render_probe(&catalog, read, "test-hash");
    assert!(read_probe.contains("ld.shared::cluster.u32"));
    assert!(read_probe.contains("~{memory}"));
    assert!(read_probe.contains("attributes #0 = { convergent }"));

    let raw = render_raw_abi(&catalog, "test-hash").unwrap();
    assert!(raw.contains("pub unsafe fn i0320(_arg0: *const u8, _arg1: u32) -> *const u8"));
    assert!(raw.contains("pub unsafe fn i0321(_arg0: *const u32, _arg1: u32) -> u32"));
    assert!(raw.contains(
        "ordinary loads and stores compile to `ld.shared::cluster` and `st.shared::cluster`"
    ));

    let reference = render_reference(&catalog, "test-hash");
    assert!(reference.contains("## Cluster-memory contracts"));
    assert!(reference.contains("preserves LLVM 22's address-space-7 result"));
    assert!(reference.contains("has no one-to-one LLVM intrinsic"));

    let outputs = all_outputs(&catalog, "{}\n".into(), "test-hash").unwrap();
    assert!(outputs.contains_key(&PathBuf::from(
        "crates/cuda-device/src/generated/cluster_memory.rs"
    )));
    assert!(outputs.contains_key(&PathBuf::from(
        "crates/dialect-nvvm/src/ops/generated/cluster_memory.rs"
    )));

    let mut wrong_as = catalog.clone();
    wrong_as
        .intrinsics
        .iter_mut()
        .find(|record| record.id == "map_shared_rank")
        .unwrap()
        .llvm
        .as_mut()
        .unwrap()
        .results = vec!["shared_ptr".into()];
    assert!(validate_renderable(&wrong_as).is_err());

    let mut wrong_source = catalog;
    wrong_source
        .intrinsics
        .iter_mut()
        .find(|record| record.id == "dsmem_read_u32")
        .unwrap()
        .source = IntrinsicSource::LlvmImported {
        source_record: "invented".into(),
    };
    assert!(validate_renderable(&wrong_source).is_err());
}

#[test]
fn tma_rendering_preserves_api_and_injects_backend_defaults() {
    let catalog = catalog_with_tma();
    validate_renderable(&catalog).unwrap();

    let compatibility = render_compat_tma(&catalog, "test-hash");
    assert!(compatibility.contains(
            "pub unsafe fn cp_async_bulk_tensor_1d_g2s(dst: *mut u8, tensor_map: *const TmaDescriptor, coord0: i32, barrier: *mut Barrier)"
        ));
    assert!(compatibility.contains(
            "pub unsafe fn cp_async_bulk_tensor_2d_g2s_multicast(dst: *mut u8, tensor_map: *const TmaDescriptor, coord0: i32, coord1: i32, barrier: *mut Barrier, cta_mask: u16)"
        ));
    assert!(compatibility.contains(
            "pub unsafe fn cp_async_bulk_tensor_5d_s2g(src: *const u8, tensor_map: *const TmaDescriptor, coord0: i32, coord1: i32, coord2: i32, coord3: i32, coord4: i32)"
        ));
    assert!(compatibility.contains("pub fn cp_async_bulk_commit_group()"));
    assert!(compatibility.contains("pub fn cp_async_bulk_wait_group(n: u32)"));
    assert!(compatibility.contains(
            "pub unsafe fn cp_async_bulk_tensor_reduce_add_tile_2d(src: *const u8, tensor_map: *const TmaDescriptor, coord0: i32, coord1: i32)"
        ));
    assert!(compatibility.contains(
            "pub unsafe fn cp_async_bulk_tensor_reduce_xor_im2col_5d(src: *const u8, tensor_map: *const TmaDescriptor, coord0: i32, coord1: i32, coord2: i32, coord3: i32, coord4: i32)"
        ));
    for dimensions in 1..=5 {
        assert!(compatibility.contains(&format!(
            "pub unsafe fn cp_async_bulk_prefetch_tensor_{dimensions}d_l2("
        )));
        assert!(compatibility.contains(&format!(
            "pub unsafe fn cp_async_bulk_prefetch_tensor_{dimensions}d_l2_cache_hint("
        )));
    }
    assert!(compatibility.contains("pub unsafe fn cp_async_bulk_prefetch_tensor_gather4_2d_l2("));
    assert!(
        compatibility
            .contains("pub unsafe fn cp_async_bulk_prefetch_tensor_gather4_2d_l2_cache_hint(")
    );
    assert!(compatibility.contains("pub unsafe fn tensormap_replace_swizzle_atomicity("));
    assert!(compatibility.contains("pub unsafe fn fence_proxy_tensormap_generic_acquire_system("));
    assert!(compatibility.contains("pub fn fence_proxy_tensormap_generic_release_system()"));

    let dialect = render_dialect_tma(&catalog, "test-hash");
    assert_eq!(dialect.matches("pub struct ").count(), 111);
    assert_eq!(dialect.matches("NResultsInterface<0>").count(), 111);
    assert!(dialect.contains("NOpdsInterface<10>"));
    assert!(dialect.contains("CpAsyncBulkWaitGroupReadOp::register(ctx)"));
    assert!(dialect.contains("CpAsyncBulkPrefetchTensorGather4TwoDimensionalL2Op::register(ctx)"));
    assert!(
        dialect
            .contains("CpAsyncBulkPrefetchTensorGather4TwoDimensionalL2CacheHintOp::register(ctx)")
    );
    assert!(dialect.contains("ReplaceTensorMapSwizzleAtomicityOp::register(ctx)"));
    assert!(dialect.contains("FenceProxyTensorMapGenericReleaseSystemOp::register(ctx)"));

    let dialect_mod = render_dialect_mod(&catalog, "test-hash");
    assert!(dialect_mod.contains("mod tma;"));
    assert!(dialect_mod.contains("pub use tma::*;"));
    assert!(dialect_mod.contains("tma::register(ctx);"));

    let importer = render_importer(&catalog, "test-hash");
    assert!(importer.contains("cuda_device::tma::cp_async_bulk_tensor_1d_g2s"));
    assert!(importer.contains("super::tma::emit_tma_g2s("));
    assert!(importer.contains("super::tma::emit_tma_g2s_multicast_cg2("));
    assert!(importer.contains("super::tma::emit_tma_s2g("));
    assert!(importer.contains("TMA wait-group count must be a compile-time constant"));
    assert!(importer.contains("require_arity(name, args.len(), 1, &loc)?;"));
    assert!(importer.contains("args.first(), Some(mir::Operand::Constant(_))"));
    assert!(importer.contains("\"v1:i0328\""));
    assert!(importer.contains("cp_async_bulk_tensor_reduce_add_tile_2d"));
    assert!(
        importer.contains(
            "dialect_nvvm::ops::CpAsyncBulkTensorReduceAddTile2dOp::get_concrete_op_info()"
        )
    );

    let lowering = render_lowering(&catalog, "test-hash");
    assert!(lowering.contains("CpAsyncBulkTensorG2sTile1dOp"));
    for tma_import in [
        "tma::convert_control",
        "tma::convert_g2s",
        "tma::convert_g2s_multicast_cg2",
        "tma::convert_prefetch_tensormap",
        "tma::convert_prefetch_tile",
        "tma::convert_reduce_s2g",
        "tma::convert_s2g",
        "tma::convert_tensormap_fence",
        "tma::convert_tensormap_replace",
        "tma::PrefetchTileConfig",
        "tma::ReduceConfig",
    ] {
        assert!(lowering.contains(tma_import), "{tma_import}");
    }
    assert!(
        lowering
            .contains("convert_g2s(ctx, rewriter, self.get_operation(), operands_info, 5, false)")
    );
    assert!(
        lowering.contains("convert_s2g(ctx, rewriter, self.get_operation(), operands_info, 5)")
    );
    assert!(lowering.contains(
            "convert_control(ctx, rewriter, self.get_operation(), operands_info, \"commit_group\", \"llvm_nvvm_cp_async_bulk_commit_group\")"
        ));
    assert!(lowering.contains(
            "convert_reduce_s2g(ctx, rewriter, self.get_operation(), operands_info, ReduceConfig::new(2, \"add\", \"tile\", \"llvm_nvvm_cp_async_bulk_tensor_reduce_add_tile_2d\"))"
        ));
    assert!(lowering.contains(
            "convert_prefetch_tile(ctx, rewriter, self.get_operation(), operands_info, PrefetchTileConfig::new(5, false, false, \"llvm_nvvm_cp_async_bulk_tensor_prefetch_tile_5d\"))"
        ));
    assert!(lowering.contains(
            "convert_tensormap_replace(ctx, rewriter, self.get_operation(), operands_info, \"llvm_nvvm_tensormap_replace_global_address\", \"global_address\", \"address\", false, false)"
        ));
    let release_system = tma_intrinsics(&catalog)
        .find(|record| record.id == "fence_proxy_tensormap_generic_release_system")
        .unwrap();
    assert!(lowering.contains(&format!(
            "convert_tensormap_fence(ctx, rewriter, self.get_operation(), operands_info, {:?}, false, \"sys\")",
            release_system.resolved_llvm_identifier()
        )));

    let g2s = tma_intrinsics(&catalog)
        .find(|record| record.id == "cp_async_bulk_tensor_2d_g2s_multicast_cg2")
        .unwrap();
    let g2s_probe = render_probe(&catalog, g2s, "test-hash");
    assert!(g2s_probe.contains("i16 %cta_mask, i64 0, i1 true, i1 false, i32 2"));
    assert!(g2s_probe.contains("addrspacecast ptr %dst_generic to ptr addrspace(7)"));

    let s2g = tma_intrinsics(&catalog)
        .find(|record| record.id == "cp_async_bulk_tensor_2d_s2g")
        .unwrap();
    let s2g_probe = render_probe(&catalog, s2g, "test-hash");
    assert!(s2g_probe.contains("ptr %tensor_map, i32 %coord0, i32 %coord1, i64 0, i1 false"));

    let reduce = tma_intrinsics(&catalog)
        .find(|record| record.id == "cp_async_bulk_tensor_reduce_add_tile_2d")
        .unwrap();
    let reduce_probe = render_probe(&catalog, reduce, "test-hash");
    assert!(reduce_probe.contains(
            "declare void @llvm.nvvm.cp.async.bulk.tensor.reduce.add.tile.2d(ptr addrspace(3), ptr, i32, i32, i64, i1)"
        ));
    assert!(reduce_probe.contains(
            "call void @llvm.nvvm.cp.async.bulk.tensor.reduce.add.tile.2d(ptr addrspace(3) %src, ptr %tensor_map, i32 %coord0, i32 %coord1, i64 0, i1 false)"
        ));

    for stem in [
        "cp_async_bulk_prefetch_tensor_1d_l2",
        "cp_async_bulk_prefetch_tensor_2d_l2",
        "cp_async_bulk_prefetch_tensor_3d_l2",
        "cp_async_bulk_prefetch_tensor_4d_l2",
        "cp_async_bulk_prefetch_tensor_5d_l2",
        "cp_async_bulk_prefetch_tensor_gather4_2d_l2",
    ] {
        let prefetch_plain = tma_intrinsics(&catalog)
            .find(|record| record.id == stem)
            .unwrap();
        let prefetch_plain_probe = render_probe(&catalog, prefetch_plain, "test-hash");
        assert!(prefetch_plain_probe.contains("i64 0, i1 false"));
        assert!(!prefetch_plain_probe.contains("%cache_hint"));

        let cache_hint_id = format!("{stem}_cache_hint");
        let prefetch_cache_hint = tma_intrinsics(&catalog)
            .find(|record| record.id == cache_hint_id)
            .unwrap();
        let prefetch_cache_hint_probe = render_probe(&catalog, prefetch_cache_hint, "test-hash");
        assert!(prefetch_cache_hint_probe.contains("i64 %cache_hint"));
        assert!(prefetch_cache_hint_probe.contains("i64 %cache_hint, i1 true"));
    }

    let outputs = all_outputs(&catalog, "{}\n".into(), "test-hash").unwrap();
    assert!(outputs.contains_key(&PathBuf::from("crates/cuda-device/src/generated/tma.rs")));
    assert!(outputs.contains_key(&PathBuf::from(
        "crates/dialect-nvvm/src/ops/generated/tma.rs"
    )));

    let mut wrong_adapter = catalog;
    wrong_adapter
        .intrinsics
        .iter_mut()
        .find(|record| record.id == "cp_async_bulk_tensor_1d_g2s")
        .unwrap()
        .tma
        .as_mut()
        .unwrap()
        .adapter = TmaAdapter::NoOperands;
    assert!(validate_renderable(&wrong_adapter).is_err());
}

#[test]
fn tcgen05_rendering_preserves_api_and_inline_ptx_routes() {
    let catalog = catalog_with_tcgen05();
    validate_renderable(&catalog).unwrap();

    let compatibility = render_compat_tcgen05(&catalog, "test-hash");
    assert!(
        compatibility
            .contains("pub unsafe fn tcgen05_alloc(dst_smem: *mut u32, n_cols: u32) -> ()")
    );
    assert!(compatibility.contains(
            "pub unsafe fn tcgen05_mma_ws_f16(d_tmem: u32, a_tmem: u32, a_desc: u64, b_desc: u64, idesc: u32, enable_d: bool) -> ()"
        ));
    for alias in ["e4m3", "e5m2", "e2m3", "e3m2", "e2m1"] {
        assert!(compatibility.contains(&format!(
                "pub unsafe fn tcgen05_mma_{alias}(d_tmem: u32, a_desc: u64, b_desc: u64, idesc: u32, enable_d: bool)"
            )));
    }
    assert!(
        compatibility
            .contains("pub unsafe fn tcgen05_ld_16x256b_x8_pure(tmem_addr: u32) -> TmemF32x32")
    );
    assert!(compatibility.contains("pub fn tcgen05_load_wait() -> ()"));
    assert!(compatibility.contains("pub fn tcgen05_relinquish_alloc_permit() -> ()"));
    assert!(compatibility.contains("pub fn tcgen05_relinquish_alloc_permit_cg2() -> ()"));
    assert!(
        compatibility.contains(
            "One full warp must execute this instruction uniformly with the same operands."
        )
    );
    assert!(compatibility.contains(
            "One full warp in each peer CTA must execute this instruction; lanes within each warp must execute uniformly with the same operands."
        ));
    assert!(compatibility.contains(
        "pub unsafe fn tcgen05_commit_multicast_cg2(mbar: *mut u64, cta_mask: u16) -> ()"
    ));
    assert!(compatibility.contains(
        "pub unsafe fn tcgen05_cp_128x128b_b4x16_p64(tmem_addr: u32, smem_desc: u64) -> ()"
    ));
    assert!(compatibility.contains(
        "pub unsafe fn tcgen05_cp_64x128b_warpx2_02_13_cg2(tmem_addr: u32, smem_desc: u64) -> ()"
    ));
    assert!(
        compatibility.contains("pub unsafe fn tcgen05_ld_16x64b_x1_raw(tmem_addr: u32) -> u32")
    );
    assert!(
        compatibility
            .contains("pub unsafe fn tcgen05_ld_16x64b_x2_raw(tmem_addr: u32) -> CuSimd<u32, 2>")
    );
    assert!(compatibility.contains(
        "pub unsafe fn tcgen05_ld_32x32b_x128_pack16(tmem_addr: u32) -> CuSimd<u32, 128>"
    ));
    assert!(
        compatibility
            .contains("pub unsafe fn tcgen05_st_16x64b_x1_raw(tmem_addr: u32, data: u32) -> ()")
    );
    assert!(compatibility.contains(
        "pub unsafe fn tcgen05_st_16x64b_x2_raw(tmem_addr: u32, data: CuSimd<u32, 2>) -> ()"
    ));
    assert!(compatibility.contains(
            "pub unsafe fn tcgen05_st_32x32b_x128_unpack16(tmem_addr: u32, data: CuSimd<u32, 128>) -> ()"
        ));
    assert!(compatibility.contains(
            "pub unsafe fn tcgen05_ld_16x32bx2_x1_raw<const HALF_SPLIT_OFFSET: i32>(tmem_addr: u32) -> u32"
        ));
    assert!(compatibility.contains(
            "pub(crate) unsafe fn __tcgen05_ld_16x32bx2_x1_raw(_tmem_addr: u32, _half_split_offset: i64) -> u32"
        ));
    assert!(
        compatibility.contains(
            "unsafe { __tcgen05_ld_16x32bx2_x1_raw(tmem_addr, HALF_SPLIT_OFFSET as i64) }"
        )
    );
    assert!(compatibility.contains(
            "pub unsafe fn tcgen05_st_16x32bx2_x128_unpack16<const HALF_SPLIT_OFFSET: i32>(tmem_addr: u32, data: CuSimd<u32, 128>)"
        ));
    assert!(compatibility.contains(
            "pub(crate) unsafe fn __tcgen05_st_16x32bx2_x128_unpack16(_tmem_addr: u32, _half_split_offset: i64, _data: CuSimd<u32, 128>)"
        ));
    assert!(
        compatibility.contains(
            "pub unsafe fn tcgen05_commit_multicast(mbar: *mut u64, cta_mask: u16) -> ()"
        )
    );
    assert!(compatibility.contains("pub unsafe fn tcgen05_shift_down(tmem_addr: u32) -> ()"));
    assert!(compatibility.contains("pub unsafe fn tcgen05_shift_down_cg2(tmem_addr: u32) -> ()"));
    assert_eq!(
        compatibility
            .matches("/// One thread in the CTA issues this instruction.\n")
            .count(),
        4
    );
    assert_eq!(
            compatibility
                .matches(
                    "/// One thread in the CTA pair issues this instruction; the peer CTA must be active and must not have exited.\n",
                )
                .count(),
            4
        );
    assert_eq!(
            compatibility
                .matches(
                    "/// The same thread that issued the tracked asynchronous tcgen05 operations must issue this commit.\n",
                )
                .count(),
            6
        );
    assert_eq!(
            compatibility
                .matches(
                    "/// Completion must be tracked by a matching commit from that same thread and observed through the selected mbarrier before relying on shifted data.\n",
                )
                .count(),
            2
        );
    assert_eq!(
            compatibility
                .matches(
                    "/// All tcgen05 operations in the kernel must use the same CTA-group mode.\n",
                )
                .count(),
            233
        );
    assert!(compatibility.contains("/// `KIND` is 0=f16, 1=tf32, 2=f8f6f4, or 3=i8."));
    assert!(compatibility.contains(
        "/// `CTA_GROUP` is 1 or 2. `COLLECTOR_A` is 0=discard, 1=lastuse, 2=fill, or 3=use."
    ));
    assert!(compatibility.contains(
        "/// `B_BUFFER` is 0 through 3. `B_USAGE` is 0=discard, 1=lastuse, 2=fill, or 3=use."
    ));
    assert_eq!(
        compatibility
            .matches("/// `legacy_a_desc` is kept for compatibility; tensor A uses `a_tmem`.\n",)
            .count(),
        5
    );
    assert_eq!(
        compatibility
            .matches("/// This uses kind f8f6f4, CTA group 1, and collector a::discard.\n",)
            .count(),
        5
    );
    assert!(!compatibility.contains("tcgen05_mma_ws_f16_with_collector"));

    let raw = render_raw_abi(&catalog, "test-hash").unwrap();
    assert!(raw.contains("pub unsafe fn i0578(_arg0: u32, _arg1: u64) -> ()"));
    assert!(raw.contains("pub unsafe fn i0611(_arg0: u32, _arg1: u64) -> ()"));
    assert!(raw.contains("pub unsafe fn i0612(_arg0: u32) -> u32"));
    assert!(raw.contains("pub unsafe fn i0614(_arg0: u32) -> [u32; 2]"));
    assert!(raw.contains("pub unsafe fn i0669(_arg0: u32) -> [u32; 128]"));
    assert!(raw.contains("pub unsafe fn i0670(_arg0: u32, _arg1: u32) -> ()"));
    assert!(raw.contains("pub unsafe fn i0672(_arg0: u32, _arg1: [u32; 2]) -> ()"));
    assert!(raw.contains("pub unsafe fn i0727(_arg0: u32, _arg1: [u32; 128]) -> ()"));
    assert!(raw.contains("pub unsafe fn i0728(_arg0: u32, _arg1: i64) -> u32"));
    assert!(raw.contains("pub unsafe fn i0743(_arg0: u32, _arg1: i64) -> [u32; 128]"));
    assert!(raw.contains("pub unsafe fn i0759(_arg0: u32, _arg1: i64, _arg2: [u32; 128]) -> ()"));
    assert!(raw.contains("pub unsafe fn i0760(_arg0: *mut u64, _arg1: u16) -> ()"));
    assert!(raw.contains("pub unsafe fn i0761(_arg0: u32) -> ()"));
    assert!(raw.contains("pub unsafe fn i0762(_arg0: u32) -> ()"));
    let mma = raw.find("pub unsafe fn i0763").unwrap();
    assert!(raw[..mma].ends_with("#[allow(clippy::too_many_arguments)]\n#[inline(never)]\n"));
    assert!(raw.contains("lane component must be a multiple of 32"));
    assert!(raw.contains("`_arg0` must name a live tensor-memory allocation"));
    assert_eq!(
        raw.matches("/// One thread in the CTA issues this instruction.\n")
            .count(),
        4
    );
    assert_eq!(
            raw.matches(
                "/// One thread in the CTA pair issues this instruction; the peer CTA must be active and must not have exited.\n",
            )
            .count(),
            4
        );
    assert_eq!(
            raw.matches(
                "/// The same thread that issued the tracked asynchronous tcgen05 operations must issue this commit.\n",
            )
            .count(),
            6
        );
    assert_eq!(
            raw.matches(
                "/// Completion must be tracked by a matching commit from that same thread and observed through the selected mbarrier before relying on shifted data.\n",
            )
            .count(),
            2
        );
    assert_eq!(
            raw.matches(
                "/// All tcgen05 operations in the kernel must use the same CTA-group mode.\n",
            )
            .count(),
            233
        );
    assert!(raw.contains("pub fn i0345() -> ()"));
    assert!(raw.contains("pub fn i0357() -> ()"));
    assert!(raw.contains("pub fn i0358() -> ()"));
    assert!(raw.contains("pub fn i0361() -> ()"));
    assert!(raw.contains("Complete the matching tensor-memory load wait"));
    assert!(raw.contains("Complete the matching tensor-memory store wait"));
    let raw_mod = render_raw_mod(&catalog, "test-hash");
    assert!(raw_mod.contains(
        "pub use crate::__cuda_oxide_intrinsic_abi_v1::i0578 as tcgen05_cp_128x128b_b4x16_p64;"
    ));
    assert!(raw_mod.contains(
            "pub use crate::__cuda_oxide_intrinsic_abi_v1::i0611 as tcgen05_cp_64x128b_warpx2_02_13_cg2;"
        ));
    assert!(raw_mod.contains(
        "pub use crate::__cuda_oxide_intrinsic_abi_v1::i0727 as tcgen05_st_32x32b_x128_unpack16;"
    ));
    assert!(raw_mod.contains(
        "pub use crate::__cuda_oxide_intrinsic_abi_v1::i0728 as tcgen05_ld_16x32bx2_x1_raw;"
    ));
    assert!(raw_mod.contains(
        "pub use crate::__cuda_oxide_intrinsic_abi_v1::i0759 as tcgen05_st_16x32bx2_x128_unpack16;"
    ));
    assert!(raw_mod.contains(
        "pub use crate::__cuda_oxide_intrinsic_abi_v1::i0760 as tcgen05_commit_multicast;"
    ));
    assert!(
        raw_mod
            .contains("pub use crate::__cuda_oxide_intrinsic_abi_v1::i0761 as tcgen05_shift_down;")
    );
    assert!(raw_mod.contains(
        "pub use crate::__cuda_oxide_intrinsic_abi_v1::i0762 as tcgen05_shift_down_cg2;"
    ));

    let dialect = render_dialect_tcgen05(&catalog, "test-hash");
    assert_eq!(dialect.matches("pub struct Tcgen05").count(), 210);
    assert_eq!(dialect.matches("impl Verify for Tcgen05").count(), 210);
    assert_eq!(dialect.matches("::register(ctx)").count(), 216);
    assert_eq!(dialect.matches("            Some(1),").count(), 32);
    assert_eq!(dialect.matches("verifier = \"succ\"").count(), 6);
    assert!(dialect.contains("Operation::get_op::<MirConstantOp>"));
    assert!(dialect.contains("Operation::get_op::<ConstantOp>"));
    assert!(dialect.contains("&[(Tcgen05Carrier::I32, 1), (Tcgen05Carrier::I64, 1)],"));
    assert!(dialect.contains(
        "&[(Tcgen05Carrier::I32, 1), (Tcgen05Carrier::I64, 1), (Tcgen05Carrier::I32, 128)],"
    ));
    assert!(dialect.contains("pub struct Tcgen05Ld16x64bX1RawOp"));
    assert!(dialect.contains("pub struct Tcgen05Ld32x32bX128Pack16Op"));
    assert!(dialect.contains("pub struct Tcgen05St16x64bX1RawOp"));
    assert!(dialect.contains("pub struct Tcgen05St32x32bX128Unpack16Op"));
    assert!(dialect.contains("pub struct Tcgen05Ld16x32bx2X1RawOp"));
    assert!(dialect.contains("pub struct Tcgen05St16x32bx2X128Unpack16Op"));
    assert!(dialect.contains("pub struct Tcgen05CommitMulticastOp"));
    assert!(dialect.contains("pub struct Tcgen05ShiftDownOp"));
    assert!(dialect.contains("pub struct Tcgen05ShiftDownCg2Op"));
    assert!(dialect.contains("NResultsInterface<128>"));
    assert!(dialect.contains("NResultsInterface<32>"));
    assert!(dialect.contains("NResultsInterface<4>"));
    assert!(dialect.contains("NOpdsInterface<6>"));

    let dialect_mod = render_dialect_mod(&catalog, "test-hash");
    assert!(dialect_mod.contains("mod tcgen05;"));
    assert!(dialect_mod.contains("pub use tcgen05::*;"));
    assert!(dialect_mod.contains("tcgen05::register(ctx);"));

    let importer = render_importer(&catalog, "test-hash");
    assert!(importer.contains("cuda_device::tcgen05::tcgen05_alloc"));
    assert!(importer.contains("Tcgen05AllocOp::get_concrete_op_info()"));
    assert!(importer.contains("Tcgen05Ld16x256bX8PureOp::get_concrete_op_info()"));
    assert!(importer.contains("MirConstructStructOp::get_concrete_op_info()"));
    assert!(importer.contains("Tcgen05Ld16x64bX1RawOp::get_concrete_op_info()"));
    assert!(importer.contains("Tcgen05Ld32x32bX128Pack16Op::get_concrete_op_info()"));
    assert!(importer.contains("Tcgen05St16x64bX1RawOp::get_concrete_op_info()"));
    assert!(importer.contains("Tcgen05St32x32bX128Unpack16Op::get_concrete_op_info()"));
    assert!(importer.contains("import_generated_tcgen05_store_operands("));
    assert!(importer.contains("downcast_ref::<MirStructType>()"));
    assert!(
        importer.contains("generated tcgen05 store data must contain {expected_len} u32 registers")
    );
    assert!(importer.contains("destination_ty == array_result.get_type(ctx)"));
    assert!(!importer.contains("super::tcgen05::emit_"));
    assert!(importer.contains("\"v1:i0343\""));
    assert!(importer.contains("\"v1:i0366\""));
    assert!(importer.contains("cuda_device::tcgen05::tcgen05_cp_128x128b_b4x16_p64"));
    assert!(importer.contains("Tcgen05Cp128x128bB4x16P64Op::get_concrete_op_info()"));
    assert!(importer.contains("\"v1:i0578\""));
    assert!(importer.contains("\"v1:i0611\""));
    assert!(importer.contains("\"v1:i0670\""));
    assert!(importer.contains("\"v1:i0727\""));
    assert!(importer.contains("cuda_device::tcgen05::__tcgen05_ld_16x32bx2_x1_raw"));
    assert!(importer.contains("cuda_device::tcgen05::__tcgen05_st_16x32bx2_x128_unpack16"));
    assert!(
        importer.contains("tcgen05 16x32bx2 half-split offset must be a compile-time constant")
    );
    assert!(importer.contains("half-split offset must lower to a constant"));
    assert!(importer.contains("args, 128, true, block_ptr"));
    assert!(importer.contains("\"v1:i0728\""));
    assert!(importer.contains("\"v1:i0759\""));
    assert!(importer.contains("\"v1:i0760\""));
    assert!(importer.contains("\"v1:i0761\""));
    assert!(importer.contains("\"v1:i0762\""));

    let lowering = render_lowering(&catalog, "test-hash");
    assert!(!lowering.contains("use crate::convert::intrinsics::tcgen05;"));
    assert!(lowering.contains("impl MirToLlvmConversion for Tcgen05AllocOp"));
    assert!(
        lowering
            .contains("tcgen05.mma.ws.cta_group::1.kind::f16 [$0], [$1], $3, $4, %enable_pred;")
    );
    assert!(!lowering.contains(".kind::bf16"));
    assert!(lowering.contains("convert_generated_tcgen05_load("));
    assert!(lowering.contains("tcgen05.ld.sync.aligned.16x64b.x1.b32 {$0}, [$1];"));
    assert!(lowering.contains("tcgen05.ld.sync.aligned.16x64b.x1.pack::16b.b32 {$0}, [$1];"));
    assert!(lowering.contains("\"=r,r,~{memory}\""));
    assert!(lowering.contains(
        "convert_generated_tcgen05_load(ctx, rewriter, self.get_operation(), 1, 1, true"
    ));
    assert!(lowering.contains("convert_generated_tcgen05_void("));
    assert!(lowering.contains("tcgen05.st.sync.aligned.16x64b.x1.b32 [$0], {$1};"));
    assert!(lowering.contains("tcgen05.st.sync.aligned.32x32b.x128.unpack::16b.b32 [$0], {$1,$2"));
    assert!(lowering.contains("\"r,r,~{memory}\""));
    assert!(lowering.contains("tcgen05.ld.sync.aligned.16x32bx2.x1.b32 {$0}, [$1], $2;"));
    assert!(lowering.contains("tcgen05.st.sync.aligned.16x32bx2.x1.b32 [$0], $1, {$2};"));
    assert!(lowering.contains("\"=r,r,n,~{memory}\""));
    assert!(lowering.contains("\"r,n,r,~{memory}\""));
    assert!(lowering.contains(
        "convert_generated_tcgen05_load(ctx, rewriter, self.get_operation(), 2, 1, true"
    ));
    assert!(lowering.contains("builtin::types::{FP32Type, IntegerType, Signedness}"));
    assert!(lowering.contains("ops as llvm_ops"));
    assert!(lowering.contains("inserter::Inserter"));
    assert!(lowering.contains("tcgen05.cp.cta_group::1.128x128b.b8x16.b4x16_p64 [$0], $1;"));
    assert!(lowering.contains("tcgen05.cp.cta_group::1.32x128b.warpx4 [$0], $1;"));
    assert!(
        lowering
            .contains("tcgen05.cp.cta_group::2.64x128b.warpx2::01_23.b8x16.b6x16_p32 [$0], $1;")
    );
    assert!(lowering.contains("tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [$0], $1;"));
    assert!(lowering.contains("\"r,l,~{memory}\""));
    assert!(lowering.contains(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [$0], $1;"
        ));
    assert!(lowering.contains("tcgen05.shift.cta_group::1.down [$0];"));
    assert!(lowering.contains("tcgen05.shift.cta_group::2.down [$0];"));
    assert!(lowering.contains("\"r,h,~{memory}\""));

    let bf16 = tcgen05_intrinsics(&catalog)
        .find(|record| record.id == "tcgen05_mma_ws_bf16")
        .unwrap();
    let bf16_probe = render_probe(&catalog, bf16, "test-hash");
    assert!(bf16_probe.contains("route: inline PTX"));
    assert!(bf16_probe.contains(".kind::f16 [$0]"));
    assert!(!bf16_probe.contains(".kind::bf16"));

    let base_commit = tcgen05_intrinsics(&catalog)
        .find(|record| record.id == "tcgen05_commit")
        .unwrap();
    let base_commit_probe = render_probe(&catalog, base_commit, "test-hash");
    assert!(base_commit_probe.contains("mbarrier::arrive::one.b64"));
    assert!(!base_commit_probe.contains("one.shared::cluster.b64"));

    let shared_commit = tcgen05_intrinsics(&catalog)
        .find(|record| record.id == "tcgen05_commit_shared_cluster")
        .unwrap();
    let shared_commit_probe = render_probe(&catalog, shared_commit, "test-hash");
    assert!(shared_commit_probe.contains("one.shared::cluster.b64"));

    for (id, spelling) in [
        (
            "tcgen05_commit_multicast",
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [$0], $1;",
        ),
        (
            "tcgen05_shift_down",
            "tcgen05.shift.cta_group::1.down [$0];",
        ),
        (
            "tcgen05_shift_down_cg2",
            "tcgen05.shift.cta_group::2.down [$0];",
        ),
    ] {
        let record = tcgen05_intrinsics(&catalog)
            .find(|record| record.id == id)
            .unwrap();
        let probe = render_probe(&catalog, record, "test-hash");
        assert!(probe.contains(spelling));
        assert!(probe.contains("asm sideeffect"));
        assert!(probe.contains("attributes #0 = { convergent }"));
    }

    for (id, spelling) in [
        (
            "tcgen05_cp_128x128b_b4x16_p64",
            "tcgen05.cp.cta_group::1.128x128b.b8x16.b4x16_p64",
        ),
        (
            "tcgen05_cp_32x128b_warpx4",
            "tcgen05.cp.cta_group::1.32x128b.warpx4",
        ),
        (
            "tcgen05_cp_64x128b_warpx2_01_23_b6x16_p32_cg2",
            "tcgen05.cp.cta_group::2.64x128b.warpx2::01_23.b8x16.b6x16_p32",
        ),
        (
            "tcgen05_cp_64x128b_warpx2_02_13",
            "tcgen05.cp.cta_group::1.64x128b.warpx2::02_13",
        ),
    ] {
        let record = tcgen05_intrinsics(&catalog)
            .find(|record| record.id == id)
            .unwrap();
        let probe = render_probe(&catalog, record, "test-hash");
        assert!(probe.contains(spelling));
        assert!(probe.contains("asm sideeffect"));
        assert!(probe.contains("\"r,l,~{memory}\""));
        assert!(probe.contains("attributes #0 = { convergent }"));
    }

    let scalar_load = tcgen05_intrinsics(&catalog)
        .find(|record| record.id == "tcgen05_ld_16x64b_x1_raw")
        .unwrap();
    let scalar_load_probe = render_probe(&catalog, scalar_load, "test-hash");
    assert!(scalar_load_probe.contains("%result = call i32 asm sideeffect"));
    assert!(scalar_load_probe.contains("tcgen05.ld.sync.aligned.16x64b.x1.b32"));
    assert!(scalar_load_probe.contains("\"=r,r,~{memory}\""));

    let packed_load = tcgen05_intrinsics(&catalog)
        .find(|record| record.id == "tcgen05_ld_32x32b_x128_pack16")
        .unwrap();
    let packed_load_probe = render_probe(&catalog, packed_load, "test-hash");
    assert!(packed_load_probe.contains("tcgen05.ld.sync.aligned.32x32b.x128.pack::16b.b32"));
    assert!(packed_load_probe.contains("{ i32, i32"));

    let scalar_store = tcgen05_intrinsics(&catalog)
        .find(|record| record.id == "tcgen05_st_16x64b_x1_raw")
        .unwrap();
    let scalar_store_probe = render_probe(&catalog, scalar_store, "test-hash");
    assert!(scalar_store_probe.contains("call void asm sideeffect"));
    assert!(scalar_store_probe.contains("tcgen05.st.sync.aligned.16x64b.x1.b32"));
    assert!(scalar_store_probe.contains("\"r,r,~{memory}\""));

    let unpacked_store = tcgen05_intrinsics(&catalog)
        .find(|record| record.id == "tcgen05_st_32x32b_x128_unpack16")
        .unwrap();
    let unpacked_store_probe = render_probe(&catalog, unpacked_store, "test-hash");
    assert!(unpacked_store_probe.contains("tcgen05.st.sync.aligned.32x32b.x128.unpack::16b.b32"));
    assert!(unpacked_store_probe.contains("i32 %d127"));

    let offset_load = tcgen05_intrinsics(&catalog)
        .find(|record| record.id == "tcgen05_ld_16x32bx2_x1_pack16")
        .unwrap();
    let offset_load_probe = render_probe(&catalog, offset_load, "test-hash");
    assert!(offset_load_probe.contains("i64 16"));
    assert!(offset_load_probe.contains("[$1], $2;"));
    assert!(offset_load_probe.contains("\"=r,r,n,~{memory}\""));

    let offset_store = tcgen05_intrinsics(&catalog)
        .find(|record| record.id == "tcgen05_st_16x32bx2_x1_unpack16")
        .unwrap();
    let offset_store_probe = render_probe(&catalog, offset_store, "test-hash");
    assert!(offset_store_probe.contains("i64 16"));
    assert!(offset_store_probe.contains("[$0], $1, {$2};"));
    assert!(offset_store_probe.contains("\"r,n,r,~{memory}\""));

    let target = render_targets(&catalog, "test-hash");
    assert!(target.contains("\"v1:i0343\""));
    assert!(target.contains("\"v1:i0578\""));
    assert!(target.contains("\"v1:i0611\""));
    assert!(target.contains("\"v1:i0727\""));
    assert!(target.contains("\"v1:i0728\""));
    assert!(target.contains("\"v1:i0759\""));
    assert!(target.contains(
        "Tcgen05MmaBUsageAttr, Tcgen05MmaCollectorAAttr, Tcgen05MmaCtaGroupAttr, Tcgen05MmaFormAttr"
    ));
    assert!(target.contains(
            "GeneratedHardwareTarget::AnyOf(&[GeneratedHardwareAlternative::ExactArchitecture(100), GeneratedHardwareAlternative::ExactArchitecture(101), GeneratedHardwareAlternative::ExactArchitecture(103), GeneratedHardwareAlternative::ExactArchitecture(110)])"
        ));
    assert!(target.contains(
            "GeneratedHardwareTarget::AnyOf(&[GeneratedHardwareAlternative::ExactArchitecture(100), GeneratedHardwareAlternative::ExactArchitecture(103), GeneratedHardwareAlternative::ExactArchitecture(110)])"
        ));

    let reference = render_reference(&catalog, "test-hash");
    assert!(reference.contains("- `tcgen05_st_16x64b_x1_raw`: runtime `unexecuted`"));
    assert!(!reference.contains("- `tcgen05_st_16x64b_x1_raw`: runtime `not recorded`"));

    let outputs = all_outputs(&catalog, "{}\n".into(), "test-hash").unwrap();
    assert!(outputs.contains_key(&PathBuf::from(
        "crates/cuda-device/src/generated/tcgen05.rs"
    )));
    assert!(outputs.contains_key(&PathBuf::from(
        "crates/dialect-nvvm/src/ops/generated/tcgen05.rs"
    )));

    let mut wrong_adapter = catalog.clone();
    wrong_adapter
        .intrinsics
        .iter_mut()
        .find(|record| record.id == "tcgen05_alloc")
        .unwrap()
        .tcgen05
        .as_mut()
        .unwrap()
        .adapter = Tcgen05Adapter::NoOperands;
    assert!(validate_renderable(&wrong_adapter).is_err());

    let mut wrong_libnvvm_target = catalog.clone();
    wrong_libnvvm_target
        .intrinsics
        .iter_mut()
        .find(|record| record.id == "tcgen05_cp_128x128b_b4x16_p64")
        .unwrap()
        .backend_lowerings[1]
        .target
        .hardware = CatalogHardwareTarget::AnyOf {
        alternatives: vec![
            CatalogHardwareAlternative::ExactArchitecture { sm: 100 },
            CatalogHardwareAlternative::ExactArchitecture { sm: 101 },
            CatalogHardwareAlternative::ExactArchitecture { sm: 103 },
            CatalogHardwareAlternative::ExactArchitecture { sm: 110 },
        ],
    };
    assert!(validate_renderable(&wrong_libnvvm_target).is_err());

    let mut wrong_load_selector = catalog.clone();
    wrong_load_selector
        .intrinsics
        .iter_mut()
        .find(|record| record.id == "tcgen05_ld_16x64b_x1_pack16")
        .unwrap()
        .expected_ptx
        .modifiers
        .remove(5);
    assert!(validate_renderable(&wrong_load_selector).is_err());

    let mut wrong_store_selector = catalog.clone();
    wrong_store_selector
        .intrinsics
        .iter_mut()
        .find(|record| record.id == "tcgen05_st_16x64b_x1_unpack16")
        .unwrap()
        .expected_ptx
        .modifiers
        .remove(5);
    assert!(validate_renderable(&wrong_store_selector).is_err());

    let mut wrong_offset_scope = catalog.clone();
    wrong_offset_scope
        .intrinsics
        .iter_mut()
        .find(|record| record.id == "tcgen05_ld_16x32bx2_x1_raw")
        .unwrap()
        .semantics
        .execution_scope = "thread".into();
    assert!(validate_renderable(&wrong_offset_scope).is_err());

    let mut wrong_copy_spelling = catalog;
    wrong_copy_spelling
        .intrinsics
        .iter_mut()
        .find(|record| record.id == "tcgen05_cp_128x128b_b4x16_p64")
        .unwrap()
        .expected_ptx
        .modifiers
        .remove(3);
    assert!(validate_renderable(&wrong_copy_spelling).is_err());
}
