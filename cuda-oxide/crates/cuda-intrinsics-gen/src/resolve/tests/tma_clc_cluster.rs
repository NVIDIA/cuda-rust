/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, CatalogHardwareAlternative, CatalogHardwareTarget, ClcAdapter,
    ClusterBarrierMode, ClusterBarrierOrdering, ClusterMemoryAdapter, ClusterMemoryOperation,
    EvidenceStageKind, ImportedFile, IntrinsicBackend, IntrinsicSource, OverlayIntrinsic,
    OverlayShardFile, RuntimeValidation, TmaAdapter, TmaOperation,
};
use crate::util::read_json;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use super::fixtures::*;
use crate::resolve::evidence::*;
use crate::resolve::families::*;
use crate::resolve::overlay::*;
use crate::resolve::policy::*;
use crate::resolve::targets::*;

#[test]
fn compact_clc_admission_matches_llvm_and_fails_closed() {
    let records = expand_clc_admission(&test_clc_admission()).unwrap();
    assert_eq!(records.len(), 6);
    assert_eq!(
        records
            .iter()
            .map(|record| (record.abi_id.as_str(), record.id.as_str()))
            .collect::<Vec<_>>(),
        [
            ("i0322", "clc_try_cancel"),
            ("i0323", "clc_try_cancel_multicast"),
            ("i0324", "clc_query_is_canceled"),
            ("i0325", "clc_query_get_first_ctaid_x"),
            ("i0326", "clc_query_get_first_ctaid_y"),
            ("i0327", "clc_query_get_first_ctaid_z"),
        ]
    );

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let imported: ImportedFile = read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
    let declarations = imported
        .intrinsics
        .iter()
        .map(|record| (record.source_record.as_str(), record))
        .collect::<BTreeMap<_, _>>();
    for record in &records {
        let declaration = declarations[record.source_record.as_deref().unwrap()];
        validate_imported_policy(record, declaration).unwrap();
    }

    assert_eq!(
        parse_hardware_target(&records[1]).unwrap(),
        CatalogHardwareTarget::AnyOf {
            alternatives: vec![
                CatalogHardwareAlternative::ExactArchitecture { sm: 100 },
                CatalogHardwareAlternative::ExactArchitecture { sm: 101 },
                CatalogHardwareAlternative::ExactArchitecture { sm: 103 },
                CatalogHardwareAlternative::ExactArchitecture { sm: 110 },
                CatalogHardwareAlternative::ExactArchitecture { sm: 120 },
                CatalogHardwareAlternative::ExactArchitecture { sm: 121 },
            ],
        }
    );

    let mut missing = test_clc_admission();
    missing.variants.pop();
    assert!(expand_clc_admission(&missing).is_err());

    let mut reordered = test_clc_admission();
    reordered.variants.swap(0, 1);
    assert!(expand_clc_admission(&reordered).is_err());

    let mut wrong_abi = test_clc_admission();
    wrong_abi.variants[0].abi_id = "i9999".into();
    assert!(expand_clc_admission(&wrong_abi).is_err());

    let mut executed = test_clc_admission();
    executed.runtime_validation = RuntimeValidation::Executed;
    assert!(expand_clc_admission(&executed).is_err());

    let declaration = declarations[records[2].source_record.as_deref().unwrap()];
    let mut wrong_adapter = records[2].clone();
    wrong_adapter.clc.as_mut().unwrap().adapter = ClcAdapter::PairU64ToI128U32;
    assert!(validate_imported_policy(&wrong_adapter, declaration).is_err());

    let mut unsorted_targets = records[1].clone();
    unsorted_targets.targets = "sm_120a|sm_100a".into();
    assert!(parse_hardware_target(&unsorted_targets).is_err());

    let mut duplicate_targets = records[1].clone();
    duplicate_targets.targets = "sm_100a|sm_100a".into();
    assert!(parse_hardware_target(&duplicate_targets).is_err());
}

#[test]
fn clc_compact_schema_is_reserved_for_aggregation() {
    let shard = |schema| OverlayShardFile {
        schema,
        family: "clc".into(),
        intrinsics: vec![],
        register_mma_int4: None,
        register_mma_int8: None,
        register_mma_b1: None,
        register_mma_f8f6f4_f32: None,
        register_mma_f8f6f4_f16: None,
        register_mma_mxf8f6f4_f32: None,
        register_mma_fp8: None,
        register_mma_ampere_float: None,
        sparse_mma_integer: None,
        sparse_mma_f8f6f4_f32: None,
        sparse_mma_f8f6f4_f16: None,
        sparse_mma_fp8_f32: None,
        sparse_mma_ordered_ampere_float: None,
        prmt: None,
        packed_conversion_fp8: None,
        packed_conversion_fp8_f16x2: None,
        scalar_conversion: None,
        scalar_arithmetic: None,
        scalar_math: None,
        extended_minmax: None,
        cluster_sreg: None,
        cluster_barrier: None,
        mbarrier_extended: None,
        special_registers: None,
        debug_control: None,
        threadfence: None,
        cluster_memory: None,
        stmatrix: None,
        clc: Some(test_clc_admission()),
        wgmma_controls: None,
        tma: None,
        tcgen05: None,
    };
    let path = Path::new("intrinsics/overlay/clc.toml");
    validate_overlay_shard_schema_with_max(&shard(CLC_SHARD_SCHEMA), path, CLC_SHARD_SCHEMA)
        .unwrap();
    assert!(
        validate_overlay_shard_schema_with_max(
            &shard(CLC_SHARD_SCHEMA - 1),
            path,
            CLC_SHARD_SCHEMA,
        )
        .unwrap_err()
        .to_string()
        .contains("requires overlay shard schema 40")
    );
}

#[test]
fn compact_tma_admission_matches_llvm_and_fails_closed() {
    let records = expand_tma_admission(&test_tma_admission()).unwrap();
    assert_eq!(
        records.len(),
        TMA_OPERATIONS.len() + tma_reduction_matrix().len()
    );
    assert!(records.iter().take(TMA_OPERATIONS.len()).all(|record| {
        let operation = record.tma.as_ref().unwrap().operation;
        let recipe = tma_recipe(operation);
        record.backend_lowerings.iter().any(|route| {
            route.backend == IntrinsicBackend::LlvmNvptx && route.mechanism == recipe.llvm_mechanism
        }) && record.backend_lowerings.iter().any(|route| {
            route.backend == IntrinsicBackend::LibNvvm
                && route.mechanism == BackendLoweringMechanism::InlinePtx
        })
    }));
    assert_eq!(
        records
            .iter()
            .filter(|record| {
                record
                    .tma
                    .as_ref()
                    .unwrap()
                    .operation
                    .prefetch_coordinate_count()
                    .is_some()
            })
            .count(),
        12,
        "the tile-prefetch subfamily must contain every plain/cache-hint pair"
    );
    assert_eq!(
        records
            .iter()
            .take(TMA_OPERATIONS.len())
            .map(|record| (record.abi_id.as_str(), record.id.as_str()))
            .collect::<Vec<_>>(),
        [
            ("i0328", "cp_async_bulk_tensor_1d_g2s"),
            ("i0329", "cp_async_bulk_tensor_2d_g2s"),
            ("i0330", "cp_async_bulk_tensor_2d_g2s_multicast"),
            ("i0331", "cp_async_bulk_tensor_2d_g2s_multicast_cg2"),
            ("i0332", "cp_async_bulk_tensor_3d_g2s"),
            ("i0333", "cp_async_bulk_tensor_4d_g2s"),
            ("i0334", "cp_async_bulk_tensor_5d_g2s"),
            ("i0335", "cp_async_bulk_tensor_1d_s2g"),
            ("i0336", "cp_async_bulk_tensor_2d_s2g"),
            ("i0337", "cp_async_bulk_tensor_3d_s2g"),
            ("i0338", "cp_async_bulk_tensor_4d_s2g"),
            ("i0339", "cp_async_bulk_tensor_5d_s2g"),
            ("i0340", "cp_async_bulk_commit_group"),
            ("i0341", "cp_async_bulk_wait_group"),
            ("i0342", "cp_async_bulk_wait_group_read"),
            ("i0887", "prefetch_tma_descriptor"),
            ("i0888", "cp_async_bulk_prefetch_tensor_1d_l2"),
            ("i0889", "cp_async_bulk_prefetch_tensor_2d_l2"),
            ("i0890", "cp_async_bulk_prefetch_tensor_3d_l2"),
            ("i0891", "cp_async_bulk_prefetch_tensor_4d_l2"),
            ("i0892", "cp_async_bulk_prefetch_tensor_5d_l2"),
            ("i0893", "cp_async_bulk_prefetch_tensor_gather4_2d_l2"),
            ("i0894", "tensormap_replace_box_dim"),
            ("i0895", "tensormap_replace_element_stride"),
            ("i0896", "tensormap_replace_element_type"),
            ("i0897", "tensormap_replace_fill_mode"),
            ("i0898", "tensormap_replace_global_address"),
            ("i0899", "tensormap_replace_global_dim"),
            ("i0900", "tensormap_replace_global_stride"),
            ("i0901", "tensormap_replace_interleave_layout"),
            ("i0902", "tensormap_replace_rank"),
            ("i0903", "tensormap_replace_swizzle_atomicity"),
            ("i0904", "tensormap_replace_swizzle_mode"),
            ("i0905", "fence_proxy_tensormap_generic_acquire_cluster"),
            ("i0906", "fence_proxy_tensormap_generic_acquire_cta"),
            ("i0907", "fence_proxy_tensormap_generic_acquire_gpu"),
            ("i0908", "fence_proxy_tensormap_generic_acquire_system"),
            ("i0909", "fence_proxy_tensormap_generic_release_cluster"),
            ("i0910", "fence_proxy_tensormap_generic_release_cta"),
            ("i0911", "fence_proxy_tensormap_generic_release_gpu"),
            ("i0912", "fence_proxy_tensormap_generic_release_system"),
            ("i0917", "cp_async_bulk_prefetch_tensor_1d_l2_cache_hint"),
            ("i0918", "cp_async_bulk_prefetch_tensor_2d_l2_cache_hint"),
            ("i0919", "cp_async_bulk_prefetch_tensor_3d_l2_cache_hint"),
            ("i0920", "cp_async_bulk_prefetch_tensor_4d_l2_cache_hint"),
            ("i0921", "cp_async_bulk_prefetch_tensor_5d_l2_cache_hint"),
            (
                "i0922",
                "cp_async_bulk_prefetch_tensor_gather4_2d_l2_cache_hint"
            ),
        ]
    );

    let reductions = &records[TMA_OPERATIONS.len()..];
    assert_eq!(reductions.len(), 64);
    assert_eq!(
        (reductions[0].abi_id.as_str(), reductions[0].id.as_str()),
        ("i0923", "cp_async_bulk_tensor_reduce_add_tile_1d")
    );
    assert_eq!(
        (reductions[7].abi_id.as_str(), reductions[7].id.as_str()),
        ("i0930", "cp_async_bulk_tensor_reduce_add_im2col_5d")
    );
    assert_eq!(
        (reductions[63].abi_id.as_str(), reductions[63].id.as_str()),
        ("i0986", "cp_async_bulk_tensor_reduce_xor_im2col_5d")
    );
    assert!(reductions.iter().all(|record| {
        record.tma.as_ref().is_some_and(|tma| {
            tma.operation == TmaOperation::Reduce
                && tma.reduction.is_some()
                && tma.adapter == TmaAdapter::ReductionPointersCoordinatesInjectDefaults
        })
    }));

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let imported: ImportedFile = read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
    let declarations = imported
        .intrinsics
        .iter()
        .map(|record| (record.source_record.as_str(), record))
        .collect::<BTreeMap<_, _>>();
    for record in &records {
        let declaration = declarations[record.source_record.as_deref().unwrap()];
        validate_imported_policy(record, declaration).unwrap();
    }

    assert_eq!(
        parse_hardware_target(&records[3]).unwrap(),
        CatalogHardwareTarget::AnyOf {
            alternatives: vec![
                CatalogHardwareAlternative::ExactArchitecture { sm: 100 },
                CatalogHardwareAlternative::ExactArchitecture { sm: 101 },
                CatalogHardwareAlternative::ExactArchitecture { sm: 103 },
                CatalogHardwareAlternative::ExactArchitecture { sm: 110 },
            ],
        }
    );

    let mut missing = test_tma_admission();
    missing.variants.pop();
    assert!(expand_tma_admission(&missing).is_err());

    let mut reordered = test_tma_admission();
    reordered.variants.swap(0, 1);
    assert!(expand_tma_admission(&reordered).is_err());

    let mut wrong_abi = test_tma_admission();
    wrong_abi.variants[0].abi_id = "i9999".into();
    assert!(expand_tma_admission(&wrong_abi).is_err());

    let mut missing_reduction = test_tma_admission();
    missing_reduction.reduce_variants.pop();
    assert!(expand_tma_admission(&missing_reduction).is_err());

    let mut reordered_reduction = test_tma_admission();
    reordered_reduction.reduce_variants.swap(0, 1);
    assert!(expand_tma_admission(&reordered_reduction).is_err());

    let mut non_contiguous_reduction_abi = test_tma_admission();
    non_contiguous_reduction_abi.reduce_variants[0].abi_id = "i9999".into();
    let records = expand_tma_admission(&non_contiguous_reduction_abi).unwrap();
    assert_eq!(records[TMA_OPERATIONS.len()].abi_id, "i9999");

    let mut executed = test_tma_admission();
    executed.runtime_validation = RuntimeValidation::Executed;
    assert!(expand_tma_admission(&executed).is_err());

    let declaration = declarations[records[0].source_record.as_deref().unwrap()];
    let mut wrong_adapter = records[0].clone();
    wrong_adapter.tma.as_mut().unwrap().adapter = TmaAdapter::NoOperands;
    assert!(validate_imported_policy(&wrong_adapter, declaration).is_err());
}

#[test]
fn tma_compact_schema_is_reserved_for_aggregation() {
    let shard = |schema: u32, include_reductions: bool| {
        let mut admission = test_tma_admission();
        if !include_reductions {
            admission.reduce_variants.clear();
        }
        OverlayShardFile {
            schema,
            family: "tma".into(),
            intrinsics: vec![],
            register_mma_int4: None,
            register_mma_int8: None,
            register_mma_b1: None,
            register_mma_f8f6f4_f32: None,
            register_mma_f8f6f4_f16: None,
            register_mma_mxf8f6f4_f32: None,
            register_mma_fp8: None,
            register_mma_ampere_float: None,
            sparse_mma_integer: None,
            sparse_mma_f8f6f4_f32: None,
            sparse_mma_f8f6f4_f16: None,
            sparse_mma_fp8_f32: None,
            sparse_mma_ordered_ampere_float: None,
            prmt: None,
            packed_conversion_fp8: None,
            packed_conversion_fp8_f16x2: None,
            scalar_conversion: None,
            scalar_arithmetic: None,
            scalar_math: None,
            extended_minmax: None,
            cluster_sreg: None,
            cluster_barrier: None,
            mbarrier_extended: None,
            special_registers: None,
            debug_control: None,
            threadfence: None,
            cluster_memory: None,
            stmatrix: None,
            clc: None,
            wgmma_controls: None,
            tma: Some(admission),
            tcgen05: None,
        }
    };
    let path = Path::new("intrinsics/overlay/tma.toml");

    validate_overlay_shard_schema_with_max(
        &shard(TMA_SHARD_SCHEMA, false),
        path,
        TMA_REDUCTION_SHARD_SCHEMA,
    )
    .unwrap();
    assert!(
        validate_overlay_shard_schema_with_max(
            &shard(TMA_SHARD_SCHEMA - 1, false),
            path,
            TMA_REDUCTION_SHARD_SCHEMA,
        )
        .unwrap_err()
        .to_string()
        .contains("compact TMA admission requires overlay shard schema 61")
    );

    validate_overlay_shard_schema_with_max(
        &shard(TMA_REDUCTION_SHARD_SCHEMA, true),
        path,
        TMA_REDUCTION_SHARD_SCHEMA,
    )
    .unwrap();
    assert!(
        validate_overlay_shard_schema_with_max(
            &shard(TMA_REDUCTION_SHARD_SCHEMA - 1, true),
            path,
            TMA_REDUCTION_SHARD_SCHEMA,
        )
        .unwrap_err()
        .to_string()
        .contains("compact TMA reduction admission requires overlay shard schema 62")
    );
}

#[test]
fn cluster_memory_compact_schema_is_reserved_and_fail_closed() {
    let shard = |schema| OverlayShardFile {
        schema,
        family: "cluster_memory".into(),
        intrinsics: vec![],
        register_mma_int4: None,
        register_mma_int8: None,
        register_mma_b1: None,
        register_mma_f8f6f4_f32: None,
        register_mma_f8f6f4_f16: None,
        register_mma_mxf8f6f4_f32: None,
        register_mma_fp8: None,
        register_mma_ampere_float: None,
        sparse_mma_integer: None,
        sparse_mma_f8f6f4_f32: None,
        sparse_mma_f8f6f4_f16: None,
        sparse_mma_fp8_f32: None,
        sparse_mma_ordered_ampere_float: None,
        prmt: None,
        packed_conversion_fp8: None,
        packed_conversion_fp8_f16x2: None,
        scalar_conversion: None,
        scalar_arithmetic: None,
        scalar_math: None,
        extended_minmax: None,
        cluster_sreg: None,
        cluster_barrier: None,
        mbarrier_extended: None,
        special_registers: None,
        debug_control: None,
        threadfence: None,
        cluster_memory: Some(test_cluster_memory_admission()),
        stmatrix: None,
        clc: None,
        wgmma_controls: None,
        tma: None,
        tcgen05: None,
    };
    let path = Path::new("intrinsics/overlay/cluster_memory.toml");
    validate_overlay_shard_schema_with_max(
        &shard(CLUSTER_MEMORY_SHARD_SCHEMA),
        path,
        CLUSTER_MEMORY_SHARD_SCHEMA,
    )
    .unwrap();
    assert!(
        validate_overlay_shard_schema_with_max(
            &shard(CLUSTER_MEMORY_SHARD_SCHEMA),
            path,
            CLUSTER_MEMORY_SHARD_SCHEMA - 1,
        )
        .is_err()
    );
    let error = validate_overlay_shard_schema_with_max(
        &shard(CLUSTER_MEMORY_SHARD_SCHEMA - 1),
        path,
        CLUSTER_MEMORY_SHARD_SCHEMA,
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("requires overlay shard schema 39")
    );
}

#[test]
fn cluster_memory_admission_preserves_mapa_identity_and_ptx_native_read() {
    let admission = test_cluster_memory_admission();
    let records = expand_cluster_memory_admission(&admission).unwrap();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].id, "map_shared_rank");
    assert_eq!(records[1].id, "dsmem_read_u32");
    assert_eq!(records[0].abi_id, "i0320");
    assert_eq!(records[1].abi_id, "i0321");
    assert_eq!(
        cluster_memory_inline_recipe(ClusterMemoryOperation::MapSharedRank),
        ("mapa.shared::cluster.u64 $0, $1, $2;", "=l,l,r")
    );
    assert_eq!(
        cluster_memory_inline_recipe(ClusterMemoryOperation::ReadU32),
        (
            "{ .reg .u64 %mapped; mapa.shared::cluster.u64 %mapped, $1, $2; ld.shared::cluster.u32 $0, [%mapped]; }",
            "=r,l,r,~{memory}"
        )
    );

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let imported: ImportedFile = read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
    let mapa = &records[0];
    let declaration = imported
        .intrinsics
        .iter()
        .find(|declaration| declaration.source_record == "int_nvvm_mapa_shared_cluster")
        .unwrap();
    assert_eq!(declaration.arguments, ["shared_ptr", "i32"]);
    assert_eq!(declaration.results, ["shared_cluster_ptr"]);
    assert_eq!(
        declaration.properties,
        ["IntrNoMem", "IntrSpeculatable", "NoCapture<arg0>"]
    );
    validate_imported_policy(mapa, declaration).unwrap();
    assert_eq!(
        declaration
            .selections
            .iter()
            .filter(|selection| selection.asm.starts_with("mapa.shared::cluster.u64"))
            .map(|selection| selection.source_record.as_str())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["mapa_shared_cluster_64", "mapa_shared_cluster_64i"])
    );

    let read = &records[1];
    validate_ptx_native_policy(read).unwrap();
    assert!(read.source_record.is_none());
    assert!(read.llvm_symbol.is_none());
    assert!(matches!(
        resolve_policy_source(read).unwrap(),
        IntrinsicSource::PtxNative { .. }
    ));
    assert_eq!(read.memory, "read");

    let mut wrong_adapter = mapa.clone();
    wrong_adapter.cluster_memory.as_mut().unwrap().adapter =
        ClusterMemoryAdapter::ConstU32PointerRankToU32;
    assert!(validate_imported_policy(&wrong_adapter, declaration).is_err());

    let mut typed_as3 = mapa.clone();
    typed_as3.llvm_results = vec!["shared_ptr".into()];
    assert!(validate_imported_policy(&typed_as3, declaration).is_err());

    let mut wrong_route = mapa.clone();
    wrong_route.backend_lowerings[0].mechanism = BackendLoweringMechanism::TypedNvvm;
    assert!(validate_imported_policy(&wrong_route, declaration).is_err());

    let mut wrong_floor = read.clone();
    wrong_floor.minimum_sm = Some("sm_80".into());
    assert!(validate_ptx_native_policy(&wrong_floor).is_err());

    let mut missing = admission.clone();
    missing.variants.pop();
    assert!(expand_cluster_memory_admission(&missing).is_err());

    let mut duplicate = admission.clone();
    duplicate.variants[1].operation = ClusterMemoryOperation::MapSharedRank;
    assert!(expand_cluster_memory_admission(&duplicate).is_err());

    let mut wrong_abi = admission.clone();
    wrong_abi.variants[0].abi_id = "i9999".into();
    assert!(expand_cluster_memory_admission(&wrong_abi).is_err());

    let mut executed = admission;
    executed.runtime_validation = RuntimeValidation::Executed;
    assert!(expand_cluster_memory_admission(&executed).is_err());
}

#[test]
fn compact_cluster_barrier_admission_and_semantics_fail_closed() {
    let records = expand_cluster_barrier_admission(&test_cluster_barrier_admission()).unwrap();
    assert_eq!(records.len(), 6);

    let imported: ImportedFile = read_json(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("intrinsics/imported.json"),
    )
    .unwrap();
    for record in &records {
        let declaration = imported
            .intrinsics
            .iter()
            .find(|declaration| {
                Some(declaration.source_record.as_str()) == record.source_record.as_deref()
            })
            .unwrap();
        validate_imported_policy(record, declaration).unwrap();
    }

    let declaration_for = |record: &OverlayIntrinsic| {
        imported
            .intrinsics
            .iter()
            .find(|declaration| {
                Some(declaration.source_record.as_str()) == record.source_record.as_deref()
            })
            .unwrap()
    };
    let base = records
        .iter()
        .find(|record| {
            record
                .cluster_barrier
                .as_ref()
                .is_some_and(|barrier| barrier.mode == ClusterBarrierMode::ArriveAligned)
        })
        .unwrap();

    let mut wrong_mode = base.clone();
    wrong_mode.cluster_barrier.as_mut().unwrap().mode = ClusterBarrierMode::WaitAligned;
    assert!(validate_imported_policy(&wrong_mode, declaration_for(base)).is_err());

    let mut wrong_order = base.clone();
    wrong_order.cluster_barrier.as_mut().unwrap().ordering = ClusterBarrierOrdering::Relaxed;
    assert!(validate_imported_policy(&wrong_order, declaration_for(base)).is_err());

    let mut wrong_alignment = base.clone();
    wrong_alignment.cluster_barrier.as_mut().unwrap().aligned = false;
    assert!(validate_imported_policy(&wrong_alignment, declaration_for(base)).is_err());

    let mut missing = test_cluster_barrier_admission();
    missing.variants.pop();
    assert!(expand_cluster_barrier_admission(&missing).is_err());

    let mut duplicate = test_cluster_barrier_admission();
    duplicate.variants[5].mode = ClusterBarrierMode::Arrive;
    assert!(expand_cluster_barrier_admission(&duplicate).is_err());

    let mut wrong_abi = test_cluster_barrier_admission();
    wrong_abi.variants[0].abi_id = "i9999".into();
    assert!(expand_cluster_barrier_admission(&wrong_abi).is_err());
}

#[test]
fn cluster_barrier_evidence_validates_both_backend_routes() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut admission = test_cluster_barrier_admission();
    admission.llvm_evidence_profile = "rust-llvm-23.1.0-16696adc".into();
    admission.libnvvm_evidence_profile = "cuda-13.3-libnvvm-13.3.33-cluster-barrier".into();
    let policies = expand_cluster_barrier_admission(&admission).unwrap();
    let evidence_files = vec![
        read_evidence_file(
            &repo_root.join("intrinsics/evidence/rust-llvm-23.1.0-16696adc-cluster-barrier.json"),
        )
        .unwrap(),
        read_evidence_file(
            &repo_root.join("intrinsics/evidence/cuda-13.3-libnvvm-13.3.33-cluster-barrier.json"),
        )
        .unwrap(),
    ];
    let indexed =
        index_evidence(&evidence_files, "16696adcd119e6ba9cc175207d984d7021211acb").unwrap();

    for policy in &policies {
        for lowering in &policy.backend_lowerings {
            let evidence = indexed
                .get(&(lowering.evidence_profile.as_str(), policy.id.as_str()))
                .unwrap();
            validate_evidence(policy, evidence, Some(lowering)).unwrap();
        }
    }

    let mut missing_typed_failure = evidence_files.clone();
    let libnvvm = missing_typed_failure
        .iter_mut()
        .find(|file| file.backend_kind == Some(IntrinsicBackend::LibNvvm))
        .unwrap();
    for record in &mut libnvvm.records {
        record.stages.retain(|stage| {
            stage.mechanism != Some(BackendLoweringMechanism::TypedNvvm)
                || stage.stage != EvidenceStageKind::DeviceLink
        });
    }
    let indexed = index_evidence(
        &missing_typed_failure,
        "16696adcd119e6ba9cc175207d984d7021211acb",
    )
    .unwrap();
    let policy = &policies[0];
    let lowering = policy
        .backend_lowerings
        .iter()
        .find(|lowering| lowering.backend == IntrinsicBackend::LibNvvm)
        .unwrap();
    let evidence = indexed
        .get(&(lowering.evidence_profile.as_str(), policy.id.as_str()))
        .unwrap();
    assert!(validate_evidence(policy, evidence, Some(lowering)).is_err());
}
