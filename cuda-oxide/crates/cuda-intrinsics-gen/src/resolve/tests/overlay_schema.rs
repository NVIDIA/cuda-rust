/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    CatalogFile, OverlayShardFile, PackedAluFormat, PackedAluOperation, RegisterMmaAccumulator,
    RegisterMmaKind,
};
use crate::util::read_json;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use super::fixtures::*;
use crate::resolve::abi_ledger::*;
use crate::resolve::driver::*;
use crate::resolve::guards::*;
use crate::resolve::overlay::*;

#[test]
fn duplicate_values_are_rejected() {
    let mut values = BTreeSet::new();
    insert_unique(&mut values, "thread_idx_x", "catalog ID").unwrap();
    let error = insert_unique(&mut values, "thread_idx_x", "catalog ID").unwrap_err();
    assert!(error.to_string().contains("duplicate catalog ID"));
}

#[test]
fn overloaded_symbols_require_distinct_resolved_identities() {
    let bf16 = packed_alu_policy(PackedAluFormat::Bf16x2, PackedAluOperation::Abs);
    let f16 = packed_alu_policy(PackedAluFormat::F16x2, PackedAluOperation::Abs);
    validate_unique_overlay(&[bf16.clone(), f16.clone()], 1).unwrap();

    let mut unresolved = f16.clone();
    unresolved.resolved_llvm_symbol = None;
    let error = validate_unique_overlay(&[bf16.clone(), unresolved], 1).unwrap_err();
    assert!(error.to_string().contains("without a resolved symbol"));

    let mut duplicate = f16;
    duplicate.resolved_llvm_symbol = bf16.resolved_llvm_symbol.clone();
    let error = validate_unique_overlay(&[bf16, duplicate], 1).unwrap_err();
    assert!(error.to_string().contains("duplicate resolved LLVM symbol"));
}

#[test]
fn overlay_manifest_loads_sorted_family_shards() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let (overlay, hash) =
        read_overlay(&repo_root, &repo_root.join("intrinsics/overlay.toml")).unwrap();
    assert_eq!(overlay.schema, OVERLAY_SCHEMA);
    assert_eq!(overlay.shards.len(), 66);
    assert_eq!(overlay.intrinsics.len(), 1029);
    assert_eq!(
        overlay
            .intrinsics
            .iter()
            .filter(|record| record.family == "scalar_arithmetic")
            .count(),
        64
    );
    assert_eq!(
        overlay
            .intrinsics
            .iter()
            .filter(|record| record.family == "scalar_conversion")
            .count(),
        10
    );
    assert_eq!(
        overlay
            .intrinsics
            .iter()
            .filter(|record| record.family == "prmt")
            .count(),
        7
    );
    assert_eq!(
        overlay
            .intrinsics
            .iter()
            .filter(|record| record.family == "debug_control")
            .count(),
        3
    );
    assert_eq!(
        overlay
            .intrinsics
            .iter()
            .filter(|record| record.family == "stmatrix")
            .count(),
        4
    );
    assert_eq!(
        overlay
            .intrinsics
            .iter()
            .filter(|record| record.family == "packed_alu")
            .count(),
        30
    );
    assert_eq!(
        overlay
            .intrinsics
            .iter()
            .filter(|record| record.family == "packed_conversion")
            .count(),
        18
    );
    assert_eq!(
        overlay
            .intrinsics
            .iter()
            .filter(|record| record.family == "active_mask")
            .count(),
        1
    );
    assert_eq!(
        overlay
            .intrinsics
            .iter()
            .filter(|record| record.family == "dotprod")
            .count(),
        4
    );
    assert_eq!(
        overlay
            .intrinsics
            .iter()
            .filter(|record| record.family == "ldmatrix")
            .count(),
        18
    );
    assert_eq!(
        overlay
            .intrinsics
            .iter()
            .filter(|record| record.family == "register_mma")
            .count(),
        154
    );
    assert_eq!(
        overlay
            .intrinsics
            .iter()
            .filter(|record| record.family == "sparse_mma")
            .count(),
        126
    );
    assert_eq!(
        overlay
            .intrinsics
            .iter()
            .filter(|record| record.family == "cp_async_copy")
            .count(),
        8
    );
    assert_eq!(
        overlay
            .intrinsics
            .iter()
            .filter(|record| record.family == "cp_async_control")
            .count(),
        3
    );
    assert_eq!(
        overlay
            .intrinsics
            .iter()
            .filter(|record| record.family == "cp_async_mbarrier")
            .count(),
        4
    );
    assert_eq!(
        overlay
            .intrinsics
            .iter()
            .filter(|record| record.family == "mbarrier_basic")
            .count(),
        5
    );
    assert_eq!(
        overlay
            .intrinsics
            .iter()
            .filter(|record| record.family == "cluster_memory")
            .count(),
        2
    );
    assert_eq!(
        overlay
            .intrinsics
            .iter()
            .filter(|record| record.family == "clc")
            .count(),
        6
    );
    assert_eq!(
        overlay
            .intrinsics
            .iter()
            .filter(|record| record.family == "sync")
            .count(),
        4
    );
    assert_eq!(
        overlay
            .intrinsics
            .iter()
            .filter(|record| record.family == "vote")
            .count(),
        4
    );
    assert_eq!(
        overlay
            .intrinsics
            .iter()
            .filter(|record| record.family == "warp_barrier")
            .count(),
        1
    );
    assert_eq!(
        overlay
            .intrinsics
            .iter()
            .filter(|record| record.family == "warp_match")
            .count(),
        4
    );
    assert_eq!(
        overlay
            .intrinsics
            .iter()
            .filter(|record| record.family == "warp_shuffle")
            .count(),
        12
    );
    assert_eq!(hash.len(), 64);

    for invalid in [
        "../outside.toml",
        "/absolute.toml",
        "other/family.toml",
        "overlay/../outside.toml",
        "overlay/not-toml.json",
    ] {
        assert!(validate_overlay_shard_path(invalid).is_err(), "{invalid}");
    }
}

#[test]
fn overlay_shard_schema_range_is_composable_and_new_fields_fail_closed() {
    let shard = |schema, sparse_mma_f8f6f4_f32, prmt| OverlayShardFile {
        schema,
        family: "sparse_mma".into(),
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
        sparse_mma_f8f6f4_f32,
        sparse_mma_f8f6f4_f16: None,
        sparse_mma_fp8_f32: None,
        sparse_mma_ordered_ampere_float: None,
        prmt,
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
        tma: None,
        tcgen05: None,
    };
    let path = Path::new("intrinsics/overlay/test.toml");
    validate_overlay_shard_schema(&shard(26, None, None), path).unwrap();
    validate_overlay_shard_schema(&shard(27, None, None), path).unwrap();
    validate_overlay_shard_schema(&shard(28, None, None), path).unwrap();
    validate_overlay_shard_schema(&shard(29, None, None), path).unwrap();
    validate_overlay_shard_schema(&shard(30, None, None), path).unwrap();
    validate_overlay_shard_schema(&shard(31, None, None), path).unwrap();
    validate_overlay_shard_schema(&shard(32, None, None), path).unwrap();
    validate_overlay_shard_schema(&shard(33, None, None), path).unwrap();
    validate_overlay_shard_schema(&shard(34, None, None), path).unwrap();
    validate_overlay_shard_schema(&shard(35, None, None), path).unwrap();
    validate_overlay_shard_schema(&shard(27, Some(test_f8f6f4_admission()), None), path).unwrap();
    validate_overlay_shard_schema_with_max(
        &shard(27, Some(test_f8f6f4_admission()), None),
        path,
        30,
    )
    .unwrap();
    validate_overlay_shard_schema(&shard(28, None, Some(test_prmt_admission())), path).unwrap();
    validate_overlay_shard_schema_with_max(&shard(28, None, Some(test_prmt_admission())), path, 30)
        .unwrap();

    assert!(validate_overlay_shard_schema(&shard(25, None, None), path).is_err());
    for schema in 35..=OVERLAY_SHARD_SCHEMA {
        validate_overlay_shard_schema(&shard(schema, None, None), path).unwrap();
    }
    assert!(
        validate_overlay_shard_schema(&shard(OVERLAY_SHARD_SCHEMA + 1, None, None), path).is_err()
    );
    let error =
        validate_overlay_shard_schema(&shard(26, Some(test_f8f6f4_admission()), None), path)
            .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("requires overlay shard schema 27")
    );
    let error = validate_overlay_shard_schema(&shard(27, None, Some(test_prmt_admission())), path)
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("requires overlay shard schema 28")
    );
    let mut f16_mma = shard(REGISTER_MMA_F8F6F4_F16_SHARD_SCHEMA, None, None);
    f16_mma.family = "register_mma".into();
    f16_mma.register_mma_f8f6f4_f16 = Some(test_register_mma_f8f6f4_admission(
        RegisterMmaAccumulator::F16,
    ));
    validate_overlay_shard_schema(&f16_mma, path).unwrap();
    f16_mma.schema -= 1;
    assert!(
        validate_overlay_shard_schema(&f16_mma, path)
            .unwrap_err()
            .to_string()
            .contains("requires overlay shard schema 47")
    );

    let mut sparse_f16_mma = shard(SPARSE_MMA_F8F6F4_F16_SHARD_SCHEMA, None, None);
    sparse_f16_mma.sparse_mma_f8f6f4_f16 = Some(test_sparse_mma_f8f6f4_f16_admission());
    validate_overlay_shard_schema(&sparse_f16_mma, path).unwrap();
    sparse_f16_mma.schema -= 1;
    assert!(
        validate_overlay_shard_schema(&sparse_f16_mma, path)
            .unwrap_err()
            .to_string()
            .contains("requires overlay shard schema 50")
    );

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let ampere_float_bytes =
        std::fs::read(repo_root.join("intrinsics/overlay/sparse_mma_ordered_ampere_float.toml"))
            .unwrap();
    let mut sparse_ampere_float: OverlayShardFile = toml::from_slice(&ampere_float_bytes).unwrap();
    assert!(
        sparse_ampere_float
            .sparse_mma_ordered_ampere_float
            .is_some()
    );
    assert_eq!(
        sparse_ampere_float.schema,
        SPARSE_MMA_AMPERE_FLOAT_SHARD_SCHEMA
    );
    validate_overlay_shard_schema(&sparse_ampere_float, path).unwrap();
    sparse_ampere_float.schema -= 1;
    assert!(
        validate_overlay_shard_schema(&sparse_ampere_float, path)
            .unwrap_err()
            .to_string()
            .contains("requires overlay shard schema 63")
    );

    let mut standard_fp8_mma = shard(REGISTER_MMA_FP8_SHARD_SCHEMA, None, None);
    standard_fp8_mma.family = "register_mma".into();
    standard_fp8_mma.register_mma_fp8 = Some(test_register_mma_fp8_admission());
    validate_overlay_shard_schema(&standard_fp8_mma, path).unwrap();
    standard_fp8_mma.schema -= 1;
    assert!(
        validate_overlay_shard_schema(&standard_fp8_mma, path)
            .unwrap_err()
            .to_string()
            .contains("requires overlay shard schema 48")
    );

    let fp8_shard = |schema| OverlayShardFile {
        schema,
        family: "packed_conversion".into(),
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
        packed_conversion_fp8: Some(test_fp8_conversion_admission()),
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
        tma: None,
        tcgen05: None,
    };
    validate_overlay_shard_schema(&fp8_shard(29), path).unwrap();
    validate_overlay_shard_schema_with_max(&fp8_shard(29), path, 30).unwrap();
    let error = validate_overlay_shard_schema(&fp8_shard(28), path).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("requires overlay shard schema 29")
    );

    let cluster_shard = OverlayShardFile {
        schema: 31,
        family: "cluster_barrier".into(),
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
        cluster_barrier: Some(test_cluster_barrier_admission()),
        mbarrier_extended: None,
        special_registers: None,
        debug_control: None,
        threadfence: None,
        cluster_memory: None,
        stmatrix: None,
        clc: None,
        wgmma_controls: None,
        tma: None,
        tcgen05: None,
    };
    validate_overlay_shard_schema_with_max(&cluster_shard, path, 31).unwrap();
    let mut old_cluster_shard = cluster_shard;
    old_cluster_shard.schema = 30;
    assert!(
        validate_overlay_shard_schema_with_max(&old_cluster_shard, path, 31)
            .unwrap_err()
            .to_string()
            .contains("requires overlay shard schema 31")
    );

    let extended_shard = OverlayShardFile {
        schema: MBARRIER_EXTENDED_SHARD_SCHEMA,
        family: "mbarrier_extended".into(),
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
        mbarrier_extended: Some(test_mbarrier_extended_admission()),
        special_registers: None,
        debug_control: None,
        threadfence: None,
        cluster_memory: None,
        stmatrix: None,
        clc: None,
        wgmma_controls: None,
        tma: None,
        tcgen05: None,
    };
    validate_overlay_shard_schema_with_max(&extended_shard, path, MBARRIER_EXTENDED_SHARD_SCHEMA)
        .unwrap();
    let mut old_extended_shard = extended_shard;
    old_extended_shard.schema = MBARRIER_EXTENDED_SHARD_SCHEMA - 1;
    assert!(
        validate_overlay_shard_schema_with_max(
            &old_extended_shard,
            path,
            MBARRIER_EXTENDED_SHARD_SCHEMA,
        )
        .unwrap_err()
        .to_string()
        .contains("requires overlay shard schema 40")
    );
}

#[test]
fn existing_catalog_intrinsics_keep_their_json_shape_without_kind() {
    fn abi_number(abi_id: &str) -> u16 {
        validate_abi_id(abi_id).unwrap();
        abi_id[1..].parse().unwrap()
    }

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let original: serde_json::Value =
        read_json(&repo_root.join("intrinsics/catalog.json")).unwrap();
    let catalog: CatalogFile = read_json(&repo_root.join("intrinsics/catalog.json")).unwrap();
    let serialized = serde_json::to_value(&catalog).unwrap();
    let original_by_abi = original["intrinsics"]
        .as_array()
        .unwrap()
        .iter()
        .map(|record| {
            let abi_id = record["rust"]["abi_id"].as_str().unwrap();
            (abi_number(abi_id), record)
        })
        .collect::<BTreeMap<_, _>>();
    let serialized_by_abi = serialized["intrinsics"]
        .as_array()
        .unwrap()
        .iter()
        .map(|record| {
            let abi_id = record["rust"]["abi_id"].as_str().unwrap();
            (abi_number(abi_id), record)
        })
        .collect::<BTreeMap<_, _>>();
    for abi in 1..=503 {
        assert_eq!(original_by_abi.get(&abi), serialized_by_abi.get(&abi));
    }

    let existing = catalog
        .intrinsics
        .iter()
        .filter(|record| abi_number(&record.rust.abi_id) <= 503)
        .collect::<Vec<_>>();
    assert_eq!(existing.len(), 503);
    assert!(existing.iter().all(|record| {
        record
            .register_mma
            .as_ref()
            .is_none_or(|mma| mma.kind.is_none())
    }));

    let standard_fp8 = catalog
        .intrinsics
        .iter()
        .filter(|record| (504..=519).contains(&abi_number(&record.rust.abi_id)))
        .collect::<Vec<_>>();
    assert!(standard_fp8.is_empty() || standard_fp8.len() == 16);
    if standard_fp8.len() == 16 {
        assert!(standard_fp8.iter().all(|record| {
            record
                .register_mma
                .as_ref()
                .is_some_and(|mma| mma.kind == Some(RegisterMmaKind::Standard))
        }));
        for abi in 504..=519 {
            assert_eq!(original_by_abi[&abi]["register_mma"]["kind"], "standard");
        }
    }
}

#[test]
fn duplicate_identity_surfaces_are_rejected_independently() {
    let first = policy();

    let mut second = distinct_policy();
    second.abi_id = first.abi_id.clone();
    assert!(
        validate_unique_overlay(&[first.clone(), second], 1)
            .unwrap_err()
            .to_string()
            .contains("duplicate intrinsic ABI ID")
    );

    let mut second = distinct_policy();
    second.operation_key = first.operation_key.clone();
    assert!(
        validate_unique_overlay(&[first.clone(), second], 1)
            .unwrap_err()
            .to_string()
            .contains("duplicate intrinsic operation key")
    );

    let mut second = distinct_policy();
    second.public_rust_path = first.public_rust_path.clone();
    assert!(
        validate_unique_overlay(&[first.clone(), second], 1)
            .unwrap_err()
            .to_string()
            .contains("duplicate public Rust path")
    );

    let mut second = distinct_policy();
    second.dialect_op_name = first.dialect_op_name.clone();
    assert!(
        validate_unique_overlay(&[first.clone(), second], 1)
            .unwrap_err()
            .to_string()
            .contains("duplicate dialect op variant")
    );

    let mut second = distinct_policy();
    second.llvm_symbol = first.llvm_symbol.clone();
    assert!(
        validate_unique_overlay(&[first, second], 1)
            .unwrap_err()
            .to_string()
            .contains("duplicate LLVM symbol")
    );
}

#[test]
fn safe_record_requires_an_allowlist_reason() {
    let mut record = policy();
    record.safe_allowlist_reason = None;
    assert!(
        validate_imported_policy(&record, &declaration())
            .unwrap_err()
            .to_string()
            .contains("safe_allowlist_reason")
    );
}

#[test]
fn candidate_resolution_is_the_only_path_that_can_omit_evidence() {
    let repo = repo_without_evidence();
    let candidate = resolve_candidate(
        &repo.0,
        "thread_idx_x",
        "LLVM version candidate",
        &"a".repeat(64),
        "sm_80",
        "+ptx70",
    )
    .unwrap();
    assert_eq!(candidate.catalog.intrinsics.len(), 1);
    assert_eq!(candidate.catalog.intrinsics[0].id, "thread_idx_x");
    assert_eq!(candidate.catalog.intrinsics[0].backend.status, "candidate");

    let scalar = resolve_candidate(
        &repo.0,
        "i0390",
        "LLVM version candidate",
        &"a".repeat(64),
        "sm_80",
        "+ptx70",
    )
    .unwrap();
    assert_eq!(scalar.catalog.intrinsics[0].id, "mul_rn_f64");
    assert!(candidate.catalog.inputs.evidence_sha256.is_empty());

    for (dense_id, abi_id) in [
        ("mma_m16n8k32_f32_e2m1_e2m1", "i0454"),
        ("mma_m16n8k32_f16_e2m1_e2m1", "i0479"),
    ] {
        let dense_by_name = resolve_candidate(
            &repo.0,
            dense_id,
            "LLVM version candidate",
            &"a".repeat(64),
            "sm_120f",
            "+ptx88",
        )
        .unwrap();
        let dense_by_abi = resolve_candidate(
            &repo.0,
            abi_id,
            "LLVM version candidate",
            &"a".repeat(64),
            "sm_120f",
            "+ptx88",
        )
        .unwrap();
        assert_eq!(
            dense_by_name.catalog.intrinsics,
            dense_by_abi.catalog.intrinsics
        );
        for lookup in [dense_id, abi_id] {
            let error = resolve_candidate(
                &repo.0,
                lookup,
                "LLVM version candidate",
                &"a".repeat(64),
                "sm_120f",
                "+ptx87",
            )
            .unwrap_err();
            assert!(error.to_string().contains(dense_id), "{error:#}");
        }
    }

    let error = resolve(&repo.0).unwrap_err();
    assert!(
        error.to_string().contains("intrinsics/evidence"),
        "{error:#}"
    );
    let error = resolve_candidate(
        &repo.0,
        "not_an_intrinsic",
        "LLVM version candidate",
        &"a".repeat(64),
        "sm_80",
        "+ptx70",
    )
    .unwrap_err();
    assert!(error.to_string().contains("unknown overlay intrinsic"));
}

#[test]
fn candidate_resolution_cannot_change_normal_catalog_bytes() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let before = crate::util::pretty_json(&resolve(&repo_root).unwrap()).unwrap();
    resolve_candidate(
        &repo_root,
        "thread_idx_x",
        "LLVM version candidate",
        &"a".repeat(64),
        "sm_80",
        "+ptx70",
    )
    .unwrap();
    let after = crate::util::pretty_json(&resolve(&repo_root).unwrap()).unwrap();
    assert_eq!(before.as_bytes(), after.as_bytes());
}
