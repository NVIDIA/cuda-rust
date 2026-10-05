/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, EvidenceStageKind, ImportedFile, IntrinsicBackend, IntrinsicSource,
    OverlayShardFile, RuntimeValidation, SpecialRegisterOutputConstraint,
};
use crate::util::read_json;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use super::fixtures::*;
use crate::resolve::driver::*;
use crate::resolve::evidence::*;
use crate::resolve::families::*;
use crate::resolve::guards::*;
use crate::resolve::overlay::*;
use crate::resolve::policy::*;

#[test]
fn cluster_sreg_admission_uses_its_fixed_introduction_schema() {
    let shard = |schema| {
        toml::from_str::<OverlayShardFile>(&format!(
            r#"
schema = {schema}
family = "sreg"

[cluster_sreg]
axes = ["x", "y", "z"]
xyz_product_count = 12
record_count = 14
"#
        ))
        .unwrap()
    };
    let path = Path::new("intrinsics/overlay/sreg_cluster.toml");

    let old = shard(CLUSTER_SREG_SHARD_SCHEMA - 1);
    assert!(
        validate_overlay_shard_schema(&old, path)
            .unwrap_err()
            .to_string()
            .contains("requires overlay shard schema 30")
    );

    let current = shard(CLUSTER_SREG_SHARD_SCHEMA);
    validate_overlay_shard_schema(&current, path).unwrap();
    let admission = current.cluster_sreg.unwrap();
    assert_eq!(expand_cluster_sreg_admission(&admission).unwrap().len(), 14);

    let mut wrong_axes = admission.clone();
    wrong_axes.axes.swap(0, 1);
    assert!(expand_cluster_sreg_admission(&wrong_axes).is_err());

    let mut wrong_count = admission;
    wrong_count.record_count = 13;
    assert!(expand_cluster_sreg_admission(&wrong_count).is_err());
}

#[test]
fn special_register_admission_is_closed_and_schema_gated() {
    let admission = test_special_register_admission();
    let records = expand_special_register_admission(&admission).unwrap();
    assert_eq!(records.len(), 12);
    assert_eq!(
        records
            .iter()
            .map(|record| record.id.as_str())
            .collect::<Vec<_>>(),
        [
            "clock",
            "clock64",
            "globaltimer",
            "envreg1",
            "envreg2",
            "smid",
            "nsmid",
            "gridid",
            "warpid",
            "nwarpid",
            "dynamic_smem_size",
            "total_smem_size",
        ]
    );

    let mut reordered = admission.clone();
    reordered.registers.swap(0, 1);
    assert!(expand_special_register_admission(&reordered).is_err());

    let mut wrong_count = admission.clone();
    wrong_count.product_count -= 1;
    assert!(expand_special_register_admission(&wrong_count).is_err());

    let mut executed = admission.clone();
    executed.runtime_validation = RuntimeValidation::Executed;
    assert!(expand_special_register_admission(&executed).is_err());

    let shard = |schema| OverlayShardFile {
        schema,
        family: "sreg".into(),
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
        special_registers: Some(admission.clone()),
        debug_control: None,
        threadfence: None,
        cluster_memory: None,
        stmatrix: None,
        clc: None,
        wgmma_controls: None,
        tma: None,
        tcgen05: None,
    };
    let path = Path::new("intrinsics/overlay/sreg_special.toml");
    validate_overlay_shard_schema(&shard(SPECIAL_REGISTER_SHARD_SCHEMA), path).unwrap();
    assert!(
        validate_overlay_shard_schema(&shard(SPECIAL_REGISTER_SHARD_SCHEMA - 1), path)
            .unwrap_err()
            .to_string()
            .contains("requires overlay shard schema 32")
    );
}

#[test]
fn pinned_special_registers_preserve_apis_widths_and_backend_routes() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let base = load_resolution_base(&repo_root).unwrap();
    let declarations = index_imported_intrinsics(&base.imported).unwrap();
    let records = base
        .overlay
        .intrinsics
        .iter()
        .filter(|record| record.special_register.is_some())
        .collect::<Vec<_>>();
    assert_eq!(records.len(), 12);
    assert_eq!(
        records
            .iter()
            .map(|record| record.abi_id.clone())
            .collect::<BTreeSet<_>>(),
        (283..=294)
            .map(|id| format!("i{id:04}"))
            .collect::<BTreeSet<_>>()
    );

    for record in &records {
        let source = resolve_policy_source(record).unwrap();
        let declaration = resolve_imported_declaration(record, &source, &declarations).unwrap();
        validate_special_register_policy(record, &source, declaration).unwrap();
        validate_special_register_llvm_exclusion(record, &declarations).unwrap();
    }

    for (id, section, anchor) in [
        ("clock", "10.23", "special-registers-clock-clock-hi"),
        ("clock64", "10.24", "special-registers-clock64"),
        ("globaltimer", "10.28", "special-registers-globaltimer"),
        ("envreg1", "10.27", "special-registers-envreg"),
        ("envreg2", "10.27", "special-registers-envreg"),
        ("smid", "10.8", "special-registers-smid"),
        ("nsmid", "10.9", "special-registers-nsmid"),
        ("gridid", "10.10", "special-registers-gridid"),
        ("warpid", "10.4", "special-registers-warpid"),
        ("nwarpid", "10.5", "special-registers-nwarpid"),
        (
            "dynamic_smem_size",
            "10.32",
            "special-registers-dynamic-smem-size",
        ),
        (
            "total_smem_size",
            "10.30",
            "special-registers-total-smem-size",
        ),
    ] {
        let record = records.iter().find(|record| record.id == id).unwrap();
        assert!(record.ptx_isa_section.starts_with(section));
        assert!(record.ptx_isa_url.ends_with(anchor));
    }

    let gridid = records.iter().find(|record| record.id == "gridid").unwrap();
    assert_eq!(gridid.rust_result, "u64");
    assert_eq!(gridid.dialect_results, ["i64"]);
    assert!(matches!(
        resolve_policy_source(gridid).unwrap(),
        IntrinsicSource::PtxNative { .. }
    ));
    assert!(
        gridid
            .special_register
            .as_ref()
            .unwrap()
            .llvm_exclusion
            .is_some()
    );

    let clock = records.iter().find(|record| record.id == "clock").unwrap();
    let source = resolve_policy_source(clock).unwrap();
    let declaration = resolve_imported_declaration(clock, &source, &declarations)
        .unwrap()
        .unwrap();

    let mut wrong_effects = (*clock).clone();
    wrong_effects.memory = "none".into();
    assert!(validate_special_register_policy(&wrong_effects, &source, Some(declaration)).is_err());

    let mut wrong_contract = (*clock).clone();
    wrong_contract
        .special_register
        .as_mut()
        .unwrap()
        .output_constraint = SpecialRegisterOutputConstraint::Register64;
    assert!(validate_special_register_policy(&wrong_contract, &source, Some(declaration)).is_err());

    let mut wrong_route = (*clock).clone();
    wrong_route.backend_lowerings[0].mechanism = BackendLoweringMechanism::InlinePtx;
    assert!(validate_special_register_policy(&wrong_route, &source, Some(declaration)).is_err());
}

#[test]
fn special_register_evidence_validates_both_backend_routes() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let policies = expand_special_register_admission(&test_special_register_admission()).unwrap();
    let mut evidence_files = vec![
        read_evidence_file(
            &repo_root.join("intrinsics/evidence/rust-llvm-23.1.0-16696adc-special-registers.json"),
        )
        .unwrap(),
        read_evidence_file(
            &repo_root.join("intrinsics/evidence/cuda-13.3-libnvvm-13.3.33-special-registers.json"),
        )
        .unwrap(),
    ];
    let llvm_revision = "16696adcd119e6ba9cc175207d984d7021211acb";
    let indexed = index_evidence(&evidence_files, llvm_revision).unwrap();
    for policy in &policies {
        for lowering in &policy.backend_lowerings {
            let evidence = indexed
                .get(&(lowering.evidence_profile.as_str(), policy.id.as_str()))
                .unwrap();
            validate_evidence(policy, evidence, Some(lowering)).unwrap();
        }
    }

    let libnvvm = evidence_files
        .iter_mut()
        .find(|file| file.backend_kind == Some(IntrinsicBackend::LibNvvm))
        .unwrap();
    libnvvm
        .records
        .iter_mut()
        .find(|record| record.id == "gridid")
        .unwrap()
        .stages
        .retain(|stage| stage.stage != EvidenceStageKind::DeviceLink);
    let indexed = index_evidence(&evidence_files, llvm_revision).unwrap();
    let policy = policies
        .iter()
        .find(|policy| policy.id == "gridid")
        .unwrap();
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

#[test]
fn threadfence_admission_is_closed_and_uses_schema_34() {
    let shard = |schema| {
        toml::from_str::<OverlayShardFile>(&format!(
            r#"
schema = {schema}
family = "sync"

[threadfence]
llvm_evidence_profile = "llvm-test"
libnvvm_evidence_profile = "libnvvm-test"
runtime_validation = "unexecuted"

[[threadfence.variant]]
abi_id = "i0298"
scope = "cta"

[[threadfence.variant]]
abi_id = "i0299"
scope = "device"

[[threadfence.variant]]
abi_id = "i0300"
scope = "system"
"#
        ))
        .unwrap()
    };
    let path = Path::new("intrinsics/overlay/threadfence.toml");

    let old = shard(THREADFENCE_SHARD_SCHEMA - 1);
    assert!(
        validate_overlay_shard_schema(&old, path)
            .unwrap_err()
            .to_string()
            .contains("requires overlay shard schema 34")
    );

    let current = shard(THREADFENCE_SHARD_SCHEMA);
    validate_overlay_shard_schema(&current, path).unwrap();
    let admission = current.threadfence.unwrap();
    let records = expand_threadfence_admission(&admission).unwrap();
    assert_eq!(records.len(), 3);
    assert_eq!(
        records
            .iter()
            .map(|record| (record.abi_id.as_str(), record.id.as_str()))
            .collect::<Vec<_>>(),
        [
            ("i0298", "threadfence_block"),
            ("i0299", "threadfence"),
            ("i0300", "threadfence_system"),
        ]
    );

    let mut reordered = admission.clone();
    reordered.variants.swap(0, 1);
    assert!(expand_threadfence_admission(&reordered).is_err());

    let mut wrong_id = admission.clone();
    wrong_id.variants[0].abi_id = "i0300".into();
    assert!(expand_threadfence_admission(&wrong_id).is_err());

    let mut executed = admission.clone();
    executed.runtime_validation = RuntimeValidation::Executed;
    assert!(expand_threadfence_admission(&executed).is_err());
}

#[test]
fn pinned_threadfences_match_llvm_and_reject_contract_drift() {
    let records = expand_threadfence_admission(&test_threadfence_admission()).unwrap();
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
        assert_eq!(
            declaration
                .selections
                .iter()
                .filter(|selection| selection_matches_policy(record, selection).unwrap())
                .count(),
            1
        );
    }

    let declaration = declarations["int_nvvm_membar_cta"];
    let mut changed = records[0].clone();
    changed.memory = "none".into();
    assert!(validate_imported_policy(&changed, declaration).is_err());

    let mut changed = records[0].clone();
    changed.convergent = true;
    assert!(validate_imported_policy(&changed, declaration).is_err());

    let mut changed = records[0].clone();
    changed.backend_lowerings[0].mechanism = BackendLoweringMechanism::InlinePtx;
    assert!(validate_imported_policy(&changed, declaration).is_err());

    let mut changed = records[0].clone();
    changed.minimum_ptx = "2.0".into();
    assert!(validate_imported_policy(&changed, declaration).is_err());
}

#[test]
fn pinned_cluster_sregs_preserve_helpers_and_reject_unused_w_components() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let (mut overlay, _) =
        read_overlay(&repo_root, &repo_root.join("intrinsics/overlay.toml")).unwrap();
    bind_pinned_abi_ids(&repo_root, &mut overlay);
    let imported: ImportedFile = read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
    let declarations = imported
        .intrinsics
        .iter()
        .map(|record| (record.source_record.as_str(), record))
        .collect::<BTreeMap<_, _>>();
    let records = overlay
        .intrinsics
        .iter()
        .filter(|record| {
            record
                .source_record
                .as_deref()
                .is_some_and(is_cluster_sreg_source)
        })
        .collect::<Vec<_>>();

    assert_eq!(records.len(), 14);
    let actual_abi_ids = records
        .iter()
        .map(|record| record.abi_id.clone())
        .collect::<BTreeSet<_>>();
    let expected_abi_ids = (263..=276)
        .map(|id| format!("i{id:04}"))
        .collect::<BTreeSet<_>>();
    assert_eq!(actual_abi_ids, expected_abi_ids);
    for record in &records {
        let declaration = declarations[record.source_record.as_deref().unwrap()];
        validate_imported_policy(record, declaration).unwrap();
        assert!(!declaration.source_record.ends_with("_w"));
    }

    let compatibility_paths = records
        .iter()
        .flat_map(|record| record.compatibility_rust_paths.iter().map(String::as_str))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        compatibility_paths,
        [
            "cuda_device::cluster::cluster_ctaidX",
            "cuda_device::cluster::cluster_ctaidY",
            "cuda_device::cluster::cluster_ctaidZ",
            "cuda_device::cluster::cluster_nctaidX",
            "cuda_device::cluster::cluster_nctaidY",
            "cuda_device::cluster::cluster_nctaidZ",
            "cuda_device::cluster::__cluster_grid_dimX",
            "cuda_device::cluster::__cluster_grid_dimY",
            "cuda_device::cluster::__cluster_grid_dimZ",
            "cuda_device::cluster::__cluster_idxX",
            "cuda_device::cluster::__cluster_idxY",
            "cuda_device::cluster::__cluster_idxZ",
        ]
        .into_iter()
        .collect()
    );

    let source = resolve_policy_source(records[0]).unwrap();
    let error = validate_sreg_policy(
        records[0],
        &source,
        Some(declarations["int_nvvm_read_ptx_sreg_cluster_ctaid_w"]),
    )
    .unwrap_err();
    assert!(error.to_string().contains("unused always-zero"));

    let mut mixed = (*records[0]).clone();
    mixed.sparse_mma = overlay
        .intrinsics
        .iter()
        .find_map(|record| record.sparse_mma.clone());
    let source = resolve_policy_source(&mixed).unwrap();
    let error = validate_sreg_policy(
        &mixed,
        &source,
        Some(declarations[mixed.source_record.as_deref().unwrap()]),
    )
    .unwrap_err();
    assert!(error.to_string().contains("mixes another generated-family"));
}
