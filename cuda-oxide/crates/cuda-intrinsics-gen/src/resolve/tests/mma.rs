/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, CatalogHardwareAlternative, CatalogHardwareTarget,
    EvidenceArtifactKind, EvidenceStageKind, ImportedFile, IntrinsicBackend, OverlayIntrinsic,
    OverlayShardFile, PreSm70MemberMaskRule, RegisterMmaAccumulator, RegisterMmaAdapter,
    RegisterMmaCompatibilitySource, RegisterMmaElement, RegisterMmaKind, RegisterMmaOperation,
    RegisterMmaOverflow, RegisterMmaShape, RuntimeValidation, SparseMmaAccumulator,
    SparseMmaAdapter, SparseMmaElement, SparseMmaLlvmAdapter, SparseMmaMetadata, SparseMmaOverflow,
    SparseMmaSelector, SparseMmaShape, WarpBarrierAdapter, WarpBarrierMaskEncoding,
    WarpBarrierMemoryOrdering, WarpBarrierParticipation,
};
use crate::ptx::OperandPattern;
use crate::util::read_json;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use super::fixtures::*;
use crate::resolve::driver::*;
use crate::resolve::families::*;
use crate::resolve::guards::*;
use crate::resolve::overlay::*;

#[test]
fn stmatrix_admission_is_closed_and_uses_schema_35() {
    let shard = |schema| {
        toml::from_str::<OverlayShardFile>(&format!(
            r#"
schema = {schema}
family = "stmatrix"

[stmatrix]
llvm_evidence_profile = "llvm-test"
libnvvm_evidence_profile = "libnvvm-test"
runtime_validation = "unexecuted"

[[stmatrix.variant]]
abi_id = "i0301"
multiplicity = "x2"
layout = "normal"

[[stmatrix.variant]]
abi_id = "i0302"
multiplicity = "x2"
layout = "transposed"

[[stmatrix.variant]]
abi_id = "i0303"
multiplicity = "x4"
layout = "normal"

[[stmatrix.variant]]
abi_id = "i0304"
multiplicity = "x4"
layout = "transposed"
"#
        ))
        .unwrap()
    };
    let path = Path::new("intrinsics/overlay/stmatrix.toml");

    let old = shard(STMATRIX_SHARD_SCHEMA - 1);
    assert!(
        validate_overlay_shard_schema(&old, path)
            .unwrap_err()
            .to_string()
            .contains("requires overlay shard schema 35")
    );

    let current = shard(STMATRIX_SHARD_SCHEMA);
    validate_overlay_shard_schema(&current, path).unwrap();
    let admission = current.stmatrix.unwrap();
    let records = expand_stmatrix_admission(&admission).unwrap();
    assert_eq!(
        records
            .iter()
            .map(|record| (record.abi_id.as_str(), record.id.as_str()))
            .collect::<Vec<_>>(),
        [
            ("i0301", "stmatrix_m8n8_x2_b16"),
            ("i0302", "stmatrix_m8n8_x2_trans_b16"),
            ("i0303", "stmatrix_m8n8_x4_b16"),
            ("i0304", "stmatrix_m8n8_x4_trans_b16"),
        ]
    );

    let mut reordered = admission.clone();
    reordered.variants.swap(0, 1);
    assert!(expand_stmatrix_admission(&reordered).is_err());

    let mut wrong_id = admission.clone();
    wrong_id.variants[0].abi_id = "i0302".into();
    assert!(expand_stmatrix_admission(&wrong_id).is_err());

    let mut executed = admission;
    executed.runtime_validation = RuntimeValidation::Executed;
    assert!(expand_stmatrix_admission(&executed).is_err());
}

#[test]
fn pinned_stmatrix_records_match_llvm_and_reject_contract_drift() {
    let records = expand_stmatrix_admission(&test_stmatrix_admission()).unwrap();
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let imported: ImportedFile = read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
    let declarations = imported
        .intrinsics
        .iter()
        .map(|record| (record.source_record.as_str(), record))
        .collect::<BTreeMap<_, _>>();

    for record in &records {
        let declaration = declarations[record.source_record.as_deref().unwrap()];
        assert!(declaration.selections.is_empty());
        validate_imported_policy(record, declaration).unwrap();
    }

    let declaration = declarations["int_nvvm_stmatrix_sync_aligned_m8n8_x2_b16"];
    let mut changed = records[0].clone();
    changed.memory = "read_write".into();
    assert!(validate_imported_policy(&changed, declaration).is_err());

    let mut changed = records[0].clone();
    changed.convergent = false;
    assert!(validate_imported_policy(&changed, declaration).is_err());

    let mut changed = records[0].clone();
    changed.backend_lowerings[0].mechanism = BackendLoweringMechanism::InlinePtx;
    assert!(validate_imported_policy(&changed, declaration).is_err());

    let mut changed = records[0].clone();
    changed.minimum_sm = Some("sm_80".into());
    assert!(validate_imported_policy(&changed, declaration).is_err());
}

#[test]
fn compact_f8f6f4_axes_require_the_exact_canonical_matrix() {
    let records = expand_sparse_mma_f8f6f4_admission(&test_f8f6f4_admission()).unwrap();
    assert_eq!(records.len(), 25);
    assert_eq!(
        records[0].id,
        "mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e2m1_e2m1_f32"
    );
    assert_eq!(records[0].rust_arguments[0], "[f32; 4]");
    assert_eq!(records[0].dialect_results, ["f32"; 4]);
    assert_eq!(records[0].llvm_results, ["f32"; 4]);

    let mut missing = test_f8f6f4_admission();
    missing.a_elements.pop();
    assert!(expand_sparse_mma_f8f6f4_admission(&missing).is_err());

    let mut duplicate = test_f8f6f4_admission();
    duplicate.a_elements[4] = SparseMmaElement::E4m3;
    assert!(expand_sparse_mma_f8f6f4_admission(&duplicate).is_err());

    let mut extra = test_f8f6f4_admission();
    extra.b_elements.push(SparseMmaElement::S4);
    assert!(expand_sparse_mma_f8f6f4_admission(&extra).is_err());

    let mut unsorted = test_f8f6f4_admission();
    unsorted.b_elements.swap(0, 1);
    assert!(expand_sparse_mma_f8f6f4_admission(&unsorted).is_err());

    let mut wrong_count = test_f8f6f4_admission();
    wrong_count.product_count = 24;
    assert!(expand_sparse_mma_f8f6f4_admission(&wrong_count).is_err());
}

#[test]
fn compact_sparse_f8f6f4_f16_admission_is_closed_and_ordered() {
    let admission = test_sparse_mma_f8f6f4_f16_admission();
    let records = expand_sparse_mma_f8f6f4_f16_admission(&admission).unwrap();
    assert_eq!(records.len(), 25);
    assert!(records.iter().all(|record| record.abi_id.is_empty()));
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut bound = overlay_file(records.clone());
    bind_pinned_abi_ids(&repo_root, &mut bound);
    assert_eq!(
        bound
            .intrinsics
            .iter()
            .map(|record| record.abi_id.clone())
            .collect::<Vec<_>>(),
        (525..=549)
            .map(|id| format!("i{id:04}"))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        records[0].id,
        "mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e2m1_e2m1_f16"
    );
    assert_eq!(
        records[24].id,
        "mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e5m2_e5m2_f16"
    );
    let first = &records[0];
    assert_eq!(
        first.source_record.as_deref(),
        Some("int_nvvm_mma_sp_ordered_metadata_m16n8k64_row_col_kind_f8f6f4_f16_e2m1_e2m1_f16")
    );
    assert_eq!(
        first.llvm_symbol.as_deref(),
        Some("llvm.nvvm.mma.sp.ordered.metadata.m16n8k64.row.col.kind.f8f6f4.f16.e2m1.e2m1.f16")
    );
    assert_eq!(
        first.rust_arguments,
        ["[u32; 2]", "[u32; 4]", "[u32; 4]", "u32", "u32"]
    );
    assert_eq!(first.rust_result, "[u32; 2]");
    assert_eq!(first.dialect_results, ["i32"; 2]);
    assert_eq!(first.llvm_results, ["v2f16"; 2]);
    assert_eq!(
        first.llvm_arguments,
        [
            "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "v2f16", "v2f16", "i32", "i32"
        ]
    );
    assert_eq!(first.minimum_ptx, "8.7");
    assert_eq!(first.minimum_sm, None);
    assert_eq!(first.targets, SPARSE_MMA_F8F6F4_TARGETS);
    assert!(first.convergent && !first.pure);
    assert!(first.backend_lowerings.iter().all(|route| {
        route.mechanism == BackendLoweringMechanism::InlinePtx
            && route.minimum_ptx.as_deref() == Some("8.7")
            && route.minimum_sm.is_none()
    }));
    let mma = first.sparse_mma.as_ref().unwrap();
    assert_eq!(mma.accumulator, SparseMmaAccumulator::F16);
    assert_eq!(mma.selector, SparseMmaSelector::ImmediateZero);
    assert_eq!(
        mma.adapter,
        SparseMmaAdapter::C2U32A4U32B4U32MetadataU32SelectorU32ToD2U32
    );
    assert_eq!(
        mma.llvm_adapter,
        SparseMmaLlvmAdapter::A4I32B4I32C2V2F16MetadataI32SelectorI32ToD2V2F16
    );
    assert_eq!(
        first.compatibility_rust_paths,
        ["cuda_device::wmma::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e2m1_e2m1_f16"]
    );
    assert_eq!(
        first.expected_ptx.operands,
        [
            OperandPattern::RegisterList { length: 2 },
            OperandPattern::RegisterList { length: 4 },
            OperandPattern::RegisterList { length: 4 },
            OperandPattern::RegisterList { length: 2 },
            OperandPattern::Register,
            OperandPattern::Immediate,
        ]
    );

    let mut missing = admission.clone();
    missing.a_elements.pop();
    assert!(expand_sparse_mma_f8f6f4_f16_admission(&missing).is_err());

    let mut reordered = admission.clone();
    reordered.b_elements.swap(0, 1);
    assert!(expand_sparse_mma_f8f6f4_f16_admission(&reordered).is_err());

    let mut wrong_count = admission.clone();
    wrong_count.product_count = 24;
    assert!(expand_sparse_mma_f8f6f4_f16_admission(&wrong_count).is_err());

    let mut legacy_first_id = admission.clone();
    legacy_first_id._legacy_first_abi_id = Some("i9999".into());
    assert!(
        expand_sparse_mma_f8f6f4_f16_admission(&legacy_first_id)
            .unwrap()
            .iter()
            .all(|record| record.abi_id.is_empty())
    );

    let mut missing_evidence = admission.clone();
    missing_evidence.llvm_evidence_profile.clear();
    assert!(expand_sparse_mma_f8f6f4_f16_admission(&missing_evidence).is_err());

    let mut executed = admission;
    executed.runtime_validation = RuntimeValidation::Executed;
    assert!(expand_sparse_mma_f8f6f4_f16_admission(&executed).is_err());
}

#[test]
fn compact_sparse_ordered_ampere_float_admission_is_closed_and_ledger_bound() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let path = repo_root.join("intrinsics/overlay/sparse_mma_ordered_ampere_float.toml");
    let bytes = fs::read(&path).unwrap();
    let shard: OverlayShardFile = toml::from_slice(&bytes).unwrap();
    let admission = shard.sparse_mma_ordered_ampere_float.unwrap();

    // Executed runtime validation is admitted for this family (GPU evidence backs it).
    assert_eq!(admission.runtime_validation, RuntimeValidation::Executed);
    let records = expand_sparse_mma_ordered_ampere_float_admission(&admission).unwrap();
    assert_eq!(records.len(), 8);
    assert!(records.iter().all(|record| record.abi_id.is_empty()));
    assert!(records.iter().all(|record| {
        record
            .sparse_mma
            .as_ref()
            .is_some_and(|mma| mma.runtime_validation == RuntimeValidation::Executed)
    }));

    let mut bound = overlay_file(records.clone());
    bind_pinned_abi_ids(&repo_root, &mut bound);
    assert_eq!(
        bound
            .intrinsics
            .iter()
            .map(|record| record.abi_id.clone())
            .collect::<Vec<_>>(),
        (1018..=1025)
            .map(|id| format!("i{id:04}"))
            .collect::<Vec<_>>()
    );

    // First form: m16n8k8 tf32 (2-reg carrier, selector 0-3).
    let first = &records[0];
    assert_eq!(first.id, "mma_sp_ordered_metadata_m16n8k8_f32_tf32");
    assert_eq!(
        first.source_record.as_deref(),
        Some("int_nvvm_mma_sp_ordered_metadata_m16n8k8_row_col_tf32")
    );
    assert_eq!(
        first.llvm_symbol.as_deref(),
        Some("llvm.nvvm.mma.sp.ordered.metadata.m16n8k8.row.col.tf32")
    );
    assert_eq!(
        first.rust_arguments,
        ["[f32; 4]", "[u32; 2]", "[u32; 2]", "u32", "u32"]
    );
    assert_eq!(first.rust_result, "[f32; 4]");
    assert_eq!(
        first.llvm_arguments,
        [
            "i32", "i32", "i32", "i32", "f32", "f32", "f32", "f32", "i32", "i32"
        ]
    );
    assert_eq!(first.minimum_ptx, "8.5");
    assert_eq!(first.minimum_sm.as_deref(), Some("sm_80"));
    assert_eq!(first.targets, "all");
    let mma = first.sparse_mma.as_ref().unwrap();
    assert_eq!(mma.selector, SparseMmaSelector::ImmediateZeroThroughThree);
    assert_eq!(
        mma.adapter,
        SparseMmaAdapter::C4F32A2U32B2U32MetadataU32SelectorU32ToD4F32
    );
    assert_eq!(
        mma.llvm_adapter,
        SparseMmaLlvmAdapter::A2I32B2I32C4F32MetadataI32SelectorI32ToD4F32
    );

    // f16 forms carry <2 x half> multiplicands; k32 pairs drop to selector 0-1.
    let k32_f16 = &records[6];
    assert_eq!(k32_f16.id, "mma_sp_ordered_metadata_m16n8k32_f16_f16");
    assert_eq!(
        k32_f16.llvm_arguments,
        [
            "v2f16", "v2f16", "v2f16", "v2f16", "v2f16", "v2f16", "v2f16", "v2f16", "v2f16",
            "v2f16", "i32", "i32"
        ]
    );
    let k32_mma = k32_f16.sparse_mma.as_ref().unwrap();
    assert_eq!(k32_mma.selector, SparseMmaSelector::ImmediateZeroOrOne);
    assert_eq!(
        k32_mma.adapter,
        SparseMmaAdapter::C2U32A4U32B4U32MetadataU32SelectorU32ToD2U32
    );

    let mut reordered = admission.clone();
    reordered.variants.swap(0, 1);
    assert!(expand_sparse_mma_ordered_ampere_float_admission(&reordered).is_err());

    let mut missing = admission.clone();
    missing.variants.pop();
    assert!(expand_sparse_mma_ordered_ampere_float_admission(&missing).is_err());

    let mut missing_evidence = admission;
    missing_evidence.llvm_evidence_profile.clear();
    assert!(expand_sparse_mma_ordered_ampere_float_admission(&missing_evidence).is_err());
}

#[test]
fn sparse_f8f6f4_f16_candidate_floor_uses_the_resolved_policy() {
    let policy = expand_sparse_mma_f8f6f4_f16_admission(&test_sparse_mma_f8f6f4_f16_admission())
        .unwrap()
        .remove(0);
    let (_, requirement) = candidate_llvm_route(&policy).unwrap();

    validate_candidate_target(&policy, &requirement, "sm_120a", "+ptx87").unwrap();
    validate_candidate_target(&policy, &requirement, "sm_120f", "+ptx88").unwrap();
    validate_candidate_target(&policy, &requirement, "sm_121a", "+ptx88").unwrap();
    validate_candidate_target(&policy, &requirement, "sm_121f", "+ptx88").unwrap();
    for target in ["sm_120f", "sm_121a", "sm_121f"] {
        assert!(
            validate_candidate_target(&policy, &requirement, target, "+ptx87").is_err(),
            "{target} must require PTX 8.8"
        );
    }
}

#[test]
fn compact_dense_f8f6f4_admission_is_closed() {
    for (accumulator, first, first_id, last_id, rust_arguments, rust_result, adapter) in [
        (
            RegisterMmaAccumulator::F32,
            454,
            "mma_m16n8k32_f32_e2m1_e2m1",
            "mma_m16n8k32_f32_e5m2_e5m2",
            ["[f32; 4]", "[u32; 4]", "[u32; 2]"],
            "[f32; 4]",
            RegisterMmaAdapter::C4F32A4U32B2U32ToD4F32,
        ),
        (
            RegisterMmaAccumulator::F16,
            479,
            "mma_m16n8k32_f16_e2m1_e2m1",
            "mma_m16n8k32_f16_e5m2_e5m2",
            ["[u32; 2]", "[u32; 4]", "[u32; 2]"],
            "[u32; 2]",
            RegisterMmaAdapter::C2U32A4U32B2U32ToD2U32,
        ),
    ] {
        let admission = test_register_mma_f8f6f4_admission(accumulator);
        let records = expand_register_mma_f8f6f4_admission(&admission, accumulator).unwrap();
        assert_eq!(records.len(), 25);
        assert!(records.iter().all(|record| record.abi_id.is_empty()));
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut bound = overlay_file(records.clone());
        bind_pinned_abi_ids(&repo_root, &mut bound);
        assert_eq!(
            bound
                .intrinsics
                .iter()
                .map(|record| record.abi_id.clone())
                .collect::<Vec<_>>(),
            (first..first + 25)
                .map(|id| format!("i{id:04}"))
                .collect::<Vec<_>>()
        );
        assert_eq!(records[0].id, first_id);
        assert_eq!(records[24].id, last_id);
        assert!(records.iter().all(|record| {
            record.rust_arguments == rust_arguments
                && record.rust_result == rust_result
                && record.register_mma.as_ref().unwrap().adapter == adapter
                && record.targets == REGISTER_MMA_F8F6F4_TARGETS
                && record.minimum_ptx == "8.7"
                && record.minimum_sm.is_none()
                && record.backend_lowerings.len() == 2
                && record
                    .backend_lowerings
                    .iter()
                    .all(|route| route.mechanism == BackendLoweringMechanism::InlinePtx)
        }));

        let mut missing = admission.clone();
        missing.a_elements.pop();
        assert!(expand_register_mma_f8f6f4_admission(&missing, accumulator).is_err());

        let mut reordered = admission.clone();
        reordered.b_elements.swap(0, 1);
        assert!(expand_register_mma_f8f6f4_admission(&reordered, accumulator).is_err());

        let mut wrong_count = admission.clone();
        wrong_count.product_count = 24;
        assert!(expand_register_mma_f8f6f4_admission(&wrong_count, accumulator).is_err());

        let mut legacy_first_id = admission.clone();
        legacy_first_id._legacy_first_abi_id = Some("i9999".into());
        assert!(
            expand_register_mma_f8f6f4_admission(&legacy_first_id, accumulator)
                .unwrap()
                .iter()
                .all(|record| record.abi_id.is_empty())
        );

        let mut reordered_targets = admission.clone();
        reordered_targets.targets.swap(0, 1);
        assert!(expand_register_mma_f8f6f4_admission(&reordered_targets, accumulator).is_err());

        let mut missing_evidence = admission.clone();
        missing_evidence.llvm_evidence_profile.clear();
        assert!(expand_register_mma_f8f6f4_admission(&missing_evidence, accumulator).is_err());

        let mut executed = admission;
        executed.runtime_validation = RuntimeValidation::Executed;
        assert!(expand_register_mma_f8f6f4_admission(&executed, accumulator).is_err());
    }

    let admission = test_register_mma_f8f6f4_admission(RegisterMmaAccumulator::F32);
    assert!(expand_register_mma_f8f6f4_admission(&admission, RegisterMmaAccumulator::F64).is_err());
}

#[test]
fn compact_standard_fp8_admission_is_closed_and_ordered() {
    let admission = test_register_mma_fp8_admission();
    let records = expand_register_mma_fp8_admission(&admission).unwrap();
    assert_eq!(records.len(), 16);
    assert!(records.iter().all(|record| record.abi_id.is_empty()));
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut bound = overlay_file(records.clone());
    bind_pinned_abi_ids(&repo_root, &mut bound);
    assert_eq!(
        bound
            .intrinsics
            .iter()
            .map(|record| record.abi_id.clone())
            .collect::<Vec<_>>(),
        (504..=519)
            .map(|id| format!("i{id:04}"))
            .collect::<Vec<_>>()
    );

    let mut index = 0;
    for &shape in &REGISTER_MMA_FP8_SHAPES {
        let (shape_name, a_count, b_count) = register_mma_fp8_shape_contract(shape).unwrap();
        for &accumulator in &REGISTER_MMA_FP8_ACCUMULATORS {
            let (scalar, arguments, result, adapter, result_count) = match accumulator {
                RegisterMmaAccumulator::F16 => (
                    "f16",
                    vec![
                        "[u32; 2]".to_owned(),
                        format!("[u32; {a_count}]"),
                        if b_count == 1 {
                            "u32".to_owned()
                        } else {
                            "[u32; 2]".to_owned()
                        },
                    ],
                    "[u32; 2]",
                    if shape == RegisterMmaShape::M16n8k16 {
                        RegisterMmaAdapter::C2U32A2U32B1U32ToD2U32
                    } else {
                        RegisterMmaAdapter::C2U32A4U32B2U32ToD2U32
                    },
                    2,
                ),
                RegisterMmaAccumulator::F32 => (
                    "f32",
                    vec![
                        "[f32; 4]".to_owned(),
                        format!("[u32; {a_count}]"),
                        if b_count == 1 {
                            "u32".to_owned()
                        } else {
                            "[u32; 2]".to_owned()
                        },
                    ],
                    "[f32; 4]",
                    if shape == RegisterMmaShape::M16n8k16 {
                        RegisterMmaAdapter::C4F32A2U32B1U32ToD4F32
                    } else {
                        RegisterMmaAdapter::C4F32A4U32B2U32ToD4F32
                    },
                    4,
                ),
                _ => unreachable!(),
            };
            for &a_element in &REGISTER_MMA_FP8_ELEMENTS {
                for &b_element in &REGISTER_MMA_FP8_ELEMENTS {
                    let record = &records[index];
                    let a = register_mma_fp8_element_name(a_element).unwrap();
                    let b = register_mma_fp8_element_name(b_element).unwrap();
                    assert_eq!(record.id, format!("mma_{shape_name}_fp8_{scalar}_{a}_{b}"));
                    assert_eq!(
                        record.operation_key,
                        format!(
                            "matrix.mma.{shape_name}.row.col.standard_fp8.{scalar}.{a}.{b}.{scalar}"
                        )
                    );
                    assert_eq!(
                        record.source_record.as_deref(),
                        Some(
                            format!("int_nvvm_mma_{shape_name}_row_col_{scalar}_{a}_{b}_{scalar}")
                                .as_str()
                        )
                    );
                    assert_eq!(record.rust_arguments, arguments);
                    assert_eq!(record.rust_result, result);
                    assert_eq!(
                        record.minimum_ptx,
                        register_mma_fp8_minimum_ptx(shape, accumulator)
                    );
                    assert_eq!(record.minimum_sm.as_deref(), Some("sm_89"));
                    assert_eq!(
                        record.expected_ptx.operands,
                        [result_count, a_count, b_count, result_count]
                            .map(|length| OperandPattern::RegisterList { length })
                    );
                    let mma = record.register_mma.as_ref().unwrap();
                    assert_eq!(mma.kind, Some(RegisterMmaKind::Standard));
                    assert_eq!(mma.adapter, adapter);
                    index += 1;
                }
            }
        }
    }
    assert_eq!(index, records.len());

    let mut reordered = admission.clone();
    reordered.shapes.swap(0, 1);
    assert!(expand_register_mma_fp8_admission(&reordered).is_err());
    let mut reordered = admission.clone();
    reordered.accumulators.swap(0, 1);
    assert!(expand_register_mma_fp8_admission(&reordered).is_err());
    let mut reordered = admission.clone();
    reordered.a_elements.swap(0, 1);
    assert!(expand_register_mma_fp8_admission(&reordered).is_err());
    let mut reordered = admission.clone();
    reordered.b_elements.swap(0, 1);
    assert!(expand_register_mma_fp8_admission(&reordered).is_err());
    let mut wrong = admission.clone();
    wrong.product_count = 15;
    assert!(expand_register_mma_fp8_admission(&wrong).is_err());
    let mut legacy_first_id = admission.clone();
    legacy_first_id._legacy_first_abi_id = Some("i9999".into());
    assert!(
        expand_register_mma_fp8_admission(&legacy_first_id)
            .unwrap()
            .iter()
            .all(|record| record.abi_id.is_empty())
    );
    let mut wrong = admission;
    wrong.runtime_validation = RuntimeValidation::Executed;
    assert!(expand_register_mma_fp8_admission(&wrong).is_err());
}

#[test]
fn compact_ampere_float_mma_admission_is_closed_and_ordered() {
    let admission = test_register_mma_ampere_float_admission();
    let records = expand_register_mma_ampere_float_admission(&admission).unwrap();
    assert_eq!(records.len(), 5);
    assert!(records.iter().all(|record| record.abi_id.is_empty()));
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut bound = overlay_file(records.clone());
    bind_pinned_abi_ids(&repo_root, &mut bound);
    assert_eq!(
        bound
            .intrinsics
            .iter()
            .map(|record| (record.abi_id.as_str(), record.id.as_str()))
            .collect::<Vec<_>>(),
        [
            ("i0520", "mma_m16n8k4_f32_tf32"),
            ("i0521", "mma_m16n8k8_f16_f16"),
            ("i0522", "mma_m16n8k8_f32_bf16"),
            ("i0523", "mma_m16n8k8_f32_f16"),
            ("i0524", "mma_m16n8k16_f16_f16"),
        ]
    );
    for record in &records {
        assert_eq!(record.minimum_ptx, "7.0");
        assert_eq!(record.minimum_sm.as_deref(), Some("sm_80"));
        assert_eq!(record.backend_lowerings.len(), 2);
        assert!(record.backend_lowerings.iter().all(|lowering| {
            lowering.mechanism == BackendLoweringMechanism::InlinePtx
                && lowering.minimum_ptx.as_deref() == Some("7.0")
                && lowering.minimum_sm.as_deref() == Some("sm_80")
        }));
        let mma = record.register_mma.as_ref().unwrap();
        assert_eq!(mma.kind, None);
        assert_eq!(
            mma.compatibility_source,
            RegisterMmaCompatibilitySource::GeneratedStub
        );
    }

    let mut reordered = admission.clone();
    reordered.variants.swap(0, 1);
    assert!(expand_register_mma_ampere_float_admission(&reordered).is_err());
    let mut legacy_first_id = admission.clone();
    legacy_first_id._legacy_first_abi_id = Some("i9999".into());
    assert!(
        expand_register_mma_ampere_float_admission(&legacy_first_id)
            .unwrap()
            .iter()
            .all(|record| record.abi_id.is_empty())
    );
    let mut wrong = admission.clone();
    wrong.product_count = 4;
    assert!(expand_register_mma_ampere_float_admission(&wrong).is_err());
    let mut wrong = admission.clone();
    wrong.llvm_evidence_profile.clear();
    assert!(expand_register_mma_ampere_float_admission(&wrong).is_err());
    let mut wrong = admission;
    wrong.runtime_validation = RuntimeValidation::Executed;
    assert!(expand_register_mma_ampere_float_admission(&wrong).is_err());

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let path = repo_root.join("intrinsics/overlay/register_mma_ampere_float.toml");
    let bytes = fs::read(&path).unwrap();
    let mut shard: OverlayShardFile = toml::from_slice(&bytes).unwrap();
    validate_overlay_shard_schema(&shard, &path).unwrap();
    shard.schema = REGISTER_MMA_AMPERE_FLOAT_SHARD_SCHEMA - 1;
    assert!(
        validate_overlay_shard_schema(&shard, &path)
            .unwrap_err()
            .to_string()
            .contains("requires overlay shard schema 49")
    );
}

#[test]
fn ampere_float_mma_policies_match_llvm_and_fail_closed() {
    let records =
        expand_register_mma_ampere_float_admission(&test_register_mma_ampere_float_admission())
            .unwrap();
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut overlay = overlay_file(records);
    bind_pinned_abi_ids(&repo_root, &mut overlay);
    let records = overlay.intrinsics;
    let imported: ImportedFile = read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
    let declarations = imported
        .intrinsics
        .iter()
        .map(|record| (record.source_record.as_str(), record))
        .collect::<BTreeMap<_, _>>();

    for record in &records {
        let declaration = declarations[record.source_record.as_deref().unwrap()];
        validate_imported_policy(record, declaration).unwrap();
        let expected_predicates = match record.id.as_str() {
            "mma_m16n8k4_f32_tf32" | "mma_m16n8k16_f16_f16" => [
                "Subtarget->getSmVersion() >= 80",
                "Subtarget->getPTXVersion() >= 70",
            ],
            _ => [
                "Subtarget->getPTXVersion() >= 65",
                "Subtarget->getSmVersion() >= 75",
            ],
        };
        assert_eq!(declaration.selections[0].predicates, expected_predicates);
        assert_eq!(record.minimum_ptx, "7.0");
        assert_eq!(record.minimum_sm.as_deref(), Some("sm_80"));
    }

    let valid = &records[2];
    assert_eq!(valid.id, "mma_m16n8k8_f32_bf16");
    assert_eq!(
        declarations[valid.source_record.as_deref().unwrap()].selections[0].predicates,
        [
            "Subtarget->getPTXVersion() >= 65",
            "Subtarget->getSmVersion() >= 75",
        ]
    );
    assert_eq!(valid.minimum_ptx, "7.0");
    assert_eq!(valid.minimum_sm.as_deref(), Some("sm_80"));
    assert!(valid.backend_lowerings.iter().all(|lowering| {
        lowering.minimum_ptx.as_deref() == Some("7.0")
            && lowering.minimum_sm.as_deref() == Some("sm_80")
    }));
    let declaration = declarations[valid.source_record.as_deref().unwrap()];
    let mut wrong = valid.clone();
    wrong.minimum_ptx = "6.5".into();
    assert!(validate_imported_policy(&wrong, declaration).is_err());
    let mut wrong = valid.clone();
    wrong.register_mma.as_mut().unwrap().adapter = RegisterMmaAdapter::C4F32A4U32B2U32ToD4F32;
    assert!(validate_imported_policy(&wrong, declaration).is_err());
    let mut wrong = valid.clone();
    wrong.register_mma.as_mut().unwrap().kind = Some(RegisterMmaKind::Standard);
    assert!(validate_imported_policy(&wrong, declaration).is_err());
    let mut wrong_declaration = declaration.clone();
    wrong_declaration.selections[0].predicates[0] = "Subtarget->getPTXVersion() >= 70".into();
    assert!(validate_imported_policy(valid, &wrong_declaration).is_err());
}

#[test]
fn standard_fp8_policies_match_llvm_and_fail_closed() {
    let records = expand_register_mma_fp8_admission(&test_register_mma_fp8_admission()).unwrap();
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut overlay = overlay_file(records);
    bind_pinned_abi_ids(&repo_root, &mut overlay);
    let records = overlay.intrinsics;
    let imported: ImportedFile = read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
    let declarations = imported
        .intrinsics
        .iter()
        .map(|record| (record.source_record.as_str(), record))
        .collect::<BTreeMap<_, _>>();

    for record in &records {
        let declaration = declarations[record.source_record.as_deref().unwrap()];
        validate_imported_policy(record, declaration).unwrap();
        let mma = record.register_mma.as_ref().unwrap();
        let source_floor = declaration.selections[0]
            .predicates
            .iter()
            .find(|predicate| predicate.contains("getPTXVersion"))
            .unwrap();
        if mma.shape == RegisterMmaShape::M16n8k32 {
            assert!(source_floor.ends_with(">= 84"));
            if mma.accumulator == RegisterMmaAccumulator::F16 {
                assert_eq!(record.minimum_ptx, "8.7");
            } else {
                assert_eq!(record.minimum_ptx, "8.4");
            }
        } else {
            assert!(source_floor.ends_with(">= 87"));
            assert_eq!(record.minimum_ptx, "8.7");
        }
    }

    let valid = &records[0];
    let declaration = declarations[valid.source_record.as_deref().unwrap()];
    let mut wrong = valid.clone();
    wrong.register_mma.as_mut().unwrap().kind = None;
    assert!(validate_imported_policy(&wrong, declaration).is_err());
    let mut wrong = valid.clone();
    wrong.register_mma.as_mut().unwrap().adapter = RegisterMmaAdapter::C2U32A4U32B2U32ToD2U32;
    assert!(validate_imported_policy(&wrong, declaration).is_err());
    let mut wrong_declaration = declaration.clone();
    wrong_declaration.selections[0].predicates[1] = "Subtarget->getPTXVersion() >= 84".into();
    assert!(validate_imported_policy(valid, &wrong_declaration).is_err());
    let mut wrong = valid.clone();
    wrong.minimum_ptx = "8.4".into();
    assert!(validate_imported_policy(&wrong, declaration).is_err());

    let mut old = expand_register_mma_f8f6f4_admission(
        &test_register_mma_f8f6f4_admission(RegisterMmaAccumulator::F32),
        RegisterMmaAccumulator::F32,
    )
    .unwrap()
    .remove(18);
    let mut standard = records[12].clone();
    assert_ne!(old.id, standard.id);
    standard.id = old.id.clone();
    assert!(validate_unique_overlay(&[old.clone(), standard], 1).is_err());
    old.register_mma.as_mut().unwrap().kind = Some(RegisterMmaKind::F8f6f4);
    assert_eq!(
        old.register_mma.as_ref().unwrap().kind,
        Some(RegisterMmaKind::F8f6f4)
    );
}

#[test]
fn standard_fp8_resolves_exact_routes_and_target_floors() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = resolve(&repo_root).unwrap();
    let records = catalog
        .intrinsics
        .iter()
        .filter(|record| {
            record.rust.abi_id.as_str() >= "i0504" && record.rust.abi_id.as_str() <= "i0519"
        })
        .collect::<Vec<_>>();
    let expected = [
        (
            "mma_m16n8k16_fp8_f16_e4m3_e4m3",
            "i0504",
            RegisterMmaShape::M16n8k16,
            RegisterMmaAccumulator::F16,
            87,
        ),
        (
            "mma_m16n8k16_fp8_f16_e4m3_e5m2",
            "i0505",
            RegisterMmaShape::M16n8k16,
            RegisterMmaAccumulator::F16,
            87,
        ),
        (
            "mma_m16n8k16_fp8_f16_e5m2_e4m3",
            "i0506",
            RegisterMmaShape::M16n8k16,
            RegisterMmaAccumulator::F16,
            87,
        ),
        (
            "mma_m16n8k16_fp8_f16_e5m2_e5m2",
            "i0507",
            RegisterMmaShape::M16n8k16,
            RegisterMmaAccumulator::F16,
            87,
        ),
        (
            "mma_m16n8k16_fp8_f32_e4m3_e4m3",
            "i0508",
            RegisterMmaShape::M16n8k16,
            RegisterMmaAccumulator::F32,
            87,
        ),
        (
            "mma_m16n8k16_fp8_f32_e4m3_e5m2",
            "i0509",
            RegisterMmaShape::M16n8k16,
            RegisterMmaAccumulator::F32,
            87,
        ),
        (
            "mma_m16n8k16_fp8_f32_e5m2_e4m3",
            "i0510",
            RegisterMmaShape::M16n8k16,
            RegisterMmaAccumulator::F32,
            87,
        ),
        (
            "mma_m16n8k16_fp8_f32_e5m2_e5m2",
            "i0511",
            RegisterMmaShape::M16n8k16,
            RegisterMmaAccumulator::F32,
            87,
        ),
        (
            "mma_m16n8k32_fp8_f16_e4m3_e4m3",
            "i0512",
            RegisterMmaShape::M16n8k32,
            RegisterMmaAccumulator::F16,
            87,
        ),
        (
            "mma_m16n8k32_fp8_f16_e4m3_e5m2",
            "i0513",
            RegisterMmaShape::M16n8k32,
            RegisterMmaAccumulator::F16,
            87,
        ),
        (
            "mma_m16n8k32_fp8_f16_e5m2_e4m3",
            "i0514",
            RegisterMmaShape::M16n8k32,
            RegisterMmaAccumulator::F16,
            87,
        ),
        (
            "mma_m16n8k32_fp8_f16_e5m2_e5m2",
            "i0515",
            RegisterMmaShape::M16n8k32,
            RegisterMmaAccumulator::F16,
            87,
        ),
        (
            "mma_m16n8k32_fp8_f32_e4m3_e4m3",
            "i0516",
            RegisterMmaShape::M16n8k32,
            RegisterMmaAccumulator::F32,
            84,
        ),
        (
            "mma_m16n8k32_fp8_f32_e4m3_e5m2",
            "i0517",
            RegisterMmaShape::M16n8k32,
            RegisterMmaAccumulator::F32,
            84,
        ),
        (
            "mma_m16n8k32_fp8_f32_e5m2_e4m3",
            "i0518",
            RegisterMmaShape::M16n8k32,
            RegisterMmaAccumulator::F32,
            84,
        ),
        (
            "mma_m16n8k32_fp8_f32_e5m2_e5m2",
            "i0519",
            RegisterMmaShape::M16n8k32,
            RegisterMmaAccumulator::F32,
            84,
        ),
    ];
    assert_eq!(records.len(), expected.len());
    let sm_89 = CatalogHardwareTarget::AnyOf {
        alternatives: vec![CatalogHardwareAlternative::MinimumSm { sm: 89 }],
    };
    let expected_routes = [
        (
            IntrinsicBackend::LlvmNvptx,
            BackendLoweringMechanism::InlinePtx,
            "rust-llvm-23.1.0-16696adc",
        ),
        (
            IntrinsicBackend::LibNvvm,
            BackendLoweringMechanism::InlinePtx,
            "cuda-13.3-libnvvm-13.3.33",
        ),
    ];
    let mut floor_groups = BTreeMap::new();

    for (record, &(id, abi_id, shape, accumulator, minimum_ptx)) in records.iter().zip(&expected) {
        assert_eq!(record.id, id);
        assert_eq!(record.rust.abi_id, abi_id);
        let mma = record.register_mma.as_ref().unwrap();
        assert_eq!((mma.shape, mma.accumulator), (shape, accumulator));
        assert_eq!(record.target.minimum_ptx.encoded(), minimum_ptx);
        assert_eq!(record.target.hardware, sm_89);
        assert_eq!(
            record
                .backend_lowerings
                .iter()
                .map(|route| (
                    route.backend,
                    route.mechanism,
                    route.evidence_profile.as_str()
                ))
                .collect::<Vec<_>>(),
            expected_routes
        );
        for route in &record.backend_lowerings {
            assert_eq!(route.target.minimum_ptx.encoded(), minimum_ptx);
            assert_eq!(route.target.hardware, sm_89);
        }
        *floor_groups
            .entry((shape, accumulator, minimum_ptx))
            .or_insert(0) += 1;

        if shape == RegisterMmaShape::M16n8k32 && accumulator == RegisterMmaAccumulator::F16 {
            assert_eq!(
                record
                    .selections
                    .iter()
                    .flat_map(|selection| &selection.predicates)
                    .filter(|predicate| predicate.contains("getPTXVersion"))
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
                ["Subtarget->getPTXVersion() >= 84"]
            );
            let llvm_route = &record.backend_lowerings[0];
            assert!(llvm_route.stages.iter().any(|stage| {
                stage.stage == EvidenceStageKind::PtxAssembly
                    && stage.mechanism == Some(BackendLoweringMechanism::InlinePtx)
                    && stage.outcome == "succeeded"
                    && stage.targets == ["sm_89", "ptx87"]
                    && stage.artifact_kind == Some(EvidenceArtifactKind::Cubin)
            }));
            assert!(llvm_route.stages.iter().any(|stage| {
                stage.stage == EvidenceStageKind::PtxAssembly
                    && stage.mechanism == Some(BackendLoweringMechanism::InlinePtx)
                    && stage.outcome == "failed"
                    && stage.targets == ["sm_89", "ptx86"]
            }));
        }
    }
    assert_eq!(
        floor_groups,
        BTreeMap::from([
            (
                (RegisterMmaShape::M16n8k16, RegisterMmaAccumulator::F16, 87),
                4
            ),
            (
                (RegisterMmaShape::M16n8k16, RegisterMmaAccumulator::F32, 87),
                4
            ),
            (
                (RegisterMmaShape::M16n8k32, RegisterMmaAccumulator::F16, 87),
                4
            ),
            (
                (RegisterMmaShape::M16n8k32, RegisterMmaAccumulator::F32, 84),
                4
            ),
        ])
    );
}

#[test]
fn dense_f8f6f4_candidate_floor_uses_the_resolved_policy() {
    for accumulator in [RegisterMmaAccumulator::F16, RegisterMmaAccumulator::F32] {
        let policy = expand_register_mma_f8f6f4_admission(
            &test_register_mma_f8f6f4_admission(accumulator),
            accumulator,
        )
        .unwrap()
        .remove(0);
        let (_, requirement) = candidate_llvm_route(&policy).unwrap();

        validate_candidate_target(&policy, &requirement, "sm_120a", "+ptx87").unwrap();
        validate_candidate_target(&policy, &requirement, "sm_120f", "+ptx88").unwrap();
        validate_candidate_target(&policy, &requirement, "sm_121a", "+ptx88").unwrap();
        validate_candidate_target(&policy, &requirement, "sm_121f", "+ptx88").unwrap();
        for target in ["sm_120f", "sm_121a", "sm_121f"] {
            assert!(
                validate_candidate_target(&policy, &requirement, target, "+ptx87").is_err(),
                "{target} must require PTX 8.8"
            );
        }
    }
}

#[test]
fn pinned_register_mma_records_match_the_closed_recipes_and_fail_closed() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let (mut overlay, _) =
        read_overlay(&repo_root, &repo_root.join("intrinsics/overlay.toml")).unwrap();
    bind_pinned_abi_ids(&repo_root, &mut overlay);
    let imported: ImportedFile = read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
    let declarations: BTreeMap<_, _> = imported
        .intrinsics
        .iter()
        .map(|record| (record.source_record.as_str(), record))
        .collect();
    let records: Vec<_> = overlay
        .intrinsics
        .iter()
        .filter(|record| record.family == "register_mma")
        .collect();
    assert_eq!(records.len(), 154);

    let dense_f8f6f4_records = records
        .iter()
        .copied()
        .filter(|record| {
            record.register_mma.as_ref().is_some_and(|mma| {
                matches!(mma.kind, None | Some(RegisterMmaKind::F8f6f4))
                    && register_mma_f8f6f4_element_name(mma.a_element).is_some()
                    && register_mma_f8f6f4_element_name(mma.b_element).is_some()
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(dense_f8f6f4_records.len(), 50);
    for (accumulator, first_abi, last_abi, adapter) in [
        (
            RegisterMmaAccumulator::F32,
            "i0454",
            "i0478",
            RegisterMmaAdapter::C4F32A4U32B2U32ToD4F32,
        ),
        (
            RegisterMmaAccumulator::F16,
            "i0479",
            "i0503",
            RegisterMmaAdapter::C2U32A4U32B2U32ToD2U32,
        ),
    ] {
        let family = dense_f8f6f4_records
            .iter()
            .copied()
            .filter(|record| record.register_mma.as_ref().unwrap().accumulator == accumulator)
            .collect::<Vec<_>>();
        assert_eq!(family.len(), 25);
        assert_eq!(family[0].abi_id, first_abi);
        assert_eq!(family[24].abi_id, last_abi);
        assert!(
            family
                .iter()
                .all(|record| record.register_mma.as_ref().unwrap().adapter == adapter)
        );
    }
    assert!(dense_f8f6f4_records.iter().all(|record| {
        record.minimum_ptx == "8.7"
            && record.minimum_sm.is_none()
            && record.targets == REGISTER_MMA_F8F6F4_TARGETS
    }));

    let mxf8f6f4_records = records
        .iter()
        .copied()
        .filter(|record| {
            record
                .register_mma
                .as_ref()
                .is_some_and(|mma| mma.kind == Some(RegisterMmaKind::Mxf8f6f4))
        })
        .collect::<Vec<_>>();
    assert_eq!(mxf8f6f4_records.len(), 25);
    assert_eq!(mxf8f6f4_records[0].abi_id, "i0858");
    assert_eq!(mxf8f6f4_records[24].abi_id, "i0882");
    assert!(mxf8f6f4_records.iter().all(|record| {
        let mma = record.register_mma.as_ref().unwrap();
        mma.shape == RegisterMmaShape::M16n8k32
            && mma.accumulator == RegisterMmaAccumulator::F32
            && mma.adapter == RegisterMmaAdapter::C4F32A4U32B2U32Scales2U32Selectors4U16ToD4F32
            && record.rust_arguments
                == [
                    "[f32; 4]", "[u32; 4]", "[u32; 2]", "u32", "u16", "u16", "u32", "u16", "u16",
                ]
            && record.rust_result == "[f32; 4]"
            && record.minimum_ptx == "8.7"
            && record.minimum_sm.is_none()
            && record.targets == REGISTER_MMA_F8F6F4_TARGETS
            && record
                .expected_ptx
                .modifiers
                .iter()
                .any(|modifier| modifier == "kind::mxf8f6f4")
    }));

    let integer_records: Vec<_> = records
        .iter()
        .copied()
        .filter(|record| {
            record.register_mma.as_ref().is_some_and(|mma| {
                mma.operation == RegisterMmaOperation::Multiply
                    && mma.accumulator == RegisterMmaAccumulator::S32
            })
        })
        .collect();
    assert_eq!(integer_records.len(), 48);
    let binary_records = records
        .iter()
        .copied()
        .filter(|record| {
            record
                .register_mma
                .as_ref()
                .is_some_and(|mma| mma.operation != RegisterMmaOperation::Multiply)
        })
        .collect::<Vec<_>>();
    assert_eq!(binary_records.len(), 6);
    let int8_records = integer_records
        .iter()
        .copied()
        .filter(|record| {
            let mma = record.register_mma.as_ref().unwrap();
            matches!(
                mma.a_element,
                RegisterMmaElement::S8 | RegisterMmaElement::U8
            ) && matches!(
                mma.b_element,
                RegisterMmaElement::S8 | RegisterMmaElement::U8
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(int8_records.len(), 24);
    let int4_records = integer_records
        .iter()
        .copied()
        .filter(|record| {
            let mma = record.register_mma.as_ref().unwrap();
            matches!(
                mma.a_element,
                RegisterMmaElement::S4 | RegisterMmaElement::U4
            ) && matches!(
                mma.b_element,
                RegisterMmaElement::S4 | RegisterMmaElement::U4
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(int4_records.len(), 24);
    let actual_variants = integer_records
        .iter()
        .map(|record| {
            let mma = record.register_mma.as_ref().unwrap();
            (mma.shape, mma.a_element, mma.b_element, mma.overflow)
        })
        .collect::<BTreeSet<_>>();
    let expected_int8_variants = [
        RegisterMmaShape::M8n8k16,
        RegisterMmaShape::M16n8k16,
        RegisterMmaShape::M16n8k32,
    ]
    .into_iter()
    .flat_map(|shape| {
        [RegisterMmaElement::S8, RegisterMmaElement::U8]
            .into_iter()
            .flat_map(move |a_element| {
                [RegisterMmaElement::S8, RegisterMmaElement::U8]
                    .into_iter()
                    .flat_map(move |b_element| {
                        [
                            RegisterMmaOverflow::Wrapping,
                            RegisterMmaOverflow::Satfinite,
                        ]
                        .into_iter()
                        .map(move |overflow| (shape, a_element, b_element, overflow))
                    })
            })
    })
    .collect::<BTreeSet<_>>();
    let expected_int4_variants = [
        RegisterMmaShape::M8n8k32,
        RegisterMmaShape::M16n8k32,
        RegisterMmaShape::M16n8k64,
    ]
    .into_iter()
    .flat_map(|shape| {
        [RegisterMmaElement::S4, RegisterMmaElement::U4]
            .into_iter()
            .flat_map(move |a_element| {
                [RegisterMmaElement::S4, RegisterMmaElement::U4]
                    .into_iter()
                    .flat_map(move |b_element| {
                        [
                            RegisterMmaOverflow::Wrapping,
                            RegisterMmaOverflow::Satfinite,
                        ]
                        .into_iter()
                        .map(move |overflow| (shape, a_element, b_element, overflow))
                    })
            })
    })
    .collect::<BTreeSet<_>>();
    let expected_variants = expected_int8_variants
        .union(&expected_int4_variants)
        .copied()
        .collect::<BTreeSet<_>>();
    assert_eq!(actual_variants, expected_variants);
    assert_eq!(
        integer_records
            .iter()
            .filter(|record| {
                record.register_mma.as_ref().unwrap().compatibility_source
                    == RegisterMmaCompatibilitySource::GeneratedStub
            })
            .count(),
        47
    );

    let actual_binary_variants = binary_records
        .iter()
        .map(|record| {
            let mma = record.register_mma.as_ref().unwrap();
            (mma.shape, mma.operation)
        })
        .collect::<BTreeSet<_>>();
    let expected_binary_variants = [
        RegisterMmaShape::M8n8k128,
        RegisterMmaShape::M16n8k128,
        RegisterMmaShape::M16n8k256,
    ]
    .into_iter()
    .flat_map(|shape| {
        [RegisterMmaOperation::XorPopc, RegisterMmaOperation::AndPopc]
            .into_iter()
            .map(move |operation| (shape, operation))
    })
    .collect::<BTreeSet<_>>();
    assert_eq!(actual_binary_variants, expected_binary_variants);
    assert!(binary_records.iter().all(|record| {
        let mma = record.register_mma.as_ref().unwrap();
        mma.accumulator == RegisterMmaAccumulator::S32
            && mma.a_element == RegisterMmaElement::B1
            && mma.b_element == RegisterMmaElement::B1
            && mma.overflow == RegisterMmaOverflow::Wrapping
            && mma.compatibility_source == RegisterMmaCompatibilitySource::GeneratedStub
            && record.expected_ptx.modifiers.ends_with(&[
                match mma.operation {
                    RegisterMmaOperation::XorPopc => "xor".into(),
                    RegisterMmaOperation::AndPopc => "and".into(),
                    RegisterMmaOperation::Multiply => unreachable!(),
                },
                "popc".into(),
            ])
    }));

    for record in &binary_records {
        let mma = record.register_mma.as_ref().unwrap();
        let (arguments, result, adapter) = match mma.shape {
            RegisterMmaShape::M8n8k128 => (
                &["[i32; 2]", "u32", "u32"] as &[_],
                "[i32; 2]",
                RegisterMmaAdapter::C2I32A1U32B1U32ToD2I32,
            ),
            RegisterMmaShape::M16n8k128 => (
                &["[i32; 4]", "[u32; 2]", "u32"] as &[_],
                "[i32; 4]",
                RegisterMmaAdapter::C4I32A2U32B1U32ToD4I32,
            ),
            RegisterMmaShape::M16n8k256 => (
                &["[i32; 4]", "[u32; 4]", "[u32; 2]"] as &[_],
                "[i32; 4]",
                RegisterMmaAdapter::C4I32A4U32B2U32ToD4I32,
            ),
            _ => unreachable!(),
        };
        assert_eq!(record.rust_arguments, arguments);
        assert_eq!(record.rust_result, result);
        assert_eq!(mma.adapter, adapter);
        let expected_floor = match (mma.shape, mma.operation) {
            (RegisterMmaShape::M8n8k128, RegisterMmaOperation::XorPopc) => ("7.0", "sm_75"),
            (_, RegisterMmaOperation::XorPopc) => ("7.0", "sm_80"),
            (_, RegisterMmaOperation::AndPopc) => ("7.1", "sm_80"),
            _ => unreachable!(),
        };
        assert_eq!(record.minimum_ptx, expected_floor.0);
        assert_eq!(record.minimum_sm.as_deref(), Some(expected_floor.1));
    }

    for record in integer_records.iter().filter(|record| {
        matches!(
            record.register_mma.as_ref().unwrap().shape,
            RegisterMmaShape::M8n8k16 | RegisterMmaShape::M8n8k32
        )
    }) {
        assert_eq!(record.rust_arguments, ["[i32; 2]", "u32", "u32"]);
        assert_eq!(record.rust_result, "[i32; 2]");
        assert_eq!(record.minimum_ptx, "6.5");
        assert_eq!(record.minimum_sm.as_deref(), Some("sm_75"));
        assert_eq!(
            record.register_mma.as_ref().unwrap().adapter,
            RegisterMmaAdapter::C2I32A1U32B1U32ToD2I32
        );
    }

    for record in int4_records
        .iter()
        .filter(|record| record.register_mma.as_ref().unwrap().shape == RegisterMmaShape::M16n8k32)
    {
        assert_eq!(record.rust_arguments, ["[i32; 4]", "[u32; 2]", "u32"]);
        assert_eq!(record.rust_result, "[i32; 4]");
        assert_eq!(record.minimum_ptx, "7.0");
        assert_eq!(record.minimum_sm.as_deref(), Some("sm_80"));
        assert_eq!(
            record.register_mma.as_ref().unwrap().adapter,
            RegisterMmaAdapter::C4I32A2U32B1U32ToD4I32
        );
    }

    for record in int4_records
        .iter()
        .filter(|record| record.register_mma.as_ref().unwrap().shape == RegisterMmaShape::M16n8k64)
    {
        assert_eq!(record.rust_arguments, ["[i32; 4]", "[u32; 4]", "[u32; 2]"]);
        assert_eq!(record.rust_result, "[i32; 4]");
        assert_eq!(record.minimum_ptx, "7.0");
        assert_eq!(record.minimum_sm.as_deref(), Some("sm_80"));
        assert_eq!(
            record.register_mma.as_ref().unwrap().adapter,
            RegisterMmaAdapter::C4I32A4U32B2U32ToD4I32
        );
    }

    let actual_int4_abi_ids = int4_records
        .iter()
        .map(|record| record.abi_id.as_str())
        .collect::<BTreeSet<_>>();
    let expected_int4_abi_ids = (133..=156)
        .map(|id| format!("i{id:04}"))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        actual_int4_abi_ids,
        expected_int4_abi_ids
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
    );

    let int8_k32 = int8_records
        .iter()
        .find(|record| record.register_mma.as_ref().unwrap().shape == RegisterMmaShape::M16n8k32)
        .unwrap();
    let int4_k32 = int4_records
        .iter()
        .find(|record| record.register_mma.as_ref().unwrap().shape == RegisterMmaShape::M16n8k32)
        .unwrap();
    assert_eq!(
        int8_k32.register_mma.as_ref().unwrap().adapter,
        RegisterMmaAdapter::C4I32A4U32B2U32ToD4I32
    );
    assert_eq!(
        int4_k32.register_mma.as_ref().unwrap().adapter,
        RegisterMmaAdapter::C4I32A2U32B1U32ToD4I32
    );

    for policy in &records {
        let declaration = declarations[policy.source_record.as_deref().unwrap()];
        assert_eq!(declaration.selections.len(), 1);
        assert!(
            selection_matches_policy(policy, &declaration.selections[0]).unwrap(),
            "{}",
            policy.id
        );
        validate_imported_policy(policy, declaration).unwrap();
    }

    let valid = records[0];
    let declaration = declarations[valid.source_record.as_deref().unwrap()];

    let mut non_convergent = valid.clone();
    non_convergent.convergent = false;
    assert!(
        validate_imported_policy(&non_convergent, declaration)
            .unwrap_err()
            .to_string()
            .contains("effects")
    );

    let mut typed_route = valid.clone();
    typed_route.backend_lowerings[0].mechanism = BackendLoweringMechanism::TypedNvvm;
    assert!(validate_imported_policy(&typed_route, declaration).is_err());

    let mut selectionless = declaration.clone();
    selectionless.selections.clear();
    assert!(validate_imported_policy(valid, &selectionless).is_err());

    let mut crossed_variant = valid.clone();
    crossed_variant.register_mma.as_mut().unwrap().a_element = RegisterMmaElement::F16;
    assert!(validate_imported_policy(&crossed_variant, declaration).is_err());

    let generated = int8_records
        .iter()
        .copied()
        .find(|record| record.id == "mma_m16n8k16_s32_s8_u8_satfinite")
        .unwrap();
    let generated_declaration = declarations[generated.source_record.as_deref().unwrap()];

    let mut wrong_stub_owner = generated.clone();
    wrong_stub_owner
        .register_mma
        .as_mut()
        .unwrap()
        .compatibility_source = RegisterMmaCompatibilitySource::ExistingStub;
    assert!(validate_imported_policy(&wrong_stub_owner, generated_declaration).is_err());

    let mut wrong_b_element = generated.clone();
    wrong_b_element.register_mma.as_mut().unwrap().b_element = RegisterMmaElement::S8;
    assert!(validate_imported_policy(&wrong_b_element, generated_declaration).is_err());

    let mut wrong_overflow = generated.clone();
    wrong_overflow.register_mma.as_mut().unwrap().overflow = RegisterMmaOverflow::Wrapping;
    assert!(validate_imported_policy(&wrong_overflow, generated_declaration).is_err());

    let mut wrong_shape = generated.clone();
    wrong_shape.register_mma.as_mut().unwrap().shape = RegisterMmaShape::M16n8k32;
    assert!(validate_imported_policy(&wrong_shape, generated_declaration).is_err());

    let mut wrong_adapter = generated.clone();
    wrong_adapter.register_mma.as_mut().unwrap().adapter =
        RegisterMmaAdapter::C4I32A4U32B2U32ToD4I32;
    assert!(validate_imported_policy(&wrong_adapter, generated_declaration).is_err());

    let binary = binary_records
        .iter()
        .copied()
        .find(|record| record.id == "mma_m8n8k128_s32_b1_xor_popc")
        .unwrap();
    let binary_declaration = declarations[binary.source_record.as_deref().unwrap()];

    let mut wrong_binary_operation = binary.clone();
    wrong_binary_operation
        .register_mma
        .as_mut()
        .unwrap()
        .operation = RegisterMmaOperation::AndPopc;
    assert!(validate_imported_policy(&wrong_binary_operation, binary_declaration).is_err());

    let mut wrong_binary_floor = binary.clone();
    wrong_binary_floor.minimum_sm = Some("sm_80".into());
    assert!(validate_imported_policy(&wrong_binary_floor, binary_declaration).is_err());

    let mut wrong_binary_element = binary.clone();
    wrong_binary_element
        .register_mma
        .as_mut()
        .unwrap()
        .a_element = RegisterMmaElement::U4;
    assert!(validate_imported_policy(&wrong_binary_element, binary_declaration).is_err());
}

#[test]
fn pinned_sparse_mma_records_close_shape_specific_selectors_and_ranges() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let (mut overlay, _) =
        read_overlay(&repo_root, &repo_root.join("intrinsics/overlay.toml")).unwrap();
    bind_pinned_abi_ids(&repo_root, &mut overlay);
    let imported: ImportedFile = read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
    let declarations: BTreeMap<_, _> = imported
        .intrinsics
        .iter()
        .map(|record| (record.source_record.as_str(), record))
        .collect();
    let records = overlay
        .intrinsics
        .iter()
        .filter(|record| record.family == "sparse_mma")
        .collect::<Vec<_>>();
    assert_eq!(records.len(), 126);
    assert_eq!(
        records
            .iter()
            .map(|record| record.abi_id.clone())
            .collect::<BTreeSet<_>>(),
        (163..=251)
            .chain(525..=549)
            .chain(1018..=1029)
            .map(|id| format!("i{id:04}"))
            .collect::<BTreeSet<_>>()
    );

    let mut derived_ids = BTreeSet::new();
    let mut derived_operation_keys = BTreeSet::new();
    let mut derived_source_records = BTreeSet::new();
    let mut derived_llvm_symbols = BTreeSet::new();
    for record in &records {
        let identity = &sparse_mma_recipe(record.sparse_mma.as_ref().unwrap())
            .unwrap()
            .identity;
        assert_eq!(record.id, identity.id);
        assert_eq!(record.operation_key, identity.operation_key);
        assert_eq!(
            record.source_record.as_deref(),
            Some(identity.source_record.as_str())
        );
        assert_eq!(
            record.llvm_symbol.as_deref(),
            Some(identity.llvm_symbol.as_str())
        );
        assert_eq!(record.expected_ptx.modifiers, identity.ptx_modifiers);
        assert!(derived_ids.insert(identity.id.clone()));
        assert!(derived_operation_keys.insert(identity.operation_key.clone()));
        assert!(derived_source_records.insert(identity.source_record.clone()));
        assert!(derived_llvm_symbols.insert(identity.llvm_symbol.clone()));
    }
    assert_eq!(derived_ids.len(), 126);
    assert_eq!(derived_operation_keys.len(), 126);
    assert_eq!(derived_source_records.len(), 126);
    assert_eq!(derived_llvm_symbols.len(), 126);

    let integer_records = records
        .iter()
        .copied()
        .filter(|record| {
            record.sparse_mma.as_ref().unwrap().accumulator == SparseMmaAccumulator::S32
        })
        .collect::<Vec<_>>();
    assert_eq!(integer_records.len(), 64);
    let f32_records = records
        .iter()
        .copied()
        .filter(|record| {
            let mma = record.sparse_mma.as_ref().unwrap();
            mma.accumulator == SparseMmaAccumulator::F32
                && mma.metadata == SparseMmaMetadata::Ordered
                && matches!(
                    mma.a_element,
                    SparseMmaElement::E2m1
                        | SparseMmaElement::E2m3
                        | SparseMmaElement::E3m2
                        | SparseMmaElement::E4m3
                        | SparseMmaElement::E5m2
                )
        })
        .collect::<Vec<_>>();
    assert_eq!(f32_records.len(), 25);
    let f16_records = records
        .iter()
        .copied()
        .filter(|record| {
            let mma = record.sparse_mma.as_ref().unwrap();
            mma.accumulator == SparseMmaAccumulator::F16
                && matches!(
                    mma.a_element,
                    SparseMmaElement::E2m1
                        | SparseMmaElement::E2m3
                        | SparseMmaElement::E3m2
                        | SparseMmaElement::E4m3
                        | SparseMmaElement::E5m2
                )
        })
        .collect::<Vec<_>>();
    assert_eq!(f16_records.len(), 25);

    let variants = integer_records
        .iter()
        .map(|record| {
            let mma = record.sparse_mma.as_ref().unwrap();
            let carrier =
                sparse_mma_carrier_recipe(mma.shape, mma.a_element, mma.b_element).unwrap();
            assert_eq!(mma.accumulator, SparseMmaAccumulator::S32);
            assert_eq!(mma.selector, carrier.selector);
            assert_eq!(mma.adapter, carrier.adapter);
            assert_eq!(mma.llvm_adapter, carrier.llvm_adapter);
            assert_eq!(record.rust_arguments, carrier.rust_arguments());
            assert_eq!(record.dialect_operands, carrier.dialect_operands());
            assert_eq!(record.llvm_arguments, carrier.llvm_arguments());
            assert_eq!(
                record.expected_ptx.operands,
                carrier.expected_ptx_operands()
            );
            assert_eq!(record.minimum_ptx, sparse_mma_minimum_ptx(mma));
            assert_eq!(record.minimum_sm.as_deref(), Some("sm_80"));
            assert_eq!(
                record.expected_ptx.operands.last(),
                Some(&OperandPattern::Immediate)
            );
            assert_eq!(
                record.expected_ptx.modifiers.first().map(String::as_str),
                Some(match mma.metadata {
                    SparseMmaMetadata::Standard => "sp",
                    SparseMmaMetadata::Ordered => "sp::ordered_metadata",
                })
            );
            (
                mma.shape,
                mma.a_element,
                mma.b_element,
                mma.overflow,
                mma.metadata,
            )
        })
        .collect::<BTreeSet<_>>();
    let mut expected_variants = BTreeSet::new();
    for shape in [SparseMmaShape::M16n8k32, SparseMmaShape::M16n8k64] {
        let metadata = match shape {
            SparseMmaShape::M16n8k8 | SparseMmaShape::M16n8k16 => {
                unreachable!("integer sparse-MMA test does not enumerate float shapes")
            }
            SparseMmaShape::M16n8k32 => [
                Some(SparseMmaMetadata::Standard),
                Some(SparseMmaMetadata::Ordered),
            ],
            SparseMmaShape::M16n8k64 => [
                Some(SparseMmaMetadata::Standard),
                Some(SparseMmaMetadata::Ordered),
            ],
            SparseMmaShape::M16n8k128 => [None, None],
        };
        for a_element in [SparseMmaElement::S8, SparseMmaElement::U8] {
            for b_element in [SparseMmaElement::S8, SparseMmaElement::U8] {
                for overflow in [SparseMmaOverflow::Wrapping, SparseMmaOverflow::Satfinite] {
                    for metadata in metadata.into_iter().flatten() {
                        expected_variants.insert((shape, a_element, b_element, overflow, metadata));
                    }
                }
            }
        }
    }
    for shape in [SparseMmaShape::M16n8k64, SparseMmaShape::M16n8k128] {
        for a_element in [SparseMmaElement::S4, SparseMmaElement::U4] {
            for b_element in [SparseMmaElement::S4, SparseMmaElement::U4] {
                for overflow in [SparseMmaOverflow::Wrapping, SparseMmaOverflow::Satfinite] {
                    for metadata in [SparseMmaMetadata::Standard, SparseMmaMetadata::Ordered] {
                        expected_variants.insert((shape, a_element, b_element, overflow, metadata));
                    }
                }
            }
        }
    }
    assert_eq!(variants, expected_variants);

    let f8f6f4_formats = [
        SparseMmaElement::E2m1,
        SparseMmaElement::E2m3,
        SparseMmaElement::E3m2,
        SparseMmaElement::E4m3,
        SparseMmaElement::E5m2,
    ];
    assert_eq!(
        f32_records
            .iter()
            .map(|record| {
                let mma = record.sparse_mma.as_ref().unwrap();
                (mma.a_element, mma.b_element)
            })
            .collect::<BTreeSet<_>>(),
        f8f6f4_formats
            .into_iter()
            .flat_map(|a| f8f6f4_formats.into_iter().map(move |b| (a, b)))
            .collect()
    );
    assert_eq!(
        f32_records
            .iter()
            .map(|record| record.abi_id.as_str())
            .collect::<BTreeSet<_>>(),
        (227..=251)
            .map(|id| format!("i{id:04}"))
            .collect::<BTreeSet<_>>()
            .iter()
            .map(String::as_str)
            .collect()
    );
    assert_eq!(
        f16_records
            .iter()
            .map(|record| {
                let mma = record.sparse_mma.as_ref().unwrap();
                (mma.a_element, mma.b_element)
            })
            .collect::<BTreeSet<_>>(),
        f8f6f4_formats
            .into_iter()
            .flat_map(|a| f8f6f4_formats.into_iter().map(move |b| (a, b)))
            .collect()
    );
    assert_eq!(
        f16_records
            .iter()
            .map(|record| record.abi_id.as_str())
            .collect::<BTreeSet<_>>(),
        (525..=549)
            .map(|id| format!("i{id:04}"))
            .collect::<BTreeSet<_>>()
            .iter()
            .map(String::as_str)
            .collect()
    );
    for record in &f32_records {
        let mma = record.sparse_mma.as_ref().unwrap();
        assert_eq!(mma.shape, SparseMmaShape::M16n8k64);
        assert_eq!(mma.accumulator, SparseMmaAccumulator::F32);
        assert_eq!(mma.overflow, SparseMmaOverflow::NotApplicable);
        assert_eq!(mma.metadata, SparseMmaMetadata::Ordered);
        assert_eq!(mma.selector, SparseMmaSelector::ImmediateZero);
        assert_eq!(
            mma.adapter,
            SparseMmaAdapter::C4F32A4U32B4U32MetadataU32SelectorU32ToD4F32
        );
        assert_eq!(
            mma.llvm_adapter,
            SparseMmaLlvmAdapter::A4I32B4I32C4F32MetadataI32SelectorI32ToD4F32
        );
        assert_eq!(
            record.rust_arguments,
            ["[f32; 4]", "[u32; 4]", "[u32; 4]", "u32", "u32"]
        );
        assert_eq!(record.rust_result, "[f32; 4]");
        assert_eq!(
            record.dialect_operands,
            [
                "f32", "f32", "f32", "f32", "u32", "u32", "u32", "u32", "u32", "u32", "u32", "u32",
                "u32", "u32"
            ]
        );
        assert_eq!(record.dialect_results, ["f32", "f32", "f32", "f32"]);
        assert_eq!(
            record.llvm_arguments,
            [
                "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "f32", "f32", "f32", "f32",
                "i32", "i32"
            ]
        );
        assert_eq!(record.llvm_results, ["f32", "f32", "f32", "f32"]);
        assert_eq!(record.minimum_ptx, "8.7");
        assert_eq!(record.minimum_sm, None);
        // Same contract as the F16 accumulator: both float forms are gated
        // on `hasMMABlockScale()`, so neither is narrower than the other.
        assert_eq!(record.targets, SPARSE_MMA_F8F6F4_TARGETS);
        assert_eq!(record.backend_lowerings.len(), 2);
        assert!(record.backend_lowerings.iter().all(|lowering| {
            lowering.mechanism == BackendLoweringMechanism::InlinePtx
                && lowering.minimum_ptx.as_deref() == Some("8.7")
                && lowering.minimum_sm.is_none()
        }));
        assert_eq!(
            record.expected_ptx.operands,
            [
                OperandPattern::RegisterList { length: 4 },
                OperandPattern::RegisterList { length: 4 },
                OperandPattern::RegisterList { length: 4 },
                OperandPattern::RegisterList { length: 4 },
                OperandPattern::Register,
                OperandPattern::Immediate,
            ]
        );
    }

    for record in &f16_records {
        let mma = record.sparse_mma.as_ref().unwrap();
        assert_eq!(mma.shape, SparseMmaShape::M16n8k64);
        assert_eq!(mma.accumulator, SparseMmaAccumulator::F16);
        assert_eq!(mma.overflow, SparseMmaOverflow::NotApplicable);
        assert_eq!(mma.metadata, SparseMmaMetadata::Ordered);
        assert_eq!(mma.selector, SparseMmaSelector::ImmediateZero);
        assert_eq!(
            mma.adapter,
            SparseMmaAdapter::C2U32A4U32B4U32MetadataU32SelectorU32ToD2U32
        );
        assert_eq!(
            mma.llvm_adapter,
            SparseMmaLlvmAdapter::A4I32B4I32C2V2F16MetadataI32SelectorI32ToD2V2F16
        );
        assert_eq!(
            record.rust_arguments,
            ["[u32; 2]", "[u32; 4]", "[u32; 4]", "u32", "u32"]
        );
        assert_eq!(record.rust_result, "[u32; 2]");
        assert_eq!(
            record.dialect_operands,
            [
                "i32", "i32", "u32", "u32", "u32", "u32", "u32", "u32", "u32", "u32", "u32", "u32"
            ]
        );
        assert_eq!(record.dialect_results, ["i32", "i32"]);
        assert_eq!(
            record.llvm_arguments,
            [
                "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "v2f16", "v2f16", "i32",
                "i32"
            ]
        );
        assert_eq!(record.llvm_results, ["v2f16", "v2f16"]);
        assert_eq!(record.minimum_ptx, "8.7");
        assert_eq!(record.minimum_sm, None);
        assert_eq!(record.targets, SPARSE_MMA_F8F6F4_TARGETS);
        assert!(record.convergent && !record.pure);
        assert!(record.backend_lowerings.iter().all(|lowering| {
            lowering.mechanism == BackendLoweringMechanism::InlinePtx
                && lowering.minimum_ptx.as_deref() == Some("8.7")
                && lowering.minimum_sm.is_none()
        }));
        assert_eq!(
            record.expected_ptx.operands,
            [
                OperandPattern::RegisterList { length: 2 },
                OperandPattern::RegisterList { length: 4 },
                OperandPattern::RegisterList { length: 4 },
                OperandPattern::RegisterList { length: 2 },
                OperandPattern::Register,
                OperandPattern::Immediate,
            ]
        );
    }

    for policy in &records {
        let declaration = declarations[policy.source_record.as_deref().unwrap()];
        assert_eq!(declaration.selections.len(), 1);
        assert!(
            selection_matches_policy(policy, &declaration.selections[0]).unwrap(),
            "{}",
            policy.id
        );
        validate_imported_policy(policy, declaration).unwrap();
    }

    let mut selectionless = declarations[records[0].source_record.as_deref().unwrap()].clone();
    selectionless.selections.clear();
    assert!(validate_imported_policy(records[0], &selectionless).is_err());

    for (id, range_prefix, wrong_range) in [
        ("mma_sp_m16n8k32_s32_s8", "Range<arg9", "Range<arg9,0,3>"),
        ("mma_sp_m16n8k64_s32_s8", "Range<arg13", "Range<arg13,0,2>"),
        (
            "mma_sp_ordered_metadata_m16n8k64_s32_s4",
            "Range<arg9",
            "Range<arg9,0,1>",
        ),
        ("mma_sp_m16n8k64_s32_s4", "Range<arg9", "Range<arg9,0,1>"),
        (
            "mma_sp_ordered_metadata_m16n8k128_s32_s4",
            "Range<arg13",
            "Range<arg13,0,2>",
        ),
        ("mma_sp_m16n8k128_s32_s4", "Range<arg13", "Range<arg13,0,2>"),
        (
            "mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e2m1_e2m1_f32",
            "Range<arg13",
            "Range<arg13,0,2>",
        ),
        (
            "mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e2m1_e2m1_f16",
            "Range<arg11",
            "Range<arg11,0,2>",
        ),
    ] {
        let valid = records
            .iter()
            .copied()
            .find(|record| record.id == id)
            .unwrap();
        let declaration = declarations[valid.source_record.as_deref().unwrap()];

        let mut runtime_selector = valid.clone();
        *runtime_selector.expected_ptx.operands.last_mut().unwrap() = OperandPattern::Register;
        assert!(
            validate_imported_policy(&runtime_selector, declaration)
                .unwrap_err()
                .to_string()
                .contains("exact selection")
        );

        let mut wrong_declaration = declaration.clone();
        *wrong_declaration
            .properties
            .iter_mut()
            .find(|property| property.starts_with(range_prefix))
            .unwrap() = wrong_range.into();
        assert!(
            validate_imported_policy(valid, &wrong_declaration)
                .unwrap_err()
                .to_string()
                .contains("immediate range")
        );
    }

    let k64 = records
        .iter()
        .copied()
        .find(|record| record.id == "mma_sp_m16n8k64_s32_s8")
        .unwrap();
    let k64_declaration = declarations[k64.source_record.as_deref().unwrap()];
    assert_eq!(k64.minimum_ptx, "7.1");
    let ordered_k64 = records
        .iter()
        .copied()
        .find(|record| record.id == "mma_sp_ordered_metadata_m16n8k64_s32_s8")
        .unwrap();
    assert_eq!(ordered_k64.minimum_ptx, "8.5");

    let f8f6f4 = f32_records[0];
    let f8f6f4_declaration = declarations[f8f6f4.source_record.as_deref().unwrap()];
    let mut widened_family = f8f6f4.clone();
    widened_family.targets = "sm_120f".into();
    assert!(validate_imported_policy(&widened_family, f8f6f4_declaration).is_err());
    let mut widened_architecture = f8f6f4.clone();
    widened_architecture.targets = "sm_121a".into();
    assert!(validate_imported_policy(&widened_architecture, f8f6f4_declaration).is_err());

    let f8f6f4_f16 = f16_records[0];
    for block_scale in [f8f6f4, f8f6f4_f16] {
        let mut missing_predicate =
            declarations[block_scale.source_record.as_deref().unwrap()].clone();
        missing_predicate.selections[0].predicates.clear();
        assert!(
            validate_imported_policy(block_scale, &missing_predicate)
                .unwrap_err()
                .to_string()
                .contains("exact selection")
        );
    }

    let ordered_k64_int4 = records
        .iter()
        .copied()
        .find(|record| record.id == "mma_sp_ordered_metadata_m16n8k64_s32_s4")
        .unwrap();
    let ordered_k64_int4_declaration =
        declarations[ordered_k64_int4.source_record.as_deref().unwrap()];
    assert_eq!(ordered_k64_int4.minimum_ptx, "8.5");
    assert_eq!(ordered_k64_int4.rust_arguments[1], "[u32; 2]");
    assert_eq!(ordered_k64_int4.rust_arguments[2], "[u32; 2]");
    assert_eq!(ordered_k64_int4.llvm_arguments.len(), 10);
    assert_eq!(
        ordered_k64_int4.sparse_mma.as_ref().unwrap().selector,
        SparseMmaSelector::ImmediateZeroOrOne
    );
    assert_eq!(
        ordered_k64_int4.sparse_mma.as_ref().unwrap().adapter,
        SparseMmaAdapter::C4I32A2U32B2U32MetadataU32SelectorU32ToD4I32
    );
    assert_eq!(
        ordered_k64_int4.sparse_mma.as_ref().unwrap().llvm_adapter,
        SparseMmaLlvmAdapter::A2I32B2I32C4I32MetadataI32SelectorI32ToD4I32
    );
    let standard_k64_int4 = records
        .iter()
        .copied()
        .find(|record| record.id == "mma_sp_m16n8k64_s32_s4")
        .unwrap();
    assert_eq!(standard_k64_int4.minimum_ptx, "7.1");
    assert_eq!(
        standard_k64_int4.rust_arguments,
        ordered_k64_int4.rust_arguments
    );
    assert_eq!(
        standard_k64_int4.llvm_arguments,
        ordered_k64_int4.llvm_arguments
    );

    let ordered_k128_int4 = records
        .iter()
        .copied()
        .find(|record| record.id == "mma_sp_ordered_metadata_m16n8k128_s32_s4")
        .unwrap();
    let ordered_k128_int4_declaration =
        declarations[ordered_k128_int4.source_record.as_deref().unwrap()];
    assert_eq!(ordered_k128_int4.minimum_ptx, "8.5");
    assert_eq!(ordered_k128_int4.rust_arguments[1], "[u32; 4]");
    assert_eq!(ordered_k128_int4.rust_arguments[2], "[u32; 4]");
    assert_eq!(ordered_k128_int4.llvm_arguments.len(), 14);
    assert_eq!(
        ordered_k128_int4.sparse_mma.as_ref().unwrap().selector,
        SparseMmaSelector::ImmediateZero
    );
    assert_eq!(
        ordered_k128_int4.sparse_mma.as_ref().unwrap().adapter,
        SparseMmaAdapter::C4I32A4U32B4U32MetadataU32SelectorU32ToD4I32
    );
    assert_eq!(
        ordered_k128_int4.sparse_mma.as_ref().unwrap().llvm_adapter,
        SparseMmaLlvmAdapter::A4I32B4I32C4I32MetadataI32SelectorI32ToD4I32
    );
    let standard_k128_int4 = records
        .iter()
        .copied()
        .find(|record| record.id == "mma_sp_m16n8k128_s32_s4")
        .unwrap();
    assert_eq!(standard_k128_int4.minimum_ptx, "7.1");
    assert_eq!(
        standard_k128_int4.rust_arguments,
        ordered_k128_int4.rust_arguments
    );
    assert_eq!(
        standard_k128_int4.llvm_arguments,
        ordered_k128_int4.llvm_arguments
    );

    let mut wrong_k128_selector = ordered_k128_int4.clone();
    wrong_k128_selector.sparse_mma.as_mut().unwrap().selector =
        SparseMmaSelector::ImmediateZeroOrOne;
    assert!(validate_imported_policy(&wrong_k128_selector, ordered_k128_int4_declaration).is_err());

    let mut mixed_k128_width = ordered_k128_int4.clone();
    mixed_k128_width.sparse_mma.as_mut().unwrap().b_element = SparseMmaElement::U8;
    assert!(
        validate_imported_policy(&mixed_k128_width, ordered_k128_int4_declaration)
            .unwrap_err()
            .to_string()
            .contains("unsupported sparse-MMA variant")
    );

    let mut mixed_width = ordered_k64_int4.clone();
    mixed_width.sparse_mma.as_mut().unwrap().b_element = SparseMmaElement::U8;
    assert!(
        validate_imported_policy(&mixed_width, ordered_k64_int4_declaration)
            .unwrap_err()
            .to_string()
            .contains("unsupported sparse-MMA variant")
    );

    let mut wrong_k64_selector = k64.clone();
    wrong_k64_selector.sparse_mma.as_mut().unwrap().selector =
        SparseMmaSelector::ImmediateZeroOrOne;
    assert!(validate_imported_policy(&wrong_k64_selector, k64_declaration).is_err());

    let mut wrong_k64_adapter = k64.clone();
    wrong_k64_adapter.sparse_mma.as_mut().unwrap().adapter =
        SparseMmaAdapter::C4I32A2U32B2U32MetadataU32SelectorU32ToD4I32;
    assert!(validate_imported_policy(&wrong_k64_adapter, k64_declaration).is_err());

    let mut wrong_k64_llvm_adapter = k64.clone();
    wrong_k64_llvm_adapter
        .sparse_mma
        .as_mut()
        .unwrap()
        .llvm_adapter = SparseMmaLlvmAdapter::A2I32B2I32C4I32MetadataI32SelectorI32ToD4I32;
    assert!(validate_imported_policy(&wrong_k64_llvm_adapter, k64_declaration).is_err());

    let mut wrong_k64_shape = k64.clone();
    wrong_k64_shape.sparse_mma.as_mut().unwrap().shape = SparseMmaShape::M16n8k32;
    assert!(validate_imported_policy(&wrong_k64_shape, k64_declaration).is_err());

    let mut wrong_k64_carriers = k64.clone();
    wrong_k64_carriers.dialect_operands.pop();
    assert!(validate_imported_policy(&wrong_k64_carriers, k64_declaration).is_err());

    let mut wrong_k64_lowering = k64.clone();
    wrong_k64_lowering.lowering = "generated_register_mma".into();
    assert!(validate_imported_policy(&wrong_k64_lowering, k64_declaration).is_err());

    let mut mismatched_metadata_identity = k64.clone();
    mismatched_metadata_identity
        .sparse_mma
        .as_mut()
        .unwrap()
        .metadata = SparseMmaMetadata::Ordered;
    assert!(validate_imported_policy(&mismatched_metadata_identity, k64_declaration).is_err());
}

#[test]
fn movmatrix_recipe_is_exact_and_fails_closed() {
    let valid = movmatrix_policy();
    validate_ptx_native_policy(&valid).unwrap();

    let reject = |policy: &OverlayIntrinsic, expected: &str| {
        let message = validate_ptx_native_policy(policy).unwrap_err().to_string();
        assert!(message.contains(expected), "unexpected error: {message}");
    };

    let mut wrong_shape = valid.clone();
    wrong_shape.expected_ptx.modifiers[2] = "m16n8".into();
    reject(&wrong_shape, "closed movmatrix recipe");

    let mut wrong_participation = valid.clone();
    wrong_participation.convergent = false;
    reject(&wrong_participation, "closed movmatrix recipe");

    // A warp collective is not a function of its own operand, so the
    // pure/no-memory pair every other collective avoids must stay rejected
    // here too.
    let mut wrong_purity = valid.clone();
    wrong_purity.pure = true;
    reject(&wrong_purity, "closed movmatrix recipe");

    let mut wrong_memory = valid.clone();
    wrong_memory.memory = "none".into();
    reject(&wrong_memory, "closed movmatrix recipe");

    let mut wrong_floor = valid.clone();
    wrong_floor.backend_lowerings[0].minimum_ptx = Some("8.0".into());
    reject(&wrong_floor, "exact movmatrix floor");

    let mut mixed = valid;
    mixed.warp_barrier = Some(crate::model::WarpBarrier {
        participation:
            WarpBarrierParticipation::ExecutingLaneNamedAllNamedLanesSameInstructionAndMask,
        legacy_pre_sm70: PreSm70MemberMaskRule::AllNamedLanesConvergedAndOnlyNamedLanesActive,
        adapter: WarpBarrierAdapter::DirectMemberMask,
        mask_encoding: WarpBarrierMaskEncoding::RegisterOrImmediate,
        memory_ordering: WarpBarrierMemoryOrdering::ParticipatingLanes,
    });
    reject(&mixed, "mixes another generated-family contract");
}

#[test]
fn plain_sm89_sparse_fp8_admission_is_the_reviewed_four_form_matrix() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let shard_bytes =
        fs::read(repo_root.join("intrinsics/overlay/sparse_mma_fp8_f32.toml")).unwrap();
    let mut shard: OverlayShardFile = toml::from_slice(&shard_bytes).unwrap();
    assert_eq!(shard.schema, SPARSE_MMA_FP8_F32_SHARD_SCHEMA);
    assert_eq!(shard.family, "sparse_mma");
    validate_overlay_shard_schema(
        &shard,
        Path::new("intrinsics/overlay/sparse_mma_fp8_f32.toml"),
    )
    .unwrap();
    let admission = shard.sparse_mma_fp8_f32.take().unwrap();

    let records = expand_sparse_mma_fp8_f32_admission(&admission).unwrap();
    assert_eq!(records.len(), 4);
    assert!(records.iter().all(|record| record.abi_id.is_empty()));

    // The matrix is emitted in A-major, B-minor order.
    assert_eq!(
        records
            .iter()
            .map(|record| record.id.as_str())
            .collect::<Vec<_>>(),
        [
            "mma_sp_m16n8k64_f32_e4m3_e4m3_f32",
            "mma_sp_m16n8k64_f32_e4m3_e5m2_f32",
            "mma_sp_m16n8k64_f32_e5m2_e4m3_f32",
            "mma_sp_m16n8k64_f32_e5m2_e5m2_f32",
        ]
    );

    for (record, (a_element, b_element)) in records.iter().zip([
        (SparseMmaElement::E4m3, SparseMmaElement::E4m3),
        (SparseMmaElement::E4m3, SparseMmaElement::E5m2),
        (SparseMmaElement::E5m2, SparseMmaElement::E4m3),
        (SparseMmaElement::E5m2, SparseMmaElement::E5m2),
    ]) {
        let mma = record.sparse_mma.as_ref().unwrap();
        assert!(is_sparse_mma_fp8_f32(mma));
        // Plain sparsity metadata, not the Blackwell `kind::f8f6f4` form.
        assert_eq!(mma.metadata, SparseMmaMetadata::Standard);
        assert_eq!(mma.shape, SparseMmaShape::M16n8k64);
        assert_eq!(mma.accumulator, SparseMmaAccumulator::F32);
        assert_eq!(mma.a_element, a_element);
        assert_eq!(mma.b_element, b_element);
        assert_eq!(mma.overflow, SparseMmaOverflow::NotApplicable);
        assert_eq!(mma.selector, SparseMmaSelector::ImmediateZero);
        assert_eq!(
            mma.adapter,
            SparseMmaAdapter::C4F32A4U32B4U32MetadataU32SelectorU32ToD4F32
        );
        assert_eq!(
            mma.llvm_adapter,
            SparseMmaLlvmAdapter::A4I32B4I32C4F32MetadataI32SelectorI32ToD4F32
        );
        assert_eq!(mma.runtime_validation, RuntimeValidation::Unexecuted);

        // PTX ISA 8.4 is the Ada floor; the qualifier must stay plain `sp`.
        assert_eq!(sparse_mma_minimum_ptx(mma), "8.4");
        assert_eq!(sparse_mma_hardware(mma), ("all", Some("sm_89")));
        assert_eq!(record.minimum_ptx, "8.4");
        assert_eq!(record.minimum_sm.as_deref(), Some("sm_89"));
        assert_eq!(record.targets, "all");
        assert_eq!(record.family, "sparse_mma");
        assert_eq!(record.lowering, "generated_sparse_mma");
        assert_eq!(
            record.rust_arguments,
            ["[f32; 4]", "[u32; 4]", "[u32; 4]", "u32", "u32"]
        );
        assert_eq!(record.rust_result, "[f32; 4]");
        assert_eq!(
            record.llvm_arguments,
            [
                "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "f32", "f32", "f32", "f32",
                "i32", "i32"
            ]
        );
        assert_eq!(record.llvm_results, ["f32", "f32", "f32", "f32"]);
        assert_eq!(
            record.llvm_symbol.as_deref(),
            Some(
                format!(
                    "llvm.nvvm.mma.sp.m16n8k64.row.col.f32.{}.{}.f32",
                    sparse_mma_element_name(a_element),
                    sparse_mma_element_name(b_element)
                )
                .as_str()
            )
        );
        assert_eq!(
            record.expected_ptx.modifiers,
            [
                "sp",
                "sync",
                "aligned",
                "m16n8k64",
                "row",
                "col",
                "f32",
                sparse_mma_element_name(a_element),
                sparse_mma_element_name(b_element),
                "f32",
            ]
        );
        assert_eq!(
            record.expected_ptx.operands,
            [
                OperandPattern::RegisterList { length: 4 },
                OperandPattern::RegisterList { length: 4 },
                OperandPattern::RegisterList { length: 4 },
                OperandPattern::RegisterList { length: 4 },
                OperandPattern::Register,
                OperandPattern::Immediate,
            ]
        );
        assert_eq!(
            record.operation_key,
            format!(
                "matrix.mma.sp.m16n8k64.row.col.f32.{}.{}.f32.not_applicable.standard_metadata",
                sparse_mma_element_name(a_element),
                sparse_mma_element_name(b_element)
            )
        );
    }

    // The operation keys stay unique against the ordered kind::f8f6f4 matrix.
    let (mut overlay, _) =
        read_overlay(&repo_root, &repo_root.join("intrinsics/overlay.toml")).unwrap();
    bind_pinned_abi_ids(&repo_root, &mut overlay);
    let keys = overlay
        .intrinsics
        .iter()
        .map(|record| record.operation_key.clone())
        .collect::<BTreeSet<_>>();
    assert_eq!(keys.len(), overlay.intrinsics.len());
    for record in &records {
        assert!(keys.contains(&record.operation_key));
    }

    // The admission is closed: only the reviewed 2x2 e4m3/e5m2 matrix is admitted.
    let mut wrong_elements = test_sparse_mma_fp8_f32_admission();
    wrong_elements.a_elements = vec![SparseMmaElement::E2m1, SparseMmaElement::E4m3];
    assert!(expand_sparse_mma_fp8_f32_admission(&wrong_elements).is_err());

    let mut wrong_product = test_sparse_mma_fp8_f32_admission();
    wrong_product.product_count = 25;
    assert!(expand_sparse_mma_fp8_f32_admission(&wrong_product).is_err());

    let mut executed = test_sparse_mma_fp8_f32_admission();
    executed.runtime_validation = RuntimeValidation::Executed;
    assert!(expand_sparse_mma_fp8_f32_admission(&executed).is_err());

    // Standard-metadata admission must not leak into the other sparse floats.
    for (accumulator, metadata) in [
        (SparseMmaAccumulator::F16, SparseMmaMetadata::Standard),
        (SparseMmaAccumulator::F32, SparseMmaMetadata::Ordered),
    ] {
        let mma = crate::model::SparseMma {
            shape: SparseMmaShape::M16n8k64,
            accumulator,
            a_element: SparseMmaElement::E4m3,
            b_element: SparseMmaElement::E4m3,
            a_layout: crate::model::SparseMmaLayout::Row,
            b_layout: crate::model::SparseMmaLayout::Col,
            overflow: SparseMmaOverflow::NotApplicable,
            metadata,
            selector: SparseMmaSelector::ImmediateZero,
            participation:
                crate::model::SparseMmaParticipation::AllWarpLanesSameInstructionAndQualifiersNoExitedLanes,
            adapter: SparseMmaAdapter::C4F32A4U32B4U32MetadataU32SelectorU32ToD4F32,
            llvm_adapter: SparseMmaLlvmAdapter::A4I32B4I32C4F32MetadataI32SelectorI32ToD4F32,
            compatibility_source: crate::model::SparseMmaCompatibilitySource::GeneratedStub,
            runtime_validation: RuntimeValidation::Unexecuted,
        };
        if metadata == SparseMmaMetadata::Ordered {
            // The ordered spelling still resolves, to the existing kind::f8f6f4 identity.
            let recipe = sparse_mma_recipe(&mma).unwrap();
            assert_eq!(
                recipe.identity.id,
                "mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e4m3_e4m3_f32"
            );
            assert_eq!(sparse_mma_minimum_ptx(&mma), "8.7");
            assert_eq!(sparse_mma_hardware(&mma), (SPARSE_MMA_F8F6F4_TARGETS, None));
        } else {
            // A packed-F16 accumulator has no plain-metadata sparse FP8 carrier.
            assert!(sparse_mma_recipe(&mma).is_none());
        }
    }
}
