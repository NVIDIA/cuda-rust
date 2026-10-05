/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, CatalogHardwareAlternative, CatalogTargetAlternative,
    DotProductOperation, DotProductSignedness, EvidenceArtifactKind, EvidenceFile, EvidenceRecord,
    EvidenceStageKind, IntrinsicBackend, LdmatrixElement, LdmatrixLayout, LdmatrixMultiplicity,
    LdmatrixShape, LdmatrixStateSpace, OverlayBackendLowering, PackedConversionDestinationFormat,
    PackedConversionRounding, PackedConversionSaturation, RuntimeValidation,
};
use crate::ptx::{InstructionPattern, OperandPattern};

use super::fixtures::*;
use crate::resolve::evidence::*;
use crate::resolve::families::*;

#[test]
fn legacy_evidence_schema_is_unchanged_and_rejects_matrix_fields() {
    let legacy = EvidenceFile {
        schema: 5,
        backend_profile: "legacy".into(),
        backend_kind: None,
        llvm_revision: "test".into(),
        backend_version: "LLVM legacy test".into(),
        backend_sha256: "0123456789abcdef".into(),
        artifact_path: None,
        build_id_prefix: None,
        nvvm_ir_version: None,
        debug_ir_version: None,
        records: vec![evidence()],
    };
    let bytes = serde_json::to_vec(&legacy).unwrap();
    assert_eq!(parse_evidence_bytes(&bytes, "legacy").unwrap(), legacy);

    let mut with_matrix_field = serde_json::to_value(&legacy).unwrap();
    with_matrix_field["matrices"] = serde_json::json!([]);
    let error = parse_synthetic_evidence(&with_matrix_field).unwrap_err();
    assert!(error.to_string().contains("legacy evidence"));
    assert!(format!("{error:#}").contains("unknown field"));
}

#[test]
fn compact_evidence_matrix_equals_explicit_records() {
    let expanded = parse_synthetic_evidence(&synthetic_matrix_json()).unwrap();
    let mut expected = vec![EvidenceRecord {
        id: "synthetic_explicit".into(),
        source: None,
        source_record: Some("int_synthetic_explicit".into()),
        llvm_symbol: Some("llvm.synthetic.explicit".into()),
        resolved_llvm_symbol: None,
        llvm_arguments: vec!["i32".into()],
        llvm_results: vec!["i32".into()],
        concrete_llvm_arguments: vec![],
        concrete_llvm_results: vec![],
        target_triple: "nvptx64-nvidia-cuda".into(),
        gpu_target: "sm_80".into(),
        ptx_feature: "+ptx71".into(),
        status: "lowered".into(),
        stages: vec![],
        declaration_attributes_canonicalized: None,
        runtime_validation: None,
        expected_ptx: InstructionPattern {
            mnemonic: "mma".into(),
            modifiers: vec!["sync".into(), "explicit".into()],
            operands: vec![OperandPattern::Register],
        },
    }];
    expected.extend(["s8", "u8"].into_iter().map(|element| EvidenceRecord {
        id: format!("synthetic_{element}"),
        source: None,
        source_record: Some(format!("int_synthetic_{element}")),
        llvm_symbol: Some(format!("llvm.synthetic.{element}")),
        resolved_llvm_symbol: None,
        llvm_arguments: vec!["i32".into()],
        llvm_results: vec!["i32".into()],
        concrete_llvm_arguments: vec![],
        concrete_llvm_results: vec![],
        target_triple: "nvptx64-nvidia-cuda".into(),
        gpu_target: "sm_80".into(),
        ptx_feature: "+ptx71".into(),
        status: "lowered".into(),
        stages: vec![shared_matrix_stage()],
        declaration_attributes_canonicalized: None,
        runtime_validation: None,
        expected_ptx: InstructionPattern {
            mnemonic: "mma".into(),
            modifiers: vec!["sync".into(), element.into()],
            operands: vec![OperandPattern::Register],
        },
    }));
    assert_eq!(expanded.schema, 6);
    assert_eq!(expanded.records, expected);
}

#[test]
fn matrix_identity_mutations_reach_existing_evidence_validation() {
    let mut expanded = parse_synthetic_evidence(&policy_matrix_json()).unwrap();
    let record = expanded.records.pop().unwrap();
    validate_test_evidence(&policy(), record.clone()).unwrap();

    let mut wrong_source = record.clone();
    wrong_source.source_record = Some("int_nvvm_read_ptx_sreg_tid_y".into());
    assert!(
        validate_test_evidence(&policy(), wrong_source)
            .unwrap_err()
            .to_string()
            .contains("source provenance mismatch")
    );

    let mut wrong_symbol = record.clone();
    wrong_symbol.llvm_symbol = Some("llvm.nvvm.read.ptx.sreg.tid.y".into());
    assert!(
        validate_test_evidence(&policy(), wrong_symbol)
            .unwrap_err()
            .to_string()
            .contains("signature mismatch")
    );

    let mut wrong_signature = record.clone();
    wrong_signature.llvm_arguments.push("i32".into());
    assert!(
        validate_test_evidence(&policy(), wrong_signature)
            .unwrap_err()
            .to_string()
            .contains("signature mismatch")
    );

    let mut wrong_ptx = record;
    wrong_ptx.expected_ptx.modifiers.push("changed".into());
    assert!(
        validate_test_evidence(&policy(), wrong_ptx)
            .unwrap_err()
            .to_string()
            .contains("PTX expectation mismatch")
    );
}

#[test]
fn evidence_matrix_rejects_bad_counts_fixtures_placeholders_and_collisions() {
    let base = synthetic_matrix_json();

    let mut bad_product = base.clone();
    bad_product["matrices"][0]["product_count"] = 3.into();
    assert_synthetic_evidence_error(&bad_product, "expands to 2 records");

    let mut unknown_fixture = base.clone();
    unknown_fixture["matrices"][0]["fixtures"][0] = "missing".into();
    assert_synthetic_evidence_error(&unknown_fixture, "unknown fixture");

    let mut uncovered_fixture = base.clone();
    let extra = uncovered_fixture["fixtures"][0].clone();
    uncovered_fixture["fixtures"]
        .as_array_mut()
        .unwrap()
        .push(extra);
    uncovered_fixture["fixtures"][1]["id"] = "unused".into();
    assert_synthetic_evidence_error(&uncovered_fixture, "not referenced");

    let mut wrong_coverage = base.clone();
    wrong_coverage["fixtures"][0]["coverage_count"] = 1.into();
    assert_synthetic_evidence_error(&wrong_coverage, "covers 2 expanded records");

    let mut malformed = base.clone();
    malformed["matrices"][0]["template"]["id"] = "synthetic_$element".into();
    assert_synthetic_evidence_error(&malformed, "malformed matrix placeholder");

    let mut unknown_axis = base.clone();
    unknown_axis["matrices"][0]["template"]["id"] = "synthetic_${other}".into();
    assert_synthetic_evidence_error(&unknown_axis, "unknown matrix axis");

    let mut collision = base.clone();
    collision["matrices"][0]["template"]["id"] = "synthetic".into();
    assert_synthetic_evidence_error(&collision, "duplicate expanded evidence ID");
}

#[test]
fn exact_operand_matrix_placeholders_fail_closed() {
    let base = policy_matrix_json();

    let mut unknown = base.clone();
    unknown["matrices"][0]["template"]["expected_ptx"]["operands"][1]["value"] =
        "%tid.${other}".into();
    assert_synthetic_evidence_error(&unknown, "unknown matrix axis other");

    let mut unterminated = base.clone();
    unterminated["matrices"][0]["template"]["expected_ptx"]["operands"][1]["value"] =
        "%tid.${axis".into();
    assert_synthetic_evidence_error(&unterminated, "unterminated matrix placeholder");

    let mut disallowed = base;
    disallowed["matrices"][0]["template"]["expected_ptx"]["mnemonic"] = "mov.${axis}".into();
    assert_synthetic_evidence_error(&disallowed, "PTX mnemonic cannot contain");
}

#[test]
fn evidence_matrix_rejects_bad_axes_fixture_ids_and_stage_conflicts() {
    let base = synthetic_matrix_json();

    let mut no_fixture = base.clone();
    no_fixture["matrices"][0]["fixtures"] = serde_json::json!([]);
    assert_synthetic_evidence_error(&no_fixture, "references no shared fixture");

    let mut empty_axes = base.clone();
    empty_axes["matrices"][0]["axes"] = serde_json::json!([]);
    assert_synthetic_evidence_error(&empty_axes, "has no axes");

    let mut duplicate_axis = base.clone();
    let axis = duplicate_axis["matrices"][0]["axes"][0].clone();
    duplicate_axis["matrices"][0]["axes"]
        .as_array_mut()
        .unwrap()
        .push(axis);
    duplicate_axis["matrices"][0]["product_count"] = 4.into();
    assert_synthetic_evidence_error(&duplicate_axis, "axes must be unique and sorted");

    let mut empty_values = base.clone();
    empty_values["matrices"][0]["axes"][0]["values"] = serde_json::json!([]);
    assert_synthetic_evidence_error(&empty_values, "has no values");

    let mut empty_axis_name = base.clone();
    empty_axis_name["matrices"][0]["axes"][0]["name"] = "".into();
    assert_synthetic_evidence_error(&empty_axis_name, "is not a safe token");

    let mut empty_value = base.clone();
    empty_value["matrices"][0]["axes"][0]["values"][0] = "".into();
    assert_synthetic_evidence_error(&empty_value, "unsafe value");

    let mut duplicate_value = base.clone();
    duplicate_value["matrices"][0]["axes"][0]["values"][1] = "s8".into();
    assert_synthetic_evidence_error(&duplicate_value, "duplicate value");

    let mut unsafe_value = base.clone();
    unsafe_value["matrices"][0]["axes"][0]["values"][0] = "../s8".into();
    assert_synthetic_evidence_error(&unsafe_value, "unsafe value");

    let mut unused_axis = base.clone();
    unused_axis["matrices"][0]["axes"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({"name": "other", "values": ["x"]}));
    assert_synthetic_evidence_error(&unused_axis, "unused axis");

    let mut duplicate_fixture = base.clone();
    let fixture = duplicate_fixture["fixtures"][0].clone();
    duplicate_fixture["fixtures"]
        .as_array_mut()
        .unwrap()
        .push(fixture);
    assert_synthetic_evidence_error(&duplicate_fixture, "duplicate evidence fixture ID");

    let mut fixture_placeholder = base.clone();
    fixture_placeholder["fixtures"][0]["stages"][0]["detail"] = "covers ${element}".into();
    assert_synthetic_evidence_error(&fixture_placeholder, "cannot contain matrix placeholders");

    let mut missing_symbol = base.clone();
    missing_symbol["matrices"][0]["template"]
        .as_object_mut()
        .unwrap()
        .remove("llvm_symbol");
    assert_synthetic_evidence_error(&missing_symbol, "missing field `llvm_symbol`");

    let mut conflicting_stage = base;
    conflicting_stage["matrices"][0]["template"]["facts"]["stages"] =
        conflicting_stage["fixtures"][0]["stages"].clone();
    assert_synthetic_evidence_error(&conflicting_stage, "conflicting duplicate");
}

#[test]
fn typed_evidence_accepts_direct_scalar_intrinsic_signatures() {
    let policy = dot_product_policy(DotProductOperation::Dp2a, DotProductSignedness::Signed);
    let mut record = dot_product_evidence(&policy);
    validate_typed_llvm_evidence(&policy, &record).unwrap();

    record.concrete_llvm_arguments.remove(2);
    let error = validate_typed_llvm_evidence(&policy, &record).unwrap_err();
    assert!(error.to_string().contains("resolved signature"));
}

#[test]
fn typed_evidence_maps_cluster_shared_pointers_to_address_space_seven() {
    let mut policy = dot_product_policy(DotProductOperation::Dp2a, DotProductSignedness::Signed);
    let mut record = dot_product_evidence(&policy);
    policy.llvm_arguments = vec![
        "shared_cluster_ptr".into(),
        "shared_ptr".into(),
        "ptr".into(),
    ];
    record.concrete_llvm_arguments = vec![
        "ptr addrspace(7)".into(),
        "ptr addrspace(3)".into(),
        "ptr".into(),
    ];

    validate_typed_llvm_evidence(&policy, &record).unwrap();
}

#[test]
fn packed_conversion_evidence_separates_llvm_declaration_facts_from_libnvvm() {
    for (destination, result) in [
        (PackedConversionDestinationFormat::Bf16x2, "v2bf16"),
        (PackedConversionDestinationFormat::F16x2, "v2f16"),
    ] {
        let policy = packed_conversion_policy(
            destination,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::None,
        );
        let llvm = policy
            .backend_lowerings
            .iter()
            .find(|lowering| lowering.backend == IntrinsicBackend::LlvmNvptx)
            .unwrap();
        let mut record = packed_conversion_evidence(&policy);
        record.status = "validated".into();
        record.stages = [
            EvidenceStageKind::DeclarationCanonicalization,
            EvidenceStageKind::BackendCodegen,
            EvidenceStageKind::PtxAssembly,
        ]
        .into_iter()
        .map(|stage| {
            evidence_stage(
                stage,
                BackendLoweringMechanism::TypedNvvm,
                &["sm_80", "ptx70"],
            )
        })
        .collect();
        let assembly = record
            .stages
            .iter_mut()
            .find(|stage| stage.stage == EvidenceStageKind::PtxAssembly)
            .unwrap();
        assembly.tool_path = Some("/usr/local/cuda/bin/ptxas".into());
        assembly.tool_version = Some("CUDA 13.3 V13.3.33".into());
        assembly.tool_sha256 =
            Some("7fdd01a4cf50e30746da98989c9272a907f491e6fd7fecfda14642e4375f88fb".into());
        assert_eq!(record.concrete_llvm_results, [result]);
        validate_packed_conversion_backend_evidence(&policy, &record, llvm).unwrap();

        let mut lowered = record.clone();
        lowered.status = "lowered".into();
        let error =
            validate_packed_conversion_backend_evidence(&policy, &lowered, llvm).unwrap_err();
        assert!(
            error.to_string().contains("validated evidence status"),
            "{error:#}"
        );

        for required in [
            EvidenceStageKind::DeclarationCanonicalization,
            EvidenceStageKind::BackendCodegen,
            EvidenceStageKind::PtxAssembly,
        ] {
            let mut missing = record.clone();
            missing.stages.retain(|stage| stage.stage != required);
            let error =
                validate_packed_conversion_backend_evidence(&policy, &missing, llvm).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("successful auxiliary typed-NVVM"),
                "{error:#}"
            );

            let mut failed = record.clone();
            failed
                .stages
                .iter_mut()
                .find(|stage| stage.stage == required)
                .unwrap()
                .outcome = "failed".into();
            let error =
                validate_packed_conversion_backend_evidence(&policy, &failed, llvm).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("successful auxiliary typed-NVVM"),
                "{error:#}"
            );

            let mut wrong_mechanism = record.clone();
            wrong_mechanism
                .stages
                .iter_mut()
                .find(|stage| stage.stage == required)
                .unwrap()
                .mechanism = Some(BackendLoweringMechanism::InlinePtx);
            let error =
                validate_packed_conversion_backend_evidence(&policy, &wrong_mechanism, llvm)
                    .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("successful auxiliary typed-NVVM"),
                "{error:#}"
            );
        }

        let mut missing_tool_identity = record.clone();
        missing_tool_identity
            .stages
            .iter_mut()
            .find(|stage| stage.stage == EvidenceStageKind::PtxAssembly)
            .unwrap()
            .tool_sha256 = None;
        let error =
            validate_packed_conversion_backend_evidence(&policy, &missing_tool_identity, llvm)
                .unwrap_err();
        assert!(
            error.to_string().contains("exact tool identity"),
            "{error:#}"
        );

        for stage_kind in [
            EvidenceStageKind::BackendCodegen,
            EvidenceStageKind::PtxAssembly,
        ] {
            let mut wrong_floor = record.clone();
            wrong_floor
                .stages
                .iter_mut()
                .find(|stage| stage.stage == stage_kind)
                .unwrap()
                .targets = vec!["sm_75".into(), "ptx70".into()];
            let error = validate_packed_conversion_backend_evidence(&policy, &wrong_floor, llvm)
                .unwrap_err();
            assert!(
                error.to_string().contains("catalog floor sm_80"),
                "{error:#}"
            );
        }

        record.declaration_attributes_canonicalized = None;
        let error =
            validate_packed_conversion_backend_evidence(&policy, &record, llvm).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("canonical declaration attributes")
        );
    }

    let policy = packed_conversion_policy(
        PackedConversionDestinationFormat::Bf16x2,
        PackedConversionRounding::NearestEven,
        PackedConversionSaturation::None,
    );
    let libnvvm = policy
        .backend_lowerings
        .iter()
        .find(|lowering| lowering.backend == IntrinsicBackend::LibNvvm)
        .unwrap();
    let mut record = packed_conversion_evidence(&policy);
    record.concrete_llvm_arguments.clear();
    record.concrete_llvm_results.clear();
    record.declaration_attributes_canonicalized = None;
    validate_packed_conversion_backend_evidence(&policy, &record, libnvvm).unwrap();

    record.concrete_llvm_arguments = policy.llvm_arguments.clone();
    let error = validate_packed_conversion_backend_evidence(&policy, &record, libnvvm).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("must not claim typed LLVM support")
    );

    record.concrete_llvm_arguments.clear();
    record.stages.push(evidence_stage(
        EvidenceStageKind::BackendCodegen,
        BackendLoweringMechanism::TypedNvvm,
        &["sm_80", "ptx70"],
    ));
    let error = validate_packed_conversion_backend_evidence(&policy, &record, libnvvm).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("must not claim typed LLVM support")
    );
}

#[test]
fn signature_and_evidence_mismatches_are_rejected() {
    let mut imported = declaration();
    imported.results = vec!["i64".into()];
    assert!(
        validate_imported_policy(&policy(), &imported)
            .unwrap_err()
            .to_string()
            .contains("LLVM result signature mismatch")
    );

    let mut backend_evidence = evidence();
    backend_evidence.llvm_results = vec!["i64".into()];
    assert!(
        validate_test_evidence(&policy(), backend_evidence)
            .unwrap_err()
            .to_string()
            .contains("evidence signature mismatch")
    );

    let mut backend_evidence = evidence();
    backend_evidence.expected_ptx = sreg_pattern("%tid.y");
    assert!(
        validate_test_evidence(&policy(), backend_evidence)
            .unwrap_err()
            .to_string()
            .contains("evidence PTX expectation mismatch")
    );
}

#[test]
fn validated_llvm_evidence_requires_exact_ptxas_identity() {
    let mut record = evidence();
    record.status = "validated".into();
    record.stages.push(crate::model::EvidenceStage {
        targets: vec!["sm_75".into()],
        representation: "probe PTX".into(),
        stage: EvidenceStageKind::PtxAssembly,
        mechanism: Some(BackendLoweringMechanism::TypedNvvm),
        outcome: "succeeded".into(),
        detail: "accepted".into(),
        artifact_kind: None,
        tool_path: Some("/usr/local/cuda/bin/ptxas".into()),
        tool_version: Some("CUDA 13.3 V13.3.33".into()),
        tool_sha256: Some(
            "7fdd01a4cf50e30746da98989c9272a907f491e6fd7fecfda14642e4375f88fb".into(),
        ),
    });
    assert!(has_valid_ptx_assembly_stage(
        &record,
        BackendLoweringMechanism::TypedNvvm
    ));

    let stage = record.stages.last_mut().unwrap();
    stage.tool_path = None;
    assert!(!has_valid_ptx_assembly_stage(
        &record,
        BackendLoweringMechanism::TypedNvvm
    ));
    record.stages.clear();
    assert!(!has_valid_ptx_assembly_stage(
        &record,
        BackendLoweringMechanism::TypedNvvm
    ));
}

#[test]
fn validated_libnvvm_evidence_requires_a_real_cubin_terminal() {
    let mut record = evidence();
    record.stages.push(crate::model::EvidenceStage {
        targets: vec!["sm_90".into(), "ptx78".into()],
        representation: "linked output".into(),
        stage: EvidenceStageKind::DeviceLink,
        mechanism: Some(BackendLoweringMechanism::InlinePtx),
        outcome: "succeeded".into(),
        detail: "test".into(),
        artifact_kind: None,
        tool_path: Some("/usr/local/cuda-13.3/lib64/libnvJitLink.so.13.3.33".into()),
        tool_version: Some("V13.3.33".into()),
        tool_sha256: Some(
            "3ba1e744347cd68617b862eccfd98b125482e882b7a6319f42abc9a768513db8".into(),
        ),
    });
    assert!(!has_valid_cubin_device_link_stage(
        &record,
        BackendLoweringMechanism::InlinePtx
    ));
    record.stages[0].artifact_kind = Some(EvidenceArtifactKind::Cubin);
    assert!(has_valid_cubin_device_link_stage(
        &record,
        BackendLoweringMechanism::InlinePtx
    ));
}

fn evidence_stage(
    stage: EvidenceStageKind,
    mechanism: BackendLoweringMechanism,
    targets: &[&str],
) -> crate::model::EvidenceStage {
    crate::model::EvidenceStage {
        targets: targets.iter().map(|target| (*target).into()).collect(),
        representation: "test".into(),
        stage,
        mechanism: Some(mechanism),
        outcome: "succeeded".into(),
        detail: "test".into(),
        artifact_kind: None,
        tool_path: None,
        tool_version: None,
        tool_sha256: None,
    }
}

#[test]
fn backend_stage_targets_and_executed_status_are_monotonic() {
    let mut target_policy = policy();
    target_policy.minimum_ptx = "6.5".into();
    target_policy.minimum_sm = Some("sm_75".into());
    let lowering = crate::model::OverlayBackendLowering {
        backend: IntrinsicBackend::LlvmNvptx,
        mechanism: BackendLoweringMechanism::TypedNvvm,
        evidence_profile: "test".into(),
        targets: None,
        minimum_ptx: None,
        minimum_sm: None,
    };
    let mut record = evidence();
    record.status = "validated".into();
    record.runtime_validation = Some(RuntimeValidation::Unexecuted);
    record.stages = vec![
        evidence_stage(
            EvidenceStageKind::BackendCodegen,
            BackendLoweringMechanism::TypedNvvm,
            &["sm_75", "ptx65"],
        ),
        evidence_stage(
            EvidenceStageKind::PtxAssembly,
            BackendLoweringMechanism::TypedNvvm,
            &["sm_75", "ptx65"],
        ),
    ];
    validate_selected_stage_targets(&target_policy, &record, &lowering).unwrap();

    record.stages[0].targets = vec!["sm_75a".into(), "ptx65".into()];
    assert!(validate_selected_stage_targets(&target_policy, &record, &lowering).is_err());
    record.stages[0].targets = vec!["sm_75".into(), "ptx65".into()];

    record.stages[1].targets = vec!["sm_80".into(), "ptx65".into()];
    validate_selected_stage_targets(&target_policy, &record, &lowering).unwrap();

    record.stages[1].targets = vec!["sm_90a".into(), "ptx65".into()];
    validate_selected_stage_targets(&target_policy, &record, &lowering).unwrap();

    record.stages[1].targets = vec!["sm_74".into(), "ptx65".into()];
    assert!(
        validate_selected_stage_targets(&target_policy, &record, &lowering)
            .unwrap_err()
            .to_string()
            .contains("catalog floor sm_75")
    );

    record.stages[1].targets = vec!["sm_75".into(), "ptx65".into()];
    record.status = "executed".into();
    record.runtime_validation = Some(RuntimeValidation::Executed);
    assert!(
        validate_selected_stage_targets(&target_policy, &record, &lowering)
            .unwrap_err()
            .to_string()
            .contains("runtime stage")
    );
}

#[test]
fn exact_and_family_evidence_targets_match_at_every_stage() {
    let lowering = crate::model::OverlayBackendLowering {
        backend: IntrinsicBackend::LlvmNvptx,
        mechanism: BackendLoweringMechanism::InlinePtx,
        evidence_profile: "test".into(),
        targets: None,
        minimum_ptx: None,
        minimum_sm: None,
    };

    for (target, wrong_targets) in [
        ("sm_120a", ["sm_120", "sm_120f", "sm_121a"]),
        ("sm_120f", ["sm_120", "sm_120a", "sm_121f"]),
    ] {
        let mut target_policy = policy();
        target_policy.minimum_ptx = "8.7".into();
        target_policy.targets = target.into();
        let mut record = evidence();
        record.status = "validated".into();
        record.runtime_validation = Some(RuntimeValidation::Unexecuted);
        record.stages = vec![
            evidence_stage(
                EvidenceStageKind::BackendCodegen,
                BackendLoweringMechanism::InlinePtx,
                &[target, "ptx87"],
            ),
            evidence_stage(
                EvidenceStageKind::PtxAssembly,
                BackendLoweringMechanism::InlinePtx,
                &[target, "ptx87"],
            ),
        ];
        validate_selected_stage_targets(&target_policy, &record, &lowering).unwrap();

        for wrong in wrong_targets {
            record.stages[0].targets = vec![wrong.into(), "ptx87".into()];
            let error = validate_selected_stage_targets(&target_policy, &record, &lowering)
                .unwrap_err()
                .to_string();
            assert!(error.contains(target), "{error}");
        }
        record.stages[0].targets = vec![target.into(), "ptx87".into()];
        for wrong in wrong_targets {
            record.stages[1].targets = vec![wrong.into(), "ptx87".into()];
            let error = validate_selected_stage_targets(&target_policy, &record, &lowering)
                .unwrap_err()
                .to_string();
            assert!(error.contains(target), "{error}");
        }
    }
}

#[test]
fn suffixed_evidence_target_spellings_are_normalized() {
    for target in ["sm_120a", "compute_120a", "sm_120f", "compute_120f"] {
        assert!(is_normalized_stage_target(target), "{target}");
    }
    for target in ["sm_120", "compute_120", "ptx87"] {
        assert!(is_normalized_stage_target(target), "{target}");
    }
    for target in ["sm_0120a", "sm_120af", "sm_120x", "compute_120A"] {
        assert!(!is_normalized_stage_target(target), "{target}");
    }
}

#[test]
fn libnvvm_stage_may_report_newer_ptx_than_the_native_instruction_floor() {
    let mut target_policy = policy();
    target_policy.minimum_ptx = "1.0".into();
    target_policy.minimum_sm = None;
    let lowering = crate::model::OverlayBackendLowering {
        backend: IntrinsicBackend::LibNvvm,
        mechanism: BackendLoweringMechanism::InlinePtx,
        evidence_profile: "test".into(),
        targets: None,
        minimum_ptx: None,
        minimum_sm: Some("sm_75".into()),
    };
    let mut record = evidence();
    record.stages = vec![evidence_stage(
        EvidenceStageKind::BackendCodegen,
        BackendLoweringMechanism::InlinePtx,
        &["sm_75", "ptx93"],
    )];
    validate_selected_stage_targets(&target_policy, &record, &lowering).unwrap();

    record.stages[0].targets = vec!["sm_75".into(), "ptx09".into()];
    assert!(
        validate_selected_stage_targets(&target_policy, &record, &lowering)
            .unwrap_err()
            .to_string()
            .contains("catalog floor sm_75 / PTX 1.0")
    );

    let llvm_lowering = crate::model::OverlayBackendLowering {
        backend: IntrinsicBackend::LlvmNvptx,
        mechanism: BackendLoweringMechanism::TypedNvvm,
        evidence_profile: "test".into(),
        targets: None,
        minimum_ptx: Some("3.2".into()),
        minimum_sm: Some("sm_20".into()),
    };
    record.stages = vec![evidence_stage(
        EvidenceStageKind::BackendCodegen,
        BackendLoweringMechanism::TypedNvvm,
        &["sm_20", "ptx93"],
    )];
    assert!(
        validate_selected_stage_targets(&target_policy, &record, &llvm_lowering)
            .unwrap_err()
            .to_string()
            .contains("catalog floor sm_20 / PTX 3.2")
    );
}

#[test]
fn imported_selection_must_match_the_full_ptx_shape() {
    let mut imported = declaration();
    imported.selections[0].asm = "mov.u32 $d, %tid.xy;".into();
    let error = validate_imported_policy(&policy(), &imported).unwrap_err();
    assert!(error.to_string().contains("does not agree"));

    imported.selections[0].asm = "mov.u32.relaxed $d, %tid.x;".into();
    let error = validate_imported_policy(&policy(), &imported).unwrap_err();
    assert!(error.to_string().contains("does not agree"));

    imported.selections[0].asm = "mov.u32 $d, %tid.x;".into();
    validate_imported_policy(&policy(), &imported).unwrap();
}

#[test]
fn blackwell_ldmatrix_evidence_covers_every_target_with_its_effective_ptx_floor() {
    let mut policy = policy();
    policy.family = "ldmatrix".into();
    policy.minimum_ptx = "8.6".into();
    policy.minimum_sm = None;
    policy.targets = BLACKWELL_LDMATRIX_LLVM_TARGETS.into();
    policy.ldmatrix_variant = Some(crate::model::LdmatrixVariant {
        shape: LdmatrixShape::M16n16,
        multiplicity: LdmatrixMultiplicity::X1,
        layout: LdmatrixLayout::Transposed,
        element: LdmatrixElement::B8,
        state_space: LdmatrixStateSpace::Shared,
    });
    let lowering = OverlayBackendLowering {
        backend: IntrinsicBackend::LlvmNvptx,
        mechanism: BackendLoweringMechanism::TypedNvvm,
        evidence_profile: "test".into(),
        targets: None,
        minimum_ptx: None,
        minimum_sm: None,
    };
    let mut record = evidence();
    record.status = "validated".into();
    record.stages.clear();
    for (target, ptx) in [
        ("sm_100a", "ptx86"),
        ("sm_100f", "ptx88"),
        ("sm_103a", "ptx88"),
        ("sm_103f", "ptx88"),
        ("sm_110a", "ptx90"),
        ("sm_110f", "ptx90"),
        ("sm_120a", "ptx87"),
        ("sm_120f", "ptx88"),
        ("sm_121a", "ptx88"),
        ("sm_121f", "ptx88"),
    ] {
        record.stages.push(evidence_stage(
            EvidenceStageKind::BackendCodegen,
            BackendLoweringMechanism::TypedNvvm,
            &[target, ptx],
        ));
        let mut assembly = evidence_stage(
            EvidenceStageKind::PtxAssembly,
            BackendLoweringMechanism::TypedNvvm,
            &[target, ptx],
        );
        assembly.tool_path = Some("/tool/ptxas".into());
        assembly.tool_version = Some("test".into());
        assembly.tool_sha256 = Some("0".repeat(64));
        record.stages.push(assembly);
    }
    validate_selected_stage_targets(&policy, &record, &lowering).unwrap();

    record.stages.retain(|stage| {
        !(stage.stage == EvidenceStageKind::PtxAssembly
            && stage.targets.iter().any(|target| target == "sm_121f"))
    });
    assert!(
        validate_selected_stage_targets(&policy, &record, &lowering)
            .unwrap_err()
            .to_string()
            .contains("one structured stage for each")
    );
}

#[test]
fn paired_target_evidence_checks_backend_and_runtime_floors() {
    let policy = policy();
    let hardware = vec![
        CatalogHardwareAlternative::ExactArchitecture { sm: 100 },
        CatalogHardwareAlternative::ExactArchitecture { sm: 103 },
    ];
    let paired = vec![
        CatalogTargetAlternative {
            minimum_ptx: "8.6".parse().unwrap(),
            hardware: hardware[0],
        },
        CatalogTargetAlternative {
            minimum_ptx: "8.8".parse().unwrap(),
            hardware: hardware[1],
        },
    ];
    let llvm = OverlayBackendLowering {
        backend: IntrinsicBackend::LlvmNvptx,
        mechanism: BackendLoweringMechanism::InlinePtx,
        evidence_profile: "test".into(),
        targets: None,
        minimum_ptx: None,
        minimum_sm: None,
    };
    let mut llvm_record = evidence();
    llvm_record.status = "validated".into();
    llvm_record.stages.clear();
    for (target, ptx) in [("sm_100a", "ptx86"), ("sm_103a", "ptx88")] {
        llvm_record.stages.push(evidence_stage(
            EvidenceStageKind::BackendCodegen,
            llvm.mechanism,
            &[target, ptx],
        ));
        let mut assembly = evidence_stage(
            EvidenceStageKind::PtxAssembly,
            llvm.mechanism,
            &[target, ptx],
        );
        assembly.tool_path = Some("/tool/ptxas".into());
        assembly.tool_version = Some("test".into());
        assembly.tool_sha256 = Some("0".repeat(64));
        llvm_record.stages.push(assembly);
    }
    validate_target_matrix_stage_targets(
        &policy,
        &llvm_record,
        &llvm,
        EvidenceStageKind::PtxAssembly,
        &hardware,
        86,
        Some(&paired),
    )
    .unwrap();

    let mut wrong_floor = llvm_record.clone();
    wrong_floor
        .stages
        .iter_mut()
        .find(|stage| {
            stage.stage == EvidenceStageKind::BackendCodegen
                && stage.targets.iter().any(|target| target == "sm_103a")
        })
        .unwrap()
        .targets = vec!["sm_103a".into(), "ptx86".into()];
    assert!(
        validate_target_matrix_stage_targets(
            &policy,
            &wrong_floor,
            &llvm,
            EvidenceStageKind::PtxAssembly,
            &hardware,
            86,
            Some(&paired),
        )
        .unwrap_err()
        .to_string()
        .contains("wrong PTX floor")
    );

    let libnvvm = OverlayBackendLowering {
        backend: IntrinsicBackend::LibNvvm,
        ..llvm.clone()
    };
    let mut libnvvm_record = evidence();
    libnvvm_record.status = "validated".into();
    libnvvm_record.stages.clear();
    for (target, ptx) in [("sm_100a", "ptx88"), ("sm_103a", "ptx90")] {
        libnvvm_record.stages.push(evidence_stage(
            EvidenceStageKind::BackendCodegen,
            libnvvm.mechanism,
            &[target, ptx],
        ));
        let mut assembly = evidence_stage(
            EvidenceStageKind::PtxAssembly,
            libnvvm.mechanism,
            &[target, ptx],
        );
        assembly.tool_path = Some("/tool/ptxas".into());
        assembly.tool_version = Some("test".into());
        assembly.tool_sha256 = Some("0".repeat(64));
        libnvvm_record.stages.push(assembly);
        let mut link = evidence_stage(
            EvidenceStageKind::DeviceLink,
            libnvvm.mechanism,
            &[target, ptx],
        );
        link.artifact_kind = Some(EvidenceArtifactKind::Cubin);
        link.tool_path = Some("/tool/nvlink".into());
        link.tool_version = Some("test".into());
        link.tool_sha256 = Some("0".repeat(64));
        libnvvm_record.stages.push(link);
    }
    validate_target_matrix_stage_targets(
        &policy,
        &libnvvm_record,
        &libnvvm,
        EvidenceStageKind::DeviceLink,
        &hardware,
        86,
        Some(&paired),
    )
    .unwrap();

    let mut executed = llvm_record;
    executed.status = "executed".into();
    executed.stages.push(evidence_stage(
        EvidenceStageKind::Runtime,
        llvm.mechanism,
        &["sm_103a", "ptx88"],
    ));
    validate_target_matrix_stage_targets(
        &policy,
        &executed,
        &llvm,
        EvidenceStageKind::PtxAssembly,
        &hardware,
        86,
        Some(&paired),
    )
    .unwrap();

    let mut wrong_runtime_ptx = executed.clone();
    wrong_runtime_ptx.stages.last_mut().unwrap().targets = vec!["sm_103a".into(), "ptx87".into()];
    assert!(
        validate_target_matrix_stage_targets(
            &policy,
            &wrong_runtime_ptx,
            &llvm,
            EvidenceStageKind::PtxAssembly,
            &hardware,
            86,
            Some(&paired),
        )
        .unwrap_err()
        .to_string()
        .contains("paired floor")
    );

    let mut wrong_runtime_hardware = executed;
    wrong_runtime_hardware.stages.last_mut().unwrap().targets =
        vec!["sm_110a".into(), "ptx90".into()];
    assert!(
        validate_target_matrix_stage_targets(
            &policy,
            &wrong_runtime_hardware,
            &llvm,
            EvidenceStageKind::PtxAssembly,
            &hardware,
            86,
            Some(&paired),
        )
        .unwrap_err()
        .to_string()
        .contains("outside its target matrix")
    );
}

#[test]
fn sparse_f8f6f4_f16_evidence_covers_every_target_and_floor() {
    let policy = expand_sparse_mma_f8f6f4_f16_admission(&test_sparse_mma_f8f6f4_f16_admission())
        .unwrap()
        .remove(0);
    let lowering = policy
        .backend_lowerings
        .iter()
        .find(|lowering| lowering.backend == IntrinsicBackend::LlvmNvptx)
        .unwrap();
    let mut record = evidence();
    record.status = "validated".into();
    record.stages.clear();
    for (target, ptx) in [
        ("sm_120a", "ptx87"),
        ("sm_120f", "ptx88"),
        ("sm_121a", "ptx88"),
        ("sm_121f", "ptx88"),
    ] {
        record.stages.push(evidence_stage(
            EvidenceStageKind::BackendCodegen,
            BackendLoweringMechanism::InlinePtx,
            &[target, ptx],
        ));
        let mut assembly = evidence_stage(
            EvidenceStageKind::PtxAssembly,
            BackendLoweringMechanism::InlinePtx,
            &[target, ptx],
        );
        assembly.artifact_kind = Some(EvidenceArtifactKind::Cubin);
        assembly.tool_path = Some("/tool/ptxas".into());
        assembly.tool_version = Some("test".into());
        assembly.tool_sha256 = Some("0".repeat(64));
        record.stages.push(assembly);
    }
    validate_selected_stage_targets(&policy, &record, lowering).unwrap();

    let valid = record.clone();
    record.stages.retain(|stage| {
        !(stage.stage == EvidenceStageKind::PtxAssembly
            && stage.targets.iter().any(|target| target == "sm_121f"))
    });
    assert!(
        validate_selected_stage_targets(&policy, &record, lowering)
            .unwrap_err()
            .to_string()
            .contains("one structured stage for each")
    );

    record = valid;
    record
        .stages
        .iter_mut()
        .find(|stage| {
            stage.stage == EvidenceStageKind::BackendCodegen
                && stage.targets.iter().any(|target| target == "sm_120f")
        })
        .unwrap()
        .targets = vec!["sm_120f".into(), "ptx87".into()];
    assert!(
        validate_selected_stage_targets(&policy, &record, lowering)
            .unwrap_err()
            .to_string()
            .contains("wrong PTX floor")
    );
}

#[test]
fn non_blackwell_evidence_rejects_multiple_target_specific_stage_pairs() {
    let policy = policy();
    let lowering = OverlayBackendLowering {
        backend: IntrinsicBackend::LlvmNvptx,
        mechanism: BackendLoweringMechanism::TypedNvvm,
        evidence_profile: "test".into(),
        targets: None,
        minimum_ptx: None,
        minimum_sm: None,
    };
    let mut record = evidence();
    record.stages = vec![
        evidence_stage(
            EvidenceStageKind::BackendCodegen,
            BackendLoweringMechanism::TypedNvvm,
            &["sm_20", "ptx20"],
        ),
        evidence_stage(
            EvidenceStageKind::BackendCodegen,
            BackendLoweringMechanism::TypedNvvm,
            &["sm_21", "ptx20"],
        ),
    ];

    assert!(
        validate_selected_stage_targets(&policy, &record, &lowering)
            .unwrap_err()
            .to_string()
            .contains("outside reviewed target-matrix evidence")
    );
}
