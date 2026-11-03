/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, CatalogHalfOpenRange, CatalogHardwareAlternative,
    CatalogHardwareTarget, CatalogTargetAlternative, CatalogTargetContract,
    CatalogTargetRequirement, IntrinsicBackend, LdmatrixElement, LdmatrixLayout,
    LdmatrixMultiplicity, LdmatrixShape, LdmatrixStateSpace, OverlayIntrinsic, PtxVersion,
    ReduxAdapter, ReduxOperation, ReduxParticipation, TargetContract, TargetSelectorBinding,
};

use super::fixtures::*;
use crate::model::ImportedSelection;
use crate::resolve::abi_ledger::*;
use crate::resolve::driver::*;
use crate::resolve::evidence::*;
use crate::resolve::families::*;
use crate::resolve::guards::*;
use crate::resolve::materialize::*;
use crate::resolve::overlay::*;
use crate::resolve::policy::*;
use crate::resolve::targets::*;

#[test]
fn selected_target_predicates_fail_closed() {
    let selection = ImportedSelection {
        source_record: "selection".into(),
        asm: "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{$r0, $r1, $r2, $r3}}, [$src];".into(),
        predicates: vec![
            "Subtarget->getPTXVersion() >= 65".into(),
            "Subtarget->getSmVersion() >= 75".into(),
        ],
        constraints: Default::default(),
    };

    let mut too_low_ptx = policy();
    too_low_ptx.minimum_ptx = "6.4".into();
    too_low_ptx.minimum_sm = Some("sm_75".into());
    assert!(
        validate_selected_target_predicates(&too_low_ptx, &selection)
            .unwrap_err()
            .to_string()
            .contains("minimum PTX")
    );

    let mut too_low_sm = policy();
    too_low_sm.minimum_ptx = "6.5".into();
    too_low_sm.minimum_sm = Some("sm_74".into());
    assert!(
        validate_selected_target_predicates(&too_low_sm, &selection)
            .unwrap_err()
            .to_string()
            .contains("minimum SM")
    );

    let mut unknown = selection;
    unknown
        .predicates
        .push("Subtarget->hasMysteryFeature()".into());
    assert!(
        validate_selected_target_predicates(&too_low_sm, &unknown)
            .unwrap_err()
            .to_string()
            .contains("fail closed")
    );
}

#[test]
fn blackwell_ldmatrix_predicate_requires_the_reviewed_exact_target_set() {
    let mut record = policy();
    record.family = "ldmatrix".into();
    record.minimum_ptx = "8.6".into();
    record.minimum_sm = None;
    record.targets = BLACKWELL_LDMATRIX_LLVM_TARGETS.into();
    record.ldmatrix_variant = Some(crate::model::LdmatrixVariant {
        shape: LdmatrixShape::M16n16,
        multiplicity: LdmatrixMultiplicity::X1,
        layout: LdmatrixLayout::Transposed,
        element: LdmatrixElement::B8,
        state_space: LdmatrixStateSpace::Shared,
    });
    let selection = ImportedSelection {
        source_record: "selection".into(),
        asm: "ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {{$r0, $r1}}, [$src];".into(),
        predicates: vec!["Subtarget->hasLdStmatrixBlackwellSupport()".into()],
        constraints: Default::default(),
    };

    validate_selected_target_predicates(&record, &selection).unwrap();

    record.targets = "sm_100a|sm_101a".into();
    assert!(validate_selected_target_predicates(&record, &selection).is_err());

    record.targets = BLACKWELL_LDMATRIX_LLVM_TARGETS.into();
    let mut conflicting = selection;
    conflicting
        .predicates
        .push("Subtarget->getPTXVersion() >= 86".into());
    assert!(validate_selected_target_predicates(&record, &conflicting).is_err());
}

#[test]
fn f32_redux_predicate_requires_the_reviewed_exact_target_matrix() {
    let mut record = policy();
    record.family = "redux".into();
    record.minimum_ptx = "8.6".into();
    record.minimum_sm = None;
    record.targets = REDUX_F32_TARGETS.into();
    record.redux = Some(crate::model::Redux {
        operation: ReduxOperation::Fmin,
        participation: ReduxParticipation::ExecutingLaneNamedAllNamedLanesSameInstructionAndMask,
        adapter: ReduxAdapter::MaskValueToSourceMemberMask,
    });
    let selection = ImportedSelection {
        source_record: "selection".into(),
        asm: "redux.sync.min.f32 \t$dst, $src, $mask;".into(),
        predicates: vec!["Subtarget->hasReduxSyncF32()".into()],
        constraints: Default::default(),
    };

    validate_selected_target_predicates(&record, &selection).unwrap();

    record.targets =
        "sm_100a|sm_100f|sm_103a|sm_103f|sm_110a|sm_110f|sm_120a|sm_120f|sm_121a|sm_121f".into();
    assert!(validate_selected_target_predicates(&record, &selection).is_err());

    record.targets = REDUX_F32_TARGETS.into();
    record.minimum_ptx = "8.8".into();
    assert!(validate_selected_target_predicates(&record, &selection).is_err());

    record.minimum_ptx = "8.6".into();
    let mut conflicting = selection;
    conflicting
        .predicates
        .push("Subtarget->getPTXVersion() >= 88".into());
    assert!(validate_selected_target_predicates(&record, &conflicting).is_err());
}

#[test]
fn intrinsic_abi_identity_is_stable_and_explicit() {
    let policy = policy();
    let declaration = declaration();
    let resolved = resolve_record(
        &policy,
        resolve_policy_source(&policy).unwrap(),
        Some(&declaration),
        &evidence(),
        "test",
        "LLVM version test",
        "0123456789abcdef",
        vec![],
        1,
    )
    .unwrap();

    assert_eq!(resolved.rust.abi_id, "i0001");
    assert_eq!(
        resolved.rust.canonical_path,
        "cuda_intrinsics::__cuda_oxide_intrinsic_abi_v1::i0001"
    );
    assert_eq!(
        resolved.rust.public_path,
        "cuda_intrinsics::sreg::thread_idx_x"
    );
    assert_eq!(
        resolved.rust.compatibility_paths,
        ["cuda_device::thread::threadIdx_x"]
    );
    assert_eq!(
        resolved.llvm.as_ref().unwrap().properties,
        [
            "IntrNoMem",
            "IntrSpeculatable",
            "NoUndef<ret>",
            "Range<ret,0,1024>"
        ]
    );
    assert!(resolved.llvm.as_ref().unwrap().result_facts.no_undef);
    assert_eq!(
        resolved.llvm.as_ref().unwrap().result_facts.range,
        Some(CatalogHalfOpenRange {
            lower: "0".into(),
            upper_exclusive: "1024".into(),
        })
    );
    assert_eq!(resolved.backend.version, "LLVM version test");
    assert_eq!(resolved.backend.sha256, "0123456789abcdef");
}

#[test]
fn malformed_intrinsic_abi_ids_are_rejected() {
    for abi_id in ["thread_idx_x", "i1", "x0001", "i00a1"] {
        let mut record = policy();
        record.abi_id = abi_id.into();
        let error = validate_unique_overlay(&[record], 1).unwrap_err();
        assert!(error.to_string().contains("stable `iNNNN` form"));
    }
}

#[test]
fn ptx_versions_are_parsed_once_and_serialize_compatibly() {
    for (text, encoded) in [("2.0", 20), ("6.5", 65), ("10.0", 100)] {
        let version = parse_ptx_version(text, "test").unwrap();
        assert_eq!(version.encoded(), encoded);
        assert_eq!(
            serde_json::to_string(&version).unwrap(),
            format!("\"{text}\"")
        );
        assert_eq!(
            serde_json::from_str::<PtxVersion>(&format!("\"{text}\"")).unwrap(),
            version
        );
    }
    for malformed in ["6", "6.05", " 6.5", "06.5", "6.5 "] {
        assert!(parse_ptx_version(malformed, "test").is_err(), "{malformed}");
    }
}

#[test]
fn hardware_targets_are_parsed_without_losing_suffix_semantics() {
    let all = policy();
    assert_eq!(
        parse_hardware_target(&all).unwrap(),
        CatalogHardwareTarget::All
    );

    let mut minimum = policy();
    minimum.minimum_sm = Some("sm_75".into());
    assert_eq!(
        parse_hardware_target(&minimum).unwrap(),
        CatalogHardwareTarget::AnyOf {
            alternatives: vec![CatalogHardwareAlternative::MinimumSm { sm: 75 }],
        }
    );

    let mut exact = policy();
    exact.targets = "sm_120a".into();
    assert_eq!(
        parse_hardware_target(&exact).unwrap(),
        CatalogHardwareTarget::AnyOf {
            alternatives: vec![CatalogHardwareAlternative::ExactArchitecture { sm: 120 }],
        }
    );

    let mut family = policy();
    family.targets = "sm_120f".into();
    assert_eq!(
        parse_hardware_target(&family).unwrap(),
        CatalogHardwareTarget::AnyOf {
            alternatives: vec![CatalogHardwareAlternative::FamilyTarget { sm: 120 }],
        }
    );
}

#[test]
fn stage_hardware_shared_parser_preserves_canonical_evidence_language() {
    for (target, expected) in [
        ("sm_75", CatalogHardwareAlternative::MinimumSm { sm: 75 }),
        (
            "sm_120a",
            CatalogHardwareAlternative::ExactArchitecture { sm: 120 },
        ),
        (
            "compute_120f",
            CatalogHardwareAlternative::FamilyTarget { sm: 120 },
        ),
    ] {
        assert_eq!(parse_stage_hardware(target), Some(expected), "{target}");
    }
    for target in ["sm_090", "sm_1000"] {
        assert_eq!(parse_stage_hardware(target), None, "{target}");
    }
}

#[test]
fn reviewed_target_floors_equal_shared_backend_derivation() {
    for (target, floor) in [
        ("sm_120a", 87),
        ("sm_120f", 88),
        ("sm_121a", 88),
        ("sm_121f", 88),
    ] {
        let hardware = parse_stage_hardware(target).unwrap();
        assert_eq!(f8f6f4_llvm_ptx_floor(hardware).unwrap(), floor, "{target}");
    }
    for (target, floor) in [
        ("sm_100a", 86),
        ("sm_100f", 88),
        ("sm_103a", 88),
        ("sm_103f", 88),
        ("sm_110a", 90),
        ("sm_110f", 90),
        ("sm_120a", 87),
        ("sm_120f", 88),
        ("sm_121a", 88),
        ("sm_121f", 88),
    ] {
        let hardware = parse_stage_hardware(target).unwrap();
        assert_eq!(
            blackwell_ldmatrix_llvm_ptx_floor(hardware).unwrap(),
            floor,
            "{target}"
        );
    }
}

fn selector(name: &str, value: &str) -> TargetSelectorBinding {
    TargetSelectorBinding {
        name: name.into(),
        value: value.into(),
    }
}

fn target_contract(
    selectors: Vec<TargetSelectorBinding>,
    alternatives: &[(&str, &str)],
) -> TargetContract {
    TargetContract {
        selectors,
        alternatives: alternatives
            .iter()
            .map(
                |(target, minimum_ptx)| crate::model::TargetContractAlternative {
                    target: (*target).into(),
                    minimum_ptx: (*minimum_ptx).into(),
                },
            )
            .collect(),
    }
}

#[test]
fn target_contracts_keep_selector_specific_ptx_hardware_pairs() {
    let contracts = [
        target_contract(
            vec![selector("kind", "f16")],
            &[
                ("sm_100a", "8.6"),
                ("sm_101a", "8.6"),
                ("sm_103a", "8.8"),
                ("sm_110a", "9.0"),
            ],
        ),
        target_contract(
            vec![selector("kind", "i8")],
            &[("sm_100a", "8.6"), ("sm_101a", "8.6"), ("sm_110a", "9.0")],
        ),
    ];
    let full_requirement = resolve_target_contracts("tcgen05_mma", &contracts).unwrap();
    assert_eq!(full_requirement.minimum_ptx.encoded(), 86);
    let CatalogHardwareTarget::TargetMatrix {
        contracts: full_contracts,
    } = &full_requirement.hardware
    else {
        panic!("full target contracts must remain a matrix")
    };
    assert_eq!(full_contracts.len(), 2);
    assert_eq!(full_contracts[0].selectors, [selector("kind", "f16")]);
    assert_eq!(full_contracts[1].selectors, [selector("kind", "i8")]);

    let requirement =
        resolve_target_contract("tcgen05_mma", &[selector("kind", "i8")], &contracts).unwrap();

    assert_eq!(requirement.minimum_ptx.encoded(), 86);
    assert_eq!(
        requirement.hardware,
        CatalogHardwareTarget::TargetMatrix {
            contracts: vec![CatalogTargetContract {
                selectors: vec![selector("kind", "i8")],
                alternatives: vec![
                    CatalogTargetAlternative {
                        minimum_ptx: "8.6".parse().unwrap(),
                        hardware: CatalogHardwareAlternative::ExactArchitecture { sm: 100 },
                    },
                    CatalogTargetAlternative {
                        minimum_ptx: "8.6".parse().unwrap(),
                        hardware: CatalogHardwareAlternative::ExactArchitecture { sm: 101 },
                    },
                    CatalogTargetAlternative {
                        minimum_ptx: "9.0".parse().unwrap(),
                        hardware: CatalogHardwareAlternative::ExactArchitecture { sm: 110 },
                    },
                ],
            }],
        }
    );

    let mut policy = policy();
    policy.id = "tcgen05_mma".into();
    validate_candidate_target(&policy, &requirement, "sm_100a", "+ptx86").unwrap();
    assert!(validate_candidate_target(&policy, &requirement, "sm_103a", "+ptx88").is_err());
    assert!(validate_candidate_target(&policy, &requirement, "sm_110a", "+ptx88").is_err());
    validate_candidate_target(&policy, &requirement, "sm_110a", "+ptx90").unwrap();

    let libnvvm = [target_contract(
        vec![selector("kind", "i8")],
        &[("sm_100a", "8.6"), ("sm_110a", "9.0")],
    )];
    let libnvvm_requirement =
        resolve_target_contract("tcgen05_mma", &[selector("kind", "i8")], &libnvvm).unwrap();
    assert_ne!(requirement, libnvvm_requirement);
    validate_candidate_target(&policy, &requirement, "sm_101a", "+ptx86").unwrap();
    assert!(validate_candidate_target(&policy, &libnvvm_requirement, "sm_101a", "+ptx86").is_err());
}

#[test]
fn target_contract_selection_and_shape_fail_closed() {
    resolve_target_contract(
        "tcgen05_mma",
        &[selector("kind", "f16")],
        &[target_contract(
            vec![selector("kind", "f16")],
            &[("sm_100a", "8.6"), ("sm_100f", "8.8"), ("sm_103a", "8.8")],
        )],
    )
    .unwrap();

    let valid = target_contract(
        vec![selector("kind", "f16"), selector("scale_d", "false")],
        &[("sm_100a", "8.6"), ("sm_103a", "8.8")],
    );
    assert!(
        resolve_target_contract(
            "tcgen05_mma",
            &[selector("kind", "i8"), selector("scale_d", "false")],
            std::slice::from_ref(&valid),
        )
        .unwrap_err()
        .to_string()
        .contains("exactly one reviewed contract")
    );

    for invalid in [
        target_contract(
            vec![selector("scale_d", "false"), selector("kind", "f16")],
            &[("sm_100a", "8.6")],
        ),
        target_contract(vec![selector("kind", "F16")], &[("sm_100a", "8.6")]),
        target_contract(vec![selector("kind_", "f16")], &[("sm_100a", "8.6")]),
        target_contract(
            vec![selector("kind", "f16")],
            &[("sm_103a", "8.8"), ("sm_100a", "8.6")],
        ),
        target_contract(
            vec![selector("kind", "f16")],
            &[("sm_100+", "8.6"), ("sm_103a", "8.8")],
        ),
        target_contract(vec![selector("kind", "f16")], &[("sm_100a", "8.60")]),
    ] {
        assert!(
            resolve_target_contract(
                "tcgen05_mma",
                &invalid.selectors,
                std::slice::from_ref(&invalid)
            )
            .is_err(),
            "{invalid:?}"
        );
    }
}

#[test]
fn legacy_target_requirement_json_is_unchanged() {
    let requirement = CatalogTargetRequirement {
        minimum_ptx: "8.6".parse().unwrap(),
        hardware: CatalogHardwareTarget::AnyOf {
            alternatives: vec![CatalogHardwareAlternative::ExactArchitecture { sm: 100 }],
        },
    };
    assert_eq!(
        serde_json::to_value(requirement).unwrap(),
        serde_json::json!({
            "minimum_ptx": "8.6",
            "hardware": {
                "kind": "any_of",
                "alternatives": [{ "kind": "exact_architecture", "sm": 100 }]
            }
        })
    );
}

#[test]
fn malformed_or_conflicting_hardware_targets_are_rejected() {
    for malformed in [
        "sm_120",
        "sm_120af",
        "sm_120A",
        "sm_0120a",
        "sm_0a",
        "sm_120+",
        "compute_120a",
        "all ",
    ] {
        let mut record = policy();
        record.targets = malformed.into();
        assert!(parse_hardware_target(&record).is_err(), "{malformed}");
    }

    let mut suffixed_minimum = policy();
    suffixed_minimum.minimum_sm = Some("sm_90a".into());
    assert!(parse_hardware_target(&suffixed_minimum).is_err());

    for target in ["sm_120a", "sm_120f"] {
        let mut conflicting = policy();
        conflicting.targets = target.into();
        conflicting.minimum_sm = Some("sm_120".into());
        let error = parse_hardware_target(&conflicting).unwrap_err().to_string();
        assert!(error.contains("cannot be combined"), "{error}");
    }
}

#[test]
fn exact_inline_ptx_routes_can_inherit_exact_or_family_targets() {
    for target in ["sm_120a", "sm_120f"] {
        let mut record = policy();
        record.minimum_ptx = "8.7".into();
        record.targets = target.into();
        record.backend_lowerings = [IntrinsicBackend::LlvmNvptx, IntrinsicBackend::LibNvvm]
            .into_iter()
            .map(|backend| crate::model::OverlayBackendLowering {
                backend,
                mechanism: BackendLoweringMechanism::InlinePtx,
                evidence_profile: "test".into(),
                targets: None,
                minimum_ptx: Some("8.7".into()),
                minimum_sm: None,
            })
            .collect();

        ensure_exact_inline_ptx_backends(
            &record,
            [
                (IntrinsicBackend::LlvmNvptx, "8.7", None),
                (IntrinsicBackend::LibNvvm, "8.7", None),
            ],
            "test",
        )
        .unwrap();
        for lowering in &record.backend_lowerings {
            assert_eq!(
                backend_target_requirement(&record, lowering)
                    .unwrap()
                    .hardware,
                parse_hardware_target(&record).unwrap()
            );
        }

        record.backend_lowerings[0].minimum_sm = Some("sm_120".into());
        assert!(
            ensure_exact_inline_ptx_backends(
                &record,
                [
                    (IntrinsicBackend::LlvmNvptx, "8.7", None),
                    (IntrinsicBackend::LibNvvm, "8.7", None),
                ],
                "test",
            )
            .is_err()
        );
    }
}

#[test]
fn backend_route_target_override_is_exact_and_does_not_inherit_the_record_sm_floor() {
    let mut record = policy();
    record.minimum_ptx = "8.6".into();
    record.minimum_sm = Some("sm_75".into());
    record.targets = "all".into();
    let lowering = crate::model::OverlayBackendLowering {
        backend: IntrinsicBackend::LibNvvm,
        mechanism: BackendLoweringMechanism::InlinePtx,
        evidence_profile: "test".into(),
        targets: Some("sm_100a|sm_120a".into()),
        minimum_ptx: None,
        minimum_sm: None,
    };

    assert_eq!(
        backend_target_requirement(&record, &lowering).unwrap(),
        CatalogTargetRequirement {
            minimum_ptx: parse_ptx_version("8.6", &record.id).unwrap(),
            hardware: CatalogHardwareTarget::AnyOf {
                alternatives: vec![
                    CatalogHardwareAlternative::ExactArchitecture { sm: 100 },
                    CatalogHardwareAlternative::ExactArchitecture { sm: 120 },
                ],
            },
        }
    );
}

#[test]
fn abi_ledger_requires_exact_active_identity() {
    let record = policy();
    let frozen_entry = ledger_entry(&record);
    validate_abi_ledger(
        &overlay_file(vec![record.clone()]),
        &ledger(vec![frozen_entry.clone()]),
    )
    .unwrap();

    let mut reassigned = record.clone();
    reassigned.id = "different_catalog_id".into();
    let error = validate_abi_ledger(
        &overlay_file(vec![reassigned]),
        &ledger(vec![frozen_entry.clone()]),
    )
    .unwrap_err();
    assert!(error.to_string().contains("catalog ID mismatch"));

    let mut reassigned = record.clone();
    reassigned.operation_key = "launch.block_index.x".into();
    let error = validate_abi_ledger(
        &overlay_file(vec![reassigned]),
        &ledger(vec![frozen_entry.clone()]),
    )
    .unwrap_err();
    assert!(error.to_string().contains("operation key mismatch"));

    for mutate in [
        |record: &mut OverlayIntrinsic| record.safe = false,
        |record: &mut OverlayIntrinsic| record.rust_arguments.push("u32".into()),
        |record: &mut OverlayIntrinsic| record.rust_result = "u64".into(),
    ] {
        let mut changed_signature = record.clone();
        mutate(&mut changed_signature);
        let error = validate_abi_ledger(
            &overlay_file(vec![changed_signature]),
            &ledger(vec![frozen_entry.clone()]),
        )
        .unwrap_err();
        assert!(error.to_string().contains("raw Rust signature mismatch"));
    }
}

#[test]
fn generated_abi_binding_uses_catalog_identity_not_axis_position() {
    let mut record = policy();
    let mut frozen = ledger_entry(&record);
    frozen.abi_id = "i9001".into();
    record.abi_id.clear();
    let mut overlay = overlay_file(vec![record]);

    bind_generated_abi_ids(&mut overlay, &ledger(vec![frozen])).unwrap();

    assert_eq!(overlay.intrinsics[0].abi_id, "i9001");
}

#[test]
fn generated_abi_binding_rejects_missing_tombstoned_or_ambiguous_identity() {
    let record = policy();
    let mut unbound = record.clone();
    unbound.abi_id.clear();

    let error = bind_generated_abi_ids(
        &mut overlay_file(vec![unbound.clone()]),
        &ledger(vec![ledger_entry(&distinct_policy())]),
    )
    .unwrap_err();
    assert!(error.to_string().contains("has no ABI ledger entry"));

    let mut tombstone = ledger_entry(&record);
    tombstone.status = "tombstone".into();
    let error = bind_generated_abi_ids(
        &mut overlay_file(vec![unbound.clone()]),
        &ledger(vec![tombstone]),
    )
    .unwrap_err();
    assert!(error.to_string().contains("non-active ABI ledger entry"));

    let first = ledger_entry(&record);
    let mut duplicate = first.clone();
    duplicate.abi_id = "i9002".into();
    let error = bind_generated_abi_ids(
        &mut overlay_file(vec![unbound]),
        &ledger(vec![first, duplicate]),
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("duplicate ABI ledger catalog ID")
    );
}

#[test]
fn generated_abi_binding_checks_derived_operation_and_raw_signature() {
    let record = policy();
    let mut unbound = record.clone();
    unbound.abi_id.clear();

    let mut wrong_operation = ledger_entry(&record);
    wrong_operation.operation_key = "launch.block_index.x".into();
    let error = bind_generated_abi_ids(
        &mut overlay_file(vec![unbound.clone()]),
        &ledger(vec![wrong_operation]),
    )
    .unwrap_err();
    assert!(error.to_string().contains("operation key mismatch"));

    let mut wrong_signature = ledger_entry(&record);
    wrong_signature
        .raw_rust_signature
        .arguments
        .push("u32".into());
    let error = bind_generated_abi_ids(
        &mut overlay_file(vec![unbound]),
        &ledger(vec![wrong_signature]),
    )
    .unwrap_err();
    assert!(error.to_string().contains("raw Rust signature mismatch"));
}

#[test]
fn abi_ledger_does_not_freeze_public_or_backend_implementation_details() {
    let record = policy();
    let frozen_entry = ledger_entry(&record);
    let mut evolved = record.clone();
    evolved.rust_module = "coordinates".into();
    evolved.rust_name = "thread_x".into();
    evolved.public_rust_path = "cuda_intrinsics::coordinates::thread_x".into();
    evolved.llvm_symbol = Some("llvm.nvvm.backend.v2.tid.x".into());
    evolved.llvm_arguments = vec!["i8".into()];
    evolved.llvm_results = vec!["i64".into()];
    evolved.dialect_op_type = "ReadThreadIndexXOpV2".into();
    evolved.dialect_op_name = "nvvm.read_thread_index_x_v2".into();
    evolved.lowering = "backend_v2_adapter".into();

    validate_abi_ledger(&overlay_file(vec![evolved]), &ledger(vec![frozen_entry])).unwrap();
}

#[test]
fn tombstoned_or_unlisted_abi_ids_cannot_reappear() {
    let record = policy();
    let mut tombstone = ledger_entry(&record);
    tombstone.status = "tombstone".into();
    let error = validate_abi_ledger(
        &overlay_file(vec![record.clone()]),
        &ledger(vec![tombstone]),
    )
    .unwrap_err();
    assert!(error.to_string().contains("cannot reappear"));

    let error = validate_abi_ledger(&overlay_file(vec![record]), &ledger(vec![])).unwrap_err();
    assert!(error.to_string().contains("contains no entries"));
}

#[test]
fn every_active_ledger_entry_requires_an_overlay_record() {
    let record = policy();
    let error = validate_abi_ledger(&overlay_file(vec![]), &ledger(vec![ledger_entry(&record)]))
        .unwrap_err();
    assert!(error.to_string().contains("has no overlay record"));
}

#[test]
fn candidate_targets_are_canonical_and_satisfy_every_floor() {
    let policy = policy();
    let requirement = CatalogTargetRequirement {
        minimum_ptx: "7.0".parse().unwrap(),
        hardware: CatalogHardwareTarget::AnyOf {
            alternatives: vec![CatalogHardwareAlternative::MinimumSm { sm: 80 }],
        },
    };
    validate_candidate_target(&policy, &requirement, "sm_80", "+ptx70").unwrap();
    validate_candidate_target(&policy, &requirement, "sm_90a", "+ptx86").unwrap();
    assert!(
        validate_candidate_target(&policy, &requirement, "sm_75", "+ptx70")
            .unwrap_err()
            .to_string()
            .contains("hardware requirement")
    );
    assert!(
        validate_candidate_target(&policy, &requirement, "sm_80", "+ptx69")
            .unwrap_err()
            .to_string()
            .contains("PTX floor")
    );
    for malformed in ["compute_80", "sm_080", "sm_80x"] {
        assert!(
            validate_candidate_target(&policy, &requirement, malformed, "+ptx70").is_err(),
            "{malformed}"
        );
    }
    for malformed in ["ptx70", "+ptx7", "+ptx070"] {
        assert!(
            validate_candidate_target(&policy, &requirement, "sm_80", malformed).is_err(),
            "{malformed}"
        );
    }

    let exact = CatalogTargetRequirement {
        minimum_ptx: "8.7".parse().unwrap(),
        hardware: CatalogHardwareTarget::AnyOf {
            alternatives: vec![CatalogHardwareAlternative::ExactArchitecture { sm: 120 }],
        },
    };
    validate_candidate_target(&policy, &exact, "sm_120a", "+ptx87").unwrap();
    assert!(validate_candidate_target(&policy, &exact, "sm_120a", "+ptx86").is_err());
    assert!(validate_candidate_target(&policy, &exact, "sm_120", "+ptx87").is_err());
    assert!(validate_candidate_target(&policy, &exact, "sm_120f", "+ptx87").is_err());

    let family = CatalogTargetRequirement {
        minimum_ptx: "8.7".parse().unwrap(),
        hardware: CatalogHardwareTarget::AnyOf {
            alternatives: vec![CatalogHardwareAlternative::FamilyTarget { sm: 120 }],
        },
    };
    validate_candidate_target(&policy, &family, "sm_120f", "+ptx87").unwrap();
    assert!(validate_candidate_target(&policy, &family, "sm_120a", "+ptx87").is_err());
}
