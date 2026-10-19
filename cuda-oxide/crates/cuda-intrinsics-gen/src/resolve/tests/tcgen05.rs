/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    CatalogHardwareAlternative, CatalogHardwareTarget, ImportedFile, OverlayIntrinsic,
    OverlayShardFile, RuntimeValidation, Tcgen05Adapter, Tcgen05MmaBUsage,
    Tcgen05MmaFixedSelectors, Tcgen05MmaForm, Tcgen05MmaKind, Tcgen05MmaSelectorLayout,
    Tcgen05Operation, Tcgen05SourceContract,
};
use crate::ptx::OperandPattern;
use crate::util::read_json;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use super::fixtures::*;
use crate::resolve::families::*;
use crate::resolve::guards::*;
use crate::resolve::overlay::*;
use crate::resolve::targets::*;

#[test]
fn compact_tcgen05_admission_matches_llvm_and_fails_closed() {
    let records = expand_tcgen05_admission(&test_tcgen05_admission()).unwrap();
    assert_eq!(records.len(), 27);
    assert_eq!(
        records
            .iter()
            .map(|record| (record.abi_id.as_str(), record.id.as_str()))
            .collect::<Vec<_>>(),
        [
            ("i0343", "tcgen05_alloc"),
            ("i0344", "tcgen05_dealloc"),
            ("i0345", "tcgen05_relinquish_alloc_permit"),
            ("i0346", "tcgen05_fence_before_thread_sync"),
            ("i0347", "tcgen05_fence_after_thread_sync"),
            ("i0348", "tcgen05_commit"),
            ("i0349", "tcgen05_commit_shared_cluster"),
            ("i0350", "tcgen05_mma_ws_f16"),
            ("i0351", "tcgen05_mma_f16"),
            ("i0352", "tcgen05_mma_ws_bf16"),
            ("i0353", "tcgen05_mma_ws_tf32"),
            ("i0354", "tcgen05_cp_smem_to_tmem"),
            ("i0355", "tcgen05_ld_16x256b_x8_pure"),
            ("i0356", "tcgen05_ld_16x256b_pure"),
            ("i0357", "tcgen05_load_wait"),
            ("i0358", "tcgen05_store_wait"),
            ("i0359", "tcgen05_alloc_cg2"),
            ("i0360", "tcgen05_dealloc_cg2"),
            ("i0361", "tcgen05_relinquish_alloc_permit_cg2"),
            ("i0362", "tcgen05_mma_f16_cg2"),
            ("i0363", "tcgen05_commit_cg2"),
            ("i0364", "tcgen05_commit_shared_cluster_cg2"),
            ("i0365", "tcgen05_commit_multicast_cg2"),
            ("i0366", "tcgen05_cp_smem_to_tmem_cg2"),
            ("i0760", "tcgen05_commit_multicast"),
            ("i0761", "tcgen05_shift_down"),
            ("i0762", "tcgen05_shift_down_cg2"),
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
        assert_tcgen05_backend_target_split(record);
        assert_eq!(
            record.execution_scope,
            record.tcgen05.as_ref().unwrap().operation.execution_scope()
        );
    }

    let bf16 = records
        .iter()
        .find(|record| record.id == "tcgen05_mma_ws_bf16")
        .unwrap();
    assert!(bf16.expected_ptx.modifiers.contains(&"kind::f16".into()));
    assert!(!bf16.expected_ptx.modifiers.contains(&"kind::bf16".into()));
    assert_eq!(
        bf16.tcgen05.as_ref().unwrap().source_contract,
        Tcgen05SourceContract::TablegenSelectionChangesPtx
    );

    let mma_f16 = records
        .iter()
        .find(|record| record.id == "tcgen05_mma_f16")
        .unwrap();
    assert!(mma_f16.expected_ptx.matches(
            "tcgen05.mma.cta_group::1.kind::f16 [%r1], %rd1, %rd2, %r2, {%z, %z, %z, %z}, %enable_pred;"
        ).unwrap());
    assert!(!mma_f16.expected_ptx.matches(
            "tcgen05.mma.cta_group::1.kind::f16 [%r1], %rd1, %rd2, %r2, {%clock64, %z, %z, %z}, %enable_pred;"
        ).unwrap());
    assert!(!mma_f16.expected_ptx.matches(
            "tcgen05.mma.cta_group::1.kind::f16 [%r1], %rd1, %rd2, %r2, {%z, %z, %z, %z}, %other_pred;"
        ).unwrap());

    let mma_f16_cg2 = records
        .iter()
        .find(|record| record.id == "tcgen05_mma_f16_cg2")
        .unwrap();
    assert!(mma_f16_cg2.expected_ptx.matches(
            "tcgen05.mma.cta_group::2.kind::f16 [%r1], %rd1, %rd2, %r2, {%z, %z, %z, %z, %z, %z, %z, %z}, %enable_pred;"
        ).unwrap());
    assert!(
        !records
            .iter()
            .any(|record| { record.rust_name == "tcgen05_mma_ws_f16_with_collector" })
    );
    assert_eq!(
        parse_hardware_target(&records[0]).unwrap(),
        CatalogHardwareTarget::AnyOf {
            alternatives: vec![
                CatalogHardwareAlternative::ExactArchitecture { sm: 100 },
                CatalogHardwareAlternative::ExactArchitecture { sm: 101 },
                CatalogHardwareAlternative::ExactArchitecture { sm: 103 },
                CatalogHardwareAlternative::ExactArchitecture { sm: 110 },
            ],
        }
    );

    let legacy_records =
        expand_tcgen05_admission(&without_tcgen05_control(test_tcgen05_admission())).unwrap();
    assert_eq!(legacy_records.len(), 24);
    assert!(legacy_records.iter().all(|record| {
        record
            .backend_lowerings
            .iter()
            .map(|lowering| lowering.evidence_profile.as_str())
            .eq(["llvm-tcgen05-test", "libnvvm-tcgen05-test"])
    }));

    let multicast = records
        .iter()
        .find(|record| record.id == "tcgen05_commit_multicast")
        .unwrap();
    assert_eq!(multicast.abi_id, "i0760");
    assert_eq!(multicast.rust_arguments, ["*mut u64", "u16"]);
    assert_eq!(multicast.llvm_arguments, ["shared_ptr", "i16"]);
    assert_eq!(multicast.execution_scope, "thread");
    assert_eq!(
        multicast.backend_lowerings[0].evidence_profile,
        "llvm-tcgen05-control-test"
    );
    assert_eq!(
        multicast.backend_lowerings[1].evidence_profile,
        "libnvvm-tcgen05-control-test"
    );

    for (id, group) in [
        ("tcgen05_shift_down", "cta_group::1"),
        ("tcgen05_shift_down_cg2", "cta_group::2"),
    ] {
        let shift = records.iter().find(|record| record.id == id).unwrap();
        assert_eq!(shift.rust_arguments, ["u32"]);
        assert_eq!(shift.llvm_arguments, ["tmem_ptr"]);
        assert_eq!(shift.execution_scope, "thread");
        assert_eq!(shift.memory, "read_write");
        assert_eq!(shift.expected_ptx.modifiers, ["shift", group, "down"]);
        assert_eq!(shift.expected_ptx.operands, [OperandPattern::Address]);
    }

    let mut missing = test_tcgen05_admission();
    missing.variants.pop();
    assert!(expand_tcgen05_admission(&missing).is_err());

    let mut reordered = test_tcgen05_admission();
    reordered.variants.swap(0, 1);
    assert!(expand_tcgen05_admission(&reordered).is_err());

    let mut wrong_abi = test_tcgen05_admission();
    wrong_abi.variants[0].abi_id = "i9999".into();
    assert!(expand_tcgen05_admission(&wrong_abi).is_err());

    let mut executed = test_tcgen05_admission();
    executed.runtime_validation = RuntimeValidation::Executed;
    assert!(expand_tcgen05_admission(&executed).is_err());

    let mut missing_control_evidence = test_tcgen05_admission();
    missing_control_evidence.control_llvm_evidence_profile = None;
    assert!(expand_tcgen05_admission(&missing_control_evidence).is_err());

    let mut partial_control = without_tcgen05_control(test_tcgen05_admission());
    partial_control
        .variants
        .push(test_tcgen05_admission().variants[24].clone());
    assert!(expand_tcgen05_admission(&partial_control).is_err());

    let shift = records
        .iter()
        .find(|record| record.id == "tcgen05_shift_down")
        .unwrap();
    let mut wrong_shift_declaration = declarations[shift.source_record.as_deref().unwrap()].clone();
    wrong_shift_declaration.selections[0].predicates =
        vec!["Subtarget->hasTcgen05InstSupport()".into()];
    assert!(validate_imported_policy(shift, &wrong_shift_declaration).is_err());

    let declaration = declarations[records[0].source_record.as_deref().unwrap()];
    let mut wrong_adapter = records[0].clone();
    wrong_adapter.tcgen05.as_mut().unwrap().adapter = Tcgen05Adapter::NoOperands;
    assert!(validate_imported_policy(&wrong_adapter, declaration).is_err());

    let mut changed_declaration = declaration.clone();
    changed_declaration.properties.pop();
    assert!(validate_imported_policy(&records[0], &changed_declaration).is_err());

    let mut broadened_libnvvm = records[0].clone();
    broadened_libnvvm.backend_lowerings[1].targets = None;
    assert!(validate_imported_policy(&broadened_libnvvm, declaration).is_err());

    let mut wrong_alloc_scope = records[0].clone();
    wrong_alloc_scope.execution_scope = "thread".into();
    assert!(validate_imported_policy(&wrong_alloc_scope, declaration).is_err());

    let pure_load = &records[12];
    let pure_declaration = declarations[pure_load.source_record.as_deref().unwrap()];
    let mut wrong_pure_load_scope = pure_load.clone();
    wrong_pure_load_scope.execution_scope = "thread".into();
    assert!(validate_imported_policy(&wrong_pure_load_scope, pure_declaration).is_err());
}

#[test]
fn compact_tcgen05_mma_admission_closes_sources_selectors_and_targets() {
    let records = expand_tcgen05_admission(&test_tcgen05_mma_admission()).unwrap();
    assert_eq!(records.len(), 51);
    let mma = &records[27..];
    assert_eq!(mma.len(), 24);
    assert_eq!(
        mma.iter()
            .map(|record| (record.abi_id.as_str(), record.id.as_str()))
            .collect::<Vec<_>>(),
        [
            ("i0763", "tcgen05_mma_shared"),
            ("i0764", "tcgen05_mma_tensor"),
            ("i0765", "tcgen05_mma_tensor_ashift"),
            ("i0766", "tcgen05_mma_sp_shared"),
            ("i0767", "tcgen05_mma_sp_tensor"),
            ("i0768", "tcgen05_mma_sp_tensor_ashift"),
            ("i0769", "tcgen05_mma_ws_shared"),
            ("i0770", "tcgen05_mma_ws_shared_zero_col_mask"),
            ("i0771", "tcgen05_mma_ws_sp_shared"),
            ("i0772", "tcgen05_mma_ws_sp_shared_zero_col_mask"),
            ("i0773", "tcgen05_mma_ws_sp_tensor"),
            ("i0774", "tcgen05_mma_ws_sp_tensor_zero_col_mask"),
            ("i0775", "tcgen05_mma_ws_tensor"),
            ("i0776", "tcgen05_mma_ws_tensor_zero_col_mask"),
            ("i0777", "tcgen05_mma_ws_e4m3"),
            ("i0778", "tcgen05_mma_ws_e5m2"),
            ("i0779", "tcgen05_mma_ws_e2m3"),
            ("i0780", "tcgen05_mma_ws_e3m2"),
            ("i0781", "tcgen05_mma_ws_e2m1"),
            ("i1011", "tcgen05_mma_e4m3"),
            ("i1012", "tcgen05_mma_e5m2"),
            ("i1013", "tcgen05_mma_e2m3"),
            ("i1014", "tcgen05_mma_e3m2"),
            ("i1015", "tcgen05_mma_e2m1"),
        ]
    );
    assert!(mma.iter().all(|record| {
        record.dialect_op_type == TCGEN05_MMA_DIALECT_OP_TYPE
            && record.dialect_op_name == TCGEN05_MMA_DIALECT_OP_NAME
    }));
    assert_eq!(
        [
            Tcgen05MmaBUsage::Discard,
            Tcgen05MmaBUsage::LastUse,
            Tcgen05MmaBUsage::Fill,
            Tcgen05MmaBUsage::Use,
        ]
        .map(Tcgen05MmaBUsage::selector_value),
        [0, 1, 2, 3]
    );

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let imported: ImportedFile = read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
    let declarations = imported
        .intrinsics
        .iter()
        .map(|record| (record.source_record.as_str(), record))
        .collect::<BTreeMap<_, _>>();
    for record in mma {
        let declaration = declarations[record.source_record.as_deref().unwrap()];
        validate_imported_policy(record, declaration).unwrap();
        assert_eq!(record.memory, "read_write");
        assert!(record.convergent);
        assert_eq!(record.execution_scope, "thread");
        assert!(!record.safe);
        assert_eq!(
            record
                .expected_ptx
                .operands
                .iter()
                .filter(|operand| {
                    **operand
                        == (OperandPattern::Exact {
                            value: "%enable_pred".into(),
                        })
                })
                .count(),
            1
        );
    }

    let dense = &mma[0];
    assert!(dense.expected_ptx.matches(
            "tcgen05.mma.cta_group::1.kind::f16.collector::a::discard [%r1], %rd1, %rd2, %r2, %enable_pred;"
        ).unwrap());
    assert!(!dense.expected_ptx.matches(
            "tcgen05.mma.cta_group::1.kind::f16.collector::a::discard [%r1], %rd1, %rd2, %r2, %clock64;"
        ).unwrap());
    assert!(mma[4].expected_ptx.matches(
            "tcgen05.mma.sp.cta_group::1.kind::f16.collector::a::discard [%r1], [%r2], %rd1, [%r3], %r4, %enable_pred;"
        ).unwrap());
    assert!(mma[7].expected_ptx.matches(
            "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::discard [%r1], %rd1, %rd2, %r2, %enable_pred, %rd3;"
        ).unwrap());
    assert!(mma[14].expected_ptx.matches(
            "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::discard [%r1], [%r2], %rd1, %r3, %enable_pred;"
        ).unwrap());
    assert!(mma[19].expected_ptx.matches(
            "tcgen05.mma.cta_group::1.kind::f8f6f4.collector::a::discard [%r1], %rd1, %rd2, %r2, %enable_pred;"
        ).unwrap());

    let canonical = &mma[..14];
    assert_eq!(
        canonical
            .iter()
            .map(|record| record.source_record.as_deref().unwrap())
            .collect::<BTreeSet<_>>()
            .len(),
        14
    );
    let selected_count = |record: &OverlayIntrinsic| {
        declarations[record.source_record.as_deref().unwrap()]
            .selections
            .iter()
            .filter(|selection| selection_matches_policy(record, selection).unwrap())
            .count()
    };
    let newly_matched = canonical
        .iter()
        .filter(|record| record.id != "tcgen05_mma_ws_tensor")
        .map(selected_count)
        .sum::<usize>();
    assert_eq!(newly_matched, 608);
    assert_eq!(selected_count(&canonical[12]), 64);
    assert_eq!(canonical.iter().map(selected_count).sum::<usize>(), 672);
    assert_eq!(mma[14..].iter().map(selected_count).sum::<usize>(), 10);
    assert_eq!(
        [2, 5]
            .into_iter()
            .map(|index| {
                declarations[canonical[index].source_record.as_deref().unwrap()]
                    .selections
                    .len()
                    - selected_count(&canonical[index])
            })
            .sum::<usize>(),
        32
    );

    for record in canonical {
        assert_eq!(
            record.compatibility_rust_paths,
            [format!("cuda_device::tcgen05::__{}", record.id)]
        );
        assert_eq!(
            &record.rust_arguments[record.rust_arguments.len() - 3..],
            ["u32", "u32", "u32"]
        );
        let contract = record.tcgen05.as_ref().unwrap().mma.as_ref().unwrap();
        let CatalogHardwareTarget::TargetMatrix { contracts } = &contract.llvm_target.hardware
        else {
            panic!("generic LLVM tcgen05 MMA target must be a matrix")
        };
        assert_eq!(contracts.len(), 4);
        assert_eq!(
            contracts
                .iter()
                .map(|contract| contract.alternatives.len())
                .collect::<Vec<_>>(),
            [8, 8, 3, 8]
        );
        let CatalogHardwareTarget::TargetMatrix { contracts } = &contract.libnvvm_target.hardware
        else {
            panic!("generic libNVVM tcgen05 MMA target must be a matrix")
        };
        assert_eq!(contracts.len(), 4);
        assert_eq!(
            contracts
                .iter()
                .map(|contract| contract.alternatives.len())
                .collect::<Vec<_>>(),
            [6, 6, 2, 6]
        );
    }
    for record in &mma[14..19] {
        assert_eq!(
            record.compatibility_rust_paths,
            [format!("cuda_device::tcgen05::{}", record.id)]
        );
        assert_eq!(
            record.rust_arguments,
            ["u32", "u32", "u64", "u64", "u32", "bool"]
        );
        assert_eq!(record.dialect_operands, ["i32", "i32", "i64", "i32", "i1"]);
        let contract = record.tcgen05.as_ref().unwrap().mma.as_ref().unwrap();
        assert_eq!(
            contract.fixed_selectors,
            Some(Tcgen05MmaFixedSelectors {
                kind: Tcgen05MmaKind::F8f6f4,
                b_buffer: 0,
                b_usage: Tcgen05MmaBUsage::Discard,
            })
        );
        let CatalogHardwareTarget::TargetMatrix { contracts } = &contract.llvm_target.hardware
        else {
            panic!("fixed LLVM tcgen05 MMA target must be a matrix")
        };
        assert_eq!(contracts.len(), 1);
        assert_eq!(contracts[0].selectors[0].value, "f8f6f4");
    }
    for record in &mma[19..] {
        assert_eq!(
            record.compatibility_rust_paths,
            [format!("cuda_device::tcgen05::{}", record.id)]
        );
        assert_eq!(record.rust_arguments, ["u32", "u64", "u64", "u32", "bool"]);
        assert_eq!(record.dialect_operands, ["i32", "i64", "i64", "i32", "i1"]);
        let contract = record.tcgen05.as_ref().unwrap().mma.as_ref().unwrap();
        assert_eq!(contract.form, Tcgen05MmaForm::Shared);
        assert_eq!(
            contract.fixed_selectors,
            Some(Tcgen05MmaFixedSelectors {
                kind: Tcgen05MmaKind::F8f6f4,
                b_buffer: 0,
                b_usage: Tcgen05MmaBUsage::Discard,
            })
        );
        assert_eq!(
            record.tcgen05.as_ref().unwrap().adapter,
            Tcgen05Adapter::MmaDirectSelectors
        );
        let CatalogHardwareTarget::TargetMatrix { contracts } = &contract.llvm_target.hardware
        else {
            panic!("fixed LLVM tcgen05 MMA target must be a matrix")
        };
        assert_eq!(contracts.len(), 1);
        assert_eq!(contracts[0].selectors[0].value, "f8f6f4");
    }

    for legacy in [
        "tcgen05_mma_ws_f16",
        "tcgen05_mma_ws_bf16",
        "tcgen05_mma_ws_tf32",
    ] {
        let old = &records[..27]
            .iter()
            .find(|record| record.id == legacy)
            .unwrap();
        assert!(old.tcgen05.as_ref().unwrap().mma.is_none());
        assert!(old.abi_id.starts_with("i035"));
        assert_eq!(
            old.expected_ptx.operands.last(),
            Some(&OperandPattern::Exact {
                value: "%enable_pred".into(),
            })
        );
    }
    let legacy_f16 = records
        .iter()
        .find(|record| record.id == "tcgen05_mma_ws_f16")
        .unwrap();
    assert!(
        legacy_f16
            .expected_ptx
            .matches("tcgen05.mma.ws.cta_group::1.kind::f16 [%r1], [%r2], %rd1, %r3, %enable_pred;")
            .unwrap()
    );
    assert!(
        !legacy_f16
            .expected_ptx
            .matches("tcgen05.mma.ws.cta_group::1.kind::f16 [%r1], [%r2], %rd1, %r3, %other_pred;")
            .unwrap()
    );
}

#[test]
fn compact_tcgen05_mma_admission_fails_closed_on_drift() {
    let mut missing = test_tcgen05_mma_admission();
    missing.mma_variants.pop();
    assert!(expand_tcgen05_admission(&missing).is_err());

    let mut reordered = test_tcgen05_mma_admission();
    reordered.mma_variants.swap(0, 1);
    assert!(expand_tcgen05_admission(&reordered).is_err());

    let mut non_contiguous_abi = test_tcgen05_mma_admission();
    non_contiguous_abi.mma_variants[0].abi_id = "i9999".into();
    let records = expand_tcgen05_admission(&non_contiguous_abi).unwrap();
    assert!(records.iter().any(|record| record.abi_id == "i9999"));

    let mut missing_target = test_tcgen05_mma_admission();
    missing_target.mma_llvm_target_contracts[0]
        .alternatives
        .pop();
    assert!(expand_tcgen05_admission(&missing_target).is_err());

    let mut broadened_i8 = test_tcgen05_mma_admission();
    broadened_i8.mma_libnvvm_target_contracts[2]
        .alternatives
        .insert(
            1,
            crate::model::TargetContractAlternative {
                target: "sm_103a".into(),
                minimum_ptx: "8.8".into(),
            },
        );
    assert!(expand_tcgen05_admission(&broadened_i8).is_err());

    let records = expand_tcgen05_admission(&test_tcgen05_mma_admission()).unwrap();
    let record = records
        .iter()
        .find(|record| record.id == "tcgen05_mma_tensor_ashift")
        .unwrap();
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let imported: ImportedFile = read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
    let declaration = imported
        .intrinsics
        .iter()
        .find(|declaration| {
            Some(declaration.source_record.as_str()) == record.source_record.as_deref()
        })
        .unwrap();

    let mut wrong_layout = record.clone();
    let mma = wrong_layout.tcgen05.as_mut().unwrap().mma.as_mut().unwrap();
    mma.selector_layout = Tcgen05MmaSelectorLayout::Base {
        kind_argument: 6,
        cta_group_argument: 5,
        collector_a_argument: 7,
        collector_a_upper_exclusive: 2,
    };
    assert!(validate_imported_policy(&wrong_layout, declaration).is_err());

    let mut wrong_properties = declaration.clone();
    wrong_properties.properties.pop();
    assert!(validate_imported_policy(record, &wrong_properties).is_err());

    let mut wrong_range = declaration.clone();
    let range = wrong_range
        .properties
        .iter_mut()
        .find(|property| property.as_str() == "Range<arg7,0,2>")
        .unwrap();
    *range = "Range<arg7,0,4>".into();
    assert!(validate_imported_policy(record, &wrong_range).is_err());

    let mut wrong_predicate = declaration.clone();
    wrong_predicate.selections[0].predicates = vec!["hasSM100a".into()];
    assert!(validate_imported_policy(record, &wrong_predicate).is_err());

    let mut wrong_memory = record.clone();
    wrong_memory.memory = "write".into();
    assert!(validate_imported_policy(&wrong_memory, declaration).is_err());

    let mut wrong_convergence = record.clone();
    wrong_convergence.convergent = false;
    assert!(validate_imported_policy(&wrong_convergence, declaration).is_err());

    let mut split_dialect_op = record.clone();
    split_dialect_op.dialect_op_type = "Tcgen05MmaTensorAshiftOp".into();
    split_dialect_op.dialect_op_name = "nvvm.tcgen05_mma_tensor_ashift".into();
    assert!(validate_imported_policy(&split_dialect_op, declaration).is_err());
}

#[test]
fn compact_tcgen05_copy_admission_matches_all_llvm_records_and_fails_closed() {
    let records = expand_tcgen05_admission(&test_tcgen05_cp_admission()).unwrap();
    assert_eq!(records.len(), 61);
    let copies = &records[27..];
    assert_eq!(copies.len(), 34);
    assert_eq!(
        (copies[0].abi_id.as_str(), copies[0].id.as_str()),
        ("i0578", "tcgen05_cp_128x128b_b4x16_p64")
    );
    assert_eq!(
        (copies[33].abi_id.as_str(), copies[33].id.as_str()),
        ("i0611", "tcgen05_cp_64x128b_warpx2_02_13_cg2")
    );

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let imported: ImportedFile = read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
    let declarations = imported
        .intrinsics
        .iter()
        .map(|record| (record.source_record.as_str(), record))
        .collect::<BTreeMap<_, _>>();
    for record in copies {
        let declaration = declarations[record.source_record.as_deref().unwrap()];
        validate_imported_policy(record, declaration).unwrap();
        assert_tcgen05_backend_target_split(record);
        assert_eq!(record.rust_arguments, ["u32", "u64"]);
        assert_eq!(record.rust_result, "()");
        assert!(!record.safe);
        assert!(record.convergent);
        assert_eq!(record.memory, "read_write");
        assert_eq!(record.targets, TCGEN05_LLVM_TARGETS);
    }

    let packed = copies
        .iter()
        .find(|record| record.id == "tcgen05_cp_128x128b_b4x16_p64")
        .unwrap();
    assert_eq!(
        packed.expected_ptx.modifiers,
        ["cp", "cta_group::1", "128x128b", "b8x16", "b4x16_p64"]
    );
    let warpx4 = copies
        .iter()
        .find(|record| record.id == "tcgen05_cp_32x128b_warpx4")
        .unwrap();
    assert_eq!(
        warpx4.expected_ptx.modifiers,
        ["cp", "cta_group::1", "32x128b", "warpx4"]
    );
    let pair_01_23 = copies
        .iter()
        .find(|record| record.id == "tcgen05_cp_64x128b_warpx2_01_23_b6x16_p32_cg2")
        .unwrap();
    assert_eq!(
        pair_01_23.expected_ptx.modifiers,
        [
            "cp",
            "cta_group::2",
            "64x128b",
            "warpx2::01_23",
            "b8x16",
            "b6x16_p32"
        ]
    );
    let pair_02_13 = copies
        .iter()
        .find(|record| record.id == "tcgen05_cp_64x128b_warpx2_02_13")
        .unwrap();
    assert_eq!(
        pair_02_13.expected_ptx.modifiers,
        ["cp", "cta_group::1", "64x128b", "warpx2::02_13"]
    );

    let mut missing = test_tcgen05_cp_admission();
    missing.cp_variants.pop();
    assert!(expand_tcgen05_admission(&missing).is_err());

    let mut reordered = test_tcgen05_cp_admission();
    reordered.cp_variants.swap(0, 1);
    assert!(expand_tcgen05_admission(&reordered).is_err());

    let mut non_contiguous_abi = test_tcgen05_cp_admission();
    non_contiguous_abi.cp_variants[0].abi_id = "i9999".into();
    let records = expand_tcgen05_admission(&non_contiguous_abi).unwrap();
    assert!(records.iter().any(|record| record.abi_id == "i9999"));

    let mut missing_evidence = test_tcgen05_cp_admission();
    missing_evidence.cp_llvm_evidence_profile = None;
    assert!(expand_tcgen05_admission(&missing_evidence).is_err());

    let declaration = declarations[copies[0].source_record.as_deref().unwrap()];
    let mut wrong_spelling = copies[0].clone();
    wrong_spelling.expected_ptx.modifiers.remove(3);
    assert!(validate_imported_policy(&wrong_spelling, declaration).is_err());

    let mut broadened_libnvvm = copies[0].clone();
    broadened_libnvvm.backend_lowerings[1].targets = None;
    assert!(validate_imported_policy(&broadened_libnvvm, declaration).is_err());
}

#[test]
fn compact_tcgen05_load_admission_matches_all_llvm_records_and_fails_closed() {
    let records = expand_tcgen05_admission(&test_tcgen05_ld_admission()).unwrap();
    assert_eq!(records.len(), 119);
    let loads = &records[61..];
    assert_eq!(loads.len(), 58);
    assert_eq!(
        (loads[0].abi_id.as_str(), loads[0].id.as_str()),
        ("i0612", "tcgen05_ld_16x64b_x1_raw")
    );
    assert_eq!(
        (loads[1].abi_id.as_str(), loads[1].id.as_str()),
        ("i0613", "tcgen05_ld_16x64b_x1_pack16")
    );
    assert_eq!(
        (loads[57].abi_id.as_str(), loads[57].id.as_str()),
        ("i0669", "tcgen05_ld_32x32b_x128_pack16")
    );

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let imported: ImportedFile = read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
    let declarations = imported
        .intrinsics
        .iter()
        .map(|record| (record.source_record.as_str(), record))
        .collect::<BTreeMap<_, _>>();
    let mut source_counts = BTreeMap::new();
    for record in loads {
        let source_record = record.source_record.as_deref().unwrap();
        *source_counts.entry(source_record).or_insert(0) += 1;
        let declaration = declarations[source_record];
        validate_imported_policy(record, declaration).unwrap();
        assert_tcgen05_backend_target_split(record);
        assert_eq!(record.rust_arguments, ["u32"]);
        assert!(record.must_use);
        assert!(!record.safe);
        assert!(!record.pure);
        assert_eq!(record.memory, "read");
        assert!(record.convergent);
        assert_eq!(record.execution_scope, "warp");
        assert!(
            !record
                .rust_arguments
                .iter()
                .any(|argument| argument == "bool")
        );
        assert_eq!(record.dialect_operands, ["i32"]);
        assert_eq!(record.llvm_arguments, ["tmem_ptr", "i1"]);
        assert!(declaration.selections.is_empty());
    }
    assert_eq!(source_counts.len(), 29);
    assert!(source_counts.values().all(|count| *count == 2));
    assert_eq!(
        records
            .iter()
            .filter(|record| {
                record.source_record.as_deref() == Some("int_nvvm_tcgen05_ld_16x256b_x1")
            })
            .count(),
        3
    );
    assert_eq!(
        records
            .iter()
            .filter(|record| {
                record.source_record.as_deref() == Some("int_nvvm_tcgen05_ld_16x256b_x8")
            })
            .count(),
        3
    );
    validate_unique_overlay(&records, 1).unwrap();

    let scalar_raw = &loads[0];
    assert_eq!(scalar_raw.rust_result, "u32");
    assert_eq!(scalar_raw.dialect_results, ["i32"]);
    assert_eq!(scalar_raw.llvm_results, ["anonymous_9933"]);
    assert_eq!(
        scalar_raw.expected_ptx.modifiers,
        ["ld", "sync", "aligned", "16x64b", "x1", "b32"]
    );
    assert_eq!(
        scalar_raw.expected_ptx.operands,
        [
            OperandPattern::RegisterList { length: 1 },
            OperandPattern::Address,
        ]
    );
    assert!(
        scalar_raw
            .expected_ptx
            .matches("tcgen05.ld.sync.aligned.16x64b.x1.b32 {%r1}, [%r2];")
            .unwrap()
    );
    assert!(
        !scalar_raw
            .expected_ptx
            .matches("tcgen05.ld.sync.aligned.16x64b.x1.b32 %r1, [%r2];")
            .unwrap()
    );
    let scalar_pack = &loads[1];
    assert_eq!(
        scalar_pack.expected_ptx.modifiers,
        ["ld", "sync", "aligned", "16x64b", "x1", "pack::16b", "b32",]
    );
    let largest = &loads[57];
    assert_eq!(largest.rust_result, "[u32; 128]");
    assert_eq!(largest.dialect_results.len(), 128);
    assert_eq!(largest.llvm_results, ["anonymous_9961"]);
    assert_eq!(
        largest.expected_ptx.operands,
        [
            OperandPattern::RegisterList { length: 128 },
            OperandPattern::Address,
        ]
    );

    let mut missing = test_tcgen05_ld_admission();
    missing.ld_variants.pop();
    assert!(expand_tcgen05_admission(&missing).is_err());

    let mut reordered = test_tcgen05_ld_admission();
    reordered.ld_variants.swap(0, 1);
    assert!(expand_tcgen05_admission(&reordered).is_err());

    let mut non_contiguous_abi = test_tcgen05_ld_admission();
    non_contiguous_abi.ld_variants[0].abi_id = "i9999".into();
    let records = expand_tcgen05_admission(&non_contiguous_abi).unwrap();
    assert!(records.iter().any(|record| record.abi_id == "i9999"));

    let mut missing_evidence = test_tcgen05_ld_admission();
    missing_evidence.ld_llvm_evidence_profile = None;
    assert!(expand_tcgen05_admission(&missing_evidence).is_err());

    let declaration = declarations[loads[0].source_record.as_deref().unwrap()];
    let mut wrong_selector = loads[0].clone();
    wrong_selector
        .tcgen05
        .as_mut()
        .unwrap()
        .ld
        .as_mut()
        .unwrap()
        .pack16 = true;
    assert!(validate_imported_policy(&wrong_selector, declaration).is_err());

    let mut changed_declaration = declaration.clone();
    changed_declaration.properties.pop();
    assert!(validate_imported_policy(&loads[0], &changed_declaration).is_err());

    let mut wrong_scope = loads[0].clone();
    wrong_scope.execution_scope = "thread".into();
    assert!(validate_imported_policy(&wrong_scope, declaration).is_err());

    let mut unreviewed_sharing = records.clone();
    unreviewed_sharing[62].tcgen05.as_mut().unwrap().operation = Tcgen05Operation::Alloc;
    assert!(validate_unique_overlay(&unreviewed_sharing, 1).is_err());
}

#[test]
fn compact_tcgen05_store_admission_matches_all_llvm_records_and_fails_closed() {
    let records = expand_tcgen05_admission(&test_tcgen05_st_admission()).unwrap();
    assert_eq!(records.len(), 177);
    let stores = &records[119..];
    assert_eq!(stores.len(), 58);
    assert_eq!(
        (stores[0].abi_id.as_str(), stores[0].id.as_str()),
        ("i0670", "tcgen05_st_16x64b_x1_raw")
    );
    assert_eq!(
        (stores[1].abi_id.as_str(), stores[1].id.as_str()),
        ("i0671", "tcgen05_st_16x64b_x1_unpack16")
    );
    assert_eq!(
        (stores[57].abi_id.as_str(), stores[57].id.as_str()),
        ("i0727", "tcgen05_st_32x32b_x128_unpack16")
    );

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let imported: ImportedFile = read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
    let declarations = imported
        .intrinsics
        .iter()
        .map(|record| (record.source_record.as_str(), record))
        .collect::<BTreeMap<_, _>>();
    let mut source_counts = BTreeMap::new();
    for record in stores {
        let source_record = record.source_record.as_deref().unwrap();
        *source_counts.entry(source_record).or_insert(0) += 1;
        let declaration = declarations[source_record];
        validate_imported_policy(record, declaration).unwrap();
        assert_tcgen05_backend_target_split(record);
        assert_eq!(record.rust_arguments.len(), 2);
        assert_eq!(record.rust_result, "()");
        assert!(!record.must_use);
        assert!(!record.safe);
        assert!(!record.pure);
        assert_eq!(record.memory, "write");
        assert!(record.convergent);
        assert_eq!(record.execution_scope, "warp");
        assert_eq!(record.dialect_results, Vec::<String>::new());
        assert_eq!(record.llvm_results, Vec::<String>::new());
        assert_eq!(record.llvm_arguments, declaration.arguments);
        assert_eq!(
            declaration.properties,
            [
                "ImmArg<arg2>",
                "IntrArgMemOnly",
                "IntrConvergent",
                "NoCapture<arg0>",
            ]
        );
        assert!(declaration.selections.is_empty());
    }
    assert_eq!(source_counts.len(), 29);
    assert!(source_counts.values().all(|count| *count == 2));
    validate_unique_overlay(&records, 1).unwrap();

    let scalar_raw = &stores[0];
    assert_eq!(scalar_raw.rust_arguments, ["u32", "u32"]);
    assert_eq!(scalar_raw.dialect_operands, ["i32", "i32"]);
    assert_eq!(
        scalar_raw.llvm_arguments,
        ["tmem_ptr", "anonymous_9933", "i1"]
    );
    assert_eq!(
        scalar_raw.expected_ptx.modifiers,
        ["st", "sync", "aligned", "16x64b", "x1", "b32"]
    );
    assert_eq!(
        scalar_raw.expected_ptx.operands,
        [
            OperandPattern::Address,
            OperandPattern::RegisterList { length: 1 },
        ]
    );
    assert!(
        scalar_raw
            .expected_ptx
            .matches("tcgen05.st.sync.aligned.16x64b.x1.b32 [%r1], {%r2};")
            .unwrap()
    );
    assert!(
        !scalar_raw
            .expected_ptx
            .matches("tcgen05.st.sync.aligned.16x64b.x1.b32 [%r1], %r2;")
            .unwrap()
    );
    assert_eq!(
        stores[1].expected_ptx.modifiers,
        [
            "st",
            "sync",
            "aligned",
            "16x64b",
            "x1",
            "unpack::16b",
            "b32",
        ]
    );
    let largest = &stores[57];
    assert_eq!(largest.rust_arguments, ["u32", "[u32; 128]"]);
    assert_eq!(largest.dialect_operands.len(), 129);
    assert_eq!(largest.llvm_arguments, ["tmem_ptr", "anonymous_9961", "i1"]);
    assert_eq!(
        largest.expected_ptx.operands,
        [
            OperandPattern::Address,
            OperandPattern::RegisterList { length: 128 },
        ]
    );

    let mut missing = test_tcgen05_st_admission();
    missing.st_variants.pop();
    assert!(expand_tcgen05_admission(&missing).is_err());

    let mut reordered = test_tcgen05_st_admission();
    reordered.st_variants.swap(0, 1);
    assert!(expand_tcgen05_admission(&reordered).is_err());

    let mut non_contiguous_abi = test_tcgen05_st_admission();
    non_contiguous_abi.st_variants[0].abi_id = "i9999".into();
    let records = expand_tcgen05_admission(&non_contiguous_abi).unwrap();
    assert!(records.iter().any(|record| record.abi_id == "i9999"));

    let mut missing_evidence = test_tcgen05_st_admission();
    missing_evidence.st_llvm_evidence_profile = None;
    assert!(expand_tcgen05_admission(&missing_evidence).is_err());

    let declaration = declarations[stores[0].source_record.as_deref().unwrap()];
    let mut wrong_selector = stores[0].clone();
    wrong_selector
        .tcgen05
        .as_mut()
        .unwrap()
        .st
        .as_mut()
        .unwrap()
        .unpack16 = true;
    assert!(validate_imported_policy(&wrong_selector, declaration).is_err());

    let mut changed_declaration = declaration.clone();
    changed_declaration.properties.pop();
    assert!(validate_imported_policy(&stores[0], &changed_declaration).is_err());

    let mut wrong_scope = stores[0].clone();
    wrong_scope.execution_scope = "thread".into();
    assert!(validate_imported_policy(&wrong_scope, declaration).is_err());

    let mut unreviewed_sharing = records.clone();
    unreviewed_sharing[120].tcgen05.as_mut().unwrap().operation = Tcgen05Operation::Alloc;
    assert!(validate_unique_overlay(&unreviewed_sharing, 1).is_err());
}

#[test]
fn compact_tcgen05_offset_admission_is_exact_and_fails_closed() {
    let records = expand_tcgen05_admission(&test_tcgen05_offset_admission()).unwrap();
    assert_eq!(records.len(), 209);
    let loads = &records[177..193];
    let stores = &records[193..209];
    assert_eq!(
        (loads[0].abi_id.as_str(), loads[0].id.as_str()),
        ("i0728", "tcgen05_ld_16x32bx2_x1_raw")
    );
    assert_eq!(
        (loads[15].abi_id.as_str(), loads[15].id.as_str()),
        ("i0743", "tcgen05_ld_16x32bx2_x128_pack16")
    );
    assert_eq!(
        (stores[0].abi_id.as_str(), stores[0].id.as_str()),
        ("i0744", "tcgen05_st_16x32bx2_x1_raw")
    );
    assert_eq!(
        (stores[15].abi_id.as_str(), stores[15].id.as_str()),
        ("i0759", "tcgen05_st_16x32bx2_x128_unpack16")
    );

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let imported: ImportedFile = read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
    let declarations = imported
        .intrinsics
        .iter()
        .map(|record| (record.source_record.as_str(), record))
        .collect::<BTreeMap<_, _>>();
    let mut load_sources = BTreeMap::new();
    for record in loads {
        let source = record.source_record.as_deref().unwrap();
        *load_sources.entry(source).or_insert(0) += 1;
        let declaration = declarations[source];
        validate_imported_policy(record, declaration).unwrap();
        assert_tcgen05_backend_target_split(record);
        assert_eq!(record.rust_arguments, ["u32", "i64"]);
        assert_eq!(record.dialect_operands, ["i32", "i64"]);
        assert_eq!(record.llvm_arguments, ["tmem_ptr", "i64", "i1"]);
        assert_eq!(record.execution_scope, "warp");
        assert!(record.must_use);
        assert_eq!(
            record.compatibility_rust_paths,
            [format!("cuda_device::tcgen05::__{}", record.id)]
        );
        assert_eq!(
            record.expected_ptx.operands.last(),
            Some(&OperandPattern::Immediate)
        );
        assert_eq!(
            declaration.properties,
            [
                "ImmArg<arg1>",
                "ImmArg<arg2>",
                "IntrArgMemOnly",
                "IntrConvergent",
                "NoCapture<arg0>",
            ]
        );
    }
    assert_eq!(load_sources.len(), 8);
    assert!(load_sources.values().all(|count| *count == 2));

    let mut store_sources = BTreeMap::new();
    for record in stores {
        let source = record.source_record.as_deref().unwrap();
        *store_sources.entry(source).or_insert(0) += 1;
        let declaration = declarations[source];
        validate_imported_policy(record, declaration).unwrap();
        assert_tcgen05_backend_target_split(record);
        assert_eq!(record.rust_arguments[0..2], ["u32", "i64"]);
        assert_eq!(record.dialect_operands[0..2], ["i32", "i64"]);
        assert_eq!(record.llvm_arguments[0..2], ["tmem_ptr", "i64"]);
        assert_eq!(record.execution_scope, "warp");
        assert!(!record.must_use);
        assert_eq!(
            record.compatibility_rust_paths,
            [format!("cuda_device::tcgen05::__{}", record.id)]
        );
        assert_eq!(
            record.expected_ptx.operands.get(1),
            Some(&OperandPattern::Immediate)
        );
        assert_eq!(
            declaration.properties,
            [
                "ImmArg<arg1>",
                "ImmArg<arg3>",
                "IntrArgMemOnly",
                "IntrConvergent",
                "NoCapture<arg0>",
            ]
        );
    }
    assert_eq!(store_sources.len(), 8);
    assert!(store_sources.values().all(|count| *count == 2));
    validate_unique_overlay(&records, 1).unwrap();

    let scalar_pack = &loads[1];
    assert_eq!(
        scalar_pack.expected_ptx.operands,
        [
            OperandPattern::RegisterList { length: 1 },
            OperandPattern::Address,
            OperandPattern::Immediate,
        ]
    );
    assert!(
        scalar_pack
            .expected_ptx
            .matches("tcgen05.ld.sync.aligned.16x32bx2.x1.pack::16b.b32 {%r1}, [%r2], 16;")
            .unwrap()
    );

    let scalar_unpack = &stores[1];
    assert_eq!(
        scalar_unpack.expected_ptx.operands,
        [
            OperandPattern::Address,
            OperandPattern::Immediate,
            OperandPattern::RegisterList { length: 1 },
        ]
    );
    assert!(
        scalar_unpack
            .expected_ptx
            .matches("tcgen05.st.sync.aligned.16x32bx2.x1.unpack::16b.b32 [%r1], 16, {%r2};")
            .unwrap()
    );

    let mut missing = test_tcgen05_offset_admission();
    missing.ld_offset_variants.pop();
    assert!(expand_tcgen05_admission(&missing).is_err());

    let mut reordered = test_tcgen05_offset_admission();
    reordered.st_offset_variants.swap(0, 1);
    assert!(expand_tcgen05_admission(&reordered).is_err());

    let mut non_contiguous_abi = test_tcgen05_offset_admission();
    non_contiguous_abi.ld_offset_variants[0].abi_id = "i9999".into();
    let records = expand_tcgen05_admission(&non_contiguous_abi).unwrap();
    assert!(records.iter().any(|record| record.abi_id == "i9999"));

    let mut missing_evidence = test_tcgen05_offset_admission();
    missing_evidence.offset_llvm_evidence_profile = None;
    assert!(expand_tcgen05_admission(&missing_evidence).is_err());

    let load_declaration = declarations[loads[0].source_record.as_deref().unwrap()];
    let mut wrong_offset_type = loads[0].clone();
    wrong_offset_type.rust_arguments[1] = "i32".into();
    assert!(validate_imported_policy(&wrong_offset_type, load_declaration).is_err());

    let mut wrong_scope = loads[0].clone();
    wrong_scope.execution_scope = "thread".into();
    assert!(validate_imported_policy(&wrong_scope, load_declaration).is_err());

    let mut missing_immediate = loads[0].clone();
    missing_immediate.expected_ptx.operands.pop();
    assert!(validate_imported_policy(&missing_immediate, load_declaration).is_err());

    let store_declaration = declarations[stores[0].source_record.as_deref().unwrap()];
    let mut wrong_adapter = stores[0].clone();
    wrong_adapter.tcgen05.as_mut().unwrap().adapter =
        Tcgen05Adapter::TmemU32RegistersInjectUnpack16ToVoid;
    assert!(validate_imported_policy(&wrong_adapter, store_declaration).is_err());

    let mut changed_declaration = store_declaration.clone();
    changed_declaration.properties.remove(0);
    assert!(validate_imported_policy(&stores[0], &changed_declaration).is_err());
}

#[test]
fn tcgen05_compact_schema_is_reserved_for_aggregation() {
    let shard = |schema, admission| OverlayShardFile {
        schema,
        family: "tcgen05".into(),
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
        tma: None,
        tcgen05: Some(admission),
    };
    let path = Path::new("intrinsics/overlay/tcgen05.toml");
    validate_overlay_shard_schema_with_max(
        &shard(
            TCGEN05_SHARD_SCHEMA,
            without_tcgen05_control(test_tcgen05_admission()),
        ),
        path,
        TCGEN05_CP_SHARD_SCHEMA,
    )
    .unwrap();
    assert!(
        validate_overlay_shard_schema_with_max(
            &shard(
                TCGEN05_SHARD_SCHEMA - 1,
                without_tcgen05_control(test_tcgen05_admission()),
            ),
            path,
            TCGEN05_CP_SHARD_SCHEMA,
        )
        .unwrap_err()
        .to_string()
        .contains("requires overlay shard schema 42")
    );
    validate_overlay_shard_schema_with_max(
        &shard(
            TCGEN05_CP_SHARD_SCHEMA,
            without_tcgen05_control(test_tcgen05_cp_admission()),
        ),
        path,
        TCGEN05_CP_SHARD_SCHEMA,
    )
    .unwrap();
    assert!(
        validate_overlay_shard_schema_with_max(
            &shard(
                TCGEN05_CP_SHARD_SCHEMA - 1,
                without_tcgen05_control(test_tcgen05_cp_admission()),
            ),
            path,
            TCGEN05_CP_SHARD_SCHEMA,
        )
        .unwrap_err()
        .to_string()
        .contains("requires overlay shard schema 52")
    );
    validate_overlay_shard_schema_with_max(
        &shard(
            TCGEN05_LD_SHARD_SCHEMA,
            without_tcgen05_control(test_tcgen05_ld_admission()),
        ),
        path,
        TCGEN05_LD_SHARD_SCHEMA,
    )
    .unwrap();
    assert!(
        validate_overlay_shard_schema_with_max(
            &shard(
                TCGEN05_LD_SHARD_SCHEMA - 1,
                without_tcgen05_control(test_tcgen05_ld_admission()),
            ),
            path,
            TCGEN05_LD_SHARD_SCHEMA,
        )
        .unwrap_err()
        .to_string()
        .contains("requires overlay shard schema 53")
    );
    validate_overlay_shard_schema_with_max(
        &shard(
            TCGEN05_ST_SHARD_SCHEMA,
            without_tcgen05_control(test_tcgen05_st_admission()),
        ),
        path,
        TCGEN05_ST_SHARD_SCHEMA,
    )
    .unwrap();
    assert!(
        validate_overlay_shard_schema_with_max(
            &shard(
                TCGEN05_ST_SHARD_SCHEMA - 1,
                without_tcgen05_control(test_tcgen05_st_admission()),
            ),
            path,
            TCGEN05_ST_SHARD_SCHEMA,
        )
        .unwrap_err()
        .to_string()
        .contains("requires overlay shard schema 54")
    );
    validate_overlay_shard_schema_with_max(
        &shard(
            TCGEN05_OFFSET_LDST_SHARD_SCHEMA,
            without_tcgen05_control(test_tcgen05_offset_admission()),
        ),
        path,
        TCGEN05_OFFSET_LDST_SHARD_SCHEMA,
    )
    .unwrap();
    assert!(
        validate_overlay_shard_schema_with_max(
            &shard(
                TCGEN05_OFFSET_LDST_SHARD_SCHEMA - 1,
                without_tcgen05_control(test_tcgen05_offset_admission()),
            ),
            path,
            TCGEN05_OFFSET_LDST_SHARD_SCHEMA,
        )
        .unwrap_err()
        .to_string()
        .contains("requires overlay shard schema 55")
    );
    validate_overlay_shard_schema_with_max(
        &shard(
            TCGEN05_CONTROL_SHARD_SCHEMA,
            test_tcgen05_offset_admission(),
        ),
        path,
        TCGEN05_CONTROL_SHARD_SCHEMA,
    )
    .unwrap();
    assert!(
        validate_overlay_shard_schema_with_max(
            &shard(
                TCGEN05_CONTROL_SHARD_SCHEMA,
                without_tcgen05_control(test_tcgen05_offset_admission()),
            ),
            path,
            TCGEN05_CONTROL_SHARD_SCHEMA,
        )
        .unwrap_err()
        .to_string()
        .contains("requires all three control variants and both backend evidence profiles")
    );
    assert!(
        validate_overlay_shard_schema_with_max(
            &shard(
                TCGEN05_CONTROL_SHARD_SCHEMA - 1,
                test_tcgen05_offset_admission(),
            ),
            path,
            TCGEN05_CONTROL_SHARD_SCHEMA,
        )
        .unwrap_err()
        .to_string()
        .contains("requires overlay shard schema 56")
    );
    validate_overlay_shard_schema_with_max(
        &shard(TCGEN05_MMA_SHARD_SCHEMA, test_tcgen05_mma_admission()),
        path,
        TCGEN05_MMA_SHARD_SCHEMA,
    )
    .unwrap();
    assert!(
        validate_overlay_shard_schema_with_max(
            &shard(TCGEN05_MMA_SHARD_SCHEMA - 1, test_tcgen05_mma_admission(),),
            path,
            TCGEN05_MMA_SHARD_SCHEMA,
        )
        .unwrap_err()
        .to_string()
        .contains("requires overlay shard schema 57")
    );
}
