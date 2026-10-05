/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::extract::IMPORTED_SCHEMA;
use crate::model::{
    BackendLoweringMechanism, CpAsyncAdapter, CpAsyncCachePolicy, CpAsyncControlAdapter,
    CpAsyncControlOperation, CpAsyncCopySize, CpAsyncMbarrierAdapter, CpAsyncMbarrierOperation,
    CpAsyncMbarrierStateSpace, CpAsyncSourceSize, DebugControlAdapter, DebugControlOperation,
    DotProductAdapter, DotProductOperation, DotProductSignedness, ImportedFile, ImportedIntrinsic,
    IntrinsicBackend, IntrinsicSource, MbarrierBasicAdapter, MbarrierBasicOperation,
    MbarrierExtendedAdapter, MbarrierExtendedOperation, MbarrierExtendedSourceContract,
    OverlayIntrinsic, OverlayShardFile, ReduxAdapter, ReduxOperation, RuntimeValidation,
    WgmmaControlAdapter, WgmmaControlMode,
};
use crate::ptx::OperandPattern;
use crate::util::read_json;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use super::fixtures::*;
use crate::resolve::evidence::*;
use crate::resolve::families::*;
use crate::resolve::guards::*;
use crate::resolve::materialize::*;
use crate::resolve::overlay::*;
use crate::resolve::policy::*;
use crate::resolve::targets::*;

#[test]
fn f32_redux_recipes_match_pinned_llvm_records() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let imported: ImportedFile = read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
    let declarations: BTreeMap<_, _> = imported
        .intrinsics
        .iter()
        .map(|record| (record.source_record.as_str(), record))
        .collect();
    let cases = [
        (
            ReduxOperation::Fmin,
            "redux.sync.min.f32 \t$dst, $src, $mask;",
            &["sync", "min", "f32"][..],
        ),
        (
            ReduxOperation::FminNan,
            "redux.sync.min.NaN.f32 \t$dst, $src, $mask;",
            &["sync", "min", "NaN", "f32"],
        ),
        (
            ReduxOperation::FminAbs,
            "redux.sync.min.abs.f32 \t$dst, $src, $mask;",
            &["sync", "min", "abs", "f32"],
        ),
        (
            ReduxOperation::FminAbsNan,
            "redux.sync.min.abs.NaN.f32 \t$dst, $src, $mask;",
            &["sync", "min", "abs", "NaN", "f32"],
        ),
        (
            ReduxOperation::Fmax,
            "redux.sync.max.f32 \t$dst, $src, $mask;",
            &["sync", "max", "f32"],
        ),
        (
            ReduxOperation::FmaxNan,
            "redux.sync.max.NaN.f32 \t$dst, $src, $mask;",
            &["sync", "max", "NaN", "f32"],
        ),
        (
            ReduxOperation::FmaxAbs,
            "redux.sync.max.abs.f32 \t$dst, $src, $mask;",
            &["sync", "max", "abs", "f32"],
        ),
        (
            ReduxOperation::FmaxAbsNan,
            "redux.sync.max.abs.NaN.f32 \t$dst, $src, $mask;",
            &["sync", "max", "abs", "NaN", "f32"],
        ),
    ];

    for (operation, expected_asm, expected_modifiers) in cases {
        let recipe = redux_recipe(operation);
        let declaration = declarations[recipe.source_record];
        assert_eq!(declaration.llvm_name, recipe.llvm_symbol);
        assert_eq!(declaration.arguments, ["f32", "i32"]);
        assert_eq!(declaration.results, ["f32"]);
        assert_eq!(declaration.selections.len(), 1);
        assert_eq!(declaration.selections[0].asm, expected_asm);
        assert_eq!(
            declaration.selections[0].predicates,
            ["Subtarget->hasReduxSyncF32()"]
        );
        assert_eq!(recipe.ptx_modifiers, expected_modifiers);
        assert_eq!(recipe.minimum_ptx, "8.6");
        assert_eq!(recipe.minimum_sm, None);
        assert_eq!(recipe.targets, REDUX_F32_TARGETS);
    }
}

#[test]
fn compact_debug_control_admission_is_closed() {
    let records = expand_debug_control_admission(&test_debug_control_admission()).unwrap();
    assert_eq!(records.len(), 3);
    assert_eq!(records[0].id, "trap");
    assert_eq!(records[1].id, "breakpoint");
    assert_eq!(records[2].id, "pmevent");
    for record in &records {
        validate_ptx_native_policy(record).unwrap();
        assert_eq!(record.backend_lowerings.len(), 2);
        assert!(
            record
                .debug_control
                .as_ref()
                .is_some_and(|debug| debug.runtime_validation == RuntimeValidation::Unexecuted)
        );
    }
    assert_eq!(records[0].minimum_ptx, "1.0");
    assert_eq!(records[0].minimum_sm, None);
    assert_eq!(records[1].minimum_ptx, "1.0");
    assert_eq!(records[1].minimum_sm.as_deref(), Some("sm_11"));
    assert_eq!(records[2].minimum_ptx, "1.4");
    assert_eq!(
        records[2].expected_ptx.operands,
        [OperandPattern::Immediate]
    );

    let mut pending = test_debug_control_admission();
    pending.abi_ids.clear();
    assert!(expand_debug_control_admission(&pending).is_err());

    let mut missing = test_debug_control_admission();
    missing.operations.pop();
    assert!(expand_debug_control_admission(&missing).is_err());

    let mut duplicate_operation = test_debug_control_admission();
    duplicate_operation.operations[2] = DebugControlOperation::Breakpoint;
    assert!(expand_debug_control_admission(&duplicate_operation).is_err());

    let mut duplicate_id = test_debug_control_admission();
    duplicate_id.abi_ids[2] = duplicate_id.abi_ids[1].clone();
    assert!(expand_debug_control_admission(&duplicate_id).is_err());

    let mut malformed_id = test_debug_control_admission();
    malformed_id.abi_ids[0] = "debug1".into();
    assert!(expand_debug_control_admission(&malformed_id).is_err());

    let mut executed = test_debug_control_admission();
    executed.runtime_validation = RuntimeValidation::Executed;
    assert!(expand_debug_control_admission(&executed).is_err());

    let mut wrong_source = records[0].clone();
    wrong_source.source = Some(IntrinsicSource::LlvmImported {
        source_record: "invented".into(),
    });
    assert!(validate_ptx_native_policy(&wrong_source).is_err());

    let mut wrong_adapter = records[2].clone();
    wrong_adapter.debug_control.as_mut().unwrap().adapter = DebugControlAdapter::Direct;
    assert!(validate_ptx_native_policy(&wrong_adapter).is_err());

    let mut wrong_immediate = records[2].clone();
    wrong_immediate.expected_ptx.operands = vec![OperandPattern::Register];
    assert!(validate_ptx_native_policy(&wrong_immediate).is_err());

    let mut wrong_floor = records[1].clone();
    wrong_floor.backend_lowerings[0].minimum_sm = Some("sm_75".into());
    assert!(validate_ptx_native_policy(&wrong_floor).is_err());
}

#[test]
fn debug_control_compact_schema_is_reserved_for_aggregation() {
    let shard = |schema| OverlayShardFile {
        schema,
        family: "debug_control".into(),
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
        debug_control: Some(test_debug_control_admission()),
        threadfence: None,
        cluster_memory: None,
        stmatrix: None,
        clc: None,
        wgmma_controls: None,
        tma: None,
        tcgen05: None,
    };
    let path = Path::new("intrinsics/overlay/debug_control.toml");
    validate_overlay_shard_schema_with_max(&shard(33), path, 33).unwrap();
    assert!(validate_overlay_shard_schema_with_max(&shard(33), path, 32).is_err());
    let error = validate_overlay_shard_schema_with_max(&shard(32), path, 33).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("requires overlay shard schema 33")
    );
}

#[test]
fn active_debug_control_sources_parse_and_prove_both_backend_routes() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let shard_path = repo_root.join("intrinsics/overlay/debug_control.toml");
    let shard: OverlayShardFile =
        toml::from_str(&fs::read_to_string(&shard_path).unwrap()).unwrap();
    validate_overlay_shard_schema_with_max(&shard, &shard_path, 33).unwrap();
    let admission = shard.debug_control.unwrap();
    assert_eq!(admission.abi_ids, ["i0295", "i0296", "i0297"]);
    let records = expand_debug_control_admission(&admission).unwrap();

    let evidence = vec![
        read_evidence_file(
            &repo_root.join("intrinsics/evidence/rust-llvm-23.1.0-16696adc-debug-control.json"),
        )
        .unwrap(),
        read_evidence_file(
            &repo_root.join("intrinsics/evidence/cuda-13.3-libnvvm-13.3.33-debug-control.json"),
        )
        .unwrap(),
    ];
    let indexed = index_evidence(&evidence, "16696adcd119e6ba9cc175207d984d7021211acb").unwrap();
    for record in &records {
        let routes = resolve_backend_lowerings(record, &indexed).unwrap();
        assert_eq!(routes.len(), 2);
        assert!(routes.iter().all(|route| {
            route.mechanism == BackendLoweringMechanism::InlinePtx && route.status == "validated"
        }));
    }
}

#[test]
fn compact_extended_mbarrier_admission_preserves_all_manual_contracts() {
    let records = expand_mbarrier_extended_admission(&test_mbarrier_extended_admission()).unwrap();
    assert_eq!(records.len(), 11);
    assert_eq!(
        records
            .iter()
            .map(|record| record.abi_id.as_str())
            .collect::<Vec<_>>(),
        (306..=316)
            .map(|id| format!("i{id:04}"))
            .collect::<Vec<_>>()
    );

    let imported: ImportedFile = read_json(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("intrinsics/imported.json"),
    )
    .unwrap();
    for record in &records {
        let contract = record.mbarrier_extended.as_ref().unwrap();
        let (template, constraints) = mbarrier_extended_inline_recipe(contract.operation);
        assert!(template.ends_with(';') || template.ends_with("; }"));
        assert!(constraints.contains("~{memory}"));
        assert!(record.convergent && record.memory == "read_write");
        assert!(
            record
                .backend_lowerings
                .iter()
                .all(|lowering| { lowering.mechanism == BackendLoweringMechanism::InlinePtx })
        );
        match contract.source_contract {
            MbarrierExtendedSourceContract::LlvmImported => {
                let declaration = imported
                    .intrinsics
                    .iter()
                    .find(|declaration| {
                        Some(declaration.source_record.as_str()) == record.source_record.as_deref()
                    })
                    .unwrap();
                validate_imported_policy(record, declaration).unwrap();
            }
            MbarrierExtendedSourceContract::PtxNativeRawClusterAddress => {
                validate_ptx_native_policy(record).unwrap();
                assert_eq!(record.rust_arguments, ["u64"]);
                assert_eq!(record.dialect_operands, ["i64"]);
            }
        }
    }

    let base = records
        .iter()
        .find(|record| record.id == "mbarrier_arrive_expect_tx")
        .unwrap();
    let declaration = imported
        .intrinsics
        .iter()
        .find(|declaration| {
            Some(declaration.source_record.as_str()) == base.source_record.as_deref()
        })
        .unwrap();
    let mut wrong_adapter = base.clone();
    wrong_adapter.mbarrier_extended.as_mut().unwrap().adapter =
        MbarrierExtendedAdapter::PointerTokenToPredicate;
    assert!(validate_imported_policy(&wrong_adapter, declaration).is_err());
    let mut wrong_floor = base.clone();
    wrong_floor.minimum_ptx = "8.6".into();
    assert!(validate_imported_policy(&wrong_floor, declaration).is_err());
    let mut lost_clobber = base.clone();
    lost_clobber.memory = "none".into();
    assert!(validate_imported_policy(&lost_clobber, declaration).is_err());

    let remote = records
        .iter()
        .find(|record| record.id == "mbarrier_arrive_cluster")
        .unwrap();
    let incompatible = imported
        .intrinsics
        .iter()
        .find(|declaration| {
            declaration.source_record == "int_nvvm_mbarrier_arrive_scope_cluster_space_cluster"
        })
        .unwrap();
    assert_eq!(incompatible.arguments, ["shared_cluster_ptr", "i32"]);
    assert!(
        validate_mbarrier_extended_policy(
            remote,
            &IntrinsicSource::LlvmImported {
                source_record: incompatible.source_record.clone(),
            },
            Some(incompatible),
        )
        .is_err()
    );

    let mut missing = test_mbarrier_extended_admission();
    missing.variants.pop();
    assert!(expand_mbarrier_extended_admission(&missing).is_err());
    let mut duplicate = test_mbarrier_extended_admission();
    duplicate.variants[10].operation = MbarrierExtendedOperation::ArriveExpectTxCta;
    assert!(expand_mbarrier_extended_admission(&duplicate).is_err());
    let mut wrong_abi = test_mbarrier_extended_admission();
    wrong_abi.variants[0].abi_id = "i9999".into();
    assert!(expand_mbarrier_extended_admission(&wrong_abi).is_err());
}

#[test]
fn compact_wgmma_control_admission_and_semantics_fail_closed() {
    let records = expand_wgmma_control_admission(&test_wgmma_control_admission()).unwrap();
    assert_eq!(records.len(), 3);

    let imported: ImportedFile = read_json(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("intrinsics/imported.json"),
    )
    .unwrap();
    let declaration_for = |record: &OverlayIntrinsic| {
        imported
            .intrinsics
            .iter()
            .find(|declaration| {
                Some(declaration.source_record.as_str()) == record.source_record.as_deref()
            })
            .unwrap()
    };
    for record in &records {
        validate_imported_policy(record, declaration_for(record)).unwrap();
        assert_eq!(record.targets, "sm_90a");
        assert_eq!(record.minimum_ptx, "8.0");
        assert_eq!(record.backend_lowerings.len(), 2);
    }

    let wait = records
        .iter()
        .find(|record| {
            record
                .wgmma_control
                .as_ref()
                .is_some_and(|control| control.mode == WgmmaControlMode::WaitGroup)
        })
        .unwrap();
    assert_eq!(wait.rust_arguments, ["u64"]);
    assert_eq!(wait.dialect_operands, ["i64"]);
    assert_eq!(
        wait.compatibility_rust_paths,
        ["cuda_device::wgmma::__wgmma_wait_group"]
    );

    let mut wrong_adapter = wait.clone();
    wrong_adapter.wgmma_control.as_mut().unwrap().adapter = WgmmaControlAdapter::NoArguments;
    assert!(validate_imported_policy(&wrong_adapter, declaration_for(wait)).is_err());

    let mut wrong_participation = wait.clone();
    wrong_participation.execution_scope = "warp".into();
    assert!(validate_imported_policy(&wrong_participation, declaration_for(wait)).is_err());

    let mut wrong_route = wait.clone();
    wrong_route.backend_lowerings[1].mechanism = BackendLoweringMechanism::TypedNvvm;
    assert!(validate_imported_policy(&wrong_route, declaration_for(wait)).is_err());

    let mut wrong_target = wait.clone();
    wrong_target.targets = "all".into();
    wrong_target.minimum_sm = Some("sm_90".into());
    assert!(validate_imported_policy(&wrong_target, declaration_for(wait)).is_err());

    let mut missing = test_wgmma_control_admission();
    missing.variants.pop();
    assert!(expand_wgmma_control_admission(&missing).is_err());

    let mut reversed = test_wgmma_control_admission();
    reversed.variants.reverse();
    assert!(expand_wgmma_control_admission(&reversed).is_err());

    let mut duplicate = test_wgmma_control_admission();
    duplicate.variants[2].mode = WgmmaControlMode::Fence;
    assert!(expand_wgmma_control_admission(&duplicate).is_err());

    let mut wrong_abi = test_wgmma_control_admission();
    wrong_abi.variants[0].abi_id = "i9999".into();
    assert!(expand_wgmma_control_admission(&wrong_abi).is_err());

    let mut executed = test_wgmma_control_admission();
    executed.runtime_validation = RuntimeValidation::Executed;
    assert!(expand_wgmma_control_admission(&executed).is_err());
}

#[test]
fn wgmma_control_compact_schema_is_reserved_for_aggregation() {
    let shard = OverlayShardFile {
        schema: WGMMA_CONTROL_SHARD_SCHEMA,
        family: "wgmma_control".into(),
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
        wgmma_controls: Some(test_wgmma_control_admission()),
        tma: None,
        tcgen05: None,
    };
    let path = Path::new("intrinsics/overlay/wgmma_control.toml");
    validate_overlay_shard_schema_with_max(&shard, path, WGMMA_CONTROL_SHARD_SCHEMA).unwrap();
    let mut old = shard;
    old.schema -= 1;
    assert!(
        validate_overlay_shard_schema_with_max(&old, path, WGMMA_CONTROL_SHARD_SCHEMA)
            .unwrap_err()
            .to_string()
            .contains("requires overlay shard schema 38")
    );
}

#[test]
fn cp_async_copy_recipe_admits_only_classic_llvm_forms() {
    let cases = [
        (
            CpAsyncCachePolicy::Ca,
            CpAsyncCopySize::B4,
            CpAsyncSourceSize::Full,
            Some("cp_async_ca_4"),
        ),
        (
            CpAsyncCachePolicy::Ca,
            CpAsyncCopySize::B4,
            CpAsyncSourceSize::Runtime,
            Some("cp_async_ca_zfill_4"),
        ),
        (
            CpAsyncCachePolicy::Ca,
            CpAsyncCopySize::B8,
            CpAsyncSourceSize::Full,
            Some("cp_async_ca_8"),
        ),
        (
            CpAsyncCachePolicy::Ca,
            CpAsyncCopySize::B8,
            CpAsyncSourceSize::Runtime,
            Some("cp_async_ca_zfill_8"),
        ),
        (
            CpAsyncCachePolicy::Ca,
            CpAsyncCopySize::B16,
            CpAsyncSourceSize::Full,
            Some("cp_async_ca_16"),
        ),
        (
            CpAsyncCachePolicy::Ca,
            CpAsyncCopySize::B16,
            CpAsyncSourceSize::Runtime,
            Some("cp_async_ca_zfill_16"),
        ),
        (
            CpAsyncCachePolicy::Cg,
            CpAsyncCopySize::B4,
            CpAsyncSourceSize::Full,
            None,
        ),
        (
            CpAsyncCachePolicy::Cg,
            CpAsyncCopySize::B4,
            CpAsyncSourceSize::Runtime,
            None,
        ),
        (
            CpAsyncCachePolicy::Cg,
            CpAsyncCopySize::B8,
            CpAsyncSourceSize::Full,
            None,
        ),
        (
            CpAsyncCachePolicy::Cg,
            CpAsyncCopySize::B8,
            CpAsyncSourceSize::Runtime,
            None,
        ),
        (
            CpAsyncCachePolicy::Cg,
            CpAsyncCopySize::B16,
            CpAsyncSourceSize::Full,
            Some("cp_async_cg_16"),
        ),
        (
            CpAsyncCachePolicy::Cg,
            CpAsyncCopySize::B16,
            CpAsyncSourceSize::Runtime,
            Some("cp_async_cg_zfill_16"),
        ),
    ];

    for (cache_policy, copy_size, source_size, expected) in cases {
        let copy = crate::model::CpAsyncCopy {
            cache_policy,
            copy_size,
            source_size,
            adapter: if source_size == CpAsyncSourceSize::Runtime {
                CpAsyncAdapter::DirectPointersAndSourceSize
            } else {
                CpAsyncAdapter::DirectPointers
            },
            runtime_validation: RuntimeValidation::Unexecuted,
        };
        assert_eq!(
            cp_async_copy_recipe(&copy).map(|recipe| recipe.id),
            expected
        );
    }
}

#[test]
fn pinned_cp_async_records_match_the_closed_recipes() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let (overlay, _) =
        read_overlay(&repo_root, &repo_root.join("intrinsics/overlay.toml")).unwrap();
    let imported: ImportedFile = read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
    let declarations: BTreeMap<_, _> = imported
        .intrinsics
        .iter()
        .map(|record| (record.source_record.as_str(), record))
        .collect();
    let policies: Vec<_> = overlay
        .intrinsics
        .iter()
        .filter(|record| matches!(record.family.as_str(), "cp_async_copy" | "cp_async_control"))
        .collect();

    assert_eq!(policies.len(), 11);
    for policy in policies {
        let declaration = declarations[policy.source_record.as_deref().unwrap()];
        validate_imported_policy(policy, declaration).unwrap();
    }
}

#[test]
fn pinned_cp_async_mbarrier_records_match_the_closed_recipes() {
    let records = pinned_cp_async_mbarrier_records();
    assert_eq!(records.len(), 4);

    for (policy, declaration) in records.values() {
        validate_imported_policy(policy, declaration).unwrap();
    }
}

#[test]
fn cp_async_mbarrier_recipes_fail_closed() {
    let records = pinned_cp_async_mbarrier_records();
    let (arrive, declaration) = &records["cp_async_mbarrier_arrive"];
    let reject = |policy: &OverlayIntrinsic, declaration: &ImportedIntrinsic, expected: &str| {
        let error = validate_imported_policy(policy, declaration).unwrap_err();
        let message = error.to_string();
        assert!(message.contains(expected), "unexpected error: {message}");
    };

    let mut wrong_symbol = arrive.clone();
    wrong_symbol.llvm_symbol = Some("llvm.nvvm.cp.async.mbarrier.changed".into());
    reject(&wrong_symbol, declaration, "LLVM symbol mismatch");

    let mut wrong_signature = arrive.clone();
    wrong_signature.rust_arguments = vec!["*const u64".into()];
    reject(
        &wrong_signature,
        declaration,
        "closed cp.async mbarrier Rust API",
    );

    let mut wrong_operation = arrive.clone();
    wrong_operation
        .cp_async_mbarrier
        .as_mut()
        .unwrap()
        .operation = CpAsyncMbarrierOperation::ArriveNoInc;
    reject(&wrong_operation, declaration, "identity does not match");

    let mut wrong_state_space = arrive.clone();
    wrong_state_space
        .cp_async_mbarrier
        .as_mut()
        .unwrap()
        .state_space = CpAsyncMbarrierStateSpace::Shared;
    reject(&wrong_state_space, declaration, "identity does not match");

    let mut wrong_adapter = arrive.clone();
    wrong_adapter.cp_async_mbarrier.as_mut().unwrap().adapter =
        CpAsyncMbarrierAdapter::PointerToVoid;
    wrong_adapter.rust_result = "u64".into();
    reject(
        &wrong_adapter,
        declaration,
        "closed cp.async mbarrier Rust API",
    );

    let mut executed_without_evidence = arrive.clone();
    executed_without_evidence
        .cp_async_mbarrier
        .as_mut()
        .unwrap()
        .runtime_validation = RuntimeValidation::Executed;
    reject(
        &executed_without_evidence,
        declaration,
        "unrecorded cp.async mbarrier runtime validation",
    );

    let mut wrong_properties = declaration.clone();
    wrong_properties.properties.pop();
    reject(arrive, &wrong_properties, "cp.async mbarrier properties");

    let mut wrong_selection = declaration.clone();
    wrong_selection.selections[0].source_record = "CP_ASYNC_MBARRIER_CHANGED".into();
    reject(
        arrive,
        &wrong_selection,
        "imported cp.async mbarrier selection changed",
    );

    let mut wrong_floor = arrive.clone();
    // Keep the PTX introduction bound valid so the recipe checks this SM.
    wrong_floor.minimum_sm = Some("sm_75".into());
    validate_arch_introduction_floor(&wrong_floor).unwrap();
    reject(&wrong_floor, declaration, "effects or target floor");

    let mut wrong_llvm_route = arrive.clone();
    wrong_llvm_route
        .backend_lowerings
        .iter_mut()
        .find(|lowering| lowering.backend == IntrinsicBackend::LlvmNvptx)
        .unwrap()
        .mechanism = BackendLoweringMechanism::InlinePtx;
    reject(&wrong_llvm_route, declaration, "reviewed typed-LLVM");

    let mut wrong_lib_route = arrive.clone();
    wrong_lib_route
        .backend_lowerings
        .iter_mut()
        .find(|lowering| lowering.backend == IntrinsicBackend::LibNvvm)
        .unwrap()
        .mechanism = BackendLoweringMechanism::TypedNvvm;
    reject(&wrong_lib_route, declaration, "reviewed typed-LLVM");

    let mut mixed_family = arrive.clone();
    mixed_family.cp_async_control = Some(crate::model::CpAsyncControl {
        operation: CpAsyncControlOperation::CommitGroup,
        adapter: CpAsyncControlAdapter::NoOperands,
        runtime_validation: RuntimeValidation::Unexecuted,
    });
    reject(
        &mixed_family,
        declaration,
        "mixes another generated-family contract",
    );
}

#[test]
fn pinned_mbarrier_basic_records_match_the_closed_recipes() {
    let records = pinned_mbarrier_basic_records();
    assert_eq!(records.len(), 5);

    for (policy, declaration) in records.values() {
        validate_imported_policy(policy, declaration).unwrap();
    }
}

#[test]
fn mbarrier_basic_recipes_fail_closed() {
    let records = pinned_mbarrier_basic_records();
    let (init, init_declaration) = &records["mbarrier_init"];
    let reject = |policy: &OverlayIntrinsic, declaration: &ImportedIntrinsic, expected: &str| {
        let error = validate_imported_policy(policy, declaration).unwrap_err();
        let message = error.to_string();
        assert!(message.contains(expected), "unexpected error: {message}");
    };

    let mut wrong_symbol = init.clone();
    wrong_symbol.llvm_symbol = Some("llvm.nvvm.mbarrier.init.changed".into());
    reject(&wrong_symbol, init_declaration, "LLVM symbol mismatch");

    let mut wrong_signature = init.clone();
    wrong_signature.rust_arguments = vec!["*mut u32".into(), "u32".into()];
    reject(
        &wrong_signature,
        init_declaration,
        "unsafe mbarrier raw and compatibility API",
    );

    let mut wrong_operation = init.clone();
    wrong_operation.mbarrier_basic.as_mut().unwrap().operation = MbarrierBasicOperation::Arrive;
    reject(
        &wrong_operation,
        init_declaration,
        "operation, state space, and adapter disagree",
    );

    let mut wrong_adapter = init.clone();
    wrong_adapter.mbarrier_basic.as_mut().unwrap().adapter =
        MbarrierBasicAdapter::InvalPointerToVoid;
    reject(
        &wrong_adapter,
        init_declaration,
        "operation, state space, and adapter disagree",
    );

    let (no_complete, no_complete_declaration) = &records["mbarrier_arrive_no_complete"];
    let mut wrong_no_complete_adapter = no_complete.clone();
    wrong_no_complete_adapter
        .mbarrier_basic
        .as_mut()
        .unwrap()
        .adapter = MbarrierBasicAdapter::ArrivePointerToToken;
    reject(
        &wrong_no_complete_adapter,
        no_complete_declaration,
        "operation, state space, and adapter disagree",
    );

    let mut executed_without_evidence = init.clone();
    executed_without_evidence
        .mbarrier_basic
        .as_mut()
        .unwrap()
        .runtime_validation = RuntimeValidation::Executed;
    reject(
        &executed_without_evidence,
        init_declaration,
        "unrecorded mbarrier runtime validation",
    );

    let mut wrong_properties = init_declaration.clone();
    wrong_properties.properties.pop();
    reject(init, &wrong_properties, "mbarrier properties");

    let mut wrong_selection = init_declaration.clone();
    wrong_selection.selections[0].source_record = "MBARRIER_INIT_CHANGED".into();
    reject(
        init,
        &wrong_selection,
        "imported mbarrier selection changed",
    );

    let mut wrong_ptx_floor = init.clone();
    wrong_ptx_floor.minimum_ptx = "7.1".into();
    reject(
        &wrong_ptx_floor,
        init_declaration,
        "effects or target floor",
    );

    let mut wrong_sm_floor = init.clone();
    // PTX 7.0 can name sm_75, but this recipe requires sm_80.
    wrong_sm_floor.minimum_sm = Some("sm_75".into());
    validate_arch_introduction_floor(&wrong_sm_floor).unwrap();
    reject(&wrong_sm_floor, init_declaration, "effects or target floor");

    let mut wrong_llvm_route = init.clone();
    wrong_llvm_route
        .backend_lowerings
        .iter_mut()
        .find(|lowering| lowering.backend == IntrinsicBackend::LlvmNvptx)
        .unwrap()
        .mechanism = BackendLoweringMechanism::InlinePtx;
    reject(
        &wrong_llvm_route,
        init_declaration,
        "reviewed mbarrier backend routes",
    );

    let mut wrong_lib_nvvm_route = init.clone();
    wrong_lib_nvvm_route
        .backend_lowerings
        .iter_mut()
        .find(|lowering| lowering.backend == IntrinsicBackend::LibNvvm)
        .unwrap()
        .mechanism = BackendLoweringMechanism::TypedNvvm;
    reject(
        &wrong_lib_nvvm_route,
        init_declaration,
        "reviewed mbarrier backend routes",
    );

    let mut route_with_unreviewed_floor = init.clone();
    route_with_unreviewed_floor.backend_lowerings[0].minimum_sm = Some("sm_90".into());
    reject(
        &route_with_unreviewed_floor,
        init_declaration,
        "reviewed mbarrier backend routes",
    );

    let mut mixed_family = init.clone();
    mixed_family.cp_async_control = Some(crate::model::CpAsyncControl {
        operation: CpAsyncControlOperation::CommitGroup,
        adapter: CpAsyncControlAdapter::NoOperands,
        runtime_validation: RuntimeValidation::Unexecuted,
    });
    reject(
        &mixed_family,
        init_declaration,
        "mixes another generated-family contract",
    );
}

#[test]
fn redux_contract_validates_effects_participation_and_operand_adapter() {
    let valid = redux_policy();
    let imported = redux_declaration();
    validate_imported_policy(&valid, &imported).unwrap();

    assert_eq!(
        valid.redux.as_ref().unwrap().adapter,
        ReduxAdapter::MaskValueToSourceMemberMask
    );

    let mut missing_contract = valid.clone();
    missing_contract.redux = None;
    assert!(
        validate_imported_policy(&missing_contract, &imported)
            .unwrap_err()
            .to_string()
            .contains("closed redux contract")
    );

    let mut wrong_effect = valid.clone();
    wrong_effect.memory = "none".into();
    assert!(
        validate_imported_policy(&wrong_effect, &imported)
            .unwrap_err()
            .to_string()
            .contains("redux effects")
    );

    let mut missing_imported_effect = imported;
    missing_imported_effect
        .properties
        .retain(|property| property != "IntrInaccessibleMemOnly");
    assert!(
        validate_imported_policy(&valid, &missing_imported_effect)
            .unwrap_err()
            .to_string()
            .contains("memory and convergence effects")
    );
}

#[test]
fn every_redux_variant_matches_its_closed_recipe() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let (overlay, _) =
        read_overlay(&repo_root, &repo_root.join("intrinsics/overlay.toml")).unwrap();
    let imported: ImportedFile = read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
    assert_eq!(imported.schema, IMPORTED_SCHEMA);
    let declarations: BTreeMap<_, _> = imported
        .intrinsics
        .iter()
        .map(|record| (record.source_record.as_str(), record))
        .collect();
    let redux: Vec<_> = overlay
        .intrinsics
        .iter()
        .filter(|record| record.family == "redux")
        .collect();
    assert_eq!(redux.len(), 16);

    for policy in redux {
        let declaration = declarations
            .get(policy.source_record.as_deref().unwrap())
            .unwrap();
        validate_imported_policy(policy, declaration).unwrap();
    }

    let mut mismatched = packed_policy("redux_sync_min_u32");
    mismatched.redux.as_mut().unwrap().operation = ReduxOperation::Umax;
    let declaration = declarations["int_nvvm_redux_sync_umin"];
    assert!(
        validate_imported_policy(&mismatched, declaration)
            .unwrap_err()
            .to_string()
            .contains("closed operation recipe")
    );
}

#[test]
fn every_dot_product_variant_matches_its_closed_recipe() {
    let variants = [
        (
            DotProductOperation::Dp4a,
            DotProductSignedness::Signed,
            "dp4a_s32",
            "int_nvvm_idp4a_s_s",
            "integer.dot_product.dp4a.s32",
        ),
        (
            DotProductOperation::Dp4a,
            DotProductSignedness::Unsigned,
            "dp4a_u32",
            "int_nvvm_idp4a_u_u",
            "integer.dot_product.dp4a.u32",
        ),
        (
            DotProductOperation::Dp2a,
            DotProductSignedness::Signed,
            "dp2a_s32",
            "int_nvvm_idp2a_s_s",
            "integer.dot_product.dp2a.lo.s32",
        ),
        (
            DotProductOperation::Dp2a,
            DotProductSignedness::Unsigned,
            "dp2a_u32",
            "int_nvvm_idp2a_u_u",
            "integer.dot_product.dp2a.lo.u32",
        ),
    ];

    for (operation, signedness, id, source_record, operation_key) in variants {
        let policy = dot_product_policy(operation, signedness);
        let declaration = dot_product_declaration(operation, signedness);
        assert_eq!(policy.id, id);
        assert_eq!(policy.source_record.as_deref(), Some(source_record));
        assert_eq!(policy.operation_key, operation_key);
        validate_imported_policy(&policy, &declaration).unwrap();
    }
}

#[test]
fn pinned_dot_product_records_match_the_reviewed_overlay() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let (overlay, _) =
        read_overlay(&repo_root, &repo_root.join("intrinsics/overlay.toml")).unwrap();
    let imported: ImportedFile = read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
    let declarations: BTreeMap<_, _> = imported
        .intrinsics
        .iter()
        .map(|record| (record.source_record.as_str(), record))
        .collect();
    let dot_products: Vec<_> = overlay
        .intrinsics
        .iter()
        .filter(|record| record.family == "dotprod")
        .collect();
    assert_eq!(dot_products.len(), 4);

    for policy in dot_products {
        let declaration = declarations[policy.source_record.as_deref().unwrap()];
        validate_imported_policy(policy, declaration).unwrap();
        let selected: Vec<_> = declaration
            .selections
            .iter()
            .filter(|selection| selection_matches_policy(policy, selection).unwrap())
            .collect();
        assert_eq!(selected.len(), 1);
        if policy.id.starts_with("dp2a") {
            assert_eq!(selected[0].constraints.immediate_bindings[0].value, 0);
        }
    }
}

#[test]
fn dp2a_selects_only_the_reviewed_low_half_binding() {
    let policy = dot_product_policy(DotProductOperation::Dp2a, DotProductSignedness::Signed);
    let declaration =
        dot_product_declaration(DotProductOperation::Dp2a, DotProductSignedness::Signed);
    let resolved = resolve_record(
        &policy,
        resolve_policy_source(&policy).unwrap(),
        Some(&declaration),
        &dot_product_evidence(&policy),
        "test",
        "LLVM version test",
        "0123456789abcdef",
        vec![],
        1,
    )
    .unwrap();

    assert_eq!(resolved.selections.len(), 1);
    assert_eq!(resolved.selections[0].source_record, "DOT2_lo");
    assert_eq!(
        resolved.selections[0].constraints.immediate_bindings,
        [crate::model::ImportedImmediateBinding {
            argument_index: 2,
            value: 0,
        }]
    );
    assert_eq!(
        resolved.dot_product.as_ref().unwrap().adapter,
        DotProductAdapter::InsertLowHalfFalse
    );

    let mut wrong_binding = declaration;
    wrong_binding.selections[1].constraints.immediate_bindings[0].value = -1;
    let error = validate_imported_policy(&policy, &wrong_binding).unwrap_err();
    assert!(error.to_string().contains("does not agree"));
}

#[test]
fn dot_product_recipe_rejects_unreviewed_api_and_adapter_changes() {
    let valid = dot_product_policy(DotProductOperation::Dp2a, DotProductSignedness::Unsigned);
    let declaration =
        dot_product_declaration(DotProductOperation::Dp2a, DotProductSignedness::Unsigned);

    let mut wrong_adapter = valid.clone();
    wrong_adapter.dot_product.as_mut().unwrap().adapter = DotProductAdapter::DirectThreeOperands;
    assert!(
        validate_imported_policy(&wrong_adapter, &declaration)
            .unwrap_err()
            .to_string()
            .contains("source adapter")
    );

    let mut must_use = valid.clone();
    must_use.must_use = true;
    assert!(
        validate_imported_policy(&must_use, &declaration)
            .unwrap_err()
            .to_string()
            .contains("non-must-use")
    );

    let mut wrong_llvm_signature = valid;
    wrong_llvm_signature.llvm_arguments = vec!["i32".into(); 3];
    assert!(
        validate_imported_policy(&wrong_llvm_signature, &declaration)
            .unwrap_err()
            .to_string()
            .contains("LLVM argument signature mismatch")
    );
}

#[test]
fn dot_product_target_predicate_is_closed_to_ptx50_and_sm61() {
    let policy = dot_product_policy(DotProductOperation::Dp4a, DotProductSignedness::Signed);
    let selection =
        &dot_product_declaration(DotProductOperation::Dp4a, DotProductSignedness::Signed)
            .selections[0];
    validate_selected_target_predicates(&policy, selection).unwrap();

    let mut wrong_ptx = policy.clone();
    wrong_ptx.minimum_ptx = "5.1".into();
    assert!(
        validate_selected_target_predicates(&wrong_ptx, selection)
            .unwrap_err()
            .to_string()
            .contains("minimum PTX")
    );

    let mut wrong_sm = policy;
    wrong_sm.minimum_sm = Some("sm_60".into());
    assert!(
        validate_selected_target_predicates(&wrong_sm, selection)
            .unwrap_err()
            .to_string()
            .contains("minimum SM")
    );
}

#[test]
fn return_range_properties_are_half_open_and_unique() {
    let facts =
        imported_result_facts(&["NoUndef<ret>".into(), "Range<ret,1,1025>".into()]).unwrap();
    assert!(facts.no_undef);
    let range = facts.range.unwrap();
    assert_eq!(range.lower, "1");
    assert_eq!(range.upper_exclusive, "1025");

    let duplicate =
        imported_result_facts(&["Range<ret,0,32>".into(), "Range<ret,0,64>".into()]).unwrap_err();
    assert!(duplicate.to_string().contains("duplicate return range"));
}
