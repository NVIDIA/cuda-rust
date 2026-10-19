/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, Clc, ClcAdapter, ClcAdmission, ClcOperation, ImportedIntrinsic,
    IntrinsicBackend, OverlayBackendLowering, OverlayIntrinsic, RuntimeValidation,
};
use crate::ptx::{InstructionPattern, OperandPattern};
use anyhow::{Context, Result, ensure};

use crate::resolve::guards::*;

#[derive(Clone, Copy)]
pub(in crate::resolve) struct ClcRecipe {
    pub(in crate::resolve) operation: ClcOperation,
    pub(in crate::resolve) abi_id: &'static str,
    pub(in crate::resolve) id: &'static str,
    pub(in crate::resolve) operation_key: &'static str,
    pub(in crate::resolve) source_record: &'static str,
    pub(in crate::resolve) llvm_symbol: &'static str,
    pub(in crate::resolve) rust_arguments: &'static [&'static str],
    pub(in crate::resolve) llvm_arguments: &'static [&'static str],
    pub(in crate::resolve) llvm_results: &'static [&'static str],
    pub(in crate::resolve) dialect_op_type: &'static str,
    pub(in crate::resolve) dialect_op_name: &'static str,
    pub(in crate::resolve) dialect_operands: &'static [&'static str],
    pub(in crate::resolve) dialect_results: &'static [&'static str],
    pub(in crate::resolve) adapter: ClcAdapter,
    pub(in crate::resolve) modifiers: &'static [&'static str],
    pub(in crate::resolve) operands: &'static [OperandPattern],
    pub(in crate::resolve) targets: &'static str,
    pub(in crate::resolve) minimum_sm: Option<&'static str>,
    pub(in crate::resolve) pure: bool,
    pub(in crate::resolve) memory: &'static str,
    pub(in crate::resolve) execution_scope: &'static str,
    pub(in crate::resolve) summary: &'static str,
}

pub(in crate::resolve) fn clc_recipe(operation: ClcOperation) -> ClcRecipe {
    const TRY_OPERANDS: &[OperandPattern] = &[OperandPattern::Address, OperandPattern::Address];
    const QUERY_OPERANDS: &[OperandPattern] = &[OperandPattern::Register, OperandPattern::Register];
    let (abi_id, id, operation_key, source_record, llvm_symbol, op_type, op_name) = match operation
    {
        ClcOperation::TryCancel => (
            "i0322",
            "clc_try_cancel",
            "cluster.launch_control.try_cancel",
            "int_nvvm_clusterlaunchcontrol_try_cancel_async_shared",
            "llvm.nvvm.clusterlaunchcontrol.try_cancel.async.shared",
            "ClcTryCancelOp",
            "nvvm.clc_try_cancel",
        ),
        ClcOperation::TryCancelMulticast => (
            "i0323",
            "clc_try_cancel_multicast",
            "cluster.launch_control.try_cancel.multicast",
            "int_nvvm_clusterlaunchcontrol_try_cancel_async_multicast_shared",
            "llvm.nvvm.clusterlaunchcontrol.try_cancel.async.multicast.shared",
            "ClcTryCancelMulticastOp",
            "nvvm.clc_try_cancel_multicast",
        ),
        ClcOperation::QueryIsCanceled => (
            "i0324",
            "clc_query_is_canceled",
            "cluster.launch_control.query.is_canceled",
            "int_nvvm_clusterlaunchcontrol_query_cancel_is_canceled",
            "llvm.nvvm.clusterlaunchcontrol.query_cancel.is_canceled",
            "ClcQueryIsCanceledOp",
            "nvvm.clc_query_is_canceled",
        ),
        ClcOperation::QueryGetFirstCtaidX => (
            "i0325",
            "clc_query_get_first_ctaid_x",
            "cluster.launch_control.query.first_ctaid.x",
            "int_nvvm_clusterlaunchcontrol_query_cancel_get_first_ctaid_x",
            "llvm.nvvm.clusterlaunchcontrol.query_cancel.get_first_ctaid.x",
            "ClcQueryGetFirstCtaidXOp",
            "nvvm.clc_query_get_first_ctaid_x",
        ),
        ClcOperation::QueryGetFirstCtaidY => (
            "i0326",
            "clc_query_get_first_ctaid_y",
            "cluster.launch_control.query.first_ctaid.y",
            "int_nvvm_clusterlaunchcontrol_query_cancel_get_first_ctaid_y",
            "llvm.nvvm.clusterlaunchcontrol.query_cancel.get_first_ctaid.y",
            "ClcQueryGetFirstCtaidYOp",
            "nvvm.clc_query_get_first_ctaid_y",
        ),
        ClcOperation::QueryGetFirstCtaidZ => (
            "i0327",
            "clc_query_get_first_ctaid_z",
            "cluster.launch_control.query.first_ctaid.z",
            "int_nvvm_clusterlaunchcontrol_query_cancel_get_first_ctaid_z",
            "llvm.nvvm.clusterlaunchcontrol.query_cancel.get_first_ctaid.z",
            "ClcQueryGetFirstCtaidZOp",
            "nvvm.clc_query_get_first_ctaid_z",
        ),
    };
    match operation {
        ClcOperation::TryCancel | ClcOperation::TryCancelMulticast => ClcRecipe {
            operation,
            abi_id,
            id,
            operation_key,
            source_record,
            llvm_symbol,
            rust_arguments: &["*mut u8", "*mut u64"],
            llvm_arguments: &["shared_ptr", "shared_ptr"],
            llvm_results: &[],
            dialect_op_type: op_type,
            dialect_op_name: op_name,
            dialect_operands: &["ptr", "ptr"],
            dialect_results: &[],
            adapter: ClcAdapter::GenericPointersToShared,
            modifiers: if operation == ClcOperation::TryCancel {
                &[
                    "try_cancel",
                    "async",
                    "shared::cta",
                    "mbarrier::complete_tx::bytes",
                    "b128",
                ]
            } else {
                &[
                    "try_cancel",
                    "async",
                    "shared::cta",
                    "mbarrier::complete_tx::bytes",
                    "multicast::cluster::all",
                    "b128",
                ]
            },
            operands: TRY_OPERANDS,
            targets: if operation == ClcOperation::TryCancel {
                "all"
            } else {
                // LLVM 22 exposes 101a; CUDA 13.3 exposes the other toolkit names.
                "sm_100a|sm_101a|sm_103a|sm_110a|sm_120a|sm_121a"
            },
            minimum_sm: if operation == ClcOperation::TryCancel {
                Some("sm_100")
            } else {
                None
            },
            pure: false,
            memory: "read_write",
            execution_scope: "cta",
            summary: if operation == ClcOperation::TryCancel {
                "Requests one pending CTA and writes its response to shared memory."
            } else {
                "Requests one pending CTA and multicasts its response across the cluster."
            },
        },
        ClcOperation::QueryIsCanceled
        | ClcOperation::QueryGetFirstCtaidX
        | ClcOperation::QueryGetFirstCtaidY
        | ClcOperation::QueryGetFirstCtaidZ => {
            let (adapter, modifiers, summary) = match operation {
                ClcOperation::QueryIsCanceled => (
                    ClcAdapter::PairU64ToI128BoolToU32,
                    &["query_cancel", "is_canceled", "pred", "b128"] as &[_],
                    "Returns whether the Cluster Launch Control request was canceled.",
                ),
                ClcOperation::QueryGetFirstCtaidX => (
                    ClcAdapter::PairU64ToI128U32,
                    &["query_cancel", "get_first_ctaid::x", "b32", "b128"] as &[_],
                    "Returns the X coordinate from a successful cancellation response.",
                ),
                ClcOperation::QueryGetFirstCtaidY => (
                    ClcAdapter::PairU64ToI128U32,
                    &["query_cancel", "get_first_ctaid::y", "b32", "b128"] as &[_],
                    "Returns the Y coordinate from a successful cancellation response.",
                ),
                ClcOperation::QueryGetFirstCtaidZ => (
                    ClcAdapter::PairU64ToI128U32,
                    &["query_cancel", "get_first_ctaid::z", "b32", "b128"] as &[_],
                    "Returns the Z coordinate from a successful cancellation response.",
                ),
                _ => unreachable!(),
            };
            ClcRecipe {
                operation,
                abi_id,
                id,
                operation_key,
                source_record,
                llvm_symbol,
                rust_arguments: &["u64", "u64"],
                llvm_arguments: &["i128"],
                llvm_results: if operation == ClcOperation::QueryIsCanceled {
                    &["i1"]
                } else {
                    &["i32"]
                },
                dialect_op_type: op_type,
                dialect_op_name: op_name,
                dialect_operands: &["i64", "i64"],
                dialect_results: &["i32"],
                adapter,
                modifiers,
                operands: QUERY_OPERANDS,
                targets: "all",
                minimum_sm: Some("sm_100"),
                pure: true,
                memory: "none",
                execution_scope: "thread",
                summary,
            }
        }
    }
}

pub(in crate::resolve) fn expand_clc_admission(
    admission: &ClcAdmission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "CLC runtime validation may be marked executed only with GPU evidence"
    );
    ensure!(
        !admission.llvm_evidence_profile.trim().is_empty()
            && !admission.libnvvm_evidence_profile.trim().is_empty(),
        "compact CLC admission requires both backend evidence profiles"
    );
    let expected = [
        ClcOperation::TryCancel,
        ClcOperation::TryCancelMulticast,
        ClcOperation::QueryIsCanceled,
        ClcOperation::QueryGetFirstCtaidX,
        ClcOperation::QueryGetFirstCtaidY,
        ClcOperation::QueryGetFirstCtaidZ,
    ];
    ensure!(
        admission
            .variants
            .iter()
            .map(|variant| variant.operation)
            .eq(expected),
        "compact CLC admission must list all six operations in canonical order"
    );

    admission
        .variants
        .iter()
        .map(|variant| {
            let recipe = clc_recipe(variant.operation);
            ensure!(
                variant.abi_id == recipe.abi_id,
                "{} must keep reserved ABI ID {}",
                recipe.id,
                recipe.abi_id
            );
            let query = recipe.pure;
            Ok(OverlayIntrinsic {
                id: recipe.id.into(),
                abi_id: variant.abi_id.clone(),
                operation_key: recipe.operation_key.into(),
                family: "clc".into(),
                source: None,
                source_record: Some(recipe.source_record.into()),
                rust_module: "clc".into(),
                rust_name: recipe.id.into(),
                rust_arguments: recipe.rust_arguments.iter().map(|value| (*value).into()).collect(),
                rust_result: if query { "u32".into() } else { "()".into() },
                safe: false,
                must_use: false,
                safe_allowlist_reason: None,
                public_rust_path: format!("cuda_intrinsics::clc::{}", recipe.id),
                compatibility_rust_paths: vec![format!("cuda_device::clc::{}", recipe.id)],
                dialect_op_type: recipe.dialect_op_type.into(),
                dialect_op_name: recipe.dialect_op_name.into(),
                dialect_operands: recipe.dialect_operands.iter().map(|value| (*value).into()).collect(),
                dialect_results: recipe.dialect_results.iter().map(|value| (*value).into()).collect(),
                llvm_symbol: Some(recipe.llvm_symbol.into()),
                resolved_llvm_symbol: None,
                llvm_arguments: recipe.llvm_arguments.iter().map(|value| (*value).into()).collect(),
                llvm_results: recipe.llvm_results.iter().map(|value| (*value).into()).collect(),
                pure: recipe.pure,
                memory: recipe.memory.into(),
                convergent: false,
                execution_scope: recipe.execution_scope.into(),
                minimum_ptx: "8.6".into(),
                minimum_sm: recipe.minimum_sm.map(Into::into),
                ptx_result: if query { "u32".into() } else { "()".into() },
                targets: recipe.targets.into(),
                ptx_isa_version: "9.3".into(),
                ptx_isa_section: "9.7.14.18-19 Cluster Launch Control".into(),
                ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel".into(),
                lowering: "generated_clc".into(),
                backend_lowerings: vec![
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LlvmNvptx,
                        mechanism: BackendLoweringMechanism::TypedNvvm,
                        evidence_profile: admission.llvm_evidence_profile.clone(),
                        targets: None,
                        minimum_ptx: Some("8.6".into()),
                        minimum_sm: recipe.minimum_sm.map(Into::into),
                    },
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LibNvvm,
                        mechanism: BackendLoweringMechanism::TypedNvvm,
                        evidence_profile: admission.libnvvm_evidence_profile.clone(),
                        targets: None,
                        minimum_ptx: Some("8.6".into()),
                        minimum_sm: recipe.minimum_sm.map(Into::into),
                    },
                ],
                packed_atomic: None,
                redux: None,
                vote: None,
                active_mask: None,
                warp_match: None,
                warp_barrier: None,
                warp_shuffle: None,
                dot_product: None,
                packed_alu: None,
                integer_minmax: None,
                packed_conversion: None,
                scalar_conversion: None,
                scalar_arithmetic: None,
                scalar_math: None,
                extended_minmax: None,
                cp_async_copy: None,
                cp_async_control: None,
                cp_async_mbarrier: None,
                mbarrier_basic: None,
                movmatrix: None,
                mbarrier_extended: None,
                register_mma: None,
                sparse_mma: None,
                prmt: None,
                cluster_barrier: None,
                wgmma_control: None,
                special_register: None,
                debug_control: None,
                cluster_memory: None,
                clc: Some(Clc {
                    operation: recipe.operation,
                    adapter: recipe.adapter,
                    runtime_validation: admission.runtime_validation,
                }),
                tma: None,
                tcgen05: None,
                ldmatrix_variant: None,
                ldmatrix_safety: None,
                ldmatrix_adapter: None,
                selected_address_space: None,
                expected_ptx: InstructionPattern {
                    mnemonic: "clusterlaunchcontrol".into(),
                    modifiers: recipe.modifiers.iter().map(|value| (*value).into()).collect(),
                    operands: if query {
                        vec![
                            OperandPattern::Register,
                            OperandPattern::Exact { value: "%clc_handle".into() },
                        ]
                    } else {
                        recipe.operands.to_vec()
                    },
                },
                summary: recipe.summary.into(),
            })
        })
        .collect()
}

pub(in crate::resolve) fn validate_clc_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let clc = policy
        .clc
        .as_ref()
        .with_context(|| format!("{} has no closed CLC contract", policy.id))?;
    let recipe = clc_recipe(clc.operation);
    let query = recipe.pure;
    ensure!(
        policy.id == recipe.id
            && policy.abi_id == recipe.abi_id
            && policy.operation_key == recipe.operation_key
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(recipe.source_record)
            && policy.llvm_symbol.as_deref() == Some(recipe.llvm_symbol)
            && policy.resolved_llvm_symbol.is_none()
            && declaration.source_record == recipe.source_record
            && declaration.llvm_name == recipe.llvm_symbol,
        "{} CLC identity changed",
        policy.id
    );
    ensure!(
        policy.rust_module == "clc"
            && policy.rust_name == recipe.id
            && policy.rust_arguments == recipe.rust_arguments
            && policy.rust_result == if query { "u32" } else { "()" }
            && !policy.safe
            && !policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path == format!("cuda_intrinsics::clc::{}", recipe.id)
            && policy.compatibility_rust_paths == [format!("cuda_device::clc::{}", recipe.id)],
        "{} CLC Rust API changed",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == recipe.dialect_operands
            && policy.dialect_results == recipe.dialect_results
            && policy.llvm_arguments == recipe.llvm_arguments
            && policy.llvm_results == recipe.llvm_results
            && policy.lowering == "generated_clc",
        "{} CLC carrier or LLVM adapter changed",
        policy.id
    );
    ensure!(
        declaration.arguments == recipe.llvm_arguments
            && declaration.results == recipe.llvm_results
            && declaration.properties
                == if query {
                    vec!["IntrNoMem", "IntrSpeculatable"]
                } else {
                    vec!["IntrArgMemOnly", "IntrHasSideEffects"]
                },
        "{} imported CLC declaration changed",
        policy.id
    );
    ensure!(
        policy.pure == recipe.pure
            && policy.memory == recipe.memory
            && !policy.convergent
            && policy.execution_scope == recipe.execution_scope
            && clc.adapter == recipe.adapter
            && clc.runtime_validation == RuntimeValidation::Unexecuted,
        "{} CLC semantics changed",
        policy.id
    );
    ensure!(
        policy.minimum_ptx == "8.6"
            && policy.minimum_sm.as_deref() == recipe.minimum_sm
            && policy.targets == recipe.targets
            && policy.ptx_isa_version == "9.3"
            && policy.ptx_result == if query { "u32" } else { "()" },
        "{} CLC target contract changed",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == "clusterlaunchcontrol"
            && policy.expected_ptx.modifiers == recipe.modifiers
            && policy.expected_ptx.operands
                == if query {
                    vec![
                        OperandPattern::Register,
                        OperandPattern::Exact {
                            value: "%clc_handle".into(),
                        },
                    ]
                } else {
                    recipe.operands.to_vec()
                }
            && policy.backend_lowerings.len() == 2
            && policy.backend_lowerings.iter().all(|route| {
                route.mechanism == BackendLoweringMechanism::TypedNvvm
                    && route.minimum_ptx.as_deref() == Some("8.6")
                    && route.minimum_sm.as_deref() == recipe.minimum_sm
                    && !route.evidence_profile.trim().is_empty()
            }),
        "{} CLC PTX shape or backend route changed",
        policy.id
    );
    ensure_no_other_family_contract(policy, "CLC")?;
    Ok(())
}
