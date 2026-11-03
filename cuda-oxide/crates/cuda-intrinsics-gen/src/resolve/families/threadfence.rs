/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, ImportedIntrinsic, IntrinsicBackend, OverlayBackendLowering,
    OverlayIntrinsic, RuntimeValidation, ThreadfenceAdmission, ThreadfenceScope,
};
use crate::ptx::InstructionPattern;
use anyhow::{Result, ensure};

#[derive(Clone, Copy)]
pub(in crate::resolve) struct ThreadfenceRecipe {
    pub(in crate::resolve) scope: ThreadfenceScope,
    pub(in crate::resolve) abi_id: &'static str,
    pub(in crate::resolve) id: &'static str,
    pub(in crate::resolve) operation_key: &'static str,
    pub(in crate::resolve) source_record: &'static str,
    pub(in crate::resolve) selection_record: &'static str,
    pub(in crate::resolve) llvm_symbol: &'static str,
    pub(in crate::resolve) dialect_op_type: &'static str,
    pub(in crate::resolve) dialect_op_name: &'static str,
    pub(in crate::resolve) ptx_level: &'static str,
    pub(in crate::resolve) execution_scope: &'static str,
    pub(in crate::resolve) minimum_ptx: &'static str,
    pub(in crate::resolve) minimum_sm: Option<&'static str>,
    pub(in crate::resolve) summary: &'static str,
}

pub(in crate::resolve) fn threadfence_recipe(scope: ThreadfenceScope) -> ThreadfenceRecipe {
    match scope {
        ThreadfenceScope::Cta => ThreadfenceRecipe {
            scope,
            abi_id: "i0298",
            id: "threadfence_block",
            operation_key: "memory.fence.cta.sc",
            source_record: "int_nvvm_membar_cta",
            selection_record: "INT_MEMBAR_CTA",
            llvm_symbol: "llvm.nvvm.membar.cta",
            dialect_op_type: "ThreadfenceBlockOp",
            dialect_op_name: "nvvm.threadfence_block",
            ptx_level: "cta",
            execution_scope: "cta",
            minimum_ptx: "1.4",
            minimum_sm: None,
            summary: "Orders this thread's memory operations for observers in its CTA.",
        },
        ThreadfenceScope::Device => ThreadfenceRecipe {
            scope,
            abi_id: "i0299",
            id: "threadfence",
            operation_key: "memory.fence.device.sc",
            source_record: "int_nvvm_membar_gl",
            selection_record: "INT_MEMBAR_GL",
            llvm_symbol: "llvm.nvvm.membar.gl",
            dialect_op_type: "ThreadfenceOp",
            dialect_op_name: "nvvm.threadfence",
            ptx_level: "gl",
            execution_scope: "device",
            minimum_ptx: "1.4",
            minimum_sm: None,
            summary: "Orders this thread's memory operations for observers on its GPU.",
        },
        ThreadfenceScope::System => ThreadfenceRecipe {
            scope,
            abi_id: "i0300",
            id: "threadfence_system",
            operation_key: "memory.fence.system.sc",
            source_record: "int_nvvm_membar_sys",
            selection_record: "INT_MEMBAR_SYS",
            llvm_symbol: "llvm.nvvm.membar.sys",
            dialect_op_type: "ThreadfenceSystemOp",
            dialect_op_name: "nvvm.threadfence_system",
            ptx_level: "sys",
            execution_scope: "system",
            minimum_ptx: "2.0",
            minimum_sm: Some("sm_20"),
            summary: "Orders this thread's memory operations for system-wide observers.",
        },
    }
}

pub(in crate::resolve) fn threadfence_scope_for_id(id: &str) -> Option<ThreadfenceScope> {
    match id {
        "threadfence_block" => Some(ThreadfenceScope::Cta),
        "threadfence" => Some(ThreadfenceScope::Device),
        "threadfence_system" => Some(ThreadfenceScope::System),
        _ => None,
    }
}

pub(in crate::resolve) fn expand_threadfence_admission(
    admission: &ThreadfenceAdmission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "threadfence runtime validation may be marked executed only with GPU evidence"
    );
    ensure!(
        !admission.llvm_evidence_profile.trim().is_empty()
            && !admission.libnvvm_evidence_profile.trim().is_empty(),
        "compact threadfence admission requires both backend evidence profiles"
    );
    let expected_scopes = [
        ThreadfenceScope::Cta,
        ThreadfenceScope::Device,
        ThreadfenceScope::System,
    ];
    let actual_scopes = admission
        .variants
        .iter()
        .map(|variant| variant.scope)
        .collect::<Vec<_>>();
    ensure!(
        actual_scopes == expected_scopes,
        "compact threadfence admission must contain each reviewed scope exactly once in canonical order"
    );

    admission
        .variants
        .iter()
        .map(|variant| {
            let recipe = threadfence_recipe(variant.scope);
            ensure!(
                variant.abi_id == recipe.abi_id,
                "{} must keep reserved ABI ID {}",
                recipe.id,
                recipe.abi_id
            );
            Ok(OverlayIntrinsic {
                id: recipe.id.into(),
                abi_id: variant.abi_id.clone(),
                operation_key: recipe.operation_key.into(),
                family: "sync".into(),
                source: None,
                source_record: Some(recipe.source_record.into()),
                rust_module: "fence".into(),
                rust_name: recipe.id.into(),
                rust_arguments: vec![],
                rust_result: "()".into(),
                safe: true,
                must_use: false,
                safe_allowlist_reason: Some(
                    "a fence only orders the calling thread's memory operations and has no caller preconditions"
                        .into(),
                ),
                public_rust_path: format!("cuda_intrinsics::fence::{}", recipe.id),
                compatibility_rust_paths: vec![
                    format!("cuda_device::fence::{}", recipe.id),
                    format!("cuda_device::{}", recipe.id),
                ],
                dialect_op_type: recipe.dialect_op_type.into(),
                dialect_op_name: recipe.dialect_op_name.into(),
                dialect_operands: vec![],
                dialect_results: vec![],
                llvm_symbol: Some(recipe.llvm_symbol.into()),
                resolved_llvm_symbol: None,
                llvm_arguments: vec![],
                llvm_results: vec![],
                pure: false,
                memory: "read_write".into(),
                convergent: false,
                execution_scope: recipe.execution_scope.into(),
                minimum_ptx: recipe.minimum_ptx.into(),
                minimum_sm: recipe.minimum_sm.map(Into::into),
                ptx_result: "()".into(),
                targets: "all".into(),
                ptx_isa_version: "9.3".into(),
                ptx_isa_section:
                    "9.7.14.4 Parallel Synchronization and Communication Instructions: membar / fence"
                        .into(),
                ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar-fence".into(),
                lowering: "direct_nvvm".into(),
                backend_lowerings: vec![
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LlvmNvptx,
                        mechanism: BackendLoweringMechanism::TypedNvvm,
                        evidence_profile: admission.llvm_evidence_profile.clone(),
                        targets: None,
                        minimum_ptx: Some("3.2".into()),
                        minimum_sm: Some("sm_20".into()),
                    },
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LibNvvm,
                        mechanism: BackendLoweringMechanism::TypedNvvm,
                        evidence_profile: admission.libnvvm_evidence_profile.clone(),
                        targets: None,
                        minimum_ptx: Some("7.0".into()),
                        minimum_sm: Some("sm_80".into()),
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
                clc: None,
                tma: None,
                tcgen05: None,
                ldmatrix_variant: None,
                ldmatrix_safety: None,
                ldmatrix_adapter: None,
                selected_address_space: None,
                expected_ptx: InstructionPattern {
                    mnemonic: "membar".into(),
                    modifiers: vec![recipe.ptx_level.into()],
                    operands: vec![],
                },
                summary: recipe.summary.into(),
            })
        })
        .collect()
}

pub(in crate::resolve) fn validate_threadfence_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
    scope: ThreadfenceScope,
) -> Result<()> {
    let recipe = threadfence_recipe(scope);
    ensure!(
        recipe.scope == scope
            && policy.id == recipe.id
            && policy.abi_id == recipe.abi_id
            && policy.operation_key == recipe.operation_key
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(recipe.source_record)
            && policy.llvm_symbol.as_deref() == Some(recipe.llvm_symbol)
            && policy.resolved_llvm_symbol.is_none(),
        "{} threadfence identity does not match its closed recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "fence"
            && policy.rust_name == recipe.id
            && policy.rust_arguments.is_empty()
            && policy.rust_result == "()"
            && policy.safe
            && !policy.must_use
            && policy.public_rust_path == format!("cuda_intrinsics::fence::{}", recipe.id)
            && policy.compatibility_rust_paths
                == [
                    format!("cuda_device::fence::{}", recipe.id),
                    format!("cuda_device::{}", recipe.id),
                ],
        "{} threadfence Rust API does not match its closed recipe",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands.is_empty()
            && policy.dialect_results.is_empty()
            && policy.llvm_arguments.is_empty()
            && policy.llvm_results.is_empty()
            && policy.lowering == "direct_nvvm",
        "{} threadfence carrier or lowering does not match its closed recipe",
        policy.id
    );
    ensure!(
        declaration.properties == ["IntrNoCallback"]
            && !policy.pure
            && policy.memory == "read_write"
            && !policy.convergent
            && policy.execution_scope == recipe.execution_scope,
        "{} threadfence effects disagree with the imported declaration",
        policy.id
    );
    ensure!(
        policy.minimum_ptx == recipe.minimum_ptx
            && policy.minimum_sm.as_deref() == recipe.minimum_sm
            && policy.ptx_result == "()"
            && policy.targets == "all"
            && policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section
                == "9.7.14.4 Parallel Synchronization and Communication Instructions: membar / fence"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar-fence",
        "{} threadfence target floor or PTX provenance changed",
        policy.id
    );
    ensure!(
        policy.expected_ptx
            == (InstructionPattern {
                mnemonic: "membar".into(),
                modifiers: vec![recipe.ptx_level.into()],
                operands: vec![],
            }),
        "{} expected PTX does not match its closed threadfence scope",
        policy.id
    );
    ensure!(
        policy.backend_lowerings.len() == 2
            && policy.backend_lowerings.iter().any(|route| {
                route.backend == IntrinsicBackend::LlvmNvptx
                    && route.mechanism == BackendLoweringMechanism::TypedNvvm
                    && route.minimum_ptx.as_deref() == Some("3.2")
                    && route.minimum_sm.as_deref() == Some("sm_20")
            })
            && policy.backend_lowerings.iter().any(|route| {
                route.backend == IntrinsicBackend::LibNvvm
                    && route.mechanism == BackendLoweringMechanism::TypedNvvm
                    && route.minimum_ptx.as_deref() == Some("7.0")
                    && route.minimum_sm.as_deref() == Some("sm_80")
            }),
        "{} must keep both reviewed typed threadfence routes",
        policy.id
    );
    ensure!(
        policy.ldmatrix_variant.is_none()
            && policy.ldmatrix_safety.is_none()
            && policy.ldmatrix_adapter.is_none()
            && policy.packed_atomic.is_none()
            && policy.redux.is_none()
            && policy.vote.is_none()
            && policy.active_mask.is_none()
            && policy.warp_match.is_none()
            && policy.warp_barrier.is_none()
            && policy.warp_shuffle.is_none()
            && policy.dot_product.is_none()
            && policy.packed_alu.is_none()
            && policy.packed_conversion.is_none()
            && policy.cp_async_copy.is_none()
            && policy.cp_async_control.is_none()
            && policy.cp_async_mbarrier.is_none()
            && policy.mbarrier_basic.is_none()
            && policy.register_mma.is_none()
            && policy.sparse_mma.is_none()
            && policy.prmt.is_none()
            && policy.cluster_barrier.is_none()
            && policy.special_register.is_none()
            && policy.debug_control.is_none()
            && policy.selected_address_space.is_none(),
        "{} mixes another generated-family contract with threadfence",
        policy.id
    );
    Ok(())
}
