/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, ClusterBarrier, ClusterBarrierAdmission, ClusterBarrierMode,
    ClusterBarrierOrdering, ClusterMemory, ClusterMemoryAdapter, ClusterMemoryAdmission,
    ClusterMemoryOperation, ClusterMemorySourceContract, ImportedIntrinsic, IntrinsicBackend,
    IntrinsicSource, OverlayBackendLowering, OverlayIntrinsic, RuntimeValidation,
};
use crate::ptx::{InstructionPattern, OperandPattern};
use anyhow::{Context, Result, ensure};
use std::collections::BTreeSet;

use crate::resolve::guards::*;

#[derive(Clone, Copy)]
pub(in crate::resolve) struct ClusterBarrierRecipe {
    pub(in crate::resolve) mode: ClusterBarrierMode,
    pub(in crate::resolve) abi_id: &'static str,
    pub(in crate::resolve) id: &'static str,
    pub(in crate::resolve) suffix: &'static str,
    pub(in crate::resolve) minimum_ptx: &'static str,
    pub(in crate::resolve) ordering: ClusterBarrierOrdering,
    pub(in crate::resolve) aligned: bool,
    pub(in crate::resolve) summary: &'static str,
}

pub(in crate::resolve) fn cluster_barrier_recipe(mode: ClusterBarrierMode) -> ClusterBarrierRecipe {
    match mode {
        ClusterBarrierMode::Arrive => ClusterBarrierRecipe {
            mode,
            abi_id: "i0277",
            id: "barrier_cluster_arrive",
            suffix: "arrive",
            minimum_ptx: "7.8",
            ordering: ClusterBarrierOrdering::Release,
            aligned: false,
            summary: "Arrives at the cluster barrier with release ordering.",
        },
        ClusterBarrierMode::ArriveAligned => ClusterBarrierRecipe {
            mode,
            abi_id: "i0278",
            id: "barrier_cluster_arrive_aligned",
            suffix: "arrive.aligned",
            minimum_ptx: "7.8",
            ordering: ClusterBarrierOrdering::Release,
            aligned: true,
            summary: "Arrives at the cluster barrier in aligned mode with release ordering.",
        },
        ClusterBarrierMode::ArriveRelaxed => ClusterBarrierRecipe {
            mode,
            abi_id: "i0279",
            id: "barrier_cluster_arrive_relaxed",
            suffix: "arrive.relaxed",
            minimum_ptx: "8.0",
            ordering: ClusterBarrierOrdering::Relaxed,
            aligned: false,
            summary: "Arrives at the cluster barrier without a release guarantee.",
        },
        ClusterBarrierMode::ArriveRelaxedAligned => ClusterBarrierRecipe {
            mode,
            abi_id: "i0280",
            id: "barrier_cluster_arrive_relaxed_aligned",
            suffix: "arrive.relaxed.aligned",
            minimum_ptx: "8.0",
            ordering: ClusterBarrierOrdering::Relaxed,
            aligned: true,
            summary: "Arrives at the cluster barrier in aligned mode without a release guarantee.",
        },
        ClusterBarrierMode::Wait => ClusterBarrierRecipe {
            mode,
            abi_id: "i0281",
            id: "barrier_cluster_wait",
            suffix: "wait",
            minimum_ptx: "7.8",
            ordering: ClusterBarrierOrdering::Acquire,
            aligned: false,
            summary: "Waits at the cluster barrier with acquire ordering.",
        },
        ClusterBarrierMode::WaitAligned => ClusterBarrierRecipe {
            mode,
            abi_id: "i0282",
            id: "barrier_cluster_wait_aligned",
            suffix: "wait.aligned",
            minimum_ptx: "7.8",
            ordering: ClusterBarrierOrdering::Acquire,
            aligned: true,
            summary: "Waits at the cluster barrier in aligned mode with acquire ordering.",
        },
    }
}

pub(in crate::resolve) fn expand_cluster_barrier_admission(
    admission: &ClusterBarrierAdmission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "cluster-barrier runtime validation may be marked executed only with GPU evidence"
    );
    ensure!(
        !admission.llvm_evidence_profile.trim().is_empty()
            && !admission.libnvvm_evidence_profile.trim().is_empty(),
        "compact cluster-barrier admission requires both backend evidence profiles"
    );
    let expected_modes = BTreeSet::from([
        ClusterBarrierMode::Arrive,
        ClusterBarrierMode::ArriveAligned,
        ClusterBarrierMode::ArriveRelaxed,
        ClusterBarrierMode::ArriveRelaxedAligned,
        ClusterBarrierMode::Wait,
        ClusterBarrierMode::WaitAligned,
    ]);
    let actual_modes: BTreeSet<_> = admission
        .variants
        .iter()
        .map(|variant| variant.mode)
        .collect();
    ensure!(
        admission.variants.len() == expected_modes.len() && actual_modes == expected_modes,
        "compact cluster-barrier admission must contain each reviewed mode exactly once"
    );

    admission
        .variants
        .iter()
        .map(|variant| {
            let recipe = cluster_barrier_recipe(variant.mode);
            ensure!(
                variant.abi_id == recipe.abi_id,
                "{} must keep reserved ABI ID {}",
                recipe.id,
                recipe.abi_id
            );
            let source_record = format!("int_nvvm_barrier_cluster_{}", recipe.suffix.replace('.', "_"));
            let llvm_symbol = format!("llvm.nvvm.barrier.cluster.{}", recipe.suffix);
            let modifiers: Vec<String> = recipe.suffix.split('.').map(str::to_owned).collect();
            Ok(OverlayIntrinsic {
                id: recipe.id.into(),
                abi_id: variant.abi_id.clone(),
                operation_key: format!("cluster.barrier.{}", recipe.suffix),
                family: "cluster_barrier".into(),
                source: None,
                source_record: Some(source_record),
                rust_module: "cluster".into(),
                rust_name: recipe.id.into(),
                rust_arguments: vec![],
                rust_result: "()".into(),
                safe: false,
                must_use: false,
                safe_allowlist_reason: None,
                public_rust_path: format!("cuda_intrinsics::cluster::{}", recipe.id),
                compatibility_rust_paths: vec![format!("cuda_device::cluster::{}", recipe.id)],
                dialect_op_type: "ClusterBarrierOp".into(),
                dialect_op_name: "nvvm.cluster_barrier".into(),
                dialect_operands: vec![],
                dialect_results: vec![],
                llvm_symbol: Some(llvm_symbol),
                resolved_llvm_symbol: None,
                llvm_arguments: vec![],
                llvm_results: vec![],
                pure: false,
                memory: "read_write".into(),
                convergent: true,
                execution_scope: "cluster".into(),
                minimum_ptx: recipe.minimum_ptx.into(),
                minimum_sm: Some("sm_90".into()),
                ptx_result: "()".into(),
                targets: "all".into(),
                ptx_isa_version: "9.3".into(),
                ptx_isa_section:
                    "Parallel Synchronization and Communication Instructions: barrier.cluster"
                        .into(),
                ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-barrier-cluster".into(),
                lowering: "generated_cluster_barrier".into(),
                backend_lowerings: vec![
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LlvmNvptx,
                        mechanism: BackendLoweringMechanism::TypedNvvm,
                        evidence_profile: admission.llvm_evidence_profile.clone(),
                        targets: None,
                        minimum_ptx: Some(recipe.minimum_ptx.into()),
                        minimum_sm: Some("sm_90".into()),
                    },
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LibNvvm,
                        mechanism: BackendLoweringMechanism::InlinePtx,
                        evidence_profile: admission.libnvvm_evidence_profile.clone(),
                        targets: None,
                        minimum_ptx: Some(recipe.minimum_ptx.into()),
                        minimum_sm: Some("sm_90".into()),
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
                cluster_barrier: Some(ClusterBarrier {
                    mode: recipe.mode,
                    ordering: recipe.ordering,
                    aligned: recipe.aligned,
                }),
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
                    mnemonic: "barrier".into(),
                    modifiers: std::iter::once("cluster".into()).chain(modifiers).collect(),
                    operands: vec![],
                },
                summary: recipe.summary.into(),
            })
        })
        .collect()
}

pub(in crate::resolve) fn validate_cluster_barrier_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let barrier = policy
        .cluster_barrier
        .as_ref()
        .with_context(|| format!("{} has no closed cluster-barrier contract", policy.id))?;
    let recipe = cluster_barrier_recipe(barrier.mode);
    let source_record = format!(
        "int_nvvm_barrier_cluster_{}",
        recipe.suffix.replace('.', "_")
    );
    let llvm_symbol = format!("llvm.nvvm.barrier.cluster.{}", recipe.suffix);
    ensure!(
        barrier.ordering == recipe.ordering
            && barrier.aligned == recipe.aligned
            && policy.id == recipe.id
            && policy.abi_id == recipe.abi_id
            && policy.operation_key == format!("cluster.barrier.{}", recipe.suffix)
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(source_record.as_str())
            && policy.llvm_symbol.as_deref() == Some(llvm_symbol.as_str())
            && policy.resolved_llvm_symbol.is_none(),
        "{} identity or semantics do not match its closed cluster-barrier recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "cluster"
            && policy.rust_name == recipe.id
            && policy.rust_arguments.is_empty()
            && policy.rust_result == "()"
            && !policy.safe
            && !policy.must_use
            && policy.public_rust_path == format!("cuda_intrinsics::cluster::{}", recipe.id)
            && policy.compatibility_rust_paths == [format!("cuda_device::cluster::{}", recipe.id)],
        "{} Rust API does not match its closed cluster-barrier recipe",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == "ClusterBarrierOp"
            && policy.dialect_op_name == "nvvm.cluster_barrier"
            && policy.dialect_operands.is_empty()
            && policy.dialect_results.is_empty()
            && policy.llvm_arguments.is_empty()
            && policy.llvm_results.is_empty()
            && policy.lowering == "generated_cluster_barrier",
        "{} carrier or lowering does not match its closed cluster-barrier recipe",
        policy.id
    );
    ensure!(
        declaration.classes == ["SDPatternOperator", "Intrinsic"]
            && declaration.properties == ["IntrConvergent", "IntrNoCallback"]
            && !policy.pure
            && policy.memory == "read_write"
            && policy.convergent
            && policy.execution_scope == "cluster",
        "{} effects disagree with the imported cluster-barrier declaration",
        policy.id
    );
    ensure!(
        policy.minimum_ptx == recipe.minimum_ptx
            && policy.minimum_sm.as_deref() == Some("sm_90")
            && policy.targets == "all"
            && policy.ptx_result == "()"
            && policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section
                == "Parallel Synchronization and Communication Instructions: barrier.cluster"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-barrier-cluster",
        "{} target floor or PTX provenance changed",
        policy.id
    );
    let expected_modifiers: Vec<_> = std::iter::once("cluster")
        .chain(recipe.suffix.split('.'))
        .collect();
    ensure!(
        policy.expected_ptx.mnemonic == "barrier"
            && policy.expected_ptx.modifiers == expected_modifiers
            && policy.expected_ptx.operands.is_empty(),
        "{} expected PTX does not match its exact cluster-barrier spelling",
        policy.id
    );
    ensure!(
        (recipe.ordering == ClusterBarrierOrdering::Relaxed) == recipe.suffix.contains(".relaxed")
            && recipe.aligned == recipe.suffix.ends_with(".aligned")
            && matches!(
                (recipe.mode, recipe.ordering),
                (
                    ClusterBarrierMode::Arrive | ClusterBarrierMode::ArriveAligned,
                    ClusterBarrierOrdering::Release
                ) | (
                    ClusterBarrierMode::ArriveRelaxed | ClusterBarrierMode::ArriveRelaxedAligned,
                    ClusterBarrierOrdering::Relaxed
                ) | (
                    ClusterBarrierMode::Wait | ClusterBarrierMode::WaitAligned,
                    ClusterBarrierOrdering::Acquire
                )
            ),
        "{} cluster-barrier semantic recipe is inconsistent",
        policy.id
    );
    let backend_pairs: BTreeSet<_> = policy
        .backend_lowerings
        .iter()
        .map(|lowering| (lowering.backend, lowering.mechanism))
        .collect();
    ensure!(
        policy.backend_lowerings.len() == 2
            && backend_pairs
                == BTreeSet::from([
                    (
                        IntrinsicBackend::LlvmNvptx,
                        BackendLoweringMechanism::TypedNvvm,
                    ),
                    (
                        IntrinsicBackend::LibNvvm,
                        BackendLoweringMechanism::InlinePtx,
                    ),
                ])
            && policy.backend_lowerings.iter().all(|lowering| {
                lowering.minimum_ptx.as_deref() == Some(recipe.minimum_ptx)
                    && lowering.minimum_sm.as_deref() == Some("sm_90")
                    && !lowering.evidence_profile.trim().is_empty()
            }),
        "{} must define exactly the reviewed cluster-barrier backend routes",
        policy.id
    );
    ensure_no_other_family_contract(policy, "cluster barrier")?;
    Ok(())
}

#[derive(Clone)]
pub(in crate::resolve) struct ClusterMemoryRecipe {
    operation: ClusterMemoryOperation,
    abi_id: &'static str,
    id: &'static str,
    operation_key: &'static str,
    source_record: Option<&'static str>,
    llvm_symbol: Option<&'static str>,
    ptx_native_instruction: Option<&'static str>,
    rust_arguments: &'static [&'static str],
    rust_result: &'static str,
    compatibility_paths: &'static [&'static str],
    dialect_op_type: &'static str,
    dialect_op_name: &'static str,
    dialect_operands: &'static [&'static str],
    dialect_results: &'static [&'static str],
    llvm_arguments: &'static [&'static str],
    llvm_results: &'static [&'static str],
    adapter: ClusterMemoryAdapter,
    source_contract: ClusterMemorySourceContract,
    memory: &'static str,
    expected_ptx: InstructionPattern,
    inline_ptx: &'static str,
    inline_constraints: &'static str,
    ptx_isa_section: &'static str,
    ptx_isa_anchor: &'static str,
    summary: &'static str,
}

pub(in crate::resolve) fn cluster_memory_recipe(
    operation: ClusterMemoryOperation,
) -> ClusterMemoryRecipe {
    match operation {
        ClusterMemoryOperation::MapSharedRank => ClusterMemoryRecipe {
            operation,
            abi_id: "i0320",
            id: "map_shared_rank",
            operation_key: "cluster.shared_address.map_rank",
            source_record: Some("int_nvvm_mapa_shared_cluster"),
            llvm_symbol: Some("llvm.nvvm.mapa.shared.cluster"),
            ptx_native_instruction: None,
            rust_arguments: &["*const u8", "u32"],
            rust_result: "*const u8",
            compatibility_paths: &[
                "cuda_device::cluster::map_shared_rank",
                "cuda_device::cluster::map_shared_rank_mut",
            ],
            dialect_op_type: "MapaSharedClusterOp",
            dialect_op_name: "nvvm.mapa_shared_cluster",
            dialect_operands: &["ptr", "i32"],
            dialect_results: &["ptr"],
            llvm_arguments: &["shared_ptr", "i32"],
            llvm_results: &["shared_cluster_ptr"],
            adapter: ClusterMemoryAdapter::GenericConstAndMutPointerRankToSamePointer,
            source_contract: ClusterMemorySourceContract::LlvmMapaSharedClusterAs7IdentityInlinePtx,
            memory: "none",
            expected_ptx: InstructionPattern {
                mnemonic: "mapa".into(),
                modifiers: vec!["shared::cluster".into(), "u64".into()],
                operands: vec![
                    OperandPattern::Register,
                    OperandPattern::Register,
                    OperandPattern::Register,
                ],
            },
            inline_ptx: "mapa.shared::cluster.u64 $0, $1, $2;",
            inline_constraints: "=l,l,r",
            ptx_isa_section: "9.7.9.24 Data Movement and Conversion Instructions: mapa",
            ptx_isa_anchor: "data-movement-and-conversion-instructions-mapa",
            summary: "Maps a CTA-shared address to the same offset in another cluster rank.",
        },
        ClusterMemoryOperation::ReadU32 => ClusterMemoryRecipe {
            operation,
            abi_id: "i0321",
            id: "dsmem_read_u32",
            operation_key: "cluster.shared_memory.map_rank_then_read_u32",
            source_record: None,
            llvm_symbol: None,
            ptx_native_instruction: Some("mapa.shared::cluster.u64 + ld.shared::cluster.u32"),
            rust_arguments: &["*const u32", "u32"],
            rust_result: "u32",
            compatibility_paths: &["cuda_device::cluster::dsmem_read_u32"],
            dialect_op_type: "DsmemReadU32Op",
            dialect_op_name: "nvvm.dsmem_read_u32",
            dialect_operands: &["ptr", "i32"],
            dialect_results: &["i32"],
            llvm_arguments: &[],
            llvm_results: &[],
            adapter: ClusterMemoryAdapter::ConstU32PointerRankToU32,
            source_contract: ClusterMemorySourceContract::PtxNativeMapaThenWeakClusterLoad,
            memory: "read",
            expected_ptx: InstructionPattern {
                mnemonic: "ld".into(),
                modifiers: vec!["shared::cluster".into(), "u32".into()],
                operands: vec![OperandPattern::Register, OperandPattern::Address],
            },
            inline_ptx: "{ .reg .u64 %mapped; mapa.shared::cluster.u64 %mapped, $1, $2; ld.shared::cluster.u32 $0, [%mapped]; }",
            inline_constraints: "=r,l,r,~{memory}",
            ptx_isa_section: "9.7.9.8 Data Movement and Conversion Instructions: ld",
            ptx_isa_anchor: "data-movement-and-conversion-instructions-ld",
            summary: "Maps a CTA-shared address to another cluster rank and reads one weak u32 value.",
        },
    }
}

pub(crate) fn cluster_memory_inline_recipe(
    operation: ClusterMemoryOperation,
) -> (&'static str, &'static str) {
    let recipe = cluster_memory_recipe(operation);
    (recipe.inline_ptx, recipe.inline_constraints)
}

pub(in crate::resolve) fn expand_cluster_memory_admission(
    admission: &ClusterMemoryAdmission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "cluster-memory runtime validation may be marked executed only with GPU evidence"
    );
    ensure!(
        !admission.llvm_evidence_profile.trim().is_empty()
            && !admission.libnvvm_evidence_profile.trim().is_empty(),
        "compact cluster-memory admission requires both backend evidence profiles"
    );
    let expected = [
        ClusterMemoryOperation::MapSharedRank,
        ClusterMemoryOperation::ReadU32,
    ];
    ensure!(
        admission.variants.len() == expected.len()
            && admission
                .variants
                .iter()
                .map(|variant| variant.operation)
                .eq(expected),
        "compact cluster-memory admission must list map_shared_rank and read_u32 once in canonical order"
    );

    admission
        .variants
        .iter()
        .map(|variant| {
            let recipe = cluster_memory_recipe(variant.operation);
            ensure!(
                variant.abi_id == recipe.abi_id,
                "{} must keep reserved ABI ID {}",
                recipe.id,
                recipe.abi_id
            );
            let source =
                recipe
                    .ptx_native_instruction
                    .map(|instruction| IntrinsicSource::PtxNative {
                        instruction: instruction.into(),
                    });
            Ok(OverlayIntrinsic {
                id: recipe.id.into(),
                abi_id: variant.abi_id.clone(),
                operation_key: recipe.operation_key.into(),
                family: "cluster_memory".into(),
                source,
                source_record: recipe.source_record.map(Into::into),
                rust_module: "cluster".into(),
                rust_name: recipe.id.into(),
                rust_arguments: recipe
                    .rust_arguments
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                rust_result: recipe.rust_result.into(),
                safe: false,
                must_use: true,
                safe_allowlist_reason: None,
                public_rust_path: format!("cuda_intrinsics::cluster::{}", recipe.id),
                compatibility_rust_paths: recipe
                    .compatibility_paths
                    .iter()
                    .map(|path| (*path).into())
                    .collect(),
                dialect_op_type: recipe.dialect_op_type.into(),
                dialect_op_name: recipe.dialect_op_name.into(),
                dialect_operands: recipe
                    .dialect_operands
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                dialect_results: recipe
                    .dialect_results
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                llvm_symbol: recipe.llvm_symbol.map(Into::into),
                resolved_llvm_symbol: None,
                llvm_arguments: recipe
                    .llvm_arguments
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                llvm_results: recipe
                    .llvm_results
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                pure: false,
                memory: recipe.memory.into(),
                convergent: true,
                execution_scope: "cluster".into(),
                minimum_ptx: "7.8".into(),
                minimum_sm: Some("sm_90".into()),
                ptx_result: recipe.rust_result.into(),
                targets: "all".into(),
                ptx_isa_version: "9.3".into(),
                ptx_isa_section: recipe.ptx_isa_section.into(),
                ptx_isa_url: format!(
                    "https://docs.nvidia.com/cuda/parallel-thread-execution/#{}",
                    recipe.ptx_isa_anchor
                ),
                lowering: "generated_cluster_memory_inline_ptx".into(),
                backend_lowerings: [IntrinsicBackend::LlvmNvptx, IntrinsicBackend::LibNvvm]
                    .into_iter()
                    .map(|backend| OverlayBackendLowering {
                        backend,
                        mechanism: BackendLoweringMechanism::InlinePtx,
                        evidence_profile: match backend {
                            IntrinsicBackend::LlvmNvptx => admission.llvm_evidence_profile.clone(),
                            IntrinsicBackend::LibNvvm => admission.libnvvm_evidence_profile.clone(),
                        },
                        targets: None,
                        minimum_ptx: Some("7.8".into()),
                        minimum_sm: Some("sm_90".into()),
                    })
                    .collect(),
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
                cluster_memory: Some(ClusterMemory {
                    operation: recipe.operation,
                    adapter: recipe.adapter,
                    source_contract: recipe.source_contract,
                    runtime_validation: admission.runtime_validation,
                }),
                clc: None,
                tma: None,
                tcgen05: None,
                ldmatrix_variant: None,
                ldmatrix_safety: None,
                ldmatrix_adapter: None,
                selected_address_space: None,
                expected_ptx: recipe.expected_ptx,
                summary: recipe.summary.into(),
            })
        })
        .collect()
}

pub(in crate::resolve) fn validate_cluster_memory_policy(
    policy: &OverlayIntrinsic,
    source: &IntrinsicSource,
    declaration: Option<&ImportedIntrinsic>,
) -> Result<()> {
    let cluster = policy
        .cluster_memory
        .as_ref()
        .with_context(|| format!("{} has no closed cluster-memory contract", policy.id))?;
    let recipe = cluster_memory_recipe(cluster.operation);
    ensure!(
        cluster.adapter == recipe.adapter
            && cluster.source_contract == recipe.source_contract
            && cluster.runtime_validation == RuntimeValidation::Unexecuted
            && policy.id == recipe.id
            && policy.operation_key == recipe.operation_key,
        "{} does not match its closed cluster-memory identity",
        policy.id
    );
    match cluster.source_contract {
        ClusterMemorySourceContract::LlvmMapaSharedClusterAs7IdentityInlinePtx => {
            let declaration = declaration.context("map_shared_rank requires its LLVM identity")?;
            ensure!(
                matches!(source, IntrinsicSource::LlvmImported { source_record }
                    if source_record == "int_nvvm_mapa_shared_cluster")
                    && policy.source.is_none()
                    && policy.source_record.as_deref() == recipe.source_record
                    && policy.llvm_symbol.as_deref() == recipe.llvm_symbol
                    && declaration.arguments == ["shared_ptr", "i32"]
                    && declaration.results == ["shared_cluster_ptr"]
                    && declaration.properties
                        == ["IntrNoMem", "IntrSpeculatable", "NoCapture<arg0>"],
                "{} must retain the AS7-returning LLVM mapa record as identity only",
                policy.id
            );
            let mut selections = Vec::new();
            for selection in &declaration.selections {
                if selection_matches_policy(policy, selection)? {
                    selections.push(selection);
                }
            }
            ensure!(
                selections.len() == 2
                    && selections
                        .iter()
                        .map(|selection| selection.source_record.as_str())
                        .collect::<BTreeSet<_>>()
                        == BTreeSet::from(["mapa_shared_cluster_64", "mapa_shared_cluster_64i",])
                    && selections.iter().all(|selection| {
                        selection.asm == "mapa.shared::cluster.u64 \t$d, $a, $b;"
                            && selection.predicates
                                == [
                                    "Subtarget->getSmVersion() >= 90",
                                    "Subtarget->getPTXVersion() >= 78",
                                ]
                    }),
                "{} must retain both exact 64-bit mapa selections",
                policy.id
            );
        }
        ClusterMemorySourceContract::PtxNativeMapaThenWeakClusterLoad => ensure!(
            matches!(source, IntrinsicSource::PtxNative { instruction }
                if Some(instruction.as_str()) == recipe.ptx_native_instruction)
                && policy.source_record.is_none()
                && policy.llvm_symbol.is_none()
                && declaration.is_none()
                && policy.llvm_arguments.is_empty()
                && policy.llvm_results.is_empty(),
            "{} must remain a PTX-native mapa plus weak cluster-load composite",
            policy.id
        ),
    }
    ensure!(
        policy.rust_module == "cluster"
            && policy.rust_name == recipe.id
            && policy.rust_arguments == recipe.rust_arguments
            && policy.rust_result == recipe.rust_result
            && !policy.safe
            && policy.must_use
            && policy.public_rust_path == format!("cuda_intrinsics::cluster::{}", recipe.id)
            && policy.compatibility_rust_paths == recipe.compatibility_paths,
        "{} Rust API or compatibility paths changed",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == recipe.dialect_operands
            && policy.dialect_results == recipe.dialect_results
            && policy.llvm_arguments == recipe.llvm_arguments
            && policy.llvm_results == recipe.llvm_results
            && policy.lowering == "generated_cluster_memory_inline_ptx",
        "{} dialect carrier or AS7 source boundary changed",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == recipe.memory
            && policy.convergent
            && policy.execution_scope == "cluster",
        "{} effects changed",
        policy.id
    );
    ensure!(
        policy.minimum_ptx == "7.8"
            && policy.minimum_sm.as_deref() == Some("sm_90")
            && policy.targets == "all"
            && policy.ptx_result == recipe.rust_result
            && policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == recipe.ptx_isa_section
            && policy.ptx_isa_url
                == format!(
                    "https://docs.nvidia.com/cuda/parallel-thread-execution/#{}",
                    recipe.ptx_isa_anchor
                ),
        "{} PTX provenance or target floor changed",
        policy.id
    );
    ensure!(
        policy.expected_ptx == recipe.expected_ptx,
        "{} expected PTX changed",
        policy.id
    );
    ensure_exact_inline_ptx_backends(
        policy,
        [
            (IntrinsicBackend::LlvmNvptx, "7.8", Some("sm_90")),
            (IntrinsicBackend::LibNvvm, "7.8", Some("sm_90")),
        ],
        "cluster-memory",
    )?;
    ensure_no_other_family_contract(policy, "cluster-memory")?;
    Ok(())
}
