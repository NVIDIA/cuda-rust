/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, DebugControl, DebugControlAdapter, DebugControlAdmission,
    DebugControlOperation, ImportedIntrinsic, IntrinsicBackend, IntrinsicSource,
    OverlayBackendLowering, OverlayIntrinsic, RuntimeValidation, WgmmaControl, WgmmaControlAdapter,
    WgmmaControlAdmission, WgmmaControlMode, WgmmaControlParticipation,
};
use crate::ptx::{InstructionPattern, OperandPattern};
use anyhow::{Context, Result, ensure};
use std::collections::BTreeSet;

use crate::resolve::abi_ledger::*;
use crate::resolve::guards::*;

#[derive(Clone, Copy)]
pub(in crate::resolve) struct DebugControlRecipe {
    id: &'static str,
    operation_key: &'static str,
    rust_name: &'static str,
    rust_arguments: &'static [&'static str],
    rust_result: &'static str,
    compatibility_path: &'static str,
    op_type: &'static str,
    op_name: &'static str,
    instruction: &'static str,
    minimum_ptx: &'static str,
    minimum_sm: Option<&'static str>,
    section: &'static str,
    anchor: &'static str,
    adapter: DebugControlAdapter,
    summary: &'static str,
}

pub(in crate::resolve) fn debug_control_recipe(
    operation: DebugControlOperation,
) -> DebugControlRecipe {
    match operation {
        DebugControlOperation::Trap => DebugControlRecipe {
            id: "trap",
            operation_key: "debug.control.trap",
            rust_name: "trap",
            rust_arguments: &[],
            rust_result: "!",
            compatibility_path: "cuda_device::debug::trap",
            op_type: "TrapOp",
            op_name: "nvvm.trap",
            instruction: "trap",
            minimum_ptx: "1.0",
            minimum_sm: None,
            section: "9.7.20.4 Miscellaneous Instructions: trap",
            anchor: "miscellaneous-instructions-trap",
            adapter: DebugControlAdapter::Direct,
            summary: "Aborts device execution and reports an interrupt to the host.",
        },
        DebugControlOperation::Breakpoint => DebugControlRecipe {
            id: "breakpoint",
            operation_key: "debug.control.breakpoint",
            rust_name: "breakpoint",
            rust_arguments: &[],
            rust_result: "()",
            compatibility_path: "cuda_device::debug::breakpoint",
            op_type: "BreakpointOp",
            op_name: "nvvm.brkpt",
            instruction: "brkpt",
            minimum_ptx: "1.0",
            minimum_sm: Some("sm_11"),
            section: "9.7.20.1 Miscellaneous Instructions: brkpt",
            anchor: "miscellaneous-instructions-brkpt",
            adapter: DebugControlAdapter::Direct,
            summary: "Suspends device execution for a debugger breakpoint.",
        },
        DebugControlOperation::Pmevent => DebugControlRecipe {
            id: "pmevent",
            operation_key: "debug.profiler.event",
            rust_name: "pmevent",
            rust_arguments: &["u32"],
            rust_result: "()",
            compatibility_path: "cuda_device::debug::__prof_trigger",
            op_type: "PmEventOp",
            op_name: "nvvm.pmevent",
            instruction: "pmevent",
            minimum_ptx: "1.4",
            minimum_sm: None,
            section: "9.7.20.3 Miscellaneous Instructions: pmevent",
            anchor: "miscellaneous-instructions-pmevent",
            adapter: DebugControlAdapter::ConstGenericToImmediateU32,
            summary: "Triggers one compile-time-selected performance monitor event.",
        },
    }
}

#[derive(Clone, Copy)]
pub(in crate::resolve) struct WgmmaControlRecipe {
    pub(in crate::resolve) mode: WgmmaControlMode,
    pub(in crate::resolve) abi_id: &'static str,
    pub(in crate::resolve) id: &'static str,
    pub(in crate::resolve) operation_key: &'static str,
    pub(in crate::resolve) source_record: &'static str,
    pub(in crate::resolve) selection_record: &'static str,
    pub(in crate::resolve) llvm_symbol: &'static str,
    pub(in crate::resolve) compatibility_path: &'static str,
    pub(in crate::resolve) dialect_op_type: &'static str,
    pub(in crate::resolve) dialect_op_name: &'static str,
    pub(in crate::resolve) suffix: &'static str,
    pub(in crate::resolve) adapter: WgmmaControlAdapter,
    pub(in crate::resolve) summary: &'static str,
}

pub(in crate::resolve) fn wgmma_control_recipe(mode: WgmmaControlMode) -> WgmmaControlRecipe {
    match mode {
        WgmmaControlMode::Fence => WgmmaControlRecipe {
            mode,
            abi_id: "i0317",
            id: "wgmma_fence",
            operation_key: "wgmma.control.fence.sync.aligned",
            source_record: "int_nvvm_wgmma_fence_sync_aligned",
            selection_record: "WGMMA_FENCE_SYNC_ALIGNED",
            llvm_symbol: "llvm.nvvm.wgmma.fence.sync.aligned",
            compatibility_path: "cuda_device::wgmma::wgmma_fence",
            dialect_op_type: "WgmmaFenceSyncAlignedOp",
            dialect_op_name: "nvvm.wgmma_fence_sync_aligned",
            suffix: "fence.sync.aligned",
            adapter: WgmmaControlAdapter::NoArguments,
            summary: "Orders register accesses before later WGMMA operations.",
        },
        WgmmaControlMode::CommitGroup => WgmmaControlRecipe {
            mode,
            abi_id: "i0318",
            id: "wgmma_commit_group",
            operation_key: "wgmma.control.commit_group.sync.aligned",
            source_record: "int_nvvm_wgmma_commit_group_sync_aligned",
            selection_record: "WGMMA_COMMIT_GROUP_SYNC_ALIGNED",
            llvm_symbol: "llvm.nvvm.wgmma.commit_group.sync.aligned",
            compatibility_path: "cuda_device::wgmma::wgmma_commit_group",
            dialect_op_type: "WgmmaCommitGroupSyncAlignedOp",
            dialect_op_name: "nvvm.wgmma_commit_group_sync_aligned",
            suffix: "commit_group.sync.aligned",
            adapter: WgmmaControlAdapter::NoArguments,
            summary: "Commits prior uncommitted WGMMA operations as one group.",
        },
        WgmmaControlMode::WaitGroup => WgmmaControlRecipe {
            mode,
            abi_id: "i0319",
            id: "wgmma_wait_group",
            operation_key: "wgmma.control.wait_group.sync.aligned",
            source_record: "int_nvvm_wgmma_wait_group_sync_aligned",
            selection_record: "WGMMA_WAIT_GROUP_SYNC_ALIGNED",
            llvm_symbol: "llvm.nvvm.wgmma.wait_group.sync.aligned",
            compatibility_path: "cuda_device::wgmma::__wgmma_wait_group",
            dialect_op_type: "WgmmaWaitGroupSyncAlignedOp",
            dialect_op_name: "nvvm.wgmma_wait_group_sync_aligned",
            suffix: "wait_group.sync.aligned",
            adapter: WgmmaControlAdapter::ConstGenericU32ToI64Immediate,
            summary: "Waits until at most the requested number of WGMMA groups remain pending.",
        },
    }
}

pub(in crate::resolve) fn expand_debug_control_admission(
    admission: &DebugControlAdmission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "debug-control runtime validation may be marked executed only with GPU evidence"
    );
    ensure!(
        !admission.llvm_evidence_profile.trim().is_empty()
            && !admission.libnvvm_evidence_profile.trim().is_empty(),
        "compact debug-control admission requires both backend evidence profiles"
    );
    ensure!(
        admission.operations
            == [
                DebugControlOperation::Trap,
                DebugControlOperation::Breakpoint,
                DebugControlOperation::Pmevent,
            ],
        "compact debug-control admission must list trap, breakpoint, and pmevent exactly once in canonical order"
    );
    ensure!(
        admission.abi_ids.len() == admission.operations.len(),
        "pending debug-control admission needs exactly three ABI IDs before aggregation"
    );
    let unique_abi_ids = admission.abi_ids.iter().collect::<BTreeSet<_>>();
    ensure!(
        unique_abi_ids.len() == admission.abi_ids.len(),
        "debug-control ABI IDs must be unique"
    );

    admission
        .operations
        .iter()
        .zip(&admission.abi_ids)
        .map(|(&operation, abi_id)| {
            validate_abi_id(abi_id)?;
            let recipe = debug_control_recipe(operation);
            let immediate = operation == DebugControlOperation::Pmevent;
            Ok(OverlayIntrinsic {
                id: recipe.id.into(),
                abi_id: abi_id.clone(),
                operation_key: recipe.operation_key.into(),
                family: "debug_control".into(),
                source: Some(IntrinsicSource::PtxNative {
                    instruction: recipe.instruction.into(),
                }),
                source_record: None,
                rust_module: "debug".into(),
                rust_name: recipe.rust_name.into(),
                rust_arguments: recipe
                    .rust_arguments
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                rust_result: recipe.rust_result.into(),
                safe: true,
                must_use: false,
                safe_allowlist_reason: Some(
                    match operation {
                        DebugControlOperation::Trap => {
                            "aborting the kernel has no memory-safety preconditions"
                        }
                        DebugControlOperation::Breakpoint => {
                            "requesting a debugger breakpoint has no memory-safety preconditions"
                        }
                        DebugControlOperation::Pmevent => {
                            "the importer accepts only the documented immediate event range"
                        }
                    }
                    .into(),
                ),
                public_rust_path: format!("cuda_intrinsics::debug::{}", recipe.rust_name),
                compatibility_rust_paths: vec![recipe.compatibility_path.into()],
                dialect_op_type: recipe.op_type.into(),
                dialect_op_name: recipe.op_name.into(),
                dialect_operands: vec![],
                dialect_results: vec![],
                llvm_symbol: None,
                resolved_llvm_symbol: None,
                llvm_arguments: vec![],
                llvm_results: vec![],
                pure: false,
                memory: "none".into(),
                convergent: false,
                execution_scope: "thread".into(),
                minimum_ptx: recipe.minimum_ptx.into(),
                minimum_sm: recipe.minimum_sm.map(Into::into),
                ptx_result: "()".into(),
                targets: "all".into(),
                ptx_isa_version: "9.3".into(),
                ptx_isa_section: recipe.section.into(),
                ptx_isa_url: format!(
                    "https://docs.nvidia.com/cuda/parallel-thread-execution/#{}",
                    recipe.anchor
                ),
                lowering: "generated_debug_control".into(),
                backend_lowerings: vec![
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LlvmNvptx,
                        mechanism: BackendLoweringMechanism::InlinePtx,
                        evidence_profile: admission.llvm_evidence_profile.clone(),
                        targets: None,
                        minimum_ptx: Some("3.2".into()),
                        minimum_sm: Some("sm_20".into()),
                    },
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LibNvvm,
                        mechanism: BackendLoweringMechanism::InlinePtx,
                        evidence_profile: admission.libnvvm_evidence_profile.clone(),
                        targets: None,
                        // PTX floor is the inline-PTX mechanism floor, same as
                        // the LlvmNvptx entry above; the hardware floor stays at
                        // the probed sm_75 (backend-codegen evidence must sit
                        // exactly at the hardware floor, and CUDA 13 cannot
                        // probe older targets). Writing the probe PTX version
                        // (9.3) here made every panic path unbuildable on the
                        // NVVM-IR route: the floor exceeded the newest PTX
                        // version cuda-oxide can request.
                        minimum_ptx: Some("3.2".into()),
                        minimum_sm: Some("sm_75".into()),
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
                debug_control: Some(DebugControl {
                    operation,
                    adapter: recipe.adapter,
                    runtime_validation: admission.runtime_validation,
                }),
                cluster_memory: None,
                clc: None,
                tma: None,
                tcgen05: None,
                ldmatrix_variant: None,
                ldmatrix_safety: None,
                ldmatrix_adapter: None,
                selected_address_space: None,
                expected_ptx: InstructionPattern {
                    mnemonic: recipe.instruction.into(),
                    modifiers: vec![],
                    operands: if immediate {
                        vec![OperandPattern::Immediate]
                    } else {
                        vec![]
                    },
                },
                summary: recipe.summary.into(),
            })
        })
        .collect()
}

pub(in crate::resolve) fn validate_debug_control_policy(
    policy: &OverlayIntrinsic,
    source: &IntrinsicSource,
) -> Result<()> {
    let debug = policy
        .debug_control
        .as_ref()
        .with_context(|| format!("{} has no closed debug-control contract", policy.id))?;
    let recipe = debug_control_recipe(debug.operation);
    let immediate = debug.operation == DebugControlOperation::Pmevent;
    ensure!(
        debug.adapter == recipe.adapter
            && debug.runtime_validation == RuntimeValidation::Unexecuted
            && policy.id == recipe.id
            && policy.operation_key == recipe.operation_key
            && source
                == &IntrinsicSource::PtxNative {
                    instruction: recipe.instruction.into(),
                }
            && policy.source_record.is_none()
            && policy.llvm_symbol.is_none()
            && policy.resolved_llvm_symbol.is_none()
            && policy.llvm_arguments.is_empty()
            && policy.llvm_results.is_empty(),
        "{} identity must remain PTX-native and match its closed debug-control recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "debug"
            && policy.rust_name == recipe.rust_name
            && policy.rust_arguments == recipe.rust_arguments
            && policy.rust_result == recipe.rust_result
            && policy.safe
            && !policy.must_use
            && policy.public_rust_path == format!("cuda_intrinsics::debug::{}", recipe.rust_name)
            && policy.compatibility_rust_paths == [recipe.compatibility_path],
        "{} Rust API or compatibility adapter changed",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.op_type
            && policy.dialect_op_name == recipe.op_name
            && policy.dialect_operands.is_empty()
            && policy.dialect_results.is_empty()
            && policy.lowering == "generated_debug_control",
        "{} dialect carrier or lowering changed",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "none"
            && !policy.convergent
            && policy.execution_scope == "thread",
        "{} debug-control effects changed",
        policy.id
    );
    ensure!(
        policy.minimum_ptx == recipe.minimum_ptx
            && policy.minimum_sm.as_deref() == recipe.minimum_sm
            && policy.targets == "all"
            && policy.ptx_result == "()"
            && policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == recipe.section
            && policy.ptx_isa_url
                == format!(
                    "https://docs.nvidia.com/cuda/parallel-thread-execution/#{}",
                    recipe.anchor
                ),
        "{} native target floor or PTX provenance changed",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == recipe.instruction
            && policy.expected_ptx.modifiers.is_empty()
            && policy.expected_ptx.operands
                == if immediate {
                    vec![OperandPattern::Immediate]
                } else {
                    vec![]
                },
        "{} expected PTX changed",
        policy.id
    );
    ensure_exact_inline_ptx_backends(
        policy,
        [
            (IntrinsicBackend::LlvmNvptx, "3.2", Some("sm_20")),
            (IntrinsicBackend::LibNvvm, "3.2", Some("sm_75")),
        ],
        "debug-control",
    )?;
    ensure_no_other_family_contract(policy, "debug-control")?;
    Ok(())
}

pub(in crate::resolve) fn expand_wgmma_control_admission(
    admission: &WgmmaControlAdmission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "WGMMA-control runtime validation may be marked executed only with GPU evidence"
    );
    ensure!(
        !admission.llvm_evidence_profile.trim().is_empty()
            && !admission.libnvvm_evidence_profile.trim().is_empty(),
        "compact WGMMA-control admission requires both backend evidence profiles"
    );
    let expected_modes = [
        WgmmaControlMode::Fence,
        WgmmaControlMode::CommitGroup,
        WgmmaControlMode::WaitGroup,
    ];
    let actual_modes = admission
        .variants
        .iter()
        .map(|variant| variant.mode)
        .collect::<Vec<_>>();
    ensure!(
        actual_modes == expected_modes,
        "compact WGMMA-control admission must contain each reviewed mode once in canonical order"
    );

    admission
        .variants
        .iter()
        .map(|variant| {
            let recipe = wgmma_control_recipe(variant.mode);
            ensure!(
                variant.abi_id == recipe.abi_id,
                "{} must keep reserved ABI ID {}",
                recipe.id,
                recipe.abi_id
            );
            let wait = recipe.mode == WgmmaControlMode::WaitGroup;
            Ok(OverlayIntrinsic {
                id: recipe.id.into(),
                abi_id: variant.abi_id.clone(),
                operation_key: recipe.operation_key.into(),
                family: "wgmma_control".into(),
                source: None,
                source_record: Some(recipe.source_record.into()),
                rust_module: "wgmma".into(),
                rust_name: recipe.id.into(),
                rust_arguments: if wait { vec!["u64".into()] } else { vec![] },
                rust_result: "()".into(),
                safe: false,
                must_use: false,
                safe_allowlist_reason: None,
                public_rust_path: format!("cuda_intrinsics::wgmma::{}", recipe.id),
                compatibility_rust_paths: vec![recipe.compatibility_path.into()],
                dialect_op_type: recipe.dialect_op_type.into(),
                dialect_op_name: recipe.dialect_op_name.into(),
                dialect_operands: if wait { vec!["i64".into()] } else { vec![] },
                dialect_results: vec![],
                llvm_symbol: Some(recipe.llvm_symbol.into()),
                resolved_llvm_symbol: None,
                llvm_arguments: if wait { vec!["i64".into()] } else { vec![] },
                llvm_results: vec![],
                pure: false,
                memory: "read_write".into(),
                convergent: true,
                execution_scope: "warpgroup".into(),
                minimum_ptx: "8.0".into(),
                minimum_sm: None,
                ptx_result: "()".into(),
                targets: "sm_90a".into(),
                ptx_isa_version: "9.3".into(),
                ptx_isa_section:
                    "Asynchronous Warpgroup Level Matrix Instructions: WGMMA control".into(),
                ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions".into(),
                lowering: "generated_wgmma_control".into(),
                backend_lowerings: vec![
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LlvmNvptx,
                        mechanism: BackendLoweringMechanism::TypedNvvm,
                        evidence_profile: admission.llvm_evidence_profile.clone(),
                        targets: None,
                        minimum_ptx: Some("8.0".into()),
                        minimum_sm: None,
                    },
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LibNvvm,
                        mechanism: BackendLoweringMechanism::InlinePtx,
                        evidence_profile: admission.libnvvm_evidence_profile.clone(),
                        targets: None,
                        minimum_ptx: Some("8.0".into()),
                        minimum_sm: None,
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
                wgmma_control: Some(WgmmaControl {
                    mode: recipe.mode,
                    adapter: recipe.adapter,
                    participation: WgmmaControlParticipation::WarpgroupAllThreadsSameInstruction,
                }),
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
                    mnemonic: "wgmma".into(),
                    modifiers: recipe.suffix.split('.').map(str::to_owned).collect(),
                    operands: if wait {
                        vec![OperandPattern::Immediate]
                    } else {
                        vec![]
                    },
                },
                summary: recipe.summary.into(),
            })
        })
        .collect()
}

pub(in crate::resolve) fn validate_wgmma_control_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let control = policy
        .wgmma_control
        .as_ref()
        .with_context(|| format!("{} has no closed WGMMA-control contract", policy.id))?;
    let recipe = wgmma_control_recipe(control.mode);
    let wait = recipe.mode == WgmmaControlMode::WaitGroup;
    ensure!(
        control.adapter == recipe.adapter
            && control.participation
                == WgmmaControlParticipation::WarpgroupAllThreadsSameInstruction
            && policy.id == recipe.id
            && policy.abi_id == recipe.abi_id
            && policy.operation_key == recipe.operation_key
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(recipe.source_record)
            && policy.llvm_symbol.as_deref() == Some(recipe.llvm_symbol)
            && policy.resolved_llvm_symbol.is_none(),
        "{} identity or semantics do not match its closed WGMMA-control recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "wgmma"
            && policy.rust_name == recipe.id
            && policy.rust_arguments
                == if wait {
                    vec!["u64"]
                } else {
                    Vec::<&str>::new()
                }
            && policy.rust_result == "()"
            && !policy.safe
            && !policy.must_use
            && policy.public_rust_path == format!("cuda_intrinsics::wgmma::{}", recipe.id)
            && policy.compatibility_rust_paths == [recipe.compatibility_path],
        "{} Rust API does not match its closed WGMMA-control recipe",
        policy.id
    );
    let expected_arguments = if wait {
        vec!["i64"]
    } else {
        Vec::<&str>::new()
    };
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == expected_arguments
            && policy.dialect_results.is_empty()
            && policy.llvm_arguments == expected_arguments
            && policy.llvm_results.is_empty()
            && policy.lowering == "generated_wgmma_control",
        "{} carrier or lowering does not match its closed WGMMA-control recipe",
        policy.id
    );
    let expected_properties = if wait {
        vec!["ImmArg<arg0>", "IntrConvergent"]
    } else {
        vec!["IntrConvergent"]
    };
    ensure!(
        declaration.classes == ["SDPatternOperator", "Intrinsic"]
            && declaration.properties == expected_properties
            && !policy.pure
            && policy.memory == "read_write"
            && policy.convergent
            && policy.execution_scope == "warpgroup",
        "{} effects disagree with the imported WGMMA-control declaration",
        policy.id
    );
    ensure!(
        policy.minimum_ptx == "8.0"
            && policy.minimum_sm.is_none()
            && policy.targets == "sm_90a"
            && policy.ptx_result == "()"
            && policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions",
        "{} target floor or PTX provenance changed",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == "wgmma"
            && policy.expected_ptx.modifiers == recipe.suffix.split('.').collect::<Vec<_>>()
            && policy.expected_ptx.operands
                == if wait {
                    vec![OperandPattern::Immediate]
                } else {
                    vec![]
                },
        "{} expected PTX does not match its exact WGMMA-control spelling",
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
                lowering.minimum_ptx.as_deref() == Some("8.0")
                    && lowering.minimum_sm.is_none()
                    && !lowering.evidence_profile.trim().is_empty()
            }),
        "{} must define exactly the reviewed WGMMA-control backend routes",
        policy.id
    );
    ensure_no_other_family_contract(policy, "WGMMA control")?;
    Ok(())
}
