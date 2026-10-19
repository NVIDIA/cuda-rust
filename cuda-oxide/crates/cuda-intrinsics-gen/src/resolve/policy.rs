/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, ClcOperation, CpAsyncSourceSize, ImportedIntrinsic, IntrinsicSource,
    MbarrierExtendedOperation, OverlayIntrinsic, ScalarMathOperation, SpecialRegisterKind,
    Tcgen05SourceContract, TmaOperation,
};
use anyhow::{Context, Result, bail, ensure};

use super::abi_ledger::*;
use super::families::*;
use super::guards::*;
use super::targets::*;

pub(super) fn resolve_policy_source(policy: &OverlayIntrinsic) -> Result<IntrinsicSource> {
    match (&policy.source, &policy.source_record) {
        (None, Some(source_record)) => Ok(IntrinsicSource::LlvmImported {
            source_record: source_record.clone(),
        }),
        (Some(source @ IntrinsicSource::PtxNative { .. }), None) => Ok(source.clone()),
        (Some(IntrinsicSource::LlvmImported { source_record }), None) => {
            ensure!(
                !source_record.trim().is_empty(),
                "{} has an empty imported LLVM source record",
                policy.id
            );
            Ok(IntrinsicSource::LlvmImported {
                source_record: source_record.clone(),
            })
        }
        (Some(_), Some(_)) => bail!(
            "{} mixes tagged source provenance with the legacy source_record field",
            policy.id
        ),
        (None, None) => bail!("{} has no intrinsic source provenance", policy.id),
    }
}

pub(super) fn validate_policy(
    policy: &OverlayIntrinsic,
    source: &IntrinsicSource,
    declaration: Option<&ImportedIntrinsic>,
    intrinsic_abi: u32,
) -> Result<()> {
    validate_abi_id(&policy.abi_id)?;
    parse_ptx_version(&policy.minimum_ptx, &policy.id)?;
    parse_hardware_target(policy)?;
    validate_arch_introduction_floor(policy)?;
    policy.expected_ptx.validate().map_err(|reason| {
        anyhow::anyhow!(
            "{} has an invalid expected PTX pattern: {reason}",
            policy.id
        )
    })?;
    let public_path = format!(
        "cuda_intrinsics::{}::{}",
        policy.rust_module, policy.rust_name
    );
    ensure!(
        policy.public_rust_path == public_path,
        "{} public Rust path must be {}",
        policy.id,
        public_path
    );
    let canonical_path = canonical_rust_path(intrinsic_abi, &policy.abi_id);
    ensure!(
        canonical_path != policy.public_rust_path
            && !policy
                .compatibility_rust_paths
                .iter()
                .any(|path| path == &canonical_path || path == &policy.public_rust_path),
        "{} must keep canonical, public, and compatibility Rust paths distinct",
        policy.id
    );
    match (source, declaration) {
        (IntrinsicSource::LlvmImported { .. }, Some(declaration)) => {
            ensure!(
                policy.llvm_symbol.as_deref() == Some(declaration.llvm_name.as_str()),
                "{} LLVM symbol mismatch: imported {}, overlay {:?}",
                policy.id,
                declaration.llvm_name,
                policy.llvm_symbol
            );
            ensure!(
                declaration.arguments == policy.llvm_arguments,
                "{} LLVM argument signature mismatch: imported {:?}, overlay {:?}",
                policy.id,
                declaration.arguments,
                policy.llvm_arguments
            );
            ensure!(
                declaration.results == policy.llvm_results,
                "{} LLVM result signature mismatch: imported {:?}, overlay {:?}",
                policy.id,
                declaration.results,
                policy.llvm_results
            );
        }
        (IntrinsicSource::PtxNative { instruction }, None) => ensure!(
            !instruction.trim().is_empty()
                && policy.llvm_symbol.is_none()
                && policy.resolved_llvm_symbol.is_none()
                && policy.llvm_arguments.is_empty()
                && policy.llvm_results.is_empty(),
            "{} PTX-native source must not invent LLVM source facts",
            policy.id
        ),
        _ => bail!(
            "{} source kind and imported declaration disagree",
            policy.id
        ),
    }
    match policy.family.as_str() {
        "sreg" => validate_sreg_policy(policy, source, declaration)?,
        "ldmatrix" => validate_ldmatrix_policy(
            policy,
            declaration.context("ldmatrix requires imported LLVM declaration")?,
        )?,
        "stmatrix" => validate_stmatrix_policy(
            policy,
            declaration.context("stmatrix requires imported LLVM declaration")?,
        )?,
        "packed_atomic" => validate_packed_atomic_policy(policy, source)?,
        "redux" => validate_redux_policy(
            policy,
            declaration.context("redux requires imported LLVM declaration")?,
        )?,
        "dotprod" => validate_dot_product_policy(
            policy,
            declaration.context("dotprod requires imported LLVM declaration")?,
        )?,
        "sync" => validate_sync_policy(
            policy,
            declaration.context("sync requires imported LLVM declaration")?,
        )?,
        "vote" => validate_vote_policy(
            policy,
            declaration.context("vote requires imported LLVM declaration")?,
        )?,
        "active_mask" => validate_active_mask_policy(
            policy,
            declaration.context("active_mask requires imported LLVM declaration")?,
        )?,
        "warp_match" => validate_warp_match_policy(
            policy,
            declaration.context("warp_match requires imported LLVM declaration")?,
        )?,
        "elect" => validate_elect_policy(
            policy,
            declaration.context("elect requires imported LLVM declaration")?,
        )?,
        "warp_barrier" => validate_warp_barrier_policy(
            policy,
            declaration.context("warp_barrier requires imported LLVM declaration")?,
        )?,
        "warp_shuffle" => validate_warp_shuffle_policy(policy, declaration)?,
        "packed_alu" => validate_packed_alu_policy(policy, source, declaration)?,
        "integer_minmax" => validate_integer_minmax_policy(policy, source, declaration)?,
        "packed_conversion" => validate_packed_conversion_policy(policy, source, declaration)?,
        "scalar_conversion" => validate_scalar_conversion_policy(
            policy,
            declaration.context("scalar_conversion requires imported LLVM declaration")?,
        )?,
        "scalar_arithmetic" => validate_scalar_arithmetic_policy(
            policy,
            declaration.context("scalar_arithmetic requires imported LLVM declaration")?,
        )?,
        "scalar_math" => validate_scalar_math_policy(policy, declaration)?,
        "extended_minmax" => validate_extended_minmax_policy(
            policy,
            declaration.context("extended_minmax requires imported LLVM declaration")?,
        )?,
        "cp_async_copy" => validate_cp_async_copy_policy(
            policy,
            declaration.context("cp_async_copy requires imported LLVM declaration")?,
        )?,
        "cp_async_control" => validate_cp_async_control_policy(
            policy,
            declaration.context("cp_async_control requires imported LLVM declaration")?,
        )?,
        "cp_async_mbarrier" => validate_cp_async_mbarrier_policy(
            policy,
            declaration.context("cp_async_mbarrier requires imported LLVM declaration")?,
        )?,
        "mbarrier_basic" => validate_mbarrier_basic_policy(
            policy,
            declaration.context("mbarrier_basic requires imported LLVM declaration")?,
        )?,
        "movmatrix" => validate_movmatrix_policy(policy, source)?,
        "mbarrier_extended" => validate_mbarrier_extended_policy(policy, source, declaration)?,
        "register_mma" => validate_register_mma_policy(
            policy,
            declaration.context("register_mma requires imported LLVM declaration")?,
        )?,
        "sparse_mma" => validate_sparse_mma_policy(
            policy,
            declaration.context("sparse_mma requires imported LLVM declaration")?,
        )?,
        "prmt" => validate_prmt_policy(
            policy,
            declaration.context("prmt requires imported LLVM declaration")?,
        )?,
        "cluster_barrier" => validate_cluster_barrier_policy(
            policy,
            declaration.context("cluster_barrier requires imported LLVM declaration")?,
        )?,
        "debug_control" => validate_debug_control_policy(policy, source)?,
        "cluster_memory" => validate_cluster_memory_policy(policy, source, declaration)?,
        "clc" => validate_clc_policy(
            policy,
            declaration.context("clc requires imported LLVM declaration")?,
        )?,
        "wgmma_control" => validate_wgmma_control_policy(
            policy,
            declaration.context("wgmma_control requires imported LLVM declaration")?,
        )?,
        "tma" => validate_tma_policy(
            policy,
            declaration.context("tma requires imported LLVM declaration")?,
        )?,
        "counted_barrier" | "grid_dependency" | "register_control" => {
            validate_execution_control_policy(
                policy,
                declaration.context("execution-control requires imported LLVM declaration")?,
            )?
        }
        "tcgen05" => validate_tcgen05_policy(
            policy,
            declaration.context("tcgen05 requires imported LLVM declaration")?,
        )?,
        family => bail!("{} uses unsupported generated family {family:?}", policy.id),
    }
    ensure!(
        (policy.family == "movmatrix") == policy.movmatrix.is_some(),
        "{} mixes the movmatrix contract with another generated family",
        policy.id
    );
    ensure!(
        (policy.family == "mbarrier_extended") == policy.mbarrier_extended.is_some(),
        "{} mixes the extended-mbarrier contract with another generated family",
        policy.id
    );
    ensure!(
        (policy.family == "register_mma") == policy.register_mma.is_some(),
        "{} mixes the register-MMA contract with another generated family",
        policy.id
    );
    ensure!(
        (policy.family == "sparse_mma") == policy.sparse_mma.is_some(),
        "{} mixes the sparse-MMA contract with another generated family",
        policy.id
    );
    ensure!(
        (policy.family == "prmt") == policy.prmt.is_some(),
        "{} mixes the prmt contract with another generated family",
        policy.id
    );
    ensure!(
        (policy.family == "cluster_barrier") == policy.cluster_barrier.is_some(),
        "{} mixes the cluster-barrier contract with another generated family",
        policy.id
    );
    ensure!(
        (policy.family == "wgmma_control") == policy.wgmma_control.is_some(),
        "{} mixes the WGMMA-control contract with another generated family",
        policy.id
    );
    ensure!(
        (policy.family == "tma") == policy.tma.is_some(),
        "{} mixes the TMA contract with another generated family",
        policy.id
    );
    ensure!(
        policy.special_register.is_none() || policy.family == "sreg",
        "{} mixes the special-register contract with another generated family",
        policy.id
    );
    ensure!(
        (policy.family == "debug_control") == policy.debug_control.is_some(),
        "{} mixes the debug-control contract with another generated family",
        policy.id
    );
    ensure!(
        (policy.family == "cluster_memory") == policy.cluster_memory.is_some(),
        "{} mixes the cluster-memory contract with another generated family",
        policy.id
    );
    ensure!(
        (policy.family == "clc") == policy.clc.is_some(),
        "{} mixes the CLC contract with another generated family",
        policy.id
    );
    ensure!(
        (policy.family == "tma") == policy.tma.is_some(),
        "{} mixes the TMA contract with another generated family",
        policy.id
    );
    ensure!(
        (policy.family == "tcgen05") == policy.tcgen05.is_some(),
        "{} mixes the tcgen05 contract with another generated family",
        policy.id
    );
    ensure!(
        (policy.family == "scalar_conversion") == policy.scalar_conversion.is_some(),
        "{} mixes the scalar-conversion contract with another generated family",
        policy.id
    );
    ensure!(
        (policy.family == "scalar_arithmetic") == policy.scalar_arithmetic.is_some(),
        "{} mixes the scalar-arithmetic contract with another generated family",
        policy.id
    );
    ensure!(
        (policy.family == "scalar_math") == policy.scalar_math.is_some(),
        "{} mixes the scalar-math contract with another generated family",
        policy.id
    );
    ensure!(
        (policy.family == "extended_minmax") == policy.extended_minmax.is_some(),
        "{} mixes the extended-minmax contract with another generated family",
        policy.id
    );
    ensure!(
        !policy.execution_scope.trim().is_empty(),
        "{} has no execution scope",
        policy.id
    );
    ensure!(
        !policy.ptx_isa_version.trim().is_empty()
            && !policy.ptx_isa_section.trim().is_empty()
            && policy.ptx_isa_url.starts_with("https://docs.nvidia.com/"),
        "{} has incomplete or non-authoritative PTX ISA provenance",
        policy.id
    );
    match (policy.safe, policy.safe_allowlist_reason.as_deref()) {
        (true, Some(reason)) if !reason.trim().is_empty() => {}
        (true, _) => bail!(
            "{} is safe but has no nonempty safe_allowlist_reason",
            policy.id
        ),
        (false, Some(reason)) if !reason.trim().is_empty() => bail!(
            "{} is unsafe but has a safe_allowlist_reason; safe exceptions apply only to safe items",
            policy.id
        ),
        (false, _) => {}
    }
    if let Some(declaration) = declaration {
        if policy.pure {
            ensure!(
                declaration
                    .classes
                    .iter()
                    // LLVM 23 added a target-generic `PureIntrinsic` class
                    // (Intrinsics.td) and migrated many NVVM declarations to
                    // it from the NVPTX-local `NVVMPureIntrinsic`; both carry
                    // the same purity contract.
                    .any(|class| class == "NVVMPureIntrinsic" || class == "PureIntrinsic")
                    || (matches!(
                        policy.family.as_str(),
                        "packed_alu" | "scalar_arithmetic" | "extended_minmax"
                    ) && declaration
                        .properties
                        .iter()
                        .any(|property| property == "IntrNoMem")
                        && declaration
                            .properties
                            .iter()
                            .any(|property| property == "IntrSpeculatable"))
                    || (policy.family == "scalar_math"
                        && declaration
                            .properties
                            .iter()
                            .any(|property| property == "IntrNoMem")),
                "{} is marked pure, but its imported declaration is not an NVVMPureIntrinsic",
                policy.id
            );
        }
        if policy.memory == "none" {
            ensure!(
                declaration
                    .properties
                    .iter()
                    .any(|property| property == "IntrNoMem"),
                "{} is marked no-memory, but its imported declaration lacks IntrNoMem",
                policy.id
            );
        }
        let imported_convergent = declaration
            .properties
            .iter()
            .any(|property| property == "IntrConvergent");
        let convergence_supplied_by_ptx =
            (matches!(policy.family.as_str(), "register_mma" | "sparse_mma")
                && (policy.register_mma.is_some() || policy.sparse_mma.is_some())
                || (policy.family == "cluster_memory"
                    && policy.cluster_memory.is_some()
                    && policy.backend_lowerings.iter().all(|lowering| {
                        lowering.mechanism == BackendLoweringMechanism::InlinePtx
                    }))
                || (policy.family == "mbarrier_extended"
                    && policy.mbarrier_extended.is_some()
                    && policy.backend_lowerings.iter().all(|lowering| {
                        lowering.mechanism == BackendLoweringMechanism::InlinePtx
                    }))
                || (policy.family == "tcgen05"
                    && policy.tcgen05.is_some()
                    && policy.backend_lowerings.iter().all(|lowering| {
                        lowering.mechanism == BackendLoweringMechanism::InlinePtx
                    })))
                && policy.convergent
                && !imported_convergent;
        ensure!(
            imported_convergent == policy.convergent || convergence_supplied_by_ptx,
            "{} convergence mismatch: imported {}, overlay {}",
            policy.id,
            imported_convergent,
            policy.convergent
        );
        let selectionless_closed_family = (policy.family == "packed_conversion"
            && policy.packed_conversion.is_some())
            || (policy.family == "prmt" && policy.prmt.is_some())
            || (policy.family == "clc"
                && policy.clc.as_ref().is_some_and(|clc| {
                    matches!(
                        clc.operation,
                        ClcOperation::QueryIsCanceled
                            | ClcOperation::QueryGetFirstCtaidX
                            | ClcOperation::QueryGetFirstCtaidY
                            | ClcOperation::QueryGetFirstCtaidZ
                    )
                }))
            || (policy.family == "sreg"
                && policy.special_register.as_ref().is_some_and(|special| {
                    matches!(
                        special.register,
                        SpecialRegisterKind::Envreg1 | SpecialRegisterKind::Envreg2
                    )
                }))
            || policy.family == "stmatrix"
            || (policy.family == "tma"
                && policy.tma.as_ref().is_some_and(|tma| {
                    matches!(
                        tma.operation,
                        TmaOperation::Reduce
                            | TmaOperation::PrefetchTensorMap
                            | TmaOperation::ReplaceBoxDim
                            | TmaOperation::ReplaceElementStride
                            | TmaOperation::ReplaceElementType
                            | TmaOperation::ReplaceFillMode
                            | TmaOperation::ReplaceGlobalAddress
                            | TmaOperation::ReplaceGlobalDim
                            | TmaOperation::ReplaceGlobalStride
                            | TmaOperation::ReplaceInterleaveLayout
                            | TmaOperation::ReplaceRank
                            | TmaOperation::ReplaceSwizzleAtomicity
                            | TmaOperation::ReplaceSwizzleMode
                    )
                }))
            || (policy.family == "tcgen05"
                && policy.tcgen05.as_ref().is_some_and(|tcgen05| {
                    tcgen05.source_contract
                        == Tcgen05SourceContract::LlvmCustomLoweringWithoutSelection
                }))
            || (policy.family == "scalar_math"
                && policy.scalar_math.as_ref().is_some_and(|sm| {
                    // Tanh is absent here only because it is PTX-native (no
                    // imported declaration), so this check never sees it.
                    matches!(
                        sm.operation,
                        ScalarMathOperation::Sin
                            | ScalarMathOperation::Cos
                            | ScalarMathOperation::Ex2
                            | ScalarMathOperation::Lg2
                            | ScalarMathOperation::Rsqrt
                    )
                }));
        ensure!(
            !declaration.selections.is_empty() || selectionless_closed_family,
            "{} has a declaration but no NVPTX TableGen selection record",
            policy.id
        );
        let mut matching_selections = Vec::new();
        for selection in &declaration.selections {
            if selection_matches_policy(policy, selection)? {
                matching_selections.push(selection);
            }
        }
        let expected_selection_count = match policy.family.as_str() {
            "vote" | "warp_barrier" | "elect" => 2,
            "warp_match" => 4,
            "warp_shuffle" => 8,
            "counted_barrier" => 4,
            "packed_conversion" | "prmt" | "stmatrix" => 0,
            "tma"
                if policy.tma.as_ref().is_some_and(|tma| {
                    matches!(
                        tma.operation,
                        TmaOperation::Reduce
                            | TmaOperation::PrefetchTensorMap
                            | TmaOperation::ReplaceBoxDim
                            | TmaOperation::ReplaceElementStride
                            | TmaOperation::ReplaceElementType
                            | TmaOperation::ReplaceFillMode
                            | TmaOperation::ReplaceGlobalAddress
                            | TmaOperation::ReplaceGlobalDim
                            | TmaOperation::ReplaceGlobalStride
                            | TmaOperation::ReplaceInterleaveLayout
                            | TmaOperation::ReplaceRank
                            | TmaOperation::ReplaceSwizzleAtomicity
                            | TmaOperation::ReplaceSwizzleMode
                    )
                }) =>
            {
                0
            }
            "tcgen05"
                if policy
                    .tcgen05
                    .as_ref()
                    .and_then(|tcgen05| tcgen05.mma.as_ref())
                    .is_some() =>
            {
                let mma = policy.tcgen05.as_ref().unwrap().mma.as_ref().unwrap();
                if mma.alias.is_some() {
                    1
                } else if tcgen05_mma_is_ws(mma.form) {
                    64
                } else if tcgen05_mma_is_ashift(mma.form) {
                    16
                } else {
                    32
                }
            }
            "tcgen05"
                if policy.tcgen05.as_ref().is_some_and(|tcgen05| {
                    tcgen05.source_contract != Tcgen05SourceContract::ExactTablegenSelection
                }) =>
            {
                0
            }
            "clc"
                if policy.clc.as_ref().is_some_and(|clc| {
                    matches!(
                        clc.operation,
                        ClcOperation::QueryIsCanceled
                            | ClcOperation::QueryGetFirstCtaidX
                            | ClcOperation::QueryGetFirstCtaidY
                            | ClcOperation::QueryGetFirstCtaidZ
                    )
                }) =>
            {
                0
            }
            "sreg"
                if policy.special_register.as_ref().is_some_and(|special| {
                    matches!(
                        special.register,
                        SpecialRegisterKind::Envreg1 | SpecialRegisterKind::Envreg2
                    )
                }) =>
            {
                0
            }
            "mbarrier_extended"
                if policy.mbarrier_extended.as_ref().is_some_and(|mbarrier| {
                    matches!(
                        mbarrier.operation,
                        MbarrierExtendedOperation::ArriveExpectTxCta
                            | MbarrierExtendedOperation::ArriveExpectTxCluster
                            | MbarrierExtendedOperation::TryWaitParityCta
                            | MbarrierExtendedOperation::TryWaitParityCluster
                    )
                }) =>
            {
                0
            }
            "mbarrier_extended" if policy.id == "nanosleep" => 2,
            "cp_async_copy"
                if policy
                    .cp_async_copy
                    .as_ref()
                    .is_some_and(|copy| copy.source_size == CpAsyncSourceSize::Runtime) =>
            {
                2
            }
            "cluster_memory" if policy.id == "map_shared_rank" => 2,
            "scalar_math"
                if policy.scalar_math.as_ref().is_some_and(|sm| {
                    matches!(
                        sm.operation,
                        ScalarMathOperation::Sin
                            | ScalarMathOperation::Cos
                            | ScalarMathOperation::Ex2
                            | ScalarMathOperation::Lg2
                            | ScalarMathOperation::Rsqrt
                    )
                }) =>
            {
                0
            }
            _ => 1,
        };
        ensure!(
            matching_selections.len() == expected_selection_count,
            "{} expected PTX {:?} does not agree with its closed imported selection set",
            policy.id,
            policy.expected_ptx
        );
        for selection in matching_selections {
            validate_selected_target_predicates(policy, selection)?;
        }
    }
    Ok(())
}
