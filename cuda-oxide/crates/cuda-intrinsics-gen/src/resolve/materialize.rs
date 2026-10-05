/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    CatalogBackend, CatalogBackendLowering, CatalogDialect, CatalogHalfOpenRange, CatalogIntrinsic,
    CatalogLdmatrix, CatalogLlvm, CatalogLlvmResultFacts, CatalogRust, CatalogSelection,
    CatalogSemantics, CatalogTarget, CatalogTargetRequirement, EvidenceRecord,
    ExecutionControlOperation, ImportedIntrinsic, IntrinsicSource, OverlayIntrinsic,
    RuntimeValidation,
};
use anyhow::{Context, Result, ensure};
use std::collections::BTreeMap;

use super::abi_ledger::*;
use super::evidence::*;
use super::guards::*;
use super::targets::*;

pub(super) fn resolve_backend_lowerings(
    policy: &OverlayIntrinsic,
    evidence_by_profile_id: &BTreeMap<(&str, &str), IndexedEvidence<'_>>,
) -> Result<Vec<CatalogBackendLowering>> {
    let mut resolved = Vec::with_capacity(policy.backend_lowerings.len());
    let mut runtime_states = Vec::with_capacity(policy.backend_lowerings.len());
    for lowering in &policy.backend_lowerings {
        let evidence = evidence_by_profile_id
            .get(&(lowering.evidence_profile.as_str(), policy.id.as_str()))
            .with_context(|| {
                format!(
                    "{} has no evidence in backend profile {}",
                    policy.id, lowering.evidence_profile
                )
            })?;
        validate_evidence(policy, evidence, Some(lowering))?;
        runtime_states.push(evidence.record.runtime_validation);
        resolved.push(CatalogBackendLowering {
            backend: lowering.backend,
            mechanism: lowering.mechanism,
            evidence_profile: lowering.evidence_profile.clone(),
            target: backend_target_requirement(policy, lowering)?,
            version: evidence.backend_version.to_owned(),
            sha256: evidence.backend_sha256.to_owned(),
            artifact_path: evidence.file.artifact_path.clone(),
            build_id_prefix: evidence.file.build_id_prefix.clone(),
            status: evidence.record.status.clone(),
            stages: evidence.record.stages.clone(),
        });
    }
    if let Some(safety) = &policy.ldmatrix_safety {
        match safety.runtime_validation {
            RuntimeValidation::Unexecuted => ensure!(
                runtime_states
                    .iter()
                    .all(|state| *state == Some(RuntimeValidation::Unexecuted)),
                "{} overlay says runtime is unexecuted but backend evidence disagrees",
                policy.id
            ),
            RuntimeValidation::Executed => ensure!(
                runtime_states.contains(&Some(RuntimeValidation::Executed)),
                "{} overlay says runtime is executed but no backend evidence has an executed runtime stage",
                policy.id
            ),
        }
    }
    if policy.family == "stmatrix" {
        ensure!(
            runtime_states
                .iter()
                .all(|state| *state == Some(RuntimeValidation::Unexecuted)),
            "{} stmatrix source contract is unexecuted but backend evidence disagrees",
            policy.id
        );
    }
    if let Some(packed) = &policy.packed_atomic {
        match packed.runtime_validation {
            RuntimeValidation::Unexecuted => ensure!(
                runtime_states
                    .iter()
                    .all(|state| *state == Some(RuntimeValidation::Unexecuted)),
                "{} packed-atomic runtime is unexecuted but backend evidence disagrees",
                policy.id
            ),
            RuntimeValidation::Executed => ensure!(
                runtime_states.contains(&Some(RuntimeValidation::Executed)),
                "{} packed-atomic runtime is executed but no backend evidence records execution",
                policy.id
            ),
        }
    }
    if let Some(mbarrier) = &policy.mbarrier_basic {
        match mbarrier.runtime_validation {
            RuntimeValidation::Unexecuted => ensure!(
                runtime_states
                    .iter()
                    .all(|state| *state == Some(RuntimeValidation::Unexecuted)),
                "{} mbarrier runtime is unexecuted but backend evidence disagrees",
                policy.id
            ),
            RuntimeValidation::Executed => ensure!(
                runtime_states.contains(&Some(RuntimeValidation::Executed)),
                "{} mbarrier runtime is executed but no backend evidence records execution",
                policy.id
            ),
        }
    }
    if let Some(movmatrix) = &policy.movmatrix {
        match movmatrix.runtime_validation {
            RuntimeValidation::Unexecuted => ensure!(
                runtime_states
                    .iter()
                    .all(|state| *state == Some(RuntimeValidation::Unexecuted)),
                "{} movmatrix runtime is unexecuted but backend evidence disagrees",
                policy.id
            ),
            RuntimeValidation::Executed => ensure!(
                runtime_states.contains(&Some(RuntimeValidation::Executed)),
                "{} movmatrix runtime is executed but no backend evidence records execution",
                policy.id
            ),
        }
    }
    if let Some(mbarrier) = &policy.mbarrier_extended {
        match mbarrier.runtime_validation {
            RuntimeValidation::Unexecuted => ensure!(
                runtime_states
                    .iter()
                    .all(|state| *state == Some(RuntimeValidation::Unexecuted)),
                "{} extended-mbarrier runtime is unexecuted but backend evidence disagrees",
                policy.id
            ),
            RuntimeValidation::Executed => ensure!(
                runtime_states.contains(&Some(RuntimeValidation::Executed)),
                "{} extended-mbarrier runtime is executed but no backend evidence records execution",
                policy.id
            ),
        }
    }
    if let Some(bridge) = &policy.cp_async_mbarrier {
        match bridge.runtime_validation {
            RuntimeValidation::Unexecuted => ensure!(
                runtime_states
                    .iter()
                    .all(|state| *state == Some(RuntimeValidation::Unexecuted)),
                "{} cp.async mbarrier runtime is unexecuted but backend evidence disagrees",
                policy.id
            ),
            RuntimeValidation::Executed => ensure!(
                runtime_states.contains(&Some(RuntimeValidation::Executed)),
                "{} cp.async mbarrier runtime is executed but no backend evidence records execution",
                policy.id
            ),
        }
    }
    if let Some(mma) = &policy.register_mma {
        match mma.runtime_validation {
            RuntimeValidation::Unexecuted => ensure!(
                runtime_states
                    .iter()
                    .all(|state| *state == Some(RuntimeValidation::Unexecuted)),
                "{} register-MMA runtime is unexecuted but backend evidence disagrees",
                policy.id
            ),
            RuntimeValidation::Executed => ensure!(
                runtime_states.contains(&Some(RuntimeValidation::Executed)),
                "{} register-MMA runtime is executed but no backend evidence records execution",
                policy.id
            ),
        }
    }
    if let Some(mma) = &policy.sparse_mma {
        match mma.runtime_validation {
            RuntimeValidation::Unexecuted => ensure!(
                runtime_states
                    .iter()
                    .all(|state| *state == Some(RuntimeValidation::Unexecuted)),
                "{} sparse-MMA runtime is unexecuted but backend evidence disagrees",
                policy.id
            ),
            RuntimeValidation::Executed => ensure!(
                runtime_states.contains(&Some(RuntimeValidation::Executed)),
                "{} sparse-MMA runtime is executed but no backend evidence records execution",
                policy.id
            ),
        }
    }
    if let Some(debug) = &policy.debug_control {
        match debug.runtime_validation {
            RuntimeValidation::Unexecuted => ensure!(
                runtime_states
                    .iter()
                    .all(|state| *state == Some(RuntimeValidation::Unexecuted)),
                "{} debug-control runtime is unexecuted but backend evidence disagrees",
                policy.id
            ),
            RuntimeValidation::Executed => ensure!(
                runtime_states.contains(&Some(RuntimeValidation::Executed)),
                "{} debug-control runtime is executed but no backend evidence records execution",
                policy.id
            ),
        }
    }
    if let Some(cluster_memory) = &policy.cluster_memory {
        match cluster_memory.runtime_validation {
            RuntimeValidation::Unexecuted => ensure!(
                runtime_states
                    .iter()
                    .all(|state| *state == Some(RuntimeValidation::Unexecuted)),
                "{} cluster-memory runtime is unexecuted but backend evidence disagrees",
                policy.id
            ),
            RuntimeValidation::Executed => ensure!(
                runtime_states.contains(&Some(RuntimeValidation::Executed)),
                "{} cluster-memory runtime is executed but no backend evidence records execution",
                policy.id
            ),
        }
    }
    if let Some(clc) = &policy.clc {
        match clc.runtime_validation {
            RuntimeValidation::Unexecuted => ensure!(
                runtime_states
                    .iter()
                    .all(|state| *state == Some(RuntimeValidation::Unexecuted)),
                "{} CLC runtime is unexecuted but backend evidence disagrees",
                policy.id
            ),
            RuntimeValidation::Executed => ensure!(
                runtime_states.contains(&Some(RuntimeValidation::Executed)),
                "{} CLC runtime is executed but no backend evidence records execution",
                policy.id
            ),
        }
    }
    if let Some(conversion) = &policy.scalar_conversion {
        match conversion.runtime_validation {
            RuntimeValidation::Unexecuted => ensure!(
                runtime_states
                    .iter()
                    .all(|state| *state == Some(RuntimeValidation::Unexecuted)),
                "{} scalar-conversion runtime is unexecuted but backend evidence disagrees",
                policy.id
            ),
            RuntimeValidation::Executed => ensure!(
                runtime_states.contains(&Some(RuntimeValidation::Executed)),
                "{} scalar-conversion runtime is executed but no backend evidence records execution",
                policy.id
            ),
        }
    }
    if let Some(arithmetic) = &policy.scalar_arithmetic {
        match arithmetic.runtime_validation {
            RuntimeValidation::Unexecuted => ensure!(
                runtime_states
                    .iter()
                    .all(|state| *state == Some(RuntimeValidation::Unexecuted)),
                "{} scalar-arithmetic runtime is unexecuted but backend evidence disagrees",
                policy.id
            ),
            RuntimeValidation::Executed => ensure!(
                runtime_states.contains(&Some(RuntimeValidation::Executed)),
                "{} scalar-arithmetic runtime is executed but no backend evidence records execution",
                policy.id
            ),
        }
    }
    resolved.sort_by_key(|lowering| lowering.backend);
    Ok(resolved)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn resolve_record(
    policy: &OverlayIntrinsic,
    source: IntrinsicSource,
    declaration: Option<&ImportedIntrinsic>,
    evidence: &EvidenceRecord,
    backend_profile: &str,
    backend_version: &str,
    backend_sha256: &str,
    backend_lowerings: Vec<CatalogBackendLowering>,
    intrinsic_abi: u32,
) -> Result<CatalogIntrinsic> {
    materialize_record(
        policy,
        source,
        declaration,
        CatalogBackend {
            profile: backend_profile.to_owned(),
            version: backend_version.to_owned(),
            sha256: backend_sha256.to_owned(),
            status: evidence.status.clone(),
            target_triple: evidence.target_triple.clone(),
            gpu_target: evidence.gpu_target.clone(),
            ptx_feature: evidence.ptx_feature.clone(),
        },
        backend_lowerings,
        intrinsic_abi,
    )
}

pub(super) fn materialize_record(
    policy: &OverlayIntrinsic,
    source: IntrinsicSource,
    declaration: Option<&ImportedIntrinsic>,
    backend: CatalogBackend,
    backend_lowerings: Vec<CatalogBackendLowering>,
    intrinsic_abi: u32,
) -> Result<CatalogIntrinsic> {
    let native_target = if let Some(mma) = policy
        .tcgen05
        .as_ref()
        .and_then(|tcgen05| tcgen05.mma.as_ref())
    {
        mma.llvm_target.clone()
    } else {
        CatalogTargetRequirement {
            minimum_ptx: parse_ptx_version(&policy.minimum_ptx, &policy.id)?,
            hardware: parse_hardware_target(policy)?,
        }
    };
    let llvm = if let Some(declaration) = declaration {
        Some(CatalogLlvm {
            symbol: policy
                .llvm_symbol
                .clone()
                .expect("validated imported LLVM symbol"),
            resolved_symbol: policy.resolved_llvm_symbol.clone(),
            arguments: policy.llvm_arguments.clone(),
            results: policy.llvm_results.clone(),
            properties: declaration.properties.clone(),
            result_facts: imported_result_facts(&declaration.properties)?,
        })
    } else {
        None
    };
    let preserves_empty_dialect_signature = (policy.family == "sync"
        && policy.id == "sync_threads")
        || matches!(
            ExecutionControlOperation::from_catalog_id(&policy.id),
            Some(
                ExecutionControlOperation::SetMaxNRegInc | ExecutionControlOperation::SetMaxNRegDec
            )
        );
    let dialect_operands =
        if policy.dialect_operands.is_empty() && !preserves_empty_dialect_signature {
            policy.llvm_arguments.clone()
        } else {
            policy.dialect_operands.clone()
        };
    let dialect_results = if policy.dialect_results.is_empty() && !preserves_empty_dialect_signature
    {
        policy.llvm_results.clone()
    } else {
        policy.dialect_results.clone()
    };
    let mut selections = Vec::new();
    for selection in declaration
        .into_iter()
        .flat_map(|declaration| declaration.selections.iter())
    {
        if selection_matches_policy(policy, selection)? {
            selections.push(CatalogSelection {
                source_record: selection.source_record.clone(),
                asm: selection.asm.clone(),
                predicates: selection.predicates.clone(),
                constraints: selection.constraints.clone(),
            });
        }
    }
    Ok(CatalogIntrinsic {
        id: policy.id.clone(),
        operation_key: policy.operation_key.clone(),
        family: policy.family.clone(),
        source,
        selections,
        rust: CatalogRust {
            abi_id: policy.abi_id.clone(),
            module: policy.rust_module.clone(),
            name: policy.rust_name.clone(),
            arguments: policy.rust_arguments.clone(),
            result: policy.rust_result.clone(),
            safe: policy.safe,
            must_use: policy.must_use,
            safe_allowlist_reason: policy.safe_allowlist_reason.clone(),
            canonical_path: canonical_rust_path(intrinsic_abi, &policy.abi_id),
            public_path: policy.public_rust_path.clone(),
            compatibility_paths: policy.compatibility_rust_paths.clone(),
        },
        dialect: CatalogDialect {
            op_type: policy.dialect_op_type.clone(),
            op_name: policy.dialect_op_name.clone(),
            operands: dialect_operands,
            results: dialect_results,
        },
        llvm,
        semantics: CatalogSemantics {
            pure: policy.pure,
            memory: policy.memory.clone(),
            convergent: policy.convergent,
            execution_scope: policy.execution_scope.clone(),
        },
        target: CatalogTarget {
            minimum_ptx: native_target.minimum_ptx,
            hardware: native_target.hardware,
            ptx_result: policy.ptx_result.clone(),
            targets: policy.targets.clone(),
            ptx_isa_version: policy.ptx_isa_version.clone(),
            ptx_isa_section: policy.ptx_isa_section.clone(),
            ptx_isa_url: policy.ptx_isa_url.clone(),
        },
        backend,
        backend_lowerings,
        packed_atomic: policy.packed_atomic.clone(),
        redux: policy.redux.clone(),
        vote: policy.vote.clone(),
        active_mask: policy.active_mask.clone(),
        warp_match: policy.warp_match.clone(),
        warp_barrier: policy.warp_barrier.clone(),
        warp_shuffle: policy.warp_shuffle.clone(),
        dot_product: policy.dot_product.clone(),
        packed_alu: policy.packed_alu.clone(),
        integer_minmax: policy.integer_minmax.clone(),
        packed_conversion: policy.packed_conversion.clone(),
        scalar_conversion: policy.scalar_conversion.clone(),
        scalar_arithmetic: policy.scalar_arithmetic.clone(),
        scalar_math: policy.scalar_math.clone(),
        extended_minmax: policy.extended_minmax.clone(),
        cp_async_copy: policy.cp_async_copy.clone(),
        cp_async_control: policy.cp_async_control.clone(),
        cp_async_mbarrier: policy.cp_async_mbarrier.clone(),
        mbarrier_basic: policy.mbarrier_basic.clone(),
        movmatrix: policy.movmatrix.clone(),
        mbarrier_extended: policy.mbarrier_extended.clone(),
        register_mma: policy.register_mma.clone(),
        sparse_mma: policy.sparse_mma.clone(),
        prmt: policy.prmt.clone(),
        cluster_barrier: policy.cluster_barrier.clone(),
        wgmma_control: policy.wgmma_control.clone(),
        special_register: policy.special_register.clone(),
        debug_control: policy.debug_control.clone(),
        cluster_memory: policy.cluster_memory.clone(),
        clc: policy.clc.clone(),
        tma: policy.tma.clone(),
        tcgen05: policy.tcgen05.clone(),
        ldmatrix: policy
            .ldmatrix_variant
            .clone()
            .map(|variant| CatalogLdmatrix {
                variant,
                safety: policy
                    .ldmatrix_safety
                    .clone()
                    .expect("validated ldmatrix safety"),
                adapter: policy.ldmatrix_adapter.expect("validated ldmatrix adapter"),
                selected_address_space: policy
                    .selected_address_space
                    .expect("validated ldmatrix address space"),
            }),
        lowering: policy.lowering.clone(),
        expected_ptx: policy.expected_ptx.clone(),
        summary: policy.summary.clone(),
    })
}

pub(super) fn imported_result_facts(properties: &[String]) -> Result<CatalogLlvmResultFacts> {
    let no_undef = properties.iter().any(|property| property == "NoUndef<ret>");
    let mut range = None;
    for property in properties {
        let Some(bounds) = property
            .strip_prefix("Range<ret,")
            .and_then(|value| value.strip_suffix('>'))
        else {
            continue;
        };
        let (lower, upper_exclusive) = bounds
            .split_once(',')
            .with_context(|| format!("malformed return range property {property:?}"))?;
        ensure!(
            !lower.is_empty() && !upper_exclusive.is_empty(),
            "malformed return range property {property:?}"
        );
        ensure!(range.is_none(), "duplicate return range properties");
        range = Some(CatalogHalfOpenRange {
            lower: lower.to_owned(),
            upper_exclusive: upper_exclusive.to_owned(),
        });
    }
    Ok(CatalogLlvmResultFacts { no_undef, range })
}
