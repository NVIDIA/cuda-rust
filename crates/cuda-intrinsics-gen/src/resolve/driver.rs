/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::extract::{IMPORTED_SCHEMA, read_upstream_lock};
use crate::model::{
    AbiLedgerFile, BackendLoweringMechanism, CatalogBackend, CatalogFile,
    CatalogHardwareAlternative, CatalogHardwareTarget, CatalogInputs, CatalogSource,
    CatalogTargetRequirement, ExecutionControlOperation, ImportedFile, ImportedIntrinsic,
    IntrinsicBackend, IntrinsicSource, OverlayFile, OverlayIntrinsic, PtxVersion,
};
#[cfg(test)]
use crate::model::{
    CatalogBackendLowering, ClcAdmission, ClcOperation, RuntimeValidation, Tcgen05Admission,
    Tcgen05LdAdmissionVariant, Tcgen05Operation, Tcgen05StAdmissionVariant, TmaAdmission,
};
use crate::util::{read_json, sha256_file};
#[cfg(test)]
use anyhow::bail;
use anyhow::{Context, Result, ensure};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use super::abi_ledger::*;
use super::evidence::*;
use super::families::*;
use super::materialize::*;
use super::overlay::*;
use super::policy::*;
use super::targets::*;

pub(super) struct ResolutionBase {
    pub(super) overlay: OverlayFile,
    pub(super) imported: ImportedFile,
    pub(super) source: CatalogSource,
    pub(super) imported_sha256: String,
    pub(super) overlay_sha256: String,
    pub(super) abi_ledger_sha256: String,
}

#[derive(Debug)]
pub(crate) struct CandidateResolution {
    pub catalog: CatalogFile,
    pub mechanism: BackendLoweringMechanism,
    pub requirement: CatalogTargetRequirement,
}

pub(super) fn primary_evidence_profile<'a>(
    policy: &'a OverlayIntrinsic,
    default_profile: &'a str,
) -> Result<&'a str> {
    let needs_family_profile = policy.tma.is_some()
        || ExecutionControlOperation::from_catalog_id(&policy.id).is_some()
        || policy
            .tcgen05
            .as_ref()
            .and_then(|tcgen05| tcgen05.mma.as_ref())
            .is_some();
    if !needs_family_profile {
        return Ok(default_profile);
    }
    policy
        .backend_lowerings
        .iter()
        .find(|lowering| lowering.backend == IntrinsicBackend::LlvmNvptx)
        .map(|lowering| lowering.evidence_profile.as_str())
        .with_context(|| format!("{} has no LLVM family evidence route", policy.id))
}

pub fn resolve(repo_root: &Path) -> Result<CatalogFile> {
    let base = load_resolution_base(repo_root)?;
    let ResolutionBase {
        overlay,
        imported,
        source,
        imported_sha256,
        overlay_sha256,
        abi_ledger_sha256,
    } = base;
    let imported_by_record = index_imported_intrinsics(&imported)?;
    let llvm_revision = source.llvm_revision.clone();
    let (evidence_files, evidence_hashes) = read_evidence(repo_root)?;
    let evidence_by_profile_id = index_evidence(&evidence_files, &llvm_revision)?;

    let mut intrinsics = Vec::with_capacity(overlay.intrinsics.len());
    for policy in &overlay.intrinsics {
        let source = resolve_policy_source(policy)?;
        let declaration = resolve_imported_declaration(policy, &source, &imported_by_record)?;
        validate_special_register_llvm_exclusion(policy, &imported_by_record)?;
        validate_policy(policy, &source, declaration, overlay.intrinsic_abi)?;
        let primary_profile = primary_evidence_profile(policy, &overlay.backend_profile)?;
        let evidence = evidence_by_profile_id
            .get(&(primary_profile, policy.id.as_str()))
            .with_context(|| {
                format!(
                    "intrinsic {} has no evidence record in selected profile {}",
                    policy.id, primary_profile
                )
            })?;
        validate_evidence(policy, evidence, None)?;
        let backend_lowerings = resolve_backend_lowerings(policy, &evidence_by_profile_id)?;
        intrinsics.push(resolve_record(
            policy,
            source,
            declaration,
            evidence.record,
            primary_profile,
            evidence.backend_version,
            evidence.backend_sha256,
            backend_lowerings,
            overlay.intrinsic_abi,
        )?);
    }
    for (_, evidence_id) in evidence_by_profile_id.keys() {
        ensure!(
            overlay
                .intrinsics
                .iter()
                .any(|record| record.id == *evidence_id),
            "evidence exists for unknown catalog ID {evidence_id}"
        );
    }

    Ok(CatalogFile {
        schema: CATALOG_SCHEMA,
        catalog_version: overlay.catalog_version,
        intrinsic_abi: overlay.intrinsic_abi,
        generator_version: env!("CARGO_PKG_VERSION").to_owned(),
        source,
        inputs: CatalogInputs {
            imported_sha256,
            overlay_sha256,
            abi_ledger_sha256,
            evidence_sha256: evidence_hashes,
        },
        intrinsics,
    })
}

#[cfg(test)]
pub(crate) fn test_catalog_with_clc(repo_root: &Path) -> Result<CatalogFile> {
    let mut catalog = resolve(repo_root)?;
    let active_count = catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "clc")
        .count();
    if active_count != 0 {
        ensure!(
            active_count == 6,
            "active CLC family must contain six records"
        );
        return Ok(catalog);
    }
    let imported: ImportedFile = read_json(&repo_root.join("intrinsics/imported.json"))?;
    let imported_by_record = index_imported_intrinsics(&imported)?;
    let operations = [
        ClcOperation::TryCancel,
        ClcOperation::TryCancelMulticast,
        ClcOperation::QueryIsCanceled,
        ClcOperation::QueryGetFirstCtaidX,
        ClcOperation::QueryGetFirstCtaidY,
        ClcOperation::QueryGetFirstCtaidZ,
    ];
    let admission = ClcAdmission {
        llvm_evidence_profile: "llvm-clc-test".into(),
        libnvvm_evidence_profile: "libnvvm-clc-test".into(),
        runtime_validation: RuntimeValidation::Unexecuted,
        variants: operations
            .into_iter()
            .map(|operation| crate::model::ClcAdmissionVariant {
                abi_id: clc_recipe(operation).abi_id.into(),
                operation,
            })
            .collect(),
    };
    for policy in expand_clc_admission(&admission)? {
        let source = resolve_policy_source(&policy)?;
        let declaration = resolve_imported_declaration(&policy, &source, &imported_by_record)?;
        validate_policy(&policy, &source, declaration, catalog.intrinsic_abi)?;
        let backend_lowerings = policy
            .backend_lowerings
            .iter()
            .map(|lowering| {
                Ok(CatalogBackendLowering {
                    backend: lowering.backend,
                    mechanism: lowering.mechanism,
                    evidence_profile: lowering.evidence_profile.clone(),
                    target: backend_target_requirement(&policy, lowering)?,
                    version: "test".into(),
                    sha256: "0".repeat(64),
                    artifact_path: None,
                    build_id_prefix: None,
                    status: "validated".into(),
                    stages: vec![],
                })
            })
            .collect::<Result<Vec<_>>>()?;
        catalog.intrinsics.push(materialize_record(
            &policy,
            source,
            declaration,
            CatalogBackend {
                profile: "clc-test".into(),
                version: "test".into(),
                sha256: "0".repeat(64),
                status: "validated".into(),
                target_triple: "nvptx64-nvidia-cuda".into(),
                gpu_target: "sm_100".into(),
                ptx_feature: "+ptx86".into(),
            },
            backend_lowerings,
            catalog.intrinsic_abi,
        )?);
    }
    catalog
        .intrinsics
        .sort_by(|left, right| left.id.cmp(&right.id));
    Ok(catalog)
}

#[cfg(test)]
pub(crate) fn test_catalog_with_tma(repo_root: &Path) -> Result<CatalogFile> {
    let mut catalog = resolve(repo_root)?;
    let active_count = catalog
        .intrinsics
        .iter()
        .filter(|record| record.tma.is_some())
        .count();
    let expected_count = TMA_OPERATIONS.len() + tma_reduction_matrix().len();
    if active_count == expected_count {
        return Ok(catalog);
    }
    ensure!(
        active_count == 0 || active_count == TMA_OPERATIONS.len(),
        "active TMA catalog has {active_count} records; expected 0, {} base records, or {expected_count} total records",
        TMA_OPERATIONS.len()
    );
    let imported: ImportedFile = read_json(&repo_root.join("intrinsics/imported.json"))?;
    let imported_by_record = index_imported_intrinsics(&imported)?;
    let admission = TmaAdmission {
        llvm_evidence_profile: "llvm-tma-test".into(),
        libnvvm_evidence_profile: "libnvvm-tma-test".into(),
        cta_llvm_evidence_profile: "llvm-tma-test".into(),
        cta_libnvvm_evidence_profile: "libnvvm-tma-test".into(),
        reduce_llvm_evidence_profile: Some("llvm-tma-reduce-test".into()),
        reduce_libnvvm_evidence_profile: Some("libnvvm-tma-reduce-test".into()),
        runtime_validation: RuntimeValidation::Unexecuted,
        variants: TMA_OPERATIONS
            .into_iter()
            .map(|operation| crate::model::TmaAdmissionVariant {
                abi_id: tma_recipe(operation).abi_id.into(),
                operation,
            })
            .collect(),
        reduce_variants: tma_reduction_admission_variants(),
    };
    for policy in expand_tma_admission(&admission)? {
        if catalog
            .intrinsics
            .iter()
            .any(|record| record.id.as_str() == policy.id.as_str())
        {
            continue;
        }
        let source = resolve_policy_source(&policy)?;
        let declaration = resolve_imported_declaration(&policy, &source, &imported_by_record)?;
        validate_policy(&policy, &source, declaration, catalog.intrinsic_abi)?;
        let backend_lowerings = policy
            .backend_lowerings
            .iter()
            .map(|lowering| {
                Ok(CatalogBackendLowering {
                    backend: lowering.backend,
                    mechanism: lowering.mechanism,
                    evidence_profile: lowering.evidence_profile.clone(),
                    target: backend_target_requirement(&policy, lowering)?,
                    version: "test".into(),
                    sha256: "0".repeat(64),
                    artifact_path: None,
                    build_id_prefix: None,
                    status: "validated".into(),
                    stages: vec![],
                })
            })
            .collect::<Result<Vec<_>>>()?;
        catalog.intrinsics.push(materialize_record(
            &policy,
            source,
            declaration,
            CatalogBackend {
                profile: "tma-test".into(),
                version: "test".into(),
                sha256: "0".repeat(64),
                status: "validated".into(),
                target_triple: "nvptx64-nvidia-cuda".into(),
                gpu_target: "sm_90".into(),
                ptx_feature: "+ptx80".into(),
            },
            backend_lowerings,
            catalog.intrinsic_abi,
        )?);
    }
    catalog
        .intrinsics
        .sort_by(|left, right| left.id.cmp(&right.id));
    Ok(catalog)
}

#[cfg(test)]
pub(crate) fn test_catalog_with_tcgen05(repo_root: &Path) -> Result<CatalogFile> {
    let mut catalog = resolve(repo_root)?;
    let active_count = catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "tcgen05")
        .count();
    match active_count {
        233 => return Ok(catalog),
        0 => {}
        count => bail!("active tcgen05 catalog has {count} of 233 records"),
    }
    let imported: ImportedFile = read_json(&repo_root.join("intrinsics/imported.json"))?;
    let imported_by_record = index_imported_intrinsics(&imported)?;
    let operations = [
        Tcgen05Operation::Alloc,
        Tcgen05Operation::Dealloc,
        Tcgen05Operation::RelinquishAllocPermit,
        Tcgen05Operation::FenceBeforeThreadSync,
        Tcgen05Operation::FenceAfterThreadSync,
        Tcgen05Operation::Commit,
        Tcgen05Operation::CommitSharedCluster,
        Tcgen05Operation::MmaWsF16,
        Tcgen05Operation::MmaF16,
        Tcgen05Operation::MmaWsBf16,
        Tcgen05Operation::MmaWsTf32,
        Tcgen05Operation::CpSmemToTmem,
        Tcgen05Operation::Ld16x256bX8Pure,
        Tcgen05Operation::Ld16x256bPure,
        Tcgen05Operation::LoadWait,
        Tcgen05Operation::StoreWait,
        Tcgen05Operation::AllocCg2,
        Tcgen05Operation::DeallocCg2,
        Tcgen05Operation::RelinquishAllocPermitCg2,
        Tcgen05Operation::MmaF16Cg2,
        Tcgen05Operation::CommitCg2,
        Tcgen05Operation::CommitSharedClusterCg2,
        Tcgen05Operation::CommitMulticastCg2,
        Tcgen05Operation::CpSmemToTmemCg2,
        Tcgen05Operation::CommitMulticast,
        Tcgen05Operation::ShiftDown,
        Tcgen05Operation::ShiftDownCg2,
    ];
    let admission = Tcgen05Admission {
        llvm_evidence_profile: "llvm-tcgen05-test".into(),
        libnvvm_evidence_profile: "libnvvm-tcgen05-test".into(),
        cp_llvm_evidence_profile: None,
        cp_libnvvm_evidence_profile: None,
        ld_llvm_evidence_profile: None,
        ld_libnvvm_evidence_profile: None,
        st_llvm_evidence_profile: None,
        st_libnvvm_evidence_profile: None,
        offset_llvm_evidence_profile: Some("llvm-tcgen05-offset-test".into()),
        offset_libnvvm_evidence_profile: Some("libnvvm-tcgen05-offset-test".into()),
        control_llvm_evidence_profile: Some("llvm-tcgen05-control-test".into()),
        control_libnvvm_evidence_profile: Some("libnvvm-tcgen05-control-test".into()),
        mma_llvm_evidence_profile: None,
        mma_libnvvm_evidence_profile: None,
        mma_llvm_target_contracts: vec![],
        mma_libnvvm_target_contracts: vec![],
        runtime_validation: RuntimeValidation::Unexecuted,
        variants: operations
            .into_iter()
            .map(|operation| crate::model::Tcgen05AdmissionVariant {
                abi_id: tcgen05_recipe(operation).abi_id.into(),
                operation,
            })
            .collect(),
        cp_variants: vec![],
        ld_variants: vec![],
        st_variants: vec![],
        ld_offset_variants: TCGEN05_OFFSET_LDST_VARIANTS
            .into_iter()
            .flat_map(|(shape, multiplicity)| {
                [false, true]
                    .into_iter()
                    .map(move |pack16| (shape, multiplicity, pack16))
            })
            .enumerate()
            .map(
                |(index, (shape, multiplicity, pack16))| Tcgen05LdAdmissionVariant {
                    abi_id: format!("i{:04}", 728 + index),
                    shape,
                    multiplicity,
                    pack16,
                },
            )
            .collect(),
        st_offset_variants: TCGEN05_OFFSET_LDST_VARIANTS
            .into_iter()
            .flat_map(|(shape, multiplicity)| {
                [false, true]
                    .into_iter()
                    .map(move |unpack16| (shape, multiplicity, unpack16))
            })
            .enumerate()
            .map(
                |(index, (shape, multiplicity, unpack16))| Tcgen05StAdmissionVariant {
                    abi_id: format!("i{:04}", 744 + index),
                    shape,
                    multiplicity,
                    unpack16,
                },
            )
            .collect(),
        mma_variants: vec![],
    };
    for policy in expand_tcgen05_admission(&admission)? {
        let source = resolve_policy_source(&policy)?;
        let declaration = resolve_imported_declaration(&policy, &source, &imported_by_record)?;
        validate_policy(&policy, &source, declaration, catalog.intrinsic_abi)?;
        let backend_lowerings = policy
            .backend_lowerings
            .iter()
            .map(|lowering| {
                Ok(CatalogBackendLowering {
                    backend: lowering.backend,
                    mechanism: lowering.mechanism,
                    evidence_profile: lowering.evidence_profile.clone(),
                    target: backend_target_requirement(&policy, lowering)?,
                    version: "test".into(),
                    sha256: "0".repeat(64),
                    artifact_path: None,
                    build_id_prefix: None,
                    status: "validated".into(),
                    stages: vec![],
                })
            })
            .collect::<Result<Vec<_>>>()?;
        catalog.intrinsics.push(materialize_record(
            &policy,
            source,
            declaration,
            CatalogBackend {
                profile: "tcgen05-test".into(),
                version: "test".into(),
                sha256: "0".repeat(64),
                status: "validated".into(),
                target_triple: "nvptx64-nvidia-cuda".into(),
                gpu_target: "sm_100a".into(),
                ptx_feature: "+ptx86".into(),
            },
            backend_lowerings,
            catalog.intrinsic_abi,
        )?);
    }
    catalog
        .intrinsics
        .sort_by(|left, right| left.id.cmp(&right.id));
    Ok(catalog)
}

pub(super) fn load_resolution_base(repo_root: &Path) -> Result<ResolutionBase> {
    let lock = read_upstream_lock(repo_root)?;
    let imported_path = repo_root.join("intrinsics/imported.json");
    let overlay_path = repo_root.join("intrinsics/overlay.toml");
    let imported: ImportedFile = read_json(&imported_path)?;
    let (mut overlay, overlay_sha256) = read_overlay(repo_root, &overlay_path)?;
    let ledger_path = repo_root.join(format!("intrinsics/abi-v{}.toml", overlay.intrinsic_abi));
    let ledger_text = fs::read_to_string(&ledger_path)
        .with_context(|| format!("read {}", ledger_path.display()))?;
    let ledger: AbiLedgerFile =
        toml::from_str(&ledger_text).with_context(|| format!("parse {}", ledger_path.display()))?;

    ensure!(
        imported.schema == IMPORTED_SCHEMA,
        "unsupported imported.json schema {}",
        imported.schema
    );
    ensure!(
        overlay.schema == OVERLAY_SCHEMA,
        "unsupported overlay.toml schema {}",
        overlay.schema
    );
    ensure!(
        overlay.intrinsic_abi > 0,
        "intrinsic_abi must be a positive integer"
    );
    ensure!(
        imported.source.llvm_revision == lock.llvm.revision,
        "imported facts use LLVM {}, but upstream.lock pins {}",
        imported.source.llvm_revision,
        lock.llvm.revision
    );
    ensure!(
        imported.source.llvm_tblgen_source_revision == lock.llvm.revision,
        "imported facts were not produced by llvm-tblgen built from the pinned source"
    );
    ensure!(
        imported.source.llvm_tblgen_version == lock.llvm_tblgen.version_line,
        "imported facts use llvm-tblgen {:?}, but upstream.lock pins {:?}",
        imported.source.llvm_tblgen_version,
        lock.llvm_tblgen.version_line
    );
    ensure!(
        imported.source.intrinsics_json_sha256 == lock.dumps.intrinsics_sha256,
        "imported intrinsic dump hash does not match upstream.lock"
    );
    ensure!(
        imported.source.nvptx_json_sha256 == lock.dumps.nvptx_sha256,
        "imported NVPTX dump hash does not match upstream.lock"
    );
    let imported_sha256 = sha256_file(&imported_path)?;
    ensure!(
        imported_sha256 == lock.dumps.normalized_imported_sha256,
        "normalized imported.json hash mismatch: upstream.lock records {}, found {}; regenerate from the pinned dumps, and refresh the lock explicitly only for an intentional normalizer change",
        lock.dumps.normalized_imported_sha256,
        imported_sha256
    );

    bind_generated_abi_ids(&mut overlay, &ledger)?;
    overlay
        .intrinsics
        .sort_by(|left, right| left.id.cmp(&right.id));
    validate_execution_control_family_completeness(&overlay.intrinsics)?;
    validate_unique_overlay(&overlay.intrinsics, overlay.intrinsic_abi)?;
    validate_abi_ledger(&overlay, &ledger)?;
    Ok(ResolutionBase {
        overlay,
        imported,
        source: CatalogSource {
            llvm_repository: lock.llvm.repository,
            llvm_revision: lock.llvm.revision,
            llvm_tblgen_version: lock.llvm_tblgen.version_line,
            llvm_tblgen_source_revision: lock
                .llvm_tblgen
                .built_from_llvm_revision
                .context("pinned llvm-tblgen has no source revision")?,
        },
        imported_sha256,
        overlay_sha256,
        abi_ledger_sha256: sha256_file(&ledger_path)?,
    })
}

pub(super) fn index_imported_intrinsics(
    imported: &ImportedFile,
) -> Result<BTreeMap<&str, &ImportedIntrinsic>> {
    let imported_by_record: BTreeMap<_, _> = imported
        .intrinsics
        .iter()
        .map(|intrinsic| (intrinsic.source_record.as_str(), intrinsic))
        .collect();
    ensure!(
        imported_by_record.len() == imported.intrinsics.len(),
        "imported.json contains duplicate source records"
    );
    Ok(imported_by_record)
}

pub(super) fn resolve_imported_declaration<'a>(
    policy: &OverlayIntrinsic,
    source: &IntrinsicSource,
    imported_by_record: &'a BTreeMap<&str, &'a ImportedIntrinsic>,
) -> Result<Option<&'a ImportedIntrinsic>> {
    match source {
        IntrinsicSource::LlvmImported { source_record } => Ok(Some(
            *imported_by_record
                .get(source_record.as_str())
                .with_context(|| {
                    format!(
                        "overlay intrinsic {} references missing imported record {}",
                        policy.id, source_record
                    )
                })?,
        )),
        IntrinsicSource::PtxNative { .. } => Ok(None),
    }
}

pub(crate) fn resolve_candidate(
    repo_root: &Path,
    intrinsic_id: &str,
    backend_version: &str,
    backend_sha256: &str,
    gpu_target: &str,
    ptx_feature: &str,
) -> Result<CandidateResolution> {
    let base = load_resolution_base(repo_root)?;
    let imported_by_record = index_imported_intrinsics(&base.imported)?;
    let policy = base
        .overlay
        .intrinsics
        .iter()
        .find(|policy| policy.id == intrinsic_id || policy.abi_id == intrinsic_id)
        .with_context(|| format!("unknown overlay intrinsic {intrinsic_id}"))?;
    let source = resolve_policy_source(policy)?;
    let declaration = resolve_imported_declaration(policy, &source, &imported_by_record)?;
    validate_special_register_llvm_exclusion(policy, &imported_by_record)?;
    validate_policy(policy, &source, declaration, base.overlay.intrinsic_abi)?;
    ensure!(
        !backend_version.trim().is_empty()
            && backend_sha256.len() == 64
            && backend_sha256.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "candidate backend identity is incomplete"
    );
    let (mechanism, target) = candidate_llvm_route(policy)?;
    validate_candidate_target(policy, &target, gpu_target, ptx_feature)?;

    let record = materialize_record(
        policy,
        source,
        declaration,
        CatalogBackend {
            profile: "candidate-comparison".into(),
            version: backend_version.into(),
            sha256: backend_sha256.into(),
            status: "candidate".into(),
            target_triple: "nvptx64-nvidia-cuda".into(),
            gpu_target: gpu_target.into(),
            ptx_feature: ptx_feature.into(),
        },
        Vec::new(),
        base.overlay.intrinsic_abi,
    )?;
    Ok(CandidateResolution {
        catalog: CatalogFile {
            schema: CATALOG_SCHEMA,
            catalog_version: base.overlay.catalog_version,
            intrinsic_abi: base.overlay.intrinsic_abi,
            generator_version: env!("CARGO_PKG_VERSION").to_owned(),
            source: base.source,
            inputs: CatalogInputs {
                imported_sha256: base.imported_sha256,
                overlay_sha256: base.overlay_sha256,
                abi_ledger_sha256: base.abi_ledger_sha256,
                evidence_sha256: Vec::new(),
            },
            intrinsics: vec![record],
        },
        mechanism,
        requirement: target,
    })
}

pub(super) fn candidate_llvm_route(
    policy: &OverlayIntrinsic,
) -> Result<(BackendLoweringMechanism, CatalogTargetRequirement)> {
    let routes = policy
        .backend_lowerings
        .iter()
        .filter(|lowering| lowering.backend == IntrinsicBackend::LlvmNvptx)
        .collect::<Vec<_>>();
    ensure!(
        routes.len() <= 1,
        "{} has more than one LLVM-NVPTX route",
        policy.id
    );
    if let Some(route) = routes.first() {
        return Ok((route.mechanism, backend_target_requirement(policy, route)?));
    }
    ensure!(
        matches!(
            resolve_policy_source(policy)?,
            IntrinsicSource::LlvmImported { .. }
        ),
        "{} has no LLVM-NVPTX candidate route",
        policy.id
    );
    Ok((
        BackendLoweringMechanism::TypedNvvm,
        CatalogTargetRequirement {
            minimum_ptx: parse_ptx_version(&policy.minimum_ptx, &policy.id)?,
            hardware: parse_hardware_target(policy)?,
        },
    ))
}

pub(super) fn validate_candidate_target(
    policy: &OverlayIntrinsic,
    requirement: &CatalogTargetRequirement,
    gpu_target: &str,
    ptx_feature: &str,
) -> Result<()> {
    ensure!(
        gpu_target.starts_with("sm_"),
        "candidate GPU target {gpu_target:?} must use sm_NN, sm_NNa, or sm_NNf"
    );
    let hardware = parse_stage_hardware(gpu_target).with_context(|| {
        format!("candidate GPU target {gpu_target:?} must use sm_NN, sm_NNa, or sm_NNf")
    })?;
    ensure!(
        describe_stage_hardware(hardware) == gpu_target,
        "candidate GPU target {gpu_target:?} is not canonical"
    );
    let ptx = parse_candidate_ptx_feature(ptx_feature)?;
    let effective_minimum_ptx = if is_f8f6f4_mma_target_matrix_policy(policy) {
        f8f6f4_llvm_ptx_floor(hardware)?
    } else {
        target_requirement_ptx_floor(requirement, hardware, true).with_context(|| {
            format!(
                "candidate GPU target {gpu_target} does not satisfy {} hardware requirement {:?}",
                policy.id, requirement.hardware
            )
        })?
    };
    ensure!(
        ptx.encoded() >= effective_minimum_ptx,
        "candidate target {gpu_target} / {ptx_feature} is below {} PTX floor {}.{}",
        policy.id,
        effective_minimum_ptx / 10,
        effective_minimum_ptx % 10
    );
    let hardware_matches = target_requirement_ptx_floor(requirement, hardware, true).is_some();
    ensure!(
        hardware_matches,
        "candidate GPU target {gpu_target} does not satisfy {} hardware requirement {:?}",
        policy.id,
        requirement.hardware
    );
    Ok(())
}

pub(super) fn target_requirement_ptx_floor(
    requirement: &CatalogTargetRequirement,
    hardware: CatalogHardwareAlternative,
    allow_forward_minimum: bool,
) -> Option<u16> {
    match &requirement.hardware {
        CatalogHardwareTarget::All => Some(requirement.minimum_ptx.encoded()),
        CatalogHardwareTarget::AnyOf { alternatives } => alternatives
            .iter()
            .any(|expected| {
                selected_stage_hardware_matches(hardware, *expected, allow_forward_minimum)
            })
            .then(|| requirement.minimum_ptx.encoded()),
        CatalogHardwareTarget::TargetMatrix { contracts } => contracts
            .iter()
            .flat_map(|contract| contract.alternatives.iter())
            .filter(|alternative| {
                selected_stage_hardware_matches(
                    hardware,
                    alternative.hardware,
                    allow_forward_minimum,
                )
            })
            .map(|alternative| alternative.minimum_ptx.encoded())
            .min(),
    }
}

pub(super) fn parse_candidate_ptx_feature(value: &str) -> Result<PtxVersion> {
    let digits = value
        .strip_prefix("+ptx")
        .with_context(|| format!("candidate PTX feature {value:?} must use +ptxNN"))?;
    ensure!(
        matches!(digits.len(), 2 | 3) && digits.bytes().all(|byte| byte.is_ascii_digit()),
        "candidate PTX feature {value:?} must use +ptxNN"
    );
    let split = digits.len() - 1;
    let version = format!("{}.{}", &digits[..split], &digits[split..]);
    version
        .parse()
        .map_err(|reason: String| anyhow::anyhow!("candidate PTX feature {value:?}: {reason}"))
}
