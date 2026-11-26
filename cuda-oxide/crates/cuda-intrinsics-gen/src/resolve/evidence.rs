/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, CatalogHardwareAlternative, CatalogHardwareTarget,
    CatalogTargetAlternative, EvidenceArtifactKind, EvidenceFile, EvidenceFileV6, EvidenceMatrix,
    EvidenceMatrixTemplate, EvidenceRecord, EvidenceRecordDefaults, EvidenceStage,
    EvidenceStageKind, ImportedAddressSpace, IntrinsicBackend, IntrinsicSource, LdmatrixShape,
    OverlayBackendLowering, OverlayIntrinsic, RuntimeValidation,
};
use crate::ptx::OperandPattern;
use crate::util::sha256_text_file;
use anyhow::{Context, Result, bail, ensure};
use cuda_target_spec::{CudaArch, recorded_ptx_floor};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use super::families::*;
use super::policy::*;
use super::targets::*;

pub(super) fn read_evidence_file(path: &Path) -> Result<EvidenceFile> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    parse_evidence_bytes(&bytes, &path.display().to_string())
}

pub(super) fn parse_evidence_bytes(bytes: &[u8], source: &str) -> Result<EvidenceFile> {
    #[derive(serde::Deserialize)]
    struct Schema {
        schema: u32,
    }

    let schema: Schema =
        serde_json::from_slice(bytes).with_context(|| format!("parse JSON {source}"))?;
    match schema.schema {
        2..=5 => serde_json::from_slice(bytes)
            .with_context(|| format!("parse legacy evidence JSON {source}")),
        6 => {
            let file: EvidenceFileV6 = serde_json::from_slice(bytes)
                .with_context(|| format!("parse matrix evidence JSON {source}"))?;
            expand_evidence_file_v6(file)
                .with_context(|| format!("expand matrix evidence {source}"))
        }
        _ => bail!("unsupported evidence schema in {source}"),
    }
}

pub(super) fn expand_evidence_file_v6(file: EvidenceFileV6) -> Result<EvidenceFile> {
    ensure!(file.schema == 6, "matrix evidence must use schema 6");
    ensure!(
        !file.records.is_empty() || !file.matrices.is_empty(),
        "schema-6 evidence contains no records or matrices"
    );
    reject_default_placeholders(&file.defaults, "evidence defaults", false)?;

    let mut fixture_by_id = BTreeMap::new();
    let mut fixture_coverage = BTreeMap::new();
    for fixture in &file.fixtures {
        ensure!(
            is_safe_matrix_token(&fixture.id),
            "evidence fixture ID {:?} is not a safe token",
            fixture.id
        );
        ensure!(
            fixture.coverage_count > 0 && !fixture.stages.is_empty(),
            "evidence fixture {} has no coverage or stages",
            fixture.id
        );
        reject_stage_placeholders(&fixture.stages, &format!("fixture {}", fixture.id))?;
        ensure!(
            fixture_by_id.insert(fixture.id.as_str(), fixture).is_none(),
            "duplicate evidence fixture ID {}",
            fixture.id
        );
        fixture_coverage.insert(fixture.id.as_str(), 0usize);
    }

    let mut records = file.records;
    let mut record_ids = BTreeSet::new();
    for record in &records {
        ensure!(
            record_ids.insert(record.id.clone()),
            "duplicate expanded evidence ID {}",
            record.id
        );
        validate_stage_pairs(&record.stages, &record.id)?;
    }

    for matrix in &file.matrices {
        let (expanded, referenced_fixtures) =
            expand_evidence_matrix(matrix, &file.defaults, &fixture_by_id)?;
        for fixture_id in referenced_fixtures {
            *fixture_coverage
                .get_mut(fixture_id.as_str())
                .expect("validated fixture reference") += expanded.len();
        }
        for record in expanded {
            ensure!(
                record_ids.insert(record.id.clone()),
                "duplicate expanded evidence ID {}",
                record.id
            );
            records.push(record);
        }
    }

    for fixture in &file.fixtures {
        let actual = fixture_coverage[fixture.id.as_str()];
        ensure!(
            actual > 0,
            "evidence fixture {} is not referenced by any matrix",
            fixture.id
        );
        ensure!(
            actual == fixture.coverage_count,
            "evidence fixture {} covers {actual} expanded records, expected {}",
            fixture.id,
            fixture.coverage_count
        );
    }

    Ok(EvidenceFile {
        schema: file.schema,
        backend_profile: file.backend_profile,
        backend_kind: file.backend_kind,
        llvm_revision: file.llvm_revision,
        backend_version: file.backend_version,
        backend_sha256: file.backend_sha256,
        artifact_path: file.artifact_path,
        build_id_prefix: file.build_id_prefix,
        nvvm_ir_version: file.nvvm_ir_version,
        debug_ir_version: file.debug_ir_version,
        records,
    })
}

pub(super) fn expand_evidence_matrix(
    matrix: &EvidenceMatrix,
    defaults: &EvidenceRecordDefaults,
    fixtures: &BTreeMap<&str, &crate::model::EvidenceFixture>,
) -> Result<(Vec<EvidenceRecord>, Vec<String>)> {
    ensure!(!matrix.axes.is_empty(), "evidence matrix has no axes");
    let mut previous_axis: Option<&str> = None;
    let mut product_count = 1usize;
    let mut bindings = vec![BTreeMap::<String, String>::new()];
    for axis in &matrix.axes {
        ensure!(
            is_safe_matrix_token(&axis.name),
            "evidence matrix axis {:?} is not a safe token",
            axis.name
        );
        if let Some(previous) = previous_axis {
            ensure!(
                previous < axis.name.as_str(),
                "evidence matrix axes must be unique and sorted: {} follows {}",
                axis.name,
                previous
            );
        }
        previous_axis = Some(&axis.name);
        ensure!(
            !axis.values.is_empty(),
            "evidence matrix axis {} has no values",
            axis.name
        );
        let mut values = BTreeSet::new();
        for value in &axis.values {
            ensure!(
                is_safe_matrix_token(value),
                "evidence matrix axis {} has unsafe value {:?}",
                axis.name,
                value
            );
            ensure!(
                values.insert(value.as_str()),
                "evidence matrix axis {} has duplicate value {:?}",
                axis.name,
                value
            );
        }
        product_count = product_count
            .checked_mul(axis.values.len())
            .context("evidence matrix product count overflow")?;
        let mut next = Vec::with_capacity(product_count);
        for binding in bindings {
            for value in &axis.values {
                let mut expanded = binding.clone();
                expanded.insert(axis.name.clone(), value.clone());
                next.push(expanded);
            }
        }
        bindings = next;
    }
    ensure!(
        product_count == matrix.product_count,
        "evidence matrix expands to {product_count} records, expected {}",
        matrix.product_count
    );
    ensure!(
        !matrix.fixtures.is_empty(),
        "evidence matrix references no shared fixture"
    );

    let mut fixture_ids = BTreeSet::new();
    let mut previous_fixture: Option<&str> = None;
    let mut fixture_stages = Vec::new();
    for fixture_id in &matrix.fixtures {
        if let Some(previous) = previous_fixture {
            ensure!(
                previous < fixture_id.as_str(),
                "evidence matrix fixtures must be unique and sorted: {fixture_id} follows {previous}"
            );
        }
        previous_fixture = Some(fixture_id);
        let fixture = fixtures
            .get(fixture_id.as_str())
            .with_context(|| format!("evidence matrix references unknown fixture {fixture_id}"))?;
        fixture_ids.insert(fixture_id.clone());
        fixture_stages.extend(fixture.stages.iter().cloned());
    }

    reject_default_placeholders(&matrix.template.facts, "matrix template facts", true)?;
    validate_matrix_identity(&matrix.template)?;
    let axis_names = matrix
        .axes
        .iter()
        .map(|axis| axis.name.as_str())
        .collect::<BTreeSet<_>>();
    let mut used_axes = BTreeSet::new();
    let mut records = Vec::with_capacity(product_count);
    for binding in &bindings {
        let record = materialize_evidence_record(
            &matrix.template,
            defaults,
            binding,
            &mut used_axes,
            &fixture_stages,
        )?;
        validate_stage_pairs(&record.stages, &record.id)?;
        records.push(record);
    }
    for axis in axis_names {
        ensure!(
            used_axes.contains(axis),
            "evidence matrix declares unused axis {axis}"
        );
    }
    Ok((records, fixture_ids.into_iter().collect()))
}

pub(super) fn validate_matrix_identity(template: &EvidenceMatrixTemplate) -> Result<()> {
    ensure!(
        !template.id.is_empty(),
        "evidence matrix template has an empty ID"
    );
    match (&template.source, &template.source_record) {
        (Some(_), None) | (None, Some(_)) => {}
        (Some(_), Some(_)) => bail!("evidence matrix template mixes source forms"),
        (None, None) => bail!("evidence matrix template has no source"),
    }
    reject_disallowed_placeholder(&template.expected_ptx.mnemonic, "PTX mnemonic")?;
    Ok(())
}

pub(super) fn materialize_evidence_record(
    template: &EvidenceMatrixTemplate,
    defaults: &EvidenceRecordDefaults,
    bindings: &BTreeMap<String, String>,
    used_axes: &mut BTreeSet<String>,
    fixture_stages: &[EvidenceStage],
) -> Result<EvidenceRecord> {
    let id = expand_axis_placeholders(&template.id, bindings, used_axes, "evidence ID")?;
    let source = template
        .source
        .as_ref()
        .map(|source| expand_evidence_source(source, bindings, used_axes))
        .transpose()?;
    let source_record = template
        .source_record
        .as_deref()
        .map(|value| expand_axis_placeholders(value, bindings, used_axes, "source record"))
        .transpose()?;
    let llvm_symbol = template
        .llvm_symbol
        .as_deref()
        .map(|value| expand_axis_placeholders(value, bindings, used_axes, "LLVM symbol"))
        .transpose()?;
    validate_expanded_matrix_identity(
        &id,
        source.as_ref(),
        source_record.as_deref(),
        llvm_symbol.as_deref(),
    )?;
    let resolved_llvm_symbol = select_fact(
        &template.facts.resolved_llvm_symbol,
        &defaults.resolved_llvm_symbol,
    )
    .map(|value| expand_axis_placeholders(&value, bindings, used_axes, "resolved LLVM symbol"))
    .transpose()?;
    let mut expected_ptx = template.expected_ptx.clone();
    for modifier in &mut expected_ptx.modifiers {
        *modifier = expand_axis_placeholders(modifier, bindings, used_axes, "PTX modifier")?;
    }
    for operand in &mut expected_ptx.operands {
        if let OperandPattern::Exact { value } = operand {
            *value = expand_axis_placeholders(value, bindings, used_axes, "exact PTX operand")?;
        }
    }

    let mut stages = defaults.stages.clone();
    stages.extend(template.facts.stages.iter().cloned());
    stages.extend(fixture_stages.iter().cloned());
    let target_triple = required_fact(
        select_fact(&template.facts.target_triple, &defaults.target_triple),
        &id,
        "target triple",
    )?;
    let gpu_target = required_fact(
        select_fact(&template.facts.gpu_target, &defaults.gpu_target),
        &id,
        "GPU target",
    )?;
    let ptx_feature = required_fact(
        select_fact(&template.facts.ptx_feature, &defaults.ptx_feature),
        &id,
        "PTX feature",
    )?;
    let status = required_fact(
        select_fact(&template.facts.status, &defaults.status),
        &id,
        "status",
    )?;
    let source_is_ptx_native = matches!(source, Some(IntrinsicSource::PtxNative { .. }));
    Ok(EvidenceRecord {
        id,
        source,
        source_record,
        llvm_symbol,
        resolved_llvm_symbol,
        llvm_arguments: select_fact(&template.facts.llvm_arguments, &defaults.llvm_arguments)
            .unwrap_or_default(),
        llvm_results: select_fact(&template.facts.llvm_results, &defaults.llvm_results)
            .unwrap_or_default(),
        concrete_llvm_arguments: select_fact(
            &template.facts.concrete_llvm_arguments,
            &defaults.concrete_llvm_arguments,
        )
        .unwrap_or_default(),
        concrete_llvm_results: select_fact(
            &template.facts.concrete_llvm_results,
            &defaults.concrete_llvm_results,
        )
        .unwrap_or_default(),
        target_triple,
        gpu_target,
        ptx_feature,
        status,
        stages,
        // Declaration canonicalization is a statement about an imported
        // LLVM declaration; a PTX-native record has none, so the file-level
        // default must not invent the fact for it.
        declaration_attributes_canonicalized: if source_is_ptx_native {
            template.facts.declaration_attributes_canonicalized
        } else {
            template
                .facts
                .declaration_attributes_canonicalized
                .or(defaults.declaration_attributes_canonicalized)
        },
        runtime_validation: template
            .facts
            .runtime_validation
            .or(defaults.runtime_validation),
        expected_ptx,
    })
}

pub(super) fn validate_expanded_matrix_identity(
    id: &str,
    source: Option<&IntrinsicSource>,
    source_record: Option<&str>,
    llvm_symbol: Option<&str>,
) -> Result<()> {
    ensure!(!id.is_empty(), "expanded evidence has an empty ID");
    let imported_source = match (source, source_record) {
        (Some(IntrinsicSource::LlvmImported { source_record }), None) => {
            ensure!(
                !source_record.is_empty(),
                "expanded evidence {id} has an empty source record"
            );
            true
        }
        (Some(IntrinsicSource::PtxNative { instruction }), None) => {
            ensure!(
                !instruction.is_empty(),
                "expanded evidence {id} has an empty PTX source"
            );
            false
        }
        (None, Some(source_record)) => {
            ensure!(
                !source_record.is_empty(),
                "expanded evidence {id} has an empty source record"
            );
            true
        }
        _ => unreachable!("matrix source shape was validated before expansion"),
    };
    if imported_source {
        ensure!(
            llvm_symbol.is_some_and(|symbol| !symbol.is_empty()),
            "expanded imported evidence {id} has no LLVM symbol"
        );
    } else {
        ensure!(
            llvm_symbol.is_none(),
            "expanded PTX-native evidence {id} invents an LLVM symbol"
        );
    }
    Ok(())
}

pub(super) fn select_fact<T: Clone>(specific: &Option<T>, default: &Option<T>) -> Option<T> {
    specific.clone().or_else(|| default.clone())
}

pub(super) fn required_fact(value: Option<String>, id: &str, field: &str) -> Result<String> {
    let value = value.with_context(|| format!("expanded evidence {id} has no {field}"))?;
    ensure!(
        !value.trim().is_empty(),
        "expanded evidence {id} has an empty {field}"
    );
    Ok(value)
}

pub(super) fn expand_evidence_source(
    source: &IntrinsicSource,
    bindings: &BTreeMap<String, String>,
    used_axes: &mut BTreeSet<String>,
) -> Result<IntrinsicSource> {
    Ok(match source {
        IntrinsicSource::LlvmImported { source_record } => IntrinsicSource::LlvmImported {
            source_record: expand_axis_placeholders(
                source_record,
                bindings,
                used_axes,
                "tagged source record",
            )?,
        },
        IntrinsicSource::PtxNative { instruction } => IntrinsicSource::PtxNative {
            instruction: expand_axis_placeholders(
                instruction,
                bindings,
                used_axes,
                "PTX-native source",
            )?,
        },
    })
}

pub(super) fn expand_axis_placeholders(
    value: &str,
    bindings: &BTreeMap<String, String>,
    used_axes: &mut BTreeSet<String>,
    field: &str,
) -> Result<String> {
    let mut output = String::with_capacity(value.len());
    let mut rest = value;
    while let Some(position) = rest.find('$') {
        output.push_str(&rest[..position]);
        let placeholder = &rest[position..];
        ensure!(
            placeholder.starts_with("${"),
            "{field} contains malformed matrix placeholder {value:?}"
        );
        let close = placeholder
            .find('}')
            .with_context(|| format!("{field} contains an unterminated matrix placeholder"))?;
        let axis = &placeholder[2..close];
        ensure!(
            is_safe_matrix_token(axis),
            "{field} contains malformed matrix axis {axis:?}"
        );
        let replacement = bindings
            .get(axis)
            .with_context(|| format!("{field} references unknown matrix axis {axis}"))?;
        output.push_str(replacement);
        used_axes.insert(axis.to_owned());
        rest = &placeholder[close + 1..];
    }
    output.push_str(rest);
    Ok(output)
}

pub(super) fn is_safe_matrix_token(value: &str) -> bool {
    let mut bytes = value.bytes();
    bytes.next().is_some_and(|byte| byte.is_ascii_lowercase())
        && bytes.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
}

pub(super) fn reject_default_placeholders(
    defaults: &EvidenceRecordDefaults,
    context: &str,
    allow_resolved_symbol: bool,
) -> Result<()> {
    if !allow_resolved_symbol && let Some(value) = &defaults.resolved_llvm_symbol {
        reject_disallowed_placeholder(value, context)?;
    }
    for value in defaults
        .target_triple
        .iter()
        .chain(defaults.gpu_target.iter())
        .chain(defaults.ptx_feature.iter())
        .chain(defaults.status.iter())
        .chain(defaults.llvm_arguments.iter().flatten())
        .chain(defaults.llvm_results.iter().flatten())
        .chain(defaults.concrete_llvm_arguments.iter().flatten())
        .chain(defaults.concrete_llvm_results.iter().flatten())
    {
        reject_disallowed_placeholder(value, context)?;
    }
    reject_stage_placeholders(&defaults.stages, context)
}

pub(super) fn reject_stage_placeholders(stages: &[EvidenceStage], context: &str) -> Result<()> {
    for stage in stages {
        for value in stage
            .targets
            .iter()
            .chain(std::iter::once(&stage.representation))
            .chain(std::iter::once(&stage.detail))
            .chain(stage.tool_path.iter())
            .chain(stage.tool_version.iter())
            .chain(stage.tool_sha256.iter())
        {
            reject_disallowed_placeholder(value, context)?;
        }
    }
    Ok(())
}

pub(super) fn reject_disallowed_placeholder(value: &str, field: &str) -> Result<()> {
    ensure!(
        !value.contains("${"),
        "{field} cannot contain matrix placeholders"
    );
    Ok(())
}

pub(super) fn validate_stage_pairs(stages: &[EvidenceStage], id: &str) -> Result<()> {
    let mut identities = Vec::new();
    for stage in stages {
        let identity = (stage.stage, stage.mechanism, stage.targets.clone());
        ensure!(
            !identities.contains(&identity),
            "expanded evidence {id} has conflicting duplicate {:?}/{:?} stage targets {:?}",
            stage.stage,
            stage.mechanism,
            stage.targets
        );
        identities.push(identity);
    }
    Ok(())
}

pub(super) fn read_evidence(repo_root: &Path) -> Result<(Vec<EvidenceFile>, Vec<String>)> {
    let directory = repo_root.join("intrinsics/evidence");
    let mut paths: Vec<PathBuf> = fs::read_dir(&directory)
        .with_context(|| format!("read {}", directory.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("json"))
        .collect();
    paths.sort();
    ensure!(
        !paths.is_empty(),
        "no evidence JSON files in {}",
        directory.display()
    );
    let mut files = Vec::with_capacity(paths.len());
    let mut hashes = Vec::with_capacity(paths.len());
    for path in paths {
        let file = read_evidence_file(&path)?;
        let name = path.file_name().unwrap().to_string_lossy();
        hashes.push(format!("{name}:{}", sha256_text_file(&path)?));
        files.push(file);
    }
    Ok((files, hashes))
}

#[derive(Debug, Clone, Copy)]
pub(super) struct IndexedEvidence<'a> {
    pub(super) file: &'a EvidenceFile,
    pub(super) record: &'a EvidenceRecord,
    pub(super) backend_version: &'a str,
    pub(super) backend_sha256: &'a str,
}

pub(super) fn index_evidence<'a>(
    files: &'a [EvidenceFile],
    llvm_revision: &str,
) -> Result<BTreeMap<(&'a str, &'a str), IndexedEvidence<'a>>> {
    let mut result = BTreeMap::new();
    for file in files {
        ensure!(
            !file.backend_profile.trim().is_empty() && !file.llvm_revision.trim().is_empty(),
            "evidence file has no concrete backend profile or LLVM revision"
        );
        ensure!(
            !file.backend_version.trim().is_empty() && !file.backend_sha256.trim().is_empty(),
            "evidence does not identify the backend binary"
        );
        if file.backend_kind != Some(IntrinsicBackend::LibNvvm) {
            ensure!(
                file.llvm_revision == llvm_revision,
                "selected evidence LLVM revision {} does not match pinned {}",
                file.llvm_revision,
                llvm_revision
            );
        }
        for record in &file.records {
            ensure!(
                result
                    .insert(
                        (file.backend_profile.as_str(), record.id.as_str()),
                        IndexedEvidence {
                            file,
                            record,
                            backend_version: &file.backend_version,
                            backend_sha256: &file.backend_sha256,
                        },
                    )
                    .is_none(),
                "duplicate evidence for catalog ID {}",
                record.id
            );
        }
    }
    Ok(result)
}

pub(super) fn validate_evidence(
    policy: &OverlayIntrinsic,
    evidence: &IndexedEvidence<'_>,
    lowering: Option<&crate::model::OverlayBackendLowering>,
) -> Result<()> {
    let record = evidence.record;
    record.expected_ptx.validate().map_err(|reason| {
        anyhow::anyhow!(
            "{} evidence has an invalid expected PTX pattern: {reason}",
            policy.id
        )
    })?;
    let policy_source = resolve_policy_source(policy)?;
    let evidence_source = match (&record.source, &record.source_record) {
        (None, Some(source_record)) => IntrinsicSource::LlvmImported {
            source_record: source_record.clone(),
        },
        (Some(source), None) => source.clone(),
        (Some(_), Some(_)) => bail!(
            "{} evidence mixes tagged source with legacy source_record",
            policy.id
        ),
        (None, None) => bail!("{} evidence has no source provenance", policy.id),
    };
    ensure!(
        evidence_source == policy_source,
        "{} evidence source provenance mismatch",
        policy.id
    );
    ensure!(
        record.llvm_symbol == policy.llvm_symbol
            && record.llvm_arguments == policy.llvm_arguments
            && record.llvm_results == policy.llvm_results,
        "{} evidence signature mismatch",
        policy.id
    );
    if matches!(policy_source, IntrinsicSource::PtxNative { .. }) {
        ensure!(
            record.llvm_symbol.is_none()
                && record.resolved_llvm_symbol.is_none()
                && record.llvm_arguments.is_empty()
                && record.llvm_results.is_empty()
                && record.concrete_llvm_arguments.is_empty()
                && record.concrete_llvm_results.is_empty()
                && record.declaration_attributes_canonicalized.is_none(),
            "{} PTX-native evidence must not invent LLVM declaration facts",
            policy.id
        );
    }
    ensure!(
        record.expected_ptx == policy.expected_ptx,
        "{} evidence PTX expectation mismatch",
        policy.id
    );
    ensure!(
        matches!(record.status.as_str(), "lowered" | "validated" | "executed"),
        "{} evidence status {} is too weak to generate a lowering",
        policy.id,
        record.status
    );
    ensure!(
        !record.target_triple.is_empty()
            && !record.gpu_target.is_empty()
            && !record.ptx_feature.is_empty(),
        "{} evidence omits its full target profile",
        policy.id
    );
    if let Some(lowering) = lowering {
        ensure!(
            evidence.file.backend_kind == Some(lowering.backend),
            "{} evidence profile {} has the wrong backend kind",
            policy.id,
            evidence.file.backend_profile
        );
        match record.status.as_str() {
            "executed" => ensure!(
                record.runtime_validation == Some(RuntimeValidation::Executed),
                "{} executed evidence must record runtime_validation = executed",
                policy.id
            ),
            _ => ensure!(
                record.runtime_validation == Some(RuntimeValidation::Unexecuted),
                "{} non-executed backend evidence must record runtime_validation = unexecuted",
                policy.id
            ),
        }
        ensure!(
            !record.stages.is_empty(),
            "{} backend evidence omits compilation stages",
            policy.id
        );
        ensure!(
            record.stages.iter().any(|stage| {
                stage.stage == EvidenceStageKind::BackendCodegen
                    && stage.mechanism == Some(lowering.mechanism)
                    && stage.outcome == "succeeded"
            }),
            "{} evidence has no successful backend-codegen stage for {:?}",
            policy.id,
            lowering.mechanism
        );
        validate_selected_stage_targets(policy, record, lowering)?;
        if lowering.mechanism == BackendLoweringMechanism::TypedNvvm {
            validate_typed_llvm_evidence(policy, record)?;
        }
        validate_packed_conversion_backend_evidence(policy, record, lowering)?;
        validate_scalar_conversion_backend_evidence(policy, record, lowering)?;
        validate_scalar_arithmetic_backend_evidence(policy, record, lowering)?;
        validate_inline_ptx_fallback_evidence(policy, record, lowering)?;
        if lowering.backend == IntrinsicBackend::LlvmNvptx
            && matches!(record.status.as_str(), "validated" | "executed")
        {
            ensure!(
                has_valid_ptx_assembly_stage(record, lowering.mechanism),
                "{} validated LLVM-NVPTX evidence requires a successful PTX-assembly stage with exact tool identity",
                policy.id
            );
        } else if lowering.backend == IntrinsicBackend::LibNvvm
            && matches!(record.status.as_str(), "validated" | "executed")
        {
            ensure!(
                has_valid_cubin_device_link_stage(record, lowering.mechanism),
                "{} validated libNVVM evidence requires a successful cubin-producing device-link stage with exact tool identity",
                policy.id
            );
        }
        if record.status == "executed" {
            ensure!(
                record.stages.iter().any(|stage| {
                    stage.stage == EvidenceStageKind::Runtime
                        && stage.mechanism == Some(lowering.mechanism)
                        && stage.outcome == "succeeded"
                }),
                "{} executed evidence requires a successful runtime stage for the selected mechanism",
                policy.id
            );
        }
    }
    Ok(())
}

pub(super) fn validate_inline_ptx_fallback_evidence(
    policy: &OverlayIntrinsic,
    record: &EvidenceRecord,
    lowering: &crate::model::OverlayBackendLowering,
) -> Result<()> {
    if !matches!(
        policy.family.as_str(),
        "cluster_barrier" | "wgmma_control" | "elect"
    ) || lowering.backend != IntrinsicBackend::LibNvvm
    {
        return Ok(());
    }
    for stage in [
        EvidenceStageKind::BackendCodegen,
        EvidenceStageKind::DeviceLink,
    ] {
        ensure!(
            record.stages.iter().any(|candidate| {
                candidate.stage == stage
                    && candidate.mechanism == Some(BackendLoweringMechanism::TypedNvvm)
                    && candidate.outcome == "failed"
            }),
            "{} libNVVM inline-PTX evidence must record the failed typed-NVVM {:?} comparison",
            policy.id,
            stage
        );
    }
    ensure!(
        !record.stages.iter().any(|candidate| {
            candidate.stage == EvidenceStageKind::DeviceLink
                && candidate.mechanism == Some(BackendLoweringMechanism::TypedNvvm)
                && candidate.outcome == "succeeded"
        }),
        "{} libNVVM evidence cannot select inline PTX after a successful typed-NVVM terminal",
        policy.id
    );
    Ok(())
}

pub(super) fn validate_packed_conversion_backend_evidence(
    policy: &OverlayIntrinsic,
    record: &EvidenceRecord,
    lowering: &crate::model::OverlayBackendLowering,
) -> Result<()> {
    if policy.family != "packed_conversion" {
        return Ok(());
    }
    match lowering.backend {
        IntrinsicBackend::LlvmNvptx => {
            validate_typed_llvm_evidence(policy, record)?;
            for stage in [
                EvidenceStageKind::DeclarationCanonicalization,
                EvidenceStageKind::BackendCodegen,
            ] {
                ensure!(
                    successful_stage(record, BackendLoweringMechanism::TypedNvvm, stage).is_some(),
                    "{} LLVM packed-conversion evidence requires a successful auxiliary typed-NVVM {:?} stage",
                    policy.id,
                    stage
                );
            }
            ensure!(
                has_valid_ptx_assembly_stage(record, BackendLoweringMechanism::TypedNvvm),
                "{} LLVM packed-conversion evidence requires a successful auxiliary typed-NVVM PTX-assembly stage with exact tool identity",
                policy.id
            );
            ensure!(
                matches!(record.status.as_str(), "validated" | "executed"),
                "{} LLVM packed-conversion evidence requires validated evidence status for its auxiliary typed-NVVM terminal stage",
                policy.id
            );
            let typed_lowering = crate::model::OverlayBackendLowering {
                mechanism: BackendLoweringMechanism::TypedNvvm,
                ..lowering.clone()
            };
            validate_selected_stage_targets(policy, record, &typed_lowering)?;
            Ok(())
        }
        IntrinsicBackend::LibNvvm => {
            ensure!(
                record.resolved_llvm_symbol.is_none()
                    && record.concrete_llvm_arguments.is_empty()
                    && record.concrete_llvm_results.is_empty()
                    && record.declaration_attributes_canonicalized.is_none()
                    && !record.stages.iter().any(|stage| {
                        stage.mechanism == Some(BackendLoweringMechanism::TypedNvvm)
                    }),
                "{} libNVVM inline-PTX evidence must not claim typed LLVM support",
                policy.id
            );
            Ok(())
        }
    }
}

pub(super) fn validate_scalar_conversion_backend_evidence(
    policy: &OverlayIntrinsic,
    record: &EvidenceRecord,
    lowering: &crate::model::OverlayBackendLowering,
) -> Result<()> {
    if policy.family != "scalar_conversion" {
        return Ok(());
    }
    match lowering.backend {
        IntrinsicBackend::LlvmNvptx => ensure!(
            successful_stage(
                record,
                BackendLoweringMechanism::TypedNvvm,
                EvidenceStageKind::DeclarationCanonicalization,
            )
            .is_some(),
            "{} LLVM scalar-conversion evidence must canonicalize the typed declaration",
            policy.id
        ),
        IntrinsicBackend::LibNvvm => ensure!(
            record.resolved_llvm_symbol.is_none()
                && record.concrete_llvm_arguments.is_empty()
                && record.concrete_llvm_results.is_empty()
                && record.declaration_attributes_canonicalized.is_none(),
            "{} libNVVM scalar-conversion evidence must describe the selected inline-PTX route",
            policy.id
        ),
    };
    Ok(())
}

pub(super) fn validate_scalar_arithmetic_backend_evidence(
    policy: &OverlayIntrinsic,
    record: &EvidenceRecord,
    lowering: &crate::model::OverlayBackendLowering,
) -> Result<()> {
    if policy.family != "scalar_arithmetic" {
        return Ok(());
    }
    match (lowering.backend, lowering.mechanism) {
        (IntrinsicBackend::LlvmNvptx, BackendLoweringMechanism::TypedNvvm) => ensure!(
            successful_stage(
                record,
                BackendLoweringMechanism::TypedNvvm,
                EvidenceStageKind::DeclarationCanonicalization,
            )
            .is_some(),
            "{} typed LLVM scalar-arithmetic evidence must canonicalize the declaration",
            policy.id
        ),
        (IntrinsicBackend::LlvmNvptx, BackendLoweringMechanism::InlinePtx) => {
            validate_typed_llvm_evidence(policy, record)?;
            ensure!(
                successful_stage(
                    record,
                    BackendLoweringMechanism::TypedNvvm,
                    EvidenceStageKind::DeclarationCanonicalization,
                )
                .is_some(),
                "{} inline LLVM scalar-arithmetic evidence must canonicalize its imported declaration",
                policy.id
            );
        }
        (IntrinsicBackend::LibNvvm, _) => ensure!(
            record.resolved_llvm_symbol.is_none()
                && record.concrete_llvm_arguments.is_empty()
                && record.concrete_llvm_results.is_empty()
                && record.declaration_attributes_canonicalized.is_none(),
            "{} libNVVM scalar-arithmetic evidence must describe the selected inline-PTX route",
            policy.id
        ),
    };
    Ok(())
}

pub(super) fn validate_typed_llvm_evidence(
    policy: &OverlayIntrinsic,
    record: &EvidenceRecord,
) -> Result<()> {
    let concrete_arguments = policy
        .llvm_arguments
        .iter()
        .map(|argument| {
            match argument.as_str() {
                "shared_cluster_ptr" => return Ok("ptr addrspace(7)".into()),
                "shared_ptr" => return Ok("ptr addrspace(3)".into()),
                "global_ptr" => return Ok("ptr addrspace(1)".into()),
                "ptr" => return Ok("ptr".into()),
                "anyptr" => {}
                _ => return Ok(argument.clone()),
            }
            match policy.selected_address_space.with_context(|| {
                format!(
                    "{} has a polymorphic LLVM pointer without a selected address space",
                    policy.id
                )
            })? {
                ImportedAddressSpace::Generic => Ok("ptr".into()),
                ImportedAddressSpace::Shared => Ok("ptr addrspace(3)".into()),
            }
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        record.resolved_llvm_symbol == policy.resolved_llvm_symbol
            && record.concrete_llvm_arguments == concrete_arguments
            && record.concrete_llvm_results == policy.llvm_results
            && record.declaration_attributes_canonicalized == Some(true),
        "{} typed LLVM evidence does not prove its resolved signature and canonical declaration attributes",
        policy.id
    );
    Ok(())
}

pub(super) fn validate_selected_stage_targets(
    policy: &OverlayIntrinsic,
    record: &EvidenceRecord,
    lowering: &crate::model::OverlayBackendLowering,
) -> Result<()> {
    let requirement = backend_target_requirement(policy, lowering)?;
    let blackwell_ldmatrix = policy.family == "ldmatrix"
        && policy
            .ldmatrix_variant
            .as_ref()
            .is_some_and(|variant| variant.shape != LdmatrixShape::M8n8);
    let f8f6f4_mma = is_f8f6f4_mma_target_matrix_policy(policy);
    let paired_alternatives = match &requirement.hardware {
        CatalogHardwareTarget::TargetMatrix { contracts } => Some(
            contracts
                .iter()
                .flat_map(|contract| contract.alternatives.iter().cloned())
                .collect::<Vec<_>>(),
        ),
        _ => None,
    };
    let target_matrix_evidence = blackwell_ldmatrix || f8f6f4_mma || paired_alternatives.is_some();
    if !target_matrix_evidence {
        let mut pairs = Vec::new();
        for stage in record.stages.iter().filter(|stage| {
            stage.mechanism == Some(lowering.mechanism) && stage.outcome == "succeeded"
        }) {
            let pair = (stage.stage, stage.mechanism);
            ensure!(
                !pairs.contains(&pair),
                "{} has multiple target-specific {:?}/{:?} stages outside reviewed target-matrix evidence",
                policy.id,
                stage.stage,
                stage.mechanism
            );
            pairs.push(pair);
        }
    }
    for stage in &record.stages {
        ensure!(
            !stage.targets.is_empty(),
            "{} evidence stage {:?} has no targets",
            policy.id,
            stage.stage
        );
        for target in &stage.targets {
            ensure!(
                is_normalized_stage_target(target),
                "{} evidence stage has unsupported target spelling {target:?}",
                policy.id
            );
        }
    }

    let terminal_kind = match lowering.backend {
        IntrinsicBackend::LlvmNvptx => EvidenceStageKind::PtxAssembly,
        IntrinsicBackend::LibNvvm => EvidenceStageKind::DeviceLink,
    };
    let expected_ptx = requirement.minimum_ptx.encoded();
    let expected_hardware = match &requirement.hardware {
        CatalogHardwareTarget::AnyOf { alternatives } if !alternatives.is_empty() => {
            alternatives.clone()
        }
        CatalogHardwareTarget::TargetMatrix { contracts } if !contracts.is_empty() => {
            let mut alternatives = contracts
                .iter()
                .flat_map(|contract| contract.alternatives.iter())
                .map(|alternative| alternative.hardware)
                .collect::<Vec<_>>();
            alternatives.sort_by_key(|hardware| target_hardware_sort_key(*hardware));
            alternatives.dedup();
            alternatives
        }
        _ => bail!(
            "{} selected backend stages require a hardware target",
            policy.id
        ),
    };
    if target_matrix_evidence {
        return validate_target_matrix_stage_targets(
            policy,
            record,
            lowering,
            terminal_kind,
            &expected_hardware,
            expected_ptx,
            paired_alternatives.as_deref(),
        );
    }
    let backend = successful_stage(
        record,
        lowering.mechanism,
        EvidenceStageKind::BackendCodegen,
    )
    .with_context(|| {
        format!(
            "{} has no successful selected backend-codegen stage",
            policy.id
        )
    })?;
    let mut required_stages = vec![backend];
    if matches!(record.status.as_str(), "validated" | "executed") {
        required_stages.push(
            successful_stage(record, lowering.mechanism, terminal_kind).with_context(|| {
                format!("{} has no successful selected terminal stage", policy.id)
            })?,
        );
    }
    for stage in required_stages {
        let (hardware, ptx) = selected_stage_floor(stage)?;
        let allow_forward_minimum = stage.stage != EvidenceStageKind::BackendCodegen;
        let hardware_matches = expected_hardware.iter().any(|expected| {
            selected_stage_hardware_matches(hardware, *expected, allow_forward_minimum)
        });
        let ptx_matches = match lowering.backend {
            IntrinsicBackend::LlvmNvptx => ptx == expected_ptx,
            IntrinsicBackend::LibNvvm => ptx >= expected_ptx,
        };
        ensure!(
            hardware_matches && ptx_matches,
            "{} evidence stage {:?} targets {} / PTX {}.{} instead of a compatible target at catalog floor {} / PTX {}.{}",
            policy.id,
            stage.stage,
            describe_stage_hardware(hardware),
            ptx / 10,
            ptx % 10,
            expected_hardware
                .iter()
                .map(|hardware| describe_stage_hardware(*hardware))
                .collect::<Vec<_>>()
                .join(" or "),
            expected_ptx / 10,
            expected_ptx % 10
        );
    }
    if record.status == "executed" {
        let runtime = successful_stage(record, lowering.mechanism, EvidenceStageKind::Runtime)
            .with_context(|| {
                format!(
                    "{} executed evidence has no successful runtime stage",
                    policy.id
                )
            })?;
        let (hardware, ptx) = selected_stage_floor(runtime)?;
        let ptx_matches = match lowering.backend {
            IntrinsicBackend::LlvmNvptx => ptx == expected_ptx,
            IntrinsicBackend::LibNvvm => ptx >= expected_ptx,
        };
        ensure!(
            expected_hardware
                .iter()
                .any(|expected| selected_stage_hardware_matches(hardware, *expected, true))
                && ptx_matches,
            "{} runtime stage target does not satisfy its catalog floor",
            policy.id
        );
    }
    Ok(())
}

pub(super) fn validate_target_matrix_stage_targets(
    policy: &OverlayIntrinsic,
    record: &EvidenceRecord,
    lowering: &OverlayBackendLowering,
    terminal_kind: EvidenceStageKind,
    expected_hardware: &[CatalogHardwareAlternative],
    minimum_ptx: u16,
    paired_alternatives: Option<&[CatalogTargetAlternative]>,
) -> Result<()> {
    let mut required_kinds = vec![EvidenceStageKind::BackendCodegen];
    if matches!(record.status.as_str(), "validated" | "executed") {
        if lowering.backend == IntrinsicBackend::LibNvvm {
            required_kinds.push(EvidenceStageKind::PtxAssembly);
        }
        required_kinds.push(terminal_kind);
    }

    for kind in required_kinds {
        let stages = record
            .stages
            .iter()
            .filter(|stage| {
                stage.stage == kind
                    && stage.mechanism == Some(lowering.mechanism)
                    && stage.outcome == "succeeded"
            })
            .collect::<Vec<_>>();
        ensure!(
            stages.len() == expected_hardware.len(),
            "{} target-matrix {:?} evidence must contain one structured stage for each of its {} reviewed targets",
            policy.id,
            kind,
            expected_hardware.len()
        );

        let mut covered = Vec::new();
        for stage in stages {
            let (hardware, ptx) = selected_stage_floor(stage)?;
            ensure!(
                expected_hardware.contains(&hardware) && !covered.contains(&hardware),
                "{} target-matrix {:?} evidence has an unexpected or duplicate target {}",
                policy.id,
                kind,
                describe_stage_hardware(hardware)
            );
            covered.push(hardware);
            let exact_floor =
                target_matrix_ptx_floor(policy, hardware, paired_alternatives, false)?;
            let ptx_matches = match lowering.backend {
                IntrinsicBackend::LlvmNvptx => ptx == exact_floor,
                IntrinsicBackend::LibNvvm => {
                    ptx >= paired_alternatives.map_or(minimum_ptx, |_| exact_floor)
                }
            };
            ensure!(
                ptx_matches,
                "{} target-matrix {:?} evidence records the wrong PTX floor for {}",
                policy.id,
                kind,
                describe_stage_hardware(hardware)
            );
            if matches!(
                kind,
                EvidenceStageKind::PtxAssembly | EvidenceStageKind::DeviceLink
            ) {
                ensure!(
                    stage
                        .tool_path
                        .as_deref()
                        .is_some_and(|value| !value.is_empty())
                        && stage
                            .tool_version
                            .as_deref()
                            .is_some_and(|value| !value.is_empty())
                        && stage
                            .tool_sha256
                            .as_deref()
                            .is_some_and(|value| !value.is_empty()),
                    "{} target-matrix {:?} evidence for {} does not identify its tool",
                    policy.id,
                    kind,
                    describe_stage_hardware(hardware)
                );
            }
            if kind == EvidenceStageKind::DeviceLink {
                ensure!(
                    stage.artifact_kind == Some(EvidenceArtifactKind::Cubin),
                    "{} target-matrix device-link evidence for {} is not a cubin",
                    policy.id,
                    describe_stage_hardware(hardware)
                );
            }
        }
    }

    if record.status == "executed" {
        let runtime = record
            .stages
            .iter()
            .filter(|stage| {
                stage.stage == EvidenceStageKind::Runtime
                    && stage.mechanism == Some(lowering.mechanism)
                    && stage.outcome == "succeeded"
            })
            .collect::<Vec<_>>();
        ensure!(
            runtime.len() == 1,
            "{} executed target-matrix evidence requires one successful runtime stage",
            policy.id
        );
        let (hardware, ptx) = selected_stage_floor(runtime[0])?;
        ensure!(
            expected_hardware
                .iter()
                .any(|expected| { selected_stage_hardware_matches(hardware, *expected, true) }),
            "{} runtime target {} is outside its target matrix",
            policy.id,
            describe_stage_hardware(hardware)
        );
        let floor = target_matrix_ptx_floor(policy, hardware, paired_alternatives, true)?;
        let ptx_matches = match lowering.backend {
            IntrinsicBackend::LlvmNvptx => ptx == floor,
            IntrinsicBackend::LibNvvm => ptx >= floor,
        };
        ensure!(
            ptx_matches,
            "{} runtime target {} records PTX {}.{} instead of its paired floor {}.{}",
            policy.id,
            describe_stage_hardware(hardware),
            ptx / 10,
            ptx % 10,
            floor / 10,
            floor % 10
        );
    }
    Ok(())
}

pub(super) fn target_matrix_ptx_floor(
    policy: &OverlayIntrinsic,
    hardware: CatalogHardwareAlternative,
    paired_alternatives: Option<&[CatalogTargetAlternative]>,
    allow_forward_minimum: bool,
) -> Result<u16> {
    if let Some(alternatives) = paired_alternatives {
        return alternatives
            .iter()
            .filter(|alternative| {
                selected_stage_hardware_matches(
                    hardware,
                    alternative.hardware,
                    allow_forward_minimum,
                )
            })
            .map(|alternative| alternative.minimum_ptx.encoded())
            .min()
            .with_context(|| {
                format!(
                    "{} target matrix has no PTX floor for {}",
                    policy.id,
                    describe_stage_hardware(hardware)
                )
            });
    }
    if is_f8f6f4_mma_target_matrix_policy(policy) {
        f8f6f4_llvm_ptx_floor(hardware)
    } else {
        blackwell_ldmatrix_llvm_ptx_floor(hardware)
    }
}

pub(super) fn f8f6f4_llvm_ptx_floor(hardware: CatalogHardwareAlternative) -> Result<u16> {
    match hardware {
        CatalogHardwareAlternative::ExactArchitecture { sm: 120 }
        | CatalogHardwareAlternative::FamilyTarget { sm: 120 }
        | CatalogHardwareAlternative::ExactArchitecture { sm: 121 }
        | CatalogHardwareAlternative::FamilyTarget { sm: 121 } => {
            recorded_stage_ptx_floor(hardware, 87)
        }
        _ => bail!(
            "{} is not a reviewed f8f6f4 MMA target",
            describe_stage_hardware(hardware)
        ),
    }
}

pub(super) fn blackwell_ldmatrix_llvm_ptx_floor(
    hardware: CatalogHardwareAlternative,
) -> Result<u16> {
    match hardware {
        CatalogHardwareAlternative::ExactArchitecture {
            sm: 100 | 103 | 110 | 120 | 121,
        }
        | CatalogHardwareAlternative::FamilyTarget {
            sm: 100 | 103 | 120 | 121,
        }
        | CatalogHardwareAlternative::FamilyTarget { sm: 110 } => {
            recorded_stage_ptx_floor(hardware, 86)
        }
        _ => bail!(
            "{} is not a reviewed Blackwell ldmatrix target",
            describe_stage_hardware(hardware)
        ),
    }
}

pub(super) fn recorded_stage_ptx_floor(
    hardware: CatalogHardwareAlternative,
    instruction_floor: u16,
) -> Result<u16> {
    let (sm, suffix) = match hardware {
        CatalogHardwareAlternative::MinimumSm { sm } => (sm, None),
        CatalogHardwareAlternative::ExactArchitecture { sm } => (sm, Some('a')),
        CatalogHardwareAlternative::FamilyTarget { sm } => (sm, Some('f')),
    };
    let arch = CudaArch::new(u32::from(sm), suffix)?;
    Ok(instruction_floor.max(recorded_ptx_floor(&arch)?))
}

pub(super) fn successful_stage(
    record: &EvidenceRecord,
    mechanism: BackendLoweringMechanism,
    kind: EvidenceStageKind,
) -> Option<&crate::model::EvidenceStage> {
    record.stages.iter().find(|stage| {
        stage.stage == kind && stage.mechanism == Some(mechanism) && stage.outcome == "succeeded"
    })
}

pub(super) fn is_normalized_stage_target(target: &str) -> bool {
    if let Some(value) = target.strip_prefix("ptx") {
        return value.len() == 2 && value.bytes().all(|byte| byte.is_ascii_digit());
    }
    parse_stage_hardware(target).is_some()
}

pub(super) fn parse_stage_hardware(target: &str) -> Option<CatalogHardwareAlternative> {
    let arch = target.parse::<CudaArch>().ok()?;
    let canonical = if target.starts_with("sm_") {
        arch.sm()
    } else if target.starts_with("compute_") {
        arch.compute()
    } else {
        return None;
    };
    if canonical != target || !matches!(arch.capability().to_string().len(), 2 | 3) {
        return None;
    }
    let sm = u16::try_from(arch.capability()).ok()?;
    Some(match arch.suffix() {
        None => CatalogHardwareAlternative::MinimumSm { sm },
        Some('a') => CatalogHardwareAlternative::ExactArchitecture { sm },
        Some('f') => CatalogHardwareAlternative::FamilyTarget { sm },
        _ => unreachable!(),
    })
}

pub(super) fn selected_stage_hardware_matches(
    actual: CatalogHardwareAlternative,
    expected: CatalogHardwareAlternative,
    allow_forward_minimum: bool,
) -> bool {
    match expected {
        CatalogHardwareAlternative::MinimumSm { sm: expected } => {
            if allow_forward_minimum {
                match actual {
                    CatalogHardwareAlternative::MinimumSm { sm }
                    | CatalogHardwareAlternative::ExactArchitecture { sm }
                    | CatalogHardwareAlternative::FamilyTarget { sm } => sm >= expected,
                }
            } else {
                actual == CatalogHardwareAlternative::MinimumSm { sm: expected }
            }
        }
        CatalogHardwareAlternative::ExactArchitecture { .. }
        | CatalogHardwareAlternative::FamilyTarget { .. } => actual == expected,
    }
}

pub(super) fn describe_stage_hardware(hardware: CatalogHardwareAlternative) -> String {
    match hardware {
        CatalogHardwareAlternative::MinimumSm { sm } => format!("sm_{sm}"),
        CatalogHardwareAlternative::ExactArchitecture { sm } => format!("sm_{sm}a"),
        CatalogHardwareAlternative::FamilyTarget { sm } => format!("sm_{sm}f"),
    }
}

pub(super) fn selected_stage_floor(
    stage: &crate::model::EvidenceStage,
) -> Result<(CatalogHardwareAlternative, u16)> {
    let mut hardware = None;
    let mut ptx = None;
    for target in &stage.targets {
        if let Some(value) = target.strip_prefix("ptx") {
            let value = value.parse::<u16>()?;
            ensure!(
                ptx.replace(value).is_none(),
                "stage has duplicate PTX targets"
            );
        } else {
            let value = parse_stage_hardware(target)
                .with_context(|| format!("stage has unsupported target spelling {target:?}"))?;
            ensure!(
                hardware.replace(value).is_none(),
                "stage has duplicate architecture targets"
            );
        }
    }
    Ok((
        hardware.context("selected stage has no architecture target")?,
        ptx.context("selected stage has no PTX target")?,
    ))
}

pub(super) fn has_valid_ptx_assembly_stage(
    record: &EvidenceRecord,
    mechanism: BackendLoweringMechanism,
) -> bool {
    has_valid_tool_stage(record, mechanism, EvidenceStageKind::PtxAssembly)
}

pub(super) fn has_valid_cubin_device_link_stage(
    record: &EvidenceRecord,
    mechanism: BackendLoweringMechanism,
) -> bool {
    has_valid_tool_stage(record, mechanism, EvidenceStageKind::DeviceLink)
        && successful_stage(record, mechanism, EvidenceStageKind::DeviceLink)
            .is_some_and(|stage| stage.artifact_kind == Some(EvidenceArtifactKind::Cubin))
}

pub(super) fn has_valid_tool_stage(
    record: &EvidenceRecord,
    mechanism: BackendLoweringMechanism,
    stage_kind: EvidenceStageKind,
) -> bool {
    record.stages.iter().any(|stage| {
        stage.stage == stage_kind
            && stage.mechanism == Some(mechanism)
            && stage.outcome == "succeeded"
            && stage
                .tool_path
                .as_deref()
                .is_some_and(|value| !value.is_empty())
            && stage
                .tool_version
                .as_deref()
                .is_some_and(|value| !value.is_empty())
            && stage.tool_sha256.as_deref().is_some_and(|value| {
                value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
            })
    })
}
