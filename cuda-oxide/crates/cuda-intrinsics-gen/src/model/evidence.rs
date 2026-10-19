/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::core::{BackendLoweringMechanism, IntrinsicBackend, IntrinsicSource, RuntimeValidation};
use crate::ptx::InstructionPattern;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceFile {
    pub schema: u32,
    pub backend_profile: String,
    #[serde(default)]
    pub backend_kind: Option<IntrinsicBackend>,
    pub llvm_revision: String,
    pub backend_version: String,
    pub backend_sha256: String,
    #[serde(default)]
    pub artifact_path: Option<String>,
    #[serde(default)]
    pub build_id_prefix: Option<String>,
    #[serde(default)]
    pub nvvm_ir_version: Option<String>,
    #[serde(default)]
    pub debug_ir_version: Option<String>,
    pub records: Vec<EvidenceRecord>,
}

/// Schema-6 evidence before matrix expansion.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceFileV6 {
    pub schema: u32,
    pub backend_profile: String,
    #[serde(default)]
    pub backend_kind: Option<IntrinsicBackend>,
    pub llvm_revision: String,
    pub backend_version: String,
    pub backend_sha256: String,
    #[serde(default)]
    pub artifact_path: Option<String>,
    #[serde(default)]
    pub build_id_prefix: Option<String>,
    #[serde(default)]
    pub nvvm_ir_version: Option<String>,
    #[serde(default)]
    pub debug_ir_version: Option<String>,
    #[serde(default)]
    pub defaults: EvidenceRecordDefaults,
    #[serde(default)]
    pub fixtures: Vec<EvidenceFixture>,
    #[serde(default)]
    pub matrices: Vec<EvidenceMatrix>,
    #[serde(default)]
    pub records: Vec<EvidenceRecord>,
}

/// Facts shared by every record in one evidence matrix.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceRecordDefaults {
    #[serde(default)]
    pub resolved_llvm_symbol: Option<String>,
    #[serde(default)]
    pub llvm_arguments: Option<Vec<String>>,
    #[serde(default)]
    pub llvm_results: Option<Vec<String>>,
    #[serde(default)]
    pub concrete_llvm_arguments: Option<Vec<String>>,
    #[serde(default)]
    pub concrete_llvm_results: Option<Vec<String>>,
    #[serde(default)]
    pub target_triple: Option<String>,
    #[serde(default)]
    pub gpu_target: Option<String>,
    #[serde(default)]
    pub ptx_feature: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub stages: Vec<EvidenceStage>,
    #[serde(default)]
    pub declaration_attributes_canonicalized: Option<bool>,
    #[serde(default)]
    pub runtime_validation: Option<RuntimeValidation>,
}

/// One shared fixture and the number of records it covers.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceFixture {
    pub id: String,
    pub coverage_count: usize,
    pub stages: Vec<EvidenceStage>,
}

/// One Cartesian evidence matrix.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceMatrix {
    pub axes: Vec<EvidenceMatrixAxis>,
    pub product_count: usize,
    #[serde(default)]
    pub fixtures: Vec<String>,
    pub template: EvidenceMatrixTemplate,
}

/// One named matrix axis.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceMatrixAxis {
    pub name: String,
    pub values: Vec<String>,
}

/// Identity and matrix-specific facts for one record template.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceMatrixTemplate {
    pub id: String,
    #[serde(default)]
    pub source: Option<IntrinsicSource>,
    #[serde(default)]
    pub source_record: Option<String>,
    #[serde(deserialize_with = "deserialize_required_optional_string")]
    pub llvm_symbol: Option<String>,
    pub expected_ptx: InstructionPattern,
    #[serde(default)]
    pub facts: EvidenceRecordDefaults,
}

fn deserialize_required_optional_string<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<String>::deserialize(deserializer)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceRecord {
    pub id: String,
    #[serde(default)]
    pub source: Option<IntrinsicSource>,
    #[serde(default)]
    pub source_record: Option<String>,
    #[serde(default)]
    pub llvm_symbol: Option<String>,
    #[serde(default)]
    pub resolved_llvm_symbol: Option<String>,
    #[serde(default)]
    pub llvm_arguments: Vec<String>,
    #[serde(default)]
    pub llvm_results: Vec<String>,
    #[serde(default)]
    pub concrete_llvm_arguments: Vec<String>,
    #[serde(default)]
    pub concrete_llvm_results: Vec<String>,
    pub target_triple: String,
    pub gpu_target: String,
    pub ptx_feature: String,
    pub status: String,
    #[serde(default)]
    pub stages: Vec<EvidenceStage>,
    #[serde(default)]
    pub declaration_attributes_canonicalized: Option<bool>,
    #[serde(default)]
    pub runtime_validation: Option<RuntimeValidation>,
    pub expected_ptx: InstructionPattern,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceStage {
    pub targets: Vec<String>,
    pub representation: String,
    pub stage: EvidenceStageKind,
    #[serde(default)]
    pub mechanism: Option<BackendLoweringMechanism>,
    pub outcome: String,
    pub detail: String,
    #[serde(default)]
    pub artifact_kind: Option<EvidenceArtifactKind>,
    #[serde(default)]
    pub tool_path: Option<String>,
    #[serde(default)]
    pub tool_version: Option<String>,
    #[serde(default)]
    pub tool_sha256: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceArtifactKind {
    Cubin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceStageKind {
    DeclarationCanonicalization,
    BackendCodegen,
    DeviceLink,
    PtxAssembly,
    Runtime,
}
