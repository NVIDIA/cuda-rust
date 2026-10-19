/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::contracts::{
    ActiveMask, Clc, ClusterBarrier, ClusterMemory, CpAsyncControl, CpAsyncCopy, CpAsyncMbarrier,
    DebugControl, DotProduct, ExtendedMinMax, IntegerMinMax, LdmatrixAdapter, LdmatrixSafety,
    LdmatrixVariant, MbarrierBasic, MbarrierExtended, Movmatrix, PackedAlu, PackedAtomic,
    PackedConversion, Prmt, Redux, RegisterMma, ScalarArithmetic, ScalarConversion, ScalarMath,
    SparseMma, SpecialRegister, Tcgen05, Tma, Vote, WarpBarrier, WarpMatch, WarpShuffle,
    WgmmaControl,
};
use super::core::{BackendLoweringMechanism, IntrinsicBackend, IntrinsicSource};
use super::evidence::EvidenceStage;
use super::imported::{ImportedAddressSpace, ImportedSelectionConstraints};
use crate::ptx::InstructionPattern;
use serde::{Deserialize, Serialize};
use std::{fmt, str::FromStr};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogFile {
    pub schema: u32,
    pub catalog_version: String,
    pub intrinsic_abi: u32,
    pub generator_version: String,
    pub source: CatalogSource,
    pub inputs: CatalogInputs,
    pub intrinsics: Vec<CatalogIntrinsic>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogSource {
    pub llvm_repository: String,
    pub llvm_revision: String,
    pub llvm_tblgen_version: String,
    pub llvm_tblgen_source_revision: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogInputs {
    pub imported_sha256: String,
    pub overlay_sha256: String,
    pub abi_ledger_sha256: String,
    pub evidence_sha256: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogIntrinsic {
    pub id: String,
    pub operation_key: String,
    pub family: String,
    pub source: IntrinsicSource,
    pub selections: Vec<CatalogSelection>,
    pub rust: CatalogRust,
    pub dialect: CatalogDialect,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llvm: Option<CatalogLlvm>,
    pub semantics: CatalogSemantics,
    pub target: CatalogTarget,
    pub backend: CatalogBackend,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub backend_lowerings: Vec<CatalogBackendLowering>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packed_atomic: Option<PackedAtomic>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub redux: Option<Redux>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vote: Option<Vote>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_mask: Option<ActiveMask>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warp_match: Option<WarpMatch>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warp_barrier: Option<WarpBarrier>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warp_shuffle: Option<WarpShuffle>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dot_product: Option<DotProduct>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packed_alu: Option<PackedAlu>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub integer_minmax: Option<IntegerMinMax>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packed_conversion: Option<PackedConversion>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scalar_conversion: Option<ScalarConversion>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scalar_arithmetic: Option<ScalarArithmetic>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scalar_math: Option<ScalarMath>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extended_minmax: Option<ExtendedMinMax>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cp_async_copy: Option<CpAsyncCopy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cp_async_control: Option<CpAsyncControl>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cp_async_mbarrier: Option<CpAsyncMbarrier>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mbarrier_basic: Option<MbarrierBasic>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub movmatrix: Option<Movmatrix>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mbarrier_extended: Option<MbarrierExtended>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub register_mma: Option<RegisterMma>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sparse_mma: Option<SparseMma>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prmt: Option<Prmt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster_barrier: Option<ClusterBarrier>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wgmma_control: Option<WgmmaControl>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub special_register: Option<SpecialRegister>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub debug_control: Option<DebugControl>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster_memory: Option<ClusterMemory>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clc: Option<Clc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tma: Option<Tma>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tcgen05: Option<Tcgen05>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ldmatrix: Option<CatalogLdmatrix>,
    pub lowering: String,
    pub expected_ptx: InstructionPattern,
    pub summary: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogSelection {
    pub source_record: String,
    pub asm: String,
    pub predicates: Vec<String>,
    #[serde(
        default,
        skip_serializing_if = "ImportedSelectionConstraints::is_empty"
    )]
    pub constraints: ImportedSelectionConstraints,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogRust {
    pub abi_id: String,
    pub module: String,
    pub name: String,
    pub arguments: Vec<String>,
    pub result: String,
    pub safe: bool,
    pub must_use: bool,
    pub safe_allowlist_reason: Option<String>,
    pub canonical_path: String,
    pub public_path: String,
    pub compatibility_paths: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogDialect {
    pub op_type: String,
    pub op_name: String,
    pub operands: Vec<String>,
    pub results: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogLlvm {
    pub symbol: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved_symbol: Option<String>,
    pub arguments: Vec<String>,
    pub results: Vec<String>,
    pub properties: Vec<String>,
    pub result_facts: CatalogLlvmResultFacts,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogLlvmResultFacts {
    pub no_undef: bool,
    pub range: Option<CatalogHalfOpenRange>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogHalfOpenRange {
    pub lower: String,
    pub upper_exclusive: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogSemantics {
    pub pure: bool,
    pub memory: String,
    pub convergent: bool,
    pub execution_scope: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogTarget {
    pub minimum_ptx: PtxVersion,
    pub hardware: CatalogHardwareTarget,
    pub ptx_result: String,
    pub targets: String,
    pub ptx_isa_version: String,
    pub ptx_isa_section: String,
    pub ptx_isa_url: String,
}

/// A PTX ISA version encoded as `major * 10 + minor`.
///
/// PTX currently uses one decimal minor digit. The resolver validates that
/// shape before constructing this value, so generated consumers compare a
/// number rather than reparsing policy text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PtxVersion(u16);

impl PtxVersion {
    pub const fn encoded(self) -> u16 {
        self.0
    }

    pub const fn major(self) -> u16 {
        self.0 / 10
    }

    pub const fn minor(self) -> u16 {
        self.0 % 10
    }
}

impl FromStr for PtxVersion {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let (major, minor) = value
            .split_once('.')
            .ok_or_else(|| "expected major.minor".to_owned())?;
        if major.is_empty()
            || !major.bytes().all(|byte| byte.is_ascii_digit())
            || minor.len() != 1
            || !minor.bytes().all(|byte| byte.is_ascii_digit())
        {
            return Err("expected numeric major.minor with one minor digit".to_owned());
        }
        let major: u16 = major.parse().map_err(|_| "major version is too large")?;
        let minor: u16 = minor.parse().unwrap();
        if format!("{major}.{minor}") != value {
            return Err("version is not in canonical major.minor form".to_owned());
        }
        let encoded = major
            .checked_mul(10)
            .and_then(|value| value.checked_add(minor))
            .ok_or_else(|| "version is too large".to_owned())?;
        Ok(Self(encoded))
    }
}

impl Serialize for PtxVersion {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for PtxVersion {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        value.parse().map_err(serde::de::Error::custom)
    }
}

impl fmt::Display for PtxVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}.{}", self.major(), self.minor())
    }
}

/// Reviewed hardware availability for an intrinsic.
///
/// Exact `a` and `f` targets stay distinct from monotonic minimums. A target
/// matrix also keeps each hardware target paired with its PTX floor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CatalogHardwareTarget {
    All,
    AnyOf {
        alternatives: Vec<CatalogHardwareAlternative>,
    },
    /// Closed selector contracts with their exact PTX and hardware pairs.
    TargetMatrix {
        contracts: Vec<CatalogTargetContract>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CatalogHardwareAlternative {
    MinimumSm { sm: u16 },
    ExactArchitecture { sm: u16 },
    FamilyTarget { sm: u16 },
}

/// One PTX floor paired with one hardware alternative.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogTargetAlternative {
    pub minimum_ptx: PtxVersion,
    pub hardware: CatalogHardwareAlternative,
}

/// One selector tuple and its exact PTX and hardware pairs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogTargetContract {
    pub selectors: Vec<TargetSelectorBinding>,
    pub alternatives: Vec<CatalogTargetAlternative>,
}

/// One field/value pair that selects a target contract.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSelectorBinding {
    pub name: String,
    pub value: String,
}

/// One selector-specific target contract from admission policy.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetContract {
    #[serde(default)]
    pub selectors: Vec<TargetSelectorBinding>,
    pub alternatives: Vec<TargetContractAlternative>,
}

/// One target spelling and PTX floor.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetContractAlternative {
    pub target: String,
    pub minimum_ptx: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogBackend {
    pub profile: String,
    pub version: String,
    pub sha256: String,
    pub status: String,
    pub target_triple: String,
    pub gpu_target: String,
    pub ptx_feature: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogBackendLowering {
    pub backend: IntrinsicBackend,
    pub mechanism: BackendLoweringMechanism,
    pub evidence_profile: String,
    pub target: CatalogTargetRequirement,
    pub version: String,
    pub sha256: String,
    pub artifact_path: Option<String>,
    pub build_id_prefix: Option<String>,
    pub status: String,
    pub stages: Vec<EvidenceStage>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogTargetRequirement {
    pub minimum_ptx: PtxVersion,
    pub hardware: CatalogHardwareTarget,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogLdmatrix {
    pub variant: LdmatrixVariant,
    pub safety: LdmatrixSafety,
    pub adapter: LdmatrixAdapter,
    pub selected_address_space: ImportedAddressSpace,
}

impl CatalogIntrinsic {
    pub fn scalar_width(&self) -> Option<u32> {
        match self.rust.result.as_str() {
            "u32" => Some(32),
            "u64" => Some(64),
            _ => None,
        }
    }

    pub fn llvm_identifier(&self) -> String {
        llvm_symbol_to_identifier(&self.llvm.as_ref().expect("LLVM-backed intrinsic").symbol)
    }

    pub fn resolved_llvm_identifier(&self) -> String {
        let llvm = self.llvm.as_ref().expect("LLVM-backed intrinsic");
        llvm_symbol_to_identifier(llvm.resolved_symbol.as_deref().unwrap_or(&llvm.symbol))
    }
}

fn llvm_symbol_to_identifier(symbol: &str) -> String {
    if !symbol.contains('_') {
        return symbol.replace('.', "_");
    }

    let suffix = symbol
        .strip_prefix("llvm.")
        .expect("LLVM intrinsic symbol must start with llvm.");
    let mut output = String::from("llvm__");
    for ch in suffix.chars() {
        match ch {
            '.' => output.push_str("_d"),
            '_' => output.push_str("_u"),
            ch if ch.is_ascii_alphanumeric() => output.push(ch),
            _ => panic!("LLVM intrinsic symbol contains an unsupported character"),
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llvm_identifier_encoding_preserves_literal_underscores() {
        assert_eq!(
            llvm_symbol_to_identifier("llvm.nvvm.read.ptx.sreg.tid.x"),
            "llvm_nvvm_read_ptx_sreg_tid_x"
        );
        assert_eq!(
            llvm_symbol_to_identifier("llvm.nvvm.wgmma.wait_group.sync.aligned"),
            "llvm__nvvm_dwgmma_dwait_ugroup_dsync_daligned"
        );
        assert_eq!(
            llvm_symbol_to_identifier(
                "llvm.nvvm.ldmatrix.sync.aligned.m16n16.x1.trans.b8x16.b4x16_p64.p3"
            ),
            "llvm__nvvm_dldmatrix_dsync_daligned_dm16n16_dx1_dtrans_db8x16_db4x16_up64_dp3"
        );
        assert_eq!(
            llvm_symbol_to_identifier(
                "llvm.nvvm.ldmatrix.sync.aligned.m8n16.x4.b8x16.b6x16_p32.p3"
            ),
            "llvm__nvvm_dldmatrix_dsync_daligned_dm8n16_dx4_db8x16_db6x16_up32_dp3"
        );
    }
}
