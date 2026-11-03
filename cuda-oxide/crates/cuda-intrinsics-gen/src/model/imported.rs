/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpstreamLock {
    pub schema: u32,
    pub rust_toolchain: LockedRustToolchain,
    pub llvm: LockedLlvm,
    pub llvm_tblgen: LockedTool,
    #[serde(default)]
    pub comparison_tools: Vec<LockedTool>,
    pub dumps: LockedDumps,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LockedRustToolchain {
    pub commit_hash: RustcCommit,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(try_from = "String")]
pub struct RustcCommit(String);

impl FromStr for RustcCommit {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.len() == 40
            && value
                .bytes()
                .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
        {
            Ok(Self(value.to_owned()))
        } else {
            Err(format!(
                "expected 40 lowercase hexadecimal characters, found {value:?}"
            ))
        }
    }
}

impl TryFrom<String> for RustcCommit {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}

impl fmt::Display for RustcCommit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LockedLlvm {
    pub repository: String,
    pub revision: String,
    pub provenance: String,
    pub public_output_allowed: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LockedTool {
    pub name: String,
    pub version_line: String,
    pub sha256: String,
    #[serde(default)]
    pub enforce_sha256: bool,
    pub provenance: String,
    pub built_from_llvm_revision: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LockedDumps {
    pub intrinsics_sha256: String,
    pub nvptx_sha256: String,
    pub normalized_imported_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportedFile {
    pub schema: u32,
    pub source: ImportedSource,
    pub intrinsics: Vec<ImportedIntrinsic>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportedSource {
    pub llvm_repository: String,
    pub llvm_revision: String,
    pub llvm_tblgen_version: String,
    pub llvm_tblgen_source_revision: String,
    pub intrinsics_json_sha256: String,
    pub nvptx_json_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportedIntrinsic {
    pub source_record: String,
    pub llvm_name: String,
    pub arguments: Vec<String>,
    pub results: Vec<String>,
    pub classes: Vec<String>,
    pub properties: Vec<String>,
    pub selections: Vec<ImportedSelection>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportedSelection {
    pub source_record: String,
    pub asm: String,
    pub predicates: Vec<String>,
    #[serde(
        default,
        skip_serializing_if = "ImportedSelectionConstraints::is_empty"
    )]
    pub constraints: ImportedSelectionConstraints,
}

/// Normalized constraints attached to an NVPTX instruction-selection record.
///
/// TableGen represents address-space-specific patterns through anonymous
/// `PatFrag` records and can bind intrinsic arguments to integer literals.
/// Keeping those facts separate from the assembly spelling lets policy select
/// an exact lowering without parsing PTX text.
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportedSelectionConstraints {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub address_space: Option<ImportedAddressSpace>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub immediate_bindings: Vec<ImportedImmediateBinding>,
}

/// One integer literal fixed by an NVPTX instruction-selection pattern.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportedImmediateBinding {
    pub argument_index: usize,
    pub value: i64,
}

impl ImportedSelectionConstraints {
    pub fn is_empty(&self) -> bool {
        self.address_space.is_none() && self.immediate_bindings.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImportedAddressSpace {
    Generic,
    Shared,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locked_tool_rejects_misspelled_security_field() {
        let input = r#"
name = "llvm-tblgen"
version_line = "LLVM version test"
sha256 = "abc"
enforce_sha25 = true
provenance = "test"
"#;
        let error = toml::from_str::<LockedTool>(input).unwrap_err();
        assert!(error.to_string().contains("enforce_sha25"));
    }

    #[test]
    fn imported_selection_rejects_misspelled_constraint() {
        let input = r#"{
            "source_record": "selection",
            "asm": "op;",
            "predicates": [],
            "constraints": { "adress_space": "shared" }
        }"#;
        let error = serde_json::from_str::<ImportedSelection>(input).unwrap_err();
        assert!(error.to_string().contains("adress_space"));
    }

    #[test]
    fn imported_selection_preserves_immediate_binding() {
        let input = r#"{
            "source_record": "DOT2_lo_ss",
            "asm": "dp2a.lo.s32.s32 $dst, $a, $b, $c;",
            "predicates": ["hasDotInstructions"],
            "constraints": {
                "immediate_bindings": [
                    { "argument_index": 2, "value": 0 }
                ]
            }
        }"#;
        let selection = serde_json::from_str::<ImportedSelection>(input).unwrap();
        assert_eq!(
            selection.constraints.immediate_bindings,
            [ImportedImmediateBinding {
                argument_index: 2,
                value: 0,
            }]
        );
        assert!(!selection.constraints.is_empty());
    }

    #[test]
    fn imported_immediate_binding_rejects_misspelled_index() {
        let input = r#"{
            "source_record": "DOT2_lo_ss",
            "asm": "dp2a.lo.s32.s32 $dst, $a, $b, $c;",
            "predicates": [],
            "constraints": {
                "immediate_bindings": [
                    { "argument_indx": 2, "value": 0 }
                ]
            }
        }"#;
        let error = serde_json::from_str::<ImportedSelection>(input).unwrap_err();
        assert!(error.to_string().contains("argument_indx"));
    }
}
