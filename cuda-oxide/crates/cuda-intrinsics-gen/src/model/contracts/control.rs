/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::super::core::RuntimeValidation;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WgmmaControlMode {
    Fence,
    CommitGroup,
    WaitGroup,
}

/// Closed semantics for one WGMMA control operation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WgmmaControl {
    pub mode: WgmmaControlMode,
    pub adapter: WgmmaControlAdapter,
    pub participation: WgmmaControlParticipation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WgmmaControlAdapter {
    NoArguments,
    ConstGenericU32ToI64Immediate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WgmmaControlParticipation {
    WarpgroupAllThreadsSameInstruction,
}

/// Closed semantic and lowering contract for one special-register read.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecialRegister {
    pub register: SpecialRegisterKind,
    pub observation: SpecialRegisterObservation,
    pub result_width: SpecialRegisterWidth,
    pub ptx_type: SpecialRegisterPtxType,
    pub output_constraint: SpecialRegisterOutputConstraint,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llvm_exclusion: Option<SpecialRegisterLlvmExclusion>,
}

/// Closed semantic contract for PTX debug controls.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DebugControl {
    pub operation: DebugControlOperation,
    pub adapter: DebugControlAdapter,
    pub runtime_validation: RuntimeValidation,
}

/// Closed semantic contract for Cluster Launch Control.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Clc {
    pub operation: ClcOperation,
    pub adapter: ClcAdapter,
    pub runtime_validation: RuntimeValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClcOperation {
    TryCancel,
    TryCancelMulticast,
    QueryIsCanceled,
    QueryGetFirstCtaidX,
    QueryGetFirstCtaidY,
    QueryGetFirstCtaidZ,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClcAdapter {
    GenericPointersToShared,
    PairU64ToI128BoolToU32,
    PairU64ToI128U32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecialRegisterKind {
    Clock,
    Clock64,
    Globaltimer,
    Envreg1,
    Envreg2,
    Smid,
    Nsmid,
    Gridid,
    Warpid,
    Nwarpid,
    DynamicSmemSize,
    TotalSmemSize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DebugControlOperation {
    Trap,
    Breakpoint,
    Pmevent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecialRegisterObservation {
    StablePure,
    VolatileObservation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecialRegisterWidth {
    B32,
    B64,
}

impl SpecialRegisterWidth {
    pub const fn bits(self) -> u32 {
        match self {
            Self::B32 => 32,
            Self::B64 => 64,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecialRegisterPtxType {
    B32,
    U32,
    U64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecialRegisterOutputConstraint {
    Register32,
    Register64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecialRegisterLlvmExclusion {
    pub source_record: String,
    pub llvm_symbol: String,
    pub imported_result_width: SpecialRegisterWidth,
    pub reason: SpecialRegisterLlvmExclusionReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecialRegisterLlvmExclusionReason {
    ResultWidthMismatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DebugControlAdapter {
    Direct,
    ConstGenericToImmediateU32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgmma_control_contract_is_closed() {
        let parsed: WgmmaControl = serde_json::from_str(
            r#"{
                "mode": "wait_group",
                "adapter": "const_generic_u32_to_i64_immediate",
                "participation": "warpgroup_all_threads_same_instruction"
            }"#,
        )
        .unwrap();
        assert_eq!(parsed.mode, WgmmaControlMode::WaitGroup);
        assert_eq!(
            parsed.adapter,
            WgmmaControlAdapter::ConstGenericU32ToI64Immediate
        );
        assert_eq!(
            parsed.participation,
            WgmmaControlParticipation::WarpgroupAllThreadsSameInstruction
        );

        let open_ended = r#"{
            "mode": "wait_group",
            "adapter": "const_generic_u32_to_i64_immediate",
            "participation": "warpgroup_all_threads_same_instruction",
            "custom_ptx": "wgmma.wait_group.sync.aligned 0;"
        }"#;
        assert!(serde_json::from_str::<WgmmaControl>(open_ended).is_err());
    }

    #[test]
    fn debug_control_contract_rejects_open_ended_policy() {
        let valid = r#"
operation = "pmevent"
adapter = "const_generic_to_immediate_u32"
runtime_validation = "unexecuted"
"#;
        let parsed = toml::from_str::<DebugControl>(valid).unwrap();
        assert_eq!(parsed.operation, DebugControlOperation::Pmevent);
        assert_eq!(
            parsed.adapter,
            DebugControlAdapter::ConstGenericToImmediateU32
        );

        for invalid in [
            valid.replace("operation = \"pmevent\"", "operation = \"profiler\""),
            valid.replace(
                "adapter = \"const_generic_to_immediate_u32\"",
                "adapter = \"runtime_u32\"",
            ),
            valid.replace(
                "runtime_validation = \"unexecuted\"",
                "runtime_validation = \"assumed\"",
            ),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(
                toml::from_str::<DebugControl>(&invalid).is_err(),
                "{invalid}"
            );
        }
    }

    #[test]
    fn clc_contract_rejects_open_ended_policy() {
        let valid = r#"
operation = "query_is_canceled"
adapter = "pair_u64_to_i128_bool_to_u32"
runtime_validation = "unexecuted"
"#;
        let parsed = toml::from_str::<Clc>(valid).unwrap();
        assert_eq!(parsed.operation, ClcOperation::QueryIsCanceled);
        assert_eq!(parsed.adapter, ClcAdapter::PairU64ToI128BoolToU32);

        for invalid in [
            valid.replace("operation = \"query_is_canceled\"", "operation = \"query\""),
            valid.replace(
                "adapter = \"pair_u64_to_i128_bool_to_u32\"",
                "adapter = \"pair_u64_to_i128\"",
            ),
            valid.replace(
                "runtime_validation = \"unexecuted\"",
                "runtime_validation = \"assumed\"",
            ),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(toml::from_str::<Clc>(&invalid).is_err(), "{invalid}");
        }
    }
}
