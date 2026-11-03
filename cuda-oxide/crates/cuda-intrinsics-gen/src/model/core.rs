/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use serde::{Deserialize, Serialize};

/// Provenance for a generated intrinsic. PTX-native operations deliberately
/// have no invented LLVM TableGen record or LLVM intrinsic symbol.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum IntrinsicSource {
    LlvmImported { source_record: String },
    PtxNative { instruction: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntrinsicBackend {
    LlvmNvptx,
    LibNvvm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendLoweringMechanism {
    TypedNvvm,
    InlinePtx,
}

/// Closed identity for the small execution-control families that share a
/// result-less MIR/dialect representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExecutionControlOperation {
    BarrierCtaSync,
    BarrierCtaSyncAligned,
    BarrierCtaArrive,
    BarrierCtaArriveAligned,
    GridDependencyLaunchDependents,
    GridDependencyWait,
    SetMaxNRegInc,
    SetMaxNRegDec,
}

impl ExecutionControlOperation {
    pub fn from_catalog_id(id: &str) -> Option<Self> {
        Some(match id {
            "barrier_cta_sync" => Self::BarrierCtaSync,
            "barrier_cta_sync_aligned" => Self::BarrierCtaSyncAligned,
            "barrier_cta_arrive" => Self::BarrierCtaArrive,
            "barrier_cta_arrive_aligned" => Self::BarrierCtaArriveAligned,
            "grid_dependency_launch_dependents" => Self::GridDependencyLaunchDependents,
            "grid_dependency_wait" => Self::GridDependencyWait,
            "setmaxnreg_inc" => Self::SetMaxNRegInc,
            "setmaxnreg_dec" => Self::SetMaxNRegDec,
            _ => return None,
        })
    }

    pub const fn family(self) -> &'static str {
        match self {
            Self::BarrierCtaSync
            | Self::BarrierCtaSyncAligned
            | Self::BarrierCtaArrive
            | Self::BarrierCtaArriveAligned => "counted_barrier",
            Self::GridDependencyLaunchDependents | Self::GridDependencyWait => "grid_dependency",
            Self::SetMaxNRegInc | Self::SetMaxNRegDec => "register_control",
        }
    }

    pub const fn operand_count(self) -> usize {
        match self {
            Self::BarrierCtaSync
            | Self::BarrierCtaSyncAligned
            | Self::BarrierCtaArrive
            | Self::BarrierCtaArriveAligned => 2,
            Self::GridDependencyLaunchDependents | Self::GridDependencyWait => 0,
            Self::SetMaxNRegInc | Self::SetMaxNRegDec => 1,
        }
    }

    pub const fn requires_immediate_operands(self) -> bool {
        matches!(self, Self::SetMaxNRegInc | Self::SetMaxNRegDec)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeValidation {
    Unexecuted,
    Executed,
}
