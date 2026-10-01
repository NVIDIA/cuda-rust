/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::super::catalog::CatalogTargetRequirement;
use super::super::core::RuntimeValidation;
use serde::{Deserialize, Serialize};

/// Closed semantic contract for one tcgen05 operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tcgen05 {
    pub operation: Tcgen05Operation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cp: Option<Tcgen05Cp>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ld: Option<Tcgen05Ld>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub st: Option<Tcgen05St>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mma: Option<Tcgen05Mma>,
    pub adapter: Tcgen05Adapter,
    pub source_contract: Tcgen05SourceContract,
    pub runtime_validation: RuntimeValidation,
}

/// Closed identity and selector contract for one tcgen05 MMA API.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tcgen05Mma {
    pub form: Tcgen05MmaForm,
    pub selector_layout: Tcgen05MmaSelectorLayout,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fixed_selectors: Option<Tcgen05MmaFixedSelectors>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alias: Option<Tcgen05MmaAlias>,
    pub llvm_target: CatalogTargetRequirement,
    pub libnvvm_target: CatalogTargetRequirement,
}

/// The 14 LLVM source forms covered by the first tcgen05 MMA batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tcgen05MmaForm {
    Shared,
    Tensor,
    TensorAshift,
    SpShared,
    SpTensor,
    SpTensorAshift,
    WsShared,
    WsSharedZeroColMask,
    WsSpShared,
    WsSpSharedZeroColMask,
    WsSpTensor,
    WsSpTensorZeroColMask,
    WsTensor,
    WsTensorZeroColMask,
}

/// Immediate arguments that select one imported tcgen05 MMA spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Tcgen05MmaSelectorLayout {
    Base {
        kind_argument: u8,
        cta_group_argument: u8,
        collector_a_argument: u8,
        collector_a_upper_exclusive: u8,
    },
    WarpSpecialized {
        kind_argument: u8,
        b_buffer_argument: u8,
        b_usage_argument: u8,
    },
}

/// A fixed warp-specialized selector tuple used by compatibility aliases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tcgen05MmaFixedSelectors {
    pub kind: Tcgen05MmaKind,
    pub b_buffer: u8,
    pub b_usage: Tcgen05MmaBUsage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tcgen05MmaKind {
    F16,
    Tf32,
    F8f6f4,
    I8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tcgen05MmaBUsage {
    Discard,
    LastUse,
    Fill,
    Use,
}

impl Tcgen05MmaBUsage {
    pub const fn selector_value(self) -> u8 {
        match self {
            Self::Discard => 0,
            Self::LastUse => 1,
            Self::Fill => 2,
            Self::Use => 3,
        }
    }
}

/// Public names proposed by PR #346 for the generic f8f6f4 carrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tcgen05MmaAlias {
    E4m3,
    E5m2,
    E2m3,
    E3m2,
    E2m1,
}

/// Closed identity for one tcgen05 shared-to-tensor-memory copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tcgen05Cp {
    pub member: Tcgen05CpMember,
    pub group: Tcgen05CpGroup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Tcgen05CpMember {
    #[serde(rename = "128x128b_b4x16_p64")]
    M128x128bB4x16P64,
    #[serde(rename = "128x128b_b6x16_p32")]
    M128x128bB6x16P32,
    #[serde(rename = "128x128b")]
    M128x128b,
    #[serde(rename = "128x256b_b4x16_p64")]
    M128x256bB4x16P64,
    #[serde(rename = "128x256b_b6x16_p32")]
    M128x256bB6x16P32,
    #[serde(rename = "32x128b_warpx4_b4x16_p64")]
    M32x128bWarpx4B4x16P64,
    #[serde(rename = "32x128b_warpx4_b6x16_p32")]
    M32x128bWarpx4B6x16P32,
    #[serde(rename = "32x128b_warpx4")]
    M32x128bWarpx4,
    #[serde(rename = "4x256b_b4x16_p64")]
    M4x256bB4x16P64,
    #[serde(rename = "4x256b_b6x16_p32")]
    M4x256bB6x16P32,
    #[serde(rename = "4x256b")]
    M4x256b,
    #[serde(rename = "64x128b_warpx2_01_23_b4x16_p64")]
    M64x128bWarpx2Pair0123B4x16P64,
    #[serde(rename = "64x128b_warpx2_01_23_b6x16_p32")]
    M64x128bWarpx2Pair0123B6x16P32,
    #[serde(rename = "64x128b_warpx2_01_23")]
    M64x128bWarpx2Pair0123,
    #[serde(rename = "64x128b_warpx2_02_13_b4x16_p64")]
    M64x128bWarpx2Pair0213B4x16P64,
    #[serde(rename = "64x128b_warpx2_02_13_b6x16_p32")]
    M64x128bWarpx2Pair0213B6x16P32,
    #[serde(rename = "64x128b_warpx2_02_13")]
    M64x128bWarpx2Pair0213,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tcgen05CpGroup {
    Cg1,
    Cg2,
}

/// Closed identity for one tcgen05 tensor-memory load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tcgen05Ld {
    pub shape: Tcgen05LdShape,
    pub multiplicity: Tcgen05LdMultiplicity,
    pub pack16: bool,
}

/// Closed identity for one tcgen05 tensor-memory store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tcgen05St {
    pub shape: Tcgen05LdShape,
    pub multiplicity: Tcgen05LdMultiplicity,
    pub unpack16: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Tcgen05LdShape {
    #[serde(rename = "16x32bx2")]
    M16x32bx2,
    #[serde(rename = "16x64b")]
    M16x64b,
    #[serde(rename = "16x128b")]
    M16x128b,
    #[serde(rename = "16x256b")]
    M16x256b,
    #[serde(rename = "32x32b")]
    M32x32b,
}

impl Tcgen05LdShape {
    pub const fn register_multiplier(self) -> usize {
        match self {
            Self::M16x32bx2 | Self::M16x64b | Self::M32x32b => 1,
            Self::M16x128b => 2,
            Self::M16x256b => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tcgen05LdMultiplicity {
    X1,
    X2,
    X4,
    X8,
    X16,
    X32,
    X64,
    X128,
}

impl Tcgen05LdMultiplicity {
    pub const fn count(self) -> usize {
        match self {
            Self::X1 => 1,
            Self::X2 => 2,
            Self::X4 => 4,
            Self::X8 => 8,
            Self::X16 => 16,
            Self::X32 => 32,
            Self::X64 => 64,
            Self::X128 => 128,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tcgen05Operation {
    Alloc,
    Dealloc,
    RelinquishAllocPermit,
    FenceBeforeThreadSync,
    FenceAfterThreadSync,
    Commit,
    CommitSharedCluster,
    MmaWsF16,
    MmaF16,
    MmaWsBf16,
    MmaWsTf32,
    CpSmemToTmem,
    Ld16x256bX8Pure,
    Ld16x256bPure,
    LoadWait,
    StoreWait,
    AllocCg2,
    DeallocCg2,
    RelinquishAllocPermitCg2,
    MmaF16Cg2,
    CommitCg2,
    CommitSharedClusterCg2,
    CommitMulticastCg2,
    CpSmemToTmemCg2,
    Ld,
    St,
    CommitMulticast,
    ShiftDown,
    ShiftDownCg2,
    Mma,
}

impl Tcgen05Operation {
    pub const fn execution_scope(self) -> &'static str {
        match self {
            Self::Alloc
            | Self::Dealloc
            | Self::RelinquishAllocPermit
            | Self::Ld16x256bX8Pure
            | Self::Ld16x256bPure
            | Self::LoadWait
            | Self::StoreWait
            | Self::AllocCg2
            | Self::DeallocCg2
            | Self::RelinquishAllocPermitCg2
            | Self::Ld
            | Self::St => "warp",
            Self::FenceBeforeThreadSync
            | Self::FenceAfterThreadSync
            | Self::Commit
            | Self::CommitSharedCluster
            | Self::MmaWsF16
            | Self::MmaF16
            | Self::MmaWsBf16
            | Self::MmaWsTf32
            | Self::CpSmemToTmem
            | Self::MmaF16Cg2
            | Self::CommitCg2
            | Self::CommitSharedClusterCg2
            | Self::CommitMulticastCg2
            | Self::CpSmemToTmemCg2
            | Self::CommitMulticast
            | Self::ShiftDown
            | Self::ShiftDownCg2
            | Self::Mma => "thread",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tcgen05Adapter {
    SharedPointerColumnsToVoid,
    TmemAddressColumnsToVoid,
    NoOperands,
    BarrierPointerToVoid,
    MmaWsDropLegacyADescriptor,
    MmaInjectZeroDisableLanes,
    TmemDescriptorToVoid,
    TmemToF32x32,
    TmemToF32x4,
    BarrierPointerMaskToVoid,
    TmemInjectPack16ToU32Registers,
    TmemU32RegistersInjectUnpack16ToVoid,
    TmemHalfSplitOffsetInjectPack16ToU32Registers,
    TmemHalfSplitOffsetU32RegistersInjectUnpack16ToVoid,
    TmemAddressToVoid,
    MmaDirectSelectors,
    MmaWsFixedSelectorsDropLegacyADescriptor,
}

/// Relationship between the public operation and LLVM's NVPTX selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tcgen05SourceContract {
    ExactTablegenSelection,
    TablegenSelectionChangesPtx,
    LlvmCustomLoweringWithoutSelection,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tcgen05_execution_scope_is_closed() {
        for operation in [
            Tcgen05Operation::Alloc,
            Tcgen05Operation::Dealloc,
            Tcgen05Operation::RelinquishAllocPermit,
            Tcgen05Operation::AllocCg2,
            Tcgen05Operation::DeallocCg2,
            Tcgen05Operation::RelinquishAllocPermitCg2,
            Tcgen05Operation::Ld16x256bX8Pure,
            Tcgen05Operation::Ld16x256bPure,
            Tcgen05Operation::LoadWait,
            Tcgen05Operation::StoreWait,
            Tcgen05Operation::Ld,
            Tcgen05Operation::St,
        ] {
            assert_eq!(operation.execution_scope(), "warp");
        }
        for operation in [
            Tcgen05Operation::FenceBeforeThreadSync,
            Tcgen05Operation::FenceAfterThreadSync,
            Tcgen05Operation::Commit,
            Tcgen05Operation::CommitSharedCluster,
            Tcgen05Operation::MmaWsF16,
            Tcgen05Operation::MmaWsBf16,
            Tcgen05Operation::MmaWsTf32,
            Tcgen05Operation::MmaF16,
            Tcgen05Operation::CpSmemToTmem,
            Tcgen05Operation::MmaF16Cg2,
            Tcgen05Operation::CommitCg2,
            Tcgen05Operation::CommitSharedClusterCg2,
            Tcgen05Operation::CommitMulticastCg2,
            Tcgen05Operation::CpSmemToTmemCg2,
            Tcgen05Operation::CommitMulticast,
            Tcgen05Operation::ShiftDown,
            Tcgen05Operation::ShiftDownCg2,
        ] {
            assert_eq!(operation.execution_scope(), "thread");
        }
    }
}
