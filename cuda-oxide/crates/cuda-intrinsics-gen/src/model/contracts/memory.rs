/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::super::core::RuntimeValidation;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MbarrierExtendedOperation {
    ArriveExpectTxCta,
    ArriveExpectTxCluster,
    ArriveRemoteCluster,
    TryWaitTokenCta,
    TryWaitParityCta,
    TryWaitParityCluster,
    FenceProxyAsyncSharedCta,
    FenceMbarrierInitReleaseCluster,
    FenceProxyAsyncGenericReleaseSharedCtaCluster,
    FenceProxyAsyncGenericAcquireSharedClusterCluster,
    Nanosleep,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClusterMemoryOperation {
    MapSharedRank,
    ReadU32,
}

/// Closed contract for cluster address mapping and remote shared reads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterMemory {
    pub operation: ClusterMemoryOperation,
    pub adapter: ClusterMemoryAdapter,
    pub source_contract: ClusterMemorySourceContract,
    pub runtime_validation: RuntimeValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClusterMemoryAdapter {
    GenericConstAndMutPointerRankToSamePointer,
    ConstU32PointerRankToU32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClusterMemorySourceContract {
    LlvmMapaSharedClusterAs7IdentityInlinePtx,
    PtxNativeMapaThenWeakClusterLoad,
}

/// Closed semantic contract for the generated packed global atomic-add
/// family. These fields are intentionally enums rather than free-form strings:
/// accepting an unreviewed state space, scope, or floating-point mode must
/// require a generator change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackedAtomic {
    pub format: PackedAtomicFormat,
    /// PTX ISA hardware floor, kept separate from cuda-oxide's admitted floor
    /// and from backend-profile floors.
    pub native_minimum_sm: u16,
    pub operation: PackedAtomicOperation,
    pub state_space: PackedAtomicStateSpace,
    pub ordering: PackedAtomicOrdering,
    pub scope: PackedAtomicScope,
    pub rounding: PackedAtomicRounding,
    pub subnormal: PackedAtomicSubnormal,
    pub atomicity: PackedAtomicAtomicity,
    pub pointer_contract: PackedAtomicPointerContract,
    pub access_contract: PackedAtomicAccessContract,
    pub scope_contract: PackedAtomicScopeContract,
    pub codegen_contract: PackedAtomicCodegenContract,
    pub return_contract: PackedAtomicReturnContract,
    pub adapter: PackedAtomicAdapter,
    pub runtime_validation: RuntimeValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAtomicFormat {
    F16x2,
    Bf16x2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAtomicOperation {
    Add,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAtomicStateSpace {
    Global,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAtomicOrdering {
    Relaxed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAtomicScope {
    Gpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAtomicRounding {
    NearestEven,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAtomicSubnormal {
    Preserve,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAtomicAtomicity {
    PerElement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAtomicPointerContract {
    MutableGlobalU32Aligned4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAtomicAccessContract {
    NoMixedWholeWordOrNonAtomicAccess,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAtomicScopeContract {
    RacingAtomicsMutuallyInclusive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAtomicCodegenContract {
    ExactNativeInstruction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAtomicReturnContract {
    OldValuesPerElementMayBeNoncoherent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAtomicAdapter {
    OldPackedU32,
}

/// Closed contract for classic global-to-shared `cp.async` copies.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CpAsyncCopy {
    pub cache_policy: CpAsyncCachePolicy,
    pub copy_size: CpAsyncCopySize,
    pub source_size: CpAsyncSourceSize,
    pub adapter: CpAsyncAdapter,
    pub runtime_validation: RuntimeValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpAsyncCachePolicy {
    Ca,
    Cg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpAsyncCopySize {
    B4,
    B8,
    B16,
}

impl CpAsyncCopySize {
    pub const fn bytes(self) -> u32 {
        match self {
            Self::B4 => 4,
            Self::B8 => 8,
            Self::B16 => 16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpAsyncSourceSize {
    Full,
    Runtime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpAsyncAdapter {
    DirectPointers,
    DirectPointersAndSourceSize,
}

/// Closed contract for classic `cp.async` group controls.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CpAsyncControl {
    pub operation: CpAsyncControlOperation,
    pub adapter: CpAsyncControlAdapter,
    pub runtime_validation: RuntimeValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpAsyncControlOperation {
    CommitGroup,
    WaitAll,
    WaitGroup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpAsyncControlAdapter {
    NoOperands,
    CompileTimeConstantMaxPending,
}

/// Closed contract for associating classic `cp.async` completion with an mbarrier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CpAsyncMbarrier {
    pub operation: CpAsyncMbarrierOperation,
    pub state_space: CpAsyncMbarrierStateSpace,
    pub adapter: CpAsyncMbarrierAdapter,
    pub runtime_validation: RuntimeValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpAsyncMbarrierOperation {
    Arrive,
    ArriveNoInc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpAsyncMbarrierStateSpace {
    Generic,
    Shared,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpAsyncMbarrierAdapter {
    PointerToVoid,
}

/// Closed contract for the basic shared-memory mbarrier lifecycle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MbarrierBasic {
    pub operation: MbarrierBasicOperation,
    pub state_space: MbarrierStateSpace,
    pub adapter: MbarrierBasicAdapter,
    pub runtime_validation: RuntimeValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MbarrierBasicOperation {
    Init,
    Arrive,
    ArriveNoComplete,
    TestWait,
    Inval,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MbarrierStateSpace {
    Shared,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MbarrierBasicAdapter {
    #[serde(rename = "pointer_count_to_void")]
    InitPointerCountToVoid,
    #[serde(rename = "pointer_to_token")]
    ArrivePointerToToken,
    #[serde(rename = "pointer_count_to_token")]
    ArriveNoCompletePointerCountToToken,
    #[serde(rename = "pointer_token_to_predicate")]
    TestWaitPointerTokenToPredicate,
    #[serde(rename = "pointer_to_void")]
    InvalPointerToVoid,
}

/// Closed contract for the remaining handwritten mbarrier operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MbarrierExtended {
    pub operation: MbarrierExtendedOperation,
    pub adapter: MbarrierExtendedAdapter,
    pub source_contract: MbarrierExtendedSourceContract,
    pub runtime_validation: RuntimeValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MbarrierExtendedAdapter {
    PointerTxCountBytesToTokenDroppingTxCount,
    RawClusterAddressToVoid,
    PointerTokenToPredicate,
    PointerParityToPredicate,
    ZeroOperandsToVoid,
    NanosecondsToVoid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MbarrierExtendedSourceContract {
    LlvmImported,
    PtxNativeRawClusterAddress,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mbarrier_basic_contract_rejects_open_ended_policy() {
        let valid = r#"
operation = "test_wait"
state_space = "shared"
adapter = "pointer_token_to_predicate"
runtime_validation = "unexecuted"
"#;
        let parsed = toml::from_str::<MbarrierBasic>(valid).unwrap();
        assert_eq!(parsed.operation, MbarrierBasicOperation::TestWait);
        assert_eq!(
            parsed.adapter,
            MbarrierBasicAdapter::TestWaitPointerTokenToPredicate
        );

        for invalid in [
            valid.replace("operation = \"test_wait\"", "operation = \"wait\""),
            valid.replace("state_space = \"shared\"", "state_space = \"global\""),
            valid.replace(
                "adapter = \"pointer_token_to_predicate\"",
                "adapter = \"direct\"",
            ),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(
                toml::from_str::<MbarrierBasic>(&invalid).is_err(),
                "{invalid}"
            );
        }
    }

    #[test]
    fn mbarrier_extended_contract_rejects_open_ended_policy() {
        let valid = r#"
operation = "arrive_expect_tx_cta"
adapter = "pointer_tx_count_bytes_to_token_dropping_tx_count"
source_contract = "llvm_imported"
runtime_validation = "unexecuted"
"#;
        let parsed = toml::from_str::<MbarrierExtended>(valid).unwrap();
        assert_eq!(
            parsed.adapter,
            MbarrierExtendedAdapter::PointerTxCountBytesToTokenDroppingTxCount
        );
        assert_eq!(
            parsed.source_contract,
            MbarrierExtendedSourceContract::LlvmImported
        );

        for invalid in [
            valid.replace("llvm_imported", "auto"),
            valid.replace(
                "pointer_tx_count_bytes_to_token_dropping_tx_count",
                "direct",
            ),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(
                toml::from_str::<MbarrierExtended>(&invalid).is_err(),
                "{invalid}"
            );
        }
    }

    #[test]
    fn cp_async_mbarrier_contract_rejects_open_ended_policy() {
        let valid = r#"
operation = "arrive_no_inc"
state_space = "shared"
adapter = "pointer_to_void"
runtime_validation = "unexecuted"
"#;
        let parsed = toml::from_str::<CpAsyncMbarrier>(valid).unwrap();
        assert_eq!(parsed.operation, CpAsyncMbarrierOperation::ArriveNoInc);
        assert_eq!(parsed.state_space, CpAsyncMbarrierStateSpace::Shared);

        for invalid in [
            valid.replace("operation = \"arrive_no_inc\"", "operation = \"wait\""),
            valid.replace("state_space = \"shared\"", "state_space = \"global\""),
            valid.replace("adapter = \"pointer_to_void\"", "adapter = \"direct\""),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(
                toml::from_str::<CpAsyncMbarrier>(&invalid).is_err(),
                "{invalid}"
            );
        }
    }

    #[test]
    fn cluster_memory_contract_rejects_open_ended_policy() {
        let valid = r#"
operation = "map_shared_rank"
adapter = "generic_const_and_mut_pointer_rank_to_same_pointer"
source_contract = "llvm_mapa_shared_cluster_as7_identity_inline_ptx"
runtime_validation = "unexecuted"
"#;
        let parsed = toml::from_str::<ClusterMemory>(valid).unwrap();
        assert_eq!(parsed.operation, ClusterMemoryOperation::MapSharedRank);
        assert_eq!(
            parsed.source_contract,
            ClusterMemorySourceContract::LlvmMapaSharedClusterAs7IdentityInlinePtx
        );

        for invalid in [
            valid.replace("map_shared_rank", "map_generic_rank"),
            valid.replace(
                "generic_const_and_mut_pointer_rank_to_same_pointer",
                "direct",
            ),
            valid.replace(
                "llvm_mapa_shared_cluster_as7_identity_inline_ptx",
                "llvm_typed_as3",
            ),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(
                toml::from_str::<ClusterMemory>(&invalid).is_err(),
                "{invalid}"
            );
        }
    }
}
