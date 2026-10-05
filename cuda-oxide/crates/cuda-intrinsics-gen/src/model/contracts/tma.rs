/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::super::core::RuntimeValidation;
use serde::{Deserialize, Serialize};

/// Closed semantic contract for a TMA operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tma {
    pub operation: TmaOperation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reduction: Option<TmaReduction>,
    pub adapter: TmaAdapter,
    pub runtime_validation: RuntimeValidation,
}

impl Tma {
    pub const fn dimensions(&self) -> Option<usize> {
        match &self.reduction {
            Some(reduction) => Some(reduction.dimensions as usize),
            None => self.operation.dimensions(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TmaOperation {
    G2sTile1d,
    G2sTile2d,
    G2sTile2dMulticast,
    G2sTile2dMulticastCg2,
    G2sTile3d,
    G2sTile4d,
    G2sTile5d,
    S2gTile1d,
    S2gTile2d,
    S2gTile3d,
    S2gTile4d,
    S2gTile5d,
    Reduce,
    CommitGroup,
    WaitGroup,
    WaitGroupRead,
    PrefetchTensorMap,
    PrefetchTile1d,
    PrefetchTile2d,
    PrefetchTile3d,
    PrefetchTile4d,
    PrefetchTile5d,
    #[serde(rename = "prefetch_tile_gather4_2d")]
    PrefetchTileGather4TwoDimensional,
    PrefetchTile1dCacheHint,
    PrefetchTile2dCacheHint,
    PrefetchTile3dCacheHint,
    PrefetchTile4dCacheHint,
    PrefetchTile5dCacheHint,
    #[serde(rename = "prefetch_tile_gather4_2d_cache_hint")]
    PrefetchTileGather4TwoDimensionalCacheHint,
    ReplaceBoxDim,
    ReplaceElementStride,
    ReplaceElementType,
    ReplaceFillMode,
    ReplaceGlobalAddress,
    ReplaceGlobalDim,
    ReplaceGlobalStride,
    ReplaceInterleaveLayout,
    ReplaceRank,
    ReplaceSwizzleAtomicity,
    ReplaceSwizzleMode,
    FenceProxyTensorMapAcquireCluster,
    FenceProxyTensorMapAcquireCta,
    FenceProxyTensorMapAcquireGpu,
    FenceProxyTensorMapAcquireSystem,
    FenceProxyTensorMapReleaseCluster,
    FenceProxyTensorMapReleaseCta,
    FenceProxyTensorMapReleaseGpu,
    FenceProxyTensorMapReleaseSystem,
}

impl TmaOperation {
    pub const fn dimensions(self) -> Option<usize> {
        match self {
            Self::G2sTile1d | Self::S2gTile1d => Some(1),
            Self::G2sTile2d
            | Self::G2sTile2dMulticast
            | Self::G2sTile2dMulticastCg2
            | Self::S2gTile2d => Some(2),
            Self::G2sTile3d | Self::S2gTile3d => Some(3),
            Self::G2sTile4d | Self::S2gTile4d => Some(4),
            Self::G2sTile5d | Self::S2gTile5d => Some(5),
            Self::Reduce
            | Self::CommitGroup
            | Self::WaitGroup
            | Self::WaitGroupRead
            | Self::PrefetchTensorMap
            | Self::PrefetchTile1d
            | Self::PrefetchTile2d
            | Self::PrefetchTile3d
            | Self::PrefetchTile4d
            | Self::PrefetchTile5d
            | Self::PrefetchTileGather4TwoDimensional
            | Self::PrefetchTile1dCacheHint
            | Self::PrefetchTile2dCacheHint
            | Self::PrefetchTile3dCacheHint
            | Self::PrefetchTile4dCacheHint
            | Self::PrefetchTile5dCacheHint
            | Self::PrefetchTileGather4TwoDimensionalCacheHint
            | Self::ReplaceBoxDim
            | Self::ReplaceElementStride
            | Self::ReplaceElementType
            | Self::ReplaceFillMode
            | Self::ReplaceGlobalAddress
            | Self::ReplaceGlobalDim
            | Self::ReplaceGlobalStride
            | Self::ReplaceInterleaveLayout
            | Self::ReplaceRank
            | Self::ReplaceSwizzleAtomicity
            | Self::ReplaceSwizzleMode
            | Self::FenceProxyTensorMapAcquireCluster
            | Self::FenceProxyTensorMapAcquireCta
            | Self::FenceProxyTensorMapAcquireGpu
            | Self::FenceProxyTensorMapAcquireSystem
            | Self::FenceProxyTensorMapReleaseCluster
            | Self::FenceProxyTensorMapReleaseCta
            | Self::FenceProxyTensorMapReleaseGpu
            | Self::FenceProxyTensorMapReleaseSystem => None,
        }
    }

    pub const fn prefetch_coordinate_count(self) -> Option<usize> {
        match self {
            Self::PrefetchTile1d | Self::PrefetchTile1dCacheHint => Some(1),
            Self::PrefetchTile2d | Self::PrefetchTile2dCacheHint => Some(2),
            Self::PrefetchTile3d | Self::PrefetchTile3dCacheHint => Some(3),
            Self::PrefetchTile4d | Self::PrefetchTile4dCacheHint => Some(4),
            Self::PrefetchTile5d
            | Self::PrefetchTile5dCacheHint
            | Self::PrefetchTileGather4TwoDimensional
            | Self::PrefetchTileGather4TwoDimensionalCacheHint => Some(5),
            _ => None,
        }
    }

    pub const fn uses_prefetch_cache_hint(self) -> bool {
        matches!(
            self,
            Self::PrefetchTile1dCacheHint
                | Self::PrefetchTile2dCacheHint
                | Self::PrefetchTile3dCacheHint
                | Self::PrefetchTile4dCacheHint
                | Self::PrefetchTile5dCacheHint
                | Self::PrefetchTileGather4TwoDimensionalCacheHint
        )
    }
}

/// Closed identity for one TMA tensor-reduction operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TmaReduction {
    pub operation: TmaReductionOperation,
    pub load_mode: TmaReductionLoadMode,
    pub dimensions: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TmaReductionOperation {
    Add,
    And,
    Dec,
    Inc,
    Max,
    Min,
    Or,
    Xor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TmaReductionLoadMode {
    Tile,
    Im2col,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TmaAdapter {
    G2sPointersCoordinatesBarrierInjectDefaults,
    G2sPointersCoordinatesBarrierMaskInjectDefaults,
    S2gPointersCoordinatesInjectDefaults,
    ReductionPointersCoordinatesInjectDefaults,
    NoOperands,
    CompileTimeConstantMaxPending,
    DescriptorPointer,
    DescriptorCoordinatesInjectDefaults,
    DescriptorCoordinatesCacheHintInjectFlag,
    DescriptorAndAddressPointers,
    DescriptorOrdinalAndU32,
    DescriptorOrdinalAndU64,
    DescriptorAndImmediateU32,
    DescriptorAndRuntimeU32,
    DescriptorPointerInjectBytes,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tma_contract_rejects_open_ended_policy() {
        let valid = r#"
operation = "g2s_tile2d_multicast"
adapter = "g2s_pointers_coordinates_barrier_mask_inject_defaults"
runtime_validation = "unexecuted"
"#;
        let parsed = toml::from_str::<Tma>(valid).unwrap();
        assert_eq!(parsed.operation, TmaOperation::G2sTile2dMulticast);
        assert_eq!(
            parsed.adapter,
            TmaAdapter::G2sPointersCoordinatesBarrierMaskInjectDefaults
        );

        for invalid in [
            valid.replace("g2s_tile2d_multicast", "g2s_multicast"),
            valid.replace(
                "g2s_pointers_coordinates_barrier_mask_inject_defaults",
                "direct",
            ),
            valid.replace(
                "runtime_validation = \"unexecuted\"",
                "runtime_validation = \"assumed\"",
            ),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(toml::from_str::<Tma>(&invalid).is_err(), "{invalid}");
        }
    }
}
