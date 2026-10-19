/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_target_spec::PtxSpelling;

/// GPU feature requirements detected in one LLVM module.
///
/// This is a set rather than a single "strongest" feature: architecture
/// families are not totally ordered. For example, WGMMA requires Hopper
/// `sm_90a`, while PTX 8.6 matrix forms require Blackwell. Keeping every bit
/// lets target validation enforce the intersection instead of silently
/// choosing whichever instruction happened to have higher detector priority.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct DetectedFeatures(u32);

#[allow(non_upper_case_globals)]
impl DetectedFeatures {
    /// tcgen05/TMEM (Blackwell datacenter, sm_100a).
    pub(crate) const Blackwell: Self = Self(1 << 0);
    /// Base TMA multicast (sm_90+, with architecture/family targets preferred).
    pub(crate) const TmaMulticast: Self = Self(1 << 1);
    /// Explicit CTA-group TMA forms (Blackwell datacenter family).
    pub(crate) const TmaCtaGroup: Self = Self(1 << 2);
    /// PTX 8.6 ldmatrix/stmatrix shapes supported on Blackwell family targets.
    pub(crate) const MatrixBlackwell: Self = Self(1 << 3);
    /// WGMMA (Hopper only, sm_90a - NOT forward-compatible).
    pub(crate) const Wgmma: Self = Self(1 << 4);
    /// TMA/mbarrier (Hopper+ compatible).
    pub(crate) const Tma: Self = Self(1 << 5);
    /// Thread Block Clusters (sm_90+, forward-compatible).
    pub(crate) const Cluster: Self = Self(1 << 6);
    /// Forward-compatible instructions with an sm_90 floor.
    pub(crate) const Sm90: Self = Self(1 << 7);
    /// Forward-compatible instructions with an sm_80 floor.
    pub(crate) const Sm80: Self = Self(1 << 8);
    /// Forward-compatible instructions with an sm_75 floor.
    pub(crate) const Sm75: Self = Self(1 << 17);
    /// Warp matrix register transpose introduced in PTX 7.8 on sm_75.
    pub(crate) const Movmatrix: Self = Self(1 << 9);
    /// Warp matrix shared-memory load introduced in PTX 6.5 on sm_75.
    pub(crate) const Ldmatrix: Self = Self(1 << 10);
    /// No special features (Volta+, with an sm_80 cross-compile default).
    pub(crate) const Basic: Self = Self(1 << 11);
    /// Generic Blackwell-or-newer operations such as base CLC and TMA cp_mask.
    pub(crate) const Sm100: Self = Self(1 << 12);
    /// Architecture/family-specific Blackwell features also available on consumers.
    pub(crate) const BlackwellFamily: Self = Self(1 << 13);
    /// Architecture/family-specific datacenter Blackwell TMA modes.
    pub(crate) const BlackwellAccelerated: Self = Self(1 << 14);
    /// Floating-point `redux.sync` (the sm_100/sm_103 architecture family).
    pub(crate) const ReduxF32: Self = Self(1 << 15);
    /// FP8 / f16-accumulator multimem forms on supported Blackwell families.
    pub(crate) const MultimemFp8: Self = Self(1 << 16);
    /// Dynamic stack save/restore, supported by LLVM NVPTX on sm_52+.
    pub(crate) const DynamicStack: Self = Self(1 << 18);

    const ALL: [Self; 19] = [
        Self::Blackwell,
        Self::TmaCtaGroup,
        Self::BlackwellAccelerated,
        Self::BlackwellFamily,
        Self::ReduxF32,
        Self::MultimemFp8,
        Self::TmaMulticast,
        Self::MatrixBlackwell,
        Self::Wgmma,
        Self::Tma,
        Self::Cluster,
        Self::Sm90,
        Self::Sm80,
        Self::Sm75,
        Self::Movmatrix,
        Self::Ldmatrix,
        Self::Sm100,
        Self::DynamicStack,
        Self::Basic,
    ];

    pub(super) const fn empty() -> Self {
        Self(0)
    }

    pub(super) const fn contains(self, feature: Self) -> bool {
        self.0 & feature.0 != 0
    }

    pub(super) fn insert(&mut self, feature: Self) {
        self.0 |= feature.0;
    }

    pub(super) fn iter(self) -> impl Iterator<Item = Self> {
        Self::ALL
            .into_iter()
            .filter(move |feature| self.contains(*feature))
    }

    fn name(self) -> &'static str {
        match self {
            Self::Blackwell => "Blackwell",
            Self::TmaMulticast => "TmaMulticast",
            Self::TmaCtaGroup => "TmaCtaGroup",
            Self::MatrixBlackwell => "MatrixBlackwell",
            Self::Wgmma => "Wgmma",
            Self::Tma => "Tma",
            Self::Cluster => "Cluster",
            Self::Sm90 => "Sm90",
            Self::Sm80 => "Sm80",
            Self::Sm75 => "Sm75",
            Self::Movmatrix => "Movmatrix",
            Self::Ldmatrix => "Ldmatrix",
            Self::Sm100 => "Sm100",
            Self::BlackwellFamily => "BlackwellFamily",
            Self::BlackwellAccelerated => "BlackwellAccelerated",
            Self::ReduxF32 => "ReduxF32",
            Self::MultimemFp8 => "MultimemFp8",
            Self::DynamicStack => "DynamicStack",
            Self::Basic => "Basic",
            _ => "Unknown",
        }
    }
}

impl std::fmt::Debug for DetectedFeatures {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;
        for feature in self.iter() {
            if !first {
                formatter.write_str(" + ")?;
            }
            formatter.write_str(feature.name())?;
            first = false;
        }
        Ok(())
    }
}

impl std::ops::BitOr for DetectedFeatures {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

/// PTX ISA requirements are independent of the GPU architecture floor.
///
/// For example, a module may need sm_80 because it uses `cp.async` and still
/// need PTX 7.8 because it also uses `movmatrix`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PtxIsaRequirement(Option<PtxSpelling>);

impl PtxIsaRequirement {
    #[allow(non_upper_case_globals)]
    pub const Default: Self = Self(None);

    pub(crate) const fn new(spelling: u16) -> Self {
        match PtxSpelling::from_spelling(spelling) {
            Some(spelling) => Self(Some(spelling)),
            None => panic!("unsupported PTX ISA spelling"),
        }
    }

    pub(super) const fn from_spelling(spelling: PtxSpelling) -> Self {
        Self(Some(spelling))
    }

    pub(super) fn spelling(self) -> Option<PtxSpelling> {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModuleRequirements {
    pub features: DetectedFeatures,
    pub ptx_isa: PtxIsaRequirement,
}
