/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared tensor placement, register carriers, and tiled matrix-instruction contracts.

use cute_layout::{ComposedLayout, Layout, OffsetUnit};
use pliron::{common_traits::Verify, context::Context, result::Error, verify_err_noloc};
use pliron_derive::pliron_attr;

use super::{
    layout::{CuteComposedLayoutAttr, CuteLayoutAttr},
    tensor::CuteTensorFormatAttr,
};

/// Register type accumulated by one matrix-multiply atom.
///
/// The first shared-memory MMA slice only needs FP32 accumulation. Keeping it
/// named makes the atom description extensible without hiding the choice in
/// an integer code.
#[pliron_attr(name = "cute.mma_accumulator", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum CuteMmaAccumulatorAttr {
    /// FP32 scalar storage or accumulation.
    F32,
}

/// What one ordinary MIR aggregate means at a semantic MMA operation.
///
/// B is deliberately absent from this first list. A B partition is a lazy
/// pointer-and-coordinate view, not an eagerly loaded register fragment:
///
/// ```text
/// ScaleStage ── K half ──► ScaleK64
/// A shared tile ─────────► A
/// zero / previous step ──► Accumulator
/// B shared tile ─────────► pointer + capacity + coordinates
/// ```
#[pliron_attr(name = "cute.mma_carrier_kind", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum CuteMmaCarrierKindAttr {
    /// All scale registers needed by one staged K region.
    ScaleStage,
    /// Scale registers selected for one K=64 matrix-instruction step.
    ScaleK64,
    /// Packed registers for one loaded A fragment.
    A,
    /// FP32 output registers carried between compute steps.
    Accumulator,
}

/// One warp-level matrix-multiply instruction shape.
///
/// Register counts are per lane. They describe the physical MIR aggregates
/// consumed by the selected backend's matrix-instruction lowering:
///
/// ```text
/// A: 4 x u32    B: 2 x u32    C: 4 x f32
///              m16n8k64
/// ```
#[pliron_attr(
    name = "cute.mma_atom",
    format = "`<` $m `,` $n `,` $k `,` $a_format `,` $b_format `,` $scale_format `,` $values_per_scale `,` $accumulator `,` $threads `,` $a_registers `,` $b_registers `,` $accumulator_registers `>`"
)]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct CuteMmaAtomAttr {
    /// Collective output-row extent M.
    pub m: u64,
    /// Collective output-column extent N.
    pub n: u64,
    /// Reduction extent K, in logical elements.
    pub k: u64,
    /// Logical encoding of packed A values.
    pub a_format: CuteTensorFormatAttr,
    /// Logical encoding of packed B values.
    pub b_format: CuteTensorFormatAttr,
    /// Encoding used by block-scale values.
    pub scale_format: CuteTensorFormatAttr,
    /// Number of consecutive logical K values sharing one scale.
    pub values_per_scale: u64,
    /// Scalar format used by the accumulator registers.
    pub accumulator: CuteMmaAccumulatorAttr,
    /// Participating threads in one matrix-instruction atom.
    pub threads: u32,
    /// Packed u32 A registers per participating lane.
    pub a_registers: u32,
    /// Packed u32 B registers per participating lane.
    pub b_registers: u32,
    /// FP32 accumulator registers per lane.
    pub accumulator_registers: u32,
}

impl CuteMmaAtomAttr {
    /// SM120's packed MXFP4 `m16n8k64` atom (`UE8M0` scales, `SF_VEC = 32`).
    #[must_use]
    pub const fn mxf4_m16n8k64() -> Self {
        Self {
            m: 16,
            n: 8,
            k: 64,
            a_format: CuteTensorFormatAttr::E2M1,
            b_format: CuteTensorFormatAttr::E2M1,
            scale_format: CuteTensorFormatAttr::UE8M0,
            values_per_scale: 32,
            accumulator: CuteMmaAccumulatorAttr::F32,
            threads: 32,
            a_registers: 4,
            b_registers: 2,
            accumulator_registers: 4,
        }
    }

    /// SM120's packed NVFP4 `m16n8k64` atom (`E4M3` scales, `SF_VEC = 16`).
    #[must_use]
    pub const fn nvf4_m16n8k64() -> Self {
        Self {
            m: 16,
            n: 8,
            k: 64,
            a_format: CuteTensorFormatAttr::E2M1,
            b_format: CuteTensorFormatAttr::E2M1,
            scale_format: CuteTensorFormatAttr::E4M3,
            values_per_scale: 16,
            accumulator: CuteMmaAccumulatorAttr::F32,
            threads: 32,
            a_registers: 4,
            b_registers: 2,
            accumulator_registers: 4,
        }
    }
}

impl Verify for CuteMmaAtomAttr {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        if self.m == 0 || self.n == 0 || self.k == 0 {
            return verify_err_noloc!("cute.mma_atom M, N, and K must be greater than zero");
        }
        if self.a_format != CuteTensorFormatAttr::E2M1
            || self.b_format != CuteTensorFormatAttr::E2M1
            || !matches!(
                self.scale_format,
                CuteTensorFormatAttr::UE8M0 | CuteTensorFormatAttr::E4M3
            )
        {
            return verify_err_noloc!(
                "cute.mma_atom v0 needs E2M1 A/B values with UE8M0 or E4M3 scales"
            );
        }
        if self.values_per_scale == 0 || !self.k.is_multiple_of(self.values_per_scale) {
            return verify_err_noloc!(
                "cute.mma_atom K must contain a whole positive number of scale groups"
            );
        }
        if self.threads == 0
            || self.a_registers == 0
            || self.b_registers == 0
            || self.accumulator_registers == 0
        {
            return verify_err_noloc!(
                "cute.mma_atom thread and per-lane register counts must be greater than zero"
            );
        }
        Ok(())
    }
}

/// How warp-level MMA atoms cover one CTA output tile.
///
/// `m_ownership` and `n_ownership` map a warp position plus a cell position
/// to the corresponding M or N atom number. The current 128 x 128 plan reads:
///
/// ```text
/// M atoms: (warp_m, cell_m)       (4,2):(1,4)
/// N atoms: ((within_pair,pair),   ((2,4),2):((1,4),2)
///           warp_n)
///
/// 4 warp-M positions x 2 warp-N positions = 8 compute warps
/// one warp owns 2 M atoms x 8 N atoms = 16 accumulator cells
/// ```
///
/// `b_load_group` is the number of B fragments loaded before immediately
/// consuming them. It keeps the performance-critical lazy-B schedule visible.
#[pliron_attr(
    name = "cute.tiled_mma_plan",
    format = "`<` $atom `,` $cta_m `,` $cta_n `,` $cta_k `,` $warp_m `,` $warp_n `,` $b_load_group `,` $shared_layout `,` $m_ownership `,` $n_ownership `>`"
)]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct CuteTiledMmaPlanAttr {
    /// Warp-level instruction shape and register contract.
    pub atom: CuteMmaAtomAttr,
    /// Output rows covered by one CTA tile.
    pub cta_m: u64,
    /// Output columns covered by one CTA tile.
    pub cta_n: u64,
    /// Reduction elements staged for one CTA compute step.
    pub cta_k: u64,
    /// Number of warp positions along the output M axis.
    pub warp_m: u32,
    /// Number of warp positions along the output N axis.
    pub warp_n: u32,
    /// B fragments loaded together before their immediate MMA use.
    pub b_load_group: u32,
    /// Physical shared placement expected by the tiled-MMA data loads.
    pub shared_layout: CuteComposedLayoutAttr,
    /// Layout mapping warp/cell coordinates to M-atom ownership.
    pub m_ownership: CuteLayoutAttr,
    /// Layout mapping warp/cell coordinates to N-atom ownership.
    pub n_ownership: CuteLayoutAttr,
}

impl CuteTiledMmaPlanAttr {
    /// Describe the current 128 x 128 x 128 SM120 MXFP4 tiled MMA.
    #[must_use]
    pub fn mxf4_128x128x128(shared_layout: ComposedLayout) -> Self {
        Self {
            atom: CuteMmaAtomAttr::mxf4_m16n8k64(),
            cta_m: 128,
            cta_n: 128,
            cta_k: 128,
            warp_m: 4,
            warp_n: 2,
            b_load_group: 2,
            shared_layout: CuteComposedLayoutAttr(shared_layout),
            m_ownership: CuteLayoutAttr(
                "(4,2):(1,4)"
                    .parse()
                    .expect("fixed M ownership layout is valid"),
            ),
            n_ownership: CuteLayoutAttr(
                "((2,4),2):((1,4),2)"
                    .parse()
                    .expect("fixed N ownership layout is valid"),
            ),
        }
    }

    /// Describe the 128 x 128 x 128 SM120 NVFP4 (`E4M3` scale, `SF_VEC = 16`) tiled MMA.
    #[must_use]
    pub fn nvf4_128x128x128(shared_layout: ComposedLayout) -> Self {
        Self {
            atom: CuteMmaAtomAttr::nvf4_m16n8k64(),
            ..Self::mxf4_128x128x128(shared_layout)
        }
    }

    /// Number of compute warps in this plan.
    #[must_use]
    pub fn compute_warps(&self) -> Option<u64> {
        u64::from(self.warp_m).checked_mul(u64::from(self.warp_n))
    }

    /// Number of M atoms owned by one warp.
    #[must_use]
    pub fn m_atoms_per_warp(&self) -> Option<u64> {
        self.cta_m
            .checked_div(self.atom.m)?
            .checked_div(u64::from(self.warp_m))
    }

    /// Number of N atoms owned by one warp.
    #[must_use]
    pub fn n_atoms_per_warp(&self) -> Option<u64> {
        self.cta_n
            .checked_div(self.atom.n)?
            .checked_div(u64::from(self.warp_n))
    }

    /// Packed scale words held by one lane for one K=128 stage.
    #[must_use]
    pub fn scale_words_per_lane(&self) -> Option<u64> {
        self.m_atoms_per_warp()?
            .checked_add(self.n_atoms_per_warp()?)
    }

    /// A-fragment u32 registers held by one lane for one K=64 step.
    #[must_use]
    pub fn a_registers_per_lane(&self) -> Option<u64> {
        self.m_atoms_per_warp()?
            .checked_mul(u64::from(self.atom.a_registers))
    }

    /// FP32 accumulator registers held by one lane for the complete warp tile.
    #[must_use]
    pub fn accumulator_registers_per_lane(&self) -> Option<u64> {
        self.m_atoms_per_warp()?
            .checked_mul(self.n_atoms_per_warp()?)?
            .checked_mul(u64::from(self.atom.accumulator_registers))
    }
}

/// Check that all logical coordinates cover one dense physical interval exactly once.
fn layout_is_dense_permutation(layout: &Layout) -> bool {
    let Some(size) = layout.checked_size() else {
        return false;
    };
    let mut offsets = (0..size)
        .map(|coordinate| layout.checked_call(&cute_layout::IntTuple::Leaf(coordinate)))
        .collect::<Option<Vec<_>>>();
    let Some(ref mut offsets) = offsets else {
        return false;
    };
    offsets.sort_unstable();
    offsets.iter().copied().eq(0..size)
}

impl Verify for CuteTiledMmaPlanAttr {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        self.atom.verify(ctx)?;
        if self.cta_m == 0 || self.cta_n == 0 || self.cta_k == 0 {
            return verify_err_noloc!(
                "cute.tiled_mma_plan CTA M, N, and K must be greater than zero"
            );
        }
        if !self.cta_m.is_multiple_of(self.atom.m)
            || !self.cta_n.is_multiple_of(self.atom.n)
            || !self.cta_k.is_multiple_of(self.atom.k)
        {
            return verify_err_noloc!("cute.tiled_mma_plan CTA tile must contain whole MMA atoms");
        }
        if self.warp_m == 0 || self.warp_n == 0 {
            return verify_err_noloc!(
                "cute.tiled_mma_plan warp grid sizes must be greater than zero"
            );
        }
        let m_atoms = self.cta_m / self.atom.m;
        let n_atoms = self.cta_n / self.atom.n;
        if !m_atoms.is_multiple_of(u64::from(self.warp_m))
            || !n_atoms.is_multiple_of(u64::from(self.warp_n))
        {
            return verify_err_noloc!(
                "cute.tiled_mma_plan atom grid must divide evenly across its warp grid"
            );
        }
        let Some(compute_warps) = self.compute_warps() else {
            return verify_err_noloc!("cute.tiled_mma_plan compute-warp count overflowed");
        };
        if compute_warps == 0 || compute_warps > 32 {
            return verify_err_noloc!(
                "cute.tiled_mma_plan must use between 1 and 32 compute warps"
            );
        }
        let Some(n_per_warp) = self.n_atoms_per_warp() else {
            return verify_err_noloc!("cute.tiled_mma_plan N ownership is not divisible");
        };
        if self.b_load_group == 0 || n_per_warp % u64::from(self.b_load_group) != 0 {
            return verify_err_noloc!(
                "cute.tiled_mma_plan B load group must divide one warp's N atoms"
            );
        }
        if self.shared_layout.0.unit() != OffsetUnit::Elements || self.shared_layout.0.offset() != 0
        {
            return verify_err_noloc!(
                "cute.tiled_mma_plan shared placement must use zero-based element offsets"
            );
        }
        let expected_shared_elements = self
            .cta_m
            .checked_mul(self.cta_k)
            .and_then(|values| values.checked_div(4));
        if expected_shared_elements.and_then(|value| i64::try_from(value).ok())
            != self.shared_layout.0.outer().checked_size()
        {
            return verify_err_noloc!(
                "cute.tiled_mma_plan shared placement must hold four packed E2M1 values per f16 carrier"
            );
        }
        if self.m_ownership.0.checked_size() != i64::try_from(m_atoms).ok()
            || self.n_ownership.0.checked_size() != i64::try_from(n_atoms).ok()
            || !layout_is_dense_permutation(&self.m_ownership.0)
            || !layout_is_dense_permutation(&self.n_ownership.0)
        {
            return verify_err_noloc!(
                "cute.tiled_mma_plan ownership layouts must cover every M/N atom exactly once"
            );
        }
        if self.m_ownership.0.shape.flatten().first().copied() != Some(i64::from(self.warp_m))
            || self.n_ownership.0.shape.flatten().last().copied() != Some(i64::from(self.warp_n))
        {
            return verify_err_noloc!(
                "cute.tiled_mma_plan ownership layouts must expose warp-M first and warp-N last"
            );
        }
        Ok(())
    }
}
