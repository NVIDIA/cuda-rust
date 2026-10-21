/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared output placement and asynchronous store synchronization contracts.

use cute_layout::{ComposedLayout, Layout, OffsetUnit, Swizzle};
use pliron::{common_traits::Verify, context::Context, result::Error, verify_err_noloc};
use pliron_derive::pliron_attr;

use super::{layout::CuteComposedLayoutAttr, mma::CuteTiledMmaPlanAttr, tensor::CuteAlignmentAttr};

/// How many result buffers the asynchronous TMA store queue may reuse.
///
/// The value has no runtime storage. It only explains the wait count used by
/// the three store-pipeline operations:
///
/// ```text
/// acquire  waits for at most stages - 1 old readers
/// commit   closes the stores issued since the previous commit
/// tail     waits for zero old readers
/// ```
#[pliron_attr(name = "cute.tma_store_pipeline", format = "`<` $stages `>`")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct CuteTmaStorePipelineAttr {
    /// Number of reusable slots in the barrier or output-buffer ring.
    pub stages: u32,
}

impl CuteTmaStorePipelineAttr {
    /// Record the number of output buffers tracked by asynchronous store groups.
    #[must_use]
    pub const fn new(stages: u32) -> Self {
        Self { stages }
    }

    /// Number passed to `cp.async.bulk.wait_group.read` before reuse.
    #[must_use]
    pub const fn max_pending(self) -> Option<u32> {
        self.stages.checked_sub(1)
    }
}

impl Verify for CuteTmaStorePipelineAttr {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        if self.stages == 0 {
            return verify_err_noloc!("cute.tma_store_pipeline needs at least one stage");
        }
        Ok(())
    }
}

/// One named CTA barrier used by a contiguous group of warps.
///
/// The current epilogue uses warps 0 through 7 and leaves warp 8 out:
///
/// ```text
/// CTA warps       0 1 2 3 4 5 6 7 | 8
/// barrier ID 2    [--- 256 threads ---] | producer
/// ```
#[pliron_attr(
    name = "cute.counted_cta_barrier",
    format = "`<` $barrier_id `,` $first_warp `,` $warp_count `,` $cta_warps `,` $lanes_per_warp `>`"
)]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct CuteCountedCtaBarrierAttr {
    /// Named CTA barrier identifier, in `0..16`.
    pub barrier_id: u32,
    /// First participating warp index within the CTA.
    pub first_warp: u32,
    /// Number of consecutive participating warps.
    pub warp_count: u32,
    /// Total warps in the CTA, including excluded producers.
    pub cta_warps: u32,
    /// Thread count contributed by each participating warp.
    pub lanes_per_warp: u32,
}

impl CuteCountedCtaBarrierAttr {
    /// Record the barrier identifier and exact participating/excluded warp ranges.
    #[must_use]
    pub const fn new(
        barrier_id: u32,
        first_warp: u32,
        warp_count: u32,
        cta_warps: u32,
        lanes_per_warp: u32,
    ) -> Self {
        Self {
            barrier_id,
            first_warp,
            warp_count,
            cta_warps,
            lanes_per_warp,
        }
    }

    /// Threads that must arrive at the counted barrier.
    #[must_use]
    pub const fn participant_threads(self) -> Option<u32> {
        self.warp_count.checked_mul(self.lanes_per_warp)
    }

    /// Warps in the CTA that deliberately do not arrive.
    #[must_use]
    pub const fn excluded_warps(self) -> Option<u32> {
        self.cta_warps.checked_sub(self.warp_count)
    }
}

impl Verify for CuteCountedCtaBarrierAttr {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        if self.barrier_id >= 16 {
            return verify_err_noloc!("cute.counted_cta_barrier ID must be between 0 and 15");
        }
        if self.warp_count == 0 || self.cta_warps == 0 || self.lanes_per_warp == 0 {
            return verify_err_noloc!(
                "cute.counted_cta_barrier warp and lane counts must be greater than zero"
            );
        }
        let Some(end_warp) = self.first_warp.checked_add(self.warp_count) else {
            return verify_err_noloc!("cute.counted_cta_barrier warp range overflowed");
        };
        if end_warp > self.cta_warps {
            return verify_err_noloc!(
                "cute.counted_cta_barrier participating warps must lie inside the CTA"
            );
        }
        let Some(threads) = self.participant_threads() else {
            return verify_err_noloc!("cute.counted_cta_barrier thread count overflowed");
        };
        let Some(cta_threads) = self.cta_warps.checked_mul(self.lanes_per_warp) else {
            return verify_err_noloc!("cute.counted_cta_barrier CTA thread count overflowed");
        };
        if threads == 0 || threads > 1024 || cta_threads > 1024 {
            return verify_err_noloc!(
                "cute.counted_cta_barrier cannot describe more than 1024 CTA threads"
            );
        }
        Ok(())
    }
}

/// Why the compute warps meet at the epilogue barrier.
#[pliron_attr(name = "cute.epilogue_sync_phase", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum CuteEpilogueSyncPhaseAttr {
    /// The previous TMA reader is done, so writers may reuse shared memory.
    Reusable,
    /// Publish generic-proxy writes, then release TMA to read shared memory.
    ReadyForTma,
}

/// Left or right 128x64 half of the current 128x128 result tile.
#[pliron_attr(name = "cute.epilogue_half_index", format = "$0")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct CuteEpilogueHalfAttr(#[doc = "Zero-based left/right half index."] pub u32);

impl Verify for CuteEpilogueHalfAttr {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        if self.0 >= 2 {
            return verify_err_noloc!("cute.epilogue_half_index v0 must be 0 or 1");
        }
        Ok(())
    }
}

/// How one warp's FP32 accumulator reaches a shared FP16 result tile.
///
/// ```text
/// 2 x 8 accumulator cells per warp
///              │
///              ▼
/// logical 128x128 f16 tile
/// ┌────── 128x64 ──────┬────── 128x64 ──────┐
/// │ one B128 TMA store │ one B128 TMA store │
/// └────────────────────┴────────────────────┘
/// ```
#[pliron_attr(
    name = "cute.epilogue_plan",
    format = "`<` $tiled_mma `,` $half_layout `,` $halves `,` $base_alignment `>`"
)]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct CuteEpiloguePlanAttr {
    /// Compute plan whose accumulator mapping feeds this output tile.
    pub tiled_mma: CuteTiledMmaPlanAttr,
    /// Physical shared placement of one output half.
    pub half_layout: CuteComposedLayoutAttr,
    /// Number of shared output halves in the complete result tile.
    pub halves: u32,
    /// Promised shared-base alignment, in bytes.
    pub base_alignment: CuteAlignmentAttr,
}

impl CuteEpiloguePlanAttr {
    /// Current SM120 two-half FP16 epilogue for the 128x128 MXFP4 plan.
    #[must_use]
    pub fn sm120_mxf4_128x128(tiled_mma: CuteTiledMmaPlanAttr) -> Self {
        let inner: Layout = "(128,64):(64,1)"
            .parse()
            .expect("fixed epilogue half layout is valid");
        let half_layout =
            ComposedLayout::new(Swizzle::new(3, 3, 3), 0, inner, OffsetUnit::Elements)
                .expect("fixed epilogue swizzle is valid");
        Self {
            tiled_mma,
            half_layout: CuteComposedLayoutAttr(half_layout),
            halves: 2,
            base_alignment: CuteAlignmentAttr(1024),
        }
    }

    /// Return the element capacity of one output half, or `None` on overflow.
    #[must_use]
    pub fn half_elements(&self) -> Option<u64> {
        u64::try_from(self.half_layout.0.inner().checked_size()?).ok()
    }

    /// Return the element capacity of the complete output tile, or `None` on overflow.
    #[must_use]
    pub fn full_elements(&self) -> Option<u64> {
        self.half_elements()?.checked_mul(u64::from(self.halves))
    }

    /// Return one output half's byte capacity, or `None` for an invalid size.
    #[must_use]
    pub fn half_bytes(&self, element_bytes: u64) -> Option<u64> {
        self.half_elements()?.checked_mul(element_bytes)
    }

    /// Return the complete output tile's byte capacity, or `None` for an invalid size.
    #[must_use]
    pub fn full_bytes(&self, element_bytes: u64) -> Option<u64> {
        self.full_elements()?.checked_mul(element_bytes)
    }
}

impl Verify for CuteEpiloguePlanAttr {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        self.tiled_mma.verify(ctx)?;
        self.base_alignment.verify(ctx)?;

        let data_inner: Layout = "(128,32):(32,1)"
            .parse()
            .expect("fixed SM120 MMA shared layout is valid");
        let data_layout =
            ComposedLayout::new(Swizzle::new(2, 3, 3), 0, data_inner, OffsetUnit::Elements)
                .expect("fixed SM120 MMA shared swizzle is valid");
        let expected_mma = CuteTiledMmaPlanAttr::mxf4_128x128x128(data_layout);
        if self.tiled_mma != expected_mma {
            return verify_err_noloc!(
                "cute.epilogue_plan v0 needs the 128x128x128 SM120 MXFP4 tiled-MMA plan"
            );
        }
        let expected = Self::sm120_mxf4_128x128(self.tiled_mma.clone());
        if self.half_layout != expected.half_layout
            || self.halves != expected.halves
            || self.base_alignment != expected.base_alignment
        {
            return verify_err_noloc!(
                "cute.epilogue_plan v0 needs two B128-swizzled 128x64 f16 halves aligned to 1024 bytes"
            );
        }
        let modes = self.half_layout.0.inner().modes();
        if modes.len() != 2
            || modes[0].checked_size() != Some(128)
            || modes[1].checked_size() != Some(64)
            || self.full_elements() != Some(128 * 128)
        {
            return verify_err_noloc!(
                "cute.epilogue_plan halves must cover one 128x128 result tile"
            );
        }
        if cute_layout::tma_phase_alignment_bytes(&self.half_layout.0, 2) != Ok(1024) {
            return verify_err_noloc!(
                "cute.epilogue_plan half layout must use the 1024-byte B128 phase alignment"
            );
        }
        if self.tiled_mma.accumulator_registers_per_lane() != Some(64) {
            return verify_err_noloc!(
                "cute.epilogue_plan v0 needs 64 FP32 accumulator registers per lane"
            );
        }
        Ok(())
    }
}
