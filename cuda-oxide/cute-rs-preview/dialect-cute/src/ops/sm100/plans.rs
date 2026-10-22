/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Typed plans for SM100 tensor memory, MMA, TMA, and collective pipelines.

use super::*;

/// The first N partition fills A, and the second consumes its last use.
#[pliron_attr(name = "cute.sm100_collector", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum CuteSm100CollectorAttr {
    /// Fill the A collector for the first product and consume its last use in the second.
    AFillLastUse,
}

/// Capacity of one two-CTA tensor-memory allocation, in columns per CTA.
///
/// Each CTA has 128 rows. The supported column count is a power of two in
/// `32..=512`; allocation and deallocation must use the same plan.
#[pliron_attr(
    name = "cute.sm100_tmem_plan",
    format = "`<` $columns `,` $cta_group `>`"
)]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct CuteSm100TmemPlanAttr {
    /// Tensor-memory columns allocated per CTA.
    pub columns: u32,
    /// Number of CTAs participating in the collective contract.
    pub cta_group: u32,
}
impl Verify for CuteSm100TmemPlanAttr {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        if self.cta_group != 2
            || !(32..=512).contains(&self.columns)
            || !self.columns.is_power_of_two()
        {
            return verify_err_noloc!(
                "SM100 tensor memory requires CTA group 2 and power-of-two columns in 32..=512"
            );
        }
        Ok(())
    }
}

/// Two FP16 MMA products sharing A, accumulated into adjacent TMEM partitions.
///
/// `m`, `n0`, `n1`, and `k` describe the complete two-CTA products. The shared
/// layouts describe each CTA's portion: A is `(m / 2, k)`, and each B is
/// `(n / 2, k)`, all K-major with a 128-byte swizzle. The current path fixes
/// `m = 256`, `k = 64`, and `tmem_columns = 512`; each N is a multiple of 16
/// in `16..=256`. B1's output follows B0's `n0` columns.
#[pliron_attr(
    name = "cute.sm100_mma_plan",
    format = "`<` $m `,` $n0 `,` $n1 `,` $k `,` $tmem_columns `,` $cta_group `,` $collector `,` $a_layout `,` $b0_layout `,` $b1_layout `>`"
)]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct CuteSm100MmaPlanAttr {
    /// Collective output-row extent M.
    pub m: u32,
    /// Output columns in the first product, starting at TMEM column zero.
    pub n0: u32,
    /// Output columns in the second product, immediately following the first.
    pub n1: u32,
    /// Reduction extent K, in logical elements.
    pub k: u32,
    /// TMEM capacity in columns available to both output partitions.
    pub tmem_columns: u32,
    /// Number of CTAs participating in the collective contract.
    pub cta_group: u32,
    /// Lifetime policy for the shared A collector across the two products.
    pub collector: CuteSm100CollectorAttr,
    /// Physical shared placement of each CTA's A tile.
    pub a_layout: CuteComposedLayoutAttr,
    /// Physical shared placement of each CTA's first B tile.
    pub b0_layout: CuteComposedLayoutAttr,
    /// Physical shared placement of each CTA's second B tile.
    pub b1_layout: CuteComposedLayoutAttr,
}

impl CuteSm100MmaPlanAttr {
    /// Bytes copied by both CTAs for one complete A/B0/B1 stage.
    pub fn transaction_bytes(&self) -> Option<u32> {
        self.m
            .checked_add(self.n0)?
            .checked_add(self.n1)?
            .checked_mul(self.k)?
            .checked_mul(2)
    }
}

impl Verify for CuteSm100MmaPlanAttr {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        CuteSm100TmemPlanAttr {
            columns: self.tmem_columns,
            cta_group: self.cta_group,
        }
        .verify(ctx)?;
        if self.m != 256
            || self.k != 64
            || self.tmem_columns != 512
            || !(16..=256).contains(&self.n0)
            || !(16..=256).contains(&self.n1)
            || !self.n0.is_multiple_of(16)
            || !self.n1.is_multiple_of(16)
            || self.n0 + self.n1 > self.tmem_columns
        {
            return verify_err_noloc!(
                "SM100 paired FP16 MMA requires M256/K64, two N partitions in 16..=256 divisible by 16, and 512 TMEM columns"
            );
        }
        for (name, layout, rows) in [
            ("A", &self.a_layout, 128),
            ("B0", &self.b0_layout, self.n0 / 2),
            ("B1", &self.b1_layout, self.n1 / 2),
        ] {
            layout.verify(ctx)?;
            if layout.0 != row_major(rows, self.k, 3) {
                return verify_err_noloc!(
                    "SM100 {name} MMA layout must be the matching K-major FP16 B128 shared tile"
                );
            }
        }
        Ok(())
    }
}

/// One CTA's FP16 input tile in a two-CTA cluster TMA copy.
///
/// The layout is K-major with 64 columns and a 128-byte swizzle; its row count
/// is a multiple of eight in `8..=128`. Completion is routed to leader rank
/// zero. The caller's barrier expectation covers bytes copied by both CTAs.
#[pliron_attr(
    name = "cute.sm100_cluster_copy_plan",
    format = "`<` $layout `,` $cta_group `,` $leader_rank `>`"
)]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct CuteSm100ClusterCopyPlanAttr {
    /// Static coordinate layout used to interpret the storage.
    pub layout: CuteComposedLayoutAttr,
    /// Number of CTAs participating in the collective contract.
    pub cta_group: u32,
    /// CTA rank receiving collective completion notifications.
    pub leader_rank: u32,
}
impl Verify for CuteSm100ClusterCopyPlanAttr {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        self.layout.verify(ctx)?;
        let dimensions = self.layout.0.outer().shape.flatten();
        if self.cta_group != 2
            || self.leader_rank != 0
            || dimensions.len() != 2
            || dimensions[0] < 8
            || dimensions[0] > 128
            || dimensions[0] % 8 != 0
            || dimensions[1] != 64
            || self.layout.0 != row_major(dimensions[0] as u32, 64, 3)
        {
            return verify_err_noloc!(
                "SM100 cluster TMA requires CTA group 2, leader 0, and a K64 FP16 B128 tile with 8..=128 rows divisible by 8"
            );
        }
        Ok(())
    }
}

/// A 128-by-32 FP16 shared output tile, with a 64-byte swizzle.
///
/// The epilogue fills this tile from FP32 TMEM, and the store operation sends
/// the same physical layout to global memory. Rows and columns count elements.
#[pliron_attr(
    name = "cute.sm100_epilogue_plan",
    format = "`<` $layout `,` $rows `,` $columns `>`"
)]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct CuteSm100EpiloguePlanAttr {
    /// Static coordinate layout used to interpret the storage.
    pub layout: CuteComposedLayoutAttr,
    /// Row extent of the shared output tile, in elements.
    pub rows: u32,
    /// Allocated or copied column extent, in elements.
    pub columns: u32,
}
impl Verify for CuteSm100EpiloguePlanAttr {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        self.layout.verify(ctx)?;
        if self.rows != 128 || self.columns != 32 || self.layout.0 != row_major(128, 32, 2) {
            return verify_err_noloc!("SM100 epilogue requires the 128x32 FP16 B64 shared layout");
        }
        Ok(())
    }
}

/// Static barrier configuration for either input stages or one accumulator.
///
/// Positive `transaction_bytes` selects the input pipeline: it counts all
/// A/B0/B1 bytes copied by both CTAs per stage, with one MMA completion arrival
/// releasing the stage. Zero selects the accumulator pipeline: one stage and
/// one release arrival per consumer warp across both CTAs. The current
/// epilogue has four consumer warps per CTA, hence eight arrivals.
/// Both forms use CTA group two, leader rank zero, and multicast mask `0b11`.
///
/// ```
/// use dialect_cute::ops::sm100::CuteSm100PipelinePlanAttr;
/// use pliron::{common_traits::Verify, context::Context};
///
/// let accumulator = CuteSm100PipelinePlanAttr {
///     stages: 1,
///     transaction_bytes: 0,
///     consumer_arrivals: 8,
///     cta_group: 2,
///     leader_rank: 0,
///     multicast_mask: 0b11,
/// };
/// accumulator.verify(&Context::new()).unwrap();
/// ```
#[pliron_attr(
    name = "cute.sm100_pipeline_plan",
    format = "`<` $stages `,` $transaction_bytes `,` $consumer_arrivals `,` $cta_group `,` $leader_rank `,` $multicast_mask `>`"
)]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct CuteSm100PipelinePlanAttr {
    /// Number of reusable slots in the barrier or output-buffer ring.
    pub stages: u32,
    /// Bytes expected by the full barrier for one complete input stage; zero selects accumulator synchronization.
    pub transaction_bytes: u32,
    /// Slot-release arrivals: one MMA completion for input, or one per epilogue warp for accumulators.
    pub consumer_arrivals: u32,
    /// Number of CTAs participating in the collective contract.
    pub cta_group: u32,
    /// CTA rank receiving collective completion notifications.
    pub leader_rank: u32,
    /// Bit mask of CTA ranks notified by a collective completion.
    pub multicast_mask: u32,
}
impl Verify for CuteSm100PipelinePlanAttr {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        if self.stages == 0
            || self.cta_group != 2
            || self.leader_rank != 0
            || self.multicast_mask != 3
        {
            return verify_err_noloc!(
                "SM100 pipeline requires positive stages, CTA group 2, leader 0, and multicast mask 3"
            );
        }
        if self.transaction_bytes == 0 {
            if self.stages != 1
                || self.consumer_arrivals == 0
                || self.consumer_arrivals > 64
                || !self.consumer_arrivals.is_multiple_of(2)
            {
                return verify_err_noloc!(
                    "SM100 accumulator pipeline requires one stage and an even number of consumer arrivals in 2..=64"
                );
            }
        } else if self.consumer_arrivals != 1 || !self.transaction_bytes.is_multiple_of(16) {
            return verify_err_noloc!(
                "SM100 input pipeline requires one MMA completion arrival and a transaction size divisible by 16"
            );
        }
        Ok(())
    }
}
