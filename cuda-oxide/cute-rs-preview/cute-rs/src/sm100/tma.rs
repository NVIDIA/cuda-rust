/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cluster copies into typed shared tiles for paired GEMM.

use super::{__compiler, Sm100SharedTile};
use crate::copy::tma::TensorMap;
use crate::markers::ReifySmem2D;
use core::marker::PhantomData;
use cuda_device::barrier::Barrier;

/// A cluster TMA copy whose descriptor and destination have the same layout.
pub struct Sm100ClusterTmaCopy<Layout>(
    #[doc = "Carries the destination layout required by the descriptor."] PhantomData<Layout>,
);
impl<Layout: ReifySmem2D> Sm100ClusterTmaCopy<Layout> {
    /// Copy an FP16 tile, routing completion to the leader CTA's barrier.
    ///
    /// Coordinates count elements. Each CTA supplies its rank and receives
    /// only its own tile; completion bytes from both CTAs accumulate at rank 0.
    ///
    /// # Safety
    ///
    /// All 32 lanes of one producer warp in each CTA call this operation
    /// together after acquiring the stage, with identical arguments within
    /// that warp. The leader must attach exactly one expectation covering both
    /// CTAs' copies to the same stage and phase. The native CuTe copy elects
    /// one lane to issue the copy and may synchronize the warp.
    /// Calling it from an already elected lane can deadlock when
    /// pipeline stages are reused.
    ///
    /// The descriptor must encode `Layout`, destination storage must be valid
    /// and aligned, and `barrier` names the local full barrier for this stage.
    #[inline(always)]
    pub unsafe fn copy<Role>(
        desc: *const TensorMap<f16, Layout>,
        dst: &mut Sm100SharedTile<Layout, Role>,
        row: u32,
        column: u32,
        barrier: *mut Barrier,
        rank: u32,
    ) {
        unsafe {
            __compiler::sm100_cluster_tma_load::<Layout>(desc, dst.base, row, column, barrier, rank)
        }
    }
}
