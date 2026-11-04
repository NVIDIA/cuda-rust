/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SM100 1-CTA and 2-CTA cluster TMA copy atoms (`cutlass.cute.nvgpu.cpasync`).

use super::TensorMap;
use crate::arch::Barrier;
use crate::layout::ReifySmem2D;
use crate::nvgpu::tcgen05::{__compiler, SharedTile, Sm100SharedTile, SmemLayout};
use core::marker::PhantomData;

/// A cluster TMA copy whose descriptor and destination have the same layout.
pub struct Sm100ClusterTmaCopy<Layout>(
    #[doc = "Carries the destination layout required by the descriptor."] PhantomData<Layout>,
);
impl<Layout: ReifySmem2D> Sm100ClusterTmaCopy<Layout> {
    /// Copy an FP16 tile, routing completion to the leader CTA's barrier.
    ///
    /// # Safety
    /// All 32 lanes of one producer warp in each CTA call this operation together.
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

/// A rank-three TMA copy into the same physical layout used by MMA.
pub struct TmaCopy<L, const GROUP: u32>(
    #[doc = "Retains the shared layout that must match the tensor-map descriptor."] PhantomData<L>,
);
impl<L: SmemLayout, const G: u32> TmaCopy<L, G> {
    /// Copy a physical row-major global tile into a typed shared tile.
    /// # Safety
    /// All producer-warp lanes participate. Coordinates are in physical global rows/columns.
    #[inline(always)]
    pub unsafe fn load<R>(
        desc: &TensorMap<f16, L>,
        dst: &mut SharedTile<L, R>,
        row: i32,
        col: i32,
        barrier: *mut Barrier,
        rank: u32,
    ) {
        unsafe {
            __compiler::tma_load::<L, G>(
                desc as *const _,
                dst.as_ptr(),
                col % L::SPAN as i32,
                row,
                col / L::SPAN as i32,
                barrier,
                rank,
            )
        }
    }
    /// # Safety
    /// One elected lane issues after stores to shared memory are visible to the async proxy.
    #[inline(always)]
    pub unsafe fn store<R>(desc: &TensorMap<f16, L>, src: &SharedTile<L, R>, row: i32, col: i32) {
        unsafe {
            __compiler::tma_store::<L, G>(
                desc as *const _,
                src.as_ptr(),
                col % L::SPAN as i32,
                row,
                col / L::SPAN as i32,
            )
        }
    }
}
