/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SM100 1-CTA and 2-CTA cluster TMA copy atoms (`cutlass.cute.nvgpu.cpasync`).

use super::TensorMap;
use crate::arch::Barrier;
use crate::layout::ReifySmem2D;
use crate::nvgpu::tcgen05::{__compiler, MmaHalfElement, SharedTile, Sm100SharedTile, SmemLayout};
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
pub struct TmaCopy<L, const GROUP: u32, T = f16>(
    #[doc = "Retains the shared layout and 16-bit element type matching the tensor-map descriptor."]
     PhantomData<(L, T)>,
);
impl<L: SmemLayout, const G: u32, T: MmaHalfElement> TmaCopy<L, G, T> {
    /// Copy a physical row-major global tile into a typed shared tile.
    /// # Safety
    /// All producer-warp lanes participate. Coordinates are in physical global rows/columns.
    #[inline(always)]
    pub unsafe fn load<R>(
        desc: &TensorMap<T, L>,
        dst: &mut SharedTile<L, R, T>,
        row: i32,
        col: i32,
        barrier: *mut Barrier,
        rank: u32,
    ) {
        unsafe {
            __compiler::tma_load::<L, G, T>(
                (desc as *const TensorMap<T, L>).cast::<TensorMap<f16, L>>(),
                dst.as_ptr().cast::<f16>(),
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
    pub unsafe fn store<R>(desc: &TensorMap<T, L>, src: &SharedTile<L, R, T>, row: i32, col: i32) {
        unsafe {
            __compiler::tma_store::<L, G, T>(
                (desc as *const TensorMap<T, L>).cast::<TensorMap<f16, L>>(),
                src.as_ptr().cast::<f16>(),
                col % L::SPAN as i32,
                row,
                col / L::SPAN as i32,
            )
        }
    }
}
