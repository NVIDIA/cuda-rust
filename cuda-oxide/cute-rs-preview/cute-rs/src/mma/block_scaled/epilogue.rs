/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Store complete SM120 accumulator fragments through the shared epilogue.

use super::{__compiler, Mxf4AccumulatorTile2x8};
use crate::epilogue::sm120::Sm120EpilogueWarp128x128;

impl Sm120EpilogueWarp128x128 {
    /// Convert and store one warp's complete 2 x 8 result tile.
    ///
    /// ```text
    /// 2 x 8 FP32 cells -> FP16 -> shared memory -> TMA output
    /// ```
    ///
    /// Every cell has a fixed position. No runtime array index is used.
    ///
    /// # Safety
    ///
    /// All 32 lanes must call this together on the slice for their current
    /// warp. The caller must follow the synchronization rules documented by
    /// [`Sm120EpilogueWarp128x128::store_atom`].
    #[inline(always)]
    pub unsafe fn store_tile(self, tile: Mxf4AccumulatorTile2x8) {
        unsafe { __compiler::epilogue_store_fragment(self, tile) };
    }
}
