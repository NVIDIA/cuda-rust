/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Independent shared/shared and tensor/shared FP16 attention MMA.

use super::{__compiler, SharedTile, SmemLayout};
use crate::sm100::{OperandA, OperandB};
use core::marker::PhantomData;

/// Independent shared/shared FP16 MMA with FP32 accumulation.
///
/// `M = 128`; `N` is a multiple of 16 in `16..=256`; `K` is a positive
/// multiple of 16; `GROUP` is 1 or 2. M/N describe the complete product.
/// Each CTA's layouts must cover A `(M / GROUP, K)` and B `(N / GROUP, K)`
/// and satisfy [`SharedLayout`](super::SharedLayout)'s restrictions.
///
/// ```
/// use cute_rs::attention::sm100::{MmaSs, SharedLayout};
/// type LocalTile = SharedLayout<64, 64>;
/// let mma = MmaSs::<LocalTile, LocalTile, 128, 128, 64, 2>::new();
/// ```
pub struct MmaSs<AL, BL, const M: u32, const N: u32, const K: u32, const GROUP: u32>(
    #[doc = "Carries both per-CTA shared operand layouts."] PhantomData<(AL, BL)>,
);
impl<AL: SmemLayout, BL: SmemLayout, const M: u32, const N: u32, const K: u32, const G: u32>
    MmaSs<AL, BL, M, N, K, G>
{
    /// Check group size, collective dimensions and per-CTA shared operand shapes.
    const VALID: () = {
        let () = AL::VALID;
        let () = BL::VALID;
        assert!(G == 1 || G == 2);
        assert!(M == 128 && N >= 16 && N <= 256 && N.is_multiple_of(16));
        assert!(K >= 16 && K.is_multiple_of(16));
        assert!(AL::ROWS == M / G && BL::ROWS == N / G);
        assert!(AL::COLS == K && BL::COLS == K);
    };
    /// Validate the static MMA dimensions and operand layouts.
    #[inline(always)]
    pub const fn new() -> Self {
        let () = Self::VALID;
        Self(PhantomData)
    }
    /// Issue the complete K tile, with the first instruction honoring `accumulate`.
    /// # Safety
    /// One elected leader lane issues MMA after input completion. Both shared
    /// views and destination TMEM must remain live until its completion barrier.
    /// Every operation using the allocation must use the same CTA group.
    #[inline(always)]
    pub unsafe fn gemm(
        &self,
        dst: u32,
        a: &SharedTile<AL, OperandA>,
        b: &SharedTile<BL, OperandB>,
        accumulate: bool,
    ) {
        unsafe {
            __compiler::mma_ss::<AL, BL, M, N, K, G>(
                dst,
                a.base,
                b.base,
                accumulate,
                a.byte_offset,
                b.byte_offset,
            )
        }
    }
}
impl<AL: SmemLayout, BL: SmemLayout, const M: u32, const N: u32, const K: u32, const G: u32> Default
    for MmaSs<AL, BL, M, N, K, G>
{
    fn default() -> Self {
        Self::new()
    }
}

/// TMEM/shared FP16 MMA, used for the probability/value product in FMHA.
///
/// `GROUP = 1`, `M = 128`, and `N` is a multiple of 16 in `16..=256`.
/// The importer supports `K = 32` or a positive multiple of 64. B's shared
/// layout must cover `(N, K)` and satisfy [`SharedLayout`](super::SharedLayout).
///
/// ```
/// use cute_rs::attention::sm100::{MmaTs, SharedLayout};
/// let mma = MmaTs::<SharedLayout<128, 64>, 128, 128, 64, 1>::new();
/// ```
pub struct MmaTs<BL, const M: u32, const N: u32, const K: u32, const GROUP: u32>(
    #[doc = "Carries the shared B operand’s physical layout."] PhantomData<BL>,
);
impl<BL: SmemLayout, const M: u32, const N: u32, const K: u32, const G: u32> MmaTs<BL, M, N, K, G> {
    /// Check the group-one TMEM/shared profile, supported K span and B layout shape.
    const VALID: () = {
        let () = BL::VALID;
        assert!(G == 1 && M == 128);
        assert!(N >= 16 && N <= 256 && N.is_multiple_of(16));
        assert!(K == 32 || (K > 0 && K.is_multiple_of(64)));
        assert!(BL::ROWS == N && BL::COLS == K);
    };
    /// Validate the static MMA dimensions and operand layouts.
    #[inline(always)]
    pub const fn new() -> Self {
        let () = Self::VALID;
        Self(PhantomData)
    }
    /// # Safety
    /// An elected lane issues this after packed FP16 probability stores and
    /// shared input copies are visible to MMA. Source and destination TMEM
    /// regions must not overlap and must remain live until completion.
    #[inline(always)]
    pub unsafe fn gemm(
        &self,
        dst: u32,
        a_tmem: u32,
        b: &SharedTile<BL, OperandB>,
        accumulate: bool,
    ) {
        unsafe {
            __compiler::mma_ts::<BL, M, N, K, G>(dst, a_tmem, b.base, accumulate, b.byte_offset)
        }
    }
}
impl<BL: SmemLayout, const M: u32, const N: u32, const K: u32, const G: u32> Default
    for MmaTs<BL, M, N, K, G>
{
    fn default() -> Self {
        Self::new()
    }
}
