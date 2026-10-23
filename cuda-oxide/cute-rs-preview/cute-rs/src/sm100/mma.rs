/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Paired N-partition FP16 MMA with shared A-collector reuse.

use super::{__compiler, OperandA, OperandB, Sm100SharedTile};
use crate::markers::ReifySmem2D;
use core::marker::PhantomData;

/// Two-CTA FP16 tiled MMA with two N partitions sharing the A collector.
///
/// The A collector holds an input so two MMA instructions can reuse it:
///
/// ```text
/// each K=16 step:  A x B0 -> first N0 columns   (save A)
///                 A x B1 -> next N1 columns    (reuse, then release A)
/// ```
///
/// Each step uses a fresh A input. The complete output is
/// `256 x (N0 + N1)` with FP32 accumulation.
/// `K` must be 64; each N partition is a multiple of 16 in `16..=256`.
/// Per-CTA layouts cover A `(128, 64)`, B0 `(N0 / 2, 64)` and B1
/// `(N1 / 2, 64)`, in FP16 elements, with the supported B128 swizzle:
///
/// ```
/// use cute_rs::{ComposedLayout, RowMajor, Sm100TiledMma, Swizzle};
/// type Tile<const R: i64> = ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<R, 64>>;
/// let mma = Sm100TiledMma::<Tile<128>, Tile<96>, Tile<80>, 192, 160>::new();
/// ```
pub struct Sm100TiledMma<AL, B0L, B1L, const N0: u32, const N1: u32, const K: u32 = 64>(
    #[doc = "Carries the three shared operand maps; no device storage is allocated."]
    PhantomData<(AL, B0L, B1L)>,
);
impl<
    AL: ReifySmem2D,
    B0L: ReifySmem2D,
    B1L: ReifySmem2D,
    const N0: u32,
    const N1: u32,
    const K: u32,
> Sm100TiledMma<AL, B0L, B1L, N0, N1, K>
{
    /// Reject unsupported N partitions, K extents and per-CTA operand shapes at instantiation.
    const VALID: () = {
        assert!(N0 >= 16 && N0 <= 256 && N0.is_multiple_of(16));
        assert!(N1 >= 16 && N1 <= 256 && N1.is_multiple_of(16));
        assert!(N0 + N1 <= 512 && K == 64);
        assert!(AL::SHAPE.0 == 128 && AL::SHAPE.1 == K as i64);
        assert!(B0L::SHAPE.0 == (N0 / 2) as i64 && B0L::SHAPE.1 == K as i64);
        assert!(B1L::SHAPE.0 == (N1 / 2) as i64 && B1L::SHAPE.1 == K as i64);
    };

    /// Check the tile dimensions and create the MMA configuration.
    #[inline(always)]
    pub const fn new() -> Self {
        let () = Self::VALID;
        Self(PhantomData)
    }

    /// Accumulate one staged K tile into tensor memory.
    ///
    /// # Safety
    ///
    /// One elected lane in the leader CTA issues this operation after the
    /// input pipeline wait. Both CTAs' tiles must be full and remain unchanged
    /// until the asynchronous consumer release completes. `address` names a
    /// 512-column allocation; `accumulate=false` starts a new output tile.
    #[inline(always)]
    pub unsafe fn gemm(
        &self,
        address: u32,
        a: &Sm100SharedTile<AL, OperandA>,
        b0: &Sm100SharedTile<B0L, OperandB>,
        b1: &Sm100SharedTile<B1L, OperandB>,
        accumulate: bool,
    ) {
        unsafe {
            __compiler::sm100_tiled_mma::<AL, B0L, B1L, N0, N1, K>(
                address, a.base, b0.base, b1.base, accumulate,
            )
        }
    }
}
impl<
    AL: ReifySmem2D,
    B0L: ReifySmem2D,
    B1L: ReifySmem2D,
    const N0: u32,
    const N1: u32,
    const K: u32,
> Default for Sm100TiledMma<AL, B0L, B1L, N0, N1, K>
{
    fn default() -> Self {
        Self::new()
    }
}
