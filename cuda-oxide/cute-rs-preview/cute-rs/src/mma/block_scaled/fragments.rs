/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Typed scale, accumulator, and lazy B fragments for SM120 MMA.

use super::{__compiler, Mxf4TiledMma};
use crate::block_scaled::{Mkl, MmaScalePair, Nkl, ScalePack4};
use crate::copy::tiled::SharedTensor;
use crate::mma::{FragmentA, FragmentB, FragmentC};
use crate::numeric::Mxf4E2M1;
use core::marker::PhantomData;

/// Scales for the five inputs used by one warp in one K=64 step.
///
/// ```text
/// one A fragment + four B fragments
///       1        +        4         = 5 scale pairs
/// ```
///
/// [`Mkl`] marks the A scale. [`Nkl`] marks the B scales, so Rust rejects an
/// A/B scale swap.
#[derive(Clone, Copy)]
pub struct Mxf4ScalePairs {
    /// Scale pair for A.
    pub a: MmaScalePair<Mkl>,
    /// Scale pair for B columns 0 through 7.
    pub b0: MmaScalePair<Nkl>,
    /// Scale pair for B columns 8 through 15.
    pub b1: MmaScalePair<Nkl>,
    /// Scale pair for B columns 16 through 23.
    pub b2: MmaScalePair<Nkl>,
    /// Scale pair for B columns 24 through 31.
    pub b3: MmaScalePair<Nkl>,
}

/// Five packed scale words for one warp and one K=128 stage.
///
/// Each word contains four 8-bit scales. The lower two scales cover the first
/// K=64 step; the upper two cover the second K=64 step.
#[derive(Clone, Copy)]
pub struct Mxf4ScaleStage {
    /// Four K32 scales for the warp’s A fragment.
    pub(super) a: ScalePack4<Mkl>,
    /// Four K32 scales for B columns 0 through 7.
    pub(super) b0: ScalePack4<Nkl>,
    /// Four K32 scales for B columns 8 through 15.
    pub(super) b1: ScalePack4<Nkl>,
    /// Four K32 scales for B columns 16 through 23.
    pub(super) b2: ScalePack4<Nkl>,
    /// Four K32 scales for B columns 24 through 31.
    pub(super) b3: ScalePack4<Nkl>,
}

/// Packed scales for one warp's part of a 128 x 128 output tile.
///
/// One warp computes two M positions and eight N positions:
///
/// ```text
/// A scales: [a0] [a1]
/// B scales: [b0] [b1] [b2] [b3] [b4] [b5] [b6] [b7]
/// ```
///
/// Each word contains the four scales needed by the two K=64 steps in one
/// K=128 stage.
#[derive(Clone, Copy)]
pub struct Mxf4ScaleTile128 {
    /// Four K32 scales for the first 16-row A band.
    pub(super) a0: ScalePack4<Mkl>,
    /// Four K32 scales for the second A band, 64 rows after the first.
    pub(super) a1: ScalePack4<Mkl>,
    /// Four K32 scales for B’s first N8 fragment.
    pub(super) b0: ScalePack4<Nkl>,
    /// Four K32 scales for B’s second N8 fragment.
    pub(super) b1: ScalePack4<Nkl>,
    /// Four K32 scales for B’s third N8 fragment.
    pub(super) b2: ScalePack4<Nkl>,
    /// Four K32 scales for B’s fourth N8 fragment.
    pub(super) b3: ScalePack4<Nkl>,
    /// Four K32 scales for B’s fifth N8 fragment.
    pub(super) b4: ScalePack4<Nkl>,
    /// Four K32 scales for B’s sixth N8 fragment.
    pub(super) b5: ScalePack4<Nkl>,
    /// Four K32 scales for B’s seventh N8 fragment.
    pub(super) b6: ScalePack4<Nkl>,
    /// Four K32 scales for B’s eighth N8 fragment.
    pub(super) b7: ScalePack4<Nkl>,
}

/// Scale pairs for one warp's 16 MMA cells in one K=64 step.
///
/// The two A pairs combine with all eight B pairs:
///
/// ```text
///             b0 b1 b2 b3 b4 b5 b6 b7
/// a0           x  x  x  x  x  x  x  x
/// a1           x  x  x  x  x  x  x  x
/// ```
#[derive(Clone, Copy)]
pub struct Mxf4ScalePairs128 {
    /// A scale for rows starting at `16 * warp_m`.
    pub a0: MmaScalePair<Mkl>,
    /// A scale for rows starting at `64 + 16 * warp_m`.
    pub a1: MmaScalePair<Mkl>,
    /// B scale for columns starting at `16 * warp_n`.
    pub b0: MmaScalePair<Nkl>,
    /// B scale for columns starting at `16 * warp_n + 8`.
    pub b1: MmaScalePair<Nkl>,
    /// B scale for columns starting at `32 + 16 * warp_n`.
    pub b2: MmaScalePair<Nkl>,
    /// B scale for columns starting at `40 + 16 * warp_n`.
    pub b3: MmaScalePair<Nkl>,
    /// B scale for columns starting at `64 + 16 * warp_n`.
    pub b4: MmaScalePair<Nkl>,
    /// B scale for columns starting at `72 + 16 * warp_n`.
    pub b5: MmaScalePair<Nkl>,
    /// B scale for columns starting at `96 + 16 * warp_n`.
    pub b6: MmaScalePair<Nkl>,
    /// B scale for columns starting at `104 + 16 * warp_n`.
    pub b7: MmaScalePair<Nkl>,
}

/// One warp's FP32 results for a 128 x 128 output tile.
///
/// One MMA cell is a 16 x 8 result. Each lane holds four FP32 values from
/// every cell.
///
/// One warp owns this 2 x 8 grid of cells:
///
/// ```text
///                 eight N positions
///              0   1   2   3   4   5   6   7
/// first M band [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ]
/// second M band [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ]
/// ```
///
/// The 16 cells have fixed names inside the type. The compiler therefore
/// keeps them in registers without a runtime array index.
#[derive(Clone, Copy)]
pub struct Mxf4AccumulatorTile2x8 {
    /// Per-lane accumulators for row band 0 and N8 fragment 0.
    pub(super) c00: FragmentC,
    /// Per-lane accumulators for row band 0 and N8 fragment 1.
    pub(super) c01: FragmentC,
    /// Per-lane accumulators for row band 0 and N8 fragment 2.
    pub(super) c02: FragmentC,
    /// Per-lane accumulators for row band 0 and N8 fragment 3.
    pub(super) c03: FragmentC,
    /// Per-lane accumulators for row band 0 and N8 fragment 4.
    pub(super) c04: FragmentC,
    /// Per-lane accumulators for row band 0 and N8 fragment 5.
    pub(super) c05: FragmentC,
    /// Per-lane accumulators for row band 0 and N8 fragment 6.
    pub(super) c06: FragmentC,
    /// Per-lane accumulators for row band 0 and N8 fragment 7.
    pub(super) c07: FragmentC,
    /// Per-lane accumulators for row band 1 and N8 fragment 0.
    pub(super) c10: FragmentC,
    /// Per-lane accumulators for row band 1 and N8 fragment 1.
    pub(super) c11: FragmentC,
    /// Per-lane accumulators for row band 1 and N8 fragment 2.
    pub(super) c12: FragmentC,
    /// Per-lane accumulators for row band 1 and N8 fragment 3.
    pub(super) c13: FragmentC,
    /// Per-lane accumulators for row band 1 and N8 fragment 4.
    pub(super) c14: FragmentC,
    /// Per-lane accumulators for row band 1 and N8 fragment 5.
    pub(super) c15: FragmentC,
    /// Per-lane accumulators for row band 1 and N8 fragment 6.
    pub(super) c16: FragmentC,
    /// Per-lane accumulators for row band 1 and N8 fragment 7.
    pub(super) c17: FragmentC,
}

/// One warp's eight B fragments for one K=64 step.
///
/// This is a view, so creating it does not load all eight fragments. During
/// [`Mxf4AccumulatorTile2x8::gemm`], it loads two at a time:
///
/// ```text
/// load b0,b1 -> compute -> load b2,b3 -> compute -> ...
/// ```
///
/// That keeps fewer temporary values in registers.
pub struct Mxf4BTileK64<'a, SharedLayout> {
    /// Shared packed-value base, addressed in f16 carrier elements.
    pub(super) base: *mut f16,
    /// Available f16 carrier elements from the base pointer.
    pub(super) capacity: usize,
    /// Warp’s N partition, selecting its group of eight B fragments.
    pub(super) warp_n: usize,
    /// K64 half of the staged K128 input: zero or one.
    pub(super) k_half: usize,
    /// Borrows the original shared tensor and preserves its layout and B-operand role.
    pub(super) source: PhantomData<&'a SharedTensor<Mxf4E2M1, f16, SharedLayout, Nkl>>,
}

impl<SharedLayout> Mxf4BTileK64<'_, SharedLayout> {
    /// Load B fragments `2 * PAIR` and `2 * PAIR + 1`.
    ///
    /// # Safety
    ///
    /// All 32 lanes must call this together. `mma` must contain each calling
    /// thread's real lane number. This view's `warp_n` and `k_half` must each
    /// be 0 or 1, and its shared tile must still contain the completed stage.
    #[inline(always)]
    pub(super) unsafe fn load<const PAIR: usize>(
        &self,
        mma: &Mxf4TiledMma<SharedLayout>,
    ) -> (FragmentB, FragmentB) {
        const { assert!(PAIR < 4) };
        let source = unsafe {
            SharedTensor::<Mxf4E2M1, f16, SharedLayout, Nkl>::from_raw_parts(
                self.base,
                self.capacity,
            )
        };
        unsafe { mma.load_b_pair_128(&source, self.warp_n, PAIR, self.k_half) }
    }
}

impl Mxf4AccumulatorTile2x8 {
    /// Start a new output tile with every FP32 value set to zero.
    #[must_use]
    #[inline(always)]
    pub const fn zero() -> Self {
        __compiler::fragment_fill(0.0)
    }

    /// Add one K=64 input step to all 16 result cells.
    ///
    /// ```text
    /// two A fragments x eight B fragments
    ///                  |
    ///                  v
    ///       this 2 x 8 FP32 result tile
    /// ```
    ///
    /// `a` contains the two A fragments already loaded for this warp. `b`
    /// selects the eight matching B fragments and loads them two at a time.
    /// `scales` must cover the same A, B, and K=64 positions.
    ///
    /// [`Mxf4TiledMma::gemm`] is the one-cell form of this operation; this
    /// method runs sixteen of those cells for the warp's 2 x 8 tile.
    ///
    /// # Safety
    ///
    /// All 32 lanes must call this together. `mma` must contain each calling
    /// thread's real lane number. The A fragments, B view, and scales must use
    /// the same warp position and K=64 half from one completed shared-memory
    /// stage. The B view's `warp_n` and `k_half` must each be 0 or 1, and that
    /// stage must not be overwritten during this call.
    #[inline(always)]
    pub unsafe fn gemm<SharedLayout>(
        &mut self,
        mma: &Mxf4TiledMma<SharedLayout>,
        a: (FragmentA, FragmentA),
        b: Mxf4BTileK64<'_, SharedLayout>,
        scales: Mxf4ScalePairs128,
    ) {
        unsafe { __compiler::tiled_gemm(self, mma, a, b, scales) };
    }
}

impl Mxf4ScaleTile128 {
    /// Take the scale pairs for one K=64 half of a K=128 stage.
    ///
    /// # Safety
    ///
    /// `half` must be `0` for the first half or `1` for the second.
    #[must_use]
    #[inline(always)]
    pub const unsafe fn pairs_at_unchecked(self, half: usize) -> Mxf4ScalePairs128 {
        unsafe { __compiler::fragment_slice_k(self, half) }
    }
}

impl Mxf4ScaleStage {
    /// Take the scale pairs for the first or second K=64 half.
    ///
    /// `HALF=0` selects the first half. `HALF=1` selects the second.
    #[must_use]
    #[inline(always)]
    pub const fn pairs<const HALF: usize>(self) -> Mxf4ScalePairs {
        Mxf4ScalePairs {
            a: self.a.pair::<HALF>(),
            b0: self.b0.pair::<HALF>(),
            b1: self.b1.pair::<HALF>(),
            b2: self.b2.pair::<HALF>(),
            b3: self.b3.pair::<HALF>(),
        }
    }

    /// Take one K=64 half using a runtime value.
    ///
    /// # Safety
    ///
    /// `half` must be `0` for the first half or `1` for the second.
    #[must_use]
    #[inline(always)]
    pub const unsafe fn pairs_at_unchecked(self, half: usize) -> Mxf4ScalePairs {
        Mxf4ScalePairs {
            a: unsafe { self.a.pair_at_unchecked(half) },
            b0: unsafe { self.b0.pair_at_unchecked(half) },
            b1: unsafe { self.b1.pair_at_unchecked(half) },
            b2: unsafe { self.b2.pair_at_unchecked(half) },
            b3: unsafe { self.b3.pair_at_unchecked(half) },
        }
    }
}
