/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Paired N-partition FP16 MMA with shared A-collector reuse and group-one
//! block-scaled NVFP4 (`MmaMXF4NVF4Op`) tiled MMA.

use super::{
    __compiler, OperandA, OperandB, SharedTile, Sm100ScaleTile, Sm100SharedTile, SmemLayout,
};
use crate::layout::ReifySmem2D;
use crate::utils::blockscaled::{Mkl, Nkl};
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

/// Group-one Blackwell block-scaled NVFP4 (`tcgen05.MmaMXF4NVF4Op`) tiled MMA with
/// `tcgen05.Cp4x32x128bOp` (`SmemToTmemCopy`) scale copy.
///
/// Mirrors CuTeDSL's `MmaMXF4NVF4Op` `TiledMma` + `Cp4x32x128bOp` (`cute.copy` + `cute.gemm`)
/// while encoding the A and B shared-memory tile layouts (`AL`, `BL`) and operand roles
/// (`OperandA`, `OperandB`, `Mkl`, `Nkl`) directly in Rust's type system.
#[derive(Clone, Copy, Debug)]
pub struct Sm100BlockScaledTiledMma<AL, BL, const M: u32 = 128, const K: u32 = 64> {
    /// Active N column count (`16..=256`, multiple of 16) for this block-scaled `TiledMma`.
    /// Lowered into the `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16` plan by `dialect-cute`.
    valid_n: u32,
    /// Leading-dimension byte stride of the B shared-memory sub-tile (`valid_n * 16`).
    /// Separates the low 16 bytes (`K = 0..31`) from the high 16 bytes (`K = 32..63`) of each row.
    b_lbo_bytes: u32,
    /// Zero-byte marker carrying the compile-time A and B shared-memory layout types (`AL`, `BL`).
    /// Checked against the [`Sm100SharedTile`] arguments passed to [`Self::gemm`].
    marker: PhantomData<(AL, BL)>,
}

impl<AL: ReifySmem2D, BL: ReifySmem2D, const M: u32, const K: u32>
    Sm100BlockScaledTiledMma<AL, BL, M, K>
{
    /// Compile-time verification that `M = 128`, `K = 64`, `AL` covers `128` rows,
    /// and `BL` covers `64`, `128`, or `256` rows.
    const VALID: () = {
        assert!(M == 128 && K == 64);
        assert!(AL::SHAPE.0 == 128);
        assert!(BL::SHAPE.0 == 64 || BL::SHAPE.0 == 128 || BL::SHAPE.0 == 256);
    };

    /// Construct a `Sm100BlockScaledTiledMma` for `valid_n` active columns (`16..=256`, multiple of 16)
    /// using `valid_n * 16` as the B sub-tile leading-dimension byte offset.
    #[must_use]
    #[inline(always)]
    pub const fn new(valid_n: u32) -> Self {
        Self::new_with_lbo(valid_n, valid_n * 16)
    }

    /// Construct a `Sm100BlockScaledTiledMma` for `valid_n` active columns (`16..=256`, multiple of 16)
    /// and an explicit B sub-tile leading-dimension byte offset (`b_lbo_bytes`, e.g. `4096` for a
    /// fixed 256-row TMA tile or `2048` for a fixed 128-row TMA tile).
    #[must_use]
    #[inline(always)]
    pub const fn new_with_lbo(valid_n: u32, b_lbo_bytes: u32) -> Self {
        let () = Self::VALID;
        Self {
            valid_n,
            b_lbo_bytes,
            marker: PhantomData,
        }
    }

    /// Copy one 512-byte `(32, 4, 4)` scale atom at `step` from shared memory (`s_scale`) into
    /// TMEM column `tmem_col` via `tcgen05.cp.32x128b.warpx4` (CuTeDSL `Cp4x32x128bOp` + `cute.copy`).
    ///
    /// # Safety
    /// One elected lane in the CTA must issue this operation after `s_scale` is published in shared memory.
    #[inline(always)]
    pub unsafe fn copy<Role>(tmem_col: u32, s_scale: &Sm100ScaleTile<Role>, step: usize) {
        unsafe {
            __compiler::sm100_block_scaled_copy(tmem_col, s_scale.base, step as u32);
        }
    }

    /// Issue one `K = 64` step (`step`) of a group-one block-scaled `mxf4nvf4` MMA (`CuTeDSL cute.gemm`),
    /// reading operand sub-tile `step` from `a` and `b` and scale columns `sfa_tmem` and `sfb_tmem`.
    ///
    /// # Safety
    /// One elected lane in the CTA must issue this operation after `a`, `b`, and the TMEM scales
    /// at `sfa_tmem` and `sfb_tmem` are ready.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub unsafe fn gemm(
        &self,
        tmem_d: u32,
        a: &Sm100SharedTile<AL, OperandA>,
        b: &Sm100SharedTile<BL, OperandB>,
        sfa_tmem: u32,
        sfb_tmem: u32,
        step: usize,
        accumulate: bool,
    ) {
        let _ = (Mkl, Nkl);
        unsafe {
            __compiler::sm100_block_scaled_gemm::<AL, BL, M, K>(
                tmem_d,
                a.base,
                b.base,
                sfa_tmem,
                sfb_tmem,
                self.valid_n,
                self.b_lbo_bytes,
                step as u32,
                accumulate,
            );
        }
    }
}

/// Independent shared/shared FP16 MMA with FP32 accumulation (`tcgen05.MmaF16BF16Op`).
///
/// `M = 128`; `N` is a multiple of 16 in `16..=256`; `K` is a positive
/// multiple of 16; `GROUP` is 1 or 2. M/N describe the complete product.
/// Each CTA's layouts must cover A `(M / GROUP, K)` and B `(N / GROUP, K)`
/// and satisfy [`SharedLayout`](super::SharedLayout)'s restrictions.
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

/// TMEM/shared FP16 MMA (`tcgen05.MmaF16BF16Op` with `OperandSource.TMEM`).
///
/// `GROUP = 1`, `M = 128`, and `N` is a multiple of 16 in `16..=256`.
/// The importer supports `K = 32` or a positive multiple of 64. B's shared
/// layout must cover `(N, K)` and satisfy [`SharedLayout`](super::SharedLayout).
pub struct MmaTs<BL, const M: u32, const N: u32, const K: u32, const GROUP: u32>(
    #[doc = "Carries the shared B operand's physical layout."] PhantomData<BL>,
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
