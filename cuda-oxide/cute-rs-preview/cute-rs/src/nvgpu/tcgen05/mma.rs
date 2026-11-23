/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Paired N-partition FP16 MMA with shared A-collector reuse and group-one
//! block-scaled NVFP4 (`MmaMXF4NVF4Op`) tiled MMA.

use super::{
    __compiler, MmaHalfElement, OperandA, OperandB, SharedTile, Sm100ScaleTile, Sm100SharedTile,
    SmemLayout,
};
use crate::layout::ReifySmem2D;
use crate::utils::blockscaled::{Mkl, Nkl};
use core::marker::PhantomData;

/// Controls how an SM107 Tensor Core collector buffer retains or releases an operand (`CollectorOp`).
///
/// Pair `Fill` with `LastUse` to load `B` from shared memory once across two `A` tiles:
///
/// ```text
/// MMA 0 (CollectorFill):    reads B from SMEM and stashes it in the B collector
/// MMA 1 (CollectorLastUse): reuses B from the collector and frees the slot
/// ```
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollectorOp {
    /// Read from shared memory and discard the collector slot (`CollectorOp.DISCARD`).
    Discard = 0,
    /// Reuse the stashed operand one final time and release the slot (`CollectorOp.LASTUSE`).
    LastUse = 1,
    /// Read from shared memory and keep the operand in the collector (`CollectorOp.FILL`).
    Fill = 2,
    /// Reuse the stashed operand and keep it in the collector (`CollectorOp.USE`).
    Use = 3,
}

/// Sealed trait restricting type-level collector markers to valid [`CollectorOp`] variants.
mod sealed_collector {
    /// Implemented only by [`super::CollectorDiscard`], [`super::CollectorLastUse`],
    /// [`super::CollectorFill`], and [`super::CollectorUse`].
    pub trait CollectorMarker {}
}

/// Compile-time [`CollectorOp`] marker carried in [`SM107BlockScaledMmaMXF8F6F4Op`]'s type parameters.
pub trait CollectorOpMarker: sealed_collector::CollectorMarker + Copy {
    /// Encoded [`CollectorOp`] discriminant (`0..=3`) read by the compiler.
    const OP: CollectorOp;
}

/// Type-level marker for [`CollectorOp::Discard`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CollectorDiscard;
/// Type-level marker for [`CollectorOp::LastUse`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CollectorLastUse;
/// Type-level marker for [`CollectorOp::Fill`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CollectorFill;
/// Type-level marker for [`CollectorOp::Use`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CollectorUse;

impl sealed_collector::CollectorMarker for CollectorDiscard {}
impl CollectorOpMarker for CollectorDiscard {
    const OP: CollectorOp = CollectorOp::Discard;
}
impl sealed_collector::CollectorMarker for CollectorLastUse {}
impl CollectorOpMarker for CollectorLastUse {
    const OP: CollectorOp = CollectorOp::LastUse;
}
impl sealed_collector::CollectorMarker for CollectorFill {}
impl CollectorOpMarker for CollectorFill {
    const OP: CollectorOp = CollectorOp::Fill;
}
impl sealed_collector::CollectorMarker for CollectorUse {}
impl CollectorOpMarker for CollectorUse {
    const OP: CollectorOp = CollectorOp::Use;
}

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
#[derive(Debug)]
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

impl<AL, BL, const M: u32, const K: u32> Copy for Sm100BlockScaledTiledMma<AL, BL, M, K> {}
impl<AL, BL, const M: u32, const K: u32> Clone for Sm100BlockScaledTiledMma<AL, BL, M, K> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
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

/// Independent shared/shared FP16 or BF16 MMA with FP32 accumulation (`tcgen05.MmaF16BF16Op`).
///
/// For `GROUP = 1`, `M` is `64` or `128`; for `GROUP = 2`, `M` is `128` or `256`.
/// `N` is a multiple of 16 in `16..=256`; `K` is a positive multiple of 16.
/// M/N describe the complete product across the CTA group. Each CTA's layouts
/// must cover A `(M / GROUP, K)` and B `(N / GROUP, K)` and satisfy [`SharedLayout`](super::SharedLayout).
pub struct MmaSs<AL, BL, const M: u32, const N: u32, const K: u32, const GROUP: u32, T = f16>(
    #[doc = "Carries both per-CTA shared operand layouts and the 16-bit element type."]
    PhantomData<(AL, BL, T)>,
);
impl<
    AL: SmemLayout,
    BL: SmemLayout,
    const M: u32,
    const N: u32,
    const K: u32,
    const G: u32,
    T: MmaHalfElement,
> MmaSs<AL, BL, M, N, K, G, T>
{
    /// Check group size, collective dimensions and per-CTA shared operand shapes.
    const VALID: () = {
        let () = AL::VALID;
        let () = BL::VALID;
        assert!(G == 1 || G == 2);
        assert!((G == 1 && (M == 64 || M == 128)) || (G == 2 && (M == 128 || M == 256)));
        assert!(N >= 16 && N <= 256 && N.is_multiple_of(16));
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
        a: &SharedTile<AL, OperandA, T>,
        b: &SharedTile<BL, OperandB, T>,
        accumulate: bool,
    ) {
        unsafe {
            __compiler::mma_ss::<AL, BL, M, N, K, G, T>(
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
impl<
    AL: SmemLayout,
    BL: SmemLayout,
    const M: u32,
    const N: u32,
    const K: u32,
    const G: u32,
    T: MmaHalfElement,
> Default for MmaSs<AL, BL, M, N, K, G, T>
{
    fn default() -> Self {
        Self::new()
    }
}

/// Vera Rubin (`sm_107a`) `MXFP8` (`Float8E4M3FN` + `Float8E8M0FNU`, `SF_VEC = 32`, `K = 64`)
/// block-scaled `tcgen05` MMA atom (`SM107BlockScaledMmaMXF8F6F4Op`).
///
/// One `K = 128` TMA stage holds two `K = 64` MMA steps (`sub_k = 0` and `1`) and one 512-byte
/// `UE8M0` scale atom per operand. Set `BCollector` to [`CollectorFill`] and [`CollectorLastUse`]
/// to reuse one `B` tile across two `M`-halves without re-reading shared memory.
#[derive(Debug)]
pub struct SM107BlockScaledMmaMXF8F6F4Op<
    AL,
    BL,
    const M: u32 = 128,
    const N: u32 = 128,
    const GROUP: u32 = 1,
    BCollector = CollectorDiscard,
>(
    #[doc = "Zero-byte marker carrying per-CTA shared layouts and the B-collector policy."]
    PhantomData<(AL, BL, BCollector)>,
);

impl<AL, BL, const M: u32, const N: u32, const GROUP: u32, BCollector> Copy
    for SM107BlockScaledMmaMXF8F6F4Op<AL, BL, M, N, GROUP, BCollector>
{
}
impl<AL, BL, const M: u32, const N: u32, const GROUP: u32, BCollector> Clone
    for SM107BlockScaledMmaMXF8F6F4Op<AL, BL, M, N, GROUP, BCollector>
{
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<
    AL: ReifySmem2D,
    BL: ReifySmem2D,
    const M: u32,
    const N: u32,
    const GROUP: u32,
    BCollector: CollectorOpMarker,
> SM107BlockScaledMmaMXF8F6F4Op<AL, BL, M, N, GROUP, BCollector>
{
    /// Compile-time verification of SM107 `MXF8F6F4` (`K = 64` per MMA, `K = 128` per TMA/scale stage)
    /// dimensions and 128-byte swizzled (`Swizzle<3, 4, 3>`) `Float8E4M3FN` shared tile shapes.
    const VALID: () = {
        assert!(GROUP == 1 || GROUP == 2);
        assert!((GROUP == 1 && M == 128) || (GROUP == 2 && M == 256));
        assert!(N >= 16 && N <= 256 && N.is_multiple_of(16));
        assert!(AL::SHAPE.0 == (M / GROUP) as i64 && AL::SHAPE.1 == 128);
        assert!(BL::SHAPE.0 == N as i64 / GROUP as i64 && BL::SHAPE.1 == 128);
        assert!(AL::NUM_BITS == 3 && AL::NUM_BASE == 4 && AL::NUM_SHIFT == 3 && AL::OFFSET == 0);
        assert!(BL::NUM_BITS == 3 && BL::NUM_BASE == 4 && BL::NUM_SHIFT == 3 && BL::OFFSET == 0);
    };

    /// Validate the static SM107 `MXF8F6F4` MMA configuration at compile time.
    #[must_use]
    #[inline(always)]
    pub const fn new() -> Self {
        let () = Self::VALID;
        Self(PhantomData)
    }

    /// Copy one 512-byte `(32, 4, 4)` `Float8E8M0FNU` scale atom (`K = 128` at `SF_VEC = 32`) at
    /// `step` from shared memory into 4 TMEM columns at `tmem_col` via `tcgen05.cp.32x128b.warpx4`.
    ///
    /// # Safety
    /// One elected lane in the leader CTA must issue this operation after `s_scale` is ready in shared memory.
    #[inline(always)]
    pub unsafe fn copy<Role>(tmem_col: u32, s_scale: &Sm100ScaleTile<Role>, step: usize) {
        unsafe {
            __compiler::sm107_block_scaled_copy::<GROUP>(tmem_col, s_scale.base, step as u32);
        }
    }

    /// Issue one `K = 64` sub-step (`sub_k` in `0..2` within the `K = 128` stage) of an SM107
    /// `tcgen05.mma.kind::mxf8f6f4.block_scale` instruction (`CuTeDSL cute.gemm`).
    ///
    /// # Safety
    /// One elected lane in the leader CTA must issue this operation on `sm_107a` after `a`, `b`,
    /// and the TMEM scales at `sfa_tmem` and `sfb_tmem` are ready.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub unsafe fn gemm(
        &self,
        tmem_d: u32,
        a: &Sm100SharedTile<AL, OperandA>,
        b: &Sm100SharedTile<BL, OperandB>,
        sfa_tmem: u32,
        sfb_tmem: u32,
        sub_k: usize,
        accumulate: bool,
    ) {
        let _ = (Mkl, Nkl);
        unsafe {
            __compiler::sm107_block_scaled_gemm::<AL, BL, M, N, GROUP, BCollector>(
                tmem_d,
                a.base,
                b.base,
                sfa_tmem,
                sfb_tmem,
                sub_k as u32,
                accumulate,
            );
        }
    }
}

impl<
    AL: ReifySmem2D,
    BL: ReifySmem2D,
    const M: u32,
    const N: u32,
    const GROUP: u32,
    BCollector: CollectorOpMarker,
> Default for SM107BlockScaledMmaMXF8F6F4Op<AL, BL, M, N, GROUP, BCollector>
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
