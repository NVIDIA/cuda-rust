/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Block-scaled MXFP4 matrix multiply for SM120 GPUs.
//!
//! This module covers the compute part of a 128 x 128 output tile:
//!
//! ```text
//! packed FP4 A + packed FP4 B + 8-bit scales
//!                       |
//!                       v
//!                shared memory
//!                       |
//!                       v
//!          fragments held by each GPU lane
//!                       |
//!                       v
//!       m16n8k64 tensor-core instructions
//!                       |
//!                       v
//!             2 x 8 FP32 result cells
//! ```
//!
//! Words used below:
//!
//! - **M** and **N** are the output row and column dimensions. **K** is the
//!   shared dimension whose products are summed.
//! - A **warp** is 32 GPU threads running one instruction together.
//! - A **lane** is one thread inside that warp.
//! - A **fragment** is the small part of a matrix held by one lane.
//! - **MMA** means matrix multiply-accumulate: `result = A * B + result`.
//! - An **MMA cell** is one 16 x 8 result made by a tensor-core instruction.
//! - A **scale pair** contains two scales: one for each K=32 half of K=64.
//! - A **stage** is one K=128 input block held in shared memory.
//! - **TMA** is the GPU copy engine used to move whole tiles.
//! - **MXFP4** stores 4-bit E2M1 numbers with an 8-bit UE8M0 scale for each
//!   group of 32 numbers.
//!
//! These Rust types add checks and names. CuTe-aware compilers retain the
//! shared-memory MMA story and choose its implementation for their backend.

use core::marker::PhantomData;

use crate::block_scaled::{Mkl, MmaScalePair, Nkl, ScalePack4, SharedScaleAtom, Sm120ScaleAtom};
use crate::epilogue::Sm120EpilogueWarp128x128;
use crate::markers::ColMajor;
use crate::mma::{AccC, FragA, FragB, load_matrix_a};
use crate::numeric::{Mxf4E2M1, UE8M0x4};
use crate::tiled_copy::SharedTensor;

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
    a: ScalePack4<Mkl>,
    b0: ScalePack4<Nkl>,
    b1: ScalePack4<Nkl>,
    b2: ScalePack4<Nkl>,
    b3: ScalePack4<Nkl>,
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
    a0: ScalePack4<Mkl>,
    a1: ScalePack4<Mkl>,
    b0: ScalePack4<Nkl>,
    b1: ScalePack4<Nkl>,
    b2: ScalePack4<Nkl>,
    b3: ScalePack4<Nkl>,
    b4: ScalePack4<Nkl>,
    b5: ScalePack4<Nkl>,
    b6: ScalePack4<Nkl>,
    b7: ScalePack4<Nkl>,
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
    c00: AccC,
    c01: AccC,
    c02: AccC,
    c03: AccC,
    c04: AccC,
    c05: AccC,
    c06: AccC,
    c07: AccC,
    c10: AccC,
    c11: AccC,
    c12: AccC,
    c13: AccC,
    c14: AccC,
    c15: AccC,
    c16: AccC,
    c17: AccC,
}

/// One warp's eight B fragments for one K=64 step.
///
/// This is a view, so creating it does not load all eight fragments. During
/// [`Mxf4AccumulatorTile2x8::accumulate_k64`], it loads two at a time:
///
/// ```text
/// load b0,b1 -> compute -> load b2,b3 -> compute -> ...
/// ```
///
/// That keeps fewer temporary values in registers.
pub struct Mxf4BTileK64<'a, SharedLayout> {
    base: *mut f16,
    capacity: usize,
    warp_n: usize,
    k_half: usize,
    source: PhantomData<&'a SharedTensor<Mxf4E2M1, f16, SharedLayout, Nkl>>,
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
    unsafe fn load<const PAIR: usize>(&self, mma: &Mxfp4TiledMma<SharedLayout>) -> (FragB, FragB) {
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
    /// # Safety
    ///
    /// All 32 lanes must call this together. `mma` must contain each calling
    /// thread's real lane number. The A fragments, B view, and scales must use
    /// the same warp position and K=64 half from one completed shared-memory
    /// stage. The B view's `warp_n` and `k_half` must each be 0 or 1, and that
    /// stage must not be overwritten during this call.
    #[inline(always)]
    pub unsafe fn accumulate_k64<SharedLayout>(
        &mut self,
        mma: &Mxfp4TiledMma<SharedLayout>,
        a: (FragA, FragA),
        b: Mxf4BTileK64<'_, SharedLayout>,
        scales: Mxf4ScalePairs128,
    ) {
        unsafe { __compiler::tiled_gemm(self, mma, a, b, scales) };
    }
}

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

// Map `(warp M position, first-or-second M block)` to a 16-row block number.
#[inline(always)]
const fn a_atom_128(warp_m: usize, atom_m: usize) -> usize {
    warp_m + 4 * atom_m
}

// Map `(warp N position, B pair number)` to a 16-row block number in N x K B.
#[inline(always)]
const fn b_pair_atom_128(warp_n: usize, pair_n: usize) -> usize {
    warp_n + 2 * pair_n
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

/// One lane's handle to the SM120 MXFP4 tensor-core operation.
///
/// All 32 lanes create this handle with their own lane number. They then use
/// it together to load fragments and run `m16n8k64`:
///
/// ```text
/// 32 lane handles -> 32 lanes load together -> one warp-wide MMA
/// ```
#[derive(Clone, Copy)]
pub struct Mxfp4TiledMma<SharedLayout> {
    lane: u32,
    layout: PhantomData<SharedLayout>,
}

impl<SharedLayout> Mxfp4TiledMma<SharedLayout> {
    /// Create the handle for one lane of the warp.
    ///
    /// The later load methods require `lane` to be the calling thread's real
    /// lane number, from 0 through 31.
    #[must_use]
    #[inline(always)]
    pub const fn get_slice(lane: u32) -> Self {
        __compiler::tiled_mma_slice(lane)
    }

    /// Create a view of the eight B fragments for one K=64 step.
    ///
    /// ```text
    /// shared B tile + warp N position + K half
    ///                      |
    ///                      v
    ///              eight-fragment view
    /// ```
    ///
    /// The view loads two fragments at a time when it is passed to
    /// [`Mxf4AccumulatorTile2x8::accumulate_k64`]. `warp_n` must be 0 or 1;
    /// `k_half` must be 0 or 1 before that unsafe operation is called.
    #[must_use]
    #[inline(always)]
    pub const fn get_b_tile_k64<'a>(
        &self,
        source: &'a SharedTensor<Mxf4E2M1, f16, SharedLayout, Nkl>,
        warp_n: usize,
        k_half: usize,
    ) -> Mxf4BTileK64<'a, SharedLayout> {
        __compiler::mma_partition_b(self, source, warp_n, k_half)
    }

    /// Load this lane's part of a 16 x 64 FP4 A block.
    ///
    /// Four FP4 values use one 16-bit storage value, so the 16 x 64 FP4 block
    /// occupies the same bytes as a 16 x 16 `f16` block.
    ///
    /// # Safety
    ///
    /// All 32 lanes must call this together. This handle must contain the
    /// calling thread's lane number. `warp_tile` must select an initialized,
    /// correctly aligned block inside `source`.
    #[must_use]
    #[inline(always)]
    pub unsafe fn load_a(
        &self,
        source: &SharedTensor<Mxf4E2M1, f16, SharedLayout, Mkl>,
        warp_tile: (usize, usize),
    ) -> FragA {
        unsafe { load_matrix_a::<SharedLayout>(source.carrier(), warp_tile, self.lane) }
    }

    /// Load two neighboring B fragments.
    ///
    /// B is stored as `N x K`, with K next to K in memory:
    ///
    /// ```text
    /// one 16-row B load -> fragment for 8 columns + fragment for 8 columns
    /// ```
    ///
    /// This lets A and B use the same `ldmatrix.x4` load.
    ///
    /// # Safety
    ///
    /// The same rules as [`Self::load_a`] apply.
    #[must_use]
    #[inline(always)]
    pub unsafe fn load_b_pair(
        &self,
        source: &SharedTensor<Mxf4E2M1, f16, SharedLayout, Nkl>,
        warp_tile: (usize, usize),
    ) -> (FragB, FragB) {
        let loaded =
            unsafe { load_matrix_a::<SharedLayout>(source.carrier(), warp_tile, self.lane) };
        (
            FragB([loaded.0[0], loaded.0[2]]),
            FragB([loaded.0[1], loaded.0[3]]),
        )
    }

    /// Load the two A fragments owned by this warp.
    ///
    /// The four M warp positions receive these row blocks:
    ///
    /// ```text
    /// warp_m   first block   second block
    ///    0          0             64
    ///    1         16             80
    ///    2         32             96
    ///    3         48            112
    /// ```
    ///
    /// # Safety
    ///
    /// The rules from [`Self::load_a`] apply. `warp_m` must be 0 through 3,
    /// `k_half` must be 0 or 1, and `source` must contain the complete staged
    /// 128 x 128 FP4 tile.
    #[must_use]
    #[inline(always)]
    pub unsafe fn load_a_128(
        &self,
        source: &SharedTensor<Mxf4E2M1, f16, SharedLayout, Mkl>,
        warp_m: usize,
        k_half: usize,
    ) -> (FragA, FragA) {
        unsafe { __compiler::mma_load_a(self, source, warp_m, k_half) }
    }

    /// Load one of this warp's four B-fragment pairs.
    ///
    /// Each pair gives two neighboring 8-column fragments:
    ///
    /// ```text
    /// pair_n       0       1       2        3
    /// base col     0      32      64       96
    /// plus                 16 * warp_n
    /// ```
    ///
    /// # Safety
    ///
    /// The rules from [`Self::load_b_pair`] apply. `warp_n` must be 0 or 1,
    /// `pair_n` must be 0 through 3, and `k_half` must be 0 or 1. `source`
    /// must contain the complete staged 128 x 128 FP4 tile.
    #[must_use]
    #[inline(always)]
    pub unsafe fn load_b_pair_128(
        &self,
        source: &SharedTensor<Mxf4E2M1, f16, SharedLayout, Nkl>,
        warp_n: usize,
        pair_n: usize,
        k_half: usize,
    ) -> (FragB, FragB) {
        unsafe { self.load_b_pair(source, (b_pair_atom_128(warp_n, pair_n), k_half)) }
    }

    /// Load five packed scale words for a 64 x 32 warp tile.
    ///
    /// One word is for A. Four words are for B. Each word contains four
    /// 8-bit scales, so both K=64 steps reuse these loads.
    ///
    /// # Safety
    ///
    /// `ROWS` must be positive. Both shared tensors must contain the requested
    /// `word`. `warp_m` and `warp_n` must select blocks inside those tensors.
    /// This handle must contain the calling thread's real lane number.
    #[must_use]
    #[inline(always)]
    pub unsafe fn load_scale_stage<const ROWS: i64>(
        &self,
        a: &SharedTensor<UE8M0x4, u32, ColMajor<ROWS, 4>, Mkl>,
        b: &SharedTensor<UE8M0x4, u32, ColMajor<ROWS, 4>, Nkl>,
        word: usize,
        warp_m: usize,
        warp_n: usize,
    ) -> Mxf4ScaleStage {
        const { assert!(ROWS > 0) };
        let rows = ROWS as usize;
        let q = (self.lane >> 2) as usize;
        let r = (self.lane & 3) as usize;
        let a_row = 16 * warp_m + q + 8 * (r & 1);
        let b_row = 32 * warp_n + q;
        let column = word * rows;
        let a_base = a.carrier().base.cast_const();
        let b_base = b.carrier().base.cast_const();

        unsafe {
            Mxf4ScaleStage {
                a: ScalePack4::from_bits(*a_base.add(column + a_row)),
                b0: ScalePack4::from_bits(*b_base.add(column + b_row)),
                b1: ScalePack4::from_bits(*b_base.add(column + b_row + 8)),
                b2: ScalePack4::from_bits(*b_base.add(column + b_row + 16)),
                b3: ScalePack4::from_bits(*b_base.add(column + b_row + 24)),
            }
        }
    }

    /// Load the ten packed scale words for one warp's 128 x 128 tile part.
    ///
    /// The words match the same two-by-eight grid as the result cells:
    ///
    /// ```text
    /// A row blocks: 16*warp_m, 64 + 16*warp_m
    /// B pair starts: 16*warp_n + {0, 32, 64, 96}
    /// ```
    ///
    /// One word contains four K=32 scales, covering both K=64 steps. Every
    /// lane reads a valid word, even when the hardware does not use that
    /// lane's scale value.
    ///
    /// # Safety
    ///
    /// Both shared tensors must contain `word`. `ROWS` must be at least 128,
    /// `word` must be 0 or 1, `warp_m` must be 0 through 3, and `warp_n` must
    /// be 0 or 1. This handle must contain the calling thread's lane number.
    #[must_use]
    #[inline(always)]
    pub unsafe fn load_scale_tile_128<const ROWS: i64>(
        &self,
        a: &SharedTensor<UE8M0x4, u32, ColMajor<ROWS, 2>, Mkl>,
        b: &SharedTensor<UE8M0x4, u32, ColMajor<ROWS, 2>, Nkl>,
        word: usize,
        warp_m: usize,
        warp_n: usize,
    ) -> Mxf4ScaleTile128 {
        const { assert!(ROWS >= 128) };
        let rows = ROWS as usize;
        let q = (self.lane >> 2) as usize;
        let r = (self.lane & 3) as usize;
        let a_provider = q + 8 * (r & 1);
        let a_row0 = 16 * warp_m + a_provider;
        let a_row1 = 64 + 16 * warp_m + a_provider;
        let b_row0 = 16 * warp_n + q;
        let column = word * rows;
        let a_base = a.carrier().base.cast_const();
        let b_base = b.carrier().base.cast_const();

        unsafe {
            Mxf4ScaleTile128 {
                a0: ScalePack4::from_bits(*a_base.add(column + a_row0)),
                a1: ScalePack4::from_bits(*a_base.add(column + a_row1)),
                b0: ScalePack4::from_bits(*b_base.add(column + b_row0)),
                b1: ScalePack4::from_bits(*b_base.add(column + b_row0 + 8)),
                b2: ScalePack4::from_bits(*b_base.add(column + b_row0 + 32)),
                b3: ScalePack4::from_bits(*b_base.add(column + b_row0 + 40)),
                b4: ScalePack4::from_bits(*b_base.add(column + b_row0 + 64)),
                b5: ScalePack4::from_bits(*b_base.add(column + b_row0 + 72)),
                b6: ScalePack4::from_bits(*b_base.add(column + b_row0 + 96)),
                b7: ScalePack4::from_bits(*b_base.add(column + b_row0 + 104)),
            }
        }
    }

    /// Load the ten packed scale words from SM120's required scale layout.
    ///
    /// Each shared input is one 512-byte block:
    ///
    /// ```text
    /// 512-byte A scale block -> 2 packed words
    /// 512-byte B scale block -> 8 packed words
    /// ```
    ///
    /// TMA copies the bytes without rearranging them. [`Sm120ScaleAtom`]
    /// converts each logical row into the physical word position required by
    /// the GPU. The ten words are reused by both K=64 steps.
    ///
    /// # Safety
    ///
    /// `a` and `b` must point to initialized, `u32`-aligned shared-memory
    /// blocks of [`Sm120ScaleAtom::BYTES`] bytes. `warp_m` must be 0 through
    /// 3, `warp_n` must be 0 or 1, and this handle must contain the calling
    /// thread's real lane number.
    #[must_use]
    #[inline(always)]
    pub unsafe fn load_scale_atom_128(
        &self,
        a: &SharedScaleAtom<Mkl>,
        b: &SharedScaleAtom<Nkl>,
        warp_m: usize,
        warp_n: usize,
    ) -> Mxf4ScaleTile128 {
        unsafe { __compiler::mma_load_scales(self, a, b, warp_m, warp_n) }
    }

    /// Run one 16 x 8 x 64 block-scaled tensor-core operation.
    ///
    /// ```text
    /// result = previous result + (A * A scale) x (B * B scale)
    /// ```
    ///
    /// The A scale has type [`MmaScalePair<Mkl>`]. The B scale has type
    /// [`MmaScalePair<Nkl>`], so Rust rejects swapping them.
    ///
    /// # Safety
    ///
    /// All 32 lanes must call this together. `a`, `b`, and `c` must use the
    /// SM120 `m16n8k64` lane layout. Each scale pair must contain the first
    /// K=32 scale followed by the second K=32 scale.
    #[must_use]
    #[inline(always)]
    pub unsafe fn gemm(
        &self,
        a: FragA,
        scale_a: MmaScalePair<Mkl>,
        b: FragB,
        scale_b: MmaScalePair<Nkl>,
        c: AccC,
    ) -> AccC {
        let _ = (self, a, scale_a, b, scale_b, c);
        unreachable!("cute-rs block-scaled MMA executed outside recognized CuTe device compilation")
    }
}

/// Stable compiler boundaries for the SM120 shared-memory MMA path.
///
/// Public methods above keep the kernel readable. These exact calls let the
/// importer retain the same tensor, fragment, and tiled-MMA meaning whether
/// rustc inlines those wrappers or leaves them as calls.
#[doc(hidden)]
pub mod __compiler {
    use super::*;

    #[inline(never)]
    pub const fn tiled_mma_slice<SharedLayout>(lane: u32) -> Mxfp4TiledMma<SharedLayout> {
        Mxfp4TiledMma {
            lane,
            layout: PhantomData,
        }
    }

    #[inline(never)]
    pub const fn fragment_fill(fill: f32) -> Mxf4AccumulatorTile2x8 {
        let cell = AccC([fill; 4]);
        Mxf4AccumulatorTile2x8 {
            c00: cell,
            c01: cell,
            c02: cell,
            c03: cell,
            c04: cell,
            c05: cell,
            c06: cell,
            c07: cell,
            c10: cell,
            c11: cell,
            c12: cell,
            c13: cell,
            c14: cell,
            c15: cell,
            c16: cell,
            c17: cell,
        }
    }

    #[inline(never)]
    pub unsafe fn epilogue_store_fragment(
        slice: Sm120EpilogueWarp128x128,
        tile: Mxf4AccumulatorTile2x8,
    ) {
        unsafe {
            slice.store_atom::<0, 0>(tile.c00);
            slice.store_atom::<0, 1>(tile.c01);
            slice.store_atom::<0, 2>(tile.c02);
            slice.store_atom::<0, 3>(tile.c03);
            slice.store_atom::<0, 4>(tile.c04);
            slice.store_atom::<0, 5>(tile.c05);
            slice.store_atom::<0, 6>(tile.c06);
            slice.store_atom::<0, 7>(tile.c07);
            slice.store_atom::<1, 0>(tile.c10);
            slice.store_atom::<1, 1>(tile.c11);
            slice.store_atom::<1, 2>(tile.c12);
            slice.store_atom::<1, 3>(tile.c13);
            slice.store_atom::<1, 4>(tile.c14);
            slice.store_atom::<1, 5>(tile.c15);
            slice.store_atom::<1, 6>(tile.c16);
            slice.store_atom::<1, 7>(tile.c17);
        }
    }

    #[inline(never)]
    pub const unsafe fn fragment_slice_k(
        scales: Mxf4ScaleTile128,
        half: usize,
    ) -> Mxf4ScalePairs128 {
        Mxf4ScalePairs128 {
            a0: unsafe { scales.a0.pair_at_unchecked(half) },
            a1: unsafe { scales.a1.pair_at_unchecked(half) },
            b0: unsafe { scales.b0.pair_at_unchecked(half) },
            b1: unsafe { scales.b1.pair_at_unchecked(half) },
            b2: unsafe { scales.b2.pair_at_unchecked(half) },
            b3: unsafe { scales.b3.pair_at_unchecked(half) },
            b4: unsafe { scales.b4.pair_at_unchecked(half) },
            b5: unsafe { scales.b5.pair_at_unchecked(half) },
            b6: unsafe { scales.b6.pair_at_unchecked(half) },
            b7: unsafe { scales.b7.pair_at_unchecked(half) },
        }
    }

    #[inline(never)]
    pub unsafe fn mma_load_scales<SharedLayout>(
        mma: &Mxfp4TiledMma<SharedLayout>,
        a: &SharedScaleAtom<Mkl>,
        b: &SharedScaleAtom<Nkl>,
        warp_m: usize,
        warp_n: usize,
    ) -> Mxf4ScaleTile128 {
        let q = (mma.lane >> 2) as usize;
        let r = (mma.lane & 3) as usize;
        let a_provider = q + 8 * (r & 1);
        let a_row0 = 16 * warp_m + a_provider;
        let a_row1 = 64 + 16 * warp_m + a_provider;
        let b_row0 = 16 * warp_n + q;
        let a_base = a.carrier().base.cast_const();
        let b_base = b.carrier().base.cast_const();

        unsafe {
            Mxf4ScaleTile128 {
                a0: ScalePack4::from_bits(*a_base.add(Sm120ScaleAtom::word_offset(a_row0))),
                a1: ScalePack4::from_bits(*a_base.add(Sm120ScaleAtom::word_offset(a_row1))),
                b0: ScalePack4::from_bits(*b_base.add(Sm120ScaleAtom::word_offset(b_row0))),
                b1: ScalePack4::from_bits(*b_base.add(Sm120ScaleAtom::word_offset(b_row0 + 8))),
                b2: ScalePack4::from_bits(*b_base.add(Sm120ScaleAtom::word_offset(b_row0 + 32))),
                b3: ScalePack4::from_bits(*b_base.add(Sm120ScaleAtom::word_offset(b_row0 + 40))),
                b4: ScalePack4::from_bits(*b_base.add(Sm120ScaleAtom::word_offset(b_row0 + 64))),
                b5: ScalePack4::from_bits(*b_base.add(Sm120ScaleAtom::word_offset(b_row0 + 72))),
                b6: ScalePack4::from_bits(*b_base.add(Sm120ScaleAtom::word_offset(b_row0 + 96))),
                b7: ScalePack4::from_bits(*b_base.add(Sm120ScaleAtom::word_offset(b_row0 + 104))),
            }
        }
    }

    #[inline(never)]
    pub unsafe fn mma_load_a<SharedLayout>(
        mma: &Mxfp4TiledMma<SharedLayout>,
        source: &SharedTensor<Mxf4E2M1, f16, SharedLayout, Mkl>,
        warp_m: usize,
        k_half: usize,
    ) -> (FragA, FragA) {
        unsafe {
            (
                mma.load_a(source, (a_atom_128(warp_m, 0), k_half)),
                mma.load_a(source, (a_atom_128(warp_m, 1), k_half)),
            )
        }
    }

    #[inline(never)]
    pub const fn mma_partition_b<'a, SharedLayout>(
        _mma: &Mxfp4TiledMma<SharedLayout>,
        source: &'a SharedTensor<Mxf4E2M1, f16, SharedLayout, Nkl>,
        warp_n: usize,
        k_half: usize,
    ) -> Mxf4BTileK64<'a, SharedLayout> {
        Mxf4BTileK64 {
            base: source.carrier().base,
            capacity: source.carrier().capacity,
            warp_n,
            k_half,
            source: PhantomData,
        }
    }

    #[inline(never)]
    pub unsafe fn tiled_gemm<SharedLayout>(
        accumulators: &mut Mxf4AccumulatorTile2x8,
        mma: &Mxfp4TiledMma<SharedLayout>,
        a: (FragA, FragA),
        b: Mxf4BTileK64<'_, SharedLayout>,
        scales: Mxf4ScalePairs128,
    ) {
        let (a0, a1) = a;

        {
            let (b0, b1) = unsafe { b.load::<0>(mma) };
            accumulators.c00 = unsafe { mma.gemm(a0, scales.a0, b0, scales.b0, accumulators.c00) };
            accumulators.c01 = unsafe { mma.gemm(a0, scales.a0, b1, scales.b1, accumulators.c01) };
            accumulators.c10 = unsafe { mma.gemm(a1, scales.a1, b0, scales.b0, accumulators.c10) };
            accumulators.c11 = unsafe { mma.gemm(a1, scales.a1, b1, scales.b1, accumulators.c11) };
        }
        {
            let (b2, b3) = unsafe { b.load::<1>(mma) };
            accumulators.c02 = unsafe { mma.gemm(a0, scales.a0, b2, scales.b2, accumulators.c02) };
            accumulators.c03 = unsafe { mma.gemm(a0, scales.a0, b3, scales.b3, accumulators.c03) };
            accumulators.c12 = unsafe { mma.gemm(a1, scales.a1, b2, scales.b2, accumulators.c12) };
            accumulators.c13 = unsafe { mma.gemm(a1, scales.a1, b3, scales.b3, accumulators.c13) };
        }
        {
            let (b4, b5) = unsafe { b.load::<2>(mma) };
            accumulators.c04 = unsafe { mma.gemm(a0, scales.a0, b4, scales.b4, accumulators.c04) };
            accumulators.c05 = unsafe { mma.gemm(a0, scales.a0, b5, scales.b5, accumulators.c05) };
            accumulators.c14 = unsafe { mma.gemm(a1, scales.a1, b4, scales.b4, accumulators.c14) };
            accumulators.c15 = unsafe { mma.gemm(a1, scales.a1, b5, scales.b5, accumulators.c15) };
        }
        {
            let (b6, b7) = unsafe { b.load::<3>(mma) };
            accumulators.c06 = unsafe { mma.gemm(a0, scales.a0, b6, scales.b6, accumulators.c06) };
            accumulators.c07 = unsafe { mma.gemm(a0, scales.a0, b7, scales.b7, accumulators.c07) };
            accumulators.c16 = unsafe { mma.gemm(a1, scales.a1, b6, scales.b6, accumulators.c16) };
            accumulators.c17 = unsafe { mma.gemm(a1, scales.a1, b7, scales.b7, accumulators.c17) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Mxf4AccumulatorTile2x8, Mxf4BTileK64, Mxfp4TiledMma, a_atom_128, b_pair_atom_128};
    use crate::block_scaled::{Mkl, Nkl, SharedScaleAtom, Sm120ScaleAtom};
    use crate::mma::AccC;

    #[test]
    fn accumulator_and_lazy_b_keep_their_physical_carriers_explicit() {
        assert_eq!(
            core::mem::size_of::<Mxf4AccumulatorTile2x8>(),
            16 * core::mem::size_of::<AccC>()
        );
        assert_eq!(
            core::mem::size_of::<Mxf4BTileK64<'static, ()>>(),
            // shared base + capacity + warp-N + K-half
            4 * core::mem::size_of::<usize>()
        );
    }

    #[test]
    fn cta_128_fragment_permutation_matches_cute() {
        assert_eq!(
            [
                [16 * a_atom_128(0, 0), 16 * a_atom_128(0, 1)],
                [16 * a_atom_128(1, 0), 16 * a_atom_128(1, 1)],
                [16 * a_atom_128(2, 0), 16 * a_atom_128(2, 1)],
                [16 * a_atom_128(3, 0), 16 * a_atom_128(3, 1)],
            ],
            [[0, 64], [16, 80], [32, 96], [48, 112]]
        );
        assert_eq!(
            [
                [
                    16 * b_pair_atom_128(0, 0),
                    16 * b_pair_atom_128(0, 1),
                    16 * b_pair_atom_128(0, 2),
                    16 * b_pair_atom_128(0, 3),
                ],
                [
                    16 * b_pair_atom_128(1, 0),
                    16 * b_pair_atom_128(1, 1),
                    16 * b_pair_atom_128(1, 2),
                    16 * b_pair_atom_128(1, 3),
                ],
            ],
            [[0, 32, 64, 96], [16, 48, 80, 112]]
        );
    }

    #[test]
    fn canonical_scale_atom_loads_exact_two_a_and_eight_b_packs() {
        let mut a_words = core::array::from_fn::<_, { Sm120ScaleAtom::WORDS }, _>(|word| {
            0xa500_0000 | word as u32
        });
        let mut b_words = core::array::from_fn::<_, { Sm120ScaleAtom::WORDS }, _>(|word| {
            0xb500_0000 | word as u32
        });
        // Both arrays are aligned, contain one complete scale block, and stay
        // alive while the views are used.
        let a = unsafe {
            SharedScaleAtom::<Mkl>::from_raw_parts(a_words.as_mut_ptr(), Sm120ScaleAtom::WORDS)
        };
        // The separate B array follows the same rules.
        let b = unsafe {
            SharedScaleAtom::<Nkl>::from_raw_parts(b_words.as_mut_ptr(), Sm120ScaleAtom::WORDS)
        };

        // Lane 0 provides A rows 0/64 and B rows
        // 0,8,32,40,64,72,96,104 for warp (0,0).
        let lane0 = Mxfp4TiledMma::<()>::get_slice(0);
        // The arrays are valid and lane/warp positions are in range.
        let first = unsafe { lane0.load_scale_atom_128(&a, &b, 0, 0) };
        assert_eq!(
            [first.a0.bits(), first.a1.bits()],
            [0xa500_0000, 0xa500_0002]
        );
        assert_eq!(
            [
                first.b0.bits(),
                first.b1.bits(),
                first.b2.bits(),
                first.b3.bits(),
                first.b4.bits(),
                first.b5.bits(),
                first.b6.bits(),
                first.b7.bits(),
            ],
            [
                0xb500_0000,
                0xb500_0020,
                0xb500_0001,
                0xb500_0021,
                0xb500_0002,
                0xb500_0022,
                0xb500_0003,
                0xb500_0023,
            ]
        );

        // Lane 31 checks the last rows and words for warp position (3,1).
        let lane31 = Mxfp4TiledMma::<()>::get_slice(31);
        // The arrays are valid and lane/warp positions are in range.
        let last = unsafe { lane31.load_scale_atom_128(&a, &b, 3, 1) };
        assert_eq!([last.a0.bits(), last.a1.bits()], [0xa500_007d, 0xa500_007f]);
        assert_eq!(
            [
                last.b0.bits(),
                last.b1.bits(),
                last.b2.bits(),
                last.b3.bits(),
                last.b4.bits(),
                last.b5.bits(),
                last.b6.bits(),
                last.b7.bits(),
            ],
            [
                0xb500_005c,
                0xb500_007c,
                0xb500_005d,
                0xb500_007d,
                0xb500_005e,
                0xb500_007e,
                0xb500_005f,
                0xb500_007f,
            ]
        );
    }
}
