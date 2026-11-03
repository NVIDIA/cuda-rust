/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp MMA slices and shared-memory fragment loading.

use super::{__compiler, Mxf4BTileK64, Mxf4ScaleStage, Mxf4ScaleTile128};
use crate::block_scaled::{
    Mkl, MmaScalePair, MmaScaleQuad, Nkl, ScalePack4, SharedNvfp4ScaleAtom, SharedScaleAtom,
    Sm120ScaleAtom,
};
use crate::copy::tiled::SharedTensor;
use crate::markers::ColMajor;
use crate::mma::{FragmentA, FragmentB, FragmentC, load_matrix_a};
use crate::numeric::{Mxf4E2M1, UE8M0x4};
use core::marker::PhantomData;

// Map `(warp M position, first-or-second M block)` to a 16-row block number.
#[inline(always)]
/// Map a warp M partition and one of its two row bands to an A atom index.
pub(super) const fn a_atom_128(warp_m: usize, atom_m: usize) -> usize {
    warp_m + 4 * atom_m
}

// Map `(warp N position, B pair number)` to a 16-row block number in N x K B.
#[inline(always)]
/// Map a warp N partition and B-pair slot to the first N8 atom index.
pub(super) const fn b_pair_atom_128(warp_n: usize, pair_n: usize) -> usize {
    warp_n + 2 * pair_n
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
pub struct Mxf4TiledMma<SharedLayout> {
    /// Calling thread’s lane index in the 32-thread warp.
    pub(super) lane: u32,
    /// Retains the shared-memory coordinate map for fragment loads.
    pub(super) layout: PhantomData<SharedLayout>,
}

impl<SharedLayout> Mxf4TiledMma<SharedLayout> {
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
    /// [`Mxf4AccumulatorTile2x8::gemm`](super::Mxf4AccumulatorTile2x8::gemm). `warp_n` must be 0 or 1;
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
    ) -> FragmentA {
        unsafe { load_matrix_a::<SharedLayout, f16>(source.carrier(), warp_tile, self.lane) }
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
    ) -> (FragmentB, FragmentB) {
        let loaded =
            unsafe { load_matrix_a::<SharedLayout, f16>(source.carrier(), warp_tile, self.lane) };
        (
            FragmentB([loaded.0[0], loaded.0[2]]),
            FragmentB([loaded.0[1], loaded.0[3]]),
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
    ) -> (FragmentA, FragmentA) {
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
    ) -> (FragmentB, FragmentB) {
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
    /// TMA copies the bytes without rearranging them. [`Sm120ScaleAtom`](crate::block_scaled::Sm120ScaleAtom)
    /// converts each logical row into the physical word position required by
    /// the GPU. The ten words are reused by both K=64 steps.
    ///
    /// # Safety
    ///
    /// `a` and `b` must point to initialized, `u32`-aligned shared-memory
    /// blocks of [`Sm120ScaleAtom::BYTES`](crate::block_scaled::Sm120ScaleAtom::BYTES) bytes. `warp_m` must be 0 through
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
        a: FragmentA,
        scale_a: MmaScalePair<Mkl>,
        b: FragmentB,
        scale_b: MmaScalePair<Nkl>,
        c: FragmentC,
    ) -> FragmentC {
        let _ = (self, a, scale_a, b, scale_b, c);
        unreachable!("cute-rs block-scaled MMA executed outside recognized CuTe device compilation")
    }

    /// Load this lane's `Float8E4M3FN` (`SF_VEC = 16`) scale quad for a 16-row A tile (`tile_m`).
    ///
    /// Maps `lane` through the SM120 `m16n8k64` SFA thread-value layout (`q + 8 * (r & 1)`)
    /// and [`Sm120ScaleAtom::word_offset`].
    ///
    /// # Safety
    /// `atom` must point to an initialized 512-byte canonical scale atom in shared
    /// or global memory, and `tile_m * 16 + 15 < 128`.
    #[must_use]
    #[inline(always)]
    pub unsafe fn load_nvfp4_scale_a(
        &self,
        atom_words: *const u32,
        tile_m: usize,
    ) -> MmaScaleQuad<Mkl> {
        let q = (self.lane >> 2) as usize;
        let r = (self.lane & 3) as usize;
        let row = (tile_m * 16 + q + 8 * (r & 1)) & 127;
        let bits = unsafe { *atom_words.add(Sm120ScaleAtom::word_offset(row)) };
        MmaScaleQuad::from_bits(bits)
    }

    /// Load this lane's `Float8E4M3FN` (`SF_VEC = 16`) scale quad for an 8-row B tile (`tile_n`).
    ///
    /// Maps `lane` through the SM120 `m16n8k64` SFB thread-value layout (`q = lane / 4`)
    /// and [`Sm120ScaleAtom::word_offset`].
    ///
    /// # Safety
    /// `atom_words` must point to an initialized 512-byte canonical scale atom in
    /// shared or global memory, and `tile_n * 8 + 7 < 128`.
    #[must_use]
    #[inline(always)]
    pub unsafe fn load_nvfp4_scale_b(
        &self,
        atom_words: *const u32,
        tile_n: usize,
    ) -> MmaScaleQuad<Nkl> {
        let q = (self.lane >> 2) as usize;
        let row = (tile_n * 8 + q) & 127;
        let bits = unsafe { *atom_words.add(Sm120ScaleAtom::word_offset(row)) };
        MmaScaleQuad::from_bits(bits)
    }

    /// Load this lane's `Float8E4M3FN` (`SF_VEC = 16`) A scale quad from a shared scale atom.
    ///
    /// # Safety
    /// `a` must be an initialized 512-byte [`SharedNvfp4ScaleAtom`], and `tile_m < 8`.
    #[must_use]
    #[inline(always)]
    pub unsafe fn load_shared_nvfp4_scale_a(
        &self,
        a: &SharedNvfp4ScaleAtom<Mkl>,
        tile_m: usize,
    ) -> MmaScaleQuad<Mkl> {
        unsafe { self.load_nvfp4_scale_a(a.carrier().base.cast_const(), tile_m) }
    }

    /// Load this lane's `Float8E4M3FN` (`SF_VEC = 16`) B scale quad from a shared scale atom.
    ///
    /// # Safety
    /// `b` must be an initialized 512-byte [`SharedNvfp4ScaleAtom`], and `tile_n < 16`.
    #[must_use]
    #[inline(always)]
    pub unsafe fn load_shared_nvfp4_scale_b(
        &self,
        b: &SharedNvfp4ScaleAtom<Nkl>,
        tile_n: usize,
    ) -> MmaScaleQuad<Nkl> {
        unsafe { self.load_nvfp4_scale_b(b.carrier().base.cast_const(), tile_n) }
    }

    /// Run one 16 x 8 x 64 NVFP4 (`Float4E2M1FN` + `Float8E4M3FN`, `SF_VEC = 16`) Tensor-Core MMA.
    ///
    /// Emits `cute_nvgpu.arch.mma.SM120.block_scaled` (`m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::4X`).
    ///
    /// # Safety
    /// All 32 lanes of the warp must execute this call convergently on SM120a with
    /// matching `m16n8k64` fragments and per-lane `E4M3x4` scale quads.
    #[must_use]
    #[inline(always)]
    pub unsafe fn gemm_nvfp4(
        &self,
        a: FragmentA,
        scale_a: MmaScaleQuad<Mkl>,
        b: FragmentB,
        scale_b: MmaScaleQuad<Nkl>,
        c: FragmentC,
    ) -> FragmentC {
        let _ = self.lane;
        unsafe { __compiler::nvfp4_mma_atom_call(c, a, b, scale_a, scale_b) }
    }
}

/// One lane's handle to the SM120 NVFP4 (`Float4E2M1FN` + `Float8E4M3FN`, `SF_VEC = 16`) tensor-core operation.
///
/// Pairs warp-cooperative `ldmatrix.x4` shared-memory loads with the SM120
/// `m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::4X` tensor-core instruction.
#[derive(Clone, Copy)]
pub struct Nvfp4TiledMma<SharedLayout> {
    /// Calling thread's hardware lane index in `0..32`.
    lane: u32,
    /// Retains the shared-memory coordinate map for fragment loads.
    layout: PhantomData<SharedLayout>,
}

impl<SharedLayout> Nvfp4TiledMma<SharedLayout> {
    /// Create the handle for one lane (`0..32`) of the warp.
    #[must_use]
    #[inline(always)]
    pub const fn get_slice(lane: u32) -> Self {
        Self {
            lane,
            layout: PhantomData,
        }
    }

    /// Load this lane's part of a 16 x 64 FP4 A block (`16 x 16` `f16` carriers) using `ldmatrix.x4`.
    ///
    /// # Safety
    /// All 32 lanes of the warp must call this convergently with their hardware
    /// lane index, and `warp_tile` must select an initialized, 16-byte-aligned
    /// 16x16 `f16` window inside `source`.
    #[must_use]
    #[inline(always)]
    pub unsafe fn load_a<T: crate::mma::LdmatrixElement>(
        &self,
        source: &crate::copy::cooperative::SmemTile<T, SharedLayout>,
        warp_tile: (usize, usize),
    ) -> FragmentA {
        unsafe { load_matrix_a::<SharedLayout, T>(source, warp_tile, self.lane) }
    }

    /// Load two neighboring 8 x 64 FP4 B fragments (`16 x 16` 16-bit carriers) using `ldmatrix.x4`.
    ///
    /// # Safety
    /// Same collective participation and alignment requirements as [`Self::load_a`].
    #[must_use]
    #[inline(always)]
    pub unsafe fn load_b_pair<T: crate::mma::LdmatrixElement>(
        &self,
        source: &crate::copy::cooperative::SmemTile<T, SharedLayout>,
        warp_tile: (usize, usize),
    ) -> (FragmentB, FragmentB) {
        let loaded = unsafe { load_matrix_a::<SharedLayout, T>(source, warp_tile, self.lane) };
        (
            FragmentB([loaded.0[0], loaded.0[2]]),
            FragmentB([loaded.0[1], loaded.0[3]]),
        )
    }

    /// Load this lane's `Float8E4M3FN` (`SF_VEC = 16`) scale quad for a 16-row A tile (`tile_m`).
    ///
    /// # Safety
    /// `atom_words` must point to an initialized 512-byte canonical scale atom.
    #[must_use]
    #[inline(always)]
    pub unsafe fn load_scale_a(&self, atom_words: *const u32, tile_m: usize) -> MmaScaleQuad<Mkl> {
        let q = (self.lane >> 2) as usize;
        let r = (self.lane & 3) as usize;
        let row = (tile_m * 16 + q + 8 * (r & 1)) & 127;
        let bits = unsafe { *atom_words.add(Sm120ScaleAtom::word_offset(row)) };
        MmaScaleQuad::from_bits(bits)
    }

    /// Load this lane's `Float8E4M3FN` (`SF_VEC = 16`) scale quad for an 8-row B tile (`tile_n`).
    ///
    /// # Safety
    /// `atom_words` must point to an initialized 512-byte canonical scale atom.
    #[must_use]
    #[inline(always)]
    pub unsafe fn load_scale_b(&self, atom_words: *const u32, tile_n: usize) -> MmaScaleQuad<Nkl> {
        let q = (self.lane >> 2) as usize;
        let row = (tile_n * 8 + q) & 127;
        let bits = unsafe { *atom_words.add(Sm120ScaleAtom::word_offset(row)) };
        MmaScaleQuad::from_bits(bits)
    }

    /// Run one 16 x 8 x 64 NVFP4 (`Float4E2M1FN` + `Float8E4M3FN`, `SF_VEC = 16`) Tensor-Core MMA.
    ///
    /// # Safety
    /// All 32 lanes of the warp must execute this call convergently on SM120a with
    /// matching `m16n8k64` fragments and per-lane `E4M3x4` scale quads.
    #[must_use]
    #[inline(always)]
    pub unsafe fn gemm(
        &self,
        a: FragmentA,
        scale_a: MmaScaleQuad<Mkl>,
        b: FragmentB,
        scale_b: MmaScaleQuad<Nkl>,
        c: FragmentC,
    ) -> FragmentC {
        let _ = self.lane;
        unsafe { __compiler::nvfp4_mma_atom_call(c, a, b, scale_a, scale_b) }
    }
}

/// CuTeDSL-aligned alias (`MmaMXF4NVF4Op`) for [`Nvfp4TiledMma`].
pub type Mxf4Nvf4TiledMma<SharedLayout> = Nvfp4TiledMma<SharedLayout>;
