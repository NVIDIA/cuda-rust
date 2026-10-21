/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Exact compiler boundaries for SM120 block-scaled MMA.

use super::tiled::a_atom_128;
use super::*;
use crate::block_scaled::{Mkl, Nkl, ScalePack4, SharedScaleAtom, Sm120ScaleAtom};
use crate::copy::tiled::SharedTensor;
use crate::epilogue::sm120::Sm120EpilogueWarp128x128;
use crate::mma::{AccC, FragA};
use crate::numeric::Mxf4E2M1;
use core::marker::PhantomData;

/// Compiler boundary assigning a lane to its typed shared-layout MMA view.
#[inline(never)]
pub const fn tiled_mma_slice<SharedLayout>(lane: u32) -> Mxfp4TiledMma<SharedLayout> {
    Mxfp4TiledMma {
        lane,
        layout: PhantomData,
    }
}

/// Compiler boundary filling all 16 per-lane accumulator fragments with one FP32 value.
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

/// Compiler boundary storing both row bands and all eight N fragments.
///
/// # Safety
/// The collective participation, shared storage and publication contracts of the epilogue slice apply.
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

/// Compiler boundary selecting the scale pairs for one K64 half.
///
/// # Safety
/// The half index must be zero or one.
#[inline(never)]
pub const unsafe fn fragment_slice_k(scales: Mxf4ScaleTile128, half: usize) -> Mxf4ScalePairs128 {
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

/// Compiler boundary loading this lane’s two A and eight B scale packs.
///
/// # Safety
/// Shared scale atoms must be initialized and live; lane and warp coordinates must match the consuming MMA.
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

/// Compiler boundary loading the warp’s two A fragments for one K64 half.
///
/// # Safety
/// Shared storage, lane participation and coordinates must satisfy Mxfp4TiledMma::load_a.
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

/// Compiler boundary borrowing a lazy B view without loading its eight fragments.
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

/// Compiler boundary issuing the four B-pair steps of a two-band, eight-column fragment tile.
///
/// # Safety
/// All lanes must supply matching live A/B fragments and scales under the public tiled-MMA collective contract.
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
