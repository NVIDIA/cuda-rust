/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Exact compiler markers for independent attention operations.

use crate::copy::tma::TensorMap;
use ctm_rs::mbarrier::Barrier;
/// Device-only shared/shared MMA marker; operand displacements count bytes from retained bases.
///
/// # Safety
/// The public MmaSs::gemm participation, completion and storage contracts apply.
#[inline(never)]
pub unsafe fn mma_ss<AL, BL, const M: u32, const N: u32, const K: u32, const G: u32>(
    dst: u32,
    a: *mut f16,
    b: *mut f16,
    accumulate: bool,
    a_byte_offset: u32,
    b_byte_offset: u32,
) {
    let _ = (dst, a, b, accumulate, a_byte_offset, b_byte_offset);
    unreachable!("CuTe compiler marker")
}
/// Device-only TMEM/shared MMA marker; B’s displacement counts bytes from its retained base.
///
/// # Safety
/// The public MmaTs::gemm participation, nonoverlap and completion contracts apply.
#[inline(never)]
pub unsafe fn mma_ts<BL, const M: u32, const N: u32, const K: u32, const G: u32>(
    dst: u32,
    a: u32,
    b: *mut f16,
    accumulate: bool,
    b_byte_offset: u32,
) {
    let _ = (dst, a, b, accumulate, b_byte_offset);
    unreachable!("CuTe compiler marker")
}
/// Device-only rank-three load marker with inner, row and outer-chunk coordinates.
///
/// # Safety
/// The producer warp must supply matching live storage and a correctly initialized group completion barrier.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn tma_load<L, const G: u32>(
    desc: *const TensorMap<f16, L>,
    dst: *mut f16,
    c0: i32,
    c1: i32,
    c2: i32,
    bar: *mut Barrier,
    rank: u32,
) {
    let _ = (desc, dst, c0, c1, c2, bar, rank);
    unreachable!("CuTe compiler marker")
}
/// Device-only rank-three store marker with inner, row and outer-chunk coordinates.
///
/// # Safety
/// The elected issuer must publish shared writes and retain storage until the copy stops reading it.
#[inline(never)]
pub unsafe fn tma_store<L, const G: u32>(
    desc: *const TensorMap<f16, L>,
    src: *mut f16,
    c0: i32,
    c1: i32,
    c2: i32,
) {
    let _ = (desc, src, c0, c1, c2);
    unreachable!("CuTe compiler marker")
}
