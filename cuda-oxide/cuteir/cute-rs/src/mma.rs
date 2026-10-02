/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level tensor-core loads and multiply-accumulate operations.
//!
//! A thread is one GPU worker. A warp is 32 threads that run together. The
//! tensor core splits each small matrix across the registers of all 32
//! threads. One thread's piece is called a *fragment*. A lane is that
//! thread's position from 0 to 31 inside the warp.
//!
//! One `mma.sync.m16n8k16` performs this operation:
//!
//! ```text
//! A: 16x16 f16 ──┐
//!                 ├── one warp ──► C: 16x8 f32
//! B: 16x8  f16 ──┘                 C = A*B + C
//!
//! each thread holds: FragA [u32; 4], FragB [u32; 2], AccC [f32; 4]
//! ```
//!
//! The compiler retains `load_matrix_*` as typed matrix-load semantics. The
//! selected backend chooses the lane mapping and target operation. The `SmemL`
//! type tells it the shared-memory layout, including any swizzle (a fixed
//! address rearrangement). A plain pointer would lose that information.
//!
use crate::SmemTile;

/// One thread's part of a 16x16 `f16` A tile.
///
/// Each of the four 32-bit registers holds two adjacent `f16` values in the
/// order expected by `mma.sync`.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct FragA(pub [u32; 4]);

/// One thread's part of a 16x8 `f16` B tile: two 32-bit registers.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct FragB(pub [u32; 2]);

/// One thread's four `f32` values from a 16x8 result tile.
///
/// `cuda_device::mma_frag` maps these four values to result coordinates.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct AccC(pub [f32; 4]);

impl AccC {
    /// Four zero values for starting `C = A*B + C`.
    pub const ZERO: AccC = AccC([0.0; 4]);
}

/// Load one 16x16 `f16` A tile into the 32 threads of a warp.
///
/// Each thread receives its [`FragA`]. `SmemL` describes the shared-memory
/// layout and any address swizzle.
///
/// `warp_tile = (r, c)` counts 16x16 windows. For example, `(2, 3)` starts at
/// element `(32, 48)`.
///
/// # Safety
///
/// All 32 lanes of one warp must call this function on the same control-flow
/// path. `lane` must be the calling thread's hardware lane number. The shared
/// tile must contain the selected window. Every selected row address must be
/// aligned to 16 bytes under `SmemL`; a composed swizzle is valid when it
/// keeps at least 16-byte chunks together. The compiler checks the parts
/// known from the types.
#[inline(never)]
pub unsafe fn load_matrix_a<SmemL>(
    src: &SmemTile<f16, SmemL>,
    warp_tile: (usize, usize),
    lane: u32,
) -> FragA {
    let _ = (src, warp_tile, lane);
    unreachable!("cute-rs `load_matrix_a` executed outside device compilation")
}

/// Load one 16x8 `f16` B tile into the 32 threads of a warp.
///
/// Each thread receives its [`FragB`]. `SmemL` describes the shared-memory
/// layout and any address swizzle.
///
/// `warp_tile = (r, c)` counts 16x8 windows. The window starts at element
/// `(16*r, 8*c)`. `ldmatrix.trans` transposes while loading, turning a
/// row-major tile into the column order expected by `mma.sync`.
///
/// # Safety
///
/// All 32 lanes of one warp must call this function on the same control-flow
/// path, and `lane` must be the calling thread's hardware lane number. The
/// shared tile must contain the selected window, with row addresses aligned
/// as required by [`load_matrix_a`].
#[inline(never)]
pub unsafe fn load_matrix_b<SmemL>(
    src: &SmemTile<f16, SmemL>,
    warp_tile: (usize, usize),
    lane: u32,
) -> FragB {
    let _ = (src, warp_tile, lane);
    unreachable!("cute-rs `load_matrix_b` executed outside device compilation")
}
