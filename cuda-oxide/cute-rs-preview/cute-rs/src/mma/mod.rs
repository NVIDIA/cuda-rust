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
//! each thread holds: FragmentA [u32; 4], FragmentB [u32; 2], FragmentC [f32; 4]
//! ```
//!
//! The compiler retains `load_matrix_*` as typed matrix-load semantics. The
//! selected backend chooses the lane mapping and target operation. The `SmemL`
//! type tells it the shared-memory layout, including any swizzle (a fixed
//! address rearrangement). A plain pointer would lose that information.
//!
//! [`bf16`] provides typed BF16 partitions and fragments; [`block_scaled`]
//! provides the SM120 MXFP4 tiled family. Their operand contracts remain
//! distinct from the common FP16 fragments defined here.
//!
//! [`FragmentC::mma_atom_call`] issues one `mma.sync` through ctm-rs
//! `mma_f16`; [`bf16::Bf16Mma16x8x16::mma_atom_call`] does the same through
//! `mma_bf16`.
//!
pub mod bf16;
pub mod block_scaled;

use crate::copy::cooperative::SmemTile;
use crate::numeric::BFloat16;

/// Seals [`LdmatrixElement`] to 16-bit floating-point storage types (`f16` and [`BFloat16`]).
mod ldmatrix_element {
    /// Implemented only by 2-byte matrix-load element types.
    pub trait Sealed {}
    impl Sealed for f16 {}
    impl Sealed for crate::numeric::BFloat16 {}
}

/// A 16-bit element type supported by warp-cooperative `ldmatrix` shared-memory loads.
pub trait LdmatrixElement: Copy + ldmatrix_element::Sealed {}
impl LdmatrixElement for f16 {}
impl LdmatrixElement for BFloat16 {}

/// One thread's part of a 16x16 `f16` A tile.
///
/// Each of the four 32-bit registers holds two adjacent `f16` values in the
/// order expected by `mma.sync`.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct FragmentA(
    #[doc = "This lane’s four packed A registers in instruction order."] pub [u32; 4],
);

/// One thread's part of a 16x8 `f16` B tile: two 32-bit registers.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct FragmentB(
    #[doc = "This lane’s two packed B registers in instruction order."] pub [u32; 2],
);

/// Two adjacent 16x8 B fragments in MMA register order.
///
/// Produced by [`load_matrix_b_pair`], [`load_matrix_b_pair_k_major`], or by
/// retiling an A-fragment load with [`FragmentA::as_b_pair`].
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct FragmentBPair(
    #[doc = "Packed B registers for two neighboring N8 fragments."] pub [u32; 4],
);

impl FragmentBPair {
    /// Select the left (`N = 0`) or right (`N = 1`) fragment.
    #[inline(always)]
    pub fn get<const N: usize>(self) -> FragmentB {
        const { assert!(N < 2) };
        FragmentB([self.0[N * 2], self.0[N * 2 + 1]])
    }
}

impl FragmentA {
    /// Interpret a non-transposed load of two K-major B tiles.
    ///
    /// The shared window contains 16 B rows and 16 reduction elements.
    #[inline(always)]
    pub fn as_b_pair(self) -> FragmentBPair {
        FragmentBPair([self.0[0], self.0[2], self.0[1], self.0[3]])
    }
}

/// One thread's four `f32` values from a 16x8 result tile.
///
/// `cuda_device::mma_frag` maps these four values to result coordinates.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct FragmentC(
    #[doc = "This lane’s four FP32 result registers in instruction order."] pub [f32; 4],
);

impl FragmentC {
    /// Four zero values for starting `C = A*B + C`.
    pub const ZERO: FragmentC = FragmentC([0.0; 4]);

    /// Accumulate one FP16 16x16 by 16x8 warp matrix product in FP32.
    ///
    /// # Safety
    /// All lanes must participate with matching fragments and supported hardware.
    #[must_use = "use the returned accumulator; mma_atom_call does not modify self in place"]
    #[inline(always)]
    pub unsafe fn mma_atom_call(self, a: FragmentA, b: FragmentB) -> Self {
        use ctm_rs::mma::{Accumulator16x8, F16A, F16B, mma_f16};

        Self(
            unsafe {
                mma_f16(
                    Accumulator16x8::from_registers(self.0),
                    F16A::from_registers(a.0),
                    F16B::from_registers(b.0),
                )
            }
            .registers(),
        )
    }
}

/// Load one 16x16 `f16` A tile into the 32 threads of a warp.
///
/// Each thread receives its [`FragmentA`]. `SmemL` describes the shared-memory
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
pub unsafe fn load_matrix_a<SmemL, T: LdmatrixElement>(
    src: &SmemTile<T, SmemL>,
    warp_tile: (usize, usize),
    lane: u32,
) -> FragmentA {
    let _ = (src, warp_tile, lane);
    unreachable!("cute-rs `load_matrix_a` executed outside device compilation")
}

/// Load one 16x8 `f16` or `BFloat16` B tile into the 32 threads of a warp.
///
/// Each thread receives its [`FragmentB`]. `SmemL` describes the shared-memory
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
pub unsafe fn load_matrix_b<SmemL, T: LdmatrixElement>(
    src: &SmemTile<T, SmemL>,
    warp_tile: (usize, usize),
    lane: u32,
) -> FragmentB {
    let _ = (src, warp_tile, lane);
    unreachable!("cute-rs `load_matrix_b` executed outside device compilation")
}

/// Load a 16x16 shared-memory window as two adjacent B fragments.
///
/// `warp_tile` counts 16x16 windows. One `ldmatrix.x4.trans` supplies both
/// 16x8 operands, retaining the source layout and swizzle in the compiler.
///
/// # Safety
/// The lane participation, alignment, and storage requirements of
/// [`load_matrix_a`] apply to the entire 16x16 window.
#[inline(never)]
pub unsafe fn load_matrix_b_pair<SmemL, T: LdmatrixElement>(
    src: &SmemTile<T, SmemL>,
    warp_tile: (usize, usize),
    lane: u32,
) -> FragmentBPair {
    let _ = (src, warp_tile, lane);
    unreachable!("cute-rs `load_matrix_b_pair` executed outside device compilation")
}

/// Load a K-major 16x16 shared-memory window as two adjacent B fragments.
///
/// Shared rows are B columns (`N`), and shared columns are reduction elements
/// (`K`). `warp_tile = (n, k)` selects the window starting at `(16*n, 16*k)`.
/// One non-transposed `ldmatrix.x4` assigns its four 8x8 submatrices in B-fragment
/// order, so each pair of result registers can feed one `mma.sync` directly.
/// This produces the same fragments as [`load_matrix_a`] followed by
/// [`FragmentA::as_b_pair`], without requiring a register permutation after loading.
///
/// # Safety
/// The lane participation, alignment, and storage requirements of
/// [`load_matrix_a`] apply to the entire 16x16 window. `lane` must be the calling
/// thread's hardware lane number; the compiler assigns the submatrix addresses.
#[inline(never)]
pub unsafe fn load_matrix_b_pair_k_major<SmemL, T: LdmatrixElement>(
    src: &SmemTile<T, SmemL>,
    warp_tile: (usize, usize),
    lane: u32,
) -> FragmentBPair {
    let _ = (src, warp_tile, lane);
    unreachable!("cute-rs `load_matrix_b_pair_k_major` executed outside device compilation")
}
