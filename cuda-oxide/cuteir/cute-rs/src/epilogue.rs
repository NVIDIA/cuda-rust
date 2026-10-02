/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Store an SM120 128x128 result tile in shared memory.
//!
//! A thread is one GPU worker. A warp is 32 threads that run together. A CTA,
//! also called a block, is a group of warps that can use the same shared
//! memory.
//! A lane is a thread's position from 0 to 31 inside its warp.
//!
//! One tensor-core operation gives each thread four `f32` results in
//! [`AccC`]. Together, one warp owns a 16x8 result block. This
//! module calls that small block an *atom*. `stmatrix` is a warp-wide
//! instruction that stores packed register values into shared memory.
//!
//! ```text
//! each thread: [c0, c1, c2, c3] f32
//!                    │ convert two at a time
//!                    ▼
//!             [c0,c1] [c2,c3] f16
//!                    │ one stmatrix for the warp
//!                    ▼
//!           one 16x8 f16 atom in shared memory
//! ```
//!
//! The Tensor Memory Accelerator (TMA) is hardware that copies complete
//! tiles. A TMA store using the 128-byte (B128) swizzle can move 64 `f16`
//! columns at once. The 128-column result therefore uses two adjacent 128x64
//! halves. A swizzle is a fixed address rearrangement that reduces cases where
//! many threads request the same shared-memory bank and have to wait; each
//! half uses its own.
//!
//! ```text
//! logical 128x128 C tile
//! ┌──────── 128x64 ────────┬──────── 128x64 ────────┐
//! │ left: 16 KiB           │ right: 16 KiB          │
//! └────────────────────────┴────────────────────────┘
//! shared allocation: 32 KiB total
//! ```
//!
//! Kernel code uses the small methods below. At the compiler boundary they
//! stay visible as one result-tile flow: select the shared tile, select a
//! warp, store the accumulator, synchronize, and expose two TMA halves.

use core::marker::PhantomData;

use crate::markers::{Composed, RowMajor, Swizzle};
use crate::{AccC, SmemTile};

/// Layout of one 128x64 half that a B128 TMA store can read.
///
/// `S<3,3,3>` describes the address swizzle in `f16` elements. Because one
/// `f16` is two bytes, the same layout is TMA's byte-based `S<3,4,3>` B128
/// mode.
pub type Sm120EpilogueHalfLayout = Composed<Swizzle<3, 3, 3>, 0, RowMajor<128, 64>>;

/// Rows in the full output tile.
pub const SM120_EPILOGUE_ROWS: usize = 128;
/// Columns in the full output tile.
pub const SM120_EPILOGUE_COLS: usize = 128;
/// Columns moved by one B128 TMA store.
pub const SM120_EPILOGUE_HALF_COLS: usize = 64;
/// `f16` elements in one 128x64 half.
pub const SM120_EPILOGUE_HALF_ELEMENTS: usize = SM120_EPILOGUE_ROWS * SM120_EPILOGUE_HALF_COLS;
/// Bytes in one 128x64 half.
pub const SM120_EPILOGUE_HALF_BYTES: usize =
    SM120_EPILOGUE_HALF_ELEMENTS * core::mem::size_of::<f16>();
/// `f16` elements in both halves.
pub const SM120_EPILOGUE_ELEMENTS: usize = SM120_EPILOGUE_ROWS * SM120_EPILOGUE_COLS;
/// Bytes in both halves.
pub const SM120_EPILOGUE_BYTES: usize = SM120_EPILOGUE_ELEMENTS * core::mem::size_of::<f16>();

const N_ATOM_OFFSETS: [usize; 8] = [0, 8, 32, 40, 64, 72, 96, 104];

/// Top-left `(row, column)` of one warp's 16x8 result atom.
///
/// Eight compute warps split the CTA tile. `warp_id % 4` selects one of four
/// 16-row positions. `warp_id / 4` selects the left or right 16-column group.
/// `m_band` adds either 0 or 64 rows. `n_slot` selects one of eight fixed
/// column offsets.
///
/// ```text
/// row = 16*(warp_id % 4) + 64*m_band
/// col = 16*(warp_id / 4) + [0,8,32,40,64,72,96,104][n_slot]
/// ```
///
/// Expected inputs are `warp_id` in `0..8`, `m_band` in `0..2`, and
/// `n_slot` in `0..8`.
#[must_use]
pub const fn sm120_epilogue_atom_origin(
    warp_id: usize,
    m_band: usize,
    n_slot: usize,
) -> (usize, usize) {
    let warp_m = warp_id % 4;
    let warp_n = warp_id / 4;
    (
        16 * warp_m + 64 * m_band,
        16 * warp_n + N_ATOM_OFFSETS[n_slot],
    )
}

/// Convert logical `(row, column)` into an `f16` offset in shared memory.
///
/// The address swizzle is applied separately inside each 128x64 half. This
/// helper is mainly for layout tests and debugging. Kernels normally call
/// [`Sm120EpilogueWarp128x128::store_atom`].
///
/// Expected inputs are `row` in `0..128` and `col` in `0..128`.
#[must_use]
pub const fn sm120_epilogue_physical_offset(row: usize, col: usize) -> usize {
    let half = col / SM120_EPILOGUE_HALF_COLS;
    let local_col = col % SM120_EPILOGUE_HALF_COLS;
    let logical = row * SM120_EPILOGUE_HALF_COLS + local_col;
    // S<3,3,3> changes address bits 3..5 using bits 6..8. It leaves bits
    // 0..2 unchanged, so each 8-f16 (16-byte) stmatrix chunk stays together.
    let swizzled = logical ^ ((logical & (0b111 << 6)) >> 3);
    half * SM120_EPILOGUE_HALF_ELEMENTS + swizzled
}

/// A typed view of the 32 KiB shared result buffer.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Sm120Epilogue128x128 {
    base: *mut f16,
}

impl Sm120Epilogue128x128 {
    /// Required alignment of the shared allocation.
    pub const ALIGNMENT: usize = 1024;
    /// Size of the full shared allocation.
    pub const BYTES: usize = SM120_EPILOGUE_BYTES;

    /// Treat an existing shared allocation as this result buffer.
    ///
    /// # Safety
    ///
    /// `base` must point to [`Self::BYTES`] writable bytes in shared memory
    /// and be aligned to 1024 bytes. The allocation must remain valid until
    /// every `stmatrix` write and asynchronous TMA read from it has finished.
    #[must_use]
    #[inline(always)]
    pub const unsafe fn from_raw(base: *mut f16) -> Self {
        unsafe { __compiler::epilogue_smem_overlay(base) }
    }

    /// Select the part of the result tile owned by one compute warp.
    ///
    /// # Safety
    ///
    /// `warp_id` must be in `0..8`. `lane` must be the calling thread's
    /// hardware lane number in `0..32`. All 32 lanes must select the same warp
    /// partition, each with its own correct lane, and take the same
    /// control-flow path for later stores.
    #[must_use]
    #[inline(always)]
    pub const unsafe fn get_slice(self, warp_id: usize, lane: u32) -> Sm120EpilogueWarp128x128 {
        unsafe { __compiler::epilogue_warp_slice(self, warp_id, lane) }
    }

    /// Wait until the previous asynchronous store no longer reads this tile.
    ///
    /// # Safety
    ///
    /// All eight compute warps must call this together. The ninth, producer
    /// warp must not participate in this counted barrier. `self` must refer
    /// to the live shared epilogue allocation used by those warps.
    #[inline(always)]
    pub unsafe fn sync_reusable(self) {
        unsafe { __compiler::epilogue_sync_reusable(self) };
    }

    /// Publish all compute-warp stores, then release the TMA issuer.
    ///
    /// This semantic boundary includes the generic-to-async shared proxy
    /// publication required before TMA reads the tile. Each backend selects
    /// its own fence and counted-barrier implementation.
    ///
    /// # Safety
    ///
    /// All eight compute warps must call this together. The producer warp
    /// must not participate, and `self` must refer to their live shared
    /// epilogue allocation.
    #[inline(always)]
    pub unsafe fn sync_ready_for_tma(self) {
        unsafe { __compiler::epilogue_sync_ready_for_tma(self) };
    }

    /// View one 128x64 half as the source of a typed TMA store.
    ///
    /// # Safety
    ///
    /// `HALF` must be 0 for the left half or 1 for the right half. The caller
    /// must provide the required synchronization and must not change this half
    /// until the asynchronous TMA store has finished reading it.
    #[must_use]
    #[inline(always)]
    pub unsafe fn tma_half<const HALF: usize>(self) -> SmemTile<f16, Sm120EpilogueHalfLayout> {
        unsafe { __compiler::epilogue_half::<HALF>(self) }
    }
}

/// Stable compiler boundaries for the SM120 output-tile protocol.
///
/// The public methods above are the readable source API. These exact helpers
/// let the importer preserve the same tile and synchronization story with or
/// without MIR inlining. Every returned value is still an ordinary Rust
/// pointer or scalar carrier.
#[doc(hidden)]
pub mod __compiler {
    use super::*;

    #[inline(never)]
    pub const unsafe fn epilogue_smem_overlay(base: *mut f16) -> Sm120Epilogue128x128 {
        Sm120Epilogue128x128 { base }
    }

    #[inline(never)]
    pub const unsafe fn epilogue_warp_slice(
        tile: Sm120Epilogue128x128,
        warp_id: usize,
        lane: u32,
    ) -> Sm120EpilogueWarp128x128 {
        Sm120EpilogueWarp128x128 {
            base: tile.base,
            warp_id,
            lane,
        }
    }

    #[inline(never)]
    pub unsafe fn epilogue_sync_reusable(tile: Sm120Epilogue128x128) {
        let _ = tile;
        unreachable!("cute-rs epilogue sync executed outside recognized device compilation")
    }

    #[inline(never)]
    pub unsafe fn epilogue_sync_ready_for_tma(tile: Sm120Epilogue128x128) {
        let _ = tile;
        unreachable!("cute-rs epilogue publication executed outside recognized device compilation")
    }

    #[inline(never)]
    pub unsafe fn epilogue_half<const HALF: usize>(
        tile: Sm120Epilogue128x128,
    ) -> SmemTile<f16, Sm120EpilogueHalfLayout> {
        const { assert!(HALF < 2) };
        SmemTile {
            base: unsafe { tile.base.add(HALF * SM120_EPILOGUE_HALF_ELEMENTS) },
            capacity: SM120_EPILOGUE_HALF_ELEMENTS,
            layout: PhantomData,
        }
    }
}

/// The result locations owned by one compute warp.
#[derive(Clone, Copy)]
pub struct Sm120EpilogueWarp128x128 {
    base: *mut f16,
    warp_id: usize,
    lane: u32,
}

impl Sm120EpilogueWarp128x128 {
    /// Convert and store one complete 16x8 result atom.
    ///
    /// `M_BAND` is 0 or 1. `N_SLOT` is in `0..8`. Each lane's four values
    /// land here inside the atom:
    ///
    /// ```text
    /// c0,c1 -> row lane/4,     columns 2*(lane%4) + {0,1}
    /// c2,c3 -> row lane/4 + 8, columns 2*(lane%4) + {0,1}
    /// ```
    ///
    /// The two packed pairs are the exact input format required by the
    /// non-transposed `stmatrix.m8n8.x2` instruction.
    ///
    /// # Safety
    ///
    /// All 32 lanes of the warp must call the same `M_BAND` and `N_SLOT` on
    /// the same control-flow path. This view must have been created with the
    /// calling hardware warp and lane. No other warp may write the same atom
    /// at the same time.
    #[inline(always)]
    pub unsafe fn store_atom<const M_BAND: usize, const N_SLOT: usize>(&self, acc: AccC) {
        const {
            assert!(M_BAND < 2);
            assert!(N_SLOT < 8);
        }

        let _ = (self.base, self.warp_id, self.lane, acc);
        unreachable!("cute-rs epilogue atom store executed outside recognized device compilation")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::markers::ReifySmem2D;
    use cuda_device::mma_frag;

    #[test]
    fn half_layout_is_exactly_one_b128_tma_tile() {
        assert_eq!(<Sm120EpilogueHalfLayout as ReifySmem2D>::ROWS, 128);
        assert_eq!(<Sm120EpilogueHalfLayout as ReifySmem2D>::COLS, 64);
        assert_eq!(<Sm120EpilogueHalfLayout as ReifySmem2D>::ROW_STRIDE, 64);
        assert_eq!(<Sm120EpilogueHalfLayout as ReifySmem2D>::COL_STRIDE, 1);
        assert_eq!(<Sm120EpilogueHalfLayout as ReifySmem2D>::SWIZZLE_B, 3);
        assert_eq!(<Sm120EpilogueHalfLayout as ReifySmem2D>::SWIZZLE_M, 3);
        assert_eq!(<Sm120EpilogueHalfLayout as ReifySmem2D>::SWIZZLE_S, 3);
        assert_eq!(SM120_EPILOGUE_HALF_BYTES, 16 * 1024);
        assert_eq!(Sm120Epilogue128x128::BYTES, 32 * 1024);
    }

    #[test]
    fn eight_warps_cover_the_logical_tile_once() {
        let mut owners = [0u8; SM120_EPILOGUE_ELEMENTS];
        for warp in 0..8 {
            for m_band in 0..2 {
                for n_slot in 0..8 {
                    let (atom_row, atom_col) = sm120_epilogue_atom_origin(warp, m_band, n_slot);
                    for lane in 0..32 {
                        for j in 0..4 {
                            let (r, c) = mma_frag::acc_coords(lane, j);
                            let logical = (atom_row + r) * SM120_EPILOGUE_COLS + atom_col + c;
                            assert_eq!(
                                owners[logical],
                                0,
                                "overlap at ({},{})",
                                atom_row + r,
                                atom_col + c
                            );
                            owners[logical] = 1;
                        }
                    }
                }
            }
        }
        assert!(owners.iter().all(|&count| count == 1));
    }

    #[test]
    fn two_swizzled_halves_are_disjoint_bijections() {
        let mut owners = [false; SM120_EPILOGUE_ELEMENTS];
        for row in 0..SM120_EPILOGUE_ROWS {
            for col in 0..SM120_EPILOGUE_COLS {
                let physical = sm120_epilogue_physical_offset(row, col);
                assert!(physical < owners.len());
                assert!(!owners[physical], "physical collision at {physical}");
                owners[physical] = true;
            }
        }
        assert!(owners.iter().all(|&claimed| claimed));
        assert_eq!(sm120_epilogue_physical_offset(0, 0), 0);
        assert_eq!(sm120_epilogue_physical_offset(0, 64), 8192);
    }

    #[test]
    fn every_stmatrix_row_address_is_sixteen_byte_aligned() {
        for warp in 0..8 {
            for m_band in 0..2 {
                for n_slot in 0..8 {
                    let (atom_row, atom_col) = sm120_epilogue_atom_origin(warp, m_band, n_slot);
                    for address_lane in 0..16 {
                        let physical =
                            sm120_epilogue_physical_offset(atom_row + address_lane, atom_col);
                        assert_eq!((physical * core::mem::size_of::<f16>()) % 16, 0);
                    }
                }
            }
        }
    }
}
