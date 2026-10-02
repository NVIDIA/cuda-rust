/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cooperative copies from global memory to shared memory.
//!
//! CUDA groups its workers like this:
//!
//! ```text
//! thread = one worker
//! warp   = 32 threads that run together
//! CTA    = one block of threads that can share memory
//! ```
//!
//! [`copy_g2s`] lets every thread in a CTA copy one piece of a tile:
//!
//! ```text
//! global matrix ── all CTA threads ──► shared tile
//! ```
//!
//! Types hold facts fixed before the kernel starts, such as layouts and
//! alignment. Function arguments hold values known only while it runs, such
//! as pointers, sizes, the tile coordinate, and the thread number.

use core::marker::PhantomData;

use crate::markers::LeadingDimMarker;

/// A row-major matrix stored in global GPU memory.
///
/// `leading_dim.elements` is the row pitch: the number of elements from the
/// start of one row to the start of the next. It can be larger than `cols`
/// when rows contain padding.
///
/// ```text
/// row 0: [ data data data | padding ]
/// row 1: [ data data data | padding ]
///          <---- row pitch ---->
/// ```
///
/// `LD` records a compile-time divisibility promise about that pitch. Build
/// this plain data value with a struct literal. [`copy_g2s`] checks the full
/// pointer, size, pitch, and layout contract.
#[repr(C)]
pub struct GmemMatrix<T, LD> {
    /// Start of the global-memory allocation.
    pub base: *const T,
    /// Logical matrix rows.
    pub rows: usize,
    /// Logical matrix columns, excluding row padding.
    pub cols: usize,
    /// Row pitch plus its compile-time divisibility marker.
    pub leading_dim: LD,
}

/// A tile in memory shared by all threads in one CTA.
///
/// `Layout` has no runtime bytes. It tells the compiler how logical rows and
/// columns map to physical addresses:
///
/// ```text
/// base pointer + capacity + zero-byte Layout marker
///                         │
///                         └── visible at the copy call
/// ```
///
/// Build this plain data value with a struct literal.
#[repr(C)]
pub struct SmemTile<T, Layout> {
    /// Start of the shared-memory allocation.
    pub base: *mut T,
    /// Number of `T` elements available from `base`.
    pub capacity: usize,
    /// Zero-byte marker that keeps the shared layout in the type.
    pub layout: PhantomData<Layout>,
}

/// Copy one global-memory tile into shared memory using the whole CTA.
///
/// `tile_coord = (r, c)` counts tiles, not elements. For a 16x8 tile,
/// `(2, 3)` starts at element `(32, 24)`.
///
/// The compiler retains this call as one typed cooperative-copy operation.
/// The selected backend chooses its realization. On a matrix edge, every
/// thread still follows the same path; a copy reports how many source bytes
/// are valid, and missing bytes become zero. The caller must still complete
/// the copy protocol and synchronize the CTA.
///
/// # Safety
///
/// The CTA must contain exactly the threads described by `ThrL`. Every one of
/// those threads must call this function on the same control-flow path, and
/// their `tidx` values must cover `ThrL` exactly once. Both raw pointers must
/// be valid. Each `Atom`, meaning one fixed-size copy piece, must be aligned.
/// The destination must have the capacity required by `SmemL`.
///
/// `src.leading_dim.elements` must be at least `src.cols` and divisible by
/// the divisor carried by its `LeadingDim<D>` type. The compiler checks that
/// the thread/value layout covers the tile exactly, that each atom maps to a
/// valid destination, and that `D * size_of::<T>()` is at least one atom.
/// Edge tiles are safe because out-of-bounds source bytes are replaced by
/// zero instead of being read.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn copy_g2s<Atom, ThrL, ValL, TileL, SmemL, T: Copy>(
    src: &GmemMatrix<T, impl LeadingDimMarker>,
    tile_coord: (usize, usize),
    dst_smem_tile: &mut SmemTile<T, SmemL>,
    tidx: u32,
) {
    let _ = (
        src,
        tile_coord,
        dst_smem_tile,
        tidx,
        PhantomData::<(Atom, ThrL, ValL, TileL)>,
    );
    unreachable!("cute-rs `copy_g2s` executed outside device compilation")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_carriers_are_only_the_visible_abi_fields() {
        // The layout marker occupies no bytes:
        //
        // SmemTile = [ pointer ][ capacity ][ zero-byte static marker ]
        assert_eq!(
            core::mem::size_of::<GmemMatrix<u32, crate::markers::LeadingDim<4>>>(),
            4 * core::mem::size_of::<usize>()
        );
        assert_eq!(
            core::mem::size_of::<SmemTile<u32, ()>>(),
            2 * core::mem::size_of::<usize>()
        );

        let tile = SmemTile::<u32, ()> {
            base: core::ptr::null_mut(),
            capacity: 24,
            layout: PhantomData,
        };
        assert!(tile.base.is_null());
        assert_eq!(tile.capacity, 24);
    }
}
