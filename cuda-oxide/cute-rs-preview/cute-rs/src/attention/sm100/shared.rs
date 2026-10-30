/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared layouts and tile arrays for independent attention MMA.

use core::marker::PhantomData;

/// Physical layout of a shared FP16 operand, with CuTe's tiled byte swizzle.
mod sealed {
    /// Marks only shared layouts recognized by the compiler.
    pub trait Layout {}
}

/// Implemented only by the layouts whose structure the compiler recognizes.
pub trait SmemLayout: sealed::Layout {
    /// Evaluating this constant rejects an unsupported physical layout.
    const VALID: ();
    /// Logical MN extent.
    const ROWS: u32;
    /// Logical K extent.
    const COLS: u32;
    /// Whether the MN mode is contiguous.
    const MN_MAJOR: bool;
    /// Contiguous FP16 values in one swizzle atom.
    const SPAN: u32;
    /// Number of XOR bits in the shared-memory swizzle.
    const SWIZZLE_BITS: u32;
    /// Map a logical coordinate into shared storage in elements (CuTeDSL
    /// `cute.crd2idx`).
    fn crd2idx(row: u32, col: u32) -> usize;
}

/// Logical `(MN,K)` layout. `MN_MAJOR` selects the contiguous logical mode.
/// Contiguous dimensions wider than 64 FP16 values are tiled in 128-byte chunks.
///
/// The contiguous extent must be 32 or a positive multiple of 64; the other
/// extent must be a positive multiple of 8. Extents and offsets count FP16
/// elements, not bytes. With `MN_MAJOR = false`, K is contiguous:
///
/// ```
/// use cute_rs::attention::sm100::SharedLayout;
/// type Tile = SharedLayout<64, 64>;
/// assert_eq!(Tile::crd2idx(0, 1), 1);
/// ```
pub struct SharedLayout<const ROWS: u32, const COLS: u32, const MN_MAJOR: bool = false>;
impl<const R: u32, const C: u32, const MN: bool> SharedLayout<R, C, MN> {
    /// Reject unsupported contiguous swizzle spans and outer tile extents.
    const VALID: () = {
        let contiguous = if MN { R } else { C };
        let outer = if MN { C } else { R };
        assert!(contiguous >= 32 && contiguous.is_multiple_of(32));
        assert!(outer > 0 && outer.is_multiple_of(8));
        assert!(contiguous == 32 || contiguous.is_multiple_of(64));
    };
    /// Offset in FP16 elements from the start of this shared tile (CuTeDSL
    /// `cute.crd2idx`).
    #[inline(always)]
    pub fn crd2idx(row: u32, col: u32) -> usize {
        let () = Self::VALID;
        let (inner, outer, outer_extent) = if MN { (row, col, C) } else { (col, row, R) };
        let span = Self::SPAN;
        let linear = (inner / span) * (span * outer_extent) + outer * span + inner % span;
        let mask = (1u32 << Self::SWIZZLE_BITS) - 1;
        (linear ^ (((linear >> 6) & mask) << 3)) as usize
    }
}
impl<const R: u32, const C: u32, const MN: bool> sealed::Layout for SharedLayout<R, C, MN> {}
impl<const R: u32, const C: u32, const MN: bool> SmemLayout for SharedLayout<R, C, MN> {
    const VALID: () = SharedLayout::<R, C, MN>::VALID;
    const ROWS: u32 = R;
    const COLS: u32 = C;
    const MN_MAJOR: bool = MN;
    const SPAN: u32 = if (if MN { R } else { C }) == 32 {
        32
    } else {
        64
    };
    const SWIZZLE_BITS: u32 = if Self::SPAN == 32 { 2 } else { 3 };
    #[inline(always)]
    fn crd2idx(row: u32, col: u32) -> usize {
        Self::crd2idx(row, col)
    }
}

/// Non-owning shared tile, typed by its physical layout and MMA operand role.
pub struct SharedTile<L, Role> {
    /// Original shared FP16 allocation base, before selecting an array tile.
    pub(super) base: *mut f16,
    /// Byte displacement of this tile from the original allocation base.
    pub(super) byte_offset: u32,
    /// Carries the physical layout and MMA operand role.
    marker: PhantomData<(L, Role)>,
}
impl<L: SmemLayout, Role> SharedTile<L, Role> {
    /// # Safety
    /// Storage must cover the layout and remain live through asynchronous use.
    /// Its base must be aligned to the layout's complete swizzle period (1024
    /// bytes for B128, 512 bytes for B64).
    #[inline(always)]
    pub const unsafe fn from_raw(base: *mut f16) -> Self {
        let () = L::VALID;
        Self {
            base,
            byte_offset: 0,
            marker: PhantomData,
        }
    }
    /// Borrow this tile's shared storage address.
    #[inline(always)]
    pub const fn as_ptr(&self) -> *mut f16 {
        // Forming a raw address does not access storage or require it to remain
        // live. Dereferencing it retains the caller's lifetime obligations.
        self.base.wrapping_add((self.byte_offset / 2) as usize)
    }

    /// Store eight contiguous FP16 values (carried as four packed `u32` words) at `(row, col)`.
    ///
    /// Both `B64` and `B128` shared swizzles keep aligned 8-element (`16`-byte)
    /// chunks contiguous, so one 128-bit store writes the complete 8-element vector
    /// at the swizzled address `L::crd2idx(row, col)`.
    ///
    /// # Safety
    /// `(row, col)` must lie within `L`, `col` must be a multiple of 8, the
    /// tile's shared storage must be live and exclusively writable for those
    /// eight elements, and callers must issue the required proxy fence or
    /// barrier before asynchronous or cross-thread readers access the tile.
    #[inline(always)]
    pub unsafe fn store_packed_8(&mut self, row: u32, col: u32, words: [u32; 4]) {
        let offset = L::crd2idx(row, col);
        unsafe {
            *self.as_ptr().add(offset).cast::<ctm_rs::memory::U32x4>() =
                ctm_rs::memory::U32x4(words);
        }
    }
}

/// Contiguous array of complete shared tiles with one layout and operand role.
///
/// Views retain the allocation base separately from their tile offset. MMA can
/// reuse the base descriptor; tensor copies address the selected physical tile.
/// Complete tiles preserve the XOR swizzle's phase at every array element.
pub struct SharedTileArray<L, Role, const TILES: u32> {
    /// Shared FP16 allocation containing every complete array tile.
    base: *mut f16,
    /// Carries the array’s common layout and operand role.
    marker: PhantomData<(L, Role)>,
}
impl<L: SmemLayout, Role, const TILES: u32> SharedTileArray<L, Role, TILES> {
    /// Bytes in one complete tile; require a whole swizzle period and a representable size.
    const TILE_BYTES: u32 = {
        let () = L::VALID;
        let elements = (L::ROWS as u64) * (L::COLS as u64);
        assert!(elements <= u32::MAX as u64 / 2);
        let bytes = elements as u32 * 2;
        assert!(bytes.is_multiple_of(1 << (L::SWIZZLE_BITS + 7)));
        bytes
    };
    /// Require a nonempty array whose total byte offset fits in u32.
    const VALID: () = {
        assert!(TILES > 0);
        assert!(TILES <= u32::MAX / Self::TILE_BYTES);
    };
    /// # Safety
    /// `base` must point to live shared storage covering `TILES` consecutive
    /// complete tiles. It must be aligned to the complete swizzle period (1024
    /// bytes for B128, 512 bytes for B64). The allocation and each accessed tile
    /// must remain live through all asynchronous operations using their views.
    #[inline(always)]
    pub const unsafe fn from_raw(base: *mut f16) -> Self {
        let () = Self::VALID;
        Self {
            base,
            marker: PhantomData,
        }
    }
    /// Select a complete tile while retaining the allocation's descriptor base.
    /// # Safety
    /// `index` must be less than `TILES`. All asynchronous accesses through this
    /// view must obey the allocation lifetime and buffer-ownership protocol.
    #[inline(always)]
    pub const unsafe fn tile_unchecked(&self, index: u32) -> SharedTile<L, Role> {
        SharedTile {
            base: self.base,
            byte_offset: index * Self::TILE_BYTES,
            marker: PhantomData,
        }
    }
}
