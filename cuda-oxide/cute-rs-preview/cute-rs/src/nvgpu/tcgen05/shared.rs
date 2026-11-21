/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared-memory tiles for paired two-CTA GEMM.

use crate::layout::ReifySmem2D;
use crate::utils::blockscaled::Sm120ScaleAtom;
use core::marker::PhantomData;
pub use cute_layout::{SharedLayout, SmemLayout};

/// A non-owning shared tile with a compile-time layout and an operand role (`OperandA` or `OperandB`).
/// Carries `Layout` and `Role` in the type system so MMA and TMA operations reject mismatched
/// tile shapes or swapped A/B operands at compile time.
pub struct Sm100SharedTile<Layout, Role> {
    /// Non-owning shared allocation base pointer (`addrspace(3)` / generic shared pointer).
    /// Reinterpreted as bytes when wrapping packed 4-bit FP4 (`E2M1`) tiles.
    pub(crate) base: *mut f16,
    /// Zero-byte marker recording the compile-time shared layout and A/B operand role.
    /// Enforces layout and role agreement between TMA copies and `TiledMma`.
    marker: PhantomData<(Layout, Role)>,
}

impl<Layout, Role> Copy for Sm100SharedTile<Layout, Role> {}
impl<Layout, Role> Clone for Sm100SharedTile<Layout, Role> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<Layout: ReifySmem2D, Role> Sm100SharedTile<Layout, Role> {
    /// Interpret existing shared storage using `Layout`.
    ///
    /// # Safety
    ///
    /// `base` must remain live and contain every element addressed by `Layout`.
    /// TMA input bases must be aligned to their full swizzle period; the
    /// supported B128 MMA layouts require 1024-byte alignment.
    #[inline(always)]
    pub const unsafe fn from_raw(base: *mut f16) -> Self {
        Self {
            base,
            marker: PhantomData,
        }
    }

    /// Interpret existing byte-addressed shared storage (such as packed FP4 `u8` tiles) using `Layout`.
    ///
    /// # Safety
    ///
    /// `base` must point to live shared memory aligned for `Layout` and cover all staged sub-tiles.
    #[must_use]
    #[inline(always)]
    pub const unsafe fn from_raw_bytes(base: *mut u8) -> Self {
        Self {
            base: base.cast::<f16>(),
            marker: PhantomData,
        }
    }

    /// Bind stage `slot` of a byte-addressed shared-memory ring at `smem_base + stage_offset_bytes`,
    /// deriving the per-stage byte stride from `Layout` (`size_in_bytes::<u8, Layout>()`).
    ///
    /// # Safety
    /// `smem_base.add(stage_offset_bytes + slot * stage_bytes)` must point to live, aligned shared memory.
    #[must_use]
    #[inline(always)]
    pub const unsafe fn from_stage_bytes(
        smem_base: *mut u8,
        stage_offset_bytes: usize,
        slot: usize,
    ) -> Self {
        let stage_bytes = crate::nvgpu::cpasync::size_in_bytes::<u8, Layout>() as usize;
        unsafe { Self::from_raw_bytes(smem_base.add(stage_offset_bytes + slot * stage_bytes)) }
    }
}

/// A non-owning shared-memory view of staged 512-byte [`Sm120ScaleAtom`] blocks with an A/B role (`Mkl` or `Nkl`).
/// Prevents swapping A-side weight scales (`Mkl`) and B-side activation scales (`Nkl`) when staging
/// scales into TMEM via `tcgen05.cp.32x128b.warpx4`.
pub struct Sm100ScaleTile<Role> {
    /// Non-owning shared `u32` scale-word base pointer (`128` words = `512` bytes per scale atom).
    /// Must be 128-byte aligned in shared memory for `tcgen05.cp`.
    pub(crate) base: *mut u32,
    /// Zero-byte marker distinguishing A-side (`Mkl`) from B-side (`Nkl`) scale tiles at compile time.
    /// Pairs with [`Sm120ScaleAtom`] to document the canonical `(32, 4, 4)` layout invariant.
    marker: PhantomData<(Sm120ScaleAtom, Role)>,
}

impl<Role> Copy for Sm100ScaleTile<Role> {}
impl<Role> Clone for Sm100ScaleTile<Role> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<Role> Sm100ScaleTile<Role> {
    /// Wrap a shared-memory buffer holding staged 512-byte [`Sm120ScaleAtom`] scale blocks.
    ///
    /// # Safety
    ///
    /// `base` must point to 128-byte-aligned shared memory containing at least one 512-byte scale atom per step.
    #[must_use]
    #[inline(always)]
    pub const unsafe fn from_raw(base: *mut u32) -> Self {
        Self {
            base,
            marker: PhantomData,
        }
    }

    /// Bind stage `slot` of a shared-memory scale ring at `smem_base + stage_offset_bytes`,
    /// deriving the per-stage byte stride (`stage_bytes`) at compile time.
    ///
    /// # Safety
    /// `smem_base.add(stage_offset_bytes + slot * stage_bytes)` must be 128-byte aligned in shared memory.
    #[must_use]
    #[inline(always)]
    pub const unsafe fn from_stage_bytes(
        smem_base: *mut u8,
        stage_offset_bytes: usize,
        stage_bytes: usize,
        slot: usize,
    ) -> Self {
        unsafe {
            Self::from_raw(
                smem_base
                    .add(stage_offset_bytes + slot * stage_bytes)
                    .cast::<u32>(),
            )
        }
    }
}

/// Sealed element marker for 16-bit (`f16` and `BFloat16`) `tcgen05` shared operand tiles.
mod sealed_elem {
    /// Implemented only for 2-byte floating-point types accepted by `tcgen05.MmaF16BF16Op`.
    pub trait MmaHalfElement {}
}

/// Element type accepted by 16-bit `tcgen05` [`SharedTile`], [`SharedTileArray`], and [`MmaSs`](super::MmaSs).
pub trait MmaHalfElement: sealed_elem::MmaHalfElement + Copy {}

impl sealed_elem::MmaHalfElement for f16 {}
impl MmaHalfElement for f16 {}
impl sealed_elem::MmaHalfElement for crate::numeric::BFloat16 {}
impl MmaHalfElement for crate::numeric::BFloat16 {}

/// Non-owning shared tile, typed by its physical layout, MMA operand role, and 16-bit element type `T`.
pub struct SharedTile<L, Role, T = f16> {
    /// Original shared 16-bit allocation base, before selecting an array tile.
    pub(crate) base: *mut f16,
    /// Byte displacement of this tile from the original allocation base.
    pub(crate) byte_offset: u32,
    /// Carries the physical layout, MMA operand role, and element type `T`.
    marker: PhantomData<(L, Role, T)>,
}
impl<L: SmemLayout, Role, T: MmaHalfElement> SharedTile<L, Role, T> {
    /// # Safety
    /// Storage must cover the layout and remain live through asynchronous use.
    /// Its base must be aligned to the layout's complete swizzle period (1024
    /// bytes for B128, 512 bytes for B64).
    #[inline(always)]
    pub const unsafe fn from_raw(base: *mut T) -> Self {
        let () = L::VALID;
        const { assert!(size_of::<T>() == 2) };
        Self {
            base: base.cast::<f16>(),
            byte_offset: 0,
            marker: PhantomData,
        }
    }
    /// Borrow this tile's shared storage address.
    #[inline(always)]
    pub const fn as_ptr(&self) -> *mut T {
        self.base
            .wrapping_add((self.byte_offset / 2) as usize)
            .cast::<T>()
    }

    /// Store eight contiguous 16-bit values (carried as four packed `u32` words) at `(row, col)`.
    ///
    /// # Safety
    /// `(row, col)` must lie within `L`, `col` must be a multiple of 8, the
    /// tile's shared storage must be live and exclusively writable for those
    /// eight elements, and callers must issue the required proxy fence or
    /// barrier before asynchronous or cross-thread readers access the tile.
    #[inline(always)]
    pub unsafe fn store_packed_8(&mut self, row: u32, col: u32, words: [u32; 4]) {
        let offset = L::crd2idx(row, col);
        let packed = (words[0] as u128)
            | ((words[1] as u128) << 32)
            | ((words[2] as u128) << 64)
            | ((words[3] as u128) << 96);
        unsafe {
            *self.as_ptr().add(offset).cast::<u128>() = packed;
        }
    }
}

/// Contiguous array of complete shared tiles with one layout, operand role, and 16-bit element type `T`.
pub struct SharedTileArray<L, Role, const TILES: u32, T = f16> {
    /// Shared 16-bit allocation containing every complete array tile.
    base: *mut f16,
    /// Carries the array's common layout, operand role, and element type `T`.
    marker: PhantomData<(L, Role, T)>,
}
impl<L: SmemLayout, Role, const TILES: u32, T: MmaHalfElement> SharedTileArray<L, Role, TILES, T> {
    /// Bytes in one complete tile; require a whole swizzle period and a representable size.
    const TILE_BYTES: u32 = {
        let () = L::VALID;
        assert!(size_of::<T>() == 2);
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
    /// bytes for B128, 512 bytes for B64).
    #[inline(always)]
    pub const unsafe fn from_raw(base: *mut T) -> Self {
        let () = Self::VALID;
        Self {
            base: base.cast::<f16>(),
            marker: PhantomData,
        }
    }
    /// Select a complete tile while retaining the allocation's descriptor base.
    /// # Safety
    /// `index` must be less than `TILES`.
    #[inline(always)]
    pub const unsafe fn tile_unchecked(&self, index: u32) -> SharedTile<L, Role, T> {
        SharedTile {
            base: self.base,
            byte_offset: index * Self::TILE_BYTES,
            marker: PhantomData,
        }
    }
}
