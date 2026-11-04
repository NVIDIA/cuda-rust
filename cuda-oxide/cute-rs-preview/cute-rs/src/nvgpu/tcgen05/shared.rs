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

    /// Return the CTA-shared byte offset of this tile's base pointer.
    /// Used by `Sm100BlockScaledTiledMma` to encode hardware `Tcgen05SmemDescriptor`s.
    ///
    /// # Safety
    /// `self.base` must point to valid shared memory belonging to the calling CTA.
    #[must_use]
    #[inline(always)]
    pub unsafe fn shared_offset(&self) -> u64 {
        unsafe { cuda_device::shared::cvta_generic_to_shared_offset(self.base.cast::<u8>()) }
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

    /// Return the CTA-shared byte offset of this scale tile's base pointer.
    /// Used to build the `Tcgen05SmemDescriptor` consumed by `tcgen05.cp.32x128b.warpx4`.
    ///
    /// # Safety
    /// `self.base` must point to valid shared memory belonging to the calling CTA.
    #[must_use]
    #[inline(always)]
    pub unsafe fn shared_offset(&self) -> u64 {
        unsafe { cuda_device::shared::cvta_generic_to_shared_offset(self.base.cast::<u8>()) }
    }
}

/// Non-owning shared tile, typed by its physical layout and MMA operand role.
pub struct SharedTile<L, Role> {
    /// Original shared FP16 allocation base, before selecting an array tile.
    pub(crate) base: *mut f16,
    /// Byte displacement of this tile from the original allocation base.
    pub(crate) byte_offset: u32,
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
        self.base.wrapping_add((self.byte_offset / 2) as usize)
    }

    /// Store eight contiguous FP16 values (carried as four packed `u32` words) at `(row, col)`.
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
            *self
                .as_ptr()
                .add(offset)
                .cast::<cuda_device::vector::U32x4>() = cuda_device::vector::U32x4(words);
        }
    }
}

/// Contiguous array of complete shared tiles with one layout and operand role.
pub struct SharedTileArray<L, Role, const TILES: u32> {
    /// Shared FP16 allocation containing every complete array tile.
    base: *mut f16,
    /// Carries the array's common layout and operand role.
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
    /// bytes for B128, 512 bytes for B64).
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
    /// `index` must be less than `TILES`.
    #[inline(always)]
    pub const unsafe fn tile_unchecked(&self, index: u32) -> SharedTile<L, Role> {
        SharedTile {
            base: self.base,
            byte_offset: index * Self::TILE_BYTES,
            marker: PhantomData,
        }
    }
}
