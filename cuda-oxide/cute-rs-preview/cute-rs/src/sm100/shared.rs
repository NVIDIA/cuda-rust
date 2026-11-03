/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared-memory tiles for paired two-CTA GEMM.

use crate::block_scaled::Sm120ScaleAtom;
use crate::markers::ReifySmem2D;
use core::marker::PhantomData;

/// A non-owning shared tile with a compile-time layout and an operand role (`OperandA` or `OperandB`).
/// Carries `Layout` and `Role` in the type system so MMA and TMA operations reject mismatched
/// tile shapes or swapped A/B operands at compile time.
pub struct Sm100SharedTile<Layout, Role> {
    /// Non-owning shared allocation base pointer (`addrspace(3)` / generic shared pointer).
    /// Reinterpreted as bytes when wrapping packed 4-bit FP4 (`E2M1`) tiles.
    pub(super) base: *mut f16,
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

    /// Return the CTA-shared byte offset of this tile's base pointer.
    /// Used by `Sm100BlockScaledTiledMma` to encode hardware `Tcgen05SmemDescriptor`s.
    ///
    /// # Safety
    /// `self.base` must point to valid shared memory belonging to the calling CTA.
    #[must_use]
    #[inline(always)]
    pub unsafe fn shared_offset(&self) -> u64 {
        unsafe { ctm_rs::memory::shared_offset(self.base.cast::<u8>()) }
    }
}

/// A non-owning shared-memory view of staged 512-byte [`Sm120ScaleAtom`] blocks with an A/B role (`Mkl` or `Nkl`).
/// Prevents swapping A-side weight scales (`Mkl`) and B-side activation scales (`Nkl`) when staging
/// scales into TMEM via `tcgen05.cp.32x128b.warpx4`.
pub struct Sm100ScaleTile<Role> {
    /// Non-owning shared `u32` scale-word base pointer (`128` words = `512` bytes per scale atom).
    /// Must be 128-byte aligned in shared memory for `tcgen05.cp`.
    pub(super) base: *mut u32,
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

    /// Return the CTA-shared byte offset of this scale tile's base pointer.
    /// Used to build the `Tcgen05SmemDescriptor` consumed by `tcgen05.cp.32x128b.warpx4`.
    ///
    /// # Safety
    /// `self.base` must point to valid shared memory belonging to the calling CTA.
    #[must_use]
    #[inline(always)]
    pub unsafe fn shared_offset(&self) -> u64 {
        unsafe { ctm_rs::memory::shared_offset(self.base.cast::<u8>()) }
    }
}
