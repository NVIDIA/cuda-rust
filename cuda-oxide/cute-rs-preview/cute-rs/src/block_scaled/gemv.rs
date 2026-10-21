/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Per-thread GEMV rows, K64 tiles, and loaded register values.

use super::{__compiler, Mkl, Nkl};
use crate::numeric::PackedE2M1x2;
use core::marker::PhantomData;

/// One thread's view of a row whose K values are adjacent.
///
/// It stores slice references plus the first value and scale offsets. It does
/// not load the row.
#[derive(Clone, Copy, Debug)]
pub struct BlockScaledThreadRow<'a, Mode> {
    /// Packed FP4 value storage, two logical values per byte.
    pub(super) values: &'a [u8],
    /// UE8M0 scale bytes in canonical Blackwell block order.
    pub(super) scales: &'a [u8],
    /// Byte offset of this row’s first packed value.
    pub(super) value_row_base: usize,
    /// Byte offset of this row’s first scale group.
    pub(super) scale_row_base: usize,
    /// Records whether the row belongs to the M or N operand.
    pub(super) mode: PhantomData<Mode>,
}

impl<'a, Mode: Copy> BlockScaledThreadRow<'a, Mode> {
    /// Select 64 neighboring K values and their four scales.
    ///
    /// ```text
    /// row ── k_tile(0) ─► K 0..63
    ///     └─ k_tile(1) ─► K 64..127
    /// ```
    ///
    /// This calculates offsets only. Call [`BlockScaledTile64::load`] to read
    /// global memory.
    #[must_use]
    #[inline(always)]
    pub fn k_tile(self, tile: usize) -> BlockScaledTile64<'a, Mode> {
        __compiler::block_scaled_k_tile(self, tile)
    }
}

/// A view of 64 packed FP4 values and four UE8M0 scales.
///
/// Each scale covers 16 values. The view stores offsets, not loaded values.
#[derive(Clone, Copy, Debug)]
pub struct BlockScaledTile64<'a, Mode> {
    /// Borrowed packed value bytes for the complete tensor.
    pub(super) values: &'a [u8],
    /// Borrowed scale bytes in canonical Blackwell block order.
    pub(super) scales: &'a [u8],
    /// Byte offset of the selected 64-value tile.
    pub(super) value_base: usize,
    /// Byte offset of the tile’s four adjacent scale bytes.
    pub(super) scale_base: usize,
    /// Retains the operand coordinate convention without runtime data.
    pub(super) mode: PhantomData<Mode>,
}

impl<'a, Mode: Copy> BlockScaledTile64<'a, Mode> {
    /// Load the packed values and scales into this thread's registers.
    ///
    /// Each scale is converted to `f32` once and then reused by its group of 16
    /// FP4 values.
    ///
    /// # Safety
    ///
    /// - `values[value_base..value_base + 32]` must be readable.
    /// - `values[value_base]` must start at a 16-byte-aligned address.
    /// - `scales[scale_base..scale_base + 4]` must be readable.
    /// - `scales[scale_base]` must start at a 4-byte-aligned address.
    #[must_use]
    #[inline(always)]
    pub unsafe fn load(self) -> LoadedBlockScaledTile64<Mode> {
        unsafe { __compiler::block_scaled_load_k64(self) }
    }
}

/// One loaded K=64 tile held in a thread's registers.
///
/// The 64 FP4 values remain packed as 32 bytes. The four scales have already
/// been converted to `f32`.
#[derive(Clone, Copy, Debug)]
pub struct LoadedBlockScaledTile64<Mode> {
    /// First 32 FP4 values, packed two per byte.
    pub(super) values_lo: [u8; 16],
    /// Last 32 FP4 values, packed two per byte.
    pub(super) values_hi: [u8; 16],
    /// Decoded scale for logical K positions 0 through 15.
    pub(super) scale0: f32,
    /// Decoded scale for logical K positions 16 through 31.
    pub(super) scale1: f32,
    /// Decoded scale for logical K positions 32 through 47.
    pub(super) scale2: f32,
    /// Decoded scale for logical K positions 48 through 63.
    pub(super) scale3: f32,
    /// Retains the operand role while values reside in registers.
    pub(super) mode: PhantomData<Mode>,
}

impl<Mode: Copy> LoadedBlockScaledTile64<Mode> {
    /// Return packed FP4 pair `PAIR` in increasing K order.
    #[must_use]
    #[inline(always)]
    pub fn value_pair<const PAIR: usize>(self) -> PackedE2M1x2 {
        const { assert!(PAIR < 32) };
        let bits = if PAIR < 16 {
            self.values_lo[PAIR]
        } else {
            self.values_hi[PAIR - 16]
        };
        PackedE2M1x2::from_bits(bits)
    }

    /// Return the `f32` scale for one of the four 16-value groups.
    #[must_use]
    #[inline(always)]
    pub fn scale<const GROUP: usize>(self) -> f32 {
        const { assert!(GROUP < 4) };
        match GROUP {
            0 => self.scale0,
            1 => self.scale1,
            2 => self.scale2,
            _ => self.scale3,
        }
    }
}

impl LoadedBlockScaledTile64<Mkl> {
    /// Add the dot product of two K=64 tiles to `acc`.
    ///
    /// The types require matrix data ([`Mkl`]) on the left and vector data
    /// ([`Nkl`]) on the right:
    ///
    /// ```text
    /// acc + sum((A[k] × A_scale[k]) × B[k] × B_scale[k])
    /// ```
    ///
    /// Each scale is reused for 16 values. Each packed byte contributes two
    /// FP4 values in increasing K order.
    #[must_use]
    #[inline(always)]
    pub fn dot_accumulate(self, rhs: LoadedBlockScaledTile64<Nkl>, acc: f32) -> f32 {
        __compiler::block_scaled_dot_k64(self, rhs, acc)
    }
}

/// One thread's view of an NVFP4 (`E2M1` + `E4M3`, `SF_VEC = 16`) row.
///
/// It stores slice references plus the row's starting value and scale offsets.
#[derive(Clone, Copy, Debug)]
pub struct Nvfp4ThreadRow<'a, Mode> {
    /// Packed FP4 value storage, two logical values per byte.
    pub(super) values: &'a [u8],
    /// E4M3 scale bytes in canonical Blackwell block order.
    pub(super) scales: &'a [u8],
    /// Byte offset of this row's first packed value.
    pub(super) value_row_base: usize,
    /// Byte offset of this row's first scale group.
    pub(super) scale_row_base: usize,
    /// Records whether the row belongs to the M or N operand.
    pub(super) mode: PhantomData<Mode>,
}

impl<'a, Mode: Copy> Nvfp4ThreadRow<'a, Mode> {
    /// Select 64 neighboring K values and their four `E4M3` scales.
    #[must_use]
    #[inline(always)]
    pub const fn k_tile(self, tile: usize) -> Nvfp4Tile64<'a, Mode> {
        __compiler::nvfp4_k_tile(self, tile)
    }
}

/// A view of 64 packed `E2M1` values and four `E4M3` scales (`SF_VEC = 16`).
#[derive(Clone, Copy, Debug)]
pub struct Nvfp4Tile64<'a, Mode> {
    /// Borrowed packed value bytes for the complete tensor.
    pub(super) values: &'a [u8],
    /// Borrowed E4M3 scale bytes in canonical Blackwell block order.
    pub(super) scales: &'a [u8],
    /// Byte offset of the selected 64-value tile.
    pub(super) value_base: usize,
    /// Byte offset of the tile's four adjacent E4M3 scale bytes.
    pub(super) scale_base: usize,
    /// Retains the operand coordinate convention without runtime data.
    pub(super) mode: PhantomData<Mode>,
}

impl<'a, Mode: Copy> Nvfp4Tile64<'a, Mode> {
    /// Load the 64 packed `E2M1` values and decode the four `E4M3` scales to `f32`.
    ///
    /// # Safety
    ///
    /// - `values[value_base..value_base + 32]` must be readable and 16-byte aligned.
    /// - `scales[scale_base..scale_base + 4]` must be readable and 4-byte aligned.
    #[must_use]
    #[inline(always)]
    pub unsafe fn load(self) -> LoadedNvfp4Tile64<Mode> {
        unsafe { __compiler::nvfp4_load_k64(self) }
    }
}

/// 64 packed `E2M1` values and four decoded `E4M3` (`f32`) scales for one K=64 tile.
#[derive(Clone, Copy, Debug)]
pub struct LoadedNvfp4Tile64<Mode> {
    /// First 16 packed bytes, covering K values 0..31.
    pub(super) values_lo: [u8; 16],
    /// Second 16 packed bytes, covering K values 32..63.
    pub(super) values_hi: [u8; 16],
    /// Decoded E4M3 scale for K values 0..15.
    pub(super) scale0: f32,
    /// Decoded E4M3 scale for K values 16..31.
    pub(super) scale1: f32,
    /// Decoded E4M3 scale for K values 32..47.
    pub(super) scale2: f32,
    /// Decoded E4M3 scale for K values 48..63.
    pub(super) scale3: f32,
    /// Retains the operand coordinate convention without runtime data.
    pub(super) mode: PhantomData<Mode>,
}

impl LoadedNvfp4Tile64<Mkl> {
    /// Add the dot product of two K=64 NVFP4 tiles (`E2M1` + `E4M3`, `SF_VEC = 16`) to `acc`.
    #[must_use]
    #[inline(always)]
    pub fn dot_accumulate(self, rhs: LoadedNvfp4Tile64<Nkl>, acc: f32) -> f32 {
        __compiler::nvfp4_dot_k64(self, rhs, acc)
    }
}
