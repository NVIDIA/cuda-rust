/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Global tensor views joining packed values and scale storage.

use super::{
    __compiler, BlockScaledThreadRow, CANONICAL_SCALE_K_GROUPS, DenseBlockScaleKMajor, KMajor,
    MmaScalePair, MmaScaleQuad, Nvfp4ThreadRow, Sm1xxBlockScaleKMajor,
};
use crate::copy::cooperative::GmemMatrix;
use crate::copy::tiled::GlobalCopyTensor;
use crate::markers::LeadingDim;
use crate::numeric::{E4M3x4, Float4E2M1FN, Float8E4M3FN, Float8E8M0FNU, Mxf4E2M1, UE8M0x4};
use crate::tensor::{Tensor, TensorElement};
use core::marker::PhantomData;

/// Packed MXFP4 values and their dense UE8M0 scales for one batch (`L = 1`).
///
/// ```text
/// logical K values: [fp4 fp4 fp4 fp4] ... 32 values ...
/// value storage:    [   one f16 box   ]
/// scale storage:    [             one UE8M0 scale      ]
/// ```
///
/// The `f16` is only a 16-bit storage box for four FP4 nibbles; it is not an
/// FP16 numeric value. Scales are stored densely, one byte per 32 K values.
#[derive(Clone, Copy, Debug)]
pub struct Mxf4BlockScaledTensor<'a, Mode: Copy> {
    /// Packed value storage viewed as four FP4 values per f16 carrier.
    values: Tensor<'a, Mxf4E2M1, KMajor<Mode>, f16>,
    /// One UE8M0 scale per logical K32 group, stored densely by row.
    pub(super) scales: Tensor<'a, Float8E8M0FNU, DenseBlockScaleKMajor<Mode, 32>, u8>,
}

impl<'a, Mode: Copy> Mxf4BlockScaledTensor<'a, Mode> {
    /// View packed FP4 values and dense scale bytes as one block-scaled tensor.
    ///
    /// This does not allocate, copy, or validate memory. Later copy and load
    /// operations still require valid buffer sizes and alignment.
    #[must_use]
    #[inline(always)]
    pub const fn from_slices(
        values: &'a [f16],
        scales: &'a [u8],
        rows: usize,
        logical_k: usize,
    ) -> Self {
        Self {
            values: Tensor::from_storage(values, KMajor::new(rows, logical_k)),
            scales: Tensor::from_storage(scales, DenseBlockScaleKMajor::new(rows, logical_k)),
        }
    }

    /// Create the global-memory view used by a block-wide value copy.
    ///
    /// Logical K counts FP4 values. The returned physical view counts `f16`
    /// storage boxes, so its width is `K / 4`:
    ///
    /// ```text
    /// 128 FP4 values ──► 32 f16 storage boxes
    /// ```
    #[must_use]
    #[inline(always)]
    pub fn values_for_copy(&self) -> GlobalCopyTensor<Mxf4E2M1, f16, LeadingDim<8>, Mode> {
        let carrier_k = self.values.layout.k / 4;
        GlobalCopyTensor::from_carrier(GmemMatrix {
            base: self.values.storage.as_ptr(),
            rows: self.values.layout.rows,
            cols: carrier_k,
            leading_dim: LeadingDim::<8> {
                elements: carrier_k,
            },
        })
    }

    /// View each four neighboring scale bytes as one `u32` copy word.
    ///
    /// ```text
    /// [s0][s1][s2][s3] ──► one u32
    /// ```
    ///
    /// This does not copy or convert the bytes. The tiled-copy layout decides
    /// which rows and K=128 sections are moved together.
    ///
    /// # Safety
    ///
    /// - The scale slice must contain at least `rows * groups` bytes.
    /// - Its first address must be aligned for `u32`.
    /// - Each row's number of scale groups must be divisible by four.
    /// - The returned view must not outlive this scale slice.
    #[must_use]
    #[inline(always)]
    pub unsafe fn scale_words_for_copy(
        &self,
    ) -> GlobalCopyTensor<UE8M0x4, u32, LeadingDim<1>, Mode> {
        let words = self.scales.layout.groups / 4;
        GlobalCopyTensor::from_carrier(GmemMatrix {
            base: self.scales.storage.as_ptr().cast::<u32>(),
            rows: self.scales.layout.rows,
            cols: words,
            leading_dim: LeadingDim::<1> { elements: words },
        })
    }

    /// Load the two neighboring scale bytes used by one K=64 MMA.
    ///
    /// The result keeps `Mode`, so the MMA API can distinguish A scales from B
    /// scales.
    #[must_use]
    #[inline(always)]
    pub fn load_scale_pair(&self, row: usize, pair: usize) -> MmaScalePair<Mode> {
        let offset = self.scales.layout.crd2idx(row, pair * 2);
        MmaScalePair {
            bits: (self.scales.storage[offset] as u32)
                | ((self.scales.storage[offset + 1] as u32) << 8),
            mode: PhantomData,
        }
    }
}

/// One view that keeps packed values and their scales together.
///
/// The Rust type records the data format, scale format, number of values per
/// scale, and value layout. This prevents mixing incompatible pieces later.
#[derive(Clone, Copy, Debug)]
pub struct BlockScaledTensor<
    'a,
    Data: TensorElement<u8>,
    Scale: TensorElement<u8>,
    const SF_VEC: usize,
    DataLayout,
> {
    /// Packed byte storage with the tensor’s logical value layout.
    pub(super) values: Tensor<'a, Data, DataLayout, u8>,
    /// Scale bytes arranged in canonical Blackwell scale blocks.
    pub(super) scales: Tensor<'a, Scale, Sm1xxBlockScaleKMajor<SF_VEC>, u8>,
}

impl<'a, Mode: Copy> BlockScaledTensor<'a, Float4E2M1FN, Float8E8M0FNU, 16, KMajor<Mode>> {
    /// View packed values and Blackwell-formatted scales as one tensor.
    ///
    /// This does not validate, allocate, convert, or copy memory. The later
    /// unsafe load still requires correctly sized and aligned slices.
    #[must_use]
    #[inline(always)]
    pub const fn from_slices(values: &'a [u8], scales: &'a [u8], rows: usize, k: usize) -> Self {
        __compiler::block_scaled_make::<Mode>(values, scales, rows, k)
    }

    /// Return the number of complete K=64 tiles.
    ///
    /// Each tile has four groups of 16 values. `k` must be divisible by 64
    /// before any tile is loaded.
    #[must_use]
    #[inline(always)]
    pub fn k_tile_count(self) -> usize {
        self.values.layout.k / (16 * CANONICAL_SCALE_K_GROUPS)
    }

    /// Select one matrix or vector row in `batch` for the current thread.
    ///
    /// This calculates starting offsets only; it does not read memory.
    #[must_use]
    #[inline(always)]
    pub fn thread_row(self, batch: usize, row: usize) -> BlockScaledThreadRow<'a, Mode> {
        __compiler::block_scaled_thread_row(self, batch, row)
    }
}

impl<'a, Mode: Copy> BlockScaledTensor<'a, Float4E2M1FN, Float8E4M3FN, 16, KMajor<Mode>> {
    /// View packed `Float4E2M1FN` values and canonical `Float8E4M3FN` (`SF_VEC = 16`) scales as one NVFP4 tensor.
    #[must_use]
    #[inline(always)]
    pub const fn from_slices(values: &'a [u8], scales: &'a [u8], rows: usize, k: usize) -> Self {
        __compiler::nvfp4_make(values, scales, rows, k)
    }

    /// Return the number of complete K=64 tiles (`4` groups of 16 values per tile).
    #[must_use]
    #[inline(always)]
    pub const fn k_tile_count(self) -> usize {
        self.values.layout.k / (16 * CANONICAL_SCALE_K_GROUPS)
    }

    /// Select one matrix or vector row in `batch` for the current thread.
    #[must_use]
    #[inline(always)]
    pub const fn thread_row(self, batch: usize, row: usize) -> Nvfp4ThreadRow<'a, Mode> {
        __compiler::nvfp4_thread_row(self, batch, row)
    }
}

/// Packed NVFP4 (`Float4E2M1FN`) values and their dense `Float8E4M3FN` scales (`SF_VEC = 16`, `L = 1`).
///
/// ```text
/// logical K values: [fp4 fp4 fp4 fp4] ... 16 values ...
/// value storage:    [   one f16 box   ] (4 boxes per 16 values)
/// scale storage:    [       one E4M3 scale byte       ]
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Nvfp4BlockScaledTensor<'a, Mode: Copy> {
    /// Packed value storage viewed as four FP4 values per f16 carrier.
    values: Tensor<'a, Mxf4E2M1, KMajor<Mode>, f16>,
    /// One E4M3 scale per logical K16 group, stored densely by row.
    pub(super) scales: Tensor<'a, Float8E4M3FN, DenseBlockScaleKMajor<Mode, 16>, u8>,
}

impl<'a, Mode: Copy> Nvfp4BlockScaledTensor<'a, Mode> {
    /// View packed FP4 values and dense E4M3 scale bytes as one NVFP4 block-scaled tensor.
    #[must_use]
    #[inline(always)]
    pub const fn from_slices(
        values: &'a [f16],
        scales: &'a [u8],
        rows: usize,
        logical_k: usize,
    ) -> Self {
        Self {
            values: Tensor::from_storage(values, KMajor::new(rows, logical_k)),
            scales: Tensor::from_storage(scales, DenseBlockScaleKMajor::new(rows, logical_k)),
        }
    }

    /// Create the global-memory view used by a block-wide value copy.
    #[must_use]
    #[inline(always)]
    pub fn values_for_copy(&self) -> GlobalCopyTensor<Mxf4E2M1, f16, LeadingDim<8>, Mode> {
        let carrier_k = self.values.layout.k / 4;
        GlobalCopyTensor::from_carrier(GmemMatrix {
            base: self.values.storage.as_ptr(),
            rows: self.values.layout.rows,
            cols: carrier_k,
            leading_dim: LeadingDim::<8> {
                elements: carrier_k,
            },
        })
    }

    /// View each four neighboring E4M3 scale bytes (covering K=64) as one `u32` copy word.
    ///
    /// # Safety
    ///
    /// - The scale slice must contain at least `rows * groups` bytes and be `u32` aligned.
    /// - Each row's number of scale groups must be divisible by four.
    #[must_use]
    #[inline(always)]
    pub unsafe fn scale_words_for_copy(
        &self,
    ) -> GlobalCopyTensor<E4M3x4, u32, LeadingDim<1>, Mode> {
        let words = self.scales.layout.groups / 4;
        GlobalCopyTensor::from_carrier(GmemMatrix {
            base: self.scales.storage.as_ptr().cast::<u32>(),
            rows: self.scales.layout.rows,
            cols: words,
            leading_dim: LeadingDim::<1> { elements: words },
        })
    }

    /// Load the four neighboring `Float8E4M3FN` scale bytes used by one K=64 NVFP4 MMA.
    #[must_use]
    #[inline(always)]
    pub fn load_scale_quad(&self, row: usize, k64_step: usize) -> MmaScaleQuad<Mode> {
        let offset = self.scales.layout.crd2idx(row, k64_step * 4);
        let bits = (self.scales.storage[offset] as u32)
            | ((self.scales.storage[offset + 1] as u32) << 8)
            | ((self.scales.storage[offset + 2] as u32) << 16)
            | ((self.scales.storage[offset + 3] as u32) << 24);
        MmaScaleQuad::from_bits(bits)
    }
}
