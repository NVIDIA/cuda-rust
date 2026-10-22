/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Table-driven BF16 split-half rotation and contiguous cache append.
use ctm_rs::memory::DisjointSlice;
use ctm_rs::{cuda_module, kernel, thread};
use cute_rs::tensor::partition::{StridedTensor, StridedTensorMut};
use cute_rs::{BFloat16, RegisterTile};

/// Independent-thread rotary and append operations over checked dense BF16 storage.
#[cuda_module]
pub mod kernels {
    use super::*;

    /// Rotate dense `[tokens, heads, dim]` BF16 tensors using split-half pairs.
    ///
    /// Coefficients have shape `[tokens, dim / 2]` and are shared across heads.
    /// Each product rounds to BF16 before the sum rounds again. One linear thread
    /// owns each pair; `dim` must be positive and even, and `heads` positive. The
    /// host must validate nonoverflowing extents, ready inputs, distinct output,
    /// and allocation lifetimes. No warp or block collective is required.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn rotate_bf16(
        input: &[BFloat16],
        cos: &[BFloat16],
        sin: &[BFloat16],
        mut output: DisjointSlice<BFloat16>,
        tokens: usize,
        heads: usize,
        dim: usize,
    ) {
        let pair = thread::index_1d().get();
        let half = dim / 2;
        if pair < tokens * heads * half {
            let row = pair / half;
            let frequency = pair % half;
            let table = (row / heads) * half + frequency;
            let left = row * dim + frequency;
            // SAFETY: the host checks even head width and dense extents;
            // each pair contains coordinates separated by one half-head.
            let values = unsafe {
                StridedTensor::from_raw_parts(input.as_ptr().add(left), 2, half)
                    .load::<2>()
                    .map(BFloat16::to_f32)
            };
            let c = StridedTensor::from_slice(cos).get(table).to_f32();
            let s = StridedTensor::from_slice(sin).get(table).to_f32();
            // Eager BF16 RoPE rounds each multiplication before addition.
            // Keeping these explicit conversions also prevents FMA contraction
            // across the observable intermediate rounding boundaries.
            let cosine = values.map(|value| BFloat16::from_f32(value * c));
            let sine = values
                .gather([1, 0])
                .zip_map(RegisterTile::from_array([true, false]), |value, negate| {
                    BFloat16::from_f32((if negate { -value } else { value }) * s)
                });
            let rotated = cosine.zip_map(sine, |a, b| BFloat16::from_f32(a.to_f32() + b.to_f32()));
            // SAFETY: checked dense extents, even head width and one thread
            // per pair give two disjoint, in-bounds coordinates per thread.
            unsafe {
                StridedTensorMut::from_raw_parts(output.as_mut_ptr().add(left), 2, half)
                    .store(rotated);
            }
        }
    }

    /// Copy rotated keys and unrotated values unchanged into matching cache intervals.
    ///
    /// `start` and `count` are BF16 element offsets/counts, not token counts. The
    /// caller supplies `count` initialized source elements and writable cache
    /// intervals `start..start + count` beyond the completed prefix. Each linear
    /// thread owns one element; inputs and destinations must not alias and must
    /// remain live through completion. No arithmetic conversion or collective occurs.
    #[kernel]
    pub fn append_bf16(
        key: &[BFloat16],
        value: &[BFloat16],
        mut cache_key: DisjointSlice<BFloat16>,
        mut cache_value: DisjointSlice<BFloat16>,
        start: usize,
        count: usize,
    ) {
        let index = thread::index_1d().get();
        if index < count {
            // SAFETY: the host checked start+count against the cache capacity.
            // Each thread owns one new slot beyond the completed prefix.
            unsafe {
                let key = StridedTensor::from_slice(key).subview(index, 1);
                let value = StridedTensor::from_slice(value).subview(index, 1);
                let mut key_destination = StridedTensorMut::from_disjoint_slice(&mut cache_key)
                    .subview_unchecked(start + index, 1);
                let mut value_destination = StridedTensorMut::from_disjoint_slice(&mut cache_value)
                    .subview_unchecked(start + index, 1);
                key.map_to(&mut key_destination, |value| value);
                value.map_to(&mut value_destination, |value| value);
            }
        }
    }
}
