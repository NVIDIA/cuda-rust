/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! BF16 tensor operations for the checked Qwen decoder composition.

use ctm_rs::memory::DisjointSlice;
use ctm_rs::{cuda_module, kernel, thread};
use cute_rs::Bf16;
use cute_rs::tensor::partition::{StridedTensor, StridedTensorMut};

/// Dense BF16 primitives; the host validates extents and keeps output disjoint.
///
/// Each active global thread owns one output element or reduction column;
/// these kernels do not require warp or CTA collectives.
#[cuda_module]
pub mod kernels {
    use super::*;

    /// Gather dense embedding rows after the host has checked every token ID.
    /// `ids` contains `rows` valid table row indices; each row has `width` BF16
    /// values. Output has `rows * width` elements and may not alias the table.
    #[kernel]
    pub fn embedding_bf16(
        table: &[Bf16],
        ids: &[u32],
        mut output: DisjointSlice<Bf16>,
        rows: usize,
        width: usize,
    ) {
        let index = thread::index_1d().get();
        if index < rows * width {
            let row = index / width;
            let column = index % width;
            let ids = StridedTensor::from_slice(ids);
            let table = StridedTensor::from_slice(table);
            let mut output = StridedTensorMut::from_disjoint_slice(&mut output);
            // SAFETY: the checked host boundary validates all token IDs and
            // extents; each global thread owns one distinct destination.
            unsafe {
                let source = table.subview_unchecked(ids.get(row) as usize * width, width);
                output.store_at(index, source.get_unchecked(column));
            }
        }
    }

    /// Add matching tensor partitions and round once to BF16.
    /// Both inputs and the separate output provide at least `length` BF16 values;
    /// addition widens operands to FP32 before narrowing the result.
    #[kernel]
    pub fn add_bf16(left: &[Bf16], right: &[Bf16], mut output: DisjointSlice<Bf16>, length: usize) {
        let index = thread::index_1d().get();
        if index < length {
            // Scalar tiles retain the existing launch assignment. The same
            // zip-map operation also supports multi-element strided partitions.
            // SAFETY: matching extents and one distinct output per global thread.
            unsafe {
                let left = StridedTensor::from_slice(left).subview_unchecked(index, 1);
                let right = StridedTensor::from_slice(right).subview_unchecked(index, 1);
                let mut output =
                    StridedTensorMut::from_disjoint_slice(&mut output).subview_unchecked(index, 1);
                left.zip_map_to(right, &mut output, |a, b| {
                    Bf16::from_f32(a.to_f32() + b.to_f32())
                });
            }
        }
    }

    /// Evaluate SiLU in FP32 and store the rounded BF16 result.
    /// Input and separate output provide `length` values; each active thread
    /// evaluates one dense coordinate through the cute-rs numerical operation.
    #[kernel]
    pub fn silu_bf16(input: &[Bf16], mut output: DisjointSlice<Bf16>, length: usize) {
        let index = thread::index_1d().get();
        if index < length {
            // SAFETY: checked equal extents and disjoint global-thread tiles.
            unsafe {
                let input = StridedTensor::from_slice(input).subview_unchecked(index, 1);
                let mut output =
                    StridedTensorMut::from_disjoint_slice(&mut output).subview_unchecked(index, 1);
                input.map_to(&mut output, cute_rs::numeric::silu_bf16);
            }
        }
    }

    /// Concatenate two dense storage ranges without changing their BF16 bits.
    /// The left range contains `left_length` values and the right contains
    /// `total_length - left_length`; output is a separate `total_length` range.
    #[kernel]
    pub fn concat_bf16(
        left: &[Bf16],
        right: &[Bf16],
        mut output: DisjointSlice<Bf16>,
        left_length: usize,
        total_length: usize,
    ) {
        let index = thread::index_1d().get();
        if index < total_length {
            // SAFETY: host checked both ranges; each thread owns its output tile.
            unsafe {
                let source = if index < left_length {
                    StridedTensor::from_slice(left).subview_unchecked(index, 1)
                } else {
                    StridedTensor::from_slice(right).subview_unchecked(index - left_length, 1)
                };
                let mut output =
                    StridedTensorMut::from_disjoint_slice(&mut output).subview_unchecked(index, 1);
                source.map_to(&mut output, |value| value);
            }
        }
    }

    /// Reduce columns in ascending row order, rounding only the final sum.
    /// Input is dense `[rows, width]`; the separate output contains `width`
    /// values. FP32 sums start at zero, so an empty row range produces zeros.
    #[kernel]
    pub fn sum_rows_bf16(
        input: &[Bf16],
        mut output: DisjointSlice<Bf16>,
        rows: usize,
        width: usize,
    ) {
        let column = thread::index_1d().get();
        if column < width {
            // SAFETY: the host checks the dense rows*width extent. Empty
            // reductions retain the original pointer without offsetting it.
            let column_view = unsafe {
                let base = if rows == 0 {
                    input.as_ptr()
                } else {
                    input.as_ptr().add(column)
                };
                StridedTensor::from_raw_parts(base, rows, width)
            };
            let sum = column_view.fold(0.0_f32, |sum, value| sum + value.to_f32());
            let mut output = StridedTensorMut::from_disjoint_slice(&mut output);
            // SAFETY: each thread owns one checked output column.
            unsafe {
                output.store_at(column, Bf16::from_f32(sum));
            }
        }
    }
}
