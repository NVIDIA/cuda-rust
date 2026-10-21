/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! BF16 linear and activation kernels used by the checked host composition.
use ctm_rs::memory::DisjointSlice;
use ctm_rs::{cuda_module, kernel, launch_contract, thread};
use cute_rs::Bf16;

/// Device entries for BF16 projections and activation, shared by FFN and decoder examples.
///
/// Matrix inputs use element strides; weights store `[output, input]`, while
/// outputs are dense. Host runners must prove bounds, context readiness, distinct
/// mutable storage, and convergent warp participation before launching.
#[cuda_module]
pub mod kernels {
    use super::*;
    use cute_rs::mma::bf16::{Bf16Mma16x8x16, MmaAccumulator, PackedDecode};
    use cute_rs::numeric::silu_bf16 as silu;

    /// Read `(row, col)` from an element-strided row-major BF16 matrix.
    ///
    /// Missing logical rows or columns return positive zero for MMA tail padding.
    /// In-range coordinates require `data` to cover `row * stride + col`; this
    /// helper does not independently validate the allocation or row stride.
    #[inline(always)]
    fn element(
        data: &[Bf16],
        row: usize,
        col: usize,
        rows: usize,
        cols: usize,
        stride: usize,
    ) -> Bf16 {
        if row < rows && col < cols {
            data[row * stride + col]
        } else {
            Bf16::from_bits(0)
        }
    }

    /// Project a short input batch using checked packed BF16 MMA operands.
    ///
    /// One 32-thread CTA owns eight output columns; K16 MMA steps accumulate in
    /// FP32. When enabled, bias is added in FP32 before one BF16 output rounding.
    ///
    /// # Safety
    /// The entire warp must enter with uniform dimensions and the same validated
    /// `source`. `width` and `outputs` must match that view. Output must contain
    /// every dense `[source rows, outputs]` element, remain distinct from inputs,
    /// and have only this grid's partitioned writers. With `HAS_BIAS`, `bias`
    /// must contain `outputs` initialized elements. Sources must be ready, and
    /// all allocations must remain live until the kernel finishes.
    #[inline(always)]
    unsafe fn packed_decode_impl<const HAS_BIAS: bool>(
        source: &PackedDecode<'_>,
        bias: &[Bf16],
        output: &mut DisjointSlice<Bf16>,
        width: usize,
        outputs: usize,
    ) {
        let lane = thread::threadIdx_x() as usize;
        let n = thread::blockIdx_x() as usize * 8;
        if n >= outputs {
            return;
        }
        let mma = Bf16Mma16x8x16::get_slice(lane);
        let a = mma.partition_a();
        let b = mma.partition_b();
        let mut accumulator = MmaAccumulator::zero();
        let mut k = 0;
        while k < width {
            // SAFETY: PackedDecode validates source extents/alignment; N and K
            // select full tiles. Every lane reaches the same instruction order.
            unsafe {
                if width - k >= 512 {
                    // Fetch 32 K16 operand pairs into a typed register tile so
                    // independent loads can overlap before their consumers.
                    // The fold visits ascending K: each MMA sees exactly the
                    // preceding accumulator, preserving FP32 rounding order.
                    let operands = cute_rs::tensor::RegisterTile::<_, 32>::from_fn(|step| {
                        (
                            a.load_decode_packed(source, k + step * 16),
                            b.load_decode_packed(source, n, k + step * 16),
                        )
                    });
                    accumulator = operands.fold(accumulator, |acc, (av, bv)| mma.mma(acc, av, bv));
                    k += 512;
                } else {
                    let av = a.load_decode_packed(source, k);
                    let bv = b.load_decode_packed(source, n, k);
                    accumulator = mma.mma(accumulator, av, bv);
                    k += 16;
                }
            }
        }
        let mut output = cute_rs::tensor::partition::StridedTensorMut::from_disjoint_slice(output);
        mma.partition_c()
            .store_decode_with(accumulator, source, |row, col, value| {
                let value = if HAS_BIAS {
                    value + bias[n + col].to_f32()
                } else {
                    value
                };
                // SAFETY: C's bijective layout and the checked grid give each lane
                // disjoint initialized output coordinates, with one BF16 rounding.
                unsafe {
                    output.store_at(row * outputs + n + col, Bf16::from_f32(value));
                }
            });
    }

    /// Project strided BF16 matrices through typed 16×8×16 warp MMA partitions.
    ///
    /// Input is `[rows, width]`, weights are `[outputs, width]`, and strides count
    /// elements. Missing M/N/K coordinates contribute zero; output is dense
    /// `[rows, outputs]`. `PREFETCH` loads four K16 steps ahead without changing
    /// accumulation order. Bias, when enabled, is added before BF16 rounding.
    ///
    /// The enclosing launch must provide exactly `32 * WARPS_PER_BLOCK` threads,
    /// complete convergent warps, valid strided source extents, and distinct
    /// writable output storage. `outputs` must be positive, grid arithmetic must
    /// fit, and `HAS_BIAS` requires at least `outputs` bias values.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn linear_bf16_impl<
        const HAS_BIAS: bool,
        const WARPS_PER_BLOCK: usize,
        const PREFETCH: bool,
    >(
        input: &[Bf16],
        weight: &[Bf16],
        bias: &[Bf16],
        mut output: DisjointSlice<Bf16>,
        rows: usize,
        width: usize,
        outputs: usize,
        input_stride: usize,
        weight_stride: usize,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let lane = tid % 32;
        let tile = thread::blockIdx_x() as usize * WARPS_PER_BLOCK + tid / 32;
        let columns = outputs.div_ceil(8);
        if tile < rows.div_ceil(16) * columns {
            let m = tile / columns * 16;
            let n = tile % columns * 8;
            let mma = Bf16Mma16x8x16::get_slice(lane);
            let load_a = |k| {
                mma.partition_a()
                    .load_with(|r, c| element(input, m + r, k + c, rows, width, input_stride))
            };
            let load_b = |k| {
                mma.partition_b()
                    .load_with(|r, c| element(weight, n + r, k + c, outputs, width, weight_stride))
            };
            let mut accumulator = MmaAccumulator::zero();
            let mut k = 0;
            while k < width {
                if PREFETCH && width - k >= 64 {
                    let a0 = load_a(k);
                    let b0 = load_b(k);
                    let a1 = load_a(k + 16);
                    let b1 = load_b(k + 16);
                    let a2 = load_a(k + 32);
                    let b2 = load_b(k + 32);
                    let a3 = load_a(k + 48);
                    let b3 = load_b(k + 48);
                    // SAFETY: the full warp uses the atom's typed fragments;
                    // arithmetic consumes K tiles in unchanged increasing order.
                    unsafe {
                        accumulator = mma.mma(accumulator, a0, b0);
                        accumulator = mma.mma(accumulator, a1, b1);
                        accumulator = mma.mma(accumulator, a2, b2);
                        accumulator = mma.mma(accumulator, a3, b3);
                    }
                    k += 64;
                    continue;
                }
                let a = load_a(k);
                let b = load_b(k);
                // SAFETY: no lane skips the collective; tails are zero-padded.
                accumulator = unsafe { mma.mma(accumulator, a, b) };
                k += 16;
            }
            let mut output =
                cute_rs::tensor::partition::StridedTensorMut::from_disjoint_slice(&mut output);
            mma.partition_c().store_with(accumulator, |r, c, value| {
                if m + r < rows && n + c < outputs {
                    let value = if HAS_BIAS {
                        value + bias[n + c].to_f32()
                    } else {
                        value
                    };
                    // SAFETY: C's bijective layout and the host-checked tile grid
                    // give disjoint output writes within the validated extent.
                    unsafe {
                        output.store_at((m + r) * outputs + n + c, Bf16::from_f32(value));
                    }
                }
            });
        }
    }

    /// Compute a bias-free projection with FP32 accumulation and BF16 output.
    ///
    /// Launch 128 threads per block, four complete warps, with one warp per 16×8
    /// output tile. The caller validates `[rows, width]` input, `[outputs, width]`
    /// weights, element strides, and dense `[rows, outputs]` nonaliasing output.
    /// Inputs must be ready and all borrows retained until completion; tails use
    /// zero padding without skipping any lane of a participating warp.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn linear_bf16(
        input: &[Bf16],
        weight: &[Bf16],
        output: DisjointSlice<Bf16>,
        rows: usize,
        width: usize,
        outputs: usize,
        input_stride: usize,
        weight_stride: usize,
    ) {
        // The const-false specialization never reads the placeholder bias slice.
        linear_bf16_impl::<false, 4, false>(
            input,
            weight,
            input,
            output,
            rows,
            width,
            outputs,
            input_stride,
            weight_stride,
        );
    }

    /// Add the projection bias in FP32 before rounding the output once to BF16.
    ///
    /// Uses the same 128-thread launch, strided matrix layout, readiness, and
    /// disjoint-output requirements as `linear_bf16`. Bias contains one BF16
    /// value per output column and remains live throughout the projection.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn linear_bias_bf16(
        input: &[Bf16],
        weight: &[Bf16],
        bias: &[Bf16],
        output: DisjointSlice<Bf16>,
        rows: usize,
        width: usize,
        outputs: usize,
        input_stride: usize,
        weight_stride: usize,
    ) {
        linear_bf16_impl::<true, 4, false>(
            input,
            weight,
            bias,
            output,
            rows,
            width,
            outputs,
            input_stride,
            weight_stride,
        );
    }

    /// The same MMA arithmetic with one warp per CTA, distributing short-row
    /// decode projections over more SMs without changing reduction order.
    ///
    /// The 32-thread launch contract supplies one full warp. Checked packed
    /// operands are used when available; otherwise the same element-strided
    /// `[rows, width]` / `[outputs, width]` matrices use predicated loads. The
    /// caller must validate those fallback extents, dense output capacity, and
    /// readiness and ownership exactly as for `linear_bf16`.
    #[launch_contract(domain=1, block=(32,1,1), min_compute_capability=(8,0))]
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn decode_linear_bf16(
        input: &[Bf16],
        weight: &[Bf16],
        mut output: DisjointSlice<Bf16>,
        rows: usize,
        width: usize,
        outputs: usize,
        input_stride: usize,
        weight_stride: usize,
    ) {
        if outputs != 0
            && rows <= output.len() / outputs
            && PackedDecode::with_validated(
                input,
                weight,
                rows,
                width,
                outputs,
                input_stride,
                weight_stride,
                |source| {
                    // SAFETY: the borrowed proof establishes packed operand
                    // bounds; output capacity is checked above, and this entry's
                    // launch contract supplies exactly one complete warp.
                    unsafe {
                        packed_decode_impl::<false>(source, input, &mut output, width, outputs);
                    }
                },
            )
        {
            return;
        }
        linear_bf16_impl::<false, 1, true>(
            input,
            weight,
            input,
            output,
            rows,
            width,
            outputs,
            input_stride,
            weight_stride,
        );
    }

    /// Decode projection with bias added to FP32 accumulators before BF16 output.
    ///
    /// Uses the one-warp packed-or-predicated schedule of `decode_linear_bf16`.
    /// Bias must contain `outputs` initialized BF16 values; all matrix bounds,
    /// source readiness, output nonaliasing, and full-warp participation apply.
    #[launch_contract(domain=1, block=(32,1,1), min_compute_capability=(8,0))]
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn decode_linear_bias_bf16(
        input: &[Bf16],
        weight: &[Bf16],
        bias: &[Bf16],
        mut output: DisjointSlice<Bf16>,
        rows: usize,
        width: usize,
        outputs: usize,
        input_stride: usize,
        weight_stride: usize,
    ) {
        if outputs != 0
            && rows <= output.len() / outputs
            && bias.len() >= outputs
            && PackedDecode::with_validated(
                input,
                weight,
                rows,
                width,
                outputs,
                input_stride,
                weight_stride,
                |source| {
                    // SAFETY: the source proof, output/bias extent checks and
                    // exact full-warp launch contract establish packed loads,
                    // disjoint stores, and collective participation.
                    unsafe {
                        packed_decode_impl::<true>(source, bias, &mut output, width, outputs);
                    }
                },
            )
        {
            return;
        }
        linear_bf16_impl::<true, 1, true>(
            input,
            weight,
            bias,
            output,
            rows,
            width,
            outputs,
            input_stride,
            weight_stride,
        );
    }

    /// Compute BF16 SiLU and gated products with a distinct rounding after each.
    ///
    /// The first `length` values of gate/up inputs and both outputs must exist
    /// and use distinct output allocations. Each linear thread owns eight BF16
    /// elements; complete tiles require 16-byte-aligned storage and tails use
    /// scalar accesses. No collectives occur. The caller establishes readiness
    /// and keeps all allocations live until completion.
    #[kernel]
    pub fn activate_bf16(
        gate: &[Bf16],
        up: &[Bf16],
        mut silu_output: DisjointSlice<Bf16>,
        mut product: DisjointSlice<Bf16>,
        length: usize,
    ) {
        let tid = thread::index_1d().get();
        let gates = cute_rs::Tensor::from_slice(gate)
            .zipped_divide::<8>()
            .slice(tid);
        let ups = cute_rs::Tensor::from_slice(up)
            .zipped_divide::<8>()
            .slice(tid);
        let mut silus = cute_rs::TensorMut::from_disjoint_slice(&mut silu_output)
            .zipped_divide::<8>()
            .slice(tid);
        let mut products = cute_rs::TensorMut::from_disjoint_slice(&mut product)
            .zipped_divide::<8>()
            .slice(tid);
        let base = gates.base();
        if base < length && length - base >= 8 {
            // SAFETY: checked buffers contain `length` values; bases are
            // aligned multiples of 8, and each thread owns one disjoint tile.
            let values = unsafe { gates.load() }.map(silu);
            let multiplied = values.zip_map(unsafe { ups.load() }, |s, u| {
                Bf16::from_f32(s.to_f32() * u.to_f32())
            });
            unsafe {
                silus.store(values);
                products.store(multiplied);
            }
        } else {
            let mut index = base;
            while index < length {
                let s = silu(gate[index]);
                let value = Bf16::from_f32(s.to_f32() * up[index].to_f32());
                unsafe {
                    silus.store_linear(index, s);
                    products.store_linear(index, value);
                }
                index += 1;
            }
        }
    }
}
