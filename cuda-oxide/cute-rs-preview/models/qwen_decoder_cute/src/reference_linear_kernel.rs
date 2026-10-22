/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Decode projections with the reference's cuBLAS reduction order.
//!
//! The reference runs `torch.nn.functional.linear` on BF16 tensors. For one
//! input row cuBLAS selects an FP32 FMA kernel: each output keeps `CHAINS`
//! accumulators over `k mod CHAINS`, visits `k` in ascending order, merges the
//! chains with XOR offsets `CHAINS/2 .. 1`, and rounds once to BF16. Two
//! shapes use a tensor-core kernel with split K instead: every slice is its
//! own K16 MMA chain, and the FP32 slice sums are added in slice order before
//! the single BF16 rounding. `reference_linear` chooses the entry per shape.
use ctm_rs::memory::DisjointSlice;
use ctm_rs::{cuda_module, kernel, launch_contract, thread};
use cute_rs::BFloat16;

/// Device entries reproducing the reference reduction order for decode shapes.
///
/// The host validates extents, strides, alignment, readiness and disjoint
/// output ownership before every launch; the entries recheck the same
/// predicates and leave the output untouched when they fail.
#[cuda_module]
pub mod kernels {
    use super::*;
    use cute_rs::mma::bf16::{APartition, BPartition, Bf16Mma16x8x16, FragmentC, PackedDecode};
    use cute_rs::tensor::partition::StridedTensorMut;
    use cute_rs::{RegisterTile, Tensor};

    /// BF16 elements each lane loads per step: one 8-byte vector.
    pub const VECTOR: usize = 4;
    /// Steps fetched into registers before they are accumulated.
    const BATCH: usize = 16;

    /// Lanes sharing one output under `chains` FP32 chains.
    pub const fn lanes_per_output(chains: usize) -> usize {
        chains / VECTOR
    }

    /// Outputs one 32-lane warp produces under `chains` FP32 chains.
    pub const fn outputs_per_warp(chains: usize) -> usize {
        32 / lanes_per_output(chains)
    }

    /// Return whether the FMA-chain entry accepts these extents and strides.
    ///
    /// `width` must be a positive multiple of `chains`; `weight_stride` must
    /// cover `width`; both operands must be 8-byte aligned, and every slice
    /// must cover its dense extent. A pitch that is not a multiple of
    /// [`VECTOR`] selects scalar weight loads with the same arithmetic.
    #[allow(clippy::too_many_arguments)]
    pub fn fma_chains_supported(
        chains: usize,
        input: &[BFloat16],
        weight: &[BFloat16],
        output_len: usize,
        bias_len: Option<usize>,
        width: usize,
        outputs: usize,
        weight_stride: usize,
    ) -> bool {
        width != 0
            && width.is_multiple_of(chains)
            && outputs != 0
            && weight_stride >= width
            && (input.as_ptr() as usize).is_multiple_of(8)
            && (weight.as_ptr() as usize).is_multiple_of(8)
            && input.len() >= width
            && outputs <= weight.len() / weight_stride
            && output_len >= outputs
            && bias_len.is_none_or(|len| len >= outputs)
    }

    /// Return whether the split-K entry accepts these extents and the slice width.
    ///
    /// This mirrors the packed decode operand proof: one or two rows, a positive
    /// multiple of sixteen for `width` and `slice_width`, eight-column groups,
    /// even strides covering `width`, 4-byte aligned operands and dense extents.
    #[allow(clippy::too_many_arguments)]
    pub fn split_k_supported(
        input: &[BFloat16],
        weight: &[BFloat16],
        output_len: usize,
        rows: usize,
        width: usize,
        outputs: usize,
        input_stride: usize,
        weight_stride: usize,
        slice_width: usize,
    ) -> bool {
        matches!(rows, 1 | 2)
            && width != 0
            && width.is_multiple_of(16)
            && slice_width != 0
            && slice_width.is_multiple_of(16)
            && outputs != 0
            && outputs.is_multiple_of(8)
            && input_stride >= width
            && weight_stride >= width
            && input_stride.is_multiple_of(2)
            && weight_stride.is_multiple_of(2)
            && (input.as_ptr() as usize).is_multiple_of(4)
            && (weight.as_ptr() as usize).is_multiple_of(4)
            && rows <= input.len() / input_stride
            && outputs <= weight.len() / weight_stride
            && rows <= output_len / outputs
    }

    /// Add one step's four exact products to the lane's four chains.
    ///
    /// A BF16 product is exact in FP32, so a fused or a separate multiply-add
    /// rounds identically: only the FP32 addition rounds.
    #[inline(always)]
    fn accumulate(
        acc: [f32; VECTOR],
        x: RegisterTile<BFloat16, VECTOR>,
        w: RegisterTile<BFloat16, VECTOR>,
    ) -> [f32; VECTOR] {
        let x = x.into_array();
        let w = w.into_array();
        [
            w[0].to_f32() * x[0].to_f32() + acc[0],
            w[1].to_f32() * x[1].to_f32() + acc[1],
            w[2].to_f32() * x[2].to_f32() + acc[2],
            w[3].to_f32() * x[3].to_f32() + acc[3],
        ]
    }

    /// Load one lane's four consecutive input values with one 8-byte read.
    ///
    /// The view is built here rather than captured by a closure so the CuTe
    /// compiler-only value stays a direct SSA operand.
    ///
    /// # Safety
    /// `offset` is a multiple of [`VECTOR`], `offset + VECTOR <= input.len()`,
    /// and the slice is 8-byte aligned.
    #[inline(always)]
    unsafe fn load_input(input: &[BFloat16], offset: usize) -> RegisterTile<BFloat16, VECTOR> {
        unsafe {
            Tensor::from_slice(input)
                .zipped_divide::<VECTOR>()
                .slice(offset / VECTOR)
                .load()
        }
    }

    /// Load one lane's four consecutive weights, by vector or scalar reads.
    ///
    /// # Safety
    /// `offset .. offset + VECTOR` lies inside `weight`; with `ALIGNED`, the
    /// offset is a multiple of [`VECTOR`] and the slice is 8-byte aligned.
    #[inline(always)]
    unsafe fn load_weights<const ALIGNED: bool>(
        weight: &[BFloat16],
        offset: usize,
    ) -> RegisterTile<BFloat16, VECTOR> {
        if ALIGNED {
            unsafe {
                Tensor::from_slice(weight)
                    .zipped_divide::<VECTOR>()
                    .slice(offset / VECTOR)
                    .load()
            }
        } else {
            RegisterTile::from_fn(|i| weight[offset + i])
        }
    }

    /// One-row projection with `CHAINS` interleaved FP32 chains per output.
    ///
    /// `CHAINS / VECTOR` lanes share one output; lane `m` of that group owns
    /// residues `m * VECTOR .. m * VECTOR + VECTOR` of `k mod CHAINS` and visits
    /// them in ascending `k`. The merge then walks XOR offsets from `CHAINS/2`
    /// down to one: offsets of at least `VECTOR` exchange registers between
    /// lanes, smaller offsets add inside the lane. Lane zero of each group
    /// adds the optional bias in FP32 and rounds once to BF16. `ALIGNED`
    /// selects 8-byte weight loads; the scalar path computes the same bits.
    ///
    /// # Safety
    /// One complete warp enters with uniform arguments that satisfy
    /// [`fma_chains_supported`], and `ALIGNED` only when `weight_stride` is a
    /// multiple of [`VECTOR`]. Operands are ready, `output` has no other writer
    /// for these columns, and every owner outlives the launch.
    #[inline(always)]
    unsafe fn fma_chains_impl<const CHAINS: usize, const HAS_BIAS: bool, const ALIGNED: bool>(
        input: &[BFloat16],
        weight: &[BFloat16],
        bias: &[BFloat16],
        output: &mut DisjointSlice<BFloat16>,
        width: usize,
        outputs: usize,
        weight_stride: usize,
    ) {
        let lanes = lanes_per_output(CHAINS);
        let lane = thread::threadIdx_x() as usize;
        let member = lane % lanes;
        let n = thread::blockIdx_x() as usize * outputs_per_warp(CHAINS) + lane / lanes;
        // Lanes past the last output keep loading a valid row and keep taking
        // part in every shuffle; only valid outputs are stored.
        let valid = n < outputs;
        let row = if valid { n } else { outputs - 1 };
        let row_base = row * weight_stride;
        let steps = width / CHAINS;
        let mut acc = [0.0f32; VECTOR];
        let mut step = 0;
        while step < steps {
            if steps - step >= BATCH {
                // SAFETY: the supported predicate makes every selected input
                // tile complete and 8-byte aligned and every weight window in
                // bounds; all lanes execute the same loads.
                let operands = RegisterTile::<_, BATCH>::from_fn(|j| unsafe {
                    let k = (step + j) * CHAINS + member * VECTOR;
                    (
                        load_input(input, k),
                        load_weights::<ALIGNED>(weight, row_base + k),
                    )
                });
                acc = operands.fold(acc, |acc, (xv, wv)| accumulate(acc, xv, wv));
                step += BATCH;
            } else {
                let k = step * CHAINS + member * VECTOR;
                // SAFETY: as above for the final partial batch.
                let (xv, wv) = unsafe {
                    (
                        load_input(input, k),
                        load_weights::<ALIGNED>(weight, row_base + k),
                    )
                };
                acc = accumulate(acc, xv, wv);
                step += 1;
            }
        }
        // Residue offsets CHAINS/2 .. VECTOR pair lanes `member` and
        // `member ^ (offset / VECTOR)`; both partners compute the same sum.
        let mut mask = lanes / 2;
        while mask >= 1 {
            acc = [
                acc[0] + ctm_rs::warp::shuffle_xor_f32(acc[0], mask as u32),
                acc[1] + ctm_rs::warp::shuffle_xor_f32(acc[1], mask as u32),
                acc[2] + ctm_rs::warp::shuffle_xor_f32(acc[2], mask as u32),
                acc[3] + ctm_rs::warp::shuffle_xor_f32(acc[3], mask as u32),
            ];
            mask /= 2;
        }
        // Residue offsets two and one stay inside the lane.
        let total = (acc[0] + acc[2]) + (acc[1] + acc[3]);
        if valid && member == 0 {
            let value = if HAS_BIAS {
                total + bias[n].to_f32()
            } else {
                total
            };
            // SAFETY: exactly one lane writes each valid output column.
            unsafe {
                StridedTensorMut::from_disjoint_slice(output)
                    .store_at(n, BFloat16::from_f32(value));
            }
        }
    }

    /// Load `STEPS` consecutive K16 operand pairs, then fold them into the chain.
    ///
    /// # Safety
    /// `k .. k + 16 * STEPS` lies within the validated packed source, `n` is
    /// this CTA's eight-column group, and all lanes call this convergently.
    #[inline(always)]
    unsafe fn chain_batch<const STEPS: usize>(
        mma: Bf16Mma16x8x16,
        a: APartition,
        b: BPartition,
        source: &PackedDecode<'_>,
        n: usize,
        k: usize,
        accumulator: FragmentC,
    ) -> FragmentC {
        unsafe {
            let operands = RegisterTile::<_, STEPS>::from_fn(|step| {
                (
                    a.load_decode_packed(source, k + step * 16),
                    b.load_decode_packed(source, n, k + step * 16),
                )
            });
            operands.fold(accumulator, |acc, (av, bv)| mma.mma_atom_call(acc, av, bv))
        }
    }

    /// Add one slice's accumulator registers to the running FP32 totals.
    ///
    /// Register order is the C partition's; slice zero lands exactly on the
    /// zero totals and every later slice rounds once per register.
    #[inline(always)]
    fn add_slice(mma: Bf16Mma16x8x16, total: [f32; 4], accumulator: FragmentC) -> [f32; 4] {
        let (mut first, mut second, mut third, mut fourth) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let mut index = 0;
        mma.partition_c().store_with(accumulator, |_, _, value| {
            match index {
                0 => first = value,
                1 => second = value,
                2 => third = value,
                _ => fourth = value,
            }
            index += 1;
        });
        [
            total[0] + first,
            total[1] + second,
            total[2] + third,
            total[3] + fourth,
        ]
    }

    /// Tensor-core K16 chains per K slice, with FP32 slice sums added in order.
    ///
    /// Every slice restarts from a zero accumulator exactly like a split-K
    /// partition; the running FP32 total adds slice sums in ascending slice
    /// order, adds the optional bias, and is rounded once to BF16 by the
    /// ordinary decode epilogue.
    ///
    /// # Safety
    /// One complete warp enters with the same validated `source`, and
    /// `slice_width` is a positive multiple of sixteen. With `HAS_BIAS`, `bias`
    /// holds `outputs` values. Output covers the dense `[source rows, outputs]`
    /// extent with no competing writer, and every owner outlives the launch.
    #[inline(always)]
    unsafe fn packed_split_k_impl<const HAS_BIAS: bool>(
        source: &PackedDecode<'_>,
        bias: &[BFloat16],
        output: &mut DisjointSlice<BFloat16>,
        width: usize,
        outputs: usize,
        slice_width: usize,
    ) {
        let lane = thread::threadIdx_x() as usize;
        let n = thread::blockIdx_x() as usize * 8;
        if n >= outputs {
            return;
        }
        let mma = Bf16Mma16x8x16::get_slice(lane);
        let a = mma.partition_a();
        let b = mma.partition_b();
        let mut total = [0.0f32; 4];
        let mut start = 0;
        while start < width {
            let end = if width - start > slice_width {
                start + slice_width
            } else {
                width
            };
            // Each slice restarts from zero like a split-K partition. Register
            // batches never cross a slice boundary and only overlap loads; every
            // MMA still sees the preceding accumulator of its own chain.
            let mut accumulator = FragmentC::ZERO;
            let mut k = start;
            while k < end {
                let remaining = end - k;
                // SAFETY: the packed proof covers every complete K16 tile below
                // `width`; all lanes follow the same ascending sequence.
                unsafe {
                    if remaining >= 512 {
                        accumulator = chain_batch::<32>(mma, a, b, source, n, k, accumulator);
                        k += 512;
                    } else if remaining >= 256 {
                        accumulator = chain_batch::<16>(mma, a, b, source, n, k, accumulator);
                        k += 256;
                    } else if remaining >= 128 {
                        accumulator = chain_batch::<8>(mma, a, b, source, n, k, accumulator);
                        k += 128;
                    } else if remaining >= 64 {
                        accumulator = chain_batch::<4>(mma, a, b, source, n, k, accumulator);
                        k += 64;
                    } else {
                        accumulator = chain_batch::<1>(mma, a, b, source, n, k, accumulator);
                        k += 16;
                    }
                }
            }
            total = add_slice(mma, total, accumulator);
            start = end;
        }
        let mut output = StridedTensorMut::from_disjoint_slice(output);
        let mut index = 0;
        // The decode epilogue visits the valid rows in the same register order
        // the totals were gathered in; the zero accumulator only drives it.
        mma.partition_c()
            .store_decode_with(FragmentC::ZERO, source, |row, col, _| {
                let value = if HAS_BIAS {
                    total[index] + bias[n + col].to_f32()
                } else {
                    total[index]
                };
                index += 1;
                // SAFETY: C's bijective layout and the checked grid give each
                // lane disjoint initialized output coordinates.
                unsafe {
                    output.store_at(row * outputs + n + col, BFloat16::from_f32(value));
                }
            });
    }

    /// Select the aligned or scalar weight path for one FMA-chain launch.
    ///
    /// # Safety
    /// As for `fma_chains_impl`, without the alignment choice.
    #[inline(always)]
    unsafe fn fma_chains<const CHAINS: usize, const HAS_BIAS: bool>(
        input: &[BFloat16],
        weight: &[BFloat16],
        bias: &[BFloat16],
        output: &mut DisjointSlice<BFloat16>,
        width: usize,
        outputs: usize,
        weight_stride: usize,
    ) {
        unsafe {
            if weight_stride.is_multiple_of(VECTOR) {
                fma_chains_impl::<CHAINS, HAS_BIAS, true>(
                    input,
                    weight,
                    bias,
                    output,
                    width,
                    outputs,
                    weight_stride,
                );
            } else {
                fma_chains_impl::<CHAINS, HAS_BIAS, false>(
                    input,
                    weight,
                    bias,
                    output,
                    width,
                    outputs,
                    weight_stride,
                );
            }
        }
    }

    /// Sixteen FP32 chains per output; eight outputs per 32-lane CTA.
    ///
    /// Input is one dense row of `width` values; weights are `[outputs, width]`
    /// with element pitch `weight_stride`; output is one dense row. The caller
    /// validates [`fma_chains_supported`], readiness and output ownership.
    #[launch_contract(domain=1, block=(32,1,1), min_compute_capability=(8,0))]
    #[kernel]
    pub fn fma16_linear_bf16(
        input: &[BFloat16],
        weight: &[BFloat16],
        mut output: DisjointSlice<BFloat16>,
        width: usize,
        outputs: usize,
        weight_stride: usize,
    ) {
        if fma_chains_supported(
            16,
            input,
            weight,
            output.len(),
            None,
            width,
            outputs,
            weight_stride,
        ) {
            // SAFETY: the launch contract supplies one complete warp and the
            // predicate above establishes the operand contract.
            unsafe {
                fma_chains::<16, false>(
                    input,
                    weight,
                    input,
                    &mut output,
                    width,
                    outputs,
                    weight_stride,
                );
            }
        }
    }

    /// Sixteen FP32 chains per output with the bias added before rounding.
    ///
    /// Same contract as `fma16_linear_bf16`; `bias` holds `outputs` values.
    #[launch_contract(domain=1, block=(32,1,1), min_compute_capability=(8,0))]
    #[kernel]
    pub fn fma16_linear_bias_bf16(
        input: &[BFloat16],
        weight: &[BFloat16],
        bias: &[BFloat16],
        mut output: DisjointSlice<BFloat16>,
        width: usize,
        outputs: usize,
        weight_stride: usize,
    ) {
        if fma_chains_supported(
            16,
            input,
            weight,
            output.len(),
            Some(bias.len()),
            width,
            outputs,
            weight_stride,
        ) {
            // SAFETY: as for `fma16_linear_bf16`, with the bias extent checked.
            unsafe {
                fma_chains::<16, true>(
                    input,
                    weight,
                    bias,
                    &mut output,
                    width,
                    outputs,
                    weight_stride,
                );
            }
        }
    }

    /// Thirty-two FP32 chains per output; four outputs per 32-lane CTA.
    ///
    /// Same contract as `fma16_linear_bf16` with `width` a multiple of 32.
    #[launch_contract(domain=1, block=(32,1,1), min_compute_capability=(8,0))]
    #[kernel]
    pub fn fma32_linear_bf16(
        input: &[BFloat16],
        weight: &[BFloat16],
        mut output: DisjointSlice<BFloat16>,
        width: usize,
        outputs: usize,
        weight_stride: usize,
    ) {
        if fma_chains_supported(
            32,
            input,
            weight,
            output.len(),
            None,
            width,
            outputs,
            weight_stride,
        ) {
            // SAFETY: as for `fma16_linear_bf16`.
            unsafe {
                fma_chains::<32, false>(
                    input,
                    weight,
                    input,
                    &mut output,
                    width,
                    outputs,
                    weight_stride,
                );
            }
        }
    }

    /// Thirty-two FP32 chains per output with the bias added before rounding.
    ///
    /// Same contract as `fma32_linear_bf16`; `bias` holds `outputs` values.
    #[launch_contract(domain=1, block=(32,1,1), min_compute_capability=(8,0))]
    #[kernel]
    pub fn fma32_linear_bias_bf16(
        input: &[BFloat16],
        weight: &[BFloat16],
        bias: &[BFloat16],
        mut output: DisjointSlice<BFloat16>,
        width: usize,
        outputs: usize,
        weight_stride: usize,
    ) {
        if fma_chains_supported(
            32,
            input,
            weight,
            output.len(),
            Some(bias.len()),
            width,
            outputs,
            weight_stride,
        ) {
            // SAFETY: as for `fma32_linear_bf16`, with the bias extent checked.
            unsafe {
                fma_chains::<32, true>(
                    input,
                    weight,
                    bias,
                    &mut output,
                    width,
                    outputs,
                    weight_stride,
                );
            }
        }
    }

    /// One/two-row tensor-core projection with split-K slices of `slice_width`.
    ///
    /// Input is `[rows, width]` and weights `[outputs, width]` with element
    /// pitches; output is dense `[rows, outputs]`. The caller validates
    /// [`split_k_supported`], readiness and output ownership, and launches one
    /// 32-thread CTA per eight output columns.
    #[launch_contract(domain=1, block=(32,1,1), min_compute_capability=(8,0))]
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn split_k_linear_bf16(
        input: &[BFloat16],
        weight: &[BFloat16],
        mut output: DisjointSlice<BFloat16>,
        rows: usize,
        width: usize,
        outputs: usize,
        input_stride: usize,
        weight_stride: usize,
        slice_width: usize,
    ) {
        if split_k_supported(
            input,
            weight,
            output.len(),
            rows,
            width,
            outputs,
            input_stride,
            weight_stride,
            slice_width,
        ) {
            PackedDecode::with_validated(
                input,
                weight,
                rows,
                width,
                outputs,
                input_stride,
                weight_stride,
                |source| {
                    // SAFETY: the packed proof validates operand extents and
                    // alignment, the predicate covered the output and slice, and
                    // the launch contract supplies one complete warp.
                    unsafe {
                        packed_split_k_impl::<false>(
                            source,
                            input,
                            &mut output,
                            width,
                            outputs,
                            slice_width,
                        );
                    }
                },
            );
        }
    }

    /// Split-K tensor-core projection with the bias added before rounding.
    ///
    /// Same contract as `split_k_linear_bf16`; `bias` holds `outputs` values.
    #[launch_contract(domain=1, block=(32,1,1), min_compute_capability=(8,0))]
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn split_k_linear_bias_bf16(
        input: &[BFloat16],
        weight: &[BFloat16],
        bias: &[BFloat16],
        mut output: DisjointSlice<BFloat16>,
        rows: usize,
        width: usize,
        outputs: usize,
        input_stride: usize,
        weight_stride: usize,
        slice_width: usize,
    ) {
        if bias.len() >= outputs
            && split_k_supported(
                input,
                weight,
                output.len(),
                rows,
                width,
                outputs,
                input_stride,
                weight_stride,
                slice_width,
            )
        {
            PackedDecode::with_validated(
                input,
                weight,
                rows,
                width,
                outputs,
                input_stride,
                weight_stride,
                |source| {
                    // SAFETY: as for `split_k_linear_bf16`, with the bias extent checked.
                    unsafe {
                        packed_split_k_impl::<true>(
                            source,
                            bias,
                            &mut output,
                            width,
                            outputs,
                            slice_width,
                        );
                    }
                },
            );
        }
    }
}
