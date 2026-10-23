/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Waveform kernels use time-major [time, channel] storage throughout.
use ctm_rs::memory::DisjointSlice;
use ctm_rs::{cuda_module, kernel, launch_contract, thread};
use cute_rs::BFloat16;

/// Device convolution, normalization and activation operations for time-major batches.
#[cuda_module]
pub mod kernels {
    use super::*;

    use cute_rs::mma::bf16::{Bf16Mma16x8x16, FragmentC};
    use cute_rs::tensor::partition::{StridedTensor, StridedTensorMut};

    /// Read one virtual im2col element, zero-filling outside the input extent.
    /// Transposed convolution separates output phases and visits the input rows
    /// selected by the current phase.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn input_value(
        input: &[BFloat16],
        row: usize,
        k: usize,
        rows: usize,
        inputs: usize,
        in_group: usize,
        k_steps: usize,
        group: usize,
        kernel: usize,
        step: usize,
        phase: usize,
        transpose: bool,
    ) -> u16 {
        if row >= rows || k >= in_group * k_steps {
            return 0;
        }
        let channel = k / k_steps;
        let tap = k % k_steps;
        if transpose && phase + tap * step >= kernel {
            return 0;
        }
        let delay = if transpose {
            tap
        } else {
            (kernel - 1 - tap) * step
        };
        if row < delay {
            0
        } else {
            input[(row - delay) * inputs + group * in_group + channel].to_bits()
        }
    }
    /// Read one BF16 weight for a logical im2col reduction coordinate.
    ///
    /// Ordinary convolution stores `[output, input/group, tap]`; transpose stores
    /// `[input, output/group, tap]` and selects `phase + tap * step`. Padded tile
    /// coordinates return zero. The caller supplies valid nonzero group geometry.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn weight_value(
        weight: &[BFloat16],
        column: usize,
        k: usize,
        in_group: usize,
        out_group: usize,
        k_steps: usize,
        group: usize,
        kernel: usize,
        step: usize,
        phase: usize,
        transpose: bool,
    ) -> u16 {
        if column >= out_group || k >= in_group * k_steps {
            return 0;
        }
        let channel = k / k_steps;
        let tap = k % k_steps;
        let index = if transpose {
            let physical_tap = phase + tap * step;
            if physical_tap >= kernel {
                return 0;
            }
            ((group * in_group + channel) * out_group + column) * kernel + physical_tap
        } else {
            ((group * out_group + column) * in_group + channel) * kernel + tap
        };
        weight[index].to_bits()
    }
    /// Direct implicit-im2col MMA; no packed activation or weight temporary.
    /// Inputs are `[rows, inputs]`; output has `rows` time steps for convolution
    /// or `rows * step` for transpose. `step` is dilation or transpose stride.
    /// Groups divide both channel counts and weight orientation follows that mode.
    /// The host validates all extents and separate output ownership. Full warps
    /// collectively execute 16x8x16 BF16 MMA with FP32 accumulation; depthwise
    /// convolution instead uses ordered scalar tap sums. Bias is added after a
    /// BF16 rounding of the dot product, then the final value is rounded again.
    #[launch_contract(domain=1, block=(128,1,1), min_compute_capability=(8,0))]
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn conv_bf16(
        input: &[BFloat16],
        weight: &[BFloat16],
        bias: &[BFloat16],
        mut output: DisjointSlice<BFloat16>,
        rows: usize,
        inputs: usize,
        outputs: usize,
        kernel: usize,
        step: usize,
        groups: usize,
        has_bias: bool,
        transpose: bool,
    ) {
        if !transpose && groups == inputs && groups == outputs {
            // Depthwise channels have only kernel taps to reduce. Padding a
            // single channel into a tensor-core tile would add needless work.
            let index = thread::index_1d().get();
            if index < rows * outputs {
                let row = index / outputs;
                let channel = index % outputs;
                // One row view contains the ordered tap reduction. Tensor
                // indexing describes convolution coordinates; no temporary
                // im2col buffer or reassociated sum is introduced.
                let input = StridedTensor::from_slice(input);
                // SAFETY: the depthwise channel selects a complete tap row.
                let weight = unsafe {
                    StridedTensor::from_slice(weight).subview_unchecked(channel * kernel, kernel)
                };
                let mut tap = 0;
                let mut sum = weight.fold(0.0_f32, |sum, weight| {
                    let delay = (kernel - 1 - tap) * step;
                    tap += 1;
                    if row >= delay {
                        // SAFETY: the delay predicate bounds the input row;
                        // channel is inside the depthwise input width.
                        sum + unsafe { input.get_unchecked((row - delay) * inputs + channel) }
                            .to_f32()
                            * weight.to_f32()
                    } else {
                        sum
                    }
                });
                if has_bias {
                    // cuDNN stores its BF16 convolution before the separate bias add.
                    sum = BFloat16::from_f32(sum).to_f32() + bias[channel].to_f32();
                }
                let mut output = StridedTensorMut::from_disjoint_slice(&mut output)
                    .local_partition::<32>(index % 32);
                // SAFETY: global threads select distinct partition coordinates.
                unsafe {
                    output.store_at(index / 32, BFloat16::from_f32(sum));
                }
            }
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let lane = tid % 32;
        let tile = thread::blockIdx_x() as usize * 4 + tid / 32;
        let in_group = inputs / groups;
        let out_group = outputs / groups;
        let columns = out_group.div_ceil(8);
        let row_tiles = rows.div_ceil(16);
        let phases = if transpose { step } else { 1 };
        if tile < row_tiles * columns * phases * groups {
            let group = tile / (row_tiles * columns * phases);
            let phase = (tile / (row_tiles * columns)) % phases;
            let m = (tile / columns) % row_tiles * 16;
            let n = tile % columns * 8;
            let mma = Bf16Mma16x8x16::get_slice(lane);
            let k_steps = if transpose {
                kernel.div_ceil(step)
            } else {
                kernel
            };
            let mut accumulator = FragmentC::ZERO;
            let mut k = 0;
            while k < in_group * k_steps {
                // The convolution supplies logical im2col coordinates. The
                // CuTe atom partitions and packs those values for each lane.
                let a = mma.partition_a().load_with(|r, reduction| {
                    BFloat16::from_bits(input_value(
                        input,
                        m + r,
                        k + reduction,
                        rows,
                        inputs,
                        in_group,
                        k_steps,
                        group,
                        kernel,
                        step,
                        phase,
                        transpose,
                    ))
                });
                let b = mma.partition_b().load_with(|c, reduction| {
                    BFloat16::from_bits(weight_value(
                        weight,
                        n + c,
                        k + reduction,
                        in_group,
                        out_group,
                        k_steps,
                        group,
                        kernel,
                        step,
                        phase,
                        transpose,
                    ))
                });
                // SAFETY: the enclosing tile predicate is warp-uniform; all
                // 32 lanes execute identical MMA counts with typed A/B tiles.
                accumulator = unsafe { mma.mma_atom_call(accumulator, a, b) };
                k += 16;
            }
            let mut output = StridedTensorMut::from_disjoint_slice(&mut output);
            mma.partition_c().store_with(accumulator, |r, c, value| {
                if m + r < rows && n + c < out_group {
                    let channel = group * out_group + n + c;
                    let value = if has_bias {
                        // Preserve the observable convolution output before bias.
                        BFloat16::from_f32(value).to_f32() + bias[channel].to_f32()
                    } else {
                        value
                    };
                    let row = if transpose {
                        (m + r) * step + phase
                    } else {
                        m + r
                    };
                    // SAFETY: the CuTe C partition is bijective within each
                    // tile; groups and phases map to distinct output positions.
                    unsafe {
                        output.store_at(row * outputs + channel, BFloat16::from_f32(value));
                    }
                }
            });
        }
    }

    /// Affine LayerNorm reduces in FP32 and rounds only the final affine value.
    /// The host supplies dense `[rows, width]` input/output, width-sized affine
    /// vectors, a positive epsilon and distinct output storage. One complete warp
    /// owns each row and participates convergently in both mean/variance sums.
    #[launch_contract(domain=1, block=(128,1,1), min_compute_capability=(8,0))]
    #[kernel]
    pub fn layer_norm_bf16(
        input: &[BFloat16],
        weight: &[BFloat16],
        bias: &[BFloat16],
        mut output: DisjointSlice<BFloat16>,
        rows: usize,
        width: usize,
        epsilon: f32,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let row = thread::blockIdx_x() as usize * 4 + tid / 32;
        let lane = tid % 32;
        if row < rows {
            // SAFETY: the launch descriptor bounds the complete row and
            // parameter vectors. Lanes partition columns in increasing order.
            let input =
                unsafe { StridedTensor::from_slice(input).subview_unchecked(row * width, width) }
                    .local_partition::<32>(lane);
            let weight = StridedTensor::from_slice(weight).local_partition::<32>(lane);
            let bias = StridedTensor::from_slice(bias).local_partition::<32>(lane);
            let sum = input.fold(0.0_f32, |sum, value| sum + value.to_f32());
            // SAFETY: complete warp participates; row predicate is warp-uniform.
            let mean = unsafe { cute_rs::arch::warp_reduction_sum_f32(sum) } / width as f32;
            let squared = input.fold(0.0_f32, |sum, value| {
                let d = value.to_f32() - mean;
                sum + d * d
            });
            // SAFETY: identical convergent warp participation as the first sum.
            let variance = unsafe { cute_rs::arch::warp_reduction_sum_f32(squared) } / width as f32;
            let inverse = cute_rs::arch::rsqrt_f32(variance + epsilon);
            let mut output = unsafe {
                StridedTensorMut::from_disjoint_slice(&mut output)
                    .subview_unchecked(row * width, width)
            }
            .local_partition::<32>(lane);
            let mut item = 0;
            // SAFETY: the matching row partitions have distinct destinations;
            // affine vectors contain the same logical columns as the input.
            unsafe {
                input.map_to(&mut output, |value| {
                    let value =
                        (value.to_f32() - mean) * inverse * weight.get_unchecked(item).to_f32()
                            + bias.get_unchecked(item).to_f32();
                    item += 1;
                    BFloat16::from_f32(value)
                });
            }
        }
    }
    /// Round an FP32 intermediate to BF16 and widen it exactly back to FP32.
    #[inline(always)]
    fn rounded(value: f32) -> f32 {
        BFloat16::from_f32(value).to_f32()
    }

    /// Erf-form GELU. The five-coefficient erf approximation has about 1.5e-7
    /// absolute approximation error before FP32 arithmetic; GPU fixtures compare
    /// the observable BF16 result with an independent double-precision libc erf.
    #[inline(always)]
    fn gelu(value: f32) -> f32 {
        let x = value * core::f32::consts::FRAC_1_SQRT_2;
        let magnitude = if x < 0.0 { -x } else { x };
        let t = cute_rs::arch::reciprocal_ftz_f32(1.0 + 0.3275911 * magnitude);
        let polynomial =
            ((((1.0614054 * t - 1.4531521) * t + 1.4214138) * t - 0.28449672) * t + 0.2548296) * t;
        let tail = polynomial * cute_rs::arch::exp2_f32(-x * x * core::f32::consts::LOG2_E);
        // Use the complementary tail for negative values to avoid cancellation
        // in 1+erf(x), particularly for small negative GELU outputs.
        let factor = if x < 0.0 { tail } else { 2.0 - tail };
        0.5 * value * factor
    }
    /// Preserve the eager BF16 operation boundaries used by SnakeBeta.
    #[inline(always)]
    fn snake(value: f32, log_alpha: f32, log_beta: f32) -> f32 {
        let alpha = rounded(cute_rs::arch::exp2_f32(
            log_alpha * core::f32::consts::LOG2_E,
        ));
        let beta = rounded(cute_rs::arch::exp2_f32(
            log_beta * core::f32::consts::LOG2_E,
        ));
        let reciprocal = rounded(cute_rs::arch::reciprocal_ftz_f32(rounded(beta + 1e-9)));
        let angle = rounded(value * alpha);
        // BF16 codec activations use modest angles. Split π reduces the angle
        // without the cancellation from a single rounded π constant. Squaring
        // sine makes the omitted odd-multiple sign immaterial.
        let turns = (angle * core::f32::consts::FRAC_1_PI) as i32;
        let reduced = (angle - turns as f32 * 3.1415925) - turns as f32 * 1.509958e-7;
        let sine = rounded(cute_rs::arch::sin_f32(reduced));
        value + rounded(reciprocal * rounded(sine * sine))
    }
    /// Apply a private operation tag to `length` dense BF16 values.
    ///
    /// Tags select SnakeBeta (0), erf GELU (1), clamp (2), row normalization (3),
    /// or elementwise multiplication (otherwise). SnakeBeta reads width-sized
    /// log-parameter vectors; normalization reads one count per row and clamps it
    /// to BF16-rounded `lower`; multiplication reads `length` alpha values. The
    /// host checks these extents and separate output storage. Each thread writes
    /// one value; SnakeBeta retains eager BF16 intermediate rounding.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn unary_bf16(
        input: &[BFloat16],
        alpha: &[BFloat16],
        beta: &[BFloat16],
        mut output: DisjointSlice<BFloat16>,
        length: usize,
        width: usize,
        operation: u32,
        lower: f32,
        upper: f32,
    ) {
        let index = thread::index_1d().get();
        if index < length {
            // The distributed tensor preserves the original one-element-per-
            // thread launch while making the thread/value partition explicit.
            let lane = index % 32;
            let item = index / 32;
            let input = StridedTensor::from_slice(input).local_partition::<32>(lane);
            let alpha = StridedTensor::from_slice(alpha);
            let beta = StridedTensor::from_slice(beta);
            let mut output =
                StridedTensorMut::from_disjoint_slice(&mut output).local_partition::<32>(lane);
            // SAFETY: index is in the checked logical range; private host
            // dispatch validates the parameter extents for each operation.
            let value = unsafe { input.get_unchecked(item) }.to_f32();
            let result = if operation == 0 {
                unsafe {
                    snake(
                        value,
                        alpha.get_unchecked(index % width).to_f32(),
                        beta.get_unchecked(index % width).to_f32(),
                    )
                }
            } else if operation == 1 {
                gelu(value)
            } else if operation == 2 {
                if value < lower {
                    lower
                } else if value > upper {
                    upper
                } else {
                    value
                }
            } else if operation == 3 {
                let count = unsafe { alpha.get_unchecked(index / width) }.to_f32();
                let minimum = rounded(lower);
                value / if count < minimum { minimum } else { count }
            } else {
                value * unsafe { alpha.get_unchecked(index) }.to_f32()
            };
            // SAFETY: global threads select distinct partition coordinates.
            unsafe {
                output.store_at(item, BFloat16::from_f32(result));
            }
        }
    }
    /// Compute `residual + BF16(input * scale)` with a final BF16 rounding.
    ///
    /// Input, residual and output each contain `length` values in time-major rows
    /// of `width` channels; scale contains one value per channel. The host checks
    /// matching extents and distinct output storage; each thread writes one value.
    #[kernel]
    pub fn scaled_add_bf16(
        input: &[BFloat16],
        residual: &[BFloat16],
        scale: &[BFloat16],
        mut output: DisjointSlice<BFloat16>,
        length: usize,
        width: usize,
    ) {
        let index = thread::index_1d().get();
        if index < length {
            let lane = index % 32;
            let item = index / 32;
            let input = StridedTensor::from_slice(input).local_partition::<32>(lane);
            let residual = StridedTensor::from_slice(residual).local_partition::<32>(lane);
            let scale = StridedTensor::from_slice(scale);
            let mut output =
                StridedTensorMut::from_disjoint_slice(&mut output).local_partition::<32>(lane);
            // SAFETY: the host checks matching input/residual/output lengths
            // and a complete per-channel scale vector.
            let product = unsafe {
                rounded(
                    input.get_unchecked(item).to_f32()
                        * scale.get_unchecked(index % width).to_f32(),
                )
            };
            let result = product + unsafe { residual.get_unchecked(item) }.to_f32();
            // SAFETY: global threads select distinct partition coordinates.
            unsafe {
                output.store_at(item, BFloat16::from_f32(result));
            }
        }
    }
}
