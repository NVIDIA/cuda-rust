/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! True NVFP4 (`Float4E2M1FN` + `Float8E4M3FN` scale, `SF_VEC = 16`) quantization and GPU projection kernels.
//!
//! Each 16-element K block is represented by:
//! - Eight packed bytes (`16` 4-bit [`Float4E2M1FN`] nibbles in `[-6.0, 6.0]`), and
//! - One 8-bit [`Float8E4M3FN`] block scale stored in Blackwell's canonical
//!   `(32, 4)` 512-byte scale-atom layout ([`cute_rs::Sm1xxBlockScaleKMajor`]).
//!
//! # Example
//!
//! ```
//! use cute_rs::BFloat16;
//! use qwen3_8b_cute::nvfp4::{dequantize_nvfp4_row, quantize_matrix_nvfp4};
//!
//! let row: Vec<BFloat16> = (0..64).map(|i| BFloat16::from_f32((i as f32 - 32.0) * 0.125)).collect();
//! let (values, scales) = quantize_matrix_nvfp4(&row, 1, 64).unwrap();
//! let reconstructed = dequantize_nvfp4_row(&values, &scales, 0, 64);
//! assert_eq!(reconstructed.len(), 64);
//! ```

use crate::model::Batch;
use crate::operator::Result;
use ctm_rs::memory::DisjointSlice;
use ctm_rs::{cuda_module, kernel, thread};
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaStream, DeviceBuffer};
use cute_rs::block_scaled::{
    CANONICAL_SCALE_ATOM_BYTES, CANONICAL_SCALE_K_GROUPS, CANONICAL_SCALE_M,
};
use cute_rs::ffn::{ProjectionRole, ProjectionWeights};
use cute_rs::tensor::partition::StridedTensorMut;
use cute_rs::{
    BFloat16, BlockScaledTensor, Float4E2M1FN, Float8E4M3FN, KMajor, Mkl, Nkl, e4m3_to_f32,
};
use std::marker::PhantomData;
use std::sync::Arc;

/// Encode a non-negative `f32` scale into the nearest finite `Float8E4M3FN` byte (`[0.0, 448.0]`).
#[must_use]
#[inline(always)]
pub fn f32_to_e4m3(value: f32) -> u8 {
    if value.is_nan() || value <= 0.0 {
        return 0;
    }
    let clamped = if value > 448.0 { 448.0 } else { value };
    let mut best_code = 0u8;
    let mut best_err = f32::INFINITY;
    let mut code = 0u8;
    while code <= 0x7e {
        let candidate = e4m3_to_f32(code);
        let err = (candidate - clamped).abs();
        if err < best_err {
            best_err = err;
            best_code = code;
        }
        code += 1;
    }
    best_code
}

/// Encode one signed `f32` value into its nearest 4-bit `Float4E2M1FN` nibble (`bits 3..0`).
#[must_use]
#[inline(always)]
pub fn f32_to_e2m1(value: f32) -> u8 {
    let sign = if value < 0.0 { 0x08u8 } else { 0x00u8 };
    let abs = if value < 0.0 { -value } else { value };
    let mag = if abs < 0.25 {
        0
    } else if abs < 0.75 {
        1
    } else if abs < 1.25 {
        2
    } else if abs < 1.75 {
        3
    } else if abs < 2.5 {
        4
    } else if abs < 3.5 {
        5
    } else if abs < 5.0 {
        6
    } else {
        7
    };
    sign | mag
}

/// Compute the byte offset of `(row, group)` inside a canonical `(32, 4)` Blackwell scale buffer.
#[must_use]
#[inline(always)]
pub const fn canonical_scale_offset(rest_k: usize, row: usize, group: usize) -> usize {
    (row >> 7) * rest_k * CANONICAL_SCALE_ATOM_BYTES
        + (group >> 2) * CANONICAL_SCALE_ATOM_BYTES
        + ((row & 31) << 4)
        + (((row >> 5) & 3) << 2)
        + (group & 3)
}

/// Quantize a dense `[rows, k]` BF16 matrix into packed `Float4E2M1FN` bytes and canonical `Float8E4M3FN` scales.
pub fn quantize_matrix_nvfp4(
    matrix: &[BFloat16],
    rows: usize,
    k: usize,
) -> Result<(Vec<u8>, Vec<u8>)> {
    if rows == 0 || k == 0 || !k.is_multiple_of(64) {
        return Err("NVFP4 quantization requires positive rows and K divisible by 64".into());
    }
    let required = rows.checked_mul(k).ok_or("NVFP4 matrix extent overflow")?;
    if matrix.len() < required {
        return Err("NVFP4 source slice is shorter than rows * k".into());
    }
    let packed_k = k / 2;
    let groups_per_row = k / 16;
    let rest_m = rows.div_ceil(CANONICAL_SCALE_M);
    let rest_k = groups_per_row.div_ceil(CANONICAL_SCALE_K_GROUPS);
    let mut values = vec![0u8; rows * packed_k];
    let mut scales = vec![0u8; rest_m * rest_k * CANONICAL_SCALE_ATOM_BYTES];

    for r in 0..rows {
        let row_slice = &matrix[r * k..(r + 1) * k];
        let out_row = &mut values[r * packed_k..(r + 1) * packed_k];
        for g in 0..groups_per_row {
            let block = &row_slice[g * 16..(g + 1) * 16];
            let mut max_abs = 0.0f32;
            for &v in block {
                let a = v.to_f32().abs();
                if a > max_abs {
                    max_abs = a;
                }
            }
            let scale_code = f32_to_e4m3(max_abs / 6.0);
            let scale_val = e4m3_to_f32(scale_code);
            let inv_scale = if scale_val > 0.0 {
                1.0 / scale_val
            } else {
                0.0
            };
            scales[canonical_scale_offset(rest_k, r, g)] = scale_code;
            for pair in 0..8 {
                let lo = f32_to_e2m1(block[pair * 2].to_f32() * inv_scale);
                let hi = f32_to_e2m1(block[pair * 2 + 1].to_f32() * inv_scale);
                out_row[g * 8 + pair] = lo | (hi << 4);
            }
        }
    }
    Ok((values, scales))
}

/// Reconstruct one logical row of `k` `f32` values from NVFP4 `values` and canonical `Float8E4M3FN` `scales`.
#[must_use]
pub fn dequantize_nvfp4_row(values: &[u8], scales: &[u8], row: usize, k: usize) -> Vec<f32> {
    let packed_k = k / 2;
    let groups_per_row = k / 16;
    let rest_k = groups_per_row.div_ceil(CANONICAL_SCALE_K_GROUPS);
    let mut out = Vec::with_capacity(k);
    for g in 0..groups_per_row {
        let scale = e4m3_to_f32(scales[canonical_scale_offset(rest_k, row, g)]);
        for pair in 0..8 {
            let byte = values[row * packed_k + g * 8 + pair];
            let (lo, hi) = cute_rs::Float4E2M1FNx2::from_bits(byte).to_f32x2();
            out.push(lo * scale);
            out.push(hi * scale);
        }
    }
    out
}

/// Device kernels for dynamic BF16-to-NVFP4 activation quantization and NVFP4 linear projection.
#[allow(missing_docs, clippy::missing_docs_in_private_items)]
#[cuda_module]
pub mod kernels {
    use super::*;

    /// Decode one E4M3FN byte to `f32` inside device code.
    #[inline(always)]
    fn dev_e4m3_to_f32(byte: u8) -> f32 {
        e4m3_to_f32(byte)
    }

    /// Encode a non-negative `f32` scale into `E4M3FN` inside device code using IEEE bit fields.
    #[inline(always)]
    fn dev_f32_to_e4m3(value: f32) -> u8 {
        if value.is_nan() || value <= 0.0 {
            return 0;
        }
        let clamped = if value > 448.0 { 448.0 } else { value };
        if clamped < (1.0 / 64.0) {
            let mant = ((clamped * 512.0) + 0.5) as u32;
            let m = mant.clamp(1, 7);
            return m as u8;
        }
        let bits = clamped.to_bits();
        let f32_exp = ((bits >> 23) & 0xff) as i32;
        let e4m3_exp = f32_exp - 120;
        if e4m3_exp <= 0 {
            let mant = ((clamped * 512.0) + 0.5) as u32;
            return (if mant > 7 { 7 } else { mant }) as u8;
        }
        let rounded_mant = (((bits >> 19) & 0x0f) + 1) >> 1;
        let (mut exp, mut mant) = if rounded_mant >= 8 {
            (e4m3_exp + 1, 0u32)
        } else {
            (e4m3_exp, rounded_mant)
        };
        if exp > 15 || (exp == 15 && mant == 7) {
            exp = 15;
            mant = 6;
        }
        (((exp as u8) & 0x0f) << 3) | ((mant as u8) & 0x07)
    }

    /// Quantize `[rows, k]` BF16 values into packed `Float4E2M1FN` bytes and canonical `Float8E4M3FN` scales.
    #[kernel]
    pub fn quantize_activations_nvfp4(
        input: &[BFloat16],
        mut out_values: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        rows: usize,
        k: usize,
        scale_boost: f32,
        inv_scale_boost: f32,
    ) {
        let tid = thread::index_1d().get();
        let tiles_per_row = k / 64;
        let total_tiles = rows * tiles_per_row;
        if tid >= total_tiles {
            return;
        }
        let row = tid / tiles_per_row;
        let tile = tid % tiles_per_row;
        let rest_k = tiles_per_row;
        let packed_k = k / 2;
        let in_base = row * k + tile * 64;
        let val_base = row * packed_k + tile * 32;
        let scale_base = (row >> 7) * rest_k * CANONICAL_SCALE_ATOM_BYTES
            + tile * CANONICAL_SCALE_ATOM_BYTES
            + ((row & 31) << 4)
            + (((row >> 5) & 3) << 2);

        let mut vals_out = StridedTensorMut::from_disjoint_slice(&mut out_values);
        let mut scales_out = StridedTensorMut::from_disjoint_slice(&mut out_scales);
        let mut g = 0usize;
        while g < 4 {
            let block_offset = in_base + g * 16;
            let mut max_abs = 0.0f32;
            let mut i = 0usize;
            while i < 16 {
                let raw = input[block_offset + i].to_f32();
                let a = if raw < 0.0 { -raw } else { raw };
                if a > max_abs {
                    max_abs = a;
                }
                i += 1;
            }
            let scale_code = dev_f32_to_e4m3(max_abs * (scale_boost / 6.0));
            let scale_val = dev_e4m3_to_f32(scale_code) * inv_scale_boost;
            let inv_scale = if scale_val > 0.0 {
                1.0 / scale_val
            } else {
                0.0
            };
            unsafe {
                scales_out.store_at(scale_base + g, scale_code);
            }
            let mut pair = 0usize;
            while pair < 8 {
                let x0 = input[block_offset + pair * 2].to_f32() * inv_scale;
                let x1 = input[block_offset + pair * 2 + 1].to_f32() * inv_scale;
                let packed = f32_to_e2m1(x0) | (f32_to_e2m1(x1) << 4);
                unsafe {
                    vals_out.store_at(val_base + g * 8 + pair, packed);
                }
                pair += 1;
            }
            g += 1;
        }
    }

    /// Project NVFP4-quantized `[rows, k]` activations against `[outputs, k]` NVFP4 weights.
    ///
    /// One 32-lane warp cooperates across the `K=64` tiles of one `(row, out_col)` output
    /// Hardware-lowered NVFP4 (`Float4E2M1FN` + `Float8E4M3FN`, `SF_VEC = 16`) linear projection.
    ///
    /// Maps `128` threads per CTA to `128` contiguous output columns (`out_col`), matching
    /// the `128`-row canonical `(32, 4)` Blackwell scale atom (`Sm1xxBlockScaleKMajor`),
    /// while `blockIdx_y` selects the token row (`0..rows`).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn nvfp4_project_bf16(
        weight_values: &[u8],
        weight_scales: &[u8],
        input_values: &[u8],
        input_scales: &[u8],
        mut output: DisjointSlice<BFloat16>,
        rows: usize,
        k: usize,
        outputs: usize,
    ) {
        let out_col = (thread::blockIdx_x() as usize) * 128 + (thread::threadIdx_x() as usize);
        let row = thread::blockIdx_y() as usize;

        let weights = BlockScaledTensor::<Float4E2M1FN, Float8E4M3FN, 16, KMajor<Mkl>>::from_slices(
            weight_values,
            weight_scales,
            outputs,
            k,
        );
        let inputs = BlockScaledTensor::<Float4E2M1FN, Float8E4M3FN, 16, KMajor<Nkl>>::from_slices(
            input_values,
            input_scales,
            rows,
            k,
        );
        let w_row = weights.thread_row(0, out_col);
        let x_row = inputs.thread_row(0, row);

        let mut acc = 0.0f32;
        let mut k_base = 0usize;
        while k_base < k {
            let tile = k_base / 64;
            let (w_frag, x_frag) =
                unsafe { (w_row.k_tile(tile).load(), x_row.k_tile(tile).load()) };
            acc = w_frag.dot_accumulate(x_frag, acc);
            k_base += 64;
        }
        let mut out_tensor = StridedTensorMut::from_disjoint_slice(&mut output);
        unsafe {
            out_tensor.store_at(
                row * outputs + out_col,
                BFloat16::from_f32(acc * (1.0 / 1024.0)),
            );
        }
    }

    /// Warp-cooperative NVFP4 (`Float4E2M1FN` + `Float8E4M3FN`, `SF_VEC = 16`) decode projection for `rows <= 2`.
    ///
    /// One 32-lane warp cooperates across `K=64` tiles for each `(row, out_col)` output element
    /// using hardware `cute.scaled_view_load` (`nvgpu.cvt_fpext`) and [`cute_rs::arch::warp_reduction_sum_f32`],
    /// launching `(rows * outputs) / 4` 128-thread CTAs to saturate all 170 SMs on RTX 5090.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn nvfp4_decode_bf16(
        weight_values: &[u8],
        weight_scales: &[u8],
        input_values: &[u8],
        input_scales: &[u8],
        mut output: DisjointSlice<BFloat16>,
        rows: usize,
        k: usize,
        outputs: usize,
    ) {
        let tid = thread::index_1d().get();
        let warp_id = tid / 32;
        let lane = tid % 32;
        let total = rows * outputs;
        if warp_id >= total {
            return;
        }
        let row = warp_id / outputs;
        let out_col = warp_id % outputs;

        let weights = BlockScaledTensor::<Float4E2M1FN, Float8E4M3FN, 16, KMajor<Mkl>>::from_slices(
            weight_values,
            weight_scales,
            outputs,
            k,
        );
        let inputs = BlockScaledTensor::<Float4E2M1FN, Float8E4M3FN, 16, KMajor<Nkl>>::from_slices(
            input_values,
            input_scales,
            rows,
            k,
        );
        let w_row = weights.thread_row(0, out_col);
        let x_row = inputs.thread_row(0, row);
        let tiles = k / 64;

        let mut acc = 0.0f32;
        let mut tile = lane;
        while tile < tiles {
            let (w_frag, x_frag) =
                unsafe { (w_row.k_tile(tile).load(), x_row.k_tile(tile).load()) };
            acc = w_frag.dot_accumulate(x_frag, acc);
            tile += 32;
        }
        let sum = unsafe { cute_rs::arch::warp_reduction_sum_f32(acc) } * (1.0 / 1024.0);
        if lane == 0 {
            let mut out_tensor = StridedTensorMut::from_disjoint_slice(&mut output);
            unsafe {
                out_tensor.store_at(row * outputs + out_col, BFloat16::from_f32(sum));
            }
        }
    }
}

/// Owned device buffers for an NVFP4-quantized `[OUT, IN]` linear projection matrix.
pub struct Nvfp4Projection<Role: ProjectionRole, const IN: usize, const OUT: usize> {
    /// Packed `Float4E2M1FN` weight values (`OUT * (IN / 2)` bytes).
    values: DeviceBuffer<u8>,
    /// Canonical `(32, 4)` `Float8E4M3FN` block scales (`1` byte per 16 K elements).
    scales: DeviceBuffer<u8>,
    /// Retains the semantic projection role (`Query`, `Key`, `Value`, `Gate`, `Up`, `Down`, etc.).
    role: PhantomData<Role>,
}

impl<Role: ProjectionRole, const IN: usize, const OUT: usize> Nvfp4Projection<Role, IN, OUT> {
    /// Quantize a checked BF16 [`ProjectionWeights`] matrix into device-resident NVFP4 storage.
    pub fn from_bf16(
        stream: &CudaStream,
        weights: &ProjectionWeights<'_, Role, IN, OUT>,
    ) -> Result<Self> {
        if weights.shape().row_stride() != IN {
            return Err("NVFP4 quantization requires dense [OUT, IN] BF16 rows".into());
        }
        stream.context().synchronize()?;
        let host_bf16 = weights.buffer().to_host_vec(stream)?;
        let (values, scales) = quantize_matrix_nvfp4(&host_bf16[..OUT * IN], OUT, IN)?;
        Ok(Self {
            values: DeviceBuffer::from_host(stream, &values)?,
            scales: DeviceBuffer::from_host(stream, &scales)?,
            role: PhantomData,
        })
    }

    /// Construct an NVFP4 projection directly from pre-quantized host byte buffers.
    pub fn from_host_parts(stream: &CudaStream, values: &[u8], scales: &[u8]) -> Result<Self> {
        let rest_m = OUT.div_ceil(CANONICAL_SCALE_M);
        let rest_k = (IN / 16).div_ceil(CANONICAL_SCALE_K_GROUPS);
        if !IN.is_multiple_of(64)
            || values.len() < OUT * (IN / 2)
            || scales.len() < rest_m * rest_k * CANONICAL_SCALE_ATOM_BYTES
        {
            return Err("invalid pre-quantized NVFP4 buffer lengths".into());
        }
        Ok(Self {
            values: DeviceBuffer::from_host(stream, values)?,
            scales: DeviceBuffer::from_host(stream, scales)?,
            role: PhantomData,
        })
    }

    /// Total device bytes occupied by the packed `Float4E2M1FN` values and `Float8E4M3FN` scales.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.values.len() + self.scales.len()
    }
}

/// Maximum pre-allocated activation rows in [`Nvfp4Runner`] before falling back to transient buffers.
const MAX_SCRATCH_ROWS: usize = 256;
/// Maximum input K dimension (`INTERMEDIATE = 12288`) for pre-allocated activation quantization scratch.
const MAX_SCRATCH_K: usize = 12_288;

/// GPU runner for NVFP4 (`Float4E2M1FN` + `Float8E4M3FN`, `SF_VEC = 16`) linear projections with pooled buffers.
pub struct Nvfp4Runner {
    /// Stream used for quantization and projection launches.
    stream: Arc<CudaStream>,
    /// Loaded NVFP4 device kernels.
    module: kernels::LoadedModule,
    /// Reusable BF16 output buffer pool.
    pool: crate::model::BufferPool,
    /// Pre-allocated device buffer for quantized activation `Float4E2M1FN` bytes.
    act_values: std::sync::Mutex<DeviceBuffer<u8>>,
    /// Pre-allocated device buffer for quantized activation `Float8E4M3FN` scales.
    act_scales: std::sync::Mutex<DeviceBuffer<u8>>,
}

impl Nvfp4Runner {
    /// Load the NVFP4 quantization and projection kernels onto `stream`'s context.
    pub fn new(stream: Arc<CudaStream>) -> Result<Self> {
        let max_val_bytes = MAX_SCRATCH_ROWS * (MAX_SCRATCH_K / 2);
        let max_scale_bytes = MAX_SCRATCH_ROWS.div_ceil(CANONICAL_SCALE_M)
            * (MAX_SCRATCH_K / 64)
            * CANONICAL_SCALE_ATOM_BYTES;
        Ok(Self {
            module: kernels::load(stream.context())?,
            pool: crate::model::BufferPool::new(64 * 1024 * 1024, 4 * 1024 * 1024, 16, |_| true),
            act_values: std::sync::Mutex::new(DeviceBuffer::<u8>::zeroed(&stream, max_val_bytes)?),
            act_scales: std::sync::Mutex::new(DeviceBuffer::<u8>::zeroed(
                &stream,
                max_scale_bytes,
            )?),
            stream,
        })
    }

    /// Borrow the runner's execution stream.
    #[must_use]
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Acquire a pooled `Batch<W>` on the runner's stream without `cuMemAlloc` / `cuMemFree`.
    pub fn allocate_batch<const W: usize>(&self, rows: usize) -> Result<Batch<W>> {
        Batch::<W>::allocate_from(&self.pool, &self.stream, rows)
    }

    /// Quantize a BF16 [`ProjectionWeights`] matrix directly on the GPU into [`Nvfp4Projection`].
    pub fn quantize_weights<Role: ProjectionRole, const IN: usize, const OUT: usize>(
        &self,
        weights: &ProjectionWeights<'_, Role, IN, OUT>,
    ) -> Result<Nvfp4Projection<Role, IN, OUT>> {
        if weights.shape().row_stride() != IN || !IN.is_multiple_of(64) {
            return Err(
                "GPU NVFP4 quantization requires dense [OUT, IN] rows with IN % 64 == 0".into(),
            );
        }
        let tiles_per_row = IN / 64;
        let rest_m = OUT.div_ceil(CANONICAL_SCALE_M);
        let rest_k = tiles_per_row;
        let mut values = DeviceBuffer::<u8>::zeroed(&self.stream, OUT * (IN / 2))?;
        let mut scales =
            DeviceBuffer::<u8>::zeroed(&self.stream, rest_m * rest_k * CANONICAL_SCALE_ATOM_BYTES)?;
        let quant_threads = OUT * tiles_per_row;
        let quant_cfg = LaunchConfig {
            grid_dim: (quant_threads.div_ceil(128) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.module.quantize_activations_nvfp4(
                &self.stream,
                quant_cfg,
                weights.buffer(),
                &mut values,
                &mut scales,
                OUT,
                IN,
                64.0f32,
                1.0f32 / 64.0f32,
            )?;
        }
        Ok(Nvfp4Projection {
            values,
            scales,
            role: PhantomData,
        })
    }

    /// Quantize `input` activations on-device and project them through `weight` in NVFP4.
    pub fn project<Role: ProjectionRole, const IN: usize, const OUT: usize>(
        &self,
        input: &Batch<IN>,
        weight: &Nvfp4Projection<Role, IN, OUT>,
    ) -> Result<Batch<OUT>> {
        self.project_buffer(input.buffer(), input.rows(), weight)
    }

    /// Quantize `rows` of `IN`-wide BF16 activations into the runner's internal scratch.
    fn quantize_into_scratch(
        &self,
        input_buffer: &DeviceBuffer<BFloat16>,
        rows: usize,
        in_width: usize,
        act_values: &mut DeviceBuffer<u8>,
        act_scales: &mut DeviceBuffer<u8>,
    ) -> Result<()> {
        let tiles_per_row = in_width / 64;
        let quant_threads = rows * tiles_per_row;
        let quant_cfg = LaunchConfig {
            grid_dim: (quant_threads.div_ceil(128) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.module.quantize_activations_nvfp4(
                &self.stream,
                quant_cfg,
                input_buffer,
                act_values,
                act_scales,
                rows,
                in_width,
                16.0f32,
                1.0f32 / 16.0f32,
            )?;
        }
        Ok(())
    }

    /// Launch `nvfp4_decode_bf16` (for `rows <= 2`) or `nvfp4_project_bf16` (for `rows > 2`).
    fn launch_project<Role: ProjectionRole, const IN: usize, const OUT: usize>(
        &self,
        act_values: &DeviceBuffer<u8>,
        act_scales: &DeviceBuffer<u8>,
        rows: usize,
        weight: &Nvfp4Projection<Role, IN, OUT>,
        output: &mut DeviceBuffer<BFloat16>,
    ) -> Result<()> {
        if rows <= 64 {
            let total_outputs = rows * OUT;
            let dec_cfg = LaunchConfig {
                grid_dim: (total_outputs.div_ceil(4) as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.module.nvfp4_decode_bf16(
                    &self.stream,
                    dec_cfg,
                    &weight.values,
                    &weight.scales,
                    act_values,
                    act_scales,
                    output,
                    rows,
                    IN,
                    OUT,
                )?;
            }
            return Ok(());
        }
        let proj_cfg = LaunchConfig {
            grid_dim: (OUT.div_ceil(128) as u32, rows as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.module.nvfp4_project_bf16(
                &self.stream,
                proj_cfg,
                &weight.values,
                &weight.scales,
                act_values,
                act_scales,
                output,
                rows,
                IN,
                OUT,
            )?;
        }
        Ok(())
    }

    /// Quantize `input` once and project Q, K, and V in NVFP4 (`Float4E2M1FN` + `Float8E4M3FN`, `SF_VEC = 16`).
    pub fn project_qkv<
        QRole: ProjectionRole,
        KRole: ProjectionRole,
        VRole: ProjectionRole,
        const IN: usize,
        const Q_OUT: usize,
        const KV_OUT: usize,
    >(
        &self,
        input: &Batch<IN>,
        q_weight: &Nvfp4Projection<QRole, IN, Q_OUT>,
        k_weight: &Nvfp4Projection<KRole, IN, KV_OUT>,
        v_weight: &Nvfp4Projection<VRole, IN, KV_OUT>,
    ) -> Result<(Batch<Q_OUT>, Batch<KV_OUT>, Batch<KV_OUT>)> {
        let rows = input.rows();
        let mut act_values = self.act_values.lock().map_err(|_| "poisoned scratch")?;
        let mut act_scales = self.act_scales.lock().map_err(|_| "poisoned scratch")?;
        self.quantize_into_scratch(input.buffer(), rows, IN, &mut act_values, &mut act_scales)?;
        let mut q = Batch::<Q_OUT>::allocate_from(&self.pool, &self.stream, rows)?;
        let mut k = Batch::<KV_OUT>::allocate_from(&self.pool, &self.stream, rows)?;
        let mut v = Batch::<KV_OUT>::allocate_from(&self.pool, &self.stream, rows)?;
        self.launch_project(&act_values, &act_scales, rows, q_weight, q.buffer_mut())?;
        self.launch_project(&act_values, &act_scales, rows, k_weight, k.buffer_mut())?;
        self.launch_project(&act_values, &act_scales, rows, v_weight, v.buffer_mut())?;
        Ok((q, k, v))
    }

    /// Quantize `input` once and project SwiGLU `gate` and `up` in NVFP4 (`Float4E2M1FN` + `Float8E4M3FN`).
    pub fn project_gate_up<
        GRole: ProjectionRole,
        URole: ProjectionRole,
        const IN: usize,
        const OUT: usize,
    >(
        &self,
        input: &Batch<IN>,
        gate_weight: &Nvfp4Projection<GRole, IN, OUT>,
        up_weight: &Nvfp4Projection<URole, IN, OUT>,
    ) -> Result<(Batch<OUT>, Batch<OUT>)> {
        let rows = input.rows();
        let mut act_values = self.act_values.lock().map_err(|_| "poisoned scratch")?;
        let mut act_scales = self.act_scales.lock().map_err(|_| "poisoned scratch")?;
        self.quantize_into_scratch(input.buffer(), rows, IN, &mut act_values, &mut act_scales)?;
        let mut gate = Batch::<OUT>::allocate_from(&self.pool, &self.stream, rows)?;
        let mut up = Batch::<OUT>::allocate_from(&self.pool, &self.stream, rows)?;
        self.launch_project(
            &act_values,
            &act_scales,
            rows,
            gate_weight,
            gate.buffer_mut(),
        )?;
        self.launch_project(&act_values, &act_scales, rows, up_weight, up.buffer_mut())?;
        Ok((gate, up))
    }

    /// Quantize `rows` of `IN`-wide BF16 activations from `input_buffer` and project through `weight`.
    pub fn project_buffer<Role: ProjectionRole, const IN: usize, const OUT: usize>(
        &self,
        input_buffer: &DeviceBuffer<BFloat16>,
        rows: usize,
        weight: &Nvfp4Projection<Role, IN, OUT>,
    ) -> Result<Batch<OUT>> {
        if rows == 0 || !IN.is_multiple_of(64) || rows > MAX_SCRATCH_ROWS || IN > MAX_SCRATCH_K {
            return Err(
                "NVFP4 projection requires 1..=256 rows and IN <= 12288 divisible by 64".into(),
            );
        }
        let mut act_values = self.act_values.lock().map_err(|_| "poisoned scratch")?;
        let mut act_scales = self.act_scales.lock().map_err(|_| "poisoned scratch")?;
        let mut output = Batch::<OUT>::allocate_from(&self.pool, &self.stream, rows)?;
        self.quantize_into_scratch(input_buffer, rows, IN, &mut act_values, &mut act_scales)?;
        self.launch_project(&act_values, &act_scales, rows, weight, output.buffer_mut())?;
        Ok(output)
    }

    /// Asynchronously quantize `input_buffer` and project Q, K, and V on `self.stream` without allocation or sync.
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_qkv<
        QRole: ProjectionRole,
        KRole: ProjectionRole,
        VRole: ProjectionRole,
        const IN: usize,
        const Q_OUT: usize,
        const KV_OUT: usize,
    >(
        &self,
        input_buffer: &DeviceBuffer<BFloat16>,
        rows: usize,
        q_weight: &Nvfp4Projection<QRole, IN, Q_OUT>,
        k_weight: &Nvfp4Projection<KRole, IN, KV_OUT>,
        v_weight: &Nvfp4Projection<VRole, IN, KV_OUT>,
        q_out: &mut DeviceBuffer<BFloat16>,
        k_out: &mut DeviceBuffer<BFloat16>,
        v_out: &mut DeviceBuffer<BFloat16>,
    ) -> Result<()> {
        let mut act_values = self.act_values.lock().map_err(|_| "poisoned scratch")?;
        let mut act_scales = self.act_scales.lock().map_err(|_| "poisoned scratch")?;
        self.quantize_into_scratch(input_buffer, rows, IN, &mut act_values, &mut act_scales)?;
        self.launch_project(&act_values, &act_scales, rows, q_weight, q_out)?;
        self.launch_project(&act_values, &act_scales, rows, k_weight, k_out)?;
        self.launch_project(&act_values, &act_scales, rows, v_weight, v_out)?;
        Ok(())
    }

    /// Asynchronously quantize `input_buffer` and project Gate and Up on `self.stream` without allocation or sync.
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_gate_up<
        GRole: ProjectionRole,
        URole: ProjectionRole,
        const IN: usize,
        const OUT: usize,
    >(
        &self,
        input_buffer: &DeviceBuffer<BFloat16>,
        rows: usize,
        gate_weight: &Nvfp4Projection<GRole, IN, OUT>,
        up_weight: &Nvfp4Projection<URole, IN, OUT>,
        gate_out: &mut DeviceBuffer<BFloat16>,
        up_out: &mut DeviceBuffer<BFloat16>,
    ) -> Result<()> {
        let mut act_values = self.act_values.lock().map_err(|_| "poisoned scratch")?;
        let mut act_scales = self.act_scales.lock().map_err(|_| "poisoned scratch")?;
        self.quantize_into_scratch(input_buffer, rows, IN, &mut act_values, &mut act_scales)?;
        self.launch_project(&act_values, &act_scales, rows, gate_weight, gate_out)?;
        self.launch_project(&act_values, &act_scales, rows, up_weight, up_out)?;
        Ok(())
    }

    /// Asynchronously quantize `input_buffer` and project through `weight` on `self.stream` without allocation or sync.
    pub fn enqueue_project<Role: ProjectionRole, const IN: usize, const OUT: usize>(
        &self,
        input_buffer: &DeviceBuffer<BFloat16>,
        rows: usize,
        weight: &Nvfp4Projection<Role, IN, OUT>,
        output: &mut DeviceBuffer<BFloat16>,
    ) -> Result<()> {
        let mut act_values = self.act_values.lock().map_err(|_| "poisoned scratch")?;
        let mut act_scales = self.act_scales.lock().map_err(|_| "poisoned scratch")?;
        self.quantize_into_scratch(input_buffer, rows, IN, &mut act_values, &mut act_scales)?;
        self.launch_project(&act_values, &act_scales, rows, weight, output)?;
        Ok(())
    }
}

/// CPU unit tests for NVFP4 (`Float4E2M1FN` + `Float8E4M3FN` scale, `SF_VEC = 16`) encoding and block scaling.
#[cfg(test)]
mod tests {
    use super::*;

    /// Verify E4M3FN encoding round-trips exact fractional powers and clamps at 448.0.
    #[test]
    fn e4m3_encoder_preserves_fractional_scales_and_clamps_overflow() {
        for &target in &[
            0.0f32, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 6.0, 64.0, 448.0,
        ] {
            let code = f32_to_e4m3(target);
            assert_eq!(e4m3_to_f32(code), target);
        }
        assert_eq!(e4m3_to_f32(f32_to_e4m3(1000.0)), 448.0);
        assert_eq!(f32_to_e4m3(-5.0), 0);
    }

    /// Verify NVFP4 row quantization and dequantization preserve sign and scale across K=64 tiles.
    #[test]
    fn nvfp4_matrix_quantize_and_dequantize_roundtrip() {
        let mut row = Vec::with_capacity(64);
        for i in 0..64 {
            let val = ((i as f32) - 32.0) * 0.25;
            row.push(BFloat16::from_f32(val));
        }
        let (values, scales) = quantize_matrix_nvfp4(&row, 1, 64).unwrap();
        assert_eq!(values.len(), 32);
        assert_eq!(scales.len(), CANONICAL_SCALE_ATOM_BYTES);
        let recon = dequantize_nvfp4_row(&values, &scales, 0, 64);
        let recon_bf16: Vec<BFloat16> = recon.iter().copied().map(BFloat16::from_f32).collect();
        let cos = crate::benchmark::cosine_similarity_bf16(&row, &recon_bf16);
        assert!(cos > 0.99, "expected high cosine similarity, got {cos}");
    }
}
