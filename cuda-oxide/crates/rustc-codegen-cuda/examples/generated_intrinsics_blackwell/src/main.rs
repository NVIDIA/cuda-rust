/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Coverage for generated high-target intrinsics.
//!
//! The packed-FP8 and TF32 conversions are launched and their results checked
//! against the two formats, so the example reports what the device computed
//! rather than that it compiled. Both are bit-exact, which leaves no tolerance
//! to choose. The sparse MMA and TMA kernels stay compile-only: they need
//! operands and descriptors an example cannot supply.
//!
//! Compile-only in CI, since it pins `sm_120a` and no runner has that device.
//! Run it on one and the conversions are verified:
//!
//!   cargo oxide run generated_intrinsics_blackwell

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{
    DisjointSlice,
    barrier::Barrier,
    convert::{
        cvt_rn_relu_satfinite_tf32_f32, cvt_rn_relu_tf32_f32, cvt_rn_satfinite_tf32_f32,
        cvt_rn_tf32_f32, cvt_rna_satfinite_tf32_f32, cvt_rna_tf32_f32,
        cvt_rz_relu_satfinite_tf32_f32, cvt_rz_relu_tf32_f32, cvt_rz_satfinite_tf32_f32,
        cvt_rz_tf32_f32,
    },
    cuda_module, kernel, thread,
    tma::{self, TmaDescriptor},
};
use cuda_intrinsics::convert::{
    cvt_rn_satfinite_e4m3x2_f32, cvt_rn_satfinite_e5m2x2_f32, cvt_rn_satfinite_relu_e4m3x2_f32,
    cvt_rn_satfinite_relu_e5m2x2_f32,
};
use cuda_intrinsics::matrix;

#[cuda_module]
mod kernels {
    use super::*;

    /// Keeps every generated packed-FP8 conversion in device code.
    ///
    /// Launched, and its four results checked against the two formats.
    #[kernel]
    pub fn compile_fp8_conversions(mut output: DisjointSlice<u16>, low: f32, high: f32) {
        let values = [
            cvt_rn_satfinite_e4m3x2_f32(low, high),
            cvt_rn_satfinite_relu_e4m3x2_f32(low, high),
            cvt_rn_satfinite_e5m2x2_f32(low, high),
            cvt_rn_satfinite_relu_e5m2x2_f32(low, high),
        ];
        let start = thread::index_1d().get() * values.len();
        if start + values.len() <= output.len() {
            for (offset, value) in values.into_iter().enumerate() {
                // SAFETY: the bounds check covers this thread's unique slots.
                unsafe { *output.get_unchecked_mut(start + offset) = value };
            }
        }
    }

    /// Keeps every generated TF32 conversion in device code.
    ///
    /// Launched, and its ten results checked against the format.
    #[kernel]
    pub fn compile_tf32_conversions(mut output: DisjointSlice<u32>, value: f32) {
        let values = [
            cvt_rna_tf32_f32(value),
            cvt_rna_satfinite_tf32_f32(value),
            cvt_rn_tf32_f32(value),
            cvt_rn_relu_tf32_f32(value),
            cvt_rn_satfinite_tf32_f32(value),
            cvt_rn_relu_satfinite_tf32_f32(value),
            cvt_rz_tf32_f32(value),
            cvt_rz_relu_tf32_f32(value),
            cvt_rz_satfinite_tf32_f32(value),
            cvt_rz_relu_satfinite_tf32_f32(value),
        ];
        let start = thread::index_1d().get() * values.len();
        if start + values.len() <= output.len() {
            for (offset, converted) in values.into_iter().enumerate() {
                // SAFETY: the bounds check covers this thread's unique slots.
                unsafe { *output.get_unchecked_mut(start + offset) = converted };
            }
        }
    }

    /// Keeps the complete ordered `kind::f8f6f4` F32 matrix in device code.
    ///
    /// This kernel is compile-only and is never launched by the example.
    #[kernel]
    pub fn compile_ordered_f8f6f4_f32(mut output: DisjointSlice<f32>) {
        let c = [0.0; 4];
        let a = [0; 4];
        let b = [0; 4];
        let metadata = 0x4444_4444;

        // SAFETY: every lane follows the same warp-synchronous sequence. The
        // selector and ordered metadata use their only admitted forms.
        let value = unsafe {
            matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e2m1_e2m1_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e2m1_e2m3_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e2m1_e3m2_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e2m1_e4m3_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e2m1_e5m2_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e2m3_e2m1_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e2m3_e2m3_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e2m3_e3m2_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e2m3_e4m3_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e2m3_e5m2_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e3m2_e2m1_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e3m2_e2m3_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e3m2_e3m2_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e3m2_e4m3_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e3m2_e5m2_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e4m3_e2m1_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e4m3_e2m3_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e4m3_e3m2_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e4m3_e4m3_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e4m3_e5m2_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e5m2_e2m1_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e5m2_e2m3_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e5m2_e3m2_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e5m2_e4m3_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e5m2_e5m2_f32(
                c, a, b, metadata, 0,
            )[0]
        };

        if let Some((slot, _)) = output.get_mut_indexed() {
            *slot = value;
        }
    }

    /// Keeps the complete ordered sparse F16 matrix in device code.
    ///
    /// This kernel is compile-only and is never launched by the example.
    #[kernel]
    pub fn compile_ordered_f8f6f4_f16(mut output: DisjointSlice<u32>) {
        let c = [0; 2];
        let a = [0; 4];
        let b = [0; 4];
        let metadata = 0x4444_4444;

        // SAFETY: every lane follows the same warp-synchronous sequence. The
        // selector and ordered metadata use their only admitted forms.
        let values = unsafe {
            [
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e2m1_e2m1_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e2m1_e2m3_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e2m1_e3m2_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e2m1_e4m3_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e2m1_e5m2_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e2m3_e2m1_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e2m3_e2m3_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e2m3_e3m2_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e2m3_e4m3_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e2m3_e5m2_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e3m2_e2m1_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e3m2_e2m3_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e3m2_e3m2_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e3m2_e4m3_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e3m2_e5m2_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e4m3_e2m1_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e4m3_e2m3_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e4m3_e3m2_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e4m3_e4m3_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e4m3_e5m2_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e5m2_e2m1_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e5m2_e2m3_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e5m2_e3m2_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e5m2_e4m3_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e5m2_e5m2_f16(
                    c, a, b, metadata, 0,
                ),
            ]
        };
        let mut value = 0;
        for lanes in values {
            value ^= lanes[0] ^ lanes[1];
        }

        if let Some((slot, _)) = output.get_mut_indexed() {
            *slot = value;
        }
    }

    /// Keeps every dense `kind::f8f6f4` F32 MMA form in device code.
    ///
    /// This kernel is compile-only and is never launched by the example.
    #[kernel]
    pub fn compile_dense_f8f6f4_f32(mut output: DisjointSlice<f32>) {
        let c = [0.0; 4];
        let a = [0; 4];
        let b = [0; 2];

        // SAFETY: every lane follows the same warp-synchronous sequence.
        let value = unsafe {
            matrix::mma_m16n8k32_f32_e2m1_e2m1(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e2m1_e2m3(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e2m1_e3m2(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e2m1_e4m3(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e2m1_e5m2(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e2m3_e2m1(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e2m3_e2m3(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e2m3_e3m2(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e2m3_e4m3(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e2m3_e5m2(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e3m2_e2m1(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e3m2_e2m3(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e3m2_e3m2(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e3m2_e4m3(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e3m2_e5m2(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e4m3_e2m1(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e4m3_e2m3(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e4m3_e3m2(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e4m3_e4m3(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e4m3_e5m2(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e5m2_e2m1(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e5m2_e2m3(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e5m2_e3m2(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e5m2_e4m3(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e5m2_e5m2(c, a, b)[0]
        };

        if let Some((slot, _)) = output.get_mut_indexed() {
            *slot = value;
        }
    }

    /// Keeps every dense `kind::f8f6f4` F16 MMA form in device code.
    ///
    /// This kernel is compile-only and is never launched by the example.
    #[kernel]
    pub fn compile_dense_f8f6f4_f16(mut output: DisjointSlice<u32>) {
        let c = [0; 2];
        let a = [0; 4];
        let b = [0; 2];

        // SAFETY: every lane follows the same warp-synchronous sequence.
        let values = unsafe {
            [
                matrix::mma_m16n8k32_f16_e2m1_e2m1(c, a, b),
                matrix::mma_m16n8k32_f16_e2m1_e2m3(c, a, b),
                matrix::mma_m16n8k32_f16_e2m1_e3m2(c, a, b),
                matrix::mma_m16n8k32_f16_e2m1_e4m3(c, a, b),
                matrix::mma_m16n8k32_f16_e2m1_e5m2(c, a, b),
                matrix::mma_m16n8k32_f16_e2m3_e2m1(c, a, b),
                matrix::mma_m16n8k32_f16_e2m3_e2m3(c, a, b),
                matrix::mma_m16n8k32_f16_e2m3_e3m2(c, a, b),
                matrix::mma_m16n8k32_f16_e2m3_e4m3(c, a, b),
                matrix::mma_m16n8k32_f16_e2m3_e5m2(c, a, b),
                matrix::mma_m16n8k32_f16_e3m2_e2m1(c, a, b),
                matrix::mma_m16n8k32_f16_e3m2_e2m3(c, a, b),
                matrix::mma_m16n8k32_f16_e3m2_e3m2(c, a, b),
                matrix::mma_m16n8k32_f16_e3m2_e4m3(c, a, b),
                matrix::mma_m16n8k32_f16_e3m2_e5m2(c, a, b),
                matrix::mma_m16n8k32_f16_e4m3_e2m1(c, a, b),
                matrix::mma_m16n8k32_f16_e4m3_e2m3(c, a, b),
                matrix::mma_m16n8k32_f16_e4m3_e3m2(c, a, b),
                matrix::mma_m16n8k32_f16_e4m3_e4m3(c, a, b),
                matrix::mma_m16n8k32_f16_e4m3_e5m2(c, a, b),
                matrix::mma_m16n8k32_f16_e5m2_e2m1(c, a, b),
                matrix::mma_m16n8k32_f16_e5m2_e2m3(c, a, b),
                matrix::mma_m16n8k32_f16_e5m2_e3m2(c, a, b),
                matrix::mma_m16n8k32_f16_e5m2_e4m3(c, a, b),
                matrix::mma_m16n8k32_f16_e5m2_e5m2(c, a, b),
            ]
        };
        let mut value = 0;
        for lanes in values {
            value ^= lanes[0] ^ lanes[1];
        }

        if let Some((slot, _)) = output.get_mut_indexed() {
            *slot = value;
        }
    }

    /// Keeps every standard FP8 register-MMA form in device code.
    ///
    /// This kernel is compile-only and is never launched by the example.
    #[kernel]
    pub fn compile_standard_fp8_mma(mut output: DisjointSlice<u32>) {
        let c_f16 = [0; 2];
        let c_f32 = [0.0; 4];
        let a_k16 = [0; 2];
        let b_k16 = 0;
        let a_k32 = [0; 4];
        let b_k32 = [0; 2];

        // SAFETY: every lane follows the same warp-synchronous sequence.
        let f16_values = unsafe {
            [
                matrix::mma_m16n8k16_fp8_f16_e4m3_e4m3(c_f16, a_k16, b_k16),
                matrix::mma_m16n8k16_fp8_f16_e4m3_e5m2(c_f16, a_k16, b_k16),
                matrix::mma_m16n8k16_fp8_f16_e5m2_e4m3(c_f16, a_k16, b_k16),
                matrix::mma_m16n8k16_fp8_f16_e5m2_e5m2(c_f16, a_k16, b_k16),
                matrix::mma_m16n8k32_fp8_f16_e4m3_e4m3(c_f16, a_k32, b_k32),
                matrix::mma_m16n8k32_fp8_f16_e4m3_e5m2(c_f16, a_k32, b_k32),
                matrix::mma_m16n8k32_fp8_f16_e5m2_e4m3(c_f16, a_k32, b_k32),
                matrix::mma_m16n8k32_fp8_f16_e5m2_e5m2(c_f16, a_k32, b_k32),
            ]
        };
        let f32_values = unsafe {
            [
                matrix::mma_m16n8k16_fp8_f32_e4m3_e4m3(c_f32, a_k16, b_k16),
                matrix::mma_m16n8k16_fp8_f32_e4m3_e5m2(c_f32, a_k16, b_k16),
                matrix::mma_m16n8k16_fp8_f32_e5m2_e4m3(c_f32, a_k16, b_k16),
                matrix::mma_m16n8k16_fp8_f32_e5m2_e5m2(c_f32, a_k16, b_k16),
                matrix::mma_m16n8k32_fp8_f32_e4m3_e4m3(c_f32, a_k32, b_k32),
                matrix::mma_m16n8k32_fp8_f32_e4m3_e5m2(c_f32, a_k32, b_k32),
                matrix::mma_m16n8k32_fp8_f32_e5m2_e4m3(c_f32, a_k32, b_k32),
                matrix::mma_m16n8k32_fp8_f32_e5m2_e5m2(c_f32, a_k32, b_k32),
            ]
        };

        let mut value = 0;
        for lanes in f16_values {
            value ^= lanes[0] ^ lanes[1];
        }
        for lanes in f32_values {
            value ^= lanes[0].to_bits() ^ lanes[1].to_bits();
            value ^= lanes[2].to_bits() ^ lanes[3].to_bits();
        }

        if let Some((slot, _)) = output.get_mut_indexed() {
            *slot = value;
        }
    }

    /// Keeps every Blackwell `ldmatrix` variant in device code.
    ///
    /// This kernel is compile-only and is never launched by the example.
    #[kernel]
    pub unsafe fn compile_blackwell_ldmatrix(input: *const u8, output: *mut u32) {
        // SAFETY: every lane follows the same sequence. A real caller must
        // provide 16-byte-aligned shared addresses with 32 readable bytes.
        let values = unsafe {
            [
                matrix::ldmatrix_m16n16_x1_trans_b8(input)[0],
                matrix::ldmatrix_m16n16_x1_trans_b8x16_b4x16_p64(input)[0],
                matrix::ldmatrix_m16n16_x1_trans_b8x16_b6x16_p32(input)[0],
                matrix::ldmatrix_m16n16_x2_trans_b8(input)[0],
                matrix::ldmatrix_m16n16_x2_trans_b8x16_b4x16_p64(input)[0],
                matrix::ldmatrix_m16n16_x2_trans_b8x16_b6x16_p32(input)[0],
                matrix::ldmatrix_m8n16_x1_b8x16_b4x16_p64(input),
                matrix::ldmatrix_m8n16_x1_b8x16_b6x16_p32(input),
                matrix::ldmatrix_m8n16_x2_b8x16_b4x16_p64(input)[0],
                matrix::ldmatrix_m8n16_x2_b8x16_b6x16_p32(input)[0],
                matrix::ldmatrix_m8n16_x4_b8x16_b4x16_p64(input)[0],
                matrix::ldmatrix_m8n16_x4_b8x16_b6x16_p32(input)[0],
            ]
        };

        for (index, value) in values.into_iter().enumerate() {
            // SAFETY: a real caller must provide space for all 12 results.
            unsafe { output.add(index).write(value) };
        }
    }

    /// Compile-only coverage for the TMA compatibility API.
    #[kernel]
    pub unsafe fn compile_tma_compatibility(
        shared: *mut u8,
        tensor_map: *const TmaDescriptor,
        cta_mask: u16,
    ) {
        // The barrier is declared here rather than taken as a parameter. It
        // lives in shared memory, which is allocated per block at launch, so
        // the host has no address it could pass. Taking one as a parameter put
        // `.ptr .shared` on the entry and left the whole module unloadable.
        static mut BAR: Barrier = Barrier::UNINIT;
        let barrier = &raw mut BAR;

        // This kernel is never launched with these placeholder addresses.
        unsafe {
            tma::cp_async_bulk_tensor_1d_g2s(shared, tensor_map, 0, barrier);
            tma::cp_async_bulk_tensor_2d_g2s(shared, tensor_map, 0, 0, barrier);
            tma::cp_async_bulk_tensor_2d_g2s_multicast(shared, tensor_map, 0, 0, barrier, cta_mask);
            tma::cp_async_bulk_tensor_3d_g2s(shared, tensor_map, 0, 0, 0, barrier);
            tma::cp_async_bulk_tensor_4d_g2s(shared, tensor_map, 0, 0, 0, 0, barrier);
            tma::cp_async_bulk_tensor_5d_g2s(shared, tensor_map, 0, 0, 0, 0, 0, barrier);

            tma::cp_async_bulk_tensor_1d_s2g(shared, tensor_map, 0);
            tma::cp_async_bulk_tensor_2d_s2g(shared, tensor_map, 0, 0);
            tma::cp_async_bulk_tensor_3d_s2g(shared, tensor_map, 0, 0, 0);
            tma::cp_async_bulk_tensor_4d_s2g(shared, tensor_map, 0, 0, 0, 0);
            tma::cp_async_bulk_tensor_5d_s2g(shared, tensor_map, 0, 0, 0, 0, 0);
        }
        tma::cp_async_bulk_commit_group();
        tma::cp_async_bulk_wait_group(0);
        tma::cp_async_bulk_wait_group_read(0);
    }
}

/// One packed-FP8 case: the two inputs and what each of the four conversions
/// must return. Derivations are in the table below.
struct Fp8Case {
    lo: f32,
    hi: f32,
    e4m3: u16,
    relu_e4m3: u16,
    e5m2: u16,
    relu_e5m2: u16,
}

/// e4m3 is s.eeee.mmm with exponent bias 7, e5m2 is s.eeeee.mm with bias 15,
/// and the first argument occupies the low byte.
///
/// | value | e4m3 | e5m2 | why |
/// |---|---|---|---|
/// | 1.0 | 0x38 | 0x3C | exponent 0, zero mantissa |
/// | 2.0 | 0x40 | 0x40 | exponent 1 |
/// | -1.0 | 0xB8 | 0xBC | sign bit over 1.0 |
/// | 448.0 | 0x7E | 0x5F | 1.75 x 2^8, the largest finite e4m3 |
/// | 1e30 | 0x7E | 0x7B | saturates to the largest finite value |
/// | 0.0 | 0x00 | 0x00 | zero |
const FP8_CASES: [Fp8Case; 3] = [
    Fp8Case {
        lo: 1.0,
        hi: 2.0,
        e4m3: 0x4038,
        relu_e4m3: 0x4038,
        e5m2: 0x403c,
        relu_e5m2: 0x403c,
    },
    // relu replaces the negative low lane with zero and leaves the high lane.
    Fp8Case {
        lo: -1.0,
        hi: 448.0,
        e4m3: 0x7eb8,
        relu_e4m3: 0x7e00,
        e5m2: 0x5fbc,
        relu_e5m2: 0x5f00,
    },
    Fp8Case {
        lo: 1.0e30,
        hi: 0.0,
        e4m3: 0x007e,
        relu_e4m3: 0x007e,
        e5m2: 0x007b,
        relu_e5m2: 0x007b,
    },
];

/// TF32 keeps the f32 exponent and truncates the mantissa to 10 bits, so each
/// value here is already representable and every rounding mode agrees. The
/// four relu forms sit at indices 3, 5, 7 and 9 of the kernel's output.
const TF32_CASES: [(f32, u32, u32); 3] = [
    (1.0, 0x3f80_0000, 0x3f80_0000),
    // 1 + 2^-10, the smallest step a 10-bit mantissa still holds exactly,
    // written as the pattern the conversion has to return unchanged.
    (f32::from_bits(0x3f80_2000), 0x3f80_2000, 0x3f80_2000),
    (-3.5, 0xc060_0000, 0x0000_0000),
];

const RELU_TF32_INDICES: [usize; 4] = [3, 5, 7, 9];

fn main() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("failed to load device module");
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut failures = 0usize;

    println!("=== generated Blackwell conversions, checked on device ===\n");

    for case in &FP8_CASES {
        let mut out = DeviceBuffer::<u16>::zeroed(&stream, 4).expect("alloc fp8 output");
        // SAFETY: the three arguments match the kernel's slice and two scalars,
        // and `out` holds the four results one thread writes.
        unsafe { module.compile_fp8_conversions(&stream, cfg, &mut out, case.lo, case.hi) }
            .expect("fp8 launch failed");
        let got = out.to_host_vec(&stream).expect("fp8 readback failed");
        let want = [case.e4m3, case.relu_e4m3, case.e5m2, case.relu_e5m2];
        let names = ["e4m3x2", "relu_e4m3x2", "e5m2x2", "relu_e5m2x2"];
        for ((got, want), name) in got.iter().zip(want).zip(names) {
            let mark = if *got == want { "ok" } else { "MISMATCH" };
            if *got != want {
                failures += 1;
            }
            println!(
                "  {name:<12} ({:>10}, {:>10}) = 0x{got:04x}  want 0x{want:04x}  {mark}",
                case.lo, case.hi
            );
        }
    }

    for (value, want_plain, want_relu) in TF32_CASES {
        let mut out = DeviceBuffer::<u32>::zeroed(&stream, 10).expect("alloc tf32 output");
        // SAFETY: the two arguments match the kernel's slice and scalar, and
        // `out` holds the ten results one thread writes.
        unsafe { module.compile_tf32_conversions(&stream, cfg, &mut out, value) }
            .expect("tf32 launch failed");
        let got = out.to_host_vec(&stream).expect("tf32 readback failed");
        for (index, got) in got.iter().enumerate() {
            let want = if RELU_TF32_INDICES.contains(&index) {
                want_relu
            } else {
                want_plain
            };
            let mark = if *got == want { "ok" } else { "MISMATCH" };
            if *got != want {
                failures += 1;
            }
            println!(
                "  tf32[{index}]      ({value:>10})              = 0x{got:08x}  want 0x{want:08x}  {mark}"
            );
        }
    }

    if failures == 0 {
        println!("\nPASS: every conversion matched, and the sparse MMA and TMA kernels compiled");
    } else {
        println!("\nFAIL: {failures} conversions disagreed with the format");
        std::process::exit(1);
    }
}
