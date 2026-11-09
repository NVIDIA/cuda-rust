/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Explicit instruction packing and numeric conversion.

/// Convert two FP32 values to packed FP16 using nearest-even rounding.
///
/// `lo` occupies bits `0..16`, and `hi` occupies bits `16..32`. This preserves
/// the argument order of `cuda_device::convert::cvt_f16x2_f32` and does not
/// introduce flush-to-zero mode or an additional conversion.
#[must_use]
#[inline(always)]
pub fn cvt_f16x2_f32(lo: f32, hi: f32) -> u32 {
    cuda_device::convert::cvt_f16x2_f32(lo, hi)
}

/// Round FP32 to raw BF16 storage bits with round-to-nearest, ties-to-even.
/// Signed zeros and infinities are preserved; NaNs remain NaNs.
#[must_use]
#[inline(always)]
pub fn f32_to_bf16_rne(value: f32) -> u16 {
    cuda_device::convert::f32_to_bf16_rne(value)
}

/// Widen raw BF16 storage bits to FP32 exactly, retaining NaN payload bits.
#[must_use]
#[inline(always)]
pub fn bf16_to_f32(bits: u16) -> f32 {
    cuda_device::convert::bf16_to_f32(bits)
}

/// Convert two FP32 values to packed FP8 E4M3 (`e4m3x2`) with nearest-even finite saturation.
///
/// `lo` occupies bits `0..8` (low byte), and `hi` occupies bits `8..16` (high byte).
/// Preserves the argument order of `cuda_device::convert::cvt_rn_satfinite_e4m3x2_f32`.
#[must_use]
#[inline(always)]
pub fn cvt_rn_satfinite_e4m3x2_f32(lo: f32, hi: f32) -> u16 {
    cuda_device::convert::cvt_rn_satfinite_e4m3x2_f32(lo, hi)
}
