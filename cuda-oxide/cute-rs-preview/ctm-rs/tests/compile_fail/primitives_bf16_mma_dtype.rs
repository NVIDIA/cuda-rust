/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Reject implicit reinterpretation between BF16 and FP16 matrix operands.
use ctm_rs::mma::{Accumulator16x8, Bf16A, Bf16B, F16A, F16B, mma_bf16, mma_f16};

/// Neither instruction accepts the other dtype's packed register carrier.
fn wrong(a: Bf16A, b: Bf16B, half_a: F16A, half_b: F16B) {
    unsafe { mma_bf16(Accumulator16x8::zero(), half_a, half_b) };
    unsafe { mma_f16(Accumulator16x8::zero(), a, b) };
}

/// The invalid device calls are type checked without executing GPU operations.
fn main() {}
