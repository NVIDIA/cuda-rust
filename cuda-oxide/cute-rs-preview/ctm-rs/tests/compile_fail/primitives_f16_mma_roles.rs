/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Compile-fail probe: reject swapped FP16 MMA A and B register operands.
use ctm_rs::mma::{Accumulator16x8, F16A, F16B, mma_f16};

/// Reject swapped FP16 MMA A and B register operands.

fn wrong(a: F16A, b: F16B) {
    unsafe { mma_f16(Accumulator16x8::zero(), b, a) };
}

/// Unused host entry point; the device or rejection probe is only type-checked.

fn main() {}
