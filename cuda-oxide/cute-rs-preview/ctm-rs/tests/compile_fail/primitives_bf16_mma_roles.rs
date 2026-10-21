/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Reject swapped BF16 matrix operand roles.
use ctm_rs::mma::{Accumulator16x8, Bf16A, Bf16B, mma_bf16};

/// Roles remain distinct even though both operands contain packed BF16 values.
fn wrong(a: Bf16A, b: Bf16B) {
    unsafe { mma_bf16(Accumulator16x8::zero(), b, a) };
}

/// The invalid device call is type checked without executing a GPU operation.
fn main() {}
