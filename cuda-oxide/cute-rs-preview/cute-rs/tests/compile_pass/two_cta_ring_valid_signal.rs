/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile-pass probe: type-check the completion signals permitted by each ring policy.
use cute_rs::sm100_attention::pipeline::{
    AsyncCompletion, SharedReuse, SharedToAsync, TensorMemory, TwoCtaRing,
};

/// Type-check the completion signals permitted by each ring policy. This device-only probe is never executed.

unsafe fn valid_completion_mechanisms(
    tensor: TwoCtaRing<2, TensorMemory>,
    shared: TwoCtaRing<2, SharedToAsync, SharedReuse>,
    copies: TwoCtaRing<2, AsyncCompletion>,
) {
    unsafe {
        tensor.mma_full(0);
        tensor.mma_empty(0);
        tensor.thread_full(0);
        tensor.thread_empty(0);
        shared.thread_full(0);
        shared.mma_empty(0);
        copies.expect(0, 16);
        copies.mma_full(0);
        copies.mma_empty(0);
    }
}

/// Unused host entry point; the device or rejection probe is only type-checked.

fn main() {}
