/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile-fail probe: reject signaling mechanisms incompatible with each ring policy.
use cute_rs::sm100_attention::pipeline::{
    AsyncCompletion, LocalMemory, SharedReuse, SharedToAsync, TensorMemory, TwoCtaRing,
};

/// Reject signaling mechanisms incompatible with each ring policy.

unsafe fn wrong_completion_mechanisms(
    tensor: TwoCtaRing<2, TensorMemory>,
    shared: TwoCtaRing<2, SharedToAsync, SharedReuse>,
    copies: TwoCtaRing<2, AsyncCompletion>,
    local: TwoCtaRing<2, LocalMemory>,
) {
    unsafe {
        tensor.expect(0, 16);
        shared.mma_full(0);
        shared.thread_empty(0);
        copies.thread_full(0);
        local.mma_empty(0);
    }
}

/// Unused host entry point; the device or rejection probe is only type-checked.

fn main() {}
