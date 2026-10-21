/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile-fail probe: prevent external code from defining a wait policy that omits required ordering.
use cuda_device::barrier::Barrier;
use cute_rs::sm100_attention::pipeline::WaitPolicy;

/// Invalid external wait policy with no ordering guarantees.

#[derive(Clone, Copy)]
struct Unordered;

impl WaitPolicy for Unordered {
    unsafe fn wait(_: *const Barrier, _: u32) {}
}

/// Unused host entry point; the device or rejection probe is only type-checked.

fn main() {}
