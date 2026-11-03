// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Well-formed `requires` relations on a standalone kernel (no #[cuda_module])
// must keep compiling: the attribute validates them for well-formedness, and
// only #[cuda_module]-generated launchers enforce them at runtime.

#[cuda_macros::kernel]
#[cuda_macros::launch_contract(
    domain = 1,
    block = (64, 1, 1),
    requires = (input.len() >= n, output.len() >= n),
)]
pub fn scaled(n: u32, input: &[f32], mut output: cuda_device::DisjointSlice<f32>) {
    let _ = (n, input, &mut output);
}

fn main() {}
