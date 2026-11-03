// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// `requires` arithmetic is limited to `+`, `-`, and `*`; division is not
// part of the v1 relation grammar.

#[cuda_macros::cuda_module]
mod kernels {
    #[cuda_macros::kernel]
    #[cuda_macros::launch_contract(
        domain = 1,
        block = (64, 1, 1),
        requires = (input.len() / 2 >= n),
    )]
    pub fn scaled(n: u32, input: &[f32], mut output: cuda_device::DisjointSlice<f32>) {
        let _ = (n, input, &mut output);
    }
}

fn main() {}
