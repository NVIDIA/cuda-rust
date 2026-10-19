// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// `requires` relations are evaluated in u64, so v1 accepts only unsigned
// integer scalar parameters; signed scalars are rejected at expansion time.

#[cuda_macros::cuda_module]
mod kernels {
    #[cuda_macros::kernel]
    #[cuda_macros::launch_contract(
        domain = 1,
        block = (64, 1, 1),
        requires = (input.len() >= n),
    )]
    pub fn scaled(n: i32, input: &[f32], mut output: cuda_device::DisjointSlice<f32>) {
        let _ = (n, input, &mut output);
    }
}

fn main() {}
