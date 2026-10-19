// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// `.len()` in a `requires` relation is only available on slice-like
// parameters; `n` is a scalar.

#[cuda_macros::cuda_module]
mod kernels {
    #[cuda_macros::kernel]
    #[cuda_macros::launch_contract(
        domain = 1,
        block = (64, 1, 1),
        requires = (n.len() >= 4),
    )]
    pub fn scaled(n: u32, mut output: cuda_device::DisjointSlice<f32>) {
        let _ = (n, &mut output);
    }
}

fn main() {}
