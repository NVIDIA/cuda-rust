// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A `requires` relation may only reference the kernel's own parameters; `n`
// does not exist on this kernel.

#[cuda_macros::cuda_module]
mod kernels {
    #[cuda_macros::kernel]
    #[cuda_macros::launch_contract(
        domain = 1,
        block = (64, 1, 1),
        requires = (input.len() >= n),
    )]
    pub fn scaled(input: &[f32], mut output: cuda_device::DisjointSlice<f32>) {
        let _ = (input, &mut output);
    }
}

fn main() {}
