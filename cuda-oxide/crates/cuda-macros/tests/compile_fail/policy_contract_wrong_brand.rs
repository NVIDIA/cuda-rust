// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};

trait Policy {
    const MAX_THREADS: u32;
}

enum SmallPolicy {}
impl Policy for SmallPolicy {
    const MAX_THREADS: u32 = 64;
}

enum WidePolicy {}
impl Policy for WidePolicy {
    const MAX_THREADS: u32 = 256;
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(P::MAX_THREADS)]
    #[launch_contract(domain = 1)]
    pub fn configured<P: Policy>() {}
}

fn needs_wide(
    _: &cuda_core::PreparedLaunch<kernels::__configured_CudaKernel<WidePolicy>>,
) {
}

fn wrong_policy(
    small: &cuda_core::PreparedLaunch<kernels::__configured_CudaKernel<SmallPolicy>>,
) {
    needs_wide(small);
}

fn main() {}
