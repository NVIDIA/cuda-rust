// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use cuda_core::KernelLaunchContract;
use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};

trait Policy {
    const MAX_THREADS: u32;
}

enum PolicyWithFiniteLimit {}

impl Policy for PolicyWithFiniteLimit {
    const MAX_THREADS: u32 = 1024;
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(P::MAX_THREADS)]
    #[launch_contract(
        domain = 3,
        block = (4_294_967_295, 4_294_967_295, 4_294_967_295),
    )]
    pub fn configured<P: Policy>() {}
}

const _: cuda_core::LaunchContractSpec =
    <kernels::__configured_CudaKernel<PolicyWithFiniteLimit> as KernelLaunchContract>::SPEC;

fn main() {}
