// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use cuda_core::{BlockRequirement, KernelLaunchContract};
use cuda_device::{cuda_module, device, kernel, launch_bounds, launch_contract};

trait Policy {
    const MAX_THREADS: u32;
    const MIN_BLOCKS: u32;
    const UNROLL: u32;
}

enum SmallPolicy {}

impl Policy for SmallPolicy {
    const MAX_THREADS: u32 = 64;
    const MIN_BLOCKS: u32 = 2;
    const UNROLL: u32 = 2;
}

enum WidePolicy {}

impl Policy for WidePolicy {
    const MAX_THREADS: u32 = 256;
    const MIN_BLOCKS: u32 = 1;
    const UNROLL: u32 = 4;
}

#[kernel]
#[launch_bounds(P::MAX_THREADS * 2, P::MIN_BLOCKS)]
fn configured<P: Policy>() {
    let mut index = 0;
    #[unroll(P::UNROLL)]
    while index < 8 {
        index += 1;
    }
}

#[device]
fn configured_helper<P: Policy>() {
    let mut index = 0;
    #[unroll(P::UNROLL)]
    while index < 8 {
        index += 1;
    }
}

#[kernel]
fn function_local_policy_value() {
    const FACTOR: u32 = 4;
    let mut index = 0;
    #[unroll(FACTOR)]
    while index < 8 {
        index += 1;
    }
}

#[cuda_module]
mod contracted {
    use super::*;

    #[kernel]
    #[launch_bounds(P::MAX_THREADS, P::MIN_BLOCKS)]
    #[launch_contract(domain = 1)]
    pub fn configured<P: Policy>() {}
}

const SMALL_CONTRACT_MAX: u32 =
    match <contracted::__configured_CudaKernel<SmallPolicy> as KernelLaunchContract>::SPEC.block() {
        BlockRequirement::MaxThreads(max) => max,
        BlockRequirement::Exact(_) => panic!("policy contract unexpectedly requires an exact block"),
    };
const WIDE_CONTRACT_MAX: u32 =
    match <contracted::__configured_CudaKernel<WidePolicy> as KernelLaunchContract>::SPEC.block() {
        BlockRequirement::MaxThreads(max) => max,
        BlockRequirement::Exact(_) => panic!("policy contract unexpectedly requires an exact block"),
    };

const _: () = assert!(SMALL_CONTRACT_MAX == 64);
const _: () = assert!(WIDE_CONTRACT_MAX == 256);

fn prepared_paths_are_policy_branded(module: &contracted::LoadedModule) {
    let _: Result<
        cuda_core::PreparedLaunch<contracted::__configured_CudaKernel<SmallPolicy>>,
        cuda_core::LaunchContractError,
    > = module.prepare_configured::<SmallPolicy>(cuda_core::LaunchConfig1D::new(1, 64, 0));
    let _: Result<
        cuda_core::PreparedLaunch<contracted::__configured_CudaKernel<WidePolicy>>,
        cuda_core::LaunchContractError,
    > = module.prepare_configured::<WidePolicy>(cuda_core::LaunchConfig1D::new(1, 256, 0));
}

#[cuda_module]
mod nested {
    use super::*;

    pub mod stage {
        use super::*;

        #[kernel]
        #[launch_bounds(P::MAX_THREADS, P::MIN_BLOCKS)]
        pub fn nested_configured<P: Policy>() {
            let mut index = 0;
            #[unroll(P::UNROLL)]
            while index < 8 {
                index += 1;
            }
        }
    }
}

fn main() {
    assert_ne!(SMALL_CONTRACT_MAX, WIDE_CONTRACT_MAX);
    let _ = prepared_paths_are_policy_branded;
}
