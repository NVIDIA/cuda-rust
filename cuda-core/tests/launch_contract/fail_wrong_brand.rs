// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_core::{
    BlockRequirement, DynamicSharedMemoryRequirement, KernelLaunchContract, LaunchConfig1D,
    LaunchContractSpec, PreparedLaunch,
};

struct KernelA;
struct KernelB;

macro_rules! contract {
    ($kernel:ty, $name:literal) => {
        impl KernelLaunchContract for $kernel {
            type Config = LaunchConfig1D;

            const SPEC: LaunchContractSpec = LaunchContractSpec::new(
                $name,
                BlockRequirement::Exact((32, 1, 1)),
                DynamicSharedMemoryRequirement::Exact {
                    bytes: 0,
                    min_alignment: 1,
                },
            );
        }
    };
}

contract!(KernelA, "kernel_a");
contract!(KernelB, "kernel_b");

fn launch_b(_: &PreparedLaunch<KernelB>) {}

fn launch_a_as_b(prepared: &PreparedLaunch<KernelA>) {
    launch_b(prepared);
}

fn main() {}
