// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_core::{
    BlockRequirement, DynamicSharedMemoryRequirement, KernelLaunchContract, LaunchConfig1D,
    LaunchConfig2D, LaunchContractSpec,
};

struct OneDimensionalKernel;

impl KernelLaunchContract for OneDimensionalKernel {
    type Config = LaunchConfig1D;

    const SPEC: LaunchContractSpec = LaunchContractSpec::new(
        "one_dimensional",
        BlockRequirement::Exact((32, 1, 1)),
        DynamicSharedMemoryRequirement::Exact {
            bytes: 0,
            min_alignment: 1,
        },
    );
}

fn prepare(_: <OneDimensionalKernel as KernelLaunchContract>::Config) {}

fn main() {
    prepare(LaunchConfig2D::new((1, 1), (32, 1), 0));
}
