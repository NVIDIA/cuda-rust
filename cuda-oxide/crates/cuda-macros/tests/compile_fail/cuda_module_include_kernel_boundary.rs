// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_core::CudaStream;
use cuda_core::simt::LaunchConfig;
use cuda_macros::cuda_module;

#[cuda_module]
mod kernels {
    #[cuda_macros::kernel]
    pub fn root(value: u32) {
        let _ = value;
    }

    include!("cuda_module_include_kernel_boundary_items.rs");
}

fn undiscovered(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    config: LaunchConfig,
) {
    let _ = module.from_include(stream, config, 1u32);
}

fn main() {}
