// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_macros::{cuda_module, kernel, launch_contract};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn contracted(input: &[u32]) {
        let _ = input;
    }
}

fn unchecked_without_unsafe(
    module: &kernels::LoadedModule,
    stream: &cuda_core::CudaStream,
    input: &cuda_core::DeviceBuffer<u32>,
) {
    module
        .contracted_unchecked(
            stream,
            cuda_core::LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: 0,
            },
            input,
        )
        .unwrap();
}

fn main() {}
