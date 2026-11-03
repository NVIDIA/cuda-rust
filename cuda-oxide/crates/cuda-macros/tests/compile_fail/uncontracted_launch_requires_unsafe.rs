// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_macros::{cuda_module, kernel};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn uncontracted(value: u32) {
        let _ = value;
    }
}

fn launch_without_unsafe(
    module: &kernels::LoadedModule,
    stream: &cuda_core::CudaStream,
) {
    module
        .uncontracted(
            stream,
            cuda_core::LaunchConfig {
                grid_dim: (1, 2, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: 0,
            },
            7,
        )
        .unwrap();
}

fn main() {}
