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

    pub mod child {
        #[cuda_macros::kernel]
        pub(super) fn scoped(value: u32) {
            let _ = value;
        }
    }
}

fn outside_parent_scope(
    child: &kernels::child::LoadedModule,
    stream: &CudaStream,
    config: LaunchConfig,
) {
    let _ = child.scoped(stream, config, 1u32);
}

fn main() {}
