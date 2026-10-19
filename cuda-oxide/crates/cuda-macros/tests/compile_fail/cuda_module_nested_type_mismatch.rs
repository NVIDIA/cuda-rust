// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_core::CudaStream;
use cuda_core::simt::LaunchConfig;
use cuda_macros::cuda_module;

#[cuda_module]
mod kernels {
    #[derive(Clone, Copy)]
    pub struct Params {
        pub value: u32,
    }

    pub mod child {
        #[derive(Clone, Copy)]
        pub struct Params {
            pub values: [u32; 4],
        }

        #[cuda_macros::kernel]
        pub fn child_typed(params: Params) {
            let _ = params.values;
        }
    }
}

fn wrong_type(
    child: &kernels::child::LoadedModule,
    stream: &CudaStream,
    config: LaunchConfig,
) {
    let _ = child.child_typed(stream, config, kernels::Params { value: 1 });
}

fn main() {}
