// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_macros::cuda_launch_async;

#[allow(non_camel_case_types)]
struct __raw_CudaKernel;

impl cuda_host::CudaKernel for __raw_CudaKernel {
    const PTX_NAME: &'static str = "raw";
}

struct FakeModule;

impl FakeModule {
    fn load_function(&self, _name: &str) -> Result<cuda_core::CudaFunction, ()> {
        unreachable!("this trybuild fixture never executes module loading")
    }
}

mod cuda_async {
    pub mod simt {
        pub mod launch {
            use cuda_core::CudaFunction;
            use cuda_core::simt::LaunchConfig;
            use std::sync::Arc;

            pub struct AsyncKernelLaunchBuilder;

            impl AsyncKernelLaunchBuilder {
                pub fn new(_function: Arc<CudaFunction>) -> Self {
                    Self
                }

                pub unsafe fn finalize_unchecked(self, _config: LaunchConfig) {}
            }
        }
    }
}

fn launch_with_unsafe(module: &FakeModule) {
    let _ = unsafe {
        cuda_launch_async! {
            kernel: raw,
            module: module,
            config: cuda_core::simt::LaunchConfig {
                grid_dim: (1, 2, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: 0,
            },
            args: [],
        }
    };
}

fn main() {}
