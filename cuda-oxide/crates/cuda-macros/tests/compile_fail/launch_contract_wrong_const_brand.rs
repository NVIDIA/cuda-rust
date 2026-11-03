// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_device::{cuda_module, kernel, launch_contract};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1))]
    pub fn const_kernel<const N: usize>(input: &[u32]) {
        let _ = (N, input);
    }
}

fn launch_eight_with_four(
    module: &kernels::LoadedModule,
    stream: &cuda_core::CudaStream,
    prepared: &cuda_core::PreparedLaunch<kernels::__const_kernel_CudaKernel<4>>,
    input: &cuda_core::DeviceBuffer<u32>,
) {
    let _ = module.const_kernel::<8>(stream, prepared, input);
}

fn main() {}
