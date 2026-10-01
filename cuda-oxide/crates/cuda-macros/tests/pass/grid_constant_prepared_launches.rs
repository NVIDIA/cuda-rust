// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]

use cuda_device::{cuda_module, kernel, launch_contract};

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Payload([u32; 32]);

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_contract(domain = 1, block = (1, 1, 1))]
    pub fn consume(#[grid_constant] payload: &Payload) {
        let _ = payload.0[31];
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (1, 1, 1))]
    pub fn generic<T: Copy>(#[grid_constant] payload: &T) {
        let _ = *payload;
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (1, 1, 1))]
    pub fn ordinary(payload: Payload) {
        let _ = payload.0[31];
    }
}

fn launch(
    module: &kernels::LoadedModule,
    stream: &cuda_core::CudaStream,
    prepared: &cuda_core::PreparedLaunch<kernels::__consume_CudaKernel>,
    generic: &cuda_core::PreparedLaunch<kernels::__generic_CudaKernel<Payload>>,
    ordinary: &cuda_core::PreparedLaunch<kernels::__ordinary_CudaKernel>,
) {
    let payload = Payload([17; 32]);
    // SAFETY: this initialized integer-only payload carries no references to
    // separate allocations; each source kernel only reads the copied value.
    unsafe {
        module.consume(stream, prepared, payload).unwrap();
        module.generic(stream, generic, payload).unwrap();
        #[cfg(feature = "async")]
        {
            let _ = module.consume_async(prepared, payload);
            let _ = module.consume_async_owned(prepared, payload);
            let _ = module.generic_async(generic, payload);
            let _ = module.generic_async_owned(generic, payload);
        }
    }
    module.ordinary(stream, ordinary, payload).unwrap();
    #[cfg(feature = "async")]
    {
        let _ = module.ordinary_async(ordinary, payload);
        let _ = module.ordinary_async_owned(ordinary, payload);
    }
    // The ordinary generic Rust helper still borrows caller-owned storage.
    kernels::generic(&payload);
}

fn main() {}
