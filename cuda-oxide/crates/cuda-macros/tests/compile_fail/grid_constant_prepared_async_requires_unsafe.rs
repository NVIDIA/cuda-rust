// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]

use core::cell::Cell;
use cuda_device::{cuda_module, kernel, launch_contract};

#[derive(Clone, Copy)]
pub struct Payload {
    read: &'static u32,
    write: &'static Cell<u32>,
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_contract(domain = 1, block = (1, 1, 1))]
    pub fn consume(#[grid_constant] payload: &Payload) {
        payload.write.set(*payload.read);
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (1, 1, 1))]
    pub fn generic<T: Copy>(#[grid_constant] payload: &T) {
        let _ = *payload;
    }
}

fn launch(
    module: &kernels::LoadedModule,
    prepared: &cuda_core::PreparedLaunch<kernels::__consume_CudaKernel>,
    generic: &cuda_core::PreparedLaunch<kernels::__generic_CudaKernel<Payload>>,
    payload: Payload,
) {
    // Owning the copied payload does not own either referenced allocation.
    let _ = module.consume_async(prepared, payload);
    let _ = module.consume_async_owned(prepared, payload);
    let _ = module.generic_async(generic, payload);
    let _ = module.generic_async_owned(generic, payload);
}

fn main() {}
