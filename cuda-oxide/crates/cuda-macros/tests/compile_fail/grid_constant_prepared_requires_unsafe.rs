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
    stream: &cuda_core::CudaStream,
    prepared: &cuda_core::PreparedLaunch<kernels::__consume_CudaKernel>,
    generic: &cuda_core::PreparedLaunch<kernels::__generic_CudaKernel<Payload>>,
    payload: Payload,
) {
    // A Copy payload and checked geometry cannot make its nested references
    // GPU-accessible or prove that the writes are synchronized.
    module.consume(stream, prepared, payload).unwrap();
    module.generic(stream, generic, payload).unwrap();
}

fn main() {}
