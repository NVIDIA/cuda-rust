// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]

use cuda_device::{cuda_module, kernel};

#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Descriptor([u32; 32]);

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn plain(#[grid_constant] descriptor: &Descriptor) {
        let _ = descriptor.0[0];
    }

    #[kernel]
    pub fn generic<T: Copy>(#[grid_constant] descriptor: &'_ Descriptor, value: T) {
        let _ = (descriptor.0[0], value);
    }
}

#[kernel(u32)]
pub fn explicit<T: Copy>(#[grid_constant] descriptor: &Descriptor, value: T) {
    let _ = (descriptor.0[0], value);
}

fn typed_launches(
    module: &kernels::LoadedModule,
    stream: &cuda_core::CudaStream,
    descriptor: Descriptor,
) {
    let config = cuda_core::simt::LaunchConfig::for_num_elems(1);
    // SAFETY: this function only type-checks the by-value host ABI; it is never run.
    unsafe {
        module.plain(stream, config, descriptor).unwrap();
        module.generic(stream, config, descriptor, 3u32).unwrap();
    }
}

fn main() {
    // The ordinary generic helper still borrows caller storage and has no
    // entry ABI marker, even when the generated entry uses grid_constant.
    kernels::generic(&Descriptor([1; 32]), 3u32);
    explicit(&Descriptor([1; 32]), 3u32);
}
