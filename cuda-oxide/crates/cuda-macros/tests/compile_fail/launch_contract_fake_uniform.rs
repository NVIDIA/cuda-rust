// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `#[cuda_module]` picks a parameter's host ABI from the spelling `Uniform<T>`,
//! so a local type of that name would otherwise be marshalled as a bare `T`
//! while presenting a different device layout. The sealed proof trait rejects
//! it before any launch is generated.

use cuda_device::{cuda_module, kernel, launch_contract};

#[repr(C)]
pub struct Uniform<T> {
    value: T,
    extra: u64,
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1))]
    pub fn lookalike(stride: Uniform<u32>) {
        let _ = stride.value;
    }
}

fn main() {}
