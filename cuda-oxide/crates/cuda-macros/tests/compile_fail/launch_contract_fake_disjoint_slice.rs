// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use core::marker::PhantomData;
use cuda_device::thread::Index1D;
use cuda_device::{cuda_module, kernel, launch_contract};

#[repr(C)]
pub struct DisjointSlice<'a, T, IndexSpace = Index1D> {
    ptr: *mut T,
    len: usize,
    marker: PhantomData<(&'a mut T, IndexSpace)>,
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1))]
    pub fn lookalike(mut out: DisjointSlice<u32>) {
        let _ = &mut out;
    }
}

fn prepare(module: &kernels::LoadedModule) {
    let _ = module.prepare_lookalike(cuda_core::LaunchConfig1D::new(1, 64, 0));
}

fn main() {}
