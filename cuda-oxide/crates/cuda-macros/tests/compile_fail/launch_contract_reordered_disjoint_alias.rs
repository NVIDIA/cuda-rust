// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_device::thread::Index1D;
use cuda_device::{cuda_module, kernel, launch_contract};

// The element is deliberately the second type argument. Host marshalling must
// not mistake `IS` for the element type and expose a safe mismatched launch.
type DisjointSlice<'a, IS, T> = cuda_device::DisjointSlice<'a, T, IS>;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1))]
    pub fn reordered(mut out: DisjointSlice<Index1D, u64>) {
        let _ = &mut out;
    }
}

fn prepare(module: &kernels::LoadedModule) {
    let _ = module.prepare_reordered(cuda_core::LaunchConfig1D::new(1, 64, 0));
}

fn main() {}
