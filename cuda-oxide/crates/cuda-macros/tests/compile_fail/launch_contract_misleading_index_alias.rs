// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_device::thread::Index1D as Index2D;
use cuda_device::{DisjointSlice, cuda_module, kernel, launch_contract};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_contract(domain = 2, block = (8, 8, 1))]
    pub fn misleading_name(mut out: DisjointSlice<u32, Index2D>) {
        let _ = &mut out;
    }
}

fn prepare(module: &kernels::LoadedModule) {
    let _ = module.prepare_misleading_name(cuda_core::LaunchConfig2D::new(
        (1, 1),
        (8, 8),
        0,
    ));
}

fn main() {}
