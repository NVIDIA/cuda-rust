// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A generic kernel's launch bounds may name its type parameters, and host
//! code generic over the same parameter needs only the parameter's own trait
//! bound to prepare a launch.

use cuda_core::{BlockRequirement, KernelLaunchContract, LaunchConfig1D, LaunchContractError};
use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};

trait Tile {
    const MAX_THREADS: u32;
    const MIN_BLOCKS: u32;
}

enum Small {}

impl Tile for Small {
    const MAX_THREADS: u32 = 64;
    const MIN_BLOCKS: u32 = 2;
}

enum Wide {}

impl Tile for Wide {
    const MAX_THREADS: u32 = 256;
    const MIN_BLOCKS: u32 = 1;
}

#[kernel]
#[launch_bounds(F::MAX_THREADS, F::MIN_BLOCKS)]
fn standalone<F: Tile>() {}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(F::MAX_THREADS * 2)]
    #[launch_contract(domain = 1)]
    pub fn tiled<F: Tile>() {}
}

fn max_threads<F: Tile>() -> u32 {
    match <kernels::__tiled_CudaKernel<F> as KernelLaunchContract>::SPEC.block() {
        BlockRequirement::MaxThreads(max) => max,
        BlockRequirement::Exact(_) => panic!("unexpected exact block"),
    }
}

/// Generic host code: `F: Tile` is the only bound it needs.
fn prepare<F: Tile>(
    module: &kernels::LoadedModule,
) -> Result<cuda_core::PreparedLaunch<kernels::__tiled_CudaKernel<F>>, LaunchContractError> {
    module.prepare_tiled::<F>(LaunchConfig1D::new(1, F::MAX_THREADS, 0))
}

fn main() {
    assert_eq!(max_threads::<Small>(), 128);
    assert_eq!(max_threads::<Wide>(), 512);
    let _ = prepare::<Small>;
    let _ = prepare::<Wide>;
    let _ = standalone::<Small>;
}
