// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `#[cuda_module]` picks the launch packet's shape (two words or three) from
//! the index space's spelling. An alias can hide a runtime row width behind
//! an innocent name: `Rt` below is `RuntimeRowMajorTiles<1, 1>`, whose slice
//! is `(ptr, len, width)` on the device, but the spelling selects the two-word
//! host ABI. Pushing two kernel parameters for a three-parameter kernel makes
//! the driver read past the argument array. The sealed
//! `__LaunchContractDisjointSliceAbi` bound is the semantic authority that
//! rejects the mismatch before any launch code exists.

use cuda_device::{DisjointSlice, RuntimeRowMajorTiles, cuda_module, kernel};

type Rt = RuntimeRowMajorTiles<1, 1>;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn alias_hides_row_width(mut out: DisjointSlice<f32, Rt>) {
        let _ = &mut out;
    }
}

fn launch(
    module: &kernels::LoadedModule,
    stream: &cuda_core::CudaStream,
    out: &mut cuda_core::DeviceBuffer<f32>,
) {
    // The spelling chose the two-word (ptr, len) marshalling, so this raw
    // launch would push two kernel parameters for a three-parameter kernel.
    // The ABI bound must reject the call.
    let cfg = cuda_core::LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };
    let _ = unsafe { module.alias_hides_row_width(stream, cfg, out) };
}

fn main() {}
