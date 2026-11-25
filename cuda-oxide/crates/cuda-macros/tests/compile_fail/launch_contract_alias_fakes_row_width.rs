// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The inverse of hiding a row width: an alias whose last segment SPELLS a
//! runtime-width index space while resolving to a flat one. The spelling
//! selects the three-word `(ptr, len, width)` host ABI, but the device slice
//! is two words, so the launch would push one parameter too many and host and
//! device would disagree about the packet. The sealed
//! `__LaunchContractDisjointSliceAbi<_, true>` bound rejects the resolved
//! type, which carries no runtime row width.

use cuda_device::thread::Index2D;
use cuda_device::{DisjointSlice, cuda_module, kernel};

type Runtime2DIndex = Index2D<64>;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn alias_fakes_row_width(mut out: DisjointSlice<f32, Runtime2DIndex>) {
        let _ = &mut out;
    }
}

fn launch(
    module: &kernels::LoadedModule,
    stream: &cuda_core::CudaStream,
    out: &mut cuda_core::DeviceBuffer<f32>,
) {
    // The spelling chose the three-word (ptr, len, width) marshalling, but
    // the resolved device slice is two words. The ABI bound must reject the
    // call before a mis-shaped packet can exist.
    let cfg = cuda_core::LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };
    let _ = unsafe { module.alias_fakes_row_width(stream, cfg, cuda_host::RowWidth::new(out, 64)) };
}

fn main() {}
