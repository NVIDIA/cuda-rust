// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// `no_data_race` on a `SharedArray` static requires `uniform_barriers` in `#[verify(...)]`.

#[cuda_macros::cuda_module]
mod kernels {
    #[cuda_macros::launch_contract(domain = 1, block = (64, 1, 1))]
    #[cuda_macros::verify(no_oob_idx(TILE), no_data_race(TILE))]
    #[cuda_macros::kernel]
    pub fn missing_uniform_barriers(mut out: cuda_device::DisjointSlice<u32>) {
        static mut TILE: cuda_device::SharedArray<u32, 64> = cuda_device::SharedArray::UNINIT;
        let _ = (&raw const TILE, &mut out);
    }
}

fn main() {}
