/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(static_mut_refs)]

//! Persistent AS3 debug-info fixture for cuda-gdb/T27.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::barrier::Barrier;
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn shared_debug(mut out: DisjointSlice<i32>) {
        static mut TILE: SharedArray<i32, 32> = SharedArray::UNINIT; // DEBUG_SHARED_TILE
        static mut OTHER: SharedArray<u16, 8> = SharedArray::UNINIT; // DEBUG_SHARED_OTHER
        static mut BAR: Barrier = Barrier::UNINIT; // DEBUG_SHARED_BARRIER

        let tid = thread::threadIdx_x() as usize;
        let gid = thread::index_1d();
        let block_base = (thread::blockIdx_x() * 100) as i32;
        let bar = &raw mut BAR;

        unsafe {
            TILE[tid] = block_base + tid as i32;
            OTHER[tid] = tid as u16;
        }
        thread::sync_threads();

        let tile_value = unsafe { TILE[tid] }; // DEBUG_SHARED_BREAK
        let other_guard = unsafe { OTHER[tid] as i32 } - tid as i32;
        let barrier_guard = if bar.addr() == 0 { -1000 } else { 0 };
        if let Some(slot) = out.get_mut(gid) {
            *slot = tile_value + other_guard + barrier_guard;
        }
    }

    /// Two block-local declarations deliberately have the same stable source
    /// path. Their mangled static instances and physical linkage names differ.
    #[kernel]
    pub fn same_leaf(flag: bool, mut out: DisjointSlice<i16>) {
        let tid = thread::threadIdx_x() as usize;
        if flag {
            static mut SAME: SharedArray<i16, 2> = SharedArray::UNINIT; // DEBUG_SHARED_SAME_LEFT
            unsafe {
                SAME[tid] = tid as i16;
                if let Some(slot) = out.get_mut(thread::index_1d()) {
                    *slot = SAME[tid];
                }
            }
        } else {
            static mut SAME: SharedArray<i16, 2> = SharedArray::UNINIT; // DEBUG_SHARED_SAME_RIGHT
            unsafe {
                SAME[tid] = 10 + tid as i16;
                if let Some(slot) = out.get_mut(thread::index_1d()) {
                    *slot = SAME[tid];
                }
            }
        }
    }

    /// Same source leaf as the T27 array, but a distinct function scope/type.
    #[kernel]
    pub fn other_scope(mut out: DisjointSlice<u64>) {
        static mut TILE: SharedArray<u64, 4> = SharedArray::UNINIT; // DEBUG_SHARED_OTHER_SCOPE
        let tid = thread::threadIdx_x() as usize;
        unsafe {
            TILE[tid] = tid as u64;
            if let Some(slot) = out.get_mut(thread::index_1d()) {
                *slot = TILE[tid];
            }
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("shared-debug module");
    let mut out = DeviceBuffer::<i32>::zeroed(&stream, 16).expect("output allocation");
    let config = LaunchConfig {
        grid_dim: (2, 1, 1),
        block_dim: (8, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { module.shared_debug((stream).as_ref(), config, &mut out) }.expect("kernel launch");
    stream.synchronize().expect("kernel completion");
    let values = out.to_host_vec(&stream).expect("copy output");
    let expected: Vec<_> = (0..2)
        .flat_map(|block| (0..8).map(move |thread| block * 100 + thread))
        .collect();
    assert_eq!(values, expected);
    println!("PASS shared_debug: runtime values = {values:?}");
}
