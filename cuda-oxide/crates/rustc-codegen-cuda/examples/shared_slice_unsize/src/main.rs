/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared-memory array unsized to a slice (`&mut [T; N]` -> `&mut [T]`).
//!
//! A `SharedArray` lives in addrspace(3). Viewing it as `&mut [f32; N]` and
//! passing it to a helper that takes `&mut [f32]` is an ordinary Rust unsize
//! coercion, but the canonical slice fat pointer stores a *generic*
//! (addrspace 0) data pointer in field 0 — so the lowering must
//! `addrspacecast` the shared pointer before the `insert_value`
//! (PTX: `cvta.shared`). Without that coercion the lowered module fails
//! verification with "Value being inserted / extracted does not match the
//! type of the indexed aggregate".
//!
//! The kernel runs a single block that stages values into shared memory,
//! then reduces them *through the slice view*, so a regression is either a
//! hard build failure (the verifier error above) or a wrong sum.
//!
//! Run: `cargo oxide run shared_slice_unsize`

#![allow(static_mut_refs)]

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, SharedArray, device, kernel, thread};
use cuda_host::cuda_module;

const N: usize = 128;

#[cuda_module]
mod kernels {
    use super::*;

    /// Sum a slice. Taking `&[f32]` (not `&[f32; N]`) is the point: the
    /// caller's shared-memory array reference must unsize into a fat
    /// pointer whose data-pointer slot is generic.
    #[inline(never)]
    #[device]
    fn sum_slice(values: &[f32]) -> f32 {
        let mut acc = 0.0;
        let mut i = 0;
        while i < values.len() {
            acc += values[i];
            i += 1;
        }
        acc
    }

    #[kernel]
    pub fn shared_slice_sum(input: &[f32], mut out: DisjointSlice<f32>) {
        // A shared 2D tile whose *rows* are arrays: indexing yields a real
        // `&mut [f32; N]` in addrspace(3) (a raw-pointer round-trip through
        // `as_mut_ptr` would erase the address space and miss the bug).
        static mut TILE: SharedArray<[f32; N], 1> = SharedArray::UNINIT;

        let tid = thread::index_1d();
        let i = tid.get();

        // Stage one element per thread into shared memory.
        if i < N {
            unsafe { TILE[0][i] = input[i] };
        }
        thread::sync_threads();

        if i == 0 {
            // `&TILE[0]` is `&[f32; N]` pointing into shared memory; the
            // call unsizes it to `&[f32]` — the repro shape.
            let row: &[f32; N] = unsafe { &TILE[0] };
            let total = sum_slice(row);
            if let Some(slot) = out.get_mut(tid) {
                *slot = total;
            }
        }
    }
}

fn main() {
    println!("=== shared_slice_unsize (shared array -> slice fat pointer) ===");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let input: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let expected: f32 = input.iter().sum();

    let input_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, 1).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let cfg = LaunchConfig::for_num_elems(N as u32);
    // SAFETY: one block of >= N threads; the kernel guards all accesses.
    unsafe { module.shared_slice_sum(&stream, cfg, &input_dev, &mut out_dev) }
        .expect("Kernel launch failed");

    let result = out_dev.to_host_vec(&stream).unwrap()[0];
    if (result - expected).abs() < 1e-3 {
        println!("PASS shared_slice_unsize: sum={result} (expected {expected})");
    } else {
        eprintln!("FAIL shared_slice_unsize: sum={result}, expected {expected}");
        std::process::exit(1);
    }
}
