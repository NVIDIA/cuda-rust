/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Array literal initialization smoke test (issue #232).
//!
//! Verifies that array literals assigned to addressable locals compile and
//! produce correct results. Before the fix, `let arr = [v0, v1, v2, v3]`
//! inside a kernel would crash because the translator tried to build a full
//! SSA aggregate for an array type, which is not a first-class SSA value.
//!
//! The kernel initializes a `[u32; 4]` array literal, modifies one element,
//! then sums all elements. The host checks the expected sum.
//!
//! Run: cargo oxide run array_init

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// Initializes a fixed array literal and sums its elements.
    ///
    /// arr = [1, 2, 3, 4]  (literal initialization into an alloca slot)
    /// arr[2] += thread_index  (dynamic modification to keep the array live)
    /// out[i] = arr[0] + arr[1] + arr[2] + arr[3]
    #[kernel]
    pub fn array_sum(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();

        // Array literal: exercises Aggregate(Array, ...) into an alloca.
        let mut arr = [1u32, 2, 3, 4];

        // Prevent the array from being optimized away.
        arr[2] += idx.get() as u32;

        if let Some(o) = out.get_mut(idx) {
            *o = arr[0] + arr[1] + arr[2] + arr[3];
        }
    }

    /// Original repro from issue #232: 2D scratch buffer with a row assignment.
    ///
    /// scratch = [[0.0; 2]; 4]  (nested repeat aggregate)
    /// scratch[2] = [0.0, 42.0]  (row store: array aggregate into indexed slot)
    /// out[0] = scratch[2][1]   (read back)
    ///
    /// Before the fix, the IR contained an `insertvalue` chain building a full
    /// `[4 x [2 x double]]` SSA value before storing it; with the fix each
    /// element is stored directly into the alloca.
    #[kernel]
    pub fn scratch_2d(mut out: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        if idx.get() != 0 {
            return;
        }

        const ROWS: usize = 4;
        const COLS: usize = 2;
        let mut scratch = [[0.0_f64; COLS]; ROWS];
        scratch[2] = [0.0, 42.0];

        if let Some(slot) = out.get_mut(idx) {
            *slot = scratch[2][1];
        }
    }

    /// Same but with a repeat expression: [0u32; 8].
    #[kernel]
    #[allow(clippy::needless_range_loop)]
    pub fn zeroed_array_sum(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;

        let mut arr = [0u32; 8];
        for k in 0..8usize {
            arr[k] = i + k as u32;
        }

        let mut sum = 0u32;
        for k in 0..8usize {
            sum += arr[k];
        }

        if let Some(o) = out.get_mut(idx) {
            *o = sum;
        }
    }
}

fn main() {
    const N: usize = 256;

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load module");
    let cfg = LaunchConfig::for_num_elems(N as u32);

    // --- array_sum ---
    // arr = [1, 2, 3, 4]; arr[2] += i; sum = 1 + 2 + (3 + i) + 4 = 10 + i
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: this is a 1D launch and the output allocation has one element
    // for every launched thread.
    unsafe { module.array_sum(&stream, cfg, &mut out_dev) }.expect("array_sum launch");
    let out = out_dev.to_host_vec(&stream).unwrap();

    let mut errors = 0usize;
    for (i, &val) in out.iter().enumerate() {
        let expected = 10 + i as u32;
        if val != expected {
            if errors < 5 {
                eprintln!("  FAIL array_sum[{}]: got {} want {}", i, val, expected);
            }
            errors += 1;
        }
    }

    // --- zeroed_array_sum ---
    // arr[k] = i + k; sum = sum_{k=0}^{7}(i + k) = 8i + 28
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: this is a 1D launch and the output allocation has one element
    // for every launched thread.
    unsafe { module.zeroed_array_sum(&stream, cfg, &mut out_dev) }
        .expect("zeroed_array_sum launch");
    let out2 = out_dev.to_host_vec(&stream).unwrap();

    for (i, &val) in out2.iter().enumerate() {
        let expected = 8 * i as u32 + 28;
        if val != expected {
            if errors < 5 {
                eprintln!(
                    "  FAIL zeroed_array_sum[{}]: got {} want {}",
                    i, val, expected
                );
            }
            errors += 1;
        }
    }

    // --- scratch_2d ---
    // scratch[2] = [0.0, 42.0]; out[0] = scratch[2][1]  => 42.0
    let mut out_dev = DeviceBuffer::<f64>::zeroed(&stream, 1).unwrap();
    // SAFETY: `scratch_2d` is launched with one thread and writes one output.
    unsafe { module.scratch_2d(&stream, LaunchConfig::for_num_elems(1), &mut out_dev) }
        .expect("scratch_2d launch");
    let out3 = out_dev.to_host_vec(&stream).unwrap();
    if (out3[0] - 42.0_f64).abs() > 1e-12 {
        eprintln!("  FAIL scratch_2d: got {} want 42.0", out3[0]);
        errors += 1;
    }

    if errors == 0 {
        println!("SUCCESS: array literal initialization produces correct results");
    } else {
        eprintln!("FAIL: {} errors", errors);
        std::process::exit(1);
    }
}
