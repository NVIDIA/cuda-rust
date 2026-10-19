/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end check that `core::intrinsics::exact_div` works in device code.
//!
//! Before that intrinsic existed, `slice::as_chunks` failed to translate:
//!
//! ```text
//! Translation failed: core::slice::as_chunks::<4>
//!   [core/src/slice/mod.rs:1345:32] Compilation error: invalid input program
//! ```
//!
//! Line 1345 is `exact_div(self.len(), N)`. This exercises the intrinsic both
//! directly (unsafe calls on unsigned and signed dividends) and through
//! `as_chunks`, the API it was blocking.
//!
//! Run: `cargo oxide run exact_div_chunks --arch sm_86`

#![feature(core_intrinsics)]
#![allow(internal_features)]

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

const N: usize = 256;
const CHUNK: usize = 4;

#[cuda_module]
mod kernels {
    use super::*;

    /// Sums each thread's chunk through `as_chunks`, the safe API `exact_div`
    /// unblocks.
    ///
    /// The weights are distinct so a wrong chunk boundary, or a permutation
    /// inside a chunk, produces a wrong value instead of passing.
    #[kernel]
    pub fn chunk_sum(input: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        // `as_chunks` computes its chunk count with exact_div(len, CHUNK).
        // This is the call that used to fail to compile.
        let (chunks, _rest) = input.as_chunks::<CHUNK>();
        if let Some(slot) = out.get_mut(idx) {
            if i < chunks.len() {
                let c = chunks[i];
                *slot = c[0] + 2.0 * c[1] + 3.0 * c[2] + 4.0 * c[3];
            } else {
                *slot = -1.0;
            }
        }
    }

    /// Calls `core::intrinsics::exact_div` directly, away from `as_chunks`.
    ///
    /// The lowering picks `udiv` or `sdiv` from the operand's signedness, so
    /// this covers both arms: an unsigned `u64` division and a signed `i64`
    /// division with a negative dividend. Every dividend is a nonzero exact
    /// multiple of its divisor; anything else is undefined behaviour under
    /// the intrinsic's contract.
    #[kernel]
    pub fn exact_div_direct(mut out_u: DisjointSlice<u64>, mut out_s: DisjointSlice<i64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        // udiv arm: ((i + 1) * 256) / 4 == (i + 1) * 64.
        let n = (i as u64 + 1) * 256;
        // sdiv arm, negative dividend: -((i + 1) * 128) / 4 == -((i + 1) * 32).
        let s = -((i as i64 + 1) * 128);
        // SAFETY: both divisors are nonzero and divide their dividends exactly.
        let q_u = unsafe { core::intrinsics::exact_div(n, 4) };
        let q_s = unsafe { core::intrinsics::exact_div(s, 4) };
        if let Some(slot) = out_u.get_mut(idx) {
            *slot = q_u;
        }
        // `get_mut` consumes its ThreadIndex witness, so the second slice
        // needs a fresh one (same pattern as the array_constants example).
        let idx_s = thread::index_1d();
        if let Some(slot) = out_s.get_mut(idx_s) {
            *slot = q_s;
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load embedded module");
    let cfg = LaunchConfig::for_num_elems(N as u32);

    // ---- as_chunks path ----
    let input: Vec<f32> = (0..N * CHUNK).map(|i| (i as f32) * 0.5).collect();
    let in_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape matches the kernel; buffers cover its accesses.
    unsafe { module.chunk_sum(&stream, cfg, &in_dev, &mut out_dev) }.expect("chunk_sum launch");
    let got = out_dev.to_host_vec(&stream).unwrap();

    let mut bad = 0;
    for (i, &g) in got.iter().enumerate().take(N) {
        let c = &input[i * CHUNK..i * CHUNK + CHUNK];
        let want = c[0] + 2.0 * c[1] + 3.0 * c[2] + 4.0 * c[3];
        if (g - want).abs() > 1e-3 {
            if bad < 5 {
                println!("  chunk_sum mismatch at {i}: got {g} want {want}");
            }
            bad += 1;
        }
    }
    println!("as_chunks::<4>  : {} / {N} correct", N - bad);

    // ---- direct intrinsic path ----
    let mut u_dev = DeviceBuffer::<u64>::zeroed(&stream, N).unwrap();
    let mut s_dev = DeviceBuffer::<i64>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape matches the kernel; buffers cover its accesses.
    unsafe { module.exact_div_direct(&stream, cfg, &mut u_dev, &mut s_dev) }
        .expect("exact_div_direct launch");
    let ugot = u_dev.to_host_vec(&stream).unwrap();
    let sgot = s_dev.to_host_vec(&stream).unwrap();

    let mut dbad = 0;
    for i in 0..N {
        let want_u = (i as u64 + 1) * 64;
        let want_s = -((i as i64 + 1) * 32);
        if ugot[i] != want_u || sgot[i] != want_s {
            if dbad < 5 {
                println!(
                    "  exact_div mismatch at {i}: got ({}, {}) want ({want_u}, {want_s})",
                    ugot[i], sgot[i]
                );
            }
            dbad += 1;
        }
    }
    println!("exact_div direct: {} / {N} correct", N - dbad);

    if bad == 0 && dbad == 0 {
        println!("\nPASS");
    } else {
        println!("\nFAIL");
        std::process::exit(1);
    }
}
