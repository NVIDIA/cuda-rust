/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `#[nounroll]` keeps a loop rolled through LLVM, libNVVM, and ptxas.
//!
//! ptxas fully unrolls small constant-trip-count loops on its own, even when the
//! PTX keeps them rolled. `#[nounroll]` lowers to `llvm.loop.unroll.disable`
//! metadata on the loop's latch, and the NVPTX backend turns that into
//! `.pragma "nounroll";` at the loop header, which ptxas honors.
//!
//! - `taps_rolled` keeps a constant 4-trip loop rolled. `taps_default` is the
//!   same loop without the annotation.
//! - `rolled_k_unrolled_inner` keeps an outer K loop rolled while `#[unroll]`
//!   expands the loop inside it.
//! - `rolled_inside_full_unroll` and `rolled_inside_partial_unroll` put a
//!   `#[nounroll]` range loop inside a loop that `#[unroll]` / `#[unroll(4)]`
//!   unrolls. Every copy of the inner loop, including the one in the partial
//!   unroll's remainder loop, stays rolled.
//! - `rolled_in_device_fn` calls a `#[device]` function whose `loop` carries
//!   the annotation.
//!
//! Every annotated loop loads from memory. With an arithmetic-only body and a
//! constant trip count, LLVM or ptxas computes the loop's result outright and
//! deletes the loop, which leaves nothing to keep rolled.
//!
//! `verify-code-shape.sh` counts the pragmas in each PTX entry. GPU-less CI runs
//! it through `scripts/smoketest.sh --compile-only`. The host checks every
//! result.
//!
//! Run: cargo oxide run nounroll

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, device, kernel, thread};
use cuda_host::cuda_module;

const TAPS: usize = 4;
const N: usize = 32;
/// Largest outer trip count the nested kernels may read inputs for.
const MAX_TRIP: u32 = 16;

#[cuda_module]
mod kernels {
    use super::*;

    /// `out[i] = sum(w[t] * x[i + t])` over `TAPS` taps, with the tap loop kept
    /// rolled.
    #[kernel]
    pub fn taps_rolled(w: &[f32], x: &[f32], mut out: DisjointSlice<f32>) {
        let tid = thread::index_1d();
        let i = tid.get();
        if let Some(slot) = out.get_mut(tid) {
            let mut acc = 0.0f32;
            let mut tap = 0usize;
            #[nounroll]
            while tap < TAPS {
                acc += w[tap] * x[i + tap];
                tap += 1;
            }
            *slot = acc;
        }
    }

    /// The same loop as `taps_rolled`, left to the default unrolling.
    #[kernel]
    pub fn taps_default(w: &[f32], x: &[f32], mut out: DisjointSlice<f32>) {
        let tid = thread::index_1d();
        let i = tid.get();
        if let Some(slot) = out.get_mut(tid) {
            let mut acc = 0.0f32;
            let mut tap = 0usize;
            while tap < TAPS {
                acc += w[tap] * x[i + tap];
                tap += 1;
            }
            *slot = acc;
        }
    }

    /// An outer K loop of 4 trips stays rolled while `#[unroll]` expands the
    /// 4-lane loop inside it. `out[i] = sum(a[k * 4 + n] * (i + 1))` over every
    /// `k < 8` step 2 and `n < 4`.
    #[kernel]
    pub fn rolled_k_unrolled_inner(a: &[u32], mut out: DisjointSlice<u32>) {
        let tid = thread::index_1d();
        let scale = tid.get() as u32 + 1;
        if let Some(slot) = out.get_mut(tid) {
            let mut acc = [0u32; 4];
            let mut k = 0usize;
            #[nounroll]
            while k < 8 {
                let mut n = 0usize;
                #[unroll]
                while n < 4 {
                    acc[n] = acc[n].wrapping_add(a[k * 4 + n].wrapping_mul(scale));
                    n += 1;
                }
                k += 2;
            }
            *slot = acc[0]
                .wrapping_add(acc[1])
                .wrapping_add(acc[2])
                .wrapping_add(acc[3]);
        }
    }

    /// Full `#[unroll]` copies the inner `#[nounroll]` loop into each of its
    /// four body copies. `out[i] = i + sum(a[i_outer * 3 + j])` over
    /// `i_outer < 4` and `j < 3`.
    ///
    /// The inner body loads from memory: LLVM would otherwise compute the sum
    /// in closed form and delete the loop, leaving nothing to keep rolled. The
    /// unchecked load and wrapping index keep panic exits out of the loop.
    #[kernel]
    pub fn rolled_inside_full_unroll(a: &[u32], mut out: DisjointSlice<u32>) {
        let tid = thread::index_1d();
        let base = tid.get() as u32;
        if let Some(slot) = out.get_mut(tid) {
            let mut acc = base;
            let mut i = 0u32;
            #[unroll]
            while i < 4 {
                #[nounroll]
                for j in 0..3u32 {
                    let index = i.wrapping_mul(3).wrapping_add(j) as usize;
                    // SAFETY: the host passes `3 * MAX_TRIP` elements, and
                    // `i < 4 <= MAX_TRIP`
                    acc = acc.wrapping_add(unsafe { *a.get_unchecked(index) });
                }
                i += 1;
            }
            *slot = acc;
        }
    }

    /// Partial `#[unroll(4)]` leaves four inner copies in its main loop and one
    /// in its remainder loop, and marks the outer main and remainder latches
    /// with `llvm.loop.unroll.disable` so LLVM does not unroll them again (7
    /// marked latches total). `out[i] = i + sum(a[i_outer * 3 + j])` over
    /// `i_outer < trip` and `j < 3`.
    ///
    /// Partial unrolling needs the loop test to be the only exit, so the body
    /// avoids bounds checks and overflow checks the same way. The inner loop
    /// is a `while`: a range `for` adds an unreachable `Option` arm, which
    /// partial unrolling counts as a second exit.
    #[kernel]
    pub fn rolled_inside_partial_unroll(a: &[u32], mut out: DisjointSlice<u32>, trip: u32) {
        let tid = thread::index_1d();
        let base = tid.get() as u32;
        if let Some(slot) = out.get_mut(tid) {
            let mut acc = base;
            let mut i = 0u32;
            #[unroll(4)]
            while i < trip {
                let mut j = 0u32;
                #[nounroll]
                while j < 3 {
                    let index = i.wrapping_mul(3).wrapping_add(j) as usize;
                    // SAFETY: the host passes `3 * MAX_TRIP` elements and
                    // `trip <= MAX_TRIP`
                    acc = acc.wrapping_add(unsafe { *a.get_unchecked(index) });
                    j = j.wrapping_add(1);
                }
                i = i.wrapping_add(1);
            }
            *slot = acc;
        }
    }

    /// `x + a[0] + a[1] + a[2] + a[3]`, summed by a rolled `loop`, so the
    /// annotation is covered on `loop` and in a `#[device]` function.
    #[inline(never)]
    #[device]
    fn rolled_sum(a: &[u32], x: u32) -> u32 {
        let mut acc = x;
        let mut step = 0usize;
        #[nounroll]
        loop {
            // SAFETY: the host passes at least four elements
            acc = acc.wrapping_add(unsafe { *a.get_unchecked(step) });
            step = step.wrapping_add(1);
            if step == 4 {
                break;
            }
        }
        acc
    }

    /// `out[i] = i + a[0] + a[1] + a[2] + a[3]`, computed by `rolled_sum`.
    #[kernel]
    pub fn rolled_in_device_fn(a: &[u32], mut out: DisjointSlice<u32>) {
        let tid = thread::index_1d();
        let base = tid.get() as u32;
        if let Some(slot) = out.get_mut(tid) {
            *slot = rolled_sum(a, base);
        }
    }
}

fn taps_expected(w: &[f32], x: &[f32]) -> Vec<f32> {
    (0..N)
        .map(|i| (0..TAPS).fold(0.0, |acc, tap| acc + w[tap] * x[i + tap]))
        .collect()
}

fn nested_expected(base: u32, outer_trips: u32, a: &[u32]) -> u32 {
    (0..outer_trips * 3).fold(base, |acc, index| acc.wrapping_add(a[index as usize]))
}

fn main() {
    println!("=== nounroll ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let stream = ctx.default_stream();
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (N as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut failures = 0usize;

    let w: Vec<f32> = (0..TAPS).map(|t| 0.5 + t as f32).collect();
    let x: Vec<f32> = (0..N + TAPS).map(|v| (v % 7) as f32 - 3.0).collect();
    let w_dev = DeviceBuffer::from_host(&stream, &w).unwrap();
    let x_dev = DeviceBuffer::from_host(&stream, &x).unwrap();
    let want_taps = taps_expected(&w, &x);

    let mut d_rolled = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    // SAFETY: each thread reads `x[i..i + TAPS]`, which `x` covers, and writes
    // its own output element.
    unsafe { module.taps_rolled(stream.as_ref(), cfg, &w_dev, &x_dev, &mut d_rolled) }
        .expect("launch taps_rolled");
    let mut d_default = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    // SAFETY: as for `taps_rolled`.
    unsafe { module.taps_default(stream.as_ref(), cfg, &w_dev, &x_dev, &mut d_default) }
        .expect("launch taps_default");
    // small integers times halves: every partial sum is exact in f32
    for (name, got) in [
        ("taps_rolled", d_rolled.to_host_vec(&stream).unwrap()),
        ("taps_default", d_default.to_host_vec(&stream).unwrap()),
    ] {
        if got != want_taps {
            println!("FAIL {name}: {got:?} expected {want_taps:?}");
            failures += 1;
        }
    }

    let a: Vec<u32> = (0..32).map(|v| v * 3 + 1).collect();
    let a_dev = DeviceBuffer::from_host(&stream, &a).unwrap();
    let mut d_k = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: every thread reads `a[0..32]` and writes its own output element.
    unsafe { module.rolled_k_unrolled_inner(stream.as_ref(), cfg, &a_dev, &mut d_k) }
        .expect("launch rolled_k_unrolled_inner");
    let a_sum: u32 = (0..8)
        .step_by(2)
        .flat_map(|k| (0..4).map(move |n| k * 4 + n))
        .map(|index| a[index])
        .sum();
    let want_k: Vec<u32> = (0..N as u32).map(|i| a_sum * (i + 1)).collect();
    if d_k.to_host_vec(&stream).unwrap() != want_k {
        println!("FAIL rolled_k_unrolled_inner");
        failures += 1;
    }

    let nested: Vec<u32> = (0..3 * MAX_TRIP).map(|v| v * 7 + 2).collect();
    let nested_dev = DeviceBuffer::from_host(&stream, &nested).unwrap();
    let mut d_full = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: the kernel reads `nested[0..12]` and each thread writes its own
    // output element.
    unsafe { module.rolled_inside_full_unroll(stream.as_ref(), cfg, &nested_dev, &mut d_full) }
        .expect("launch rolled_inside_full_unroll");
    let want_full: Vec<u32> = (0..N as u32)
        .map(|i| nested_expected(i, 4, &nested))
        .collect();
    if d_full.to_host_vec(&stream).unwrap() != want_full {
        println!("FAIL rolled_inside_full_unroll");
        failures += 1;
    }

    // 10 = two trips of the main loop plus a remainder of two
    let trip = 10u32;
    let mut d_partial = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    assert!(trip <= MAX_TRIP);
    // SAFETY: the kernel reads `nested[0..3 * trip]`, which `nested` covers,
    // and each thread writes its own output element.
    unsafe {
        module.rolled_inside_partial_unroll(stream.as_ref(), cfg, &nested_dev, &mut d_partial, trip)
    }
    .expect("launch rolled_inside_partial_unroll");
    let want_partial: Vec<u32> = (0..N as u32)
        .map(|i| nested_expected(i, trip, &nested))
        .collect();
    if d_partial.to_host_vec(&stream).unwrap() != want_partial {
        println!("FAIL rolled_inside_partial_unroll");
        failures += 1;
    }

    let mut d_device = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: the kernel reads `nested[0..4]` and each thread writes its own
    // output element.
    unsafe { module.rolled_in_device_fn(stream.as_ref(), cfg, &nested_dev, &mut d_device) }
        .expect("launch rolled_in_device_fn");
    let want_device: Vec<u32> = (0..N as u32)
        .map(|i| nested[..4].iter().fold(i, |acc, &v| acc.wrapping_add(v)))
        .collect();
    if d_device.to_host_vec(&stream).unwrap() != want_device {
        println!("FAIL rolled_in_device_fn");
        failures += 1;
    }

    if failures == 0 {
        println!("nounroll: PASS ({N} threads; every rolled loop computes the same results)");
    } else {
        println!("nounroll: FAIL ({failures} kernels mismatched)");
        std::process::exit(1);
    }
}
