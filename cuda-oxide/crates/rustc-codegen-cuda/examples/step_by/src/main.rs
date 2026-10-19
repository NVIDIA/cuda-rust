/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression test for issue #21.
//!
//! `for i in (a..b).step_by(s)` lowers (after `StepBy::next` is inlined) to
//! a Transmute involving the one-word, niche-optimised representation of
//! `Option<NonZeroUsize>`. The historical synthetic-tag model turned the
//! destination into an unrelated aggregate, and the build crashed in llc
//! with
//!
//!   error: invalid cast opcode for cast from 'i64' to '{ i8, { { i64 } } }'
//!     %v23 = bitcast i64 %v22 to { i8, { { i64 } } }
//!
//! cuda-oxide now records rustc's physical niche carrier directly and lowers
//! the equal-size Transmute without assigning synthetic enum fields, so the
//! resulting PTX keeps the original one-word representation.
//!
//! The kernel below uses `step_by`; the second kernel is the `while`-loop
//! form from the original report and acts as a value-correctness control.
//!
//! Run: cargo oxide run step_by

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// Per-thread sum of `i` over `(tid..N).step_by(blockDim.x)`.
    #[kernel]
    pub fn step_by_sum(mut out: DisjointSlice<u64>) {
        let tid = thread::index_1d();
        let start = tid.get();
        if let Some(out_elem) = out.get_mut(tid) {
            let mut acc: u64 = 0;
            for i in (start..256usize).step_by(thread::blockDim_x() as usize) {
                acc += i as u64;
            }
            *out_elem = acc;
        }
    }

    /// `while`-loop equivalent. If `step_by_sum` and this disagree, the bug
    /// is in the `step_by` codegen (the case issue #21 was about).
    #[kernel]
    pub fn step_by_sum_control(mut out: DisjointSlice<u64>) {
        let tid = thread::index_1d();
        let start = tid.get();
        if let Some(out_elem) = out.get_mut(tid) {
            let step = thread::blockDim_x() as usize;
            let mut i = start;
            let mut acc: u64 = 0;
            while i < 256 {
                acc += i as u64;
                i += step;
            }
            *out_elem = acc;
        }
    }
}

fn expected(tid: usize, n: usize, step: usize) -> u64 {
    let mut i = tid;
    let mut acc: u64 = 0;
    while i < n {
        acc += i as u64;
        i += step;
    }
    acc
}

fn main() {
    println!("=== step_by regression (issue #21) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let stream = ctx.default_stream();

    const BLOCK: u32 = 32;
    const N: usize = BLOCK as usize;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut d_out = DeviceBuffer::<u64>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.step_by_sum(stream.as_ref(), cfg, &mut d_out) }.expect("launch step_by_sum");
    let got_step_by = d_out.to_host_vec(&stream).unwrap();

    let mut d_ctrl = DeviceBuffer::<u64>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.step_by_sum_control(stream.as_ref(), cfg, &mut d_ctrl) }
        .expect("launch step_by_sum_control");
    let got_ctrl = d_ctrl.to_host_vec(&stream).unwrap();

    let mut failures = 0usize;
    for tid in 0..N {
        let want = expected(tid, 256, BLOCK as usize);
        let s = got_step_by[tid];
        let c = got_ctrl[tid];
        if s != want || c != want || s != c {
            println!("FAIL tid={tid}: step_by={s} control={c} expected={want}");
            failures += 1;
        }
    }

    if failures == 0 {
        println!(
            "step_by: PASS ({} threads, step_by and while-loop produced the expected sums)",
            N
        );
    } else {
        println!("step_by: FAIL ({} mismatches)", failures);
        std::process::exit(1);
    }
}
