/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression coverage for overlap-safe device memory moves.
//!
//! Unlike `copy_nonoverlapping` (which reaches MIR as a `CopyNonOverlapping`
//! statement and lowers to `llvm.memcpy`), `core::ptr::copy` bottoms out in the
//! `core::intrinsics::copy` intrinsic. cuda-oxide lowers that path to the
//! overlap-safe `llvm.memmove`.
//!
//! The example also covers `[T]::copy_within`, whose libcore implementation
//! performs its checked range arithmetic and then reaches the same `ptr::copy`
//! path. Both forward and backward overlapping copies are verified.
//!
//! Usage:
//!   cargo oxide run ptr_copy
//!   CUDA_OXIDE_NO_OPT=1 cargo oxide run ptr_copy

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    /// Thread 0 loads `input` into `out`, then shifts `out[0..n-1]` up by one
    /// with `ptr::copy` (dst = src + 1, a forward-overlapping move that a plain
    /// memcpy would corrupt). Result: `out[0] == input[0]`, `out[k] == input[k-1]`.
    #[kernel]
    pub fn shift_right_one(input: &[i32], mut out: DisjointSlice<i32>, n: usize) {
        if thread::index_1d().get() == 0 {
            unsafe {
                let p = out.as_mut_ptr();
                core::ptr::copy_nonoverlapping(input.as_ptr(), p, n);
                core::ptr::copy(p, p.add(1), n - 1);
            }
        }
    }

    /// Exercise `slice::copy_within` with overlap in both relative directions.
    #[kernel]
    pub fn copy_within_overlap(seed: i32, mut out: DisjointSlice<i32>) {
        if thread::index_1d().get() != 0 {
            return;
        }

        let original = [
            seed,
            seed + 1,
            seed + 2,
            seed + 3,
            seed + 4,
            seed + 5,
            seed + 6,
            seed + 7,
        ];

        let mut forward = original;
        let mut backward = original;

        // Destination begins inside the source range: dst > src.
        forward.copy_within(0..6, 2);

        // Source begins inside the destination range: dst < src.
        backward.copy_within(2..8, 0);

        unsafe {
            // One active thread owns the complete output region. Keep result
            // export explicit so this regression does not depend on iterator
            // lowering in addition to `copy_within`.
            let ptr = out.as_mut_ptr();

            ptr.write(forward[0]);
            ptr.add(1).write(forward[1]);
            ptr.add(2).write(forward[2]);
            ptr.add(3).write(forward[3]);
            ptr.add(4).write(forward[4]);
            ptr.add(5).write(forward[5]);
            ptr.add(6).write(forward[6]);
            ptr.add(7).write(forward[7]);

            ptr.add(8).write(backward[0]);
            ptr.add(9).write(backward[1]);
            ptr.add(10).write(backward[2]);
            ptr.add(11).write(backward[3]);
            ptr.add(12).write(backward[4]);
            ptr.add(13).write(backward[5]);
            ptr.add(14).write(backward[6]);
            ptr.add(15).write(backward[7]);
        }
    }
}

fn main() {
    println!("=== ptr_copy ===");

    const N: usize = 96;
    let input: Vec<i32> = (0..N as i32).map(|i| i * 3 - 7).collect();

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let cfg = LaunchConfig::for_num_elems(N as u32);

    let din = DeviceBuffer::from_host(&stream, &input).unwrap();
    let mut out = DeviceBuffer::<i32>::zeroed(&stream, N).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.shift_right_one(&stream, cfg, &din, &mut out, N) }
        .expect("shift_right_one launch");

    let got = out.to_host_vec(&stream).unwrap();

    let mut want = input.clone();
    for k in (1..N).rev() {
        want[k] = want[k - 1];
    }

    assert_eq!(got, want, "shift_right_one (overlapping ptr::copy)");
    println!("PASS: ptr::copy (overlapping forward move via memmove)");

    const COPY_WITHIN_SEED: i32 = 100;
    const COPY_WITHIN_WORDS: usize = 16;

    let mut copy_within_out = DeviceBuffer::<i32>::zeroed(&stream, COPY_WITHIN_WORDS).unwrap();

    // SAFETY: one thread is sufficient and the output buffer has all 16 values.
    unsafe {
        module.copy_within_overlap(
            &stream,
            LaunchConfig::for_num_elems(1),
            COPY_WITHIN_SEED,
            &mut copy_within_out,
        )
    }
    .expect("copy_within_overlap launch");

    let got = copy_within_out.to_host_vec(&stream).unwrap();
    let expected = vec![
        100, 101, 100, 101, 102, 103, 104, 105, // forward overlap
        102, 103, 104, 105, 106, 107, 106, 107, // backward overlap
    ];

    assert_eq!(got, expected, "slice::copy_within forward/backward overlap");
    println!("PASS: slice::copy_within (forward and backward overlap)");

    println!("PASS: ptr_copy");
}
