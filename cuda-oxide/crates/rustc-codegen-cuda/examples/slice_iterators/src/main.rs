/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Device-side conformance coverage for slice iterators carrying runtime
//! fat-slice state.
//!
//! `slice::windows` advances an overlapping slice view on every `next()`.
//! `slice::chunks_exact` advances a complete-chunk view while preserving an
//! independent remainder slice. `slice::chunks_exact_mut` carries mutable
//! slice state and yields non-overlapping mutable chunks before returning the
//! mutable remainder.
//!
//! This regression intentionally stays on forward iterator paths. It does not
//! exercise reverse/from-end iteration, dedicated MIR `Subslice` projection
//! regressions, local-array iterator scalarization, or `as_chunks`.
//!
//! Usage:
//!   cargo oxide run slice_iterators
//!   CUDA_OXIDE_NO_OPT=1 cargo oxide run slice_iterators

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// Encode every 3-element overlapping window and the number of windows.
    #[allow(clippy::while_let_on_iterator)] // Explicit next() calls are the behavior under test.
    #[kernel]
    pub fn windows_forward(input: &[u32], window_size: usize, mut out: DisjointSlice<u32>) {
        if thread::index_1d().get() != 0 {
            return;
        }

        let mut windows = input.windows(window_size);
        let mut count = 0usize;

        unsafe {
            // One active thread owns the complete output region.
            let ptr = out.as_mut_ptr();

            while let Some(window) = windows.next() {
                if count < 5 {
                    let signature = if window.len() == 3 {
                        window[0] * 10_000 + window[1] * 100 + window[2]
                    } else {
                        u32::MAX
                    };
                    ptr.add(count).write(signature);
                }
                count += 1;
            }

            ptr.add(5).write(count as u32);
        }
    }

    /// Encode complete chunks plus the remainder before and after iteration.
    #[allow(clippy::while_let_on_iterator)] // Explicit next() calls are the behavior under test.
    #[kernel]
    pub fn chunks_exact_forward(input: &[u32], chunk_size: usize, mut out: DisjointSlice<u32>) {
        if thread::index_1d().get() != 0 {
            return;
        }

        let mut chunks = input.chunks_exact(chunk_size);
        let remainder_before = chunks.remainder();
        let mut count = 0usize;

        unsafe {
            // One active thread owns the complete output region.
            let ptr = out.as_mut_ptr();

            while let Some(chunk) = chunks.next() {
                if count < 2 {
                    let signature = if chunk.len() == 3 {
                        chunk[0] * 10_000 + chunk[1] * 100 + chunk[2]
                    } else {
                        u32::MAX
                    };
                    ptr.add(count).write(signature);
                }
                count += 1;
            }

            let remainder_after = chunks.remainder();

            ptr.add(2).write(count as u32);

            ptr.add(3).write(remainder_before.len() as u32);
            ptr.add(4).write(if remainder_before.len() == 2 {
                remainder_before[0]
            } else {
                u32::MAX
            });
            ptr.add(5).write(if remainder_before.len() == 2 {
                remainder_before[1]
            } else {
                u32::MAX
            });

            ptr.add(6).write(remainder_after.len() as u32);
            ptr.add(7).write(if remainder_after.len() == 2 {
                remainder_after[0]
            } else {
                u32::MAX
            });
            ptr.add(8).write(if remainder_after.len() == 2 {
                remainder_after[1]
            } else {
                u32::MAX
            });
        }
    }

    /// Mutate complete chunks with chunk-specific deltas, then mutate the
    /// remainder through `ChunksExactMut::into_remainder`.
    #[allow(clippy::while_let_on_iterator)] // Explicit next() calls are the behavior under test.
    #[kernel]
    pub fn chunks_exact_mut_forward(mut buffer: DisjointSlice<u32>, n: usize, chunk_size: usize) {
        if thread::index_1d().get() != 0 {
            return;
        }

        unsafe {
            // SAFETY: the host passes `n == buffer.len()`, the backing device
            // allocation remains live for the kernel, and only thread 0 enters
            // this path.
            let slice = core::slice::from_raw_parts_mut(buffer.as_mut_ptr(), n);
            let mut chunks = slice.chunks_exact_mut(chunk_size);
            let mut ordinal = 1u32;

            while let Some(chunk) = chunks.next() {
                let delta = ordinal * 10;
                let mut i = 0usize;
                while i < chunk.len() {
                    chunk[i] = chunk[i].wrapping_add(delta);
                    i += 1;
                }
                ordinal += 1;
            }

            let remainder = chunks.into_remainder();
            let mut i = 0usize;
            while i < remainder.len() {
                remainder[i] = remainder[i].wrapping_add(100);
                i += 1;
            }
        }
    }
}

fn main() {
    println!("=== slice_iterators ===");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let one_thread = LaunchConfig::for_num_elems(1);

    // --- slice::windows ---
    let windows_input = vec![10u32, 20, 30, 40, 50, 60, 70];
    let windows_dev = DeviceBuffer::from_host(&stream, &windows_input).unwrap();
    let mut windows_out = DeviceBuffer::<u32>::zeroed(&stream, 6).unwrap();

    // SAFETY: one thread owns the output and all buffers cover the kernel accesses.
    unsafe { module.windows_forward(&stream, one_thread, &windows_dev, 3, &mut windows_out) }
        .expect("windows_forward launch");

    let got = windows_out.to_host_vec(&stream).unwrap();
    let expected = vec![102_030, 203_040, 304_050, 405_060, 506_070, 5];
    assert_eq!(got, expected, "slice::windows forward iterator state");
    println!("PASS: slice::windows (overlapping forward views)");

    // --- slice::chunks_exact ---
    let chunks_input = vec![10u32, 20, 30, 40, 50, 60, 70, 80];
    let chunks_dev = DeviceBuffer::from_host(&stream, &chunks_input).unwrap();
    let mut chunks_out = DeviceBuffer::<u32>::zeroed(&stream, 9).unwrap();

    // SAFETY: one thread owns the output and all buffers cover the kernel accesses.
    unsafe { module.chunks_exact_forward(&stream, one_thread, &chunks_dev, 3, &mut chunks_out) }
        .expect("chunks_exact_forward launch");

    let got = chunks_out.to_host_vec(&stream).unwrap();
    let expected = vec![
        102_030, 405_060, // complete chunk signatures
        2,       // complete chunk count
        2, 70, 80, // remainder before iteration
        2, 70, 80, // remainder after iteration
    ];
    assert_eq!(
        got, expected,
        "slice::chunks_exact forward state and persistent remainder"
    );
    println!("PASS: slice::chunks_exact (forward chunks + stable remainder)");

    // --- slice::chunks_exact_mut ---
    let mutable_input = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
    let mut mutable_dev = DeviceBuffer::from_host(&stream, &mutable_input).unwrap();

    // SAFETY: one thread owns the complete buffer mutation and `n` matches its length.
    unsafe {
        module.chunks_exact_mut_forward(
            &stream,
            one_thread,
            &mut mutable_dev,
            mutable_input.len(),
            3,
        )
    }
    .expect("chunks_exact_mut_forward launch");

    let got = mutable_dev.to_host_vec(&stream).unwrap();
    let expected = vec![11u32, 12, 13, 24, 25, 26, 107, 108];
    assert_eq!(
        got, expected,
        "slice::chunks_exact_mut chunk and remainder mutation"
    );
    println!("PASS: slice::chunks_exact_mut (mutable chunks + remainder)");

    println!("PASS: slice_iterators");
}
