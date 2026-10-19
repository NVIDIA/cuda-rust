/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression test for a niche-`Option` whose payload carries two pointers —
//! `<[T]>::split_at_mut_checked`, which returns `Option<(&mut [T], &mut [T])>`.
//!
//! That composed payload is two fat slice pointers `{ptr, len, ptr, len}`; the
//! `None` niche lives in the first data pointer. The enum slot map used to back
//! only ONE pointer (the niche carrier), so the second slice pointer had no
//! provenance-preserving `ptr` slot and lowering failed closed with "refusing
//! to erase LLVM pointer provenance". The fix gives each extra pointer leaf its
//! own `ptr` slot, so BOTH slice pointers survive the memory round-trip.
//!
//! The kernel splits a buffer, bumps the two halves by different amounts through
//! the two returned slice pointers, and the host verifies both — so a dropped or
//! provenance-stripped second pointer produces a wrong result, not just a
//! codegen abort.
//!
//! Usage:
//!   cargo oxide run split_at_mut

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    /// Thread 0 splits `buf[..n]` at `k` via `split_at_mut_checked` and bumps the
    /// left half by 1 and the right half by 100 through the two returned slices.
    #[kernel]
    pub fn split_and_bump(mut buf: DisjointSlice<u32>, n: usize, k: usize) {
        if thread::index_1d().get() == 0 {
            unsafe {
                let s = core::slice::from_raw_parts_mut(buf.as_mut_ptr(), n);
                if let Some((left, right)) = s.split_at_mut_checked(k) {
                    for x in left.iter_mut() {
                        *x = x.wrapping_add(1);
                    }
                    for x in right.iter_mut() {
                        *x = x.wrapping_add(100);
                    }
                }
            }
        }
    }
}

fn main() {
    println!("=== split_at_mut ===");
    const N: usize = 64;
    const K: usize = 25;

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load");
    let cfg = LaunchConfig::for_num_elems(1);

    let host: Vec<u32> = (0..N as u32).collect();
    let mut buf = DeviceBuffer::from_host(&stream, &host).unwrap();
    // SAFETY: launch shape/resources match the kernel; the buffer covers its writes.
    unsafe { module.split_and_bump(&stream, cfg, &mut buf, N, K) }.expect("split_and_bump launch");
    let got = buf.to_host_vec(&stream).unwrap();

    let mut want = host.clone();
    for (i, w) in want.iter_mut().enumerate() {
        *w = w.wrapping_add(if i < K { 1 } else { 100 });
    }
    assert_eq!(
        got, want,
        "split_at_mut: both halves must be bumped through their own pointer"
    );

    println!("PASS: split_at_mut (Option<(&mut [T], &mut [T])> both pointers live)");
}
