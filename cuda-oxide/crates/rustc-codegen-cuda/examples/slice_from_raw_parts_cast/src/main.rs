/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression test for `slice::from_raw_parts` over a reinterpret-cast pointer.
//!
//! Building `&mut [(u64, u64)]` from a `*mut u64` via `p as *mut (u64, u64)`
//! left the fat pointer's data operand typed to the pre-cast pointee (`*mut u64`)
//! while the slice element is `(u64, u64)`, so `mir.construct_slice` failed to
//! verify: "MirConstructSliceOp data pointer pointee mismatch". The fix coerces
//! the data pointer to a generic-address-space pointer to the slice element type
//! at the from_raw_parts site.
//!
//! Two kernels: a global-memory reinterpret (address space 0) and a shared-memory
//! reinterpret (address space 3), so the coercion is exercised for both — the
//! shared case is why the coercion must target the generic address space, not the
//! source pointer's.
//!
//! Usage:
//!   cargo oxide run slice_from_raw_parts_cast

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    /// Reinterpret `buf` (`2*n` u64s, global memory) as `n` `(u64, u64)` pairs via
    /// a pointer cast + `from_raw_parts_mut`, then bump each lane of every pair.
    #[kernel]
    pub fn bump_pairs(mut buf: DisjointSlice<u64>, n: usize) {
        if thread::index_1d().get() == 0 {
            unsafe {
                let p = buf.as_mut_ptr() as *mut (u64, u64);
                let pairs = core::slice::from_raw_parts_mut(p, n);
                let mut i = 0;
                while i < n {
                    pairs[i] = (pairs[i].0 + 10, pairs[i].1 + 20);
                    i += 1;
                }
            }
        }
    }

    /// Load 128 u64s into shared memory, reinterpret that shared buffer as 64
    /// `(u64, u64)` pairs via `from_raw_parts` over a *shared* (addrspace 3)
    /// pointer, and sum both lanes of every pair into `out[0]`.
    #[kernel]
    pub fn shared_pair_sum(input: &[u64], mut out: DisjointSlice<u64>) {
        static mut SMEM: SharedArray<u64, 128> = SharedArray::UNINIT;
        if thread::index_1d().get() == 0 {
            unsafe {
                let sp = core::ptr::addr_of_mut!(SMEM) as *mut u64;
                let mut k = 0;
                while k < 128 {
                    *sp.add(k) = input[k];
                    k += 1;
                }
                let pairs =
                    core::slice::from_raw_parts(core::ptr::addr_of!(SMEM) as *const (u64, u64), 64);
                let mut s = 0u64;
                let mut j = 0;
                while j < 64 {
                    s = s.wrapping_add(pairs[j].0).wrapping_add(pairs[j].1);
                    j += 1;
                }
                if let Some((slot, _)) = out.get_mut_indexed() {
                    *slot = s;
                }
            }
        }
    }
}

fn main() {
    println!("=== slice_from_raw_parts_cast ===");
    const N: usize = 64; // pairs
    let host: Vec<u64> = (0..(2 * N) as u64).collect();

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load");
    let cfg = LaunchConfig::for_num_elems(1);

    // Global-memory reinterpret.
    let mut buf = DeviceBuffer::from_host(&stream, &host).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffer covers its accesses.
    unsafe { module.bump_pairs(&stream, cfg, &mut buf, N) }.expect("bump_pairs launch");
    let got = buf.to_host_vec(&stream).unwrap();
    let mut want = host.clone();
    for i in 0..N {
        want[2 * i] += 10;
        want[2 * i + 1] += 20;
    }
    assert_eq!(got, want, "bump_pairs");

    // Shared-memory reinterpret.
    let din = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut out = DeviceBuffer::<u64>::zeroed(&stream, 1).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.shared_pair_sum(&stream, cfg, &din, &mut out) }
        .expect("shared_pair_sum launch");
    let got_sum = out.to_host_vec(&stream).unwrap()[0];
    let want_sum: u64 = host.iter().copied().sum();
    assert_eq!(got_sum, want_sum, "shared_pair_sum");

    println!("PASS: slice_from_raw_parts_cast (global + shared reinterpret)");
}
