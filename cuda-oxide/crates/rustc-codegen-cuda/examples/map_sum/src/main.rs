/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression test for `iter().map(...).sum()` — a struct with two zero-sized
//! fields addressed off a shared base pointer.
//!
//! `Iterator::sum` composes closures via `core`'s `map_fold`:
//! `move |acc, elt| g(acc, f(elt))`, which captures the map closure `f` and the
//! `Sum::sum` fold closure `g` as upvars. Both are zero-sized, so the composed
//! closure is a ZST struct with two ZST fields, and its body borrows both upvars
//! off the same base pointer.
//!
//! In `convert_field_addr`, the ZST-field branch used to forward the base SSA
//! value directly for the first field. That type-punned the base pointer to the
//! field's pointee in dialect conversion's type history, so the *sibling*
//! `field_addr` for the second field resolved its base pointee to the wrong
//! (zero-field) type and failed to lower:
//!
//! ```text
//! field_addr index 1 out of bounds for struct with 0 fields
//! ```
//!
//! The fix emits an explicit zero-offset GEP (a distinct result) for the ZST
//! field, mirroring the union branch, so the base pointer's recorded type stays
//! intact for the sibling access.
//!
//! Two kernels cover both element types that appear in practice (`i64` and
//! `usize` sums), each host-verified.
//!
//! Usage:
//!   cargo oxide run map_sum

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    /// `out[tid] = sum(2*x for x in data)` via `iter().map(...).sum::<i64>()`.
    #[kernel]
    pub fn map_sum_i64(data: &[i64], mut out: DisjointSlice<i64>) {
        if let Some(slot) = out.get_mut(thread::index_1d()) {
            *slot = data.iter().map(|&x| x.wrapping_mul(2)).sum();
        }
    }

    /// `out[tid] = sum(x+1 for x in data)` via `iter().map(...).sum::<usize>()`
    /// (the `usize` composed closure the chaining kernels hit).
    #[kernel]
    pub fn map_sum_usize(data: &[u64], mut out: DisjointSlice<u64>) {
        if let Some(slot) = out.get_mut(thread::index_1d()) {
            let s: usize = data.iter().map(|&x| x as usize + 1).sum();
            *slot = s as u64;
        }
    }
}

fn main() {
    println!("=== map_sum ===");
    const N: usize = 64;

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load");
    let cfg = LaunchConfig::for_num_elems(N as u32);

    // i64 map+sum.
    let data_i: Vec<i64> = (0..N as i64).collect();
    let din = DeviceBuffer::from_host(&stream, &data_i).unwrap();
    let mut out = DeviceBuffer::<i64>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.map_sum_i64(&stream, cfg, &din, &mut out) }.expect("map_sum_i64 launch");
    let got = out.to_host_vec(&stream).unwrap();
    let want_i: i64 = data_i.iter().map(|&x| x.wrapping_mul(2)).sum();
    for (tid, &g) in got.iter().enumerate() {
        assert_eq!(g, want_i, "map_sum_i64 thread {tid}");
    }

    // usize map+sum.
    let data_u: Vec<u64> = (0..N as u64).collect();
    let dinu = DeviceBuffer::from_host(&stream, &data_u).unwrap();
    let mut outu = DeviceBuffer::<u64>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.map_sum_usize(&stream, cfg, &dinu, &mut outu) }.expect("map_sum_usize launch");
    let gotu = outu.to_host_vec(&stream).unwrap();
    let want_u: u64 = data_u.iter().map(|&x| x + 1).sum();
    for (tid, &g) in gotu.iter().enumerate() {
        assert_eq!(g, want_u, "map_sum_usize thread {tid}");
    }

    println!("PASS: map_sum (i64 + usize iter().map(..).sum())");
}
