/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Scalar -> aggregate transmute lowering (regression).
//!
//! Transmuting an integer into a single-field struct that the importer does
//! not classify as a niche-optimised enum (e.g. `usize -> NonNull<T>`, which
//! appears in `core::fmt`) used to be refused in `mir-lower` with
//! `scalar -> aggregate Transmute without niche encoding`. The fix lowers it
//! as a faithful, size-checked memory round-trip via `emit_transmute_via_memory`.
//!
//! The kernel transmutes a (non-zero) address `usize -> NonNull<u8>` and back
//! `NonNull<u8> -> usize` without dereferencing, so the host can verify the
//! address survives the round-trip:
//!   out[i] = in[i]
//!
//! Run: cargo oxide run transmute_scalar_struct

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// usize -> NonNull<u8> : scalar -> aggregate transmute (no niche).
    #[inline(never)]
    fn to_nonnull(addr: usize) -> core::ptr::NonNull<u8> {
        unsafe { core::mem::transmute::<usize, core::ptr::NonNull<u8>>(addr) }
    }

    /// NonNull<u8> -> usize : aggregate -> scalar transmute (the reverse).
    #[inline(never)]
    fn from_nonnull(p: core::ptr::NonNull<u8>) -> usize {
        unsafe { core::mem::transmute::<core::ptr::NonNull<u8>, usize>(p) }
    }

    #[kernel]
    pub fn roundtrip(addrs: &[u64], mut out: DisjointSlice<u64>) {
        let t = thread::index_1d();
        let i = t.get();
        // Pointer is never dereferenced; we only round-trip the bits.
        let nn = to_nonnull(addrs[i] as usize);
        let back = from_nonnull(nn);
        if let Some(slot) = out.get_mut(t) {
            *slot = back as u64;
        }
    }
}

const N: usize = 64;

fn main() {
    println!("=== scalar <-> aggregate transmute (usize <-> NonNull) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let stream = ctx.default_stream();

    // Non-zero fabricated addresses (never dereferenced).
    let addrs: Vec<u64> = (0..N as u64).map(|i| 0x1000 + i * 8).collect();
    let d_in = DeviceBuffer::from_host(&stream, &addrs).unwrap();
    let mut d_out = DeviceBuffer::<u64>::zeroed(&stream, N).unwrap();

    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (N as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.roundtrip(stream.as_ref(), config, &d_in, &mut d_out) }
        .expect("Kernel launch failed");

    let out = d_out.to_host_vec(&stream).unwrap();
    let mut ok = true;
    for i in 0..N {
        if out[i] != addrs[i] {
            println!("FAIL: lane {i}: out={:#x} (want {:#x})", out[i], addrs[i]);
            ok = false;
            break;
        }
    }
    if ok {
        println!("SUCCESS: all {N} addresses survived usize<->NonNull round-trip");
    } else {
        std::process::exit(1);
    }
}
