/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Field-then-index assignment smoke test (issue #235).
//!
//! Exercises the `(Field, ConstantIndex)` and `(Field, Index)` 2-level
//! projection chains in store position. Before the fix, assigning to
//! `local.field[i]` (struct field followed by an array index) crashed with
//! "Unsupported construct: 2-level projection with first elem Field".
//!
//! The kernel builds a local `Packet` struct whose `data` field is a `[u32; 4]`
//! array, writes to each element via both constant and variable indices, then
//! sums them into the output.
//!
//! Run: cargo oxide run field_array_assign

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

/// A plain struct with an array field. Lives only on the device stack.
#[derive(Copy, Clone)]
struct Packet {
    data: [u32; 4],
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Each thread builds a local Packet, assigns to its array field via both
    /// constant indices and a runtime variable index, then writes the sum out.
    #[kernel]
    pub fn sum_packet(scale: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;

        let mut pkt = Packet { data: [0u32; 4] };

        // (Field, ConstantIndex) projections: pkt.data[0], pkt.data[3]
        pkt.data[0] = scale * i;
        pkt.data[3] = scale * i + 3;

        // (Field, Index) projections via a runtime variable. The indexed
        // range loop is the pattern under test: an iterator rewrite would
        // erase the runtime-index MIR projection this regression exercises.
        #[allow(clippy::needless_range_loop)]
        for k in 1usize..3 {
            pkt.data[k] = scale * i + k as u32;
        }

        if let Some(o) = out.get_mut(idx) {
            *o = pkt.data[0] + pkt.data[1] + pkt.data[2] + pkt.data[3];
        }
    }
}

fn main() {
    const N: usize = 256;
    let scale: u32 = 3;

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load module");

    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.sum_packet(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            scale,
            &mut out_dev,
        )
    }
    .expect("kernel launch");

    let out = out_dev.to_host_vec(&stream).unwrap();

    // Each thread i computes: scale*i + (scale*i+1) + (scale*i+2) + (scale*i+3)
    //                        = 4*scale*i + 6
    let mut errors = 0usize;
    for (i, &val) in out.iter().enumerate() {
        let expected = 4 * scale * (i as u32) + 6;
        if val != expected {
            if errors < 5 {
                eprintln!("  FAIL [{}]: got {} want {}", i, val, expected);
            }
            errors += 1;
        }
    }

    if errors == 0 {
        println!("SUCCESS: all {} field-array assignments correct", N);
    } else {
        eprintln!("FAIL: {} errors", errors);
        std::process::exit(1);
    }
}
