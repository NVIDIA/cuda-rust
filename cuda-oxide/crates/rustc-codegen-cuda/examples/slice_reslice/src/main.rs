/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Minimal repro for issue #126: re-slicing a slice parameter inside a kernel
// (e.g. `&bytes[2..]`) used to fail device codegen with
// "Unsupported construct: Aggregate kind RawPtr(...) not yet supported".
//
// Range indexing on a slice goes through core's
// `slice::index::get_offset_len_noubcheck`, whose MIR builds the new fat
// pointer with `Rvalue::Aggregate(AggregateKind::RawPtr(..), [data_ptr, len])`.
// The importer had no arm for that aggregate kind, so any kernel that passed
// a sub-slice to a helper function failed to compile.
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    /// Reads the first two bytes of `stuff` as a little-endian u16 value.
    /// Taking `&[u8]` (not an offset) is the point: callers must be able to
    /// hand it a re-sliced view of a kernel parameter.
    fn first_two_le(stuff: &[u8]) -> u32 {
        (stuff[0] as u32) | ((stuff[1] as u32) << 8)
    }

    #[kernel]
    pub fn slice_reslice(bytes: &[u8], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            // Constant offset, like the issue's `&y0[0..]`.
            let head = first_two_le(&bytes[0..]);
            // Runtime offset: each thread reads its own 2-byte window.
            let window = first_two_le(&bytes[2 * i..]);
            *out_elem = head.wrapping_add(window);
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let m = kernels::load(&ctx).expect("load");

    const N: usize = 64;
    // Enough bytes that every thread's `2 * i ..` window holds two bytes.
    let bytes: Vec<u8> = (0..(2 * N + 2)).map(|v| (v * 7 % 251) as u8).collect();

    let bytes_dev = DeviceBuffer::from_host(&stream, &bytes).unwrap();
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        m.slice_reslice(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &bytes_dev,
            &mut out_dev,
        )
    }
    .expect("launch slice_reslice");

    let out = out_dev.to_host_vec(&stream).unwrap();

    let le16 = |lo: u8, hi: u8| (lo as u32) | ((hi as u32) << 8);
    let head = le16(bytes[0], bytes[1]);
    let mut errors = 0;
    for (i, &got) in out.iter().enumerate() {
        let expected = head.wrapping_add(le16(bytes[2 * i], bytes[2 * i + 1]));
        if got != expected {
            if errors < 5 {
                eprintln!("  Error at [{i}]: expected {expected}, got {got}");
            }
            errors += 1;
        }
    }

    if errors == 0 {
        println!("SUCCESS: all {N} re-sliced windows correct");
    } else {
        println!("FAILURE: {errors} mismatches");
        std::process::exit(1);
    }
}
