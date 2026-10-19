// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Regression: a `#[kernel]` calls a RECURSIVE function defined in a dependency crate.
//!
//! Before `-Zalways-encode-mir` was added to cargo-oxide's device rustflags, this failed with
//! `Symbol reprolib__rec1 not found`: rustc encodes cross-crate MIR only for `#[inline]`/generic
//! items, so the non-inline, non-generic (and un-inlinable, because recursive) `reprolib::rec1` was
//! called but its definition was never emitted into the device module. The identical function
//! defined LOCALLY in this crate always compiled, which isolates the axis to "cross-crate + not
//! inlinable", not the function body. See PR "pass -Zalways-encode-mir ...".

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;
    #[kernel]
    pub unsafe fn recurse_sum(inp: &[u64], mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        let r = reprolib::rec1(inp);
        if let Some(slot) = out.get_mut(idx) {
            *slot = r;
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load");

    let host: Vec<u64> = (1..=16u64).collect();
    let expected: u64 = host.iter().copied().sum();
    let inp = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut out = DeviceBuffer::<u64>::zeroed(&stream, 1).unwrap();

    // SAFETY: one thread; the kernel reads `inp` and writes `out[0]`.
    unsafe { module.recurse_sum(&stream, LaunchConfig::for_num_elems(1), &inp, &mut out) }
        .expect("launch");

    let got = out.to_host_vec(&stream).unwrap()[0];
    if got == expected {
        println!("PASS cross_crate_recursion: sum={got}");
    } else {
        println!("FAIL cross_crate_recursion: got {got}, expected {expected}");
        std::process::exit(1);
    }
}
