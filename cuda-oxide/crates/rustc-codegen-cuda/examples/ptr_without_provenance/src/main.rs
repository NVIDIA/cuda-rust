// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Regression coverage for non-null integer-to-pointer constants.
//!
//! `core::ptr::without_provenance::<T>(N)` const-evaluates to a pointer
//! constant whose bytes carry no provenance. The importer must decode the
//! raw address exactly; an old Debug-string byte parser mis-decoded these
//! partially-initialized constant bytes.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

const LOW_ADDR: usize = 0x1000;
const TAG_ADDR: usize = 0xDEAD_BEE0;

#[cuda_module]
mod kernels {
    use super::*;

    // Pointers minted from plain integers at const-eval time.
    const LOW: *const u32 = core::ptr::without_provenance::<u32>(LOW_ADDR);
    const TAG: *mut u8 = core::ptr::without_provenance_mut::<u8>(TAG_ADDR);

    #[kernel]
    pub fn check(low_addr: usize, tag_addr: usize, mut out: DisjointSlice<u64>) {
        if thread::index_1d().get() == 0 {
            unsafe {
                // The constants must decode to their exact addresses.
                *out.get_unchecked_mut(0) = LOW.addr() as u64;
                *out.get_unchecked_mut(1) = TAG.addr() as u64;
                // And they must match the same addresses computed at runtime.
                let low_rt = core::ptr::without_provenance::<u32>(low_addr);
                let tag_rt = core::ptr::with_exposed_provenance_mut::<u8>(tag_addr);
                *out.get_unchecked_mut(2) = (LOW == low_rt) as u64;
                *out.get_unchecked_mut(3) = (TAG.addr() == tag_rt.addr()) as u64;
            }
        }
    }
}

const SLOTS: usize = 4;

fn main() {
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load");

    let mut out_dev = DeviceBuffer::<u64>::zeroed(&stream, SLOTS).unwrap();

    // SAFETY: launch shape/resources match the kernel; the buffer covers its accesses.
    unsafe {
        module.check(
            &stream,
            LaunchConfig::for_num_elems(1),
            LOW_ADDR,
            TAG_ADDR,
            &mut out_dev,
        )
    }
    .expect("launch check");

    let got = out_dev.to_host_vec(&stream).unwrap();
    let expected = [LOW_ADDR as u64, TAG_ADDR as u64, 1, 1];

    let mut errors = 0;
    for (i, (&actual, &want)) in got.iter().zip(expected.iter()).enumerate() {
        if actual != want {
            eprintln!("slot {i}: expected {want:#x}, got {actual:#x}");
            errors += 1;
        }
    }

    if errors == 0 {
        println!("SUCCESS: without_provenance constants decode to their exact addresses");
    } else {
        println!("FAILURE: {errors} without_provenance mismatches");
        std::process::exit(1);
    }
}
