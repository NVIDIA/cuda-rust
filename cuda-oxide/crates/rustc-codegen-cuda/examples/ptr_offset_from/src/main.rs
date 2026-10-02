// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Regression coverage for raw pointer distance, wrapping-offset, and provenance APIs.
//!
//! The stable raw pointer methods lower through rustc intrinsics in MIR:
//! `offset_from_unsigned` uses `ptr_offset_from_unsigned`, while
//! `offset_from` and the byte-oriented wrappers use `ptr_offset_from`.
//! Safe `wrapping_offset`/`wrapping_add` use `arith_offset`. Strict-provenance
//! `addr`, `with_addr`, and `map_addr`, plus exposed-provenance
//! `expose_provenance` and `with_exposed_provenance`, must preserve the expected
//! pointer/address semantics. cuda-oxide must preserve signed distances, scale
//! offsets by the pointee type, retain pointer mutability through libcore's cast,
//! and leave zero-sized pointees stationary.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(always)]
    fn wrapping_offset<T>(pointer: *const T, count: isize) -> *const T {
        pointer.wrapping_offset(count)
    }

    #[kernel]
    pub fn pointer_offsets(
        values: &[u32],
        lo: usize,
        hi: usize,
        extreme: isize,
        mut out: DisjointSlice<i64>,
    ) {
        if thread::index_1d().get() == 0 {
            let base = values.as_ptr();
            unsafe {
                let lo_ptr = base.add(lo);
                let hi_ptr = base.add(hi);

                *out.get_unchecked_mut(0) = hi_ptr.offset_from_unsigned(lo_ptr) as i64;
                *out.get_unchecked_mut(1) = lo_ptr.offset_from(hi_ptr) as i64;
                *out.get_unchecked_mut(2) = hi_ptr.byte_offset_from(lo_ptr) as i64;
                *out.get_unchecked_mut(3) = hi_ptr.byte_offset_from_unsigned(lo_ptr) as i64;
                *out.get_unchecked_mut(4) = *base.wrapping_add(hi) as i64;
                *out.get_unchecked_mut(5) = *hi_ptr.wrapping_offset(-((hi - lo) as isize)) as i64;
                *out.get_unchecked_mut(6) =
                    *base.cast::<[u32; 3]>().wrapping_add(2).cast::<u32>() as i64;
                *out.get_unchecked_mut(7) = *(base as *mut u32).wrapping_add(lo) as i64;
                *out.get_unchecked_mut(8) =
                    (wrapping_offset(base.cast::<()>(), hi as isize) == base.cast::<()>()) as i64;

                let wrapped = base.wrapping_offset(extreme).addr();
                let byte_delta = wrapped.wrapping_sub(base.addr());
                let expected_delta = (extreme as usize).wrapping_mul(core::mem::size_of::<u32>());
                *out.get_unchecked_mut(9) = (byte_delta == expected_delta) as i64;
                *out.get_unchecked_mut(10) =
                    (wrapping_offset(base.cast::<()>(), extreme) == base.cast::<()>()) as i64;

                let with_addr = base.with_addr(hi_ptr.addr());
                *out.get_unchecked_mut(11) = (*with_addr == values[hi]) as i64;

                let mapped = base.map_addr(|addr| {
                    addr.wrapping_add(hi.wrapping_mul(core::mem::size_of::<u32>()))
                });
                *out.get_unchecked_mut(12) = (*mapped == values[hi]) as i64;

                let exposed = hi_ptr.expose_provenance();
                *out.get_unchecked_mut(13) = (exposed == hi_ptr.addr()) as i64;

                let reconstructed = core::ptr::with_exposed_provenance::<u32>(exposed);
                *out.get_unchecked_mut(14) = (*reconstructed == values[hi]) as i64;
            }
        }
    }
}

const SLOTS: usize = 15;

fn main() {
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load");

    let values: Vec<u32> = (0..16).map(|v| v * 11).collect();
    let lo = 3usize;
    let hi = 9usize;
    let extreme = isize::MAX;

    let values_dev = DeviceBuffer::from_host(&stream, &values).unwrap();
    let mut out_dev = DeviceBuffer::<i64>::zeroed(&stream, SLOTS).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.pointer_offsets(
            &stream,
            LaunchConfig::for_num_elems(1),
            &values_dev,
            lo,
            hi,
            extreme,
            &mut out_dev,
        )
    }
    .expect("launch pointer_offsets");

    let got = out_dev.to_host_vec(&stream).unwrap();
    let element_distance = (hi - lo) as i64;
    let byte_distance = ((hi - lo) * core::mem::size_of::<u32>()) as i64;
    let expected = [
        element_distance,
        -element_distance,
        byte_distance,
        byte_distance,
        values[hi] as i64,
        values[lo] as i64,
        values[6] as i64,
        values[lo] as i64,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ];

    let mut errors = 0;
    for (i, (&actual, &want)) in got.iter().zip(expected.iter()).enumerate() {
        if actual != want {
            eprintln!("slot {i}: expected {want}, got {actual}");
            errors += 1;
        }
    }

    if errors == 0 {
        println!("SUCCESS: pointer distance, wrapping, and provenance operations are correct");
    } else {
        println!("FAILURE: {errors} pointer/provenance mismatches");
        std::process::exit(1);
    }
}
