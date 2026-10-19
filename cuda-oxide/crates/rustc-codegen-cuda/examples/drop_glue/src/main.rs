/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Positive test: device-side drop execution and suppression semantics.
//!
//! Verifies ordinary device-side drop glue together with core wrappers that
//! intentionally suppress or explicitly trigger destruction:
//!
//! - `ManuallyDrop::new`
//! - `ManuallyDrop::drop`
//! - `mem::forget`
//! - `MaybeUninit::new`
//! - `MaybeUninit::write` + `assume_init_drop`
//! - `MaybeUninit::assume_init_read`
//!
//! Usage:
//!   cargo oxide run drop_glue
//!
//! Expected: all drop execution and suppression checks pass.

use core::mem::{ManuallyDrop, MaybeUninit};

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

const DROP_SENTINEL: u32 = 0xDEAD_BEEF;

pub struct DropMarker {
    target: *mut u32,
}

impl Drop for DropMarker {
    fn drop(&mut self) {
        unsafe {
            self.target.write(DROP_SENTINEL);
        }
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn drop_glue_kernel(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(slot) = out.get_mut(idx) {
            *slot = 0;
            let _m = DropMarker {
                target: slot as *mut u32,
            };
            // `_m` drops at end of scope and writes `DROP_SENTINEL`.
        }
    }

    #[kernel]
    pub fn drop_control_kernel(mut out: DisjointSlice<u32>) {
        if thread::index_1d().get() != 0 {
            return;
        }

        unsafe {
            // SAFETY: the host launches exactly one active thread and supplies
            // a seven-element output allocation owned by this thread.
            let out_ptr = out.as_mut_ptr();

            // Lane 0: ordinary scope exit runs drop glue.
            let slot = out_ptr.add(0);
            slot.write(0x1111_0000);
            {
                let _marker = DropMarker { target: slot };
            }

            // Lane 1: ManuallyDrop suppresses automatic drop glue.
            let slot = out_ptr.add(1);
            slot.write(0x2222_0000);
            {
                let _marker = ManuallyDrop::new(DropMarker { target: slot });
            }

            // Lane 2: ManuallyDrop::drop explicitly runs drop glue.
            let slot = out_ptr.add(2);
            slot.write(0x3333_0000);
            let mut marker = ManuallyDrop::new(DropMarker { target: slot });
            ManuallyDrop::drop(&mut marker);

            // Lane 3: mem::forget consumes the value without running Drop.
            let slot = out_ptr.add(3);
            slot.write(0x4444_0000);
            core::mem::forget(DropMarker { target: slot });

            // Lane 4: dropping an initialized MaybeUninit<T> does not drop T.
            let slot = out_ptr.add(4);
            slot.write(0x5555_0000);
            {
                let _marker = MaybeUninit::new(DropMarker { target: slot });
            }

            // Lane 5: initialize through MaybeUninit and explicitly drop T.
            let slot = out_ptr.add(5);
            slot.write(0x6666_0000);
            let mut marker = MaybeUninit::<DropMarker>::uninit();
            marker.write(DropMarker { target: slot });
            marker.assume_init_drop();

            // Lane 6: assume_init_read returns an owned T that drops normally.
            let slot = out_ptr.add(6);
            slot.write(0x7777_0000);
            {
                let marker = MaybeUninit::new(DropMarker { target: slot });
                let _owned = marker.assume_init_read();
            }
        }
    }
}

fn main() {
    println!("=== drop_glue ===\n");

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load module");

    // Historical regression: ordinary device-side Drop over many threads.
    const N: usize = 256;
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffer covers its accesses.
    unsafe {
        module
            .drop_glue_kernel(&stream, LaunchConfig::for_num_elems(N as u32), &mut out_dev)
            .expect("drop_glue_kernel launch");
    }

    let out = out_dev.to_host_vec(&stream).unwrap();

    let mut errors = 0usize;
    for (i, &val) in out.iter().enumerate() {
        if val != DROP_SENTINEL {
            if errors < 5 {
                eprintln!(
                    "  FAIL drop_glue[{}]: got {:#010X} expected {:#010X}",
                    i, val, DROP_SENTINEL
                );
            }
            errors += 1;
        }
    }

    if errors == 0 {
        println!("SUCCESS: drop glue wrote sentinel in all {} elements", N);
    } else {
        eprintln!("FAIL: {} errors", errors);
        std::process::exit(1);
    }

    // Focused initialization/drop conformance matrix.
    let mut controls_dev = DeviceBuffer::<u32>::zeroed(&stream, 7).unwrap();

    // SAFETY: exactly one thread owns the seven-element output allocation.
    unsafe {
        module
            .drop_control_kernel(&stream, LaunchConfig::for_num_elems(1), &mut controls_dev)
            .expect("drop_control_kernel launch");
    }

    let got = controls_dev.to_host_vec(&stream).unwrap();
    let expected = [
        DROP_SENTINEL,
        0x2222_0000,
        DROP_SENTINEL,
        0x4444_0000,
        0x5555_0000,
        DROP_SENTINEL,
        DROP_SENTINEL,
    ];

    if got.as_slice() != expected {
        eprintln!("FAIL: initialization/drop conformance mismatch");
        for (i, (&actual, &wanted)) in got.iter().zip(expected.iter()).enumerate() {
            if actual != wanted {
                eprintln!(
                    "  lane {}: got {:#010X} expected {:#010X}",
                    i, actual, wanted
                );
            }
        }
        std::process::exit(1);
    }

    println!("PASS: ordinary scope exit runs drop glue");
    println!("PASS: ManuallyDrop suppresses automatic drop");
    println!("PASS: ManuallyDrop::drop runs drop glue");
    println!("PASS: mem::forget suppresses drop");
    println!("PASS: MaybeUninit suppresses contained drop");
    println!("PASS: MaybeUninit::assume_init_drop runs drop glue");
    println!("PASS: MaybeUninit::assume_init_read preserves inhabited drop path");
    println!("PASS: initialization/drop conformance");
}
