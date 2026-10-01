/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `assert_inhabited` on an uninhabited type must trap on the GPU
//!
//! `MaybeUninit::<Infallible>::uninit().assume_init()` reaches rustc's
//! `assert_inhabited::<Infallible>` guard, which panics at runtime
//! ("attempted to instantiate uninhabited type"), so the kernel must trap.
//! If the launch succeeds instead, the panic path was compiled away.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn assume_init_guard(flag: u32, mut out: DisjointSlice<u32>) {
        if let Some(slot) = out.get_mut(thread::index_1d()) {
            if flag != 0 {
                // assert_inhabited::<Infallible> panics ("attempted to instantiate uninhabited type")
                // the kernel must trap
                //
                // The invalid `assume_init` is this example's entire point:
                // rustc's assert_inhabited guard must survive into PTX as a
                // trap. No Infallible value ever materializes; execution
                // stops at the guard. Hence the targeted lint allows.
                #[allow(invalid_value, clippy::uninit_assumed_init)]
                unsafe {
                    core::mem::MaybeUninit::<core::convert::Infallible>::uninit().assume_init()
                };
            }
            *slot = 1;
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load module");

    let mut out = DeviceBuffer::from_host(&stream, &[0u32]).expect("out");

    // SAFETY: one thread, one output element
    let result = unsafe {
        module.assume_init_guard(&stream, LaunchConfig::for_num_elems(1), 1u32, &mut out)
    }
    .and_then(|()| out.to_host_vec(&stream));

    match result {
        Err(e) => println!("PASS (kernel trapped: {})", e),
        Ok(out) => {
            println!(
                "FAIL (uninhabited panic path did not trap, out[0] = {})",
                out[0]
            );
            std::process::exit(1);
        }
    }
}
