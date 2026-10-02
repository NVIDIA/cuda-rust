/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Message-carrying device panic lowers to a trap
//!
//! A `#[kernel]` that reaches a panic carrying a message must compile. The
//! panic path is dropped and replaced by `nvvm.trap` (`trap;` in PTX), which
//! aborts the kernel; the message itself is discarded, because there is no
//! panic runtime and no formatting machinery on the device.
//!
//! The trigger is a `panic!("...")` guarded by a value read from device
//! memory, so the panic path is data-dependent and survives `-C opt-level=3`.
//!
//! Test 1 feeds in-range inputs and verifies the scaled outputs.
//! Test 2 plants one out-of-range input; that thread must trap, which the
//! driver reports as a launch failure. A launch that instead *succeeds* would
//! mean the panic path fell through rather than trapping.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

const N: usize = 1024;
/// Largest input the kernel accepts; anything above it panics.
const LIMIT: u32 = 1_000;
const SCALE: u64 = 3;
const SENTINEL: u64 = u64::MAX;

#[cuda_module]
mod kernels {
    use super::*;

    /// Scale `inputs[i]` into `out[i]`, panicking on an out-of-range input.
    ///
    /// The `panic!` builds a message string, so before the message-eliding fix
    /// the backend rejected this kernel outright instead of compiling the path
    /// to a trap.
    #[kernel]
    pub fn scale_checked(inputs: &[u32], mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        let value = inputs[idx.get()];

        if value > LIMIT {
            panic!("input exceeds the supported range");
        }

        if let Some(slot) = out.get_mut(idx) {
            *slot = value as u64 * SCALE;
        }
    }
}

fn launch_scale_checked(
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
    inputs_host: &[u32],
) -> Result<Vec<u64>, cuda_core::DriverError> {
    let inputs = DeviceBuffer::from_host(stream, inputs_host)?;
    let mut out = DeviceBuffer::from_host(stream, &vec![SENTINEL; N])?;

    let config = LaunchConfig {
        grid_dim: ((N as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: one thread per output element
    unsafe { module.scale_checked(stream.as_ref(), config, &inputs, &mut out) }?;
    out.to_host_vec(stream)
}

fn in_range_inputs() -> Vec<u32> {
    (0..N).map(|i| (i as u32) % (LIMIT + 1)).collect()
}

fn run_in_range(module: &kernels::LoadedModule, stream: &Arc<CudaStream>) -> bool {
    let inputs_host = in_range_inputs();
    let out = match launch_scale_checked(module, stream, &inputs_host) {
        Ok(out) => out,
        Err(e) => {
            println!("in_range: FAIL (launch error: {})", e);
            return false;
        }
    };

    for (i, &value) in out.iter().enumerate() {
        let expected = inputs_host[i] as u64 * SCALE;
        if value != expected {
            println!(
                "in_range: FAIL (out[{}] = {}, expected {})",
                i, value, expected
            );
            return false;
        }
    }

    println!("in_range: PASS");
    true
}

fn run_out_of_range(module: &kernels::LoadedModule, stream: &Arc<CudaStream>) -> bool {
    let mut inputs_host = in_range_inputs();
    inputs_host[0] = LIMIT + 1; // thread 0 reaches the `panic!`

    match launch_scale_checked(module, stream, &inputs_host) {
        // The panic path lowered to something that fell through instead of
        // trapping: the kernel produced a value for an input it rejects.
        Ok(out) => {
            println!(
                "out_of_range: FAIL (kernel did not trap, out[0] = {})",
                out[0]
            );
            false
        }
        // PTX `trap` surfaces as a generic launch failure.
        Err(e) => {
            println!("out_of_range: PASS (thread trapped: {})", e);
            true
        }
    }
}

fn main() {
    println!("=== Message-Carrying Device Panic Traps ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    println!("Device ordinal: {}\n", ctx.ordinal());

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let stream = ctx.default_stream();

    // The trapping test runs last: a trap poisons the context for every
    // launch after it.
    let in_range_ok = run_in_range(&module, &stream);
    let trap_ok = run_out_of_range(&module, &stream);

    if in_range_ok && trap_ok {
        println!("\nSUCCESS");
    } else {
        println!("\nFAILURE");
        std::process::exit(1);
    }
}
