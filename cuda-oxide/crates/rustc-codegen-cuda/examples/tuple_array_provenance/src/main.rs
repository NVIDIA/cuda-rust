/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Array of tuples whose elements hold thin pointers to device statics.
//!
//! Aggregate const values materialize each thin pointer field via
//! `MirGlobalAllocOp`.
//!
//! Run: `cargo oxide run tuple_array_provenance`

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

static FIRST: u32 = 11;
static SECOND: u32 = 17;
const POINTERS: [(&u32, bool); 2] = [(&FIRST, false), (&SECOND, true)];

#[cuda_module]
mod kernels {
    use super::*;

    /// # Safety
    ///
    /// `output` must point to writable device-accessible storage for one `u32` per
    /// launched thread.
    #[kernel]
    pub unsafe fn tuple_array_pointer(output: *mut u32) {
        let index = thread::index_1d().get();
        let (pointer, flag) = POINTERS[index & 1];
        unsafe {
            output.add(index).write(*pointer + flag as u32);
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load module");

    let out = DeviceBuffer::<u32>::zeroed(&stream, 2).expect("alloc out");
    // SAFETY: two-thread launch writing two u32s.
    unsafe {
        module
            .tuple_array_pointer(
                &stream,
                LaunchConfig::for_num_elems(2),
                out.cu_deviceptr() as *mut u32,
            )
            .expect("launch");
    }
    stream.synchronize().expect("sync");

    let host = out.to_host_vec(&stream).expect("dtoh");
    let expected = [FIRST + false as u32, SECOND + true as u32];
    assert_eq!(host.as_slice(), expected.as_slice());
    println!("tuple_array_provenance: PASS ({:?})", host);
}
