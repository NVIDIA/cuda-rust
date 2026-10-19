/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Interior-addend device-static array-to-slice unsize.
//!
//! `const PAIR_SLICE: &[f32] = &TABLE[2]` produces a fat slice pointer whose
//! data word carries a non-zero byte addend into `TABLE` and whose metadata
//! word stores the slice length.
//!
//! Run: `cargo oxide run static_slice_addend`

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::kernel;
use cuda_host::cuda_module;

static TABLE: [[f32; 2]; 4] = [[0.25, 0.5], [1.0, 2.0], [4.0, 8.0], [16.0, 32.0]];

/// `&TABLE[2]` selects the third nested array. The resulting slice data
/// pointer has a 16-byte addend and its metadata stores length 2.
const PAIR_SLICE: &[f32] = &TABLE[2];

#[inline(never)]
fn pair_slice() -> &'static [f32] {
    PAIR_SLICE
}

#[cuda_module]
mod kernels {
    use super::*;

    /// # Safety
    ///
    /// `out` must point to device-accessible storage that is properly aligned
    /// and writable for two `f32` values. No other thread may race with these
    /// writes.
    #[kernel]
    pub unsafe fn slice_addend(out: *mut f32) {
        let pair = pair_slice();

        unsafe {
            *out = pair[0] + pair[1];
            *out.add(1) = pair.len() as f32;
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load module");

    let out = DeviceBuffer::<f32>::zeroed(&stream, 2).expect("alloc out");

    // SAFETY: one-thread launch writing two f32 values.
    unsafe {
        module
            .slice_addend(
                &stream,
                LaunchConfig::for_num_elems(1),
                out.cu_deviceptr() as *mut f32,
            )
            .expect("launch slice_addend");
    }

    stream.synchronize().expect("sync");

    let host = out.to_host_vec(&stream).expect("dtoh");
    let expected_sum = TABLE[2][0] + TABLE[2][1];

    assert!(
        (host[0] - expected_sum).abs() < 1e-6,
        "sum: got {} expected {}",
        host[0],
        expected_sum
    );
    assert!(
        (host[1] - 2.0).abs() < 1e-6,
        "len: got {} expected 2",
        host[1]
    );

    println!(
        "static_slice_addend: PASS (sum {}, len {})",
        host[0], host[1]
    );
}
