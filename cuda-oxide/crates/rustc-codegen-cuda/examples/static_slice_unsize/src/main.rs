/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Zero-addend device-static array→slice unsize.
//!
//! `const TABLE_SLICE: &[f32] = &TABLE` coerces `&[f32; 4]` to a fat `&[f32]`.
//! The importer materializes the thin global pointer plus the array length.
//!
//! Run: `cargo oxide run static_slice_unsize`

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::kernel;
use cuda_host::cuda_module;

static TABLE: [f32; 4] = [0.25, 0.5, 1.0, 2.0];

/// `&TABLE` is `&[f32; 4]`; the unsize coercion to `&[f32]` keeps a zero
/// addend and adds the length metadata a thin static pointer cannot carry.
const TABLE_SLICE: &[f32] = &TABLE;

/// A zero-addend prefix subslice: the same base pointer as `TABLE_SLICE`,
/// but the fat pointer stores length 2. The importer must source the slice
/// length from that metadata word, not from the static array's type.
const TABLE_HEAD: &[f32] = {
    let s: &[f32] = &TABLE;
    s.split_at(2).0
};

#[inline(never)]
fn table_slice() -> &'static [f32] {
    TABLE_SLICE
}

#[inline(never)]
fn table_head() -> &'static [f32] {
    TABLE_HEAD
}

#[cuda_module]
mod kernels {
    use super::*;

    /// # Safety
    ///
    /// `out` must point to device-accessible storage that is properly aligned
    /// and writable for one `f32`. No other thread may race with this write.
    #[kernel]
    pub unsafe fn slice_unsize(out: *mut f32) {
        let table = table_slice();
        unsafe {
            *out = table[0] + table[3];
        }
    }

    /// # Safety
    ///
    /// `out` must point to device-accessible storage that is properly aligned
    /// and writable for two `f32`s. No other thread may race with these writes.
    #[kernel]
    pub unsafe fn head_slice(out: *mut f32) {
        let head = table_head();
        unsafe {
            *out = head[0] + head[1];
            *out.add(1) = head.len() as f32;
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load module");

    let out = DeviceBuffer::<f32>::zeroed(&stream, 1).expect("alloc out");
    // SAFETY: one-thread launch writing a single f32.
    unsafe {
        module
            .slice_unsize(
                &stream,
                LaunchConfig::for_num_elems(1),
                out.cu_deviceptr() as *mut f32,
            )
            .expect("launch");
    }
    stream.synchronize().expect("sync");

    let host = out.to_host_vec(&stream).expect("dtoh");
    let expected = TABLE[0] + TABLE[3];
    assert!(
        (host[0] - expected).abs() < 1e-6,
        "got {} expected {}",
        host[0],
        expected
    );

    // Prefix subslice: the stored fat-pointer length (2) must reach the
    // device, not the static array's length (4).
    let head_out = DeviceBuffer::<f32>::zeroed(&stream, 2).expect("alloc head_out");
    // SAFETY: one-thread launch writing two f32s.
    unsafe {
        module
            .head_slice(
                &stream,
                LaunchConfig::for_num_elems(1),
                head_out.cu_deviceptr() as *mut f32,
            )
            .expect("launch head_slice");
    }
    stream.synchronize().expect("sync");

    let head_host = head_out.to_host_vec(&stream).expect("dtoh head");
    let head_expected = TABLE[0] + TABLE[1];
    assert!(
        (head_host[0] - head_expected).abs() < 1e-6,
        "head sum: got {} expected {}",
        head_host[0],
        head_expected
    );
    assert!(
        (head_host[1] - 2.0).abs() < 1e-6,
        "head len: got {} expected 2",
        head_host[1]
    );
    println!(
        "static_slice_unsize: PASS ({}, head {} len {})",
        host[0], head_host[0], head_host[1]
    );
}
