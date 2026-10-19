/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! cuda-oxide device stage of the native-cubin interop example.
//!
//! Compiled by `cargo oxide run interop_cubin_identity` through the
//! rustc-codegen-cuda backend in NVVM IR mode, then finalized into
//! `scale_offset_device.cubin` by the libNVVM + nvJitLink finalizer. The
//! host crate loads that cubin at run time and launches the kernel by name.

use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// `output[i] = input[i] * scale + offset` for the first `n` elements.
    ///
    /// `input` and `output` are ordinary device pointers owned by the host
    /// runtime that loaded this cubin.
    #[kernel]
    pub unsafe fn scale_offset_f32(
        n: u32,
        scale: f32,
        offset: f32,
        input: *const f32,
        output: *mut f32,
    ) {
        let idx = thread::index_1d().get();
        if idx < n as usize {
            let x = unsafe { *input.add(idx) };
            unsafe {
                *output.add(idx) = x * scale + offset;
            }
        }
    }
}

fn main() {}
