/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! cuda-oxide SIMT stage for the cutile-rs inter-kernel interop example.

use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// Applies a branchy per-element post-process to a Tile-produced buffer.
    ///
    /// `input` and `output` are ordinary device pointers owned by the cutile-rs
    /// host runtime. The example launches this after a cutile-rs row-softmax
    /// kernel on the same CUDA stream.
    #[kernel]
    pub unsafe fn threshold_scale_f32(
        n: u32,
        threshold: f32,
        scale: f32,
        input: *const f32,
        output: *mut f32,
    ) {
        let idx = thread::index_1d().get();
        if idx < n as usize {
            let x = unsafe { *input.add(idx) };
            let y = if x >= threshold { x * scale } else { 0.0 };
            unsafe {
                *output.add(idx) = y;
            }
        }
    }
}

fn main() {}
