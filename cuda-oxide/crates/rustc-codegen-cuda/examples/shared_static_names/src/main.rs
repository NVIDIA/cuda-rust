/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Stable shared-memory and device-global names (#1477)
//!
//! Each `SharedArray` static becomes a `.shared` global, and each ordinary
//! `static` a device kernel reads becomes a `.global`. Their PTX names are
//! derived from the Rust static's mangled symbol, not from a counter over the
//! whole crate, so adding, removing or reordering a kernel does not rename
//! the other kernels' globals:
//!
//! ```text
//!   __shared_mem_<crate id>_<hash of the static's mangled symbol>
//!   __device_global_<crate id>_<hash of the static's mangled symbol>
//! ```
//!
//! The two kernels here rotate a block's thread ids through shared tiles:
//!
//! ```text
//!   rotate_128:  TILE[tid] = tid         ──► sync ──► out[tid] = TILE[(tid + 1) % 128]
//!   rotate_256:  TILE[tid] = tid         ──► sync ──► out[tid] = TILE[(tid + 1) % 256]
//!                TWICE[tid] = 2 * tid                           + TWICE[(tid + 2) % 256] * SCALE[tid % 4]
//! ```
//!
//! The host checks both outputs on the GPU. The PTX contract itself is checked
//! by `scripts/check-shared-static-names.sh`, which builds copies of this
//! crate with extra kernels spliced in at one of the two `ADDED KERNEL SLOT`
//! comments, in release and full-debug builds, and requires the PTX of
//! both kernels and their globals' declarations to stay the same.
//!
//! LLVM numbers basic-block labels by the function's position in the module
//! (`$L__BB<function>_<block>`), which an added kernel can shift, so the check
//! ignores that function number; a release build's PTX for these kernels must
//! otherwise stay byte-identical.
//!
//! Run: `cargo oxide run shared_static_names`

use cuda_core::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;

/// Per-lane weights read by `rotate_256`.
///
/// An ordinary immutable `static`, so it lowers to a `__device_global_*`
/// global rather than shared memory and exercises that naming path too.
static SCALE: [f32; 4] = [1.0, 2.0, 3.0, 4.0];

#[cuda_module]
mod kernels {
    use super::*;

    // ADDED KERNEL SLOT (first)

    /// Writes `(tid + 1) % 128` for each of 128 threads via a 128-entry tile.
    ///
    /// Launch with one block of exactly 128 threads; `out` must hold 128
    /// elements. Every thread writes its own tile slot before the barrier
    /// and reads its neighbor's after it.
    #[kernel]
    pub fn rotate_128(mut out: DisjointSlice<f32>) {
        static mut TILE: SharedArray<f32, 128, 16> = SharedArray::UNINIT;

        let tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TILE) };
        let tid = thread::threadIdx_x() as usize;
        unsafe { *tile.add(tid & 127) = tid as f32 };
        thread::sync_threads();
        unsafe { *out.get_unchecked_mut(tid) = *tile.add((tid + 1) & 127) };
    }

    /// Writes `(tid + 1) % 256 + 2 * ((tid + 2) % 256) * SCALE[tid % 4]`
    /// for each of 256 threads.
    ///
    /// Uses two shared statics, so the check also covers several statics in
    /// one kernel. Launch with one block of exactly 256 threads; `out` must
    /// hold 256 elements. Every thread writes its own slot of both tiles
    /// before the barrier and reads its neighbors' after it.
    #[kernel]
    pub fn rotate_256(mut out: DisjointSlice<f32>) {
        static mut TILE: SharedArray<f32, 256, 16> = SharedArray::UNINIT;
        static mut TWICE: SharedArray<f32, 256> = SharedArray::UNINIT;

        let tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TILE) };
        let twice = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TWICE) };
        let tid = thread::threadIdx_x() as usize;
        unsafe {
            *tile.add(tid & 255) = tid as f32;
            *twice.add(tid & 255) = (2 * tid) as f32;
        }
        thread::sync_threads();
        let scale = SCALE[tid & 3];
        unsafe {
            *out.get_unchecked_mut(tid) =
                *tile.add((tid + 1) & 255) + *twice.add((tid + 2) & 255) * scale;
        }
    }

    // ADDED KERNEL SLOT (last)
}

/// Launches `rotate_128` and `rotate_256` and checks their outputs on the host.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== shared_static_names (#1477) ===");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    let mut failures = 0;
    for threads in [128usize, 256] {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut out = DeviceBuffer::<f32>::zeroed(&stream, threads)?;

        // SAFETY: one block of `threads` threads, each writing only out[tid],
        // and `out` holds exactly `threads` elements
        if threads == 128 {
            unsafe { module.rotate_128(stream.as_ref(), cfg, &mut out) }?;
        } else {
            unsafe { module.rotate_256(stream.as_ref(), cfg, &mut out) }?;
        }

        let result = out.to_host_vec(&stream)?;
        for (tid, &got) in result.iter().enumerate() {
            let neighbor = ((tid + 1) % threads) as f32;
            let expected = if threads == 128 {
                neighbor
            } else {
                neighbor + (2 * ((tid + 2) % threads)) as f32 * SCALE[tid % 4]
            };
            if got != expected {
                eprintln!("FAIL rotate_{threads}: out[{tid}] = {got}, expected {expected}");
                failures += 1;
            }
        }
    }

    if failures > 0 {
        return Err(format!("{failures} mismatched outputs").into());
    }
    println!("SUCCESS: rotate_128 and rotate_256 match the host reference");
    Ok(())
}
