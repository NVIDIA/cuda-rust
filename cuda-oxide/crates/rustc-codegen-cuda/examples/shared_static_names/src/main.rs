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
//! A third kernel, `field_tiles`, takes constant pointers to the
//! `SharedArray` fields of two tuple statics. The fields are zero-sized, so
//! each one sits at offset zero of its static, yet each must still get its
//! own global. A field's global is named after the static and the field's
//! position, found from the pointee type, so it is just as stable:
//!
//! ```text
//!   __shared_mem_<crate id>_<hash of the static's mangled symbol and field path>
//! ```
//!
//! The host checks all three outputs on the GPU. The PTX contract itself is checked
//! by `scripts/check-shared-static-names.sh`, which builds copies of this
//! crate with extra kernels spliced in at one of the two `ADDED KERNEL SLOT`
//! comments, in release and full-debug builds, and requires the PTX of
//! all three kernels and their globals' declarations to stay the same.
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

/// Values `field_tiles` writes into its four `SharedArray` fields.
///
/// Distinct powers of two, so their sum tells which fields shared storage:
/// a merged pair keeps only the later write and loses the other bit.
const FIELD_VALUES: [u32; 4] = [1, 2, 4, 8];

/// `field_tiles`' output when all four fields are separate allocations.
const FIELD_SUM: u32 = 15;

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

    /// Writes `FIELD_SUM` to `out[0]` when every `SharedArray` field of the
    /// two tuple statics is its own allocation.
    ///
    /// A `SharedArray` is zero-sized, so every field of `SAME` and `MIXED`
    /// sits at byte offset zero of its static. Naming a pointer by the static
    /// it points into would then fold both fields into one global: `SAME`'s
    /// fields have one physical layout and would silently share storage,
    /// while `MIXED`'s would declare one global twice with different sizes.
    /// The fields of each static differ in type, which is what tells them
    /// apart; two fields of one type are rejected at compile time. `SAME.0`
    /// is read back through a `*const` pointer, which must name the same
    /// global as the `*mut` pointer that wrote it.
    /// Launch with one block of 64 threads; `out` must hold one element.
    #[kernel]
    pub fn field_tiles(mut out: DisjointSlice<u32>) {
        static mut SAME: (SharedArray<u32, 64>, SharedArray<u32, 64, 4>) =
            (SharedArray::UNINIT, SharedArray::UNINIT);
        static mut MIXED: (SharedArray<u32, 64>, SharedArray<u32, 32>) =
            (SharedArray::UNINIT, SharedArray::UNINIT);

        // constant field pointers carry the static's provenance at offset
        // zero, which is what reaches the shared-memory translation
        const SAME_A: *mut SharedArray<u32, 64> = unsafe { &raw mut SAME.0 };
        const SAME_B: *mut SharedArray<u32, 64, 4> = unsafe { &raw mut SAME.1 };
        const MIXED_A: *mut SharedArray<u32, 64> = unsafe { &raw mut MIXED.0 };
        const MIXED_B: *mut SharedArray<u32, 32> = unsafe { &raw mut MIXED.1 };
        // a `*const` to a field already written through a `*mut` must reach
        // the same storage, since a static has one address
        const SAME_A_READ: *const SharedArray<u32, 64> = unsafe { &raw const SAME.0 };

        let same_a = unsafe { SharedArray::as_raw_mut_ptr(SAME_A) };
        let same_b = unsafe { SharedArray::as_raw_mut_ptr(SAME_B) };
        let mixed_a = unsafe { SharedArray::as_raw_mut_ptr(MIXED_A) };
        let mixed_b = unsafe { SharedArray::as_raw_mut_ptr(MIXED_B) };
        let tid = thread::threadIdx_x() as usize;
        unsafe {
            *same_a.add(tid & 63) = FIELD_VALUES[0];
            *same_b.add(tid & 63) = FIELD_VALUES[1];
            *mixed_a.add(tid & 63) = FIELD_VALUES[2];
        }
        // `mixed_b` has 32 slots for 64 threads; one writer per slot keeps
        // the stores race-free, and every thread still reaches the barrier
        if tid < 32 {
            unsafe { *mixed_b.add(tid) = FIELD_VALUES[3] };
        }
        thread::sync_threads();
        if tid == 0 {
            unsafe {
                let read_a = *(*SAME_A_READ).as_ptr();
                *out.get_unchecked_mut(0) = read_a + *same_b + *mixed_a + *mixed_b;
            }
        }
    }

    // ADDED KERNEL SLOT (last)
}

/// Launches `rotate_128`, `rotate_256` and `field_tiles` and checks their
/// outputs on the host.
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

    let field_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut field_out = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
    // SAFETY: one block of 64 threads, matching the 64-entry tiles; only
    // thread 0 writes, and `field_out` holds its one element
    unsafe { module.field_tiles(stream.as_ref(), field_cfg, &mut field_out) }?;
    let field_sum = field_out.to_host_vec(&stream)?[0];
    if field_sum != FIELD_SUM {
        eprintln!("FAIL field_tiles: got {field_sum}, expected {FIELD_SUM}");
        failures += 1;
    }

    if failures > 0 {
        return Err(format!("{failures} mismatched outputs").into());
    }
    println!("SUCCESS: rotate_128, rotate_256 and field_tiles match the host reference");
    Ok(())
}
