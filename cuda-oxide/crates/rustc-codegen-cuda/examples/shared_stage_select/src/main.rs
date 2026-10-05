/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Runtime stage selection over per-stage shared-memory buffers.
//!
//! This is the multi-stage pipeline pattern the tcgen05 GEMM examples use:
//! a `match stage & 3 { ... }` returning raw pointers into one of four
//! shared-memory buffers per arm, inside a loop whose trip count is only
//! known at runtime.
//!
//! Regression guard: LLVM 23's SimplifyCFG (default NVPTX subtarget bumped
//! from sm_30 to sm_75, legalizing `brx.idx` and enabling
//! switch-to-lookup-table) converts exactly this shape into `.global` data
//! arrays of shared-memory addresses. ptxas rejects `.shared` symbols in
//! `.global`/`.const` initializers in both the bare and `generic()` forms
//! ("Variable used as initial value not in .global or .const state space"),
//! and driver JIT fails module load with CUDA_ERROR_INVALID_PTX. cuda-oxide
//! passes `-switch-to-lookup=false` to `opt` and scans the produced PTX for
//! shared symbols in data-space initializers; if either guard regresses,
//! this example fails at build time (PTX scan) and at run time (module
//! load), on every GPU, not just tcgen05 hardware.
//!
//! Run: `cargo oxide run shared_stage_select`

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;

const LANES: usize = 32;

#[cuda_module]
mod kernels {
    use super::*;

    /// Each iteration selects the current stage's A/B buffers by matching on
    /// `i & 3`, writes a lane-unique value through both raw pointers, syncs,
    /// then reads the neighbor lane's values back into an accumulator.
    #[kernel]
    pub fn stage_select_roundtrip(iters: u32, mut out: DisjointSlice<u32>) {
        static mut STAGE_A0: SharedArray<u32, LANES> = SharedArray::UNINIT;
        static mut STAGE_A1: SharedArray<u32, LANES> = SharedArray::UNINIT;
        static mut STAGE_A2: SharedArray<u32, LANES> = SharedArray::UNINIT;
        static mut STAGE_A3: SharedArray<u32, LANES> = SharedArray::UNINIT;
        static mut STAGE_B0: SharedArray<u32, LANES> = SharedArray::UNINIT;
        static mut STAGE_B1: SharedArray<u32, LANES> = SharedArray::UNINIT;
        static mut STAGE_B2: SharedArray<u32, LANES> = SharedArray::UNINIT;
        static mut STAGE_B3: SharedArray<u32, LANES> = SharedArray::UNINIT;

        let lane = thread::threadIdx_x() as usize;
        let next = (lane + 1) % LANES;
        let mut acc: u32 = 0;
        let mut i: u32 = 0;
        while i < iters {
            let stage = i & 3;
            // The per-arm pointer pairs are compile-time shared addresses;
            // the merged values are what SimplifyCFG would turn into a
            // lookup table of `.shared` symbols.
            let (a, b): (*mut u32, *mut u32) = match stage {
                0 => (
                    unsafe { SharedArray::as_raw_mut_ptr(&raw mut STAGE_A0) },
                    unsafe { SharedArray::as_raw_mut_ptr(&raw mut STAGE_B0) },
                ),
                1 => (
                    unsafe { SharedArray::as_raw_mut_ptr(&raw mut STAGE_A1) },
                    unsafe { SharedArray::as_raw_mut_ptr(&raw mut STAGE_B1) },
                ),
                2 => (
                    unsafe { SharedArray::as_raw_mut_ptr(&raw mut STAGE_A2) },
                    unsafe { SharedArray::as_raw_mut_ptr(&raw mut STAGE_B2) },
                ),
                _ => (
                    unsafe { SharedArray::as_raw_mut_ptr(&raw mut STAGE_A3) },
                    unsafe { SharedArray::as_raw_mut_ptr(&raw mut STAGE_B3) },
                ),
            };

            // SAFETY: each thread writes only its own lane, reads the
            // neighbor lane strictly after a block-wide barrier, and the
            // second barrier orders the reads before the next iteration's
            // writes to the same stage buffers.
            unsafe {
                a.add(lane).write(i * 100 + lane as u32);
                b.add(lane).write(i + lane as u32);
            }
            thread::sync_threads();
            unsafe {
                acc = acc
                    .wrapping_add(a.add(next).read())
                    .wrapping_add(b.add(next).read());
            }
            thread::sync_threads();
            i += 1;
        }

        let gid = thread::index_1d();
        if let Some(slot) = out.get_mut(gid) {
            *slot = acc;
        }
    }
}

fn main() {
    println!("=== shared_stage_select (stage-switch over shared buffers) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    // If SimplifyCFG lookup tables of shared addresses ever come back, this
    // load fails with CUDA_ERROR_INVALID_PTX before any kernel runs.
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (LANES as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let iters: u32 = 9; // not a multiple of 4, so every stage count differs
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, LANES).expect("alloc");

    // SAFETY: one 32-thread block; the kernel's shared accesses are barrier
    // ordered and `out` has one element per thread.
    unsafe { module.stage_select_roundtrip(stream.as_ref(), cfg, iters, &mut out) }
        .expect("Kernel launch failed");
    stream.synchronize().expect("sync");

    let result = out.to_host_vec(&stream).expect("copy back");
    let mut failures = 0;
    for (lane, &got) in result.iter().enumerate().take(LANES) {
        let next = ((lane + 1) % LANES) as u32;
        let mut expected: u32 = 0;
        for i in 0..iters {
            expected = expected.wrapping_add(i * 100 + next).wrapping_add(i + next);
        }
        if got != expected {
            eprintln!("✗ lane {lane}: got {got}, expected {expected}");
            failures += 1;
        }
    }
    if failures > 0 {
        eprintln!("✗ {failures} lanes mismatched");
        std::process::exit(1);
    }
    println!("✓ all {LANES} lanes correct across {iters} staged iterations");
    println!("\n✓ SUCCESS: shared_stage_select passed!");
}
