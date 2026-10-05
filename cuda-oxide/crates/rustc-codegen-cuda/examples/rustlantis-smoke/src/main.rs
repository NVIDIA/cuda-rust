/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![feature(custom_mir, core_intrinsics)]
#![allow(internal_features)]
#![allow(unused_assignments, unused_parens, overflowing_literals)]
// rustlantis emits fuzz-shaped Rust whose style trips clippy: similar
// variable names, `()` arguments, `transmute<isize, isize>`, `a ^ a`,
// and short variable names like `_0` / `_1` are all part of the test
// corpus, not bugs to clean up.
#![allow(
    clippy::similar_names,
    clippy::unit_arg,
    clippy::useless_transmute,
    clippy::eq_op,
    clippy::just_underscores_and_digits
)]

//! rustlantis smoke test (Stage 1b + Stage 2a).
//!
//! Deterministic CPU-vs-GPU comparison for:
//! 1. a hand-written custom MIR function, and
//! 2. an auto-generated rustlantis custom MIR function.
//!
//! The reusable trace machinery (`trace_reset`, `trace_finish`, `dump_var`, the
//! `TraceValue`/`TraceDump` traits, and the `RL_TRACE` global) lives in
//! `crates/fuzzer`. This example only contains the two MIR bodies, the host
//! oracle, and the kernel wiring. See `crates/fuzzer` for the
//! differential-testing infrastructure that drives this smoke example.
//!
//! Build and run with:
//!   cargo oxide run rustlantis-smoke

use core::intrinsics::mir::*;
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use fuzzer::{dump_var, trace_finish, trace_reset};

mod generated_case;

// =============================================================================
// Stage 1b: hand-written #[custom_mir] function
// =============================================================================
//
// This is the minimal local proof that a custom MIR arithmetic/bitwise body can
// compile and run through both backends.

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn fn0(a: i32) -> (i32, i32, i32, i32, i32) {
    mir!(
        let b: i32;
        let c: i32;
        let d: i32;
        let e: i32;
        let eu: u32;
        let lo: u32;
        let hi: u32;
        let f_u: u32;
        let f: i32;

        {
            b = a + 7_i32;
            c = b * 2_i32;
            d = c - 3_i32;
            e = d ^ 1515870810_i32;
            eu = e as u32;
            lo = eu << 11_u32;
            hi = eu >> 21_u32;
            f_u = lo | hi;
            f = f_u as i32;
            RET = (b, c, d, e, f);
            Return()
        }
    )
}

#[inline(never)]
fn compute_stage1_trace(seed: i32) -> u64 {
    trace_reset();
    let (b, c, d, e, f) = fn0(seed);
    dump_var((b, c, d, e, f));
    trace_finish()
}

// =============================================================================
// GPU kernel
// =============================================================================
//
// Launched as <<<1, 1>>> from the host. Writes one hash per stage.
//
// Loaded through `#[cuda_module]` rather than by reading a `.ptx` off disk. A
// generated case that calls into libdevice (any `f64` maths, for one) makes the
// pipeline emit NVVM IR and leave the PTX to the consumer, so there is no `.ptx`
// to read and every such seed died on module load instead of being compared.

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn rl_smoke(mut stage1_out: DisjointSlice<u64>, mut stage2_out: DisjointSlice<u64>) {
        if let Some(slot) = stage1_out.get_mut(thread::index_1d()) {
            *slot = compute_stage1_trace(10);
        }

        if let Some(slot) = stage2_out.get_mut(thread::index_1d()) {
            *slot = generated_case::compute_rustlantis_trace();
        }
    }
}

// =============================================================================
// Host driver
// =============================================================================

fn main() {
    println!("=== rustlantis smoke test (Stage 1b + Stage 2a) ===\n");

    let cpu_stage1 = compute_stage1_trace(core::hint::black_box(10));
    let cpu_stage2 = generated_case::compute_rustlantis_trace();
    println!("Stage 1b CPU hash: 0x{:016x}  ({})", cpu_stage1, cpu_stage1);
    println!("Stage 2a CPU hash: 0x{:016x}  ({})", cpu_stage2, cpu_stage2);

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let mut stage1_out = DeviceBuffer::<u64>::zeroed(&stream, 1).expect("alloc stage1_out");
    let mut stage2_out = DeviceBuffer::<u64>::zeroed(&stream, 1).expect("alloc stage2_out");
    let module = kernels::load(&ctx).expect("Failed to load device module");

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: the two arguments match `rl_smoke`'s two slice parameters, and
    // both are live DeviceBuffers allocated above.
    unsafe { module.rl_smoke(&stream, cfg, &mut stage1_out, &mut stage2_out) }
        .expect("Kernel launch failed");

    let gpu_stage1 = stage1_out.to_host_vec(&stream).expect("readback stage1")[0];
    let gpu_stage2 = stage2_out.to_host_vec(&stream).expect("readback stage2")[0];
    println!("Stage 1b GPU hash: 0x{:016x}  ({})", gpu_stage1, gpu_stage1);
    println!("Stage 2a GPU hash: 0x{:016x}  ({})", gpu_stage2, gpu_stage2);

    if cpu_stage1 == gpu_stage1 && cpu_stage2 == gpu_stage2 {
        println!("\nPASS: CPU/GPU traces match");
    } else {
        println!("\nMISMATCH — INVESTIGATE");
        std::process::exit(1);
    }
}
