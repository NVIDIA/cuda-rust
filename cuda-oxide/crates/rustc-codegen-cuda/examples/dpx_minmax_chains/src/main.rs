/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression coverage for the s32/u32 DPX items of issue #338 Phase 4.
//!
//! DPX (SASS `VIMNMX`) has no dedicated PTX instructions: ptxas fuses
//! dataflow-adjacent native integer min/max chains. This example pins the
//! PTX shape that fusion needs — min/max instructions, never
//! compare/select — and verifies each chain's semantics on the device.
//! One kernel per issue item; `vimnmx` names the issue's mixed min→max
//! pattern, not a CUDA API function.
//!
//! Run: cargo oxide run dpx_minmax_chains

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// Deterministic per-thread operand, shared with the host reference.
    /// Lane 0 pairs MAX + 1 and lane 2 pairs MIN + -1 (explicit wrapping);
    /// later lanes get a multiplicative hash.
    pub fn input_s32(t: u32, operand: u32) -> i32 {
        const EDGES: [i32; 6] = [i32::MAX, 1, i32::MIN, -1, 0, 0x5555_5555];
        if t < 6 {
            EDGES[((t + operand) % 6) as usize]
        } else {
            let v = t
                .wrapping_add(operand.wrapping_mul(0x85EB_CA6B))
                .wrapping_mul(0x9E37_79B1);
            (v ^ (v >> 15)) as i32
        }
    }

    /// Unsigned twin of `input_s32`; lane 0 adds MAX + 1 (wraps to 0).
    pub fn input_u32(t: u32, operand: u32) -> u32 {
        const EDGES: [u32; 6] = [u32::MAX, 1, 0, 0x8000_0000, 0x7FFF_FFFF, 0xAAAA_AAAA];
        if t < 6 {
            EDGES[((t + operand) % 6) as usize]
        } else {
            let v = t
                .wrapping_add(operand.wrapping_mul(0xC2B2_AE35))
                .wrapping_mul(0x27D4_EB2F);
            v ^ (v >> 13)
        }
    }

    #[kernel]
    pub fn vimax3_s32(mut out: DisjointSlice<i32>) {
        let tid = thread::index_1d();
        let t = tid.get() as u32;
        if let Some(out_elem) = out.get_mut(tid) {
            let (x, y, z) = (input_s32(t, 0), input_s32(t, 1), input_s32(t, 2));
            *out_elem = x.max(y).max(z);
        }
    }

    #[kernel]
    pub fn vimin3_s32(mut out: DisjointSlice<i32>) {
        let tid = thread::index_1d();
        let t = tid.get() as u32;
        if let Some(out_elem) = out.get_mut(tid) {
            let (x, y, z) = (input_s32(t, 0), input_s32(t, 1), input_s32(t, 2));
            *out_elem = x.min(y).min(z);
        }
    }

    #[kernel]
    pub fn vimnmx_s32(mut out: DisjointSlice<i32>) {
        let tid = thread::index_1d();
        let t = tid.get() as u32;
        if let Some(out_elem) = out.get_mut(tid) {
            let (x, y, z) = (input_s32(t, 0), input_s32(t, 1), input_s32(t, 2));
            *out_elem = x.min(y).max(z);
        }
    }

    #[kernel]
    pub fn viaddmax_s32(mut out: DisjointSlice<i32>) {
        let tid = thread::index_1d();
        let t = tid.get() as u32;
        if let Some(out_elem) = out.get_mut(tid) {
            let (x, y, z) = (input_s32(t, 0), input_s32(t, 1), input_s32(t, 2));
            *out_elem = x.wrapping_add(y).max(z);
        }
    }

    #[kernel]
    pub fn viaddmin_s32(mut out: DisjointSlice<i32>) {
        let tid = thread::index_1d();
        let t = tid.get() as u32;
        if let Some(out_elem) = out.get_mut(tid) {
            let (x, y, z) = (input_s32(t, 0), input_s32(t, 1), input_s32(t, 2));
            *out_elem = x.wrapping_add(y).min(z);
        }
    }

    #[kernel]
    pub fn vimax3_u32(mut out: DisjointSlice<u32>) {
        let tid = thread::index_1d();
        let t = tid.get() as u32;
        if let Some(out_elem) = out.get_mut(tid) {
            let (x, y, z) = (input_u32(t, 0), input_u32(t, 1), input_u32(t, 2));
            *out_elem = x.max(y).max(z);
        }
    }

    #[kernel]
    pub fn vimin3_u32(mut out: DisjointSlice<u32>) {
        let tid = thread::index_1d();
        let t = tid.get() as u32;
        if let Some(out_elem) = out.get_mut(tid) {
            let (x, y, z) = (input_u32(t, 0), input_u32(t, 1), input_u32(t, 2));
            *out_elem = x.min(y).min(z);
        }
    }

    #[kernel]
    pub fn vimnmx_u32(mut out: DisjointSlice<u32>) {
        let tid = thread::index_1d();
        let t = tid.get() as u32;
        if let Some(out_elem) = out.get_mut(tid) {
            let (x, y, z) = (input_u32(t, 0), input_u32(t, 1), input_u32(t, 2));
            *out_elem = x.min(y).max(z);
        }
    }

    #[kernel]
    pub fn viaddmax_u32(mut out: DisjointSlice<u32>) {
        let tid = thread::index_1d();
        let t = tid.get() as u32;
        if let Some(out_elem) = out.get_mut(tid) {
            let (x, y, z) = (input_u32(t, 0), input_u32(t, 1), input_u32(t, 2));
            *out_elem = x.wrapping_add(y).max(z);
        }
    }

    #[kernel]
    pub fn viaddmin_u32(mut out: DisjointSlice<u32>) {
        let tid = thread::index_1d();
        let t = tid.get() as u32;
        if let Some(out_elem) = out.get_mut(tid) {
            let (x, y, z) = (input_u32(t, 0), input_u32(t, 1), input_u32(t, 2));
            *out_elem = x.wrapping_add(y).min(z);
        }
    }
}

const BLOCK: u32 = 256;
const N: usize = BLOCK as usize;

/// Hand-computed single-lane expectation, independent of the shared helpers.
fn pin<T: PartialEq + Copy + std::fmt::Display>(name: &str, got: T, want: T) -> usize {
    if got != want {
        println!("FAIL pin: {name}={got} expected={want}");
        1
    } else {
        0
    }
}

fn check<T: PartialEq + Copy + std::fmt::Display>(
    name: &str,
    got: &[T],
    want: impl Fn(u32) -> T,
) -> usize {
    let mut failures = 0;
    for (tid, &value) in got.iter().enumerate() {
        let expected = want(tid as u32);
        if value != expected {
            println!("FAIL tid={tid}: {name}={value} expected={expected}");
            failures += 1;
        }
    }
    failures
}

fn main() {
    use kernels::{input_s32 as s, input_u32 as u};

    println!("=== dpx_minmax_chains: issue-338 Phase 4 s32/u32 semantics ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let stream = ctx.default_stream();

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    // Every kernel writes one element per thread of a single 256-thread block,
    // matching the 256-element allocations below.
    macro_rules! run {
        ($kernel:ident, $ty:ty) => {{
            let mut out = DeviceBuffer::<$ty>::zeroed(&stream, N).unwrap();
            // SAFETY: the 256-thread 1D block matches the kernel's indexing
            // model and the 256-element output allocation.
            unsafe { module.$kernel(stream.as_ref(), cfg, &mut out) }
                .expect(concat!("launch ", stringify!($kernel)));
            out.to_host_vec(&stream).unwrap()
        }};
    }

    let mut failures = 0;
    failures += check("vimax3_s32", &run!(vimax3_s32, i32), |t| {
        s(t, 0).max(s(t, 1)).max(s(t, 2))
    });
    failures += check("vimin3_s32", &run!(vimin3_s32, i32), |t| {
        s(t, 0).min(s(t, 1)).min(s(t, 2))
    });
    failures += check("vimnmx_s32", &run!(vimnmx_s32, i32), |t| {
        s(t, 0).min(s(t, 1)).max(s(t, 2))
    });
    let addmax_s32 = run!(viaddmax_s32, i32);
    // Lane 0 is (MAX, 1, MIN): MAX + 1 wraps to MIN. Lane 2 is (MIN, -1, 0):
    // MIN - 1 wraps to MAX.
    failures += pin("viaddmax_s32[0] wraps MAX+1", addmax_s32[0], i32::MIN);
    failures += pin("viaddmax_s32[2] wraps MIN-1", addmax_s32[2], i32::MAX);
    failures += check("viaddmax_s32", &addmax_s32, |t| {
        s(t, 0).wrapping_add(s(t, 1)).max(s(t, 2))
    });
    let addmin_s32 = run!(viaddmin_s32, i32);
    failures += pin("viaddmin_s32[2] wraps MIN-1", addmin_s32[2], 0);
    failures += check("viaddmin_s32", &addmin_s32, |t| {
        s(t, 0).wrapping_add(s(t, 1)).min(s(t, 2))
    });
    failures += check("vimax3_u32", &run!(vimax3_u32, u32), |t| {
        u(t, 0).max(u(t, 1)).max(u(t, 2))
    });
    failures += check("vimin3_u32", &run!(vimin3_u32, u32), |t| {
        u(t, 0).min(u(t, 1)).min(u(t, 2))
    });
    failures += check("vimnmx_u32", &run!(vimnmx_u32, u32), |t| {
        u(t, 0).min(u(t, 1)).max(u(t, 2))
    });
    let addmax_u32 = run!(viaddmax_u32, u32);
    // Lane 0 is (MAX, 1, 0): MAX + 1 wraps to 0.
    failures += pin("viaddmax_u32[0] wraps MAX+1", addmax_u32[0], 0);
    failures += check("viaddmax_u32", &addmax_u32, |t| {
        u(t, 0).wrapping_add(u(t, 1)).max(u(t, 2))
    });
    failures += check("viaddmin_u32", &run!(viaddmin_u32, u32), |t| {
        u(t, 0).wrapping_add(u(t, 1)).min(u(t, 2))
    });

    if failures == 0 {
        println!("dpx_minmax_chains: PASS ({N} threads, 10 DPX-shaped chains agree)");
    } else {
        println!("dpx_minmax_chains: FAIL ({failures} mismatches)");
        std::process::exit(1);
    }
}
