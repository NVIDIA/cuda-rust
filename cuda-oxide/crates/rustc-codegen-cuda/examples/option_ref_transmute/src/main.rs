/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Sibling regression for issue #21: pointer-source niched Transmute.
//!
//! `step_by` exercises the *integer-source* form of the cast: rustc stores
//! `Option<NonZeroUsize>` in one `usize` (niche = 0), so cuda-oxide must keep
//! the enum's physical bytes instead of inventing a separate tag.
//!
//! This kernel exercises the *pointer-source* form: rustc stores
//! `Option<&T>` as a single `ptr` (niche = null), and the `Cast(Transmute)`
//! source is a pointer rather than an integer.
//!
//! The historical bug sent this through the generic pointer-to-struct path,
//! which was written for fat-pointer construction (`&[T] = { ptr, i64 }`).
//! Enum Transmute now uses the enum's rustc-derived physical type and an
//! exact-size, aligned memory round-trip, so the pointer remains the niche
//! carrier and no synthetic discriminant is introduced.
//!
//! Run with:
//!   cargo oxide run option_ref_transmute

use core::mem;
use core::ptr;

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// For each tid, build a raw pointer (null when `tid >= input.len()`),
    /// transmute it to `Option<&u32>`, and write `*x as u64` for `Some(x)`
    /// or `u64::MAX` for `None`. The `mem::transmute` is the cast under
    /// test: source is `*const u32`, destination is `Option<&u32>`, which
    /// rustc represents as a niched single-pointer scalar.
    #[kernel]
    pub fn opt_ref_transmute(input: &[u32], mut out: DisjointSlice<u64>) {
        let tid = thread::index_1d();
        let i = tid.get();
        if let Some(out_elem) = out.get_mut(tid) {
            let p: *const u32 = if i < input.len() {
                unsafe { input.as_ptr().add(i) }
            } else {
                ptr::null()
            };
            let opt: Option<&u32> = unsafe { mem::transmute(p) };
            *out_elem = match opt {
                Some(x) => *x as u64,
                None => u64::MAX,
            };
        }
    }

    /// Same shape but constructed without `mem::transmute`: rustc's own
    /// `Some(&...)` / `None` lowering. This is the control: if the bug is
    /// specific to the niched-scalar Transmute path, this kernel still
    /// produces correct values even when `opt_ref_transmute` does not.
    #[kernel]
    pub fn opt_ref_control(input: &[u32], mut out: DisjointSlice<u64>) {
        let tid = thread::index_1d();
        let i = tid.get();
        if let Some(out_elem) = out.get_mut(tid) {
            let opt: Option<&u32> = if i < input.len() {
                Some(unsafe { &*input.as_ptr().add(i) })
            } else {
                None
            };
            *out_elem = match opt {
                Some(x) => *x as u64,
                None => u64::MAX,
            };
        }
    }
}

fn expected(i: usize, input: &[u32]) -> u64 {
    if i < input.len() {
        input[i] as u64
    } else {
        u64::MAX
    }
}

fn main() {
    println!("=== option_ref_transmute regression (issue #21 sibling) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let stream = ctx.default_stream();

    const BLOCK: u32 = 32;
    const N: usize = BLOCK as usize;
    const INPUT_LEN: usize = 20;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let host_input: Vec<u32> = (0..INPUT_LEN as u32).map(|x| x * 7 + 1).collect();
    let d_input = DeviceBuffer::from_host(&stream, &host_input).unwrap();

    let mut d_xmute = DeviceBuffer::<u64>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.opt_ref_transmute(stream.as_ref(), cfg, &d_input, &mut d_xmute) }
        .expect("launch opt_ref_transmute");
    let got_xmute = d_xmute.to_host_vec(&stream).unwrap();

    let mut d_ctrl = DeviceBuffer::<u64>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.opt_ref_control(stream.as_ref(), cfg, &d_input, &mut d_ctrl) }
        .expect("launch opt_ref_control");
    let got_ctrl = d_ctrl.to_host_vec(&stream).unwrap();

    let mut failures = 0usize;
    for tid in 0..N {
        let want = expected(tid, &host_input);
        let x = got_xmute[tid];
        let c = got_ctrl[tid];
        if x != want || c != want || x != c {
            println!("FAIL tid={tid}: transmute={x} control={c} expected={want}");
            failures += 1;
        }
    }

    if failures == 0 {
        println!(
            "option_ref_transmute: PASS ({N} threads, transmute and Some/None paths match expected, input_len={INPUT_LEN})"
        );
    } else {
        println!("option_ref_transmute: FAIL ({failures} mismatches)");
        std::process::exit(1);
    }
}
