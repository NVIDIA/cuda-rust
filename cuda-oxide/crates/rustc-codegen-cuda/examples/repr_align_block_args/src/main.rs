/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `repr(align(N))` recovery for values in block-arg positions.
//!
//! The alignment lives on MIR aggregate types and is not expressible on
//! the converted LLVM struct types. The dialect conversion driver may
//! convert block-argument types before any per-op rewrite runs (pliron
//! PR #182 made this eager), so mir-lower recovers the alignment of
//! stored/referenced block args from the `OperandsInfo` conversion
//! history at each op's own rewrite. These kernels place over-aligned
//! aggregates in block-arg positions (function params, mem2reg join and
//! loop-header args) so any recovery loss shows up in the emitted `.ll`
//! as missing `align 16` annotations. Runtime results are also checked,
//! but the sharp assertion is the artifact diff between pliron revisions.
//!
//! Run: `cargo oxide run repr_align_block_args`

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Over-aligned struct: abi_align = 16 on the MirStructType.
#[repr(align(16))]
#[derive(Clone, Copy)]
pub struct Al16 {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
}

/// `p` arrives as an entry block arg of MirStructType(align 16). Taking
/// `&p` keeps the param local in memory, so the incoming block arg is
/// stored to a param alloca: a `mir.store` whose operand(1) is a block
/// arg. That store (and the alloca) must carry `align 16`.
#[inline(never)]
fn consume(p: Al16) -> f32 {
    let r = &p;
    r.a + p.d
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Join-block struct: `s` is never address-taken, so mem2reg promotes
    /// it to a block arg of the if/else join block. Passing it to
    /// `consume` moves the block arg into a call operand and then into
    /// the callee's entry spill.
    #[kernel]
    pub fn align_probe(params: &[f32], mut out: DisjointSlice<f32>) {
        let i = thread::index_1d();
        if let Some(slot) = out.get_mut(i) {
            let s = if params[0] > 0.0 {
                Al16 {
                    a: params[0],
                    b: params[1],
                    c: 1.0,
                    d: 2.0,
                }
            } else {
                Al16 {
                    a: params[1],
                    b: params[0],
                    c: 3.0,
                    d: 4.0,
                }
            };
            *slot = consume(s) + s.b;
        }
    }

    /// Dead params: `_dead_slice` reconstructs a (ptr, len) pair that is
    /// never read; `_dead_n` is an unused scalar. Under the old driver
    /// these entry args were converted only via the entry BrOp trick;
    /// under PR #182 the new up-front phase is the only converter.
    #[kernel]
    pub fn unused_params(
        a: &[f32],
        _dead_slice: &[f32],
        _dead_n: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d();
        let i_raw = i.get();
        if let Some(slot) = out.get_mut(i)
            && i_raw < a.len()
        {
            *slot = a[i_raw] * 2.0;
        }
    }

    /// Loop-carried tuple: mem2reg turns `acc` into a MirTupleType
    /// loop-header block arg; the old driver converted it when the latch
    /// terminator was dequeued, the new driver converts it up front.
    #[kernel]
    pub fn loop_carried(mut out: DisjointSlice<f32>, n: u32) {
        let i = thread::index_1d();
        if let Some(slot) = out.get_mut(i) {
            let mut acc = (0.0f32, 1.0f32);
            let mut k = 0u32;
            while k < n {
                acc = (acc.0 + k as f32, acc.1 * 0.5);
                k += 1;
            }
            *slot = acc.0 + acc.1;
        }
    }
}

fn main() {
    println!("=== repr_align_block_args (alignment recovery for block-arg values) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 256;
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let cfg = LaunchConfig::for_num_elems(N as u32);
    let mut errors = 0usize;

    // align_probe: s = {3, 41, 1, 2} -> consume = 3 + 2 = 5, + b = 46.
    let params = DeviceBuffer::from_host(&stream, &[3.0f32, 41.0]).unwrap();
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.align_probe(&stream, cfg, &params, &mut out) }.expect("launch align_probe");
    for (i, &got) in out.to_host_vec(&stream).unwrap().iter().enumerate() {
        if got != 46.0 {
            if errors < 5 {
                eprintln!("  align_probe[{i}]: expected 46.0, got {got}");
            }
            errors += 1;
        }
    }

    // unused_params: out = a * 2.
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let a_dev = DeviceBuffer::from_host(&stream, &a_host).unwrap();
    let dead = DeviceBuffer::from_host(&stream, &[0.0f32; 4]).unwrap();
    let mut out2 = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.unused_params(&stream, cfg, &a_dev, &dead, 7u32, &mut out2) }
        .expect("launch unused_params");
    for (i, &got) in out2.to_host_vec(&stream).unwrap().iter().enumerate() {
        let expected = a_host[i] * 2.0;
        if got != expected {
            if errors < 5 {
                eprintln!("  unused_params[{i}]: expected {expected}, got {got}");
            }
            errors += 1;
        }
    }

    // loop_carried: n = 4 -> acc.0 = 0+1+2+3 = 6, acc.1 = 0.0625 -> 6.0625.
    let mut out3 = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.loop_carried(&stream, cfg, &mut out3, 4u32) }.expect("launch loop_carried");
    for (i, &got) in out3.to_host_vec(&stream).unwrap().iter().enumerate() {
        if got != 6.0625 {
            if errors < 5 {
                eprintln!("  loop_carried[{i}]: expected 6.0625, got {got}");
            }
            errors += 1;
        }
    }

    if errors == 0 {
        println!("✓ SUCCESS: all probes produced expected values");
    } else {
        println!("✗ FAILED: {errors} mismatches");
        std::process::exit(1);
    }
}
