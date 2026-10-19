/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression for issue #131: `match xs[i]` over an array of enums.
//!
//! The place for a match payload binding like `E::A(x)` on `xs[i]` carries
//! the projection chain `[Index(i), Downcast(variant), Field(0)]`. The
//! value-producing place walker in `mir-importer`
//! (`translate_place_iterative`) used to advance its running Rust type only
//! on `Field` projections, so `Downcast` saw the stale outer Array type and
//! bailed with "Downcast on non-ADT type: Array". The same staleness
//! affected `ConstantIndex` (literal `xs[0]`) and `Deref` (`(*p)[i]`).
//!
//! The fix folds every projection element through rustc_public's own
//! `ProjectionElem::ty`, so the three kernels below pin all three chains:
//!
//! | Kernel              | Projection chain                        |
//! |---------------------|-----------------------------------------|
//! | match_runtime_index | [Index, Downcast, Field]                |
//! | match_const_index   | [ConstantIndex, Downcast, Field]        |
//! | match_deref_index   | [Deref, Index, Downcast, Field]         |
//!
//! The enum has both payload (`A(u32)`, `B(u32)`) and fieldless (`C`)
//! variants, so the match exercises `MirEnumPayloadOp` extraction as well
//! as the bare discriminant switch.
//!
//! Note: a fully-constant `match xs[0]` over a constant array is folded
//! away by rustc's MIR optimizations before it reaches the importer, so the
//! constant-index kernel derives the array contents from a kernel parameter
//! to keep the projection chain alive.
//!
//! Run with:
//!   cargo oxide run enum_array_match

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, device, kernel, thread};
use cuda_host::cuda_module;

#[derive(Clone, Copy)]
pub enum E {
    A(u32),
    B(u32),
    C,
}

/// Deref-then-index: the place is `(*p)[i]` with projection chain
/// [Deref, Index, Downcast, Field]. The reference parameter is opaque to
/// MIR optimizations (ReferencePropagation cannot see through it), so the
/// Deref projection reaches the importer intact; an inline `let p = &xs;`
/// would be propagated away before translation.
#[device]
pub fn pick_payload(p: &[E; 4], i: usize) -> u32 {
    match (*p)[i] {
        E::A(x) => x,
        E::B(y) => y + 1000,
        E::C => 9999,
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Runtime index: projection chain [Index(local), Downcast, Field].
    #[kernel]
    pub fn match_runtime_index(index: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let xs: [E; 4] = [E::A(7), E::B(8), E::C, E::A(100)];
            *out_elem = match xs[index as usize] {
                E::A(x) => x,
                E::B(y) => y + 1000,
                E::C => 9999,
            };
        }
    }

    /// Literal index with a runtime-unknown discriminant: exercises the
    /// [ConstantIndex, Downcast, Field] flavor of the same walk.
    #[kernel]
    pub fn match_const_index(val: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let xs: [E; 2] = if val > 5 {
                [E::A(val), E::C]
            } else {
                [E::B(val), E::C]
            };
            *out_elem = match xs[0] {
                E::A(x) => x,
                E::B(y) => y + 1000,
                E::C => 9999,
            };
        }
    }

    /// Deref-then-index via the #[device] helper: pins the Deref arm of
    /// the projection-type fold.
    #[kernel]
    pub fn match_deref_index(index: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let xs: [E; 4] = [E::A(7), E::B(8), E::C, E::A(100)];
            *out_elem = pick_payload(&xs, index as usize);
        }
    }
}

/// Host-side mirror of the device-side match over
/// `[E::A(7), E::B(8), E::C, E::A(100)]`.
fn expected_for_index(index: u32) -> u32 {
    match index {
        0 => 7,        // A(7)   -> payload
        1 => 8 + 1000, // B(8)   -> payload + 1000
        2 => 9999,     // C      -> fieldless arm
        _ => 100,      // A(100) -> payload
    }
}

/// Host-side mirror of `match_const_index`.
fn expected_for_val(val: u32) -> u32 {
    if val > 5 { val } else { val + 1000 }
}

fn main() {
    println!("=== enum_array_match regression (issue #131) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let stream = ctx.default_stream();

    const BLOCK: u32 = 32;
    const N: usize = BLOCK as usize;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut failures = 0usize;

    // Runtime index + deref-then-index: sweep every element of the array,
    // covering both payload variants and the fieldless variant.
    for index in 0u32..4 {
        let want = expected_for_index(index);

        let mut d_out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.match_runtime_index(stream.as_ref(), cfg, index, &mut d_out) }
            .expect("launch match_runtime_index");
        let got = d_out.to_host_vec(&stream).unwrap();
        for (tid, &g) in got.iter().enumerate() {
            if g != want {
                println!("FAIL match_runtime_index index={index} tid={tid}: got {g}, want {want}");
                failures += 1;
            }
        }

        let mut d_out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.match_deref_index(stream.as_ref(), cfg, index, &mut d_out) }
            .expect("launch match_deref_index");
        let got = d_out.to_host_vec(&stream).unwrap();
        for (tid, &g) in got.iter().enumerate() {
            if g != want {
                println!("FAIL match_deref_index index={index} tid={tid}: got {g}, want {want}");
                failures += 1;
            }
        }
    }

    // Constant index: both sides of the discriminant-selecting branch.
    for val in [3u32, 9u32] {
        let want = expected_for_val(val);

        let mut d_out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.match_const_index(stream.as_ref(), cfg, val, &mut d_out) }
            .expect("launch match_const_index");
        let got = d_out.to_host_vec(&stream).unwrap();
        for (tid, &g) in got.iter().enumerate() {
            if g != want {
                println!("FAIL match_const_index val={val} tid={tid}: got {g}, want {want}");
                failures += 1;
            }
        }
    }

    if failures == 0 {
        println!(
            "enum_array_match: PASS ({N} threads; runtime/const/deref index chains all match)"
        );
    } else {
        println!("enum_array_match: FAIL ({failures} mismatches)");
        std::process::exit(1);
    }
}
