/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Field access on reordered/padded repr(Rust) structs.
//!
//! `Arena` declares fields (layout, cap, stride, big). rustc reorders them
//! in memory, and the enum field `Layout` occupies 8 bytes in rustc's
//! layout but only 5 bytes in its lowered `{ i8, i32 }` form, so the
//! padded LLVM struct gains an interior `[3 x i8]` slot:
//!
//! ```text
//! rustc:  layout@0 (8 bytes), big@8, cap@16, stride@20, size 24
//! LLVM:   { { i8, i32 }, [3 x i8], i64, i32, i32 }
//!           layout=0     pad=1     big=2 cap=3 stride=4
//! ```
//!
//! Every aggregate site in mir-lower shares the type converter's slot map
//! (declaration index -> LLVM slot, padding-aware, ZST-aware), so field
//! reads must land on the right slot regardless of reorder and padding.
//! Guards the fix for issue #128: each lane must compute
//! `cap * 1000 + stride + big` from the right fields; the host verifies
//! the value and exits 1 on any mismatch.
//!
//! Run: `cargo oxide run struct_field_layout`

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Multi-word, align-4 enum: the payload variant forces `{ i8, i32 }`.
#[derive(Copy, Clone)]
pub enum Layout {
    Aos,
    Soa,
    AoSoA(u32),
}

/// Declaration order: layout, cap, stride, big.
/// rustc memory order: layout first (8 bytes incl. enum tail padding),
/// then big (align 8), then cap, stride.
pub struct Arena {
    layout: Layout,
    cap: u32,
    stride: u32,
    big: u64,
}

/// Keep the receiver a pointer-to-struct so field reads go through the
/// field-address (GEP) and extract paths instead of being SROA'd away.
#[inline(never)]
fn pick(a: &Arena) -> u32 {
    match a.layout {
        Layout::Soa => a
            .cap
            .wrapping_mul(1000)
            .wrapping_add(a.stride)
            .wrapping_add(a.big as u32),
        Layout::Aos => a.stride,
        Layout::AoSoA(w) => w,
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn fill(params: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(slot) = out.get_mut(idx) {
            let arena = Arena {
                layout: Layout::Soa,
                cap: params[0],
                stride: params[1],
                big: 7,
            };
            // Expected: params[0] * 1000 + params[1] + 7.
            *slot = pick(&arena);
        }
    }
}

fn main() {
    println!("=== struct_field_layout (issue #128) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 256;
    let params_host: Vec<u32> = vec![3, 41];
    let params_dev = DeviceBuffer::from_host(&stream, &params_host).unwrap();
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.fill(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &params_dev,
            &mut out_dev,
        )
    }
    .expect("Kernel launch failed");

    let out_host = out_dev.to_host_vec(&stream).unwrap();

    // Layout::Soa arm: cap * 1000 + stride + big = 3 * 1000 + 41 + 7.
    // A wrong-field read produces a visibly different value (e.g. swapped
    // cap/stride gives 41 * 1000 + 3 + 7; reading the pad gives garbage).
    let expected = params_host[0]
        .wrapping_mul(1000)
        .wrapping_add(params_host[1])
        .wrapping_add(7);

    let mut errors = 0;
    for (i, &got) in out_host.iter().enumerate() {
        if got != expected {
            if errors < 5 {
                eprintln!("  Error at [{i}]: expected {expected}, got {got}");
            }
            errors += 1;
        }
    }

    if errors == 0 {
        println!("✓ SUCCESS: all {N} lanes read the right fields (value {expected})");
    } else {
        println!("✗ FAILED: {errors} of {N} lanes read the wrong field");
        std::process::exit(1);
    }
}
