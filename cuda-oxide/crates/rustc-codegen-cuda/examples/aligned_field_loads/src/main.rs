/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Adjacent field reads of an over-aligned element fuse into one wide load.
//!
//! A load takes its alignment from its own result type, and a scalar records
//! none, so `p.x` used to export with LLVM's default `align 4` even when the
//! element type promised `align(8)`. LoadStoreVectorizer will not widen past an
//! alignment it cannot prove, so the pair stayed two `ld.global.b32`.
//!
//! | kernel                      | element             | expected |
//! |-----------------------------|---------------------|----------|
//! | `packed_pair`               | `repr(C)`, align 4  | 2 loads -- align 4 is all rustc guarantees |
//! | `aligned_pair`              | `repr(C, align(8))` | **1 wide load** |
//! | `hot_packed` / `hot_aligned`| same, cache-resident| the timed comparison |
//!
//! `packed_pair` is the control that must NOT change: nothing proves 8-byte
//! alignment there, and widening it would be wrong.
//!
//! Streaming access is bandwidth-bound, so fusing loads does not speed it up --
//! two loads from consecutive addresses already coalesce into the same
//! transactions. The win is in the cache-resident kernels, where load count is
//! the limit rather than bandwidth.
//!
//! Run: `cargo oxide run aligned_field_loads`

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Two f32, natural alignment 4.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PackedPair {
    pub x: f32,
    pub y: f32,
}

/// The same payload, promising 8-byte alignment.
#[repr(C, align(8))]
#[derive(Clone, Copy)]
pub struct AlignedPair {
    pub x: f32,
    pub y: f32,
}

/// The same payload and alignment, but the two lanes live in an array rather
/// than in named fields. Reading them goes through the array-element address
/// path instead of the field path.
#[repr(C, align(8))]
#[derive(Clone, Copy)]
pub struct AlignedLanes(pub [f32; 2]);

// SAFETY: `AlignedLanes` is a `repr(C)` array of two f32 with no padding.
unsafe impl cuda_core::DeviceCopy for AlignedLanes {}

// SAFETY: both are plain `repr(C)` f32 pairs -- no padding holes, pointers, or
// interior mutability -- so a byte copy to the device is valid.
unsafe impl cuda_core::DeviceCopy for PackedPair {}
unsafe impl cuda_core::DeviceCopy for AlignedPair {}

/// Table entries for the cache-resident kernels.
const TABLE: usize = 4096;
/// Lookups per thread.
const ROUNDS: u32 = 64;

#[cuda_module]
mod kernels {
    use super::*;

    /// Control: align 4 is all that is proven, so this must stay two loads.
    #[kernel]
    pub fn packed_pair(pts: &[PackedPair], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let p = pts[i];
            *o = p.x + p.y;
        }
    }

    /// The fused case: `align(8)` is proven, so the pair becomes one wide load.
    #[kernel]
    pub fn aligned_pair(pts: &[AlignedPair], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let p = pts[i];
            *o = p.x + p.y;
        }
    }

    /// The array-lane form, read through the slice rather than copied into a
    /// local first. Copying the element to a local already fused, because SROA
    /// then loads the whole aggregate; reading through the place uses the
    /// address path, which is what this needed.
    #[kernel]
    pub fn lanes_through_ref(pts: &[AlignedLanes], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = pts[i].0[0] + pts[i].0[1];
        }
    }

    /// Cache-resident array-lane form, read through a reference: the kernel
    /// this change speeds up.
    #[kernel]
    pub fn hot_lanes(pts: &[AlignedLanes], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let mut acc = 0.0f32;
            let mut k = i as u32;
            let mut r = 0;
            while r < ROUNDS {
                let lanes = &pts[(k as usize) & (TABLE - 1)].0;
                acc += lanes[0] + lanes[1];
                k = k.wrapping_mul(1664525).wrapping_add(1013904223);
                r += 1;
            }
            *o = acc;
        }
    }

    /// Cache-resident control.
    #[kernel]
    pub fn hot_packed(pts: &[PackedPair], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let mut acc = 0.0f32;
            let mut k = i as u32;
            let mut r = 0;
            while r < ROUNDS {
                let p = pts[(k as usize) & (TABLE - 1)];
                acc += p.x + p.y;
                k = k.wrapping_mul(1664525).wrapping_add(1013904223);
                r += 1;
            }
            *o = acc;
        }
    }

    /// Cache-resident, over-aligned: the kernel this change speeds up.
    #[kernel]
    pub fn hot_aligned(pts: &[AlignedPair], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let mut acc = 0.0f32;
            let mut k = i as u32;
            let mut r = 0;
            while r < ROUNDS {
                let p = pts[(k as usize) & (TABLE - 1)];
                acc += p.x + p.y;
                k = k.wrapping_mul(1664525).wrapping_add(1013904223);
                r += 1;
            }
            *o = acc;
        }
    }
}

const N: usize = 1 << 22;
const ITERS: usize = 30;
const WARMUP: usize = 5;

fn main() {
    let ctx = CudaContext::new(0).expect("context");
    let stream = ctx.default_stream();

    let packed: Vec<PackedPair> = (0..N)
        .map(|i| PackedPair {
            x: i as f32,
            y: (i * 2) as f32,
        })
        .collect();
    let aligned: Vec<AlignedPair> = (0..N)
        .map(|i| AlignedPair {
            x: i as f32,
            y: (i * 2) as f32,
        })
        .collect();

    let stream_expect: Vec<f32> = (0..N).map(|i| i as f32 + (i * 2) as f32).collect();
    // Same LCG and order as the device loop, so the f32 accumulation matches
    // bit for bit rather than approximately.
    let hot_expect: Vec<f32> = (0..N)
        .map(|i| {
            let mut acc = 0.0f32;
            let mut k = i as u32;
            for _ in 0..ROUNDS {
                let j = (k as usize) & (TABLE - 1);
                acc += j as f32 + (j * 2) as f32;
                k = k.wrapping_mul(1664525).wrapping_add(1013904223);
            }
            acc
        })
        .collect();

    let lanes: Vec<AlignedLanes> = (0..N)
        .map(|i| AlignedLanes([i as f32, (i * 2) as f32]))
        .collect();

    let packed_dev = DeviceBuffer::from_host(&stream, &packed).unwrap();
    let lanes_dev = DeviceBuffer::from_host(&stream, &lanes).unwrap();
    let aligned_dev = DeviceBuffer::from_host(&stream, &aligned).unwrap();
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("module");
    let cfg = LaunchConfig::for_num_elems(N as u32);

    println!(
        "aligned_field_loads -- {} threads, {} timed launches after {} warmup",
        N, ITERS, WARMUP
    );
    println!("{:<22} {:>12}   correct", "kernel", "us/launch");
    println!("{}", "-".repeat(50));

    let mut all_ok = true;
    macro_rules! bench {
        ($name:literal, $call:expr, $want:ident) => {{
            for _ in 0..WARMUP {
                $call;
            }
            stream.synchronize().unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                $call;
            }
            stream.synchronize().unwrap();
            let us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
            let ok = out.to_host_vec(&stream).unwrap() == $want;
            all_ok &= ok;
            println!(
                "{:<22} {:>12.1}   {}",
                $name,
                us,
                if ok { "yes" } else { "NO" }
            );
        }};
    }

    // SAFETY: 1-D launch, one thread per element; both buffers cover every
    // access, and the table index is masked to TABLE entries.
    unsafe {
        bench!(
            "packed_pair",
            {
                module
                    .packed_pair(&stream, cfg, &packed_dev, &mut out)
                    .unwrap()
            },
            stream_expect
        );
        bench!(
            "aligned_pair",
            {
                module
                    .aligned_pair(&stream, cfg, &aligned_dev, &mut out)
                    .unwrap()
            },
            stream_expect
        );
        bench!(
            "lanes_through_ref",
            {
                module
                    .lanes_through_ref(&stream, cfg, &lanes_dev, &mut out)
                    .unwrap()
            },
            stream_expect
        );
        bench!(
            "hot_lanes",
            {
                module
                    .hot_lanes(&stream, cfg, &lanes_dev, &mut out)
                    .unwrap()
            },
            hot_expect
        );
        bench!(
            "hot_packed",
            {
                module
                    .hot_packed(&stream, cfg, &packed_dev, &mut out)
                    .unwrap()
            },
            hot_expect
        );
        bench!(
            "hot_aligned",
            {
                module
                    .hot_aligned(&stream, cfg, &aligned_dev, &mut out)
                    .unwrap()
            },
            hot_expect
        );
    }

    if all_ok {
        println!("\nSUCCESS: all 6 kernels bit-correct");
    } else {
        eprintln!("\nMISMATCH");
        std::process::exit(1);
    }
}
