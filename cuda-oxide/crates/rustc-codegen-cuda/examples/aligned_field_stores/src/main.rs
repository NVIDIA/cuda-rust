/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Adjacent field writes of an over-aligned element fuse into one wide store.
//!
//! The mirror of `aligned_field_loads`. A field address records the alignment
//! it can prove while the aggregate's `abi_align` is still in hand, and
//! `convert_load` consults that record when its own result type reports none.
//! `convert_store` did not: it asked only the stored value's type, and a scalar
//! answers nothing, so an `f32` field write exported at LLVM's default `align 4`
//! even when the element promised `align(8)`. LoadStoreVectorizer will not widen
//! past an alignment it cannot prove, so the pair stayed two `st.global.b32`.
//!
//! | kernel                          | element             | expected |
//! |---------------------------------|---------------------|----------|
//! | `packed_store`                  | `repr(C)`, align 4  | 2 stores -- align 4 is all rustc guarantees |
//! | `aligned_store`                 | `repr(C, align(8))` | **1 wide store** |
//! | `lanes_store`                   | array lanes, align 8| **1 wide store** (element path) |
//! | `hot_packed` / `hot_aligned`    | same, cache-resident| the timed comparison |
//!
//! `packed_store` is the control that must NOT change: nothing proves 8-byte
//! alignment there, and widening it would be wrong.
//!
//! Streaming writes are bandwidth-bound, so fusing them does not speed anything
//! up -- two stores to consecutive addresses already coalesce into the same
//! transactions, and the same bytes still have to reach memory. The win is in
//! the cache-resident kernels, where the store *instruction* count is the limit
//! rather than bandwidth: each thread rewrites its own two-element window
//! `ROUNDS` times, so the traffic is absorbed by cache and what remains is
//! issue.
//!
//! Run: `cargo oxide run aligned_field_stores`

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Two f32, natural alignment 4.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PackedPair {
    pub x: f32,
    pub y: f32,
}

/// The same payload, promising 8-byte alignment.
#[repr(C, align(8))]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct AlignedPair {
    pub x: f32,
    pub y: f32,
}

/// The same payload and alignment, but the two lanes live in an array rather
/// than in named fields, so writing them goes through the array-element
/// address path instead of the field path.
#[repr(C, align(8))]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct AlignedLanes(pub [f32; 2]);

// SAFETY: all three are plain `repr(C)` f32 pairs -- no padding holes,
// pointers, or interior mutability -- so a byte copy to the device is valid.
unsafe impl cuda_core::DeviceCopy for PackedPair {}
unsafe impl cuda_core::DeviceCopy for AlignedPair {}
unsafe impl cuda_core::DeviceCopy for AlignedLanes {}

/// Elements each thread owns in the cache-resident kernels. A power of two so
/// the LCG index masks cleanly, and small enough that every thread's window
/// stays in cache across all `ROUNDS` passes.
const WINDOW: usize = 2;
/// Writes per thread in the cache-resident kernels.
const ROUNDS: u32 = 64;

#[cuda_module]
mod kernels {
    use super::*;

    /// Control: align 4 is all that is proven, so this must stay two stores.
    #[kernel]
    pub fn packed_store(mut out: DisjointSlice<PackedPair>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            o.x = i as f32;
            o.y = (i * 2) as f32;
        }
    }

    /// The fused case: `align(8)` is proven, so the pair becomes one wide store.
    #[kernel]
    pub fn aligned_store(mut out: DisjointSlice<AlignedPair>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            o.x = i as f32;
            o.y = (i * 2) as f32;
        }
    }

    /// The array-lane form: same two f32, reached by index rather than by name.
    #[kernel]
    pub fn lanes_store(mut out: DisjointSlice<AlignedLanes>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            o.0[0] = i as f32;
            o.0[1] = (i * 2) as f32;
        }
    }

    /// Cache-resident control, natural alignment.
    ///
    /// Each thread rewrites its own `WINDOW` elements `ROUNDS` times. The index
    /// comes from an LCG so the compiler cannot prove which writes are dead and
    /// drop them, and the window is private to the thread so no two threads
    /// race for a slot.
    #[kernel]
    pub fn hot_packed(mut out: DisjointSlice<PackedPair>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if out.get_mut(idx).is_some() {
            let base = i * WINDOW;
            let mut k = i as u32;
            let mut r = 0u32;
            while r < ROUNDS {
                let slot = base + ((k as usize) & (WINDOW - 1));
                // SAFETY: `slot` lies in [base, base + WINDOW), this thread's
                // own window, and the buffer holds WINDOW elements per thread.
                let o = unsafe { out.get_unchecked_mut(slot) };
                o.x = r as f32;
                o.y = (r * 2) as f32;
                k = k.wrapping_mul(1664525).wrapping_add(1013904223);
                r += 1;
            }
        }
    }

    /// Cache-resident, over-aligned: the kernel this change speeds up.
    #[kernel]
    pub fn hot_aligned(mut out: DisjointSlice<AlignedPair>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if out.get_mut(idx).is_some() {
            let base = i * WINDOW;
            let mut k = i as u32;
            let mut r = 0u32;
            while r < ROUNDS {
                let slot = base + ((k as usize) & (WINDOW - 1));
                // SAFETY: as in `hot_packed`.
                let o = unsafe { out.get_unchecked_mut(slot) };
                o.x = r as f32;
                o.y = (r * 2) as f32;
                k = k.wrapping_mul(1664525).wrapping_add(1013904223);
                r += 1;
            }
        }
    }
}

/// Threads, and elements written by the streaming kernels.
const N: usize = 1 << 18;
const ITERS: usize = 30;
const WARMUP: usize = 5;

/// Replay one thread's LCG walk on the host: the last value written to each
/// slot of its window, in the same order the device writes them.
fn hot_expect<T: Copy>(threads: usize, make: impl Fn(u32) -> T, fill: T) -> Vec<T> {
    let mut out = vec![fill; threads * WINDOW];
    for i in 0..threads {
        let base = i * WINDOW;
        let mut k = i as u32;
        for r in 0..ROUNDS {
            out[base + ((k as usize) & (WINDOW - 1))] = make(r);
            k = k.wrapping_mul(1664525).wrapping_add(1013904223);
        }
    }
    out
}

fn main() {
    let ctx = CudaContext::new(0).expect("context");
    let stream = ctx.default_stream();

    // The cache-resident kernels address WINDOW elements per thread, so every
    // buffer is sized for the widest use and the streaming kernels write the
    // first N entries of it.
    let slots = N * WINDOW;

    let mut packed = DeviceBuffer::<PackedPair>::zeroed(&stream, slots).unwrap();
    let mut aligned = DeviceBuffer::<AlignedPair>::zeroed(&stream, slots).unwrap();
    let mut lanes = DeviceBuffer::<AlignedLanes>::zeroed(&stream, slots).unwrap();

    let module = kernels::load(&ctx).expect("module");
    let cfg = LaunchConfig::for_num_elems(N as u32);

    println!(
        "aligned_field_stores -- {} threads, {} timed launches after {} warmup",
        N, ITERS, WARMUP
    );
    println!("{:<16} {:>12}   correct", "kernel", "us/launch");
    println!("{}", "-".repeat(44));

    let mut all_ok = true;
    macro_rules! bench {
        ($name:literal, $call:expr, $check:expr) => {{
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
            let ok: bool = $check;
            all_ok &= ok;
            println!(
                "{:<16} {:>12.1}   {}",
                $name,
                us,
                if ok { "yes" } else { "NO" }
            );
        }};
    }

    // Streaming: one element per thread, written once. Expected to be
    // bandwidth-bound and therefore unchanged by fusion.
    let stream_packed: Vec<PackedPair> = (0..N)
        .map(|i| PackedPair {
            x: i as f32,
            y: (i * 2) as f32,
        })
        .collect();
    let stream_aligned: Vec<AlignedPair> = (0..N)
        .map(|i| AlignedPair {
            x: i as f32,
            y: (i * 2) as f32,
        })
        .collect();
    let stream_lanes: Vec<AlignedLanes> = (0..N)
        .map(|i| AlignedLanes([i as f32, (i * 2) as f32]))
        .collect();

    // SAFETY: 1-D launch, one thread per element; every buffer holds
    // WINDOW * N elements, which covers both the one-per-thread streaming
    // writes and the per-thread windows of the cache-resident kernels.
    unsafe {
        bench!(
            "packed_store",
            { module.packed_store(&stream, cfg, &mut packed).unwrap() },
            packed.to_host_vec(&stream).unwrap()[..N] == stream_packed[..]
        );
        bench!(
            "aligned_store",
            { module.aligned_store(&stream, cfg, &mut aligned).unwrap() },
            aligned.to_host_vec(&stream).unwrap()[..N] == stream_aligned[..]
        );
        bench!(
            "lanes_store",
            { module.lanes_store(&stream, cfg, &mut lanes).unwrap() },
            lanes.to_host_vec(&stream).unwrap()[..N] == stream_lanes[..]
        );

        let want_packed = hot_expect(
            N,
            |r| PackedPair {
                x: r as f32,
                y: (r * 2) as f32,
            },
            PackedPair { x: 0.0, y: 0.0 },
        );
        bench!(
            "hot_packed",
            { module.hot_packed(&stream, cfg, &mut packed).unwrap() },
            packed.to_host_vec(&stream).unwrap() == want_packed
        );

        let want_aligned = hot_expect(
            N,
            |r| AlignedPair {
                x: r as f32,
                y: (r * 2) as f32,
            },
            AlignedPair { x: 0.0, y: 0.0 },
        );
        bench!(
            "hot_aligned",
            { module.hot_aligned(&stream, cfg, &mut aligned).unwrap() },
            aligned.to_host_vec(&stream).unwrap() == want_aligned
        );
    }

    if all_ok {
        println!("\nSUCCESS: all 5 kernels bit-correct");
    } else {
        eprintln!("\nMISMATCH");
        std::process::exit(1);
    }
}
