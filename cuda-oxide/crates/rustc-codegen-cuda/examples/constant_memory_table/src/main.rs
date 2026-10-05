/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Reading a `ConstantMemory` table at a runtime index, and where `.const` wins.
//!
//! [`ConstantMemory::get`] returns `T` *by value*. For a scalar that is exactly
//! right, but a value has no address, so indexing a `ConstantMemory<[f32; N]>`
//! copy at runtime spills the whole array to the thread's local depot first —
//! one `st.local` per element in *every thread* — and the lookup then reads
//! thread-private memory. [`ConstantMemory::get_ref`] borrows the storage
//! instead, so the index stays in constant space and the read is one `ld.const`.
//!
//! Each row below reads the same 256-entry table with the same arithmetic. Two
//! things vary: how the table is reached, and how much the lanes of a warp agree
//! about the index.
//!
//! | kernel               | table reached by             | index         |
//! |----------------------|------------------------------|---------------|
//! | `const_getval_div`   | `TABLE.get()[i]`             | divergent     |
//! | `const_ref_div`      | `TABLE.get_ref()[i]`         | divergent     |
//! | `global_div`         | `const T: [f32; N]`          | divergent     |
//! | `const_getval_uni`   | `TABLE.get()[i]`             | warp-uniform  |
//! | `const_ref_uni`      | `TABLE.get_ref()[i]`         | warp-uniform  |
//! | `global_uni`         | `const T: [f32; N]`          | warp-uniform  |
//!
//! The index axis is the point. Constant memory is served by a
//! broadcast-oriented cache: when every lane of a warp wants the same entry it
//! is one broadcast, and when lanes want different entries the distinct
//! addresses are served in sequence. Ordinary global memory behaves the other
//! way round — a warp's divergent reads of a small table coalesce and the table
//! stays resident in L1. So `.const` is not simply "faster memory", and the
//! right choice follows the access pattern rather than the qualifier.
//!
//! `global_*` is the comparison point: a plain `const T: [f32; N]` is
//! materialized as one immutable device global and read with `ld.global.nc`, with
//! no host upload and no `#[constant]` at all.
//!
//! Each thread performs `ROUNDS` dependent lookups, so the kernel is
//! lookup-bound rather than bandwidth-bound (4 bytes in, 4 bytes out).
//!
//! Run: `cargo oxide run constant_memory_table`

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use cuda_device::cuda_module;

/// Threads, and `u32` elements in the input buffer.
const ELEMS: usize = 8 << 20;

/// Table entries. A power of two so the index mask is a single `and`.
const N: usize = 256;

/// Lookups per thread.
const ROUNDS: u32 = 64;

/// Timed launches per kernel.
const ITERS: usize = 50;

/// Untimed launches first, so clocks and caches settle.
const WARMUP: usize = 10;

/// Lanes per warp. The warp-uniform kernels seed from a warp-aligned index, so
/// all lanes of a warp walk the same index stream.
const WARP: usize = 32;

/// The table. The `.const` variants receive it by upload; `global_*` carries the
/// identical values as a compile-time constant.
const TABLE: [f32; N] = {
    let mut t = [0.0_f32; N];
    let mut i = 0;
    while i < N {
        t[i] = (i as f32) * 0.125 - 16.0;
        i += 1;
    }
    t
};

#[cuda_module]
mod kernels {
    use cuda_device::{ConstantMemory, DisjointSlice, constant, kernel, thread};

    /// Host-uploaded constant memory.
    #[constant]
    static TABLE_CONST: ConstantMemory<[f32; super::N]> = ConstantMemory::UNINIT;

    /// The identical values as a bare array constant, which is materialized as
    /// one immutable device global.
    const TABLE_GLOBAL: [f32; super::N] = super::TABLE;

    /// One LCG step. Shared by every kernel so only the table read and the
    /// index's divergence differ.
    #[inline(always)]
    fn step(h: u32) -> u32 {
        h.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
    }

    // Six kernels, each the same loop with one expression changed. Written out
    // longhand because `#[cuda_module]` expands before any inner `macro_rules!`
    // would, and because keeping the loop textually identical is the whole point
    // of the comparison.
    //
    // The `_div` kernels seed from `input[i]`, so each lane walks its own index
    // stream. The `_uni` kernels seed from `input[i & !(WARP - 1)]`, which is one
    // warp-uniform load, so every lane of a warp asks for the same entry. Both
    // read the input buffer once and write the output once.

    #[kernel]
    pub fn const_getval_div(input: &[u32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let mut h = input[i];
        let mut acc = 0.0_f32;
        for _ in 0..super::ROUNDS {
            h = step(h);
            acc += TABLE_CONST.get()[(h >> 24) as usize & (super::N - 1)];
        }
        if let Some(o) = output.get_mut(idx) {
            *o = acc;
        }
    }

    #[kernel]
    pub fn const_ref_div(input: &[u32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let mut h = input[i];
        let mut acc = 0.0_f32;
        for _ in 0..super::ROUNDS {
            h = step(h);
            acc += TABLE_CONST.get_ref()[(h >> 24) as usize & (super::N - 1)];
        }
        if let Some(o) = output.get_mut(idx) {
            *o = acc;
        }
    }

    #[kernel]
    pub fn global_div(input: &[u32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let mut h = input[i];
        let mut acc = 0.0_f32;
        for _ in 0..super::ROUNDS {
            h = step(h);
            acc += TABLE_GLOBAL[(h >> 24) as usize & (super::N - 1)];
        }
        if let Some(o) = output.get_mut(idx) {
            *o = acc;
        }
    }

    #[kernel]
    pub fn const_getval_uni(input: &[u32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let mut h = input[i & !(super::WARP - 1)];
        let mut acc = 0.0_f32;
        for _ in 0..super::ROUNDS {
            h = step(h);
            acc += TABLE_CONST.get()[(h >> 24) as usize & (super::N - 1)];
        }
        if let Some(o) = output.get_mut(idx) {
            *o = acc;
        }
    }

    #[kernel]
    pub fn const_ref_uni(input: &[u32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let mut h = input[i & !(super::WARP - 1)];
        let mut acc = 0.0_f32;
        for _ in 0..super::ROUNDS {
            h = step(h);
            acc += TABLE_CONST.get_ref()[(h >> 24) as usize & (super::N - 1)];
        }
        if let Some(o) = output.get_mut(idx) {
            *o = acc;
        }
    }

    #[kernel]
    pub fn global_uni(input: &[u32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let mut h = input[i & !(super::WARP - 1)];
        let mut acc = 0.0_f32;
        for _ in 0..super::ROUNDS {
            h = step(h);
            acc += TABLE_GLOBAL[(h >> 24) as usize & (super::N - 1)];
        }
        if let Some(o) = output.get_mut(idx) {
            *o = acc;
        }
    }
}

/// Host-side seed for thread `i`.
fn seed(i: usize) -> u32 {
    (i as u32).wrapping_mul(2_654_435_761).wrapping_add(12_345)
}

/// The host reference for thread `i`. `warp_uniform` selects the same
/// warp-aligned seed the `_uni` kernels use. Only adds, so no FMA contraction
/// can change the result, and a correct kernel matches bit for bit.
fn expected(i: usize, warp_uniform: bool) -> f32 {
    let mut h = seed(if warp_uniform { i & !(WARP - 1) } else { i });
    let mut acc = 0.0_f32;
    for _ in 0..ROUNDS {
        h = h.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        acc += TABLE[(h >> 24) as usize & (N - 1)];
    }
    acc
}

/// One measured kernel.
struct Row {
    name: &'static str,
    reached_by: &'static str,
    index: &'static str,
    us: f64,
    correct: bool,
}

/// Billions of table lookups per second.
fn glookups(us: f64) -> f64 {
    (ELEMS as f64 * f64::from(ROUNDS)) / (us * 1e-6) / 1e9
}

fn zero(stream: &CudaStream, buf: &mut DeviceBuffer<f32>) {
    let zeros = vec![0.0f32; ELEMS];
    buf.copy_from_host(stream, &zeros).expect("zero fill");
    stream.synchronize().expect("zero sync");
}

fn main() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    module
        .set_table_const(&stream, &TABLE)
        .expect("upload the constant-memory table");

    let host_in: Vec<u32> = (0..ELEMS).map(seed).collect();
    let input = DeviceBuffer::from_host(&stream, &host_in).expect("input alloc");
    let mut output = DeviceBuffer::<f32>::zeroed(&stream, ELEMS).expect("output alloc");

    let cfg = LaunchConfig::for_num_elems(ELEMS as u32);
    let ref_div: Vec<f32> = (0..ELEMS).map(|i| expected(i, false)).collect();
    let ref_uni: Vec<f32> = (0..ELEMS).map(|i| expected(i, true)).collect();

    let mut rows: Vec<Row> = Vec::new();

    macro_rules! measure {
        ($name:literal, $reached:literal, $index:literal, $call:ident, $reference:expr) => {{
            zero(&stream, &mut output);
            for _ in 0..WARMUP {
                // SAFETY: launch shape/resources match the kernel; the buffers
                // cover exactly the `ELEMS` elements it accesses.
                unsafe { module.$call(&stream, cfg, &input, &mut output) }
                    .expect(concat!($name, " warmup"));
            }
            stream.synchronize().expect("warmup sync");
            let start = stream
                .record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
                .expect("start event");
            for _ in 0..ITERS {
                // SAFETY: as above.
                unsafe { module.$call(&stream, cfg, &input, &mut output) }
                    .expect(concat!($name, " timed"));
            }
            let end = stream
                .record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
                .expect("end event");
            let us = f64::from(start.elapsed_ms(&end).expect("elapsed")) * 1000.0 / ITERS as f64;
            let got = output.to_host_vec(&stream).expect("readback");
            let correct = got.len() == ELEMS && got == *$reference;
            rows.push(Row {
                name: $name,
                reached_by: $reached,
                index: $index,
                us,
                correct,
            });
            us
        }};
    }

    let gv_div = measure!(
        "const_getval_div",
        "get()",
        "divergent",
        const_getval_div,
        &ref_div
    );
    let rf_div = measure!(
        "const_ref_div",
        "get_ref()",
        "divergent",
        const_ref_div,
        &ref_div
    );
    let gl_div = measure!(
        "global_div",
        "const [f32; N]",
        "divergent",
        global_div,
        &ref_div
    );
    let gv_uni = measure!(
        "const_getval_uni",
        "get()",
        "warp-uniform",
        const_getval_uni,
        &ref_uni
    );
    let rf_uni = measure!(
        "const_ref_uni",
        "get_ref()",
        "warp-uniform",
        const_ref_uni,
        &ref_uni
    );
    let gl_uni = measure!(
        "global_uni",
        "const [f32; N]",
        "warp-uniform",
        global_uni,
        &ref_uni
    );

    println!();
    println!(
        "constant_memory_table -- {ELEMS} threads x {ROUNDS} lookups into a {N}-entry f32 table"
    );
    println!("{ITERS} timed launches after {WARMUP} warmup\n");
    println!(
        "{:<18} {:>16} {:>14} {:>11} {:>12} {:>9}",
        "kernel", "reached by", "index", "us/launch", "Glookups/s", "correct"
    );
    println!("{:-<86}", "");
    for r in &rows {
        println!(
            "{:<18} {:>16} {:>14} {:>11.1} {:>12.2} {:>9}",
            r.name,
            r.reached_by,
            r.index,
            r.us,
            glookups(r.us),
            if r.correct { "yes" } else { "NO" }
        );
    }

    println!("\nwhat get_ref() buys, per index pattern (time vs the get() copy):");
    println!("{:-<86}", "");
    println!("  divergent     get_ref() {:>6.2}x faster", gv_div / rf_div);
    println!("  warp-uniform  get_ref() {:>6.2}x faster", gv_uni / rf_uni);

    println!("\nwhich address space suits which pattern (time, lower is better):");
    println!("{:-<86}", "");
    println!(
        "  divergent     .const {:>9.1}us   vs  .global {:>9.1}us   -> .global {:>5.2}x",
        rf_div,
        gl_div,
        rf_div / gl_div
    );
    println!(
        "  warp-uniform  .const {:>9.1}us   vs  .global {:>9.1}us   -> .const  {:>5.2}x",
        rf_uni,
        gl_uni,
        gl_uni / rf_uni
    );

    let wrong: Vec<&str> = rows.iter().filter(|r| !r.correct).map(|r| r.name).collect();
    if wrong.is_empty() {
        println!("\n\u{2713} SUCCESS: all {} kernels bit-correct", rows.len());
    } else {
        println!("\n\u{2717} FAILED: incorrect output from {wrong:?}");
        std::process::exit(1);
    }
}
