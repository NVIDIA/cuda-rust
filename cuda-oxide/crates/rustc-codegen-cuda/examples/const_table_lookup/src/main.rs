/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! A compile-time lookup table should cost the same however it is spelled.
//!
//! Every kernel reads the *same table* with the *same divergent index* and does
//! the same arithmetic. Only the spelling differs:
//!
//! | kernel             | table spelling                  |
//! |--------------------|---------------------------------|
//! | `lutN_value`       | `const T: [f32; N]`             |
//! | `lutN_ref`         | `const T: &[f32; N]`            |
//! | `lutN_constant`    | `ConstantMemory<[f32; N]>`      |
//! | `lut256_mut_copy`  | `let mut t = T; t[j] = x;`      |
//!
//! `lutN_ref` is the reference: a promoted array constant behind a reference has
//! always been materialized as one immutable device global, read with a single
//! `ld.global.nc`. `lutN_value` is the same data written without the `&`, and it
//! used to be materialized *per thread* into the local depot — one `st.local`
//! per element in every thread, for data the module image already carries — then
//! read back from thread-private memory. Measured on an A10G (sm_86), that cost
//! 3.4x at 16 entries and 49.9x at 256. The two spellings now agree to within
//! 1%, which is what this example exists to keep true.
//!
//! Two sizes are measured because `ptxas` can rescue a small table on its own
//! (it reports no stack frame at 16 entries either way), so the 256-entry row is
//! the one that shows what the lowering costs.
//!
//! `lutN_constant` is *not* fixed, and is kept here as the contrast:
//! [`ConstantMemory::get`] returns `T` by value, so a `ConstantMemory<[f32; N]>`
//! reads all N entries out of `.const` and writes them into a local depot. It
//! stays far off the other two, and a lookup table cannot use `.const` today.
//!
//! `lut256_mut_copy` covers the other side of the write-once guard: it writes
//! its copy of the table before reading it, so the memcpy-from-immutable-global
//! lowering must not fire. Each thread overwrites a different entry with a
//! thread-unique value, so its bit-exact check proves the copy stays private to
//! the thread; its timing is expected to look like the old `lut256_value`.
//!
//! Each thread performs `ROUNDS` lookups against a per-thread LCG, so the index
//! is unpredictable and lane-divergent and the kernel is lookup-bound rather
//! than bandwidth-bound (4 bytes in, 4 bytes out per thread).
//!
//! Run: `cargo oxide run const_table_lookup`

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use cuda_device::cuda_module;

/// Threads, and `u32` elements in the input buffer.
const ELEMS: usize = 8 << 20;

/// Small table: a power of two so the index mask is a single `and`.
const N16: usize = 16;

/// Realistic lookup-table size, and too large for a select chain.
const N256: usize = 256;

/// Table lookups per thread. High enough that the lookup, not the 8 bytes of
/// I/O, sets the kernel's time.
const ROUNDS: u32 = 64;

/// Timed launches per kernel.
const ITERS: usize = 50;

/// Untimed launches first, so clocks and caches settle.
const WARMUP: usize = 10;

/// The 16-entry table, shared by every variant and by the host reference.
const TABLE16: [f32; N16] = [
    1.25, -2.5, 5.0, 10.5, 0.75, -1.5, 3.25, 7.0, 2.5, -0.25, 4.5, 9.75, 1.0, -3.75, 6.25, 8.5,
];

/// The 256-entry table. Built in a const block so it is a compile-time constant
/// with no host upload, exactly like `TABLE16`.
const TABLE256: [f32; N256] = {
    let mut t = [0.0_f32; N256];
    let mut i = 0;
    while i < N256 {
        t[i] = (i as f32) * 0.125 - 16.0;
        i += 1;
    }
    t
};

#[cuda_module]
mod kernels {
    use cuda_device::{ConstantMemory, DisjointSlice, constant, kernel, thread};

    /// Bare array constants: materialized per thread into `.local`.
    const T16: [f32; super::N16] = super::TABLE16;
    const T256: [f32; super::N256] = super::TABLE256;

    /// The identical data behind a reference: reaches `.global` today.
    const T16_REF: &[f32; super::N16] = &super::TABLE16;
    const T256_REF: &[f32; super::N256] = &super::TABLE256;

    /// The same data in `.const`, uploaded by the host.
    #[constant]
    static T16_CONST: ConstantMemory<[f32; super::N16]> = ConstantMemory::UNINIT;
    #[constant]
    static T256_CONST: ConstantMemory<[f32; super::N256]> = ConstantMemory::UNINIT;

    /// One LCG step. Shared by every kernel so the index stream is identical and
    /// only the table read differs.
    #[inline(always)]
    fn step(h: u32) -> u32 {
        h.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
    }

    // Every kernel below is the same loop with one expression changed: the table
    // read. Written out longhand because `#[cuda_module]` expands before any
    // inner `macro_rules!` would, and because keeping the loop textually
    // identical is the point of the comparison.

    #[kernel]
    pub fn lut16_value(input: &[u32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let mut h = input[i];
        let mut acc = 0.0_f32;
        for _ in 0..super::ROUNDS {
            h = step(h);
            acc += T16[(h >> 24) as usize & (super::N16 - 1)];
        }
        if let Some(o) = output.get_mut(idx) {
            *o = acc;
        }
    }

    #[kernel]
    pub fn lut16_ref(input: &[u32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let mut h = input[i];
        let mut acc = 0.0_f32;
        for _ in 0..super::ROUNDS {
            h = step(h);
            acc += T16_REF[(h >> 24) as usize & (super::N16 - 1)];
        }
        if let Some(o) = output.get_mut(idx) {
            *o = acc;
        }
    }

    #[kernel]
    pub fn lut16_constant(input: &[u32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let mut h = input[i];
        let mut acc = 0.0_f32;
        for _ in 0..super::ROUNDS {
            h = step(h);
            acc += T16_CONST.get()[(h >> 24) as usize & (super::N16 - 1)];
        }
        if let Some(o) = output.get_mut(idx) {
            *o = acc;
        }
    }

    #[kernel]
    pub fn lut256_value(input: &[u32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let mut h = input[i];
        let mut acc = 0.0_f32;
        for _ in 0..super::ROUNDS {
            h = step(h);
            acc += T256[(h >> 24) as usize & (super::N256 - 1)];
        }
        if let Some(o) = output.get_mut(idx) {
            *o = acc;
        }
    }

    #[kernel]
    pub fn lut256_ref(input: &[u32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let mut h = input[i];
        let mut acc = 0.0_f32;
        for _ in 0..super::ROUNDS {
            h = step(h);
            acc += T256_REF[(h >> 24) as usize & (super::N256 - 1)];
        }
        if let Some(o) = output.get_mut(idx) {
            *o = acc;
        }
    }

    #[kernel]
    pub fn lut256_constant(input: &[u32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let mut h = input[i];
        let mut acc = 0.0_f32;
        for _ in 0..super::ROUNDS {
            h = step(h);
            acc += T256_CONST.get()[(h >> 24) as usize & (super::N256 - 1)];
        }
        if let Some(o) = output.get_mut(idx) {
            *o = acc;
        }
    }

    /// The write-once guard's other half: this kernel *writes* its copy of the
    /// table before reading it, so the memcpy-from-immutable-global lowering
    /// must not fire and the per-thread materialization must survive. Its row
    /// is about the `correct` column, not the timing one: thread `i` overwrites
    /// entry `i & (N - 1)` with a thread-unique value, so if the lowering ever
    /// handed every thread one shared table, threads would observe each other's
    /// writes and the bit-exact check would fail.
    #[kernel]
    pub fn lut256_mut_copy(input: &[u32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let mut t = T256;
        t[i & (super::N256 - 1)] = input[i] as f32;
        let mut h = input[i];
        let mut acc = 0.0_f32;
        for _ in 0..super::ROUNDS {
            h = step(h);
            acc += t[(h >> 24) as usize & (super::N256 - 1)];
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

/// Reference for `lut256_mut_copy`: thread `i` first overwrites one entry of
/// its private copy, so every thread's table differs by one element.
fn expected_mut(i: usize) -> f32 {
    let mut table = TABLE256;
    table[i & (N256 - 1)] = seed(i) as f32;
    expected(i, &table)
}

/// The host reference: the same LCG, the same table, the same add order, so a
/// correct kernel matches bit for bit (only adds, so no FMA contraction).
fn expected(i: usize, table: &[f32]) -> f32 {
    let mask = table.len() - 1;
    let mut h = seed(i);
    let mut acc = 0.0_f32;
    for _ in 0..ROUNDS {
        h = h.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        acc += table[(h >> 24) as usize & mask];
    }
    acc
}

/// One measured kernel.
struct Row {
    name: &'static str,
    entries: usize,
    /// How the table is written in the source, which is the only difference
    /// between the kernels.
    spelling: &'static str,
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
    // The `.const` variants need their tables uploaded; the others carry theirs
    // in the module image.
    module
        .set_t16_const(&stream, &TABLE16)
        .expect("upload 16-entry .const table");
    module
        .set_t256_const(&stream, &TABLE256)
        .expect("upload 256-entry .const table");

    let host_in: Vec<u32> = (0..ELEMS).map(seed).collect();
    let input = DeviceBuffer::from_host(&stream, &host_in).expect("input alloc");
    let mut output = DeviceBuffer::<f32>::zeroed(&stream, ELEMS).expect("output alloc");

    let cfg = LaunchConfig::for_num_elems(ELEMS as u32);
    let ref16: Vec<f32> = (0..ELEMS).map(|i| expected(i, &TABLE16)).collect();
    let ref256: Vec<f32> = (0..ELEMS).map(|i| expected(i, &TABLE256)).collect();
    let refmut: Vec<f32> = (0..ELEMS).map(expected_mut).collect();

    let mut rows: Vec<Row> = Vec::new();

    macro_rules! measure {
        ($name:literal, $entries:expr, $spelling:literal, $call:ident, $reference:expr) => {{
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
                entries: $entries,
                spelling: $spelling,
                us,
                correct,
            });
            us
        }};
    }

    let v16 = measure!("lut16_value", N16, "[f32; N]", lut16_value, &ref16);
    let r16 = measure!("lut16_ref", N16, "&[f32; N]", lut16_ref, &ref16);
    let c16 = measure!("lut16_constant", N16, "ConstantMem", lut16_constant, &ref16);
    let v256 = measure!("lut256_value", N256, "[f32; N]", lut256_value, &ref256);
    let r256 = measure!("lut256_ref", N256, "&[f32; N]", lut256_ref, &ref256);
    let c256 = measure!(
        "lut256_constant",
        N256,
        "ConstantMem",
        lut256_constant,
        &ref256
    );
    // Correctness coverage for the write-once guard's fallback, not a timing
    // contest: a written copy must stay per-thread and element-materialized.
    measure!(
        "lut256_mut_copy",
        N256,
        "mut copy",
        lut256_mut_copy,
        &refmut
    );

    println!();
    println!(
        "const_table_lookup -- {ELEMS} threads x {ROUNDS} divergent lookups per thread \
         ({} lookups per launch)",
        ELEMS * ROUNDS as usize
    );
    println!("{ITERS} timed launches after {WARMUP} warmup\n");
    println!(
        "{:<16} {:>8} {:>12} {:>11} {:>12} {:>9}",
        "kernel", "entries", "spelling", "us/launch", "Glookups/s", "correct"
    );
    println!("{:-<74}", "");
    for r in &rows {
        println!(
            "{:<16} {:>8} {:>12} {:>11.1} {:>12.2} {:>9}",
            r.name,
            r.entries,
            r.spelling,
            r.us,
            glookups(r.us),
            if r.correct { "yes" } else { "NO" }
        );
    }

    // Each spelling against `&[f32; N]`, the one that has always been a single
    // immutable device global. `[f32; N]` should now sit at 1.0x; `ConstantMem`
    // is expected to be far off, and stays here as the contrast.
    println!("\ntime relative to the `&[f32; N]` spelling (1.00x = same cost):");
    println!("{:-<74}", "");
    println!(
        "  {N16:>3} entries:  [f32; N] {:>6.2}x   ConstantMem {:>7.2}x",
        v16 / r16,
        c16 / r16
    );
    println!(
        "  {N256:>3} entries:  [f32; N] {:>6.2}x   ConstantMem {:>7.2}x",
        v256 / r256,
        c256 / r256
    );

    let wrong: Vec<&str> = rows.iter().filter(|r| !r.correct).map(|r| r.name).collect();
    if wrong.is_empty() {
        println!("\n\u{2713} SUCCESS: all {} kernels bit-correct", rows.len());
    } else {
        println!("\n\u{2717} FAILED: incorrect output from {wrong:?}");
        std::process::exit(1);
    }
}
