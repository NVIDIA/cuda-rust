/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! A field-less enum table should cost what the equivalent integer table costs.
//!
//! A `const OPS: [Op; N]` where `Op` is a field-less `#[repr(u32)]` enum holds
//! nothing but discriminants, so reading one entry is a single load — exactly
//! like reading a `const CODES: [u32; N]` of the same values. Each pair of
//! kernels below does the same arithmetic over the same index stream and differs
//! only in whether the table's element type is the enum or its discriminant:
//!
//! | kernel              | table                          |
//! |---------------------|--------------------------------|
//! | `enumN_table`       | `const OPS: [Op; N]`           |
//! | `u32N_table`        | `const CODES: [u32; N]`        |
//! | `enum256_ref_table` | `const R: &[Op; 256] = &OPS`   |
//!
//! The integer form has been materialized as one immutable device global, read
//! with a single `ld.global.nc`, since bare array constants stopped being built
//! per thread. The enum form was left out of that: it was materialized into the
//! per-thread local depot with one `st.local` per element in *every* thread, for
//! data the module image already carries, and the lookup then read
//! thread-private memory. At 256 entries that was 299 PTX instructions and a
//! 1 KiB depot against 42 instructions and none.
//!
//! There was no way to spell around it either. `&[Op; N]` — the reference form
//! that works for scalar tables — is rejected outright, so a field-less enum
//! table had no fast spelling at all. Both spellings are exercised here:
//! `enum256_ref_table` reads through `const R: &[Op; 256] = &OPS`, which used
//! to hard-error with "invalid input program" and now lowers to the same
//! immutable global as the bare table.
//!
//! Both sizes are measured because `ptxas` can rescue a small table on its own,
//! so the 256-entry row is the one that shows what the lowering costs.
//!
//! Each thread performs `ROUNDS` dependent lookups against a per-thread LCG, so
//! the index is unpredictable and lane-divergent and the kernel is lookup-bound
//! rather than bandwidth-bound (4 bytes in, 4 bytes out per thread).
//!
//! Run: `cargo oxide run enum_table_lookup`

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use cuda_device::cuda_module;

/// Threads, and `u32` elements in the input buffer.
const ELEMS: usize = 8 << 20;

/// Small table, which `ptxas` can rescue by itself.
const N16: usize = 16;

/// Realistic table size, past anything `ptxas` will unpack.
const N256: usize = 256;

/// Lookups per thread.
const ROUNDS: u32 = 64;

/// Timed launches per kernel.
const ITERS: usize = 50;

/// Untimed launches first, so clocks and caches settle.
const WARMUP: usize = 10;

/// A field-less enum: every variant is just a discriminant, so an `[Op; N]`
/// holds exactly the same bytes as the `[u32; N]` of its values.
#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(u32)]
pub enum Op {
    Add = 3,
    Sub = 5,
    Mul = 7,
    Div = 11,
    Rem = 13,
    Shl = 17,
    Shr = 19,
    Xor = 23,
}

/// The discriminant an entry carries at index `i`. Used to build both tables, so
/// the enum table and the integer table are byte-identical by construction.
const fn code_at(i: usize) -> u32 {
    match i % 8 {
        0 => 3,
        1 => 5,
        2 => 7,
        3 => 11,
        4 => 13,
        5 => 17,
        6 => 19,
        _ => 23,
    }
}

const fn op_at(i: usize) -> Op {
    match i % 8 {
        0 => Op::Add,
        1 => Op::Sub,
        2 => Op::Mul,
        3 => Op::Div,
        4 => Op::Rem,
        5 => Op::Shl,
        6 => Op::Shr,
        _ => Op::Xor,
    }
}

macro_rules! build_tables {
    ($n:expr, $ops:ident, $codes:ident) => {
        const $ops: [Op; $n] = {
            let mut t = [Op::Add; $n];
            let mut i = 0;
            while i < $n {
                t[i] = op_at(i);
                i += 1;
            }
            t
        };
        const $codes: [u32; $n] = {
            let mut t = [0u32; $n];
            let mut i = 0;
            while i < $n {
                t[i] = code_at(i);
                i += 1;
            }
            t
        };
    };
}

build_tables!(N16, OPS16, CODES16);
build_tables!(N256, OPS256, CODES256);

/// The reference spelling of the 256-entry enum table. This form used to be
/// rejected outright; it must lower to the same immutable global the bare
/// table gets, and the kernel reading through it must match the same host
/// reference.
const OPS256_REF: &[Op; N256] = &OPS256;

#[cuda_module]
mod kernels {
    use cuda_device::{DisjointSlice, kernel, thread};

    /// One LCG step. Shared by every kernel so the index stream is identical and
    /// only the table's element type differs.
    #[inline(always)]
    fn step(h: u32) -> u32 {
        h.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
    }

    // Four kernels, each the same loop with one expression changed. Written out
    // longhand because `#[cuda_module]` expands before any inner `macro_rules!`
    // would, and because keeping the loop textually identical is the point.

    #[kernel]
    pub fn enum16_table(input: &[u32], mut output: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let mut h = input[i];
        let mut acc = 0u32;
        for _ in 0..super::ROUNDS {
            h = step(h);
            acc = acc.wrapping_add(super::OPS16[(h >> 24) as usize & (super::N16 - 1)] as u32);
        }
        if let Some(o) = output.get_mut(idx) {
            *o = acc;
        }
    }

    #[kernel]
    pub fn u3216_table(input: &[u32], mut output: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let mut h = input[i];
        let mut acc = 0u32;
        for _ in 0..super::ROUNDS {
            h = step(h);
            acc = acc.wrapping_add(super::CODES16[(h >> 24) as usize & (super::N16 - 1)]);
        }
        if let Some(o) = output.get_mut(idx) {
            *o = acc;
        }
    }

    #[kernel]
    pub fn enum256_table(input: &[u32], mut output: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let mut h = input[i];
        let mut acc = 0u32;
        for _ in 0..super::ROUNDS {
            h = step(h);
            acc = acc.wrapping_add(super::OPS256[(h >> 24) as usize & (super::N256 - 1)] as u32);
        }
        if let Some(o) = output.get_mut(idx) {
            *o = acc;
        }
    }

    #[kernel]
    pub fn u32256_table(input: &[u32], mut output: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let mut h = input[i];
        let mut acc = 0u32;
        for _ in 0..super::ROUNDS {
            h = step(h);
            acc = acc.wrapping_add(super::CODES256[(h >> 24) as usize & (super::N256 - 1)]);
        }
        if let Some(o) = output.get_mut(idx) {
            *o = acc;
        }
    }

    /// The same loop as `enum256_table`, read through the reference spelling
    /// `const R: &[Op; 256] = &OPS` instead of the bare table.
    #[kernel]
    pub fn enum256_ref_table(input: &[u32], mut output: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let mut h = input[i];
        let mut acc = 0u32;
        for _ in 0..super::ROUNDS {
            h = step(h);
            acc =
                acc.wrapping_add(super::OPS256_REF[(h >> 24) as usize & (super::N256 - 1)] as u32);
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

/// The host reference: the same LCG over the same discriminants, so both members
/// of a pair must match it bit for bit.
fn expected(i: usize, entries: usize) -> u32 {
    let mut h = seed(i);
    let mut acc = 0u32;
    for _ in 0..ROUNDS {
        h = h.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        acc = acc.wrapping_add(code_at((h >> 24) as usize & (entries - 1)));
    }
    acc
}

/// One measured kernel.
struct Row {
    name: &'static str,
    entries: usize,
    element: &'static str,
    us: f64,
    correct: bool,
}

/// Billions of table lookups per second.
fn glookups(us: f64) -> f64 {
    (ELEMS as f64 * f64::from(ROUNDS)) / (us * 1e-6) / 1e9
}

fn zero(stream: &CudaStream, buf: &mut DeviceBuffer<u32>) {
    let zeros = vec![0u32; ELEMS];
    buf.copy_from_host(stream, &zeros).expect("zero fill");
    stream.synchronize().expect("zero sync");
}

fn main() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let host_in: Vec<u32> = (0..ELEMS).map(seed).collect();
    let input = DeviceBuffer::from_host(&stream, &host_in).expect("input alloc");
    let mut output = DeviceBuffer::<u32>::zeroed(&stream, ELEMS).expect("output alloc");

    let cfg = LaunchConfig::for_num_elems(ELEMS as u32);
    let ref16: Vec<u32> = (0..ELEMS).map(|i| expected(i, N16)).collect();
    let ref256: Vec<u32> = (0..ELEMS).map(|i| expected(i, N256)).collect();

    let mut rows: Vec<Row> = Vec::new();

    macro_rules! measure {
        ($name:literal, $entries:expr, $element:literal, $call:ident, $reference:expr) => {{
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
                element: $element,
                us,
                correct,
            });
            us
        }};
    }

    let e16 = measure!("enum16_table", N16, "Op", enum16_table, &ref16);
    let c16 = measure!("u3216_table", N16, "u32", u3216_table, &ref16);
    let e256 = measure!("enum256_table", N256, "Op", enum256_table, &ref256);
    let c256 = measure!("u32256_table", N256, "u32", u32256_table, &ref256);
    let r256 = measure!("enum256_ref_table", N256, "&Op", enum256_ref_table, &ref256);

    println!();
    println!(
        "enum_table_lookup -- {ELEMS} threads x {ROUNDS} divergent lookups per thread \
         ({} lookups per launch)",
        ELEMS * ROUNDS as usize
    );
    println!("{ITERS} timed launches after {WARMUP} warmup\n");
    println!(
        "{:<18} {:>8} {:>9} {:>11} {:>12} {:>9}",
        "kernel", "entries", "element", "us/launch", "Glookups/s", "correct"
    );
    println!("{:-<72}", "");
    for r in &rows {
        println!(
            "{:<18} {:>8} {:>9} {:>11.1} {:>12.2} {:>9}",
            r.name,
            r.entries,
            r.element,
            r.us,
            glookups(r.us),
            if r.correct { "yes" } else { "NO" }
        );
    }

    // The enum table should now cost what the integer table of the same
    // discriminants costs: 1.00x is the target, not a speedup over it. The
    // reference spelling reads the same global, so it is held to the same bar.
    println!("\ntime of the enum table relative to the u32 table of the same values:");
    println!("{:-<72}", "");
    println!("  {N16:>3} entries:  {:>6.2}x", e16 / c16);
    println!("  {N256:>3} entries:  {:>6.2}x", e256 / c256);
    println!(
        "  {N256:>3} entries through &[Op; {N256}]:  {:>6.2}x",
        r256 / c256
    );

    let wrong: Vec<&str> = rows.iter().filter(|r| !r.correct).map(|r| r.name).collect();
    if wrong.is_empty() {
        println!("\n\u{2713} SUCCESS: all {} kernels bit-correct", rows.len());
    } else {
        println!("\n\u{2717} FAILED: incorrect output from {wrong:?}");
        std::process::exit(1);
    }
}
