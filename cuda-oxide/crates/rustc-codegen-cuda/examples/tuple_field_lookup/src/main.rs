/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Reading one field of a tuple array element, without copying the array.
//!
//! A place read like `PAIRS[i].1` used to be lowered by loading the *whole
//! array* as a value and projecting into it, because `mir.field_addr` verified
//! struct pointees only. Each such read therefore spilled the entire table to a
//! fresh per-thread stack slot before touching one field, and a second field read
//! spilled it again. A tuple carries the same layout facts a struct does, so the
//! address path can now walk to the field and load just that.
//!
//! | kernel             | reads                                   |
//! |--------------------|-----------------------------------------|
//! | `tuple_one_field`  | `PAIRS[i].1`                            |
//! | `tuple_two_fields` | `let (a, b) = PAIRS[i]`                 |
//! | `scalar_control`   | `CODES[i]`, a plain `[u32; N]` table    |
//!
//! `scalar_control` reads a scalar table over the same index stream and should
//! not move at all: it never had an element to project into, so it is here to
//! show the measurement is sound rather than to be improved.
//!
//! Each thread performs `ROUNDS` dependent lookups against a per-thread LCG, so
//! the index is unpredictable and lane-divergent and the kernel is lookup-bound
//! rather than bandwidth-bound (4 bytes in, 4 bytes out per thread).
//!
//! Run: `cargo oxide run tuple_field_lookup`

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use cuda_device::cuda_module;

/// Threads, and `u32` elements in the input buffer.
const ELEMS: usize = 1 << 20;

/// Table entries. `(u8, u32)` is 8 bytes with three bytes of interior padding.
const N: usize = 256;

/// Lookups per thread.
const ROUNDS: u32 = 64;

/// Timed launches per kernel. Kept small: until this lowering was fixed, one
/// launch of the tuple kernels took the better part of a second.
const ITERS: usize = 10;

/// Untimed launches first, so clocks and caches settle.
const WARMUP: usize = 3;

const fn first_at(i: usize) -> u8 {
    (i & 0xff) as u8
}
const fn second_at(i: usize) -> u32 {
    (i as u32) * 7 + 1
}

const PAIRS: [(u8, u32); N] = {
    let mut t = [(0u8, 0u32); N];
    let mut i = 0;
    while i < N {
        t[i] = (first_at(i), second_at(i));
        i += 1;
    }
    t
};

const CODES: [u32; N] = {
    let mut t = [0u32; N];
    let mut i = 0;
    while i < N {
        t[i] = second_at(i);
        i += 1;
    }
    t
};

#[cuda_module]
mod kernels {
    use cuda_device::{DisjointSlice, kernel, thread};

    #[inline(always)]
    fn step(h: u32) -> u32 {
        h.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
    }

    #[kernel]
    pub fn tuple_one_field(input: &[u32], mut output: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let mut h = input[i];
        let mut acc = 0u32;
        for _ in 0..super::ROUNDS {
            h = step(h);
            acc = acc.wrapping_add(super::PAIRS[(h >> 24) as usize & (super::N - 1)].1);
        }
        if let Some(o) = output.get_mut(idx) {
            *o = acc;
        }
    }

    #[kernel]
    pub fn tuple_two_fields(input: &[u32], mut output: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let mut h = input[i];
        let mut acc = 0u32;
        for _ in 0..super::ROUNDS {
            h = step(h);
            let (a, b) = super::PAIRS[(h >> 24) as usize & (super::N - 1)];
            acc = acc.wrapping_add(b).wrapping_add(a as u32);
        }
        if let Some(o) = output.get_mut(idx) {
            *o = acc;
        }
    }

    #[kernel]
    pub fn scalar_control(input: &[u32], mut output: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let mut h = input[i];
        let mut acc = 0u32;
        for _ in 0..super::ROUNDS {
            h = step(h);
            acc = acc.wrapping_add(super::CODES[(h >> 24) as usize & (super::N - 1)]);
        }
        if let Some(o) = output.get_mut(idx) {
            *o = acc;
        }
    }
}

fn seed(i: usize) -> u32 {
    (i as u32).wrapping_mul(2_654_435_761).wrapping_add(12_345)
}

/// Host reference. `with_first` adds the tuple's first field too, matching
/// `tuple_two_fields`. Only wrapping adds, so a correct kernel matches exactly.
fn expected(i: usize, with_first: bool) -> u32 {
    let mut h = seed(i);
    let mut acc = 0u32;
    for _ in 0..ROUNDS {
        h = h.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let e = (h >> 24) as usize & (N - 1);
        acc = acc.wrapping_add(second_at(e));
        if with_first {
            acc = acc.wrapping_add(u32::from(first_at(e)));
        }
    }
    acc
}

struct Row {
    name: &'static str,
    reads: &'static str,
    us: f64,
    correct: bool,
}

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
    let ref_second: Vec<u32> = (0..ELEMS).map(|i| expected(i, false)).collect();
    let ref_both: Vec<u32> = (0..ELEMS).map(|i| expected(i, true)).collect();

    let mut rows: Vec<Row> = Vec::new();

    macro_rules! measure {
        ($name:literal, $reads:literal, $call:ident, $reference:expr) => {{
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
                reads: $reads,
                us,
                correct,
            });
        }};
    }

    measure!(
        "tuple_one_field",
        "PAIRS[i].1",
        tuple_one_field,
        &ref_second
    );
    measure!(
        "tuple_two_fields",
        "let (a,b)=PAIRS[i]",
        tuple_two_fields,
        &ref_both
    );
    measure!("scalar_control", "CODES[i]", scalar_control, &ref_second);

    println!();
    println!(
        "tuple_field_lookup -- {ELEMS} threads x {ROUNDS} divergent lookups into a \
         {N}-entry (u8, u32) table"
    );
    println!("{ITERS} timed launches after {WARMUP} warmup\n");
    println!(
        "{:<18} {:>21} {:>11} {:>12} {:>9}",
        "kernel", "reads", "us/launch", "Glookups/s", "correct"
    );
    println!("{:-<76}", "");
    for r in &rows {
        println!(
            "{:<18} {:>21} {:>11.1} {:>12.2} {:>9}",
            r.name,
            r.reads,
            r.us,
            glookups(r.us),
            if r.correct { "yes" } else { "NO" }
        );
    }

    let wrong: Vec<&str> = rows.iter().filter(|r| !r.correct).map(|r| r.name).collect();
    if wrong.is_empty() {
        println!("\n\u{2713} SUCCESS: all {} kernels bit-correct", rows.len());
    } else {
        println!("\n\u{2717} FAILED: incorrect output from {wrong:?}");
        std::process::exit(1);
    }
}
