// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * Regression coverage for rustc MIR `ProjectionElem::Subslice`.
 *
 * rustc emits Subslice for the `middle @ ..` part of array/slice patterns:
 *
 *     [head, middle @ .., tail]
 *
 * Arrays use `from_end = false` and produce a sized `[T; K]` place. Slices
 * use `from_end = true` and produce an unsized `[T]` place whose fat-pointer
 * metadata must become `old_len - from - to`.
 *
 * The helpers are `#[inline(never)]` so optimized MIR keeps the projection
 * in the callee instead of dissolving it into the kernel. One case slices a
 * local buffer at a lane-dependent length, so the PTX keeps the from-end
 * arithmetic that the constant-length cases fold away.
 *
 * Build/run:
 *     cargo oxide run subslice_projection
 *     cargo oxide pipeline subslice_projection
 */

use cuda_core::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

#[derive(Copy, Clone)]
struct Pair {
    x: u32,
    y: u32,
}

#[inline(never)]
fn middle_last_field(values: &[Pair]) -> u32 {
    let [_, middle @ .., _] = values else {
        return 0;
    };
    let [.., last] = middle else {
        return 0;
    };
    last.x + last.y
}

#[inline(never)]
fn array_middle_value(values: [u32; 4]) -> [u32; 2] {
    let [_, middle @ .., _] = values;
    middle
}

#[inline(never)]
fn array_middle_mut(values: &mut [u32; 4]) -> &mut [u32; 2] {
    let [_, middle @ .., _] = values;
    middle
}

#[inline(never)]
fn slice_middle(values: &[u32]) -> &[u32] {
    match values {
        [_, middle @ .., _] => middle,
        _ => values,
    }
}

#[inline(never)]
fn bump_slice_middle(values: &mut [u32]) {
    if let [_, middle @ .., _] = values {
        middle[0] += 9;
    }
}

// The two-step pattern is deliberate: `[_, middle @ .., _]` followed by
// `[.., last]` makes rustc emit a from-end `ConstantIndex` after a slice
// `Subslice`, `(*_1)[1:-1][-1 of 1]`. The single pattern `[_, .., last, _]`
// would index the original slice directly and skip the path under test.
#[inline(never)]
fn bump_slice_middle_last(values: &mut [u32]) {
    if let [_, middle @ .., _] = values
        && let [.., last] = middle
    {
        *last += 11;
    }
}

#[inline(never)]
fn bump_slice_middle_second_last(values: &mut [u32]) {
    if let [_, middle @ .., _] = values
        && let [.., second_last, _] = middle
    {
        *second_last += 13;
    }
}

#[inline(never)]
fn middle_last(values: &[u32]) -> u32 {
    let [_, middle @ .., _] = values else {
        return 0;
    };
    let [.., last] = middle else {
        return 0;
    };
    *last
}

/// Elements in the local buffer of the runtime-length case.
const BUFFER_LEN: usize = 8;

/// Slice length for the runtime-length case. It comes from the row data, which
/// the compiler cannot see, so the from-end index stays a runtime computation.
/// `make_input` puts `10 + i` in column 0, so lane `i` works on `3 + i`
/// elements: from a one-element middle, where only `[.., last]` matches, up to
/// the whole buffer.
fn runtime_len(row: [u32; 4]) -> usize {
    (row[0] - 7) as usize
}

/// Seeds the local buffer from `row`, runs the from-end helpers on its first
/// `n` elements (`n <= BUFFER_LEN`) and packs the outcome into one word: every
/// element's delta weighted by its 1-based position, times 1000, plus the
/// value `middle_last` reads back. A write to the wrong element, a collateral
/// write and a read from the wrong element each change the word. The kernel
/// and the host oracle both call this function, so the device result is
/// checked against the same helpers compiled for the host.
fn from_end_at_runtime_len(row: [u32; 4], n: usize) -> u32 {
    let mut values: [u32; BUFFER_LEN] = [
        row[0],
        row[1],
        row[2],
        row[3],
        row[0] + 50,
        row[1] + 50,
        row[2] + 50,
        row[3] + 50,
    ];
    let before = values;

    let window = &mut values[..n];
    bump_slice_middle_last(window);
    bump_slice_middle_second_last(window);
    let read = middle_last(&values[..n]);

    let mut weighted = 0;
    for k in 0..values.len() {
        weighted += (values[k] - before[k]) * (k as u32 + 1);
    }
    weighted * 1000 + read
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn test_slice_subslice_from_end_field(input: &[[u32; 4]], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }

        let row = input[i];
        let values = [
            Pair {
                x: row[0],
                y: row[0] + 100,
            },
            Pair {
                x: row[1],
                y: row[1] + 100,
            },
            Pair {
                x: row[2],
                y: row[2] + 100,
            },
            Pair {
                x: row[3],
                y: row[3] + 100,
            },
        ];

        if let Some(slot) = out.get_mut(idx) {
            *slot = middle_last_field(&values);
        }
    }

    #[kernel]
    pub fn test_array_subslice_value(input: &[[u32; 4]], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }

        let middle = array_middle_value(input[i]);
        if let Some(slot) = out.get_mut(idx) {
            *slot = middle[0] + middle[1];
        }
    }

    #[kernel]
    pub fn test_array_subslice_ref(input: &[[u32; 4]], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }

        // Keep this pattern directly on local storage. The local address is
        // an Erased+writable compiler carrier; the Subslice projection must
        // preserve that carrier until the final shared Reborrow boundary.
        let values = input[i];
        let [_, ref middle @ .., _] = values;
        if let Some(slot) = out.get_mut(idx) {
            *slot = middle[0] + middle[1];
        }
    }

    #[kernel]
    pub fn test_array_subslice_mut(input: &[[u32; 4]], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }

        let mut values = input[i];
        let before = values[1];
        let middle = array_middle_mut(&mut values);
        middle[0] += 7;

        if let Some(slot) = out.get_mut(idx) {
            // Proves the returned reference aliases the original array.
            *slot = values[1] - before;
        }
    }

    #[kernel]
    pub fn test_slice_subslice_metadata(input: &[[u32; 4]], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }

        let values = input[i];
        let middle = slice_middle(&values);
        if let Some(slot) = out.get_mut(idx) {
            // Checks both the advanced data pointer and rebuilt length metadata.
            *slot = (middle.len() as u32) * 1000 + middle[0];
        }
    }

    #[kernel]
    pub fn test_slice_subslice_from_end_mut(input: &[[u32; 4]], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }

        let mut values = input[i];
        let before = values;

        // Exercise two distinct from-end ConstantIndex offsets on the rebuilt
        // middle subslice: offset 1 selects original element 2, offset 2 selects
        // original element 1.
        bump_slice_middle_last(&mut values);
        bump_slice_middle_second_last(&mut values);

        if let Some(slot) = out.get_mut(idx) {
            let d0 = values[0] - before[0];
            let d1 = values[1] - before[1];
            let d2 = values[2] - before[2];
            let d3 = values[3] - before[3];

            // Pack every element's delta so collateral writes also fail the test.
            *slot = d0 | (d1 << 8) | (d2 << 16) | (d3 << 24);
        }
    }

    #[kernel]
    pub fn test_slice_subslice_from_end_runtime_len(
        input: &[[u32; 4]],
        mut out: DisjointSlice<u32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }

        // The slice length is only known at run time, so the from-end index
        // cannot fold to a constant element position. The kernel checks the
        // buffer bound itself; the host oracle has no such check and panics
        // on a bad length instead.
        let row = input[i];
        let n = runtime_len(row);
        if let Some(slot) = out.get_mut(idx) {
            *slot = if n <= BUFFER_LEN {
                from_end_at_runtime_len(row, n)
            } else {
                u32::MAX
            };
        }
    }

    #[kernel]
    pub fn test_slice_subslice_mut(input: &[[u32; 4]], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }

        let mut values = input[i];
        let before = values[1];
        bump_slice_middle(&mut values);

        if let Some(slot) = out.get_mut(idx) {
            // Proves mutable slice subslices write through to original storage.
            *slot = values[1] - before;
        }
    }
}

/// Six lanes: column 0 of the rows runs from 10 to 15, so the runtime-length
/// case sees every length from 3 to 8.
const N: usize = 6;

fn input_row(i: usize) -> [u32; 4] {
    let i = i as u32;
    [10 + i, 20 + i, 30 + i, 40 + i]
}

fn make_input() -> Vec<[u32; 4]> {
    (0..N).map(input_row).collect()
}

fn run_and_report<F, E>(name: &str, stream: &Arc<CudaStream>, launch: F, expected: E) -> bool
where
    F: FnOnce(&Arc<CudaStream>, LaunchConfig, &DeviceBuffer<[u32; 4]>, &mut DeviceBuffer<u32>),
    E: Fn(usize) -> u32,
{
    let input = make_input();
    let dev_in = DeviceBuffer::from_host(stream, &input).unwrap();
    let mut dev_out = DeviceBuffer::<u32>::zeroed(stream, N).unwrap();

    launch(
        stream,
        LaunchConfig::for_num_elems(N as u32),
        &dev_in,
        &mut dev_out,
    );

    let host_out = dev_out.to_host_vec(stream).unwrap();
    let expected_values: Vec<u32> = (0..N).map(expected).collect();
    let pass = host_out == expected_values;

    println!(
        "  {name:<32} {}  got={host_out:?} expected={expected_values:?}",
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

fn main() {
    println!("=== subslice_projection: ProjectionElem::Subslice ===\n");

    let ctx = CudaContext::new(0).expect("CUDA init");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Load embedded PTX");

    let mut all_pass = true;

    all_pass &= run_and_report(
        "slice subslice from-end field",
        &stream,
        |s, cfg, i, o| {
            // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
            unsafe { module.test_slice_subslice_from_end_field(s, cfg, i, o) }.expect("launch")
        },
        |i| 160 + 2 * i as u32,
    );

    all_pass &= run_and_report(
        "array value",
        &stream,
        |s, cfg, i, o| {
            // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
            unsafe { module.test_array_subslice_value(s, cfg, i, o) }.expect("launch")
        },
        |i| 50 + 2 * i as u32,
    );

    all_pass &= run_and_report(
        "array shared ref",
        &stream,
        |s, cfg, i, o| {
            // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
            unsafe { module.test_array_subslice_ref(s, cfg, i, o) }.expect("launch")
        },
        |i| 50 + 2 * i as u32,
    );

    all_pass &= run_and_report(
        "array mutable ref",
        &stream,
        |s, cfg, i, o| {
            // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
            unsafe { module.test_array_subslice_mut(s, cfg, i, o) }.expect("launch")
        },
        |_| 7,
    );

    all_pass &= run_and_report(
        "slice metadata",
        &stream,
        |s, cfg, i, o| {
            // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
            unsafe { module.test_slice_subslice_metadata(s, cfg, i, o) }.expect("launch")
        },
        |i| 2020 + i as u32,
    );

    all_pass &= run_and_report(
        "slice mutable ref",
        &stream,
        |s, cfg, i, o| {
            // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
            unsafe { module.test_slice_subslice_mut(s, cfg, i, o) }.expect("launch")
        },
        |_| 9,
    );

    all_pass &= run_and_report(
        "slice subslice from-end mutable",
        &stream,
        |s, cfg, i, o| {
            // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
            unsafe { module.test_slice_subslice_from_end_mut(s, cfg, i, o) }.expect("launch")
        },
        |_| (13 << 8) | (11 << 16),
    );

    all_pass &= run_and_report(
        "slice subslice from-end runtime",
        &stream,
        |s, cfg, i, o| {
            // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
            unsafe { module.test_slice_subslice_from_end_runtime_len(s, cfg, i, o) }
                .expect("launch")
        },
        |i| {
            let row = input_row(i);
            from_end_at_runtime_len(row, runtime_len(row))
        },
    );

    if all_pass {
        println!("\nSUCCESS: all Subslice projection cases passed");
    } else {
        eprintln!("\nFAILURE: at least one Subslice projection case failed");
        std::process::exit(1);
    }
}
