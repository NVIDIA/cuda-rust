/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression coverage for MIR `ConstantIndex { from_end: true }` on slices.
//!
//! The important distinction is that rustc resolves from-end indexing for
//! fixed-size arrays before the MIR importer sees it. A runtime-length slice
//! retains `from_end: true`, and codegen must materialize `slice.len() - offset`.
//!
//! Covered shapes:
//! - last element of `&[u32]` (`offset = 1`),
//! - penultimate element of `&[u32]` (`offset = 2`),
//! - last row of `&[[u32; 3]]`, guarding against confusing the row width with
//!   the outer slice length,
//! - mutable last element of `&mut [u32]`, exercising the address/store path.
//!
//! Run with:
//!   CUDA_OXIDE_DUMP_MIR=1 cargo oxide run constant_index_from_end

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

const THREADS: usize = 32;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn read_last(input: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            if let [.., last] = *input {
                *out_elem = last;
            } else {
                *out_elem = u32::MAX;
            }
        }
    }

    #[kernel]
    pub fn read_penultimate(input: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            if let [.., penultimate, _] = *input {
                *out_elem = penultimate;
            } else {
                *out_elem = u32::MAX;
            }
        }
    }

    /// The element type is itself an array. The from-end index must use the
    /// OUTER slice length, not the inner row width (`3`).
    #[kernel]
    pub fn read_last_row(rows: &[[u32; 3]], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            if let [.., row] = *rows {
                *out_elem = row[2];
            } else {
                *out_elem = u32::MAX;
            }
        }
    }

    /// Mutable/address path: the destination must be the original slice's last
    /// element, not a copied value.
    #[kernel]
    pub fn write_last(values: &mut [u32], value: u32) {
        if let [.., ref mut last] = *values {
            *last = value;
        }
    }
}

fn check_all(name: &str, values: &[u32], expected: u32) -> usize {
    let mut errors = 0;
    for (i, &value) in values.iter().enumerate() {
        if value != expected {
            if errors < 5 {
                eprintln!("  FAIL {name}[{i}]: got {value}, expected {expected}");
            }
            errors += 1;
        }
    }
    errors
}

fn main() {
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load module");

    let input_host = vec![11u32, 22, 33, 44, 55];
    let input_dev = DeviceBuffer::from_host(&stream, &input_host).unwrap();

    let rows_host = vec![
        [10u32, 11, 12],
        [20u32, 21, 22],
        [30u32, 31, 32],
        [40u32, 41, 42],
    ];
    let rows_dev = DeviceBuffer::from_host(&stream, &rows_host).unwrap();

    let mut errors = 0usize;

    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, THREADS).unwrap();
    // SAFETY: every thread reads a shared input slice and owns one output slot.
    unsafe {
        module.read_last(
            &stream,
            LaunchConfig::for_num_elems(THREADS as u32),
            &input_dev,
            &mut out_dev,
        )
    }
    .expect("read_last launch");
    errors += check_all(
        "read_last",
        &out_dev.to_host_vec(&stream).unwrap(),
        *input_host.last().unwrap(),
    );

    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, THREADS).unwrap();
    // SAFETY: every thread reads a shared input slice and owns one output slot.
    unsafe {
        module.read_penultimate(
            &stream,
            LaunchConfig::for_num_elems(THREADS as u32),
            &input_dev,
            &mut out_dev,
        )
    }
    .expect("read_penultimate launch");
    errors += check_all(
        "read_penultimate",
        &out_dev.to_host_vec(&stream).unwrap(),
        input_host[input_host.len() - 2],
    );

    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, THREADS).unwrap();
    // SAFETY: every thread reads a shared row slice and owns one output slot.
    unsafe {
        module.read_last_row(
            &stream,
            LaunchConfig::for_num_elems(THREADS as u32),
            &rows_dev,
            &mut out_dev,
        )
    }
    .expect("read_last_row launch");
    errors += check_all(
        "read_last_row",
        &out_dev.to_host_vec(&stream).unwrap(),
        rows_host.last().unwrap()[2],
    );

    let values_host = vec![1u32, 2, 3, 4, 5];
    let mut values_dev = DeviceBuffer::from_host(&stream, &values_host).unwrap();
    const NEW_LAST: u32 = 99;
    // SAFETY: one thread performs the single mutable write under test.
    unsafe {
        module.write_last(
            &stream,
            LaunchConfig::for_num_elems(1),
            &mut values_dev,
            NEW_LAST,
        )
    }
    .expect("write_last launch");
    let values = values_dev.to_host_vec(&stream).unwrap();
    if values[..values.len() - 1] != values_host[..values_host.len() - 1]
        || values.last().copied() != Some(NEW_LAST)
    {
        eprintln!(
            "  FAIL write_last: got {:?}, expected {:?} with last={NEW_LAST}",
            values, values_host
        );
        errors += 1;
    }

    if errors == 0 {
        println!("PASS: constant_index_from_end");
    } else {
        eprintln!("FAIL: constant_index_from_end ({errors} errors)");
        std::process::exit(1);
    }
}
