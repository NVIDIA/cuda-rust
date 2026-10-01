/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression test for issue #231: projected reads from addressable local
//! arrays should walk the place to an address, then load once at the end.
//!
//! Before the fix, a read like `scratch[row][col]` loaded/copied the whole
//! `[4 x [2 x double]]` array into a temporary, then loaded/copied a
//! `[2 x double]` row temporary before the final scalar read.
//! The setup below uses a row assignment only to create a known value; the
//! regression under test is the final dynamic projected read.
//!
//! Run: cargo oxide run projected_place_read

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, device, kernel, thread};
use cuda_host::cuda_module;

#[derive(Clone, Copy)]
pub struct Point {
    x: f64,
    y: f64,
}

#[device]
#[inline(never)]
pub fn read_point(point: &Point) -> f64 {
    point.x + point.y
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Original repro from issue #231: 2D scratch buffer with dynamic indexing.
    ///
    /// scratch = [[0.0; 2]; 4], then scratch[1] = [3.0, 0.0]
    /// row = row_arg & 3, col = col_arg & 1  (dynamic indices)
    /// out[0] = scratch[row][col]  (projected read)
    ///
    /// `row_arg` and `col_arg` intentionally keep the indices dynamic so the
    /// read cannot be constant-folded. The expected lowering computes the row
    /// address, computes the column address, and emits one final scalar load.
    #[kernel]
    pub fn projected_read_2d(row_arg: usize, col_arg: usize, mut out: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        if idx.get() != 0 {
            return;
        }

        const ROWS: usize = 4;
        const COLS: usize = 2;

        let mut scratch = [[0.0_f64; COLS]; ROWS];
        scratch[1] = [3.0, 0.0];

        let row = row_arg & (ROWS - 1);
        let col = col_arg & (COLS - 1);
        let value = scratch[row][col];

        if let Some(slot) = out.get_mut(idx) {
            *slot = value;
        }
    }

    /// Struct field reads through a reference.
    ///
    /// This pins the broader read model after the #231 fix: field reads that
    /// have a real backing address should use `Deref -> Field`, then emit one
    /// final scalar load for each field.
    #[kernel]
    pub fn projected_read_struct_field(x_arg: usize, y_arg: usize, mut out: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        if idx.get() != 0 {
            return;
        }

        let point = Point {
            x: x_arg as f64,
            y: y_arg as f64,
        };
        let value = read_point(&point);

        if let Some(slot) = out.get_mut(idx) {
            *slot = value;
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load module");

    // row = 5 & 3 = 1; col = 6 & 1 = 0; out[0] = scratch[1][0]  => 3.0
    let mut out_dev = DeviceBuffer::<f64>::zeroed(&stream, 1).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.projected_read_2d(
            &stream,
            LaunchConfig::for_num_elems(1),
            5usize,
            6usize,
            &mut out_dev,
        )
    }
    .expect("projected_read_2d launch");

    let out = out_dev.to_host_vec(&stream).unwrap();
    assert!(
        (out[0] - 3.0_f64).abs() < 1e-12,
        "projected_read_2d: got {}, want 3.0",
        out[0]
    );

    // value = 6 + 5 = 11
    let mut out_dev = DeviceBuffer::<f64>::zeroed(&stream, 1).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.projected_read_struct_field(
            &stream,
            LaunchConfig::for_num_elems(1),
            6usize,
            5usize,
            &mut out_dev,
        )
    }
    .expect("projected_read_struct_field launch");

    let out = out_dev.to_host_vec(&stream).unwrap();
    assert!(
        (out[0] - 11.0_f64).abs() < 1e-12,
        "projected_read_struct_field: got {}, want 11.0",
        out[0]
    );

    println!("SUCCESS: projected place read produces correct results");
}
