/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Block-per-row LayerNorm with no raw indexing, out of place and in place.
//!
//! `DisjointSlice<T, BlockRows>` gives block `b` row `b` and thread `t` the
//! columns `t, t + blockDim.x, ...` of it. The host binds the row width with
//! `cuda_host::RowWidth`, so the device reads one width for every block.
//!
//! - `layernorm` reads `x` and writes `out`, reducing the row with
//!   `block_reduce` before each thread writes its columns.
//! - `layernorm_inplace` normalizes a buffer over itself. The buffer cannot be
//!   both `&[f32]` and `DisjointSlice<f32>`, so every read and write goes
//!   through the column witness.
//!
//! The row width is not a multiple of the block size, so the last column pass
//! is partial. The output buffer also ends in a partial row, and the grid has
//! a block past the last row. Both of those blocks get `None` from
//! `block_row32` and return together, and the host checks that the partial
//! row was left alone.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::cooperative_groups::{ThreadGroup, block_reduce, ops::Sum, this_thread_block};
use cuda_device::{
    BlockRows, DisjointSlice, SharedArray, cuda_module, kernel, launch_bounds, launch_contract,
};

const BLOCK: u32 = 256;
/// Warp totals `block_reduce` needs for a `BLOCK`-thread block.
const WARPS: usize = (BLOCK / 32) as usize;
/// Not a multiple of `BLOCK`, so each thread's last column pass is partial.
const COLS: u32 = 1000;
const ROWS: u32 = 37;
/// Elements past the last whole row: a partial row that no block may touch.
const TAIL: usize = 5;
const EPS: f32 = 1e-5;
const SENTINEL: f32 = 12345.0;

#[cuda_module]
mod kernels {
    use super::*;

    /// Normalize each row of `x` into the same row of `out`.
    #[kernel(launch_context = lc)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub fn layernorm(mut out: DisjointSlice<f32, BlockRows>, x: &[f32], eps: f32) {
        static mut SMEM: SharedArray<f32, WARPS> = SharedArray::UNINIT;

        // uniform within the block, because every thread calls this on the
        // unchanged parameter the host bound, so the whole block returns or
        // none does
        let Some(mut row) = out.block_row32(lc) else {
            return;
        };
        let width = row.width() as usize;
        let base = row.row() as usize * width;
        // also uniform: every thread of the block compares the same values
        if base + width > x.len() {
            return;
        }
        let block = this_thread_block();
        let n = row.width() as f32;

        let mut sum = 0.0f32;
        for y in row.thread_columns() {
            sum += x[base + y.column() as usize];
        }
        // SAFETY: every thread of the block reaches this call (both early
        // returns above are uniform per block), and only these calls use SMEM
        let mean = unsafe { block_reduce::<f32, Sum, WARPS>(&block, sum, &raw mut SMEM) } / n;
        block.sync();

        let mut squares = 0.0f32;
        for y in row.thread_columns() {
            let d = x[base + y.column() as usize] - mean;
            squares += d * d;
        }
        // SAFETY: as above, and the barrier separates the reuse of SMEM
        let var = unsafe { block_reduce::<f32, Sum, WARPS>(&block, squares, &raw mut SMEM) } / n;
        let rstd = 1.0 / (var + eps).sqrt();

        for mut y in row.thread_columns() {
            let v = x[base + y.column() as usize];
            y.write((v - mean) * rstd);
        }
    }

    /// Normalize each row of `xy` over itself.
    #[kernel(launch_context = lc)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub fn layernorm_inplace(mut xy: DisjointSlice<f32, BlockRows>, eps: f32) {
        static mut SMEM: SharedArray<f32, WARPS> = SharedArray::UNINIT;

        // uniform within the block, as in `layernorm`: `xy` is the unchanged
        // host-bound parameter
        let Some(mut row) = xy.block_row32(lc) else {
            return;
        };
        let block = this_thread_block();
        let n = row.width() as f32;

        let mut sum = 0.0f32;
        for y in row.thread_columns() {
            sum += y.read();
        }
        // SAFETY: every thread of the block reaches this call (the early
        // return above is uniform per block), and only these calls use SMEM
        let mean = unsafe { block_reduce::<f32, Sum, WARPS>(&block, sum, &raw mut SMEM) } / n;
        block.sync();

        let mut squares = 0.0f32;
        for y in row.thread_columns() {
            let d = y.read() - mean;
            squares += d * d;
        }
        // SAFETY: as above, and the barrier separates the reuse of SMEM
        let var = unsafe { block_reduce::<f32, Sum, WARPS>(&block, squares, &raw mut SMEM) } / n;
        let rstd = 1.0 / (var + eps).sqrt();

        // each thread rewrites only the columns it read, so no barrier is
        // needed between the reads above and these writes
        for mut y in row.thread_columns() {
            let v = y.read();
            y.write((v - mean) * rstd);
        }
    }
}

fn input(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 7919) % 1013) as f32 / 101.0 - 5.0)
        .collect()
}

fn reference(x: &[f32]) -> Vec<f32> {
    let cols = COLS as usize;
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(cols) {
        let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / cols as f64;
        let var = row
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / cols as f64;
        let rstd = 1.0 / (var + f64::from(EPS)).sqrt();
        out.extend(row.iter().map(|&v| ((f64::from(v) - mean) * rstd) as f32));
    }
    out
}

/// Compare the whole rows with the reference and check the tail is untouched.
fn check(name: &str, got: &[f32], expected: &[f32]) -> Result<(), String> {
    if got.len() != expected.len() + TAIL {
        return Err(format!(
            "{name}: read back {} elements, expected {}",
            got.len(),
            expected.len() + TAIL
        ));
    }
    let (rows, tail) = got.split_at(expected.len());
    for (i, (g, e)) in rows.iter().zip(expected).enumerate() {
        // a NaN fails every comparison, so the tolerance test alone would
        // pass it
        if !g.is_finite() || (g - e).abs() > 1e-4 {
            return Err(format!("{name}: element {i} is {g}, expected {e}"));
        }
    }
    if let Some((i, v)) = tail.iter().enumerate().find(|(_, v)| **v != SENTINEL) {
        return Err(format!(
            "{name}: partial-row element {i} is {v}, but no block owns that row"
        ));
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: the embedded module is the one built from this crate's
    // `kernels`, so every generated launch method matches its kernel
    let module = unsafe { kernels::load(&ctx) }?;

    let whole = (ROWS * COLS) as usize;
    let x_host = input(whole);
    let expected = reference(&x_host);
    // one block per whole row, one for the partial row, and one past the end
    let config = LaunchConfig1D::new(ROWS + 2, BLOCK, 0);

    // ---- out of place ----------------------------------------------------
    let x = DeviceBuffer::from_host(&stream, &x_host)?;
    let mut out = DeviceBuffer::from_host(&stream, &vec![SENTINEL; whole + TAIL])?;
    let prepared = module.prepare_layernorm(config)?;
    module.layernorm(
        &stream,
        &prepared,
        cuda_host::RowWidth::new(&mut out, COLS),
        &x,
        EPS,
    )?;
    stream.synchronize()?;
    check("layernorm", &out.to_host_vec(&stream)?, &expected)?;
    println!("layernorm: {ROWS} rows of {COLS} match the reference");

    // ---- in place --------------------------------------------------------
    let mut xy_host = x_host.clone();
    xy_host.extend([SENTINEL; TAIL]);
    let mut xy = DeviceBuffer::from_host(&stream, &xy_host)?;
    let prepared = module.prepare_layernorm_inplace(config)?;
    module.layernorm_inplace(
        &stream,
        &prepared,
        cuda_host::RowWidth::new(&mut xy, COLS),
        EPS,
    )?;
    stream.synchronize()?;
    check("layernorm_inplace", &xy.to_host_vec(&stream)?, &expected)?;
    println!("layernorm_inplace: {ROWS} rows of {COLS} match the reference");

    println!("\nSUCCESS: block-owned rows normalized with no raw indexing");
    Ok(())
}
