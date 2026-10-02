/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression tests for the runtime row width carried inside
//! `DisjointSlice<T, Runtime2DIndex>`.
//!
//! Three properties, each of which failed loudly at some point in the
//! runtime-width slice design:
//!
//! 1. **Nonzero width readback**: the host binds a row width via
//!    `cuda_host::RowWidth`, and every device thread must read that exact
//!    value back. An entry prologue that dropped the third kernel parameter
//!    compiled and ran while giving every thread width 0; only checking a
//!    NONZERO value catches that.
//!
//! 2. **Two-width witness mixing stays sound**: a `Runtime2DIndex` witness
//!    carries the thread's `(row, col)` coordinates, and `get_mut` resolves
//!    them against the ADDRESSED slice's own row width. Safe code that mints
//!    a witness from each of two slices with different widths and selects one
//!    under a thread-varying condition must still write each thread to its
//!    own cell of the addressed grid. A flat-index witness fails this test:
//!    the minting slice's grid would leak through the selection.
//!
//! 3. **By-value runtime-width slice across a non-inlined call boundary**:
//!    passing a runtime-width `DisjointSlice` by value to an
//!    `#[inline(never)]` device helper must marshal all three fields
//!    (ptr, len, width) through the internal call ABI, matching the
//!    three-parameter callee signature.
//!
//! Run: cargo oxide run row_width_slice

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_host::cuda_module;

const SENTINEL: u32 = 0xDEAD_BEEF;

#[cuda_module]
mod kernels {
    use cuda_device::thread::{Runtime2DIndex, ThreadIndex};
    use cuda_device::{DisjointSlice, kernel, thread};

    /// Property 1: every in-grid thread writes the row width it read back
    /// from the slice. A dropped or zeroed width mints no witness at all
    /// (width 0 resolves nothing), leaving the sentinel behind for the host
    /// to catch.
    #[kernel]
    pub fn write_width_readback(mut out: DisjointSlice<u32, Runtime2DIndex>) {
        let width = out.row_width();
        if let Some(idx) = thread::index_2d_runtime(&out)
            && let Some(cell) = out.get_mut(idx)
        {
            *cell = width;
        }
    }

    /// Property 2: mint witnesses from BOTH slices (row widths 5 and 100),
    /// select one under a thread-varying condition, and address `b` through
    /// whichever witness won. Per-slice resolution must land every thread on
    /// `b`'s own cell `(row, col)`, so writing the expected flat index makes
    /// any grid leakage or aliasing a value mismatch the host can see.
    #[kernel]
    pub fn two_width_selection(
        a: DisjointSlice<u32, Runtime2DIndex>,
        mut b: DisjointSlice<u32, Runtime2DIndex>,
    ) {
        let row = thread::index_2d_row();
        let col = thread::index_2d_col();
        let b_width = b.row_width() as usize;
        let wa = thread::index_2d_runtime(&a);
        let wb = thread::index_2d_runtime(&b);
        if let (Some(wa), Some(wb)) = (wa, wb) {
            // Thread-varying selection between two witnesses of one type:
            // exactly the shape that aliased flat-index witnesses.
            let chosen = if (row + col).is_multiple_of(2) {
                wa
            } else {
                wb
            };
            if let Some(cell) = b.get_mut(chosen) {
                *cell = (row * b_width + col) as u32;
            }
        }
    }

    /// Property 3 callee: takes the runtime-width slice BY VALUE.
    /// `inline(never)` keeps the call visible to the device pipeline, so the
    /// three-field slice must survive the internal call ABI intact.
    #[inline(never)]
    fn write_width_by_value(
        mut c: DisjointSlice<u32, Runtime2DIndex>,
        idx: ThreadIndex<Runtime2DIndex>,
    ) {
        let width = c.row_width();
        if let Some(cell) = c.get_mut(idx) {
            *cell = width;
        }
    }

    /// Property 3 caller: mints the witness, then moves the slice and the
    /// witness across the non-inlined call boundary.
    #[kernel]
    pub fn byvalue_helper_width(out: DisjointSlice<u32, Runtime2DIndex>) {
        if let Some(idx) = thread::index_2d_runtime(&out) {
            write_width_by_value(out, idx);
        }
    }
}

fn launch_cfg(grid: (u32, u32), block: (u32, u32)) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (grid.0, grid.1, 1),
        block_dim: (block.0, block.1, 1),
        shared_mem_bytes: 0,
    }
}

fn device_buffer_of_sentinels(stream: &cuda_core::CudaStream, len: usize) -> DeviceBuffer<u32> {
    DeviceBuffer::from_host(stream, &vec![SENTINEL; len]).expect("sentinel buffer")
}

fn main() {
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load module");

    // ── Property 1: nonzero width readback ─────────────────────────────
    const WIDTH: u32 = 37;
    const ROWS: usize = 8;
    let len = WIDTH as usize * ROWS;
    let mut out = device_buffer_of_sentinels(&stream, len);
    // SAFETY: 2D launch; the kernel bounds itself by the slice's width/len.
    unsafe {
        module.write_width_readback(
            &stream,
            launch_cfg((3, 1), (16, 16)),
            cuda_host::RowWidth::new(&mut out, WIDTH),
        )
    }
    .expect("write_width_readback launch");
    let host = out.to_host_vec(&stream).unwrap();
    for (i, &v) in host.iter().enumerate() {
        assert_eq!(
            v, WIDTH,
            "out[{i}]: got {v:#x}, want width {WIDTH}; a zero or dropped row width reached the device"
        );
    }
    println!("width readback: all {len} cells saw row width {WIDTH}");

    // ── Property 2: two-width witness mixing ───────────────────────────
    const WIDTH_A: u32 = 5;
    const WIDTH_B: u32 = 100;
    const ROWS_B: usize = 4;
    let mut a = device_buffer_of_sentinels(&stream, WIDTH_A as usize * ROWS_B);
    let mut b = device_buffer_of_sentinels(&stream, WIDTH_B as usize * ROWS_B);
    // SAFETY: 2D launch; each thread writes at most one cell of `b`.
    unsafe {
        module.two_width_selection(
            &stream,
            launch_cfg((1, 1), (16, 16)),
            cuda_host::RowWidth::new(&mut a, WIDTH_A),
            cuda_host::RowWidth::new(&mut b, WIDTH_B),
        )
    }
    .expect("two_width_selection launch");
    let host_a = a.to_host_vec(&stream).unwrap();
    let host_b = b.to_host_vec(&stream).unwrap();
    for (i, &v) in host_a.iter().enumerate() {
        assert_eq!(v, SENTINEL, "a[{i}] was written; `a` must stay untouched");
    }
    // Threads with col < 5 hold both witnesses; whichever they select must
    // resolve against b's row width of 100. Everything else stays sentinel.
    for row in 0..16usize {
        for col in 0..WIDTH_B as usize {
            let flat = row * WIDTH_B as usize + col;
            if flat >= host_b.len() {
                continue;
            }
            let expected = if col < WIDTH_A as usize && row < ROWS_B {
                flat as u32
            } else {
                SENTINEL
            };
            assert_eq!(
                host_b[flat], expected,
                "b[{flat}] (row {row}, col {col}): got {:#x}, want {expected:#x}; \
                 a witness resolved against the wrong slice's row width",
                host_b[flat]
            );
        }
    }
    println!("two-width selection: every thread landed on b's own (row, col) cell");

    // ── Property 3: by-value runtime-width slice through a helper ──────
    const WIDTH_C: u32 = 13;
    const ROWS_C: usize = 4;
    let len_c = WIDTH_C as usize * ROWS_C;
    let mut c = device_buffer_of_sentinels(&stream, len_c);
    // SAFETY: 2D launch; the helper bounds itself by the slice's width/len.
    unsafe {
        module.byvalue_helper_width(
            &stream,
            launch_cfg((1, 1), (16, 16)),
            cuda_host::RowWidth::new(&mut c, WIDTH_C),
        )
    }
    .expect("byvalue_helper_width launch");
    let host_c = c.to_host_vec(&stream).unwrap();
    for (i, &v) in host_c.iter().enumerate() {
        assert_eq!(
            v, WIDTH_C,
            "c[{i}]: got {v:#x}, want width {WIDTH_C}; the by-value call ABI dropped the row width"
        );
    }
    println!("by-value helper: row width {WIDTH_C} survived the internal call ABI");

    println!("SUCCESS: runtime row width bound, resolved per-slice, and marshalled by value");
}
