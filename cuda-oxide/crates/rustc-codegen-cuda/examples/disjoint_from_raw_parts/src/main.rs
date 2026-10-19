/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Building a `DisjointSlice` inside a kernel, for both index-space shapes.
//!
//! `DisjointSlice::from_raw_parts` writes a struct literal. The literal
//! usually folds away before import, and it survives when the slice crosses a
//! call the optimiser keeps, such as a `#[device]` helper taking `&mut
//! DisjointSlice`. Import then met an aggregate whose type is the slice's own
//! rather than a struct, and refused it as a scalar-lowered ADT with no
//! runtime field (issue #667).
//!
//! The two kernels below cover the shapes that differ in construction:
//!
//! - `increment_from_raw_parts` builds the two-word slice, whose index space
//!   carries no runtime layout.
//! - `scale_row_width_slice` builds the three-word form, where the row width
//!   read at the access site is the third operand, and a width written into
//!   the length slot would resolve every row somewhere else.
//!
//! Each element's expected value depends on its own index, so a kernel that
//! wrote a constant, or that addressed the wrong element, fails the check
//! rather than passing quietly.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, Runtime2DIndex, cuda_module, device, kernel, thread};

const LEN: u32 = 1 << 12;
const BLOCK: u32 = 128;

/// Deliberately not a power of two, so a row width read as a length, or a
/// length read as a row width, lands somewhere visible.
const ROW_WIDTH: u32 = 37;
const ROWS: u32 = 24;

#[cuda_module]
mod kernels {
    use super::*;

    /// Take the slice by `&mut` through a call, which is what keeps the
    /// literal from folding into its use.
    #[device]
    fn add_one<'a>(vector: &mut DisjointSlice<'a, f32>) {
        if let Some(value) = vector.get_mut(thread::index_1d()) {
            *value += 1.0;
        }
    }

    /// Rebuild the slice from a raw pointer and length inside the kernel.
    ///
    /// # Safety
    ///
    /// `ptr` addresses `len` writable elements, and the launch gives each
    /// thread one of them.
    #[kernel]
    pub unsafe fn increment_from_raw_parts(ptr: *mut f32, len: usize) {
        let mut view = unsafe { DisjointSlice::from_raw_parts(ptr, len) };
        add_one(&mut view);
    }

    /// Scale through a slice whose row width is a runtime value.
    #[device]
    fn scale_row<'a>(grid: &mut DisjointSlice<'a, f32, Runtime2DIndex>, factor: f32) {
        let Some(index) = thread::index_2d_runtime(grid) else {
            return;
        };
        if let Some(value) = grid.get_mut(index) {
            *value *= factor;
        }
    }

    /// Rebuild the three-word slice, row width included.
    ///
    /// # Safety
    ///
    /// `ptr` addresses `len` writable elements and `row_width` describes the
    /// grid the launch covers.
    #[kernel]
    pub unsafe fn scale_row_width_slice(ptr: *mut f32, len: usize, row_width: u32) {
        let mut grid = unsafe { DisjointSlice::from_raw_parts_with_space(ptr, len, row_width) };
        scale_row(&mut grid, 2.0);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    // Two-word slice: every element carries its own index, so a constant
    // write or a misaddressed element shows up as a mismatch.
    let host: Vec<f32> = (0..LEN).map(|i| i as f32).collect();
    let data = DeviceBuffer::from_host(&stream, &host)?;
    let config = LaunchConfig {
        grid_dim: (LEN.div_ceil(BLOCK), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: the grid covers exactly LEN elements and the buffer holds them.
    unsafe {
        module.increment_from_raw_parts(
            &stream,
            config,
            data.cu_deviceptr() as *mut f32,
            data.len(),
        )?;
    }
    stream.synchronize()?;
    let got = data.to_host_vec(&stream)?;
    for (i, value) in got.iter().enumerate() {
        let expected = host[i] + 1.0;
        if *value != expected {
            return Err(format!(
                "increment_from_raw_parts: element {i} is {value}, expected {expected}"
            )
            .into());
        }
    }
    println!("increment_from_raw_parts: {LEN} elements, exact match");

    // Three-word slice: the row width is a runtime value the kernel rebuilds
    // into the slice it addresses.
    let grid_len = (ROWS * ROW_WIDTH) as usize;
    let grid_host: Vec<f32> = (0..grid_len).map(|i| i as f32).collect();
    let grid = DeviceBuffer::from_host(&stream, &grid_host)?;
    let grid_config = LaunchConfig {
        grid_dim: (ROW_WIDTH.div_ceil(32), ROWS, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: the grid covers every cell of the ROWS by ROW_WIDTH region once,
    // and the buffer holds exactly that many elements.
    unsafe {
        module.scale_row_width_slice(
            &stream,
            grid_config,
            grid.cu_deviceptr() as *mut f32,
            grid.len(),
            ROW_WIDTH,
        )?;
    }
    stream.synchronize()?;
    let scaled = grid.to_host_vec(&stream)?;
    for (i, value) in scaled.iter().enumerate() {
        let expected = grid_host[i] * 2.0;
        if *value != expected {
            return Err(format!(
                "scale_row_width_slice: element {i} is {value}, expected {expected}"
            )
            .into());
        }
    }
    println!("scale_row_width_slice: {grid_len} elements at row width {ROW_WIDTH}, exact match");

    println!("SUCCESS: disjoint slices built in-kernel behave like parameter slices");
    Ok(())
}
