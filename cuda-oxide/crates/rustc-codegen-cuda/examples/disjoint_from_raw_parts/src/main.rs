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
//! - `skip_partial` passes `DisjointSlice::empty()` and
//!   `DisjointSlice::default()` to helpers whose second output is optional.
//!   rustc folds the empty slice into a constant, which import has to
//!   rebuild as a slice value; the helpers then keep it across a call, by
//!   value and by `&mut`, and read its length.
//! - `skip_partial_consts` passes named empty-slice constants by value: a flat
//!   slice, a runtime-width slice whose image ends in padding, a tuple
//!   holding a slice beside a scalar, and an array of two slices. Each
//!   reaches import as a constant to decode, and the padding and the nested
//!   fields and elements are the shapes that need care. It also reads null and dangling raw pointers to a slice back from
//!   helpers, which stay plain address constants.
//!
//! Each element's expected value depends on its own index, so a kernel that
//! wrote a constant, or that addressed the wrong element, fails the check
//! rather than passing quietly.

use core::ptr::NonNull;
use cuda_core::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, Runtime2DIndex, cuda_module, device, kernel, thread};

const LEN: u32 = 1 << 12;
const BLOCK: u32 = 128;

/// Deliberately not a power of two, so a row width read as a length, or a
/// length read as a row width, lands somewhere visible.
const ROW_WIDTH: u32 = 37;
const ROWS: u32 = 24;
/// Added by `skip_partial_consts`, so its output differs from `skip_partial`'s
const TAG: u64 = 7;

/// An empty flat slice that `skip_partial_consts` passes as a named constant.
/// Its image is a dangling pointer and a length of zero.
const NO_PARTIAL: DisjointSlice<'static, f32> = DisjointSlice::empty();
/// An empty runtime-width slice: pointer, length and a row width of zero,
/// then four bytes of padding the decoder has to skip.
const NO_GRID: DisjointSlice<'static, f32, Runtime2DIndex> = DisjointSlice::empty();
/// An empty slice beside a scalar, so the slice is a field of a constant
/// tuple rather than the whole constant. `add_tag` reads both.
const TAGGED_PARTIAL: (DisjointSlice<'static, f32>, u64) = (DisjointSlice::empty(), TAG);
/// Two empty slices as the elements of a constant array, which import
/// decodes element by element. `mark_partials` takes the array by value.
const NO_PARTIALS: [DisjointSlice<'static, f32>; 2] =
    [DisjointSlice::empty(), DisjointSlice::empty()];

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

    /// Write this thread's index plus the partial buffer's length to `out`,
    /// and a marker to `partial` when it has a slot for this thread.
    #[device]
    fn index_with_partial<'a>(
        out: &mut DisjointSlice<'a, f32>,
        mut partial: DisjointSlice<'a, f32>,
    ) {
        let extra = partial.len() as f32;
        if let Some(slot) = partial.get_mut(thread::index_1d()) {
            *slot = -1.0;
        }
        if let Some((value, index)) = out.get_mut_indexed() {
            *value = index.get() as f32 + extra;
        }
    }

    /// Callers that do not want the partial output pass an empty slice, with
    /// no `unsafe` block at the call site. The runtime-width call must find
    /// no cell, because an empty slice carries a row width of zero.
    #[kernel]
    pub fn skip_partial(mut out: DisjointSlice<f32>) {
        index_with_partial(&mut out, DisjointSlice::empty());
        scale_row(&mut DisjointSlice::default(), 0.0);
    }

    /// Take the runtime-width slice by value, so a constant argument stays a
    /// constant at the call.
    #[device]
    fn scale_owned_row<'a>(mut grid: DisjointSlice<'a, f32, Runtime2DIndex>, factor: f32) {
        scale_row(&mut grid, factor);
    }

    /// Add the tag, plus the slice's length of zero, to this thread's element,
    /// and write a marker to the slice when it has a slot for this thread.
    #[device]
    fn add_tag<'a>(out: &mut DisjointSlice<'a, f32>, tagged: (DisjointSlice<'a, f32>, u64)) {
        let (mut partial, tag) = tagged;
        if let Some(slot) = partial.get_mut(thread::index_1d()) {
            *slot = -1.0;
        }
        if let Some(value) = out.get_mut(thread::index_1d()) {
            *value += tag as f32 + partial.len() as f32;
        }
    }

    /// Write a marker through each slice that has a slot for this thread.
    /// Kept out of line, so the array arrives as a constant at the call.
    #[device]
    #[inline(never)]
    fn mark_partials(partials: [DisjointSlice<'_, f32>; 2]) {
        let [mut first, mut second] = partials;
        if let Some(slot) = first.get_mut(thread::index_1d()) {
            *slot = -1.0;
        }
        if let Some(slot) = second.get_mut(thread::index_1d()) {
            *slot = -1.0;
        }
    }

    /// A null raw pointer to a slice: an address constant with no storage
    /// behind it, unlike a reference to a slice constant.
    #[device]
    #[inline(never)]
    fn null_grid() -> *mut DisjointSlice<'static, f32, Runtime2DIndex> {
        core::ptr::null_mut()
    }

    /// A dangling raw pointer to a slice, likewise only an address.
    #[device]
    #[inline(never)]
    fn dangling_grid() -> *mut DisjointSlice<'static, f32, Runtime2DIndex> {
        NonNull::dangling().as_ptr()
    }

    /// `skip_partial` with named constants in place of the calls. The tag is
    /// added only when both raw pointer constants come back as written.
    #[kernel]
    pub fn skip_partial_consts(mut out: DisjointSlice<f32>) {
        index_with_partial(&mut out, NO_PARTIAL);
        scale_owned_row(NO_GRID, 0.0);
        mark_partials(NO_PARTIALS);
        if null_grid().is_null() && !dangling_grid().is_null() {
            add_tag(&mut out, TAGGED_PARTIAL);
        }
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

    // Empty optional output: each element holds its own index, plus the
    // empty slice's length of zero, and nothing else was written.
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, LEN as usize)?;
    // SAFETY: a 1D launch over exactly LEN elements gives each thread one.
    unsafe { module.skip_partial(&stream, config, &mut out)? };
    stream.synchronize()?;
    let got = out.to_host_vec(&stream)?;
    for (i, value) in got.iter().enumerate() {
        let expected = i as f32;
        if *value != expected {
            return Err(
                format!("skip_partial: element {i} is {value}, expected {expected}").into(),
            );
        }
    }
    println!("skip_partial: {LEN} elements, empty optional output skipped");

    // Named constants: each element holds its own index plus the tag
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, LEN as usize)?;
    // SAFETY: a 1D launch over exactly LEN elements gives each thread one.
    unsafe { module.skip_partial_consts(&stream, config, &mut out)? };
    stream.synchronize()?;
    let got = out.to_host_vec(&stream)?;
    for (i, value) in got.iter().enumerate() {
        let expected = (i as u64 + TAG) as f32;
        if *value != expected {
            return Err(format!(
                "skip_partial_consts: element {i} is {value}, expected {expected}"
            )
            .into());
        }
    }
    println!("skip_partial_consts: {LEN} elements, constant empty slices skipped");

    println!("SUCCESS: disjoint slices built in-kernel behave like parameter slices");
    Ok(())
}
