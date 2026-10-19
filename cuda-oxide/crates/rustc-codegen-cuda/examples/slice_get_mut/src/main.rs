// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The indexed loops and the `Option::map` write are the MIR shapes under
// test (issue #58); the iterator rewrites clippy suggests would dissolve
// the very projections this example guards.
#![allow(clippy::needless_range_loop, clippy::option_map_unit_fn)]

/*
 * Mutable element writes through slice-shaped (fat) pointers.
 *
 * A `&mut [T]` is a FAT pointer: at the ABI level it is a (data pointer,
 * length) pair, not a single address. Rust's `slice::get_mut` inlines to a
 * bounds check plus the place `&mut (*fat)[i]`, i.e. the projection chain
 * `[Deref, Index(i)]` over a fat-pointer local. The importer's address
 * walker must extract the thin data pointer (field 0 of the pair) before
 * indexing; treating the fat value as a thin address would be a
 * miscompile, and refusing it outright rejected valid kernels.
 *
 * Kernels cover every write shape from issue #58:
 *
 *     a.get_mut(i).map(|e| *e = v)      // the issue's exact shape
 *     if let Some(e) = a.get_mut(i)     // same projection, if-let form
 *     a[i] = v   (a: &mut [f32; N])     // [Deref, Index] assignment, thin
 *     s[i] = v   (s: &mut [f32])        // [Deref, Index] assignment, fat
 *     let e = &mut a[i]; *e = v         // pre-existing working shape
 *     cells.get_mut(i) -> &mut Cell,    // fat walk composed with the
 *     then c.lo = v                     // (Deref, Field) field write
 *     let e = &mut rows[row][0]; *e = v // fat slice element is an array
 *     let e = &mut rows[1][0]; *e = v   // same, with ConstantIndex row
 *     rows[row][0] = v                  // same shape as projected assignment
 *
 * Each kernel writes a distinct, index-dependent pattern; the host reads
 * the buffer back and checks every lane, so a write that lands in a
 * temporary copy (or in lane 0) fails loudly.
 *
 * Build:
 *     cargo oxide run slice_get_mut
 *
 * Guards the fix for issue #58 (writes through `get_mut` on a
 * `&mut [f32; N]` were rejected with "cannot compute a mutable in-memory
 * address through fat-pointer deref").
 */

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

const SIZE: usize = 8;
const N: usize = 4;

/// Two-field element for the field-write-through-fat-pointer kernel.
#[repr(C)]
#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct Cell {
    pub lo: f32,
    pub hi: f32,
}

// Plain pair of f32s with no padding or pointers: safe to memcpy to the
// device.
unsafe impl cuda_core::DeviceCopy for Cell {}

#[cuda_module]
mod kernels {
    use super::*;

    /// The exact issue #58 kernel shape: `a.get_mut(i).map(|e| *e = 42.)`
    /// over `a: &mut [f32; SIZE]`. The inlined `slice::get_mut` body takes
    /// `&mut (*fat)[i]` through a fat `&mut [f32]` local.
    #[kernel]
    pub fn write_get_mut_map(mut out: DisjointSlice<[f32; SIZE]>) {
        let idx = thread::index_1d();
        let a = match out.get_mut(idx) {
            Some(a) => a,
            None => return,
        };
        for i in 0..SIZE {
            a.get_mut(i).map(|e| *e = 42.0);
        }
    }

    /// Same projection chain, `if let` form, with an index-dependent value
    /// so a write that always lands in lane 0 is caught.
    #[kernel]
    pub fn write_get_mut_if_let(mut out: DisjointSlice<[f32; SIZE]>) {
        let idx = thread::index_1d();
        let a = match out.get_mut(idx) {
            Some(a) => a,
            None => return,
        };
        for i in 0..SIZE {
            if let Some(e) = a.get_mut(i) {
                *e = 100.0 + i as f32;
            }
        }
    }

    /// Direct indexed assignment `a[i] = v` through `a: &mut [f32; SIZE]`:
    /// the assignment place is `[Deref, Index(i)]` over a THIN pointer to
    /// an array (the 2-level assignment arm, array pointee).
    #[kernel]
    pub fn write_index_assign(mut out: DisjointSlice<[f32; SIZE]>) {
        let idx = thread::index_1d();
        let a = match out.get_mut(idx) {
            Some(a) => a,
            None => return,
        };
        for i in 0..SIZE {
            a[i] = 7.0 + i as f32;
        }
    }

    /// Indexed assignment `s[i] = v` through a FAT `&mut [f32]` (unsizing
    /// reborrow of the array): the assignment place is `[Deref, Index(i)]`
    /// over the fat pointer itself.
    #[kernel]
    pub fn write_slice_index_assign(mut out: DisjointSlice<[f32; SIZE]>) {
        let idx = thread::index_1d();
        let a = match out.get_mut(idx) {
            Some(a) => a,
            None => return,
        };
        let s: &mut [f32] = a;
        for i in 0..s.len() {
            s[i] = 3.0 * i as f32;
        }
    }

    /// Pre-existing working shape kept as a guard: `&mut a[i]` borrows the
    /// element address through the thin array pointer.
    #[kernel]
    pub fn write_mut_ref_index(mut out: DisjointSlice<[f32; SIZE]>) {
        let idx = thread::index_1d();
        let a = match out.get_mut(idx) {
            Some(a) => a,
            None => return,
        };
        for i in 0..SIZE {
            let e = &mut a[i];
            *e = 50.0 - i as f32;
        }
    }

    /// Field writes through an element reference produced by the inlined
    /// `slice::get_mut` over a fat `&mut [Cell]`: composes the fat-pointer
    /// address walk with the `(Deref, Field)` assignment path.
    #[kernel]
    pub fn write_struct_field_get_mut(mut out: DisjointSlice<[Cell; 2]>) {
        let idx = thread::index_1d();
        let cells = match out.get_mut(idx) {
            Some(c) => c,
            None => return,
        };
        for i in 0..2 {
            if let Some(c) = cells.get_mut(i) {
                c.lo = 1.5 + i as f32;
                c.hi = 2.5 + i as f32;
            }
        }
    }

    /// Mutable write through a fat slice whose element type is itself an
    /// array. The row index must step by one `[f32; 2]` row, then the
    /// column index selects inside that row.
    #[kernel]
    pub fn write_nested_slice_row_ref(mut out: DisjointSlice<[[f32; 2]; 2]>) {
        let idx = thread::index_1d();
        let lane = idx.get();
        let matrix = match out.get_mut(idx) {
            Some(m) => m,
            None => return,
        };

        matrix[0][0] = 10.0;
        matrix[0][1] = 77.0;
        matrix[1][0] = 0.0;
        matrix[1][1] = 99.0;

        let rows: &mut [[f32; 2]] = matrix;
        let row = ((rows[0][0] as usize) & 1) + 1;
        let elem = &mut rows[row][0];
        *elem = 42.0 + lane as f32;
    }

    /// Same nested-array fat-slice reference shape, but with a literal row
    /// index. This covers the forward `ConstantIndex` address-walker arm.
    #[kernel]
    pub fn write_nested_slice_const_row_ref(mut out: DisjointSlice<[[f32; 2]; 2]>) {
        let idx = thread::index_1d();
        let lane = idx.get();
        let matrix = match out.get_mut(idx) {
            Some(m) => m,
            None => return,
        };

        matrix[0][0] = 10.0;
        matrix[0][1] = 77.0;
        matrix[1][0] = 0.0;
        matrix[1][1] = 99.0;

        let rows: &mut [[f32; 2]] = matrix;
        let elem = &mut rows[1][0];
        *elem = 62.0 + lane as f32;
    }

    /// Same nested-array fat-slice shape as `write_nested_slice_row_ref`,
    /// but as a direct projected assignment. This exercises the
    /// statement-level store path that reuses the same address walker.
    #[kernel]
    pub fn write_nested_slice_row_assign(mut out: DisjointSlice<[[f32; 2]; 2]>) {
        let idx = thread::index_1d();
        let lane = idx.get();
        let matrix = match out.get_mut(idx) {
            Some(m) => m,
            None => return,
        };

        matrix[0][0] = 10.0;
        matrix[0][1] = 77.0;
        matrix[1][0] = 0.0;
        matrix[1][1] = 99.0;

        let rows: &mut [[f32; 2]] = matrix;
        let row = ((rows[0][0] as usize) & 1) + 1;
        rows[row][0] = 52.0 + lane as f32;
    }

    /// Nested fat-slice element that is a WIDER array (`[f32; 3]`) written at a
    /// NON-ZERO inner column. The row index must stride by one whole `[f32; 3]`
    /// row (3 floats, not 2), then the column index must select column 2 inside
    /// that row. This pins both the per-element stride and the inner
    /// array-aware index away from the leading element, the case a
    /// column-`[0]`-only test would miss.
    #[kernel]
    pub fn write_nested_slice_wide_nonzero_col(mut out: DisjointSlice<[[f32; 3]; 2]>) {
        let idx = thread::index_1d();
        let lane = idx.get();
        let matrix = match out.get_mut(idx) {
            Some(m) => m,
            None => return,
        };

        matrix[0][0] = 10.0;
        matrix[0][1] = 11.0;
        matrix[0][2] = 12.0;
        matrix[1][0] = 20.0;
        matrix[1][1] = 21.0;
        matrix[1][2] = 22.0;

        let rows: &mut [[f32; 3]] = matrix;
        let row = ((rows[0][0] as usize) & 1) + 1; // 10 & 1 == 0 -> runtime row 1
        rows[row][2] = 70.0 + lane as f32; // write row 1, column 2
    }
}

/// Launches one `[f32; SIZE]`-shaped kernel on a zeroed buffer and checks
/// every lane of every element against `expected(i)`.
fn run_and_check<F>(
    name: &str,
    stream: &Arc<CudaStream>,
    expected: fn(usize) -> f32,
    launch: F,
) -> bool
where
    F: FnOnce(&Arc<CudaStream>, LaunchConfig, &mut DeviceBuffer<[f32; SIZE]>),
{
    let mut dev_out = DeviceBuffer::<[f32; SIZE]>::zeroed(stream, N).unwrap();
    launch(stream, LaunchConfig::for_num_elems(N as u32), &mut dev_out);
    let host_out = dev_out.to_host_vec(stream).unwrap();

    let ok = host_out
        .iter()
        .all(|elem| (0..SIZE).all(|i| (elem[i] - expected(i)).abs() < 1e-6));
    let verdict = if ok { "PASS" } else { "FAIL" };
    println!("  {name:<30} {verdict}    (elem[0] = {:?})", host_out[0]);
    ok
}

/// Launches one nested-array fat-slice kernel and checks that the write
/// reaches row 1 column 0 rather than row 0 column 1.
fn run_nested_matrix_check<F>(
    name: &str,
    stream: &Arc<CudaStream>,
    expected_base: f32,
    launch: F,
) -> bool
where
    F: FnOnce(&Arc<CudaStream>, LaunchConfig, &mut DeviceBuffer<[[f32; 2]; 2]>),
{
    let mut dev_mats = DeviceBuffer::<[[f32; 2]; 2]>::zeroed(stream, N).unwrap();
    launch(stream, LaunchConfig::for_num_elems(N as u32), &mut dev_mats);
    let host_mats = dev_mats.to_host_vec(stream).unwrap();
    let ok = host_mats.iter().enumerate().all(|(lane, mat)| {
        (mat[0][0] - 10.0).abs() < 1e-6
            && (mat[0][1] - 77.0).abs() < 1e-6
            && (mat[1][0] - (expected_base + lane as f32)).abs() < 1e-6
            && (mat[1][1] - 99.0).abs() < 1e-6
    });
    let verdict = if ok { "PASS" } else { "FAIL" };
    println!("  {name:<30} {verdict}    (elem[0] = {:?})", host_mats[0]);
    ok
}

fn main() {
    println!("=== slice_get_mut: writes through fat-pointer places ===\n");

    let ctx = CudaContext::new(0).expect("CUDA init");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Load embedded PTX");

    let mut all_pass = true;

    all_pass &= run_and_check(
        "write_get_mut_map",
        &stream,
        |_| 42.0,
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        |s, cfg, o| unsafe { module.write_get_mut_map(s, cfg, o) }.expect("launch"),
    );
    all_pass &= run_and_check(
        "write_get_mut_if_let",
        &stream,
        |i| 100.0 + i as f32,
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        |s, cfg, o| unsafe { module.write_get_mut_if_let(s, cfg, o) }.expect("launch"),
    );
    all_pass &= run_and_check(
        "write_index_assign",
        &stream,
        |i| 7.0 + i as f32,
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        |s, cfg, o| unsafe { module.write_index_assign(s, cfg, o) }.expect("launch"),
    );
    all_pass &= run_and_check(
        "write_slice_index_assign",
        &stream,
        |i| 3.0 * i as f32,
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        |s, cfg, o| unsafe { module.write_slice_index_assign(s, cfg, o) }.expect("launch"),
    );
    all_pass &= run_and_check(
        "write_mut_ref_index",
        &stream,
        |i| 50.0 - i as f32,
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        |s, cfg, o| unsafe { module.write_mut_ref_index(s, cfg, o) }.expect("launch"),
    );

    // Struct-field variant has its own buffer shape; check it inline.
    {
        let mut dev_cells = DeviceBuffer::<[Cell; 2]>::zeroed(&stream, N).unwrap();
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.write_struct_field_get_mut(
                &stream,
                LaunchConfig::for_num_elems(N as u32),
                &mut dev_cells,
            )
        }
        .expect("launch");
        let host_cells = dev_cells.to_host_vec(&stream).unwrap();
        let ok = host_cells.iter().all(|pair| {
            (0..2).all(|i| {
                pair[i]
                    == Cell {
                        lo: 1.5 + i as f32,
                        hi: 2.5 + i as f32,
                    }
            })
        });
        let verdict = if ok { "PASS" } else { "FAIL" };
        println!(
            "  {:<30} {verdict}    (elem[0] = {:?})",
            "write_struct_field_get_mut", host_cells[0]
        );
        all_pass &= ok;
    }

    // Nested-array slice element variants: catch row/column stride mixups
    // when a fat slice's element type is itself an array.
    all_pass &=
        run_nested_matrix_check("write_nested_slice_row_ref", &stream, 42.0, |s, cfg, o| {
            // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
            unsafe { module.write_nested_slice_row_ref(s, cfg, o) }.expect("launch")
        });
    all_pass &= run_nested_matrix_check(
        "write_nested_slice_const_row_ref",
        &stream,
        62.0,
        |s, cfg, o| {
            // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
            unsafe { module.write_nested_slice_const_row_ref(s, cfg, o) }.expect("launch")
        },
    );
    all_pass &= run_nested_matrix_check(
        "write_nested_slice_row_assign",
        &stream,
        52.0,
        |s, cfg, o| {
            // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
            unsafe { module.write_nested_slice_row_assign(s, cfg, o) }.expect("launch")
        },
    );

    // Wider nested element (`[f32; 3]`) with a non-zero inner column: the row
    // index must stride by a whole `[f32; 3]` and the column index must land in
    // column 2, not column 0.
    {
        let mut dev = DeviceBuffer::<[[f32; 3]; 2]>::zeroed(&stream, N).unwrap();
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.write_nested_slice_wide_nonzero_col(
                &stream,
                LaunchConfig::for_num_elems(N as u32),
                &mut dev,
            )
        }
        .expect("launch");
        let host = dev.to_host_vec(&stream).unwrap();
        let ok = host.iter().enumerate().all(|(lane, m)| {
            (m[0][0] - 10.0).abs() < 1e-6
                && (m[0][1] - 11.0).abs() < 1e-6
                && (m[0][2] - 12.0).abs() < 1e-6
                && (m[1][0] - 20.0).abs() < 1e-6
                && (m[1][1] - 21.0).abs() < 1e-6
                && (m[1][2] - (70.0 + lane as f32)).abs() < 1e-6
        });
        let verdict = if ok { "PASS" } else { "FAIL" };
        println!(
            "  {:<30} {verdict}    (elem[0] = {:?})",
            "write_nested_slice_wide_nonzero_col", host[0]
        );
        all_pass &= ok;
    }

    if all_pass {
        println!("\nSUCCESS: all kernels passed");
    } else {
        println!("\nFAILURE: at least one kernel wrote the wrong lanes");
        std::process::exit(1);
    }
}
