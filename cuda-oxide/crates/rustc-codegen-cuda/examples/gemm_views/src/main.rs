/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::too_many_arguments)]

//! SGEMM through read-side proof-carrying views, beside raw-pointer twins.
//!
//! `C = alpha * A * B + beta * C` with runtime dimensions `m`, `n`, `k`.
//! The safe kernels check each input strip once up front
//! (`MatrixView32::row` / `col`), then read with no further checks; the
//! fused `zip_exact` pair drives the naive dot product with a single
//! compare per iteration (the loop's own exit test). The C write goes
//! through a 1x1 tile with a runtime row width (`tile_2d32_rt`).

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig2D};
use cuda_device::{
    ColView32, DisjointSlice, MatrixView32, RowView32, RuntimeRowMajorTiles, SharedArray, Uniform,
    cuda_module, kernel, launch_bounds, launch_contract, thread,
};

const M: usize = 128;
const N: usize = 96;
const K: usize = 64;
const ALPHA: f32 = 1.5;
const BETA: f32 = 0.5;
const BLOCK: u32 = 16;

#[cuda_module]
mod kernels {
    use super::*;

    const TILE: u32 = 16;

    // The raw twins re-derive the exact scalar proofs that the view
    // constructors run, so the safe and raw entries keep the same guard
    // structure in PTX.

    #[inline(always)]
    fn checked_raw_row_band_start(row: u32, stride: u32, cols: u32, len: usize) -> u64 {
        if cols == 0 || cols > stride {
            return u64::MAX;
        }
        let start = row as u64 * stride as u64;
        let last = start + (cols as u64 - 1);
        if last < len as u64 { start } else { u64::MAX }
    }

    #[inline(always)]
    fn checked_raw_col_band_start(col: u32, stride: u32, rows: u32, len: usize) -> u64 {
        if rows == 0 || col >= stride {
            return u64::MAX;
        }
        let last = (rows as u64 - 1) * stride as u64 + col as u64;
        if last < len as u64 {
            col as u64
        } else {
            u64::MAX
        }
    }

    #[inline(always)]
    fn checked_raw_cell_start(row: u32, col: u32, stride: u32, len: usize) -> u64 {
        if stride == 0 || col >= stride {
            return u64::MAX;
        }
        if row as u64 > (u64::MAX - col as u64) / stride as u64 {
            return u64::MAX;
        }
        let flat = row as u64 * stride as u64 + col as u64;
        if flat < len as u64 { flat } else { u64::MAX }
    }

    /// Naive SGEMM where every input read goes through a proof-carrying view.
    ///
    /// One up-front check for A's row, one for B's column, one
    /// length-equality check to fuse them; the dot-product loop then compiles
    /// to load / load / fma / advance / compare / branch with no bounds
    /// traps.
    #[kernel(launch_context = lc)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1),
        requires = (k >= 1, a.len() >= m * k, b.len() >= k * n, c.len() >= m * n))]
    pub fn sgemm_naive_views(
        m: u32,
        n: Uniform<u32>,
        k: u32,
        alpha: f32,
        a: &[f32], // M x K matrix
        b: &[f32], // K x N matrix
        beta: f32,
        mut c: DisjointSlice<f32, RuntimeRowMajorTiles<1, 1>>, // M x N matrix
    ) {
        let coord = thread::coord_2d_u32(lc);
        let row = coord.row();
        let col = coord.col();
        // No barriers in this kernel, so out-of-range threads may leave early.
        if row >= m || col >= n.get() {
            return;
        }

        let a_mat = MatrixView32::new(a, k);
        let b_mat = MatrixView32::new(b, n.get());
        // The `requires` contract proves the buffer sizes and `k >= 1` on
        // the host, so for a contracted launch every proof below succeeds and
        // the beta epilogue always runs.
        if let Some(a_row) = a_mat.row(row, k)
            && let Some(b_col) = b_mat.col(col, k)
            && let Some(pair) = a_row.zip_exact(b_col)
        {
            let mut sum = 0.0f32;
            for (x, y) in pair {
                sum += x * y;
            }
            // `c` carries its own row width, bound on the host to the same `n`
            // this kernel reads, so no stride crosses the call and no `unsafe`
            // is needed here.
            if let Some(mut cell) = c.tile_2d32_rt(coord) {
                let previous = cell.at_const::<0, 0>().read();
                cell.at_const::<0, 0>().write(alpha * sum + beta * previous);
            }
        }
    }

    /// The same naive SGEMM with manually proved raw pointers.
    ///
    /// # Safety
    ///
    /// `a`, `b`, and `c` must reference `a_len`, `b_len`, and `c_len` readable
    /// (and, for `c`, writable) device `f32` elements for the duration of the
    /// launch.
    /// Callers must pass `k >= 1`, mirroring the safe twin's contract.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1))]
    pub unsafe fn sgemm_naive_raw(
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        a: *const f32,
        a_len: usize,
        b: *const f32,
        b_len: usize,
        beta: f32,
        c: *mut f32,
        c_len: usize,
    ) {
        let row = thread::blockIdx_y()
            .wrapping_mul(thread::blockDim_y())
            .wrapping_add(thread::threadIdx_y());
        let col = thread::blockIdx_x()
            .wrapping_mul(thread::blockDim_x())
            .wrapping_add(thread::threadIdx_x());
        if row >= m || col >= n {
            return;
        }

        // Sequential strip checks, mirroring the safe kernel's `if let` chain
        // (B's strip is only checked after A's check succeeds). Like the safe
        // kernel's contract, callers must pass `k >= 1`; a failed check skips
        // the whole update.
        let a_start = checked_raw_row_band_start(row, k, k, a_len);
        if a_start == u64::MAX {
            return;
        }
        let b_start = checked_raw_col_band_start(col, n, k, b_len);
        if b_start == u64::MAX {
            return;
        }
        let mut sum = 0.0f32;
        let mut i = 0u32;
        while i < k {
            // SAFETY: the strip checks above cover offsets `i` (row) and
            // `i * n` (column) for every `i < k`.
            unsafe {
                let x = *a.add((a_start + i as u64) as usize);
                let y = *b.add((b_start + i as u64 * n as u64) as usize);
                sum += x * y;
            }
            i = i.wrapping_add(1);
        }

        let cell = checked_raw_cell_start(row, col, n, c_len);
        if cell == u64::MAX {
            return;
        }
        // SAFETY: the cell proof covers this element, and distinct (row, col)
        // pairs map to distinct cells under the shared row width `n`.
        unsafe {
            let ptr = c.add(cell as usize);
            let previous = *ptr;
            *ptr = alpha * sum + beta * previous;
        }
    }

    /// Tiled SGEMM whose staging loads go through hoisted read views.
    ///
    /// Both strip checks run once before the tile loop; a failed check leaves
    /// an empty view instead of returning, so every thread reaches both
    /// barriers (barrier uniformity). Each staging read is then a single
    /// compare with a zero-fill fallback: `.get(i).unwrap_or(0.0)`.
    #[kernel(launch_context = lc)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1),
        requires = (k >= 1, a.len() >= m * k, b.len() >= k * n, c.len() >= m * n))]
    pub fn sgemm_tiled_views(
        m: u32,
        n: Uniform<u32>,
        k: u32,
        alpha: f32,
        a: &[f32], // M x K matrix
        b: &[f32], // K x N matrix
        beta: f32,
        mut c: DisjointSlice<f32, RuntimeRowMajorTiles<1, 1>>, // M x N matrix
    ) {
        static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, 256> = SharedArray::UNINIT;

        let _ = m;
        let coord = thread::coord_2d_u32(lc);
        let row = coord.row();
        let col = coord.col();
        let tx = thread::threadIdx_x();
        let ty = thread::threadIdx_y();

        let a_mat = MatrixView32::new(a, k);
        let b_mat = MatrixView32::new(b, n.get());
        let a_row = a_mat.row(row, k).unwrap_or(RowView32::empty());
        let b_col = b_mat.col(col, k).unwrap_or(ColView32::empty());

        let num_tiles = k.div_ceil(TILE);
        let mut sum = 0.0f32;
        let mut tile = 0u32;
        while tile < num_tiles {
            let tile_start = tile.wrapping_mul(TILE);
            let smem_idx = ty.wrapping_mul(TILE).wrapping_add(tx) as usize;

            let a_value = a_row.get(tile_start.wrapping_add(tx)).unwrap_or(0.0);
            let b_value = b_col.get(tile_start.wrapping_add(ty)).unwrap_or(0.0);
            // SAFETY: smem_idx = ty * 16 + tx < 256 under the contracted
            // (16, 16, 1) block, so each thread stages a unique slot.
            unsafe {
                TILE_A[smem_idx] = a_value;
                TILE_B[smem_idx] = b_value;
            }

            thread::sync_threads();

            // SAFETY: ty, i, and tx are all below 16, so every index stays
            // inside the 256-element shared tiles.
            unsafe {
                let mut i = 0usize;
                while i < TILE as usize {
                    sum += TILE_A[ty as usize * 16 + i] * TILE_B[i * 16 + tx as usize];
                    i += 1;
                }
            }

            thread::sync_threads();
            tile = tile.wrapping_add(1);
        }

        // Epilogue after the final barrier: one rectangle proof for this
        // thread's C cell. Out-of-range threads get `None` and simply skip.
        // `c` carries its own row width, bound on the host to the same `n`
        // this kernel reads, so no stride crosses the call.
        if let Some(mut cell) = c.tile_2d32_rt(coord) {
            let previous = cell.at_const::<0, 0>().read();
            cell.at_const::<0, 0>().write(alpha * sum + beta * previous);
        }
    }

    /// The same tiled SGEMM with manually proved raw pointers.
    ///
    /// # Safety
    ///
    /// `a`, `b`, and `c` must reference `a_len`, `b_len`, and `c_len` readable
    /// (and, for `c`, writable) device `f32` elements for the duration of the
    /// launch.
    /// Callers must pass `k >= 1`, mirroring the safe twin's contract.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1))]
    pub unsafe fn sgemm_tiled_raw(
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        a: *const f32,
        a_len: usize,
        b: *const f32,
        b_len: usize,
        beta: f32,
        c: *mut f32,
        c_len: usize,
    ) {
        static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, 256> = SharedArray::UNINIT;

        let _ = m;
        let tx = thread::threadIdx_x();
        let ty = thread::threadIdx_y();
        let row = thread::blockIdx_y()
            .wrapping_mul(thread::blockDim_y())
            .wrapping_add(ty);
        let col = thread::blockIdx_x()
            .wrapping_mul(thread::blockDim_x())
            .wrapping_add(tx);

        // Hoisted strip checks mirroring the safe kernel's view constructors.
        // A failed check degrades to a zero-length strip instead of an early
        // return, so every thread reaches both barriers.
        let a_start = checked_raw_row_band_start(row, k, k, a_len);
        let b_start = checked_raw_col_band_start(col, n, k, b_len);
        let (a_base, a_eff): (*const f32, u32) = if a_start != u64::MAX {
            // SAFETY: the row strip check covers `a_start .. a_start + k`.
            (unsafe { a.add(a_start as usize) }, k)
        } else {
            (core::ptr::null(), 0)
        };
        let (b_base, b_eff): (*const f32, u32) = if b_start != u64::MAX {
            // SAFETY: the column strip check covers `b_start + i * n`, i < k.
            (unsafe { b.add(b_start as usize) }, k)
        } else {
            (core::ptr::null(), 0)
        };

        let num_tiles = k.div_ceil(TILE);
        let mut sum = 0.0f32;
        let mut tile = 0u32;
        while tile < num_tiles {
            let tile_start = tile.wrapping_mul(TILE);
            let smem_idx = ty.wrapping_mul(TILE).wrapping_add(tx) as usize;

            let a_col = tile_start.wrapping_add(tx);
            let a_value = if a_col < a_eff {
                // SAFETY: the hoisted row strip check covers offsets 0..a_eff.
                unsafe { *a_base.add(a_col as usize) }
            } else {
                0.0
            };
            let b_row = tile_start.wrapping_add(ty);
            let b_value = if b_row < b_eff {
                // SAFETY: the hoisted column strip check covers `i * n` for
                // every i below b_eff.
                unsafe { *b_base.add((b_row as u64 * n as u64) as usize) }
            } else {
                0.0
            };
            // SAFETY: smem_idx = ty * 16 + tx < 256 under the contracted
            // (16, 16, 1) block, so each thread stages a unique slot.
            unsafe {
                TILE_A[smem_idx] = a_value;
                TILE_B[smem_idx] = b_value;
            }

            thread::sync_threads();

            // SAFETY: ty, i, and tx are all below 16, so every index stays
            // inside the 256-element shared tiles.
            unsafe {
                let mut i = 0usize;
                while i < TILE as usize {
                    sum += TILE_A[ty as usize * 16 + i] * TILE_B[i * 16 + tx as usize];
                    i += 1;
                }
            }

            thread::sync_threads();
            tile = tile.wrapping_add(1);
        }

        let cell = checked_raw_cell_start(row, col, n, c_len);
        if cell == u64::MAX {
            return;
        }
        // SAFETY: the cell proof covers this element, and distinct (row, col)
        // pairs map to distinct cells under the shared row width `n`.
        unsafe {
            let ptr = c.add(cell as usize);
            let previous = *ptr;
            *ptr = alpha * sum + beta * previous;
        }
    }
}

/// Time all four kernels at 1024^3 with the same methodology as the `gemm`
/// example (untimed warmup, then 5 timed launches, one sync, average).
fn bench() -> Result<(), Box<dyn std::error::Error>> {
    const BM: usize = 1024;
    const BN: usize = 1024;
    const BK: usize = 1024;
    const WARMUP: u32 = 2;
    const NUM_RUNS: u32 = 5;
    let flops = 2.0 * BM as f64 * BN as f64 * BK as f64;

    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    // SAFETY: this standalone example owns the package-named device bundle,
    // whose four entry definitions are generated by the module above.
    let module = unsafe { kernels::load(&context)? };

    let mut a = vec![0.0f32; BM * BK];
    let mut b = vec![0.0f32; BK * BN];
    let mut c_init = vec![0.0f32; BM * BN];
    for (index, value) in a.iter_mut().enumerate() {
        *value = ((index * 3 + 1) % 10) as f32 * 0.1;
    }
    for (index, value) in b.iter_mut().enumerate() {
        *value = ((index * 7 + 2) % 10) as f32 * 0.1;
    }
    for (index, value) in c_init.iter_mut().enumerate() {
        *value = ((index * 5 + 3) % 13) as f32 * 0.25;
    }

    let a_dev = DeviceBuffer::from_host(&stream, &a)?;
    let b_dev = DeviceBuffer::from_host(&stream, &b)?;
    let mut c_views_naive = DeviceBuffer::from_host(&stream, &c_init)?;
    let c_raw_naive = DeviceBuffer::from_host(&stream, &c_init)?;
    let mut c_views_tiled = DeviceBuffer::from_host(&stream, &c_init)?;
    let c_raw_tiled = DeviceBuffer::from_host(&stream, &c_init)?;

    let grid = ((BN as u32) / BLOCK, (BM as u32) / BLOCK);
    let config = LaunchConfig2D::new(grid, (BLOCK, BLOCK), 0);
    let (m, n, k) = (BM as u32, BN as u32, BK as u32);

    let views_naive_launch = module.prepare_sgemm_naive_views(config)?;
    let raw_naive_launch = module.prepare_sgemm_naive_raw(config)?;
    let views_tiled_launch = module.prepare_sgemm_tiled_views(config)?;
    let raw_tiled_launch = module.prepare_sgemm_tiled_raw(config)?;

    fn time_runs(
        stream: &cuda_core::CudaStream,
        label: &str,
        flops: f64,
        mut launch: impl FnMut() -> Result<(), Box<dyn std::error::Error>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for _ in 0..WARMUP {
            launch()?;
        }
        stream.synchronize()?;
        let start = std::time::Instant::now();
        for _ in 0..NUM_RUNS {
            launch()?;
        }
        stream.synchronize()?;
        let avg_ms = start.elapsed().as_secs_f64() * 1000.0 / NUM_RUNS as f64;
        let gflops = flops / (avg_ms / 1000.0) / 1e9;
        println!("  {label:<20} {avg_ms:8.3} ms  {gflops:9.1} GFLOPS");
        Ok(())
    }

    println!("gemm_views bench: {BM}x{BN}x{BK}, {NUM_RUNS} timed runs");
    time_runs(&stream, "naive views (safe)", flops, || {
        module.sgemm_naive_views(
            &stream,
            &views_naive_launch,
            m,
            n,
            k,
            ALPHA,
            &a_dev,
            &b_dev,
            BETA,
            // C's row width is bound to the slice here, once for the launch.
            cuda_host::RowWidth::new(&mut c_views_naive, n),
        )?;
        Ok(())
    })?;
    // SAFETY: each buffer owns exactly the element count passed beside it and
    // stays alive until the final synchronize below.
    time_runs(&stream, "naive raw (unsafe)", flops, || {
        unsafe {
            module.sgemm_naive_raw(
                &stream,
                &raw_naive_launch,
                m,
                n,
                k,
                ALPHA,
                a_dev.cu_deviceptr() as *const f32,
                a.len(),
                b_dev.cu_deviceptr() as *const f32,
                b.len(),
                BETA,
                c_raw_naive.cu_deviceptr() as *mut f32,
                c_init.len(),
            )?;
        }
        Ok(())
    })?;
    time_runs(&stream, "tiled views (safe)", flops, || {
        module.sgemm_tiled_views(
            &stream,
            &views_tiled_launch,
            m,
            n,
            k,
            ALPHA,
            &a_dev,
            &b_dev,
            BETA,
            // C's row width is bound to the slice here, once for the launch.
            cuda_host::RowWidth::new(&mut c_views_tiled, n),
        )?;
        Ok(())
    })?;
    // SAFETY: as above; the raw tiled twin reads a/b and writes its own C.
    time_runs(&stream, "tiled raw (unsafe)", flops, || {
        unsafe {
            module.sgemm_tiled_raw(
                &stream,
                &raw_tiled_launch,
                m,
                n,
                k,
                ALPHA,
                a_dev.cu_deviceptr() as *const f32,
                a.len(),
                b_dev.cu_deviceptr() as *const f32,
                b.len(),
                BETA,
                c_raw_tiled.cu_deviceptr() as *mut f32,
                c_init.len(),
            )?;
        }
        Ok(())
    })?;

    // Safe and raw twins ran the same launch count from identical C, so the
    // accumulated results must still match bitwise.
    let views_naive = c_views_naive.to_host_vec(&stream)?;
    let raw_naive = c_raw_naive.to_host_vec(&stream)?;
    let views_tiled = c_views_tiled.to_host_vec(&stream)?;
    let raw_tiled = c_raw_tiled.to_host_vec(&stream)?;
    if views_naive != raw_naive || views_tiled != raw_tiled {
        return Err("bench results diverged between safe and raw twins".into());
    }
    println!("bench sanity: safe == raw bitwise for both pairs");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().any(|arg| arg == "--verify-ptx") {
        return verify_ptx();
    }
    if std::env::args().any(|arg| arg == "--bench") {
        return bench();
    }

    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    // SAFETY: this standalone example owns the package-named device bundle,
    // whose four entry definitions are generated by the module above.
    let module = unsafe { kernels::load(&context)? };

    // Runtime sizes deliberately distinct: A is 128x64, B is 64x96.
    let mut a = vec![0.0f32; M * K];
    let mut b = vec![0.0f32; K * N];
    let mut c_init = vec![0.0f32; M * N];
    for row in 0..M {
        for col in 0..K {
            a[row * K + col] = ((row + col) % 10) as f32 * 0.1;
        }
    }
    for row in 0..K {
        for col in 0..N {
            b[row * N + col] = ((row * col) % 10) as f32 * 0.1;
        }
    }
    for (index, value) in c_init.iter_mut().enumerate() {
        *value = ((index * 7 + 3) % 13) as f32 * 0.25;
    }

    let reference = cpu_reference(&a, &b, &c_init);

    let a_dev = DeviceBuffer::from_host(&stream, &a)?;
    let b_dev = DeviceBuffer::from_host(&stream, &b)?;

    let grid = ((N as u32) / BLOCK, (M as u32) / BLOCK);
    let config = LaunchConfig2D::new(grid, (BLOCK, BLOCK), 0);

    let (m_arg, n_arg, k_arg) = (M as u32, N as u32, K as u32);

    // Naive pair.
    let mut c_safe_naive = DeviceBuffer::from_host(&stream, &c_init)?;
    let c_raw_naive = DeviceBuffer::from_host(&stream, &c_init)?;
    let safe_naive_launch = module.prepare_sgemm_naive_views(config)?;
    let raw_naive_launch = module.prepare_sgemm_naive_raw(config)?;
    module.sgemm_naive_views(
        &stream,
        &safe_naive_launch,
        m_arg,
        n_arg,
        k_arg,
        ALPHA,
        &a_dev,
        &b_dev,
        BETA,
        cuda_host::RowWidth::new(&mut c_safe_naive, n_arg),
    )?;
    // SAFETY: each buffer owns exactly the element count passed beside it and
    // stays alive until after stream synchronization in `to_host_vec`.
    unsafe {
        module.sgemm_naive_raw(
            &stream,
            &raw_naive_launch,
            m_arg,
            n_arg,
            k_arg,
            ALPHA,
            a_dev.cu_deviceptr() as *const f32,
            a.len(),
            b_dev.cu_deviceptr() as *const f32,
            b.len(),
            BETA,
            c_raw_naive.cu_deviceptr() as *mut f32,
            c_init.len(),
        )?;
    }
    // The requires contract rejects a launch whose claimed sizes exceed the
    // buffers, on the CPU, before any GPU work: claim twice the real K.
    match module.sgemm_naive_views(
        &stream,
        &safe_naive_launch,
        m_arg,
        n_arg,
        k_arg * 2,
        ALPHA,
        &a_dev,
        &b_dev,
        BETA,
        cuda_host::RowWidth::new(&mut c_safe_naive, n_arg),
    ) {
        Err(err) => {
            let text = err.to_string();
            if !text.contains("size requirement") {
                return Err(format!("expected a size-requirement rejection, got: {text}").into());
            }
            println!("contract rejected oversized-K launch on the CPU: {text}");
        }
        Ok(()) => {
            return Err("requires contract accepted a K twice the real size".into());
        }
    }

    let safe_naive = c_safe_naive.to_host_vec(&stream)?;
    let raw_naive = c_raw_naive.to_host_vec(&stream)?;

    // Tiled pair.
    let mut c_safe_tiled = DeviceBuffer::from_host(&stream, &c_init)?;
    let c_raw_tiled = DeviceBuffer::from_host(&stream, &c_init)?;
    let safe_tiled_launch = module.prepare_sgemm_tiled_views(config)?;
    let raw_tiled_launch = module.prepare_sgemm_tiled_raw(config)?;
    module.sgemm_tiled_views(
        &stream,
        &safe_tiled_launch,
        m_arg,
        n_arg,
        k_arg,
        ALPHA,
        &a_dev,
        &b_dev,
        BETA,
        cuda_host::RowWidth::new(&mut c_safe_tiled, n_arg),
    )?;
    // SAFETY: each buffer owns exactly the element count passed beside it and
    // stays alive until after stream synchronization in `to_host_vec`.
    unsafe {
        module.sgemm_tiled_raw(
            &stream,
            &raw_tiled_launch,
            m_arg,
            n_arg,
            k_arg,
            ALPHA,
            a_dev.cu_deviceptr() as *const f32,
            a.len(),
            b_dev.cu_deviceptr() as *const f32,
            b.len(),
            BETA,
            c_raw_tiled.cu_deviceptr() as *mut f32,
            c_init.len(),
        )?;
    }
    let safe_tiled = c_safe_tiled.to_host_vec(&stream)?;
    let raw_tiled = c_raw_tiled.to_host_vec(&stream)?;

    const TOLERANCE: f32 = 1e-3;
    for (name, result) in [
        ("sgemm_naive_views", &safe_naive),
        ("sgemm_naive_raw", &raw_naive),
        ("sgemm_tiled_views", &safe_tiled),
        ("sgemm_tiled_raw", &raw_tiled),
    ] {
        let error = max_abs_diff(result, &reference);
        println!("{name:>18}: max |gpu - cpu| = {error:.3e}");
        if error > TOLERANCE {
            return Err(format!("{name} diverges from the CPU reference by {error}").into());
        }
    }
    let naive_pair_diff = max_abs_diff(&safe_naive, &raw_naive);
    let tiled_pair_diff = max_abs_diff(&safe_tiled, &raw_tiled);
    println!("naive safe vs raw: max diff = {naive_pair_diff:.3e}");
    println!("tiled safe vs raw: max diff = {tiled_pair_diff:.3e}");
    if naive_pair_diff > TOLERANCE || tiled_pair_diff > TOLERANCE {
        return Err("safe and raw twins disagree".into());
    }

    // Correctness proven above on small, non-square sizes; now show what the
    // views buy at a real size.
    println!();
    bench()?;

    println!();
    println!("SUCCESS: view-based SGEMM matched raw twins and CPU reference");
    Ok(())
}

fn cpu_reference(a: &[f32], b: &[f32], c_init: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; M * N];
    for row in 0..M {
        for col in 0..N {
            let mut sum = 0.0f32;
            for i in 0..K {
                sum += a[row * K + i] * b[i * N + col];
            }
            out[row * N + col] = ALPHA * sum + BETA * c_init[row * N + col];
        }
    }
    out
}

fn max_abs_diff(x: &[f32], y: &[f32]) -> f32 {
    x.iter()
        .zip(y.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

// =============================================================================
// PTX structure verification (no GPU required)
// =============================================================================

fn verify_ptx() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("gemm_views.ptx");
    let ptx = std::fs::read_to_string(&path)?;
    let document = ptx_parse::Document::parse(&ptx)?;

    for marker in [
        "__launch_contract_config",
        "__launch_contract_block_config",
        "__launch_bounds_config",
        "make_kernel_scope",
    ] {
        if ptx.contains(marker) {
            return Err(
                format!("compile-time marker `{marker}` leaked into the PTX module").into(),
            );
        }
    }

    let safe_naive = entry(&document, "sgemm_naive_views")?;
    let raw_naive = entry(&document, "sgemm_naive_raw")?;
    let safe_tiled = entry(&document, "sgemm_tiled_views")?;
    let raw_tiled = entry(&document, "sgemm_tiled_raw")?;

    for (name, definition) in [
        ("sgemm_naive_views", safe_naive),
        ("sgemm_naive_raw", raw_naive),
        ("sgemm_tiled_views", safe_tiled),
        ("sgemm_tiled_raw", raw_tiled),
    ] {
        let body = definition.text();
        // These kernels declare `block = (16, 16, 1)`, so the exact shape
        // reaches the device compiler as `.reqntid` and the driver rejects any
        // other block on any axis. A thread maximum cannot express this shape:
        // it bounds the product alone, and its per-axis fields would claim one
        // thread on Y for a block that has sixteen. `.maxntid` must therefore
        // be absent, and ptxas rejects an entry carrying both regardless.
        if !body.contains(".reqntid 16, 16, 1") {
            return Err(format!("{name} lost its exact 16x16 block shape").into());
        }
        if body.contains(".maxntid") {
            return Err(
                format!("{name} declares both .maxntid and .reqntid, which ptxas rejects").into(),
            );
        }
        verify_no_calls(name, definition)?;
        // Every bounds fact is proven through sentinel scalar checks; nothing
        // in these kernels may lower to a panic trap.
        let traps = trap_count(definition);
        if traps != 0 {
            return Err(format!("{name} contains {traps} trap instructions").into());
        }
        if !body.contains("fma.rn.f32") {
            return Err(format!("{name} lost its fused multiply-add loop").into());
        }
    }

    // The naive pair must keep the identical guard anatomy: same number of
    // conditional branches (strip checks + loop bound), no re-checks inside
    // the proven dot-product loop.
    let safe_branches = conditional_branches(safe_naive);
    let raw_branches = conditional_branches(raw_naive);
    if safe_branches != raw_branches {
        return Err(format!(
            "naive guard branches differ: safe={safe_branches}, raw={raw_branches}"
        )
        .into());
    }
    compare_memory_operations("naive", safe_naive, raw_naive)?;

    // The tiled pair must actually stage through shared memory and keep the
    // same global-memory traffic as the raw twin.
    for (name, definition) in [
        ("sgemm_tiled_views", safe_tiled),
        ("sgemm_tiled_raw", raw_tiled),
    ] {
        let body = definition.text();
        for operation in ["ld.shared", "st.shared"] {
            if !body.contains(operation) {
                return Err(format!("{name} has no {operation} traffic").into());
            }
        }
    }
    compare_memory_operations("tiled", safe_tiled, raw_tiled)?;

    println!(
        "SUCCESS: gemm_views safe kernels match raw PTX structure \
         (naive branches: {safe_branches})"
    );
    Ok(())
}

fn entry<'document, 'source>(
    document: &'document ptx_parse::Document<'source>,
    name: &str,
) -> Result<ptx_parse::CallableDefinition<'document, 'source>, Box<dyn std::error::Error>> {
    document
        .definitions_named(name)
        .find(|definition| definition.callable().kind() == ptx_parse::CallableKind::Entry)
        .ok_or_else(|| format!("missing or incomplete PTX entry `{name}`").into())
}

fn trap_count(definition: ptx_parse::CallableDefinition<'_, '_>) -> usize {
    definition
        .instructions()
        .filter(|instruction| instruction.base_opcode() == "trap")
        .count()
}

fn conditional_branches(definition: ptx_parse::CallableDefinition<'_, '_>) -> usize {
    definition
        .instructions()
        .filter(|instruction| {
            instruction.base_opcode() == "bra" && instruction.predicate().is_some()
        })
        .count()
}

fn verify_no_calls(
    name: &str,
    definition: ptx_parse::CallableDefinition<'_, '_>,
) -> Result<(), Box<dyn std::error::Error>> {
    if definition
        .instructions()
        .any(|instruction| instruction.base_opcode() == "call")
    {
        return Err(format!("{name} contains an out-of-line device call").into());
    }
    Ok(())
}

fn compare_memory_operations(
    pair: &str,
    safe: ptx_parse::CallableDefinition<'_, '_>,
    raw: ptx_parse::CallableDefinition<'_, '_>,
) -> Result<(), Box<dyn std::error::Error>> {
    for operation in ["ld", "st"] {
        let safe_ops = data_memory_operations(safe, operation);
        let raw_ops = data_memory_operations(raw, operation);
        if safe_ops.is_empty() {
            return Err(format!("safe {pair} entry has no `{operation}` data operation").into());
        }
        if safe_ops != raw_ops {
            return Err(format!(
                "{pair} {operation} operations differ: safe={safe_ops:?}, raw={raw_ops:?}"
            )
            .into());
        }
    }
    Ok(())
}

fn data_memory_operations(
    definition: ptx_parse::CallableDefinition<'_, '_>,
    operation: &str,
) -> Vec<String> {
    let mut operations: Vec<String> = definition
        .instructions()
        .filter_map(|instruction| data_memory_operation(instruction, operation))
        .collect();
    operations.sort();
    operations
}

fn data_memory_operation(
    instruction: &ptx_parse::Instruction<'_>,
    operation: &str,
) -> Option<String> {
    if instruction.base_opcode() != operation {
        return None;
    }
    let mnemonic = instruction.head();
    if mnemonic.contains(".param.") || mnemonic.contains(".shared.") || mnemonic.contains(".local.")
    {
        return None;
    }
    Some(mnemonic.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptx_parser_counts_traps_branches_and_data_operations() {
        let ptx = ".visible .entry test() {\n\
                    @%p1 bra $L__BB0_2;\n\
                    ld.global.f32 %f1, [%rd1];\n\
                    ld.shared.f32 %f2, [%r1];\n\
                    st.global.f32 [%rd2], %f3;\n\
                    trap;\n\
                    @!%p2 bra.uni $L__BB0_3;\n\
                    call.uni helper;\n}";
        let document = ptx_parse::Document::parse(ptx).unwrap();
        let definition = entry(&document, "test").unwrap();
        assert_eq!(trap_count(definition), 1);
        assert_eq!(conditional_branches(definition), 2);
        assert!(
            definition
                .instructions()
                .any(|instruction| instruction.base_opcode() == "call")
        );
        assert_eq!(
            data_memory_operations(definition, "ld"),
            vec!["ld.global.f32"]
        );
        assert_eq!(
            data_memory_operations(definition, "st"),
            vec!["st.global.f32"]
        );
    }
}
