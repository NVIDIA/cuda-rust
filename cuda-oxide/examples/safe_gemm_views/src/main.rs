/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::too_many_arguments)]

//! Provably safe 2D SGEMM through read-side proof-carrying views + Lean 4
//! shared-memory safety proofs (`#[verify(...)]`), beside raw-pointer twins.
//!
//! `C = alpha * A * B + beta * C` with runtime dimensions `m`, `n`, `k`.
//! The safe kernels check each input strip once up front
//! (`MatrixView32::row` / `col`), while `sgemm_tiled_views` additionally
//! proves `no_oob_idx(TILE_A, TILE_B)`, `no_data_race(TILE_A, TILE_B)`,
//! and `uniform_barriers` in Lean 4 at `PostMem2Reg`.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig2D};
use cuda_device::{
    ColView32, DisjointSlice, MatrixView32, RowView32, RuntimeRowMajorTiles, SharedArray, Uniform,
    cuda_module, kernel, launch_bounds, launch_contract, thread, verify,
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
    #[kernel(launch_context = lc)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1),
        requires = (k >= 1, a.len() >= m * k, b.len() >= k * n, c.len() >= m * n))]
    pub fn sgemm_naive_views(
        m: u32,
        n: Uniform<u32>,
        k: u32,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        mut c: DisjointSlice<f32, RuntimeRowMajorTiles<1, 1>>,
    ) {
        let coord = thread::coord_2d_u32(lc);
        let row = coord.row();
        let col = coord.col();
        if row >= m || col >= n.get() {
            return;
        }

        let a_mat = MatrixView32::new(a, k);
        let b_mat = MatrixView32::new(b, n.get());
        if let Some(a_row) = a_mat.row(row, k)
            && let Some(b_col) = b_mat.col(col, k)
            && let Some(pair) = a_row.zip_exact(b_col)
        {
            let mut sum = 0.0f32;
            for (x, y) in pair {
                sum += x * y;
            }
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
    /// launch. Callers must pass `k >= 1`.
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

    /// Tiled SGEMM whose staging loads go through hoisted read views and whose
    /// shared-memory tiles (`TILE_A`, `TILE_B`) and `sync_threads()` barriers
    /// are formally verified by Lean 4 at `PostMem2Reg`.
    #[kernel(launch_context = lc)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1),
        requires = (k >= 1, a.len() >= m * k, b.len() >= k * n, c.len() >= m * n))]
    #[verify(
        no_oob_idx(TILE_A, TILE_B),
        no_data_race(TILE_A, TILE_B),
        uniform_barriers
    )]
    pub fn sgemm_tiled_views(
        m: u32,
        n: Uniform<u32>,
        k: u32,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        mut c: DisjointSlice<f32, RuntimeRowMajorTiles<1, 1>>,
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
            // Phase 0 (provably safe `no_oob_idx` + `no_data_race`):
            // Each thread stages its own unique slot `smem_idx = ty * 16 + tx < 256`
            // under the contracted (16, 16, 1) block.
            {
                TILE_A[smem_idx] = a_value;
                TILE_B[smem_idx] = b_value;
            }

            thread::sync_threads();

            // Phase 1 (provably safe `no_oob_idx` + `no_data_race` + `uniform_barriers`):
            // Barrier #1 separates Phase 0 writes from Phase 1 reads, and `ty`, `i`, `tx`
            // are all `< 16` so `ty * 16 + i < 256` and `i * 16 + tx < 256`.
            {
                let mut i = 0usize;
                while i < TILE as usize {
                    sum += TILE_A[ty as usize * 16 + i] * TILE_B[i * 16 + tx as usize];
                    i += 1;
                }
            }

            thread::sync_threads();
            tile = tile.wrapping_add(1);
        }

        if let Some(mut cell) = c.tile_2d32_rt(coord) {
            let previous = cell.at_const::<0, 0>().read();
            cell.at_const::<0, 0>().write(alpha * sum + beta * previous);
        }
    }

    /// Tiled SGEMM without `MatrixView32` (`RowView32` / `ColView32`), made
    /// 100% safe by Lean 4 (`#[verify(...)]`) at `PostMem2Reg`!
    ///
    /// Unlike `gemm_views` where `sgemm_tiled_raw` had to be `pub unsafe fn`
    /// taking raw pointers `*const f32` / `*mut f32`, Lean 4 proves:
    /// - `no_oob_idx(a, b)`: the hoisted strip checks (`checked_raw_row_band_start`
    ///   and `checked_raw_col_band_start`) keep `a[a_start + a_col]` and
    ///   `b[b_start + b_row * n]` in bounds (`NoOobRowStrip`, `NoOobColStrip`)
    /// - `no_oob_idx(TILE_A, TILE_B)` & `no_data_race(TILE_A, TILE_B)`: safe
    ///   shared-memory staging and scanning across `sync_threads()` phases
    /// - `uniform_barriers`: all threads reach both `sync_threads()` barriers
    #[kernel(launch_context = lc)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1),
        requires = (k >= 1, a.len() >= m * k, b.len() >= k * n, c.len() >= m * n))]
    #[verify(
        no_oob_idx(a, b, TILE_A, TILE_B),
        no_data_race(TILE_A, TILE_B),
        uniform_barriers
    )]
    pub fn sgemm_tiled_raw(
        m: u32,
        n: Uniform<u32>,
        k: u32,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        mut c: DisjointSlice<f32, RuntimeRowMajorTiles<1, 1>>,
    ) {
        static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, 256> = SharedArray::UNINIT;

        let _ = m;
        let coord = thread::coord_2d_u32(lc);
        let row = coord.row();
        let col = coord.col();
        let tx = thread::threadIdx_x();
        let ty = thread::threadIdx_y();
        let n_val = n.get();

        let a_start = checked_raw_row_band_start(row, k, k, a.len());
        let b_start = checked_raw_col_band_start(col, n_val, k, b.len());
        let a_eff = if a_start != u64::MAX { k } else { 0 };
        let b_eff = if b_start != u64::MAX { k } else { 0 };

        let num_tiles = k.div_ceil(TILE);
        let mut sum = 0.0f32;
        let mut tile = 0u32;
        while tile < num_tiles {
            let tile_start = tile.wrapping_mul(TILE);
            let smem_idx = ty.wrapping_mul(TILE).wrapping_add(tx) as usize;

            let a_col = tile_start.wrapping_add(tx);
            let a_value = if a_col < a_eff {
                a[(a_start + a_col as u64) as usize]
            } else {
                0.0
            };
            let b_row = tile_start.wrapping_add(ty);
            let b_value = if b_row < b_eff {
                b[(b_start + b_row as u64 * n_val as u64) as usize]
            } else {
                0.0
            };
            // Phase 0 (provably safe `no_oob_idx` + `no_data_race`):
            // Each thread stages its own unique slot `smem_idx = ty * 16 + tx < 256`.
            {
                TILE_A[smem_idx] = a_value;
                TILE_B[smem_idx] = b_value;
            }

            thread::sync_threads();

            // Phase 1 (provably safe `no_oob_idx` + `no_data_race` + `uniform_barriers`):
            // Barrier #1 separates Phase 0 writes from Phase 1 reads (`< 256`).
            {
                let mut i = 0usize;
                while i < TILE as usize {
                    sum += TILE_A[ty as usize * 16 + i] * TILE_B[i * 16 + tx as usize];
                    i += 1;
                }
            }

            thread::sync_threads();
            tile = tile.wrapping_add(1);
        }

        if let Some(mut cell) = c.tile_2d32_rt(coord) {
            let previous = cell.at_const::<0, 0>().read();
            cell.at_const::<0, 0>().write(alpha * sum + beta * previous);
        }
    }
}

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
    let mut c_raw_tiled = DeviceBuffer::from_host(&stream, &c_init)?;

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
        println!("  {label:<24} {avg_ms:8.3} ms  {gflops:9.1} GFLOPS");
        Ok(())
    }

    println!("safe_gemm_views bench: {BM}x{BN}x{BK}, {NUM_RUNS} timed runs");
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
            cuda_host::RowWidth::new(&mut c_views_tiled, n),
        )?;
        Ok(())
    })?;
    time_runs(&stream, "tiled raw (now safe!)", flops, || {
        module.sgemm_tiled_raw(
            &stream,
            &raw_tiled_launch,
            m,
            n,
            k,
            ALPHA,
            &a_dev,
            &b_dev,
            BETA,
            cuda_host::RowWidth::new(&mut c_raw_tiled, n),
        )?;
        Ok(())
    })?;

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
    if std::env::args().any(|arg| arg == "--bench") {
        return bench();
    }

    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    // SAFETY: this standalone example owns the package-named device bundle,
    // whose four entry definitions are generated by the module above.
    let module = unsafe { kernels::load(&context)? };

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

    let safe_naive = c_safe_naive.to_host_vec(&stream)?;
    let raw_naive = c_raw_naive.to_host_vec(&stream)?;

    let mut c_safe_tiled = DeviceBuffer::from_host(&stream, &c_init)?;
    let mut c_raw_tiled = DeviceBuffer::from_host(&stream, &c_init)?;
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
    module.sgemm_tiled_raw(
        &stream,
        &raw_tiled_launch,
        m_arg,
        n_arg,
        k_arg,
        ALPHA,
        &a_dev,
        &b_dev,
        BETA,
        cuda_host::RowWidth::new(&mut c_raw_tiled, n_arg),
    )?;
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

    println!();
    bench()?;

    println!();
    println!("=== Provably Safe 2D Tiled SGEMM (`safe_gemm_views`) ===\n");
    println!("Contract & Lean 4 Proof Summary:");
    println!("  • Launch contract:   domain = 2, block = (16, 16, 1)");
    println!("  • Host requires:     k >= 1, a.len() >= m * k, b.len() >= k * n, c.len() >= m * n");
    println!("  • sgemm_tiled_views: SAFE (MatrixView32 for `a, b` + Lean 4 for `TILE_A, TILE_B`)");
    println!(
        "                       Proved: no_oob_idx(TILE_A, TILE_B), no_data_race(TILE_A, TILE_B), uniform_barriers"
    );
    println!("                       Elided: 4 SharedArray `mir.assert` bounds checks");
    println!(
        "  • sgemm_tiled_raw:   NOW SAFE! (Was `unsafe` in `gemm_views`; now 100% safe slices without MatrixView32)"
    );
    println!(
        "                       Proved: no_oob_idx(a, b, TILE_A, TILE_B), no_data_race(TILE_A, TILE_B), uniform_barriers"
    );
    println!(
        "                       Elided: 6 `mir.assert` bounds checks (2 global slice strips + 4 SharedArray tiles)\n"
    );
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
