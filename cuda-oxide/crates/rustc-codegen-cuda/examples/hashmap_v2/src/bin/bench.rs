/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `hashmap_v2` performance bench — head-to-head GPU vs CPU `hashbrown`.
//!
//! Measures three operation classes across three load factors:
//!
//! - **insert**:      Protocol B (payload-first) and Protocol A
//!   (RESERVED-tag handshake), one thread per key.
//! - **lookup**:      every query hits. Protocol-B-built map; queries
//!   the inserted keys via single-thread find and warp-cooperative find.
//! - **lookup_fail**: every query misses. Same maps, fresh disjoint
//!   random keys.
//!
//! Load factors: 50%, 75%, 90% of `CAPACITY` slots.
//!
//! GPU timing uses CUDA events around the kernel launch loop only —
//! no H2D upload of keys, no D2H of results — so the comparison is
//! pure compute on both sides. Insert benches include the `memset_d8`
//! reset of `ctrl` + `slots` between iterations (same overhead for B
//! and A, cancels in the comparison).
//!
//! CPU baseline:
//!
//! - insert:               single-threaded `hashbrown::HashMap::insert`
//!   (hashbrown is not concurrent — insert needs `&mut self`).
//! - lookup / lookup_fail: rayon-parallel `.get(&k)` across all CPU
//!   cores — hashbrown allows any number of `&self` readers, so this
//!   is hashbrown's actual lookup ceiling.
//!
//! Run with `./run-bench.sh` from the crate directory (sets the
//! cuda-oxide RUSTFLAGS and invokes `cargo run --release --bin bench`).

use std::sync::Arc;
use std::time::Instant;

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use hashbrown::HashMap as HbMap;
use hashmap_v2::*;
use rayon::prelude::*;

// =============================================================================
// CONFIG
// =============================================================================

/// GPU table capacity, in slots. Power of two so the probe `& mask` works.
/// 1 << 20 = ~1 M slots = 8 MiB of slot storage + 1 MiB of ctrl bytes.
const CAPACITY: usize = 1 << 20;

/// Load factors swept by every insert and find bench.
const LOAD_FACTORS: [(f32, &str); 3] = [(0.50, "50%"), (0.75, "75%"), (0.90, "90%")];

/// Untimed warmup iterations to settle the GPU clock and CPU caches.
const WARMUP: usize = 3;

/// Measured iterations averaged into the reported number.
const ITERS: usize = 10;

// =============================================================================
// CPU KEY GENERATOR (matches hashbrown's `RandomKeys` in benches/general_ops.rs)
// =============================================================================

/// `state = (state + 1) * 3787392781` — the same generator hashbrown's own
/// bench suite uses for its "random" key distribution. We dedup before
/// returning so the bench inputs are exact set sizes.
fn random_distinct_u32_keys(n: usize, seed: u64) -> Vec<u32> {
    let mut state = seed;
    let mut seen = HbMap::with_capacity(n);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        state = state.wrapping_add(1).wrapping_mul(3_787_392_781);
        let k = state as u32;
        if k != FORBIDDEN_KEY && seen.insert(k, ()).is_none() {
            out.push(k);
        }
    }
    out
}

// =============================================================================
// TIMING HELPERS
// =============================================================================

/// Convert ops + per-iter milliseconds → Mops/s.
fn mops(n_ops: usize, ms_per_iter: f64) -> f64 {
    (n_ops as f64 / 1e6) / (ms_per_iter / 1000.0)
}

/// Reset both ctrl and slots to all-`0xFF` (= EMPTY everywhere).
/// Async on `stream`. Used between insert iterations.
unsafe fn reset_table_async(
    map: &GpuSwissMap,
    stream: &Arc<CudaStream>,
) -> Result<(), Box<dyn std::error::Error>> {
    unsafe {
        cuda_core::simt::memory::memset_d8_async(
            map.ctrl.cu_deviceptr(),
            0xFF,
            map.ctrl.num_bytes(),
            stream.cu_stream(),
        )?;
        cuda_core::simt::memory::memset_d8_async(
            map.slots.cu_deviceptr(),
            0xFF,
            map.slots.num_bytes(),
            stream.cu_stream(),
        )?;
    }
    Ok(())
}

/// Time `iters` repetitions of `f` on `stream` with CUDA events. Returns
/// the average milliseconds per iteration. The closure is called once
/// per iteration with a single sub-stream synchronization implied by
/// the surrounding event record.
fn time_gpu_iters<F>(
    stream: &Arc<CudaStream>,
    iters: usize,
    mut f: F,
) -> Result<f64, Box<dyn std::error::Error>>
where
    F: FnMut() -> Result<(), Box<dyn std::error::Error>>,
{
    let start = stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    for _ in 0..iters {
        f()?;
    }
    let end = stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let elapsed = start.elapsed_ms(&end)? as f64;
    Ok(elapsed / iters as f64)
}

// =============================================================================
// GPU INSERT BENCH (one cell of the matrix)
// =============================================================================

/// Bench a single GPU insert kernel cell: warmup + timed iterations.
/// Each iteration resets the table and inserts every key in `keys`.
/// Returns Mops/s.
fn bench_gpu_insert<F>(
    map: &GpuSwissMap,
    keys_dev: &DeviceBuffer<u32>,
    values_dev: &DeviceBuffer<u32>,
    n_keys: usize,
    stream: &Arc<CudaStream>,
    mut launch: F,
) -> Result<f64, Box<dyn std::error::Error>>
where
    F: FnMut(
        &GpuSwissMap,
        &DeviceBuffer<u32>,
        &DeviceBuffer<u32>,
    ) -> Result<(), Box<dyn std::error::Error>>,
{
    for _ in 0..WARMUP {
        unsafe { reset_table_async(map, stream)? };
        launch(map, keys_dev, values_dev)?;
    }
    stream.synchronize()?;

    let ms = time_gpu_iters(stream, ITERS, || {
        unsafe { reset_table_async(map, stream)? };
        launch(map, keys_dev, values_dev)?;
        Ok(())
    })?;
    Ok(mops(n_keys, ms))
}

/// Bench a single GPU find kernel cell. Map is pre-built; only the
/// kernel launch is timed.
fn bench_gpu_find<F>(
    map: &GpuSwissMap,
    keys_dev: &DeviceBuffer<u32>,
    out_dev: &mut DeviceBuffer<u32>,
    n_keys: usize,
    stream: &Arc<CudaStream>,
    mut launch: F,
) -> Result<f64, Box<dyn std::error::Error>>
where
    F: FnMut(
        &GpuSwissMap,
        &DeviceBuffer<u32>,
        &mut DeviceBuffer<u32>,
    ) -> Result<(), Box<dyn std::error::Error>>,
{
    for _ in 0..WARMUP {
        launch(map, keys_dev, out_dev)?;
    }
    stream.synchronize()?;

    let ms = time_gpu_iters(stream, ITERS, || {
        launch(map, keys_dev, out_dev)?;
        Ok(())
    })?;
    Ok(mops(n_keys, ms))
}

// =============================================================================
// CPU BENCH HELPERS
// =============================================================================

/// Build a fresh `HashMap` of `n_keys` entries. Returns the median Mops/s
/// across `ITERS` rebuilds (single-threaded; hashbrown insert needs
/// `&mut self`).
fn bench_cpu_insert(keys: &[u32], values: &[u32]) -> f64 {
    let n = keys.len();
    for _ in 0..WARMUP {
        let mut m = HbMap::with_capacity(n);
        for i in 0..n {
            m.insert(keys[i], values[i]);
        }
        std::hint::black_box(&m);
    }

    let mut total_ms = 0.0;
    for _ in 0..ITERS {
        let mut m = HbMap::with_capacity(n);
        let t0 = Instant::now();
        for i in 0..n {
            m.insert(keys[i], values[i]);
        }
        total_ms += t0.elapsed().as_secs_f64() * 1000.0;
        std::hint::black_box(&m);
    }
    let avg_ms = total_ms / ITERS as f64;
    mops(n, avg_ms)
}

/// Time `map.get(&k)` for every key in `query_keys`, parallelized via
/// rayon across all available CPU cores. `&self` only — hashbrown
/// allows any number of concurrent readers.
fn bench_cpu_find(map: &HbMap<u32, u32>, query_keys: &[u32]) -> f64 {
    for _ in 0..WARMUP {
        query_keys.par_iter().for_each(|k| {
            std::hint::black_box(map.get(k));
        });
    }

    let mut total_ms = 0.0;
    for _ in 0..ITERS {
        let t0 = Instant::now();
        query_keys.par_iter().for_each(|k| {
            std::hint::black_box(map.get(k));
        });
        total_ms += t0.elapsed().as_secs_f64() * 1000.0;
    }
    let avg_ms = total_ms / ITERS as f64;
    mops(query_keys.len(), avg_ms)
}

// =============================================================================
// MAIN
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("GPU Hashmap v2 — Performance vs CPU hashbrown");
    println!("============================================================");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // The device artifact is embedded in this binary, so the bench needs no
    // loose `hashmap_v2.ptx` beside the manifest -- the same load `main.rs`
    // already uses.
    let module = kernels::load(&ctx)?;

    print_environment_banner(&ctx)?;

    println!("Bench config:");
    println!(
        "  capacity:    {CAPACITY} slots (= 1 << {})",
        CAPACITY.trailing_zeros()
    );
    println!("  warmup:      {WARMUP} iterations");
    println!("  measured:    {ITERS} iterations");
    println!("  GPU timing:  CUDA events, kernel-only (no H2D/D2H)");
    println!(
        "  CPU find:    rayon ({} threads)",
        rayon::current_num_threads()
    );
    println!();

    // Single key pool large enough to cover the largest load factor + a
    // disjoint miss-set of the same size. RandomKeys with hashbrown's prime.
    let max_n = (CAPACITY as f32 * LOAD_FACTORS[2].0) as usize;
    let pool = random_distinct_u32_keys(2 * max_n, 0xCAFE_F00D_DEAD_BEEF);

    // Per-load-factor cells.
    let mut row_b_insert = [0.0f64; 3];
    let mut row_a_insert = [0.0f64; 3];
    let mut row_single_lookup = [0.0f64; 3];
    let mut row_warp_lookup = [0.0f64; 3];
    let mut row_warp_typed_lookup = [0.0f64; 3];
    let mut row_single_lookup_fail = [0.0f64; 3];
    let mut row_warp_lookup_fail = [0.0f64; 3];
    let mut row_warp_typed_lookup_fail = [0.0f64; 3];
    let mut row_cpu_insert = [0.0f64; 3];
    let mut row_cpu_lookup = [0.0f64; 3];
    let mut row_cpu_lookup_fail = [0.0f64; 3];

    for (col_idx, &(load, label)) in LOAD_FACTORS.iter().enumerate() {
        let n_keys = (CAPACITY as f32 * load) as usize;
        let keys: Vec<u32> = pool[..n_keys].to_vec();
        let absent: Vec<u32> = pool[n_keys..2 * n_keys].to_vec();
        let values: Vec<u32> = (0..n_keys as u32).collect();

        println!(
            "[load = {:>3}, {:>10} inserted keys, {:>10} miss-query keys]",
            label, n_keys, n_keys
        );

        // ---- Pre-upload device buffers (untimed) ------------------------
        let keys_dev = DeviceBuffer::from_host(&stream, &keys)?;
        let absent_dev = DeviceBuffer::from_host(&stream, &absent)?;
        let values_dev = DeviceBuffer::from_host(&stream, &values)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, n_keys)?;

        // ---- GPU INSERT — Protocol B -----------------------------------
        let map = GpuSwissMap::new(CAPACITY, &stream)?;
        let cfg = LaunchConfig::for_num_elems(n_keys as u32);
        row_b_insert[col_idx] =
            bench_gpu_insert(&map, &keys_dev, &values_dev, n_keys, &stream, |m, k, v| {
                // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
                unsafe { module.insert_kernel(&stream, cfg, &m.ctrl, &m.slots, k, v) }?;
                Ok(())
            })?;

        // ---- GPU INSERT — Protocol A -----------------------------------
        row_a_insert[col_idx] =
            bench_gpu_insert(&map, &keys_dev, &values_dev, n_keys, &stream, |m, k, v| {
                // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
                unsafe { module.insert_kernel_proto_a(&stream, cfg, &m.ctrl, &m.slots, k, v) }?;
                Ok(())
            })?;

        // ---- Build a final B-protocol map for the find benches ----------
        unsafe { reset_table_async(&map, &stream)? };
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.insert_kernel(&stream, cfg, &map.ctrl, &map.slots, &keys_dev, &values_dev)
        }?;
        stream.synchronize()?;

        let cfg_warp = LaunchConfig::for_num_elems((n_keys * PROBE_TILE) as u32);

        // ---- GPU LOOKUP (hits) — single-thread --------------------------
        row_single_lookup[col_idx] =
            bench_gpu_find(&map, &keys_dev, &mut out_dev, n_keys, &stream, |m, k, o| {
                // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
                unsafe { module.find_kernel(&stream, cfg, &m.ctrl, &m.slots, k, o) }?;
                Ok(())
            })?;

        // ---- GPU LOOKUP (hits) — warp-cooperative -----------------------
        row_warp_lookup[col_idx] =
            bench_gpu_find(&map, &keys_dev, &mut out_dev, n_keys, &stream, |m, k, o| {
                // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
                unsafe { module.find_kernel_warp(&stream, cfg_warp, &m.ctrl, &m.slots, k, o) }?;
                Ok(())
            })?;

        // ---- GPU LOOKUP (hits) — warp-cooperative, typed CG API ---------
        row_warp_typed_lookup[col_idx] =
            bench_gpu_find(&map, &keys_dev, &mut out_dev, n_keys, &stream, |m, k, o| {
                // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
                unsafe {
                    module.find_kernel_warp_typed(&stream, cfg_warp, &m.ctrl, &m.slots, k, o)
                }?;
                Ok(())
            })?;

        // ---- GPU LOOKUP_FAIL (misses) — single-thread -------------------
        row_single_lookup_fail[col_idx] = bench_gpu_find(
            &map,
            &absent_dev,
            &mut out_dev,
            n_keys,
            &stream,
            |m, k, o| {
                // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
                unsafe { module.find_kernel(&stream, cfg, &m.ctrl, &m.slots, k, o) }?;
                Ok(())
            },
        )?;

        // ---- GPU LOOKUP_FAIL (misses) — warp-cooperative ----------------
        row_warp_lookup_fail[col_idx] = bench_gpu_find(
            &map,
            &absent_dev,
            &mut out_dev,
            n_keys,
            &stream,
            |m, k, o| {
                // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
                unsafe { module.find_kernel_warp(&stream, cfg_warp, &m.ctrl, &m.slots, k, o) }?;
                Ok(())
            },
        )?;

        // ---- GPU LOOKUP_FAIL (misses) — warp-cooperative, typed CG API --
        row_warp_typed_lookup_fail[col_idx] = bench_gpu_find(
            &map,
            &absent_dev,
            &mut out_dev,
            n_keys,
            &stream,
            |m, k, o| {
                // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
                unsafe {
                    module.find_kernel_warp_typed(&stream, cfg_warp, &m.ctrl, &m.slots, k, o)
                }?;
                Ok(())
            },
        )?;

        // ---- CPU INSERT — single-threaded hashbrown ---------------------
        row_cpu_insert[col_idx] = bench_cpu_insert(&keys, &values);

        // ---- Build the CPU map once for both find benches ---------------
        let mut cpu_map: HbMap<u32, u32> = HbMap::with_capacity(n_keys);
        for i in 0..n_keys {
            cpu_map.insert(keys[i], values[i]);
        }

        // ---- CPU LOOKUP (hits) — rayon-parallel hashbrown ---------------
        row_cpu_lookup[col_idx] = bench_cpu_find(&cpu_map, &keys);

        // ---- CPU LOOKUP_FAIL (misses) — rayon-parallel hashbrown --------
        row_cpu_lookup_fail[col_idx] = bench_cpu_find(&cpu_map, &absent);

        println!("    ... done.");
    }

    print_section_header("Insert (Mops/s; higher is better)");
    print_row("GPU Protocol B           ", &row_b_insert);
    print_row("GPU Protocol A           ", &row_a_insert);
    print_row("CPU hashbrown (1 thread) ", &row_cpu_insert);
    print_ratios("GPU-B / CPU              ", &row_b_insert, &row_cpu_insert);
    print_ratios("GPU-A / CPU              ", &row_a_insert, &row_cpu_insert);

    print_section_header("Find — lookup (every query hits)");
    print_row("GPU single-thread        ", &row_single_lookup);
    print_row("GPU warp-cooperative     ", &row_warp_lookup);
    print_row("GPU warp-typed (CG API)  ", &row_warp_typed_lookup);
    print_row("CPU hashbrown (rayon)    ", &row_cpu_lookup);
    print_ratios(
        "GPU-single     / CPU     ",
        &row_single_lookup,
        &row_cpu_lookup,
    );
    print_ratios(
        "GPU-warp       / CPU     ",
        &row_warp_lookup,
        &row_cpu_lookup,
    );
    print_ratios(
        "GPU-warp-typed / CPU     ",
        &row_warp_typed_lookup,
        &row_cpu_lookup,
    );
    print_ratios(
        "typed / hand-written     ",
        &row_warp_typed_lookup,
        &row_warp_lookup,
    );

    print_section_header("Find — lookup_fail (every query misses)");
    print_row("GPU single-thread        ", &row_single_lookup_fail);
    print_row("GPU warp-cooperative     ", &row_warp_lookup_fail);
    print_row("GPU warp-typed (CG API)  ", &row_warp_typed_lookup_fail);
    print_row("CPU hashbrown (rayon)    ", &row_cpu_lookup_fail);
    print_ratios(
        "GPU-single     / CPU     ",
        &row_single_lookup_fail,
        &row_cpu_lookup_fail,
    );
    print_ratios(
        "GPU-warp       / CPU     ",
        &row_warp_lookup_fail,
        &row_cpu_lookup_fail,
    );
    print_ratios(
        "GPU-warp-typed / CPU     ",
        &row_warp_typed_lookup_fail,
        &row_cpu_lookup_fail,
    );
    print_ratios(
        "typed / hand-written     ",
        &row_warp_typed_lookup_fail,
        &row_warp_lookup_fail,
    );

    println!();
    Ok(())
}

fn print_section_header(title: &str) {
    println!();
    println!("------------------------------------------------------------");
    println!("{title}");
    println!("------------------------------------------------------------");
    println!("                          load=50%   load=75%   load=90%");
}

fn print_row(label: &str, row: &[f64; 3]) {
    println!("{label}{:>9.1} {:>9.1} {:>9.1}", row[0], row[1], row[2]);
}

fn print_ratios(label: &str, gpu: &[f64; 3], cpu: &[f64; 3]) {
    let ratio = |g: f64, c: f64| if c > 0.0 { g / c } else { 0.0 };
    println!(
        "{label}{:>8.1}x {:>8.1}x {:>8.1}x",
        ratio(gpu[0], cpu[0]),
        ratio(gpu[1], cpu[1]),
        ratio(gpu[2], cpu[2])
    );
}

/// Print a banner line matching the smoketest's convention:
///
///   hashmap_v2 bench @ <git-sha> (<branch>)
///   GPU: <name>, <cap.major>.<cap.minor>
///   PTX arch: sm_<NNN>
///
/// The git data shells out to `git`. The GPU data uses the cuda-core
/// driver wrappers so it works even without `nvidia-smi` on PATH.
fn print_environment_banner(
    ctx: &Arc<cuda_core::CudaContext>,
) -> Result<(), Box<dyn std::error::Error>> {
    let git_head = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_else(|| "?".to_string());
    let git_branch = std::process::Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_else(|| "?".to_string());

    let gpu_name = ctx.device_name().unwrap_or_else(|_| "?".to_string());
    let (cap_major, cap_minor) = ctx.compute_capability().unwrap_or((0, 0));

    println!("hashmap_v2 bench @ {git_head} ({git_branch})");
    println!("GPU: {gpu_name}, {cap_major}.{cap_minor}");
    println!("PTX arch: sm_{cap_major}{cap_minor}");
    println!();
    Ok(())
}
