/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Verification benchmarks for warmup and in-memory compilation caching.
//!
//! Tests assert cache behavior via [`jit_compile_count`] (process-global counter,
//! +1 per real JIT compile, +0 on cache hits) — not wall-clock timing. Durations
//! are printed for inspection but never asserted.
//!
//! All tests hold [`common::cache_test_lock`] to prevent concurrent tests from
//! moving the counter during the measured window.
//!
//! Run with `CUTILE_JIT_LOG=1 cargo test -p cutile --test warmup_suite warmup_bench -- --nocapture`
//! to see per-compile vs cache-hit logs.

use crate::common;
use cutile::api;
use cutile::prelude::{DeviceOp, PartitionOp};
use cutile::tile_kernel::{
    contains_cuda_function, get_default_device, jit_compile_count, TileFunctionKey, TileKernel,
};
use cutile_compiler::cuda_tile_runtime_utils::{
    get_compiler_version, get_gpu_name, tileiras_fingerprint,
};
use cutile_compiler::specialization::SpecializationBits;
use std::time::Instant;

// Distinct module name avoids cache key collisions with warmup.rs tests.
#[cutile::module]
mod bench_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn vector_add<T: ElementType, const N: i32>(
        z: &mut Tensor<T, { [N] }>,
        x: &Tensor<T, { [-1] }>,
        y: &Tensor<T, { [-1] }>,
    ) {
        let tile_x = load_tile_like(x, z);
        let tile_y = load_tile_like(y, z);
        z.store(tile_x + tile_y);
    }
}

fn stride_args() -> Vec<(String, Vec<i32>)> {
    vec![
        ("z".to_string(), vec![1]),
        ("x".to_string(), vec![1]),
        ("y".to_string(), vec![1]),
    ]
}

fn vector_add_spec_args(len: usize, tile: usize) -> Vec<(String, SpecializationBits)> {
    let x = api::ones::<f32>(&[len]).sync().unwrap();
    let y = api::ones::<f32>(&[len]).sync().unwrap();
    let z = api::zeros::<f32>(&[len]).partition([tile]).sync().unwrap();
    let z_spec = z.unpartition().spec().clone();
    vec![
        ("z".to_string(), z_spec),
        ("x".to_string(), x.spec().clone()),
        ("y".to_string(), y.spec().clone()),
    ]
}

fn bench_key(
    generics: Vec<String>,
    spec_args: Vec<(String, SpecializationBits)>,
) -> TileFunctionKey {
    let device_id = get_default_device();
    TileFunctionKey::builder("bench_module", "vector_add")
        .generics(generics)
        .stride_args(stride_args())
        .spec_args(spec_args)
        .source_hash(bench_module::_SOURCE_HASH)
        .device_id(device_id)
        .gpu_name(get_gpu_name(device_id))
        .compiler_version(get_compiler_version())
        .tileiras_fingerprint(tileiras_fingerprint())
        .build()
}

fn timed_kernel_call(tile_size: &str) -> std::time::Duration {
    let n: usize = tile_size.parse().unwrap();
    let t0 = Instant::now();
    let x = api::ones::<f32>(&[256]).sync().unwrap();
    let y = api::ones::<f32>(&[256]).sync().unwrap();
    let z = api::zeros::<f32>(&[256]).partition([n]).sync().unwrap();
    let _result = bench_module::vector_add(z, &x, &y)
        .generics(vec!["f32".into(), tile_size.into()])
        .sync()
        .unwrap();
    t0.elapsed()
}

// tile=32: called without warmup → first call is the JIT compile (miss).
// tile=64: pre-compiled by warmup → first real call is a cache hit.
// Fill kernel primed via spec_args_64 so only vector_add moves the counter.
#[test]
fn warmup_eliminates_first_call_jit() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();

        // Prime fill kernel so only vector_add moves the counter below.
        let spec_args_64 = vector_add_spec_args(256, 64);

        let c0 = jit_compile_count();
        let cold_duration = timed_kernel_call("32");
        let c_after_cold = jit_compile_count();
        assert_eq!(
            c_after_cold,
            c0 + 1,
            "un-warmed first call to tile=32 must perform exactly one JIT \
             compile (only bench_module::vector_add; full_apply was primed)"
        );

        let warmup_t0 = Instant::now();
        let z = api::meta::<f32>(&[256]).partition([64]);
        let x = api::meta::<f32>(&[256]);
        let y = api::meta::<f32>(&[256]);
        bench_module::vector_add(z, x, y)
            .generics(vec!["f32".into(), "64".into()])
            .compile()
            .expect("meta .compile() warmup failed");
        let warmup_duration = warmup_t0.elapsed();
        let c_after_warmup = jit_compile_count();
        assert_eq!(
            c_after_warmup,
            c_after_cold + 1,
            ".compile() warmup for tile=64 must perform exactly one JIT compile"
        );

        let warm_duration = timed_kernel_call("64");
        let c_after_warm = jit_compile_count();
        assert_eq!(
            c_after_warm, c_after_warmup,
            "warmed-up first call to tile=64 must NOT compile (cache hit): \
             counter moved from {c_after_warmup} to {c_after_warm}"
        );

        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║         Warmup Verification: First-Call Latency         ║");
        println!("╠══════════════════════════════════════════════════════════╣");
        println!(
            "║  Without warmup (tile=32): {:>10.1?}  (includes JIT)  ║",
            cold_duration
        );
        println!(
            "║  Warmup step     (tile=64): {:>10.1?}  (pre-compile)   ║",
            warmup_duration
        );
        println!(
            "║  With warmup     (tile=64): {:>10.1?}  (cache hit)     ║",
            warm_duration
        );
        println!("╠══════════════════════════════════════════════════════════╣");
        println!("║  JIT compiles: cold +1, warmup +1, warmed call +0       ║");
        println!("╚══════════════════════════════════════════════════════════╝\n");

        let key = bench_key(vec!["f32".into(), "64".into()], spec_args_64);
        assert!(
            contains_cuda_function(&key),
            "kernel should be in memory cache after warmup"
        );
    });
}

#[test]
fn second_call_hits_memory_cache() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();

        // tile=16 is unique to this test; priming fill kernel keeps counter clean.
        let spec_args_16 = vector_add_spec_args(256, 16);
        let c0 = jit_compile_count();

        let first = timed_kernel_call("16");
        let c_after_first = jit_compile_count();
        assert_eq!(
            c_after_first,
            c0 + 1,
            "first call to tile=16 must perform exactly one JIT compile \
             (only bench_module::vector_add; full_apply was primed)"
        );

        let second = timed_kernel_call("16");
        let c_after_second = jit_compile_count();
        assert_eq!(
            c_after_second, c_after_first,
            "second call to tile=16 must NOT compile (cache hit): \
             counter moved from {c_after_first} to {c_after_second}"
        );

        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║       Memory Cache Verification: 1st vs 2nd Call        ║");
        println!("╠══════════════════════════════════════════════════════════╣");
        println!(
            "║  First  call (tile=16): {:>10.1?}  (JIT: +1 compile)  ║",
            first
        );
        println!(
            "║  Second call (tile=16): {:>10.1?}  (cache: +0 compile)║",
            second
        );
        println!("╚══════════════════════════════════════════════════════════╝\n");

        let key = bench_key(vec!["f32".into(), "16".into()], spec_args_16);
        assert!(
            contains_cuda_function(&key),
            "tile=16 kernel should be in memory cache after first call"
        );
    });
}

// Acceptance test for the new `.compile()` terminal + `api::meta` warmup path.
//
// Warms the cache with zero-allocation meta tensors and no launch, then proves a
// real `.sync()` launch of the same shape/generics hits that entry (+0 compiles).
// tile=8 is unique to this test so its key never collides with the others.
#[test]
fn meta_compile_terminal_warms_cache() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();

        let _prime = api::ones::<f32>(&[256]).sync().unwrap();

        let generics = vec!["f32".to_string(), "8".to_string()];
        let c0 = jit_compile_count();

        // Warmup: same call you would launch, but meta inputs + `.compile()`.
        // No GPU allocation, no launch — just JIT-compile and cache.
        let z = api::meta::<f32>(&[256]).partition([8]);
        let x = api::meta::<f32>(&[256]);
        let y = api::meta::<f32>(&[256]);
        bench_module::vector_add(z, x, y)
            .generics(generics.clone())
            .compile()
            .expect("meta .compile() warmup failed");
        let c_after_compile = jit_compile_count();
        assert_eq!(
            c_after_compile,
            c0 + 1,
            "meta .compile() must JIT-compile exactly once"
        );

        // Real launch of the same specialization must hit the warmed entry.
        let real_x = api::ones::<f32>(&[256]).sync().unwrap();
        let real_y = api::ones::<f32>(&[256]).sync().unwrap();
        let real_z = api::zeros::<f32>(&[256]).partition([8]).sync().unwrap();
        let _ = bench_module::vector_add(real_z, &real_x, &real_y)
            .generics(generics)
            .sync()
            .unwrap();
        let c_after_launch = jit_compile_count();
        assert_eq!(
            c_after_launch, c_after_compile,
            "real launch after meta .compile() must NOT recompile (key must \
             match): counter moved from {c_after_compile} to {c_after_launch}"
        );
    });
}

// Summary statistics over a slice of per-iteration durations.
fn report(label: &str, samples: &[std::time::Duration]) {
    assert!(!samples.is_empty(), "no samples for {label}");
    let mut ns: Vec<u128> = samples.iter().map(|d| d.as_nanos()).collect();
    ns.sort_unstable();
    let n = ns.len();
    let pct = |p: f64| ns[((p * (n as f64 - 1.0)).round() as usize).min(n - 1)];
    let mean = ns.iter().sum::<u128>() / n as u128;
    println!(
        "  {label:<48} n={n:>5}  min={:>8.3?}  median={:>8.3?}  mean={:>8.3?}  p99={:>8.3?}  max={:>8.3?}",
        std::time::Duration::from_nanos(ns[0] as u64),
        std::time::Duration::from_nanos(pct(0.50) as u64),
        std::time::Duration::from_nanos(mean as u64),
        std::time::Duration::from_nanos(pct(0.99) as u64),
        std::time::Duration::from_nanos(ns[n - 1] as u64),
    );
}

/// Measures the per-launch cost of building the hardened key on the cache-hit
/// path, three ways:
///
///   (A) End-to-end warmed launch — real per-call latency. `jit_compile_count`
///       is asserted flat across the loop, so every iteration is a cache hit.
///   (B) Build hardened key, no launch — the extra CPU work the hit path does
///       before the lookup. The key *is* the lookup key now, so no digest is
///       computed here.
///   (C) `get_gpu_name()` alone — cached `OnceLock` lookup + `String` clone.
///
/// Run with:
///   cargo test -p cutile --test warmup_suite warmup_bench::cache_hit_path_cost -- --nocapture
#[test]
fn cache_hit_path_cost() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();

        const LAUNCH_ITERS: usize = 500;
        const CPU_ITERS: usize = 5000;
        let tile = "256";
        let generics = vec!["f32".to_string(), tile.to_string()];

        // Prime: first call to this tile JIT-compiles (miss); everything after
        // is a cache hit. Fill kernel primed so only vector_add moves the counter.
        let spec_args = vector_add_spec_args(256, 256);
        let c0 = jit_compile_count();
        let _ = timed_kernel_call(tile);
        let c_after_prime = jit_compile_count();
        assert_eq!(
            c_after_prime,
            c0 + 1,
            "prime call must JIT-compile exactly once (only bench_module::vector_add)"
        );

        // (A) End-to-end warmed launches. Counter must stay flat => all hits.
        let mut launch_samples = Vec::with_capacity(LAUNCH_ITERS);
        for _ in 0..LAUNCH_ITERS {
            launch_samples.push(timed_kernel_call(tile));
        }
        let c_after_launch = jit_compile_count();
        assert_eq!(
            c_after_launch, c_after_prime,
            "every launch in the loop must be a cache hit (no recompile): \
             jit_compile_count moved from {c_after_prime} to {c_after_launch}"
        );

        // (B) Isolated added cost: build the hardened key, no launch. Clones are
        // hoisted out of the timed region so we measure the build (which includes
        // get_gpu_name + version lookups), not the clones.
        let device_id = get_default_device();
        let mut key_samples = Vec::with_capacity(CPU_ITERS);
        for _ in 0..CPU_ITERS {
            let g = generics.clone();
            let s = spec_args.clone();
            let t0 = Instant::now();
            let key = std::hint::black_box(bench_key(g, s));
            key_samples.push(t0.elapsed());
            drop(key);
        }

        // (C) get_gpu_name() in isolation — cached OnceLock lookup + String clone per launch.
        let mut gpu_name_samples = Vec::with_capacity(CPU_ITERS);
        for _ in 0..CPU_ITERS {
            let t0 = Instant::now();
            let name = std::hint::black_box(get_gpu_name(device_id));
            gpu_name_samples.push(t0.elapsed());
            drop(name);
        }

        println!("\n=== cache-hit-path cost (tile={tile}, f32) ===");
        report(
            "(A) end-to-end warmed launch (real per-call)",
            &launch_samples,
        );
        report(
            "(B) build hardened key (added per-launch CPU)",
            &key_samples,
        );
        report(
            "(C) get_gpu_name() only (OnceLock lookup + String clone)",
            &gpu_name_samples,
        );
    });
}

/// (B) under concurrency: runs the full hit-path key build across thread
/// counts. `get_gpu_name()` is lock-free on the hot path, while cached
/// `tileiras` resolution and fingerprint lookups use short-lived mutexes. This
/// measures their aggregate contention together with the rest of key building.
///
///   threads=1 per-call ties back to (B) (~5.5µs).
///   Flat per-call as threads grow => no material contention in the full path.
///
/// Run with:
///   cargo test -p cutile --test warmup_suite warmup_bench::hit_path_contention -- --nocapture
#[test]
fn hit_path_contention() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();

        let generics = std::sync::Arc::new(vec!["f32".to_string(), "128".to_string()]);
        let spec_args = std::sync::Arc::new(vector_add_spec_args(256, 128));

        // Prime the complete path so workers measure steady-state key building,
        // not one-time GPU queries, tileiras resolution, or `--version` probing.
        drop(bench_key((*generics).clone(), (*spec_args).clone()));

        const CALLS_PER_THREAD: usize = 20_000;
        println!("\n=== hit-path (build key) contention ===");
        for threads in [1usize, 2, 4, 8, 16] {
            let barrier = std::sync::Arc::new(std::sync::Barrier::new(threads));
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    let b = barrier.clone();
                    let g = generics.clone();
                    let s = spec_args.clone();
                    std::thread::spawn(move || {
                        // Release all threads at once for real contention.
                        b.wait();
                        let t0 = Instant::now();
                        for _ in 0..CALLS_PER_THREAD {
                            drop(std::hint::black_box(bench_key((*g).clone(), (*s).clone())));
                        }
                        t0.elapsed()
                    })
                })
                .collect();

            let mut wall = std::time::Duration::ZERO; // slowest thread
            let mut sum = std::time::Duration::ZERO; // for mean per-call
            for h in handles {
                let e = h.join().unwrap();
                sum += e;
                wall = wall.max(e);
            }
            let total_calls = (threads * CALLS_PER_THREAD) as u32;
            let per_call = sum / total_calls;
            let throughput = total_calls as f64 / wall.as_secs_f64();
            println!(
                "  threads={threads:>2}  per-call(mean)={per_call:>9.3?}  wall={wall:>9.3?}  throughput={throughput:>12.0}/s",
            );
        }
        println!(
            "Read as: threads=1 ties back to (B); flat per-call as threads grow \
             => no material contention in the full key path.\n"
        );
    });
}
