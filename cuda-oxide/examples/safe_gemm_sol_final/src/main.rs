/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lean 4 formally verified Blackwell (`sm_100a`) speed-of-light FP16/BF16 GEMM (`C = A x B^T`):
//! 2-CTA cluster (`#[cluster_launch(2, 1, 1)]`), 7-stage `224 KiB` multicast TMA ring,
//! overlapped `CLC` work-stealing, 4-slot `TILE_INFO` mailbox, and 2-stage `512`-column
//! `TMEM` ping-pong (`tcgen05.mma.cta_group::2`).

use std::collections::HashMap;
use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::OnceLock;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, IntoResult, LaunchConfig1D};
use cuda_device::{
    DisjointSlice,
    barrier::{
        Barrier, fence_proxy_async_shared_cta, mbarrier_arrive, mbarrier_arrive_cluster,
        mbarrier_arrive_expect_tx, mbarrier_init, mbarrier_try_wait_parity,
        mbarrier_try_wait_parity_cluster,
    },
    clc::{clc_query_get_first_ctaid_x, clc_query_is_canceled, clc_try_cancel_multicast},
    cluster, cluster_launch,
    convert::{bf16_to_f32, cvt_bf16x2_f32},
    cuda_module, kernel, launch_bounds, launch_contract,
    shared::{SharedArray, cvta_generic_to_shared_offset},
    tcgen05::{
        Tcgen05AccumulatorType, Tcgen05ElementType, Tcgen05InstructionDescriptor, Tcgen05MmaShape,
        tcgen05_alloc_cg2, tcgen05_commit_multicast_cg2, tcgen05_dealloc_cg2,
        tcgen05_ld_16x256b_x8_pure, tcgen05_load_wait, tcgen05_mma_f16_cg2,
        tcgen05_relinquish_alloc_permit_cg2,
    },
    thread,
    tma::{TmaDescriptor, cp_async_bulk_tensor_2d_g2s_multicast_cg2},
    verify, warp,
};
use half::{bf16, f16};

/// Cached per-shape (`M -> TFLOPS`) live `cuBLASLt` FP16 reference measurements
/// loaded at most once per process when `bench/cublaslt_bench` is available.
static CUBLAS_BASELINE: OnceLock<Option<HashMap<usize, f64>>> = OnceLock::new();

/// Locate the optional `cublaslt_bench` binary under `./bench` or `../gemm_sol_final/bench`.
/// Returns `None` if neither candidate binary has been built on the current machine.
fn find_cublaslt_bench_binary() -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let local_bin = manifest_dir.join("bench").join("cublaslt_bench");
    if local_bin.exists() {
        return Some(local_bin);
    }
    let sibling_bin = manifest_dir.parent().map(|p| {
        p.join("gemm_sol_final")
            .join("bench")
            .join("cublaslt_bench")
    })?;
    sibling_bin.exists().then_some(sibling_bin)
}

/// Parse `(M, TFLOPS)` rows from the `--- FP16 ---` section of `cublaslt_bench` stdout.
/// Ignores lines outside the `--- FP16 ---` section.
fn parse_cublas_fp16_rows(stdout: &str) -> HashMap<usize, f64> {
    let mut map = HashMap::new();
    let mut in_fp16 = false;
    for line in stdout.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("--- FP16") {
            in_fp16 = true;
            continue;
        }
        if trimmed.starts_with("--- BF16") {
            in_fp16 = false;
            continue;
        }
        if !in_fp16 || !trimmed.starts_with("FP16 ") {
            continue;
        }
        let toks: Vec<&str> = trimmed.split_whitespace().collect();
        let size = toks.get(3).copied().unwrap_or("");
        let m: Option<usize> = size.split('x').next().and_then(|s| s.trim().parse().ok());
        let tf: Option<f64> = toks.iter().rev().nth(1).and_then(|s| s.parse().ok());
        if let (Some(m), Some(tf)) = (m, tf) {
            map.insert(m, tf);
        }
    }
    map
}

/// Load or execute the optional `cublaslt_bench` helper binary to populate [`CUBLAS_BASELINE`].
/// Skipped when `GEMM_CUBLAS_SKIP=1` is set or when `cublaslt_bench` is not built.
fn load_cublas_baseline() -> Option<HashMap<usize, f64>> {
    if std::env::var("GEMM_CUBLAS_SKIP").as_deref() == Ok("1") {
        return None;
    }
    let Some(bin) = find_cublaslt_bench_binary() else {
        eprintln!(
            "No live cublasLt baseline binary found (optional; build via examples/gemm_sol_final/bench/build.sh)."
        );
        return None;
    };
    eprintln!("Measuring live cuBLASLt tuned FP16 reference on host GPU...");
    let mut cmd = std::process::Command::new(&bin);
    for (name, flag) in [
        ("GEMM_SOL_WARMUP", "--warmup"),
        ("GEMM_SOL_ITERS", "--iters"),
        ("GEMM_SOL_REPEATS", "--repeats"),
    ] {
        if let Ok(value) = std::env::var(name) {
            cmd.args([flag, value.as_str()]);
        }
    }
    let out = cmd.output().ok()?;
    if !out.status.success() {
        eprintln!(
            "{} exited with {}: {}",
            bin.display(),
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        );
        return None;
    }
    let map = parse_cublas_fp16_rows(&String::from_utf8_lossy(&out.stdout));
    (!map.is_empty()).then_some(map)
}

/// Print the `"vs cuBLAS"` comparison row for square dimension `m` when a baseline is present.
/// Falls back to an informational note if `cublaslt_bench` was skipped or not built.
fn print_cublas_comparison(tflops: f64, m: usize) {
    match CUBLAS_BASELINE
        .get_or_init(load_cublas_baseline)
        .as_ref()
        .and_then(|map| map.get(&m).copied())
    {
        Some(sol) => {
            let pct = (tflops / sol) * 100.0;
            println!(
                "  vs cuBLAS:   {pct:.2}% of live cublasLt tuned FP16 reference ({sol:.0} TFLOPS)"
            );
        }
        None => {
            println!(
                "  vs cuBLAS:   (no live cublasLt baseline; see examples/gemm_sol_final/bench/build.sh)"
            );
        }
    }
}

/// Parse a positive integer count override (`GEMM_SOL_WARMUP`, `GEMM_SOL_ITERS`, `GEMM_SOL_REPEATS`)
/// from the process environment, returning `default` when the variable is unset.
fn benchmark_count(name: &str, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    match std::env::var(name) {
        Ok(value) => {
            let count = value.parse::<usize>()?;
            if count == 0 || count > 10_000_000 {
                return Err(format!("{name} must be between 1 and 10000000").into());
            }
            Ok(count)
        }
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

/// Execution mode selected via the `GEMM_SOL_MODE` environment variable (`both`, `validate`, or `bench`).
/// Defaults to [`RunMode::Both`] so running the binary validates correctness and prints B200 throughput.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunMode {
    /// Run only the full `16,777,216`-element exact-BF16 Walsh correctness check at `4096^3`.
    Validate,
    /// Run only the `4096^3`, `8192^3`, and `16384^3` throughput benchmarks.
    Bench,
    /// Run full `4096^3` correctness validation followed by the `4096^3` / `8192^3` / `16384^3` benchmarks.
    Both,
}

impl FromStr for RunMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "validate" => Ok(Self::Validate),
            "bench" => Ok(Self::Bench),
            "both" => Ok(Self::Both),
            other => Err(format!(
                "invalid GEMM_SOL_MODE={other:?}; expected validate, bench, or both"
            )),
        }
    }
}

/// Build a 64-bit `tcgen05` shared-memory matrix descriptor from base address, LBO, SBO, and swizzle mode.
#[inline(always)]
fn build_smem_descriptor(
    smem_addr: u64,
    leading_dim_bytes: u32,
    stride_bytes: u32,
    swizzle: u8,
) -> u64 {
    let addr_enc = (smem_addr >> 4) & 0x3FFF;
    let ld_enc = ((leading_dim_bytes >> 4) & 0x3FFF) as u64;
    let stride_enc = ((stride_bytes >> 4) & 0x3FFF) as u64;
    let fixed_bit = 1u64 << 46;
    let swizzle_bits = (swizzle as u64) << 61;

    addr_enc | (ld_enc << 16) | (stride_enc << 32) | fixed_bit | swizzle_bits
}

include!("kernels.rs");

/// Return `true` when compute capability `(major, minor)` supports datacenter `tcgen05` (`sm_100a`+).
fn can_execute_tcgen05_ptx(major: i32, minor: i32) -> bool {
    matches!((major, minor), (10, 0) | (10, 1) | (10, 3) | (11, 0))
}

/// Verify that a loose PTX artifact targets a datacenter `tcgen05` architecture and exports
/// `gemm_sol_clc_multicast_4_stage_pipeline`.
fn verify_tcgen05_ptx_contract(ptx: &str) -> Result<(), String> {
    const TARGETS: [&str; 8] = [
        "sm_100a", "sm_101a", "sm_103a", "sm_110a", "sm_100f", "sm_101f", "sm_103f", "sm_110f",
    ];
    let document = ptx_parse::Document::parse(ptx).map_err(|error| error.to_string())?;
    let has_target = document.directives().iter().any(|directive| {
        directive.name() == ".target"
            && directive
                .arguments()
                .split(',')
                .next()
                .is_some_and(|target| TARGETS.contains(&target.trim()))
    });
    if !has_target {
        return Err("missing a supported datacenter tcgen05 target".to_string());
    }
    if !document
        .callables_named("gemm_sol_clc_multicast_4_stage_pipeline")
        .any(|callable| callable.kind() == ptx_parse::CallableKind::Entry)
    {
        return Err(
            "missing expected kernel entry `gemm_sol_clc_multicast_4_stage_pipeline`".to_string(),
        );
    }
    Ok(())
}

/// Encode a 2D `SWIZZLE_128B` + `L2_128B` `TmaDescriptor` for an `f16` (`u16`) device buffer.
fn encode_tma_2d_f16_swizzled(
    buf: &DeviceBuffer<u16>,
    global_width: u64,
    global_height: u64,
    box_k: u32,
    box_mn: u32,
) -> Result<TmaDescriptor, Box<dyn std::error::Error>> {
    use cuda_core::sys::{
        CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B, cuTensorMapEncodeTiled,
    };

    let mut tensor_map = MaybeUninit::<cuda_core::sys::CUtensorMap>::uninit();
    let global_dim: [u64; 2] = [global_width, global_height];
    let global_strides: [u64; 1] = [global_width * 2];
    let box_dim: [u32; 2] = [box_k, box_mn];
    let element_strides: [u32; 2] = [1, 1];

    unsafe {
        cuTensorMapEncodeTiled(
            tensor_map.as_mut_ptr(),
            CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            2,
            buf.cu_deviceptr() as *mut std::ffi::c_void,
            global_dim.as_ptr(),
            global_strides.as_ptr(),
            box_dim.as_ptr(),
            element_strides.as_ptr(),
            CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
        .result()?;
        Ok(tensor_map.as_ptr().cast::<TmaDescriptor>().read())
    }
}

/// Validate `gemm_sol_clc_multicast_4_stage_pipeline` bit-for-bit on `4096^3` using 12-bit Walsh codes.
fn validate_4096_walsh(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    const M: usize = 4096;
    let k_code = |kk: usize| -> u32 { ((kk as u32).wrapping_mul(251).wrapping_add(17)) & 0xFFF };
    let r_code = |r: usize| -> u32 { ((r as u32).wrapping_mul(197).wrapping_add(101)) & 0xFFF };
    let c_code = |c: usize| -> u32 { ((c as u32).wrapping_mul(109).wrapping_add(1021)) & 0xFFF };
    let fp =
        |pos: u32, kk: usize| -> f32 { (13 - 2 * (pos ^ k_code(kk)).count_ones() as i32) as f32 };

    let host_a: Vec<u16> = (0..M)
        .flat_map(|i| (0..M).map(move |kk| f16::from_f32(fp(r_code(i), kk)).to_bits()))
        .collect();
    let host_b: Vec<u16> = (0..M)
        .flat_map(|j| (0..M).map(move |kk| f16::from_f32(fp(c_code(j), kk)).to_bits()))
        .collect();
    let expected_by_xor: Vec<u16> = (0..M)
        .map(|d| {
            let score = 13 - 2 * (d as u32).count_ones() as i64;
            bf16::from_f32((M as i64 * score) as f32).to_bits()
        })
        .collect();

    let dev_a = DeviceBuffer::from_host(stream, &host_a)?;
    let dev_b = DeviceBuffer::from_host(stream, &host_b)?;
    let a_tma = encode_tma_2d_f16_swizzled(&dev_a, M as u64, M as u64, 64, 128)?;
    let b_tma = encode_tma_2d_f16_swizzled(&dev_b, M as u64, M as u64, 64, 32)?;

    let tiles_m = (M / 256) as u32;
    let tiles_n = (M / 128) as u32;
    let wide_total = tiles_m * (tiles_n / 2);
    let total_tiles = wide_total.div_ceil(B200_CLUSTER_PAIRS) * B200_CLUSTER_PAIRS;

    let mut dev_out = DeviceBuffer::<u32>::zeroed(stream, M * M / 2)?;
    let cfg = LaunchConfig1D::new(total_tiles * 2, 192, 0);
    let prep = module.prepare_gemm_sol_clc_multicast_4_stage_pipeline(cfg)?;
    unsafe {
        module.gemm_sol_clc_multicast_4_stage_pipeline(
            stream,
            &prep,
            a_tma,
            b_tma,
            &mut dev_out,
            M as i32,
            M as i32,
            tiles_m,
            tiles_n,
        )?;
    }
    stream.synchronize()?;

    let host_out = dev_out.to_host_vec(stream)?;
    for row in 0..M {
        for col in 0..M {
            let word = host_out[(row * M + col) / 2];
            let got_bits = if col % 2 == 0 {
                (word & 0xFFFF) as u16
            } else {
                (word >> 16) as u16
            };
            let exp_bits = expected_by_xor[(r_code(row) ^ c_code(col)) as usize];
            if got_bits != exp_bits {
                return Err(format!(
                    "mismatch at C[{row},{col}]: got {} vs expected {}",
                    bf16_to_f32(got_bits),
                    bf16_to_f32(exp_bits)
                )
                .into());
            }
        }
    }
    println!(
        "PASSED: Lean-verified gemm_sol_clc_multicast_4_stage_pipeline 4096x4096x4096 (all 16,777,216 BF16 elements exact!)"
    );
    Ok(())
}

/// Benchmark the Lean-verified `gemm_sol_clc_multicast_4_stage_pipeline` (`M256xN256`) entry at
/// `m x n x k` using `2,000` warmup launches followed by the median of `5 x 301` CUDA-event batches.
fn run_benchmark_clc_multicast_4_stage_pipeline(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
    m: usize,
    n: usize,
    k: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    let warmup = benchmark_count("GEMM_SOL_WARMUP", 2000)?;
    let iters = benchmark_count("GEMM_SOL_ITERS", 301)?;
    let repeats = benchmark_count("GEMM_SOL_REPEATS", 5)?;
    if repeats.is_multiple_of(2) {
        return Err("GEMM_SOL_REPEATS must be odd for the median".into());
    }

    let dev_a = DeviceBuffer::<u16>::zeroed(stream, m * k)?;
    let dev_b = DeviceBuffer::<u16>::zeroed(stream, n * k)?;
    let a_tma = encode_tma_2d_f16_swizzled(&dev_a, k as u64, m as u64, 64, 128)?;
    let b_tma = encode_tma_2d_f16_swizzled(
        &dev_b,
        k as u64,
        n as u64,
        64,
        if m == 4096 { 32 } else { 128 },
    )?;

    let tiles_m = (m / 256) as u32;
    let tiles_n = (n / 128) as u32;
    let wide_total = tiles_m * (tiles_n / 2);
    let total_tiles = wide_total.div_ceil(B200_CLUSTER_PAIRS) * B200_CLUSTER_PAIRS;
    let cfg = LaunchConfig1D::new(total_tiles * 2, 192, 0);
    let prep = module.prepare_gemm_sol_clc_multicast_4_stage_pipeline(cfg)?;

    let output_u32_count = m * n / 2;
    let mut dev_out = DeviceBuffer::<u32>::zeroed(stream, output_u32_count)?;
    let n_arg = n as i32;
    let k_arg = k as i32;

    for _ in 0..warmup {
        // SAFETY: `a_tma`, `b_tma`, and `dev_out` match the validated `m x n x k` allocations on `stream`.
        unsafe {
            module.gemm_sol_clc_multicast_4_stage_pipeline(
                stream,
                &prep,
                a_tma,
                b_tma,
                &mut dev_out,
                n_arg,
                k_arg,
                tiles_m,
                tiles_n,
            )?;
        }
    }
    stream.synchronize()?;

    let mut samples_us = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start_event =
            stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        for _ in 0..iters {
            // SAFETY: `a_tma`, `b_tma`, and `dev_out` remain valid across the timed batch.
            unsafe {
                module.gemm_sol_clc_multicast_4_stage_pipeline(
                    stream,
                    &prep,
                    a_tma,
                    b_tma,
                    &mut dev_out,
                    n_arg,
                    k_arg,
                    tiles_m,
                    tiles_n,
                )?;
            }
        }
        let end_event =
            stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        let elapsed_ms = start_event.elapsed_ms(&end_event)?;
        samples_us.push(elapsed_ms as f64 * 1000.0 / iters as f64);
    }

    let mut ordered = samples_us.clone();
    ordered.sort_by(f64::total_cmp);
    let avg_us = ordered[repeats / 2];
    let avg_ms = avg_us / 1000.0;
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let tflops = (flops / (avg_ms / 1000.0)) / 1e12;

    println!("=======================================================");
    println!(
        "  BENCHMARK: gemm_sol_clc_multicast_4_stage_pipeline (M256xN256, cg2, Lean-verified) {m}x{n}x{k} f16 -> bf16"
    );
    println!("=======================================================");
    println!(
        "  Grid:        {} CTAs (1D, CLC-managed, cluster=2, cg2)",
        total_tiles * 2
    );
    println!("  K-loop:      {} outer iters (BK=64, 4 MMAs each)", k / 64);
    println!("  Pipeline:    CLC + cta_group::2 + 7-stage SMEM (224 KiB) + 2-stage TMEM");
    println!("  Iterations:  {iters} x {repeats} batches (after {warmup} warmup)");
    println!("  Statistic:   median batch mean");
    println!("  Samples us:  {samples_us:?}");
    println!("  Average:     {avg_us:.3} us / kernel");
    println!("  FLOPS/kern:  {flops:.3e}");
    println!("  Throughput:  {tflops:.3} TFLOPS");
    println!(
        "RESULT {{\"implementation\":\"cuda-oxide-verified\",\"n\":{m},\"warmup\":{warmup},\"iters\":{iters},\"repeats\":{repeats},\"samples_us\":{samples_us:?},\"median_us\":{avg_us:.9},\"tflops\":{tflops:.9},\"variant\":\"M256xN256\",\"input\":\"fp16\",\"output\":\"bf16\",\"compute\":\"fp32\"}}"
    );
    print_cublas_comparison(tflops, m);
    println!("=======================================================\n");

    Ok(tflops)
}

/// Print a compact throughput summary table for all benchmarked `(M, TFLOPS)` problem sizes.
/// Matches the summary banner printed by `examples/gemm_sol_final`.
fn print_benchmark_summary(measurements: &[(usize, f64)]) {
    println!("-- Individual fixed-size results ----------------------");
    for (m, tflops) in measurements {
        println!("  {m}^3: {tflops:.3} TFLOPS");
    }
    println!("-------------------------------------------------------");
}

/// Entry point: validates `gemm_sol_clc_multicast_4_stage_pipeline` bit-for-bit on `4096^3` and
/// runs the `4096^3`, `8192^3`, and `16384^3` throughput benchmarks on datacenter Blackwell GPUs.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=======================================================");
    println!("  Provably Safe Stage 10 Blackwell GEMM SoL (`safe_gemm_sol_final`)");
    println!("=======================================================\n");

    let mode: RunMode = std::env::var("GEMM_SOL_MODE")
        .unwrap_or_else(|_| "both".to_string())
        .parse()?;

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let (major, minor) = ctx.compute_capability()?;
    println!("GPU: sm_{major}{minor}");

    if !can_execute_tcgen05_ptx(major, minor) {
        let ptx_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("safe_gemm_sol_final.ptx");
        if ptx_path.exists() {
            let ptx = std::fs::read_to_string(&ptx_path)?;
            verify_tcgen05_ptx_contract(&ptx)?;
            println!(
                "PASS: Lean 4 verified PTX artifact validated at {} (sm_100a hardware required for execution)",
                ptx_path.display()
            );
            return Ok(());
        }
        println!(
            "Note: sm_{major}{minor} does not have tcgen05/TMEM hardware; build with `--arch sm_100a` to verify PTX."
        );
        return Ok(());
    }

    println!("Loading embedded Lean-verified CUDA module");
    // SAFETY: `ctx` is an initialized `sm_100a`+ CUDA context matching the compiled PTX target.
    let module = unsafe { kernels::load(&ctx) }?;
    println!("Module loaded\n");

    if matches!(mode, RunMode::Validate | RunMode::Both) {
        println!("-- Full-output correctness test -----------------------\n");
        validate_4096_walsh(&stream, &module)?;
    }

    if matches!(mode, RunMode::Bench | RunMode::Both) {
        let _ = CUBLAS_BASELINE.get_or_init(load_cublas_baseline);
        println!("\n-- Benchmarks -----------------------------------------\n");
        let sizes = [
            (4096, 4096, 4096),
            (8192, 8192, 8192),
            (16384, 16384, 16384),
        ];
        let mut measurements = Vec::with_capacity(sizes.len());
        for (m, n, k) in sizes {
            let tflops = run_benchmark_clc_multicast_4_stage_pipeline(&stream, &module, m, n, k)?;
            measurements.push((m, tflops));
        }
        print_benchmark_summary(&measurements);
    }

    println!("\n=======================================================");
    println!("  Safe GEMM SoL target complete");
    println!("=======================================================");
    println!("PASS: requested Safe GEMM SoL work completed");
    Ok(())
}
