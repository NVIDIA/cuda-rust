/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Canonical `cuda-oxide` Blackwell (`sm_100a`, B200) speed-of-light FP16/BF16 GEMM (`C = A × B^T`).
//!
//! # 1. 2-CTA Cluster + 7-Stage TMA Ring + 2-Stage `TMEM` Ping-Pong
//!
//! ```text
//! 2-CTA CLUSTER (`cta_group::2`, 192 threads / CTA = 6 warps)
//! ────────────────────────────────────────────────────────────────────────────────────────────
//! Warp 4 (TMA + CLC):  [ 7-Stage Multicast TMA Ring (224 KiB SMEM) + Overlapped CLC Steal   ]
//! Warp 5 (Leader MMA): [ Tile 0 -> TMEM_0 ] [ Tile 1 -> TMEM_1 ] [ Tile 2 -> TMEM_0 ] ...
//! Warps 0..3 (Epilog):                      [ ld(TMEM_0) -> Free TMEM_0 -> Butterfly -> U32x4]
//! ```
//!
//! # 2. Data Layout
//! - `A`: `[M, K]` `f16`, row-major (`K` contiguous, 128B swizzled TMA).
//! - `B`: `[N, K]` `f16`, row-major (`K` contiguous, transposed storage, 128B swizzled TMA).
//! - `C`: `[M, N]` `bf16` output, row-major (packed as `u32` pairs and stored via 128-bit `U32x4`).

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, IntoResult, LaunchConfig};
use cuda_device::barrier::{
    Barrier, fence_proxy_async_shared_cta, mbarrier_arrive, mbarrier_arrive_cluster,
    mbarrier_arrive_expect_tx, mbarrier_init, mbarrier_try_wait_parity,
};
use cuda_device::clc::{
    clc_query_get_first_ctaid_x, clc_query_is_canceled, clc_try_cancel_multicast,
};
use cuda_device::cluster;
use cuda_device::convert::{bf16_to_f32, cvt_bf16x2_f32};
use cuda_device::shared::{SharedArray, cvta_generic_to_shared_offset};
use cuda_device::tcgen05::{
    Tcgen05AccumulatorType, Tcgen05ElementType, Tcgen05InstructionDescriptor, Tcgen05MmaShape,
    tcgen05_alloc_cg2, tcgen05_commit_multicast_cg2, tcgen05_dealloc_cg2,
    tcgen05_ld_16x256b_x8_pure, tcgen05_load_wait, tcgen05_mma_f16_cg2,
    tcgen05_relinquish_alloc_permit_cg2,
};
use cuda_device::tma::{TmaDescriptor, cp_async_bulk_tensor_2d_g2s_multicast_cg2};
use cuda_device::{DisjointSlice, cluster_launch, kernel, thread, warp};
use cuda_host::{
    KernelFamily, KernelFamilyBuildError, KernelFamilyId, KernelProblem, KernelSelector,
    KernelVariant, NoKernelSelectionCache, SelectedVariant, SelectionMode, cuda_module,
};
use half::{bf16, f16};
use std::mem::{MaybeUninit, size_of};
use std::str::FromStr;

// =============================================================================
// LIVE cuBLASLt BASELINE
// =============================================================================

/// Live `cuBLASLt` FP16-input / FP16-output / FP32-compute reference harness used
/// to report `"% of SoL"` on the same GPU.
mod cublas_baseline {
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::OnceLock;

    /// Cached per-shape (`M -> TFLOPS`) measurements parsed from `bench/cublaslt_bench`.
    /// Populated at most once per process via [`warmup`] or [`reference_tflops`].
    static BASELINE: OnceLock<Option<HashMap<usize, f64>>> = OnceLock::new();

    /// Return the live `cuBLASLt` FP16 TFLOPS for an `M × M × M` GEMM on the host GPU,
    /// or `None` if `bench/cublaslt_bench` was skipped or has not been built.
    pub fn reference_tflops(m: usize) -> Option<f64> {
        BASELINE.get_or_init(load).as_ref()?.get(&m).copied()
    }

    /// Pre-warm the `cuBLASLt` reference before the timed Rust benchmark loop starts
    /// so one-time measurement output does not interleave with benchmark results.
    pub fn warmup() {
        let _ = BASELINE.get_or_init(load);
    }

    /// Resolve the path to the locally compiled `bench/cublaslt_bench` helper binary.
    /// Built on each machine via `cd examples/gemm_sol_final/bench && bash build.sh`.
    fn bench_binary() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("bench")
            .join("cublaslt_bench")
    }

    /// Execute `bench/cublaslt_bench` with matching warmup/iteration/repeat overrides
    /// and parse its `--- FP16 ---` throughput rows.
    fn load() -> Option<HashMap<usize, f64>> {
        if std::env::var("GEMM_CUBLAS_SKIP").as_deref() == Ok("1") {
            return None;
        }
        let bin = bench_binary();
        if !bin.exists() {
            eprintln!(
                "ℹ️  No live cublasLt baseline at {} -- % of SoL column will be omitted.",
                bin.display()
            );
            eprintln!(
                "    Build it once with: cd {} && bash build.sh",
                bin.parent().unwrap_or(Path::new(".")).display(),
            );
            return None;
        }

        eprintln!("ℹ️  Measuring live cuBLASLt tuned FP16 reference on host GPU...");
        let mut command = std::process::Command::new(&bin);
        for (name, flag) in [
            ("GEMM_SOL_WARMUP", "--warmup"),
            ("GEMM_SOL_ITERS", "--iters"),
            ("GEMM_SOL_REPEATS", "--repeats"),
        ] {
            if let Ok(value) = std::env::var(name) {
                command.args([flag, value.as_str()]);
            }
        }
        let out = match command.output() {
            Ok(o) => o,
            Err(e) => {
                eprintln!("⚠️  Failed to run {}: {e}", bin.display());
                return None;
            }
        };
        if !out.status.success() {
            eprintln!(
                "⚠️  {} exited with status {}: {}",
                bin.display(),
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            );
            return None;
        }
        let stdout = String::from_utf8_lossy(&out.stdout);
        let map = parse_fp16_reference(&stdout);
        if map.is_empty() {
            eprintln!("⚠️  Could not parse FP16 rows from cublaslt_bench output:\n{stdout}");
            None
        } else {
            let mut sizes: Vec<(usize, f64)> = map.iter().map(|(m, t)| (*m, *t)).collect();
            sizes.sort_by_key(|(m, _)| *m);
            let pretty = sizes
                .iter()
                .map(|(m, t)| format!("{m}={t:.1}"))
                .collect::<Vec<_>>()
                .join(", ");
            eprintln!("✓ live cuBLASLt FP16 reference (TFLOPS): {pretty}");
            Some(map)
        }
    }

    /// Extract `(M, TFLOPS)` pairs from the `--- FP16 ---` section of `cublaslt_bench` stdout.
    /// Ignores lines outside the `--- FP16 ---` block.
    fn parse_fp16_reference(s: &str) -> HashMap<usize, f64> {
        let mut map = HashMap::new();
        let mut in_fp16 = false;
        for line in s.lines() {
            let l = line.trim_start();
            if l.starts_with("--- FP16") {
                in_fp16 = true;
                continue;
            }
            if l.starts_with("--- BF16") {
                in_fp16 = false;
                continue;
            }
            if !in_fp16 || !l.starts_with("FP16 ") {
                continue;
            }
            let toks: Vec<&str> = l.split_whitespace().collect();
            let size = toks.get(3).copied().unwrap_or("");
            let m: Option<usize> = size.split('x').next().and_then(|s| s.trim().parse().ok());
            let tf: Option<f64> = toks.iter().rev().nth(1).and_then(|s| s.parse().ok());
            if let (Some(m), Some(tf)) = (m, tf) {
                map.insert(m, tf);
            }
        }
        map
    }
}

/// Print the `"vs cuBLAS"` comparison line for size `m` using the live `cuBLASLt`
/// baseline if `bench/cublaslt_bench` is present.
fn print_cublas_comparison(tflops: f64, m: usize) {
    match cublas_baseline::reference_tflops(m) {
        Some(sol) => {
            let pct = (tflops / sol) * 100.0;
            println!(
                "  vs cuBLAS:   {pct:.2}% of live cublasLt tuned FP16 reference ({sol:.0} TFLOPS)"
            );
        }
        None => {
            println!("  vs cuBLAS:   (no live cublasLt baseline; see bench/build.sh)");
        }
    }
}

/// Print a compact summary of the measured fixed-size `4096³`, `8192³`, and `16384³` results.
fn print_benchmark_summary(measurements: &[(usize, f64)]) {
    println!("── Individual fixed-size results ────────────────────");
    for (m, tflops) in measurements {
        println!("  {m}³: {tflops:.3} TFLOPS");
    }
    println!("─────────────────────────────────────────────────────");
}

/// Parse a positive integer benchmark count override (`GEMM_SOL_WARMUP`, `GEMM_SOL_ITERS`,
/// `GEMM_SOL_REPEATS`) from the environment, or return `default` when unset.
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

/// Execution mode selected via the `GEMM_SOL_MODE` environment variable.
/// Controls whether the binary runs validation, benchmarks, both, or dry-run dispatch planning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunMode {
    /// Run only the full 16,777,216-element exact-BF16 Walsh correctness checks at `4096³`.
    Validate,
    /// Run only the `4096³`, `8192³`, and `16384³` throughput benchmarks.
    Bench,
    /// Run full correctness validation followed by the throughput benchmarks (default).
    Both,
    /// Print the `KernelFamily` variant selection for each size without opening a CUDA context.
    Plan,
}

impl FromStr for RunMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "validate" => Ok(Self::Validate),
            "bench" => Ok(Self::Bench),
            "both" => Ok(Self::Both),
            "plan" => Ok(Self::Plan),
            other => Err(format!(
                "invalid GEMM_SOL_MODE={other:?}; expected validate, bench, both, or plan"
            )),
        }
    }
}

// =============================================================================
// KERNEL
// =============================================================================

/// Build a 64-bit `tcgen05` shared-memory matrix descriptor from base address, LBO, SBO, and swizzle mode.
///
/// Bit layout of the 64-bit descriptor:
/// - `[0:13]`:  `smem_addr >> 4`
/// - `[16:29]`: `leading_dim_bytes >> 4` (LBO: byte stride to next core matrix right)
/// - `[32:45]`: `stride_bytes >> 4` (SBO: byte stride to next core matrix down)
/// - `[46]`:    fixed `0b1`
/// - `[61:63]`: `swizzle` mode (`2` for 128-byte swizzle)
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

/// Typed host launcher function pointer shared by both `KernelFamily` variants.
/// Accepts the loaded CUDA module, stream, launch config, TMA descriptor pointers, output buffer, and dimensions.
type GemmKernelLauncher = unsafe fn(
    &kernels::LoadedModule,
    &CudaStream,
    LaunchConfig,
    *const TmaDescriptor,
    *const TmaDescriptor,
    &mut DeviceBuffer<u32>,
    i32,
    i32,
    u32,
    u32,
) -> Result<(), cuda_core::DriverError>;

/// Stable identifier for each compiled variant in the `gemm_sol_final/output_tile@1` family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GemmVariantId {
    /// `256 × 256` macro tile per 2-CTA cluster, selected for `4096³` and `8192³`.
    M256xN256,
    /// `512 × 256` macro tile per 2-CTA cluster, selected for `16384³` (`tiles_m >= 64`).
    M512xN256,
}

/// Static shape contract and preference threshold attached to each [`GemmVariant`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GemmVariantMetadata {
    /// PTX entry symbol name in the compiled device module.
    kernel_name: &'static str,
    /// Human-readable macro-tile label (`"M256xN256"` or `"M512xN256"`).
    output_tile: &'static str,
    /// Required `M` divisibility and minimum `M` dimension for this variant.
    m_tile: usize,
    /// Required `N` divisibility and minimum `N` dimension for this variant.
    n_tile: usize,
    /// Required `K` divisibility and minimum `K` dimension (`256` = 4 outer K-iterations of 64).
    k_tile_multiple: usize,
    /// Minimum `tiles_m = M / 256` at which the automatic selector prefers this variant.
    preferred_from_tiles_m: usize,
}

/// Two-variant `KernelFamily` table for `gemm_sol_final/output_tile@1`.
type GemmKernelFamily = KernelFamily<GemmVariantId, GemmKernelLauncher, GemmVariantMetadata, 2>;

/// Single entry in [`GemmKernelFamily`] pairing a [`GemmVariantId`], launcher, and [`GemmVariantMetadata`].
type GemmVariant = KernelVariant<GemmVariantId, GemmKernelLauncher, GemmVariantMetadata>;

/// Construct the two-entry `gemm_sol_final/output_tile@1` kernel family.
/// Registers `M256xN256` (`preferred_from_tiles_m = 0`) and `M512xN256` (`preferred_from_tiles_m = 64`).
fn gemm_kernel_family() -> Result<GemmKernelFamily, KernelFamilyBuildError> {
    KernelFamily::try_new(
        "gemm_sol_final/output_tile",
        1,
        [
            KernelVariant::new(
                GemmVariantId::M256xN256,
                kernels::LoadedModule::gemm_sol_clc_multicast_4_stage_pipeline
                    as GemmKernelLauncher,
                GemmVariantMetadata {
                    kernel_name: "gemm_sol_clc_multicast_4_stage_pipeline",
                    output_tile: "M256xN256",
                    m_tile: 256,
                    n_tile: 256,
                    k_tile_multiple: 256,
                    preferred_from_tiles_m: 0,
                },
            ),
            KernelVariant::new(
                GemmVariantId::M512xN256,
                kernels::LoadedModule::gemm_sol_clc_multicast_4_stage_pipeline_large
                    as GemmKernelLauncher,
                GemmVariantMetadata {
                    kernel_name: "gemm_sol_clc_multicast_4_stage_pipeline_large",
                    output_tile: "M512xN256",
                    m_tile: 512,
                    n_tile: 256,
                    k_tile_multiple: 256,
                    preferred_from_tiles_m: 64,
                },
            ),
        ],
    )
}

/// Problem descriptor passed to [`KernelFamily::select`] for eligibility validation and tile selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GemmProblem {
    /// Number of rows `M` in `A` and `C`.
    m: usize,
    /// Number of columns `N` in `B^T` and `C`.
    n: usize,
    /// Inner reduction dimension `K` shared by `A` and `B`.
    k: usize,
    /// Number of `256`-row M-tiles (`m / 256`).
    tiles_m: usize,
}

impl GemmProblem {
    /// Create a [`GemmProblem`] for an `m × n × k` matrix multiplication.
    fn new(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            tiles_m: m / 256,
        }
    }
}

/// Structured error returned when a [`GemmProblem`] violates a variant's shape or launch-grid contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GemmVariantIneligible {
    /// Variant that rejected the problem.
    variant: GemmVariantId,
    /// Static explanation of which contract check failed.
    reason: &'static str,
    /// Requested `M` dimension.
    m: usize,
    /// Requested `N` dimension.
    n: usize,
    /// Requested `K` dimension.
    k: usize,
    /// Variant's required `M` tile multiple.
    m_tile: usize,
    /// Variant's required `N` tile multiple.
    n_tile: usize,
    /// Variant's required `K` tile multiple.
    k_tile_multiple: usize,
}

impl std::fmt::Display for GemmVariantIneligible {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?} rejected {}x{}x{}: {}; tile contract is M >= {} and divisible by {}, N >= {} and divisible by {}, K >= {} and divisible by {}",
            self.variant,
            self.m,
            self.n,
            self.k,
            self.reason,
            self.m_tile,
            self.m_tile,
            self.n_tile,
            self.n_tile,
            self.k_tile_multiple,
            self.k_tile_multiple,
        )
    }
}

impl std::error::Error for GemmVariantIneligible {}

impl KernelProblem<GemmVariant> for GemmProblem {
    type Rejection = GemmVariantIneligible;

    fn validate(&self, variant: &GemmVariant) -> Result<(), Self::Rejection> {
        let metadata = variant.metadata();
        let reject = |reason| GemmVariantIneligible {
            variant: *variant.id(),
            reason,
            m: self.m,
            n: self.n,
            k: self.k,
            m_tile: metadata.m_tile,
            n_tile: metadata.n_tile,
            k_tile_multiple: metadata.k_tile_multiple,
        };

        if !(self.m >= metadata.m_tile
            && self.m.is_multiple_of(metadata.m_tile)
            && self.n >= metadata.n_tile
            && self.n.is_multiple_of(metadata.n_tile)
            && self.k >= metadata.k_tile_multiple
            && self.k.is_multiple_of(metadata.k_tile_multiple))
        {
            return Err(reject("dimensions violate the compiled tile shape"));
        }
        let max_m = i32::MAX as usize + 1;
        if self.m > max_m || self.n > i32::MAX as usize || self.k > i32::MAX as usize {
            return Err(reject(
                "M, N, or K exceeds the kernel's signed coordinate/launch ABI",
            ));
        }

        let a_fits = self
            .m
            .checked_mul(self.k)
            .is_some_and(|elements| elements <= usize::MAX / size_of::<u16>());
        let b_fits = self
            .n
            .checked_mul(self.k)
            .is_some_and(|elements| elements <= usize::MAX / size_of::<u16>());
        let output_fits = self
            .m
            .checked_mul(self.n)
            .is_some_and(|elements| elements <= usize::MAX / 2);
        if !(a_fits && b_fits && output_fits) {
            return Err(reject("matrix allocation size overflows usize"));
        }

        let grid_x = (self.m / 256)
            .checked_mul(self.n / 128)
            .and_then(|tiles| tiles.checked_mul(2));
        if grid_x.is_none_or(|grid_x| grid_x > i32::MAX as usize) {
            return Err(reject("grid.x exceeds the CUDA launch limit"));
        }

        Ok(())
    }
}

/// Threshold-based selector that picks the eligible variant with the highest
/// `preferred_from_tiles_m <= problem.tiles_m`.
struct GemmSelector;

impl KernelSelector<GemmProblem, GemmVariant, GemmVariantId> for GemmSelector {
    type Error = std::convert::Infallible;

    fn select(
        &mut self,
        _family: KernelFamilyId,
        problem: &GemmProblem,
        eligible: &[&GemmVariant],
    ) -> Result<GemmVariantId, Self::Error> {
        let preferred = eligible
            .iter()
            .copied()
            .filter(|variant| variant.metadata().preferred_from_tiles_m <= problem.tiles_m)
            .max_by_key(|variant| variant.metadata().preferred_from_tiles_m)
            .unwrap_or(eligible[0]);
        Ok(*preferred.id())
    }
}

/// Parse a `GEMM_SOL_VARIANT` string (`"auto"`, `"m256xn256"`, or `"m512xn256"`) into a [`SelectionMode`].
fn gemm_selection_mode(value: Option<&str>) -> Result<SelectionMode<GemmVariantId>, String> {
    match value.unwrap_or("auto") {
        "auto" => Ok(SelectionMode::Auto),
        "m256xn256" => Ok(SelectionMode::Force(GemmVariantId::M256xN256)),
        "m512xn256" => Ok(SelectionMode::Force(GemmVariantId::M512xN256)),
        other => Err(format!(
            "invalid GEMM_SOL_VARIANT={other:?}; expected auto, m256xn256, or m512xn256"
        )),
    }
}

/// Convert `std::env::var("GEMM_SOL_VARIANT")` into a validated [`SelectionMode`].
fn gemm_selection_mode_from_var(
    value: Result<String, std::env::VarError>,
) -> Result<SelectionMode<GemmVariantId>, String> {
    match value {
        Ok(value) => gemm_selection_mode(Some(&value)),
        Err(std::env::VarError::NotPresent) => gemm_selection_mode(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err("GEMM_SOL_VARIANT must be valid UTF-8".to_string())
        }
    }
}

/// Concrete `KernelSelectionError` instantiation for [`GemmKernelFamily`].
type GemmSelectionError = cuda_host::KernelSelectionError<
    GemmVariantId,
    GemmVariantIneligible,
    std::convert::Infallible,
    std::convert::Infallible,
>;

/// Borrowed selection result returned by [`select_gemm_variant`].
type GemmSelectedVariant<'family> =
    SelectedVariant<'family, GemmVariantId, GemmKernelLauncher, GemmVariantMetadata>;

/// Run eligibility validation and variant selection on `family` for `problem` under `mode`.
fn select_gemm_variant<'family>(
    family: &'family GemmKernelFamily,
    problem: &GemmProblem,
    mode: SelectionMode<GemmVariantId>,
) -> Result<GemmSelectedVariant<'family>, GemmSelectionError> {
    let mut selector = GemmSelector;
    let mut cache = NoKernelSelectionCache;
    family.select(problem, mode, &mut selector, &mut cache)
}

/// Print the selected variant for `4096³`, `8192³`, and `16384³` without initializing CUDA.
fn print_gemm_dispatch_plan(
    family: &GemmKernelFamily,
    mode: SelectionMode<GemmVariantId>,
) -> Result<(), GemmSelectionError> {
    println!("Kernel family: {}", family.id());
    for size in [4096, 8192, 16384] {
        let problem = GemmProblem::new(size, size, size);
        let selected = select_gemm_variant(family, &problem, mode)?;
        println!(
            "{size}x{size}x{size} -> {} ({}) [{}]",
            selected.variant().metadata().output_tile,
            selected.variant().metadata().kernel_name,
            selected.source(),
        );
    }
    Ok(())
}

// =============================================================================
// HOST CODE
// =============================================================================

/// Return `true` when compute capability `(major, minor)` supports datacenter `tcgen05`
/// (`sm_100a`, `sm_101a`, `sm_103a`, `sm_110a`).
fn can_execute_tcgen05_ptx(major: i32, minor: i32) -> bool {
    matches!((major, minor), (10, 0) | (10, 1) | (10, 3) | (11, 0))
}

/// Verify that a loose PTX artifact targets a datacenter `tcgen05` architecture and exports
/// both `gemm_sol_clc_multicast_4_stage_pipeline` and `gemm_sol_clc_multicast_4_stage_pipeline_large`.
fn verify_tcgen05_ptx_contract(ptx: &str) -> Result<(), String> {
    const TARGETS: [&str; 8] = [
        "sm_100a", "sm_101a", "sm_103a", "sm_110a", "sm_100f", "sm_101f", "sm_103f", "sm_110f",
    ];
    const ENTRIES: [&str; 2] = [
        "gemm_sol_clc_multicast_4_stage_pipeline",
        "gemm_sol_clc_multicast_4_stage_pipeline_large",
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
    for entry in ENTRIES {
        if !document
            .callables_named(entry)
            .any(|callable| callable.kind() == ptx_parse::CallableKind::Entry)
        {
            return Err(format!("missing expected kernel entry `{entry}`"));
        }
    }
    Ok(())
}

/// Entry point: parses `GEMM_SOL_MODE` and `GEMM_SOL_VARIANT`, validates both kernels on `4096³`,
/// and benchmarks `4096³`, `8192³`, and `16384³` against live `cuBLASLt`.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("  GEMM SoL -- gemm_sol_final (size-specialized)");
    println!("═══════════════════════════════════════════════════════\n");

    let mode: RunMode = std::env::var("GEMM_SOL_MODE")
        .unwrap_or_else(|_| "both".to_string())
        .parse()?;
    let family = gemm_kernel_family()?;
    let selection_mode = if matches!(mode, RunMode::Bench | RunMode::Both | RunMode::Plan) {
        gemm_selection_mode_from_var(std::env::var("GEMM_SOL_VARIANT"))?
    } else {
        SelectionMode::Auto
    };

    if mode == RunMode::Plan {
        print_gemm_dispatch_plan(&family, selection_mode)?;
        return Ok(());
    }

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let (major, minor) = ctx.compute_capability()?;
    println!("GPU: sm_{major}{minor}");

    if !can_execute_tcgen05_ptx(major, minor) {
        println!("\nWARNING: tcgen05 requires a compatible datacenter Blackwell GPU");
        return verify_ptx_only();
    }

    println!("Loading embedded CUDA module");
    let module = kernels::load(&ctx)?;
    println!("Module loaded\n");

    if matches!(mode, RunMode::Validate | RunMode::Both) {
        println!("── Full-output correctness tests ────────────────────\n");
        run_correctness_test_clc_multicast_4_stage_pipeline(
            &stream,
            &module,
            kernels::LoadedModule::gemm_sol_clc_multicast_4_stage_pipeline,
            "gemm_sol_clc_multicast_4_stage_pipeline",
            "M256xN256",
            4096,
            4096,
            4096,
        )?;
        run_correctness_test_clc_multicast_4_stage_pipeline(
            &stream,
            &module,
            kernels::LoadedModule::gemm_sol_clc_multicast_4_stage_pipeline_large,
            "gemm_sol_clc_multicast_4_stage_pipeline_large",
            "M512xN256",
            4096,
            4096,
            4096,
        )?;
    }

    if matches!(mode, RunMode::Bench | RunMode::Both) {
        cublas_baseline::warmup();
        println!("\n── Benchmarks ───────────────────────────────────────\n");
        let sizes = [
            (4096, 4096, 4096),
            (8192, 8192, 8192),
            (16384, 16384, 16384),
        ];
        let mut measurements = Vec::with_capacity(sizes.len());
        for (m, n, k) in sizes {
            let tflops = run_benchmark_clc_multicast_4_stage_pipeline(
                &stream,
                &module,
                &family,
                selection_mode,
                m,
                n,
                k,
            )?;
            measurements.push((m, tflops));
        }
        print_benchmark_summary(&measurements);
    }

    println!("\n═══════════════════════════════════════════════════════");
    println!("  GEMM SoL target complete");
    println!("═══════════════════════════════════════════════════════");
    println!("PASS: requested GEMM SoL work completed");
    Ok(())
}

/// Number of distinct 12-bit Walsh codes (`4,096`) used by the `4096³` exact-BF16 validator.
/// Gives every row and column in `4096 × 4096` a unique signature with an `O(1)` analytic dot product.
const VALIDATION_CODE_COUNT: usize = 1 << 12;

/// Bitmask (`0xFFF`) isolating a 12-bit Walsh code index (`0..4096`).
const VALIDATION_CODE_MASK: u32 = VALIDATION_CODE_COUNT as u32 - 1;

/// Compute a 12-bit affine permutation `(x * multiplier + addend) mod 4096`.
/// Odd multipliers are bijections over `0..4096`.
#[inline]
fn validation_affine12(x: usize, multiplier: u32, addend: u32) -> u32 {
    ((x as u32).wrapping_mul(multiplier).wrapping_add(addend)) & VALIDATION_CODE_MASK
}

/// Map reduction index `kk` (`0..4096`) to its unique 12-bit Walsh basis index.
#[inline]
fn validation_k_code(kk: usize) -> u32 {
    validation_affine12(kk, 251, 17)
}

/// Map matrix `A` row index `row` (`0..4096`) to its unique 12-bit Walsh code.
#[inline]
fn validation_row_code(row: usize) -> u32 {
    validation_affine12(row, 197, 101)
}

/// Map matrix `B` column index `col` (`0..4096`) to its unique 12-bit Walsh code.
#[inline]
fn validation_col_code(col: usize) -> u32 {
    validation_affine12(col, 109, 1021)
}

/// Evaluate the signed integer Walsh fingerprint (`13 - 2 * popcount(code ^ k_code)`) in `f32`.
/// Because every value is a small odd integer in `[-11, 13]`, all `f16` inputs and `f32` products are exact.
#[inline]
fn validation_fingerprint(position_code: u32, kk: usize) -> f32 {
    let distance = (position_code ^ validation_k_code(kk)).count_ones() as i32;
    (13 - 2 * distance) as f32
}

/// Return the exact `f32` test value for `A[row, kk]`.
#[inline]
fn validation_a_value(row: usize, kk: usize) -> f32 {
    validation_fingerprint(validation_row_code(row), kk)
}

/// Return the exact `f32` test value for `B[col, kk]`.
#[inline]
fn validation_b_value(col: usize, kk: usize) -> f32 {
    validation_fingerprint(validation_col_code(col), kk)
}

/// Validate `launch_kernel` at `4096³` by checking all `16,777,216` BF16 output elements
/// bit-for-bit against the exact analytical Walsh dot product.
#[expect(
    clippy::too_many_arguments,
    reason = "validation runner takes the stream, module, launcher, labels, and M/N/K dimensions"
)]
fn run_correctness_test_clc_multicast_4_stage_pipeline(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
    launch_kernel: GemmKernelLauncher,
    kernel_name: &str,
    output_tile: &str,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(
        [m, n, k],
        [VALIDATION_CODE_COUNT; 3],
        "the bit-exact Walsh validator is defined for the fixed 4096³ contract"
    );

    println!("Entry: {kernel_name} ({output_tile})");
    println!("Matrix: {m}x{n}x{k} (f16 -> bf16)");
    println!("CLC + cta_group::2 + 7-stage SMEM pipeline (224 KiB) + 2-stage TMEM.");
    println!("Warps: 4=TMA, 5=MMA (leader only), 0-3=epilogue (both CTAs).");

    let host_a: Vec<u16> = (0..m)
        .flat_map(|i| (0..k).map(move |kk| f16::from_f32(validation_a_value(i, kk)).to_bits()))
        .collect();
    let host_b: Vec<u16> = (0..n)
        .flat_map(|j| (0..k).map(move |kk| f16::from_f32(validation_b_value(j, kk)).to_bits()))
        .collect();

    // Walsh orthogonality gives C[row,col] = K * (13 - 2 * HammingDistance)
    // for each row/column code pair. Cache 4096 XOR-indexed expected entries.
    let expected_by_xor: Vec<u16> = (0..VALIDATION_CODE_COUNT)
        .map(|difference| {
            let score = 13 - 2 * (difference as u32).count_ones() as i64;
            let fp32 = (k as i64 * score) as f32;
            bf16::from_f32(fp32).to_bits()
        })
        .collect();

    println!(
        "Validation data: 12-bit Walsh fingerprints varying along K and covering \
         each K code once, with unique signatures for all {VALIDATION_CODE_COUNT} rows and columns"
    );
    println!("Expected: analytic exact FP32 dot product rounded once to BF16\n");

    let dev_a = DeviceBuffer::from_host(stream, &host_a)?;
    let dev_b = DeviceBuffer::from_host(stream, &host_b)?;

    let a_tma = encode_tma_2d_f16_swizzled(&dev_a, k as u64, m as u64, 64, 128)?;
    let b_tma = encode_tma_2d_f16_swizzled(
        &dev_b,
        k as u64,
        n as u64,
        64,
        if m == 4096 { 32 } else { 128 },
    )?;

    let dev_a_tma = DeviceBuffer::from_host(stream, &a_tma.opaque)?;
    let dev_b_tma = DeviceBuffer::from_host(stream, &b_tma.opaque)?;

    let tiles_m = (m / 256) as u32;
    let tiles_n = (n / 128) as u32;
    let wide_total = tiles_m * (tiles_n / 2);
    let total_tiles = wide_total.div_ceil(B200_CLUSTER_PAIRS) * B200_CLUSTER_PAIRS;

    let cfg = LaunchConfig {
        grid_dim: (total_tiles * 2, 1, 1),
        block_dim: (192, 1, 1),
        shared_mem_bytes: 0,
    };

    println!(
        "Grid: {} CTAs (1D, CLC-managed), cluster: 2x1x1 (cg2), block: 192 threads (6 warps)",
        total_tiles * 2
    );
    let swizzle_g = if tiles_m <= 16 { 2 } else { 8 };
    println!("Host work IDs: {total_tiles} ({tiles_m}x{tiles_n}), logical L2 band G={swizzle_g}");
    println!("K-loop: {} outer iters (BK=64, 4 MMAs each)", k / 64);

    let output_u32_count = m * n / 2;
    let mut dev_output = DeviceBuffer::<u32>::zeroed(stream, output_u32_count)?;

    let a_tma_ptr = dev_a_tma.cu_deviceptr() as *const TmaDescriptor;
    let b_tma_ptr = dev_b_tma.cu_deviceptr() as *const TmaDescriptor;
    let n_arg = n as i32;
    let k_arg = k as i32;

    println!("\nLaunching {kernel_name} (cg2)...");

    // SAFETY: `a_tma_ptr`, `b_tma_ptr`, and `dev_output` live in device memory on `stream`
    // and match the `m × n × k` dimensions validated above.
    unsafe {
        launch_kernel(
            module,
            stream,
            cfg,
            a_tma_ptr,
            b_tma_ptr,
            &mut dev_output,
            n_arg,
            k_arg,
            tiles_m,
            tiles_n,
        )
    }?;

    stream.synchronize()?;

    let host_output: Vec<u32> = dev_output.to_host_vec(stream)?;

    let expected_bits = |row: usize, col: usize| -> u16 {
        expected_by_xor[(validation_row_code(row) ^ validation_col_code(col)) as usize]
    };
    let read_c_bits = |row: usize, col: usize| -> u16 {
        let packed = host_output[row * (n / 2) + col / 2];
        if col.is_multiple_of(2) {
            packed as u16
        } else {
            (packed >> 16) as u16
        }
    };

    let mut mismatches: u64 = 0;
    let mut first_bad: Option<(usize, usize, f32, f32)> = None;
    let mut max_rel_err: f32 = 0.0;
    for row in 0..m {
        for col in 0..n {
            let val_bits = read_c_bits(row, col);
            let exp_bits = expected_bits(row, col);
            let val = bf16_to_f32(val_bits);
            let exp = bf16_to_f32(exp_bits);
            let diff = (val - exp).abs();
            let rel = diff / (exp.abs() + 1.0);
            if rel > max_rel_err {
                max_rel_err = rel;
            }
            if val_bits != exp_bits {
                mismatches += 1;
                if first_bad.is_none() {
                    first_bad = Some((row, col, val, exp));
                }
            }
        }
    }
    let total = (m * n) as u64;
    let all_ok = mismatches == 0;
    println!(
        "\nFull check: {} / {} exact BF16 matches (max rel err {:.4})",
        total - mismatches,
        total,
        max_rel_err
    );
    if let Some((r, c, val, exp)) = first_bad {
        println!("  first mismatch: C[{r},{c}] = {val} (expected {exp})");
    }

    println!("\n═══════════════════════════════════════════════════════");
    if all_ok {
        println!("PASSED: {kernel_name} {m}x{n}x{k} (full {total}-element check)");
    } else {
        println!("FAILED: {kernel_name} {m}x{n}x{k}: {mismatches} / {total} elements wrong");
        return Err(
            format!("Correctness check failed: {mismatches} / {total} elements wrong").into(),
        );
    }
    println!("═══════════════════════════════════════════════════════");

    Ok(())
}

/// Benchmark the selected `KernelFamily` entry at `m × n × k` using `2,000` warmup launches
/// followed by the median of `5 × 301` CUDA-event timed batches.
fn run_benchmark_clc_multicast_4_stage_pipeline(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
    family: &GemmKernelFamily,
    selection_mode: SelectionMode<GemmVariantId>,
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

    let problem = GemmProblem::new(m, n, k);
    let selected = select_gemm_variant(family, &problem, selection_mode)?;
    let launch_kernel = *selected.variant().entry();
    let kernel_name = selected.variant().metadata().kernel_name;
    let output_tile = selected.variant().metadata().output_tile;
    let selection_source = selected.source();

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

    let dev_a_tma = DeviceBuffer::from_host(stream, &a_tma.opaque)?;
    let dev_b_tma = DeviceBuffer::from_host(stream, &b_tma.opaque)?;
    let a_tma_ptr = dev_a_tma.cu_deviceptr() as *const TmaDescriptor;
    let b_tma_ptr = dev_b_tma.cu_deviceptr() as *const TmaDescriptor;
    let n_arg = n as i32;
    let k_arg = k as i32;

    let tiles_m = (m / 256) as u32;
    let tiles_n = (n / 128) as u32;
    let wide_total = tiles_m * (tiles_n / 2);
    let total_tiles = wide_total.div_ceil(B200_CLUSTER_PAIRS) * B200_CLUSTER_PAIRS;

    let cfg = LaunchConfig {
        grid_dim: (total_tiles * 2, 1, 1),
        block_dim: (192, 1, 1),
        shared_mem_bytes: 0,
    };

    let output_u32_count = m * n / 2;
    let mut dev_output = DeviceBuffer::<u32>::zeroed(stream, output_u32_count)?;

    for _ in 0..warmup {
        // SAFETY: `a_tma_ptr`, `b_tma_ptr`, and `dev_output` are valid device allocations on `stream`.
        unsafe {
            launch_kernel(
                module,
                stream,
                cfg,
                a_tma_ptr,
                b_tma_ptr,
                &mut dev_output,
                n_arg,
                k_arg,
                tiles_m,
                tiles_n,
            )
        }?;
    }
    stream.synchronize()?;

    let mut samples_us = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start_event =
            stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        for _ in 0..iters {
            // SAFETY: `a_tma_ptr`, `b_tma_ptr`, and `dev_output` remain valid across the timed loop.
            unsafe {
                launch_kernel(
                    module,
                    stream,
                    cfg,
                    a_tma_ptr,
                    b_tma_ptr,
                    &mut dev_output,
                    n_arg,
                    k_arg,
                    tiles_m,
                    tiles_n,
                )
            }?;
        }
        let end_event =
            stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        let elapsed_ms = start_event.elapsed_ms(&end_event)?;
        samples_us.push(elapsed_ms as f64 * 1000.0 / iters as f64);
    }
    let mut ordered = samples_us.clone();
    ordered.sort_by(f64::total_cmp);
    let avg_ms = ordered[repeats / 2] / 1000.0;
    let avg_us = avg_ms * 1000.0;
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let tflops = (flops / (avg_ms / 1000.0)) / 1e12;

    println!("═══════════════════════════════════════════════════════");
    println!(
        "  BENCHMARK: {kernel_name} ({output_tile}, cg2) {m}x{n}x{k} f16 -> bf16 [{selection_source}]"
    );
    println!("═══════════════════════════════════════════════════════");
    println!(
        "  Grid:        {} CTAs (1D, CLC-managed, cluster=2, cg2)",
        total_tiles * 2
    );
    println!("  K-loop:      {} outer iters (BK=64, 4 MMAs each)", k / 64);
    println!("  Pipeline:    CLC + cta_group::2 + 7-stage SMEM (224 KiB) + 2-stage TMEM");
    println!("  Iterations:  {iters} × {repeats} batches (after {warmup} warmup)");
    println!("  Statistic:   median batch mean");
    println!("  Samples us:  {samples_us:?}");
    println!("  Average:     {avg_us:.3} us / kernel");
    println!("  FLOPS/kern:  {flops:.3e}");
    println!("  Throughput:  {tflops:.3} TFLOPS");
    println!(
        "RESULT {{\"implementation\":\"cuda-oxide\",\"n\":{m},\"warmup\":{warmup},\"iters\":{iters},\"repeats\":{repeats},\"samples_us\":{samples_us:?},\"median_us\":{avg_us:.9},\"tflops\":{tflops:.9},\"variant\":\"{output_tile}\",\"input\":\"fp16\",\"output\":\"bf16\",\"compute\":\"fp32\"}}"
    );
    print_cublas_comparison(tflops, m);
    println!("═══════════════════════════════════════════════════════\n");

    Ok(tflops)
}

/// Fallback for non-`sm_100a` hosts: verify that the emitted `gemm_sol_final.ptx` artifact
/// targets a datacenter `tcgen05` architecture and contains both kernel entries.
fn verify_ptx_only() -> Result<(), Box<dyn std::error::Error>> {
    let ptx_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("gemm_sol_final.ptx");
    let ptx = std::fs::read_to_string(&ptx_path)?;
    verify_tcgen05_ptx_contract(&ptx)
        .map_err(|error| format!("PTX verification failed: {error}"))?;

    println!("\nPTX verification (loose build artifact):");
    println!("   PTX file generated at: {}", ptx_path.display());
    println!("\n   To inspect generated PTX:");
    println!("   cat {}", ptx_path.display());
    println!("PASS: PTX was generated successfully; GPU execution skipped");

    Ok(())
}

/// Encode a 2D `SWIZZLE_128B` + `L2_128B` `CUtensorMap` descriptor for an `f16` (`u16`) device buffer
/// of shape `[global_height, global_width]` with tile box `[box_mn, box_k]`.
fn encode_tma_2d_f16_swizzled(
    buf: &DeviceBuffer<u16>,
    global_width: u64,
    global_height: u64,
    box_k: u32,
    box_mn: u32,
) -> Result<cuda_core::sys::CUtensorMap, Box<dyn std::error::Error>> {
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

    // SAFETY: `buf.cu_deviceptr()` is a 256B-aligned device allocation of `global_width * global_height`
    // `u16` elements, and all dimension/stride slices match `tensorRank = 2`.
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
        Ok(tensor_map.assume_init())
    }
}

#[cfg(test)]
mod kernel_family_tests {
    use super::*;
    use cuda_host::SelectionSource;

    fn selected_id(
        family: &GemmKernelFamily,
        m: usize,
        mode: SelectionMode<GemmVariantId>,
    ) -> (GemmVariantId, SelectionSource) {
        let problem = GemmProblem::new(m, m, m);
        let selected = select_gemm_variant(family, &problem, mode).unwrap();
        (*selected.variant().id(), selected.source())
    }

    #[test]
    fn tcgen05_execution_gate_matches_backend_targets() {
        for capability in [(10, 0), (10, 1), (10, 3), (11, 0)] {
            assert!(can_execute_tcgen05_ptx(capability.0, capability.1));
        }
        for capability in [(9, 0), (12, 0), (12, 1)] {
            assert!(!can_execute_tcgen05_ptx(capability.0, capability.1));
        }
    }

    #[test]
    fn ptx_only_pass_requires_target_and_both_kernel_entries() {
        let target = ".target sm_100a";
        let small = ".visible .entry gemm_sol_clc_multicast_4_stage_pipeline(";
        let large = ".visible .entry gemm_sol_clc_multicast_4_stage_pipeline_large(";

        assert!(verify_tcgen05_ptx_contract(&format!("{target}\n{small}\n{large}")).is_ok());
        assert!(verify_tcgen05_ptx_contract(&format!("{target}, debug\n{small}\n{large}")).is_ok());
        assert!(verify_tcgen05_ptx_contract("").is_err());
        assert!(verify_tcgen05_ptx_contract(&format!("{target}\n{small}")).is_err());
        assert!(
            verify_tcgen05_ptx_contract(&format!(".target sm_120a\n{small}\n{large}")).is_err()
        );
        assert!(
            verify_tcgen05_ptx_contract(&format!("// {target}\n// {small}\n// {large}")).is_err()
        );
    }

    #[test]
    fn automatic_policy_preserves_the_measured_size_threshold() {
        let family = gemm_kernel_family().unwrap();

        assert_eq!(
            selected_id(&family, 4096, SelectionMode::Auto),
            (GemmVariantId::M256xN256, SelectionSource::Selector)
        );
        assert_eq!(
            selected_id(&family, 8192, SelectionMode::Auto),
            (GemmVariantId::M256xN256, SelectionSource::Selector)
        );
        assert_eq!(
            selected_id(&family, 16384, SelectionMode::Auto),
            (GemmVariantId::M512xN256, SelectionSource::Selector)
        );
    }

    #[test]
    fn manual_override_can_choose_either_valid_resource_envelope() {
        let family = gemm_kernel_family().unwrap();

        assert_eq!(
            selected_id(
                &family,
                4096,
                SelectionMode::Force(GemmVariantId::M512xN256),
            ),
            (GemmVariantId::M512xN256, SelectionSource::Override)
        );
        assert_eq!(
            selected_id(
                &family,
                16384,
                SelectionMode::Force(GemmVariantId::M256xN256),
            ),
            (GemmVariantId::M256xN256, SelectionSource::Override)
        );
    }

    #[test]
    fn forced_large_variant_rejects_an_odd_m256_tile_count() {
        let family = gemm_kernel_family().unwrap();
        let problem = GemmProblem::new(16640, 16384, 16384);
        let error = select_gemm_variant(
            &family,
            &problem,
            SelectionMode::Force(GemmVariantId::M512xN256),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            cuda_host::KernelSelectionError::IneligibleForcedVariant {
                id: GemmVariantId::M512xN256,
                ..
            }
        ));
    }

    #[test]
    fn family_rejects_shapes_outside_both_kernel_contracts() {
        let family = gemm_kernel_family().unwrap();
        for problem in [
            GemmProblem::new(0, 0, 0),
            GemmProblem::new(4096, 4224, 4096),
            GemmProblem::new(4096, 4096, 4224),
        ] {
            let error = select_gemm_variant(&family, &problem, SelectionMode::Auto).unwrap_err();
            assert!(matches!(
                error,
                cuda_host::KernelSelectionError::NoEligibleVariants { .. }
            ));
        }
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn family_rejects_shapes_that_cannot_fit_the_launch_abi() {
        let family = gemm_kernel_family().unwrap();
        let n_too_large = (i32::MAX as usize / 256 + 1) * 256;
        let problem = GemmProblem::new(4096, n_too_large, 4096);
        let error = select_gemm_variant(
            &family,
            &problem,
            SelectionMode::Force(GemmVariantId::M256xN256),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            cuda_host::KernelSelectionError::IneligibleForcedVariant {
                rejection: GemmVariantIneligible {
                    reason: "M, N, or K exceeds the kernel's signed coordinate/launch ABI",
                    ..
                },
                ..
            }
        ));

        let m_too_large = (i32::MAX as usize / 256 + 2) * 256;
        let problem = GemmProblem::new(m_too_large, 256, 256);
        let error = select_gemm_variant(
            &family,
            &problem,
            SelectionMode::Force(GemmVariantId::M256xN256),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            cuda_host::KernelSelectionError::IneligibleForcedVariant {
                rejection: GemmVariantIneligible {
                    reason: "M, N, or K exceeds the kernel's signed coordinate/launch ABI",
                    ..
                },
                ..
            }
        ));

        let largest_tiled_i32 = (i32::MAX as usize / 256) * 256;
        let problem = GemmProblem::new(16640, largest_tiled_i32, 256);
        let error = select_gemm_variant(
            &family,
            &problem,
            SelectionMode::Force(GemmVariantId::M256xN256),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            cuda_host::KernelSelectionError::IneligibleForcedVariant {
                rejection: GemmVariantIneligible {
                    reason: "grid.x exceeds the CUDA launch limit",
                    ..
                },
                ..
            }
        ));
    }

    #[test]
    fn environment_values_map_to_explicit_selection_modes() {
        assert_eq!(gemm_selection_mode(None).unwrap(), SelectionMode::Auto);
        assert_eq!(
            gemm_selection_mode(Some("m256xn256")).unwrap(),
            SelectionMode::Force(GemmVariantId::M256xN256)
        );
        assert_eq!(
            gemm_selection_mode(Some("m512xn256")).unwrap(),
            SelectionMode::Force(GemmVariantId::M512xN256)
        );
        assert!(gemm_selection_mode(Some("largest")).is_err());
        assert_eq!(
            gemm_selection_mode_from_var(Err(std::env::VarError::NotPresent)).unwrap(),
            SelectionMode::Auto
        );
        assert_eq!("validate".parse::<RunMode>().unwrap(), RunMode::Validate);
        assert_eq!("bench".parse::<RunMode>().unwrap(), RunMode::Bench);
        assert_eq!("both".parse::<RunMode>().unwrap(), RunMode::Both);
        assert_eq!("plan".parse::<RunMode>().unwrap(), RunMode::Plan);
        assert!("unknown".parse::<RunMode>().is_err());

        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;
            let non_utf8 = std::ffi::OsString::from_vec(vec![0xff]);
            assert!(
                gemm_selection_mode_from_var(Err(std::env::VarError::NotUnicode(non_utf8)))
                    .is_err()
            );
        }
    }
}
