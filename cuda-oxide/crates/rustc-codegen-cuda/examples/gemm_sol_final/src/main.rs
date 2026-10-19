/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(
    clippy::not_unsafe_ptr_arg_deref,
    clippy::needless_range_loop,
    clippy::unnecessary_cast,
    clippy::too_many_arguments
)]

//! Canonical cuda-oxide Blackwell GEMM speed-of-light example.
//!
//! Two resource-specialized entry points share the same CLC + `cta_group::2`
//! design: M256xN256 for 4096/8192 and M512xN256 for 16384. Both use a
//! four-stage shared-memory pipeline, compiler-directed K-loop unrolling,
//! L2-aware output-tile ordering, and vectorized 64-bit epilogue stores.
//!
//! Data layout:
//! - A: M×K f16, row-major (K contiguous)
//! - B: N×K f16, row-major (transposed storage, K contiguous)
//! - C: M×N bf16 output, row-major (packed as u32 pairs)

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use cuda_device::barrier::{
    Barrier, fence_proxy_async_shared_cta, mbarrier_arrive, mbarrier_arrive_cluster,
    mbarrier_arrive_expect_tx, mbarrier_init, mbarrier_inval, mbarrier_try_wait_parity,
};
use cuda_device::clc::{
    clc_query_get_first_ctaid_x, clc_query_is_canceled, clc_try_cancel_multicast,
};
use cuda_device::cluster;
use cuda_device::convert::{bf16_to_f32, cvt_bf16x2_f32};
use cuda_device::shared::{SharedArray, cvta_generic_to_shared_offset};
use cuda_device::tcgen05::{
    Tcgen05AccumulatorType, Tcgen05ElementType, Tcgen05InstructionDescriptor, Tcgen05MmaShape,
    stmatrix_m8n8_x2, tcgen05_alloc_cg2, tcgen05_commit_multicast_cg2, tcgen05_dealloc_cg2,
    tcgen05_ld_16x256b_pure, tcgen05_load_wait, tcgen05_mma_f16_cg2,
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
use std::sync::Arc;

// =============================================================================
// LIVE cuBLASLt BASELINE (replaces the previous hardcoded B200 constants)
// =============================================================================

/// Live cuBLASLt FP16-input/FP16-output reference used to compute "% of SoL"
/// in benchmark reports.
///
/// The target kernel writes BF16, but the pinned cuBLASLt stack does not
/// expose an FP16-input/BF16-output matrix combination. Its FP16-output path is the
/// closest supported reference: it uses the same input type, FP32 compute, and
/// two-byte output width, while differing in the final 16-bit conversion. The
/// benchmark is always measured live on the same GPU.
mod cublas_baseline {
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::OnceLock;

    static BASELINE: OnceLock<Option<HashMap<usize, f64>>> = OnceLock::new();

    /// FP16-input / FP16-output / FP32-compute cuBLASLt TFLOPS for an M×M×M
    /// GEMM on the host GPU, or `None` if the bench could not be measured.
    pub fn reference_tflops(m: usize) -> Option<f64> {
        BASELINE.get_or_init(load).as_ref()?.get(&m).copied()
    }

    /// Pre-warm the baseline so the one-time measurement runs at startup, not in
    /// the middle of a benchmark print.
    pub fn warmup() {
        let _ = BASELINE.get_or_init(load);
    }

    fn bench_binary() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("bench")
            .join("cublaslt_bench")
    }

    fn load() -> Option<HashMap<usize, f64>> {
        let bin = bench_binary();
        if !bin.exists() {
            eprintln!(
                "ℹ️  No live cublasLt baseline at {} — % of SoL column will be omitted.",
                bin.display()
            );
            eprintln!(
                "    Build it once with: cd {} && bash build.sh",
                bin.parent().unwrap_or(Path::new(".")).display(),
            );
            return None;
        }

        eprintln!("ℹ️  Measuring live cuBLASLt FP16 reference on host GPU...");
        let out = match std::process::Command::new(&bin).output() {
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
                .map(|(m, t)| format!("{m}={:.1}", t))
                .collect::<Vec<_>>()
                .join(", ");
            eprintln!("✓ live cuBLASLt FP16 reference (TFLOPS): {pretty}");
            Some(map)
        }
    }

    /// Extract `(M, TFLOPS)` pairs from the `--- FP16 ---` section of
    /// `cublaslt_bench` output. Lines look like:
    ///
    /// ```text
    /// FP16 FP32 compute 16384x16384x16384 3.9000 ms 2255.0 TFLOPS
    /// ```
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
            // ["FP16", "FP32", "compute", "MxNxK", "X.XXXX", "ms",
            //  "Y.Y", "TFLOPS"]
            let toks: Vec<&str> = l.split_whitespace().collect();
            let size = toks.get(3).copied().unwrap_or("");
            let m: Option<usize> = size.split('x').next().and_then(|s| s.trim().parse().ok());
            // TFLOPS value is the second-to-last token (last is the literal "TFLOPS")
            let tf: Option<f64> = toks.iter().rev().nth(1).and_then(|s| s.parse().ok());
            if let (Some(m), Some(tf)) = (m, tf) {
                map.insert(m, tf);
            }
        }
        map
    }
}

/// Print the "vs cuBLAS" line for a benchmark result, using the live baseline
/// from `bench/cublaslt_bench` if available, otherwise an explanatory
/// placeholder. Replaces the previous hardcoded `match m { ... }` blocks
/// that compared every host GPU against B200's cublasLt SoL.
fn print_cublas_comparison(tflops: f64, m: usize) {
    match cublas_baseline::reference_tflops(m) {
        Some(sol) => {
            let pct = (tflops / sol) * 100.0;
            println!(
                "  vs cuBLAS:   {:.2}% of live cublasLt FP16 reference ({:.0} TFLOPS)",
                pct, sol
            );
        }
        None => {
            println!("  vs cuBLAS:   (no live cublasLt baseline; see bench/build.sh)");
        }
    }
}

fn print_benchmark_summary(measurements: &[(usize, f64)]) {
    let count = measurements.len() as f64;
    let kernel_geomean = (measurements
        .iter()
        .map(|(_, value)| value.ln())
        .sum::<f64>()
        / count)
        .exp();

    println!("── Fixed-size geomean summary ───────────────────────");
    println!("  Kernel:      {:.3} TFLOPS", kernel_geomean);

    let reference_values: Option<Vec<f64>> = measurements
        .iter()
        .map(|(m, _)| cublas_baseline::reference_tflops(*m))
        .collect();
    if let Some(reference_values) = reference_values {
        let reference_geomean =
            (reference_values.iter().map(|value| value.ln()).sum::<f64>() / count).exp();
        println!("  cuBLAS ref:  {:.3} TFLOPS", reference_geomean);
        println!(
            "  Ratio:       {:.2}% of live FP16 reference",
            kernel_geomean / reference_geomean * 100.0
        );
    } else {
        println!("  cuBLAS ref:  unavailable");
    }
    println!("─────────────────────────────────────────────────────");
}

// =============================================================================
// KERNEL
// =============================================================================

/// Build a tcgen05 SMEM descriptor from components.
///
/// Bit layout of the 64-bit descriptor:
///   [0:13]  base_addr >> 4
///   [16:29] LBO >> 4 (leading byte offset — stride to next core matrix RIGHT)
///   [32:45] SBO >> 4 (stride byte offset — stride to next core matrix DOWN)
///   [46]    fixed 0b1
///   [61:63] swizzle mode
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

// Device kernels live in kernels.rs (the autocuda-editable surface), include!d
// here so #[cuda_module] sees an inline module with a brace body.
include!("kernels.rs");

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GemmVariantId {
    M256xN256,
    M512xN256,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GemmVariantMetadata {
    kernel_name: &'static str,
    output_tile: &'static str,
    m_tile: usize,
    n_tile: usize,
    k_tile_multiple: usize,
    preferred_from_tiles_m: usize,
}

type GemmKernelFamily = KernelFamily<GemmVariantId, GemmKernelLauncher, GemmVariantMetadata, 2>;
type GemmVariant = KernelVariant<GemmVariantId, GemmKernelLauncher, GemmVariantMetadata>;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GemmProblem {
    m: usize,
    n: usize,
    k: usize,
    tiles_m: usize,
}

impl GemmProblem {
    fn new(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            tiles_m: m / 256,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GemmVariantIneligible {
    variant: GemmVariantId,
    reason: &'static str,
    m: usize,
    n: usize,
    k: usize,
    m_tile: usize,
    n_tile: usize,
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

type GemmSelectionError = cuda_host::KernelSelectionError<
    GemmVariantId,
    GemmVariantIneligible,
    std::convert::Infallible,
    std::convert::Infallible,
>;
type GemmSelectedVariant<'family> =
    SelectedVariant<'family, GemmVariantId, GemmKernelLauncher, GemmVariantMetadata>;

fn select_gemm_variant<'family>(
    family: &'family GemmKernelFamily,
    problem: &GemmProblem,
    mode: SelectionMode<GemmVariantId>,
) -> Result<GemmSelectedVariant<'family>, GemmSelectionError> {
    let mut selector = GemmSelector;
    let mut cache = NoKernelSelectionCache;
    family.select(problem, mode, &mut selector, &mut cache)
}

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

fn can_execute_tcgen05_ptx(major: i32, minor: i32) -> bool {
    matches!((major, minor), (10, 0) | (10, 1) | (10, 3) | (11, 0))
}

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("  GEMM SoL — gemm_sol_final (size-specialized)");
    println!("═══════════════════════════════════════════════════════\n");

    let mode = std::env::var("GEMM_SOL_MODE").unwrap_or_else(|_| "both".to_string());
    let (do_validate, do_bench, do_plan) = match mode.as_str() {
        "validate" => (true, false, false),
        "bench" => (false, true, false),
        "both" => (true, true, false),
        "plan" => (false, false, true),
        other => {
            return Err(format!(
                "invalid GEMM_SOL_MODE={other:?}; expected validate, bench, both, or plan"
            )
            .into());
        }
    };
    let family = gemm_kernel_family()?;
    let selection_mode = if do_bench || do_plan {
        gemm_selection_mode_from_var(std::env::var("GEMM_SOL_VARIANT"))?
    } else {
        SelectionMode::Auto
    };

    if do_plan {
        print_gemm_dispatch_plan(&family, selection_mode)?;
        return Ok(());
    }

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let (major, minor) = ctx.compute_capability()?;
    println!("GPU: sm_{}{}", major, minor);

    // Keep this execution set in sync with mir-importer's tcgen05 target
    // support. Other GPU generations may inspect and assemble the generated
    // artifact, but must not turn a module-load failure into an execution pass.
    if !can_execute_tcgen05_ptx(major, minor) {
        println!("\nWARNING: tcgen05 requires a compatible datacenter Blackwell GPU");
        return verify_ptx_only();
    }

    println!("Loading embedded CUDA module");
    let module = kernels::load(&ctx)?;
    println!("Module loaded\n");

    if do_validate {
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

    if do_bench {
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

// A 12-bit Walsh fingerprint gives every row and column in the fixed 4096²
// validator a unique whole-vector signature while retaining an O(MK + NK + MN)
// analytic reference. The odd affine multipliers are permutations mod 4096.
const VALIDATION_CODE_COUNT: usize = 1 << 12;
const VALIDATION_CODE_MASK: u32 = VALIDATION_CODE_COUNT as u32 - 1;

#[inline]
fn validation_affine12(x: usize, multiplier: u32, addend: u32) -> u32 {
    ((x as u32).wrapping_mul(multiplier).wrapping_add(addend)) & VALIDATION_CODE_MASK
}

#[inline]
fn validation_k_code(kk: usize) -> u32 {
    validation_affine12(kk, 251, 17)
}

#[inline]
fn validation_row_code(row: usize) -> u32 {
    validation_affine12(row, 197, 101)
}

#[inline]
fn validation_col_code(col: usize) -> u32 {
    validation_affine12(col, 109, 1021)
}

#[inline]
fn validation_fingerprint(position_code: u32, kk: usize) -> f32 {
    let distance = (position_code ^ validation_k_code(kk)).count_ones() as i32;
    (13 - 2 * distance) as f32
}

#[inline]
fn validation_a_value(row: usize, kk: usize) -> f32 {
    validation_fingerprint(validation_row_code(row), kk)
}

#[inline]
fn validation_b_value(col: usize, kk: usize) -> f32 {
    validation_fingerprint(validation_col_code(col), kk)
}

fn run_correctness_test_clc_multicast_4_stage_pipeline(
    stream: &Arc<CudaStream>,
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
    println!("Matrix: {}x{}x{} (f16 -> bf16)", m, n, k);
    println!("CLC + cta_group::2 + 4-stage SMEM pipeline.");
    println!("Warps: 4=TMA, 5=MMA (leader only), 0-3=epilogue (both CTAs).");

    let mut host_a: Vec<u16> = vec![0u16; m * k];
    for i in 0..m {
        for kk in 0..k {
            host_a[i * k + kk] = f16::from_f32(validation_a_value(i, kk)).to_bits();
        }
    }
    let mut host_b: Vec<u16> = vec![0u16; n * k];
    for j in 0..n {
        for kk in 0..k {
            host_b[j * k + kk] = f16::from_f32(validation_b_value(j, kk)).to_bits();
        }
    }

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
         each K code once, with unique signatures for all {} rows and columns",
        VALIDATION_CODE_COUNT
    );
    println!("Expected: analytic exact FP32 dot product rounded once to BF16\n");

    let dev_a = DeviceBuffer::from_host(stream, &host_a)?;
    let dev_b = DeviceBuffer::from_host(stream, &host_b)?;

    let a_ptr = dev_a.cu_deviceptr();
    let b_ptr = dev_b.cu_deviceptr();

    let a_tma =
        create_tma_descriptor_f16_swizzled(a_ptr as *mut std::ffi::c_void, k as u64, m as u64)?;
    // B TMA: 64 rows per CTA (each CTA loads half the N tile, split by rank)
    let b_tma = create_tma_descriptor_f16_swizzled_box(
        b_ptr as *mut std::ffi::c_void,
        k as u64,
        n as u64,
        64,
        64,
    )?;

    let dev_a_tma = DeviceBuffer::from_host(stream, &a_tma.opaque)?;
    let dev_b_tma = DeviceBuffer::from_host(stream, &b_tma.opaque)?;

    let tiles_m = (m / 256) as u32;
    let tiles_n = (n / 128) as u32;
    let total_tiles = tiles_m * tiles_n;

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
    println!(
        "Host work IDs: {} ({}x{}), logical L2 band G={}",
        total_tiles, tiles_m, tiles_n, swizzle_g
    );
    println!("K-loop: {} outer iters (BK=64, 4 MMAs each)", k / 64);

    let output_u32_count = m * n / 2;
    let mut dev_output = DeviceBuffer::<u32>::zeroed(stream, output_u32_count)?;

    let a_tma_ptr = dev_a_tma.cu_deviceptr();
    let b_tma_ptr = dev_b_tma.cu_deviceptr();
    let a_tma_ptr = a_tma_ptr as *const TmaDescriptor;
    let b_tma_ptr = b_tma_ptr as *const TmaDescriptor;
    let n_arg = n as i32;
    let k_arg = k as i32;

    println!("\nLaunching {kernel_name} (cg2)...");

    unsafe {
        launch_kernel(
            module,
            stream.as_ref(),
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

    // Full-output correctness check (anti reward-hack): compare EVERY BF16
    // element against the exact rounded reference, not a sampled subset. The
    // signed integer inputs make every FP32 product and partial sum exact, so a
    // correct kernel must match the reference BF16 bits.
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
        println!(
            "  first mismatch: C[{},{}] = {} (expected {})",
            r, c, val, exp
        );
    }

    println!("\n═══════════════════════════════════════════════════════");
    if all_ok {
        println!(
            "PASSED: {} {}x{}x{} (full {}-element check)",
            kernel_name, m, n, k, total
        );
    } else {
        println!(
            "FAILED: {} {}x{}x{}: {} / {} elements wrong",
            kernel_name, m, n, k, mismatches, total
        );
        return Err(format!(
            "Correctness check failed: {} / {} elements wrong",
            mismatches, total
        )
        .into());
    }
    println!("═══════════════════════════════════════════════════════");

    Ok(())
}

fn run_benchmark_clc_multicast_4_stage_pipeline(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
    family: &GemmKernelFamily,
    selection_mode: SelectionMode<GemmVariantId>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    const WARMUP: usize = 10;
    const ITERS: usize = 100;

    let problem = GemmProblem::new(m, n, k);
    let selected = select_gemm_variant(family, &problem, selection_mode)?;
    let launch_kernel = *selected.variant().entry();
    let kernel_name = selected.variant().metadata().kernel_name;
    let output_tile = selected.variant().metadata().output_tile;
    let selection_source = selected.source();

    let dev_a = DeviceBuffer::<u16>::zeroed(stream, m * k)?;
    let dev_b = DeviceBuffer::<u16>::zeroed(stream, n * k)?;

    let a_ptr = dev_a.cu_deviceptr();
    let b_ptr = dev_b.cu_deviceptr();

    let a_tma =
        create_tma_descriptor_f16_swizzled(a_ptr as *mut std::ffi::c_void, k as u64, m as u64)?;
    let b_tma = create_tma_descriptor_f16_swizzled_box(
        b_ptr as *mut std::ffi::c_void,
        k as u64,
        n as u64,
        64,
        64,
    )?;

    let dev_a_tma = DeviceBuffer::from_host(stream, &a_tma.opaque)?;
    let dev_b_tma = DeviceBuffer::from_host(stream, &b_tma.opaque)?;
    let a_tma_ptr = dev_a_tma.cu_deviceptr();
    let b_tma_ptr = dev_b_tma.cu_deviceptr();
    let a_tma_ptr = a_tma_ptr as *const TmaDescriptor;
    let b_tma_ptr = b_tma_ptr as *const TmaDescriptor;
    let n_arg = n as i32;
    let k_arg = k as i32;

    let tiles_m = (m / 256) as u32;
    let tiles_n = (n / 128) as u32;
    let total_tiles = tiles_m * tiles_n;

    let cfg = LaunchConfig {
        grid_dim: (total_tiles * 2, 1, 1),
        block_dim: (192, 1, 1),
        shared_mem_bytes: 0,
    };

    let output_u32_count = m * n / 2;
    let mut dev_output = DeviceBuffer::<u32>::zeroed(stream, output_u32_count)?;

    for _ in 0..WARMUP {
        unsafe {
            launch_kernel(
                module,
                stream.as_ref(),
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

    let start_event =
        stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;

    for _ in 0..ITERS {
        unsafe {
            launch_kernel(
                module,
                stream.as_ref(),
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

    let avg_ms = elapsed_ms as f64 / ITERS as f64;
    let avg_us = avg_ms * 1000.0;
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let tflops = (flops / (avg_ms / 1000.0)) / 1e12;

    // cuBLAS comparison is printed by print_cublas_comparison(tflops, m) below.

    println!("═══════════════════════════════════════════════════════");
    println!(
        "  BENCHMARK: {} ({}, cg2) {}x{}x{} f16 -> bf16 [{}]",
        kernel_name, output_tile, m, n, k, selection_source
    );
    println!("═══════════════════════════════════════════════════════");
    println!(
        "  Grid:        {} CTAs (1D, CLC-managed, cluster=2, cg2)",
        total_tiles * 2
    );
    println!("  K-loop:      {} outer iters (BK=64, 4 MMAs each)", k / 64);
    println!("  Pipeline:    CLC + cta_group::2 + 4-stage SMEM + unroll(4)");
    println!("  Iterations:  {} (after {} warmup)", ITERS, WARMUP);
    println!("  Total time:  {:.3} ms", elapsed_ms);
    println!("  Average:     {:.3} us / kernel", avg_us);
    println!("  FLOPS/kern:  {:.3e}", flops);
    println!("  Throughput:  {:.3} TFLOPS", tflops);
    print_cublas_comparison(tflops, m);
    println!("═══════════════════════════════════════════════════════\n");

    Ok(tflops)
}

/// Fallback for GPUs that cannot execute tcgen05: verify the loose PTX build
/// artifact beside this crate against the tcgen05 contract. The main path
/// loads the module embedded in the binary instead; only this fallback reads
/// the loose file.
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

/// Create a 2D TMA descriptor for f16 data with SWIZZLE_128B.
///
/// Single copy of 64 K-elements × 128 M/N rows per TMA instruction.
/// The TMA hardware applies a 128-byte XOR swizzle during the transfer.
fn create_tma_descriptor_f16_swizzled(
    global_address: *mut std::ffi::c_void,
    global_width: u64,
    global_height: u64,
) -> Result<cuda_core::sys::CUtensorMap, Box<dyn std::error::Error>> {
    use cuda_core::sys::{
        CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B, cuTensorMapEncodeTiled,
        cudaError_enum_CUDA_SUCCESS,
    };

    let mut tensor_map = MaybeUninit::<cuda_core::sys::CUtensorMap>::uninit();
    let global_dim: [u64; 2] = [global_width, global_height];
    let global_strides: [u64; 1] = [global_width * 2];
    let box_dim: [u32; 2] = [64, 128]; // 64 K-elements × 128 M/N rows
    let element_strides: [u32; 2] = [1, 1];

    let result = unsafe {
        cuTensorMapEncodeTiled(
            tensor_map.as_mut_ptr(),
            CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            2,
            global_address,
            global_dim.as_ptr(),
            global_strides.as_ptr(),
            box_dim.as_ptr(),
            element_strides.as_ptr(),
            CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };

    if result != cudaError_enum_CUDA_SUCCESS {
        return Err(format!("cuTensorMapEncodeTiled (SWIZZLE_128B) failed: {:?}", result).into());
    }

    Ok(unsafe { tensor_map.assume_init() })
}

fn create_tma_descriptor_f16_swizzled_box(
    global_address: *mut std::ffi::c_void,
    global_width: u64,
    global_height: u64,
    box_k: u32,
    box_mn: u32,
) -> Result<cuda_core::sys::CUtensorMap, Box<dyn std::error::Error>> {
    use cuda_core::sys::{
        CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B, cuTensorMapEncodeTiled,
        cudaError_enum_CUDA_SUCCESS,
    };

    let mut tensor_map = MaybeUninit::<cuda_core::sys::CUtensorMap>::uninit();
    let global_dim: [u64; 2] = [global_width, global_height];
    let global_strides: [u64; 1] = [global_width * 2];
    let box_dim: [u32; 2] = [box_k, box_mn];
    let element_strides: [u32; 2] = [1, 1];

    let result = unsafe {
        cuTensorMapEncodeTiled(
            tensor_map.as_mut_ptr(),
            CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            2,
            global_address,
            global_dim.as_ptr(),
            global_strides.as_ptr(),
            box_dim.as_ptr(),
            element_strides.as_ptr(),
            CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };

    if result != cudaError_enum_CUDA_SUCCESS {
        return Err(format!(
            "cuTensorMapEncodeTiled (SWIZZLE_128B, box {}x{}) failed: {:?}",
            box_k, box_mn, result
        )
        .into());
    }

    Ok(unsafe { tensor_map.assume_init() })
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
