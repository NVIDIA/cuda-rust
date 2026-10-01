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

//! GEMM Speed-of-Light kernels (SM100+ / Blackwell)
//!
//! Eight kernels that implement GEMM on arbitrary M×N×K matrices:
//!
//! - `gemm_sol_tiled` (Phase 1): K-loop + grid tiling, SWIZZLE_NONE (8 TMA copies/tile)
//! - `gemm_sol_swizzled` (Phase 1.5): Same structure, SWIZZLE_128B (1 TMA copy/tile)
//! - `gemm_sol_pipelined` (Phase 2): Double-buffered SMEM, TMA/MMA overlap
//! - `gemm_sol_warp_spec` (Phase 3): Warp-specialized producer/consumer pipeline
//! - `gemm_sol_persistent` (Phase 4A): Persistent + TMEM accum pipeline
//! - `gemm_sol_clc` (Phase 4B): CLC tile scheduling (no multicast)
//! - `gemm_sol_clc_multicast` (Phase 4C): CLC + TMA multicast for B tiles
//! - `gemm_sol_clc_multicast_4_stage_pipeline` (Phase 4D): 4 SMEM stages, no MCAST_BAR
//!
//! All use:
//! - K-loop with accumulation (BK=64, 4 MMAs per K-tile)
//! - Grid-level M/N tiling with blockIdx
//! - No host-side pre-tiling (flat row-major buffers, TMA handles the rest)
//!
//! Data layout:
//! - A: M×K f16, row-major (K contiguous)
//! - B: N×K f16, row-major (transposed storage, K contiguous)
//! - C: M×N bf16 output, row-major (packed as u32 pairs)
//!
//! Build and run with:
//!   cargo oxide run gemm_sol

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::barrier::{
    Barrier, fence_mbarrier_init_release_cluster,
    fence_proxy_async_generic_acquire_shared_cluster_cluster,
    fence_proxy_async_generic_release_shared_cta_cluster, fence_proxy_async_shared_cta,
    mbarrier_arrive, mbarrier_arrive_cluster, mbarrier_arrive_expect_tx,
    mbarrier_arrive_expect_tx_cluster, mbarrier_init, mbarrier_inval, mbarrier_try_wait_parity,
    mbarrier_try_wait_parity_cluster,
};
use cuda_device::clc::{
    clc_query_get_first_ctaid_x, clc_query_is_canceled, clc_try_cancel, clc_try_cancel_multicast,
};
use cuda_device::cluster;
use cuda_device::convert::{bf16_to_f32, cvt_bf16x2_f32};
use cuda_device::shared::{SharedArray, cvta_generic_to_shared_offset};
use cuda_device::tcgen05::{
    Tcgen05AccumulatorType, Tcgen05ElementType, Tcgen05InstructionDescriptor, Tcgen05MmaShape,
    stmatrix_m8n8_x2, tcgen05_alloc, tcgen05_alloc_cg2, tcgen05_commit_multicast_cg2,
    tcgen05_commit_shared_cluster, tcgen05_dealloc, tcgen05_dealloc_cg2, tcgen05_ld_16x256b_pure,
    tcgen05_load_wait, tcgen05_mma_f16, tcgen05_mma_f16_cg2, tcgen05_relinquish_alloc_permit_cg2,
};
use cuda_device::tma::{
    TmaDescriptor, cp_async_bulk_tensor_2d_g2s, cp_async_bulk_tensor_2d_g2s_multicast,
    cp_async_bulk_tensor_2d_g2s_multicast_cg2,
};
use cuda_device::{DisjointSlice, cluster_launch, kernel, thread, warp};
use cuda_host::cuda_module;
use half::{bf16, f16};
use std::mem::MaybeUninit;
use std::sync::Arc;

// =============================================================================
// LIVE cuBLASLt BASELINE (replaces the previous hardcoded B200 constants)
// =============================================================================

/// Live cublasLt FP16 GEMM baseline used to compute "% of SoL" in benchmark
/// reports.
///
/// The baseline is measured by `bench/cublaslt_bench` (a tiny C program that
/// calls `cublasLtMatmul` with the same shapes/dtypes gemm_sol uses). On
/// first access we invoke that binary, parse the FP16 section of its output,
/// and cache the result. If the binary is missing or fails, the per-phase
/// reports omit the "% of SoL" column rather than printing a misleading
/// number against a baseline measured on different silicon.
mod cublas_baseline {
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::OnceLock;

    static BASELINE: OnceLock<Option<HashMap<usize, f64>>> = OnceLock::new();

    /// FP16-input / FP32-compute cublasLtMatmul TFLOPS for an M×M×M GEMM on
    /// the host GPU, or `None` if the bench could not be measured.
    pub fn fp16_tflops(m: usize) -> Option<f64> {
        BASELINE.get_or_init(load).as_ref()?.get(&m).copied()
    }

    /// Pre-warm the baseline so the ~25s measurement runs at startup, not in
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

        eprintln!("ℹ️  Measuring cublasLt FP16 baseline on host GPU (one-shot, ~25s)...");
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
        let map = parse_fp16(&stdout);
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
            eprintln!("✓ cublasLt FP16 baseline (TFLOPS): {pretty}");
            Some(map)
        }
    }

    /// Extract `(M, TFLOPS)` pairs from the `--- FP16 ---` section of
    /// `cublaslt_bench` output. Lines look like:
    ///
    /// ```text
    /// FP16 FP32 compute   16384x16384x16384   20.5993 ms     427.0 TFLOPS
    /// ```
    fn parse_fp16(s: &str) -> HashMap<usize, f64> {
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
            // ["FP16", "FP32", "compute", "MxNxK", "X.XXXX", "ms", "Y.Y", "TFLOPS"]
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

/// Print the "vs cuBLAS" line for a benchmark phase, using the live baseline
/// from `bench/cublaslt_bench` if available, otherwise an explanatory
/// placeholder. Replaces the previous hardcoded `match m { ... }` blocks
/// that compared every host GPU against B200's cublasLt SoL.
fn print_cublas_comparison(tflops: f64, m: usize) {
    match cublas_baseline::fp16_tflops(m) {
        Some(sol) => {
            let pct = (tflops / sol) * 100.0;
            println!(
                "  vs cuBLAS:   {:.2}% of live cublasLt SoL ({:.0} TFLOPS)",
                pct, sol
            );
        }
        None => {
            println!("  vs cuBLAS:   (no live cublasLt baseline; see bench/build.sh)");
        }
    }
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
#[cuda_module]
mod kernels {
    use super::*;

    /// Phase 1 GEMM kernel: K-loop + grid tiling, SWIZZLE_NONE.
    ///
    /// Each CTA computes one 128×128 output tile of C.
    /// The K dimension is processed in BK=64 outer iterations, each containing
    /// 4 sequential MMA ops (128×128×16 each) with accumulation.
    ///
    /// All 128 threads participate in every phase — no warp specialization.
    /// TMA and MMA are serialized: load → wait → compute → wait → next iteration.
    ///
    /// ```text
    ///   All 128 threads (4 warps, no specialization)
    ///   ┌──────────────────────────────────────────────────────┐
    ///   │  for k_idx in 0..K/64:                               │
    ///   │    Thread 0: issue 8 TMA copies (128×8 each)         │
    ///   │    Thread 0: arrive_expect_tx(TMA_BAR, 32KB)         │
    ///   │    ALL:      spin on TMA_BAR[parity] ← wait for DMA  │
    ///   │    ALL:      sync_threads                            │
    ///   │    Thread 0: issue 4 MMAs (128×128×16 each)          │
    ///   │    Thread 0: commit → MMA_BAR                        │
    ///   │    ALL:      spin on MMA_BAR[parity] ← wait for TC   │
    ///   │    ALL:      sync_threads                            │
    ///   │                                                      │
    ///   │  Epilogue: read TMEM → cvt f32→bf16 → stmatrix       │
    ///   │  Global write: SMEM_OUT → GMEM                       │
    ///   └──────────────────────────────────────────────────────┘
    /// ```
    ///
    /// SMEM layout constraint:
    /// The tcgen05 MMA reads 8×8 "core matrices" from SMEM with a hardcoded
    /// 16-byte row stride — there is no descriptor field to override this.
    /// A plain row-major TMA copy (128×64) produces 128-byte row strides,
    /// causing the MMA to read garbage for every row > 0 within each core matrix.
    ///
    /// Fix: split TMA into 8 copies of 128×8 (one per K-group). Each copy
    /// has 16-byte rows in SMEM, matching the hardware expectation. This is
    /// correct but slow (8× more TMA transactions). Phase 1.5 solves this
    /// with SWIZZLE_128B.
    ///
    /// Grid launch: grid_dim = (M/128, N/128, 1), block_dim = (128, 1, 1)
    ///   blockIdx.x → which 128-row block of C (tile_m)
    ///   blockIdx.y → which 128-col block of C (tile_n)
    #[kernel]
    pub unsafe fn gemm_sol_tiled(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        mut out: DisjointSlice<u32>,
        n: i32,
        k: i32,
    ) {
        unsafe {
            // ── Tile dimensions (compile-time constants) ──
            // BM=128, BN=128: output tile per CTA
            // BK=64: K-tile loaded by TMA per outer iteration
            // MMA_K=16: K processed per single MMA instruction
            // NUM_K_MMAS=4: MMAs per K-tile (64/16)

            // SMEM tiles: 128×64 f16 = 16,384 bytes each
            static mut SMEM_A: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_B: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            // Output: 128×128 bf16 = 16,384 elements = 8,192 packed u32
            static mut SMEM_OUT: SharedArray<u32, 8192, 128> = SharedArray::UNINIT;
            static mut TMEM_ADDR: SharedArray<u32, 1, 4> = SharedArray::UNINIT;
            static mut TMA_BAR: Barrier = Barrier::UNINIT;
            static mut MMA_BAR: Barrier = Barrier::UNINIT;

            const A_TILE_BYTES: u32 = 128 * 64 * 2; // 16,384
            const B_TILE_BYTES: u32 = 128 * 64 * 2; // 16,384

            // 8 TMA copies of 128×8 (one per K-group), producing a tiled SMEM layout:
            //   Each K-group: 128 rows × 8 elements × 2 bytes = 2048 bytes
            //   Row stride within K-group: 16 bytes (satisfies MMA's hardcoded 16B stride)
            //
            // MMA navigates this via the SMEM descriptor:
            //   SBO = 128 bytes: stride between 8-row core matrix groups (8 rows × 16 B/row)
            //   LBO = 2048 bytes: stride between K-groups
            //
            // Element (m,k): byte = (k/8)*2048 + m*16 + (k%8)*2
            const SBO_BYTES: u32 = 128;
            const LBO_BYTES: u32 = 2048;
            const SWIZZLE_NONE: u8 = 0;
            const KGROUP_BYTES: u32 = 2048; // 128 rows × 8 elems × 2 bytes

            // Cast i32 kernel params to u32 for unsigned arithmetic
            let n = n as u32;
            let k = k as u32;

            let tid = thread::threadIdx_x();
            let warp_id = warp::warp_id();
            let lane_id = tid % 32;
            let is_thread0 = tid == 0;

            // Grid tiling: each CTA owns a 128×128 output tile
            let tile_m = thread::blockIdx_x(); // which 128-row block of C
            let tile_n = thread::blockIdx_y(); // which 128-col block of C

            // ── PHASE 0: Initialize barriers + allocate TMEM ──
            if is_thread0 {
                mbarrier_init(&raw mut TMA_BAR, 1);
                mbarrier_init(&raw mut MMA_BAR, 1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();

            if warp_id == 0 {
                tcgen05_alloc(&raw mut TMEM_ADDR as *mut u32, 512);
            }
            thread::sync_threads();
            let tmem_addr = *(&raw const TMEM_ADDR as *const u32);

            // Pre-build the MMA instruction descriptor (constant across K-loop)
            let idesc = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape::M128_N128)
                .element_type(Tcgen05ElementType::F16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .build()
                .raw();

            // ── PHASE 1: K-loop ──
            // Outer loop: K/64 iterations. Each loads 128×64 tiles via TMA.
            // Inner loop: 4 MMAs per tile (128×128×16 each), accumulating in TMEM.
            let k_iters = k / 64;
            let mut k_idx: u32 = 0;

            while k_idx < k_iters {
                let phase = k_idx & 1;

                // ── TMA load: fetch 128×64 tiles via 8 copies of 128×8 each ──
                // Each TMA copy fetches 8 K-elements for all 128 M/N rows.
                // The 8 copies are placed at SMEM offsets 0, 2048, 4096, ... bytes,
                // producing a tiled layout where each 128×8 block has 16-byte row strides.
                if is_thread0 {
                    let k_base = (k_idx * 64) as i32;
                    let m_offset = (tile_m * 128) as i32;
                    let n_offset = (tile_n * 128) as i32;
                    let smem_a_ptr = &raw mut SMEM_A as *mut u8;
                    let smem_b_ptr = &raw mut SMEM_B as *mut u8;

                    let mut g: u32 = 0;
                    while g < 8 {
                        let k_offset = k_base + (g * 8) as i32;
                        let smem_off = (g * KGROUP_BYTES) as usize;
                        cp_async_bulk_tensor_2d_g2s(
                            smem_a_ptr.add(smem_off),
                            a_tma,
                            k_offset,
                            m_offset,
                            &raw mut TMA_BAR,
                        );
                        cp_async_bulk_tensor_2d_g2s(
                            smem_b_ptr.add(smem_off),
                            b_tma,
                            k_offset,
                            n_offset,
                            &raw mut TMA_BAR,
                        );
                        g += 1;
                    }
                    mbarrier_arrive_expect_tx(&raw const TMA_BAR, 1, A_TILE_BYTES + B_TILE_BYTES);
                }

                // Wait for TMA completion (all threads spin on parity)
                while !mbarrier_try_wait_parity(&raw const TMA_BAR, phase) {}
                thread::sync_threads();

                // ── 4 MMAs within this K-tile (64 K-elements total) ──
                //
                // We have a 128x64 f16 tile loaded into SMEM. But its layout is as follows:
                // SMEM holds 8 K-groups of 8 elements each (8 × 8 = 64 K-elements).
                // Each MMA instruction (M128_N128) consumes K=16 internally, which
                // spans TWO consecutive K-groups. The hardware automatically reads
                // both K-groups in a single MMA call:
                //   - Descriptor base → first K-group  (e.g. K=0..7)
                //   - base + LBO      → second K-group (e.g. K=8..15)
                //
                // So 8 K-groups ÷ 2 per MMA = 4 MMA calls to cover all 64 K-elements:
                //   j=0: K-groups 0,1 (K= 0..15)  — base offset =     0
                //   j=1: K-groups 2,3 (K=16..31)  — base offset =  4096
                //   j=2: K-groups 4,5 (K=32..47)  — base offset =  8192
                //   j=3: K-groups 6,7 (K=48..63)  — base offset = 12288
                if is_thread0 {
                    let smem_a_base = cvta_generic_to_shared_offset(&raw const SMEM_A as *const u8);
                    let smem_b_base = cvta_generic_to_shared_offset(&raw const SMEM_B as *const u8);

                    let mut j: u32 = 0;
                    while j < 4 {
                        // Point descriptor base at K-group 2j; LBO (2048) lets the
                        // hardware reach K-group 2j+1 automatically within the same op.
                        let byte_offset = (j * 2 * KGROUP_BYTES) as u64;
                        let a_desc = build_smem_descriptor(
                            smem_a_base + byte_offset,
                            LBO_BYTES,
                            SBO_BYTES,
                            SWIZZLE_NONE,
                        );
                        let b_desc = build_smem_descriptor(
                            smem_b_base + byte_offset,
                            LBO_BYTES,
                            SBO_BYTES,
                            SWIZZLE_NONE,
                        );

                        // First MMA ever (k_idx=0, j=0): overwrite TMEM
                        // All subsequent: accumulate into TMEM
                        let accumulate = k_idx > 0 || j > 0;
                        tcgen05_mma_f16(tmem_addr, a_desc, b_desc, idesc, accumulate);
                        j += 1;
                    }

                    tcgen05_commit_shared_cluster(&raw mut MMA_BAR as *mut u64);
                }

                // Wait for MMA completion
                while !mbarrier_try_wait_parity(&raw const MMA_BAR, phase) {}
                thread::sync_threads();

                k_idx += 1;
            }

            // ── PHASE 2: Epilogue — TMEM → registers → SMEM ──
            //
            // TMEM holds 128×128 f32 accumulators. We need to read them, convert to bf16,
            // and write to SMEM_OUT as packed u32 pairs (2 bf16 per u32).
            //
            // TMEM addressing: tmem_addr + (row << 16) + col
            //   - row: TMEM row index (0..127). Bits [31:16] of the TMEM address.
            //   - col: column offset in 32-bit units (0..127). Bits [15:0].
            //
            // tcgen05_ld_16x256b_pure reads a 16-row × 8-column block (256 bytes = 16×8×f32).
            // Returns [4 x u32] per thread — 4 f32 values from the thread's assigned positions.
            //
            // Each warp handles 32 output rows (warp_id * 32 .. warp_id * 32 + 31).
            // Within that, we process 2 blocks of 16 rows × 8 column blocks of 16 columns.
            //
            // stmatrix_m8n8_x2 lane mapping:
            //   lane_id % 8  → which row within an 8-row group (0..7)
            //   lanes  0..7  → first  8×8 matrix (col_offset + 0..7)
            //   lanes  8..15 → second 8×8 matrix (col_offset + 8..15), offset by 16 bytes
            //   lanes 16..31 → address operand ignored; fragment registers still participate
            const TILE_N: usize = 128;
            let warp_row_base = (warp_id * 32) as usize;
            let row_stride_bytes = TILE_N * 2; // 128 bf16 = 256 bytes per row

            let row_within_8 = (lane_id % 8) as usize;
            let is_second_matrix = (8..16).contains(&lane_id);
            let col_offset_for_matrix2 = if is_second_matrix { 16usize } else { 0usize };

            let mut tmem_row_block = 0u32;
            while tmem_row_block < 2 {
                let tmem_row = warp_id * 32 + tmem_row_block * 16;

                let mut col_block = 0u32;
                while col_block < 8 {
                    let col_offset = (col_block * 16) as usize;

                    // Two TMEM loads per column block: columns [0..7] and [8..15]
                    let regs_a =
                        tcgen05_ld_16x256b_pure(tmem_addr + (tmem_row << 16) + col_offset as u32);
                    tcgen05_load_wait();

                    let regs_b = tcgen05_ld_16x256b_pure(
                        tmem_addr + (tmem_row << 16) + col_offset as u32 + 8,
                    );
                    tcgen05_load_wait();

                    // Convert f32 pairs → packed bf16 pairs, write via stmatrix
                    let p0_lo = cvt_bf16x2_f32(regs_a[0], regs_a[1]);
                    let p1_lo = cvt_bf16x2_f32(regs_b[0], regs_b[1]);

                    let out_row_lo = warp_row_base + (tmem_row_block as usize * 16) + row_within_8;
                    let smem_addr_lo = (&raw mut SMEM_OUT as *mut u8).add(
                        out_row_lo * row_stride_bytes + col_offset * 2 + col_offset_for_matrix2,
                    );
                    stmatrix_m8n8_x2(smem_addr_lo, p0_lo, p1_lo);

                    // Same for the upper 8 rows of this 16-row block
                    let p0_hi = cvt_bf16x2_f32(regs_a[2], regs_a[3]);
                    let p1_hi = cvt_bf16x2_f32(regs_b[2], regs_b[3]);

                    let out_row_hi =
                        warp_row_base + (tmem_row_block as usize * 16) + 8 + row_within_8;
                    let smem_addr_hi = (&raw mut SMEM_OUT as *mut u8).add(
                        out_row_hi * row_stride_bytes + col_offset * 2 + col_offset_for_matrix2,
                    );
                    stmatrix_m8n8_x2(smem_addr_hi, p0_hi, p1_hi);

                    col_block += 1;
                }
                tmem_row_block += 1;
            }

            thread::sync_threads();

            // ── PHASE 3: Copy output tile to global memory (scattered write) ──
            // SMEM_OUT layout: 128 rows × 64 u32 per row (= 128 bf16 per row)
            // Global C layout: M rows × (N/2) u32 per row
            let n_u32 = (n / 2) as usize; // u32 values per row in global C
            let tile_row_base = (tile_m * 128) as usize;
            let tile_col_base = (tile_n * 64) as usize; // tile_n * 128 / 2

            let mut local_idx = tid as usize;
            while local_idx < 8192 {
                let local_row = local_idx >> 6; // / 64
                let local_col = local_idx & 63; // % 64

                let global_row = tile_row_base + local_row;
                let global_col = tile_col_base + local_col;
                let global_idx = global_row * n_u32 + global_col;

                *out.get_unchecked_mut(global_idx) = SMEM_OUT[local_idx];
                local_idx += 128;
            }

            // ── PHASE 4: Cleanup ──
            thread::sync_threads();
            if warp_id == 0 {
                tcgen05_dealloc(tmem_addr, 512);
            }
            if is_thread0 {
                mbarrier_inval(&raw mut TMA_BAR);
                mbarrier_inval(&raw mut MMA_BAR);
            }
        }
    }

    /// Phase 1.5 GEMM kernel: SWIZZLE_128B — single TMA copy per matrix per K-tile.
    ///
    /// Phase 1 used 8 TMA copies of 128×8 to produce 16-byte row strides matching
    /// the MMA's hardcoded core matrix layout. This works but costs 8× TMA overhead.
    ///
    /// SWIZZLE_128B solves the same problem in hardware: TMA copies a full 128×64
    /// tile in one instruction, but XOR-swizzles byte addresses during the transfer.
    /// The MMA applies the matching de-swizzle when reading. The net effect: data
    /// appears to have 16-byte core matrix row strides from the MMA's perspective,
    /// even though SMEM physically has 128-byte rows.
    ///
    /// Bank conflict elimination is a secondary benefit of the swizzle pattern.
    ///
    /// Changes from Phase 1:
    ///   - TMA: 1 copy of 128×64 per matrix (was 8 copies of 128×8)
    ///   - SMEM descriptor: SBO=1024, LBO=16, swizzle=2 (was SBO=128, LBO=2048, swizzle=0)
    ///   - MMA byte offsets: j*32 (was j*4096)
    ///   - Epilogue + grid tiling: identical to Phase 1
    ///
    /// The net effect: same computation, same correctness, but ~45% faster TMA throughput
    /// from fewer DMA transactions and correct core matrix layout via hardware swizzle.
    #[kernel]
    pub unsafe fn gemm_sol_swizzled(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        mut out: DisjointSlice<u32>,
        n: i32,
        k: i32,
    ) {
        unsafe {
            static mut SMEM_A: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_B: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_OUT: SharedArray<u32, 8192, 128> = SharedArray::UNINIT;
            static mut TMEM_ADDR: SharedArray<u32, 1, 4> = SharedArray::UNINIT;
            static mut TMA_BAR: Barrier = Barrier::UNINIT;
            static mut MMA_BAR: Barrier = Barrier::UNINIT;

            const A_TILE_BYTES: u32 = 128 * 64 * 2; // 16,384
            const B_TILE_BYTES: u32 = 128 * 64 * 2; // 16,384

            // With SWIZZLE_128B, TMA copies a contiguous 128×64 f16 tile (one instruction).
            // The swizzle hardware rearranges bytes so the MMA sees core matrices correctly.
            // SBO/LBO describe inter-core-matrix strides in the logical (swizzled) space:
            //   SBO = 1024: 8 rows × 128 bytes/row (stride to next M-group)
            //   LBO = 16:   8 elements × 2 bytes   (stride to next K-group)
            // The MMA's hardcoded 16-byte intra-row stride is satisfied by the swizzle.
            const SBO_BYTES: u32 = 1024;
            const LBO_BYTES: u32 = 16;
            const SWIZZLE_128B: u8 = 2;

            let n = n as u32;
            let k = k as u32;

            let tid = thread::threadIdx_x();
            let warp_id = warp::warp_id();
            let lane_id = tid % 32;
            let is_thread0 = tid == 0;

            let tile_m = thread::blockIdx_x();
            let tile_n = thread::blockIdx_y();

            // ── PHASE 0: Initialize barriers + allocate TMEM ──
            if is_thread0 {
                mbarrier_init(&raw mut TMA_BAR, 1);
                mbarrier_init(&raw mut MMA_BAR, 1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();

            if warp_id == 0 {
                tcgen05_alloc(&raw mut TMEM_ADDR as *mut u32, 512);
            }
            thread::sync_threads();
            let tmem_addr = *(&raw const TMEM_ADDR as *const u32);

            let idesc = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape::M128_N128)
                .element_type(Tcgen05ElementType::F16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .build()
                .raw();

            // ── PHASE 1: K-loop ──
            let k_iters = k / 64;
            let mut k_idx: u32 = 0;

            while k_idx < k_iters {
                let phase = k_idx & 1;

                // ── TMA load: single copy of 128×64 per matrix ──
                // With SWIZZLE_128B, TMA fetches the entire K-tile in one instruction
                // and applies a byte-level XOR swizzle during the GMEM→SMEM transfer.
                if is_thread0 {
                    let k_base = (k_idx * 64) as i32;
                    let m_offset = (tile_m * 128) as i32;
                    let n_offset = (tile_n * 128) as i32;
                    let smem_a_ptr = &raw mut SMEM_A as *mut u8;
                    let smem_b_ptr = &raw mut SMEM_B as *mut u8;

                    cp_async_bulk_tensor_2d_g2s(
                        smem_a_ptr,
                        a_tma,
                        k_base,
                        m_offset,
                        &raw mut TMA_BAR,
                    );
                    cp_async_bulk_tensor_2d_g2s(
                        smem_b_ptr,
                        b_tma,
                        k_base,
                        n_offset,
                        &raw mut TMA_BAR,
                    );
                    mbarrier_arrive_expect_tx(&raw const TMA_BAR, 1, A_TILE_BYTES + B_TILE_BYTES);
                }

                while !mbarrier_try_wait_parity(&raw const TMA_BAR, phase) {}
                thread::sync_threads();

                // ── 4 MMAs within this K-tile ──
                // Each MMA consumes K=16 (two K-groups of 8 elements).
                // In the swizzled layout, consecutive K-groups are LBO=16 bytes apart,
                // so two K-groups span 32 bytes. The 4 MMAs step through:
                //   j=0: byte offset  0 (K= 0..15)
                //   j=1: byte offset 32 (K=16..31)
                //   j=2: byte offset 64 (K=32..47)
                //   j=3: byte offset 96 (K=48..63)
                if is_thread0 {
                    let smem_a_base = cvta_generic_to_shared_offset(&raw const SMEM_A as *const u8);
                    let smem_b_base = cvta_generic_to_shared_offset(&raw const SMEM_B as *const u8);

                    let mut j: u32 = 0;
                    while j < 4 {
                        let byte_offset = (j * 32) as u64;
                        let a_desc = build_smem_descriptor(
                            smem_a_base + byte_offset,
                            LBO_BYTES,
                            SBO_BYTES,
                            SWIZZLE_128B,
                        );
                        let b_desc = build_smem_descriptor(
                            smem_b_base + byte_offset,
                            LBO_BYTES,
                            SBO_BYTES,
                            SWIZZLE_128B,
                        );

                        let accumulate = k_idx > 0 || j > 0;
                        tcgen05_mma_f16(tmem_addr, a_desc, b_desc, idesc, accumulate);
                        j += 1;
                    }

                    tcgen05_commit_shared_cluster(&raw mut MMA_BAR as *mut u64);
                }

                while !mbarrier_try_wait_parity(&raw const MMA_BAR, phase) {}
                thread::sync_threads();

                k_idx += 1;
            }

            // ── PHASE 2: Epilogue — TMEM → registers → SMEM (identical to Phase 1) ──
            const TILE_N: usize = 128;
            let warp_row_base = (warp_id * 32) as usize;
            let row_stride_bytes = TILE_N * 2;

            let row_within_8 = (lane_id % 8) as usize;
            let is_second_matrix = (8..16).contains(&lane_id);
            let col_offset_for_matrix2 = if is_second_matrix { 16usize } else { 0usize };

            let mut tmem_row_block = 0u32;
            while tmem_row_block < 2 {
                let tmem_row = warp_id * 32 + tmem_row_block * 16;

                let mut col_block = 0u32;
                while col_block < 8 {
                    let col_offset = (col_block * 16) as usize;

                    let regs_a =
                        tcgen05_ld_16x256b_pure(tmem_addr + (tmem_row << 16) + col_offset as u32);
                    tcgen05_load_wait();

                    let regs_b = tcgen05_ld_16x256b_pure(
                        tmem_addr + (tmem_row << 16) + col_offset as u32 + 8,
                    );
                    tcgen05_load_wait();

                    let p0_lo = cvt_bf16x2_f32(regs_a[0], regs_a[1]);
                    let p1_lo = cvt_bf16x2_f32(regs_b[0], regs_b[1]);

                    let out_row_lo = warp_row_base + (tmem_row_block as usize * 16) + row_within_8;
                    let smem_addr_lo = (&raw mut SMEM_OUT as *mut u8).add(
                        out_row_lo * row_stride_bytes + col_offset * 2 + col_offset_for_matrix2,
                    );
                    stmatrix_m8n8_x2(smem_addr_lo, p0_lo, p1_lo);

                    let p0_hi = cvt_bf16x2_f32(regs_a[2], regs_a[3]);
                    let p1_hi = cvt_bf16x2_f32(regs_b[2], regs_b[3]);

                    let out_row_hi =
                        warp_row_base + (tmem_row_block as usize * 16) + 8 + row_within_8;
                    let smem_addr_hi = (&raw mut SMEM_OUT as *mut u8).add(
                        out_row_hi * row_stride_bytes + col_offset * 2 + col_offset_for_matrix2,
                    );
                    stmatrix_m8n8_x2(smem_addr_hi, p0_hi, p1_hi);

                    col_block += 1;
                }
                tmem_row_block += 1;
            }

            thread::sync_threads();

            // ── PHASE 3: Copy output tile to global memory ──
            let n_u32 = (n / 2) as usize;
            let tile_row_base = (tile_m * 128) as usize;
            let tile_col_base = (tile_n * 64) as usize;

            let mut local_idx = tid as usize;
            while local_idx < 8192 {
                let local_row = local_idx >> 6;
                let local_col = local_idx & 63;

                let global_row = tile_row_base + local_row;
                let global_col = tile_col_base + local_col;
                let global_idx = global_row * n_u32 + global_col;

                *out.get_unchecked_mut(global_idx) = SMEM_OUT[local_idx];
                local_idx += 128;
            }

            // ── PHASE 4: Cleanup ──
            thread::sync_threads();
            if warp_id == 0 {
                tcgen05_dealloc(tmem_addr, 512);
            }
            if is_thread0 {
                mbarrier_inval(&raw mut TMA_BAR);
                mbarrier_inval(&raw mut MMA_BAR);
            }
        }
    }

    /// Phase 2 GEMM kernel: Double-buffered SMEM with TMA/MMA overlap.
    ///
    /// Same MMA and epilogue as Phase 1.5 (SWIZZLE_128B), but the K-loop now
    /// uses two SMEM buffers for A and B. While MMA computes on one buffer,
    /// TMA prefetches the next K-tile into the other buffer.
    ///
    /// ```text
    ///   Timeline (all 128 threads, single-threaded control):
    ///
    ///   Prologue: TMA kick k=0 → buf0
    ///   ┌─────────────────────────────────────────────────┐
    ///   │ k=0: wait TMA_BAR0 │ MMA on buf0 │ TMA k=1→buf1 │
    ///   │ k=1: wait TMA_BAR1 │ MMA on buf1 │ TMA k=2→buf0 │
    ///   │ k=2: wait TMA_BAR0 │ MMA on buf0 │ TMA k=3→buf1 │
    ///   │ ...                                             │
    ///   │ k=N: wait TMA_BARx │ MMA on bufx │ (no prefetch)│
    ///   └─────────────────────────────────────────────────┘
    ///   Epilogue: TMEM → bf16 → GMEM
    /// ```
    ///
    /// Barrier parity: TMA_BAR0 and TMA_BAR1 alternate. Each fires every 2 iterations,
    /// so its parity is `(k_idx / 2) & 1`. The MMA barrier fires every iteration,
    /// so its parity is `k_idx & 1`.
    ///
    /// Changes from Phase 1.5:
    ///   - 2× SMEM for A and B (32KB each → 64KB total data buffers)
    ///   - 2 TMA barriers (one per buffer) instead of 1
    ///   - Prologue loads k=0 before entering the loop
    ///   - Steady state: MMA on buf[k%2], TMA prefetch into buf[(k+1)%2]
    ///   - Epilogue + grid tiling: identical to Phase 1.5
    #[kernel]
    pub unsafe fn gemm_sol_pipelined(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        mut out: DisjointSlice<u32>,
        n: i32,
        k: i32,
    ) {
        unsafe {
            // Double SMEM buffers: two tiles each for A and B
            static mut SMEM_A0: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_A1: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_B0: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_B1: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_OUT: SharedArray<u32, 8192, 128> = SharedArray::UNINIT;
            static mut TMEM_ADDR: SharedArray<u32, 1, 4> = SharedArray::UNINIT;
            // One TMA barrier per buffer + one MMA barrier
            static mut TMA_BAR0: Barrier = Barrier::UNINIT;
            static mut TMA_BAR1: Barrier = Barrier::UNINIT;
            static mut MMA_BAR: Barrier = Barrier::UNINIT;

            const A_TILE_BYTES: u32 = 128 * 64 * 2; // 16,384
            const B_TILE_BYTES: u32 = 128 * 64 * 2; // 16,384

            const SBO_BYTES: u32 = 1024;
            const LBO_BYTES: u32 = 16;
            const SWIZZLE_128B: u8 = 2;

            let n = n as u32;
            let k = k as u32;

            let tid = thread::threadIdx_x();
            let warp_id = warp::warp_id();
            let lane_id = tid % 32;
            let is_thread0 = tid == 0;

            let tile_m = thread::blockIdx_x();
            let tile_n = thread::blockIdx_y();

            // ── PHASE 0: Initialize barriers + allocate TMEM ──
            if is_thread0 {
                mbarrier_init(&raw mut TMA_BAR0, 1);
                mbarrier_init(&raw mut TMA_BAR1, 1);
                mbarrier_init(&raw mut MMA_BAR, 1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();

            if warp_id == 0 {
                tcgen05_alloc(&raw mut TMEM_ADDR as *mut u32, 512);
            }
            thread::sync_threads();
            let tmem_addr = *(&raw const TMEM_ADDR as *const u32);

            let idesc = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape::M128_N128)
                .element_type(Tcgen05ElementType::F16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .build()
                .raw();

            let k_iters = k / 64;
            let m_offset = (tile_m * 128) as i32;
            let n_offset = (tile_n * 128) as i32;

            // ── Prologue: kick off TMA for k=0 into buffer 0 ──
            if is_thread0 {
                cp_async_bulk_tensor_2d_g2s(
                    &raw mut SMEM_A0 as *mut u8,
                    a_tma,
                    0,
                    m_offset,
                    &raw mut TMA_BAR0,
                );
                cp_async_bulk_tensor_2d_g2s(
                    &raw mut SMEM_B0 as *mut u8,
                    b_tma,
                    0,
                    n_offset,
                    &raw mut TMA_BAR0,
                );
                mbarrier_arrive_expect_tx(&raw const TMA_BAR0, 1, A_TILE_BYTES + B_TILE_BYTES);
            }

            // ── PHASE 1: K-loop with double buffering ──
            //
            // Each iteration:
            //   1. Wait for current buffer's TMA to complete
            //   2. Issue MMA on current buffer (async → tensor core)
            //   3. Prefetch k+1 into the other buffer (async → DMA, overlaps with MMA)
            //   4. Wait for MMA to complete
            //
            // TMA parity = (k_idx / 2) & 1 because each TMA barrier fires every other iter.
            // MMA parity = k_idx & 1 because MMA barrier fires every iter.
            let mut k_idx: u32 = 0;

            while k_idx < k_iters {
                let buf = k_idx & 1;
                let tma_parity = (k_idx >> 1) & 1;
                let mma_parity = k_idx & 1;

                // Step 1: Wait for current buffer's TMA to complete
                if buf == 0 {
                    while !mbarrier_try_wait_parity(&raw const TMA_BAR0, tma_parity) {}
                } else {
                    while !mbarrier_try_wait_parity(&raw const TMA_BAR1, tma_parity) {}
                }
                thread::sync_threads();

                // Step 2: Issue MMA on current buffer (async → tensor core)
                if is_thread0 {
                    let smem_a_base = if buf == 0 {
                        cvta_generic_to_shared_offset(&raw const SMEM_A0 as *const u8)
                    } else {
                        cvta_generic_to_shared_offset(&raw const SMEM_A1 as *const u8)
                    };
                    let smem_b_base = if buf == 0 {
                        cvta_generic_to_shared_offset(&raw const SMEM_B0 as *const u8)
                    } else {
                        cvta_generic_to_shared_offset(&raw const SMEM_B1 as *const u8)
                    };

                    let mut j: u32 = 0;
                    while j < 4 {
                        let byte_offset = (j * 32) as u64;
                        let a_desc = build_smem_descriptor(
                            smem_a_base + byte_offset,
                            LBO_BYTES,
                            SBO_BYTES,
                            SWIZZLE_128B,
                        );
                        let b_desc = build_smem_descriptor(
                            smem_b_base + byte_offset,
                            LBO_BYTES,
                            SBO_BYTES,
                            SWIZZLE_128B,
                        );

                        let accumulate = k_idx > 0 || j > 0;
                        tcgen05_mma_f16(tmem_addr, a_desc, b_desc, idesc, accumulate);
                        j += 1;
                    }

                    tcgen05_commit_shared_cluster(&raw mut MMA_BAR as *mut u64);
                }

                // Step 3: Prefetch k+1 into the other buffer (overlaps with MMA)
                if is_thread0 && k_idx + 1 < k_iters {
                    let next_k_base = ((k_idx + 1) * 64) as i32;
                    if buf == 0 {
                        cp_async_bulk_tensor_2d_g2s(
                            &raw mut SMEM_A1 as *mut u8,
                            a_tma,
                            next_k_base,
                            m_offset,
                            &raw mut TMA_BAR1,
                        );
                        cp_async_bulk_tensor_2d_g2s(
                            &raw mut SMEM_B1 as *mut u8,
                            b_tma,
                            next_k_base,
                            n_offset,
                            &raw mut TMA_BAR1,
                        );
                        mbarrier_arrive_expect_tx(
                            &raw const TMA_BAR1,
                            1,
                            A_TILE_BYTES + B_TILE_BYTES,
                        );
                    } else {
                        cp_async_bulk_tensor_2d_g2s(
                            &raw mut SMEM_A0 as *mut u8,
                            a_tma,
                            next_k_base,
                            m_offset,
                            &raw mut TMA_BAR0,
                        );
                        cp_async_bulk_tensor_2d_g2s(
                            &raw mut SMEM_B0 as *mut u8,
                            b_tma,
                            next_k_base,
                            n_offset,
                            &raw mut TMA_BAR0,
                        );
                        mbarrier_arrive_expect_tx(
                            &raw const TMA_BAR0,
                            1,
                            A_TILE_BYTES + B_TILE_BYTES,
                        );
                    }
                }

                // Step 4: Wait for MMA to complete
                while !mbarrier_try_wait_parity(&raw const MMA_BAR, mma_parity) {}
                thread::sync_threads();

                k_idx += 1;
            }

            // ── PHASE 2: Epilogue — TMEM → registers → SMEM (identical to Phase 1.5) ──
            const TILE_N: usize = 128;
            let warp_row_base = (warp_id * 32) as usize;
            let row_stride_bytes = TILE_N * 2;

            let row_within_8 = (lane_id % 8) as usize;
            let is_second_matrix = (8..16).contains(&lane_id);
            let col_offset_for_matrix2 = if is_second_matrix { 16usize } else { 0usize };

            let mut tmem_row_block = 0u32;
            while tmem_row_block < 2 {
                let tmem_row = warp_id * 32 + tmem_row_block * 16;

                let mut col_block = 0u32;
                while col_block < 8 {
                    let col_offset = (col_block * 16) as usize;

                    let regs_a =
                        tcgen05_ld_16x256b_pure(tmem_addr + (tmem_row << 16) + col_offset as u32);
                    tcgen05_load_wait();

                    let regs_b = tcgen05_ld_16x256b_pure(
                        tmem_addr + (tmem_row << 16) + col_offset as u32 + 8,
                    );
                    tcgen05_load_wait();

                    let p0_lo = cvt_bf16x2_f32(regs_a[0], regs_a[1]);
                    let p1_lo = cvt_bf16x2_f32(regs_b[0], regs_b[1]);

                    let out_row_lo = warp_row_base + (tmem_row_block as usize * 16) + row_within_8;
                    let smem_addr_lo = (&raw mut SMEM_OUT as *mut u8).add(
                        out_row_lo * row_stride_bytes + col_offset * 2 + col_offset_for_matrix2,
                    );
                    stmatrix_m8n8_x2(smem_addr_lo, p0_lo, p1_lo);

                    let p0_hi = cvt_bf16x2_f32(regs_a[2], regs_a[3]);
                    let p1_hi = cvt_bf16x2_f32(regs_b[2], regs_b[3]);

                    let out_row_hi =
                        warp_row_base + (tmem_row_block as usize * 16) + 8 + row_within_8;
                    let smem_addr_hi = (&raw mut SMEM_OUT as *mut u8).add(
                        out_row_hi * row_stride_bytes + col_offset * 2 + col_offset_for_matrix2,
                    );
                    stmatrix_m8n8_x2(smem_addr_hi, p0_hi, p1_hi);

                    col_block += 1;
                }
                tmem_row_block += 1;
            }

            thread::sync_threads();

            // ── PHASE 3: Copy output tile to global memory ──
            let n_u32 = (n / 2) as usize;
            let tile_row_base = (tile_m * 128) as usize;
            let tile_col_base = (tile_n * 64) as usize;

            let mut local_idx = tid as usize;
            while local_idx < 8192 {
                let local_row = local_idx >> 6;
                let local_col = local_idx & 63;

                let global_row = tile_row_base + local_row;
                let global_col = tile_col_base + local_col;
                let global_idx = global_row * n_u32 + global_col;

                *out.get_unchecked_mut(global_idx) = SMEM_OUT[local_idx];
                local_idx += 128;
            }

            // ── PHASE 4: Cleanup ──
            thread::sync_threads();
            if warp_id == 0 {
                tcgen05_dealloc(tmem_addr, 512);
            }
            if is_thread0 {
                mbarrier_inval(&raw mut TMA_BAR0);
                mbarrier_inval(&raw mut TMA_BAR1);
                mbarrier_inval(&raw mut MMA_BAR);
            }
        }
    }

    /// Phase 3 GEMM kernel: Warp-specialized pipelined producer/consumer.
    ///
    /// The K-loop is split across dedicated warps that run independent loops,
    /// coordinated purely through mbarriers — no sync_threads inside the K-loop.
    ///
    /// ```text
    ///   6 warps (192 threads), 1 output tile per CTA:
    ///
    ///   Warp 4 (TMA Producer)     Warp 5 (MMA Consumer)     Warps 0-3 (Epilogue)
    ///   ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
    ///   │ for k in 0..K/64:│      │ for k in 0..K/64:│      │                  │
    ///   │  wait MMA_BAR[s] │◄─────│  wait TMA_BAR[s] │      │   (idle during   │
    ///   │  TMA A,B → buf[s]│      │  4× MMA on buf[s]│      │    K-loop)       │
    ///   │  signal TMA_BAR  │─────▶│  signal MMA_BAR  │─┐    │                  │
    ///   │                  │      │                  │ │    │                  │
    ///   │ (exit loop)      │      │  last k: commit  │ │    │  wait COMPUTE_BAR│
    ///   │                  │      │  → COMPUTE_BAR   │─┼───▶│  TMEM→bf16→GMEM  │
    ///   └──────────────────┘      └──────────────────┘ │    └──────────────────┘
    ///                                                  │
    ///                              MMA_BAR releases ───┘
    ///                              buffer for TMA reuse
    /// ```
    ///
    /// Pre-signal trick: MMA_BAR0/1 are arrived once before the loops start.
    /// Without this, the TMA producer would deadlock on iteration 0 waiting for
    /// MMA to release a buffer that was never consumed. The pre-signal gives
    /// the producer permission to fill both buffers before MMA has started.
    ///
    /// Changes from Phase 2:
    ///   - block_dim: 192 (6 warps) instead of 128 (4 warps)
    ///   - Warp 4 = TMA producer, Warp 5 = MMA consumer, Warps 0-3 = epilogue
    ///   - 5 barriers: TMA_BAR0/1 (producer→consumer), MMA_BAR0/1 (consumer→producer),
    ///     COMPUTE_BAR (consumer→epilogue)
    ///   - No sync_threads in K-loop (warps are fully decoupled)
    #[kernel]
    pub unsafe fn gemm_sol_warp_spec(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        mut out: DisjointSlice<u32>,
        n: i32,
        k: i32,
    ) {
        unsafe {
            // Double SMEM buffers (same as Phase 2)
            static mut SMEM_A0: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_A1: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_B0: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_B1: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_OUT: SharedArray<u32, 8192, 128> = SharedArray::UNINIT;
            static mut TMEM_ADDR: SharedArray<u32, 1, 4> = SharedArray::UNINIT;
            // Per-stage barriers + compute barrier
            static mut TMA_BAR0: Barrier = Barrier::UNINIT;
            static mut TMA_BAR1: Barrier = Barrier::UNINIT;
            static mut MMA_BAR0: Barrier = Barrier::UNINIT;
            static mut MMA_BAR1: Barrier = Barrier::UNINIT;
            static mut COMPUTE_BAR: Barrier = Barrier::UNINIT;

            const A_TILE_BYTES: u32 = 128 * 64 * 2;
            const B_TILE_BYTES: u32 = 128 * 64 * 2;

            const SBO_BYTES: u32 = 1024;
            const LBO_BYTES: u32 = 16;
            const SWIZZLE_128B: u8 = 2;

            const TMA_WARP: u32 = 4;
            const MMA_WARP: u32 = 5;

            let n = n as u32;
            let k = k as u32;

            let tid = thread::threadIdx_x();
            let warp_id = warp::warp_id();
            let lane_id = tid % 32;

            let tile_m = thread::blockIdx_x();
            let tile_n = thread::blockIdx_y();

            // ── PHASE 0: Initialize barriers + allocate TMEM ──
            if tid == 0 {
                mbarrier_init(&raw mut TMA_BAR0, 1);
                mbarrier_init(&raw mut TMA_BAR1, 1);
                mbarrier_init(&raw mut MMA_BAR0, 1);
                mbarrier_init(&raw mut MMA_BAR1, 1);
                mbarrier_init(&raw mut COMPUTE_BAR, 1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();

            // Pre-signal MMA_BAR0/1 so the producer can start filling buffers
            // immediately without waiting for a consumer release.
            if tid == 0 {
                mbarrier_arrive(&raw const MMA_BAR0);
                mbarrier_arrive(&raw const MMA_BAR1);
            }
            thread::sync_threads();

            if warp_id == 0 {
                tcgen05_alloc(&raw mut TMEM_ADDR as *mut u32, 512);
            }
            thread::sync_threads();
            let tmem_addr = *(&raw const TMEM_ADDR as *const u32);

            let idesc = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape::M128_N128)
                .element_type(Tcgen05ElementType::F16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .build()
                .raw();

            let k_iters = k / 64;
            let m_offset = (tile_m * 128) as i32;
            let n_offset = (tile_n * 128) as i32;

            // ── PHASE 1: Warp-specialized K-loop ──
            //
            // Warp 4 (TMA producer): loads K-tiles into double-buffered SMEM
            // Warp 5 (MMA consumer): computes on loaded tiles, accumulates into TMEM
            // Warps 0-3: idle during K-loop, active during epilogue
            //
            // No sync_threads inside — coordination is purely through mbarriers.

            // ── TMA Producer (warp 4) ──
            if warp_id == TMA_WARP {
                let is_lane0 = lane_id == 0;
                let mut k_idx: u32 = 0;

                while k_idx < k_iters {
                    let stage = k_idx & 1;
                    let mma_parity = (k_idx >> 1) & 1;

                    // Wait for consumer to release this buffer
                    if stage == 0 {
                        while !mbarrier_try_wait_parity(&raw const MMA_BAR0, mma_parity) {}
                    } else {
                        while !mbarrier_try_wait_parity(&raw const MMA_BAR1, mma_parity) {}
                    }

                    // Issue TMA load into this buffer
                    if is_lane0 {
                        let k_base = (k_idx * 64) as i32;
                        if stage == 0 {
                            cp_async_bulk_tensor_2d_g2s(
                                &raw mut SMEM_A0 as *mut u8,
                                a_tma,
                                k_base,
                                m_offset,
                                &raw mut TMA_BAR0,
                            );
                            cp_async_bulk_tensor_2d_g2s(
                                &raw mut SMEM_B0 as *mut u8,
                                b_tma,
                                k_base,
                                n_offset,
                                &raw mut TMA_BAR0,
                            );
                            mbarrier_arrive_expect_tx(
                                &raw const TMA_BAR0,
                                1,
                                A_TILE_BYTES + B_TILE_BYTES,
                            );
                        } else {
                            cp_async_bulk_tensor_2d_g2s(
                                &raw mut SMEM_A1 as *mut u8,
                                a_tma,
                                k_base,
                                m_offset,
                                &raw mut TMA_BAR1,
                            );
                            cp_async_bulk_tensor_2d_g2s(
                                &raw mut SMEM_B1 as *mut u8,
                                b_tma,
                                k_base,
                                n_offset,
                                &raw mut TMA_BAR1,
                            );
                            mbarrier_arrive_expect_tx(
                                &raw const TMA_BAR1,
                                1,
                                A_TILE_BYTES + B_TILE_BYTES,
                            );
                        }
                    }

                    k_idx += 1;
                }
            }

            // ── MMA Consumer (warp 5) ──
            if warp_id == MMA_WARP {
                let is_lane0 = lane_id == 0;
                let mut k_idx: u32 = 0;

                while k_idx < k_iters {
                    let stage = k_idx & 1;
                    let tma_parity = (k_idx >> 1) & 1;
                    let is_last = k_idx + 1 == k_iters;

                    // Wait for producer to fill this buffer
                    if stage == 0 {
                        while !mbarrier_try_wait_parity(&raw const TMA_BAR0, tma_parity) {}
                    } else {
                        while !mbarrier_try_wait_parity(&raw const TMA_BAR1, tma_parity) {}
                    }

                    // Issue 4 MMAs on this buffer
                    if is_lane0 {
                        let smem_a_base = if stage == 0 {
                            cvta_generic_to_shared_offset(&raw const SMEM_A0 as *const u8)
                        } else {
                            cvta_generic_to_shared_offset(&raw const SMEM_A1 as *const u8)
                        };
                        let smem_b_base = if stage == 0 {
                            cvta_generic_to_shared_offset(&raw const SMEM_B0 as *const u8)
                        } else {
                            cvta_generic_to_shared_offset(&raw const SMEM_B1 as *const u8)
                        };

                        let mut j: u32 = 0;
                        while j < 4 {
                            let byte_offset = (j * 32) as u64;
                            let a_desc = build_smem_descriptor(
                                smem_a_base + byte_offset,
                                LBO_BYTES,
                                SBO_BYTES,
                                SWIZZLE_128B,
                            );
                            let b_desc = build_smem_descriptor(
                                smem_b_base + byte_offset,
                                LBO_BYTES,
                                SBO_BYTES,
                                SWIZZLE_128B,
                            );

                            let accumulate = k_idx > 0 || j > 0;
                            tcgen05_mma_f16(tmem_addr, a_desc, b_desc, idesc, accumulate);
                            j += 1;
                        }

                        // Signal buffer release (or epilogue on last iteration).
                        // Last iter: commit to COMPUTE_BAR so epilogue warps know
                        // all MMA is done. Producer has already exited its loop.
                        if is_last {
                            tcgen05_commit_shared_cluster(&raw mut COMPUTE_BAR as *mut u64);
                        } else if stage == 0 {
                            tcgen05_commit_shared_cluster(&raw mut MMA_BAR0 as *mut u64);
                        } else {
                            tcgen05_commit_shared_cluster(&raw mut MMA_BAR1 as *mut u64);
                        }
                    }

                    k_idx += 1;
                }
            }

            // ── PHASE 2: Epilogue — TMEM → registers → SMEM (warps 0-3) ──
            // Warps 0-3 wait for COMPUTE_BAR, then read TMEM and write to SMEM_OUT.
            // Warps 4-5 skip the TMEM readback (their warp_id*32 would be out of bounds).
            if warp_id < 4 {
                while !mbarrier_try_wait_parity(&raw const COMPUTE_BAR, 0) {}

                const TILE_N: usize = 128;
                let warp_row_base = (warp_id * 32) as usize;
                let row_stride_bytes = TILE_N * 2;

                let row_within_8 = (lane_id % 8) as usize;
                let is_second_matrix = (8..16).contains(&lane_id);
                let col_offset_for_matrix2 = if is_second_matrix { 16usize } else { 0usize };

                let mut tmem_row_block = 0u32;
                while tmem_row_block < 2 {
                    let tmem_row = warp_id * 32 + tmem_row_block * 16;

                    let mut col_block = 0u32;
                    while col_block < 8 {
                        let col_offset = (col_block * 16) as usize;

                        let regs_a = tcgen05_ld_16x256b_pure(
                            tmem_addr + (tmem_row << 16) + col_offset as u32,
                        );
                        tcgen05_load_wait();

                        let regs_b = tcgen05_ld_16x256b_pure(
                            tmem_addr + (tmem_row << 16) + col_offset as u32 + 8,
                        );
                        tcgen05_load_wait();

                        let p0_lo = cvt_bf16x2_f32(regs_a[0], regs_a[1]);
                        let p1_lo = cvt_bf16x2_f32(regs_b[0], regs_b[1]);

                        let out_row_lo =
                            warp_row_base + (tmem_row_block as usize * 16) + row_within_8;
                        let smem_addr_lo = (&raw mut SMEM_OUT as *mut u8).add(
                            out_row_lo * row_stride_bytes + col_offset * 2 + col_offset_for_matrix2,
                        );
                        stmatrix_m8n8_x2(smem_addr_lo, p0_lo, p1_lo);

                        let p0_hi = cvt_bf16x2_f32(regs_a[2], regs_a[3]);
                        let p1_hi = cvt_bf16x2_f32(regs_b[2], regs_b[3]);

                        let out_row_hi =
                            warp_row_base + (tmem_row_block as usize * 16) + 8 + row_within_8;
                        let smem_addr_hi = (&raw mut SMEM_OUT as *mut u8).add(
                            out_row_hi * row_stride_bytes + col_offset * 2 + col_offset_for_matrix2,
                        );
                        stmatrix_m8n8_x2(smem_addr_hi, p0_hi, p1_hi);

                        col_block += 1;
                    }
                    tmem_row_block += 1;
                }
            }

            thread::sync_threads();

            // ── PHASE 3: Copy output tile to global memory (all 192 threads) ──
            let n_u32 = (n / 2) as usize;
            let tile_row_base = (tile_m * 128) as usize;
            let tile_col_base = (tile_n * 64) as usize;

            let mut local_idx = tid as usize;
            while local_idx < 8192 {
                let local_row = local_idx >> 6;
                let local_col = local_idx & 63;

                let global_row = tile_row_base + local_row;
                let global_col = tile_col_base + local_col;
                let global_idx = global_row * n_u32 + global_col;

                *out.get_unchecked_mut(global_idx) = SMEM_OUT[local_idx];
                local_idx += 192; // all 6 warps participate
            }

            // ── PHASE 4: Cleanup ──
            thread::sync_threads();
            if warp_id == 0 {
                tcgen05_dealloc(tmem_addr, 512);
            }
            if tid == 0 {
                mbarrier_inval(&raw mut TMA_BAR0);
                mbarrier_inval(&raw mut TMA_BAR1);
                mbarrier_inval(&raw mut MMA_BAR0);
                mbarrier_inval(&raw mut MMA_BAR1);
                mbarrier_inval(&raw mut COMPUTE_BAR);
            }
        }
    }

    /// Phase 4A: Persistent GEMM with TMEM accumulator pipeline.
    ///
    /// ```text
    ///   148 persistent CTAs (37 clusters × 4), each loops over many tiles:
    ///
    ///   Warp 4 (TMA)               Warp 5 (MMA)             Warps 0-3 (Epilogue)
    ///   ┌───────────────────┐      ┌───────────────────┐    ┌────────────────────┐
    ///   │ loop:             │      │ loop:             │    │ loop:              │
    ///   │  atomicAdd(ctr)   │      │  wait TILE_READY  │    │  wait TILE_READY   │
    ///   │  → tile_id        │      │  wait ACCUM_EMPTY │    │  wait ACCUM_FULL   │
    ///   │  write TILE_INFO  │      │                   │    │                    │
    ///   │  signal TILE_READY│─────▶│  K-loop:          │    │  drain TMEM stg N  │
    ///   │                   │      │   wait TMA_BAR[s] │    │  TMEM→bf16→GMEM    │
    ///   │  K-loop:          │      │   4× MMA → stg[N] │    │  signal ACCUM_EMPTY│
    ///   │   wait MMA_BAR[s] │◄─────│   signal MMA_BAR  │    │                    │
    ///   │   TMA A,B→buf[s]  │      │  signal ACCUM_FULL│───▶│  (next tile...)    │
    ///   │   signal TMA_BAR  │─────▶│  (next tile...)   │    │                    │
    ///   └───────────────────┘      └───────────────────┘    └────────────────────┘
    /// ```
    ///
    /// Key differences from Phase 3:
    /// - **Persistent kernel**: fixed CTA count (148 = all SMs × 4 per cluster),
    ///   each loops over many output tiles via a global atomic counter.
    /// - **TMEM accumulator pipeline**: 2 TMEM stages — MMA fills stage N while
    ///   epilogue drains stage N-1. ACCUM_FULL signals "results ready",
    ///   ACCUM_EMPTY signals "stage freed". Hides epilogue latency behind MMA.
    /// - **TILE_READY barrier**: TMA writes tile coordinates to TILE_INFO, then
    ///   signals TILE_READY so MMA and epilogue know which tile to process.
    /// - **global_k counter**: spans tile boundaries (never resets) so that
    ///   TMA_BAR/MMA_BAR parity tracking stays consistent across tiles.
    ///   k_idx resets per tile; global_k = sum of all k_idx across all tiles.
    ///
    /// ACCUM_EMPTY is initialized with arrival count = 128 (all 4 epilogue warps ×
    /// 32 threads). Every epilogue thread must arrive before MMA can reuse a stage.
    ///
    /// Grid launch: grid_dim = (num_CTAs, 1, 1), cluster_dim = (4, 1, 1)
    #[kernel]
    #[cluster_launch(4, 1, 1)]
    pub unsafe fn gemm_sol_persistent(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        mut out: DisjointSlice<u32>,
        tile_counter: *const u32,
        n: i32,
        k: i32,
        tiles_m: u32,
        tiles_n: u32,
    ) {
        unsafe {
            // ── SMEM layout (same as Phase 3) ──
            static mut SMEM_A0: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_A1: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_B0: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_B1: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_OUT: SharedArray<u32, 8192, 128> = SharedArray::UNINIT;
            static mut TMEM_ADDR: SharedArray<u32, 1, 4> = SharedArray::UNINIT;

            // TMA writes tile coords here; MMA and epilogue read them
            static mut TILE_INFO: SharedArray<u32, 4, 4> = SharedArray::UNINIT;

            // Barriers: TMA↔MMA (same as Phase 3)
            static mut TMA_BAR0: Barrier = Barrier::UNINIT;
            static mut TMA_BAR1: Barrier = Barrier::UNINIT;
            static mut MMA_BAR0: Barrier = Barrier::UNINIT;
            static mut MMA_BAR1: Barrier = Barrier::UNINIT;

            // Barriers: TMEM accumulator pipeline
            static mut ACCUM_FULL0: Barrier = Barrier::UNINIT;
            static mut ACCUM_FULL1: Barrier = Barrier::UNINIT;
            static mut ACCUM_EMPTY0: Barrier = Barrier::UNINIT;
            static mut ACCUM_EMPTY1: Barrier = Barrier::UNINIT;

            // TMA → MMA+epilogue: "tile coords are ready in TILE_INFO"
            static mut TILE_READY: Barrier = Barrier::UNINIT;
            // The producer must not overwrite TILE_INFO until every reader warp
            // has copied the current tile coordinates/has_work flag.
            static mut TILE_INFO_FREE: Barrier = Barrier::UNINIT;

            const A_TILE_BYTES: u32 = 128 * 64 * 2;
            const B_TILE_BYTES: u32 = 128 * 64 * 2;
            const SBO_BYTES: u32 = 1024;
            const LBO_BYTES: u32 = 16;
            const SWIZZLE_128B: u8 = 2;

            const TMA_WARP: u32 = 4;
            const MMA_WARP: u32 = 5;
            const NUM_ACCUM_STAGES: u32 = 2;
            const ACCUM_STAGE_COLS: u32 = 128;

            let n = n as u32;
            let k = k as u32;
            let tid = thread::threadIdx_x();
            let warp_id = warp::warp_id();
            let lane_id = tid % 32;

            // ── Initialize barriers + allocate TMEM ──
            // Barrier arrival counts:
            //   TMA/MMA/ACCUM_FULL/TILE_READY: 1 (single thread signals via commit or arrive)
            //   ACCUM_EMPTY: 128 (all 4 epilogue warps × 32 threads must arrive)
            if tid == 0 {
                mbarrier_init(&raw mut TMA_BAR0, 1);
                mbarrier_init(&raw mut TMA_BAR1, 1);
                mbarrier_init(&raw mut MMA_BAR0, 1);
                mbarrier_init(&raw mut MMA_BAR1, 1);
                mbarrier_init(&raw mut ACCUM_FULL0, 1);
                mbarrier_init(&raw mut ACCUM_FULL1, 1);
                mbarrier_init(&raw mut ACCUM_EMPTY0, 128);
                mbarrier_init(&raw mut ACCUM_EMPTY1, 128);
                mbarrier_init(&raw mut TILE_READY, 1);
                // One synchronized leader per reader warp acknowledges that all
                // lanes have copied the mailbox before the producer reuses it.
                mbarrier_init(&raw mut TILE_INFO_FREE, 5);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();

            // Pre-signal MMA_BAR0/1 so TMA producer can start loading the first tile
            // without waiting for MMA to release a buffer it never consumed.
            if tid == 0 {
                mbarrier_arrive(&raw const MMA_BAR0);
                mbarrier_arrive(&raw const MMA_BAR1);
            }

            thread::sync_threads();

            if warp_id == 0 {
                tcgen05_alloc(&raw mut TMEM_ADDR as *mut u32, 512);
            }
            thread::sync_threads();
            let tmem_addr = *(&raw const TMEM_ADDR as *const u32);

            let idesc = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape::M128_N128)
                .element_type(Tcgen05ElementType::F16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .build()
                .raw();

            let k_iters = k / 64;
            let total_tiles = tiles_m * tiles_n;

            cluster::cluster_sync();

            // ════════════════════════════════════════════════════════════════════
            // TMA Producer (warp 4): fetch tiles atomically, load A and B
            // ════════════════════════════════════════════════════════════════════
            if warp_id == TMA_WARP {
                let is_lane0 = lane_id == 0;
                let counter = &*(tile_counter as *const DeviceAtomicU32);

                // global_k counts K-iterations across ALL tiles (never resets).
                // This keeps TMA_BAR/MMA_BAR parity correct: stage = global_k & 1,
                // parity = (global_k >> 1) & 1. If we reset per tile, parity would
                // collide with the previous tile's last iteration.
                let mut global_k: u32 = 0;
                let mut tile_seq: u32 = 0;
                let mut tile_info_free_parity: u32 = 0;

                loop {
                    if is_lane0 {
                        let tile_id = counter.fetch_add(1, AtomicOrdering::Relaxed);
                        // tile_seq counts prior publishes: the first one has no
                        // readers to wait for, every later one does.
                        if tile_seq > 0 {
                            while !mbarrier_try_wait_parity(
                                &raw const TILE_INFO_FREE,
                                tile_info_free_parity,
                            ) {}
                            tile_info_free_parity ^= 1;
                        }
                        if tile_id < total_tiles {
                            *(&raw mut TILE_INFO as *mut u32).add(0) = tile_id / tiles_n;
                            *(&raw mut TILE_INFO as *mut u32).add(1) = tile_id % tiles_n;
                            *(&raw mut TILE_INFO as *mut u32).add(2) = 1;
                        } else {
                            *(&raw mut TILE_INFO as *mut u32).add(2) = 0;
                        }
                        mbarrier_arrive(&raw const TILE_READY);
                        tile_seq += 1;
                    }

                    // Lanes 1-31 read the mailbox lane 0 just wrote, and lane 0
                    // can stall above waiting for reader acks; without this sync
                    // they race ahead and can read the previous tile's has_work.
                    warp::sync_mask(u32::MAX);
                    let has_work = *(&raw const TILE_INFO as *const u32).add(2);
                    if has_work == 0 {
                        break;
                    }

                    let tile_m = *(&raw const TILE_INFO as *const u32).add(0);
                    let tile_n = *(&raw const TILE_INFO as *const u32).add(1);
                    let m_offset = (tile_m * 128) as i32;
                    let n_offset = (tile_n * 128) as i32;

                    let mut k_idx: u32 = 0;
                    while k_idx < k_iters {
                        let stage = global_k & 1;
                        let mma_parity = (global_k >> 1) & 1;

                        if stage == 0 {
                            while !mbarrier_try_wait_parity(&raw const MMA_BAR0, mma_parity) {}
                        } else {
                            while !mbarrier_try_wait_parity(&raw const MMA_BAR1, mma_parity) {}
                        }
                        // All lanes wait, then sync before lane 0 issues the
                        // loads that let MMA re-arm this barrier (same
                        // no-stranding rule as the MMA warp's K-loop).
                        warp::sync_mask(u32::MAX);

                        if is_lane0 {
                            let k_base = (k_idx * 64) as i32;
                            if stage == 0 {
                                cp_async_bulk_tensor_2d_g2s(
                                    &raw mut SMEM_A0 as *mut u8,
                                    a_tma,
                                    k_base,
                                    m_offset,
                                    &raw mut TMA_BAR0,
                                );
                                cp_async_bulk_tensor_2d_g2s(
                                    &raw mut SMEM_B0 as *mut u8,
                                    b_tma,
                                    k_base,
                                    n_offset,
                                    &raw mut TMA_BAR0,
                                );
                                mbarrier_arrive_expect_tx(
                                    &raw const TMA_BAR0,
                                    1,
                                    A_TILE_BYTES + B_TILE_BYTES,
                                );
                            } else {
                                cp_async_bulk_tensor_2d_g2s(
                                    &raw mut SMEM_A1 as *mut u8,
                                    a_tma,
                                    k_base,
                                    m_offset,
                                    &raw mut TMA_BAR1,
                                );
                                cp_async_bulk_tensor_2d_g2s(
                                    &raw mut SMEM_B1 as *mut u8,
                                    b_tma,
                                    k_base,
                                    n_offset,
                                    &raw mut TMA_BAR1,
                                );
                                mbarrier_arrive_expect_tx(
                                    &raw const TMA_BAR1,
                                    1,
                                    A_TILE_BYTES + B_TILE_BYTES,
                                );
                            }
                        }

                        k_idx += 1;
                        global_k += 1;
                    }
                }
            }

            // ════════════════════════════════════════════════════════════════════
            // MMA Consumer (warp 5): K-loop with 2-stage TMEM accumulator pipeline
            //
            // TMEM has 2 stages of 128 columns each. MMA writes to stage[tile_iter % 2]
            // while epilogue reads from the other stage. This hides epilogue latency
            // behind tensor core compute.
            //
            // Parity tracking for ACCUM_FULL/EMPTY:
            //   stage 0 fires on tiles 0, 2, 4, ... → parity = (tile_iter / 2) & 1
            //   stage 1 fires on tiles 1, 3, 5, ... → parity = (tile_iter / 2) & 1
            //   ACCUM_EMPTY waits start at tile_iter >= 2 (first 2 stages are free).
            // ════════════════════════════════════════════════════════════════════
            if warp_id == MMA_WARP {
                let is_lane0 = lane_id == 0;
                let mut tile_iter: u32 = 0;
                let mut tile_parity: u32 = 0;
                let mut global_k: u32 = 0;

                loop {
                    while !mbarrier_try_wait_parity(&raw const TILE_READY, tile_parity) {}
                    tile_parity ^= 1;

                    let has_work = *(&raw const TILE_INFO as *const u32).add(2);
                    warp::sync_mask(u32::MAX);
                    if is_lane0 {
                        mbarrier_arrive(&raw const TILE_INFO_FREE);
                    }
                    if has_work == 0 {
                        break;
                    }

                    let accum_stage = tile_iter % NUM_ACCUM_STAGES;
                    let tmem_stage_offset = accum_stage * ACCUM_STAGE_COLS;

                    // First 2 tiles fill fresh stages; after that, wait for epilogue to drain
                    // All lanes may wait here: the next ACCUM_EMPTY phase needs
                    // an ACCUM_FULL commit from a later tile, which sits behind
                    // this warp's tile-top sync, so the barrier cannot advance
                    // past a lane that is still waiting on it.
                    if tile_iter >= NUM_ACCUM_STAGES {
                        let empty_parity = ((tile_iter - NUM_ACCUM_STAGES) / NUM_ACCUM_STAGES) & 1;
                        if accum_stage == 0 {
                            while !mbarrier_try_wait_parity(&raw const ACCUM_EMPTY0, empty_parity) {
                            }
                        } else {
                            while !mbarrier_try_wait_parity(&raw const ACCUM_EMPTY1, empty_parity) {
                            }
                        }
                    }

                    // K-loop
                    let mut k_idx: u32 = 0;
                    while k_idx < k_iters {
                        let stage = global_k & 1;
                        let tma_parity = (global_k >> 1) & 1;

                        if stage == 0 {
                            while !mbarrier_try_wait_parity(&raw const TMA_BAR0, tma_parity) {}
                        } else {
                            while !mbarrier_try_wait_parity(&raw const TMA_BAR1, tma_parity) {}
                        }
                        // All lanes wait, then sync before lane 0 consumes the
                        // stage. The commit below is what lets the producer
                        // re-arm this barrier, so ordering it after the sync
                        // means the barrier can never advance past a lane that
                        // is still waiting on it -- no lane can strand one
                        // parity short of a stopped barrier.
                        warp::sync_mask(u32::MAX);

                        if is_lane0 {
                            let smem_a_base = if stage == 0 {
                                cvta_generic_to_shared_offset(&raw const SMEM_A0 as *const u8)
                            } else {
                                cvta_generic_to_shared_offset(&raw const SMEM_A1 as *const u8)
                            };
                            let smem_b_base = if stage == 0 {
                                cvta_generic_to_shared_offset(&raw const SMEM_B0 as *const u8)
                            } else {
                                cvta_generic_to_shared_offset(&raw const SMEM_B1 as *const u8)
                            };

                            let mut j: u32 = 0;
                            while j < 4 {
                                let byte_offset = (j * 32) as u64;
                                let a_desc = build_smem_descriptor(
                                    smem_a_base + byte_offset,
                                    LBO_BYTES,
                                    SBO_BYTES,
                                    SWIZZLE_128B,
                                );
                                let b_desc = build_smem_descriptor(
                                    smem_b_base + byte_offset,
                                    LBO_BYTES,
                                    SBO_BYTES,
                                    SWIZZLE_128B,
                                );

                                let accumulate = k_idx > 0 || j > 0;
                                tcgen05_mma_f16(
                                    tmem_addr + tmem_stage_offset,
                                    a_desc,
                                    b_desc,
                                    idesc,
                                    accumulate,
                                );
                                j += 1;
                            }

                            if stage == 0 {
                                tcgen05_commit_shared_cluster(&raw mut MMA_BAR0 as *mut u64);
                            } else {
                                tcgen05_commit_shared_cluster(&raw mut MMA_BAR1 as *mut u64);
                            }
                        }

                        k_idx += 1;
                        global_k += 1;
                    }

                    // Signal that this TMEM stage has results
                    if is_lane0 {
                        if accum_stage == 0 {
                            tcgen05_commit_shared_cluster(&raw mut ACCUM_FULL0 as *mut u64);
                        } else {
                            tcgen05_commit_shared_cluster(&raw mut ACCUM_FULL1 as *mut u64);
                        }
                    }

                    tile_iter += 1;
                }
            }

            // ════════════════════════════════════════════════════════════════════
            // Epilogue warps (0-3): drain TMEM stages, write to GMEM
            // No sync_threads inside — each warp writes its own 32 rows independently.
            // All 128 threads arrive on ACCUM_EMPTY to signal MMA that TMEM is free.
            // ════════════════════════════════════════════════════════════════════
            if warp_id < 4 {
                let is_lane0 = lane_id == 0;
                let mut epi_tile_iter: u32 = 0;
                let mut tile_parity: u32 = 0;

                const TILE_N: usize = 128;
                let warp_row_base = (warp_id * 32) as usize;
                let row_stride_bytes = TILE_N * 2;
                let row_within_8 = (lane_id % 8) as usize;
                let is_second_matrix = (8..16).contains(&lane_id);
                let col_offset_for_matrix2 = if is_second_matrix { 16usize } else { 0usize };

                loop {
                    while !mbarrier_try_wait_parity(&raw const TILE_READY, tile_parity) {}
                    tile_parity ^= 1;

                    let has_work = *(&raw const TILE_INFO as *const u32).add(2);
                    // Save tile coords to local regs (TILE_INFO may be overwritten by TMA later)
                    let tile_m = *(&raw const TILE_INFO as *const u32).add(0);
                    let tile_n = *(&raw const TILE_INFO as *const u32).add(1);
                    warp::sync_mask(u32::MAX);
                    if is_lane0 {
                        mbarrier_arrive(&raw const TILE_INFO_FREE);
                    }
                    if has_work == 0 {
                        break;
                    }

                    let accum_stage = epi_tile_iter % NUM_ACCUM_STAGES;
                    let tmem_stage_offset = accum_stage * ACCUM_STAGE_COLS;

                    // Wait for MMA to finish filling this TMEM stage
                    let full_parity = (epi_tile_iter / NUM_ACCUM_STAGES) & 1;
                    if accum_stage == 0 {
                        while !mbarrier_try_wait_parity(&raw const ACCUM_FULL0, full_parity) {}
                    } else {
                        while !mbarrier_try_wait_parity(&raw const ACCUM_FULL1, full_parity) {}
                    }

                    // Drain TMEM stage → SMEM_OUT (each warp handles its own 32 rows)
                    let mut tmem_row_block = 0u32;
                    while tmem_row_block < 2 {
                        let tmem_row = warp_id * 32 + tmem_row_block * 16;

                        let mut col_block = 0u32;
                        while col_block < 8 {
                            let col_offset = (col_block * 16) as usize;

                            let regs_a = tcgen05_ld_16x256b_pure(
                                tmem_addr
                                    + tmem_stage_offset
                                    + (tmem_row << 16)
                                    + col_offset as u32,
                            );
                            tcgen05_load_wait();

                            let regs_b = tcgen05_ld_16x256b_pure(
                                tmem_addr
                                    + tmem_stage_offset
                                    + (tmem_row << 16)
                                    + col_offset as u32
                                    + 8,
                            );
                            tcgen05_load_wait();

                            let p0_lo = cvt_bf16x2_f32(regs_a[0], regs_a[1]);
                            let p1_lo = cvt_bf16x2_f32(regs_b[0], regs_b[1]);
                            let out_row_lo =
                                warp_row_base + (tmem_row_block as usize * 16) + row_within_8;
                            let smem_addr_lo = (&raw mut SMEM_OUT as *mut u8).add(
                                out_row_lo * row_stride_bytes
                                    + col_offset * 2
                                    + col_offset_for_matrix2,
                            );
                            stmatrix_m8n8_x2(smem_addr_lo, p0_lo, p1_lo);

                            let p0_hi = cvt_bf16x2_f32(regs_a[2], regs_a[3]);
                            let p1_hi = cvt_bf16x2_f32(regs_b[2], regs_b[3]);
                            let out_row_hi =
                                warp_row_base + (tmem_row_block as usize * 16) + 8 + row_within_8;
                            let smem_addr_hi = (&raw mut SMEM_OUT as *mut u8).add(
                                out_row_hi * row_stride_bytes
                                    + col_offset * 2
                                    + col_offset_for_matrix2,
                            );
                            stmatrix_m8n8_x2(smem_addr_hi, p0_hi, p1_hi);

                            col_block += 1;
                        }
                        tmem_row_block += 1;
                    }

                    // Write SMEM_OUT → GMEM (each warp writes only its own 32 rows)
                    let n_u32 = (n / 2) as usize;
                    let tile_row_base = (tile_m * 128) as usize;
                    let tile_col_base = (tile_n * 64) as usize;
                    let base_row = warp_id as usize * 32;

                    let mut elem = lane_id as usize;
                    while elem < 2048 {
                        let local_row = elem / 64;
                        let local_col = elem % 64;
                        let smem_idx = (base_row + local_row) * 64 + local_col;
                        let global_row = tile_row_base + base_row + local_row;
                        let global_col = tile_col_base + local_col;
                        let global_idx = global_row * n_u32 + global_col;

                        *out.get_unchecked_mut(global_idx) = SMEM_OUT[smem_idx];
                        elem += 32;
                    }

                    // Signal MMA that this TMEM stage is free (all 128 epilogue threads arrive)
                    if accum_stage == 0 {
                        mbarrier_arrive(&raw const ACCUM_EMPTY0);
                    } else {
                        mbarrier_arrive(&raw const ACCUM_EMPTY1);
                    }

                    epi_tile_iter += 1;
                }
            }

            // ── Cleanup: all warps converge here after their loops exit ──
            thread::sync_threads();
            if warp_id == 0 {
                tcgen05_dealloc(tmem_addr, 512);
            }
            if tid == 0 {
                mbarrier_inval(&raw mut TMA_BAR0);
                mbarrier_inval(&raw mut TMA_BAR1);
                mbarrier_inval(&raw mut MMA_BAR0);
                mbarrier_inval(&raw mut MMA_BAR1);
                mbarrier_inval(&raw mut ACCUM_FULL0);
                mbarrier_inval(&raw mut ACCUM_FULL1);
                mbarrier_inval(&raw mut ACCUM_EMPTY0);
                mbarrier_inval(&raw mut ACCUM_EMPTY1);
                mbarrier_inval(&raw mut TILE_READY);
                mbarrier_inval(&raw mut TILE_INFO_FREE);
            }
        }
    }

    /// Phase 4B: CLC tile scheduling + TMEM accumulator pipeline (no multicast).
    ///
    /// ```text
    ///   Full grid launch: 1 CTA per tile, cluster_dim=(4,1,1)
    ///   Hardware launches clusters of 4 CTAs onto SMs.
    ///   CTAs that finish steal more work from the pending queue.
    ///
    ///   CTA (any rank)                  Hardware Pending Queue
    ///   ┌───────────────────────┐      ┌─────────────────────┐
    ///   │ 1. Process own tile   │      │ [CTA4..CTA7]        │
    ///   │    (blockIdx.x)       │      │ [CTA8..CTA11]       │
    ///   │                       │      │ [CTA12..CTA15]      │
    ///   │ 2. CLC work-stealing: │      │ ...                 │
    ///   │    lane 0: arm + issue│      │                     │
    ///   │    clc_try_cancel ────┼─────▶│ steal [CTA4..CTA7]  │
    ///   │    lane 0: wait/decode│      │ (removed from queue)│
    ///   │    warp: shuffle result│     │                     │
    ///   │                       │      └─────────────────────┘
    ///   │ 3. Process all 4 tiles│
    ///   │    from stolen cluster│      Each CTA independently steals
    ///   │    serially:          │      and processes CLUSTER_SIZE tiles.
    ///   │    ci=0: tile 4       │      No coordination between CTAs
    ///   │    ci=1: tile 5       │      in the same cluster.
    ///   │    ci=2: tile 6       │
    ///   │    ci=3: tile 7       │
    ///   │                       │
    ///   │ 4. Repeat until       │
    ///   │    is_canceled = 0    │
    ///   └───────────────────────┘
    /// ```
    ///
    /// Changes from Phase 4A (`gemm_sol_persistent`):
    /// - **CLC replaces atomic counter**: full grid launch (1 CTA per tile), running CTAs
    ///   steal pending work via `clc_try_cancel` instead of `atomicAdd` on a global counter.
    /// - **Cluster-aware stealing**: `clc_try_cancel` returns the first ctaid of a stolen
    ///   cluster. Each CTA serially processes all `CLUSTER_SIZE` tiles from that cluster.
    /// - **Single-lane response ownership**: lane 0 waits for and decodes the async CLC
    ///   response, then broadcasts the result to its warp so every lane takes the same path.
    /// - **Column-major tile rasterization**: linear ctaid maps to (row, col) for L2 locality.
    /// - **No `tile_counter` parameter**: hardware manages the pending queue, zero contention.
    ///
    /// Each CTA loads its own A and B tiles via unicast TMA (no multicast).
    ///
    /// Grid launch: grid_dim = (total_tiles, 1, 1), cluster_dim = (4, 1, 1)
    #[kernel]
    #[cluster_launch(4, 1, 1)]
    pub unsafe fn gemm_sol_clc(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        mut out: DisjointSlice<u32>,
        n: i32,
        k: i32,
        tiles_m: u32,
        _tiles_n: u32,
    ) {
        unsafe {
            // ── SMEM layout (same as Phase 4A) ──
            static mut SMEM_A0: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_A1: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_B0: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_B1: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_OUT: SharedArray<u32, 8192, 128> = SharedArray::UNINIT;
            static mut TMEM_ADDR: SharedArray<u32, 1, 4> = SharedArray::UNINIT;

            static mut TILE_INFO: SharedArray<u32, 4, 4> = SharedArray::UNINIT;

            // Barriers: TMA↔MMA double-buffered
            static mut TMA_BAR0: Barrier = Barrier::UNINIT;
            static mut TMA_BAR1: Barrier = Barrier::UNINIT;
            static mut MMA_BAR0: Barrier = Barrier::UNINIT;
            static mut MMA_BAR1: Barrier = Barrier::UNINIT;

            // Barriers: TMEM accumulator pipeline
            static mut ACCUM_FULL0: Barrier = Barrier::UNINIT;
            static mut ACCUM_FULL1: Barrier = Barrier::UNINIT;
            static mut ACCUM_EMPTY0: Barrier = Barrier::UNINIT;
            static mut ACCUM_EMPTY1: Barrier = Barrier::UNINIT;

            static mut TILE_READY: Barrier = Barrier::UNINIT;
            // The producer must not overwrite TILE_INFO until every reader warp
            // has copied the current tile coordinates/has_work flag.
            static mut TILE_INFO_FREE: Barrier = Barrier::UNINIT;

            // CLC: 16-byte response buffer + mbarrier
            static mut CLC_RESPONSE: SharedArray<u64, 2, 16> = SharedArray::UNINIT;
            static mut CLC_BAR: Barrier = Barrier::UNINIT;

            const A_TILE_BYTES: u32 = 128 * 64 * 2;
            const B_TILE_BYTES: u32 = 128 * 64 * 2;
            const SBO_BYTES: u32 = 1024;
            const LBO_BYTES: u32 = 16;
            const SWIZZLE_128B: u8 = 2;

            const TMA_WARP: u32 = 4;
            const MMA_WARP: u32 = 5;
            const NUM_ACCUM_STAGES: u32 = 2;
            const ACCUM_STAGE_COLS: u32 = 128;
            const CLUSTER_SIZE: u32 = 4;

            let n = n as u32;
            let k = k as u32;
            let tid = thread::threadIdx_x();
            let warp_id = warp::warp_id();
            let lane_id = tid % 32;

            // ── Initialize barriers + allocate TMEM ──
            if tid == 0 {
                mbarrier_init(&raw mut TMA_BAR0, 1);
                mbarrier_init(&raw mut TMA_BAR1, 1);
                mbarrier_init(&raw mut MMA_BAR0, 1);
                mbarrier_init(&raw mut MMA_BAR1, 1);
                mbarrier_init(&raw mut ACCUM_FULL0, 1);
                mbarrier_init(&raw mut ACCUM_FULL1, 1);
                mbarrier_init(&raw mut ACCUM_EMPTY0, 128);
                mbarrier_init(&raw mut ACCUM_EMPTY1, 128);
                mbarrier_init(&raw mut TILE_READY, 1);
                // One synchronized leader per reader warp acknowledges that all
                // lanes have copied the mailbox before the producer reuses it.
                mbarrier_init(&raw mut TILE_INFO_FREE, 5);
                mbarrier_init(&raw mut CLC_BAR, 1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();

            // Pre-arrive MMA_BARs so TMA can proceed on the first K-iteration
            if tid == 0 {
                mbarrier_arrive(&raw const MMA_BAR0);
                mbarrier_arrive(&raw const MMA_BAR1);
            }

            thread::sync_threads();

            if warp_id == 0 {
                tcgen05_alloc(&raw mut TMEM_ADDR as *mut u32, 512);
            }
            thread::sync_threads();
            let tmem_addr = *(&raw const TMEM_ADDR as *const u32);

            let idesc = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape::M128_N128)
                .element_type(Tcgen05ElementType::F16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .build()
                .raw();

            let k_iters = k / 64;

            cluster::cluster_sync();

            // ════════════════════════════════════════════════════════════════════
            // TMA Producer (warp 4): CLC tile scheduling + per-CTA TMA loads
            // ════════════════════════════════════════════════════════════════════
            if warp_id == TMA_WARP {
                let is_lane0 = lane_id == 0;
                let mut global_k: u32 = 0;
                let mut clc_iter: u32 = 0;
                let mut tile_info_free_parity: u32 = 0;

                // ── First tile: use our own blockIdx (hardware-assigned) ──
                let first_ctaid = thread::blockIdx_x();
                let first_tile_m = first_ctaid % tiles_m;
                let first_tile_n = first_ctaid / tiles_m;

                if is_lane0 {
                    *(&raw mut TILE_INFO as *mut u32).add(0) = first_tile_m;
                    *(&raw mut TILE_INFO as *mut u32).add(1) = first_tile_n;
                    *(&raw mut TILE_INFO as *mut u32).add(2) = 1;
                    mbarrier_arrive(&raw const TILE_READY);
                }

                let m_offset = (first_tile_m * 128) as i32;
                let n_offset = (first_tile_n * 128) as i32;

                let mut k_idx: u32 = 0;
                while k_idx < k_iters {
                    let stage = global_k & 1;
                    let mma_parity = (global_k >> 1) & 1;

                    if stage == 0 {
                        while !mbarrier_try_wait_parity(&raw const MMA_BAR0, mma_parity) {}
                    } else {
                        while !mbarrier_try_wait_parity(&raw const MMA_BAR1, mma_parity) {}
                    }
                    // All lanes wait, then sync before lane 0 issues the loads
                    // that let MMA re-arm this barrier (no-stranding rule).
                    warp::sync_mask(u32::MAX);

                    if is_lane0 {
                        let k_base = (k_idx * 64) as i32;
                        if stage == 0 {
                            mbarrier_arrive_expect_tx(
                                &raw const TMA_BAR0,
                                1,
                                A_TILE_BYTES + B_TILE_BYTES,
                            );
                            cp_async_bulk_tensor_2d_g2s(
                                &raw mut SMEM_A0 as *mut u8,
                                a_tma,
                                k_base,
                                m_offset,
                                &raw mut TMA_BAR0,
                            );
                            cp_async_bulk_tensor_2d_g2s(
                                &raw mut SMEM_B0 as *mut u8,
                                b_tma,
                                k_base,
                                n_offset,
                                &raw mut TMA_BAR0,
                            );
                        } else {
                            mbarrier_arrive_expect_tx(
                                &raw const TMA_BAR1,
                                1,
                                A_TILE_BYTES + B_TILE_BYTES,
                            );
                            cp_async_bulk_tensor_2d_g2s(
                                &raw mut SMEM_A1 as *mut u8,
                                a_tma,
                                k_base,
                                m_offset,
                                &raw mut TMA_BAR1,
                            );
                            cp_async_bulk_tensor_2d_g2s(
                                &raw mut SMEM_B1 as *mut u8,
                                b_tma,
                                k_base,
                                n_offset,
                                &raw mut TMA_BAR1,
                            );
                        }
                    }

                    k_idx += 1;
                    global_k += 1;
                }

                // ── Subsequent tiles: CLC work-stealing loop ──
                // Each CTA independently steals a cluster from the pending queue.
                // clc_try_cancel returns the first ctaid of the stolen cluster;
                // this CTA then serially processes all CLUSTER_SIZE tiles.
                let resp_ptr = &raw mut CLC_RESPONSE as *mut u64;

                loop {
                    let clc_parity = clc_iter & 1;

                    // CLC writes the response through the async proxy. Keep all
                    // response access in lane 0, then broadcast registers to the
                    // warp so no lane can observe stale shared memory.
                    let mut is_canceled = 0u32;
                    let mut first_stolen = 0u32;
                    if is_lane0 {
                        fence_proxy_async_shared_cta();
                        mbarrier_arrive_expect_tx(&raw const CLC_BAR, 1, 16);
                        clc_try_cancel(resp_ptr as *mut u8, &raw mut CLC_BAR);

                        while !mbarrier_try_wait_parity(&raw const CLC_BAR, clc_parity) {}

                        let resp_lo = *resp_ptr;
                        let resp_hi = *resp_ptr.add(1);
                        is_canceled = clc_query_is_canceled(resp_lo, resp_hi);
                        if is_canceled != 0 {
                            first_stolen = clc_query_get_first_ctaid_x(resp_lo, resp_hi);
                        }
                        fence_proxy_async_shared_cta();
                    }

                    is_canceled = warp::shuffle_sync(u32::MAX, is_canceled, 0);
                    first_stolen = warp::shuffle_sync(u32::MAX, first_stolen, 0);

                    if is_canceled == 0 {
                        if is_lane0 {
                            while !mbarrier_try_wait_parity(
                                &raw const TILE_INFO_FREE,
                                tile_info_free_parity,
                            ) {}
                            *(&raw mut TILE_INFO as *mut u32).add(2) = 0;
                            mbarrier_arrive(&raw const TILE_READY);
                        }
                        break;
                    }

                    let mut ci: u32 = 0;
                    while ci < CLUSTER_SIZE {
                        let stolen_ctaid = first_stolen + ci;
                        let tile_m = stolen_ctaid % tiles_m;
                        let tile_n = stolen_ctaid / tiles_m;

                        if is_lane0 {
                            while !mbarrier_try_wait_parity(
                                &raw const TILE_INFO_FREE,
                                tile_info_free_parity,
                            ) {}
                            tile_info_free_parity ^= 1;
                            *(&raw mut TILE_INFO as *mut u32).add(0) = tile_m;
                            *(&raw mut TILE_INFO as *mut u32).add(1) = tile_n;
                            *(&raw mut TILE_INFO as *mut u32).add(2) = 1;
                            mbarrier_arrive(&raw const TILE_READY);
                        }

                        let m_off = (tile_m * 128) as i32;
                        let n_off = (tile_n * 128) as i32;

                        let mut k_idx: u32 = 0;
                        while k_idx < k_iters {
                            let stage = global_k & 1;
                            let mma_parity = (global_k >> 1) & 1;

                            if stage == 0 {
                                while !mbarrier_try_wait_parity(&raw const MMA_BAR0, mma_parity) {}
                            } else {
                                while !mbarrier_try_wait_parity(&raw const MMA_BAR1, mma_parity) {}
                            }
                            // All lanes wait, then sync before lane 0 issues the
                            // loads that let MMA re-arm this barrier
                            // (no-stranding rule).
                            warp::sync_mask(u32::MAX);

                            if is_lane0 {
                                let k_base = (k_idx * 64) as i32;
                                if stage == 0 {
                                    mbarrier_arrive_expect_tx(
                                        &raw const TMA_BAR0,
                                        1,
                                        A_TILE_BYTES + B_TILE_BYTES,
                                    );
                                    cp_async_bulk_tensor_2d_g2s(
                                        &raw mut SMEM_A0 as *mut u8,
                                        a_tma,
                                        k_base,
                                        m_off,
                                        &raw mut TMA_BAR0,
                                    );
                                    cp_async_bulk_tensor_2d_g2s(
                                        &raw mut SMEM_B0 as *mut u8,
                                        b_tma,
                                        k_base,
                                        n_off,
                                        &raw mut TMA_BAR0,
                                    );
                                } else {
                                    mbarrier_arrive_expect_tx(
                                        &raw const TMA_BAR1,
                                        1,
                                        A_TILE_BYTES + B_TILE_BYTES,
                                    );
                                    cp_async_bulk_tensor_2d_g2s(
                                        &raw mut SMEM_A1 as *mut u8,
                                        a_tma,
                                        k_base,
                                        m_off,
                                        &raw mut TMA_BAR1,
                                    );
                                    cp_async_bulk_tensor_2d_g2s(
                                        &raw mut SMEM_B1 as *mut u8,
                                        b_tma,
                                        k_base,
                                        n_off,
                                        &raw mut TMA_BAR1,
                                    );
                                }
                            }

                            k_idx += 1;
                            global_k += 1;
                        }

                        ci += 1;
                    }

                    clc_iter += 1;
                }
            }

            // ════════════════════════════════════════════════════════════════════
            // MMA Consumer (warp 5): identical to Phase 4A
            // ════════════════════════════════════════════════════════════════════
            if warp_id == MMA_WARP {
                let is_lane0 = lane_id == 0;
                let mut tile_iter: u32 = 0;
                let mut tile_parity: u32 = 0;
                let mut global_k: u32 = 0;

                loop {
                    while !mbarrier_try_wait_parity(&raw const TILE_READY, tile_parity) {}
                    tile_parity ^= 1;

                    let has_work = *(&raw const TILE_INFO as *const u32).add(2);
                    warp::sync_mask(u32::MAX);
                    if is_lane0 {
                        mbarrier_arrive(&raw const TILE_INFO_FREE);
                    }
                    if has_work == 0 {
                        break;
                    }

                    let accum_stage = tile_iter % NUM_ACCUM_STAGES;
                    let tmem_stage_offset = accum_stage * ACCUM_STAGE_COLS;

                    // All lanes may wait here: the next ACCUM_EMPTY phase needs
                    // an ACCUM_FULL commit from a later tile, which sits behind
                    // this warp's tile-top sync, so the barrier cannot advance
                    // past a lane that is still waiting on it.
                    if tile_iter >= NUM_ACCUM_STAGES {
                        let empty_parity = ((tile_iter - NUM_ACCUM_STAGES) / NUM_ACCUM_STAGES) & 1;
                        if accum_stage == 0 {
                            while !mbarrier_try_wait_parity(&raw const ACCUM_EMPTY0, empty_parity) {
                            }
                        } else {
                            while !mbarrier_try_wait_parity(&raw const ACCUM_EMPTY1, empty_parity) {
                            }
                        }
                    }

                    let mut k_idx: u32 = 0;
                    while k_idx < k_iters {
                        let stage = global_k & 1;
                        let tma_parity = (global_k >> 1) & 1;

                        if stage == 0 {
                            while !mbarrier_try_wait_parity(&raw const TMA_BAR0, tma_parity) {}
                        } else {
                            while !mbarrier_try_wait_parity(&raw const TMA_BAR1, tma_parity) {}
                        }
                        // All lanes wait, then sync before lane 0 consumes the
                        // stage. The commit below is what lets the producer
                        // re-arm this barrier, so ordering it after the sync
                        // means the barrier can never advance past a lane that
                        // is still waiting on it -- no lane can strand one
                        // parity short of a stopped barrier.
                        warp::sync_mask(u32::MAX);

                        if is_lane0 {
                            let smem_a_base = if stage == 0 {
                                cvta_generic_to_shared_offset(&raw const SMEM_A0 as *const u8)
                            } else {
                                cvta_generic_to_shared_offset(&raw const SMEM_A1 as *const u8)
                            };
                            let smem_b_base = if stage == 0 {
                                cvta_generic_to_shared_offset(&raw const SMEM_B0 as *const u8)
                            } else {
                                cvta_generic_to_shared_offset(&raw const SMEM_B1 as *const u8)
                            };

                            let mut j: u32 = 0;
                            while j < 4 {
                                let byte_offset = (j * 32) as u64;
                                let a_desc = build_smem_descriptor(
                                    smem_a_base + byte_offset,
                                    LBO_BYTES,
                                    SBO_BYTES,
                                    SWIZZLE_128B,
                                );
                                let b_desc = build_smem_descriptor(
                                    smem_b_base + byte_offset,
                                    LBO_BYTES,
                                    SBO_BYTES,
                                    SWIZZLE_128B,
                                );

                                let accumulate = k_idx > 0 || j > 0;
                                tcgen05_mma_f16(
                                    tmem_addr + tmem_stage_offset,
                                    a_desc,
                                    b_desc,
                                    idesc,
                                    accumulate,
                                );
                                j += 1;
                            }

                            if stage == 0 {
                                tcgen05_commit_shared_cluster(&raw mut MMA_BAR0 as *mut u64);
                            } else {
                                tcgen05_commit_shared_cluster(&raw mut MMA_BAR1 as *mut u64);
                            }
                        }

                        k_idx += 1;
                        global_k += 1;
                    }

                    if is_lane0 {
                        if accum_stage == 0 {
                            tcgen05_commit_shared_cluster(&raw mut ACCUM_FULL0 as *mut u64);
                        } else {
                            tcgen05_commit_shared_cluster(&raw mut ACCUM_FULL1 as *mut u64);
                        }
                    }

                    tile_iter += 1;
                }
            }

            // ════════════════════════════════════════════════════════════════════
            // Epilogue warps (0-3): identical to Phase 4A
            // ════════════════════════════════════════════════════════════════════
            if warp_id < 4 {
                let is_lane0 = lane_id == 0;
                let mut epi_tile_iter: u32 = 0;
                let mut tile_parity: u32 = 0;

                const TILE_N: usize = 128;
                let warp_row_base = (warp_id * 32) as usize;
                let row_stride_bytes = TILE_N * 2;
                let row_within_8 = (lane_id % 8) as usize;
                let is_second_matrix = (8..16).contains(&lane_id);
                let col_offset_for_matrix2 = if is_second_matrix { 16usize } else { 0usize };

                loop {
                    while !mbarrier_try_wait_parity(&raw const TILE_READY, tile_parity) {}
                    tile_parity ^= 1;

                    let has_work = *(&raw const TILE_INFO as *const u32).add(2);
                    let tile_m = *(&raw const TILE_INFO as *const u32).add(0);
                    let tile_n = *(&raw const TILE_INFO as *const u32).add(1);
                    warp::sync_mask(u32::MAX);
                    if is_lane0 {
                        mbarrier_arrive(&raw const TILE_INFO_FREE);
                    }
                    if has_work == 0 {
                        break;
                    }

                    let accum_stage = epi_tile_iter % NUM_ACCUM_STAGES;
                    let tmem_stage_offset = accum_stage * ACCUM_STAGE_COLS;

                    let full_parity = (epi_tile_iter / NUM_ACCUM_STAGES) & 1;
                    if accum_stage == 0 {
                        while !mbarrier_try_wait_parity(&raw const ACCUM_FULL0, full_parity) {}
                    } else {
                        while !mbarrier_try_wait_parity(&raw const ACCUM_FULL1, full_parity) {}
                    }

                    let mut tmem_row_block = 0u32;
                    while tmem_row_block < 2 {
                        let tmem_row = warp_id * 32 + tmem_row_block * 16;

                        let mut col_block = 0u32;
                        while col_block < 8 {
                            let col_offset = (col_block * 16) as usize;

                            let regs_a = tcgen05_ld_16x256b_pure(
                                tmem_addr
                                    + tmem_stage_offset
                                    + (tmem_row << 16)
                                    + col_offset as u32,
                            );
                            tcgen05_load_wait();

                            let regs_b = tcgen05_ld_16x256b_pure(
                                tmem_addr
                                    + tmem_stage_offset
                                    + (tmem_row << 16)
                                    + col_offset as u32
                                    + 8,
                            );
                            tcgen05_load_wait();

                            let p0_lo = cvt_bf16x2_f32(regs_a[0], regs_a[1]);
                            let p1_lo = cvt_bf16x2_f32(regs_b[0], regs_b[1]);
                            let out_row_lo =
                                warp_row_base + (tmem_row_block as usize * 16) + row_within_8;
                            let smem_addr_lo = (&raw mut SMEM_OUT as *mut u8).add(
                                out_row_lo * row_stride_bytes
                                    + col_offset * 2
                                    + col_offset_for_matrix2,
                            );
                            stmatrix_m8n8_x2(smem_addr_lo, p0_lo, p1_lo);

                            let p0_hi = cvt_bf16x2_f32(regs_a[2], regs_a[3]);
                            let p1_hi = cvt_bf16x2_f32(regs_b[2], regs_b[3]);
                            let out_row_hi =
                                warp_row_base + (tmem_row_block as usize * 16) + 8 + row_within_8;
                            let smem_addr_hi = (&raw mut SMEM_OUT as *mut u8).add(
                                out_row_hi * row_stride_bytes
                                    + col_offset * 2
                                    + col_offset_for_matrix2,
                            );
                            stmatrix_m8n8_x2(smem_addr_hi, p0_hi, p1_hi);

                            col_block += 1;
                        }
                        tmem_row_block += 1;
                    }

                    let n_u32 = (n / 2) as usize;
                    let tile_row_base = (tile_m * 128) as usize;
                    let tile_col_base = (tile_n * 64) as usize;
                    let base_row = warp_id as usize * 32;

                    let mut elem = lane_id as usize;
                    while elem < 2048 {
                        let local_row = elem / 64;
                        let local_col = elem % 64;
                        let smem_idx = (base_row + local_row) * 64 + local_col;
                        let global_row = tile_row_base + base_row + local_row;
                        let global_col = tile_col_base + local_col;
                        let global_idx = global_row * n_u32 + global_col;

                        *out.get_unchecked_mut(global_idx) = SMEM_OUT[smem_idx];
                        elem += 32;
                    }

                    if accum_stage == 0 {
                        mbarrier_arrive(&raw const ACCUM_EMPTY0);
                    } else {
                        mbarrier_arrive(&raw const ACCUM_EMPTY1);
                    }

                    epi_tile_iter += 1;
                }
            }

            // ── Cleanup ──
            thread::sync_threads();
            cluster::cluster_sync();
            if warp_id == 0 {
                tcgen05_dealloc(tmem_addr, 512);
            }
            if tid == 0 {
                mbarrier_inval(&raw mut TMA_BAR0);
                mbarrier_inval(&raw mut TMA_BAR1);
                mbarrier_inval(&raw mut MMA_BAR0);
                mbarrier_inval(&raw mut MMA_BAR1);
                mbarrier_inval(&raw mut ACCUM_FULL0);
                mbarrier_inval(&raw mut ACCUM_FULL1);
                mbarrier_inval(&raw mut ACCUM_EMPTY0);
                mbarrier_inval(&raw mut ACCUM_EMPTY1);
                mbarrier_inval(&raw mut TILE_READY);
                mbarrier_inval(&raw mut TILE_INFO_FREE);
                mbarrier_inval(&raw mut CLC_BAR);
            }
        }
    }

    /// Phase 4C: CLC + TMA multicast for B tiles.
    ///
    /// ```text
    ///   Cluster of 4 CTAs co-scheduled across SMs:
    ///
    ///   Every CTA (warp 4)                    CTA 0 / rank 0
    ///   ┌──────────────────────────┐          ┌──────────────────────────┐
    ///   │ wait MMA_BAR            │          │                          │
    ///   │ arm local TMA_BAR       │          │                          │
    ///   │ TMA A → own SMEM        │          │                          │
    ///   │ arrive rank-0 MCAST_BAR ├─────────▶│ cluster-acquire wait     │
    ///   │                          │          │ for all four CTAs        │
    ///   │                          │          │ TMA B multicast ────────┼──▶ all CTA SMEM
    ///   └──────────────────────────┘          └──────────────────────────┘
    ///
    ///   CLC work-stealing (clc_try_cancel_multicast):
    ///     Every CTA arms its local CLC_BAR and arrives at CLC_READY
    ///     Rank 0 waits for all CTAs, then multicasts one cancellation response
    ///     Lane 0 in each CTA decodes and warp-broadcasts the result
    ///     Each CTA derives: my_tile = first_stolen + my_rank
    /// ```
    ///
    /// Builds on Phase 4B by multicasting B tiles from rank 0 to all cluster CTAs via
    /// `cp_async_bulk_tensor_2d_g2s_multicast`. Each CTA still loads its own A tile
    /// (row-specific), but B tiles (shared column) are loaded once and broadcast.
    ///
    /// MCAST_BAR protocol: before rank 0 can overwrite a B buffer stage, ALL 4 CTAs
    /// must signal they've consumed the previous B data from that stage. Each non-rank-0
    /// CTA arrives at rank 0's MCAST_BAR via `mbarrier_arrive_cluster`.
    ///
    /// Critical ordering: each CTA arms its local `TMA_BAR` before issuing its A load
    /// and before advertising readiness through `MCAST_BAR`. Rank 0 uses a cluster-acquire
    /// wait before issuing B multicast, so remote B bytes cannot reach an unarmed barrier.
    ///
    /// Grid launch: grid_dim = (total_tiles, 1, 1), cluster_dim = (4, 1, 1)
    #[kernel]
    #[cluster_launch(4, 1, 1)]
    pub unsafe fn gemm_sol_clc_multicast(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        mut out: DisjointSlice<u32>,
        n: i32,
        k: i32,
        tiles_m: u32,
        _tiles_n: u32,
    ) {
        unsafe {
            // ── SMEM layout (same as Phase 4A) ──
            static mut SMEM_A0: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_A1: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_B0: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_B1: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_OUT: SharedArray<u32, 8192, 128> = SharedArray::UNINIT;
            static mut TMEM_ADDR: SharedArray<u32, 1, 4> = SharedArray::UNINIT;

            static mut TILE_INFO: SharedArray<u32, 4, 4> = SharedArray::UNINIT;

            // Barriers: TMA↔MMA double-buffered
            static mut TMA_BAR0: Barrier = Barrier::UNINIT;
            static mut TMA_BAR1: Barrier = Barrier::UNINIT;
            static mut MMA_BAR0: Barrier = Barrier::UNINIT;
            static mut MMA_BAR1: Barrier = Barrier::UNINIT;

            // Barriers: TMEM accumulator pipeline
            static mut ACCUM_FULL0: Barrier = Barrier::UNINIT;
            static mut ACCUM_FULL1: Barrier = Barrier::UNINIT;
            static mut ACCUM_EMPTY0: Barrier = Barrier::UNINIT;
            static mut ACCUM_EMPTY1: Barrier = Barrier::UNINIT;

            static mut TILE_READY: Barrier = Barrier::UNINIT;
            // The producer must not overwrite TILE_INFO until every reader warp
            // has copied the current tile coordinates/has_work flag.
            static mut TILE_INFO_FREE: Barrier = Barrier::UNINIT;

            // CLC: 16-byte response buffer + mbarrier
            static mut CLC_RESPONSE: SharedArray<u64, 2, 16> = SharedArray::UNINIT;
            static mut CLC_BAR: Barrier = Barrier::UNINIT;
            static mut CLC_READY: Barrier = Barrier::UNINIT;

            // TMA multicast: cluster-wide consumer barriers.
            // Rank 0's TMA warp waits on these before multicasting B to ensure
            // ALL cluster CTAs have consumed the previous B from this stage.
            static mut MCAST_BAR0: Barrier = Barrier::UNINIT;
            static mut MCAST_BAR1: Barrier = Barrier::UNINIT;
            const A_TILE_BYTES: u32 = 128 * 64 * 2;
            const B_TILE_BYTES: u32 = 128 * 64 * 2;
            const SBO_BYTES: u32 = 1024;
            const LBO_BYTES: u32 = 16;
            const SWIZZLE_128B: u8 = 2;

            const TMA_WARP: u32 = 4;
            const MMA_WARP: u32 = 5;
            const NUM_ACCUM_STAGES: u32 = 2;
            const ACCUM_STAGE_COLS: u32 = 128;
            const CLUSTER_SIZE: u32 = 4;
            const CTA_MASK_ALL: u16 = 0b1111;

            let n = n as u32;
            let k = k as u32;
            let tid = thread::threadIdx_x();
            let warp_id = warp::warp_id();
            let lane_id = tid % 32;

            // ── Initialize barriers + allocate TMEM ──
            if tid == 0 {
                mbarrier_init(&raw mut TMA_BAR0, 1);
                mbarrier_init(&raw mut TMA_BAR1, 1);
                mbarrier_init(&raw mut MMA_BAR0, 1);
                mbarrier_init(&raw mut MMA_BAR1, 1);
                mbarrier_init(&raw mut ACCUM_FULL0, 1);
                mbarrier_init(&raw mut ACCUM_FULL1, 1);
                mbarrier_init(&raw mut ACCUM_EMPTY0, 128);
                mbarrier_init(&raw mut ACCUM_EMPTY1, 128);
                mbarrier_init(&raw mut TILE_READY, 1);
                // One synchronized leader per reader warp acknowledges that all
                // lanes have copied the mailbox before the producer reuses it.
                mbarrier_init(&raw mut TILE_INFO_FREE, 5);
                mbarrier_init(&raw mut CLC_BAR, 1);
                mbarrier_init(&raw mut CLC_READY, CLUSTER_SIZE);
                // MCAST_BARs: all 4 cluster CTAs must arrive before rank 0 can reuse B buffer
                mbarrier_init(&raw mut MCAST_BAR0, CLUSTER_SIZE);
                mbarrier_init(&raw mut MCAST_BAR1, CLUSTER_SIZE);
                fence_mbarrier_init_release_cluster();
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();

            let my_rank = cluster::cluster_ctaidX();

            // map_shared_rank translates a local SMEM pointer to the address in rank 0's
            // shared memory, for use with mbarrier_arrive_cluster (cross-CTA barrier arrive).
            let rank0_mcast_bar0_addr = cvta_generic_to_shared_offset(cluster::map_shared_rank(
                &raw const MCAST_BAR0,
                0,
            ) as *const u8);
            let rank0_mcast_bar1_addr = cvta_generic_to_shared_offset(cluster::map_shared_rank(
                &raw const MCAST_BAR1,
                0,
            ) as *const u8);
            let rank0_clc_ready_addr = cvta_generic_to_shared_offset(cluster::map_shared_rank(
                &raw const CLC_READY,
                0,
            ) as *const u8);

            // Pre-arrive MMA_BARs so TMA can proceed on the first K-iteration
            if tid == 0 {
                mbarrier_arrive(&raw const MMA_BAR0);
                mbarrier_arrive(&raw const MMA_BAR1);
            }

            thread::sync_threads();

            if warp_id == 0 {
                tcgen05_alloc(&raw mut TMEM_ADDR as *mut u32, 512);
            }
            thread::sync_threads();
            let tmem_addr = *(&raw const TMEM_ADDR as *const u32);

            let idesc = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape::M128_N128)
                .element_type(Tcgen05ElementType::F16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .build()
                .raw();

            let k_iters = k / 64;

            cluster::cluster_sync();

            // No synthetic pre-arrive is needed for MCAST_BARs. On the first use
            // of each stage, every CTA arms its local TMA_BAR, arrives at rank 0's
            // MCAST_BAR, and rank 0 performs the normal cluster-acquire wait.

            // ════════════════════════════════════════════════════════════════════
            // TMA Producer (warp 4): CLC tile scheduling + TMA multicast
            // ════════════════════════════════════════════════════════════════════
            if warp_id == TMA_WARP {
                let is_lane0 = lane_id == 0;
                let is_rank0 = my_rank == 0;
                let mut global_k: u32 = 0;
                let mut clc_iter: u32 = 0;
                let mut tile_info_free_parity: u32 = 0;

                // ── First tile: use our own blockIdx (hardware-assigned) ──
                let first_ctaid = thread::blockIdx_x();
                let first_tile_m = first_ctaid % tiles_m;
                let first_tile_n = first_ctaid / tiles_m;

                if is_lane0 {
                    *(&raw mut TILE_INFO as *mut u32).add(0) = first_tile_m;
                    *(&raw mut TILE_INFO as *mut u32).add(1) = first_tile_n;
                    *(&raw mut TILE_INFO as *mut u32).add(2) = 1;
                    mbarrier_arrive(&raw const TILE_READY);
                }
                let m_offset = (first_tile_m * 128) as i32;
                let n_offset = (first_tile_n * 128) as i32;

                let mut k_idx: u32 = 0;
                while k_idx < k_iters {
                    let stage = global_k & 1;
                    let mma_parity = (global_k >> 1) & 1;

                    // Wait for local MMA to finish consuming this stage
                    if stage == 0 {
                        while !mbarrier_try_wait_parity(&raw const MMA_BAR0, mma_parity) {}
                    } else {
                        while !mbarrier_try_wait_parity(&raw const MMA_BAR1, mma_parity) {}
                    }
                    // All lanes wait, then sync before lane 0 issues the loads
                    // that let MMA re-arm this barrier (no-stranding rule).
                    warp::sync_mask(u32::MAX);

                    // Arm the local completion barrier and issue the per-CTA A
                    // copy before advertising that this CTA is ready for B multicast.
                    let k_base = (k_idx * 64) as i32;
                    if is_lane0 {
                        if stage == 0 {
                            mbarrier_arrive_expect_tx(
                                &raw const TMA_BAR0,
                                1,
                                A_TILE_BYTES + B_TILE_BYTES,
                            );
                            cp_async_bulk_tensor_2d_g2s(
                                &raw mut SMEM_A0 as *mut u8,
                                a_tma,
                                k_base,
                                m_offset,
                                &raw mut TMA_BAR0,
                            );
                            fence_proxy_async_shared_cta();
                            mbarrier_arrive_cluster(rank0_mcast_bar0_addr);
                        } else {
                            mbarrier_arrive_expect_tx(
                                &raw const TMA_BAR1,
                                1,
                                A_TILE_BYTES + B_TILE_BYTES,
                            );
                            cp_async_bulk_tensor_2d_g2s(
                                &raw mut SMEM_A1 as *mut u8,
                                a_tma,
                                k_base,
                                m_offset,
                                &raw mut TMA_BAR1,
                            );
                            fence_proxy_async_shared_cta();
                            mbarrier_arrive_cluster(rank0_mcast_bar1_addr);
                        }
                    }

                    // Remote arrivals require a cluster-acquire wait at rank 0.
                    let mcast_parity = (global_k >> 1) & 1;
                    if is_rank0 {
                        if stage == 0 {
                            while !mbarrier_try_wait_parity_cluster(
                                &raw const MCAST_BAR0,
                                mcast_parity,
                            ) {}
                        } else {
                            while !mbarrier_try_wait_parity_cluster(
                                &raw const MCAST_BAR1,
                                mcast_parity,
                            ) {}
                        }
                    }

                    if is_rank0 && is_lane0 {
                        if stage == 0 {
                            cp_async_bulk_tensor_2d_g2s_multicast(
                                &raw mut SMEM_B0 as *mut u8,
                                b_tma,
                                k_base,
                                n_offset,
                                &raw mut TMA_BAR0,
                                CTA_MASK_ALL,
                            );
                        } else {
                            cp_async_bulk_tensor_2d_g2s_multicast(
                                &raw mut SMEM_B1 as *mut u8,
                                b_tma,
                                k_base,
                                n_offset,
                                &raw mut TMA_BAR1,
                                CTA_MASK_ALL,
                            );
                        }
                    }

                    k_idx += 1;
                    global_k += 1;
                }

                {
                    // ── Subsequent tiles: CLC work-stealing ──
                    // Rank 0 issues clc_try_cancel_multicast, and CTAs derive per-rank tile IDs
                    // from the shared response (first_stolen + rank).
                    let resp_ptr = &raw mut CLC_RESPONSE as *mut u64;

                    loop {
                        let clc_parity = clc_iter & 1;

                        // Arm every CTA-local completion barrier before rank 0 can
                        // multicast the next response. CLC_READY also proves that
                        // every CTA is still alive and has released its prior read.
                        let mut is_canceled = 0u32;
                        let mut first_stolen = 0u32;
                        if is_lane0 {
                            mbarrier_arrive_expect_tx_cluster(&raw const CLC_BAR, 1, 16);
                            mbarrier_arrive_cluster(rank0_clc_ready_addr);

                            if is_rank0 {
                                while !mbarrier_try_wait_parity_cluster(
                                    &raw const CLC_READY,
                                    clc_parity,
                                ) {}
                                fence_proxy_async_generic_acquire_shared_cluster_cluster();
                                clc_try_cancel_multicast(resp_ptr as *mut u8, &raw mut CLC_BAR);
                            }

                            while !mbarrier_try_wait_parity_cluster(&raw const CLC_BAR, clc_parity)
                            {
                            }

                            let resp_lo = *resp_ptr;
                            let resp_hi = *resp_ptr.add(1);
                            is_canceled = clc_query_is_canceled(resp_lo, resp_hi);
                            if is_canceled != 0 {
                                first_stolen = clc_query_get_first_ctaid_x(resp_lo, resp_hi);
                            }
                            fence_proxy_async_generic_release_shared_cta_cluster();
                        }

                        is_canceled = warp::shuffle_sync(u32::MAX, is_canceled, 0);
                        first_stolen = warp::shuffle_sync(u32::MAX, first_stolen, 0);

                        if is_canceled == 0 {
                            if is_lane0 {
                                while !mbarrier_try_wait_parity(
                                    &raw const TILE_INFO_FREE,
                                    tile_info_free_parity,
                                ) {}
                                *(&raw mut TILE_INFO as *mut u32).add(2) = 0;
                                mbarrier_arrive(&raw const TILE_READY);
                            }
                            break;
                        }
                        let my_ctaid = first_stolen + my_rank;
                        let tile_m = my_ctaid % tiles_m;
                        let tile_n = my_ctaid / tiles_m;

                        if is_lane0 {
                            while !mbarrier_try_wait_parity(
                                &raw const TILE_INFO_FREE,
                                tile_info_free_parity,
                            ) {}
                            tile_info_free_parity ^= 1;
                            *(&raw mut TILE_INFO as *mut u32).add(0) = tile_m;
                            *(&raw mut TILE_INFO as *mut u32).add(1) = tile_n;
                            *(&raw mut TILE_INFO as *mut u32).add(2) = 1;
                            mbarrier_arrive(&raw const TILE_READY);
                        }

                        let m_off = (tile_m * 128) as i32;
                        let n_off = (tile_n * 128) as i32;

                        let mut k_idx: u32 = 0;
                        while k_idx < k_iters {
                            let stage = global_k & 1;
                            let mma_parity = (global_k >> 1) & 1;

                            // Wait for local MMA to finish consuming this stage
                            if stage == 0 {
                                while !mbarrier_try_wait_parity(&raw const MMA_BAR0, mma_parity) {}
                            } else {
                                while !mbarrier_try_wait_parity(&raw const MMA_BAR1, mma_parity) {}
                            }
                            // All lanes wait, then sync before lane 0 issues the
                            // loads that let MMA re-arm this barrier
                            // (no-stranding rule).
                            warp::sync_mask(u32::MAX);

                            // Arm the local barrier and start A before advertising
                            // readiness for the cluster-wide B multicast.
                            let k_base = (k_idx * 64) as i32;
                            if is_lane0 {
                                if stage == 0 {
                                    mbarrier_arrive_expect_tx(
                                        &raw const TMA_BAR0,
                                        1,
                                        A_TILE_BYTES + B_TILE_BYTES,
                                    );
                                    cp_async_bulk_tensor_2d_g2s(
                                        &raw mut SMEM_A0 as *mut u8,
                                        a_tma,
                                        k_base,
                                        m_off,
                                        &raw mut TMA_BAR0,
                                    );
                                    fence_proxy_async_shared_cta();
                                    mbarrier_arrive_cluster(rank0_mcast_bar0_addr);
                                } else {
                                    mbarrier_arrive_expect_tx(
                                        &raw const TMA_BAR1,
                                        1,
                                        A_TILE_BYTES + B_TILE_BYTES,
                                    );
                                    cp_async_bulk_tensor_2d_g2s(
                                        &raw mut SMEM_A1 as *mut u8,
                                        a_tma,
                                        k_base,
                                        m_off,
                                        &raw mut TMA_BAR1,
                                    );
                                    fence_proxy_async_shared_cta();
                                    mbarrier_arrive_cluster(rank0_mcast_bar1_addr);
                                }
                            }

                            let mcast_parity = (global_k >> 1) & 1;
                            if is_rank0 {
                                if stage == 0 {
                                    while !mbarrier_try_wait_parity_cluster(
                                        &raw const MCAST_BAR0,
                                        mcast_parity,
                                    ) {}
                                } else {
                                    while !mbarrier_try_wait_parity_cluster(
                                        &raw const MCAST_BAR1,
                                        mcast_parity,
                                    ) {}
                                }
                            }

                            if is_rank0 && is_lane0 {
                                if stage == 0 {
                                    cp_async_bulk_tensor_2d_g2s_multicast(
                                        &raw mut SMEM_B0 as *mut u8,
                                        b_tma,
                                        k_base,
                                        n_off,
                                        &raw mut TMA_BAR0,
                                        CTA_MASK_ALL,
                                    );
                                } else {
                                    cp_async_bulk_tensor_2d_g2s_multicast(
                                        &raw mut SMEM_B1 as *mut u8,
                                        b_tma,
                                        k_base,
                                        n_off,
                                        &raw mut TMA_BAR1,
                                        CTA_MASK_ALL,
                                    );
                                }
                            }

                            k_idx += 1;
                            global_k += 1;
                        }

                        clc_iter += 1;
                    }
                }
            }

            // ════════════════════════════════════════════════════════════════════
            // MMA Consumer (warp 5): identical to Phase 4A
            // ════════════════════════════════════════════════════════════════════
            if warp_id == MMA_WARP {
                let is_lane0 = lane_id == 0;
                let mut tile_iter: u32 = 0;
                let mut tile_parity: u32 = 0;
                let mut global_k: u32 = 0;

                loop {
                    while !mbarrier_try_wait_parity(&raw const TILE_READY, tile_parity) {}
                    tile_parity ^= 1;

                    let has_work = *(&raw const TILE_INFO as *const u32).add(2);
                    warp::sync_mask(u32::MAX);
                    if is_lane0 {
                        mbarrier_arrive(&raw const TILE_INFO_FREE);
                    }
                    if has_work == 0 {
                        break;
                    }

                    let accum_stage = tile_iter % NUM_ACCUM_STAGES;
                    let tmem_stage_offset = accum_stage * ACCUM_STAGE_COLS;

                    // All lanes may wait here: the next ACCUM_EMPTY phase needs
                    // an ACCUM_FULL commit from a later tile, which sits behind
                    // this warp's tile-top sync, so the barrier cannot advance
                    // past a lane that is still waiting on it.
                    if tile_iter >= NUM_ACCUM_STAGES {
                        let empty_parity = ((tile_iter - NUM_ACCUM_STAGES) / NUM_ACCUM_STAGES) & 1;
                        if accum_stage == 0 {
                            while !mbarrier_try_wait_parity(&raw const ACCUM_EMPTY0, empty_parity) {
                            }
                        } else {
                            while !mbarrier_try_wait_parity(&raw const ACCUM_EMPTY1, empty_parity) {
                            }
                        }
                    }

                    let mut k_idx: u32 = 0;
                    while k_idx < k_iters {
                        let stage = global_k & 1;
                        let tma_parity = (global_k >> 1) & 1;

                        if stage == 0 {
                            while !mbarrier_try_wait_parity(&raw const TMA_BAR0, tma_parity) {}
                        } else {
                            while !mbarrier_try_wait_parity(&raw const TMA_BAR1, tma_parity) {}
                        }
                        // All lanes wait, then sync before lane 0 consumes the
                        // stage. The commit below is what lets the producer
                        // re-arm this barrier, so ordering it after the sync
                        // means the barrier can never advance past a lane that
                        // is still waiting on it -- no lane can strand one
                        // parity short of a stopped barrier.
                        warp::sync_mask(u32::MAX);

                        if is_lane0 {
                            let smem_a_base = if stage == 0 {
                                cvta_generic_to_shared_offset(&raw const SMEM_A0 as *const u8)
                            } else {
                                cvta_generic_to_shared_offset(&raw const SMEM_A1 as *const u8)
                            };
                            let smem_b_base = if stage == 0 {
                                cvta_generic_to_shared_offset(&raw const SMEM_B0 as *const u8)
                            } else {
                                cvta_generic_to_shared_offset(&raw const SMEM_B1 as *const u8)
                            };

                            let mut j: u32 = 0;
                            while j < 4 {
                                let byte_offset = (j * 32) as u64;
                                let a_desc = build_smem_descriptor(
                                    smem_a_base + byte_offset,
                                    LBO_BYTES,
                                    SBO_BYTES,
                                    SWIZZLE_128B,
                                );
                                let b_desc = build_smem_descriptor(
                                    smem_b_base + byte_offset,
                                    LBO_BYTES,
                                    SBO_BYTES,
                                    SWIZZLE_128B,
                                );

                                let accumulate = k_idx > 0 || j > 0;
                                tcgen05_mma_f16(
                                    tmem_addr + tmem_stage_offset,
                                    a_desc,
                                    b_desc,
                                    idesc,
                                    accumulate,
                                );
                                j += 1;
                            }

                            if stage == 0 {
                                tcgen05_commit_shared_cluster(&raw mut MMA_BAR0 as *mut u64);
                            } else {
                                tcgen05_commit_shared_cluster(&raw mut MMA_BAR1 as *mut u64);
                            }
                        }

                        k_idx += 1;
                        global_k += 1;
                    }

                    if is_lane0 {
                        if accum_stage == 0 {
                            tcgen05_commit_shared_cluster(&raw mut ACCUM_FULL0 as *mut u64);
                        } else {
                            tcgen05_commit_shared_cluster(&raw mut ACCUM_FULL1 as *mut u64);
                        }
                    }

                    tile_iter += 1;
                }
            }

            // ════════════════════════════════════════════════════════════════════
            // Epilogue warps (0-3): identical to Phase 4A
            // ════════════════════════════════════════════════════════════════════
            if warp_id < 4 {
                let is_lane0 = lane_id == 0;
                let mut epi_tile_iter: u32 = 0;
                let mut tile_parity: u32 = 0;

                const TILE_N: usize = 128;
                let warp_row_base = (warp_id * 32) as usize;
                let row_stride_bytes = TILE_N * 2;
                let row_within_8 = (lane_id % 8) as usize;
                let is_second_matrix = (8..16).contains(&lane_id);
                let col_offset_for_matrix2 = if is_second_matrix { 16usize } else { 0usize };

                loop {
                    while !mbarrier_try_wait_parity(&raw const TILE_READY, tile_parity) {}
                    tile_parity ^= 1;

                    let has_work = *(&raw const TILE_INFO as *const u32).add(2);
                    let tile_m = *(&raw const TILE_INFO as *const u32).add(0);
                    let tile_n = *(&raw const TILE_INFO as *const u32).add(1);
                    warp::sync_mask(u32::MAX);
                    if is_lane0 {
                        mbarrier_arrive(&raw const TILE_INFO_FREE);
                    }
                    if has_work == 0 {
                        break;
                    }

                    let accum_stage = epi_tile_iter % NUM_ACCUM_STAGES;
                    let tmem_stage_offset = accum_stage * ACCUM_STAGE_COLS;

                    let full_parity = (epi_tile_iter / NUM_ACCUM_STAGES) & 1;
                    if accum_stage == 0 {
                        while !mbarrier_try_wait_parity(&raw const ACCUM_FULL0, full_parity) {}
                    } else {
                        while !mbarrier_try_wait_parity(&raw const ACCUM_FULL1, full_parity) {}
                    }

                    let mut tmem_row_block = 0u32;
                    while tmem_row_block < 2 {
                        let tmem_row = warp_id * 32 + tmem_row_block * 16;

                        let mut col_block = 0u32;
                        while col_block < 8 {
                            let col_offset = (col_block * 16) as usize;

                            let regs_a = tcgen05_ld_16x256b_pure(
                                tmem_addr
                                    + tmem_stage_offset
                                    + (tmem_row << 16)
                                    + col_offset as u32,
                            );
                            tcgen05_load_wait();

                            let regs_b = tcgen05_ld_16x256b_pure(
                                tmem_addr
                                    + tmem_stage_offset
                                    + (tmem_row << 16)
                                    + col_offset as u32
                                    + 8,
                            );
                            tcgen05_load_wait();

                            let p0_lo = cvt_bf16x2_f32(regs_a[0], regs_a[1]);
                            let p1_lo = cvt_bf16x2_f32(regs_b[0], regs_b[1]);
                            let out_row_lo =
                                warp_row_base + (tmem_row_block as usize * 16) + row_within_8;
                            let smem_addr_lo = (&raw mut SMEM_OUT as *mut u8).add(
                                out_row_lo * row_stride_bytes
                                    + col_offset * 2
                                    + col_offset_for_matrix2,
                            );
                            stmatrix_m8n8_x2(smem_addr_lo, p0_lo, p1_lo);

                            let p0_hi = cvt_bf16x2_f32(regs_a[2], regs_a[3]);
                            let p1_hi = cvt_bf16x2_f32(regs_b[2], regs_b[3]);
                            let out_row_hi =
                                warp_row_base + (tmem_row_block as usize * 16) + 8 + row_within_8;
                            let smem_addr_hi = (&raw mut SMEM_OUT as *mut u8).add(
                                out_row_hi * row_stride_bytes
                                    + col_offset * 2
                                    + col_offset_for_matrix2,
                            );
                            stmatrix_m8n8_x2(smem_addr_hi, p0_hi, p1_hi);

                            col_block += 1;
                        }
                        tmem_row_block += 1;
                    }

                    let n_u32 = (n / 2) as usize;
                    let tile_row_base = (tile_m * 128) as usize;
                    let tile_col_base = (tile_n * 64) as usize;
                    let base_row = warp_id as usize * 32;

                    let mut elem = lane_id as usize;
                    while elem < 2048 {
                        let local_row = elem / 64;
                        let local_col = elem % 64;
                        let smem_idx = (base_row + local_row) * 64 + local_col;
                        let global_row = tile_row_base + base_row + local_row;
                        let global_col = tile_col_base + local_col;
                        let global_idx = global_row * n_u32 + global_col;

                        *out.get_unchecked_mut(global_idx) = SMEM_OUT[smem_idx];
                        elem += 32;
                    }

                    if accum_stage == 0 {
                        mbarrier_arrive(&raw const ACCUM_EMPTY0);
                    } else {
                        mbarrier_arrive(&raw const ACCUM_EMPTY1);
                    }

                    epi_tile_iter += 1;
                }
            }

            // ── Cleanup ──
            thread::sync_threads();
            cluster::cluster_sync();
            if warp_id == 0 {
                tcgen05_dealloc(tmem_addr, 512);
            }
            if tid == 0 {
                mbarrier_inval(&raw mut TMA_BAR0);
                mbarrier_inval(&raw mut TMA_BAR1);
                mbarrier_inval(&raw mut MMA_BAR0);
                mbarrier_inval(&raw mut MMA_BAR1);
                mbarrier_inval(&raw mut ACCUM_FULL0);
                mbarrier_inval(&raw mut ACCUM_FULL1);
                mbarrier_inval(&raw mut ACCUM_EMPTY0);
                mbarrier_inval(&raw mut ACCUM_EMPTY1);
                mbarrier_inval(&raw mut TILE_READY);
                mbarrier_inval(&raw mut TILE_INFO_FREE);
                mbarrier_inval(&raw mut CLC_BAR);
                mbarrier_inval(&raw mut CLC_READY);
                mbarrier_inval(&raw mut MCAST_BAR0);
                mbarrier_inval(&raw mut MCAST_BAR1);
            }
        }
    }

    /// Phase 4D: CLC + TMA multicast with 4 SMEM stages and no MCAST_BAR.
    ///
    /// Blackwell GEMM kernel: CLC + cta_group::2 + 4-stage SMEM pipeline.
    ///
    /// Architecture: 6 warps per CTA, cluster size 2 (CTA pairs).
    ///   - Warp 4 (TMA): loads A/B tiles from global via TMA into shared memory.
    ///   - Warp 5 (MMA): consumes SMEM tiles via pair-UMMA (tcgen05, cta_group::2).
    ///   - Warps 0-3 (Epilogue): read accumulators from TMEM, convert f32->bf16, store to global.
    ///
    /// Pipeline stages: 4 SMEM stages (TMA_BAR0..3 / MMA_BAR0..3), 2 accumulator stages
    /// (ACCUM_FULL0/1, ACCUM_EMPTY0/1), plus TILE_READY for TMA→MMA/Epilogue tile handoff.
    ///
    /// Key synchronization protocol (cta_group::2 barrier aliasing):
    ///   Both CTAs in a pair issue TMA loads, but the barrier pointer is masked with
    ///   PEER_BIT_MASK (0xFEFFFFF8) before being passed to the TMA instruction. This
    ///   clears bit 24, redirecting both CTAs' completion signals to the leader CTA's
    ///   (rank 0) barrier. Consequently:
    ///     - Only the leader sets expect_tx (doubled: both CTAs' bytes land on one barrier).
    ///     - Only the leader waits on TMA barriers in the MMA warp.
    ///     - The follower's MMA warp skips TMA waits since pair-UMMA is leader-issued.
    ///   MMA barriers still use normal multicast (tcgen05_commit_multicast_cg2 with
    ///   CTA_MASK_PAIR=0b11), so both CTAs receive MMA completion signals.
    ///
    /// CLC work-stealing: both CTAs arm their local CLC_BAR and arrive at the leader's
    /// CLC_READY; rank 0 waits for the pair, then issues clc_try_cancel_multicast so both
    /// CTAs receive the response via CLC_BAR. Lane 0 in each CTA decodes the response and
    /// warp-broadcasts the result. Tile indices are derived by dividing the CLC
    /// first_ctaid_x by the cluster size (2), NOT using raw CTA IDs.
    ///
    /// # Shape contract
    ///
    /// `k` must be a multiple of 256. With a K tile size of 64, this makes `k_iters`
    /// a multiple of four, so every output tile starts on pipeline stage 0. The MMA
    /// consumer relies on that reset to keep its unrolled stage selection constant.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    pub unsafe fn gemm_sol_clc_multicast_4_stage_pipeline(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        mut out: DisjointSlice<u32>,
        n: i32,
        k: i32,
        tiles_m: u32,
        _tiles_n: u32,
    ) {
        unsafe {
            static mut SMEM_A0: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_A1: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_A2: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_A3: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_B0: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_B1: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_B2: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_B3: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_OUT: SharedArray<u32, 8192, 128> = SharedArray::UNINIT;
            static mut TMEM_ADDR: SharedArray<u32, 1, 4> = SharedArray::UNINIT;

            static mut TILE_INFO: SharedArray<u32, 4, 4> = SharedArray::UNINIT;

            // 4-stage TMA <-> MMA pipeline
            static mut TMA_BAR0: Barrier = Barrier::UNINIT;
            static mut TMA_BAR1: Barrier = Barrier::UNINIT;
            static mut TMA_BAR2: Barrier = Barrier::UNINIT;
            static mut TMA_BAR3: Barrier = Barrier::UNINIT;
            static mut MMA_BAR0: Barrier = Barrier::UNINIT;
            static mut MMA_BAR1: Barrier = Barrier::UNINIT;
            static mut MMA_BAR2: Barrier = Barrier::UNINIT;
            static mut MMA_BAR3: Barrier = Barrier::UNINIT;
            static mut ACCUM_FULL0: Barrier = Barrier::UNINIT;
            static mut ACCUM_FULL1: Barrier = Barrier::UNINIT;
            static mut ACCUM_EMPTY0: Barrier = Barrier::UNINIT;
            static mut ACCUM_EMPTY1: Barrier = Barrier::UNINIT;
            static mut TILE_READY: Barrier = Barrier::UNINIT;
            // The producer must not overwrite TILE_INFO until every reader warp
            // has copied the current tile coordinates/has_work flag.
            //
            // CTA-LOCALITY INVARIANT: TILE_INFO / TILE_READY / TILE_INFO_FREE
            // are accessed only by warps of the owning CTA -- never via
            // cluster::map_shared_rank/mapa and never by the TMA/async proxy.
            // Both CTAs of the pair compute tile metadata independently from
            // the multicast CLC response, each into its own mailbox, so the
            // CTA-scope mbarrier_arrive/mbarrier_try_wait_parity handshake is
            // sufficient. The init count of 5 is the CTA-local reader set:
            // epilogue warps 0-3 plus the MMA warp's tile-top ack, none gated
            // by elect_one_cta (gating the MMA ack would leave the follower
            // CTA one arrival short and deadlock its producer). If any access
            // ever becomes cross-CTA (e.g. a leader-only publish shared via
            // map_shared_rank, or TILE_INFO delivered by multicast), the acks
            // must become mbarrier_arrive_cluster, the producer wait
            // mbarrier_try_wait_parity_cluster, and
            // fence_mbarrier_init_release_cluster must follow mbarrier_init.
            static mut TILE_INFO_FREE: Barrier = Barrier::UNINIT;

            static mut CLC_RESPONSE: SharedArray<u64, 2, 16> = SharedArray::UNINIT;
            static mut CLC_BAR: Barrier = Barrier::UNINIT;
            static mut CLC_READY: Barrier = Barrier::UNINIT;

            const A_TILE_BYTES: u32 = 128 * 64 * 2;
            const B_TILE_BYTES: u32 = 64 * 64 * 2; // 64 B rows per CTA (split by rank)
            const SBO_BYTES: u32 = 1024;
            const LBO_BYTES: u32 = 16;
            const SWIZZLE_128B: u8 = 2;

            const TMA_WARP: u32 = 4;
            const MMA_WARP: u32 = 5;
            const NUM_ACCUM_STAGES: u32 = 2;
            const ACCUM_STAGE_COLS: u32 = 128;
            const CLUSTER_SIZE: u32 = 2;
            const CTA_MASK_PAIR: u16 = 0b11;
            // Clears bit 24 (CTA rank within pair) + alignment bits 2:0 of a shared
            // memory barrier address, redirecting TMA completions to the leader CTA's
            // barrier.
            const PEER_BIT_MASK: u32 = 0xFEFFFFF8;
            // L2 cache-blocking: visit tiles in vertical N-bands SWIZZLE_G wide so
            // tiles done near-in-time share A-rows + a small set of B-cols in L2.
            // This is the dominant data-movement lever at large sizes (~+88% @16384).
            const SWIZZLE_G: u32 = 8;

            let n = n as u32;
            let k = k as u32;
            let tid = thread::threadIdx_x();
            let _ctaid = thread::blockIdx_x();
            let warp_id = warp::warp_id();
            let lane_id = tid % 32;

            let my_rank = cluster::cluster_ctaidX();
            let self_mask: u16 = 1u16 << (my_rank as u16);
            if tid == 0 {
                mbarrier_init(&raw mut TMA_BAR0, 1);
                mbarrier_init(&raw mut TMA_BAR1, 1);
                mbarrier_init(&raw mut TMA_BAR2, 1);
                mbarrier_init(&raw mut TMA_BAR3, 1);
                mbarrier_init(&raw mut MMA_BAR0, 1);
                mbarrier_init(&raw mut MMA_BAR1, 1);
                mbarrier_init(&raw mut MMA_BAR2, 1);
                mbarrier_init(&raw mut MMA_BAR3, 1);
                mbarrier_init(&raw mut ACCUM_FULL0, 1);
                mbarrier_init(&raw mut ACCUM_FULL1, 1);
                mbarrier_init(&raw mut ACCUM_EMPTY0, 256);
                mbarrier_init(&raw mut ACCUM_EMPTY1, 256);
                mbarrier_init(&raw mut TILE_READY, 1);
                // One synchronized leader per reader warp acknowledges that all
                // lanes have copied the mailbox before the producer reuses it.
                mbarrier_init(&raw mut TILE_INFO_FREE, 5);
                mbarrier_init(&raw mut CLC_BAR, 1);
                mbarrier_init(&raw mut CLC_READY, CLUSTER_SIZE);
                // The follower's lane 0 arrives at the leader's CLC_READY via a
                // map_shared_rank address, so remote arrivals must be able to
                // observe the initialized barrier state.
                fence_mbarrier_init_release_cluster();
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();

            // Pre-arrive all MMA stage barriers so producer can start immediately.
            if tid == 0 {
                mbarrier_arrive(&raw const MMA_BAR0);
                mbarrier_arrive(&raw const MMA_BAR1);
                mbarrier_arrive(&raw const MMA_BAR2);
                mbarrier_arrive(&raw const MMA_BAR3);
            }
            thread::sync_threads();

            if warp_id == 0 {
                tcgen05_alloc_cg2(&raw mut TMEM_ADDR as *mut u32, 512);
            }
            thread::sync_threads();
            let tmem_addr = *(&raw const TMEM_ADDR as *const u32);
            let elect_one_cta = my_rank == 0;

            let idesc = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape::M256_N128)
                .element_type(Tcgen05ElementType::F16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .build()
                .raw();

            let k_iters = k / 64;
            cluster::cluster_sync();

            if warp_id == TMA_WARP {
                let is_lane0 = lane_id == 0;
                let mut tile_seq: u32 = 0;
                let mut clc_iter: u32 = 0;
                let mut tile_info_free_parity: u32 = 0;

                // BUG FIX: CLC assigns consecutive blockIdx values to CTAs in a cluster.
                // For a cluster of size 2: CTA pair (0,1) has blockIdx (0,1), pair (2,3) has
                // blockIdx (2,3), etc. The tile index is cluster_base_id / cluster_size, NOT
                // the raw blockIdx. Using raw blockIdx caused only row 0 and row 2048 to be
                // correct (each CTA computed its own rank's 128 rows at the wrong tile).
                let cluster_base_id = thread::blockIdx_x() - my_rank;
                let tile_idx = cluster_base_id / 2;
                // L2 CACHE-BLOCKING SWIZZLE (CUTLASS-style tile ordering).
                //
                // Change the order in which output tiles are assigned to clusters. Split
                // the N direction into bands of at most SWIZZLE_G columns, visit every M
                // row in one band, and then move to the next band.
                //
                // For SWIZZLE_G = 3, the first band is visited in this order:
                //
                //   tile_idx:  0      1      2      3      4      5
                //   C tile:   (0,0)  (0,1)  (0,2)  (1,0)  (1,1)  (1,2)  ...
                //
                // Within each M row, these tiles reuse the same A tile while reading
                // neighboring B tiles. The same small set of B tiles is then revisited for
                // the next M row, so the data is more likely to still be in the L2 cache.
                //
                // This changes only the visit order. The mapping from tile_idx to
                // (tile_m, tile_n) is one-to-one, and TILE_INFO tells the epilogue where to
                // write C. Every output tile is still computed exactly once with the same
                // result.
                let tiles_n = _tiles_n;
                let group_tiles = SWIZZLE_G * tiles_m;
                let group = tile_idx / group_tiles;
                let in_group = tile_idx % group_tiles;
                let n_start = group * SWIZZLE_G;
                let band_w = if SWIZZLE_G < tiles_n - n_start {
                    SWIZZLE_G
                } else {
                    tiles_n - n_start
                };
                let first_tile_m = in_group / band_w;
                let first_tile_n = n_start + in_group % band_w;

                if is_lane0 {
                    *(&raw mut TILE_INFO as *mut u32).add(0) = first_tile_m;
                    *(&raw mut TILE_INFO as *mut u32).add(1) = first_tile_n;
                    *(&raw mut TILE_INFO as *mut u32).add(2) = 1;
                    mbarrier_arrive(&raw const TILE_READY);
                }

                let m_offset = (first_tile_m * 256 + my_rank * 128) as i32;
                let b_n_offset = (first_tile_n * 128 + my_rank * 64) as i32;

                let mut k_idx: u32 = 0;
                while k_idx < k_iters {
                    let global_k = tile_seq * k_iters + k_idx;
                    let stage = global_k & 3;
                    let mma_parity = (global_k >> 2) & 1;

                    let (smem_a_ptr, smem_b_ptr, tma_bar_const, tma_bar_mut, mma_bar_const): (
                        *mut u8,
                        *mut u8,
                        *const Barrier,
                        *mut Barrier,
                        *const Barrier,
                    ) = match stage {
                        0 => (
                            &raw mut SMEM_A0 as *mut u8,
                            &raw mut SMEM_B0 as *mut u8,
                            &raw const TMA_BAR0 as *const Barrier,
                            &raw mut TMA_BAR0 as *mut Barrier,
                            &raw const MMA_BAR0 as *const Barrier,
                        ),
                        1 => (
                            &raw mut SMEM_A1 as *mut u8,
                            &raw mut SMEM_B1 as *mut u8,
                            &raw const TMA_BAR1 as *const Barrier,
                            &raw mut TMA_BAR1 as *mut Barrier,
                            &raw const MMA_BAR1 as *const Barrier,
                        ),
                        2 => (
                            &raw mut SMEM_A2 as *mut u8,
                            &raw mut SMEM_B2 as *mut u8,
                            &raw const TMA_BAR2 as *const Barrier,
                            &raw mut TMA_BAR2 as *mut Barrier,
                            &raw const MMA_BAR2 as *const Barrier,
                        ),
                        _ => (
                            &raw mut SMEM_A3 as *mut u8,
                            &raw mut SMEM_B3 as *mut u8,
                            &raw const TMA_BAR3 as *const Barrier,
                            &raw mut TMA_BAR3 as *mut Barrier,
                            &raw const MMA_BAR3 as *const Barrier,
                        ),
                    };

                    // TMA warp waits for MMA to finish consuming the previous stage's data.
                    // This is an MMA barrier wait (not TMA) — safe for both CTAs because
                    // tcgen05_commit_multicast_cg2 multicasts to both CTAs' MMA barriers.
                    while !mbarrier_try_wait_parity(mma_bar_const, mma_parity) {}
                    // All lanes wait, then sync before lane 0 issues the loads
                    // that let MMA re-arm this barrier (no-stranding rule).
                    warp::sync_mask(u32::MAX);

                    // BARRIER ALIASING PROTOCOL (cta_group::2):
                    // PEER_BIT_MASK clears bit 24 of the barrier address, redirecting both
                    // CTAs' TMA completions to the leader's (rank 0) barrier. Therefore:
                    //   1. Only the leader arms the barrier with doubled expect_tx (both CTAs'
                    //      A+B tile bytes = 24576*2 = 49152).
                    //   2. Both CTAs still issue their own TMA loads (each contributes 24576 bytes).
                    if is_lane0 {
                        if elect_one_cta {
                            mbarrier_arrive_expect_tx(
                                tma_bar_const,
                                1,
                                (A_TILE_BYTES + B_TILE_BYTES) * 2,
                            );
                        }
                        let aliased_bar = ((tma_bar_mut as u32) & PEER_BIT_MASK) as *mut Barrier;
                        let k_base = (k_idx * 64) as i32;
                        cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                            smem_a_ptr,
                            a_tma,
                            k_base,
                            m_offset,
                            aliased_bar,
                            self_mask,
                        );
                        cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                            smem_b_ptr,
                            b_tma,
                            k_base,
                            b_n_offset,
                            aliased_bar,
                            self_mask,
                        );
                    }

                    k_idx += 1;
                }
                tile_seq += 1;

                // map_shared_rank translates a local SMEM pointer to the address in
                // the leader's shared memory, for use with mbarrier_arrive_cluster
                // (cross-CTA barrier arrive).
                let leader_clc_ready_addr = cvta_generic_to_shared_offset(
                    cluster::map_shared_rank(&raw const CLC_READY, 0) as *const u8,
                );

                let resp_ptr = &raw mut CLC_RESPONSE as *mut u64;
                loop {
                    let clc_parity = clc_iter & 1;

                    // Arm every CTA-local completion barrier before the leader can
                    // multicast the next response. CLC_READY also proves that both
                    // CTAs are still alive and have released their prior read.
                    //
                    // CLC writes the response through the async proxy. Keep all
                    // response access in lane 0, then broadcast registers to the
                    // warp so no lane can observe stale shared memory.
                    let mut is_canceled = 0u32;
                    let mut first_stolen = 0u32;
                    if is_lane0 {
                        mbarrier_arrive_expect_tx_cluster(&raw const CLC_BAR, 1, 16);
                        mbarrier_arrive_cluster(leader_clc_ready_addr);

                        if elect_one_cta {
                            while !mbarrier_try_wait_parity_cluster(
                                &raw const CLC_READY,
                                clc_parity,
                            ) {}
                            fence_proxy_async_generic_acquire_shared_cluster_cluster();
                            clc_try_cancel_multicast(resp_ptr as *mut u8, &raw mut CLC_BAR);
                        }

                        while !mbarrier_try_wait_parity_cluster(&raw const CLC_BAR, clc_parity) {}

                        let resp_lo = *resp_ptr;
                        let resp_hi = *resp_ptr.add(1);
                        is_canceled = clc_query_is_canceled(resp_lo, resp_hi);
                        if is_canceled != 0 {
                            first_stolen = clc_query_get_first_ctaid_x(resp_lo, resp_hi);
                        }
                        fence_proxy_async_generic_release_shared_cta_cluster();
                    }

                    is_canceled = warp::shuffle_sync(u32::MAX, is_canceled, 0);
                    first_stolen = warp::shuffle_sync(u32::MAX, first_stolen, 0);

                    if is_canceled == 0 {
                        if is_lane0 {
                            while !mbarrier_try_wait_parity(
                                &raw const TILE_INFO_FREE,
                                tile_info_free_parity,
                            ) {}
                            *(&raw mut TILE_INFO as *mut u32).add(2) = 0;
                            mbarrier_arrive(&raw const TILE_READY);
                        }
                        break;
                    }

                    // BUG FIX: Same cluster_size division as the initial tile. CLC returns a
                    // raw first_ctaid_x which represents the first CTA in the stolen cluster
                    // pair. Divide by 2 to get the tile index.
                    let tile_idx = first_stolen / 2;
                    // Same cache-blocking swizzle as the initial tile (see above).
                    let tiles_n = _tiles_n;
                    let group_tiles = SWIZZLE_G * tiles_m;
                    let group = tile_idx / group_tiles;
                    let in_group = tile_idx % group_tiles;
                    let n_start = group * SWIZZLE_G;
                    let band_w = if SWIZZLE_G < tiles_n - n_start {
                        SWIZZLE_G
                    } else {
                        tiles_n - n_start
                    };
                    let tile_m = in_group / band_w;
                    let tile_n = n_start + in_group % band_w;

                    if is_lane0 {
                        while !mbarrier_try_wait_parity(
                            &raw const TILE_INFO_FREE,
                            tile_info_free_parity,
                        ) {}
                        tile_info_free_parity ^= 1;
                        *(&raw mut TILE_INFO as *mut u32).add(0) = tile_m;
                        *(&raw mut TILE_INFO as *mut u32).add(1) = tile_n;
                        *(&raw mut TILE_INFO as *mut u32).add(2) = 1;
                        mbarrier_arrive(&raw const TILE_READY);
                    }

                    let m_off = (tile_m * 256 + my_rank * 128) as i32;
                    let b_n_off = (tile_n * 128 + my_rank * 64) as i32;

                    let mut k_idx: u32 = 0;
                    while k_idx < k_iters {
                        let global_k = tile_seq * k_iters + k_idx;
                        let stage = global_k & 3;
                        let mma_parity = (global_k >> 2) & 1;

                        let (smem_a_ptr, smem_b_ptr, tma_bar_const, tma_bar_mut, mma_bar_const): (
                            *mut u8,
                            *mut u8,
                            *const Barrier,
                            *mut Barrier,
                            *const Barrier,
                        ) = match stage {
                            0 => (
                                &raw mut SMEM_A0 as *mut u8,
                                &raw mut SMEM_B0 as *mut u8,
                                &raw const TMA_BAR0 as *const Barrier,
                                &raw mut TMA_BAR0 as *mut Barrier,
                                &raw const MMA_BAR0 as *const Barrier,
                            ),
                            1 => (
                                &raw mut SMEM_A1 as *mut u8,
                                &raw mut SMEM_B1 as *mut u8,
                                &raw const TMA_BAR1 as *const Barrier,
                                &raw mut TMA_BAR1 as *mut Barrier,
                                &raw const MMA_BAR1 as *const Barrier,
                            ),
                            2 => (
                                &raw mut SMEM_A2 as *mut u8,
                                &raw mut SMEM_B2 as *mut u8,
                                &raw const TMA_BAR2 as *const Barrier,
                                &raw mut TMA_BAR2 as *mut Barrier,
                                &raw const MMA_BAR2 as *const Barrier,
                            ),
                            _ => (
                                &raw mut SMEM_A3 as *mut u8,
                                &raw mut SMEM_B3 as *mut u8,
                                &raw const TMA_BAR3 as *const Barrier,
                                &raw mut TMA_BAR3 as *mut Barrier,
                                &raw const MMA_BAR3 as *const Barrier,
                            ),
                        };

                        while !mbarrier_try_wait_parity(mma_bar_const, mma_parity) {}
                        // All lanes wait, then sync before lane 0 issues the
                        // loads that let MMA re-arm this barrier
                        // (no-stranding rule).
                        warp::sync_mask(u32::MAX);

                        if is_lane0 {
                            if elect_one_cta {
                                mbarrier_arrive_expect_tx(
                                    tma_bar_const,
                                    1,
                                    (A_TILE_BYTES + B_TILE_BYTES) * 2,
                                );
                            }
                            let aliased_bar =
                                ((tma_bar_mut as u32) & PEER_BIT_MASK) as *mut Barrier;
                            let k_base = (k_idx * 64) as i32;
                            cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                smem_a_ptr,
                                a_tma,
                                k_base,
                                m_off,
                                aliased_bar,
                                self_mask,
                            );
                            cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                smem_b_ptr,
                                b_tma,
                                k_base,
                                b_n_off,
                                aliased_bar,
                                self_mask,
                            );
                        }

                        k_idx += 1;
                    }
                    tile_seq += 1;

                    clc_iter += 1;
                }
            }

            if warp_id == MMA_WARP {
                let is_lane0 = lane_id == 0;
                let mut tile_iter: u32 = 0;
                let mut tile_parity: u32 = 0;

                loop {
                    while !mbarrier_try_wait_parity(&raw const TILE_READY, tile_parity) {}
                    tile_parity ^= 1;

                    let has_work = *(&raw const TILE_INFO as *const u32).add(2);
                    warp::sync_mask(u32::MAX);
                    if is_lane0 {
                        mbarrier_arrive(&raw const TILE_INFO_FREE);
                    }
                    if has_work == 0 {
                        break;
                    }

                    let accum_stage = tile_iter % NUM_ACCUM_STAGES;
                    let tmem_stage_offset = accum_stage * ACCUM_STAGE_COLS;

                    // All leader-warp lanes may wait here: the next ACCUM_EMPTY
                    // phase needs an ACCUM_FULL commit from a later tile, which
                    // sits behind this warp's tile-top sync, so the barrier
                    // cannot advance past a lane that is still waiting on it.
                    if elect_one_cta && tile_iter >= NUM_ACCUM_STAGES {
                        let empty_parity = ((tile_iter - NUM_ACCUM_STAGES) / NUM_ACCUM_STAGES) & 1;
                        if accum_stage == 0 {
                            while !mbarrier_try_wait_parity(&raw const ACCUM_EMPTY0, empty_parity) {
                            }
                        } else {
                            while !mbarrier_try_wait_parity(&raw const ACCUM_EMPTY1, empty_parity) {
                            }
                        }
                    }

                    let tile_k_base = tile_iter * k_iters;
                    let mut k_idx: u32 = 0;
                    // Unroll one full pipeline cycle. The launch contract guarantees
                    // k_iters % 4 == 0, so the producer's global stage and this local
                    // stage agree at every tile boundary. Keeping this expression
                    // loop-local lets the unroll pass fold each stage match.
                    #[unroll(4)]
                    while k_idx < k_iters {
                        let global_k = tile_k_base + k_idx;
                        let stage = k_idx & 3;
                        let tma_parity = (global_k >> 2) & 1;

                        let (smem_a_base, smem_b_base, tma_bar_const, mma_bar_mut): (
                            u64,
                            u64,
                            *const Barrier,
                            *mut Barrier,
                        ) = match stage {
                            0 => (
                                cvta_generic_to_shared_offset(&raw const SMEM_A0 as *const u8),
                                cvta_generic_to_shared_offset(&raw const SMEM_B0 as *const u8),
                                &raw const TMA_BAR0 as *const Barrier,
                                &raw mut MMA_BAR0 as *mut Barrier,
                            ),
                            1 => (
                                cvta_generic_to_shared_offset(&raw const SMEM_A1 as *const u8),
                                cvta_generic_to_shared_offset(&raw const SMEM_B1 as *const u8),
                                &raw const TMA_BAR1 as *const Barrier,
                                &raw mut MMA_BAR1 as *mut Barrier,
                            ),
                            2 => (
                                cvta_generic_to_shared_offset(&raw const SMEM_A2 as *const u8),
                                cvta_generic_to_shared_offset(&raw const SMEM_B2 as *const u8),
                                &raw const TMA_BAR2 as *const Barrier,
                                &raw mut MMA_BAR2 as *mut Barrier,
                            ),
                            _ => (
                                cvta_generic_to_shared_offset(&raw const SMEM_A3 as *const u8),
                                cvta_generic_to_shared_offset(&raw const SMEM_B3 as *const u8),
                                &raw const TMA_BAR3 as *const Barrier,
                                &raw mut MMA_BAR3 as *mut Barrier,
                            ),
                        };

                        // LEADER-ONLY TMA WAIT + MMA:
                        // Because TMA completions are aliased to the leader's barrier, only
                        // the leader can (and should) wait on tma_bar_const. The follower's
                        // TMA barrier is never signaled -- pair-UMMA is issued by the leader
                        // and operates on both CTAs' SMEM simultaneously.
                        if elect_one_cta {
                            while !mbarrier_try_wait_parity(tma_bar_const, tma_parity) {}
                            // All lanes of the leader warp wait, then sync before
                            // lane 0 issues the MMAs whose commit re-arms this
                            // barrier: the barrier can then never advance past a
                            // lane still waiting on it, so no lane can strand one
                            // parity short (the seed-0 hang in the B200 mbarrier
                            // perturbation sweep).
                            warp::sync_mask(u32::MAX);

                            if is_lane0 {
                                let mut j: u32 = 0;
                                while j < 4 {
                                    let byte_offset = (j * 32) as u64;
                                    let a_desc = build_smem_descriptor(
                                        smem_a_base + byte_offset,
                                        LBO_BYTES,
                                        SBO_BYTES,
                                        SWIZZLE_128B,
                                    );
                                    let b_desc = build_smem_descriptor(
                                        smem_b_base + byte_offset,
                                        LBO_BYTES,
                                        SBO_BYTES,
                                        SWIZZLE_128B,
                                    );

                                    let accumulate = k_idx > 0 || j > 0;
                                    tcgen05_mma_f16_cg2(
                                        tmem_addr + tmem_stage_offset,
                                        a_desc,
                                        b_desc,
                                        idesc,
                                        accumulate,
                                    );
                                    j += 1;
                                }

                                tcgen05_commit_multicast_cg2(
                                    mma_bar_mut as *mut u64,
                                    CTA_MASK_PAIR,
                                );
                            }
                        }

                        k_idx += 1;
                    }

                    if elect_one_cta && is_lane0 {
                        if accum_stage == 0 {
                            tcgen05_commit_multicast_cg2(
                                &raw mut ACCUM_FULL0 as *mut u64,
                                CTA_MASK_PAIR,
                            );
                        } else {
                            tcgen05_commit_multicast_cg2(
                                &raw mut ACCUM_FULL1 as *mut u64,
                                CTA_MASK_PAIR,
                            );
                        }
                    }

                    tile_iter += 1;
                }

                tcgen05_relinquish_alloc_permit_cg2();
            }

            if warp_id < 4 {
                let is_lane0 = lane_id == 0;
                let mut epi_tile_iter: u32 = 0;
                let mut tile_parity: u32 = 0;

                let leader_accum_empty0_addr = cvta_generic_to_shared_offset(
                    cluster::map_shared_rank(&raw const ACCUM_EMPTY0, 0) as *const u8,
                );
                let leader_accum_empty1_addr = cvta_generic_to_shared_offset(
                    cluster::map_shared_rank(&raw const ACCUM_EMPTY1, 0) as *const u8,
                );

                const TILE_N: usize = 128;
                let warp_row_base = (warp_id * 32) as usize;
                let row_stride_bytes = TILE_N * 2;
                let row_within_8 = (lane_id % 8) as usize;
                let is_second_matrix = (8..16).contains(&lane_id);
                let col_offset_for_matrix2 = if is_second_matrix { 16usize } else { 0usize };

                loop {
                    while !mbarrier_try_wait_parity(&raw const TILE_READY, tile_parity) {}
                    tile_parity ^= 1;

                    let has_work = *(&raw const TILE_INFO as *const u32).add(2);
                    let tile_m = *(&raw const TILE_INFO as *const u32).add(0);
                    let tile_n = *(&raw const TILE_INFO as *const u32).add(1);
                    warp::sync_mask(u32::MAX);
                    if is_lane0 {
                        mbarrier_arrive(&raw const TILE_INFO_FREE);
                    }
                    if has_work == 0 {
                        break;
                    }

                    let accum_stage = epi_tile_iter % NUM_ACCUM_STAGES;
                    let tmem_stage_offset = accum_stage * ACCUM_STAGE_COLS;

                    let full_parity = (epi_tile_iter / NUM_ACCUM_STAGES) & 1;
                    if accum_stage == 0 {
                        while !mbarrier_try_wait_parity(&raw const ACCUM_FULL0, full_parity) {}
                    } else {
                        while !mbarrier_try_wait_parity(&raw const ACCUM_FULL1, full_parity) {}
                    }

                    let mut tmem_row_block = 0u32;
                    while tmem_row_block < 2 {
                        let tmem_row = warp_id * 32 + tmem_row_block * 16;

                        let mut col_block = 0u32;
                        while col_block < 8 {
                            let col_offset = (col_block * 16) as usize;

                            let regs_a = tcgen05_ld_16x256b_pure(
                                tmem_addr
                                    + tmem_stage_offset
                                    + (tmem_row << 16)
                                    + col_offset as u32,
                            );
                            tcgen05_load_wait();

                            let regs_b = tcgen05_ld_16x256b_pure(
                                tmem_addr
                                    + tmem_stage_offset
                                    + (tmem_row << 16)
                                    + col_offset as u32
                                    + 8,
                            );
                            tcgen05_load_wait();

                            let p0_lo = cvt_bf16x2_f32(regs_a[0], regs_a[1]);
                            let p1_lo = cvt_bf16x2_f32(regs_b[0], regs_b[1]);
                            let out_row_lo =
                                warp_row_base + (tmem_row_block as usize * 16) + row_within_8;
                            let smem_addr_lo = (&raw mut SMEM_OUT as *mut u8).add(
                                out_row_lo * row_stride_bytes
                                    + col_offset * 2
                                    + col_offset_for_matrix2,
                            );
                            stmatrix_m8n8_x2(smem_addr_lo, p0_lo, p1_lo);

                            let p0_hi = cvt_bf16x2_f32(regs_a[2], regs_a[3]);
                            let p1_hi = cvt_bf16x2_f32(regs_b[2], regs_b[3]);
                            let out_row_hi =
                                warp_row_base + (tmem_row_block as usize * 16) + 8 + row_within_8;
                            let smem_addr_hi = (&raw mut SMEM_OUT as *mut u8).add(
                                out_row_hi * row_stride_bytes
                                    + col_offset * 2
                                    + col_offset_for_matrix2,
                            );
                            stmatrix_m8n8_x2(smem_addr_hi, p0_hi, p1_hi);

                            col_block += 1;
                        }
                        tmem_row_block += 1;
                    }

                    let n_u32 = (n / 2) as usize;
                    let tile_row_base = (tile_m * 256 + my_rank * 128) as usize;
                    let tile_col_base = (tile_n * 64) as usize;
                    let base_row = warp_id as usize * 32;

                    let mut elem = lane_id as usize;
                    while elem < 2048 {
                        let local_row = elem / 64;
                        let local_col = elem % 64;
                        let smem_idx = (base_row + local_row) * 64 + local_col;
                        let global_row = tile_row_base + base_row + local_row;
                        let global_col = tile_col_base + local_col;
                        let global_idx = global_row * n_u32 + global_col;

                        *out.get_unchecked_mut(global_idx) = SMEM_OUT[smem_idx];
                        elem += 32;
                    }

                    if elect_one_cta {
                        if accum_stage == 0 {
                            mbarrier_arrive(&raw const ACCUM_EMPTY0);
                        } else {
                            mbarrier_arrive(&raw const ACCUM_EMPTY1);
                        }
                    } else {
                        if accum_stage == 0 {
                            mbarrier_arrive_cluster(leader_accum_empty0_addr);
                        } else {
                            mbarrier_arrive_cluster(leader_accum_empty1_addr);
                        }
                    }

                    epi_tile_iter += 1;
                }
            }

            // BUG FIX: cluster_sync before exit prevents "Cluster target block not present"
            // (CUDA_EXCEPTION_17). Without this, a fast CTA can exit while its partner is still
            // executing cross-CTA operations (e.g., mbarrier_arrive_cluster in the epilogue).
            cluster::cluster_sync();

            if warp_id == 0 {
                tcgen05_dealloc_cg2(tmem_addr, 512);
            }
            if tid == 0 {
                mbarrier_inval(&raw mut TMA_BAR0);
                mbarrier_inval(&raw mut TMA_BAR1);
                mbarrier_inval(&raw mut TMA_BAR2);
                mbarrier_inval(&raw mut TMA_BAR3);
                mbarrier_inval(&raw mut MMA_BAR0);
                mbarrier_inval(&raw mut MMA_BAR1);
                mbarrier_inval(&raw mut MMA_BAR2);
                mbarrier_inval(&raw mut MMA_BAR3);
                mbarrier_inval(&raw mut ACCUM_FULL0);
                mbarrier_inval(&raw mut ACCUM_FULL1);
                mbarrier_inval(&raw mut ACCUM_EMPTY0);
                mbarrier_inval(&raw mut ACCUM_EMPTY1);
                mbarrier_inval(&raw mut TILE_READY);
                mbarrier_inval(&raw mut TILE_INFO_FREE);
                mbarrier_inval(&raw mut CLC_BAR);
                mbarrier_inval(&raw mut CLC_READY);
            }
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("  GEMM Speed-of-Light — All Phases");
    println!("═══════════════════════════════════════════════════════\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let (major, minor) = ctx.compute_capability()?;
    println!("GPU: sm_{}{}", major, minor);

    // Optional phase filter for isolating CLC kernels while debugging. The normal
    // no-variable path below still runs the complete benchmark suite.
    let phase_filter = std::env::var("GEMM_SOL_PHASE").ok();

    // Run the cublasLt baseline once up front so the ~25s measurement isn't
    // sandwiched between benchmark prints. Skipped silently if the bench
    // binary isn't built (the per-phase reports will omit the % SoL column).
    if phase_filter.is_none() {
        cublas_baseline::warmup();
    }

    // Gate on the GPUs that can actually execute this module BEFORE trying
    // to load it (same set as gemm_sol_final; keep in sync with
    // mir-importer's tcgen05 target support). Deciding "wrong GPU" from a
    // module-load failure is not sound: the driver reports arch-incompatible
    // PTX and genuinely malformed PTX with the same CUDA_ERROR_INVALID_PTX,
    // so a load-error fallback silently converts compiler bugs (e.g.
    // `.global` initializers referencing `.shared` symbols) into a
    // PTX-only "pass".
    if !can_execute_tcgen05_ptx(major, minor) {
        println!("\nWARNING: tcgen05 requires sm_100 (datacenter Blackwell)");
        if major >= 10 {
            println!(
                "   Your GPU is sm_{}{} (consumer Blackwell has no tcgen05).",
                major, minor
            );
        } else {
            println!("   Your GPU is sm_{}{} (pre-Blackwell).", major, minor);
        }
        println!("   PTX was generated successfully; run on sm_100 to execute kernels.");
        return verify_ptx_only();
    }

    let module = match kernels::load(&ctx) {
        Ok(m) => m,
        Err(e) => {
            // This GPU passed the capability gate above, so the module must
            // load. CUDA_ERROR_INVALID_PTX here means the driver rejected
            // the generated PTX itself: a compiler bug, never a "wrong GPU"
            // situation. Fail loudly instead of degrading to the PTX-only
            // verification path.
            let driver_status = match &e {
                cuda_host::EmbeddedModuleError::Driver(driver) => Some(driver.0),
                _ => None,
            };
            if driver_status == Some(cuda_core::sys::cudaError_enum_CUDA_ERROR_INVALID_PTX) {
                return Err(format!(
                    "driver rejected the generated PTX as invalid on sm_{major}{minor}, \
                     which should execute it (CUDA_ERROR_INVALID_PTX): {e:?}"
                )
                .into());
            }
            return Err(e.into());
        }
    };
    println!("PTX loaded\n");

    let sizes: [(usize, usize, usize); 3] = [
        (4096, 4096, 4096),
        (8192, 8192, 8192),
        (16384, 16384, 16384),
    ];

    if let Some(phase) = phase_filter.as_deref() {
        match phase {
            "4a-correctness" => {
                run_correctness_test_persistent(&stream, &module, 4096, 4096, 4096)?;
            }
            "4a" => {
                run_correctness_test_persistent(&stream, &module, 4096, 4096, 4096)?;
                for (m, n, k) in sizes {
                    run_benchmark_persistent(&stream, &module, m, n, k)?;
                }
            }
            "4b-correctness" => {
                run_correctness_test_clc(&stream, &module, 4096, 4096, 4096)?;
            }
            "4b" => {
                run_correctness_test_clc(&stream, &module, 4096, 4096, 4096)?;
                for (m, n, k) in sizes {
                    run_benchmark_clc(&stream, &module, m, n, k)?;
                }
            }
            "4c-correctness" => {
                run_correctness_test_clc_multicast(&stream, &module, 4096, 4096, 4096)?;
            }
            "4c" => {
                run_correctness_test_clc_multicast(&stream, &module, 4096, 4096, 4096)?;
                for (m, n, k) in sizes {
                    run_benchmark_clc_multicast(&stream, &module, m, n, k)?;
                }
            }
            "4d-correctness" => {
                run_correctness_test_clc_multicast_4_stage_pipeline(
                    &stream, &module, 4096, 4096, 4096,
                )?;
            }
            "4d" => {
                run_correctness_test_clc_multicast_4_stage_pipeline(
                    &stream, &module, 4096, 4096, 4096,
                )?;
                for (m, n, k) in sizes {
                    run_benchmark_clc_multicast_4_stage_pipeline(&stream, &module, m, n, k)?;
                }
            }
            _ => {
                return Err(format!(
                    "unknown GEMM_SOL_PHASE={phase:?}; expected 4a, 4a-correctness, 4b, \
                     4b-correctness, 4c, 4c-correctness, 4d, or 4d-correctness"
                )
                .into());
            }
        }
        return Ok(());
    }

    // Run the complete suite by default. GEMM_SOL_PHASE above can isolate a CLC
    // phase for targeted correctness and repeated-launch testing.
    if true {
        // ── Phase 1: K-loop + grid tiling ──
        println!("\n\n═══════════════════════════════════════════════════════");
        println!("  Phase 1: K-loop + Grid Tiling (gemm_sol_tiled)");
        println!("═══════════════════════════════════════════════════════\n");

        println!("── Correctness Test ─────────────────────────────────\n");
        run_correctness_test(&stream, &module, 4096, 4096, 4096)?;

        println!("\n── Benchmarks ──────────────────────────────────────-\n");
        for (m, n, k) in sizes {
            if let Err(e) = run_benchmark(&stream, &module, m, n, k) {
                eprintln!("Benchmark (tiled) {}x{}x{} failed: {}", m, n, k, e);
            }
        }

        // ── Phase 1.5: SWIZZLE_128B ──
        println!("\n\n═══════════════════════════════════════════════════════");
        println!("  Phase 1.5: SWIZZLE_128B (gemm_sol_swizzled)");
        println!("═══════════════════════════════════════════════════════\n");

        println!("── Correctness Test ─────────────────────────────────\n");
        run_correctness_test_swizzled(&stream, &module, 4096, 4096, 4096)?;

        println!("\n── Benchmarks ──────────────────────────────────────-\n");
        for (m, n, k) in sizes {
            if let Err(e) = run_benchmark_swizzled(&stream, &module, m, n, k) {
                eprintln!("Benchmark (swizzled) {}x{}x{} failed: {}", m, n, k, e);
            }
        }

        // ── Phase 2: Double-buffered SMEM pipeline ──
        println!("\n\n═══════════════════════════════════════════════════════");
        println!("  Phase 2: Double-Buffered Pipeline (gemm_sol_pipelined)");
        println!("═══════════════════════════════════════════════════════\n");

        println!("── Correctness Test ─────────────────────────────────\n");
        run_correctness_test_pipelined(&stream, &module, 4096, 4096, 4096)?;

        println!("\n── Benchmarks ──────────────────────────────────────-\n");
        for (m, n, k) in sizes {
            if let Err(e) = run_benchmark_pipelined(&stream, &module, m, n, k) {
                eprintln!("Benchmark (pipelined) {}x{}x{} failed: {}", m, n, k, e);
            }
        }

        // ── Phase 3: Warp-specialized producer/consumer ──
        println!("\n\n═══════════════════════════════════════════════════════");
        println!("  Phase 3: Warp Specialization (gemm_sol_warp_spec)");
        println!("═══════════════════════════════════════════════════════\n");

        println!("── Correctness Test ─────────────────────────────────\n");
        run_correctness_test_warp_spec(&stream, &module, 4096, 4096, 4096)?;

        println!("\n── Benchmarks ──────────────────────────────────────-\n");
        for (m, n, k) in sizes {
            if let Err(e) = run_benchmark_warp_spec(&stream, &module, m, n, k) {
                eprintln!("Benchmark (warp_spec) {}x{}x{} failed: {}", m, n, k, e);
            }
        }

        // ── Phase 4A: Persistent + TMEM accumulator pipeline ──
        println!("\n\n═══════════════════════════════════════════════════════");
        println!("  Phase 4A: Persistent Kernel (gemm_sol_persistent)");
        println!("═══════════════════════════════════════════════════════\n");

        println!("── Correctness Test ─────────────────────────────────\n");
        run_correctness_test_persistent(&stream, &module, 4096, 4096, 4096)?;

        println!("\n── Benchmarks ──────────────────────────────────────-\n");
        for (m, n, k) in sizes {
            if let Err(e) = run_benchmark_persistent(&stream, &module, m, n, k) {
                eprintln!("Benchmark (persistent) {}x{}x{} failed: {}", m, n, k, e);
            }
        }

        // ── Phase 4B: CLC tile scheduling (no multicast) ──
        println!("\n\n═══════════════════════════════════════════════════════");
        println!("  Phase 4B: CLC Tile Scheduling (gemm_sol_clc)");
        println!("═══════════════════════════════════════════════════════\n");

        println!("── Correctness Test ─────────────────────────────────\n");
        run_correctness_test_clc(&stream, &module, 4096, 4096, 4096)?;

        println!("\n── Benchmarks ──────────────────────────────────────-\n");
        for (m, n, k) in sizes {
            if let Err(e) = run_benchmark_clc(&stream, &module, m, n, k) {
                eprintln!("Benchmark (clc) {}x{}x{} failed: {}", m, n, k, e);
            }
        }

        // ── Phase 4C: CLC + TMA multicast ──
        println!("\n\n═══════════════════════════════════════════════════════");
        println!("  Phase 4C: CLC + TMA Multicast (gemm_sol_clc_multicast)");
        println!("═══════════════════════════════════════════════════════\n");

        println!("── Correctness Test ─────────────────────────────────\n");
        run_correctness_test_clc_multicast(&stream, &module, 4096, 4096, 4096)?;

        println!("\n── Benchmarks ──────────────────────────────────────-\n");
        for (m, n, k) in sizes {
            if let Err(e) = run_benchmark_clc_multicast(&stream, &module, m, n, k) {
                eprintln!("Benchmark (clc_multicast) {}x{}x{} failed: {}", m, n, k, e);
            }
        }

        // ── Phase 4D: 4-stage pipeline + CTA pairs + TMA multicast + CLC + no MCAST_BAR ──
        println!("\n\n═══════════════════════════════════════════════════════");
        println!("  Phase 4D: CLC + 4-stage multicast (no MCAST_BAR)");
        println!("═══════════════════════════════════════════════════════\n");

        println!("── Correctness Test ─────────────────────────────────\n");
        run_correctness_test_clc_multicast_4_stage_pipeline(&stream, &module, 4096, 4096, 4096)?;

        println!("\n── Benchmarks ──────────────────────────────────────-\n");
        for (m, n, k) in sizes {
            if let Err(e) = run_benchmark_clc_multicast_4_stage_pipeline(&stream, &module, m, n, k)
            {
                eprintln!(
                    "Benchmark (clc_multicast_4_stage_pipeline) {}x{}x{} failed: {}",
                    m, n, k, e
                );
            }
        }
    } // end all phases

    println!("\n═══════════════════════════════════════════════════════");
    println!("  GEMM SoL — All Phases Complete");
    println!("═══════════════════════════════════════════════════════");

    Ok(())
}

/// Run correctness test: A=1.0, B=1.0 → C[i,j] = K for all (i,j).
fn run_correctness_test(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    assert!(m.is_multiple_of(128) && n.is_multiple_of(128) && k.is_multiple_of(64));

    println!("Matrix: {}x{}x{} (f16 -> bf16)", m, n, k);
    println!("No pre-tiling. Flat row-major buffers.");

    // A: M×K f16, A[i,k] = (i%8+1) (K contiguous, row-major)
    let mut host_a: Vec<u16> = vec![0u16; m * k];
    for i in 0..m {
        let val = f16::from_f32((i % 8 + 1) as f32).to_bits();
        for kk in 0..k {
            host_a[i * k + kk] = val;
        }
    }
    // B: N×K f16, B[n,k] = (n%8+1) (transposed storage: N×K, K contiguous)
    let mut host_b: Vec<u16> = vec![0u16; n * k];
    for j in 0..n {
        let val = f16::from_f32((j % 8 + 1) as f32).to_bits();
        for kk in 0..k {
            host_b[j * k + kk] = val;
        }
    }

    println!("A[i,k] = (i%8+1), B[n,k] = (n%8+1)");
    println!("Expected: C[i,j] = (i%8+1)*(j%8+1)*K\n");

    // Upload to GPU
    let dev_a = DeviceBuffer::from_host(stream, &host_a)?;
    let dev_b = DeviceBuffer::from_host(stream, &host_b)?;

    let a_ptr = dev_a.cu_deviceptr();
    let b_ptr = dev_b.cu_deviceptr();

    // Create TMA descriptors for full global tensors
    // TMA box = [8, 128]: 8 K-elements × 128 M/N rows per copy
    // Kernel issues 8 copies per K-tile to fill the 128×64 SMEM tile
    let a_tma =
        create_tma_descriptor_f16(a_ptr as *mut std::ffi::c_void, k as u64, m as u64, 8, 128)?;
    let b_tma =
        create_tma_descriptor_f16(b_ptr as *mut std::ffi::c_void, k as u64, n as u64, 8, 128)?;

    let dev_a_tma = DeviceBuffer::from_host(stream, &a_tma.opaque)?;
    let dev_b_tma = DeviceBuffer::from_host(stream, &b_tma.opaque)?;

    // Grid tiling: one CTA per 128×128 output tile
    let grid_m = (m / 128) as u32;
    let grid_n = (n / 128) as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid_m, grid_n, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };

    println!(
        "Grid: {}x{} CTAs ({} total), block: 128 threads",
        grid_m,
        grid_n,
        grid_m * grid_n
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

    println!("\nLaunching gemm_sol_tiled...");

    unsafe {
        module.gemm_sol_tiled(
            (stream).as_ref(),
            cfg,
            a_tma_ptr,
            b_tma_ptr,
            &mut dev_output,
            n_arg,
            k_arg,
        )
    }?;

    stream.synchronize()?;

    // Verify a sample of output values
    let host_output: Vec<u32> = dev_output.to_host_vec(stream)?;

    // Expected value at (row, col): C[i,j] = (i%8+1)*(j%8+1)*K
    let expected = |row: usize, col: usize| -> f32 { ((row % 8 + 1) * (col % 8 + 1) * k) as f32 };
    let read_c = |row: usize, col: usize| -> f32 {
        let packed_idx = row * (n / 2) + col / 2;
        let packed = host_output[packed_idx];
        let (lo, hi) = unpack_bf16_pair(packed);
        if col.is_multiple_of(2) { lo } else { hi }
    };

    // ── Optional: dump 16×16 sub-matrices (comment out when not needed) ──
    // print_16x16_got_vs_expected(&host_output, n, k);

    // ── Spot checks: corners, centre, random positions ──
    let check_positions = [
        (0, 0),
        (0, n - 1),
        (m - 1, 0),
        (m - 1, n - 1),
        (m / 2, n / 2),
        (3, 5),
        (7, 7),
        (127, 127),
    ];

    let mut all_ok = true;
    println!("\nSpot checks:");
    for (row, col) in check_positions {
        let val = read_c(row, col);
        let exp = expected(row, col);
        let ok = (val - exp).abs() < (exp * 0.02 + 1.0);
        println!(
            "  C[{:>4},{:>4}] = {:>10.0}  (expected {:>10.0})  {}",
            row,
            col,
            val,
            exp,
            if ok { "OK" } else { "FAIL" }
        );
        if !ok {
            all_ok = false;
        }
    }

    // ── Sum of first row ──
    let mut first_row_sum: f64 = 0.0;
    for col_pair in 0..(n / 2) {
        let packed = host_output[col_pair];
        let (lo, hi) = unpack_bf16_pair(packed);
        first_row_sum += lo as f64 + hi as f64;
    }
    // Row 0: C[0,j] = 1*(j%8+1)*K. Sum = K * sum_{j=0..N-1}(j%8+1) = K * (N/8) * 36
    let expected_row_sum = k as f64 * (n as f64 / 8.0) * 36.0;
    let row_sum_ok =
        (first_row_sum - expected_row_sum).abs() < (expected_row_sum * 0.02 + n as f64);
    println!(
        "\n  Row 0 sum: {:.0} (expected {:.0}) {}",
        first_row_sum,
        expected_row_sum,
        if row_sum_ok { "OK" } else { "FAIL" }
    );

    println!("\n═══════════════════════════════════════════════════════");
    if all_ok && row_sum_ok {
        println!("PASSED: gemm_sol {}x{}x{}", m, n, k);
    } else {
        println!("FAILED: gemm_sol {}x{}x{}", m, n, k);
        return Err("Correctness check failed".into());
    }
    println!("═══════════════════════════════════════════════════════");

    Ok(())
}

#[allow(dead_code)]
/// Print the first 16×16 sub-matrix of the GOT output alongside the EXPECTED output.
/// Useful for visual debugging. Call from `run_correctness_test` and comment out when not needed.
fn print_16x16_got_vs_expected(host_output: &[u32], n: usize, k: usize) {
    let read_c = |row: usize, col: usize| -> f32 {
        let packed_idx = row * (n / 2) + col / 2;
        let packed = host_output[packed_idx];
        let (lo, hi) = unpack_bf16_pair(packed);
        if col.is_multiple_of(2) { lo } else { hi }
    };

    println!("\n── Got (first 16 rows × 16 cols) ──");
    print!("       ");
    for col in 0..16 {
        print!("{:>8}", col);
    }
    println!();
    for row in 0..16 {
        print!("  r{:2}: ", row);
        for col in 0..16 {
            print!("{:8.0}", read_c(row, col));
        }
        println!();
    }

    println!("\n── Expected (first 16 rows × 16 cols) ──");
    print!("       ");
    for col in 0..16 {
        print!("{:>8}", col);
    }
    println!();
    for row in 0..16 {
        print!("  r{:2}: ", row);
        for col in 0..16 {
            let expected_val = ((row % 8 + 1) * (col % 8 + 1) * k) as f32;
            print!("{:8.0}", expected_val);
        }
        println!();
    }
}

/// Benchmark the GEMM kernel with CUDA event timing.
fn run_benchmark(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    const WARMUP: usize = 10;
    const ITERS: usize = 100;

    assert!(m.is_multiple_of(128) && n.is_multiple_of(128) && k.is_multiple_of(64));

    // Allocate device memory (zeros — values don't matter for timing)
    let dev_a = DeviceBuffer::<u16>::zeroed(stream, m * k)?;
    let dev_b = DeviceBuffer::<u16>::zeroed(stream, n * k)?;

    let a_ptr = dev_a.cu_deviceptr();
    let b_ptr = dev_b.cu_deviceptr();

    // TMA descriptors (8 K-elems × 128 M/N rows per copy)
    let a_tma =
        create_tma_descriptor_f16(a_ptr as *mut std::ffi::c_void, k as u64, m as u64, 8, 128)?;
    let b_tma =
        create_tma_descriptor_f16(b_ptr as *mut std::ffi::c_void, k as u64, n as u64, 8, 128)?;

    let dev_a_tma = DeviceBuffer::from_host(stream, &a_tma.opaque)?;
    let dev_b_tma = DeviceBuffer::from_host(stream, &b_tma.opaque)?;
    let a_tma_ptr = dev_a_tma.cu_deviceptr();
    let b_tma_ptr = dev_b_tma.cu_deviceptr();
    let a_tma_ptr = a_tma_ptr as *const TmaDescriptor;
    let b_tma_ptr = b_tma_ptr as *const TmaDescriptor;
    let n_arg = n as i32;
    let k_arg = k as i32;

    let grid_m = (m / 128) as u32;
    let grid_n = (n / 128) as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid_m, grid_n, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };

    let output_u32_count = m * n / 2;
    let mut dev_output = DeviceBuffer::<u32>::zeroed(stream, output_u32_count)?;

    // ── Warmup ──
    for _ in 0..WARMUP {
        unsafe {
            module.gemm_sol_tiled(
                (stream).as_ref(),
                cfg,
                a_tma_ptr,
                b_tma_ptr,
                &mut dev_output,
                n_arg,
                k_arg,
            )
        }?;
    }
    stream.synchronize()?;

    // ── Timed iterations with CUDA events ──
    let start_event =
        stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;

    for _ in 0..ITERS {
        unsafe {
            module.gemm_sol_tiled(
                (stream).as_ref(),
                cfg,
                a_tma_ptr,
                b_tma_ptr,
                &mut dev_output,
                n_arg,
                k_arg,
            )
        }?;
    }

    let end_event =
        stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let elapsed_ms = start_event.elapsed_ms(&end_event)?;

    // ── Results ──
    let avg_ms = elapsed_ms as f64 / ITERS as f64;
    let avg_us = avg_ms * 1000.0;
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let tflops = (flops / (avg_ms / 1000.0)) / 1e12;

    // cuBLAS SoL is now measured live via bench/cublaslt_bench (parsed by
    // the cublas_baseline module). The comparison line is printed by
    // print_cublas_comparison(tflops, m) below.

    println!("═══════════════════════════════════════════════════════");
    println!("  BENCHMARK: gemm_sol {}x{}x{} f16 -> bf16", m, n, k);
    println!("═══════════════════════════════════════════════════════");
    println!(
        "  Grid:        {}x{} CTAs ({} total)",
        grid_m,
        grid_n,
        grid_m * grid_n
    );
    println!("  K-loop:      {} outer iters (BK=64, 4 MMAs each)", k / 64);
    println!("  Iterations:  {} (after {} warmup)", ITERS, WARMUP);
    println!("  Total time:  {:.3} ms", elapsed_ms);
    println!("  Average:     {:.3} us / kernel", avg_us);
    println!("  FLOPS/kern:  {:.3e}", flops);
    println!("  Throughput:  {:.3} TFLOPS", tflops);
    print_cublas_comparison(tflops, m);
    println!("═══════════════════════════════════════════════════════\n");

    Ok(())
}

/// Run correctness test for the SWIZZLE_128B kernel.
fn run_correctness_test_swizzled(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    assert!(m.is_multiple_of(128) && n.is_multiple_of(128) && k.is_multiple_of(64));

    println!("Matrix: {}x{}x{} (f16 -> bf16)", m, n, k);
    println!("SWIZZLE_128B: single TMA copy per matrix per K-tile.");

    let mut host_a: Vec<u16> = vec![0u16; m * k];
    for i in 0..m {
        let val = f16::from_f32((i % 8 + 1) as f32).to_bits();
        for kk in 0..k {
            host_a[i * k + kk] = val;
        }
    }
    let mut host_b: Vec<u16> = vec![0u16; n * k];
    for j in 0..n {
        let val = f16::from_f32((j % 8 + 1) as f32).to_bits();
        for kk in 0..k {
            host_b[j * k + kk] = val;
        }
    }

    println!("A[i,k] = (i%8+1), B[n,k] = (n%8+1)");
    println!("Expected: C[i,j] = (i%8+1)*(j%8+1)*K\n");

    let dev_a = DeviceBuffer::from_host(stream, &host_a)?;
    let dev_b = DeviceBuffer::from_host(stream, &host_b)?;

    let a_ptr = dev_a.cu_deviceptr();
    let b_ptr = dev_b.cu_deviceptr();

    let a_tma =
        create_tma_descriptor_f16_swizzled(a_ptr as *mut std::ffi::c_void, k as u64, m as u64)?;
    let b_tma =
        create_tma_descriptor_f16_swizzled(b_ptr as *mut std::ffi::c_void, k as u64, n as u64)?;

    let dev_a_tma = DeviceBuffer::from_host(stream, &a_tma.opaque)?;
    let dev_b_tma = DeviceBuffer::from_host(stream, &b_tma.opaque)?;

    let grid_m = (m / 128) as u32;
    let grid_n = (n / 128) as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid_m, grid_n, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };

    println!(
        "Grid: {}x{} CTAs ({} total), block: 128 threads",
        grid_m,
        grid_n,
        grid_m * grid_n
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

    println!("\nLaunching gemm_sol_swizzled...");

    unsafe {
        module.gemm_sol_swizzled(
            (stream).as_ref(),
            cfg,
            a_tma_ptr,
            b_tma_ptr,
            &mut dev_output,
            n_arg,
            k_arg,
        )
    }?;

    stream.synchronize()?;

    let host_output: Vec<u32> = dev_output.to_host_vec(stream)?;

    let expected = |row: usize, col: usize| -> f32 { ((row % 8 + 1) * (col % 8 + 1) * k) as f32 };
    let read_c = |row: usize, col: usize| -> f32 {
        let packed_idx = row * (n / 2) + col / 2;
        let packed = host_output[packed_idx];
        let (lo, hi) = unpack_bf16_pair(packed);
        if col.is_multiple_of(2) { lo } else { hi }
    };

    // ── Optional: dump 16×16 sub-matrices (comment out when not needed) ──
    // print_16x16_got_vs_expected(&host_output, n, k);

    let check_positions = [
        (0, 0),
        (0, n - 1),
        (m - 1, 0),
        (m - 1, n - 1),
        (m / 2, n / 2),
        (3, 5),
        (7, 7),
        (127, 127),
    ];

    let mut all_ok = true;
    println!("\nSpot checks:");
    for (row, col) in check_positions {
        let val = read_c(row, col);
        let exp = expected(row, col);
        let ok = (val - exp).abs() < (exp * 0.02 + 1.0);
        println!(
            "  C[{:>4},{:>4}] = {:>10.0}  (expected {:>10.0})  {}",
            row,
            col,
            val,
            exp,
            if ok { "OK" } else { "FAIL" }
        );
        if !ok {
            all_ok = false;
        }
    }

    let mut first_row_sum: f64 = 0.0;
    for col_pair in 0..(n / 2) {
        let packed = host_output[col_pair];
        let (lo, hi) = unpack_bf16_pair(packed);
        first_row_sum += lo as f64 + hi as f64;
    }
    let expected_row_sum = k as f64 * (n as f64 / 8.0) * 36.0;
    let row_sum_ok =
        (first_row_sum - expected_row_sum).abs() < (expected_row_sum * 0.02 + n as f64);
    println!(
        "\n  Row 0 sum: {:.0} (expected {:.0}) {}",
        first_row_sum,
        expected_row_sum,
        if row_sum_ok { "OK" } else { "FAIL" }
    );

    println!("\n═══════════════════════════════════════════════════════");
    if all_ok && row_sum_ok {
        println!("PASSED: gemm_sol_swizzled {}x{}x{}", m, n, k);
    } else {
        println!("FAILED: gemm_sol_swizzled {}x{}x{}", m, n, k);
        return Err("Correctness check failed".into());
    }
    println!("═══════════════════════════════════════════════════════");

    Ok(())
}

/// Benchmark the SWIZZLE_128B GEMM kernel with CUDA event timing.
fn run_benchmark_swizzled(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    const WARMUP: usize = 10;
    const ITERS: usize = 100;

    assert!(m.is_multiple_of(128) && n.is_multiple_of(128) && k.is_multiple_of(64));

    let dev_a = DeviceBuffer::<u16>::zeroed(stream, m * k)?;
    let dev_b = DeviceBuffer::<u16>::zeroed(stream, n * k)?;

    let a_ptr = dev_a.cu_deviceptr();
    let b_ptr = dev_b.cu_deviceptr();

    let a_tma =
        create_tma_descriptor_f16_swizzled(a_ptr as *mut std::ffi::c_void, k as u64, m as u64)?;
    let b_tma =
        create_tma_descriptor_f16_swizzled(b_ptr as *mut std::ffi::c_void, k as u64, n as u64)?;

    let dev_a_tma = DeviceBuffer::from_host(stream, &a_tma.opaque)?;
    let dev_b_tma = DeviceBuffer::from_host(stream, &b_tma.opaque)?;
    let a_tma_ptr = dev_a_tma.cu_deviceptr();
    let b_tma_ptr = dev_b_tma.cu_deviceptr();
    let a_tma_ptr = a_tma_ptr as *const TmaDescriptor;
    let b_tma_ptr = b_tma_ptr as *const TmaDescriptor;
    let n_arg = n as i32;
    let k_arg = k as i32;

    let grid_m = (m / 128) as u32;
    let grid_n = (n / 128) as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid_m, grid_n, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };

    let output_u32_count = m * n / 2;
    let mut dev_output = DeviceBuffer::<u32>::zeroed(stream, output_u32_count)?;

    for _ in 0..WARMUP {
        unsafe {
            module.gemm_sol_swizzled(
                (stream).as_ref(),
                cfg,
                a_tma_ptr,
                b_tma_ptr,
                &mut dev_output,
                n_arg,
                k_arg,
            )
        }?;
    }
    stream.synchronize()?;

    let start_event =
        stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;

    for _ in 0..ITERS {
        unsafe {
            module.gemm_sol_swizzled(
                (stream).as_ref(),
                cfg,
                a_tma_ptr,
                b_tma_ptr,
                &mut dev_output,
                n_arg,
                k_arg,
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

    // cuBLAS SoL is now measured live via bench/cublaslt_bench (parsed by
    // the cublas_baseline module). The comparison line is printed by
    // print_cublas_comparison(tflops, m) below.

    println!("═══════════════════════════════════════════════════════");
    println!(
        "  BENCHMARK: gemm_sol_swizzled {}x{}x{} f16 -> bf16",
        m, n, k
    );
    println!("═══════════════════════════════════════════════════════");
    println!(
        "  Grid:        {}x{} CTAs ({} total)",
        grid_m,
        grid_n,
        grid_m * grid_n
    );
    println!("  K-loop:      {} outer iters (BK=64, 4 MMAs each)", k / 64);
    println!("  TMA:         1 copy/matrix (SWIZZLE_128B)");
    println!("  Iterations:  {} (after {} warmup)", ITERS, WARMUP);
    println!("  Total time:  {:.3} ms", elapsed_ms);
    println!("  Average:     {:.3} us / kernel", avg_us);
    println!("  FLOPS/kern:  {:.3e}", flops);
    println!("  Throughput:  {:.3} TFLOPS", tflops);
    print_cublas_comparison(tflops, m);
    println!("═══════════════════════════════════════════════════════\n");

    Ok(())
}

/// Run correctness test for the double-buffered pipelined kernel.
fn run_correctness_test_pipelined(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    assert!(m.is_multiple_of(128) && n.is_multiple_of(128) && k.is_multiple_of(64));

    println!("Matrix: {}x{}x{} (f16 -> bf16)", m, n, k);
    println!("Double-buffered SMEM, TMA/MMA overlap.");

    let mut host_a: Vec<u16> = vec![0u16; m * k];
    for i in 0..m {
        let val = f16::from_f32((i % 8 + 1) as f32).to_bits();
        for kk in 0..k {
            host_a[i * k + kk] = val;
        }
    }
    let mut host_b: Vec<u16> = vec![0u16; n * k];
    for j in 0..n {
        let val = f16::from_f32((j % 8 + 1) as f32).to_bits();
        for kk in 0..k {
            host_b[j * k + kk] = val;
        }
    }

    println!("A[i,k] = (i%8+1), B[n,k] = (n%8+1)");
    println!("Expected: C[i,j] = (i%8+1)*(j%8+1)*K\n");

    let dev_a = DeviceBuffer::from_host(stream, &host_a)?;
    let dev_b = DeviceBuffer::from_host(stream, &host_b)?;

    let a_ptr = dev_a.cu_deviceptr();
    let b_ptr = dev_b.cu_deviceptr();

    let a_tma =
        create_tma_descriptor_f16_swizzled(a_ptr as *mut std::ffi::c_void, k as u64, m as u64)?;
    let b_tma =
        create_tma_descriptor_f16_swizzled(b_ptr as *mut std::ffi::c_void, k as u64, n as u64)?;

    let dev_a_tma = DeviceBuffer::from_host(stream, &a_tma.opaque)?;
    let dev_b_tma = DeviceBuffer::from_host(stream, &b_tma.opaque)?;

    let grid_m = (m / 128) as u32;
    let grid_n = (n / 128) as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid_m, grid_n, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };

    println!(
        "Grid: {}x{} CTAs ({} total), block: 128 threads",
        grid_m,
        grid_n,
        grid_m * grid_n
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

    println!("\nLaunching gemm_sol_pipelined...");

    unsafe {
        module.gemm_sol_pipelined(
            (stream).as_ref(),
            cfg,
            a_tma_ptr,
            b_tma_ptr,
            &mut dev_output,
            n_arg,
            k_arg,
        )
    }?;

    stream.synchronize()?;

    let host_output: Vec<u32> = dev_output.to_host_vec(stream)?;

    let expected = |row: usize, col: usize| -> f32 { ((row % 8 + 1) * (col % 8 + 1) * k) as f32 };
    let read_c = |row: usize, col: usize| -> f32 {
        let packed_idx = row * (n / 2) + col / 2;
        let packed = host_output[packed_idx];
        let (lo, hi) = unpack_bf16_pair(packed);
        if col.is_multiple_of(2) { lo } else { hi }
    };

    let check_positions = [
        (0, 0),
        (0, n - 1),
        (m - 1, 0),
        (m - 1, n - 1),
        (m / 2, n / 2),
        (3, 5),
        (7, 7),
        (127, 127),
    ];

    let mut all_ok = true;
    println!("\nSpot checks:");
    for (row, col) in check_positions {
        let val = read_c(row, col);
        let exp = expected(row, col);
        let ok = (val - exp).abs() < (exp * 0.02 + 1.0);
        println!(
            "  C[{:>4},{:>4}] = {:>10.0}  (expected {:>10.0})  {}",
            row,
            col,
            val,
            exp,
            if ok { "OK" } else { "FAIL" }
        );
        if !ok {
            all_ok = false;
        }
    }

    let mut first_row_sum: f64 = 0.0;
    for col_pair in 0..(n / 2) {
        let packed = host_output[col_pair];
        let (lo, hi) = unpack_bf16_pair(packed);
        first_row_sum += lo as f64 + hi as f64;
    }
    let expected_row_sum = k as f64 * (n as f64 / 8.0) * 36.0;
    let row_sum_ok =
        (first_row_sum - expected_row_sum).abs() < (expected_row_sum * 0.02 + n as f64);
    println!(
        "\n  Row 0 sum: {:.0} (expected {:.0}) {}",
        first_row_sum,
        expected_row_sum,
        if row_sum_ok { "OK" } else { "FAIL" }
    );

    println!("\n═══════════════════════════════════════════════════════");
    if all_ok && row_sum_ok {
        println!("PASSED: gemm_sol_pipelined {}x{}x{}", m, n, k);
    } else {
        println!("FAILED: gemm_sol_pipelined {}x{}x{}", m, n, k);
        return Err("Correctness check failed".into());
    }
    println!("═══════════════════════════════════════════════════════");

    Ok(())
}

/// Benchmark the double-buffered pipelined GEMM kernel with CUDA event timing.
fn run_benchmark_pipelined(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    const WARMUP: usize = 10;
    const ITERS: usize = 100;

    assert!(m.is_multiple_of(128) && n.is_multiple_of(128) && k.is_multiple_of(64));

    let dev_a = DeviceBuffer::<u16>::zeroed(stream, m * k)?;
    let dev_b = DeviceBuffer::<u16>::zeroed(stream, n * k)?;

    let a_ptr = dev_a.cu_deviceptr();
    let b_ptr = dev_b.cu_deviceptr();

    let a_tma =
        create_tma_descriptor_f16_swizzled(a_ptr as *mut std::ffi::c_void, k as u64, m as u64)?;
    let b_tma =
        create_tma_descriptor_f16_swizzled(b_ptr as *mut std::ffi::c_void, k as u64, n as u64)?;

    let dev_a_tma = DeviceBuffer::from_host(stream, &a_tma.opaque)?;
    let dev_b_tma = DeviceBuffer::from_host(stream, &b_tma.opaque)?;
    let a_tma_ptr = dev_a_tma.cu_deviceptr();
    let b_tma_ptr = dev_b_tma.cu_deviceptr();
    let a_tma_ptr = a_tma_ptr as *const TmaDescriptor;
    let b_tma_ptr = b_tma_ptr as *const TmaDescriptor;
    let n_arg = n as i32;
    let k_arg = k as i32;

    let grid_m = (m / 128) as u32;
    let grid_n = (n / 128) as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid_m, grid_n, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };

    let output_u32_count = m * n / 2;
    let mut dev_output = DeviceBuffer::<u32>::zeroed(stream, output_u32_count)?;

    for _ in 0..WARMUP {
        unsafe {
            module.gemm_sol_pipelined(
                (stream).as_ref(),
                cfg,
                a_tma_ptr,
                b_tma_ptr,
                &mut dev_output,
                n_arg,
                k_arg,
            )
        }?;
    }
    stream.synchronize()?;

    let start_event =
        stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;

    for _ in 0..ITERS {
        unsafe {
            module.gemm_sol_pipelined(
                (stream).as_ref(),
                cfg,
                a_tma_ptr,
                b_tma_ptr,
                &mut dev_output,
                n_arg,
                k_arg,
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

    // cuBLAS SoL is now measured live via bench/cublaslt_bench (parsed by
    // the cublas_baseline module). The comparison line is printed by
    // print_cublas_comparison(tflops, m) below.

    println!("═══════════════════════════════════════════════════════");
    println!(
        "  BENCHMARK: gemm_sol_pipelined {}x{}x{} f16 -> bf16",
        m, n, k
    );
    println!("═══════════════════════════════════════════════════════");
    println!(
        "  Grid:        {}x{} CTAs ({} total)",
        grid_m,
        grid_n,
        grid_m * grid_n
    );
    println!("  K-loop:      {} outer iters (BK=64, 4 MMAs each)", k / 64);
    println!("  Pipeline:    2-stage double-buffered (SWIZZLE_128B)");
    println!("  Iterations:  {} (after {} warmup)", ITERS, WARMUP);
    println!("  Total time:  {:.3} ms", elapsed_ms);
    println!("  Average:     {:.3} us / kernel", avg_us);
    println!("  FLOPS/kern:  {:.3e}", flops);
    println!("  Throughput:  {:.3} TFLOPS", tflops);
    print_cublas_comparison(tflops, m);
    println!("═══════════════════════════════════════════════════════\n");

    Ok(())
}

fn run_correctness_test_warp_spec(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    assert!(m.is_multiple_of(128) && n.is_multiple_of(128) && k.is_multiple_of(64));

    println!("Matrix: {}x{}x{} (f16 -> bf16)", m, n, k);
    println!("Warp-specialized pipeline: warp 4 = TMA, warp 5 = MMA, warps 0-3 = epilogue.");

    let mut host_a: Vec<u16> = vec![0u16; m * k];
    for i in 0..m {
        let val = f16::from_f32((i % 8 + 1) as f32).to_bits();
        for kk in 0..k {
            host_a[i * k + kk] = val;
        }
    }
    let mut host_b: Vec<u16> = vec![0u16; n * k];
    for j in 0..n {
        let val = f16::from_f32((j % 8 + 1) as f32).to_bits();
        for kk in 0..k {
            host_b[j * k + kk] = val;
        }
    }

    println!("A[i,k] = (i%8+1), B[n,k] = (n%8+1)");
    println!("Expected: C[i,j] = (i%8+1)*(j%8+1)*K\n");

    let dev_a = DeviceBuffer::from_host(stream, &host_a)?;
    let dev_b = DeviceBuffer::from_host(stream, &host_b)?;

    let a_ptr = dev_a.cu_deviceptr();
    let b_ptr = dev_b.cu_deviceptr();

    let a_tma =
        create_tma_descriptor_f16_swizzled(a_ptr as *mut std::ffi::c_void, k as u64, m as u64)?;
    let b_tma =
        create_tma_descriptor_f16_swizzled(b_ptr as *mut std::ffi::c_void, k as u64, n as u64)?;

    let dev_a_tma = DeviceBuffer::from_host(stream, &a_tma.opaque)?;
    let dev_b_tma = DeviceBuffer::from_host(stream, &b_tma.opaque)?;

    let grid_m = (m / 128) as u32;
    let grid_n = (n / 128) as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid_m, grid_n, 1),
        block_dim: (192, 1, 1),
        shared_mem_bytes: 0,
    };

    println!(
        "Grid: {}x{} CTAs ({} total), block: 192 threads (6 warps)",
        grid_m,
        grid_n,
        grid_m * grid_n
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

    println!("\nLaunching gemm_sol_warp_spec...");

    unsafe {
        module.gemm_sol_warp_spec(
            (stream).as_ref(),
            cfg,
            a_tma_ptr,
            b_tma_ptr,
            &mut dev_output,
            n_arg,
            k_arg,
        )
    }?;

    stream.synchronize()?;

    let host_output: Vec<u32> = dev_output.to_host_vec(stream)?;

    let expected = |row: usize, col: usize| -> f32 { ((row % 8 + 1) * (col % 8 + 1) * k) as f32 };
    let read_c = |row: usize, col: usize| -> f32 {
        let packed_idx = row * (n / 2) + col / 2;
        let packed = host_output[packed_idx];
        let (lo, hi) = unpack_bf16_pair(packed);
        if col.is_multiple_of(2) { lo } else { hi }
    };

    let check_positions = [
        (0, 0),
        (0, n - 1),
        (m - 1, 0),
        (m - 1, n - 1),
        (m / 2, n / 2),
        (3, 5),
        (7, 7),
        (127, 127),
    ];

    let mut all_ok = true;
    println!("\nSpot checks:");
    for (row, col) in check_positions {
        let val = read_c(row, col);
        let exp = expected(row, col);
        let ok = (val - exp).abs() < (exp * 0.02 + 1.0);
        println!(
            "  C[{:>4},{:>4}] = {:>10.0}  (expected {:>10.0})  {}",
            row,
            col,
            val,
            exp,
            if ok { "OK" } else { "FAIL" }
        );
        if !ok {
            all_ok = false;
        }
    }

    let mut first_row_sum: f64 = 0.0;
    for col_pair in 0..(n / 2) {
        let packed = host_output[col_pair];
        let (lo, hi) = unpack_bf16_pair(packed);
        first_row_sum += lo as f64 + hi as f64;
    }
    let expected_row_sum = k as f64 * (n as f64 / 8.0) * 36.0;
    let row_sum_ok =
        (first_row_sum - expected_row_sum).abs() < (expected_row_sum * 0.02 + n as f64);
    println!(
        "\n  Row 0 sum: {:.0} (expected {:.0}) {}",
        first_row_sum,
        expected_row_sum,
        if row_sum_ok { "OK" } else { "FAIL" }
    );

    println!("\n═══════════════════════════════════════════════════════");
    if all_ok && row_sum_ok {
        println!("PASSED: gemm_sol_warp_spec {}x{}x{}", m, n, k);
    } else {
        println!("FAILED: gemm_sol_warp_spec {}x{}x{}", m, n, k);
        return Err("Correctness check failed".into());
    }
    println!("═══════════════════════════════════════════════════════");

    Ok(())
}

/// Benchmark the warp-specialized pipelined GEMM kernel with CUDA event timing.
fn run_benchmark_warp_spec(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    const WARMUP: usize = 10;
    const ITERS: usize = 100;

    assert!(m.is_multiple_of(128) && n.is_multiple_of(128) && k.is_multiple_of(64));

    let dev_a = DeviceBuffer::<u16>::zeroed(stream, m * k)?;
    let dev_b = DeviceBuffer::<u16>::zeroed(stream, n * k)?;

    let a_ptr = dev_a.cu_deviceptr();
    let b_ptr = dev_b.cu_deviceptr();

    let a_tma =
        create_tma_descriptor_f16_swizzled(a_ptr as *mut std::ffi::c_void, k as u64, m as u64)?;
    let b_tma =
        create_tma_descriptor_f16_swizzled(b_ptr as *mut std::ffi::c_void, k as u64, n as u64)?;

    let dev_a_tma = DeviceBuffer::from_host(stream, &a_tma.opaque)?;
    let dev_b_tma = DeviceBuffer::from_host(stream, &b_tma.opaque)?;
    let a_tma_ptr = dev_a_tma.cu_deviceptr();
    let b_tma_ptr = dev_b_tma.cu_deviceptr();
    let a_tma_ptr = a_tma_ptr as *const TmaDescriptor;
    let b_tma_ptr = b_tma_ptr as *const TmaDescriptor;
    let n_arg = n as i32;
    let k_arg = k as i32;

    let grid_m = (m / 128) as u32;
    let grid_n = (n / 128) as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid_m, grid_n, 1),
        block_dim: (192, 1, 1),
        shared_mem_bytes: 0,
    };

    let output_u32_count = m * n / 2;
    let mut dev_output = DeviceBuffer::<u32>::zeroed(stream, output_u32_count)?;

    for _ in 0..WARMUP {
        unsafe {
            module.gemm_sol_warp_spec(
                (stream).as_ref(),
                cfg,
                a_tma_ptr,
                b_tma_ptr,
                &mut dev_output,
                n_arg,
                k_arg,
            )
        }?;
    }
    stream.synchronize()?;

    let start_event =
        stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;

    for _ in 0..ITERS {
        unsafe {
            module.gemm_sol_warp_spec(
                (stream).as_ref(),
                cfg,
                a_tma_ptr,
                b_tma_ptr,
                &mut dev_output,
                n_arg,
                k_arg,
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

    // cuBLAS SoL is now measured live via bench/cublaslt_bench (parsed by
    // the cublas_baseline module). The comparison line is printed by
    // print_cublas_comparison(tflops, m) below.

    println!("═══════════════════════════════════════════════════════");
    println!(
        "  BENCHMARK: gemm_sol_warp_spec {}x{}x{} f16 -> bf16",
        m, n, k
    );
    println!("═══════════════════════════════════════════════════════");
    println!(
        "  Grid:        {}x{} CTAs ({} total)",
        grid_m,
        grid_n,
        grid_m * grid_n
    );
    println!("  K-loop:      {} outer iters (BK=64, 4 MMAs each)", k / 64);
    println!("  Pipeline:    warp-specialized 2-stage (SWIZZLE_128B)");
    println!("  Iterations:  {} (after {} warmup)", ITERS, WARMUP);
    println!("  Total time:  {:.3} ms", elapsed_ms);
    println!("  Average:     {:.3} us / kernel", avg_us);
    println!("  FLOPS/kern:  {:.3e}", flops);
    println!("  Throughput:  {:.3} TFLOPS", tflops);
    print_cublas_comparison(tflops, m);
    println!("═══════════════════════════════════════════════════════\n");

    Ok(())
}

fn run_correctness_test_persistent(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    assert!(m.is_multiple_of(128) && n.is_multiple_of(128) && k.is_multiple_of(64));

    println!("Matrix: {}x{}x{} (f16 -> bf16)", m, n, k);
    println!("Persistent kernel: TMEM accum pipeline (2 stages).");
    println!("Warps: 4=TMA, 5=MMA, 0-3=epilogue.");

    let mut host_a: Vec<u16> = vec![0u16; m * k];
    for i in 0..m {
        let val = f16::from_f32((i % 8 + 1) as f32).to_bits();
        for kk in 0..k {
            host_a[i * k + kk] = val;
        }
    }
    let mut host_b: Vec<u16> = vec![0u16; n * k];
    for j in 0..n {
        let val = f16::from_f32((j % 8 + 1) as f32).to_bits();
        for kk in 0..k {
            host_b[j * k + kk] = val;
        }
    }

    println!("A[i,k] = (i%8+1), B[n,k] = (n%8+1)");
    println!("Expected: C[i,j] = (i%8+1)*(j%8+1)*K\n");

    let dev_a = DeviceBuffer::from_host(stream, &host_a)?;
    let dev_b = DeviceBuffer::from_host(stream, &host_b)?;

    let a_ptr = dev_a.cu_deviceptr();
    let b_ptr = dev_b.cu_deviceptr();

    let a_tma =
        create_tma_descriptor_f16_swizzled(a_ptr as *mut std::ffi::c_void, k as u64, m as u64)?;
    let b_tma =
        create_tma_descriptor_f16_swizzled(b_ptr as *mut std::ffi::c_void, k as u64, n as u64)?;

    let dev_a_tma = DeviceBuffer::from_host(stream, &a_tma.opaque)?;
    let dev_b_tma = DeviceBuffer::from_host(stream, &b_tma.opaque)?;

    let tiles_m = (m / 128) as u32;
    let tiles_n = (n / 128) as u32;
    let total_tiles = tiles_m * tiles_n;

    // Persistent: launch enough CTAs to fill the GPU, in clusters of 4
    let cluster_size = 4u32;
    let num_clusters = 37u32; // 148 SMs / 4
    let num_ctas = num_clusters * cluster_size;

    let cfg = LaunchConfig {
        grid_dim: (num_ctas, 1, 1),
        block_dim: (192, 1, 1),
        shared_mem_bytes: 0,
    };

    println!(
        "Grid: {} CTAs ({} clusters of {}), block: 192 threads (6 warps)",
        num_ctas, num_clusters, cluster_size
    );
    println!(
        "Total tiles: {} ({}x{}), ~{} tiles/CTA",
        total_tiles,
        tiles_m,
        tiles_n,
        total_tiles / num_ctas
    );
    println!("K-loop: {} outer iters (BK=64, 4 MMAs each)", k / 64);

    let output_u32_count = m * n / 2;
    let mut dev_output = DeviceBuffer::<u32>::zeroed(stream, output_u32_count)?;

    // Tile counter: single u32 initialized to 0
    let dev_tile_counter = DeviceBuffer::from_host(stream, &[0u32])?;

    let a_tma_ptr = dev_a_tma.cu_deviceptr();
    let b_tma_ptr = dev_b_tma.cu_deviceptr();
    let a_tma_ptr = a_tma_ptr as *const TmaDescriptor;
    let b_tma_ptr = b_tma_ptr as *const TmaDescriptor;
    let counter_ptr = dev_tile_counter.cu_deviceptr();
    let counter_ptr = counter_ptr as *const u32;
    let n_arg = n as i32;
    let k_arg = k as i32;

    println!("\nLaunching gemm_sol_persistent...");

    unsafe {
        module.gemm_sol_persistent(
            (stream).as_ref(),
            cfg,
            a_tma_ptr,
            b_tma_ptr,
            &mut dev_output,
            counter_ptr,
            n_arg,
            k_arg,
            tiles_m,
            tiles_n,
        )
    }?;

    stream.synchronize()?;

    let host_output: Vec<u32> = dev_output.to_host_vec(stream)?;

    let expected = |row: usize, col: usize| -> f32 { ((row % 8 + 1) * (col % 8 + 1) * k) as f32 };
    let read_c = |row: usize, col: usize| -> f32 {
        let packed_idx = row * (n / 2) + col / 2;
        let packed = host_output[packed_idx];
        let (lo, hi) = unpack_bf16_pair(packed);
        if col.is_multiple_of(2) { lo } else { hi }
    };

    let check_positions = [
        (0, 0),
        (0, n - 1),
        (m - 1, 0),
        (m - 1, n - 1),
        (m / 2, n / 2),
        (3, 5),
        (7, 7),
        (127, 127),
    ];

    let mut all_ok = true;
    println!("\nSpot checks:");
    for (row, col) in check_positions {
        let val = read_c(row, col);
        let exp = expected(row, col);
        let ok = (val - exp).abs() < (exp * 0.02 + 1.0);
        println!(
            "  C[{:>4},{:>4}] = {:>10.0}  (expected {:>10.0})  {}",
            row,
            col,
            val,
            exp,
            if ok { "OK" } else { "FAIL" }
        );
        if !ok {
            all_ok = false;
        }
    }

    let mut first_row_sum: f64 = 0.0;
    for col_pair in 0..(n / 2) {
        let packed = host_output[col_pair];
        let (lo, hi) = unpack_bf16_pair(packed);
        first_row_sum += lo as f64 + hi as f64;
    }
    let expected_row_sum = k as f64 * (n as f64 / 8.0) * 36.0;
    let row_sum_ok =
        (first_row_sum - expected_row_sum).abs() < (expected_row_sum * 0.02 + n as f64);
    println!(
        "\n  Row 0 sum: {:.0} (expected {:.0}) {}",
        first_row_sum,
        expected_row_sum,
        if row_sum_ok { "OK" } else { "FAIL" }
    );

    println!("\n═══════════════════════════════════════════════════════");
    if all_ok && row_sum_ok {
        println!("PASSED: gemm_sol_persistent {}x{}x{}", m, n, k);
    } else {
        println!("FAILED: gemm_sol_persistent {}x{}x{}", m, n, k);
        return Err("Correctness check failed".into());
    }
    println!("═══════════════════════════════════════════════════════");

    Ok(())
}

fn run_benchmark_persistent(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    const WARMUP: usize = 10;
    const ITERS: usize = 100;

    assert!(m.is_multiple_of(128) && n.is_multiple_of(128) && k.is_multiple_of(64));

    let dev_a = DeviceBuffer::<u16>::zeroed(stream, m * k)?;
    let dev_b = DeviceBuffer::<u16>::zeroed(stream, n * k)?;

    let a_ptr = dev_a.cu_deviceptr();
    let b_ptr = dev_b.cu_deviceptr();

    let a_tma =
        create_tma_descriptor_f16_swizzled(a_ptr as *mut std::ffi::c_void, k as u64, m as u64)?;
    let b_tma =
        create_tma_descriptor_f16_swizzled(b_ptr as *mut std::ffi::c_void, k as u64, n as u64)?;

    let dev_a_tma = DeviceBuffer::from_host(stream, &a_tma.opaque)?;
    let dev_b_tma = DeviceBuffer::from_host(stream, &b_tma.opaque)?;
    let a_tma_ptr = dev_a_tma.cu_deviceptr();
    let b_tma_ptr = dev_b_tma.cu_deviceptr();
    let a_tma_ptr = a_tma_ptr as *const TmaDescriptor;
    let b_tma_ptr = b_tma_ptr as *const TmaDescriptor;
    let n_arg = n as i32;
    let k_arg = k as i32;

    let tiles_m = (m / 128) as u32;
    let tiles_n = (n / 128) as u32;

    let cluster_size = 4u32;
    let num_clusters = 37u32;
    let num_ctas = num_clusters * cluster_size;

    let cfg = LaunchConfig {
        grid_dim: (num_ctas, 1, 1),
        block_dim: (192, 1, 1),
        shared_mem_bytes: 0,
    };

    let output_u32_count = m * n / 2;
    let mut dev_output = DeviceBuffer::<u32>::zeroed(stream, output_u32_count)?;
    let dev_tile_counter = DeviceBuffer::<u32>::zeroed(stream, 1)?;

    let counter_ptr = {
        let ptr = dev_tile_counter.cu_deviceptr();
        ptr as *const u32
    };

    for _ in 0..WARMUP {
        let z = 0u32;
        unsafe {
            cuda_core::simt::memory::memcpy_htod_async(
                dev_tile_counter.cu_deviceptr(),
                &z as *const u32,
                std::mem::size_of::<u32>(),
                stream.cu_stream(),
            )?;
        }
        unsafe {
            module.gemm_sol_persistent(
                (stream).as_ref(),
                cfg,
                a_tma_ptr,
                b_tma_ptr,
                &mut dev_output,
                counter_ptr,
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
        let z = 0u32;
        unsafe {
            cuda_core::simt::memory::memcpy_htod_async(
                dev_tile_counter.cu_deviceptr(),
                &z as *const u32,
                std::mem::size_of::<u32>(),
                stream.cu_stream(),
            )?;
        }
        unsafe {
            module.gemm_sol_persistent(
                (stream).as_ref(),
                cfg,
                a_tma_ptr,
                b_tma_ptr,
                &mut dev_output,
                counter_ptr,
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

    // cuBLAS SoL is now measured live via bench/cublaslt_bench (parsed by
    // the cublas_baseline module). The comparison line is printed by
    // print_cublas_comparison(tflops, m) below.

    println!("═══════════════════════════════════════════════════════");
    println!(
        "  BENCHMARK: gemm_sol_persistent {}x{}x{} f16 -> bf16",
        m, n, k
    );
    println!("═══════════════════════════════════════════════════════");
    println!(
        "  Grid:        {} CTAs ({} clusters×4)",
        num_ctas, num_clusters
    );
    println!("  K-loop:      {} outer iters (BK=64, 4 MMAs each)", k / 64);
    println!("  Pipeline:    persistent + 2-stage TMEM accum");
    println!("  Iterations:  {} (after {} warmup)", ITERS, WARMUP);
    println!("  Total time:  {:.3} ms", elapsed_ms);
    println!("  Average:     {:.3} us / kernel", avg_us);
    println!("  FLOPS/kern:  {:.3e}", flops);
    println!("  Throughput:  {:.3} TFLOPS", tflops);
    print_cublas_comparison(tflops, m);
    println!("═══════════════════════════════════════════════════════\n");

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
    // The quadrant scale breaks the affine-code twin symmetry: since
    // code(x + 2048) = code(x) ^ 0x800, the XOR-based expected values would
    // otherwise be invariant under (row, col) -> (row + 2048, col + 2048),
    // giving every 128x128 output tile a byte-identical twin 16 tiles away
    // on both axes — exactly the tile-permutation blindness this validator
    // exists to remove. Doubling the top half of A keeps every value an
    // exact small f16 integer while making twin tiles differ everywhere
    // (scores are odd, so never zero).
    let quadrant_scale = (1 + (row >> 11)) as f32;
    quadrant_scale * validation_fingerprint(validation_row_code(row), kk)
}

#[inline]
fn validation_b_value(col: usize, kk: usize) -> f32 {
    validation_fingerprint(validation_col_code(col), kk)
}

/// Builds the Walsh-fingerprint A/B inputs for the fixed 4096³ CLC correctness
/// contract. Every row of A and column of B carries a unique whole-vector
/// signature, so tile-permutation failures (stale TILE_INFO work-ids, tile
/// swaps between readers, wrong accumulators) produce output bytes that cannot
/// match the analytic reference — unlike periodic inputs, where all 128×128
/// output tiles are byte-identical.
fn validation_build_inputs(m: usize, n: usize, k: usize) -> (Vec<u16>, Vec<u16>) {
    assert_eq!(
        [m, n, k],
        [VALIDATION_CODE_COUNT; 3],
        "the bit-exact Walsh validator is defined for the fixed 4096³ contract"
    );

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

    println!(
        "Validation data: 12-bit Walsh fingerprints varying along K and covering \
         each K code once, with unique signatures for all {} rows and columns",
        VALIDATION_CODE_COUNT
    );
    println!("Expected: analytic exact FP32 dot product rounded once to BF16\n");

    (host_a, host_b)
}

/// Full-output correctness check: compares EVERY BF16 element against the
/// exact rounded reference, not a sampled subset. The fingerprints are small
/// signed integers, so every FP32 product and partial sum is exact in any
/// accumulation order (magnitudes stay far below 2^24) and a correct kernel
/// must match the reference BF16 bits.
fn validation_check_full_output(
    kernel_name: &str,
    host_output: &[u32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Walsh orthogonality gives C[row,col] = scale(row) * K * (13 - 2 *
    // HammingDistance) for each row/column code pair, where scale is the
    // quadrant scale from validation_a_value (1 for rows 0..2047, 2 above).
    // Cache 4096 XOR-indexed entries per row half; K * score * scale keeps
    // at most 5 significant bits, so both are exact in BF16.
    let expected_by_xor: Vec<[u16; 2]> = (0..VALIDATION_CODE_COUNT)
        .map(|difference| {
            let score = 13 - 2 * (difference as u32).count_ones() as i64;
            let fp32 = (k as i64 * score) as f32;
            [
                bf16::from_f32(fp32).to_bits(),
                bf16::from_f32(2.0 * fp32).to_bits(),
            ]
        })
        .collect();

    let expected_bits = |row: usize, col: usize| -> u16 {
        expected_by_xor[(validation_row_code(row) ^ validation_col_code(col)) as usize][row >> 11]
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
    if mismatches == 0 {
        println!(
            "PASSED: {} {}x{}x{} (full {}-element check)",
            kernel_name, m, n, k, total
        );
    } else {
        println!("FAILED: {} {}x{}x{}", kernel_name, m, n, k);
        return Err(format!(
            "Correctness check failed: {} / {} elements wrong",
            mismatches, total
        )
        .into());
    }
    println!("═══════════════════════════════════════════════════════");

    Ok(())
}

fn run_correctness_test_clc(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    assert!(m.is_multiple_of(128) && n.is_multiple_of(128) && k.is_multiple_of(64));

    println!("Matrix: {}x{}x{} (f16 -> bf16)", m, n, k);
    println!("CLC tile scheduling + TMEM accum pipeline.");
    println!("Warps: 4=TMA, 5=MMA, 0-3=epilogue.");

    let (host_a, host_b) = validation_build_inputs(m, n, k);

    let dev_a = DeviceBuffer::from_host(stream, &host_a)?;
    let dev_b = DeviceBuffer::from_host(stream, &host_b)?;

    let a_ptr = dev_a.cu_deviceptr();
    let b_ptr = dev_b.cu_deviceptr();

    let a_tma =
        create_tma_descriptor_f16_swizzled(a_ptr as *mut std::ffi::c_void, k as u64, m as u64)?;
    let b_tma =
        create_tma_descriptor_f16_swizzled(b_ptr as *mut std::ffi::c_void, k as u64, n as u64)?;

    let dev_a_tma = DeviceBuffer::from_host(stream, &a_tma.opaque)?;
    let dev_b_tma = DeviceBuffer::from_host(stream, &b_tma.opaque)?;

    let tiles_m = (m / 128) as u32;
    let tiles_n = (n / 128) as u32;
    let total_tiles = tiles_m * tiles_n;

    let cfg = LaunchConfig {
        grid_dim: (total_tiles, 1, 1),
        block_dim: (192, 1, 1),
        shared_mem_bytes: 0,
    };

    println!(
        "Grid: {} CTAs (1D, CLC-managed), cluster: 4x1x1, block: 192 threads (6 warps)",
        total_tiles
    );
    println!(
        "Total tiles: {} ({}x{}), column-major rasterization",
        total_tiles, tiles_m, tiles_n
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

    println!("\nLaunching gemm_sol_clc...");

    unsafe {
        module.gemm_sol_clc(
            (stream).as_ref(),
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

    validation_check_full_output("gemm_sol_clc", &host_output, m, n, k)
}

fn run_benchmark_clc(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    const WARMUP: usize = 10;
    const ITERS: usize = 100;

    assert!(m.is_multiple_of(128) && n.is_multiple_of(128) && k.is_multiple_of(64));

    let dev_a = DeviceBuffer::<u16>::zeroed(stream, m * k)?;
    let dev_b = DeviceBuffer::<u16>::zeroed(stream, n * k)?;

    let a_ptr = dev_a.cu_deviceptr();
    let b_ptr = dev_b.cu_deviceptr();

    let a_tma =
        create_tma_descriptor_f16_swizzled(a_ptr as *mut std::ffi::c_void, k as u64, m as u64)?;
    let b_tma =
        create_tma_descriptor_f16_swizzled(b_ptr as *mut std::ffi::c_void, k as u64, n as u64)?;

    let dev_a_tma = DeviceBuffer::from_host(stream, &a_tma.opaque)?;
    let dev_b_tma = DeviceBuffer::from_host(stream, &b_tma.opaque)?;
    let a_tma_ptr = dev_a_tma.cu_deviceptr();
    let b_tma_ptr = dev_b_tma.cu_deviceptr();
    let a_tma_ptr = a_tma_ptr as *const TmaDescriptor;
    let b_tma_ptr = b_tma_ptr as *const TmaDescriptor;
    let n_arg = n as i32;
    let k_arg = k as i32;

    let tiles_m = (m / 128) as u32;
    let tiles_n = (n / 128) as u32;
    let total_tiles = tiles_m * tiles_n;

    let cfg = LaunchConfig {
        grid_dim: (total_tiles, 1, 1),
        block_dim: (192, 1, 1),
        shared_mem_bytes: 0,
    };

    let output_u32_count = m * n / 2;
    let mut dev_output = DeviceBuffer::<u32>::zeroed(stream, output_u32_count)?;

    for _ in 0..WARMUP {
        unsafe {
            module.gemm_sol_clc(
                (stream).as_ref(),
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
            module.gemm_sol_clc(
                (stream).as_ref(),
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

    // cuBLAS SoL is now measured live via bench/cublaslt_bench (parsed by
    // the cublas_baseline module). The comparison line is printed by
    // print_cublas_comparison(tflops, m) below.

    println!("═══════════════════════════════════════════════════════");
    println!("  BENCHMARK: gemm_sol_clc {}x{}x{} f16 -> bf16", m, n, k);
    println!("═══════════════════════════════════════════════════════");
    println!(
        "  Grid:        {} CTAs (1D, CLC-managed, cluster=4)",
        total_tiles
    );
    println!("  K-loop:      {} outer iters (BK=64, 4 MMAs each)", k / 64);
    println!("  Pipeline:    CLC + 2-stage TMEM accum");
    println!("  Iterations:  {} (after {} warmup)", ITERS, WARMUP);
    println!("  Total time:  {:.3} ms", elapsed_ms);
    println!("  Average:     {:.3} us / kernel", avg_us);
    println!("  FLOPS/kern:  {:.3e}", flops);
    println!("  Throughput:  {:.3} TFLOPS", tflops);
    print_cublas_comparison(tflops, m);
    println!("═══════════════════════════════════════════════════════\n");

    Ok(())
}

fn run_correctness_test_clc_multicast(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    assert!(m.is_multiple_of(128) && n.is_multiple_of(128) && k.is_multiple_of(64));

    println!("Matrix: {}x{}x{} (f16 -> bf16)", m, n, k);
    println!("CLC + TMA multicast for B tiles + TMEM accum pipeline.");
    println!("Warps: 4=TMA, 5=MMA, 0-3=epilogue.");

    let (host_a, host_b) = validation_build_inputs(m, n, k);

    let dev_a = DeviceBuffer::from_host(stream, &host_a)?;
    let dev_b = DeviceBuffer::from_host(stream, &host_b)?;

    let a_ptr = dev_a.cu_deviceptr();
    let b_ptr = dev_b.cu_deviceptr();

    let a_tma =
        create_tma_descriptor_f16_swizzled(a_ptr as *mut std::ffi::c_void, k as u64, m as u64)?;
    let b_tma =
        create_tma_descriptor_f16_swizzled(b_ptr as *mut std::ffi::c_void, k as u64, n as u64)?;

    let dev_a_tma = DeviceBuffer::from_host(stream, &a_tma.opaque)?;
    let dev_b_tma = DeviceBuffer::from_host(stream, &b_tma.opaque)?;

    let tiles_m = (m / 128) as u32;
    let tiles_n = (n / 128) as u32;
    let total_tiles = tiles_m * tiles_n;

    let cfg = LaunchConfig {
        grid_dim: (total_tiles, 1, 1),
        block_dim: (192, 1, 1),
        shared_mem_bytes: 0,
    };

    println!(
        "Grid: {} CTAs (1D, CLC-managed), cluster: 4x1x1, block: 192 threads (6 warps)",
        total_tiles
    );
    println!(
        "Total tiles: {} ({}x{}), column-major rasterization",
        total_tiles, tiles_m, tiles_n
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

    println!("\nLaunching gemm_sol_clc_multicast...");

    unsafe {
        module.gemm_sol_clc_multicast(
            (stream).as_ref(),
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

    validation_check_full_output("gemm_sol_clc_multicast", &host_output, m, n, k)
}

fn run_benchmark_clc_multicast(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    const WARMUP: usize = 10;
    const ITERS: usize = 100;

    assert!(m.is_multiple_of(128) && n.is_multiple_of(128) && k.is_multiple_of(64));

    let dev_a = DeviceBuffer::<u16>::zeroed(stream, m * k)?;
    let dev_b = DeviceBuffer::<u16>::zeroed(stream, n * k)?;

    let a_ptr = dev_a.cu_deviceptr();
    let b_ptr = dev_b.cu_deviceptr();

    let a_tma =
        create_tma_descriptor_f16_swizzled(a_ptr as *mut std::ffi::c_void, k as u64, m as u64)?;
    let b_tma =
        create_tma_descriptor_f16_swizzled(b_ptr as *mut std::ffi::c_void, k as u64, n as u64)?;

    let dev_a_tma = DeviceBuffer::from_host(stream, &a_tma.opaque)?;
    let dev_b_tma = DeviceBuffer::from_host(stream, &b_tma.opaque)?;
    let a_tma_ptr = dev_a_tma.cu_deviceptr();
    let b_tma_ptr = dev_b_tma.cu_deviceptr();
    let a_tma_ptr = a_tma_ptr as *const TmaDescriptor;
    let b_tma_ptr = b_tma_ptr as *const TmaDescriptor;
    let n_arg = n as i32;
    let k_arg = k as i32;

    let tiles_m = (m / 128) as u32;
    let tiles_n = (n / 128) as u32;
    let total_tiles = tiles_m * tiles_n;

    let cfg = LaunchConfig {
        grid_dim: (total_tiles, 1, 1),
        block_dim: (192, 1, 1),
        shared_mem_bytes: 0,
    };

    let output_u32_count = m * n / 2;
    let mut dev_output = DeviceBuffer::<u32>::zeroed(stream, output_u32_count)?;

    for _ in 0..WARMUP {
        unsafe {
            module.gemm_sol_clc_multicast(
                (stream).as_ref(),
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
            module.gemm_sol_clc_multicast(
                (stream).as_ref(),
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
        "  BENCHMARK: gemm_sol_clc_multicast {}x{}x{} f16 -> bf16",
        m, n, k
    );
    println!("═══════════════════════════════════════════════════════");
    println!(
        "  Grid:        {} CTAs (1D, CLC-managed, cluster=4)",
        total_tiles
    );
    println!("  K-loop:      {} outer iters (BK=64, 4 MMAs each)", k / 64);
    println!("  Pipeline:    CLC + TMA multicast + 2-stage TMEM accum");
    println!("  Iterations:  {} (after {} warmup)", ITERS, WARMUP);
    println!("  Total time:  {:.3} ms", elapsed_ms);
    println!("  Average:     {:.3} us / kernel", avg_us);
    println!("  FLOPS/kern:  {:.3e}", flops);
    println!("  Throughput:  {:.3} TFLOPS", tflops);
    print_cublas_comparison(tflops, m);
    println!("═══════════════════════════════════════════════════════\n");

    Ok(())
}

fn run_correctness_test_clc_multicast_4_stage_pipeline(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    assert!(m.is_multiple_of(256) && n.is_multiple_of(128) && k.is_multiple_of(256));

    println!("Matrix: {}x{}x{} (f16 -> bf16)", m, n, k);
    println!("CLC + cta_group::2 + 4-stage SMEM pipeline.");
    println!("Warps: 4=TMA, 5=MMA (leader only), 0-3=epilogue (both CTAs).");

    let (host_a, host_b) = validation_build_inputs(m, n, k);

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
    println!(
        "Total tiles: {} ({}x{}), column-major rasterization",
        total_tiles, tiles_m, tiles_n
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

    println!("\nLaunching gemm_sol_clc_multicast_4_stage_pipeline (cg2)...");

    unsafe {
        module.gemm_sol_clc_multicast_4_stage_pipeline(
            (stream).as_ref(),
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

    validation_check_full_output(
        "gemm_sol_clc_multicast_4_stage_pipeline",
        &host_output,
        m,
        n,
        k,
    )
}

fn run_benchmark_clc_multicast_4_stage_pipeline(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    const WARMUP: usize = 10;
    const ITERS: usize = 100;

    assert!(m.is_multiple_of(256) && n.is_multiple_of(128) && k.is_multiple_of(256));

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
            module.gemm_sol_clc_multicast_4_stage_pipeline(
                (stream).as_ref(),
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
            module.gemm_sol_clc_multicast_4_stage_pipeline(
                (stream).as_ref(),
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
        "  BENCHMARK: gemm_sol_clc_multicast_4_stage_pipeline (cg2) {}x{}x{} f16 -> bf16",
        m, n, k
    );
    println!("═══════════════════════════════════════════════════════");
    println!(
        "  Grid:        {} CTAs (1D, CLC-managed, cluster=2, cg2)",
        total_tiles * 2
    );
    println!("  K-loop:      {} outer iters (BK=64, 4 MMAs each)", k / 64);
    println!("  Pipeline:    CLC + cta_group::2 + 4-stage SMEM");
    println!("  Iterations:  {} (after {} warmup)", ITERS, WARMUP);
    println!("  Total time:  {:.3} ms", elapsed_ms);
    println!("  Average:     {:.3} us / kernel", avg_us);
    println!("  FLOPS/kern:  {:.3e}", flops);
    println!("  Throughput:  {:.3} TFLOPS", tflops);
    print_cublas_comparison(tflops, m);
    println!("═══════════════════════════════════════════════════════\n");

    Ok(())
}

// Keep this execution set in sync with mir-importer's tcgen05 target
// support (and with gemm_sol_final's copy). Other GPU generations may
// inspect and assemble the generated artifact, but must not turn a
// module-load failure into an execution pass.
fn can_execute_tcgen05_ptx(major: i32, minor: i32) -> bool {
    matches!((major, minor), (10, 0) | (10, 1) | (10, 3) | (11, 0))
}

fn verify_ptx_only() -> Result<(), Box<dyn std::error::Error>> {
    let ptx_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("gemm_sol.ptx");

    if !ptx_path.exists() {
        return Err("PTX file not found".into());
    }

    println!("\nPTX Verification:");
    println!("   PTX file generated at: {}", ptx_path.display());
    println!("\n   To inspect generated PTX:");
    println!("   cat {}", ptx_path.display());

    Ok(())
}

/// Create a 2D TMA descriptor for f16 data.
///
/// `global_width` / `global_height` are the full tensor dimensions (in elements).
/// `tile_width` / `tile_height` are the TMA box dimensions (what each copy fetches).
///
/// For our GEMM:
///   A: width=K, height=M, tile=[8, 128]
///   B: width=K, height=N, tile=[8, 128]
fn create_tma_descriptor_f16(
    global_address: *mut std::ffi::c_void,
    global_width: u64,
    global_height: u64,
    tile_width: u32,
    tile_height: u32,
) -> Result<cuda_core::sys::CUtensorMap, Box<dyn std::error::Error>> {
    use cuda_core::sys::{
        CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_NONE, cuTensorMapEncodeTiled,
        cudaError_enum_CUDA_SUCCESS,
    };

    let mut tensor_map = MaybeUninit::<cuda_core::sys::CUtensorMap>::uninit();
    let global_dim: [u64; 2] = [global_width, global_height];
    let global_strides: [u64; 1] = [global_width * 2]; // byte stride between rows
    let box_dim: [u32; 2] = [tile_width, tile_height];
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
            CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_NONE,
            CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };

    if result != cudaError_enum_CUDA_SUCCESS {
        return Err(format!("cuTensorMapEncodeTiled failed: {:?}", result).into());
    }

    Ok(unsafe { tensor_map.assume_init() })
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

fn unpack_bf16_pair(packed: u32) -> (f32, f32) {
    let lo = (packed & 0xFFFF) as u16;
    let hi = ((packed >> 16) & 0xFFFF) as u16;
    (bf16_to_f32(lo), bf16_to_f32(hi))
}
