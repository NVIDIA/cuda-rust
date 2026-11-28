/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Provably safe GPU atomic operations and warp vote collectives (`safe_atomics`)
//! verified at compile time by Lean 4 with zero `unsafe` blocks inside kernels.
//!
//! ```text
//!   [1] `two_level_histogram`      Shared `BlockAtomicU32` (256 bins) -> Global `DeviceAtomicU32`
//!   [2] `ballot_warp_compaction`   `warp::ballot` + `lanemask_lt` -> 1 `fetch_add` per warp
//!   [3] `cas_max_and_fault_ticket` CAS spin-loop + `fetch_min`/`or` + last-CTA `AcqRel` ticket
//!   [4] `per_block_cta_counters`   Global `BlockAtomicU32` owned per CTA (`bins[blockIdx_x()]`)
//!   [5] `float_and_system_atomics` `DeviceAtomicF32` bucket sums + `SystemAtomicU32` count
//! ```

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{
    AtomicOrdering, BlockAtomicU32, DeviceAtomicF32, DeviceAtomicU32, SystemAtomicU32,
};
use cuda_device::{
    SharedArray, cuda_module, kernel, launch_bounds, launch_contract, thread, verify, warp,
};

/// Threads per CTA (`256` threads = `8` warps of `32` lanes).
const BLOCK_SIZE: u32 = 256;
/// Number of histogram bins (`256`, matching `BLOCK_SIZE`).
const NUM_BINS: usize = 256;
/// Number of launched CTAs across the 1D grid.
const NUM_BLOCKS: u32 = 8;
/// Total input elements processed across all CTAs (`4` elements per thread).
const NUM_ELEMS: usize = (BLOCK_SIZE * NUM_BLOCKS * 4) as usize;

/// Device kernel module containing all five Lean-verified atomic and warp-vote kernels.
#[cuda_module]
mod kernels {
    use super::*;

    /// Two-level 256-bin histogram combining `.cta`-scoped shared atomics with
    /// `.gpu`-scoped global atomics (zero `unsafe` blocks):
    /// - Phase 0: Each thread zeroes `LOCAL_BINS[tid]` via `BlockAtomicU32::store`.
    /// - Phase 1: Strided loop accumulates `LOCAL_BINS[val & 255].fetch_add(1)`.
    /// - Phase 2: Each thread flushes non-zero `LOCAL_BINS[tid]` into `bins[tid]`.
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (bins.len() >= 256),
    )]
    #[verify(
        no_oob_idx(data, bins, LOCAL_BINS),
        no_data_race(LOCAL_BINS, bins),
        uniform_barriers
    )]
    #[kernel]
    pub fn two_level_histogram(data: &[u32], bins: &mut [DeviceAtomicU32]) {
        /// Per-CTA 256-bin shared histogram backed by `.cta`-scoped `BlockAtomicU32`.
        static mut LOCAL_BINS: SharedArray<BlockAtomicU32, NUM_BINS> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        LOCAL_BINS[tid].store(0, AtomicOrdering::Relaxed);
        thread::sync_threads();

        let stride = (NUM_BLOCKS * BLOCK_SIZE) as usize;
        let mut i = (thread::blockIdx_x() * BLOCK_SIZE) as usize + tid;
        while i < data.len() {
            let bin = (data[i] as usize) & 255;
            LOCAL_BINS[bin].fetch_add(1, AtomicOrdering::Relaxed);
            i += stride;
        }
        thread::sync_threads();

        let local_count = LOCAL_BINS[tid].load(AtomicOrdering::Relaxed);
        if local_count > 0 && tid < bins.len() {
            bins[tid].fetch_add(local_count, AtomicOrdering::Relaxed);
        }
    }

    /// Warp-aggregated stream compaction using `warp::ballot` and `warp::lanemask_lt`:
    /// - All 32 lanes vote on `keep = val > threshold` (`CfgOp.warpSyncFull`).
    /// - Lane 0 reserves `mask.count_ones()` slots in `out_count[0]` with a single
    ///   `DeviceAtomicU32::fetch_add` and broadcasts `warp_base` with `shuffle_sync`.
    /// - Each matching lane writes its value to `out_vals[warp_base + lane_rank]`.
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (out_count.len() >= 1, out_vals.len() >= input.len()),
    )]
    #[verify(
        no_oob_idx(input, out_count, out_vals),
        no_data_race(out_count, out_vals),
        uniform_barriers
    )]
    #[kernel]
    pub fn ballot_warp_compaction(
        input: &[u32],
        threshold: u32,
        out_count: &mut [DeviceAtomicU32],
        out_vals: &mut [DeviceAtomicU32],
    ) {
        let gid = thread::index_1d().get();
        let val = if gid < input.len() { input[gid] } else { 0 };
        let keep = gid < input.len() && val > threshold;

        let mask = warp::ballot(keep);
        let warp_pop = mask.count_ones();
        let mut base = 0u32;
        if warp::lane_id() == 0 && warp_pop > 0 && !out_count.is_empty() {
            base = out_count[0].fetch_add(warp_pop, AtomicOrdering::Relaxed);
        }
        let warp_base = warp::shuffle(base, 0);

        if keep {
            let rank = (mask & warp::lanemask_lt()).count_ones();
            let dst = (warp_base + rank) as usize;
            if dst < out_vals.len() {
                out_vals[dst].store(val, AtomicOrdering::Relaxed);
            }
        }
    }

    /// Compare-and-exchange (`compare_exchange`) maximum tracker, `fetch_min` /
    /// `fetch_or` sticky fault word, CTA-uniform early `return`, and last-CTA
    /// `AcqRel` ticket completion barrier (`argmax_rows_finite_fault` pattern).
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (max_val.len() >= 1, fault_words.len() >= 2, done_ticket.len() >= 1),
    )]
    #[verify(
        no_oob_idx(scores, max_val, fault_words, done_ticket, block_seen, final_snap),
        no_data_race(max_val, fault_words, done_ticket, final_snap),
        disjoint_writes(block_seen),
        uniform_barriers
    )]
    #[kernel]
    pub fn cas_max_and_fault_ticket(
        scores: &[f32],
        active_blocks: u32,
        max_val: &mut [DeviceAtomicU32],
        fault_words: &mut [DeviceAtomicU32],
        done_ticket: &mut [DeviceAtomicU32],
        block_seen: &mut [u32],
        final_snap: &mut [DeviceAtomicU32],
    ) {
        let blk = thread::blockIdx_x();
        if blk >= active_blocks {
            return;
        }

        let tid = thread::threadIdx_x() as usize;
        let gid = (blk as usize) * (BLOCK_SIZE as usize) + tid;
        let v = if gid < scores.len() { scores[gid] } else { 0.0 };
        let finite = v.is_finite();
        let bad_mask = warp::ballot(!finite);

        if warp::lane_id() == 0 && bad_mask != 0 && fault_words.len() >= 2 {
            let first_bad_lane = bad_mask.trailing_zeros();
            let bad_gid = (blk * BLOCK_SIZE) + (thread::threadIdx_x() & !31) + first_bad_lane;
            fault_words[0].fetch_min(bad_gid, AtomicOrdering::Relaxed);
            fault_words[1].fetch_or(1u32 << (blk & 31), AtomicOrdering::Relaxed);
        }

        if finite && v > 0.0 && !max_val.is_empty() {
            let cand = v.to_bits();
            let mut cur = max_val[0].load(AtomicOrdering::Relaxed);
            while cur < cand {
                match max_val[0].compare_exchange(
                    cur,
                    cand,
                    AtomicOrdering::Relaxed,
                    AtomicOrdering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(actual) => cur = actual,
                }
            }
        }
        thread::sync_threads();

        if tid == 0 {
            let blk_idx = blk as usize;
            if blk_idx < block_seen.len() {
                block_seen[blk_idx] = 1;
            }
            if !done_ticket.is_empty() {
                let prev = done_ticket[0].fetch_add(1, AtomicOrdering::AcqRel);
                if prev + 1 == active_blocks {
                    done_ticket[0].store(0, AtomicOrdering::Relaxed);
                    if !final_snap.is_empty() && !max_val.is_empty() {
                        final_snap[0].store(
                            max_val[0].load(AtomicOrdering::Relaxed),
                            AtomicOrdering::Relaxed,
                        );
                    }
                }
            }
        }
    }

    /// Global slice of `.cta`-scoped `BlockAtomicU32` counters partitioned by
    /// `blockIdx_x()`: Lean 4 proves `b1 = b2 -> bx1 = bx2 && by1 = by2`, so
    /// `.cta` scope is morally strong without paying `.gpu` L2 coherence traffic!
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
    )]
    #[verify(no_oob_idx(data, block_sums), no_data_race(block_sums))]
    #[kernel]
    pub fn per_block_cta_counters(data: &[u32], block_sums: &mut [BlockAtomicU32]) {
        let gid = thread::index_1d().get();
        let blk = thread::blockIdx_x() as usize;
        if gid < data.len() && blk < block_sums.len() {
            block_sums[blk].fetch_add(data[gid], AtomicOrdering::Relaxed);
        }
    }

    /// Floating-point `DeviceAtomicF32::fetch_add` bucket accumulation paired
    /// with `.sys`-scoped `SystemAtomicU32` invocation counting.
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (buckets.len() >= 8, sys_counter.len() >= 1),
    )]
    #[verify(
        no_oob_idx(values, buckets, sys_counter),
        no_data_race(buckets, sys_counter)
    )]
    #[kernel]
    pub fn float_and_system_atomics(
        values: &[f32],
        buckets: &mut [DeviceAtomicF32],
        sys_counter: &mut [SystemAtomicU32],
    ) {
        let gid = thread::index_1d().get();
        if gid < values.len() {
            let b = gid & 7;
            if b < buckets.len() {
                buckets[b].fetch_add(values[gid], AtomicOrdering::Relaxed);
            }
            if !sys_counter.is_empty() {
                sys_counter[0].fetch_add(1, AtomicOrdering::Relaxed);
            }
        }
    }
}

/// Host test runner verifying all five Lean-proved atomic & warp-vote kernels against CPU oracles.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Provably Safe GPU Atomics & Warp Votes (`safe_atomics`) ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = unsafe { kernels::load(&ctx) }?;
    let cfg_all = LaunchConfig1D::new(NUM_BLOCKS, BLOCK_SIZE, 0);
    let cfg_1x = LaunchConfig1D::new(NUM_BLOCKS, BLOCK_SIZE, 0);

    // 1. Two-level 256-bin histogram (`SharedArray<BlockAtomicU32, 256>` -> `&mut [DeviceAtomicU32]`):
    let host_data: Vec<u32> = (0..NUM_ELEMS)
        .map(|i| ((i * 37 + 13) % 256) as u32)
        .collect();
    let mut expected_bins = vec![0u32; NUM_BINS];
    for &v in &host_data {
        expected_bins[(v as usize) & 255] += 1;
    }
    let d_data = DeviceBuffer::from_host(&stream, &host_data)?;
    let mut d_bins =
        DeviceBuffer::from_host(&stream, &[0u32; NUM_BINS])?.wrap_elem::<DeviceAtomicU32>();
    let prep_hist = module.prepare_two_level_histogram(cfg_all)?;
    module.two_level_histogram(&stream, &prep_hist, &d_data, &mut d_bins)?;
    let got_bins = d_bins.peel_elem::<u32>().to_host_vec(&stream)?;
    assert_eq!(got_bins, expected_bins, "two_level_histogram mismatch");
    println!(
        "  [1/5] two_level_histogram:      PASS (256 bins matched CPU oracle across {NUM_ELEMS} items)"
    );

    // 2. Warp-ballot stream compaction (`warp::ballot` + `lanemask_lt` + `DeviceAtomicU32`):
    let n_compact = (NUM_BLOCKS * BLOCK_SIZE) as usize;
    let host_in: Vec<u32> = (0..n_compact).map(|i| (i as u32) % 100).collect();
    let threshold = 70u32;
    let mut expected_kept: Vec<u32> = host_in.iter().copied().filter(|&x| x > threshold).collect();
    expected_kept.sort_unstable();
    let d_in = DeviceBuffer::from_host(&stream, &host_in)?;
    let mut d_cnt = DeviceBuffer::from_host(&stream, &[0u32])?.wrap_elem::<DeviceAtomicU32>();
    let mut d_vals =
        DeviceBuffer::from_host(&stream, &vec![0u32; n_compact])?.wrap_elem::<DeviceAtomicU32>();
    let prep_comp = module.prepare_ballot_warp_compaction(cfg_1x)?;
    module.ballot_warp_compaction(
        &stream,
        &prep_comp,
        &d_in,
        threshold,
        &mut d_cnt,
        &mut d_vals,
    )?;
    let got_cnt = d_cnt.peel_elem::<u32>().to_host_vec(&stream)?[0] as usize;
    let mut got_vals = d_vals.peel_elem::<u32>().to_host_vec(&stream)?;
    got_vals.truncate(got_cnt);
    got_vals.sort_unstable();
    assert_eq!(got_cnt, expected_kept.len(), "compacted count mismatch");
    assert_eq!(got_vals, expected_kept, "compacted elements mismatch");
    println!(
        "  [2/5] ballot_warp_compaction:   PASS ({got_cnt} > {threshold} elements compacted cleanly)"
    );

    // 3. CAS max + sticky fault mask + last-CTA AcqRel ticket reset:
    let active_blocks = NUM_BLOCKS - 2;
    let active_len = (active_blocks * BLOCK_SIZE) as usize;
    let mut host_scores: Vec<f32> = (0..n_compact)
        .map(|i| ((i % 500) as f32) * 0.25 + 1.0)
        .collect();
    host_scores[300] = f32::NAN;
    host_scores[900] = f32::INFINITY;
    let expected_max_bits = host_scores[..active_len]
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(0.0f32, f32::max)
        .to_bits();
    let d_scores = DeviceBuffer::from_host(&stream, &host_scores)?;
    let mut d_max = DeviceBuffer::from_host(&stream, &[0u32])?.wrap_elem::<DeviceAtomicU32>();
    let mut d_fault =
        DeviceBuffer::from_host(&stream, &[u32::MAX, 0u32])?.wrap_elem::<DeviceAtomicU32>();
    let mut d_ticket = DeviceBuffer::from_host(&stream, &[0u32])?.wrap_elem::<DeviceAtomicU32>();
    let mut d_seen = DeviceBuffer::from_host(&stream, &[0u32; NUM_BLOCKS as usize])?;
    let mut d_snap = DeviceBuffer::from_host(&stream, &[0u32])?.wrap_elem::<DeviceAtomicU32>();
    let prep_cas = module.prepare_cas_max_and_fault_ticket(cfg_1x)?;
    module.cas_max_and_fault_ticket(
        &stream,
        &prep_cas,
        &d_scores,
        active_blocks,
        &mut d_max,
        &mut d_fault,
        &mut d_ticket,
        &mut d_seen,
        &mut d_snap,
    )?;
    let got_max = d_max.peel_elem::<u32>().to_host_vec(&stream)?[0];
    let got_fault = d_fault.peel_elem::<u32>().to_host_vec(&stream)?;
    let got_ticket = d_ticket.peel_elem::<u32>().to_host_vec(&stream)?[0];
    let got_seen = d_seen.to_host_vec(&stream)?;
    let got_snap = d_snap.peel_elem::<u32>().to_host_vec(&stream)?[0];
    assert_eq!(got_max, expected_max_bits, "CAS max mismatch");
    assert_eq!(got_snap, expected_max_bits, "last-CTA final_snap mismatch");
    assert_eq!(
        &got_seen[..active_blocks as usize],
        &vec![1u32; active_blocks as usize][..],
        "active block_seen mismatch"
    );
    assert_eq!(
        &got_seen[active_blocks as usize..],
        &[0u32, 0u32],
        "early-returned blocks should not write block_seen"
    );
    assert_eq!(got_fault[0], 300, "sticky fault earliest index mismatch");
    assert_eq!(
        got_fault[1],
        (1u32 << (300 / BLOCK_SIZE)) | (1u32 << (900 / BLOCK_SIZE)),
        "sticky fault block mask mismatch"
    );
    assert_eq!(got_ticket, 0, "last-CTA ticket did not reset to 0");
    println!(
        "  [3/5] cas_max_and_fault_ticket: PASS (max = {}, earliest fault = {}, ticket reset = 0)",
        f32::from_bits(got_max),
        got_fault[0]
    );

    // 4. Per-CTA `.cta`-scoped `BlockAtomicU32` global slice (`block_sums[blockIdx_x()]`):
    let mut d_blk_sums = DeviceBuffer::from_host(&stream, &[0u32; NUM_BLOCKS as usize])?
        .wrap_elem::<BlockAtomicU32>();
    let prep_cta = module.prepare_per_block_cta_counters(cfg_1x)?;
    module.per_block_cta_counters(&stream, &prep_cta, &d_in, &mut d_blk_sums)?;
    let got_blk_sums = d_blk_sums.peel_elem::<u32>().to_host_vec(&stream)?;
    for (b, &got) in got_blk_sums.iter().enumerate() {
        let start = b * (BLOCK_SIZE as usize);
        let exp: u32 = host_in[start..start + (BLOCK_SIZE as usize)].iter().sum();
        assert_eq!(got, exp, "per_block_cta_counters mismatch at block {b}");
    }
    println!(
        "  [4/5] per_block_cta_counters:   PASS ({NUM_BLOCKS} per-CTA `.cta` counters matched)"
    );

    // 5. Floating-point `DeviceAtomicF32` + `.sys`-scoped `SystemAtomicU32`:
    let host_f32: Vec<f32> = (0..n_compact).map(|i| ((i & 7) + 1) as f32 * 0.5).collect();
    let d_f32 = DeviceBuffer::from_host(&stream, &host_f32)?;
    let mut d_buckets =
        DeviceBuffer::from_host(&stream, &[0.0f32; 8])?.wrap_elem::<DeviceAtomicF32>();
    let mut d_sys = DeviceBuffer::from_host(&stream, &[0u32])?.wrap_elem::<SystemAtomicU32>();
    let prep_f32 = module.prepare_float_and_system_atomics(cfg_1x)?;
    module.float_and_system_atomics(&stream, &prep_f32, &d_f32, &mut d_buckets, &mut d_sys)?;
    let got_buckets = d_buckets.peel_elem::<f32>().to_host_vec(&stream)?;
    let got_sys = d_sys.peel_elem::<u32>().to_host_vec(&stream)?[0];
    assert_eq!(got_sys, n_compact as u32, "SystemAtomicU32 count mismatch");
    let items_per_bucket = (n_compact / 8) as f32;
    for (b, &got) in got_buckets.iter().enumerate() {
        let exp = items_per_bucket * ((b + 1) as f32 * 0.5);
        assert!(
            (got - exp).abs() < 1e-2,
            "float_and_system_atomics mismatch at bucket {b}: got {got}, expected {exp}"
        );
    }
    println!(
        "  [5/5] float_and_system_atomics: PASS (8 FP32 buckets + SystemAtomicU32 = {got_sys})"
    );

    println!(
        "\nAll 5 verified atomic & warp-vote kernels passed with 0 `unsafe` blocks inside kernels!"
    );
    Ok(())
}
