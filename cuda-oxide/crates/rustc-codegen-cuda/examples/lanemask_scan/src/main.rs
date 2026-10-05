/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lane-position masks (`%lanemask_lt`/`le`/`eq`/`ge`/`gt`) and warp prefix sums.
//!
//! These five read-only special registers return a 32-bit mask describing the
//! calling lane's position within its warp. Their headline use is the warp-level
//! exclusive prefix sum that powers stream compaction:
//!
//! ```text
//!   ballot = ballot_sync(mask, keep)      // bit k = lane k's predicate
//!   rank   = (ballot & lanemask_lt()).count_ones()
//! ```
//!
//! `rank` is exactly the number of earlier lanes that also kept their element,
//! i.e. the compacted output slot for this lane — computed warp-wide with one
//! ballot and one popcount, no shared memory and no loop.
//!
//! Build and run with:
//!   cargo oxide run lanemask_scan

use cuda_device::{DisjointSlice, kernel, thread, warp};
use cuda_host::cuda_module;

const FULL_MASK: u32 = 0xffff_ffff;

// =============================================================================
// KERNELS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Every lane writes its own `%lanemask_lt` register. For lane `i` this is
    /// `(1 << i) - 1` — the set of lanes strictly before it. A direct, per-lane
    /// readout of the special register with no collective involved.
    #[kernel]
    pub fn lanemask_lt_values(mut out: DisjointSlice<u32>) {
        let gid = thread::index_1d();
        if gid.in_bounds(out.len()) {
            unsafe {
                *out.get_unchecked_mut(gid.get()) = warp::lanemask_lt();
            }
        }
    }

    /// Each lane writes all five lanemask register values into a flat buffer
    /// laid out as [lt, le, eq, ge, gt] per lane (5 * 32 = 160 u32s per warp).
    ///
    /// Used to verify that each of the five registers reads correctly for every
    /// lane position, not just `lt`.
    #[kernel]
    pub fn all_lanemasks(mut out: DisjointSlice<u32>) {
        let gid = thread::index_1d();
        let n_lanes = out.len() / 5;
        if gid.in_bounds(n_lanes) {
            let base = gid.get() * 5;
            unsafe {
                *out.get_unchecked_mut(base) = warp::lanemask_lt();
                *out.get_unchecked_mut(base + 1) = warp::lanemask_le();
                *out.get_unchecked_mut(base + 2) = warp::lanemask_eq();
                *out.get_unchecked_mut(base + 3) = warp::lanemask_ge();
                *out.get_unchecked_mut(base + 4) = warp::lanemask_gt();
            }
        }
    }

    /// Warp exclusive prefix sum / stream-compaction rank.
    ///
    /// Each lane "keeps" its element when `data[gid] != 0`. The kept lanes are
    /// gathered with `ballot_sync`, and `(ballot & lanemask_lt()).count_ones()`
    /// gives each lane the number of kept lanes *before* it — its slot in the
    /// compacted output.
    #[kernel]
    pub fn warp_compact_rank(data: &[u32], mut ranks: DisjointSlice<u32>) {
        let gid = thread::index_1d();
        let n = ranks.len();

        // Launched with exactly `n` threads (a multiple of 32), so every lane is
        // in bounds and joins the full-warp ballot.
        let keep = gid.in_bounds(n) && data[gid.get()] != 0;
        let lane = warp::lane_id();
        let group = lane / 4;
        let expected_group_mask = 0xfu32 << (group * 4);
        // High-only values catch an accidental 32-bit match lowering.
        let wide_group = (group as u64) << 32;
        let wide_lane = (lane as u64) << 32;
        let active = warp::active_mask();
        let ballot = warp::ballot_sync(FULL_MASK, keep);
        let all_lanes_joined = warp::all_sync(FULL_MASK, true);
        let lane_zero_joined = warp::any_sync(FULL_MASK, lane == 0);
        let matches_ok = (warp::match_any_sync(FULL_MASK, group) == expected_group_mask)
            & (warp::match_any_i64_sync(FULL_MASK, wide_group) == expected_group_mask)
            & (warp::match_all_sync(FULL_MASK, 42) == FULL_MASK)
            & (warp::match_all_i64_sync(FULL_MASK, wide_lane) == 0);
        let rank = if all_lanes_joined & lane_zero_joined & (active == FULL_MASK) & matches_ok {
            (ballot & warp::lanemask_lt()).count_ones()
        } else {
            u32::MAX
        };

        if gid.in_bounds(n) {
            unsafe {
                *ranks.get_unchecked_mut(gid.get()) = rank;
            }
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() {
    use cuda_core::simt::LaunchConfig;
    use cuda_core::{CudaContext, DeviceBuffer};

    println!("=== Lane-Position Masks & Warp Prefix Sum ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let (major, minor) = ctx.compute_capability().expect("compute capability");
    println!("GPU Compute Capability: sm_{}{}", major, minor);

    const N: usize = 256;
    const WARPS: usize = N / 32;

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let cfg = LaunchConfig {
        block_dim: (32, 1, 1),
        grid_dim: (WARPS as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    // ===== Test 1: raw lanemask_lt readout =====
    println!("\n--- Test 1: lanemask_lt() per lane ---");
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.lanemask_lt_values((stream).as_ref(), cfg, &mut out_dev) }
        .expect("Kernel launch failed");

    let out = out_dev.to_host_vec(&stream).unwrap();
    // For lane i: lanemask_lt == (1 << i) - 1  (lane 0 -> 0, lane 31 -> 0x7fffffff).
    let lt_ok = out.iter().enumerate().all(|(i, &v)| {
        let lane = (i % 32) as u32;
        v == ((1u32 << lane) - 1)
    });
    println!("lane 0..4   : {:08x?}", &out[..4]);
    println!("lane 30,31  : {:08x?}", &out[30..32]);
    if lt_ok {
        println!("✓ lanemask_lt matches (1 << lane) - 1 for all {} lanes", N);
    } else {
        println!("✗ lanemask_lt mismatch!");
        std::process::exit(1);
    }

    // ===== Test 2: warp prefix sum / compaction rank =====
    println!("\n--- Test 2: warp_compact_rank over a keep-mask ---");
    // Keep every 3rd element (arbitrary, just needs a mix of 0/non-0).
    let data_host: Vec<u32> = (0..N).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();
    let data_dev = DeviceBuffer::from_host(&stream, &data_host).unwrap();
    let mut ranks_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.warp_compact_rank((stream).as_ref(), cfg, &data_dev, &mut ranks_dev) }
        .expect("Kernel launch failed");

    let ranks = ranks_dev.to_host_vec(&stream).unwrap();

    // CPU reference: exclusive prefix count of kept lanes, reset at each warp.
    let mut expected = vec![0u32; N];
    for w in 0..WARPS {
        let mut acc = 0u32;
        for lane in 0..32 {
            let idx = w * 32 + lane;
            expected[idx] = acc;
            if data_host[idx] != 0 {
                acc += 1;
            }
        }
    }

    if ranks == expected {
        println!("ranks[0..8]: {:?}", &ranks[..8]);
        println!("✓ warp_compact_rank matches CPU exclusive prefix sum");
    } else {
        println!("✗ rank mismatch!");
        println!("  gpu: {:?}", &ranks[..8]);
        println!("  cpu: {:?}", &expected[..8]);
        std::process::exit(1);
    }

    // ===== Test 3: all five lanemask registers =====
    println!("\n--- Test 3: all five lanemask registers (lt/le/eq/ge/gt) ---");
    // 5 values per lane, one warp (32 lanes).
    let n_lanes: usize = 32;
    let mut all_dev = DeviceBuffer::<u32>::zeroed(&stream, n_lanes * 5).unwrap();
    let cfg1 = LaunchConfig {
        block_dim: (32, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.all_lanemasks((stream).as_ref(), cfg1, &mut all_dev) }
        .expect("Kernel launch failed");

    let all = all_dev.to_host_vec(&stream).unwrap();

    let mut masks_ok = true;
    for lane in 0u32..32 {
        let base = (lane as usize) * 5;
        let lt = all[base];
        let le = all[base + 1];
        let eq = all[base + 2];
        let ge = all[base + 3];
        let gt = all[base + 4];

        // Mathematical invariants for lane i:
        //   lt = bits 0..(i-1)  = (1 << i) - 1
        //   le = bits 0..i      = (2 << i) - 1  [lane 31 wraps to 0xFFFF_FFFF]
        //   eq = bit i only     = 1 << i
        //   ge = bits i..31     = !(lt)
        //   gt = bits (i+1)..31 = !(le)
        let exp_lt = (1u32 << lane).wrapping_sub(1);
        let exp_le = (2u32 << lane).wrapping_sub(1);
        let exp_eq = 1u32 << lane;
        let exp_ge = !exp_lt;
        let exp_gt = !exp_le;

        if lt != exp_lt || le != exp_le || eq != exp_eq || ge != exp_ge || gt != exp_gt {
            println!(
                "✗ lane {lane}: got lt={lt:#010x} le={le:#010x} eq={eq:#010x} ge={ge:#010x} gt={gt:#010x}"
            );
            println!(
                "  expected    lt={exp_lt:#010x} le={exp_le:#010x} eq={exp_eq:#010x} ge={exp_ge:#010x} gt={exp_gt:#010x}"
            );
            masks_ok = false;
        }
    }

    // Print a representative row for readability.
    let sample = |lane: usize| {
        let b = lane * 5;
        format!(
            "lane {:2}: lt={:#010x} le={:#010x} eq={:#010x} ge={:#010x} gt={:#010x}",
            lane,
            all[b],
            all[b + 1],
            all[b + 2],
            all[b + 3],
            all[b + 4]
        )
    };
    println!("{}", sample(0));
    println!("{}", sample(1));
    println!("{}", sample(16));
    println!("{}", sample(31));

    if masks_ok {
        println!("✓ all five lanemask registers correct for all 32 lanes");
    } else {
        std::process::exit(1);
    }

    println!("\nSUCCESS: lane-position masks produced correct warp prefix sums");
}
