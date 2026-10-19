/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cooperative-groups smoke test.
//!
//! 21 device-verified checks for `cuda_device::cooperative_groups`,
//! grouped into three layers:
//!
//! 1. Raw intrinsics (`active_mask`, `match_any_sync`, `match_all_sync`,
//!    `grid::sync`).
//! 2. Typed cooperative-groups handles (`WarpTile<32>`, `WarpTile<16>`,
//!    `this_grid()`).
//! 3. Reductions and scans (`warp_reduce`, `warp_scan`, `block_reduce`,
//!    `block_scan` × `Sum`/`Min`/`Max`/`BitAnd`/`BitOr`/`BitXor` ×
//!    `u32`/`i32`/`f32`).
//!
//! See `README.md` in this crate for the full check list and the
//! op/type matrix. The generated `coop_groups_demo.ptx` is also handy
//! for inspecting how each kernel actually lowers.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::cooperative_groups::{
    ThreadGroup, WarpCollective, block_reduce, block_scan, coalesced_threads,
    ops::{BitAnd, BitOr, BitXor, Max, Min, Sum},
    this_grid, this_thread_block, warp_reduce, warp_scan,
};
use cuda_device::{
    DisjointSlice, SharedArray, cluster_launch, cooperative_launch, grid, kernel, launch_bounds,
    launch_contract, thread, warp,
};
use cuda_host::{cuda_launch, cuda_module};

// =============================================================================
// KERNELS
// =============================================================================

/// Each thread writes the warp's `active_mask()` (full warp → 0xFFFFFFFF).
#[kernel]
pub fn test_active_mask(mut out: DisjointSlice<u32>) {
    let gid = thread::index_1d();
    let mask = warp::active_mask();
    if let Some(slot) = out.get_mut(gid) {
        *slot = mask;
    }
}

/// Each thread reports the lane mask of every other thread sharing its
/// `value`. With `value = lane / 4` the warp is partitioned into 8 buckets
/// of 4 contiguous lanes each, so every thread should see `0xF << (group*4)`.
#[kernel]
pub fn test_match_any(mut out: DisjointSlice<u32>) {
    let gid = thread::index_1d();
    let lane = warp::lane_id();
    let value: u32 = lane / 4;
    let mask = warp::match_any_sync(u32::MAX, value);
    if let Some(slot) = out.get_mut(gid) {
        *slot = mask;
    }
}

/// All lanes use the same value, so `match_all_sync` returns the full mask.
#[kernel]
pub fn test_match_all(mut out: DisjointSlice<u32>) {
    let gid = thread::index_1d();
    let mask = warp::match_all_sync(u32::MAX, 42u32);
    if let Some(slot) = out.get_mut(gid) {
        *slot = mask;
    }
}

// The grid-sync kernels live in a `#[cuda_module]` module so their launches
// go through the typed path. `#[cooperative_launch]` makes every generated
// launch method use a cooperative launch (`cuLaunchKernelEx` with
// `CU_LAUNCH_ATTRIBUTE_COOPERATIVE`), which `grid::sync()` requires.
#[cuda_module]
mod grid_sync_kernels {
    use super::*;

    /// Smoke test for `grid::sync()`. Each block's thread 0 writes a marker
    /// (`blockIdx.x + 1`), the grid synchronises, then thread 0 reads every
    /// other block's marker via the raw base pointer and writes the sum into
    /// `out[blockIdx.x]`. Expected value: `gridDim.x * (gridDim.x + 1) / 2`.
    #[kernel]
    #[cooperative_launch]
    pub fn test_grid_sync(mut markers: DisjointSlice<u32>, mut out: DisjointSlice<u32>) {
        let block_id = thread::blockIdx_x();
        let n = thread::gridDim_x();

        if thread::threadIdx_x() == 0 {
            unsafe {
                *markers.get_unchecked_mut(block_id as usize) = block_id + 1;
            }
        }

        grid::sync();

        if thread::threadIdx_x() == 0 {
            let base = markers.as_mut_ptr() as *const u32;
            let mut sum: u32 = 0;
            let mut i: u32 = 0;
            while i < n {
                unsafe {
                    sum = sum.wrapping_add(*base.add(i as usize));
                }
                i += 1;
            }
            unsafe {
                *out.get_unchecked_mut(block_id as usize) = sum;
            }
        }
    }

    /// `this_grid().sync()` must produce the same observable result as the
    /// raw `grid::sync()` test above: every block sees every other block's
    /// pre-barrier marker write.
    #[kernel]
    #[cooperative_launch]
    pub fn test_typed_grid_sync(mut markers: DisjointSlice<u32>, mut out: DisjointSlice<u32>) {
        let grid_handle = this_grid();
        let block_id = thread::blockIdx_x();
        let n = thread::gridDim_x();

        if thread::threadIdx_x() == 0 {
            unsafe {
                *markers.get_unchecked_mut(block_id as usize) = block_id + 1;
            }
        }

        grid_handle.sync();

        if thread::threadIdx_x() == 0 {
            let base = markers.as_mut_ptr() as *const u32;
            let mut sum: u32 = 0;
            let mut i: u32 = 0;
            while i < n {
                unsafe {
                    sum = sum.wrapping_add(*base.add(i as usize));
                }
                i += 1;
            }
            unsafe {
                *out.get_unchecked_mut(block_id as usize) = sum;
            }
        }
    }

    /// Combined cluster + cooperative launch: each cluster of 2 blocks writes
    /// markers, then `grid::sync()` publishes them to every block. Preparation
    /// must accept both attributes and prove the grid fits in concurrent
    /// cluster capacity (Hopper+).
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[cooperative_launch]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn test_cluster_coop_grid_sync(
        mut markers: DisjointSlice<u32>,
        mut out: DisjointSlice<u32>,
    ) {
        let block_id = thread::blockIdx_x();
        let n = thread::gridDim_x();

        if thread::threadIdx_x() == 0 {
            // SAFETY: the host sizes `markers` to exactly gridDim.x elements,
            // so `block_id` indexes in bounds, and only thread 0 of each
            // block writes its own slot, so the writes are disjoint.
            unsafe {
                *markers.get_unchecked_mut(block_id as usize) = block_id + 1;
            }
        }

        grid::sync();

        if thread::threadIdx_x() == 0 {
            let base = markers.as_mut_ptr() as *const u32;
            let mut sum: u32 = 0;
            let mut i: u32 = 0;
            while i < n {
                // SAFETY: `i < n = gridDim.x` and `markers` holds gridDim.x
                // elements, so `base.add(i)` stays in bounds. Every block's
                // marker write happened before `grid::sync()`, so the
                // cross-block reads are data-race free.
                unsafe {
                    sum = sum.wrapping_add(*base.add(i as usize));
                }
                i += 1;
            }
            // SAFETY: the host sizes `out` to exactly gridDim.x elements, so
            // `block_id` indexes in bounds, and only thread 0 of each block
            // writes its own slot.
            unsafe {
                *out.get_unchecked_mut(block_id as usize) = sum;
            }
        }
    }
}

// =============================================================================
// Typed cooperative-groups kernels
// =============================================================================

/// Check the full-warp fast path for typed ballot and match collectives.
/// With `predicate = lane_id() & 1`, every lane should report `0xAAAAAAAA`.
#[kernel]
pub fn test_typed_warp32_ballot(mut out: DisjointSlice<u32>) {
    let gid = thread::index_1d();
    let warp_tile = this_thread_block().tiled_partition::<32>();
    let rank = warp_tile.thread_rank();
    let mask = warp_tile.ballot((warp::lane_id() & 1) != 0);
    let expected_match = 0xFu32 << ((rank / 4) * 4);

    let ok = (warp_tile.match_any(rank / 4) == expected_match)
        & (warp_tile.match_any_i64(((rank / 4) as u64) << 40) == expected_match)
        & (warp_tile.match_all(42) == u32::MAX)
        & (warp_tile.match_all_i64(42u64 << 40) == u32::MAX)
        & (warp_tile.match_all(rank) == 0)
        & (warp_tile.match_all_i64((rank as u64) << 40) == 0);

    if let Some(slot) = out.get_mut(gid) {
        *slot = if ok { mask } else { u32::MAX };
    }
}

/// `WarpTile<16>::ballot(predicate)` returns a *tile-relative* mask: bit
/// `k` is set iff the lane at tile-rank `k` had `predicate == true`.
/// With `predicate = lane_id() & 1` every tile should see `0xAAAA`.
#[kernel]
pub fn test_typed_warp16_ballot(mut out: DisjointSlice<u32>) {
    let gid = thread::index_1d();
    let tile = this_thread_block().tiled_partition::<16>();
    let mask = tile.ballot((warp::lane_id() & 1) != 0);
    if let Some(slot) = out.get_mut(gid) {
        *slot = mask;
    }
}

/// Check tile-relative shuffles plus sparse-group ballot packing and shuffle
/// source handling. The stored value remains each 16-lane tile's lane-0 broadcast.
#[kernel]
pub fn test_typed_warp16_shfl(mut out: DisjointSlice<u32>) {
    let gid = thread::index_1d();
    let tile = this_thread_block().tiled_partition::<16>();
    let lane = warp::lane_id();
    tile.sync();
    let broadcast = tile.shfl(lane, 0);
    let rank = tile.thread_rank();
    let expected_match = 0xFu32 << ((rank / 4) * 4);

    let tile_ok = (tile.shfl_xor(lane, 1) == (lane ^ 1))
        & (tile.shfl_xor(lane, 16) == lane)
        & (tile.match_any(rank / 4) == expected_match)
        & (tile.match_any_i64(((rank / 4) as u64) << 40) == expected_match)
        & (tile.match_all(42) == 0xFFFF)
        & (tile.match_all_i64(42u64 << 40) == 0xFFFF)
        & (tile.match_all(rank) == 0)
        & (tile.match_all_i64((rank as u64) << 40) == 0)
        & (tile.shfl_down(lane, 1) == if rank < 15 { lane + 1 } else { lane })
        & (tile.shfl_up(lane, 1) == if rank > 0 { lane - 1 } else { lane })
        & (tile.shfl_xor_f32(lane as f32, 16) == lane as f32)
        & (tile.shfl_down_f32(lane as f32, 1)
            == if rank < 15 {
                (lane + 1) as f32
            } else {
                lane as f32
            })
        & (tile.shfl_up_f32(lane as f32, 1)
            == if rank > 0 {
                (lane - 1) as f32
            } else {
                lane as f32
            });
    let coalesced_ok = if lane & 1 == 0 {
        let group = coalesced_threads();
        let group_rank = group.thread_rank();
        let expected_match = 0xFu32 << ((group_rank / 4) * 4);

        (group.ballot((lane & 2) != 0) == 0xAAAA)
            & (group.match_any(group_rank / 4) == expected_match)
            & (group.match_any_i64(((group_rank / 4) as u64) << 40) == expected_match)
            & (group.match_all(42) == 0xFFFF)
            & (group.match_all_i64(42u64 << 40) == 0xFFFF)
            & (group.match_all(group_rank) == 0)
            & (group.match_all_i64((group_rank as u64) << 40) == 0)
            & (group.shfl(lane, group.size()) == lane)
            & (group.shfl_xor(lane, 1) == lane)
            & (group.shfl_down(lane, 1) == lane)
            & (group.shfl_up(lane, 1) == lane)
            & (group.shfl_down(lane, 2) == if lane < 30 { lane + 2 } else { lane })
            & (group.shfl_up(lane, 2) == if lane >= 2 { lane - 2 } else { lane })
            & (group.shfl_xor_f32(lane as f32, 1) == lane as f32)
            & (group.shfl_down_f32(lane as f32, 1) == lane as f32)
            & (group.shfl_up_f32(lane as f32, 1) == lane as f32)
    } else {
        true
    };
    // Include the top physical lane and gaps of different sizes, so the
    // device regression cannot pass by merely shifting an even-lane ballot.
    let irregular_ok = if (0x8010_0089u32 & (1u32 << lane)) != 0 {
        let group = coalesced_threads();
        let rank = group.thread_rank();
        let expected_match = if rank & 1 == 0 { 0x15 } else { 0x0A };

        (group.ballot(lane == 3 || lane == 20 || lane == 31) == 0x1A)
            & (group.ballot(true) == 0x1F)
            & (group.ballot(false) == 0)
            & (group.match_any(rank & 1) == expected_match)
            & (group.match_any_i64(((rank & 1) as u64) << 40) == expected_match)
            & (group.match_all(42) == 0x1F)
            & (group.match_all_i64(42u64 << 40) == 0x1F)
            & (group.match_all(rank) == 0)
            & (group.match_all_i64((rank as u64) << 40) == 0)
    } else {
        true
    };
    if let Some(slot) = out.get_mut(gid) {
        *slot = if tile_ok & coalesced_ok & irregular_ok {
            broadcast
        } else {
            u32::MAX
        };
    }
}

// NOTE: `test_typed_grid_sync` lives in the `grid_sync_kernels` module above,
// next to its raw `grid::sync()` counterpart, because both need the
// `#[cooperative_launch]` typed launch path.

/// Probe `Grid::size()` / `Grid::thread_rank()` from every thread.
/// `out[i]` records `thread_rank()` for the thread whose `index_1d == i`;
/// for a 1D launch the recorded value should equal `i`.
#[kernel]
pub fn test_typed_grid_rank(mut out: DisjointSlice<u32>) {
    let gid = thread::index_1d();
    let g = this_grid();
    let rank = g.thread_rank();
    if let Some(slot) = out.get_mut(gid) {
        *slot = rank;
    }
}

// =============================================================================
// Reductions and scans
// =============================================================================
//
// Tests use this layout:
//
//   warp_reduce_<T>: 1 row per warp,   6 cells (u32) or 3 cells (i32, f32).
//   warp_scan_<T>:   1 row per thread, 6 cells (u32) or 3 cells (i32, f32).
//   block_reduce_<T>: 1 row per block, 6 cells (u32) or 3 cells (i32, f32).
//   block_scan_<T>:  1 row per thread, 6 cells (u32) or 3 cells (i32, f32).
//
// Block kernels launch with block_dim = 96 (NUM_WARPS = 3) so:
//   - BitAnd input `!(1<<lane)`   → AND across 3 of each bit clears all bits.
//   - BitOr  input `1<<lane`      → OR  across 3 of each bit sets   all bits.
//   - BitXor input `1<<lane`      → XOR across 3 of each bit (odd) sets all
//                                    bits — non-degenerate (a 32-warp block
//                                    would have an even count and degenerate
//                                    to identity, masking bugs).

// ----- warp_reduce -----

/// `warp_reduce` over `WarpTile<32>` for every supported `u32` op.
/// One row per warp; cells = `[Sum, Min, Max, BitAnd, BitOr, BitXor]`.
#[kernel]
pub fn test_warp_reduce_u32(mut out: DisjointSlice<u32>) {
    let warp = this_thread_block().tiled_partition::<32>();
    let lane = warp.thread_rank();
    let warp_idx = thread::index_1d().get() / 32;

    let v_lane = lane;
    let v_one = 1u32 << lane;
    let v_inv = !v_one;

    let r_sum = warp_reduce::<u32, Sum, _>(&warp, v_lane);
    let r_min = warp_reduce::<u32, Min, _>(&warp, v_lane);
    let r_max = warp_reduce::<u32, Max, _>(&warp, v_lane);
    let r_and = warp_reduce::<u32, BitAnd, _>(&warp, v_inv);
    let r_or = warp_reduce::<u32, BitOr, _>(&warp, v_one);
    let r_xor = warp_reduce::<u32, BitXor, _>(&warp, v_one);

    if lane == 0 {
        let base = warp_idx * 6;
        unsafe {
            *out.get_unchecked_mut(base) = r_sum;
            *out.get_unchecked_mut(base + 1) = r_min;
            *out.get_unchecked_mut(base + 2) = r_max;
            *out.get_unchecked_mut(base + 3) = r_and;
            *out.get_unchecked_mut(base + 4) = r_or;
            *out.get_unchecked_mut(base + 5) = r_xor;
        }
    }
}

/// `warp_reduce` over `WarpTile<32>` for every supported `i32` op.
/// One row per warp; cells = `[Sum, Min, Max]`.
#[kernel]
pub fn test_warp_reduce_i32(mut out: DisjointSlice<i32>) {
    let warp = this_thread_block().tiled_partition::<32>();
    let lane = warp.thread_rank();
    let warp_idx = thread::index_1d().get() / 32;

    // value = lane - 16: range [-16, 15] gives meaningful Min < 0 < Max.
    let v = (lane as i32) - 16;

    let r_sum = warp_reduce::<i32, Sum, _>(&warp, v);
    let r_min = warp_reduce::<i32, Min, _>(&warp, v);
    let r_max = warp_reduce::<i32, Max, _>(&warp, v);

    if lane == 0 {
        let base = warp_idx * 3;
        unsafe {
            *out.get_unchecked_mut(base) = r_sum;
            *out.get_unchecked_mut(base + 1) = r_min;
            *out.get_unchecked_mut(base + 2) = r_max;
        }
    }
}

/// `warp_reduce` over `WarpTile<32>` for every supported `f32` op.
/// One row per warp; cells = `[Sum, Min, Max]`.
#[kernel]
pub fn test_warp_reduce_f32(mut out: DisjointSlice<f32>) {
    let warp = this_thread_block().tiled_partition::<32>();
    let lane = warp.thread_rank();
    let warp_idx = thread::index_1d().get() / 32;

    let v = lane as f32;

    let r_sum = warp_reduce::<f32, Sum, _>(&warp, v);
    let r_min = warp_reduce::<f32, Min, _>(&warp, v);
    let r_max = warp_reduce::<f32, Max, _>(&warp, v);

    if lane == 0 {
        let base = warp_idx * 3;
        unsafe {
            *out.get_unchecked_mut(base) = r_sum;
            *out.get_unchecked_mut(base + 1) = r_min;
            *out.get_unchecked_mut(base + 2) = r_max;
        }
    }
}

// ----- warp_scan -----

/// `warp_scan` (inclusive) over `WarpTile<32>` for every supported `u32` op.
/// One row per thread; cells = `[Sum, Min, Max, BitAnd, BitOr, BitXor]`.
#[kernel]
pub fn test_warp_scan_u32(mut out: DisjointSlice<u32>) {
    let warp = this_thread_block().tiled_partition::<32>();
    let lane = warp.thread_rank();
    let gid = thread::index_1d().get();

    let v_lane = lane;
    let v_one = 1u32 << lane;
    let v_inv = !v_one;

    let r_sum = warp_scan::<u32, Sum, _>(&warp, v_lane);
    let r_min = warp_scan::<u32, Min, _>(&warp, v_lane);
    let r_max = warp_scan::<u32, Max, _>(&warp, v_lane);
    let r_and = warp_scan::<u32, BitAnd, _>(&warp, v_inv);
    let r_or = warp_scan::<u32, BitOr, _>(&warp, v_one);
    let r_xor = warp_scan::<u32, BitXor, _>(&warp, v_one);

    let base = gid * 6;
    unsafe {
        *out.get_unchecked_mut(base) = r_sum;
        *out.get_unchecked_mut(base + 1) = r_min;
        *out.get_unchecked_mut(base + 2) = r_max;
        *out.get_unchecked_mut(base + 3) = r_and;
        *out.get_unchecked_mut(base + 4) = r_or;
        *out.get_unchecked_mut(base + 5) = r_xor;
    }
}

/// `warp_scan` (inclusive) over `WarpTile<32>` for every supported `i32` op.
/// One row per thread; cells = `[Sum, Min, Max]`.
#[kernel]
pub fn test_warp_scan_i32(mut out: DisjointSlice<i32>) {
    let warp = this_thread_block().tiled_partition::<32>();
    let lane = warp.thread_rank();
    let gid = thread::index_1d().get();

    let v = (lane as i32) - 16;

    let r_sum = warp_scan::<i32, Sum, _>(&warp, v);
    let r_min = warp_scan::<i32, Min, _>(&warp, v);
    let r_max = warp_scan::<i32, Max, _>(&warp, v);

    let base = gid * 3;
    unsafe {
        *out.get_unchecked_mut(base) = r_sum;
        *out.get_unchecked_mut(base + 1) = r_min;
        *out.get_unchecked_mut(base + 2) = r_max;
    }
}

/// `warp_scan` (inclusive) over `WarpTile<32>` for every supported `f32` op.
/// One row per thread; cells = `[Sum, Min, Max]`.
#[kernel]
pub fn test_warp_scan_f32(mut out: DisjointSlice<f32>) {
    let warp = this_thread_block().tiled_partition::<32>();
    let lane = warp.thread_rank();
    let gid = thread::index_1d().get();

    let v = lane as f32;

    let r_sum = warp_scan::<f32, Sum, _>(&warp, v);
    let r_min = warp_scan::<f32, Min, _>(&warp, v);
    let r_max = warp_scan::<f32, Max, _>(&warp, v);

    let base = gid * 3;
    unsafe {
        *out.get_unchecked_mut(base) = r_sum;
        *out.get_unchecked_mut(base + 1) = r_min;
        *out.get_unchecked_mut(base + 2) = r_max;
    }
}

// ----- block_reduce -----
//
// Block dim 96 → NUM_WARPS = 3. SMEM declared once per kernel and reused
// across ops; an explicit `block.sync()` between calls satisfies the
// reuse contract documented on `block_reduce`.

/// `block_reduce` for every supported `u32` op.
#[kernel]
pub fn test_block_reduce_u32(mut out: DisjointSlice<u32>) {
    static mut SMEM: SharedArray<u32, 3> = SharedArray::UNINIT;
    let block = this_thread_block();
    let tid = thread::threadIdx_x();
    let block_id = thread::blockIdx_x() as usize;

    let v = tid;
    let v_one = 1u32 << (tid & 31);
    let v_inv = !v_one;

    // SAFETY: every block thread uses this allocation; barriers separate reuse.

    let r_sum = unsafe { block_reduce::<u32, Sum, _>(&block, v, &raw mut SMEM) };
    block.sync();
    // SAFETY: every block thread uses this allocation; barriers separate reuse.
    let r_min = unsafe { block_reduce::<u32, Min, _>(&block, v, &raw mut SMEM) };
    block.sync();
    // SAFETY: every block thread uses this allocation; barriers separate reuse.
    let r_max = unsafe { block_reduce::<u32, Max, _>(&block, v, &raw mut SMEM) };
    block.sync();
    // SAFETY: every block thread uses this allocation; barriers separate reuse.
    let r_and = unsafe { block_reduce::<u32, BitAnd, _>(&block, v_inv, &raw mut SMEM) };
    block.sync();
    // SAFETY: every block thread uses this allocation; barriers separate reuse.
    let r_or = unsafe { block_reduce::<u32, BitOr, _>(&block, v_one, &raw mut SMEM) };
    block.sync();
    // SAFETY: every block thread uses this allocation; barriers separate reuse.
    let r_xor = unsafe { block_reduce::<u32, BitXor, _>(&block, v_one, &raw mut SMEM) };

    if tid == 0 {
        let base = block_id * 6;
        unsafe {
            *out.get_unchecked_mut(base) = r_sum;
            *out.get_unchecked_mut(base + 1) = r_min;
            *out.get_unchecked_mut(base + 2) = r_max;
            *out.get_unchecked_mut(base + 3) = r_and;
            *out.get_unchecked_mut(base + 4) = r_or;
            *out.get_unchecked_mut(base + 5) = r_xor;
        }
    }
}

/// `block_reduce` for every supported `i32` op.
#[kernel]
pub fn test_block_reduce_i32(mut out: DisjointSlice<i32>) {
    static mut SMEM: SharedArray<i32, 3> = SharedArray::UNINIT;
    let block = this_thread_block();
    let tid = thread::threadIdx_x();
    let block_id = thread::blockIdx_x() as usize;

    // value = tid - 48: range [-48, 47] over a 96-thread block.
    let v = (tid as i32) - 48;

    // SAFETY: every block thread uses this allocation; barriers separate reuse.

    let r_sum = unsafe { block_reduce::<i32, Sum, _>(&block, v, &raw mut SMEM) };
    block.sync();
    // SAFETY: every block thread uses this allocation; barriers separate reuse.
    let r_min = unsafe { block_reduce::<i32, Min, _>(&block, v, &raw mut SMEM) };
    block.sync();
    // SAFETY: every block thread uses this allocation; barriers separate reuse.
    let r_max = unsafe { block_reduce::<i32, Max, _>(&block, v, &raw mut SMEM) };

    if tid == 0 {
        let base = block_id * 3;
        unsafe {
            *out.get_unchecked_mut(base) = r_sum;
            *out.get_unchecked_mut(base + 1) = r_min;
            *out.get_unchecked_mut(base + 2) = r_max;
        }
    }
}

/// `block_reduce` for every supported `f32` op.
#[kernel]
pub fn test_block_reduce_f32(mut out: DisjointSlice<f32>) {
    static mut SMEM: SharedArray<f32, 3> = SharedArray::UNINIT;
    let block = this_thread_block();
    let tid = thread::threadIdx_x();
    let block_id = thread::blockIdx_x() as usize;

    let v = tid as f32;

    // SAFETY: every block thread uses this allocation; barriers separate reuse.

    let r_sum = unsafe { block_reduce::<f32, Sum, _>(&block, v, &raw mut SMEM) };
    block.sync();
    // SAFETY: every block thread uses this allocation; barriers separate reuse.
    let r_min = unsafe { block_reduce::<f32, Min, _>(&block, v, &raw mut SMEM) };
    block.sync();
    // SAFETY: every block thread uses this allocation; barriers separate reuse.
    let r_max = unsafe { block_reduce::<f32, Max, _>(&block, v, &raw mut SMEM) };

    if tid == 0 {
        let base = block_id * 3;
        unsafe {
            *out.get_unchecked_mut(base) = r_sum;
            *out.get_unchecked_mut(base + 1) = r_min;
            *out.get_unchecked_mut(base + 2) = r_max;
        }
    }
}

// ----- block_scan -----

/// `block_scan` (inclusive) for every supported `u32` op.
/// One row per thread; cells = `[Sum, Min, Max, BitAnd, BitOr, BitXor]`.
#[kernel]
pub fn test_block_scan_u32(mut out: DisjointSlice<u32>) {
    static mut SMEM: SharedArray<u32, 3> = SharedArray::UNINIT;
    let block = this_thread_block();
    let tid = thread::threadIdx_x();
    let block_id = thread::blockIdx_x();
    let global_tid = block_id * thread::blockDim_x() + tid;

    let v = tid;
    let v_one = 1u32 << (tid & 31);
    let v_inv = !v_one;

    // SAFETY: every block thread uses this allocation; barriers separate reuse.

    let r_sum = unsafe { block_scan::<u32, Sum, _>(&block, v, &raw mut SMEM) };
    block.sync();
    // SAFETY: every block thread uses this allocation; barriers separate reuse.
    let r_min = unsafe { block_scan::<u32, Min, _>(&block, v, &raw mut SMEM) };
    block.sync();
    // SAFETY: every block thread uses this allocation; barriers separate reuse.
    let r_max = unsafe { block_scan::<u32, Max, _>(&block, v, &raw mut SMEM) };
    block.sync();
    // SAFETY: every block thread uses this allocation; barriers separate reuse.
    let r_and = unsafe { block_scan::<u32, BitAnd, _>(&block, v_inv, &raw mut SMEM) };
    block.sync();
    // SAFETY: every block thread uses this allocation; barriers separate reuse.
    let r_or = unsafe { block_scan::<u32, BitOr, _>(&block, v_one, &raw mut SMEM) };
    block.sync();
    // SAFETY: every block thread uses this allocation; barriers separate reuse.
    let r_xor = unsafe { block_scan::<u32, BitXor, _>(&block, v_one, &raw mut SMEM) };

    let base = (global_tid as usize) * 6;
    unsafe {
        *out.get_unchecked_mut(base) = r_sum;
        *out.get_unchecked_mut(base + 1) = r_min;
        *out.get_unchecked_mut(base + 2) = r_max;
        *out.get_unchecked_mut(base + 3) = r_and;
        *out.get_unchecked_mut(base + 4) = r_or;
        *out.get_unchecked_mut(base + 5) = r_xor;
    }
}

/// `block_scan` (inclusive) for every supported `i32` op.
#[kernel]
pub fn test_block_scan_i32(mut out: DisjointSlice<i32>) {
    static mut SMEM: SharedArray<i32, 3> = SharedArray::UNINIT;
    let block = this_thread_block();
    let tid = thread::threadIdx_x();
    let block_id = thread::blockIdx_x();
    let global_tid = block_id * thread::blockDim_x() + tid;

    let v = (tid as i32) - 48;

    // SAFETY: every block thread uses this allocation; barriers separate reuse.

    let r_sum = unsafe { block_scan::<i32, Sum, _>(&block, v, &raw mut SMEM) };
    block.sync();
    // SAFETY: every block thread uses this allocation; barriers separate reuse.
    let r_min = unsafe { block_scan::<i32, Min, _>(&block, v, &raw mut SMEM) };
    block.sync();
    // SAFETY: every block thread uses this allocation; barriers separate reuse.
    let r_max = unsafe { block_scan::<i32, Max, _>(&block, v, &raw mut SMEM) };

    let base = (global_tid as usize) * 3;
    unsafe {
        *out.get_unchecked_mut(base) = r_sum;
        *out.get_unchecked_mut(base + 1) = r_min;
        *out.get_unchecked_mut(base + 2) = r_max;
    }
}

/// `block_scan` (inclusive) for every supported `f32` op.
#[kernel]
pub fn test_block_scan_f32(mut out: DisjointSlice<f32>) {
    static mut SMEM: SharedArray<f32, 3> = SharedArray::UNINIT;
    let block = this_thread_block();
    let tid = thread::threadIdx_x();
    let block_id = thread::blockIdx_x();
    let global_tid = block_id * thread::blockDim_x() + tid;

    let v = tid as f32;

    // SAFETY: every block thread uses this allocation; barriers separate reuse.

    let r_sum = unsafe { block_scan::<f32, Sum, _>(&block, v, &raw mut SMEM) };
    block.sync();
    // SAFETY: every block thread uses this allocation; barriers separate reuse.
    let r_min = unsafe { block_scan::<f32, Min, _>(&block, v, &raw mut SMEM) };
    block.sync();
    // SAFETY: every block thread uses this allocation; barriers separate reuse.
    let r_max = unsafe { block_scan::<f32, Max, _>(&block, v, &raw mut SMEM) };

    let base = (global_tid as usize) * 3;
    unsafe {
        *out.get_unchecked_mut(base) = r_sum;
        *out.get_unchecked_mut(base + 1) = r_min;
        *out.get_unchecked_mut(base + 2) = r_max;
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() {
    println!("=== Cooperative Groups Demo ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

    // Thread block clusters require sm_90 (Hopper) or later.
    let (major, minor) = ctx.compute_capability().expect("compute capability");
    if major < 9 {
        println!("skipping: thread block clusters require sm_90+ (device is sm_{major}{minor})");
        return;
    }

    let stream = ctx.default_stream();

    // Typed handle for the `#[cuda_module]` grid-sync kernels, loaded from the
    // artifact embedded in this binary rather than a loose `.ptx`, so there is
    // nothing to find on disk and `--materialize-cubin` works. Their
    // `#[cooperative_launch]` attribute makes the generated launch methods
    // submit cooperative launches, so no per-call flag is needed.
    //
    // SAFETY: `load` selects this package's own embedded bundle, which is
    // compiled from the module below and so carries exactly the entry points
    // and ABI the generated launch methods expect.
    let grid_sync_module =
        unsafe { grid_sync_kernels::load(&ctx) }.expect("Failed to load embedded grid-sync module");

    // The kernels outside the `#[cuda_module]` are launched through
    // `cuda_launch!`, which takes the raw module. It is the same load, so both
    // paths run the same image -- one `cuModuleLoad`, not two.
    let module = grid_sync_module.as_cuda_module().clone();

    const N: usize = 256;
    let cfg = LaunchConfig {
        block_dim: (32, 1, 1),
        grid_dim: ((N / 32) as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    // --- active_mask ---
    println!("--- active_mask() ---");
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: `slice_mut(out)` is the (ptr, len) pair for `test_active_mask`'s single
    // slice parameter (see the #[kernel] fn above); `out` is a live DeviceBuffer.
    unsafe {
        cuda_launch! {
            kernel: test_active_mask,
            stream: stream,
            module: module,
            config: cfg,
            args: [slice_mut(out)]
        }
    }
    .expect("test_active_mask launch failed");
    let host = out.to_host_vec(&stream).unwrap();
    let ok = host.iter().all(|&m| m == u32::MAX);
    println!(
        "  every lane saw 0xFFFFFFFF: {}",
        if ok { "yes" } else { "NO" }
    );

    // --- match_any_sync ---
    println!("\n--- match_any_sync(value = lane / 4) ---");
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: `slice_mut(out)` is the (ptr, len) pair for `test_match_any`'s single
    // slice parameter (see the #[kernel] fn above); `out` is a live DeviceBuffer.
    unsafe {
        cuda_launch! {
            kernel: test_match_any,
            stream: stream,
            module: module,
            config: cfg,
            args: [slice_mut(out)]
        }
    }
    .expect("test_match_any launch failed");
    let host = out.to_host_vec(&stream).unwrap();
    let ok = host.iter().enumerate().all(|(i, &m)| {
        let group = (i % 32) / 4;
        m == 0xF << (group * 4)
    });
    println!(
        "  every lane saw its 4-bucket mask: {}",
        if ok { "yes" } else { "NO" }
    );

    // --- match_all_sync ---
    println!("\n--- match_all_sync(constant) ---");
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: `slice_mut(out)` is the (ptr, len) pair for `test_match_all`'s single
    // slice parameter (see the #[kernel] fn above); `out` is a live DeviceBuffer.
    unsafe {
        cuda_launch! {
            kernel: test_match_all,
            stream: stream,
            module: module,
            config: cfg,
            args: [slice_mut(out)]
        }
    }
    .expect("test_match_all launch failed");
    let host = out.to_host_vec(&stream).unwrap();
    let ok = host.iter().all(|&m| m == u32::MAX);
    println!(
        "  every lane saw 0xFFFFFFFF: {}",
        if ok { "yes" } else { "NO" }
    );

    // --- grid::sync ---
    println!("\n--- grid::sync() (cooperative launch) ---");
    const BLOCKS: u32 = 32;
    let block_threads = 128u32;
    let coop_cfg = LaunchConfig {
        block_dim: (block_threads, 1, 1),
        grid_dim: (BLOCKS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut markers = DeviceBuffer::<u32>::zeroed(&stream, BLOCKS as usize).unwrap();
    let mut sums = DeviceBuffer::<u32>::zeroed(&stream, BLOCKS as usize).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { grid_sync_module.test_grid_sync(stream.as_ref(), coop_cfg, &mut markers, &mut sums) }
        .expect("test_grid_sync cooperative launch failed");
    let host = sums.to_host_vec(&stream).unwrap();
    let expected: u32 = (1..=BLOCKS).sum();
    let ok = host.iter().all(|&s| s == expected);
    println!(
        "  every block saw the full barrier-flushed marker sum {} : {}",
        expected,
        if ok { "yes" } else { "NO" }
    );
    if !ok {
        println!("  observed sums: {:?}", host);
        std::process::exit(1);
    }

    // --- cluster + cooperative (Hopper+) ---
    println!("\n--- cluster + cooperative grid::sync() (PreparedLaunch) ---");
    const CLUSTER_COOP_BLOCKS: u32 = 4; // divisible by #[cluster_launch(2, 1, 1)]
    const CLUSTER_COOP_THREADS: u32 = 128;
    match ctx.supports_cluster_launch() {
        Ok(false) => {
            println!(
                "  cluster+cooperative: not exercised (device lacks cluster launch / Hopper+)"
            );
        }
        Err(error) => {
            println!("  cluster support query failed: {error}");
            std::process::exit(1);
        }
        Ok(true) => {
            let cluster_config = LaunchConfig1D::new(CLUSTER_COOP_BLOCKS, CLUSTER_COOP_THREADS, 0);
            let prepared =
                match grid_sync_module.prepare_test_cluster_coop_grid_sync(cluster_config) {
                    Ok(prepared) => prepared,
                    Err(error) => {
                        println!("  prepare failed: {error}");
                        std::process::exit(1);
                    }
                };
            let mut markers =
                DeviceBuffer::<u32>::zeroed(&stream, CLUSTER_COOP_BLOCKS as usize).unwrap();
            let mut sums =
                DeviceBuffer::<u32>::zeroed(&stream, CLUSTER_COOP_BLOCKS as usize).unwrap();
            grid_sync_module
                .test_cluster_coop_grid_sync(stream.as_ref(), &prepared, &mut markers, &mut sums)
                .expect("test_cluster_coop_grid_sync launch failed");
            let host = sums.to_host_vec(&stream).unwrap();
            let expected: u32 = (1..=CLUSTER_COOP_BLOCKS).sum();
            let ok = host.iter().all(|&s| s == expected);
            println!(
                "  every block saw the full barrier-flushed marker sum {} : {}",
                expected,
                if ok { "yes" } else { "NO" }
            );
            if !ok {
                println!("  observed sums: {:?}", host);
                std::process::exit(1);
            }

            // Oversized cooperative clustered grid must fail preparation with a
            // clear residency error rather than hanging at launch.
            let too_large = LaunchConfig1D::new(1_048_576, CLUSTER_COOP_THREADS, 0);
            match grid_sync_module.prepare_test_cluster_coop_grid_sync(too_large) {
                Err(error) => {
                    let message = error.to_string();
                    let ok = message.contains("cooperative grid")
                        || message.contains("cluster")
                        || message.contains("resident");
                    println!(
                        "  oversized prepare rejected cleanly: {}",
                        if ok { "yes" } else { "NO" }
                    );
                    if !ok {
                        println!("  unexpected prepare error: {message}");
                        std::process::exit(1);
                    }
                }
                Ok(_) => {
                    println!("  oversized prepare unexpectedly succeeded");
                    std::process::exit(1);
                }
            }
        }
    }

    // =========================================================================
    // TYPED COOPERATIVE-GROUPS API
    // =========================================================================

    println!("\n=== Typed cooperative_groups API ===");

    // --- WarpTile<32>::ballot ---
    println!("\n--- WarpTile<32>::ballot(lane_id & 1) ---");
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: `slice_mut(out)` is the (ptr, len) pair for `test_typed_warp32_ballot`'s single
    // slice parameter (see the #[kernel] fn above); `out` is a live DeviceBuffer.
    unsafe {
        cuda_launch! {
            kernel: test_typed_warp32_ballot,
            stream: stream,
            module: module,
            config: cfg,
            args: [slice_mut(out)]
        }
    }
    .expect("test_typed_warp32_ballot launch failed");
    let host = out.to_host_vec(&stream).unwrap();
    let ok = host.iter().all(|&m| m == 0xAAAAAAAA);
    println!(
        "  every lane saw 0xAAAAAAAA: {}",
        if ok { "yes" } else { "NO" }
    );
    if !ok {
        println!("  observed: {:?}", &host[..32]);
        std::process::exit(1);
    }

    // --- WarpTile<16>::ballot ---
    println!("\n--- WarpTile<16>::ballot(lane_id & 1) ---");
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: `slice_mut(out)` is the (ptr, len) pair for `test_typed_warp16_ballot`'s single
    // slice parameter (see the #[kernel] fn above); `out` is a live DeviceBuffer.
    unsafe {
        cuda_launch! {
            kernel: test_typed_warp16_ballot,
            stream: stream,
            module: module,
            config: cfg,
            args: [slice_mut(out)]
        }
    }
    .expect("test_typed_warp16_ballot launch failed");
    let host = out.to_host_vec(&stream).unwrap();
    let ok = host.iter().all(|&m| m == 0xAAAA);
    println!(
        "  every lane in every 16-lane tile saw 0xAAAA: {}",
        if ok { "yes" } else { "NO" }
    );
    if !ok {
        println!("  observed: {:?}", &host[..32]);
        std::process::exit(1);
    }

    // --- WarpTile<16>::shfl ---
    println!("\n--- WarpTile<16>::shfl(lane_id, 0) ---");
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: `slice_mut(out)` is the (ptr, len) pair for `test_typed_warp16_shfl`'s single
    // slice parameter (see the #[kernel] fn above); `out` is a live DeviceBuffer.
    unsafe {
        cuda_launch! {
            kernel: test_typed_warp16_shfl,
            stream: stream,
            module: module,
            config: cfg,
            args: [slice_mut(out)]
        }
    }
    .expect("test_typed_warp16_shfl launch failed");
    let host = out.to_host_vec(&stream).unwrap();
    let ok = host.iter().enumerate().all(|(i, &v)| {
        let lane = (i as u32) % 32;
        let expected = if lane < 16 { 0 } else { 16 };
        v == expected
    });
    println!(
        "  tile 0 broadcasts 0, tile 1 broadcasts 16: {}",
        if ok { "yes" } else { "NO" }
    );
    if !ok {
        println!("  observed (first 32): {:?}", &host[..32]);
        std::process::exit(1);
    }

    // --- this_grid().sync() ---
    println!("\n--- this_grid().sync() (cooperative launch) ---");
    let mut markers = DeviceBuffer::<u32>::zeroed(&stream, BLOCKS as usize).unwrap();
    let mut sums = DeviceBuffer::<u32>::zeroed(&stream, BLOCKS as usize).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        grid_sync_module.test_typed_grid_sync(stream.as_ref(), coop_cfg, &mut markers, &mut sums)
    }
    .expect("test_typed_grid_sync cooperative launch failed");
    let host = sums.to_host_vec(&stream).unwrap();
    let expected: u32 = (1..=BLOCKS).sum();
    let ok = host.iter().all(|&s| s == expected);
    println!(
        "  typed grid sync produces the same sum {} : {}",
        expected,
        if ok { "yes" } else { "NO" }
    );
    if !ok {
        println!("  observed sums: {:?}", host);
        std::process::exit(1);
    }

    // --- this_grid().thread_rank() ---
    println!("\n--- this_grid().thread_rank() ---");
    let total = (BLOCKS * block_threads) as usize;
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, total).unwrap();
    // SAFETY: `slice_mut(out)` is the (ptr, len) pair for `test_typed_grid_rank`'s single
    // slice parameter (see the #[kernel] fn above); `out` is a live DeviceBuffer.
    unsafe {
        cuda_launch! {
            kernel: test_typed_grid_rank,
            stream: stream,
            module: module,
            config: coop_cfg,
            args: [slice_mut(out)]
        }
    }
    .expect("test_typed_grid_rank launch failed");
    let host = out.to_host_vec(&stream).unwrap();
    let ok = host.iter().enumerate().all(|(i, r)| (*r as usize) == i);
    println!(
        "  thread_rank() forms the identity permutation 0..{}: {}",
        total,
        if ok { "yes" } else { "NO" }
    );
    if !ok {
        let mismatches: Vec<_> = host
            .iter()
            .enumerate()
            .filter(|(i, r)| (**r as usize) != *i)
            .take(8)
            .collect();
        println!("  first mismatches (idx, observed): {:?}", mismatches);
        std::process::exit(1);
    }

    // =========================================================================
    // Reductions and scans
    // =========================================================================

    // Warp tests reuse `cfg` (block_dim 32, 8 blocks → 8 warps total).

    let warp_count = N / 32;

    // --- warp_reduce<u32> ---
    println!("\n--- warp_reduce::<u32, _>, all 6 ops ---");
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, warp_count * 6).unwrap();
    // SAFETY: `slice_mut(out)` is the (ptr, len) pair for `test_warp_reduce_u32`'s single
    // slice parameter (see the #[kernel] fn above); `out` is a live DeviceBuffer.
    unsafe {
        cuda_launch! {
            kernel: test_warp_reduce_u32, stream: stream, module: module,
            config: cfg, args: [slice_mut(out)]
        }
    }
    .expect("test_warp_reduce_u32 launch failed");
    let host = out.to_host_vec(&stream).unwrap();
    let expected = [
        496u32,     // Sum: 0+1+...+31
        0,          // Min
        31,         // Max
        0,          // BitAnd: !(all bits) = 0
        0xFFFFFFFF, // BitOr:  all bits set
        0xFFFFFFFF, // BitXor: 32 distinct bits XORed
    ];
    let mut ok = true;
    for w in 0..warp_count {
        for (j, &want) in expected.iter().enumerate() {
            if host[w * 6 + j] != want {
                println!(
                    "  warp {} op {}: expected 0x{:08x}, got 0x{:08x}",
                    w,
                    j,
                    want,
                    host[w * 6 + j]
                );
                ok = false;
            }
        }
    }
    println!(
        "  all 8 warps × [Sum,Min,Max,BitAnd,BitOr,BitXor] match: {}",
        if ok { "yes" } else { "NO" }
    );
    if !ok {
        std::process::exit(1);
    }

    // --- warp_reduce<i32> ---
    println!("\n--- warp_reduce::<i32, _>, [Sum, Min, Max] ---");
    let mut out = DeviceBuffer::<i32>::zeroed(&stream, warp_count * 3).unwrap();
    // SAFETY: `slice_mut(out)` is the (ptr, len) pair for `test_warp_reduce_i32`'s single
    // slice parameter (see the #[kernel] fn above); `out` is a live DeviceBuffer.
    unsafe {
        cuda_launch! {
            kernel: test_warp_reduce_i32, stream: stream, module: module,
            config: cfg, args: [slice_mut(out)]
        }
    }
    .expect("test_warp_reduce_i32 launch failed");
    let host = out.to_host_vec(&stream).unwrap();
    let expected_i = [-16i32, -16, 15];
    let mut ok = true;
    for w in 0..warp_count {
        for (j, &want) in expected_i.iter().enumerate() {
            if host[w * 3 + j] != want {
                println!(
                    "  warp {} op {}: expected {}, got {}",
                    w,
                    j,
                    want,
                    host[w * 3 + j]
                );
                ok = false;
            }
        }
    }
    println!(
        "  all 8 warps × [Sum,Min,Max] match: {}",
        if ok { "yes" } else { "NO" }
    );
    if !ok {
        std::process::exit(1);
    }

    // --- warp_reduce<f32> ---
    println!("\n--- warp_reduce::<f32, _>, [Sum, Min, Max] ---");
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, warp_count * 3).unwrap();
    // SAFETY: `slice_mut(out)` is the (ptr, len) pair for `test_warp_reduce_f32`'s single
    // slice parameter (see the #[kernel] fn above); `out` is a live DeviceBuffer.
    unsafe {
        cuda_launch! {
            kernel: test_warp_reduce_f32, stream: stream, module: module,
            config: cfg, args: [slice_mut(out)]
        }
    }
    .expect("test_warp_reduce_f32 launch failed");
    let host = out.to_host_vec(&stream).unwrap();
    let expected_f = [496.0f32, 0.0, 31.0];
    let mut ok = true;
    for w in 0..warp_count {
        for (j, &want) in expected_f.iter().enumerate() {
            if (host[w * 3 + j] - want).abs() > 1e-4 {
                println!(
                    "  warp {} op {}: expected {}, got {}",
                    w,
                    j,
                    want,
                    host[w * 3 + j]
                );
                ok = false;
            }
        }
    }
    println!(
        "  all 8 warps × [Sum,Min,Max] match: {}",
        if ok { "yes" } else { "NO" }
    );
    if !ok {
        std::process::exit(1);
    }

    // --- warp_scan<u32> ---
    println!("\n--- warp_scan::<u32, _> (inclusive), all 6 ops ---");
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N * 6).unwrap();
    // SAFETY: `slice_mut(out)` is the (ptr, len) pair for `test_warp_scan_u32`'s single
    // slice parameter (see the #[kernel] fn above); `out` is a live DeviceBuffer.
    unsafe {
        cuda_launch! {
            kernel: test_warp_scan_u32, stream: stream, module: module,
            config: cfg, args: [slice_mut(out)]
        }
    }
    .expect("test_warp_scan_u32 launch failed");
    let host = out.to_host_vec(&stream).unwrap();
    // For each lane k in 0..32, expected per-op inclusive scan.
    let scan_u32 = |k: u32| -> [u32; 6] {
        let mask: u64 = (1u64 << (k + 1)) - 1;
        let mask_u32 = mask as u32;
        [
            (k * (k + 1)) / 2, // Sum
            0,                 // Min: lane 0 contributes 0
            k,                 // Max
            !mask_u32,         // BitAnd of !(1<<i) for i in 0..=k
            mask_u32,          // BitOr of (1<<i) for i in 0..=k
            mask_u32,          // BitXor of (1<<i) — distinct bits, == OR
        ]
    };
    let mut ok = true;
    for tid in 0..N {
        let lane = (tid as u32) & 31;
        let want = scan_u32(lane);
        for (j, &w) in want.iter().enumerate() {
            if host[tid * 6 + j] != w {
                println!(
                    "  tid {} (lane {}) op {}: expected 0x{:08x}, got 0x{:08x}",
                    tid,
                    lane,
                    j,
                    w,
                    host[tid * 6 + j]
                );
                ok = false;
            }
        }
    }
    println!(
        "  all {} threads × [Sum,Min,Max,BitAnd,BitOr,BitXor] match: {}",
        N,
        if ok { "yes" } else { "NO" }
    );
    if !ok {
        std::process::exit(1);
    }

    // --- warp_scan<i32> ---
    println!("\n--- warp_scan::<i32, _> (inclusive), [Sum, Min, Max] ---");
    let mut out = DeviceBuffer::<i32>::zeroed(&stream, N * 3).unwrap();
    // SAFETY: `slice_mut(out)` is the (ptr, len) pair for `test_warp_scan_i32`'s single
    // slice parameter (see the #[kernel] fn above); `out` is a live DeviceBuffer.
    unsafe {
        cuda_launch! {
            kernel: test_warp_scan_i32, stream: stream, module: module,
            config: cfg, args: [slice_mut(out)]
        }
    }
    .expect("test_warp_scan_i32 launch failed");
    let host = out.to_host_vec(&stream).unwrap();
    let scan_i32 = |k: i32| -> [i32; 3] {
        // value at lane i is (i - 16); inclusive scan to lane k.
        let sum_0_to_k = k * (k + 1) / 2;
        let scan_sum = sum_0_to_k - 16 * (k + 1);
        [scan_sum, -16, k - 16]
    };
    let mut ok = true;
    for tid in 0..N {
        let lane = ((tid as u32) & 31) as i32;
        let want = scan_i32(lane);
        for (j, &w) in want.iter().enumerate() {
            if host[tid * 3 + j] != w {
                println!(
                    "  tid {} (lane {}) op {}: expected {}, got {}",
                    tid,
                    lane,
                    j,
                    w,
                    host[tid * 3 + j]
                );
                ok = false;
            }
        }
    }
    println!(
        "  all {} threads × [Sum,Min,Max] match: {}",
        N,
        if ok { "yes" } else { "NO" }
    );
    if !ok {
        std::process::exit(1);
    }

    // --- warp_scan<f32> ---
    println!("\n--- warp_scan::<f32, _> (inclusive), [Sum, Min, Max] ---");
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, N * 3).unwrap();
    // SAFETY: `slice_mut(out)` is the (ptr, len) pair for `test_warp_scan_f32`'s single
    // slice parameter (see the #[kernel] fn above); `out` is a live DeviceBuffer.
    unsafe {
        cuda_launch! {
            kernel: test_warp_scan_f32, stream: stream, module: module,
            config: cfg, args: [slice_mut(out)]
        }
    }
    .expect("test_warp_scan_f32 launch failed");
    let host = out.to_host_vec(&stream).unwrap();
    let scan_f32 = |k: u32| -> [f32; 3] {
        let sum = (k * (k + 1) / 2) as f32;
        [sum, 0.0, k as f32]
    };
    let mut ok = true;
    for tid in 0..N {
        let lane = (tid as u32) & 31;
        let want = scan_f32(lane);
        for (j, &w) in want.iter().enumerate() {
            if (host[tid * 3 + j] - w).abs() > 1e-3 {
                println!(
                    "  tid {} (lane {}) op {}: expected {}, got {}",
                    tid,
                    lane,
                    j,
                    w,
                    host[tid * 3 + j]
                );
                ok = false;
            }
        }
    }
    println!(
        "  all {} threads × [Sum,Min,Max] match: {}",
        N,
        if ok { "yes" } else { "NO" }
    );
    if !ok {
        std::process::exit(1);
    }

    // --- block tests: block_dim = 96 (NUM_WARPS = 3) ---
    const BLOCK_SIZE: usize = 96;
    const NUM_BLOCKS: usize = 4;
    const TOTAL_THREADS: usize = BLOCK_SIZE * NUM_BLOCKS;
    let block_cfg = LaunchConfig {
        block_dim: (BLOCK_SIZE as u32, 1, 1),
        grid_dim: (NUM_BLOCKS as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    // Helper: per-tid running scan over BLOCK_SIZE values.
    fn block_scan_u32_expected() -> Vec<[u32; 6]> {
        let mut acc = [0u32, u32::MAX, 0u32, u32::MAX, 0u32, 0u32];
        let mut out = Vec::with_capacity(BLOCK_SIZE);
        for tid in 0..BLOCK_SIZE {
            let v = tid as u32;
            let v_one = 1u32 << ((tid as u32) & 31);
            let v_inv = !v_one;
            acc[0] = acc[0].wrapping_add(v);
            acc[1] = acc[1].min(v);
            acc[2] = acc[2].max(v);
            acc[3] &= v_inv;
            acc[4] |= v_one;
            acc[5] ^= v_one;
            out.push(acc);
        }
        out
    }
    fn block_scan_i32_expected() -> Vec<[i32; 3]> {
        let mut acc = [0i32, i32::MAX, i32::MIN];
        let mut out = Vec::with_capacity(BLOCK_SIZE);
        for tid in 0..BLOCK_SIZE {
            let v = (tid as i32) - 48;
            acc[0] = acc[0].wrapping_add(v);
            acc[1] = acc[1].min(v);
            acc[2] = acc[2].max(v);
            out.push(acc);
        }
        out
    }
    fn block_scan_f32_expected() -> Vec<[f32; 3]> {
        let mut acc = [0.0f32, f32::INFINITY, f32::NEG_INFINITY];
        let mut out = Vec::with_capacity(BLOCK_SIZE);
        for tid in 0..BLOCK_SIZE {
            let v = tid as f32;
            acc[0] += v;
            acc[1] = acc[1].min(v);
            acc[2] = acc[2].max(v);
            out.push(acc);
        }
        out
    }

    let scan_u32_expected = block_scan_u32_expected();
    let scan_i32_expected = block_scan_i32_expected();
    let scan_f32_expected = block_scan_f32_expected();
    // Block reduce expected = scan at last tid.
    let red_u32_expected = scan_u32_expected[BLOCK_SIZE - 1];
    let red_i32_expected = scan_i32_expected[BLOCK_SIZE - 1];
    let red_f32_expected = scan_f32_expected[BLOCK_SIZE - 1];

    // --- block_reduce<u32> ---
    println!(
        "\n--- block_reduce::<u32, _> (block_dim={}, {} warps), all 6 ops ---",
        BLOCK_SIZE,
        BLOCK_SIZE / 32
    );
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, NUM_BLOCKS * 6).unwrap();
    // SAFETY: `slice_mut(out)` is the (ptr, len) pair for `test_block_reduce_u32`'s single
    // slice parameter (see the #[kernel] fn above); `out` is a live DeviceBuffer.
    unsafe {
        cuda_launch! {
            kernel: test_block_reduce_u32, stream: stream, module: module,
            config: block_cfg, args: [slice_mut(out)]
        }
    }
    .expect("test_block_reduce_u32 launch failed");
    let host = out.to_host_vec(&stream).unwrap();
    let mut ok = true;
    for b in 0..NUM_BLOCKS {
        for (j, &want) in red_u32_expected.iter().enumerate() {
            if host[b * 6 + j] != want {
                println!(
                    "  block {} op {}: expected 0x{:08x}, got 0x{:08x}",
                    b,
                    j,
                    want,
                    host[b * 6 + j]
                );
                ok = false;
            }
        }
    }
    println!(
        "  all {} blocks × [Sum,Min,Max,BitAnd,BitOr,BitXor] match: {}",
        NUM_BLOCKS,
        if ok { "yes" } else { "NO" }
    );
    if !ok {
        std::process::exit(1);
    }

    // --- block_reduce<i32> ---
    println!(
        "\n--- block_reduce::<i32, _> (block_dim={}), [Sum, Min, Max] ---",
        BLOCK_SIZE
    );
    let mut out = DeviceBuffer::<i32>::zeroed(&stream, NUM_BLOCKS * 3).unwrap();
    // SAFETY: `slice_mut(out)` is the (ptr, len) pair for `test_block_reduce_i32`'s single
    // slice parameter (see the #[kernel] fn above); `out` is a live DeviceBuffer.
    unsafe {
        cuda_launch! {
            kernel: test_block_reduce_i32, stream: stream, module: module,
            config: block_cfg, args: [slice_mut(out)]
        }
    }
    .expect("test_block_reduce_i32 launch failed");
    let host = out.to_host_vec(&stream).unwrap();
    let mut ok = true;
    for b in 0..NUM_BLOCKS {
        for (j, &want) in red_i32_expected.iter().enumerate() {
            if host[b * 3 + j] != want {
                println!(
                    "  block {} op {}: expected {}, got {}",
                    b,
                    j,
                    want,
                    host[b * 3 + j]
                );
                ok = false;
            }
        }
    }
    println!(
        "  all {} blocks × [Sum,Min,Max] match: {}",
        NUM_BLOCKS,
        if ok { "yes" } else { "NO" }
    );
    if !ok {
        std::process::exit(1);
    }

    // --- block_reduce<f32> ---
    println!(
        "\n--- block_reduce::<f32, _> (block_dim={}), [Sum, Min, Max] ---",
        BLOCK_SIZE
    );
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, NUM_BLOCKS * 3).unwrap();
    // SAFETY: `slice_mut(out)` is the (ptr, len) pair for `test_block_reduce_f32`'s single
    // slice parameter (see the #[kernel] fn above); `out` is a live DeviceBuffer.
    unsafe {
        cuda_launch! {
            kernel: test_block_reduce_f32, stream: stream, module: module,
            config: block_cfg, args: [slice_mut(out)]
        }
    }
    .expect("test_block_reduce_f32 launch failed");
    let host = out.to_host_vec(&stream).unwrap();
    let mut ok = true;
    for b in 0..NUM_BLOCKS {
        for (j, &want) in red_f32_expected.iter().enumerate() {
            if (host[b * 3 + j] - want).abs() > 1e-2 {
                println!(
                    "  block {} op {}: expected {}, got {}",
                    b,
                    j,
                    want,
                    host[b * 3 + j]
                );
                ok = false;
            }
        }
    }
    println!(
        "  all {} blocks × [Sum,Min,Max] match: {}",
        NUM_BLOCKS,
        if ok { "yes" } else { "NO" }
    );
    if !ok {
        std::process::exit(1);
    }

    // --- block_scan<u32> ---
    println!(
        "\n--- block_scan::<u32, _> (block_dim={}, inclusive), all 6 ops ---",
        BLOCK_SIZE
    );
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, TOTAL_THREADS * 6).unwrap();
    // SAFETY: `slice_mut(out)` is the (ptr, len) pair for `test_block_scan_u32`'s single
    // slice parameter (see the #[kernel] fn above); `out` is a live DeviceBuffer.
    unsafe {
        cuda_launch! {
            kernel: test_block_scan_u32, stream: stream, module: module,
            config: block_cfg, args: [slice_mut(out)]
        }
    }
    .expect("test_block_scan_u32 launch failed");
    let host = out.to_host_vec(&stream).unwrap();
    let mut ok = true;
    for b in 0..NUM_BLOCKS {
        for (tid, want) in scan_u32_expected.iter().enumerate() {
            let global_tid = b * BLOCK_SIZE + tid;
            for (j, &w) in want.iter().enumerate() {
                if host[global_tid * 6 + j] != w {
                    println!(
                        "  block {} tid {} op {}: expected 0x{:08x}, got 0x{:08x}",
                        b,
                        tid,
                        j,
                        w,
                        host[global_tid * 6 + j]
                    );
                    ok = false;
                }
            }
        }
    }
    println!(
        "  all {} threads × [Sum,Min,Max,BitAnd,BitOr,BitXor] match: {}",
        TOTAL_THREADS,
        if ok { "yes" } else { "NO" }
    );
    if !ok {
        std::process::exit(1);
    }

    // --- block_scan<i32> ---
    println!(
        "\n--- block_scan::<i32, _> (block_dim={}), [Sum, Min, Max] ---",
        BLOCK_SIZE
    );
    let mut out = DeviceBuffer::<i32>::zeroed(&stream, TOTAL_THREADS * 3).unwrap();
    // SAFETY: `slice_mut(out)` is the (ptr, len) pair for `test_block_scan_i32`'s single
    // slice parameter (see the #[kernel] fn above); `out` is a live DeviceBuffer.
    unsafe {
        cuda_launch! {
            kernel: test_block_scan_i32, stream: stream, module: module,
            config: block_cfg, args: [slice_mut(out)]
        }
    }
    .expect("test_block_scan_i32 launch failed");
    let host = out.to_host_vec(&stream).unwrap();
    let mut ok = true;
    for b in 0..NUM_BLOCKS {
        for (tid, want) in scan_i32_expected.iter().enumerate() {
            let global_tid = b * BLOCK_SIZE + tid;
            for (j, &w) in want.iter().enumerate() {
                if host[global_tid * 3 + j] != w {
                    println!(
                        "  block {} tid {} op {}: expected {}, got {}",
                        b,
                        tid,
                        j,
                        w,
                        host[global_tid * 3 + j]
                    );
                    ok = false;
                }
            }
        }
    }
    println!(
        "  all {} threads × [Sum,Min,Max] match: {}",
        TOTAL_THREADS,
        if ok { "yes" } else { "NO" }
    );
    if !ok {
        std::process::exit(1);
    }

    // --- block_scan<f32> ---
    println!(
        "\n--- block_scan::<f32, _> (block_dim={}), [Sum, Min, Max] ---",
        BLOCK_SIZE
    );
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, TOTAL_THREADS * 3).unwrap();
    // SAFETY: `slice_mut(out)` is the (ptr, len) pair for `test_block_scan_f32`'s single
    // slice parameter (see the #[kernel] fn above); `out` is a live DeviceBuffer.
    unsafe {
        cuda_launch! {
            kernel: test_block_scan_f32, stream: stream, module: module,
            config: block_cfg, args: [slice_mut(out)]
        }
    }
    .expect("test_block_scan_f32 launch failed");
    let host = out.to_host_vec(&stream).unwrap();
    let mut ok = true;
    for b in 0..NUM_BLOCKS {
        for (tid, want) in scan_f32_expected.iter().enumerate() {
            let global_tid = b * BLOCK_SIZE + tid;
            for (j, &w) in want.iter().enumerate() {
                if (host[global_tid * 3 + j] - w).abs() > 1e-2 {
                    println!(
                        "  block {} tid {} op {}: expected {}, got {}",
                        b,
                        tid,
                        j,
                        w,
                        host[global_tid * 3 + j]
                    );
                    ok = false;
                }
            }
        }
    }
    println!(
        "  all {} threads × [Sum,Min,Max] match: {}",
        TOTAL_THREADS,
        if ok { "yes" } else { "NO" }
    );
    if !ok {
        std::process::exit(1);
    }

    println!("\n=== All cooperative-groups checks PASSED ===");
}
