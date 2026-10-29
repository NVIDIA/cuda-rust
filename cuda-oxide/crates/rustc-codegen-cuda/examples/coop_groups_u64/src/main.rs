/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! 64-bit reductions, scans and shuffles through `cooperative_groups`.
//!
//! `warp_reduce`, `warp_scan`, `block_reduce` and `block_scan` accept `u64`
//! with every combiner in `ops`, and `i64` with `Sum`, `Min` and `Max`. PTX
//! has no 64-bit shuffle, so each exchange moves the two 32-bit halves. A
//! collective that loses the high word, drops a carry between the halves,
//! orders values by their low words, or reads across a tile boundary gives
//! a different answer from the host reference here.
//!
//! The kernels cover:
//!
//! - `tile_1` .. `tile_32`: every reduction and inclusive scan on
//!   `WarpTile<N>` for each tile size, plus the four `u64` group shuffles,
//!   including offsets that leave the tile and must return the caller's own
//!   value. Each block of 128 threads holds several warps and, below 32
//!   lanes, several tiles per warp.
//! - `block_collectives`: every block reduction and inclusive scan, launched
//!   with block sizes from 1 to 1024 threads, most of them leaving a partial
//!   final warp.
//! - `coalesced_shuffles`: the `u64` shuffles on sparse `CoalescedThreads`
//!   groups, whose ranks and lanes differ.
//! - `defaults_8` and `defaults_32`: the same shuffles on a group that
//!   implements only the required `WarpCollective` methods, so they run the
//!   trait's default bodies.
//! - `warp_total`: a caller that sums `u64` values over a full warp through
//!   the public API alone.
//!
//! The inputs follow fixed patterns per warp or per block: full-width random
//! values, values whose low words alone would pick the wrong `Min` and
//! `Max`, values just below `u64::MAX` whose `Sum` wraps, the `u64` and
//! `i64` extremes, and low words whose `Sum` carries into the high word.
//!
//! Build and run with:
//!   cargo oxide run coop_groups_u64

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::cooperative_groups::{
    ThreadGroup, WarpCollective, WarpTile, block_reduce, block_scan, coalesced_threads,
    ops::{BitAnd, BitOr, BitXor, Max, Min, Sum},
    this_thread_block, warp_reduce, warp_scan,
};
use cuda_device::{DisjointSlice, SharedArray, device, kernel, thread, warp};
use cuda_host::cuda_module;

/// Reductions and scans per thread: six `u64` combiners and three `i64`
/// combiners, each reduced and scanned.
const COLLECTIVE_OUTPUTS: usize = 18;
/// Group shuffles per thread: one index, three butterfly, three down and
/// three up shuffles.
const SHUFFLE_OUTPUTS: usize = 10;
/// Results per thread from a tile kernel: the collectives, then the shuffles.
const TILE_OUTPUTS: usize = COLLECTIVE_OUTPUTS + SHUFFLE_OUTPUTS;
/// Results per member thread from `coalesced_shuffles`.
const COALESCED_OUTPUTS: usize = 7;

/// Pre-filled into every output so a thread that never writes is visible.
const SENTINEL: u64 = 0xA5A5_A5A5_A5A5_A5A5;

#[cuda_module]
mod kernels {
    use super::*;

    /// Stores `value` at column `col` of the row that starts at `row`.
    #[device]
    fn put(out: &mut DisjointSlice<u64>, row: usize, col: usize, value: u64) {
        // SAFETY: the host sizes `out` to one full row per launched thread,
        // and each thread passes the start of its own row
        unsafe { *out.get_unchecked_mut(row + col) = value };
    }

    /// Every reduction, scan and `u64` shuffle a `WarpTile<N>` offers, in the
    /// column order the host reference uses.
    #[device]
    fn tile_row<const N: u32>(input: &[u64], mut out: DisjointSlice<u64>) {
        let gid = thread::index_1d().get();
        let tile = this_thread_block().tiled_partition::<N>();
        let v = input[gid];
        let s = v as i64;
        let row = gid * TILE_OUTPUTS;

        put(&mut out, row, 0, warp_reduce::<u64, Sum, N>(&tile, v));
        put(&mut out, row, 1, warp_reduce::<u64, Min, N>(&tile, v));
        put(&mut out, row, 2, warp_reduce::<u64, Max, N>(&tile, v));
        put(&mut out, row, 3, warp_reduce::<u64, BitAnd, N>(&tile, v));
        put(&mut out, row, 4, warp_reduce::<u64, BitOr, N>(&tile, v));
        put(&mut out, row, 5, warp_reduce::<u64, BitXor, N>(&tile, v));
        put(&mut out, row, 6, warp_scan::<u64, Sum, N>(&tile, v));
        put(&mut out, row, 7, warp_scan::<u64, Min, N>(&tile, v));
        put(&mut out, row, 8, warp_scan::<u64, Max, N>(&tile, v));
        put(&mut out, row, 9, warp_scan::<u64, BitAnd, N>(&tile, v));
        put(&mut out, row, 10, warp_scan::<u64, BitOr, N>(&tile, v));
        put(&mut out, row, 11, warp_scan::<u64, BitXor, N>(&tile, v));
        put(
            &mut out,
            row,
            12,
            warp_reduce::<i64, Sum, N>(&tile, s) as u64,
        );
        put(
            &mut out,
            row,
            13,
            warp_reduce::<i64, Min, N>(&tile, s) as u64,
        );
        put(
            &mut out,
            row,
            14,
            warp_reduce::<i64, Max, N>(&tile, s) as u64,
        );
        put(&mut out, row, 15, warp_scan::<i64, Sum, N>(&tile, s) as u64);
        put(&mut out, row, 16, warp_scan::<i64, Min, N>(&tile, s) as u64);
        put(&mut out, row, 17, warp_scan::<i64, Max, N>(&tile, s) as u64);

        shuffle_cols::<_, N>(&tile, v, &mut out, row + COLLECTIVE_OUTPUTS);
    }

    /// The four `u64` group shuffles of an `N`-lane group, written to the
    /// `SHUFFLE_OUTPUTS` columns that start at `start`.
    #[device]
    fn shuffle_cols<G: WarpCollective, const N: u32>(
        group: &G,
        v: u64,
        out: &mut DisjointSlice<u64>,
        start: usize,
    ) {
        let rank = group.thread_rank();

        // the third offset of each kind reaches past a sub-warp tile, where
        // the group must hand back the caller's own value
        let wide_xor = if N < 32 { N } else { 16 };
        let wide_delta = if N < 32 { N } else { 17 };
        put(out, start, 0, group.shfl_u64(v, rank * 7 + 3));
        put(out, start, 1, group.shfl_xor_u64(v, 1));
        put(out, start, 2, group.shfl_xor_u64(v, N - 1));
        put(out, start, 3, group.shfl_xor_u64(v, wide_xor));
        put(out, start, 4, group.shfl_down_u64(v, 1));
        put(out, start, 5, group.shfl_down_u64(v, N - 1));
        put(out, start, 6, group.shfl_down_u64(v, wide_delta));
        put(out, start, 7, group.shfl_up_u64(v, 1));
        put(out, start, 8, group.shfl_up_u64(v, N - 1));
        put(out, start, 9, group.shfl_up_u64(v, wide_delta));
    }

    /// A group that implements only the required `WarpCollective` methods,
    /// as one defined outside `cuda-device` would, so its `u64` shuffles run
    /// the trait's default bodies over two 32-bit shuffles.
    #[derive(Clone, Copy)]
    pub struct HalvesOnly<const N: u32>(WarpTile<N>);

    impl<const N: u32> ThreadGroup for HalvesOnly<N> {
        fn size(&self) -> u32 {
            self.0.size()
        }

        fn thread_rank(&self) -> u32 {
            self.0.thread_rank()
        }

        fn sync(&self) {
            self.0.sync()
        }
    }

    impl<const N: u32> WarpCollective for HalvesOnly<N> {
        fn ballot(&self, predicate: bool) -> u32 {
            self.0.ballot(predicate)
        }

        fn all(&self, predicate: bool) -> bool {
            self.0.all(predicate)
        }

        fn any(&self, predicate: bool) -> bool {
            self.0.any(predicate)
        }

        fn shfl(&self, var: u32, src_rank: u32) -> u32 {
            self.0.shfl(var, src_rank)
        }

        fn shfl_xor(&self, var: u32, lane_mask: u32) -> u32 {
            self.0.shfl_xor(var, lane_mask)
        }

        fn shfl_down(&self, var: u32, delta: u32) -> u32 {
            self.0.shfl_down(var, delta)
        }

        fn shfl_up(&self, var: u32, delta: u32) -> u32 {
            self.0.shfl_up(var, delta)
        }

        fn shfl_f32(&self, var: f32, src_rank: u32) -> f32 {
            self.0.shfl_f32(var, src_rank)
        }

        fn shfl_xor_f32(&self, var: f32, lane_mask: u32) -> f32 {
            self.0.shfl_xor_f32(var, lane_mask)
        }

        fn shfl_down_f32(&self, var: f32, delta: u32) -> f32 {
            self.0.shfl_down_f32(var, delta)
        }

        fn shfl_up_f32(&self, var: f32, delta: u32) -> f32 {
            self.0.shfl_up_f32(var, delta)
        }

        fn match_any(&self, value: u32) -> u32 {
            self.0.match_any(value)
        }

        fn match_any_i64(&self, value: u64) -> u32 {
            self.0.match_any_i64(value)
        }

        fn match_all(&self, value: u32) -> u32 {
            self.0.match_all(value)
        }

        fn match_all_i64(&self, value: u64) -> u32 {
            self.0.match_all_i64(value)
        }
    }

    /// The `u64` shuffles through the trait's default bodies.
    #[device]
    fn default_row<const N: u32>(input: &[u64], mut out: DisjointSlice<u64>) {
        let gid = thread::index_1d().get();
        let group = HalvesOnly(this_thread_block().tiled_partition::<N>());
        shuffle_cols::<_, N>(&group, input[gid], &mut out, gid * SHUFFLE_OUTPUTS);
    }

    #[kernel]
    pub fn defaults_8(input: &[u64], out: DisjointSlice<u64>) {
        default_row::<8>(input, out);
    }

    #[kernel]
    pub fn defaults_32(input: &[u64], out: DisjointSlice<u64>) {
        default_row::<32>(input, out);
    }

    #[kernel]
    pub fn tile_1(input: &[u64], out: DisjointSlice<u64>) {
        tile_row::<1>(input, out);
    }

    #[kernel]
    pub fn tile_2(input: &[u64], out: DisjointSlice<u64>) {
        tile_row::<2>(input, out);
    }

    #[kernel]
    pub fn tile_4(input: &[u64], out: DisjointSlice<u64>) {
        tile_row::<4>(input, out);
    }

    #[kernel]
    pub fn tile_8(input: &[u64], out: DisjointSlice<u64>) {
        tile_row::<8>(input, out);
    }

    #[kernel]
    pub fn tile_16(input: &[u64], out: DisjointSlice<u64>) {
        tile_row::<16>(input, out);
    }

    #[kernel]
    pub fn tile_32(input: &[u64], out: DisjointSlice<u64>) {
        tile_row::<32>(input, out);
    }

    /// Every block reduction and inclusive scan, in the same column order as
    /// the first `COLLECTIVE_OUTPUTS` columns of a tile row.
    #[kernel]
    pub fn block_collectives(input: &[u64], mut out: DisjointSlice<u64>) {
        static mut SMEM: SharedArray<u64, 32> = SharedArray::UNINIT;
        static mut SMEM_I64: SharedArray<i64, 32> = SharedArray::UNINIT;

        let block = this_thread_block();
        let gid = thread::index_1d().get();
        let v = input[gid];
        let s = v as i64;
        let row = gid * COLLECTIVE_OUTPUTS;

        // each call leaves its scratch in an arbitrary state, so a barrier
        // separates it from the next call on the same allocation
        macro_rules! block_op {
            ($f:ident, $t:ty, $op:ty, $x:expr, $smem:ident) => {{
                // SAFETY: every thread of the block makes this call with the
                // same scratch, and the barrier below orders its reuse
                let r = unsafe { $f::<$t, $op, 32>(&block, $x, &raw mut $smem) };
                block.sync();
                r
            }};
        }

        put(&mut out, row, 0, block_op!(block_reduce, u64, Sum, v, SMEM));
        put(&mut out, row, 1, block_op!(block_reduce, u64, Min, v, SMEM));
        put(&mut out, row, 2, block_op!(block_reduce, u64, Max, v, SMEM));
        put(
            &mut out,
            row,
            3,
            block_op!(block_reduce, u64, BitAnd, v, SMEM),
        );
        put(
            &mut out,
            row,
            4,
            block_op!(block_reduce, u64, BitOr, v, SMEM),
        );
        put(
            &mut out,
            row,
            5,
            block_op!(block_reduce, u64, BitXor, v, SMEM),
        );
        put(&mut out, row, 6, block_op!(block_scan, u64, Sum, v, SMEM));
        put(&mut out, row, 7, block_op!(block_scan, u64, Min, v, SMEM));
        put(&mut out, row, 8, block_op!(block_scan, u64, Max, v, SMEM));
        put(
            &mut out,
            row,
            9,
            block_op!(block_scan, u64, BitAnd, v, SMEM),
        );
        put(
            &mut out,
            row,
            10,
            block_op!(block_scan, u64, BitOr, v, SMEM),
        );
        put(
            &mut out,
            row,
            11,
            block_op!(block_scan, u64, BitXor, v, SMEM),
        );
        put(
            &mut out,
            row,
            12,
            block_op!(block_reduce, i64, Sum, s, SMEM_I64) as u64,
        );
        put(
            &mut out,
            row,
            13,
            block_op!(block_reduce, i64, Min, s, SMEM_I64) as u64,
        );
        put(
            &mut out,
            row,
            14,
            block_op!(block_reduce, i64, Max, s, SMEM_I64) as u64,
        );
        put(
            &mut out,
            row,
            15,
            block_op!(block_scan, i64, Sum, s, SMEM_I64) as u64,
        );
        put(
            &mut out,
            row,
            16,
            block_op!(block_scan, i64, Min, s, SMEM_I64) as u64,
        );
        put(
            &mut out,
            row,
            17,
            block_op!(block_scan, i64, Max, s, SMEM_I64) as u64,
        );
    }

    /// The `u64` shuffles on a sparse coalesced group. Lanes named in
    /// `masks[warp]` enter the branch together and form the group; the
    /// others write nothing.
    #[kernel]
    pub fn coalesced_shuffles(
        input: &[u64],
        masks: &[u32],
        mut out: DisjointSlice<u64>,
        mut seen: DisjointSlice<u32>,
    ) {
        let gid = thread::index_1d().get();
        let members = masks[gid / 32];

        if members & (1u32 << warp::lane_id()) != 0 {
            let group = coalesced_threads();
            let v = input[gid];
            let row = gid * COALESCED_OUTPUTS;

            put(
                &mut out,
                row,
                0,
                group.shfl_u64(v, group.thread_rank() * 7 + 3),
            );
            put(&mut out, row, 1, group.shfl_xor_u64(v, 1));
            put(&mut out, row, 2, group.shfl_xor_u64(v, 6));
            put(&mut out, row, 3, group.shfl_down_u64(v, 1));
            put(&mut out, row, 4, group.shfl_down_u64(v, 5));
            put(&mut out, row, 5, group.shfl_up_u64(v, 1));
            put(&mut out, row, 6, group.shfl_up_u64(v, 5));

            // SAFETY: the host sizes `seen` to one entry per launched thread
            unsafe { *seen.get_unchecked_mut(gid) = group.raw_mask() };
        }
    }

    /// Sums one `u64` per lane over a full warp, written against the public
    /// API alone.
    #[device]
    pub fn total(tile: &WarpTile<32>, value: u64) -> u64 {
        warp_reduce::<u64, Sum, 32>(tile, value)
    }

    #[kernel]
    pub fn warp_total(input: &[u64], mut out: DisjointSlice<u64>) {
        let gid = thread::index_1d().get();
        let tile = this_thread_block().tiled_partition::<32>();
        put(&mut out, gid, 0, total(&tile, input[gid]));
    }
}

// =============================================================================
// HOST REFERENCE
// =============================================================================

#[derive(Clone, Copy)]
enum Op {
    Sum,
    Min,
    Max,
    And,
    Or,
    Xor,
}

const U64_OPS: [Op; 6] = [Op::Sum, Op::Min, Op::Max, Op::And, Op::Or, Op::Xor];
const I64_OPS: [Op; 3] = [Op::Sum, Op::Min, Op::Max];

const COLLECTIVE_LABELS: [&str; COLLECTIVE_OUTPUTS] = [
    "reduce u64 Sum",
    "reduce u64 Min",
    "reduce u64 Max",
    "reduce u64 BitAnd",
    "reduce u64 BitOr",
    "reduce u64 BitXor",
    "scan u64 Sum",
    "scan u64 Min",
    "scan u64 Max",
    "scan u64 BitAnd",
    "scan u64 BitOr",
    "scan u64 BitXor",
    "reduce i64 Sum",
    "reduce i64 Min",
    "reduce i64 Max",
    "scan i64 Sum",
    "scan i64 Min",
    "scan i64 Max",
];

const SHUFFLE_LABELS: [&str; SHUFFLE_OUTPUTS] = [
    "shfl_u64 rank*7+3",
    "shfl_xor_u64 1",
    "shfl_xor_u64 N-1",
    "shfl_xor_u64 wide",
    "shfl_down_u64 1",
    "shfl_down_u64 N-1",
    "shfl_down_u64 wide",
    "shfl_up_u64 1",
    "shfl_up_u64 N-1",
    "shfl_up_u64 wide",
];

const COALESCED_LABELS: [&str; COALESCED_OUTPUTS] = [
    "shfl_u64 rank*7+3",
    "shfl_xor_u64 1",
    "shfl_xor_u64 6",
    "shfl_down_u64 1",
    "shfl_down_u64 5",
    "shfl_up_u64 1",
    "shfl_up_u64 5",
];

fn combine_u64(op: Op, a: u64, b: u64) -> u64 {
    match op {
        Op::Sum => a.wrapping_add(b),
        Op::Min => a.min(b),
        Op::Max => a.max(b),
        Op::And => a & b,
        Op::Or => a | b,
        Op::Xor => a ^ b,
    }
}

fn combine_i64(op: Op, a: i64, b: i64) -> i64 {
    match op {
        Op::Sum => a.wrapping_add(b),
        Op::Min => a.min(b),
        Op::Max => a.max(b),
        Op::And | Op::Or | Op::Xor => unreachable!("i64 has no bitwise combiners"),
    }
}

/// The reductions and inclusive scans one member of `group` at `rank`
/// should see, in kernel column order.
fn collective_row(group: &[u64], rank: usize) -> Vec<u64> {
    let fold_u64 = |op, values: &[u64]| {
        values[1..]
            .iter()
            .fold(values[0], |acc, &x| combine_u64(op, acc, x))
    };
    let fold_i64 = |op, values: &[u64]| {
        values[1..]
            .iter()
            .fold(values[0] as i64, |acc, &x| combine_i64(op, acc, x as i64)) as u64
    };
    let prefix = &group[..=rank];

    let mut row = Vec::with_capacity(COLLECTIVE_OUTPUTS);
    row.extend(U64_OPS.iter().map(|&op| fold_u64(op, group)));
    row.extend(U64_OPS.iter().map(|&op| fold_u64(op, prefix)));
    row.extend(I64_OPS.iter().map(|&op| fold_i64(op, group)));
    row.extend(I64_OPS.iter().map(|&op| fold_i64(op, prefix)));
    row
}

/// Expected rows for a tile kernel with `n`-lane tiles. Launches use whole
/// warps, so consecutive chunks of `n` threads are exactly the tiles.
fn tile_expected(input: &[u64], n: usize) -> Vec<u64> {
    let wide_xor = if n < 32 { n } else { 16 };
    let wide_delta = if n < 32 { n } else { 17 };

    let mut expected = Vec::with_capacity(input.len() * TILE_OUTPUTS);
    for tile in input.chunks(n) {
        for rank in 0..n {
            let own = tile[rank];
            let xor = |mask: usize| tile.get(rank ^ mask).copied().unwrap_or(own);
            let down = |delta: usize| tile.get(rank + delta).copied().unwrap_or(own);
            let up = |delta: usize| rank.checked_sub(delta).map_or(own, |src| tile[src]);

            expected.extend(collective_row(tile, rank));
            expected.push(tile[(rank * 7 + 3) & (n - 1)]);
            expected.extend([xor(1), xor(n - 1), xor(wide_xor)]);
            expected.extend([down(1), down(n - 1), down(wide_delta)]);
            expected.extend([up(1), up(n - 1), up(wide_delta)]);
        }
    }
    expected
}

/// Expected rows for `block_collectives` launched with `block`-thread blocks.
fn block_expected(input: &[u64], block: usize) -> Vec<u64> {
    let mut expected = Vec::with_capacity(input.len() * COLLECTIVE_OUTPUTS);
    for group in input.chunks(block) {
        for rank in 0..group.len() {
            expected.extend(collective_row(group, rank));
        }
    }
    expected
}

/// Expected rows for `coalesced_shuffles`: a member at `lane` reads from the
/// lane its offset names when that lane is in the group, and otherwise
/// keeps its own value. Non-members keep the sentinel.
fn coalesced_expected(input: &[u64], masks: &[u32]) -> Vec<u64> {
    let mut expected = vec![SENTINEL; input.len() * COALESCED_OUTPUTS];
    for (w, &members) in masks.iter().enumerate() {
        let values = &input[w * 32..(w + 1) * 32];
        let member = |lane: usize| lane < 32 && members & (1u32 << lane) != 0;

        for lane in (0..32).filter(|&lane| member(lane)) {
            let own = values[lane];
            let from = |src: Option<usize>| match src {
                Some(src) if member(src) => values[src],
                _ => own,
            };
            let rank = (members & ((1u32 << lane) - 1)).count_ones() as usize;
            let rank_lane = (0..32).filter(|&l| member(l)).nth(rank * 7 + 3);
            let row = [
                from(rank_lane),
                from(Some(lane ^ 1)),
                from(Some(lane ^ 6)),
                from(Some(lane + 1)),
                from(Some(lane + 5)),
                from(lane.checked_sub(1)),
                from(lane.checked_sub(5)),
            ];
            let start = (w * 32 + lane) * COALESCED_OUTPUTS;
            expected[start..start + COALESCED_OUTPUTS].copy_from_slice(&row);
        }
    }
    expected
}

// =============================================================================
// INPUTS
// =============================================================================

/// Number of value patterns; warp `w` (or block `b`) uses pattern `w % PATTERNS`.
const PATTERNS: usize = 8;

fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Value `i` of `pattern`. Each pattern targets one way a 64-bit collective
/// built from 32-bit shuffles can go wrong.
fn pattern_value(pattern: usize, i: u64) -> u64 {
    let r = splitmix64(i ^ ((pattern as u64) << 40));
    match pattern {
        // full-width random values
        0 => r,
        // the high word decides Min and Max, and the low words alone would
        // pick the opposite value; the low words also carry when summed
        1 if i.is_multiple_of(2) => (1 << 32) | (r & 0xF),
        1 => 0xFFFF_FFF0 | (r & 0xF),
        // just below u64::MAX, so Sum wraps
        2 => u64::MAX - (r & 0xFFFF),
        // the u64 and i64 extremes
        3 => [0, 1, u64::MAX, u64::MAX - 1, 1 << 63, (1 << 63) - 1][(r % 6) as usize],
        // all u64::MAX, the Min and BitAnd identity
        4 => u64::MAX,
        // only the high words differ
        5 => (r & 0xFFFF_FFFF_0000_0000) | 0xDEAD_BEEF,
        // negative as i64, with low words that carry into the high word
        6 => 0xFFFF_FFFF_8000_0000 | (r & 0x7FFF_FFFF),
        // one bit each, spread over both words
        _ => 1 << (r % 64),
    }
}

/// `len` values where every consecutive run of `run` values shares a
/// pattern, starting from pattern `first`.
fn patterned_input(len: usize, run: usize, first: usize) -> Vec<u64> {
    (0..len)
        .map(|i| pattern_value((first + i / run) % PATTERNS, i as u64))
        .collect()
}

// =============================================================================
// HOST CODE
// =============================================================================

const TILE_BLOCK: u32 = 128;
const TILE_BLOCKS: u32 = 8;
const TILE_THREADS: usize = (TILE_BLOCK * TILE_BLOCKS) as usize;

/// Block sizes for `block_collectives`, most with a partial final warp.
const BLOCK_SIZES: [u32; 14] = [
    1, 7, 31, 32, 33, 45, 64, 100, 255, 256, 257, 640, 1000, 1024,
];
const BLOCKS_PER_SIZE: u32 = 3;

/// Coalesced group per warp of `coalesced_shuffles`.
const COALESCED_MASKS: [u32; 8] = [
    u32::MAX,
    0x8010_0089,
    0x5555_5555,
    0xAAAA_AAAA,
    0x0000_0001,
    0xFFFF_0000,
    0x7FFF_FFFE,
    0xF0F0_0F0F,
];

/// Compares `got` with `expected` row by row and prints the first few
/// mismatches. Returns how many values differ.
fn compare(case: &str, got: &[u64], expected: &[u64], labels: &[&str]) -> usize {
    let row_len = labels.len();
    let mut mismatches = 0;
    for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
        if g != e {
            if mismatches < 5 {
                println!(
                    "  {case}: thread {} {}: got {g:#018x}, expected {e:#018x}",
                    i / row_len,
                    labels[i % row_len],
                );
            }
            mismatches += 1;
        }
    }
    if got.len() != expected.len() {
        println!(
            "  {case}: {} results, expected {}",
            got.len(),
            expected.len()
        );
        mismatches += 1;
    }
    if mismatches == 0 {
        println!("  {case}: {} results match", got.len());
    } else {
        println!("  {case}: {mismatches} mismatches");
    }
    mismatches
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let (major, minor) = ctx.compute_capability()?;
    println!("=== cooperative_groups u64 / i64 collectives on sm_{major}{minor} ===");

    let module = kernels::load(&ctx)?;
    let mut failures = 0;

    // ----- warp tiles -----
    println!("\n--- WarpTile<N> reductions, scans and u64 shuffles ---");
    let tile_cfg = LaunchConfig {
        block_dim: (TILE_BLOCK, 1, 1),
        grid_dim: (TILE_BLOCKS, 1, 1),
        shared_mem_bytes: 0,
    };
    let host_input = patterned_input(TILE_THREADS, 32, 0);
    let input = DeviceBuffer::from_host(&stream, &host_input)?;
    let tile_labels: Vec<&str> = COLLECTIVE_LABELS
        .iter()
        .chain(SHUFFLE_LABELS.iter())
        .copied()
        .collect();

    for n in [1usize, 2, 4, 8, 16, 32] {
        let mut out =
            DeviceBuffer::from_host(&stream, &vec![SENTINEL; TILE_THREADS * TILE_OUTPUTS])?;
        let s = stream.as_ref();
        // SAFETY: each launch covers TILE_THREADS threads in blocks of 128,
        // a multiple of every tile size, and `input` and `out` hold one value
        // and one row per thread
        unsafe {
            match n {
                1 => module.tile_1(s, tile_cfg, &input, &mut out),
                2 => module.tile_2(s, tile_cfg, &input, &mut out),
                4 => module.tile_4(s, tile_cfg, &input, &mut out),
                8 => module.tile_8(s, tile_cfg, &input, &mut out),
                16 => module.tile_16(s, tile_cfg, &input, &mut out),
                _ => module.tile_32(s, tile_cfg, &input, &mut out),
            }
        }?;
        stream.synchronize()?;
        failures += compare(
            &format!("WarpTile<{n}>"),
            &out.to_host_vec(&stream)?,
            &tile_expected(&host_input, n),
            &tile_labels,
        );
    }

    // ----- blocks -----
    println!("\n--- block_reduce / block_scan ---");
    for (launch, block) in BLOCK_SIZES.into_iter().enumerate() {
        let threads = (block * BLOCKS_PER_SIZE) as usize;
        let host_input =
            patterned_input(threads, block as usize, launch * BLOCKS_PER_SIZE as usize);
        let input = DeviceBuffer::from_host(&stream, &host_input)?;
        let mut out =
            DeviceBuffer::from_host(&stream, &vec![SENTINEL; threads * COLLECTIVE_OUTPUTS])?;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (BLOCKS_PER_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: blocks hold at most 1024 threads, the 32 warp totals the
        // scratch holds, and `input` and `out` hold one value and one row per
        // thread
        unsafe { module.block_collectives(stream.as_ref(), cfg, &input, &mut out) }?;
        stream.synchronize()?;
        failures += compare(
            &format!("{BLOCKS_PER_SIZE} blocks of {block}"),
            &out.to_host_vec(&stream)?,
            &block_expected(&host_input, block as usize),
            &COLLECTIVE_LABELS,
        );
    }

    // ----- default trait bodies -----
    println!("\n--- WarpCollective default u64 shuffles ---");
    let host_input = patterned_input(TILE_THREADS, 32, 0);
    let input = DeviceBuffer::from_host(&stream, &host_input)?;
    for n in [8usize, 32] {
        let mut out =
            DeviceBuffer::from_host(&stream, &vec![SENTINEL; TILE_THREADS * SHUFFLE_OUTPUTS])?;
        let s = stream.as_ref();
        // SAFETY: as for the tile kernels, with one shuffle row per thread
        unsafe {
            match n {
                8 => module.defaults_8(s, tile_cfg, &input, &mut out),
                _ => module.defaults_32(s, tile_cfg, &input, &mut out),
            }
        }?;
        stream.synchronize()?;
        // the shuffle columns of the tile reference, which used the same calls
        let expected: Vec<u64> = tile_expected(&host_input, n)
            .chunks(TILE_OUTPUTS)
            .flat_map(|row| &row[COLLECTIVE_OUTPUTS..])
            .copied()
            .collect();
        failures += compare(
            &format!("HalvesOnly<{n}>"),
            &out.to_host_vec(&stream)?,
            &expected,
            &SHUFFLE_LABELS,
        );
    }

    // ----- coalesced groups -----
    println!("\n--- CoalescedThreads u64 shuffles ---");
    let threads = COALESCED_MASKS.len() * 32;
    let host_input = patterned_input(threads, threads, 0);
    let input = DeviceBuffer::from_host(&stream, &host_input)?;
    let masks = DeviceBuffer::from_host(&stream, &COALESCED_MASKS)?;
    let mut out = DeviceBuffer::from_host(&stream, &vec![SENTINEL; threads * COALESCED_OUTPUTS])?;
    let mut seen = DeviceBuffer::from_host(&stream, &vec![0u32; threads])?;
    let cfg = LaunchConfig {
        block_dim: (threads as u32, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: one block of whole warps, with one mask per warp and one value,
    // row and mask slot per thread
    unsafe {
        module.coalesced_shuffles(stream.as_ref(), cfg, &input, &masks, &mut out, &mut seen)
    }?;
    stream.synchronize()?;
    let seen = seen.to_host_vec(&stream)?;
    for (gid, &mask) in seen.iter().enumerate() {
        let want = COALESCED_MASKS[gid / 32];
        let observed = if want & (1 << (gid % 32)) != 0 {
            want
        } else {
            0
        };
        if mask != observed {
            println!("  thread {gid} saw group mask {mask:#010x}, expected {observed:#010x}");
            failures += 1;
        }
    }
    failures += compare(
        "sparse groups",
        &out.to_host_vec(&stream)?,
        &coalesced_expected(&host_input, &COALESCED_MASKS),
        &COALESCED_LABELS,
    );

    // ----- public-API caller -----
    println!("\n--- total(&WarpTile<32>, u64) ---");
    let host_input = patterned_input(TILE_THREADS, 32, 0);
    let input = DeviceBuffer::from_host(&stream, &host_input)?;
    let mut out = DeviceBuffer::from_host(&stream, &vec![SENTINEL; TILE_THREADS])?;
    // SAFETY: TILE_THREADS threads in whole warps, one value and one result
    // per thread
    unsafe { module.warp_total(stream.as_ref(), tile_cfg, &input, &mut out) }?;
    stream.synchronize()?;
    let expected: Vec<u64> = host_input
        .chunks(32)
        .flat_map(|warp| {
            let sum = warp.iter().fold(0u64, |acc, &x| acc.wrapping_add(x));
            [sum; 32]
        })
        .collect();
    failures += compare(
        "warp sums",
        &out.to_host_vec(&stream)?,
        &expected,
        &["total"],
    );

    if failures != 0 {
        return Err(format!("{failures} results disagree with the host reference").into());
    }
    println!("\nSUCCESS: u64 and i64 collectives match the host reference");
    Ok(())
}
