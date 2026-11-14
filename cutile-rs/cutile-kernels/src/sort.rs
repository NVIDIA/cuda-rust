/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tile-level stable sort by small key, and the rank-by-key building block
//! under it ([#1658]).
//!
//! All kernels work on one `[N]` tile of `u32` keys in `[0, K)`, where `N`
//! and `K` are powers of two (Tile IR partition views require power-of-two
//! tile dimensions). `rank_by_key_probe` exposes the building block: each
//! element's rank among earlier elements with the same key, plus the per-key
//! counts. `sort_by_key` composes it into a stable counting sort —
//! exclusive-scan the counts into per-key base offsets, add each element's
//! within-key rank, and scatter keys and payload to the resulting positions.
//!
//! Two formulations of the ranking core exist, matching the alternatives
//! measured in [#1658]:
//!
//! - The one-hot form (used by `rank_by_key_probe` and `sort_by_key`) builds
//!   `onehot[i][k] = (keys[i] == k)` so the per-key work becomes dense
//!   `[N, K]` tile ops: one inclusive prefix sum down `N` yields within-key
//!   ranks, reductions yield the counts and per-element selections.
//! - `rank_by_key_loop` runs one `[N]` prefix sum per key value, the
//!   formulation the issue's radix sort shipped with.
//!
//! On an A10G (sm_86) the one-hot form measured 11-15% faster across
//! `N`x`K` of 1024x8, 1024x16, and 512x32 (`tests/rank_sort_bench_gpu.rs`),
//! so it backs `rank_by_key_probe` and `sort_by_key`. Its `[N, K]` working
//! set grows with both parameters, though, and Tile IR assembly fails for
//! `N * K = 32768` i32 elements on sm_86; the loop form carries only `[N]`
//! tiles and keeps compiling as `K` grows.
//!
//! Both are scan-bound; the issue's intended fix is a compiler primitive on
//! `match.any` and warp ballots that kernel code cannot express. These
//! kernels pin down the API and correctness oracles for it.
//!
//! Single tile block: launch with a `(1, 1, 1)` grid.
//!
//! [#1658]: https://github.com/NVIDIA/cuda-rust/issues/1658

#[cutile::module]
pub mod sort_kernels {
    use cutile::core::*;

    /// Within-key rank and per-key counts for a `[N]` tile of keys in `[0, K)`.
    ///
    /// `rank[i] = #{ j < i : digit[j] == digit[i] }` and
    /// `counts[k] = #{ i : digit[i] == k }`. `N` and `K` must be powers of two.
    ///
    /// # Safety
    /// Every element of `digit` must be in `[0, K)`.
    #[cutile::entry]
    pub unsafe fn rank_by_key_probe<const N: i32, const K: i32>(
        digit: &Tensor<u32, { [N] }>,
        rank: &mut Tensor<i32, { [N] }>,
        counts: &mut Tensor<i32, { [K] }>,
    ) {
        let digit_u32: Tile<u32, { [N] }> = digit.load_tile(const_shape![N], [0i32]);
        let d: Tile<i32, { [N] }> = bitcast(digit_u32);

        let d_nk: Tile<i32, { [N, K] }> =
            d.reshape(const_shape![N, 1]).broadcast(const_shape![N, K]);
        let key_ids: Tile<i32, { [K] }> = iota(const_shape![K]);
        let key_nk: Tile<i32, { [N, K] }> = key_ids
            .reshape(const_shape![1, K])
            .broadcast(const_shape![N, K]);
        let is_key: Tile<bool, { [N, K] }> = eq_tile(d_nk, key_nk);

        let ones: Tile<i32, { [N, K] }> = constant(1i32, const_shape![N, K]);
        let zeros: Tile<i32, { [N, K] }> = constant(0i32, const_shape![N, K]);
        let onehot: Tile<i32, { [N, K] }> = select(is_key, ones, zeros);

        // The inclusive prefix sum down N counts same-key elements up to and
        // including each position; subtracting one gives the earlier-only rank.
        let incl: Tile<i32, { [N, K] }> = scan_sum(onehot, 0i32, reverse::Forward, 0i32);
        let rank_nk: Tile<i32, { [N, K] }> = select(is_key, incl - ones, zeros);
        let rank_n: Tile<i32, { [N] }> = reduce_sum(rank_nk, 1i32);
        rank.store(rank_n);

        let counts_k: Tile<i32, { [K] }> = reduce_sum(onehot, 0i32);
        counts.store(counts_k);
    }

    /// `rank_by_key_probe` computed with one `[N]` prefix sum per key value
    /// instead of a single `[N, K]` one-hot scan — the formulation #1658's
    /// radix sort shipped with, kept so the two can be benchmarked against
    /// each other. Same contract as [`rank_by_key_probe`].
    ///
    /// # Safety
    /// Every element of `digit` must be in `[0, K)`.
    #[cutile::entry]
    pub unsafe fn rank_by_key_loop<const N: i32, const K: i32>(
        digit: &Tensor<u32, { [N] }>,
        rank: &mut Tensor<i32, { [N] }>,
        counts: &mut Tensor<i32, { [K] }>,
    ) {
        let digit_u32: Tile<u32, { [N] }> = digit.load_tile(const_shape![N], [0i32]);
        let d: Tile<i32, { [N] }> = bitcast(digit_u32);

        let ones_n: Tile<i32, { [N] }> = constant(1i32, const_shape![N]);
        let zeros_n: Tile<i32, { [N] }> = constant(0i32, const_shape![N]);
        let key_ids: Tile<i32, { [K] }> = iota(const_shape![K]);
        let mut rank_t: Tile<i32, { [N] }> = constant(0i32, const_shape![N]);
        let mut counts_t: Tile<i32, { [K] }> = constant(0i32, const_shape![K]);
        for k in 0i32..K {
            let k_n: Tile<i32, { [N] }> = k.broadcast(const_shape![N]);
            let is_k: Tile<bool, { [N] }> = eq_tile(d, k_n);
            let onehot_n: Tile<i32, { [N] }> = select(is_k, ones_n, zeros_n);
            let incl: Tile<i32, { [N] }> = scan_sum(onehot_n, 0i32, reverse::Forward, 0i32);
            rank_t = select(is_k, incl - ones_n, rank_t);
            // Reduce results drop rank in the IR, so go through a scalar
            // rather than broadcasting the [1] tile directly.
            let count_1: Tile<i32, { [1] }> = reduce_sum(onehot_n, 0i32);
            let count_s: i32 = tile_to_scalar(count_1.reshape(const_shape![]));
            let count_k: Tile<i32, { [K] }> = count_s.broadcast(const_shape![K]);
            let k_k: Tile<i32, { [K] }> = k.broadcast(const_shape![K]);
            let is_slot: Tile<bool, { [K] }> = eq_tile(key_ids, k_k);
            counts_t = select(is_slot, count_k, counts_t);
        }
        rank.store(rank_t);
        counts.store(counts_t);
    }

    /// Stable sort of a `[N]` tile by `u32` keys in `[0, K)`, carrying an
    /// `i32` payload tile along. `N` and `K` must be powers of two.
    ///
    /// # Safety
    /// Every element of `keys` must be in `[0, K)`; out-of-range keys scatter
    /// through out-of-bounds pointers.
    #[cutile::entry]
    pub unsafe fn sort_by_key<const N: i32, const K: i32>(
        keys: &Tensor<u32, { [N] }>,
        payload: &Tensor<i32, { [N] }>,
        sorted_keys: &mut Tensor<u32, { [N] }>,
        sorted_payload: &mut Tensor<i32, { [N] }>,
    ) {
        let keys_u32: Tile<u32, { [N] }> = keys.load_tile(const_shape![N], [0i32]);
        let payload_t: Tile<i32, { [N] }> = payload.load_tile(const_shape![N], [0i32]);
        let d: Tile<i32, { [N] }> = bitcast(keys_u32);

        let d_nk: Tile<i32, { [N, K] }> =
            d.reshape(const_shape![N, 1]).broadcast(const_shape![N, K]);
        let key_ids: Tile<i32, { [K] }> = iota(const_shape![K]);
        let key_nk: Tile<i32, { [N, K] }> = key_ids
            .reshape(const_shape![1, K])
            .broadcast(const_shape![N, K]);
        let is_key: Tile<bool, { [N, K] }> = eq_tile(d_nk, key_nk);

        let ones: Tile<i32, { [N, K] }> = constant(1i32, const_shape![N, K]);
        let zeros: Tile<i32, { [N, K] }> = constant(0i32, const_shape![N, K]);
        let onehot: Tile<i32, { [N, K] }> = select(is_key, ones, zeros);

        let incl: Tile<i32, { [N, K] }> = scan_sum(onehot, 0i32, reverse::Forward, 0i32);
        let rank_nk: Tile<i32, { [N, K] }> = select(is_key, incl - ones, zeros);
        let rank_n: Tile<i32, { [N] }> = reduce_sum(rank_nk, 1i32);

        // Exclusive scan of the per-key counts gives each key's base offset.
        let counts_k: Tile<i32, { [K] }> = reduce_sum(onehot, 0i32);
        let ends: Tile<i32, { [K] }> = scan_sum(counts_k, 0i32, reverse::Forward, 0i32);
        let base_k: Tile<i32, { [K] }> = ends - counts_k;

        // Select base[keys[i]] for each element through the one-hot mask.
        let base_nk: Tile<i32, { [N, K] }> = base_k
            .reshape(const_shape![1, K])
            .broadcast(const_shape![N, K]);
        let base_sel: Tile<i32, { [N, K] }> = select(is_key, base_nk, zeros);
        let base_n: Tile<i32, { [N] }> = reduce_sum(base_sel, 1i32);

        // pos is a permutation of 0..N: bases partition the output range and
        // within-key ranks are unique, so the scatters below cannot race.
        let pos: Tile<i32, { [N] }> = base_n + rank_n;

        let keys_base: PointerTile<*mut u32, { [] }> = get_tensor_base(sorted_keys);
        let keys_base_1: PointerTile<*mut u32, { [1] }> = keys_base.reshape(const_shape![1]);
        let keys_base_n: PointerTile<*mut u32, { [N] }> = keys_base_1.broadcast(const_shape![N]);
        let keys_ptrs: PointerTile<*mut u32, { [N] }> = keys_base_n.offset_tile(pos);
        let keys_token: Token = get_tensor_token(sorted_keys);
        // SAFETY: every key is in [0, K), so pos stays a permutation of 0..N
        // and each lane writes one distinct in-bounds element of sorted_keys.
        let _: Token = unsafe {
            store_ptr_tko(
                keys_ptrs,
                keys_u32,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                Some(keys_token),
                Latency::<1>,
            )
        };

        let payload_base: PointerTile<*mut i32, { [] }> = get_tensor_base(sorted_payload);
        let payload_base_1: PointerTile<*mut i32, { [1] }> = payload_base.reshape(const_shape![1]);
        let payload_base_n: PointerTile<*mut i32, { [N] }> =
            payload_base_1.broadcast(const_shape![N]);
        let payload_ptrs: PointerTile<*mut i32, { [N] }> = payload_base_n.offset_tile(pos);
        let payload_token: Token = get_tensor_token(sorted_payload);
        // SAFETY: same permutation argument as the key scatter above.
        let _: Token = unsafe {
            store_ptr_tko(
                payload_ptrs,
                payload_t,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                Some(payload_token),
                Latency::<1>,
            )
        };
    }
}

pub use sort_kernels::{rank_by_key_loop, rank_by_key_probe, sort_by_key};
