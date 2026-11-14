/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Outside-in tests for the tile-level sort / rank-by-key primitive (#1658).
//!
//! These are written test-first and WILL NOT COMPILE until the primitives
//! exist. They drive this intended API in `cutile_kernels::sort`:
//!
//! ```ignore
//! // Stable sort of a [N] tile by a small key in [0, K), carrying an i32 payload.
//! #[cutile::entry]
//! unsafe fn sort_by_key<const N: i32, const K: i32>(
//!     keys: &Tensor<u32, { [N] }>,
//!     payload: &Tensor<i32, { [N] }>,
//!     sorted_keys: &mut Tensor<u32, { [N] }>,
//!     sorted_payload: &mut Tensor<i32, { [N] }>,
//! );
//!
//! // Probe exposing the rank_by_key building block: each element's rank among
//! // earlier elements with the same key, plus per-key counts.
//! #[cutile::entry]
//! unsafe fn rank_by_key_probe<const N: i32, const K: i32>(
//!     digit: &Tensor<u32, { [N] }>,
//!     rank: &mut Tensor<i32, { [N] }>,
//!     counts: &mut Tensor<i32, { [K] }>,
//! );
//! ```
//!
//! Layering (see #1658): `sort_by_key` is the production composition
//! (exclusive-scan the per-key counts into base offsets, add each element's
//! within-key rank, scatter); `rank_by_key_probe` exposes the hard building
//! block directly. Each level is checked against its own INDEPENDENT host
//! oracle so a bug in one cannot be masked by the other. Device-only; run via
//! `scripts/run_gpu_tests.sh`.

use std::sync::Arc;

use cutile::cuda_core::Stream;
use cutile::prelude::*;
use cutile_kernels::sort::{rank_by_key_loop, rank_by_key_probe, sort_by_key};

fn on_test_thread(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(32_000_000)
        .spawn(f)
        .unwrap()
        .join()
        .unwrap();
}

fn gen(tags: &[&str]) -> Vec<String> {
    tags.iter().map(|s| s.to_string()).collect()
}

fn up_u32(values: &[u32], shape: &[usize], stream: &Arc<Stream>) -> Tensor<u32> {
    api::copy_host_vec_to_device(&Arc::new(values.to_vec()))
        .reshape(shape)
        .sync_on(stream)
        .unwrap()
}
fn up_i32(values: &[i32], shape: &[usize], stream: &Arc<Stream>) -> Tensor<i32> {
    api::copy_host_vec_to_device(&Arc::new(values.to_vec()))
        .reshape(shape)
        .sync_on(stream)
        .unwrap()
}
fn read_u32(t: Tensor<u32>, stream: &Arc<Stream>) -> Vec<u32> {
    t.to_host_vec().sync_on(stream).unwrap()
}
fn read_i32(t: Tensor<i32>, stream: &Arc<Stream>) -> Vec<i32> {
    t.to_host_vec().sync_on(stream).unwrap()
}

// ---- Independent host oracles ----

// Stable sort by key, carrying payload. Rust's sort_by_key is stable, so equal
// keys keep their original order; this is deliberately NOT the scan+scatter
// composition, so it is an independent oracle for `sort_by_key`.
fn stable_sort_ref(keys: &[u32], payload: &[i32]) -> (Vec<u32>, Vec<i32>) {
    let mut idx: Vec<usize> = (0..keys.len()).collect();
    idx.sort_by_key(|&i| keys[i]);
    (
        idx.iter().map(|&i| keys[i]).collect(),
        idx.iter().map(|&i| payload[i]).collect(),
    )
}

// rank[i] = #{ j < i : digit[j] == digit[i] }; counts[k] = #{ i : digit[i]==k }.
fn rank_ref(digit: &[u32], k: usize) -> (Vec<i32>, Vec<i32>) {
    let mut counts = vec![0i32; k];
    let mut rank = vec![0i32; digit.len()];
    for (i, &d) in digit.iter().enumerate() {
        let d = d as usize;
        rank[i] = counts[d];
        counts[d] += 1;
    }
    (rank, counts)
}

// Deterministic, host-provided key stream (no device RNG) spanning several warps.
fn pattern(n: usize, k: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * 7 + 3) % k) as u32).collect()
}

// Assert the within-key ranks are a permutation of 0..counts[k] (uniqueness +
// completeness), and that ranks increase with index within a key (stability).
fn assert_rank_invariants(digit: &[u32], rank: &[i32], counts: &[i32], k: usize) {
    let mut seen: Vec<Vec<i32>> = vec![Vec::new(); k];
    let mut last_idx_for_key: Vec<Option<usize>> = vec![None; k];
    for (i, &d) in digit.iter().enumerate() {
        let d = d as usize;
        if let Some(prev) = last_idx_for_key[d] {
            assert!(
                rank[i] > rank[prev],
                "stability: key {d} at idx {i} must out-rank earlier idx {prev}"
            );
        }
        last_idx_for_key[d] = Some(i);
        seen[d].push(rank[i]);
    }
    for key in 0..k {
        let mut got = seen[key].clone();
        got.sort_unstable();
        let want: Vec<i32> = (0..counts[key]).collect();
        assert_eq!(got, want, "key {key} ranks must be 0..counts[{key}]");
    }
}

// Sort-readiness: base = exclusive scan of counts; pos[i] = base[digit[i]] +
// rank[i] must be a permutation of 0..N. This is the property that makes
// rank_by_key usable to build a stable sort.
fn assert_sort_ready(digit: &[u32], rank: &[i32], counts: &[i32], k: usize) {
    let mut base = vec![0i32; k];
    let mut acc = 0i32;
    for key in 0..k {
        base[key] = acc;
        acc += counts[key];
    }
    let mut hit = vec![false; digit.len()];
    for (i, &d) in digit.iter().enumerate() {
        let pos = (base[d as usize] + rank[i]) as usize;
        assert!(!hit[pos], "position {pos} written twice");
        hit[pos] = true;
    }
    assert!(hit.iter().all(|&b| b), "positions must cover 0..N exactly");
}

// =====================================================================
// Integration level (outside-in): production `sort_by_key` vs an
// independent host stable sort.
// =====================================================================

#[test]
#[ignore = "requires a CUDA GPU and Tile IR assembler; run via run_gpu_tests.sh"]
fn sort_by_key_matches_host_stable_sort() {
    on_test_thread(|| {
        const N: usize = 256; // several warps
        const K: usize = 16;
        let keys = pattern(N, K);
        let payload: Vec<i32> = (0..N as i32).collect(); // original indices
        let (want_keys, want_payload) = stable_sort_ref(&keys, &payload);

        let ctx = Context::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let k_t = up_u32(&keys, &[N], &stream);
        let p_t = up_i32(&payload, &[N], &stream);
        let mut sk = up_u32(&vec![0u32; N], &[N], &stream);
        let mut sp = up_i32(&vec![0i32; N], &[N], &stream);

        let _ = unsafe {
            sort_by_key(
                &k_t,
                &p_t,
                (&mut sk).partition([N]),
                (&mut sp).partition([N]),
            )
        }
        .generics(gen(&["256", "16"]))
        .sync_on(&stream)
        .unwrap();

        assert_eq!(read_u32(sk, &stream), want_keys, "sorted keys");
        // Payload carries original indices; matching the stable oracle's payload
        // is exactly the stability check (equal keys keep input order).
        assert_eq!(
            read_i32(sp, &stream),
            want_payload,
            "sorted payload / stability"
        );
    });
}

// =====================================================================
// Unit level: `rank_by_key` building block vs the exact host reference,
// plus structural invariants (independent of rank_ref).
// =====================================================================

// Which ranking core to launch; both share the contract of rank_by_key_probe.
#[derive(Clone, Copy)]
enum RankImpl {
    OneHot,
    Loop,
}

fn run_rank_with(
    which: RankImpl,
    digit: &[u32],
    k: usize,
    n_tag: &str,
    k_tag: &str,
) -> (Vec<i32>, Vec<i32>) {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let d_t = up_u32(digit, &[digit.len()], &stream);
    let mut rank_t = up_i32(&vec![0i32; digit.len()], &[digit.len()], &stream);
    let mut counts_t = up_i32(&vec![0i32; k], &[k], &stream);
    match which {
        RankImpl::OneHot => unsafe {
            rank_by_key_probe(
                &d_t,
                (&mut rank_t).partition([digit.len()]),
                (&mut counts_t).partition([k]),
            )
        }
        .generics(gen(&[n_tag, k_tag]))
        .sync_on(&stream)
        .unwrap(),
        RankImpl::Loop => unsafe {
            rank_by_key_loop(
                &d_t,
                (&mut rank_t).partition([digit.len()]),
                (&mut counts_t).partition([k]),
            )
        }
        .generics(gen(&[n_tag, k_tag]))
        .sync_on(&stream)
        .unwrap(),
    };
    (read_i32(rank_t, &stream), read_i32(counts_t, &stream))
}

fn run_rank(digit: &[u32], k: usize, n_tag: &str, k_tag: &str) -> (Vec<i32>, Vec<i32>) {
    run_rank_with(RankImpl::OneHot, digit, k, n_tag, k_tag)
}

#[test]
#[ignore = "requires a CUDA GPU and Tile IR assembler; run via run_gpu_tests.sh"]
fn rank_by_key_small_exact() {
    on_test_thread(|| {
        // Hand-checkable: rank=[0,0,1,0,1,2,0,2], counts=[3,1,3,1]. Shapes are
        // powers of two: Tile IR partition views reject other tile extents.
        let digit = [2u32, 0, 2, 1, 0, 2, 3, 0];
        let (want_rank, want_counts) = rank_ref(&digit, 4);
        let (rank, counts) = run_rank(&digit, 4, "8", "4");
        assert_eq!(rank, want_rank, "rank");
        assert_eq!(counts, want_counts, "counts");
    });
}

#[test]
#[ignore = "requires a CUDA GPU and Tile IR assembler; run via run_gpu_tests.sh"]
fn rank_by_key_loop_matches_reference() {
    on_test_thread(|| {
        const N: usize = 1024;
        const K: usize = 16;
        let digit = pattern(N, K);
        let (want_rank, want_counts) = rank_ref(&digit, K);
        let (rank, counts) = run_rank_with(RankImpl::Loop, &digit, K, "1024", "16");
        assert_eq!(rank, want_rank, "rank");
        assert_eq!(counts, want_counts, "counts");
    });
}

#[test]
#[ignore = "requires a CUDA GPU and Tile IR assembler; run via run_gpu_tests.sh"]
fn rank_by_key_multiwarp_matches_reference_and_invariants() {
    on_test_thread(|| {
        const N: usize = 1024; // 32 warps: exercises the cross-warp stitch
        const K: usize = 16;
        let digit = pattern(N, K);
        let (want_rank, want_counts) = rank_ref(&digit, K);
        let (rank, counts) = run_rank(&digit, K, "1024", "16");
        assert_eq!(rank, want_rank, "rank");
        assert_eq!(counts, want_counts, "counts");
        assert_eq!(counts.iter().sum::<i32>(), N as i32, "counts sum to N");
        assert_rank_invariants(&digit, &rank, &counts, K);
        assert_sort_ready(&digit, &rank, &counts, K);
    });
}

#[test]
#[ignore = "requires a CUDA GPU and Tile IR assembler; run via run_gpu_tests.sh"]
fn rank_by_key_edge_cases() {
    on_test_thread(|| {
        const N: usize = 256;
        const K: usize = 8;
        // All one key: rank = 0..N, one nonzero count.
        let all_same = vec![3u32; N];
        let (rank, counts) = run_rank(&all_same, K, "256", "8");
        assert_eq!(rank, (0..N as i32).collect::<Vec<_>>(), "all-same ranks");
        assert_eq!(counts[3], N as i32, "all-same count");
        assert_eq!(counts.iter().sum::<i32>(), N as i32, "no stray counts");

        // Round-robin: every key equally frequent; a key is never absent here,
        // but ranks must still form 0..counts[k] per key.
        let rr: Vec<u32> = (0..N).map(|i| (i % K) as u32).collect();
        let (rank, counts) = run_rank(&rr, K, "256", "8");
        let (wr, wc) = rank_ref(&rr, K);
        assert_eq!(rank, wr, "round-robin rank");
        assert_eq!(counts, wc, "round-robin counts");
        assert_sort_ready(&rr, &rank, &counts, K);
    });
}
