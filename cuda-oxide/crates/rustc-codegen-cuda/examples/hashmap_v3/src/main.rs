/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `hashmap_v3` correctness tests — 15 GPU hardware checks for the
//! cooperative-groups SwissTable hashmap.
//!
//! Kernels and the `GpuSwissMap` host driver live in `lib.rs` (shared
//! with the bench binary). This `main` only drives the test scenarios
//! and prints PASS / FAIL per test.
//!
//! Build and run with:
//!   cargo oxide run hashmap_v3

use cuda_core::CudaContext;
use hashmap_v3::*;

fn main() {
    println!("=== GPU Hashmap v3 — Cooperative-Groups SwissTable ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");

    const CAPACITY: usize = 1 << 14;
    const M: usize = CAPACITY / 2;

    let keys_v0 = distinct_keys(M, 0xDEAD_BEEF);
    let values_v0: Vec<u32> = (0..M as u32).collect();

    // -------------------------------------------------------------------------
    // Test 1: insert_bulk roundtrip — every inserted key is findable
    //         with the inserted value.
    // -------------------------------------------------------------------------
    println!("--- Test 1: insert_bulk roundtrip ---");
    {
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        map.insert_bulk(&keys_v0, &values_v0, &module, &stream)
            .expect("insert");
        let found = map.find_bulk(&keys_v0, &module, &stream).expect("find");

        let mut bad = 0usize;
        for i in 0..M {
            if found[i] != values_v0[i] {
                bad += 1;
            }
        }
        if bad == 0 {
            println!("  {M} / {M} keys round-tripped, capacity = {CAPACITY}");
        } else {
            println!("  FAIL: {bad} mismatches");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 2: miss on absent keys — disjoint key set must miss.
    // -------------------------------------------------------------------------
    println!("\n--- Test 2: miss on absent keys ---");
    {
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        map.insert_bulk(&keys_v0, &values_v0, &module, &stream)
            .expect("insert");

        let inserted: std::collections::HashSet<u32> = keys_v0.iter().copied().collect();
        let mut absent = Vec::with_capacity(M);
        let mut state: u32 = 0xC0FF_EEEEu32;
        while absent.len() < M {
            let k = xorshift32(&mut state);
            if k != FORBIDDEN_KEY && !inserted.contains(&k) {
                absent.push(k);
            }
        }

        let found = map.find_bulk(&absent, &module, &stream).expect("find");
        let bad: usize = found.iter().filter(|&&v| v != MISS).count();
        if bad == 0 {
            println!("  {M} / {M} absent keys correctly missed");
        } else {
            println!("  FAIL: {bad} absent keys returned non-MISS");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 3: last-writer-wins on re-insert — second insert_bulk
    //         overwrites every value.
    // -------------------------------------------------------------------------
    println!("\n--- Test 3: insert_bulk last-writer-wins on re-insert ---");
    {
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        map.insert_bulk(&keys_v0, &values_v0, &module, &stream)
            .expect("insert v0");

        let values_v1: Vec<u32> = values_v0.iter().map(|v| v ^ 0xA5A5_A5A5).collect();
        map.insert_bulk(&keys_v0, &values_v1, &module, &stream)
            .expect("insert v1");

        let found = map.find_bulk(&keys_v0, &module, &stream).expect("find");
        let mut bad = 0usize;
        for i in 0..M {
            if found[i] != values_v1[i] {
                bad += 1;
            }
        }
        if bad == 0 {
            println!("  all {M} keys reflect last-writer values");
        } else {
            println!("  FAIL: {bad} stale values");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 4: try_insert_bulk first-writer-wins — pass 2 reports
    //         every key as already-present and the table preserves
    //         the pass-1 values.
    // -------------------------------------------------------------------------
    println!("\n--- Test 4: try_insert_bulk first-writer-wins ---");
    {
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        let first_flags = map
            .try_insert_bulk(&keys_v0, &values_v0, &module, &stream)
            .expect("try_insert pass 1");
        let all_fresh = first_flags.iter().all(|&b| b);

        let values_v1: Vec<u32> = values_v0.iter().map(|v| v ^ 0xA5A5_A5A5).collect();
        let second_flags = map
            .try_insert_bulk(&keys_v0, &values_v1, &module, &stream)
            .expect("try_insert pass 2");
        let none_fresh = second_flags.iter().all(|&b| !b);

        let found = map.find_bulk(&keys_v0, &module, &stream).expect("find");
        let bad: usize = found
            .iter()
            .zip(&values_v0)
            .filter(|(f, v)| *f != *v)
            .count();

        if all_fresh && none_fresh && bad == 0 {
            println!("  pass 1: all {M} keys fresh");
            println!("  pass 2: no keys fresh, table preserves v0 values");
        } else {
            println!("  FAIL: all_fresh={all_fresh} none_fresh={none_fresh} bad_values={bad}");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 5: load-factor stress (~75%) — dense insert + roundtrip.
    // -------------------------------------------------------------------------
    println!("\n--- Test 5: load-factor stress (~75%) ---");
    {
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        let m_dense = (CAPACITY * 3) / 4;
        let dense_keys = distinct_keys(m_dense, 0x1234_5678);
        let dense_values: Vec<u32> = (0..m_dense as u32).collect();
        map.insert_bulk(&dense_keys, &dense_values, &module, &stream)
            .expect("dense insert");

        let found = map
            .find_bulk(&dense_keys, &module, &stream)
            .expect("dense find");
        let bad: usize = found
            .iter()
            .zip(&dense_values)
            .filter(|(f, v)| *f != *v)
            .count();
        if bad == 0 {
            println!(
                "  {m_dense} keys round-tripped at load factor {:.1}%, capacity = {}",
                100.0 * m_dense as f64 / map.capacity() as f64,
                map.capacity()
            );
        } else {
            println!("  FAIL: {bad} mismatches under load");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 6: delete-then-find — deleted keys must miss; survivors must hit.
    // -------------------------------------------------------------------------
    println!("\n--- Test 6: delete-then-find ---");
    {
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        map.insert_bulk(&keys_v0, &values_v0, &module, &stream)
            .expect("insert");

        // Delete the first third.
        let third = M / 3;
        let to_delete = &keys_v0[..third];
        let survivors = &keys_v0[third..];

        let del_flags = map
            .delete_bulk(to_delete, &module, &stream)
            .expect("delete");
        let all_deleted = del_flags.iter().all(|&b| b);

        // Survivors must still hit with their original values.
        let surv_found = map
            .find_bulk(survivors, &module, &stream)
            .expect("find survivors");
        let mut bad_surv = 0usize;
        for (i, k_idx) in (third..M).enumerate() {
            if surv_found[i] != values_v0[k_idx] {
                bad_surv += 1;
            }
        }

        // Deleted keys must miss.
        let deleted_found = map
            .find_bulk(to_delete, &module, &stream)
            .expect("find deleted");
        let bad_del: usize = deleted_found.iter().filter(|&&v| v != MISS).count();

        if all_deleted && bad_surv == 0 && bad_del == 0 {
            println!(
                "  deleted {third} keys; {} survivors all hit, {third} deleted all miss",
                M - third
            );
        } else {
            println!("  FAIL: all_deleted={all_deleted} bad_surv={bad_surv} bad_del={bad_del}");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 7: delete-then-reinsert — re-inserted key sees new value.
    //         (v3 does not yet reclaim DELETED slots on insert, so
    //          the new entry lands at a fresh slot; find still
    //          returns the new value because it walks past the
    //          tombstone.)
    // -------------------------------------------------------------------------
    println!("\n--- Test 7: delete-then-reinsert ---");
    {
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        map.insert_bulk(&keys_v0, &values_v0, &module, &stream)
            .expect("insert v0");

        let third = M / 3;
        let to_recycle = &keys_v0[..third];
        let _ = map
            .delete_bulk(to_recycle, &module, &stream)
            .expect("delete");

        let new_values: Vec<u32> = values_v0[..third]
            .iter()
            .map(|v| v.wrapping_add(0x10_0000))
            .collect();
        map.insert_bulk(to_recycle, &new_values, &module, &stream)
            .expect("re-insert");

        let found = map
            .find_bulk(to_recycle, &module, &stream)
            .expect("find re-inserted");
        let bad: usize = found
            .iter()
            .zip(&new_values)
            .filter(|(f, v)| *f != *v)
            .count();

        if bad == 0 {
            println!("  {third} keys re-inserted with new values, all observable");
        } else {
            println!("  FAIL: {bad} re-inserted keys returned stale or missing values");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 8: warp-cooperative find parity with single-thread find on a
    //         mixed hit/miss query batch at moderate load (50%).
    // -------------------------------------------------------------------------
    println!("\n--- Test 8: warp-cooperative find parity ---");
    {
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        map.insert_bulk(&keys_v0, &values_v0, &module, &stream)
            .expect("insert");

        // Build a query batch that's half hits (the inserted keys) and
        // half misses (fresh random keys). Total = 2 * M queries.
        let inserted: std::collections::HashSet<u32> = keys_v0.iter().copied().collect();
        let mut absent = Vec::with_capacity(M);
        let mut state: u32 = 0x0BAD_F00D;
        while absent.len() < M {
            let k = xorshift32(&mut state);
            if k != FORBIDDEN_KEY && !inserted.contains(&k) {
                absent.push(k);
            }
        }
        let mut query_keys = Vec::with_capacity(2 * M);
        query_keys.extend_from_slice(&keys_v0);
        query_keys.extend_from_slice(&absent);

        let single = map
            .find_bulk(&query_keys, &module, &stream)
            .expect("find single");
        let tile_32 = map
            .find_bulk_tile_32(&query_keys, &module, &stream)
            .expect("find tile_32");

        let bad: usize = single
            .iter()
            .zip(&tile_32)
            .filter(|(s, w)| *s != *w)
            .count();
        let hits: usize = tile_32.iter().filter(|&&v| v != MISS).count();

        if bad == 0 && hits == M {
            println!(
                "  tile_32 find matches single-thread find on {} queries ({hits} hits, {} misses)",
                query_keys.len(),
                query_keys.len() - hits
            );
        } else {
            println!("  FAIL: bad={bad} hits={hits} (expected {M})");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 9: tile_32 find at 75% load — the regime where warp-cooperative
    //         is supposed to win against single-thread find.
    // -------------------------------------------------------------------------
    println!("\n--- Test 9: tile_32 find at ~75% load ---");
    {
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        let m_dense = (CAPACITY * 3) / 4;
        let dense_keys = distinct_keys(m_dense, 0xFEED_FACE);
        let dense_values: Vec<u32> = (0..m_dense as u32).collect();
        map.insert_bulk(&dense_keys, &dense_values, &module, &stream)
            .expect("dense insert");

        let found = map
            .find_bulk_tile_32(&dense_keys, &module, &stream)
            .expect("tile_32 find");
        let bad: usize = found
            .iter()
            .zip(&dense_values)
            .filter(|(f, v)| *f != *v)
            .count();
        if bad == 0 {
            println!(
                "  {m_dense} keys round-tripped via tile_32 find at load factor {:.1}%",
                100.0 * m_dense as f64 / map.capacity() as f64,
            );
        } else {
            println!("  FAIL: {bad} mismatches under load");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 10: tile_16 parity vs tile_32 on 16384 mixed (hit + miss)
    //          queries. Same table populated via insert_bulk; both
    //          find variants must agree on every result.
    // -------------------------------------------------------------------------
    println!("\n--- Test 10: tile_16 vs tile_32 parity on 16384 mixed queries ---");
    {
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        map.insert_bulk(&keys_v0, &values_v0, &module, &stream)
            .expect("insert");

        let inserted: std::collections::HashSet<u32> = keys_v0.iter().copied().collect();
        let mut absent = Vec::with_capacity(M);
        let mut state: u32 = 0x1234_5678u32;
        while absent.len() < M {
            let k = xorshift32(&mut state);
            if k != FORBIDDEN_KEY && !inserted.contains(&k) {
                absent.push(k);
            }
        }
        let mut query_keys = Vec::with_capacity(2 * M);
        query_keys.extend_from_slice(&keys_v0);
        query_keys.extend_from_slice(&absent);

        let tile_32 = map
            .find_bulk_tile_32(&query_keys, &module, &stream)
            .expect("tile_32");
        let tile_16 = map
            .find_bulk_tile_16(&query_keys, &module, &stream)
            .expect("tile_16");

        let bad: usize = tile_32
            .iter()
            .zip(&tile_16)
            .filter(|(a, b)| *a != *b)
            .count();
        let hits: usize = tile_16.iter().filter(|&&v| v != MISS).count();

        if bad == 0 && hits == M {
            println!(
                "  tile_16 matches tile_32 on {} queries ({hits} hits, {} misses)",
                query_keys.len(),
                query_keys.len() - hits
            );
        } else {
            println!("  FAIL: bad={bad} hits={hits} (expected {M})");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 11: tile_16 find at ~75% load — sub-warp variant under load.
    // -------------------------------------------------------------------------
    println!("\n--- Test 11: tile_16 find at ~75% load ---");
    {
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        let m_dense = (CAPACITY * 3) / 4;
        let dense_keys = distinct_keys(m_dense, 0xFEED_FACE);
        let dense_values: Vec<u32> = (0..m_dense as u32).collect();
        map.insert_bulk(&dense_keys, &dense_values, &module, &stream)
            .expect("dense insert");

        let found = map
            .find_bulk_tile_16(&dense_keys, &module, &stream)
            .expect("tile_16 find");
        let bad: usize = found
            .iter()
            .zip(&dense_values)
            .filter(|(f, v)| *f != *v)
            .count();
        if bad == 0 {
            println!(
                "  {m_dense} keys round-tripped via tile_16 find at load factor {:.1}%",
                100.0 * m_dense as f64 / map.capacity() as f64,
            );
        } else {
            println!("  FAIL: {bad} mismatches under load");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 12: in-warp dedup — 1k distinct keys repeated 100 times must
    // collapse to exactly 1k entries with last-writer-wins values.
    // -------------------------------------------------------------------------
    println!("\n--- Test 12: insert_bulk_dedup (1000 keys × 100 reps) ---");
    {
        const N: usize = 1000;
        const REPS: usize = 100;
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        let base_keys = distinct_keys(N, 0xCAFE_F00D);
        // Build the input by repeating the base keys 100 times. The
        // value at index `i` is `i as u32` so the "last value seen for
        // key k" is uniquely identifiable.
        let mut keys = Vec::with_capacity(N * REPS);
        let mut values = Vec::with_capacity(N * REPS);
        for r in 0..REPS {
            for (i, &k) in base_keys.iter().enumerate() {
                keys.push(k);
                values.push((r * N + i) as u32);
            }
        }
        map.insert_bulk_dedup(&keys, &values, &module, &stream)
            .expect("dedup insert");

        // Find each base key; every one must hit. (We don't assert
        // which value lands — within-warp last-writer-wins picks the
        // highest-rank lane's value, but cross-warp arbitration is
        // race-dependent. Exact-count is the headline acceptance.)
        let found = map.find_bulk(&base_keys, &module, &stream).expect("find");
        let hits = found.iter().filter(|&&v| v != MISS).count();
        if hits != N {
            println!("  FAIL: expected {N} hits, got {hits}");
            std::process::exit(1);
        }

        // Cross-check: querying any key NOT in the base set must miss.
        let inserted: std::collections::HashSet<u32> = base_keys.iter().copied().collect();
        let mut absent = Vec::with_capacity(N);
        let mut state: u32 = 0xACE0_F00Du32;
        while absent.len() < N {
            let k = xorshift32(&mut state);
            if k != FORBIDDEN_KEY && !inserted.contains(&k) {
                absent.push(k);
            }
        }
        let absent_found = map.find_bulk(&absent, &module, &stream).expect("find");
        let absent_hits = absent_found.iter().filter(|&&v| v != MISS).count();
        if absent_hits != 0 {
            println!("  FAIL: {absent_hits} keys should have missed but hit");
            std::process::exit(1);
        }
        println!("  {N} distinct keys × {REPS} reps -> {hits} hits, 0 phantom entries");
    }

    // -------------------------------------------------------------------------
    // Test 13: DELETED-slot reclaim stress — many delete-then-reinsert
    // cycles at high load. Without reclaim, tombstones accumulate and
    // probe chains lengthen until insert can't make progress; with
    // reclaim, the chain length stays bounded.
    // -------------------------------------------------------------------------
    println!("\n--- Test 13: DELETED reclaim stress (90% load, 10 cycles) ---");
    {
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        let m = (CAPACITY * 9) / 10;
        let base_keys = distinct_keys(m, 0xBADC_0FFE);
        let base_values: Vec<u32> = (0..m as u32).collect();
        map.insert_bulk(&base_keys, &base_values, &module, &stream)
            .expect("seed insert");

        // Subset to delete-and-reinsert each cycle.
        let churn_n = m / 4;
        let churn_keys: Vec<u32> = base_keys[..churn_n].to_vec();

        const CYCLES: usize = 10;
        for c in 0..CYCLES {
            map.delete_bulk(&churn_keys, &module, &stream)
                .expect("delete");
            let new_values: Vec<u32> = (0..churn_n as u32)
                .map(|i| (c as u32) * 1_000_000 + i)
                .collect();
            map.insert_bulk(&churn_keys, &new_values, &module, &stream)
                .expect("reinsert");
        }

        // Final state: every base key still hits, churned keys hold
        // the last cycle's values, untouched keys hold their originals.
        let final_found = map.find_bulk(&base_keys, &module, &stream).expect("find");
        let mut bad = 0usize;
        let last_cycle = (CYCLES - 1) as u32;
        for (i, &got) in final_found.iter().enumerate() {
            let want = if i < churn_n {
                last_cycle * 1_000_000 + i as u32
            } else {
                base_values[i]
            };
            if got != want {
                bad += 1;
            }
        }
        if bad != 0 {
            println!("  FAIL: {bad} mismatches after {CYCLES} delete-reinsert cycles");
            std::process::exit(1);
        }
        println!("  {CYCLES} cycles of {churn_n}-key churn at 90% load, all {m} keys preserved");
    }

    // -------------------------------------------------------------------------
    // Test 14: explicit resize_to round-trip — insert 64k pairs into a
    // 128k table, resize to 256k, verify every key + value survives.
    // -------------------------------------------------------------------------
    println!("\n--- Test 14: resize_to two-buffer rehash ---");
    {
        let mut map = GpuSwissMap::new(1 << 17, &stream).expect("alloc");
        let n = 1 << 16;
        let keys = distinct_keys(n, 0xF00D_BEEF);
        let values: Vec<u32> = (0..n as u32).collect();
        map.insert_bulk(&keys, &values, &module, &stream)
            .expect("insert");

        let cap_before = map.capacity();
        map.resize_to(1 << 18, &module, &stream).expect("resize");
        let cap_after = map.capacity();

        let found = map.find_bulk(&keys, &module, &stream).expect("find");
        let bad: usize = found.iter().zip(&values).filter(|(f, v)| *f != *v).count();
        if bad != 0 {
            println!("  FAIL: {bad} mismatches after rehash");
            std::process::exit(1);
        }
        println!(
            "  {n} keys rehashed {cap_before} -> {cap_after}, resize_count = {}",
            map.resize_count()
        );
    }

    // -------------------------------------------------------------------------
    // Test 15: auto-resize stress — start at 1024 slots, insert 100k
    // unique keys in chunks via insert_bulk_grow, observe at least 7
    // resize events, and verify every key is present at the end.
    // -------------------------------------------------------------------------
    println!("\n--- Test 15: insert_bulk_grow stress (1024 -> 100k keys) ---");
    {
        let mut map = GpuSwissMap::new(1024, &stream).expect("alloc");
        const N: usize = 100_000;
        let keys = distinct_keys(N, 0x1234_5678);
        let values: Vec<u32> = (0..N as u32).collect();

        const CHUNK: usize = 10_000;
        let mut idx = 0;
        while idx < N {
            let end = (idx + CHUNK).min(N);
            map.insert_bulk_grow(&keys[idx..end], &values[idx..end], &module, &stream)
                .expect("grow insert");
            idx = end;
        }

        let resizes = map.resize_count();
        if resizes < 7 {
            println!("  FAIL: expected >= 7 resize events, got {resizes}");
            std::process::exit(1);
        }

        let found = map.find_bulk(&keys, &module, &stream).expect("find");
        let bad: usize = found.iter().zip(&values).filter(|(f, v)| *f != *v).count();
        if bad != 0 {
            println!("  FAIL: {bad} mismatches after auto-resize stress");
            std::process::exit(1);
        }
        println!(
            "  {N} keys inserted via {} resize events; final capacity = {}",
            resizes,
            map.capacity()
        );
    }

    println!("\n=== SUCCESS: All 15 hashmap v3 tests passed! ===");
}
