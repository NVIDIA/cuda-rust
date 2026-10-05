/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU Hashmap v1 — Open-Addressed Static Map
//!
//! A fixed-capacity, open-addressed `u32 -> u32` hashmap that runs entirely
//! on the GPU. This is the v1 baseline: the smallest possible algorithm
//! that proves the end-to-end pipeline — host allocation, kernel launch,
//! atomic CAS, correctness harness — so the moving parts are obvious.
//! `hashmap_v2` and `hashmap_v3` are sibling example crates that layer
//! SwissTable machinery (control-byte fingerprints, triangular probing,
//! warp-cooperative probe, cooperative-groups insert/find) on top.
//!
//! Storage is a single `DeviceBuffer<u64>` of length `N` (power of two) where
//! each `u64` slot packs `(key as u64) << 32 | value as u64`. The empty
//! sentinel is `u64::MAX` (i.e. `(0xFFFFFFFF, 0xFFFFFFFF)`), which the user
//! is forbidden from inserting.
//!
//! Two insert contracts are exposed:
//!   - `insert_bulk`     — last-writer-wins. Matches `hashbrown::HashMap::insert`.
//!   - `try_insert_bulk` — first-writer-wins. Reports per-key "fresh vs present".
//!
//! Build and run with:
//!   cargo oxide run hashmap

use std::sync::Arc;

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU64};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

// =============================================================================
// SHARED CONSTANTS AND HELPERS (compiled both host- and device-side)
// =============================================================================

/// Sentinel for an empty slot. The packed pair `(u32::MAX, u32::MAX)` is
/// reserved for this purpose; users may not insert it.
const EMPTY: u64 = u64::MAX;

/// Sentinel value the host returns from `find_bulk` for missing keys.
const MISS: u32 = u32::MAX;

/// FxHash multiplier (golden-ratio-derived constant). Same one hashbrown
/// uses behind `FxHashMap`. Cheap on GPU: one widening multiply.
const FX_K: u64 = 0x517c_c1b7_2722_0a95;

/// FxHash-style single-multiply hash from a `u32` key to a 64-bit hash.
///
/// Returns 64 bits so the SwissTable variants in `hashmap_v2` and
/// `hashmap_v3` can split into a low-bit `h1` (probe position) and
/// a high-bit `h2` (fingerprint) without a second hash call.
#[inline(always)]
fn hash_u32(key: u32) -> u64 {
    (key as u64).wrapping_mul(FX_K)
}

/// Pack `(key, value)` into a single `u64` slot. Key occupies the upper 32
/// bits, value the lower 32. Inverse of [`unpack_key`] / [`unpack_value`].
///
/// Putting both fields in one machine word is what lets a single
/// `compare_exchange` publish a `(k, v)` atomically: no thread can observe
/// "key written but value still EMPTY".
#[inline(always)]
fn pack(key: u32, value: u32) -> u64 {
    ((key as u64) << 32) | (value as u64)
}

/// Recover the key from a packed slot (upper 32 bits).
#[inline(always)]
fn unpack_key(slot: u64) -> u32 {
    (slot >> 32) as u32
}

/// Recover the value from a packed slot (lower 32 bits).
#[inline(always)]
fn unpack_value(slot: u64) -> u32 {
    (slot & 0xFFFF_FFFF) as u32
}

// =============================================================================
// KERNELS
// =============================================================================

#[cuda_module]
mod kernels {
    use super::*;

    /// `insert_kernel` — last-writer-wins.
    ///
    /// One thread per input key. Linear-probes from `hash(key) & mask` until it
    /// either CAS-claims an EMPTY slot or finds the key already present, in
    /// which case it overwrites the value via an inner CAS loop.
    ///
    /// `table.len()` must be a power of two (host-side invariant).
    #[kernel]
    pub fn insert_kernel(table: &[DeviceAtomicU64], keys: &[u32], values: &[u32]) {
        let tid = thread::index_1d().get();
        if tid >= keys.len() {
            return;
        }

        let key = keys[tid];
        let value = values[tid];
        let mask = table.len() - 1;
        let mut idx = (hash_u32(key) as usize) & mask;

        loop {
            let slot = &table[idx];

            match slot.compare_exchange(
                EMPTY,
                pack(key, value),
                AtomicOrdering::AcqRel,
                AtomicOrdering::Relaxed,
            ) {
                Ok(_) => return,
                Err(observed) if unpack_key(observed) == key => {
                    let mut expected = observed;
                    let desired = pack(key, value);
                    loop {
                        match slot.compare_exchange(
                            expected,
                            desired,
                            AtomicOrdering::AcqRel,
                            AtomicOrdering::Relaxed,
                        ) {
                            Ok(_) => return,
                            // Fires only when another thread in this same
                            // batch holds the same key and CAS'd in first.
                            // Re-read its value and retry so last-writer-wins.
                            Err(actual) => expected = actual,
                        }
                    }
                }
                // Hash collision: different key already lives here. Probe forward.
                Err(_) => idx = (idx + 1) & mask,
            }
        }
    }

    /// `try_insert_kernel` — first-writer-wins.
    ///
    /// Same outer probe as `insert_kernel`, but on duplicate it leaves the
    /// existing slot untouched and reports `1 = already present` via `out`.
    /// On a fresh insert it reports `0`.
    #[kernel]
    pub fn try_insert_kernel(
        table: &[DeviceAtomicU64],
        keys: &[u32],
        values: &[u32],
        mut out: DisjointSlice<u32>,
    ) {
        let tid = thread::index_1d();
        let tid_raw = tid.get();
        let i = tid_raw;
        if i >= keys.len() {
            return;
        }

        let key = keys[i];
        let value = values[i];
        let mask = table.len() - 1;
        let mut idx = (hash_u32(key) as usize) & mask;

        loop {
            let slot = &table[idx];

            match slot.compare_exchange(
                EMPTY,
                pack(key, value),
                AtomicOrdering::AcqRel,
                AtomicOrdering::Relaxed,
            ) {
                Ok(_) => {
                    if let Some(o) = out.get_mut(tid) {
                        *o = 0;
                    }
                    return;
                }
                Err(observed) if unpack_key(observed) == key => {
                    if let Some(o) = out.get_mut(tid) {
                        *o = 1;
                    }
                    return;
                }
                Err(_) => idx = (idx + 1) & mask,
            }
        }
    }

    /// `find_kernel` — one key per thread.
    ///
    /// Linear-probe from `hash(key) & mask` and return the value if a key match
    /// is found, or `MISS = u32::MAX` if an EMPTY slot is hit first.
    #[kernel]
    pub fn find_kernel(table: &[DeviceAtomicU64], keys: &[u32], mut out: DisjointSlice<u32>) {
        let tid = thread::index_1d();
        let tid_raw = tid.get();
        let i = tid_raw;
        if i >= keys.len() {
            return;
        }

        let key = keys[i];
        let mask = table.len() - 1;
        let mut idx = (hash_u32(key) as usize) & mask;

        loop {
            let slot = &table[idx];
            let observed = slot.load(AtomicOrdering::Acquire);

            if observed == EMPTY {
                if let Some(o) = out.get_mut(tid) {
                    *o = MISS;
                }
                return;
            }
            if unpack_key(observed) == key {
                if let Some(o) = out.get_mut(tid) {
                    *o = unpack_value(observed);
                }
                return;
            }
            idx = (idx + 1) & mask;
        }
    }
}

// =============================================================================
// HOST DRIVER
// =============================================================================

/// Forbidden user key. `(FORBIDDEN_KEY, FORBIDDEN_VALUE)` packs to `EMPTY`,
/// so we refuse any insert that uses `u32::MAX` as the key. Documented as a
/// host-side invariant to match `cuCollections`-style sentinel discipline.
const FORBIDDEN_KEY: u32 = u32::MAX;

/// Host-side handle to a v1 GPU hashmap.
///
/// Owns one device-resident `DeviceBuffer<u64>` of length `capacity`, where
/// each `u64` packs a `(key, value)` pair. The buffer is initialized to
/// all-`0xFF` bytes (every slot reads as `EMPTY = u64::MAX`) and stays
/// alive for the lifetime of this handle.
///
/// All bulk operations launch one of the three kernels above and copy
/// inputs/outputs across the host/device boundary on the supplied stream.
struct GpuHashMap {
    /// Device-side packed `(key, value)` slot array. Length = `capacity`.
    slots: DeviceBuffer<DeviceAtomicU64>,
    /// Number of slots. Always a power of two so `hash & (capacity - 1)`
    /// can serve as the probe-position mask.
    capacity: usize,
}

impl GpuHashMap {
    /// Allocate a fresh, empty table of `capacity` slots. `capacity` must be
    /// a non-zero power of two.
    ///
    /// We allocate via `DeviceBuffer::zeroed` (which gives us a buffer of
    /// length `capacity` worth of zero-initialized `u64`s), then immediately
    /// `memset_d8_async(0xFF)` the bytes so every slot reads as `u64::MAX =
    /// EMPTY`.
    fn new(capacity: usize, stream: &Arc<CudaStream>) -> Result<Self, cuda_core::DriverError> {
        assert!(
            capacity.is_power_of_two(),
            "capacity must be a power of two"
        );
        assert!(capacity > 0, "capacity must be positive");

        let slots = DeviceBuffer::<u64>::zeroed(stream, capacity)?;
        unsafe {
            cuda_core::simt::memory::memset_d8_async(
                slots.cu_deviceptr(),
                0xFF,
                slots.num_bytes(),
                stream.cu_stream(),
            )?;
        }

        // View the storage as atomics so the kernels can soundly mutate it
        // through a shared `&[DeviceAtomicU64]` (issue #151). `DeviceAtomicU64`
        // is `#[repr(transparent)]` over `u64`, so the pointer, length, and the
        // `0xFF`-filled bytes are all unchanged.
        let slots = slots.cast_elem::<DeviceAtomicU64>();

        Ok(Self { slots, capacity })
    }

    /// Number of slots in the table. Fixed at construction time.
    fn capacity(&self) -> usize {
        self.capacity
    }

    /// Last-writer-wins bulk insert. Overwrites existing values.
    fn insert_bulk(
        &self,
        keys: &[u32],
        values: &[u32],
        module: &kernels::LoadedModule,
        stream: &Arc<CudaStream>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(keys.len(), values.len());
        if keys.is_empty() {
            return Ok(());
        }
        debug_assert!(
            keys.iter().all(|&k| k != FORBIDDEN_KEY),
            "u32::MAX is reserved as the EMPTY sentinel and may not be inserted"
        );

        let keys_dev = DeviceBuffer::from_host(stream, keys)?;
        let values_dev = DeviceBuffer::from_host(stream, values)?;

        let cfg = LaunchConfig::for_num_elems(keys.len() as u32);
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.insert_kernel(stream, cfg, &self.slots, &keys_dev, &values_dev) }?;

        Ok(())
    }

    /// First-writer-wins bulk insert. Returns a `Vec<bool>` of length
    /// `keys.len()` where `true` means "this key was fresh, the table now
    /// contains the corresponding value" and `false` means "key was already
    /// present, the table is unchanged for this key".
    fn try_insert_bulk(
        &self,
        keys: &[u32],
        values: &[u32],
        module: &kernels::LoadedModule,
        stream: &Arc<CudaStream>,
    ) -> Result<Vec<bool>, Box<dyn std::error::Error>> {
        assert_eq!(keys.len(), values.len());
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        debug_assert!(
            keys.iter().all(|&k| k != FORBIDDEN_KEY),
            "u32::MAX is reserved as the EMPTY sentinel and may not be inserted"
        );

        let keys_dev = DeviceBuffer::from_host(stream, keys)?;
        let values_dev = DeviceBuffer::from_host(stream, values)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(stream, keys.len())?;

        let cfg = LaunchConfig::for_num_elems(keys.len() as u32);
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.try_insert_kernel(
                stream,
                cfg,
                &self.slots,
                &keys_dev,
                &values_dev,
                &mut out_dev,
            )
        }?;

        let raw = out_dev.to_host_vec(stream)?;
        Ok(raw.into_iter().map(|x| x == 0).collect())
    }

    /// Bulk find. Returns `Vec<u32>` of length `keys.len()`; entries equal
    /// to `MISS = u32::MAX` mean "key not present".
    fn find_bulk(
        &self,
        keys: &[u32],
        module: &kernels::LoadedModule,
        stream: &Arc<CudaStream>,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }

        let keys_dev = DeviceBuffer::from_host(stream, keys)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(stream, keys.len())?;

        let cfg = LaunchConfig::for_num_elems(keys.len() as u32);
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.find_kernel(stream, cfg, &self.slots, &keys_dev, &mut out_dev) }?;

        let out = out_dev.to_host_vec(stream)?;
        Ok(out)
    }
}

// =============================================================================
// CORRECTNESS TESTS
// =============================================================================

/// Tiny xorshift32 RNG so we don't pull in a crate for randomness. Must
/// produce keys in `[0, u32::MAX)` (excluding the forbidden sentinel).
fn xorshift32(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

/// Sample `n` distinct keys, all `< u32::MAX`, deterministically seeded.
fn distinct_keys(n: usize, seed: u32) -> Vec<u32> {
    let mut state = seed;
    let mut seen = std::collections::HashSet::with_capacity(n);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let k = xorshift32(&mut state);
        if k != FORBIDDEN_KEY && seen.insert(k) {
            out.push(k);
        }
    }
    out
}

fn main() {
    println!("=== GPU Hashmap v1 — Open-Addressed Static Map ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    const CAPACITY: usize = 1 << 14;
    const M: usize = CAPACITY / 2;

    let keys_v0 = distinct_keys(M, 0xDEAD_BEEF);
    let values_v0: Vec<u32> = (0..M as u32).collect();

    // -------------------------------------------------------------------------
    // Test 1: insert_bulk roundtrip — every inserted key must be findable
    //         and return the value we inserted.
    // -------------------------------------------------------------------------
    println!("--- Test 1: insert_bulk roundtrip ---");
    {
        let map = GpuHashMap::new(CAPACITY, &stream).expect("alloc");
        map.insert_bulk(&keys_v0, &values_v0, &module, &stream)
            .expect("insert");
        let found = map.find_bulk(&keys_v0, &module, &stream).expect("find");

        let mut hits = 0usize;
        let mut mismatches = 0usize;
        for i in 0..M {
            if found[i] == MISS {
                if mismatches < 5 {
                    eprintln!(
                        "  miss at i={i} key=0x{:08X} (should be 0x{:08X})",
                        keys_v0[i], values_v0[i]
                    );
                }
                mismatches += 1;
            } else if found[i] != values_v0[i] {
                if mismatches < 5 {
                    eprintln!(
                        "  bad value at i={i} key=0x{:08X}: got 0x{:08X}, want 0x{:08X}",
                        keys_v0[i], found[i], values_v0[i]
                    );
                }
                mismatches += 1;
            } else {
                hits += 1;
            }
        }
        if mismatches == 0 {
            println!("  {hits} / {M} keys round-tripped, capacity = {CAPACITY}");
        } else {
            println!("  FAIL: {mismatches} mismatches");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 2: miss on absent keys — disjoint key set, every find must miss.
    // -------------------------------------------------------------------------
    println!("\n--- Test 2: miss on absent keys ---");
    {
        let map = GpuHashMap::new(CAPACITY, &stream).expect("alloc");
        map.insert_bulk(&keys_v0, &values_v0, &module, &stream)
            .expect("insert");

        // Sample a fresh key set that doesn't overlap the inserted one.
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
            println!("  {} / {M} absent keys correctly missed", M);
        } else {
            println!("  FAIL: {bad} absent keys returned non-MISS");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 3: last-writer-wins on re-insert — re-insert the same keys with
    //         a different value set; finds must return the new values.
    // -------------------------------------------------------------------------
    println!("\n--- Test 3: insert_bulk last-writer-wins on re-insert ---");
    {
        let map = GpuHashMap::new(CAPACITY, &stream).expect("alloc");
        map.insert_bulk(&keys_v0, &values_v0, &module, &stream)
            .expect("insert v0");

        let values_v1: Vec<u32> = values_v0.iter().map(|v| v ^ 0xA5A5_A5A5).collect();
        map.insert_bulk(&keys_v0, &values_v1, &module, &stream)
            .expect("insert v1");

        let found = map.find_bulk(&keys_v0, &module, &stream).expect("find");
        let mut bad = 0usize;
        for i in 0..M {
            if found[i] != values_v1[i] {
                if bad < 5 {
                    eprintln!(
                        "  i={i} key=0x{:08X}: got 0x{:08X}, want v1=0x{:08X} (v0 was 0x{:08X})",
                        keys_v0[i], found[i], values_v1[i], values_v0[i],
                    );
                }
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
    // Test 4: try_insert_bulk first-writer-wins — second call with same
    //         keys but new values must report all-present and find must
    //         still return the first-call values.
    // -------------------------------------------------------------------------
    println!("\n--- Test 4: try_insert_bulk first-writer-wins ---");
    {
        let map = GpuHashMap::new(CAPACITY, &stream).expect("alloc");
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
        let mut bad = 0usize;
        for i in 0..M {
            if found[i] != values_v0[i] {
                bad += 1;
            }
        }

        if all_fresh && none_fresh && bad == 0 {
            println!("  pass 1: all {M} keys fresh");
            println!("  pass 2: no keys fresh, table preserves v0 values");
        } else {
            println!("  FAIL: all_fresh={all_fresh} none_fresh={none_fresh} bad_values={bad}");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 5: mixed batch — half already-present, half fresh. The flag
    //         array must be exactly true on the fresh half, and the
    //         already-present half must keep its original values.
    // -------------------------------------------------------------------------
    println!("\n--- Test 5: try_insert_bulk mixed dup/fresh batch ---");
    {
        let map = GpuHashMap::new(CAPACITY, &stream).expect("alloc");
        let initial_keys = &keys_v0[..M / 2];
        let initial_values = &values_v0[..M / 2];
        map.insert_bulk(initial_keys, initial_values, &module, &stream)
            .expect("seed");

        let mixed_keys = keys_v0.clone();
        let mixed_values: Vec<u32> = values_v0.iter().map(|v| v.wrapping_add(0x1000)).collect();
        let flags = map
            .try_insert_bulk(&mixed_keys, &mixed_values, &module, &stream)
            .expect("try_insert mixed");

        let mut bad_flags = 0usize;
        for (i, &flag) in flags.iter().enumerate() {
            let expected_fresh = i >= M / 2;
            if flag != expected_fresh {
                bad_flags += 1;
            }
        }

        let found = map.find_bulk(&mixed_keys, &module, &stream).expect("find");
        let mut bad_values = 0usize;
        for i in 0..M {
            let expected = if i < M / 2 {
                values_v0[i]
            } else {
                mixed_values[i]
            };
            if found[i] != expected {
                bad_values += 1;
            }
        }

        if bad_flags == 0 && bad_values == 0 {
            println!(
                "  flags exactly true on the fresh half ({} fresh / {} dup)",
                M - M / 2,
                M / 2
            );
            println!("  values unchanged on dup half, populated on fresh half");
        } else {
            println!("  FAIL: bad_flags={bad_flags} bad_values={bad_values}");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 6: capacity-stress — load to ~75% and round-trip everything.
    //         Probe walks should still terminate cleanly under load.
    // -------------------------------------------------------------------------
    println!("\n--- Test 6: load-factor stress (~75%) ---");
    {
        let map = GpuHashMap::new(CAPACITY, &stream).expect("alloc");
        let m_dense = (CAPACITY * 3) / 4;
        let dense_keys = distinct_keys(m_dense, 0x1234_5678);
        let dense_values: Vec<u32> = (0..m_dense as u32).collect();
        map.insert_bulk(&dense_keys, &dense_values, &module, &stream)
            .expect("dense insert");

        let found = map
            .find_bulk(&dense_keys, &module, &stream)
            .expect("dense find");
        let mut bad = 0usize;
        for i in 0..m_dense {
            if found[i] != dense_values[i] {
                bad += 1;
            }
        }
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

    println!("\n=== SUCCESS: All 6 hashmap v1 tests passed! ===");
}
