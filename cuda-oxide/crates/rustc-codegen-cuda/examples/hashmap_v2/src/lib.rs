/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU Hashmap v2 — SwissTable-Inspired
//!
//! A `u32 -> u32` GPU hashmap that ports the three structural ideas
//! behind hashbrown's SwissTable to a CUDA backend:
//!   - A separate control-byte array (`ctrl: DeviceBuffer<u32>` packing 4
//!     1-byte tags per word) so probe walks examine fingerprints, not the
//!     full `(key, value)` payload.
//!   - hashbrown's h1/h2 hash split — h1 picks the probe position, h2 is a
//!     7-bit per-slot fingerprint stored in the tag.
//!   - Triangular probing in 32-tag-byte tiles (`PROBE_TILE`). Insert,
//!     find, and delete all walk the same triangular sequence so any
//!     key insert places is always reachable by find.
//!   - Tombstone delete (`FULL(h2)` -> `DELETED (0x80)` via `u32` CAS).
//!   - Warp-cooperative find (`find_kernel_warp`) — one warp per key,
//!     32 tag bytes inspected in parallel via `warp::ballot`.
//!
//! Two insert protocols ship side by side so the bench binary can
//! measure them head-to-head:
//!
//!   - **Protocol B (payload-first)** — one `DeviceAtomicU64::
//!     compare_exchange` on the slot followed by one
//!     `DeviceAtomicU32::compare_exchange` on the ctrl word. The slot
//!     CAS is the serialization point. Concurrent inserts of the same
//!     key in the same launch always see each other via `Err(actual)`
//!     and degenerate into the duplicate-handling path.
//!   - **Protocol A (ctrl-first, RESERVED handshake)** — one ctrl-byte
//!     CAS `EMPTY -> RESERVED` claims the slot exclusively, a plain
//!     release store writes the slot payload, then a second ctrl-byte
//!     CAS `RESERVED -> FULL(h2)` publishes the entry. No CAS on the
//!     slot itself. Across kernel launches (stream sync between insert
//!     and any further insert/find), Protocol A is observably
//!     equivalent to Protocol B; within a single launch, two threads
//!     racing to insert the same key may each claim a different
//!     RESERVED byte and end up publishing two slots for that key.
//!     Subsequent finds remain internally consistent (same probe
//!     order, same hit). The duplicate is the documented cost of
//!     skipping the slot CAS.
//!
//! Find observers (single-thread and warp-cooperative) treat the four
//! tag values as: `FULL(h2)` -> peek the slot, `EMPTY` -> terminate
//! probe with MISS, `DELETED` and `RESERVED` -> advance within tile.
//! `RESERVED` does NOT terminate (the slot may be in flight) and does
//! NOT match h2 (its top bit is 1, h2's is 0), so the existing find
//! kernels handle it correctly without special-casing.
//!
//! Library crate: kernels, device-side helpers, and the host-side
//! `GpuSwissMap` driver are defined here so two binaries in the same
//! package can reuse them — `main` (correctness tests) and `bench`
//! (head-to-head perf vs CPU `hashbrown::HashMap`).
//!
//! Build and run the tests with:
//!   cargo oxide run hashmap_v2
//!
//! Run the bench with:
//!   ./crates/rustc-codegen-cuda/examples/hashmap_v2/run-bench.sh

use std::sync::Arc;

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaStream, DeviceBuffer};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32, DeviceAtomicU64};
use cuda_device::cooperative_groups::{ThreadGroup, WarpCollective, this_thread_block};
use cuda_device::{DisjointSlice, kernel, thread, warp};
use cuda_host::cuda_module;

// =============================================================================
// SHARED CONSTANTS AND HELPERS (compiled both host- and device-side)
// =============================================================================

/// Number of tag bytes packed into one `ctrl` word. Fixed at 4 because
/// cuda-oxide has no 8-bit atomics: a single 32-bit CAS is the smallest
/// instruction that can transition one tag while leaving the other three
/// untouched. This is a *storage* constant — it does NOT determine the
/// probe-step width.
pub const GROUP: usize = 4;

/// Probe-step width in tag bytes. Both single-thread and warp-cooperative
/// kernels examine `PROBE_TILE` consecutive tag bytes per probe step and
/// advance triangularly in `PROBE_TILE` units. Fixed at 32 because:
///   - The warp-cooperative kernel uses 32 lanes (one tag byte per lane),
///     so a 32-byte tile maps one-to-one onto cuda-oxide's full-warp
///     `warp::ballot` and `warp::shuffle` primitives.
///   - Insert and find MUST use the same `PROBE_TILE` so they walk the
///     same triangular sequence — otherwise find can terminate early
///     on an `EMPTY` slot that insert had skipped, missing valid keys.
///   - `PROBE_TILE` must be a multiple of `GROUP` (one ctrl word covers
///     `GROUP` tag bytes; we read `PROBE_TILE / GROUP` ctrl words per
///     step).
///
/// Sub-warp `WarpTile<N>` masks via `cuda_device::cooperative_groups`
/// shipped after v2 was frozen; `hashmap_v3`'s `find_kernel_tile_16`
/// uses them for two-keys-per-warp packing.
pub const PROBE_TILE: usize = 32;

/// Tag byte = "this slot is free". All slots start as `EMPTY_TAG`. The
/// initial all-`0xFF` ctrl array gives us this for free via
/// `memset_d8_async(0xFF, ...)`.
pub const EMPTY_TAG: u8 = 0xFF;

/// Tag byte = "this slot was once occupied; do not stop probing here, but
/// also do not treat it as live". Insert in v2 does **not** reclaim
/// these (no `DELETED -> FULL` CAS path); they linger until v3's rehash.
pub const DELETED_TAG: u8 = 0x80;

/// Tag byte = "an insert has claimed this slot's ctrl byte but has not
/// yet published a key-value payload". Used only by Protocol A's
/// `EMPTY -> RESERVED -> FULL(h2)` handshake.
///
/// Top bit is set (0xFE > 0x7F) so it can never collide with a `FULL(h2)`
/// fingerprint. It is also distinct from `EMPTY_TAG (0xFF)` and
/// `DELETED_TAG (0x80)`, so existing find/delete logic (look for h2 to
/// peek, look for EMPTY to terminate) skips RESERVED automatically as
/// "neither — advance".
pub const RESERVED_TAG: u8 = 0xFE;

/// Tag byte for an occupied slot is `FULL(h2)` — the top bit is clear and
/// the low seven bits are `h2`, the high-byte fingerprint of the key's
/// hash. h2 thus lives in `[0x00, 0x7F]` and can never collide with
/// `EMPTY_TAG (0xFF)` or `DELETED_TAG (0x80)`.
///
/// Format:
/// ```text
///   bit:   7   6   5   4   3   2   1   0
///        +---+---+---+---+---+---+---+---+
/// EMPTY  | 1   1   1   1   1   1   1   1 |   0xFF
/// DELETED| 1   0   0   0   0   0   0   0 |   0x80
/// FULL   | 0 |       h2 (7 bits)         |   0x00..0x7F
///        +---+---+---+---+---+---+---+---+
/// ```
///
/// Helper to build a `FULL(h2)` byte; the input is already 7-bit so this is
/// a no-op, but we name it to make intent obvious in the kernel.
#[inline(always)]
pub fn full_tag(h2: u8) -> u8 {
    h2
}

/// Slot sentinel for "this slot is unclaimed" — `u64::MAX`, which packs
/// as `(u32::MAX, u32::MAX)`. The `slots` buffer is `memset` to all-`0xFF`
/// bytes at construction so every slot reads as this.
pub const EMPTY_SLOT: u64 = u64::MAX;

/// Sentinel returned by `find_bulk` for missing keys.
pub const MISS: u32 = u32::MAX;

/// Per-key flag value for "this key was already present" (`try_insert_bulk`)
/// or "this key was not in the table" (`delete_bulk`). The host narrows this
/// to a `bool` before returning.
pub const FLAG_PRESENT: u32 = 1;
/// Per-key flag value for "fresh insert" / "successful delete".
pub const FLAG_FRESH_OR_OK: u32 = 0;

/// FxHash multiplier.
pub const FX_K: u64 = 0x517c_c1b7_2722_0a95;

/// FxHash-style single-multiply hash. Returns 64 bits so we can split into
/// h1 (low bits, probe position) and h2 (top 7 bits, fingerprint) without
/// a second hash call.
#[inline(always)]
pub fn hash_u32(key: u32) -> u64 {
    (key as u64).wrapping_mul(FX_K)
}

/// Extract the 7-bit fingerprint stored in the FULL tag. We pull from the
/// **top** of the hash so it's statistically independent of the low-bit
/// position — the same split hashbrown uses (`raw.rs:60`).
#[inline(always)]
pub fn h2_from_hash(hash: u64) -> u8 {
    ((hash >> 57) & 0x7F) as u8
}

/// Extract the tag byte at index `i` (0..4) from a packed ctrl word.
#[inline(always)]
pub fn get_tag(word: u32, i: usize) -> u8 {
    ((word >> (8 * i)) & 0xFF) as u8
}

/// Replace the tag byte at index `i` (0..4) inside a packed ctrl word.
/// Returns the new word; the other three bytes are preserved.
#[inline(always)]
pub fn set_tag(word: u32, i: usize, tag: u8) -> u32 {
    let shift = 8 * i;
    (word & !(0xFFu32 << shift)) | ((tag as u32) << shift)
}

/// Pack `(key, value)` into a single `u64` slot — key in the upper 32
/// bits, value in the lower 32. The packed sentinel `u64::MAX` matches
/// `(FORBIDDEN_KEY, u32::MAX)` and is reserved for `EMPTY_SLOT`.
#[inline(always)]
pub fn pack(key: u32, value: u32) -> u64 {
    ((key as u64) << 32) | (value as u64)
}

/// Recover the key from a packed slot (upper 32 bits).
#[inline(always)]
pub fn unpack_key(slot: u64) -> u32 {
    (slot >> 32) as u32
}

/// Recover the value from a packed slot (lower 32 bits).
#[inline(always)]
pub fn unpack_value(slot: u64) -> u32 {
    (slot & 0xFFFF_FFFF) as u32
}

// =============================================================================
// KERNELS
// =============================================================================

#[cuda_module]
pub mod kernels {
    use super::*;

    /// `insert_kernel` — last-writer-wins, one thread per input key.
    ///
    /// Storage:
    ///   - `ctrl[g]` is the `u32` packing tags for slots `g*GROUP .. g*GROUP+GROUP`.
    ///   - `slots[s]` is the packed `(key, value)` for slot `s`.
    ///
    /// Probe shape: triangular in `PROBE_TILE`-byte tiles. Each step
    /// examines `PROBE_TILE` consecutive tag bytes (= `PROBE_TILE / GROUP`
    /// ctrl words) starting at `probe_base`, and advances by `stride *
    /// PROBE_TILE` (with `stride += 1` per step). All four kernels — and
    /// the warp-cooperative find — share this exact shape so they walk
    /// the same sequence and EMPTY-termination remains correct.
    ///
    /// Insert protocol (Protocol B, payload-first):
    ///   1. Phase 1 — walk the tile ctrl word by ctrl word looking for any
    ///      `FULL(h2)` tag whose slot holds our key. On match, slot-CAS
    ///      overwrite the value (last-writer-wins) and return.
    ///   2. Phase 2 — re-walk the tile looking for any `EMPTY_TAG` byte.
    ///      For each one, try the slot CAS `EMPTY_SLOT -> pack(k, v)`.
    ///      The slot CAS is the serialization point: concurrent inserts of
    ///      the same key see `Err(actual)` with a matching key and
    ///      degenerate to the overwrite path; concurrent inserts of
    ///      *different* keys see `Err(actual)` with a mismatched key and
    ///      skip past.
    ///   3. After a successful slot CAS, publish via a ctrl-word CAS retry
    ///      loop: `set_tag(current_word, j, FULL(h2))` on the specific
    ///      ctrl word containing byte `j`. Other bytes in that word may
    ///      have changed concurrently, so the loop re-reads on failure;
    ///      byte `j` itself cannot change under us because no other thread
    ///      can claim a slot we already own.
    ///   4. No FULL(h2) match anywhere in the tile and no EMPTY_TAG to
    ///      claim → triangular advance and repeat.
    #[kernel]
    pub fn insert_kernel(
        ctrl: &[DeviceAtomicU32],
        slots: &[DeviceAtomicU64],
        keys: &[u32],
        values: &[u32],
    ) {
        let tid = thread::index_1d().get();
        if tid >= keys.len() {
            return;
        }

        let key = keys[tid];
        let value = values[tid];
        let hash = hash_u32(key);
        let h2 = h2_from_hash(hash);
        let mask = slots.len() - 1;
        let mut probe_base = (hash as usize) & mask & !(PROBE_TILE - 1);
        let mut stride = 0usize;

        loop {
            // Phase 1: walk the entire 32-byte tile, ctrl word by ctrl word,
            // checking for an already-published FULL(h2) entry holding our key.
            let mut g = 0usize;
            while g < PROBE_TILE {
                let ctrl_word_idx = (probe_base + g) / GROUP;
                let ctrl_atomic = &ctrl[ctrl_word_idx];
                let word = ctrl_atomic.load(AtomicOrdering::Acquire);
                let mut j = 0;
                while j < GROUP {
                    if get_tag(word, j) == h2 {
                        let slot_idx = probe_base + g + j;
                        let slot_atomic = &slots[slot_idx];
                        let observed = slot_atomic.load(AtomicOrdering::Acquire);
                        if unpack_key(observed) == key {
                            insert_overwrite(slot_atomic, observed, key, value);
                            return;
                        }
                    }
                    j += 1;
                }
                g += GROUP;
            }

            // Phase 2: try to claim an EMPTY tag anywhere in this tile.
            let mut g = 0usize;
            while g < PROBE_TILE {
                let ctrl_word_idx = (probe_base + g) / GROUP;
                let ctrl_atomic = &ctrl[ctrl_word_idx];
                // Re-read this word: another thread may have mutated it
                // between Phase 1 and now.
                let word = ctrl_atomic.load(AtomicOrdering::Acquire);
                let mut j = 0;
                while j < GROUP {
                    if get_tag(word, j) == EMPTY_TAG {
                        let slot_idx = probe_base + g + j;
                        let slot_atomic = &slots[slot_idx];
                        match slot_atomic.compare_exchange(
                            EMPTY_SLOT,
                            pack(key, value),
                            AtomicOrdering::AcqRel,
                            AtomicOrdering::Relaxed,
                        ) {
                            Ok(_) => {
                                publish_full_tag(ctrl_atomic, word, j, h2);
                                return;
                            }
                            Err(actual) => {
                                if unpack_key(actual) == key {
                                    insert_overwrite(slot_atomic, actual, key, value);
                                    return;
                                }
                                // Different key already in this slot; skip.
                            }
                        }
                    }
                    j += 1;
                }
                g += GROUP;
            }

            // No FULL(h2) match and no claimable EMPTY in this tile: advance.
            stride += 1;
            probe_base = (probe_base + stride * PROBE_TILE) & mask;
        }
    }

    /// `try_insert_kernel` — first-writer-wins variant.
    ///
    /// Same probe / claim shape as `insert_kernel`, but on duplicate it leaves
    /// the existing slot untouched and writes per-thread output:
    ///   `out[tid] = FLAG_FRESH_OR_OK (0)`  -> we claimed a fresh slot
    ///   `out[tid] = FLAG_PRESENT (1)`      -> key was already in the table
    #[kernel]
    pub fn try_insert_kernel(
        ctrl: &[DeviceAtomicU32],
        slots: &[DeviceAtomicU64],
        keys: &[u32],
        values: &[u32],
        mut out: DisjointSlice<u32>,
    ) {
        let tid = thread::index_1d();
        let tid_raw = tid.get();
        let i_thread = tid_raw;
        if i_thread >= keys.len() {
            return;
        }

        let key = keys[i_thread];
        let value = values[i_thread];
        let hash = hash_u32(key);
        let h2 = h2_from_hash(hash);
        let mask = slots.len() - 1;
        let mut probe_base = (hash as usize) & mask & !(PROBE_TILE - 1);
        let mut stride = 0usize;

        loop {
            // Phase 1: published-FULL duplicate detection across the tile.
            let mut g = 0usize;
            while g < PROBE_TILE {
                let ctrl_word_idx = (probe_base + g) / GROUP;
                let ctrl_atomic = &ctrl[ctrl_word_idx];
                let word = ctrl_atomic.load(AtomicOrdering::Acquire);
                let mut j = 0;
                while j < GROUP {
                    if get_tag(word, j) == h2 {
                        let slot_idx = probe_base + g + j;
                        let slot_atomic = &slots[slot_idx];
                        let observed = slot_atomic.load(AtomicOrdering::Acquire);
                        if unpack_key(observed) == key {
                            if let Some(o) = out.get_mut(tid) {
                                *o = FLAG_PRESENT;
                            }
                            return;
                        }
                    }
                    j += 1;
                }
                g += GROUP;
            }

            // Phase 2: claim an EMPTY slot.
            let mut g = 0usize;
            while g < PROBE_TILE {
                let ctrl_word_idx = (probe_base + g) / GROUP;
                let ctrl_atomic = &ctrl[ctrl_word_idx];
                let word = ctrl_atomic.load(AtomicOrdering::Acquire);
                let mut j = 0;
                while j < GROUP {
                    if get_tag(word, j) == EMPTY_TAG {
                        let slot_idx = probe_base + g + j;
                        let slot_atomic = &slots[slot_idx];
                        match slot_atomic.compare_exchange(
                            EMPTY_SLOT,
                            pack(key, value),
                            AtomicOrdering::AcqRel,
                            AtomicOrdering::Relaxed,
                        ) {
                            Ok(_) => {
                                publish_full_tag(ctrl_atomic, word, j, h2);
                                if let Some(o) = out.get_mut(tid) {
                                    *o = FLAG_FRESH_OR_OK;
                                }
                                return;
                            }
                            Err(actual) => {
                                if unpack_key(actual) == key {
                                    // Same-key race; some other thread is the
                                    // first-writer. Report PRESENT, leave slot.
                                    if let Some(o) = out.get_mut(tid) {
                                        *o = FLAG_PRESENT;
                                    }
                                    return;
                                }
                            }
                        }
                    }
                    j += 1;
                }
                g += GROUP;
            }

            stride += 1;
            probe_base = (probe_base + stride * PROBE_TILE) & mask;
        }
    }

    /// `insert_kernel_proto_a` — last-writer-wins, Protocol A
    /// (ctrl-first / RESERVED handshake), one thread per input key.
    ///
    /// Probe shape is identical to Protocol B (`PROBE_TILE = 32`,
    /// triangular advance). The protocol differs only in how a slot is
    /// claimed:
    ///   1. **Phase 1** — walk the tile looking for `FULL(h2)` whose slot
    ///      already holds our key; on match, take the same overwrite path
    ///      as Protocol B (`insert_overwrite`).
    ///   2. **Phase 2 — handshake claim**. For each `EMPTY_TAG` byte in
    ///      the tile, attempt a ctrl-word CAS `EMPTY_TAG -> RESERVED_TAG`
    ///      at byte `j`. On success: the slot is exclusively ours.
    ///   3. **Plain release store** of `pack(k, v)` into the slot — no
    ///      slot CAS needed; observers can't peek a RESERVED slot, and no
    ///      other insert can claim this byte.
    ///   4. **Publish** with a ctrl-word CAS retry loop
    ///      `RESERVED_TAG -> FULL(h2)` at byte `j`. Same `publish_full_tag`
    ///      helper as Protocol B; byte `j` is owned by us so it cannot
    ///      change under the loop.
    ///
    /// On a CAS collision in step 2 (someone mutated a different byte of
    /// the same word), re-read the word and re-scan it before moving on.
    ///
    /// Duplicate-key races within a single launch: two threads inserting
    /// the same key in the same launch may each win the handshake at
    /// different bytes and publish two slots holding that key. The
    /// resulting state is internally consistent (every find walks the
    /// probe in the same order and returns the first published slot it
    /// hits), but it is NOT single-occurrence. Across kernel launches
    /// the stream-sync release-acquire boundary publishes earlier inserts
    /// before Phase 1 of a later launch reads, so cross-launch dedup is
    /// correct.
    #[kernel]
    pub fn insert_kernel_proto_a(
        ctrl: &[DeviceAtomicU32],
        slots: &[DeviceAtomicU64],
        keys: &[u32],
        values: &[u32],
    ) {
        let tid = thread::index_1d().get();
        if tid >= keys.len() {
            return;
        }

        let key = keys[tid];
        let value = values[tid];
        let hash = hash_u32(key);
        let h2 = h2_from_hash(hash);
        let mask = slots.len() - 1;
        let mut probe_base = (hash as usize) & mask & !(PROBE_TILE - 1);
        let mut stride = 0usize;

        loop {
            // Phase 1: scan the tile for an already-published FULL(h2)
            // entry whose slot key matches ours. If found, take the
            // overwrite path. RESERVED bytes naturally fall through here
            // (top bit set, won't equal h2).
            let mut g = 0usize;
            while g < PROBE_TILE {
                let ctrl_word_idx = (probe_base + g) / GROUP;
                let ctrl_atomic = &ctrl[ctrl_word_idx];
                let word = ctrl_atomic.load(AtomicOrdering::Acquire);
                let mut j = 0;
                while j < GROUP {
                    if get_tag(word, j) == h2 {
                        let slot_idx = probe_base + g + j;
                        let slot_atomic = &slots[slot_idx];
                        let observed = slot_atomic.load(AtomicOrdering::Acquire);
                        if unpack_key(observed) == key {
                            insert_overwrite(slot_atomic, observed, key, value);
                            return;
                        }
                    }
                    j += 1;
                }
                g += GROUP;
            }

            // Phase 2: walk the tile word by word; for each word, try to
            // claim the first EMPTY byte via EMPTY_TAG -> RESERVED_TAG CAS.
            // CAS collisions on a different byte of the same word force a
            // re-read of that word.
            let mut g = 0usize;
            while g < PROBE_TILE {
                let ctrl_word_idx = (probe_base + g) / GROUP;
                let ctrl_atomic = &ctrl[ctrl_word_idx];
                'word: loop {
                    let word = ctrl_atomic.load(AtomicOrdering::Acquire);
                    let mut j = 0;
                    while j < GROUP {
                        if get_tag(word, j) == EMPTY_TAG {
                            let claimed = set_tag(word, j, RESERVED_TAG);
                            match ctrl_atomic.compare_exchange(
                                word,
                                claimed,
                                AtomicOrdering::AcqRel,
                                AtomicOrdering::Relaxed,
                            ) {
                                Ok(_) => {
                                    // We exclusively own slot (probe_base + g + j).
                                    // Plain release store: no observer can load
                                    // the slot until we publish FULL(h2), and no
                                    // concurrent inserter can race the byte.
                                    let slot_idx = probe_base + g + j;
                                    let slot_atomic = &slots[slot_idx];
                                    slot_atomic.store(pack(key, value), AtomicOrdering::Release);
                                    publish_full_tag(ctrl_atomic, claimed, j, h2);
                                    return;
                                }
                                Err(_) => {
                                    continue 'word;
                                }
                            }
                        }
                        j += 1;
                    }
                    // Word fully scanned, no EMPTY remains. Move on.
                    break 'word;
                }
                g += GROUP;
            }

            stride += 1;
            probe_base = (probe_base + stride * PROBE_TILE) & mask;
        }
    }

    /// `try_insert_kernel_proto_a` — first-writer-wins, Protocol A.
    ///
    /// Same probe / claim shape as `insert_kernel_proto_a`. Differences:
    ///   - On Phase 1 hit (key already published), writes
    ///     `FLAG_PRESENT (1)` and returns without touching the slot.
    ///   - On successful handshake claim and publish, writes
    ///     `FLAG_FRESH_OR_OK (0)`.
    ///
    /// Same single-launch duplicate-key caveat as `insert_kernel_proto_a`:
    /// two threads racing on the same key in the same launch may each
    /// publish a slot. The first slot encountered in probe order then
    /// shadows the second; both threads will report `FRESH` (each thinks
    /// it claimed first). Use `insert_kernel` (Protocol B) when strict
    /// single-launch first-writer dedup is required.
    #[kernel]
    pub fn try_insert_kernel_proto_a(
        ctrl: &[DeviceAtomicU32],
        slots: &[DeviceAtomicU64],
        keys: &[u32],
        values: &[u32],
        mut out: DisjointSlice<u32>,
    ) {
        let tid = thread::index_1d();
        let tid_raw = tid.get();
        let i_thread = tid_raw;
        if i_thread >= keys.len() {
            return;
        }

        let key = keys[i_thread];
        let value = values[i_thread];
        let hash = hash_u32(key);
        let h2 = h2_from_hash(hash);
        let mask = slots.len() - 1;
        let mut probe_base = (hash as usize) & mask & !(PROBE_TILE - 1);
        let mut stride = 0usize;

        loop {
            // Phase 1: published-FULL duplicate detection across the tile.
            let mut g = 0usize;
            while g < PROBE_TILE {
                let ctrl_word_idx = (probe_base + g) / GROUP;
                let ctrl_atomic = &ctrl[ctrl_word_idx];
                let word = ctrl_atomic.load(AtomicOrdering::Acquire);
                let mut j = 0;
                while j < GROUP {
                    if get_tag(word, j) == h2 {
                        let slot_idx = probe_base + g + j;
                        let slot_atomic = &slots[slot_idx];
                        let observed = slot_atomic.load(AtomicOrdering::Acquire);
                        if unpack_key(observed) == key {
                            if let Some(o) = out.get_mut(tid) {
                                *o = FLAG_PRESENT;
                            }
                            return;
                        }
                    }
                    j += 1;
                }
                g += GROUP;
            }

            // Phase 2: handshake claim with per-word retry.
            let mut g = 0usize;
            while g < PROBE_TILE {
                let ctrl_word_idx = (probe_base + g) / GROUP;
                let ctrl_atomic = &ctrl[ctrl_word_idx];
                'word: loop {
                    let word = ctrl_atomic.load(AtomicOrdering::Acquire);
                    let mut j = 0;
                    while j < GROUP {
                        if get_tag(word, j) == EMPTY_TAG {
                            let claimed = set_tag(word, j, RESERVED_TAG);
                            match ctrl_atomic.compare_exchange(
                                word,
                                claimed,
                                AtomicOrdering::AcqRel,
                                AtomicOrdering::Relaxed,
                            ) {
                                Ok(_) => {
                                    let slot_idx = probe_base + g + j;
                                    let slot_atomic = &slots[slot_idx];
                                    slot_atomic.store(pack(key, value), AtomicOrdering::Release);
                                    publish_full_tag(ctrl_atomic, claimed, j, h2);
                                    if let Some(o) = out.get_mut(tid) {
                                        *o = FLAG_FRESH_OR_OK;
                                    }
                                    return;
                                }
                                Err(_) => continue 'word,
                            }
                        }
                        j += 1;
                    }
                    break 'word;
                }
                g += GROUP;
            }

            stride += 1;
            probe_base = (probe_base + stride * PROBE_TILE) & mask;
        }
    }

    /// `find_kernel` — single-thread find, one thread per key.
    ///
    /// Walks the same triangular probe sequence as the insert kernels (same
    /// `PROBE_TILE = 32` width), so EMPTY-termination is sound — see
    /// `find_kernel_warp` below for why probe-width coherence matters.
    ///
    /// At each tile (32 consecutive tag bytes):
    ///   - For every byte tagged `FULL(h2)` matching our key's fingerprint,
    ///     load the slot and key-compare; on match return the value.
    ///   - If any byte in the tile is `EMPTY_TAG`, the key cannot live past
    ///     this point in its triangular chain (insert would have stopped
    ///     at this same EMPTY), so return `MISS`.
    ///   - Otherwise (tile holds only FULL-mismatch + DELETED), triangular
    ///     advance and repeat. DELETED never terminates find.
    #[kernel]
    pub fn find_kernel(
        ctrl: &[DeviceAtomicU32],
        slots: &[DeviceAtomicU64],
        keys: &[u32],
        mut out: DisjointSlice<u32>,
    ) {
        let tid = thread::index_1d();
        let tid_raw = tid.get();
        let i_thread = tid_raw;
        if i_thread >= keys.len() {
            return;
        }

        let key = keys[i_thread];
        let hash = hash_u32(key);
        let h2 = h2_from_hash(hash);
        let mask = slots.len() - 1;
        let mut probe_base = (hash as usize) & mask & !(PROBE_TILE - 1);
        let mut stride = 0usize;

        loop {
            let mut g = 0usize;
            let mut has_empty = false;
            while g < PROBE_TILE {
                let ctrl_word_idx = (probe_base + g) / GROUP;
                let ctrl_atomic = &ctrl[ctrl_word_idx];
                let word = ctrl_atomic.load(AtomicOrdering::Acquire);
                let mut j = 0;
                while j < GROUP {
                    let tag = get_tag(word, j);
                    if tag == h2 {
                        let slot_idx = probe_base + g + j;
                        let slot_atomic = &slots[slot_idx];
                        let observed = slot_atomic.load(AtomicOrdering::Acquire);
                        if unpack_key(observed) == key {
                            if let Some(o) = out.get_mut(tid) {
                                *o = unpack_value(observed);
                            }
                            return;
                        }
                    } else if tag == EMPTY_TAG {
                        has_empty = true;
                    }
                    j += 1;
                }
                g += GROUP;
            }

            if has_empty {
                if let Some(o) = out.get_mut(tid) {
                    *o = MISS;
                }
                return;
            }

            stride += 1;
            probe_base = (probe_base + stride * PROBE_TILE) & mask;
        }
    }

    /// `find_kernel_warp` — warp-cooperative find, one warp (32 lanes) per key.
    ///
    /// Each probe step pulls 32 tag bytes (= `PROBE_TILE`) into the warp via a
    /// single coalesced ctrl load: lanes 0..3 share `ctrl[probe_base/4 + 0]`,
    /// lanes 4..7 share `+1`, ..., lanes 28..31 share `+7`. Each lane then
    /// extracts its byte and the warp uses ballot/shuffle to:
    ///   1. `m_h2 = ballot(tag == h2)` — a 32-bit fingerprint match mask.
    ///   2. For each set bit in `m_h2` (lowest first), the matching lane
    ///      loads its slot, broadcasts the packed `(key, value)` via two
    ///      `shuffle`s (one per `u32` half), and all lanes key-compare.
    ///      On hit, lane 0 writes `out[warp_idx] = value` and the warp returns.
    ///   3. `m_empty = ballot(tag == EMPTY_TAG)` — if non-zero, the key
    ///      cannot live past an EMPTY in this hash chain; lane 0 writes MISS.
    ///   4. Otherwise, triangular advance `probe_base` by `stride * PROBE_TILE`
    ///      and repeat.
    ///
    /// Launch with `blockDim.x` a multiple of 32; the host driver does
    /// `LaunchConfig::for_num_elems(keys.len() * 32)` so each warp receives
    /// exactly one key. DELETED tags are skipped (they don't terminate the
    /// probe), same as the single-thread `find_kernel`.
    #[kernel]
    pub fn find_kernel_warp(
        ctrl: &[DeviceAtomicU32],
        slots: &[DeviceAtomicU64],
        keys: &[u32],
        mut out: DisjointSlice<u32>,
    ) {
        let lane = warp::lane_id();
        let global_tid = thread::index_1d().get();
        let warp_idx = global_tid / PROBE_TILE;
        if warp_idx >= keys.len() {
            return;
        }

        let key = keys[warp_idx];
        let hash = hash_u32(key);
        let h2 = h2_from_hash(hash);
        let mask = slots.len() - 1;
        let mut probe_base = (hash as usize) & mask & !(PROBE_TILE - 1);
        let mut stride = 0usize;

        loop {
            // Each lane owns exactly one tag byte at (probe_base + lane).
            let tag_pos = probe_base + (lane as usize);
            let ctrl_word_idx = tag_pos / GROUP;
            let byte_in_word = tag_pos % GROUP;
            let word = ctrl[ctrl_word_idx].load(AtomicOrdering::Acquire);
            let tag: u8 = ((word >> (8 * byte_in_word)) & 0xFF) as u8;

            let mut m_h2 = warp::ballot(tag == h2);
            let m_empty = warp::ballot(tag == EMPTY_TAG);

            // Walk h2 candidates lowest-bit-first.
            while m_h2 != 0 {
                let cand = m_h2.trailing_zeros();
                // Only the candidate lane materializes the slot; others feed
                // a dummy value into shuffle. The shuffle's source is `cand`
                // so all lanes converge on the candidate's slot value.
                let local_slot: u64 = if lane == cand {
                    let slot_idx = probe_base + (cand as usize);
                    slots[slot_idx].load(AtomicOrdering::Acquire)
                } else {
                    0
                };
                // cuda-oxide's warp::shuffle is u32-only, so broadcast the
                // packed slot in two halves and reassemble.
                let lo = warp::shuffle(local_slot as u32, cand);
                let hi = warp::shuffle((local_slot >> 32) as u32, cand);
                let observed: u64 = ((hi as u64) << 32) | (lo as u64);

                if unpack_key(observed) == key {
                    if lane == 0 {
                        // SAFETY: warp_idx < keys.len() == out.len(), and
                        // each warp has a unique warp_idx so writes by lane
                        // 0 across warps are disjoint.
                        unsafe {
                            *out.get_unchecked_mut(warp_idx) = unpack_value(observed);
                        }
                    }
                    return;
                }
                m_h2 &= m_h2 - 1;
            }

            if m_empty != 0 {
                if lane == 0 {
                    // SAFETY: same uniqueness argument as above.
                    unsafe {
                        *out.get_unchecked_mut(warp_idx) = MISS;
                    }
                }
                return;
            }

            // Triangular advance in tile units. Both operands are
            // PROBE_TILE-aligned, so the sum stays aligned and `& mask`
            // keeps it in [0, N).
            stride += 1;
            probe_base = (probe_base + stride * PROBE_TILE) & mask;
        }
    }

    /// `find_kernel_warp_typed` — `find_kernel_warp` rewritten on the typed
    /// cooperative-groups API.
    ///
    /// Identical algorithm to `find_kernel_warp`; the only differences are:
    ///
    /// - `warp::lane_id()`               -> `tile.thread_rank()`
    /// - `warp::ballot(p)`               -> `tile.ballot(p)`
    /// - `warp::shuffle(v, src)`         -> `tile.shfl(v, src)`
    ///
    /// where `tile = this_thread_block().tiled_partition::<32>()`.
    ///
    /// The emitted *instruction* under each typed call is byte-identical
    /// to the raw form (one `vote.sync.ballot.b32` or one
    /// `shfl.sync.idx.b32`), but at the time of writing each typed
    /// wrapper is large enough to clear rustc's MIR `Inline` cost
    /// threshold and ends up as its own `.visible .func`. The four typed
    /// sites in the inner probe loop (`tile.ballot ×2`, `tile.shfl ×2`)
    /// therefore add four `call.uni` round-trips per probe step, which
    /// the bench binary measures at ~12–17 % runtime overhead vs the
    /// raw kernel.
    #[kernel]
    pub fn find_kernel_warp_typed(
        ctrl: &[DeviceAtomicU32],
        slots: &[DeviceAtomicU64],
        keys: &[u32],
        mut out: DisjointSlice<u32>,
    ) {
        let block = this_thread_block();
        let tile = block.tiled_partition::<32>();

        let lane = tile.thread_rank();
        let global_tid = thread::index_1d().get();
        let warp_idx = global_tid / PROBE_TILE;
        if warp_idx >= keys.len() {
            return;
        }

        let key = keys[warp_idx];
        let hash = hash_u32(key);
        let h2 = h2_from_hash(hash);
        let mask = slots.len() - 1;
        let mut probe_base = (hash as usize) & mask & !(PROBE_TILE - 1);
        let mut stride = 0usize;

        loop {
            let tag_pos = probe_base + (lane as usize);
            let ctrl_word_idx = tag_pos / GROUP;
            let byte_in_word = tag_pos % GROUP;
            let word = ctrl[ctrl_word_idx].load(AtomicOrdering::Acquire);
            let tag: u8 = ((word >> (8 * byte_in_word)) & 0xFF) as u8;

            let mut m_h2 = tile.ballot(tag == h2);
            let m_empty = tile.ballot(tag == EMPTY_TAG);

            while m_h2 != 0 {
                let cand = m_h2.trailing_zeros();
                let local_slot: u64 = if lane == cand {
                    let slot_idx = probe_base + (cand as usize);
                    slots[slot_idx].load(AtomicOrdering::Acquire)
                } else {
                    0
                };
                let lo = tile.shfl(local_slot as u32, cand);
                let hi = tile.shfl((local_slot >> 32) as u32, cand);
                let observed: u64 = ((hi as u64) << 32) | (lo as u64);

                if unpack_key(observed) == key {
                    if lane == 0 {
                        // SAFETY: warp_idx < keys.len() == out.len(), and
                        // each warp has a unique warp_idx so writes by lane
                        // 0 across warps are disjoint.
                        unsafe {
                            *out.get_unchecked_mut(warp_idx) = unpack_value(observed);
                        }
                    }
                    return;
                }
                m_h2 &= m_h2 - 1;
            }

            if m_empty != 0 {
                if lane == 0 {
                    // SAFETY: same uniqueness argument as above.
                    unsafe {
                        *out.get_unchecked_mut(warp_idx) = MISS;
                    }
                }
                return;
            }

            stride += 1;
            probe_base = (probe_base + stride * PROBE_TILE) & mask;
        }
    }

    /// `delete_kernel` — tombstone the slot for each input key.
    ///
    /// One thread per key. Probes the same `PROBE_TILE`-wide triangular
    /// sequence as insert and find. When it locates the key it CAS-flips
    /// the byte in the containing ctrl word from `FULL(h2)` to
    /// `DELETED_TAG`. The `(key, value)` payload is **not** cleared:
    /// readers only ever materialize slots whose tag is `FULL(h2)`, so a
    /// stale slot under a `DELETED` tag is unreachable.
    ///
    /// The CAS targets the specific ctrl word containing the matching tag
    /// byte (not "the group's word" — there are now `PROBE_TILE / GROUP`
    /// ctrl words per tile). On CAS failure (some other thread mutated
    /// this word), the inner loop re-reads and re-scans this word before
    /// moving on to the next word in the tile.
    ///
    /// Output:
    ///   `out[tid] = FLAG_FRESH_OR_OK (0)` -> deleted successfully
    ///   `out[tid] = FLAG_PRESENT (1)`     -> key was not in the table
    #[kernel]
    pub fn delete_kernel(
        ctrl: &[DeviceAtomicU32],
        slots: &[DeviceAtomicU64],
        keys: &[u32],
        mut out: DisjointSlice<u32>,
    ) {
        let tid = thread::index_1d();
        let tid_raw = tid.get();
        let i_thread = tid_raw;
        if i_thread >= keys.len() {
            return;
        }

        let key = keys[i_thread];
        let hash = hash_u32(key);
        let h2 = h2_from_hash(hash);
        let mask = slots.len() - 1;
        let mut probe_base = (hash as usize) & mask & !(PROBE_TILE - 1);
        let mut stride = 0usize;

        'tile: loop {
            let mut has_empty = false;
            let mut g = 0usize;
            while g < PROBE_TILE {
                let ctrl_word_idx = (probe_base + g) / GROUP;
                let ctrl_atomic = &ctrl[ctrl_word_idx];

                // Per-word retry loop: if our CAS to flip a tag to DELETED
                // fails because someone else mutated this same word, re-read
                // and re-scan this word.
                loop {
                    let word = ctrl_atomic.load(AtomicOrdering::Acquire);
                    let mut j = 0;
                    let mut cas_collided = false;
                    while j < GROUP {
                        let tag = get_tag(word, j);
                        if tag == h2 {
                            let slot_idx = probe_base + g + j;
                            let slot_atomic = &slots[slot_idx];
                            let observed = slot_atomic.load(AtomicOrdering::Acquire);
                            if unpack_key(observed) == key {
                                let new_word = set_tag(word, j, DELETED_TAG);
                                match ctrl_atomic.compare_exchange(
                                    word,
                                    new_word,
                                    AtomicOrdering::AcqRel,
                                    AtomicOrdering::Relaxed,
                                ) {
                                    Ok(_) => {
                                        if let Some(o) = out.get_mut(tid) {
                                            *o = FLAG_FRESH_OR_OK;
                                        }
                                        return;
                                    }
                                    Err(_) => {
                                        cas_collided = true;
                                        break;
                                    }
                                }
                            }
                        } else if tag == EMPTY_TAG {
                            has_empty = true;
                        }
                        j += 1;
                    }
                    if !cas_collided {
                        break; // word fully scanned; move to next word in the tile
                    }
                    // else: retry this same word with a freshly-loaded view.
                }
                g += GROUP;
            }

            if has_empty {
                if let Some(o) = out.get_mut(tid) {
                    *o = FLAG_PRESENT;
                }
                return;
            }

            stride += 1;
            probe_base = (probe_base + stride * PROBE_TILE) & mask;
            continue 'tile;
        }
    }
}

// =============================================================================
// DEVICE-SIDE HELPERS
// =============================================================================

/// Inner CAS loop that overwrites a slot's value while preserving its key.
/// Used by both `insert_kernel` and `try_insert_kernel`'s last-writer-wins
/// branches, and only ever entered when the slot already holds our key.
#[inline(always)]
fn insert_overwrite(slot_atomic: &DeviceAtomicU64, mut expected: u64, key: u32, value: u32) {
    let desired = pack(key, value);
    loop {
        match slot_atomic.compare_exchange(
            expected,
            desired,
            AtomicOrdering::AcqRel,
            AtomicOrdering::Relaxed,
        ) {
            Ok(_) => return,
            // Someone else's overwrite landed; re-read and retry so the
            // last-writer-wins guarantee survives concurrent duplicates.
            Err(actual) => expected = actual,
        }
    }
}

/// Publish a tag byte to `FULL(h2)` via a ctrl-word CAS retry loop. The
/// slot at byte `i` is already exclusively ours — either via a winning
/// slot CAS (Protocol B, byte transitions `EMPTY -> FULL`) or via a
/// winning ctrl-byte CAS to RESERVED (Protocol A, byte transitions
/// `RESERVED -> FULL`). In both cases byte `i` cannot change under us,
/// so the only reason this CAS fails is concurrent mutation of a
/// *different* byte in the same word, in which case we re-read and
/// rebuild the new word.
#[inline(always)]
fn publish_full_tag(ctrl_atomic: &DeviceAtomicU32, mut current_word: u32, i: usize, h2: u8) {
    loop {
        let new_word = set_tag(current_word, i, full_tag(h2));
        match ctrl_atomic.compare_exchange(
            current_word,
            new_word,
            AtomicOrdering::Release,
            AtomicOrdering::Relaxed,
        ) {
            Ok(_) => return,
            Err(actual) => current_word = actual,
        }
    }
}

// =============================================================================
// HOST DRIVER
// =============================================================================

/// Forbidden user key. `(u32::MAX, u32::MAX)` collides with the
/// `EMPTY_SLOT` sentinel, so the simplest invariant is to forbid
/// `u32::MAX` as a key outright.
pub const FORBIDDEN_KEY: u32 = u32::MAX;

/// Host-side handle to a v2 SwissTable-style GPU hashmap.
///
/// Owns two device-resident buffers:
///   - `ctrl`: `DeviceBuffer<u32>` of length `capacity / GROUP`. Each `u32`
///     holds 4 tag bytes packed little-endian.
///   - `slots`: `DeviceBuffer<u64>` of length `capacity`. Each `u64` packs
///     `(key, value)` with the key in the upper 32 bits.
///
/// Both buffers are `memset_d8_async(0xFF)` at construction so every tag
/// reads as `EMPTY_TAG` and every slot reads as `EMPTY_SLOT`.
pub struct GpuSwissMap {
    /// Packed tag bytes; 4 tags per `u32` word. Typed as
    /// `DeviceBuffer<DeviceAtomicU32>` so the kernels can take a
    /// `&[DeviceAtomicU32]` and mutate it soundly (issue #151:
    /// mutating through a shared readonly `&[u32]` is UB).
    pub ctrl: DeviceBuffer<DeviceAtomicU32>,
    /// Packed `(key, value)` slots — key in the upper 32 bits. Typed as
    /// `DeviceBuffer<DeviceAtomicU64>` for the same reason as `ctrl`.
    pub slots: DeviceBuffer<DeviceAtomicU64>,
    /// Number of slots. Power of two, multiple of `GROUP`.
    capacity: usize,
}

impl GpuSwissMap {
    /// Allocate a fresh, empty table of `capacity` slots. `capacity` must
    /// be a non-zero power of two and at least `GROUP` (so the ctrl array
    /// has at least one word).
    pub fn new(capacity: usize, stream: &Arc<CudaStream>) -> Result<Self, cuda_core::DriverError> {
        assert!(
            capacity.is_power_of_two(),
            "capacity must be a power of two"
        );
        assert!(
            capacity >= PROBE_TILE,
            "capacity must be >= PROBE_TILE ({PROBE_TILE})"
        );

        let ctrl = DeviceBuffer::<u32>::zeroed(stream, capacity / GROUP)?;
        let slots = DeviceBuffer::<u64>::zeroed(stream, capacity)?;
        unsafe {
            cuda_core::simt::memory::memset_d8_async(
                ctrl.cu_deviceptr(),
                0xFF,
                ctrl.num_bytes(),
                stream.cu_stream(),
            )?;
            cuda_core::simt::memory::memset_d8_async(
                slots.cu_deviceptr(),
                0xFF,
                slots.num_bytes(),
                stream.cu_stream(),
            )?;
        }

        // Reinterpret the allocations as their atomic element types
        // (`#[repr(transparent)]`, identical layout). The kernels mutate
        // these buffers through shared `&[DeviceAtomic*]` slices; a plain
        // `&[u32]` / `&[u64]` would be readonly+noalias and writing
        // through it is UB (issue #151). The memset above is byte-level so
        // it stays valid across the cast.
        let ctrl = ctrl.cast_elem::<DeviceAtomicU32>();
        let slots = slots.cast_elem::<DeviceAtomicU64>();

        Ok(Self {
            ctrl,
            slots,
            capacity,
        })
    }

    /// Number of slots in the table. Fixed at construction time.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Last-writer-wins bulk insert. Overwrites existing values.
    pub fn insert_bulk(
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
            "u32::MAX is reserved and may not be used as a key"
        );

        let keys_dev = DeviceBuffer::from_host(stream, keys)?;
        let values_dev = DeviceBuffer::from_host(stream, values)?;

        let cfg = LaunchConfig::for_num_elems(keys.len() as u32);
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.insert_kernel(stream, cfg, &self.ctrl, &self.slots, &keys_dev, &values_dev)
        }?;

        Ok(())
    }

    /// First-writer-wins bulk insert. Returns a `Vec<bool>` of length
    /// `keys.len()`; `true` means the key was fresh (and the table now
    /// contains the new value), `false` means the key was already present
    /// (and the table is unchanged for that key).
    pub fn try_insert_bulk(
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
            "u32::MAX is reserved and may not be used as a key"
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
                &self.ctrl,
                &self.slots,
                &keys_dev,
                &values_dev,
                &mut out_dev,
            )
        }?;

        let raw = out_dev.to_host_vec(stream)?;
        Ok(raw.into_iter().map(|x| x == FLAG_FRESH_OR_OK).collect())
    }

    /// Last-writer-wins bulk insert using **Protocol A** (ctrl-first
    /// RESERVED handshake). Same return contract as `insert_bulk`.
    /// See `insert_kernel_proto_a` for the duplicate-key caveat
    /// (cross-launch dedup is correct; same-launch same-key races may
    /// publish multiple slots).
    pub fn insert_bulk_proto_a(
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
            "u32::MAX is reserved and may not be used as a key"
        );

        let keys_dev = DeviceBuffer::from_host(stream, keys)?;
        let values_dev = DeviceBuffer::from_host(stream, values)?;

        let cfg = LaunchConfig::for_num_elems(keys.len() as u32);
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.insert_kernel_proto_a(
                stream,
                cfg,
                &self.ctrl,
                &self.slots,
                &keys_dev,
                &values_dev,
            )
        }?;

        Ok(())
    }

    /// First-writer-wins bulk insert using **Protocol A**. Same return
    /// contract as `try_insert_bulk`. Same caveat as
    /// `insert_bulk_proto_a` for same-launch same-key races.
    pub fn try_insert_bulk_proto_a(
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
            "u32::MAX is reserved and may not be used as a key"
        );

        let keys_dev = DeviceBuffer::from_host(stream, keys)?;
        let values_dev = DeviceBuffer::from_host(stream, values)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(stream, keys.len())?;

        let cfg = LaunchConfig::for_num_elems(keys.len() as u32);
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.try_insert_kernel_proto_a(
                stream,
                cfg,
                &self.ctrl,
                &self.slots,
                &keys_dev,
                &values_dev,
                &mut out_dev,
            )
        }?;

        let raw = out_dev.to_host_vec(stream)?;
        Ok(raw.into_iter().map(|x| x == FLAG_FRESH_OR_OK).collect())
    }

    /// Bulk find. Returns `Vec<u32>` of length `keys.len()`; entries equal
    /// to `MISS = u32::MAX` mean "key not present".
    pub fn find_bulk(
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
        unsafe {
            module.find_kernel(
                stream,
                cfg,
                &self.ctrl,
                &self.slots,
                &keys_dev,
                &mut out_dev,
            )
        }?;

        Ok(out_dev.to_host_vec(stream)?)
    }

    /// Bulk find using the warp-cooperative kernel — one warp (32 lanes)
    /// per query key, 32 tag bytes inspected in parallel per probe step
    /// via `warp::ballot`. Same return contract as `find_bulk`.
    ///
    /// Requires `capacity >= PROBE_TILE = 32`, which holds for any
    /// power-of-two capacity at or above 32.
    pub fn find_bulk_warp(
        &self,
        keys: &[u32],
        module: &kernels::LoadedModule,
        stream: &Arc<CudaStream>,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        debug_assert!(
            self.capacity >= PROBE_TILE,
            "warp-cooperative find requires capacity >= {PROBE_TILE} slots"
        );

        let keys_dev = DeviceBuffer::from_host(stream, keys)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(stream, keys.len())?;

        // One warp per key: launch (keys.len() * 32) threads, block size
        // 256 means 8 warps per block, 8 keys per block.
        let total_threads = (keys.len() as u32).saturating_mul(PROBE_TILE as u32);
        let cfg = LaunchConfig::for_num_elems(total_threads);
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.find_kernel_warp(
                stream,
                cfg,
                &self.ctrl,
                &self.slots,
                &keys_dev,
                &mut out_dev,
            )
        }?;

        Ok(out_dev.to_host_vec(stream)?)
    }

    /// Bulk delete (tombstone). Returns a `Vec<bool>` of length `keys.len()`;
    /// `true` means the key was present and is now tombstoned, `false` means
    /// the key was not in the table.
    pub fn delete_bulk(
        &self,
        keys: &[u32],
        module: &kernels::LoadedModule,
        stream: &Arc<CudaStream>,
    ) -> Result<Vec<bool>, Box<dyn std::error::Error>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }

        let keys_dev = DeviceBuffer::from_host(stream, keys)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(stream, keys.len())?;

        let cfg = LaunchConfig::for_num_elems(keys.len() as u32);
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.delete_kernel(
                stream,
                cfg,
                &self.ctrl,
                &self.slots,
                &keys_dev,
                &mut out_dev,
            )
        }?;

        let raw = out_dev.to_host_vec(stream)?;
        Ok(raw.into_iter().map(|x| x == FLAG_FRESH_OR_OK).collect())
    }
}

// =============================================================================
// SHARED UTILITIES (used by both `main` tests and `bench`)
// =============================================================================

/// Tiny xorshift32 — avoids pulling in a crate just for deterministic
/// pseudo-random keys in tests and benches.
pub fn xorshift32(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

/// Sample `n` distinct keys, all `< u32::MAX`, deterministically seeded.
pub fn distinct_keys(n: usize, seed: u32) -> Vec<u32> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip() {
        for (k, v) in [
            (0u32, 0u32),
            (1, 0xDEAD_BEEF),
            (0xCAFE_BABE, 0x5555_5555),
            (FORBIDDEN_KEY - 1, u32::MAX),
        ] {
            let s = pack(k, v);
            assert_eq!(unpack_key(s), k, "key roundtrip for ({k:#x}, {v:#x})");
            assert_eq!(unpack_value(s), v, "value roundtrip for ({k:#x}, {v:#x})");
        }
    }

    #[test]
    fn empty_slot_matches_forbidden_pair() {
        // The slot sentinel and the forbidden key must agree: this is
        // the invariant that lets `memset_d8(0xFF)` safely initialise
        // the slots buffer to all-EMPTY.
        assert_eq!(EMPTY_SLOT, pack(FORBIDDEN_KEY, u32::MAX));
        assert_eq!(EMPTY_SLOT, u64::MAX);
    }

    #[test]
    fn set_tag_preserves_siblings() {
        let word = 0xAABB_CCDDu32;
        for i in 0..GROUP {
            let updated = set_tag(word, i, 0x42);
            assert_eq!(get_tag(updated, i), 0x42, "byte {i} read-back");
            for j in 0..GROUP {
                if j != i {
                    assert_eq!(
                        get_tag(updated, j),
                        get_tag(word, j),
                        "sibling byte {j} clobbered when writing {i}"
                    );
                }
            }
        }
    }

    #[test]
    fn h2_from_hash_in_range() {
        // h2 must always be representable as a FULL(h2) tag, i.e. fit
        // in 7 bits (top bit clear). Sample a range of inputs.
        let mut state = 0xDEAD_BEEFu32;
        for _ in 0..1024 {
            let key = xorshift32(&mut state);
            let h2 = h2_from_hash(hash_u32(key));
            assert!(h2 <= 0x7F, "h2 out of range for key {key:#x}: {h2:#x}");
        }
    }

    #[test]
    fn tag_namespaces_disjoint() {
        // EMPTY, DELETED, RESERVED, and any FULL(h2) fingerprint must
        // all be distinguishable by tag-byte value alone.
        assert_ne!(EMPTY_TAG, DELETED_TAG);
        assert_ne!(EMPTY_TAG, RESERVED_TAG);
        assert_ne!(DELETED_TAG, RESERVED_TAG);
        for h2 in 0u8..=0x7F {
            let full = full_tag(h2);
            assert_ne!(EMPTY_TAG, full, "EMPTY collides with FULL({h2:#x})");
            assert_ne!(DELETED_TAG, full, "DELETED collides with FULL({h2:#x})");
            assert_ne!(RESERVED_TAG, full, "RESERVED collides with FULL({h2:#x})");
        }
    }

    #[test]
    fn full_tag_is_identity_on_valid_h2() {
        for h2 in 0u8..=0x7F {
            assert_eq!(full_tag(h2), h2);
        }
    }

    #[test]
    fn distinct_keys_excludes_forbidden() {
        let ks = distinct_keys(4096, 0xABCD_1234);
        assert_eq!(ks.len(), 4096);
        let unique: std::collections::HashSet<u32> = ks.iter().copied().collect();
        assert_eq!(unique.len(), ks.len(), "duplicate keys produced");
        assert!(!unique.contains(&FORBIDDEN_KEY));
    }
}
