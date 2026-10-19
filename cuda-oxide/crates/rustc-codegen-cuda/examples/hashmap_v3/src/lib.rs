/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU Hashmap v3 — SwissTable, Cooperative-Groups Edition
//!
//! A `u32 -> u32` GPU hashmap built on the typed cooperative-groups API:
//!   - A separate control-byte array (`ctrl: DeviceBuffer<u32>` packing 4
//!     1-byte tags per word) so probe walks examine fingerprints, not the
//!     full `(key, value)` payload.
//!   - hashbrown's h1/h2 hash split — h1 picks the probe position, h2 is a
//!     7-bit per-slot fingerprint stored in the tag.
//!   - Triangular probing in `PROBE_TILE`-byte tiles. Insert, find,
//!     and delete all walk the same triangular sequence so any key
//!     insert places is always reachable by find.
//!   - Tombstone delete (`FULL(h2)` -> `DELETED (0x80)` via `u32` CAS).
//!   - Warp-cooperative find on the typed cooperative-groups API,
//!     parameterised over a `WarpTile<N>` lane-tile size (`N = 32`
//!     for one query per warp, `N = 16` for two queries per warp).
//!
//! Insert is payload-first when claiming a fresh `EMPTY` slot:
//! `DeviceAtomicU64::compare_exchange` on the slot first, then
//! `DeviceAtomicU32::compare_exchange` to flip the ctrl byte to
//! `FULL(h2)`. The slot CAS is the serialization point —
//! concurrent inserts of the same key in the same launch always see
//! each other via `Err(actual)` and degenerate into the duplicate-
//! handling path.
//!
//! Insert also reclaims `DELETED` slots. While walking Phase 2,
//! insert remembers the first `DELETED` byte it sees in the chain.
//! On hitting `EMPTY` it claims the remembered `DELETED` (if any)
//! via a two-stage handshake: tag-CAS `DELETED -> RESERVED` to
//! exclude other claimers, plain release-store the new
//! `(key, value)` payload over the stale data, then tag-CAS
//! `RESERVED -> FULL(h2)` to publish. The reclaim slot lies before
//! the chain's first `EMPTY`, so find's `EMPTY`-termination invariant
//! is preserved (find walks the chain in order, hits the
//! now-`FULL` reclaimed slot before any `EMPTY` further down).
//!
//! Find observers treat the four tag values as: `FULL(h2)` -> peek
//! the slot, `EMPTY` -> terminate probe with MISS, `DELETED` and
//! `RESERVED` -> advance within tile. `RESERVED` (top bit set,
//! distinct from `EMPTY` and `DELETED`) does NOT terminate find and
//! does NOT match h2, so the find kernels need no special-casing.
//!
//! Library crate: kernels, device-side helpers, and the host-side
//! `GpuSwissMap` driver are defined here so two binaries in the same
//! package can reuse them — `main` (correctness tests) and `bench`
//! (head-to-head perf vs CPU `hashbrown::HashMap`).
//!
//! Build and run the tests with:
//!   cargo oxide run hashmap_v3
//!
//! Run the bench with:
//!   ./crates/rustc-codegen-cuda/examples/hashmap_v3/run-bench.sh

use std::sync::Arc;

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaStream, DeviceBuffer};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32, DeviceAtomicU64};
use cuda_device::cooperative_groups::{ThreadGroup, WarpCollective, this_thread_block};
use cuda_device::{DisjointSlice, kernel, thread};
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
///   - The full-warp insert and find kernels use 32 lanes (one tag byte
///     per lane), so a 32-byte tile maps one-to-one onto the warp.
///   - Insert and find MUST share `PROBE_TILE` so they walk the same
///     triangular sequence — otherwise find can terminate early on an
///     `EMPTY` slot that insert had skipped, missing valid keys.
///   - `PROBE_TILE` must be a multiple of `GROUP` (one ctrl word covers
///     `GROUP` tag bytes; we read `PROBE_TILE / GROUP` ctrl words per
///     step).
///
/// `find_kernel_tile_16` uses a 16-lane `WarpTile` but still walks the
/// same probe sequence: each 32-byte insert tile is scanned as **two**
/// consecutive 16-byte sub-tiles before the triangular advance, so the
/// same table is queryable by either tile size.
pub const PROBE_TILE: usize = 32;

/// Maximum number of threads used by the non-cooperative rehash launch.
///
/// Resize walks the old table with a grid-stride loop, so correctness does
/// not depend on launching one thread per slot. Keeping the launch bounded
/// avoids cooperative-launch resident-block limits while still giving enough
/// parallelism for large tables.
pub const REHASH_MAX_THREADS: usize = 65_536;

/// Tag byte = "this slot is free". All slots start as `EMPTY_TAG`. The
/// initial all-`0xFF` ctrl array gives us this for free via
/// `memset_d8_async(0xFF, ...)`.
pub const EMPTY_TAG: u8 = 0xFF;

/// Tag byte = "this slot was once occupied; do not stop probing here,
/// but also do not treat it as live". Insert reclaims these slots
/// when the probe chain it is walking has a `DELETED` byte before
/// the chain's first `EMPTY`; otherwise they linger until the next
/// rehash.
pub const DELETED_TAG: u8 = 0x80;

/// Tag byte = "an insert has claimed this slot's ctrl byte and is
/// about to publish a fresh payload". Used only by the
/// `DELETED -> RESERVED -> FULL(h2)` reclaim handshake — the regular
/// `EMPTY -> FULL(h2)` claim path goes through the slot CAS first
/// and never visits this state.
///
/// Top bit is set (`0xFE > 0x7F`) so it can never collide with a
/// `FULL(h2)` fingerprint. It is also distinct from `EMPTY_TAG
/// (0xFF)` and `DELETED_TAG (0x80)`, so existing find/delete logic
/// (look for h2 to peek, look for EMPTY to terminate) skips
/// `RESERVED` automatically as "neither — advance".
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
    /// Thin wrapper around [`insert_into_table_core`]: bounds-check the
    /// thread, look up its `(key, value)`, and dispatch with
    /// `overwrite = true` (existing values for the same key are replaced
    /// last-writer-wins). Probe and reclaim mechanics live in the helper.
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
        let _ = insert_into_table_core(ctrl, slots, keys[tid], values[tid], true);
    }

    /// `try_insert_kernel` — first-writer-wins variant.
    ///
    /// Same probe / claim mechanics as [`insert_kernel`] (delegates to
    /// [`insert_into_table_core`] with `overwrite = false`), but writes
    /// per-thread output reflecting whether the slot was fresh:
    ///   `out[tid] = FLAG_FRESH_OR_OK (0)`  -> we claimed a fresh slot
    ///                                         (or reclaimed a `DELETED` one)
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
        let fresh = insert_into_table_core(ctrl, slots, keys[i_thread], values[i_thread], false);
        if let Some(o) = out.get_mut(tid) {
            *o = if fresh {
                FLAG_FRESH_OR_OK
            } else {
                FLAG_PRESENT
            };
        }
    }

    /// `insert_kernel_dedup` — bulk insert with intra-warp duplicate
    /// detection via `tile.match_any(my_key)`.
    ///
    /// One thread per input. Each warp tile partitions its 32 lanes into
    /// duplicate-groups (by key) using `match_any`; only the highest-rank
    /// lane in each group performs the global insert via
    /// [`insert_into_table_core`]. Other lanes in the group drop their
    /// duplicate insert silently. Cross-warp duplicates still race on the
    /// global path and are arbitrated by the slot CAS plus the Phase 2
    /// `FULL(h2)+key` re-check.
    ///
    /// Picking the highest-rank lane per group gives "last-writer-wins
    /// within a warp" (the highest-rank lane has the largest input index
    /// in that warp, so its value is the most-recent for the warp).
    #[kernel]
    pub fn insert_kernel_dedup(
        ctrl: &[DeviceAtomicU32],
        slots: &[DeviceAtomicU64],
        keys: &[u32],
        values: &[u32],
    ) {
        let block = this_thread_block();
        let tile = block.tiled_partition::<32>();
        let lane = tile.thread_rank();
        let global_tid = thread::index_1d().get();

        // Tail lanes (past keys.len()) must still participate in `match_any`
        // — `WarpCollective::match_any` is a sync collective and requires
        // all lanes in the tile to call it. Inactive lanes contribute the
        // sentinel `FORBIDDEN_KEY` (= `u32::MAX`), which user keys are
        // forbidden from using, so they form their own dup group disjoint
        // from any active key.
        let active = global_tid < keys.len();
        let key = if active {
            keys[global_tid]
        } else {
            FORBIDDEN_KEY
        };
        let value = if active { values[global_tid] } else { 0 };

        let dup_mask = tile.match_any(key);

        if !active {
            return;
        }

        // Highest set bit in the dup mask is the in-warp leader for this
        // duplicate group. All lanes in the group agree on the same
        // leader because they all see the same dup_mask.
        let leader = 31u32 - dup_mask.leading_zeros();
        if lane == leader {
            let _ = insert_into_table_core(ctrl, slots, key, value, true);
        }
    }

    /// `rehash_kernel` — single-kernel two-buffer rehash.
    ///
    /// v3 ships the two-buffer mode only (`old != new`); the new buffer
    /// is `memset`-cleared by the host before launch, and the old buffer
    /// remains read-only for the duration of the kernel. That means no
    /// grid-wide barrier is required: each thread can read a live old
    /// slot and immediately insert it into the new table. Threads stride
    /// across the old table so the host can bound the launch size without
    /// depending on cooperative-launch residency limits.
    #[kernel]
    pub fn rehash_kernel(
        old_ctrl: &[DeviceAtomicU32],
        old_slots: &[DeviceAtomicU64],
        new_ctrl: &[DeviceAtomicU32],
        new_slots: &[DeviceAtomicU64],
    ) {
        let mut tid = thread::index_1d().get();
        let stride = (thread::gridDim_x() * thread::blockDim_x()) as usize;

        while tid < old_slots.len() {
            let ctrl_word_idx = tid / GROUP;
            let byte_in_word = tid % GROUP;
            let ctrl_atomic = &old_ctrl[ctrl_word_idx];
            let ctrl_word = ctrl_atomic.load(AtomicOrdering::Acquire);
            let tag = get_tag(ctrl_word, byte_in_word);
            if tag <= 0x7F {
                let slot_atomic = &old_slots[tid];
                let slot = slot_atomic.load(AtomicOrdering::Acquire);
                let _ = insert_into_table_core(
                    new_ctrl,
                    new_slots,
                    unpack_key(slot),
                    unpack_value(slot),
                    true,
                );
            }

            tid += stride;
        }
    }

    /// `find_kernel` — single-thread find, one thread per key.
    ///
    /// Walks the same triangular probe sequence as the insert kernels
    /// (same `PROBE_TILE = 32` width), so EMPTY-termination is sound —
    /// see `find_tile_impl` below for why probe-width coherence matters.
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

    /// `find_kernel_tile_32` — full-warp find (32-lane tile per query).
    ///
    /// One concrete instantiation of [`find_tile_impl`] at `N = 32`. Each
    /// warp partitions into a single 32-lane tile that scans one
    /// `PROBE_TILE`-byte insert tile per `tile.ballot` round, identical
    /// to v2's `find_kernel_warp_typed`.
    ///
    /// Launch with `LaunchConfig::for_num_elems(keys.len() * 32)`.
    #[kernel]
    pub fn find_kernel_tile_32(
        ctrl: &[DeviceAtomicU32],
        slots: &[DeviceAtomicU64],
        keys: &[u32],
        out: DisjointSlice<u32>,
    ) {
        let global_tid = thread::index_1d().get();
        find_tile_impl::<32>(ctrl, slots, keys, out, global_tid);
    }

    /// `find_kernel_tile_16` — sub-warp find (16-lane tile per query).
    ///
    /// One concrete instantiation of [`find_tile_impl`] at `N = 16`. Each
    /// warp partitions into two 16-lane tiles, each handling one query.
    /// Two queries per warp instead of one; each query scans
    /// `PROBE_TILE = 32` insert-tile bytes in **two** 16-byte ballot
    /// rounds rather than one 32-byte round, before triangular advance.
    ///
    /// The motivating regime is moderate load (75 % is the bench
    /// crossover): probe chains long enough that single-thread loses on
    /// per-key serialization, but short enough that a full-warp 32-lane
    /// scan per key is over-provisioned and leaves throughput on the
    /// table. Two queries per warp recover the headroom.
    ///
    /// Launch with `LaunchConfig::for_num_elems(keys.len() * 16)`.
    #[kernel]
    pub fn find_kernel_tile_16(
        ctrl: &[DeviceAtomicU32],
        slots: &[DeviceAtomicU64],
        keys: &[u32],
        out: DisjointSlice<u32>,
    ) {
        let global_tid = thread::index_1d().get();
        find_tile_impl::<16>(ctrl, slots, keys, out, global_tid);
    }

    /// Const-generic warp-cooperative find body, parameterised over the
    /// `N`-lane tile size. The two `find_kernel_tile_*` kernels above are
    /// the only callers; both inline this body via `#[inline(always)]`,
    /// so the const-generic monomorphises to two distinct PTX symbols
    /// with `N` folded as an integer literal.
    ///
    /// Algorithm (one tile per query):
    ///   1. Walk the `PROBE_TILE`-byte insert tile in `N`-byte sub-tiles.
    ///   2. Each sub-tile pulls `N` tag bytes into the tile via a single
    ///      coalesced ctrl load (lane `l` reads byte at `probe_base + sub
    ///      + l`).
    ///   3. `m_h2 = tile.ballot(tag == h2)` — N-bit fingerprint match
    ///      mask. For each set bit (lowest first) the matching lane
    ///      loads its slot and broadcasts the packed `(key, value)` via
    ///      two `shfl`s; on key match, lane 0 writes `out[tile_idx]` and
    ///      the tile returns.
    ///   4. `m_empty = tile.ballot(tag == EMPTY_TAG)` — if non-zero, the
    ///      key cannot live past an `EMPTY` in this hash chain; lane 0
    ///      writes `MISS` and the tile returns.
    ///   5. Else advance to the next `N`-byte sub-tile within the same
    ///      `PROBE_TILE` insert tile. After `PROBE_TILE / N` sub-tiles,
    ///      triangular-advance `probe_base` by `stride * PROBE_TILE` and
    ///      repeat.
    ///
    /// `N = 32` collapses the inner sub-tile loop to one iteration per
    /// probe step (identical to v2's `find_kernel_warp_typed`). `N = 16`
    /// runs two iterations per probe step but doubles the number of
    /// queries per warp.
    ///
    /// Insert and find share `PROBE_TILE`-aligned probe sequences; only
    /// the *granularity* of the find ballot changes between `N = 32` and
    /// `N = 16`. The same v3 table is queryable by either kernel.
    #[inline(always)]
    fn find_tile_impl<const N: u32>(
        ctrl: &[DeviceAtomicU32],
        slots: &[DeviceAtomicU64],
        keys: &[u32],
        mut out: DisjointSlice<u32>,
        global_tid: usize,
    ) {
        let block = this_thread_block();
        let tile = block.tiled_partition::<N>();

        let lane = tile.thread_rank();
        let tile_idx = global_tid / (N as usize);
        if tile_idx >= keys.len() {
            return;
        }

        let key = keys[tile_idx];
        let hash = hash_u32(key);
        let h2 = h2_from_hash(hash);
        let mask = slots.len() - 1;
        let mut probe_base = (hash as usize) & mask & !(PROBE_TILE - 1);
        let mut stride = 0usize;

        loop {
            let mut sub = 0usize;
            while sub < PROBE_TILE {
                let tag_pos = probe_base + sub + (lane as usize);
                let ctrl_word_idx = tag_pos / GROUP;
                let byte_in_word = tag_pos % GROUP;
                let word = ctrl[ctrl_word_idx].load(AtomicOrdering::Acquire);
                let tag: u8 = ((word >> (8 * byte_in_word)) & 0xFF) as u8;

                let mut m_h2 = tile.ballot(tag == h2);
                let m_empty = tile.ballot(tag == EMPTY_TAG);

                while m_h2 != 0 {
                    let cand = m_h2.trailing_zeros();
                    let local_slot: u64 = if lane == cand {
                        let slot_idx = probe_base + sub + (cand as usize);
                        slots[slot_idx].load(AtomicOrdering::Acquire)
                    } else {
                        0
                    };
                    let lo = tile.shfl(local_slot as u32, cand);
                    let hi = tile.shfl((local_slot >> 32) as u32, cand);
                    let observed: u64 = ((hi as u64) << 32) | (lo as u64);

                    if unpack_key(observed) == key {
                        if lane == 0 {
                            // SAFETY: tile_idx < keys.len() == out.len(),
                            // and each tile has a unique tile_idx so
                            // writes by lane 0 across tiles are disjoint.
                            unsafe {
                                *out.get_unchecked_mut(tile_idx) = unpack_value(observed);
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
                            *out.get_unchecked_mut(tile_idx) = MISS;
                        }
                    }
                    return;
                }

                sub += N as usize;
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

/// Outcome of [`try_reclaim_deleted`]: a remembered `DELETED` byte
/// either becomes ours, was reclaimed by a peer with our key (we're
/// already done), or was reclaimed by a peer with a different key
/// (we have to fall back to the standard `EMPTY` claim path).
enum ReclaimOutcome {
    Reclaimed,
    AlreadyHasOurKey,
    Lost,
}

/// Try to reclaim a `DELETED` slot at `(del_word_idx, del_j)` for
/// `(key, value)` via a two-stage handshake:
///
///   1. Tag-CAS `DELETED -> RESERVED` to claim exclusivity over the
///      byte. Concurrent inserts and finds skip `RESERVED` because it
///      is neither `h2` nor `EMPTY` — only the publisher (us) cares
///      about it.
///   2. Plain release-store `(key, value)` into the slot. The byte is
///      `RESERVED`, so the slot is invisible to all other readers.
///   3. Tag-CAS `RESERVED -> FULL(h2)` to publish via the standard
///      [`publish_full_tag`] retry loop.
///
/// On a CAS conflict at stage 1, re-load and re-classify the byte:
///   - Still `DELETED` -> retry the CAS (a sibling byte mutated).
///   - `RESERVED` -> spin until the racing publisher finishes.
///   - `FULL(h2')` -> the racing publisher landed; if its key equals
///     ours, treat as already-present (overwrite if `overwrite=true`),
///     else return `Lost` so the caller falls back to `EMPTY` claim.
///
/// `del_word_idx * GROUP + del_j` is the absolute slot index.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn try_reclaim_deleted(
    ctrl: &[DeviceAtomicU32],
    slots: &[DeviceAtomicU64],
    del_word_idx: usize,
    del_j: usize,
    key: u32,
    value: u32,
    h2: u8,
    overwrite: bool,
) -> ReclaimOutcome {
    let ctrl_atomic = &ctrl[del_word_idx];
    let slot_idx = del_word_idx * GROUP + del_j;
    let slot_atomic = &slots[slot_idx];

    loop {
        let cur = ctrl_atomic.load(AtomicOrdering::Acquire);
        let cur_tag = get_tag(cur, del_j);
        if cur_tag == DELETED_TAG {
            let new_word = set_tag(cur, del_j, RESERVED_TAG);
            if ctrl_atomic
                .compare_exchange(
                    cur,
                    new_word,
                    AtomicOrdering::AcqRel,
                    AtomicOrdering::Relaxed,
                )
                .is_ok()
            {
                slot_atomic.store(pack(key, value), AtomicOrdering::Release);
                let post = ctrl_atomic.load(AtomicOrdering::Relaxed);
                publish_full_tag(ctrl_atomic, post, del_j, h2);
                return ReclaimOutcome::Reclaimed;
            }
            // Else: a sibling byte changed; retry.
        } else if cur_tag == RESERVED_TAG {
            // A concurrent reclaim is publishing here; spin.
            continue;
        } else if cur_tag <= 0x7F {
            // Reclaim landed before us. If their key matches ours,
            // treat as already-present.
            let observed = slot_atomic.load(AtomicOrdering::Acquire);
            if unpack_key(observed) == key {
                if overwrite {
                    insert_overwrite(slot_atomic, observed, key, value);
                }
                return ReclaimOutcome::AlreadyHasOurKey;
            }
            return ReclaimOutcome::Lost;
        } else {
            // EMPTY_TAG (impossible under our state machine) or any
            // other unknown value: bail out.
            return ReclaimOutcome::Lost;
        }
    }
}

/// Core insert routine shared by every insert path
/// (`insert_kernel`, `try_insert_kernel`, `insert_kernel_dedup`,
/// `rehash_kernel`).
///
/// Returns `true` if the slot was fresh (claimed an `EMPTY` or
/// reclaimed a `DELETED`); `false` if the key was already present
/// (overwritten if `overwrite = true`, otherwise left untouched).
///
/// Probe shape: triangular in `PROBE_TILE`-byte tiles, identical to
/// every other v3 kernel so EMPTY-termination remains correct.
///
/// Per-tile algorithm:
///
///   * **Phase 1** — full tile scan. For each `FULL(h2)` tag whose
///     slot holds our key, take the duplicate path (overwrite or
///     report present, depending on `overwrite`). For each
///     `DELETED` tag, remember the first one for potential reclaim.
///
///   * **Reclaim attempt** — if we remembered a `DELETED`, try
///     [`try_reclaim_deleted`]. On `Reclaimed` / `AlreadyHasOurKey`,
///     return immediately. On `Lost`, forget the remembered byte
///     (don't second-guess) and fall through to the EMPTY claim.
///
///   * **Phase 2** — re-walk the tile looking for an `EMPTY` byte
///     to claim. Critically, also re-check `FULL(h2)` bytes for our
///     key here: a concurrent insert (or reclaim) may have published
///     our key into one of them in the window between Phase 1 and
///     Phase 2, and last-writer-wins requires we observe it instead
///     of landing a phantom duplicate at a later EMPTY.
///
///   * On EMPTY, slot-CAS `EMPTY_SLOT -> pack(k, v)`. On success,
///     publish via [`publish_full_tag`] and return fresh. On
///     same-key CAS failure, take the duplicate path. On
///     different-key CAS failure, keep scanning.
///
///   * **Advance** — no claim possible in this tile: triangular step
///     and repeat.
#[inline(always)]
fn insert_into_table_core(
    ctrl: &[DeviceAtomicU32],
    slots: &[DeviceAtomicU64],
    key: u32,
    value: u32,
    overwrite: bool,
) -> bool {
    let hash = hash_u32(key);
    let h2 = h2_from_hash(hash);
    let mask = slots.len() - 1;
    let mut probe_base = (hash as usize) & mask & !(PROBE_TILE - 1);
    let mut stride = 0usize;
    let mut first_deleted_word: usize = 0;
    let mut first_deleted_byte: usize = 0;
    let mut have_first_deleted: bool = false;

    loop {
        // Phase 1: scan tile for FULL(h2)+key (already-present check)
        // and remember the first DELETED byte for potential reclaim.
        let mut g = 0usize;
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
                        if overwrite {
                            insert_overwrite(slot_atomic, observed, key, value);
                        }
                        return false;
                    }
                } else if tag == DELETED_TAG && !have_first_deleted {
                    first_deleted_word = ctrl_word_idx;
                    first_deleted_byte = j;
                    have_first_deleted = true;
                }
                j += 1;
            }
            g += GROUP;
        }

        // Reclaim path (only if we saw a DELETED in Phase 1).
        if have_first_deleted {
            match try_reclaim_deleted(
                ctrl,
                slots,
                first_deleted_word,
                first_deleted_byte,
                key,
                value,
                h2,
                overwrite,
            ) {
                ReclaimOutcome::Reclaimed => return true,
                ReclaimOutcome::AlreadyHasOurKey => return false,
                ReclaimOutcome::Lost => {
                    have_first_deleted = false; // don't retry on later tiles
                }
            }
        }

        // Phase 2: claim an EMPTY byte. Also re-check FULL(h2)+key in
        // case a concurrent insert or reclaim published our key between
        // Phase 1 and now.
        let mut g = 0usize;
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
                        if overwrite {
                            insert_overwrite(slot_atomic, observed, key, value);
                        }
                        return false;
                    }
                } else if tag == EMPTY_TAG {
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
                            return true;
                        }
                        Err(actual) => {
                            if unpack_key(actual) == key {
                                if overwrite {
                                    insert_overwrite(slot_atomic, actual, key, value);
                                }
                                return false;
                            }
                            // Different key landed; continue scanning.
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

/// Inner CAS loop that overwrites a slot's value while preserving its key.
/// Used by every insert path's last-writer-wins branch; only ever entered
/// when the slot already holds our key.
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
/// slot at byte `i` is already exclusively ours via a winning slot CAS
/// (byte transitions `EMPTY -> FULL`); byte `i` cannot change under us,
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

/// Host-side handle to a v3 SwissTable-style GPU hashmap.
///
/// Owns two device-resident buffers:
///   - `ctrl`: `DeviceBuffer<u32>` of length `capacity / GROUP`. Each `u32`
///     holds 4 tag bytes packed little-endian.
///   - `slots`: `DeviceBuffer<u64>` of length `capacity`. Each `u64` packs
///     `(key, value)` with the key in the upper 32 bits.
///
/// Both buffers are `memset_d8_async(0xFF)` at construction so every tag
/// reads as `EMPTY_TAG` and every slot reads as `EMPTY_SLOT`.
///
/// `live_estimate` is a conservative upper bound on the number of live
/// keys (over-counts duplicates and re-inserts of the same key). It
/// is incremented only by the auto-resize-aware paths
/// ([`Self::insert_bulk_grow`]), never by [`Self::insert_bulk`] or
/// [`Self::insert_bulk_dedup`]; the bound is correct as a load-factor
/// trigger but should not be treated as a key count.
pub struct GpuSwissMap {
    /// Packed tag bytes; 4 tags per `u32` word. Element type is the
    /// atomic wrapper so kernels can mutate it through a shared slice
    /// without violating the `&[u32]` readonly/noalias contract.
    pub ctrl: DeviceBuffer<DeviceAtomicU32>,
    /// Packed `(key, value)` slots — key in the upper 32 bits. Atomic
    /// element type for the same reason as `ctrl`.
    pub slots: DeviceBuffer<DeviceAtomicU64>,
    /// Number of slots. Power of two, multiple of `GROUP`.
    capacity: usize,
    /// Conservative upper bound on live entries; only updated by
    /// auto-resize-aware insert paths.
    live_estimate: usize,
    /// Number of `resize_to` invocations completed on this table.
    resize_count: usize,
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
        // Reinterpret the allocations as slices of the atomic wrapper
        // (`#[repr(transparent)]`, identical layout). The memset above
        // is byte-level so it is unaffected by the element type.
        let ctrl = ctrl.cast_elem::<DeviceAtomicU32>();
        let slots = slots.cast_elem::<DeviceAtomicU64>();

        Ok(Self {
            ctrl,
            slots,
            capacity,
            live_estimate: 0,
            resize_count: 0,
        })
    }

    /// Number of slots in the table. May change across [`Self::resize_to`].
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of [`Self::resize_to`] invocations completed on this
    /// table. Used by the resize stress tests to verify that
    /// auto-resize is actually firing.
    pub fn resize_count(&self) -> usize {
        self.resize_count
    }

    /// Conservative upper bound on the number of live entries.
    /// Tracks only inserts that went through the auto-resize-aware
    /// path; see the [`GpuSwissMap`] doc for the exact contract.
    pub fn live_estimate(&self) -> usize {
        self.live_estimate
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

    /// Last-writer-wins bulk insert with intra-warp duplicate dedup
    /// via `tile.match_any(my_key)`.
    ///
    /// For inputs with high duplicate rates this is a strict win over
    /// [`insert_bulk`]: in-warp dups collapse to a single global insert
    /// per duplicate group (highest-rank lane wins), which both
    /// reduces global atomic traffic and concentrates the
    /// last-writer-wins arbitration into the per-group leader. Across
    /// warps the standard slot-CAS + Phase 2 `FULL(h2)+key` re-check
    /// still arbitrates correctly.
    ///
    /// For low-duplicate inputs the per-thread cost of `match_any`
    /// (one warp-sync vote + a `leading_zeros`) shows up; prefer
    /// [`insert_bulk`] when duplicates are statistically rare.
    pub fn insert_bulk_dedup(
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
            module.insert_kernel_dedup(stream, cfg, &self.ctrl, &self.slots, &keys_dev, &values_dev)
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

    /// Bulk find using the full-warp `find_kernel_tile_32` kernel —
    /// one 32-lane tile per query, 32 tag bytes inspected in parallel
    /// per `tile.ballot` round. Same return contract as `find_bulk`.
    ///
    /// Requires `capacity >= PROBE_TILE = 32`, which holds for any
    /// power-of-two capacity at or above 32.
    pub fn find_bulk_tile_32(
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

        // One 32-lane tile per key: launch keys.len() * 32 threads,
        // block size 256 means 8 tiles per block, 8 keys per block.
        let total_threads = (keys.len() as u32).saturating_mul(32);
        let cfg = LaunchConfig::for_num_elems(total_threads);
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.find_kernel_tile_32(
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

    /// Bulk find using the sub-warp `find_kernel_tile_16` kernel —
    /// two 16-lane tiles per warp, each handling one query. Each
    /// `tile.ballot` covers 16 tag bytes, so a full `PROBE_TILE = 32`
    /// insert tile is scanned in two ballot rounds before triangular
    /// advance.
    ///
    /// Same return contract as `find_bulk_tile_32`. The same v3
    /// table is queryable by either kernel — only the find ballot
    /// granularity differs.
    pub fn find_bulk_tile_16(
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

        // One 16-lane tile per key: launch keys.len() * 16 threads,
        // block size 256 means 16 tiles per block, 16 keys per block.
        let total_threads = (keys.len() as u32).saturating_mul(16);
        let cfg = LaunchConfig::for_num_elems(total_threads);
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.find_kernel_tile_16(
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

    /// Reallocate `ctrl` and `slots` to `new_capacity` slots and
    /// rehash all live entries from the old buffers into the new
    /// via the strided two-buffer [`rehash_kernel`].
    ///
    /// Both growing and shrinking are supported. `new_capacity` must
    /// be a power of two and at least `PROBE_TILE`. After this call,
    /// the old buffers are dropped (freed once the launch completes
    /// and the borrowed device pointers go out of scope).
    ///
    /// The old buffers are read-only and the new buffers start empty,
    /// so rehash does not require a cooperative grid-wide barrier.
    /// The launch size is capped and each thread handles a strided
    /// subset of old slots, which keeps resize portable across GPUs
    /// with smaller cooperative-launch residency budgets.
    pub fn resize_to(
        &mut self,
        new_capacity: usize,
        module: &kernels::LoadedModule,
        stream: &Arc<CudaStream>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert!(
            new_capacity.is_power_of_two(),
            "new_capacity must be a power of two"
        );
        assert!(
            new_capacity >= PROBE_TILE,
            "new_capacity must be >= PROBE_TILE ({PROBE_TILE})"
        );

        let new_ctrl = DeviceBuffer::<u32>::zeroed(stream, new_capacity / GROUP)?;
        let new_slots = DeviceBuffer::<u64>::zeroed(stream, new_capacity)?;
        unsafe {
            cuda_core::simt::memory::memset_d8_async(
                new_ctrl.cu_deviceptr(),
                0xFF,
                new_ctrl.num_bytes(),
                stream.cu_stream(),
            )?;
            cuda_core::simt::memory::memset_d8_async(
                new_slots.cu_deviceptr(),
                0xFF,
                new_slots.num_bytes(),
                stream.cu_stream(),
            )?;
        }
        let new_ctrl = new_ctrl.cast_elem::<DeviceAtomicU32>();
        let new_slots = new_slots.cast_elem::<DeviceAtomicU64>();

        let rehash_threads = self.capacity.min(REHASH_MAX_THREADS) as u32;
        let cfg = LaunchConfig::for_num_elems(rehash_threads);
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.rehash_kernel(stream, cfg, &self.ctrl, &self.slots, &new_ctrl, &new_slots)
        }?;

        self.ctrl = new_ctrl;
        self.slots = new_slots;
        self.capacity = new_capacity;
        self.resize_count += 1;
        // live_estimate stays as-is: rehash drops DELETED slots but
        // doesn't change the live-key count, and the estimate was
        // already a conservative upper bound.
        Ok(())
    }

    /// Last-writer-wins bulk insert with automatic resize-on-load.
    ///
    /// Before launching `insert_kernel`, doubles `capacity` (one or
    /// more times in a loop) until `(live_estimate + keys.len()) * 8
    /// <= capacity * 7`, i.e. the load factor would stay under 7/8.
    /// Each doubling invokes [`Self::resize_to`] and increments
    /// `resize_count`.
    ///
    /// Use this when input size isn't known in advance. For known
    /// upper bounds, sizing the table once via [`Self::new`] and
    /// using [`Self::insert_bulk`] is strictly cheaper.
    pub fn insert_bulk_grow(
        &mut self,
        keys: &[u32],
        values: &[u32],
        module: &kernels::LoadedModule,
        stream: &Arc<CudaStream>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(keys.len(), values.len());

        let projected = self.live_estimate + keys.len();
        while projected * 8 > self.capacity * 7 {
            let target = self.capacity * 2;
            self.resize_to(target, module, stream)?;
        }
        if !keys.is_empty() {
            self.insert_bulk(keys, values, module, stream)?;
            self.live_estimate = projected;
        }
        Ok(())
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
        // all be distinguishable by tag-byte value alone — the
        // DELETED -> RESERVED -> FULL(h2) reclaim handshake relies on
        // every transient state being distinguishable from FULL.
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
