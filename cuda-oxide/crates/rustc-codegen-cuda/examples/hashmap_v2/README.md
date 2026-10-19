# hashmap_v2

## GPU Hashmap v2 — SwissTable-Inspired

A `u32 -> u32` hashmap that adds the three structural ideas behind
hashbrown's SwissTable to the open-addressed v1 baseline (the sibling
`hashmap` example crate):

1. **Control-byte array.** Probing reads a packed `ctrl: DeviceBuffer<u32>`
   of 1-byte tags (4 tags per word) instead of the full `(key, value)`
   payload. Most probe steps now touch one cache-line of metadata.
2. **h1 / h2 hash split.** The same hash drives two roles — `h1` (low
   bits, probe position) and `h2` (top 7 bits, in-tag fingerprint).
3. **Triangular probing.** Probe step `i` advances by `i` groups, which
   visits every group exactly once and avoids primary clustering.

Plus the operation the v1 baseline didn't have:

4. **Tombstone delete.** A successful delete CAS-flips the slot's tag
   from `FULL(h2)` to `DELETED`. Find skips past tombstones; v2's insert
   does **not** reclaim them — `hashmap_v3` adds reclaim and rehash on top.

v2 ships **two find kernels and two insert protocols** side by side so
you can mix and match for your workload:

- **Single-thread find** — one CUDA thread per query key. Cheapest at
  moderate load, where probes terminate fast.
- **Warp-cooperative find** — one warp (32 lanes) per query key,
  inspecting 32 tag bytes per probe step in parallel via `warp::ballot`
  (the GPU analog of hashbrown's SSE2 `_mm_movemask_epi8` ctrl scan).
  Worth it under long probe chains — high load and miss-heavy workloads.
- **Protocol B insert** — slot-CAS-first, one atomic on the payload
  followed by one publish-write to the ctrl byte. The default.
- **Protocol A insert** — ctrl-byte-CAS-first with a `RESERVED ->
  FULL` handshake. No slot CAS; concurrent finders see "in flight"
  via the `RESERVED` tag.

The bench binary in this crate measures all four GPU configurations
against single-threaded `hashbrown::HashMap` insert and rayon-parallel
`hashbrown` find at three load factors.

## Build and Run

```bash
# Correctness tests (12 hardware checks):
cargo oxide run hashmap_v2

# Performance bench vs CPU `hashbrown::HashMap`:
./cuda-oxide/crates/rustc-codegen-cuda/examples/hashmap_v2/run-bench.sh
# (or directly: cargo oxide run hashmap_v2 --bin bench)
```

The crate ships two binaries in one Cargo package:

- `hashmap_v2` (default) — 12 correctness tests on real hardware. Picked
  up by the workspace smoketest harness.
- `bench` — head-to-head perf bench: GPU insert (Protocol B vs A) ×
  find (single-thread vs warp-coop) × load factor (50/75/90 %), with a
  single-threaded CPU `hashbrown` insert baseline and a rayon-parallel
  CPU `hashbrown` find baseline (hashbrown allows any number of
  concurrent `&self` readers, so rayon-parallel is the honest CPU
  ceiling for find).

## Storage Layout

Two device-resident buffers, both `memset_d8_async(0xFF, ...)` at
construction so every tag reads `EMPTY` and every slot reads
`EMPTY_SLOT`:

```text
ctrl: DeviceBuffer<u32>   length N / GROUP   (GROUP = 4)

   one u32 = 4 packed tag bytes:

      31              24 23              16 15               8 7                0
      +------------------+------------------+------------------+------------------+
      |   tag for slot 3 |   tag for slot 2 |   tag for slot 1 |   tag for slot 0 |
      +------------------+------------------+------------------+------------------+
      \_______________________________ ctrl[group_idx] ___________________________/

   tag byte encoding:

      bit:   7   6   5   4   3   2   1   0
            +---+---+---+---+---+---+---+---+
   EMPTY    | 1   1   1   1   1   1   1   1 |   0xFF
   DELETED  | 1   0   0   0   0   0   0   0 |   0x80
   FULL(h2) | 0 |       h2 (7 bits)         |   0x00..0x7F
            +---+---+---+---+---+---+---+---+

slots: DeviceBuffer<u64>   length N (power of two, multiple of GROUP)

   each u64 packs (key, value):

      63                 32 31                  0
      +--------------------+--------------------+
      |       key (u32)    |     value (u32)    |
      +--------------------+--------------------+

   sentinel:                0xFFFF_FFFF_FFFF_FFFF      (= u64::MAX)
   forbidden user pair:     key = u32::MAX            (would collide with EMPTY_SLOT)
```

The slot layout matches the open-addressed v1 baseline (`hashmap` crate);
the new piece is the parallel `ctrl` array — probing reads tags first
and only touches a slot when a tag's fingerprint matches.

## Hash Split

```rust
let hash = (key as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
let h1   = hash as usize;                    // probe position
let h2   = ((hash >> 57) & 0x7F) as u8;      // 7-bit fingerprint, top bit clear
```

`h2` is taken from the top of the hash so it's statistically independent
of the low-bit position used for probing — same split hashbrown uses.
The 7-bit format guarantees `h2` can never collide with `EMPTY (0xFF)`
or `DELETED (0x80)`.

## Insert Protocols — B (Payload-First) and A (Ctrl-First Handshake)

Two insert protocols ship side by side so the bench harness can
measure them head-to-head. Same probe shape, same `PROBE_TILE`, same
find/delete kernels — they differ only in *how* a thread takes
ownership of an empty slot.

### Protocol B — payload-first

cuCollections-style. Two atomics per insert, no RESERVED state.

```text
1. group_idx = (h1 / GROUP) & ((N / GROUP) - 1)
2. word = atomic_load(ctrl[group_idx], Acquire)

   Phase 1 -- already-present check:
   for each byte i with tag(word, i) == h2:
     observed = atomic_load(slots[group_idx*GROUP + i], Acquire)
     if unpack_key(observed) == key:
       last-writer-wins:    slot-CAS overwrite loop, return
       first-writer-wins:   report PRESENT, return

   Phase 2 -- claim an EMPTY slot:
   for each byte i with tag(word, i) == EMPTY:
     match slots[group_idx*GROUP + i].cas(EMPTY_SLOT -> pack(k, v)):
       Ok          => publish: ctrl-word CAS sets byte i to FULL(h2). return
       Err(actual) where unpack_key(actual) == k => same-key race:
                       last-writer-wins:  slot-CAS overwrite loop, return
                       first-writer-wins: report PRESENT, return
       Err(actual) => different key in flight; skip byte, try next

3. No FULL(h2) match and no EMPTY in this group: stride += 1;
   group_idx = (group_idx + stride) & ((N / GROUP) - 1); loop.
```

The slot CAS itself is the serialization point. Two threads racing on
the same key always converge on the same slot (deterministic probe
order), and exactly one wins the CAS; the other observes `Err(actual)`
with a matching key and falls into the duplicate-handling path. Two
threads racing on different keys at the same slot likewise have one
winner, and the loser sees the mismatched key and probes past.

The publish step is a small CAS retry loop because **other** threads
may concurrently mutate **other** bytes in the same ctrl word for their
own inserts. Byte `i` itself can never change under us — no other
thread can claim a slot we already own — so the loop terminates as soon
as we observe a stable view of the other three bytes.

### Protocol A — ctrl-first (RESERVED handshake)

A new tag value `RESERVED (0xFE)` advertises "this slot's ctrl byte is
claimed but the payload isn't published yet". The handshake is one
ctrl-byte CAS to claim, one plain release store of the slot, one
ctrl-byte CAS to publish — no CAS on the slot itself.

```text
1. Phase 1 -- already-present check (same as Protocol B).

2. Phase 2 -- handshake claim:
   for each EMPTY byte i in the tile (per-word retry loop on collision):
     match ctrl_word.cas(byte i: EMPTY -> RESERVED):
       Ok      => slots[base + i] = pack(k, v)            // plain release store
                  ctrl_word.cas(byte i: RESERVED -> FULL(h2))   // publish
                  return
       Err(_)  => re-read this word, re-scan; continue

3. No match, no claimable EMPTY: triangular advance, loop.
```

Tag-value space:

```text
   0xFF  EMPTY
   0xFE  RESERVED      (Protocol A only; top bit set, never == h2)
   0x80  DELETED
0x00..0x7F  FULL(h2)
```

`RESERVED` is wedged between `EMPTY` and `DELETED` in numeric value but
it is functionally distinct: existing find / delete kernels treat it as
"neither h2 nor EMPTY" and advance, which is exactly the right behavior
— don't peek the slot (the payload isn't published), don't terminate
the probe (the byte isn't truly empty).

#### Why no slot CAS

Phase 2's ctrl-byte CAS is the serialization point. Once a thread wins
`EMPTY -> RESERVED` at byte `i`, no other inserter can reach that slot
(every other inserter sees `RESERVED` and treats it as "not for me").
The slot write is therefore exclusive — a plain release store suffices.

#### Cost — the same-launch duplicate caveat

In a single kernel launch, two threads inserting the same key may each
win the handshake at *different* bytes (T1 reads word, sees EMPTY at
j1, CAS-claims j1; T2 reads the same word concurrently, sees EMPTY at
j2, CAS-claims j2; both succeed). Both publish slots holding K. Find
returns whichever of the two slots it encounters first in the probe
order — internally consistent, but K appears twice in the table.

Across kernel launches the stream-sync release-acquire boundary
publishes the first launch's `FULL(h2)` before the second launch's
Phase 1 reads, so cross-launch dedup is correct (Test 12 verifies
this for `try_insert`).

If strict single-launch dedup is required, use Protocol B — the slot
CAS arbitrates same-key races deterministically.

## Find

```text
1. group_idx = (h1 / GROUP) & ((N / GROUP) - 1)
2. word = atomic_load(ctrl[group_idx], Acquire)

   for each byte i in 0..GROUP:
     if tag(word, i) == h2:
       observed = atomic_load(slots[group_idx*GROUP + i], Acquire)
       if unpack_key(observed) == key:
         return unpack_value(observed)

   if any byte in word is EMPTY: return MISS  (key cannot be in this chain)

3. stride += 1; group_idx = (group_idx + stride) & mask; loop.
```

Find skips past `DELETED` tags (they don't terminate the probe) and
stops on `EMPTY` (which does, because no later insert could legally
land beyond an empty slot in this hash chain).

## Delete

Tombstone-only. Probe like find; on key match, CAS the ctrl word from
the byte's current value to `DELETED`. The `(key, value)` payload is
left as-is — readers only ever load slots whose tag is `FULL(h2)`, so a
stale slot with a `DELETED` tag is unreachable.

v2 deliberately does **not** reclaim deleted slots on insert. A
delete-then-reinsert sequence still works (the new entry lands at a
fresh slot, find walks past the tombstone), at the cost of effective
capacity erosion under churn. Reclaim arrives with v3's rehash path.

## Probe-Step Width

All four kernels use the **same** triangular probe sequence with a
32-tag-byte tile per step (`PROBE_TILE`). Single-thread kernels iterate
the 32 bytes serially via a `GROUP`-sized inner loop (one ctrl word at
a time, four bytes each). The warp-cooperative find kernel inspects all
32 bytes in parallel — one lane per byte, decision via `ballot`.

Insert and find **must** share the probe shape: if one walked
4-byte tiles and the other walked 32-byte tiles, find could terminate
on an `EMPTY` slot in the wider window that insert had skipped over,
and miss perfectly valid keys.

## Warp-Cooperative Find

`find_bulk_warp` launches one warp (32 lanes) per query key. Each lane
owns one tag byte at `(probe_base + lane)`. Per probe step:

```text
1. Coalesced load of 8 ctrl words (= 32 tag bytes) into the warp.
2. m_h2    = ballot(tag == h2(K))   -> 32-bit fingerprint match mask
3. m_empty = ballot(tag == EMPTY)   -> 32-bit empty mask
4. while m_h2 != 0:
       cand = trailing_zeros(m_h2)
       lane `cand` loads slots[probe_base + cand]
       broadcast (key, value) via two `shuffle`s, all lanes key-compare
       hit  -> lane 0 writes out[warp_idx], return
       miss -> m_h2 &= m_h2 - 1, try next candidate
5. if m_empty != 0: lane 0 writes MISS, return
6. else: triangular advance, repeat
```

The single-thread `find_kernel` stays in the binary as the comparison
baseline. Both produce identical results on every input (verified by
Test 8); the bench binary measures the throughput crossover.

`find_kernel_warp_typed` is the same algorithm rewritten on
`cuda_device::cooperative_groups::WarpTile<32>` and ships in the bench
as a third row. It currently runs ~12–17 % slower than the hand-written
kernel because the four typed-API sites in the inner probe loop
(`tile.ballot ×2`, `tile.shfl ×2`) clear rustc's MIR `Inline` cost
threshold and end up as standalone `.visible .func` calls in the
generated PTX — each one a single PTX instruction wrapped in
`ld.param`/`ret`. Two-stage plan to close the gap: short term, shrink
the wrappers below the MIR threshold so the existing pass folds them;
long term, add a MIR-level inliner pass to mir-importer that applies
our own policy independent of rustc's.

## Correctness Tests (twelve)

| #  | Name                                 | What it verifies                                                  |
|:---|:-------------------------------------|:------------------------------------------------------------------|
|  1 | `insert_bulk` roundtrip              | every inserted key is findable with the inserted value            |
|  2 | miss on absent keys                  | disjoint key set must miss                                        |
|  3 | last-writer-wins on re-insert        | second `insert_bulk` overwrites every value                       |
|  4 | `try_insert_bulk` first-writer       | pass 2 reports all-present, table preserves pass-1 values         |
|  5 | load-factor stress (~75%)            | 12288 keys at 75% load all round-trip                             |
|  6 | delete-then-find                     | survivors hit with original values, deleted keys all return MISS  |
|  7 | delete-then-reinsert                 | re-inserted keys observable with new values, even past tombstones |
|  8 | warp-coop find parity                | warp-coop and single-thread find agree on 16384 mixed queries     |
|  9 | warp-coop find at ~75% load          | 12288 keys round-trip via the warp-coop kernel under load         |
| 10 | Protocol A insert round-trip         | A inserts round-trip and disjoint queries miss                    |
| 11 | Protocol A vs B parity at 75%        | both protocols round-trip at 75% load via single + warp find      |
| 12 | Protocol A try_insert (cross-launch) | pass 2 reports all-present, table preserves pass-1 values         |

Default capacity is `1 << 14` slots.

## Intentionally Out of Scope for v2

- **16-lane sub-warp tiles** — v2's warp-coop find uses a full 32-lane
  tile via the raw `warp::ballot` / `warp::shuffle` primitives. The
  typed `WarpTile<16>` path (masked `ballot_sync` / `shuffle_sync`)
  shipped in `cuda_device::cooperative_groups` after v2 was frozen and
  is exercised by `hashmap_v3`'s `find_kernel_tile_16`.
- **DELETED slot reclaim on insert** — v2 leaves tombstones in place;
  with enough delete-insert churn the table fragments and effective
  capacity shrinks. Reclaim and rehash ship in `hashmap_v3`.
- **Resize / rehash** — v2 has no resize path. Strided two-buffer
  rehash ships in `hashmap_v3`.
- **Generic `<K, V>`** — deferred for API design reasons.
- **Float keys** — PTX has no `compare_exchange` for floats.
- **Single-launch dedup under Protocol A** — best-effort by design;
  use Protocol B if strict single-launch single-occurrence is required.

## Performance vs CPU `hashbrown`

`./run-bench.sh` runs the head-to-head bench against single-threaded
`hashbrown` insert and rayon-parallel `hashbrown` find. GPU timings are
CUDA-event kernel-only (no H2D/D2H), CPU timings are wall-clock
`Instant::now()`. Numbers below are representative on a 24-core host
with one Ada-class GPU at 1 M slot capacity, 10 measured iterations
plus 3 warmup, in **Mops/s** (millions of operations per second):

| Operation                        | load=50% | load=75% | load=90% | speedup vs CPU hashbrown |
|:---------------------------------|:---------|:---------|:---------|:-------------------------|
| GPU insert — Protocol B          |     4732 |     4852 |     4847 | ~21–27×                  |
| GPU insert — Protocol A          |     3880 |     3952 |     3923 | ~17–22×                  |
| CPU insert — `HashMap::insert`   |      222 |      206 |      181 | (1×, baseline)           |
| GPU find  — single-thread (hits) |    22355 |    19357 |    17683 | ~12–34×                  |
| GPU find  — warp-coop (hits)     |     9483 |     9361 |     9219 | ~6–15×                   |
| CPU find  — rayon `.get` (hits)  |      650 |     1531 |      733 | (1×, baseline)           |
| GPU find  — single-thread (miss) |    12158 |    12303 |    11409 | ~3–4×                    |
| GPU find  — warp-coop    (miss)  |    11649 |    11608 |    10978 | ~3–4×                    |
| CPU find  — rayon `.get` (miss)  |     2795 |     2823 |     3773 | (1×, baseline)           |

Reading the table:

- **Insert** is the most lopsided. Single-threaded CPU `hashbrown`
  caps out around 200 Mops/s; the GPU pumps in ~5 G inserts/s using
  one thread per key. Protocol B beats Protocol A by ~20 % at every
  load — the slot-CAS does cost less than the two-stage RESERVED
  handshake on this hardware, even though Protocol A skips the slot
  CAS entirely. Protocol A's win is correctness latitude (`RESERVED`
  decouples claim from publish), not raw insert throughput.
- **Find on hits** is where single-thread find shines: probes
  terminate fast, so spinning up 32 lanes per warp for warp-coop is
  pure overhead at moderate load. Warp-coop only catches up under
  long probe chains.
- **Find on misses** is where CPU `hashbrown` claws back the most
  ground — its SIMD ctrl scan checks 16 tag bytes per `pcmpeqb` and
  hits an `EMPTY` quickly. The GPU still wins, but only ~3–4×.

If you're comparing to a *concurrent* CPU hashmap (`DashMap`,
`scc::HashMap`, etc.) the absolute CPU numbers above will shift, but
the shape of the table — GPU inserts are 1–2 orders of magnitude
faster, find ratios are tighter on misses than hits — stays the same.

## See Also

- **`hashmap_v3`** — the next iteration of this design. Same SwissTable
  storage layout and probe shape; adds 16-lane sub-warp find tiles via
  `WarpTile<16>`, intra-warp insert dedup via `tile.match_any`,
  `DELETED`-slot reclaim on insert, and a strided two-buffer rehash
  with auto-resize. v2 stays in tree as the frozen baseline.

## What v2 Ships in One Crate

- Split `ctrl` / `slots` arrays with hashbrown's h1/h2 hash split.
- Triangular probing on `PROBE_TILE = 32` tag bytes per step.
- Tombstone delete (`FULL(h2) -> DELETED` via 32-bit ctrl-word CAS).
- Two insert protocols — Protocol B (slot-CAS-first) and Protocol A
  (ctrl-byte `RESERVED -> FULL` handshake).
- Two find kernels — single-thread per key and warp-cooperative
  (32 lanes per key, `ballot` + `shuffle`).
- A perf bench binary measuring all four GPU configurations against
  CPU `hashbrown::HashMap`.
