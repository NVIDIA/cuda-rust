# hashmap

## GPU Hashmap v1 — Open-Addressed Static Map

A fixed-capacity `u32 -> u32` hashmap that lives entirely in device memory
and supports concurrent insert and find from thousands of threads. This
is the v1 baseline — `hashmap_v2` and `hashmap_v3` are sibling example
crates that layer SwissTable-style control bytes, triangular probing,
warp-cooperative probe, and cooperative-groups insert/find on top of it.

The point of v1 is to prove the end-to-end pipeline — host allocation,
kernel launch, atomic CAS protocol, correctness harness — with the
smallest possible algorithm so the moving parts are obvious.

## Build and Run

```bash
cargo oxide run hashmap
```

## What's in the Box

One device-resident array of packed slots:

```text
DeviceBuffer<u64>   length N (power of two)

slot layout:
   63                 32 31                  0
   +--------------------+--------------------+
   |      key (u32)     |     value (u32)    |
   +--------------------+--------------------+

empty sentinel:        0xFFFF_FFFF_FFFF_FFFF      (= u64::MAX)
forbidden user pair:   key = u32::MAX            (would collide with EMPTY)
```

A single 64-bit `compare_exchange` publishes a whole `(key, value)` pair
atomically, so no thread can ever observe a half-written slot. That is
the entire reason v1 stays at `u32 -> u32`: it fits in one 64-bit CAS.

## Insert Contracts

Two are exposed side-by-side, both backed by the same outer probe loop:

| Method            | On duplicate key | Returns                              | CPU analog                          |
|:------------------|:-----------------|:------------------------------------ |:------------------------------------|
| `insert_bulk`     | overwrite value  | `()`                                 | `hashbrown::HashMap::insert`        |
| `try_insert_bulk` | leave slot alone | `Vec<bool>` — `true` = key was fresh | `HashMap::try_insert` / `or_insert` |

Find is straightforward:

| Method      | Returns                             |
|:------------|:------------------------------------|
| `find_bulk` | `Vec<u32>`; `u32::MAX` marks a miss |

## Probe Loop in 30 Seconds

Linear probing on an open-addressed table:

```text
1. idx = hash(key) & (N - 1)
2. CAS(slot[idx], EMPTY -> pack(k, v)):
     Ok           => done                              -- claimed an empty slot
     Err(other) where unpack_key(other) == k => duplicate-key path:
                       insert_bulk:     value-CAS overwrite loop
                       try_insert_bulk: report "already present"
     Err(other)   => idx = (idx + 1) & mask; loop      -- hash collision, probe forward
```

Find walks the same path, terminates on the first key match (hit) or
the first EMPTY slot (miss). v1 holds load factor under 7/8 host-side
so probe walks stay short.

## Hash Function

FxHash-style single multiply, same constant `hashbrown` uses for
`FxHashMap`:

```rust
fn hash_u32(key: u32) -> u64 {
    (key as u64).wrapping_mul(0x517c_c1b7_2722_0a95)
}
```

Cheap on GPU (one widening multiply), produces 64 bits of hash so the
SwissTable variants in `hashmap_v2` / `hashmap_v3` can split into a
low-bit probe position and a high-bit fingerprint without a second
hash call.

## Correctness Tests (six)

| # | Name                           | What it verifies                                          |
|:--|:-------------------------------|:----------------------------------------------------------|
| 1 | `insert_bulk` roundtrip        | every inserted key is findable with the inserted value    |
| 2 | miss on absent keys            | disjoint key set must miss                                |
| 3 | last-writer-wins on re-insert  | second `insert_bulk` overwrites every value               |
| 4 | `try_insert_bulk` first-writer | pass 2 reports all-present, table preserves pass-1 values |
| 5 | mixed dup/fresh batch          | flags exactly true on the fresh half, values preserved    |
| 6 | load-factor stress             | 75% load factor, all keys still round-trip                |

Default capacity is `1 << 14` slots; tests run at 50% and 75% load.

### Test 6 at a glance

Test 6 only exercises the **hit** path; it never queries an absent key.
The exact workflow:

```text
1. start with a fresh empty table (16384 slots, all EMPTY)
2. generate 12288 DISTINCT random keys  K[0..12288]
3. insert_bulk(K, V) where V[i] = i        <-- table is now 75% full
4. find_bulk(K)                            <-- query the SAME 12288 keys
5. assert found[i] == V[i] for every i     <-- every key must hit
```

Step 4 is the load-bearing one. The query batch is the same set of
keys we just inserted, so every find should return a hit. What makes
this a **stress** test is that at 75% load probe walks are no longer
trivial:

```text
load:        12.5%        50%          75%
avg probe:   ~1.07        ~1.5         ~2.5..4
max probe:   ~5           ~10          ~20..30   (linear-probing rules of thumb)
```

At low load most lookups land on the right slot on probe 0. At 75%
load a typical find walks several "wrong key" slots before finding
its match, and that probe walk is where bugs hide — off-by-one in
`(idx + 1) & mask`, mishandled wrap-around, ordering issues that let
a finder skip past a slot mid-publish. Test 6 says: even with that
pressure, every inserted key still round-trips to the right value.

What test 6 does **not** cover:

- **Miss queries at 75% load.** A find for an absent key has to walk
  until it sees an EMPTY slot, which at high load can be much longer
  than a hit probe. Worth adding as a follow-up test.
- **Concurrent insert + find.** Insert and find run in separate
  kernel launches with stream sync between them, so there are no
  insert-vs-find races. Mid-flight concurrency is a SwissTable concern
  (handled in `hashmap_v2` / `hashmap_v3` via the `RESERVED`-tag
  handshake).

## Intentionally Out of Scope for v1

- **Deletion** — `hashmap_v2` and `hashmap_v3` introduce tombstone tags.
- **Resize / rehash** — `hashmap_v3` adds a single-kernel strided
  two-buffer rehash.
- **Generic `<K, V>`** — deferred for API design reasons (sentinel
  selection, slot-packing atomicity, hasher choice), not because of any
  cuda-oxide compiler limitation. `#[kernel]` already supports generics.
- **Float keys** — PTX has no `compare_exchange` for floats, so they
  cannot be slot discriminants in this design.

## See Also

The SwissTable variants live in **separate sibling example crates** so
this one stays a clean, minimal reference:

- **`hashmap_v2`** — adds a parallel `ctrl: DeviceBuffer<u32>` of
  packed 4-tags-per-word control bytes, an `h1` / `h2` split off the
  same hash, triangular probing, tombstone delete, and a warp-cooperative
  `find` that uses `ballot` as the GPU analog of hashbrown's SSE2
  `_mm_movemask_epi8`. Two insert protocols (slot-CAS-first and
  ctrl-CAS-first with a `RESERVED` handshake) ship side by side.
- **`hashmap_v3`** — keeps v2's storage layout and probe shape, drops
  the second insert protocol, and adds 16-lane sub-warp find tiles
  (`WarpTile<16>`), intra-warp insert dedup via `tile.match_any`,
  `DELETED`-slot reclaim on insert, and a strided two-buffer rehash
  with auto-resize.

None of that retrofits onto v1; v1 stands on its own.
