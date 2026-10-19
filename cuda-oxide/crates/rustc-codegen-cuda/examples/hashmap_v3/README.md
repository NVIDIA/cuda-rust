# hashmap_v3

## GPU Hashmap v3 — Cooperative-Groups SwissTable

A `u32 -> u32` hashmap that builds on the v2 SwissTable design — packed
control-byte array, `h1` / `h2` hash split, triangular probing,
tombstone delete — and replaces every "manual warp intrinsic" path with
the typed cooperative-groups API. The same surface unlocks four new
capabilities that v2 did not have:

| Capability                    | Surface                  | Payoff                  |
|:------------------------------|:-------------------------|:------------------------|
| Sub-warp find tile            | `find_bulk_tile_16`      | Two queries per warp    |
| Intra-warp insert dedup       | `insert_bulk_dedup`      | Duplicate-heavy inputs  |
| `DELETED`-slot reclaim        | Every insert path        | Churn at high load      |
| Single-kernel rehash + resize | `resize_to` / grow path  | Strided auto-resize |

- `WarpTile<16>` is 1.2-1.3x faster than the full-warp tile at
  moderate load.
- `match_any` dedup is 17.5x faster than naive insert at 99 %
  key-clustered duplication.
- Rehash uses `rehash_kernel` under `GpuSwissMap::resize_to` and
  `insert_bulk_grow`.

What v2 had that v3 dropped (with the empirical reason):

- **Protocol A insert** (ctrl-CAS-first with a `RESERVED` handshake).
  v2's perf table showed Protocol B (payload-first) beat it by ~20 %
  at every load. The `RESERVED` tag is reused in v3 for a different
  purpose — the `DELETED -> RESERVED -> FULL(h2)` reclaim handshake.
- **Raw `warp::ballot` / `warp::shuffle` find kernel.** v3 uses only the
  typed cooperative-groups API; the `~12 %` non-inlining penalty
  (documented in the v2 perf section) is real, but `WarpTile<16>` more
  than makes it back at moderate load.

## Build and Run

```bash
# Correctness tests (15 hardware checks):
cargo oxide run hashmap_v3

# Performance bench vs CPU `hashbrown::HashMap`:
./cuda-oxide/crates/rustc-codegen-cuda/examples/hashmap_v3/run-bench.sh
# (or directly: cargo oxide run hashmap_v3 --bin bench)
```

The crate ships two binaries in one Cargo package:

- `hashmap_v3` (default) — 15 correctness tests on real hardware.
  Picked up by the workspace smoketest harness. Covers basic
  insert/find/delete, both find tile sizes (`tile_32`, `tile_16`),
  in-warp dedup correctness, `DELETED` reclaim under churn, two-buffer
  rehash via `resize_to`, and auto-resize-driven growth.
- `bench` — head-to-head perf bench across three load factors plus a
  dedicated "insert with duplicates" section measuring `match_any`
  dedup at 50 / 90 / 99 % duplicate rates.

## API Surface

```rust
let mut map = GpuSwissMap::new(capacity, &stream)?;          // power-of-two, >= 32

// Last-writer-wins bulk insert (one thread per key).
map.insert_bulk(&keys, &values, &module, &stream)?;

// Last-writer-wins with intra-warp dedup via tile.match_any().
// Clustered duplicate inputs see large speedups.
map.insert_bulk_dedup(&keys, &values, &module, &stream)?;

// First-writer-wins variant; returns Vec<bool> of fresh-or-already-present.
let fresh = map.try_insert_bulk(&keys, &values, &module, &stream)?;

// Three find variants. All take MISS = u32::MAX for absent keys.
let by_thread  = map.find_bulk(&query, &module, &stream)?;             // 1 thread / key
let by_warp    = map.find_bulk_tile_32(&query, &module, &stream)?;     // 1 warp / key
let by_subwarp = map.find_bulk_tile_16(&query, &module, &stream)?;     // 2 keys / warp

// Tombstone delete (FULL(h2) -> DELETED). Subsequent inserts may reclaim.
let deleted = map.delete_bulk(&keys, &module, &stream)?;       // Vec<bool>

// Resize: strided rehash kernel into freshly memset'd buffers.
map.resize_to(new_capacity, &module, &stream)?;

// Auto-resize wrapper: doubles capacity in a loop until projected load
// would stay under 7/8, then runs insert_bulk.
map.insert_bulk_grow(&keys, &values, &module, &stream)?;
```

## Algorithm Notes

### Tag-byte state machine

Every slot has a 1-byte tag, packed 4-per-`u32` into the `ctrl` array:

| Tag            | Value        | Meaning                  |
|:---------------|:-------------|:-------------------------|
| `EMPTY_TAG`    | `0xFF`       | Unclaimed slot           |
| `DELETED_TAG`  | `0x80`       | Reclaimable tombstone    |
| `RESERVED_TAG` | `0xFE`       | Transient insert claim   |
| `FULL(h2)`     | `0x00..0x7F` | Live entry               |

Top bit clear iff the slot is live, so `tag <= 0x7F` is the
`FULL(h2)` test. `EMPTY` and `DELETED` differ in the bottom bits so
find can stop on `EMPTY` and walk past `DELETED` with one mask.
`RESERVED` (`0xFE`) is distinct from both and from any `h2`, so find
and concurrent inserts skip it without special-casing.

`FULL(h2)` stores the bottom seven bits of the key's h2 fingerprint.
`RESERVED_TAG` means an insert has claimed the byte and is publishing
the payload.

### Insert (every path goes through `insert_into_table_core`)

Probe the chain in `PROBE_TILE`-byte tiles (`PROBE_TILE = 32` to match
the warp-cooperative find). Per tile:

1. **Phase 1** — full tile scan. For each `FULL(h2)` byte whose slot
   holds our key, take the duplicate path (overwrite or report
   present). Remember the first `DELETED` byte for potential reclaim.
2. **Reclaim attempt** — if Phase 1 saw a `DELETED`, run the
   `DELETED -> RESERVED -> FULL(h2)` handshake on it. On `Reclaimed`
   or `AlreadyHasOurKey`, return. On `Lost` (a peer claimed it for a
   different key), forget the byte and fall through.
3. **Phase 2** — re-walk the tile looking for an `EMPTY` byte.
   Critically, also re-check `FULL(h2)+key` here: a concurrent insert
   or reclaim may have published our key into a byte we already saw
   in Phase 1. Without the re-check, last-writer-wins could land a
   phantom duplicate at a later EMPTY.

The slot CAS is the global serialization point for `EMPTY`-claim, the
ctrl-byte CAS handshake is the serialization point for `DELETED`-claim,
and Phase 2's `FULL(h2)` re-check is the synchronization point for
already-published duplicates.

### Find (single-thread and both warp-cooperative tiles)

Same triangular probe sequence as insert. Per tile:

- Find any `FULL(h2)` byte whose slot holds the query key — return its
  value.
- If no h2 match and any byte in the tile is `EMPTY`, the key cannot
  live past this point in its chain (insert would have claimed the
  same `EMPTY`) — return `MISS`.
- Otherwise advance and repeat.

The `find_kernel_tile_16` variant runs a 16-lane warp tile that scans
each 32-byte insert-tile in two consecutive 16-byte ballot rounds
before the triangular advance. Insert and find share the
`PROBE_TILE = 32` walk granularity so the same v3 table is queryable
by either tile size.

### Delete

Walk the same probe sequence; on key match, tag-CAS the byte from
`FULL(h2)` to `DELETED_TAG` (with a per-word retry loop for sibling
mutations). The `(key, value)` payload is left in the slot — readers
only ever materialize slots whose tag is `FULL(h2)`, and a future
insert may reclaim the slot via the handshake above.

### Resize and rehash

`resize_to(new_capacity)`:

1. Allocate fresh `ctrl` and `slots` for `new_capacity`,
   `memset_d8_async(0xFF, ...)` so they read all-`EMPTY`.
2. Launch `rehash_kernel(old_ctrl, old_slots, new_ctrl, new_slots)`.
   The launch is capped, and each thread walks old slots in a grid-stride loop.
3. Each thread reads a live old slot and immediately re-inserts `(key, value)`
   into the new table via the standard `insert_into_table_core`.
4. Replace `self.ctrl` / `self.slots` / `self.capacity` with the new
   buffers; old buffers drop.

Because resize always rehashes into separate freshly cleared buffers, no
cooperative grid-wide barrier is needed: the old table is read-only while the
new table is being populated.

`insert_bulk_grow` is a thin wrapper: doubles `capacity` (in a loop)
until the projected post-insert load would stay under 7/8, then runs
`insert_bulk`. Each doubling increments `resize_count` so stress tests
can verify the trigger.

## Bench Snapshot — RTX 5090, sm\_120

Captured live by `./run-bench.sh`. Numbers fluctuate ~5 % across runs;
ratios are stable.

### Standard insert / find (1 M-slot table)

```text
Insert (Mops/s; higher is better)
                          load=50%   load=75%   load=90%
GPU                         4819.8    4799.7    4697.6
GPU dedup (no dups)         4808.4    4791.8    4693.2
CPU hashbrown (1 thread)     219.5     214.3     178.0
GPU            / CPU         22.0x     22.4x     26.4x
dedup / naive                 1.0x      1.0x      1.0x

Find — lookup (every query hits)
                          load=50%   load=75%   load=90%
GPU single-thread          22717.7   19819.4   17724.1
GPU tile_32 (1 key/warp)    7305.2    7102.3    7080.7
GPU tile_16 (2 keys/warp)   9485.9    8704.7    8380.3
CPU hashbrown (rayon)       1024.2    1671.9    1087.2
tile_16 / tile_32             1.3x      1.2x      1.2x

Find — lookup_fail (every query misses)
                          load=50%   load=75%   load=90%
GPU single-thread          12199.6   12388.3   11383.9
GPU tile_32 (1 key/warp)    9150.0    8919.5    8540.0
GPU tile_16 (2 keys/warp)  10221.5    9117.8    8230.2
tile_16 / tile_32             1.1x      1.0x      1.0x
```

The dedup overhead on zero-duplicate input is < 1 %; the
`tile_16` / `tile_32` win is consistent at every load and operation
type. Single-thread find still wins overall on hits because per-key
work is tiny relative to warp-coordination overhead — see v2's perf
discussion for the same observation.

### Insert with duplicates (1 M inputs, key-clustered layout)

```text
                             50% dup    90% dup    99% dup
GPU naive insert             8386.8     7159.6     1630.7
GPU dedup (match_any)        9819.6    23185.5    28511.3
dedup / naive                  1.2x       3.2x      17.5x
```

The naive path collapses at 99 % clustered duplication because
~32 lanes per warp hammer the same slot CAS. The deduped path
collapses each warp's same-key cluster to a single global insert via
`tile.match_any(my_key)` and the leader's CAS lands uncontended.

Random-permuted duplicates (the same key set, no clustering) show
much smaller wins — 32 random picks from N >> 32 keys are nearly
distinct, so intra-warp dedup has little to collapse. Cluster
duplicates (sorted-by-key bulk-load) are the canonical workload where
this kernel earns its keep.

## Cross-Version Comparison

The find-path table below pulls v2 numbers from the v2 bench (same
hardware, same session) for the apples-to-apples comparison:

```text
Lookup hits (Mops/s; higher is better)
                                  load=50%   load=75%   load=90%
v2 hand-written warp-coop          9452       9366       9215
v2 typed CG (tiled_partition::<32>)7999       7840       7811
v3 tile_32 (typed CG, full warp)   7305       7102       7081
v3 tile_16 (typed CG, sub-warp)    9486       8705       8380
```

`v3 tile_16` is at parity with v2's hand-written warp-coop at low
load and ~9 % behind at high load — the typed-API non-inlining cost
(documented in v2's perf section) is paid back by the sub-warp
parallelism. `v3 tile_32` runs the same algorithm as v2 typed CG but
through a const-generic body; the ~10 % gap there is the const-generic
dispatch shape (the outer sub-tile loop runs once for `N=32` but still
introduces a phi the compiler doesn't strip). Closing that gap is a
follow-up micro-optimisation; `tile_16` is the headline.
