# Compile-Time Race Detection Example

**Paper section**: Section 3, "Example: Preventing Data Races"
**Purpose**: Demonstrate a data race that cuTile Rust prevents structurally.

<a id="the-example-kernel-internal-index-bug-in-head-permutation"></a>
## Incorrect destination index

A kernel permutes head dimensions of a `(b, h, m, d)` attention tensor.
Grid `(B*H, H, 1)` assigns each block a `(b, h1, h2)` triple: load from
`src[b, h1, m, :]`, store to `dst[b, h2, m, :]`.

The store index has `m` and `h2` swapped, so
the write is to `dst[b, m, h2, 0]` instead of `dst[b, h2, m, 0]`. The
launcher is correct (distinct `src` and `dst` tensors). The launcher's
borrow checker is satisfied.

For fixed `(b, m, h2)`, H blocks with different `h1` each load a different
source tile and store to the same destination `dst[b, m, h2, 0]`.
Concurrent writes of different values to the same address: a data race.

## Why this is a data race per the Tile IR memory model

The Tile IR memory model (see §2.3 of the paper) defines:

- **Tokens** order memory operations *within a single tile thread*. They
  do not provide inter-tile-thread ordering.
- **Scopes** order memory operations *across* tile threads. Scopes are
  `weak` (no scope), `tile_block`, `device`, and `sys`.
- A pair of conflicting accesses is *morally strong* only if each
  operation specifies a scope that includes the other's tile block.
  Accesses that are neither related in happens-before nor morally strong
  constitute a **data race**, with **undefined behaviour**.

The buggy kernel's concurrent stores are not ordered by tokens (tokens are
intra-tile-thread only) and not scoped to `device` (the stores are `weak`
in cuTile Python). These accesses are not morally strong, so the race causes
undefined behavior.

## Why cuTile Rust makes this bug inexpressible

In cuTile Rust, `dst` is partitioned on the host side:

```rust
let dst = zeros(&[b, h, m, d]).partition([1, 1, BM, BD]);
kernels::permute_heads(dst, src.clone()).sync()?;
```

Each tile thread receives `dst: &mut Tensor<f32, {[1, 1, BM, BD]}>` — a
partition view bound to exactly its assigned destination tile. The kernel
writes to its assigned tile with `dst.store(tile)`. It cannot select another
block's destination by swapping `m` and `h2`.

### Python (races, produces non-deterministic output)

```bash
python3 permute_race.py
```

Expected: `=== Data race confirmed. ===` with 17–35% of elements differing
across runs on each configuration.

### Rust (no race possible, semantic bugs at most)

```bash
cd /path/to/cutile-rs
cargo build -p cutile-examples --example permute_race_rust
cargo run  -p cutile-examples --example permute_race_rust
```

The Rust kernel uses a partition view for `dst`. The destination of each
store is fixed by the partition; the kernel cannot construct a wrong
destination index. Any bug the programmer introduces in the source load
produces a semantic bug (wrong output), not a data race.

## Files

| File | Description |
|------|------------|
| `permute_race.py` | Python: kernel-internal bug, demonstrates the race |
| `permute_race_rust.rs` | Rust: partition view makes the bug inexpressible |
