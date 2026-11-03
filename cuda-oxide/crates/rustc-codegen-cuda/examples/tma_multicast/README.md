# tma_multicast

## TMA Multicast Cluster Broadcast

Demonstrates TMA multicast: a single `cp.async.bulk.tensor` load broadcasts a
tile from global memory into the shared memory of **every CTA** in a thread
block cluster. One instruction, N copies — no extra bandwidth or thread work.

For basic TMA copies (sm_90+), see the [`tma_copy`](../tma_copy/) example.

## What This Example Does

Launches a cluster of 4 CTAs. CTA-0 thread-0 issues one multicast TMA copy.
The hardware delivers an identical 64x64 tile to all 4 CTAs' shared memory.
Each CTA then writes its tile to global memory for host-side verification.

## Key Concepts

### Multicast TMA Instruction

The multicast variant adds a `cta_mask` bitmask that selects which CTAs in
the cluster receive the tile:

```rust
#[kernel]
#[cluster_launch(4, 1, 1)]
pub fn tma_multicast_test(tensor_map: *const TmaDescriptor, ...) {
    // ...init barriers, cluster_sync, arrive...

    if cluster::block_rank() == 0 && thread::threadIdx_x() == 0 {
        let cta_mask: u16 = 0b1111; // all 4 CTAs
        cp_async_bulk_tensor_2d_g2s_multicast(
            &raw mut TILE as *mut u8,
            tensor_map,
            tile_x, tile_y,  // element offsets, not tile indices!
            &raw mut BAR,
            cta_mask,
        );
    }

    // ALL CTAs wait on their local barrier — each gets the tile
    while !mbarrier_try_wait(&raw const BAR, token) {}
}
```

### Cluster Synchronization

Every CTA must have its mbarrier initialized before the multicast fires,
because the TMA writes to all CTAs' shared memory and signals all their
barriers:

```text
CTA-0: mbarrier_init → fence ─┐
CTA-1: mbarrier_init → fence ─┤
CTA-2: mbarrier_init → fence ─┼→ cluster_sync() → CTA-0 issues multicast
CTA-3: mbarrier_init → fence ─┘
```

### CTA Mask

The `cta_mask` is a bitmask over cluster ranks. With a `(4,1,1)` cluster:

| Mask     | Effect                             |
|----------|------------------------------------|
| `0b1111` | All 4 CTAs receive the tile        |
| `0b0101` | Only CTAs 0 and 2                  |
| `0b0001` | Only CTA 0 (equivalent to unicast) |

## Generated PTX

```ptx
.target sm_120a          // follows the detected GPU; see "Which GPUs Run This?"
.explicitcluster
.reqnctapercluster 4, 1, 1

// Multicast TMA: broadcasts tile to all CTAs matching cta_mask
cp.async.bulk.tensor.2d.shared::cluster.global.tile
    .mbarrier::complete_tx::bytes.multicast::cluster
    [%rd_smem], [%rd_tensor_map, {%r_x, %r_y}], [%rd_mbar], %rs_mask;
```

Key PTX differences from unicast TMA:
- `.multicast::cluster` qualifier on the instruction
- Extra `%rs_mask` operand (16-bit CTA bitmask)
- `.explicitcluster` and `.reqnctapercluster` directives on the entry point

## Build and Run

```bash
cargo oxide run tma_multicast
```

## Expected Output

### On Blackwell Datacenter (sm_100)

`cargo oxide run` builds `.target sm_100a` for a B100/B200/GB200 and the
module loads and runs:

```text
=== TMA Multicast Example ===

GPU Compute Capability: sm_100
✓ embedded module loaded successfully

--- TMA Multicast (tma_multicast_test) ---

1. Setup: 4 CTAs in cluster, tile 64x64 (4096 floats)
2. Launching tma_multicast_test (cluster=(4,1,1), block=256)...
3. Verifying all 4 CTAs received the same tile...
   ✓ All 4 CTAs have identical tile data (4096 values each)!

🎉 TMA multicast successful — one load, 4 CTAs served!

=== TMA Multicast Test Complete ===
```

### On Consumer Blackwell (sm_120)

`cargo oxide run` builds `.target sm_120a` for an RTX 5090 (first verified
in #668). Transcript captured on an RTX 5090:

```text
=== TMA Multicast Example ===

GPU Compute Capability: sm_120
✓ embedded module loaded successfully

--- TMA Multicast (tma_multicast_test) ---

1. Setup: 4 CTAs in cluster, tile 64x64 (4096 floats)
2. Launching tma_multicast_test (cluster=(4,1,1), block=256)...
3. Verifying all 4 CTAs received the same tile...
   ✓ All 4 CTAs have identical tile data (4096 values each)!

🎉 TMA multicast successful — one load, 4 CTAs served!

=== TMA Multicast Test Complete ===
```

### On Hopper (sm_90/sm_90a)

Multicast is sm_90+ in the PTX ISA, and `cargo oxide run` on an H100 builds
`.target sm_90a`, so the module is expected to load and execute there. That
run is not yet captured; #966 tracks it.

The host gate is `major < 9`, the same floor as `tma_copy`. Below sm_90 the
example prints `skipping: TMA multicast requires sm_90 or newer` and exits
cleanly.

## Hardware Requirements

- **Multicast floor**: sm_90 or newer (PTX ISA). Verified on consumer
  Blackwell sm_120 (RTX 5090, #668); Hopper sm_90a run not yet captured (#966)
- **Build target**: `cargo oxide run` builds for the detected GPU; there is
  no fixed target baked into the example (see "Which GPUs Run This?")
- **Not supported**: anything below sm_90
- **CUDA Driver**: 12.0+
- **Cluster launch**: Required (`cuLaunchKernelEx` with cluster dimensions)

## Multicast vs Unicast TMA

| Aspect              | Unicast (`tma_copy`)             | Multicast (`tma_multicast`)         |
|---------------------|----------------------------------|-------------------------------------|
| Destination         | One CTA's shared memory          | All CTAs in cluster                 |
| Bandwidth           | 1x tile transfer                 | 1x transfer, N copies               |
| Architecture        | sm_90+ (Hopper+)                 | sm_90+ (Hopper+), seen on sm_120    |
| Use case            | Single-CTA tile loads            | GEMM/convolution with shared tiles  |
| `cluster_launch`    | Optional                         | Required                            |

## Pitfalls

**TMA coordinates are element offsets, not tile indices.** Passing `{1, 0}`
instead of `{64, 0}` for tile (1,0) causes `CUDA_EXCEPTION_27: Warp Illegal
Instruction Parameter` — the hardware requires coordinates aligned to the tile
(box) dimensions. This is easy to miss because `{0, 0}` is trivially aligned
and always works.

**`cluster_sync()` before the multicast is mandatory.** The multicast TMA
writes to every CTA's shared memory and signals every CTA's mbarrier. If any
CTA hasn't finished `mbarrier_init` + `fence_proxy_async_shared_cta` before
the multicast fires, the barrier tracking will be silently corrupt.

## Which GPUs Run This?

There is no fixed `.target` in this example. `cargo oxide run` forwards the
detected GPU as a hint (`CUDA_OXIDE_DEVICE_ARCH`, `sm_XYa` form for cc >= 9),
and rustc-codegen-cuda builds for that GPU whenever the kernel's features run
on it. sm_100a is only the fallback when no compatible GPU is detected:

```text
cargo oxide run on an RTX 5090 -> hint sm_120a -> .target sm_120a -> loads, runs
cargo oxide run on an H100     -> hint sm_90a  -> .target sm_90a  -> expected to run (#966)
no compatible GPU detected     -> no hint      -> .target sm_100a (fallback)
  (cargo oxide build, or run on a pre-Hopper GPU)
```

The host binary cannot see which target it embeds. If a module built on
another machine, or with `--arch`, does not JIT on the local GPU, the
load-failure arm prints the driver error and a `skipping:` line rather than
guessing at a target.

The multicast instruction itself has an sm_90+ floor in the intrinsic catalog,
which matches the PTX ISA: `sm_90a` is advised there for performance, not
required for legality. The host gate is therefore sm_90, not sm_100.
