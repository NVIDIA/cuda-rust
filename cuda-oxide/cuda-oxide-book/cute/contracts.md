# Checks and Obligations

The compiler verifies the whole module before any MLIR is built. Types carry the
static facts, the verifier checks how operations connect, and everything dynamic stays
with the author at `unsafe` boundaries.

```{figure} images/cutlass-primitives-at-a-glance.svg
:align: center
:width: 100%

Where cutlass-primitives sits: typed carriers select instruction families, one of two compiler profiles consumes them, and the author keeps the schedule.
```

## What is checked

| Layer | Checks |
| :--- | :--- |
| Rust types | layout, swizzle, operand role, element format, MMA shape, pipeline stage count, barrier ids; `TensorMut` over `DisjointSlice`; `TmemGuard` typestate; sealed traits |
| cute-layout proofs | cooperative copy coverage and per-thread contiguity, `ldmatrix` source validity, TMA encodability and swizzle phase alignment |
| dialect-cute verifier | tensor provenance, one pipeline init with complete producer and consumer state, cursor provenance, scheduler grids, MMA placement, epilogue phases, no compiler-only type reaches runtime storage |
| Target gate | `sm_100a`, `sm_103a`, `sm_107a`, or `sm_120a`; `tcgen05` on `sm_100a`/`103a`/`107a`; SM107 `MXF8F6F4` on `sm_107a`; MXF4 `mma.sync` on `sm_120a` |
| Launch ABI | cubin argument sizes and offsets match the exported MLIR; `#[grid_constant] &TensorMap` is a 128-byte by-value argument aligned to 64 |

Any failure stops the build. The importer is closed-world: a `cute_rs::` call it does not
recognise is an error, not a fallback.

## What you own

- Thread convergence and which lanes take part in each collective.
- Barrier phases, arrival counts and transaction bytes at run time.
- Asynchronous completion before a buffer or TMEM column is reused.
- Pointer lifetimes behind raw views and descriptors; the tensors a descriptor points
  to must stay alive until the GPU is done with them.
- Disjoint writes across threads.

## Two rules worth remembering

**SM100 collective TMA loads.** `Sm100ClusterTmaCopy::copy` needs all 32 lanes of the
producer warp with identical arguments; the operation elects a lane internally. An
outer elected-lane branch deadlocks once a stage is reused. TMA stores have one issuing
thread, so this rule does not apply to them.

**Collector buffers on SM100 and SM107.** Tensor Cores have a small internal scratchpad
called a **collector** that can stash an input operand so the next MMA reuses it without
reading shared memory again:

```text
SM100 paired FP16 (`Sm100TiledMma`, automatic per K16 step):
  first N partition   ──►  fill the A collector
  second N partition  ──►  last use of the A collector

SM107 MXFP8 (`SM107BlockScaledMmaMXF8F6F4Op`, explicit `BCollector` type marker):
  top M half    (`CollectorFill`)    ──►  reads B from SMEM and stashes B in the collector
  bottom M half (`CollectorLastUse`) ──►  reuses B from the collector and frees the slot
```

In `Sm100TiledMma`, the paired operation owns both halves so you cannot leave the A
collector dangling. In `SM107BlockScaledMmaMXF8F6F4Op`, you choose `CollectorDiscard`,
`CollectorFill`, or `CollectorLastUse` in the type parameters, and you own pairing each
`CollectorFill` with a matching `CollectorLastUse` on the same `K = 64` sub-step.

## Grid-constant launch arguments

```text
#[grid_constant] desc: &TensorMap<...>

host descriptor  ──►  128-byte launch argument  ──►  device &TensorMap<...>
                      aligned to 64 bytes            same address for every thread
```

The descriptor needs no device allocation. The full launch contract is under
[Kernels and Device Functions](../gpu-programming/kernels-and-device-functions.md).
