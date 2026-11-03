# Checks and Obligations

The compiler verifies the whole module before any MLIR is built. Types carry the
static facts, the verifier checks how operations connect, and everything dynamic stays
with the author at `unsafe` boundaries.

```{figure} images/ctm-rs-at-a-glance.svg
:align: center
:width: 100%

Where ctm-rs sits: typed carriers select instruction families, one of two compiler profiles consumes them, and the author keeps the schedule.
```

## What is checked

| Layer | Checks |
| :--- | :--- |
| Rust types | layout, swizzle, operand role, element format, MMA shape, pipeline stage count, barrier ids; `TensorMut` over `DisjointSlice`; `TmemGuard` typestate; sealed traits |
| cute-layout proofs | cooperative copy coverage and per-thread contiguity, `ldmatrix` source validity, TMA encodability and swizzle phase alignment |
| dialect-cute verifier | tensor provenance, one pipeline init with complete producer and consumer state, cursor provenance, scheduler grids, MMA placement, epilogue phases, no compiler-only type reaches runtime storage |
| Target gate | `sm_100a` or `sm_120a`; `tcgen05` only on `sm_100a`; MXF4 register MMA only on `sm_120a` |
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

**The A collector on SM100.** CUTLASS 4.7's MMA atom cannot choose when to fill or
release the A collector, so the backend emits the `tcgen05` MMA instruction from the
verified plan:

```text
each K16:   first N partition   ──►  fill the A collector
            second N partition  ──►  last use of the A collector
```

Translation owns the descriptor bits, instruction selection, barrier arrivals,
completion multicast and TMEM fences.

## Grid-constant launch arguments

```text
#[grid_constant] desc: &TensorMap<...>

host descriptor  ──►  128-byte launch argument  ──►  device &TensorMap<...>
                      aligned to 64 bytes            same address for every thread
```

The descriptor needs no device allocation. The full launch contract is under
[Kernels and Device Functions](../gpu-programming/kernels-and-device-functions.md).
