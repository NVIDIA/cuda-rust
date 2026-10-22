# FP16 GEMM architecture

The [Rust kernel](../src/kernel.rs) combines `cute-rs` layouts, tensor
operations and pipelines with `ctm-rs` thread/warp coordinates, descriptor
prefetch, barriers, fences, cluster addressing and TMEM allocation/release.
`cuda-device` supplies kernel annotations and dynamic shared storage. CuTe
owns its composite pipeline and tensor operations, including their internal
synchronization; the host encodes tensor maps from the same CuTe layout types.
The [CTM-only version](../../fp16_gemm_256x352_ctm/README.md) expresses the
schedule through instruction types and physical formats.

## Tile and pipeline

```text
                 one 256 × 352 output tile
                 192 columns       160 columns
                ┌─────────────────┬────────────────┐
CTA 0: 128 rows │                 │                │
                ├─────────────────┼────────────────┤
CTA 1: 128 rows │                 │                │
                └─────────────────┴────────────────┘
                 two-CTA MMA pair; A collector reused

A/B global ──cluster TMA──▶ shared [5 stages] ──MMA──▶ TMEM
C global   ◀─────TMA────── shared [2 stages] ◀──FP16 + bias──┘
```

- **Cooperative MMA:** 256×192 and 256×160 outputs share A for each K16
  instruction pair; each mainloop stage covers K64.
- **Warp roles:** each CTA has 192 threads: four epilogue warps, one MMA
  warp, and one TMA warp.
- **Input pipeline:** five shared stages overlap cluster TMA and MMA.
  Every lane of the producer warp participates in each native CuTe copy.
- **Accumulator and epilogue:** 512 TMEM columns, then eleven 128×32 slices
  per CTA; optional row bias, FP16 conversion and vector stores feed two
  shared C stages.
- **Persistent schedule:** clusters visit tiles in M-first order. The host
  uses the smaller of tile count and resident capacity; `--clusters` overrides it.
- **Shared memory:** 211,072 bytes per CTA, aligned to 1,024 bytes.

Host tensor maps and device tiles use the same Rust layout types:

| Layout | Shared tile (rows × columns) | TMA swizzle |
| --- | --- | --- |
| `ASmem` | 128 × 64 | 128 bytes |
| `B0Smem` | 96 × 64 | 128 bytes |
| `B1Smem` | 80 × 64 | 128 bytes |
| `CSmem` | 128 × 32 | 64 bytes |

## Typed layouts and descriptors

These kernel declarations bind shared operands, tensor maps, and paired MMA
dimensions. Layout and operand types do not prove collective participation or
asynchronous completion; kernel synchronization and caller obligations still
apply.

```rust
use cute_rs::{ComposedLayout, RowMajor, Sm100TiledMma, PipelineTmaUmma, Swizzle};

type ASmem = ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<128, 64>>;
type B0Smem = ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<96, 64>>;
type B1Smem = ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<80, 64>>;
type TiledMma = Sm100TiledMma<ASmem, B0Smem, B1Smem, 192, 160>;
type MainloopPipeline = PipelineTmaUmma<5, 77_824>;
```

All four typed TMA descriptors use `#[grid_constant]` parameters. The kernel
takes typed references:

```rust
#[grid_constant] a: &TensorMap<f16, ASmem>
```

The generated host launcher accepts each descriptor **by value**, passing
128 bytes directly without allocating a descriptor buffer in device global
memory:

```text
host descriptor -> 128 launch-argument bytes -> shared read-only descriptor
```

Each thread borrows the same descriptor storage. Element and layout types stay
explicit in the signature. Tensor data stays in device memory and must remain
live until GPU work completes; the unsafe host launch makes this caller
obligation explicit.

## Compiler flow and source map

The compiler preserves layout and pipeline information through verification
and lowering:

![The cute-rs stack from Rust types through verification to a CUDA cubin](../../../../cuda-oxide-book/cute/images/cute-rs-preview-stack-and-flow.svg)

- [SM100 APIs](../../../cute-rs/src/sm100/mod.rs): `Sm100SharedTile`,
  `Sm100ClusterTmaCopy`, `Sm100TiledMma`, `Sm100TmemEpilogue`, and pipelines.
- [Host setup](../src/main.rs): typed TMA descriptors, cluster launch, and validation.
- [Dialect operations](../../../dialect-cute/src/ops/sm100/mod.rs): layout,
  collector, and pipeline verification before lowering.
- [Example README](../README.md): computation, run commands, and performance.
