# CUTLASS translation contracts

See the [translation guide](../translation.md) for setup and a first run,
and [the compiler flow](../pliron-mlir-flow.md) for the translation stages.
This reference covers build commands, verified mappings, and caller obligations.

## Build examples

After installing the managed CUTLASS compiler, run from `cuda-oxide/`:

```bash
export CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir

cargo oxide build elementwise_cute --arch sm_120a
cargo oxide build nvfp4_gemv_cute --arch sm_120a
cargo oxide build blockscale_gemm_cute --arch sm_120a
cargo oxide build fp16_gemm_256x352_cute --arch sm_100a
```

### Compiler selection and optimization

Installation verifies both archive and library digests. `cargo oxide`
resolves the managed compiler and fingerprints its library digest.
An explicit absolute path overrides discovery; library loading still
enforces the pinned digest:

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
CUDA_OXIDE_CUTLASS_COMPILER=/absolute/path/to/libCutlassCompiler.so \
  cargo oxide build elementwise_cute --arch sm_120a
```

Attention comparisons use the public reference's optimization level 2:

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 \
  cargo oxide build sm100_fmha_prefill_cute --arch sm_100a
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 \
  cargo oxide build sm100_mla_decode_fp16_cute --arch sm_100a
```

The execution manifest records the chosen level and the resulting pipeline
hash; changing this option invalidates the device build fingerprint.

## Supported mappings

| Example | Compiler-visible operations |
| --- | --- |
| [Elementwise](../../examples/elementwise_cute) | Tensor tiling and copying |
| [NVFP4 GEMV](../../examples/nvfp4_gemv_cute) | Scaled tensor views and GEMV |
| [Block-scaled GEMM](../../examples/blockscale_gemm_cute) | Scheduler, work tiles, TMA pipelines, shared-memory MMA, epilogue stores |
| [SM100 FP16 GEMM](../../examples/fp16_gemm_256x352_cute) | Typed shared tiles, TMEM, paired two-CTA MMA, cluster TMA, producer/consumer pipelines |

The [SM100 attention API](../../cute-rs/src/attention/sm100/mod.rs) supplies independent
QK and PV plans: group-one/group-two shared/shared MMA, group-one TMEM/shared MMA,
K-major and MN-major tiled shared layouts, rank-three typed TMA copies, and
FP32 TMEM fragments with packed FP16 stores. The semantic verifier checks
operand roles, layout extents, major order, and MMA group compatibility.
Attention kernels keep their warp schedule, online softmax, barrier phases,
and asynchronous buffer lifetimes explicit. Those dynamic obligations remain
with unsafe callers; a local operation order does not prove a cross-warp handoff.

At a block-scaled epilogue hand-off, `ReadyForTma` makes shared-memory writes
visible to TMA with an async-shared proxy fence, then emits a counted CTA
barrier. `Reusable` emits only the barrier.

## Grid-constant launch arguments

Grid-constant references use the same launch convention as the native backend:

```text
#[grid_constant] desc: &TmaDesc<...>
host descriptor -> 128-byte launch argument -> device &TmaDesc<...>
                   aligned to 64 bytes        same address for every thread
```

The backend checks the cubin's argument sizes and offsets against this layout.
The descriptor needs no separate device allocation; the tensor allocations it
points to must remain valid until GPU work completes. See the
[grid-constant safety requirements](../../../cuda-oxide-book/gpu-programming/kernels-and-device-functions.md#grid-constant-parameters)
for the full launch contract.

## SM100 contracts

The [Rust APIs](../../cute-rs/src/sm100/mod.rs) become verified `cute.sm100_*` plans:

| API | Contract |
| --- | --- |
| `Sm100SharedTile`, `TmaDesc<f16, Layout>` | Operand role and matching host/device layout |
| `Sm100TiledMma` | Two N partitions sharing A's collector lifetime |
| `Sm100TmaMmaPipeline`, `Sm100AccumulatorPipeline` | Buffer ownership and completion |
| `Sm100Tmem`, `Sm100TmemEpilogue` | TMEM allocation and FP32 → FP16 epilogue |

- **Compiler checks:** matching layouts and partitions, cluster transaction
  bytes, barrier arrivals, and the presence of allocation, MMA, copy, and
  release operations.
- **Caller responsibilities:** correct dynamic phases, participating lanes,
  and pointer lifetimes at each unsafe operation.
- **Profile:** two CTAs; FP16 inputs, FP32 accumulation; M256/K64;
  128-byte input swizzle; 512 TMEM columns; 128×32 epilogue with 64-byte swizzle.
  The example uses N192+160 and checks CPU FP32 accumulation rounded to FP16.

### CuTe and the A collector

- CUTLASS CuTe operations build shared descriptors, TMEM operations, and cluster TMA.
- CUTLASS 4.7's MMA atom cannot select when to fill or release the A collector.
  The backend emits an **NVVM TCGen05 MMA instruction** from the verified plan:

```text
Each K16:  first N partition → fill A collector
           second partition → last use of A collector
```

- Translation owns descriptor bits, instruction selection, barrier arrivals,
  completion multicast, and TMEM fences.

### Collective TMA loads

- `Sm100ClusterTmaCopy::copy`: all 32 producer lanes per CTA participate,
  with identical arguments within each warp; native CuTe elects internally.
- An outer elected-lane branch can deadlock on stage reuse. Regression cases
  cover more than five K stages and multiple persistent tiles.
- TMA **stores** have one issuing thread; the collective load rule does not apply.

## Inspect prepared MLIR

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
CUDA_OXIDE_MLIR_OUTPUT="$PWD/module.mlir" \
  cargo oxide build elementwise_cute --arch sm_120a
```

The [translation guide](../translation.md#safety-and-failures) describes failure
behavior and lists the supported environment options.
