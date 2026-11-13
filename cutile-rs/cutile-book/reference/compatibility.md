# CUDA and Tile IR compatibility

A kernel needs both a supported Tile IR version and a compatible target GPU.
The tables below come from the same requirements registry used by the compiler,
bytecode writer, launch guard, and example/test prerequisites.

The version is the **emitted Tile IR version**, not the version reported by
`cuDriverGetVersion`. We ask the selected `tileiras` which formats it accepts
and choose the newest one our writer also supports. Older assemblers without
`--list-versions` are checked by compiling a small representative kernel.
An explicit `CUTILE_BYTECODE_VERSION` must be supported; we do not downgrade it.

`CUTILE_TILEIRAS_PATH` takes precedence over the toolkit directory. A newer
driver does not make an older assembler understand newer bytecode.

Calls to tile-library functions ([Tile Libraries](../guide/tile-libraries.md))
need the selected version to be 13.5.

<!-- BEGIN TILE IR REQUIREMENTS -->
| Feature | 13.2 | 13.3 | 13.4 | 13.5 | Target requirement |
|---|---|---|---|---|---|
| baseline Tile IR | Yes | Yes | Yes | Yes | `sm_80+` |
| alloca | — | Yes | Yes | Yes | Any supported target |
| pack/unpack | — | Yes | Yes | Yes | Any supported target |
| block-scaled MMA | — | Yes | Yes | Yes | `sm_100+` |
| gather_scatter_view | — | Yes | Yes | Yes | Any supported target |
| strided_view | — | Yes | Yes | Yes | Any supported target |
| atomic_red_view_tko | — | Yes | Yes | Yes | Any supported target |
| insert | — | — | Yes | Yes | Any supported target |
| fpowi | — | — | Yes | Yes | Any supported target |
| GDC tokens | — | — | Yes | Yes | Any supported target |
| memory_fence_alias_tko | — | — | Yes | Yes | Any supported target |
| i4 | — | Yes | Yes | Yes | Any supported target |
| f8E4M3FN/f8E5M2 | Yes | Yes | Yes | Yes | `sm_90+` |
| f8E8M0FNU | Yes | Yes | Yes | Yes | `sm_100+` |
| f4E2M1FN | — | Yes | Yes | Yes | `sm_100+` |
| f8E5M3FNU | — | — | Yes | Yes | `sm_107+` |
| cuda_tile.mmaf_scaled with f8E5M3FNU scales | — | — | Yes | Yes | `sm_107` only |
| ptr_attr | — | — | Yes | Yes | Any supported target |
| unicast/multicast pointer | — | — | — | Yes | `sm_100` / `sm_103` / `sm_107` |
| multimem view operations | — | — | — | Yes | `sm_100` / `sm_103` / `sm_107` |
| atomic_red_view_tko.inbounds=true | — | — | — | Yes | Any supported target |
| if/select optimization hints | — | — | — | Yes | Any supported target |
| min_ctas_per_sm | — | — | — | Yes | Any supported target |
| argument attributes | — | — | — | Yes | Any supported target |
| rounding_mode=nearest_away | — | — | Yes | Yes | Any supported target |
| ftof additional rounding modes (type-dependent) | — | — | Yes | Yes | Any supported target |
| ftoi.saturating | — | — | Yes | Yes | Any supported target |
| inbounds=true | — | — | Yes | Yes | Any supported target |
| mmaf.fast_acc | — | Yes | Yes | Yes | Any supported target |
| exp.rounding_mode (non-approximate) | — | Yes | Yes | Yes | Any supported target |
| module.producer | — | Yes | Yes | Yes | Any supported target |
| global.constant | — | Yes | Yes | Yes | Any supported target |
| global.symbol_visibility | — | Yes | Yes | Yes | Any supported target |
| num_worker_warps_per_cta | — | Yes | Yes | Yes | Any supported target |
| num_cta_in_cga greater than 1 | Yes | Yes | Yes | Yes | `sm_90+` |
| bf16 atomic add | — | Yes | Yes | Yes | `sm_90+` |
| return inside loop | — | — | Yes | Yes | Any supported target |
| target sm_90 | — | Yes | Yes | Yes | `sm_90` only |
| target sm_107 | — | — | Yes | Yes | `sm_107` only |
| programmatic dependent launch | — | — | Yes | Yes | `sm_90+` |
| preview$call (foreign tile-library call) | — | — | — | Yes | Any supported target |

| Target | Minimum emitted Tile IR version |
|---|---|
| `sm_80` | 13.2 |
| `sm_86` | 13.2 |
| `sm_87` | 13.2 |
| `sm_88` | 13.2 |
| `sm_89` | 13.2 |
| `sm_90` | 13.3 |
| `sm_100` | 13.2 |
| `sm_103` | 13.2 |
| `sm_107` | 13.4 |
| `sm_110` | 13.2 |
| `sm_120` | 13.2 |
| `sm_121` | 13.2 |
<!-- END TILE IR REQUIREMENTS -->

The if/select optimization-hints row describes acceptance in the frontend IR.
These hints are discarded when emitting 13.5 bytecode and have no backend
performance effect.

Requirements combine. For example, FP4 packing needs both the `pack/unpack`
operations and the FP4 element type. A Hopper target requires Tile IR 13.3
even when the operations themselves were available earlier.

Architecture numbers are not a feature hierarchy: `sm_120` supports NVFP4,
but not scaled MMA using `f8E5M3FNU` scales. Operand types, shapes and attribute
values still have to satisfy the operation's verifier. In particular,
`num_worker_warps_per_cta` accepts 1, 2, 4, 8, 16 or 32 in 13.3 and 13.5,
and only 4 or 8 in 13.4.

PDL additionally needs the driver's `cuLaunchKernelEx` entry point. Optional
host APIs are resolved at runtime; their availability is separate from the
device IR requirements. See the [PDL launch contract](host-api.md#programmatic-dependent-launch).

The driver/toolkit combination must meet
[NVIDIA's CUDA compatibility rules](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
We do not impose a blanket driver-version-at-least-toolkit-version check.

## CUDA 13.5 features

The 13.5 surface includes unicast/multicast pointer classification,
`multimem_load_reduce_view_tko`, `multimem_reduce_view_tko`,
`multimem_store_view_tko`, atomic-view inbounds promises,
minimum residency hints, and function argument attributes.
The named `simt` preview is separate from this stable surface. cuTile Rust
supports [foreign tile-library calls](../guide/tile-libraries.md) through that
preview; see the guide for the supported operands and current limitations.

The raw Rust APIs are exposed as unsafe functions in `cutile::tileir`, defined
in `cutile/src/_tileir.rs`: `ptr_with_attr_unicast`, `ptr_with_attr_multicast`,
the three `multimem_*_view_tko` operations, and `atomic_red_view_tko_with_inbounds`.
The `multimem` module provides Add, Min, Max, And,
Or, Xor, Umin, and Umax reduction markers. These unsafe APIs require live,
correctly classified mappings and an explicit memory-ordering protocol.
A Tile IR token does not synchronize independent GPUs.

See [Unsafe GPU communication](unsafe-comms.md) for unicast and multicast
contracts, available synchronization primitives, and the runnable two-GPU example.

Likelihood uses the safe `Likely` extension trait, imported by
`use cutile::core::*`. The receiver must be a scalar `bool` or a boolean
`Tile`, and the method preserves its type and value:

```rust
if condition.likely::<true>() {
    // Ordinary control flow; the existing DSL restrictions still apply.
}
let result = select(mask.likely::<false>(), a, b);
```

The prediction is advisory: an incorrect hint never affects correctness.
Copies and inline helpers preserve the IR hint; boolean computations and
control-flow joins may discard it. Apply it to the final condition.

**Tile IR 13.5 discards likelihood hints during bytecode emission.** The
methods currently require 13.5 but have no backend performance effect on
that target. Ordinary `if` and `select` use the standard 13.5 encoding,
without an optimization-hints field.

Entry annotations accept
`optimization_hints = (sm_100 = (min_ctas_per_sm = 2,),)` and a `default` group.
`CompileOptions::min_ctas_per_sm` overrides the entry hint for the active GPU
and participates in JIT cache identity. Minimum residency is in the range
0–32; zero disables the constraint. Hints do not change program semantics.
Function argument attributes are available through the low-level IR builder.
`DISubprogram::is_declaration` exposes the 13.5 debug declaration flag; ordinary
definitions keep the flag clear, and older writers reject declarations.

The named `simt` preview has its own feature gate and is separate from the
versioned CTK 13.5 API surface.

Multicast operations require an NVLink SHARP-capable fabric and the supported
`sm_100`, `sm_103`, or `sm_107` families. All-false atomic inbounds arrays retain
their older version floor.

### Selecting the toolkit

Install a CUDA 13.5 toolkit for the host architecture and select its assembler
explicitly when the default environment uses an older toolkit:

```bash
export CUDA_TOOLKIT_PATH=/path/to/cuda-13.5
export CUTILE_TILEIRAS_PATH="$CUDA_TOOLKIT_PATH/bin/tileiras"
export CUTILE_BYTECODE_VERSION=13.5
```

Use one toolkit installation across projects. The assembler and its compiler
backend must support the requested bytecode version; changing a runtime library
path does not upgrade the assembler. Tile libraries must be built for the
toolkit that links them.

### Validation

Run from `cutile-rs/`. The normal CPU runner includes the new frontend,
bytecode, validation, and GPU-kernel compilation tests. When a matching CUDA Tile IR
translator is available, the following also checks the emitted images with its
independent C++ bytecode reader and verifier:

```bash
CUTILE_REFERENCE_TRANSLATE=/path/to/cuda-tile-translate \
  cargo test -p cutile --test tile_ir_13_5 --test tile_ir_13_5_multicast
```

With the shared 13.5 environment above, assemble the feature and multicast
execution kernels targeting `sm_100` without a GPU:

```bash
cargo test -p cutile --test tile_ir_13_5 --test tile_ir_13_5_multicast \
  assemble_ -- --ignored
```

Then on the two-or-more-GPU B200 instance with NVLink SHARP:

```bash
cargo test -p cutile --test tile_ir_13_5_multicast \
  multicast_load_reduce_and_store_execute_from_each_gpu -- --ignored --test-threads=1
bash scripts/run_gpu_tests.sh
```

The dedicated multicast test exercises each GPU as initiator, checks the
load-reduced output and both devices' storage after reduction/store, and fails
if the required devices or fabric are absent. It is ignored by default because
single-GPU machines cannot run it. Assembler and hardware execution remain
separate from passing the CPU suite.

## Keeping the tables current

Requirements live in `cutile-ir/src/requirements.rs`, based on the
[versioned Tile IR specification](https://docs.nvidia.com/cuda/tile-ir/13.4/sections/stability.html).
When changing them, print the README and reference tables with:

```bash
cargo run -p cutile-ir --example feature_matrix
```

The `documented_matrices_match_the_registry` test checks both checked-in tables.
CPU tests exercise the requirements without a driver; assembler CI covers
13.2, 13.3, and 13.4. See the
[Tile workflow](https://github.com/NVIDIA/cuda-rust/blob/main/.github/workflows/cutile-rs.yml)
for the current image pins and enabled jobs.
Examples report unsupported prerequisites as `SKIP`; discovery, compilation,
and execution errors remain failures.
