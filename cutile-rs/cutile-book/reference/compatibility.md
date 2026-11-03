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

<!-- BEGIN TILE IR REQUIREMENTS -->
| Feature | 13.2 | 13.3 | 13.4 | Target requirement |
|---|---|---|---|---|
| baseline Tile IR | Yes | Yes | Yes | `sm_80+` |
| alloca | — | Yes | Yes | Any supported target |
| pack/unpack | — | Yes | Yes | Any supported target |
| block-scaled MMA | — | Yes | Yes | `sm_100+` |
| gather_scatter_view | — | Yes | Yes | Any supported target |
| strided_view | — | Yes | Yes | Any supported target |
| atomic_red_view_tko | — | Yes | Yes | Any supported target |
| insert | — | — | Yes | Any supported target |
| fpowi | — | — | Yes | Any supported target |
| GDC tokens | — | — | Yes | Any supported target |
| memory_fence_alias_tko | — | — | Yes | Any supported target |
| i4 | — | Yes | Yes | Any supported target |
| f8E4M3FN/f8E5M2 | Yes | Yes | Yes | `sm_90+` |
| f8E8M0FNU | Yes | Yes | Yes | `sm_100+` |
| f4E2M1FN | — | Yes | Yes | `sm_100+` |
| f8E5M3FNU | — | — | Yes | `sm_107+` |
| cuda_tile.mmaf_scaled with f8E5M3FNU scales | — | — | Yes | `sm_107` only |
| ptr_attr | — | — | Yes | Any supported target |
| rounding_mode=nearest_away | — | — | Yes | Any supported target |
| ftof additional rounding modes (type-dependent) | — | — | Yes | Any supported target |
| ftoi.saturating | — | — | Yes | Any supported target |
| inbounds=true | — | — | Yes | Any supported target |
| mmaf.fast_acc | — | Yes | Yes | Any supported target |
| exp.rounding_mode (non-approximate) | — | Yes | Yes | Any supported target |
| module.producer | — | Yes | Yes | Any supported target |
| global.constant | — | Yes | Yes | Any supported target |
| global.symbol_visibility | — | Yes | Yes | Any supported target |
| num_worker_warps_per_cta | — | Yes | Yes | Any supported target |
| num_cta_in_cga greater than 1 | Yes | Yes | Yes | `sm_90+` |
| num_worker_warps_per_cta restricted to 4 or 8 | — | — | Yes | Any supported target |
| bf16 atomic add | — | Yes | Yes | `sm_90+` |
| return inside loop | — | — | Yes | Any supported target |
| target sm_90 | — | Yes | Yes | `sm_90` only |
| target sm_107 | — | — | Yes | `sm_107` only |
| programmatic dependent launch | — | — | Yes | `sm_90+` |

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

Requirements combine. For example, FP4 packing needs both the `pack/unpack`
operations and the FP4 element type. A Hopper target requires Tile IR 13.3
even when the operations themselves were available earlier.

Architecture numbers are not a feature hierarchy: `sm_120` supports NVFP4,
but not scaled MMA using `f8E5M3FNU` scales. Operand types, shapes and attribute
values still have to satisfy the operation's verifier. In particular,
`num_worker_warps_per_cta` accepts 1, 2, 4, 8, 16 or 32 in 13.3, and only 4
or 8 in 13.4.

PDL additionally needs the driver's `cuLaunchKernelEx` entry point. Optional
host APIs are resolved at runtime; their availability is separate from the
device IR requirements. See the [PDL launch contract](host-api.md#programmatic-dependent-launch).

The driver/toolkit combination must meet
[NVIDIA's CUDA compatibility rules](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
We do not impose a blanket driver-version-at-least-toolkit-version check.

## Keeping the tables current

Requirements live in `cutile-ir/src/requirements.rs`, based on the
[versioned Tile IR specification](https://docs.nvidia.com/cuda/tile-ir/13.4/sections/stability.html).
When changing them, print the README and reference tables with:

```bash
cargo run -p cutile-ir --example feature_matrix
```

The `documented_matrices_match_the_registry` test checks both checked-in tables.
CPU tests exercise the requirements without a driver; assembler CI covers
13.2 and 13.3, with a 13.4 lane enabled when its image is configured.
Examples report unsupported prerequisites as `SKIP`; discovery, compilation,
and execution errors remain failures.
