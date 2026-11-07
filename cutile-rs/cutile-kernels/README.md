# cutile-kernels

Reusable cuTile Rust kernels.

This crate contains Rust source kernels launched from the host. To find
`.tilelib` packages whose device functions can be called from inside a kernel,
see [Finding tile libraries](../cutile-book/guide/tile-libraries.md#finding-tile-libraries).

Kernel modules are grouped by function: attention, norms, positional encoding,
KV-cache updates, argmax/sampling, embeddings, and pointwise transformer
utilities.

`experimental/` contains raw-pointer kernels and benchmark-oriented kernels,
including grouped GEMM, fused QKV, NVFP4 MoE, split-K prefill, persistent RMSNorm, and f16
KVBM block-layout conversions.

**Peak performance requires CUDA Toolkit (CTK) 13.5 and its Tile IR assembler.**
This is a performance recommendation, not a new minimum for all kernels.
NVFP4 MoE requires block-scaled MMA support: Tile IR 13.3 or newer and SM100+
(including SM120/121). Other kernels retain their existing target requirements.
Select the 13.5 `tileiras` binary, for example via `CUTILE_TILEIRAS_PATH`;
the kernels can use the existing supported bytecode formats.

## Grout kernels

The corpus includes the kernels from
[Grout `23ae5a3`](https://github.com/elibol/grout/blob/23ae5a3e78dc39e242918824839ba8a35e4adce9/src/kernels.rs),
organized into separate modules. The current cuTile API adaptations are retained,
including guarded blocks in place of early returns in the LPT attention kernels.
The refresh adds vector operations, single-token embedding/RoPE/cache updates,
head-major attention, baseline GEMM, split-K prefill and merge, and persistent
RMSNorm. The baseline GEMM and persistent RMSNorm variants are experimental;
their presence does not imply they are Grout's default dispatch choices.

Kernel implementations belong here; runnable host-side examples in
[`cutile-examples`](../cutile-examples/README.md) import them instead of keeping
separate kernel copies.

`experimental::fmha_prefill_gqa_lpt_split` and `prefill_splitk_reduce_merge`
form a two-launch causal attention pipeline. Use matching specialization and
scheduling parameters. With `M_EFF = BM * GROUP` and
`NS_M = NUM_KV_SPLITS * M_EFF`, allocate partial attention as
`[num_q_blocks * num_head_groups, NS_M, D]` (f16) and partial log-sum-exp as
`[num_q_blocks * num_head_groups, NS_M]` (f32). The output is
`[q_len, num_head_groups * GROUP, D]`. See the entry-point safety contracts for
alignment, grid, and dimension requirements.

## NVFP4 MoE

`experimental::nvfp4_moe` exposes six kernels by Emmet Zhang et al.:

| Entry | Role |
| --- | --- |
| `routing_group_pairwise_kernel` | Group up to 16 token/expert routes |
| `routing_group_histogram_kernel` | Group larger batches using expert counts and offsets |
| `nvfp4_quantize_kernel` | Quantize f16 activations to packed FP4 with FP8 block scales |
| `grouped_fc1_kernel` | Grouped FC1, ReLU-squared, and requantization |
| `grouped_fc2_kernel` | Grouped FC2 with BF16 output |
| `finalize_kernel` | Combine routed expert outputs with routing weights |

These are low-level, unchecked entry points. The
[`nvfp4_moe` example](../cutile-examples/examples/nvfp4_moe.rs) demonstrates the
buffer layouts, generic arguments, launch order, numerical checks, and CUDA graph
capture using this library. From `cutile-rs/`, run:

```sh
cargo run --release -p cutile-examples --example nvfp4_moe
```

The example is a synthetic correctness and timing fixture; its timings are not
a comparison against a production MoE library. See the
[current validation gap](../cutile-examples/README.md#nvfp4-moe-validation-gap)
observed with public `tileiras` 13.5.37 on SM120 before using its results.

## Fused QKV

`experimental::fused_qkv` fuses FP16 QKV projection, FP32 per-head Q/K RMSNorm,
FP8 conversion, Q storage, and paged K/V scatter. V is not normalized. The
[runnable example and benchmark guide](../cutile-examples/fused_qkv.md) documents
the unsafe buffer/slot contract, full-output correctness fixture, and opt-in
CUDA graph timing sweep. Use CUDA 13.5's assembler for performance evaluation;
the starting schedule has not been tuned on B200 and overlap is not guaranteed.
The adapted TileGym traversal retains its MIT license in the kernel source.

## Validation

From `cutile-rs/`, `cargo test -p cutile-kernels --test smoke` lowers all 48
entry points to Tile IR and serializes bytecode. It does not launch GPU kernels.
NVFP4 compilation is checked for both SM100 and SM120 with explicit Tile IR
13.3 and 13.4 bytecode versions, independent of the selected assembler.
QKV compilation covers SM90/100/120 and the minimum supported formats as well
as SM100 with 13.5 bytecode. Its default example checks direct and graph launches.

`cargo test -p cutile-kernels --test gpu -- --ignored --test-threads=1` compares
split-K prefill and persistent RMSNorm against CPU references on a CUDA GPU.
It covers all four prefill schedules, tail rows, cached prefixes, empty splits,
and multiple grid strides. Both suites are included in the corresponding
`scripts/run_cpu_tests.sh` / `scripts/run_gpu_tests.sh` runners. The NVFP4 example
validates the complete MoE pipeline on supported devices.
