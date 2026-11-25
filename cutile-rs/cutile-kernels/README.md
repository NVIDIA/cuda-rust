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

## Choosing a starting point

| Operation | Source | What to check before reuse |
| --- | --- | --- |
| Prefill and decode attention | [`attention`](src/attention.rs), [`experimental::attention`](src/experimental/attention.rs) | Causal masking, query groups, tensor layout, active KV length versus reserved capacity, and split/merge scratch |
| GEMM and short causal attention | [`gemm`](src/experimental/gemm.rs), [`gemm_variants`](src/experimental/gemm_variants.rs), [`attention_f16`](src/experimental/attention_f16.rs) | Input/output precision, row alignment, masked tails, and tile working set; see the variant table below |
| Expert computation | [`moe_f16`](src/experimental/moe_f16.rs), [`nvfp4_moe`](src/experimental/nvfp4_moe.rs), [`moe`](src/experimental/moe.rs) | FP16 versus block-scaled FP4, routing layout, quantization, and all stages needed for the final weighted output |
| Normalization and fused epilogues | [`norms`](src/norms.rs), [`experimental::norms`](src/experimental/norms.rs), [`fused_transformer`](src/experimental/fused_transformer.rs) | Single-token versus multi-row work, in-place/aliasing rules, and whether fusion eliminates an intermediate |
| Embeddings, RoPE and KV movement | [`embeddings`](src/embeddings.rs), [`positional`](src/positional.rs), [`kv_cache`](src/kv_cache.rs), [`kvbm`](src/experimental/kvbm.rs) | Physical strides, position source, and cache/block layout; convenience variants have different calling contracts |
| Pointwise operations and selection | [`pointwise`](src/pointwise.rs), [`argmax`](src/argmax.rs) | Vector versus matrix layout, output type, and any required reduction launch |
| Fused QKV projection | [`qkv`](src/experimental/qkv.rs) | Head layout, quantization scales, slot bounds and schedule compatibility; use the host guide below |

Start with an entry that matches the operation and buffer contract, then tune
its configuration. Neither inclusion here nor the `experimental` label is a
performance ranking. Compare with the appropriate library baseline on the
target workload; the validation commands below check correctness, not optimality.

## Grout kernels

The corpus includes the kernels from
[Grout `23ae5a3`](https://github.com/elibol/grout/blob/23ae5a3e78dc39e242918824839ba8a35e4adce9/src/kernels.rs),
organized into separate modules. The current cuTile API adaptations are retained,
including guarded blocks in place of early returns in the LPT attention kernels.
The refresh adds vector operations, single-token embedding/RoPE/cache updates,
head-major attention, mixed-precision GEMM, split-K prefill and merge, and persistent
RMSNorm. The GEMM and persistent RMSNorm variants are experimental;
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

The raw-pointer `fmha_prefill_gqa_lpt` and `fmha_prefill_gqa_lpt_split` entries
retain their contiguous K/V contract: physical head stride `kv_len * D`.
For caches with reserved capacity, use `fmha_prefill_gqa_lpt_strided` or
`fmha_prefill_gqa_lpt_split_strided`. Immediately after `kv_len`, these take
`k_head_stride` and `v_head_stride`, independently, **in elements**. For example,
K/V storage `[kv_heads, capacity, D]` has head stride `capacity * D`, while
`kv_len` remains the active length. Token rows must still be contiguous with
stride `D`; this interface does not support arbitrary layouts. Split scratch
and the merge call are unchanged. The tensor-view attention entries are another
option when the existing view already carries the required strides.

## GEMM, causal attention and FP16 MoE variants

| Module | Choices and output |
| --- | --- |
| `experimental::gemm` | FP16 operands and output, FP32 accumulation; K must be divisible by the K tile size |
| `experimental::gemm_variants` | FP16 inputs, FP32 output; TMA/non-TMA access, transposed MMA, masked gathers for contiguous matrices with unaligned row strides |
| `experimental::attention_f16` | FP16 causal attention with FP32 output; single-tile, narrow-tail and streaming paths |
| `experimental::moe_f16` | Routed FP16 ReLU-squared MoE; direct gathers or packed input, FP32 weighted atomic combine |

These are experimental building blocks, not an automatic dispatch policy. Tile
sizes, layout constraints and scratch requirements are documented beside the
entries. Raw-pointer and gather entry points have explicit unsafe contracts.
The FP16 MoE path is separate from the block-scaled NVFP4 pipeline below.
It requires Tile IR 13.4 or newer for explicit inbounds TMA loads; its host
example and GPU regression skip only the MoE variants on older assemblers.

Choose a variant for its operation and layout. The FP32-output GEMMs are not
drop-in replacements for the FP16-output GEMM. Single-tile and narrow-tail
attention target short sequences; streaming attention bounds the working set
for longer sequences. Direct-gather MoE avoids input materialization, while
packed input adds a stage that can pay off through reuse. Measure the complete
pipeline, including scratch initialization and output combination. Example
configurations are starting points, not a restricted set of tuning choices.

From `cutile-rs/`, run the [host example](examples/kernel_variants.rs):

```sh
cargo run --release -p cutile-kernels --example kernel_variants -- all
# Or select gemm, attention or moe instead of all.
```

The example compares synthetic inputs against CPU references and exercises tail
tiles, empty experts, skewed routing and packed/unpacked execution. It does not
report benchmark rankings. Its host code is shared with the GPU regression
suite, while kernel implementations stay in this crate's `src/experimental/`.
Use stock CUDA 13.5 tooling for evaluation; available schedules depend on the
GPU. See the book's [performance guide](../cutile-book/guide/performance.md) for
choosing and measuring configurations.

```sh
cargo test -p cutile-kernels --test variants_gpu -- --ignored --test-threads=1
CUTILE_JIT_SANITIZE=memcheck compute-sanitizer --tool memcheck --error-exitcode 1 \
  target/release/examples/kernel_variants all
```

The sanitizer command requires the example binary built above and matching
CUDA Toolkit tools. Sanitized execution is for correctness, not timing.

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
[validation instructions](../cutile-examples/README.md#validation) and
[known SM120 limitation](../cutile-examples/README.md#known-sm120-limitation)
before using its results. Opt-in B200 launch profiles and the large synthetic fixture
are documented in the [host example guide](../cutile-examples/README.md#b200-reference-profiles).

## Fused QKV

`experimental::fused_qkv` fuses FP16 QKV projection, FP32 per-head Q/K RMSNorm,
FP8 conversion, Q storage, and paged K/V scatter. V is not normalized. The
[runnable example and benchmark guide](../cutile-examples/fused_qkv.md) documents
the unsafe buffer/slot contract, full-output correctness fixture, and opt-in
CUDA graph timing sweep. Its opt-in `--profile b200` schedules are intended for
evaluation with CUDA 13.5 tooling; compiler support for individual schedules can
vary. Defaults retain the original schedule. Overlap is not guaranteed.
The adapted TileGym traversal retains its MIT license in the kernel source.

## Validation

From `cutile-rs/`, `cargo test -p cutile-kernels --test smoke` lowers the kernel
entry points to Tile IR and serializes bytecode. It does not launch GPU kernels.
NVFP4 compilation is checked for both SM100 and SM120 with explicit Tile IR
13.3 and 13.4 bytecode versions, independent of the selected assembler.
QKV compilation covers SM90/100/120 and the minimum supported formats as well
as SM100 with 13.5 bytecode. Its default example checks direct and graph launches.

`cargo test -p cutile-kernels --test gpu -- --ignored --test-threads=1` compares
ordinary/split-K prefill and persistent RMSNorm against CPU references on a CUDA
GPU. It covers all four prefill schedules, tail rows, cached prefixes, independent
K/V reserved capacities, empty splits, and multiple grid strides. Both suites
are included in the corresponding
`scripts/run_cpu_tests.sh` / `scripts/run_gpu_tests.sh` runners. The NVFP4 example
validates the complete MoE pipeline on supported devices.
