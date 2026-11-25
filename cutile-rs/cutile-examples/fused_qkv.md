# Fused QKV projection and FP8 epilogue

The [`fused_qkv` example](examples/fused_qkv.rs) uses
[`cutile-kernels::experimental::fused_qkv`](../cutile-kernels/src/experimental/qkv.rs)
to compute FP16 `X[M,K] @ W[N,K].T` with FP32 accumulation, per-head Q/K RMSNorm,
FP8 quantization, a contiguous Q output, and paged K/V cache stores in one kernel.
V is quantized and stored without normalization. There is no FP16 intermediate
rounding, bias, RoPE, residual, attention, or communication.

Use a complete **public CUDA Toolkit 13.5** installation for evaluation.
Compiler support for individual opt-in SM100 schedules can vary.
Execution requires FP8 support (SM90+); the normal target/version rules still
apply. The schedules were evaluated on B200; they are not universal optima. Source
fusion does not establish overlap between GEMM and the epilogue. Its grouped-M
traversal and TMA loading derive from TileGym; the kernel retains the MIT notice.

## Run

From `cutile-rs/`, after configuring your toolkit and `CUTILE_TILEIRAS_PATH`:

```sh
# Small full-output correctness check, including direct and graph launches.
cargo run --release -p cutile-examples --example fused_qkv

# Opt-in B200 profiles, with per-sample CUDA-event measurements.
cargo run --release --locked -p cutile-examples --example fused_qkv -- --bench --profile b200

# Same sweep with non-unit gamma/scales and permuted physical cache slots.
cargo run --release --locked -p cutile-examples --example fused_qkv -- --bench --profile b200 --nontrivial

# Inspect dimensions, schedule controls and measurement options without a GPU.
cargo run -p cutile-examples --example fused_qkv -- --help
```

The benchmark defaults to M=128,256,512,1024,2048,4096,8192, K=4096,
32 Q heads, 8 KV heads, and head dimension 128 (N=6144). Override dimensions
with `--m-list`, `--k`, `--q-heads`, and `--kv-heads`. K must be a multiple of
8 so FP16 input/weight row strides meet 16-byte TMA alignment.

With `--profile b200`, the standard benchmark dimensions on SM100 use
BM=128, BN=128 and occupancy=1. BK=128 through M=2048 and BK=64 for larger M.
GROUP_M=1 at M=2048 and 8 otherwise. M=256 uses one CTA per output tile; other
sizes use a persistent grid with one CTA per SM. M=128 uses two CTAs per cluster
(with the grid rounded up to a complete cluster); other sizes use one.

The default profile retains BM=64, BN=128, BK=64, GROUP_M=8, occupancy=1 and
one CTA per cluster on all architectures. `--profile b200` rejects other
architectures/nonstandard shapes and cannot mix with manual schedule controls.
With the default profile, `--bm`, `--bk`, `--group-m` and `--programs` start from
the original defaults. `--bm` and `--bk` accept 16,32,64,128; `--group-m` controls
grouped-M traversal. `--programs 0` launches one CTA per output tile; a positive
count sets a persistent grid. These controls enable evaluation, not an implicit
claim that a configuration is optimal. This kernel does not explicitly implement
the CuTe example's two-CTA, two-accumulator, warp-specialized schedule.

See [B200 setup](README.md#b200-reference-profiles) for matching public toolkit
tools and libraries, and [validation](README.md#validation) for
compiler-instrumented sanitizer commands. Comparing compilers with different
profiles is not an isolated compiler comparison.

## Data and validation

Weights have contiguous `[N,K]` layout, ordered Q, K, V. Each head has 128
elements. Q/K use FP32 gamma vectors of length 128, shared across heads of
their kind, and epsilon=1e-6. Outputs multiply by separate inverse scales,
clamp to [-448,448], and convert to E4M3FN with round-to-nearest-even.

Q is `[M,Q_heads,128]`. Separate contiguous NHD K/V caches are
`[pages,page_size,KV_heads,128]`, flattened to `[capacity,KV_heads*128]`.
Slots are int64 physical token offsets, not page IDs. Negative slots skip KV
writes but still produce Q. Nonnegative slots must be unique and in bounds;
the example checks that contract before launch. A different cache layout needs
an adapter.

The no-argument fixture preserves M=67, K=96, 4 Q heads, 2 KV heads and 96 cache
slots. It exercises M/K tails, a zero row, non-unit and negative gamma, separate
inverse scales, permuted/negative slots and untouched-cache sentinels, with both
ordinary and persistent grids. It checks every Q/K/V element against an
independent FP64 projection/norm and FP8 converter, allowing one neighboring
FP8 spacing for FP32 reduction/rsqrt rounding. Untouched bytes must match exactly.
Output buffers are reset before direct and graph checks.
Inactive tail rows retain their slot-load mask and use an in-bounds pointer
before the load. The fixture remains unpadded and retains its numerical checks
when running compiler-instrumented memcheck.

Benchmark inputs/weights use the same deterministic, nonuniform recipe in the
source, including a zero first input row. By default gamma is one, all inverse
scales are 0.0625, and slots are identity-mapped. `--nontrivial` uses non-unit
gamma, inverse scales 0.0625/0.125/0.25, and a unique permutation; all benchmark
rows still write KV. Cache capacity rounds M+16 up to a multiple of 16.

To keep large sweeps practical, benchmark mode checks all columns of up to eight
evenly spaced rows against the FP64 reference, all untouched cache slots exactly,
and all output elements for finiteness. It prints the selected rows, differing
FP8 codes and maximum absolute errors. These sampled checks are not a replacement
for full-output validation of a shared fixture when comparing implementations.

## Timing and comparisons

Each sample warms up 50 graph launches and times 1,000 replays with CUDA events
on the graph's stream; 20 samples are collected. Configure these with
`--warmups`, `--replays`, and `--samples`. Compilation, allocation, uploads,
capture, validation and output resets are outside timing. The complete fused
kernel is inside the graph. Replays overwrite outputs, so no per-replay reset is
needed. Outputs are checked again after the timing sequence.

The example prints launch/fixture settings, `sample_csv` records containing
every latency sample, and mean/median/min/max microseconds. Save the output plus
the source SHA, actual toolkit/assembler versions, GPU and driver information.

The public
[`cute-rs` QKV example](../../cuda-oxide/cute-rs-preview/examples/sm100_qkv_epilogue_cute/README.md)
includes a Python CuTeDSL reference. Match inputs, gamma, scales, physical cache
layout, slot mapping, shapes and timing boundaries before computing speedups;
matching dimensions alone is insufficient. Run each implementation on the same
GPU and keep profiler runs separate from latency measurements.
