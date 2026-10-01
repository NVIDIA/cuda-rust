# gemm_sol_final

Canonical cuda-oxide Blackwell GEMM speed-of-light example. It packages the
best fully validated kernel.

The goal is simple: show how close a Rust/cuda-oxide kernel can get to the
closest supported live cuBLASLt GEMM on the same GPU, without weakening the
workload or correctness check.

## Final design

```text
KernelFamily: gemm_sol_final/output_tile@1
├── M256xN256 -> existing small resource envelope
└── M512xN256 -> existing large resource envelope

Auto selector:
4096 / 8192  -> M256xN256
16384        -> M512xN256
```

Both entries use CLC work scheduling, a two-CTA cluster, `cta_group::2` UMMA,
four TMA/shared-memory pipeline stages, two TMEM accumulator halves,
compiler-directed K-loop unrolling, and L2-aware output-tile ordering.

## What moved performance

### 1. Produce more output per cooperative CTA pair

```text
Earlier:  M256xN128   [============]
Final:    M256xN256   [========================]
                         2x output columns
```

Widening the cooperative tile to M256xN256 was the dominant optimization:
**+50.06% geomean** in the originating fixed-size experiment.

### 2. Give 16K a larger M tile

```text
4K / 8K:  [ M256xN256 ]
16K:      [ M256xN256 ]
           [ M256xN256 ]  -> one M512xN256 macro tile, shared B work
```

The dedicated M512xN256 entry improved 16384 by **5.70%** while the smaller
sizes retained their better M256xN256 resource envelope.

Eligibility and preference stay separate:

```text
Eligibility: can this compiled entry safely run M×N×K?
Preference:  which eligible entry measured best for this size?
```

Both variants are valid for the fixed 4K size, so a manual override can compare
them. The automatic selector keeps the accepted performance threshold without
hard-coding dispatch inside the timed loop.

The family revision is part of its cache identity. A future change to this
threshold or tuning methodology must bump revision `1`.

### 3. Drain the epilogue with wider stores

Each `u32` contains two BF16 outputs:

```text
Before: [C0 C1] -> STG.32    [C2 C3] -> STG.32
Final:  [C0 C1 C2 C3]        -> STG.E.64
```

Aligned 64-bit stores improved the large 16K path by **4.90%** and the small
4K path by **2.03%** in their respective accepted trials.

### 4. Tune—not introduce—the existing L2 ordering

The cache-blocking tile remap already existed. The accepted work only tuned its
physical N-band width by problem size. After N tiles widened from 128 to 256,
logical `G=2` at 4K and `G=8` at larger sizes preserve 512- and 2048-column
bands respectively.

## Accepted B300 provenance

With the fixed 10-warmup/100-iteration protocol, the accepted checkpoint
measured:

| M=N=K | TFLOPS |
|---:|---:|
| 4096 | 1524.171 |
| 8192 | 1868.354 |
| 16384 | 2161.894 |
| Geomean | 1832.775 |

These numbers describe the originating B300/toolchain and are not hardcoded as
a performance claim for other systems. The example reports fresh timings and
a live cuBLASLt reference on the host GPU.

## Run

From the cuda-oxide repository root:

```bash
# Full 16,777,216-output exact-BF16 checks for both entry points.
GEMM_SOL_MODE=validate cargo oxide run gemm_sol_final

# Fixed 4096/8192/16384 benchmark; 16K dispatches to M512xN256.
# Defaults: 2000 warmup, median of five 301-launch CUDA-event batches.
GEMM_SOL_MODE=bench cargo oxide run gemm_sol_final

# Print the same dispatch decisions without opening a CUDA context.
GEMM_SOL_MODE=plan cargo oxide run gemm_sol_final

# Override the automatic policy with a validated, stable variant ID.
GEMM_SOL_MODE=plan GEMM_SOL_VARIANT=m512xn256 cargo oxide run gemm_sol_final
```

`GEMM_SOL_MODE=both` is the default. This example requires datacenter
Blackwell `tcgen05`, CLC, and `cta_group::2` support. On unsupported GPUs it
still verifies that PTX generation succeeded.

`GEMM_SOL_VARIANT` accepts `auto` (the default), `m256xn256`, or
`m512xn256`. Forced choices bypass the performance selector but still pass the
kernel shape contract; an unsafe shape is rejected before launch.

## Live cuBLASLt reference

Build the helper inside the same CUDA development environment used to run the
example; rebuilding avoids accidentally comparing against a stale cuBLASLt
version:

```bash
cd crates/rustc-codegen-cuda/examples/gemm_sol_final/bench
bash build.sh
```

The target writes BF16 while the closest supported reference uses FP16 input,
FP32 compute, and FP16 output. The output conversion differs, but both paths
write two bytes per element. The helper defaults to timing and selecting among cuBLASLt's top eight
heuristic candidates. Run both that mode and the wider configuration search:

```bash
./cublaslt_bench --mode both
```

The exhaustive-grid mode tests every returned algorithm ID and its exposed
tile, stage, cluster, custom-option, swizzle, and supported reduction settings,
with the finite split-K grid `0,1,2,3,4,5,6,8,12,16,32`. Unlike NVIDIA's
illustrative custom-find sample, it does not stop after 16 IDs or 100 successful
configurations. This is an exhaustive search of the stated grid, not a claim
that every possible cuBLASLt implementation or split-K integer was explored.
It retains the heuristic candidates in final selection.

Both the Rust kernel and the helper default to 2,000 warmup launches, then five
independent CUDA-event batches of 301 launches. Reported throughput uses the
median batch mean, with tuning and validation excluded. Timing inputs are
zeroed in both paths. Each selected cuBLASLt algorithm additionally passes 64
nonzero CPU dot-product checks at each measured size; the Rust full-output
validator checks both kernel variants at 4096³. Those 4K checks do not establish
full-output correctness at 8K or 16K.

Rust timing overrides are `GEMM_SOL_WARMUP`, `GEMM_SOL_ITERS`, and
`GEMM_SOL_REPEATS` (positive odd repeat count); they are forwarded to the live
helper. `GEMM_CUBLAS_MODE=top8|exhaustive|both` selects its search mode.
`GEMM_CUBLAS_SKIP=1` omits the helper for a separately collected comparison.
The helper accepts corresponding `--warmup`, `--iters`, and `--repeats`
options, `--n` for one square size, and emits `RESULT` JSON lines including raw
batch timings, selected configuration, workspace, and validation scope. Results
are reported per shape without aggregating the three shapes into a geomean.
See [bench/README.md](bench/README.md) for search and output-layout details.
