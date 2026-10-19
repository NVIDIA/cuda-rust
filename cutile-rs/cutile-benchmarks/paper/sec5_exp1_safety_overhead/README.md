# Experiment 1: Safety Overhead

**Paper section**: §5.1
**Claim**: safe disjoint bounded partition indices add no measurable
overhead relative to raw-pointer persistent kernels while remaining
competitive with cuTile Python and cuBLAS.

The paper's microbenchmark results were collected on NVIDIA B200. The plots
use the B200 result files in this directory. RTX 5090 reproduction
scripts are retained, but their generated CSVs are not part of the checked-in
paper artifact. Hardware details and nominal memory bandwidth inputs live in
[`../MACHINE.md`](../MACHINE.md).

## Layout

```
sec5_exp1_safety_overhead/
├── paper/                 # paper-facing run scripts, Python benchmarks, plot scripts
│   └── results/
│       └── b200/          # committed B200 CSVs consumed by paper plots
├── rust/
│   ├── gemm/              # Rust GEMM benchmark crate
│   └── elemwise/          # Rust elementwise benchmark crate
├── tuning/                # tile-shape and optimization-hint sweep scripts
├── diagnostics/           # JIT, swizzle, IR, and bandwidth diagnostics
└── scripts/               # shared local wrappers, e.g. clock locking
```

Shared benchmark helpers live one level up in [`../tools`](../tools). In
particular, the nominal bandwidth helper is:

```bash
python3 ../tools/query_nominal_memory_bandwidth.py --device 0 --markdown
```

## Paper Runs

Run from this directory to generate the B200 results:

```bash
./paper/run_b200_persistent_gemm_elemwise.sh
```

It writes to [`paper/results/b200`](paper/results/b200). The plot scripts
default to that B200 bundle.

Paper result CSVs are committed under `paper/results`. Keep raw tuning sweeps,
diagnostics, and scratch reruns under `tuning/results` or
`diagnostics/results`; those paths are ignored locally.

For smoke tests or local reruns that must not replace the paper data, redirect
outputs explicitly:

```bash
RESULTS_DIR=/tmp/cutile-paper-exp1 FIGURES_DIR=/tmp/cutile-paper-figures PY=/path/to/python3 ./paper/run_b200_persistent_gemm_elemwise.sh
python3 paper/plot_exp1.py --target b200 --out /tmp/exp1_safety_overhead.pdf
python3 paper/plot_exp1_elemwise.py --target b200 --out /tmp/exp1_elemwise.pdf
```

B200 paper CSVs:

- `paper/results/b200/gemm_rust_persistent_safe_results.csv`
- `paper/results/b200/gemm_rust_persistent_raw_results.csv`
- `paper/results/b200/gemm_python_persistent_results.csv`
- `paper/results/b200/gemm_cublas_results.csv`
- `paper/results/b200/gemm_cublas_fraction.csv`
- `paper/results/b200/elemwise_rust_runtime_results.csv`
- `paper/results/b200/elemwise_python_results.csv`
- `paper/results/b200/metadata.csv`

Generated plots:

- `figures/generated/exp1_safety_overhead.pdf`
- `figures/generated/exp1_elemwise.pdf`
- `figures/generated/exp1_jit_breakdown.pdf`

`RESULTS_DIR=/path/to/dir` can be set to redirect paper CSV output, but the
default should be used for paper regeneration.

The paper states that the B200 safety-overhead microbenchmarks lock SM clocks
for reproducibility. The B200 runner uses an 1830 MHz SM clock target, records
clock metadata in `metadata.csv`, and can be made strict with
`REQUIRE_CLOCK_LOCK=1`.

For B200 dense FP16/BF16 GEMM SoL, use 2250 TFLOP/s per GPU. NVIDIA's
HGX B200 table reports 36 PFLOP/s FP16/BF16 Tensor Core for an 8-GPU
HGX B200 in sparse mode, and the same table notes dense is half sparse:
36 / 2 / 8 = 2.25 PFLOP/s per GPU. In the committed B200 bundle, the
paper-highlighted M=N=K=8192 row has the mapped safe persistent Rust
kernel at 2068.88 TFLOP/s (92.0% of dense SoL, 96.35% of cuBLAS).
The raw-pointer Rust variant reaches 2074.08 TFLOP/s, so the safe
mapped kernel is within 0.3% of the raw-pointer baseline. The comparable
cuTile Python persistent result reaches 2037.83 TFLOP/s (90.6% of
dense SoL, 94.90% of cuBLAS), and the logged cuBLASLt baseline reaches
2147.29 TFLOP/s (95.4% of dense SoL). At M=N=K=32768, the committed
CSV reports 2121.46 TFLOP/s for safe Rust, 2072.37 TFLOP/s for raw
Rust, 2111.20 TFLOP/s for cuTile Python, and 2187.30 TFLOP/s for
cuBLASLt. The paper compares safe and raw-pointer overhead using the 8192 row.
Source:
<https://www.nvidia.com/en-us/data-center/hgx/>.

Generate the B200 plots from the committed B200 result bundle:

```bash
python3 paper/plot_exp1.py --target b200
python3 paper/plot_exp1_elemwise.py --target b200
```

## Paper Code Audit

The `paper/` directory contains the code paths that directly contribute to
paper results:

| File | Paper role |
|---|---|
| `run_b200_persistent_gemm_elemwise.sh` | B200 persistent GEMM + elementwise paper runner |
| `gemm_python_persistent.py` | B200 cuTile Python persistent GEMM baseline |
| `elemwise_python.py` | cuTile Python elementwise baseline |
| `summarize_gemm_cublas_fraction.py` | cuTile/cublas fraction tables |
| `plot_exp1.py` | GEMM plot; defaults to B200 |
| `plot_exp1_elemwise.py` | Elementwise plot; defaults to B200 |

The B200 runner calls `diagnostics/run_cublaslt_cpp_gemm.sh` for the direct
C++ cuBLASLt baseline, but only the curated `gemm_cublas_results.csv` copied
into `paper/results/b200` is part of the checked-in paper result bundle. Raw
cuBLASLt heuristic dumps are local diagnostics and are not included.

## Kernel Variants

| Config | Frontend | Safety surface | Source |
|---|---|---|---|
| Rust `persistent_safe` | cuTile Rust | mapped partition, checked safe API | `rust/gemm/src/main.rs` |
| Rust `persistent_raw` | cuTile Rust | raw-pointer output access | `rust/gemm/src/main.rs` |
| Python `persistent` | cuTile Python | Python persistent baseline | `paper/gemm_python_persistent.py` |
| cuBLAS | cuBLASLt C++ | baseline denominator | `diagnostics/run_cublaslt_cpp_gemm.sh` |

JIT timing, legacy static/dynamic/unchecked sweeps, full-output diagnostics,
and swizzle probes are not part of the paper throughput path.

## Fixed Tile Schedule

Rust and Python use the same tile shapes and `num_cta_in_cga`/`num_ctas=2`
for paper GEMM runs. Occupancy is left unset.

| M=N=K | (BM, BN, BK) | CTA hint |
|---:|---|---:|
| 1024 | (128, 128, 64) | 2 |
| 2048 | (256, 256, 128) | 2 |
| 4096 | (128, 256, 128) | 2 |
| 8192 | (128, 256, 128) | 2 |
| 16384 | (256, 256, 128) | 2 |
| 32768 | (128, 256, 128) | 2 |

Elementwise uses a shared Rust/Python schedule: `B=16384` with the CTA
hint left unset/default. The paper runners invoke each Rust variant as a
separate single-variant process so optimized/safe use the same measurement
structure as the Python single-kernel baseline: independent allocations,
warmup, fixed-iteration timed windows, and cache-rotation chunks. Set `ELEM_B` and
`ELEM_NUM_CTAS` to compare an explicit architecture-specific CTA hint.
Rust-only `max_divisibility` is left unset in the cross-frontend paper
script so the compared kernels use the same algorithmic tile schedule.

## Tuning

Tuning scripts are opt-in and write to `tuning/results` by default. They do
not update the paper CSVs unless you explicitly re-run the paper scripts after
choosing a configuration. Do not commit raw sweep CSVs.

```bash
./tuning/run_rust_tuning_sweep.sh gemm
./tuning/run_rust_tuning_sweep.sh elemwise
```

The sweep runs in two stages:

1. Tune tile shape with default optimization hints.
2. Pick the best tile shape and tune optimization hints.

Common controls:

- `GEMM_VARIANTS="full-output optimized static"`
- `GEMM_SIZES="1024 2048 4096 8192 16384 32768"`
- `GEMM_TILE_HINTS="2,default,default 1,1,default"`
- `GEMM_HINT_SWEEP=0`
- `FAST=1`
- `MAX_HINT=8`

For locked-clock static sweeps matching the optimized sweep shape:

```bash
./tuning/run_static_gemm_locked_sweep.sh
```

## Diagnostics

Diagnostics are separate from the paper throughput measurements. The only
checked-in diagnostic output is
`diagnostics/results/jit/gemm_rust_jit_timing.log`, which is read by
`diagnostics/plot_jit_breakdown.py` to regenerate the paper JIT panel.

```bash
./diagnostics/run_jit_breakdown.sh
./diagnostics/run_python_gemms.sh swizzle
./diagnostics/run_python_gemms.sh sweep
```

Diagnostics write regenerated outputs under `diagnostics/results` by default.
Do not commit raw diagnostic outputs unless they are consumed by a paper plot.

The empirical memory-bandwidth script is diagnostic only:

```bash
python3 diagnostics/measure_memory_roofline.py --method cuda-vector-copy
```

The paper-facing elementwise roofline uses nominal device-memory bandwidth
queried from local NVML fields through `../tools/query_nominal_memory_bandwidth.py`.

## Static Variant Note

The static GEMM variant keeps static shape information in the front end so
the compiler can eliminate bounds checks. Its kernel body mirrors the unchecked
and dynamic GEMM bodies, including the `K / BK` loop bound, so the throughput
comparison does not include an algorithmic loop-bound difference.

The persistent GEMM paths keep static M/N/K shape information where available.
The default B200 persistent path uses the safe mapped scheduler; the raw-pointer
variant is kept as a diagnostic reference for compiler/codegen comparisons.
