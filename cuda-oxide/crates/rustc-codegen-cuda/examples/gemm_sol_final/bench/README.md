# Benchmarks

GPU GEMM benchmarks for measuring Speed-of-Light (SoL) performance on
Blackwell GPUs.

| Script                 | What it measures                       | Dependencies              |
|------------------------|----------------------------------------|---------------------------|
| `cublaslt_bench.c`     | Top-8 and exhaustive-grid tuned cuBLASLt reference | CUDA toolkit (C compiler) |
| `cublas_sol_bench.py`  | Legacy `cublasGemmEx` comparison       | numpy, CUDA toolkit       |
| `cutlass_sol_bench.py` | CUTLASS CuTe DSL FP16 GEMM throughput  | nvidia-cutlass-dsl, torch |

## Reference choice

Use `cublaslt_bench.c` for the optimization score. It runs
`cublasLtMatmulAlgoGetHeuristic` on the same GPU at every fixed size; the
legacy `cublasGemmEx` script is retained only for historical comparison.

The target kernel uses FP16 A/B, FP32 accumulation, and BF16 output. On the
pinned stack, cuBLASLt does not expose that exact mixed input/output
combination, so the
harness parses the benchmark's FP16 A/B, FP32-compute, FP16-output section.
That is the closest supported path with the same input type and two-byte output
width, but its final conversion differs from the kernel. The helper's TN
column-major output also differs in storage order from the Rust row-major
output; all measured problems are square. Both compute A×Bᵀ with contiguous K
inputs. `--output bf16` probes exact output-type support; cuBLASLt 13.8.0 on the
B200 returned `CUBLAS_STATUS_NOT_SUPPORTED` for FP16 inputs and BF16 output.

Historical throughput from another GPU, driver, or toolkit is not a valid
baseline. Build and run the benchmark on the same host as every kernel trial;
the originating B300 checkpoint is recorded in `../README.md`, and the B200
presentation evidence is linked below.

## Requirements

- **GPU**: Datacenter Blackwell. Measured on B200 (sm_100a); originating results used B300 (sm_103a).
- **CUDA Toolkit**: 13.0+ at build time, plus a compatible NVIDIA driver at runtime
- **Python**: 3.12+ (for Python benchmarks only)

## Setup

### cublasLt benchmark (recommended, no Python deps)

The packaged `build.sh` figures out CUDA paths (honoring `CUDA_HOME` /
`CUDA_PATH`, then falling back to `/usr/local/cuda`) and rpath-pins the
toolkit libraries ahead of any unrelated CUDA installation inherited through
`LD_LIBRARY_PATH`. Enter the cuda-oxide Nix shell from the repository root;
a system CTK also needs a compatible driver library on its runtime path:

```bash
cd cuda-oxide
nix develop
cd crates/rustc-codegen-cuda/examples/gemm_sol_final/bench
bash build.sh
./cublaslt_bench
```

`src/main.rs` picks this binary up automatically and uses its live FP16
section as the closest supported reference. Use `build.sh`; it handles both
the Nix toolkit `lib` layout and classic CTK `lib64`.

### cuBLAS (legacy) + CUTLASS benchmarks

```bash
cd bench/
python3 -m venv venv
source venv/bin/activate
pip install numpy                          # for cublas_sol_bench.py
pip install nvidia-cutlass-dsl torch       # for cutlass_sol_bench.py
```

## Running

### Live cuBLASLt reference

```bash
./cublaslt_bench
```

Tests FP16 A/B, FP32 compute, FP16 output at 4K, 8K, and 16K in TN format.
The default `--mode top8` measures all eight heuristic candidates, then selects
by the median of five 31-launch refinement batches after ten warmups per batch.

```bash
# Top-8 and the wider search, for every fixed shape:
./cublaslt_bench --mode both
# One shape with an explicit final timing protocol:
./cublaslt_bench --mode exhaustive --n 8192 --warmup 2000 --iters 301 --repeats 5
```

The wider search follows NVIDIA's
[custom-find sample](https://github.com/NVIDIA/CUDALibrarySamples/blob/main/cuBLASLt/Common/LtMatmulCustomFind.h),
without its 16-ID or 100-result truncation. It enumerates all returned algorithm
IDs, exposed tile/stage lists, cluster shapes, custom options, swizzles, and
supported reductions over the finite split-K grid
`0,1,2,3,4,5,6,8,12,16,32`. Both 0 and 1 mean no split; current algorithms may
require 1. Only configurations that pass `cublasLtMatmulAlgoCheck` and fit the
32 MiB workspace limit are timed. The search is exhaustive over this documented
grid, not over every possible split-K integer. The inner MMA shape uses its
library default, as in NVIDIA's custom-find sample.

Grid screening uses one warmup and five launches per valid configuration.
The fastest 16 screening candidates plus **all** heuristic candidates undergo
five 31-launch refinement batches; the lowest median wins. The winner is
validated on nonzero signed 1/8 data using 64 CPU dot products at distributed
coordinates, then remeasured from fresh zeroed inputs with the final protocol.
The final number is not the minimum tuning sample. Every matmul return status
is checked; failed final timing or correctness checks prevent a result.

`RESULT` JSON lines preserve the five raw final batch means, selected algorithm
configuration, workspace usage, search counts, and validation scope. cuBLASLt
only needs the winning configuration after tuning, so search time is excluded
from steady-state throughput. The Rust consumer defaults to top-8 tuning and
forwards any `GEMM_SOL_WARMUP`, `GEMM_SOL_ITERS`, or `GEMM_SOL_REPEATS` override.
Set `GEMM_CUBLAS_MODE` to `top8`, `exhaustive`, or `both` to change its mode;
`GEMM_CUBLAS_SKIP=1` supports collecting the two implementations separately.
When the helper runs in `both` mode, the Rust summary consumes the top-8 row;
the standalone `RESULT` records retain both independently measured results.

### cuBLAS (legacy) SoL

```bash
source venv/bin/activate
python cublas_sol_bench.py
```

Uses `cublasGemmEx` — significantly slower on Blackwell. Kept for
historical comparison only.

### CUTLASS SoL

```bash
cd bench/
source venv/bin/activate
PYTHONPATH=. python cutlass_sol_bench.py
```

Tests FP16 GEMM at 4K and 8K using a tcgen05 MMA kernel with
software-pipelined K-loop and TMA loads.

## Notes

- All benchmarks use GPU-side timing (CUDA events), not wall-clock.
- Rust and cuBLASLt defaults: 2,000 warmup launches followed by five independent
  batches of 301 launches, reported as median batch mean. Timing uses zero data.
- GPU clocks remain under the driver's control; record GPU model, SM count,
  driver/toolkit/library versions, power limit, and other active GPU workloads.
- Historical B300 results and legacy Python scripts retain their original
  10-warmup/100-iteration protocol; do not mix them with current measurements.
- The `venv/` directory is gitignored — each machine creates its own.
- The compiled `cublaslt_bench` binary is gitignored — rebuild from
  `cublaslt_bench.c` on each machine.

## B200 presentation evidence (2026-10-01)

[Comparison JSON](results/b200-2026-10-01/comparison.json) records the three
individual shapes, top-8 and exhaustive-grid results, five raw final timing
samples, algorithm configurations, and complete methodology/provenance.
[Raw cuBLASLt output](results/b200-2026-10-01/cublas-final.log) and
[raw Rust output](results/b200-2026-10-01/oxide.log) retain search counts and
correctness results. [Reproduction commands](results/b200-2026-10-01/reproduce.sh)
use the matching final timing protocol. Source hashes and the benchmark patch
identify the uncommitted code used for this run; clocks were not locked.
