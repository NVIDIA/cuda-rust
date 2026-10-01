# Benchmarks

GPU GEMM benchmarks for measuring Speed-of-Light (SoL) performance on
Blackwell GPUs.

| Script                 | What it measures                       | Dependencies              |
|------------------------|----------------------------------------|---------------------------|
| `cublaslt_bench.c`     | Live heuristic cuBLASLt GEMM reference | CUDA toolkit (C compiler) |
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
width, but its final conversion differs from the kernel. The benchmark also
prints a BF16-input/BF16-output section for context.

Historical throughput from another GPU, driver, or toolkit is not a valid
baseline. Build and run the benchmark on the same host as every kernel trial;
the accepted final measurements are recorded in `../README.md`.

## Requirements

- **GPU**: Datacenter Blackwell. Verified on the current B300 (sm_103a).
- **CUDA Toolkit**: 13.0+ at build time, plus a compatible NVIDIA driver at runtime
- **Python**: 3.12+ (for Python benchmarks only)

## Setup

### cublasLt benchmark (recommended, no Python deps)

The packaged `build.sh` figures out CUDA paths (honoring `CUDA_HOME` /
`CUDA_PATH`, then falling back to `/usr/local/cuda`) and rpath-pins the
toolkit libraries ahead of any unrelated CUDA installation inherited through
`LD_LIBRARY_PATH`. Enter `nix develop path:.` at the repository root; a system
CTK also needs a compatible driver library on its runtime path:

```bash
cd bench/
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

Tests BF16 and FP16 regular GEMM with FP32 compute at 4K, 8K, and 16K
using TN format and heuristic algorithm selection. The harness consumes only
the FP16 section described above.

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
- Warmup: 10 iterations. Timed: 100 iterations.
- The `venv/` directory is gitignored — each machine creates its own.
- The compiled `cublaslt_bench` binary is gitignored — rebuild from
  `cublaslt_bench.c` on each machine.
