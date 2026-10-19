# Benchmarks

GPU GEMM benchmarks for measuring Speed-of-Light (SoL) performance on
Blackwell GPUs.

| Script                  | What it measures                           | Dependencies              |
|-------------------------|--------------------------------------------|---------------------------|
| `cublaslt_bench.c`      | **cublasLtMatmul** GEMM ceiling (correct!) | CUDA toolkit (C compiler) |
| `cublas_sol_bench.py`   | cublasGemmEx GEMM (legacy, ~2.5x slower)   | numpy, CUDA toolkit       |
| `cutlass_sol_bench.py`  | CUTLASS CuTe DSL FP16 GEMM throughput      | nvidia-cutlass-dsl, torch |

## IMPORTANT: cublasGemmEx vs cublasLtMatmul

On Blackwell, **`cublasLtMatmul` is ~2.5x faster than `cublasGemmEx`**
for FP16/BF16 GEMM. This is because cublasLt uses heuristic-based
algorithm selection that picks Blackwell-optimized kernels (2SM MMA,
CLC-based persistent kernels), while `cublasGemmEx` with
`CUBLAS_GEMM_DEFAULT_TENSOR_OP` selects legacy algorithms.

Modular's `vendor_blas.matmul` (their cuBLAS reference) uses
`cublasLtMatmul` with `cublasLtMatmulAlgoGetHeuristic` — the same API.
**Use `cublaslt_bench.c` as the correct baseline, not `cublas_sol_bench.py`.**

Results on B200 (sm_100, 148 SMs, CUDA 13.1):

| API             | BF16 4K     | BF16 8K     | BF16 16K    | FP16 4K     | FP16 8K     | FP16 16K    |
|:----------------|:------------|:------------|:------------|:------------|:------------|:------------|
| cublasLtMatmul  | 1470 TFLOPS | 1389 TFLOPS | 1515 TFLOPS | 1502 TFLOPS | 1402 TFLOPS | 1526 TFLOPS |
| cublasGemmEx    |  528 TFLOPS |  626 TFLOPS |  692 TFLOPS |  534 TFLOPS |     —       |     —       |

Our gemm_sol uses FP16 input / FP32 accumulate, so the FP16 columns are
the correct baseline: **4K=1502, 8K=1402, 16K=1526 TFLOPS**.

## Requirements

- **GPU**: Blackwell (sm_100+). Tested on B200.
- **CUDA Toolkit**: 13.0+ (needs `libcudart.so`, `libcublasLt.so` on `LD_LIBRARY_PATH`)
- **Python**: 3.12+ (for Python benchmarks only)

## Setup

### cublasLt benchmark (recommended, no Python deps)

The packaged `build.sh` figures out CUDA paths (honoring `CUDA_HOME` /
`CUDA_PATH`, then falling back to `/usr/local/cuda`) and rpath-pins the
right lib directory. Use it inside `nix develop` or with a system CTK:

```bash
cd bench/
bash build.sh
./cublaslt_bench
```

`gemm_sol/src/main.rs` will pick this binary up automatically and use it as
its live cublasLt baseline (replacing the previously hardcoded B200
constants). The raw `gcc` invocation still works if you prefer:

```bash
gcc -O2 -o cublaslt_bench cublaslt_bench.c \
    -I"$CUDA_HOME/include" -L"$CUDA_HOME/lib" \
    -lcublasLt -lcudart -lm \
    -Wl,-rpath,"$CUDA_HOME/lib"
```

### cuBLAS (legacy) + CUTLASS benchmarks

```bash
cd bench/
python3 -m venv venv
source venv/bin/activate
pip install numpy                          # for cublas_sol_bench.py
pip install nvidia-cutlass-dsl torch       # for cutlass_sol_bench.py
```

## Running

### cublasLt SoL (correct baseline)

```bash
./cublaslt_bench
```

Tests BF16 and FP16 GEMM with FP32 compute at 4K, 8K, and 16K using
`cublasLtMatmul` with TN format and heuristic algorithm selection.
This is the same API path that Modular and production GEMM
implementations use.

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
