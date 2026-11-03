# MXFP4 GEMM with C++ instruction primitives

This CUDA C++ example implements the same fixed SM120a GEMM as
[`blockscale_gemm_ctm`](../../../blockscale_gemm_ctm): M=N=2048, K=1024,
packed E2M1 inputs, UE8M0 scales per 32 K values, FP32 accumulators, and FP16 output.
It keeps the 128×128×128 tile, three-stage TMA ring, eight compute warps and one
producer warp, persistent tile schedule, and STMATRIX/TMA epilogue. The five
tensor maps are passed by value with `__grid_constant__`.

The kernel uses narrow inline-PTX wrappers and explicit physical addresses;
it requires no CuTe algebra or CUTLASS headers. A/B fragments and scale carriers
have distinct C++ types. This is a new C++ counterpart to the Rust example,
not an upstream CUTLASS benchmark.

Build with a CUDA toolkit supporting SM120a and a C++17 host compiler:

```sh
python3 build.py --output-dir /tmp/blockscale-cpp
```

The script runs CPU layout checks and compiles the executable in a fresh
`run-*` directory. Its manifest records exact commands, source/binary hashes,
compiler versions, retained PTX/cubin intermediates, and resource usage. The
explicit `-gencode=arch=compute_120a,code=sm_120a` target avoids a generic SM120
fallback, which cannot express the architecture-specific blockscale instruction.
`--nvcc` and `--cxx` accept executable paths or names on `PATH`; `--cxx` selects
the compiler for both CPU tests and nvcc's host compilation.

Use the existing shared validation harness with the reported binary path:

```sh
CUDA_VISIBLE_DEVICES=1 python3 ../../../blockscale_gemm_ctm/validate.py \
  --binary /tmp/blockscale-cpp/run-*/blockscale_gemm_cpp \
  --output-dir /tmp/blockscale-cpp-validation --sanitizer
```

Replace the wildcard with one exact run directory if multiple builds exist.
Each process uses the same deterministic input fixture as Rust, computes an
independent dense CPU oracle, checks all 4,194,304 FP16 bit patterns from poisoned
output, and rechecks after repeated launches. Validation covers the resident
grid, forced one-CTA and 17-CTA grids, memcheck, and synccheck. CPU tests check
physical-layout coverage, alignment, and nonaliasing; they do not require a GPU.

The executable accepts `--ctas`, `--warmup`, `--samples`, and `--json`. Defaults
are the occupancy-based resident grid, 100 warmups, and 31 single-kernel CUDA
event samples. Use the same explicit CTA count for comparisons (170 on the
tested RTX 5090). Compare Nsight **kernel-active** durations with Rust under the
same protocol; CUDA event intervals include submission gaps and are a separate
metric. Compiler versions and generated resource usage remain relevant to the
comparison. No SOL claim is made.

## Compare with the Rust CTM

Build both executables freshly, then run the exact-oracle comparison runner:

```sh
python3 ../../../blockscale_gemm_ctm/compare.py \
  --binary rust=/path/to/blockscale_gemm_ctm \
  --binary cpp=/path/to/blockscale_gemm_cpp \
  --output-dir /tmp/blockscale-comparison --device 1 \
  --ctas 170 --rounds 3 --warmup 100 --samples 31
```

The runner requires 132 kernel launches and full correctness in every timing
process and reports Nsight kernel-active time, excluding host submission gaps.
