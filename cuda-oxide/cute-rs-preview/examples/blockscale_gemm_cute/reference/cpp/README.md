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

## Local correctness and timing (2026-10-11)

Both fresh executables passed all 4,194,304 bit-exact FP16 checks and replay
checks with resident 170-CTA, forced 1-CTA, and forced 17-CTA grids. Both passed
memcheck with tensor-operation checking and synccheck with zero errors.
The C++ CPU physical-layout tests and 12 comparison-runner tests also passed.

The comparison uses physical GPU 1, an RTX 5090, driver 615.71.09, and Nsight
Systems 2026.3.2. Each implementation runs three rounds, serially in rotating
order, with 100 warmups and 31 measured launches per round. Both use 170 CTAs,
hot buffer reuse, no graphs, and unlocked clocks. The reported duration is
Nsight kernel-active time, excluding host submission gaps. The runner requires
exactly 132 kernel launches and full correctness in every timing process.

| Implementation | Round medians (µs) | Median of round medians (µs) |
| --- | --- | --- |
| Rust primitives, CUTLASS compiler 4.7.0 | 11.712, 11.680, 11.648 | 11.680 |
| C++ instruction primitives, nvcc 13.4.92 | 11.072, 11.072, 10.976 | 11.072 |

Rust takes 5.5% longer in this experiment. This compares the two implementations
and compiler pipelines; it does not isolate a language cost. Rust uses 128
registers per thread and C++ uses 136; both have 8 bytes of stack and no local
memory, with 86,016 bytes of dynamic shared memory plus 1,024 bytes of static
alignment overhead per CTA. The differing instruction schedules and compilers
need further investigation before attributing the timing gap.

The Rust build uses `nightly-2026-10-11` after rebasing onto main `d141118c05`.
Its cubin is byte-for-byte identical to the pre-rebase cubin, so the new
parameter attributes have not improved this kernel's generated code. See the
[Rust example](../../../blockscale_gemm_ctm#c-primitives-comparison-after-the-main-rebase-2026-10-11)
for the current attribute-export boundary.

[Recorded results](benchmark-results.json) retain the raw samples, source and
binary hashes, compiler provenance, validation reports, and local trace paths.
To reproduce, build both executables freshly and run:

```sh
python3 ../../../blockscale_gemm_ctm/compare.py \
  --binary rust=/path/to/blockscale_gemm_ctm \
  --binary cpp=/path/to/blockscale_gemm_cpp \
  --output-dir /tmp/blockscale-comparison --device 1 \
  --ctas 170 --rounds 3 --warmup 100 --samples 31
```

The paths in `benchmark-results.json` describe the original measurement checkout
before this reference was relocated. Source and artifact hashes are preserved.
