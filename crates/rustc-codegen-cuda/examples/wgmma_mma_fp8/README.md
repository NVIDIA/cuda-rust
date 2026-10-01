# FP8 WGMMA

This runnable Hopper example checks the complete cuda-oxide path for:

```text
wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3
```

It also measures the existing BF16 `m64n64k16` path under the same effective
matrix workload.

## Run

```bash
cargo oxide run wgmma_mma_fp8 --arch sm_90a
```

Execution requires an H100 or H200. On another architecture the binary checks
the generated PTX for the FP8 and BF16 instructions and exits without printing
the runtime `SUCCESS` marker.

Pass `-- --check-only` to run the all-output checks and one launch of each
benchmark kernel, without warmups or timing. This mode fails on a non-Hopper GPU.

Run correctness and Compute Sanitizer checks from the repository root:

```bash
cargo oxide run wgmma_mma_fp8 --arch sm_90a -- --check-only
cargo oxide sanitize wgmma_mma_fp8 --arch sm_90a --lineinfo --tool memcheck -- --error-exitcode 86 --print-limit 0 --check-warpgroup-mma yes -- --check-only
cargo oxide sanitize wgmma_mma_fp8 --arch sm_90a --lineinfo --tool racecheck -- --error-exitcode 86 --print-limit 0 --racecheck-report all --print-level info -- --check-only
cargo oxide sanitize wgmma_mma_fp8 --arch sm_90a --lineinfo --tool initcheck -- --error-exitcode 86 --print-limit 0 -- --check-only
cargo oxide sanitize wgmma_mma_fp8 --arch sm_90a --lineinfo --tool synccheck -- --error-exitcode 86 --print-limit 0 --check-warpgroup-mma yes -- --check-only
```

Each run should print `SUCCESS`; sanitizer summaries should report zero errors
or hazards. Collect timings separately with the normal run command above,
without sanitizer instrumentation or `CUDA_OXIDE_DEBUG` enabled. Include the
GPU model, driver and CUDA toolkit versions when reporting results.

## Numeric checks

The host creates small integers in `[-3, 3]`, encodes them directly as E4M3 or
BF16, and copies their raw bytes to the device. Both formats represent these
values exactly. Each kernel uses one 4096-byte, 256-byte-aligned shared buffer:

- A occupies bytes `0..2048`; B occupies bytes `2048..4096`.
- Both operands are K-major with a 32-byte K span.
- Logical byte offset `row * 32 + byte` is stored at
  `Swizzle::<1, 4, 7>::apply(offset)`, matching the descriptor's 32-byte
  swizzle.
- Every writer issues `fence.proxy.async.shared::cta`, then the CTA executes
  `sync_threads`, before WGMMA reads shared memory through the async proxy.

The accumulator starts at `1.0`, so the check covers the input accumulator as
well as the matrix product. After `commit_group` and `wait_group<0>`, each of
the 128 threads stores its 32 accumulator registers contiguously. The host
scatters those fragments into the 64x64 result layout and compares every value
exactly with an F32 reference decoded from the raw input bytes. A one-MMA BF16
result is checked the same way.

Exact comparison is intentional for this dataset: products are integers with
absolute value at most 6, and even the effective-K=64 benchmark's partial sums
are bounded by `1 + 64 * 6 = 385`. These stay within the exact-integer range
of half precision, below the documented FP8 WGMMA accumulation precision.
Failures report the mismatch count, maximum absolute error, and first mismatch;
non-finite mismatches report an infinite error.

## Benchmark

The benchmark is an end-to-end tile microbenchmark, not a large GEMM or a peak
throughput claim. Every launch includes global-to-shared staging, proxy and CTA
synchronization, WGMMA, and one checksum store per CTA.

Both variants compute the same effective `64x64x64` workload from the same
logical values. The FP8 K=32 tile contains two copies of the base K=16
pattern, with a common cyclic permutation of A and B in the second half.
This preserves the dot product while making missing SW32 half-swaps visible
to the numeric check. The FP8 kernel reuses that shared tile for two MMAs; the BF16 kernel
reuses its K=16 tile for four MMAs. Their decoded F32 references and device
checksums must agree before timing begins.

Timing uses 8192 CTAs, 10 alternating warmups, and 11 alternating samples of
100 CUDA-event-timed launches. The report uses the median average launch time
and counts `2 * 64 * 64 * 64` operations per CTA for both variants.
Raw sample times are printed, and every CTA's checksum is rechecked after timing.

Expected final marker:

```text
SUCCESS: FP8 WGMMA numeric check and BF16 comparison passed
```

## Large-GEMM FP8 versus BF16 comparison

For one command collecting the PR correctness checks, four sanitizers, three
tile benchmark runs, and three large-GEMM sweeps into a shareable archive, use:

```bash
CUDA_TOOLKIT_PATH=/usr/local/cuda-13.0 CUDA_VISIBLE_DEVICES=0 bash scripts/validate-fp8-wgmma.sh
```

See [HOPPER_VALIDATION.md](HOPPER_VALIDATION.md) for prerequisites and archive contents.

From the repository root, on H100/H200 with a CUDA toolkit including cuBLASLt:

```bash
bash scripts/bench-fp8-vs-bf16.sh | tee fp8-vs-bf16.log
```

This builds `bench/fp8_vs_bf16.cu` and sweeps square GEMMs with dimensions
64, 256, 1024, 4096 and 8192, followed by a narrow `64x8192x8192` case,
once with FP8 fast accumulation off and once with it on (12 comparisons).
To choose one M/N/K shape, enable FP8 fast accumulation explicitly, or run
the GPU-independent self-check:

```bash
bash scripts/bench-fp8-vs-bf16.sh 4096 4096 4096
bash scripts/bench-fp8-vs-bf16.sh 4096 4096 4096 1
bash scripts/bench-fp8-vs-bf16.sh --self-test
```

Set `NVCC=/path/to/nvcc` or `CUDA_TOOLKIT_PATH=/path/to/cuda` if needed.
The selected toolkit must include its matching cuBLASLt headers and library.

Both paths use the same logical inputs, TN layouts, FP32 compute, BF16 output,
and a 32 MiB workspace limit. FP8 uses E4M3 and unit input scales. Inputs are
preconverted; timing excludes conversion, allocation, CPU/GPU transfers and
validation. It includes the complete cuBLASLt GEMM, including output writes.
Ten warmups precede seven alternating samples of 50 launches, timed with CUDA
events; results show each sample and the median speedup `BF16_ms / FP8_ms`.
Each format uses the first supported cuBLASLt heuristic, not an exhaustive
algorithm search. Small cases can include host launch submission overhead.

Every output is checked before and after timing against a CPU reference rounded
to BF16, with `atol=0.03125`, `rtol=0.01`, and rejection of nonfinite results.
The inputs are signed, exactly representable values with a 32-element period;
this makes full-output validation cheap but does not test real-model FP8
quantization accuracy. The maximum absolute error is printed. Fast accumulation
defaults to off; turning it on can trade numerical accuracy for speed on Hopper.

Large square GEMMs are candidates for a larger FP8 advantage because optimized
kernels can sustain Tensor Core work and reuse inputs. Size alone guarantees
neither a speedup nor 2x performance. Narrow shapes, staging, synchronization,
output traffic, library algorithm selection and conversion costs can limit it.
The original Rust tile benchmark already launches 8192 CTAs; adding more CTAs
does not turn its short two/four-MMA sequence into a sustained-compute kernel.

This is an independent library experiment, not runtime validation of this PR's
Rust WGMMA lowering or a replacement for its sanitizer results. cuBLASLt chooses
its own kernel; identifying its exact instructions requires profiling/disassembly.
No large-GEMM GPU results are recorded yet.

References: [NVIDIA H100 specifications](https://www.nvidia.com/en-us/data-center/h100/)
and [cuBLASLt documentation](https://docs.nvidia.com/cuda/archive/13.0.0/cublas/index.html).
