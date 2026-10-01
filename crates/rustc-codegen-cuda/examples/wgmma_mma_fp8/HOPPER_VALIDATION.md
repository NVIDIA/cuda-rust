# H100/H200 run for PR #1306

1. Use a Linux H100/H200 machine with CUDA 13+ (including Compute Sanitizer), driver 580+, and the
   [repo prerequisites](https://nvlabs.github.io/cuda-oxide/getting-started/installation.html).
   Include cuBLASLt headers/libraries in that toolkit. Keep other GPU jobs stopped.
   Allow approximately 20–40 minutes for the first build and PR checks, plus
   2–5 minutes for the additional GEMM sweeps; build and host speeds vary.

2. Use a checkout containing the updated runner and `bench/fp8_vs_bf16.cu`.
   If these changes have been pushed to the branch, clone it with:

   ```bash
   git clone -b feat/fp8-wgmma https://github.com/Vishalkulkarni45/cuda-oxide.git
   cd cuda-oxide
   ```

3. Select the CUDA installation and GPU, then run:

   ```bash
   CUDA_TOOLKIT_PATH=/usr/local/cuda-13.0 CUDA_VISIBLE_DEVICES=0 bash scripts/validate-fp8-wgmma.sh
   ```

4. Send back the printed `fp8-wgmma.*.tar.gz` file, **even if a check fails**.
   A successful archive contains `result.txt` starting with `PASS` and
   `status.txt` with `exit_code=0`. `summary.txt` contains the tile timings and
   all 36 GEMM results with shape, accumulation mode and speedup. Do not use
   sanitizer timings as benchmarks. If the library experiment fails after the
   PR checks, `pr-validation.txt` records that those earlier checks passed.

The archive covers [PR #1306](https://github.com/NVIDIA/cuda-rust/pull/1306)
and the benchmark promised in [issue #1281](https://github.com/NVIDIA/cuda-rust/issues/1281):

| Required evidence | Captured result |
|---|---|
| All-output correctness | All 4,096 FP8 and 4,096 BF16 values checked against decoded F32 references; maximum error and `SUCCESS` |
| Compute Sanitizer | memcheck, racecheck, initcheck, synccheck; WGMMA checks enabled for memcheck/synccheck |
| Comparable timings | Three runs with raw CUDA-event samples, median launch ms, and BF16/FP8 TFLOPS |
| Reproducibility | Commit, local diff, source, GPU/driver/toolkit versions, GPU state before/after, PTX and assembler report |
| Independent large-GEMM comparison | Three cuBLASLt sweeps: five square shapes and one narrow shape, each with FP8 fast accumulation off/on; correctness, raw samples, median ms, TFLOPS, speedup and library version |

Both paths use 8,192 CTAs of 128 threads, the same 64×64×64 logical work,
and 4 KiB shared memory: two FP8 MMAs versus four BF16 MMAs. Each run uses
10 warmups and 11 alternating samples of 100 launches. This measures a tile
microbenchmark including staging and synchronization, not peak GEMM throughput.

The independent cuBLASLt comparison uses preconverted inputs, FP32 compute and
BF16 output. It excludes conversion and transfers. It does not exercise the PR's
Rust lowering. Large square shapes are candidates for higher FP8 speedup, not a
guarantee: `speedup > 1` means FP8 is faster, and `speedup < 1` means BF16 is faster.
Both are valid results. Fast accumulation can trade accuracy for speed; its two
modes are labeled separately. See README.md for the reference and tolerances.

The runner stops on a failed command or missing evidence and prints the archive
path even on failure. CPU-only runner regression check:

```bash
python3 scripts/test-fp8-wgmma-runner.py
```
