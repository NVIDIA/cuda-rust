# B200 performance: Rust CuTe, Rust CTM and Python CTM

The [2026-10-18 summary](results/b200-20261018/summary.json) compares both Rust
GEMMs with the same checked-in [Python CTM reference](gemm.py). All six ordinary
comparisons and both profiled comparisons pass the independent PyTorch oracle.
Inputs, reference hashes and cluster counts match. This public reference uses
CUTLASS instruction primitives for the hardware operations and CuTe compilation,
layout and epilogue helpers; see [the API description](README.md).

## Conditions and CUDA-event results

Source revision `a0d5776b2b7a4ecd387c14d1408d879a2ff42201`, with only the
comparison's printed Rust label corrected before measurement. Kernel sources
are unchanged. The reports retain executable, reference and oracle hashes.
The host was `6u1g-0001`, NVIDIA B200, driver 615.71.09, CUDA toolkit 13.4.92,
CUTLASS 4.7.0, Python 3.12.3 and PyTorch 2.14.1+cu130. GPU clocks were not locked;
the power limit was 1,000 W. Builds use the default CUTLASS optimization setting,
with `cutlass-primitives-mlir22` selected for Rust CTM.

All runs use M8192/N8096/K8192, no bias, seed 0, 74 two-CTA clusters, hot buffers,
2,000 warmups and 301 direct CUDA-event samples. Each sample brackets one kernel
on its actual launch stream, then synchronizes. Python runs first, followed by
input export and the Rust subprocess. Rust has one additional preliminary
untimed launch. Compilation, data export and correctness checks are not timed.
No CUDA graphs are used.

The table aggregates three complete comparisons per Rust version: the median
of their three run medians. Runs alternated CuTe and CTM; the Python entry in
each pair belongs to those particular comparison runs, not a different kernel.

| Pair | Implementation | Median (µs) | Range of run medians (µs) | TFLOP/s at median |
| --- | --- | ---: | ---: | ---: |
| CuTe | Rust CuTe | 770.784 | 767.968–781.472 | 1409.768 |
| CuTe | Python CTM | 774.592 | 751.136–777.664 | 1402.838 |
| CTM | Rust CTM | 780.320 | 778.336–780.576 | 1392.540 |
| CTM | Python CTM | 770.848 | 765.824–772.032 | 1409.651 |

The observed run-to-run variation limits claims about small differences.
The JSON clock snapshot is taken after timing, so it does not establish clock
conditions throughout either implementation's measurement. Repeat comparisons
on an otherwise idle GPU before interpreting a ranking. A short warmup can
sample a different power/clock regime; 2,000 warmups is a recorded protocol,
not a guarantee that every device or shape has reached steady state.

## Nsight Systems kernel durations

Separate captures use Nsight Systems **2026.3.2**, software CUDA tracing and
device-side event timestamps. Each result below is the median of the **301
individual kernel execution intervals** enclosed by the timed event pairs.
The range is over individual kernels, unlike the range of run medians above.

| Capture | Implementation | Median kernel (µs) | Individual-kernel range (µs) |
| --- | --- | ---: | ---: |
| CuTe | Rust CuTe | 766.752 | 735.263–794.943 |
| CuTe | Python CTM | 732.416 | 587.711–847.424 |
| CTM | Rust CTM | 774.207 | 731.360–812.256 |
| CTM | Python CTM | 751.392 | 600.672–819.455 |

Kernel durations exclude idle intervals before or after a kernel, but include
memory waits and synchronization during execution. CUDA-event intervals also
include time inside the event pair when no kernel is executing. Profiling can
change host submission, clocks and execution conditions. Do not subtract the
ordinary event and separate profiler columns to estimate launch overhead, or
interpret these two captures as a stable kernel-speed ranking.

[extract_nsight.py](extract_nsight.py) matches process, device, context, green
context and stream. It requires exactly 602 event records and 301 complete
GEMMs per implementation, one kernel inside every pair, with the expected
192-thread blocks and 148-block grid. It also checks total GEMM counts (2,301
Python and 2,302 Rust), excludes all warmups, and rejects dropped-record
diagnostics. Captured event medians agree with the profiled comparison reports
within 0.033 µs. The two CSV files retain every selected event/kernel timestamp;
the extracted JSON records counts, timing statistics, names and trace hashes.
Raw `.nsys-rep` and SQLite captures remain under `/tmp/fp16-readme-20261018` on
the measurement host rather than in Git.

## Reproduce

Build either Rust example using its entry README and prepare the shared Python
environment. From `cuda-oxide/`, select the executable and a fresh output folder:

```bash
GEMM_PYTHON=cute-rs-preview/examples/fp16_gemm_256x352_cute/.venv/bin/python
GEMM_COMPARE=cute-rs-preview/examples/fp16_gemm_256x352_cute/compare.py
GEMM_BINARY=cute-rs-preview/examples/fp16_gemm_256x352_ctm/target/release/fp16_gemm_256x352_ctm
GEMM_RESULTS=$(mktemp -d /tmp/fp16-gemm-measurement.XXXXXX)

for run in 1 2 3; do
  "$GEMM_PYTHON" "$GEMM_COMPARE" --rust-bin "$GEMM_BINARY" \
    --mnk 8192,8096,8192 --seed 0 --warmup 2000 --iters 301 \
    --json "$GEMM_RESULTS/comparison-$run.json"
done

nsys profile --trace=cuda-sw --sample=none --cpuctxsw=none \
  --cuda-event-trace=true --discard-environment=true --export=sqlite \
  --output="$GEMM_RESULTS/gemm-nsight" \
  "$GEMM_PYTHON" "$GEMM_COMPARE" --rust-bin "$GEMM_BINARY" \
  --mnk 8192,8096,8192 --seed 0 --warmup 2000 --iters 301 \
  --json "$GEMM_RESULTS/gemm-nsight.json"

python3 cute-rs-preview/examples/fp16_gemm_256x352_cute/public_reference/extract_nsight.py \
  "$GEMM_RESULTS/gemm-nsight.sqlite"
```

For Rust CuTe, change `GEMM_BINARY` to
`cute-rs-preview/examples/fp16_gemm_256x352_cute/target/release/fp16_gemm_256x352_cute`.
The [recorded commands](results/b200-20261018/commands.json) preserve the exact
alternating run order used here. Run GPU workloads sequentially.

## Earlier results and validation

The [2026-10-11 CTM investigation](../../fp16_gemm_256x352_ctm/public_reference/README.md#historical-b200-measurements-2026-10-11)
preserves earlier event timings, controlled-clock Nsight Compute results and
CuTe migration checks. Its original Python reference is absent from the
checkout. Those values are historical evidence, not measurements of the new
public Python kernel or a same-session comparison against it.

Numerical correctness and sanitizer status are separate from performance.
[Validation notes](README.md#validation-and-known-limitation) retain passing
numerical results and the unresolved Python TMA memcheck diagnostic.
