# CUDA-event and Nsight Systems decode measurements

The [2026-10-18 evidence](results/nsight-rtx5090-20261018/summary.json) records
fresh builds at `a0d5776b2`, with runtime and static shapes at
`B=1, Sq=1, Sk=1024, H=8, D=64`. Both implementations passed the CPU FP64 oracle,
replay checks, matching input hashes and cross-implementation output checks.
The CuTeDSL reference is independently authored here, using the same split-KV
algorithm; it is not an upstream CUTLASS decode kernel.

Each implementation performs two correctness invocations, 100 warmups and 31
timed invocations. Each invocation always launches **partial, then merge** on
one stream. The README's CUDA-event column is the median of 31 unprofiled
intervals around those two launches, synchronizing after each sample. It includes
submission gaps. Compilation, allocation, transfers and validation are outside
the intervals.

The same comparison command runs separately under Nsight Systems. **Active sum**
adds the two kernel durations within each invocation; **sequence span** runs from
the partial kernel's start to the merge kernel's end, including their gap.
Kernel durations include device stalls. Medians in microseconds:

| Shapes | Implementation | Partial | Merge | Active sum | Sequence span |
| --- | --- | ---: | ---: | ---: | ---: |
| Runtime | Rust (cute-rs) | 1.984 | 1.408 | 3.392 | 3.680 |
| Runtime | CuTeDSL | 1.856 | 1.664 | 3.520 | 3.808 |
| Static | Rust (cute-rs) | 1.888 | 0.992 | 2.880 | 3.168 |
| Static | CuTeDSL | 1.728 | 0.960 | 2.656 | 2.976 |

The active sum is computed **before** taking the median. It need not equal the
sum of the separate component medians. The shorter Rust event intervals in the
README do not imply faster kernels in every case: static decode has a smaller
CuTeDSL active sum in this capture. One run per method with fixed order and
unlocked clocks does not establish a stable speed ranking.

## Selection and conditions

The shared [extractor](../../fmha_prefill_sm120a_cute/reference/extract_nsight.py)
requires exactly **133 partial and 133 merge launches** in each child stream,
alternating in that order, plus **62 GPU event markers**. Every event pair must
enclose one complete partial/merge invocation. The selected 31 pairs must also
equal the final 31 invocations by ordinal count, excluding two correctness pairs
and 100 warmups. It checks process/device/context/green-context/stream identity,
rejects overlapping or missing kernels and dropped-event diagnostics, and
retains active sums and spans separately.

GPU completion-marker intervals differ from profiled CUDA elapsed-time reports
in this capture (maximum absolute difference **4.288 µs**). Every difference is
retained in [samples.json](results/nsight-rtx5090-20261018/samples.json); their cause
is not established. Event enclosure and ordinal selection agree for every pair.
The table uses **unprofiled** CUDA events and separately captured kernel
durations, so subtracting columns does not isolate host overhead. See the
[shared method and event-marker audit](../../fmha_prefill_sm120a_cute/reference/nsight-measurements.md).

The environment is RTX 5090, driver 615.71.09, CUDA toolkit 13.4.92, CUTLASS 4.7.0
at optimization level 2 on both paths, Python 3.12.3, PyTorch 2.14.1+cu130 and
Nsight Systems 2026.3.2. Buffers are reused; no CUDA graphs or clock locking were
applied. CuTeDSL runs first, then Rust for each case. GPU runs were serial.
The [historical notes](measurement-notes.md) use a different three-round protocol.
The summary retains validation, commands, hashes and actual trace diagnostics;
full binary traces and SQLite exports remain local, identified by their hashes.

## Reproduce

After [reference setup](README.md), run from `cuda-oxide/`:

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 \
  cargo oxide build fmha_decode_sm120a_cute --arch sm_120a

export CUTE_DSL_ARCH=sm_120a CUTE_DSL_COMPILER_OPT=--opt-level=2
export CUDA_VISIBLE_DEVICES=0
FMHA_BIN=cute-rs-preview/examples/fmha_decode_sm120a_cute/target/release/fmha_decode_sm120a_cute
FMHA_RESULTS=$(mktemp -d /tmp/sm120-decode.XXXXXX)

"$FMHA_BIN" --all-cases --warmup 100 --iters 31 \
  --json "$FMHA_RESULTS/decode-events.json"

nsys profile --trace=cuda-sw --sample=none --cpuctxsw=none \
  --cuda-event-trace=true --discard-environment=true --export=sqlite \
  --output="$FMHA_RESULTS/decode" \
  "$FMHA_BIN" --all-cases --warmup 100 --iters 31 \
  --json "$FMHA_RESULTS/decode-profiled.json"

python3 cute-rs-preview/examples/fmha_prefill_sm120a_cute/reference/extract_nsight.py \
  "$FMHA_RESULTS/decode.sqlite" \
  --comparison "$FMHA_RESULTS/decode-profiled.json" \
  --output "$FMHA_RESULTS/decode-kernels.json"
```

Select an idle RTX 5090 with `CUDA_VISIBLE_DEVICES`; the recorded run used its
UUID, retained in the evidence. The binary finds `cute-rs-preview/.venv/bin/python`
automatically. Keep each benchmark and profiler run serial on that GPU.
