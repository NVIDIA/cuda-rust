# CUDA-event and Nsight Systems decode measurements

The [2026-10-18 results](results/nsight-b200-20261018/summary.json) compare
unprofiled CUDA-event latency with GPU kernel durations from a separate Nsight
Systems capture. The source revision is `029fda8e3c41a7cfd2654b11cb3ebdb6f5da2a32`.
The [older measurement notes](measurement-notes.md) retain the historical
optimization results and their original settings.

Median and range of the five window means, in microseconds:

| Profile | Implementation | Unprofiled CUDA events | Nsight kernel sum |
| --- | --- | ---: | ---: |
| Small | CuTe-RS | 16.769 (16.765–16.782) | 16.959 (16.955–16.965) |
| Small | CuTeDSL | 27.827 (18.132–40.883) | 18.093 (18.087–18.110) |
| Persistent | CuTe-RS | 64.211 (48.251–66.707) | 59.148 (48.297–62.647) |
| Persistent | CuTeDSL | 52.636 (49.265–69.966) | 61.988 (49.036–64.389) |

Small CuTeDSL event windows also vary substantially, despite stable kernel
durations in the separate profiled run. Persistent decode changes its apparent
ranking between the two runs, with overlapping ranges. This is not evidence of
a stable speedup or a precise estimate of launch overhead.

For persistent decode, the median window means for main/reduction are
**51.411/7.737 µs** (CuTe-RS) and **52.769/9.218 µs** (CuTeDSL).
The aggregate above is computed by summing paired durations within each
invocation before aggregating; it does not add independently computed medians.

## What each number measures

The CUDA-event measurement encloses 100 complete invocations on the launch
stream. It includes time between launches and, for split decode, the reduction
kernel. Compilation, allocations, transfers and correctness checks happen
outside the interval. Each reported number is the median of five window means,
with 300 warmup invocations before each window.

The Nsight measurement adds the GPU durations of the selected kernels in each
window and divides by 100. The small case has one main kernel per invocation.
The persistent case has a main kernel followed by a reduction kernel, and both
durations contribute. This sum excludes gaps between kernels. Kernel duration
still includes device stalls and synchronization inside a kernel; it does not
measure only time issuing arithmetic instructions. The reported value is again
the median of five window means, rather than the median individual launch.

The trace also retains CUDA-event intervals from the profiled run. Comparing
those intervals with the kernel sum in the same window identifies time outside
the measured kernel intervals. Subtracting independently collected unprofiled
and profiled results does not isolate launch overhead: profiling and changes
in GPU clock or cache state can affect the runs differently.

This capture uses Nsight Systems software CUDA tracing, `--trace=cuda-sw`, with
device-side event tracing enabled. Initial captures using the default hardware
trace had incomplete kernel records and were excluded. Accepted captures must
account for every measured invocation. NVIDIA documents both the
[CUDA trace methods and event-trace overhead](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cuda-trace-methods).

Nsight Compute is useful for performance counters, but its replay, cache and
clock controls can change measurement conditions. These results use Nsight
Systems; they do not substitute an `ncu` replay duration for normal application
latency. See NVIDIA's [profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#replay).

## Conditions and cases

Measurements use NVIDIA B200, driver 615.71.09, CUTLASS 4.7.0 at optimization
level 2 for both implementations, and CUDA-enabled PyTorch 2.13.0. The Python
environment matches [requirements.txt](requirements.txt). GPU clocks are
unlocked, buffers are reused, and neither implementation uses CUDA graphs.
Both implementations pass the independent oracle before timing and are checked
again afterward.

| Case | Batch | KV length | Splits | Clusters | Kernels per invocation |
| --- | ---: | ---: | ---: | ---: | --- |
| Small | 1 | 1024 | 1 | 1 | Main |
| Persistent | 16 | 8192 | 8 | 74 | Main and reduction |

Each case runs once without profiling and then in a separate profiled process.
Both runs use CuTeDSL first, followed by CuTe-RS, with 300 warmups, 100 timed
invocations and five windows per implementation. These runs do not balance
both timing orders. Keep this ordering and the unlocked clocks in mind when
interpreting small differences; the historical results used different warmups
and both orders.

## Reproduce the capture

Build from `cuda-oxide/`:

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 \
  cargo oxide build sm100_mla_decode_fp16_cute --arch sm_100a
```

From the example directory, after [setting up the Python environment](README.md),
capture the small case:

```bash
DECODE_PYTHON=.venv/bin/python
DECODE_RESULTS=$(mktemp -d /tmp/sm100-decode-nsight.XXXXXX)

"$DECODE_PYTHON" compare.py \
  --rust-bin target/release/sm100_mla_decode_fp16_cute \
  --batch 1 --kv-length 1024 --splits 1 --clusters 1 \
  --warmup 300 --iters 100 --windows 5 --timing-order cutedsl-first \
  --json "$DECODE_RESULTS/decode-small-events.json"

nsys profile --trace=cuda-sw --sample=none --cpuctxsw=none \
  --cuda-event-trace=true --discard-environment=true --export=sqlite \
  --output="$DECODE_RESULTS/decode-small" \
  "$DECODE_PYTHON" compare.py \
  --rust-bin target/release/sm100_mla_decode_fp16_cute \
  --batch 1 --kv-length 1024 --splits 1 --clusters 1 \
  --warmup 300 --iters 100 --windows 5 --timing-order cutedsl-first \
  --json "$DECODE_RESULTS/decode-small-profiled.json"
```

For the persistent case, repeat both commands with
`--batch 16 --kv-length 8192 --splits 8 --clusters 74`, using
`decode-persistent` in the output filenames. Run cases sequentially. The
recorded session used `/tmp/sm100-ctm-validation/venv/bin/python` as its
configured interpreter.

## Select and aggregate the measured windows

Use the shared, standard-library-only
[analysis helper](../../sm100_fmha_prefill_cute/reference/extract_nsight.py)
from the example directory, pairing the SQLite export with the comparison JSON
from the same profiled run:

```bash
python3 ../sm100_fmha_prefill_cute/reference/extract_nsight.py \
  "$DECODE_RESULTS/decode-small.sqlite" \
  --comparison "$DECODE_RESULTS/decode-small-profiled.json" \
  --output "$DECODE_RESULTS/decode-small-kernels.json"
```

The helper checks GPU event pairs, launch counts, main/reduction ordering and
capture completeness. Its JSON retains selected kernel timestamps, component
durations and per-window statistics for both implementations.

Inspect the exported SQLite kernel records with their process, context, stream
and GPU timestamps. CUDA-event boundaries distinguish the five timed windows
from validation and warmup launches. Keep the timed Rust child process separate
from the earlier validation child process. Match these kernel names:

| Implementation | Main kernel | Reduction kernel |
| --- | --- | --- |
| CuTe-RS | `sm100_mla_decode_fp16_cute_rs` | `sm100_mla_decode_fp16_reduce_cute_rs` |
| CuTeDSL | `split_kv_kernel` | `reduction_kernel` |

Each accepted window contains exactly 100 main launches, plus 100 reduction
launches when splits exceed one, on the expected stream. Exclude all other
kernels, validation launches and 300 warmups per window. For each window,
calculate `sum(end - start) / 100`, converting nanoseconds to microseconds,
then take the median of the five window means. Preserve main and reduction
components separately as well as their sum.

A whole-capture `nsys stats --report cuda_gpu_kern_sum` summary includes warmup
and validation work, so it does not reproduce the selected-window result.
NVIDIA's [analysis guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html#cuda-gpu-kern-sum-nvtx-name-base-mangled-cuda-gpu-kernel-summary)
describes the summary fields. Keep the per-window values and count checks with
the [recorded evidence](results/nsight-b200-20261018/summary.json); raw binary
traces remain local.
