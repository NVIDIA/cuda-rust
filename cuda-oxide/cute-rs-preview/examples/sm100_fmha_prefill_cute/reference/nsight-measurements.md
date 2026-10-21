# CUDA events and Nsight kernel durations

The [2026-10-18 measurements](results/nsight-b200-20261018/summary.json) cover
all three README profiles at commit `029fda8e3`. Both implementations passed
the existing independent FP32 oracle and cross-comparison checks before timing,
and their outputs were checked again afterward. These measurements supplement
the [historical results](measurement-notes.md); they use a newly built executable.

Median and range of the five window means, in microseconds:

| Profile | Implementation | Unprofiled CUDA events | Nsight kernel duration |
| --- | --- | ---: | ---: |
| Dense | CuTe-RS | 18.552 (18.445–18.556) | 18.682 (18.612–18.692) |
| Dense | CuTeDSL | 73.692 (44.035–76.597) | 18.444 (18.428–18.465) |
| Dense persistent | CuTe-RS | 279.571 (224.909–281.912) | 276.195 (229.597–280.872) |
| Dense persistent | CuTeDSL | 277.812 (226.037–298.726) | 279.437 (225.240–294.804) |
| Causal GQA | CuTe-RS | 155.356 (144.736–158.855) | 155.967 (145.177–163.405) |
| Causal GQA | CuTeDSL | 170.114 (168.127–171.612) | 167.412 (167.305–169.431) |

The small dense case illustrates the distinction: similar GPU kernel durations
can produce very different event-window throughput. Persistent profiles drift
across windows; these runs do not establish a stable speed ranking.

## What each time measures

The CUDA-event result is the median of five windows, each containing 100
complete invocations after 100 warmups. The elapsed stream time is divided by
100. It includes idle gaps when the host cannot submit launches fast enough.
Compilation, allocation, input copies and validation are outside the windows.

The Nsight Systems result comes from a separate run with the same settings.
For each timed window, sum the GPU kernel end-minus-start durations and divide
by 100, then take the median of those five means. Prefill launches one kernel
per invocation. This excludes gaps between kernels, but includes memory waits,
synchronization and other stalls during kernel execution; it is not a measure
of arithmetic instructions alone. NVIDIA's [CUDA GPU kernel report](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html#cuda-gpu-kern-sum)
describes the execution-duration metric.

Only kernels enclosed by the GPU-side CUDA event pairs on the matching
process, context and stream are included. Extraction requires five windows and
exactly 100 kernels in each, excluding warmups, the initial validation launch
and Rust's separate validation subprocess. The evidence retains each window
and selected kernel timestamps. A whole-process `cuda_gpu_kern_sum` report
also includes warmups and must not be substituted for this selection.

Profiling can perturb execution and host submission. The unprofiled event
result and profiled kernel result therefore describe separate observations;
subtracting them does not isolate host overhead. The instrumented event
measurements are retained separately for checking the trace boundaries.

## Environment and reproduction

NVIDIA B200, driver 615.71.09, CUDA toolkit 13.4.92, CUTLASS 4.7.0 at optimization
level 2 on both paths, Python 3.12.3, PyTorch 2.13.0+cu130, and Nsight Systems
2026.3.2. Buffers are hot, no CUDA graphs are used, and clocks are not locked.
CuTeDSL is measured first, then Rust, as in the current harness. Clock/thermal
drift and this fixed order limit conclusions about small timing differences.
The evidence records executable hashes and environment details.

Build from `cuda-oxide/`:

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 \
  cargo oxide build sm100_fmha_prefill_cute --arch sm_100a
```

From the example directory, with `python` selecting the configured reference
environment, run the same case once normally and once under Nsight:

```bash
python compare.py --rust-bin target/release/sm100_fmha_prefill_cute \
  --shape 1,1024,1024,8,8 --warmup 100 --iters 100 --windows 5 \
  --json /tmp/prefill-dense-events.json

nsys profile --trace=cuda-sw --sample=none --cpuctxsw=none \
  --cuda-event-trace=true --discard-environment=true --export=sqlite \
  --output=/tmp/prefill-dense \
  python compare.py --rust-bin target/release/sm100_fmha_prefill_cute \
  --shape 1,1024,1024,8,8 --warmup 100 --iters 100 --windows 5 \
  --json /tmp/prefill-dense-profiled.json
```

Repeat with `--shape 4,4096,4096,8,8` for dense persistent, or
`--shape 4,4096,4096,8,2 --causal` for causal GQA, using distinct output paths.
The recorded run used `/tmp/sm100-ctm-validation/venv/bin/python`; use your own
configured interpreter. Commands and raw benchmark reports accompany the summary.

Extract the timed windows with the shared, standard-library-only
[analysis helper](extract_nsight.py). It matches GPU event pairs to their
process, context and stream, checks launch counts and component ordering,
and rejects incomplete captures. From the example directory:

```bash
python3 reference/extract_nsight.py /tmp/prefill-dense.sqlite \
  --comparison /tmp/prefill-dense-profiled.json \
  --output /tmp/prefill-dense-kernels.json
```

Use the comparison JSON from the same profiled run. The output retains selected
kernel timestamps, event boundaries, per-window statistics and their aggregate.

Software CUDA tracing was selected explicitly: initial default hardware traces
dropped repeated CuTeDSL kernel records and reported incomplete CUPTI events.
Those incomplete captures are excluded. GPU-side event tracing itself adds
overhead, which is why event benchmarks are also run without the profiler.
See NVIDIA's [CUDA trace methods](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cuda-trace-methods).

Nsight Compute is useful for instruction and hardware-counter analysis, but
its replay, cache and clock controls can change the conditions being measured.
The table uses Nsight Systems execution intervals, not a mixture of metrics
from the two tools. See the [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#reproducibility).
