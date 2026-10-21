# CUDA-event and Nsight Systems measurements

The [2026-10-18 evidence](results/nsight-rtx5090-20261018/summary.json) records
fresh opt-level-2 builds at `a0d5776b2` and the four matched README cases:
runtime/static shapes crossed with causal/noncausal attention, all with
`B=1, Sq=Sk=1024, H=8, D=64`. Both implementations passed the CPU FP64 oracle,
replay checks, matching input hashes and cross-implementation output checks.
The pinned public CuTeDSL kernel is unchanged.

## What the table measures

Each implementation performs two correctness launches, 100 warmups and 31 timed
invocations. CUDA events surround one launch, with synchronization after each
sample. The README reports the median of those 31 intervals from an unprofiled
run. Compilation, allocation, transfers and validation are outside the intervals.
GPU idle time while the host submits work can fall inside them.

A separate run of the **same comparison command** supplies Nsight Systems kernel
durations. For each timed invocation, take `kernel.end - kernel.start` in
microseconds, then take the median of 31 values. Kernel duration includes memory
waits, synchronization and other device stalls; it excludes gaps outside kernel
execution. It does not mean time spent only on arithmetic instructions.

The [extractor](extract_nsight.py) matches process, device, context, green context
and stream. Each child must contain exactly **133 attention launches** and
**62 GPU event markers**. Each pair must enclose exactly one complete kernel;
those kernels must also equal the final 31 attention launches by ordinal count.
The first two correctness launches and 100 warmups are excluded. Missing records,
unexpected kernels inside a pair, overlaps, and dropped-event diagnostics fail
extraction. A whole-capture `cuda_gpu_kern_sum` includes warmups and is unsuitable.

The capture uses software tracing (`--trace=cuda-sw`) and device-side event
markers. Nsight documents [event-trace overhead](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cuda-event-trace).
Profiler event markers and CUDA elapsed-time reports did **not** agree exactly
in this capture. All differences remain in [samples.json](results/nsight-rtx5090-20261018/samples.json).
For Rust static-causal sample 21, CUDA reported **33.120 µs**, the marker span
was **19.168 µs**, and the enclosed kernel was **15.904 µs**. The associated
start-event API call took 15.830 µs; the cause of the discrepancy is unproven.
The ordinal and event-enclosure selections agree for every sample. No sample
was removed and no equality between the two event measurements is asserted.

## Conditions and limitations

RTX 5090, driver 615.71.09, CUDA toolkit 13.4.92, CUTLASS 4.7.0, Python 3.12.3,
PyTorch 2.14.1+cu130 and Nsight Systems 2026.3.2. Both compilers use optimization
level 2. Buffers are reused, no CUDA graphs or clock locking were applied, and
the comparison runs CuTeDSL first, then Rust for each case. GPU runs were serial.
This is one run per method, not the historical three alternating rounds.
Clock, thermal and fixed-order effects limit interpretation of small differences.
Subtracting separately collected event and kernel timings does not isolate host
overhead. The [historical measurements](measurement-notes.md) retain their original
protocol and must not be mixed into the current table.

The summary retains binary/source hashes, commands, validation and trace
diagnostics. `samples.json` retains every selected timing, stream identity and
timestamp bounds. Full `.nsys-rep` and SQLite captures remain local, identified
by their hashes. The extractor can regenerate the selected results from them.

## Reproduce

After [reference setup](README.md), run from `cuda-oxide/`:

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 \
  cargo oxide build fmha_prefill_sm120a_cute --arch sm_120a

export CUTE_DSL_ARCH=sm_120a CUTE_DSL_COMPILER_OPT=--opt-level=2
export CUDA_VISIBLE_DEVICES=0
FMHA_BIN=cute-rs-preview/examples/fmha_prefill_sm120a_cute/target/release/fmha_prefill_sm120a_cute
FMHA_RESULTS=$(mktemp -d /tmp/sm120-prefill.XXXXXX)

"$FMHA_BIN" --all-cases --warmup 100 --iters 31 \
  --json "$FMHA_RESULTS/prefill-events.json"

nsys profile --trace=cuda-sw --sample=none --cpuctxsw=none \
  --cuda-event-trace=true --discard-environment=true --export=sqlite \
  --output="$FMHA_RESULTS/prefill" \
  "$FMHA_BIN" --all-cases --warmup 100 --iters 31 \
  --json "$FMHA_RESULTS/prefill-profiled.json"

python3 cute-rs-preview/examples/fmha_prefill_sm120a_cute/reference/extract_nsight.py \
  "$FMHA_RESULTS/prefill.sqlite" \
  --comparison "$FMHA_RESULTS/prefill-profiled.json" \
  --output "$FMHA_RESULTS/prefill-kernels.json"
```

Select an idle RTX 5090 with `CUDA_VISIBLE_DEVICES`; the recorded run used its
UUID, retained in the evidence. The binary finds `cute-rs-preview/.venv/bin/python`
automatically. Keep each benchmark and profiler run serial on that GPU.

Run the extractor's CPU-only checks from `cuda-oxide/`:

```bash
python3 -m unittest discover \
  -s cute-rs-preview/examples/fmha_prefill_sm120a_cute/reference -p test_extract_nsight.py
```
