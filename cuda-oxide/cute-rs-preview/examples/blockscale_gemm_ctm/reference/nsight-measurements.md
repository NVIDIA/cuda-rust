# Rust CTM versus CuTeDSL: CUDA events and Nsight Systems

This is a supplementary comparison with a higher-level CuTeDSL kernel. The
primary CTM comparison uses our [Python CUTLASS Primitives reference](ctm-python-measurements.md).

These [2026-10-18 results](results/nsight-rtx5090-20261018.json) measure the
current `blockscale_gemm_ctm` executable at
`a0d5776b2b7a4ecd387c14d1408d879a2ff42201` against the
[public CuTeDSL GEMM](../../blockscale_gemm_cute/reference/dense_blockscaled_gemm_persistent_cooperative.py).
Both compilers use CUTLASS 4.7.0 at optimization level 2. The Rust compiler
profile is `cutlass-primitives-mlir22`; the CuTeDSL source is the unchanged
upstream v4.6.2 example running on the 4.7.0 runtime.

Median and range of three round medians, in microseconds:

| Implementation | Unprofiled CUDA events | Nsight kernel duration |
| --- | ---: | ---: |
| Rust CTM | 13.536 (13.376–13.536) | 11.616 (11.584–11.616) |
| CuTeDSL | 16.768 (16.736–16.800) | 10.144 (10.144–10.175) |

This comparator is different from the repository-authored
[CUTLASS CTM port](README.md), which needs compiler 4.8.0. Its earlier
measurements remain in [measurement notes](measurement-notes.md) and
[compiler-fix results](compiler-fix-results.json). No old timings or Rust
`cute-rs` measurements are used in the new table.

## Workload and checks

Both kernels compute `M=N=2048`, `K=1024`, `L=1`, with a
`128 × 128 × 128` CTA tile, packed E2M1 inputs, UE8M0 scales per 32 K values,
FP32 accumulation and FP16 output. Rust uses 170 CTAs, matching the CuTeDSL
scheduler capacity on this RTX 5090. The trace records the actual grids and
block dimensions in the evidence. Buffers are reused, CUDA graphs are disabled
and clocks are unlocked.

The measured grids are `[170,1,1]` for Rust CTM and `[1,1,170]` for CuTeDSL,
both with 288 threads per CTA. The current Rust cubin is 58,880 bytes with
SHA-256 `8cde8a431a70494fa892ba74aba49a480bfa0645752700fdb76f8c74f1afdf2d`.

Rust checks all 4,194,304 FP16 output bit patterns against its deterministic
CPU oracle before timing, then checks the output again after the repeated
launches. CuTeDSL uses its seed-2026 fixture and upstream reference check
(`atol=0.1`, `rtol=0.01`). Both pass, but their inputs differ.

Each round has 100 warmups and 31 single-launch CUDA-event samples. Rust runs
in three fresh processes, followed by one CuTeDSL process with three rounds.
The reported value is the median of the three round medians. This order and
process grouping apply to both the unprofiled and profiled sequences; the
comparison does not establish a stable speed ranking or isolate compiler
or language overhead.

CUDA-event intervals include gaps caused by submission. Nsight measures only
the GEMM kernel's GPU duration, including stalls inside the kernel. Setup,
compilation, input generation, transfers, validation and warmups are excluded.
The event and kernel columns come from separate captures; subtracting them
does not isolate host overhead.

## Reproduce

From `cuda-oxide/`, with the pinned compiler installed:

```bash
cargo oxide toolchain install cutlass
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
CUDA_OXIDE_MLIR_PROFILE=cutlass-primitives-mlir22 \
CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 \
  cargo oxide build blockscale_gemm_ctm --arch sm_120a
```

Use Python 3.12 and a CUDA-enabled PyTorch installation. The measured packages
are `nvidia-cutlass-dsl==4.7.0`, `cuda-python==13.4.1`, `cuda-bindings==13.4.3`,
`torch==2.14.1` and `numpy==2.5.4`. These are the public CuTeDSL comparator's
dependencies, separate from the authored CTM port's 4.8.0 requirements.

From this example's directory, choose an idle SM120 GPU and the configured
Python interpreter:

```bash
export BLOCKSCALE_PYTHON="$PWD/../../.venv/bin/python"
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUTE_DSL_COMPILER_OPT=--opt-level=2
BLOCKSCALE_RESULTS=$(mktemp -d /tmp/blockscale-ctm-nsight.XXXXXX)
mkdir "$BLOCKSCALE_RESULTS/events" "$BLOCKSCALE_RESULTS/profiled"

cat > "$BLOCKSCALE_RESULTS/compare.sh" <<'SH'
set -eu
for round in 0 1 2; do
    target/release/blockscale_gemm_ctm --ctas 170 --warmup 100 --samples 31 \
      --json "$1/rust-round-$round.json" > "$1/rust-round-$round.txt" 2>&1
done
"$BLOCKSCALE_PYTHON" ../blockscale_gemm_cute/reference/benchmark.py > "$1/cutedsl.txt" 2>&1
if [ "${BENCH_PROFILED:-0}" = 1 ]; then
    cp ../blockscale_gemm_cute/reference/cutedsl-profiled-events.json "$1/cutedsl-events.json"
else
    cp ../blockscale_gemm_cute/reference/cutedsl-events.json "$1/cutedsl-events.json"
fi
SH

bash "$BLOCKSCALE_RESULTS/compare.sh" "$BLOCKSCALE_RESULTS/events"
BENCH_PROFILED=1 nsys profile --trace=cuda-sw,nvtx --sample=none \
  --cpuctxsw=none --cuda-event-trace=true --discard-environment=true \
  --export=sqlite --output="$BLOCKSCALE_RESULTS/blockscale" \
  bash "$BLOCKSCALE_RESULTS/compare.sh" "$BLOCKSCALE_RESULTS/profiled"
```

The recorded session used driver 615.71.09, Nsight Systems 2026.3.2 and GPU UUID
`GPU-04a6d226-f1d6-0fc8-a95a-3ede21981e72`. Run the commands sequentially
without concurrent work on that GPU. Raw traces stay outside the repository.

## Select and aggregate

Use the [standard-library extraction snippet](../../blockscale_gemm_cute/reference/nsight-measurements.md#select-measured-launches)
with this capture's `BLOCKSCALE_RESULTS`. Both blockscale executables expose
`blockscale_gemm` and have the same event-pair and launch-count contract, so
that extraction applies directly to the fresh CTM trace. The CuTeDSL kernel
name and sampling schedule are also the same; the measured data are separate.

The snippet matches the full process/device/context/green-context/stream
identity, requires exactly one whole GEMM per event pair, rejects missing or
extra records and checks the expected launch ordinal. Each Rust process has
132 GEMMs and 62 GPU event markers; it selects zero-based launches 101–131.
CuTeDSL has 394 GEMMs and 186 markers; it selects 101–131, 232–262 and 363–393.
This retains exactly 93 measured kernels per implementation and excludes
validation, warmups and conversion kernels. A whole-capture kernel summary
would include those untimed launches.

For each selected kernel, compute `(end - start) / 1000` in microseconds.
Take the median within each set of 31, then the median of those three values.
The compact evidence retains all selected durations, round statistics,
correctness reports, source and cubin hashes, grid dimensions and raw-trace
hashes. It also retains observed differences between profiled
`cuEventElapsedTime` reports and GPU-event completion-marker intervals;
these timers are not assumed identical.

The marker-interval medians exceed the corresponding profiled API-event
medians by 0.384–0.704 µs in this capture. The trace has no dropped-event
diagnostic and contains complete kernel/event counts for every workload
process. Generic NVTX and no-CUDA warnings are retained in the evidence;
selection uses CUDA records, not NVTX ranges.
