# CUDA-event and Nsight Systems blockscale measurements

The [2026-10-18 evidence](results/nsight-rtx5090-20261018.json) compares
unprofiled CUDA events with kernel durations from a separate Nsight Systems
capture at source revision `a0d5776b2b7a4ecd387c14d1408d879a2ff42201`.
The Rust implementation uses `cute-rs` through the CUTLASS translation backend.
The CuTeDSL source remains the unmodified CUTLASS v4.6.2 reference, compiled
with the installed 4.7.0 runtime. Both compilers use optimization level 2.

Median and range of the three round medians, in microseconds:

| Implementation | Unprofiled CUDA events | Nsight kernel duration |
| --- | ---: | ---: |
| Rust (`cute-rs`) | 13.536 (13.504–13.600) | 11.360 (11.360–11.360) |
| CuTeDSL | 16.832 (16.704–16.832) | 10.176 (10.176–10.176) |

All 93 selected launches per implementation are retained in the compact
evidence. The Rust cubin is 29,656 bytes with SHA-256
`4d292bcf454db514905e85533ed8a35149b8d8081e2fd811ec39e9c5521c22e7`.

## Measurement contract

Both implementations compute `M=N=2048`, `K=1024`, `L=1`, with a
`128 × 128 × 128` CTA tile, packed E2M1 inputs, UE8M0 scales per 32 K values,
FP32 accumulation and FP16 output. Buffers are reused and CUDA graphs are
disabled. Each round has 100 warmup launches and 31 single-launch event pairs.
The reported value is the median of three round medians, in microseconds.

The comparison uses the existing programs without changing their fixtures:
Rust checks all 4,194,304 FP16 output bit patterns against its deterministic
CPU oracle; CuTeDSL checks its seed-2026 inputs against the upstream numerical
reference (`atol=0.1`, `rtol=0.01`). The inputs are different. Rust runs in a
fresh process for each round; CuTeDSL runs all three rounds in one process.
Rust runs first in both captures. GPU clocks are unlocked, so these measurements
do not establish a stable speed ranking or isolate a compiler-only difference.

CUDA-event intervals include gaps caused by submission between the recorded
events. Nsight measures only the GEMM kernel's GPU duration, which still includes
stalls within the kernel. Compilation, input generation, validation, transfers
and warmups are excluded. The unprofiled and profiled results come from separate
runs; subtracting their columns does not isolate host overhead.

## Reproduce

Build from `cuda-oxide/` after installing the CUTLASS compiler with
`cargo oxide toolchain install cutlass`:

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 \
  cargo oxide build blockscale_gemm_cute --arch sm_120a
```

Use Python 3.12 with a CUDA-enabled PyTorch installation. The measured package
versions are `nvidia-cutlass-dsl==4.7.0`, `cuda-python==13.4.1`,
`cuda-bindings==13.4.3`, `torch==2.14.1` and `numpy==2.5.4`; the older
[requirements.txt](requirements.txt) describes the historical 4.6.2 environment.

From this example's directory, select an idle SM120 GPU and prepare a comparison
command. `BLOCKSCALE_PYTHON` should name the configured Python interpreter:

```bash
export BLOCKSCALE_PYTHON="$PWD/../../.venv/bin/python"
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUTE_DSL_COMPILER_OPT=--opt-level=2
BLOCKSCALE_RESULTS=$(mktemp -d /tmp/blockscale-nsight.XXXXXX)
mkdir "$BLOCKSCALE_RESULTS/events" "$BLOCKSCALE_RESULTS/profiled"

cat > "$BLOCKSCALE_RESULTS/compare.sh" <<'SH'
set -eu
for round in 0 1 2; do
    target/release/blockscale_gemm_cute > "$1/rust-round-$round.txt" 2>&1
done
"$BLOCKSCALE_PYTHON" reference/benchmark.py > "$1/cutedsl.txt" 2>&1
if [ "${BENCH_PROFILED:-0}" = 1 ]; then
    cp reference/cutedsl-profiled-events.json "$1/cutedsl-events.json"
else
    cp reference/cutedsl-events.json "$1/cutedsl-events.json"
fi
SH

bash "$BLOCKSCALE_RESULTS/compare.sh" "$BLOCKSCALE_RESULTS/events"

BENCH_PROFILED=1 nsys profile --trace=cuda-sw,nvtx --sample=none \
  --cpuctxsw=none --cuda-event-trace=true --discard-environment=true \
  --export=sqlite --output="$BLOCKSCALE_RESULTS/blockscale" \
  bash "$BLOCKSCALE_RESULTS/compare.sh" "$BLOCKSCALE_RESULTS/profiled"
```

The recorded session used driver 615.71.09, GPU UUID
`GPU-04a6d226-f1d6-0fc8-a95a-3ede21981e72` and Nsight Systems 2026.3.2.
Run the two commands sequentially without other work on that GPU. Raw logs,
SQLite exports and binary traces stay outside the repository.

## Select measured launches

Use the SQLite `CUPTI_ACTIVITY_KIND_CUDA_EVENT` completion timestamps to pair
each start/end event on its full stream identity: process, device, context,
green context and stream. Select kernels wholly enclosed by each pair and
require exactly one GEMM launch. Match the demangled Rust name
`blockscale_gemm` and the CuTeDSL `Sm120BlockScaledGemmKernel` main kernel.
Reject kernels crossing a boundary, extra timed streams, missing events,
incomplete launch counts or Nsight diagnostics reporting dropped events.

Each of the three Rust processes must have 132 GEMM launches: one correctness
launch, 100 warmups and 31 measured launches, with exactly 62 GPU event records.
The CuTeDSL process has three sets of 31 measured pairs, with exactly 186
GPU event records on its GEMM stream. It has 394 GEMM launches: one validation
and three rounds of 100 warmups plus 31 samples. Its 13 conversion kernels
are excluded. Do not use a whole-capture
`cuda_gpu_kern_sum` average, which includes untimed launches.

For each selected pair, calculate `(kernel.end - kernel.start) / 1000` to get
microseconds. Group consecutive samples into rounds of 31, take each round's
median, then take the median of those three values. The selected zero-based
GEMM indices are 101–131 in each Rust process and 101–131, 232–262 and 363–393
in the CuTeDSL process. Those index checks agree with event enclosure.

The median intervals between Nsight's GPU-event completion timestamps exceed
the same profiled round's `cuEventElapsedTime` report by 0.384–0.480 µs.
The evidence retains these observed differences; it does not equate the two
timers. The unprofiled event table uses the separate `events/` output.
Nsight warns that
[device-side event tracing can add overhead](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cuda-trace-methods).
The capture has no dropped-event diagnostic. Generic NVTX warnings and a
no-CUDA warning for the final file-copy subprocess remain in the evidence;
all four workload processes have complete kernel and event counts.

Run this standard-library-only extraction from the same shell as the capture.
It checks the event-enclosed launch selection against those fixed ordinals:

```bash
python3 - "$BLOCKSCALE_RESULTS/blockscale.sqlite" <<'PY'
from collections import defaultdict
import json
import sqlite3
import statistics
import sys

db = sqlite3.connect(sys.argv[1])
db.row_factory = sqlite3.Row
identity = lambda row: tuple(row[k] for k in
    ('globalPid', 'deviceId', 'contextId', 'greenContextId', 'streamId'))
def implementation(name):
    if name == 'blockscale_gemm':
        return 'Rust'
    if name.startswith('kernel_cutlass_kernel_dense_blockscaled_gemm_persistent_cooperativeSm120BlockScaledGemmKernel_'):
        return 'CuTeDSL'
    return None

kernels, events = defaultdict(list), defaultdict(list)
for row in db.execute('SELECT k.*, s.value AS name FROM CUPTI_ACTIVITY_KIND_KERNEL k '
                      'JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start'):
    kernels[identity(row)].append(row)
for row in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_CUDA_EVENT ORDER BY timestamp'):
    events[identity(row)].append(row)
assert len(events) == 4, 'expected three Rust streams and one CuTeDSL stream'
assert not any('dropped' in row['text'].lower()
               for row in db.execute('SELECT text FROM DIAGNOSTIC_EVENT'))
rounds = defaultdict(list)
for stream, markers in events.items():
    workload = [k for k in kernels[stream] if implementation(k['name'])]
    labels = {implementation(k['name']) for k in workload}
    assert len(labels) == 1, 'ambiguous workload stream'
    label = labels.pop()
    rust = label == 'Rust'
    assert len(workload) == (132 if rust else 394)
    assert len(markers) == (62 if rust else 186)
    samples = []
    for i in range(0, len(markers), 2):
        begin, end = markers[i]['timestamp'], markers[i + 1]['timestamp']
        assert 0 < begin < end
        assert not i or markers[i - 1]['timestamp'] < begin
        overlap = [k for k in kernels[stream] if k['start'] < end and begin < k['end']]
        assert len(overlap) == 1, 'missing, extra or overlapping kernel'
        kernel = overlap[0]
        assert begin <= kernel['start'] < kernel['end'] <= end
        sample = i // 2
        ordinal = 101 + sample if rust else 101 + (sample // 31) * 131 + sample % 31
        assert kernel['start'] == workload[ordinal]['start'], 'selected an untimed launch'
        samples.append((kernel['end'] - kernel['start']) / 1000)
    rounds[label].extend(statistics.median(samples[i:i + 31])
                         for i in range(0, len(samples), 31))
assert {label: len(values) for label, values in rounds.items()} == {'Rust': 3, 'CuTeDSL': 3}
print(json.dumps({label: {'round_medians_us': values, 'median_us': statistics.median(values)}
                  for label, values in rounds.items()}, indent=2))
PY
```

The compact evidence retains round statistics, selected and excluded counts,
kernel samples, correctness results, source hashes and raw-trace hashes.
Historical results remain in the [CTM comparison record](../../blockscale_gemm_ctm/benchmark-results.json)
and its [measurement notes](../../blockscale_gemm_ctm/reference/measurement-notes.md);
their timings are not combined with this capture.

## Historical entry-README result

The earlier README recorded a 2026-08-18 run at `aafe6e47fb4e` on RTX 5090:
Nsight Systems 2026.1.3 reported 11.264 µs for the CUTLASS 4.7 translation
and 10.272 µs for CuTeDSL 4.6.2. The translated cubin was 29,608 bytes with
SHA-256 `69e1f1aad53a5889a40f6d7bfce0c6f66607eaeb1119640aa94f8ed3467dfbee`.
Those historical kernel-only values are superseded in the entry README by the
fresh, paired event/profiler measurements above.
