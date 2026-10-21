# Rust CTM versus Python CUTLASS Primitives

This is the direct comparison of [`ctm-rs`](../src/kernel.rs) with the
repository's [Python CTM kernel](kernel.py), using
`cutlass.experimental.primitives`. Both use the same deterministic packed
inputs and exact FP16 oracle. The separate [CuTeDSL comparison](nsight-measurements.md)
uses the higher-level CuTe kernel and is supplementary.

Measured on **2026-10-18** at source revision
`a0d5776b2b7a4ecd387c14d1408d879a2ff42201`. Values are medians of three
round medians, followed by the range of those round medians, in microseconds.

| Implementation | Unprofiled CUDA events | Nsight kernel duration |
| --- | ---: | ---: |
| Rust CTM | 13.536 (13.472–13.728) | 11.616 (11.584–11.647) |
| Python CTM | 19.520 (19.328–19.648) | 10.880 (10.848–10.880) |

[Compact measurement record](results/ctm-python-rtx5090-20261018.json).
Both implementations passed the full-output and replay checks in all six
processes each. These timings use their existing launch paths; see below.

## Workload and timing

- RTX 5090 (SM120), M=N=2048, K=1024; packed E2M1 inputs, UE8M0 scales per
  32 K values, FP32 accumulation and FP16 output.
- 170 CTAs, 288 threads per CTA, 128×128×128 tiles, three input stages,
  86,016 dynamic shared bytes; hot buffers, no CUDA graphs, unlocked clocks.
- Rust uses CUTLASS compiler **4.7.0**, profile `cutlass-primitives-mlir22`;
  Python uses CUTLASS DSL/compiler **4.8.0**. Both request optimization level 2.
- Three rounds, each with 100 warmups and 31 timed launches. Each implementation
  runs in a fresh process. Order rotates: Rust/Python, Python/Rust, Rust/Python.
  Report the median of the three round medians and retain their range.

CUDA events come from runs **without a profiler**. Separate Nsight Systems runs
repeat that sequence and measure GPU kernel duration, including device stalls.
Every process first checks all 4,194,304 FP16 output bit patterns and checks
the output again after its repeated launches. Setup, fixture generation,
compilation, transfers and verification are outside the measured intervals.

The Python [host wrapper](launch.py) rebuilds five tensor-map descriptors on
each launch; Rust prepares them before timing. CUDA-event intervals can include
GPU idle gaps while the host prepares and submits work. These event results
therefore compare the existing launch paths as well as their kernels.
Nsight kernel durations exclude that host work. Subtracting the two columns
does not measure host overhead because the captures are separate.

The [full-buffer parity record](benchmark-results.json) documents identical
A, B, both scale buffers and expected output. The current Rust fixture helpers
and oracle retain those semantics; the Python fixture source and all five
generated buffer hashes match the record. The new evidence retains the hashes
and correctness reports from this comparison.

## Reproduce

From `cuda-oxide/`, build the Rust executable with the pinned compiler:

```bash
cargo oxide toolchain install cutlass
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
CUDA_OXIDE_MLIR_PROFILE=cutlass-primitives-mlir22 \
CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 \
  cargo oxide build blockscale_gemm_ctm --arch sm_120a
```

From this example's directory, create the Python CTM environment and choose an
idle SM120 GPU. Use the actual executable path if Cargo's target directory was
overridden. The measured environment uses Python 3.12, CUDA bindings 13.4.3,
NumPy 2.5.4, driver 615.71.09 and Nsight Systems 2026.3.2.

```bash
python3.12 -m venv /tmp/cutlass-ctm-venv
/tmp/cutlass-ctm-venv/bin/python -m pip install -r reference/requirements.txt
export CTM_PYTHON=/tmp/cutlass-ctm-venv/bin/python
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUTE_DSL_ARCH=sm_120a CUTE_DSL_COMPILER_OPT=--opt-level=2
```

Run the following with standard-library Python from the example directory.
It creates a fresh results directory, checks every executable report, and
uses the existing comparison helper to select measured kernels. Run it
without other work on the selected GPU.

```python
import json
import os
from pathlib import Path
import statistics
import tempfile
import time

from compare import execution_order, kernel_samples, run_command, validate_report

results = Path(tempfile.mkdtemp(prefix="blockscale-ctm-python-"))
commands = {
    "rust_ctm": [str(Path("target/release/blockscale_gemm_ctm").resolve())],
    "python_ctm": [os.environ["CTM_PYTHON"], str(Path("reference/benchmark.py").resolve())],
}
names = {
    "rust_ctm": "blockscale_gemm",
    "python_ctm": "kernel_cutlass_blockscale_gemm_1...",
}
summary = {phase: {label: [] for label in commands} for phase in ("events", "profiled")}
print(f"Results: {results}", flush=True)
for phase in summary:
    (results / phase).mkdir()
    for round_index, order in enumerate(execution_order(list(commands), 3)):
        for label in order:
            prefix = results / phase / f"round-{round_index}-{label}"
            report_path = prefix.with_suffix(".json")
            command = commands[label] + ["--ctas", "170", "--warmup", "100",
                                         "--samples", "31", "--json", str(report_path)]
            if phase == "profiled":
                command = ["nsys", "profile", "--trace=cuda-sw,nvtx", "--sample=none",
                           "--cpuctxsw=none", "--cuda-event-trace=true",
                           "--discard-environment=true", "--export=sqlite",
                           "--output=" + str(prefix)] + command
            started = time.time_ns()
            code = run_command(command, 300, os.environ.copy(), prefix.with_suffix(".log"))
            if code:
                raise RuntimeError(f"Exit {code}: see {prefix.with_suffix('.log')}")
            report = validate_report(report_path, 170, 100, 31, started)
            value = report["median_ms"] * 1000
            if phase == "profiled":
                samples = kernel_samples(prefix.with_suffix(".sqlite"), 100, 31, names[label])
                value = statistics.median(samples)
            summary[phase][label].append(value)
for phase, implementations in summary.items():
    for label, values in implementations.items():
        print(phase, label, f"{statistics.median(values):.3f} us", values)
(results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
```

The Python kernel's `...` suffix is literal in this Nsight version, not a
wildcard. If a different version changes its name, inspect the trace and
update the exact expected name; do not silently accept a different kernel.

## Trace selection and evidence

Every profiled process must contain exactly **132** launches of its expected
GEMM: one initial correctness launch, 100 warmups and 31 measured launches.
The strict helper rejects missing/extra kernels and invalid timestamps, then
selects zero-based launches **101–131**. This gives **93 kernel samples per
implementation** across three rounds. Durations are `(end - start) / 1000`
microseconds from `CUPTI_ACTIVITY_KIND_KERNEL`.

An independent trace check confirmed all six captures contain 62 GPU event
markers on one complete process/device/context/green-context/stream identity.
Each pair encloses exactly one measured kernel, selecting those same final
31 launches with no overlaps. No dropped/incomplete-capture diagnostic was
found. The GPU-marker interval medians differ from the profiled
`cuEventElapsedTime` medians by 0.064–0.640 µs; the evidence retains both,
without treating them as identical timers or using them for the unprofiled
CUDA-event column.

The recorded GPU was `GPU-04a6d226-f1d6-0fc8-a95a-3ede21981e72`.
Raw evidence is under `/tmp/cuda-oxide-ctm-python-comparison-20261018/`.

The compact result preserves per-round event reports, all selected kernel
durations, correctness, source hashes, the Rust cubin hash, compiler versions and raw-trace
hashes. The Python runtime did not expose a generated cubin in these runs;
its package, compiler-library and source hashes identify the toolchain instead.
Raw traces remain outside the repository. The version difference and
existing primitive-lowering differences remain part of this comparison;
these results do not isolate language overhead.
