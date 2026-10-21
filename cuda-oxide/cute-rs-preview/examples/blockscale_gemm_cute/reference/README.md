# Block-scaled GEMM references

These are the comparison sources for [CuTe-RS](../) and the separate
[Rust CTM example](../../blockscale_gemm_ctm). All use
M=N=2048, K=1024, L=1, E2M1 inputs, UE8M0 scales per 32 K values,
FP32 accumulation, and FP16 output on SM120a.

For the current Rust (`cute-rs`) versus CuTeDSL comparison, see
[CUDA-event and Nsight Systems measurements](nsight-measurements.md). That
record uses CUTLASS 4.7.0 for both compilers and keeps its environment and
measurements separate from the historical 4.6.2 results below.

| Path | Implementation |
| --- | --- |
| [`dense_blockscaled_gemm_persistent_cooperative.py`](dense_blockscaled_gemm_persistent_cooperative.py) | Upstream Python CuTeDSL kernel. |
| [`blockscaled_gemm_dispatch.py`](blockscaled_gemm_dispatch.py) | Its required upstream instruction-selection helper. |
| [`benchmark.py`](benchmark.py) | The Python harness used for the recorded comparison. |
| [CTM reference](../../blockscale_gemm_ctm/reference/) | Matching GEMM port using the actual CUTLASS experimental primitives API, with the Rust fixture and exact oracle. |
| [`cpp/`](cpp) | The CUDA C++ instruction-primitives counterpart, including build and layout checks. |

The C++ counterpart was written for cuda-oxide using custom inline PTX. Its
historical results compare Rust with custom CUDA C++. Use [CTM reference](../../blockscale_gemm_ctm/reference/) for the
comparison against CUTLASS's actual primitives API. That port uses the upstream
SM120 operations; the available complete upstream blockscale primitives example
targets SM100.

## Python setup and execution

From this `reference/` directory, use Python 3.12 and a CUDA-enabled PyTorch
installation supporting the GPU. The pinned packages reproduce the versions
used by the recorded CuTeDSL 4.6.2 run:

```sh
python3.12 -m venv /tmp/blockscale-reference-venv
/tmp/blockscale-reference-venv/bin/python -m pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  /tmp/blockscale-reference-venv/bin/python benchmark.py
```

The harness performs the upstream numerical reference check (`atol=0.1`,
`rtol=0.01`) before timing. It uses seed 2026, hot buffer reuse, no CUDA graphs,
three rounds, 100 warmups and 31 single-kernel samples per round. Its generated
`cutedsl-events.json` contains **CUDA-event intervals**, which include submission
gaps. Those values are not the Nsight kernel-active times in the comparison
tables. Generated reports are ignored by Git.

This command runs CuTeDSL only. The current measurement notes give the combined
sequence for the Rust executable and CuTeDSL harness.

To measure kernel-active time, profile the same harness:

```sh
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID BENCH_PROFILED=1 \
  nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output=/tmp/blockscale-cutedsl \
  /tmp/blockscale-reference-venv/bin/python benchmark.py
nsys export --type=sqlite --output=/tmp/blockscale-cutedsl.sqlite \
  /tmp/blockscale-cutedsl.nsys-rep
```

Use a fresh output basename for each run. The harness marks each measured
window with `cutedsl_samples_round_*`. Extract only the 31 main GEMM launches
within each window; setup, conversion, and correctness kernels are excluded:

```sh
python3 - <<'PY'
import sqlite3
import statistics
with sqlite3.connect('/tmp/blockscale-cutedsl.sqlite') as db:
    windows = db.execute("SELECT start,end,text FROM NVTX_EVENTS "
                         "WHERE text LIKE 'cutedsl_samples_round_%' ORDER BY start").fetchall()
    assert len(windows) == 3
    medians = []
    for start, end, name in windows:
        samples = [row[0] for row in db.execute(
            "SELECT (k.end-k.start)/1000.0 FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON s.id=k.shortName WHERE k.start>=? AND k.end<=? "
            "AND s.value LIKE 'kernel_cutlass_kernel_dense%' ORDER BY k.start", (start, end))]
        assert len(samples) == 31
        medians.append(statistics.median(samples))
        print(name, medians[-1], 'us kernel-active')
    print('Median of round medians:', statistics.median(medians), 'us')
PY
```

The exact-oracle [comparison runner](../../blockscale_gemm_ctm/compare.py) accepts
the Rust and C++ binaries and the new [CTM executable harness](../../blockscale_gemm_ctm/reference/benchmark.py).
The older CuTeDSL `benchmark.py` above has a different interface.

## Source provenance

The two upstream Python files are unmodified copies from NVIDIA CUTLASS
`v4.6.2`, commit `6c65a175668952f09bcbf66cb97a8de1b734b4a0`, under
`examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/`.
They retain their full BSD-3-Clause notices. The harness and C++ counterpart
use cuda-oxide's Apache-2.0 license.

| File | SHA-256 |
| --- | --- |
| `dense_blockscaled_gemm_persistent_cooperative.py` | `e3e8503239716b4d68db7dfec4971dcff7b386470c8515c3d2cc8698c761540e` |
| `blockscaled_gemm_dispatch.py` | `2d4ab5d6c150d938c0dbae3f63f5ddbbc9abf33975d66810cd6fd47a45ff3996` |

The [CuTe/Rust comparison record](../../blockscale_gemm_ctm/benchmark-results.json)
and [C++/Rust comparison record](cpp/benchmark-results.json) preserve the original
measurements and hashes. Paths inside those records describe the checkout at
measurement time, before this reorganization; they are historical provenance.
