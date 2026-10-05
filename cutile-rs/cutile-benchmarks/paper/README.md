# Paper Benchmarks

This directory contains reproducibility artifacts for the cuTile Rust paper
*Fearless Concurrency on the GPU*.

The harnesses and checked-in result files here correspond to the paper's
evaluation section and supplementary work-distribution appendix. The companion
Grout source artifact for the end-to-end inference experiments is published in
the Hugging Face Grout repository: https://github.com/huggingface/grout

## Provenance

The paper's measurements were run against **cuTile Rust 0.2.0**.
External engines and sibling projects used by the comparisons are recorded
below. Commit hashes are full Git object IDs where the source checkout is
available locally.

| Component | Version / commit | Used for |
| --- | --- | --- |
| cuTile Rust | `0.2.0` | Rust kernel implementation, host API, and paper-result provenance. |
| Grout | [v0.1.0](https://github.com/huggingface/grout/releases/tag/v0.1.0) (`23ae5a3e78dc39e242918824839ba8a35e4adce9`) | Qwen3 inference engine and benchmark harness for Section 5.3. |
| cuTile Python | `4f9e5c99ac1c0f2a794d36a99fc1921da84f7de7` | Python frontend baseline for Section 5.1 safety-overhead comparisons. |
| vLLM | package `0.18.0`; local source checkout `69c9f19951026a7d6f2ddb425e7b1b5e6926f453` | Section 5.3 inference baseline, run with CUDA graphs and prefix cache disabled. |
| SGLang | package `0.5.9`; local source checkout `8686f42ac2c7abccfa4239ab6a5d6eaa715a4ef1` | Section 5.3 inference baseline, run with RadixAttention/prefix cache disabled. |

The vLLM and SGLang benchmark environments were installed from package
versions listed above; the local source checkout hashes are included as
source references, but the result JSONL does not encode package build hashes.

## Requirements

Use a local `python3` with the plotting dependencies installed. Set `PY` when
the benchmark should use a specific Python environment:

```bash
PY=/path/to/python3 ./sec5_exp1_safety_overhead/paper/run_b200_persistent_gemm_elemwise.sh
```

Section 5.1 requires an NVIDIA B200 to reproduce the committed paper numbers.
The runner locks SM clocks by default and uses `nvidia-smi`; clock locking may
require elevated permissions. Section 5.2 and the work-distribution appendix
were run on an RTX 5090. Section 5.3 uses the Grout artifact and model weights
from the companion Grout repository.

The embedded Rust benchmark crates depend on the published cuTile Rust crates
at version `0.2.0`; they do not require a sibling `cutile-rs` checkout.

Checked-in CSV and JSONL files are the paper data. For smoke tests or local
reruns, write results and generated figures outside this tree. Shell runners
use `RESULTS_DIR` for CSV output and `FIGURES_DIR` for generated plots when
they render plots as part of the run. Plot scripts with an `--out` argument
can also be pointed directly at a scratch file:

```bash
RESULTS_DIR=/tmp/cutile-paper-exp1 FIGURES_DIR=/tmp/cutile-paper-figures ./sec5_exp1_safety_overhead/paper/run_b200_persistent_gemm_elemwise.sh
python3 sec5_exp1_safety_overhead/paper/plot_exp1.py --target b200 --out /tmp/exp1_safety_overhead.pdf
python3 sec5_exp3_end_to_end_inference/plot_grout_sweep.py --target b200 --out /tmp/exp3_grout_sweep_b200.pdf
```

## Layout

| Path | Paper location | Paper figures | Contents |
| --- | --- | --- | --- |
| `sec3_example1_race_detection/` | Section 3, "Example: Preventing Data Races" | Listing/discussion around the cuTile Python race example | Head-permutation race example: Python permits the race; the equivalent cuTile Rust partitioned destination makes the data race inexpressible. |
| `sec5_exp1_safety_overhead/` | Section 5.1, "Safety Overhead" | `fig:safety_overhead`, `fig:elemwise_safety`, `fig:gemm_safety`, `fig:jit_breakdown`; generated files `exp1_elemwise.pdf`, `exp1_safety_overhead.pdf`, `exp1_jit_breakdown.pdf` | B200 safety-overhead microbenchmarks for memory-bound element-wise add and compute-bound persistent GEMM, plus the JIT timing input for the paper panel. |
| `sec5_exp2_execution_mode_overhead/` | Section 5.2, "Execution Mode Overhead"; appendix async-overlap panel | `fig:exec_mode`, `fig:async_throughput`; generated files `exp2_execmode_latency.pdf`, `exp2_async_throughput.pdf` | RTX 5090 pipeline latency for sync, async, and CUDA graph replay, plus async overlap with host work. |
| `sec5_exp3_end_to_end_inference/` | Section 5.3, "End-to-End Inference" | `fig:grout_sweeps`, `fig:grout_sweep_5090`, `fig:grout_sweep_b200`; generated files `exp3_grout_sweep.pdf`, `exp3_grout_sweep_b200.pdf` | Aggregated Grout, vLLM, and SGLang single-request Qwen3 inference results for RTX 5090/Qwen3-4B and B200/Qwen3-32B. |
| `app_exp1_work_distribution/` | Supplementary Work-Distribution Data appendix | `fig:work_distribution_appendix`, `fig:bimodal_throughput`, `fig:bimodal_w`; generated files `exp3_bimodal_throughput.pdf`, `exp3_bimodal_w.pdf` | RTX 5090 bimodal-GEMM queue experiments comparing serial, thread-per-stream, and single-host-thread async scheduling. |

Additional support files:

- `MACHINE.md`: hardware, clocking, and software environment notes.
- `tools/query_nominal_memory_bandwidth.py`: helper for nominal memory
  bandwidth roofline inputs.
- `lock_clocks.sh`: clock-locking helper for microbenchmark reproduction.

## Notes

The safety-overhead microbenchmarks use locked B200 SM clocks for
reproducibility. The execution-mode experiments use RTX 5090. End-to-end
inference uses RTX 5090/Qwen3-4B and B200/Qwen3-32B at default GPU clocks.

Raw profiling traces, model weights, build artifacts, and large generated
outputs are omitted. Checked-in result files are the
inputs needed to reproduce the plots and tables.
