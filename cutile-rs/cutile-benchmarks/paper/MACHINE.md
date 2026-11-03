# Paper Machine Specification

This file records the paper-facing machine setup for the cuTile Rust paper
artifacts. The evaluation used two systems, not a single local host:

- an NVIDIA DGX B200 datacenter system, using one B200 GPU
- a workstation containing an NVIDIA GeForce RTX 5090

## Workload Placement

| Paper result | Artifact directory | System | GPU | Clock and pinning policy |
| --- | --- | --- | --- | --- |
| Section 5.1 safety-overhead microbenchmarks | `sec5_exp1_safety_overhead/` | NVIDIA DGX B200, single GPU | NVIDIA B200 | The paper states that these microbenchmarks lock SM clocks for reproducibility. The B200 runner uses an 1830 MHz clock target and records run metadata in `paper/results/b200/metadata.csv`. |
| Section 5.2 execution-mode overhead | `sec5_exp2_execution_mode_overhead/` | RTX 5090 workstation | NVIDIA GeForce RTX 5090 | The paper identifies the platform as RTX 5090; it does not specify a clock lock for this result. |
| Section 5.3 end-to-end inference | `sec5_exp3_end_to_end_inference/` | RTX 5090 workstation and DGX B200, single GPU | RTX 5090 for Qwen3-4B; B200 for Qwen3-32B | Default GPU clocks. Do not apply microbenchmark clock caps to Grout, vLLM, or SGLang measurements. |
| Appendix work-distribution data | `app_exp1_work_distribution/` | RTX 5090 workstation, as recorded by the artifact README | NVIDIA GeForce RTX 5090 | The paper appendix specifies CPU pinning: serial mode pins to one CPU core; async and threaded modes pin to CPUs 1-16, the workstation's 16 physical cores, avoiding SMT siblings. |

## B200 DGX System

| Property | Value |
| --- | --- |
| GPU | NVIDIA B200 |
| Role | Section 5.1 safety-overhead microbenchmarks; Section 5.3 Qwen3-32B inference |
| CUDA version recorded in B200 safety metadata | 13.2 |
| Driver version recorded in B200 safety metadata | 595.58.03 |
| Safety-microbenchmark SM clock target | 1830 MHz |
| Clock metadata | Recorded in `sec5_exp1_safety_overhead/paper/results/b200/metadata.csv` |
| Recorded max SM clock | 1965 MHz |
| Recorded memory clock | 3996 MHz |
| Recorded power limit | 1000 W |

For B200 dense FP16/BF16 GEMM SoL, use 2250 TFLOP/s per GPU. The basis used
by the artifact is NVIDIA's HGX B200 table: 36 PFLOP/s FP16/BF16 Tensor Core
for an 8-GPU HGX B200 in sparse mode, dense is half sparse, so
`36 / 2 / 8 = 2.25 PFLOP/s` per GPU.

## RTX 5090 Workstation

| Property | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090 |
| Architecture | Blackwell (`sm_120`) |
| SMs | 170 |
| Max graphics clock | 3090 MHz |
| Memory | 32 GB GDDR7 (`nvidia-smi`: 32607 MiB available FB memory) |
| Compute capability | 12.0 |
| Max threads per SM | 1536 |

The RTX 5090 workstation is used for the Section 5.2 execution-mode
microbenchmark, the Qwen3-4B inference measurements, and the appendix
work-distribution experiment.

## Nominal Device-Memory Bandwidth Inputs

For bandwidth rooflines, use the nominal device-memory bandwidth computed from
local NVML fields when those fields are available:

```text
BW_nominal_GB_s =
  max_memory_clock_mhz * transfers_per_clock * memory_bus_width_bits / 8 / 1000
```

Collect the inputs for the active benchmark GPU with:

```bash
python3 tools/query_nominal_memory_bandwidth.py --device 0 --markdown
```

Recorded paper inputs:

| GPU | Source | Max memory clock | Bus width | Transfers/clock | Calculation | Nominal BW |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| NVIDIA GeForce RTX 5090 | local NVML query | 14001 MHz | 512 bit | 2 | `14001 * 2 * 512 / 8 / 1000` | 1792.128 GB/s |
| NVIDIA B200 | derived from `sec5_exp1_safety_overhead/paper/results/b200/metadata.csv` | 3996 MHz | 7680 bit | 2 | `3996 * 2 * 7680 / 8 / 1000` | 7672.320 GB/s |

The optional Section 5.3 Grout request-level roofline diagnostics use separate
model-level parameters recorded in the aggregate files: RTX 5090/Qwen3-4B uses
`BW_nominal=1792.0 GB/s`, and B200/Qwen3-32B uses `BW_nominal=8000.0 GB/s`.
Those roofline diagnostics are not the same thing as the Section 5.1
elementwise bandwidth denominator.

## Software

| Component | Version / source |
| --- | --- |
| cuTile Rust | 0.2.0 for the paper-facing measurements |
| CUDA | 13.2 in recorded benchmark metadata |
| Python | 3.12 virtual environments in the benchmark harnesses |
| cuda-tile Python | development checkout recorded in `README.md` provenance |
| tileiras | 13.2.78 in local RTX 5090 diagnostic notes |
| torch | 2.9.1 in local RTX 5090 diagnostic notes |

## Notes

- JIT caches are warmed before steady-state performance measurement.
- Criterion, pytest-benchmark, and benchmark-local loops handle warmup and
  sample iteration depending on the experiment.
- The paper text is the authority for paper-facing clock policy; checked-in
  metadata records per-run device fields for committed result bundles.
