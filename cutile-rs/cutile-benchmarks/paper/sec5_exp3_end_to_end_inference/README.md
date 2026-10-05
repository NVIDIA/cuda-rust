# Experiment 3: Grout Case Study

**Paper section**: §5.3
**Claim**: The system is general enough for end-to-end LLM inference.

## Paper Data

End-to-end inference runs use default GPU clocks. Do not apply a
microbenchmark clock cap to Grout/vLLM/SGLang measurements.

Current committed result bundles:

- `data/sweep_pp_20260508_114340/`: RTX 5090, Qwen3-4B, `tg=36`,
  `pp={18,128,512,2048,8192}`. The paper plot uses `aggregate.csv` and
  `run.jsonl`.
- `data/sweep_tg_20260508_111703_plus_115728_tg8192/`: RTX 5090,
  Qwen3-4B, `pp=18`, `tg={36,128,512,2048,8192}`, includes
  request-level HBM roofline columns. The paper plot uses `aggregate.csv`.
- `data/b200_qwen3_32b/pp_sweep_tg36_8k/`: B200, Qwen3-32B, `tg=36`,
  `pp={18,128,512,2048,8192}`. The paper plot uses `aggregate.csv` and
  `run.jsonl`.
- `data/b200_qwen3_32b/tg_sweep_pp18_8k/`: B200, Qwen3-32B, `pp=18`,
  `tg={36,128,512,2048,8192}`, includes request-level HBM roofline
  columns. The paper plot uses `aggregate.csv`.

Human-readable `aggregate.md` and `summary.txt` files from the original sweep
runs are not part of the public artifact; the paper figures read the CSV and
JSONL files listed above.

Raw timestamped source directories live in the sibling `grout-v2`
checkout. The B200 bundle is copied from
`grout-v2:benchmark/results/final/b200_qwen3_32b`.

## Plots

Generate the RTX 5090 plot:

```bash
python3 plot_grout_sweep.py --target rtx5090
```

Generate the B200 plot:

```bash
python3 plot_grout_sweep.py --target b200
```

Outputs:

- `figures/generated/exp3_grout_sweep.pdf`
- `figures/generated/exp3_grout_sweep_b200.pdf`

By default, paper plots omit roofline curves. Use `--roofline nominal` or
`--roofline both` for diagnostics if you want to visualize the nominal
request-level HBM bound or the historical 85% bandwidth reference.

## What Is Measured

- `request_gen_tps`: generated tokens divided by full end-to-end request
  time. This is the main cross-engine metric.
- Prefill/TTFT latency: direct per-request measurement when available;
  vLLM uses the benchmark harness's TTFT-style probe.
- TG roofline: request-level nominal HBM upper bound with the same
  generated-token denominator, using the ideal decode payload model emitted by
  the Grout aggregation script.

## Scope Notes

- Grout is an inference test bed, not a general-purpose serving stack.
- Grout uses a mix of safe kernels, `unchecked_accesses`, raw-pointer kernels,
  and cuBLAS GEMM fallbacks.
- Large model GEMMs fall back to cuBLAS; the cuTile Rust kernels cover the
  custom fused/model-specific operations around the GEMM core.
