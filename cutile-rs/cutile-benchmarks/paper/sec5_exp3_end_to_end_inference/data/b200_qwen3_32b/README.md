# B200 Qwen3-32B Final Inference Results

This directory contains the B200/Qwen3-32B inference results used in the paper.
The raw timestamped source directories were generated under
`benchmark/results/sweep/`, which is ignored by default.

## Bundles

- `tg_sweep_pp18_8k/`: source run `benchmark/results/sweep/20260507_213032`
  - `pp=18`
  - `tg={36,128,512,2048,8192}`
  - `reps=10`, `warmup=3`
  - includes roofline columns
- `pp_sweep_tg36_8k/`: source run `benchmark/results/sweep/20260507_210643`
  - `pp={18,128,512,2048,8192}`
  - `tg=36`
  - `reps=10`, `warmup=3`

The public artifact keeps only result files consumed by the paper plot:
`tg_sweep_pp18_8k/aggregate.csv`, plus `pp_sweep_tg36_8k/aggregate.csv` and
`pp_sweep_tg36_8k/run.jsonl`.

<a id="headline-results"></a>
## Throughput

TG sweep through 8192 generated tokens, request generation throughput:

| engine | variant | pp | tg | request_gen_tps | nominal roof | % nominal | % effective |
|---|---|---:|---:|---:|---:|---:|---:|
| grout-v2 | default | 18 | 36 | 80.9 | 120.9 | 66.9% | 78.6% |
| grout-v2 | default | 18 | 128 | 81.3 | 121.7 | 66.8% | 78.5% |
| grout-v2 | default | 18 | 512 | 81.6 | 121.9 | 66.9% | 78.7% |
| grout-v2 | default | 18 | 2048 | 81.2 | 121.6 | 66.8% | 78.6% |
| grout-v2 | default | 18 | 8192 | 80.1 | 120.1 | 66.7% | 78.5% |
| sglang | no-radix | 18 | 36 | 75.5 | 120.9 | 62.4% | 73.3% |
| sglang | no-radix | 18 | 128 | 77.2 | 121.7 | 63.5% | 74.6% |
| sglang | no-radix | 18 | 512 | 77.7 | 121.9 | 63.8% | 75.0% |
| sglang | no-radix | 18 | 2048 | 77.4 | 121.6 | 63.7% | 74.9% |
| sglang | no-radix | 18 | 8192 | 76.5 | 120.1 | 63.7% | 74.9% |
| vllm | cuda-graph | 18 | 36 | 78.3 | 120.9 | 64.8% | 76.1% |
| vllm | cuda-graph | 18 | 128 | 78.7 | 121.7 | 64.7% | 76.1% |
| vllm | cuda-graph | 18 | 512 | 78.8 | 121.9 | 64.6% | 76.0% |
| vllm | cuda-graph | 18 | 2048 | 78.5 | 121.6 | 64.6% | 76.0% |
| vllm | cuda-graph | 18 | 8192 | 77.5 | 120.1 | 64.5% | 75.9% |

PP sweep through 8192 prompt tokens, request generation throughput:

| engine | variant | pp | tg | request_gen_tps |
|---|---|---:|---:|---:|
| grout-v2 | default | 18 | 36 | 80.9 |
| grout-v2 | default | 128 | 36 | 80.0 |
| grout-v2 | default | 512 | 36 | 76.5 |
| grout-v2 | default | 2048 | 36 | 63.3 |
| grout-v2 | default | 8192 | 36 | 37.2 |
| sglang | no-radix | 18 | 36 | 75.4 |
| sglang | no-radix | 128 | 36 | 75.4 |
| sglang | no-radix | 512 | 36 | 72.4 |
| sglang | no-radix | 2048 | 36 | 61.3 |
| sglang | no-radix | 8192 | 36 | 36.7 |
| vllm | cuda-graph | 18 | 36 | 77.8 |
| vllm | cuda-graph | 128 | 36 | 77.6 |
| vllm | cuda-graph | 512 | 36 | 73.9 |
| vllm | cuda-graph | 2048 | 36 | 61.8 |
| vllm | cuda-graph | 8192 | 36 | 37.3 |
