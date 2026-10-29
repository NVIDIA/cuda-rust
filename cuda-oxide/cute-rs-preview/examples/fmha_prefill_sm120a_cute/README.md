# FP16 prefill with cute-rs — SM120a (RTX 5090)

This cuda-oxide example computes `softmax(Q @ K.T / sqrt(64)) @ V` with
FP16 inputs/output, FP32 accumulation, and 64-by-64 tiles.

```text
Q/K/V tensor views → typed shared-memory copies → warp MMA fragments
                  → online softmax → weighted V accumulation → FP16 output
```

- Batch **1**, head dimension **64**, contiguous BSHD tensors and equal Q/K/V heads.
- Sequence lengths are positive multiples of 64. **Causal** (default) requires
  equal query/KV lengths; `--noncausal` permits unequal lengths and `Sq=1`.
- Runtime shapes are the default. `--shape-mode static` requires **Sq=Sk=1024, H=8**.

From `cuda-oxide/`, once per checkout:

```bash
cargo oxide toolchain install cutlass
python3 -m venv cute-rs-preview/.venv
cute-rs-preview/.venv/bin/pip install -r cute-rs-preview/requirements.txt
```

Compare all four cases on RTX 5090:

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 \
CUTE_DSL_ARCH=sm_120a CUTE_DSL_COMPILER_OPT=--opt-level=2 \
  cargo oxide run fmha_prefill_sm120a_cute --arch sm_120a -- \
  --all-cases --warmup 100 --iters 31
```

Both implementations check every output against a CPU FP64 oracle. The command
also verifies matching input hashes and output parity, then prints a table of
CUDA-event medians. `--all-cases` requires **Sq=Sk=1024, H=8** and overrides
the shape mode and mask.

For one case, replace `--all-cases` with `--shape-mode static --causal`; choose
`runtime` or `--noncausal` for the other cases. Add `--json PATH` to save the
comparison, `--iters 0` to verify without timing, or `--rust-only` for a standalone
Rust report. `--all-cases` cannot be combined with `--rust-only`.
Pass `--help` after `--` for all options.

Use `--python PATH` to select another interpreter and `--device N` to select
another visible GPU; device 0 is the default.

RTX 5090 results at `a0d5776b2` (2026-10-18), **B=1, Sq=Sk=1024, H=8, D=64**,
CUTLASS 4.7 at optimization level 2 on both sides. Each value is the median of
**31 single-invocation samples after 100 warmups**, with reused buffers.
Both implementations passed correctness checks. Times are microseconds:

| Shapes | Attention | CUDA events: Rust (cute-rs) | CUDA events: CuTeDSL | Nsight kernel: Rust (cute-rs) | Nsight kernel: CuTeDSL |
| --- | --- | ---: | ---: | ---: | ---: |
| Runtime | Causal | 18.048 | 23.520 | 16.320 | 16.287 |
| Runtime | Noncausal | 17.952 | 23.872 | 16.416 | 16.576 |
| Static | Causal | 17.984 | 22.752 | 15.904 | 15.936 |
| Static | Noncausal | 17.984 | 23.936 | 15.936 | 16.991 |

CUDA events include GPU idle time during host submission. Nsight Systems
measures the kernel's execution duration, including device stalls and excluding
submission gaps. These are separate runs; subtracting columns does not isolate
host overhead. Clocks were unlocked and CuTeDSL ran first in each case.
This single run per method does not establish a stable speed ranking.

- [Reference setup and validation commands](reference/README.md)
- [Dedicated decode example](../fmha_decode_sm120a_cute/README.md)
