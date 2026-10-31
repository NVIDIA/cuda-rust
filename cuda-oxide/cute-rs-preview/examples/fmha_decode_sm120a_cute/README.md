# Single-token decode with cute-rs — SM120a (RTX 5090)

This cuda-oxide example computes attention for one query per head against an
existing FP16 KV cache. A dedicated kernel processes 64 cache entries per block;
a second kernel combines the partial softmax results and writes FP16 output.

```text
Q/K/V tensor views → typed copies → FP32 dot products and local softmax
                  → split states → merge kernel → FP16 output
```

- Batch **1**, head dimension **64**, contiguous BSHD tensors, FP32 accumulation.
- Any positive cache length is supported, including partial final tiles.
- `--specialization auto` (default) uses static `(Sk,H)` shapes **(64,8), (65,3),
  (1024,8), (4097,8)** when available; otherwise it uses runtime dimensions.
  `static` requires one of those shapes; `runtime` keeps dimensions dynamic.
- The cache contains only visible tokens. Cache management, paging and grouped-query
  attention are outside this example.

From `cuda-oxide/`, once per checkout:

```bash
cargo oxide toolchain install cutlass
python3 -m venv cute-rs-preview/.venv
cute-rs-preview/.venv/bin/pip install -r cute-rs-preview/requirements.txt
```

Compare runtime and static specialization on RTX 5090:

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 \
CUTE_DSL_ARCH=sm_120a CUTE_DSL_COMPILER_OPT=--opt-level=2 \
  cargo oxide run fmha_decode_sm120a_cute --arch sm_120a -- \
  --all-cases --warmup 100 --iters 31
```

Both implementations check every output against a CPU FP64 oracle. The command
also verifies matching input hashes and output parity, then prints two rows of
CUDA-event medians at the default **Sk=1024, H=8** shape. Each interval covers
the partial and merge kernels. `--all-cases` requires a listed static shape and
overrides `--specialization`; attention is noncausal and single-token.

For one case, replace `--all-cases` with `--specialization static` or `runtime`.
Add `--json PATH` to save the comparison, `--iters 0` to verify without timing,
or `--rust-only` for a standalone Rust report. `--all-cases` cannot be combined
with `--rust-only`. Pass `--help` after `--` for all options.

Use `--python PATH` to select another interpreter and `--device N` to select
another visible GPU; device 0 is the default.
The CuTeDSL reference is independently authored here using the same split-KV
algorithm; it is **not an upstream CUTLASS decode kernel**.

RTX 5090 results at `a0d5776b2` (2026-10-18), **B=1, Sq=1, Sk=1024, H=8, D=64**,
CUTLASS 4.7 at optimization level 2 on both sides. Each value is the median of
**31 single-invocation samples after 100 warmups**, with reused buffers.
Both implementations passed correctness checks. Times are microseconds:

| Shapes | CUDA events: Rust (cute-rs) | CUDA events: CuTeDSL | Nsight kernel sum: Rust (cute-rs) | Nsight kernel sum: CuTeDSL |
| --- | ---: | ---: | ---: | ---: |
| Runtime | 4.768 | 11.808 | 3.392 | 3.520 |
| Static | 4.416 | 11.808 | 2.880 | 2.656 |

Both metrics include the partial and merge kernels. CUDA events include gaps
during host submission; Nsight Systems sums the two GPU execution durations,
excluding those gaps but including device stalls. These are separate runs;
subtracting columns does not isolate host overhead. Clocks were unlocked and
CuTeDSL ran first in each case. This single run per method does not establish
a stable speed ranking.

- [CuTeDSL reference, setup and validation commands](reference/README.md)
