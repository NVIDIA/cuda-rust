# CuTe DSL FlashAttention v2 reference

`flash_attention_v2.py` is an unchanged copy of the public CUTLASS 4.7 example:

- Repository commit: `dcf215af68a2d08d305076c152a06f201728cd53`.
- [Original source](https://github.com/NVIDIA/cutlass/blob/dcf215af68a2d08d305076c152a06f201728cd53/examples/python/CuTeDSL/cute/ampere/kernel/attention/flash_attention_v2.py).
- SHA-256: `263df1fd783729286cd6fa5ec994ec86e7614837bd1ff204eb6687192a8bae2d`.
- License: BSD-3-Clause; the original copyright, conditions, and disclaimer
  remain in the source header.

This is the ordinary CuTe DSL implementation using `@cute.jit`, `cp.async`,
warp-level FP16/BF16 MMA, and fused online softmax. It requires no other
repository helper files. The installed DSL runtime has its own NVIDIA EULA;
runtime source is not vendored here.

Use Python 3.10 or later supported by CUTLASS DSL 4.7, a CUDA-enabled PyTorch
build supporting the selected GPU, CUDA bindings, and `nvidia-cutlass-dsl==4.7.0`.
The shared [`requirements.txt`](../../../requirements.txt) selects the CUDA 13
package extra. The upstream CUDA 12.9 alternative is `nvidia-cutlass-dsl==4.7.0`
without that extra. Match the CUDA runtime, toolkit, and driver to the package
variant.

Prepare the shared Python environment once for both prefill and decode,
from `cuda-oxide/`:

```bash
python3 -m venv cute-rs-preview/.venv
cute-rs-preview/.venv/bin/pip install -r cute-rs-preview/requirements.txt
```

Select a CUDA-enabled PyTorch build that supports SM120 and matches your CUDA
runtime. The normal `cargo oxide run` command finds the shared environment
automatically; no activation or Python environment-variable export is needed.
Python selection is `--python PATH` after `--`, `CUDA_OXIDE_CUTE_PYTHON`, the
example's `.venv/bin/python`, the shared `cute-rs-preview/.venv/bin/python`, then `python3`.
See the [example README](../README.md) for direct-event comparisons.
The main command with `--all-cases --warmup 100 --iters 31` prints direct
CUDA-event medians for all four runtime/static and causal/noncausal cases.
These intervals can include host launch/submission delays, which Nsight Systems
kernel durations exclude. Comparing those two methods does not establish a
regression, and runs with different warmup and sample counts use different
sampling protocols.

`compare.py` retains the separate Nsight Systems profiling workflow and
invokes the Rust binary with `--rust-only`. With Nsight Systems installed,
reproduce its four-case measurement protocol from `cuda-oxide/` using the
shared Python environment:

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
  cargo oxide build --arch sm_120a --device-codegen-crate fmha_prefill_sm120a_cute \
  --cargo-target-dir /tmp/cuda-oxide-fmha-prefill -- \
  --manifest-path cute-rs-preview/examples/fmha_prefill_sm120a_cute/Cargo.toml --release

for shape in runtime static; do
  for mask in causal noncausal; do
    CUTE_DSL_ARCH=sm_120a cute-rs-preview/.venv/bin/python \
      cute-rs-preview/examples/fmha_prefill_sm120a_cute/reference/compare.py \
      --rust-binary /tmp/cuda-oxide-fmha-prefill/release/fmha_prefill_sm120a_cute \
      --seq-q 1024 --seq-k 1024 --heads 8 --shape-mode "$shape" --"$mask" \
      --warmup 100 --samples 31 --rounds 3 \
      --output-dir "/tmp/prefill-nsys-$shape-$mask"
  done
done
```

The harness reports the median of round medians and retains traces and raw
samples in fresh evidence directories. Hardware state, clocks and software
versions can affect new results; reproducing the protocol does not guarantee
the archived numbers.

From this directory, run the causal prefill workload on the default device:

```bash
CUTE_DSL_ARCH=sm_120a ../../../.venv/bin/python flash_attention_v2.py \
  --dtype Float16 --batch_size 1 --num_head 8 \
  --seqlen_q 1024 --seqlen_k 1024 --head_dim 64 \
  --m_block_size 64 --n_block_size 64 --num_threads 128 \
  --softmax_scale 0.125 --is_causal
```

Omit `--is_causal` for the noncausal FMHA workload. Both commands retain the
upstream PyTorch output check. The default benchmark uses three warmups and
ten iterations; add `--use_cold_l2` only when comparing the corresponding
cache policy.

The selected tile allocates three 64-by-64 FP16 shared-memory arrays, totaling
24 KiB. Each array is already a multiple of its 1024-byte alignment. This fits
the SM120 per-block limit of 99 KiB documented in the
[Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/).
The source's `can_implement` checks the larger SM80 capacity, so other tile
choices must separately respect the actual target's capacity.

The reference accepts FP16/BF16 with a head dimension divisible by eight,
16-byte contiguous alignment, a warp-multiple thread count, and
`2 * m_block_size` divisible by the thread count. Head dimensions are padded
to multiples of 32 for storage. It implements equal Q/K/V head counts and
optional top-left causal masking; it produces no training LSE output and
does not implement GQA, paged KV, or split-KV decode. Its use of the older
warp MMA instruction makes this an SM120 candidate without `tcgen05` or TMEM;
compatibility and performance still require a GPU run.

`launch.py` invokes this unchanged kernel with the Rust example's deterministic
inputs and an independent CPU FP64 oracle. It supports the same `--seq-q`,
`--seq-k`, `--heads`, `--causal`/`--noncausal`, `--input-scale`, `--zero-query`,
`--warmup`, `--iters`, `--output`, and `--json` options as the Rust host.
`benchmark.py` is the same checked runner. The original script's own CLI above
uses its upstream random input generator instead of the shared fixture.

## Validation

`validate.py` in the example directory checks lengths 1, 64, 128, 192 and
1024, both masks, unequal head-local inputs, rectangular noncausal shapes,
large logits and zero queries. With `--reference` it also checks that the
input hashes agree and compares every Rust and CuTeDSL output element,
retaining per-case logs and JSON reports. From the example directory, with
the shared environment's interpreter as `python`:

```bash
CUTE_DSL_ARCH=sm_120a python validate.py \
  --binary /path/to/fmha_prefill_sm120a_cute --reference --output-dir /tmp/prefill-validation
CUTE_DSL_ARCH=sm_120a python validate.py \
  --binary /path/to/fmha_prefill_sm120a_cute --reference --shape-mode static \
  --output-dir /tmp/prefill-static-validation
python -m unittest discover -s reference -p test_fixture.py
```

`--rust-only` runs the Rust binary without the comparison; with `--warmup 0
--iters 0` it performs two checked launches without timing.
