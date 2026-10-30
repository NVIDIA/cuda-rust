# CuTeDSL Blackwell FMHA reference

[`fmha.py`](cute/blackwell/kernel/attention/fmha/fmha.py) and its complete local
helper closure are unchanged copies of the public CUTLASS 4.7 example at
commit `dcf215af68a2d08d305076c152a06f201728cd53`. Their original BSD-3-Clause
notices remain intact. [SOURCE.md](SOURCE.md) records upstream links and SHA-256
digests; the comparison runner verifies those digests before execution.

This is the ordinary CuTe/NVVM programming model using `@cute.jit`, TMA,
`tcgen05`, TMEM, and fused online softmax for data center Blackwell. The Rust
kernel is `sm100_fmha_prefill_cute_rs`; [`../compare.py`](../compare.py) invokes
the pinned source through `sm100_fmha_prefill_cutedsl`. The `cute_ext` GQA
examples use a separate LIR programming model. The Rust kernel combines
structured CuTe operations with explicit CTM instructions; see the
[programming boundary](../README.md).

## Setup and matched comparison

Use Python 3.12 and the shared environment pinned in
[`cute-rs-preview/requirements.txt`](../../../requirements.txt); the recorded B200
results used torch `2.13.0+cu130`. From `cuda-oxide/`:

```bash
python3 -m venv cute-rs-preview/.venv
cute-rs-preview/.venv/bin/pip install -r cute-rs-preview/requirements.txt
```

The Rust runner selects `CUDA_OXIDE_CUTE_PYTHON`, then this example's `.venv`,
then the shared `cute-rs-preview/.venv`, then `python3`. Build the Rust example as shown
in the [example README](../README.md). To invoke the comparison directly from
the example directory:

```bash
../../.venv/bin/python compare.py \
  --rust-bin target/release/sm100_fmha_prefill_cute \
  --shape 1,257,385,4,2 --causal --check-only
```

Comparison runs print the CuTeDSL and CuTe-RS median invocation times on two
lines. `--json FILE` saves the complete report; `--check-only` prints PASS for
each implementation after validation, without timing. Diagnostics go to stderr.

The selected profile uses FP16 contiguous BSHD tensors, head dimension 128,
FP32 QK/PV accumulation, FP16 probabilities and output, MHA/GQA, and either
dense or top-left causal attention. Sequence lengths are positive runtime
parameters and may include partial tiles. The scale is `1/sqrt(128)`.
The public source supports additional profiles; the shared
[feature-gap roadmap](../../sm100_mla_decode_fp16_cute/reference/attention-feature-gaps.md)
separates omitted kernel/host functionality from compiler work. Unsupported
flags and shapes are rejected before allocation.

## Validation

From `cuda-oxide/`:

```bash
cute-rs-preview/.venv/bin/python -m unittest discover -s cute-rs-preview/examples/sm100_fmha_prefill_cute -p test_compare.py
cute-rs-preview/.venv/bin/python cute-rs-preview/examples/sm100_fmha_prefill_cute/validate.py --rust-bin /path/to/binary
cute-rs-preview/.venv/bin/python cute-rs-preview/examples/sm100_fmha_prefill_cute/validate.py --rust-bin /path/to/binary \
  --case both-tails-gqa --sanitizer memcheck
cute-rs-preview/.venv/bin/python cute-rs-preview/examples/sm100_fmha_prefill_cute/validate.py --rust-bin /path/to/binary \
  --case kv-ring-wrap --sanitizer synccheck
```

Validation covers one-key attention, causal masks, GQA, tails, all stage-ring
wraps, and more work tiles than resident SMs. Subprocess timeouts bound barrier
failures.
