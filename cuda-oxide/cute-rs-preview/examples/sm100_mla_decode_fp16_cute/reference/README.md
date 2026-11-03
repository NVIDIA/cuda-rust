# Public CUTLASS CuTeDSL MLA decode reference

The files under `cute/blackwell/kernel/attention/mla/` are the unmodified public
CUTLASS v4.7.0 FP16 MLA decode kernel and helpers. [SOURCE.md](SOURCE.md) records
the exact upstream commit, file hashes and license notices. The first-party
[adapter](../cutedsl.py) verifies those hashes before compilation and exposes
`sm100_mla_decode_fp16_cutedsl`. The Rust-generated cubin exports
`sm100_mla_decode_fp16_cute_rs` and `sm100_mla_decode_fp16_reduce_cute_rs`.
The Rust kernels combine structured CuTe operations with explicit CTM
instructions; see the [programming boundary](../README.md).

The matched profile uses 128 heads, one query, latent dimension 512, RoPE
dimension 64 and 128-token pages. The latent cache supplies both K and V.
Inputs, probabilities and output are FP16; QK/PV accumulation is FP32 and LSE
is base-2. Attention is noncausal with explicit length masking. Batch and KV
lengths must be positive, lengths are uniform across the batch, and split
counts range from 1 to 256. Unsupported profiles fail before allocation or
launch. Softmax scale defaults to 0.0416 and output scale to 1.0.

Use Python 3.12 and the shared environment pinned in
[`cute-rs-preview/requirements.txt`](../../../requirements.txt). From `cuda-oxide/`,
create it once, then run from the example directory:

```bash
python3 -m venv cute-rs-preview/.venv
cute-rs-preview/.venv/bin/pip install -r cute-rs-preview/requirements.txt
cd cute-rs-preview/examples/sm100_mla_decode_fp16_cute
. ../../.venv/bin/activate
python compare.py --rust-bin target/release/sm100_mla_decode_fp16_cute \
  --batch 1 --kv-length 1024 --splits 1 --clusters 1 \
  --warmup 10 --iterations 100 --rounds 5 --json comparison-results.json
python validate.py --rust-bin target/release/sm100_mla_decode_fp16_cute \
  --json validation-results.json
python -m unittest discover -p 'test_*.py'
cargo test
```

The default `cargo oxide run` command calls `compare.py` with the executable
it just built. Python is selected from `CUDA_OXIDE_CUTE_PYTHON`, this example's
`.venv/bin/python`, the shared `cute-rs-preview/.venv/bin/python`, then `python3`.
The terminal ends with the CuTeDSL and CuTe-RS median times; reference diagnostics
go to stderr. `--json FILE` preserves the full report. `--iters`, `--windows`
and `--check-only` alias `--iterations`, `--rounds` and `--validation-only`.
Validation-only runs print `PASS` for both implementations without timing.

`compare.py` generates one exact set of little-endian input bytes with
nonidentity pages and supplies it to both implementations. Output and LSE are
checked against an independent FP64, length-masked oracle and each other.
The reported maximum absolute and RMS errors use `atol=0.01, rtol=1e-5`;
deterministic zero-query cases use a tighter absolute tolerance. `validate.py`
covers page permutations, tail keys, unused splits, pipeline-ring wrapping,
persistent cluster reuse and a non-unit output scale. Rust subprocesses have
a timeout to bound barrier failures.

Timing starts only after correctness passes. CUDA events surround repeated
complete invocations on the launch stream, including split reduction when
enabled, with identical warmups, iterations, rounds, cluster counts and hot
buffers. Both implementations compile at CUTLASS optimization level 2, as
requested by the upstream source; comparison rejects a Rust executable built
at another level. Compilation, allocation, transfers and validation are outside
timing. No CUDA graphs are used. Results retain every window and configuration;
event intervals include submission gaps and are distinct from profiler
kernel-active times. Both implementations are checked again after timing.

The [Rust kernel](../src/kernel.rs) retains two CTAs per cluster and 384 threads
per CTA: softmax warps 0–3, correction/epilogue 4–7, MMA 8, TMA 9, page-table
copy 10 and control 11. Group-two M128 MMA computes QK128×128×64 and
PV128×256×32. Pipeline stages are Q1, KV15, S2, P2, correction2, O1 and
page-table4, with the shared K/V ring aliased as in the reference. MMA ordering
is QK0, then QK-next/PV-current, then PV-last. Each CTA allocates 512 TMEM
columns and 232,448 bytes of dynamic shared memory. Register budgets are
192 for softmax, 208 for correction and 96 for other roles, with one resident
block per SM specified to activate redistribution. Packed FP32 pair arithmetic
retains the source operation order; deferred TMEM transfers wait at buffer
handoff boundaries. Layout-typed operands and TMA descriptors preserve K-major
Q/K/P and MN-major V interpretation through compilation.

For the serialized-input adapter mode, the Rust executable accepts
`--input-dir DIR --output-dir DIR` and reads
`config.json`, `q_latent.f16`, `q_rope.f16`, `c_latent.f16`, `c_rope.f16` and
`page_table.i32`. It writes `output.f16` and `lse.f32`. `--validation-only`
performs one complete invocation without timing. Use the comparison runner
for the independent oracle check; adapter mode checks input sizes,
page bounds, finite outputs and repeatability, and retains JSON on stdout.

The shared [feature-gap roadmap](attention-feature-gaps.md) records additional
attention functionality and compiler work. The shared
[architecture notes](attention-design.md) explain the two source schedules and
the reusable API boundaries.
