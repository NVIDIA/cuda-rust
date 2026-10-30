# Public Python CTM reference

[gemm.py](gemm.py) implements FP16 GEMM with public CUTLASS 4.7.0
`cutlass.experimental.primitives` operations for TMA, barriers, TMEM and MMA.
CuTe supplies compilation, runtime tensor wrappers, and epilogue layouts and
vector copies. This is the Python CTM comparison for both
[Rust CTM](../../fp16_gemm_256x352_ctm/README.md) and
[Rust CuTe](../README.md); it is not a second high-level CuTe GEMM port.
The CLI label is `CUTLASS (CuTe + CTM)`. [SOURCE.md](SOURCE.md) records provenance
and licenses. These experimental APIs are pinned in [requirements.txt](requirements.txt).

## Python setup

The comparison uses the shared environment pinned in
[`cute-rs-preview/requirements.txt`](../../../requirements.txt). From `cuda-oxide/`:

```bash
python3 -m venv cute-rs-preview/.venv
cute-rs-preview/.venv/bin/pip install -r cute-rs-preview/requirements.txt
```

The CuTe executable honors `--python PATH`, then `CUDA_OXIDE_CUTE_PYTHON`, then
this example's `.venv`, then the shared `cute-rs-preview/.venv`. Without any of
them, it creates the shared environment itself through `setup_reference.py`,
which needs `python3` with `venv`/`ensurepip` (`python3-venv` on Debian/Ubuntu)
or `uv` on `PATH`, plus network access. `--rust-only` and `--help` bypass Python
setup. Invoking `compare.py` directly uses the interpreter that starts it.

The CTM executable is a standalone Rust runner. Its comparison command invokes
this shared Python harness with `--rust-bin` pointing at the CTM executable.
Both entry READMEs give complete commands. No ignored `../reference/` files
are needed for either comparison.

## Inputs, correctness and output

The harness generates seeded FP16 inputs once and exports those exact bytes to
Rust. A and C are contiguous; B is stored as contiguous `[N,K]` and viewed as
`[1,K,N]` by the oracle. Optional bias has one FP16 value per output row.
M must be a positive multiple of 256, and N/K of 8. TMA handles partial N352
and K64 tiles. Index/allocation bounds are checked before launch.

[oracle.py](oracle.py) computes an independent FP32 PyTorch result with TF32
and autocast disabled, adds bias, then rounds once to FP16. Both implementations
must pass `atol=0.1, rtol=1e-5`. `--json` saves `verification: "both-pass"`,
input hashes, binary/reference hashes, launch settings and both timings.
A standalone Rust result marked `external-oracle-required` is not independent
verification. JSON keys remain `cuda_oxide` and `python_reference`; the Rust
`implementation` distinguishes `cute-rs` and `ctm-rs`.

Diagnostics go to stderr. The last two stdout lines identify the Python and
Rust implementations, with microseconds and TFLOP/s. The latter uses
`2 * M * N * K / time`; optional bias is not counted as additional FLOPs.
The default tiny shape is useful for correctness and launch overhead, while
`8192,8096,8192` is the throughput case used in the performance tables.

## Timing options

Comparisons use direct CUDA events: one kernel launch per sample, followed by
synchronization. Each event pair uses the actual launch stream. Python runs
first; input export, correctness work and Rust process startup separate the
implementations. The recorded post-timing clock snapshot is not a measurement
of clock state throughout either timing interval.

Standalone Rust timing defaults to graph replay, with `--graph-launches`
launches per sample. Use `--timing-mode direct --warmup 0 --iters 1` for exactly
one Rust launch, then verify its output. With the CuTe executable, add
`--rust-only`. CPU verification is exhaustive and expensive for large matrices;
use the comparison harness for those shapes. With nonzero warmup, Rust also
performs one preliminary untimed launch, while Python uses exactly the requested
warmups. The throughput tables use 2,000 warmups; cold-start comparisons need
this distinction.

## Validation and known limitation

Six public Python cases (explicit streams and graph replay), six Rust CuTe
cases, and four fractional/cancellation/bias-rounding comparisons pass their
numerical checks; the fractional Rust/Python outputs match byte-for-byte.
Rust CuTe memcheck and synccheck pass, as does Python synccheck.

**The Python reference is not memcheck-clean.** Compute Sanitizer 2026.3 reported
an invalid remote-shared write involving a group-two TMA peer barrier. A local
self-multicast experiment removed that diagnostic, but is not applied to the
checked-in kernel. A sanitizer compatibility issue is suspected, not confirmed.
Correctness passes do not resolve it.

From the CuTe example directory, after setup:

```bash
../../.venv/bin/python -m unittest discover -v
../../.venv/bin/python -m public_reference.validate --output /tmp/fp16-reference-checks.json
compute-sanitizer --tool memcheck --check-tensor-ops yes --error-exitcode 1 \
  ../../.venv/bin/python -m public_reference.validate
compute-sanitizer --tool synccheck --error-exitcode 1 \
  ../../.venv/bin/python -m public_reference.validate
```

For Rust CuTe, from `cuda-oxide/`, after building:

```bash
python3 cute-rs-preview/examples/fp16_gemm_256x352_cute/validate.py --sanitizer
```

This standard-library runner covers full tiles, row bias, N/K tails, persistent
tiles, five-stage reuse, phase wraps and graph replay. `--binary PATH` selects
a nondefault build location; omit `--sanitizer` for numerical checks alone.
[Rust CTM validation and historical evidence](../../fp16_gemm_256x352_ctm/public_reference/README.md)
are documented separately.
