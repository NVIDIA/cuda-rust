# CUTLASS CTM blockscale GEMM reference

This is a port of cuda-oxide's [Rust CTM GEMM](../)
using the actual `cutlass.experimental.primitives` API from CUTLASS 4.8.0.
The kernel is written in Python, as exposed by that upstream API; the Rust
implementation remains in `blockscale_gemm_ctm`. The older [C++ reference](../../blockscale_gemm_cute/reference/cpp/)
reference uses custom CUDA C++ inline PTX and has separate historical results.

The entry README shows the current
[Rust CTM versus this Python CTM comparison](ctm-python-measurements.md), with
both CUDA-event and Nsight kernel timings on the same inputs. The
[higher-level CuTeDSL comparison](nsight-measurements.md) is supplementary;
the historical CTM measurements below remain separate.

Both CTM kernels use M=N=2048, K=1024, packed E2M1 inputs, UE8M0 scales per
32 K elements, FP32 accumulation, and FP16 output. Physical offsets, the
128×128×128 tile, three pipeline stages, eight consumer warps plus one producer,
persistent tile traversal, and the STMATRIX/TMA epilogue match. Each launch has
288 threads, five grid-constant tensor maps, 86,016 dynamic shared bytes and
1,024 static shared bytes for alignment.

[`kernel.py`](kernel.py) calls upstream TMA, barriers, LDMATRIX, block-scaled MMA,
STMATRIX, and conversion operations. Its small MMA adapter packs/unpacks the
upstream API's raw LLVM register carriers. CuTe supplies decorators, tracing,
and launch/allocation infrastructure; there is no CuTe tensor/layout algebra
in this programming model. Compiler IR still contains CuTe launch/allocation
plumbing, so this is not a claim that every intermediate `cute.*` operation
has disappeared.

## Run and reproduce

From this directory, with Python 3.12 and an SM120 GPU:

```sh
python3.12 -m venv /tmp/cutlass-ctm-venv
/tmp/cutlass-ctm-venv/bin/python -m pip install -r requirements.txt
/tmp/cutlass-ctm-venv/bin/python -m unittest discover
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  /tmp/cutlass-ctm-venv/bin/python benchmark.py --ctas 170 \
  --warmup 100 --samples 31 --json /tmp/cutlass-ctm-result.json
```

The harness creates all inputs on the CPU, poisons output, checks every FP16
bit before timing, and checks replay afterward. It launches only GEMM kernels.
The independent original Rust fixture generator and this NumPy implementation
were compared byte for byte for all five complete buffers, including all
4,194,304 expected outputs. CLI JSON includes fixture/source/package hashes.

For fresh **Nsight kernel-active** measurements, build Rust with
`scripts/check-primitives-api-compile.py --example blockscale`, then use:

```sh
PATH=/tmp/cutlass-ctm-venv/bin:$PATH \
CUTE_DSL_KEEP=ir,ptx,cubin CUTE_DSL_DUMP_DIR=/tmp/cutlass-ctm-artifacts \
python3 ../compare.py \
  --binary rust=/absolute/path/to/blockscale_gemm_ctm \
  --binary cutlass="$PWD/benchmark.py" \
  --kernel-name 'cutlass=kernel_cutlass_blockscale_gemm_1...' \
  --output-dir /tmp/blockscale-ctm-comparison --device 1 \
  --ctas 170 --rounds 3 --warmup 100 --samples 31
```

The ellipsis is part of the literal Nsight 2026.3 `shortName`, not a wildcard.
Other profiler/compiler versions may name it differently; inspect their trace
and supply its exact value. Every process must contain exactly 132 launches of
its expected kernel, plus a fresh passing correctness report. Runs are serial
and rotate order; timing uses `(end-start)/1000` from CUPTI kernel activities.
CUDA event intervals in executable JSON include host submission gaps and are
a separate metric. The Python host rebuilds descriptors on each invocation;
that host work is outside the kernel-active measurement.

Use `../validate.py --binary "$PWD/benchmark.py"`
with the same Python PATH, GPU selection, `--output-dir`, and `--sanitizer` to
check 170-, 1-, and 17-CTA grids, memcheck with tensor checks, and synccheck.

## Scope and provenance

The official 4.8.0 wheel's primitives wrapper matches the local CUTLASS checkout
`0b55a2f691d69981583568fd9eb69687b1f0de8a` byte for byte (SHA-256
`e11a53bfe7dc6f27ea80d189ae84996147f153660d6d97ac1d8fcaede66c7499`).
This GEMM port is repository-authored, rather than an upstream tuned example.

The current Rust compiler is pinned to CUTLASS 4.7.0; this primitives API needs
4.8.0. Upstream barrier waits also emit an explicit 10,000,000-nanosecond
maximum suspension hint, while Rust omits that optional operand. Phase and
ordering semantics match; emitted instructions and compiler optimizations can
differ. Results compare these implementations/toolchains and do not isolate
language overhead or establish SOL. No global compiler pin was changed.

## Historical comparison (2026-10-11)

These measurements precede the Rust source cleanup. See the
[current paired comparison](ctm-python-measurements.md) for fresh results and
[measurement notes](measurement-notes.md) for the optimization history.

RTX 5090, physical GPU 1, 170 CTAs, Nsight Systems 2026.3.2. Three serial
rounds with rotating order, 100 warmups and 31 samples per round; hot buffer
reuse, no graphs, unlocked clocks. Both implementations passed the complete
bit-exact oracle and replay check in every timing process. Both also passed
170-, 1-, and 17-CTA checks, memcheck, and synccheck.

| Implementation | Round medians (µs) | Median of round medians (µs) |
| --- | --- | --- |
| Rust CTM, CUTLASS compiler 4.7.0 | 11.680, 11.648, 11.616 | **11.648** |
| CUTLASS primitives API, compiler 4.8.0 | 10.752, 10.752, 10.752 | **10.752** |

Rust takes 8.33% longer in this comparison, subject to the toolchain/wait-hint
qualifications above. Resource usage is 128 registers/8 stack bytes for Rust
and 122 registers/0 stack bytes for CUTLASS. Matching shared-memory allocation
is verified in both cubins. These results supersede using the custom-C++ row
as evidence about CUTLASS CTM performance.

[`benchmark-results.json`](benchmark-results.json) preserves every sample,
commands, source and cubin hashes, compiler versions, fixture parity evidence,
and correctness/sanitizer reports. Local raw traces are under
`/tmp/cutlass-ctm-correction/timing/run-h2597zg8/`.

The reference originally lived at `blockscale_gemm_cute/reference/ctm/`.
Paths and hashes inside the recorded JSON preserve that measurement-time
location; the executable and run instructions above use the current location.
