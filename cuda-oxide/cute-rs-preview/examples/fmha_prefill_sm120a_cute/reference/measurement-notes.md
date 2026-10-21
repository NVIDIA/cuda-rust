# Prefill validation and performance notes

Build reproduction, numerical checks, timing methodology, and optimization
history for the [prefill example](../README.md). Paths and commands below
refer to the example directory unless another working directory is specified.

For the current matched CUDA-event and Nsight kernel timings, see
[Nsight measurements](nsight-measurements.md). The
[profiling follow-up](profiling-notes.md) and [epilogue-results.json](epilogue-results.json)
record the earlier load-scheduling improvement. The evidence below is historical.

## Implementation

This cuda-oxide example computes `softmax(Q @ K.T / sqrt(64)) @ V` with
64-by-64 tiles, warp-level FP16 MMA, FP32 accumulation, and online softmax.
The launch uses 128 threads and 24,576 bytes of shared memory per block.
It targets the warp MMA path available on RTX 5090, without `tcgen05` or TMEM.

The initial interface fixes batch size to one and head dimension to 64.
Q/K/V and the FP16 output use contiguous BSHD order. Sequence lengths must
be positive multiples of 64, with `Sq=1` additionally accepted for noncausal
attention. Causal mode is the default and currently
requires equal query and KV lengths; noncausal FMHA permits rectangular
sequences. Q/K/V use the same number of heads.

The [public CuTeDSL reference](README.md) is vendored unchanged
from CUTLASS 4.7, including its BSD license. `reference/launch.py` constructs
the same inputs and invokes that original kernel class. The Rust device
implementation is in `src/kernel.rs`; the CPU fixture and oracle are in
`src/fixture.rs`. The host checks and timing live in `src/runner.rs`.
The separate [decode example](../../fmha_decode_sm120a_cute/README.md) has dedicated
split-KV kernels and its own runner; only the CPU fixture and oracle are shared.

Both sides use CUTLASS 4.7.0: Rust's pinned compiler library and the matching
CuTeDSL package. The comparison measures these complete stacks on identical
inputs, tiles, precision and masks.
Both implementations specialize the causal flag at compilation. Rust uses a
`CAUSAL` const generic and dispatches to the matching compiled kernel.
`--shape-mode runtime` is the default: sequence lengths and head count remain
runtime parameters on both sides. `--shape-mode static` fixes `Sq=Sk=1024`
and `H=8` in both compilers; other shapes are rejected in this mode. Rust uses
const generics, while the Python launcher preserves concrete tensor shapes
and strides. The vendored reference kernel stays unchanged. These modes
describe kernel specialization, not measurements of compilation time.

The first KV tile initializes online softmax and handles the causal diagonal.
Later tiles use an unmasked loop. Checked, complete K/V tiles are represented
as 64-by-64 views so the compiler can remove redundant copy bounds checks;
Q retains bounds checks for the supported single-query case. The explicit
FTZ exponential matches the reference's `math.exp2(fastmath=True)`, including
flushing subnormal results to zero.
K-major B-fragment loads produce the MMA register order directly. Output
conversion uses nearest-even packed FP16 conversion, and shared-tile offsets
use 32-bit arithmetic before conversion to pointer offsets.

## Build and run

The [example README](../README.md) shows the normal command, which now compares
both implementations with matched direct CUDA events. The direct binary
commands here use `--rust-only` to retain the standalone reports used by this
profiling workflow.

From `cuda-oxide/`, using an already configured CUTLASS backend:

Rebuild the CLI and backend from this branch first. The example requires the
new FP16 MMA, copy/load, softmax, slice-pointer and scalar-parameter mappings;
an older installed backend cannot compile it.

```bash
export CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir
export CUDA_OXIDE_MLIR_PROFILE=cutlass-full-cute-mlir22
cargo oxide build --arch sm_120a --device-codegen-crate fmha_prefill_sm120a_cute \
  --cargo-target-dir /tmp/cuda-oxide-fmha-prefill -- \
  --manifest-path cute-rs-preview/examples/fmha_prefill_sm120a_cute/Cargo.toml --release

# Physical GPU 1 becomes ordinal 0 inside the process.
CUDA_VISIBLE_DEVICES=1 /tmp/cuda-oxide-fmha-prefill/release/fmha_prefill_sm120a_cute \
  --rust-only --seq-q 1024 --seq-k 1024 --heads 8 --causal --json /tmp/prefill-rust.json

# Two checked launches, without timing, for noncausal FMHA.
CUDA_VISIBLE_DEVICES=1 /tmp/cuda-oxide-fmha-prefill/release/fmha_prefill_sm120a_cute \
  --rust-only --seq-q 128 --seq-k 128 --noncausal --warmup 0 --iters 0
```

From this example's directory, with the matching Python environment described
under `reference/`, run:

```bash
CUDA_VISIBLE_DEVICES=1 CUTE_DSL_ARCH=sm_120a python reference/launch.py \
  --seq-q 1024 --seq-k 1024 --heads 8 --causal --json /tmp/prefill-cutlass.json

CUDA_VISIBLE_DEVICES=1 CUTE_DSL_ARCH=sm_120a python validate.py \
  --binary /path/to/fmha_prefill_sm120a_cute --reference --output-dir /tmp/prefill-validation

CUDA_VISIBLE_DEVICES=1 CUTE_DSL_ARCH=sm_120a python validate.py \
  --binary /path/to/fmha_prefill_sm120a_cute --reference --shape-mode static \
  --output-dir /tmp/prefill-static-validation
```

## Numerical checks

Inputs are signed dyadic FP16 values derived from a wrapping 32-bit hash.
The default range is `[-0.25, 0.2421875]`. `--input-scale 4` or `16` scales
Q/K by an exact power of two while leaving V unchanged; `--zero-query`
exercises uniform softmax. Both implementations report FNV-1a hashes over
the exact little-endian input and output bytes.

Every output element must be finite and satisfy
`abs(actual - expected) <= 2e-4 + 2e-3 * abs(expected)`. The independent CPU
oracle evaluates stable softmax and both matrix products in FP64. It keeps
probabilities in FP64; the tolerance accounts for the device's FP16
probability conversion and final FP16 output rounding. This checks
mathematical attention rather than requiring identical rounding schedules.
The output starts as NaNs and is poisoned again before a second launch;
that replay must reproduce the checked output bits exactly.

## Timing methodology

Timing is optional. The default uses 100 warmups and 31 single-kernel CUDA
event samples with reused input buffers. Compilation, CPU verification,
transfers, and output checking are outside the event intervals. JSON records
every sample and its median, device identity, launch dimensions, tolerance,
and validation results. `--iters 0` emits a null timing metric and performs
only validation. These direct-event measurements do not claim cold-cache or
graph throughput. `--output PATH` writes the checked output as raw FP16 BSHD.

For the separate profiling comparison, use Nsight Systems kernel-active durations:

```bash
CUDA_VISIBLE_DEVICES=1 CUTE_DSL_ARCH=sm_120a python reference/compare.py \
  --rust-binary /path/to/fmha_prefill_sm120a_cute --python /path/to/python \
  --seq-q 1024 --seq-k 1024 --heads 8 --causal --output-dir /tmp/prefill-comparison
```

The runner alternates implementations over three rounds, each with 100 warmups
and 31 measured launches. It excludes the two correctness launches and warmups,
and reports the median of round medians. Direct-event samples remain separate.
Raw traces, SQLite, output checks and input hashes are retained. An optional
`--rust-build-manifest` records compiler/backend provenance and must contain a
`binary_sha256` matching the executable.
Add `--shape-mode static` to compare the two shape-specialized implementations.
The harness forwards the selected mode to both runners and rejects missing
or mismatched specialization metadata before reporting timings.

## Earlier optimization results

Results before the matrix-offset narrowing follow-up on RTX 5090 GPU 1, CUTLASS 4.7.0 on both sides,
`B=1, Sq=Sk=1024, H=8, D=64`, measured on 2026-10-12:

| Shape mode on both sides | Mask | CuTe-RS | CuTeDSL | Rust slower by |
| --- | --- | ---: | ---: | ---: |
| Runtime | Causal | 16.768 µs | 15.712 µs | 6.7% |
| Runtime | Noncausal | 16.512 µs | 15.744 µs | 4.9% |
| Static | Causal | 16.447 µs | 15.520 µs | 6.0% |
| Static | Noncausal | 15.744 µs | 15.520 µs | 1.4% |

The original approximately 16% gap has narrowed, but performance parity is
not established. These are three alternating rounds with 31 kernel-active
samples per implementation per round, after 100 warmups. They exclude
compilation and host launch overhead. See [optimization-results.json](optimization-results.json)
for build hashes, every timing sample, validation, and discarded experiments.
Static and runtime modes are separate comparisons: specializing Rust alone
would not be a fair comparison with the runtime-shape reference.

Original baseline on GPU 1, RTX 5090, on 2026-10-12;
`B=1, Sq=Sk=1024, H=8, D=64`:

| Workload | CuTe-RS kernel-active | CuTeDSL kernel-active |
| --- | ---: | ---: |
| Causal prefill | 18.176 µs | 15.648 µs |
| Noncausal FMHA | 18.240 µs | 15.680 µs |

The baseline Rust kernel used 179 registers per thread; the pinned 4.7 reference
uses 152 for causal and 151 for noncausal attention. The previously listed
128-register reference count came from a different compiler version and did
not describe these timings. The Rust cubin has no stack or local-memory
allocation. See the pinned build
hashes, per-round samples and validation in [benchmark-results.json](benchmark-results.json).

## Validation workflow and results

`validate.py` checks lengths 1, 64, 128, 192, and 1024, both masks, unequal
head-local inputs, rectangular noncausal shapes, large logits, and zero queries. With
`--reference`, it also checks that the input hashes agree and compares every
Rust and CuTeDSL output element. It retains per-case logs and JSON reports.
CPU-only Python fixture tests require no GPU:

```bash
python -m unittest discover -s reference -p test_fixture.py
```

The optimized build passed 15 runtime-shape cases and four static-shape cases
for Rust, CuTeDSL, and their pairwise output comparisons: 57 numerical checks.
All 16 Compute Sanitizer runs passed: memcheck, racecheck, synccheck, and
initcheck (global and shared memory), for both masks in both shape modes.
These results and the tested binary hash are in
[optimization-results.json](optimization-results.json).

The original recorded run passed all 13 cases for Rust and CuTeDSL, including their
cross-checks. Compute Sanitizer memcheck, racecheck, synccheck and initcheck
(global and shared memory) passed causal and noncausal 128-token Rust cases.
See [validation-results.json](validation-results.json).

## B200 handoff

For a later B200 session, public ordinary CuTe references are
`cute/blackwell/kernel/attention/fmha/fmha.py` plus `helpers/fmha_helpers.py`,
and `cute/blackwell/kernel/attention/mla/mla_decode_fp16.py` plus its sibling
`mla_helpers.py`, under CUTLASS's `examples/python/CuTeDSL/` tree. The former
uses one-CTA `tcgen05`; the latter provides two-CTA FP16 MLA split-KV decode.
Public `cute_ext/blackwell/attention/gqa_decode_simple.py` uses the
experimental/LIR frontend and is a different porting target. Internal DKG
GQA source is not included here.


## Historical artifact paths

The example/package was renamed to include the tested `sm_120a` target.
Existing benchmark and validation JSON records preserve the original package
names, paths and hashes from their measurement runs; they are historical
provenance, not build commands for the renamed package.
