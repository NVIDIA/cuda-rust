# FP16 CTM architecture and validation

The [entry README](../README.md) contains the run commands and performance
comparisons. This directory describes the CTM implementation and its
validation commands; the comparison uses the checked-in
[public Python CTM implementation](../../fp16_gemm_256x352_cute/public_reference/gemm.py).

## Instruction schedule

The [kernel](../src/kernel.rs) expresses the schedule through typed `ctm_rs`
instructions and physical formats without CuTe operations or layout algebra.
Types distinguish operand roles, CTA groups and TMEM views; the kernel
coordinates election, phase changes, fences and lifetimes explicitly.
The types do not prove collective participation or asynchronous completion.

Two CTAs cooperate on each 256×352 output tile. Five A/B stages feed paired
256×192 and 256×160 MMA operations with FILL/LASTUSE collector reuse.
The accumulator occupies 512 TMEM columns, and two C stages serve the epilogue.
The [CuTe architecture notes](../../fp16_gemm_256x352_cute/public_reference/architecture.md)
describe the shared tile geometry and warp roles.

The kernel contains no local macros. Helpers are ordinary
`#[inline(always)]` functions, with const generics for K steps and output
chunks; the MLIR exporter preserves the inlining request for pointer-bearing
helpers. The epilogue writes four aligned 128-bit chunks per lane, and invariant
TMEM views are checked once per output tile.

## Build and validation commands

From `cuda-oxide/`, the primitives compiler check creates a fresh build and
records compiler hashes, MLIR, cubin and the build directory in its manifest:

```sh
python3 scripts/check-primitives-api-compile.py --example fp16 \
  --output-dir /tmp/primitives-fp16-evidence
```

On an SM100 GPU, pass that directory's
`target/release/fp16_gemm_256x352_ctm` executable to the validator:

```sh
python3 cute-rs-preview/examples/fp16_gemm_256x352_ctm/validate.py \
  --binary /path/to/fp16_gemm_256x352_ctm \
  --output-dir /tmp/fp16-results --sanitizer
```

The six cases exercise tails, bias, persistent tiles, five-stage reuse and phase
wraps. `--sanitizer` adds Compute Sanitizer memcheck with tensor-operation checks
and synccheck. The host supports direct/graph timing and an exhaustive CPU
oracle; see `--help`. Large shapes make the CPU oracle expensive.
This kernel requires SM100; the
[blockscaled CTM example](../../blockscale_gemm_ctm/README.md) covers SM120a.

The shared comparison harness accepts `--rust-bin` for this executable and
verifies both outputs against an independent PyTorch oracle using identical
input bytes. Its aggregate keys are `cuda_oxide` and `python_reference`.
Check `input_sha256`, cluster counts, launch counts and timing settings, and
require `verification: "both-pass"`. A standalone report marked
`external-oracle-required` is not an independent correctness result.
