# FP16 CTM architecture and historical validation

The [entry README](../README.md) contains current run commands and performance
comparisons. This directory preserves the CTM implementation details and the
**2026-10-11** investigation. Its Python timings used an older local reference,
not the current checked-in
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

## Historical B200 measurements: 2026-10-11

The [original report](results/b200-validation-2026-10-11.json) is preserved
byte-for-byte. Its relative `protocol.reference_file` path is recorded from the
original example directory, before this report moved. It identifies the old,
ignored `fp16_gemm_256x352_cute/reference/fp16_gemm_3_256x352.py` reference and
its SHA-256. That file is absent from a fresh checkout. Absolute `/tmp` paths
inside the report identify the original measurement artifacts; they are not
checked-in evidence or paths needed by the current public reference.

The report used CUTLASS 4.7.0, driver 615.71.09 and a 1,000 W power limit.
The final comparison used M8192/N8096/K8192 without bias, seed 0, 2,000 warmups,
301 timed launches, 74 clusters and direct CUDA events. Each table entry
summarizes three run medians. All comparisons verified both outputs against
the independent PyTorch oracle using identical input bytes.

| Historical implementation | Median (µs) | Run range (µs) |
| --- | ---: | ---: |
| Final Rust CTM | 778.976 | 776.864–780.576 |
| Previous local Python CTM, paired with final Rust | 779.456 | 771.168–779.872 |
| Earlier Rust CTM candidate with macros | 842.464 | 836.352–849.312 |
| Original Rust CTM | 1,846.848 | 1,846.624–1,847.072 |

The final event timings overlapped the Python reference's run range. These
runs were separate from profiling and did not record ambient clock state.
The earlier-candidate and original rows were measured in different runs;
they preserve the history of the implementation.

### Compiler and descriptor investigation

Hardware profiling identified local-memory traffic from temporary
enum/`Result` values. Moving the compiler's enum scratch allocations to the
function entry block allowed scalar replacement to eliminate that traffic
while preserving validation and error handling. Computing shared-memory
descriptors across the issuing warp, outside elected-lane branches, reduced
executed instructions further. Each of the eight MMA instructions and the
completion commit still executes in an elected lane, matching the CUTLASS
primitive sequence.

A separate Nsight Compute comparison used controlled SM clocks of approximately
845 MHz. These profile timings must not be mixed with the normal event timings:

| Historical profiled implementation | Time (ms) | Executed warp instructions (millions) | Tensor-pipe activity (% elapsed) | Local-memory sectors (millions) |
| --- | ---: | ---: | ---: | ---: |
| Rust CTM before compiler/descriptor fixes | 1.420960 | 52.641 | 74.760 | 12.389 |
| Final Rust CTM | 1.160704 | 32.277 | 91.517 | 0 |
| Previous local Python CTM | 1.133056 | 15.684 | 93.609 | 0 |

All three profiles reported zero register spilling. The eliminated traffic
came from temporary local storage. The final Rust profile still took 2.44%
longer than the local Python reference at the controlled clock, so the normal
event results did not establish identical performance.

### Correctness and migration checks

The original passed the six numerical cases, but memcheck reported a stack
overflow in its outlined epilogue. The final kernel passed all six cases,
memcheck with tensor-operation checks, and synccheck. Two additional fractional
FP16 input cases, including bias and N/K tails, passed a CPU float64 oracle
with maximum absolute errors of 0.00390625 and 0.0078125 using the existing
tolerance. The primitives compiler check verified no CuTe operations/types,
the eight paired MMA issues, the TMEM load/wait, and the direct kernel ABI.
Its final fresh-build cubin was byte-identical to the measured cubin.

The Rust CuTe migration was checked separately: its three-run median moved
from 762.624 to 764.672 µs, within the measured run ranges. It passed the same
six cases and sanitizers; its fractional outputs matched the final CTM output
bytes. Rebuilding it with the final compiler produced the same cubin. This
was a migration regression check, separate from the CTM performance baseline.
