# MXFP4 GEMM with Rust CTM (`ctm-rs`)

Detailed implementation, validation, and measurement history for the
[example](../README.md). Commands run from `cuda-oxide/` unless a section
specifies another working directory. Historical temporary paths identify
measurement artifacts; the checked-in JSON records preserve their original
paths and hashes.

SM120a port of [`blockscale_gemm_cute`](../../blockscale_gemm_cute) for the RTX 5090:
packed E2M1 inputs, UE8M0 scales per 32 K values, FP32 accumulators and FP16 output.
It retains the 128×128 tile, three-stage TMA ring, eight compute warps and one
producer warp, persistent scheduling, and shared-memory STMATRIX/TMA epilogue.
The fixed problem is M=N=2048, K=1024.

The kernel uses `ctm_rs`, with distinct A/B register and scale
carriers and typed shared formats/tensor maps. Election, physical lane addresses,
barrier phases and publication remain explicit Rust. No CuTe/Python dependency
or CuTe operations/types are required. Five grid-constant tensor maps replace the
CuTe example's device pointers to descriptor allocations.

The accumulator is an `AccumulatorTile16x8<2, 8>`. Compile-time fragment iteration
expresses one B-pair MMA body and one output-store body; the type checks fragment
coordinates and operand roles. The B-scale helper reads two aligned `u128` groups,
then selects typed scale pairs from `u32` words. Bounded shared-memory offsets use
32-bit arithmetic and a separate K-half selector.

## FP16 compiler fix checked on blockscale (2026-10-11)

Commit `3a00962d6` hoists private enum scratch allocations and preserves
`#[inline(always)]` on pointer-bearing helpers. Rebuilding this example with that
commit produces a **byte-identical cubin** to the cleanup build below. Blockscale
has no enum scratch to hoist; the 24 new inline attributes do not change its final
GPU code. Both builds include the same local accumulator/address cleanup.

Fresh Nsight Systems kernel-active timings use the same common host, workload
and actual CUTLASS CTM reference, with three rotating rounds of 100 warmups and
31 samples on GPU 1 (hot buffers, unlocked clocks):

| Build | Round medians (µs) | Median (µs) |
| --- | --- | --- |
| Rust before compiler fix | 11.328, 11.328, 11.360 | 11.328 |
| Rust after compiler fix | 11.359, 11.360, 11.328 | 11.359 |
| CUTLASS CTM | 10.592, 10.624, 10.624 | 10.624 |

The 0.031 µs before/after difference is measurement variation between identical
binaries. This fix does not close the roughly 6.9% gap to CUTLASS CTM here.
The freshly built Rust executable passes all 4,194,304 FP16 output bits and replay
checks at 170/1/17 CTAs; all 68 exporter tests pass. The identical cubin retains
the cleanup build's sanitizer coverage. [Results and hashes](compiler-fix-results.json).

## Cleanup and source optimizations (2026-10-11)

Fresh RTX 5090 measurements use the actual Rust and CUTLASS CTM cubins loaded by
one common host harness, with the same complete oracle and five tensor maps.
These are Nsight Systems **kernel-active** times for M=N=2048, K=1024, 170 CTAs:
three rotating-order rounds, 100 warmups and 31 samples per round, hot buffers,
unlocked clocks. Rust uses the pinned CUTLASS compiler 4.7; the CUTLASS CTM reference
uses 4.8. A separate controlled Rust-IR compiler-version experiment measured
11.712 µs with either compiler, so an upgrade alone did not remove the gap.

| Rust source / reference | Round medians (µs) | Median (µs) |
| --- | --- | --- |
| Before cleanup | 11.744, 11.679, 11.680 | 11.680 |
| Typed accumulator cleanup only | 11.584, 11.551, 11.551 | 11.551 |
| Cleanup plus vector scale loads | 11.488, 11.456, 11.456 | 11.456 |
| Final: also simplify address arithmetic | 11.360, 11.296, 11.328 | 11.328 |
| Actual CUTLASS CTM reference | 10.592, 10.592, 10.560 | 10.592 |

The final source reduces Rust time by 3.0%, leaving 0.736 µs (6.9%) versus CUTLASS
in this run. All 4,194,304 FP16 outputs match exactly at 170/1/17 CTAs, including
replay; memory/tensor checks and synchronization checks report zero errors.
[Recorded results](cleanup-results.json) separate these source builds from the
isolated IR experiments. CTA-local TMA lowering and the barrier-wait hint are
diagnostic variants, not changes included in this source. The earlier tables
below describe separate experiments and should not be combined numerically.

Hardware counters show executed warp instructions falling from 5.016 million to
3.988 million; CUTLASS executes 2.959 million. Scale-load instruction counts now
match CUTLASS. Neither profile shows local-memory traffic or register spilling;
both still allow one resident CTA per SM. The remaining difference includes
additional address/control work and the existing cluster-capable TMA route.
Matching CUTLASS's explicit barrier-wait hint slowed the combined experimental
variant (11.072 to 11.199 µs), so it was not adopted.

Related implementations live alongside this example: the
[CUDA C++ primitives reference](../../blockscale_gemm_cute/reference/cpp/),
[Python CuTeDSL reference](../../blockscale_gemm_cute/reference/dense_blockscaled_gemm_persistent_cooperative.py),
and its [benchmark harness](../../blockscale_gemm_cute/reference/benchmark.py).

From `cuda-oxide/`, with pinned CUTLASS compiler 4.7 installed:

```sh
python3 scripts/check-primitives-api-compile.py --example blockscale \
  --output-dir /tmp/primitives-blockscale-evidence
```

The manifest records the fresh executable directory, source/compiler hashes,
MLIR and validated cubin. Use the resulting
`run-*/target/release/blockscale_gemm_ctm` executable:

```sh
CUDA_VISIBLE_DEVICES=1 python3 cute-rs-preview/examples/blockscale_gemm_ctm/validate.py \
  --binary /path/to/blockscale_gemm_ctm \
  --output-dir /tmp/blockscale-results --sanitizer
CUDA_VISIBLE_DEVICES=1 /path/to/blockscale_gemm_ctm --json /tmp/blockscale-timing.json
```

Each validation case compares all 4,194,304 output bit patterns with the exact
CPU oracle, starts with NaN output and rechecks after repeated launches. Resident,
one-CTA and 17-CTA grids exercise persistent stage/phase wrapping. The optional
checks run memcheck with tensor-operation checking and synccheck. Timeout/failure
reports cannot reuse previous JSON output. The full CPU reference is substantial.

A normal run performs 100 warmups and 31 individual event-timed launches, matching
the CuTe example's protocol; flags are `--ctas`, `--warmup`, `--samples`, `--json`.
CUDA-event intervals include gaps between event execution and kernel execution;
they must not be compared with Nsight kernel-active durations. The report
distinguishes correctness from timing. Throughput is not a SOL claim.

## Local validation (2026-10-11)

On an RTX 5090, all 4,194,304 FP16 outputs matched the exact CPU oracle,
including repeat launches, with the resident 170-CTA grid and forced 1/17-CTA
grids. CUDA memcheck (`--check-tensor-ops yes`) and synccheck reported zero errors.
For this optimization, the fresh `cuda-device` suite passed 123 tests with
110 ignored and no failures. The production compile check and independent
code review also passed. Constructor tests cover valid pointer preservation
and invalid null, misaligned and undersized storage for all three shared formats
used here. Invalid-input tests use the host trap stub; generated MLIR was also
checked for validation branches ending in trap/unreachable.

## Earlier CuTe comparison (2026-10-11)

Nsight Systems 2026.3.2, driver 615.71.09, physical GPU 1 (RTX 5090),
M=N=2048, K=1024, L=1, 128×128×128 tiles, E2M1 inputs, UE8M0 scales per
32 K values, FP32 accumulation and FP16 output. Each round used 100 warmups
and 31 measured launches, hot buffer reuse, no graphs and unlocked clocks.
The three Rust implementations ran serially in rotating order over three rounds;
Python ran three rounds in one process afterwards. All durations below are
Nsight kernel-active time, excluding host submission gaps.

| Implementation | Run medians (µs) | Median of run medians (µs) |
| --- | --- | --- |
| Rust primitives before this optimization | 12.960, 12.992, 12.960 | 12.960 |
| Rust primitives with trapping constructor | 11.648, 11.680, 11.616 | 11.648 |
| CuTe-RS, CUTLASS compiler 4.7.0 | 11.264, 11.264, 11.232 | 11.264 |
| Python CuTeDSL 4.6.2 | 10.080, 10.112, 10.112 | 10.112 |

The optimization reduces primitives kernel time by 10.1%; it remains 3.4%
slower than CuTe-RS and 15.2% slower than Python CuTeDSL in this measurement.
The Rust runs check all 4,194,304 FP16 bits against the CPU oracle. Python uses
the upstream reference check (`atol=0.1`, `rtol=0.01`) before timing a separate
generated workspace. This comparison matches shape, formats and timing method;
compiler versions and correctness oracles differ. It does not establish SOL.

Earlier event timings of 15.328 µs (Rust primitives) and 13.600 µs (CuTe-RS)
used the same problem size. They are a different timing metric and are not
the baseline for the improvement above.

`SharedRegion::from_raw_parts_or_trap` retains the existing constructor's null,
alignment and capacity validation and unsafe storage/lifetime contract. It
directly constructs the typed region after validation, avoiding the
pointer-bearing `Result` carrier in kernels that already trap on failure.
The Result-returning API remains available. The final cubin has no direct calls
to the three shared-region helper specializations; conditional TMA syscall
fallback calls remain. Resource usage is 128 registers and 8 bytes of stack
(previously 126/8; CuTe-RS uses 118/0). Kernel scheduling and raw TMA semantics
are unchanged.

The Python row above is CuTeDSL. A separate
[matching CUTLASS CTM port](README.md) now uses the
actual `cutlass.experimental.primitives` API with the Rust fixture, exact oracle,
and physical schedule. It was written for this comparison because the complete
upstream blockscale primitives example targets SM100 TCGen05/TMEM.

To reproduce one Rust timing round, use its freshly built executable:

```sh
CUDA_VISIBLE_DEVICES=1 nsys profile --trace=cuda,nvtx --sample=none \
  --cpuctxsw=none --output=/tmp/blockscale-active \
  /path/to/blockscale_gemm_ctm --warmup 100 --samples 31
nsys export --type sqlite --output=/tmp/blockscale-active.sqlite \
  /tmp/blockscale-active.nsys-rep
```

In `CUPTI_ACTIVITY_KIND_KERNEL`, select the last 31 `blockscale_gemm` launches
ordered by `start` (there are 132 total: one correctness launch, 100 warmups,
31 samples). Their `(end-start)/1000.0` values are microseconds. Repeat serially
for three rounds and report the median of round medians. Python sample windows
are delimited by `cutedsl_samples_round_*` NVTX ranges in the local harness.

[The recorded results](../benchmark-results.json) preserve all samples, tool/source
hashes, correctness scope and provenance. Paths in those records refer to the
original locations and the former `blockscale_gemm_primitives` executable name.
The example first moved from
`crates/rustc-codegen-cuda/examples/blockscale_gemm_primitives/` to
`cute-rs-preview/examples/blockscale_gemm_primitives/`, then was renamed to
`cute-rs-preview/examples/blockscale_gemm_ctm/`. The historical names, paths and hashes
are unchanged. The current executable is `blockscale_gemm_ctm`.
The Python reproduction harness is now checked in at
[`reference/benchmark.py`](../../blockscale_gemm_cute/reference/benchmark.py), with
[setup and profiling instructions](../../blockscale_gemm_cute/reference/README.md).
The original combined-run harness remains at `/tmp/primitives-optimization/compare.py`;
the checked-in [comparison runner](../compare.py) handles the Rust/C++ binaries and
the new CUTLASS CTM executable harness.

Validated primitives cubin SHA-256:
`d55b1fe9b00bdaaf743165dd1c9fdfb046194536c094e4b48a6c53c0ff26b533`.
The sanitizer build and final formatted build have identical cubin hashes.
Local detailed evidence: `/tmp/primitives-optimization/final/manifest.json`,
`/tmp/primitives-optimization/validation/run-nuiwesaq/report.json`, and
`/tmp/primitives-optimization/timing/comparison.json` (alongside the Nsight traces).

## Custom CUDA C++ comparison after the main rebase (2026-10-11)

[`blockscale_gemm_cpp`](../../blockscale_gemm_cute/reference/cpp/) implements the same GEMM with
CUDA C++ instruction primitives. Its README and
[recorded results](../../blockscale_gemm_cute/reference/cpp/benchmark-results.json) document the
fresh comparison after rebasing onto main `d141118c05` and moving to
`nightly-2026-10-11`. From `cuda-oxide/cute-rs-preview/examples/blockscale_gemm_ctm/`,
use the checked-in runner to compare fresh executables:

```sh
python3 compare.py \
  --binary rust=/path/to/blockscale_gemm_ctm \
  --binary cpp=/path/to/blockscale_gemm_cpp \
  --output-dir /tmp/blockscale-comparison --device 1 \
  --ctas 170 --rounds 3 --warmup 100 --samples 31
```

This runs the binaries serially in rotating order, checks full bit-exact and
replay correctness in every process, then extracts the final 31 kernel-active
samples from each Nsight Systems trace. The earlier CuTe table above is a
separate experiment; use the fresh C++ comparison for the current Rust/C++ gap.

The post-rebase Rust cubin is byte-for-byte identical to the pre-rebase cubin
whose SHA-256 is recorded above. Main's new reference-parameter validity
attributes currently reach the native LLVM export route; the CUTLASS MLIR
export route used here drops `ReferenceParamValidityAttr`. No generated-code
speedup from those attributes is present in this kernel. Its five reference
parameters point to tensor-map descriptor objects; their validity does not
establish aliasing or size guarantees for the allocations encoded in those maps.
