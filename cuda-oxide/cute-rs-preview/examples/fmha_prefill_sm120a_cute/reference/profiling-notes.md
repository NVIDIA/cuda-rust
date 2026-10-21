# Prefill profiling results

The retained cuda-oxide implementation combines verified matrix-address
simplification with cooperative-copy thread bounds. On RTX 5090, physical GPU 1,
the remaining gap averages **1.38%** across four matched cases; full parity is
not established. Both implementations use SM120a and CUTLASS 4.7.0, with
`B=1, Sq=Sk=1024, H=8, D=64`. The vendored CuTeDSL kernel is unchanged.

## Retained changes

The compiler uses 32-bit matrix offsets only after proving every intermediate
and swizzle field lies in `0..=i32::MAX`. The existing unsafe API contract
supplies hardware lanes and complete in-bounds matrix windows. Negative,
oversized or otherwise unproven layouts retain 64-bit arithmetic; pointer
addition remains 64-bit.

Within that proven domain, a bounded exhaustive check permits a carry-free
split of window and lane offsets only when it equals the full composed layout
map for every legal window and lane. This handles nested layouts and fixed
offsets conservatively. The high window offset remains additive, exposing
constant shared-memory displacements; failed proofs use the general path.

Cooperative-copy lowering applies `thread % verified_thread_count`. This is
an identity for every valid caller and exposes the existing thread bound to
the compiler. General partial-atom zero filling remains intact. The measured
prefill kernels now emit full-byte `LDGSTS.E.BYPASS.128` copies; the reference
also carries the `LTC128B` cache hint.

The kernel applies the negative-infinity repair in every causal iteration,
matching CuTeDSL; only the first causal tile applies the diagonal mask.
The algorithm, tile sizes and arithmetic precision are unchanged. These remain
complete-implementation comparisons: Rust uses a 1D launch grid and CuTeDSL
a 3D grid, with equal CTA counts.

## Why the combination matters

The original causal assembly consumed 12 of 20 steady Q/K matrix loads with
at most one intervening instruction. Nsight Compute observed 1,449 static and
1,518 runtime causal HMMA short-scoreboard samples. Proven address narrowing
restored early loads and reduced those observations to zero in the isolated
candidate; its instruction count fell only about 1%.

Standalone address factoring and standalone copy bounding did not consistently
improve all four cases. Copy bounding alone reintroduced immediate matrix-load
dependencies. Combining them preserves early causal QK/PV loads while matching
the reference's reported copy shared-wavefront counts. The combined profiles
observed 0 static causal, 0 static noncausal, 3 runtime causal and 213 runtime
noncausal HMMA short-scoreboard samples. Runtime noncausal retains a localized
load-use wait. An earlier output-swizzle probe regressed before these
scheduling changes; the final combination was checked separately below.

[Combined profile analysis](address-copy-profile-analysis.json) records these
schedules, counters and build provenance. Its one-round exploratory timings
are separate from the repeated results below. Lower instruction/register counts
and fewer reported wavefronts do not establish proportional latency savings.
Zero observed stall samples does not prove zero actual stalls.

The output-address helper now separates disjoint row and column bits. Its
formula agrees with the original for all 4,096 shared-tile coordinates and
removes 32 instructions per kernel. Both runtime kernels and static noncausal
retain every instruction encoding before the epilogue. Static causal changes
register allocation, but preserves early matrix loads. The final profiles
show 0/0 static and 1/226 runtime HMMA short-scoreboard observations; runtime
noncausal remains localized. See [final profiles](epilogue-profile-analysis.json).
The [cache-hint probe](cache-hint-probe.md) showed no consistent meaningful
benefit and was not retained.

## Repeated results and validation

| Shapes | Attention | CuTe-RS | CuTeDSL | Remaining gap |
| --- | --- | ---: | ---: | ---: |
| Runtime | Causal | 15.871 µs | 15.648 µs | 1.43% |
| Runtime | Noncausal | 16.127 µs | 15.680 µs | 2.85% |
| Static | Causal | 15.519 µs | 15.455 µs | 0.41% |
| Static | Noncausal | 15.584 µs | 15.456 µs | 0.83% |

Each row is the median of three alternating round medians, using 100 warmups
and 31 Nsight Systems kernel-active samples per implementation per round.
The **1.38%** average weights the four percentage gaps equally.
[Final results](epilogue-results.json) preserve all 744 samples, commands,
hashes and validation reports. Nsight Compute replay durations are excluded.

The retained build passed 57 numerical checks: 15 runtime and four static cases
for Rust, CuTeDSL and their output cross-checks. All 16 sanitizer runs passed
across both masks and shape modes (memcheck, racecheck, synccheck and
global/shared initcheck). Independent CPU validation passed 235 tests:
95 exporter, 53 layout and 87 dialect tests, including 73,899 checked copy
transactions. Eight prefill host tests also passed.

The final results embed the independent `review.json`: all 14 source hashes
and build-artifact hashes matched, and all four kernels' instruction encodings
matched the profiled final candidate exactly. The final output-address
change received fresh numerical, sanitizer, host-test and profiler checks.

Subsequent host cleanup removed unused decode compatibility. The simplified
`cargo oxide run` commands passed correctness checks for both examples, and
all four prefill and ten decode kernel instruction streams still matched the
measured builds. Recorded source hashes describe the original measured snapshot.

## Historical evidence and reproduction

[Baseline and narrowing analysis](profile-analysis.json) and
[narrowing-stage results](profiled-results.json) preserve the earlier stage,
whose average gap was 2.30%. The [address/copy stage](address-copy-results.json)
measured 1.49% before the final output calculation. [Residual findings](residual-findings.md) describe
that earlier snapshot and the standalone bounded-copy probe; references there
to the retained build refer to that historical stage.

Nsight Compute collected source, warp-state, scheduler, instruction, launch,
throughput and memory sections for one post-warmup launch, with cache and clock
control set to `none`. Sampling is stochastic; category shifts cannot be added
or converted into portions of the elapsed-time gain. The remaining gap is not
fully attributed. Use the build, validation and matched comparison commands
in [measurement notes](measurement-notes.md).
