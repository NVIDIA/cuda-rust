# B200/SM100 MLA decode performance

For current CUDA-event and Nsight Systems kernel-duration comparisons, see the
[2026-10-18 measurements](nsight-measurements.md). The results below retain their
original historical conditions.

The historical measurements show CuTe-RS reaching parity with the pinned
CuTeDSL decode reference on both profiles; the small profile is faster.
Shapes, FP16/FP32 precision, length masks,
page permutations, split counts, two-CTA schedule and pipeline stage counts are
unchanged. The CuTeDSL source remains pinned and unmodified.

The historical reports linked below predate the CTM migration and describe the
source revision recorded in each artifact. Current migration evidence follows
in the next section; the reproduction commands validate and measure new builds.
Structured CuTe operations keep their existing dialect paths, while explicit
instruction calls now use CTM as described in the [example README](../README.md).

## CTM migration validation

The [migration report](results/ctm-migration/summary.json) records **9 passing
B200 correctness cases** against pinned CuTeDSL and the independent oracle on
2026-10-14, plus source hashes and the compiler comparison with `d642dcbfbf`.
Decode main and reduction kernel text is byte-identical to the baseline.
Both memcheck and synccheck report zero errors. This report is separate from
the historical performance results below.
Decode timing was not repeated because both kernel text sections are unchanged.

## Complete-invocation results

Measurements use B200/SM100a, driver 615.71.09, CUTLASS 4.7.0 at optimization
level 2, and CPU affinity 8. Both implementations receive identical input bytes
and pass correctness before timing. CUDA events enclose complete invocations,
including split reduction when enabled, with no CUDA graphs. Allocations,
compilation, input transfers and validation are outside the event interval.
GPU clocks are not locked; state snapshots and every timing window are retained.

Each profile was run in both timing orders. The table reports the median of all
10 windows: five per order, 10 warmups and 100 invocations per window. No samples
were discarded. These settings match the original recorded benchmark.

| Batch / KV length / splits / clusters | Previous CuTe-RS µs | Optimized CuTe-RS µs | CuTeDSL µs | CuTe-RS latency reduction vs CuTeDSL |
|---|---:|---:|---:|---:|
| 1 / 1024 / 1 / 1 | 29.613 | 16.775 | 18.022 | 6.9% |
| 16 / 8192 / 8 / 74 | 82.737 | 48.514 | 49.213 | 1.4% |

The previous column is the preserved initial measurement, not a new run of the
old executable. Optimized/reference ranges are respectively 16.728–16.793 /
17.992–18.317 µs for the small profile and 48.477–48.634 / 49.137–49.276 µs for
the large profile. The practical conclusion for large decode is parity.

A separate sustained check uses 2,000 warmups and 301 invocations per window,
seven windows in each order, retaining all 14 windows:

| Profile | CuTe-RS median µs (range) | CuTeDSL median µs (range) |
|---|---:|---:|
| Small | 16.745 (16.722–16.761) | 17.926 (17.910–17.947) |
| Large | 65.633 (63.661–69.815) | 66.047 (62.889–69.952) |

Both large kernels slow under prolonged warmup. Nsight Systems confirms that
this shift occurs in GPU kernel-active time, rather than only host submission
gaps: Rust main/reduction pairs move from roughly 48.7 to 65.6 µs in the trace.
The cause of that shared shift was not isolated. Parity holds in both regimes;
the overlapping sustained ranges do not establish a large-profile speedup.
Instrumented profiler durations are diagnostics, not substitutes for the
unprofiled CUDA-event measurements above.

All configurations, timing orders, samples, errors, hardware snapshots and
input/executable hashes are in the
[recorded results](results/optimization-b200/summary.json)
and its adjacent per-run reports. The measured executable SHA-256 is
`59129818212f33f8e485e932da1d4d17d32b8566a6d2b8677eef908fddbcde79`.

## What caused the gap

1. **Synchronization invalidated unrelated cache state.** The original generic
   cluster handoffs generated excessive cache invalidations. Typed full/empty
   pipeline policies now distinguish TMEM transfers, ordinary shared-memory
   publication, and reuse by asynchronous copies. They retain the required
   cluster scope, proxy ordering, completion waits and TC fences. Saved main
   profiles show `CCTL.IVALL` executions falling from 1,046,742 to 3,552. This
   was the largest early improvement, taking large decode from about 83 to
   60 µs in development measurements.
2. **Descriptor and ring bookkeeping was repeated.** Typed shared-tile arrays
   retain a common allocation base separately from tile offsets. Producer and
   consumer cursors retain a bounded slot and phase instead of repeatedly
   recovering them from a growing ticket. The compiler can reuse descriptor
   bases while preserving layout, operand role and stage constraints.
3. **Output copies were narrower than the reference.** Rust emitted ordinary
   128-bit stores; CuTeDSL emitted 256-bit stores with the L1 no-allocate hint.
   The final typed copy path emits the same `STG.E.NA.ENL2.256` instruction.
   Executed wide output stores drop from 65,536 to 32,768 in the large main
   profile. This change closed the remaining roughly 55-to-49 µs gap.

Smaller changes retain reference FTZ exp2 and packed arithmetic, reduce register
pressure in correction/epilogue fragments, specialize final-tile softmax, and
avoid work on fully valid mask tiles. The split-reduction workspace layout and
increasing-split accumulation order remain unchanged. Its host-precomputed
nonempty split count avoids repeating uniform geometry work in each block.

Rejected experiments included narrower V TMA copies, descriptor address-field
rewrites and wider epilogue fragments. Fewer instructions or a closer textual
translation alone did not consistently improve measured performance.

## Rust abstractions and safety

The example still expresses MMA using layout- and operand-typed CuTe tiles and
TMA using typed tensor maps. It contains no hand-written PTX kernel. New reusable
abstractions live in CuTe-RS:

- [SharedTileArray](../../../cute-rs/src/attention/sm100/shared.rs) carries layout, operand
  role and extent while exposing complete, swizzle-preserving tile views.
- [`TwoCtaRing` and `RingCursor`](../../../cute-rs/src/attention/sm100/pipeline.rs) encode
  stage count, cursor role and permitted synchronization/completion policies.
- [`GlobalStoreTile` and `store_global_no_allocate`](../../../cute-rs/src/copy/vector.rs) encode
  source alignment and supported scalar/width choices. The semantic CuTe
  operation lowers through the native cache-aware register-to-global copy atom.

Unsafe boundaries retain explicit obligations for live storage, destination
alignment, warp participation, asynchronous completion and buffer ownership.
Types restrict legal combinations; they do not claim to prove a dynamic
multi-warp schedule. The raw tile-address accessor uses wrapping arithmetic,
so merely retrieving an address does not require a still-live allocation.

## Validation and reproduction

The optimized executable passes all nine decode cases, including tails,
permuted pages, ring wrapping, persistent reuse, 256 splits, empty trailing
splits and output scale 1.73. Output and base-2 LSE are bitwise identical to
CuTeDSL in every case and pass the independent FP64 oracle. Both implementations
are checked again after timing to detect iteration-dependent changes.

Compute Sanitizer memcheck and synccheck report zero errors on B2/K2049/splits3
with one cluster and B1/K129/splits256 with two clusters. Racecheck reports zero
errors and warnings on the persistent tail case. Logs and input hashes are
stored with the results. These checks supplement the synchronization contracts.

At measurement time, the affected API/dialect/export suite passed 241 tests, including pointer
provenance, alignment and width rejection. Native compiler probes and real Rust
f16/f32/u32 fixtures confirm the wide-store instruction. Host tests (5), Python
orchestration/contract tests (12), tile/ring tests and scoped clippy checks pass.
Three prefill regression cases pass after rebuilding with the shared changes:
causal GQA tails, KV-ring wrap and persistent reuse. The rebuilt prefill binary
matches the validated regression binary byte-for-byte.

From `cuda-oxide/`, build with the pinned toolchain:

```sh
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 \
  cargo oxide build sm100_mla_decode_fp16_cute --arch sm_100a
```

From the example directory, using its Python dependencies:

```sh
python validate.py --rust-bin target/release/sm100_mla_decode_fp16_cute \
  --json validation.json
taskset -c 8 python compare.py --rust-bin target/release/sm100_mla_decode_fp16_cute \
  --batch 16 --kv-length 8192 --splits 8 --clusters 74 \
  --warmup 10 --iterations 100 --rounds 5 --timing-order cutedsl-first \
  --json large-cutedsl-first.json
# Repeat with --timing-order rust-first. For sustained runs use:
# --warmup 2000 --iterations 301 --rounds 7
```

For the small profile use B1/K1024/splits1/clusters1. These results establish
parity for the measured matched profiles, not every upstream configuration.
The separate [feature-gap roadmap](attention-feature-gaps.md) remains
the source of truth for omitted attention functionality.

The subsequent [documentation and compiler cleanup checks](results/cleanup-b200/summary.json)
rebuilt both examples with byte-identical executables and GPU images. The
recorded correctness and performance evidence therefore still applies.
