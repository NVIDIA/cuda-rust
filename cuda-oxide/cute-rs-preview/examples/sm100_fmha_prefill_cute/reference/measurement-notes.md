# SM100 prefill validation and performance notes

Schedule details, numerical checks, timing methodology, and evidence for the
[prefill example](../README.md). [Reference setup](README.md) describes the
pinned CuTeDSL source and Python environment. Commands below run from the
example directory unless stated otherwise.

For current CUDA-event results alongside GPU kernel execution durations, see
the [2026-10-18 Nsight measurements](nsight-measurements.md).

The historical reports linked below predate the CTM migration and describe the
source revision recorded in each artifact. Current migration evidence follows
in the next section; the reproduction commands validate and measure new builds.
Structured CuTe operations keep their existing dialect paths, while explicit
instruction calls now use CTM as described in the [example README](../README.md).

## CTM migration validation

The [migration report](results/ctm-migration/summary.json) records **8 passing
B200 correctness cases** against pinned CuTeDSL and the independent oracle on
2026-10-14, plus source hashes and the compiler comparison with `d642dcbfbf`.
Resources are unchanged; typed TMEM teardown adds two non-padding instructions.
Both memcheck and synccheck report zero errors. This report is separate from
the historical performance results below.
A matched dense persistent run at `(B,Sq,Sk,Hq,Hkv)=(4,4096,4096,8,8)` used
2,000 warmups, 301 invocations per window and five windows. Baseline and migrated
medians were **290.322 µs** and **290.476 µs** (+0.053%), within the observed
spread. This one-profile invocation-throughput check found no measurable
regression; it does not establish broad performance parity. Both raw timing
reports are linked from the migration summary.

## Build and reproduce

From `cuda-oxide/`:

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 \
  cargo oxide build sm100_fmha_prefill_cute --arch sm_100a
```

From this example's directory, run the built executable. Its default entry
point invokes `compare.py`, which supplies identical FP16 bytes to both paths:

```bash
target/release/sm100_fmha_prefill_cute --shape 1,1024,1024,8,8 --check-only
target/release/sm100_fmha_prefill_cute --shape 1,257,385,4,2 --causal --check-only
target/release/sm100_fmha_prefill_cute --shape 4,4096,4096,8,8 \
  --warmup 10 --iters 100 --windows 5 --json /tmp/prefill-comparison.json
target/release/sm100_fmha_prefill_cute --shape 4,4096,4096,8,8 \
  --warmup 2000 --iters 301 --windows 5 --json /tmp/prefill-sustained.json
```

## Matched schedule

The kernel retains 512 threads: two four-warp softmax groups, four correction
warps, and separate MMA/load/epilogue/idle warps. Q uses 2 stages, aliased K/V 3,
and O 2. Twelve pipeline identities coordinate these groups. TMEM 512 columns
hold S0/S1 at 0/128, O0/O1 at 256/384, and packed P0/P1 at 160/32, overlapping the
opposite score buffer. Score loads and packed probability stores use the source's
32-register TMEM chunks. Correction uses eight deferred 16-register load/scale/store
steps and one terminal store wait. The runtime epilogue loop uses dependent
16-register loads and 128-bit shared stores. Packed FP32 FMA/add/multiply and the
softmax FMA/exp2/conversion interleave follow the source. Unmasked tiles interleave
32-register loads with two max-reduction chains; masked tiles use a balanced
reduction. The masked path retains one conservative TMEM load-completion wait
before masking; the source relies on register dependencies there. The
initial/terminal overlap handoffs are explicit.
Persistent scheduling visits query tiles first and strides by resident CTAs.
Typed shared layouts select B128 swizzling, both QK shared/shared and PV
TMEM/shared MMA, and matching rank-three tensor-map copies. The launch preserves
one minimum CTA per SM and role-local register budgets of 192/96/32. The compiled
cubin has zero stack and local-memory instructions; see
[resource inspection](results/kernel-resources.json).

## Numerical checks and timing

`compare.py` uses an independent chunked FP32 Torch oracle, explicitly disables
TF32, and validates both
outputs and their cross-comparison. Ordinary inputs retain upstream
`atol=0.1, rtol=1e-5`; deterministic uniform-score tests use `atol=0.005`.
Every output must be finite. Reports include actual maximum/RMS errors,
source provenance, hardware state, raw timing windows, and launch resources.
The Rust adapter also checks the supplied FP32 oracle before recording events.

Timing uses events on each implementation's actual launch stream around a
window of repeated complete invocations, divided by the iteration count.
Both implementations use compiler optimization level 2, identical
warmups/iterations/windows, and hot cache.
Compilation, allocation, transfers, and validation are outside event intervals.
Event windows include host submission gaps and measure complete invocation
throughput. Profiler kernel-active durations are reported separately.
No relative performance claim should be made from unvalidated runs or from
differences smaller than the observed spread.

## Validation workflow

Validation commands from `cuda-oxide/`, using the same Python runtime:

```bash
python -m unittest discover -s cute-rs-preview/examples/sm100_fmha_prefill_cute -p test_compare.py
python cute-rs-preview/examples/sm100_fmha_prefill_cute/validate.py --rust-bin /path/to/binary
python cute-rs-preview/examples/sm100_fmha_prefill_cute/validate.py --rust-bin /path/to/binary \
  --case both-tails-gqa --sanitizer memcheck
python cute-rs-preview/examples/sm100_fmha_prefill_cute/validate.py --rust-bin /path/to/binary \
  --case kv-ring-wrap --sanitizer synccheck
```

Validation covers one-key attention, causal masks, GQA, tails, all stage-ring
wraps, and more work tiles than resident SMs. Subprocess timeouts bound barrier
failures. All eight cases passed on B200 with both compilers at optimization
level 2; see [correctness reports](results/correctness/). The additional causal
GQA timing profile also passed. Across all runs, the largest Rust error against
the FP32 oracle was 0.00058413 and the largest difference from CuTeDSL was
0.00012207. The one-key case matched exactly. Final memcheck and synccheck each
reported zero errors; see [sanitizer logs](results/sanitizer/).
[CUDA event tracing](results/stream-trace.json) confirms all three measured test
launches fall between both events on the actual launch stream.

## Recorded measurements

B200 measurements on 2026-10-12 used 10 warmups, 100 invocations per window,
and 5 windows. Values are median microseconds, with the observed range:

| Profile (B, Sq, Sk, Hq, Hkv) | Rust µs | CuTeDSL µs |
| --- | ---: | ---: |
| Dense (1, 1024, 1024, 8, 8) | 18.45 (18.35–18.47) | 75.40 (75.21–76.09) |
| Dense persistent (4, 4096, 4096, 8, 8) | 246.17 (225.49–286.05) | 236.80 (226.66–303.35) |
| Causal GQA persistent (4, 4096, 4096, 8, 2) | 150.32 (144.85–153.33) | 168.25 (168.13–170.18) |

These are complete invocation event windows, including host submission gaps.
The small-case difference does not establish a kernel-active speedup. The dense
persistent ranges overlap substantially, so that short-warmup run does not
establish a speed ranking. A separate sustained dense measurement used 2,000
warmups, 301 invocations per window, and 5 windows: Rust **289.98 µs**
(289.33–290.87), CuTeDSL **292.13 µs** (291.06–292.26). The longer warmup removed
the broad drift; the median difference is 0.73%, consistent with comparable
invocation throughput. Both the initial and sustained reports are preserved.
[Raw timing reports](results/timing/) retain all samples and GPU state.
The kernel preserves the selected source schedule with the conservative masked
score-load wait noted above; it does not claim identical generated instructions.

These historical reports identify the measured executable SHA-256
`090480d8ce79bac74ca65048ec1d2488d047a78f5ec9c4964d640a028b5c7637`.
Its SM100a cubin SHA-256 is
`648600133c912456355e682e5c6821f8c54362d47c47176c1a32e5638b3a3f00`.
The [resource report](results/kernel-resources.json) records source provenance,
register budgets, shared memory, and instruction inspection.


The reports retain paths and hashes from their measurement runs. They were
moved from the example's `results/` directory to `reference/results/` without
changing their contents. These are historical provenance records, not paths
for new build commands. The shared [initial validation](initial-validation.md)
report records the corresponding compiler and joint attention checks.

The subsequent [documentation and compiler cleanup checks](../../sm100_mla_decode_fp16_cute/reference/results/cleanup-b200/summary.json)
preserved the prefill GPU image and the later correctness-regression executable
(`4f9ffbb5a7e3194a37425516cc28c12a5c0bfe0a3324019f5350e0e2e06ce6b2`).
That executable differs from the historical timing executable identified above;
the timing reports remain measurements of that earlier host build. The later
CLI comparison changes also preserve the GPU image, but change the host executable.
