# Residual copy and scheduling findings

These results describe the `profiled-final` snapshot and the separate
`bounded-copy` probe captured on 2026-10-12. The records preserve those builds;
subsequent experiments are outside this comparison.

[Residual profile evidence](residual-profile-analysis.json) compares the
narrowed matrix-address and causal-only maximum-guard implementation with the
unchanged CUTLASS 4.7 reference. [Bounded-copy evidence](bounded-copy-profile-analysis.json)
examines a separate copy specialization. Neither analysis decomposes the full
remaining timing gap.

## Copy instruction difference

The retained implementation emits `LDGSTS.E.BYPASS.128.ZFILL`; the bounded-copy
probe emits `LDGSTS.E.BYPASS.128`; the reference emits
`LDGSTS.E.BYPASS.LTC128B.128`.

| Static mode | Copy warp instructions, each implementation | Reported copy shared wavefronts: retained | Bounded copy / reference |
| --- | ---: | ---: | ---: |
| Causal | 36,864 | 294,912 | 147,456 |
| Noncausal | 67,584 | 540,672 | 270,336 |

Global theoretical sector counts also match: 589,824 for causal copies and
1,081,344 for noncausal copies. Each implementation's reported shared
wavefronts equal its reported ideal, with zero excessive shared wavefronts.
The final noncausal comparison attributes the entire shared-wavefront
difference to copies; other shared-memory opcode counts match. These counters
provide no evidence of shared bank-conflict overhead. The difference in
reported wavefronts does **not** establish a proportional latency difference.

## Copy specialization changes the matrix schedule

The bounded-copy probe also changes register allocation and restores immediate
LDSM-to-HMMA dependencies:

| Static mode | Steady QK short-scoreboard samples: retained → bounded | Steady PV | Registers per thread | Dynamic instruction change |
| --- | ---: | ---: | ---: | ---: |
| Causal | 0 → 940 | 0 → 459 | 152 → 138 | −3.16% |
| Noncausal | 0 → 1,881 | 0 → 856 | 152 → 140 | −1.06% |

For example, bounded causal code loads `R80` at `0x23a0` and consumes it in
HMMA at `0x23b0`; the PV sequence loads `R40` at `0x2fa0` and consumes it at
`0x2fb0`. Noncausal equivalents are `0x1cc0 → 0x1cd0` and
`0x28c0 → 0x28d0`. Offsets are relative to each kernel's first instruction.

The instruction dependencies corroborate the sampling change. They support a
scheduling regression that can offset a copy-form benefit, even though total
instruction and register counts fall. Neither effect's exact contribution to
elapsed time is isolated. **Zero observed samples does not guarantee zero
actual stalls.**

## Other residual evidence

The retained noncausal kernel executes 2,948,096 dynamic warp instructions in
static mode versus 2,953,728 for the reference, a 0.19% reduction. Runtime mode
executes 3,317,760 versus 3,204,608, a 3.53% excess, concentrated in address and
control operations. Static mode's remaining timing gap therefore cannot be
explained by total instruction-count excess alone.

Rust has higher MIO and barrier sample shares, while counts of barriers,
waits and async commits match. The epilogue also retains more coordinate and
address instructions, although both implementations execute the same shared
stores, shared vector loads and global vector stores. Its sampled share is
small. Copy/load issue grouping, barrier arrival timing, runtime indexing and
epilogue address reuse remain specific candidates, with unproven timing gains.
Any full-atom copy specialization must preserve partial-atom zero filling
unless complete source validity is proved.

## Method and provenance

The workload is batch 1, `Sq=Sk=1024`, eight heads and head dimension 64 on an
RTX 5090 (`sm_120a`), with 128 threads per CTA and 24,576 bytes of dynamic shared
memory. Rust and the reference use CUTLASS 4.7.0. Nsight Compute profiles one
launch after correctness checks and warmup, with cache and clock control set
to `none`. Dynamic instruction totals are checked against raw hardware
counters. Stall samples are normalized within each profile; sampling is
stochastic, and category differences are not additive time savings.

Separate three-round Nsight Systems measurements for the `profiled-final`
snapshot recorded runtime causal/noncausal gaps of 2.24% / 3.66%, and about
1.65% for both static modes. Nsight Compute duration was not used. The causal
instruction encodings match the earlier narrowed causal kernels; that
equivalence record is preserved in the evidence.

Both JSON records embed profiler commands and environment settings, profile
hashes, build manifests, binary/cubin hashes, opcode counts and per-PC
observations. Historical temporary paths remain provenance, not portable
artifact links. The residual record also identifies the final timing reports;
the bounded-copy record includes the representative load/consumer pairs.
