# mcast_barrier_test

Minimal test that isolates the cluster-wide barrier protocol used in
`gemm_sol_clc_multicast` for TMA multicast synchronization.

## What it tests

4 CTAs form a cluster. Every iteration, each CTA's thread 0 sends an
`mbarrier_arrive_cluster` to **rank 0's** barrier. Only rank 0 waits on
the barrier via `mbarrier_try_wait_parity`. The loop runs N iterations
with 2-stage double-buffering (BAR0 for even k, BAR1 for odd k).

This is the exact synchronization pattern `gemm_sol_clc_multicast` uses to
coordinate TMA multicast: all CTAs signal "I'm done reading B from
this buffer" and rank 0 waits for all 4 signals before overwriting
the buffer with the next multicast load.

## Build and run

```sh
cargo oxide run mcast_barrier_test
```

## The bug: why it deadlocks without `cluster_sync()`

### How the barrier works

An mbarrier is initialized with an expected arrival count (4 in our
case, one per CTA). It has a **parity bit** that starts at 0.

```text
Phase 0: counter = 4
  CTA 0 arrives → counter = 3
  CTA 1 arrives → counter = 2
  CTA 2 arrives → counter = 1
  CTA 3 arrives → counter = 0 → parity flips (0 → 1), counter resets to 4

Phase 1: counter = 4
  ... same thing, parity flips back (1 → 0)
```

`try_wait_parity(bar, P)` returns true when the barrier's current
parity is NOT P, meaning phase P has completed.

### Double-buffered parity scheme

With 2 barriers and N iterations:

```text
k=0: BAR0, wait phase 0     k=4: BAR0, wait phase 0  (cycle repeats)
k=1: BAR1, wait phase 0     k=5: BAR1, wait phase 0
k=2: BAR0, wait phase 1     k=6: BAR0, wait phase 1
k=3: BAR1, wait phase 1     k=7: BAR1, wait phase 1
```

Each barrier cycles through phase 0 → phase 1 → phase 0 → ...

### Why 4 iterations works fine

With 4 iterations, each barrier sees exactly 2 phases (phase 0 and
phase 1), totaling 8 arrivals per barrier. Even if CTAs are on
different iterations, every arrival still lands in the correct phase
because there are only 2 phases total.

### Why 64 iterations deadlocks

Non-rank-0 CTAs don't wait on anything. They arrive and immediately
loop to the next iteration. CTA 1 can reach k=8 while rank 0 is
still on k=0.

Here's what happens to BAR0 (expected count = 4 arrivals per phase):

```text
BAR0 starts at phase 0, counter = 4.

CTA 1 arrives for k=0   → counter = 3   (phase 0)
CTA 2 arrives for k=0   → counter = 2   (phase 0)
CTA 3 arrives for k=0   → counter = 1   (phase 0)
CTA 1 races to k=2, arrives again
                         → counter = 0   (phase 0) ← STOLEN from phase 1!

Phase 0 completes. BAR0 flips to phase 1, counter = 4.
Rank 0 waits phase 0 → sees phase 1 → passes. Looks fine!

CTA 0 arrives for k=0   → counter = 3   (phase 1) ← was meant for phase 0!
CTA 2 arrives for k=2   → counter = 2   (phase 1)
CTA 3 arrives for k=2   → counter = 1   (phase 1)
CTA 1 races to k=4, arrives
                         → counter = 0   (phase 1) ← STOLEN from phase 0!

Phase 1 completes. BAR0 flips to phase 0, counter = 4.
Rank 0 waits phase 1 → sees phase 0 → passes. Looks fine again!
```

But the counts are now permanently corrupted. CTA 1's arrivals
leaked into future phases, displacing other CTAs' arrivals.
Eventually some phase only receives 3 arrivals instead of 4.
That phase never completes. Rank 0 spins forever. **Deadlock.**

### The fix: `cluster_sync()` as a barrier between iterations

Adding `cluster_sync()` at the end of each iteration forces all 4
CTAs to be on the same iteration before any of them can advance.
CTA 1 cannot race to k=2 while CTA 0 is still on k=0.

With this fix, all tests pass (4, 8, 16, 32, 64, 256, 1024 iters).

### What this means for `gemm_sol_clc_multicast`

In the real GEMM kernel, `cluster_sync()` every iteration would be
too expensive. Instead, the pipeline itself prevents racing:

```text
TMA warp:  wait MMA_BAR[stage]  →  arrive MCAST_BAR  →  TMA loads  →  set TMA_BAR
MMA warp:  wait TMA_BAR[stage]  →  do MMA             →  commit MMA_BAR[stage]
```

A CTA's TMA warp can't re-arrive at MCAST_BAR[stage] until its MMA
warp has consumed the data from that stage — which requires rank 0's
multicast to have completed first. This limits how far any CTA can
get ahead and keeps arrivals in the correct barrier phase.
