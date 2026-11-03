# Reactor optimization A/B — combinatorial report

Combinatorial A/B of the deferred "measure-first" reactor optimizations,
run against the real `SlotTable` (loom/miri/TSan-verified core). Harness:
`cuda-async/src/slot_table.rs::ab_bench` (an `#[ignore]` test).

```
cargo test -p cuda-async --release --lib ab_reactor_variants -- --nocapture --ignored
```

## Method

Two orthogonal axes, tested as a full matrix:

- **Unpark policy** — `always` (then current: unpark the scanner on every
  registration) vs `empty→wake` (llist-style: unpark only on the idle→active
  transition). At measurement time the `n_armed` counter behind `empty→wake`
  lived in the harness so `SlotTable` stayed untouched; since the policy was
  adopted, `SlotTable` carries its own `n_armed` (`publish` reports the
  idle→active transition, `is_idle` gates the scanner's park), and the
  harness keeps a separate counter only so both policies can still be
  toggled against the shipped table.
- **Scanner idle** — `park` (current) vs `never-park` (hot spin).

Two workload regimes: **saturated** (8 threads registering + completing as
fast as possible) and **bursty** (a periodic yield opening idle windows).
8 registrant threads × 150k ops; RTX-5090 host (16-core). Metric: registration
throughput (Mops/s), plus unpark count and scanner scan-passes. Best of 3.

## Results

### SATURATED
| config | Mops/s | unparks | scan passes |
|---|---|---|---|
| always+park | 5.43 | 1,200,000 | 79,831 |
| **empty→wake+park** | **7.52** | 13,526 | 91,372 |
| always+never-park | 6.72 | 1,200,000 | 323,697 |
| empty→wake+never-park | 7.08 | 6,622 | 169,287 |

### BURSTY
| config | Mops/s | unparks | scan passes |
|---|---|---|---|
| always+park | 6.84 | 1,200,000 | 112,180 |
| **empty→wake+park** | **9.52** | 17,168 | 97,865 |
| always+never-park | 5.92 | 1,200,000 | 140,937 |
| empty→wake+never-park | 6.74 | 8,863 | 252,140 |

## Findings

1. **empty→wake is a real +38–39% throughput win — my earlier "marginal"
   analysis was wrong.** I reasoned that std `unpark()` on a not-parked thread
   is a cheap atomic, so skipping it saves little. That missed the actual cost:
   8 threads calling `unpark()` on the **same** scanner `Thread` hammer one
   `Parker` cache line, and the cross-core coherence ping-pong dominates.
   empty→wake cuts unparks ~99% (1.2M → ~13k) and removes the contention. The
   linux `llist` "wake-only-if-was-empty" trick applies for exactly this reason.

2. **The combinatorial interaction mattered, and it favors `park`, not
   `never-park`.** I had flagged that empty→wake might make never-park viable.
   The data says the opposite: `park` beats `never-park` in every cell once
   contention is removed, because the never-park scanner burns a core spinning
   (169k–323k scan passes vs ~90k) that then competes with the registrant
   threads for CPU. Best config in both regimes is **empty→wake + park**.

3. **Regime caveat.** This measures registration *throughput* at 5–9 M ops/s —
   far above real GPU completion rates (4-stream pools, µs–ms kernels →
   hundreds of k/s from a few threads). So the practical win is smaller than
   +38%, but the change is cheap and strictly helps under any concurrent
   registration load. It never hurts latency (fewer atomics on the register
   path).

## Recommendation — ADOPTED

**`empty→wake` + `park` shipped** (`e62632c`). `SlotTable` gained an `n_armed`
counter; `publish` reports the idle→active transition and `register` unparks
only then; the scanner parks on `is_idle()`. The ordering subtlety (increment
`n_armed` before the bit, so the scanner can't decrement before it) is
loom-verified: the wrong order trips an underflow `debug_assert` under loom,
the correct order passes; also clean under TSan and miri. N=1 latency
unchanged (single registration still unparks). `never-park` was **not**
adopted — it loses to `park` under CPU contention.

## Not tested this round (need their own implementation)

- **Per-stream monotonic counter + prefix-wake** (io_uring CQ-tail): a
  different harvest data structure, not a toggle. Its win (cache lines per
  stream vs per op) is orthogonal to unpark policy; worth a separate A/B once
  implemented.
- **Adaptive spin budget** (Shenango / `mutex_spin_on_owner`) and **inline-spin
  budget auto-tuning**: the DeviceFuture poll layer and scanner backoff, a
  different axis from this reactor-harvest matrix.
