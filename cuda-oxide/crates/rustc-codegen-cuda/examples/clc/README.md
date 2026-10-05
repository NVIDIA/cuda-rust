# CLC — Cluster Launch Control Test

Tests Blackwell (SM 100+) Cluster Launch Control intrinsics for hardware-managed
persistent kernel work-stealing.

## What CLC Does

You launch a **normal grid** (one CTA per tile, just like a non-persistent kernel).
But CTAs that finish fast can **steal** not-yet-launched CTAs' work directly from
the hardware scheduler — no atomics, no persistent CTAs.

### Concrete Example

10 tiles, 3 SMs:

```text
Hardware pending queue:  [CTA0, CTA1, ..., CTA9]

Step 1: Hardware launches 3 CTAs onto 3 SMs
  SM0: CTA0    SM1: CTA1    SM2: CTA2
  Pending: [CTA3, CTA4, CTA5, CTA6, CTA7, CTA8, CTA9]

Step 2: Each CTA processes its own blockIdx tile

Step 3: CTA0 finishes first, calls try_cancel
  Hardware: "CTA3 was pending. Cancelled it. Its blockIdx=(3,0). It's yours."
  CTA0 processes tile (3,0)
  Pending: [CTA4, CTA5, CTA6, CTA7, CTA8, CTA9]

Step 4: CTA0 finishes, calls try_cancel again → steals CTA4, etc.

Step 5: All tiles done. Next try_cancel returns "nothing left."
  CTA0 exits.
```

The hardware scheduler manages the pending queue. No global atomics. No contention.

### The try_cancel + mbarrier Flow

`try_cancel` is **asynchronous** (like TMA) — the hardware writes a 16-byte response
to shared memory and signals an mbarrier when done:

```text
arrive_expect_tx(bar, 16)  →  "I arrive, AND expect 16 bytes of async data"
                                barrier state: pending=0, tx-count=16
clc_try_cancel(resp, bar)  →  "hardware, write response to resp, signal bar when done"
                                (returns immediately)
wait_parity(bar, phase)    →  spin until hardware writes 16 bytes → tx-count=0
                                barrier completes, auto-reinits for next phase
```

You must call `arrive_expect_tx` **before** `try_cancel` so the barrier knows how
many bytes to expect. Same pattern as TMA loads.

### try_cancel vs try_cancel_multicast

Both variants steal a **cluster-sized group** of pending CTAs from the hardware
queue. The difference is where the 16-byte response lands:

```text
clc_try_cancel           →  response written to CALLING CTA's SMEM only
                             barrier signaled on CALLING CTA's mbarrier only
                             ⇒ each CTA steals independently

clc_try_cancel_multicast →  response written to ALL CTAs' SMEM in the cluster
                             barrier signaled on ALL CTAs' mbarriers
                             ⇒ one CTA steals, entire cluster gets the result
```

This determines the work distribution pattern:

**Unicast (`clc_try_cancel`)** — each CTA acts alone:
```text
Cluster of 4 CTAs on one SM, grid has 1024 pending tiles:

  CTA0: steal cluster [CTA4-CTA7]  → process all 4 tiles serially
  CTA1: steal cluster [CTA8-CTA11] → process all 4 tiles serially
  CTA2: steal cluster [CTA12-CTA15] → process all 4 tiles serially
  CTA3: steal cluster [CTA16-CTA19] → process all 4 tiles serially

Each CTA calls clc_try_cancel independently. Each gets a different
stolen cluster. Each serially processes all CLUSTER_SIZE tiles from
its stolen cluster. No coordination between CTAs needed.
```

**Multicast (`clc_try_cancel_multicast`)** — one steals, all share:
```text
Cluster of 4 CTAs on one SM, grid has 1024 pending tiles:

  CTA0 (rank 0): steal cluster [CTA4-CTA7]
    → response multicast to CTA0, CTA1, CTA2, CTA3
  CTA0: process tile 4 (first_stolen + 0)
  CTA1: process tile 5 (first_stolen + 1)
  CTA2: process tile 6 (first_stolen + 2)
  CTA3: process tile 7 (first_stolen + 3)

Only rank 0 calls clc_try_cancel_multicast. The hardware
broadcasts the response to every CTA in the cluster. Each CTA
reads first_ctaid from its own SMEM and derives its tile.
```

**Critical:** with the multicast variant, **every CTA** must call
`arrive_expect_tx` on its own `CLC_BAR` before rank 0 calls
`clc_try_cancel_multicast`. Otherwise the multicast response arrives
at an un-armed barrier (same pitfall as TMA multicast loads).

### Progression: Phase 3 → Phase 4A → CLC

```text
Phase 3:  grid = (tiles_m, tiles_n)     hardware assigns tiles, done
Phase 4A: grid = (148 CTAs, 1)          software atomic counter for tile IDs
CLC:      grid = (tiles_m, tiles_n)     normal grid + hardware work-stealing
```

CLC gets the simplicity of Phase 3's grid launch with the load-balancing of Phase 4A —
done in hardware with zero software overhead.

## Intrinsics Tested

| Intrinsic                       | PTX Instruction                                 |
|---------------------------------|-------------------------------------------------|
| `clc_try_cancel`                | `clusterlaunchcontrol.try_cancel.async...b128`  |
| `clc_try_cancel_multicast`      | `...multicast::cluster::all.b128`               |
| `clc_query_is_canceled`         | `...query_cancel.is_canceled.pred.b128`         |
| `clc_query_get_first_ctaid_x`   | `...query_cancel.get_first_ctaid::x.b32.b128`   |
| `clc_query_get_first_ctaid_y`   | `...query_cancel.get_first_ctaid::y.b32.b128`   |
| `clc_query_get_first_ctaid_z`   | `...query_cancel.get_first_ctaid::z.b32.b128`   |

### is_canceled semantics

- `is_canceled = 1` → a pending CTA was successfully canceled → **work available** (decode coords)
- `is_canceled = 0` → no pending CTAs to cancel → **done** (exit the loop)

Once you observe `is_canceled = 0`, you **must not** call `try_cancel` again (UB per PTX spec).

## Build and Run

```bash
cargo oxide run clc
```

## Hardware Requirements

- **GPU:** Blackwell (B200, GB200) with SM 100+
- **PTX ISA:** 8.6+
- **Driver:** CUDA 13.x (R580+)
