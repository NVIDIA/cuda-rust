/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide.Mir.ControlFlow

/-!
# Declarative Control-Flow & Lifecycle Contracts (`CudaOxide.Contracts.ControlFlow`)

Defines the path-level and operational reachability propositions (`Prop`) proved
in Lean 4 over a kernel's transcribed `ControlFlowCfg`, mapping 1-to-1 to the
control-flow clauses written in `#[verify(...)]`:

```text
  Rust `#[verify(...)]` Clause     Lean Proposition (`CudaOxide.Contracts.ControlFlow`)
  ────────────────────────────     ────────────────────────────────────────────────────
  `uniform_barriers`               `UniformBarriers cfg`
  `tmem_lifecycle`                 `TmemLifecycle cfg`
```

Every proposition quantifies over inductive CFG paths (`PathAvoiding`) and
inductively reachable CTA states (`CtaReachable`), never over a bare `Bool`
checker. Reusable decision checkers and soundness proofs live in
`CudaOxide.Proofs.ControlFlow`.
-/
namespace CudaOxide.Contracts.ControlFlow
open CudaOxide.Mir.ControlFlow

/-- Valid non-exclusive Blackwell `tcgen05.alloc` column counts (PTX ISA Table 58:
    power of two in `[32, 512]`). -/
def TmemValidCols (nCols : Nat) : Prop :=
  nCols = 32 ∨                                              -- 32 columns (minimum allocation unit), OR
  nCols = 64 ∨                                              -- 64 columns, OR
  nCols = 128 ∨                                             -- 128 columns, OR
  nCols = 256 ∨                                             -- 256 columns, OR
  nCols = 512                                               -- 512 columns (full per-SM TMEM capacity).

/-- Inductive directed path from block `u` to block `v` in `cfg` where every
    visited block is in bounds (`< cfg.blocks.length`) and outside `avoid`. -/
inductive PathAvoiding (cfg : ControlFlowCfg) (avoid : List Nat) : Nat → Nat → Prop where
  /-- Zero-step path at `u` when `u` is a valid block outside `avoid`. -/
  | refl {u : Nat}
      (hbounds : u < cfg.blocks.length)
      (havoid  : u ∉ avoid) :
      PathAvoiding cfg avoid u u
  /-- Single CFG edge `u -> w` followed by a valid `avoid`-free path `w ->* v`. -/
  | step {u w v : Nat} {cond : BranchCond}
      (hbounds : u < cfg.blocks.length)
      (havoid  : u ∉ avoid)
      (hedge   : (cond, w) ∈ (cfg.blockAt u).succs)
      (hrest   : PathAvoiding cfg avoid w v) :
      PathAvoiding cfg avoid u v

/-- Inductive small-step CTA reachability over `(blockIdx, CtaState)` starting
    at entry block `0` in `CtaState.initial cfg`. -/
inductive CtaReachable (cfg : ControlFlowCfg) : Nat → CtaState → Prop where
  /-- Kernel entry reaches block `0` in the initial clean/unallocated CTA state. -/
  | entry (hpos : 0 < cfg.blocks.length) :
      CtaReachable cfg 0 (CtaState.initial cfg)
  /-- Stepping through block `b`'s ops and taking enabled CTA edge `(cond, dst)` reaches `(dst, nextSt)`. -/
  | step {b dst : Nat} {st outSt nextSt : CtaState} {cond : BranchCond}
      (hreach : CtaReachable cfg b st)
      (hblk   : b < cfg.blocks.length)
      (hops   : execBlockOps cfg b st (cfg.blockAt b).ops = some outSt)
      (hedge  : (cond, dst) ∈ enabledCtaSuccs cfg b outSt)
      (hdst   : dst < cfg.blocks.length)
      (htrans : stepEdge cfg b outSt cond = some nextSt) :
      CtaReachable cfg dst nextSt

/-- Every reachable CTA state `(b, st)` in `cfg` executes all operations in block `b`
    without violating any hardware precondition, satisfies clean return invariants
    on `mir.return`, and transitions cleanly across every outgoing enabled CTA edge. -/
def SafeCtaExecution (requireTmem : Bool) (cfg : ControlFlowCfg) : Prop :=
  0 < cfg.blocks.length ∧                                   -- The kernel CFG has an entry block `0`, AND
  ∀ b : Nat, ∀ st : CtaState,                               -- For ANY block `b` and CTA state `st`:
    CtaReachable cfg b st →                                 -- IF `(b, st)` is reachable along valid CTA edges,
    ∃ outSt : CtaState,                                     -- THEN (1) all ops in `b` succeed (`some outSt`),
      execBlockOps cfg b st (cfg.blockAt b).ops = some outSt ∧
      ((cfg.blockAt b).isReturn = true →                    --      (2) if `b` is a `return` block, `TMEM` is freed
        validReturnState requireTmem outSt = true) ∧        --          and no async MMA or `ld` remains pending,
      (∀ cond : BranchCond, ∀ dst : Nat,                    --      (3) for every outgoing enabled CTA edge `(cond, dst)`,
        (cond, dst) ∈ enabledCtaSuccs cfg b outSt →         --          `dst` is a valid block index AND the edge
        dst < cfg.blocks.length ∧                           --          precondition (e.g. `mbarReady` armed) holds.
        ∃ nextSt : CtaState, stepEdge cfg b outSt cond = some nextSt)

/-- Every cyclic asynchronous TMA or `tcgen05` loop in `cfg` enforces reverse
    back-pressure (`EMPTY_BAR` wait or `thread::sync_threads()`) between consumer
    reads and the next iteration's producer overwrite. -/
def NoUnbackpressuredLoop (cfg : ControlFlowCfg) : Prop :=
  ∀ prod cons : Nat, ∀ bp : List Nat,                       -- For ANY cyclic `(producer, consumer, bp)` triple:
    (prod, cons, bp) ∈ loopBackpressurePairs cfg →          -- IF `cons` reads a buffer/TMEM written by cyclic `prod`,
    ¬ PathAvoiding cfg bp cons prod                         -- THEN NO CFG path `cons ->* prod` bypasses `bp`.

/-- **`uniform_barriers`**: At least one barrier synchronizes the kernel, every
    `thread::sync_threads()` and signaled `mbarrier` wait executes in thread-uniform
    control flow across all paths from entry `0` to `return`, and (when `Barrier`s
    are present without `TMEM`) `mbarrier` init/sync, PTX proxy fences, and loop
    back-pressure hold across all reachable CTA states. -/
def UniformBarriers (cfg : ControlFlowCfg) : Prop :=
  0 < countEffectiveBarriers cfg ∧                          -- At least one barrier synchronizes the phases, AND
  (∀ b : Nat, b ∈ syncBarrierBlocks cfg →                   -- For EVERY block `b` containing `sync_threads()`:
    activeGuardAt cfg b = .tr ∧                             --   (1) `b` sits outside thread-divergent `if` guards,
    (isCyclicBlockBool cfg b = false →                      --   (2) if `b` is acyclic, no path `0 ->* ret`
      ∀ ret : Nat, ret ∈ returnBlocks cfg →                 --       bypasses `b`, AND
        ¬ PathAvoiding cfg [b] 0 ret)) ∧
  ((cyclicBarrierBlocks cfg ≠ [] →                          --   (3) if cyclic barriers exist, an enclosing loop
    cyclicBarrierLoopBodyStarts cfg ≠ []) ∧                 --       header dominates them, AND
   ∀ hdr body : Nat,                                        --       from every loop body start `(hdr, body)`:
    (hdr, body) ∈ cyclicBarrierLoopBodyStarts cfg →
    (∀ ret : Nat, ret ∈ returnBlocks cfg →                  --       • entry `0` cannot bypass `hdr` to `return`,
      ¬ PathAvoiding cfg [hdr] 0 ret) ∧
    (∀ cb : Nat, cb ∈ cyclicBarrierBlocks cfg →             --       • no loop iteration `body ->* hdr` skips any
      ¬ PathAvoiding cfg [cb] body hdr) ∧                   --         cyclic barrier `cb`, AND
    (∀ ret : Nat, ret ∈ returnBlocks cfg →                  --       • no mid-loop exit `body ->* ret` bypasses
      ¬ PathAvoiding cfg [hdr] body ret)) ∧                 --         the loop header `hdr`, AND
  (∀ rBlk : Nat, rBlk ∈ signaledMbarReadyBlocks cfg →       --   (4) every signaled `mbarrier` ready-block intercepts
    ∀ ret : Nat, ret ∈ returnBlocks cfg →                   --       all paths from entry `0` to `return`, AND
      ¬ PathAvoiding cfg [rBlk] 0 ret) ∧
  (∀ dBlk : Nat, dBlk ∈ unsyncedDsmemBlocks cfg →           --   (5) every cross-CTA DSMEM access executes `cluster_sync()`
    ∀ ret : Nat, ret ∈ returnBlocks cfg →                   --       on all paths before any CTA exits (`mir.return`), AND
      ¬ PathAvoiding cfg ((clusterSyncBlocks cfg).filter (· != dBlk)) dBlk ret) ∧
  (∀ s rBlk : Nat, (s, rBlk) ∈ sharedReadSites cfg →        --   (6) every upward-exposed `SharedArray` read is dominated
    ¬ PathAvoiding cfg (sharedInitDominators cfg s) 0 rBlk) ∧ --     from entry `0` by a guaranteed initializer/barrier, AND
  validMbarArrivalCounts cfg = true ∧                       --   (7) 1-CTA `mbarrier` arrival sites match `mbarrier_init` counts, AND
  (0 < cfg.numBarriers ∧ tmemAllocSites cfg = [] →          --   (8) when `Barrier`s are used (outside `tmem_lifecycle`),
    SafeCtaExecution false cfg ∧ NoUnbackpressuredLoop cfg) --       `mbarrier` init/proxy/back-pressure hold on all paths.

/-- **`tmem_lifecycle`**: Blackwell Tensor Memory (`TMEM`) is allocated once by a
    single full 32-lane warp with a valid power-of-two column count in `[32, 512]`,
    ordered across `tcgen05` (`mma -> commit -> wait -> ld -> load_wait -> use`),
    proxy-coherent across `generic -> async`, and freed with the exact same column
    count after `sync_threads()` on every kernel exit path. -/
def TmemLifecycle (cfg : ControlFlowCfg) : Prop :=
  (∃ cols : Nat,                                            -- There EXISTS an allocation column count `cols` with:
    TmemValidCols cols ∧                                    --   (1) `cols ∈ {32, 64, 128, 256, 512}`,
    tmemAllocSites cfg = [cols] ∧                           --   (2) a single `tcgen05_alloc` of `cols` columns,
    tmemDeallocSites cfg ≠ [] ∧                             --   (3) at least one `tcgen05_dealloc` site, AND
    (∀ d : Nat, d ∈ tmemDeallocSites cfg → d = cols)) ∧     --   (4) every `tcgen05_dealloc` frees `cols`, AND
  (∀ aBlk : Nat, aBlk ∈ unlaunchedCg2AllocBlocks cfg →      --   (5) in 2-CTA `_cg2` mode, peer CTA launch is synchronized
    ¬ PathAvoiding cfg ((clusterSyncBlocks cfg).filter (· != aBlk)) 0 aBlk) ∧ -- via `cluster_sync()` from entry `0` before `alloc_cg2`, AND
  (∀ iBlk aBlk : Nat,                                       --   (6) in 2-CTA `_cg2` mode, any prior SMEM/`mbarrier_init`
    (iBlk, aBlk) ∈ cg2PreAllocInitPairs cfg →               --       synchronizes via `cluster_sync()` before `alloc_cg2`,
    ¬ PathAvoiding cfg ((clusterSyncBlocks cfg).filter (· != iBlk)) iBlk aBlk) ∧
  (∀ dBlk : Nat, dBlk ∈ unsyncedCg2DeallocBlocks cfg →      --   (6) in 2-CTA `_cg2` mode, every `dealloc_cg2` executes
    ∀ ret : Nat, ret ∈ returnBlocks cfg →                   --       `cluster_sync()` on all paths before `mir.return`,
      ¬ PathAvoiding cfg ((clusterSyncBlocks cfg).filter (· != dBlk)) dBlk ret) ∧
  (∀ s rBlk : Nat, (s, rBlk) ∈ sharedReadSites cfg →        --   (7) every upward-exposed `SharedArray` read is initialized,
    ¬ PathAvoiding cfg (sharedInitDominators cfg s) 0 rBlk) ∧
  validMbarArrivalCounts cfg = true ∧                       --   (8) 1-CTA `mbarrier` arrival sites match `mbarrier_init` counts,
  SafeCtaExecution true cfg ∧                               -- Full `tcgen05` / `TMEM` / proxy / `mbarrier` state machine holds,
  NoUnbackpressuredLoop cfg                                 -- AND cyclic `TMEM` / TMA loops enforce back-pressure.

end CudaOxide.Contracts.ControlFlow
