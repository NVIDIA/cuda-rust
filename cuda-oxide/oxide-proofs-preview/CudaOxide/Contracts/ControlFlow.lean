/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide.Mir.ControlFlow
import CudaOxide.Mir.Program

/-!
# Declarative Control-Flow & Lifecycle Contracts (`CudaOxide.Contracts.ControlFlow`)

Defines the path-level and operational reachability propositions (`Prop`) proved
in Lean 4 over a kernel's transcribed `MirFunc` and `ControlFlowCfg`, mapping
1-to-1 to the control-flow clauses written in `#[verify(...)]`:

```text
  Rust `#[verify(...)]` Clause     Lean Proposition (`CudaOxide.Contracts.ControlFlow`)
  ────────────────────────────     ────────────────────────────────────────────────────
  `uniform_barriers`               `UniformBarriers mirFunc controlFlowCfg`
  `tmem_lifecycle`                 `TmemLifecycle controlFlowCfg`
```

Every proposition quantifies over inductive CFG paths (`PathAvoiding`) and
inductively reachable CTA states (`CtaReachable`), never over a bare `Bool`
checker. Reusable decision checkers and soundness proofs live in
`CudaOxide.Proofs.ControlFlow`.
-/
namespace CudaOxide.Contracts.ControlFlow
open CudaOxide.Mir.ControlFlow
open CudaOxide.Mir.Program

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

/-- Structural CFG obligations for `uniform_barriers` on a refined `ControlFlowCfg`:
    at least one barrier synchronizes the kernel, every `sync_threads()` and signaled
    `mbarrier` wait executes in thread-uniform control flow across all paths from entry `0`
    to `return`, and `mbarrier`/proxy/back-pressure invariants hold. -/
def UniformBarriersCfg (cfg : ControlFlowCfg) : Prop :=
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

/-- Build the 1-to-1 structural `ControlFlowCfg` skeleton directly from `f.blocks`. -/
def mirToCfg (f : MirFunc) : ControlFlowCfg :=
  { blockDimX := f.blockDimX, blockDimY := f.blockDimY, clusterDimX := f.clusterDimX
  , numShared := f.sharedDecls.length, numBarriers := 0
  , blocks := f.blocks.map fun blk =>
      match blk.term with
      | .ret => { isReturn := true }
      | .unreachable => {}
      | .goto dst _ => { succs := [(.always, dst)] }
      | .condBr cond tBlk _ fBlk _ =>
          { condVarOpt := some cond, succs := [(.unknown, tBlk), (.unknown, fBlk)] } }

/-- Branch arm `(u, armBlk, armIdx)` reaches incoming parameter edge `e = (dstBlk, k, argVar, srcBlk, edgeArm, val)`
    before hitting `avoid`: either `e` is the direct outgoing arm `armIdx` of `u`, or a path
    in `cfg` avoiding `avoid` connects `armBlk` to `srcBlk`. -/
def armReachesEdge (cfg : ControlFlowCfg) (u armBlk armIdx : Nat) (avoid : List Nat)
    (e : Nat × Nat × Nat × Nat × Nat × Nat) : Prop :=
  (e.2.2.2.1 = u ∧ e.2.2.2.2.1 = armIdx) ∨ PathAvoiding cfg avoid armBlk e.2.2.2.1

/-- Declarative `Split -> Assign -> Meet` CTA-uniformity invariant over `MirFunc`:
    `uniVars` and `rfPtrs` name the cross-block SSA variables guaranteed CTA-uniform
    (resp. race-free `SharedArray` pointers), while `checkExecUniStmts` steps block-local
    scratch registers sequentially to each block's terminator (`exitAt b`). -/
def MirValidUniSet (f : MirFunc) (uniVars rfPtrs uniConds : List Nat) : Prop :=
  let mcfg := mirToCfg f
  let allEdges := collectAllArgEdges f
  let exitAt := fun b => checkExecUniStmts f uniVars rfPtrs uniVars rfPtrs (f.blockAt b).stmts
  (∀ v : Nat, v ∈ (f.blockAt 0).args.map BlockArg.varId → v ∉ rfPtrs) ∧ -- (1) Entry params are not SMEM pointers,
  (∀ b : Nat, b < f.blocks.length → (exitAt b).1 = true) ∧  -- (2) Every assignment to `uniVars`/`rfPtrs` is valid,
  (∀ e : Nat × Nat × Nat × Nat × Nat × Nat, e ∈ allEdges →  -- (3) Incoming block-argument edges preserve uniformity:
    (e.2.2.1 ∈ uniVars → e.2.2.2.2.2 ∈ (exitAt e.2.2.2.1).2.1) ∧
    (e.2.2.1 ∈ rfPtrs → e.2.2.2.2.2 ∈ (exitAt e.2.2.2.1).2.2)) ∧
  (∀ mb : Nat, mb < f.blocks.length →                       -- (4) Every `cond ∈ uniConds` is exit-uniform at its `condBr`:
    ∀ cond tBlk fBlk : Nat, ∀ tArgs fArgs : List Nat,
      (f.blockAt mb).term = .condBr cond tBlk tArgs fBlk fArgs →
      cond ∈ uniConds → cond ∈ (exitAt mb).2.1) ∧
  (∀ u : Nat, u < f.blocks.length →                         -- (5) `Split -> Assign -> Meet`: for every divergent `condBr` at `u`,
    ∀ cond tBlk fBlk : Nat, ∀ tArgs fArgs : List Nat,
      (f.blockAt u).term = .condBr cond tBlk tArgs fBlk fArgs →
      cond ∉ (exitAt u).2.1 →
      ∃ mOpt : Option Nat,                                  --     there exists an optional single reconvergence block `mOpt` with:
        (∀ m : Nat, mOpt = some m →                         --     • if `mOpt = some m`, `m ≠ u` post-dominates `u` to all `return`s,
          m ≠ u ∧ ∀ ret : Nat, ret ∈ returnBlocks mcfg → ¬ PathAvoiding mcfg [m] u ret) ∧
        let avoid := match mOpt with | some m => [m] | none => []
        (∀ e0 e1 : Nat × Nat × Nat × Nat × Nat × Nat,       --     • (5a) Two-arm pre-reconvergence join on `(b, k)`:
          e0 ∈ allEdges → e1 ∈ allEdges →
          e0.1 = e1.1 → e0.2.1 = e1.2.1 →
          e0.2.2.1 ∈ uniVars →
          armReachesEdge mcfg u tBlk 0 avoid e0 →
          armReachesEdge mcfg u fBlk 1 avoid e1 →
          e0.2.2.2.2.2 = e1.2.2.2.2.2) ∧
        (∀ e : Nat × Nat × Nat × Nat × Nat × Nat,           --     • (5b) Divergent loop cycle through `e.dstBlk`:
          e ∈ allEdges →
          e.2.2.1 ∈ uniVars →
          PathAvoiding mcfg avoid e.1 u →
          (armReachesEdge mcfg u tBlk 0 avoid e ∨ armReachesEdge mcfg u fBlk 1 avoid e) →
          e.2.2.2.2.2 = e.2.2.1))

/-- Downgrade `.uniform` and `.posLoopExit` branch labels at block `cb` to `.unknown` whenever
    its recorded `condVarOpt` is not in the certified `uniConds` list. -/
def refineCfgWithUni (cfg : ControlFlowCfg) (uniConds : List Nat) : ControlFlowCfg :=
  { cfg with
    blocks := cfg.blocks.map fun cb =>
      let condUni :=
        match cb.condVarOpt with
        | some cond => uniConds.contains cond
        | none => false
      if condUni then cb
      else
        { cb with
          succs := cb.succs.map fun (bc, dst) =>
            match bc with
            | .uniform | .posLoopExit _ => (.unknown, dst)
            | _ => (bc, dst) } }

/-- **`uniform_barriers`**: There exists a CTA-uniformity set `(uniVars, rfPtrs, uniConds)`
    certified over `f` (`MirValidUniSet f uniVars rfPtrs uniConds`) under which the refined
    `ControlFlowCfg` satisfies all structural barrier-uniformity and lifecycle obligations. -/
def UniformBarriers (f : MirFunc) (cfg : ControlFlowCfg) : Prop :=
  ∃ uniVars rfPtrs uniConds : List Nat,                     -- Certified CTA-uniform SSA variables & condition IDs:
    MirValidUniSet f uniVars rfPtrs uniConds ∧              --   (1) Satisfies `Split -> Assign -> Meet` over `f`, AND
    UniformBarriersCfg (refineCfgWithUni cfg uniConds)      --   (2) Proves `UniformBarriersCfg` after downgrading divergent branches.

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
  (∀ iBlk aBlk : Nat,                                       --   (5) in 2-CTA `_cg2` mode, any prior SMEM/`mbarrier_init`
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
