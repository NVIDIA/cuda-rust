/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide.Mir.Program
import CudaOxide.Contracts.ControlFlow

/-!
# Operational Kernel Safety Contracts (`CudaOxide.Contracts.Program`)

States the declarative safety contracts (`no_oob_idx`, `disjoint_writes`,
`no_data_race`, and `mir.assert` redundancy) directly over the small-step
operational semantics (`Reachable` / `EmitsEvent`) of a line-by-line
`PostMem2Reg` `MirFunc` - with zero pattern-matched `KernelSummary` middleman.
-/
namespace CudaOxide.Contracts.Program
open CudaOxide.Mir.Program
open CudaOxide.Mir.ControlFlow
open CudaOxide.Contracts.ControlFlow

/-- **`no_oob_idx(slice_param)`**: On every valid launch input `env`, for every
    valid GPU thread `t`, every reachable global memory load or store targeting
    slice parameter `paramId` accesses an element index strictly `< env.sliceLen paramId`. -/
def MirNoOobGlobal (f : MirFunc) (paramId : Nat) : Prop :=
  ∀ env : InputEnv, ∀ t : ThreadCtx, ∀ ev : MemEvent,       -- Pick ANY valid launch `env`, thread `t`, and memory event `ev`:
    f.validInput env →                                      -- IF `env` satisfies scalar widths and host `requires` clauses,
    f.validThread t →                                       -- AND `t` is within the CTA's `(blockDimX, blockDimY)` bounds,
    EmitsEvent f env t ev →                                 -- AND `t` reaches and executes `ev` during `f`,
    ev.space = .global →                                    -- AND `ev` accesses global slice memory,
    ev.bufId = paramId →                                    -- AND `ev` targets parameter `paramId`,
    ev.index < env.sliceLen paramId                         -- THEN the element offset is strictly `< paramId.len()`.

/-- **`no_oob_idx(SHARED)`**: On every valid launch input `env`, for every valid
    GPU thread `t`, every reachable shared-memory load or store targeting
    `sharedId` accesses an element index strictly below its declared `size`. -/
def MirNoOobShared (f : MirFunc) (sharedId : Nat) : Prop :=
  ∀ env : InputEnv, ∀ t : ThreadCtx, ∀ ev : MemEvent,       -- Pick ANY valid launch `env`, thread `t`, and memory event `ev`:
    f.validInput env →                                      -- IF `env` satisfies launch preconditions,
    f.validThread t →                                       -- AND `t` is a valid CTA thread,
    EmitsEvent f env t ev →                                 -- AND `t` emits `ev` on a reachable execution path,
    ev.space = .shared →                                    -- AND `ev` accesses CTA shared memory,
    ev.bufId = sharedId →                                   -- AND `ev` targets `SharedArray` `sharedId`,
    ev.index < (f.sharedDecls.getD sharedId { size := 0 }).size -- THEN the element offset is strictly `< SHARED.size`.

/-- **`disjoint_writes(out)`**: On every valid launch input `env`, if two valid
    GPU threads `t₁, t₂` both perform a global access on `paramId` (at least one
    being a write) that resolves to the exact same element index, they must be
    the exact same GPU thread (`t₁ = t₂`). -/
def MirDisjointWrites (f : MirFunc) (paramId : Nat) : Prop :=
  ∀ env : InputEnv, ∀ t₁ t₂ : ThreadCtx, ∀ ev₁ ev₂ : MemEvent, -- Pick ANY launch `env`, two threads `t₁, t₂`, and two events:
    f.validInput env →                                      -- IF `env` satisfies launch preconditions,
    f.validThread t₁ → f.validThread t₂ →                   -- AND both `t₁` and `t₂` are valid GPU threads,
    EmitsEvent f env t₁ ev₁ → EmitsEvent f env t₂ ev₂ →     -- AND `t₁` emits `ev₁` while `t₂` emits `ev₂`,
    ev₁.space = .global → ev₂.space = .global →             -- AND both events access global memory,
    ev₁.bufId = paramId → ev₂.bufId = paramId →             -- AND both target mutable slice parameter `paramId`,
    (ev₁.isWrite = true ∨ ev₂.isWrite = true) →             -- AND at least one of the two accesses is a write,
    ev₁.index = ev₂.index →                                 -- AND both compute the exact same element index,
    t₁ = t₂                                                 -- THEN they are the exact same GPU thread.

/-- Hardware `mbarrier` completion ordering between a producer write released on `BAR[s]`
    and a subsequent consumer access after `mbarrier_try_wait_parity(&BAR[s])` succeeded. -/
def asyncOrdered (f : MirFunc) (env : InputEnv) (ev₁ ev₂ : MemEvent) : Prop :=
  (∃ b s : Nat, ev₁.syncBar = some (b, s) ∧ ev₂.syncBar = none ∧   -- Either `ev₁` released `BAR[s]`
                f.hasValidBarSignal b = true ∧ env.mbarFlag b s = true) ∨ -- and `BAR[s]` completed before `ev₂`, OR
  (∃ b s : Nat, ev₂.syncBar = some (b, s) ∧ ev₁.syncBar = none ∧   -- `ev₂` released `BAR[s]`
                f.hasValidBarSignal b = true ∧ env.mbarFlag b s = true)   -- and `BAR[s]` completed before `ev₁`.

/-- **`no_data_race(SHARED)`**: Barrier synchronization is uniform across the CTA
    (`UniformBarriers cfg`, so per-thread `sync_threads()` counters represent the same
    CTA epoch), AND within any single CTA (`t₁.bx = t₂.bx ∧ t₁.by_ = t₂.by_`), if two
    valid threads `t₁, t₂` access the same `SharedArray` `sharedId` in the same barrier
    phase (`ev₁.phase = ev₂.phase`), are not ordered by an `mbarrier` completion handoff
    (`¬ asyncOrdered f env ev₁ ev₂`), at least one is a write, and their element indices
    collide (`ev₁.index = ev₂.index`), then `t₁ = t₂`. -/
def MirNoSharedRace (f : MirFunc) (cfg : ControlFlowCfg) (sharedId : Nat) : Prop :=
  UniformBarriers f cfg ∧                                   -- CTA barriers are uniform so `ev.phase` epochs align, AND
  ∀ env : InputEnv, ∀ t₁ t₂ : ThreadCtx, ∀ ev₁ ev₂ : MemEvent, -- Pick ANY launch `env`, two CTA threads, and two events:
    f.validInput env →                                      -- IF `env` satisfies launch preconditions,
    f.validThread t₁ → f.validThread t₂ →                   -- AND both `t₁` and `t₂` are valid CTA threads,
    t₁.bx = t₂.bx → t₁.by_ = t₂.by_ →                       -- AND both belong to the same thread block (CTA),
    EmitsEvent f env t₁ ev₁ → EmitsEvent f env t₂ ev₂ →     -- AND `t₁` emits `ev₁` while `t₂` emits `ev₂`,
    ev₁.space = .shared → ev₂.space = .shared →             -- AND both access CTA shared memory,
    ev₁.bufId = sharedId → ev₂.bufId = sharedId →           -- AND both target `SharedArray` `sharedId`,
    ev₁.phase = ev₂.phase →                                 -- AND both execute in the same `sync_threads()` phase,
    ((t₁.ty * f.blockDimX + t₁.tx) / 32 =                   -- AND if both belong to the same 32-lane hardware warp,
      (t₂.ty * f.blockDimX + t₂.tx) / 32 →                  --     they execute in the same `warp::sync_mask(u32::MAX)` phase,
      ev₁.warpPhase = ev₂.warpPhase) →
    ¬ asyncOrdered f env ev₁ ev₂ →                          -- AND neither is ordered after the other by `mbarrier` wait,
    (ev₁.isWrite = true ∨ ev₂.isWrite = true) →             -- AND at least one access is a write,
    ev₁.index = ev₂.index →                                 -- AND both hit the exact same shared-memory slot,
    t₁ = t₂                                                 -- THEN they are the exact same thread (`tx₁ = tx₂ ∧ ty₁ = ty₂`).

/-- **`tmem_lifecycle` (`TMEM` bounds & warpgroup slice)**: Every reachable Blackwell
    Tensor Memory (`TMEM`) access on a valid thread stays within `128` lanes, within
    `allocCols ≤ 512` columns, and (for warp-restricted `tcgen05.ld` / `tcgen05.st`)
    within that warp's 32-lane warpgroup slice (PTX ISA 9.7.18.8.1). -/
def MirNoOobTmem (f : MirFunc) : Prop :=
  ∀ env : InputEnv, ∀ t : ThreadCtx, ∀ tev : TmemEvent,     -- Pick ANY valid launch `env`, thread `t`, and `TMEM` event:
    f.validInput env →                                      -- IF `env` satisfies launch preconditions,
    f.validThread t →                                       -- AND `t` is a valid CTA thread,
    EmitsTmemEvent f env t tev →                            -- AND `t` reaches and executes `tev`,
    (tev.relAddr / 65536) + tev.heightLanes ≤ 128 ∧         -- THEN (1) lane span fits inside the 128 TMEM lanes, AND
    (tev.relAddr % 65536) + tev.widthCols ≤ tev.allocCols ∧ --      (2) column span fits inside `allocCols`, AND
    tev.allocCols ≤ 512 ∧                                   --      (3) `allocCols` fits inside the 512-column SM limit, AND
    (tev.warpRestricted = true →                            --      (4) for warp-restricted `tcgen05.ld` / `tcgen05.st`,
      ((t.tx / 32) % 4) * 32 ≤ tev.relAddr / 65536 ∧        --          starting lane ≥ this warp's 32-lane slice base, AND
      (tev.relAddr / 65536) + tev.heightLanes ≤             --          ending lane ≤ this warp's 32-lane slice ceiling.
        ((t.tx / 32) % 4) * 32 + 32)

/-- **Verified `mir.assert` elision**: Every reachable execution reaching
    `Stmt.assertCond assertId cond` evaluates `cond` to `.bool true`, proving
    the runtime panic check can be safely erased before LLVM/PTX lowering. -/
def MirAssertRedundant (f : MirFunc) (assertId : Nat) : Prop :=
  ∀ env : InputEnv, ∀ t : ThreadCtx, ∀ b : Nat, ∀ phases : Nat × Nat, ∀ regs : RegFile, -- Pick ANY reachable block entry `(b, regs, phases)`:
    f.validInput env →                                      -- IF `env` satisfies launch preconditions,
    f.validThread t →                                       -- AND `t` is a valid GPU thread,
    Reachable f env t b regs phases →                       -- AND `(b, regs, phases)` is reachable,
    ∀ pfx sfx : List Stmt, ∀ cond : Nat,                    -- AND `Stmt.assertCond assertId cond` appears inside `b.stmts`,
      (f.blockAt b).stmts = pfx ++ (.assertCond assertId cond :: sfx) →
      (execStmts f env t regs phases (f.blockAt b).term pfx).1 cond = .bool true -- THEN `cond` is guaranteed `.bool true` at that program point.

/-- **Cluster DSMEM peer-rank safety (`map_shared_rank`)**: Every reachable
    `Stmt.clusterRankCheck rankVar` evaluates `rankVar` to an integer rank strictly
    below `f.clusterDimX` (guaranteeing valid Distributed Shared Memory addressing). -/
def MirNoOobClusterRank (f : MirFunc) : Prop :=
  ∀ env : InputEnv, ∀ t : ThreadCtx, ∀ b : Nat, ∀ phases : Nat × Nat, ∀ regs : RegFile, -- Pick ANY reachable block entry `(b, regs, phases)`:
    f.validInput env →                                      -- IF `env` satisfies launch preconditions,
    f.validThread t →                                       -- AND `t` is a valid GPU thread,
    Reachable f env t b regs phases →                       -- AND `(b, regs, phases)` is reachable,
    ∀ pfx sfx : List Stmt, ∀ rankVar r : Nat,               -- AND `Stmt.clusterRankCheck rankVar` appears inside `b.stmts`,
      (f.blockAt b).stmts = pfx ++ (.clusterRankCheck rankVar :: sfx) →
      (execStmts f env t regs phases (f.blockAt b).term pfx).1 rankVar = .int r →
      r < f.clusterDimX                                     -- THEN `r` is a valid peer CTA rank `< f.clusterDimX`.

/-- **`writes_all_elems(out)`**: Whenever the launched grid has at least one Y row
    (`0 < gridDimY`) and covers `out.len()` (`env.sliceLen paramId ≤ gridDimX * f.blockDimX`,
    with `env.sliceLen paramId < 2^32` when `f.u32Coords = true`), EVERY element index
    `elemIdx < env.sliceLen paramId` is written without hitting any `mir.assert` trap by some
    valid launched GPU thread `t` (`t.bx < gridDimX ∧ t.by_ < gridDimY`) starting from block `0`. -/
def MirWritesAllElems (f : MirFunc) (paramId : Nat) : Prop :=
  ∀ env : InputEnv, ∀ gridDimX gridDimY : Nat,              -- Pick ANY valid launch `env` and `(gridDimX, gridDimY)`:
    f.validInput env →                                      -- IF `env` satisfies launch preconditions,
    gridDimX ≤ 4294967296 →                                 -- AND `grid.x` fits in CUDA's 32-bit `u32` launch dimension,
    0 < gridDimY →                                          -- AND the launched grid has at least one Y row,
    (f.u32Coords = true → env.sliceLen paramId < 4294967296) → -- AND 32-bit coordinate mode fits `< 2^32`,
    env.sliceLen paramId ≤ gridDimX * f.blockDimX →         -- AND `grid.x * block.x` covers `out.len()` (host-checked),
    ∀ elemIdx : Nat, elemIdx < env.sliceLen paramId →       -- THEN for EVERY output slot `elemIdx < out.len()`,
      ∃ t : ThreadCtx, ∃ ev : MemEvent,                     -- there EXISTS a GPU thread `t` and write event `ev` such that:
        f.validThread t ∧                                   --   (1) `t` is within the CTA's `(blockDimX, blockDimY)` bounds,
        t.bx < gridDimX ∧ t.by_ < gridDimY ∧                --   (2) `t` belongs to an active launched CTA in the grid,
        EmitsTrapFreeEvent f env t ev ∧                     --   (3) `t` reaches and emits `ev` trap-free from entry block `0`,
        ev.isWrite = true ∧ ev.space = .global ∧            --   (4) `ev` is a global write to `paramId` at `ev.index = elemIdx`.
        ev.bufId = paramId ∧ ev.index = elemIdx

end CudaOxide.Contracts.Program
