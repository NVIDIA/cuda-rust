/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide.Contracts.ControlFlow

/-!
# Verified Control-Flow Checkers and Soundness Proofs (`CudaOxide.Proofs.ControlFlow`)

Provides `O(blocks)` decidable invariant checkers and proves once (`0 sorryAx`)
that whenever a checker returns `true`, the corresponding declarative path/trace
contract in `CudaOxide.Contracts.ControlFlow` holds across all valid CFG paths:

- `uniform_barriers_sound`:       `checkUniformBarriers cfg = true → UniformBarriers cfg`
- `tmem_lifecycle_sound`:         `checkTmemLifecycle cfg = true → TmemLifecycle cfg`
- `writes_on_active_paths_sound`: `checkWritesOnActivePaths cfg entries writes = true → WritesOnAllActivePaths cfg entries writes`
-/
namespace CudaOxide.Proofs.ControlFlow
open CudaOxide.Mir.Kernel
open CudaOxide.Mir.ControlFlow
open CudaOxide.Contracts.ControlFlow

/-- Closed inductive witness certifying that no `avoid`-free path from any `u ∈ vis`
    can ever reach `dst`. -/
def validUnreachWitness (cfg : ControlFlowCfg) (avoid : List Nat) (dst : Nat)
    (vis : List Nat) : Bool :=
  !(dst ∈ vis) &&
  vis.all fun u =>
    !(u ∈ avoid) &&
    ((cfg.blockAt u).succs.all fun (_, w) => decide (w ∈ avoid) || decide (w ∈ vis))

/-- If `vis` is a closed unreachability witness avoiding `avoid` and `dst`, then
    no `PathAvoiding cfg avoid u dst` can start from any `u ∈ vis`. -/
theorem pathAvoiding_false_of_witness {cfg : ControlFlowCfg} {avoid vis : List Nat} {u dst : Nat}
    (hwit : validUnreachWitness cfg avoid dst vis = true)
    (hp : PathAvoiding cfg avoid u dst)
    (hmem : u ∈ vis) : False := by
  induction hp with                                         -- Induct over the valid CFG path `u ->* dst`
  | refl _ _ =>
      unfold validUnreachWitness at hwit                    -- Expand `!(u ∈ vis) && vis.all ...`
      rcases Bool.and_eq_true_iff.mp hwit with ⟨hnotdst, _⟩ -- Extract `!(u ∈ vis) = true`
      simp [hmem] at hnotdst                                -- Contradiction with `u ∈ vis`
  | @step x w _ cond _ _ hedge hrest ih =>
      have hwit' := hwit                                    -- Preserve `validUnreachWitness` for the IH
      unfold validUnreachWitness at hwit                    -- Expand witness conjunction
      rcases Bool.and_eq_true_iff.mp hwit with ⟨_, hall⟩    -- Extract edge-closure `vis.all`
      have hx := List.all_eq_true.mp hall x hmem            -- Instantiate closure invariant at `x ∈ vis`
      rcases Bool.and_eq_true_iff.mp hx with ⟨_, hsuccs⟩    -- Extract outgoing-edge closure for `x`
      have hw := List.all_eq_true.mp hsuccs (cond, w) hedge -- Instantiate at taken CFG edge `(cond, w)`
      have hw_not_avoid : w ∉ avoid := by                   -- Extract `w ∉ avoid` from remaining path `hrest`
        cases hrest with
        | refl _ ha => exact ha
        | step _ ha _ _ => exact ha
      have hw_mem : w ∈ vis := by                           -- Since `w ∉ avoid`, `(w ∈ avoid) || (w ∈ vis)` forces `w ∈ vis`
        rcases Bool.or_eq_true_iff.mp hw with hav | hvis
        · exact False.elim (hw_not_avoid (of_decide_eq_true hav))
        · exact of_decide_eq_true hvis
      exact ih hwit' hw_mem                                 -- Apply induction hypothesis to suffix `w ->* dst`

/-- Computable certificate checker proving `¬ PathAvoiding cfg avoid src dst`. -/
def certifyUnreachable (cfg : ControlFlowCfg) (avoid : List Nat) (src dst : Nat) : Bool :=
  decide (src ∈ avoid) ||
  decide (src ≥ cfg.blocks.length) ||
  let vis := reachableNodesAvoiding cfg src avoid
  decide (src ∈ vis) && validUnreachWitness cfg avoid dst vis

/-- Soundness of `certifyUnreachable`: returning `true` guarantees no valid path
    `PathAvoiding cfg avoid src dst` exists in `cfg`. -/
theorem not_pathAvoiding_of_certify {cfg : ControlFlowCfg} {avoid : List Nat} {src dst : Nat}
    (hcert : certifyUnreachable cfg avoid src dst = true) :
    ¬ PathAvoiding cfg avoid src dst := by
  intro hp                                                  -- Assume a path `hp : PathAvoiding cfg avoid src dst` exists
  unfold certifyUnreachable at hcert                        -- Expand the three certificate branches
  rcases Bool.or_eq_true_iff.mp hcert with hleft | hright   -- Split `(src ∈ avoid || src ≥ len)` vs BFS witness
  · rcases Bool.or_eq_true_iff.mp hleft with hav | hge
    · have hna : src ∉ avoid := by                          -- First node of `hp` must satisfy `src ∉ avoid`
        cases hp with
        | refl _ ha => exact ha
        | step _ ha _ _ => exact ha
      exact hna (of_decide_eq_true hav)                     -- Contradiction with `src ∈ avoid`
    · have hlt : src < cfg.blocks.length := by              -- First node of `hp` must satisfy `src < blocks.length`
        cases hp with
        | refl hb _ => exact hb
        | step hb _ _ _ => exact hb
      have hge' : src ≥ cfg.blocks.length := of_decide_eq_true hge
      omega                                                 -- Contradiction between `src < len` and `src ≥ len`
  · rcases Bool.and_eq_true_iff.mp hright with ⟨hsrc, hwit⟩ -- Extract `src ∈ vis` and `validUnreachWitness`
    exact pathAvoiding_false_of_witness hwit hp (of_decide_eq_true hsrc) -- Discharge by inductive witness closure

/-- Per-block finite list of reachable `CtaState`s forming an inductive invariant map. -/
abbrev StateMap := List (List CtaState)

/-- Insert `st` into block `b`'s state list in `inv` if absent (capped at 16 states/block). -/
def addStateAt (inv : StateMap) (b : Nat) (st : CtaState) : StateMap :=
  let cur := inv.getD b []
  if st ∈ cur || cur.length ≥ 16 then
    inv
  else
    inv.set b (st :: cur)

/-- Execute a block's ops using precomputed `(blkGuard, inCycle)` metadata during inference. -/
def execBlockOpsCached (cfg : ControlFlowCfg) (blkGuard : GuardExpr) (inCycle : Bool)
    (st : CtaState) : List CfgOp → Option CtaState
  | [] => some st
  | op :: rest =>
      match stepOp cfg blkGuard inCycle st op with
      | none => none
      | some st' => execBlockOpsCached cfg blkGuard inCycle st' rest

/-- Propagate one sweep across all blocks using precomputed block guards and CTA edges. -/
def propagateOnce (cfg : ControlFlowCfg)
    (metaList : List (Nat × GuardExpr × Bool × List (BranchCond × Nat)))
    (inv : StateMap) : StateMap :=
  metaList.foldl (fun acc (b, blkGuard, inCycle, succEdges) =>
    let cur := acc.getD b []
    let ops := (cfg.blockAt b).ops
    cur.foldl (fun acc2 st =>
      match execBlockOpsCached cfg blkGuard inCycle st ops with
      | none => acc2
      | some outSt =>
          succEdges.foldl (fun acc3 (cond, dst) =>
            match stepEdge outSt cond with
            | none => acc3
            | some nextSt => addStateAt acc3 dst nextSt) acc2) acc) inv

/-- Iterate `propagateOnce` `fuel` times to reach a closed `StateMap`. -/
def propagateLoop (cfg : ControlFlowCfg)
    (metaList : List (Nat × GuardExpr × Bool × List (BranchCond × Nat))) :
    Nat → StateMap → StateMap
  | 0, inv => inv
  | fuel + 1, inv =>
      let next := propagateOnce cfg metaList inv
      if next == inv then inv else propagateLoop cfg metaList fuel next

/-- Infer the closed inductive `StateMap` for `cfg` starting from `(0, CtaState.initial cfg)`. -/
def inferStateMap (cfg : ControlFlowCfg) : StateMap :=
  match cfg.blocks.length with
  | 0 => []
  | n + 1 =>
      let metaList := (List.range (n + 1)).map fun b =>
        (b, activeGuardAt cfg b, isCyclicBlockBool cfg b, ctaSuccs cfg b)
      let init : StateMap := [CtaState.initial cfg] :: List.replicate n []
      propagateLoop cfg metaList (n + 3) init

/-- Check whether `inv` is an inductive state invariant for `cfg` where every
    reachable block succeeds, returns cleanly, and closes under all CTA edges. -/
def validStateMap (requireTmem : Bool) (cfg : ControlFlowCfg) (inv : StateMap) : Bool :=
  decide (0 < cfg.blocks.length) &&
  decide (CtaState.initial cfg ∈ inv.getD 0 []) &&
  (List.range cfg.blocks.length).all fun b =>
    (inv.getD b []).all fun st =>
      match execBlockOps cfg b st (cfg.blockAt b).ops with
      | none => false
      | some outSt =>
          (!(cfg.blockAt b).isReturn || validReturnState requireTmem outSt) &&
          ((ctaSuccs cfg b).all fun (cond, dst) =>
            decide (dst < cfg.blocks.length) &&
            match stepEdge outSt cond with
            | none => false
            | some nextSt => decide (nextSt ∈ inv.getD dst []))

/-- Every inductively reachable `(b, st)` (`CtaReachable cfg b st`) belongs to `inv.getD b []`
    whenever `validStateMap` holds. -/
theorem ctaReachable_mem_stateMap {requireTmem : Bool} {cfg : ControlFlowCfg} {inv : StateMap}
    {b : Nat} {st : CtaState}
    (hvalid : validStateMap requireTmem cfg inv = true)
    (hreach : CtaReachable cfg b st) :
    st ∈ inv.getD b [] := by
  unfold validStateMap at hvalid                            -- Unpack `validStateMap` into entry + inductive closure
  rcases Bool.and_eq_true_iff.mp hvalid with ⟨h01, hallB⟩
  rcases Bool.and_eq_true_iff.mp h01 with ⟨_, hinit⟩
  induction hreach with                                     -- Induct on the small-step `CtaReachable` derivation
  | entry _ =>
      exact of_decide_eq_true hinit                         -- Base: `CtaState.initial cfg ∈ inv.getD 0 []`
  | @step blk dst st0 outSt nextSt cond _ hblk hops hedge _ htrans ih =>
      have hb_range : blk ∈ List.range cfg.blocks.length := List.mem_range.mpr hblk -- `blk` is in `0..len-1`
      have hblk_ok := List.all_eq_true.mp hallB blk hb_range -- Instantiate block invariant at `blk`
      have hst_ok := List.all_eq_true.mp hblk_ok st0 ih     -- Instantiate state invariant at `st0 ∈ inv.getD blk []`
      rw [hops] at hst_ok                                   -- Substitute `execBlockOps ... = some outSt`
      rcases Bool.and_eq_true_iff.mp hst_ok with ⟨_, hedges⟩ -- Extract outgoing-edge closure for `outSt`
      have hedge_ok := List.all_eq_true.mp hedges (cond, dst) hedge -- Instantiate at edge `(cond, dst)`
      rcases Bool.and_eq_true_iff.mp hedge_ok with ⟨_, hstep_ok⟩
      rw [htrans] at hstep_ok                               -- Substitute `stepEdge outSt cond = some nextSt`
      exact of_decide_eq_true hstep_ok                      -- Conclude `nextSt ∈ inv.getD dst []`

/-- Computable decision checker for `SafeCtaExecution requireTmem cfg`. -/
def checkCtaExecution (requireTmem : Bool) (cfg : ControlFlowCfg) : Bool :=
  validStateMap requireTmem cfg (inferStateMap cfg)

/-- Soundness of `checkCtaExecution`: returning `true` proves `SafeCtaExecution`. -/
theorem safeCtaExecution_of_check {requireTmem : Bool} {cfg : ControlFlowCfg}
    (hcheck : checkCtaExecution requireTmem cfg = true) :
    SafeCtaExecution requireTmem cfg := by
  unfold checkCtaExecution at hcheck                        -- Let `inv := inferStateMap cfg`
  have hvalid := hcheck                                     -- Preserve full `validStateMap` hypothesis for `ctaReachable_mem_stateMap`
  unfold validStateMap at hcheck                            -- Unpack `0 < blocks.length` and per-block invariant
  rcases Bool.and_eq_true_iff.mp hcheck with ⟨h01, hallB⟩
  rcases Bool.and_eq_true_iff.mp h01 with ⟨hpos, _⟩
  refine ⟨of_decide_eq_true hpos, ?_⟩                       -- Prove `0 < cfg.blocks.length`, then unpack `∀ b st`
  intro b st hreach                                         -- Take ANY reachable `(b, st)`
  have hblk : b < cfg.blocks.length := by                   -- Every `CtaReachable` target block `b` is `< blocks.length`
    cases hreach with
    | entry hp => exact hp
    | step _ _ _ _ hdst _ => exact hdst
  have hmem := ctaReachable_mem_stateMap hvalid hreach      -- By inductive closure, `st ∈ (inferStateMap cfg).getD b []`
  have hb_ok := List.all_eq_true.mp hallB b (List.mem_range.mpr hblk)
  have hst_ok := List.all_eq_true.mp hb_ok st hmem          -- Instantiate validated check on `(b, st)`
  cases hops : execBlockOps cfg b st (cfg.blockAt b).ops with -- Case-split on `execBlockOps`
  | none =>
      rw [hops] at hst_ok; contradiction                    -- `none` evaluates to `false = true`, impossible
  | some outSt =>
      rw [hops] at hst_ok                                   -- Simplify `hst_ok` under `some outSt`
      rcases Bool.and_eq_true_iff.mp hst_ok with ⟨hret_ok, hedges_ok⟩
      refine ⟨outSt, rfl, ?_, ?_⟩                           -- Witness `outSt`
      · intro hret                                          -- Goal 1: if `isReturn = true`, `validReturnState` holds
        rcases Bool.or_eq_true_iff.mp hret_ok with hnot | hv
        · rw [hret] at hnot; contradiction                  -- `!true = true` is impossible
        · exact hv
      · intro cond dst hedge                                -- Goal 2: every outgoing edge `(cond, dst)` steps cleanly
        have he := List.all_eq_true.mp hedges_ok (cond, dst) hedge
        rcases Bool.and_eq_true_iff.mp he with ⟨hdst, hstep⟩
        refine ⟨of_decide_eq_true hdst, ?_⟩
        cases hs : stepEdge outSt cond with
        | none => rw [hs] at hstep; contradiction           -- `none` evaluates to `false = true`, impossible
        | some nextSt => exact ⟨nextSt, rfl⟩                -- Witness `nextSt`

/-- Computable decision checker for `NoUnbackpressuredLoop cfg`. -/
def checkLoopBackpressure (cfg : ControlFlowCfg) : Bool :=
  (loopBackpressurePairs cfg).all fun (prod, cons, bp) =>
    certifyUnreachable cfg bp cons prod

/-- Soundness of `checkLoopBackpressure`: returning `true` proves `NoUnbackpressuredLoop`. -/
theorem noUnbackpressuredLoop_of_check {cfg : ControlFlowCfg}
    (hcheck : checkLoopBackpressure cfg = true) :
    NoUnbackpressuredLoop cfg := by
  intro prod cons bp hmem                                   -- Pick ANY cyclic `(prod, cons, bp)` pair
  have hcert := List.all_eq_true.mp hcheck (prod, cons, bp) hmem -- Extract its unreachability certificate
  exact not_pathAvoiding_of_certify hcert                   -- Apply `not_pathAvoiding_of_certify`

/-- Computable decision checker for `UniformBarriers cfg` (covering `sync_threads()`
    uniformity plus `mbarrier`/proxy/back-pressure safety when `Barrier`s exist). -/
def checkUniformBarriers (cfg : ControlFlowCfg) : Bool :=
  decide (0 < countEffectiveBarriers cfg) &&
  ((syncBarrierBlocks cfg).all fun b =>
    decide (activeGuardAt cfg b = .tr) &&
    (isCyclicBlockBool cfg b ||
      (returnBlocks cfg).all fun ret => certifyUnreachable cfg [b] 0 ret)) &&
  ((decide (cyclicBarrierBlocks cfg = []) || decide (cyclicBarrierLoopBodyStarts cfg ≠ [])) &&
   (cyclicBarrierLoopBodyStarts cfg).all fun (hdr, body) =>
    ((returnBlocks cfg).all fun ret => certifyUnreachable cfg [hdr] 0 ret) &&
    ((cyclicBarrierBlocks cfg).all fun cb => certifyUnreachable cfg [cb] body hdr) &&
    ((returnBlocks cfg).all fun ret => certifyUnreachable cfg [hdr] body ret)) &&
  ((signaledMbarReadyBlocks cfg).all fun rBlk =>
    (returnBlocks cfg).all fun ret => certifyUnreachable cfg [rBlk] 0 ret) &&
  (!(decide (0 < cfg.numBarriers) && decide (tmemAllocSites cfg = [])) ||
    (checkCtaExecution false cfg && checkLoopBackpressure cfg))

/-- Reusable soundness theorem for `uniform_barriers`: `checkUniformBarriers cfg = true`
    proves `UniformBarriers cfg` across all CFG paths. -/
theorem uniform_barriers_sound {cfg : ControlFlowCfg}
    (hcheck : checkUniformBarriers cfg = true) :
    UniformBarriers cfg := by
  unfold checkUniformBarriers at hcheck                     -- Unpack the five boolean conjuncts of `checkUniformBarriers`
  rcases Bool.and_eq_true_iff.mp hcheck with ⟨h0123, hasync⟩
  rcases Bool.and_eq_true_iff.mp h0123 with ⟨h012, hmbar⟩
  rcases Bool.and_eq_true_iff.mp h012 with ⟨h01, hcyc⟩
  rcases Bool.and_eq_true_iff.mp h01 with ⟨hcnt, hsync⟩
  rcases Bool.and_eq_true_iff.mp hcyc with ⟨hhasHdr, hloops⟩
  refine ⟨of_decide_eq_true hcnt, ?_, ⟨?_, ?_⟩, ?_, ?_⟩     -- Prove each of the five `UniformBarriers` clauses
  · intro b hb                                              -- Clause 1: every `sync_threads()` block `b`
    have hb_ok := List.all_eq_true.mp hsync b hb
    rcases Bool.and_eq_true_iff.mp hb_ok with ⟨htr, hacyc⟩
    refine ⟨of_decide_eq_true htr, ?_⟩
    intro hnotcyc ret hret
    rcases Bool.or_eq_true_iff.mp hacyc with hcyc_true | hall_ret
    · rw [hnotcyc] at hcyc_true; contradiction              -- `false = true` impossible when `b` is acyclic
    · exact not_pathAvoiding_of_certify (List.all_eq_true.mp hall_ret ret hret)
  · intro hne                                               -- Clause 2a: non-empty cyclic barriers have an enclosing header
    rcases Bool.or_eq_true_iff.mp hhasHdr with hemp | hnonemp
    · exact False.elim (hne (of_decide_eq_true hemp))
    · exact of_decide_eq_true hnonemp
  · intro hdr body hpair                                    -- Clause 2b: cyclic loop barrier & exit checks
    have hp_ok := List.all_eq_true.mp hloops (hdr, body) hpair
    rcases Bool.and_eq_true_iff.mp hp_ok with ⟨h12, hmid⟩
    rcases Bool.and_eq_true_iff.mp h12 with ⟨hentry, hcb⟩
    refine ⟨?_, ?_, ?_⟩
    · intro ret hret
      exact not_pathAvoiding_of_certify (List.all_eq_true.mp hentry ret hret)
    · intro cb hcb_mem
      exact not_pathAvoiding_of_certify (List.all_eq_true.mp hcb cb hcb_mem)
    · intro ret hret
      exact not_pathAvoiding_of_certify (List.all_eq_true.mp hmid ret hret)
  · intro rBlk hrBlk ret hret                               -- Clause 3: signaled `mbarrier` ready-block uniformity
    have hr := List.all_eq_true.mp hmbar rBlk hrBlk
    exact not_pathAvoiding_of_certify (List.all_eq_true.mp hr ret hret)
  · intro ⟨hbars, hnotmem⟩                                  -- Clause 4: `mbarrier` CTA execution & loop back-pressure
    have hcond : (decide (0 < cfg.numBarriers) && decide (tmemAllocSites cfg = [])) = true := by
      simp [hbars, hnotmem]
    rcases Bool.or_eq_true_iff.mp hasync with hneg | hboth
    · rw [hcond] at hneg; contradiction                     -- `!true = true` is impossible
    · rcases Bool.and_eq_true_iff.mp hboth with ⟨hcta, hloop⟩
      exact ⟨safeCtaExecution_of_check hcta, noUnbackpressuredLoop_of_check hloop⟩

/-- Bridge `validTmemColsBool cols = true` to the declarative `TmemValidCols cols` proposition. -/
theorem tmemValidCols_of_bool {cols : Nat}
    (h : validTmemColsBool cols = true) :
    CudaOxide.Contracts.TmemValidCols cols := by
  unfold validTmemColsBool at h                             -- Unfold `cols == 32 || ... || cols == 512`
  unfold CudaOxide.Contracts.TmemValidCols                  -- Unfold `cols = 32 ∨ ... ∨ cols = 512`
  simp only [Bool.or_eq_true, beq_iff_eq] at h              -- Convert boolean `||` / `==` into logical `∨` / `=`
  rcases h with (((h32 | h64) | h128) | h256) | h512        -- Reassociate into right-nested `∨`
  · exact Or.inl h32
  · exact Or.inr (Or.inl h64)
  · exact Or.inr (Or.inr (Or.inl h128))
  · exact Or.inr (Or.inr (Or.inr (Or.inl h256)))
  · exact Or.inr (Or.inr (Or.inr (Or.inr h512)))

/-- Computable decision checker for `TmemLifecycle cfg`. -/
def checkTmemLifecycle (cfg : ControlFlowCfg) : Bool :=
  let allocOk :=
    match tmemAllocSites cfg with
    | [cols] =>
        validTmemColsBool cols &&
        decide (tmemDeallocSites cfg ≠ []) &&
        (tmemDeallocSites cfg).all (· == cols)
    | _ => false
  allocOk && checkCtaExecution true cfg && checkLoopBackpressure cfg

/-- Reusable soundness theorem for `tmem_lifecycle`: `checkTmemLifecycle cfg = true`
    proves `TmemLifecycle cfg` across all reachable CTA states and paths. -/
theorem tmem_lifecycle_sound {cfg : ControlFlowCfg}
    (hcheck : checkTmemLifecycle cfg = true) :
    TmemLifecycle cfg := by
  unfold checkTmemLifecycle at hcheck                       -- Unpack `allocOk`, `checkCtaExecution`, `checkLoopBackpressure`
  rcases Bool.and_eq_true_iff.mp hcheck with ⟨ha1, hloop⟩
  rcases Bool.and_eq_true_iff.mp ha1 with ⟨halloc, hcta⟩
  refine ⟨?_, safeCtaExecution_of_check hcta, noUnbackpressuredLoop_of_check hloop⟩
  cases hs : tmemAllocSites cfg with                        -- Match `tmemAllocSites cfg` against `[cols]`
  | nil =>
      rw [hs] at halloc; contradiction                      -- `[]` evaluates `allocOk` to `false`, impossible
  | cons cols rest =>
      cases rest with
      | cons _ _ =>
          rw [hs] at halloc; contradiction                  -- `≥ 2` alloc sites evaluates `allocOk` to `false`
      | nil =>
          rw [hs] at halloc                                 -- Single alloc site `[cols]`
          rcases Bool.and_eq_true_iff.mp halloc with ⟨hv1, hallDealloc⟩
          rcases Bool.and_eq_true_iff.mp hv1 with ⟨hcols, hnonempty⟩
          refine ⟨cols, tmemValidCols_of_bool hcols, rfl, of_decide_eq_true hnonempty, ?_⟩
          intro d hd                                        -- Every dealloc site `d` satisfies `(d == cols) = true`
          exact beq_iff_eq.mp (List.all_eq_true.mp hallDealloc d hd)

/-- Computable decision checker for `WritesOnAllActivePaths cfg guardEntries writeBlocks`. -/
def checkWritesOnActivePaths (cfg : ControlFlowCfg)
    (guardEntries writeBlocks : List Nat) : Bool :=
  decide (guardEntries ≠ []) &&
  decide (writeBlocks ≠ []) &&
  guardEntries.all fun entry =>
    (returnBlocks cfg).all fun ret =>
      certifyUnreachable cfg writeBlocks entry ret

/-- Reusable soundness theorem for `writes_all_elems` CFG post-dominance:
    proves every path from `idx < len` to `return` executes a write block. -/
theorem writes_on_active_paths_sound {cfg : ControlFlowCfg}
    {guardEntries writeBlocks : List Nat}
    (hcheck : checkWritesOnActivePaths cfg guardEntries writeBlocks = true) :
    WritesOnAllActivePaths cfg guardEntries writeBlocks := by
  unfold checkWritesOnActivePaths at hcheck                 -- Unpack non-emptiness + all-pairs `certifyUnreachable`
  rcases Bool.and_eq_true_iff.mp hcheck with ⟨hne12, hall⟩
  rcases Bool.and_eq_true_iff.mp hne12 with ⟨hg, hw⟩
  refine ⟨of_decide_eq_true hg, of_decide_eq_true hw, ?_⟩   -- Prove non-empty lists, then unpack `∀ entry ret`
  intro entry hentry ret hret
  have he := List.all_eq_true.mp hall entry hentry          -- Instantiate at `entry ∈ guardEntries`
  have hr := List.all_eq_true.mp he ret hret                -- Instantiate at `ret ∈ returnBlocks cfg`
  exact not_pathAvoiding_of_certify hr                      -- Discharge via `not_pathAvoiding_of_certify`

end CudaOxide.Proofs.ControlFlow
