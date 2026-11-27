/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import Lean
import CudaOxide.Proofs.ControlFlow.Lattice

/-!
# Certified `ControlFlowCfg` State-Map Soundness (`CudaOxide.Proofs.ControlFlow.Soundness`)
-/
namespace CudaOxide.Proofs.ControlFlow
open CudaOxide.Mir.ControlFlow
open CudaOxide.Contracts.ControlFlow

/-- Check whether `vis` is a closed subgraph avoiding `avoid` from which no path
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

/-- Abstract invariant map keeping at most one joined `CtaState` per `visitedPosBodies` phase at each block. -/
abbrev StateMap := List (List CtaState)

/-- Join `st` into `curList` whenever an entry shares `st.visitedPosBodies` (collapsing `2^k` `if/else`
    diamond states into 1 summary while keeping pre-body vs. post-body positive-trip loop states exact). -/
def joinStateInList (st : CtaState) : List CtaState → List CtaState
  | [] => [st]
  | cur :: rest =>
      if cur.visitedPosBodies == st.visitedPosBodies then
        CtaState.join cur st :: rest
      else
        cur :: joinStateInList st rest

/-- Join `st` into block `b`'s abstract summaries in `inv`. -/
def joinStateAt (inv : StateMap) (b : Nat) (st : CtaState) : StateMap :=
  let cur := inv.getD b []
  let next := joinStateInList st cur
  if next == cur then inv else inv.set b next

/-- Alias for `execBlockOpsWith` used during state-map inference and validation. -/
abbrev execBlockOpsCached := execBlockOpsWith

/-- Definitional equivalence between `execBlockOpsCached` and `execBlockOps`. -/
theorem execBlockOpsCached_eq (cfg : ControlFlowCfg) (b : Nat) (st : CtaState) (ops : List CfgOp) :
    execBlockOpsCached cfg b (decide (b ∈ posLoopExitBodies cfg)) (activeGuardAt cfg b)
      (isCyclicBlockBool cfg b) st ops = execBlockOps cfg b st ops := rfl


/-- Propagate one sweep across all blocks joining `if/else` diamonds per `visitedPosBodies` phase. -/
def propagateOnce (cfg : ControlFlowCfg) (clusterBars commitBars : List Nat)
    (metaList : List (Nat × Bool × GuardExpr × Bool × List Nat × List (BranchCond × Nat)))
    (inv : StateMap) : StateMap :=
  metaList.foldl (fun acc (b, isPosBody, blkGuard, inCycle, warpSigBars, succEdges) =>
    let cur := acc.getD b []
    let ops := (cfg.blockAt b).ops
    cur.foldl (fun acc2 st =>
      match execBlockOpsCached cfg b isPosBody blkGuard inCycle st ops with
      | none => acc2
      | some outSt =>
          succEdges.foldl (fun acc3 (cond, dst) =>
            let enabled := match cond with
              | .posLoopExit bodyBlk => decide (bodyBlk ∈ outSt.visitedPosBodies)
              | _ => true
            if !enabled then acc3
            else
              match stepEdgeWith clusterBars commitBars warpSigBars outSt cond with
              | none => acc3
              | some nextSt => joinStateAt acc3 dst nextSt) acc2) acc) inv

/-- Iterate `propagateOnce` `fuel` times until the abstract `StateMap` fixpoint stabilizes. -/
def propagateLoop (cfg : ControlFlowCfg) (clusterBars commitBars : List Nat)
    (metaList : List (Nat × Bool × GuardExpr × Bool × List Nat × List (BranchCond × Nat))) :
    Nat → StateMap → StateMap
  | 0, inv => inv
  | fuel + 1, inv =>
      let next := propagateOnce cfg clusterBars commitBars metaList inv
      if next == inv then inv else propagateLoop cfg clusterBars commitBars metaList fuel next

/-- Infer the abstract inductive `StateMap` for `cfg` (`O(blocks)` space & time). -/
def inferStateMap (cfg : ControlFlowCfg) : StateMap :=
  match cfg.blocks.length with
  | 0 => []
  | n + 1 =>
      let posBodies := posLoopExitBodies cfg
      let clusterBars := clusterSignaledBarrierIds cfg
      let commitBars := committedTmemBarrierIds cfg
      let metaList := (List.range (n + 1)).map fun b =>
        (b, decide (b ∈ posBodies), activeGuardAt cfg b, isCyclicBlockBool cfg b,
         warpWaitSignaledBarsAt cfg b, ctaSuccs cfg b)
      let init : StateMap := [CtaState.initial cfg] :: List.replicate n []
      propagateLoop cfg clusterBars commitBars metaList (4 * (n + 2)) init

/-- Verify in `O(blocks)` time that `inv` is an inductive abstract certificate for `cfg`. -/
def validStateMap (requireTmem : Bool) (cfg : ControlFlowCfg) (inv : StateMap) : Bool :=
  decide (0 < cfg.blocks.length) &&
  ((inv.getD 0 []).any fun s0 => s0.subsumes (CtaState.initial cfg)) &&
  let posBodies := posLoopExitBodies cfg
  let clusterBars := clusterSignaledBarrierIds cfg
  let commitBars := committedTmemBarrierIds cfg
  (List.range cfg.blocks.length).all fun b =>
    let cur := inv.getD b []
    if cur.isEmpty then true
    else
      let blkGuard := activeGuardAt cfg b
      let inCycle := isCyclicBlockBool cfg b
      let isPos := decide (b ∈ posBodies)
      let warpSigBars := warpWaitSignaledBarsAt cfg b
      let blk := cfg.blockAt b
      cur.all fun st =>
        match execBlockOpsCached cfg b isPos blkGuard inCycle st blk.ops with
        | none => false
        | some outSt =>
            (!blk.isReturn || validReturnState requireTmem outSt) &&
            ((enabledCtaSuccs cfg b outSt).all fun (cond, dst) =>
              decide (dst < cfg.blocks.length) &&
              match stepEdgeWith clusterBars commitBars warpSigBars outSt cond with
              | none => false
              | some nextSt => (inv.getD dst []).any fun dstSt => dstSt.subsumes nextSt)

/-- Inductive simulation theorem: every concrete reachable state `(b, st)` (`CtaReachable cfg b st`)
    is subsumed by an abstract summary `abs ∈ inv.getD b []`. -/
theorem ctaReachable_subsumed {requireTmem : Bool} {cfg : ControlFlowCfg} {inv : StateMap}
    {b : Nat} {st : CtaState}
    (hvalid : validStateMap requireTmem cfg inv = true)
    (hreach : CtaReachable cfg b st) :
    ∃ abs ∈ inv.getD b [], abs.subsumes st = true := by
  unfold validStateMap at hvalid                          -- Unpack `validStateMap` entry & per-block checks
  rcases Bool.and_eq_true_iff.mp hvalid with ⟨h01, hallB⟩
  rcases Bool.and_eq_true_iff.mp h01 with ⟨_, hinit⟩
  induction hreach with                                   -- Induction on concrete `CtaReachable` steps
  | entry _ =>                                            -- Base case (`block 0`): `CtaState.initial` is subsumed by `inv[0]`
      exact List.any_eq_true.mp hinit
  | @step blk dst st0 outSt nextSt cond _ hblk hops hedge _ htrans ih => -- Inductive step (`blk -> dst`)
      obtain ⟨abs0, habs0, hsub0⟩ := ih                   --   Step 1: Obtain abstract `abs0` subsuming `st0` at `blk`
      have hblk_ok := List.all_eq_true.mp hallB blk (List.mem_range.mpr hblk)
      dsimp only at hblk_ok
      have hne : (inv.getD blk []).isEmpty = false := by
        cases hl : inv.getD blk [] <;> simp_all
      rw [hne] at hblk_ok; simp only [Bool.false_eq_true, ↓reduceIte] at hblk_ok
      have hst_ok := List.all_eq_true.mp hblk_ok abs0 habs0
      rw [execBlockOpsCached_eq] at hst_ok
      cases hopsA : execBlockOps cfg blk abs0 (cfg.blockAt blk).ops with
      | none => rw [hopsA] at hst_ok; contradiction
      | some outAbs =>                                    --   Step 2: `execBlockOps_mono` preserves subsumption across `blk.ops`
          rw [hopsA] at hst_ok
          obtain ⟨outConc', hopsC', hsubOut⟩ := execBlockOps_mono cfg blk (cfg.blockAt blk).ops hsub0 hopsA
          rw [hops] at hopsC'; injection hopsC' with heqOut; subst heqOut
          rcases Bool.and_eq_true_iff.mp hst_ok with ⟨_, hedges⟩
          have hedgeA := enabledCtaSuccs_mono hsubOut hedge
          have hedge_ok := List.all_eq_true.mp hedges (cond, dst) hedgeA
          rcases Bool.and_eq_true_iff.mp hedge_ok with ⟨_, hstep_ok⟩
          have hse : stepEdgeWith (clusterSignaledBarrierIds cfg) (committedTmemBarrierIds cfg)
              (warpWaitSignaledBarsAt cfg blk) outAbs cond = stepEdge cfg blk outAbs cond := rfl
          rw [hse] at hstep_ok
          cases htrA : stepEdge cfg blk outAbs cond with
          | none => rw [htrA] at hstep_ok; contradiction
          | some nextAbs =>                               --   Step 3: `stepEdge_mono` + `ctaSubsumes_trans` cover `nextSt` at `dst`
              rw [htrA] at hstep_ok
              obtain ⟨dstAbs, hdstMem, hdstSub⟩ := List.any_eq_true.mp hstep_ok
              have hconcEn : branchCondPosEn outSt cond = true := by
                unfold enabledCtaSuccs at hedge; rw [List.mem_filter] at hedge
                cases cond <;> exact hedge.2
              obtain ⟨nextConc', htrC', hsubNext⟩ := stepEdge_mono cfg blk cond hsubOut hconcEn htrA
              rw [htrans] at htrC'; injection htrC' with heqNext; subst heqNext
              exact ⟨dstAbs, hdstMem, ctaSubsumes_trans hdstSub hsubNext⟩

/-- Soundness of `validStateMap`: any valid abstract `StateMap` proves `SafeCtaExecution`. -/
theorem safeCtaExecution_of_stateMap {requireTmem : Bool} {cfg : ControlFlowCfg} {inv : StateMap}
    (hvalid : validStateMap requireTmem cfg inv = true) :
    SafeCtaExecution requireTmem cfg := by
  have hcheck := hvalid
  unfold validStateMap at hcheck
  rcases Bool.and_eq_true_iff.mp hcheck with ⟨h01, hallB⟩
  rcases Bool.and_eq_true_iff.mp h01 with ⟨hpos, _⟩
  refine ⟨of_decide_eq_true hpos, ?_⟩
  intro b st hreach
  have hblk : b < cfg.blocks.length := by
    cases hreach with
    | entry hp => exact hp
    | step _ _ _ _ hdst _ => exact hdst
  obtain ⟨abs, habs, hsub⟩ := ctaReachable_subsumed hvalid hreach
  have hblk_ok := List.all_eq_true.mp hallB b (List.mem_range.mpr hblk)
  dsimp only at hblk_ok
  have hne : (inv.getD b []).isEmpty = false := by
    cases hl : inv.getD b [] <;> simp_all
  rw [hne] at hblk_ok; simp only [Bool.false_eq_true, ↓reduceIte] at hblk_ok
  have hst_ok := List.all_eq_true.mp hblk_ok abs habs
  rw [execBlockOpsCached_eq] at hst_ok
  cases hopsA : execBlockOps cfg b abs (cfg.blockAt b).ops with
  | none => rw [hopsA] at hst_ok; contradiction
  | some outAbs =>
      rw [hopsA] at hst_ok
      obtain ⟨outConc, hopsC, hsubOut⟩ := execBlockOps_mono cfg b (cfg.blockAt b).ops hsub hopsA
      rcases Bool.and_eq_true_iff.mp hst_ok with ⟨hret_ok, hedges_ok⟩
      refine ⟨outConc, hopsC, ?_, ?_⟩
      · intro hret
        rcases Bool.or_eq_true_iff.mp hret_ok with hnot | hv
        · rw [hret] at hnot; contradiction
        · exact validReturnState_mono hsubOut hv
      · intro cond dst hedge
        have hedgeA := enabledCtaSuccs_mono hsubOut hedge
        have he := List.all_eq_true.mp hedges_ok (cond, dst) hedgeA
        rcases Bool.and_eq_true_iff.mp he with ⟨hdst, hstep⟩
        refine ⟨of_decide_eq_true hdst, ?_⟩
        have hse : stepEdgeWith (clusterSignaledBarrierIds cfg) (committedTmemBarrierIds cfg)
            (warpWaitSignaledBarsAt cfg b) outAbs cond = stepEdge cfg b outAbs cond := rfl
        rw [hse] at hstep
        cases hsA : stepEdge cfg b outAbs cond with
        | none => rw [hsA] at hstep; contradiction
        | some nextAbs =>
            have hconcEn : branchCondPosEn outConc cond = true := by
              unfold enabledCtaSuccs at hedge; rw [List.mem_filter] at hedge
              cases cond <;> exact hedge.2
            obtain ⟨nextConc, htrC, _⟩ := stepEdge_mono cfg b cond hsubOut hconcEn hsA
            exact ⟨nextConc, htrC⟩

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

/-- Check `UniformBarriers cfg` against a precomputed `inv : StateMap` in `O(blocks)` time. -/
def checkUniformBarriersWith (cfg : ControlFlowCfg) (inv : StateMap) : Bool :=
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
  ((unsyncedDsmemBlocks cfg).all fun dBlk =>
    let postSyncs := (clusterSyncBlocks cfg).filter (· != dBlk)
    (returnBlocks cfg).all fun ret => certifyUnreachable cfg postSyncs dBlk ret) &&
  ((sharedReadSites cfg).all fun (s, rBlk) =>
    certifyUnreachable cfg (sharedInitDominators cfg s) 0 rBlk) &&
  validMbarArrivalCounts cfg &&
  (!(decide (0 < cfg.numBarriers) && decide (tmemAllocSites cfg = [])) ||
    (validStateMap false cfg inv && checkLoopBackpressure cfg))

/-- Reusable soundness theorem for `uniform_barriers` using a precomputed `inv : StateMap`. -/
theorem uniform_barriers_of_stateMap {cfg : ControlFlowCfg} {inv : StateMap}
    (hcheck : checkUniformBarriersWith cfg inv = true) :
    UniformBarriers cfg := by
  unfold checkUniformBarriersWith at hcheck
  rcases Bool.and_eq_true_iff.mp hcheck with ⟨h0123456, hasync⟩
  rcases Bool.and_eq_true_iff.mp h0123456 with ⟨h012345, hmbarCnt⟩
  rcases Bool.and_eq_true_iff.mp h012345 with ⟨h01234, hsinit⟩
  rcases Bool.and_eq_true_iff.mp h01234 with ⟨h0123, hdsmem⟩
  rcases Bool.and_eq_true_iff.mp h0123 with ⟨h012, hmbar⟩
  rcases Bool.and_eq_true_iff.mp h012 with ⟨h01, hcyc⟩
  rcases Bool.and_eq_true_iff.mp h01 with ⟨hcnt, hsync⟩
  rcases Bool.and_eq_true_iff.mp hcyc with ⟨hhasHdr, hloops⟩
  refine ⟨of_decide_eq_true hcnt, ?_, ⟨?_, ?_⟩, ?_, ?_, ?_, hmbarCnt, ?_⟩
  · intro b hb
    have hb_ok := List.all_eq_true.mp hsync b hb
    rcases Bool.and_eq_true_iff.mp hb_ok with ⟨htr, hacyc⟩
    refine ⟨of_decide_eq_true htr, ?_⟩
    intro hnotcyc ret hret
    rcases Bool.or_eq_true_iff.mp hacyc with hcyc_true | hall_ret
    · rw [hnotcyc] at hcyc_true; contradiction
    · exact not_pathAvoiding_of_certify (List.all_eq_true.mp hall_ret ret hret)
  · intro hne
    rcases Bool.or_eq_true_iff.mp hhasHdr with hemp | hnonemp
    · exact False.elim (hne (of_decide_eq_true hemp))
    · exact of_decide_eq_true hnonemp
  · intro hdr body hpair
    have hp_ok := List.all_eq_true.mp hloops (hdr, body) hpair
    rcases Bool.and_eq_true_iff.mp hp_ok with ⟨h12, hmid⟩
    rcases Bool.and_eq_true_iff.mp h12 with ⟨hentry, hcb⟩
    refine ⟨?_, ?_, ?_⟩
    · intro ret hret; exact not_pathAvoiding_of_certify (List.all_eq_true.mp hentry ret hret)
    · intro cb hcb_mem; exact not_pathAvoiding_of_certify (List.all_eq_true.mp hcb cb hcb_mem)
    · intro ret hret; exact not_pathAvoiding_of_certify (List.all_eq_true.mp hmid ret hret)
  · intro rBlk hrBlk ret hret
    have hr := List.all_eq_true.mp hmbar rBlk hrBlk
    exact not_pathAvoiding_of_certify (List.all_eq_true.mp hr ret hret)
  · intro dBlk hdBlk ret hret
    have hd := List.all_eq_true.mp hdsmem dBlk hdBlk
    exact not_pathAvoiding_of_certify (List.all_eq_true.mp hd ret hret)
  · intro s rBlk hsite
    exact not_pathAvoiding_of_certify (List.all_eq_true.mp hsinit (s, rBlk) hsite)
  · intro ⟨hbars, hnotmem⟩
    have hcond : (decide (0 < cfg.numBarriers) && decide (tmemAllocSites cfg = [])) = true := by
      simp [hbars, hnotmem]
    rcases Bool.or_eq_true_iff.mp hasync with hneg | hboth
    · rw [hcond] at hneg; contradiction
    · rcases Bool.and_eq_true_iff.mp hboth with ⟨hvalid, hloop⟩
      exact ⟨safeCtaExecution_of_stateMap hvalid, noUnbackpressuredLoop_of_check hloop⟩

/-- Bridge `validTmemColsBool cols = true` to the declarative `TmemValidCols cols` proposition. -/
theorem tmemValidCols_of_bool {cols : Nat}
    (h : validTmemColsBool cols = true) :
    TmemValidCols cols := by
  unfold validTmemColsBool at h                             -- Unfold `cols == 32 || ... || cols == 512`
  unfold TmemValidCols                                      -- Unfold `cols = 32 ∨ ... ∨ cols = 512`
  simp only [Bool.or_eq_true, beq_iff_eq] at h              -- Convert boolean `||` / `==` into logical `∨` / `=`
  rcases h with (((h32 | h64) | h128) | h256) | h512        -- Reassociate into right-nested `∨`
  · exact Or.inl h32
  · exact Or.inr (Or.inl h64)
  · exact Or.inr (Or.inr (Or.inl h128))
  · exact Or.inr (Or.inr (Or.inr (Or.inl h256)))
  · exact Or.inr (Or.inr (Or.inr (Or.inr h512)))

/-- Check `TmemLifecycle cfg` against a precomputed `inv : StateMap` in `O(blocks)` time. -/
def checkTmemLifecycleWith (cfg : ControlFlowCfg) (inv : StateMap) : Bool :=
  let allocOk := match tmemAllocSites cfg with
    | [cols] =>
        validTmemColsBool cols &&
        decide (tmemDeallocSites cfg ≠ []) &&
        (tmemDeallocSites cfg).all (· == cols)
    | _ => false
  let cg2LaunchOk := (unlaunchedCg2AllocBlocks cfg).all fun aBlk =>
    certifyUnreachable cfg ((clusterSyncBlocks cfg).filter (· != aBlk)) 0 aBlk
  let cg2EntryOk := (cg2PreAllocInitPairs cfg).all fun (iBlk, aBlk) =>
    certifyUnreachable cfg ((clusterSyncBlocks cfg).filter (· != iBlk)) iBlk aBlk
  let cg2ExitOk := (unsyncedCg2DeallocBlocks cfg).all fun dBlk =>
    let postSyncs := (clusterSyncBlocks cfg).filter (· != dBlk)
    (returnBlocks cfg).all fun ret => certifyUnreachable cfg postSyncs dBlk ret
  let sinitOk := (sharedReadSites cfg).all fun (s, rBlk) =>
    certifyUnreachable cfg (sharedInitDominators cfg s) 0 rBlk
  allocOk && cg2LaunchOk && cg2EntryOk && cg2ExitOk && sinitOk && validMbarArrivalCounts cfg && validStateMap true cfg inv && checkLoopBackpressure cfg

/-- Reusable soundness theorem for `tmem_lifecycle` using a precomputed `inv : StateMap`. -/
theorem tmem_lifecycle_of_stateMap {cfg : ControlFlowCfg} {inv : StateMap}
    (hcheck : checkTmemLifecycleWith cfg inv = true) :
    TmemLifecycle cfg := by
  unfold checkTmemLifecycleWith at hcheck
  rcases Bool.and_eq_true_iff.mp hcheck with ⟨ha123456, hloop⟩
  rcases Bool.and_eq_true_iff.mp ha123456 with ⟨ha12345, hvalid⟩
  rcases Bool.and_eq_true_iff.mp ha12345 with ⟨ha1234, hmbarCnt⟩
  rcases Bool.and_eq_true_iff.mp ha1234 with ⟨ha123, hsinit⟩
  rcases Bool.and_eq_true_iff.mp ha123 with ⟨ha12, hexit⟩
  rcases Bool.and_eq_true_iff.mp ha12 with ⟨ha1, hentry⟩
  rcases Bool.and_eq_true_iff.mp ha1 with ⟨halloc, hlaunch⟩
  refine ⟨?_, ?_, ?_, ?_, ?_, hmbarCnt, safeCtaExecution_of_stateMap hvalid, noUnbackpressuredLoop_of_check hloop⟩
  · cases hs : tmemAllocSites cfg with
    | nil => rw [hs] at halloc; contradiction
    | cons cols rest =>
        cases rest with
        | cons _ _ => rw [hs] at halloc; contradiction
        | nil =>
            rw [hs] at halloc
            rcases Bool.and_eq_true_iff.mp halloc with ⟨hv1, hallDealloc⟩
            rcases Bool.and_eq_true_iff.mp hv1 with ⟨hcols, hnonempty⟩
            refine ⟨cols, tmemValidCols_of_bool hcols, rfl, of_decide_eq_true hnonempty, ?_⟩
            intro d hd
            exact beq_iff_eq.mp (List.all_eq_true.mp hallDealloc d hd)
  · intro aBlk haBlk
    exact not_pathAvoiding_of_certify (List.all_eq_true.mp hlaunch aBlk haBlk)
  · intro iBlk aBlk hpair
    exact not_pathAvoiding_of_certify (List.all_eq_true.mp hentry (iBlk, aBlk) hpair)
  · intro dBlk hdBlk ret hret
    have hd := List.all_eq_true.mp hexit dBlk hdBlk
    exact not_pathAvoiding_of_certify (List.all_eq_true.mp hd ret hret)
  · intro s rBlk hsite
    exact not_pathAvoiding_of_certify (List.all_eq_true.mp hsinit (s, rBlk) hsite)

open Lean Elab Command Meta in
/-- Precomputes `cfgStateMap : StateMap` in compiled `MetaM` bytecode (`addDecl` certificate
    checked in Lean's kernel by `validStateMap` inside `tmem_lifecycle_of_stateMap`). -/
elab "cfg_def_state_map " c:ident : command => do
  Command.liftTermElabM do                                -- Enter Lean's term-elaboration environment
    let ns ← getCurrNamespace                             -- Read current namespace (`CudaOxide.Generated`)
    let cExpr ← Term.elabTerm (← `($c:ident)) (some (mkConst ``ControlFlowCfg)) -- Resolve `controlFlowCfg`
    let cVal ← unsafe evalExpr ControlFlowCfg (mkConst ``ControlFlowCfg) cExpr  -- Evaluate CFG in native code
    let smVal := inferStateMap cVal                       -- Compute 1-summary-per-block `StateMap` in native code
    let stListTy ← Meta.mkAppM ``List #[mkConst ``CtaState]
    let smTy ← Meta.mkAppM ``List #[stListTy]
    addDecl <| Declaration.defnDecl {                     -- Emit `cfgStateMap` constant (verified by `validStateMap` via `rfl`)
      name := ns ++ `cfgStateMap, levelParams := [], type := smTy, value := toExpr smVal,
      hints := ReducibilityHints.abbrev, safety := DefinitionSafety.safe }

end CudaOxide.Proofs.ControlFlow
