/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import Lean
import CudaOxide.Proofs.ControlFlow.Soundness

/-!
# Certified `Split -> Assign -> Meet` CTA Uniformity (`CudaOxide.Proofs.ControlFlow.Uniformity`)

Certifies a single cross-block SSA uniformity set `(uniVars, rfPtrs)` over `MirFunc`
(`MirValidUniSet`, stepping block-local scratch registers sequentially per block) with
minimal certificate footprint so Lean's kernel runs with near-zero time and memory overhead.
-/
namespace CudaOxide.Proofs.ControlFlow
open CudaOxide.Mir.ControlFlow
open CudaOxide.Mir.Program
open CudaOxide.Contracts.ControlFlow

/-- Per-block `Split -> Assign -> Meet` reachability certificate `(mOpt, visU, visT, visF, cantReachU)`. -/
structure SplitCert where
  /-- Optional single reconvergence block `m ≠ u` post-dominating `u`. -/
  mOpt : Option Nat := none
  /-- Closed subgraph of blocks reachable from `u` avoiding `[m]` (disjoint from `returnBlocks`). -/
  visU : List Nat := []
  /-- Closed subgraph of blocks reachable from `thenBlk` avoiding `avoid`. -/
  visT : List Nat := []
  /-- Closed subgraph of blocks reachable from `elseBlk` avoiding `avoid`. -/
  visF : List Nat := []
  /-- Closed subgraph of blocks avoiding `avoid` that cannot reach `u` (refuting `PathAvoiding avoid b u`). -/
  cantReachU : List Nat := []
  deriving Repr, DecidableEq, Lean.ToExpr

/-- Minimal per-block summary (~4 scalars per block instead of 6 lists!). -/
structure BlockUniSummary where
  /-- Outgoing target block indices of `mirToCfg f` at `b`. -/
  succDsts : List Nat := []
  /-- `some (cond, tBlk, fBlk)` when `(f.blockAt b).term = .condBr`, else `none`. -/
  condBrOpt : Option (Nat × Nat × Nat) := none
  /-- Only those `e.val` / `condBr` registers uniform at exit of block `b` that are actually queried! -/
  exitUni : List Nat := []
  /-- Exit race-free `SharedArray` pointers at block `b`. -/
  exitRf : List Nat := []
  /-- Precomputed `SplitCert` when `b` is a divergent `condBr`, else `{}`. -/
  split : SplitCert := {}
  deriving Repr, DecidableEq, Lean.ToExpr

/-- Chunk a flat list into blocks of 16 elements so `(chunks.getD (x / 16) []).getD (x % 16) []`
    runs in `O(N / 16 + 16)` tiny pointer hops instead of `O(N)` struct traversals in WHNF. -/
def chunk16 {α : Type} : Nat → List α → List (List α)
  | 0, _ => []
  | _, [] => []
  | fuel + 1, xs => xs.take 16 :: chunk16 fuel (xs.drop 16)

/-- Lookup index `x` in a 16-way chunked table. -/
def lookupChunk16 {α : Type} (d : α) (chunks : List (List α)) (x : Nat) : α :=
  (chunks.getD (x / 16) []).getD (x % 16) d

/-- Soundness of `lookupChunk16` when `chunks = chunk16 fuel xs` with `xs.length ≤ 16 * fuel`. -/
theorem lookupChunk16_eq_getD {α : Type} (d : α) :
    ∀ (fuel : Nat) (xs : List α) (x : Nat), xs.length ≤ 16 * fuel →
      lookupChunk16 d (chunk16 fuel xs) x = xs.getD x d := by
  intro fuel
  induction fuel with
  | zero =>
      intro xs x hlen
      cases xs with
      | nil => simp [lookupChunk16, chunk16]
      | cons _ _ => simp at hlen
  | succ f ih =>
      intro xs x hlen
      cases xs with
      | nil => simp [lookupChunk16, chunk16]
      | cons hd tl =>
          let xs := hd :: tl
          change (((xs.take 16 :: chunk16 f (xs.drop 16)).getD (x / 16) []).getD (x % 16) d) = xs.getD x d
          by_cases hx : x < 16
          · have hdiv : x / 16 = 0 := Nat.div_eq_of_lt hx
            have hmod : x % 16 = x := Nat.mod_eq_of_lt hx
            rw [hdiv, hmod, List.getD_cons_zero]
            by_cases hlt : x < xs.length
            · have ht : x < (xs.take 16).length := by simp; omega
              rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem ht,
                  List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hlt, List.getElem_take]
            · have ht : (xs.take 16).length ≤ x := by simp; omega
              have hge : xs.length ≤ x := by omega
              rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none ht,
                  List.getD_eq_getElem?_getD, List.getElem?_eq_none hge]
          · obtain ⟨q, hq⟩ : ∃ q, x / 16 = q + 1 := ⟨x / 16 - 1, by omega⟩
            rw [hq, List.getD_cons_succ]
            have hdropLen : (xs.drop 16).length ≤ 16 * f := by simp; omega
            have hih := ih (xs.drop 16) (x - 16) hdropLen
            unfold lookupChunk16 at hih
            have hq2 : (x - 16) / 16 = q := by omega
            have hm2 : (x - 16) % 16 = x % 16 := by omega
            rw [hq2, hm2] at hih
            rw [hih]
            by_cases hlt : x < xs.length
            · have hdlt : x - 16 < (xs.drop 16).length := by simp; omega
              rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hdlt,
                  List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hlt, List.getElem_drop]
              have hadd : 16 + (x - 16) = x := by omega
              simp [hadd]
            · have hdge : (xs.drop 16).length ≤ x - 16 := by simp; omega
              have hge : xs.length ≤ x := by omega
              rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none hdge,
                  List.getD_eq_getElem?_getD, List.getElem?_eq_none hge]

/-- Fast 16-way chunked check that `vis` is closed under `succChunks` outside `avoid`. -/
def isClosedSuccs (succChunks : List (List (List Nat))) (avoid vis : List Nat) : Bool :=
  vis.all fun x =>
    !avoid.contains x &&
    ((lookupChunk16 [] succChunks x).all fun w => avoid.contains w || vis.contains w)

/-- Extract `(succDsts, condBrOpt)` per block in one linear pass over `f.blocks`. -/
def extractTermSkeleton (f : MirFunc) : List (List Nat × Option (Nat × Nat × Nat)) :=
  f.blocks.map fun blk =>
    match blk.term with
    | .ret | .unreachable => ([], none)
    | .goto dst _ => ([dst], none)
    | .condBr c tBlk _ fBlk _ => ([tBlk, fBlk], some (c, tBlk, fBlk))

/-- Generic helper: `(xs.map g).getD i (g d) = g (xs.getD i d)`. -/
theorem getD_map_eq {α β : Type} (xs : List α) (g : α → β) (d : α) (i : Nat) :
    (xs.map g).getD i (g d) = g (xs.getD i d) := by
  by_cases hi : i < xs.length
  · have hm : i < (xs.map g).length := by simp [hi]
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hm,
        List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi]
    simp
  · have hm : (xs.map g).length ≤ i := by simp; omega
    have hx : xs.length ≤ i := by omega
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none hm,
        List.getD_eq_getElem?_getD, List.getElem?_eq_none hx]
    rfl

/-- Helper: `(extractTermSkeleton f).getD x ([], none)` matches `f.blockAt x`. -/
theorem extractTermSkeleton_getD (f : MirFunc) (x : Nat) :
    (extractTermSkeleton f).getD x ([], none) =
      (match (f.blockAt x).term with
       | .ret | .unreachable => ([], none)
       | .goto dst _ => ([dst], none)
       | .condBr c tBlk _ fBlk _ => ([tBlk, fBlk], some (c, tBlk, fBlk))) := by
  unfold extractTermSkeleton MirFunc.blockAt
  exact getD_map_eq f.blocks _ { term := .unreachable } x

/-- Helper: `((mirToCfg f).blockAt x).succs.map Prod.snd` matches `(extractTermSkeleton f).getD x`. -/
theorem mirToCfg_succs_snd (f : MirFunc) (x : Nat) :
    (((mirToCfg f).blockAt x).succs.map Prod.snd) = ((extractTermSkeleton f).getD x ([], none)).1 := by
  rw [extractTermSkeleton_getD f x]
  unfold mirToCfg ControlFlowCfg.blockAt MirFunc.blockAt
  by_cases hx : x < f.blocks.length
  · simp only [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_eq_getElem hx, Option.map_some, Option.getD_some]
    cases f.blocks[x].term <;> rfl
  · have hx' : f.blocks.length ≤ x := by omega
    simp only [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_eq_none hx', Option.map_none, Option.getD_none]
    rfl

/-- Bridge `isClosedSuccs` + `!vis.contains dst` to `validUnreachWitness (mirToCfg f)`. -/
theorem validUnreachWitness_of_succs {f : MirFunc} {sums : List BlockUniSummary}
    {succChunks : List (List (List Nat))} {avoid vis : List Nat} {dst : Nat}
    (hsk : sums.map (fun s => (s.succDsts, s.condBrOpt)) = extractTermSkeleton f)
    (hch : chunk16 sums.length (sums.map BlockUniSummary.succDsts) = succChunks)
    (hcl : isClosedSuccs succChunks avoid vis = true)
    (hndst : (!vis.contains dst) = true) :
    validUnreachWitness (mirToCfg f) avoid dst vis = true := by
  subst hch
  unfold validUnreachWitness
  refine Bool.and_eq_true_iff.mpr ⟨?_, List.all_eq_true.mpr ?_⟩
  · simp only [Bool.not_eq_true'] at hndst ⊢
    exact decide_eq_false (fun hm => by rw [List.contains_iff_mem.mpr hm] at hndst; contradiction)
  · intro u hu
    have huOk := List.all_eq_true.mp hcl u hu
    rcases Bool.and_eq_true_iff.mp huOk with ⟨hna, hsuccs⟩
    have hmapLen : (sums.map BlockUniSummary.succDsts).length ≤ 16 * sums.length := by simp; omega
    rw [lookupChunk16_eq_getD [] sums.length _ u hmapLen,
        getD_map_eq sums BlockUniSummary.succDsts {} u] at hsuccs
    refine Bool.and_eq_true_iff.mpr ⟨?_, List.all_eq_true.mpr ?_⟩
    · simp only [Bool.not_eq_true'] at hna ⊢
      exact decide_eq_false (fun hm => by rw [List.contains_iff_mem.mpr hm] at hna; contradiction)
    · intro (bc, w) hcw
      have hwMem : w ∈ ((mirToCfg f).blockAt u).succs.map Prod.snd :=
        List.mem_map.mpr ⟨(bc, w), hcw, rfl⟩
      rw [mirToCfg_succs_snd f u, ← hsk,
          getD_map_eq sums (fun s => (s.succDsts, s.condBrOpt)) {} u] at hwMem
      have hwOk := List.all_eq_true.mp hsuccs w hwMem
      rcases Bool.or_eq_true_iff.mp hwOk with hav | hvi
      · exact Bool.or_eq_true_iff.mpr (Or.inl (decide_eq_true (List.contains_iff_mem.mp hav)))
      · exact Bool.or_eq_true_iff.mpr (Or.inr (decide_eq_true (List.contains_iff_mem.mp hvi)))

/-- Soundness of `isClosedSuccs` refuting `PathAvoiding (mirToCfg f) avoid src dst`. -/
theorem not_pathAvoiding_of_succs {f : MirFunc} {sums : List BlockUniSummary}
    {succChunks : List (List (List Nat))} {avoid vis : List Nat} {src dst : Nat}
    (hlen : sums.length = f.blocks.length)
    (hsk : sums.map (fun s => (s.succDsts, s.condBrOpt)) = extractTermSkeleton f)
    (hch : chunk16 sums.length (sums.map BlockUniSummary.succDsts) = succChunks)
    (hcl : isClosedSuccs succChunks avoid vis = true)
    (hsrc : (avoid.contains src || Nat.ble sums.length src || vis.contains src) = true)
    (hndst : (!vis.contains dst) = true) :
    ¬ PathAvoiding (mirToCfg f) avoid src dst := by
  intro hp
  have hmLen : (mirToCfg f).blocks.length = sums.length := by simp [mirToCfg, hlen]
  rcases Bool.or_eq_true_iff.mp hsrc with hleft | hmem
  · rcases Bool.or_eq_true_iff.mp hleft with hav | hge
    · have haMem : src ∈ avoid := List.contains_iff_mem.mp hav
      cases hp with
      | refl _ ha => exact ha haMem
      | step _ ha _ _ => exact ha haMem
    · have hge' : sums.length ≤ src := Nat.ble_eq.mp hge
      cases hp with
      | refl hb _ => omega
      | step hb _ _ _ => omega
  · exact pathAvoiding_false_of_witness (validUnreachWitness_of_succs hsk hch hcl hndst) hp (List.contains_iff_mem.mp hmem)

/-- Fast `Bool`-only `O(1)` test refuting `armReachesEdge` via `visArm`. -/
def checkArmDisjointCert (u armIdx : Nat) (visArm : List Nat) (e : Nat × Nat × Nat × Nat × Nat × Nat) : Bool :=
  !(e.2.2.2.1 == u && e.2.2.2.2.1 == armIdx) && !visArm.contains e.2.2.2.1

/-- Soundness of `checkArmDisjointCert`. -/
theorem not_armReachesEdge_of_succs {f : MirFunc} {sums : List BlockUniSummary}
    {succChunks : List (List (List Nat))} {u armBlk armIdx : Nat}
    {avoid visArm : List Nat} {e : Nat × Nat × Nat × Nat × Nat × Nat}
    (hlen : sums.length = f.blocks.length)
    (hsk : sums.map (fun s => (s.succDsts, s.condBrOpt)) = extractTermSkeleton f)
    (hch : chunk16 sums.length (sums.map BlockUniSummary.succDsts) = succChunks)
    (hcl : isClosedSuccs succChunks avoid visArm = true)
    (hsrc : (avoid.contains armBlk || Nat.ble sums.length armBlk || visArm.contains armBlk) = true)
    (hchk : checkArmDisjointCert u armIdx visArm e = true) :
    ¬ armReachesEdge (mirToCfg f) u armBlk armIdx avoid e := by
  unfold checkArmDisjointCert at hchk
  rcases Bool.and_eq_true_iff.mp hchk with ⟨hdir, hnin⟩
  intro harm
  rcases harm with ⟨he1, he2⟩ | hp
  · have hb : (e.2.2.2.1 == u && e.2.2.2.2.1 == armIdx) = true := by simp [he1, he2]
    rw [hb] at hdir; contradiction
  · exact not_pathAvoiding_of_succs hlen hsk hch hcl hsrc hnin hp

/-- Validate one `SplitCert` using 16-way chunked `succChunks` and `rets` directly. -/
def validSplitCertFast (numBlocks : Nat) (succChunks : List (List (List Nat)))
    (rets : List Nat) (u tBlk fBlk : Nat) (sc : SplitCert) : Bool :=
  let avoid := match sc.mOpt with | some m => [m] | none => []
  let pdomOk := match sc.mOpt with
    | none => true
    | some m =>
        (m != u) &&
        isClosedSuccs succChunks [m] sc.visU &&
        (([m] : List Nat).contains u || Nat.ble numBlocks u || sc.visU.contains u) &&
        rets.all fun ret => !sc.visU.contains ret
  let tOk :=
    isClosedSuccs succChunks avoid sc.visT &&
    (avoid.contains tBlk || Nat.ble numBlocks tBlk || sc.visT.contains tBlk)
  let fOk :=
    isClosedSuccs succChunks avoid sc.visF &&
    (avoid.contains fBlk || Nat.ble numBlocks fBlk || sc.visF.contains fBlk)
  let cycOk := isClosedSuccs succChunks avoid sc.cantReachU && !sc.cantReachU.contains u
  pdomOk && tOk && fOk && cycOk

/-- Check `(5a)` and `(5b)` over pre-filtered `uniEdges`. -/
def checkSplitUniEdges (u : Nat) (sc : SplitCert)
    (uniEdges allEdges : List (Nat × Nat × Nat × Nat × Nat × Nat)) : Bool :=
  let joinOk :=
    uniEdges.all fun e0 =>
      checkArmDisjointCert u 0 sc.visT e0 ||
      (allEdges.all fun e1 =>
        !(e0.1 == e1.1 && e0.2.1 == e1.2.1) ||
        (e0.2.2.2.2.2 == e1.2.2.2.2.2) ||
        checkArmDisjointCert u 1 sc.visF e1)
  let loopOk :=
    uniEdges.all fun e =>
      sc.cantReachU.contains e.1 ||
      (e.2.2.2.2.2 == e.2.2.1) ||
      (checkArmDisjointCert u 0 sc.visT e && checkArmDisjointCert u 1 sc.visF e)
  joinOk && loopOk

/-- Linear scan checking `validSplitCertFast` & `checkSplitUniEdges` without `sums.getD u`. -/
def checkSplitsRec (numBlocks : Nat) (succChunks : List (List (List Nat))) (rets : List Nat)
    (uniEdges allEdges : List (Nat × Nat × Nat × Nat × Nat × Nat)) :
    Nat → List BlockUniSummary → Bool
  | _, [] => true
  | u, su :: rest =>
      (match su.condBrOpt with
       | some (cond, tBlk, fBlk) =>
           su.exitUni.contains cond ||
           (validSplitCertFast numBlocks succChunks rets u tBlk fBlk su.split &&
            checkSplitUniEdges u su.split uniEdges allEdges)
       | none => true) &&
      checkSplitsRec numBlocks succChunks rets uniEdges allEdges (u + 1) rest

/-- Extract block `u` from `checkSplitsRec k ss`. -/
theorem checkSplitsRec_at {numBlocks : Nat} {succChunks : List (List (List Nat))} {rets : List Nat}
    {uniEdges allEdges : List (Nat × Nat × Nat × Nat × Nat × Nat)} :
    ∀ (k : Nat) (ss : List BlockUniSummary) (i : Nat), i < ss.length →
      checkSplitsRec numBlocks succChunks rets uniEdges allEdges k ss = true →
      (let su := ss.getD i {}
       match su.condBrOpt with
       | some (cond, tBlk, fBlk) =>
           su.exitUni.contains cond ||
           (validSplitCertFast numBlocks succChunks rets (k + i) tBlk fBlk su.split &&
            checkSplitUniEdges (k + i) su.split uniEdges allEdges)
       | none => true) = true := by
  intro k ss
  induction ss generalizing k with
  | nil => intro i hi; nomatch hi
  | cons hd tl ih =>
      intro i hi hrec
      simp only [checkSplitsRec] at hrec
      rcases Bool.and_eq_true_iff.mp hrec with ⟨hcur, hrest⟩
      cases i with
      | zero => simpa using hcur
      | succ j =>
          have hj : j < tl.length := Nat.lt_of_succ_lt_succ hi
          have hih := ih (k + 1) j hj hrest
          have hk : k + 1 + j = k + (j + 1) := by omega
          rw [hk] at hih
          simpa using hih

/-- Linear recursion verifying `checkExecUniStmts` on every block against `(su.exitUni, su.exitRf)` (subset!). -/
def verifyBlockExits (f : MirFunc) (uniVars rfPtrs : List Nat) :
    List BasicBlock → List BlockUniSummary → Bool
  | [], [] => true
  | blk :: bs, su :: ss =>
      let (ok, exU, exR) := checkExecUniStmts f uniVars rfPtrs uniVars rfPtrs blk.stmts
      ok &&
      (su.exitUni.all exU.contains) &&
      (match su.condBrOpt with | some (c, _, _) => !exU.contains c || su.exitUni.contains c | none => true) &&
      (su.exitRf.all exR.contains) &&
      verifyBlockExits f uniVars rfPtrs bs ss
  | _, _ => false

/-- Pointwise extraction from `verifyBlockExits`. -/
theorem verifyBlockExits_at {f : MirFunc} {uniVars rfPtrs : List Nat} {sums : List BlockUniSummary}
    (hv : verifyBlockExits f uniVars rfPtrs f.blocks sums = true) :
    sums.length = f.blocks.length ∧
    ∀ b < f.blocks.length,
      let (ok, exU, exR) := checkExecUniStmts f uniVars rfPtrs uniVars rfPtrs (f.blockAt b).stmts
      ok = true ∧
      (∀ x ∈ (sums.getD b {}).exitUni, x ∈ exU) ∧
      (∀ c t fb, (sums.getD b {}).condBrOpt = some (c, t, fb) → c ∈ exU → c ∈ (sums.getD b {}).exitUni) ∧
      (∀ x ∈ (sums.getD b {}).exitRf, x ∈ exR) := by
  unfold MirFunc.blockAt
  suffices ∀ (bs : List BasicBlock) (ss : List BlockUniSummary),
      verifyBlockExits f uniVars rfPtrs bs ss = true →
      ss.length = bs.length ∧
      ∀ b < bs.length,
        let (ok, exU, exR) := checkExecUniStmts f uniVars rfPtrs uniVars rfPtrs (bs.getD b { term := .unreachable }).stmts
        ok = true ∧
        (∀ x ∈ (ss.getD b {}).exitUni, x ∈ exU) ∧
        (∀ c t fb, (ss.getD b {}).condBrOpt = some (c, t, fb) → c ∈ exU → c ∈ (ss.getD b {}).exitUni) ∧
        (∀ x ∈ (ss.getD b {}).exitRf, x ∈ exR) from
    this f.blocks sums hv
  intro bs
  induction bs with
  | nil =>
      intro ss h
      cases ss with
      | nil => refine ⟨rfl, fun b hb => nomatch hb⟩
      | cons _ _ => simp [verifyBlockExits] at h
  | cons blk bs ih =>
      intro ss h
      cases ss with
      | nil => simp [verifyBlockExits] at h
      | cons su ss =>
          simp only [verifyBlockExits] at h
          rcases Bool.and_eq_true_iff.mp h with ⟨h0_3, htail⟩
          rcases Bool.and_eq_true_iff.mp h0_3 with ⟨h0_2, hallR⟩
          rcases Bool.and_eq_true_iff.mp h0_2 with ⟨h0_1, hcbImp⟩
          rcases Bool.and_eq_true_iff.mp h0_1 with ⟨hok, hallU⟩
          have ⟨hlen, hrest⟩ := ih ss htail
          refine ⟨by simp [hlen], ?_⟩
          intro b hb
          cases b with
          | zero =>
              simp only [List.getD_cons_zero]
              refine ⟨hok, ?_, ?_, ?_⟩
              · intro x hx; exact List.contains_iff_mem.mp (List.all_eq_true.mp hallU x hx)
              · intro c t fb hcb hex
                rw [hcb] at hcbImp; dsimp only at hcbImp
                rw [List.contains_iff_mem.mpr hex] at hcbImp
                simp only [Bool.not_true, Bool.false_or] at hcbImp
                exact List.contains_iff_mem.mp hcbImp
              · intro x hx; exact List.contains_iff_mem.mp (List.all_eq_true.mp hallR x hx)
          | succ j =>
              have hj : j < bs.length := Nat.lt_of_succ_lt_succ hb
              simp only [List.getD_cons_succ]
              exact hrest j hj

/-- Helper: every edge in `collectAllArgEdges f` has `e.2.2.2.1 < f.blocks.length`. -/
theorem collectAllArgEdges_src_lt {f : MirFunc} {e : Nat × Nat × Nat × Nat × Nat × Nat}
    (he : e ∈ collectAllArgEdges f) : e.2.2.2.1 < f.blocks.length := by
  unfold collectAllArgEdges at he
  let dstArgLists := f.blocks.map fun blk => blk.args.map BlockArg.varId
  let pushArgs := fun (dst src arm : Nat) (args : List Nat) (acc : List (Nat × Nat × Nat × Nat × Nat × Nat)) =>
    if dst == 0 then acc
    else
      let dArgs := dstArgLists.getD dst []
      (List.range (min dArgs.length args.length)).foldr (fun k a =>
        match dArgs[k]?, args[k]? with
        | some argId, some v => (dst, k, argId, src, arm, v) :: a
        | _, _ => a) acc
  have hpush : ∀ dst src arm args acc,
      e ∈ pushArgs dst src arm args acc → e.2.2.2.1 = src ∨ e ∈ acc := by
    intro dst src arm args acc hm
    dsimp only [pushArgs] at hm
    split at hm
    · exact Or.inr hm
    · suffices ∀ ks : List Nat,
          e ∈ ks.foldr (fun k a =>
            match (dstArgLists.getD dst [])[k]?, args[k]? with
            | some argId, some v => (dst, k, argId, src, arm, v) :: a
            | _, _ => a) acc → e.2.2.2.1 = src ∨ e ∈ acc from
        this _ hm
      intro ks hks
      induction ks with
      | nil => exact Or.inr hks
      | cons k ks ihk =>
          dsimp only [List.foldr] at hks
          split at hks
          · cases hks with
            | head => exact Or.inl rfl
            | tail _ hr => exact ihk hr
          · exact ihk hks
  suffices ∀ (p : Nat) (bs : List BasicBlock),
      e ∈ collectAllArgEdges.go pushArgs p bs → p ≤ e.2.2.2.1 ∧ e.2.2.2.1 < p + bs.length by
    have hres := (this 0 f.blocks he).2; omega
  intro p bs
  induction bs generalizing p with
  | nil => intro hm; nomatch hm
  | cons blk rest ih =>
      intro hm
      dsimp only [collectAllArgEdges.go] at hm
      cases ht : blk.term <;> rw [ht] at hm
      · have ⟨_, h2⟩ := ih (p + 1) hm; simp only [List.length_cons]; omega
      · have ⟨_, h2⟩ := ih (p + 1) hm; simp only [List.length_cons]; omega
      · rcases hpush _ p 0 _ _ hm with hsrc | hrest
        · simp [hsrc]
        · have ⟨_, h2⟩ := ih (p + 1) hrest; simp only [List.length_cons]; omega
      · rcases hpush _ p 0 _ _ hm with hsrc | hmid
        · simp [hsrc]
        · rcases hpush _ p 1 _ _ hmid with hsrc | hrest
          · simp [hsrc]
          · have ⟨_, h2⟩ := ih (p + 1) hrest; simp only [List.length_cons]; omega

/-- Ultra-compact kernel validator for `MirValidUniSet f uniVars rfPtrs uniConds` using 16-way `succChunks`. -/
def checkMirValidUniSetFast (f : MirFunc) (uniVars rfPtrs : List Nat)
    (sums : List BlockUniSummary) (succChunks : List (List (List Nat))) (rets : List Nat)
    (allEdges : List (Nat × Nat × Nat × Nat × Nat × Nat))
    (uniConds : List Nat) : Bool :=
  let uniEdges := allEdges.filter fun e => uniVars.contains e.2.2.1
  (((f.blockAt 0).args.map BlockArg.varId).all fun v => !rfPtrs.contains v) &&
  verifyBlockExits f uniVars rfPtrs f.blocks sums &&
  (allEdges.all fun e =>
    let ss := sums.getD e.2.2.2.1 {}
    (!uniVars.contains e.2.2.1 || ss.exitUni.contains e.2.2.2.2.2) &&
    (!rfPtrs.contains e.2.2.1 || ss.exitRf.contains e.2.2.2.2.2)) &&
  (sums.all fun su =>
    match su.condBrOpt with
    | some (cond, _, _) => !uniConds.contains cond || su.exitUni.contains cond
    | none => true) &&
  checkSplitsRec sums.length succChunks rets uniEdges allEdges 0 sums

/-- Soundness of `checkMirValidUniSetFast`: establishes `MirValidUniSet`. -/
theorem mirValidUniSet_of_check {f : MirFunc} {uniVars rfPtrs : List Nat}
    {sums : List BlockUniSummary} {succChunks : List (List (List Nat))} {rets : List Nat}
    {allEdges : List (Nat × Nat × Nat × Nat × Nat × Nat)} {uniConds : List Nat}
    (hsk : sums.map (fun s => (s.succDsts, s.condBrOpt)) = extractTermSkeleton f)
    (hch : chunk16 sums.length (sums.map BlockUniSummary.succDsts) = succChunks)
    (hrets : returnBlocks (mirToCfg f) = rets)
    (hae : collectAllArgEdges f = allEdges)
    (hchk : checkMirValidUniSetFast f uniVars rfPtrs sums succChunks rets allEdges uniConds = true) :
    MirValidUniSet f uniVars rfPtrs uniConds := by
  subst hrets; subst hae
  unfold checkMirValidUniSetFast at hchk
  rcases Bool.and_eq_true_iff.mp hchk with ⟨h0_3, hsplits⟩
  rcases Bool.and_eq_true_iff.mp h0_3 with ⟨h0_2, hconds⟩
  rcases Bool.and_eq_true_iff.mp h0_2 with ⟨h0_1, hedges⟩
  rcases Bool.and_eq_true_iff.mp h0_1 with ⟨hrf0, hvex⟩
  have ⟨hlen, hexAt⟩ := verifyBlockExits_at hvex
  have hskAt : ∀ i,
      ((sums.getD i {}).succDsts, (sums.getD i {}).condBrOpt) =
        (extractTermSkeleton f).getD i ([], none) := by
    intro i; rw [← hsk]; exact (getD_map_eq sums (fun s => (s.succDsts, s.condBrOpt)) {} i).symm
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · intro v hv hmem
    have hnv := List.all_eq_true.mp hrf0 v hv
    rw [List.contains_iff_mem.mpr hmem] at hnv; contradiction
  · intro b hb
    exact (hexAt b hb).1
  · intro e he
    have heOk := List.all_eq_true.mp hedges e he
    have hsrcLt : e.2.2.2.1 < f.blocks.length := collectAllArgEdges_src_lt he
    have ⟨_, hsubU, _, hsubR⟩ := hexAt e.2.2.2.1 hsrcLt
    rcases Bool.and_eq_true_iff.mp heOk with ⟨huni, hrf⟩
    refine ⟨?_, ?_⟩
    · intro ha
      rw [List.contains_iff_mem.mpr ha] at huni; simp only [Bool.not_true, Bool.false_or] at huni
      exact hsubU _ (List.contains_iff_mem.mp huni)
    · intro ha
      rw [List.contains_iff_mem.mpr ha] at hrf; simp only [Bool.not_true, Bool.false_or] at hrf
      exact hsubR _ (List.contains_iff_mem.mp hrf)
  · intro mb hmb cond tBlk fBlk tArgs fArgs hterm hmem
    have hmbS : mb < sums.length := by omega
    have hsMem : sums.getD mb {} ∈ sums := by
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hmbS]; exact List.getElem_mem _
    have hmOk := List.all_eq_true.mp hconds (sums.getD mb {}) hsMem
    have hcb : (sums.getD mb {}).condBrOpt = some (cond, tBlk, fBlk) := by
      have h2 := congrArg Prod.snd (hskAt mb)
      dsimp only at h2; rw [h2, extractTermSkeleton_getD f mb]; rw [hterm]
    rw [hcb] at hmOk; dsimp only at hmOk
    rw [List.contains_iff_mem.mpr hmem] at hmOk
    simp only [Bool.not_true, Bool.false_or] at hmOk
    exact (hexAt mb hmb).2.1 _ (List.contains_iff_mem.mp hmOk)
  · intro u hu cond tBlk fBlk tArgs fArgs hterm hnotUni
    have huS : u < sums.length := by omega
    have huOk := checkSplitsRec_at 0 sums u huS hsplits
    simp only [Nat.zero_add] at huOk
    have hcb : (sums.getD u {}).condBrOpt = some (cond, tBlk, fBlk) := by
      have h2 := congrArg Prod.snd (hskAt u)
      dsimp only at h2; rw [h2, extractTermSkeleton_getD f u]; rw [hterm]
    rw [hcb] at huOk; dsimp only at huOk
    have hnc : (sums.getD u {}).exitUni.contains cond = false := by
      cases hc : (sums.getD u {}).exitUni.contains cond
      · rfl
      · exact False.elim (hnotUni ((hexAt u hu).2.1 _ (List.contains_iff_mem.mp hc)))
    rw [hnc] at huOk; simp only [Bool.false_or] at huOk
    rcases Bool.and_eq_true_iff.mp huOk with ⟨hvsc, hse⟩
    unfold validSplitCertFast at hvsc
    rcases Bool.and_eq_true_iff.mp hvsc with ⟨hpTF, hcycFast⟩
    rcases Bool.and_eq_true_iff.mp hpTF with ⟨hpT, hfOk⟩
    rcases Bool.and_eq_true_iff.mp hpT with ⟨hpdom, htOk⟩
    rcases Bool.and_eq_true_iff.mp htOk with ⟨hclT, hsrcT⟩
    rcases Bool.and_eq_true_iff.mp hfOk with ⟨hclF, hsrcF⟩
    rcases Bool.and_eq_true_iff.mp hcycFast with ⟨hclCyc, hnuCyc⟩
    have hcycWit := validUnreachWitness_of_succs hsk hch hclCyc hnuCyc
    unfold checkSplitUniEdges at hse
    rcases Bool.and_eq_true_iff.mp hse with ⟨hjoin, hloop⟩
    refine ⟨(sums.getD u {}).split.mOpt, ?_, ?_⟩
    · intro m hm
      rw [hm] at hpdom; dsimp only at hpdom
      rcases Bool.and_eq_true_iff.mp hpdom with ⟨hp12, hallRet⟩
      rcases Bool.and_eq_true_iff.mp hp12 with ⟨hp1, hsrcU⟩
      rcases Bool.and_eq_true_iff.mp hp1 with ⟨hne, hclU⟩
      refine ⟨bne_iff_ne.mp hne, ?_⟩
      intro ret hret
      exact not_pathAvoiding_of_succs hlen hsk hch hclU hsrcU (List.all_eq_true.mp hallRet ret hret)
    · intro avoid
      refine ⟨?_, ?_⟩
      · intro e0 e1 he0 he1 hbEq hkEq hargUni hr0 hr1
        have he0U : e0 ∈ (collectAllArgEdges f).filter (fun e => uniVars.contains e.2.2.1) :=
          List.mem_filter.mpr ⟨he0, List.contains_iff_mem.mpr hargUni⟩
        have h0 := List.all_eq_true.mp hjoin e0 he0U
        rcases Bool.or_eq_true_iff.mp h0 with hc0 | hall1
        · exact False.elim (not_armReachesEdge_of_succs hlen hsk hch hclT hsrcT hc0 hr0)
        · have h1 := List.all_eq_true.mp hall1 e1 he1
          have hbk : (e0.1 == e1.1 && e0.2.1 == e1.2.1) = true := by simp [hbEq, hkEq]
          rw [hbk] at h1; simp only [Bool.not_true, Bool.false_or] at h1
          rcases Bool.or_eq_true_iff.mp h1 with heq | hc1
          · exact beq_iff_eq.mp heq
          · exact False.elim (not_armReachesEdge_of_succs hlen hsk hch hclF hsrcF hc1 hr1)
      · intro e he hargUni hcyc hreach
        have heU : e ∈ (collectAllArgEdges f).filter (fun e => uniVars.contains e.2.2.1) :=
          List.mem_filter.mpr ⟨he, List.contains_iff_mem.mpr hargUni⟩
        have heOk := List.all_eq_true.mp hloop e heU
        rcases Bool.or_eq_true_iff.mp heOk with hleft | hboth
        · rcases Bool.or_eq_true_iff.mp hleft with hcant | heq
          · exact False.elim (pathAvoiding_false_of_witness hcycWit hcyc (List.contains_iff_mem.mp hcant))
          · exact beq_iff_eq.mp heq
        · rcases Bool.and_eq_true_iff.mp hboth with ⟨hc0, hc1⟩
          rcases hreach with hr0 | hr1
          · exact False.elim (not_armReachesEdge_of_succs hlen hsk hch hclT hsrcT hc0 hr0)
          · exact False.elim (not_armReachesEdge_of_succs hlen hsk hch hclF hsrcF hc1 hr1)

/-- Reusable soundness theorem combining separate `MirValidUniSet` and `UniformBarriersCfg` proofs
    (allowing Lean 4's parallel theorem checker to verify them concurrently in separate WHNF arenas). -/
theorem uniform_barriers_of_certs {f : MirFunc} {cfg cfgRefined : ControlFlowCfg}
    {uniVars rfPtrs : List Nat}
    {sums : List BlockUniSummary} {succChunks : List (List (List Nat))} {rets : List Nat}
    {allEdges : List (Nat × Nat × Nat × Nat × Nat × Nat)}
    {uniConds : List Nat} {inv : StateMap}
    (hsk : sums.map (fun s => (s.succDsts, s.condBrOpt)) = extractTermSkeleton f)
    (hch : chunk16 sums.length (sums.map BlockUniSummary.succDsts) = succChunks)
    (hrets : returnBlocks (mirToCfg f) = rets)
    (href : refineCfgWithUni cfg uniConds = cfgRefined)
    (hae : collectAllArgEdges f = allEdges)
    (huni : checkMirValidUniSetFast f uniVars rfPtrs sums succChunks rets allEdges uniConds = true)
    (hcfg : checkUniformBarriersWith cfgRefined inv = true) :
    UniformBarriers f cfg := by
  have hbar := uniform_barriers_cfg_of_stateMap hcfg
  rw [← href] at hbar
  exact ⟨uniVars, rfPtrs, uniConds, mirValidUniSet_of_check hsk hch hrets hae huni, hbar⟩

/-- Single-theorem wrapper retained for compact negative tests. -/
def checkUniformBarriersMir (f : MirFunc) (uniVars rfPtrs : List Nat)
    (sums : List BlockUniSummary) (succChunks : List (List (List Nat))) (rets : List Nat)
    (cfgRefined : ControlFlowCfg)
    (allEdges : List (Nat × Nat × Nat × Nat × Nat × Nat))
    (uniConds : List Nat) (inv : StateMap) : Bool :=
  checkMirValidUniSetFast f uniVars rfPtrs sums succChunks rets allEdges uniConds &&
  checkUniformBarriersWith cfgRefined inv

/-- Reusable soundness theorem for `UniformBarriers f cfg`. -/
theorem uniform_barriers_of_stateMap {f : MirFunc} {cfg cfgRefined : ControlFlowCfg}
    {uniVars rfPtrs : List Nat}
    {sums : List BlockUniSummary} {succChunks : List (List (List Nat))} {rets : List Nat}
    {allEdges : List (Nat × Nat × Nat × Nat × Nat × Nat)}
    {uniConds : List Nat} {inv : StateMap}
    (hsk : sums.map (fun s => (s.succDsts, s.condBrOpt)) = extractTermSkeleton f)
    (hch : chunk16 sums.length (sums.map BlockUniSummary.succDsts) = succChunks)
    (hrets : returnBlocks (mirToCfg f) = rets)
    (href : refineCfgWithUni cfg uniConds = cfgRefined)
    (hae : collectAllArgEdges f = allEdges)
    (hcheck : checkUniformBarriersMir f uniVars rfPtrs sums succChunks rets cfgRefined allEdges uniConds inv = true) :
    UniformBarriers f cfg := by
  unfold checkUniformBarriersMir at hcheck
  rcases Bool.and_eq_true_iff.mp hcheck with ⟨huni, hcfg⟩
  exact uniform_barriers_of_certs hsk hch hrets href hae huni hcfg

/-- Variables referenced inside a single `Rhs` expression. -/
def uniRhsVars : Rhs → List Nat
  | .copy s | .notBool s | .intCast _ s | .load s | .cvtaShared s | .loadTmemBase s _ | .atomicOp _ _ s _ _ | .ctaUint1 _ _ s | .ctaUnary s => [s]
  | .binOp _ _ l r | .checkedOvf _ _ l r | .cmpOp _ l r | .ptrOffset l r | .ctaBin l r => [l, r]
  | .mbarTryWait _ sv _ => [sv]
  | _ => []

/-- Collect variables read before being locally assigned in `stmts` + `term` (the cross-block variables!). -/
def collectCrossBlockVars (f : MirFunc) : List Nat :=
  f.blocks.foldl (fun acc blk =>
    let acc1 := blk.args.foldl (fun a ba => if a.contains ba.varId then a else ba.varId :: a) acc
    let (defs, used) := blk.stmts.foldl (fun (ds, us) st =>
      match st with
      | .assign dst rhs =>
          let us' := (uniRhsVars rhs).foldl (fun a v => if ds.contains v || a.contains v then a else v :: a) us
          (dst :: ds, us')
      | _ => (ds, us)) ([], [])
    let termVars := match blk.term with
      | .goto _ args => args
      | .condBr c _ ta _ fa => c :: (ta ++ fa)
      | .ret | .unreachable => []
    termVars.foldl (fun a v => if defs.contains v || a.contains v then a else v :: a)
      (used.foldl (fun a v => if a.contains v then a else v :: a) acc1)) []

/-- Compute strict post-dominator candidate lists for all blocks via fast reverse fixed-point intersection. -/
def computePostDomLists (mcfg : ControlFlowCfg) (rets indices : List Nat) : List (List Nat) :=
  let succsTbl := indices.map fun b => ((mcfg.blockAt b).succs.map Prod.snd)
  let canReachRet := indices.filter fun b => rets.any fun r => canReachAvoiding mcfg b r []
  let initPdom := indices.map fun b =>
    if rets.contains b then [b]
    else if canReachRet.contains b then canReachRet
    else [b]
  let rec loop : Nat → List (List Nat) → List (List Nat)
    | 0, cur => cur
    | fuel + 1, cur =>
        let next := indices.map fun b =>
          if rets.contains b || !canReachRet.contains b then cur.getD b []
          else
            let liveSuccs := (succsTbl.getD b []).filter canReachRet.contains
            match liveSuccs with
            | [] => [b]
            | s0 :: rest =>
                let inter := rest.foldl (fun acc s =>
                  let ps := cur.getD s []
                  acc.filter ps.contains) (cur.getD s0 [])
                if inter.contains b then inter else b :: inter
        let sumLen := fun (xs : List (List Nat)) => xs.foldl (fun s l => s + l.length) 0
        if sumLen next == sumLen cur then cur else loop fuel next
  loop (indices.length + 2) initPdom

/-- Compute one `SplitCert` on demand for block `u`. -/
def inferSplitCertFor (f : MirFunc) (mcfg : ControlFlowCfg) (pdomLists : List (List Nat))
    (rets indices : List Nat)
    (allEdges : List (Nat × Nat × Nat × Nat × Nat × Nat)) (u : Nat) : SplitCert :=
  match (f.blockAt u).term with
  | .condBr _ tBlk _ fBlk _ =>
      if (f.blockAt fBlk).term == .unreachable && tBlk != u then
        let fList := if fBlk == tBlk || fBlk ∈ rets then [] else [fBlk]
        let visU := if fBlk == u || fList.isEmpty then [u] else [u, fBlk]
        if fList.isEmpty then
          { mOpt := none }
        else
          { mOpt := some tBlk, visU := visU, visT := [], visF := fList, cantReachU := [] }
      else if (f.blockAt tBlk).term == .unreachable && fBlk != u then
        let tList := if tBlk == fBlk || tBlk ∈ rets then [] else [tBlk]
        let visU := if tBlk == u || tList.isEmpty then [u] else [u, tBlk]
        if tList.isEmpty then
          { mOpt := none }
        else
          { mOpt := some fBlk, visU := visU, visT := tList, visF := [], cantReachU := [] }
      else
      let cands := ((pdomLists.getD u []).filter (· != u)).filter fun m =>
        rets.all fun ret => !(reachableNodesAvoiding mcfg u [m]).contains ret
      let mOpt := match cands.find? (fun m => canReachAvoiding mcfg u m (cands.filter (· != m))) with
        | some ipdom => some ipdom
        | none => cands.head?
      let avoid := match mOpt with | some m => [m] | none => []
      let visU := match mOpt with
        | some m => reachableNodesAvoiding mcfg u [m]
        | none => []
      let visT := reachableNodesAvoiding mcfg tBlk avoid
      let visF := reachableNodesAvoiding mcfg fBlk avoid
      let needsCyc := allEdges.any fun e =>
        e.2.2.2.2.2 != e.2.2.1 &&
        !(checkArmDisjointCert u 0 visT e && checkArmDisjointCert u 1 visF e)
      let cantReachU :=
        if needsCyc then indices.filter fun b => !avoid.contains b && !canReachAvoiding mcfg b u avoid
        else []
      { mOpt := mOpt, visU := visU, visT := visT, visF := visF, cantReachU := cantReachU }
  | _ => {}

/-- Simulate one block without failing early (for fixed-point shrinking in `MetaM`). -/
def simBlockUni (f : MirFunc) (curU curR : List Nat) (stmts : List Stmt) :
    List Nat × List Nat × List Nat × List Nat :=
  stmts.foldl (fun (badU, badR, uSet, rSet) st =>
    match st with
    | .assign dst rhs =>
        let uOk := rhs.isCtaUni f uSet.contains rSet.contains
        let rOk := rhs.isRfSmemPtr f rSet.contains
        let bu := if !uOk && !badU.contains dst then dst :: badU else badU
        let br := if !rOk && !badR.contains dst then dst :: badR else badR
        let nu := if uOk then (if uSet.contains dst then uSet else dst :: uSet) else uSet.filter (· != dst)
        let nr := if rOk then (if rSet.contains dst then rSet else dst :: rSet) else rSet.filter (· != dst)
        (bu, br, nu, nr)
    | _ => (badU, badR, uSet, rSet)) ([], [], curU, curR)

/-- Monotonic shrink computing `(uniVars, rfPtrs, splitCerts)` over cross-block variables only! -/
def inferGlobalUniLoop (f : MirFunc) (mcfg : ControlFlowCfg) (pdomLists : List (List Nat))
    (rets indices : List Nat)
    (allEdges : List (Nat × Nat × Nat × Nat × Nat × Nat)) :
    Nat → List Nat → List Nat → List (Option SplitCert) →
    List Nat × List Nat × List SplitCert
  | 0, curU, curR, scOpts => (curU, curR, scOpts.map (·.getD {}))
  | fuel + 1, curU, curR, scOpts =>
      let sims := f.blocks.map fun blk => simBlockUni f curU curR blk.stmts
      let exitAt := fun b =>
        let (_, _, exU, exR) := sims.getD b ([], [], [], [])
        (exU, exR)
      let badAssignU := sims.foldl (fun acc (bu, _, _, _) =>
        bu.foldl (fun a v => if a.contains v then a else v :: a) acc) []
      let badAssignR := sims.foldl (fun acc (_, br, _, _) =>
        br.foldl (fun a v => if a.contains v then a else v :: a) acc) []
      let scOpts' := indices.map fun u =>
        match scOpts.getD u none, (f.blockAt u).term with
        | some sc, _ => some sc
        | none, .condBr cond _ _ _ _ =>
            if (exitAt u).1.contains cond then none
            else some (inferSplitCertFor f mcfg pdomLists rets indices allEdges u)
        | none, _ => none
      let activeSplits := indices.filter fun u =>
        match (f.blockAt u).term with
        | .condBr cond _ _ _ _ => !(exitAt u).1.contains cond
        | _ => false
      let droppedArgs : List Nat := activeSplits.foldl (fun acc u =>
        let sc := (scOpts'.getD u none).getD {}
        let step1 := allEdges.foldl (fun a e0 =>
          if checkArmDisjointCert u 0 sc.visT e0 then a
          else if allEdges.any (fun e1 => e0.1 == e1.1 && e0.2.1 == e1.2.1 && e0.2.2.2.2.2 != e1.2.2.2.2.2 && !checkArmDisjointCert u 1 sc.visF e1) then
            if a.contains e0.2.2.1 then a else e0.2.2.1 :: a
          else a) acc
        allEdges.foldl (fun a e =>
          if sc.cantReachU.contains e.1 || e.2.2.2.2.2 == e.2.2.1 ||
             (checkArmDisjointCert u 0 sc.visT e && checkArmDisjointCert u 1 sc.visF e) then a
          else if a.contains e.2.2.1 then a else e.2.2.1 :: a) step1) badAssignU
      let badU := allEdges.foldl (fun a e =>
        if (exitAt e.2.2.2.1).1.contains e.2.2.2.2.2 then a
        else if a.contains e.2.2.1 then a else e.2.2.1 :: a) droppedArgs
      let badR := allEdges.foldl (fun a e =>
        if (exitAt e.2.2.2.1).2.contains e.2.2.2.2.2 then a
        else if a.contains e.2.2.1 then a else e.2.2.1 :: a) badAssignR
      let nextU := curU.filter fun v => !badU.contains v
      let nextR := curR.filter fun v => !badR.contains v
      if nextU.length == curU.length && nextR.length == curR.length then
        (curU, curR, scOpts'.map (·.getD {}))
      else inferGlobalUniLoop f mcfg pdomLists rets indices allEdges fuel nextU nextR scOpts'

/-- Infer the compact uniformity certificate bundle in native `MetaM` code. -/
def inferUniformityBundle (f : MirFunc) (cfg : ControlFlowCfg) :
    List Nat × List Nat × List BlockUniSummary × List Nat ×
    List (Nat × Nat × Nat × Nat × Nat × Nat) × List Nat × ControlFlowCfg × StateMap :=
  let mcfg := mirToCfg f
  let skel := extractTermSkeleton f
  let rets := returnBlocks mcfg
  let indices := List.range f.blocks.length
  let pdomLists := computePostDomLists mcfg rets indices
  let crossVars := collectCrossBlockVars f
  let entryArgs := (f.blockAt 0).args.map BlockArg.varId
  let initR := crossVars.filter fun v => !entryArgs.contains v
  let allEdges := collectAllArgEdges f
  let initSc := indices.map fun _ => (none : Option SplitCert)
  let (uniVars, rfPtrs, splitCerts) :=
    inferGlobalUniLoop f mcfg pdomLists rets indices allEdges (crossVars.length + 3) crossVars initR initSc
  let queriedExitAt := indices.map fun b =>
    let edgeQs := allEdges.foldl (fun a e =>
      if e.2.2.2.1 == b && !a.contains e.2.2.2.2.2 then e.2.2.2.2.2 :: a else a) []
    match (f.blockAt b).term with
    | .condBr c _ _ _ _ => if edgeQs.contains c then edgeQs else c :: edgeQs
    | _ => edgeQs
  let sums := indices.map fun b =>
    let (sDsts, cbOpt) := skel.getD b ([], none)
    let (_, exU, exR) := checkExecUniStmts f uniVars rfPtrs uniVars rfPtrs (f.blockAt b).stmts
    let qs := queriedExitAt.getD b []
    { succDsts := sDsts, condBrOpt := cbOpt
    , exitUni := exU.filter qs.contains
    , exitRf := exR.filter qs.contains
    , split := splitCerts.getD b {} }
  let allConds := indices.foldl (fun acc mb =>
    match (f.blockAt mb).term with
    | .condBr c _ _ _ _ => if acc.contains c then acc else c :: acc
    | _ => acc) []
  let uniConds := allConds.filter fun c =>
    indices.all fun mb =>
      match (f.blockAt mb).term with
      | .condBr mc _ _ _ _ => mc != c || (sums.getD mb {}).exitUni.contains c
      | _ => true
  let refined := refineCfgWithUni cfg uniConds
  (uniVars, rfPtrs, sums, rets, allEdges, uniConds, refined, inferStateMap refined)

open Lean Elab Command Meta in
/-- Collapse `@OfNat.ofNat Nat (lit n) _` wrappers generated by `toExpr` into 1-node raw `Literal.natVal n`
    kernel literals (`8x` smaller `Expr` trees and zero `OfNat.ofNat` unfolding in WHNF). -/
partial def stripOfNatWrappers (e : Expr) : Expr :=
  match e with
  | .app (.app (.app (.const ``OfNat.ofNat _) (.const ``Nat _)) lit@(.lit (.natVal _))) _ => lit
  | .app f a => .app (stripOfNatWrappers f) (stripOfNatWrappers a)
  | .lam n t b bi => .lam n (stripOfNatWrappers t) (stripOfNatWrappers b) bi
  | .forallE n t b bi => .forallE n (stripOfNatWrappers t) (stripOfNatWrappers b) bi
  | .letE n t v b nd => .letE n (stripOfNatWrappers t) (stripOfNatWrappers v) (stripOfNatWrappers b) nd
  | .mdata m b => .mdata m (stripOfNatWrappers b)
  | .proj s i b => .proj s i (stripOfNatWrappers b)
  | _ => e

open Lean Elab Command Meta in
/-- Emits compact `cfgUniVars`, `cfgRfPtrs`, `cfgUniSums`, `cfgSuccChunks`, `cfgRets`, `cfgAllEdges`,
    `cfgUniConds`, `cfgRefined`, and `cfgUniStateMap` constants. -/
elab "cfg_def_uni_state_map " fnId:ident cfgId:ident : command => do
  Command.liftTermElabM do
    let ns ← getCurrNamespace
    let fExpr ← Term.elabTerm (← `($fnId:ident)) (some (mkConst ``MirFunc))
    let cExpr ← Term.elabTerm (← `($cfgId:ident)) (some (mkConst ``ControlFlowCfg))
    let fVal ← unsafe evalExpr MirFunc (mkConst ``MirFunc) fExpr
    let cVal ← unsafe evalExpr ControlFlowCfg (mkConst ``ControlFlowCfg) cExpr
    let (uniVars, rfPtrs, sums, rets, allEdges, uniConds, refinedCfg, smVal) := inferUniformityBundle fVal cVal
    let natTy := mkConst ``Nat
    let natListTy ← Meta.mkAppM ``List #[natTy]
    let n2 ← Meta.mkAppM ``Prod #[natTy, natTy]
    let n3 ← Meta.mkAppM ``Prod #[natTy, n2]
    let n4 ← Meta.mkAppM ``Prod #[natTy, n3]
    let n5 ← Meta.mkAppM ``Prod #[natTy, n4]
    let edgeTy ← Meta.mkAppM ``Prod #[natTy, n5]
    let allEdgesTy ← Meta.mkAppM ``List #[edgeTy]
    let sumListTy ← Meta.mkAppM ``List #[mkConst ``BlockUniSummary]
    let natList2Ty ← Meta.mkAppM ``List #[natListTy]
    let natList3Ty ← Meta.mkAppM ``List #[natList2Ty]
    let succChunks := chunk16 sums.length (sums.map BlockUniSummary.succDsts)
    let stListTy ← Meta.mkAppM ``List #[mkConst ``CtaState]
    let smTy ← Meta.mkAppM ``List #[stListTy]
    addDecl <| Declaration.defnDecl {
      name := ns ++ `cfgUniVars, levelParams := [], type := natListTy, value := stripOfNatWrappers (toExpr uniVars),
      hints := ReducibilityHints.regular 1, safety := DefinitionSafety.safe }
    addDecl <| Declaration.defnDecl {
      name := ns ++ `cfgRfPtrs, levelParams := [], type := natListTy, value := stripOfNatWrappers (toExpr rfPtrs),
      hints := ReducibilityHints.regular 1, safety := DefinitionSafety.safe }
    addDecl <| Declaration.defnDecl {
      name := ns ++ `cfgUniSums, levelParams := [], type := sumListTy, value := stripOfNatWrappers (toExpr sums),
      hints := ReducibilityHints.regular 1, safety := DefinitionSafety.safe }
    addDecl <| Declaration.defnDecl {
      name := ns ++ `cfgSuccChunks, levelParams := [], type := natList3Ty, value := stripOfNatWrappers (toExpr succChunks),
      hints := ReducibilityHints.regular 1, safety := DefinitionSafety.safe }
    addDecl <| Declaration.defnDecl {
      name := ns ++ `cfgRets, levelParams := [], type := natListTy, value := stripOfNatWrappers (toExpr rets),
      hints := ReducibilityHints.regular 1, safety := DefinitionSafety.safe }
    addDecl <| Declaration.defnDecl {
      name := ns ++ `cfgAllEdges, levelParams := [], type := allEdgesTy, value := stripOfNatWrappers (toExpr allEdges),
      hints := ReducibilityHints.regular 1, safety := DefinitionSafety.safe }
    addDecl <| Declaration.defnDecl {
      name := ns ++ `cfgUniConds, levelParams := [], type := natListTy, value := stripOfNatWrappers (toExpr uniConds),
      hints := ReducibilityHints.regular 1, safety := DefinitionSafety.safe }
    addDecl <| Declaration.defnDecl {
      name := ns ++ `cfgRefined, levelParams := [], type := mkConst ``ControlFlowCfg, value := stripOfNatWrappers (toExpr refinedCfg),
      hints := ReducibilityHints.regular 1, safety := DefinitionSafety.safe }
    let origSmVal := if refinedCfg == cVal then smVal else inferStateMap cVal
    addDecl <| Declaration.defnDecl {
      name := ns ++ `cfgUniStateMap, levelParams := [], type := smTy, value := stripOfNatWrappers (toExpr smVal),
      hints := ReducibilityHints.regular 1, safety := DefinitionSafety.safe }
    addDecl <| Declaration.defnDecl {
      name := ns ++ `cfgStateMap, levelParams := [], type := smTy, value := stripOfNatWrappers (toExpr origSmVal),
      hints := ReducibilityHints.regular 1, safety := DefinitionSafety.safe }

end CudaOxide.Proofs.ControlFlow
