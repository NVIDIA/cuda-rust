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

/-! ## Lattice & Monotonicity Lemmas for Symbolic `CtaState` Summarization -/

theorem listSubsumes_refl {α : Type} {sub : α → α → Bool} (hrefl : ∀ x, sub x x = true) :
    ∀ xs : List α, listSubsumes sub xs xs = true
  | [] => rfl
  | x :: xs => by simp [listSubsumes, hrefl x, listSubsumes_refl hrefl xs]

theorem listSubsumes_trans {α : Type} {sub : α → α → Bool}
    (htrans : ∀ a b c, sub a b = true → sub b c = true → sub a c = true) :
    ∀ {xs ys zs : List α}, listSubsumes sub xs ys = true → listSubsumes sub ys zs = true →
      listSubsumes sub xs zs = true
  | [], [], [], _, _ => rfl
  | x :: xs, y :: ys, z :: zs, h1, h2 => by
      simp only [listSubsumes, Bool.and_eq_true] at h1 h2 ⊢
      exact ⟨htrans x y z h1.1 h2.1, listSubsumes_trans htrans h1.2 h2.2⟩
  | [], _ :: _, _, h1, _ | _ :: _, [], _, h1, _ => by simp [listSubsumes] at h1
  | _ :: _, _ :: _, [], _, h2 => by simp [listSubsumes] at h2

theorem listSubsumes_length {α : Type} {sub : α → α → Bool} :
    ∀ {xs ys : List α}, listSubsumes sub xs ys = true → xs.length = ys.length
  | [], [], _ => rfl
  | _ :: xs, _ :: ys, h => by
      simp only [listSubsumes, Bool.and_eq_true] at h
      exact congrArg (· + 1) (listSubsumes_length h.2)
  | [], _ :: _, h | _ :: _, [], h => by simp [listSubsumes] at h

theorem listSubsumes_getD {α : Type} {sub : α → α → Bool} {d : α} (hd : sub d d = true) :
    ∀ {xs ys : List α} (i : Nat), listSubsumes sub xs ys = true → sub (xs.getD i d) (ys.getD i d) = true
  | [], [], _, _ => by simp [List.getD, hd]
  | x :: _, y :: _, 0, h => by simp only [listSubsumes, Bool.and_eq_true] at h; simpa [List.getD] using h.1
  | _ :: xs, _ :: ys, i + 1, h => by
      simp only [listSubsumes, Bool.and_eq_true] at h
      simpa [List.getD] using listSubsumes_getD hd i h.2
  | [], _ :: _, _, h | _ :: _, [], _, h => by simp [listSubsumes] at h

theorem listSubsumes_getElem? {α : Type} {sub : α → α → Bool} :
    ∀ {xs ys : List α} (i : Nat), listSubsumes sub xs ys = true →
      match xs[i]?, ys[i]? with
      | some a, some c => sub a c = true
      | none, none => True
      | _, _ => False
  | [], [], _, _ => by simp
  | x :: _, y :: _, 0, h => by simp only [listSubsumes, Bool.and_eq_true] at h; simpa using h.1
  | _ :: xs, _ :: ys, i + 1, h => by
      simp only [listSubsumes, Bool.and_eq_true] at h
      simpa using listSubsumes_getElem? i h.2
  | [], _ :: _, _, h | _ :: _, [], _, h => by simp [listSubsumes] at h

theorem listSubsumes_set {α : Type} {sub : α → α → Bool} {a c : α} (hac : sub a c = true) :
    ∀ {xs ys : List α} (i : Nat), listSubsumes sub xs ys = true →
      listSubsumes sub (xs.set i a) (ys.set i c) = true
  | [], [], _, _ => rfl
  | _ :: xs, _ :: ys, 0, h => by simp only [listSubsumes, Bool.and_eq_true] at h; simp [listSubsumes, hac, h.2]
  | x :: xs, y :: ys, i + 1, h => by
      simp only [listSubsumes, Bool.and_eq_true] at h
      simp [listSubsumes, h.1, listSubsumes_set hac i h.2]
  | [], _ :: _, _, h | _ :: _, [], _, h => by simp [listSubsumes] at h

theorem listSubsumes_map {α : Type} {sub : α → α → Bool} {f : α → α}
    (hf : ∀ a c, sub a c = true → sub (f a) (f c) = true) :
    ∀ {xs ys : List α}, listSubsumes sub xs ys = true → listSubsumes sub (xs.map f) (ys.map f) = true
  | [], [], _ => rfl
  | x :: xs, y :: ys, h => by
      simp only [listSubsumes, Bool.and_eq_true] at h
      simp [listSubsumes, hf x y h.1, listSubsumes_map hf h.2]
  | [], _ :: _, h | _ :: _, [], h => by simp [listSubsumes] at h

theorem natSubset_mem {xs ys : List Nat} {x : Nat} (h : natSubset xs ys = true) (hx : x ∈ xs) :
    x ∈ ys :=
  of_decide_eq_true (List.all_eq_true.mp h x hx)

theorem natSubset_refl (xs : List Nat) : natSubset xs xs = true :=
  List.all_eq_true.mpr fun _ hx => decide_eq_true hx

theorem natSubset_trans {xs ys zs : List Nat} (h1 : natSubset xs ys = true) (h2 : natSubset ys zs = true) :
    natSubset xs zs = true :=
  List.all_eq_true.mpr fun _ hx => decide_eq_true (natSubset_mem h2 (natSubset_mem h1 hx))

theorem natSubset_isEmpty {conc abs : List Nat} (h : natSubset conc abs = true) (hemp : abs.isEmpty = true) :
    conc.isEmpty = true := by
  cases conc with
  | nil => rfl
  | cons x _ =>
      have hx := natSubset_mem h (.head _)
      cases abs <;> simp_all

theorem mem_insertSortedNat {x z : Nat} : ∀ ys : List Nat, z ∈ insertSortedNat x ys ↔ z = x ∨ z ∈ ys
  | [] => by simp [insertSortedNat]
  | y :: ys => by
      unfold insertSortedNat
      by_cases hxy : (x == y) = true
      · have hxy' : x = y := beq_iff_eq.mp hxy
        subst hxy'; simp [List.mem_cons]
      · simp only [hxy, Bool.false_eq_true, ↓reduceIte]
        by_cases hlt : x < y
        · simp [hlt, List.mem_cons]
        · simp only [hlt, ↓reduceIte, List.mem_cons, mem_insertSortedNat ys]
          constructor
          · rintro (rfl | rfl | hm); exact Or.inr (Or.inl rfl); exact Or.inl rfl; exact Or.inr (Or.inr hm)
          · rintro (rfl | rfl | hm); exact Or.inr (Or.inl rfl); exact Or.inl rfl; exact Or.inr (Or.inr hm)

theorem natSubset_insert (x : Nat) {c a : List Nat} (h : natSubset c a = true) :
    natSubset (insertSortedNat x c) (insertSortedNat x a) = true := by
  apply List.all_eq_true.mpr
  intro z hz
  apply decide_eq_true
  rw [mem_insertSortedNat] at hz ⊢
  rcases hz with rfl | hzc
  · exact Or.inl rfl
  · exact Or.inr (natSubset_mem h hzc)

theorem natSubset_filter (p : Nat → Bool) {c a : List Nat} (h : natSubset c a = true) :
    natSubset (c.filter p) (a.filter p) = true := by
  apply List.all_eq_true.mpr
  intro z hz
  rw [List.mem_filter] at hz
  exact decide_eq_true (List.mem_filter.mpr ⟨natSubset_mem h hz.1, hz.2⟩)

theorem proxySub_refl : ∀ x : ProxyState, ProxyState.subsumes x x = true := by
  intro x; cases x <;> decide
theorem proxySub_trans : ∀ a b c : ProxyState,
    ProxyState.subsumes a b = true → ProxyState.subsumes b c = true → ProxyState.subsumes a c = true := by
  intro a b c; cases a <;> cases b <;> cases c <;> decide
theorem proxyMerge_mono : ∀ a c d : ProxyState,
    ProxyState.subsumes a c = true → ProxyState.subsumes (ProxyState.merge a d) (ProxyState.merge c d) = true := by
  intro a c d; cases a <;> cases c <;> cases d <;> decide
theorem proxyStepSync_mono : ∀ a c : ProxyState,
    ProxyState.subsumes a c = true →
    ProxyState.subsumes (match a with | .needsSyncAndFence => .needsFence | o => o)
      (match c with | .needsSyncAndFence => .needsFence | o => o) = true := by
  intro a c; cases a <;> cases c <;> decide
theorem proxyStepFence_mono : ∀ a c : ProxyState,
    ProxyState.subsumes a c = true →
    ProxyState.subsumes (match a with | .needsFence => .clean | o => o)
      (match c with | .needsFence => .clean | o => o) = true := by
  intro a c; cases a <;> cases c <;> decide
theorem proxySub_clean_eq : ∀ c : ProxyState, ProxyState.subsumes .clean c = true → c = .clean := by
  intro c; cases c <;> decide

theorem mbarSub_refl : ∀ x : MbarState, MbarState.subsumes x x = true := by
  intro x; cases x <;> decide
theorem mbarSub_trans : ∀ a b c : MbarState,
    MbarState.subsumes a b = true → MbarState.subsumes b c = true → MbarState.subsumes a c = true := by
  intro a b c; cases a <;> cases b <;> cases c <;> decide
theorem mbarStepSync_mono : ∀ a c : MbarState,
    MbarState.subsumes a c = true →
    MbarState.subsumes (match a with | .initUnsynced => .syncedIdle | o => o)
      (match c with | .initUnsynced => .syncedIdle | o => o) = true := by
  intro a c; cases a <;> cases c <;> decide
theorem mbarSub_armedTx_eq : ∀ c : MbarState, MbarState.subsumes .armedTx c = true → c = .armedTx := by
  intro c; cases c <;> decide
theorem mbarSub_validArrive : ∀ a c : MbarState, MbarState.subsumes a c = true →
    (a = .syncedIdle ∨ a = .armedPlain ∨ a = .armedTx ∨ a = .completed) →
    (c = .syncedIdle ∨ c = .armedPlain ∨ c = .armedTx ∨ c = .completed) := by
  intro a c; cases a <;> cases c <;> decide

theorem mbarSub_readyOk : ∀ (va vc : MbarState) (f : Bool),
    MbarState.subsumes va vc = true →
    (match some va with | some .armedPlain | some .armedTx | some .completed => true | some .syncedIdle => f | _ => false) = true →
    (match some vc with | some .armedPlain | some .armedTx | some .completed => true | some .syncedIdle => f | _ => false) = true := by
  intro va vc f; cases va <;> cases vc <;> cases f <;> decide

theorem tmemReadyStep_mono : ∀ (memA memC x mhA mhC trA trC : Bool),
    (!memA || memC) = true → (!mhC || mhA) = true → (!trA || trC) = true →
    (!(if (memA || x) && !mhA then true else trA) || (if (memC || x) && !mhC then true else trC)) = true := by
  intro memA memC x mhA mhC trA trC; cases memA <;> cases memC <;> cases x <;> cases mhA <;> cases mhC <;> cases trA <;> cases trC <;> decide

theorem ite_none_some_iff {α : Type} {c : Bool} {x y : α} :
    (if (!c) = true then none else some x) = some y ↔ c = true ∧ x = y := by
  cases c <;> simp

theorem mbarSub_tmaArmOk : ∀ (va vc : MbarState) (cg : Bool) (ex : Bool),
    MbarState.subsumes va vc = true →
    (match va with | .armedTx => true | .syncedIdle | .armedPlain | .completed => cg && ex | _ => false) = true →
    (match vc with | .armedTx => true | .syncedIdle | .armedPlain | .completed => cg && ex | _ => false) = true := by
  intro va vc cg ex; cases va <;> cases vc <;> cases cg <;> cases ex <;> decide

theorem tmemPhaseSub_refl (p : TmemPhase) : TmemPhase.subsumes p p = true := by
  simp [TmemPhase.subsumes]

theorem tmemPhaseSub_trans {a b c : TmemPhase}
    (h1 : TmemPhase.subsumes a b = true) (h2 : TmemPhase.subsumes b c = true) :
    TmemPhase.subsumes a c = true := by
  simp only [TmemPhase.subsumes, Bool.or_eq_true, beq_iff_eq] at h1 h2 ⊢
  rcases h1 with rfl | rfl
  · exact Or.inl rfl
  · exact h2

theorem ctaSubsumes_refl (st : CtaState) : st.subsumes st = true := by
  unfold CtaState.subsumes
  have hBool : ∀ b : Bool, (!b || b) = true := by decide
  have hNat : listSubsumes (fun a c => natSubset c a) st.sharedPendingTma st.sharedPendingTma = true :=
    listSubsumes_refl (sub := fun a c => natSubset c a) (fun x => natSubset_refl x) _
  have hBList : listSubsumes (fun a c => !a || c) st.sharedClusterSynced st.sharedClusterSynced = true :=
    listSubsumes_refl (sub := fun a c => !a || c) hBool _
  simp [listSubsumes_refl proxySub_refl, hNat, hBList, listSubsumes_refl mbarSub_refl,
        tmemPhaseSub_refl, natSubset_refl, hBool]

theorem ctaSubsumes_trans {a b c : CtaState} (h1 : a.subsumes b = true) (h2 : b.subsumes c = true) :
    a.subsumes c = true := by
  unfold CtaState.subsumes at h1 h2 ⊢
  have hbTrans : ∀ x y z : Bool, (!x || y) = true → (!y || z) = true → (!x || z) = true := by decide
  simp only [Bool.and_eq_true] at h1 h2 ⊢
  rcases h1 with ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨hp1, ht1⟩, hc1⟩, hbp1⟩, hbs1⟩, htp1⟩, hac1⟩, hmh1⟩, hmi1⟩, hcb1⟩, htr1⟩, hpl1⟩, hss1⟩, hcs1⟩, hrp1⟩, hvp1⟩
  rcases h2 with ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨hp2, ht2⟩, hc2⟩, hbp2⟩, hbs2⟩, htp2⟩, hac2⟩, hmh2⟩, hmi2⟩, hcb2⟩, htr2⟩, hpl2⟩, hss2⟩, hcs2⟩, hrp2⟩, hvp2⟩
  have hptT : listSubsumes (fun x y => natSubset y x) a.sharedPendingTma c.sharedPendingTma = true :=
    listSubsumes_trans (sub := fun x y => natSubset y x) (fun _ _ _ u v => natSubset_trans v u) ht1 ht2
  exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
    listSubsumes_trans proxySub_trans hp1 hp2, hptT⟩,
    listSubsumes_trans (sub := fun x y => !x || y) hbTrans hc1 hc2⟩,
    listSubsumes_trans proxySub_trans hbp1 hbp2⟩,
    listSubsumes_trans mbarSub_trans hbs1 hbs2⟩,
    tmemPhaseSub_trans htp1 htp2⟩,
    hbTrans _ _ _ hac1 hac2⟩,
    hbTrans _ _ _ hmh2 hmh1⟩,
    hbTrans _ _ _ hmi1 hmi2⟩,
    natSubset_trans hcb1 hcb2⟩,
    hbTrans _ _ _ htr1 htr2⟩,
    natSubset_trans hpl2 hpl1⟩,
    hbTrans _ _ _ hss1 hss2⟩,
    hbTrans _ _ _ hcs1 hcs2⟩,
    hbTrans _ _ _ hrp1 hrp2⟩,
    natSubset_trans hvp2 hvp1⟩

/-- Monotonicity of `validReturnState` under `CtaState.subsumes`. -/
theorem validReturnState_mono {req : Bool} {abs conc : CtaState}
    (hsub : abs.subsumes conc = true) (hret : validReturnState req abs = true) :
    validReturnState req conc = true := by
  have hs := hsub; unfold CtaState.subsumes at hs
  simp only [Bool.and_eq_true] at hs
  rcases hs with ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨_, _⟩, _⟩, _⟩, _⟩, htp⟩, _⟩, hmh⟩, _⟩, _⟩, _⟩, hpl⟩, _⟩, _⟩, _⟩, _⟩
  have hbFalse : ∀ {x y : Bool}, (!x || y) = true → y = false → x = false := by decide
  unfold validReturnState at hret ⊢
  simp only [Bool.and_eq_true, Bool.not_eq_true'] at hret ⊢
  rcases hret with ⟨⟨hph, hmha⟩, hpla⟩
  refine ⟨⟨?_, hbFalse hmh hmha⟩, natSubset_isEmpty hpl hpla⟩
  simp only [TmemPhase.subsumes, Bool.or_eq_true, beq_iff_eq] at htp
  rcases htp with hconf | heq
  · rw [hconf] at hph; cases req <;> simp at hph
  · rw [← heq]; exact hph

/-- Monotonicity of `enabledCtaSuccs`: every edge enabled in `conc` is also enabled in `abs`. -/
theorem enabledCtaSuccs_mono {cfg : ControlFlowCfg} {b : Nat} {abs conc : CtaState}
    {cond : BranchCond} {dst : Nat}
    (hsub : abs.subsumes conc = true)
    (he : (cond, dst) ∈ enabledCtaSuccs cfg b conc) :
    (cond, dst) ∈ enabledCtaSuccs cfg b abs := by
  have hs := hsub; unfold CtaState.subsumes at hs; simp only [Bool.and_eq_true] at hs
  have hvp := hs.2
  unfold enabledCtaSuccs at he ⊢
  rw [List.mem_filter] at he ⊢
  refine ⟨he.1, ?_⟩
  cases cond with
  | posLoopExit bodyBlk =>
      dsimp only at he ⊢
      exact decide_eq_true (natSubset_mem hvp (of_decide_eq_true he.2))
  | _ => rfl

/-- Monotonicity of `stepOp` under `CtaState.subsumes`. -/
theorem stepOp_mono (cfg : ControlFlowCfg) (curBlk : Nat) (blkGuard : GuardExpr) (inCycle : Bool)
    {abs conc abs' : CtaState} (op : CfgOp)
    (hsub : abs.subsumes conc = true)
    (hstep : stepOp cfg curBlk blkGuard inCycle abs op = some abs') :
    ∃ conc', stepOp cfg curBlk blkGuard inCycle conc op = some conc' ∧ abs'.subsumes conc' = true := by
  have hs := hsub; unfold CtaState.subsumes at hs; simp only [Bool.and_eq_true] at hs
  rcases hs with ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨hsp, hpt⟩, hsc⟩, hbp⟩, hbs⟩, htp⟩, hac⟩, hmh⟩, hmi⟩, hcb⟩, htr⟩, hpl⟩, hss⟩, hcs⟩, hrp⟩, hvp⟩
  have hlenSP := listSubsumes_length hsp
  have hlenPT := listSubsumes_length hpt
  have hlenSC := listSubsumes_length hsc
  have hlenBS := listSubsumes_length hbs
  have hbImp : ∀ {x y : Bool}, (!x || y) = true → x = true → y = true := by decide
  have hbFalse : ∀ {x y : Bool}, (!x || y) = true → y = false → x = false := by decide
  cases op with
  | syncThreads =>
      unfold stepOp at hstep ⊢
      injection hstep with hstep; subst hstep
      refine ⟨_, rfl, ?_⟩
      unfold CtaState.subsumes; simp only [Bool.and_eq_true]
      refine ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
        listSubsumes_map proxyStepSync_mono hsp, hpt⟩, hsc⟩,
        listSubsumes_map proxyStepSync_mono hbp⟩,
        listSubsumes_map mbarStepSync_mono hbs⟩, htp⟩, hac⟩, hmh⟩, hmi⟩, hcb⟩, htr⟩, hpl⟩, ?_⟩, hcs⟩, hrp⟩, hvp⟩
      cases ha : (!abs.mmaHazard && abs.pendingLoads.isEmpty)
      · simp
      · simp only [Bool.and_eq_true, Bool.not_eq_true'] at ha
        have hc1 : conc.mmaHazard = false := hbFalse hmh ha.1
        have hc2 : conc.pendingLoads.isEmpty = true := natSubset_isEmpty hpl ha.2
        simp [hc1, hc2]
  | clusterSync =>
      dsimp only [stepOp] at hstep ⊢
      by_cases hcdim : cfg.clusterDimX ≤ 1
      · rw [if_pos hcdim] at hstep; contradiction
      · rw [if_neg hcdim] at hstep ⊢
        injection hstep with hstep; subst hstep
        refine ⟨_, rfl, ?_⟩
        have hcd : (!(!abs.mmaHazard && abs.pendingLoads.isEmpty) || (!conc.mmaHazard && conc.pendingLoads.isEmpty)) = true := by
          cases ha : (!abs.mmaHazard && abs.pendingLoads.isEmpty)
          · simp
          · simp only [Bool.and_eq_true, Bool.not_eq_true'] at ha
            simp [hbFalse hmh ha.1, natSubset_isEmpty hpl ha.2]
        have hrep : listSubsumes (fun a c => !a || c)
            (List.replicate abs.sharedClusterSynced.length true)
            (List.replicate conc.sharedClusterSynced.length true) = true := by
          rw [← hlenSC]; exact listSubsumes_refl (sub := fun a c => !a || c) (by decide) _
        unfold CtaState.subsumes; simp only [Bool.and_eq_true]
        exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
          listSubsumes_map proxyStepSync_mono hsp, hpt⟩, hrep⟩,
          listSubsumes_map proxyStepSync_mono hbp⟩,
          listSubsumes_map mbarStepSync_mono hbs⟩, htp⟩, rfl⟩, hmh⟩, hmi⟩, hcb⟩, htr⟩, hpl⟩, hcd⟩, hcd⟩, hrp⟩, hvp⟩
  | proxyFenceAsyncShared =>
      dsimp only [stepOp] at hstep ⊢
      injection hstep with hstep; subst hstep
      refine ⟨_, rfl, ?_⟩
      unfold CtaState.subsumes; simp only [Bool.and_eq_true]
      exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
        listSubsumes_map proxyStepFence_mono hsp, hpt⟩, hsc⟩,
        listSubsumes_map proxyStepFence_mono hbp⟩, hbs⟩, htp⟩, hac⟩, hmh⟩, hmi⟩, hcb⟩, htr⟩, hpl⟩, hss⟩, hcs⟩, hrp⟩, hvp⟩
  | sharedRead s =>
      dsimp only [stepOp] at hstep ⊢
      rw [← hlenSP]
      by_cases hslt : s < abs.sharedProxy.length
      · rw [if_pos hslt] at hstep ⊢; injection hstep with hstep; subst hstep; exact ⟨conc, rfl, hsub⟩
      · rw [if_neg hslt] at hstep; contradiction
  | dsmemAccess s =>
      dsimp only [stepOp] at hstep ⊢
      rw [← hlenSC]
      by_cases hpre : (decide (cfg.clusterDimX ≤ 1) || decide (s ≥ abs.sharedClusterSynced.length)) = true
      · rw [if_pos hpre] at hstep; contradiction
      · rw [if_neg hpre] at hstep ⊢
        cases hok : abs.sharedClusterSynced.getD s false
        · rw [hok] at hstep; contradiction
        · rw [hok] at hstep; simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte] at hstep
          have hget := listSubsumes_getD (sub := fun a c => !a || c) (d := false) rfl s hsc
          have hcOk : conc.sharedClusterSynced.getD s false = true := hbImp hget hok
          injection hstep with hstep; subst hstep
          simp only [hcOk, Bool.not_true, Bool.false_eq_true, ↓reduceIte]
          exact ⟨conc, rfl, hsub⟩
  | sharedWrite s =>
      dsimp only [stepOp] at hstep ⊢
      rw [← hlenSP, ← hlenPT]
      by_cases hpre : (decide (s ≥ abs.sharedProxy.length) || decide (s ≥ abs.sharedPendingTma.length)) = true
      · rw [if_pos hpre] at hstep; contradiction
      · rw [if_neg hpre] at hstep ⊢
        cases hemp : (abs.sharedPendingTma.getD s []).isEmpty
        · rw [hemp] at hstep; contradiction
        · rw [hemp] at hstep; simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte] at hstep
          have hptS := listSubsumes_getD (sub := fun a c => natSubset c a) (d := []) (natSubset_refl []) s hpt
          have hcEmp : (conc.sharedPendingTma.getD s []).isEmpty = true := natSubset_isEmpty hptS hemp
          injection hstep with hstep; subst hstep
          simp only [hcEmp, Bool.not_true, Bool.false_eq_true, ↓reduceIte]
          refine ⟨_, rfl, ?_⟩
          have hspS := proxyMerge_mono _ _ (if (blkGuard.hasZeroLeaderCheck && !blkGuard.hasUnknown) || (cfg.blockDimY == 1 && blkGuard.selectsAtMostOneThread) then .needsFence else .needsSyncAndFence) (listSubsumes_getD (d := .clean) rfl s hsp)
          unfold CtaState.subsumes; simp only [Bool.and_eq_true]
          exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
            listSubsumes_set hspS s hsp, hpt⟩,
            listSubsumes_set (sub := fun x y => !x || y) rfl s hsc⟩, hbp⟩, hbs⟩, htp⟩, hac⟩, hmh⟩, hmi⟩, hcb⟩, htr⟩, hpl⟩, hss⟩, hcs⟩, hrp⟩, hvp⟩
  | mbarInit barId count =>
      dsimp only [stepOp] at hstep ⊢
      rw [← hlenBS]
      by_cases hpre : (decide (barId ≥ abs.barState.length) || count == 0) = true
      · rw [if_pos hpre] at hstep; contradiction
      · rw [if_neg hpre] at hstep ⊢
        injection hstep with hstep; subst hstep
        refine ⟨_, rfl, ?_⟩
        have hbpB := proxyMerge_mono _ _ (if (blkGuard.hasZeroLeaderCheck && !blkGuard.hasUnknown) || (cfg.blockDimY == 1 && blkGuard.selectsAtMostOneThread) then .needsFence else .needsSyncAndFence) (listSubsumes_getD (d := .clean) rfl barId hbp)
        unfold CtaState.subsumes; simp only [Bool.and_eq_true]
        exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
          hsp, hpt⟩, hsc⟩,
          listSubsumes_set hbpB barId hbp⟩,
          listSubsumes_set rfl barId hbs⟩, htp⟩, hac⟩, hmh⟩, hmi⟩, hcb⟩, htr⟩, hpl⟩, hss⟩, hcs⟩, hrp⟩, hvp⟩
  | mbarArrive barId =>
      dsimp only [stepOp] at hstep ⊢
      have hge := listSubsumes_getElem? barId hbs
      cases ha : abs.barState[barId]? with
      | none => rw [ha] at hstep; contradiction
      | some va =>
          cases hc : conc.barState[barId]? with
          | none => rw [ha, hc] at hge; contradiction
          | some vc =>
              rw [ha] at hstep; rw [ha, hc] at hge; dsimp only at hge
              have hva : va = .syncedIdle ∨ va = .armedPlain ∨ va = .armedTx ∨ va = .completed := by
                cases va <;> first | contradiction | simp
              have heqA : abs' = { abs with barState := abs.barState.set barId .armedPlain } := by
                rcases hva with rfl | rfl | rfl | rfl <;> (injection hstep with h1; exact h1.symm)
              subst heqA
              have hvc := mbarSub_validArrive va vc hge hva
              have hset : listSubsumes MbarState.subsumes (abs.barState.set barId .armedPlain)
                  (conc.barState.set barId .armedPlain) = true := listSubsumes_set rfl barId hbs
              rcases hvc with rfl | rfl | rfl | rfl <;>
                (dsimp only; refine ⟨_, rfl, ?_⟩; unfold CtaState.subsumes; simp only [Bool.and_eq_true];
                 exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨hsp, hpt⟩, hsc⟩, hbp⟩, hset⟩, htp⟩, hac⟩, hmh⟩, hmi⟩, hcb⟩, htr⟩, hpl⟩, hss⟩, hcs⟩, hrp⟩, hvp⟩)
  | mbarArriveExpectTx barId txBytes =>
      dsimp only [stepOp] at hstep ⊢
      by_cases htx : (txBytes == 0) = true
      · rw [if_pos htx] at hstep; contradiction
      · rw [if_neg htx] at hstep ⊢
        have hge := listSubsumes_getElem? barId hbs
        cases ha : abs.barState[barId]? with
        | none => rw [ha] at hstep; contradiction
        | some va =>
            cases hc : conc.barState[barId]? with
            | none => rw [ha, hc] at hge; contradiction
            | some vc =>
                rw [ha] at hstep; rw [ha, hc] at hge; dsimp only at hge
                have hva : va = .syncedIdle ∨ va = .armedPlain ∨ va = .armedTx ∨ va = .completed := by
                  cases va <;> first | contradiction | simp
                have heqA : abs' = { abs with barState := abs.barState.set barId .armedTx } := by
                  rcases hva with rfl | rfl | rfl | rfl <;> (injection hstep with h1; exact h1.symm)
                subst heqA
                have hvc := mbarSub_validArrive va vc hge hva
                have hset : listSubsumes MbarState.subsumes (abs.barState.set barId .armedTx)
                    (conc.barState.set barId .armedTx) = true := listSubsumes_set rfl barId hbs
                rcases hvc with rfl | rfl | rfl | rfl <;>
                  (dsimp only; refine ⟨_, rfl, ?_⟩; unfold CtaState.subsumes; simp only [Bool.and_eq_true];
                   exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨hsp, hpt⟩, hsc⟩, hbp⟩, hset⟩, htp⟩, hac⟩, hmh⟩, hmi⟩, hcb⟩, htr⟩, hpl⟩, hss⟩, hcs⟩, hrp⟩, hvp⟩)
  | tmaCopyG2S sharedId barId isCg2 =>
      dsimp only [stepOp] at hstep ⊢
      rw [← hlenSP, ← hlenPT, ← hlenBS]
      by_cases hp1 : ((isCg2 && cfg.clusterDimX != 2) || !((blkGuard.hasZeroLeaderCheck && !blkGuard.hasUnknown) || (cfg.blockDimY == 1 && blkGuard.selectsAtMostOneThread)) || decide (sharedId ≥ abs.sharedProxy.length) || decide (sharedId ≥ abs.sharedPendingTma.length) || decide (barId ≥ abs.barState.length)) = true
      · rw [if_pos hp1] at hstep; contradiction
      · rw [if_neg hp1] at hstep ⊢
        by_cases hp2 : (abs.sharedProxy.getD sharedId .needsFence != .clean || abs.barProxy.getD barId .needsFence != .clean) = true
        · rw [if_pos hp2] at hstep; contradiction
        · rw [if_neg hp2] at hstep
          have hp2f := Bool.eq_false_iff.mpr hp2
          simp only [Bool.or_eq_false_iff, bne_eq_false_iff_eq] at hp2f
          have hspSub := listSubsumes_getD (d := ProxyState.needsFence) rfl sharedId hsp
          rw [hp2f.1] at hspSub
          have hspC := proxySub_clean_eq _ hspSub
          have hbpSub := listSubsumes_getD (d := ProxyState.needsFence) rfl barId hbp
          rw [hp2f.2] at hbpSub
          have hbpC := proxySub_clean_eq _ hbpSub
          rcases ite_none_some_iff.mp hstep with ⟨hba, hstep⟩
          subst hstep
          have hbsB := listSubsumes_getD (d := MbarState.uninit) rfl barId hbs
          have hcArm := mbarSub_tmaArmOk _ _ isCg2 (cfgHasExpectTx cfg barId) hbsB hba
          simp only [hspC, hbpC, bne_self_eq_false, Bool.or_self, Bool.false_eq_true, ↓reduceIte]
          refine ⟨_, ite_none_some_iff.mpr ⟨hcArm, rfl⟩, ?_⟩
          have hptUpd := natSubset_insert barId (listSubsumes_getD (sub := fun a c => natSubset c a) (d := []) (natSubset_refl []) sharedId hpt)
          unfold CtaState.subsumes; simp only [Bool.and_eq_true]
          exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
            hsp, listSubsumes_set (sub := fun a c => natSubset c a) hptUpd sharedId hpt⟩,
            listSubsumes_set (sub := fun a c => !a || c) rfl sharedId hsc⟩, hbp⟩, hbs⟩, htp⟩, hac⟩, hmh⟩, hmi⟩, hcb⟩, htr⟩, hpl⟩, hss⟩, hcs⟩, hrp⟩, hvp⟩
  | clcTryCancel sharedId barId =>
      dsimp only [stepOp] at hstep ⊢
      rw [← hlenSP, ← hlenPT, ← hlenBS]
      by_cases hp1 : (decide (cfg.clusterDimX < 1) || !((blkGuard.hasZeroLeaderCheck && !blkGuard.hasUnknown) || (cfg.blockDimY == 1 && blkGuard.selectsAtMostOneThread)) || decide (sharedId ≥ abs.sharedProxy.length) || decide (sharedId ≥ abs.sharedPendingTma.length) || decide (barId ≥ abs.barState.length)) = true
      · rw [if_pos hp1] at hstep; contradiction
      · rw [if_neg hp1] at hstep ⊢
        by_cases hp2 : (abs.barProxy.getD barId .needsFence != .clean) = true
        · rw [if_pos hp2] at hstep; contradiction
        · rw [if_neg hp2] at hstep
          have hp2eq := bne_eq_false_iff_eq.mp (Bool.eq_false_iff.mpr hp2)
          have hbpSub := listSubsumes_getD (d := ProxyState.needsFence) rfl barId hbp
          rw [hp2eq] at hbpSub
          have hbpC := proxySub_clean_eq _ hbpSub
          by_cases hp3 : (abs.barState.getD barId .uninit != .armedTx) = true
          · rw [if_pos hp3] at hstep; contradiction
          · rw [if_neg hp3] at hstep
            have hp3eq := bne_eq_false_iff_eq.mp (Bool.eq_false_iff.mpr hp3)
            have hbsSub := listSubsumes_getD (d := MbarState.uninit) rfl barId hbs
            rw [hp3eq] at hbsSub
            have hbsC := mbarSub_armedTx_eq _ hbsSub
            injection hstep with hstep; subst hstep
            simp only [hbpC, hbsC, bne_self_eq_false, Bool.false_eq_true, ↓reduceIte]
            refine ⟨_, rfl, ?_⟩
            have hptUpd := natSubset_insert barId (listSubsumes_getD (sub := fun a c => natSubset c a) (d := []) (natSubset_refl []) sharedId hpt)
            unfold CtaState.subsumes; simp only [Bool.and_eq_true]
            exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
              hsp, listSubsumes_set (sub := fun a c => natSubset c a) hptUpd sharedId hpt⟩,
              listSubsumes_set (sub := fun a c => !a || c) rfl sharedId hsc⟩, hbp⟩, hbs⟩, htp⟩, hac⟩, hmh⟩, hmi⟩, hcb⟩, htr⟩, hpl⟩, hss⟩, hcs⟩, hrp⟩, hvp⟩
  | tmemAlloc sharedId cols isCg2 =>
      dsimp only [stepOp] at hstep ⊢
      rw [← hlenSP]
      by_cases hp1 : ((isCg2 && cfg.clusterDimX != 2) || inCycle || decide (sharedId ≥ abs.sharedProxy.length) || !validTmemColsBool cols) = true
      · rw [if_pos hp1] at hstep; contradiction
      · rw [if_neg hp1] at hstep ⊢
        by_cases hp2 : (blkGuard.selectsSingleFullWarp cfg.blockDimX cfg.blockDimY).isNone = true
        · rw [if_pos hp2] at hstep; contradiction
        · rw [if_neg hp2] at hstep ⊢
          by_cases hp3 : (abs.tmemPhase != .uninit) = true
          · rw [if_pos hp3] at hstep; contradiction
          · rw [if_neg hp3] at hstep
            have hp3eq := bne_eq_false_iff_eq.mp (Bool.eq_false_iff.mpr hp3)
            have hcp : conc.tmemPhase = .uninit := by
              simp only [TmemPhase.subsumes, hp3eq, Bool.or_eq_true, beq_iff_eq] at htp
              rcases htp with h1 | h2; contradiction; exact h2.symm
            injection hstep with hstep; subst hstep
            simp only [hcp, bne_self_eq_false, Bool.false_eq_true, ↓reduceIte]
            refine ⟨_, rfl, ?_⟩
            unfold CtaState.subsumes; simp only [Bool.and_eq_true]
            exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
              hsp, hpt⟩, hsc⟩, hbp⟩, hbs⟩, tmemPhaseSub_refl _⟩, by cases isCg2 <;> rfl⟩,
              hmh⟩, hmi⟩, hcb⟩, htr⟩, hpl⟩, rfl⟩, rfl⟩, rfl⟩, hvp⟩
  | tmemRelinquishPermit isCg2 =>
      dsimp only [stepOp] at hstep ⊢
      by_cases hp1 : (isCg2 && cfg.clusterDimX != 2) = true
      · rw [if_pos hp1] at hstep; contradiction
      · rw [if_neg hp1] at hstep ⊢
        by_cases hp2 : (blkGuard.selectsSingleFullWarp cfg.blockDimX cfg.blockDimY).isNone = true
        · rw [if_pos hp2] at hstep; contradiction
        · rw [if_neg hp2] at hstep ⊢
          cases hap : abs.tmemPhase <;> rw [hap] at hstep htp <;> try contradiction
          rename_i cols allocCg2
          have hcp : conc.tmemPhase = .allocated cols allocCg2 := by
            simp only [TmemPhase.subsumes, Bool.or_eq_true, beq_iff_eq] at htp
            rcases htp with h1 | h2; contradiction; exact h2.symm
          rw [hcp]; dsimp only at hstep ⊢
          by_cases hcg : (allocCg2 != isCg2) = true
          · rw [if_pos hcg] at hstep; contradiction
          · rw [if_neg hcg] at hstep ⊢
            injection hstep with hstep; subst hstep
            refine ⟨_, rfl, ?_⟩
            unfold CtaState.subsumes; simp only [Bool.and_eq_true]
            exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
              hsp, hpt⟩, hsc⟩, hbp⟩, hbs⟩, tmemPhaseSub_refl _⟩, hac⟩, hmh⟩, hmi⟩, hcb⟩, htr⟩, hpl⟩, hss⟩, hcs⟩,
              by cases isCg2 <;> rfl⟩, hvp⟩
  | tmemAsyncIssue smemIds isCg2 =>
      dsimp only [stepOp] at hstep ⊢
      by_cases hp1 : ((isCg2 && cfg.clusterDimX != 2) || !((blkGuard.hasZeroLeaderCheck && !blkGuard.hasUnknown) || (cfg.blockDimY == 1 && blkGuard.selectsAtMostOneThread)) || !abs.allocClusterSynced) = true
      · rw [if_pos hp1] at hstep; contradiction
      · rw [if_neg hp1] at hstep
        have hp1f := Bool.eq_false_iff.mpr hp1
        simp only [Bool.or_eq_false_iff, Bool.not_eq_false'] at hp1f
        have hcAc := hbImp hac hp1f.2
        have hcp1 : ((isCg2 && cfg.clusterDimX != 2) || !((blkGuard.hasZeroLeaderCheck && !blkGuard.hasUnknown) || (cfg.blockDimY == 1 && blkGuard.selectsAtMostOneThread)) || !conc.allocClusterSynced) = false := by
          simp [hp1f.1.1, hp1f.1.2, hcAc]
        rw [hcp1]; simp only [Bool.false_eq_true, ↓reduceIte]
        cases hap : abs.tmemPhase <;> rw [hap] at hstep htp <;> try contradiction
        rename_i cols allocCg2
        have hcp : conc.tmemPhase = .allocated cols allocCg2 := by
          simp only [TmemPhase.subsumes, Bool.or_eq_true, beq_iff_eq] at htp
          rcases htp with h1 | h2; contradiction; exact h2.symm
        rw [hcp]; dsimp only at hstep ⊢
        by_cases hcg : (allocCg2 != isCg2) = true
        · rw [if_pos hcg] at hstep; contradiction
        · rw [if_neg hcg] at hstep ⊢
          cases hsmem : (smemIds.all fun s =>
              decide (s < abs.sharedProxy.length) && decide (s < abs.sharedPendingTma.length) &&
              (abs.sharedProxy.getD s .needsFence == .clean) &&
              (abs.sharedPendingTma.getD s []).isEmpty)
          · rw [hsmem] at hstep; contradiction
          · rw [hsmem] at hstep; simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte] at hstep
            have hcSmem : (smemIds.all fun s =>
                decide (s < conc.sharedProxy.length) && decide (s < conc.sharedPendingTma.length) &&
                (conc.sharedProxy.getD s .needsFence == .clean) &&
                (conc.sharedPendingTma.getD s []).isEmpty) = true := by
              apply List.all_eq_true.mpr
              intro s hsMem
              have hsA := List.all_eq_true.mp hsmem s hsMem
              simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hsA ⊢
              rcases hsA with ⟨⟨⟨hl1, hl2⟩, hcl⟩, hemp⟩
              exact ⟨⟨⟨hlenSP ▸ hl1, hlenPT ▸ hl2⟩,
                proxySub_clean_eq _ (hcl ▸ listSubsumes_getD (d := .needsFence) rfl s hsp)⟩,
                natSubset_isEmpty (listSubsumes_getD (sub := fun a c => natSubset c a) (d := []) (natSubset_refl []) s hpt) hemp⟩
            injection hstep with hstep; subst hstep
            simp only [hcSmem, Bool.not_true, Bool.false_eq_true, ↓reduceIte]
            refine ⟨_, rfl, ?_⟩
            unfold CtaState.subsumes; simp only [Bool.and_eq_true]
            exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
              hsp, hpt⟩, hsc⟩, hbp⟩, hbs⟩, tmemPhaseSub_refl _⟩, hac⟩, rfl⟩, rfl⟩, hcb⟩, rfl⟩, hpl⟩, rfl⟩, rfl⟩, hrp⟩, hvp⟩
  | tmemCommit barId isCg2 =>
      dsimp only [stepOp] at hstep ⊢
      rw [← hlenBS]
      by_cases hp1 : ((isCg2 && cfg.clusterDimX != 2) || !((blkGuard.hasZeroLeaderCheck && !blkGuard.hasUnknown) || (cfg.blockDimY == 1 && blkGuard.selectsAtMostOneThread)) || !(abs.mmaIssued || priorTmemIssueReaches cfg curBlk) || decide (barId ≥ abs.barState.length)) = true
      · rw [if_pos hp1] at hstep; contradiction
      · rw [if_neg hp1] at hstep
        have hp1f := Bool.eq_false_iff.mpr hp1
        simp only [Bool.or_eq_false_iff, Bool.not_eq_false'] at hp1f
        have hcMma : (conc.mmaIssued || priorTmemIssueReaches cfg curBlk) = true := by
          rcases Bool.or_eq_true_iff.mp hp1f.1.2 with h1 | h2
          · simp [hbImp hmi h1]
          · simp [h2]
        by_cases hp2 : (abs.barProxy.getD barId .needsFence != .clean) = true
        · rw [if_pos hp2] at hstep; contradiction
        · rw [if_neg hp2] at hstep
          have hp2eq := bne_eq_false_iff_eq.mp (Bool.eq_false_iff.mpr hp2)
          have hbpSub := listSubsumes_getD (d := ProxyState.needsFence) rfl barId hbp
          rw [hp2eq] at hbpSub
          have hbpC := proxySub_clean_eq _ hbpSub
          cases hap : abs.tmemPhase <;> rw [hap] at hstep htp <;> try contradiction
          rename_i cols allocCg2
          have hcp : conc.tmemPhase = .allocated cols allocCg2 := by
            simp only [TmemPhase.subsumes, Bool.or_eq_true, beq_iff_eq] at htp
            rcases htp with h1 | h2; contradiction; exact h2.symm
          have hbsB := listSubsumes_getD (d := .uninit) rfl barId hbs
          cases hab : abs.barState.getD barId .uninit <;> rw [hab] at hstep hbsB <;> try contradiction
          all_goals (
            dsimp only at hstep
            have hcbValid := mbarSub_validArrive _ _ hbsB (by simp)
            by_cases hcg : (allocCg2 != isCg2) = true
            · rw [if_pos hcg] at hstep; contradiction
            · rw [if_neg hcg] at hstep
              injection hstep with hstep; subst hstep
              have hcp1 : ((isCg2 && cfg.clusterDimX != 2) || !((blkGuard.hasZeroLeaderCheck && !blkGuard.hasUnknown) || (cfg.blockDimY == 1 && blkGuard.selectsAtMostOneThread)) || !(conc.mmaIssued || priorTmemIssueReaches cfg curBlk) || decide (barId ≥ abs.barState.length)) = false := by
                simp [hp1f.1.1.1, hp1f.1.1.2, hcMma, hp1f.2]
              rw [hcp1, hbpC, hcp]; simp only [bne_self_eq_false, Bool.false_eq_true, ↓reduceIte]
              have hset := listSubsumes_set (a := .armedPlain) (c := .armedPlain) rfl barId hbs
              have hcbIns := natSubset_insert barId hcb
              rcases hcbValid with hcv | hcv | hcv | hcv <;>
                (rw [hcv]; dsimp only; rw [if_neg hcg]; refine ⟨_, rfl, ?_⟩; unfold CtaState.subsumes; simp only [Bool.and_eq_true];
                 exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨hsp, hpt⟩, hsc⟩, hbp⟩, hset⟩, tmemPhaseSub_refl _⟩, hac⟩, rfl⟩, hmi⟩, hcbIns⟩, rfl⟩, hpl⟩, rfl⟩, rfl⟩, hrp⟩, hvp⟩))
  | tmemLoad ldId =>
      dsimp only [stepOp] at hstep ⊢
      cases hap : abs.tmemPhase <;> rw [hap] at hstep htp <;> try contradiction
      rename_i cols allocCg2
      have hcp : conc.tmemPhase = .allocated cols allocCg2 := by
        simp only [TmemPhase.subsumes, Bool.or_eq_true, beq_iff_eq] at htp
        rcases htp with h1 | h2; contradiction; exact h2.symm
      rw [hcp]; dsimp only at hstep ⊢
      by_cases hp1 : (!abs.allocClusterSynced || !abs.tmemReady || abs.mmaHazard) = true
      · rw [if_pos hp1] at hstep; contradiction
      · rw [if_neg hp1] at hstep
        have hp1f := Bool.eq_false_iff.mpr hp1
        simp only [Bool.or_eq_false_iff, Bool.not_eq_false'] at hp1f
        have hcAc := hbImp hac hp1f.1.1
        have hcTr := hbImp htr hp1f.1.2
        have hcMh : conc.mmaHazard = false := hbFalse hmh hp1f.2
        injection hstep with hstep; subst hstep
        simp only [hcAc, hcTr, hcMh, Bool.not_true, Bool.or_self, Bool.false_eq_true, ↓reduceIte]
        refine ⟨_, rfl, ?_⟩
        unfold CtaState.subsumes; simp only [Bool.and_eq_true]
        exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
          hsp, hpt⟩, hsc⟩, hbp⟩, hbs⟩, tmemPhaseSub_refl _⟩, by simp⟩, by simp⟩, hmi⟩, hcb⟩, by simp⟩,
          natSubset_insert ldId hpl⟩, rfl⟩, rfl⟩, hrp⟩, hvp⟩
  | tmemLoadWait =>
      dsimp only [stepOp] at hstep ⊢
      cases hap : abs.tmemPhase <;> rw [hap] at hstep htp <;> try contradiction
      rename_i cols allocCg2
      have hcp : conc.tmemPhase = .allocated cols allocCg2 := by
        simp only [TmemPhase.subsumes, Bool.or_eq_true, beq_iff_eq] at htp
        rcases htp with h1 | h2; contradiction; exact h2.symm
      rw [hcp]; dsimp only at hstep ⊢
      injection hstep with hstep; subst hstep
      refine ⟨_, rfl, ?_⟩
      unfold CtaState.subsumes; simp only [Bool.and_eq_true]
      exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
        hsp, hpt⟩, hsc⟩, hbp⟩, hbs⟩, tmemPhaseSub_refl _⟩, hac⟩, hmh⟩, hmi⟩, hcb⟩, htr⟩,
        natSubset_refl []⟩, hss⟩, hcs⟩, hrp⟩, hvp⟩
  | tmemLoadUse ldId =>
      dsimp only [stepOp] at hstep ⊢
      by_cases hin : ldId ∈ abs.pendingLoads
      · rw [if_pos hin] at hstep; contradiction
      · rw [if_neg hin] at hstep
        have hcNotIn : ¬(ldId ∈ conc.pendingLoads) := fun hm => hin (natSubset_mem hpl hm)
        injection hstep with hstep; subst hstep
        rw [if_neg hcNotIn]; exact ⟨conc, rfl, hsub⟩
  | tmemDealloc cols isCg2 =>
      dsimp only [stepOp] at hstep ⊢
      by_cases hp1 : ((isCg2 && cfg.clusterDimX != 2) || inCycle || (blkGuard.selectsSingleFullWarp cfg.blockDimX cfg.blockDimY).isNone) = true
      · rw [if_pos hp1] at hstep; contradiction
      · rw [if_neg hp1] at hstep ⊢
        by_cases hp2 : (abs.tmemPhase != .allocated cols isCg2) = true
        · rw [if_pos hp2] at hstep; contradiction
        · rw [if_neg hp2] at hstep
          have hp2eq := bne_eq_false_iff_eq.mp (Bool.eq_false_iff.mpr hp2)
          have hcp : conc.tmemPhase = .allocated cols isCg2 := by
            simp only [TmemPhase.subsumes, hp2eq, Bool.or_eq_true, beq_iff_eq] at htp
            rcases htp with h1 | h2; contradiction; exact h2.symm
          by_cases hp3 : (abs.mmaHazard || !abs.pendingLoads.isEmpty || !abs.syncedSinceTmemUse) = true
          · rw [if_pos hp3] at hstep; contradiction
          · rw [if_neg hp3] at hstep
            have hp3f := Bool.eq_false_iff.mpr hp3
            simp only [Bool.or_eq_false_iff, Bool.not_eq_false'] at hp3f
            have hcMh : conc.mmaHazard = false := hbFalse hmh hp3f.1.1
            have hcEmp : conc.pendingLoads.isEmpty = true := natSubset_isEmpty hpl hp3f.1.2
            have hcSs := hbImp hss hp3f.2
            by_cases hp4 : (isCg2 && (!abs.clusterSyncedSinceTmemUse || !abs.relinquishedPermitCg2)) = true
            · rw [if_pos hp4] at hstep; contradiction
            · rw [if_neg hp4] at hstep
              have hcCgF : (isCg2 && (!conc.clusterSyncedSinceTmemUse || !conc.relinquishedPermitCg2)) = false := by
                cases isCg2
                · rfl
                · have hp4f := Bool.eq_false_iff.mpr hp4
                  simp only [Bool.true_and, Bool.or_eq_false_iff, Bool.not_eq_false'] at hp4f
                  simp [hbImp hcs hp4f.1, hbImp hrp hp4f.2]
              injection hstep with hstep; subst hstep
              simp only [hcp, bne_self_eq_false, hcMh, hcEmp, hcSs, Bool.not_true, Bool.or_self, hcCgF, Bool.false_eq_true, ↓reduceIte]
              refine ⟨_, rfl, ?_⟩
              unfold CtaState.subsumes; simp only [Bool.and_eq_true]
              exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
                hsp, hpt⟩, hsc⟩, hbp⟩, hbs⟩, tmemPhaseSub_refl _⟩, hac⟩, by simp⟩, hmi⟩, hcb⟩, htr⟩, hpl⟩, by simp⟩, hcs⟩, hrp⟩, hvp⟩

/-- Monotonicity of `execBlockOps` across an entire basic block's operation sequence. -/
theorem execBlockOps_mono (cfg : ControlFlowCfg) (b : Nat) {abs conc outAbs : CtaState} (ops : List CfgOp)
    (hsub : abs.subsumes conc = true)
    (hops : execBlockOps cfg b abs ops = some outAbs) :
    ∃ outConc, execBlockOps cfg b conc ops = some outConc ∧ outAbs.subsumes outConc = true := by
  unfold execBlockOps execBlockOpsWith at hops ⊢
  have hsub0 : (if decide (b ∈ posLoopExitBodies cfg) && !(b ∈ abs.visitedPosBodies) then
      { abs with visitedPosBodies := insertSortedNat b abs.visitedPosBodies } else abs).subsumes
    (if decide (b ∈ posLoopExitBodies cfg) && !(b ∈ conc.visitedPosBodies) then
      { conc with visitedPosBodies := insertSortedNat b conc.visitedPosBodies } else conc) = true := by
    have hs := hsub; unfold CtaState.subsumes at hs; simp only [Bool.and_eq_true] at hs
    have hvp := hs.2
    cases hpos : decide (b ∈ posLoopExitBodies cfg)
    · simp only [Bool.false_and, Bool.false_eq_true, ↓reduceIte]; exact hsub
    · simp only [Bool.true_and]
      by_cases hba : b ∈ abs.visitedPosBodies <;> by_cases hbc : b ∈ conc.visitedPosBodies <;>
        simp only [hba, hbc, decide_true, decide_false, Bool.not_true, Bool.not_false, Bool.false_eq_true, ↓reduceIte]
      · exact hsub
      · unfold CtaState.subsumes; simp only [Bool.and_eq_true]
        refine ⟨hs.1, List.all_eq_true.mpr fun z hz => decide_eq_true ?_⟩
        rcases (mem_insertSortedNat _).mp hz with rfl | hzc
        · exact hba
        · exact natSubset_mem hvp hzc
      · exact False.elim (hba (natSubset_mem hvp hbc))
      · unfold CtaState.subsumes; simp only [Bool.and_eq_true]
        exact ⟨hs.1, natSubset_insert b hvp⟩
  let rec go_mono : ∀ {a0 c0 : CtaState} (os : List CfgOp),
      a0.subsumes c0 = true →
      execBlockOpsWith.go cfg b (activeGuardAt cfg b) (isCyclicBlockBool cfg b) a0 os = some outAbs →
      ∃ outConc, execBlockOpsWith.go cfg b (activeGuardAt cfg b) (isCyclicBlockBool cfg b) c0 os = some outConc ∧
        outAbs.subsumes outConc = true
    | a0, c0, [], hs0, hgo => by
        simp only [execBlockOpsWith.go, Option.some.injEq] at hgo ⊢
        subst hgo; exact ⟨c0, rfl, hs0⟩
    | a0, c0, op :: rest, hs0, hgo => by
        simp only [execBlockOpsWith.go] at hgo ⊢
        cases h1 : stepOp cfg b (activeGuardAt cfg b) (isCyclicBlockBool cfg b) a0 op with
        | none => rw [h1] at hgo; contradiction
        | some a1 =>
            rw [h1] at hgo
            obtain ⟨c1, hc1, hs1⟩ := stepOp_mono cfg b _ _ op hs0 h1
            rw [hc1]
            exact go_mono rest hs1 hgo
  exact go_mono ops hsub0 hops

/-- Boolean predicate confirming `conc` has visited `bodyBlk` when `cond = .posLoopExit bodyBlk`. -/
def branchCondPosEn (st : CtaState) : BranchCond → Bool
  | .posLoopExit bodyBlk => decide (bodyBlk ∈ st.visitedPosBodies)
  | _ => true

/-- Monotonicity of `stepEdge` under `CtaState.subsumes`. -/
theorem stepEdge_mono (cfg : ControlFlowCfg) (srcBlk : Nat) {abs conc nextAbs : CtaState} (cond : BranchCond)
    (hsub : abs.subsumes conc = true)
    (hconcEn : branchCondPosEn conc cond = true)
    (hstep : stepEdge cfg srcBlk abs cond = some nextAbs) :
    ∃ nextConc, stepEdge cfg srcBlk conc cond = some nextConc ∧ nextAbs.subsumes nextConc = true := by
  have hs := hsub; unfold CtaState.subsumes at hs; simp only [Bool.and_eq_true] at hs
  rcases hs with ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨hsp, hpt⟩, hsc⟩, hbp⟩, hbs⟩, htp⟩, hac⟩, hmh⟩, hmi⟩, hcb⟩, htr⟩, hpl⟩, hss⟩, hcs⟩, hrp⟩, hvp⟩
  have hbImp : ∀ {x y : Bool}, (!x || y) = true → x = true → y = true := by decide
  have hbFalse : ∀ {x y : Bool}, (!x || y) = true → y = false → x = false := by decide
  cases cond with
  | always | coord _ | uniform | unknown =>
      dsimp only [stepEdge, stepEdgeWith] at hstep ⊢
      injection hstep with hstep; subst hstep; exact ⟨conc, rfl, hsub⟩
  | posLoopExit bodyBlk =>
      dsimp only [stepEdge, stepEdgeWith, branchCondPosEn] at hconcEn hstep ⊢
      have hcMem := of_decide_eq_true hconcEn
      rw [if_pos hcMem]
      by_cases haMem : bodyBlk ∈ abs.visitedPosBodies
      · rw [if_pos haMem] at hstep; injection hstep with hstep; subst hstep; exact ⟨conc, rfl, hsub⟩
      · rw [if_neg haMem] at hstep; contradiction
  | mbarSpin barId =>
      dsimp only [stepEdge, stepEdgeWith] at hstep ⊢
      have hge := listSubsumes_getElem? barId hbs
      cases ha : abs.barState[barId]? with
      | none => rw [ha] at hstep; contradiction
      | some va =>
          cases hc : conc.barState[barId]? with
          | none => rw [ha, hc] at hge; contradiction
          | some vc =>
              rw [ha] at hstep; rw [ha, hc] at hge; dsimp only at hge
              have hva : va = .syncedIdle ∨ va = .armedPlain ∨ va = .armedTx ∨ va = .completed := by
                cases va <;> first | contradiction | simp
              have heqA : nextAbs = abs := by
                rcases hva with rfl | rfl | rfl | rfl <;> (injection hstep with h1; exact h1.symm)
              subst heqA
              rcases mbarSub_validArrive va vc hge hva with rfl | rfl | rfl | rfl <;>
                (dsimp only; exact ⟨conc, rfl, hsub⟩)
  | mbarReady barId initFree =>
      dsimp only [stepEdge, stepEdgeWith] at hstep ⊢
      have hge := listSubsumes_getElem? barId hbs
      cases ha : abs.barState[barId]? with
      | none => rw [ha] at hstep; contradiction
      | some va =>
          cases hc : conc.barState[barId]? with
          | none => rw [ha, hc] at hge; contradiction
          | some vc =>
              rw [ha] at hstep; rw [ha, hc] at hge; dsimp only at hge hstep ⊢
              rcases ite_none_some_iff.mp hstep with ⟨haReadyOk, hstep⟩
              subst hstep
              have hcReadyOk := mbarSub_readyOk va vc
                (initFree || (decide (barId ∈ signaledBarrierIds cfg) && (activeGuardAt cfg srcBlk).hasWarpFilter))
                hge haReadyOk
              refine ⟨_, ite_none_some_iff.mpr ⟨hcReadyOk, rfl⟩, ?_⟩
              have hptMap : listSubsumes (fun a c => natSubset c a)
                  (abs.sharedPendingTma.map (·.filter (· != barId)))
                  (conc.sharedPendingTma.map (·.filter (· != barId))) = true :=
                listSubsumes_map (sub := fun a c => natSubset c a) (fun _ _ h => natSubset_filter (· != barId) h) hpt
              have hbsSet := listSubsumes_set (a := .completed) (c := .completed) rfl barId hbs
              have hmemImp : (!decide (barId ∈ abs.tmemCommitBars) || decide (barId ∈ conc.tmemCommitBars)) = true := by
                by_cases hm : barId ∈ abs.tmemCommitBars
                · simp [hm, natSubset_mem hcb hm]
                · simp [hm]
              have htrUpd := tmemReadyStep_mono
                (decide (barId ∈ abs.tmemCommitBars))
                (decide (barId ∈ conc.tmemCommitBars))
                (decide (barId ∈ signaledBarrierIds cfg) && (activeGuardAt cfg srcBlk).hasWarpFilter && decide (barId ∈ committedTmemBarrierIds cfg))
                abs.mmaHazard conc.mmaHazard abs.tmemReady conc.tmemReady hmemImp hmh htr
              unfold CtaState.subsumes; simp only [Bool.and_eq_true]
              exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
                hsp, hptMap⟩, hsc⟩, hbp⟩, hbsSet⟩, htp⟩, hac⟩, hmh⟩, hmi⟩, hcb⟩,
                by simpa only [Bool.and_eq_true] using htrUpd⟩, hpl⟩, hss⟩, hcs⟩, hrp⟩, hvp⟩

/-- Propagate one sweep across all blocks joining `if/else` diamonds per `visitedPosBodies` phase. -/
def propagateOnce (cfg : ControlFlowCfg) (sig commitBars : List Nat)
    (metaList : List (Nat × Bool × GuardExpr × Bool × List (BranchCond × Nat)))
    (inv : StateMap) : StateMap :=
  metaList.foldl (fun acc (b, isPosBody, blkGuard, inCycle, succEdges) =>
    let cur := acc.getD b []
    let ops := (cfg.blockAt b).ops
    let hasWarp := blkGuard.hasWarpFilter
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
              match stepEdgeWith sig commitBars hasWarp outSt cond with
              | none => acc3
              | some nextSt => joinStateAt acc3 dst nextSt) acc2) acc) inv

/-- Iterate `propagateOnce` `fuel` times until the abstract `StateMap` fixpoint stabilizes. -/
def propagateLoop (cfg : ControlFlowCfg) (sig commitBars : List Nat)
    (metaList : List (Nat × Bool × GuardExpr × Bool × List (BranchCond × Nat))) :
    Nat → StateMap → StateMap
  | 0, inv => inv
  | fuel + 1, inv =>
      let next := propagateOnce cfg sig commitBars metaList inv
      if next == inv then inv else propagateLoop cfg sig commitBars metaList fuel next

/-- Infer the abstract inductive `StateMap` for `cfg` (`O(blocks)` space & time). -/
def inferStateMap (cfg : ControlFlowCfg) : StateMap :=
  match cfg.blocks.length with
  | 0 => []
  | n + 1 =>
      let posBodies := posLoopExitBodies cfg
      let sig := signaledBarrierIds cfg
      let commitBars := committedTmemBarrierIds cfg
      let metaList := (List.range (n + 1)).map fun b =>
        (b, decide (b ∈ posBodies), activeGuardAt cfg b, isCyclicBlockBool cfg b, ctaSuccs cfg b)
      let init : StateMap := [CtaState.initial cfg] :: List.replicate n []
      propagateLoop cfg sig commitBars metaList (4 * (n + 2)) init

/-- Verify in `O(blocks)` time that `inv` is an inductive abstract certificate for `cfg`. -/
def validStateMap (requireTmem : Bool) (cfg : ControlFlowCfg) (inv : StateMap) : Bool :=
  decide (0 < cfg.blocks.length) &&
  ((inv.getD 0 []).any fun s0 => s0.subsumes (CtaState.initial cfg)) &&
  let posBodies := posLoopExitBodies cfg
  let sig := signaledBarrierIds cfg
  let commitBars := committedTmemBarrierIds cfg
  (List.range cfg.blocks.length).all fun b =>
    let cur := inv.getD b []
    if cur.isEmpty then true
    else
      let blkGuard := activeGuardAt cfg b
      let inCycle := isCyclicBlockBool cfg b
      let isPos := decide (b ∈ posBodies)
      let hasWarp := blkGuard.hasWarpFilter
      let blk := cfg.blockAt b
      cur.all fun st =>
        match execBlockOpsCached cfg b isPos blkGuard inCycle st blk.ops with
        | none => false
        | some outSt =>
            (!blk.isReturn || validReturnState requireTmem outSt) &&
            ((enabledCtaSuccs cfg b outSt).all fun (cond, dst) =>
              decide (dst < cfg.blocks.length) &&
              match stepEdgeWith sig commitBars hasWarp outSt cond with
              | none => false
              | some nextSt => (inv.getD dst []).any fun dstSt => dstSt.subsumes nextSt)

/-- Inductive simulation theorem: every concrete reachable state `(b, st)` (`CtaReachable cfg b st`)
    is subsumed by an abstract summary `abs ∈ inv.getD b []`. -/
theorem ctaReachable_subsumed {requireTmem : Bool} {cfg : ControlFlowCfg} {inv : StateMap}
    {b : Nat} {st : CtaState}
    (hvalid : validStateMap requireTmem cfg inv = true)
    (hreach : CtaReachable cfg b st) :
    ∃ abs ∈ inv.getD b [], abs.subsumes st = true := by
  unfold validStateMap at hvalid
  rcases Bool.and_eq_true_iff.mp hvalid with ⟨h01, hallB⟩
  rcases Bool.and_eq_true_iff.mp h01 with ⟨_, hinit⟩
  induction hreach with
  | entry _ =>
      exact List.any_eq_true.mp hinit
  | @step blk dst st0 outSt nextSt cond _ hblk hops hedge _ htrans ih =>
      obtain ⟨abs0, habs0, hsub0⟩ := ih
      have hblk_ok := List.all_eq_true.mp hallB blk (List.mem_range.mpr hblk)
      dsimp only at hblk_ok
      have hne : (inv.getD blk []).isEmpty = false := by
        cases hl : inv.getD blk [] <;> simp_all
      rw [hne] at hblk_ok; simp only [Bool.false_eq_true, ↓reduceIte] at hblk_ok
      have hst_ok := List.all_eq_true.mp hblk_ok abs0 habs0
      rw [execBlockOpsCached_eq] at hst_ok
      cases hopsA : execBlockOps cfg blk abs0 (cfg.blockAt blk).ops with
      | none => rw [hopsA] at hst_ok; contradiction
      | some outAbs =>
          rw [hopsA] at hst_ok
          obtain ⟨outConc', hopsC', hsubOut⟩ := execBlockOps_mono cfg blk (cfg.blockAt blk).ops hsub0 hopsA
          rw [hops] at hopsC'; injection hopsC' with heqOut; subst heqOut
          rcases Bool.and_eq_true_iff.mp hst_ok with ⟨_, hedges⟩
          have hedgeA := enabledCtaSuccs_mono hsubOut hedge
          have hedge_ok := List.all_eq_true.mp hedges (cond, dst) hedgeA
          rcases Bool.and_eq_true_iff.mp hedge_ok with ⟨_, hstep_ok⟩
          have hse : stepEdgeWith (signaledBarrierIds cfg) (committedTmemBarrierIds cfg)
              (activeGuardAt cfg blk).hasWarpFilter outAbs cond = stepEdge cfg blk outAbs cond := rfl
          rw [hse] at hstep_ok
          cases htrA : stepEdge cfg blk outAbs cond with
          | none => rw [htrA] at hstep_ok; contradiction
          | some nextAbs =>
              rw [htrA] at hstep_ok
              obtain ⟨dstAbs, hdstMem, hdstSub⟩ := List.any_eq_true.mp hstep_ok
              have hconcEn : branchCondPosEn outSt cond = true := by
                unfold enabledCtaSuccs at hedge; rw [List.mem_filter] at hedge
                cases cond <;> exact hedge.2
              obtain ⟨nextConc', htrC', hsubNext⟩ := stepEdge_mono cfg blk cond hsubOut hconcEn htrA
              rw [htrans] at htrC'; injection htrC' with heqNext; subst heqNext
              exact ⟨dstAbs, hdstMem, ctaSubsumes_trans hdstSub hsubNext⟩

/-- Computable decision checker for `SafeCtaExecution requireTmem cfg`. -/
def checkCtaExecution (requireTmem : Bool) (cfg : ControlFlowCfg) : Bool :=
  validStateMap requireTmem cfg (inferStateMap cfg)

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
        have hse : stepEdgeWith (signaledBarrierIds cfg) (committedTmemBarrierIds cfg)
            (activeGuardAt cfg b).hasWarpFilter outAbs cond = stepEdge cfg b outAbs cond := rfl
        rw [hse] at hstep
        cases hsA : stepEdge cfg b outAbs cond with
        | none => rw [hsA] at hstep; contradiction
        | some nextAbs =>
            have hconcEn : branchCondPosEn outConc cond = true := by
              unfold enabledCtaSuccs at hedge; rw [List.mem_filter] at hedge
              cases cond <;> exact hedge.2
            obtain ⟨nextConc, htrC, _⟩ := stepEdge_mono cfg b cond hsubOut hconcEn hsA
            exact ⟨nextConc, htrC⟩

/-- Soundness of `checkCtaExecution`: returning `true` proves `SafeCtaExecution`. -/
theorem safeCtaExecution_of_check {requireTmem : Bool} {cfg : ControlFlowCfg}
    (hcheck : checkCtaExecution requireTmem cfg = true) :
    SafeCtaExecution requireTmem cfg :=
  safeCtaExecution_of_stateMap hcheck

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
  (!(decide (0 < cfg.numBarriers) && decide (tmemAllocSites cfg = [])) ||
    (validStateMap false cfg inv && checkLoopBackpressure cfg))

/-- Reusable soundness theorem for `uniform_barriers` using a precomputed `inv : StateMap`. -/
theorem uniform_barriers_of_stateMap {cfg : ControlFlowCfg} {inv : StateMap}
    (hcheck : checkUniformBarriersWith cfg inv = true) :
    UniformBarriers cfg := by
  unfold checkUniformBarriersWith at hcheck
  rcases Bool.and_eq_true_iff.mp hcheck with ⟨h0123, hasync⟩
  rcases Bool.and_eq_true_iff.mp h0123 with ⟨h012, hmbar⟩
  rcases Bool.and_eq_true_iff.mp h012 with ⟨h01, hcyc⟩
  rcases Bool.and_eq_true_iff.mp h01 with ⟨hcnt, hsync⟩
  rcases Bool.and_eq_true_iff.mp hcyc with ⟨hhasHdr, hloops⟩
  refine ⟨of_decide_eq_true hcnt, ?_, ⟨?_, ?_⟩, ?_, ?_⟩
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

/-- Check `TmemLifecycle cfg` against a precomputed `inv : StateMap` in `O(blocks)` time. -/
def checkTmemLifecycleWith (cfg : ControlFlowCfg) (inv : StateMap) : Bool :=
  let allocOk := match tmemAllocSites cfg with
    | [cols] =>
        validTmemColsBool cols &&
        decide (tmemDeallocSites cfg ≠ []) &&
        (tmemDeallocSites cfg).all (· == cols)
    | _ => false
  allocOk && validStateMap true cfg inv && checkLoopBackpressure cfg

/-- Reusable soundness theorem for `tmem_lifecycle` using a precomputed `inv : StateMap`. -/
theorem tmem_lifecycle_of_stateMap {cfg : ControlFlowCfg} {inv : StateMap}
    (hcheck : checkTmemLifecycleWith cfg inv = true) :
    TmemLifecycle cfg := by
  unfold checkTmemLifecycleWith at hcheck
  rcases Bool.and_eq_true_iff.mp hcheck with ⟨ha1, hloop⟩
  rcases Bool.and_eq_true_iff.mp ha1 with ⟨halloc, hvalid⟩
  refine ⟨?_, safeCtaExecution_of_stateMap hvalid, noUnbackpressuredLoop_of_check hloop⟩
  cases hs : tmemAllocSites cfg with
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

open Lean Elab Command Meta in
/-- Precomputes `cfgStateMap : StateMap` in compiled `MetaM` bytecode (`addDecl` certificate
    checked in Lean's kernel by `validStateMap` inside `tmem_lifecycle_of_stateMap`). -/
elab "cfg_def_state_map " c:ident : command => do
  Command.liftTermElabM do
    let ns ← getCurrNamespace
    let cExpr ← Term.elabTerm (← `($c:ident)) (some (mkConst ``ControlFlowCfg))
    let cVal ← unsafe evalExpr ControlFlowCfg (mkConst ``ControlFlowCfg) cExpr
    let smVal := inferStateMap cVal
    let stListTy ← Meta.mkAppM ``List #[mkConst ``CtaState]
    let smTy ← Meta.mkAppM ``List #[stListTy]
    addDecl <| Declaration.defnDecl {
      name := ns ++ `cfgStateMap, levelParams := [], type := smTy, value := toExpr smVal,
      hints := ReducibilityHints.abbrev, safety := DefinitionSafety.safe }

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
