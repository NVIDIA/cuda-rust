/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide.Contracts.ControlFlow

/-!
# Abstract `CtaState` Lattice & Monotonicity (`CudaOxide.Proofs.ControlFlow.Lattice`)
-/
namespace CudaOxide.Proofs.ControlFlow
open CudaOxide.Mir.ControlFlow
open CudaOxide.Contracts.ControlFlow

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

theorem listSubsumes_any_mem_false {barId : Nat} :
    ∀ {absL concL : List (List Nat)},
      listSubsumes (fun a c => natSubset c a) absL concL = true →
      (absL.any fun xs => decide (barId ∈ xs)) = false →
      (concL.any fun xs => decide (barId ∈ xs)) = false
  | [], [], _, _ => rfl
  | _ :: as_, _ :: cs, hsub, hany => by
      simp only [listSubsumes, Bool.and_eq_true] at hsub
      simp only [List.any_cons, Bool.or_eq_false_iff, decide_eq_false_iff_not] at hany ⊢
      exact ⟨fun hm => hany.1 (natSubset_mem hsub.1 hm), listSubsumes_any_mem_false hsub.2 hany.2⟩
  | [], _ :: _, hsub, _ | _ :: _, [], hsub, _ => by simp [listSubsumes] at hsub

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
  rcases h1 with ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨hp1, ht1⟩, hc1⟩, hbp1⟩, hbs1⟩, htp1⟩, hcl1⟩, hac1⟩, hmh1⟩, hmi1⟩, hcb1⟩, htcr1⟩, htr1⟩, hufw1⟩, hufl1⟩, hpl1⟩, hss1⟩, hcs1⟩, hrp1⟩, hvp1⟩
  rcases h2 with ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨hp2, ht2⟩, hc2⟩, hbp2⟩, hbs2⟩, htp2⟩, hcl2⟩, hac2⟩, hmh2⟩, hmi2⟩, hcb2⟩, htcr2⟩, htr2⟩, hufw2⟩, hufl2⟩, hpl2⟩, hss2⟩, hcs2⟩, hrp2⟩, hvp2⟩
  have hptT : listSubsumes (fun x y => natSubset y x) a.sharedPendingTma c.sharedPendingTma = true :=
    listSubsumes_trans (sub := fun x y => natSubset y x) (fun _ _ _ u v => natSubset_trans v u) ht1 ht2
  exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
    listSubsumes_trans proxySub_trans hp1 hp2, hptT⟩,
    listSubsumes_trans (sub := fun x y => !x || y) hbTrans hc1 hc2⟩,
    listSubsumes_trans proxySub_trans hbp1 hbp2⟩,
    listSubsumes_trans mbarSub_trans hbs1 hbs2⟩,
    tmemPhaseSub_trans htp1 htp2⟩,
    hbTrans _ _ _ hcl1 hcl2⟩,
    hbTrans _ _ _ hac1 hac2⟩,
    hbTrans _ _ _ hmh2 hmh1⟩,
    hbTrans _ _ _ hmi1 hmi2⟩,
    natSubset_trans hcb1 hcb2⟩,
    hbTrans _ _ _ htcr1 htcr2⟩,
    hbTrans _ _ _ htr1 htr2⟩,
    hbTrans _ _ _ hufw2 hufw1⟩,
    hbTrans _ _ _ hufl2 hufl1⟩,
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
  rcases hs with ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨_, _⟩, _⟩, _⟩, _⟩, htp⟩, _⟩, _⟩, hmh⟩, _⟩, _⟩, _⟩, _⟩, _⟩, _⟩, hpl⟩, _⟩, _⟩, _⟩, _⟩
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
  | posLoopExit bodyBlk =>                                -- Case 2: Positive-trip loop exit requires `bodyBlk ∈ visitedPosBodies`
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
  rcases hs with ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨hsp, hpt⟩, hsc⟩, hbp⟩, hbs⟩, htp⟩, hcl⟩, hac⟩, hmh⟩, hmi⟩, hcb⟩, htcr⟩, htr⟩, hufw⟩, hufl⟩, hpl⟩, hss⟩, hcs⟩, hrp⟩, hvp⟩
  have hlenSP := listSubsumes_length hsp
  have hlenPT := listSubsumes_length hpt
  have hlenSC := listSubsumes_length hsc
  have hlenBS := listSubsumes_length hbs
  have hbImp : ∀ {x y : Bool}, (!x || y) = true → x = true → y = true := by decide
  have hbFalse : ∀ {x y : Bool}, (!x || y) = true → y = false → x = false := by decide
  cases op with                                           -- Case-split on each hardware/memory `CfgOp`
  | syncThreads =>                                        -- Case 1: CTA `sync_threads()` advances proxy & mbarrier sync states
      unfold stepOp at hstep ⊢
      injection hstep with hstep; subst hstep
      refine ⟨_, rfl, ?_⟩
      have hcd : (!(!abs.mmaHazard && abs.pendingLoads.isEmpty) || (!conc.mmaHazard && conc.pendingLoads.isEmpty)) = true := by
        cases ha : (!abs.mmaHazard && abs.pendingLoads.isEmpty)
        · simp
        · simp only [Bool.and_eq_true, Bool.not_eq_true'] at ha
          simp [hbFalse hmh ha.1, natSubset_isEmpty hpl ha.2]
      have huflUpd : (!(if (!conc.mmaHazard && conc.pendingLoads.isEmpty) = true then false else conc.unFencedTmemLoad) ||
          (if (!abs.mmaHazard && abs.pendingLoads.isEmpty) = true then false else abs.unFencedTmemLoad)) = true := by
        cases hc : (!conc.mmaHazard && conc.pendingLoads.isEmpty)
        · have ha : (!abs.mmaHazard && abs.pendingLoads.isEmpty) = false := hbFalse hcd hc
          simp [ha, hufl]
        · simp
      unfold CtaState.subsumes; simp only [Bool.and_eq_true]
      exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
        listSubsumes_map proxyStepSync_mono hsp, hpt⟩, hsc⟩,
        listSubsumes_map proxyStepSync_mono hbp⟩,
        listSubsumes_map mbarStepSync_mono hbs⟩, htp⟩, hcl⟩, hac⟩, hmh⟩, hmi⟩, hcb⟩, htcr⟩, htr⟩, hufw⟩,
        by simpa only [Bool.and_eq_true] using huflUpd⟩, hpl⟩, hcd⟩, hcs⟩, hrp⟩, hvp⟩
  | clusterSync =>                                        -- Case 2: 2-CTA `cluster_sync()` enables DSMEM & `_cg2` TMEM access/dealloc
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
        have huflUpd : (!(if (!conc.mmaHazard && conc.pendingLoads.isEmpty) = true then false else conc.unFencedTmemLoad) ||
            (if (!abs.mmaHazard && abs.pendingLoads.isEmpty) = true then false else abs.unFencedTmemLoad)) = true := by
          cases hc : (!conc.mmaHazard && conc.pendingLoads.isEmpty)
          · have ha : (!abs.mmaHazard && abs.pendingLoads.isEmpty) = false := hbFalse hcd hc
            simp [ha, hufl]
          · simp
        have hrep : listSubsumes (fun a c => !a || c)
            (List.replicate abs.sharedClusterSynced.length true)
            (List.replicate conc.sharedClusterSynced.length true) = true := by
          rw [← hlenSC]; exact listSubsumes_refl (sub := fun a c => !a || c) (by decide) _
        unfold CtaState.subsumes; simp only [Bool.and_eq_true]
        exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
          listSubsumes_map proxyStepSync_mono hsp, hpt⟩, hrep⟩,
          listSubsumes_map proxyStepSync_mono hbp⟩,
          listSubsumes_map mbarStepSync_mono hbs⟩, htp⟩, rfl⟩, rfl⟩, hmh⟩, hmi⟩, hcb⟩, htcr⟩, htr⟩, hufw⟩,
          by simpa only [Bool.and_eq_true] using huflUpd⟩, hpl⟩, hcd⟩, hcd⟩, hrp⟩, hvp⟩
  | proxyFenceAsyncShared =>                              -- Case 3: `fence_proxy_async_shared_cta()` flushes generic writes to `.clean`
      dsimp only [stepOp] at hstep ⊢
      injection hstep with hstep; subst hstep
      refine ⟨_, rfl, ?_⟩
      unfold CtaState.subsumes; simp only [Bool.and_eq_true]
      exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
        listSubsumes_map proxyStepFence_mono hsp, hpt⟩, hsc⟩,
        listSubsumes_map proxyStepFence_mono hbp⟩, hbs⟩, htp⟩, hcl⟩, hac⟩, hmh⟩, hmi⟩, hcb⟩, htcr⟩, htr⟩, hufw⟩, hufl⟩, hpl⟩, hss⟩, hcs⟩, hrp⟩, hvp⟩
  | sharedRead s =>                                       -- Case 4: Generic-proxy shared read requires no pending TMA on `s`
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
          exact ⟨conc, rfl, hsub⟩
  | dsmemAccess s =>                                      -- Case 5: Cross-CTA DSMEM access requires `sharedClusterSynced[s]`
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
  | sharedWrite s =>                                      -- Case 6: Generic-proxy shared write dirties `sharedProxy[s]`
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
          have hspS := proxyMerge_mono _ _ (if blkGuard.isSingleLeader cfg.blockDimX cfg.blockDimY then .needsFence else .needsSyncAndFence) (listSubsumes_getD (d := .clean) rfl s hsp)
          unfold CtaState.subsumes; simp only [Bool.and_eq_true]
          exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
            listSubsumes_set hspS s hsp, hpt⟩,
            listSubsumes_set (sub := fun x y => !x || y) rfl s hsc⟩, hbp⟩, hbs⟩, htp⟩, hcl⟩, hac⟩, hmh⟩, hmi⟩, hcb⟩, htcr⟩, htr⟩, hufw⟩, hufl⟩, hpl⟩, hss⟩, hcs⟩, hrp⟩, hvp⟩
  | mbarInit barId count =>                               -- Case 7: `mbarrier_init` transitions `barState[barId]` to `.initNeedsSync`
      dsimp only [stepOp] at hstep ⊢
      rw [← hlenBS]
      by_cases hpre : (decide (barId ≥ abs.barState.length) || count == 0) = true
      · rw [if_pos hpre] at hstep; contradiction
      · rw [if_neg hpre] at hstep ⊢
        injection hstep with hstep; subst hstep
        refine ⟨_, rfl, ?_⟩
        have hbpB := proxyMerge_mono _ _ (if blkGuard.isSingleLeader cfg.blockDimX cfg.blockDimY then .needsFence else .needsSyncAndFence) (listSubsumes_getD (d := .clean) rfl barId hbp)
        unfold CtaState.subsumes; simp only [Bool.and_eq_true]
        exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
          hsp, hpt⟩, hsc⟩,
          listSubsumes_set hbpB barId hbp⟩,
          listSubsumes_set rfl barId hbs⟩, htp⟩, hcl⟩, hac⟩, hmh⟩, hmi⟩, hcb⟩, htcr⟩, htr⟩, hufw⟩, hufl⟩, hpl⟩, hss⟩, hcs⟩, hrp⟩, hvp⟩
  | mbarInval barId =>                                    -- Case 7b: `mbarrier_inval` resets `barState[barId]` to `.uninit`
      dsimp only [stepOp] at hstep ⊢
      rw [← hlenBS]
      by_cases hpre : barId ≥ abs.barState.length
      · rw [if_pos hpre] at hstep; contradiction
      · rw [if_neg hpre] at hstep ⊢
        injection hstep with hstep; subst hstep
        refine ⟨_, rfl, ?_⟩
        unfold CtaState.subsumes; simp only [Bool.and_eq_true]
        exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
          hsp, hpt⟩, hsc⟩, hbp⟩,
          listSubsumes_set rfl barId hbs⟩, htp⟩, hcl⟩, hac⟩, hmh⟩, hmi⟩, hcb⟩, htcr⟩, htr⟩, hufw⟩, hufl⟩, hpl⟩, hss⟩, hcs⟩, hrp⟩, hvp⟩
  | mbarArrive barId isCluster _ =>                       -- Case 8: `mbarrier_arrive*` checks fenced TMEM loads & arms `barId` (`.armedPlain`)
      dsimp only [stepOp] at hstep ⊢
      by_cases hp0 : ((isCluster && decide (cfg.clusterDimX ≤ 1)) || abs.unFencedTmemLoad) = true
      · rw [if_pos hp0] at hstep; contradiction
      · rw [if_neg hp0] at hstep
        have hp0f := Bool.eq_false_iff.mpr hp0
        simp only [Bool.or_eq_false_iff] at hp0f
        have hcufl := hbFalse hufl hp0f.2
        have hcp0 : ((isCluster && decide (cfg.clusterDimX ≤ 1)) || conc.unFencedTmemLoad) = false := by
          simp [hp0f.1, hcufl]
        rw [hcp0]; simp only [Bool.false_eq_true, ↓reduceIte]
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
                   exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨hsp, hpt⟩, hsc⟩, hbp⟩, hset⟩, htp⟩, hcl⟩, hac⟩, hmh⟩, hmi⟩, hcb⟩, htcr⟩, htr⟩, hufw⟩, hufl⟩, hpl⟩, hss⟩, hcs⟩, hrp⟩, hvp⟩)
  | mbarArriveExpectTx barId txBytes isCluster =>         -- Case 9: `mbarrier_arrive_expect_tx*` checks fenced TMEM loads & arms `barId` (`.armedTx`)
      dsimp only [stepOp] at hstep ⊢
      by_cases htx : ((txBytes == 0) || (isCluster && decide (cfg.clusterDimX ≤ 1)) || abs.unFencedTmemLoad) = true
      · rw [if_pos htx] at hstep; contradiction
      · rw [if_neg htx] at hstep
        have htxf := Bool.eq_false_iff.mpr htx
        simp only [Bool.or_eq_false_iff] at htxf
        have hcufl := hbFalse hufl htxf.2
        have hctx : ((txBytes == 0) || (isCluster && decide (cfg.clusterDimX ≤ 1)) || conc.unFencedTmemLoad) = false := by
          simp [htxf.1.1, htxf.1.2, hcufl]
        rw [hctx]; simp only [Bool.false_eq_true, ↓reduceIte]
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
                   exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨hsp, hpt⟩, hsc⟩, hbp⟩, hset⟩, htp⟩, hcl⟩, hac⟩, hmh⟩, hmi⟩, hcb⟩, htcr⟩, htr⟩, hufw⟩, hufl⟩, hpl⟩, hss⟩, hcs⟩, hrp⟩, hvp⟩)
  | tmaCopyG2S sharedId barId isCg2 isMc =>               -- Case 10: Async 2D TMA `g2s` checks `.clean` proxies & records pending TMA hazard
      dsimp only [stepOp] at hstep ⊢
      rw [← hlenSP, ← hlenPT, ← hlenBS]
      by_cases hp1 : ((isCg2 && cfg.clusterDimX != 2) || (isMc && decide (cfg.clusterDimX ≤ 1)) || !(blkGuard.isSingleLeader cfg.blockDimX cfg.blockDimY) || decide (sharedId ≥ abs.sharedProxy.length) || decide (sharedId ≥ abs.sharedPendingTma.length) || decide (barId ≥ abs.barState.length)) = true
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
          exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
            hsp, listSubsumes_set (sub := fun a c => natSubset c a) hptUpd sharedId hpt⟩,
            listSubsumes_set (sub := fun a c => !a || c) rfl sharedId hsc⟩, hbp⟩, hbs⟩, htp⟩, hcl⟩, hac⟩, hmh⟩, hmi⟩, hcb⟩, htcr⟩, htr⟩, hufw⟩, hufl⟩, hpl⟩, hss⟩, hcs⟩, hrp⟩, hvp⟩
  | clcTryCancel sharedId barId isMc =>                   -- Case 11: Blackwell `clc_try_cancel` checks `.armedTx` & records pending hazard
      dsimp only [stepOp] at hstep ⊢
      rw [← hlenSP, ← hlenPT, ← hlenBS]
      by_cases hp1 : (decide (cfg.clusterDimX < 1) || (isMc && decide (cfg.clusterDimX ≤ 1)) || !(blkGuard.isSingleLeader cfg.blockDimX cfg.blockDimY) || decide (sharedId ≥ abs.sharedProxy.length) || decide (sharedId ≥ abs.sharedPendingTma.length) || decide (barId ≥ abs.barState.length)) = true
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
            exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
              hsp, listSubsumes_set (sub := fun a c => natSubset c a) hptUpd sharedId hpt⟩,
              listSubsumes_set (sub := fun a c => !a || c) rfl sharedId hsc⟩, hbp⟩, hbs⟩, htp⟩, hcl⟩, hac⟩, hmh⟩, hmi⟩, hcb⟩, htcr⟩, htr⟩, hufw⟩, hufl⟩, hpl⟩, hss⟩, hcs⟩, hrp⟩, hvp⟩
  | tmemAlloc sharedId cols isCg2 =>                      -- Case 12: Full-warp `tcgen05_alloc*` checks `clusterLaunched` & transitions `.uninit` -> `.allocated`
      dsimp only [stepOp] at hstep ⊢
      rw [← hlenSP]
      by_cases hp1 : ((isCg2 && cfg.clusterDimX != 2) || inCycle || decide (sharedId ≥ abs.sharedProxy.length) || !validTmemColsBool cols) = true
      · rw [if_pos hp1] at hstep; contradiction
      · rw [if_neg hp1] at hstep ⊢
        by_cases hp2 : (blkGuard.selectsSingleFullWarp cfg.blockDimX cfg.blockDimY).isNone = true
        · rw [if_pos hp2] at hstep; contradiction
        · rw [if_neg hp2] at hstep ⊢
          by_cases hp3 : ((isCg2 && !abs.clusterLaunched) || abs.tmemPhase != .uninit) = true
          · rw [if_pos hp3] at hstep; contradiction
          · rw [if_neg hp3] at hstep
            have hp3f := Bool.eq_false_iff.mpr hp3
            simp only [Bool.or_eq_false_iff, bne_eq_false_iff_eq] at hp3f
            have hcCl : (isCg2 && !conc.clusterLaunched) = false := by
              cases isCg2
              · rfl
              · simp only [Bool.true_and, Bool.not_eq_false'] at hp3f ⊢
                exact hbImp hcl hp3f.1
            have hcp : conc.tmemPhase = .uninit := by
              simp only [TmemPhase.subsumes, hp3f.2, Bool.or_eq_true, beq_iff_eq] at htp
              rcases htp with h1 | h2; contradiction; exact h2.symm
            injection hstep with hstep; subst hstep
            simp only [hcCl, hcp, bne_self_eq_false, Bool.or_self, Bool.false_eq_true, ↓reduceIte]
            refine ⟨_, rfl, ?_⟩
            unfold CtaState.subsumes; simp only [Bool.and_eq_true]
            exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
              hsp, hpt⟩, hsc⟩, hbp⟩, hbs⟩, tmemPhaseSub_refl _⟩, hcl⟩, by cases isCg2 <;> rfl⟩,
              hmh⟩, hmi⟩, hcb⟩, htcr⟩, htr⟩, hufw⟩, hufl⟩, hpl⟩, rfl⟩, rfl⟩, rfl⟩, hvp⟩
  | tmemRelinquishPermit isCg2 =>                         -- Case 13: Full-warp `tcgen05_relinquish_alloc_permit_cg2` before dealloc
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
            exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
              hsp, hpt⟩, hsc⟩, hbp⟩, hbs⟩, tmemPhaseSub_refl _⟩, hcl⟩, hac⟩, hmh⟩, hmi⟩, hcb⟩, htcr⟩, htr⟩, hufw⟩, hufl⟩, hpl⟩, hss⟩, hcs⟩,
              by cases isCg2 <;> rfl⟩, hvp⟩
  | tmemAsyncIssue smemIds isCg2 =>                       -- Case 14: Leader `tcgen05_mma*` checks fenced wait + clean SMEM & sets `mmaHazard`
      dsimp only [stepOp] at hstep ⊢
      by_cases hp1 : ((isCg2 && cfg.clusterDimX != 2) || !(blkGuard.isSingleLeader cfg.blockDimX cfg.blockDimY) || !abs.allocClusterSynced || abs.unFencedMbarWait) = true
      · rw [if_pos hp1] at hstep; contradiction
      · rw [if_neg hp1] at hstep
        have hp1f := Bool.eq_false_iff.mpr hp1
        simp only [Bool.or_eq_false_iff, Bool.not_eq_false'] at hp1f
        have hcAc := hbImp hac hp1f.1.2
        have hcUfw := hbFalse hufw hp1f.2
        have hcp1 : ((isCg2 && cfg.clusterDimX != 2) || !(blkGuard.isSingleLeader cfg.blockDimX cfg.blockDimY) || !conc.allocClusterSynced || conc.unFencedMbarWait) = false := by
          simp [hp1f.1.1.1, hp1f.1.1.2, hcAc, hcUfw]
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
              rcases hsA with ⟨⟨⟨hl1, hl2⟩, hcl'⟩, hemp⟩
              exact ⟨⟨⟨hlenSP ▸ hl1, hlenPT ▸ hl2⟩,
                proxySub_clean_eq _ (hcl' ▸ listSubsumes_getD (d := .needsFence) rfl s hsp)⟩,
                natSubset_isEmpty (listSubsumes_getD (sub := fun a c => natSubset c a) (d := []) (natSubset_refl []) s hpt) hemp⟩
            injection hstep with hstep; subst hstep
            simp only [hcSmem, Bool.not_true, Bool.false_eq_true, ↓reduceIte]
            refine ⟨_, rfl, ?_⟩
            unfold CtaState.subsumes; simp only [Bool.and_eq_true]
            exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
              hsp, hpt⟩, hsc⟩, hbp⟩, hbs⟩, tmemPhaseSub_refl _⟩, hcl⟩, hac⟩, rfl⟩, rfl⟩, hcb⟩, rfl⟩, rfl⟩, hufw⟩, hufl⟩, hpl⟩, rfl⟩, rfl⟩, hrp⟩, hvp⟩
  | tmemCommit barId isCg2 isMc =>                        -- Case 15: `tcgen05_commit*` ties prior MMA to `barId` & clears `mmaHazard`
      dsimp only [stepOp] at hstep ⊢
      rw [← hlenBS]
      by_cases hp1 : ((isCg2 && cfg.clusterDimX != 2) || (isMc && decide (cfg.clusterDimX ≤ 1)) || !(blkGuard.isSingleLeader cfg.blockDimX cfg.blockDimY) || !(abs.mmaIssued || priorTmemIssueReaches cfg curBlk) || decide (barId ≥ abs.barState.length)) = true
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
              have hcp1 : ((isCg2 && cfg.clusterDimX != 2) || (isMc && decide (cfg.clusterDimX ≤ 1)) || !(blkGuard.isSingleLeader cfg.blockDimX cfg.blockDimY) || !(conc.mmaIssued || priorTmemIssueReaches cfg curBlk) || decide (barId ≥ abs.barState.length)) = false := by
                simp [hp1f.1.1.1.1, hp1f.1.1.1.2, hp1f.1.1.2, hcMma, hp1f.2]
              rw [hcp1, hbpC, hcp]; simp only [bne_self_eq_false, Bool.false_eq_true, ↓reduceIte]
              have hset := listSubsumes_set (a := .armedPlain) (c := .armedPlain) rfl barId hbs
              have hcbIns := natSubset_insert barId hcb
              rcases hcbValid with hcv | hcv | hcv | hcv <;>
                (rw [hcv]; dsimp only; rw [if_neg hcg]; refine ⟨_, rfl, ?_⟩; unfold CtaState.subsumes; simp only [Bool.and_eq_true];
                 exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨hsp, hpt⟩, hsc⟩, hbp⟩, hset⟩, tmemPhaseSub_refl _⟩, hcl⟩, hac⟩, rfl⟩, hmi⟩, hcbIns⟩, rfl⟩, rfl⟩, hufw⟩, hufl⟩, hpl⟩, rfl⟩, rfl⟩, hrp⟩, hvp⟩))
  | tmemLoad ldId =>                                      -- Case 16: `tcgen05_ld` checks warp convergence + `tmemReady && !mmaHazard && !unFencedMbarWait`
      dsimp only [stepOp] at hstep ⊢
      by_cases hwc : (!blkGuard.isWarpConvergent cfg.blockDimX cfg.blockDimY) = true
      · rw [if_pos hwc] at hstep; contradiction
      · rw [if_neg hwc] at hstep ⊢
        cases hap : abs.tmemPhase <;> rw [hap] at hstep htp <;> try contradiction
        rename_i cols allocCg2
        have hcp : conc.tmemPhase = .allocated cols allocCg2 := by
          simp only [TmemPhase.subsumes, Bool.or_eq_true, beq_iff_eq] at htp
          rcases htp with h1 | h2; contradiction; exact h2.symm
        rw [hcp]; dsimp only at hstep ⊢
        by_cases hp1 : (!abs.allocClusterSynced || !abs.tmemReady || abs.mmaHazard || abs.unFencedMbarWait) = true
        · rw [if_pos hp1] at hstep; contradiction
        · rw [if_neg hp1] at hstep
          have hp1f := Bool.eq_false_iff.mpr hp1
          simp only [Bool.or_eq_false_iff, Bool.not_eq_false'] at hp1f
          have hcAc := hbImp hac hp1f.1.1.1
          have hcTr := hbImp htr hp1f.1.1.2
          have hcMh : conc.mmaHazard = false := hbFalse hmh hp1f.1.2
          have hcUfw : conc.unFencedMbarWait = false := hbFalse hufw hp1f.2
          injection hstep with hstep; subst hstep
          simp only [hcAc, hcTr, hcMh, hcUfw, Bool.not_true, Bool.or_self, Bool.false_eq_true, ↓reduceIte]
          refine ⟨_, rfl, ?_⟩
          unfold CtaState.subsumes; simp only [Bool.and_eq_true]
          exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
            hsp, hpt⟩, hsc⟩, hbp⟩, hbs⟩, tmemPhaseSub_refl _⟩, hcl⟩, by simp⟩, by simp⟩, hmi⟩, hcb⟩, htcr⟩, by simp⟩, by simp⟩, rfl⟩,
            natSubset_insert ldId hpl⟩, rfl⟩, rfl⟩, hrp⟩, hvp⟩
  | tmemLoadWait =>                                       -- Case 17: `tcgen05_load_wait()` drains `pendingLoads` to `[]`
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
      exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
        hsp, hpt⟩, hsc⟩, hbp⟩, hbs⟩, tmemPhaseSub_refl _⟩, hcl⟩, hac⟩, hmh⟩, hmi⟩, hcb⟩, htcr⟩, htr⟩, hufw⟩, hufl⟩,
        natSubset_refl []⟩, hss⟩, hcs⟩, hrp⟩, hvp⟩
  | tmemLoadUse ldId =>                                   -- Case 18: Register read of `ldId` requires `ldId ∉ pendingLoads`
      dsimp only [stepOp] at hstep ⊢
      by_cases hin : ldId ∈ abs.pendingLoads
      · rw [if_pos hin] at hstep; contradiction
      · rw [if_neg hin] at hstep
        have hcNotIn : ¬(ldId ∈ conc.pendingLoads) := fun hm => hin (natSubset_mem hpl hm)
        injection hstep with hstep; subst hstep
        rw [if_neg hcNotIn]; exact ⟨conc, rfl, hsub⟩
  | tmemStoreWait =>                                      -- Case 18b: `tcgen05_store_wait()` checks warp convergence + allocated `TMEM`
      dsimp only [stepOp] at hstep ⊢
      by_cases hwc : (!blkGuard.isWarpConvergent cfg.blockDimX cfg.blockDimY) = true
      · rw [if_pos hwc] at hstep; contradiction
      · rw [if_neg hwc] at hstep ⊢
        cases hap : abs.tmemPhase <;> rw [hap] at hstep htp <;> try contradiction
        rename_i cols allocCg2
        have hcp : conc.tmemPhase = .allocated cols allocCg2 := by
          simp only [TmemPhase.subsumes, Bool.or_eq_true, beq_iff_eq] at htp
          rcases htp with h1 | h2; contradiction; exact h2.symm
        rw [hcp]; dsimp only at hstep ⊢
        injection hstep with hstep; subst hstep
        exact ⟨conc, rfl, hsub⟩
  | warpSyncFull =>                                       -- Case 18b': Full-warp `*.sync 0xFFFF_FFFF` checks 32-lane warp convergence
      dsimp only [stepOp] at hstep ⊢
      by_cases hwc : (!blkGuard.isWarpConvergent cfg.blockDimX cfg.blockDimY) = true
      · rw [if_pos hwc] at hstep; contradiction
      · rw [if_neg hwc] at hstep ⊢
        injection hstep with hstep; subst hstep
        exact ⟨conc, rfl, hsub⟩
  | tmemFenceBeforeSync =>                                -- Case 18c: `tcgen05_fence_before_thread_sync` checks warp convergence + retired `ld`s
      dsimp only [stepOp] at hstep ⊢
      by_cases hp1 : (!blkGuard.isWarpConvergent cfg.blockDimX cfg.blockDimY || !abs.pendingLoads.isEmpty) = true
      · rw [if_pos hp1] at hstep; contradiction
      · rw [if_neg hp1] at hstep
        have hp1f := Bool.eq_false_iff.mpr hp1
        simp only [Bool.or_eq_false_iff, Bool.not_eq_false'] at hp1f
        have hcEmp := natSubset_isEmpty hpl hp1f.2
        have hcp1 : (!blkGuard.isWarpConvergent cfg.blockDimX cfg.blockDimY || !conc.pendingLoads.isEmpty) = false := by
          simp [hp1f.1, hcEmp]
        rw [hcp1]; simp only [Bool.false_eq_true, ↓reduceIte]
        cases hap : abs.tmemPhase <;> rw [hap] at hstep htp <;> try contradiction
        rename_i cols allocCg2
        have hcp : conc.tmemPhase = .allocated cols allocCg2 := by
          simp only [TmemPhase.subsumes, Bool.or_eq_true, beq_iff_eq] at htp
          rcases htp with h1 | h2; contradiction; exact h2.symm
        rw [hcp]; dsimp only at hstep ⊢
        injection hstep with hstep; subst hstep
        refine ⟨_, rfl, ?_⟩
        unfold CtaState.subsumes; simp only [Bool.and_eq_true]
        exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
          hsp, hpt⟩, hsc⟩, hbp⟩, hbs⟩, tmemPhaseSub_refl _⟩, hcl⟩, hac⟩, hmh⟩, hmi⟩, hcb⟩, htcr⟩, htr⟩, hufw⟩, rfl⟩, hpl⟩, hss⟩, hcs⟩, hrp⟩, hvp⟩
  | tmemFenceAfterSync =>                                 -- Case 18d: `tcgen05_fence_after_thread_sync` promotes `tmemCommitReady` -> `tmemReady` & clears `unFencedMbarWait`
      dsimp only [stepOp] at hstep ⊢
      cases hap : abs.tmemPhase <;> rw [hap] at hstep htp <;> try contradiction
      rename_i cols allocCg2
      have hcp : conc.tmemPhase = .allocated cols allocCg2 := by
        simp only [TmemPhase.subsumes, Bool.or_eq_true, beq_iff_eq] at htp
        rcases htp with h1 | h2; contradiction; exact h2.symm
      rw [hcp]; dsimp only at hstep ⊢
      injection hstep with hstep; subst hstep
      refine ⟨_, rfl, ?_⟩
      have htrUpd := tmemReadyStep_mono abs.tmemCommitReady conc.tmemCommitReady false abs.mmaHazard conc.mmaHazard abs.tmemReady conc.tmemReady htcr hmh htr
      unfold CtaState.subsumes; simp only [Bool.and_eq_true]
      exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
        hsp, hpt⟩, hsc⟩, hbp⟩, hbs⟩, tmemPhaseSub_refl _⟩, hcl⟩, hac⟩, hmh⟩, hmi⟩, hcb⟩, htcr⟩,
        by simpa using htrUpd⟩, rfl⟩, hufl⟩, hpl⟩, hss⟩, hcs⟩, hrp⟩, hvp⟩
  | tmemDealloc cols isCg2 =>                             -- Case 19: Full-warp `tcgen05_dealloc*` checks sync + permit & transitions `.deallocated`
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
              exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
                hsp, hpt⟩, hsc⟩, hbp⟩, hbs⟩, tmemPhaseSub_refl _⟩, hcl⟩, hac⟩, by simp⟩, hmi⟩, hcb⟩, htcr⟩, htr⟩, hufw⟩, hufl⟩, hpl⟩, by simp⟩, hcs⟩, hrp⟩, hvp⟩

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
  rcases hs with ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨hsp, hpt⟩, hsc⟩, hbp⟩, hbs⟩, htp⟩, hcl⟩, hac⟩, hmh⟩, hmi⟩, hcb⟩, htcr⟩, htr⟩, hufw⟩, hufl⟩, hpl⟩, hss⟩, hcs⟩, hrp⟩, hvp⟩
  have hbImp : ∀ {x y : Bool}, (!x || y) = true → x = true → y = true := by decide
  have hbFalse : ∀ {x y : Bool}, (!x || y) = true → y = false → x = false := by decide
  cases cond with                                         -- Case-split on outgoing `BranchCond` edge
  | always | coord _ | uniform | warpUniform | unknown => -- Case 1: Non-`mbarrier` branch preserves `CtaState` unchanged
      dsimp only [stepEdge, stepEdgeWith] at hstep ⊢
      injection hstep with hstep; subst hstep; exact ⟨conc, rfl, hsub⟩
  | posLoopExit bodyBlk =>
      dsimp only [stepEdge, stepEdgeWith, branchCondPosEn] at hconcEn hstep ⊢
      have hcMem := of_decide_eq_true hconcEn
      rw [if_pos hcMem]
      by_cases haMem : bodyBlk ∈ abs.visitedPosBodies
      · rw [if_pos haMem] at hstep; injection hstep with hstep; subst hstep; exact ⟨conc, rfl, hsub⟩
      · rw [if_neg haMem] at hstep; contradiction
  | mbarSpin barId =>                                     -- Case 3: `mbarrier` spin-wait backedge requires initialized barrier
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
  | mbarReady barId initFree waitKind =>                  -- Case 4: `mbarrier` ready edge checks scope & retires pending TMA/MMA on `barId`
      dsimp only [stepEdge, stepEdgeWith] at hstep ⊢
      by_cases hclScope : (decide (barId ∈ clusterSignaledBarrierIds cfg) && waitKind == .ctaAcquire) = true
      · rw [if_pos hclScope] at hstep; contradiction
      · rw [if_neg hclScope] at hstep ⊢
        by_cases hrelTma : (waitKind == .clusterRelaxed && (abs.sharedPendingTma.any fun xs => decide (barId ∈ xs))) = true
        · rw [if_pos hrelTma] at hstep; contradiction
        · rw [if_neg hrelTma] at hstep
          have hrelC : (waitKind == .clusterRelaxed && (conc.sharedPendingTma.any fun xs => decide (barId ∈ xs))) = false := by
            cases hwk : waitKind == .clusterRelaxed
            · rfl
            · have hf := Bool.eq_false_iff.mpr hrelTma
              rw [hwk] at hf
              rw [Bool.true_and] at hf ⊢
              exact listSubsumes_any_mem_false hpt hf
          rw [hrelC]; simp only [Bool.false_eq_true, ↓reduceIte]
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
                    (initFree || decide (barId ∈ warpWaitSignaledBarsAt cfg srcBlk))
                    hge haReadyOk
                  refine ⟨_, ite_none_some_iff.mpr ⟨hcReadyOk, rfl⟩, ?_⟩
                  have hptNext : listSubsumes (fun a c => natSubset c a)
                      (if waitKind == .clusterRelaxed then abs.sharedPendingTma else abs.sharedPendingTma.map (·.filter (· != barId)))
                      (if waitKind == .clusterRelaxed then conc.sharedPendingTma else conc.sharedPendingTma.map (·.filter (· != barId))) = true := by
                    cases waitKind == .clusterRelaxed
                    · exact listSubsumes_map (sub := fun a c => natSubset c a) (fun _ _ h => natSubset_filter (· != barId) h) hpt
                    · exact hpt
                  have hbsSet := listSubsumes_set (a := .completed) (c := .completed) rfl barId hbs
                  have hmemImp : (!decide (barId ∈ abs.tmemCommitBars) || decide (barId ∈ conc.tmemCommitBars)) = true := by
                    by_cases hm : barId ∈ abs.tmemCommitBars
                    · simp [hm, natSubset_mem hcb hm]
                    · simp [hm]
                  have htcrUpd := tmemReadyStep_mono
                    (decide (barId ∈ abs.tmemCommitBars))
                    (decide (barId ∈ conc.tmemCommitBars))
                    (decide (barId ∈ warpWaitSignaledBarsAt cfg srcBlk) && decide (barId ∈ committedTmemBarrierIds cfg))
                    abs.mmaHazard conc.mmaHazard abs.tmemCommitReady conc.tmemCommitReady hmemImp hmh htcr
                  unfold CtaState.subsumes; simp only [Bool.and_eq_true]
                  exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
                    hsp, hptNext⟩, hsc⟩, hbp⟩, hbsSet⟩, htp⟩, hcl⟩, hac⟩, hmh⟩, hmi⟩, hcb⟩,
                    by simpa only [Bool.and_eq_true] using htcrUpd⟩, htr⟩, rfl⟩, hufl⟩, hpl⟩, hss⟩, hcs⟩, hrp⟩, hvp⟩


end CudaOxide.Proofs.ControlFlow
