/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide.Proofs.Program.Invariant

/-!
# Verification Condition Bridge Theorems (`CudaOxide.Proofs.Program.Vc`)
-/
namespace CudaOxide.Proofs.Program
open CudaOxide.Mir.Program
open CudaOxide.Mir.ControlFlow
open CudaOxide.Contracts.Program
open CudaOxide.Contracts.ControlFlow

/-- Helper: extract a list-conjunction obligation at `x ∈ xs`. -/
theorem foldr_and_of_mem {α : Type} {P : α → Prop} :
    ∀ xs : List α, xs.foldr (fun x acc => P x ∧ acc) True → ∀ x ∈ xs, P x
  | [], _, _, hmem => nomatch hmem
  | _ :: _, ⟨hp, _⟩, _, List.Mem.head _ => hp
  | _ :: tl, ⟨_, htl⟩, x, List.Mem.tail _ hm => foldr_and_of_mem tl htl x hm

/-- Remove exact duplicate symbolic events before generating VC obligations. -/
def dedupEvents {α : Type} [DecidableEq α] : List α → List α
  | [] => []
  | x :: xs =>
      let rest := dedupEvents xs
      if decide (x ∈ rest) then rest else x :: rest

theorem mem_dedupEvents {α : Type} [DecidableEq α] {x : α} :
    ∀ {xs : List α}, x ∈ xs → x ∈ dedupEvents xs
  | [], h => nomatch h
  | y :: tl, h => by
      dsimp only [dedupEvents]
      rcases List.mem_cons.mp h with rfl | htl
      · split
        · rename_i hin; exact of_decide_eq_true hin
        · exact List.Mem.head _
      · have ih := mem_dedupEvents htl
        split
        · exact ih
        · exact List.Mem.tail _ ih

/-- Helper: bound `(base + k) / elemDiv < bound` from either `base < bound` (`span = 1, elemDiv = 1`)
    or `base + span ≤ bound * elemDiv` (`k < span, 0 < elemDiv`). -/
def symSpanInBounds (sev : SymMemEvent) (base bound : Nat) : Prop :=
  if sev.span == 1 && sev.elemDiv == 1 then base < bound
  else base + sev.span ≤ bound * sev.elemDiv

/-- Soundness of `symSpanInBounds`: implies `(base + k) / sev.elemDiv < bound` for every `k < sev.span`. -/
theorem symSpanInBounds_sound {sev : SymMemEvent} {base bound k : Nat}
    (hk : k < sev.span) (hdiv : 0 < sev.elemDiv)
    (hin : symSpanInBounds sev base bound) :
    (base + k) / sev.elemDiv < bound := by
  unfold symSpanInBounds at hin
  split at hin                                              -- Split: 1-element pointer (`span = 1, elemDiv = 1`) vs. byte span
  · rename_i h11
    rcases Bool.and_eq_true_iff.mp h11 with ⟨hs1, hd1⟩
    have hs1' : sev.span = 1 := beq_iff_eq.mp hs1
    have hd1' : sev.elemDiv = 1 := beq_iff_eq.mp hd1
    have hk0 : k = 0 := by omega                            -- `k < 1` forces `k = 0`
    rw [hk0, hd1', Nat.add_zero, Nat.div_one]; exact hin    -- `(base + 0) / 1 = base < bound`
  · have hlt : base + k < bound * sev.elemDiv := by omega   -- Span case: `base + k < base + span ≤ bound * elemDiv`
    exact (Nat.div_lt_iff_lt_mul hdiv).mpr hlt              -- Divide both sides by `elemDiv > 0`

/-- Extract `some slotSym` when `hyp` syntactically entails `.mbarReady barId slotSym` (`fuel`-bounded). -/
def findMbarReadySlotFuel : Nat → SymBool → Nat → Option SymInt
  | 0, _, _ => none
  | fuel + 1, hyp, barId =>
      match hyp with
      | .mbarReady b s => if b == barId then some s else none
      | .and p q =>
          match findMbarReadySlotFuel fuel p barId with
          | some s => some s
          | none => findMbarReadySlotFuel fuel q barId
      | .or p q =>
          match findMbarReadySlotFuel fuel p barId, findMbarReadySlotFuel fuel q barId with
          | some s1, some s2 => if s1 == s2 then some s1 else none
          | _, _ => none
      | .not (.not p) => findMbarReadySlotFuel fuel p barId
      | .not (.or p q) =>
          match findMbarReadySlotFuel fuel (.not p) barId with
          | some s => some s
          | none => findMbarReadySlotFuel fuel (.not q) barId
      | _ => none

/-- Extract `some slotSym` when `hyp` entails `mbarReady barId slotSym`. -/
def findMbarReadySlot (hyp : SymBool) (barId : Nat) : Option SymInt :=
  findMbarReadySlotFuel 32 hyp barId

/-- Soundness of `findMbarReadySlotFuel`. -/
theorem findMbarReadySlotFuel_sound {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat} :
    ∀ (fuel : Nat) (hyp : SymBool) (barId : Nat) (s : SymInt),
      findMbarReadySlotFuel fuel hyp barId = some s →
      hyp.eval env t havoc = true →
      env.mbarFlag barId (s.eval env t havoc) = true
  | 0, _, _, _, hfind, _ => by contradiction
  | fuel + 1, hyp, barId, s, hfind, hhyp => by
      cases hyp with
      | mbarReady b s0 =>
          dsimp only [findMbarReadySlotFuel] at hfind
          split at hfind
          · rename_i heq
            injection hfind with hs; subst hs
            rw [← beq_iff_eq.mp heq]
            simpa [SymBool.eval] using hhyp
          · contradiction
      | and p q =>
          dsimp only [findMbarReadySlotFuel] at hfind
          simp only [SymBool.eval, Bool.and_eq_true] at hhyp
          cases hp : findMbarReadySlotFuel fuel p barId with
          | some sp =>
              rw [hp] at hfind; injection hfind with hs; subst hs
              exact findMbarReadySlotFuel_sound fuel p barId sp hp hhyp.1
          | none =>
              rw [hp] at hfind; dsimp only at hfind
              exact findMbarReadySlotFuel_sound fuel q barId s hfind hhyp.2
      | or p q =>
          dsimp only [findMbarReadySlotFuel] at hfind
          cases hp : findMbarReadySlotFuel fuel p barId <;>
            cases hq : findMbarReadySlotFuel fuel q barId <;>
            rw [hp, hq] at hfind
          all_goals try contradiction
          dsimp only at hfind
          split at hfind
          · rename_i hsEq
            injection hfind with hs; subst hs
            simp only [SymBool.eval, Bool.or_eq_true] at hhyp
            rcases hhyp with hpv | hqv
            · exact findMbarReadySlotFuel_sound fuel p barId _ hp hpv
            · rw [beq_iff_eq.mp hsEq]
              exact findMbarReadySlotFuel_sound fuel q barId _ hq hqv
          · contradiction
      | not inner =>
          cases inner with
          | not p =>
              dsimp only [findMbarReadySlotFuel] at hfind
              have hpv : p.eval env t havoc = true := by
                simp only [SymBool.eval] at hhyp; cases hx : p.eval env t havoc <;> simp_all
              exact findMbarReadySlotFuel_sound fuel p barId s hfind hpv
          | or p q =>
              dsimp only [findMbarReadySlotFuel] at hfind
              have hnpq : (SymBool.not p).eval env t havoc = true ∧ (SymBool.not q).eval env t havoc = true := by
                simp only [SymBool.eval] at hhyp ⊢
                cases hpv : p.eval env t havoc <;> cases hqv : q.eval env t havoc <;> simp_all
              cases hnp : findMbarReadySlotFuel fuel (.not p) barId with
              | some sp =>
                  rw [hnp] at hfind; injection hfind with hs; subst hs
                  exact findMbarReadySlotFuel_sound fuel (.not p) barId sp hnp hnpq.1
              | none =>
                  rw [hnp] at hfind; dsimp only at hfind
                  exact findMbarReadySlotFuel_sound fuel (.not q) barId s hfind hnpq.2
          | _ => dsimp only [findMbarReadySlotFuel] at hfind; contradiction
      | lit _ | cmp _ _ _ | ite _ _ _ =>
          dsimp only [findMbarReadySlotFuel] at hfind; contradiction

/-- Soundness of `findMbarReadySlot`. -/
theorem findMbarReadySlot_sound {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat}
    {hyp : SymBool} {barId : Nat} {s : SymInt}
    (hfind : findMbarReadySlot hyp barId = some s)
    (hhyp : hyp.eval env t havoc = true) :
    env.mbarFlag barId (s.eval env t havoc) = true :=
  findMbarReadySlotFuel_sound 32 hyp barId s hfind hhyp

/-- Computable syntactic test checking whether `(s1, s2)` is unconditionally ordered on a constant `Barrier` slot. -/
def symAsyncOrdered (f : MirFunc) (s1 s2 : SymMemEvent) : Bool :=
  match s1.syncBar, s2.syncBar with
  | some (b, .const c1), none =>
      f.hasValidBarSignal b && (findMbarReadySlot s2.pathCond b == some (.const c1))
  | none, some (b, .const c2) =>
      f.hasValidBarSignal b && (findMbarReadySlot s1.pathCond b == some (.const c2))
  | _, _ => false

/-- Soundness of `symAsyncOrdered`: returning `true` guarantees `asyncOrdered f env ev₁ ev₂`. -/
theorem symAsyncOrdered_sound {f : MirFunc} {env : InputEnv} {t₁ t₂ : ThreadCtx}
    {h₁ h₂ : Nat → Nat} {s1 s2 : SymMemEvent} {ev₁ ev₂ : MemEvent}
    (hord : symAsyncOrdered f s1 s2 = true)
    (hab1 : evalSyncBar env t₁ h₁ s1.syncBar = ev₁.syncBar)
    (hab2 : evalSyncBar env t₂ h₂ s2.syncBar = ev₂.syncBar)
    (hpc1 : s1.pathCond.eval env t₁ h₁ = true) (hpc2 : s2.pathCond.eval env t₂ h₂ = true) :
    asyncOrdered f env ev₁ ev₂ := by
  unfold symAsyncOrdered at hord
  cases hb1 : s1.syncBar <;> cases hb2 : s2.syncBar <;> rw [hb1, hb2] at hord
  · contradiction
  · rename_i bs2
    rcases bs2 with ⟨b, slot2⟩
    cases slot2 <;> try contradiction
    rename_i c2
    rcases Bool.and_eq_true_iff.mp hord with ⟨harm, hfindEq⟩
    have hfind : findMbarReadySlot s1.pathCond b = some (.const c2) := beq_iff_eq.mp hfindEq
    have hflag : env.mbarFlag b c2 = true := by
      simpa [SymInt.eval] using findMbarReadySlot_sound hfind hpc1
    rw [hb1] at hab1; rw [hb2] at hab2
    dsimp only [evalSyncBar, SymInt.eval] at hab1 hab2
    exact Or.inr ⟨b, c2, hab2.symm, hab1.symm, harm, hflag⟩
  · rename_i bs1
    rcases bs1 with ⟨b, slot1⟩
    cases slot1 <;> try contradiction
    rename_i c1
    rcases Bool.and_eq_true_iff.mp hord with ⟨harm, hfindEq⟩
    have hfind : findMbarReadySlot s2.pathCond b = some (.const c1) := beq_iff_eq.mp hfindEq
    have hflag : env.mbarFlag b c1 = true := by
      simpa [SymInt.eval] using findMbarReadySlot_sound hfind hpc2
    rw [hb1] at hab1; rw [hb2] at hab2
    dsimp only [evalSyncBar, SymInt.eval] at hab1 hab2
    exact Or.inl ⟨b, c1, hab1.symm, hab2.symm, harm, hflag⟩
  · rename_i bs1 bs2
    rcases bs1 with ⟨b1, sl1⟩; rcases bs2 with ⟨b2, sl2⟩
    cases sl1 <;> cases sl2 <;> contradiction

/-- Precomputable `SharedArray<Barrier, N>` `(slot₁, slot₂)` pair when one event signals
    `BAR[slot₁]` and the other event's `pathCond` awaits `BAR[slot₂]` on a signaled `Barrier`. -/
def extractAsyncSlotPair (f : MirFunc) (s1 s2 : SymMemEvent) : Option (SymInt × SymInt) :=
  match s1.syncBar, s2.syncBar with
  | some (b, slot₁), none =>
      if f.hasValidBarSignal b then
        (findMbarReadySlot s2.pathCond b).map fun slot₂ => (slot₁, slot₂)
      else none
  | none, some (b, slot₂) =>
      if f.hasValidBarSignal b then
        (findMbarReadySlot s1.pathCond b).map fun slot₁ => (slot₁, slot₂)
      else none
  | _, _ => none

/-- Propositional guard from a precomputed `extractAsyncSlotPair` result (`slot₁ ≠ slot₂ → goal`). -/
def symSlotGuardOpt (opt : Option (SymInt × SymInt))
    (env : InputEnv) (t₁ t₂ : ThreadCtx) (h₁ h₂ : Nat → Nat) (goal : Prop) : Prop :=
  match opt with
  | some (slot₁, slot₂) => (slot₁.eval env t₁ h₁ ≠ slot₂.eval env t₂ h₂) → goal
  | none => goal

/-- Relational `SharedArray<Barrier, N>` slot premise: when producer `s1` signals `(b, slot₁)`
    and consumer `s2` awaits `(b, slot₂)`, `¬ asyncOrdered` implies `slot₁ ≠ slot₂`! -/
def symAsyncSlotGuard (f : MirFunc) (s1 s2 : SymMemEvent)
    (env : InputEnv) (t₁ t₂ : ThreadCtx) (h₁ h₂ : Nat → Nat) (goal : Prop) : Prop :=
  symSlotGuardOpt (extractAsyncSlotPair f s1 s2) env t₁ t₂ h₁ h₂ goal

/-- Soundness of `symAsyncSlotGuard`: under `¬ asyncOrdered`, any colliding barrier slot `slot₁ = slot₂`
    contradicts `¬ asyncOrdered`, discharging the `slot₁ ≠ slot₂` premise! -/
theorem symAsyncSlotGuard_sound {f : MirFunc} {s1 s2 : SymMemEvent}
    {env : InputEnv} {t₁ t₂ : ThreadCtx} {h₁ h₂ : Nat → Nat}
    {ev₁ ev₂ : MemEvent} {goal : Prop}
    (hab1 : evalSyncBar env t₁ h₁ s1.syncBar = ev₁.syncBar)
    (hab2 : evalSyncBar env t₂ h₂ s2.syncBar = ev₂.syncBar)
    (hpc1 : s1.pathCond.eval env t₁ h₁ = true)
    (hpc2 : s2.pathCond.eval env t₂ h₂ = true)
    (hnotOrd : ¬ asyncOrdered f env ev₁ ev₂)
    (hguard : symAsyncSlotGuard f s1 s2 env t₁ t₂ h₁ h₂ goal) :
    goal := by
  unfold symAsyncSlotGuard extractAsyncSlotPair at hguard
  cases hb1 : s1.syncBar <;> cases hb2 : s2.syncBar <;> rw [hb1, hb2] at hguard
  · exact hguard
  · rename_i bs2
    rcases bs2 with ⟨b, slot₂⟩
    dsimp only at hguard
    by_cases harm : f.hasValidBarSignal b = true
    · rw [if_pos harm] at hguard
      cases hf1 : findMbarReadySlot s1.pathCond b with
      | none => rw [hf1] at hguard; exact hguard
      | some slot₁ =>
          rw [hf1] at hguard; dsimp only [Option.map, symSlotGuardOpt] at hguard
          apply hguard
          intro hsameSlot
          have hflag := findMbarReadySlot_sound hf1 hpc1
          rw [hsameSlot] at hflag
          rw [hb1] at hab1; rw [hb2] at hab2
          dsimp only [evalSyncBar] at hab1 hab2
          exact hnotOrd (Or.inr ⟨b, slot₂.eval env t₂ h₂, hab2.symm, hab1.symm, harm, hflag⟩)
    · rw [if_neg harm] at hguard; exact hguard
  · rename_i bs1
    rcases bs1 with ⟨b, slot₁⟩
    dsimp only at hguard
    by_cases harm : f.hasValidBarSignal b = true
    · rw [if_pos harm] at hguard
      cases hf2 : findMbarReadySlot s2.pathCond b with
      | none => rw [hf2] at hguard; exact hguard
      | some slot₂ =>
          rw [hf2] at hguard; dsimp only [Option.map, symSlotGuardOpt] at hguard
          apply hguard
          intro hsameSlot
          have hflag := findMbarReadySlot_sound hf2 hpc2
          rw [← hsameSlot] at hflag
          rw [hb1] at hab1; rw [hb2] at hab2
          dsimp only [evalSyncBar] at hab1 hab2
          exact hnotOrd (Or.inl ⟨b, slot₁.eval env t₁ h₁, hab1.symm, hab2.symm, harm, hflag⟩)
    · rw [if_neg harm] at hguard; exact hguard
  · exact hguard

/-- Obligation asserting `t₁` and `t₂` coincide on intra-CTA `(tx, ty)` whenever two span accesses overlap. -/
def symPairRaceOblig (s1 s2 : SymMemEvent) (b1 b2 : Nat) (goal : Prop) : Prop :=
  if s1.span == 1 && s1.elemDiv == 1 && s2.span == 1 && s2.elemDiv == 1 then
    b1 = b2 → goal
  else
    ∀ idx : Nat,
      idx * s1.elemDiv < b1 + s1.span →
      b1 < (idx + 1) * s1.elemDiv →
      idx * s2.elemDiv < b2 + s2.span →
      b2 < (idx + 1) * s2.elemDiv →
      goal

/-- Soundness of `symPairRaceOblig`: when `(b1 + k1) / d1 = (b2 + k2) / d2`, `goal` follows. -/
theorem symPairRaceOblig_sound {s1 s2 : SymMemEvent} {b1 b2 k1 k2 : Nat} {goal : Prop}
    (hk1 : k1 < s1.span) (hd1 : 0 < s1.elemDiv)
    (hk2 : k2 < s2.span) (hd2 : 0 < s2.elemDiv)
    (hob : symPairRaceOblig s1 s2 b1 b2 goal)
    (hsame : (b1 + k1) / s1.elemDiv = (b2 + k2) / s2.elemDiv) :
    goal := by
  unfold symPairRaceOblig at hob
  split at hob
  · rename_i h11
    simp only [Bool.and_eq_true, beq_iff_eq] at h11
    rcases h11 with ⟨⟨⟨hs1, hdv1⟩, hs2⟩, hdv2⟩
    have hk1z : k1 = 0 := by omega
    have hk2z : k2 = 0 := by omega
    rw [hk1z, hk2z, hdv1, hdv2, Nat.add_zero, Nat.add_zero, Nat.div_one, Nat.div_one] at hsame
    exact hob hsame
  · let q := (b1 + k1) / s1.elemDiv
    have hq2 : (b2 + k2) / s2.elemDiv = q := hsame.symm
    have h1a : q * s1.elemDiv < b1 + s1.span := by
      have hle : q * s1.elemDiv ≤ b1 + k1 := Nat.div_mul_le_self (b1 + k1) s1.elemDiv
      omega
    have h1b : b1 < (q + 1) * s1.elemDiv := by
      have hdm : s1.elemDiv * q + (b1 + k1) % s1.elemDiv = b1 + k1 :=
        Nat.div_add_mod (b1 + k1) s1.elemDiv
      have hrm := Nat.mod_lt (b1 + k1) hd1
      have hmul : s1.elemDiv * q = q * s1.elemDiv := Nat.mul_comm _ _
      have hexp : (q + 1) * s1.elemDiv = q * s1.elemDiv + s1.elemDiv := by
        rw [Nat.add_mul, Nat.one_mul]
      omega
    have h2a : q * s2.elemDiv < b2 + s2.span := by
      have hle : ((b2 + k2) / s2.elemDiv) * s2.elemDiv ≤ b2 + k2 := Nat.div_mul_le_self (b2 + k2) s2.elemDiv
      rw [hq2] at hle; omega
    have h2b : b2 < (q + 1) * s2.elemDiv := by
      have hdm0 := Nat.div_add_mod (b2 + k2) s2.elemDiv
      rw [hq2] at hdm0
      have hrm := Nat.mod_lt (b2 + k2) hd2
      have hmul : s2.elemDiv * q = q * s2.elemDiv := Nat.mul_comm _ _
      have hexp : (q + 1) * s2.elemDiv = q * s2.elemDiv + s2.elemDiv := by
        rw [Nat.add_mul, Nat.one_mul]
      omega
    exact hob q h1a h1b h2a h2b


/-- Propositional interpretation of `CmpOp` for direct `omega` consumption (zero `decide` wrappers!). -/
def evalCmpProp (op : CmpOp) (a b : Nat) : Prop :=
  match op with
  | .lt => a < b
  | .le => a ≤ b
  | .gt => b < a
  | .ge => b ≤ a
  | .eq => a = b
  | .ne => a ≠ b

/-- Equivalence of `evalCmpOp op a b = true` and `evalCmpProp op a b`. -/
theorem evalCmpOp_iff_prop (op : CmpOp) (a b : Nat) :
    evalCmpOp op a b = true ↔ evalCmpProp op a b := by
  cases op <;> simp [evalCmpOp, evalCmpProp]

/-- Rewrite `evalCmpOp op a b = true` directly to `evalCmpProp` so `@decide` never blocks congruence! -/
theorem evalCmpOp_eq_true (op : CmpOp) (a b : Nat) :
    (evalCmpOp op a b = true) = evalCmpProp op a b :=
  propext (evalCmpOp_iff_prop op a b)

/-- Rewrite `evalCmpOp op a b = false` directly to `¬evalCmpProp` so `@decide` never blocks congruence! -/
theorem evalCmpOp_eq_false (op : CmpOp) (a b : Nat) :
    (evalCmpOp op a b = false) = (¬evalCmpProp op a b) := by
  cases op <;> simp [evalCmpOp, evalCmpProp]

/-- Peel `Option` discriminant `(if c then 0/1 else 1/0) == 1` comparisons at the top of a path condition
    so `SymBool.holds` produces pure `<, ≤, =, ≠, ∧, ∨, ¬` atoms with zero `if` wrappers. -/
def SymBool.normDisc : SymBool → SymBool
  | .and p q => .and p.normDisc q.normDisc
  | p =>
      match extractIteOneCond p with
      | some (c, true) => c
      | some (c, false) => .not c
      | none => p

/-- Bidirectional equivalence for `extractIteOneCond`: matching `Option` discriminant comparisons
    evaluate to `true` iff the underlying condition `c` evaluates to `b`. -/
theorem extractIteOneCond_iff {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat} :
    ∀ {p c : SymBool} {b : Bool},
      extractIteOneCond p = some (c, b) →
      (p.eval env t havoc = true ↔ c.eval env t havoc = b)
  | .cmp .eq (.ite c0 (.const 1) (.const 0)) (.const 1), _, _, hext
  | .cmp .eq (.ite c0 (.const 1) (.const 0)) (.const 0), _, _, hext
  | .cmp .ne (.ite c0 (.const 1) (.const 0)) (.const 0), _, _, hext
  | .cmp .ne (.ite c0 (.const 1) (.const 0)) (.const 1), _, _, hext
  | .cmp .eq (.ite c0 (.const 0) (.const 1)) (.const 1), _, _, hext
  | .cmp .eq (.ite c0 (.const 0) (.const 1)) (.const 0), _, _, hext
  | .cmp .ne (.ite c0 (.const 0) (.const 1)) (.const 0), _, _, hext
  | .cmp .ne (.ite c0 (.const 0) (.const 1)) (.const 1), _, _, hext
  | .cmp .eq (.const 1) (.ite c0 (.const 1) (.const 0)), _, _, hext
  | .cmp .eq (.const 0) (.ite c0 (.const 1) (.const 0)), _, _, hext
  | .cmp .eq (.const 1) (.ite c0 (.const 0) (.const 1)), _, _, hext
  | .cmp .eq (.const 0) (.ite c0 (.const 0) (.const 1)), _, _, hext => by
      simp only [extractIteOneCond, Option.some.injEq, Prod.mk.injEq] at hext
      rcases hext with ⟨rfl, rfl⟩
      simp only [SymBool.eval, SymInt.eval, evalCmpOp]
      by_cases hcv : SymBool.eval env t havoc c0 = true <;> simp [hcv]

/-- Soundness of `SymBool.normDisc`: peeling top-level `Option` discriminant `0/1` wrappers
    preserves exact boolean truth under `(env, t, havoc)`. -/
theorem symBool_normDisc_iff (env : InputEnv) (t : ThreadCtx) (havoc : Nat → Nat) :
    ∀ p : SymBool, p.normDisc.eval env t havoc = true ↔ p.eval env t havoc = true
  | .and p q => by
      simp only [SymBool.normDisc, SymBool.eval, Bool.and_eq_true,
                 symBool_normDisc_iff env t havoc p, symBool_normDisc_iff env t havoc q]
  | .lit _ | .mbarReady _ _ | .not _ | .or _ _ | .ite _ _ _ => by simp [SymBool.normDisc, extractIteOneCond]
  | .cmp op l r => by
      dsimp only [SymBool.normDisc]
      cases hext : extractIteOneCond (.cmp op l r) with
      | none => rfl
      | some cb =>
          rcases cb with ⟨c, b⟩
          have hiff := extractIteOneCond_iff (env := env) (t := t) (havoc := havoc) hext
          cases b
          · dsimp only; rw [hiff]; simp only [SymBool.eval]
            cases SymBool.eval env t havoc c <;> simp
          · dsimp only; exact hiff.symm

/-- Propositional interpretation of `SymBool` (after `normDisc`) unfolding directly to `<, ≤, =, ≠, ∧, ∨, ¬`. -/
def SymBool.holdsRaw (env : InputEnv) (t : ThreadCtx) (havoc : Nat → Nat) : SymBool → Prop
  | .lit b => b = true
  | .mbarReady barId slot => env.mbarFlag barId (slot.eval env t havoc) = true
  | .cmp op a b => evalCmpProp op (a.eval env t havoc) (b.eval env t havoc)
  | .not p => ¬(p.holdsRaw env t havoc)
  | .and p q => p.holdsRaw env t havoc ∧ q.holdsRaw env t havoc
  | .or p q => p.holdsRaw env t havoc ∨ q.holdsRaw env t havoc
  | .ite c tb fb => if c.eval env t havoc then tb.holdsRaw env t havoc else fb.holdsRaw env t havoc

/-- Propositional truth of `p : SymBool` under `(env, t, havoc)`, normalizing `Option`
    discriminants first so `mir_vc` hands pure linear arithmetic directly to `omega`. -/
def SymBool.holds (env : InputEnv) (t : ThreadCtx) (havoc : Nat → Nat) (p : SymBool) : Prop :=
  p.normDisc.holdsRaw env t havoc

/-- Exact equivalence between `SymBool.eval = true` and `SymBool.holdsRaw`. -/
theorem symBool_eval_iff_holdsRaw (env : InputEnv) (t : ThreadCtx) (havoc : Nat → Nat) :
    ∀ p : SymBool, p.eval env t havoc = true ↔ p.holdsRaw env t havoc
  | .lit _ | .mbarReady _ _ => Iff.rfl
  | .cmp op a b => evalCmpOp_iff_prop op _ _
  | .not p => by
      simp only [SymBool.eval, SymBool.holdsRaw, Bool.not_eq_true']
      rw [← Bool.not_eq_true, symBool_eval_iff_holdsRaw env t havoc p]
  | .and p q => by
      simp only [SymBool.eval, SymBool.holdsRaw, Bool.and_eq_true,
                 symBool_eval_iff_holdsRaw env t havoc p, symBool_eval_iff_holdsRaw env t havoc q]
  | .or p q => by
      simp only [SymBool.eval, SymBool.holdsRaw, Bool.or_eq_true,
                 symBool_eval_iff_holdsRaw env t havoc p, symBool_eval_iff_holdsRaw env t havoc q]
  | .ite c tb fb => by
      simp only [SymBool.eval, SymBool.holdsRaw]
      split
      · exact symBool_eval_iff_holdsRaw env t havoc tb
      · exact symBool_eval_iff_holdsRaw env t havoc fb

/-- Exact equivalence between `SymBool.eval = true` and `SymBool.holds`. -/
theorem symBool_eval_iff_holds (env : InputEnv) (t : ThreadCtx) (havoc : Nat → Nat) (p : SymBool) :
    p.eval env t havoc = true ↔ p.holds env t havoc := by
  unfold SymBool.holds
  rw [← symBool_normDisc_iff env t havoc p]
  exact symBool_eval_iff_holdsRaw env t havoc p.normDisc

/-- Helper: lift `foldr` implications from `SymBool.holds` to `SymBool.eval = true`. -/
theorem foldr_holds_to_eval {α : Type} {c : α → SymBool} {Q : α → Prop}
    {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat} :
    ∀ xs : List α,
      xs.foldr (fun x acc => ((c x).holds env t havoc → Q x) ∧ acc) True →
      xs.foldr (fun x acc => ((c x).eval env t havoc = true → Q x) ∧ acc) True
  | [], _ => trivial
  | x :: tl, ⟨hhd, htl⟩ =>
      ⟨fun he => hhd ((symBool_eval_iff_holds env t havoc (c x)).mp he),
       foldr_holds_to_eval tl htl⟩

/-- Helper: lift two-thread `foldr` implications from `SymBool.holds` to `SymBool.eval = true`. -/
theorem foldr_holds2_to_eval {α : Type} {c1 c2 : α → SymBool} {Q : α → Prop}
    {env : InputEnv} {t₁ t₂ : ThreadCtx} {h₁ h₂ : Nat → Nat} :
    ∀ xs : List α,
      xs.foldr (fun x acc => ((c1 x).holds env t₁ h₁ → (c2 x).holds env t₂ h₂ → Q x) ∧ acc) True →
      xs.foldr (fun x acc => ((c1 x).eval env t₁ h₁ = true → (c2 x).eval env t₂ h₂ = true → Q x) ∧ acc) True
  | [], _ => trivial
  | x :: tl, ⟨hhd, htl⟩ =>
      ⟨fun he1 he2 => hhd ((symBool_eval_iff_holds env t₁ h₁ (c1 x)).mp he1)
                          ((symBool_eval_iff_holds env t₂ h₂ (c2 x)).mp he2),
       foldr_holds2_to_eval tl htl⟩

/-- Bridge theorem for `MirNoOobGlobal`: reduces concrete operational global bounds safety on
    `paramId` to pure `Nat` inequalities over the pre-cached `sevs` slice events.
    - **How generated kernels call it**:
      `apply vcNoOobGlobal_of_table mirFunc_valid mirEvents_eq (by mir_bind_events); mir_vc mirFunc` -/
theorem vcNoOobGlobal_of_table {f : MirFunc} {tbl : List SymBlockInv}
    {memEvs sevs : List SymMemEvent} {paramId : Nat}
    (hvalid : validSymFunc f tbl = true)
    (hmem : collectSymMemEvents f tbl = memEvs)
    (hevs : dedupEvents (memEvs.filter fun s => s.space == .global && s.bufId == paramId) = sevs)
    (hvc : ∀ env : InputEnv, ∀ t : ThreadCtx, ∀ havoc : Nat → Nat, -- For ANY valid input, thread, and loop state:
      f.validInput env → f.validThread t →                         -- under launch & CTA bounds,
      sevs.foldr (fun sev acc =>
        (sev.pathCond.holds env t havoc →                          -- IF the event's path condition holds,
         symSpanInBounds sev ((sev.indexOpt.getD (.const 0)).eval env t havoc) (env.sliceLen paramId)) ∧ acc) True) :
    MirNoOobGlobal f paramId := by
  intro env t ev hIn hTh hemits hsp hbid                    -- Step 1: Pick any GPU thread `t` and global access `ev`
  obtain ⟨havoc, sev, hsev, idxS, k, hks, hdiv, _, hssp, hsbid, _, hopt, hidxEv, hpc, _⟩ :=
    emitsEvent_mem_collect hvalid hIn hemits                -- Step 2: Map concrete `ev` to its symbolic event `sev`
  have hfMem : sev ∈ sevs := by                             -- Step 3: Show `sev` belongs to the filtered `paramId` slice list
    rw [← hevs]; apply mem_dedupEvents; rw [← hmem, List.mem_filter]; exact ⟨hsev, by simp [hssp, hsp, hsbid, hbid]⟩
  have hp := foldr_and_of_mem _ (foldr_holds_to_eval sevs (hvc env t havoc hIn hTh)) sev hfMem hpc -- Step 4: Extract `omega`'s VC proof for `sev`
  rw [hopt] at hp; dsimp only [Option.getD] at hp           -- Step 5: Unwrap `sev.indexOpt = some idxS`
  rw [← hidxEv]; exact symSpanInBounds_sound hks hdiv hp    -- Step 6: Conclude concrete `ev.index < env.sliceLen paramId`

/-- Soundness of `vcNoOobShared` from pre-computed `memEvs = collectSymMemEvents f tbl`. -/
theorem vcNoOobShared_of_table {f : MirFunc} {tbl : List SymBlockInv}
    {memEvs sevs : List SymMemEvent} {sharedId cap : Nat}
    (hvalid : validSymFunc f tbl = true)
    (hmem : collectSymMemEvents f tbl = memEvs)
    (hcap : (f.sharedDecls.getD sharedId { size := 0 }).size = cap)
    (hevs : dedupEvents (memEvs.filter fun s => s.space == .shared && s.bufId == sharedId) = sevs)
    (hvc : ∀ env : InputEnv, ∀ t : ThreadCtx, ∀ havoc : Nat → Nat, -- For ANY valid input, thread, and loop state:
      f.validInput env → f.validThread t →                         -- under launch & CTA bounds,
      sevs.foldr (fun sev acc =>
        (sev.pathCond.holds env t havoc →                          -- IF the event's path condition holds,
         symSpanInBounds sev ((sev.indexOpt.getD (.const 0)).eval env t havoc) cap) ∧ acc) True) :
    MirNoOobShared f sharedId := by
  intro env t ev hIn hTh hemits hsp hbid                    -- Step 1: Pick any CTA thread `t` and shared access `ev`
  obtain ⟨havoc, sev, hsev, idxS, k, hks, hdiv, _, hssp, hsbid, _, hopt, hidxEv, hpc, _⟩ :=
    emitsEvent_mem_collect hvalid hIn hemits                -- Step 2: Map concrete `ev` to its symbolic event `sev`
  have hfMem : sev ∈ sevs := by                             -- Step 3: Show `sev` belongs to `SharedArray` `sharedId`'s list
    rw [← hevs]; apply mem_dedupEvents; rw [← hmem, List.mem_filter]; exact ⟨hsev, by simp [hssp, hsp, hsbid, hbid]⟩
  have hp := foldr_and_of_mem _ (foldr_holds_to_eval sevs (hvc env t havoc hIn hTh)) sev hfMem hpc -- Step 4: Extract `omega`'s VC proof for `sev`
  rw [hopt] at hp; dsimp only [Option.getD] at hp           -- Step 5: Unwrap `sev.indexOpt = some idxS`
  rw [← hidxEv, hcap]; exact symSpanInBounds_sound hks hdiv hp -- Step 6: Conclude concrete `ev.index < SHARED.size`

/-- Soundness of `vcDisjointWrites` from pre-computed `memEvs = collectSymMemEvents f tbl`. -/
theorem vcDisjointWrites_of_table {f : MirFunc} {tbl : List SymBlockInv}
    {memEvs : List SymMemEvent} {paramId : Nat}
    {pairs : List (SymMemEvent × SymMemEvent)}
    (hvalid : validSymFunc f tbl = true)
    (hmem : collectSymMemEvents f tbl = memEvs)
    (hpairs : (let sevs := memEvs.filter fun s => s.space == .global && s.bufId == paramId
               (sevs.flatMap fun s1 => sevs.map fun s2 => (s1, s2)).filter fun p =>
                 p.1.isWrite || p.2.isWrite) = pairs)
    (hvc : ∀ env : InputEnv, ∀ t₁ t₂ : ThreadCtx, ∀ h₁ h₂ : Nat → Nat, -- For ANY two threads `t₁, t₂` and loop states `h₁, h₂`:
      f.validInput env → f.validThread t₁ → f.validThread t₂ →
      pairs.foldr (fun p acc =>
        (p.1.pathCond.holds env t₁ h₁ →                                -- IF both events' path conditions hold
         p.2.pathCond.holds env t₂ h₂ →                                -- AND their global element indices collide,
         symPairRaceOblig p.1 p.2
           ((p.1.indexOpt.getD (.const 0)).eval env t₁ h₁)
           ((p.2.indexOpt.getD (.const 0)).eval env t₂ h₂)
           (t₁.bx = t₂.bx ∧ t₁.by_ = t₂.by_ ∧ t₁.tx = t₂.tx ∧ t₁.ty = t₂.ty)) ∧ acc) True) :
    MirDisjointWrites f paramId := by
  intro env t₁ t₂ ev₁ ev₂ hIn hT1 hT2 hem1 hem2 hsp1 hsp2 hb1 hb2 hw hidx -- Step 1: Pick two threads `t₁, t₂` colliding on `ev₁.index = ev₂.index`
  obtain ⟨h₁, s1, hs1, i1, k1, hk1, hd1, hw1, hssp1, hsbid1, _, ho1, hi1, hpc1, _⟩ :=
    emitsEvent_mem_collect hvalid hIn hem1                  -- Step 2a: Map `ev₁` to symbolic event `s1`
  obtain ⟨h₂, s2, hs2, i2, k2, hk2, hd2, hw2, hssp2, hsbid2, _, ho2, hi2, hpc2, _⟩ :=
    emitsEvent_mem_collect hvalid hIn hem2                  -- Step 2b: Map `ev₂` to symbolic event `s2`
  rw [hmem] at hs1 hs2
  have hm1 : s1 ∈ memEvs.filter (fun s => s.space == .global && s.bufId == paramId) := by
    rw [List.mem_filter]; exact ⟨hs1, by simp [hssp1, hsp1, hsbid1, hb1]⟩
  have hm2 : s2 ∈ memEvs.filter (fun s => s.space == .global && s.bufId == paramId) := by
    rw [List.mem_filter]; exact ⟨hs2, by simp [hssp2, hsp2, hsbid2, hb2]⟩
  have hpair : (s1, s2) ∈ pairs := by                       -- Step 3: Show `(s1, s2)` is in `pairs` since one is a write
    rw [← hpairs, List.mem_filter]
    refine ⟨List.mem_flatMap.mpr ⟨s1, hm1, List.mem_map.mpr ⟨s2, hm2, rfl⟩⟩, ?_⟩
    dsimp only; rw [hw1, hw2]; rcases hw with h1 | h2
    · simp [h1]
    · simp [h2]
  have hp := foldr_and_of_mem _ (foldr_holds2_to_eval pairs (hvc env t₁ t₂ h₁ h₂ hIn hT1 hT2)) (s1, s2) hpair hpc1 hpc2 -- Step 4: Extract VC for `(s1, s2)`
  rw [ho1, ho2] at hp; dsimp only [Option.getD] at hp
  have heqIdx : (i1.eval env t₁ h₁ + k1) / s1.elemDiv = (i2.eval env t₂ h₂ + k2) / s2.elemDiv := by
    rw [hi1, hi2, hidx]                                     -- Step 5: Rewrite `ev₁.index = ev₂.index` into symbolic form
  obtain ⟨hbx, hby, htx, hty⟩ := symPairRaceOblig_sound hk1 hd1 hk2 hd2 hp heqIdx -- Step 6: Deduce all 4 coordinates match
  cases t₁; cases t₂; simp_all                              -- Step 7: Conclude `t₁ = t₂`

/-- Extract `some c` when `hyp` syntactically entails `t.tx = c` (direct `tid == c` or `warp == w ∧ lane == l`). -/
def findEqTidX : SymBool → Option Nat
  | .and p q => (findEqTidX p).orElse fun _ => findEqTidX q
  | .cmp op a b =>
      if op == .eq then
        if a == .tidX then asConstInt b
        else if b == .tidX then asConstInt a
        else if a == .intCast 64 .tidX then
          match asConstInt b with | some c => if decide (c < 4294967296) then some c else none | none => none
        else if b == .intCast 64 .tidX then
          match asConstInt a with | some c => if decide (c < 4294967296) then some c else none | none => none
        else none
      else none
  | _ => none

def findEqWarp32 : SymBool → Option Nat
  | .and p q => (findEqWarp32 p).orElse fun _ => findEqWarp32 q
  | .cmp op a b =>
      let wExpr : SymInt := .binOp .div 32 .tidX (.const 32)
      if op == .eq then
        if a == wExpr then asConstInt b
        else if b == wExpr then asConstInt a
        else none
      else none
  | _ => none

def findEqLane32 (blockDimX : Nat) : SymBool → Option Nat
  | .and p q => (findEqLane32 blockDimX p).orElse fun _ => findEqLane32 blockDimX q
  | .cmp op a b =>
      let lExpr1 : SymInt := .binOp .rem 32 .tidX (.const 32)
      let lExpr2 : SymInt := .binOp .rem 0 (.binOp .add 0 (.binOp .mul 0 .tidY (.const blockDimX)) .tidX) (.const 32)
      if op == .eq then
        if a == lExpr1 || a == lExpr2 then
          match asConstInt b with | some l => if decide (l < 32) then some l else none | none => none
        else if b == lExpr1 || b == lExpr2 then
          match asConstInt a with | some l => if decide (l < 32) then some l else none | none => none
        else none
      else none
  | _ => none

/-- Extract `some tx` when `hyp` pins `t.tx` to a single constant lane inside a 1D CTA (`blockDimY = 1`). -/
def findPinnedSingleThread (f : MirFunc) (hyp : SymBool) : Option Nat :=
  if f.blockDimY != 1 || decide (4294967296 < f.blockDimX) then none
  else
    match findEqTidX hyp with
    | some c => some c
    | none =>
        match findEqWarp32 hyp, findEqLane32 f.blockDimX hyp with
        | some w, some l => some (32 * w + l)
        | _, _ => none

theorem wordMod64_of_u32 {n : Nat} (h : n < 4294967296) : wordMod 64 n = n := by
  unfold wordMod; exact Nat.mod_eq_of_lt (Nat.lt_trans h (by decide))

theorem findEqTidX_sound {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat} (htx : t.tx < 4294967296) :
    ∀ (hyp : SymBool) {c : Nat}, findEqTidX hyp = some c → hyp.eval env t havoc = true → t.tx = c
  | .and p q, _, hf, he => by
      dsimp only [findEqTidX] at hf
      simp only [SymBool.eval, Bool.and_eq_true] at he
      cases hp : findEqTidX p with
      | some cp => rw [hp] at hf; injection hf with hc; subst hc; exact findEqTidX_sound htx p hp he.1
      | none => rw [hp] at hf; exact findEqTidX_sound htx q hf he.2
  | .cmp op a b, c, hf, he => by
      dsimp only [findEqTidX] at hf
      have hm := wordMod64_of_u32 htx
      by_cases hop : (op == .eq) = true
      · rw [if_pos hop] at hf; rw [beq_iff_eq.mp hop] at he
        by_cases ha1 : (a == .tidX) = true
        · rw [if_pos ha1] at hf; rw [beq_iff_eq.mp ha1, asConstInt_eq hf] at he
          simp only [SymBool.eval, SymInt.eval, evalCmpOp_eq_true, evalCmpProp] at he; exact he
        · rw [if_neg ha1] at hf
          by_cases hb1 : (b == .tidX) = true
          · rw [if_pos hb1] at hf; rw [beq_iff_eq.mp hb1, asConstInt_eq hf] at he
            simp only [SymBool.eval, SymInt.eval, evalCmpOp_eq_true, evalCmpProp] at he; exact he.symm
          · rw [if_neg hb1] at hf
            by_cases ha2 : (a == .intCast 64 .tidX) = true
            · rw [if_pos ha2] at hf
              cases hcb : asConstInt b <;> rw [hcb] at hf <;> try contradiction
              dsimp only at hf; split at hf <;> try contradiction
              injection hf with hc; subst hc
              rw [beq_iff_eq.mp ha2, asConstInt_eq hcb] at he
              simp only [SymBool.eval, SymInt.eval, evalCmpOp_eq_true, evalCmpProp, hm] at he; exact he
            · rw [if_neg ha2] at hf
              by_cases hb2 : (b == .intCast 64 .tidX) = true
              · rw [if_pos hb2] at hf
                cases hca : asConstInt a <;> rw [hca] at hf <;> try contradiction
                dsimp only at hf; split at hf <;> try contradiction
                injection hf with hc; subst hc
                rw [beq_iff_eq.mp hb2, asConstInt_eq hca] at he
                simp only [SymBool.eval, SymInt.eval, evalCmpOp_eq_true, evalCmpProp, hm] at he; exact he.symm
              · rw [if_neg hb2] at hf; contradiction
      · rw [if_neg hop] at hf; contradiction
  | .lit _, _, hf, _ | .mbarReady _ _, _, hf, _ | .not _, _, hf, _ | .or _ _, _, hf, _ | .ite _ _ _, _, hf, _ => by
      contradiction

theorem findEqWarp32_sound {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat} :
    ∀ (hyp : SymBool) {w : Nat}, findEqWarp32 hyp = some w → hyp.eval env t havoc = true → t.tx / 32 = w
  | .and p q, _, hf, he => by
      dsimp only [findEqWarp32] at hf
      simp only [SymBool.eval, Bool.and_eq_true] at he
      cases hp : findEqWarp32 p with
      | some wp => rw [hp] at hf; injection hf with hw; subst hw; exact findEqWarp32_sound p hp he.1
      | none => rw [hp] at hf; exact findEqWarp32_sound q hf he.2
  | .cmp op a b, w, hf, he => by
      dsimp only [findEqWarp32] at hf
      by_cases hop : (op == .eq) = true
      · rw [if_pos hop] at hf; rw [beq_iff_eq.mp hop] at he
        by_cases ha : (a == .binOp .div 32 .tidX (.const 32)) = true
        · rw [if_pos ha] at hf; rw [beq_iff_eq.mp ha, asConstInt_eq hf] at he
          simp only [SymBool.eval, SymInt.eval, evalBinOp, evalCmpOp_eq_true, evalCmpProp] at he; exact he
        · rw [if_neg ha] at hf
          by_cases hb : (b == .binOp .div 32 .tidX (.const 32)) = true
          · rw [if_pos hb] at hf; rw [beq_iff_eq.mp hb, asConstInt_eq hf] at he
            simp only [SymBool.eval, SymInt.eval, evalBinOp, evalCmpOp_eq_true, evalCmpProp] at he; exact he.symm
          · rw [if_neg hb] at hf; contradiction
      · rw [if_neg hop] at hf; contradiction
  | .lit _, _, hf, _ | .mbarReady _ _, _, hf, _ | .not _, _, hf, _ | .or _ _, _, hf, _ | .ite _ _ _, _, hf, _ => by
      contradiction

theorem findEqLane32_sound {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat}
    {blockDimX : Nat} (hty0 : t.ty = 0) :
    ∀ (hyp : SymBool) {l : Nat}, findEqLane32 blockDimX hyp = some l → hyp.eval env t havoc = true → t.tx % 32 = l
  | .and p q, _, hf, he => by
      dsimp only [findEqLane32] at hf
      simp only [SymBool.eval, Bool.and_eq_true] at he
      cases hp : findEqLane32 blockDimX p with
      | some lp => rw [hp] at hf; injection hf with hl; subst hl; exact findEqLane32_sound hty0 p hp he.1
      | none => rw [hp] at hf; exact findEqLane32_sound hty0 q hf he.2
  | .cmp op a b, l, hf, he => by
      dsimp only [findEqLane32] at hf
      by_cases hop : (op == .eq) = true
      · rw [if_pos hop] at hf; rw [beq_iff_eq.mp hop] at he
        by_cases ha : (a == .binOp .rem 32 .tidX (.const 32) || a == .binOp .rem 0 (.binOp .add 0 (.binOp .mul 0 .tidY (.const blockDimX)) .tidX) (.const 32)) = true
        · rw [if_pos ha] at hf
          cases hcb : asConstInt b <;> rw [hcb] at hf <;> try contradiction
          dsimp only at hf; split at hf <;> try contradiction
          injection hf with hl; subst hl
          rw [asConstInt_eq hcb] at he
          rcases Bool.or_eq_true_iff.mp ha with ha1 | ha2
          · rw [beq_iff_eq.mp ha1] at he
            simp only [SymBool.eval, SymInt.eval, evalBinOp, evalCmpOp_eq_true, evalCmpProp] at he; exact he
          · rw [beq_iff_eq.mp ha2] at he
            simp only [SymBool.eval, SymInt.eval, evalBinOp, wordMod, evalCmpOp_eq_true, evalCmpProp, hty0, Nat.zero_mul, Nat.zero_add] at he; exact he
        · rw [if_neg ha] at hf
          by_cases hb : (b == .binOp .rem 32 .tidX (.const 32) || b == .binOp .rem 0 (.binOp .add 0 (.binOp .mul 0 .tidY (.const blockDimX)) .tidX) (.const 32)) = true
          · rw [if_pos hb] at hf
            cases hca : asConstInt a <;> rw [hca] at hf <;> try contradiction
            dsimp only at hf; split at hf <;> try contradiction
            injection hf with hl; subst hl
            rw [asConstInt_eq hca] at he
            rcases Bool.or_eq_true_iff.mp hb with hb1 | hb2
            · rw [beq_iff_eq.mp hb1] at he
              simp only [SymBool.eval, SymInt.eval, evalBinOp, evalCmpOp_eq_true, evalCmpProp] at he; exact he.symm
            · rw [beq_iff_eq.mp hb2] at he
              simp only [SymBool.eval, SymInt.eval, evalBinOp, wordMod, evalCmpOp_eq_true, evalCmpProp, hty0, Nat.zero_mul, Nat.zero_add] at he; exact he.symm
          · rw [if_neg hb] at hf; contradiction
      · rw [if_neg hop] at hf; contradiction
  | .lit _, _, hf, _ | .mbarReady _ _, _, hf, _ | .not _, _, hf, _ | .or _ _, _, hf, _ | .ite _ _ _, _, hf, _ => by
      contradiction

/-- Computable syntactic test: do `s1` and `s2` execute exclusively on the exact same single CTA thread? -/
def symPinsSameThread (f : MirFunc) (s1 s2 : SymMemEvent) : Bool :=
  match findPinnedSingleThread f s1.pathCond, findPinnedSingleThread f s2.pathCond with
  | some c1, some c2 => c1 == c2
  | _, _ => false

theorem symPinsSameThread_sound {f : MirFunc} {env : InputEnv} {t₁ t₂ : ThreadCtx}
    {h₁ h₂ : Nat → Nat} {s1 s2 : SymMemEvent}
    (hT1 : f.validThread t₁) (hT2 : f.validThread t₂)
    (hbx : t₁.bx = t₂.bx) (hby : t₁.by_ = t₂.by_)
    (hpin : symPinsSameThread f s1 s2 = true)
    (hpc1 : s1.pathCond.eval env t₁ h₁ = true) (hpc2 : s2.pathCond.eval env t₂ h₂ = true) :
    t₁ = t₂ := by
  have singlePin : ∀ (t : ThreadCtx) (h : Nat → Nat) (p : SymBool) (c : Nat),
      f.validThread t → findPinnedSingleThread f p = some c → p.eval env t h = true → t.tx = c ∧ t.ty = 0 := by
    intro t h p c hTh hf he
    unfold findPinnedSingleThread at hf
    split at hf
    · contradiction
    · rename_i hok
      simp only [Bool.or_eq_true, bne_iff_ne, decide_eq_true_eq, not_or, Classical.not_not] at hok
      have htxB : t.tx < 4294967296 := by have := hTh.1; omega
      have hty0 : t.ty = 0 := by have := hTh.2.1; rw [hok.1] at this; omega
      refine ⟨?_, hty0⟩
      cases heq : findEqTidX p with
      | some c0 => rw [heq] at hf; injection hf with hc; subst hc; exact findEqTidX_sound htxB p heq he
      | none =>
          rw [heq] at hf
          cases hw : findEqWarp32 p <;> cases hl : findEqLane32 f.blockDimX p <;> rw [hw, hl] at hf <;> try contradiction
          injection hf with hc; subst hc
          have hdw := findEqWarp32_sound p hw he
          have hdl := findEqLane32_sound hty0 p hl he
          have hdiv := Nat.div_add_mod t.tx 32
          omega
  unfold symPinsSameThread at hpin
  cases hp1 : findPinnedSingleThread f s1.pathCond <;>
    cases hp2 : findPinnedSingleThread f s2.pathCond <;> rw [hp1, hp2] at hpin <;> try contradiction
  rename_i c1 c2
  have hcEq : c1 = c2 := beq_iff_eq.mp hpin
  subst hcEq
  obtain ⟨htx1, hty1⟩ := singlePin t₁ h₁ s1.pathCond c1 hT1 hp1 hpc1
  obtain ⟨htx2, hty2⟩ := singlePin t₂ h₂ s2.pathCond c1 hT2 hp2 hpc2
  cases t₁; cases t₂; simp_all

/-- Upper-triangular pair enumeration `{(x_i, x_j) | i ≤ j}` avoiding symmetric `(s1, s2)` / `(s2, s1)` duplication. -/
def upperTriPairs {α : Type} : List α → List (α × α)
  | [] => []
  | x :: xs => (x :: xs).map (x, ·) ++ upperTriPairs xs

theorem mem_upperTriPairs {α : Type} {x y : α} :
    ∀ {xs : List α}, x ∈ xs → y ∈ xs → (x, y) ∈ upperTriPairs xs ∨ (y, x) ∈ upperTriPairs xs
  | [], hx, _ => nomatch hx
  | _ :: _, List.Mem.head _, hy =>
      Or.inl (List.mem_append_left _ (List.mem_map.mpr ⟨y, hy, rfl⟩))
  | _ :: _, List.Mem.tail _ hx, List.Mem.head _ =>
      Or.inr (List.mem_append_left _ (List.mem_map.mpr ⟨x, List.Mem.tail _ hx, rfl⟩))
  | _ :: _, List.Mem.tail _ hx, List.Mem.tail _ hy =>
      match mem_upperTriPairs hx hy with
      | Or.inl h1 => Or.inl (List.mem_append_right _ h1)
      | Or.inr h2 => Or.inr (List.mem_append_right _ h2)

/-- Bridge theorem for `MirNoSharedRace`: requires `UniformBarriers cfg` to align barrier
    phases across CTA threads, then prunes barrier-disjoint, mbarrier-ordered, same-single-thread,
    and symmetric duplicate event pairs, reducing remaining conflicts on `sharedId` to `tx₁ = tx₂ ∧ ty₁ = ty₂`. -/
theorem vcNoSharedRace_of_table {f : MirFunc} {cfg : ControlFlowCfg} {tbl : List SymBlockInv}
    {memEvs : List SymMemEvent} {sharedId : Nat}
    {pairs : List (SymMemEvent × SymMemEvent × Option (SymInt × SymInt))}
    (hunif : UniformBarriers f cfg)
    (hvalid : validSymFunc f tbl = true)
    (hmem : collectSymMemEvents f tbl = memEvs)
    (hpairs : (let sevs := dedupEvents (memEvs.filter fun s => s.space == .shared && s.bufId == sharedId)
               ((upperTriPairs sevs).filter fun p =>
                 (p.1.isWrite || p.2.isWrite) &&
                 !(p.1.phaseSpec.disjoint p.2.phaseSpec) &&
                 !(symAsyncOrdered f p.1 p.2) &&
                 !(symPinsSameThread f p.1 p.2)).map fun p =>
                   (p.1, p.2, extractAsyncSlotPair f p.1 p.2)) = pairs)
    (hvc : ∀ env : InputEnv, ∀ t₁ t₂ : ThreadCtx, ∀ h₁ h₂ : Nat → Nat, -- For ANY two threads `t₁, t₂` in the same CTA:
      f.validInput env → f.validThread t₁ → f.validThread t₂ →
      t₁.bx = t₂.bx → t₁.by_ = t₂.by_ →
      pairs.foldr (fun p acc =>
        (p.1.pathCond.holds env t₁ h₁ →                                -- IF both events' path conditions hold
         p.2.1.pathCond.holds env t₂ h₂ →                              -- AND they are not ordered on the same `Barrier[slot]`,
         symSlotGuardOpt p.2.2 env t₁ t₂ h₁ h₂
           (symPairRaceOblig p.1 p.2.1
             ((p.1.indexOpt.getD (.const 0)).eval env t₁ h₁)
             ((p.2.1.indexOpt.getD (.const 0)).eval env t₂ h₂)
             (t₁.tx = t₂.tx ∧ t₁.ty = t₂.ty))) ∧ acc) True) :
    MirNoSharedRace f cfg sharedId := by
  refine ⟨hunif, ?_⟩                                        -- Supply `UniformBarriers cfg` for phase alignment
  intro env t₁ t₂ ev₁ ev₂ hIn hT1 hT2 hbx hby hem1 hem2 hsp1 hsp2 hb1 hb2 hph hnotOrd hw hidx
  obtain ⟨h₁, s1, hs1, i1, k1, hk1, hd1, hw1, hssp1, hsbid1, hab1, ho1, hi1, hpc1, hps1⟩ :=
    emitsEvent_mem_collect hvalid hIn hem1
  obtain ⟨h₂, s2, hs2, i2, k2, hk2, hd2, hw2, hssp2, hsbid2, hab2, ho2, hi2, hpc2, hps2⟩ :=
    emitsEvent_mem_collect hvalid hIn hem2
  by_cases hdisj : s1.phaseSpec.disjoint s2.phaseSpec = true
  · exact False.elim (phaseSpec_disjoint_sound hdisj hps1 hps2 hph)
  · by_cases haord : symAsyncOrdered f s1 s2 = true
    · exact False.elim (hnotOrd (symAsyncOrdered_sound haord hab1 hab2 hpc1 hpc2))
    · by_cases hpin : symPinsSameThread f s1 s2 = true
      · exact symPinsSameThread_sound hT1 hT2 hbx hby hpin hpc1 hpc2
      · rw [hmem] at hs1 hs2
        have hm1 : s1 ∈ dedupEvents (memEvs.filter (fun s => s.space == .shared && s.bufId == sharedId)) := by
          apply mem_dedupEvents; rw [List.mem_filter]; exact ⟨hs1, by simp [hssp1, hsp1, hsbid1, hb1]⟩
        have hm2 : s2 ∈ dedupEvents (memEvs.filter (fun s => s.space == .shared && s.bufId == sharedId)) := by
          apply mem_dedupEvents; rw [List.mem_filter]; exact ⟨hs2, by simp [hssp2, hsp2, hsbid2, hb2]⟩
        have hwOr : (s1.isWrite || s2.isWrite) = true := by
          rw [hw1, hw2]; rcases hw with hwL | hwR <;> simp_all
        have hnd : (!(s1.phaseSpec.disjoint s2.phaseSpec)) = true := by
          cases hc : s1.phaseSpec.disjoint s2.phaseSpec <;> simp_all
        have hna : (!(symAsyncOrdered f s1 s2)) = true := by
          cases hc : symAsyncOrdered f s1 s2 <;> simp_all
        have hnp : (!(symPinsSameThread f s1 s2)) = true := by
          cases hc : symPinsSameThread f s1 s2 <;> simp_all
        rcases mem_upperTriPairs hm1 hm2 with h12 | h21
        · have hpair : (s1, s2, extractAsyncSlotPair f s1 s2) ∈ pairs := by
            rw [← hpairs, List.mem_map]
            refine ⟨(s1, s2), List.mem_filter.mpr ⟨h12, ?_⟩, rfl⟩
            simp [hwOr, hnd, hna, hnp]
          have hp0 := foldr_and_of_mem _ (foldr_holds2_to_eval pairs (hvc env t₁ t₂ h₁ h₂ hIn hT1 hT2 hbx hby))
            (s1, s2, extractAsyncSlotPair f s1 s2) hpair hpc1 hpc2
          have hp := symAsyncSlotGuard_sound hab1 hab2 hpc1 hpc2 hnotOrd hp0
          rw [ho1, ho2] at hp; dsimp only [Option.getD] at hp
          have heqIdx : (i1.eval env t₁ h₁ + k1) / s1.elemDiv = (i2.eval env t₂ h₂ + k2) / s2.elemDiv := by
            rw [hi1, hi2, hidx]
          obtain ⟨htx, hty⟩ := symPairRaceOblig_sound hk1 hd1 hk2 hd2 hp heqIdx
          cases t₁; cases t₂; simp_all
        · have hnotOrd2 : ¬ asyncOrdered f env ev₂ ev₁ := fun ho => hnotOrd ho.symm
          by_cases hdisj2 : s2.phaseSpec.disjoint s1.phaseSpec = true
          · exact False.elim (phaseSpec_disjoint_sound hdisj2 hps2 hps1 hph.symm)
          · by_cases haord2 : symAsyncOrdered f s2 s1 = true
            · exact False.elim (hnotOrd2 (symAsyncOrdered_sound haord2 hab2 hab1 hpc2 hpc1))
            · by_cases hpin2 : symPinsSameThread f s2 s1 = true
              · exact (symPinsSameThread_sound hT2 hT1 hbx.symm hby.symm hpin2 hpc2 hpc1).symm
              · have hpair : (s2, s1, extractAsyncSlotPair f s2 s1) ∈ pairs := by
                  rw [← hpairs, List.mem_map]
                  refine ⟨(s2, s1), List.mem_filter.mpr ⟨h21, ?_⟩, rfl⟩
                  have hwOr2 : (s2.isWrite || s1.isWrite) = true := by rw [Bool.or_comm]; exact hwOr
                  have hnd2 : (!(s2.phaseSpec.disjoint s1.phaseSpec)) = true := by
                    cases hc : s2.phaseSpec.disjoint s1.phaseSpec <;> simp_all
                  have hna2 : (!(symAsyncOrdered f s2 s1)) = true := by
                    cases hc : symAsyncOrdered f s2 s1 <;> simp_all
                  have hnp2 : (!(symPinsSameThread f s2 s1)) = true := by
                    cases hc : symPinsSameThread f s2 s1 <;> simp_all
                  simp [hwOr2, hnd2, hna2, hnp2]
                have hp0 := foldr_and_of_mem _ (foldr_holds2_to_eval pairs (hvc env t₂ t₁ h₂ h₁ hIn hT2 hT1 hbx.symm hby.symm))
                  (s2, s1, extractAsyncSlotPair f s2 s1) hpair hpc2 hpc1
                have hp := symAsyncSlotGuard_sound hab2 hab1 hpc2 hpc1 hnotOrd2 hp0
                rw [ho2, ho1] at hp; dsimp only [Option.getD] at hp
                have heqIdx : (i2.eval env t₂ h₂ + k2) / s2.elemDiv = (i1.eval env t₁ h₁ + k1) / s1.elemDiv := by
                  rw [hi2, hi1, hidx]
                obtain ⟨htx, hty⟩ := symPairRaceOblig_sound hk2 hd2 hk1 hd1 hp heqIdx
                cases t₁; cases t₂; simp_all

/-- Soundness of `vcAssertRedundant` from pre-computed `assEvs = collectSymAssertEvents f tbl`. -/
theorem vcAssertRedundant_of_table {f : MirFunc} {tbl : List SymBlockInv}
    {assEvs aevs : List SymAssertEvent} {assertId : Nat}
    (hvalid : validSymFunc f tbl = true)
    (hass : collectSymAssertEvents f tbl = assEvs)
    (haevs : (assEvs.filter (·.assertId == assertId)) = aevs)
    (hvc : ∀ env : InputEnv, ∀ t : ThreadCtx, ∀ havoc : Nat → Nat,
      f.validInput env → f.validThread t →
      aevs.foldr (fun aev acc =>
        (aev.pathCond.holds env t havoc →
         (aev.condOpt.getD (.lit false)).holds env t havoc) ∧ acc) True) :
    MirAssertRedundant f assertId := by
  intro env t b phase regs hIn hTh hreach pfx sfx cond hstmts -- Step 1: Pick any reachable block `b` containing `assertCond assertId cond`
  obtain ⟨hb, hr, havoc, hpc, hps, hm⟩ := symReachable_sound hvalid hIn hreach -- Step 2: Obtain block `b`'s symbolic invariant match
  have hvf := hvalid
  unfold validSymFunc at hvf
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hvf
  obtain ⟨_, _, hallB⟩ := hvf
  have hbOk := List.all_eq_true.mp hallB b (List.mem_range.mpr hb)
  rw [hr] at hbOk; simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true] at hbOk
  obtain ⟨_, _, _, hmA, _⟩ := symExecStmts_correct f env t havoc (tbl.getD b {}).effCond hpc (f.blockAt b).term
    (f.blockAt b).stmts regs (tbl.getD b {}).regs phase (tbl.getD b {}).phaseSpec hm hps hbOk.1.1.1 hbOk.1.2
  obtain ⟨aev, haevMem, haid, hpce, hcondEval⟩ := hmA pfx sfx assertId cond hstmts -- Step 3: Map `assertCond` to symbolic `aev`
  have haevOk := List.all_eq_true.mp hbOk.1.1.2 aev haevMem
  cases hcOpt : aev.condOpt with                            -- Step 4: `validSymFunc` guarantees `aev.condOpt = some bSym`
  | none => rw [hcOpt] at haevOk; contradiction
  | some bSym =>
      have hfMem : aev ∈ aevs := by                         -- Step 5: Show `aev` belongs to the filtered `assertId` list
        rw [← haevs, ← hass, List.mem_filter]
        refine ⟨?_, beq_iff_eq.mpr haid⟩
        unfold collectSymAssertEvents
        rw [List.mem_flatMap]
        refine ⟨b, List.mem_range.mpr hb, ?_⟩
        dsimp only; rw [hr]; simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte]; exact haevMem
      suffices hgen : ∀ xs : List SymAssertEvent,
          xs.foldr (fun x acc => (x.pathCond.holds env t havoc → (x.condOpt.getD (.lit false)).holds env t havoc) ∧ acc) True →
          xs.foldr (fun x acc => (x.pathCond.eval env t havoc = true → (x.condOpt.getD (.lit false)).eval env t havoc = true) ∧ acc) True by
        have hp := foldr_and_of_mem _ (hgen aevs (hvc env t havoc hIn hTh)) aev hfMem (by rw [hpce]; exact hpc)
        rw [hcOpt] at hp; dsimp only [Option.getD] at hp
        rw [hcondEval bSym hcOpt, hp]                       -- Step 6: Conclude concrete `cond = .bool true` (safe to elide!)
      intro xs; induction xs with
      | nil => intro _; trivial
      | cons a tl ih =>
          intro ⟨hhd, htl⟩
          exact ⟨fun hp => (symBool_eval_iff_holds env t havoc _).mpr
                  (hhd ((symBool_eval_iff_holds env t havoc _).mp hp)), ih htl⟩

/-- Bridge theorem for `MirNoOobTmem`: reduces operational Blackwell `TMEM` bounds and warpgroup
    slice safety to pure `Nat` inequalities over the materialized `tevs` `SymTmemEvent` list. -/
theorem vcNoOobTmem_of_table {f : MirFunc} {tbl : List SymBlockInv}
    {tmemEvs tevs : List SymTmemEvent}
    (hvalid : validSymFunc f tbl = true)
    (htmem : collectSymTmemEvents f tbl = tmemEvs)
    (hevs : tmemEvs = tevs)
    (hvc : ∀ env : InputEnv, ∀ t : ThreadCtx, ∀ havoc : Nat → Nat,
      f.validInput env → f.validThread t →
      tevs.foldr (fun stev acc =>
        (stev.pathCond.holds env t havoc →
         (((stev.relAddrOpt.getD (.const 0)).eval env t havoc) / 65536) + stev.heightLanes ≤ 128 ∧
         (((stev.relAddrOpt.getD (.const 0)).eval env t havoc) % 65536) + stev.widthCols ≤ stev.allocCols ∧
         stev.allocCols ≤ 512 ∧
         (stev.warpRestricted = true →
           ((t.tx / 32) % 4) * 32 ≤ ((stev.relAddrOpt.getD (.const 0)).eval env t havoc) / 65536 ∧
           (((stev.relAddrOpt.getD (.const 0)).eval env t havoc) / 65536) + stev.heightLanes ≤
             ((t.tx / 32) % 4) * 32 + 32)) ∧ acc) True) :
    MirNoOobTmem f := by
  intro env t tev hIn hTh hemits                            -- Step 1: Pick any CTA thread `t` and `TMEM` access `tev`
  obtain ⟨havoc, stev, hstev, relS, hac, hhl, hwc, hwr, hro, hre, hpc⟩ :=
    emitsTmemEvent_mem_collect hvalid hIn hemits            -- Step 2: Map `tev` to its symbolic `SymTmemEvent` `stev`
  rw [htmem, hevs] at hstev
  have hp := foldr_and_of_mem _ (foldr_holds_to_eval tevs (hvc env t havoc hIn hTh)) stev hstev hpc -- Step 3: Extract `omega`'s VC proof
  rw [hro] at hp; dsimp only [Option.getD] at hp            -- Step 4: Unwrap `stev.relAddrOpt = some relS`
  rw [hre, hac, hhl, hwc, hwr] at hp                        -- Step 5: Transfer symbolic lane/col/warpgroup bounds to `tev`
  exact hp

/-- Symbolic cluster-rank check recorded at `Stmt.clusterRankCheck` (`map_shared_rank`). -/
structure SymRankEvent where
  /-- Symbolic rank expression (`none` if unresolvable, which fails closed to `False`). -/
  rankOpt : Option SymInt
  /-- Active path condition at the `clusterRankCheck` site. -/
  pathCond : SymBool
  deriving Repr, DecidableEq, Lean.ToExpr

/-- Extract all `SymRankEvent`s from a block's statement list. -/
def symRankEventsOfStmts (f : MirFunc) (pCond : SymBool) (sregs : SymRegList) :
    List Stmt → List SymRankEvent
  | [] => []
  | .assign dst (.load _) :: rest =>
      symRankEventsOfStmts f pCond (sregs.setVar dst .unk) rest
  | .assign dst (.loadTmemBase _ cols) :: rest =>
      symRankEventsOfStmts f pCond (sregs.setVar dst (.tmemPtr cols (.const 0))) rest
  | .assign dst rhs :: rest =>
      symRankEventsOfStmts f pCond (sregs.setVar dst (symEvalRhs f sregs rhs)) rest
  | .clusterRankCheck rv :: rest =>
      let rOpt := match sregs.getVar rv with | .int e => some e | _ => none
      { rankOpt := rOpt, pathCond := pCond } :: symRankEventsOfStmts f pCond sregs rest
  | _ :: rest =>
      symRankEventsOfStmts f pCond sregs rest

/-- Correspondence of `symRankEventsOfStmts` with `pfx ++ (.clusterRankCheck rv :: sfx)` in `execStmts`. -/
theorem symRankEventsOfStmts_correct (f : MirFunc) (env : InputEnv) (t : ThreadCtx)
    (havoc : Nat → Nat) (pCond : SymBool) (term : Terminator) :
    ∀ stmts pfx sfx : List Stmt, ∀ regs : RegFile, ∀ sregs : SymRegList,
      ∀ phase rv r : Nat,
      (∀ v, (sregs.getVar v).matchesVal env t havoc (regs v)) →
      stmts = pfx ++ (.clusterRankCheck rv :: sfx) →
      (execStmts f env t regs phase term pfx).1 rv = .int r →
      ∃ rev ∈ symRankEventsOfStmts f pCond sregs stmts,
        rev.pathCond = pCond ∧
        ∀ rSym : SymInt, rev.rankOpt = some rSym → rSym.eval env t havoc = r
  | _, [], sfx, regs, sregs, phase, rv, r, hmatch, rfl, hr => by
      dsimp only [List.nil_append, execStmts] at hr ⊢
      refine ⟨{ rankOpt := match sregs.getVar rv with | .int e => some e | _ => none, pathCond := pCond },
              List.Mem.head _, rfl, ?_⟩
      intro rSym hopt
      have hm := hmatch rv
      cases hsv : sregs.getVar rv with
      | int e0 =>
          rw [hsv] at hopt hm; injection hopt with heq; subst heq
          rw [hr] at hm; dsimp only [SymVal.matchesVal] at hm; injection hm with hmr; exact hmr.symm
      | _ => rw [hsv] at hopt; contradiction
  | stmt :: rest, hd :: tl, sfx, regs, sregs, phase, rv, r, hmatch, heq, hr => by
      simp only [List.cons_append, List.cons.injEq] at heq
      rcases heq with ⟨rfl, htl⟩
      cases stmt with
      | assign dst rhs =>
          have hnextGen : ∀ v, ((sregs.setVar dst (symEvalRhs f sregs rhs)).getVar v).matchesVal env t havoc
              ((regs.set dst (evalRhs f env t regs rhs)) v) := by
            intro v; unfold RegFile.set; rw [getVar_setVar]
            by_cases hv : (v == dst) = true
            · simp [hv]; exact symEvalRhs_correct f env t havoc regs sregs hmatch rhs
            · simp [hv]; exact hmatch v
          cases rhs with
          | load ptr =>
              have hnext : ∀ v, ((sregs.setVar dst .unk).getVar v).matchesVal env t havoc ((regs.set dst .unk) v) := by
                intro v; unfold RegFile.set; rw [getVar_setVar]
                by_cases hv : (v == dst) = true <;> simp [hv, SymVal.matchesVal]; exact hmatch v
              obtain ⟨rev, hmem, hp, hrEq⟩ := symRankEventsOfStmts_correct f env t havoc pCond term
                rest tl sfx (regs.set dst .unk) (sregs.setVar dst .unk) phase rv r hnext htl (by simpa [execStmts] using hr)
              exact ⟨rev, hmem, hp, hrEq⟩
          | loadTmemBase ptr cols =>
              have hnext : ∀ v, ((sregs.setVar dst (.tmemPtr cols (.const 0))).getVar v).matchesVal env t havoc
                  ((regs.set dst (.tmemPtr cols 0)) v) := by
                intro v; unfold RegFile.set; rw [getVar_setVar]
                by_cases hv : (v == dst) = true <;> simp [hv, SymVal.matchesVal, SymInt.eval]; exact hmatch v
              obtain ⟨rev, hmem, hp, hrEq⟩ := symRankEventsOfStmts_correct f env t havoc pCond term
                rest tl sfx (regs.set dst (.tmemPtr cols 0)) (sregs.setVar dst (.tmemPtr cols (.const 0))) phase rv r hnext htl (by simpa [execStmts] using hr)
              exact ⟨rev, hmem, hp, hrEq⟩
          | copy _ | constInt _ _ | constBool _ | sreg _ | index1D | coord2DU32 | paramScalar _ | slicePtr _
          | sliceLen _ | sharedBase _ | binOp _ _ _ _ | checkedOvf _ _ _ _ | cmpOp _ _ _ | notBool _ | intCast _ _
          | ptrOffset _ _ | mbarTryWait _ _ _ | cvtaShared _ | opaqueUint _ _ | ctaUint1 _ _ _ | ctaConst | ctaUnary _ | ctaBin _ _ | opaqueVal =>
              obtain ⟨rev, hmem, hp, hrEq⟩ := symRankEventsOfStmts_correct f env t havoc pCond term
                rest tl sfx _ _ phase rv r hnextGen htl (by simpa [execStmts] using hr)
              exact ⟨rev, hmem, hp, hrEq⟩
      | clusterRankCheck rv0 =>
          obtain ⟨rev, hmem, hp, hrEq⟩ := symRankEventsOfStmts_correct f env t havoc pCond term
            rest tl sfx regs sregs phase rv r hmatch htl (by simpa [execStmts] using hr)
          exact ⟨rev, List.Mem.tail _ hmem, hp, hrEq⟩
      | store _ _ | barrierSlotCheck _ | assertCond _ _ | mbarArrive _ _ _ | mbarArriveExpectTx _ _ _
      | tmaCopyG2S _ _ _ _ | tmemAlloc _ _ | smemDescRead _ _ | tmemAccess _ _ _ _ =>
          obtain ⟨rev, hmem, hp, hrEq⟩ := symRankEventsOfStmts_correct f env t havoc pCond term
            rest tl sfx regs sregs phase rv r hmatch htl (by simpa [execStmts] using hr)
          exact ⟨rev, hmem, hp, hrEq⟩
      | syncThreads =>
          obtain ⟨rev, hmem, hp, hrEq⟩ := symRankEventsOfStmts_correct f env t havoc pCond term
            rest tl sfx regs sregs (phase + 1) rv r hmatch htl (by simpa [execStmts] using hr)
          exact ⟨rev, hmem, hp, hrEq⟩

/-- Collect all `SymRankEvent`s across reachable blocks in `tbl`. -/
def collectSymRankEvents (f : MirFunc) (tbl : List SymBlockInv) : List SymRankEvent :=
  (List.range f.blocks.length).flatMap fun b =>
    let inv := tbl.getD b {}
    if !inv.reachable then []
    else symRankEventsOfStmts f inv.effCond inv.regs (f.blockAt b).stmts

/-- Propositional rank-bounds obligation for one `SymRankEvent` (failing closed to `False` on `none`). -/
def symRankInBounds (rev : SymRankEvent) (env : InputEnv) (t : ThreadCtx) (havoc : Nat → Nat)
    (clusterDimX : Nat) : Prop :=
  match rev.rankOpt with
  | some rSym => rSym.eval env t havoc < clusterDimX
  | none => False

/-- Bridge theorem for `MirNoOobClusterRank`: reduces `map_shared_rank` peer-rank checks to `omega`. -/
theorem vcNoOobClusterRank_of_table {f : MirFunc} {tbl : List SymBlockInv}
    {rankEvs revs : List SymRankEvent}
    (hvalid : validSymFunc f tbl = true)
    (hranks : collectSymRankEvents f tbl = rankEvs)
    (hevs : rankEvs = revs)
    (hvc : ∀ env : InputEnv, ∀ t : ThreadCtx, ∀ havoc : Nat → Nat,
      f.validInput env → f.validThread t →
      revs.foldr (fun rev acc =>
        (rev.pathCond.holds env t havoc →
         symRankInBounds rev env t havoc f.clusterDimX) ∧ acc) True) :
    MirNoOobClusterRank f := by
  intro env t b phase regs hIn hTh hreach pfx sfx rankVar r hstmts hr -- Step 1: Pick any reachable `clusterRankCheck rankVar`
  obtain ⟨hb, hrch, havoc, hpc, _, hm⟩ := symReachable_sound hvalid hIn hreach -- Step 2: Obtain block `b`'s symbolic invariant match
  obtain ⟨rev, hrevMem, hpce, hrEq⟩ := symRankEventsOfStmts_correct f env t havoc
    (tbl.getD b {}).effCond (f.blockAt b).term (f.blockAt b).stmts pfx sfx regs (tbl.getD b {}).regs phase rankVar r hm hstmts hr
  have hfMem : rev ∈ revs := by                             -- Step 3: Show `rev` belongs to `symRankEvents`
    rw [← hevs, ← hranks]
    unfold collectSymRankEvents
    rw [List.mem_flatMap]
    refine ⟨b, List.mem_range.mpr hb, ?_⟩
    dsimp only; rw [hrch]; simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte]; exact hrevMem
  have hp := foldr_and_of_mem _ (foldr_holds_to_eval revs (hvc env t havoc hIn hTh)) rev hfMem (by rw [hpce]; exact hpc) -- Step 4: Extract VC
  unfold symRankInBounds at hp
  cases hro : rev.rankOpt with                              -- Step 5: `none` fails closed (`False`), `some rSym` proves `r < clusterDimX`
  | none => rw [hro] at hp; contradiction
  | some rSym => rw [hro] at hp; dsimp only at hp; rw [← hrEq rSym hro]; exact hp

/-- Converse of `symBool_conj_true`: `SymBool.conj a b` evaluates to `true` only if both `a` and `b` do. -/
theorem symBool_conj_true_elim {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat}
    {a b : SymBool} (h : (SymBool.conj a b).eval env t havoc = true) :
    a.eval env t havoc = true ∧ b.eval env t havoc = true := by
  unfold SymBool.conj at h
  split at h                                                  -- Case `a == .lit true`
  · rename_i ha; rw [beq_iff_eq.mp ha]; exact ⟨rfl, h⟩
  · split at h                                                -- Case `b == .lit true`
    · rename_i _ hb; rw [beq_iff_eq.mp hb]; exact ⟨h, rfl⟩
    · split at h                                              -- Case `a == .lit false || b == .lit false`
      · contradiction
      · split at h                                            -- Case `a == b`
        · rename_i _ _ _ hab; rw [← beq_iff_eq.mp hab]; exact ⟨h, h⟩
        · simpa [SymBool.eval] using h                        -- General `.and a b` case

/-- Scan a basic block's statements for a guaranteed 1-element global store to `paramId` while
    conjoining every intra-block `Stmt.assertCond` predicate (`none` if any assert condition is `.unk`). -/
def findBlockGlobalWrite (f : MirFunc) (paramId : Nat) (pCond : SymBool) (sregs : SymRegList) :
    List Stmt → SymRegList × Option SymInt × Option SymBool
  | [] => (sregs, none, some (.lit true))
  | .assign dst (.load _) :: rest =>
      findBlockGlobalWrite f paramId pCond (sregs.setVar dst .unk) rest
  | .assign dst (.loadTmemBase _ cols) :: rest =>
      findBlockGlobalWrite f paramId pCond (sregs.setVar dst (.tmemPtr cols (.const 0))) rest
  | .assign dst rhs :: rest =>
      findBlockGlobalWrite f paramId pCond (sregs.setVar dst (symEvalRhs f sregs rhs)) rest
  | .assertCond _ cond :: rest =>
      let (r', wOpt, aOpt) := findBlockGlobalWrite f paramId pCond sregs rest
      let curA := match sregs.getVar cond, aOpt with
        | .bool cSym, some restSym => some (SymBool.conj cSym restSym)
        | _, _ => none
      (r', wOpt, curA)
  | .store ptr _ :: rest =>
      let sev := symSinglePtrEvent pCond .any true (sregs.getVar ptr)
      let cur := if sev.space == .global && sev.bufId == paramId then sev.indexOpt else none
      let (r', later, aOpt) := findBlockGlobalWrite f paramId pCond sregs rest
      (r', cur.orElse (fun _ => later), aOpt)
  | _ :: rest =>
      findBlockGlobalWrite f paramId pCond sregs rest

/-- Soundness of `findBlockGlobalWrite`: preserves symbolic register correspondence, certifies
    `stmtsNoTrap` whenever the returned assertion predicate holds, and witnesses a concrete global
    write `MemEvent` to `paramId` whenever `some idxSym` is returned. -/
theorem findBlockGlobalWrite_sound (f : MirFunc) (paramId : Nat) (env : InputEnv) (t : ThreadCtx)
    (havoc : Nat → Nat) (pCond : SymBool) (hpc : pCond.eval env t havoc = true) (term : Terminator) :
    ∀ stmts : List Stmt, ∀ regs : RegFile, ∀ sregs : SymRegList, ∀ phase : Nat,
      (∀ v, (sregs.getVar v).matchesVal env t havoc (regs v)) →
      (∀ v, ((findBlockGlobalWrite f paramId pCond sregs stmts).1.getVar v).matchesVal env t havoc
              ((execStmts f env t regs phase term stmts).1 v)) ∧
      (∀ aSym : SymBool, (findBlockGlobalWrite f paramId pCond sregs stmts).2.2 = some aSym →
        aSym.eval env t havoc = true → stmtsNoTrap f env t regs phase term stmts = true) ∧
      (∀ idxSym : SymInt, (findBlockGlobalWrite f paramId pCond sregs stmts).2.1 = some idxSym →
        ∃ ev ∈ (execStmts f env t regs phase term stmts).2.2.1,
          ev.isWrite = true ∧ ev.space = .global ∧ ev.bufId = paramId ∧ ev.index = idxSym.eval env t havoc)
  | [], _, _, _, hmatch => ⟨hmatch, fun _ _ _ => rfl, fun _ h => by contradiction⟩ -- Base case: empty block
  | stmt :: rest, regs, sregs, phase, hmatch => by
      cases stmt with                                         -- Step through the head statement `stmt`
      | assign dst rhs =>
          have hnextGen : ∀ v, ((sregs.setVar dst (symEvalRhs f sregs rhs)).getVar v).matchesVal env t havoc
              ((regs.set dst (evalRhs f env t regs rhs)) v) := by
            intro v; unfold RegFile.set; rw [getVar_setVar]
            by_cases hv : (v == dst) = true
            · simp [hv]; exact symEvalRhs_correct f env t havoc regs sregs hmatch rhs
            · simp [hv]; exact hmatch v
          cases rhs with
          | load ptr =>
              have hnext : ∀ v, ((sregs.setVar dst .unk).getVar v).matchesVal env t havoc ((regs.set dst .unk) v) := by
                intro v; unfold RegFile.set; rw [getVar_setVar]
                by_cases hv : (v == dst) = true <;> simp [hv, SymVal.matchesVal]; exact hmatch v
              obtain ⟨hr, hnt, hw⟩ := findBlockGlobalWrite_sound f paramId env t havoc pCond hpc term rest (regs.set dst .unk) (sregs.setVar dst .unk) phase hnext
              exact ⟨hr, hnt, fun idxSym hs => by
                obtain ⟨ev, hev, hprops⟩ := hw idxSym hs
                exact ⟨ev, List.mem_append_right _ hev, hprops⟩⟩
          | loadTmemBase ptr cols =>
              have hnext : ∀ v, ((sregs.setVar dst (.tmemPtr cols (.const 0))).getVar v).matchesVal env t havoc
                  ((regs.set dst (.tmemPtr cols 0)) v) := by
                intro v; unfold RegFile.set; rw [getVar_setVar]
                by_cases hv : (v == dst) = true <;> simp [hv, SymVal.matchesVal, SymInt.eval]; exact hmatch v
              obtain ⟨hr, hnt, hw⟩ := findBlockGlobalWrite_sound f paramId env t havoc pCond hpc term rest (regs.set dst (.tmemPtr cols 0)) (sregs.setVar dst (.tmemPtr cols (.const 0))) phase hnext
              exact ⟨hr, hnt, fun idxSym hs => by
                obtain ⟨ev, hev, hprops⟩ := hw idxSym hs
                exact ⟨ev, List.mem_append_right _ hev, hprops⟩⟩
          | copy _ | constInt _ _ | constBool _ | sreg _ | index1D | coord2DU32 | paramScalar _ | slicePtr _
          | sliceLen _ | sharedBase _ | binOp _ _ _ _ | checkedOvf _ _ _ _ | cmpOp _ _ _ | notBool _ | intCast _ _
          | ptrOffset _ _ | mbarTryWait _ _ _ | cvtaShared _ | opaqueUint _ _ | ctaUint1 _ _ _ | ctaConst | ctaUnary _ | ctaBin _ _ | opaqueVal =>
              exact findBlockGlobalWrite_sound f paramId env t havoc pCond hpc term rest _ _ phase hnextGen
      | assertCond aid cond =>
          obtain ⟨hr, hnt, hw⟩ := findBlockGlobalWrite f paramId pCond sregs rest |> fun _ =>
            findBlockGlobalWrite_sound f paramId env t havoc pCond hpc term rest regs sregs phase hmatch
          refine ⟨hr, fun aSym ha hev => ?_, hw⟩
          dsimp only [findBlockGlobalWrite] at ha
          have hmc := hmatch cond
          cases hsc : sregs.getVar cond <;>
            cases hrestA : (findBlockGlobalWrite f paramId pCond sregs rest).2.2 <;>
            rw [hsc, hrestA] at ha <;> try contradiction
          rename_i cSym restSym
          injection ha with haEq; subst haEq
          obtain ⟨hcTrue, hrestTrue⟩ := symBool_conj_true_elim hev
          rw [hsc] at hmc; dsimp only [SymVal.matchesVal] at hmc
          rw [hcTrue] at hmc
          dsimp only [stmtsNoTrap]
          rw [hmc]
          simpa using hnt restSym hrestA hrestTrue
      | store ptr val =>
          obtain ⟨hr, hnt, hw⟩ := findBlockGlobalWrite_sound f paramId env t havoc pCond hpc term rest regs sregs phase hmatch
          refine ⟨hr, hnt, fun idxSym hs => ?_⟩
          dsimp only [findBlockGlobalWrite] at hs
          let sev := symSinglePtrEvent pCond .any true (sregs.getVar ptr)
          cases hcur : (if sev.space == .global && sev.bufId == paramId then sev.indexOpt else none) with
          | none =>
              rw [hcur] at hs
              obtain ⟨ev, hev, hprops⟩ := hw idxSym hs
              exact ⟨ev, List.mem_append_right _ hev, hprops⟩
          | some curIdx =>
              rw [hcur] at hs; injection hs with heq; subst heq
              have hmp := hmatch ptr
              cases hspv : sregs.getVar ptr with
              | ptr sp bid off =>
                  rw [hspv] at hmp
                  simp only [sev, symSinglePtrEvent, hspv] at hcur
                  split at hcur <;> try contradiction
                  rename_i hgb
                  injection hcur with hoff; subst hoff
                  rcases Bool.and_eq_true_iff.mp hgb with ⟨hspEq, hbidEq⟩
                  have hsp' : sp = .global := by cases sp <;> simp_all
                  have hbid' : bid = paramId := beq_iff_eq.mp hbidEq
                  subst hsp'; subst hbid'
                  dsimp only [SymVal.matchesVal] at hmp
                  simp only [execStmts, hmp]
                  exact ⟨_, List.mem_append_left _ (List.Mem.head _), rfl, rfl, rfl, rfl⟩
              | condPtr g sp bid off =>
                  rw [hspv] at hmp
                  simp only [sev, symSinglePtrEvent, hspv] at hcur
                  by_cases himp : symImpliesVal pCond g true = true
                  · rw [if_pos himp] at hcur
                    split at hcur <;> try contradiction
                    rename_i hgb
                    injection hcur with hoff; subst hoff
                    rcases Bool.and_eq_true_iff.mp hgb with ⟨hspEq, hbidEq⟩
                    have hsp' : sp = .global := by cases sp <;> simp_all
                    have hbid' : bid = paramId := beq_iff_eq.mp hbidEq
                    subst hsp'; subst hbid'
                    have hmp' := hmp (symImpliesVal_sound hpc g true himp)
                    simp only [execStmts, hmp']
                    exact ⟨_, List.mem_append_left _ (List.Mem.head _), rfl, rfl, rfl, rfl⟩
                  · rw [if_neg himp] at hcur; split at hcur <;> contradiction
              | _ =>
                  simp only [sev, symSinglePtrEvent, hspv] at hcur
                  split at hcur <;> contradiction
      | syncThreads =>
          exact findBlockGlobalWrite_sound f paramId env t havoc pCond hpc term rest regs sregs (phase + 1) hmatch
      | barrierSlotCheck _ | tmaCopyG2S _ _ _ _ | tmemAlloc _ _ | smemDescRead _ _ =>
          obtain ⟨hr, hnt, hw⟩ := findBlockGlobalWrite_sound f paramId env t havoc pCond hpc term rest regs sregs phase hmatch
          exact ⟨hr, hnt, fun idxSym hs => by
            obtain ⟨ev, hev, hprops⟩ := hw idxSym hs
            exact ⟨ev, List.mem_append_right _ hev, hprops⟩⟩
      | clusterRankCheck _ | mbarArrive _ _ _ | mbarArriveExpectTx _ _ _ | tmemAccess _ _ _ _ =>
          exact findBlockGlobalWrite_sound f paramId env t havoc pCond hpc term rest regs sregs phase hmatch

/-- Symbolic leaf collected from entry block `0` for `MirWritesAllElems`:
    `writeIdxOpt = some idxSym` when the path wrote `paramId` at `idxSym`, or `none` when it exited without writing. -/
structure SymWriteLeaf where
  /-- Conjoined branch predicate accumulated from entry block `0` to this leaf. -/
  pathCond : SymBool
  /-- Conjoined `Stmt.assertCond` predicates accumulated along the path up through this leaf. -/
  assertsCond : SymBool := .lit true
  /-- `some idxSym` when the path hit a global store to `paramId`, or `none` on a non-writing exit. -/
  writeIdxOpt : Option SymInt
  deriving Repr, DecidableEq, Lean.ToExpr

/-- Exhaustively enumerate symbolic execution paths from `(b, pCond, aAcc, sregs)` until hitting a
    global write to `paramId` (`writeIdxOpt = some idx`) or a non-writing exit (`writeIdxOpt = none`).
    Returns `none` (failing closed) if `fuel` runs out (cycles), a branch condition is `.unk`, or an
    intermediate `Stmt.assertCond` condition is `.unk`. -/
def collectWriteLeaves (f : MirFunc) (paramId : Nat) :
    Nat → Nat → SymBool → SymBool → SymRegList → Option (List SymWriteLeaf)
  | 0, _, _, _, _ => none
  | fuel + 1, b, pCond, aAcc, sregs =>
      if b ≥ f.blocks.length then none
      else
        let blk := f.blockAt b
        let (outSR, wOpt, aOpt) := findBlockGlobalWrite f paramId pCond sregs blk.stmts
        match aOpt with
        | none => none
        | some blkA =>
            let aAcc' := SymBool.conj aAcc blkA
            match wOpt with
            | some idxSym =>
                some [{ pathCond := pCond, assertsCond := aAcc', writeIdxOpt := some idxSym }]
            | none =>
                match blk.term with
                | .ret | .unreachable =>
                    some [{ pathCond := pCond, assertsCond := aAcc', writeIdxOpt := none }]
                | .goto dst args =>
                    collectWriteLeaves f paramId fuel dst pCond aAcc'
                      (bindSymBlockArgs outSR outSR (f.blockAt dst).args args)
                | .condBr cv tBlk tArgs fBlk fArgs =>
                    match outSR.getVar cv with
                    | .bool cSym =>
                        match collectWriteLeaves f paramId fuel tBlk (.and pCond cSym) aAcc'
                                (bindSymBlockArgs outSR outSR (f.blockAt tBlk).args tArgs),
                              collectWriteLeaves f paramId fuel fBlk (.and pCond (.not cSym)) aAcc'
                                (bindSymBlockArgs outSR outSR (f.blockAt fBlk).args fArgs) with
                        | some l1, some l2 => some (l1 ++ l2)
                        | _, _ => none
                    | _ => none

/-- Soundness of `collectWriteLeaves`: every concrete execution from `(b, regs, phase)` lands in at
    least one symbolic leaf `leaf ∈ leaves` whose `pathCond` holds and whose `some idxSym` (once
    `leaf.assertsCond` holds) witnesses a trap-free `EmitsTrapFreeEvent` global write to `paramId`. -/
theorem collectWriteLeaves_sound (f : MirFunc) (paramId : Nat) (env : InputEnv) (t : ThreadCtx)
    (havoc : Nat → Nat) :
    ∀ (fuel b phase : Nat) (pCond aAcc : SymBool) (regs : RegFile) (sregs : SymRegList) (leaves : List SymWriteLeaf),
      (aAcc.eval env t havoc = true → ReachableTrapFree f env t b regs phase) →
      pCond.eval env t havoc = true →
      (∀ v, (sregs.getVar v).matchesVal env t havoc (regs v)) →
      collectWriteLeaves f paramId fuel b pCond aAcc sregs = some leaves →
      ∃ leaf ∈ leaves,
        leaf.pathCond.eval env t havoc = true ∧
        (leaf.assertsCond.eval env t havoc = true →
          ∀ idxSym : SymInt, leaf.writeIdxOpt = some idxSym →
            ∃ ev : MemEvent, EmitsTrapFreeEvent f env t ev ∧
              ev.isWrite = true ∧ ev.space = .global ∧ ev.bufId = paramId ∧ ev.index = idxSym.eval env t havoc)
  | 0, _, _, _, _, _, _, _, _, _, _, hcol => by contradiction  -- Exhausted fuel returns `none`, contradiction
  | fuel + 1, b, phase, pCond, aAcc, regs, sregs, leaves, hreach, hpc, hmatch, hcol => by
      dsimp only [collectWriteLeaves] at hcol                 -- Unfold one step of `collectWriteLeaves`
      by_cases hge : b ≥ f.blocks.length
      · rw [if_pos hge] at hcol; contradiction                -- Out-of-range block index fails closed
      · rw [if_neg hge] at hcol
        obtain ⟨hmOut, hntOut, hwOut⟩ := findBlockGlobalWrite_sound f paramId env t havoc pCond hpc (f.blockAt b).term (f.blockAt b).stmts regs sregs phase hmatch
        cases ha : (findBlockGlobalWrite f paramId pCond sregs (f.blockAt b).stmts).2.2 with
        | none => rw [ha] at hcol; contradiction              -- Unresolved assert condition fails closed
        | some blkA =>
            rw [ha] at hcol; dsimp only at hcol
            cases hw : (findBlockGlobalWrite f paramId pCond sregs (f.blockAt b).stmts).2.1 with
            | some idxSym =>                                  -- Case 1: Block `b` itself writes `paramId` at `idxSym`
                rw [hw] at hcol; injection hcol with hl; subst hl
                refine ⟨{ pathCond := pCond, assertsCond := SymBool.conj aAcc blkA, writeIdxOpt := some idxSym },
                        List.Mem.head _, hpc, ?_⟩
                intro haAll i0 hi0; injection hi0 with hi; subst hi
                obtain ⟨haAccT, hblkAT⟩ := symBool_conj_true_elim haAll
                obtain ⟨ev, hev, hprops⟩ := hwOut idxSym hw
                exact ⟨ev, ⟨b, phase, regs, hreach haAccT, hntOut blkA ha hblkAT, hev⟩, hprops⟩
            | none =>                                         -- Case 2: Block `b` did not write; follow terminator
                rw [hw] at hcol; dsimp only at hcol
                cases hterm : (f.blockAt b).term with
                | ret | unreachable =>                        -- Non-writing exit records `writeIdxOpt := none`
                    rw [hterm] at hcol; injection hcol with hl; subst hl
                    exact ⟨{ pathCond := pCond, assertsCond := SymBool.conj aAcc blkA, writeIdxOpt := none },
                           List.Mem.head _, hpc, fun _ _ h => by contradiction⟩
                | goto dst args =>                            -- Unconditional `goto` steps via `ReachableTrapFree.stepGoto`
                    rw [hterm] at hcol
                    have hdstLt : dst < f.blocks.length := by
                      cases fuel with
                      | zero => simp [collectWriteLeaves] at hcol
                      | succ _ =>
                          dsimp only [collectWriteLeaves] at hcol
                          by_cases hd : dst ≥ f.blocks.length
                          · rw [if_pos hd] at hcol; contradiction
                          · exact Nat.lt_of_not_ge hd
                    cases hex : execStmts f env t regs phase (f.blockAt b).term (f.blockAt b).stmts with
                    | mk outR restPair =>
                        rcases restPair with ⟨outP, evs, tevs⟩
                        have hmOut' : ∀ v, ((findBlockGlobalWrite f paramId pCond sregs (f.blockAt b).stmts).1.getVar v).matchesVal env t havoc (outR v) := by
                          intro v; have := hmOut v; rw [hex] at this; exact this
                        have hrNext : (SymBool.conj aAcc blkA).eval env t havoc = true →
                            ReachableTrapFree f env t dst (bindBlockArgs outR outR (f.blockAt dst).args args) outP := by
                          intro hconj
                          obtain ⟨ha1, ha2⟩ := symBool_conj_true_elim hconj
                          exact ReachableTrapFree.stepGoto (hreach ha1) (hntOut blkA ha ha2) hex hterm hdstLt
                        have hbind := bindSymBlockArgs_correct hmOut' (f.blockAt dst).args args outR _ hmOut'
                        exact collectWriteLeaves_sound f paramId env t havoc fuel dst outP pCond (SymBool.conj aAcc blkA) _ _ leaves hrNext hpc hbind hcol
                | condBr cv tBlk tArgs fBlk fArgs =>          -- Conditional branch splits into `tBlk` (`cSym`) and `fBlk` (`¬cSym`)
                    rw [hterm] at hcol; dsimp only at hcol
                    cases hcv : (findBlockGlobalWrite f paramId pCond sregs (f.blockAt b).stmts).1.getVar cv <;>
                      rw [hcv] at hcol <;> try contradiction
                    rename_i cSym
                    dsimp only at hcol
                    cases hl1 : collectWriteLeaves f paramId fuel tBlk (.and pCond cSym) (SymBool.conj aAcc blkA)
                            (bindSymBlockArgs (findBlockGlobalWrite f paramId pCond sregs (f.blockAt b).stmts).1
                                              (findBlockGlobalWrite f paramId pCond sregs (f.blockAt b).stmts).1
                                              (f.blockAt tBlk).args tArgs) <;>
                      cases hl2 : collectWriteLeaves f paramId fuel fBlk (.and pCond (.not cSym)) (SymBool.conj aAcc blkA)
                            (bindSymBlockArgs (findBlockGlobalWrite f paramId pCond sregs (f.blockAt b).stmts).1
                                              (findBlockGlobalWrite f paramId pCond sregs (f.blockAt b).stmts).1
                                              (f.blockAt fBlk).args fArgs) <;>
                      rw [hl1, hl2] at hcol <;> try contradiction
                    rename_i l1 l2
                    injection hcol with hl; subst hl
                    cases hex : execStmts f env t regs phase (f.blockAt b).term (f.blockAt b).stmts with
                    | mk outR restPair =>
                        rcases restPair with ⟨outP, evs, tevs⟩
                        have hmOut' : ∀ v, ((findBlockGlobalWrite f paramId pCond sregs (f.blockAt b).stmts).1.getVar v).matchesVal env t havoc (outR v) := by
                          intro v; have := hmOut v; rw [hex] at this; exact this
                        have hcvMatch := hmOut' cv
                        rw [hcv] at hcvMatch; dsimp only [SymVal.matchesVal] at hcvMatch
                        cases hce : cSym.eval env t havoc
                        · rw [hce] at hcvMatch
                          have hfl : fBlk < f.blocks.length := by
                            cases fuel with
                            | zero => simp [collectWriteLeaves] at hl2
                            | succ _ =>
                                dsimp only [collectWriteLeaves] at hl2
                                by_cases hd : fBlk ≥ f.blocks.length
                                · rw [if_pos hd] at hl2; contradiction
                                · exact Nat.lt_of_not_ge hd
                          have hrF : (SymBool.conj aAcc blkA).eval env t havoc = true →
                              ReachableTrapFree f env t fBlk (bindBlockArgs outR outR (f.blockAt fBlk).args fArgs) outP := by
                            intro hconj
                            obtain ⟨ha1, ha2⟩ := symBool_conj_true_elim hconj
                            exact ReachableTrapFree.stepCondFalse (hreach ha1) (hntOut blkA ha ha2) hex hterm hcvMatch hfl
                          have hpcF : (SymBool.and pCond (.not cSym)).eval env t havoc = true := by
                            simp [SymBool.eval, hpc, hce]
                          have hbind := bindSymBlockArgs_correct hmOut' (f.blockAt fBlk).args fArgs outR _ hmOut'
                          obtain ⟨lf, hlfMem, hlfProps⟩ := collectWriteLeaves_sound f paramId env t havoc fuel fBlk outP (.and pCond (.not cSym)) (SymBool.conj aAcc blkA) _ _ l2 hrF hpcF hbind hl2
                          exact ⟨lf, List.mem_append_right _ hlfMem, hlfProps⟩
                        · rw [hce] at hcvMatch
                          have htl : tBlk < f.blocks.length := by
                            cases fuel with
                            | zero => simp [collectWriteLeaves] at hl1
                            | succ _ =>
                                dsimp only [collectWriteLeaves] at hl1
                                by_cases hd : tBlk ≥ f.blocks.length
                                · rw [if_pos hd] at hl1; contradiction
                                · exact Nat.lt_of_not_ge hd
                          have hrT : (SymBool.conj aAcc blkA).eval env t havoc = true →
                              ReachableTrapFree f env t tBlk (bindBlockArgs outR outR (f.blockAt tBlk).args tArgs) outP := by
                            intro hconj
                            obtain ⟨ha1, ha2⟩ := symBool_conj_true_elim hconj
                            exact ReachableTrapFree.stepCondTrue (hreach ha1) (hntOut blkA ha ha2) hex hterm hcvMatch htl
                          have hpcT' : (SymBool.and pCond cSym).eval env t havoc = true := by
                            simp [SymBool.eval, hpc, hce]
                          have hbind := bindSymBlockArgs_correct hmOut' (f.blockAt tBlk).args tArgs outR _ hmOut'
                          obtain ⟨lf, hlfMem, hlfProps⟩ := collectWriteLeaves_sound f paramId env t havoc fuel tBlk outP (.and pCond cSym) (SymBool.conj aAcc blkA) _ _ l1 hrT hpcT' hbind hl1
                          exact ⟨lf, List.mem_append_left _ hlfMem, hlfProps⟩

/-- Propositional obligation for one `SymWriteLeaf` on witness thread `t` for output element `elemIdx`:
    non-writing leaves must be unreachable (`False`), every `Stmt.assertCond` along the path must pass,
    and the written element index must equal `elemIdx`. -/
def symWriteLeafOblig (leaf : SymWriteLeaf) (env : InputEnv) (t : ThreadCtx) (elemIdx : Nat) : Prop :=
  match leaf.writeIdxOpt with
  | none => False
  | some idxSym =>
      leaf.assertsCond.holds env t (fun _ => 0) ∧
      idxSym.eval env t (fun _ => 0) = elemIdx

/-- Synthesize a candidate 1D linear witness-thread index expression from `leaves`:
    evaluates the writing leaf's `SymInt` index on `t₀ = (elemIdx / blockDimX, elemIdx % blockDimX)`,
    supporting identity (`idx`), reverse (`len - 1 - idx`), XOR swaps (`idx ^ mask`), and piecewise
    involutions uniformly while `vcWritesAllElems` + `omega` verify bounds and surjectivity. -/
def buildWriteWitnessSym (f : MirFunc) : List SymWriteLeaf → SymInt
  | [] => .index1D f.blockDimX
  | [lf] => lf.writeIdxOpt.getD (.index1D f.blockDimX)
  | lf :: rest =>
      match lf.writeIdxOpt with
      | some idxSym =>
          if rest.all (·.writeIdxOpt.isNone) then idxSym
          else .ite lf.pathCond idxSym (buildWriteWitnessSym f rest)
      | none => buildWriteWitnessSym f rest

/-- Bridge theorem for `MirWritesAllElems`: instantiates the candidate 1D witness thread `wt` from
    `buildWriteWitnessSym f leaves` evaluated at `t₀ = (elemIdx / blockDimX, elemIdx % blockDimX, 0, 0)`
    and reduces trap-free full output coverage from entry block `0` to `omega` obligations. -/
theorem vcWritesAllElems {f : MirFunc} {paramId : Nat} {leaves : List SymWriteLeaf}
    (hdims : (decide (0 < f.blocks.length) && decide (0 < f.blockDimX) && decide (0 < f.blockDimY)) = true)
    (hleaves : collectWriteLeaves f paramId f.blocks.length 0 (.lit true) (.lit true) (initialSymRegs f) = some leaves)
    (hvc : ∀ env : InputEnv, ∀ elemIdx : Nat, ∀ t₀ wt : ThreadCtx,
      f.validInput env →
      (f.u32Coords = true → env.sliceLen paramId < 4294967296) →
      elemIdx < env.sliceLen paramId →
      f.validThread t₀ →
      t₀.ty = 0 → t₀.by_ = 0 →
      t₀.bx * f.blockDimX + t₀.tx = elemIdx →
      let lin := (buildWriteWitnessSym f leaves).eval env t₀ (fun _ => 0)
      (lin < env.sliceLen paramId) ∧
      (f.validThread wt →
       wt.ty = 0 → wt.by_ = 0 →
       wt.bx * f.blockDimX + wt.tx = lin →
       leaves.foldr (fun lf acc =>
         (lf.pathCond.holds env wt (fun _ => 0) →
          symWriteLeafOblig lf env wt elemIdx) ∧ acc) True)) :
    MirWritesAllElems f paramId := by
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hdims    -- Unpack positive block/CTA dimension bounds
  rcases hdims with ⟨⟨hblkPos, hbxPos⟩, hbyPos⟩
  intro env gridDimX gridDimY hIn hgxU32 hgyPos hu32 hcov elemIdx helem
  let t₀ : ThreadCtx := { bx := elemIdx / f.blockDimX, by_ := 0, tx := elemIdx % f.blockDimX, ty := 0 } -- Seed coordinate `t₀`
  have hlin0 : t₀.bx * f.blockDimX + t₀.tx = elemIdx := by   -- Prove `t₀`'s linear coordinate equals `elemIdx`
    dsimp only [t₀]; rw [Nat.mul_comm]; exact Nat.div_add_mod elemIdx f.blockDimX
  have hbx0Lt : t₀.bx < gridDimX :=                           -- Prove `t₀.bx < gridDimX` from `elemIdx < out.len()`
    (Nat.div_lt_iff_lt_mul hbxPos).mpr (Nat.lt_of_lt_of_le helem hcov)
  have ht0Valid : f.validThread t₀ := by                      -- Prove `t₀` satisfies `MirFunc.validThread`
    refine ⟨Nat.mod_lt elemIdx hbxPos, hbyPos, Nat.lt_of_lt_of_le hbx0Lt hgxU32, by dsimp [t₀]; omega, ?_, ?_⟩
    · split <;> trivial
    · split
      · rename_i hc32; rw [hlin0]; exact Nat.lt_trans helem (hu32 hc32)
      · trivial
  let lin := (buildWriteWitnessSym f leaves).eval env t₀ (fun _ => 0) -- Evaluate inverse witness index `lin`
  let wt : ThreadCtx := { bx := lin / f.blockDimX, by_ := 0, tx := lin % f.blockDimX, ty := 0 } -- Witness thread `wt`
  obtain ⟨hlinLt, hvcWt⟩ := hvc env elemIdx t₀ wt hIn hu32 helem ht0Valid rfl rfl hlin0
  have hlinWt : wt.bx * f.blockDimX + wt.tx = lin := by       -- Prove `wt`'s linear coordinate equals `lin`
    dsimp only [wt]; rw [Nat.mul_comm]; exact Nat.div_add_mod lin f.blockDimX
  have hbxLt : wt.bx < gridDimX :=                            -- Prove `wt.bx` falls within the host-checked `gridDimX`
    (Nat.div_lt_iff_lt_mul hbxPos).mpr (Nat.lt_of_lt_of_le hlinLt hcov)
  have hwtValid : f.validThread wt := by                      -- Prove `wt` satisfies `MirFunc.validThread`
    refine ⟨Nat.mod_lt lin hbxPos, hbyPos, Nat.lt_of_lt_of_le hbxLt hgxU32, by dsimp [wt]; omega, ?_, ?_⟩
    · split <;> trivial
    · split
      · rename_i hc32; rw [hlinWt]; exact Nat.lt_trans hlinLt (hu32 hc32)
      · trivial
  have hinitM := initialSymRegs_correct f env wt (fun _ => 0) -- Match concrete `initialRegs` with `initialSymRegs`
  obtain ⟨lf, hlfMem, hlfPc, hlfEv⟩ := collectWriteLeaves_sound f paramId env wt (fun _ => 0)
    f.blocks.length 0 0 (.lit true) (.lit true) (initialRegs f env) (initialSymRegs f) leaves
    (fun _ => ReachableTrapFree.entry hblkPos) rfl hinitM hleaves -- Trace `wt` from block `0` to an active `lf ∈ leaves`
  have hp := foldr_and_of_mem _ (foldr_holds_to_eval leaves (hvcWt hwtValid rfl rfl hlinWt)) lf hlfMem hlfPc
  unfold symWriteLeafOblig at hp
  cases hwo : lf.writeIdxOpt with                             -- Non-writing leaf (`none`) contradicts `omega`'s `False` proof
  | none => rw [hwo] at hp; contradiction
  | some idxSym =>                                            -- Writing leaf (`some idxSym`) certifies trap-free global write
      rw [hwo] at hp; dsimp only at hp
      obtain ⟨haHolds, hidxEq⟩ := hp
      have haEval := (symBool_eval_iff_holds env wt (fun _ => 0) lf.assertsCond).mpr haHolds
      obtain ⟨ev, hem, hw, hsp, hbid, hidx⟩ := hlfEv haEval idxSym hwo
      exact ⟨wt, ev, hwtValid, hbxLt, hgyPos, hem, hw, hsp, hbid, hidx.trans hidxEq⟩

end CudaOxide.Proofs.Program
