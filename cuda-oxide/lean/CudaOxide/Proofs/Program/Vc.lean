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
open CudaOxide.Contracts.Program

/-- Helper: extract a list-conjunction obligation at `x ∈ xs`. -/
theorem foldr_and_of_mem {α : Type} {P : α → Prop} :
    ∀ xs : List α, xs.foldr (fun x acc => P x ∧ acc) True → ∀ x ∈ xs, P x
  | [], _, _, hmem => nomatch hmem
  | _ :: _, ⟨hp, _⟩, _, List.Mem.head _ => hp
  | _ :: tl, ⟨_, htl⟩, x, List.Mem.tail _ hm => foldr_and_of_mem tl htl x hm

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
    (hevs : (memEvs.filter fun s => s.space == .global && s.bufId == paramId) = sevs)
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
    rw [← hevs, ← hmem, List.mem_filter]; exact ⟨hsev, by simp [hssp, hsp, hsbid, hbid]⟩
  have hp := foldr_and_of_mem _ (foldr_holds_to_eval sevs (hvc env t havoc hIn hTh)) sev hfMem hpc -- Step 4: Extract `omega`'s VC proof for `sev`
  rw [hopt] at hp; dsimp only [Option.getD] at hp           -- Step 5: Unwrap `sev.indexOpt = some idxS`
  rw [← hidxEv]; exact symSpanInBounds_sound hks hdiv hp    -- Step 6: Conclude concrete `ev.index < env.sliceLen paramId`

/-- Soundness of `vcNoOobShared` from pre-computed `memEvs = collectSymMemEvents f tbl`. -/
theorem vcNoOobShared_of_table {f : MirFunc} {tbl : List SymBlockInv}
    {memEvs sevs : List SymMemEvent} {sharedId cap : Nat}
    (hvalid : validSymFunc f tbl = true)
    (hmem : collectSymMemEvents f tbl = memEvs)
    (hcap : (f.sharedDecls.getD sharedId { size := 0 }).size = cap)
    (hevs : (memEvs.filter fun s => s.space == .shared && s.bufId == sharedId) = sevs)
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
    rw [← hevs, ← hmem, List.mem_filter]; exact ⟨hsev, by simp [hssp, hsp, hsbid, hbid]⟩
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

/-- Bridge theorem for `MirNoSharedRace`: prunes barrier-disjoint event pairs via `PhaseSpec.disjoint`
    and `symAsyncOrdered`, reducing remaining conflicts on `sharedId` to `tx₁ = tx₂ ∧ ty₁ = ty₂`. -/
theorem vcNoSharedRace_of_table {f : MirFunc} {tbl : List SymBlockInv}
    {memEvs : List SymMemEvent} {sharedId : Nat}
    {pairs : List (SymMemEvent × SymMemEvent × Option (SymInt × SymInt))}
    (hvalid : validSymFunc f tbl = true)
    (hmem : collectSymMemEvents f tbl = memEvs)
    (hpairs : (let sevs := memEvs.filter fun s => s.space == .shared && s.bufId == sharedId
               ((sevs.flatMap fun s1 => sevs.map fun s2 => (s1, s2)).filter fun p =>
                 (p.1.isWrite || p.2.isWrite) &&
                 !(p.1.phaseSpec.disjoint p.2.phaseSpec) &&
                 !(symAsyncOrdered f p.1 p.2)).map fun p =>
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
    MirNoSharedRace f sharedId := by
  intro env t₁ t₂ ev₁ ev₂ hIn hT1 hT2 hbx hby hem1 hem2 hsp1 hsp2 hb1 hb2 hph hnotOrd hw hidx -- Step 1: Pick two same-CTA events in the same phase
  obtain ⟨h₁, s1, hs1, i1, k1, hk1, hd1, hw1, hssp1, hsbid1, hab1, ho1, hi1, hpc1, hps1⟩ :=
    emitsEvent_mem_collect hvalid hIn hem1                  -- Step 2a: Map `ev₁` to symbolic event `s1`
  obtain ⟨h₂, s2, hs2, i2, k2, hk2, hd2, hw2, hssp2, hsbid2, hab2, ho2, hi2, hpc2, hps2⟩ :=
    emitsEvent_mem_collect hvalid hIn hem2                  -- Step 2b: Map `ev₂` to symbolic event `s2`
  by_cases hdisj : s1.phaseSpec.disjoint s2.phaseSpec = true -- Step 3: Prune if separated by `sync_threads()` (`ev₁.phase ≠ ev₂.phase`)
  · exact False.elim (phaseSpec_disjoint_sound hdisj hps1 hps2 hph)
  · by_cases haord : symAsyncOrdered f s1 s2 = true         -- Step 4: Prune if ordered by a constant `mbarrier` handoff
    · exact False.elim (hnotOrd (symAsyncOrdered_sound haord hab1 hab2 hpc1 hpc2))
    · rw [hmem] at hs1 hs2                                  -- Step 5: Otherwise `(s1, s2)` was kept in `pairs` for `omega`
      have hm1 : s1 ∈ memEvs.filter (fun s => s.space == .shared && s.bufId == sharedId) := by
        rw [List.mem_filter]; exact ⟨hs1, by simp [hssp1, hsp1, hsbid1, hb1]⟩
      have hm2 : s2 ∈ memEvs.filter (fun s => s.space == .shared && s.bufId == sharedId) := by
        rw [List.mem_filter]; exact ⟨hs2, by simp [hssp2, hsp2, hsbid2, hb2]⟩
      have hpair : (s1, s2, extractAsyncSlotPair f s1 s2) ∈ pairs := by
        rw [← hpairs, List.mem_map]
        refine ⟨(s1, s2), ?_, rfl⟩
        rw [List.mem_filter]
        refine ⟨List.mem_flatMap.mpr ⟨s1, hm1, List.mem_map.mpr ⟨s2, hm2, rfl⟩⟩, ?_⟩
        dsimp only
        have hwOr : (s1.isWrite || s2.isWrite) = true := by
          rw [hw1, hw2]; rcases hw with hwL | hwR
          · simp [hwL]
          · simp [hwR]
        have hnd : (!(s1.phaseSpec.disjoint s2.phaseSpec)) = true := by
          cases hc : s1.phaseSpec.disjoint s2.phaseSpec <;> simp_all
        have hna : (!(symAsyncOrdered f s1 s2)) = true := by
          cases hc : symAsyncOrdered f s1 s2 <;> simp_all
        simp [hwOr, hnd, hna]
      have hp0 := foldr_and_of_mem _ (foldr_holds2_to_eval pairs (hvc env t₁ t₂ h₁ h₂ hIn hT1 hT2 hbx hby))
        (s1, s2, extractAsyncSlotPair f s1 s2) hpair hpc1 hpc2 -- Step 6: Extract `omega`'s proof for `(s1, s2)`
      have hp := symAsyncSlotGuard_sound hab1 hab2 hpc1 hpc2 hnotOrd hp0 -- Step 7: Discharge `slot₁ ≠ slot₂` from `¬ asyncOrdered`
      rw [ho1, ho2] at hp; dsimp only [Option.getD] at hp
      have heqIdx : (i1.eval env t₁ h₁ + k1) / s1.elemDiv = (i2.eval env t₂ h₂ + k2) / s2.elemDiv := by
        rw [hi1, hi2, hidx]
      obtain ⟨htx, hty⟩ := symPairRaceOblig_sound hk1 hd1 hk2 hd2 hp heqIdx -- Step 8: Deduce `tx₁ = tx₂ ∧ ty₁ = ty₂`
      cases t₁; cases t₂; simp_all                          -- Step 9: Combine with `bx₁ = bx₂, by₁ = by₂` to get `t₁ = t₂`

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
          | copy _ | constInt _ _ | constBool _ | sreg _ | index1D | paramScalar _ | slicePtr _
          | sliceLen _ | sharedBase _ | binOp _ _ _ _ | cmpOp _ _ _ | notBool _ | intCast _ _
          | ptrOffset _ _ | mbarTryWait _ _ | cvtaShared _ | opaqueUint _ _ | opaqueVal =>
              obtain ⟨rev, hmem, hp, hrEq⟩ := symRankEventsOfStmts_correct f env t havoc pCond term
                rest tl sfx _ _ phase rv r hnextGen htl (by simpa [execStmts] using hr)
              exact ⟨rev, hmem, hp, hrEq⟩
      | clusterRankCheck rv0 =>
          obtain ⟨rev, hmem, hp, hrEq⟩ := symRankEventsOfStmts_correct f env t havoc pCond term
            rest tl sfx regs sregs phase rv r hmatch htl (by simpa [execStmts] using hr)
          exact ⟨rev, List.Mem.tail _ hmem, hp, hrEq⟩
      | store _ _ | barrierSlotCheck _ | assertCond _ _ | mbarArrive _ _ | mbarArriveExpectTx _ _ _
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

end CudaOxide.Proofs.Program
