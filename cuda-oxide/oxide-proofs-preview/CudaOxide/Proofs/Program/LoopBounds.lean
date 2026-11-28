/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide.Proofs.Program.Expr

/-!
# Inductive Loop/Ring Bounds & Word-Width Checks (`CudaOxide.Proofs.Program.LoopBounds`)
-/
namespace CudaOxide.Proofs.Program
open CudaOxide.Mir.Program
open CudaOxide.Contracts.Program

/-- Extract a strict constant upper bound `target < B` syntactically present in conjunction `hyp`. -/
def hypImpliesLt (hyp : SymBool) (target : SymInt) : Option Nat :=
  match hyp with
  | .cmp .lt lhs (.const b) => if lhs == target then some b else none
  | .and p q =>
      match hypImpliesLt p target with
      | some b => some b
      | none => hypImpliesLt q target
  | _ => none

/-- Helper: `wordMod w (wordMod w n) = wordMod w n`. -/
theorem wordMod_idem : ∀ w n : Nat, wordMod w (wordMod w n) = wordMod w n
  | 0, _ => rfl
  | _ + 1, _ => by simp [wordMod]

/-- Helper: for `0 < w`, `wordMod w n = n` iff `n < 2 ^ w`. -/
theorem wordMod_eq_iff_lt {w n : Nat} (hw : 0 < w) :
    wordMod w n = n ↔ n < 2 ^ w := by
  cases w with
  | zero => omega
  | succ k =>
      simp only [wordMod]
      constructor
      · intro heq
        rw [← heq]
        exact Nat.mod_lt n (Nat.two_pow_pos (k + 1))
      · exact Nat.mod_eq_of_lt

/-- Soundness of `hypImpliesLt`. -/
theorem hypImpliesLt_sound {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat} :
    ∀ {hyp : SymBool} {target : SymInt} {b : Nat},
      hypImpliesLt hyp target = some b →
      hyp.eval env t havoc = true →
      target.eval env t havoc < b
  | .cmp .lt lhs (.const b0), target, b, hlt, hhyp => by
      unfold hypImpliesLt at hlt
      split at hlt
      · rename_i heq
        injection hlt with hb; subst hb
        rw [← beq_iff_eq.mp heq]
        simp only [SymBool.eval, SymInt.eval, evalCmpOp] at hhyp
        exact of_decide_eq_true hhyp
      · contradiction
  | .and p q, target, b, hlt, hhyp => by
      unfold hypImpliesLt at hlt
      simp only [SymBool.eval, Bool.and_eq_true] at hhyp
      cases hp : hypImpliesLt p target with
      | some b0 =>
          rw [hp] at hlt; injection hlt with hb; subst hb
          exact hypImpliesLt_sound hp hhyp.1
      | none =>
          rw [hp] at hlt; dsimp only at hlt
          exact hypImpliesLt_sound hlt hhyp.2
  | .lit _, _, _, hlt, _ | .mbarReady _ _, _, _, hlt, _
  | .cmp .lt _ .tidX, _, _, hlt, _ | .cmp .lt _ .tidY, _, _, hlt, _
  | .cmp .lt _ .ctaidX, _, _, hlt, _ | .cmp .lt _ .ctaidY, _, _, hlt, _ | .cmp .lt _ (.nctaid _), _, _, hlt, _
  | .cmp .lt _ (.index1D _), _, _, hlt, _ | .cmp .lt _ (.paramScalar _), _, _, hlt, _
  | .cmp .lt _ (.sliceLen _), _, _, hlt, _ | .cmp .lt _ (.havoc _ _), _, _, hlt, _
  | .cmp .lt _ (.opaqueUint _ _), _, _, hlt, _ | .cmp .lt _ (.binOp _ _ _ _), _, _, hlt, _
  | .cmp .lt _ (.intCast _ _), _, _, hlt, _ | .cmp .lt _ (.ite _ _ _), _, _, hlt, _
  | .cmp .le _ _, _, _, hlt, _ | .cmp .gt _ _, _, _, hlt, _
  | .cmp .ge _ _, _, _, hlt, _ | .cmp .eq _ _, _, _, hlt, _ | .cmp .ne _ _, _, _, hlt, _
  | .not _, _, _, hlt, _ | .or _ _, _, _, hlt, _ | .ite _ _ _, _, _, hlt, _ => by
      unfold hypImpliesLt at hlt; contradiction

/-- Extract `(aw, x)` when `e` has the exact shape `.binOp .add aw x (.const 1)`. -/
def asAddOne : SymInt → Option (Nat × SymInt)
  | .binOp .add aw x (.const 1) => some (aw, x)
  | _ => none

/-- Soundness of `asAddOne`. -/
theorem asAddOne_eq {e x : SymInt} {aw : Nat} (h : asAddOne e = some (aw, x)) :
    e = .binOp .add aw x (.const 1) := by
  cases e with
  | binOp op w l r =>
      cases op <;> try contradiction
      cases r with
      | const c =>
          unfold asAddOne at h
          by_cases hc1 : c = 1
          · subst hc1; simp only [Option.some.injEq, Prod.mk.injEq] at h
            rcases h with ⟨rfl, rfl⟩; rfl
          · simp [hc1] at h
      | _ => contradiction
  | _ => contradiction

/-- Extract `some c` when `e` is `.const c`. -/
def asConstInt : SymInt → Option Nat
  | .const c => some c
  | _ => none

/-- Soundness of `asConstInt`. -/
theorem asConstInt_eq {e : SymInt} {c : Nat} (h : asConstInt e = some c) :
    e = .const c := by
  cases e <;> simp [asConstInt] at h
  subst h; rfl

/-- Decidable strict upper-bound check `e < lim` under path hypothesis `hyp`
    (covering constants, `% m`, `& mask`, and wrap-on-limit `if x + 1 == B { 0 } else { x + 1 }` rings). -/
def symIntBoundedUnder (hyp : SymBool) : SymInt → Nat → Bool
  | .const c, lim => decide (c < lim)
  | .binOp op w a b, lim =>
      (match hypImpliesLt hyp (.binOp op w a b) with
       | some bnd => decide (bnd ≤ lim)
       | none => false) ||
      (op == .rem && match asConstInt b with | some m => decide (0 < m) && decide (m ≤ lim) | none => false) ||
      (op == .bitAnd && match asConstInt b with | some mask => decide (mask < lim) | none => false) ||
      (op == .bitAnd && match asConstInt a with | some mask => decide (mask < lim) | none => false)
  | .ite c tv fv, lim =>
      (symIntBoundedUnder (.and hyp c) tv lim && symIntBoundedUnder (.and hyp (.not c)) fv lim) ||
      (match asAddOne fv with
       | some (aw, x) =>
           match hypImpliesLt hyp x with
           | some bnd =>
               decide (0 < aw) && (wordMod aw bnd == bnd) && decide (bnd ≤ lim) &&
               symIntBoundedUnder hyp tv lim &&
               ((c == .cmp .eq fv (.const bnd)) || (c == .cmp .eq (.const bnd) fv) ||
                (c == .cmp .ge fv (.const bnd)))
           | none => false
       | none => false) ||
      (match asAddOne tv with
       | some (aw, x) =>
           match hypImpliesLt hyp x with
           | some bnd =>
               decide (0 < aw) && (wordMod aw bnd == bnd) && decide (bnd ≤ lim) &&
               symIntBoundedUnder hyp fv lim &&
               ((c == .cmp .ne tv (.const bnd)) || (c == .cmp .ne (.const bnd) tv) ||
                (c == .cmp .lt tv (.const bnd)))
           | none => false
       | none => false)
  | e, lim =>
      match hypImpliesLt hyp e with
      | some b => decide (b ≤ lim)
      | none => false

/-- Soundness of `symIntBoundedUnder`: guarantees `e.eval env t havoc < lim` whenever `hyp` holds. -/
theorem symIntBoundedUnder_sound {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat}
    {hyp : SymBool} (hhyp : hyp.eval env t havoc = true) :
    ∀ e : SymInt, ∀ lim : Nat,
      symIntBoundedUnder hyp e lim = true →
      e.eval env t havoc < lim
  | .const c, lim, hb => by                                 -- Case: constant `c < lim` directly
      unfold symIntBoundedUnder at hb; exact of_decide_eq_true hb
  | .ite c tv fv, lim, hb => by                             -- Case: conditional `if c then tv else fv` (incl. ring wraparound)
      unfold symIntBoundedUnder at hb
      have stepEqHelper : ∀ {aw bnd : Nat} {x : SymInt},    -- Helper: `x < bnd < 2^aw` guarantees `wordMod aw (x + 1) = x + 1 ≤ bnd`
          0 < aw → wordMod aw bnd = bnd → bnd ≤ lim →
          hypImpliesLt hyp x = some bnd →
          (SymInt.eval env t havoc (.binOp .add aw x (.const 1)) = x.eval env t havoc + 1 ∧
           x.eval env t havoc + 1 ≤ bnd) := by
        intro aw bnd x haw hwm hble hxlt
        have hxv := hypImpliesLt_sound hxlt hhyp
        have hbndPow : bnd < 2 ^ aw := (wordMod_eq_iff_lt haw).mp hwm
        have hsumPow : x.eval env t havoc + 1 < 2 ^ aw := by omega
        exact ⟨by simp [SymInt.eval, evalBinOp, (wordMod_eq_iff_lt haw).mpr hsumPow], by omega⟩
      rcases Bool.or_eq_true_iff.mp hb with h12 | htvRing
      · rcases Bool.or_eq_true_iff.mp h12 with hboth | hfvRing
        · rcases Bool.and_eq_true_iff.mp hboth with ⟨htv, hfv⟩ -- Subcase 1: both `tv < lim` and `fv < lim` hold independently
          simp only [SymInt.eval]
          by_cases hce : c.eval env t havoc = true
          · rw [if_pos hce]
            exact symIntBoundedUnder_sound (by simp [SymBool.eval, hhyp, hce]) tv lim htv
          · rw [if_neg hce]
            have hcf : c.eval env t havoc = false := Bool.eq_false_iff.mpr hce
            exact symIntBoundedUnder_sound (by simp [SymBool.eval, hhyp, hcf]) fv lim hfv
        · cases hfvA : asAddOne fv with                     -- Subcase 2: `if x + 1 == B { 0 } else { x + 1 }` ring wrap
          | none => rw [hfvA] at hfvRing; contradiction
          | some p =>
              rcases p with ⟨aw, x⟩
              rw [hfvA] at hfvRing; dsimp only at hfvRing
              cases hxlt : hypImpliesLt hyp x with
              | none => rw [hxlt] at hfvRing; contradiction
              | some bnd =>
                  rw [hxlt] at hfvRing; dsimp only at hfvRing
                  simp only [Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq] at hfvRing
                  rcases hfvRing with ⟨⟨⟨⟨haw, hwm⟩, hble⟩, htvB⟩, hcCases⟩
                  have hfvEq := asAddOne_eq hfvA
                  obtain ⟨haddEval, haddLe⟩ := stepEqHelper haw hwm hble hxlt
                  have haddRaw : evalBinOp .add aw (x.eval env t havoc) 1 = x.eval env t havoc + 1 := haddEval
                  by_cases hce : c.eval env t havoc = true
                  · simp only [SymInt.eval]; rw [if_pos hce]; exact symIntBoundedUnder_sound hhyp tv lim htvB
                  · have hgoal : SymInt.eval env t havoc fv < lim := by -- False arm (`x + 1 != bnd` and `x + 1 ≤ bnd`) gives `x + 1 < bnd ≤ lim`
                      rw [hfvEq, haddEval]
                      rcases hcCases with (hc1 | hc2) | hc3
                      · rw [hc1, hfvEq] at hce
                        simp only [SymBool.eval, evalCmpOp, SymInt.eval, haddRaw] at hce
                        have hne : x.eval env t havoc + 1 ≠ bnd := fun he => hce (decide_eq_true he); omega
                      · rw [hc2, hfvEq] at hce
                        simp only [SymBool.eval, evalCmpOp, SymInt.eval, haddRaw] at hce
                        have hne : bnd ≠ x.eval env t havoc + 1 := fun he => hce (decide_eq_true he); omega
                      · rw [hc3, hfvEq] at hce
                        simp only [SymBool.eval, evalCmpOp, SymInt.eval, haddRaw] at hce
                        have hnge : ¬(bnd ≤ x.eval env t havoc + 1) := fun he => hce (decide_eq_true he); omega
                    simp only [SymInt.eval]; rw [if_neg hce]; exact hgoal
      · cases htvA : asAddOne tv with                       -- Subcase 3: inverted `if x + 1 != B { x + 1 } else { 0 }` ring wrap
        | none => rw [htvA] at htvRing; contradiction
        | some p =>
            rcases p with ⟨aw, x⟩
            rw [htvA] at htvRing; dsimp only at htvRing
            cases hxlt : hypImpliesLt hyp x with
            | none => rw [hxlt] at htvRing; contradiction
            | some bnd =>
                rw [hxlt] at htvRing; dsimp only at htvRing
                simp only [Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq] at htvRing
                rcases htvRing with ⟨⟨⟨⟨haw, hwm⟩, hble⟩, hfvB⟩, hcCases⟩
                have htvEq := asAddOne_eq htvA
                obtain ⟨haddEval, haddLe⟩ := stepEqHelper haw hwm hble hxlt
                have haddRaw : evalBinOp .add aw (x.eval env t havoc) 1 = x.eval env t havoc + 1 := haddEval
                by_cases hce : c.eval env t havoc = true
                · have hgoal : SymInt.eval env t havoc tv < lim := by
                    rw [htvEq, haddEval]
                    rcases hcCases with (hc1 | hc2) | hc3
                    · rw [hc1, htvEq] at hce
                      simp only [SymBool.eval, evalCmpOp, SymInt.eval, haddRaw] at hce
                      have hne : x.eval env t havoc + 1 ≠ bnd := of_decide_eq_true hce; omega
                    · rw [hc2, htvEq] at hce
                      simp only [SymBool.eval, evalCmpOp, SymInt.eval, haddRaw] at hce
                      have hne : bnd ≠ x.eval env t havoc + 1 := of_decide_eq_true hce; omega
                    · rw [hc3, htvEq] at hce
                      simp only [SymBool.eval, evalCmpOp, SymInt.eval, haddRaw] at hce
                      have hltB : x.eval env t havoc + 1 < bnd := of_decide_eq_true hce; omega
                  simp only [SymInt.eval]; rw [if_pos hce]; exact hgoal
                · simp only [SymInt.eval]; rw [if_neg hce]; exact symIntBoundedUnder_sound hhyp fv lim hfvB
  | .binOp op w a b, lim, hb => by                          -- Case: `a % m` (`< m ≤ lim`) or `a & mask` (`≤ mask < lim`)
      unfold symIntBoundedUnder at hb
      rcases Bool.or_eq_true_iff.mp hb with h123 | handL
      · clear hb
        rcases Bool.or_eq_true_iff.mp h123 with h12 | handR
        · clear h123
          rcases Bool.or_eq_true_iff.mp h12 with hh | hrem
          · clear h12                                       -- Subcase: explicit `< bnd` guard in `hyp`
            revert hh
            cases hlt : hypImpliesLt hyp (.binOp op w a b)
            · intro hh; contradiction
            · intro hh; exact Nat.lt_of_lt_of_le (hypImpliesLt_sound hlt hhyp) (of_decide_eq_true hh)
          · clear h12                                       -- Subcase: remainder `a % m < m ≤ lim` (`Nat.mod_lt`)
            rcases Bool.and_eq_true_iff.mp hrem with ⟨hop, hmB⟩
            have hop' : op = .rem := beq_iff_eq.mp hop; subst hop'
            clear hrem hop
            revert hmB
            cases hc : asConstInt b
            · intro hmB; contradiction
            · rename_i m; intro hmB
              rcases Bool.and_eq_true_iff.mp hmB with ⟨hm0, hmle⟩
              rw [asConstInt_eq hc]
              have hmod : a.eval env t havoc % m < m := Nat.mod_lt _ (of_decide_eq_true hm0)
              simp only [SymInt.eval, evalBinOp]
              exact Nat.lt_of_lt_of_le hmod (of_decide_eq_true hmle)
        · clear h123                                        -- Subcase: bitwise mask `a & mask ≤ mask < lim` (`Nat.and_le_right`)
          rcases Bool.and_eq_true_iff.mp handR with ⟨hop, hmB⟩
          have hop' : op = .bitAnd := beq_iff_eq.mp hop; subst hop'
          clear handR hop
          revert hmB
          cases hc : asConstInt b
          · intro hmB; contradiction
          · intro hmB; rw [asConstInt_eq hc]
            exact Nat.lt_of_le_of_lt Nat.and_le_right (of_decide_eq_true hmB)
      · clear hb                                            -- Subcase: commuted bitwise mask `mask & b ≤ mask < lim`
        rcases Bool.and_eq_true_iff.mp handL with ⟨hop, hmA⟩
        have hop' : op = .bitAnd := beq_iff_eq.mp hop; subst hop'
        clear handL hop
        revert hmA
        cases hc : asConstInt a
        · intro hmA; contradiction
        · intro hmA; rw [asConstInt_eq hc]
          exact Nat.lt_of_le_of_lt Nat.and_le_left (of_decide_eq_true hmA)
  | .tidX, lim, hb | .tidY, lim, hb | .ctaidX, lim, hb | .ctaidY, lim, hb | .nctaid _, lim, hb
  | .index1D _, lim, hb | .paramScalar _, lim, hb | .sliceLen _, lim, hb
  | .havoc _ _, lim, hb | .opaqueUint _ _, lim, hb | .intCast _ _, lim, hb => by -- Case: leaf bounded directly by `hypImpliesLt`
      unfold symIntBoundedUnder at hb
      cases hlt : hypImpliesLt hyp _ <;> rw [hlt] at hb
      · contradiction
      · exact Nat.lt_of_lt_of_le (hypImpliesLt_sound hlt hhyp) (of_decide_eq_true hb)

/-- Inspect a backedge `SymInt` expression to suggest a candidate ring upper bound `lim`. -/
def suggestRingBound : SymInt → Option Nat
  | .binOp .rem _ _ (.const m) => if 0 < m then some m else none
  | .binOp .bitAnd _ _ (.const mask) | .binOp .bitAnd _ (.const mask) _ => some (mask + 1)
  | .ite (.cmp .eq (.binOp .add _ _ (.const 1)) (.const bnd)) (.const 0) _
  | .ite (.cmp .eq (.const bnd) (.binOp .add _ _ (.const 1))) (.const 0) _
  | .ite (.cmp .ge (.binOp .add _ _ (.const 1)) (.const bnd)) (.const 0) _
  | .ite (.cmp .ne (.binOp .add _ _ (.const 1)) (.const bnd)) _ (.const 0)
  | .ite (.cmp .ne (.const bnd) (.binOp .add _ _ (.const 1))) _ (.const 0)
  | .ite (.cmp .lt (.binOp .add _ _ (.const 1)) (.const bnd)) _ (.const 0) =>
      if 0 < bnd then some bnd else none
  | _ => none


/-- True when `e` is syntactically guaranteed to evaluate to a `Nat` fixed by `wordMod w`
    (covering wrapping arithmetic, quotient/remainder/shift/bitwise ops, and `.ite` joins). -/
def symIntFitsWidth (f : MirFunc) (w : Nat) : SymInt → Bool
  | .const c => decide (0 < w) && (wordMod w c == c)
  | .tidX => decide (32 ≤ w) && decide (f.blockDimX ≤ 4294967296)
  | .tidY => decide (32 ≤ w) && decide (f.blockDimY ≤ 4294967296)
  | .ctaidX | .ctaidY | .nctaid _ => decide (32 ≤ w)
  | .paramScalar p => decide (0 < w) && (f.params[p]? == some (.scalar w))
  | .havoc _ hw | .opaqueUint _ hw => decide (0 < w) && (hw == w)
  | .binOp .add bw _ _ | .binOp .sub bw _ _ | .binOp .mul bw _ _ | .binOp .shl bw _ _ =>
      decide (0 < w) && (bw == w)
  | .binOp .div _ a _ | .binOp .shr _ a _ =>
      symIntFitsWidth f w a
  | .binOp .rem _ a b =>
      symIntFitsWidth f w a ||
      match asConstInt b with
      | some m => decide (0 < w) && decide (0 < m) && (wordMod w (m - 1) == m - 1)
      | none => false
  | .binOp .bitAnd _ a b =>
      symIntFitsWidth f w a || symIntFitsWidth f w b
  | .binOp .bitOr _ a b | .binOp .bitXor _ a b =>
      symIntFitsWidth f w a && symIntFitsWidth f w b
  | .intCast dw _ => decide (0 < w) && (dw == w)
  | .ite _ tv fv => symIntFitsWidth f w tv && symIntFitsWidth f w fv
  | _ => false

/-- Extract `env.scalar p < 2 ^ w` from `paramBoundsProp`. -/
theorem scalarBounds_get {env : InputEnv} :
    ∀ (ps : List ParamKind) (off p w : Nat),
      paramBoundsProp env off ps →
      ps[p]? = some (.scalar w) →
      0 < w →
      env.scalar (off + p) < 2 ^ w
  | [], _, _, _, _, hget, _ => by simp at hget
  | (.scalar w0) :: tl, off, 0, w, ⟨hhd, _⟩, hget, hw => by
      simp only [List.getElem?_cons_zero, Option.some.injEq, ParamKind.scalar.injEq] at hget
      subst hget; rw [if_pos hw] at hhd; simpa using hhd
  | (.scalar _) :: tl, off, p + 1, w, ⟨_, htl⟩, hget, hw => by
      simp only [List.getElem?_cons_succ] at hget
      have ih := scalarBounds_get tl (off + 1) p w htl hget hw
      simpa [Nat.add_assoc, Nat.add_comm 1 p] using ih
  | (.slice _) :: tl, off, 0, _, _, hget, _ => by simp at hget
  | (.slice _) :: tl, off, p + 1, w, ⟨_, htl⟩, hget, hw => by
      simp only [List.getElem?_cons_succ] at hget
      have ih := scalarBounds_get tl (off + 1) p w htl hget hw
      simpa [Nat.add_assoc, Nat.add_comm 1 p] using ih
  | .disjointSlice :: tl, off, 0, _, _, hget, _ => by simp at hget
  | .disjointSlice :: tl, off, p + 1, w, ⟨_, htl⟩, hget, hw => by
      simp only [List.getElem?_cons_succ] at hget
      have ih := scalarBounds_get tl (off + 1) p w htl hget hw
      simpa [Nat.add_assoc, Nat.add_comm 1 p] using ih

/-- Helper: every `SymInt` accepted by `symIntFitsWidth` has `0 < w` and `< 2 ^ w`. -/
theorem symIntFitsWidth_lt {f : MirFunc} (env : InputEnv) (t : ThreadCtx) (havoc : Nat → Nat)
    (hvalidIn : f.validInput env) (hTh : f.validThread t) {w : Nat} :
    ∀ e : SymInt, symIntFitsWidth f w e = true → 0 < w ∧ e.eval env t havoc < 2 ^ w
  | .const c, hfits => by
      simp only [symIntFitsWidth, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hfits
      exact ⟨hfits.1, (wordMod_eq_iff_lt hfits.1).mp hfits.2⟩
  | .tidX, hfits => by
      simp only [symIntFitsWidth, Bool.and_eq_true, decide_eq_true_eq] at hfits
      have hpow : (4294967296 : Nat) ≤ 2 ^ w := Nat.le_trans (by decide : 4294967296 ≤ 2 ^ 32) (Nat.pow_le_pow_right (by omega) hfits.1)
      exact ⟨by omega, Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le hTh.1 hfits.2) hpow⟩
  | .tidY, hfits => by
      simp only [symIntFitsWidth, Bool.and_eq_true, decide_eq_true_eq] at hfits
      have hpow : (4294967296 : Nat) ≤ 2 ^ w := Nat.le_trans (by decide : 4294967296 ≤ 2 ^ 32) (Nat.pow_le_pow_right (by omega) hfits.1)
      exact ⟨by omega, Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le hTh.2.1 hfits.2) hpow⟩
  | .ctaidX, hfits => by
      simp only [symIntFitsWidth, decide_eq_true_eq] at hfits
      have hpow : (4294967296 : Nat) ≤ 2 ^ w := Nat.le_trans (by decide : 4294967296 ≤ 2 ^ 32) (Nat.pow_le_pow_right (by omega) hfits)
      exact ⟨by omega, Nat.lt_of_lt_of_le hTh.2.2.1 hpow⟩
  | .ctaidY, hfits => by
      simp only [symIntFitsWidth, decide_eq_true_eq] at hfits
      have hpow : (4294967296 : Nat) ≤ 2 ^ w := Nat.le_trans (by decide : 4294967296 ≤ 2 ^ 32) (Nat.pow_le_pow_right (by omega) hfits)
      exact ⟨by omega, Nat.lt_of_lt_of_le hTh.2.2.2.1 hpow⟩
  | .nctaid d, hfits => by
      simp only [symIntFitsWidth, decide_eq_true_eq] at hfits
      have hpow : (4294967296 : Nat) ≤ 2 ^ w := Nat.le_trans (by decide : 4294967296 ≤ 2 ^ 32) (Nat.pow_le_pow_right (by omega) hfits)
      have hmod : wordMod 32 (env.nctaid d) < 4294967296 := by
        unfold wordMod; exact Nat.mod_lt _ (by decide)
      exact ⟨by omega, Nat.lt_of_lt_of_le hmod hpow⟩
  | .paramScalar p, hfits => by
      simp only [symIntFitsWidth, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hfits
      exact ⟨hfits.1, by simpa [SymInt.eval] using scalarBounds_get f.params 0 p w hvalidIn.1 hfits.2 hfits.1⟩
  | .havoc vid hw, hfits => by
      simp only [symIntFitsWidth, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hfits
      rw [hfits.2]
      exact ⟨hfits.1, (wordMod_eq_iff_lt hfits.1).mp (wordMod_idem w _)⟩
  | .opaqueUint id hw, hfits => by
      simp only [symIntFitsWidth, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hfits
      rw [hfits.2]
      exact ⟨hfits.1, (wordMod_eq_iff_lt hfits.1).mp (wordMod_idem w _)⟩
  | .intCast dw s, hfits => by
      simp only [symIntFitsWidth, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hfits
      rw [hfits.2]
      exact ⟨hfits.1, (wordMod_eq_iff_lt hfits.1).mp (wordMod_idem w _)⟩
  | .ite c tv fv, hfits => by
      simp only [symIntFitsWidth, Bool.and_eq_true] at hfits
      have ⟨hw, htv⟩ := symIntFitsWidth_lt env t havoc hvalidIn hTh tv hfits.1
      have ⟨_, hfv⟩ := symIntFitsWidth_lt env t havoc hvalidIn hTh fv hfits.2
      refine ⟨hw, ?_⟩
      simp only [SymInt.eval]
      split
      · exact htv
      · exact hfv
  | .binOp op bw a b, hfits => by
      cases op with
      | add =>
          simp only [symIntFitsWidth, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hfits
          rw [hfits.2]
          exact ⟨hfits.1, (wordMod_eq_iff_lt hfits.1).mp (wordMod_idem w _)⟩
      | sub =>
          simp only [symIntFitsWidth, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hfits
          refine ⟨hfits.1, (wordMod_eq_iff_lt hfits.1).mp ?_⟩
          rw [hfits.2]; simp only [SymInt.eval, evalBinOp]
          split <;> exact wordMod_idem w _
      | mul =>
          simp only [symIntFitsWidth, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hfits
          rw [hfits.2]
          exact ⟨hfits.1, (wordMod_eq_iff_lt hfits.1).mp (wordMod_idem w _)⟩
      | shl =>
          simp only [symIntFitsWidth, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hfits
          rw [hfits.2]
          exact ⟨hfits.1, (wordMod_eq_iff_lt hfits.1).mp (wordMod_idem w _)⟩
      | div =>
          simp only [symIntFitsWidth] at hfits
          have ⟨hw, ha⟩ := symIntFitsWidth_lt env t havoc hvalidIn hTh a hfits
          exact ⟨hw, Nat.lt_of_le_of_lt (Nat.div_le_self _ _) ha⟩
      | rem =>
          simp only [symIntFitsWidth] at hfits
          rcases Bool.or_eq_true_iff.mp hfits with haF | hmF
          · have ⟨hw, ha⟩ := symIntFitsWidth_lt env t havoc hvalidIn hTh a haF
            exact ⟨hw, Nat.lt_of_le_of_lt (Nat.mod_le _ _) ha⟩
          · revert hmF
            cases hc : asConstInt b
            · intro hmF; contradiction
            · rename_i m; intro hmF
              simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hmF
              rcases hmF with ⟨⟨hw, hm0⟩, hwm⟩
              have hm1lt : m - 1 < 2 ^ w := (wordMod_eq_iff_lt hw).mp hwm
              rw [asConstInt_eq hc]
              have hmod : a.eval env t havoc % m < m := Nat.mod_lt _ hm0
              simp only [SymInt.eval, evalBinOp]
              exact ⟨hw, by omega⟩
      | shr =>
          simp only [symIntFitsWidth] at hfits
          have ⟨hw, ha⟩ := symIntFitsWidth_lt env t havoc hvalidIn hTh a hfits
          exact ⟨hw, Nat.lt_of_le_of_lt (Nat.div_le_self _ _) ha⟩
      | bitAnd =>
          simp only [symIntFitsWidth] at hfits
          rcases Bool.or_eq_true_iff.mp hfits with haF | hbF
          · have ⟨hw, ha⟩ := symIntFitsWidth_lt env t havoc hvalidIn hTh a haF
            exact ⟨hw, Nat.lt_of_le_of_lt Nat.and_le_left ha⟩
          · have ⟨hw, hb⟩ := symIntFitsWidth_lt env t havoc hvalidIn hTh b hbF
            exact ⟨hw, Nat.lt_of_le_of_lt Nat.and_le_right hb⟩
      | bitOr =>
          simp only [symIntFitsWidth, Bool.and_eq_true] at hfits
          have ⟨hw, ha⟩ := symIntFitsWidth_lt env t havoc hvalidIn hTh a hfits.1
          have ⟨_, hb⟩ := symIntFitsWidth_lt env t havoc hvalidIn hTh b hfits.2
          exact ⟨hw, Nat.or_lt_two_pow ha hb⟩
      | bitXor =>
          simp only [symIntFitsWidth, Bool.and_eq_true] at hfits
          have ⟨hw, ha⟩ := symIntFitsWidth_lt env t havoc hvalidIn hTh a hfits.1
          have ⟨_, hb⟩ := symIntFitsWidth_lt env t havoc hvalidIn hTh b hfits.2
          exact ⟨hw, Nat.xor_lt_two_pow ha hb⟩
  | .index1D _, hfits | .sliceLen _, hfits => by
      simp [symIntFitsWidth] at hfits

/-- Soundness of `symIntFitsWidth`: any matching `SymInt` is fixed by `wordMod w`. -/
theorem symIntFitsWidth_sound {f : MirFunc} {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat}
    (hvalidIn : f.validInput env) (hTh : f.validThread t) {w : Nat} {e : SymInt}
    (hfits : symIntFitsWidth f w e = true) :
    wordMod w (e.eval env t havoc) = e.eval env t havoc := by
  have ⟨hw, hlt⟩ := symIntFitsWidth_lt env t havoc hvalidIn hTh e hfits
  exact (wordMod_eq_iff_lt hw).mpr hlt


end CudaOxide.Proofs.Program
