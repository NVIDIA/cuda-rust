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
  | .cmp .lt _ .ctaidX, _, _, hlt, _ | .cmp .lt _ .ctaidY, _, _, hlt, _
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

/-- Helper: `wordMod w n ≤ n` for every width `w` and natural number `n`. -/
theorem wordMod_le_self : ∀ w n : Nat, wordMod w n ≤ n
  | 0, _ => Nat.le_refl _
  | _ + 1, n => Nat.mod_le n _

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

/-- Extract `env.sliceLen p ≤ 9223372036854775807` from `paramBoundsProp` for slice/disjointSlice params. -/
theorem sliceBounds_get {env : InputEnv} :
    ∀ (ps : List ParamKind) (off p : Nat),
      paramBoundsProp env off ps →
      (match ps[p]? with | some (.slice _) | some .disjointSlice => true | _ => false) = true →
      env.sliceLen (off + p) ≤ 9223372036854775807
  | [], _, _, _, hget => by simp at hget
  | (.scalar _) :: _, _, 0, _, hget => by simp at hget
  | (.scalar _) :: tl, off, p + 1, ⟨_, htl⟩, hget => by
      simp only [List.getElem?_cons_succ] at hget
      simpa [Nat.add_assoc, Nat.add_comm 1 p] using sliceBounds_get tl (off + 1) p htl hget
  | (.slice _) :: _, off, 0, ⟨hhd, _⟩, _ => by simpa using hhd
  | (.slice _) :: tl, off, p + 1, ⟨_, htl⟩, hget => by
      simp only [List.getElem?_cons_succ] at hget
      simpa [Nat.add_assoc, Nat.add_comm 1 p] using sliceBounds_get tl (off + 1) p htl hget
  | .disjointSlice :: _, off, 0, ⟨hhd, _⟩, _ => by simpa using hhd
  | .disjointSlice :: tl, off, p + 1, ⟨_, htl⟩, hget => by
      simp only [List.getElem?_cons_succ] at hget
      simpa [Nat.add_assoc, Nat.add_comm 1 p] using sliceBounds_get tl (off + 1) p htl hget

/-- Non-wrapping upper bound `e.eval ≤ ub` derivable statically from `f`'s thread/parameter widths. -/
def symIntStaticLe (f : MirFunc) : SymInt → Option Nat
  | .const c => some c
  | .tidX => if 0 < f.blockDimX then some (f.blockDimX - 1) else none
  | .tidY => if 0 < f.blockDimY then some (f.blockDimY - 1) else none
  | .ctaidX | .ctaidY => some 4294967295
  | .paramScalar p =>
      match f.params[p]? with
      | some (.scalar w) => if 0 < w && w ≤ 32 then some (2 ^ w - 1) else none
      | _ => none
  | .sliceLen p =>
      match f.params[p]? with
      | some (.slice _) | some .disjointSlice => some 9223372036854775807
      | _ => none
  | .intCast _ s => symIntStaticLe f s
  | _ => none

/-- Soundness of `symIntStaticLe`: returns `some ub` only when `e.eval env t havoc ≤ ub`. -/
theorem symIntStaticLe_sound {f : MirFunc} {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat}
    (hvalidIn : f.validInput env) (hth : f.validThread t) :
    ∀ {e : SymInt} {ub : Nat},
      symIntStaticLe f e = some ub →
      e.eval env t havoc ≤ ub
  | .const _, _, h => by injection h with hc; subst hc; exact Nat.le_refl _
  | .tidX, _, h => by
      unfold symIntStaticLe at h
      split at h <;> try contradiction
      injection h with hc; subst hc
      have htx := hth.1; simp only [SymInt.eval]; omega
  | .tidY, _, h => by
      unfold symIntStaticLe at h
      split at h <;> try contradiction
      injection h with hc; subst hc
      have hty := hth.2.1; simp only [SymInt.eval]; omega
  | .ctaidX, _, h => by
      injection h with hc; subst hc
      have hbx := hth.2.2.1; simp only [SymInt.eval]; omega
  | .ctaidY, _, h => by
      injection h with hc; subst hc
      have hby := hth.2.2.2.1; simp only [SymInt.eval]; omega
  | .paramScalar p, _, h => by
      unfold symIntStaticLe at h
      cases hp : f.params[p]? with
      | none => rw [hp] at h; contradiction
      | some pk =>
          rw [hp] at h; cases pk <;> try contradiction
          rename_i w; dsimp only at h
          split at h <;> try contradiction
          rename_i hbounds; injection h with hc; subst hc
          rcases Bool.and_eq_true_iff.mp hbounds with ⟨hw0, _⟩
          have hlt := scalarBounds_get f.params 0 p w hvalidIn.1 hp (of_decide_eq_true hw0)
          simp only [Nat.zero_add] at hlt
          simp only [SymInt.eval]; omega
  | .sliceLen p, ub, h => by
      unfold symIntStaticLe at h
      have hok : (match f.params[p]? with | some (.slice _) | some .disjointSlice => true | _ => false) = true := by
        cases hp : f.params[p]? <;> rw [hp] at h <;> try contradiction
        rename_i pk; cases pk <;> try contradiction
        all_goals rfl
      have hub : ub = 9223372036854775807 := by
        cases hp : f.params[p]? <;> rw [hp] at h <;> try contradiction
        rename_i pk; cases pk <;> simp at h <;> exact h.symm
      subst hub
      have hsl := sliceBounds_get f.params 0 p hvalidIn.1 hok
      simpa [SymInt.eval] using hsl
  | .intCast dw s, _, h => by
      unfold symIntStaticLe at h
      have hs := symIntStaticLe_sound (havoc := havoc) hvalidIn hth h
      unfold SymInt.eval
      exact Nat.le_trans (wordMod_le_self dw _) hs
  | .index1D _, _, h | .havoc _ _, _, h | .opaqueUint _ _, _, h
  | .binOp _ _ _ _, _, h | .ite _ _ _, _, h => by
      unfold symIntStaticLe at h; contradiction

/-- Check whether `atom` is `lhs < rhs` (peeling double negation `.not (.not ...)`), where
    `lhs` is `x` or `x + off` (`64`-bit, `off ≤ 2^31`) and `rhs ≤ 2^63 - 1`. -/
def atomBoundsBase64 (f : MirFunc) (atom : SymBool) (x : SymInt) : Bool :=
  let core := match atom with | .not (.not p) => p | p => p
  match core with
  | .cmp .lt lhs rhs =>
      let lhsOk :=
        (lhs == x) ||
        match lhs with
        | .binOp .add 64 base (.const off) => (base == x) && decide (off ≤ 2147483648)
        | _ => false
      lhsOk &&
      match symIntStaticLe f rhs with
      | some ub => decide (ub ≤ 9223372036854775807)
      | none => false
  | _ => false

/-- Search conjunction `hyp` for a 64-bit strided loop guard bounding `x < 2^63` when `x < 2^63 + 2^32`. -/
def hypBoundsBase64 (f : MirFunc) : SymBool → SymInt → Bool
  | .and p q, x => hypBoundsBase64 f p x || hypBoundsBase64 f q x
  | atom, x => atomBoundsBase64 f atom x

/-- Soundness of `atomBoundsBase64`: when `x.eval < 9223372041149743104` and `atom` holds, `x.eval < 2^63`. -/
theorem atomBoundsBase64_sound {f : MirFunc} {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat}
    (hvalidIn : f.validInput env) (hth : f.validThread t)
    {atom : SymBool} {x : SymInt}
    (hx : x.eval env t havoc < 9223372041149743104)
    (hatom : atomBoundsBase64 f atom x = true)
    (heval : atom.eval env t havoc = true) :
    x.eval env t havoc < 9223372036854775808 := by
  have hcore : (match atom with | .not (.not p) => p | p => p).eval env t havoc = true := by
    cases atom with
    | not inner =>
        cases inner with
        | not p =>
            dsimp only
            simp only [SymBool.eval] at heval
            cases hpv : p.eval env t havoc <;> simp_all
        | _ => exact heval
    | _ => exact heval
  unfold atomBoundsBase64 at hatom
  generalize hc : (match atom with | .not (.not p) => p | p => p) = core at hatom hcore
  cases core with
  | cmp op lhs rhs =>
      cases op <;> try contradiction
      dsimp only at hatom
      rcases Bool.and_eq_true_iff.mp hatom with ⟨hlhsOk, hrhsOk⟩
      cases hrub : symIntStaticLe f rhs with
      | none => rw [hrub] at hrhsOk; contradiction
      | some ub =>
          rw [hrub] at hrhsOk; dsimp only at hrhsOk
          have hubLe : ub ≤ 9223372036854775807 := of_decide_eq_true hrhsOk
          have hrhsLe := symIntStaticLe_sound (havoc := havoc) hvalidIn hth hrub
          simp only [SymBool.eval, evalCmpOp] at hcore
          have hltRhs : lhs.eval env t havoc < rhs.eval env t havoc := of_decide_eq_true hcore
          rcases Bool.or_eq_true_iff.mp hlhsOk with heqX | haddOff
          · rw [beq_iff_eq.mp heqX] at hltRhs; omega
          · cases lhs with
            | binOp bop bw base rop =>
                cases bop <;> try contradiction
                by_cases hbw : bw = 64
                · subst hbw
                  cases rop with
                  | const off =>
                      rcases Bool.and_eq_true_iff.mp haddOff with ⟨hbaseEq, hoffLe⟩
                      have hoffB : off ≤ 2147483648 := of_decide_eq_true hoffLe
                      rw [beq_iff_eq.mp hbaseEq] at hltRhs
                      have hnoWrap : x.eval env t havoc + off < 2 ^ 64 := by omega
                      have hmodEq : wordMod 64 (x.eval env t havoc + off) = x.eval env t havoc + off :=
                        (wordMod_eq_iff_lt (by decide : 0 < 64)).mpr hnoWrap
                      simp only [SymInt.eval, evalBinOp, hmodEq] at hltRhs
                      omega
                  | _ => contradiction
                · simp [hbw] at haddOff
            | _ => contradiction
  | _ => contradiction

/-- Soundness of `hypBoundsBase64`. -/
theorem hypBoundsBase64_sound {f : MirFunc} {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat}
    (hvalidIn : f.validInput env) (hth : f.validThread t) :
    ∀ {hyp : SymBool} {x : SymInt},
      x.eval env t havoc < 9223372041149743104 →
      hypBoundsBase64 f hyp x = true →
      hyp.eval env t havoc = true →
      x.eval env t havoc < 9223372036854775808
  | .and p q, x, hx, hok, hev => by
      unfold hypBoundsBase64 at hok
      simp only [SymBool.eval, Bool.and_eq_true] at hev
      rcases Bool.or_eq_true_iff.mp hok with hp | hq
      · exact hypBoundsBase64_sound hvalidIn hth hx hp hev.1
      · exact hypBoundsBase64_sound hvalidIn hth hx hq hev.2
  | .lit _, _, hx, hok, hev | .mbarReady _ _, _, hx, hok, hev
  | .cmp _ _ _, _, hx, hok, hev | .not _, _, hx, hok, hev
  | .or _ _, _, hx, hok, hev | .ite _ _ _, _, hx, hok, hev =>
      atomBoundsBase64_sound hvalidIn hth hx hok hev

/-- Decidable strict upper-bound check `e < lim` under path hypothesis `hyp`
    (covering constants, static width bounds, `% m`, `& mask`, strided 64-bit loop steps,
    and wrap-on-limit `if x + 1 == B { 0 } else { x + 1 }` rings). -/
def symIntBoundedUnder (f : MirFunc) (hyp : SymBool) : SymInt → Nat → Bool
  | .const c, lim => decide (c < lim)
  | .binOp op w a b, lim =>
      (match hypImpliesLt hyp (.binOp op w a b) with
       | some bnd => decide (bnd ≤ lim)
       | none => false) ||
      (op == .rem && match asConstInt b with | some m => decide (0 < m) && decide (m ≤ lim) | none => false) ||
      (op == .bitAnd && match asConstInt b with | some mask => decide (mask < lim) | none => false) ||
      (op == .bitAnd && match asConstInt a with | some mask => decide (mask < lim) | none => false) ||
      (op == .add && w == 64 && decide (9223372041149743104 ≤ lim) &&
       match asConstInt b, hypImpliesLt hyp a with
       | some step, some curLim =>
           decide (step ≤ 2147483648) && decide (curLim ≤ 9223372041149743104) &&
           hypBoundsBase64 f hyp a
       | _, _ => false)
  | .ite c tv fv, lim =>
      (symIntBoundedUnder f (.and hyp c) tv lim && symIntBoundedUnder f (.and hyp (.not c)) fv lim) ||
      (match asAddOne fv with
       | some (aw, x) =>
           match hypImpliesLt hyp x with
           | some bnd =>
               decide (0 < aw) && (wordMod aw bnd == bnd) && decide (bnd ≤ lim) &&
               symIntBoundedUnder f hyp tv lim &&
               ((c == .cmp .eq fv (.const bnd)) || (c == .cmp .eq (.const bnd) fv) ||
                (c == .cmp .ge fv (.const bnd)))
           | none => false
       | none => false) ||
      (match asAddOne tv with
       | some (aw, x) =>
           match hypImpliesLt hyp x with
           | some bnd =>
               decide (0 < aw) && (wordMod aw bnd == bnd) && decide (bnd ≤ lim) &&
               symIntBoundedUnder f hyp fv lim &&
               ((c == .cmp .ne tv (.const bnd)) || (c == .cmp .ne (.const bnd) tv) ||
                (c == .cmp .lt tv (.const bnd)))
           | none => false
       | none => false)
  | e, lim =>
      (match hypImpliesLt hyp e with
       | some b => decide (b ≤ lim)
       | none => false) ||
      (match symIntStaticLe f e with
       | some ub => decide (ub < lim)
       | none => false)

/-- Soundness of `symIntBoundedUnder`: guarantees `e.eval env t havoc < lim` whenever `hyp` holds. -/
theorem symIntBoundedUnder_sound {f : MirFunc} {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat}
    (hvalidIn : f.validInput env) (hth : f.validThread t)
    {hyp : SymBool} (hhyp : hyp.eval env t havoc = true) :
    ∀ e : SymInt, ∀ lim : Nat,
      symIntBoundedUnder f hyp e lim = true →
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
            exact symIntBoundedUnder_sound hvalidIn hth (by simp [SymBool.eval, hhyp, hce]) tv lim htv
          · rw [if_neg hce]
            have hcf : c.eval env t havoc = false := Bool.eq_false_iff.mpr hce
            exact symIntBoundedUnder_sound hvalidIn hth (by simp [SymBool.eval, hhyp, hcf]) fv lim hfv
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
                  · simp only [SymInt.eval]; rw [if_pos hce]; exact symIntBoundedUnder_sound hvalidIn hth hhyp tv lim htvB
                  · have hgoal : SymInt.eval env t havoc fv < lim := by
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
                · simp only [SymInt.eval]; rw [if_neg hce]; exact symIntBoundedUnder_sound hvalidIn hth hhyp fv lim hfvB
  | .binOp op w a b, lim, hb => by                          -- Case: `% m`, `& mask`, or 64-bit strided loop step `x + step`
      unfold symIntBoundedUnder at hb
      rcases Bool.or_eq_true_iff.mp hb with h1234 | hstride
      · clear hb
        rcases Bool.or_eq_true_iff.mp h1234 with h123 | handL
        · clear h1234
          rcases Bool.or_eq_true_iff.mp h123 with h12 | handR
          · clear h123
            rcases Bool.or_eq_true_iff.mp h12 with hh | hrem
            · clear h12
              revert hh
              cases hlt : hypImpliesLt hyp (.binOp op w a b)
              · intro hh; contradiction
              · intro hh; exact Nat.lt_of_lt_of_le (hypImpliesLt_sound hlt hhyp) (of_decide_eq_true hh)
            · clear h12
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
          · clear h123
            rcases Bool.and_eq_true_iff.mp handR with ⟨hop, hmB⟩
            have hop' : op = .bitAnd := beq_iff_eq.mp hop; subst hop'
            clear handR hop
            revert hmB
            cases hc : asConstInt b
            · intro hmB; contradiction
            · intro hmB; rw [asConstInt_eq hc]
              exact Nat.lt_of_le_of_lt Nat.and_le_right (of_decide_eq_true hmB)
        · clear h1234
          rcases Bool.and_eq_true_iff.mp handL with ⟨hop, hmA⟩
          have hop' : op = .bitAnd := beq_iff_eq.mp hop; subst hop'
          clear handL hop
          revert hmA
          cases hc : asConstInt a
          · intro hmA; contradiction
          · intro hmA; rw [asConstInt_eq hc]
            exact Nat.lt_of_le_of_lt Nat.and_le_left (of_decide_eq_true hmA)
      · clear hb                                            -- Subcase: strided 64-bit loop step `a + step < 2^63 + 2^32 ≤ lim`
        rcases Bool.and_eq_true_iff.mp hstride with ⟨h123, hrest⟩
        rcases Bool.and_eq_true_iff.mp h123 with ⟨h12, hlim⟩
        rcases Bool.and_eq_true_iff.mp h12 with ⟨hop, hw64⟩
        have hop' : op = .add := beq_iff_eq.mp hop; subst hop'
        have hw' : w = 64 := beq_iff_eq.mp hw64; subst hw'
        have hlim' : 9223372041149743104 ≤ lim := of_decide_eq_true hlim
        cases hc : asConstInt b with
        | none => rw [hc] at hrest; contradiction
        | some step =>
            rw [hc] at hrest
            cases hcur : hypImpliesLt hyp a with
            | none => rw [hcur] at hrest; contradiction
            | some curLim =>
                rw [hcur] at hrest
                rcases Bool.and_eq_true_iff.mp hrest with ⟨hstepCur, hbase⟩
                rcases Bool.and_eq_true_iff.mp hstepCur with ⟨hstep, hcurLe⟩
                have hstep' : step ≤ 2147483648 := of_decide_eq_true hstep
                have hcurLe' : curLim ≤ 9223372041149743104 := of_decide_eq_true hcurLe
                have haCur := hypImpliesLt_sound hcur hhyp
                have haLt63 := hypBoundsBase64_sound hvalidIn hth (by omega) hbase hhyp
                rw [asConstInt_eq hc]
                have hnoWrap : a.eval env t havoc + step < 2 ^ 64 := by omega
                have hmodEq : wordMod 64 (a.eval env t havoc + step) = a.eval env t havoc + step :=
                  (wordMod_eq_iff_lt (by decide : 0 < 64)).mpr hnoWrap
                simp only [SymInt.eval, evalBinOp, hmodEq]
                omega
  | .tidX, lim, hb | .tidY, lim, hb | .ctaidX, lim, hb | .ctaidY, lim, hb
  | .index1D _, lim, hb | .paramScalar _, lim, hb | .sliceLen _, lim, hb
  | .havoc _ _, lim, hb | .opaqueUint _ _, lim, hb | .intCast _ _, lim, hb => by -- Case: leaf bounded by `hypImpliesLt` or `symIntStaticLe`
      unfold symIntBoundedUnder at hb
      rcases Bool.or_eq_true_iff.mp hb with hh | hst
      · cases hlt : hypImpliesLt hyp _ <;> rw [hlt] at hh
        · contradiction
        · exact Nat.lt_of_lt_of_le (hypImpliesLt_sound hlt hhyp) (of_decide_eq_true hh)
      · cases hle : symIntStaticLe f _ <;> rw [hle] at hst
        · contradiction
        · exact Nat.lt_of_le_of_lt (symIntStaticLe_sound hvalidIn hth hle) (of_decide_eq_true hst)

/-- Inspect a backedge `SymInt` expression to suggest a candidate ring/strided upper bound `lim`. -/
def suggestRingBound : SymInt → Option Nat
  | .binOp .rem _ _ (.const m) => if 0 < m then some m else none
  | .binOp .bitAnd _ _ (.const mask) | .binOp .bitAnd _ (.const mask) _ => some (mask + 1)
  | .binOp .add 64 (.havoc _ 64) (.const step) =>
      if 1 < step && step ≤ 2147483648 then some 9223372041149743104 else none
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

/-- Helper: every `SymInt` accepted by `symIntFitsWidth` has `0 < w` and `< 2 ^ w`. -/
theorem symIntFitsWidth_lt {f : MirFunc} (env : InputEnv) (t : ThreadCtx) (havoc : Nat → Nat)
    (hvalidIn : f.validInput env) {w : Nat} :
    ∀ e : SymInt, symIntFitsWidth f w e = true → 0 < w ∧ e.eval env t havoc < 2 ^ w
  | .const c, hfits => by
      simp only [symIntFitsWidth, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hfits
      exact ⟨hfits.1, (wordMod_eq_iff_lt hfits.1).mp hfits.2⟩
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
      have ⟨hw, htv⟩ := symIntFitsWidth_lt env t havoc hvalidIn tv hfits.1
      have ⟨_, hfv⟩ := symIntFitsWidth_lt env t havoc hvalidIn fv hfits.2
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
          have ⟨hw, ha⟩ := symIntFitsWidth_lt env t havoc hvalidIn a hfits
          exact ⟨hw, Nat.lt_of_le_of_lt (Nat.div_le_self _ _) ha⟩
      | rem =>
          simp only [symIntFitsWidth] at hfits
          rcases Bool.or_eq_true_iff.mp hfits with haF | hmF
          · have ⟨hw, ha⟩ := symIntFitsWidth_lt env t havoc hvalidIn a haF
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
          have ⟨hw, ha⟩ := symIntFitsWidth_lt env t havoc hvalidIn a hfits
          exact ⟨hw, Nat.lt_of_le_of_lt (Nat.div_le_self _ _) ha⟩
      | bitAnd =>
          simp only [symIntFitsWidth] at hfits
          rcases Bool.or_eq_true_iff.mp hfits with haF | hbF
          · have ⟨hw, ha⟩ := symIntFitsWidth_lt env t havoc hvalidIn a haF
            exact ⟨hw, Nat.lt_of_le_of_lt Nat.and_le_left ha⟩
          · have ⟨hw, hb⟩ := symIntFitsWidth_lt env t havoc hvalidIn b hbF
            exact ⟨hw, Nat.lt_of_le_of_lt Nat.and_le_right hb⟩
      | bitOr =>
          simp only [symIntFitsWidth, Bool.and_eq_true] at hfits
          have ⟨hw, ha⟩ := symIntFitsWidth_lt env t havoc hvalidIn a hfits.1
          have ⟨_, hb⟩ := symIntFitsWidth_lt env t havoc hvalidIn b hfits.2
          exact ⟨hw, Nat.or_lt_two_pow ha hb⟩
      | bitXor =>
          simp only [symIntFitsWidth, Bool.and_eq_true] at hfits
          have ⟨hw, ha⟩ := symIntFitsWidth_lt env t havoc hvalidIn a hfits.1
          have ⟨_, hb⟩ := symIntFitsWidth_lt env t havoc hvalidIn b hfits.2
          exact ⟨hw, Nat.xor_lt_two_pow ha hb⟩
  | .tidX, hfits | .tidY, hfits | .ctaidX, hfits | .ctaidY, hfits
  | .index1D _, hfits | .sliceLen _, hfits => by
      simp [symIntFitsWidth] at hfits

/-- Soundness of `symIntFitsWidth`: any matching `SymInt` is fixed by `wordMod w`. -/
theorem symIntFitsWidth_sound {f : MirFunc} {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat}
    (hvalidIn : f.validInput env) {w : Nat} {e : SymInt}
    (hfits : symIntFitsWidth f w e = true) :
    wordMod w (e.eval env t havoc) = e.eval env t havoc := by
  have ⟨hw, hlt⟩ := symIntFitsWidth_lt env t havoc hvalidIn e hfits
  exact (wordMod_eq_iff_lt hw).mpr hlt


end CudaOxide.Proofs.Program
