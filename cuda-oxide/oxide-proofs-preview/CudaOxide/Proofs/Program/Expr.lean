/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import Lean
import CudaOxide.Contracts.Program

/-!
# Symbolic Expressions, Subsumption & Barrier Phases (`CudaOxide.Proofs.Program.Expr`)
-/
namespace CudaOxide.Proofs.Program
open CudaOxide.Mir.Program
open CudaOxide.Contracts.Program

mutual
/-- Symbolic unsigned integer expression over `(env, t, havoc)`. -/
inductive SymInt where
  /-- Constant literal `n`. -/
  | const (n : Nat)
  /-- Hardware `threadIdx.x`. -/
  | tidX
  /-- Hardware `threadIdx.y`. -/
  | tidY
  /-- Hardware `blockIdx.x`. -/
  | ctaidX
  /-- Hardware `blockIdx.y`. -/
  | ctaidY
  /-- Hardware 32-bit `gridDim.{x, y, z}` (`dim = 0, 1, 2`). -/
  | nctaid (dim : Nat)
  /-- 64-bit 1D global thread index `(bx * blockDimX + tx) % 2^64`. -/
  | index1D (blockDimX : Nat)
  /-- Scalar kernel parameter `param`. -/
  | paramScalar (param : Nat)
  /-- Slice `.len()` of kernel parameter `param`. -/
  | sliceLen (param : Nat)
  /-- Universally quantified loop-header block parameter `(varId, width)`. -/
  | havoc (varId width : Nat)
  /-- Per-thread opaque `width`-bit unsigned value `(id, width)`. -/
  | opaqueUint (id width : Nat)
  /-- Binary integer operation `evalBinOp op width a b`. -/
  | binOp (op : BinOp) (width : Nat) (lhs rhs : SymInt)
  /-- Unsigned integer cast `wordMod dstWidth src`. -/
  | intCast (dstWidth : Nat) (src : SymInt)
  /-- Conditional join value `if cond then tVal else fVal`. -/
  | ite (cond : SymBool) (tVal fVal : SymInt)
  deriving Repr, DecidableEq, Lean.ToExpr

/-- Symbolic boolean predicate over `(env, t, havoc)`. -/
inductive SymBool where
  /-- Constant boolean (`true` or `false`). -/
  | lit (b : Bool)
  /-- Hardware `mbarrier_try_wait_parity(&BAR[slot], parity)` ready predicate on `(barId, slot)`. -/
  | mbarReady (barId : Nat) (slot : SymInt)
  /-- Integer comparison `evalCmpOp op lhs rhs`. -/
  | cmp (op : CmpOp) (lhs rhs : SymInt)
  /-- Logical negation `!p`. -/
  | not (p : SymBool)
  /-- Conjunction `p && q`. -/
  | and (p q : SymBool)
  /-- Disjunction `p || q`. -/
  | or (p q : SymBool)
  /-- Conditional boolean `if c then tB else fB`. -/
  | ite (c tB fB : SymBool)
  deriving Repr, DecidableEq, Lean.ToExpr
end

mutual
/-- Fast non-dependent boolean equality on `SymInt` (avoiding `DecidableEq` proof-term construction in `whnf`). -/
def SymInt.beq : SymInt → SymInt → Bool
  | .const a, .const b => a == b
  | .tidX, .tidX | .tidY, .tidY | .ctaidX, .ctaidX | .ctaidY, .ctaidY => true
  | .nctaid a, .nctaid b | .index1D a, .index1D b | .paramScalar a, .paramScalar b | .sliceLen a, .sliceLen b => a == b
  | .havoc v1 w1, .havoc v2 w2 | .opaqueUint v1 w1, .opaqueUint v2 w2 => v1 == v2 && w1 == w2
  | .binOp o1 w1 l1 r1, .binOp o2 w2 l2 r2 =>
      o1 == o2 && w1 == w2 && SymInt.beq l1 l2 && SymInt.beq r1 r2
  | .intCast w1 s1, .intCast w2 s2 => w1 == w2 && SymInt.beq s1 s2
  | .ite c1 t1 f1, .ite c2 t2 f2 =>
      SymBool.beq c1 c2 && SymInt.beq t1 t2 && SymInt.beq f1 f2
  | _, _ => false

/-- Fast non-dependent boolean equality on `SymBool` (avoiding `DecidableEq` proof-term construction in `whnf`). -/
def SymBool.beq : SymBool → SymBool → Bool
  | .lit a, .lit b => a == b
  | .mbarReady b1 s1, .mbarReady b2 s2 => b1 == b2 && SymInt.beq s1 s2
  | .cmp o1 l1 r1, .cmp o2 l2 r2 => o1 == o2 && SymInt.beq l1 l2 && SymInt.beq r1 r2
  | .not p1, .not p2 => SymBool.beq p1 p2
  | .and p1 q1, .and p2 q2 | .or p1 q1, .or p2 q2 => SymBool.beq p1 p2 && SymBool.beq q1 q2
  | .ite c1 t1 f1, .ite c2 t2 f2 =>
      SymBool.beq c1 c2 && SymBool.beq t1 t2 && SymBool.beq f1 f2
  | _, _ => false
end

instance : BEq SymInt := ⟨SymInt.beq⟩
instance : BEq SymBool := ⟨SymBool.beq⟩


mutual
theorem symInt_beq_refl : ∀ a : SymInt, SymInt.beq a a = true
  | .const _ => by simp [SymInt.beq]
  | .tidX | .tidY | .ctaidX | .ctaidY => rfl
  | .nctaid _ | .index1D _ | .paramScalar _ | .sliceLen _ => by simp [SymInt.beq]
  | .havoc _ _ | .opaqueUint _ _ => by simp [SymInt.beq]
  | .binOp _ _ l r => by simp [SymInt.beq, symInt_beq_refl l, symInt_beq_refl r]
  | .intCast _ s => by simp [SymInt.beq, symInt_beq_refl s]
  | .ite c t f => by simp [SymInt.beq, symBool_beq_refl c, symInt_beq_refl t, symInt_beq_refl f]

theorem symBool_beq_refl : ∀ p : SymBool, SymBool.beq p p = true
  | .lit _ => by simp [SymBool.beq]
  | .mbarReady _ s => by simp [SymBool.beq, symInt_beq_refl s]
  | .cmp _ l r => by simp [SymBool.beq, symInt_beq_refl l, symInt_beq_refl r]
  | .not p => by simp [SymBool.beq, symBool_beq_refl p]
  | .and p q | .or p q => by simp [SymBool.beq, symBool_beq_refl p, symBool_beq_refl q]
  | .ite c t f => by simp [SymBool.beq, symBool_beq_refl c, symBool_beq_refl t, symBool_beq_refl f]
end

mutual
theorem symInt_eq_of_beq : (a b : SymInt) → SymInt.beq a b = true → a = b
  | .const _, b, h => by cases b <;> simp [SymInt.beq] at h; exact congrArg _ h
  | .tidX, b, h => by cases b <;> simp [SymInt.beq] at h; rfl
  | .tidY, b, h => by cases b <;> simp [SymInt.beq] at h; rfl
  | .ctaidX, b, h => by cases b <;> simp [SymInt.beq] at h; rfl
  | .ctaidY, b, h => by cases b <;> simp [SymInt.beq] at h; rfl
  | .nctaid _, b, h => by cases b <;> simp [SymInt.beq] at h; exact congrArg _ h
  | .index1D _, b, h => by cases b <;> simp [SymInt.beq] at h; exact congrArg _ h
  | .paramScalar _, b, h => by cases b <;> simp [SymInt.beq] at h; exact congrArg _ h
  | .sliceLen _, b, h => by cases b <;> simp [SymInt.beq] at h; exact congrArg _ h
  | .havoc _ _, b, h => by cases b <;> simp [SymInt.beq] at h; rw [h.1, h.2]
  | .opaqueUint _ _, b, h => by cases b <;> simp [SymInt.beq] at h; rw [h.1, h.2]
  | .binOp _ _ l1 r1, b, h => by
      cases b <;> simp [SymInt.beq] at h
      rename_i o2 w2 l2 r2
      rw [h.1.1.1, h.1.1.2, symInt_eq_of_beq l1 l2 h.1.2, symInt_eq_of_beq r1 r2 h.2]
  | .intCast _ s1, b, h => by
      cases b <;> simp [SymInt.beq] at h
      rename_i w2 s2
      rw [h.1, symInt_eq_of_beq s1 s2 h.2]
  | .ite c1 t1 f1, b, h => by
      cases b <;> simp [SymInt.beq] at h
      rename_i c2 t2 f2
      rw [symBool_eq_of_beq c1 c2 h.1.1, symInt_eq_of_beq t1 t2 h.1.2, symInt_eq_of_beq f1 f2 h.2]

theorem symBool_eq_of_beq : (p q : SymBool) → SymBool.beq p q = true → p = q
  | .lit _, q, h => by cases q <;> simp [SymBool.beq] at h; exact congrArg _ h
  | .mbarReady _ s1, q, h => by
      cases q <;> simp [SymBool.beq] at h
      rename_i b2 s2
      rw [h.1, symInt_eq_of_beq s1 s2 h.2]
  | .cmp _ l1 r1, q, h => by
      cases q <;> simp [SymBool.beq] at h
      rename_i o2 l2 r2
      rw [h.1.1, symInt_eq_of_beq l1 l2 h.1.2, symInt_eq_of_beq r1 r2 h.2]
  | .not p1, q, h => by
      cases q <;> simp [SymBool.beq] at h
      rename_i p2
      exact congrArg _ (symBool_eq_of_beq p1 p2 h)
  | .and p1 q1, q, h => by
      cases q <;> simp [SymBool.beq] at h
      rename_i p2 q2
      rw [symBool_eq_of_beq p1 p2 h.1, symBool_eq_of_beq q1 q2 h.2]
  | .or p1 q1, q, h => by
      cases q <;> simp [SymBool.beq] at h
      rename_i p2 q2
      rw [symBool_eq_of_beq p1 p2 h.1, symBool_eq_of_beq q1 q2 h.2]
  | .ite c1 t1 f1, q, h => by
      cases q <;> simp [SymBool.beq] at h
      rename_i c2 t2 f2
      rw [symBool_eq_of_beq c1 c2 h.1.1, symBool_eq_of_beq t1 t2 h.1.2, symBool_eq_of_beq f1 f2 h.2]
end

instance : LawfulBEq SymInt where
  rfl := symInt_beq_refl _
  eq_of_beq := symInt_eq_of_beq _ _

instance : LawfulBEq SymBool where
  rfl := symBool_beq_refl _
  eq_of_beq := symBool_eq_of_beq _ _

mutual
/-- Evaluate `SymInt` to a concrete `Nat` under `(env, t, havoc)`. -/
def SymInt.eval (env : InputEnv) (t : ThreadCtx) (havoc : Nat → Nat) : SymInt → Nat
  | .const n => n
  | .tidX => t.tx
  | .tidY => t.ty
  | .ctaidX => t.bx
  | .ctaidY => t.by_
  | .nctaid d => wordMod 32 (env.nctaid d)
  | .index1D bdx => wordMod 64 (t.bx * bdx + t.tx)
  | .paramScalar p => env.scalar p
  | .sliceLen p => env.sliceLen p
  | .havoc vid w => wordMod w (havoc vid)
  | .opaqueUint id w => wordMod w (env.opaqueUint t id)
  | .binOp op w a b => evalBinOp op w (a.eval env t havoc) (b.eval env t havoc)
  | .intCast dw s => wordMod dw (s.eval env t havoc)
  | .ite c tv fv => if c.eval env t havoc then tv.eval env t havoc else fv.eval env t havoc

/-- Evaluate `SymBool` to a concrete `Bool` under `(env, t, havoc)`. -/
def SymBool.eval (env : InputEnv) (t : ThreadCtx) (havoc : Nat → Nat) : SymBool → Bool
  | .lit b => b
  | .mbarReady barId slot => env.mbarFlag barId (slot.eval env t havoc)
  | .cmp op a b => evalCmpOp op (a.eval env t havoc) (b.eval env t havoc)
  | .not p => !(p.eval env t havoc)
  | .and p q => p.eval env t havoc && q.eval env t havoc
  | .or p q => p.eval env t havoc || q.eval env t havoc
  | .ite c tb fb => if c.eval env t havoc then tb.eval env t havoc else fb.eval env t havoc
end

mutual
/-- True when `SymInt` references no `.havoc` variable in `keys`. -/
def SymInt.disjointHavoc (keys : List Nat) : SymInt → Bool
  | .const _ | .tidX | .tidY | .ctaidX | .ctaidY | .nctaid _ | .index1D _ | .paramScalar _ | .sliceLen _ | .opaqueUint _ _ => true
  | .havoc vid _ => !(keys.contains vid)
  | .binOp _ _ a b => a.disjointHavoc keys && b.disjointHavoc keys
  | .intCast _ s => s.disjointHavoc keys
  | .ite c tv fv => c.disjointHavoc keys && tv.disjointHavoc keys && fv.disjointHavoc keys

/-- True when `SymBool` references no `.havoc` variable in `keys`. -/
def SymBool.disjointHavoc (keys : List Nat) : SymBool → Bool
  | .lit _ => true
  | .mbarReady _ s => s.disjointHavoc keys
  | .cmp _ a b => a.disjointHavoc keys && b.disjointHavoc keys
  | .not p => p.disjointHavoc keys
  | .and p q | .or p q => p.disjointHavoc keys && q.disjointHavoc keys
  | .ite c tb fb => c.disjointHavoc keys && tb.disjointHavoc keys && fb.disjointHavoc keys
end

mutual
/-- Evaluating a `SymInt` disjoint from `keys` is invariant when `havoc₁` and `havoc₂` agree outside `keys`. -/
theorem symInt_eval_congr {keys : List Nat} {env : InputEnv} {t : ThreadCtx}
    {h₁ h₂ : Nat → Nat} (hagree : ∀ k, k ∉ keys → h₁ k = h₂ k) :
    ∀ e : SymInt, e.disjointHavoc keys = true → e.eval env t h₁ = e.eval env t h₂
  | .const _, _ | .tidX, _ | .tidY, _ | .ctaidX, _ | .ctaidY, _ | .nctaid _, _
  | .index1D _, _ | .paramScalar _, _ | .sliceLen _, _ | .opaqueUint _ _, _ => rfl
  | .havoc vid _, hd => by
      simp only [SymInt.disjointHavoc, Bool.not_eq_true', List.contains_eq_mem,
                 decide_eq_false_iff_not] at hd
      simp only [SymInt.eval, hagree vid hd]
  | .binOp _ _ a b, hd => by
      simp only [SymInt.disjointHavoc, Bool.and_eq_true] at hd
      simp only [SymInt.eval, symInt_eval_congr hagree a hd.1, symInt_eval_congr hagree b hd.2]
  | .intCast _ s, hd => by
      simp only [SymInt.disjointHavoc] at hd
      simp only [SymInt.eval, symInt_eval_congr hagree s hd]
  | .ite c tv fv, hd => by
      simp only [SymInt.disjointHavoc, Bool.and_eq_true] at hd
      simp only [SymInt.eval, symBool_eval_congr hagree c hd.1.1,
                 symInt_eval_congr hagree tv hd.1.2, symInt_eval_congr hagree fv hd.2]

/-- Evaluating a `SymBool` disjoint from `keys` is invariant when `havoc₁` and `havoc₂` agree outside `keys`. -/
theorem symBool_eval_congr {keys : List Nat} {env : InputEnv} {t : ThreadCtx}
    {h₁ h₂ : Nat → Nat} (hagree : ∀ k, k ∉ keys → h₁ k = h₂ k) :
    ∀ p : SymBool, p.disjointHavoc keys = true → p.eval env t h₁ = p.eval env t h₂
  | .lit _, _ => rfl
  | .mbarReady _ s, hd => by
      simp only [SymBool.disjointHavoc] at hd
      simp only [SymBool.eval, symInt_eval_congr hagree s hd]
  | .cmp _ a b, hd => by
      simp only [SymBool.disjointHavoc, Bool.and_eq_true] at hd
      simp only [SymBool.eval, symInt_eval_congr hagree a hd.1, symInt_eval_congr hagree b hd.2]
  | .not p, hd => by
      simp only [SymBool.disjointHavoc] at hd
      simp only [SymBool.eval, symBool_eval_congr hagree p hd]
  | .and p q, hd => by
      simp only [SymBool.disjointHavoc, Bool.and_eq_true] at hd
      simp only [SymBool.eval, symBool_eval_congr hagree p hd.1, symBool_eval_congr hagree q hd.2]
  | .or p q, hd => by
      simp only [SymBool.disjointHavoc, Bool.and_eq_true] at hd
      simp only [SymBool.eval, symBool_eval_congr hagree p hd.1, symBool_eval_congr hagree q hd.2]
  | .ite c tb fb, hd => by
      simp only [SymBool.disjointHavoc, Bool.and_eq_true] at hd
      simp only [SymBool.eval, symBool_eval_congr hagree c hd.1.1,
                 symBool_eval_congr hagree tb hd.1.2, symBool_eval_congr hagree fb hd.2]
end

/-- Symbolic value tracked per SSA variable during block-level symbolic execution. -/
inductive SymVal where
  /-- Symbolic integer value. -/
  | int (e : SymInt)
  /-- Symbolic boolean value. -/
  | bool (b : SymBool)
  /-- Symbolic pointer `(space, bufId, symOffset)`. -/
  | ptr (space : AddrSpace) (bufId : Nat) (offset : SymInt)
  /-- Guarded `Option` pointer `(guard, space, bufId, symOffset)` valid when `guard` holds. -/
  | condPtr (guard : SymBool) (space : AddrSpace) (bufId : Nat) (offset : SymInt)
  /-- `.shared::cta` byte offset into `SharedArray` `sharedId`. -/
  | smemOff (sharedId : Nat) (byteOff : SymInt)
  /-- 64-bit `tcgen05` shared-memory matrix descriptor into `SharedArray` `sharedId`. -/
  | smemDesc (sharedId : Nat) (byteOff : SymInt)
  /-- 32-bit Blackwell `TMEM` address `(lane << 16) + col` relative to an `allocCols` allocation. -/
  | tmemPtr (allocCols : Nat) (relAddr : SymInt)
  /-- Non-index float/aggregate value. -/
  | unk
  deriving Repr, BEq, DecidableEq, Lean.ToExpr

instance : LawfulBEq SymVal where
  rfl := by
    intro v; cases v <;> (try rfl) <;>
      (change instBEqSymVal.beq _ _ = true; simp [instBEqSymVal.beq])
  eq_of_beq := by
    intro a b h
    cases a <;> cases b <;>
      (try contradiction) <;>
      (change instBEqSymVal.beq _ _ = true at h; simp [instBEqSymVal.beq] at h; try simp_all)

/-- True when `SymVal` references no `.havoc` variable in `keys`. -/
def SymVal.disjointHavoc (keys : List Nat) : SymVal → Bool
  | .int e => e.disjointHavoc keys
  | .bool b => b.disjointHavoc keys
  | .ptr _ _ off => off.disjointHavoc keys
  | .condPtr g _ _ off => g.disjointHavoc keys && off.disjointHavoc keys
  | .smemOff _ b | .smemDesc _ b | .tmemPtr _ b => b.disjointHavoc keys
  | .unk => true

/-- Relational match between a symbolic `SymVal` and a concrete runtime `Val`
    (`.unk` leaves dead/float/untracked registers unconstrained; `.condPtr` requires equality under `guard`). -/
def SymVal.matchesVal (env : InputEnv) (t : ThreadCtx) (havoc : Nat → Nat)
    (sv : SymVal) (v : Val) : Prop :=
  match sv with
  | .unk => True
  | .int e => v = .int (e.eval env t havoc)
  | .bool b => v = .bool (b.eval env t havoc)
  | .ptr sp bid off => v = .ptr sp bid (off.eval env t havoc)
  | .condPtr g sp bid off => g.eval env t havoc = true → v = .ptr sp bid (off.eval env t havoc)
  | .smemOff sid b => v = .smemOff sid (b.eval env t havoc)
  | .smemDesc sid b => v = .smemDesc sid (b.eval env t havoc)
  | .tmemPtr cols b => v = .tmemPtr cols (b.eval env t havoc)

/-- Congruence of `SymVal.matchesVal` on havoc updates outside `keys`. -/
theorem symVal_matches_congr {keys : List Nat} {env : InputEnv} {t : ThreadCtx}
    {h₁ h₂ : Nat → Nat} (hagree : ∀ k, k ∉ keys → h₁ k = h₂ k)
    (sv : SymVal) (v : Val) (hd : sv.disjointHavoc keys = true)
    (hm : sv.matchesVal env t h₂ v) :
    sv.matchesVal env t h₁ v := by
  cases sv with
  | unk => trivial
  | int e =>
      simp only [SymVal.matchesVal] at hm ⊢
      rw [symInt_eval_congr hagree e hd]; exact hm
  | bool b =>
      simp only [SymVal.matchesVal] at hm ⊢
      rw [symBool_eval_congr hagree b hd]; exact hm
  | ptr sp bid off =>
      simp only [SymVal.matchesVal] at hm ⊢
      rw [symInt_eval_congr hagree off hd]; exact hm
  | condPtr g sp bid off =>
      simp only [SymVal.disjointHavoc, Bool.and_eq_true] at hd
      simp only [SymVal.matchesVal] at hm ⊢
      intro hg1
      rw [symBool_eval_congr hagree g hd.1] at hg1
      rw [symInt_eval_congr hagree off hd.2]
      exact hm hg1
  | smemOff sid b =>
      simp only [SymVal.matchesVal] at hm ⊢
      rw [symInt_eval_congr hagree b hd]; exact hm
  | smemDesc sid b =>
      simp only [SymVal.matchesVal] at hm ⊢
      rw [symInt_eval_congr hagree b hd]; exact hm
  | tmemPtr cols b =>
      simp only [SymVal.matchesVal] at hm ⊢
      rw [symInt_eval_congr hagree b hd]; exact hm

/-- Helper lemma: direct match check `(want && hyp == atom) || (!want && (hyp == .not atom || atom == .not hyp))`. -/
theorem directAtomMatch_sound {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat}
    {hyp atom : SymBool} {want : Bool}
    (hm : ((want && (hyp == atom)) || (!want && ((hyp == .not atom) || (atom == .not hyp)))) = true)
    (hhyp : hyp.eval env t havoc = true) :
    atom.eval env t havoc = want := by
  rcases Bool.or_eq_true_iff.mp hm with h1 | h2
  · rcases Bool.and_eq_true_iff.mp h1 with ⟨hw, heq⟩
    rw [← beq_iff_eq.mp heq, hw]; exact hhyp
  · rcases Bool.and_eq_true_iff.mp h2 with ⟨hnw, hor⟩
    have hw : want = false := by cases want <;> simp at hnw ⊢
    rcases Bool.or_eq_true_iff.mp hor with hl | hr
    · rw [beq_iff_eq.mp hl] at hhyp
      simp only [SymBool.eval] at hhyp
      rw [hw]; cases hv : atom.eval env t havoc <;> simp [hv] at hhyp ⊢
    · rw [beq_iff_eq.mp hr, hw]; simp [SymBool.eval, hhyp]

/-- Extract the underlying boolean condition `(c, b)` from a Rust `Option` discriminant check
    such as `(if c then 1 else 0) == 1` or `(if c then 0 else 1) != 0`.
    - **Why it exists**: `if let Some(elem) = c.get_mut(idx)` lowers in MIR to constructing an
      `Option` enum (`1` when `idx < len`, `0` otherwise) and branching on `disc == 1`.
    - **How it is used**: Lets `hypImpliesAtomFuel` and `SymBool.normDisc` recover `idx < len`
      directly from the discriminant comparison without leaving nested `0/1` `ite` terms. -/
def extractIteOneCond : SymBool → Option (SymBool × Bool)
  | .cmp .eq (.ite c (.const 1) (.const 0)) (.const 1) => some (c, true)
  | .cmp .eq (.ite c (.const 1) (.const 0)) (.const 0) => some (c, false)
  | .cmp .ne (.ite c (.const 1) (.const 0)) (.const 0) => some (c, true)
  | .cmp .ne (.ite c (.const 1) (.const 0)) (.const 1) => some (c, false)
  | .cmp .eq (.ite c (.const 0) (.const 1)) (.const 1) => some (c, false)
  | .cmp .eq (.ite c (.const 0) (.const 1)) (.const 0) => some (c, true)
  | .cmp .ne (.ite c (.const 0) (.const 1)) (.const 0) => some (c, false)
  | .cmp .ne (.ite c (.const 0) (.const 1)) (.const 1) => some (c, true)
  | .cmp .eq (.const 1) (.ite c (.const 1) (.const 0)) => some (c, true)
  | .cmp .eq (.const 0) (.ite c (.const 1) (.const 0)) => some (c, false)
  | .cmp .eq (.const 1) (.ite c (.const 0) (.const 1)) => some (c, false)
  | .cmp .eq (.const 0) (.ite c (.const 0) (.const 1)) => some (c, true)
  | _ => none

/-- Soundness of `extractIteOneCond`: whenever `extractIteOneCond hyp = some (c, b)` and `hyp`
    evaluates to `true`, the inner predicate `c` evaluates to `b`. -/
theorem extractIteOneCond_sound {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat} :
    ∀ {hyp c : SymBool} {b : Bool},
      extractIteOneCond hyp = some (c, b) →
      hyp.eval env t havoc = true →
      c.eval env t havoc = b
  | .cmp .eq (.ite c0 (.const 1) (.const 0)) (.const 1), _, _, hext, hhyp
  | .cmp .eq (.ite c0 (.const 1) (.const 0)) (.const 0), _, _, hext, hhyp
  | .cmp .ne (.ite c0 (.const 1) (.const 0)) (.const 0), _, _, hext, hhyp
  | .cmp .ne (.ite c0 (.const 1) (.const 0)) (.const 1), _, _, hext, hhyp
  | .cmp .eq (.ite c0 (.const 0) (.const 1)) (.const 1), _, _, hext, hhyp
  | .cmp .eq (.ite c0 (.const 0) (.const 1)) (.const 0), _, _, hext, hhyp
  | .cmp .ne (.ite c0 (.const 0) (.const 1)) (.const 0), _, _, hext, hhyp
  | .cmp .ne (.ite c0 (.const 0) (.const 1)) (.const 1), _, _, hext, hhyp
  | .cmp .eq (.const 1) (.ite c0 (.const 1) (.const 0)), _, _, hext, hhyp
  | .cmp .eq (.const 0) (.ite c0 (.const 1) (.const 0)), _, _, hext, hhyp
  | .cmp .eq (.const 1) (.ite c0 (.const 0) (.const 1)), _, _, hext, hhyp
  | .cmp .eq (.const 0) (.ite c0 (.const 0) (.const 1)), _, _, hext, hhyp => by
      simp only [extractIteOneCond, Option.some.injEq, Prod.mk.injEq] at hext
      rcases hext with ⟨rfl, rfl⟩
      simp only [SymBool.eval, SymInt.eval, evalCmpOp] at hhyp
      cases hcv : SymBool.eval env t havoc c0 <;> simp [hcv] at hhyp ⊢

/-- Check whether `hyp` syntactically entails `atom == want` (including through `Option` discriminant `.ite`s
    and De Morgan `!(p || q)` / `!!p` negations) by bounded structural fuel (`fuel = 16`). -/
def hypImpliesAtomFuel : Nat → SymBool → SymBool → Bool → Bool
  | 0, _, _, _ => false
  | fuel + 1, hyp, atom, want =>
      if (want && (hyp == atom)) || (!want && ((hyp == .not atom) || (atom == .not hyp))) then true
      else
        match hyp with
        | .lit false => true
        | .and p q => hypImpliesAtomFuel fuel p atom want || hypImpliesAtomFuel fuel q atom want
        | .or p q => hypImpliesAtomFuel fuel p atom want && hypImpliesAtomFuel fuel q atom want
        | .not (.or p q) => hypImpliesAtomFuel fuel (.not p) atom want || hypImpliesAtomFuel fuel (.not q) atom want
        | .not (.not p) => hypImpliesAtomFuel fuel p atom want
        | .cmp op l r =>
            match extractIteOneCond (.cmp op l r) with
            | some (c, true) => hypImpliesAtomFuel fuel c atom want
            | some (c, false) => hypImpliesAtomFuel fuel (.not c) atom want
            | none => false
        | _ => false

/-- Check whether `hyp` syntactically entails `atom == want`. -/
def hypImpliesAtom (hyp atom : SymBool) (want : Bool) : Bool :=
  hypImpliesAtomFuel 24 hyp atom want

/-- Soundness of `hypImpliesAtomFuel` by structural induction on `fuel`. -/
theorem hypImpliesAtomFuel_sound {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat}
    {atom : SymBool} {want : Bool} :
    ∀ fuel : Nat, ∀ hyp : SymBool,
      hypImpliesAtomFuel fuel hyp atom want = true →
      hyp.eval env t havoc = true →
      atom.eval env t havoc = want
  | 0, _, himp, _ => by contradiction
  | fuel + 1, hyp, himp, hhyp => by
      unfold hypImpliesAtomFuel at himp
      split at himp
      · rename_i hm; exact directAtomMatch_sound hm hhyp
      · cases hyp with
        | lit b => cases b <;> simp [SymBool.eval] at hhyp himp
        | mbarReady _ _ | ite _ _ _ => simp at himp
        | and p q =>
            dsimp only at himp
            simp only [SymBool.eval, Bool.and_eq_true] at hhyp
            rcases Bool.or_eq_true_iff.mp himp with hp | hq
            · exact hypImpliesAtomFuel_sound fuel p hp hhyp.1
            · exact hypImpliesAtomFuel_sound fuel q hq hhyp.2
        | or p q =>
            dsimp only at himp
            rcases Bool.and_eq_true_iff.mp himp with ⟨hp, hq⟩
            simp only [SymBool.eval, Bool.or_eq_true] at hhyp
            rcases hhyp with hpv | hqv
            · exact hypImpliesAtomFuel_sound fuel p hp hpv
            · exact hypImpliesAtomFuel_sound fuel q hq hqv
        | not inner =>
            cases inner with
            | or p q =>
                dsimp only at himp
                have hnp : (SymBool.not p).eval env t havoc = true ∧ (SymBool.not q).eval env t havoc = true := by
                  simp only [SymBool.eval] at hhyp ⊢
                  cases hp : p.eval env t havoc <;> cases hq : q.eval env t havoc <;> simp_all
                rcases Bool.or_eq_true_iff.mp himp with h1 | h2
                · exact hypImpliesAtomFuel_sound fuel (.not p) h1 hnp.1
                · exact hypImpliesAtomFuel_sound fuel (.not q) h2 hnp.2
            | not p =>
                dsimp only at himp
                have hp : p.eval env t havoc = true := by
                  simp only [SymBool.eval] at hhyp; cases hpv : p.eval env t havoc <;> simp_all
                exact hypImpliesAtomFuel_sound fuel p himp hp
            | _ => simp at himp
        | cmp op l r =>
            dsimp only at himp
            cases hext : extractIteOneCond (.cmp op l r) with
            | none => rw [hext] at himp; contradiction
            | some cb =>
                rw [hext] at himp
                rcases cb with ⟨c, b⟩
                cases b
                · dsimp only at himp
                  have hnc : (SymBool.not c).eval env t havoc = true := by
                    simp [SymBool.eval, extractIteOneCond_sound hext hhyp]
                  exact hypImpliesAtomFuel_sound fuel (.not c) himp hnc
                · dsimp only at himp
                  exact hypImpliesAtomFuel_sound fuel c himp (extractIteOneCond_sound hext hhyp)

/-- Soundness of `hypImpliesAtom`. -/
theorem hypImpliesAtom_sound {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat}
    {atom : SymBool} {want : Bool} (hyp : SymBool)
    (himp : hypImpliesAtom hyp atom want = true)
    (hhyp : hyp.eval env t havoc = true) :
    atom.eval env t havoc = want :=
  hypImpliesAtomFuel_sound 24 hyp himp hhyp

/-- Syntactic boolean implication `(target == want)` under single-conjunct `hyp` (no top-level `.or`). -/
def symImpliesTarget (hyp target : SymBool) (want : Bool) : Bool :=
  if hypImpliesAtom hyp target want then true
  else
    match target with
    | .lit b => b == want
    | .not inner => symImpliesTarget hyp inner (!want)
    | .and a b =>
        if want then symImpliesTarget hyp a true && symImpliesTarget hyp b true
        else symImpliesTarget hyp a false || symImpliesTarget hyp b false
    | .or a b =>
        if want then symImpliesTarget hyp a true || symImpliesTarget hyp b true
        else symImpliesTarget hyp a false && symImpliesTarget hyp b false
    | _ => false

/-- Soundness of `symImpliesTarget`. -/
theorem symImpliesTarget_sound {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat}
    {hyp : SymBool} (hhyp : hyp.eval env t havoc = true) :
    ∀ target : SymBool, ∀ want : Bool,
      symImpliesTarget hyp target want = true →
      target.eval env t havoc = want
  | .lit b, want, himp => by
      unfold symImpliesTarget at himp
      split at himp
      · rename_i ha; exact hypImpliesAtom_sound hyp ha hhyp
      · exact beq_iff_eq.mp himp
  | .mbarReady _ _, want, himp => by
      unfold symImpliesTarget at himp
      split at himp
      · rename_i ha; exact hypImpliesAtom_sound hyp ha hhyp
      · contradiction
  | .cmp _ _ _, want, himp => by
      unfold symImpliesTarget at himp
      split at himp
      · rename_i ha; exact hypImpliesAtom_sound hyp ha hhyp
      · contradiction
  | .ite _ _ _, want, himp => by
      unfold symImpliesTarget at himp
      split at himp
      · rename_i ha; exact hypImpliesAtom_sound hyp ha hhyp
      · contradiction
  | .not inner, want, himp => by
      unfold symImpliesTarget at himp
      split at himp
      · rename_i ha; exact hypImpliesAtom_sound hyp ha hhyp
      · have hi := symImpliesTarget_sound hhyp inner (!want) himp
        simp [SymBool.eval, hi]
  | .and a b, want, himp => by
      unfold symImpliesTarget at himp
      split at himp
      · rename_i ha; exact hypImpliesAtom_sound hyp ha hhyp
      · cases want
        · simp only [Bool.false_eq_true, ↓reduceIte, Bool.or_eq_true] at himp
          rcases himp with hla | hlb
          · simp [SymBool.eval, symImpliesTarget_sound hhyp a false hla]
          · simp [SymBool.eval, symImpliesTarget_sound hhyp b false hlb]
        · simp only [↓reduceIte, Bool.and_eq_true] at himp
          simp [SymBool.eval, symImpliesTarget_sound hhyp a true himp.1,
                symImpliesTarget_sound hhyp b true himp.2]
  | .or a b, want, himp => by
      unfold symImpliesTarget at himp
      split at himp
      · rename_i ha; exact hypImpliesAtom_sound hyp ha hhyp
      · cases want
        · simp only [Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true] at himp
          simp [SymBool.eval, symImpliesTarget_sound hhyp a false himp.1,
                symImpliesTarget_sound hhyp b false himp.2]
        · simp only [↓reduceIte, Bool.or_eq_true] at himp
          rcases himp with hla | hlb
          · simp [SymBool.eval, symImpliesTarget_sound hhyp a true hla]
          · simp [SymBool.eval, symImpliesTarget_sound hhyp b true hlb]

/-- Fuel-bounded syntactic boolean implication `hyp => (target == want)` distributing `.or`
    at top level and right-nested under `.and` (`(p ∧ (q1 ∨ q2)) => ...`) before decomposing `target`. -/
def symImpliesValFuel : Nat → SymBool → SymBool → Bool → Bool
  | 0, hyp, target, want => symImpliesTarget hyp target want
  | fuel + 1, hyp, target, want =>
      if symImpliesTarget hyp target want then true
      else
        match hyp with
        | .or p q => symImpliesValFuel fuel p target want && symImpliesValFuel fuel q target want
        | .and p (.or q1 q2) =>
            symImpliesValFuel fuel (.and p q1) target want && symImpliesValFuel fuel (.and p q2) target want
        | _ => false

/-- Syntactic boolean implication `hyp => (target == want)` with disjunction-splitting fuel `2`. -/
def symImpliesVal (hyp target : SymBool) (want : Bool) : Bool :=
  symImpliesValFuel 2 hyp target want

/-- Soundness of `symImpliesValFuel`. -/
theorem symImpliesValFuel_sound {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat} :
    ∀ fuel : Nat, ∀ {hyp : SymBool}, hyp.eval env t havoc = true →
      ∀ target : SymBool, ∀ want : Bool,
        symImpliesValFuel fuel hyp target want = true →
        target.eval env t havoc = want
  | 0, _, hhyp, target, want, himp => symImpliesTarget_sound hhyp target want himp
  | fuel + 1, hyp, hhyp, target, want, himp => by
      unfold symImpliesValFuel at himp
      split at himp
      · rename_i ht; exact symImpliesTarget_sound hhyp target want ht
      · cases hyp with
        | or p q =>
            dsimp only at himp
            rcases Bool.and_eq_true_iff.mp himp with ⟨hp, hq⟩
            simp only [SymBool.eval, Bool.or_eq_true] at hhyp
            rcases hhyp with hpv | hqv
            · exact symImpliesValFuel_sound fuel hpv target want hp
            · exact symImpliesValFuel_sound fuel hqv target want hq
        | and p q =>
            cases q with
            | or q1 q2 =>
                dsimp only at himp
                rcases Bool.and_eq_true_iff.mp himp with ⟨h1, h2⟩
                simp only [SymBool.eval, Bool.and_eq_true, Bool.or_eq_true] at hhyp
                rcases hhyp.2 with hq1 | hq2
                · exact symImpliesValFuel_sound fuel (by simp [SymBool.eval, hhyp.1, hq1]) target want h1
                · exact symImpliesValFuel_sound fuel (by simp [SymBool.eval, hhyp.1, hq2]) target want h2
            | _ => contradiction
        | lit _ | mbarReady _ _ | cmp _ _ _ | not _ | ite _ _ _ => contradiction

/-- Soundness of `symImpliesVal`. -/
theorem symImpliesVal_sound {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat}
    {hyp : SymBool} (hhyp : hyp.eval env t havoc = true)
    (target : SymBool) (want : Bool)
    (himp : symImpliesVal hyp target want = true) :
    target.eval env t havoc = want :=
  symImpliesValFuel_sound 2 hhyp target want himp

/-- Equivalence of `SymInt` under path condition `hyp` (unfolding `.ite` trees). -/
def symIntEquivUnder (hyp : SymBool) : SymInt → SymInt → Bool
  | e1, e2 =>
      if e1 == e2 then true
      else
        match e1 with
        | .ite c tv fv =>
            (symImpliesVal hyp c true && symIntEquivUnder hyp tv e2) ||
            (symImpliesVal hyp c false && symIntEquivUnder hyp fv e2)
        | _ => false

/-- Soundness of `symIntEquivUnder`. -/
theorem symIntEquivUnder_sound {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat}
    {hyp : SymBool} (hhyp : hyp.eval env t havoc = true) :
    ∀ expected actual : SymInt,
      symIntEquivUnder hyp expected actual = true →
      expected.eval env t havoc = actual.eval env t havoc
  | .ite c tv fv, act, heq => by
      unfold symIntEquivUnder at heq
      split at heq
      · rename_i h1; rw [beq_iff_eq.mp h1]
      · simp only [Bool.or_eq_true, Bool.and_eq_true] at heq
        rcases heq with ⟨hct, htv⟩ | ⟨hcf, hfv⟩
        · have hc := symImpliesVal_sound hhyp c true hct
          simp [SymInt.eval, hc, symIntEquivUnder_sound hhyp tv act htv]
        · have hc := symImpliesVal_sound hhyp c false hcf
          simp [SymInt.eval, hc, symIntEquivUnder_sound hhyp fv act hfv]
  | .const _, _, heq | .tidX, _, heq | .tidY, _, heq | .ctaidX, _, heq | .ctaidY, _, heq | .nctaid _, _, heq
  | .index1D _, _, heq | .paramScalar _, _, heq | .sliceLen _, _, heq | .havoc _ _, _, heq
  | .opaqueUint _ _, _, heq | .binOp _ _ _ _, _, heq | .intCast _ _, _, heq => by
      unfold symIntEquivUnder at heq
      split at heq
      · rename_i h1; rw [beq_iff_eq.mp h1]
      · contradiction

/-- Equivalence of `SymBool` under path condition `hyp` (unfolding `.ite` trees). -/
def symBoolEquivUnder (hyp : SymBool) : SymBool → SymBool → Bool
  | b1, b2 =>
      if b1 == b2 then true
      else
        match b1 with
        | .ite c tb fb =>
            (symImpliesVal hyp c true && symBoolEquivUnder hyp tb b2) ||
            (symImpliesVal hyp c false && symBoolEquivUnder hyp fb b2)
        | _ => false

/-- Soundness of `symBoolEquivUnder`. -/
theorem symBoolEquivUnder_sound {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat}
    {hyp : SymBool} (hhyp : hyp.eval env t havoc = true) :
    ∀ expected actual : SymBool,
      symBoolEquivUnder hyp expected actual = true →
      expected.eval env t havoc = actual.eval env t havoc
  | .ite c tb fb, act, heq => by
      unfold symBoolEquivUnder at heq
      split at heq
      · rename_i h1; rw [beq_iff_eq.mp h1]
      · simp only [Bool.or_eq_true, Bool.and_eq_true] at heq
        rcases heq with ⟨hct, htb⟩ | ⟨hcf, hfb⟩
        · have hc := symImpliesVal_sound hhyp c true hct
          simp [SymBool.eval, hc, symBoolEquivUnder_sound hhyp tb act htb]
        · have hc := symImpliesVal_sound hhyp c false hcf
          simp [SymBool.eval, hc, symBoolEquivUnder_sound hhyp fb act hfb]
  | .lit _, _, heq | .mbarReady _ _, _, heq | .cmp _ _ _, _, heq | .not _, _, heq
  | .and _ _, _, heq | .or _ _, _, heq => by
      unfold symBoolEquivUnder at heq
      split at heq
      · rename_i h1; rw [beq_iff_eq.mp h1]
      · contradiction

/-- Syntactic subsumption of `SymVal`s under path condition `hyp`.
    - **What it does**: Checks whether an incoming edge value `actual` refines the destination
      block's summary value `expected` whenever `hyp` holds.
    - **Why we need it**: At `if/else` joins (`.ite` / `.condPtr`) and loop backedges, `validSymEdge`
      must confirm in `O(1)` that every block's `SymRegList` over-approximates all incoming edges.
    - **How it is used**: Called per variable inside `validSymEdge`; justified by `symValEquivUnder_sound`. -/
def symValEquivUnder (hyp : SymBool) (expected actual : SymVal) : Bool :=
  match expected, actual with
  | .unk, _ => true
  | .int e1, .int e2 => symIntEquivUnder hyp e1 e2
  | .bool b1, .bool b2 => symBoolEquivUnder hyp b1 b2
  | .ptr sp1 b1 o1, .ptr sp2 b2 o2 => sp1 == sp2 && b1 == b2 && symIntEquivUnder hyp o1 o2
  | .smemOff s1 b1, .smemOff s2 b2 => s1 == s2 && symIntEquivUnder hyp b1 b2
  | .smemDesc s1 b1, .smemDesc s2 b2 => s1 == s2 && symIntEquivUnder hyp b1 b2
  | .tmemPtr c1 b1, .tmemPtr c2 b2 => c1 == c2 && symIntEquivUnder hyp b1 b2
  | .condPtr _ sp1 b1 o1, .ptr sp2 b2 o2 =>
      sp1 == sp2 && b1 == b2 && symIntEquivUnder hyp o1 o2
  | .condPtr g _ _ _, .unk =>
      symImpliesVal hyp g false
  | .condPtr g1 sp1 b1 o1, .condPtr g2 sp2 b2 o2 =>
      sp1 == sp2 && b1 == b2 &&
      ((g1 == g2 && o1 == o2) ||
       (symImpliesVal (.and hyp g1) g2 true && symIntEquivUnder (.and hyp g1) o1 o2))
  | _, _ => false

/-- Soundness of `symValEquivUnder`: if `actual` matches concrete `v` and `expected` subsumes `actual`,
    then `expected` also matches `v`. -/
theorem symValEquivUnder_sound {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat}
    {hyp : SymBool} (hhyp : hyp.eval env t havoc = true)
    (expected actual : SymVal) (v : Val)
    (heq : symValEquivUnder hyp expected actual = true)
    (hact : actual.matchesVal env t havoc v) :
    expected.matchesVal env t havoc v := by
  unfold symValEquivUnder at heq
  cases expected with
  | unk => trivial                                        -- `.unk` over-approximates any concrete runtime `Val`
  | int e1 =>                                             -- `.int e1` subsumes `.int e2` via `symIntEquivUnder`
      cases actual with
      | int e2 =>
          simp only [SymVal.matchesVal] at heq hact ⊢
          rw [symIntEquivUnder_sound hhyp e1 e2 heq]; exact hact
      | _ => contradiction
  | bool b1 =>                                            -- `.bool b1` subsumes `.bool b2` via `symBoolEquivUnder`
      cases actual with
      | bool b2 =>
          simp only [SymVal.matchesVal] at heq hact ⊢
          rw [symBoolEquivUnder_sound hhyp b1 b2 heq]; exact hact
      | _ => contradiction
  | ptr sp1 b1 o1 =>                                      -- `.ptr` requires identical space/buffer + equal offset under `hyp`
      cases actual with
      | ptr sp2 b2 o2 =>
          simp only [Bool.and_eq_true, beq_iff_eq, SymVal.matchesVal] at heq hact ⊢
          rcases heq with ⟨⟨rfl, rfl⟩, ho⟩
          rw [symIntEquivUnder_sound hhyp o1 o2 ho]; exact hact
      | _ => contradiction
  | smemOff s1 b1 =>                                      -- `.smemOff` requires identical `SharedArray` ID + equal byte offset
      cases actual with
      | smemOff s2 b2 =>
          simp only [Bool.and_eq_true, beq_iff_eq, SymVal.matchesVal] at heq hact ⊢
          rcases heq with ⟨rfl, hb⟩
          rw [symIntEquivUnder_sound hhyp b1 b2 hb]; exact hact
      | _ => contradiction
  | smemDesc s1 b1 =>                                     -- `.smemDesc` requires identical `SharedArray` ID + equal descriptor base
      cases actual with
      | smemDesc s2 b2 =>
          simp only [Bool.and_eq_true, beq_iff_eq, SymVal.matchesVal] at heq hact ⊢
          rcases heq with ⟨rfl, hb⟩
          rw [symIntEquivUnder_sound hhyp b1 b2 hb]; exact hact
      | _ => contradiction
  | tmemPtr c1 b1 =>                                      -- `.tmemPtr` requires identical `allocCols` + equal `(lane << 16) + col`
      cases actual with
      | tmemPtr c2 b2 =>
          simp only [Bool.and_eq_true, beq_iff_eq, SymVal.matchesVal] at heq hact ⊢
          rcases heq with ⟨rfl, hb⟩
          rw [symIntEquivUnder_sound hhyp b1 b2 hb]; exact hact
      | _ => contradiction
  | condPtr g1 sp1 b1 o1 =>                               -- `Option` `.condPtr` join (`Some(ptr)` vs. `None` `.unk`)
      cases actual with
      | ptr sp2 b2 o2 =>                                  --   `Some` branch: valid pointer matches unconditionally
          simp only [Bool.and_eq_true, beq_iff_eq, SymVal.matchesVal] at heq hact ⊢
          rcases heq with ⟨⟨rfl, rfl⟩, ho⟩
          intro _; rw [symIntEquivUnder_sound hhyp o1 o2 ho]; exact hact
      | unk =>                                            --   `None` branch: `hyp` forces guard `g1` to `false` (vacuously valid)
          simp only [SymVal.matchesVal] at heq ⊢
          intro hg1
          rw [symImpliesVal_sound hhyp g1 false heq] at hg1; contradiction
      | condPtr g2 sp2 b2 o2 =>                           --   Nested `condPtr` join: `hyp ∧ g1` implies `g2` and equal offsets
          simp only [Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq, SymVal.matchesVal] at heq hact ⊢
          rcases heq with ⟨⟨rfl, rfl⟩, ⟨rfl, rfl⟩ | ⟨hg2t, ho⟩⟩
          · exact hact
          · intro hg1
            have hhg1 : (SymBool.and hyp g1).eval env t havoc = true := by
              simp [SymBool.eval, hhyp, hg1]
            rw [symIntEquivUnder_sound hhg1 o1 o2 ho]
            exact hact (symImpliesVal_sound hhg1 g2 true hg2t)
      | int _ | bool _ | smemOff _ _ | smemDesc _ _ | tmemPtr _ _ => contradiction

/-! ## Section 2: Abstract Barrier Phases (`PhaseSpec`) -/

/-- Abstract barrier-phase descriptor tracking `sync_threads()` counts across acyclic and cyclic blocks.
    - **What it does**: Summarizes the set of runtime `phase` integers a thread can have at a block:
      exact count `.exact k` outside loops, or modular residue `.modGe minP m r` (`phase % m = r`)
      inside a loop containing `m` barriers per iteration.
    - **Why we need it**: Proves cross-iteration shared-memory race freedom (e.g., Tile `t + 1` write
      at even phase `2t` vs. Tile `t` read at odd phase `2t + 1`) without unrolling the loop.
    - **How it is used**: Attached to each `SymBlockInv` and `SymMemEvent`; `vcNoSharedRace` prunes
      pairs whose `PhaseSpec.disjoint` returns `true` via `phaseSpec_disjoint_sound`. -/
inductive PhaseSpec where
  /-- Unreachable block (not yet visited). -/
  | bot
  /-- Exact barrier count `phase = k` (acyclic blocks). -/
  | exact (k : Nat)
  /-- Cyclic modular invariant `minP ≤ phase ∧ phase % m = r` (`m > 0`). -/
  | modGe (minP m r : Nat)
  /-- Unconstrained barrier phase (`True`). -/
  | any
  deriving Repr, BEq, DecidableEq, Lean.ToExpr

instance : LawfulBEq PhaseSpec where
  rfl := by
    intro a; cases a <;> (try rfl) <;>
      (change instBEqPhaseSpec.beq _ _ = true; simp [instBEqPhaseSpec.beq])
  eq_of_beq := by
    intro a b h; cases a <;> cases b <;> (try contradiction) <;>
      (change instBEqPhaseSpec.beq _ _ = true at h; simp [instBEqPhaseSpec.beq] at h; try simp_all)

/-- True when concrete barrier count `phase` satisfies `ps`. -/
def PhaseSpec.matchesPhase (ps : PhaseSpec) (phase : Nat) : Bool :=
  match ps with
  | .bot => false
  | .exact k => phase == k
  | .modGe minP m r => decide (minP ≤ phase) && decide (0 < m) && (phase % m == r)
  | .any => true

/-- Advance `PhaseSpec` across one `Stmt.syncThreads` barrier (`phase -> phase + 1`). -/
def PhaseSpec.stepSync : PhaseSpec → PhaseSpec
  | .bot => .bot
  | .exact k => .exact (k + 1)
  | .modGe minP m r => .modGe (minP + 1) m ((r + 1) % m)
  | .any => .any

/-- Soundness of `PhaseSpec.stepSync`. -/
theorem phaseSpec_stepSync_sound {ps : PhaseSpec} {phase : Nat}
    (hm : ps.matchesPhase phase = true) :
    ps.stepSync.matchesPhase (phase + 1) = true := by
  cases ps with
  | bot => contradiction
  | any => rfl
  | exact k =>
      simp only [PhaseSpec.matchesPhase, PhaseSpec.stepSync, beq_iff_eq] at hm ⊢
      omega
  | modGe minP m r =>
      simp only [PhaseSpec.matchesPhase, PhaseSpec.stepSync, Bool.and_eq_true,
                 decide_eq_true_eq, beq_iff_eq] at hm ⊢
      refine ⟨⟨by omega, hm.1.2⟩, ?_⟩
      rw [← hm.2, Nat.add_mod (phase % m), Nat.mod_mod, ← Nat.add_mod]

/-- Check syntactic subsumption `srcPs ⊆ dstPs` on a CFG edge. -/
def PhaseSpec.subsumes (dstPs srcPs : PhaseSpec) : Bool :=
  match dstPs, srcPs with
  | _, .bot => true
  | .any, _ => true
  | .exact k1, .exact k2 => k1 == k2
  | .modGe min1 m1 r1, .exact k2 => decide (min1 ≤ k2) && decide (0 < m1) && (k2 % m1 == r1)
  | .modGe min1 m1 r1, .modGe min2 m2 r2 =>
      decide (min1 ≤ min2) && decide (0 < m1) && (m1 == m2) && (r1 == r2)
  | _, _ => false

/-- Soundness of `PhaseSpec.subsumes`. -/
theorem phaseSpec_subsumes_sound {dstPs srcPs : PhaseSpec} {phase : Nat}
    (hsub : dstPs.subsumes srcPs = true)
    (hm : srcPs.matchesPhase phase = true) :
    dstPs.matchesPhase phase = true := by
  cases dstPs <;> cases srcPs <;>
    simp only [PhaseSpec.subsumes, PhaseSpec.matchesPhase, Bool.and_eq_true,
               decide_eq_true_eq, beq_iff_eq] at hsub hm ⊢
  all_goals try contradiction
  all_goals try trivial
  · omega
  · refine ⟨⟨by omega, hsub.1.2⟩, ?_⟩; rw [hm]; exact hsub.2
  · refine ⟨⟨by omega, hsub.1.1.2⟩, ?_⟩; rw [hsub.1.2, hm.2]; exact hsub.2.symm

/-- True when `ps1` and `ps2` cannot possibly hold for the same `phase` (`phase₁ ≠ phase₂`). -/
def PhaseSpec.disjoint (ps1 ps2 : PhaseSpec) : Bool :=
  match ps1, ps2 with
  | .bot, _ | _, .bot => true
  | .exact k1, .exact k2 => k1 != k2
  | .exact k1, .modGe min2 m2 r2 => decide (k1 < min2) || (decide (0 < m2) && (k1 % m2 != r2))
  | .modGe min1 m1 r1, .exact k2 => decide (k2 < min1) || (decide (0 < m1) && (k2 % m1 != r1))
  | .modGe _ m1 r1, .modGe _ m2 r2 => decide (0 < m1) && (m1 == m2) && (r1 != r2)
  | _, _ => false

/-- Soundness of `PhaseSpec.disjoint`: no single `phase` can match both `ps1` and `ps2`. -/
theorem phaseSpec_disjoint_sound {ps1 ps2 : PhaseSpec} {p1 p2 : Nat}
    (hdisj : ps1.disjoint ps2 = true)
    (hm1 : ps1.matchesPhase p1 = true)
    (hm2 : ps2.matchesPhase p2 = true)
    (heq : p1 = p2) : False := by
  subst heq                                               -- Unify `p2` with `p1` so one `phase` must satisfy both specs
  cases ps1 <;> cases ps2 <;>                             -- Case-split on `.exact` vs `.modGe` combinations
    simp only [PhaseSpec.disjoint, PhaseSpec.matchesPhase, Bool.and_eq_true, Bool.or_eq_true,
               decide_eq_true_eq, beq_iff_eq, bne_iff_ne] at hdisj hm1 hm2
  all_goals try contradiction
  · exact hdisj (hm1.symm.trans hm2)                      -- `.exact k1` vs `.exact k2` with `k1 != k2`
  · rcases hdisj with hlt | ⟨_, hne⟩                      -- `.exact k1` vs `.modGe min2 m2 r2`: either `< min2` or `% m2 != r2`
    · subst hm1; omega
    · subst hm1; exact hne hm2.2
  · rcases hdisj with hlt | ⟨_, hne⟩                      -- Symmetric `.modGe` vs `.exact` case
    · subst hm2; omega
    · subst hm2; exact hne hm1.2
  · rw [← hdisj.1.2] at hm2                               -- `.modGe` vs `.modGe` with equal modulus `m` and distinct residues `r1 != r2`
    exact hdisj.2 (hm1.2.symm.trans hm2.2)

/-- Pairwise `(phaseSpec, warpPhaseSpec)` match against `(phase, warpPhase)`. -/
def matchesPhasePair (pss : PhaseSpec × PhaseSpec) (phases : Nat × Nat) : Bool :=
  pss.1.matchesPhase phases.1 && pss.2.matchesPhase phases.2

/-- Soundness of stepping CTA `phase + 1` in `matchesPhasePair`. -/
theorem matchesPhasePair_stepCta {pss : PhaseSpec × PhaseSpec} {phases : Nat × Nat}
    (hm : matchesPhasePair pss phases = true) :
    matchesPhasePair (pss.1.stepSync, pss.2) (phases.1 + 1, phases.2) = true := by
  unfold matchesPhasePair at hm ⊢
  rcases Bool.and_eq_true_iff.mp hm with ⟨hm1, hm2⟩
  exact Bool.and_eq_true_iff.mpr ⟨phaseSpec_stepSync_sound hm1, hm2⟩

/-- Soundness of stepping intra-warp `warpPhase + 1` in `matchesPhasePair`. -/
theorem matchesPhasePair_stepWarp {pss : PhaseSpec × PhaseSpec} {phases : Nat × Nat}
    (hm : matchesPhasePair pss phases = true) :
    matchesPhasePair (pss.1, pss.2.stepSync) (phases.1, phases.2 + 1) = true := by
  unfold matchesPhasePair at hm ⊢
  rcases Bool.and_eq_true_iff.mp hm with ⟨hm1, hm2⟩
  exact Bool.and_eq_true_iff.mpr ⟨hm1, phaseSpec_stepSync_sound hm2⟩

/-- Soundness of pairwise subsumption on `(PhaseSpec × PhaseSpec)`. -/
theorem matchesPhasePair_subsumes {dst src : PhaseSpec × PhaseSpec} {phases : Nat × Nat}
    (h1 : dst.1.subsumes src.1 = true)
    (h2 : dst.2.subsumes src.2 = true)
    (hm : matchesPhasePair src phases = true) :
    matchesPhasePair dst phases = true := by
  unfold matchesPhasePair at hm ⊢
  rcases Bool.and_eq_true_iff.mp hm with ⟨hm1, hm2⟩
  exact Bool.and_eq_true_iff.mpr ⟨phaseSpec_subsumes_sound h1 hm1, phaseSpec_subsumes_sound h2 hm2⟩


end CudaOxide.Proofs.Program
