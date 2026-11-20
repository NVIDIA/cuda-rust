/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import Lean
import CudaOxide.Contracts.Program
import CudaOxide.Proofs.Launch

/-!
# Verified Symbolic VC Generator & `mir_vc` Tactic (`CudaOxide.Proofs.Program`)

Proves declarative safety contracts (`MirNoOobGlobal`, `MirNoOobShared`,
`MirDisjointWrites`, `MirNoSharedRace`, `MirAssertRedundant`) directly over a
line-by-line `MirFunc` with zero per-kernel manual proofs (`0 sorryAx`).

## File Map (5 Sections)

```text
  def mirFunc : MirFunc   (1-to-1 basic blocks from `PostMem2Reg` `dialect-mir`)
         │
         ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ SECTION 1: Symbolic Domain & Subsumption (`SymInt`, `SymBool`, `SymVal`)    │
  │   • Symbolic integer/boolean ASTs + `.havoc` loop variables + `.condPtr`    │
  │   • Syntactic implication (`symImpliesVal`) & join equivalence (`*_sound`)  │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ SECTION 2: Abstract Barrier Phases (`PhaseSpec`)                            │
  │   • Exact (`phase = k`) & cyclic (`phase % m = r`) `sync_threads()` counts  │
  │   • Sound barrier-phase disjointness (`phaseSpec_disjoint_sound`)           │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ SECTION 3: Intra-Block Symbolic Execution (`symEvalRhs`, `symExecStmts`)    │
  │   • Steps a block's statements symbolically, emitting `SymMemEvent`s and    │
  │     `SymAssertEvent`s (`symExecStmts_correct`)                              │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ SECTION 4: Inductive CFG Invariant Inference & Checker (`validSymFunc`)     │
  │   • `inferSymTable`: 1-pass RPO inference cutting loops via `.havoc`        │
  │   • `validSymFunc` -> `symReachable_sound`: proves every concrete GPU path  │
  │     satisfies `symTable` (checked per kernel in `O(blocks)` via `rfl`)      │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ SECTION 5: Verification Conditions, `MetaM` Caching & `mir_vc`              │
  │   • `mir_def_sym_table`: precomputes `symTable` & events once at elab time  │
  │   • `vc*_of_table`: reduces operational contracts to pure `Nat` inequalities│
  │   • `mir_vc`: unfolds formulas, injects `mir_mul_mono`, and runs `omega`    │
  └─────────────────────────────────────────────────────────────────────────────┘
```
-/
namespace CudaOxide.Proofs.Program
open CudaOxide.Mir.Program
open CudaOxide.Contracts.Program

/-! ## Section 1: Symbolic Expressions & Conditional Subsumption -/

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
  | .index1D a, .index1D b | .paramScalar a, .paramScalar b | .sliceLen a, .sliceLen b => a == b
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
  | .index1D _ | .paramScalar _ | .sliceLen _ => by simp [SymInt.beq]
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
theorem symInt_eq_of_beq : ∀ {a b : SymInt}, SymInt.beq a b = true → a = b
  | .const _, .const _, h => by simp [SymInt.beq] at h; exact congrArg _ h
  | .tidX, .tidX, _ | .tidY, .tidY, _ | .ctaidX, .ctaidX, _ | .ctaidY, .ctaidY, _ => rfl
  | .index1D _, .index1D _, h | .paramScalar _, .paramScalar _, h | .sliceLen _, .sliceLen _, h => by
      simp [SymInt.beq] at h; exact congrArg _ h
  | .havoc _ _, .havoc _ _, h | .opaqueUint _ _, .opaqueUint _ _, h => by
      simp [SymInt.beq] at h; rw [h.1, h.2]
  | .binOp _ _ _ _, .binOp _ _ _ _, h => by
      simp [SymInt.beq] at h; rw [h.1.1.1, h.1.1.2, symInt_eq_of_beq h.1.2, symInt_eq_of_beq h.2]
  | .intCast _ _, .intCast _ _, h => by
      simp [SymInt.beq] at h; rw [h.1, symInt_eq_of_beq h.2]
  | .ite _ _ _, .ite _ _ _, h => by
      simp [SymInt.beq] at h; rw [symBool_eq_of_beq h.1.1, symInt_eq_of_beq h.1.2, symInt_eq_of_beq h.2]
  | .const _, .tidX, h | .const _, .tidY, h | .const _, .ctaidX, h | .const _, .ctaidY, h
  | .const _, .index1D _, h | .const _, .paramScalar _, h | .const _, .sliceLen _, h
  | .const _, .havoc _ _, h | .const _, .opaqueUint _ _, h | .const _, .binOp _ _ _ _, h
  | .const _, .intCast _ _, h | .const _, .ite _ _ _, h
  | .tidX, .const _, h | .tidX, .tidY, h | .tidX, .ctaidX, h | .tidX, .ctaidY, h
  | .tidX, .index1D _, h | .tidX, .paramScalar _, h | .tidX, .sliceLen _, h
  | .tidX, .havoc _ _, h | .tidX, .opaqueUint _ _, h | .tidX, .binOp _ _ _ _, h
  | .tidX, .intCast _ _, h | .tidX, .ite _ _ _, h
  | .tidY, .const _, h | .tidY, .tidX, h | .tidY, .ctaidX, h | .tidY, .ctaidY, h
  | .tidY, .index1D _, h | .tidY, .paramScalar _, h | .tidY, .sliceLen _, h
  | .tidY, .havoc _ _, h | .tidY, .opaqueUint _ _, h | .tidY, .binOp _ _ _ _, h
  | .tidY, .intCast _ _, h | .tidY, .ite _ _ _, h
  | .ctaidX, .const _, h | .ctaidX, .tidX, h | .ctaidX, .tidY, h | .ctaidX, .ctaidY, h
  | .ctaidX, .index1D _, h | .ctaidX, .paramScalar _, h | .ctaidX, .sliceLen _, h
  | .ctaidX, .havoc _ _, h | .ctaidX, .opaqueUint _ _, h | .ctaidX, .binOp _ _ _ _, h
  | .ctaidX, .intCast _ _, h | .ctaidX, .ite _ _ _, h
  | .ctaidY, .const _, h | .ctaidY, .tidX, h | .ctaidY, .tidY, h | .ctaidY, .ctaidX, h
  | .ctaidY, .index1D _, h | .ctaidY, .paramScalar _, h | .ctaidY, .sliceLen _, h
  | .ctaidY, .havoc _ _, h | .ctaidY, .opaqueUint _ _, h | .ctaidY, .binOp _ _ _ _, h
  | .ctaidY, .intCast _ _, h | .ctaidY, .ite _ _ _, h
  | .index1D _, .const _, h | .index1D _, .tidX, h | .index1D _, .tidY, h | .index1D _, .ctaidX, h
  | .index1D _, .ctaidY, h | .index1D _, .paramScalar _, h | .index1D _, .sliceLen _, h
  | .index1D _, .havoc _ _, h | .index1D _, .opaqueUint _ _, h | .index1D _, .binOp _ _ _ _, h
  | .index1D _, .intCast _ _, h | .index1D _, .ite _ _ _, h
  | .paramScalar _, .const _, h | .paramScalar _, .tidX, h | .paramScalar _, .tidY, h | .paramScalar _, .ctaidX, h
  | .paramScalar _, .ctaidY, h | .paramScalar _, .index1D _, h | .paramScalar _, .sliceLen _, h
  | .paramScalar _, .havoc _ _, h | .paramScalar _, .opaqueUint _ _, h | .paramScalar _, .binOp _ _ _ _, h
  | .paramScalar _, .intCast _ _, h | .paramScalar _, .ite _ _ _, h
  | .sliceLen _, .const _, h | .sliceLen _, .tidX, h | .sliceLen _, .tidY, h | .sliceLen _, .ctaidX, h
  | .sliceLen _, .ctaidY, h | .sliceLen _, .index1D _, h | .sliceLen _, .paramScalar _, h
  | .sliceLen _, .havoc _ _, h | .sliceLen _, .opaqueUint _ _, h | .sliceLen _, .binOp _ _ _ _, h
  | .sliceLen _, .intCast _ _, h | .sliceLen _, .ite _ _ _, h
  | .havoc _ _, .const _, h | .havoc _ _, .tidX, h | .havoc _ _, .tidY, h | .havoc _ _, .ctaidX, h
  | .havoc _ _, .ctaidY, h | .havoc _ _, .index1D _, h | .havoc _ _, .paramScalar _, h | .havoc _ _, .sliceLen _, h
  | .havoc _ _, .opaqueUint _ _, h | .havoc _ _, .binOp _ _ _ _, h | .havoc _ _, .intCast _ _, h | .havoc _ _, .ite _ _ _, h
  | .opaqueUint _ _, .const _, h | .opaqueUint _ _, .tidX, h | .opaqueUint _ _, .tidY, h | .opaqueUint _ _, .ctaidX, h
  | .opaqueUint _ _, .ctaidY, h | .opaqueUint _ _, .index1D _, h | .opaqueUint _ _, .paramScalar _, h | .opaqueUint _ _, .sliceLen _, h
  | .opaqueUint _ _, .havoc _ _, h | .opaqueUint _ _, .binOp _ _ _ _, h | .opaqueUint _ _, .intCast _ _, h | .opaqueUint _ _, .ite _ _ _, h
  | .binOp _ _ _ _, .const _, h | .binOp _ _ _ _, .tidX, h | .binOp _ _ _ _, .tidY, h | .binOp _ _ _ _, .ctaidX, h
  | .binOp _ _ _ _, .ctaidY, h | .binOp _ _ _ _, .index1D _, h | .binOp _ _ _ _, .paramScalar _, h | .binOp _ _ _ _, .sliceLen _, h
  | .binOp _ _ _ _, .havoc _ _, h | .binOp _ _ _ _, .opaqueUint _ _, h | .binOp _ _ _ _, .intCast _ _, h | .binOp _ _ _ _, .ite _ _ _, h
  | .intCast _ _, .const _, h | .intCast _ _, .tidX, h | .intCast _ _, .tidY, h | .intCast _ _, .ctaidX, h
  | .intCast _ _, .ctaidY, h | .intCast _ _, .index1D _, h | .intCast _ _, .paramScalar _, h | .intCast _ _, .sliceLen _, h
  | .intCast _ _, .havoc _ _, h | .intCast _ _, .opaqueUint _ _, h | .intCast _ _, .binOp _ _ _ _, h | .intCast _ _, .ite _ _ _, h
  | .ite _ _ _, .const _, h | .ite _ _ _, .tidX, h | .ite _ _ _, .tidY, h | .ite _ _ _, .ctaidX, h
  | .ite _ _ _, .ctaidY, h | .ite _ _ _, .index1D _, h | .ite _ _ _, .paramScalar _, h | .ite _ _ _, .sliceLen _, h
  | .ite _ _ _, .havoc _ _, h | .ite _ _ _, .opaqueUint _ _, h | .ite _ _ _, .binOp _ _ _ _, h | .ite _ _ _, .intCast _ _, h => by
      contradiction

theorem symBool_eq_of_beq : ∀ {p q : SymBool}, SymBool.beq p q = true → p = q
  | .lit _, .lit _, h => by simp [SymBool.beq] at h; exact congrArg _ h
  | .mbarReady _ _, .mbarReady _ _, h => by simp [SymBool.beq] at h; rw [h.1, symInt_eq_of_beq h.2]
  | .cmp _ _ _, .cmp _ _ _, h => by simp [SymBool.beq] at h; rw [h.1.1, symInt_eq_of_beq h.1.2, symInt_eq_of_beq h.2]
  | .not _, .not _, h => by simp [SymBool.beq] at h; exact congrArg _ (symBool_eq_of_beq h)
  | .and _ _, .and _ _, h => by simp [SymBool.beq] at h; rw [symBool_eq_of_beq h.1, symBool_eq_of_beq h.2]
  | .or _ _, .or _ _, h => by simp [SymBool.beq] at h; rw [symBool_eq_of_beq h.1, symBool_eq_of_beq h.2]
  | .ite _ _ _, .ite _ _ _, h => by
      simp [SymBool.beq] at h; rw [symBool_eq_of_beq h.1.1, symBool_eq_of_beq h.1.2, symBool_eq_of_beq h.2]
  | .lit _, .mbarReady _ _, h | .lit _, .cmp _ _ _, h | .lit _, .not _, h | .lit _, .and _ _, h | .lit _, .or _ _, h | .lit _, .ite _ _ _, h
  | .mbarReady _ _, .lit _, h | .mbarReady _ _, .cmp _ _ _, h | .mbarReady _ _, .not _, h | .mbarReady _ _, .and _ _, h | .mbarReady _ _, .or _ _, h | .mbarReady _ _, .ite _ _ _, h
  | .cmp _ _ _, .lit _, h | .cmp _ _ _, .mbarReady _ _, h | .cmp _ _ _, .not _, h | .cmp _ _ _, .and _ _, h | .cmp _ _ _, .or _ _, h | .cmp _ _ _, .ite _ _ _, h
  | .not _, .lit _, h | .not _, .mbarReady _ _, h | .not _, .cmp _ _ _, h | .not _, .and _ _, h | .not _, .or _ _, h | .not _, .ite _ _ _, h
  | .and _ _, .lit _, h | .and _ _, .mbarReady _ _, h | .and _ _, .cmp _ _ _, h | .and _ _, .not _, h | .and _ _, .or _ _, h | .and _ _, .ite _ _ _, h
  | .or _ _, .lit _, h | .or _ _, .mbarReady _ _, h | .or _ _, .cmp _ _ _, h | .or _ _, .not _, h | .or _ _, .and _ _, h | .or _ _, .ite _ _ _, h
  | .ite _ _ _, .lit _, h | .ite _ _ _, .mbarReady _ _, h | .ite _ _ _, .cmp _ _ _, h | .ite _ _ _, .not _, h | .ite _ _ _, .and _ _, h | .ite _ _ _, .or _ _, h => by
      contradiction
end

instance : LawfulBEq SymInt where
  rfl := symInt_beq_refl _
  eq_of_beq := symInt_eq_of_beq

instance : LawfulBEq SymBool where
  rfl := symBool_beq_refl _
  eq_of_beq := symBool_eq_of_beq

mutual
/-- Evaluate `SymInt` to a concrete `Nat` under `(env, t, havoc)`. -/
def SymInt.eval (env : InputEnv) (t : ThreadCtx) (havoc : Nat → Nat) : SymInt → Nat
  | .const n => n
  | .tidX => t.tx
  | .tidY => t.ty
  | .ctaidX => t.bx
  | .ctaidY => t.by_
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
  | .const _ | .tidX | .tidY | .ctaidX | .ctaidY | .index1D _ | .paramScalar _ | .sliceLen _ | .opaqueUint _ _ => true
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
  | .const _, _ | .tidX, _ | .tidY, _ | .ctaidX, _ | .ctaidY, _
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
  | .const _, _, heq | .tidX, _, heq | .tidY, _, heq | .ctaidX, _, heq | .ctaidY, _, heq
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
  | unk => trivial
  | int e1 =>
      cases actual with
      | int e2 =>
          simp only [SymVal.matchesVal] at heq hact ⊢
          rw [symIntEquivUnder_sound hhyp e1 e2 heq]; exact hact
      | _ => contradiction
  | bool b1 =>
      cases actual with
      | bool b2 =>
          simp only [SymVal.matchesVal] at heq hact ⊢
          rw [symBoolEquivUnder_sound hhyp b1 b2 heq]; exact hact
      | _ => contradiction
  | ptr sp1 b1 o1 =>
      cases actual with
      | ptr sp2 b2 o2 =>
          simp only [Bool.and_eq_true, beq_iff_eq, SymVal.matchesVal] at heq hact ⊢
          rcases heq with ⟨⟨rfl, rfl⟩, ho⟩
          rw [symIntEquivUnder_sound hhyp o1 o2 ho]; exact hact
      | _ => contradiction
  | smemOff s1 b1 =>
      cases actual with
      | smemOff s2 b2 =>
          simp only [Bool.and_eq_true, beq_iff_eq, SymVal.matchesVal] at heq hact ⊢
          rcases heq with ⟨rfl, hb⟩
          rw [symIntEquivUnder_sound hhyp b1 b2 hb]; exact hact
      | _ => contradiction
  | smemDesc s1 b1 =>
      cases actual with
      | smemDesc s2 b2 =>
          simp only [Bool.and_eq_true, beq_iff_eq, SymVal.matchesVal] at heq hact ⊢
          rcases heq with ⟨rfl, hb⟩
          rw [symIntEquivUnder_sound hhyp b1 b2 hb]; exact hact
      | _ => contradiction
  | tmemPtr c1 b1 =>
      cases actual with
      | tmemPtr c2 b2 =>
          simp only [Bool.and_eq_true, beq_iff_eq, SymVal.matchesVal] at heq hact ⊢
          rcases heq with ⟨rfl, hb⟩
          rw [symIntEquivUnder_sound hhyp b1 b2 hb]; exact hact
      | _ => contradiction
  | condPtr g1 sp1 b1 o1 =>
      cases actual with
      | ptr sp2 b2 o2 =>
          simp only [Bool.and_eq_true, beq_iff_eq, SymVal.matchesVal] at heq hact ⊢
          rcases heq with ⟨⟨rfl, rfl⟩, ho⟩
          intro _; rw [symIntEquivUnder_sound hhyp o1 o2 ho]; exact hact
      | unk =>
          simp only [SymVal.matchesVal] at heq ⊢
          intro hg1
          rw [symImpliesVal_sound hhyp g1 false heq] at hg1; contradiction
      | condPtr g2 sp2 b2 o2 =>
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

/-! ## Section 3: Block-Level Symbolic Execution (`symEvalRhs`, `symExecStmts`) -/

/-- Dense symbolic register file stored as a `List SymVal` indexed by `varId`. -/
abbrev SymRegList := List SymVal

/-- Read symbolic variable `v` from `sregs` (defaulting to `.unk`). -/
def SymRegList.getVar (sregs : SymRegList) (v : Nat) : SymVal :=
  sregs.getD v .unk

/-- Single-pass structural register write extending with `.unk` if `dst ≥ sregs.length`
    (zero `List.length` scans or dependent `Nat.decLt` proof construction in Lean's kernel `whnf`). -/
def SymRegList.setVar : SymRegList → Nat → SymVal → SymRegList
  | [], 0, sv => [sv]
  | [], d + 1, sv => .unk :: SymRegList.setVar [] d sv
  | _ :: rest, 0, sv => sv :: rest
  | x :: rest, d + 1, sv => x :: SymRegList.setVar rest d sv

/-- Lemma: reading `k` after `sregs.setVar dst sv` behaves identically to `RegFile.set`. -/
theorem getVar_setVar : ∀ (sregs : SymRegList) (dst k : Nat) (sv : SymVal),
    (sregs.setVar dst sv).getVar k = if k == dst then sv else sregs.getVar k
  | [], 0, 0, _ => rfl
  | [], 0, _ + 1, _ => rfl
  | [], d + 1, 0, _ => rfl
  | [], d + 1, k + 1, sv => by
      have ih := getVar_setVar [] d k sv
      have hbeq : ((k + 1) == (d + 1)) = (k == d) := by simp
      simpa [SymRegList.setVar, SymRegList.getVar, List.getD, hbeq] using ih
  | _ :: _, 0, 0, _ => rfl
  | _ :: _, 0, _ + 1, _ => rfl
  | _ :: _, d + 1, 0, _ => rfl
  | x :: rest, d + 1, k + 1, sv => by
      have ih := getVar_setVar rest d k sv
      have hbeq : ((k + 1) == (d + 1)) = (k == d) := by simp
      simpa [SymRegList.setVar, SymRegList.getVar, List.getD, hbeq] using ih

/-- Bind formal `BlockArg`s in parallel from `callerRegs` onto `baseRegs`. -/
def bindSymBlockArgs (callerRegs baseRegs : SymRegList) :
    List BlockArg → List Nat → SymRegList
  | [], _ | _, [] => baseRegs
  | fa :: fas, av :: avs =>
      bindSymBlockArgs callerRegs (baseRegs.setVar fa.varId (callerRegs.getVar av)) fas avs

/-- Exact correspondence between `bindBlockArgs` and `bindSymBlockArgs`. -/
theorem bindSymBlockArgs_correct {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat}
    {callerR : RegFile} {callerSR : SymRegList}
    (hcaller : ∀ v, (callerSR.getVar v).matchesVal env t havoc (callerR v)) :
    ∀ fargs : List BlockArg, ∀ actuals : List Nat,
      ∀ baseR : RegFile, ∀ baseSR : SymRegList,
      (∀ v, (baseSR.getVar v).matchesVal env t havoc (baseR v)) →
      ∀ v, ((bindSymBlockArgs callerSR baseSR fargs actuals).getVar v).matchesVal env t havoc
             ((bindBlockArgs callerR baseR fargs actuals) v)
  | [], _, _, _, hbase, v => hbase v
  | _ :: _, [], _, _, hbase, v => hbase v
  | fa :: fas, av :: avs, baseR, baseSR, hbase, v => by
      dsimp only [bindBlockArgs, bindSymBlockArgs]
      apply bindSymBlockArgs_correct hcaller fas avs
      intro k; unfold RegFile.set; rw [getVar_setVar]
      by_cases hk : (k == fa.varId) = true
      · simp [hk]; exact hcaller av
      · simp [hk]; exact hbase k

mutual
/-- Fuel-threaded node budget check on `SymInt` (`O(budget)` steps, short-circuiting on `0`). -/
def symIntRemNodes : Nat → SymInt → Option Nat
  | 0, _ => none
  | fuel + 1, .binOp _ _ a b =>
      match symIntRemNodes fuel a with
      | none => none
      | some f1 => symIntRemNodes f1 b
  | fuel + 1, .intCast _ a => symIntRemNodes fuel a
  | fuel + 1, .ite _ a b =>
      match symIntRemNodes fuel a with
      | none => none
      | some f1 => symIntRemNodes f1 b
  | fuel + 1, _ => some fuel

/-- Fuel-threaded node budget check on `SymBool` (`O(budget)` steps, short-circuiting on `0`). -/
def symBoolRemNodes : Nat → SymBool → Option Nat
  | 0, _ => none
  | fuel + 1, .not p => symBoolRemNodes fuel p
  | fuel + 1, .mbarReady _ s => symIntRemNodes fuel s
  | fuel + 1, .and p q | fuel + 1, .or p q =>
      match symBoolRemNodes fuel p with
      | none => none
      | some f1 => symBoolRemNodes f1 q
  | fuel + 1, .ite _ p q =>
      match symBoolRemNodes fuel p with
      | none => none
      | some f1 => symBoolRemNodes f1 q
  | fuel + 1, .cmp _ a b =>
      match symIntRemNodes fuel a with
      | none => none
      | some f1 => symIntRemNodes f1 b
  | fuel + 1, .lit _ => some fuel
end

/-- Syntactic AST depth of a `SymInt` (used only to bound DAG-to-tree duplication on non-index values). -/
def symIntDepth : SymInt → Nat
  | .binOp _ _ a b => 1 + max (symIntDepth a) (symIntDepth b)
  | .intCast _ a => 1 + symIntDepth a
  | .ite _ a b => 1 + max (symIntDepth a) (symIntDepth b)
  | _ => 1

/-- Syntactic AST depth of a `SymBool`. -/
def symBoolDepth : SymBool → Nat
  | .not p => 1 + symBoolDepth p
  | .and p q | .or p q => 1 + max (symBoolDepth p) (symBoolDepth q)
  | .ite c p q => 1 + max (symBoolDepth c) (max (symBoolDepth p) (symBoolDepth q))
  | .cmp _ a b => 1 + max (symIntDepth a) (symIntDepth b)
  | _ => 1

/-- Soundly widen large (`> 36` AST nodes) non-index `SymInt` / `SymBool` expressions to `.unk` in `O(1)`. -/
def widenDeepVal (sv : SymVal) : SymVal :=
  match sv with
  | .int e => if (symIntRemNodes 36 e).isNone then .unk else sv
  | .bool b => if (symBoolRemNodes 48 b).isNone then .unk else sv
  | _ => sv

/-- Soundness of `widenDeepVal`: widening to `.unk` preserves `SymVal.matchesVal`. -/
theorem widenDeepVal_matches {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat}
    {sv : SymVal} {v : Val} (h : sv.matchesVal env t havoc v) :
    (widenDeepVal sv).matchesVal env t havoc v := by
  unfold widenDeepVal
  cases sv <;> simp only <;> try exact h
  all_goals (split <;> first | trivial | exact h)

/-- Raw 1-step symbolic evaluator for `Rhs` before `widenDeepVal`. -/
def symEvalRhsRaw (f : MirFunc) (sregs : SymRegList) : Rhs → SymVal
  | .copy src => sregs.getVar src
  | .constInt w val => .int (.const (wordMod w val))
  | .constBool b => .bool (.lit b)
  | .sreg .tidX => .int .tidX
  | .sreg .tidY => .int .tidY
  | .sreg .ctaidX => .int .ctaidX
  | .sreg .ctaidY => .int .ctaidY
  | .sreg .ntidX => .int (.const f.blockDimX)
  | .sreg .ntidY => .int (.const f.blockDimY)
  | .sreg .laneId => .int (.binOp .rem 32 .tidX (.const 32))
  | .sreg .clusterCtaidX => .int (.binOp .rem 0 .ctaidX (.const (max 1 f.clusterDimX)))
  | .index1D => .int (.index1D f.blockDimX)
  | .paramScalar p => .int (.paramScalar p)
  | .slicePtr p => .ptr .global p (.const 0)
  | .sliceLen p => .int (.sliceLen p)
  | .sharedBase s => .ptr .shared s (.const 0)
  | .binOp op w l r =>
      match sregs.getVar l, sregs.getVar r with
      | .int a, .int b => .int (.binOp op w a b)
      | .smemOff sid base, .int d =>
          if op == .add then .smemOff sid (.binOp .add 0 base d)
          else if op == .shr && d == .const 4 then .smemDesc sid base
          else .unk
      | .int d, .smemOff sid base =>
          if op == .add then .smemOff sid (.binOp .add 0 d base) else .unk
      | .smemDesc sid base, .int d =>
          if op == .add then .smemDesc sid (.binOp .add 0 base (.binOp .mul 0 d (.const 16)))
          else if op == .bitAnd || op == .bitOr then .smemDesc sid base
          else .unk
      | .int d, .smemDesc sid base =>
          if op == .add then .smemDesc sid (.binOp .add 0 (.binOp .mul 0 d (.const 16)) base)
          else if op == .bitAnd || op == .bitOr then .smemDesc sid base
          else .unk
      | .tmemPtr cols rel, .int d =>
          if op == .add then .tmemPtr cols (.binOp .add w rel d) else .unk
      | .int d, .tmemPtr cols rel =>
          if op == .add then .tmemPtr cols (.binOp .add w d rel) else .unk
      | _, _ => .unk
  | .cmpOp op l r =>
      match sregs.getVar l, sregs.getVar r with
      | .int a, .int b => .bool (.cmp op a b)
      | _, _ => .unk
  | .notBool src =>
      match sregs.getVar src with
      | .bool b => .bool (.not b)
      | _ => .unk
  | .intCast dstW src =>
      match sregs.getVar src with
      | .int a => .int (.intCast dstW a)
      | .smemOff sid b => if decide (32 ≤ dstW) then .smemOff sid b else .unk
      | _ => .unk
  | .ptrOffset base idx =>
      match sregs.getVar base, sregs.getVar idx with
      | .ptr sp bid off, .int delta => .ptr sp bid (.binOp .add 0 off delta)
      | _, _ => .unk
  | .load _ | .loadTmemBase _ _ => .unk
  | .mbarTryWait barId slotVar =>
      match sregs.getVar slotVar with
      | .int s => .bool (.mbarReady barId s)
      | _ => .unk
  | .cvtaShared ptr =>
      match sregs.getVar ptr with
      | .ptr .shared sid off =>
          let eb := max 1 (f.sharedDecls.getD sid { size := 0 }).elemBytes
          .smemOff sid (.binOp .mul 0 off (.const eb))
      | _ => .unk
  | .opaqueUint id w => .int (.opaqueUint id w)
  | .opaqueVal => .unk

/-- Symbolically evaluate one `Rhs` expression with depth-10 widening on deep arithmetic chains. -/
def symEvalRhs (f : MirFunc) (sregs : SymRegList) (rhs : Rhs) : SymVal :=
  widenDeepVal (symEvalRhsRaw f sregs rhs)

/-- Soundness of `symEvalRhs`: evaluating `rhs` symbolically matches `evalRhs`. -/
theorem symEvalRhs_correct (f : MirFunc) (env : InputEnv) (t : ThreadCtx)
    (havoc : Nat → Nat) (regs : RegFile) (sregs : SymRegList)
    (hmatch : ∀ v, (sregs.getVar v).matchesVal env t havoc (regs v)) (rhs : Rhs) :
    (symEvalRhs f sregs rhs).matchesVal env t havoc (evalRhs f env t regs rhs) := by
  apply widenDeepVal_matches
  cases rhs with
  | copy src => exact hmatch src
  | constInt _ _ | constBool _ | index1D | paramScalar _ | slicePtr _
  | sliceLen _ | sharedBase _ | load _ | loadTmemBase _ _
  | opaqueUint _ _ | opaqueVal =>
      simp [symEvalRhsRaw, evalRhs, SymVal.matchesVal, SymInt.eval, SymBool.eval]
  | mbarTryWait barId slotVar =>
      have hs := hmatch slotVar
      dsimp only [symEvalRhsRaw]
      cases hss : sregs.getVar slotVar <;> simp [hss, SymVal.matchesVal] at hs ⊢
      simp [evalRhs, hs, SymBool.eval]
  | sreg r =>
      cases r <;> simp [symEvalRhsRaw, evalRhs, SymVal.matchesVal, SymInt.eval, evalBinOp]
  | cvtaShared ptr =>
      have hp := hmatch ptr
      dsimp only [symEvalRhsRaw]
      cases hsp : sregs.getVar ptr with
      | ptr sp bid off =>
          cases sp <;> simp [hsp, SymVal.matchesVal] at hp ⊢
          simp [evalRhs, hp, SymInt.eval, evalBinOp, wordMod]
      | _ => simp [SymVal.matchesVal]
  | binOp op w l r =>
      have hl := hmatch l; have hr := hmatch r
      dsimp only [symEvalRhsRaw]
      cases hsl : sregs.getVar l <;> cases hsr : sregs.getVar r <;>
        simp [hsl, hsr, SymVal.matchesVal] at hl hr ⊢
      · simp [evalRhs, hl, hr, SymInt.eval]
      · cases op <;> simp [evalRhs, hl, hr, SymInt.eval, evalBinOp, wordMod]
      · cases op <;> simp [evalRhs, hl, hr, SymInt.eval, evalBinOp, wordMod]
      · cases op <;> simp [evalRhs, hl, hr, SymInt.eval, evalBinOp]
      · rename_i sid base d
        cases op <;> try simp [evalRhs, hl, hr, SymInt.eval, evalBinOp, wordMod]
        by_cases hd4 : d = .const 4
        · subst hd4; simp [SymInt.eval]
        · simp [hd4]
      · cases op <;> simp [evalRhs, hl, hr, SymInt.eval, evalBinOp, wordMod]
      · cases op <;> simp [evalRhs, hl, hr, SymInt.eval, evalBinOp]
  | cmpOp op l r =>
      have hl := hmatch l; have hr := hmatch r
      dsimp only [symEvalRhsRaw]
      cases hsl : sregs.getVar l <;> cases hsr : sregs.getVar r <;>
        simp [hsl, hsr, SymVal.matchesVal] at hl hr ⊢
      simp [evalRhs, hl, hr, SymBool.eval]
  | notBool src =>
      have hs := hmatch src
      dsimp only [symEvalRhsRaw]
      cases hss : sregs.getVar src <;> simp [hss, SymVal.matchesVal] at hs ⊢
      simp [evalRhs, hs, SymBool.eval]
  | intCast dw src =>
      have hs := hmatch src
      dsimp only [symEvalRhsRaw]
      cases hss : sregs.getVar src <;> simp [hss, SymVal.matchesVal] at hs ⊢
      · simp [evalRhs, hs, SymInt.eval]
      · by_cases h32 : 32 ≤ dw
        · simp [h32, evalRhs, hs]
        · simp [h32]
  | ptrOffset base idx =>
      have hb := hmatch base; have hi := hmatch idx
      dsimp only [symEvalRhsRaw]
      cases hsb : sregs.getVar base <;> cases hsi : sregs.getVar idx <;>
        simp [hsb, hsi, SymVal.matchesVal] at hb hi ⊢
      simp [evalRhs, hb, hi, SymInt.eval, evalBinOp, wordMod]

/-- Symbolic memory event recorded during `symExecStmts`. -/
structure SymMemEvent where
  /-- `true` for store / TMA / `tmemAlloc`, `false` for load / `smemDescRead`. -/
  isWrite : Bool
  /-- Target address space (`.global` or `.shared`). -/
  space : AddrSpace
  /-- Buffer / parameter ID. -/
  bufId : Nat
  /-- Symbolic base offset (`none` if pointer/descriptor provenance was unknown). -/
  indexOpt : Option SymInt
  /-- Number of elements (`elemDiv = 1`) or bytes (`elemDiv = elemBytes`) accessed (`≥ 1`). -/
  span : Nat := 1
  /-- Divisor converting raw offset units into element indices (`1` for element pointers, `elemBytes` for byte descriptors). -/
  elemDiv : Nat := 1
  /-- `some (barId, slot)` when released by an `mbarrier` arrival or TMA transfer on `BAR[slot]`. -/
  syncBar : Option (Nat × SymInt) := none
  /-- Path condition active at this access site. -/
  pathCond : SymBool
  /-- Symbolic barrier phase active at this access site. -/
  phaseSpec : PhaseSpec
  deriving Repr, DecidableEq, Lean.ToExpr

/-- Evaluate an optional symbolic `(barId, slotSym)` handle to `Option (Nat × Nat)`. -/
def evalSyncBar (env : InputEnv) (t : ThreadCtx) (havoc : Nat → Nat) :
    Option (Nat × SymInt) → Option (Nat × Nat)
  | none => none
  | some (b, s) => some (b, s.eval env t havoc)

/-- Symbolic assertion event recorded during `symExecStmts`. -/
structure SymAssertEvent where
  /-- Deterministic `assertId` of the `Stmt.assertCond` instruction. -/
  assertId : Nat
  /-- Active path condition at the assertion site. -/
  pathCond : SymBool
  /-- Symbolic boolean condition `cond` tested by `Stmt.assertCond`. -/
  condOpt : Option SymBool
  deriving Repr, DecidableEq, Lean.ToExpr

/-- Symbolic Blackwell Tensor Memory (`TMEM`) event recorded during `symExecStmts`. -/
structure SymTmemEvent where
  /-- Dynamically allocated `TMEM` column count (`tcgen05_alloc`). -/
  allocCols : Nat
  /-- Symbolic 32-bit `(lane << 16) + col` address (`none` if provenance was unknown). -/
  relAddrOpt : Option SymInt
  /-- Number of `TMEM` lanes accessed. -/
  heightLanes : Nat
  /-- Number of 32-bit `TMEM` columns accessed. -/
  widthCols : Nat
  /-- True for warp-restricted `tcgen05.ld` / `tcgen05.st`. -/
  warpRestricted : Bool
  /-- Active path condition at the `TMEM` operation site. -/
  pathCond : SymBool
  deriving Repr, DecidableEq, Lean.ToExpr

/-- Resolve a pointer `SymVal` under `pCond` into a 1-element `SymMemEvent` (or `indexOpt := none` on failure). -/
def symSinglePtrEvent (pCond : SymBool) (ps : PhaseSpec) (isWrite : Bool) (sv : SymVal) : SymMemEvent :=
  match sv with
  | .ptr sp bid off =>
      { isWrite := isWrite, space := sp, bufId := bid, indexOpt := some off, pathCond := pCond, phaseSpec := ps }
  | .condPtr g sp bid off =>
      if symImpliesVal pCond g true then
        { isWrite := isWrite, space := sp, bufId := bid, indexOpt := some off, pathCond := pCond, phaseSpec := ps }
      else
        { isWrite := isWrite, space := .global, bufId := 0, indexOpt := none, pathCond := pCond, phaseSpec := ps }
  | _ =>
      { isWrite := isWrite, space := .global, bufId := 0, indexOpt := none, pathCond := pCond, phaseSpec := ps }

/-- Execute a block's statements symbolically from `(pCond, sregs, ps)`.
    - **What it does**: Steps through a basic block's `List Stmt`, updating `sregs` and `ps`
      and recording every global/shared memory event (`SymMemEvent`), assertion (`SymAssertEvent`),
      and Blackwell `TMEM` event (`SymTmemEvent`).
    - **Fail-closed invariant**: Any memory or `TMEM` instruction on an unresolved handle emits
      `indexOpt := none` / `relAddrOpt := none`, causing `validSymFunc` to return `false`. -/
def symExecStmts (f : MirFunc) (pCond : SymBool) (sregs : SymRegList) (ps : PhaseSpec) (term : Terminator) :
    List Stmt → SymRegList × PhaseSpec × List SymMemEvent × List SymAssertEvent × List SymTmemEvent
  | [] => (sregs, ps, [], [], [])
  | stmt :: rest =>
      match stmt with
      | .assign dst (.load ptr) =>
          let mev := [symSinglePtrEvent pCond ps false (sregs.getVar ptr)]
          let (r', ps', mevs, aevs, tevs) := symExecStmts f pCond (sregs.setVar dst .unk) ps term rest
          (r', ps', mev ++ mevs, aevs, tevs)
      | .assign dst (.loadTmemBase ptr cols) =>
          let mev := [symSinglePtrEvent pCond ps false (sregs.getVar ptr)]
          let (r', ps', mevs, aevs, tevs) := symExecStmts f pCond (sregs.setVar dst (.tmemPtr cols (.const 0))) ps term rest
          (r', ps', mev ++ mevs, aevs, tevs)
      | .assign dst rhs =>
          let sv := symEvalRhs f sregs rhs
          symExecStmts f pCond (sregs.setVar dst sv) ps term rest
      | .store ptr _ =>
          let baseEv := symSinglePtrEvent pCond ps true (sregs.getVar ptr)
          let mev := match findTrailArrive f 8 term rest with
            | none => [baseEv]
            | some (b, sv) =>
                match sregs.getVar sv with
                | .int sSym =>
                    [{ baseEv with syncBar := if baseEv.space == .shared then some (b, sSym) else none }]
                | _ =>
                    [{ baseEv with indexOpt := none }]
          let (r', ps', mevs, aevs, tevs) := symExecStmts f pCond sregs ps term rest
          (r', ps', mev ++ mevs, aevs, tevs)
      | .barrierSlotCheck ptr =>
          let mev := [symSinglePtrEvent pCond ps false (sregs.getVar ptr)]
          let (r', ps', mevs, aevs, tevs) := symExecStmts f pCond sregs ps term rest
          (r', ps', mev ++ mevs, aevs, tevs)
      | .assertCond aid cond =>
          let cOpt := match sregs.getVar cond with | .bool b => some b | _ => none
          let aev : SymAssertEvent := { assertId := aid, pathCond := pCond, condOpt := cOpt }
          let (r', ps', mevs, aevs, tevs) := symExecStmts f pCond sregs ps term rest
          (r', ps', mevs, aev :: aevs, tevs)
      | .clusterRankCheck _ | .mbarArrive _ _ | .mbarArriveExpectTx _ _ _ =>
          symExecStmts f pCond sregs ps term rest
      | .syncThreads =>
          symExecStmts f pCond sregs ps.stepSync term rest
      | .tmaCopyG2S ptr barId slotVar spanElems =>
          let span := max 1 spanElems
          let mev := match sregs.getVar ptr, sregs.getVar slotVar with
            | .ptr sp bid off, .int sSym =>
                [{ isWrite := true, space := sp, bufId := bid, indexOpt := some off,
                   span := span, elemDiv := 1, syncBar := some (barId, sSym), pathCond := pCond, phaseSpec := ps }]
            | _, _ =>
                [{ isWrite := true, space := .shared, bufId := 0, indexOpt := none,
                   span := span, elemDiv := 1, pathCond := pCond, phaseSpec := ps }]
          let (r', ps', mevs, aevs, tevs) := symExecStmts f pCond sregs ps term rest
          (r', ps', mev ++ mevs, aevs, tevs)
      | .tmemAlloc ptr _cols =>
          let lane0Cond : SymBool :=
            .and pCond (.cmp .eq (.binOp .rem 0 .tidX (.const 32)) (.const 0))
          let mev := [symSinglePtrEvent lane0Cond ps true (sregs.getVar ptr)]
          let (r', ps', mevs, aevs, tevs) := symExecStmts f pCond sregs ps term rest
          (r', ps', mev ++ mevs, aevs, tevs)
      | .smemDescRead descVar tileBytes =>
          let span := max 1 tileBytes
          let mev := match sregs.getVar descVar with
            | .smemDesc sid byteOff =>
                let eb := max 1 (f.sharedDecls.getD sid { size := 0 }).elemBytes
                [{ isWrite := false, space := .shared, bufId := sid, indexOpt := some byteOff,
                   span := span, elemDiv := eb, pathCond := pCond, phaseSpec := ps }]
            | _ =>
                [{ isWrite := false, space := .shared, bufId := 0, indexOpt := none,
                   span := span, elemDiv := 1, pathCond := pCond, phaseSpec := ps }]
          let (r', ps', mevs, aevs, tevs) := symExecStmts f pCond sregs ps term rest
          (r', ps', mev ++ mevs, aevs, tevs)
      | .tmemAccess tmemVar hl wc wr =>
          let tev := match sregs.getVar tmemVar with
            | .tmemPtr cols rel =>
                [{ allocCols := cols, relAddrOpt := some rel, heightLanes := hl,
                   widthCols := wc, warpRestricted := wr, pathCond := pCond }]
            | _ =>
                [{ allocCols := 0, relAddrOpt := none, heightLanes := hl,
                   widthCols := wc, warpRestricted := wr, pathCond := pCond }]
          let (r', ps', mevs, aevs, tevs) := symExecStmts f pCond sregs ps term rest
          (r', ps', mevs, aevs, tev ++ tevs)

/-- Helper lemma discharging single-pointer `symSinglePtrEvent` correspondence in `symExecStmts_correct`. -/
theorem symSinglePtrEvent_sound {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat}
    {pCond : SymBool} (hpc : pCond.eval env t havoc = true)
    {ps : PhaseSpec} {phase : Nat} (hps : ps.matchesPhase phase = true)
    {isW : Bool} {sv : SymVal} {v : Val} {ev : MemEvent}
    (hmatch : sv.matchesVal env t havoc v)
    (hsome : (symSinglePtrEvent pCond ps isW sv).indexOpt.isSome = true)
    (hmem : ev ∈ (match v with | .ptr sp bid off => [{ isWrite := isW, space := sp, bufId := bid, index := off, phase := phase : MemEvent }] | _ => [])) :
    ∃ idxSym : SymInt, ∃ k : Nat,
      let sev := symSinglePtrEvent pCond ps isW sv
      k < sev.span ∧ 0 < sev.elemDiv ∧
      sev.isWrite = ev.isWrite ∧ sev.space = ev.space ∧ sev.bufId = ev.bufId ∧
      evalSyncBar env t havoc sev.syncBar = ev.syncBar ∧ sev.indexOpt = some idxSym ∧
      (idxSym.eval env t havoc + k) / sev.elemDiv = ev.index ∧
      sev.pathCond.eval env t havoc = true ∧ sev.phaseSpec.matchesPhase ev.phase = true := by
  cases sv with
  | ptr sp bid off =>
      dsimp only [symSinglePtrEvent]
      rw [SymVal.matchesVal] at hmatch; subst hmatch
      simp only [List.mem_singleton] at hmem; subst hmem
      exact ⟨off, 0, Nat.zero_lt_one, Nat.zero_lt_one, rfl, rfl, rfl, rfl, rfl, by simp, hpc, hps⟩
  | condPtr g sp bid off =>
      dsimp only [symSinglePtrEvent] at hsome ⊢
      by_cases himp : symImpliesVal pCond g true = true
      · rw [if_pos himp] at hsome ⊢
        rw [SymVal.matchesVal] at hmatch
        rw [hmatch (symImpliesVal_sound hpc g true himp)] at hmem
        simp only [List.mem_singleton] at hmem; subst hmem
        exact ⟨off, 0, Nat.zero_lt_one, Nat.zero_lt_one, rfl, rfl, rfl, rfl, rfl, by simp, hpc, hps⟩
      · rw [if_neg himp] at hsome; simp at hsome
  | int _ | bool _ | smemOff _ _ | smemDesc _ _ | tmemPtr _ _ | unk =>
      dsimp only [symSinglePtrEvent] at hsome; simp at hsome

/-- Exact correspondence between `symExecStmts` and concrete `execStmts` when `pCond` holds and all
    symbolic memory and `TMEM` events in the block carry `isSome` offsets. -/
theorem symExecStmts_correct (f : MirFunc) (env : InputEnv) (t : ThreadCtx)
    (havoc : Nat → Nat) (pCond : SymBool) (hpc : pCond.eval env t havoc = true) (term : Terminator) :
    ∀ stmts : List Stmt, ∀ regs : RegFile, ∀ sregs : SymRegList,
      ∀ phase : Nat, ∀ ps : PhaseSpec,
      (∀ v, (sregs.getVar v).matchesVal env t havoc (regs v)) →
      ps.matchesPhase phase = true →
      ((symExecStmts f pCond sregs ps term stmts).2.2.1.all (fun m => m.indexOpt.isSome)) = true →
      ((symExecStmts f pCond sregs ps term stmts).2.2.2.2.all (fun tm => tm.relAddrOpt.isSome)) = true →
      (∀ v, ((symExecStmts f pCond sregs ps term stmts).1.getVar v).matchesVal env t havoc
              ((execStmts f env t regs phase term stmts).1 v)) ∧
      (symExecStmts f pCond sregs ps term stmts).2.1.matchesPhase
        (execStmts f env t regs phase term stmts).2.1 = true ∧
      (∀ ev ∈ (execStmts f env t regs phase term stmts).2.2.1,
        ∃ sev ∈ (symExecStmts f pCond sregs ps term stmts).2.2.1, ∃ idxSym : SymInt, ∃ k : Nat,
          k < sev.span ∧ 0 < sev.elemDiv ∧
          sev.isWrite = ev.isWrite ∧ sev.space = ev.space ∧ sev.bufId = ev.bufId ∧
          evalSyncBar env t havoc sev.syncBar = ev.syncBar ∧ sev.indexOpt = some idxSym ∧
          (idxSym.eval env t havoc + k) / sev.elemDiv = ev.index ∧
          sev.pathCond.eval env t havoc = true ∧ sev.phaseSpec.matchesPhase ev.phase = true) ∧
      (∀ pfx sfx : List Stmt, ∀ aid cond : Nat,
        stmts = pfx ++ (.assertCond aid cond :: sfx) →
        ∃ aev ∈ (symExecStmts f pCond sregs ps term stmts).2.2.2.1,
          aev.assertId = aid ∧ aev.pathCond = pCond ∧
          ∀ bSym : SymBool, aev.condOpt = some bSym →
            (execStmts f env t regs phase term pfx).1 cond = .bool (bSym.eval env t havoc)) ∧
      (∀ tev ∈ (execStmts f env t regs phase term stmts).2.2.2,
        ∃ stev ∈ (symExecStmts f pCond sregs ps term stmts).2.2.2.2, ∃ relSym : SymInt,
          stev.allocCols = tev.allocCols ∧ stev.heightLanes = tev.heightLanes ∧
          stev.widthCols = tev.widthCols ∧ stev.warpRestricted = tev.warpRestricted ∧
          stev.relAddrOpt = some relSym ∧ relSym.eval env t havoc = tev.relAddr ∧
          stev.pathCond = pCond) := by
  intro stmts
  induction stmts with                                    -- Structural induction on `stmts`
  | nil =>
      intro regs sregs phase ps hmatch hps _ _
      refine ⟨hmatch, hps, ?_, ?_, ?_⟩
      · intro _ hmem; cases hmem
      · intro pfx _ _ _ heq; cases pfx <;> contradiction
      · intro _ hmem; cases hmem
  | cons stmt rest ih =>
      intro regs sregs phase ps hmatch hps hallSome hallTmem
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
              simp only [symExecStmts, List.all_append, List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true] at hallSome hallTmem
              have hnext : ∀ v, ((sregs.setVar dst .unk).getVar v).matchesVal env t havoc
                  ((regs.set dst .unk) v) := by
                intro v; unfold RegFile.set; rw [getVar_setVar]
                by_cases hv : (v == dst) = true
                · simp [hv, SymVal.matchesVal]
                · simp [hv]; exact hmatch v
              obtain ⟨hmR, hmP, hmE, hmA, hmT⟩ := ih (regs.set dst .unk) (sregs.setVar dst .unk) phase ps hnext hps hallSome.2 hallTmem
              simp only [execStmts, symExecStmts]
              refine ⟨hmR, hmP, ?_, ?_, hmT⟩
              · intro ev hev
                rw [List.mem_append] at hev
                rcases hev with hhead | htail
                · obtain ⟨idxS, k, hrest⟩ := symSinglePtrEvent_sound hpc hps (hmatch ptr) hallSome.1 hhead
                  exact ⟨_, List.mem_append_left _ (List.Mem.head _), idxS, k, hrest⟩
                · obtain ⟨sev, hsev, idxS, k, hrest⟩ := hmE ev htail
                  exact ⟨sev, List.mem_append_right _ hsev, idxS, k, hrest⟩
              · intro pfx sfx aid cond heq
                cases pfx with
                | nil => simp at heq
                | cons hd tl =>
                    simp only [List.cons_append, List.cons.injEq] at heq
                    rcases heq with ⟨rfl, htl⟩
                    simpa [execStmts] using hmA tl sfx aid cond htl
          | loadTmemBase ptr cols =>
              simp only [symExecStmts, List.all_append, List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true] at hallSome hallTmem
              have hnext : ∀ v, ((sregs.setVar dst (.tmemPtr cols (.const 0))).getVar v).matchesVal env t havoc
                  ((regs.set dst (.tmemPtr cols 0)) v) := by
                intro v; unfold RegFile.set; rw [getVar_setVar]
                by_cases hv : (v == dst) = true
                · simp [hv, SymVal.matchesVal, SymInt.eval]
                · simp [hv]; exact hmatch v
              obtain ⟨hmR, hmP, hmE, hmA, hmT⟩ :=
                ih (regs.set dst (.tmemPtr cols 0)) (sregs.setVar dst (.tmemPtr cols (.const 0))) phase ps hnext hps hallSome.2 hallTmem
              simp only [execStmts, symExecStmts]
              refine ⟨hmR, hmP, ?_, ?_, hmT⟩
              · intro ev hev
                rw [List.mem_append] at hev
                rcases hev with hhead | htail
                · obtain ⟨idxS, k, hrest⟩ := symSinglePtrEvent_sound hpc hps (hmatch ptr) hallSome.1 hhead
                  exact ⟨_, List.mem_append_left _ (List.Mem.head _), idxS, k, hrest⟩
                · obtain ⟨sev, hsev, idxS, k, hrest⟩ := hmE ev htail
                  exact ⟨sev, List.mem_append_right _ hsev, idxS, k, hrest⟩
              · intro pfx sfx aid cond heq
                cases pfx with
                | nil => simp at heq
                | cons hd tl =>
                    simp only [List.cons_append, List.cons.injEq] at heq
                    rcases heq with ⟨rfl, htl⟩
                    simpa [execStmts] using hmA tl sfx aid cond htl
          | _ =>
              simp only [execStmts, symExecStmts] at hallSome hallTmem ⊢
              obtain ⟨hmR, hmP, hmE, hmA, hmT⟩ := ih _ _ phase ps hnextGen hps hallSome hallTmem
              refine ⟨hmR, hmP, hmE, ?_, hmT⟩
              intro pfx sfx aid cond heq
              cases pfx with
              | nil => simp at heq
              | cons hd tl =>
                  simp only [List.cons_append, List.cons.injEq] at heq
                  rcases heq with ⟨rfl, htl⟩
                  simpa [execStmts] using hmA tl sfx aid cond htl
      | store ptr _ =>
          simp only [symExecStmts, List.all_append, Bool.and_eq_true] at hallSome hallTmem
          obtain ⟨hmR, hmP, hmE, hmA, hmT⟩ := ih regs sregs phase ps hmatch hps hallSome.2 hallTmem
          simp only [execStmts, symExecStmts]
          refine ⟨hmR, hmP, ?_, ?_, hmT⟩
          · intro ev hev
            rw [List.mem_append] at hev
            rcases hev with hhead | htail
            · cases hfba : findTrailArrive f 8 term rest with
              | none =>
                  rw [hfba] at hallSome hhead
                  simp only [ite_self] at hhead
                  simp only [List.all_cons, List.all_nil, Bool.and_true] at hallSome
                  obtain ⟨idxS, k, hrest⟩ := symSinglePtrEvent_sound hpc hps (hmatch ptr) hallSome.1 hhead
                  exact ⟨_, List.mem_append_left _ (List.Mem.head _), idxS, k, hrest⟩
              | some bsv =>
                  rcases bsv with ⟨b, sv⟩
                  rw [hfba] at hallSome hhead
                  dsimp only at hallSome hhead
                  have hsvM := hmatch sv
                  cases hssv : sregs.getVar sv with
                  | int sSym =>
                      dsimp only
                      rw [hssv] at hallSome ⊢
                      rw [hssv, SymVal.matchesVal] at hsvM
                      rw [hsvM] at hhead
                      simp only [List.all_cons, List.all_nil, Bool.and_true] at hallSome
                      have hp := hmatch ptr
                      cases hsp : sregs.getVar ptr with
                      | ptr sp bid off =>
                          dsimp only [symSinglePtrEvent] at hallSome ⊢
                          rw [hsp, SymVal.matchesVal] at hp; rw [hp] at hhead
                          simp only [List.mem_singleton] at hhead; subst hhead
                          refine ⟨_, List.mem_append_left _ (List.Mem.head _),
                                  off, 0, Nat.zero_lt_one, Nat.zero_lt_one, rfl, rfl, rfl, ?_, rfl, by simp, hpc, hps⟩
                          dsimp only [evalSyncBar]
                          cases sp <;> rfl
                      | condPtr g sp bid off =>
                          rw [hsp] at hallSome
                          by_cases himp : symImpliesVal pCond g true = true
                          · simp only [symSinglePtrEvent, himp, ↓reduceIte] at hallSome ⊢
                            rw [hsp, SymVal.matchesVal] at hp
                            rw [hp (symImpliesVal_sound hpc g true himp)] at hhead
                            simp only [List.mem_singleton] at hhead; subst hhead
                            refine ⟨_, List.mem_append_left _ (List.Mem.head _),
                                    off, 0, Nat.zero_lt_one, Nat.zero_lt_one, rfl, rfl, rfl, ?_, rfl, by simp, hpc, hps⟩
                            dsimp only [evalSyncBar]
                            cases sp <;> rfl
                          · simp [symSinglePtrEvent, himp] at hallSome
                      | _ =>
                          rw [hsp] at hallSome; dsimp only [symSinglePtrEvent] at hallSome; simp at hallSome
                  | _ => rw [hssv] at hallSome; simp at hallSome
            · obtain ⟨sev, hsev, idxS, k, hrest⟩ := hmE ev htail
              exact ⟨sev, List.mem_append_right _ hsev, idxS, k, hrest⟩
          · intro pfx sfx aid cond heq
            cases pfx with
            | nil => simp at heq
            | cons hd tl =>
                simp only [List.cons_append, List.cons.injEq] at heq
                rcases heq with ⟨rfl, htl⟩
                simpa [execStmts] using hmA tl sfx aid cond htl
      | barrierSlotCheck ptr =>
          simp only [symExecStmts, List.all_append, List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true] at hallSome hallTmem
          obtain ⟨hmR, hmP, hmE, hmA, hmT⟩ := ih regs sregs phase ps hmatch hps hallSome.2 hallTmem
          simp only [execStmts, symExecStmts]
          refine ⟨hmR, hmP, ?_, ?_, hmT⟩
          · intro ev hev
            rw [List.mem_append] at hev
            rcases hev with hhead | htail
            · obtain ⟨idxS, k, hrest⟩ := symSinglePtrEvent_sound hpc hps (hmatch ptr) hallSome.1 hhead
              exact ⟨_, List.mem_append_left _ (List.Mem.head _), idxS, k, hrest⟩
            · obtain ⟨sev, hsev, idxS, k, hrest⟩ := hmE ev htail
              exact ⟨sev, List.mem_append_right _ hsev, idxS, k, hrest⟩
          · intro pfx sfx aid cond heq
            cases pfx with
            | nil => simp at heq
            | cons hd tl =>
                simp only [List.cons_append, List.cons.injEq] at heq
                rcases heq with ⟨rfl, htl⟩
                simpa [execStmts] using hmA tl sfx aid cond htl
      | assertCond aid0 cond0 =>
          simp only [symExecStmts] at hallSome hallTmem
          obtain ⟨hmR, hmP, hmE, hmA, hmT⟩ := ih regs sregs phase ps hmatch hps hallSome hallTmem
          simp only [execStmts, symExecStmts]
          refine ⟨hmR, hmP, hmE, ?_, hmT⟩
          intro pfx sfx aid cond heq
          cases pfx with
          | nil =>
              simp only [List.nil_append, List.cons.injEq, Stmt.assertCond.injEq] at heq
              rcases heq with ⟨⟨ha, hc⟩, _⟩
              subst ha hc
              refine ⟨_, List.Mem.head _, rfl, rfl, ?_⟩
              intro bSym hcOpt
              have hm := hmatch cond0
              cases hsv : sregs.getVar cond0 with
              | bool b0 =>
                  rw [hsv] at hcOpt hm
                  dsimp only at hcOpt
                  injection hcOpt with hb; subst hb
                  simpa [execStmts, SymVal.matchesVal] using hm
              | _ =>
                  rw [hsv] at hcOpt; contradiction
          | cons hd tl =>
              simp only [List.cons_append, List.cons.injEq] at heq
              rcases heq with ⟨rfl, htl⟩
              obtain ⟨aev, haev, ha, hp, hc⟩ := hmA tl sfx aid cond htl
              exact ⟨aev, List.Mem.tail _ haev, ha, hp, by simpa [execStmts] using hc⟩
      | clusterRankCheck _ =>
          simp only [execStmts, symExecStmts] at hallSome hallTmem ⊢
          obtain ⟨hmR, hmP, hmE, hmA, hmT⟩ := ih regs sregs phase ps hmatch hps hallSome hallTmem
          refine ⟨hmR, hmP, hmE, ?_, hmT⟩
          intro pfx sfx aid cond heq
          cases pfx with
          | nil => simp at heq
          | cons hd tl =>
              simp only [List.cons_append, List.cons.injEq] at heq
              rcases heq with ⟨rfl, htl⟩
              simpa [execStmts] using hmA tl sfx aid cond htl
      | mbarArrive _ _ =>
          simp only [execStmts, symExecStmts] at hallSome hallTmem ⊢
          obtain ⟨hmR, hmP, hmE, hmA, hmT⟩ := ih regs sregs phase ps hmatch hps hallSome hallTmem
          refine ⟨hmR, hmP, hmE, ?_, hmT⟩
          intro pfx sfx aid cond heq
          cases pfx with
          | nil => simp at heq
          | cons hd tl =>
              simp only [List.cons_append, List.cons.injEq] at heq
              rcases heq with ⟨rfl, htl⟩
              simpa [execStmts] using hmA tl sfx aid cond htl
      | mbarArriveExpectTx _ _ _ =>
          simp only [execStmts, symExecStmts] at hallSome hallTmem ⊢
          obtain ⟨hmR, hmP, hmE, hmA, hmT⟩ := ih regs sregs phase ps hmatch hps hallSome hallTmem
          refine ⟨hmR, hmP, hmE, ?_, hmT⟩
          intro pfx sfx aid cond heq
          cases pfx with
          | nil => simp at heq
          | cons hd tl =>
              simp only [List.cons_append, List.cons.injEq] at heq
              rcases heq with ⟨rfl, htl⟩
              simpa [execStmts] using hmA tl sfx aid cond htl
      | syncThreads =>
          simp only [execStmts, symExecStmts] at hallSome hallTmem ⊢
          obtain ⟨hmR, hmP, hmE, hmA, hmT⟩ := ih regs sregs (phase + 1) ps.stepSync hmatch (phaseSpec_stepSync_sound hps) hallSome hallTmem
          refine ⟨hmR, hmP, hmE, ?_, hmT⟩
          intro pfx sfx aid cond heq
          cases pfx with
          | nil => simp at heq
          | cons hd tl =>
              simp only [List.cons_append, List.cons.injEq] at heq
              rcases heq with ⟨rfl, htl⟩
              simpa [execStmts] using hmA tl sfx aid cond htl
      | tmaCopyG2S ptr barId slotVar spanElems =>
          simp only [symExecStmts, List.all_append, Bool.and_eq_true] at hallSome hallTmem
          obtain ⟨hmR, hmP, hmE, hmA, hmT⟩ := ih regs sregs phase ps hmatch hps hallSome.2 hallTmem
          simp only [execStmts, symExecStmts]
          refine ⟨hmR, hmP, ?_, ?_, hmT⟩
          · intro ev hev
            rw [List.mem_append] at hev
            rcases hev with hhead | htail
            · have hp := hmatch ptr
              have hs := hmatch slotVar
              cases hsp : sregs.getVar ptr <;> cases hss : sregs.getVar slotVar
              all_goals try (rw [hsp, hss] at hallSome; simp at hallSome)
              rename_i sp bid off sSym
              rw [hsp, SymVal.matchesVal] at hp; rw [hss, SymVal.matchesVal] at hs
              rw [hp, hs] at hhead
              dsimp only at hhead
              obtain ⟨k, hkRange, hkEq⟩ := List.mem_map.mp hhead
              subst hkEq
              exact ⟨{ isWrite := true, space := sp, bufId := bid, indexOpt := some off,
                       span := max 1 spanElems, elemDiv := 1, syncBar := some (barId, sSym),
                       pathCond := pCond, phaseSpec := ps },
                     List.mem_append_left _ (List.Mem.head _), off, k,
                     List.mem_range.mp hkRange, Nat.zero_lt_one, rfl, rfl, rfl, rfl, rfl,
                     by simp, hpc, hps⟩
            · obtain ⟨sev, hsev, idxS, k, hrest⟩ := hmE ev htail
              exact ⟨sev, List.mem_append_right _ hsev, idxS, k, hrest⟩
          · intro pfx sfx aid cond heq
            cases pfx with
            | nil => simp at heq
            | cons hd tl =>
                simp only [List.cons_append, List.cons.injEq] at heq
                rcases heq with ⟨rfl, htl⟩
                simpa [execStmts] using hmA tl sfx aid cond htl
      | tmemAlloc ptr _cols =>
          simp only [symExecStmts, List.all_append, List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true] at hallSome hallTmem
          obtain ⟨hmR, hmP, hmE, hmA, hmT⟩ := ih regs sregs phase ps hmatch hps hallSome.2 hallTmem
          simp only [execStmts, symExecStmts]
          refine ⟨hmR, hmP, ?_, ?_, hmT⟩
          · intro ev hev
            rw [List.mem_append] at hev
            rcases hev with hhead | htail
            · by_cases hl0 : (t.tx % 32 == 0) = true
              · rw [if_pos hl0] at hhead
                have hlane0 : (SymBool.and pCond (.cmp .eq (.binOp .rem 0 .tidX (.const 32)) (.const 0))).eval env t havoc = true := by
                  simp only [SymBool.eval, SymInt.eval, evalCmpOp, evalBinOp, Bool.and_eq_true]
                  exact ⟨hpc, decide_eq_true (beq_iff_eq.mp hl0)⟩
                obtain ⟨idxS, k, hrest⟩ := symSinglePtrEvent_sound hlane0 hps (hmatch ptr) hallSome.1 hhead
                exact ⟨_, List.mem_append_left _ (List.Mem.head _), idxS, k, hrest⟩
              · rw [if_neg hl0] at hhead; cases hhead
            · obtain ⟨sev, hsev, idxS, k, hrest⟩ := hmE ev htail
              exact ⟨sev, List.mem_append_right _ hsev, idxS, k, hrest⟩
          · intro pfx sfx aid cond heq
            cases pfx with
            | nil => simp at heq
            | cons hd tl =>
                simp only [List.cons_append, List.cons.injEq] at heq
                rcases heq with ⟨rfl, htl⟩
                simpa [execStmts] using hmA tl sfx aid cond htl
      | smemDescRead descVar tileBytes =>
          simp only [symExecStmts, List.all_append, Bool.and_eq_true] at hallSome hallTmem
          obtain ⟨hmR, hmP, hmE, hmA, hmT⟩ := ih regs sregs phase ps hmatch hps hallSome.2 hallTmem
          simp only [execStmts, symExecStmts]
          refine ⟨hmR, hmP, ?_, ?_, hmT⟩
          · intro ev hev
            rw [List.mem_append] at hev
            rcases hev with hhead | htail
            · have hd := hmatch descVar
              cases hsd : sregs.getVar descVar with
              | smemDesc sid byteOff =>
                  rw [hsd, SymVal.matchesVal] at hd; rw [hd] at hhead
                  dsimp only at hhead
                  obtain ⟨k, hkRange, hkEq⟩ := List.mem_map.mp hhead
                  subst hkEq
                  let eb := max 1 (f.sharedDecls.getD sid { size := 0 }).elemBytes
                  have hebPos : 0 < eb := by dsimp only [eb]; omega
                  exact ⟨{ isWrite := false, space := .shared, bufId := sid, indexOpt := some byteOff,
                           span := max 1 tileBytes, elemDiv := eb, pathCond := pCond, phaseSpec := ps },
                         List.mem_append_left _ (List.Mem.head _), byteOff, k,
                         List.mem_range.mp hkRange, hebPos, rfl, rfl, rfl, rfl, rfl,
                         rfl, hpc, hps⟩
              | _ => rw [hsd] at hallSome; simp at hallSome
            · obtain ⟨sev, hsev, idxS, k, hrest⟩ := hmE ev htail
              exact ⟨sev, List.mem_append_right _ hsev, idxS, k, hrest⟩
          · intro pfx sfx aid cond heq
            cases pfx with
            | nil => simp at heq
            | cons hd tl =>
                simp only [List.cons_append, List.cons.injEq] at heq
                rcases heq with ⟨rfl, htl⟩
                simpa [execStmts] using hmA tl sfx aid cond htl
      | tmemAccess tmemVar hl wc wr =>
          simp only [symExecStmts, List.all_append, Bool.and_eq_true] at hallSome hallTmem
          obtain ⟨hmR, hmP, hmE, hmA, hmT⟩ := ih regs sregs phase ps hmatch hps hallSome hallTmem.2
          simp only [execStmts, symExecStmts]
          refine ⟨hmR, hmP, hmE, ?_, ?_⟩
          · intro pfx sfx aid cond heq
            cases pfx with
            | nil => simp at heq
            | cons hd tl =>
                simp only [List.cons_append, List.cons.injEq] at heq
                rcases heq with ⟨rfl, htl⟩
                simpa [execStmts] using hmA tl sfx aid cond htl
          · intro tev htev
            rw [List.mem_append] at htev
            rcases htev with hhead | htail
            · have ht := hmatch tmemVar
              cases hst : sregs.getVar tmemVar with
              | tmemPtr cols rel =>
                  rw [hst, SymVal.matchesVal] at ht; rw [ht] at hhead
                  simp only [List.mem_singleton] at hhead; subst hhead
                  exact ⟨{ allocCols := cols, relAddrOpt := some rel, heightLanes := hl,
                           widthCols := wc, warpRestricted := wr, pathCond := pCond },
                         List.mem_append_left _ (List.Mem.head _), rel, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩
              | _ => rw [hst] at hallTmem; simp at hallTmem
            · obtain ⟨stev, hstev, relS, hrest⟩ := hmT tev htail
              exact ⟨stev, List.mem_append_right _ hstev, relS, hrest⟩

/-! ## Section 4: Inductive CFG Invariant Inference & Checker (`validSymFunc`) -/

/-- Symbolic entry invariant for one basic block `b` in `MirFunc`. -/
structure SymBlockInv where
  /-- True when block `b` is reachable in the symbolic invariant table. -/
  reachable : Bool := false
  /-- Outer path condition satisfied on entry to block `b` (disjoint from `b`'s loop args). -/
  pathCond : SymBool := .lit false
  /-- Symbolic register bindings on entry to block `b`. -/
  regs : SymRegList := []
  /-- Barrier-phase invariant on entry to block `b`. -/
  phaseSpec : PhaseSpec := .bot
  /-- Verified inductive `(varId, width, bound)` upper bounds on loop-header `BlockArg`s. -/
  loopBounds : List (Nat × Nat × Nat) := []
  deriving Repr, DecidableEq, Lean.ToExpr

/-- Conjoin two `SymBool` predicates, simplifying `.lit` identities. -/
def SymBool.conj (a b : SymBool) : SymBool :=
  if a == .lit true then b
  else if b == .lit true then a
  else if a == .lit false || b == .lit false then .lit false
  else if a == b then a
  else .and a b

/-- Convert `loopBounds` triples `(vid, width, lim)` into `vid < lim` conjunctions. -/
def loopBoundsCond : List (Nat × Nat × Nat) → SymBool
  | [] => .lit true
  | (vid, w, lim) :: rest =>
      SymBool.conj (.cmp .lt (.havoc vid w) (.const lim)) (loopBoundsCond rest)

/-- Effective block-entry path condition combining `pathCond` with verified `loopBounds`. -/
def SymBlockInv.effCond (inv : SymBlockInv) : SymBool :=
  SymBool.conj inv.pathCond (loopBoundsCond inv.loopBounds)

/-- Helper: `SymBool.conj a b` evaluates to `true` when both `a` and `b` evaluate to `true`. -/
theorem symBool_conj_true {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat}
    {a b : SymBool} (ha : a.eval env t havoc = true) (hb : b.eval env t havoc = true) :
    (SymBool.conj a b).eval env t havoc = true := by
  unfold SymBool.conj
  split
  · exact hb
  · split
    · exact ha
    · split
      · rename_i hf
        rcases Bool.or_eq_true_iff.mp hf with h1 | h2
        · rw [beq_iff_eq.mp h1] at ha; contradiction
        · rw [beq_iff_eq.mp h2] at hb; contradiction
      · split
        · exact ha
        · simp [SymBool.eval, ha, hb]

/-- Logical complement peeling double negation (`.not x -> x`, `y -> .not y`). -/
def SymBool.neg : SymBool → SymBool
  | .lit b => .lit (!b)
  | .not inner => inner
  | other => .not other

/-- Flatten nested `.and` trees into a conjunction list (`true -> []`). -/
def SymBool.toConjList : SymBool → List SymBool
  | .lit true => []
  | .and p q => p.toConjList ++ q.toConjList
  | other => [other]

/-- Rebuild a right-associated `SymBool` conjunction from a list (`[] -> true`). -/
def SymBool.ofConjList : List SymBool → SymBool
  | [] => .lit true
  | [x] => x
  | x :: rest => SymBool.conj x (SymBool.ofConjList rest)

/-- Flatten nested `.or` trees into a disjunction list (`false -> []`). -/
def SymBool.toDisjList : SymBool → List SymBool
  | .lit false => []
  | .or p q => p.toDisjList ++ q.toDisjList
  | other => [other]

/-- Rebuild a right-associated `SymBool` disjunction from a list (`[] -> false`). -/
def SymBool.ofDisjList : List SymBool → SymBool
  | [] => .lit false
  | [x] => x
  | x :: rest => .or x (SymBool.ofDisjList rest)

/-- Strip shared leading conjuncts from two conjunction lists, returning `(common, remA, remB)`. -/
def stripCommonPrefix : List SymBool → List SymBool → (List SymBool × List SymBool × List SymBool)
  | x :: ra, y :: rb =>
      if x == y then
        let (c, r1, r2) := stripCommonPrefix ra rb
        (x :: c, r1, r2)
      else
        ([], x :: ra, y :: rb)
  | ra, rb => ([], ra, rb)

/-- Disjoin two `SymBool` predicates, factoring shared prefix conjuncts `(P ∧ Q1) ∨ (P ∧ Q2) -> P ∧ (Q1 ∨ Q2)`,
    collapsing `k`-way `if/else` diamonds, and widening deep reconverged diamond residuals (`depth > 6`) to
    their dominating conjunctive prefix `P` so `pathCond` stays strictly `O(nesting depth)` on huge kernels. -/
def SymBool.disj (a b : SymBool) : SymBool :=
  if a == .lit false then b
  else if b == .lit false then a
  else if a == .lit true || b == .lit true then .lit true
  else if a == b then a
  else if a == b.neg || b == a.neg then .lit true
  else
    let (common, ra, rb) := stripCommonPrefix a.toConjList b.toConjList
    let pref := SymBool.ofConjList common
    if ra.isEmpty || rb.isEmpty then
      pref
    else
      let da := (SymBool.ofConjList ra).toDisjList
      let db := (SymBool.ofConjList rb).toDisjList
      let rbReduced := rb.filter fun c => !(da.any fun d => c == d.neg || d == c.neg)
      let raReduced := ra.filter fun c => !(db.any fun d => c == d.neg || d == c.neg)
      if rbReduced.isEmpty || raReduced.isEmpty then
        pref
      else
        let remA := SymBool.ofConjList raReduced
        let remB := SymBool.ofConjList rbReduced
        if remA == .lit true || remB == .lit true || remA == remB.neg || remB == remA.neg then
          pref
        else if max (symBoolDepth remA) (symBoolDepth remB) > 8 then
          pref
        else
          SymBool.conj pref (.or remA remB)

/-- Merge two `SymVal` register lists using `iteCond` for 2-branch `.ite`s and `(c1, c2)` for `.condPtr`s. -/
def mergeSymRegLists (iteCond c1 c2 : SymBool) (r1 r2 : SymRegList) : SymRegList :=
  let len := max r1.length r2.length
  let condSmall := iteCond != .lit true && iteCond != .lit false && symBoolDepth iteCond ≤ 32
  (List.range len).map fun v =>
    let v1 := r1.getVar v
    let v2 := r2.getVar v
    if v1 == v2 then v1
    else
      match v1, v2 with
      | .ptr sp bid off, .unk => if c1 == .lit true || c1 == .lit false then .unk else .condPtr c1 sp bid off
      | .unk, .ptr sp bid off => if c2 == .lit true || c2 == .lit false then .unk else .condPtr c2 sp bid off
      | .condPtr g sp bid off, .unk => .condPtr (SymBool.conj c1 g) sp bid off
      | .unk, .condPtr g sp bid off => .condPtr (SymBool.conj c2 g) sp bid off
      | _, _ =>
          if !condSmall then .unk
          else
            match v1, v2 with
            | .int e1, .int e2 =>
                if max (symIntDepth e1) (symIntDepth e2) > 8 then .unk
                else
                  match e2 with
                  | .havoc _ _ =>
                      if iteCond == c1 && c2 != .lit true && c2 != .lit false && symBoolDepth c2 ≤ 32 then
                        .int (.ite c2 e2 e1)
                      else
                        .int (.ite iteCond e1 e2)
                  | _ => .int (.ite iteCond e1 e2)
            | .bool b1, .bool b2 =>
                if max (symBoolDepth b1) (symBoolDepth b2) > 6 then .unk
                else .bool (.ite iteCond b1 b2)
            | .smemOff s1 b1, .smemOff s2 b2 =>
                if s1 == s2 && max (symIntDepth b1) (symIntDepth b2) ≤ 8 then .smemOff s1 (.ite iteCond b1 b2) else .unk
            | .smemDesc s1 b1, .smemDesc s2 b2 =>
                if s1 == s2 && max (symIntDepth b1) (symIntDepth b2) ≤ 8 then .smemDesc s1 (.ite iteCond b1 b2) else .unk
            | .tmemPtr m1 b1, .tmemPtr m2 b2 =>
                if m1 == m2 && max (symIntDepth b1) (symIntDepth b2) ≤ 8 then .tmemPtr m1 (.ite iteCond b1 b2) else .unk
            | _, _ => .unk

/-- Merge two `PhaseSpec` values at an acyclic join block. -/
def mergePhaseSpecs (p1 p2 : PhaseSpec) : PhaseSpec :=
  match p1, p2 with
  | .bot, x | x, .bot => x
  | .exact k1, .exact k2 => if k1 == k2 then .exact k1 else .any
  | .modGe min1 m1 r1, .modGe min2 m2 r2 =>
      if m1 == m2 && r1 == r2 then .modGe (min min1 min2) m1 r1 else .any
  | _, _ => .any

/-- Count `Stmt.syncThreads` barriers along a simple `O(blocks)` cycle starting at loop header `hdr`. -/
def inferLoopBarrierModulus (f : MirFunc) (hdr : Nat) : Nat :=
  let n := f.blocks.length
  let rec dfs : Nat → Nat → List Nat → Nat → Option Nat × List Nat
    | 0, _, seen, _ => (none, seen)
    | fuel + 1, u, seen, accSync =>
        let blk := f.blockAt u
        let blkSyncs := (blk.stmts.filter (· == .syncThreads)).length
        let tot := accSync + blkSyncs
        let succs := match blk.term with
          | .goto d _ => [d]
          | .condBr _ t _ e _ => [t, e]
          | _ => []
        if succs.contains hdr && u ≥ hdr then
          (some tot, seen)
        else
          succs.foldl (fun (found, sn) s =>
            match found with
            | some _ => (found, sn)
            | none =>
                if s > hdr && s < n && !(sn.contains s) then
                  dfs fuel s (s :: sn) tot
                else
                  (none, sn)) (none, seen)
  match (dfs (n + 1) hdr [hdr] 0).1 with
  | some m => m
  | none => 0

/-- Construct initial `SymRegList` for `block 0` from `f.params`. -/
def initialSymRegs (f : MirFunc) : SymRegList :=
  let b0 := f.blockAt 0
  ((List.range b0.args.length).zip b0.args).foldl (fun acc (idx, ba) =>
    match f.params[idx]? with
    | some (.scalar _) => acc.setVar ba.varId (.int (.paramScalar idx))
    | _ => acc.setVar ba.varId .unk) []

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
  | .const c, lim, hb => by
      unfold symIntBoundedUnder at hb; exact of_decide_eq_true hb
  | .ite c tv fv, lim, hb => by
      unfold symIntBoundedUnder at hb
      have stepEqHelper : ∀ {aw bnd : Nat} {x : SymInt},
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
        · rcases Bool.and_eq_true_iff.mp hboth with ⟨htv, hfv⟩
          simp only [SymInt.eval]
          by_cases hce : c.eval env t havoc = true
          · rw [if_pos hce]
            exact symIntBoundedUnder_sound (by simp [SymBool.eval, hhyp, hce]) tv lim htv
          · rw [if_neg hce]
            have hcf : c.eval env t havoc = false := Bool.eq_false_iff.mpr hce
            exact symIntBoundedUnder_sound (by simp [SymBool.eval, hhyp, hcf]) fv lim hfv
        · cases hfvA : asAddOne fv with
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
      · cases htvA : asAddOne tv with
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
  | .binOp op w a b, lim, hb => by
      unfold symIntBoundedUnder at hb
      rcases Bool.or_eq_true_iff.mp hb with h123 | handL
      · clear hb
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
      · clear hb
        rcases Bool.and_eq_true_iff.mp handL with ⟨hop, hmA⟩
        have hop' : op = .bitAnd := beq_iff_eq.mp hop; subst hop'
        clear handL hop
        revert hmA
        cases hc : asConstInt a
        · intro hmA; contradiction
        · intro hmA; rw [asConstInt_eq hc]
          exact Nat.lt_of_le_of_lt Nat.and_le_left (of_decide_eq_true hmA)
  | .tidX, lim, hb | .tidY, lim, hb | .ctaidX, lim, hb | .ctaidY, lim, hb
  | .index1D _, lim, hb | .paramScalar _, lim, hb | .sliceLen _, lim, hb
  | .havoc _ _, lim, hb | .opaqueUint _ _, lim, hb | .intCast _ _, lim, hb => by
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

/-- SSA variables referenced inside a single `Rhs`. -/
def rhsUsedVars : Rhs → List Nat
  | .copy s | .notBool s | .intCast _ s | .load s | .cvtaShared s | .loadTmemBase s _ => [s]
  | .binOp _ _ l r | .cmpOp _ l r | .ptrOffset l r => [l, r]
  | .mbarTryWait _ sv => [sv]
  | _ => []

/-- SSA variables referenced inside a single `Stmt`. -/
def stmtUsedVars : Stmt → List Nat
  | .assign _ rhs => rhsUsedVars rhs
  | .store p v => [p, v]
  | .assertCond _ c => [c]
  | .barrierSlotCheck p | .tmemAlloc p _ | .smemDescRead p _ | .tmemAccess p _ _ _ => [p]
  | .clusterRankCheck r => [r]
  | .mbarArrive _ sv | .mbarArriveExpectTx _ sv _ => [sv]
  | .tmaCopyG2S p _ sv _ => [p, sv]
  | .syncThreads => []

/-- Collect variables read before being overwritten inside `stmts`. -/
def upwardExposedVars : List Nat → List Stmt → List Nat
  | _, [] => []
  | defined, stmt :: rest =>
      let used := (stmtUsedVars stmt).filter fun v => !(defined.contains v)
      let defined' := match stmt with
        | .assign dst _ => if defined.contains dst then defined else dst :: defined
        | _ => defined
      let tail := upwardExposedVars defined' rest
      used.foldl (fun acc v => if acc.contains v then acc else v :: acc) tail

/-- Compute the set of register IDs live across basic-block boundaries
    (any register upward-exposed at block entry or read by a block terminator). -/
def crossBlockUsedVars (f : MirFunc) : List Nat :=
  f.blocks.foldl (fun acc blk =>
    let termVars := match blk.term with
      | .goto _ args => args
      | .condBr cv _ tArgs _ fArgs => cv :: (tArgs ++ fArgs)
      | _ => []
    let exposed := upwardExposedVars (blk.args.map BlockArg.varId) blk.stmts ++ termVars
    exposed.foldl (fun acc2 v => if acc2.contains v then acc2 else v :: acc2) acc) []

/-- Drop intra-block temporary registers (`v ∉ crossLive`) to `.unk` and trim trailing `.unk`s. -/
def pruneIntraBlockRegs (crossLive : List Nat) (regs : SymRegList) : SymRegList :=
  let pruned := ((List.range regs.length).zip regs).map fun (v, sv) =>
    match sv with
    | .unk => .unk
    | _ => if crossLive.contains v then sv else .unk
  pruned.reverse.dropWhile (· == .unk) |>.reverse

/-- Single forward pass of `inferSymTable` parameterized by pre-seeded per-header `loopBounds`. -/
def inferSymTablePass (f : MirFunc) (hdrBounds : Nat → List (Nat × Nat × Nat)) : List SymBlockInv :=
  let n := f.blocks.length
  if n == 0 then []
  else
    let crossLive := crossBlockUsedVars f
    let init0 : SymBlockInv :=
      { reachable := true, pathCond := .lit true, regs := initialSymRegs f, phaseSpec := .exact 0 }
    let initTable := init0 :: List.replicate (n - 1) {}
    (List.range n).foldl (fun tbl b =>
      let inv := tbl.getD b {}
      if !inv.reachable then tbl
      else
        let blk := f.blockAt b
        let effC := inv.effCond
        let (outRegs, outPs, _, _, _) := symExecStmts f effC inv.regs inv.phaseSpec blk.term blk.stmts
        let pushEdge := fun (acc : List SymBlockInv) (edgeCond : SymBool) (dst : Nat) (argVars : List Nat) =>
          if dst ≤ b || dst ≥ n then acc
          else
            let dstBlk := f.blockAt dst
            let curDst := acc.getD dst {}
            let dstKeep := crossLive ++ (dstBlk.args.map BlockArg.varId)
            let boundRegs := pruneIntraBlockRegs dstKeep (bindSymBlockArgs outRegs outRegs dstBlk.args argVars)
            let nextRegs :=
              if dstBlk.isLoopHeader then
                dstBlk.args.foldl (fun r ba =>
                  if ba.width ≤ 1 then r.setVar ba.varId .unk
                  else r.setVar ba.varId (.int (.havoc ba.varId ba.width))) boundRegs
              else
                boundRegs
            let nextPs :=
              if dstBlk.isLoopHeader then
                let m := inferLoopBarrierModulus f dst
                if m == 0 then outPs
                else
                  match outPs with
                  | .exact k => .modGe k m (k % m)
                  | .modGe minP m0 r0 => if m == m0 then .modGe minP m0 r0 else .any
                  | _ => .any
              else
                outPs
            let lbs := if dstBlk.isLoopHeader then hdrBounds dst else []
            let merged : SymBlockInv :=
              if !curDst.reachable then
                { reachable := true, pathCond := edgeCond, regs := nextRegs,
                  phaseSpec := nextPs, loopBounds := lbs }
              else
                let (_, raRel, rbRel) := stripCommonPrefix curDst.pathCond.toConjList edgeCond.toConjList
                let relCond :=
                  if raRel.isEmpty && rbRel.isEmpty then .lit true
                  else
                    let q1 := SymBool.ofConjList raRel
                    let q2 := SymBool.ofConjList rbRel
                    if q1 != .lit true && (q1 == q2.neg || q2 == q1.neg) then q1 else curDst.pathCond
                { reachable := true
                , pathCond := SymBool.disj curDst.pathCond edgeCond
                , regs := mergeSymRegLists relCond curDst.pathCond edgeCond curDst.regs nextRegs
                , phaseSpec := mergePhaseSpecs curDst.phaseSpec nextPs
                , loopBounds := curDst.loopBounds }
            acc.set dst merged
        match blk.term with
        | .goto dst args => pushEdge tbl effC dst args
        | .condBr cv tBlk tArgs fBlk fArgs =>
            let (tCond, fCond) := match outRegs.getVar cv with
              | .bool cSym => (SymBool.conj effC cSym, SymBool.conj effC (.not cSym))
              | _ => (effC, effC)
            let tbl1 := pushEdge tbl tCond tBlk tArgs
            pushEdge tbl1 fCond fBlk fArgs
        | .ret | .unreachable => tbl) initTable

/-- True when `term` has `target` as one of its outgoing successor blocks. -/
def termTargets (term : Terminator) (target : Nat) : Bool :=
  match term with
  | .goto dst _ => dst == target
  | .condBr _ tBlk _ fBlk _ => tBlk == target || fBlk == target
  | .ret | .unreachable => false

/-- Prune `cand` to only those `(hdr, lbs)` whose bounds hold across EVERY incoming edge in `tbl`. -/
def pruneHdrBounds (f : MirFunc) (tbl : List SymBlockInv)
    (cand : List (Nat × List (Nat × Nat × Nat))) : List (Nat × List (Nat × Nat × Nat)) :=
  let n := f.blocks.length
  cand.filterMap fun (hdr, lbs) =>
    let hdrBlk := f.blockAt hdr
    let inBlocks := (List.range n).filter fun b =>
      (tbl.getD b {}).reachable && termTargets (f.blockAt b).term hdr
    let lbsOk := lbs.filter fun (vid, _, lim) =>
      inBlocks.all fun b =>
        let inv := tbl.getD b {}
        let blk := f.blockAt b
        let effC := inv.effCond
        let (outRegs, _, _, _, _) := symExecStmts f effC inv.regs inv.phaseSpec blk.term blk.stmts
        let checkOne := fun ec dst args =>
          if dst != hdr then true
          else
            let pr := bindSymBlockArgs outRegs outRegs hdrBlk.args args
            match pr.getVar vid with
            | .int e => symIntBoundedUnder ec e lim
            | _ => false
        match blk.term with
        | .goto dst args => checkOne effC dst args
        | .condBr cv tBlk tArgs fBlk fArgs =>
            let (tC, fC) := match outRegs.getVar cv with
              | .bool cSym => (SymBool.conj effC cSym, SymBool.conj effC (.not cSym))
              | _ => (effC, effC)
            checkOne tC tBlk tArgs && checkOne fC fBlk fArgs
        | _ => true
    if lbsOk.isEmpty then none else some (hdr, lbsOk)

/-- Infer the `List SymBlockInv` invariant table for `f` (including nested inductive ring bounds
    discovered via Houdini greatest-fixpoint pruning and independently certified by `validSymFunc`). -/
def inferSymTable (f : MirFunc) : List SymBlockInv :=
  let baseTbl := inferSymTablePass f (fun _ => [])
  let n := f.blocks.length
  let hdrs := (List.range n).filter fun hdr =>
    (f.blockAt hdr).isLoopHeader && (baseTbl.getD hdr {}).reachable
  if hdrs.isEmpty then baseTbl
  else
    -- 1. Collect all wrap/mod bounds (`suggestRingBound`) from backedges across all loop headers:
    let perHdrBackEdges : List (Nat × List (Nat × SymRegList × List Nat)) :=
      hdrs.map fun hdr =>
        let edges := (List.range n).flatMap fun b =>
          let inv := baseTbl.getD b {}
          if !inv.reachable || !termTargets (f.blockAt b).term hdr then []
          else
            let blk := f.blockAt b
            let (outRegs, _, _, _, _) := symExecStmts f inv.effCond inv.regs inv.phaseSpec blk.term blk.stmts
            match blk.term with
            | .goto dst args => if dst == hdr then [(b, outRegs, args)] else []
            | .condBr _ tBlk tArgs fBlk fArgs =>
                (if tBlk == hdr then [(b, outRegs, tArgs)] else []) ++
                (if fBlk == hdr then [(b, outRegs, fArgs)] else [])
            | _ => []
        (hdr, edges)
    let globalLims : List Nat :=
      perHdrBackEdges.foldl (fun acc (hdr, edges) =>
        let hdrBlk := f.blockAt hdr
        hdrBlk.args.foldl (fun acc2 fa =>
          edges.foldl (fun acc3 (b, outRegs, args) =>
            if b < hdr then acc3
            else
              let pr := bindSymBlockArgs outRegs outRegs hdrBlk.args args
              match pr.getVar fa.varId with
              | .int e =>
                  match suggestRingBound e with
                  | some lim => if acc3.contains lim then acc3 else lim :: acc3
                  | none => acc3
              | _ => acc3) acc2) acc) []
    -- 2. Seed optimistic candidate bounds (direct wrap bound OR outer loop arg initialized `< lim`):
    let seedCand : List (Nat × List (Nat × Nat × Nat)) :=
      perHdrBackEdges.filterMap fun (hdr, edges) =>
        let hdrBlk := f.blockAt hdr
        let lbs := hdrBlk.args.filterMap fun fa =>
          if fa.width ≤ 1 then none
          else
            let directLim := edges.findSome? fun (b, outRegs, args) =>
              if b < hdr then none
              else
                let pr := bindSymBlockArgs outRegs outRegs hdrBlk.args args
                match pr.getVar fa.varId with
                | .int e => suggestRingBound e
                | _ => none
            match directLim with
            | some lim => some (fa.varId, fa.width, lim)
            | none =>
                -- Check if forward entry edges pass a constant `< lim` for a `lim ∈ globalLims`:
                globalLims.findSome? fun lim =>
                  let fwdOk := edges.all fun (b, outRegs, args) =>
                    if b ≥ hdr then true
                    else
                      let pr := bindSymBlockArgs outRegs outRegs hdrBlk.args args
                      match pr.getVar fa.varId with
                      | .int (.const c) => decide (c < lim)
                      | _ => false
                  if fwdOk then some (fa.varId, fa.width, lim) else none
        if lbs.isEmpty then none else some (hdr, lbs)
    if seedCand.isEmpty then baseTbl
    else
      let rec houdini (fuel : Nat) (cand : List (Nat × List (Nat × Nat × Nat))) : List SymBlockInv :=
        let tbl := inferSymTablePass f fun hdr =>
          match cand.find? (fun p => p.1 == hdr) with
          | some (_, lbs) => lbs
          | none => []
        match fuel with
        | 0 => inferSymTablePass f (fun _ => [])
        | fuel' + 1 =>
            let pruned := pruneHdrBounds f tbl cand
            if pruned == cand then tbl
            else if pruned.isEmpty then baseTbl
            else houdini fuel' pruned
      houdini 5 seedCand

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

/-- Check pairwise distinctness of formal `BlockArg.varId`s in `args`. -/
def distinctBlockArgIds : List BlockArg → Bool
  | [] => true
  | fa :: rest => !(rest.map BlockArg.varId).contains fa.varId && distinctBlockArgIds rest

/-- Linear `O(vars)` register-list subsumption checker avoiding quadratic `getVar` indexing in `whnf`. -/
def checkRegsEquiv (edgeCond : SymBool) : List SymVal → List SymVal → Bool
  | [], _ => true
  | d :: ds, p :: ps =>
      (match d with | .unk => true | _ => symValEquivUnder edgeCond d p) &&
      checkRegsEquiv edgeCond ds ps
  | d :: ds, [] =>
      (match d with | .unk => true | _ => symValEquivUnder edgeCond d .unk) &&
      checkRegsEquiv edgeCond ds []

/-- Linear `O(vars)` loop-header outer-register subsumption checker (`idx` tracks current `varId`). -/
def checkLoopRegsEquiv (hdrArgIds : List Nat) (edgeCond : SymBool) :
    Nat → List SymVal → List SymVal → Bool
  | _, [], _ => true
  | idx, d :: ds, p :: ps =>
      let ok :=
        if hdrArgIds.contains idx then true
        else match d with
          | .unk => true
          | _ => d.disjointHavoc hdrArgIds && symValEquivUnder edgeCond d p
      ok && checkLoopRegsEquiv hdrArgIds edgeCond (idx + 1) ds ps
  | idx, d :: ds, [] =>
      let ok :=
        if hdrArgIds.contains idx then true
        else match d with
          | .unk => true
          | _ => d.disjointHavoc hdrArgIds && symValEquivUnder edgeCond d .unk
      ok && checkLoopRegsEquiv hdrArgIds edgeCond (idx + 1) ds []

/-- Soundness of `checkRegsEquiv`: implies `symValEquivUnder` at every `v : Nat`. -/
theorem checkRegsEquiv_getVar {edgeCond : SymBool} :
    ∀ (ds ps : List SymVal) (v : Nat),
      checkRegsEquiv edgeCond ds ps = true →
      symValEquivUnder edgeCond (ds.getD v .unk) (ps.getD v .unk) = true
  | [], _, _, _ => by simp [List.getD, symValEquivUnder]
  | d :: _, p :: _, 0, h => by
      simp only [checkRegsEquiv, Bool.and_eq_true] at h
      cases d <;> first | exact h.1 | simp [List.getD, symValEquivUnder]
  | _ :: ds, _ :: ps, v + 1, h => by
      simp only [checkRegsEquiv, Bool.and_eq_true] at h
      exact checkRegsEquiv_getVar ds ps v h.2
  | d :: _, [], 0, h => by
      simp only [checkRegsEquiv, Bool.and_eq_true] at h
      cases d <;> first | exact h.1 | simp [List.getD, symValEquivUnder]
  | _ :: ds, [], v + 1, h => by
      simp only [checkRegsEquiv, Bool.and_eq_true] at h
      simpa [List.getD] using checkRegsEquiv_getVar ds [] v h.2

/-- Soundness of `checkLoopRegsEquiv`: implies `disjointHavoc` and `symValEquivUnder` outside `hdrArgIds`. -/
theorem checkLoopRegsEquiv_getVar {hdrArgIds : List Nat} {edgeCond : SymBool} :
    ∀ (idx : Nat) (ds ps : List SymVal) (k : Nat),
      checkLoopRegsEquiv hdrArgIds edgeCond idx ds ps = true →
      hdrArgIds.contains (idx + k) = false →
      (ds.getD k .unk).disjointHavoc hdrArgIds = true ∧
      symValEquivUnder edgeCond (ds.getD k .unk) (ps.getD k .unk) = true
  | _, [], _, _, _, _ => by simp [List.getD, SymVal.disjointHavoc, symValEquivUnder]
  | idx, d :: _, p :: _, 0, h, hnc => by
      simp only [Nat.add_zero] at hnc
      simp only [checkLoopRegsEquiv, hnc, Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true] at h
      cases d <;> first | exact Bool.and_eq_true_iff.mp h.1 | simp [List.getD, SymVal.disjointHavoc, symValEquivUnder]
  | idx, _ :: ds, _ :: ps, k + 1, h, hnc => by
      simp only [checkLoopRegsEquiv, Bool.and_eq_true] at h
      have hnc' : hdrArgIds.contains ((idx + 1) + k) = false := by
        simpa [Nat.add_assoc, Nat.add_comm 1 k] using hnc
      exact checkLoopRegsEquiv_getVar (idx + 1) ds ps k h.2 hnc'
  | idx, d :: _, [], 0, h, hnc => by
      simp only [Nat.add_zero] at hnc
      simp only [checkLoopRegsEquiv, hnc, Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true] at h
      cases d <;> first | exact Bool.and_eq_true_iff.mp h.1 | simp [List.getD, SymVal.disjointHavoc, symValEquivUnder]
  | idx, _ :: ds, [], k + 1, h, hnc => by
      simp only [checkLoopRegsEquiv, Bool.and_eq_true] at h
      have hnc' : hdrArgIds.contains ((idx + 1) + k) = false := by
        simpa [Nat.add_assoc, Nat.add_comm 1 k] using hnc
      simpa [List.getD] using checkLoopRegsEquiv_getVar (idx + 1) ds [] k h.2 hnc'

/-- Validate a single CFG edge `(b -> dst)` passing `argVars` under `edgeCond` and `(outRegs, outPs)`. -/
def validSymEdge (f : MirFunc) (tbl : List SymBlockInv)
    (edgeCond : SymBool) (outRegs : SymRegList) (outPs : PhaseSpec)
    (dst : Nat) (argVars : List Nat) : Bool :=
  decide (dst < f.blocks.length) &&
  let dstBlk := f.blockAt dst
  let dstInv := tbl.getD dst {}
  decide (argVars.length == dstBlk.args.length) &&
  distinctBlockArgIds dstBlk.args &&
  dstInv.reachable &&
  dstInv.phaseSpec.subsumes outPs &&
  let passedRegs := bindSymBlockArgs outRegs outRegs dstBlk.args argVars
  let maxVars := max dstInv.regs.length passedRegs.length
  if !dstBlk.isLoopHeader then
    decide (dstInv.loopBounds == []) &&
    symImpliesVal edgeCond dstInv.pathCond true &&
    checkRegsEquiv edgeCond dstInv.regs passedRegs
  else
    let hdrArgIds := dstBlk.args.map BlockArg.varId
    (hdrArgIds.all fun vid => decide (vid < maxVars)) &&
    dstInv.pathCond.disjointHavoc hdrArgIds &&
    symImpliesVal edgeCond dstInv.pathCond true &&
    (dstInv.loopBounds.all fun (vid, w, lim) =>
      hdrArgIds.contains vid &&
      decide (0 < w) &&
      (dstInv.regs.getVar vid == .int (.havoc vid w)) &&
      match passedRegs.getVar vid with
      | .int e => symIntFitsWidth f w e && symIntBoundedUnder edgeCond e lim
      | _ => false) &&
    (dstBlk.args.all fun fa =>
      if fa.width == 0 then
        dstInv.regs.getVar fa.varId == .unk
      else
        dstInv.regs.getVar fa.varId == .int (.havoc fa.varId fa.width) &&
        match passedRegs.getVar fa.varId with
        | .int e => symIntFitsWidth f fa.width e
        | _ => false) &&
    checkLoopRegsEquiv hdrArgIds edgeCond 0 dstInv.regs passedRegs

/-- Computable certificate checker verifying that `tbl` is a sound inductive symbolic
    invariant for every block and CFG edge of `f` (`O(blocks * vars)`). -/
def validSymFunc (f : MirFunc) (tbl : List SymBlockInv) : Bool :=
  decide (0 < f.blocks.length) &&
  decide (tbl.length == f.blocks.length) &&
  let inv0 := tbl.getD 0 {}
  let initSR := initialSymRegs f
  inv0.reachable &&
  decide (inv0.pathCond == .lit true) &&
  decide (inv0.loopBounds == []) &&
  decide (inv0.phaseSpec == .exact 0) &&
  decide (inv0.regs == initSR) &&
  (List.range f.blocks.length).all fun b =>
    let inv := tbl.getD b {}
    if !inv.reachable then true
    else
      let blk := f.blockAt b
      let effC := inv.effCond
      let (outRegs, outPs, mevs, aevs, tevs) := symExecStmts f effC inv.regs inv.phaseSpec blk.term blk.stmts
      mevs.all (fun mev => mev.indexOpt.isSome) &&
      aevs.all (fun aev => aev.condOpt.isSome) &&
      tevs.all (fun tev => tev.relAddrOpt.isSome) &&
      match blk.term with
      | .ret | .unreachable => true
      | .goto dst args =>
          validSymEdge f tbl effC outRegs outPs dst args
      | .condBr cv tBlk tArgs fBlk fArgs =>
          match outRegs.getVar cv with
          | .bool cSym =>
              validSymEdge f tbl (SymBool.conj effC cSym) outRegs outPs tBlk tArgs &&
              validSymEdge f tbl (SymBool.conj effC (.not cSym)) outRegs outPs fBlk fArgs
          | .unk =>
              validSymEdge f tbl effC outRegs outPs tBlk tArgs &&
              validSymEdge f tbl effC outRegs outPs fBlk fArgs
          | _ => false

/-- Extract the `Nat` payload of a concrete `Val` (`0` otherwise). -/
def Val.asInt : Val → Nat
  | .int n => n
  | _ => 0

/-- Lemma: reading `getVar v` past `sregs.length` returns `.unk`. -/
theorem getVar_ge_length (sregs : SymRegList) {v : Nat} (hge : sregs.length ≤ v) :
    sregs.getVar v = .unk := by
  simp [SymRegList.getVar, List.getD, List.getElem?_eq_none hge]

/-- Correspondence of `initialRegs` and `initialSymRegs` at kernel entry (`block 0`). -/
theorem initialSymRegs_correct (f : MirFunc) (env : InputEnv) (t : ThreadCtx) (havoc : Nat → Nat) :
    ∀ v, ((initialSymRegs f).getVar v).matchesVal env t havoc (initialRegs f env v) := by
  unfold initialRegs initialSymRegs
  dsimp only
  have hbase : ∀ (pairs : List (Nat × BlockArg)) (r : RegFile) (sr : SymRegList),
      (∀ v, (sr.getVar v).matchesVal env t havoc (r v)) →
      ∀ v, ((pairs.foldl (fun acc (idx, ba) =>
              match f.params[idx]? with
              | some (.scalar _) => acc.setVar ba.varId (.int (.paramScalar idx))
              | _ => acc.setVar ba.varId .unk) sr).getVar v).matchesVal env t havoc
            ((pairs.foldl (fun acc (idx, ba) =>
              match f.params[idx]? with
              | some (.scalar _) => acc.set ba.varId (.int (env.scalar idx))
              | _ => acc.set ba.varId .unk) r) v) := by
    intro pairs
    induction pairs with
    | nil => intro r sr hm v; exact hm v
    | cons hd tl ih =>
        intro r sr hm v
        rcases hd with ⟨idx, ba⟩
        dsimp only [List.foldl_cons]
        apply ih
        intro k
        cases f.params[idx]? with
        | none =>
            unfold RegFile.set; rw [getVar_setVar]
            by_cases hk : (k == ba.varId) = true
            · simp [hk, SymVal.matchesVal]
            · simp [hk]; exact hm k
        | some pk =>
            cases pk <;> (
              unfold RegFile.set; rw [getVar_setVar]
              by_cases hk : (k == ba.varId) = true
              · simp [hk, SymVal.matchesVal, SymInt.eval]
              · simp [hk]; exact hm k)
  apply hbase
  intro v; simp [SymRegList.getVar, List.getD, SymVal.matchesVal]

/-- Helper: prove `(loopBoundsCond lbs).eval env t havoc' = true` from per-triple `havoc'` bounds. -/
theorem loopBoundsCond_eval_true {env : InputEnv} {t : ThreadCtx} {havoc' : Nat → Nat} :
    ∀ lbs : List (Nat × Nat × Nat),
      (∀ p ∈ lbs, wordMod p.2.1 (havoc' p.1) < p.2.2) →
      (loopBoundsCond lbs).eval env t havoc' = true
  | [], _ => rfl
  | (vid, w, lim) :: rest, hall => by
      dsimp only [loopBoundsCond]
      apply symBool_conj_true
      · have hhd := hall (vid, w, lim) (List.Mem.head _)
        simp only [SymBool.eval, SymInt.eval, evalCmpOp]
        exact decide_eq_true hhd
      · exact loopBoundsCond_eval_true rest (fun p hp => hall p (List.Mem.tail _ hp))

/-- Edge preservation lemma: if `validSymEdge` holds on `(b -> dst)`, concrete `(outRegs, outPhase)`
    transitions to `dstInv` under an updated `havoc'`. -/
theorem validSymEdge_sound {f : MirFunc} {tbl : List SymBlockInv} {env : InputEnv} {t : ThreadCtx}
    (hvalidIn : f.validInput env)
    {edgeCond : SymBool} {outRegs : RegFile} {outSR : SymRegList}
    {outPhase : Nat} {outPs : PhaseSpec} {havoc : Nat → Nat}
    {dst : Nat} {argVars : List Nat}
    (hedge : validSymEdge f tbl edgeCond outSR outPs dst argVars = true)
    (hecond : edgeCond.eval env t havoc = true)
    (hmatch : ∀ v, (outSR.getVar v).matchesVal env t havoc (outRegs v))
    (hps : outPs.matchesPhase outPhase = true) :
    (tbl.getD dst {}).reachable = true ∧
    ∃ havoc' : Nat → Nat,
      (tbl.getD dst {}).effCond.eval env t havoc' = true ∧
      (tbl.getD dst {}).phaseSpec.matchesPhase outPhase = true ∧
      ∀ v, ((tbl.getD dst {}).regs.getVar v).matchesVal env t havoc'
             ((bindBlockArgs outRegs outRegs (f.blockAt dst).args argVars) v) := by
  unfold validSymEdge at hedge
  dsimp only at hedge
  have hpassed : ∀ v, ((bindSymBlockArgs outSR outSR (f.blockAt dst).args argVars).getVar v).matchesVal env t havoc
      ((bindBlockArgs outRegs outRegs (f.blockAt dst).args argVars) v) :=
    bindSymBlockArgs_correct hmatch (f.blockAt dst).args argVars outRegs outSR hmatch
  by_cases hhdr : (f.blockAt dst).isLoopHeader = false    -- Split: ordinary forward join vs. loop header
  · rw [hhdr] at hedge                                    -- Case 1: Acyclic destination (`havoc' := havoc`)
    simp only [Bool.not_false, ↓reduceIte, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hedge
    obtain ⟨_, ⟨⟨⟨_, _⟩, hreach⟩, hsub⟩, ⟨hlbsEmpty, hpcond⟩, hallV⟩ := hedge
    refine ⟨hreach, havoc, ?_, phaseSpec_subsumes_sound hsub hps, ?_⟩
    · unfold SymBlockInv.effCond
      rw [hlbsEmpty]
      exact symBool_conj_true (symImpliesVal_sound hecond _ true hpcond) rfl
    · intro v
      let passedSR := bindSymBlockArgs outSR outSR (f.blockAt dst).args argVars
      let dstInv := tbl.getD dst {}
      have heqv := checkRegsEquiv_getVar dstInv.regs passedSR v hallV
      exact symValEquivUnder_sound hecond _ _ _ heqv (hpassed v)
  · have hhdrT : (f.blockAt dst).isLoopHeader = true := by -- Case 2: Loop header (`rebind havoc' on hdrArgIds`)
      cases hhb : (f.blockAt dst).isLoopHeader
      · exact False.elim (hhdr hhb)
      · rfl
    rw [hhdrT] at hedge
    simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true, decide_eq_true_eq] at hedge
    obtain ⟨_, ⟨⟨⟨_, _⟩, hreach⟩, hsub⟩, ⟨⟨⟨⟨_, hpcDisj⟩, hpcond⟩, hlbsOk⟩, hargsOk⟩, hallV⟩ := hedge
    let dstBlk := f.blockAt dst
    let dstInv := tbl.getD dst {}
    let passedSR := bindSymBlockArgs outSR outSR dstBlk.args argVars
    let hdrArgIds := dstBlk.args.map BlockArg.varId
    -- Update `havoc'` at `hdrArgIds` to the concrete runtime integers passed along this edge:
    let havoc' : Nat → Nat := fun k =>
      if hdrArgIds.contains k then
        Val.asInt ((bindBlockArgs outRegs outRegs dstBlk.args argVars) k)
      else havoc k
    have hagree : ∀ k, k ∉ hdrArgIds → havoc' k = havoc k := by
      intro k hk
      have hc : hdrArgIds.contains k = false := by
        cases hck : hdrArgIds.contains k
        · rfl
        · exact False.elim (hk (List.contains_iff_mem.mp hck))
      dsimp only [havoc']; rw [hc]; simp
    refine ⟨hreach, havoc', ?_, phaseSpec_subsumes_sound hsub hps, ?_⟩
    · apply symBool_conj_true
      · rw [symBool_eval_congr hagree dstInv.pathCond hpcDisj]
        exact symImpliesVal_sound hecond dstInv.pathCond true hpcond
      · apply loopBoundsCond_eval_true
        intro ⟨vid, w, lim⟩ hmemLb
        have hlb := List.all_eq_true.mp hlbsOk (vid, w, lim) hmemLb
        simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hlb
        rcases hlb with ⟨⟨⟨hcontains, _⟩, _⟩, hpassLb⟩
        cases hpsv : passedSR.getVar vid <;> rw [hpsv] at hpassLb
        all_goals try contradiction
        rename_i eSym
        dsimp only at hpassLb
        rcases Bool.and_eq_true_iff.mp hpassLb with ⟨hfitsW, hbndE⟩
        have hbVal := hpassed vid
        rw [hpsv, SymVal.matchesVal] at hbVal
        dsimp only [havoc', dstBlk]
        rw [hcontains, if_pos rfl, hbVal]
        dsimp only [Val.asInt]
        rw [symIntFitsWidth_sound hvalidIn hfitsW]
        exact symIntBoundedUnder_sound hecond eSym lim hbndE
    · intro v
      by_cases hmem : hdrArgIds.contains v = true         -- Split: loop-carried `BlockArg` vs. outer invariant var
      · have hvMem : v ∈ hdrArgIds := List.contains_iff_mem.mp hmem
        obtain ⟨fa, hfaMem, hfaId⟩ := List.mem_map.mp hvMem
        have hpOk := List.all_eq_true.mp hargsOk fa hfaMem
        subst hfaId
        by_cases hw0 : (fa.width == 0) = true
        · rw [hw0] at hpOk; simp only [↓reduceIte, beq_iff_eq] at hpOk
          rw [hpOk]; trivial
        · rw [if_neg hw0] at hpOk; simp only [Bool.and_eq_true, beq_iff_eq] at hpOk
          rcases hpOk with ⟨hdstH, hfitsM⟩
          rw [hdstH]
          cases hpsv : passedSR.getVar fa.varId <;> rw [hpsv] at hfitsM
          all_goals try contradiction
          rename_i eSym
          have hbVal := hpassed fa.varId
          rw [hpsv, SymVal.matchesVal] at hbVal
          dsimp only at hfitsM
          dsimp only [SymVal.matchesVal, SymInt.eval, havoc', dstBlk]
          rw [hmem, if_pos rfl, hbVal]
          dsimp only [Val.asInt]
          rw [symIntFitsWidth_sound hvalidIn hfitsM]
      · have hmf : hdrArgIds.contains (0 + v) = false := by
          cases hc : hdrArgIds.contains v <;> simp_all
        have hvOk := checkLoopRegsEquiv_getVar 0 dstInv.regs passedSR v hallV hmf
        have hmBase := symValEquivUnder_sound hecond _ _ _ hvOk.2 (hpassed v)
        exact symVal_matches_congr hagree _ _ hvOk.1 hmBase

/-- Core inductive soundness theorem: every reachable concrete state `(b, regs, phase)`
    of `f` satisfies the validated symbolic invariant `tbl.getD b {}` for some `havoc`. -/
theorem symReachable_sound {f : MirFunc} {tbl : List SymBlockInv}
    (hvalid : validSymFunc f tbl = true)
    {env : InputEnv} {t : ThreadCtx} (hvalidIn : f.validInput env)
    {b phase : Nat} {regs : RegFile}
    (hreach : Reachable f env t b regs phase) :
    b < f.blocks.length ∧
    (tbl.getD b {}).reachable = true ∧
    ∃ havoc : Nat → Nat,
      (tbl.getD b {}).effCond.eval env t havoc = true ∧
      (tbl.getD b {}).phaseSpec.matchesPhase phase = true ∧
      ∀ v, ((tbl.getD b {}).regs.getVar v).matchesVal env t havoc (regs v) := by
  have hvf := hvalid
  unfold validSymFunc at hvf
  dsimp only at hvf
  simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hvf
  obtain ⟨⟨hpos, _⟩, ⟨⟨⟨⟨h0r, h0c⟩, h0lbs⟩, h0p⟩, h0regs⟩, hallB⟩ := hvf
  induction hreach with                                   -- Induction on `Reachable` execution steps
  | entry _ =>
      refine ⟨hpos, h0r, fun _ => 0, ?_, ?_, ?_⟩
      · unfold SymBlockInv.effCond; rw [h0c, h0lbs]; rfl
      · rw [h0p]; rfl
      · intro v; rw [h0regs]; exact initialSymRegs_correct f env t (fun _ => 0) v
  | @stepGoto b0 dst p0 outP r0 outR evs tevs args _ hexec hterm hdst ih =>
      obtain ⟨hb0, hr0, havoc, hpc0, hps0, hm0⟩ := ih
      have hbOk := List.all_eq_true.mp hallB b0 (List.mem_range.mpr hb0)
      rw [hr0] at hbOk; simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true] at hbOk
      obtain ⟨hmOut, hpsOut, _, _, _⟩ := symExecStmts_correct f env t havoc (tbl.getD b0 {}).effCond hpc0 (f.blockAt b0).term
        (f.blockAt b0).stmts r0 (tbl.getD b0 {}).regs p0 (tbl.getD b0 {}).phaseSpec hm0 hps0 hbOk.1.1.1 hbOk.1.2
      rw [hexec] at hmOut hpsOut
      rw [hterm] at hbOk hmOut hpsOut
      obtain ⟨hrDst, havoc', hpcDst, hpsDst, hmDst⟩ :=
        validSymEdge_sound hvalidIn hbOk.2 hpc0 hmOut hpsOut
      exact ⟨hdst, hrDst, havoc', hpcDst, hpsDst, hmDst⟩
  | @stepCondTrue b0 tBlk fBlk cond p0 outP r0 outR evs tevs tArgs fArgs _ hexec hterm hcond hdst ih =>
      obtain ⟨hb0, hr0, havoc, hpc0, hps0, hm0⟩ := ih
      have hbOk := List.all_eq_true.mp hallB b0 (List.mem_range.mpr hb0)
      rw [hr0] at hbOk; simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true] at hbOk
      obtain ⟨hmOut, hpsOut, _, _, _⟩ := symExecStmts_correct f env t havoc (tbl.getD b0 {}).effCond hpc0 (f.blockAt b0).term
        (f.blockAt b0).stmts r0 (tbl.getD b0 {}).regs p0 (tbl.getD b0 {}).phaseSpec hm0 hps0 hbOk.1.1.1 hbOk.1.2
      rw [hexec] at hmOut hpsOut
      rw [hterm] at hbOk hmOut hpsOut
      rcases hbOk with ⟨_, hbrOk⟩
      dsimp only at hbrOk
      have hcv := hmOut cond; dsimp only at hcv
      cases hget : (symExecStmts f (tbl.getD b0 {}).effCond (tbl.getD b0 {}).regs (tbl.getD b0 {}).phaseSpec (.condBr cond tBlk tArgs fBlk fArgs) (f.blockAt b0).stmts).1.getVar cond with
      | bool cSym =>
          rw [hget] at hbrOk hcv
          rw [hcond, SymVal.matchesVal] at hcv
          injection hcv with hcTrue
          simp only [Bool.and_eq_true] at hbrOk
          have hec := symBool_conj_true hpc0 hcTrue.symm
          obtain ⟨hrDst, havoc', hpcDst, hpsDst, hmDst⟩ :=
            validSymEdge_sound hvalidIn hbrOk.1 hec hmOut hpsOut
          exact ⟨hdst, hrDst, havoc', hpcDst, hpsDst, hmDst⟩
      | unk =>
          rw [hget] at hbrOk
          simp only [Bool.and_eq_true] at hbrOk
          obtain ⟨hrDst, havoc', hpcDst, hpsDst, hmDst⟩ :=
            validSymEdge_sound hvalidIn hbrOk.1 hpc0 hmOut hpsOut
          exact ⟨hdst, hrDst, havoc', hpcDst, hpsDst, hmDst⟩
      | int _ | ptr _ _ _ | condPtr _ _ _ _ | smemOff _ _ | smemDesc _ _ | tmemPtr _ _ =>
          rw [hget] at hbrOk; contradiction
  | @stepCondFalse b0 tBlk fBlk cond p0 outP r0 outR evs tevs tArgs fArgs _ hexec hterm hcond hdst ih =>
      obtain ⟨hb0, hr0, havoc, hpc0, hps0, hm0⟩ := ih
      have hbOk := List.all_eq_true.mp hallB b0 (List.mem_range.mpr hb0)
      rw [hr0] at hbOk; simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true] at hbOk
      obtain ⟨hmOut, hpsOut, _, _, _⟩ := symExecStmts_correct f env t havoc (tbl.getD b0 {}).effCond hpc0 (f.blockAt b0).term
        (f.blockAt b0).stmts r0 (tbl.getD b0 {}).regs p0 (tbl.getD b0 {}).phaseSpec hm0 hps0 hbOk.1.1.1 hbOk.1.2
      rw [hexec] at hmOut hpsOut
      rw [hterm] at hbOk hmOut hpsOut
      rcases hbOk with ⟨_, hbrOk⟩
      dsimp only at hbrOk
      have hcv := hmOut cond; dsimp only at hcv
      cases hget : (symExecStmts f (tbl.getD b0 {}).effCond (tbl.getD b0 {}).regs (tbl.getD b0 {}).phaseSpec (.condBr cond tBlk tArgs fBlk fArgs) (f.blockAt b0).stmts).1.getVar cond with
      | bool cSym =>
          rw [hget] at hbrOk hcv
          rw [hcond, SymVal.matchesVal] at hcv
          injection hcv with hcFalse
          simp only [Bool.and_eq_true] at hbrOk
          have hnotTrue : (SymBool.not cSym).eval env t havoc = true := by
            simp [SymBool.eval, hcFalse.symm]
          have hec := symBool_conj_true hpc0 hnotTrue
          obtain ⟨hrDst, havoc', hpcDst, hpsDst, hmDst⟩ :=
            validSymEdge_sound hvalidIn hbrOk.2 hec hmOut hpsOut
          exact ⟨hdst, hrDst, havoc', hpcDst, hpsDst, hmDst⟩
      | unk =>
          rw [hget] at hbrOk
          simp only [Bool.and_eq_true] at hbrOk
          obtain ⟨hrDst, havoc', hpcDst, hpsDst, hmDst⟩ :=
            validSymEdge_sound hvalidIn hbrOk.2 hpc0 hmOut hpsOut
          exact ⟨hdst, hrDst, havoc', hpcDst, hpsDst, hmDst⟩
      | int _ | ptr _ _ _ | condPtr _ _ _ _ | smemOff _ _ | smemDesc _ _ | tmemPtr _ _ =>
          rw [hget] at hbrOk; contradiction

/-- Collect all `SymMemEvent`s emitted across reachable blocks in `tbl`. -/
def collectSymMemEvents (f : MirFunc) (tbl : List SymBlockInv) : List SymMemEvent :=
  (List.range f.blocks.length).flatMap fun b =>
    let inv := tbl.getD b {}
    if !inv.reachable then []
    else (symExecStmts f inv.effCond inv.regs inv.phaseSpec (f.blockAt b).term (f.blockAt b).stmts).2.2.1

/-- Collect all `SymAssertEvent`s emitted across reachable blocks in `tbl`. -/
def collectSymAssertEvents (f : MirFunc) (tbl : List SymBlockInv) : List SymAssertEvent :=
  (List.range f.blocks.length).flatMap fun b =>
    let inv := tbl.getD b {}
    if !inv.reachable then []
    else (symExecStmts f inv.effCond inv.regs inv.phaseSpec (f.blockAt b).term (f.blockAt b).stmts).2.2.2.1

/-- Collect all `SymTmemEvent`s emitted across reachable blocks in `tbl`. -/
def collectSymTmemEvents (f : MirFunc) (tbl : List SymBlockInv) : List SymTmemEvent :=
  (List.range f.blocks.length).flatMap fun b =>
    let inv := tbl.getD b {}
    if !inv.reachable then []
    else (symExecStmts f inv.effCond inv.regs inv.phaseSpec (f.blockAt b).term (f.blockAt b).stmts).2.2.2.2

/-- Lemma: every concrete `EmitsEvent` corresponds to a `SymMemEvent` in `collectSymMemEvents`. -/
theorem emitsEvent_mem_collect {f : MirFunc} {tbl : List SymBlockInv}
    (hvalid : validSymFunc f tbl = true)
    {env : InputEnv} {t : ThreadCtx} (hvalidIn : f.validInput env)
    {ev : MemEvent} (hemits : EmitsEvent f env t ev) :
    ∃ havoc : Nat → Nat, ∃ sev ∈ collectSymMemEvents f tbl, ∃ idxSym : SymInt, ∃ k : Nat,
      k < sev.span ∧ 0 < sev.elemDiv ∧
      sev.isWrite = ev.isWrite ∧ sev.space = ev.space ∧ sev.bufId = ev.bufId ∧
      evalSyncBar env t havoc sev.syncBar = ev.syncBar ∧ sev.indexOpt = some idxSym ∧
      (idxSym.eval env t havoc + k) / sev.elemDiv = ev.index ∧
      sev.pathCond.eval env t havoc = true ∧ sev.phaseSpec.matchesPhase ev.phase = true := by
  obtain ⟨b, phase, regs, hreach, hevMem⟩ := hemits
  obtain ⟨hb, hr, havoc, hpc, hps, hm⟩ := symReachable_sound hvalid hvalidIn hreach
  have hvf := hvalid
  unfold validSymFunc at hvf; dsimp only at hvf
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hvf
  obtain ⟨_, _, hallB⟩ := hvf
  have hbOk := List.all_eq_true.mp hallB b (List.mem_range.mpr hb)
  rw [hr] at hbOk; simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true] at hbOk
  obtain ⟨_, _, hmE, _, _⟩ := symExecStmts_correct f env t havoc (tbl.getD b {}).effCond hpc (f.blockAt b).term
    (f.blockAt b).stmts regs (tbl.getD b {}).regs phase (tbl.getD b {}).phaseSpec hm hps hbOk.1.1.1 hbOk.1.2
  obtain ⟨sev, hsev, idxS, k, hks, hdiv, hw, hsp, hbid, hab, hidxOpt, hidxEv, hpce, hpse⟩ := hmE ev hevMem
  refine ⟨havoc, sev, ?_, idxS, k, hks, hdiv, hw, hsp, hbid, hab, hidxOpt, hidxEv, hpce, hpse⟩
  unfold collectSymMemEvents
  rw [List.mem_flatMap]
  refine ⟨b, List.mem_range.mpr hb, ?_⟩
  dsimp only; rw [hr]; simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte]; exact hsev

/-- Lemma: every concrete `EmitsTmemEvent` corresponds to a `SymTmemEvent` in `collectSymTmemEvents`. -/
theorem emitsTmemEvent_mem_collect {f : MirFunc} {tbl : List SymBlockInv}
    (hvalid : validSymFunc f tbl = true)
    {env : InputEnv} {t : ThreadCtx} (hvalidIn : f.validInput env)
    {tev : TmemEvent} (hemits : EmitsTmemEvent f env t tev) :
    ∃ havoc : Nat → Nat, ∃ stev ∈ collectSymTmemEvents f tbl, ∃ relSym : SymInt,
      stev.allocCols = tev.allocCols ∧ stev.heightLanes = tev.heightLanes ∧
      stev.widthCols = tev.widthCols ∧ stev.warpRestricted = tev.warpRestricted ∧
      stev.relAddrOpt = some relSym ∧ relSym.eval env t havoc = tev.relAddr ∧
      stev.pathCond.eval env t havoc = true := by
  obtain ⟨b, phase, regs, hreach, htevMem⟩ := hemits
  obtain ⟨hb, hr, havoc, hpc, hps, hm⟩ := symReachable_sound hvalid hvalidIn hreach
  have hvf := hvalid
  unfold validSymFunc at hvf; dsimp only at hvf
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hvf
  obtain ⟨_, _, hallB⟩ := hvf
  have hbOk := List.all_eq_true.mp hallB b (List.mem_range.mpr hb)
  rw [hr] at hbOk; simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true] at hbOk
  obtain ⟨_, _, _, _, hmT⟩ := symExecStmts_correct f env t havoc (tbl.getD b {}).effCond hpc (f.blockAt b).term
    (f.blockAt b).stmts regs (tbl.getD b {}).regs phase (tbl.getD b {}).phaseSpec hm hps hbOk.1.1.1 hbOk.1.2
  obtain ⟨stev, hstev, relS, hac, hhl, hwc, hwr, hro, hre, hpce⟩ := hmT tev htevMem
  refine ⟨havoc, stev, ?_, relS, hac, hhl, hwc, hwr, hro, hre, ?_⟩
  · unfold collectSymTmemEvents
    rw [List.mem_flatMap]
    refine ⟨b, List.mem_range.mpr hb, ?_⟩
    dsimp only; rw [hr]; simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte]; exact hstev
  · rw [hpce]; exact hpc

/-! ## Section 5: Verification Conditions, `MetaM` Precomputation & `mir_vc` -/

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
  split at hin
  · rename_i h11
    rcases Bool.and_eq_true_iff.mp h11 with ⟨hs1, hd1⟩
    have hs1' : sev.span = 1 := beq_iff_eq.mp hs1
    have hd1' : sev.elemDiv = 1 := beq_iff_eq.mp hd1
    have hk0 : k = 0 := by omega
    rw [hk0, hd1', Nat.add_zero, Nat.div_one]; exact hin
  · have hlt : base + k < bound * sev.elemDiv := by omega
    exact (Nat.div_lt_iff_lt_mul hdiv).mpr hlt

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

/-- Pure arithmetic Verification Condition (`Prop`) for `MirNoOobGlobal f paramId`. -/
def vcNoOobGlobal (f : MirFunc) (paramId : Nat) : Prop :=
  let sevs := (collectSymMemEvents f (inferSymTable f)).filter fun s =>
    s.space == .global && s.bufId == paramId
  ∀ env : InputEnv, ∀ t : ThreadCtx, ∀ havoc : Nat → Nat,
    f.validInput env → f.validThread t →
    sevs.foldr (fun sev acc =>
      (sev.pathCond.eval env t havoc = true →
       symSpanInBounds sev ((sev.indexOpt.getD (.const 0)).eval env t havoc) (env.sliceLen paramId)) ∧ acc) True

/-- Soundness of `vcNoOobGlobal` -> `MirNoOobGlobal`. -/
theorem vcNoOobGlobal_sound {f : MirFunc} {paramId : Nat}
    (hvalid : validSymFunc f (inferSymTable f) = true)
    (hvc : vcNoOobGlobal f paramId) :
    MirNoOobGlobal f paramId := by
  intro env t ev hIn hTh hemits hsp hbid
  obtain ⟨havoc, sev, hsev, idxS, k, hks, hdiv, _, hssp, hsbid, _, hopt, hidxEv, hpc, _⟩ :=
    emitsEvent_mem_collect hvalid hIn hemits
  have hfMem : sev ∈ (collectSymMemEvents f (inferSymTable f)).filter
      (fun s => s.space == .global && s.bufId == paramId) := by
    rw [List.mem_filter]; exact ⟨hsev, by simp [hssp, hsp, hsbid, hbid]⟩
  have hp := foldr_and_of_mem _ (hvc env t havoc hIn hTh) sev hfMem hpc
  rw [hopt] at hp; dsimp only [Option.getD] at hp
  rw [← hidxEv]; exact symSpanInBounds_sound hks hdiv hp

/-- Pure arithmetic Verification Condition (`Prop`) for `MirNoOobShared f sharedId`. -/
def vcNoOobShared (f : MirFunc) (sharedId : Nat) : Prop :=
  let cap := (f.sharedDecls.getD sharedId { size := 0 }).size
  let sevs := (collectSymMemEvents f (inferSymTable f)).filter fun s =>
    s.space == .shared && s.bufId == sharedId
  ∀ env : InputEnv, ∀ t : ThreadCtx, ∀ havoc : Nat → Nat,
    f.validInput env → f.validThread t →
    sevs.foldr (fun sev acc =>
      (sev.pathCond.eval env t havoc = true →
       symSpanInBounds sev ((sev.indexOpt.getD (.const 0)).eval env t havoc) cap) ∧ acc) True

/-- Soundness of `vcNoOobShared` -> `MirNoOobShared`. -/
theorem vcNoOobShared_sound {f : MirFunc} {sharedId : Nat}
    (hvalid : validSymFunc f (inferSymTable f) = true)
    (hvc : vcNoOobShared f sharedId) :
    MirNoOobShared f sharedId := by
  intro env t ev hIn hTh hemits hsp hbid
  obtain ⟨havoc, sev, hsev, idxS, k, hks, hdiv, _, hssp, hsbid, _, hopt, hidxEv, hpc, _⟩ :=
    emitsEvent_mem_collect hvalid hIn hemits
  have hfMem : sev ∈ (collectSymMemEvents f (inferSymTable f)).filter
      (fun s => s.space == .shared && s.bufId == sharedId) := by
    rw [List.mem_filter]; exact ⟨hsev, by simp [hssp, hsp, hsbid, hbid]⟩
  have hp := foldr_and_of_mem _ (hvc env t havoc hIn hTh) sev hfMem hpc
  rw [hopt] at hp; dsimp only [Option.getD] at hp
  rw [← hidxEv]; exact symSpanInBounds_sound hks hdiv hp

/-- Pure arithmetic Verification Condition (`Prop`) for `MirDisjointWrites f paramId`. -/
def vcDisjointWrites (f : MirFunc) (paramId : Nat) : Prop :=
  let sevs := (collectSymMemEvents f (inferSymTable f)).filter fun s =>
    s.space == .global && s.bufId == paramId
  let pairs := (sevs.flatMap fun s1 => sevs.map fun s2 => (s1, s2)).filter fun p =>
    p.1.isWrite || p.2.isWrite
  ∀ env : InputEnv, ∀ t₁ t₂ : ThreadCtx, ∀ h₁ h₂ : Nat → Nat,
    f.validInput env → f.validThread t₁ → f.validThread t₂ →
    pairs.foldr (fun p acc =>
      (p.1.pathCond.eval env t₁ h₁ = true →
       p.2.pathCond.eval env t₂ h₂ = true →
       symPairRaceOblig p.1 p.2
         ((p.1.indexOpt.getD (.const 0)).eval env t₁ h₁)
         ((p.2.indexOpt.getD (.const 0)).eval env t₂ h₂)
         (t₁.bx = t₂.bx ∧ t₁.by_ = t₂.by_ ∧ t₁.tx = t₂.tx ∧ t₁.ty = t₂.ty)) ∧ acc) True

/-- Soundness of `vcDisjointWrites` -> `MirDisjointWrites`. -/
theorem vcDisjointWrites_sound {f : MirFunc} {paramId : Nat}
    (hvalid : validSymFunc f (inferSymTable f) = true)
    (hvc : vcDisjointWrites f paramId) :
    MirDisjointWrites f paramId := by
  intro env t₁ t₂ ev₁ ev₂ hIn hT1 hT2 hem1 hem2 hsp1 hsp2 hb1 hb2 hw hidx
  obtain ⟨h₁, s1, hs1, i1, k1, hk1, hd1, hw1, hssp1, hsbid1, _, ho1, hi1, hpc1, _⟩ :=
    emitsEvent_mem_collect hvalid hIn hem1
  obtain ⟨h₂, s2, hs2, i2, k2, hk2, hd2, hw2, hssp2, hsbid2, _, ho2, hi2, hpc2, _⟩ :=
    emitsEvent_mem_collect hvalid hIn hem2
  have hm1 : s1 ∈ (collectSymMemEvents f (inferSymTable f)).filter
      (fun s => s.space == .global && s.bufId == paramId) := by
    rw [List.mem_filter]; exact ⟨hs1, by simp [hssp1, hsp1, hsbid1, hb1]⟩
  have hm2 : s2 ∈ (collectSymMemEvents f (inferSymTable f)).filter
      (fun s => s.space == .global && s.bufId == paramId) := by
    rw [List.mem_filter]; exact ⟨hs2, by simp [hssp2, hsp2, hsbid2, hb2]⟩
  have hpair : (s1, s2) ∈ ((collectSymMemEvents f (inferSymTable f)).filter
      (fun s => s.space == .global && s.bufId == paramId) |>.flatMap fun a =>
      (collectSymMemEvents f (inferSymTable f)).filter
      (fun s => s.space == .global && s.bufId == paramId) |>.map fun b => (a, b)).filter
      (fun p => p.1.isWrite || p.2.isWrite) := by
    rw [List.mem_filter]
    refine ⟨List.mem_flatMap.mpr ⟨s1, hm1, List.mem_map.mpr ⟨s2, hm2, rfl⟩⟩, ?_⟩
    dsimp only; rw [hw1, hw2]
    rcases hw with h1 | h2
    · simp [h1]
    · simp [h2]
  have hp := foldr_and_of_mem _ (hvc env t₁ t₂ h₁ h₂ hIn hT1 hT2) (s1, s2) hpair hpc1 hpc2
  rw [ho1, ho2] at hp; dsimp only [Option.getD] at hp
  have heqIdx : (i1.eval env t₁ h₁ + k1) / s1.elemDiv = (i2.eval env t₂ h₂ + k2) / s2.elemDiv := by
    rw [hi1, hi2, hidx]
  obtain ⟨hbx, hby, htx, hty⟩ := symPairRaceOblig_sound hk1 hd1 hk2 hd2 hp heqIdx
  cases t₁; cases t₂; simp_all

/-- Pure arithmetic Verification Condition (`Prop`) for `MirNoSharedRace f sharedId`
    (pruning barrier-disjoint pairs via `PhaseSpec.disjoint` and `symAsyncOrdered`). -/
def vcNoSharedRace (f : MirFunc) (sharedId : Nat) : Prop :=
  let sevs := (collectSymMemEvents f (inferSymTable f)).filter fun s =>
    s.space == .shared && s.bufId == sharedId
  let pairs := (sevs.flatMap fun s1 => sevs.map fun s2 => (s1, s2)).filter fun p =>
    (p.1.isWrite || p.2.isWrite) &&
    !(p.1.phaseSpec.disjoint p.2.phaseSpec) &&
    !(symAsyncOrdered f p.1 p.2)
  ∀ env : InputEnv, ∀ t₁ t₂ : ThreadCtx, ∀ h₁ h₂ : Nat → Nat,
    f.validInput env → f.validThread t₁ → f.validThread t₂ →
    t₁.bx = t₂.bx → t₁.by_ = t₂.by_ →
    pairs.foldr (fun p acc =>
      (p.1.pathCond.eval env t₁ h₁ = true →
       p.2.pathCond.eval env t₂ h₂ = true →
       symAsyncSlotGuard f p.1 p.2 env t₁ t₂ h₁ h₂
         (symPairRaceOblig p.1 p.2
           ((p.1.indexOpt.getD (.const 0)).eval env t₁ h₁)
           ((p.2.indexOpt.getD (.const 0)).eval env t₂ h₂)
           (t₁.tx = t₂.tx ∧ t₁.ty = t₂.ty))) ∧ acc) True

/-- Soundness of `vcNoSharedRace` -> `MirNoSharedRace`. -/
theorem vcNoSharedRace_sound {f : MirFunc} {sharedId : Nat}
    (hvalid : validSymFunc f (inferSymTable f) = true)
    (hvc : vcNoSharedRace f sharedId) :
    MirNoSharedRace f sharedId := by
  intro env t₁ t₂ ev₁ ev₂ hIn hT1 hT2 hbx hby hem1 hem2 hsp1 hsp2 hb1 hb2 hph hnotOrd hw hidx
  obtain ⟨h₁, s1, hs1, i1, k1, hk1, hd1, hw1, hssp1, hsbid1, hab1, ho1, hi1, hpc1, hps1⟩ :=
    emitsEvent_mem_collect hvalid hIn hem1
  obtain ⟨h₂, s2, hs2, i2, k2, hk2, hd2, hw2, hssp2, hsbid2, hab2, ho2, hi2, hpc2, hps2⟩ :=
    emitsEvent_mem_collect hvalid hIn hem2
  by_cases hdisj : s1.phaseSpec.disjoint s2.phaseSpec = true
  · exact False.elim (phaseSpec_disjoint_sound hdisj hps1 hps2 hph)
  · by_cases haord : symAsyncOrdered f s1 s2 = true
    · exact False.elim (hnotOrd (symAsyncOrdered_sound haord hab1 hab2 hpc1 hpc2))
    · have hm1 : s1 ∈ (collectSymMemEvents f (inferSymTable f)).filter
          (fun s => s.space == .shared && s.bufId == sharedId) := by
        rw [List.mem_filter]; exact ⟨hs1, by simp [hssp1, hsp1, hsbid1, hb1]⟩
      have hm2 : s2 ∈ (collectSymMemEvents f (inferSymTable f)).filter
          (fun s => s.space == .shared && s.bufId == sharedId) := by
        rw [List.mem_filter]; exact ⟨hs2, by simp [hssp2, hsp2, hsbid2, hb2]⟩
      have hpair : (s1, s2) ∈ ((collectSymMemEvents f (inferSymTable f)).filter
          (fun s => s.space == .shared && s.bufId == sharedId) |>.flatMap fun a =>
          (collectSymMemEvents f (inferSymTable f)).filter
          (fun s => s.space == .shared && s.bufId == sharedId) |>.map fun b => (a, b)).filter
          (fun p => (p.1.isWrite || p.2.isWrite) && !(p.1.phaseSpec.disjoint p.2.phaseSpec) && !(symAsyncOrdered f p.1 p.2)) := by
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
      have hp0 := foldr_and_of_mem _ (hvc env t₁ t₂ h₁ h₂ hIn hT1 hT2 hbx hby) (s1, s2) hpair hpc1 hpc2
      have hp := symAsyncSlotGuard_sound hab1 hab2 hpc1 hpc2 hnotOrd hp0
      rw [ho1, ho2] at hp; dsimp only [Option.getD] at hp
      have heqIdx : (i1.eval env t₁ h₁ + k1) / s1.elemDiv = (i2.eval env t₂ h₂ + k2) / s2.elemDiv := by
        rw [hi1, hi2, hidx]
      obtain ⟨htx, hty⟩ := symPairRaceOblig_sound hk1 hd1 hk2 hd2 hp heqIdx
      cases t₁; cases t₂; simp_all

/-- Pure arithmetic Verification Condition (`Prop`) for `MirAssertRedundant f assertId`. -/
def vcAssertRedundant (f : MirFunc) (assertId : Nat) : Prop :=
  let aevs := (collectSymAssertEvents f (inferSymTable f)).filter (·.assertId == assertId)
  ∀ env : InputEnv, ∀ t : ThreadCtx, ∀ havoc : Nat → Nat,
    f.validInput env → f.validThread t →
    aevs.foldr (fun aev acc =>
      (aev.pathCond.eval env t havoc = true →
       (aev.condOpt.getD (.lit false)).eval env t havoc = true) ∧ acc) True

/-- Soundness of `vcAssertRedundant` -> `MirAssertRedundant`. -/
theorem vcAssertRedundant_sound {f : MirFunc} {assertId : Nat}
    (hvalid : validSymFunc f (inferSymTable f) = true)
    (hvc : vcAssertRedundant f assertId) :
    MirAssertRedundant f assertId := by
  let tbl := inferSymTable f
  intro env t b phase regs hIn hTh hreach pfx sfx cond hstmts
  obtain ⟨hb, hr, havoc, hpc, hps, hm⟩ := symReachable_sound hvalid hIn hreach
  have hvf := hvalid
  unfold validSymFunc at hvf
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hvf
  obtain ⟨_, _, hallB⟩ := hvf
  have hbOk := List.all_eq_true.mp hallB b (List.mem_range.mpr hb)
  rw [hr] at hbOk; simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true] at hbOk
  obtain ⟨_, _, _, hmA, _⟩ := symExecStmts_correct f env t havoc (tbl.getD b {}).effCond hpc (f.blockAt b).term
    (f.blockAt b).stmts regs (tbl.getD b {}).regs phase (tbl.getD b {}).phaseSpec hm hps hbOk.1.1.1 hbOk.1.2
  obtain ⟨aev, haevMem, haid, hpce, hcondEval⟩ := hmA pfx sfx assertId cond hstmts
  have haevOk := List.all_eq_true.mp hbOk.1.1.2 aev haevMem
  cases hcOpt : aev.condOpt with
  | none => rw [hcOpt] at haevOk; contradiction
  | some bSym =>
      have hfMem : aev ∈ (collectSymAssertEvents f tbl).filter (·.assertId == assertId) := by
        rw [List.mem_filter]
        refine ⟨?_, beq_iff_eq.mpr haid⟩
        unfold collectSymAssertEvents
        rw [List.mem_flatMap]
        refine ⟨b, List.mem_range.mpr hb, ?_⟩
        dsimp only; rw [hr]; simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte]; exact haevMem
      have hp := foldr_and_of_mem _ (hvc env t havoc hIn hTh) aev hfMem (by rw [hpce]; exact hpc)
      rw [hcOpt] at hp; dsimp only [Option.getD] at hp
      rw [hcondEval bSym hcOpt, hp]

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
  intro env t ev hIn hTh hemits hsp hbid
  obtain ⟨havoc, sev, hsev, idxS, k, hks, hdiv, _, hssp, hsbid, _, hopt, hidxEv, hpc, _⟩ :=
    emitsEvent_mem_collect hvalid hIn hemits              -- Map concrete `EmitsEvent` to its symbolic `sev`
  have hfMem : sev ∈ sevs := by
    rw [← hevs, ← hmem, List.mem_filter]; exact ⟨hsev, by simp [hssp, hsp, hsbid, hbid]⟩
  have hp := foldr_and_of_mem _ (foldr_holds_to_eval sevs (hvc env t havoc hIn hTh)) sev hfMem hpc
  rw [hopt] at hp; dsimp only [Option.getD] at hp
  rw [← hidxEv]; exact symSpanInBounds_sound hks hdiv hp  -- Transfer symbolic `< sliceLen` back to `ev.index`

/-- Soundness of `vcNoOobShared` from pre-computed `memEvs = collectSymMemEvents f tbl`. -/
theorem vcNoOobShared_of_table {f : MirFunc} {tbl : List SymBlockInv}
    {memEvs sevs : List SymMemEvent} {sharedId cap : Nat}
    (hvalid : validSymFunc f tbl = true)
    (hmem : collectSymMemEvents f tbl = memEvs)
    (hcap : (f.sharedDecls.getD sharedId { size := 0 }).size = cap)
    (hevs : (memEvs.filter fun s => s.space == .shared && s.bufId == sharedId) = sevs)
    (hvc : ∀ env : InputEnv, ∀ t : ThreadCtx, ∀ havoc : Nat → Nat,
      f.validInput env → f.validThread t →
      sevs.foldr (fun sev acc =>
        (sev.pathCond.holds env t havoc →
         symSpanInBounds sev ((sev.indexOpt.getD (.const 0)).eval env t havoc) cap) ∧ acc) True) :
    MirNoOobShared f sharedId := by
  intro env t ev hIn hTh hemits hsp hbid
  obtain ⟨havoc, sev, hsev, idxS, k, hks, hdiv, _, hssp, hsbid, _, hopt, hidxEv, hpc, _⟩ :=
    emitsEvent_mem_collect hvalid hIn hemits
  have hfMem : sev ∈ sevs := by
    rw [← hevs, ← hmem, List.mem_filter]; exact ⟨hsev, by simp [hssp, hsp, hsbid, hbid]⟩
  have hp := foldr_and_of_mem _ (foldr_holds_to_eval sevs (hvc env t havoc hIn hTh)) sev hfMem hpc
  rw [hopt] at hp; dsimp only [Option.getD] at hp
  rw [← hidxEv, hcap]; exact symSpanInBounds_sound hks hdiv hp

/-- Soundness of `vcDisjointWrites` from pre-computed `memEvs = collectSymMemEvents f tbl`. -/
theorem vcDisjointWrites_of_table {f : MirFunc} {tbl : List SymBlockInv}
    {memEvs : List SymMemEvent} {paramId : Nat}
    {pairs : List (SymMemEvent × SymMemEvent)}
    (hvalid : validSymFunc f tbl = true)
    (hmem : collectSymMemEvents f tbl = memEvs)
    (hpairs : (let sevs := memEvs.filter fun s => s.space == .global && s.bufId == paramId
               (sevs.flatMap fun s1 => sevs.map fun s2 => (s1, s2)).filter fun p =>
                 p.1.isWrite || p.2.isWrite) = pairs)
    (hvc : ∀ env : InputEnv, ∀ t₁ t₂ : ThreadCtx, ∀ h₁ h₂ : Nat → Nat,
      f.validInput env → f.validThread t₁ → f.validThread t₂ →
      pairs.foldr (fun p acc =>
        (p.1.pathCond.holds env t₁ h₁ →
         p.2.pathCond.holds env t₂ h₂ →
         symPairRaceOblig p.1 p.2
           ((p.1.indexOpt.getD (.const 0)).eval env t₁ h₁)
           ((p.2.indexOpt.getD (.const 0)).eval env t₂ h₂)
           (t₁.bx = t₂.bx ∧ t₁.by_ = t₂.by_ ∧ t₁.tx = t₂.tx ∧ t₁.ty = t₂.ty)) ∧ acc) True) :
    MirDisjointWrites f paramId := by
  intro env t₁ t₂ ev₁ ev₂ hIn hT1 hT2 hem1 hem2 hsp1 hsp2 hb1 hb2 hw hidx
  obtain ⟨h₁, s1, hs1, i1, k1, hk1, hd1, hw1, hssp1, hsbid1, _, ho1, hi1, hpc1, _⟩ :=
    emitsEvent_mem_collect hvalid hIn hem1
  obtain ⟨h₂, s2, hs2, i2, k2, hk2, hd2, hw2, hssp2, hsbid2, _, ho2, hi2, hpc2, _⟩ :=
    emitsEvent_mem_collect hvalid hIn hem2
  rw [hmem] at hs1 hs2
  have hm1 : s1 ∈ memEvs.filter (fun s => s.space == .global && s.bufId == paramId) := by
    rw [List.mem_filter]; exact ⟨hs1, by simp [hssp1, hsp1, hsbid1, hb1]⟩
  have hm2 : s2 ∈ memEvs.filter (fun s => s.space == .global && s.bufId == paramId) := by
    rw [List.mem_filter]; exact ⟨hs2, by simp [hssp2, hsp2, hsbid2, hb2]⟩
  have hpair : (s1, s2) ∈ pairs := by
    rw [← hpairs, List.mem_filter]
    refine ⟨List.mem_flatMap.mpr ⟨s1, hm1, List.mem_map.mpr ⟨s2, hm2, rfl⟩⟩, ?_⟩
    dsimp only; rw [hw1, hw2]; rcases hw with h1 | h2
    · simp [h1]
    · simp [h2]
  have hp := foldr_and_of_mem _ (foldr_holds2_to_eval pairs (hvc env t₁ t₂ h₁ h₂ hIn hT1 hT2)) (s1, s2) hpair hpc1 hpc2
  rw [ho1, ho2] at hp; dsimp only [Option.getD] at hp
  have heqIdx : (i1.eval env t₁ h₁ + k1) / s1.elemDiv = (i2.eval env t₂ h₂ + k2) / s2.elemDiv := by
    rw [hi1, hi2, hidx]
  obtain ⟨hbx, hby, htx, hty⟩ := symPairRaceOblig_sound hk1 hd1 hk2 hd2 hp heqIdx
  cases t₁; cases t₂; simp_all

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
    (hvc : ∀ env : InputEnv, ∀ t₁ t₂ : ThreadCtx, ∀ h₁ h₂ : Nat → Nat,
      f.validInput env → f.validThread t₁ → f.validThread t₂ →
      t₁.bx = t₂.bx → t₁.by_ = t₂.by_ →
      pairs.foldr (fun p acc =>
        (p.1.pathCond.holds env t₁ h₁ →
         p.2.1.pathCond.holds env t₂ h₂ →
         symSlotGuardOpt p.2.2 env t₁ t₂ h₁ h₂
           (symPairRaceOblig p.1 p.2.1
             ((p.1.indexOpt.getD (.const 0)).eval env t₁ h₁)
             ((p.2.1.indexOpt.getD (.const 0)).eval env t₂ h₂)
             (t₁.tx = t₂.tx ∧ t₁.ty = t₂.ty))) ∧ acc) True) :
    MirNoSharedRace f sharedId := by
  intro env t₁ t₂ ev₁ ev₂ hIn hT1 hT2 hbx hby hem1 hem2 hsp1 hsp2 hb1 hb2 hph hnotOrd hw hidx
  obtain ⟨h₁, s1, hs1, i1, k1, hk1, hd1, hw1, hssp1, hsbid1, hab1, ho1, hi1, hpc1, hps1⟩ :=
    emitsEvent_mem_collect hvalid hIn hem1
  obtain ⟨h₂, s2, hs2, i2, k2, hk2, hd2, hw2, hssp2, hsbid2, hab2, ho2, hi2, hpc2, hps2⟩ :=
    emitsEvent_mem_collect hvalid hIn hem2
  by_cases hdisj : s1.phaseSpec.disjoint s2.phaseSpec = true
  · exact False.elim (phaseSpec_disjoint_sound hdisj hps1 hps2 hph)
  · by_cases haord : symAsyncOrdered f s1 s2 = true
    · exact False.elim (hnotOrd (symAsyncOrdered_sound haord hab1 hab2 hpc1 hpc2))
    · rw [hmem] at hs1 hs2
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
        (s1, s2, extractAsyncSlotPair f s1 s2) hpair hpc1 hpc2
      have hp := symAsyncSlotGuard_sound hab1 hab2 hpc1 hpc2 hnotOrd hp0
      rw [ho1, ho2] at hp; dsimp only [Option.getD] at hp
      have heqIdx : (i1.eval env t₁ h₁ + k1) / s1.elemDiv = (i2.eval env t₂ h₂ + k2) / s2.elemDiv := by
        rw [hi1, hi2, hidx]
      obtain ⟨htx, hty⟩ := symPairRaceOblig_sound hk1 hd1 hk2 hd2 hp heqIdx
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
  intro env t b phase regs hIn hTh hreach pfx sfx cond hstmts
  obtain ⟨hb, hr, havoc, hpc, hps, hm⟩ := symReachable_sound hvalid hIn hreach
  have hvf := hvalid
  unfold validSymFunc at hvf
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hvf
  obtain ⟨_, _, hallB⟩ := hvf
  have hbOk := List.all_eq_true.mp hallB b (List.mem_range.mpr hb)
  rw [hr] at hbOk; simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true] at hbOk
  obtain ⟨_, _, _, hmA, _⟩ := symExecStmts_correct f env t havoc (tbl.getD b {}).effCond hpc (f.blockAt b).term
    (f.blockAt b).stmts regs (tbl.getD b {}).regs phase (tbl.getD b {}).phaseSpec hm hps hbOk.1.1.1 hbOk.1.2
  obtain ⟨aev, haevMem, haid, hpce, hcondEval⟩ := hmA pfx sfx assertId cond hstmts
  have haevOk := List.all_eq_true.mp hbOk.1.1.2 aev haevMem
  cases hcOpt : aev.condOpt with
  | none => rw [hcOpt] at haevOk; contradiction
  | some bSym =>
      have hfMem : aev ∈ aevs := by
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
        rw [hcondEval bSym hcOpt, hp]
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
  intro env t tev hIn hTh hemits
  obtain ⟨havoc, stev, hstev, relS, hac, hhl, hwc, hwr, hro, hre, hpc⟩ :=
    emitsTmemEvent_mem_collect hvalid hIn hemits
  rw [htmem, hevs] at hstev
  have hp := foldr_and_of_mem _ (foldr_holds_to_eval tevs (hvc env t havoc hIn hTh)) stev hstev hpc
  rw [hro] at hp; dsimp only [Option.getD] at hp
  rw [hre, hac, hhl, hwc, hwr] at hp
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
  intro env t b phase regs hIn hTh hreach pfx sfx rankVar r hstmts hr
  obtain ⟨hb, hrch, havoc, hpc, _, hm⟩ := symReachable_sound hvalid hIn hreach
  obtain ⟨rev, hrevMem, hpce, hrEq⟩ := symRankEventsOfStmts_correct f env t havoc
    (tbl.getD b {}).effCond (f.blockAt b).term (f.blockAt b).stmts pfx sfx regs (tbl.getD b {}).regs phase rankVar r hm hstmts hr
  have hfMem : rev ∈ revs := by
    rw [← hevs, ← hranks]
    unfold collectSymRankEvents
    rw [List.mem_flatMap]
    refine ⟨b, List.mem_range.mpr hb, ?_⟩
    dsimp only; rw [hrch]; simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte]; exact hrevMem
  have hp := foldr_and_of_mem _ (foldr_holds_to_eval revs (hvc env t havoc hIn hTh)) rev hfMem (by rw [hpce]; exact hpc)
  unfold symRankInBounds at hp
  cases hro : rev.rankOpt with
  | none => rw [hro] at hp; contradiction
  | some rSym => rw [hro] at hp; dsimp only at hp; rw [← hrEq rSym hro]; exact hp

open Lean Elab Tactic Meta in
/-- Precomputes `symTable`, `symMemEvents`, `symAssertEvents`, `symTmemEvents`, and `symRankEvents`
    ONCE per module using a single native `evalExpr MirFunc` call. -/
elab "mir_def_sym_table " f:ident : command => do
  Command.liftTermElabM do
    let ns ← getCurrNamespace
    let fExpr ← Term.elabTerm (← `($f:ident)) (some (mkConst ``MirFunc))
    let fVal ← unsafe evalExpr MirFunc (mkConst ``MirFunc) fExpr
    let tblVal := inferSymTable fVal
    let invTy := mkConst ``SymBlockInv
    let mut invConsts : Array Expr := #[]
    for h : i in [0 : tblVal.length] do
      let invName := ns ++ Name.mkSimple s!"symInv_{i}"
      addDecl <| Declaration.defnDecl {
        name := invName, levelParams := [], type := invTy, value := toExpr tblVal[i],
        hints := ReducibilityHints.abbrev, safety := DefinitionSafety.safe }
      invConsts := invConsts.push (mkConst invName)
    let tblTy ← Meta.mkAppM ``List #[invTy]
    let tblSpine ← Meta.mkListLit invTy invConsts.toList
    addDecl <| Declaration.defnDecl {
      name := ns ++ `symTable, levelParams := [], type := tblTy, value := tblSpine,
      hints := ReducibilityHints.abbrev, safety := DefinitionSafety.safe }
    let memTy ← Meta.mkAppM ``List #[mkConst ``SymMemEvent]
    let memVal := collectSymMemEvents fVal tblVal
    addAndCompile <| Declaration.defnDecl {
      name := ns ++ `symMemEvents, levelParams := [], type := memTy, value := toExpr memVal,
      hints := ReducibilityHints.abbrev, safety := DefinitionSafety.safe }
    let assTy ← Meta.mkAppM ``List #[mkConst ``SymAssertEvent]
    let assVal := collectSymAssertEvents fVal tblVal
    addAndCompile <| Declaration.defnDecl {
      name := ns ++ `symAssertEvents, levelParams := [], type := assTy, value := toExpr assVal,
      hints := ReducibilityHints.abbrev, safety := DefinitionSafety.safe }
    let tmemTy ← Meta.mkAppM ``List #[mkConst ``SymTmemEvent]
    let tmemVal := collectSymTmemEvents fVal tblVal
    addAndCompile <| Declaration.defnDecl {
      name := ns ++ `symTmemEvents, levelParams := [], type := tmemTy, value := toExpr tmemVal,
      hints := ReducibilityHints.abbrev, safety := DefinitionSafety.safe }
    let rankTy ← Meta.mkAppM ``List #[mkConst ``SymRankEvent]
    let rankVal := collectSymRankEvents fVal tblVal
    addAndCompile <| Declaration.defnDecl {
      name := ns ++ `symRankEvents, levelParams := [], type := rankTy, value := toExpr rankVal,
      hints := ReducibilityHints.abbrev, safety := DefinitionSafety.safe }

open Lean Elab Tactic Meta in
/-- Evaluates a closed `Nat` witness in compiled `MetaM` and closes `LHS = rhs` in the kernel via `rfl`. -/
elab "mir_bind_nat" : tactic => withMainContext do
  let tgt ← instantiateMVars (← getMainTarget)
  let some (ty, lhs, rhs) := tgt.eq? | throwError "mir_bind_nat: expected equality goal"
  let v ← unsafe evalExpr Nat ty lhs
  let _ ← isDefEq rhs (toExpr v)
  evalTactic (← `(tactic| rfl))

open Lean Elab Tactic Meta in
/-- Evaluates a closed `List SymMemEvent` witness in compiled `MetaM` and closes equality via `rfl`. -/
elab "mir_bind_events" : tactic => withMainContext do
  let tgt ← instantiateMVars (← getMainTarget)
  let some (ty, lhs, rhs) := tgt.eq? | throwError "mir_bind_events: expected equality goal"
  let v ← unsafe evalExpr (List SymMemEvent) ty lhs
  let _ ← isDefEq rhs (toExpr v)
  evalTactic (← `(tactic| rfl))

open Lean Elab Tactic Meta in
/-- Evaluates a closed `List (SymMemEvent × SymMemEvent)` witness in compiled `MetaM` and closes equality via `rfl`. -/
elab "mir_bind_pairs" : tactic => withMainContext do
  let tgt ← instantiateMVars (← getMainTarget)
  let some (ty, lhs, rhs) := tgt.eq? | throwError "mir_bind_pairs: expected equality goal"
  let v ← unsafe evalExpr (List (SymMemEvent × SymMemEvent)) ty lhs
  let _ ← isDefEq rhs (toExpr v)
  evalTactic (← `(tactic| rfl))

open Lean Elab Tactic Meta in
/-- Evaluates a closed `List (SymMemEvent × SymMemEvent × Option (SymInt × SymInt))` witness in `MetaM` and closes equality via `rfl`. -/
elab "mir_bind_shared_pairs" : tactic => withMainContext do
  let tgt ← instantiateMVars (← getMainTarget)
  let some (ty, lhs, rhs) := tgt.eq? | throwError "mir_bind_shared_pairs: expected equality goal"
  let v ← unsafe evalExpr (List (SymMemEvent × SymMemEvent × Option (SymInt × SymInt))) ty lhs
  let _ ← isDefEq rhs (toExpr v)
  evalTactic (← `(tactic| rfl))

open Lean Elab Tactic Meta in
/-- Evaluates a closed `List SymAssertEvent` witness in compiled `MetaM` and closes equality via `rfl`. -/
elab "mir_bind_asserts" : tactic => withMainContext do
  let tgt ← instantiateMVars (← getMainTarget)
  let some (ty, lhs, rhs) := tgt.eq? | throwError "mir_bind_asserts: expected equality goal"
  let v ← unsafe evalExpr (List SymAssertEvent) ty lhs
  let _ ← isDefEq rhs (toExpr v)
  evalTactic (← `(tactic| rfl))

open Lean Elab Tactic Meta in
/-- Evaluates a closed `List SymTmemEvent` witness in compiled `MetaM` and closes equality via `rfl`. -/
elab "mir_bind_tmem" : tactic => withMainContext do
  let tgt ← instantiateMVars (← getMainTarget)
  let some (ty, lhs, rhs) := tgt.eq? | throwError "mir_bind_tmem: expected equality goal"
  let v ← unsafe evalExpr (List SymTmemEvent) ty lhs
  let _ ← isDefEq rhs (toExpr v)
  evalTactic (← `(tactic| rfl))

open Lean Elab Tactic Meta in
/-- Evaluates a closed `List SymRankEvent` witness in compiled `MetaM` and closes equality via `rfl`. -/
elab "mir_bind_ranks" : tactic => withMainContext do
  let tgt ← instantiateMVars (← getMainTarget)
  let some (ty, lhs, rhs) := tgt.eq? | throwError "mir_bind_ranks: expected equality goal"
  let v ← unsafe evalExpr (List SymRankEvent) ty lhs
  let _ ← isDefEq rhs (toExpr v)
  evalTactic (← `(tactic| rfl))

/-- Concrete power-of-two constants unfolded by `mir_vc` so `omega` handles `% (2 ^ w)` directly. -/
theorem pow2_8       : (2 : Nat) ^ 8        = 256 := rfl
theorem pow2_16      : (2 : Nat) ^ 16       = 65536 := rfl
theorem pow2_32      : (2 : Nat) ^ 32       = 4294967296 := rfl
theorem pow2_64      : (2 : Nat) ^ 64       = 18446744073709551616 := rfl
theorem pow2_succ_8  : (2 : Nat) ^ (7 + 1)  = 256 := rfl
theorem pow2_succ_16 : (2 : Nat) ^ (15 + 1) = 65536 := rfl
theorem pow2_succ_32 : (2 : Nat) ^ (31 + 1) = 4294967296 := rfl
theorem pow2_succ_64 : (2 : Nat) ^ (63 + 1) = 18446744073709551616 := rfl

/-- Power-of-two bitmask identities (`x &&& (2^k - 1) = x % 2^k`) unfolded in `O(1)` by `mir_vc`. -/
theorem land_mask_1          (a : Nat) : a &&& 1          = a % 2          := Nat.and_two_pow_sub_one_eq_mod a 1
theorem land_mask_3          (a : Nat) : a &&& 3          = a % 4          := Nat.and_two_pow_sub_one_eq_mod a 2
theorem land_mask_7          (a : Nat) : a &&& 7          = a % 8          := Nat.and_two_pow_sub_one_eq_mod a 3
theorem land_mask_15         (a : Nat) : a &&& 15         = a % 16         := Nat.and_two_pow_sub_one_eq_mod a 4
theorem land_mask_31         (a : Nat) : a &&& 31         = a % 32         := Nat.and_two_pow_sub_one_eq_mod a 5
theorem land_mask_63         (a : Nat) : a &&& 63         = a % 64         := Nat.and_two_pow_sub_one_eq_mod a 6
theorem land_mask_127        (a : Nat) : a &&& 127        = a % 128        := Nat.and_two_pow_sub_one_eq_mod a 7
theorem land_mask_255        (a : Nat) : a &&& 255        = a % 256        := Nat.and_two_pow_sub_one_eq_mod a 8
theorem land_mask_511        (a : Nat) : a &&& 511        = a % 512        := Nat.and_two_pow_sub_one_eq_mod a 9
theorem land_mask_1023       (a : Nat) : a &&& 1023       = a % 1024       := Nat.and_two_pow_sub_one_eq_mod a 10
theorem land_mask_4095       (a : Nat) : a &&& 4095       = a % 4096       := Nat.and_two_pow_sub_one_eq_mod a 12
theorem land_mask_16383      (a : Nat) : a &&& 16383      = a % 16384      := Nat.and_two_pow_sub_one_eq_mod a 14
theorem land_mask_65535      (a : Nat) : a &&& 65535      = a % 65536      := Nat.and_two_pow_sub_one_eq_mod a 16
theorem land_mask_1073741823 (a : Nat) : a &&& 1073741823 = a % 1073741824 := Nat.and_two_pow_sub_one_eq_mod a 30
theorem land_mask_4294967295 (a : Nat) : a &&& 4294967295 = a % 4294967296 := Nat.and_two_pow_sub_one_eq_mod a 32
theorem land_mask_1_rev      (a : Nat) : 1          &&& a = a % 2          := by rw [Nat.and_comm, land_mask_1]
theorem land_mask_3_rev      (a : Nat) : 3          &&& a = a % 4          := by rw [Nat.and_comm, land_mask_3]
theorem land_mask_7_rev      (a : Nat) : 7          &&& a = a % 8          := by rw [Nat.and_comm, land_mask_7]
theorem land_mask_15_rev     (a : Nat) : 15         &&& a = a % 16         := by rw [Nat.and_comm, land_mask_15]
theorem land_mask_31_rev     (a : Nat) : 31         &&& a = a % 32         := by rw [Nat.and_comm, land_mask_31]
theorem land_mask_63_rev     (a : Nat) : 63         &&& a = a % 64         := by rw [Nat.and_comm, land_mask_63]
theorem land_mask_127_rev    (a : Nat) : 127        &&& a = a % 128        := by rw [Nat.and_comm, land_mask_127]
theorem land_mask_255_rev    (a : Nat) : 255        &&& a = a % 256        := by rw [Nat.and_comm, land_mask_255]
theorem land_mask_65535_rev  (a : Nat) : 65535      &&& a = a % 65536      := by rw [Nat.and_comm, land_mask_65535]

/-- Disjoint bitwise-OR identity: when `a` is a multiple of `2^k` (low `k` bits zero) and
    `b < 2^k` (bits `≥ k` zero), `a ||| b = a + b`. Used by `mir_mul_mono` on bit-split indices. -/
theorem or_eq_add_of_mod_two_pow {a b : Nat} (k : Nat) (ha : a % (2 ^ k) = 0) (hb : b < 2 ^ k) :
    a ||| b = a + b := by
  let q := a / 2 ^ k
  have ha1 : a = 2 ^ k * q := by
    have hd := Nat.div_add_mod a (2 ^ k)
    rw [ha, Nat.add_zero] at hd
    exact hd.symm
  have ha2 : a = q * 2 ^ k := by rw [ha1, Nat.mul_comm]
  apply Nat.eq_of_testBit_eq                              -- Prove `a ||| b = a + b` bit-by-bit across all bit indices `j`
  intro j
  have lhs_eq : (a ||| b).testBit j = ((q * 2 ^ k).testBit j || b.testBit j) := by
    rw [← ha2, Nat.testBit_or]
  have rhs_eq : (a + b).testBit j = if j < k then b.testBit j else q.testBit (j - k) := by
    rw [ha1, Nat.testBit_two_pow_mul_add q hb j]
  rw [lhs_eq, rhs_eq, Nat.testBit_mul_two_pow]
  by_cases hjk : j < k                                    -- Split: low bit `j < k` vs. high bit `k ≤ j`
  · have hnk : ¬(k ≤ j) := by omega
    simp [hjk, hnk]
  · have hkj : k ≤ j := by omega
    have hbj : b.testBit j = false :=
      Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le hb (Nat.pow_le_pow_right (by omega) hkj))
    simp [hjk, hkj, hbj]

/-- Symmetric disjoint bitwise-OR identity when `a < 2^k` and `b % 2^k = 0`. -/
theorem or_eq_add_of_lt_two_pow {a b : Nat} (k : Nat) (ha : a < 2 ^ k) (hb : b % (2 ^ k) = 0) :
    a ||| b = a + b := by
  rw [Nat.or_comm, or_eq_add_of_mod_two_pow k hb ha, Nat.add_comm]

/-- Tight upper bound `(2^32 - 1)^2` on the product of any two 32-bit quantities, leaving
    `≥ 2^32` headroom below `2^64` so `a * b + c < 2^64` holds for every 32-bit offset `c`. -/
theorem mul_lt_u64_of_u32 {a b : Nat} (ha : a < 4294967296) (hb : b < 4294967296) :
    a * b ≤ 18446744065119617025 := by
  have ha' : a ≤ 4294967295 := by omega
  have hb' : b ≤ 4294967295 := by omega
  exact Nat.mul_le_mul ha' hb'

open Lean Elab Tactic Meta in
/-- Injects `Nat.mul_le_mul_right` / `Nat.mul_le_mul_left` monotonicity facts, 32x32->64
    non-overflow bounds (`mul_lt_u64_of_u32`), `Nat.and_le_left`/`right` bounds on residual
    `&&&` terms, and disjoint `a ||| b = a + b` identities (`or_eq_add_of_mod_two_pow`) for `omega`. -/
elab "mir_mul_mono" : tactic => withMainContext do
  let mulRef ← IO.mkRef (#[] : Array (Expr × Expr))
  let andRef ← IO.mkRef (#[] : Array (Expr × Expr))
  let orRef  ← IO.mkRef (#[] : Array (Expr × Expr))
  let xorRef ← IO.mkRef (#[] : Array (Expr × Expr))
  let collect (e : Expr) : MetaM Unit := do
    e.forEach fun sub => do
      if sub.isAppOfArity ``HMul.hMul 6 then
        let args := sub.getAppArgs
        mulRef.modify fun acc => acc.push (args[4]!, args[5]!)
      else if sub.isAppOfArity ``HAnd.hAnd 6 then
        let args := sub.getAppArgs
        andRef.modify fun acc => acc.push (args[4]!, args[5]!)
      else if sub.isAppOfArity ``HOr.hOr 6 then
        let args := sub.getAppArgs
        orRef.modify fun acc => acc.push (args[4]!, args[5]!)
      else if sub.isAppOfArity ``HXor.hXor 6 then
        let args := sub.getAppArgs
        xorRef.modify fun acc => acc.push (args[4]!, args[5]!)
  collect (← instantiateMVars (← getMainTarget))
  for decl in (← getLCtx) do
    if !decl.isImplementationDetail then
      collect (← instantiateMVars decl.type)
  let mulTerms ← mulRef.get
  let andTerms ← andRef.get
  let orTerms  ← orRef.get
  let xorTerms ← xorRef.get
  let proveByOmega (prop : Expr) : TacticM (Option Expr) := do
    try
      let sg ← mkFreshExprMVar prop
      let saved ← getGoals
      setGoals [sg.mvarId!]
      evalTactic (← `(tactic| omega))
      setGoals saved
      pure (some sg)
    catch _ => pure none
  let mut k := 0
  for i in [:andTerms.size] do
    let (a, b) := andTerms[i]!
    k := k + 1
    let pfL ← mkAppOptM ``Nat.and_le_left #[some a, some b]
    let g1 ← (← getMainGoal).assert (Name.mkSimple s!"handL_{k}") (← inferType pfL) pfL
    let (_, g1') ← g1.intro1P
    let pfR ← mkAppOptM ``Nat.and_le_right #[some a, some b]
    let g2 ← g1'.assert (Name.mkSimple s!"handR_{k}") (← inferType pfR) pfR
    let (_, g2') ← g2.intro1P
    replaceMainGoal [g2']
  let zeroNat := toExpr (0 : Nat)
  let proveReduced (pr : Expr) : TacticM (Option Expr) := do
    try
      let sg ← mkFreshExprMVar pr
      let saved ← getGoals
      setGoals [sg.mvarId!]
      evalTactic (← `(tactic| (simp only [Nat.reducePow]; omega)))
      setGoals saved
      pure (some sg)
    catch _ => pure none
  for i in [:orTerms.size] do
    let (a, b) := orTerms[i]!
    let mut matched := false
    for shift in ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20, 24, 30, 32] : List Nat) do
      if !matched then
        let powVal : Nat := 2 ^ shift
        let powExpr := toExpr powVal
        let shiftExpr := toExpr shift
        let modA ← mkAppM ``HMod.hMod #[a, powExpr]
        if (← proveByOmega (← mkAppM ``Eq #[modA, zeroNat])).isSome then
          if (← proveByOmega (← mkAppM ``LT.lt #[b, powExpr])).isSome then
            let pModGoal ← mkAppM ``Eq #[← mkAppM ``HMod.hMod #[a, ← mkAppM ``HPow.hPow #[toExpr (2 : Nat), shiftExpr]], zeroNat]
            let pLtGoal  ← mkAppM ``LT.lt #[b, ← mkAppM ``HPow.hPow #[toExpr (2 : Nat), shiftExpr]]
            if let some h1 ← proveReduced pModGoal then
              if let some h2 ← proveReduced pLtGoal then
                k := k + 1
                let pf ← mkAppOptM ``or_eq_add_of_mod_two_pow #[some a, some b, some shiftExpr, some h1, some h2]
                let g' ← (← getMainGoal).assert (Name.mkSimple s!"hor_{k}") (← inferType pf) pf
                let (_, g'') ← g'.intro1P
                replaceMainGoal [g'']
                matched := true
        if !matched then
          let modB ← mkAppM ``HMod.hMod #[b, powExpr]
          if (← proveByOmega (← mkAppM ``LT.lt #[a, powExpr])).isSome then
            if (← proveByOmega (← mkAppM ``Eq #[modB, zeroNat])).isSome then
              let pLtGoal  ← mkAppM ``LT.lt #[a, ← mkAppM ``HPow.hPow #[toExpr (2 : Nat), shiftExpr]]
              let pModGoal ← mkAppM ``Eq #[← mkAppM ``HMod.hMod #[b, ← mkAppM ``HPow.hPow #[toExpr (2 : Nat), shiftExpr]], zeroNat]
              if let some h1 ← proveReduced pLtGoal then
                if let some h2 ← proveReduced pModGoal then
                  k := k + 1
                  let pf ← mkAppOptM ``or_eq_add_of_lt_two_pow #[some a, some b, some shiftExpr, some h1, some h2]
                  let g' ← (← getMainGoal).assert (Name.mkSimple s!"hor_{k}") (← inferType pf) pf
                  let (_, g'') ← g'.intro1P
                  replaceMainGoal [g'']
                  matched := true
  for i in [:xorTerms.size] do
    let (a, b) := xorTerms[i]!
    for shift in ([1, 2, 3, 4, 8, 16, 32] : List Nat) do
      let powVal : Nat := 2 ^ shift
      let powExpr := toExpr powVal
      let shiftExpr := toExpr shift
      if (← proveByOmega (← mkAppM ``LT.lt #[a, powExpr])).isSome then
        if (← proveByOmega (← mkAppM ``LT.lt #[b, powExpr])).isSome then
          let pA ← mkAppM ``LT.lt #[a, ← mkAppM ``HPow.hPow #[toExpr (2 : Nat), shiftExpr]]
          let pB ← mkAppM ``LT.lt #[b, ← mkAppM ``HPow.hPow #[toExpr (2 : Nat), shiftExpr]]
          if let some h1 ← proveReduced pA then
            if let some h2 ← proveReduced pB then
              k := k + 1
              let pf ← mkAppM ``Nat.xor_lt_two_pow #[h1, h2]
              let g' ← (← getMainGoal).assert (Name.mkSimple s!"hxor_{k}") (← inferType pf) pf
              let (_, g'') ← g'.intro1P
              replaceMainGoal [g'']
  let u32Bound := toExpr (4294967296 : Nat)
  for i in [:mulTerms.size] do
    let (a1, b1) := mulTerms[i]!
    if let some pa ← proveByOmega (← mkAppM ``LT.lt #[a1, u32Bound]) then
      if let some pb ← proveByOmega (← mkAppM ``LT.lt #[b1, u32Bound]) then
        k := k + 1
        let pf ← mkAppM ``mul_lt_u64_of_u32 #[pa, pb]
        let pfTy ← inferType pf
        let g' ← (← getMainGoal).assert (Name.mkSimple s!"hmul64_{k}") pfTy pf
        let (_, g'') ← g'.intro1P
        replaceMainGoal [g'']
    for j in [:mulTerms.size] do
      if i != j then
        let (a2, b2) := mulTerms[j]!
        if (← isDefEq b1 b2) && !(← isDefEq a1 a2) then
          if let some ple ← proveByOmega (← mkAppM ``LE.le #[a1, a2]) then
            k := k + 1
            let pf ← mkAppM ``Nat.mul_le_mul_right #[b1, ple]
            let pfTy ← inferType pf
            let g' ← (← getMainGoal).assert (Name.mkSimple s!"hmul_{k}") pfTy pf
            let (_, g'') ← g'.intro1P
            replaceMainGoal [g'']
        if (← isDefEq a1 a2) && !(← isDefEq b1 b2) then
          if let some ple ← proveByOmega (← mkAppM ``LE.le #[b1, b2]) then
            k := k + 1
            let pf ← mkAppM ``Nat.mul_le_mul_left #[a1, ple]
            let pfTy ← inferType pf
            let g' ← (← getMainGoal).assert (Name.mkSimple s!"hmul_{k}") pfTy pf
            let (_, g'') ← g'.intro1P
            replaceMainGoal [g'']

/-- Classical double-negation elimination as an unconditional equality for `simp only` (`0` typeclass search). -/
theorem prop_not_not (p : Prop) : (¬¬p) = p := propext Classical.not_not

open Lean Elab Tactic Meta in
/-- Recursively discharges structured VC goals (`∧`, `→`, `ite` splits via native `MVarId.byCases`,
    and `omega` / `mir_mul_mono`) with bounded `iteFuel` (`≤ 6`) and zero `delab` fragility. -/
partial def solveVcGoals (fuel iteFuel : Nat) : TacticM Unit := do
  if fuel == 0 then return
  if (← getGoals).isEmpty then return
  if (← try evalTactic (← `(tactic| trivial)); pure true catch _ => pure false) then return
  let tgt ← withMainContext do whnfR (← instantiateMVars (← getMainTarget))
  if tgt.isAppOfArity ``And 2 then
    evalTactic (← `(tactic| constructor))
    let gs ← getGoals
    for g in gs do
      setGoals [g]
      solveVcGoals (fuel - 1) iteFuel
    setGoals []
    return
  if tgt.isForall then
    let (_, g') ← (← getMainGoal).intro (Name.mkSimple s!"hvc_{fuel}")
    replaceMainGoal [g']
    solveVcGoals (fuel - 1) iteFuel
    return
  let mut gRen ← getMainGoal
  let mut rIdx := 0
  for d in (← gRen.getDecl).lctx do
    let nm := d.userName.toString
    if !d.isImplementationDetail && !(nm.startsWith "hvc_") && !(nm.startsWith "hacc_") && !(nm.startsWith "hite_") then
      rIdx := rIdx + 1
      gRen ← gRen.rename d.fvarId (Name.mkSimple s!"hacc_{fuel}_{rIdx}")
  replaceMainGoal [gRen]
  try evalTactic (← `(tactic|
    simp only [CudaOxide.Mir.Program.paramBoundsProp, SymInt.eval, SymBool.holds,
               SymBool.normDisc, extractIteOneCond, SymBool.holdsRaw,
               SymBool.eval, symSpanInBounds, symRankInBounds, symPairRaceOblig, symSlotGuardOpt,
               evalCmpOp_eq_true, evalCmpOp_eq_false, evalCmpProp,
               CudaOxide.Mir.Program.evalBinOp, CudaOxide.Mir.Program.wordMod, Option.getD,
               ↓reduceIte, eq_self, ne_eq, Nat.not_lt_zero,
               Nat.zero_add, Nat.add_zero, Nat.reduceAdd, Nat.lt_irrefl,
               Nat.max_def, Nat.reduceLeDiff, Nat.reduceMod, Nat.reducePow,
               pow2_8, pow2_16, pow2_32, pow2_64,
               pow2_succ_8, pow2_succ_16, pow2_succ_32, pow2_succ_64,
               land_mask_1, land_mask_3, land_mask_7, land_mask_15, land_mask_31,
               land_mask_63, land_mask_127, land_mask_255, land_mask_511, land_mask_1023,
               land_mask_4095, land_mask_16383, land_mask_65535, land_mask_1073741823,
               land_mask_4294967295,
               land_mask_1_rev, land_mask_3_rev, land_mask_7_rev, land_mask_15_rev,
               land_mask_31_rev, land_mask_63_rev, land_mask_127_rev, land_mask_255_rev,
               land_mask_65535_rev,
               and_true, true_and, and_false, false_and,
               or_true, true_or, or_false, false_or, not_true_eq_false, not_false_eq_true,
               not_or, prop_not_not,
               List.mem_cons, List.not_mem_nil, forall_eq_or_imp, forall_eq,
               Bool.and_eq_true, Bool.or_eq_true, Bool.not_eq_true',
               Bool.and_eq_false_iff, Bool.or_eq_false_iff, Bool.not_eq_false',
               Bool.not_false, Bool.not_true, Bool.not_eq_false, Bool.not_eq_true,
               decide_false, decide_true, decide_eq_true_eq, decide_eq_true_iff,
               decide_eq_false_iff_not, beq_iff_eq, bne_iff_ne] at *))
  catch _ => pure ()
  if (← getGoals).isEmpty then return
  withMainContext do
    if (← try evalTactic (← `(tactic| omega)); pure true catch _ => pure false) then return
    for d in (← getLCtx) do
      if !d.isImplementationDetail then
        let dTy ← whnfR (← instantiateMVars d.type)
        if dTy.isAppOfArity ``And 2 then
          let subs ← (← getMainGoal).cases d.fvarId
          for sub in subs do
            setGoals [sub.mvarId]
            solveVcGoals (fuel - 1) iteFuel
          setGoals []
          return
    if iteFuel > 0 then
      let findInnerIte (root : Expr) : Option Expr :=
        let rec go (e : Expr) : Option Expr :=
          if e.isAppOfArity ``ite 5 then
            let args := e.getAppArgs
            (go args[1]!).orElse fun _ =>
            (go args[3]!).orElse fun _ =>
            (go args[4]!).orElse fun _ => some args[1]!
          else
            match e with
            | .app f a => (go f).orElse fun _ => go a
            | _ => none
        go root
      let mut iteCondOpt := findInnerIte (← instantiateMVars (← getMainTarget))
      if iteCondOpt.isNone then
        for decl in (← getLCtx) do
          if !decl.isImplementationDetail && iteCondOpt.isNone then
            iteCondOpt := findInnerIte (← instantiateMVars decl.type)
      if let some condExpr := iteCondOpt then
        let hName := Name.mkSimple s!"hite_{fuel}_{iteFuel}"
        let hIte := mkIdent hName
        let (sPos, sNeg) ← (← getMainGoal).byCases condExpr hName
        let simpExcept (thmStx : TSyntax `Lean.Parser.Tactic.simpLemma) : TacticM Unit := do
          if (← getGoals).isEmpty then return
          try evalTactic (← `(tactic| simp only [$thmStx, ↓reduceIte, eq_self, Nat.not_lt_zero, and_false, false_and, or_false, false_or, and_true, true_and, or_true, true_or])) catch _ => pure ()
          if (← getGoals).isEmpty then return
          let decls ← withMainContext do pure (← getLCtx).decls.toList
          for dOpt in decls do
            if (← getGoals).isEmpty then return
            if let some d := dOpt then
              if !d.isImplementationDetail && d.userName != hName then
                let dId := mkIdent d.userName
                try evalTactic (← `(tactic| simp only [$thmStx, ↓reduceIte, eq_self, Nat.not_lt_zero, and_false, false_and, or_false, false_or, and_true, true_and, or_true, true_or] at $dId:ident)) catch _ => pure ()
        setGoals [sPos.mvarId]
        simpExcept (← `(Lean.Parser.Tactic.simpLemma| eq_true $hIte:ident))
        solveVcGoals (fuel - 1) (iteFuel - 1)
        setGoals [sNeg.mvarId]
        simpExcept (← `(Lean.Parser.Tactic.simpLemma| eq_false $hIte:ident))
        solveVcGoals (fuel - 1) (iteFuel - 1)
        setGoals []
        return
      for d in (← getLCtx) do
        if !d.isImplementationDetail then
          let dTy ← whnfR (← instantiateMVars d.type)
          if dTy.isAppOfArity ``Or 2 then
            let subs ← (← getMainGoal).cases d.fvarId
            for sub in subs do
              setGoals [sub.mvarId]
              solveVcGoals (fuel - 1) iteFuel
            setGoals []
            return
    if (← try evalTactic (← `(tactic| (mir_mul_mono; omega))); pure true catch _ => pure false) then return
    evalTactic (← `(tactic| omega))

/-- Invoke `solveVcGoals` with structural fuel `512` and `ite` case-split fuel `16`. -/
elab "mir_solve_vc" : tactic => solveVcGoals 512 16

/-- Primary proof tactic for `MirFunc` arithmetic obligations (`no_oob_idx`, `disjoint_writes`,
    `no_data_race`, `MirAssertRedundant`). Unfolds `List.foldr` into per-event subgoals first
    (avoiding monolithic `simp` congruence proofs across large event-pair lists), then unfolds
    symbolic evaluators into `Nat` arithmetic and closes every leaf with `omega`. -/
macro "mir_vc" f:ident : tactic =>
  `(tactic| (
      unfold CudaOxide.Mir.Program.MirFunc.validInput CudaOxide.Mir.Program.MirFunc.validThread
      dsimp only [$f:ident, List.foldr, symSlotGuardOpt, symPairRaceOblig, Option.getD]
      mir_solve_vc))

end CudaOxide.Proofs.Program
