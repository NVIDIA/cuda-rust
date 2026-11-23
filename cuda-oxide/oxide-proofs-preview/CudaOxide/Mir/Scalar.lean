/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import Init.Data.BitVec.Lemmas

/-!
# Scalar `dialect-mir` Expression AST & Evaluator (`CudaOxide.Mir.Scalar`)

Defines the Lean representation for pure single-block `dialect-mir` expressions
over same-width unsigned machine words (`ui8`, `ui16`, `ui32`, `ui64`):

```text
  dialect-mir operation                  CudaOxide.Mir.Scalar.Expr
  ──────────────────────────────────     ──────────────────────────────────────
  ^entry(arg_i: uiW)                     `Expr.arg ⟨i, by decide⟩`
  mir.constant / builtin.constant        `Expr.literal (BitVec.ofNat W v)`
  mir.add (lhs, rhs)                     `Expr.add lhs rhs`
  mir.sub (lhs, rhs)                     `Expr.sub lhs rhs`
  mir.mul (lhs, rhs)                     `Expr.mul lhs rhs`
```

Arithmetic wraps modulo `2 ^ width` via `BitVec width`. When a 1D launch bound
proves `block * blockSize + lane < 2 ^ width`, `mul_add_toNat_of_lt` bridges
the `BitVec` evaluation to the exact natural formula `CudaOxide.Mir.globalIndex`.
-/
namespace CudaOxide.Mir.Scalar

/-- An SSA scalar expression whose arguments and intermediate values share one
    unsigned machine-word `width` (`8`, `16`, `32`, or `64`). Each argument
    reference carries a `Fin arity` proof that its index is in range. -/
inductive Expr (width arity : Nat) where
  /-- Entry block argument at bounded position `index` (`0 ≤ index < arity`). -/
  | arg (index : Fin arity)
  /-- Unsigned integer constant literal of `width` bits. -/
  | literal (value : BitVec width)
  /-- Wrapping `width`-bit integer addition (`mir.add`). -/
  | add (lhs rhs : Expr width arity)
  /-- Wrapping `width`-bit integer subtraction (`mir.sub`). -/
  | sub (lhs rhs : Expr width arity)
  /-- Wrapping `width`-bit integer multiplication (`mir.mul`). -/
  | mul (lhs rhs : Expr width arity)

/-- Evaluate a scalar MIR expression with exact wrapping `BitVec` arithmetic. -/
def Expr.eval {width arity : Nat} (expr : Expr width arity)
    (inputs : Fin arity → BitVec width) : BitVec width :=
  match expr with
  | .arg index => inputs index
  | .literal value => value
  | .add lhs rhs => lhs.eval inputs + rhs.eval inputs
  | .sub lhs rhs => lhs.eval inputs - rhs.eval inputs
  | .mul lhs rhs => lhs.eval inputs * rhs.eval inputs

/-- Evaluating an argument reference `.arg index` looks up `inputs index`. -/
@[simp] theorem eval_arg {width arity : Nat} (index : Fin arity)
    (inputs : Fin arity → BitVec width) :
    (Expr.arg index).eval inputs = inputs index := rfl

/-- Evaluating a constant literal `.literal value` returns `value`. -/
@[simp] theorem eval_literal {width arity : Nat} (value : BitVec width)
    (inputs : Fin arity → BitVec width) :
    (Expr.literal value).eval inputs = value := rfl

/-- Evaluating `.add lhs rhs` adds the evaluated operands in `BitVec width`. -/
@[simp] theorem eval_add {width arity : Nat} (lhs rhs : Expr width arity)
    (inputs : Fin arity → BitVec width) :
    (Expr.add lhs rhs).eval inputs = lhs.eval inputs + rhs.eval inputs := rfl

/-- Evaluating `.sub lhs rhs` subtracts the evaluated operands in `BitVec width`. -/
@[simp] theorem eval_sub {width arity : Nat} (lhs rhs : Expr width arity)
    (inputs : Fin arity → BitVec width) :
    (Expr.sub lhs rhs).eval inputs = lhs.eval inputs - rhs.eval inputs := rfl

/-- Evaluating `.mul lhs rhs` multiplies the evaluated operands in `BitVec width`. -/
@[simp] theorem eval_mul {width arity : Nat} (lhs rhs : Expr width arity)
    (inputs : Fin arity → BitVec width) :
    (Expr.mul lhs rhs).eval inputs = lhs.eval inputs * rhs.eval inputs := rfl

/-- Wrapping `BitVec` multiply-add agrees with natural arithmetic when the
    unwrapped result fits in `2 ^ width`. -/
theorem mul_add_toNat_of_lt {width : Nat} (block blockSize lane : Nat)
    (hfit : block * blockSize + lane < 2 ^ width) :
    (BitVec.ofNat width block * BitVec.ofNat width blockSize +
      BitVec.ofNat width lane).toNat = block * blockSize + lane := by
  rw [← BitVec.ofNat_mul, ← BitVec.ofNat_add]             -- Fold BitVec `*` and `+` into `BitVec.ofNat width (block * blockSize + lane)`
  exact Nat.mod_eq_of_lt hfit                             -- `.toNat` is `% 2^width`, which is a no-op since `hfit` proves `< 2^width`

end CudaOxide.Mir.Scalar
