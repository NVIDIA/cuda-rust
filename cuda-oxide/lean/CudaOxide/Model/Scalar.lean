/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import Init.Data.BitVec.Lemmas

/-!
Expression semantics for the first scalar exporter subset: pure functions with
one basic block and same-width unsigned words (`u8`, `u16`, `u32`, or `u64`).
The model permits any width; the exporter is responsible for rejecting widths
and operations outside its supported subset.

Arithmetic wraps at the selected width. This module has no GPU execution or
memory semantics and does not prove that a Rust-to-model bridge is correct.
-/
namespace CudaOxide.Model.Scalar

/-- An expression whose arguments and intermediate values share one word width.
    An argument reference carries evidence that its index is below the arity. -/
inductive Expr (width arity : Nat) where
  | arg (index : Fin arity)
  | literal (value : BitVec width)
  | add (lhs rhs : Expr width arity)
  | sub (lhs rhs : Expr width arity)
  | mul (lhs rhs : Expr width arity)

/-- Evaluate a scalar expression with exact wrapping word arithmetic. -/
def Expr.eval {width arity : Nat} (expr : Expr width arity)
    (inputs : Fin arity → BitVec width) : BitVec width :=
  match expr with
  | .arg index => inputs index
  | .literal value => value
  | .add lhs rhs => lhs.eval inputs + rhs.eval inputs
  | .sub lhs rhs => lhs.eval inputs - rhs.eval inputs
  | .mul lhs rhs => lhs.eval inputs * rhs.eval inputs

@[simp] theorem eval_arg {width arity : Nat} (index : Fin arity)
    (inputs : Fin arity → BitVec width) :
    (Expr.arg index).eval inputs = inputs index := rfl

@[simp] theorem eval_literal {width arity : Nat} (value : BitVec width)
    (inputs : Fin arity → BitVec width) :
    (Expr.literal value).eval inputs = value := rfl

@[simp] theorem eval_add {width arity : Nat} (lhs rhs : Expr width arity)
    (inputs : Fin arity → BitVec width) :
    (Expr.add lhs rhs).eval inputs = lhs.eval inputs + rhs.eval inputs := rfl

@[simp] theorem eval_sub {width arity : Nat} (lhs rhs : Expr width arity)
    (inputs : Fin arity → BitVec width) :
    (Expr.sub lhs rhs).eval inputs = lhs.eval inputs - rhs.eval inputs := rfl

@[simp] theorem eval_mul {width arity : Nat} (lhs rhs : Expr width arity)
    (inputs : Fin arity → BitVec width) :
    (Expr.mul lhs rhs).eval inputs = lhs.eval inputs * rhs.eval inputs := rfl

/-- Wrapping multiply-add agrees with natural arithmetic when the full natural
    result fits. This is an arithmetic lemma, not a source translation proof. -/
theorem mul_add_toNat_of_lt {width : Nat} (block blockSize lane : Nat)
    (hfit : block * blockSize + lane < 2 ^ width) :
    (BitVec.ofNat width block * BitVec.ofNat width blockSize +
      BitVec.ofNat width lane).toNat = block * blockSize + lane := by
  rw [← BitVec.ofNat_mul, ← BitVec.ofNat_add]
  exact Nat.mod_eq_of_lt hfit

end CudaOxide.Model.Scalar
