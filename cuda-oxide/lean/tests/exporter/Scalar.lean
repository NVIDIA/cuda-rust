/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide.Model.Scalar
import CudaOxide.Model.Launch

/-! Checks for word overflow, expression structure, and argument ordering. -/
namespace ScalarTests
open CudaOxide.Model.Scalar

private def noInputs {width : Nat} : Fin 0 → BitVec width := Fin.elim0

-- Overflow is wrapping, not natural-number addition.
example :
    (Expr.add (.literal (BitVec.ofNat 8 255)) (.literal (BitVec.ofNat 8 1))
      : Expr 8 0).eval noInputs = BitVec.ofNat 8 0 := by
  decide

-- Subtraction wraps below zero as an unsigned word.
example :
    (Expr.sub (.literal (BitVec.ofNat 8 0)) (.literal (BitVec.ofNat 8 1))
      : Expr 8 0).eval noInputs = BitVec.ofNat 8 255 := by
  decide

-- 200 * 2 = 400, which reduces to 144 modulo 256.
example :
    (Expr.mul (.literal (BitVec.ofNat 8 200)) (.literal (BitVec.ofNat 8 2))
      : Expr 8 0).eval noInputs = BitVec.ofNat 8 144 := by
  decide

private def orderedInputs (index : Fin 2) : BitVec 8 :=
  if index.val = 0 then BitVec.ofNat 8 9 else BitVec.ofNat 8 4

private def firstMinusSecond : Expr 8 2 :=
  .sub (.arg ⟨0, by decide⟩) (.arg ⟨1, by decide⟩)

example : firstMinusSecond.eval orderedInputs = BitVec.ofNat 8 5 := by
  decide

-- Reversing argument references changes subtraction rather than commuting it.
example :
    (Expr.sub (.arg ⟨1, by decide⟩) (.arg ⟨0, by decide⟩)
      : Expr 8 2).eval orderedInputs = BitVec.ofNat 8 251 := by
  decide

-- A nested expression exercises SSA-expression composition.
example :
    (Expr.add (.mul (.arg ⟨0, by decide⟩) (.literal (BitVec.ofNat 8 30)))
      (.arg ⟨1, by decide⟩) : Expr 8 2).eval orderedInputs = BitVec.ofNat 8 18 := by
  decide

-- The arithmetic mapping to the natural launch model needs an explicit bound.
-- It still does not establish that any particular Rust function was exported
-- correctly, or that a computed address satisfies the stricter sentinel bound.
example (block blockSize lane : Nat)
    (hfit : block * blockSize + lane < 2 ^ 64) :
    (BitVec.ofNat 64 block * BitVec.ofNat 64 blockSize +
      BitVec.ofNat 64 lane).toNat =
      CudaOxide.Model.globalIndex blockSize block lane := by
  exact mul_add_toNat_of_lt block blockSize lane hfit

#print axioms mul_add_toNat_of_lt

end ScalarTests
