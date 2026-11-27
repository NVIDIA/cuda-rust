/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import Lean
import CudaOxide

/-!
Regression tests for `CudaOxide.Proofs.Program.Tactic.solveVcGoals`:
Step 8 must still case-split `Or` hypotheses in context when `iteFuel` reaches `0`.
-/
namespace OrSplitAfterIteFuelTests
open CudaOxide.Proofs.Program

open Lean Elab Tactic in
local elab "mir_solve_vc_ite0" : tactic => solveVcGoals 512 0

/-- Disjunction hypothesis splitting still fires when `iteFuel = 0` on a nonlinear goal
    that plain `omega` and `mir_mul_mono; omega` without `Or`-splitting cannot solve. -/
theorem orSplit_zero_iteFuel (a b c d M : Nat)
    (h : (a ≤ c ∧ c * b ≤ M) ∨ (a ≤ d ∧ d * b ≤ M)) : a * b ≤ M := by
  mir_solve_vc_ite0

/-- Public `mir_solve_vc` (`iteFuel = 16`) still splits `Or` hypotheses after 16 `ite`
    splits exhaust `iteFuel`. -/
theorem orSplit_after_iteFuel_exhausted (t a b c d M : Nat) (ht : 16 ≤ t)
    (h : (a ≤ c ∧ c * b ≤ M) ∨ (a ≤ d ∧ d * b ≤ M))
    (h16 : (if t = 0 then 0 else if t = 1 then 0 else if t = 2 then 0 else if t = 3 then 0 else
            if t = 4 then 0 else if t = 5 then 0 else if t = 6 then 0 else if t = 7 then 0 else
            if t = 8 then 0 else if t = 9 then 0 else if t = 10 then 0 else if t = 11 then 0 else
            if t = 12 then 0 else if t = 13 then 0 else if t = 14 then 0 else if t = 15 then 0 else 1) = 1) :
    a * b ≤ M := by
  mir_solve_vc

#print axioms orSplit_zero_iteFuel
#print axioms orSplit_after_iteFuel_exhausted

end OrSplitAfterIteFuelTests
