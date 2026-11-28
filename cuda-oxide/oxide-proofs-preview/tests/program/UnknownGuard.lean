/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide

/-! Checks that `Reachable` follows both arms of a branch whose condition is `.unk`. -/
namespace UnknownGuardTests
open CudaOxide.Mir.Program CudaOxide.Contracts.Program

/-- Block 0 branches on an opaque value. The then-arm asserts `false`. -/
def unknownGuard : MirFunc :=
  { domain := 1, blockDimX := 32, blockDimY := 1, params := [.slice false]
  , blocks := [
      { args := [⟨0, 0⟩], stmts := [.assign 2 .opaqueVal], term := .condBr 2 1 [] 2 [] },
      { stmts := [.assign 3 (.constBool false), .assertCond 0 3], term := .ret },
      { term := .ret } ] }

/-- The then-arm is reachable on every thread. -/
theorem then_arm_reachable (env : InputEnv) (t : ThreadCtx) :
    ∃ regs, Reachable unknownGuard env t 1 regs (0, 0) :=
  ⟨_, .stepCondTrue (.entry (by decide)) rfl rfl (by simp [evalRhs, RegFile.set]) (by decide)⟩

/-- The else-arm is reachable on every thread. -/
theorem else_arm_reachable (env : InputEnv) (t : ThreadCtx) :
    ∃ regs, Reachable unknownGuard env t 2 regs (0, 0) :=
  ⟨_, .stepCondFalse (.entry (by decide)) rfl rfl (by simp [evalRhs, RegFile.set]) (by decide)⟩

/-- The `assert(false)` in the then-arm of a `.unk` branch is not redundant. -/
theorem assert_not_redundant (env : InputEnv) (t : ThreadCtx)
    (henv : unknownGuard.validInput env) (ht : unknownGuard.validThread t) :
    ¬ MirAssertRedundant unknownGuard 0 := by
  intro h                                                          -- Assume `MirAssertRedundant unknownGuard 0` holds
  obtain ⟨regs, hr⟩ := then_arm_reachable env t                    -- Witness reachable `then`-arm state `(1, regs, (0, 0))`
  have hcond := h env t 1 (0, 0) regs henv ht hr [.assign 3 (.constBool false)] [] 3 rfl -- Assert `0` reads register `3`
  simp [execStmts, evalRhs, RegFile.set] at hcond                  -- Register `3` evaluates to `.bool false`, contradiction

#print axioms then_arm_reachable
#print axioms else_arm_reachable
#print axioms assert_not_redundant

end UnknownGuardTests
