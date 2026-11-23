/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide

/-!
Block 0 branches on an opaque value. `Reachable` must reach both arms. A proof that
only block 0 is reachable must fail.
-/
open CudaOxide.Mir.Program

/-- Block 0 branches on an opaque value. -/
def unknownGuard : MirFunc :=
  { domain := 1, blockDimX := 32, blockDimY := 1, params := [.slice false]
  , blocks := [
      { args := [⟨0, 0⟩], stmts := [.assign 2 .opaqueVal], term := .condBr 2 1 [] 2 [] },
      { stmts := [.assign 3 (.constBool false), .assertCond 0 3], term := .ret },
      { term := .ret } ] }

-- This claim is false because a `.unk` condition prunes neither arm.
example {env t b regs phase} (h : Reachable unknownGuard env t b regs phase) : b = 0 := by
  induction h with
  | entry => rfl
  | stepGoto _ _ hterm _ ih => subst ih; simp [unknownGuard, MirFunc.blockAt] at hterm
  | stepCondTrue _ hexec hterm hcond _ ih | stepCondFalse _ hexec hterm hcond _ ih =>
      subst ih
      simp [unknownGuard, MirFunc.blockAt] at hterm
      obtain ⟨rfl, -⟩ := hterm                              -- The branch condition is register `2`
      simp [unknownGuard, MirFunc.blockAt, execStmts, evalRhs] at hexec
      obtain ⟨rfl, -⟩ := hexec                              -- Block 0 leaves `.unk` in register `2`
      simp [RegFile.set] at hcond <;> omega                 -- `.unk` excludes neither arm
