/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide

set_option maxRecDepth 65536

/-!
Negative model test: reading `x[4 * idx + 3]` under `idx < out.len()` with an insufficient
scaled launch precondition `requires = (x.len() >= 3 * out.len())` can walk out of bounds
when `out.len() = 1` and `x.len() = 3`, and must be rejected by Lean 4.
-/
namespace RejectedInsufficientRequiresScale
open CudaOxide.Proofs.Program

def badQuadGatherMirFunc : CudaOxide.Mir.Program.MirFunc :=
  { domain := 1
  , blockDimX := 256
  , blockDimY := 1
  , params := [(.slice false), (.slice true)]
  , requires := [
      { cmp := .ge
      , lhs := .sliceLen 0
      , rhs := .mul (.const 3) (.sliceLen 1) } ]
  , blocks := [
      { args := [⟨0, 0⟩, ⟨1, 0⟩], stmts := [
          (.assign 2 .index1D),
          (.assign 3 (.sliceLen 1)),
          (.assign 4 (.cmpOp .lt 2 3))
        ], term := (.condBr 4 1 [] 2 []) },
      { stmts := [
          (.assign 5 (.constInt 64 4)),
          (.assign 6 (.binOp .mul 64 5 2)),
          (.assign 7 (.constInt 64 3)),
          (.assign 8 (.binOp .add 64 6 7)),
          (.assign 9 (.slicePtr 0)),
          (.assign 10 (.ptrOffset 9 8)),
          (.assign 11 (.load 10))
        ], term := (.goto 2 []) },
      { stmts := [], term := .ret }
    ] }

mir_def_sym_table badQuadGatherMirFunc

theorem badQuad_valid : validSymFunc badQuadGatherMirFunc symTable = true := rfl
theorem badQuad_events : collectSymMemEvents badQuadGatherMirFunc symTable = symMemEvents := rfl

theorem rejected_insufficient_requires_scale :
    CudaOxide.Contracts.Program.MirNoOobGlobal badQuadGatherMirFunc 0 := by
  apply vcNoOobGlobal_of_table badQuad_valid badQuad_events (by mir_bind_events)
  mir_vc badQuadGatherMirFunc

end RejectedInsufficientRequiresScale
