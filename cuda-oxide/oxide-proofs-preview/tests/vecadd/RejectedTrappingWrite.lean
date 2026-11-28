/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide

set_option maxRecDepth 65536

/-!
Negative model test: a kernel with `writes_all_elems(c)` that executes an unproven
`mir.assert(idx < a.len())` before writing `c[idx]` (without `requires = (a.len() >= c.len())`)
traps when `a.len() == 0 < c.len()` and must be rejected by Lean 4.
-/
namespace RejectedTrappingWrite
open CudaOxide.Proofs.Program

def trappingMirFunc : CudaOxide.Mir.Program.MirFunc :=
  { domain := 1
  , blockDimX := 256
  , blockDimY := 1
  , params := [(.slice false), (.slice true)]
  , requires := []
  , blocks := [
      { args := [⟨0, 0⟩, ⟨1, 0⟩], stmts := [
          (.assign 2 .index1D),
          (.assign 3 (.sliceLen 1)),
          (.assign 4 (.cmpOp .lt 2 3))
        ], term := (.condBr 4 1 [] 2 []) },
      { stmts := [
          (.assign 5 (.sliceLen 0)),
          (.assign 6 (.cmpOp .lt 2 5)),
          (.assertCond 0 6),
          (.assign 7 (.slicePtr 1)),
          (.assign 8 (.ptrOffset 7 2)),
          (.assign 9 .opaqueVal),
          (.store 8 9)
        ], term := (.goto 2 []) },
      { stmts := [], term := .ret }
    ] }

theorem rejected_trapping_assert_before_write :
    CudaOxide.Contracts.Program.MirWritesAllElems trappingMirFunc 1 := by
  apply vcWritesAllElems rfl (by mir_bind_writes)
  mir_vc trappingMirFunc

end RejectedTrappingWrite
