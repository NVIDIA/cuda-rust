/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide

namespace RejectedTmemColOverflow
open CudaOxide.Proofs.Program

-- Intentionally false: 128-column MMA on a 64-column TMEM allocation overflows `allocCols`.
def badFunc : CudaOxide.Mir.Program.MirFunc :=
  { domain := 1
  , blockDimX := 128
  , blockDimY := 1
  , params := [.disjointSlice]
  , sharedDecls := [{ size := 1, elemBytes := 4 }]
  , blocks := [
      { args := [⟨0, 0⟩]
      , stmts := [
          (.assign 1 (.sharedBase 0)),
          (.assign 2 (.loadTmemBase 1 64)),
          (.tmemAccess 2 128 128 false)
        ]
      , term := .ret }
    ] }

mir_def_sym_table badFunc

theorem badFunc_valid : validSymFunc badFunc symTable = true := rfl
theorem badTmem_eq : collectSymTmemEvents badFunc symTable = symTmemEvents := rfl

theorem false_claim : CudaOxide.Contracts.Program.MirNoOobTmem badFunc := by
  apply vcNoOobTmem_of_table badFunc_valid badTmem_eq (by mir_bind_tmem)
  mir_vc badFunc

end RejectedTmemColOverflow
