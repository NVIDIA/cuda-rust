/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide

namespace RejectedTmemWarpSlice
open CudaOxide.Proofs.Program

-- Intentionally false: all 4 warps (`tx < 128`) execute warp-restricted `tcgen05.ld`
-- at fixed TMEM lane 0 instead of `(tx / 32) * 32`, violating PTX ISA 9.7.18.8.1.
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
          (.assign 2 (.loadTmemBase 1 128)),
          (.tmemAccess 2 32 16 true)
        ]
      , term := .ret }
    ] }

mir_def_sym_table badFunc

theorem badFunc_valid : validSymFunc badFunc symTable = true := rfl
theorem badTmem_eq : collectSymTmemEvents badFunc symTable = symTmemEvents := rfl

theorem false_claim : CudaOxide.Contracts.Program.MirNoOobTmem badFunc := by
  apply vcNoOobTmem_of_table badFunc_valid badTmem_eq (by mir_bind_tmem)
  mir_vc badFunc

end RejectedTmemWarpSlice
