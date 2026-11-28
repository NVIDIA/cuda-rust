/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide

/-!
Negative model test: mixing a `.device` atomic `fetch_add` on `buf[0]` with a non-atomic `Stmt.store`
on `buf[0]` across threads violates PTX ISA §8.4 (`morallyStrongAtomic = False`) and must be rejected
by `MirNoGlobalRace`.
-/
namespace RejectedMixedAtomicRace
open CudaOxide.Mir.Program CudaOxide.Contracts.Program CudaOxide.Proofs.Program

def mixedAtomicFunc : MirFunc :=
  { domain := 1
  , blockDimX := 256
  , blockDimY := 1
  , params := [(.slice true), (.scalar 32)]
  , blocks := [
      { args := [⟨0, 0⟩, ⟨1, 32⟩], stmts := [
          (.assign 2 (.constInt 64 0)),
          (.assign 3 (.sliceLen 0)),
          (.assign 4 (.cmpOp .lt 2 3))
        ], term := (.condBr 4 1 [] 3 []) },
      { stmts := [
          (.assign 5 (.slicePtr 0)),
          (.assign 6 (.ptrOffset 5 2)),
          (.assign 7 (.atomicOp true .device 6 0 32)),
          (.assign 8 (.cmpOp .eq 7 1))
        ], term := (.condBr 8 2 [] 3 []) },
      { stmts := [
          (.assign 9 (.constInt 32 0)),
          (.store 6 9)
        ], term := (.goto 3 []) },
      { stmts := [], term := .ret } ] }

mir_def_sym_table mixedAtomicFunc

theorem mixed_valid : validSymFunc mixedAtomicFunc symTable = true := rfl
theorem mixed_events : collectSymMemEvents mixedAtomicFunc symTable = symMemEvents := rfl

theorem rejected_mixed_atomic_race : MirNoGlobalRace mixedAtomicFunc 0 := by
  apply vcNoGlobalRace_of_table mixed_valid mixed_events (by mir_bind_pairs)
  mir_vc mixedAtomicFunc

end RejectedMixedAtomicRace
