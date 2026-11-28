/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide

/-!
Negative model test: `.block` (`BlockAtomic*`, PTX `.cta`) atomic operations on a global slice slot
`buf[0]` accessed by multiple CTAs (`bx₁ ≠ bx₂`) fail `morallyStrongAtomic` and must be rejected by
`MirNoGlobalRace`.
-/
namespace RejectedCrossBlockCtaAtomic
open CudaOxide.Mir.Program CudaOxide.Contracts.Program CudaOxide.Proofs.Program

def crossBlockCtaAtomicFunc : MirFunc :=
  { domain := 1
  , blockDimX := 256
  , blockDimY := 1
  , params := [(.slice true)]
  , blocks := [
      { args := [⟨0, 0⟩], stmts := [
          (.assign 1 (.constInt 64 0)),
          (.assign 2 (.sliceLen 0)),
          (.assign 3 (.cmpOp .lt 1 2))
        ], term := (.condBr 3 1 [] 2 []) },
      { stmts := [
          (.assign 4 (.slicePtr 0)),
          (.assign 5 (.ptrOffset 4 1)),
          (.assign 6 (.atomicOp true .block 5 0 32))
        ], term := (.goto 2 []) },
      { stmts := [], term := .ret } ] }

mir_def_sym_table crossBlockCtaAtomicFunc

theorem cta_valid : validSymFunc crossBlockCtaAtomicFunc symTable = true := rfl
theorem cta_events : collectSymMemEvents crossBlockCtaAtomicFunc symTable = symMemEvents := rfl

theorem rejected_cross_block_cta_atomic : MirNoGlobalRace crossBlockCtaAtomicFunc 0 := by
  apply vcNoGlobalRace_of_table cta_valid cta_events (by mir_bind_pairs)
  mir_vc crossBlockCtaAtomicFunc

end RejectedCrossBlockCtaAtomic
