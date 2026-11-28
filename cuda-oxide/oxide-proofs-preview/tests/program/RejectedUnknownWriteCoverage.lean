/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide

/-!
Block 0 branches on an opaque runtime condition; only block 1 writes `out[bx]`, while block 2
returns without writing. `MirWritesAllElems` requires `EmitsTrapFreeEvent` (`ReachableTrapFree`),
so an angelic attempt to pick block 1 across `.unk` must fail.
-/
open CudaOxide.Mir.Program CudaOxide.Contracts.Program

/-- Kernel that writes `out[bx]` only on the `then` arm of an untracked `.opaqueVal` branch. -/
def conditionalWrite : MirFunc :=
  { domain := 1, blockDimX := 1, blockDimY := 1, params := [.disjointSlice]
  , blocks := [
      { args := [⟨0, 0⟩], stmts := [.assign 2 .opaqueVal], term := .condBr 2 1 [] 2 [] },
      { stmts := [ .assign 3 (.slicePtr 0)
                 , .assign 4 (.sreg .ctaidX)
                 , .assign 5 (.ptrOffset 3 4)
                 , .assign 6 (.constInt 32 1)
                 , .store 5 6 ], term := .ret },
      { term := .ret } ] }

-- This claim is false because `.unk` does not guarantee taking block 1 (`ReachableTrapFree` requires `.bool true`).
example : MirWritesAllElems conditionalWrite 0 := by
  intro env gridDimX gridDimY _ _ hgyPos _ hcov elemIdx helem -- Unpack launch & element bounds
  refine ⟨{ bx := elemIdx, by_ := 0, tx := 0, ty := 0 },      -- Propose witness thread `bx = elemIdx`
          { isWrite := true, space := .global, bufId := 0, index := elemIdx, phase := 0 }, ?_⟩
  have hbxLt : elemIdx < gridDimX := by                       -- `blockDimX = 1` gives `elemIdx < gridDimX`
    simp [conditionalWrite] at hcov; omega
  refine ⟨?_, hbxLt, hgyPos, ?_, rfl, rfl, rfl, rfl⟩          -- Split thread validity vs `EmitsTrapFreeEvent`
  · refine ⟨by decide, by decide, ?_, by decide, by simp [conditionalWrite], by simp [conditionalWrite]⟩
    omega                                                     -- `elemIdx < gridDimX <= 2^32`
  · refine ⟨1, (0, 0), _, .stepCondTrue (.entry (by decide)) rfl rfl rfl ?_ (by decide), rfl, ?_⟩
    · simp [conditionalWrite, MirFunc.blockAt, execStmts, evalRhs, RegFile.set] <;> omega -- `.unk = .bool true` fails
    · simp [conditionalWrite, MirFunc.blockAt, execStmts, evalRhs, RegFile.set]           -- Block 1 emits the store
