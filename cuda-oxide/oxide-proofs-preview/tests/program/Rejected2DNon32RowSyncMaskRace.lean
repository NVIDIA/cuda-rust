/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide

/-!
Negative model test: on a 2D thread block whose X dimension is NOT a multiple of 32
(`block = (48, 2, 1)`, `96` threads = `3` warps), threads `(tx = 0, ty = 1)` (`linear_tid = 48`,
Warp `1`) and `(tx = 20, ty = 1)` (`linear_tid = 68`, Warp `2`) both have `tx / 32 == 0` AND
`ty == 1`, yet belong to DIFFERENT hardware warps! Attempting to prove `MirNoSharedRace` across
`Stmt.syncWarp` on `TILE[0]` must fail in Lean 4.
-/
open CudaOxide.Mir.Program CudaOxide.Mir.ControlFlow
open CudaOxide.Contracts.Program CudaOxide.Contracts.ControlFlow

def non32RowMir : MirFunc :=
  { domain := 2, blockDimX := 48, blockDimY := 2
  , sharedDecls := [{ size := 4, elemBytes := 4 }]
  , blocks := [
      { stmts := [.assign 0 (.sreg .tidX), .assign 1 (.sreg .tidY),
                  .assign 2 (.constInt 32 0), .assign 3 (.constInt 32 1),
                  .assign 4 (.cmpOp .eq 0 2), .assign 5 (.cmpOp .eq 1 3)]
      , term := .condBr 4 1 [] 3 [] },
      { stmts := [], term := .condBr 5 2 [] 3 [] },
      { stmts := [.assign 6 (.sharedBase 0), .assign 7 (.constInt 64 0),
                  .assign 8 (.ptrOffset 6 7), .assign 9 (.constInt 32 99), .store 8 9]
      , term := .goto 3 [] },
      { stmts := [.syncThreads], term := .goto 4 [] },
      { stmts := [.assign 10 (.sharedBase 0), .assign 11 (.constInt 64 0),
                  .assign 12 (.ptrOffset 10 11), .assign 13 (.constInt 32 100)]
      , term := .condBr 4 5 [] 7 [] },
      { stmts := [], term := .condBr 5 6 [] 7 [] },
      { stmts := [.store 12 13], term := .goto 7 [] },
      { stmts := [.syncWarp], term := .goto 8 [] },
      { stmts := [.assign 14 (.constInt 32 20), .assign 15 (.cmpOp .eq 0 14)]
      , term := .condBr 15 9 [] 11 [] },
      { stmts := [], term := .condBr 5 10 [] 11 [] },
      { stmts := [.assign 16 (.load 12)], term := .goto 11 [] },
      { term := .ret } ] }

def non32RowCfg : ControlFlowCfg :=
  { blockDimX := 48, blockDimY := 2, numShared := 1, numBarriers := 0
  , blocks := [
      { succs := [(.coord (.eq .tx (.const 0)), 1), (.coord (.ne .tx (.const 0)), 3)] },
      { succs := [(.coord (.eq .ty (.const 1)), 2), (.coord (.ne .ty (.const 1)), 3)] },
      { ops := [.sharedWrite 0], succs := [(.always, 3)] },
      { ops := [.syncThreads], succs := [(.always, 4)] },
      { succs := [(.coord (.eq .tx (.const 0)), 5), (.coord (.ne .tx (.const 0)), 7)] },
      { succs := [(.coord (.eq .ty (.const 1)), 6), (.coord (.ne .ty (.const 1)), 7)] },
      { ops := [.sharedWrite 0], succs := [(.always, 7)] },
      { ops := [.syncWarp true], succs := [(.always, 8)] },
      { succs := [(.coord (.eq .tx (.const 20)), 9), (.coord (.ne .tx (.const 20)), 11)] },
      { succs := [(.coord (.eq .ty (.const 1)), 10), (.coord (.ne .ty (.const 1)), 11)] },
      { ops := [.sharedRead 0], succs := [(.always, 11)] },
      { isReturn := true } ] }

mir_def_sym_table non32RowMir

theorem non32RowMir_valid :
    CudaOxide.Proofs.Program.validSymFunc non32RowMir symTable = true := rfl

theorem non32RowEvents_eq :
    CudaOxide.Proofs.Program.collectSymMemEvents non32RowMir symTable = symMemEvents := rfl

-- Intentionally false: `(0, 1)` (`linear_tid = 48`, Warp 1) and `(20, 1)` (`linear_tid = 68`, Warp 2)
-- are NOT in the same hardware warp (`48 / 32 != 68 / 32`), so `MirNoSharedRace` fails!
theorem rejected_2d_non32_row_sync_mask_race (hunif : UniformBarriers non32RowMir non32RowCfg) :
    MirNoSharedRace non32RowMir non32RowCfg 0 := by
  apply CudaOxide.Proofs.Program.vcNoSharedRace_of_table hunif non32RowMir_valid non32RowEvents_eq (by mir_bind_shared_pairs)
  mir_vc non32RowMir
