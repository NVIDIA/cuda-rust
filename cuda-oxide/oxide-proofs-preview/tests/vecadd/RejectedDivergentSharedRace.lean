/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide

/-!
Negative model test: a divergent `if tx < 32 { sync_threads() }` puts `tx = 0` in `phase = 1`
and `tx = 32` in `phase = 0` so both access `TILE[0]` concurrently in unequal local phases.
`MirNoSharedRace` requires `UniformBarriers divRaceCfg`, so attempting to prove `MirNoSharedRace`
must fail in Lean 4.
-/
open CudaOxide.Mir.Program CudaOxide.Mir.ControlFlow
open CudaOxide.Contracts.Program CudaOxide.Contracts.ControlFlow

/-- Kernel where only `tx < 32` executes `syncThreads`, then `tx = 0` writes `TILE[0]` (phase 1)
    while `tx = 32` reads `TILE[0]` (phase 0). -/
def divRaceMir : MirFunc :=
  { domain := 1, blockDimX := 64, blockDimY := 1
  , sharedDecls := [{ size := 64, elemBytes := 4 }]
  , blocks := [
      { stmts := [.assign 0 (.sreg .tidX), .assign 1 (.constInt 32 32), .assign 2 (.cmpOp .lt 0 1)]
      , term := .condBr 2 1 [] 2 [] },
      { stmts := [.syncThreads, .assign 3 (.constInt 32 0), .assign 4 (.cmpOp .eq 0 3)]
      , term := .condBr 4 3 [] 2 [] },
      { stmts := [.assign 5 (.cmpOp .eq 0 1)], term := .condBr 5 4 [] 5 [] },
      { stmts := [.assign 6 (.sharedBase 0), .assign 7 (.constInt 64 0), .assign 8 (.ptrOffset 6 7),
                  .assign 9 (.constInt 32 42), .store 8 9], term := .goto 2 [] },
      { stmts := [.assign 10 (.sharedBase 0), .assign 11 (.constInt 64 0), .assign 12 (.ptrOffset 10 11),
                  .assign 13 (.load 12)], term := .goto 5 [] },
      { term := .ret } ] }

/-- Matching `ControlFlowCfg` with `syncThreads` in block `1` under `tx < 32`. -/
def divRaceCfg : ControlFlowCfg :=
  { blockDimX := 64, blockDimY := 1, numShared := 1, numBarriers := 0
  , blocks := [
      { succs := [(.coord (.lt .tx (.const 32)), 1), (.coord (.ge .tx (.const 32)), 2)] },
      { ops := [.syncThreads], succs := [(.coord (.eq .tx (.const 0)), 3), (.coord (.ne .tx (.const 0)), 2)] },
      { succs := [(.coord (.eq .tx (.const 32)), 4), (.coord (.ne .tx (.const 32)), 5)] },
      { ops := [.sharedWrite 0], succs := [(.always, 2)] },
      { ops := [.sharedRead 0], succs := [(.always, 5)] },
      { isReturn := true } ] }

mir_def_sym_table divRaceMir

theorem divRaceMir_valid :
    CudaOxide.Proofs.Program.validSymFunc divRaceMir symTable = true :=
  rfl

theorem divRaceEvents_eq :
    CudaOxide.Proofs.Program.collectSymMemEvents divRaceMir symTable = symMemEvents :=
  rfl

-- `vcNoSharedRace_of_table` requires `UniformBarriers divRaceMir divRaceCfg`, which fails on block `1` (`tx < 32`).
theorem rejected_divergent_shared_race :
    MirNoSharedRace divRaceMir divRaceCfg 0 := by
  apply CudaOxide.Proofs.Program.vcNoSharedRace_of_table ?_ divRaceMir_valid divRaceEvents_eq (by mir_bind_shared_pairs)
  · mir_vc divRaceMir                                       -- Equal-phase pair check is empty,
  · have h : (let (uv, rf, su, rs, ie, uc, ref, sm) := CudaOxide.Proofs.ControlFlow.inferUniformityBundle divRaceMir divRaceCfg
              CudaOxide.Proofs.ControlFlow.checkUniformBarriersMir divRaceMir uv rf su rs ref ie uc sm) = true := by
      decide +revert; omega                                 -- BUT `UniformBarriers` fails on divergent block `1`!
    exact CudaOxide.Proofs.ControlFlow.uniform_barriers_of_stateMap rfl rfl rfl rfl h
