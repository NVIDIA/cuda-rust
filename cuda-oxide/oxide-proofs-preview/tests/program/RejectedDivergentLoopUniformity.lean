/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide

/-!
Negative model test: `let n = if tx < 16 { 1 } else { 3 }; while i < n { sync_threads(); i += 1 }`
labels `while i < n` as `.uniform` in `ControlFlowCfg`, and `UniformBarriers` must reject it in
Lean 4 because `MirValidUniSet` excludes `n` (`1 ≠ 3` across split arms) and downgrades `i < n`
to `.unknown`.
-/
open CudaOxide.Mir.Program CudaOxide.Mir.ControlFlow
open CudaOxide.Contracts.ControlFlow CudaOxide.Proofs.ControlFlow

/-- `MirFunc` for `let n = if tx < 16 { 1 } else { 3 }; while i < n { sync_threads(); i += 1 }`. -/
def splitLoopMir : MirFunc :=
  { domain := 1, blockDimX := 64, blockDimY := 1, params := [.disjointSlice]
  , blocks := [
      { args := [⟨0, 0⟩], stmts := [.assign 1 (.sreg .tidX), .assign 2 (.constInt 32 16), .assign 3 (.cmpOp .lt 1 2)]
      , term := .condBr 3 1 [] 2 [] },
      { stmts := [.assign 4 (.constInt 32 1)], term := .goto 3 [4] },
      { stmts := [.assign 5 (.constInt 32 3)], term := .goto 3 [5] },
      { args := [⟨6, 32⟩], stmts := [.assign 7 (.constInt 32 0)], term := .goto 4 [7] },
      { args := [⟨8, 32⟩], isLoopHeader := true, stmts := [.assign 9 (.cmpOp .lt 8 6)]
      , term := .condBr 9 5 [] 6 [] },
      { stmts := [.syncThreads, .assign 10 (.constInt 32 1), .assign 11 (.binOp .add 32 8 10)]
      , term := .goto 4 [11] },
      { stmts := [.syncThreads], term := .ret } ] }

/-- `ControlFlowCfg` claiming `.uniform` on block `4` (`condVarOpt := some 9`). -/
def splitLoopCfg : ControlFlowCfg :=
  { blockDimX := 64, blockDimY := 1, numShared := 0, numBarriers := 0
  , blocks := [
      { condVarOpt := some 3, succs := [(.coord (.lt .tx (.const 16)), 1), (.coord (.ge .tx (.const 16)), 2)] },
      { succs := [(.always, 3)] },
      { succs := [(.always, 3)] },
      { succs := [(.always, 4)] },
      { condVarOpt := some 9, succs := [(.uniform, 5), (.uniform, 6)] },
      { ops := [.syncThreads], succs := [(.always, 4)] },
      { ops := [.syncThreads], isReturn := true } ] }

cfg_def_uni_state_map splitLoopMir splitLoopCfg

-- Attempting to prove `UniformBarriers splitLoopMir splitLoopCfg` fails because `9 ∉ cfgUniVars`.
theorem rejected_split_loop_uniform_barriers :
    UniformBarriers splitLoopMir splitLoopCfg := by
  have h : checkUniformBarriersMir splitLoopMir cfgUniVars cfgRfPtrs cfgUniSums cfgRets cfgRefined cfgAllEdges cfgUniConds cfgUniStateMap = true := by
    have hf : checkUniformBarriersMir splitLoopMir cfgUniVars cfgRfPtrs cfgUniSums cfgRets cfgRefined cfgAllEdges cfgUniConds cfgUniStateMap = false := rfl
    omega                                                   -- `false = true` fails in `omega`!
  exact uniform_barriers_of_stateMap rfl rfl rfl rfl h
