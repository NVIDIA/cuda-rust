/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide

/-!
Negative model test: `warp::sync_mask(u32::MAX)` (`Stmt.syncWarp`) only orders shared-memory
accesses within a single 32-lane hardware warp, NEVER across distinct warps in the CTA.
Here `tx` writes `TILE[tx]`, executes `Stmt.syncWarp`, and reads `TILE[(tx + 32) % 64]`
(reading the neighboring warp's slot without `thread::sync_threads()`). `MirNoSharedRace`
must fail in Lean 4.
-/
open CudaOxide.Mir.Program CudaOxide.Mir.ControlFlow
open CudaOxide.Contracts.Program CudaOxide.Contracts.ControlFlow

def crossWarpMir : MirFunc :=
  { domain := 1, blockDimX := 64, blockDimY := 1
  , sharedDecls := [{ size := 64, elemBytes := 4 }]
  , blocks := [
      { stmts := [.assign 0 (.sreg .tidX), .assign 1 (.intCast 64 0),
                  .assign 2 (.sharedBase 0), .assign 3 (.ptrOffset 2 1),
                  .assign 4 (.constInt 32 7), .store 3 4]
      , term := .goto 1 [] },
      { stmts := [.syncWarp], term := .goto 2 [] },
      { stmts := [.assign 5 (.constInt 64 32), .assign 6 (.binOp .add 64 1 5),
                  .assign 7 (.constInt 64 64), .assign 8 (.binOp .rem 64 6 7),
                  .assign 9 (.ptrOffset 2 8), .assign 10 (.load 9)]
      , term := .ret } ] }

def crossWarpCfg : ControlFlowCfg :=
  { blockDimX := 64, blockDimY := 1, numShared := 1, numBarriers := 0
  , blocks := [
      { ops := [.sharedWrite 0], succs := [(.always, 1)] },
      { ops := [.syncWarp true], succs := [(.always, 2)] },
      { ops := [.sharedRead 0], isReturn := true } ] }

cfg_def_uni_state_map crossWarpMir crossWarpCfg
mir_def_sym_table crossWarpMir

theorem crossWarpMir_valid :
    CudaOxide.Proofs.Program.validSymFunc crossWarpMir symTable = true := rfl

theorem crossWarpEvents_eq :
    CudaOxide.Proofs.Program.collectSymMemEvents crossWarpMir symTable = symMemEvents := rfl

theorem crossWarpUni_valid :
    CudaOxide.Proofs.ControlFlow.checkMirValidUniSetFast crossWarpMir cfgUniVars cfgRfPtrs cfgUniSums cfgSuccChunks cfgRets cfgAllEdges cfgUniConds = true := rfl

theorem crossWarpBarriers_valid :
    CudaOxide.Proofs.ControlFlow.checkUniformBarriersWith cfgRefined cfgUniStateMap = true := rfl

theorem crossWarp_uniform_barriers :
    UniformBarriers crossWarpMir crossWarpCfg :=
  CudaOxide.Proofs.ControlFlow.uniform_barriers_of_certs rfl rfl rfl rfl rfl crossWarpUni_valid crossWarpBarriers_valid

-- Intentionally false: `(tx + 32) % 64` crosses from Warp 0 into Warp 1 across `syncWarp` alone.
theorem rejected_cross_warp_sync_mask_race :
    MirNoSharedRace crossWarpMir crossWarpCfg 0 := by
  apply CudaOxide.Proofs.Program.vcNoSharedRace_of_table crossWarp_uniform_barriers crossWarpMir_valid crossWarpEvents_eq (by mir_bind_shared_pairs)
  mir_vc crossWarpMir
