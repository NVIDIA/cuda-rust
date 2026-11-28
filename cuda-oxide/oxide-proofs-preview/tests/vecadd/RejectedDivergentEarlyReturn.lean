/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide

/-!
Negative model test: replacing a block-uniform early `return` before `sync_threads()` with a
thread-divergent early `return` (`if tid == 0 { return; }`) leaves threads `1..255` deadlocked
at `sync_threads()` and must be rejected by `UniformBarriersCfg`.
-/
namespace RejectedDivergentEarlyReturn
open CudaOxide.Mir CudaOxide.Contracts

def divergentEarlyReturnCfg : ControlFlow.ControlFlowCfg :=
  { blockDimX := 256, blockDimY := 1, numShared := 1, numBarriers := 0
  , blocks := [
      { succs := [(.coord (.eq .tx (.const 0)), 2), (.coord (.ne .tx (.const 0)), 1)] },
      { ops := [.sharedWrite 0], succs := [(.always, 3)] },
      { isReturn := true },
      { ops := [.syncThreads], succs := [(.always, 4)] },
      { ops := [.sharedRead 0], succs := [(.always, 2)] } ] }

theorem rejected_divergent_early_return : ControlFlow.UniformBarriersCfg divergentEarlyReturnCfg :=
  CudaOxide.Proofs.ControlFlow.uniform_barriers_cfg_of_stateMap
    (inv := CudaOxide.Proofs.ControlFlow.inferStateMap divergentEarlyReturnCfg) (by decide)

end RejectedDivergentEarlyReturn
