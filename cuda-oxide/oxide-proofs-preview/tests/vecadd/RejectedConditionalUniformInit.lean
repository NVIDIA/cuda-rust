/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide

/-!
Negative model test: an acyclic `.uniform` diamond `if flag { SHARED[tid] = ...; } sync_threads(); read SHARED[tid];`
leaves `SHARED` uninitialized when `!flag` and must be rejected by `UniformBarriersCfg`.
-/
namespace RejectedConditionalUniformInit
open CudaOxide.Mir CudaOxide.Contracts

def condUniformInitCfg : ControlFlow.ControlFlowCfg :=
  { blockDimX := 256, blockDimY := 1, numShared := 1, numBarriers := 0
  , blocks := [
      { succs := [(.uniform, 1), (.uniform, 2)] },
      { ops := [.sharedWrite 0], succs := [(.always, 2)] },
      { ops := [.syncThreads], succs := [(.always, 3)] },
      { ops := [.sharedRead 0], succs := [(.always, 4)] },
      { isReturn := true } ] }

theorem rejected_cond_uniform_init : ControlFlow.UniformBarriersCfg condUniformInitCfg :=
  CudaOxide.Proofs.ControlFlow.uniform_barriers_cfg_of_stateMap
    (inv := CudaOxide.Proofs.ControlFlow.inferStateMap condUniformInitCfg) (by decide)

end RejectedConditionalUniformInit
