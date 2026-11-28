/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide

/-!
Negative model test: executing a full-warp `0xFFFF_FFFF` synchronous collective (`CfgOp.warpSyncFull`,
e.g. `warp::ballot(pred)`) inside a sub-warp lane-divergent guard (`if lane == 0`) violates 32-lane
warp convergence and must be rejected by `UniformBarriersCfg`.
-/
namespace RejectedDivergentWarpVote
open CudaOxide.Mir CudaOxide.Contracts

def divergentWarpVoteCfg : ControlFlow.ControlFlowCfg :=
  { blockDimX := 256, blockDimY := 1, numShared := 0, numBarriers := 0
  , blocks := [
      { succs := [(.coord (.eq (.mod .tx 32) (.const 0)), 1),
                  (.coord (.ne (.mod .tx 32) (.const 0)), 2)] },
      { ops := [.warpSyncFull], succs := [(.always, 2)] },
      { isReturn := true } ] }

theorem rejected_divergent_warp_vote : ControlFlow.UniformBarriersCfg divergentWarpVoteCfg :=
  CudaOxide.Proofs.ControlFlow.uniform_barriers_cfg_of_stateMap
    (inv := CudaOxide.Proofs.ControlFlow.inferStateMap divergentWarpVoteCfg) (by decide)

end RejectedDivergentWarpVote
