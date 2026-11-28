/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide

/-!
Negative model test: `warp::sync_mask` executed under a sub-warp divergent guard (`tx < 16`)
must be refused by `UniformBarriersCfg` (`checkUniformBarriersWith` evaluates to `false`).
-/
open CudaOxide.Mir.ControlFlow
open CudaOxide.Contracts.ControlFlow
open CudaOxide.Proofs.ControlFlow

namespace RejectedSubwarpSyncMask

def cfg : ControlFlowCfg :=
  { blockDimX := 32, blockDimY := 1, numShared := 0, numBarriers := 0
  , blocks := [
      { succs := [(.coord (.lt .tx (.const 16)), 1), (.coord (.ge .tx (.const 16)), 2)] },
      { ops := [.syncWarp true], succs := [(.always, 2)] },
      { isReturn := true } ] }

-- Intentionally false: `syncWarp` in block `1` is guarded by sub-warp `tx < 16`.
theorem false_claim : UniformBarriersCfg cfg :=
  uniform_barriers_cfg_of_stateMap (cfg := cfg) (inv := inferStateMap cfg) (by decide)

end RejectedSubwarpSyncMask
