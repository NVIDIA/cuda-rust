/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide

/-!
Negative model test: thread `0` arms `BAR` and starts a TMA copy into `TILE` (block `3`), then every
thread executes `sync_threads()` (block `4`) and reads `TILE` (block `5`). Nobody waits on `BAR`.
A CTA barrier orders the instruction that starts the copy, not its completion, so the read can see
stale data. This is `examples/safe_async_copy` with `thread::sync_threads()` in place of the
`mbarrier` wait. `Accepted.tma_read_after_wait` is the same graph with the wait.
-/
open CudaOxide.Mir.ControlFlow
open CudaOxide.Contracts.ControlFlow
open CudaOxide.Proofs.ControlFlow

namespace RejectedTmaReadAfterSync

def cfg : ControlFlowCfg :=
  { blockDimX := 256, blockDimY := 1, numShared := 1, numBarriers := 1
  , blocks := [
      { succs := [(.coord (.eq .tx (.const 0)), 1), (.coord (.ne .tx (.const 0)), 2)] },
      { ops := [.mbarInit 0 1, .proxyFenceAsyncShared], succs := [(.always, 2)] },
      { ops := [.syncThreads]
      , succs := [(.coord (.eq .tx (.const 0)), 3), (.coord (.ne .tx (.const 0)), 4)] },
      { ops := [.mbarArriveExpectTx 0 16384, .tmaCopyG2S 0 0], succs := [(.always, 4)] },
      { ops := [.syncThreads], succs := [(.always, 5)] },
      { ops := [.sharedRead 0], succs := [(.always, 6)] },
      { isReturn := true } ] }

-- Intentionally false: the read in block `5` is refused while the copy on `BAR` is still pending.
theorem false_claim : UniformBarriersCfg cfg :=
  uniform_barriers_cfg_of_stateMap (cfg := cfg) (inv := inferStateMap cfg) (by decide)

end RejectedTmaReadAfterSync
