/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import Init

/-!
# 1D GPU Launch Coordinates (`CudaOxide.Mir`)

Defines the Lean representation of 1D SIMT launch coordinates (`blockIdx.x`,
`blockDim.x`, `threadIdx.x`, and `thread::index_1d`) used by exported
`dialect-mir` kernels:

```text
  CUDA / dialect-mir                     CudaOxide.Mir
  ──────────────────────────────────     ──────────────────────────────────────
  blockIdx.x * blockDim.x + threadIdx.x  `globalIndex blockSize block lane`
  Launched 1D thread `(block, lane)`     `Thread1D blocks blockSize` (`Fin`-bounded)
```

Values use exact natural numbers; fixed-width register wraparound is modeled in
`CudaOxide.Mir.Scalar` (`BitVec`) and bridged via `mul_add_toNat_of_lt`.
-/
namespace CudaOxide.Mir

/-- Logical linear index `block * blockSize + lane` of a 1D launch. -/
def globalIndex (blockSize block lane : Nat) : Nat :=
  block * blockSize + lane

/-- A 1D GPU thread whose block and lane bounds are carried in `Fin` types.
    Assumes launch dimensions Y and Z are 1 (`#[launch_contract(domain = 1)]`). -/
structure Thread1D (blocks blockSize : Nat) where
  /-- Block index along the X axis (`0 ≤ block < blocks`). -/
  block : Fin blocks
  /-- Thread lane index inside the block (`0 ≤ lane < blockSize`). -/
  lane : Fin blockSize

/-- The linear global index of a `Thread1D` with checked launch coordinates. -/
def Thread1D.index {blocks blockSize : Nat}
    (thread : Thread1D blocks blockSize) : Nat :=
  globalIndex blockSize thread.block.val thread.lane.val

end CudaOxide.Mir
