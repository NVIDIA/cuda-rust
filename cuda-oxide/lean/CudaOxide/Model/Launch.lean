/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import Init

/-!
A mathematical model of a one-dimensional launch. Values use exact natural
numbers; this module does not model checked machine arithmetic or CUDA execution.
The model must be connected to the Rust kernel before its proofs apply to code.
-/
namespace CudaOxide.Model

/-- Logical index used by a one-dimensional launch. -/
def globalIndex (blockSize block lane : Nat) : Nat :=
  block * blockSize + lane

/-- A thread whose block and lane bounds are carried in the field types.
    The model assumes that all launch dimensions other than x are one. -/
structure Thread1D (blocks blockSize : Nat) where
  block : Fin blocks
  lane : Fin blockSize

/-- The logical index of a thread with checked launch coordinates. -/
def Thread1D.index {blocks blockSize : Nat}
    (thread : Thread1D blocks blockSize) : Nat :=
  globalIndex blockSize thread.block.val thread.lane.val

end CudaOxide.Model
