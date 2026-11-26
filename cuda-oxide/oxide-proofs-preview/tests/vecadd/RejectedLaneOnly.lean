/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide.Proofs.Launch

private def brokenLaneOnlyIndex (_blockSize _block lane : Nat) : Nat := lane

-- Intentionally false: two different blocks can share a lane number.
example {blockSize block₁ block₂ lane₁ lane₂ : Nat}
    (hlane₁ : lane₁ < blockSize) (hlane₂ : lane₂ < blockSize)
    (hsame : brokenLaneOnlyIndex blockSize block₁ lane₁ =
      brokenLaneOnlyIndex blockSize block₂ lane₂) :
    block₁ = block₂ ∧ lane₁ = lane₂ := by
  simp only [brokenLaneOnlyIndex] at hsame
  omega
