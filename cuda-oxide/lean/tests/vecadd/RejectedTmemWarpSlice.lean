/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide.Proofs.Bounds

open CudaOxide.Contracts CudaOxide.Mir.Kernel

-- Intentionally false: all 4 warps (`tx < 128`) read fixed TMEM lane 0 instead of
-- `(tx / 32) * 32`, violating PTX ISA 9.7.18.8.1 for warps 1..3 (`tx = 32..127`).
example :
    NoOobTmemAccess 128 1 128
      { laneExpr := .const 0,
        colExpr := .const 0,
        heightLanes := 32, widthCols := 16,
        loopBound := 1, warpRestricted := true } := by
  intro tx ty i htx hty hi hguard
  simp only [IdxExpr.eval, GuardExpr.eval] at *
  omega
