/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide.Proofs.Bounds

open CudaOxide.Contracts CudaOxide.Mir.Kernel

-- Intentionally false: 5 steps of 16 columns (`0..80`) exceed a 64-column TMEM allocation.
example :
    NoOobTmemAccess 128 1 64
      { laneExpr := .mul (.div .tx 32) (.const 32),
        colExpr := .mul .loopVar (.const 16),
        heightLanes := 32, widthCols := 16,
        loopBound := 5, warpRestricted := true } := by
  intro tx ty i htx hty hi hguard
  simp only [IdxExpr.eval, GuardExpr.eval] at *
  omega
