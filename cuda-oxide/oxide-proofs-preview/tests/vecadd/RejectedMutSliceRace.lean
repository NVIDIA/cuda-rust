/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide.Contracts.Access

/-!
Negative model test: reading `c[idx + 1]` while writing `c[idx]` on `c: &mut [f32]`
creates a cross-thread read-write race (thread `0` reads `c[1]` while thread `1`
writes `c[1]`) and must be rejected by Lean 4.
-/
theorem rejected_mut_slice_cross_thread_race :
    CudaOxide.Contracts.DisjointSliceAccessPair 256 0 1 := by
  intro block₁ block₂ lane₁ lane₂ hlane₁ hlane₂ hsame
  omega
