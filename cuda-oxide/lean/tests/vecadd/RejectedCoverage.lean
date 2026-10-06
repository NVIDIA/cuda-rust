/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide.Proofs.Launch

-- Intentionally false: this launch has only four lanes for five outputs.
-- The required launch_coverage premise fails; Counterexamples.lean proves index 4
-- is an output index with no launched writer for this configuration.
example : (5 : Nat) ≤ 2 * 2 := by
  omega
