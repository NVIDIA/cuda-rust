/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide.Proofs.Bounds

-- Intentionally false: a valid output index can exceed a shorter input.
example {aLen outLen i : Nat} (active : i < outLen) : i < aLen := by
  omega
