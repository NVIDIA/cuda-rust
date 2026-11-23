/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide.Proofs.Bounds

-- Intentionally false: replacing `<` with `≤` permits the invalid boundary.
-- A real generated obligation must match the exported guard exactly.
example {outLen i : Nat} (active : i ≤ outLen) : i < outLen := by
  omega
