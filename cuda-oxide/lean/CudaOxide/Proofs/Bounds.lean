/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import Lean

/-!
Reusable buffer-bounds facts for generated access obligations.

These facts concern natural-number indices and lengths. Applying them to a
kernel requires a source-to-model connection and justified length assumptions;
they do not establish pointer validity, aliasing, or machine arithmetic behavior.
-/
namespace CudaOxide.Proofs.Bounds

/-- A strict guard supplies a buffer bound when its limit fits the buffer.
    A generated obligation can apply this fact separately to every access. -/
theorem index_lt_of_guard {i guardLen bufferLen : Nat}
    (active : i < guardLen) (fits : guardLen ≤ bufferLen) : i < bufferLen := by
  exact Nat.lt_of_lt_of_le active fits

end CudaOxide.Proofs.Bounds
