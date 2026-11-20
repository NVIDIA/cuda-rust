/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide.Contracts.Access
import CudaOxide.Proofs.Launch

/-!
# Reusable 1D Buffer-Bounds Proofs (`CudaOxide.Proofs.Bounds`)

Reusable lemmas over `CudaOxide.Contracts.NoOobIdx` and `NoOobScaledGuard`.
Line-by-line kernel bounds proofs live in `CudaOxide.Proofs.Program`.
-/
namespace CudaOxide.Proofs.Bounds
open CudaOxide.Contracts

/-- If `i < guardLen` (from an `if` check) and `guardLen ≤ bufferLen` (from
    `requires`), then `i < bufferLen`. -/
theorem index_lt_of_guard {i guardLen bufferLen : Nat}
    (active : i < guardLen) (fits : guardLen ≤ bufferLen) : i < bufferLen := by
  exact Nat.lt_of_lt_of_le active fits                    -- Chain `i < guardLen` and `guardLen ≤ bufferLen`

/-- Proves `no_oob_idx(buf)` for `buf[idx]` when the kernel checks `if idx < guardLen`
    and the launch contract guarantees `guardLen ≤ bufferLen` (e.g., `c.len() ≤ a.len()`). -/
theorem no_oob_idx_of_guard_le {blockSize guardLen bufferLen : Nat}
    (fits : guardLen ≤ bufferLen) :
    NoOobIdx blockSize 0 guardLen bufferLen := by
  intro block lane active                                 -- Unpack `∀ block lane` and guard `active : idx < guardLen`
  simpa using index_lt_of_guard active fits               -- Simplify `idx + 0` to `idx` and apply `index_lt_of_guard`

/-- Proves `no_oob_idx(buf)` for `buf[idx]` when the kernel checks `buf`'s own
    length directly (`if idx < buf.len()`). -/
theorem no_oob_idx_of_same_len {blockSize bufferLen : Nat} :
    NoOobIdx blockSize 0 bufferLen bufferLen := by
  exact no_oob_idx_of_guard_le (Nat.le_refl bufferLen)    -- Special case where `bufferLen ≤ bufferLen` by reflexivity

/-- Proves `no_oob_idx(data)` when the kernel checks `idx / groupSize < guardLen`
    and the launch contract guarantees `guardLen * groupSize ≤ bufferLen`. -/
theorem no_oob_idx_of_div_guard_le {blockSize groupSize guardLen bufferLen : Nat}
    (hpos : 0 < groupSize) (hfit : guardLen * groupSize ≤ bufferLen) :
    NoOobScaledGuard blockSize groupSize guardLen bufferLen := by
  intro block lane hdiv                                   -- Unpack `idx / groupSize < guardLen`
  have hlt : CudaOxide.Mir.globalIndex blockSize block lane < guardLen * groupSize :=
    (Nat.div_lt_iff_lt_mul hpos).1 hdiv                   -- Rewrite `idx / W < guardLen` as `idx < guardLen * W`
  exact Nat.lt_of_lt_of_le hlt hfit                       -- Chain with `guardLen * groupSize ≤ bufferLen`

end CudaOxide.Proofs.Bounds
