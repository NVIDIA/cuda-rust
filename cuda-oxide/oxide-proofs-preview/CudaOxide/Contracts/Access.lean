/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide.Mir.Launch

/-!
# Standalone 1D Access & Grid-Coverage Contracts (`CudaOxide.Contracts`)

Foundational 1D launch-coverage and index-disjointness propositions over
`CudaOxide.Mir.globalIndex`. End-to-end kernel exports use the operational
line-by-line contracts in `CudaOxide.Contracts.Program` and
`CudaOxide.Contracts.ControlFlow`.
-/
namespace CudaOxide.Contracts
open CudaOxide.Mir

/-- **`no_oob_idx(buf)`**: Every thread `(block, lane)` that passes the strict
    guard `globalIndex blockSize block lane < guardLen` indexes `buf` at
    `globalIndex blockSize block lane + offset < bufferLen`. -/
def NoOobIdx (blockSize offset guardLen bufferLen : Nat) : Prop :=
  ∀ block lane : Nat,                                     -- Pick ANY thread (block, lane):
    globalIndex blockSize block lane < guardLen →         -- IF it passes the branch guard (`idx < guardLen`),
    globalIndex blockSize block lane + offset < bufferLen -- THEN its buffer access is in-bounds (`< bufferLen`).

/-- **`disjoint_writes(buf)`**: Distinct launched 1D threads `(block₁, lane₁)`
    and `(block₂, lane₂)` never write to the same element index in `buf`. -/
def DisjointWrites (blockSize offset : Nat) : Prop :=
  ∀ block₁ block₂ lane₁ lane₂ : Nat,                      -- Pick ANY two threads (block₁, lane₁) and (block₂, lane₂):
    lane₁ < blockSize →                                   -- IF thread 1's lane is valid,
    lane₂ < blockSize →                                   -- AND thread 2's lane is valid,
    globalIndex blockSize block₁ lane₁ + offset =         -- AND both threads write to the exact
      globalIndex blockSize block₂ lane₂ + offset →       --     same buffer index (`idx₁ + offset = idx₂ + offset`),
    block₁ = block₂ ∧ lane₁ = lane₂                       -- THEN they are the exact same thread (`∧` = AND).

/-- **`disjoint_writes(buf)` (cross-access pair on `&mut [T]`)**: Across a write
    at `offset₁` and any read or write at `offset₂` on the same writable slice,
    distinct 1D threads never touch the same element index. -/
def DisjointSliceAccessPair (blockSize offset₁ offset₂ : Nat) : Prop :=
  ∀ block₁ block₂ lane₁ lane₂ : Nat,                      -- Pick ANY two threads (block₁, lane₁) and (block₂, lane₂):
    lane₁ < blockSize →                                   -- IF thread 1's lane is valid,
    lane₂ < blockSize →                                   -- AND thread 2's lane is valid,
    globalIndex blockSize block₁ lane₁ + offset₁ =        -- AND thread 1's access (`idx₁ + offset₁`) hits the
      globalIndex blockSize block₂ lane₂ + offset₂ →      --     same element as thread 2's (`idx₂ + offset₂`),
    block₁ = block₂ ∧ lane₁ = lane₂                       -- THEN they must be the exact same thread (`∧` = AND).

/-- **`writes_all_elems(buf)`**: Every output element index `i < outLen` in `buf`
    is written by some launched thread `(block, lane)` in the 1D grid. -/
def WritesAllElems (blocks blockSize outLen : Nat) : Prop :=
  ∀ i : Nat,                                              -- Pick ANY output element index `i`:
    i < outLen →                                          -- IF `i` is within the output length (`i < outLen`),
    ∃ block lane : Nat,                                   -- THEN there EXISTS a thread (block, lane) such that:
      block < blocks ∧                                    --   1. `block` is a valid launched block, AND
      lane < blockSize ∧                                  --   2. `lane` is a valid lane in that block, AND
      globalIndex blockSize block lane = i                --   3. that thread's global index equals `i`.

/-- **`disjoint_writes(buf)` (warp / sub-group leader)**: Distinct 1D threads
    `(block₁, lane₁)` and `(block₂, lane₂)` that both pass the group-leader
    guard `idx % groupSize = 0` never write to the same group slot `idx / groupSize`. -/
def DisjointGroupWrites (blockSize groupSize : Nat) : Prop :=
  ∀ block₁ block₂ lane₁ lane₂ : Nat,                      -- Pick ANY two threads (block₁, lane₁) and (block₂, lane₂):
    lane₁ < blockSize →                                   -- IF thread 1's lane is valid,
    lane₂ < blockSize →                                   -- AND thread 2's lane is valid,
    globalIndex blockSize block₁ lane₁ % groupSize = 0 →  -- AND thread 1 is its group's leader (`idx₁ % W = 0`),
    globalIndex blockSize block₂ lane₂ % groupSize = 0 →  -- AND thread 2 is its group's leader (`idx₂ % W = 0`),
    globalIndex blockSize block₁ lane₁ / groupSize =      -- AND both leaders write to the exact same
      globalIndex blockSize block₂ lane₂ / groupSize →    --     group slot (`idx₁ / W = idx₂ / W`),
    block₁ = block₂ ∧ lane₁ = lane₂                       -- THEN they are the exact same thread (`∧` = AND).

/-- **`no_oob_idx(buf)` (group-scaled guard)**: Every 1D thread `(block, lane)`
    whose group index `idx / groupSize < guardLen` indexes `buf[idx]` strictly
    below `bufferLen` when `guardLen * groupSize ≤ bufferLen`. -/
def NoOobScaledGuard (blockSize groupSize guardLen bufferLen : Nat) : Prop :=
  ∀ block lane : Nat,                                     -- Pick ANY thread (block, lane):
    globalIndex blockSize block lane / groupSize < guardLen → -- IF its group slot is `< guardLen`,
    globalIndex blockSize block lane < bufferLen          -- THEN its element index `idx` is `< bufferLen`.

end CudaOxide.Contracts
