/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide.Mir.Launch
import CudaOxide.Mir.Kernel

/-!
# Kernel Verification Contracts (`CudaOxide.Contracts`)

Defines the Lean propositions that correspond 1-to-1 with the clauses written
by a kernel author in Rust's `#[verify(...)]` attribute:

```text
  Rust `#[verify(...)]` Clause           Lean Proposition (`CudaOxide.Contracts`)
  ──────────────────────────────────     ──────────────────────────────────────────
  `no_oob_idx(buf)`                      `NoOobIdx` / `NoOobTileAccess` / `NoOobTmemAccess`
  `disjoint_writes(buf)`                 `DisjointWrites blockSize offset`
  `writes_all_elems(buf)`                `WritesAllElems blocks blockSize outLen`
  `no_data_race(buf)`                    `NoDataRace blockDimX blockDimY a₁ a₂`
  `uniform_barriers`                     `UniformBarriers barriersUniform numBarriers`
  `tmem_lifecycle`                       `TmemLifecycle allocCols deallocCols balanced`
```

Each proposition states a mathematical obligation over the kernel's exported
`CudaOxide.Mir` model. Reusable proofs live in `CudaOxide.Proofs`.
-/
namespace CudaOxide.Contracts
open CudaOxide.Mir
open CudaOxide.Mir.Kernel

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

/-- **`writes_all_elems(buf)`**: Every output element index `i < outLen` in `buf`
    is written by some launched thread `(block, lane)` in the 1D grid. -/
def WritesAllElems (blocks blockSize outLen : Nat) : Prop :=
  ∀ i : Nat,                                              -- Pick ANY output element index `i`:
    i < outLen →                                          -- IF `i` is within the output length (`i < outLen`),
    ∃ block lane : Nat,                                   -- THEN there EXISTS a thread (block, lane) such that:
      block < blocks ∧                                    --   1. `block` is a valid launched block, AND
      lane < blockSize ∧                                  --   2. `lane` is a valid lane in that block, AND
      globalIndex blockSize block lane = i                --   3. that thread's global index equals `i`.

/-- **`no_oob_idx(TILE)`**: Every 2D thread `(tx, ty)` and loop step `i < loopBound`
    indexes `TILE` strictly below `bufferLen`. -/
def NoOobTileAccess (blockDimX blockDimY loopBound : Nat)
    (pat : TileAccessPattern) (bufferLen : Nat) : Prop :=
  ∀ tx ty i : Nat,                                        -- Pick ANY 2D thread (tx, ty) and loop step `i`:
    tx < blockDimX →                                      -- IF `tx` is within the block X dimension,
    ty < blockDimY →                                      -- AND `ty` is within the block Y dimension,
    i < loopBound →                                       -- AND `i` is within the loop bound,
    pat.address tx ty i < bufferLen                       -- THEN the tile index is strictly `< bufferLen`.

/-- **`no_oob_idx(TILE)` (with branch guard)**: Every 2D thread `(tx, ty)` and
    loop step `i < loopBound` that satisfies `guard` indexes `TILE` strictly
    below `bufferLen`. -/
def NoOobGuardedTileAccess (blockDimX blockDimY loopBound : Nat)
    (guard : GuardExpr) (pat : TileAccessPattern) (bufferLen : Nat) : Prop :=
  ∀ tx ty i : Nat,                                        -- Pick ANY 2D thread (tx, ty) and loop step `i`:
    tx < blockDimX →                                      -- IF `tx` is within the block X dimension,
    ty < blockDimY →                                      -- AND `ty` is within the block Y dimension,
    i < loopBound →                                       -- AND `i` is within the loop bound,
    guard.eval tx ty i →                                  -- AND `(tx, ty, i)` satisfies the branch guard,
    pat.address tx ty i < bufferLen                       -- THEN the tile index is strictly `< bufferLen`.

/-- **`no_data_race(buf)`**: Two accesses `a₁` and `a₂` on the same buffer never
    cause a write-write or read-write conflict between distinct active threads in
    the same barrier phase. -/
def NoDataRace (blockDimX blockDimY : Nat)
    (a₁ a₂ : PhasedTileAccess) : Prop :=
  ∀ tx₁ ty₁ i₁ tx₂ ty₂ i₂ : Nat,                            -- Pick ANY two threads (tx₁, ty₁), (tx₂, ty₂) and steps i₁, i₂:
    tx₁ < blockDimX → ty₁ < blockDimY → i₁ < a₁.loopBound → -- IF thread 1 and step i₁ are within bounds,
    tx₂ < blockDimX → ty₂ < blockDimY → i₂ < a₂.loopBound → -- AND thread 2 and step i₂ are within bounds,
    a₁.active tx₁ → a₂.active tx₂ →                         -- AND both threads satisfy their access guard (e.g. `lane == 0`),
    a₁.bufferId = a₂.bufferId →                             -- AND both accesses target the same buffer,
    a₁.phase = a₂.phase →                                   -- AND both accesses execute in the SAME barrier phase,
    (a₁.kind = .write ∨ a₂.kind = .write) →                 -- AND at least one of the accesses is a write,
    a₁.pattern.address tx₁ ty₁ i₁ =                         -- AND both threads compute the exact same
      a₂.pattern.address tx₂ ty₂ i₂ →                       --     element address in that buffer,
    tx₁ = tx₂ ∧ ty₁ = ty₂                                   -- THEN they must be the exact same thread (`∧` = AND).

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

/-- **`uniform_barriers`**: All `sync_threads()` barriers in the kernel execute
    in thread-uniform control flow. -/
def UniformBarriers (barriersUniform : Bool) (numBarriers : Nat) : Prop :=
  barriersUniform = true ∧                                -- Every barrier is outside thread-divergent branches, AND
  0 < numBarriers                                         -- at least one barrier synchronizes the phases.

/-- **`no_oob_idx(a)` (2D row strip)**: Every element `start + col + offset` with
    `col < cols` is `< bufferLen` when the strip's last element
    `start + (cols - 1) < bufferLen` (and `offset = 0`). -/
def NoOobRowStrip (bufferLen : Nat) (offset : Nat := 0) : Prop :=
  ∀ start cols col : Nat,                                 -- Pick ANY row strip start, width `cols`, and offset `col`:
    0 < cols →                                            -- IF the strip is non-empty (`cols > 0`),
    start + (cols - 1) < bufferLen →                      -- AND the strip's last element fits in `bufferLen`,
    col < cols →                                          -- AND the thread's column offset `col < cols`,
    start + col + offset < bufferLen                      -- THEN `start + col + offset` is strictly `< bufferLen`.

/-- **`no_oob_idx(b)` (2D column strip)**: Every strided element
    `start + row * stride + offset` with `row < rows` is `< bufferLen` when the
    strip's last element `(rows - 1) * stride + start < bufferLen` (and `offset = 0`). -/
def NoOobColStrip (bufferLen : Nat) (offset : Nat := 0) : Prop :=
  ∀ start stride rows row : Nat,                          -- Pick ANY col strip start, row stride, height `rows`, and `row`:
    0 < rows →                                            -- IF the strip is non-empty (`rows > 0`),
    (rows - 1) * stride + start < bufferLen →             -- AND the strip's last strided element fits in `bufferLen`,
    row < rows →                                          -- AND the thread's row offset `row < rows`,
    start + row * stride + offset < bufferLen             -- THEN `start + row * stride + offset` is strictly `< bufferLen`.

/-- Valid non-exclusive Blackwell `tcgen05.alloc` column counts (PTX ISA Table 58:
    power of two in `[32, 512]`). -/
def TmemValidCols (nCols : Nat) : Prop :=
  nCols = 32 ∨                                            -- 32 columns (minimum allocation unit), OR
  nCols = 64 ∨                                            -- 64 columns, OR
  nCols = 128 ∨                                           -- 128 columns, OR
  nCols = 256 ∨                                           -- 256 columns, OR
  nCols = 512                                             -- 512 columns (full per-SM TMEM capacity).

/-- **`tmem_lifecycle`**: Blackwell Tensor Memory (`TMEM`) is allocated with a
    valid power-of-two column count in `[32, 512]`, token-ordered across async
    `tcgen05` operations, and deallocated with the exact same column count on
    every kernel exit path. -/
def TmemLifecycle (allocCols deallocCols : Nat) (balanced : Bool) : Prop :=
  TmemValidCols allocCols ∧                               -- Allocated column count is a power of two in [32, 512], AND
  allocCols = deallocCols ∧                               -- `tcgen05.dealloc` frees the exact `allocCols`, AND
  balanced = true                                         -- Alloc/commit/wait/dealloc CFG lifecycle holds on all exit paths.

/-- **`tmem_lifecycle` / `no_oob_idx` (`TMEM` bounds & warpgroup slice)**:
    Every active thread `(tx, ty)` and loop step `i < a.loopBound` accesses
    `TMEM` strictly within `128` lanes, within `allocCols ≤ 512` columns, and
    (for `tcgen05.ld` / `tcgen05.st`) within its warp's 32-lane warpgroup slice. -/
def NoOobTmemAccess (blockDimX blockDimY allocCols : Nat)
    (a : TmemAccess) : Prop :=
  ∀ tx ty i : Nat,                                        -- Pick ANY 2D thread (tx, ty) and loop step `i`:
    tx < blockDimX →                                      -- IF `tx` is within the block X dimension,
    ty < blockDimY →                                      -- AND `ty` is within the block Y dimension,
    i < a.loopBound →                                     -- AND `i` is within the loop bound,
    a.guard.eval tx ty i →                                -- AND `(tx, ty, i)` satisfies the dominating branch guard,
    a.laneExpr.eval tx ty i + a.heightLanes ≤ 128 ∧       -- THEN (1) lane span fits inside the 128 TMEM lanes, AND
    a.colExpr.eval tx ty i + a.widthCols ≤ allocCols ∧    --      (2) column span fits inside `allocCols`, AND
    allocCols ≤ 512 ∧                                     --      (3) `allocCols` fits inside the 512-column SM limit, AND
    (a.warpRestricted = true →                            --      (4) for warp-restricted `tcgen05.ld` / `tcgen05.st` ops,
      ((tx / 32) % 4) * 32 ≤ a.laneExpr.eval tx ty i ∧    --          starting lane ≥ this warp's 32-lane slice base, AND
      a.laneExpr.eval tx ty i + a.heightLanes ≤           --          ending lane ≤ this warp's 32-lane slice ceiling
        ((tx / 32) % 4) * 32 + 32)                        --          (PTX ISA 9.7.18.8.1 warpgroup lane restriction).

end CudaOxide.Contracts
