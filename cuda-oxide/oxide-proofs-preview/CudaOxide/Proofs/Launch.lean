/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide.Mir.Launch
import CudaOxide.Contracts.Access

/-!
# Reusable 1D Launch & Coordinate Lemmas (`CudaOxide.Proofs.Launch`)

Reusable theorems over `CudaOxide.Mir.globalIndex` and `Thread1D` proving
linear-index bounds, injectivity, and grid coverage.
-/
namespace CudaOxide.Proofs.Launch
open CudaOxide.Mir CudaOxide.Contracts

/-- If `block < blocks` and `lane < blockSize`, then the linear index
    `block * blockSize + lane` is strictly below `blocks * blockSize`. -/
theorem launched_index_bound {blocks blockSize block lane : Nat}
    (hblock : block < blocks) (hlane : lane < blockSize) :
    globalIndex blockSize block lane < blocks * blockSize := by
  have hstep : block + 1 ≤ blocks := by omega             -- Step 1: `block < blocks` on Nat means `block + 1 ≤ blocks`
  have hmul := Nat.mul_le_mul_right blockSize hstep       -- Step 2: Multiply both sides by `blockSize`
  simp only [Nat.add_mul, Nat.one_mul] at hmul            -- Step 3: Expand `(block + 1) * blockSize` to `block * blockSize + blockSize`
  unfold globalIndex                                      -- Step 4: Replace `globalIndex` in goal with `block * blockSize + lane`
  omega                                                   -- Step 5: Linear solver finishes using `hlane : lane < blockSize`

/-- If `block < blocks`, `lane < blockSize`, and the total thread count fits
    inside `limit` (`blocks * blockSize ≤ limit`), then `block * blockSize + lane < limit`. -/
theorem launched_index_fits {blocks blockSize block lane limit : Nat}
    (hblock : block < blocks) (hlane : lane < blockSize)
    (hlimit : blocks * blockSize ≤ limit) :
    globalIndex blockSize block lane < limit := by
  exact Nat.lt_of_lt_of_le (launched_index_bound hblock hlane) hlimit -- Chain `idx < blocks * blockSize` with `≤ limit`

/-- If the total thread count fits inside `sentinel` (`blocks * blockSize ≤ sentinel`),
    no valid thread's index can ever equal `sentinel`. -/
theorem launched_index_not_sentinel {blocks blockSize block lane sentinel : Nat}
    (hblock : block < blocks) (hlane : lane < blockSize)
    (hlimit : blocks * blockSize ≤ sentinel) :
    globalIndex blockSize block lane ≠ sentinel := by
  exact Nat.ne_of_lt (launched_index_fits hblock hlane hlimit)        -- Strict `< sentinel` implies `≠ sentinel`

/-- Two threads with the same linear index (`block₁ * B + lane₁ = block₂ * B + lane₂`)
    must have the exact same `block` and `lane`. -/
theorem globalIndex_injective {blockSize block₁ block₂ lane₁ lane₂ : Nat}
    (hlane₁ : lane₁ < blockSize) (hlane₂ : lane₂ < blockSize)
    (hsame : globalIndex blockSize block₁ lane₁ =
      globalIndex blockSize block₂ lane₂) :
    block₁ = block₂ ∧ lane₁ = lane₂ := by
  have hpositive : 0 < blockSize := by omega              -- Step 1: Since `0 ≤ lane₁ < blockSize`, `blockSize > 0`
  have hlane : lane₁ = lane₂ := by                        -- Step 2: Prove `lane₁ = lane₂` by taking `% blockSize`
    have hmod := congrArg (fun n => n % blockSize) hsame  --   Apply `% blockSize` to both sides of `hsame`
    simpa [globalIndex, Nat.mul_add_mod_self_right,       --   `(block * blockSize + lane) % blockSize` simplifies
      Nat.mod_eq_of_lt hlane₁, Nat.mod_eq_of_lt hlane₂] using hmod -- to `lane` because `lane < blockSize`
  have hblock : block₁ = block₂ := by                     -- Step 3: Prove `block₁ = block₂` by taking `/ blockSize`
    have hdiv := congrArg (fun n => n / blockSize) hsame  --   Apply `/ blockSize` to both sides of `hsame`
    simpa [globalIndex, Nat.mul_comm block₁ blockSize,    --   `(block * blockSize + lane) / blockSize` simplifies
      Nat.mul_comm block₂ blockSize, Nat.mul_add_div hpositive,
      Nat.div_eq_of_lt hlane₁, Nat.div_eq_of_lt hlane₂] using hdiv -- to `block + 0 = block`
  exact ⟨hblock, hlane⟩                                   -- Step 4: Pair `block₁ = block₂` and `lane₁ = lane₂`

/-- Proves `disjoint_writes(buf)`: two different 1D threads writing to `idx + offset`
    never write to the same buffer element. -/
theorem disjoint_writes_1d {blockSize offset : Nat} :
    DisjointWrites blockSize offset := by
  intro block₁ block₂ lane₁ lane₂ hlane₁ hlane₂ hsame     -- Unpack the `∀` variables and `→` premises of `DisjointWrites`
  have hbase :                                            -- Cancel `+ offset` from both sides of `idx₁ + offset = idx₂ + offset`
      globalIndex blockSize block₁ lane₁ =
        globalIndex blockSize block₂ lane₂ := Nat.add_right_cancel hsame
  exact globalIndex_injective hlane₁ hlane₂ hbase         -- Apply injectivity of `globalIndex`

/-- Proves `DisjointSliceAccessPair` when a thread reads and writes its own exact
    slot (`offset₁ = offset₂ = offset`, e.g. `c[idx] += a[idx]` on `c: &mut [T]`). -/
theorem disjoint_slice_pair_same_offset {blockSize offset : Nat} :
    DisjointSliceAccessPair blockSize offset offset :=
  disjoint_writes_1d

/-- When `outLen ≤ blocks * blockSize`, every output element `i < outLen` is
    covered by thread `(block = i / blockSize, lane = i % blockSize)`. -/
theorem launch_coverage {blocks blockSize outLen i : Nat}
    (hpositive : 0 < blockSize) (hcoverage : outLen ≤ blocks * blockSize)
    (hi : i < outLen) :
    ∃ block lane : Nat, block < blocks ∧ lane < blockSize ∧
      globalIndex blockSize block lane = i := by
  refine ⟨i / blockSize, i % blockSize, ?_, ?_, ?_⟩       -- Witness `block := i / blockSize` and `lane := i % blockSize`
  · apply (Nat.div_lt_iff_lt_mul hpositive).2             -- Goal 1 (`i / blockSize < blocks`): rewrite as `i < blocks * blockSize`
    omega                                                 --   Follows from `i < outLen ≤ blocks * blockSize`
  · exact Nat.mod_lt i hpositive                          -- Goal 2 (`i % blockSize < blockSize`): remainder is `< divisor`
  · simpa [globalIndex, Nat.mul_comm] using Nat.div_add_mod i blockSize -- Goal 3: `(i / B) * B + (i % B) = i`

/-- Proves `writes_all_elems(buf)`: every element `i < outLen` is written by
    some thread when the grid has at least `outLen` threads. -/
theorem writes_all_elems_1d {blocks blockSize outLen : Nat}
    (hpositive : 0 < blockSize) (hcoverage : outLen ≤ blocks * blockSize) :
    WritesAllElems blocks blockSize outLen := by
  intro i hi                                              -- Unpack `∀ i, i < outLen → ...` from `WritesAllElems`
  exact launch_coverage hpositive hcoverage hi            -- Discharge with `launch_coverage`

/-- Every typed `Thread1D` has a linear index `< blocks * blockSize`. -/
theorem thread_index_bound {blocks blockSize : Nat}
    (thread : Thread1D blocks blockSize) :
    thread.index < blocks * blockSize := by
  exact launched_index_bound thread.block.isLt thread.lane.isLt -- `.isLt` extracts `val < n` from each `Fin n` field

/-- Bundle a `Thread1D`'s linear index with its `< blocks * blockSize` proof as a `Fin`. -/
def thread_address {blocks blockSize : Nat}
    (thread : Thread1D blocks blockSize) : Fin (blocks * blockSize) :=
  ⟨thread.index, thread_index_bound thread⟩

/-- Two `Thread1D` values with the same linear index are the exact same thread. -/
theorem thread_index_injective {blocks blockSize : Nat}
    (first second : Thread1D blocks blockSize)
    (hsame : first.index = second.index) : first = second := by
  rcases first with ⟨firstBlock, firstLane⟩               -- Destructure `first` into its `(block, lane)` fields
  rcases second with ⟨secondBlock, secondLane⟩            -- Destructure `second` into its `(block, lane)` fields
  have hcoords := globalIndex_injective firstLane.isLt secondLane.isLt hsame -- Prove underlying Nat coordinates equal
  have hblock : firstBlock = secondBlock := Fin.ext hcoords.1 -- Lift Nat equality to `Fin blocks` equality
  have hlane : firstLane = secondLane := Fin.ext hcoords.2    -- Lift Nat equality to `Fin blockSize` equality
  cases hblock                                            -- Substitute `secondBlock` with `firstBlock`
  cases hlane                                             -- Substitute `secondLane` with `firstLane`
  rfl                                                     -- Both sides are now identical (`⟨firstBlock, firstLane⟩`)

/-- Two numbers with the same quotient `a / W = b / W` and remainder `0`
    (`a % W = 0` and `b % W = 0`) are equal. -/
theorem eq_of_div_eq_of_mod_zero {a b groupSize : Nat}
    (hmod₁ : a % groupSize = 0) (hmod₂ : b % groupSize = 0)
    (hdiv : a / groupSize = b / groupSize) : a = b := by
  have ha := Nat.div_add_mod a groupSize                  -- `groupSize * (a / groupSize) + a % groupSize = a`
  have hb := Nat.div_add_mod b groupSize                  -- `groupSize * (b / groupSize) + b % groupSize = b`
  rw [hmod₁, Nat.add_zero] at ha                          -- Simplify `ha` to `groupSize * (a / groupSize) = a`
  rw [hmod₂, Nat.add_zero, ← hdiv] at hb                  -- Substitute `b % W = 0` and `a / W = b / W` into `hb`
  exact ha.symm.trans hb                                  -- Chain `a = W * (a / W) = b`

/-- Proves `disjoint_writes(sums)` for warp/group leaders: two 1D threads that
    both satisfy `idx % groupSize = 0` and write to `idx / groupSize` are the
    exact same thread. -/
theorem disjoint_group_writes_1d {blockSize groupSize : Nat} :
    DisjointGroupWrites blockSize groupSize := by
  intro block₁ block₂ lane₁ lane₂ hlane₁ hlane₂ hmod₁ hmod₂ hdiv -- Unpack leader guards and `idx₁ / W = idx₂ / W`
  have hsame := eq_of_div_eq_of_mod_zero hmod₁ hmod₂ hdiv -- Reconstruct `idx₁ = idx₂` from quotient and remainder `0`
  exact globalIndex_injective hlane₁ hlane₂ hsame         -- Apply injectivity of `globalIndex`

end CudaOxide.Proofs.Launch
