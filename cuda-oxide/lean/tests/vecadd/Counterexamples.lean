/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import Lean
import CudaOxide.Model.Launch

/-! Proved counterexamples for broken mathematical access patterns. -/
namespace Counterexamples
open CudaOxide.Model

/-- If the output guard uses `≤` instead of `<`, its boundary lane is invalid. -/
theorem broken_inclusive_guard :
    (4 : Nat) ≤ 4 ∧ ¬ ((4 : Nat) < 4) := by
  decide

/-- Ignoring blockId creates duplicate writers: block 0 lane 0 and
    block 1 lane 0 are different launched threads writing the same index. -/
def brokenLaneOnlyIndex (_blockSize _block lane : Nat) : Nat := lane

theorem broken_lane_only_index :
    (0 : Nat) < 2 ∧ (1 : Nat) < 2 ∧ (0 : Nat) < 4 ∧
    (0 : Nat) ≠ 1 ∧
    brokenLaneOnlyIndex 4 0 0 = brokenLaneOnlyIndex 4 1 0 := by
  decide

/-- Merely using the lane bound without an output guard is unsafe for a
    padded launch: the last lane of 4 blocks of 256 exceeds length 1000. -/
theorem broken_missing_guard :
    (3 : Nat) < 4 ∧ (255 : Nat) < 256 ∧
    ¬ (globalIndex 256 3 255 < 1000) := by
  decide

/-- A guarded output access can still read beyond a shorter input. -/
theorem broken_missing_length :
    (4 : Nat) < 5 ∧ ¬ ((4 : Nat) < 4) := by
  decide

/-- Two blocks of two lanes cannot cover index 4 of an output of length 5. -/
theorem broken_insufficient_coverage :
    (4 : Nat) < 5 ∧ ¬ (∃ block lane : Nat,
      block < 2 ∧ lane < 2 ∧ globalIndex 2 block lane = 4) := by
  constructor
  · decide
  · rintro ⟨block, lane, hblock, hlane, hindex⟩
    unfold globalIndex at hindex
    omega

#print axioms broken_inclusive_guard
#print axioms broken_lane_only_index
#print axioms broken_missing_guard
#print axioms broken_missing_length
#print axioms broken_insufficient_coverage

end Counterexamples
