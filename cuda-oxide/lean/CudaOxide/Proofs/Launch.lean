/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import Lean
import CudaOxide.Model.Launch

/-!
Reusable launch-index proofs for one-dimensional access models.

These theorems describe an explicit mathematical index model. They do not
establish that a Rust function, exported IR, or PTX implements that model.
No claim is made about floating-point arithmetic or pointer aliasing.
-/
namespace CudaOxide.Proofs.Launch
open CudaOxide.Model

/-- The global index of a launched lane is below the total thread count. -/
theorem launched_index_bound {blocks blockSize block lane : Nat}
    (hblock : block < blocks) (hlane : lane < blockSize) :
    globalIndex blockSize block lane < blocks * blockSize := by
  have hstep : block + 1 ≤ blocks := by omega
  have hmul := Nat.mul_le_mul_right blockSize hstep
  simp only [Nat.add_mul, Nat.one_mul] at hmul
  unfold globalIndex
  omega

/-- The bound is explicit. For cuda-oxide index_1d on 64-bit usize, use
    `limit = 2^64 - 1`: the helper reserves usize::MAX as an invalid sentinel.
    The weaker `limit = 2^64` proves arithmetic fit but allows that sentinel.
    Neither instantiation establishes the source-to-model connection. -/
theorem launched_index_fits {blocks blockSize block lane limit : Nat}
    (hblock : block < blocks) (hlane : lane < blockSize)
    (hlimit : blocks * blockSize ≤ limit) :
    globalIndex blockSize block lane < limit := by
  exact Nat.lt_of_lt_of_le (launched_index_bound hblock hlane) hlimit

/-- A strict bound below the reserved index value also excludes the
    invalid sentinel used by the current cuda-oxide index_1d helper. -/
theorem launched_index_not_sentinel {blocks blockSize block lane sentinel : Nat}
    (hblock : block < blocks) (hlane : lane < blockSize)
    (hlimit : blocks * blockSize ≤ sentinel) :
    globalIndex blockSize block lane ≠ sentinel := by
  exact Nat.ne_of_lt (launched_index_fits hblock hlane hlimit)

/-- Distinct launched (block,lane) pairs cannot share the same global index.
    Bounding both lanes is essential. -/
theorem globalIndex_injective {blockSize block₁ block₂ lane₁ lane₂ : Nat}
    (hlane₁ : lane₁ < blockSize) (hlane₂ : lane₂ < blockSize)
    (hsame : globalIndex blockSize block₁ lane₁ =
      globalIndex blockSize block₂ lane₂) :
    block₁ = block₂ ∧ lane₁ = lane₂ := by
  have hpositive : 0 < blockSize := by omega
  have hlane : lane₁ = lane₂ := by
    have hmod := congrArg (fun n => n % blockSize) hsame
    simpa [globalIndex, Nat.mul_add_mod_self_right,
      Nat.mod_eq_of_lt hlane₁, Nat.mod_eq_of_lt hlane₂] using hmod
  have hblock : block₁ = block₂ := by
    have hdiv := congrArg (fun n => n / blockSize) hsame
    simpa [globalIndex, Nat.mul_comm block₁ blockSize,
      Nat.mul_comm block₂ blockSize, Nat.mul_add_div hpositive,
      Nat.div_eq_of_lt hlane₁, Nat.div_eq_of_lt hlane₂] using hdiv
  exact ⟨hblock, hlane⟩

/-- Sufficient launch coverage: every output index belongs to a launched
    block and lane. This is a separate obligation from unique writes. -/
theorem launch_coverage {blocks blockSize outLen i : Nat}
    (hpositive : 0 < blockSize) (hcoverage : outLen ≤ blocks * blockSize)
    (hi : i < outLen) :
    ∃ block lane : Nat, block < blocks ∧ lane < blockSize ∧
      globalIndex blockSize block lane = i := by
  refine ⟨i / blockSize, i % blockSize, ?_, ?_, ?_⟩
  · apply (Nat.div_lt_iff_lt_mul hpositive).2
    omega
  · exact Nat.mod_lt i hpositive
  · simpa [globalIndex, Nat.mul_comm] using Nat.div_add_mod i blockSize

/-- A typed thread supplies its launch-bounds proofs from its fields. -/
theorem thread_index_bound {blocks blockSize : Nat}
    (thread : Thread1D blocks blockSize) :
    thread.index < blocks * blockSize := by
  exact launched_index_bound thread.block.isLt thread.lane.isLt

/-- Convert a launched thread into an address carrying the proved total bound.
    This is an address in the padded launch domain, not necessarily the output:
    a separate `index < outLen` guard is still required for output accesses. -/
def thread_address {blocks blockSize : Nat}
    (thread : Thread1D blocks blockSize) : Fin (blocks * blockSize) :=
  ⟨thread.index, thread_index_bound thread⟩

/-- Equal logical indices identify the same typed thread. -/
theorem thread_index_injective {blocks blockSize : Nat}
    (first second : Thread1D blocks blockSize)
    (hsame : first.index = second.index) : first = second := by
  rcases first with ⟨firstBlock, firstLane⟩
  rcases second with ⟨secondBlock, secondLane⟩
  have hcoords := globalIndex_injective firstLane.isLt secondLane.isLt hsame
  have hblock : firstBlock = secondBlock := Fin.ext hcoords.1
  have hlane : firstLane = secondLane := Fin.ext hcoords.2
  cases hblock
  cases hlane
  rfl

end CudaOxide.Proofs.Launch
