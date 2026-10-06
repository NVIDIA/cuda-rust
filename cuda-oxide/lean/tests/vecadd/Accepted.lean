/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide.Proofs.Bounds
import CudaOxide.Proofs.Launch

-- This fixture instantiates reusable facts for the mathematical access
-- obligations of guarded vecadd. It illustrates the shape of future generated
-- obligations; the intended verifier will generate these applications.
-- This fixture does not parse Rust or connect the model to an actual kernel.
namespace Accepted
open CudaOxide.Model CudaOxide.Proofs.Bounds CudaOxide.Proofs.Launch

theorem bounds {aLen bLen outLen blockSize block lane : Nat}
    (ha : outLen ≤ aLen) (hb : outLen ≤ bLen)
    (active : globalIndex blockSize block lane < outLen) :
    globalIndex blockSize block lane < aLen ∧
    globalIndex blockSize block lane < bLen ∧
    globalIndex blockSize block lane < outLen := by
  exact ⟨index_lt_of_guard active ha, index_lt_of_guard active hb, active⟩

theorem uniqueWrites {blockSize block₁ block₂ lane₁ lane₂ : Nat}
    (hlane₁ : lane₁ < blockSize) (hlane₂ : lane₂ < blockSize)
    (hsame : globalIndex blockSize block₁ lane₁ =
      globalIndex blockSize block₂ lane₂) :
    block₁ = block₂ ∧ lane₁ = lane₂ := by
  exact globalIndex_injective hlane₁ hlane₂ hsame

theorem coverage {blocks blockSize outLen i : Nat}
    (hpositive : 0 < blockSize) (hlaunch : outLen ≤ blocks * blockSize)
    (hi : i < outLen) :
    ∃ block lane : Nat, block < blocks ∧ lane < blockSize ∧
      globalIndex blockSize block lane = i := by
  exact launch_coverage hpositive hlaunch hi

-- Construct a thread using checked block and lane coordinates.
def lastThread : Thread1D 4 256 :=
  ⟨⟨3, by omega⟩, ⟨255, by omega⟩⟩

example : lastThread.index = 1023 := by
  rfl

example : (thread_address lastThread).val = 1023 := by
  rfl

-- Padding does not prevent the launch from covering all 1000 outputs.
example (i : Nat) (hi : i < 1000) :
    ∃ block lane : Nat, block < 4 ∧ lane < 256 ∧
      globalIndex 256 block lane = i := by
  exact launch_coverage (by omega) (by omega) hi

#print axioms bounds
#print axioms uniqueWrites
#print axioms coverage
#print axioms index_lt_of_guard
#print axioms launched_index_bound
#print axioms launched_index_fits
#print axioms launched_index_not_sentinel
#print axioms globalIndex_injective
#print axioms launch_coverage
#print axioms thread_index_bound
#print axioms thread_address
#print axioms thread_index_injective

end Accepted
