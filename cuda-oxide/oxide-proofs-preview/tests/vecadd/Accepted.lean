/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide

-- Instantiates reusable facts for the access contracts of guarded vecadd:
-- `no_oob_idx(a, b, c)`, `disjoint_writes(c)`, and `writes_all_elems(c)`.
namespace Accepted
open CudaOxide.Mir CudaOxide.Contracts CudaOxide.Proofs.Bounds CudaOxide.Proofs.Launch

theorem no_oob_idx_all {aLen bLen outLen blockSize : Nat}
    (ha : outLen ≤ aLen) (hb : outLen ≤ bLen) :
    NoOobIdx blockSize 0 outLen aLen ∧
    NoOobIdx blockSize 0 outLen bLen ∧
    NoOobIdx blockSize 0 outLen outLen := by
  exact ⟨no_oob_idx_of_guard_le ha, no_oob_idx_of_guard_le hb, no_oob_idx_of_same_len⟩

theorem disjoint_writes_c {blockSize : Nat} :
    DisjointWrites blockSize 0 := by
  exact disjoint_writes_1d

theorem writes_all_elems_c {blocks blockSize outLen : Nat}
    (hpositive : 0 < blockSize) (hlaunch : outLen ≤ blocks * blockSize) :
    WritesAllElems blocks blockSize outLen := by
  exact writes_all_elems_1d hpositive hlaunch

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

#print axioms no_oob_idx_all
#print axioms disjoint_writes_c
#print axioms writes_all_elems_c
#print axioms index_lt_of_guard
#print axioms no_oob_idx_of_guard_le
#print axioms no_oob_idx_of_same_len
#print axioms launched_index_bound
#print axioms launched_index_fits
#print axioms launched_index_not_sentinel
#print axioms globalIndex_injective
#print axioms disjoint_writes_1d
#print axioms launch_coverage
#print axioms writes_all_elems_1d
#print axioms thread_index_bound
#print axioms thread_address
#print axioms thread_index_injective
#print axioms CudaOxide.Proofs.ControlFlow.uniform_barriers_of_stateMap
#print axioms CudaOxide.Proofs.ControlFlow.tmem_lifecycle_of_stateMap
#print axioms CudaOxide.Proofs.ControlFlow.writes_on_active_paths_sound
#print axioms CudaOxide.Proofs.Program.symReachable_sound
#print axioms CudaOxide.Proofs.Program.vcNoOobGlobal_of_table
#print axioms CudaOxide.Proofs.Program.vcNoOobShared_of_table
#print axioms CudaOxide.Proofs.Program.vcDisjointWrites_of_table
#print axioms CudaOxide.Proofs.Program.vcNoSharedRace_of_table
#print axioms CudaOxide.Proofs.Program.vcAssertRedundant_of_table
#print axioms CudaOxide.Proofs.Program.vcNoOobTmem_of_table
#print axioms CudaOxide.Proofs.Program.or_eq_add_of_mod_two_pow

end Accepted
