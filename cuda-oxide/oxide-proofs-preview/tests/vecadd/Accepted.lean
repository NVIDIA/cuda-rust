/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide

set_option maxRecDepth 65536

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

-- Reverse-indexed 1D write `if idx < c.len() { c[c.len() - 1 - idx] = ... }` satisfies `MirWritesAllElems`.
def revMirFunc : CudaOxide.Mir.Program.MirFunc :=
  { domain := 1
  , blockDimX := 256
  , blockDimY := 1
  , params := [(.slice false), (.slice true)]
  , requires := [{ cmp := .ge, lhs := .sliceLen 0, rhs := .sliceLen 1 }]
  , blocks := [
      { args := [⟨0, 0⟩, ⟨1, 0⟩], stmts := [
          (.assign 2 .index1D),
          (.assign 3 (.sliceLen 1)),
          (.assign 4 (.cmpOp .lt 2 3))
        ], term := (.condBr 4 1 [] 2 []) },
      { stmts := [
          (.assign 5 (.sliceLen 0)),
          (.assign 6 (.cmpOp .lt 2 5)),
          (.assertCond 0 6),
          (.assign 7 (.constInt 64 1)),
          (.assign 8 (.binOp .sub 64 3 7)),
          (.assign 9 (.binOp .sub 64 8 2)),
          (.assign 10 (.slicePtr 1)),
          (.assign 11 (.ptrOffset 10 9)),
          (.assign 12 .opaqueVal),
          (.store 11 12)
        ], term := (.goto 2 []) },
      { stmts := [], term := .ret }
    ] }

theorem rev_writes_all_elems :
    CudaOxide.Contracts.Program.MirWritesAllElems revMirFunc 1 := by
  apply CudaOxide.Proofs.Program.vcWritesAllElems rfl (by mir_bind_writes)
  mir_vc revMirFunc

/-- Thread `0` arms `BAR` and starts a TMA copy into `TILE` (block `3`), every thread waits on `BAR`
    (block `4`), then reads `TILE` (block `5`): the shape of `examples/safe_async_copy`.
    `RejectedTmaReadAfterSync.lean` replaces the wait with `sync_threads()`. -/
def tmaWaitCfg : ControlFlow.ControlFlowCfg :=
  { blockDimX := 256, blockDimY := 1, numShared := 1, numBarriers := 1
  , blocks := [
      { succs := [(.coord (.eq .tx (.const 0)), 1), (.coord (.ne .tx (.const 0)), 2)] },
      { ops := [.mbarInit 0 1, .proxyFenceAsyncShared], succs := [(.always, 2)] },
      { ops := [.syncThreads]
      , succs := [(.coord (.eq .tx (.const 0)), 3), (.coord (.ne .tx (.const 0)), 4)] },
      { ops := [.mbarArriveExpectTx 0 16384, .tmaCopyG2S 0 0], succs := [(.always, 4)] },
      { succs := [(.mbarSpin 0, 4), (.mbarReady 0 false, 5)] },
      { ops := [.sharedRead 0], succs := [(.always, 6)] },
      { isReturn := true } ] }

theorem tma_read_after_wait : ControlFlow.UniformBarriersCfg tmaWaitCfg :=
  CudaOxide.Proofs.ControlFlow.uniform_barriers_cfg_of_stateMap
    (inv := CudaOxide.Proofs.ControlFlow.inferStateMap tmaWaitCfg) (by decide)

/-- Scaled-extent 1D gather `if idx < out.len() { out[idx] = x[4 * idx + 3] }` under
    `requires = (x.len() >= 4 * out.len())` (`#1869` shape, including 64-bit `checked_mul`/`checked_add`). -/
def quadGatherMirFunc : CudaOxide.Mir.Program.MirFunc :=
  { domain := 1
  , blockDimX := 256
  , blockDimY := 1
  , params := [(.slice false), (.slice true)]
  , requires := [
      { cmp := .ge
      , lhs := .sliceLen 0
      , rhs := .mul (.const 4) (.sliceLen 1) } ]
  , blocks := [
      { args := [⟨0, 0⟩, ⟨1, 0⟩], stmts := [
          (.assign 2 .index1D),
          (.assign 3 (.sliceLen 1)),
          (.assign 4 (.cmpOp .lt 2 3))
        ], term := (.condBr 4 1 [] 2 []) },
      { stmts := [
          (.assign 5 (.constInt 64 4)),
          (.assign 6 (.binOp .mul 64 5 2)),
          (.assign 7 (.checkedOvf .mul 64 5 2)),
          (.assign 8 (.notBool 7)),
          (.assertCond 0 8),
          (.assign 9 (.constInt 64 3)),
          (.assign 10 (.binOp .add 64 6 9)),
          (.assign 11 (.checkedOvf .add 64 6 9)),
          (.assign 12 (.notBool 11)),
          (.assertCond 1 12),
          (.assign 13 (.sliceLen 0)),
          (.assign 14 (.cmpOp .lt 10 13)),
          (.assertCond 2 14),
          (.assign 15 (.slicePtr 0)),
          (.assign 16 (.ptrOffset 15 10)),
          (.assign 17 (.load 16)),
          (.assign 18 (.slicePtr 1)),
          (.assign 19 (.ptrOffset 18 2)),
          (.store 19 17)
        ], term := (.goto 2 []) },
      { stmts := [], term := .ret }
    ] }

mir_def_sym_table quadGatherMirFunc

theorem quadGather_valid :
    CudaOxide.Proofs.Program.validSymFunc quadGatherMirFunc symTable = true := rfl

theorem quadGather_events :
    CudaOxide.Proofs.Program.collectSymMemEvents quadGatherMirFunc symTable = symMemEvents := rfl

theorem quadGather_asserts :
    CudaOxide.Proofs.Program.collectSymAssertEvents quadGatherMirFunc symTable = symAssertEvents := rfl

theorem quadGather_no_oob_x :
    CudaOxide.Contracts.Program.MirNoOobGlobal quadGatherMirFunc 0 := by
  apply CudaOxide.Proofs.Program.vcNoOobGlobal_of_table quadGather_valid quadGather_events (by mir_bind_events)
  mir_vc quadGatherMirFunc

theorem quadGather_asserts_redundant :
    CudaOxide.Contracts.Program.MirAssertRedundant quadGatherMirFunc 0 ∧
    CudaOxide.Contracts.Program.MirAssertRedundant quadGatherMirFunc 1 ∧
    CudaOxide.Contracts.Program.MirAssertRedundant quadGatherMirFunc 2 := by
  refine ⟨?_, ?_, ?_⟩
  · apply CudaOxide.Proofs.Program.vcAssertRedundant_of_table quadGather_valid quadGather_asserts (by mir_bind_asserts)
    mir_vc quadGatherMirFunc
  · apply CudaOxide.Proofs.Program.vcAssertRedundant_of_table quadGather_valid quadGather_asserts (by mir_bind_asserts)
    mir_vc quadGatherMirFunc
  · apply CudaOxide.Proofs.Program.vcAssertRedundant_of_table quadGather_valid quadGather_asserts (by mir_bind_asserts)
    mir_vc quadGatherMirFunc

namespace RowProduct2D
/-- Commuted 2D scalar-product `requires = (x.len() >= k * m)` proving `t * k + col < x.len()`
    when `t < m` and `col < k`. -/
def rowProduct2DMirFunc : CudaOxide.Mir.Program.MirFunc :=
  { domain := 1
  , blockDimX := 256
  , blockDimY := 1
  , params := [(.slice false), (.scalar 32), (.scalar 32)]
  , requires := [
      { cmp := .ge
      , lhs := .sliceLen 0
      , rhs := .mul (.scalar 1) (.scalar 2) } ]
  , blocks := [
      { args := [⟨0, 0⟩, ⟨1, 32⟩, ⟨2, 32⟩], stmts := [
          (.assign 3 (.sreg .ctaidX)),
          (.assign 4 (.sreg .tidX)),
          (.assign 5 (.cmpOp .lt 3 2)),
          (.assign 6 (.cmpOp .lt 4 1))
        ], term := (.condBr 5 1 [] 3 []) },
      { stmts := [], term := (.condBr 6 2 [] 3 []) },
      { stmts := [
          (.assign 7 (.intCast 64 3)),
          (.assign 8 (.intCast 64 1)),
          (.assign 9 (.intCast 64 4)),
          (.assign 10 (.binOp .mul 64 7 8)),
          (.assign 11 (.binOp .add 64 10 9)),
          (.assign 12 (.slicePtr 0)),
          (.assign 13 (.ptrOffset 12 11)),
          (.assign 14 (.load 13))
        ], term := (.goto 3 []) },
      { stmts := [], term := .ret }
    ] }

mir_def_sym_table rowProduct2DMirFunc

theorem rowProduct2D_valid :
    CudaOxide.Proofs.Program.validSymFunc rowProduct2DMirFunc symTable = true := rfl

theorem rowProduct2D_events :
    CudaOxide.Proofs.Program.collectSymMemEvents rowProduct2DMirFunc symTable = symMemEvents := rfl

theorem rowProduct2D_no_oob_x :
    CudaOxide.Contracts.Program.MirNoOobGlobal rowProduct2DMirFunc 0 := by
  apply CudaOxide.Proofs.Program.vcNoOobGlobal_of_table rowProduct2D_valid rowProduct2D_events (by mir_bind_events)
  mir_vc rowProduct2DMirFunc
end RowProduct2D

#print axioms no_oob_idx_all
#print axioms disjoint_writes_c
#print axioms writes_all_elems_c
#print axioms rev_writes_all_elems
#print axioms tma_read_after_wait
#print axioms quadGather_no_oob_x
#print axioms quadGather_asserts_redundant
#print axioms RowProduct2D.rowProduct2D_no_oob_x
#print axioms CudaOxide.Mir.Program.ReachableTrapFree.toReachable
#print axioms CudaOxide.Mir.Program.EmitsTrapFreeEvent.toEmitsEvent
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
#print axioms CudaOxide.Proofs.Program.vcWritesAllElems
#print axioms CudaOxide.Proofs.Program.symReachable_sound
#print axioms CudaOxide.Proofs.Program.vcNoOobGlobal_of_table
#print axioms CudaOxide.Proofs.Program.vcNoOobShared_of_table
#print axioms CudaOxide.Proofs.Program.vcDisjointWrites_of_table
#print axioms CudaOxide.Proofs.Program.vcNoSharedRace_of_table
#print axioms CudaOxide.Proofs.Program.vcAssertRedundant_of_table
#print axioms CudaOxide.Proofs.Program.vcNoOobTmem_of_table
#print axioms CudaOxide.Proofs.Program.or_eq_add_of_mod_two_pow

end Accepted
