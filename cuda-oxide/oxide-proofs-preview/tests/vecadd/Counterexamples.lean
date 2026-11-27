/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import Lean
import CudaOxide.Mir.Launch
import CudaOxide.Proofs.ControlFlow

/-! Proved counterexamples for broken mathematical access and Blackwell control-flow patterns. -/
namespace Counterexamples
open CudaOxide.Mir
open CudaOxide.Mir.ControlFlow
open CudaOxide.Contracts.ControlFlow
open CudaOxide.Proofs.ControlFlow

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

/-- 2-CTA `tcgen05_alloc_cg2` before `cluster_sync()` fails even with zero prior `mbarrier_init`. -/
def brokenCg2AllocBeforeClusterSyncCfg : ControlFlowCfg :=
  { blockDimX := 32, blockDimY := 1, clusterDimX := 2, numShared := 1, numBarriers := 0
  , blocks := [
      { ops := [.tmemAlloc 0 128 true, .clusterSync, .tmemRelinquishPermit true,
                .clusterSync, .tmemDealloc 128 true, .clusterSync]
      , isReturn := true } ] }

theorem broken_cg2_alloc_before_cluster_sync :
    checkTmemLifecycleWith brokenCg2AllocBeforeClusterSyncCfg
      (inferStateMap brokenCg2AllocBeforeClusterSyncCfg) = false := by
  rfl

/-- Waiting with `.ctaAcquire` on a remote-arrived cluster barrier fails `validStateMap`. -/
def brokenCtaWaitOnClusterBarCfg : ControlFlowCfg :=
  { blockDimX := 32, blockDimY := 1, clusterDimX := 2, numShared := 0, numBarriers := 1
  , blocks := [
      { ops := [.mbarInit 0 2, .proxyFenceAsyncShared, .clusterSync, .mbarArrive 0 true false]
      , succs := [(.mbarReady 0 false .ctaAcquire, 1), (.mbarSpin 0, 0)] },
      { ops := [.clusterSync], isReturn := true } ] }

theorem broken_cta_wait_on_cluster_bar :
    validStateMap false brokenCtaWaitOnClusterBarCfg
      (inferStateMap brokenCtaWaitOnClusterBarCfg) = false := by
  rfl

/-- `tcgen05.ld` after `mbarrier_try_wait` without `tcgen05.fence::after_thread_sync` fails. -/
def brokenMissingTmemFenceAfterCfg : ControlFlowCfg :=
  { blockDimX := 32, blockDimY := 1, clusterDimX := 1, numShared := 2, numBarriers := 1
  , blocks := [
      { ops := [.mbarInit 0 1, .proxyFenceAsyncShared, .syncThreads,
                .tmemAlloc 0 128 false, .syncThreads, .sharedWrite 1,
                .proxyFenceAsyncShared, .syncThreads]
      , succs := [(.coord (.eq .tx (.const 0)), 1), (.coord (.ne .tx (.const 0)), 2)] },
      { ops := [.tmemAsyncIssue [1] false, .tmemCommit 0 false false]
      , succs := [(.always, 2)] },
      { succs := [(.mbarReady 0 false .ctaAcquire, 3), (.mbarSpin 0, 2)] },
      { ops := [.tmemLoad 0, .tmemLoadWait, .syncThreads, .tmemDealloc 128 false]
      , isReturn := true } ] }

theorem broken_missing_tmem_fence_after :
    validStateMap true brokenMissingTmemFenceAfterCfg
      (inferStateMap brokenMissingTmemFenceAfterCfg) = false := by
  rfl

/-- `mbarrier_arrive` after `tcgen05.ld` without `tcgen05.fence::before_thread_sync` fails. -/
def brokenMissingTmemFenceBeforeCfg : ControlFlowCfg :=
  { blockDimX := 32, blockDimY := 1, clusterDimX := 1, numShared := 2, numBarriers := 2
  , blocks := [
      { ops := [.mbarInit 0 1, .mbarInit 1 1, .proxyFenceAsyncShared, .syncThreads,
                .tmemAlloc 0 128 false, .syncThreads, .sharedWrite 1,
                .proxyFenceAsyncShared, .syncThreads]
      , succs := [(.coord (.eq .tx (.const 0)), 1), (.coord (.ne .tx (.const 0)), 2)] },
      { ops := [.tmemAsyncIssue [1] false, .tmemCommit 0 false false]
      , succs := [(.always, 2)] },
      { succs := [(.mbarReady 0 false .ctaAcquire, 3), (.mbarSpin 0, 2)] },
      { ops := [.tmemFenceAfterSync, .tmemLoad 0, .tmemLoadWait]
      , succs := [(.coord (.eq .tx (.const 0)), 4), (.coord (.ne .tx (.const 0)), 5)] },
      { ops := [.mbarArrive 1 false false], succs := [(.always, 5)] },
      { ops := [.syncThreads, .tmemDealloc 128 false], isReturn := true } ] }

theorem broken_missing_tmem_fence_before :
    validStateMap true brokenMissingTmemFenceBeforeCfg
      (inferStateMap brokenMissingTmemFenceBeforeCfg) = false := by
  rfl

/-- Intra-warp `mbarrier_try_wait` before `mbarrier_arrive` in the same warp fails `validStateMap`. -/
def brokenIntraWarpWaitBeforeArriveCfg : ControlFlowCfg :=
  { blockDimX := 64, blockDimY := 1, clusterDimX := 1, numShared := 0, numBarriers := 1
  , blocks := [
      { ops := [.mbarInit 0 1, .proxyFenceAsyncShared, .syncThreads]
      , succs := [(.coord (.eq (.div .tx 32) (.const 1)), 1),
                  (.coord (.ne (.div .tx 32) (.const 1)), 3)] },
      { succs := [(.mbarReady 0 false .ctaAcquire, 2), (.mbarSpin 0, 1)] },
      { ops := [.mbarArrive 0 false false], succs := [(.always, 3)] },
      { isReturn := true } ] }

theorem broken_intra_warp_wait_before_arrive :
    validStateMap false brokenIntraWarpWaitBeforeArriveCfg
      (inferStateMap brokenIntraWarpWaitBeforeArriveCfg) = false := by
  rfl

#print axioms broken_inclusive_guard
#print axioms broken_lane_only_index
#print axioms broken_missing_guard
#print axioms broken_missing_length
#print axioms broken_insufficient_coverage
#print axioms broken_cg2_alloc_before_cluster_sync
#print axioms broken_cta_wait_on_cluster_bar
#print axioms broken_missing_tmem_fence_after
#print axioms broken_missing_tmem_fence_before
#print axioms broken_intra_warp_wait_before_arrive

end Counterexamples
