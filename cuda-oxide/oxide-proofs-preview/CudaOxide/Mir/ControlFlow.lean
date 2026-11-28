/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import Lean

/-!
# Transcribed Kernel CFG & Hardware State Transitions (`CudaOxide.Mir.ControlFlow`)

Represents the compact basic-block control-flow graph transcribed 1-to-1 by
`lean-exporter` from `PostMem2Reg` `dialect-mir`, together with small-step CTA
state transitions for barriers, PTX `generic -> async` proxies, `mbarrier`
handoffs, and Blackwell `tcgen05` / Tensor Memory (`TMEM`) operations:

```text
  PostMem2Reg `dialect-mir`              `CudaOxide.Mir.ControlFlow`
  ----------------------------------     --------------------------------------
  Basic block `^bb`                      `CfgBlock { ops, succs, isReturn }`
  `nvvm.barrier0`                        `CfgOp.syncThreads`
  `nvvm.mbarrier_init_shared`            `CfgOp.mbarInit barId count`
  `nvvm.fence_proxy_async_shared_cta`    `CfgOp.proxyFenceAsyncShared`
  `nvvm.cp_async_bulk_tensor_*_g2s`      `CfgOp.tmaCopyG2S sharedId barId`
  `nvvm.tcgen05_alloc` / `dealloc`       `CfgOp.tmemAlloc` / `CfgOp.tmemDealloc`
  `nvvm.tcgen05_mma*` / `commit`         `CfgOp.tmemAsyncIssue` / `CfgOp.tmemCommit`
  `nvvm.tcgen05_ld_*` / `load_wait`      `CfgOp.tmemLoad` / `CfgOp.tmemLoadWait`
```
-/
namespace CudaOxide.Mir.ControlFlow

/-- Compositional coordinate expression over intra-CTA thread coordinates `(tx, ty)` and step `i`
    used to label thread-selection branch guards (`tid == 0`, `warp_id == 0`, `tid < 64`). -/
inductive IdxExpr where
  /-- Thread X coordinate (`threadIdx.x`). -/
  | tx
  /-- Thread Y coordinate (`threadIdx.y`). -/
  | ty
  /-- CTA cluster X rank (`cluster_ctaidX()`). -/
  | clusterCtaidX
  /-- Loop step index `i`. -/
  | loopVar
  /-- Constant natural-number literal. -/
  | const (n : Nat)
  /-- Sum `a + b`. -/
  | add (a b : IdxExpr)
  /-- Product `a * b`. -/
  | mul (a b : IdxExpr)
  /-- Euclidean quotient `a / d`. -/
  | div (a : IdxExpr) (d : Nat)
  /-- Euclidean remainder `a % d`. -/
  | mod (a : IdxExpr) (d : Nat)
  deriving Repr, DecidableEq, Lean.ToExpr

/-- Evaluate an `IdxExpr` on concrete `(tx, ty, i, cr)`. -/
def IdxExpr.eval (e : IdxExpr) (tx ty i : Nat) (cr : Nat := 0) : Nat :=
  match e with
  | .tx => tx
  | .ty => ty
  | .clusterCtaidX => cr
  | .loopVar => i
  | .const n => n
  | .add a b => a.eval tx ty i cr + b.eval tx ty i cr
  | .mul a b => a.eval tx ty i cr * b.eval tx ty i cr
  | .div a d => a.eval tx ty i cr / d
  | .mod a d => a.eval tx ty i cr % d

/-- Compositional thread-coordinate branch predicate over `(tx, ty, i)`. -/
inductive GuardExpr where
  /-- Unguarded (`True`). -/
  | tr
  /-- Strict upper bound `a < b` (e.g. `tid < 64`). -/
  | lt (a b : IdxExpr)
  /-- Lower bound `a ≥ b` (negation of `a < b`). -/
  | ge (a b : IdxExpr)
  /-- Equality guard `a = b` (e.g. `lane == 0` or `warp_id == 0`). -/
  | eq (a b : IdxExpr)
  /-- Inequality guard `a ≠ b` (negation of `a = b`). -/
  | ne (a b : IdxExpr)
  /-- Conjunction of two dominating branch guards (`p ∧ q`). -/
  | and (p q : GuardExpr)
  /-- Warp-convergent runtime branch (`!lane_divergent`, e.g. `if tile_kind == 0` inside `warp_id < 4`).
      Blocks CTA-wide `sync_threads()`, `isSingleLeader`, and unconditional `sharedWriterBlocks`,
      while preserving 32-lane intra-warp convergence for `tcgen05.ld` / `tcgen05.st`. -/
  | warpDiv
  /-- Per-lane data-dependent or unmodeled branch condition (`if data[tid] > 0.0`).
      Poisons both warp-collective and block-uniformity checks until reconvergence. -/
  | unknown
  deriving Repr, DecidableEq, Lean.ToExpr

/-- Logical negation of a `GuardExpr` across a `cond_br` false arm. -/
def GuardExpr.negate (g : GuardExpr) : GuardExpr :=
  match g with
  | .tr => .tr
  | .lt a b => .ge a b
  | .ge a b => .lt a b
  | .eq a b => .ne a b
  | .ne a b => .eq a b
  | .and _ _ => .unknown
  | .warpDiv => .warpDiv
  | .unknown => .unknown

/-- Conjoin two `GuardExpr`s, simplifying `.tr` identities. -/
def GuardExpr.conj (p q : GuardExpr) : GuardExpr :=
  match p, q with
  | .tr, _ => q
  | _, .tr => p
  | _, _ => if p == q then p else .and p q

/-- True when `g` contains any runtime data-dependent (`.warpDiv` or `.unknown`) branch guard. -/
def GuardExpr.hasUnknown (g : GuardExpr) : Bool :=
  match g with
  | .warpDiv | .unknown => true
  | .and p q => p.hasUnknown || q.hasUnknown
  | _ => false

/-- True when `g` includes an exact `tx = C` thread check. -/
def GuardExpr.hasExactTxEq (g : GuardExpr) : Bool :=
  match g with
  | .eq .tx (.const _) | .eq (.const _) .tx => true
  | .and p q => p.hasExactTxEq || q.hasExactTxEq
  | _ => false

/-- True when `g` includes an exact `ty = C` thread check. -/
def GuardExpr.hasExactTyEq (g : GuardExpr) : Bool :=
  match g with
  | .eq .ty (.const _) | .eq (.const _) .ty => true
  | .and p q => p.hasExactTyEq || q.hasExactTyEq
  | _ => false

/-- True when `g` includes an exact `tx / 32 = W` warp-selection check. -/
def GuardExpr.hasWarpEq (g : GuardExpr) : Bool :=
  match g with
  | .eq (.div .tx 32) (.const _) | .eq (.const _) (.div .tx 32) => true
  | .and p q => p.hasWarpEq || q.hasWarpEq
  | _ => false

/-- True when `g` restricts execution to a specialized warp subset (`warp_id == W`, `< W`, or `>= W`). -/
def GuardExpr.hasWarpFilter (g : GuardExpr) : Bool :=
  match g with
  | .eq (.div .tx 32) (.const _) | .eq (.const _) (.div .tx 32) => true
  | .lt (.div .tx 32) (.const _) | .ge (.div .tx 32) (.const _) => true
  | .and p q => p.hasWarpFilter || q.hasWarpFilter
  | _ => false

/-- True when `g` includes an exact `tx % 32 = L` intra-warp lane check. -/
def GuardExpr.hasLaneEq (g : GuardExpr) : Bool :=
  match g with
  | .eq (.mod .tx 32) (.const _) | .eq (.const _) (.mod .tx 32) => true
  | .and p q => p.hasLaneEq || q.hasLaneEq
  | _ => false

/-- True when `g` narrows execution to at most one thread in a CTA (`(blockDimX, blockDimY)`).
    In multi-warp CTAs (`blockDimX > 32`), bare `lane == 0` selects multiple warps' lane-0 threads
    and is NOT accepted without `warp == W`! -/
def GuardExpr.selectsAtMostOneThread (g : GuardExpr) (blockDimX : Nat := 32) (blockDimY : Nat := 1) : Bool :=
  if blockDimY == 1 then
    g.hasExactTxEq || (g.hasWarpEq && g.hasLaneEq) || (decide (blockDimX ≤ 32) && g.hasLaneEq)
  else
    g.hasExactTxEq && g.hasExactTyEq

/-- True when `g` narrows execution to at most one single thread in the CTA (`{t | g(t)}` has size `≤ 1`,
    preserved under conjoining any inner runtime branch condition). -/
def GuardExpr.isSingleLeader (g : GuardExpr) (blockDimX blockDimY : Nat) : Bool :=
  g.selectsAtMostOneThread blockDimX blockDimY

/-- True when `g` splits lanes within a 32-lane hardware warp or depends on lane-divergent `.unknown`. -/
def GuardExpr.hasSubwarpDivergence (g : GuardExpr) : Bool :=
  match g with
  | .tr | .warpDiv => false
  | .lt .tx (.const k) | .ge .tx (.const k) => k % 32 != 0
  | .eq (.div .tx 32) (.const _) | .eq (.const _) (.div .tx 32) => false
  | .lt (.div .tx 32) (.const _) | .ge (.div .tx 32) (.const _) => false
  | .eq .clusterCtaidX (.const _) | .eq (.const _) .clusterCtaidX => false
  | .and p q => p.hasSubwarpDivergence || q.hasSubwarpDivergence
  | _ => true

/-- True when every active warp under `g` executes with all 32 lanes converged. -/
def GuardExpr.isWarpConvergent (g : GuardExpr) (blockDimX blockDimY : Nat) : Bool :=
  if blockDimY == 1 then
    blockDimX % 32 == 0 && !g.hasSubwarpDivergence
  else
    (blockDimX * blockDimY) % 32 == 0 && (g == .tr || g == .warpDiv)

/-- True when `g` filters lanes within a 32-lane hardware warp or depends on `.unknown`. -/
def GuardExpr.hasSubwarpLaneFilter (g : GuardExpr) : Bool :=
  match g with
  | .tr => false
  | .lt .tx (.const bound) => bound != 32
  | .eq (.div .tx 32) (.const _) | .eq (.const _) (.div .tx 32) => false
  | .eq .clusterCtaidX (.const _) | .eq (.const _) .clusterCtaidX => false
  | .and p q => p.hasSubwarpLaneFilter || q.hasSubwarpLaneFilter
  | _ => true

/-- Return `some warpIdx` when `g` selects exactly one full 32-lane hardware warp
    (`(warpIdx + 1) * 32 ≤ blockDimX` in a 1D block `blockDimY = 1`) with zero
    sub-warp lane filtering or `.unknown` branch nesting. -/
def GuardExpr.selectsSingleFullWarp (g : GuardExpr) (blockDimX blockDimY : Nat) : Option Nat :=
  if blockDimY != 1 || g.hasSubwarpLaneFilter then
    none
  else
    match g with
    | .tr => if blockDimX == 32 then some 0 else none
    | .lt .tx (.const 32) => if 32 ≤ blockDimX then some 0 else none
    | .eq (.div .tx 32) (.const w) | .eq (.const w) (.div .tx 32) =>
        if (w + 1) * 32 ≤ blockDimX then some w else none
    | .and p q =>
        match p.selectsSingleFullWarp blockDimX blockDimY,
              q.selectsSingleFullWarp blockDimX blockDimY with
        | some w1, some w2 => if w1 == w2 then some w1 else none
        | some w, none | none, some w => some w
        | none, none => none
    | _ => none

/-- Evaluate a known coordinate `GuardExpr` to a concrete `Bool` on `(tx, ty, i, cr)`,
    returning `none` if `g` depends on `.unknown`. -/
def GuardExpr.evalBool (g : GuardExpr) (tx ty i : Nat) (cr : Nat := 0) : Option Bool :=
  match g with
  | .tr => some true
  | .lt a b => some (decide (a.eval tx ty i cr < b.eval tx ty i cr))
  | .ge a b => some (decide (a.eval tx ty i cr ≥ b.eval tx ty i cr))
  | .eq a b => some (decide (a.eval tx ty i cr = b.eval tx ty i cr))
  | .ne a b => some (decide (a.eval tx ty i cr ≠ b.eval tx ty i cr))
  | .and p q =>
      match p.evalBool tx ty i cr, q.evalBool tx ty i cr with
      | some false, _ | _, some false => some false
      | some true, some true => some true
      | _, _ => none
  | .warpDiv | .unknown => none

/-- Return `some warpIdx` when `g` restricts execution to a single warp `tx / 32 = warpIdx`. -/
def GuardExpr.exactWarpOpt (g : GuardExpr) (blockDimX blockDimY : Nat) : Option Nat :=
  if blockDimY != 1 then
    none
  else
    match g with
    | .tr => if blockDimX ≤ 32 then some 0 else none
    | .lt .tx (.const 32) => some 0
    | .eq .tx (.const t) | .eq (.const t) .tx => some (t / 32)
    | .eq (.div .tx 32) (.const w) | .eq (.const w) (.div .tx 32) => some w
    | .and p q =>
        match p.exactWarpOpt blockDimX blockDimY, q.exactWarpOpt blockDimX blockDimY with
        | some w1, some w2 => if w1 == w2 then some w1 else none
        | some w, none | none, some w => some w
        | none, none => none
    | _ => none

/-- Memory-consistency scope and ordering qualifier on an `mbarrier` wait edge. -/
inductive MbarWaitKind where
  /-- `.acquire.cta` (`mbarrier_try_wait_parity`, `mbarrier_try_wait`, `mbarrier_test_wait`). -/
  | ctaAcquire
  /-- `.acquire.cluster` (`mbarrier_try_wait_parity_cluster`). -/
  | clusterAcquire
  /-- `.relaxed.cluster` (`mbarrier_try_wait_parity_cluster_relaxed`, execution-only). -/
  | clusterRelaxed
  deriving Repr, DecidableEq, Lean.ToExpr

/-- Branch label on an outgoing basic-block terminator edge in `PostMem2Reg` MIR. -/
inductive BranchCond where
  /-- Unconditional `mir.goto`. -/
  | always
  /-- Thread-coordinate predicate (`tid == 0`, `warp_id == 0`, `tid < 64`, `lane == 0`). -/
  | coord (g : GuardExpr)
  /-- `mbarrier_try_wait*` ready edge (`initFree = true` when initial ring parity is `1`). -/
  | mbarReady (barId : Nat) (initFree : Bool := false) (waitKind : MbarWaitKind := .ctaAcquire)
  /-- `mbarrier_try_wait*` spin edge (`ready = false`). -/
  | mbarSpin (barId : Nat)
  /-- Exit edge of a positive-trip counted loop (`0 < bound`) whose body starts at `bodyBlk`. -/
  | posLoopExit (bodyBlk : Nat)
  /-- CTA-uniform loop bound branch (`tile < num_tiles`, `col < 128`). -/
  | uniform
  /-- Warp-convergent runtime branch (`!lane_divergent`, all 32 lanes of each warp take same arm). -/
  | warpUniform
  /-- Per-thread data-dependent or unmodeled runtime branch (`data[tid] > 0.0`). -/
  | unknown
  deriving Repr, DecidableEq, Lean.ToExpr

/-- Control-flow-sensitive GPU or shared-memory operation inside a basic block,
    transcribed in exact program order from `PostMem2Reg` `dialect-mir`. -/
inductive CfgOp where
  /-- CTA-wide `thread::sync_threads()` (`nvvm.barrier0`). -/
  | syncThreads
  /-- Cluster-wide `cluster::cluster_sync()` (`nvvm.cluster_sync` / `nvvm.cluster_barrier`). -/
  | clusterSync
  /-- Generic-proxy shared-memory load on `sharedId`. -/
  | sharedRead (sharedId : Nat)
  /-- Generic-proxy shared-memory store on `sharedId`. -/
  | sharedWrite (sharedId : Nat)
  /-- Cross-CTA Distributed Shared Memory (`map_shared_rank`) access on `sharedId`. -/
  | dsmemAccess (sharedId : Nat)
  /-- `mbarrier_init(&BAR, count)` via generic proxy. -/
  | mbarInit (barId count : Nat)
  /-- `mbarrier_inval(&BAR)` resetting `BAR` back to `.uninit`. -/
  | mbarInval (barId : Nat)
  /-- `fence_proxy_async_shared_cta()` bridging generic writes to the async proxy. -/
  | proxyFenceAsyncShared
  /-- Plain thread arrival `mbarrier_arrive*(&BAR)` (`isCluster` for remote cluster, `isRelaxed` for `.relaxed`). -/
  | mbarArrive (barId : Nat) (isCluster : Bool := false) (isRelaxed : Bool := false)
  /-- Byte-counted arrival `mbarrier_arrive_expect_tx*(&BAR, count, txBytes)` (`isCluster` for `.cluster` scope). -/
  | mbarArriveExpectTx (barId txBytes : Nat) (isCluster : Bool := false)
  /-- Asynchronous TMA copy `cp_async_bulk_tensor_*_g2s(SMEM, BAR, ...)` (`isCg2` = 2-CTA, `isMulticast` = cluster multicast). -/
  | tmaCopyG2S (sharedId barId : Nat) (isCg2 : Bool := false) (isMulticast : Bool := false)
  /-- Asynchronous Blackwell CLC `clc_try_cancel*` writing `sharedId` on `barId` (`isMulticast` = `.multicast::cluster::all`). -/
  | clcTryCancel (sharedId barId : Nat) (isMulticast : Bool := false)
  /-- Warp-collective `tcgen05_alloc(&TMEM_ADDR[0], cols)` (`isCg2` = 2-CTA mode). -/
  | tmemAlloc (sharedId cols : Nat) (isCg2 : Bool := false)
  /-- Warp-collective `tcgen05_dealloc(tmem_addr, cols)` (`isCg2` = 2-CTA mode). -/
  | tmemDealloc (cols : Nat) (isCg2 : Bool := false)
  /-- Warp-collective `tcgen05_relinquish_alloc_permit*` (`isCg2` = 2-CTA mode). -/
  | tmemRelinquishPermit (isCg2 : Bool := false)
  /-- Single-thread async `tcgen05_mma*` or `tcgen05_cp*` reading `smemIds` into `TMEM`. -/
  | tmemAsyncIssue (smemIds : List Nat) (isCg2 : Bool := false)
  /-- Single-thread `tcgen05_commit*(&BAR)` tracking prior `tmemAsyncIssue` on `barId` (`isMulticast` = `.multicast::cluster`). -/
  | tmemCommit (barId : Nat) (isCg2 : Bool := false) (isMulticast : Bool := false)
  /-- Warp-restricted async `tcgen05_ld_*` writing register token `ldId`. -/
  | tmemLoad (ldId : Nat)
  /-- `tcgen05_load_wait()` retiring all in-flight `tmemLoad` operations in the thread. -/
  | tmemLoadWait
  /-- Consumer read of the register result produced by `tmemLoad ldId`. -/
  | tmemLoadUse (ldId : Nat)
  /-- Warp-convergent `tcgen05_store_wait()` retiring async `tcgen05.st` operations. -/
  | tmemStoreWait
  /-- `tcgen05_fence_before_thread_sync()` ordering `tcgen05` prior to thread synchronization. -/
  | tmemFenceBeforeSync
  /-- `tcgen05_fence_after_thread_sync()` ordering `tcgen05` after thread synchronization. -/
  | tmemFenceAfterSync
  deriving Repr, DecidableEq, Lean.ToExpr

/-- One basic block in the kernel's transcribed `PostMem2Reg` control-flow graph. -/
structure CfgBlock where
  /-- Ordered GPU and shared-memory operations inside this block. -/
  ops : List CfgOp := []
  /-- Outgoing CFG edges `(branchCondition, targetBlockIndex)`. -/
  succs : List (BranchCond × Nat) := []
  /-- Optional `MirFunc` SSA condition variable ID when this block ends in `mir.cond_br`. -/
  condVarOpt : Option Nat := none
  /-- True when this block terminates in `mir.return`. -/
  isReturn : Bool := false
  deriving Repr, DecidableEq, Lean.ToExpr

/-- Compact kernel control-flow graph transcribed from `PostMem2Reg` `dialect-mir`. -/
structure ControlFlowCfg where
  /-- Static X dimension of the thread block (`reqntid_x`). -/
  blockDimX : Nat
  /-- Static Y dimension of the thread block (`reqntid_y`). -/
  blockDimY : Nat
  /-- Static X dimension of the CTA cluster (`cluster_dim_x`, `1` when non-clustered). -/
  clusterDimX : Nat := 1
  /-- Number of static `SharedArray` allocations in the kernel. -/
  numShared : Nat
  /-- Number of static `Barrier` (`mbarrier`) allocations in the kernel. -/
  numBarriers : Nat
  /-- Basic blocks indexed `0 .. blocks.length - 1` (`0` is the entry block). -/
  blocks : List CfgBlock
  deriving Repr, DecidableEq, Lean.ToExpr

/-- Look up basic block `b` in `cfg`, returning an empty block if out of range. -/
def ControlFlowCfg.blockAt (cfg : ControlFlowCfg) (b : Nat) : CfgBlock :=
  cfg.blocks.getD b {}

/-- Computable test for valid Blackwell `tcgen05.alloc` column counts (`32..512` powers of two). -/
def validTmemColsBool (cols : Nat) : Bool :=
  cols == 32 || cols == 64 || cols == 128 || cols == 256 || cols == 512

/-- Generic-to-async PTX proxy coherence state of a shared-memory resource. -/
inductive ProxyState where
  /-- Clean: visible to both `generic` and `async` hardware proxies. -/
  | clean
  /-- Written by a single leader thread; requires `fence_proxy_async_shared_cta()`. -/
  | needsFence
  /-- Written across multiple threads; requires `sync_threads()` then `fence_proxy_async_shared_cta()`. -/
  | needsSyncAndFence
  deriving Repr, DecidableEq, Lean.ToExpr

/-- Join (worst-case dirty state) of two `ProxyState` values. -/
def ProxyState.merge (a b : ProxyState) : ProxyState :=
  match a, b with
  | .needsSyncAndFence, _ | _, .needsSyncAndFence => .needsSyncAndFence
  | .needsFence, _ | _, .needsFence => .needsFence
  | .clean, .clean => .clean

/-- True when abstract proxy state `abs` is at least as conservative (dirty) as `conc`. -/
def ProxyState.subsumes (abs conc : ProxyState) : Bool :=
  match abs, conc with
  | .needsSyncAndFence, _ => true
  | .needsFence, .needsFence | .needsFence, .clean => true
  | .clean, .clean => true
  | _, _ => false

/-- Lifecycle state of a 64-bit shared-memory `Barrier` (`mbarrier`). -/
inductive MbarState where
  /-- Uninitialized shared memory (`Barrier::UNINIT`). -/
  | uninit
  /-- Initialized by `mbarrier_init`, awaiting `sync_threads()` before cross-warp use. -/
  | initUnsynced
  /-- Synchronized across the CTA and ready to be armed. -/
  | syncedIdle
  /-- Armed by `mbarrier_arrive` or `tcgen05_commit`. -/
  | armedPlain
  /-- Armed by `mbarrier_arrive_expect_tx`. -/
  | armedTx
  /-- Awaited along an `mbarrier_try_wait*` ready edge. -/
  | completed
  deriving Repr, DecidableEq, Lean.ToExpr

/-- Abstract join of two `MbarState` values across reconverging CFG branches. -/
def MbarState.join (a b : MbarState) : MbarState :=
  if a == b then a
  else
    match a, b with
    | .uninit, _ | _, .uninit => .uninit
    | .initUnsynced, _ | _, .initUnsynced => .initUnsynced
    | _, _ => .syncedIdle

/-- True when abstract `MbarState` `abs` safely over-approximates concrete `conc`. -/
def MbarState.subsumes (abs conc : MbarState) : Bool :=
  if abs == conc then true
  else
    match abs, conc with
    | .uninit, _ => true
    | .initUnsynced, .uninit => false
    | .initUnsynced, _ => true
    | .syncedIdle, .armedPlain | .syncedIdle, .armedTx | .syncedIdle, .completed => true
    | _, _ => false

/-- Allocation phase of CTA-wide Blackwell Tensor Memory (`TMEM`). -/
inductive TmemPhase where
  /-- Not yet allocated. -/
  | uninit
  /-- Dynamically allocated with `(cols, isCg2)`. -/
  | allocated (cols : Nat) (isCg2 : Bool)
  /-- Deallocated with `(cols, isCg2)`. -/
  | freed (cols : Nat) (isCg2 : Bool)
  /-- Incompatible `TMEM` phases joined across divergent CFG paths (rejected on use/return). -/
  | conflict
  deriving Repr, DecidableEq, Lean.ToExpr

/-- Abstract join of two `TmemPhase` values. -/
def TmemPhase.join (a b : TmemPhase) : TmemPhase :=
  if a == b then a else .conflict

/-- True when abstract `TmemPhase` `abs` subsumes `conc`. -/
def TmemPhase.subsumes (abs conc : TmemPhase) : Bool :=
  abs == .conflict || abs == conc

/-- Insert `x` into an ascending `List Nat` without duplicates so set-valued `CtaState` fields are canonical. -/
def insertSortedNat (x : Nat) : List Nat → List Nat
  | [] => [x]
  | y :: ys =>
      if x == y then y :: ys
      else if x < y then x :: y :: ys
      else y :: insertSortedNat x ys

/-- Sorted union of two `List Nat` sets. -/
def unionSortedNat (xs ys : List Nat) : List Nat :=
  xs.foldl (fun acc x => insertSortedNat x acc) ys

/-- Intersection of two `List Nat` sets. -/
def interSortedNat (xs ys : List Nat) : List Nat :=
  xs.filter (· ∈ ys)

/-- Decidable subset check `xs ⊆ ys` (`∀ x ∈ xs, x ∈ ys`). -/
def natSubset (xs ys : List Nat) : Bool :=
  xs.all (· ∈ ys)

/-- Pointwise equal-length list subsumption check. -/
def listSubsumes {α : Type} (sub : α → α → Bool) : List α → List α → Bool
  | [], [] => true
  | x :: xs, y :: ys => sub x y && listSubsumes sub xs ys
  | _, _ => false

/-- Concrete and abstract CTA state tracked across basic blocks and control-flow edges.
    Each basic block in `Proofs/ControlFlow.lean` carries at most ONE joined `CtaState`
    (`O(blocks)` symbolic summarization, zero powerset state explosion). -/
structure CtaState where
  /-- Generic-proxy dirty state for each `SharedArray` (`0 .. numShared - 1`). -/
  sharedProxy : List ProxyState
  /-- In-flight (un-awaited) TMA/CLC `Barrier` IDs writing each `SharedArray` (`0 .. numShared - 1`). -/
  sharedPendingTma : List (List Nat)
  /-- True for each `SharedArray` when no unsynchronized local write precedes a peer `dsmemAccess`. -/
  sharedClusterSynced : List Bool
  /-- Generic-proxy dirty state for each `Barrier` (`0 .. numBarriers - 1`). -/
  barProxy : List ProxyState
  /-- Lifecycle state for each `Barrier` (`0 .. numBarriers - 1`). -/
  barState : List MbarState
  /-- Current `TMEM` allocation phase. -/
  tmemPhase : TmemPhase := .uninit
  /-- True once peer CTAs have synchronized via `cluster_sync()` (`true` for 1-CTA; required before `alloc_cg2`). -/
  clusterLaunched : Bool := true
  /-- True once `tcgen05_alloc_cg2` has been synchronized across the 2-CTA cluster (`true` for 1-CTA). -/
  allocClusterSynced : Bool := true
  /-- True when any incoming path has an uncommitted `tcgen05_mma` / `tcgen05_cp` hazard. -/
  mmaHazard : Bool := false
  /-- True when all incoming paths have issued at least one `tcgen05_mma` / `tcgen05_cp`. -/
  mmaIssued : Bool := false
  /-- `Barrier` IDs guaranteed armed by `tcgen05_commit` on all incoming paths. -/
  tmemCommitBars : List Nat := []
  /-- True once `mbarReady` has fired on a `tmemCommitBars` barrier (awaiting `tcgen05.fence::after_thread_sync`). -/
  tmemCommitReady : Bool := false
  /-- True once `tcgen05.fence::after_thread_sync` has ordered a completed `tmemCommitReady` handoff. -/
  tmemReady : Bool := false
  /-- True when an `mbarReady` edge has not yet been followed by `tcgen05.fence::after_thread_sync`. -/
  unFencedMbarWait : Bool := false
  /-- True when a `tcgen05.ld` has not yet been followed by `tcgen05.fence::before_thread_sync` (or CTA sync). -/
  unFencedTmemLoad : Bool := false
  /-- In-flight `tcgen05_ld` register IDs not yet retired by `tcgen05_load_wait()`. -/
  pendingLoads : List Nat := []
  /-- True once `sync_threads()` or `cluster_sync()` has executed after the latest `TMEM` use. -/
  syncedSinceTmemUse : Bool := false
  /-- True once `cluster_sync()` has executed after the latest `TMEM` use (required by `_cg2` dealloc). -/
  clusterSyncedSinceTmemUse : Bool := false
  /-- True once `tcgen05_relinquish_alloc_permit_cg2` has executed (required by `_cg2` dealloc). -/
  relinquishedPermitCg2 : Bool := false
  /-- Positive-trip counted loop body blocks visited along any incoming CTA path. -/
  visitedPosBodies : List Nat := []
  deriving Repr, DecidableEq, Lean.ToExpr

/-- Initial CTA state at kernel entry (`block 0`). -/
def CtaState.initial (cfg : ControlFlowCfg) : CtaState :=
  { sharedProxy := List.replicate cfg.numShared .clean
  , sharedPendingTma := List.replicate cfg.numShared []
  , sharedClusterSynced := List.replicate cfg.numShared true
  , barProxy := List.replicate cfg.numBarriers .clean
  , barState := List.replicate cfg.numBarriers .uninit
  , clusterLaunched := decide (cfg.clusterDimX ≤ 1) }

/-- Sound abstract join of two `CtaState`s at a CFG merge block (`1` summary per block). -/
def CtaState.join (a b : CtaState) : CtaState :=
  { sharedProxy := List.zipWith ProxyState.merge a.sharedProxy b.sharedProxy
  , sharedPendingTma := List.zipWith unionSortedNat a.sharedPendingTma b.sharedPendingTma
  , sharedClusterSynced := List.zipWith (· && ·) a.sharedClusterSynced b.sharedClusterSynced
  , barProxy := List.zipWith ProxyState.merge a.barProxy b.barProxy
  , barState := List.zipWith MbarState.join a.barState b.barState
  , tmemPhase := TmemPhase.join a.tmemPhase b.tmemPhase
  , clusterLaunched := a.clusterLaunched && b.clusterLaunched
  , allocClusterSynced := a.allocClusterSynced && b.allocClusterSynced
  , mmaHazard := a.mmaHazard || b.mmaHazard
  , mmaIssued := a.mmaIssued && b.mmaIssued
  , tmemCommitBars := interSortedNat a.tmemCommitBars b.tmemCommitBars
  , tmemCommitReady := a.tmemCommitReady && b.tmemCommitReady
  , tmemReady := a.tmemReady && b.tmemReady
  , unFencedMbarWait := a.unFencedMbarWait || b.unFencedMbarWait
  , unFencedTmemLoad := a.unFencedTmemLoad || b.unFencedTmemLoad
  , pendingLoads := unionSortedNat a.pendingLoads b.pendingLoads
  , syncedSinceTmemUse := a.syncedSinceTmemUse && b.syncedSinceTmemUse
  , clusterSyncedSinceTmemUse := a.clusterSyncedSinceTmemUse && b.clusterSyncedSinceTmemUse
  , relinquishedPermitCg2 := a.relinquishedPermitCg2 && b.relinquishedPermitCg2
  , visitedPosBodies := unionSortedNat a.visitedPosBodies b.visitedPosBodies }

/-- Decidable subsumption `abs.subsumes conc`: `abs` is at least as conservative as `conc`
    on every safety/hazard component, so `abs` passing `stepOp`/`stepEdge`/`validReturnState`
    guarantees `conc` passes too. -/
def CtaState.subsumes (abs conc : CtaState) : Bool :=
  listSubsumes ProxyState.subsumes abs.sharedProxy conc.sharedProxy &&
  listSubsumes (fun a c => natSubset c a) abs.sharedPendingTma conc.sharedPendingTma &&
  listSubsumes (fun a c => !a || c) abs.sharedClusterSynced conc.sharedClusterSynced &&
  listSubsumes ProxyState.subsumes abs.barProxy conc.barProxy &&
  listSubsumes MbarState.subsumes abs.barState conc.barState &&
  TmemPhase.subsumes abs.tmemPhase conc.tmemPhase &&
  (!abs.clusterLaunched || conc.clusterLaunched) &&
  (!abs.allocClusterSynced || conc.allocClusterSynced) &&
  (!conc.mmaHazard || abs.mmaHazard) &&
  (!abs.mmaIssued || conc.mmaIssued) &&
  natSubset abs.tmemCommitBars conc.tmemCommitBars &&
  (!abs.tmemCommitReady || conc.tmemCommitReady) &&
  (!abs.tmemReady || conc.tmemReady) &&
  (!conc.unFencedMbarWait || abs.unFencedMbarWait) &&
  (!conc.unFencedTmemLoad || abs.unFencedTmemLoad) &&
  natSubset conc.pendingLoads abs.pendingLoads &&
  (!abs.syncedSinceTmemUse || conc.syncedSinceTmemUse) &&
  (!abs.clusterSyncedSinceTmemUse || conc.clusterSyncedSinceTmemUse) &&
  (!abs.relinquishedPermitCg2 || conc.relinquishedPermitCg2) &&
  natSubset conc.visitedPosBodies abs.visitedPosBodies

/-- Wave-by-wave breadth-first search avoiding `avoid` (recursion depth ≤ `cfg.blocks.length`,
    using non-dependent `List.contains` so `whnf` never constructs `DecidableEq` proof terms). -/
def bfsWaveAvoiding (cfg : ControlFlowCfg) (avoid : List Nat) :
    Nat → List Nat → List Nat → List Nat
  | 0, _, visited => visited
  | _ + 1, [], visited => visited
  | fuel + 1, frontier, visited =>
      let valid := frontier.foldl (fun acc u =>
        if avoid.contains u || visited.contains u || acc.contains u || u ≥ cfg.blocks.length then acc else u :: acc) []
      if valid.isEmpty then
        visited
      else
        let visited' := valid ++ visited
        let nextFrontier := valid.foldl (fun acc u =>
          ((cfg.blockAt u).succs.map Prod.snd).foldl (fun acc2 v =>
            if avoid.contains v || visited'.contains v || acc2.contains v then acc2 else v :: acc2) acc) []
        bfsWaveAvoiding cfg avoid fuel nextFrontier visited'

/-- Compute all reachable block indices from `src` without stepping on any block in `avoid`. -/
def reachableNodesAvoiding (cfg : ControlFlowCfg) (src : Nat) (avoid : List Nat) : List Nat :=
  if avoid.contains src || src ≥ cfg.blocks.length then
    []
  else
    bfsWaveAvoiding cfg avoid (cfg.blocks.length + 1) [src] []

/-- Computable reachability check: can `src` reach `dst` without touching `avoid`? -/
def canReachAvoiding (cfg : ControlFlowCfg) (src dst : Nat) (avoid : List Nat) : Bool :=
  (reachableNodesAvoiding cfg src avoid).contains dst

/-- Computable dominance test: every path from entry `0` to `b` visits `dom`. -/
def dominatesBool (cfg : ControlFlowCfg) (dom b : Nat) : Bool :=
  canReachAvoiding cfg 0 b [] && !canReachAvoiding cfg 0 b [dom]

/-- True when basic block `b` lies on a directed cycle (`b ->+ b`). -/
def isCyclicBlockBool (cfg : ControlFlowCfg) (b : Nat) : Bool :=
  (cfg.blockAt b).succs.any (fun (_, s) => canReachAvoiding cfg s b [])

/-- True when every path from `src` to any `isReturn` block passes through `post`. -/
def postDominatesFromBool (cfg : ControlFlowCfg) (src post : Nat) : Bool :=
  let indices := List.range cfg.blocks.length
  indices.all fun ret =>
    if (cfg.blockAt ret).isReturn && canReachAvoiding cfg src ret [] then
      !canReachAvoiding cfg src ret [post]
    else
      true

/-- Extract the branch-guard contribution of `cond` as a `GuardExpr`. -/
def BranchCond.toGuardExpr (c : BranchCond) : GuardExpr :=
  match c with
  | .coord g => g
  | .warpUniform => .warpDiv
  | .unknown => .unknown
  | .always | .mbarReady _ _ _ | .mbarSpin _ | .posLoopExit _ | .uniform => .tr

/-- Compute the active dominating `GuardExpr` at basic block `b` across all 2-way branches.
    Any nested `.unknown` branch (`1..N` levels deep) attaches `.unknown` until reconvergence.
    Pre-barrier loops (`!canReachAvoiding cfg b brBlk []`) reconverge at loop exit. -/
def activeGuardAt (cfg : ControlFlowCfg) (b : Nat) : GuardExpr :=
  let reach0 := reachableNodesAvoiding cfg 0 []
  if !(b ∈ reach0) then .tr
  else
    let indices := List.range cfg.blocks.length
    indices.foldl (fun acc brBlk =>
      match (cfg.blockAt brBlk).succs with
      | [(c0, d0), (c1, d1)] =>
          let g0e := c0.toGuardExpr
          let g1e := c1.toGuardExpr
          if brBlk == b || d0 == d1
              || (g0e == .tr && g1e == .tr)
              || !(brBlk ∈ reach0)
              || canReachAvoiding cfg 0 b [brBlk] then
            acc
          else
            let g0 :=
              if canReachAvoiding cfg d1 b [brBlk] then .tr
              else if (c0 == .unknown || c1 == .unknown)
                  && !canReachAvoiding cfg b brBlk []
                  && canReachAvoiding cfg d1 b [] then .tr
              else g0e
            let g1 :=
              if canReachAvoiding cfg d0 b [brBlk] then .tr
              else if (c0 == .unknown || c1 == .unknown)
                  && !canReachAvoiding cfg b brBlk []
                  && canReachAvoiding cfg d0 b [] then .tr
              else g1e
            acc.conj (g0.conj g1)
      | _ => acc) .tr

/-- True when at least one valid thread `(tx < blockDimX, ty < blockDimY)` satisfies `g`
    (checks lanes `0..31` plus warp leaders `w * 32` to keep WHNF reduction compact). -/
def existsValidThread (cfg : ControlFlowCfg) (g : GuardExpr) : Bool :=
  let warpBases := (List.range ((cfg.blockDimX + 31) / 32)).map (· * 32)
  let txCandidates := (List.range (min cfg.blockDimX 32) ++ warpBases).filter (· < cfg.blockDimX)
  let tyCandidates := List.range (min cfg.blockDimY 32)
  let crCandidates := List.range (max 1 cfg.clusterDimX)
  txCandidates.any fun tx =>
    tyCandidates.any fun ty =>
      crCandidates.any fun cr =>
        g.evalBool tx ty 0 cr == some true

/-- Outgoing CTA edges from block `b`: when `b` splits on a non-empty thread-coordinate
    guard `.coord g` into `(armBlk, joinBlk)` where `armBlk` reconverges at `joinBlk`, the CTA
    always executes `armBlk` on the matching threads before reaching `joinBlk`. -/
def ctaSuccs (cfg : ControlFlowCfg) (b : Nat) : List (BranchCond × Nat) :=
  let succs := (cfg.blockAt b).succs
  match succs with
  | [(.coord g0, d0), (.coord g1, d1)] =>
      if d0 != d1 && existsValidThread cfg g0 && !g0.hasUnknown
          && canReachAvoiding cfg d0 d1 [b] && postDominatesFromBool cfg d0 d1 then
        [(.coord g0, d0)]
      else if d0 != d1 && existsValidThread cfg g1 && !g1.hasUnknown
          && canReachAvoiding cfg d1 d0 [b] && postDominatesFromBool cfg d1 d0 then
        [(.coord g1, d1)]
      else
        succs
  | _ => succs

/-- Outgoing CTA edges from block `b` enabled in post-block state `st`: filters out
    `.posLoopExit bodyBlk` before the positive-trip loop body `bodyBlk` has executed at least once. -/
def enabledCtaSuccs (cfg : ControlFlowCfg) (b : Nat) (st : CtaState) : List (BranchCond × Nat) :=
  (ctaSuccs cfg b).filter fun
    | (.posLoopExit bodyBlk, _) => decide (bodyBlk ∈ st.visitedPosBodies)
    | _ => true

/-- True when `cfg` contains at least one `mbarArriveExpectTx` targeting `barId`. -/
def cfgHasExpectTx (cfg : ControlFlowCfg) (barId : Nat) : Bool :=
  cfg.blocks.any fun blk =>
    blk.ops.any fun
      | .mbarArriveExpectTx b _ _ => b == barId
      | _ => false

/-- True when a distinct basic block `bi ≠ curBlk` containing `.tmemAsyncIssue` can reach `curBlk`. -/
def priorTmemIssueReaches (cfg : ControlFlowCfg) (curBlk : Nat) : Bool :=
  (List.range cfg.blocks.length).any fun bi =>
    bi != curBlk &&
    ((cfg.blockAt bi).ops.any fun | .tmemAsyncIssue _ _ => true | _ => false) &&
    canReachAvoiding cfg bi curBlk []

/-- Single-step operational semantics for one `CfgOp` inside block `curBlk`.
    Returns `none` immediately if any hardware ordering, proxy, barrier, or warp rule fails. -/
def stepOp (cfg : ControlFlowCfg) (curBlk : Nat) (blkGuard : GuardExpr) (inCycle : Bool)
    (st : CtaState) (op : CfgOp) : Option CtaState :=
  let leaderClean := blkGuard.isSingleLeader cfg.blockDimX cfg.blockDimY
  match op with
  | .syncThreads =>
      let cleanDone := !st.mmaHazard && st.pendingLoads.isEmpty
      some
        { st with
          sharedProxy := st.sharedProxy.map fun
            | .needsSyncAndFence => .needsFence
            | other => other
          barProxy := st.barProxy.map fun
            | .needsSyncAndFence => .needsFence
            | other => other
          barState := st.barState.map fun
            | .initUnsynced => .syncedIdle
            | other => other
          unFencedTmemLoad := if cleanDone then false else st.unFencedTmemLoad
          syncedSinceTmemUse := cleanDone }
  | .clusterSync =>
      if cfg.clusterDimX ≤ 1 then
        none
      else
        let cleanDone := !st.mmaHazard && st.pendingLoads.isEmpty
        some
          { st with
            sharedProxy := st.sharedProxy.map fun
              | .needsSyncAndFence => .needsFence
              | other => other
            sharedClusterSynced := List.replicate st.sharedClusterSynced.length true
            barProxy := st.barProxy.map fun
              | .needsSyncAndFence => .needsFence
              | other => other
            barState := st.barState.map fun
              | .initUnsynced => .syncedIdle
              | other => other
            clusterLaunched := true
            allocClusterSynced := true
            unFencedTmemLoad := if cleanDone then false else st.unFencedTmemLoad
            syncedSinceTmemUse := cleanDone
            clusterSyncedSinceTmemUse := cleanDone }
  | .proxyFenceAsyncShared =>
      some
        { st with
          sharedProxy := st.sharedProxy.map fun
            | .needsFence => .clean
            | other => other
          barProxy := st.barProxy.map fun
            | .needsFence => .clean
            | other => other }
  | .sharedRead s =>
      if s ≥ st.sharedProxy.length || s ≥ st.sharedPendingTma.length then
        none
      else if !(st.sharedPendingTma.getD s []).isEmpty then
        none
      else
        some st
  | .dsmemAccess s =>
      if cfg.clusterDimX ≤ 1 || s ≥ st.sharedClusterSynced.length then
        none
      else if !st.sharedClusterSynced.getD s false then
        none
      else
        some st
  | .sharedWrite s =>
      if s ≥ st.sharedProxy.length || s ≥ st.sharedPendingTma.length then
        none
      else if !(st.sharedPendingTma.getD s []).isEmpty then
        none
      else
        let dirty : ProxyState := if leaderClean then .needsFence else .needsSyncAndFence
        let merged := ProxyState.merge (st.sharedProxy.getD s .clean) dirty
        some
          { st with
            sharedProxy := st.sharedProxy.set s merged
            sharedClusterSynced := st.sharedClusterSynced.set s false }
  | .mbarInit barId count =>
      if barId ≥ st.barState.length || count == 0 then
        none
      else
        let dirty : ProxyState := if leaderClean then .needsFence else .needsSyncAndFence
        let merged := ProxyState.merge (st.barProxy.getD barId .clean) dirty
        some
          { st with
            barProxy := st.barProxy.set barId merged
            barState := st.barState.set barId .initUnsynced }
  | .mbarInval barId =>
      if barId ≥ st.barState.length then
        none
      else
        some { st with barState := st.barState.set barId .uninit }
  | .mbarArrive barId isCluster _ =>
      if (isCluster && cfg.clusterDimX ≤ 1) || st.unFencedTmemLoad then
        none
      else
        match st.barState[barId]? with
        | some .syncedIdle | some .armedPlain | some .armedTx | some .completed =>
            some { st with barState := st.barState.set barId .armedPlain }
        | _ => none
  | .mbarArriveExpectTx barId txBytes isCluster =>
      if txBytes == 0 || (isCluster && cfg.clusterDimX ≤ 1) || st.unFencedTmemLoad then
        none
      else
        match st.barState[barId]? with
        | some .syncedIdle | some .armedPlain | some .armedTx | some .completed =>
            some { st with barState := st.barState.set barId .armedTx }
        | _ => none
  | .tmaCopyG2S sharedId barId isCg2 isMc =>
      let barArmedOk :=
        match st.barState.getD barId .uninit with
        | .armedTx => true
        | .syncedIdle | .armedPlain | .completed => isCg2 && cfgHasExpectTx cfg barId
        | _ => false
      if (isCg2 && cfg.clusterDimX != 2) || (isMc && cfg.clusterDimX ≤ 1)
          || !leaderClean || sharedId ≥ st.sharedProxy.length
          || sharedId ≥ st.sharedPendingTma.length || barId ≥ st.barState.length then
        none
      else if st.sharedProxy.getD sharedId .needsFence != .clean
          || st.barProxy.getD barId .needsFence != .clean then
        none
      else if !barArmedOk then
        none
      else
        some
          { st with
            sharedPendingTma := st.sharedPendingTma.set sharedId
              (insertSortedNat barId (st.sharedPendingTma.getD sharedId []))
            sharedClusterSynced := st.sharedClusterSynced.set sharedId false }
  | .clcTryCancel sharedId barId isMc =>
      if cfg.clusterDimX < 1 || (isMc && cfg.clusterDimX ≤ 1)
          || !leaderClean || sharedId ≥ st.sharedProxy.length
          || sharedId ≥ st.sharedPendingTma.length || barId ≥ st.barState.length then
        none
      else if st.barProxy.getD barId .needsFence != .clean then
        none
      else if st.barState.getD barId .uninit != .armedTx then
        none
      else
        some
          { st with
            sharedPendingTma := st.sharedPendingTma.set sharedId
              (insertSortedNat barId (st.sharedPendingTma.getD sharedId []))
            sharedClusterSynced := st.sharedClusterSynced.set sharedId false }
  | .tmemAlloc sharedId cols isCg2 =>
      if (isCg2 && cfg.clusterDimX != 2)
          || inCycle || sharedId ≥ st.sharedProxy.length || !validTmemColsBool cols then
        none
      else if (blkGuard.selectsSingleFullWarp cfg.blockDimX cfg.blockDimY).isNone then
        none
      else if (isCg2 && !st.clusterLaunched) || st.tmemPhase != .uninit then
        none
      else
        some
          { st with
            tmemPhase := .allocated cols isCg2
            allocClusterSynced := !isCg2
            syncedSinceTmemUse := false
            clusterSyncedSinceTmemUse := false
            relinquishedPermitCg2 := false }
  | .tmemRelinquishPermit isCg2 =>
      if isCg2 && cfg.clusterDimX != 2 then
        none
      else if (blkGuard.selectsSingleFullWarp cfg.blockDimX cfg.blockDimY).isNone then
        none
      else
        match st.tmemPhase with
        | .allocated _ allocCg2 =>
            if allocCg2 != isCg2 then none
            else some { st with relinquishedPermitCg2 := isCg2 }
        | _ => none
  | .tmemAsyncIssue smemIds isCg2 =>
      if (isCg2 && cfg.clusterDimX != 2) || !leaderClean
          || !st.allocClusterSynced || st.unFencedMbarWait then
        none
      else
        match st.tmemPhase with
        | .allocated _ allocCg2 =>
            if allocCg2 != isCg2 then
              none
            else
              let smemOk := smemIds.all fun s =>
                s < st.sharedProxy.length
                && s < st.sharedPendingTma.length
                && st.sharedProxy.getD s .needsFence == .clean
                && (st.sharedPendingTma.getD s []).isEmpty
              if !smemOk then
                none
              else
                some
                  { st with
                    mmaHazard := true
                    mmaIssued := true
                    tmemCommitReady := false
                    tmemReady := false
                    syncedSinceTmemUse := false
                    clusterSyncedSinceTmemUse := false }
        | _ => none
  | .tmemCommit barId isCg2 isMc =>
      let hasPriorMma := st.mmaIssued || priorTmemIssueReaches cfg curBlk
      if (isCg2 && cfg.clusterDimX != 2) || (isMc && cfg.clusterDimX ≤ 1)
          || !leaderClean || !hasPriorMma || barId ≥ st.barState.length then
        none
      else if st.barProxy.getD barId .needsFence != .clean then
        none
      else
        match st.tmemPhase, st.barState.getD barId .uninit with
        | .allocated _ allocCg2, .syncedIdle
        | .allocated _ allocCg2, .armedPlain
        | .allocated _ allocCg2, .armedTx
        | .allocated _ allocCg2, .completed =>
            if allocCg2 != isCg2 then
              none
            else
              let nextCommitBars := insertSortedNat barId st.tmemCommitBars
              some
                { st with
                  barState := st.barState.set barId .armedPlain
                  mmaHazard := false
                  tmemCommitBars := nextCommitBars
                  tmemCommitReady := false
                  tmemReady := false
                  syncedSinceTmemUse := false
                  clusterSyncedSinceTmemUse := false }
        | _, _ => none
  | .tmemLoad ldId =>
      if !blkGuard.isWarpConvergent cfg.blockDimX cfg.blockDimY then
        none
      else
        match st.tmemPhase with
        | .allocated _ _ =>
            if !st.allocClusterSynced || !st.tmemReady || st.mmaHazard || st.unFencedMbarWait then
              none
            else
              let nextPending := insertSortedNat ldId st.pendingLoads
              some
                { st with
                  unFencedTmemLoad := true
                  pendingLoads := nextPending
                  syncedSinceTmemUse := false
                  clusterSyncedSinceTmemUse := false }
        | _ => none
  | .tmemLoadWait =>
      match st.tmemPhase with
      | .allocated _ _ => some { st with pendingLoads := [] }
      | _ => none
  | .tmemLoadUse ldId =>
      if ldId ∈ st.pendingLoads then none else some st
  | .tmemStoreWait =>
      if !blkGuard.isWarpConvergent cfg.blockDimX cfg.blockDimY then
        none
      else
        match st.tmemPhase with
        | .allocated _ _ => some st
        | _ => none
  | .tmemFenceBeforeSync =>
      if !blkGuard.isWarpConvergent cfg.blockDimX cfg.blockDimY || !st.pendingLoads.isEmpty then
        none
      else
        match st.tmemPhase with
        | .allocated _ _ => some { st with unFencedTmemLoad := false }
        | _ => none
  | .tmemFenceAfterSync =>
      match st.tmemPhase with
      | .allocated _ _ =>
          let nextReady := if st.tmemCommitReady && !st.mmaHazard then true else st.tmemReady
          some { st with tmemReady := nextReady, unFencedMbarWait := false }
      | _ => none
  | .tmemDealloc cols isCg2 =>
      if (isCg2 && cfg.clusterDimX != 2)
          || inCycle || (blkGuard.selectsSingleFullWarp cfg.blockDimX cfg.blockDimY).isNone then
        none
      else if st.tmemPhase != .allocated cols isCg2 then
        none
      else if st.mmaHazard || !st.pendingLoads.isEmpty || !st.syncedSinceTmemUse then
        none
      else if isCg2 && (!st.clusterSyncedSinceTmemUse || !st.relinquishedPermitCg2) then
        none
      else
        some { st with tmemPhase := .freed cols isCg2 }

/-- All loop-body block indices referenced by `.posLoopExit` edges in `cfg`. -/
def posLoopExitBodies (cfg : ControlFlowCfg) : List Nat :=
  cfg.blocks.foldl (fun acc blk =>
    blk.succs.foldl (fun acc2 (c, _) =>
      match c with
      | .posLoopExit bodyBlk => if bodyBlk ∈ acc2 then acc2 else bodyBlk :: acc2
      | _ => acc2) acc) []

/-- Execute a block's ops using precomputed `(isPosBody, blkGuard, inCycle)` metadata. -/
def execBlockOpsWith (cfg : ControlFlowCfg) (b : Nat) (isPosBody : Bool)
    (blkGuard : GuardExpr) (inCycle : Bool)
    (st : CtaState) (ops : List CfgOp) : Option CtaState :=
  let st0 :=
    if isPosBody && !(b ∈ st.visitedPosBodies) then
      { st with visitedPosBodies := insertSortedNat b st.visitedPosBodies }
    else st
  let rec go (cur : CtaState) : List CfgOp → Option CtaState
    | [] => some cur
    | op :: rest =>
        match stepOp cfg b blkGuard inCycle cur op with
        | none => none
        | some cur' => go cur' rest
  go st0 ops

/-- Execute all operations in a basic block sequentially from `st`. -/
def execBlockOps (cfg : ControlFlowCfg) (b : Nat) (st : CtaState) (ops : List CfgOp) : Option CtaState :=
  execBlockOpsWith cfg b (decide (b ∈ posLoopExitBodies cfg)) (activeGuardAt cfg b)
    (isCyclicBlockBool cfg b) st ops

/-- Barrier IDs signaled anywhere in `cfg` by non-relaxed `mbarArrive`, `mbarArriveExpectTx`, or `tmemCommit`. -/
def signaledBarrierIds (cfg : ControlFlowCfg) : List Nat :=
  (List.range cfg.numBarriers).filter fun barId =>
    cfg.blocks.any fun blk =>
      blk.ops.any fun
        | .mbarArrive b _ isRelaxed => b == barId && !isRelaxed
        | .mbarArriveExpectTx b _ _ | .tmemCommit b _ _ => b == barId
        | _ => false

/-- Basic blocks in `cfg` that signal `barId` via non-relaxed `mbarrier_arrive`, `mbarrier_arrive_expect_tx`, or `tmemCommit`. -/
def barSignalBlocks (cfg : ControlFlowCfg) (barId : Nat) : List Nat :=
  (List.range cfg.blocks.length).filter fun b =>
    (cfg.blockAt b).ops.any fun
      | .mbarArrive bid _ isRelaxed => bid == barId && !isRelaxed
      | .mbarArriveExpectTx bid _ _ | .tmemCommit bid _ _ => bid == barId
      | _ => false

/-- True when `barId` is signaled either by a distinct warp outside `srcBlk`'s warp guard, or
    along an intra-warp path reachable from entry `0` strictly before `srcBlk`. -/
def validWarpWaitSignal (cfg : ControlFlowCfg) (srcBlk barId : Nat) : Bool :=
  let waitGuard := activeGuardAt cfg srcBlk
  if !waitGuard.hasWarpFilter then
    false
  else
    let sigBlks := barSignalBlocks cfg barId
    let waitWarp := waitGuard.exactWarpOpt cfg.blockDimX cfg.blockDimY
    sigBlks.any fun sBlk =>
      let sigGuard := activeGuardAt cfg sBlk
      let sigWarp := sigGuard.exactWarpOpt cfg.blockDimX cfg.blockDimY
      let otherWarp :=
        match waitWarp, sigWarp with
        | some wW, some wS => wW != wS
        | some wW, none =>
            sigGuard.evalBool (wW * 32) 0 0 == some false ||
            existsValidThread cfg (.and sigGuard (waitGuard.negate))
        | none, some wS => waitGuard.evalBool (wS * 32) 0 0 == some false
        | none, none => false
      otherWarp ||
        (sBlk != srcBlk && canReachAvoiding cfg 0 sBlk [srcBlk] && canReachAvoiding cfg sBlk srcBlk [])

/-- Barrier IDs signaled across CTAs (remote `mbarrier_arrive_cluster`, `_cg2`/multicast TMA,
    multicast `clc_try_cancel_multicast`, or `_cg2`/multicast `tcgen05_commit`), requiring a
    cluster-scoped wait (`mbarrier_try_wait_parity_cluster*`) per PTX ISA Section 8.7. -/
def clusterSignaledBarrierIds (cfg : ControlFlowCfg) : List Nat :=
  if cfg.clusterDimX ≤ 1 then []
  else
    (List.range cfg.numBarriers).filter fun barId =>
      cfg.blocks.any fun blk =>
        blk.ops.any fun
          | .mbarArrive b true _ => b == barId
          | .mbarArriveExpectTx b _ true => b == barId
          | .tmaCopyG2S _ b isCg2 isMc => b == barId && (isCg2 || isMc)
          | .clcTryCancel _ b isMc => b == barId && isMc
          | .tmemCommit b isCg2 isMc => b == barId && (isCg2 || isMc)
          | _ => false

/-- Barrier IDs signaled by `tcgen05_commit` in `cfg`. -/
def committedTmemBarrierIds (cfg : ControlFlowCfg) : List Nat :=
  (List.range cfg.numBarriers).filter fun barId =>
    cfg.blocks.any fun blk =>
      blk.ops.any fun
        | .tmemCommit b _ _ => b == barId
        | _ => false

/-- Apply an outgoing terminator edge label `cond` using precomputed `(clusterBars, commitBars, warpSigBars)`. -/
def stepEdgeWith (clusterBars commitBars warpSigBars : List Nat) (st : CtaState) (cond : BranchCond) :
    Option CtaState :=
  match cond with
  | .mbarReady barId initFree waitKind =>
      if barId ∈ clusterBars && waitKind == .ctaAcquire then
        none
      else if waitKind == .clusterRelaxed && st.sharedPendingTma.any (barId ∈ ·) then
        none
      else
        let crossWarpSig := barId ∈ warpSigBars
        let readyOk := match st.barState[barId]? with
          | some .armedPlain | some .armedTx | some .completed => true
          | some .syncedIdle => initFree || crossWarpSig
          | _ => false
        if !readyOk then
          none
        else
          let isCommitBar :=
            barId ∈ st.tmemCommitBars || (crossWarpSig && barId ∈ commitBars)
          let nextCommitReady :=
            if isCommitBar && !st.mmaHazard then true else st.tmemCommitReady
          let nextPendingTma :=
            if waitKind == .clusterRelaxed then st.sharedPendingTma
            else st.sharedPendingTma.map (·.filter (· != barId))
          some
            { st with
              sharedPendingTma := nextPendingTma
              barState := st.barState.set barId .completed
              tmemCommitReady := nextCommitReady
              unFencedMbarWait := true }
  | .mbarSpin barId =>
      match st.barState[barId]? with
      | some .syncedIdle | some .armedPlain | some .armedTx | some .completed => some st
      | _ => none
  | .posLoopExit bodyBlk =>
      if bodyBlk ∈ st.visitedPosBodies then some st else none
  | .always | .coord _ | .uniform | .warpUniform | .unknown => some st

/-- Precompute the list of barriers satisfying `validWarpWaitSignal cfg srcBlk` at block `srcBlk`. -/
def warpWaitSignaledBarsAt (cfg : ControlFlowCfg) (srcBlk : Nat) : List Nat :=
  (List.range cfg.numBarriers).filter (validWarpWaitSignal cfg srcBlk)

/-- Apply an outgoing terminator edge label `cond` from `srcBlk` to post-block state `st`. -/
def stepEdge (cfg : ControlFlowCfg) (srcBlk : Nat) (st : CtaState) (cond : BranchCond) : Option CtaState :=
  stepEdgeWith (clusterSignaledBarrierIds cfg) (committedTmemBarrierIds cfg)
    (warpWaitSignaledBarsAt cfg srcBlk) st cond

/-- True when `st` is a clean CTA return state (no leaked `TMEM`, no uncommitted MMA,
    and no un-awaited `tcgen05_ld` registers). -/
def validReturnState (requireTmem : Bool) (st : CtaState) : Bool :=
  let phaseOk :=
    if requireTmem then
      match st.tmemPhase with
      | .freed _ _ => true
      | _ => false
    else
      match st.tmemPhase with
      | .uninit | .freed _ _ => true
      | .allocated _ _ | .conflict => false
  phaseOk && !st.mmaHazard && st.pendingLoads.isEmpty

/-- Indices of all basic blocks in `cfg` terminating in `mir.return`. -/
def returnBlocks (cfg : ControlFlowCfg) : List Nat :=
  (List.range cfg.blocks.length).filter fun b => (cfg.blockAt b).isReturn

/-- True when every outgoing edge of block `u` is CTA-uniform (`.always`, `.uniform`, or `.posLoopExit`). -/
def isCtaUniformSplit (cfg : ControlFlowCfg) (u : Nat) : Bool :=
  ((cfg.blockAt u).succs).all fun (bc, _) =>
    match bc with
    | .always | .uniform | .posLoopExit _ => true
    | _ => false

/-- Build the reversed CFG (`v -> u` for every edge `u -> v` in `cfg`, `u, v < cfg.blocks.length`)
    in a single `O(V + E)` pass for backward reachability walks. -/
def reverseCfg (cfg : ControlFlowCfg) : ControlFlowCfg :=
  let n := cfg.blocks.length
  let revBlocks := (List.range n).foldl (fun acc u =>
    ((cfg.blockAt u).succs).foldl (fun acc2 (_, v) =>
      if v < n then
        let cur := acc2.getD v {}
        acc2.set v { cur with succs := (.always, u) :: cur.succs }
      else acc2) acc) (List.replicate n {})
  { cfg with blocks := revBlocks }

/-- Indices of all basic blocks in `cfg` containing at least one `.syncThreads` or `.clusterSync` barrier. -/
def syncBarrierBlocks (cfg : ControlFlowCfg) : List Nat :=
  (List.range cfg.blocks.length).filter fun b =>
    (cfg.blockAt b).ops.any fun op => op == .syncThreads || op == .clusterSync

/-- Indices of all basic blocks in `cfg` containing `.clusterSync`. -/
def clusterSyncBlocks (cfg : ControlFlowCfg) : List Nat :=
  (List.range cfg.blocks.length).filter fun b =>
    (cfg.blockAt b).ops.any (· == .clusterSync)

/-- Indices of all basic blocks in `cfg` whose `.dsmemAccess` is not followed by `.clusterSync` in the same block. -/
def unsyncedDsmemBlocks (cfg : ControlFlowCfg) : List Nat :=
  (List.range cfg.blocks.length).filter fun b =>
    (cfg.blockAt b).ops.foldl (fun dirty op =>
      match op with
      | .dsmemAccess _ => true
      | .clusterSync => false
      | _ => dirty) false

/-- Cyclic basic blocks in `cfg` containing `.syncThreads` or `.clusterSync`. -/
def cyclicBarrierBlocks (cfg : ControlFlowCfg) : List Nat :=
  (syncBarrierBlocks cfg).filter (isCyclicBlockBool cfg)

/-- Pairs `(hdr, bodyStart)` for enclosing cyclic-barrier loops in `cfg`. -/
def cyclicBarrierLoopBodyStarts (cfg : ControlFlowCfg) : List (Nat × Nat) :=
  match cyclicBarrierBlocks cfg with
  | [] => []
  | sampleBar :: _ =>
      let candidates := (List.range cfg.blocks.length).filter fun h =>
        canReachAvoiding cfg h sampleBar []
        && canReachAvoiding cfg sampleBar h []
        && !canReachAvoiding cfg 0 sampleBar [h]
      let headers := candidates.filter fun h =>
        candidates.all fun other => other == h || !canReachAvoiding cfg 0 other [h]
      headers.flatMap fun hdr =>
        let bodyStarts := ((cfg.blockAt hdr).succs.map Prod.snd).filter fun s =>
          canReachAvoiding cfg s hdr []
        bodyStarts.map fun s => (hdr, s)

/-- Target blocks of CTA-uniform `.mbarReady` edges for signaled `Barrier`s in `cfg`. -/
def signaledMbarReadyBlocks (cfg : ControlFlowCfg) : List Nat :=
  let sig := signaledBarrierIds cfg
  let raw := cfg.blocks.flatMap fun blk =>
    blk.succs.filterMap fun
      | (.mbarReady barId _ _, dst) => if barId ∈ sig then some dst else none
      | _ => none
  raw.filter fun dst => activeGuardAt cfg dst == .tr

/-- Total number of `sync_threads()` / `cluster_sync()` operations plus signaled `mbarrier` wait edges in `cfg`. -/
def countEffectiveBarriers (cfg : ControlFlowCfg) : Nat :=
  let syncCount := (cfg.blocks.map fun blk =>
    (blk.ops.filter fun op => op == .syncThreads || op == .clusterSync).length).sum
  let sig := signaledBarrierIds cfg
  let mbarCount := (cfg.blocks.flatMap fun blk =>
    blk.succs.filterMap fun
      | (.mbarReady barId _ _, dst) => if barId ∈ sig then some dst else none
      | _ => none).length
  syncCount + mbarCount

/-- Triples `(prodBlk, consBlk, backpressureBlocks)` for cyclic async TMA and `TMEM` loops.
    Every loop backedge from `consBlk` (and every self-cycle of a `TmaCopyG2S` producer when
    consumers read its buffer) must cross `backpressureBlocks`. -/
def loopBackpressurePairs (cfg : ControlFlowCfg) : List (Nat × Nat × List Nat) :=
  let indices := List.range cfg.blocks.length
  let syncs := syncBarrierBlocks cfg
  let sig := signaledBarrierIds cfg
  let committed := committedTmemBarrierIds cfg
  let tmaPairs := (List.range cfg.numShared).flatMap fun s =>
    let prodPair := indices.findSome? fun b =>
      (cfg.blockAt b).ops.findSome? fun
        | .tmaCopyG2S sid bid _ _ => if sid == s then some (b, bid) else none
        | _ => none
    match prodPair with
    | none => []
    | some (prodBlk, tmaBar) =>
        if !isCyclicBlockBool cfg prodBlk then
          []
        else
          let emptyReadyBlks := cfg.blocks.flatMap fun blk =>
            blk.succs.filterMap fun
              | (.mbarReady bid _ _, dst) => if bid != tmaBar && bid ∈ sig then some dst else none
              | _ => none
          let bp := syncs ++ emptyReadyBlks
          let consumers := indices.filter fun b =>
            b != prodBlk &&
            (cfg.blockAt b).ops.any fun
              | .sharedRead sid => sid == s
              | .tmemAsyncIssue smems _ => s ∈ smems
              | _ => false
          let crossPairs := consumers.filterMap fun consBlk =>
            if canReachAvoiding cfg consBlk prodBlk [] then some (prodBlk, consBlk, bp) else none
          let ringSelfPairs :=
            if consumers.isEmpty then []
            else
              ((cfg.blockAt prodBlk).succs.map Prod.snd).filterMap fun succBlk =>
                if canReachAvoiding cfg succBlk prodBlk [] then some (prodBlk, succBlk, bp) else none
          crossPairs ++ ringSelfPairs
  let tmemProds := indices.filter fun b =>
    isCyclicBlockBool cfg b &&
    (cfg.blockAt b).ops.any fun
      | .tmemAsyncIssue _ _ => true
      | _ => false
  let tmemCons := indices.filter fun b =>
    (cfg.blockAt b).ops.any fun
      | .tmemLoad _ => true
      | _ => false
  let tmemEmptyReady := cfg.blocks.flatMap fun blk =>
    blk.succs.filterMap fun
      | (.mbarReady bid _ _, dst) => if !(bid ∈ committed) && bid ∈ sig then some dst else none
      | _ => none
  let tmemBp := syncs ++ tmemEmptyReady
  let tmemPairs := tmemProds.flatMap fun pBlk =>
    tmemCons.filterMap fun cBlk =>
      if pBlk != cBlk && canReachAvoiding cfg cBlk pBlk [] then some (pBlk, cBlk, tmemBp) else none
  tmaPairs ++ tmemPairs

/-- Column counts passed to all `tcgen05_alloc` operations in `cfg`. -/
def tmemAllocSites (cfg : ControlFlowCfg) : List Nat :=
  cfg.blocks.flatMap fun blk =>
    blk.ops.filterMap fun
      | .tmemAlloc _ cols _ => some cols
      | _ => none

/-- Column counts passed to all `tcgen05_dealloc` operations in `cfg`. -/
def tmemDeallocSites (cfg : ControlFlowCfg) : List Nat :=
  cfg.blocks.flatMap fun blk =>
    blk.ops.filterMap fun
      | .tmemDealloc cols _ => some cols
      | _ => none

/-- Blocks `b` whose first access to `SharedArray` `s` inside `b` is `.sharedRead s`
    (i.e. upward-exposed `.sharedRead s` not preceded by an intra-block write to `s`). -/
def exposedSharedReadBlocks (cfg : ControlFlowCfg) (s : Nat) : List Nat :=
  (List.range cfg.blocks.length).filter fun b =>
    let (_, exposed) := (cfg.blockAt b).ops.foldl (fun (written, exp) op =>
      match op with
      | .sharedWrite sid | .tmaCopyG2S sid _ _ _ | .clcTryCancel sid _ _ | .tmemAlloc sid _ _ =>
          if sid == s then (true, exp) else (written, exp)
      | .sharedRead sid =>
          if sid == s && !written then (written, true) else (written, exp)
      | _ => (written, exp)) (false, false)
    exposed

/-- Non-`.unknown`-guarded blocks in `cfg` that initialize `SharedArray` `s`. -/
def sharedWriterBlocks (cfg : ControlFlowCfg) (s : Nat) : List Nat :=
  (List.range cfg.blocks.length).filter fun b =>
    ((cfg.blockAt b).ops.any fun
      | .sharedWrite sid | .tmaCopyG2S sid _ _ _ | .clcTryCancel sid _ _ | .tmemAlloc sid _ _ => sid == s
      | _ => false) &&
    !(activeGuardAt cfg b).hasUnknown

/-- True when an acyclic reachable `.uniform` split block `u` can reach `sb` via an arm avoiding
    `writers` (meaning an entire CTA can reach `sb` without any thread in that CTA executing `writers`). -/
def hasUniformBypassTo (cfg : ControlFlowCfg) (writers : List Nat) (sb : Nat) : Bool :=
  if sb ∈ writers then false
  else
    let reach0 := reachableNodesAvoiding cfg 0 writers
    reach0.any fun u =>
      match (cfg.blockAt u).succs with
      | [(.uniform, d0), (.uniform, d1)] =>
          !isCyclicBlockBool cfg u &&
          ((canReachAvoiding cfg d0 sb writers && writers.any (canReachAvoiding cfg d1 · [])) ||
           (canReachAvoiding cfg d1 sb writers && writers.any (canReachAvoiding cfg d0 · [])))
      | _ => false

/-- Blocks guaranteeing `SharedArray` `s` has been written before an upward-exposed `.sharedRead s`
    (either a direct non-`.unknown` writer block, a CTA barrier reachable after a writer with no
    CTA-uniform bypass, or an `.mbarReady` completion edge when `s` has a writer). -/
def sharedInitDominators (cfg : ControlFlowCfg) (s : Nat) : List Nat :=
  let writers := sharedWriterBlocks cfg s
  if writers.isEmpty then []
  else
    let syncsAfterWrite := (syncBarrierBlocks cfg).filter fun sb =>
      !hasUniformBypassTo cfg writers sb &&
      writers.any fun w => canReachAvoiding cfg 0 w [] && canReachAvoiding cfg w sb []
    let sig := signaledBarrierIds cfg
    let readyBlks := cfg.blocks.flatMap fun blk =>
      blk.succs.filterMap fun
        | (.mbarReady bid _ wk, dst) => if bid ∈ sig && wk != .clusterRelaxed then some dst else none
        | _ => none
    writers ++ syncsAfterWrite ++ readyBlks

/-- Pairs `(s, rBlk)` where `rBlk` has an upward-exposed `.sharedRead s`. -/
def sharedReadSites (cfg : ControlFlowCfg) : List (Nat × Nat) :=
  (List.range cfg.numShared).flatMap fun s =>
    (exposedSharedReadBlocks cfg s).map fun rBlk => (s, rBlk)

/-- Computable check that every 1-CTA `Barrier`'s arrival sites match its `mbarrier_init` thread count
    (rejecting multi-thread arrivals on `count = 1` barriers and single non-cyclic leader arrivals on `count > 1` barriers). -/
def validMbarArrivalCounts (cfg : ControlFlowCfg) : Bool :=
  if cfg.clusterDimX != 1 then true
  else
    (List.range cfg.numBarriers).all fun barId =>
      let inits := cfg.blocks.flatMap fun blk =>
        blk.ops.filterMap fun
          | .mbarInit b c => if b == barId then some c else none
          | _ => none
      let arrBlks := (List.range cfg.blocks.length).filter fun blkIdx =>
        (cfg.blockAt blkIdx).ops.any fun
          | .mbarArrive b _ _ | .mbarArriveExpectTx b _ _ | .tmemCommit b _ _ => b == barId
          | _ => false
      let count1Ok :=
        !(1 ∈ inits) ||
        arrBlks.all fun b => (activeGuardAt cfg b).isSingleLeader cfg.blockDimX cfg.blockDimY
      let countGt1Ok :=
        match arrBlks with
        | [b] =>
            inits.isEmpty || !(inits.all (· > 1)) || isCyclicBlockBool cfg b ||
            !(activeGuardAt cfg b).isSingleLeader cfg.blockDimX cfg.blockDimY
        | _ => true
      count1Ok && countGt1Ok

/-- Basic blocks executing 2-CTA `tcgen05_alloc_cg2` (`isCg2 = true`) without a preceding
    same-block `clusterSync` (requiring `cluster_sync()` on all paths from kernel entry `0`). -/
def unlaunchedCg2AllocBlocks (cfg : ControlFlowCfg) : List Nat :=
  (List.range cfg.blocks.length).filter fun b =>
    let (_, bad) := (cfg.blockAt b).ops.foldl (fun (synced, err) op =>
      match op with
      | .clusterSync => (true, err)
      | .tmemAlloc _ _ true => (synced, err || !synced)
      | _ => (synced, err)) (false, false)
    bad

/-- Pairs `(initBlk, allocBlk)` where `initBlk` performs `mbarInit` or `sharedWrite` without a
    same-block `clusterSync` and reaches a 2-CTA `tcgen05_alloc_cg2` (`isCg2 = true`) at `allocBlk`. -/
def cg2PreAllocInitPairs (cfg : ControlFlowCfg) : List (Nat × Nat) :=
  let indices := List.range cfg.blocks.length
  let allocBlks := indices.filter fun b =>
    (cfg.blockAt b).ops.any fun
      | .tmemAlloc _ _ true => true
      | _ => false
  let initBlks := indices.filter fun b =>
    (cfg.blockAt b).ops.foldl (fun dirty op =>
      match op with
      | .mbarInit _ _ | .sharedWrite _ => true
      | .clusterSync => false
      | _ => dirty) false
  initBlks.flatMap fun iBlk =>
    allocBlks.filterMap fun aBlk =>
      if iBlk != aBlk && canReachAvoiding cfg iBlk aBlk [] then some (iBlk, aBlk) else none

/-- Basic blocks containing a 2-CTA `tcgen05_dealloc_cg2` (`isCg2 = true`) not followed by
    `clusterSync` in the same block (requiring `cluster_sync()` on all paths to `mir.return`). -/
def unsyncedCg2DeallocBlocks (cfg : ControlFlowCfg) : List Nat :=
  (List.range cfg.blocks.length).filter fun b =>
    (cfg.blockAt b).ops.foldl (fun dirty op =>
      match op with
      | .tmemDealloc _ true => true
      | .clusterSync => false
      | _ => dirty) false

end CudaOxide.Mir.ControlFlow
