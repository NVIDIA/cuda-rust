/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide.Mir.Kernel

/-!
# Transcribed Kernel CFG & Hardware State Transitions (`CudaOxide.Mir.ControlFlow`)

Represents the compact basic-block control-flow graph transcribed 1-to-1 by
`lean-exporter` from `PostMem2Reg` `dialect-mir`, together with small-step CTA
state transitions for barriers, PTX `generic -> async` proxies, `mbarrier`
handoffs, and Blackwell `tcgen05` / Tensor Memory (`TMEM`) operations:

```text
  PostMem2Reg `dialect-mir`              `CudaOxide.Mir.ControlFlow`
  ──────────────────────────────────     ──────────────────────────────────────
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
open CudaOxide.Mir.Kernel

/-- Branch label on an outgoing basic-block terminator edge in `PostMem2Reg` MIR. -/
inductive BranchCond where
  /-- Unconditional `mir.goto`. -/
  | always
  /-- Thread-coordinate predicate (`tid == 0`, `warp_id == 0`, `tid < 64`, `lane == 0`). -/
  | coord (g : GuardExpr)
  /-- `mbarrier_try_wait*` ready edge (`ready = true`). -/
  | mbarReady (barId : Nat)
  /-- `mbarrier_try_wait*` spin edge (`ready = false`). -/
  | mbarSpin (barId : Nat)
  /-- CTA-uniform loop bound branch (`tile < num_tiles`, `col < 128`). -/
  | uniform
  /-- Per-thread data-dependent or unmodeled runtime branch (`data[tid] > 0.0`). -/
  | unknown
  deriving Repr, DecidableEq

/-- Control-flow-sensitive GPU or shared-memory operation inside a basic block,
    transcribed in exact program order from `PostMem2Reg` `dialect-mir`. -/
inductive CfgOp where
  /-- CTA-wide `thread::sync_threads()` (`nvvm.barrier0`). -/
  | syncThreads
  /-- Generic-proxy shared-memory load on `sharedId`. -/
  | sharedRead (sharedId : Nat)
  /-- Generic-proxy shared-memory store on `sharedId`. -/
  | sharedWrite (sharedId : Nat)
  /-- `mbarrier_init(&BAR, count)` via generic proxy. -/
  | mbarInit (barId count : Nat)
  /-- `fence_proxy_async_shared_cta()` bridging generic writes to the async proxy. -/
  | proxyFenceAsyncShared
  /-- Plain thread arrival `mbarrier_arrive(&BAR)`. -/
  | mbarArrive (barId : Nat)
  /-- Byte-counted arrival `mbarrier_arrive_expect_tx(&BAR, count, txBytes)`. -/
  | mbarArriveExpectTx (barId txBytes : Nat)
  /-- Asynchronous TMA copy `cp_async_bulk_tensor_*_g2s(SMEM, BAR, ...)`. -/
  | tmaCopyG2S (sharedId barId : Nat)
  /-- Warp-collective `tcgen05_alloc(&TMEM_ADDR[0], cols)` (`isCg2` = 2-CTA mode). -/
  | tmemAlloc (sharedId cols : Nat) (isCg2 : Bool := false)
  /-- Warp-collective `tcgen05_dealloc(tmem_addr, cols)` (`isCg2` = 2-CTA mode). -/
  | tmemDealloc (cols : Nat) (isCg2 : Bool := false)
  /-- Single-thread async `tcgen05_mma*` or `tcgen05_cp*` reading `smemIds` into `TMEM`. -/
  | tmemAsyncIssue (smemIds : List Nat)
  /-- Single-thread `tcgen05_commit*(&BAR)` tracking prior `tmemAsyncIssue` on `barId`. -/
  | tmemCommit (barId : Nat)
  /-- Warp-restricted async `tcgen05_ld_*` writing register token `ldId`. -/
  | tmemLoad (ldId : Nat)
  /-- `tcgen05_load_wait()` retiring all in-flight `tmemLoad` operations in the thread. -/
  | tmemLoadWait
  /-- Consumer read of the register result produced by `tmemLoad ldId`. -/
  | tmemLoadUse (ldId : Nat)
  deriving Repr, DecidableEq

/-- One basic block in the kernel's transcribed `PostMem2Reg` control-flow graph. -/
structure CfgBlock where
  /-- Ordered GPU and shared-memory operations inside this block. -/
  ops : List CfgOp := []
  /-- Outgoing CFG edges `(branchCondition, targetBlockIndex)`. -/
  succs : List (BranchCond × Nat) := []
  /-- True when this block terminates in `mir.return`. -/
  isReturn : Bool := false
  deriving Repr, DecidableEq

/-- Compact kernel control-flow graph transcribed from `PostMem2Reg` `dialect-mir`. -/
structure ControlFlowCfg where
  /-- Static X dimension of the thread block (`reqntid_x`). -/
  blockDimX : Nat
  /-- Static Y dimension of the thread block (`reqntid_y`). -/
  blockDimY : Nat
  /-- Number of static `SharedArray` allocations in the kernel. -/
  numShared : Nat
  /-- Number of static `Barrier` (`mbarrier`) allocations in the kernel. -/
  numBarriers : Nat
  /-- Basic blocks indexed `0 .. blocks.length - 1` (`0` is the entry block). -/
  blocks : List CfgBlock
  deriving Repr, DecidableEq

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
  deriving Repr, DecidableEq

/-- Join (worst-case dirty state) of two `ProxyState` values. -/
def ProxyState.merge (a b : ProxyState) : ProxyState :=
  match a, b with
  | .needsSyncAndFence, _ | _, .needsSyncAndFence => .needsSyncAndFence
  | .needsFence, _ | _, .needsFence => .needsFence
  | .clean, .clean => .clean

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
  deriving Repr, DecidableEq

/-- Allocation phase of CTA-wide Blackwell Tensor Memory (`TMEM`). -/
inductive TmemPhase where
  /-- Not yet allocated. -/
  | uninit
  /-- Dynamically allocated with `(cols, isCg2)`. -/
  | allocated (cols : Nat) (isCg2 : Bool)
  /-- Deallocated with `(cols, isCg2)`. -/
  | freed (cols : Nat) (isCg2 : Bool)
  deriving Repr, DecidableEq

/-- Concrete CTA state tracked across basic blocks and control-flow edges. -/
structure CtaState where
  /-- Generic-proxy dirty state for each `SharedArray` (`0 .. numShared - 1`). -/
  sharedProxy : List ProxyState
  /-- Active TMA `Barrier` for each `SharedArray` (`none` if not written by TMA). -/
  sharedTmaBar : List (Option Nat)
  /-- Generic-proxy dirty state for each `Barrier` (`0 .. numBarriers - 1`). -/
  barProxy : List ProxyState
  /-- Lifecycle state for each `Barrier` (`0 .. numBarriers - 1`). -/
  barState : List MbarState
  /-- Current `TMEM` allocation phase. -/
  tmemPhase : TmemPhase := .uninit
  /-- True when a `tcgen05_mma` / `tcgen05_cp` has issued without a matching `tcgen05_commit`. -/
  mmaUncommitted : Bool := false
  /-- `Barrier` armed by the latest `tcgen05_commit`. -/
  tmemCommitBar : Option Nat := none
  /-- True once `mbarReady` has fired on `tmemCommitBar` with no newer uncommitted MMA. -/
  tmemReady : Bool := false
  /-- In-flight `tcgen05_ld` register IDs not yet retired by `tcgen05_load_wait()`. -/
  pendingLoads : List Nat := []
  /-- True once `sync_threads()` has executed after the latest `TMEM` alloc/issue/load. -/
  syncedSinceTmemUse : Bool := false
  deriving Repr, DecidableEq

/-- Initial CTA state at kernel entry (`block 0`). -/
def CtaState.initial (cfg : ControlFlowCfg) : CtaState :=
  { sharedProxy := List.replicate cfg.numShared .clean
  , sharedTmaBar := List.replicate cfg.numShared none
  , barProxy := List.replicate cfg.numBarriers .clean
  , barState := List.replicate cfg.numBarriers .uninit }

/-- Wave-by-wave breadth-first search avoiding `avoid` (recursion depth ≤ `cfg.blocks.length`). -/
def bfsWaveAvoiding (cfg : ControlFlowCfg) (avoid : List Nat) :
    Nat → List Nat → List Nat → List Nat
  | 0, _, visited => visited
  | _ + 1, [], visited => visited
  | fuel + 1, frontier, visited =>
      let valid := frontier.foldl (fun acc u =>
        if u ∈ avoid || u ∈ visited || u ∈ acc || u ≥ cfg.blocks.length then acc else u :: acc) []
      if valid.isEmpty then
        visited
      else
        let visited' := valid ++ visited
        let nextFrontier := valid.foldl (fun acc u =>
          ((cfg.blockAt u).succs.map Prod.snd).foldl (fun acc2 v =>
            if v ∈ avoid || v ∈ visited' || v ∈ acc2 then acc2 else v :: acc2) acc) []
        bfsWaveAvoiding cfg avoid fuel nextFrontier visited'

/-- Compute all reachable block indices from `src` without stepping on any block in `avoid`. -/
def reachableNodesAvoiding (cfg : ControlFlowCfg) (src : Nat) (avoid : List Nat) : List Nat :=
  if src ∈ avoid || src ≥ cfg.blocks.length then
    []
  else
    bfsWaveAvoiding cfg avoid (cfg.blocks.length + 1) [src] []

/-- Computable reachability check: can `src` reach `dst` without touching `avoid`? -/
def canReachAvoiding (cfg : ControlFlowCfg) (src dst : Nat) (avoid : List Nat) : Bool :=
  dst ∈ reachableNodesAvoiding cfg src avoid

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
  | .unknown => .unknown
  | .always | .mbarReady _ | .mbarSpin _ | .uniform => .tr

/-- Compute the active dominating `GuardExpr` at basic block `b` across all 2-way branches.
    Any nested `.unknown` branch (`1..N` levels deep) attaches `.unknown` until reconvergence. -/
def activeGuardAt (cfg : ControlFlowCfg) (b : Nat) : GuardExpr :=
  let indices := List.range cfg.blocks.length
  indices.foldl (fun acc brBlk =>
    match (cfg.blockAt brBlk).succs with
    | [(c0, d0), (c1, d1)] =>
        if brBlk == b || d0 == d1
            || (c0.toGuardExpr == .tr && c1.toGuardExpr == .tr)
            || !dominatesBool cfg brBlk b then
          acc
        else
          let g0 := if !canReachAvoiding cfg d1 b [brBlk] then c0.toGuardExpr else .tr
          let g1 := if !canReachAvoiding cfg d0 b [brBlk] then c1.toGuardExpr else .tr
          acc.conj (g0.conj g1)
    | _ => acc) .tr

/-- True when at least one valid thread `(tx < blockDimX, ty < blockDimY)` satisfies `g`
    (checks lanes `0..31` plus warp leaders `w * 32` to keep WHNF reduction compact). -/
def existsValidThread (cfg : ControlFlowCfg) (g : GuardExpr) : Bool :=
  let warpBases := (List.range ((cfg.blockDimX + 31) / 32)).map (· * 32)
  let txCandidates := (List.range (min cfg.blockDimX 32) ++ warpBases).filter (· < cfg.blockDimX)
  let tyCandidates := List.range (min cfg.blockDimY 32)
  txCandidates.any fun tx =>
    tyCandidates.any fun ty =>
      g.evalBool tx ty 0 == some true

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

/-- Single-step operational semantics for one `CfgOp` inside block `b`.
    Returns `none` immediately if any hardware ordering, proxy, barrier, or warp rule fails. -/
def stepOp (cfg : ControlFlowCfg) (blkGuard : GuardExpr) (inCycle : Bool)
    (st : CtaState) (op : CfgOp) : Option CtaState :=
  let leaderClean := blkGuard.hasZeroLeaderCheck && !blkGuard.hasUnknown
  match op with
  | .syncThreads =>
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
          syncedSinceTmemUse := !st.mmaUncommitted && st.pendingLoads.isEmpty }
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
      if s < st.sharedProxy.length then some st else none
  | .sharedWrite s =>
      if s ≥ st.sharedProxy.length then
        none
      else
        let tmaOk := match st.sharedTmaBar.getD s none with
          | none => true
          | some barId => st.barState.getD barId .uninit == .completed
        if !tmaOk then
          none
        else
          let dirty : ProxyState := if leaderClean then .needsFence else .needsSyncAndFence
          let merged := ProxyState.merge (st.sharedProxy.getD s .clean) dirty
          some { st with sharedProxy := st.sharedProxy.set s merged }
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
  | .mbarArrive barId =>
      match st.barState[barId]? with
      | some .syncedIdle | some .armedPlain | some .completed =>
          some { st with barState := st.barState.set barId .armedPlain }
      | _ => none
  | .mbarArriveExpectTx barId txBytes =>
      if txBytes == 0 then
        none
      else
        match st.barState[barId]? with
        | some .syncedIdle | some .armedTx | some .completed =>
            some { st with barState := st.barState.set barId .armedTx }
        | _ => none
  | .tmaCopyG2S sharedId barId =>
      if !leaderClean || sharedId ≥ st.sharedProxy.length || barId ≥ st.barState.length then
        none
      else if st.sharedProxy.getD sharedId .needsFence != .clean
          || st.barProxy.getD barId .needsFence != .clean then
        none
      else if st.barState.getD barId .uninit != .armedTx then
        none
      else
        some { st with sharedTmaBar := st.sharedTmaBar.set sharedId (some barId) }
  | .tmemAlloc sharedId cols isCg2 =>
      if inCycle || sharedId ≥ st.sharedProxy.length || !validTmemColsBool cols then
        none
      else if (blkGuard.selectsSingleFullWarp cfg.blockDimX cfg.blockDimY).isNone then
        none
      else if st.tmemPhase != .uninit then
        none
      else
        some
          { st with
            tmemPhase := .allocated cols isCg2
            syncedSinceTmemUse := false }
  | .tmemAsyncIssue smemIds =>
      if !leaderClean then
        none
      else
        match st.tmemPhase with
        | .allocated _ _ =>
            let smemOk := smemIds.all fun s =>
              s < st.sharedProxy.length
              && st.sharedProxy.getD s .needsFence == .clean
              && match st.sharedTmaBar.getD s none with
                 | none => true
                 | some barId => st.barState.getD barId .uninit == .completed
            if !smemOk then
              none
            else
              some
                { st with
                  mmaUncommitted := true
                  tmemCommitBar := none
                  tmemReady := false
                  syncedSinceTmemUse := false }
        | _ => none
  | .tmemCommit barId =>
      if !leaderClean || !st.mmaUncommitted || barId ≥ st.barState.length then
        none
      else if st.barProxy.getD barId .needsFence != .clean then
        none
      else
        match st.tmemPhase, st.barState.getD barId .uninit with
        | .allocated _ _, .syncedIdle | .allocated _ _, .completed =>
            some
              { st with
                barState := st.barState.set barId .armedPlain
                mmaUncommitted := false
                tmemCommitBar := some barId
                tmemReady := false
                syncedSinceTmemUse := false }
        | _, _ => none
  | .tmemLoad ldId =>
      match st.tmemPhase with
      | .allocated _ _ =>
          if !st.tmemReady || st.mmaUncommitted then
            none
          else
            let nextPending := if ldId ∈ st.pendingLoads then st.pendingLoads else ldId :: st.pendingLoads
            some { st with pendingLoads := nextPending, syncedSinceTmemUse := false }
      | _ => none
  | .tmemLoadWait =>
      match st.tmemPhase with
      | .allocated _ _ => some { st with pendingLoads := [] }
      | _ => none
  | .tmemLoadUse ldId =>
      if ldId ∈ st.pendingLoads then none else some st
  | .tmemDealloc cols isCg2 =>
      if inCycle || (blkGuard.selectsSingleFullWarp cfg.blockDimX cfg.blockDimY).isNone then
        none
      else if st.tmemPhase != .allocated cols isCg2 then
        none
      else if st.mmaUncommitted || !st.pendingLoads.isEmpty || !st.syncedSinceTmemUse then
        none
      else
        some { st with tmemPhase := .freed cols isCg2 }

/-- Execute all operations in a basic block sequentially from `st`. -/
def execBlockOps (cfg : ControlFlowCfg) (b : Nat) (st : CtaState) : List CfgOp → Option CtaState
  | [] => some st
  | op :: rest =>
      match stepOp cfg (activeGuardAt cfg b) (isCyclicBlockBool cfg b) st op with
      | none => none
      | some st' => execBlockOps cfg b st' rest

/-- Apply an outgoing terminator edge label `cond` to post-block state `st`. -/
def stepEdge (st : CtaState) (cond : BranchCond) : Option CtaState :=
  match cond with
  | .mbarReady barId =>
      match st.barState[barId]? with
      | some .armedPlain | some .armedTx | some .completed =>
          let nextReady :=
            if st.tmemCommitBar == some barId && !st.mmaUncommitted then true else st.tmemReady
          some
            { st with
              barState := st.barState.set barId .completed
              tmemReady := nextReady }
      | _ => none
  | .mbarSpin barId =>
      match st.barState[barId]? with
      | some .armedPlain | some .armedTx | some .completed => some st
      | _ => none
  | .always | .coord _ | .uniform | .unknown => some st

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
      | .allocated _ _ => false
  phaseOk && !st.mmaUncommitted && st.pendingLoads.isEmpty

/-- Indices of all basic blocks in `cfg` terminating in `mir.return`. -/
def returnBlocks (cfg : ControlFlowCfg) : List Nat :=
  (List.range cfg.blocks.length).filter fun b => (cfg.blockAt b).isReturn

/-- Indices of all basic blocks in `cfg` containing at least one `.syncThreads` barrier. -/
def syncBarrierBlocks (cfg : ControlFlowCfg) : List Nat :=
  (List.range cfg.blocks.length).filter fun b =>
    (cfg.blockAt b).ops.any (· == .syncThreads)

/-- Cyclic basic blocks in `cfg` containing `.syncThreads`. -/
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

/-- Barrier IDs signaled anywhere in `cfg` by `mbarArrive`, `mbarArriveExpectTx`, or `tmemCommit`. -/
def signaledBarrierIds (cfg : ControlFlowCfg) : List Nat :=
  (List.range cfg.numBarriers).filter fun barId =>
    cfg.blocks.any fun blk =>
      blk.ops.any fun
        | .mbarArrive b | .mbarArriveExpectTx b _ | .tmemCommit b => b == barId
        | _ => false

/-- Target blocks of `.mbarReady barId` edges for signaled `Barrier`s in `cfg`. -/
def signaledMbarReadyBlocks (cfg : ControlFlowCfg) : List Nat :=
  let sig := signaledBarrierIds cfg
  cfg.blocks.flatMap fun blk =>
    blk.succs.filterMap fun
      | (.mbarReady barId, dst) => if barId ∈ sig then some dst else none
      | _ => none

/-- Total number of `sync_threads()` operations plus signaled `mbarrier` wait edges in `cfg`. -/
def countEffectiveBarriers (cfg : ControlFlowCfg) : Nat :=
  let syncCount := (cfg.blocks.map fun blk => (blk.ops.filter (· == .syncThreads)).length).sum
  syncCount + (signaledMbarReadyBlocks cfg).length

/-- Barrier IDs signaled by `tcgen05_commit` in `cfg`. -/
def committedTmemBarrierIds (cfg : ControlFlowCfg) : List Nat :=
  (List.range cfg.numBarriers).filter fun barId =>
    cfg.blocks.any fun blk =>
      blk.ops.any fun
        | .tmemCommit b => b == barId
        | _ => false

/-- Triples `(prodBlk, consBlk, backpressureBlocks)` for cyclic async TMA and `TMEM` loops.
    Every loop backedge from `consBlk` to `prodBlk` must cross `backpressureBlocks`. -/
def loopBackpressurePairs (cfg : ControlFlowCfg) : List (Nat × Nat × List Nat) :=
  let indices := List.range cfg.blocks.length
  let syncs := syncBarrierBlocks cfg
  let sig := signaledBarrierIds cfg
  let tmemCommitted := committedTmemBarrierIds cfg
  let smemPairs := (List.range cfg.numShared).flatMap fun s =>
    let prods := indices.flatMap fun b =>
      if isCyclicBlockBool cfg b then
        (cfg.blockAt b).ops.filterMap fun
          | .tmaCopyG2S sid barId => if sid == s then some (b, barId) else none
          | _ => none
      else
        []
    let cons := indices.filter fun b =>
      (cfg.blockAt b).ops.any fun
        | .sharedRead sid => sid == s
        | .tmemAsyncIssue sids => s ∈ sids
        | _ => false
    prods.flatMap fun (pBlk, fullBar) =>
      let emptyReady := cfg.blocks.flatMap fun blk =>
        blk.succs.filterMap fun
          | (.mbarReady bId, dst) => if bId != fullBar && bId ∈ sig then some dst else none
          | _ => none
      let bp := syncs ++ emptyReady
      (cons.filter fun cBlk => canReachAvoiding cfg cBlk pBlk []).map fun cBlk =>
        (pBlk, cBlk, bp)
  let tmemProds := indices.filter fun b =>
    isCyclicBlockBool cfg b &&
    (cfg.blockAt b).ops.any fun
      | .tmemAsyncIssue _ => true
      | _ => false
  let tmemCons := indices.filter fun b =>
    (cfg.blockAt b).ops.any fun
      | .tmemLoad _ => true
      | _ => false
  let tmemEmptyReady := cfg.blocks.flatMap fun blk =>
    blk.succs.filterMap fun
      | (.mbarReady bId, dst) => if !(bId ∈ tmemCommitted) && bId ∈ sig then some dst else none
      | _ => none
  let tmemBp := syncs ++ tmemEmptyReady
  let tmemPairs := tmemProds.flatMap fun pBlk =>
    (tmemCons.filter fun cBlk => canReachAvoiding cfg cBlk pBlk []).map fun cBlk =>
      (pBlk, cBlk, tmemBp)
  smemPairs ++ tmemPairs

/-- Column counts requested by all `.tmemAlloc` operations in `cfg`. -/
def tmemAllocSites (cfg : ControlFlowCfg) : List Nat :=
  cfg.blocks.flatMap fun blk =>
    blk.ops.filterMap fun
      | .tmemAlloc _ cols _ => some cols
      | _ => none

/-- Column counts passed to all `.tmemDealloc` operations in `cfg`. -/
def tmemDeallocSites (cfg : ControlFlowCfg) : List Nat :=
  cfg.blocks.flatMap fun blk =>
    blk.ops.filterMap fun
      | .tmemDealloc cols _ => some cols
      | _ => none

end CudaOxide.Mir.ControlFlow
