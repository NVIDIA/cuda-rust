/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide.Mir.Launch

/-!
# Guarded 1D Kernel Access Representation (`CudaOxide.Mir.Kernel`)

Represents the buffer-access structure extracted by `lean-exporter` from a
`PostMem2Reg` `dialect-mir` kernel function:

```text
  dialect-mir CFG (PostMem2Reg)          CudaOxide.Mir.Kernel
  ──────────────────────────────────     ──────────────────────────────────────
  idx = thread::index_1d()               `globalIndex blockSize block lane`
  if idx < guardParam.len() {            `guardParam : Fin arity`
    ... load  buf[idx] ...               `AccessKind.read`  on `param : Fin arity`
    ... store buf[idx] ...               `AccessKind.write` on `param : Fin arity`
  }
```
-/
namespace CudaOxide.Mir.Kernel

/-- Direction of a buffer element operation in `dialect-mir`. -/
inductive AccessKind where
  /-- Read from a slice or buffer parameter (`mir.load` on `mir.ptr_offset`). -/
  | read
  /-- Write to a mutable slice or `DisjointSlice` (`mir.store` on `mir.ptr_offset`). -/
  | write
  deriving Repr, DecidableEq

/-- A single guarded 1D buffer access extracted from `PostMem2Reg` `dialect-mir`.
    Every access occurs under a strict dominating branch guard
    `globalIndex blockSize block lane + indexOffset < lengths guardParam`. -/
structure GuardedAccess1D (arity : Nat) where
  /-- Whether this operation loads from or stores to the target buffer. -/
  kind : AccessKind
  /-- Kernel parameter position of the accessed buffer (`0 ≤ targetParam < arity`). -/
  targetParam : Fin arity
  /-- Kernel parameter position whose `.len()` bounds the dominating branch guard. -/
  guardParam : Fin arity
  /-- Constant element offset added to the 1D thread index (`0` for `buf[idx]`). -/
  indexOffset : Nat := 0
  deriving Repr, DecidableEq

/-- Summary of a 1D kernel extracted from `dialect-mir` at `PostMem2Reg`. -/
structure KernelSummary1D (arity : Nat) where
  /-- Static X dimension of the thread block (`#[launch_contract(block = (x, 1, 1))]`). -/
  blockSize : Nat
  /-- All buffer reads and writes reachable in the kernel CFG. -/
  accesses : List (GuardedAccess1D arity)
  deriving Repr, DecidableEq

/-- Effective element index accessed by thread `(block, lane)` for `access`. -/
def GuardedAccess1D.address {arity : Nat}
    (access : GuardedAccess1D arity) (blockSize block lane : Nat) : Nat :=
  CudaOxide.Mir.globalIndex blockSize block lane + access.indexOffset

/-- Memory address space of a buffer access. -/
inductive MemorySpace where
  /-- Global device memory (`&[T]`, `DisjointSlice`). -/
  | global
  /-- Per-block shared memory (`SharedArray<T, N>`). -/
  | shared
  /-- Per-SM Blackwell Tensor Memory (`TMEM`, 128 lanes x up to 512 columns). -/
  | tmem
  deriving Repr, DecidableEq

/-- Compositional index expression over thread coordinates `(tx, ty)` and loop step `i`. -/
inductive IdxExpr where
  /-- Thread X coordinate (`threadIdx.x`). -/
  | tx
  /-- Thread Y coordinate (`threadIdx.y`). -/
  | ty
  /-- Inner scan loop induction variable `i` (`0 ≤ i < loopBound`). -/
  | loopVar
  /-- Constant natural-number literal. -/
  | const (n : Nat)
  /-- Sum of two index expressions (`a + b`). -/
  | add (a b : IdxExpr)
  /-- Product of two index expressions (`a * b`). -/
  | mul (a b : IdxExpr)
  /-- Euclidean quotient `a / d` (from `/` or `>> k`). -/
  | div (a : IdxExpr) (d : Nat)
  /-- Euclidean remainder `a % d` (from `%` or `& (2^k - 1)`). -/
  | mod (a : IdxExpr) (d : Nat)
  deriving Repr, DecidableEq

/-- Unfold an `IdxExpr` syntax tree into a `Nat` formula over `(tx, ty, i)`. -/
def IdxExpr.eval (e : IdxExpr) (tx ty i : Nat) : Nat :=
  match e with
  | .tx => tx
  | .ty => ty
  | .loopVar => i
  | .const n => n
  | .add a b => a.eval tx ty i + b.eval tx ty i
  | .mul a b => a.eval tx ty i * b.eval tx ty i
  | .div a d => a.eval tx ty i / d
  | .mod a d => a.eval tx ty i % d

/-- Compositional branch-guard predicate over `(tx, ty, i)`. -/
inductive GuardExpr where
  /-- Unguarded (`True`). -/
  | tr
  /-- Strict upper bound `a < b` (e.g. `tid < 64`). -/
  | lt (a b : IdxExpr)
  /-- Lower bound `a ≥ b` (negation of `a < b`). -/
  | ge (a b : IdxExpr)
  /-- Equality guard `a = b` (e.g. `lane == 0` or `lane == 7`). -/
  | eq (a b : IdxExpr)
  /-- Inequality guard `a ≠ b` (negation of `a = b`). -/
  | ne (a b : IdxExpr)
  /-- Conjunction of two dominating branch guards (`p ∧ q`). -/
  | and (p q : GuardExpr)
  deriving Repr, DecidableEq

/-- Unfold a `GuardExpr` syntax tree into a `Prop` over `(tx, ty, i)`. -/
def GuardExpr.eval (g : GuardExpr) (tx ty i : Nat) : Prop :=
  match g with
  | .tr => True
  | .lt a b => a.eval tx ty i < b.eval tx ty i
  | .ge a b => a.eval tx ty i ≥ b.eval tx ty i
  | .eq a b => a.eval tx ty i = b.eval tx ty i
  | .ne a b => a.eval tx ty i ≠ b.eval tx ty i
  | .and p q => p.eval tx ty i ∧ q.eval tx ty i

/-- How threads in a block index into a shared-memory array (`SharedArray`). -/
inductive TileAccessPattern where
  /-- One fixed slot per thread (`TILE[ty * stride + tx]`). -/
  | threadSlot (stride : Nat)
  /-- Walk across row `ty`, one element per loop step `i` (`TILE[ty * stride + i]`). -/
  | rowScan (stride : Nat)
  /-- Walk down column `tx`, one element per loop step `i` (`TILE[i * stride + tx]`). -/
  | colScan (stride : Nat)
  /-- One slot per warp or sub-warp team of `groupSize` threads (`WARP_SUMS[tx / groupSize]`). -/
  | groupSlot (groupSize : Nat)
  /-- Walk straight across a 1D array, one element per loop step `i` (`WARP_SUMS[i]`). -/
  | linearScan
  /-- A contiguous chunk of `pack` slots owned by each thread `tx` (`S_X[tx * pack + i]`). -/
  | packedSlot (pack : Nat)
  /-- A chunk of `pack` slots indexed by a thread's lane `tx % groupSize` inside its sub-warp (`S_XS[(tx % groupSize) * pack + i]`). -/
  | groupLaneScan (groupSize pack : Nat)
  /-- A 1D thread ID `tx` split into a 2D `(row = tx / groupSize, col = tx % groupSize)` cell (`TILE[(tx / groupSize) * stride + (tx % groupSize)]`). -/
  | bitSplitSlot (groupSize stride : Nat)
  /-- General compositional index expression `e` built from SSA operations. -/
  | expr (e : IdxExpr)
  deriving Repr, DecidableEq

/-- Evaluate the flat element index of a shared-memory access for thread `(tx, ty)` at step `i`. -/
def TileAccessPattern.address (pat : TileAccessPattern) (tx ty i : Nat) : Nat :=
  match pat with
  | .threadSlot stride => CudaOxide.Mir.globalIndex stride ty tx
  | .rowScan stride => CudaOxide.Mir.globalIndex stride ty i
  | .colScan stride => CudaOxide.Mir.globalIndex stride i tx
  | .groupSlot groupSize => tx / groupSize
  | .linearScan => i
  | .packedSlot pack => CudaOxide.Mir.globalIndex pack tx i
  | .groupLaneScan groupSize pack => CudaOxide.Mir.globalIndex pack (tx % groupSize) i
  | .bitSplitSlot groupSize stride =>
      CudaOxide.Mir.globalIndex stride (tx / groupSize) (tx % groupSize)
  | .expr e => e.eval tx ty i

/-- A shared-memory tile access tagged with its barrier phase (`0`, `1`, ...). -/
structure PhasedTileAccess where
  /-- Read or write access. -/
  kind : AccessKind
  /-- Memory space (`.shared` for `SharedArray`). -/
  space : MemorySpace := .shared
  /-- Identifier of the target buffer (`0` for `TILE_A`, `1` for `TILE_B`, ...). -/
  bufferId : Nat
  /-- Barrier phase index (`0` before first `sync_threads()`, `1` after first, etc.). -/
  phase : Nat
  /-- Coordinate formula used to index the buffer. -/
  pattern : TileAccessPattern
  /-- Exclusive upper bound on loop index `i` (`1` if not inside an inner scan loop). -/
  loopBound : Nat := 1
  /-- True when guarded by a leader check (`tx % groupSize = 0` or `tx = 0`). -/
  leaderOnly : Bool := false
  /-- Optional compositional branch guard (`GuardExpr.tr` when unguarded or covered by `leaderOnly`). -/
  guard : GuardExpr := .tr
  deriving Repr, DecidableEq

/-- Predicate describing which threads `tx` execute `a` (all threads when
    unguarded; only threads satisfying `a.guard` and/or `leaderOnly` when guarded). -/
def PhasedTileAccess.active (a : PhasedTileAccess) (tx : Nat) : Prop :=
  let leaderProp :=
    match a.leaderOnly, a.pattern with
    | true, .groupSlot groupSize => tx % groupSize = 0
    | true, _ => tx = 0
    | false, _ => True
  match a.guard with
  | .tr => leaderProp
  | g => leaderProp ∧ g.eval tx 0 0

/-- Summary of a 2D phased shared-memory kernel extracted at `PostMem2Reg`. -/
structure TiledKernelSummary2D where
  /-- Static X dimension of the thread block (`reqntid_x`). -/
  blockDimX : Nat
  /-- Static Y dimension of the thread block (`reqntid_y`). -/
  blockDimY : Nat
  /-- Number of `sync_threads()` barriers in the tile loop. -/
  numBarriers : Nat
  /-- Whether all barriers execute in thread-uniform control flow. -/
  barriersUniform : Bool
  /-- All phased shared-memory accesses extracted from the kernel CFG. -/
  sharedAccesses : List PhasedTileAccess
  deriving Repr, DecidableEq

/-- A Blackwell Tensor Memory (`TMEM`) access (`tcgen05.mma`, `tcgen05.cp`,
    `tcgen05.ld`, or `tcgen05.st`) extracted at `PostMem2Reg`. -/
structure TmemAccess where
  /-- Starting TMEM lane expression (`(taddr >> 16) & 0xFFFF`, within `0..128`). -/
  laneExpr : IdxExpr
  /-- Starting TMEM column expression (`taddr & 0xFFFF`, within `0..allocCols`). -/
  colExpr : IdxExpr
  /-- Number of TMEM lanes touched by the instruction (`16`, `32`, `64`, or `128`). -/
  heightLanes : Nat
  /-- Number of 32-bit TMEM columns touched by the instruction (`1..256`). -/
  widthCols : Nat
  /-- Exclusive upper bound on loop index `i` (`1` when outside a loop). -/
  loopBound : Nat := 1
  /-- True for `tcgen05.ld` / `tcgen05.st`, which must stay inside the warp's 32-lane slice. -/
  warpRestricted : Bool := true
  /-- Dominating thread guard (`GuardExpr.tr` when all block threads participate). -/
  guard : GuardExpr := .tr
  deriving Repr, DecidableEq

end CudaOxide.Mir.Kernel
