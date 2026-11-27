/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import Lean.ToExpr

/-!
# Line-by-Line `PostMem2Reg` MIR Program & Semantics (`CudaOxide.Mir.Program`)

Represents a `PostMem2Reg` `dialect-mir` GPU kernel function 1-to-1 as basic
blocks of SSA statements and terminators, together with its small-step per-thread
operational semantics:

```text
  PostMem2Reg `dialect-mir`              `CudaOxide.Mir.Program`
  ----------------------------------     --------------------------------------
  `^bb(v0: ui32, v1: fp32)`              `BasicBlock { args, stmts, term }`
  `v2 = mir.add (v0, c1)`                `Stmt.assign 2 (.binOp .add 32 0 1)`
  `p  = mir.ptr_offset (base, idx)`      `Stmt.assign p (.ptrOffset base idx)`
  `x  = mir.load (p)` / `mir.store`      `Stmt.assign x (.load p)` / `Stmt.store`
  `nvvm.barrier0`                        `Stmt.syncThreads`
  `mir.cond_br` / `mir.goto` / `return`  `Terminator.condBr` / `.goto` / `.ret`
```
-/
namespace CudaOxide.Mir.Program

/-- Memory address space of a pointer value in `dialect-mir`. -/
inductive AddrSpace where
  /-- Global device memory (`&[T]`, `&mut [T]`, `DisjointSlice<T>`). -/
  | global
  /-- CTA shared memory (`SharedArray<T, N>`). -/
  | shared
  /-- Thread-local stack slot (`mir.alloca`). -/
  | localStack
  deriving Repr, BEq, DecidableEq, Lean.ToExpr

instance : LawfulBEq AddrSpace where
  rfl := by intro a; cases a <;> rfl
  eq_of_beq := by intro a b; cases a <;> cases b <;> decide

/-- Parameter kind on a kernel `mir.func` entry signature. -/
inductive ParamKind where
  /-- Scalar unsigned integer parameter of `width` bits (`0` for float/opaque scalars). -/
  | scalar (width : Nat)
  /-- Slice parameter `&[T]` (`mutable = false`) or `&mut [T]` (`mutable = true`). -/
  | slice (mutable : Bool)
  /-- Host-partitioned `DisjointSlice<T>` parameter. -/
  | disjointSlice
  deriving Repr, BEq, DecidableEq

instance : LawfulBEq ParamKind where
  rfl := by
    intro a; cases a <;> (try rfl) <;>
      (change instBEqParamKind.beq _ _ = true; simp [instBEqParamKind.beq])
  eq_of_beq := by
    intro a b h; cases a <;> cases b <;> (try contradiction) <;>
      (change instBEqParamKind.beq _ _ = true at h; simp [instBEqParamKind.beq] at h; try simp_all)

/-- Static `SharedArray` declaration inside a `mir.func`. -/
structure SharedDecl where
  /-- Element capacity of the `SharedArray`. -/
  size : Nat
  /-- Byte width of one element. -/
  elemBytes : Nat := 4
  deriving Repr, BEq, DecidableEq

/-- Hardware PTX special register read inside a kernel. -/
inductive Sreg where
  /-- `threadIdx.x` (`nvvm.read_ptx_sreg_tid_x`). -/
  | tidX
  /-- `threadIdx.y` (`nvvm.read_ptx_sreg_tid_y`). -/
  | tidY
  /-- `blockIdx.x` (`nvvm.read_ptx_sreg_ctaid_x`). -/
  | ctaidX
  /-- `blockIdx.y` (`nvvm.read_ptx_sreg_ctaid_y`). -/
  | ctaidY
  /-- `blockDim.x` (`nvvm.read_ptx_sreg_ntid_x`). -/
  | ntidX
  /-- `blockDim.y` (`nvvm.read_ptx_sreg_ntid_y`). -/
  | ntidY
  /-- Warp lane index `(threadIdx.y * blockDim.x + threadIdx.x) % 32` (`nvvm.read_ptx_sreg_laneid`). -/
  | laneId
  /-- CTA cluster X rank `blockIdx.x % clusterDimX` (`nvvm.read_ptx_sreg_cluster_ctaid_x`). -/
  | clusterCtaidX
  deriving Repr, BEq, DecidableEq

/-- Unsigned integer binary arithmetic / bitwise operator in `dialect-mir`. -/
inductive BinOp where
  /-- Wrapping `width`-bit addition (`mir.add`). -/
  | add
  /-- Wrapping `width`-bit subtraction (`mir.sub`). -/
  | sub
  /-- Wrapping `width`-bit multiplication (`mir.mul`). -/
  | mul
  /-- Unsigned integer quotient (`mir.div`). -/
  | div
  /-- Unsigned integer remainder (`mir.rem`). -/
  | rem
  /-- Bitwise AND (`mir.bitand`). -/
  | bitAnd
  /-- Bitwise OR (`mir.bitor`). -/
  | bitOr
  /-- Bitwise XOR (`mir.bitxor`). -/
  | bitXor
  /-- Left bit-shift (`mir.shl`). -/
  | shl
  /-- Logical right bit-shift (`mir.shr`). -/
  | shr
  deriving Repr, BEq, DecidableEq, Lean.ToExpr

instance : LawfulBEq BinOp where
  rfl := by intro a; cases a <;> rfl
  eq_of_beq := by intro a b; cases a <;> cases b <;> decide

/-- Unsigned integer comparison operator in `dialect-mir`. -/
inductive CmpOp where
  /-- Strict less-than `<` (`mir.lt`). -/
  | lt
  /-- Less-than-or-equal `<=` (`mir.le`). -/
  | le
  /-- Strict greater-than `>` (`mir.gt`). -/
  | gt
  /-- Greater-than-or-equal `>=` (`mir.ge`). -/
  | ge
  /-- Equality `==` (`mir.eq`). -/
  | eq
  /-- Inequality `!=` (`mir.ne`). -/
  | ne
  deriving Repr, BEq, DecidableEq, Lean.ToExpr

instance : LawfulBEq CmpOp where
  rfl := by intro a; cases a <;> rfl
  eq_of_beq := by intro a b; cases a <;> cases b <;> decide

/-- Right-hand side of a single SSA assignment statement in `PostMem2Reg` MIR. -/
inductive Rhs where
  /-- Copy / alias an existing SSA variable (`mir.cast` pointer reborrow, etc.). -/
  | copy (src : Nat)
  /-- Unsigned integer constant literal of `width` bits (`mir.constant`). -/
  | constInt (width val : Nat)
  /-- Boolean constant literal (`true` or `false`). -/
  | constBool (b : Bool)
  /-- Hardware special register read (`threadIdx`, `blockIdx`, `blockDim`, `lane_id`). -/
  | sreg (reg : Sreg)
  /-- 64-bit 1D global thread index `blockIdx.x * blockDim.x + threadIdx.x` (`thread::index_1d`). -/
  | index1D
  /-- Packed 2D `u32` coordinates `((row as u64) << 32) + (col as u64)` (`thread::coords_2d_u32`). -/
  | coord2DU32
  /-- Scalar entry parameter `param` (`0 .. params.length - 1`). -/
  | paramScalar (param : Nat)
  /-- Base pointer of slice parameter `param` (`mir.extract_field` index `0`). -/
  | slicePtr (param : Nat)
  /-- Element length of slice parameter `param` (`mir.extract_field` index `1`). -/
  | sliceLen (param : Nat)
  /-- Base pointer of static `SharedArray` `sharedId` (`mir.shared_alloc`). -/
  | sharedBase (sharedId : Nat)
  /-- Unsigned `width`-bit binary operation on SSA variables `lhs` and `rhs`. -/
  | binOp (op : BinOp) (width lhs rhs : Nat)
  /-- Unsigned `width`-bit overflow boolean (`true` iff `lhs op rhs` wraps/underflows `width` bits). -/
  | checkedOvf (op : BinOp) (width lhs rhs : Nat)
  /-- Unsigned integer comparison on SSA variables `lhs` and `rhs`. -/
  | cmpOp (op : CmpOp) (lhs rhs : Nat)
  /-- Boolean negation (`mir.not`). -/
  | notBool (src : Nat)
  /-- Unsigned integer width cast to `dstWidth` bits (`mir.cast IntToInt`). -/
  | intCast (dstWidth src : Nat)
  /-- Element pointer offset `base + idx` (`mir.ptr_offset`). -/
  | ptrOffset (base idx : Nat)
  /-- Memory load from pointer variable `ptr` (`mir.load`). -/
  | load (ptr : Nat)
  /-- Hardware `mbarrier_try_wait*(&BAR[slot], parity)` poll (`isAcquire = false` for `.relaxed` waits). -/
  | mbarTryWait (barId slotVar : Nat) (isAcquire : Bool := true)
  /-- `.shared::cta` byte offset `cvta_generic_to_shared_offset(ptr)` (`nvvm.cvta_generic_to_shared_offset`). -/
  | cvtaShared (ptr : Nat)
  /-- Load the base `TMEM` handle of `cols` columns from shared-memory slot `ptr` (`TMEM_ADDR[0]`). -/
  | loadTmemBase (ptr cols : Nat)
  /-- Arbitrary `width`-bit unsigned integer produced by an opaque call/extract (`id`, `width`). -/
  | opaqueUint (id width : Nat)
  /-- Arbitrary `width`-bit unsigned integer whose CTA-uniformity follows `src` (`clc_query_*`, race-free SMEM `u32`). -/
  | ctaUint1 (id width src : Nat)
  /-- Zero-operand CTA-uniform non-index constant (`mir.float_constant`). -/
  | ctaConst
  /-- Single-operand pure combinational / CTA-query op (`clc_query_is_canceled`). -/
  | ctaUnary (src : Nat)
  /-- Two-operand pure combinational non-index op (floating-point comparison). -/
  | ctaBin (lhs rhs : Nat)
  /-- Non-index floating-point or aggregate value (`fp16`, `bf16`, `fp32`, etc.). -/
  | opaqueVal
  deriving Repr, DecidableEq

/-- Non-terminating statement inside a `BasicBlock`, in exact program order. -/
inductive Stmt where
  /-- SSA variable definition `dst = rhs`. -/
  | assign (dst : Nat) (rhs : Rhs)
  /-- Memory store `*ptr = val` (`mir.store`). -/
  | store (ptr val : Nat)
  /-- Bounds or invariant assertion (`mir.assert`), carrying a deterministic `assertId`. -/
  | assertCond (assertId cond : Nat)
  /-- CTA-wide barrier `thread::sync_threads()` (`nvvm.barrier0`). -/
  | syncThreads
  /-- Bounds-checked slot reference on a `SharedArray<Barrier, N>` pointer `ptr`. -/
  | barrierSlotCheck (ptr : Nat)
  /-- Bounds-checked peer CTA rank (`rank < clusterDimX`) on `cluster::map_shared_rank*`. -/
  | clusterRankCheck (rankVar : Nat)
  /-- Plain thread arrival `mbarrier_arrive*(&BAR[slot])` (`isRelease = false` for `.relaxed` arrives). -/
  | mbarArrive (barId slotVar : Nat) (isRelease : Bool := true)
  /-- Byte-counted arrival `mbarrier_arrive_expect_tx(&BAR[slot], count, txBytes)`. -/
  | mbarArriveExpectTx (barId slotVar txBytes : Nat)
  /-- Asynchronous TMA global-to-shared copy writing `spanElems` elements at `ptr` on `(barId, slotVar)`. -/
  | tmaCopyG2S (ptr barId slotVar spanElems : Nat)
  /-- Warp-collective `tcgen05_alloc(ptr, cols)` writing `TMEM` handle slot `ptr`. -/
  | tmemAlloc (ptr cols : Nat)
  /-- Asynchronous `tcgen05` shared-memory descriptor read of `tileBytes` bytes from `descVar`. -/
  | smemDescRead (descVar tileBytes : Nat)
  /-- Blackwell Tensor Memory (`TMEM`) access (`tcgen05.mma`, `tcgen05.cp`, or `tcgen05.ld`). -/
  | tmemAccess (tmemVar heightLanes widthCols : Nat) (warpRestricted : Bool)
  deriving Repr, DecidableEq

/-- Block terminator transferring control to successor blocks or returning. -/
inductive Terminator where
  /-- Kernel return (`mir.return`). -/
  | ret
  /-- Unreachable exhaustiveness sink (`mir.unreachable`). -/
  | unreachable
  /-- Unconditional jump passing `args` to `target`'s block parameters (`mir.goto`). -/
  | goto (target : Nat) (args : List Nat)
  /-- Conditional branch on boolean `cond` (`mir.cond_br`). -/
  | condBr (cond : Nat) (thenBlk : Nat) (thenArgs : List Nat) (elseBlk : Nat) (elseArgs : List Nat)
  deriving Repr, DecidableEq

/-- Block parameter `(varId, width)` on a `BasicBlock` (`width = 0` for opaque/float, `1` for bool). -/
structure BlockArg where
  /-- SSA variable identifier bound on entry to this block. -/
  varId : Nat
  /-- Integer bit width (`8`, `16`, `32`, `64`), `1` for `i1` bool, or `0` for opaque/float. -/
  width : Nat
  deriving Repr, DecidableEq

/-- One basic block in a transcribed `PostMem2Reg` `MirFunc`. -/
structure BasicBlock where
  /-- SSA block parameters populated by predecessor `goto` / `condBr` operands. -/
  args : List BlockArg := []
  /-- True when this block is a cyclic loop header (`dst <= src` backedge target). -/
  isLoopHeader : Bool := false
  /-- Ordered non-terminating statements inside this block. -/
  stmts : List Stmt := []
  /-- Block-ending control-flow terminator. -/
  term : Terminator := .ret
  deriving Repr, DecidableEq

/-- Complete `PostMem2Reg` kernel function transcribed line by line from `dialect-mir`. -/
structure MirFunc where
  /-- Launch domain dimension (`1` or `2`). -/
  domain : Nat := 1
  /-- Static CTA X thread count (`reqntid_x`). -/
  blockDimX : Nat
  /-- Static CTA Y thread count (`reqntid_y`, `1` for 1D). -/
  blockDimY : Nat := 1
  /-- Static CTA cluster X count (`cluster_dim_x`, `1` when non-clustered). -/
  clusterDimX : Nat := 1
  /-- True when `u32_coordinates` launch contract guarantees `bx * blockDimX + tx < 2^32`. -/
  u32Coords : Bool := false
  /-- Kernel entry parameter kinds (`0 .. params.length - 1`). -/
  params : List ParamKind
  /-- Host-checked slice length relations `(lhs, rhs)` meaning `len(lhs) >= len(rhs)`. -/
  requiresLenGe : List (Nat × Nat) := []
  /-- Static `SharedArray` declarations (`0 .. sharedDecls.length - 1`). -/
  sharedDecls : List SharedDecl := []
  /-- `SharedArray` IDs proved race-free (`no_data_race`), whose uniform-index reads are CTA-uniform. -/
  raceFreeShared : List Nat := []
  /-- Topologically ordered basic blocks (`0` is the entry block). -/
  blocks : List BasicBlock
  deriving Repr, DecidableEq

/-- Look up basic block `b` in `f` (returning an empty `unreachable` block if out of bounds). -/
def MirFunc.blockAt (f : MirFunc) (b : Nat) : BasicBlock :=
  f.blocks.getD b { term := .unreachable }

/-- True when `f` signals `Barrier` group `barId` via release `mbarrier_arrive` or positive `mbarrier_arrive_expect_tx`. -/
def MirFunc.hasValidBarSignal (f : MirFunc) (barId : Nat) : Bool :=
  f.blocks.any fun blk =>
    blk.stmts.any fun
      | .mbarArrive b _ rel => b == barId && rel
      | .mbarArriveExpectTx b _ tx => b == barId && decide (0 < tx)
      | _ => false

/-- Coordinates of one executing GPU hardware thread. -/
structure ThreadCtx where
  /-- CTA grid X coordinate (`blockIdx.x`). -/
  bx : Nat
  /-- CTA grid Y coordinate (`blockIdx.y`). -/
  by_ : Nat
  /-- Intra-CTA thread X coordinate (`threadIdx.x`). -/
  tx : Nat
  /-- Intra-CTA thread Y coordinate (`threadIdx.y`). -/
  ty : Nat
  deriving Repr, DecidableEq

/-- True when `t` lies within the valid `(blockDimX, blockDimY)` CTA bounds and 32-bit PTX
    `ctaid` hardware range of `f` (`by_ = 0` on 1D kernels). -/
def MirFunc.validThread (f : MirFunc) (t : ThreadCtx) : Prop :=
  t.tx < f.blockDimX ∧
  t.ty < f.blockDimY ∧
  t.bx < 4294967296 ∧
  t.by_ < 4294967296 ∧
  (if f.domain == 1 then t.by_ = 0 else True) ∧
  (if f.u32Coords then t.bx * f.blockDimX + t.tx < 4294967296 else True)

/-- Uniform kernel launch arguments (`scalar` values, `sliceLen` element counts, `mbarFlag`
    completion witnesses, and per-thread opaque uints). -/
structure InputEnv where
  /-- Runtime unsigned value of scalar parameter `p`. -/
  scalar : Nat → Nat
  /-- Runtime element count `.len()` of slice parameter `p`. -/
  sliceLen : Nat → Nat
  /-- Completion witness for `mbarrier_try_wait_parity` on `(barId, slot)`. -/
  mbarFlag : Nat → Nat → Bool := fun _ _ => false
  /-- Arbitrary runtime unsigned value returned by an opaque operation `id` on thread `t`. -/
  opaqueUint : ThreadCtx → Nat → Nat := fun _ _ => 0

/-- Conjunction of bit-width bounds `env.scalar i < 2 ^ w` and `env.sliceLen i < 2 ^ 64` over `params`. -/
def paramBoundsProp (env : InputEnv) : Nat → List ParamKind → Prop
  | _, [] => True
  | idx, (.scalar w) :: rest =>
      (if 0 < w then env.scalar idx < 2 ^ w else True) ∧ paramBoundsProp env (idx + 1) rest
  | idx, _ :: rest =>
      env.sliceLen idx ≤ 9223372036854775807 ∧ paramBoundsProp env (idx + 1) rest

/-- True when `env` satisfies scalar/slice bit-width bounds and `requiresLenGe` relations of `f`. -/
def MirFunc.validInput (f : MirFunc) (env : InputEnv) : Prop :=
  paramBoundsProp env 0 f.params ∧
  (∀ req : Nat × Nat, req ∈ f.requiresLenGe → env.sliceLen req.2 ≤ env.sliceLen req.1)

/-- Runtime value held by an SSA variable during concrete execution of `MirFunc`. -/
inductive Val where
  /-- Unsigned machine integer value. -/
  | int (n : Nat)
  /-- Boolean predicate value (`i1`). -/
  | bool (b : Bool)
  /-- Typed element pointer `(space, bufId, elemOffset)`. -/
  | ptr (space : AddrSpace) (bufId offset : Nat)
  /-- `.shared::cta` byte offset into `SharedArray` `sharedId` (`cvta_generic_to_shared_offset`). -/
  | smemOff (sharedId byteOff : Nat)
  /-- 64-bit `tcgen05` shared-memory matrix descriptor into `SharedArray` `sharedId` at `byteOff`. -/
  | smemDesc (sharedId byteOff : Nat)
  /-- 32-bit Blackwell `TMEM` address `(lane << 16) + col` relative to an `allocCols` allocation. -/
  | tmemPtr (allocCols relAddr : Nat)
  /-- Non-index float/aggregate value. -/
  | unk
  deriving Repr, DecidableEq

/-- Wrap `n` modulo `2 ^ width` for fixed-width unsigned MIR arithmetic (`width = 0` means unbounded `Nat`). -/
def wordMod : Nat → Nat → Nat
  | 0, n => n
  | w + 1, n => n % (2 ^ (w + 1))

/-- Evaluate an unsigned `width`-bit binary MIR operator on concrete `Nat` values. -/
def evalBinOp (op : BinOp) (width a b : Nat) : Nat :=
  match op with
  | .add => wordMod width (a + b)
  | .sub =>
      if b ≤ a then wordMod width (a - b)
      else wordMod width (2 ^ width - ((b - a) % (2 ^ width)))
  | .mul => wordMod width (a * b)
  | .div => a / b
  | .rem => a % b
  | .bitAnd => a &&& b
  | .bitOr => a ||| b
  | .bitXor => a ^^^ b
  | .shl => wordMod width (a * (2 ^ (b % max 1 width)))
  | .shr => a / (2 ^ (b % max 1 width))

/-- Evaluate an unsigned integer comparison operator on concrete `Nat` values. -/
def evalCmpOp (op : CmpOp) (a b : Nat) : Bool :=
  match op with
  | .lt => decide (a < b)
  | .le => decide (a ≤ b)
  | .gt => decide (b < a)
  | .ge => decide (b ≤ a)
  | .eq => decide (a = b)
  | .ne => decide (a ≠ b)

/-- Concrete register map from SSA variable ID to its runtime `Val`. -/
abbrev RegFile := Nat → Val

/-- Update one SSA variable `dst` in `regs`. -/
def RegFile.set (regs : RegFile) (dst : Nat) (v : Val) : RegFile :=
  fun k => if k == dst then v else regs k

/-- Bind a block's formal `BlockArg` list in parallel from `callerRegs` onto `baseRegs`. -/
def bindBlockArgs (callerRegs baseRegs : RegFile) : List BlockArg → List Nat → RegFile
  | [], _ | _, [] => baseRegs
  | fa :: fas, av :: avs =>
      bindBlockArgs callerRegs (baseRegs.set fa.varId (callerRegs av)) fas avs

/-- Evaluate the unsigned `width`-bit overflow flag of `mir.checked_add` / `checked_sub` / `checked_mul`. -/
def evalCheckedOvf (op : BinOp) (width a b : Nat) : Bool :=
  if width == 0 then false
  else
    let m := 2 ^ width
    match op with
    | .add => decide (m ≤ a % m + b % m)
    | .sub => decide (a % m < b % m)
    | .mul => decide (m ≤ (a % m) * (b % m))
    | _ => true

/-- Evaluate one `Rhs` expression in `(f, env, t, regs)`. -/
def evalRhs (f : MirFunc) (env : InputEnv) (t : ThreadCtx) (regs : RegFile) : Rhs → Val
  | .copy src => regs src
  | .constInt w val => .int (wordMod w val)
  | .constBool b => .bool b
  | .sreg .tidX => .int t.tx
  | .sreg .tidY => .int t.ty
  | .sreg .ctaidX => .int t.bx
  | .sreg .ctaidY => .int t.by_
  | .sreg .ntidX => .int f.blockDimX
  | .sreg .ntidY => .int f.blockDimY
  | .sreg .laneId => .int ((t.ty * f.blockDimX + t.tx) % 32)
  | .sreg .clusterCtaidX => .int (t.bx % max 1 f.clusterDimX)
  | .index1D => .int (wordMod 64 (t.bx * f.blockDimX + t.tx))
  | .coord2DU32 =>
      let col := wordMod 64 (evalBinOp .add 32 (evalBinOp .mul 32 t.bx f.blockDimX) t.tx)
      let row := wordMod 64 (evalBinOp .add 32 (evalBinOp .mul 32 t.by_ f.blockDimY) t.ty)
      .int (evalBinOp .add 64 (evalBinOp .shl 64 row 32) col)
  | .paramScalar p => .int (env.scalar p)
  | .slicePtr p => .ptr .global p 0
  | .sliceLen p => .int (env.sliceLen p)
  | .sharedBase s => .ptr .shared s 0
  | .binOp op w l r =>
      match regs l, regs r with
      | .int a, .int b => .int (evalBinOp op w a b)
      | .smemOff sid base, .int d =>
          if op == .add then .smemOff sid (base + d)
          else if op == .shr && d == 4 then .smemDesc sid base
          else .unk
      | .int d, .smemOff sid base =>
          if op == .add then .smemOff sid (d + base) else .unk
      | .smemDesc sid base, .int d =>
          if op == .add then .smemDesc sid (base + d * 16)
          else if op == .bitAnd || op == .bitOr then .smemDesc sid base
          else .unk
      | .int d, .smemDesc sid base =>
          if op == .add then .smemDesc sid (d * 16 + base)
          else if op == .bitAnd || op == .bitOr then .smemDesc sid base
          else .unk
      | .tmemPtr cols rel, .int d =>
          if op == .add then .tmemPtr cols (wordMod w (rel + d)) else .unk
      | .int d, .tmemPtr cols rel =>
          if op == .add then .tmemPtr cols (wordMod w (d + rel)) else .unk
      | _, _ => .unk
  | .cmpOp op l r =>
      match regs l, regs r with
      | .int a, .int b => .bool (evalCmpOp op a b)
      | _, _ => .unk
  | .checkedOvf op w l r =>
      match regs l, regs r with
      | .int a, .int b => .bool (evalCheckedOvf op w a b)
      | _, _ => .unk
  | .notBool src =>
      match regs src with
      | .bool b => .bool (!b)
      | _ => .unk
  | .intCast dstW src =>
      match regs src with
      | .int a => .int (wordMod dstW a)
      | .smemOff sid b => if 32 ≤ dstW then .smemOff sid b else .unk
      | _ => .unk
  | .ptrOffset base idx =>
      match regs base, regs idx with
      | .ptr sp bid off, .int delta => .ptr sp bid (off + delta)
      | _, _ => .unk
  | .load _ => .unk
  | .mbarTryWait barId slotVar isAcquire =>
      if !isAcquire then .unk
      else
        let s := match regs slotVar with | .int k => k | _ => 0
        .bool (env.mbarFlag barId s)
  | .cvtaShared ptr =>
      match regs ptr with
      | .ptr .shared sid off =>
          let eb := max 1 (f.sharedDecls.getD sid { size := 0 }).elemBytes
          .smemOff sid (off * eb)
      | _ => .unk
  | .loadTmemBase _ _ => .unk
  | .opaqueUint id w | .ctaUint1 id w _ => .int (wordMod w (env.opaqueUint t id))
  | .ctaConst | .ctaUnary _ | .ctaBin _ _ | .opaqueVal => .unk

/-- Memory access event observed while executing a basic block. -/
structure MemEvent where
  /-- `true` for `Stmt.store` / `tmaCopyG2S` / `tmemAlloc`, `false` for `load` / `smemDescRead` / `barrierSlotCheck`. -/
  isWrite : Bool
  /-- Target address space (`.global` or `.shared`). -/
  space : AddrSpace
  /-- Parameter index (for `.global`) or `SharedArray` ID (for `.shared`). -/
  bufId : Nat
  /-- Evaluated element offset (`0`-based). -/
  index : Nat
  /-- Monotonic CTA barrier phase (`syncThreads` count executed prior to this event). -/
  phase : Nat
  /-- `some (barId, slot)` when released by an `mbarrier` arrival or TMA transfer on `BAR[slot]`. -/
  syncBar : Option (Nat × Nat) := none
  deriving Repr, DecidableEq

/-- Blackwell Tensor Memory (`TMEM`) access event observed while executing a basic block. -/
structure TmemEvent where
  /-- Dynamically allocated `TMEM` column capacity (`tcgen05_alloc`). -/
  allocCols : Nat
  /-- Evaluated 32-bit `(lane << 16) + col` address relative to the allocation base. -/
  relAddr : Nat
  /-- Number of `TMEM` lanes accessed (`4..128`). -/
  heightLanes : Nat
  /-- Number of 32-bit `TMEM` columns accessed (`1..256`). -/
  widthCols : Nat
  /-- True for `tcgen05.ld` / `tcgen05.st`, which must stay inside the warp's 32-lane slice. -/
  warpRestricted : Bool
  deriving Repr, DecidableEq

/-- Scan `stmts` within the current basic block for a release `Stmt.mbarArrive`, continuing across up
    to `hopFuel` unconditional zero-arg `.goto` blocks (`SharedArray::index` hops) if not reassigned. -/
def findBlockArriveLocal : List Stmt → Option (Nat × Nat) × Bool × List Nat
  | [] => (none, false, [])
  | .mbarArrive b s true :: _ => (some (b, s), false, [])
  | .syncThreads :: _ => (none, true, [])
  | .assign dst _ :: rest =>
      let (res, stopped, defs) := findBlockArriveLocal rest
      match res with
      | some (b, s) => if dst == s then (none, stopped, dst :: defs) else (some (b, s), stopped, dst :: defs)
      | none => (none, stopped, dst :: defs)
  | _ :: rest => findBlockArriveLocal rest

/-- Multi-hop `.goto` extension of `findBlockArriveLocal` bounded by `hopFuel` block transitions. -/
def findTrailArrive (f : MirFunc) : Nat → Terminator → List Stmt → Option (Nat × Nat)
  | hopFuel, term, stmts =>
      let (localRes, stopped, localDefs) := findBlockArriveLocal stmts
      match localRes with
      | some bs => some bs
      | none =>
          if stopped then none
          else
            match hopFuel, term with
            | fuel + 1, .goto dst [] =>
                let dstBlk := f.blockAt dst
                if dst < f.blocks.length && dstBlk.args.isEmpty then
                  match findTrailArrive f fuel dstBlk.term dstBlk.stmts with
                  | some (b, s) => if localDefs.contains s then none else some (b, s)
                  | none => none
                else none
            | _, _ => none

/-- Execute a list of statements sequentially from `(regs, phase)`, accumulating `MemEvent`s and `TmemEvent`s. -/
def execStmts (f : MirFunc) (env : InputEnv) (t : ThreadCtx)
    (regs : RegFile) (phase : Nat) (term : Terminator) : List Stmt → RegFile × Nat × List MemEvent × List TmemEvent
  | [] => (regs, phase, [], [])
  | stmt :: rest =>
      match stmt with
      | .assign dst (.load ptr) =>
          let ev := match regs ptr with
            | .ptr sp bid off => [{ isWrite := false, space := sp, bufId := bid, index := off, phase := phase }]
            | _ => []
          let (r', p', evs, tevs) := execStmts f env t (regs.set dst .unk) phase term rest
          (r', p', ev ++ evs, tevs)
      | .assign dst (.loadTmemBase ptr cols) =>
          let ev := match regs ptr with
            | .ptr sp bid off => [{ isWrite := false, space := sp, bufId := bid, index := off, phase := phase }]
            | _ => []
          let (r', p', evs, tevs) := execStmts f env t (regs.set dst (.tmemPtr cols 0)) phase term rest
          (r', p', ev ++ evs, tevs)
      | .assign dst rhs =>
          let v := evalRhs f env t regs rhs
          execStmts f env t (regs.set dst v) phase term rest
      | .store ptr _val =>
          let sbar := match findTrailArrive f 8 term rest with
            | some (b, sv) => match regs sv with | .int sVal => some (b, sVal) | _ => none
            | none => none
          let ev := match regs ptr with
            | .ptr sp bid off =>
                [{ isWrite := true, space := sp, bufId := bid, index := off,
                   phase := phase, syncBar := if sp == .shared then sbar else none }]
            | _ => []
          let (r', p', evs, tevs) := execStmts f env t regs phase term rest
          (r', p', ev ++ evs, tevs)
      | .barrierSlotCheck ptr =>
          let ev := match regs ptr with
            | .ptr sp bid off => [{ isWrite := false, space := sp, bufId := bid, index := off, phase := phase }]
            | _ => []
          let (r', p', evs, tevs) := execStmts f env t regs phase term rest
          (r', p', ev ++ evs, tevs)
      | .assertCond _ _ | .clusterRankCheck _ | .mbarArrive _ _ _ | .mbarArriveExpectTx _ _ _ =>
          execStmts f env t regs phase term rest
      | .syncThreads =>
          execStmts f env t regs (phase + 1) term rest
      | .tmaCopyG2S ptr barId slotVar spanElems =>
          let span := max 1 spanElems
          let ev := match regs ptr, regs slotVar with
            | .ptr sp bid off, .int sVal =>
                (List.range span).map fun k =>
                  { isWrite := true, space := sp, bufId := bid, index := off + k,
                    phase := phase, syncBar := some (barId, sVal) }
            | _, _ => []
          let (r', p', evs, tevs) := execStmts f env t regs phase term rest
          (r', p', ev ++ evs, tevs)
      | .tmemAlloc ptr _cols =>
          let ev :=
            if t.tx % 32 == 0 then
              match regs ptr with
              | .ptr sp bid off => [{ isWrite := true, space := sp, bufId := bid, index := off, phase := phase }]
              | _ => []
            else []
          let (r', p', evs, tevs) := execStmts f env t regs phase term rest
          (r', p', ev ++ evs, tevs)
      | .smemDescRead descVar tileBytes =>
          let span := max 1 tileBytes
          let ev := match regs descVar with
            | .smemDesc sid byteOff =>
                let eb := max 1 (f.sharedDecls.getD sid { size := 0 }).elemBytes
                (List.range span).map fun k =>
                  { isWrite := false, space := .shared, bufId := sid,
                    index := (byteOff + k) / eb, phase := phase }
            | _ => []
          let (r', p', evs, tevs) := execStmts f env t regs phase term rest
          (r', p', ev ++ evs, tevs)
      | .tmemAccess tmemVar hl wc wr =>
          let tev := match regs tmemVar with
            | .tmemPtr cols rel =>
                [{ allocCols := cols, relAddr := rel, heightLanes := hl,
                   widthCols := wc, warpRestricted := wr }]
            | _ => []
          let (r', p', evs, tevs) := execStmts f env t regs phase term rest
          (r', p', evs, tev ++ tevs)

/-- Initial register file at kernel entry (`block 0`). -/
def initialRegs (f : MirFunc) (env : InputEnv) : RegFile :=
  let b0 := f.blockAt 0
  ((List.range b0.args.length).zip b0.args).foldl (fun acc (idx, ba) =>
    match f.params[idx]? with
    | some (.scalar _) => acc.set ba.varId (.int (env.scalar idx))
    | _ => acc.set ba.varId .unk) (fun _ => .unk)

/-- Inductive small-step reachability of `(blockIdx, entryRegs, entryPhase)` for thread `t`
    executing `f` on `env` across arbitrary branches and loop iterations. -/
inductive Reachable (f : MirFunc) (env : InputEnv) (t : ThreadCtx) :
    Nat → RegFile → Nat → Prop where
  /-- Kernel starts at `block 0` in phase `0` with `initialRegs f env`. -/
  | entry (hpos : 0 < f.blocks.length) :
      Reachable f env t 0 (initialRegs f env) 0
  /-- Unconditional `mir.goto target args` transition. -/
  | stepGoto {b target phase outPhase : Nat} {regs outRegs : RegFile}
      {evs : List MemEvent} {tevs : List TmemEvent} {args : List Nat}
      (hreach : Reachable f env t b regs phase)
      (hexec : execStmts f env t regs phase (f.blockAt b).term (f.blockAt b).stmts = (outRegs, outPhase, evs, tevs))
      (hterm : (f.blockAt b).term = .goto target args)
      (hdst : target < f.blocks.length) :
      Reachable f env t target (bindBlockArgs outRegs outRegs (f.blockAt target).args args) outPhase
  /-- Conditional `mir.cond_br` taking `thenBlk` unless `outRegs cond = .bool false`.
      A `.unk` condition, such as a comparison on a loaded value, may take either arm. -/
  | stepCondTrue {b tBlk fBlk cond phase outPhase : Nat} {regs outRegs : RegFile}
      {evs : List MemEvent} {tevs : List TmemEvent} {tArgs fArgs : List Nat}
      (hreach : Reachable f env t b regs phase)
      (hexec : execStmts f env t regs phase (f.blockAt b).term (f.blockAt b).stmts = (outRegs, outPhase, evs, tevs))
      (hterm : (f.blockAt b).term = .condBr cond tBlk tArgs fBlk fArgs)
      (hcond : outRegs cond ≠ .bool false)
      (hdst : tBlk < f.blocks.length) :
      Reachable f env t tBlk (bindBlockArgs outRegs outRegs (f.blockAt tBlk).args tArgs) outPhase
  /-- Conditional `mir.cond_br` taking `elseBlk` unless `outRegs cond = .bool true`.
      A `.unk` condition may take either arm. -/
  | stepCondFalse {b tBlk fBlk cond phase outPhase : Nat} {regs outRegs : RegFile}
      {evs : List MemEvent} {tevs : List TmemEvent} {tArgs fArgs : List Nat}
      (hreach : Reachable f env t b regs phase)
      (hexec : execStmts f env t regs phase (f.blockAt b).term (f.blockAt b).stmts = (outRegs, outPhase, evs, tevs))
      (hterm : (f.blockAt b).term = .condBr cond tBlk tArgs fBlk fArgs)
      (hcond : outRegs cond ≠ .bool true)
      (hdst : fBlk < f.blocks.length) :
      Reachable f env t fBlk (bindBlockArgs outRegs outRegs (f.blockAt fBlk).args fArgs) outPhase

/-- Shorthand: `MemEvent` `ev` is emitted during some reachable execution step of `(f, env, t)`. -/
def EmitsEvent (f : MirFunc) (env : InputEnv) (t : ThreadCtx) (ev : MemEvent) : Prop :=
  ∃ b phase : Nat, ∃ regs : RegFile,
    Reachable f env t b regs phase ∧
    ev ∈ (execStmts f env t regs phase (f.blockAt b).term (f.blockAt b).stmts).2.2.1

/-- Shorthand: `TmemEvent` `tev` is emitted during some reachable execution step of `(f, env, t)`. -/
def EmitsTmemEvent (f : MirFunc) (env : InputEnv) (t : ThreadCtx) (tev : TmemEvent) : Prop :=
  ∃ b phase : Nat, ∃ regs : RegFile,
    Reachable f env t b regs phase ∧
    tev ∈ (execStmts f env t regs phase (f.blockAt b).term (f.blockAt b).stmts).2.2.2

/-- True when every `Stmt.assertCond _ cond` in `stmts` evaluates `cond` to `.bool true`
    (meaning no retained runtime `mir.assert` check traps while executing `stmts`). -/
def stmtsNoTrap (f : MirFunc) (env : InputEnv) (t : ThreadCtx)
    (regs : RegFile) (phase : Nat) (term : Terminator) : List Stmt → Bool
  | [] => true
  | stmt :: rest =>
      match stmt with
      | .assertCond _ cond =>
          (regs cond == .bool true) && stmtsNoTrap f env t regs phase term rest
      | .assign dst (.load _) =>
          stmtsNoTrap f env t (regs.set dst .unk) phase term rest
      | .assign dst (.loadTmemBase _ cols) =>
          stmtsNoTrap f env t (regs.set dst (.tmemPtr cols 0)) phase term rest
      | .assign dst rhs =>
          stmtsNoTrap f env t (regs.set dst (evalRhs f env t regs rhs)) phase term rest
      | .syncThreads =>
          stmtsNoTrap f env t regs (phase + 1) term rest
      | _ =>
          stmtsNoTrap f env t regs phase term rest

/-- Inductive small-step reachability of `(blockIdx, entryRegs, entryPhase)` where every
    `Stmt.assertCond` executed along predecessor blocks evaluated to `.bool true`. -/
inductive ReachableTrapFree (f : MirFunc) (env : InputEnv) (t : ThreadCtx) :
    Nat → RegFile → Nat → Prop where
  /-- Kernel starts at `block 0` in phase `0` with `initialRegs f env`. -/
  | entry (hpos : 0 < f.blocks.length) :
      ReachableTrapFree f env t 0 (initialRegs f env) 0
  /-- Trap-free unconditional `mir.goto target args` transition. -/
  | stepGoto {b target phase outPhase : Nat} {regs outRegs : RegFile}
      {evs : List MemEvent} {tevs : List TmemEvent} {args : List Nat}
      (hreach : ReachableTrapFree f env t b regs phase)
      (hnotrap : stmtsNoTrap f env t regs phase (f.blockAt b).term (f.blockAt b).stmts = true)
      (hexec : execStmts f env t regs phase (f.blockAt b).term (f.blockAt b).stmts = (outRegs, outPhase, evs, tevs))
      (hterm : (f.blockAt b).term = .goto target args)
      (hdst : target < f.blocks.length) :
      ReachableTrapFree f env t target (bindBlockArgs outRegs outRegs (f.blockAt target).args args) outPhase
  /-- Trap-free conditional `mir.cond_br` taking `thenBlk` when `outRegs cond = .bool true`. -/
  | stepCondTrue {b tBlk fBlk cond phase outPhase : Nat} {regs outRegs : RegFile}
      {evs : List MemEvent} {tevs : List TmemEvent} {tArgs fArgs : List Nat}
      (hreach : ReachableTrapFree f env t b regs phase)
      (hnotrap : stmtsNoTrap f env t regs phase (f.blockAt b).term (f.blockAt b).stmts = true)
      (hexec : execStmts f env t regs phase (f.blockAt b).term (f.blockAt b).stmts = (outRegs, outPhase, evs, tevs))
      (hterm : (f.blockAt b).term = .condBr cond tBlk tArgs fBlk fArgs)
      (hcond : outRegs cond = .bool true)
      (hdst : tBlk < f.blocks.length) :
      ReachableTrapFree f env t tBlk (bindBlockArgs outRegs outRegs (f.blockAt tBlk).args tArgs) outPhase
  /-- Trap-free conditional `mir.cond_br` taking `elseBlk` when `outRegs cond = .bool false`. -/
  | stepCondFalse {b tBlk fBlk cond phase outPhase : Nat} {regs outRegs : RegFile}
      {evs : List MemEvent} {tevs : List TmemEvent} {tArgs fArgs : List Nat}
      (hreach : ReachableTrapFree f env t b regs phase)
      (hnotrap : stmtsNoTrap f env t regs phase (f.blockAt b).term (f.blockAt b).stmts = true)
      (hexec : execStmts f env t regs phase (f.blockAt b).term (f.blockAt b).stmts = (outRegs, outPhase, evs, tevs))
      (hterm : (f.blockAt b).term = .condBr cond tBlk tArgs fBlk fArgs)
      (hcond : outRegs cond = .bool false)
      (hdst : fBlk < f.blocks.length) :
      ReachableTrapFree f env t fBlk (bindBlockArgs outRegs outRegs (f.blockAt fBlk).args fArgs) outPhase

/-- Trap-free reachability implies general `Reachable`. -/
theorem ReachableTrapFree.toReachable {f : MirFunc} {env : InputEnv} {t : ThreadCtx}
    {b phase : Nat} {regs : RegFile} (h : ReachableTrapFree f env t b regs phase) :
    Reachable f env t b regs phase := by
  induction h with
  | entry hpos => exact Reachable.entry hpos                 -- Base entry block `0`
  | stepGoto _ _ hex hterm hdst ih => exact Reachable.stepGoto ih hex hterm hdst
  | stepCondTrue _ _ hex hterm hc hdst ih =>
      exact Reachable.stepCondTrue ih hex hterm (by rw [hc]; simp) hdst
  | stepCondFalse _ _ hex hterm hc hdst ih =>
      exact Reachable.stepCondFalse ih hex hterm (by rw [hc]; simp) hdst

/-- `MemEvent` `ev` is emitted on a trap-free execution path (every `Stmt.assertCond`
    along predecessor blocks and inside the emitting block evaluates to `.bool true`). -/
def EmitsTrapFreeEvent (f : MirFunc) (env : InputEnv) (t : ThreadCtx) (ev : MemEvent) : Prop :=
  ∃ b phase : Nat, ∃ regs : RegFile,
    ReachableTrapFree f env t b regs phase ∧
    stmtsNoTrap f env t regs phase (f.blockAt b).term (f.blockAt b).stmts = true ∧
    ev ∈ (execStmts f env t regs phase (f.blockAt b).term (f.blockAt b).stmts).2.2.1

/-- Every `EmitsTrapFreeEvent` is also a reachable `EmitsEvent`. -/
theorem EmitsTrapFreeEvent.toEmitsEvent {f : MirFunc} {env : InputEnv} {t : ThreadCtx} {ev : MemEvent} :
    EmitsTrapFreeEvent f env t ev → EmitsEvent f env t ev
  | ⟨b, phase, regs, hr, _, hmem⟩ => ⟨b, phase, regs, hr.toReachable, hmem⟩

/-- True when `rhs` produces a CTA-uniform value given uniform variable predicate `isUni`
    and race-free `SharedArray` pointer predicate `isRfPtr`. -/
def Rhs.isCtaUni (_f : MirFunc) (isUni isRfPtr : Nat → Bool) : Rhs → Bool
  | .copy src | .notBool src | .intCast _ src | .cvtaShared src | .ctaUint1 _ _ src | .ctaUnary src => isUni src
  | .constInt _ _ | .constBool _ | .paramScalar _ | .slicePtr _ | .sliceLen _ | .sharedBase _ | .ctaConst => true
  | .sreg r =>
      match r with
      | .ctaidX | .ctaidY | .ntidX | .ntidY | .clusterCtaidX => true
      | .tidX | .tidY | .laneId => false
  | .binOp _ _ l r | .checkedOvf _ _ l r | .cmpOp _ l r | .ptrOffset l r | .ctaBin l r => isUni l && isUni r
  | .load ptr => isRfPtr ptr && isUni ptr
  | .index1D | .coord2DU32 | .mbarTryWait _ _ _ | .loadTmemBase _ _ | .opaqueUint _ _ | .opaqueVal => false

/-- True when `rhs` produces a pointer into a race-free `SharedArray` (`sid ∈ f.raceFreeShared`). -/
def Rhs.isRfSmemPtr (f : MirFunc) (isRfPtr : Nat → Bool) : Rhs → Bool
  | .sharedBase sid => f.raceFreeShared.contains sid
  | .copy src | .ptrOffset src _ => isRfPtr src
  | _ => false

/-- Step `(ok, curUni, curRf)` sequentially across `stmts` starting from `(globalUni, globalRf)`:
    any assignment to a variable in `globalUni` (resp. `globalRf`) must be locally uniform (resp. `rfPtr`),
    while block-local scratch registers update `(curUni, curRf)` locally within the block. -/
def checkExecUniStmts (f : MirFunc) (globalUni globalRf : List Nat) :
    List Nat → List Nat → List Stmt → Bool × List Nat × List Nat
  | curUni, curRf, [] => (true, curUni, curRf)
  | curUni, curRf, stmt :: rest =>
      match stmt with
      | .assign dst rhs =>
          let uOk := rhs.isCtaUni f curUni.contains curRf.contains
          let rOk := rhs.isRfSmemPtr f curRf.contains
          if (globalUni.contains dst && !uOk) || (globalRf.contains dst && !rOk) then
            (false, [], [])
          else
            let nextUni := if uOk then (if curUni.contains dst then curUni else dst :: curUni) else curUni.filter (· != dst)
            let nextRf := if rOk then (if curRf.contains dst then curRf else dst :: curRf) else curRf.filter (· != dst)
            checkExecUniStmts f globalUni globalRf nextUni nextRf rest
      | _ => checkExecUniStmts f globalUni globalRf curUni curRf rest

/-- Collect every valid non-entry block-argument edge `(dstBlk, paramIdx, argVarId, srcBlk, armIdx, passedVarId)`
    across `f` in a single pass (`armIdx = 0` for `goto` / `condBr` then-arm, `1` for else-arm). -/
def collectAllArgEdges (f : MirFunc) : List (Nat × Nat × Nat × Nat × Nat × Nat) :=
  let dstArgLists := f.blocks.map fun blk => blk.args.map BlockArg.varId
  let pushArgs := fun (dst src arm : Nat) (args : List Nat) (acc : List (Nat × Nat × Nat × Nat × Nat × Nat)) =>
    if dst == 0 then acc
    else
      let dArgs := dstArgLists.getD dst []
      (List.range (min dArgs.length args.length)).foldr (fun k a =>
        match dArgs[k]?, args[k]? with
        | some argId, some v => (dst, k, argId, src, arm, v) :: a
        | _, _ => a) acc
  let rec go : Nat → List BasicBlock → List (Nat × Nat × Nat × Nat × Nat × Nat)
    | _, [] => []
    | p, blk :: rest =>
        let tail := go (p + 1) rest
        match blk.term with
        | .goto dst args => pushArgs dst p 0 args tail
        | .condBr _ tBlk tArgs fBlk fArgs =>
            pushArgs tBlk p 0 tArgs (pushArgs fBlk p 1 fArgs tail)
        | .ret | .unreachable => tail
  go 0 f.blocks

end CudaOxide.Mir.Program
