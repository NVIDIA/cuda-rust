/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide.Proofs.Program.Expr

/-!
# Block-Level Symbolic Execution (`CudaOxide.Proofs.Program.Exec`)
-/
namespace CudaOxide.Proofs.Program
open CudaOxide.Mir.Program
open CudaOxide.Contracts.Program

/-- Dense symbolic register file stored as a `List SymVal` indexed by `varId`. -/
abbrev SymRegList := List SymVal

/-- Read symbolic variable `v` from `sregs` (defaulting to `.unk`). -/
def SymRegList.getVar (sregs : SymRegList) (v : Nat) : SymVal :=
  sregs.getD v .unk

/-- Single-pass structural register write extending with `.unk` if `dst ≥ sregs.length`
    (zero `List.length` scans or dependent `Nat.decLt` proof construction in Lean's kernel `whnf`). -/
def SymRegList.setVar : SymRegList → Nat → SymVal → SymRegList
  | [], 0, sv => [sv]
  | [], d + 1, sv => .unk :: SymRegList.setVar [] d sv
  | _ :: rest, 0, sv => sv :: rest
  | x :: rest, d + 1, sv => x :: SymRegList.setVar rest d sv

/-- Lemma: reading `k` after `sregs.setVar dst sv` behaves identically to `RegFile.set`. -/
theorem getVar_setVar : ∀ (sregs : SymRegList) (dst k : Nat) (sv : SymVal),
    (sregs.setVar dst sv).getVar k = if k == dst then sv else sregs.getVar k
  | [], 0, 0, _ => rfl
  | [], 0, _ + 1, _ => rfl
  | [], d + 1, 0, _ => rfl
  | [], d + 1, k + 1, sv => by
      have ih := getVar_setVar [] d k sv
      have hbeq : ((k + 1) == (d + 1)) = (k == d) := by simp
      simpa [SymRegList.setVar, SymRegList.getVar, List.getD, hbeq] using ih
  | _ :: _, 0, 0, _ => rfl
  | _ :: _, 0, _ + 1, _ => rfl
  | _ :: _, d + 1, 0, _ => rfl
  | x :: rest, d + 1, k + 1, sv => by
      have ih := getVar_setVar rest d k sv
      have hbeq : ((k + 1) == (d + 1)) = (k == d) := by simp
      simpa [SymRegList.setVar, SymRegList.getVar, List.getD, hbeq] using ih

/-- Bind formal `BlockArg`s in parallel from `callerRegs` onto `baseRegs`. -/
def bindSymBlockArgs (callerRegs baseRegs : SymRegList) :
    List BlockArg → List Nat → SymRegList
  | [], _ | _, [] => baseRegs
  | fa :: fas, av :: avs =>
      bindSymBlockArgs callerRegs (baseRegs.setVar fa.varId (callerRegs.getVar av)) fas avs

/-- Exact correspondence between `bindBlockArgs` and `bindSymBlockArgs`. -/
theorem bindSymBlockArgs_correct {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat}
    {callerR : RegFile} {callerSR : SymRegList}
    (hcaller : ∀ v, (callerSR.getVar v).matchesVal env t havoc (callerR v)) :
    ∀ fargs : List BlockArg, ∀ actuals : List Nat,
      ∀ baseR : RegFile, ∀ baseSR : SymRegList,
      (∀ v, (baseSR.getVar v).matchesVal env t havoc (baseR v)) →
      ∀ v, ((bindSymBlockArgs callerSR baseSR fargs actuals).getVar v).matchesVal env t havoc
             ((bindBlockArgs callerR baseR fargs actuals) v)
  | [], _, _, _, hbase, v => hbase v
  | _ :: _, [], _, _, hbase, v => hbase v
  | fa :: fas, av :: avs, baseR, baseSR, hbase, v => by
      dsimp only [bindBlockArgs, bindSymBlockArgs]
      apply bindSymBlockArgs_correct hcaller fas avs
      intro k; unfold RegFile.set; rw [getVar_setVar]
      by_cases hk : (k == fa.varId) = true
      · simp [hk]; exact hcaller av
      · simp [hk]; exact hbase k

mutual
/-- Fuel-threaded node budget check on `SymInt` (`O(budget)` steps, short-circuiting on `0`). -/
def symIntRemNodes : Nat → SymInt → Option Nat
  | 0, _ => none
  | fuel + 1, .binOp _ _ a b =>
      match symIntRemNodes fuel a with
      | none => none
      | some f1 => symIntRemNodes f1 b
  | fuel + 1, .intCast _ a => symIntRemNodes fuel a
  | fuel + 1, .ite _ a b =>
      match symIntRemNodes fuel a with
      | none => none
      | some f1 => symIntRemNodes f1 b
  | fuel + 1, _ => some fuel

/-- Fuel-threaded node budget check on `SymBool` (`O(budget)` steps, short-circuiting on `0`). -/
def symBoolRemNodes : Nat → SymBool → Option Nat
  | 0, _ => none
  | fuel + 1, .not p => symBoolRemNodes fuel p
  | fuel + 1, .mbarReady _ s => symIntRemNodes fuel s
  | fuel + 1, .and p q | fuel + 1, .or p q =>
      match symBoolRemNodes fuel p with
      | none => none
      | some f1 => symBoolRemNodes f1 q
  | fuel + 1, .ite _ p q =>
      match symBoolRemNodes fuel p with
      | none => none
      | some f1 => symBoolRemNodes f1 q
  | fuel + 1, .cmp _ a b =>
      match symIntRemNodes fuel a with
      | none => none
      | some f1 => symIntRemNodes f1 b
  | fuel + 1, .lit _ => some fuel
end

/-- Soundly widen large (`> 36` AST nodes) non-index `SymInt` / `SymBool` expressions to `.unk` in `O(1)`. -/
def widenDeepVal (sv : SymVal) : SymVal :=
  match sv with
  | .int e => if (symIntRemNodes 36 e).isNone then .unk else sv
  | .bool b => if (symBoolRemNodes 48 b).isNone then .unk else sv
  | _ => sv

/-- Soundness of `widenDeepVal`: widening to `.unk` preserves `SymVal.matchesVal`. -/
theorem widenDeepVal_matches {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat}
    {sv : SymVal} {v : Val} (h : sv.matchesVal env t havoc v) :
    (widenDeepVal sv).matchesVal env t havoc v := by
  unfold widenDeepVal
  cases sv <;> simp only <;> try exact h
  all_goals (split <;> first | trivial | exact h)

/-- Raw 1-step symbolic evaluator for `Rhs` before `widenDeepVal`. -/
def symEvalRhsRaw (f : MirFunc) (sregs : SymRegList) : Rhs → SymVal
  | .copy src => sregs.getVar src
  | .constInt w val => .int (.const (wordMod w val))
  | .constBool b => .bool (.lit b)
  | .sreg .tidX => .int .tidX
  | .sreg .tidY => .int .tidY
  | .sreg .ctaidX => .int .ctaidX
  | .sreg .ctaidY => .int .ctaidY
  | .sreg .ntidX => .int (.const f.blockDimX)
  | .sreg .ntidY => .int (.const f.blockDimY)
  | .sreg .laneId => .int (.binOp .rem 32 .tidX (.const 32))
  | .sreg .clusterCtaidX => .int (.binOp .rem 0 .ctaidX (.const (max 1 f.clusterDimX)))
  | .index1D => .int (.index1D f.blockDimX)
  | .coord2DU32 =>
      let col := .intCast 64 (.binOp .add 32 (.binOp .mul 32 .ctaidX (.const f.blockDimX)) .tidX)
      let row := .intCast 64 (.binOp .add 32 (.binOp .mul 32 .ctaidY (.const f.blockDimY)) .tidY)
      .int (.binOp .add 64 (.binOp .shl 64 row (.const 32)) col)
  | .paramScalar p => .int (.paramScalar p)
  | .slicePtr p => .ptr .global p (.const 0)
  | .sliceLen p => .int (.sliceLen p)
  | .sharedBase s => .ptr .shared s (.const 0)
  | .binOp op w l r =>
      match sregs.getVar l, sregs.getVar r with
      | .int a, .int b => .int (.binOp op w a b)
      | .smemOff sid base, .int d =>
          if op == .add then .smemOff sid (.binOp .add 0 base d)
          else if op == .shr && d == .const 4 then .smemDesc sid base
          else .unk
      | .int d, .smemOff sid base =>
          if op == .add then .smemOff sid (.binOp .add 0 d base) else .unk
      | .smemDesc sid base, .int d =>
          if op == .add then .smemDesc sid (.binOp .add 0 base (.binOp .mul 0 d (.const 16)))
          else if op == .bitAnd || op == .bitOr then .smemDesc sid base
          else .unk
      | .int d, .smemDesc sid base =>
          if op == .add then .smemDesc sid (.binOp .add 0 (.binOp .mul 0 d (.const 16)) base)
          else if op == .bitAnd || op == .bitOr then .smemDesc sid base
          else .unk
      | .tmemPtr cols rel, .int d =>
          if op == .add then .tmemPtr cols (.binOp .add w rel d) else .unk
      | .int d, .tmemPtr cols rel =>
          if op == .add then .tmemPtr cols (.binOp .add w d rel) else .unk
      | _, _ => .unk
  | .cmpOp op l r =>
      match sregs.getVar l, sregs.getVar r with
      | .int a, .int b => .bool (.cmp op a b)
      | _, _ => .unk
  | .notBool src =>
      match sregs.getVar src with
      | .bool b => .bool (.not b)
      | _ => .unk
  | .intCast dstW src =>
      match sregs.getVar src with
      | .int a => .int (.intCast dstW a)
      | .smemOff sid b => if decide (32 ≤ dstW) then .smemOff sid b else .unk
      | _ => .unk
  | .ptrOffset base idx =>
      match sregs.getVar base, sregs.getVar idx with
      | .ptr sp bid off, .int delta => .ptr sp bid (.binOp .add 0 off delta)
      | _, _ => .unk
  | .load _ | .loadTmemBase _ _ => .unk
  | .mbarTryWait barId slotVar =>
      match sregs.getVar slotVar with
      | .int s => .bool (.mbarReady barId s)
      | _ => .unk
  | .cvtaShared ptr =>
      match sregs.getVar ptr with
      | .ptr .shared sid off =>
          let eb := max 1 (f.sharedDecls.getD sid { size := 0 }).elemBytes
          .smemOff sid (.binOp .mul 0 off (.const eb))
      | _ => .unk
  | .opaqueUint id w => .int (.opaqueUint id w)
  | .opaqueVal => .unk

/-- Symbolically evaluate one `Rhs` expression with depth-10 widening on deep arithmetic chains. -/
def symEvalRhs (f : MirFunc) (sregs : SymRegList) (rhs : Rhs) : SymVal :=
  widenDeepVal (symEvalRhsRaw f sregs rhs)

/-- Soundness of `symEvalRhs`: evaluating `rhs` symbolically matches `evalRhs`. -/
theorem symEvalRhs_correct (f : MirFunc) (env : InputEnv) (t : ThreadCtx)
    (havoc : Nat → Nat) (regs : RegFile) (sregs : SymRegList)
    (hmatch : ∀ v, (sregs.getVar v).matchesVal env t havoc (regs v)) (rhs : Rhs) :
    (symEvalRhs f sregs rhs).matchesVal env t havoc (evalRhs f env t regs rhs) := by
  apply widenDeepVal_matches
  cases rhs with
  | copy src => exact hmatch src
  | constInt _ _ | constBool _ | index1D | coord2DU32 | paramScalar _ | slicePtr _
  | sliceLen _ | sharedBase _ | load _ | loadTmemBase _ _
  | opaqueUint _ _ | opaqueVal =>
      simp [symEvalRhsRaw, evalRhs, SymVal.matchesVal, SymInt.eval, SymBool.eval]
  | mbarTryWait barId slotVar =>
      have hs := hmatch slotVar
      dsimp only [symEvalRhsRaw]
      cases hss : sregs.getVar slotVar <;> simp [hss, SymVal.matchesVal] at hs ⊢
      simp [evalRhs, hs, SymBool.eval]
  | sreg r =>
      cases r <;> simp [symEvalRhsRaw, evalRhs, SymVal.matchesVal, SymInt.eval, evalBinOp]
  | cvtaShared ptr =>
      have hp := hmatch ptr
      dsimp only [symEvalRhsRaw]
      cases hsp : sregs.getVar ptr with
      | ptr sp bid off =>
          cases sp <;> simp [hsp, SymVal.matchesVal] at hp ⊢
          simp [evalRhs, hp, SymInt.eval, evalBinOp, wordMod]
      | _ => simp [SymVal.matchesVal]
  | binOp op w l r =>
      have hl := hmatch l; have hr := hmatch r
      dsimp only [symEvalRhsRaw]
      cases hsl : sregs.getVar l <;> cases hsr : sregs.getVar r <;>
        simp [hsl, hsr, SymVal.matchesVal] at hl hr ⊢
      · simp [evalRhs, hl, hr, SymInt.eval]
      · cases op <;> simp [evalRhs, hl, hr, SymInt.eval, evalBinOp, wordMod]
      · cases op <;> simp [evalRhs, hl, hr, SymInt.eval, evalBinOp, wordMod]
      · cases op <;> simp [evalRhs, hl, hr, SymInt.eval, evalBinOp]
      · rename_i sid base d
        cases op <;> try simp [evalRhs, hl, hr, SymInt.eval, evalBinOp, wordMod]
        by_cases hd4 : d = .const 4
        · subst hd4; simp [SymInt.eval]
        · simp [hd4]
      · cases op <;> simp [evalRhs, hl, hr, SymInt.eval, evalBinOp, wordMod]
      · cases op <;> simp [evalRhs, hl, hr, SymInt.eval, evalBinOp]
  | cmpOp op l r =>
      have hl := hmatch l; have hr := hmatch r
      dsimp only [symEvalRhsRaw]
      cases hsl : sregs.getVar l <;> cases hsr : sregs.getVar r <;>
        simp [hsl, hsr, SymVal.matchesVal] at hl hr ⊢
      simp [evalRhs, hl, hr, SymBool.eval]
  | notBool src =>
      have hs := hmatch src
      dsimp only [symEvalRhsRaw]
      cases hss : sregs.getVar src <;> simp [hss, SymVal.matchesVal] at hs ⊢
      simp [evalRhs, hs, SymBool.eval]
  | intCast dw src =>
      have hs := hmatch src
      dsimp only [symEvalRhsRaw]
      cases hss : sregs.getVar src <;> simp [hss, SymVal.matchesVal] at hs ⊢
      · simp [evalRhs, hs, SymInt.eval]
      · by_cases h32 : 32 ≤ dw
        · simp [h32, evalRhs, hs]
        · simp [h32]
  | ptrOffset base idx =>
      have hb := hmatch base; have hi := hmatch idx
      dsimp only [symEvalRhsRaw]
      cases hsb : sregs.getVar base <;> cases hsi : sregs.getVar idx <;>
        simp [hsb, hsi, SymVal.matchesVal] at hb hi ⊢
      simp [evalRhs, hb, hi, SymInt.eval, evalBinOp, wordMod]

/-- Symbolic memory event recorded during `symExecStmts`. -/
structure SymMemEvent where
  /-- `true` for store / TMA / `tmemAlloc`, `false` for load / `smemDescRead`. -/
  isWrite : Bool
  /-- Target address space (`.global` or `.shared`). -/
  space : AddrSpace
  /-- Buffer / parameter ID. -/
  bufId : Nat
  /-- Symbolic base offset (`none` if pointer/descriptor provenance was unknown). -/
  indexOpt : Option SymInt
  /-- Number of elements (`elemDiv = 1`) or bytes (`elemDiv = elemBytes`) accessed (`≥ 1`). -/
  span : Nat := 1
  /-- Divisor converting raw offset units into element indices (`1` for element pointers, `elemBytes` for byte descriptors). -/
  elemDiv : Nat := 1
  /-- `some (barId, slot)` when released by an `mbarrier` arrival or TMA transfer on `BAR[slot]`. -/
  syncBar : Option (Nat × SymInt) := none
  /-- Path condition active at this access site. -/
  pathCond : SymBool
  /-- Symbolic barrier phase active at this access site. -/
  phaseSpec : PhaseSpec
  deriving Repr, DecidableEq, Lean.ToExpr

/-- Evaluate an optional symbolic `(barId, slotSym)` handle to `Option (Nat × Nat)`. -/
def evalSyncBar (env : InputEnv) (t : ThreadCtx) (havoc : Nat → Nat) :
    Option (Nat × SymInt) → Option (Nat × Nat)
  | none => none
  | some (b, s) => some (b, s.eval env t havoc)

/-- Symbolic assertion event recorded during `symExecStmts`. -/
structure SymAssertEvent where
  /-- Deterministic `assertId` of the `Stmt.assertCond` instruction. -/
  assertId : Nat
  /-- Active path condition at the assertion site. -/
  pathCond : SymBool
  /-- Symbolic boolean condition `cond` tested by `Stmt.assertCond`. -/
  condOpt : Option SymBool
  deriving Repr, DecidableEq, Lean.ToExpr

/-- Symbolic Blackwell Tensor Memory (`TMEM`) event recorded during `symExecStmts`. -/
structure SymTmemEvent where
  /-- Dynamically allocated `TMEM` column count (`tcgen05_alloc`). -/
  allocCols : Nat
  /-- Symbolic 32-bit `(lane << 16) + col` address (`none` if provenance was unknown). -/
  relAddrOpt : Option SymInt
  /-- Number of `TMEM` lanes accessed. -/
  heightLanes : Nat
  /-- Number of 32-bit `TMEM` columns accessed. -/
  widthCols : Nat
  /-- True for warp-restricted `tcgen05.ld` / `tcgen05.st`. -/
  warpRestricted : Bool
  /-- Active path condition at the `TMEM` operation site. -/
  pathCond : SymBool
  deriving Repr, DecidableEq, Lean.ToExpr

/-- Resolve a pointer `SymVal` under `pCond` into a 1-element `SymMemEvent` (or `indexOpt := none` on failure). -/
def symSinglePtrEvent (pCond : SymBool) (ps : PhaseSpec) (isWrite : Bool) (sv : SymVal) : SymMemEvent :=
  match sv with
  | .ptr sp bid off =>
      { isWrite := isWrite, space := sp, bufId := bid, indexOpt := some off, pathCond := pCond, phaseSpec := ps }
  | .condPtr g sp bid off =>
      if symImpliesVal pCond g true then
        { isWrite := isWrite, space := sp, bufId := bid, indexOpt := some off, pathCond := pCond, phaseSpec := ps }
      else
        { isWrite := isWrite, space := .global, bufId := 0, indexOpt := none, pathCond := pCond, phaseSpec := ps }
  | _ =>
      { isWrite := isWrite, space := .global, bufId := 0, indexOpt := none, pathCond := pCond, phaseSpec := ps }

/-- Execute a block's statements symbolically from `(pCond, sregs, ps)`.
    - **What it does**: Steps through a basic block's `List Stmt`, updating `sregs` and `ps`
      and recording every global/shared memory event (`SymMemEvent`), assertion (`SymAssertEvent`),
      and Blackwell `TMEM` event (`SymTmemEvent`).
    - **Fail-closed invariant**: Any memory or `TMEM` instruction on an unresolved handle emits
      `indexOpt := none` / `relAddrOpt := none`, causing `validSymFunc` to return `false`. -/
def symExecStmts (f : MirFunc) (pCond : SymBool) (sregs : SymRegList) (ps : PhaseSpec) (term : Terminator) :
    List Stmt → SymRegList × PhaseSpec × List SymMemEvent × List SymAssertEvent × List SymTmemEvent
  | [] => (sregs, ps, [], [], [])
  | stmt :: rest =>
      match stmt with
      | .assign dst (.load ptr) =>
          let mev := [symSinglePtrEvent pCond ps false (sregs.getVar ptr)]
          let (r', ps', mevs, aevs, tevs) := symExecStmts f pCond (sregs.setVar dst .unk) ps term rest
          (r', ps', mev ++ mevs, aevs, tevs)
      | .assign dst (.loadTmemBase ptr cols) =>
          let mev := [symSinglePtrEvent pCond ps false (sregs.getVar ptr)]
          let (r', ps', mevs, aevs, tevs) := symExecStmts f pCond (sregs.setVar dst (.tmemPtr cols (.const 0))) ps term rest
          (r', ps', mev ++ mevs, aevs, tevs)
      | .assign dst rhs =>
          let sv := symEvalRhs f sregs rhs
          symExecStmts f pCond (sregs.setVar dst sv) ps term rest
      | .store ptr _ =>
          let baseEv := symSinglePtrEvent pCond ps true (sregs.getVar ptr)
          let mev := match findTrailArrive f 8 term rest with
            | none => [baseEv]
            | some (b, sv) =>
                match sregs.getVar sv with
                | .int sSym =>
                    [{ baseEv with syncBar := if baseEv.space == .shared then some (b, sSym) else none }]
                | _ =>
                    [{ baseEv with indexOpt := none }]
          let (r', ps', mevs, aevs, tevs) := symExecStmts f pCond sregs ps term rest
          (r', ps', mev ++ mevs, aevs, tevs)
      | .barrierSlotCheck ptr =>
          let mev := [symSinglePtrEvent pCond ps false (sregs.getVar ptr)]
          let (r', ps', mevs, aevs, tevs) := symExecStmts f pCond sregs ps term rest
          (r', ps', mev ++ mevs, aevs, tevs)
      | .assertCond aid cond =>
          let cOpt := match sregs.getVar cond with | .bool b => some b | _ => none
          let aev : SymAssertEvent := { assertId := aid, pathCond := pCond, condOpt := cOpt }
          let (r', ps', mevs, aevs, tevs) := symExecStmts f pCond sregs ps term rest
          (r', ps', mevs, aev :: aevs, tevs)
      | .clusterRankCheck _ | .mbarArrive _ _ | .mbarArriveExpectTx _ _ _ =>
          symExecStmts f pCond sregs ps term rest
      | .syncThreads =>
          symExecStmts f pCond sregs ps.stepSync term rest
      | .tmaCopyG2S ptr barId slotVar spanElems =>
          let span := max 1 spanElems
          let mev := match sregs.getVar ptr, sregs.getVar slotVar with
            | .ptr sp bid off, .int sSym =>
                [{ isWrite := true, space := sp, bufId := bid, indexOpt := some off,
                   span := span, elemDiv := 1, syncBar := some (barId, sSym), pathCond := pCond, phaseSpec := ps }]
            | _, _ =>
                [{ isWrite := true, space := .shared, bufId := 0, indexOpt := none,
                   span := span, elemDiv := 1, pathCond := pCond, phaseSpec := ps }]
          let (r', ps', mevs, aevs, tevs) := symExecStmts f pCond sregs ps term rest
          (r', ps', mev ++ mevs, aevs, tevs)
      | .tmemAlloc ptr _cols =>
          let lane0Cond : SymBool :=
            .and pCond (.cmp .eq (.binOp .rem 0 .tidX (.const 32)) (.const 0))
          let mev := [symSinglePtrEvent lane0Cond ps true (sregs.getVar ptr)]
          let (r', ps', mevs, aevs, tevs) := symExecStmts f pCond sregs ps term rest
          (r', ps', mev ++ mevs, aevs, tevs)
      | .smemDescRead descVar tileBytes =>
          let span := max 1 tileBytes
          let mev := match sregs.getVar descVar with
            | .smemDesc sid byteOff =>
                let eb := max 1 (f.sharedDecls.getD sid { size := 0 }).elemBytes
                [{ isWrite := false, space := .shared, bufId := sid, indexOpt := some byteOff,
                   span := span, elemDiv := eb, pathCond := pCond, phaseSpec := ps }]
            | _ =>
                [{ isWrite := false, space := .shared, bufId := 0, indexOpt := none,
                   span := span, elemDiv := 1, pathCond := pCond, phaseSpec := ps }]
          let (r', ps', mevs, aevs, tevs) := symExecStmts f pCond sregs ps term rest
          (r', ps', mev ++ mevs, aevs, tevs)
      | .tmemAccess tmemVar hl wc wr =>
          let tev := match sregs.getVar tmemVar with
            | .tmemPtr cols rel =>
                [{ allocCols := cols, relAddrOpt := some rel, heightLanes := hl,
                   widthCols := wc, warpRestricted := wr, pathCond := pCond }]
            | _ =>
                [{ allocCols := 0, relAddrOpt := none, heightLanes := hl,
                   widthCols := wc, warpRestricted := wr, pathCond := pCond }]
          let (r', ps', mevs, aevs, tevs) := symExecStmts f pCond sregs ps term rest
          (r', ps', mevs, aevs, tev ++ tevs)

/-- Helper lemma discharging single-pointer `symSinglePtrEvent` correspondence in `symExecStmts_correct`. -/
theorem symSinglePtrEvent_sound {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat}
    {pCond : SymBool} (hpc : pCond.eval env t havoc = true)
    {ps : PhaseSpec} {phase : Nat} (hps : ps.matchesPhase phase = true)
    {isW : Bool} {sv : SymVal} {v : Val} {ev : MemEvent}
    (hmatch : sv.matchesVal env t havoc v)
    (hsome : (symSinglePtrEvent pCond ps isW sv).indexOpt.isSome = true)
    (hmem : ev ∈ (match v with | .ptr sp bid off => [{ isWrite := isW, space := sp, bufId := bid, index := off, phase := phase : MemEvent }] | _ => [])) :
    ∃ idxSym : SymInt, ∃ k : Nat,
      let sev := symSinglePtrEvent pCond ps isW sv
      k < sev.span ∧ 0 < sev.elemDiv ∧
      sev.isWrite = ev.isWrite ∧ sev.space = ev.space ∧ sev.bufId = ev.bufId ∧
      evalSyncBar env t havoc sev.syncBar = ev.syncBar ∧ sev.indexOpt = some idxSym ∧
      (idxSym.eval env t havoc + k) / sev.elemDiv = ev.index ∧
      sev.pathCond.eval env t havoc = true ∧ sev.phaseSpec.matchesPhase ev.phase = true := by
  cases sv with
  | ptr sp bid off =>
      dsimp only [symSinglePtrEvent]
      rw [SymVal.matchesVal] at hmatch; subst hmatch
      simp only [List.mem_singleton] at hmem; subst hmem
      exact ⟨off, 0, Nat.zero_lt_one, Nat.zero_lt_one, rfl, rfl, rfl, rfl, rfl, by simp, hpc, hps⟩
  | condPtr g sp bid off =>
      dsimp only [symSinglePtrEvent] at hsome ⊢
      by_cases himp : symImpliesVal pCond g true = true
      · rw [if_pos himp] at hsome ⊢
        rw [SymVal.matchesVal] at hmatch
        rw [hmatch (symImpliesVal_sound hpc g true himp)] at hmem
        simp only [List.mem_singleton] at hmem; subst hmem
        exact ⟨off, 0, Nat.zero_lt_one, Nat.zero_lt_one, rfl, rfl, rfl, rfl, rfl, by simp, hpc, hps⟩
      · rw [if_neg himp] at hsome; simp at hsome
  | int _ | bool _ | smemOff _ _ | smemDesc _ _ | tmemPtr _ _ | unk =>
      dsimp only [symSinglePtrEvent] at hsome; simp at hsome

/-- Exact correspondence between `symExecStmts` and concrete `execStmts` when `pCond` holds and all
    symbolic memory and `TMEM` events in the block carry `isSome` offsets. -/
theorem symExecStmts_correct (f : MirFunc) (env : InputEnv) (t : ThreadCtx)
    (havoc : Nat → Nat) (pCond : SymBool) (hpc : pCond.eval env t havoc = true) (term : Terminator) :
    ∀ stmts : List Stmt, ∀ regs : RegFile, ∀ sregs : SymRegList,
      ∀ phase : Nat, ∀ ps : PhaseSpec,
      (∀ v, (sregs.getVar v).matchesVal env t havoc (regs v)) →
      ps.matchesPhase phase = true →
      ((symExecStmts f pCond sregs ps term stmts).2.2.1.all (fun m => m.indexOpt.isSome)) = true →
      ((symExecStmts f pCond sregs ps term stmts).2.2.2.2.all (fun tm => tm.relAddrOpt.isSome)) = true →
      (∀ v, ((symExecStmts f pCond sregs ps term stmts).1.getVar v).matchesVal env t havoc
              ((execStmts f env t regs phase term stmts).1 v)) ∧
      (symExecStmts f pCond sregs ps term stmts).2.1.matchesPhase
        (execStmts f env t regs phase term stmts).2.1 = true ∧
      (∀ ev ∈ (execStmts f env t regs phase term stmts).2.2.1,
        ∃ sev ∈ (symExecStmts f pCond sregs ps term stmts).2.2.1, ∃ idxSym : SymInt, ∃ k : Nat,
          k < sev.span ∧ 0 < sev.elemDiv ∧
          sev.isWrite = ev.isWrite ∧ sev.space = ev.space ∧ sev.bufId = ev.bufId ∧
          evalSyncBar env t havoc sev.syncBar = ev.syncBar ∧ sev.indexOpt = some idxSym ∧
          (idxSym.eval env t havoc + k) / sev.elemDiv = ev.index ∧
          sev.pathCond.eval env t havoc = true ∧ sev.phaseSpec.matchesPhase ev.phase = true) ∧
      (∀ pfx sfx : List Stmt, ∀ aid cond : Nat,
        stmts = pfx ++ (.assertCond aid cond :: sfx) →
        ∃ aev ∈ (symExecStmts f pCond sregs ps term stmts).2.2.2.1,
          aev.assertId = aid ∧ aev.pathCond = pCond ∧
          ∀ bSym : SymBool, aev.condOpt = some bSym →
            (execStmts f env t regs phase term pfx).1 cond = .bool (bSym.eval env t havoc)) ∧
      (∀ tev ∈ (execStmts f env t regs phase term stmts).2.2.2,
        ∃ stev ∈ (symExecStmts f pCond sregs ps term stmts).2.2.2.2, ∃ relSym : SymInt,
          stev.allocCols = tev.allocCols ∧ stev.heightLanes = tev.heightLanes ∧
          stev.widthCols = tev.widthCols ∧ stev.warpRestricted = tev.warpRestricted ∧
          stev.relAddrOpt = some relSym ∧ relSym.eval env t havoc = tev.relAddr ∧
          stev.pathCond = pCond) := by
  intro stmts
  induction stmts with                                    -- Structural induction on `stmts`
  | nil =>                                                -- Base case (`[]`): registers & phase match immediately
      intro regs sregs phase ps hmatch hps _ _
      refine ⟨hmatch, hps, ?_, ?_, ?_⟩
      · intro _ hmem; cases hmem
      · intro pfx _ _ _ heq; cases pfx <;> contradiction
      · intro _ hmem; cases hmem
  | cons stmt rest ih =>                                  -- Inductive step (`stmt :: rest`): case-split by MIR statement
      intro regs sregs phase ps hmatch hps hallSome hallTmem
      cases stmt with
      | assign dst rhs =>                                 -- Case 1: `dst = rhs` register assignment
          have hnextGen : ∀ v, ((sregs.setVar dst (symEvalRhs f sregs rhs)).getVar v).matchesVal env t havoc
              ((regs.set dst (evalRhs f env t regs rhs)) v) := by
            intro v; unfold RegFile.set; rw [getVar_setVar]
            by_cases hv : (v == dst) = true
            · simp [hv]; exact symEvalRhs_correct f env t havoc regs sregs hmatch rhs
            · simp [hv]; exact hmatch v
          cases rhs with
          | load ptr =>                                   --   Subcase 1a: `mir.load(ptr)` emits 1 read `SymMemEvent` + sets `dst := .unk`
              simp only [symExecStmts, List.all_append, List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true] at hallSome hallTmem
              have hnext : ∀ v, ((sregs.setVar dst .unk).getVar v).matchesVal env t havoc
                  ((regs.set dst .unk) v) := by
                intro v; unfold RegFile.set; rw [getVar_setVar]
                by_cases hv : (v == dst) = true
                · simp [hv, SymVal.matchesVal]
                · simp [hv]; exact hmatch v
              obtain ⟨hmR, hmP, hmE, hmA, hmT⟩ := ih (regs.set dst .unk) (sregs.setVar dst .unk) phase ps hnext hps hallSome.2 hallTmem
              simp only [execStmts, symExecStmts]
              refine ⟨hmR, hmP, ?_, ?_, hmT⟩
              · intro ev hev
                rw [List.mem_append] at hev
                rcases hev with hhead | htail
                · obtain ⟨idxS, k, hrest⟩ := symSinglePtrEvent_sound hpc hps (hmatch ptr) hallSome.1 hhead
                  exact ⟨_, List.mem_append_left _ (List.Mem.head _), idxS, k, hrest⟩
                · obtain ⟨sev, hsev, idxS, k, hrest⟩ := hmE ev htail
                  exact ⟨sev, List.mem_append_right _ hsev, idxS, k, hrest⟩
              · intro pfx sfx aid cond heq
                cases pfx with
                | nil => simp at heq
                | cons hd tl =>
                    simp only [List.cons_append, List.cons.injEq] at heq
                    rcases heq with ⟨rfl, htl⟩
                    simpa [execStmts] using hmA tl sfx aid cond htl
          | loadTmemBase ptr cols =>                      --   Subcase 1b: `TMEM_ADDR` load emits 1 shared read + sets `.tmemPtr cols 0`
              simp only [symExecStmts, List.all_append, List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true] at hallSome hallTmem
              have hnext : ∀ v, ((sregs.setVar dst (.tmemPtr cols (.const 0))).getVar v).matchesVal env t havoc
                  ((regs.set dst (.tmemPtr cols 0)) v) := by
                intro v; unfold RegFile.set; rw [getVar_setVar]
                by_cases hv : (v == dst) = true
                · simp [hv, SymVal.matchesVal, SymInt.eval]
                · simp [hv]; exact hmatch v
              obtain ⟨hmR, hmP, hmE, hmA, hmT⟩ :=
                ih (regs.set dst (.tmemPtr cols 0)) (sregs.setVar dst (.tmemPtr cols (.const 0))) phase ps hnext hps hallSome.2 hallTmem
              simp only [execStmts, symExecStmts]
              refine ⟨hmR, hmP, ?_, ?_, hmT⟩
              · intro ev hev
                rw [List.mem_append] at hev
                rcases hev with hhead | htail
                · obtain ⟨idxS, k, hrest⟩ := symSinglePtrEvent_sound hpc hps (hmatch ptr) hallSome.1 hhead
                  exact ⟨_, List.mem_append_left _ (List.Mem.head _), idxS, k, hrest⟩
                · obtain ⟨sev, hsev, idxS, k, hrest⟩ := hmE ev htail
                  exact ⟨sev, List.mem_append_right _ hsev, idxS, k, hrest⟩
              · intro pfx sfx aid cond heq
                cases pfx with
                | nil => simp at heq
                | cons hd tl =>
                    simp only [List.cons_append, List.cons.injEq] at heq
                    rcases heq with ⟨rfl, htl⟩
                    simpa [execStmts] using hmA tl sfx aid cond htl
          | _ =>                                          --   Subcase 1c: Pure arithmetic/pointer `Rhs` via `symEvalRhs_correct`
              simp only [execStmts, symExecStmts] at hallSome hallTmem ⊢
              obtain ⟨hmR, hmP, hmE, hmA, hmT⟩ := ih _ _ phase ps hnextGen hps hallSome hallTmem
              refine ⟨hmR, hmP, hmE, ?_, hmT⟩
              intro pfx sfx aid cond heq
              cases pfx with
              | nil => simp at heq
              | cons hd tl =>
                  simp only [List.cons_append, List.cons.injEq] at heq
                  rcases heq with ⟨rfl, htl⟩
                  simpa [execStmts] using hmA tl sfx aid cond htl
      | store ptr _ =>                                    -- Case 2: `mir.store(ptr, val)` emits 1 write `SymMemEvent` (+ optional `syncBar`)
          simp only [symExecStmts, List.all_append, Bool.and_eq_true] at hallSome hallTmem
          obtain ⟨hmR, hmP, hmE, hmA, hmT⟩ := ih regs sregs phase ps hmatch hps hallSome.2 hallTmem
          simp only [execStmts, symExecStmts]
          refine ⟨hmR, hmP, ?_, ?_, hmT⟩
          · intro ev hev
            rw [List.mem_append] at hev
            rcases hev with hhead | htail
            · cases hfba : findTrailArrive f 8 term rest with
              | none =>
                  rw [hfba] at hallSome hhead
                  simp only [ite_self] at hhead
                  simp only [List.all_cons, List.all_nil, Bool.and_true] at hallSome
                  obtain ⟨idxS, k, hrest⟩ := symSinglePtrEvent_sound hpc hps (hmatch ptr) hallSome.1 hhead
                  exact ⟨_, List.mem_append_left _ (List.Mem.head _), idxS, k, hrest⟩
              | some bsv =>
                  rcases bsv with ⟨b, sv⟩
                  rw [hfba] at hallSome hhead
                  dsimp only at hallSome hhead
                  have hsvM := hmatch sv
                  cases hssv : sregs.getVar sv with
                  | int sSym =>
                      dsimp only
                      rw [hssv] at hallSome ⊢
                      rw [hssv, SymVal.matchesVal] at hsvM
                      rw [hsvM] at hhead
                      simp only [List.all_cons, List.all_nil, Bool.and_true] at hallSome
                      have hp := hmatch ptr
                      cases hsp : sregs.getVar ptr with
                      | ptr sp bid off =>
                          dsimp only [symSinglePtrEvent] at hallSome ⊢
                          rw [hsp, SymVal.matchesVal] at hp; rw [hp] at hhead
                          simp only [List.mem_singleton] at hhead; subst hhead
                          refine ⟨_, List.mem_append_left _ (List.Mem.head _),
                                  off, 0, Nat.zero_lt_one, Nat.zero_lt_one, rfl, rfl, rfl, ?_, rfl, by simp, hpc, hps⟩
                          dsimp only [evalSyncBar]
                          cases sp <;> rfl
                      | condPtr g sp bid off =>
                          rw [hsp] at hallSome
                          by_cases himp : symImpliesVal pCond g true = true
                          · simp only [symSinglePtrEvent, himp, ↓reduceIte] at hallSome ⊢
                            rw [hsp, SymVal.matchesVal] at hp
                            rw [hp (symImpliesVal_sound hpc g true himp)] at hhead
                            simp only [List.mem_singleton] at hhead; subst hhead
                            refine ⟨_, List.mem_append_left _ (List.Mem.head _),
                                    off, 0, Nat.zero_lt_one, Nat.zero_lt_one, rfl, rfl, rfl, ?_, rfl, by simp, hpc, hps⟩
                            dsimp only [evalSyncBar]
                            cases sp <;> rfl
                          · simp [symSinglePtrEvent, himp] at hallSome
                      | _ =>
                          rw [hsp] at hallSome; dsimp only [symSinglePtrEvent] at hallSome; simp at hallSome
                  | _ => rw [hssv] at hallSome; simp at hallSome
            · obtain ⟨sev, hsev, idxS, k, hrest⟩ := hmE ev htail
              exact ⟨sev, List.mem_append_right _ hsev, idxS, k, hrest⟩
          · intro pfx sfx aid cond heq
            cases pfx with
            | nil => simp at heq
            | cons hd tl =>
                simp only [List.cons_append, List.cons.injEq] at heq
                rcases heq with ⟨rfl, htl⟩
                simpa [execStmts] using hmA tl sfx aid cond htl
      | barrierSlotCheck ptr =>                           -- Case 3: `SharedArray<Barrier, N>` slot bounds check
          simp only [symExecStmts, List.all_append, List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true] at hallSome hallTmem
          obtain ⟨hmR, hmP, hmE, hmA, hmT⟩ := ih regs sregs phase ps hmatch hps hallSome.2 hallTmem
          simp only [execStmts, symExecStmts]
          refine ⟨hmR, hmP, ?_, ?_, hmT⟩
          · intro ev hev
            rw [List.mem_append] at hev
            rcases hev with hhead | htail
            · obtain ⟨idxS, k, hrest⟩ := symSinglePtrEvent_sound hpc hps (hmatch ptr) hallSome.1 hhead
              exact ⟨_, List.mem_append_left _ (List.Mem.head _), idxS, k, hrest⟩
            · obtain ⟨sev, hsev, idxS, k, hrest⟩ := hmE ev htail
              exact ⟨sev, List.mem_append_right _ hsev, idxS, k, hrest⟩
          · intro pfx sfx aid cond heq
            cases pfx with
            | nil => simp at heq
            | cons hd tl =>
                simp only [List.cons_append, List.cons.injEq] at heq
                rcases heq with ⟨rfl, htl⟩
                simpa [execStmts] using hmA tl sfx aid cond htl
      | assertCond aid0 cond0 =>                          -- Case 4: `mir.assert` records `SymAssertEvent` for `MirAssertRedundant`
          simp only [symExecStmts] at hallSome hallTmem
          obtain ⟨hmR, hmP, hmE, hmA, hmT⟩ := ih regs sregs phase ps hmatch hps hallSome hallTmem
          simp only [execStmts, symExecStmts]
          refine ⟨hmR, hmP, hmE, ?_, hmT⟩
          intro pfx sfx aid cond heq
          cases pfx with
          | nil =>
              simp only [List.nil_append, List.cons.injEq, Stmt.assertCond.injEq] at heq
              rcases heq with ⟨⟨ha, hc⟩, _⟩
              subst ha hc
              refine ⟨_, List.Mem.head _, rfl, rfl, ?_⟩
              intro bSym hcOpt
              have hm := hmatch cond0
              cases hsv : sregs.getVar cond0 with
              | bool b0 =>
                  rw [hsv] at hcOpt hm
                  dsimp only at hcOpt
                  injection hcOpt with hb; subst hb
                  simpa [execStmts, SymVal.matchesVal] using hm
              | _ =>
                  rw [hsv] at hcOpt; contradiction
          | cons hd tl =>
              simp only [List.cons_append, List.cons.injEq] at heq
              rcases heq with ⟨rfl, htl⟩
              obtain ⟨aev, haev, ha, hp, hc⟩ := hmA tl sfx aid cond htl
              exact ⟨aev, List.Mem.tail _ haev, ha, hp, by simpa [execStmts] using hc⟩
      | clusterRankCheck _ =>                             -- Case 5: `map_shared_rank` peer-rank check (no register/phase change)
          simp only [execStmts, symExecStmts] at hallSome hallTmem ⊢
          obtain ⟨hmR, hmP, hmE, hmA, hmT⟩ := ih regs sregs phase ps hmatch hps hallSome hallTmem
          refine ⟨hmR, hmP, hmE, ?_, hmT⟩
          intro pfx sfx aid cond heq
          cases pfx with
          | nil => simp at heq
          | cons hd tl =>
              simp only [List.cons_append, List.cons.injEq] at heq
              rcases heq with ⟨rfl, htl⟩
              simpa [execStmts] using hmA tl sfx aid cond htl
      | mbarArrive _ _ =>                                 -- Case 6: `mbarrier_arrive` (tracked in `ControlFlowCfg` & `findTrailArrive`)
          simp only [execStmts, symExecStmts] at hallSome hallTmem ⊢
          obtain ⟨hmR, hmP, hmE, hmA, hmT⟩ := ih regs sregs phase ps hmatch hps hallSome hallTmem
          refine ⟨hmR, hmP, hmE, ?_, hmT⟩
          intro pfx sfx aid cond heq
          cases pfx with
          | nil => simp at heq
          | cons hd tl =>
              simp only [List.cons_append, List.cons.injEq] at heq
              rcases heq with ⟨rfl, htl⟩
              simpa [execStmts] using hmA tl sfx aid cond htl
      | mbarArriveExpectTx _ _ _ =>                       -- Case 7: `mbarrier_arrive_expect_tx`
          simp only [execStmts, symExecStmts] at hallSome hallTmem ⊢
          obtain ⟨hmR, hmP, hmE, hmA, hmT⟩ := ih regs sregs phase ps hmatch hps hallSome hallTmem
          refine ⟨hmR, hmP, hmE, ?_, hmT⟩
          intro pfx sfx aid cond heq
          cases pfx with
          | nil => simp at heq
          | cons hd tl =>
              simp only [List.cons_append, List.cons.injEq] at heq
              rcases heq with ⟨rfl, htl⟩
              simpa [execStmts] using hmA tl sfx aid cond htl
      | syncThreads =>                                    -- Case 8: `sync_threads()` increments `phase` and `ps.stepSync`
          simp only [execStmts, symExecStmts] at hallSome hallTmem ⊢
          obtain ⟨hmR, hmP, hmE, hmA, hmT⟩ := ih regs sregs (phase + 1) ps.stepSync hmatch (phaseSpec_stepSync_sound hps) hallSome hallTmem
          refine ⟨hmR, hmP, hmE, ?_, hmT⟩
          intro pfx sfx aid cond heq
          cases pfx with
          | nil => simp at heq
          | cons hd tl =>
              simp only [List.cons_append, List.cons.injEq] at heq
              rcases heq with ⟨rfl, htl⟩
              simpa [execStmts] using hmA tl sfx aid cond htl
      | tmaCopyG2S ptr barId slotVar spanElems =>         -- Case 9: 2D TMA `g2s` emits `spanElems`-wide write released on `BAR[slot]`
          simp only [symExecStmts, List.all_append, Bool.and_eq_true] at hallSome hallTmem
          obtain ⟨hmR, hmP, hmE, hmA, hmT⟩ := ih regs sregs phase ps hmatch hps hallSome.2 hallTmem
          simp only [execStmts, symExecStmts]
          refine ⟨hmR, hmP, ?_, ?_, hmT⟩
          · intro ev hev
            rw [List.mem_append] at hev
            rcases hev with hhead | htail
            · have hp := hmatch ptr
              have hs := hmatch slotVar
              cases hsp : sregs.getVar ptr <;> cases hss : sregs.getVar slotVar
              all_goals try (rw [hsp, hss] at hallSome; simp at hallSome)
              rename_i sp bid off sSym
              rw [hsp, SymVal.matchesVal] at hp; rw [hss, SymVal.matchesVal] at hs
              rw [hp, hs] at hhead
              dsimp only at hhead
              obtain ⟨k, hkRange, hkEq⟩ := List.mem_map.mp hhead
              subst hkEq
              exact ⟨{ isWrite := true, space := sp, bufId := bid, indexOpt := some off,
                       span := max 1 spanElems, elemDiv := 1, syncBar := some (barId, sSym),
                       pathCond := pCond, phaseSpec := ps },
                     List.mem_append_left _ (List.Mem.head _), off, k,
                     List.mem_range.mp hkRange, Nat.zero_lt_one, rfl, rfl, rfl, rfl, rfl,
                     by simp, hpc, hps⟩
            · obtain ⟨sev, hsev, idxS, k, hrest⟩ := hmE ev htail
              exact ⟨sev, List.mem_append_right _ hsev, idxS, k, hrest⟩
          · intro pfx sfx aid cond heq
            cases pfx with
            | nil => simp at heq
            | cons hd tl =>
                simp only [List.cons_append, List.cons.injEq] at heq
                rcases heq with ⟨rfl, htl⟩
                simpa [execStmts] using hmA tl sfx aid cond htl
      | tmemAlloc ptr _cols =>                            -- Case 10: `tcgen05_alloc` writes `TMEM_ADDR` on warp lane `tx % 32 == 0`
          simp only [symExecStmts, List.all_append, List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true] at hallSome hallTmem
          obtain ⟨hmR, hmP, hmE, hmA, hmT⟩ := ih regs sregs phase ps hmatch hps hallSome.2 hallTmem
          simp only [execStmts, symExecStmts]
          refine ⟨hmR, hmP, ?_, ?_, hmT⟩
          · intro ev hev
            rw [List.mem_append] at hev
            rcases hev with hhead | htail
            · by_cases hl0 : (t.tx % 32 == 0) = true
              · rw [if_pos hl0] at hhead
                have hlane0 : (SymBool.and pCond (.cmp .eq (.binOp .rem 0 .tidX (.const 32)) (.const 0))).eval env t havoc = true := by
                  simp only [SymBool.eval, SymInt.eval, evalCmpOp, evalBinOp, Bool.and_eq_true]
                  exact ⟨hpc, decide_eq_true (beq_iff_eq.mp hl0)⟩
                obtain ⟨idxS, k, hrest⟩ := symSinglePtrEvent_sound hlane0 hps (hmatch ptr) hallSome.1 hhead
                exact ⟨_, List.mem_append_left _ (List.Mem.head _), idxS, k, hrest⟩
              · rw [if_neg hl0] at hhead; cases hhead
            · obtain ⟨sev, hsev, idxS, k, hrest⟩ := hmE ev htail
              exact ⟨sev, List.mem_append_right _ hsev, idxS, k, hrest⟩
          · intro pfx sfx aid cond heq
            cases pfx with
            | nil => simp at heq
            | cons hd tl =>
                simp only [List.cons_append, List.cons.injEq] at heq
                rcases heq with ⟨rfl, htl⟩
                simpa [execStmts] using hmA tl sfx aid cond htl
      | smemDescRead descVar tileBytes =>                 -- Case 11: `tcgen05.mma` SMEM descriptor read (`tileBytes` span)
          simp only [symExecStmts, List.all_append, Bool.and_eq_true] at hallSome hallTmem
          obtain ⟨hmR, hmP, hmE, hmA, hmT⟩ := ih regs sregs phase ps hmatch hps hallSome.2 hallTmem
          simp only [execStmts, symExecStmts]
          refine ⟨hmR, hmP, ?_, ?_, hmT⟩
          · intro ev hev
            rw [List.mem_append] at hev
            rcases hev with hhead | htail
            · have hd := hmatch descVar
              cases hsd : sregs.getVar descVar with
              | smemDesc sid byteOff =>
                  rw [hsd, SymVal.matchesVal] at hd; rw [hd] at hhead
                  dsimp only at hhead
                  obtain ⟨k, hkRange, hkEq⟩ := List.mem_map.mp hhead
                  subst hkEq
                  let eb := max 1 (f.sharedDecls.getD sid { size := 0 }).elemBytes
                  have hebPos : 0 < eb := by dsimp only [eb]; omega
                  exact ⟨{ isWrite := false, space := .shared, bufId := sid, indexOpt := some byteOff,
                           span := max 1 tileBytes, elemDiv := eb, pathCond := pCond, phaseSpec := ps },
                         List.mem_append_left _ (List.Mem.head _), byteOff, k,
                         List.mem_range.mp hkRange, hebPos, rfl, rfl, rfl, rfl, rfl,
                         rfl, hpc, hps⟩
              | _ => rw [hsd] at hallSome; simp at hallSome
            · obtain ⟨sev, hsev, idxS, k, hrest⟩ := hmE ev htail
              exact ⟨sev, List.mem_append_right _ hsev, idxS, k, hrest⟩
          · intro pfx sfx aid cond heq
            cases pfx with
            | nil => simp at heq
            | cons hd tl =>
                simp only [List.cons_append, List.cons.injEq] at heq
                rcases heq with ⟨rfl, htl⟩
                simpa [execStmts] using hmA tl sfx aid cond htl
      | tmemAccess tmemVar hl wc wr =>
          simp only [symExecStmts, List.all_append, Bool.and_eq_true] at hallSome hallTmem
          obtain ⟨hmR, hmP, hmE, hmA, hmT⟩ := ih regs sregs phase ps hmatch hps hallSome hallTmem.2
          simp only [execStmts, symExecStmts]
          refine ⟨hmR, hmP, hmE, ?_, ?_⟩
          · intro pfx sfx aid cond heq
            cases pfx with
            | nil => simp at heq
            | cons hd tl =>
                simp only [List.cons_append, List.cons.injEq] at heq
                rcases heq with ⟨rfl, htl⟩
                simpa [execStmts] using hmA tl sfx aid cond htl
          · intro tev htev
            rw [List.mem_append] at htev
            rcases htev with hhead | htail
            · have ht := hmatch tmemVar
              cases hst : sregs.getVar tmemVar with
              | tmemPtr cols rel =>
                  rw [hst, SymVal.matchesVal] at ht; rw [ht] at hhead
                  simp only [List.mem_singleton] at hhead; subst hhead
                  exact ⟨{ allocCols := cols, relAddrOpt := some rel, heightLanes := hl,
                           widthCols := wc, warpRestricted := wr, pathCond := pCond },
                         List.mem_append_left _ (List.Mem.head _), rel, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩
              | _ => rw [hst] at hallTmem; simp at hallTmem
            · obtain ⟨stev, hstev, relS, hrest⟩ := hmT tev htail
              exact ⟨stev, List.mem_append_right _ hstev, relS, hrest⟩


end CudaOxide.Proofs.Program
