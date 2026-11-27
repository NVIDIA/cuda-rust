/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide.Proofs.Program.Exec
import CudaOxide.Proofs.Program.LoopBounds

/-!
# CFG Abstract Invariant Inference & Soundness (`CudaOxide.Proofs.Program.Invariant`)
-/
namespace CudaOxide.Proofs.Program
open CudaOxide.Mir.Program
open CudaOxide.Contracts.Program

/-- Symbolic entry invariant for one basic block `b` in `MirFunc`. -/
structure SymBlockInv where
  /-- True when block `b` is reachable in the symbolic invariant table. -/
  reachable : Bool := false
  /-- Outer path condition satisfied on entry to block `b` (disjoint from `b`'s loop args). -/
  pathCond : SymBool := .lit false
  /-- Symbolic register bindings on entry to block `b`. -/
  regs : SymRegList := []
  /-- Barrier-phase invariant on entry to block `b`. -/
  phaseSpec : PhaseSpec := .bot
  /-- Verified inductive `(varId, width, bound)` upper bounds on loop-header `BlockArg`s. -/
  loopBounds : List (Nat × Nat × Nat) := []
  deriving Repr, DecidableEq, Lean.ToExpr

/-- Conjoin two `SymBool` predicates, simplifying `.lit` identities. -/
def SymBool.conj (a b : SymBool) : SymBool :=
  if a == .lit true then b
  else if b == .lit true then a
  else if a == .lit false || b == .lit false then .lit false
  else if a == b then a
  else .and a b

/-- Convert `loopBounds` triples `(vid, width, lim)` into `vid < lim` conjunctions. -/
def loopBoundsCond : List (Nat × Nat × Nat) → SymBool
  | [] => .lit true
  | (vid, w, lim) :: rest =>
      SymBool.conj (.cmp .lt (.havoc vid w) (.const lim)) (loopBoundsCond rest)

/-- Effective block-entry path condition combining `pathCond` with verified `loopBounds`. -/
def SymBlockInv.effCond (inv : SymBlockInv) : SymBool :=
  SymBool.conj inv.pathCond (loopBoundsCond inv.loopBounds)

/-- Helper: `SymBool.conj a b` evaluates to `true` when both `a` and `b` evaluate to `true`. -/
theorem symBool_conj_true {env : InputEnv} {t : ThreadCtx} {havoc : Nat → Nat}
    {a b : SymBool} (ha : a.eval env t havoc = true) (hb : b.eval env t havoc = true) :
    (SymBool.conj a b).eval env t havoc = true := by
  unfold SymBool.conj
  split
  · exact hb
  · split
    · exact ha
    · split
      · rename_i hf
        rcases Bool.or_eq_true_iff.mp hf with h1 | h2
        · rw [beq_iff_eq.mp h1] at ha; contradiction
        · rw [beq_iff_eq.mp h2] at hb; contradiction
      · split
        · exact ha
        · simp [SymBool.eval, ha, hb]

/-- Logical complement peeling double negation (`.not x -> x`, `y -> .not y`). -/
def SymBool.neg : SymBool → SymBool
  | .lit b => .lit (!b)
  | .not inner => inner
  | other => .not other

/-- Flatten nested `.and` trees into a conjunction list (`true -> []`). -/
def SymBool.toConjList : SymBool → List SymBool
  | .lit true => []
  | .and p q => p.toConjList ++ q.toConjList
  | other => [other]

/-- Rebuild a right-associated `SymBool` conjunction from a list (`[] -> true`). -/
def SymBool.ofConjList : List SymBool → SymBool
  | [] => .lit true
  | [x] => x
  | x :: rest => SymBool.conj x (SymBool.ofConjList rest)

/-- Flatten nested `.or` trees into a disjunction list (`false -> []`). -/
def SymBool.toDisjList : SymBool → List SymBool
  | .lit false => []
  | .or p q => p.toDisjList ++ q.toDisjList
  | other => [other]

/-- Rebuild a right-associated `SymBool` disjunction from a list (`[] -> false`). -/
def SymBool.ofDisjList : List SymBool → SymBool
  | [] => .lit false
  | [x] => x
  | x :: rest => .or x (SymBool.ofDisjList rest)

/-- Strip shared leading conjuncts from two conjunction lists, returning `(common, remA, remB)`. -/
def stripCommonPrefix : List SymBool → List SymBool → (List SymBool × List SymBool × List SymBool)
  | x :: ra, y :: rb =>
      if x == y then
        let (c, r1, r2) := stripCommonPrefix ra rb
        (x :: c, r1, r2)
      else
        ([], x :: ra, y :: rb)
  | ra, rb => ([], ra, rb)

/-- Disjoin two `SymBool` predicates, factoring shared prefix conjuncts `(P ∧ Q1) ∨ (P ∧ Q2) -> P ∧ (Q1 ∨ Q2)`,
    collapsing `k`-way `if/else` diamonds, and widening deep reconverged diamond residuals (`depth > 6`) to
    their dominating conjunctive prefix `P` so `pathCond` stays strictly `O(nesting depth)` on huge kernels. -/
def SymBool.disj (a b : SymBool) : SymBool :=
  if a == .lit false then b
  else if b == .lit false then a
  else if a == .lit true || b == .lit true then .lit true
  else if a == b then a
  else if a == b.neg || b == a.neg then .lit true
  else
    let (common, ra, rb) := stripCommonPrefix a.toConjList b.toConjList
    let pref := SymBool.ofConjList common
    if ra.isEmpty || rb.isEmpty then
      pref
    else
      let da := (SymBool.ofConjList ra).toDisjList
      let db := (SymBool.ofConjList rb).toDisjList
      let rbReduced := rb.filter fun c => !(da.any fun d => c == d.neg || d == c.neg)
      let raReduced := ra.filter fun c => !(db.any fun d => c == d.neg || d == c.neg)
      if rbReduced.isEmpty || raReduced.isEmpty then
        pref
      else
        let remA := SymBool.ofConjList raReduced
        let remB := SymBool.ofConjList rbReduced
        if remA == .lit true || remB == .lit true || remA == remB.neg || remB == remA.neg then
          pref
        else if (symBoolRemNodes 28 remA).isNone || (symBoolRemNodes 28 remB).isNone then
          pref
        else
          SymBool.conj pref (.or remA remB)

/-- Merge two `SymVal` register lists using `iteCond` for 2-branch `.ite`s and `(c1, c2)` for `.condPtr`s. -/
def mergeSymRegLists (iteCond c1 c2 : SymBool) (r1 r2 : SymRegList) : SymRegList :=
  let len := max r1.length r2.length
  let condSmall := iteCond != .lit true && iteCond != .lit false && (symBoolRemNodes 96 iteCond).isSome
  (List.range len).map fun v =>
    let v1 := r1.getVar v
    let v2 := r2.getVar v
    if v1 == v2 then v1
    else
      match v1, v2 with
      | .ptr sp bid off, .unk => if c1 == .lit true || c1 == .lit false then .unk else .condPtr c1 sp bid off
      | .unk, .ptr sp bid off => if c2 == .lit true || c2 == .lit false then .unk else .condPtr c2 sp bid off
      | .condPtr g sp bid off, .unk => .condPtr (SymBool.conj c1 g) sp bid off
      | .unk, .condPtr g sp bid off => .condPtr (SymBool.conj c2 g) sp bid off
      | _, _ =>
          if !condSmall then .unk
          else
            match v1, v2 with
            | .int e1, .int e2 =>
                if (symIntRemNodes 36 e1).isNone || (symIntRemNodes 36 e2).isNone then .unk
                else if iteCond == c1 && symImpliesVal c1 c2 false
                    && c2 != .lit true && c2 != .lit false && (symBoolRemNodes 96 c2).isSome
                    && !(match c2 with | .or _ _ => true | _ => false) then
                  .int (.ite c2 e2 e1)
                else
                  match e2 with
                  | .havoc _ _ =>
                      if iteCond == c1 && c2 != .lit true && c2 != .lit false && (symBoolRemNodes 96 c2).isSome then
                        .int (.ite c2 e2 e1)
                      else
                        .int (.ite iteCond e1 e2)
                  | _ => .int (.ite iteCond e1 e2)
            | .bool b1, .bool b2 =>
                if (symBoolRemNodes 28 b1).isNone || (symBoolRemNodes 28 b2).isNone then .unk
                else .bool (.ite iteCond b1 b2)
            | .smemOff s1 b1, .smemOff s2 b2 =>
                if s1 == s2 && (symIntRemNodes 36 b1).isSome && (symIntRemNodes 36 b2).isSome then .smemOff s1 (.ite iteCond b1 b2) else .unk
            | .smemDesc s1 b1, .smemDesc s2 b2 =>
                if s1 == s2 && (symIntRemNodes 36 b1).isSome && (symIntRemNodes 36 b2).isSome then .smemDesc s1 (.ite iteCond b1 b2) else .unk
            | .tmemPtr m1 b1, .tmemPtr m2 b2 =>
                if m1 == m2 && (symIntRemNodes 36 b1).isSome && (symIntRemNodes 36 b2).isSome then .tmemPtr m1 (.ite iteCond b1 b2) else .unk
            | _, _ => .unk

/-- Merge two `PhaseSpec` values at an acyclic join block. -/
def mergePhaseSpecs (p1 p2 : PhaseSpec) : PhaseSpec :=
  match p1, p2 with
  | .bot, x | x, .bot => x
  | .exact k1, .exact k2 => if k1 == k2 then .exact k1 else .any
  | .modGe min1 m1 r1, .modGe min2 m2 r2 =>
      if m1 == m2 && r1 == r2 then .modGe (min min1 min2) m1 r1 else .any
  | _, _ => .any

/-- Count `Stmt.syncThreads` barriers along a simple `O(blocks)` cycle starting at loop header `hdr`. -/
def inferLoopBarrierModulus (f : MirFunc) (hdr : Nat) : Nat :=
  let n := f.blocks.length
  let rec dfs : Nat → Nat → List Nat → Nat → Option Nat × List Nat
    | 0, _, seen, _ => (none, seen)
    | fuel + 1, u, seen, accSync =>
        let blk := f.blockAt u
        let blkSyncs := (blk.stmts.filter (· == .syncThreads)).length
        let tot := accSync + blkSyncs
        let succs := match blk.term with
          | .goto d _ => [d]
          | .condBr _ t _ e _ => [t, e]
          | _ => []
        if succs.contains hdr && u ≥ hdr then
          (some tot, seen)
        else
          succs.foldl (fun (found, sn) s =>
            match found with
            | some _ => (found, sn)
            | none =>
                if s > hdr && s < n && !(sn.contains s) then
                  dfs fuel s (s :: sn) tot
                else
                  (none, sn)) (none, seen)
  match (dfs (n + 1) hdr [hdr] 0).1 with
  | some m => m
  | none => 0

/-- Construct initial `SymRegList` for `block 0` from `f.params`. -/
def initialSymRegs (f : MirFunc) : SymRegList :=
  let b0 := f.blockAt 0
  ((List.range b0.args.length).zip b0.args).foldl (fun acc (idx, ba) =>
    match f.params[idx]? with
    | some (.scalar _) => acc.setVar ba.varId (.int (.paramScalar idx))
    | _ => acc.setVar ba.varId .unk) []


/-- SSA variables referenced inside a single `Rhs`. -/
def rhsUsedVars : Rhs → List Nat
  | .copy s | .notBool s | .intCast _ s | .load s | .cvtaShared s | .loadTmemBase s _ | .ctaUint1 _ _ s | .ctaUnary s => [s]
  | .binOp _ _ l r | .checkedOvf _ _ l r | .cmpOp _ l r | .ptrOffset l r | .ctaBin l r => [l, r]
  | .mbarTryWait _ sv _ => [sv]
  | _ => []

/-- SSA variables referenced inside a single `Stmt`. -/
def stmtUsedVars : Stmt → List Nat
  | .assign _ rhs => rhsUsedVars rhs
  | .store p v => [p, v]
  | .assertCond _ c => [c]
  | .barrierSlotCheck p | .tmemAlloc p _ | .smemDescRead p _ | .tmemAccess p _ _ _ => [p]
  | .clusterRankCheck r => [r]
  | .mbarArrive _ sv _ | .mbarArriveExpectTx _ sv _ => [sv]
  | .tmaCopyG2S p _ sv _ => [p, sv]
  | .syncThreads => []

/-- Collect variables read before being overwritten inside `stmts`. -/
def upwardExposedVars : List Nat → List Stmt → List Nat
  | _, [] => []
  | defined, stmt :: rest =>
      let used := (stmtUsedVars stmt).filter fun v => !(defined.contains v)
      let defined' := match stmt with
        | .assign dst _ => if defined.contains dst then defined else dst :: defined
        | _ => defined
      let tail := upwardExposedVars defined' rest
      used.foldl (fun acc v => if acc.contains v then acc else v :: acc) tail

/-- Compute the set of register IDs live across basic-block boundaries
    (any register upward-exposed at block entry or read by a block terminator). -/
def crossBlockUsedVars (f : MirFunc) : List Nat :=
  f.blocks.foldl (fun acc blk =>
    let termVars := match blk.term with
      | .goto _ args => args
      | .condBr cv _ tArgs _ fArgs => cv :: (tArgs ++ fArgs)
      | _ => []
    let exposed := upwardExposedVars (blk.args.map BlockArg.varId) blk.stmts ++ termVars
    exposed.foldl (fun acc2 v => if acc2.contains v then acc2 else v :: acc2) acc) []

/-- Drop intra-block temporary registers (`v ∉ crossLive`) to `.unk` and trim trailing `.unk`s. -/
def pruneIntraBlockRegs (crossLive : List Nat) (regs : SymRegList) : SymRegList :=
  let pruned := ((List.range regs.length).zip regs).map fun (v, sv) =>
    match sv with
    | .unk => .unk
    | _ => if crossLive.contains v then sv else .unk
  pruned.reverse.dropWhile (· == .unk) |>.reverse

/-- Single forward pass of `inferSymTable` parameterized by pre-seeded per-header `loopBounds`. -/
def inferSymTablePass (f : MirFunc) (hdrBounds : Nat → List (Nat × Nat × Nat)) : List SymBlockInv :=
  let n := f.blocks.length
  if n == 0 then []
  else
    let crossLive := crossBlockUsedVars f
    let init0 : SymBlockInv :=
      { reachable := true, pathCond := .lit true, regs := initialSymRegs f, phaseSpec := .exact 0 }
    let initTable := init0 :: List.replicate (n - 1) {}
    (List.range n).foldl (fun tbl b =>
      let inv := tbl.getD b {}
      if !inv.reachable then tbl
      else
        let blk := f.blockAt b
        let effC := inv.effCond
        let (outRegs, outPs, _, _, _) := symExecStmts f effC inv.regs inv.phaseSpec blk.term blk.stmts
        let pushEdge := fun (acc : List SymBlockInv) (edgeCond : SymBool) (dst : Nat) (argVars : List Nat) =>
          if dst ≤ b || dst ≥ n then acc
          else
            let dstBlk := f.blockAt dst
            let curDst := acc.getD dst {}
            let dstKeep := crossLive ++ (dstBlk.args.map BlockArg.varId)
            let boundRegs := pruneIntraBlockRegs dstKeep (bindSymBlockArgs outRegs outRegs dstBlk.args argVars)
            let nextRegs :=
              if dstBlk.isLoopHeader then
                dstBlk.args.foldl (fun r ba =>
                  if ba.width ≤ 1 then r.setVar ba.varId .unk
                  else r.setVar ba.varId (.int (.havoc ba.varId ba.width))) boundRegs
              else
                boundRegs
            let nextPs :=
              if dstBlk.isLoopHeader then
                let m := inferLoopBarrierModulus f dst
                if m == 0 then outPs
                else
                  match outPs with
                  | .exact k => .modGe k m (k % m)
                  | .modGe minP m0 r0 => if m == m0 then .modGe minP m0 r0 else .any
                  | _ => .any
              else
                outPs
            let lbs := if dstBlk.isLoopHeader then hdrBounds dst else []
            let merged : SymBlockInv :=
              if !curDst.reachable then
                { reachable := true, pathCond := edgeCond, regs := nextRegs,
                  phaseSpec := nextPs, loopBounds := lbs }
              else
                let (_, raRel, rbRel) := stripCommonPrefix curDst.pathCond.toConjList edgeCond.toConjList
                let relCond :=
                  if raRel.isEmpty && rbRel.isEmpty then .lit true
                  else
                    let q1 := SymBool.ofConjList raRel
                    let q2 := SymBool.ofConjList rbRel
                    if q1 != .lit true && (q1 == q2.neg || q2 == q1.neg) then q1 else curDst.pathCond
                { reachable := true
                , pathCond := SymBool.disj curDst.pathCond edgeCond
                , regs := mergeSymRegLists relCond curDst.pathCond edgeCond curDst.regs nextRegs
                , phaseSpec := mergePhaseSpecs curDst.phaseSpec nextPs
                , loopBounds := curDst.loopBounds }
            acc.set dst merged
        match blk.term with
        | .goto dst args => pushEdge tbl effC dst args
        | .condBr cv tBlk tArgs fBlk fArgs =>
            let (tCond, fCond) := match outRegs.getVar cv with
              | .bool cSym => (SymBool.conj effC cSym, SymBool.conj effC (.not cSym))
              | _ => (effC, effC)
            let tbl1 := pushEdge tbl tCond tBlk tArgs
            pushEdge tbl1 fCond fBlk fArgs
        | .ret | .unreachable => tbl) initTable

/-- True when `term` has `target` as one of its outgoing successor blocks. -/
def termTargets (term : Terminator) (target : Nat) : Bool :=
  match term with
  | .goto dst _ => dst == target
  | .condBr _ tBlk _ fBlk _ => tBlk == target || fBlk == target
  | .ret | .unreachable => false

/-- Prune `cand` to only those `(hdr, lbs)` whose bounds hold across EVERY incoming edge in `tbl`. -/
def pruneHdrBounds (f : MirFunc) (tbl : List SymBlockInv)
    (cand : List (Nat × List (Nat × Nat × Nat))) : List (Nat × List (Nat × Nat × Nat)) :=
  let n := f.blocks.length
  cand.filterMap fun (hdr, lbs) =>
    let hdrBlk := f.blockAt hdr
    let inBlocks := (List.range n).filter fun b =>
      (tbl.getD b {}).reachable && termTargets (f.blockAt b).term hdr
    let lbsOk := lbs.filter fun (vid, _, lim) =>
      inBlocks.all fun b =>
        let inv := tbl.getD b {}
        let blk := f.blockAt b
        let effC := inv.effCond
        let (outRegs, _, _, _, _) := symExecStmts f effC inv.regs inv.phaseSpec blk.term blk.stmts
        let checkOne := fun ec dst args =>
          if dst != hdr then true
          else
            let pr := bindSymBlockArgs outRegs outRegs hdrBlk.args args
            match pr.getVar vid with
            | .int e => symIntBoundedUnder ec e lim
            | _ => false
        match blk.term with
        | .goto dst args => checkOne effC dst args
        | .condBr cv tBlk tArgs fBlk fArgs =>
            let (tC, fC) := match outRegs.getVar cv with
              | .bool cSym => (SymBool.conj effC cSym, SymBool.conj effC (.not cSym))
              | _ => (effC, effC)
            checkOne tC tBlk tArgs && checkOne fC fBlk fArgs
        | _ => true
    if lbsOk.isEmpty then none else some (hdr, lbsOk)

/-- Infer the `List SymBlockInv` invariant table for `f` (including nested inductive ring bounds
    discovered via Houdini greatest-fixpoint pruning and independently certified by `validSymFunc`). -/
def inferSymTable (f : MirFunc) : List SymBlockInv :=
  let baseTbl := inferSymTablePass f (fun _ => [])
  let n := f.blocks.length
  let hdrs := (List.range n).filter fun hdr =>
    (f.blockAt hdr).isLoopHeader && (baseTbl.getD hdr {}).reachable
  if hdrs.isEmpty then baseTbl
  else
    -- 1. Collect all wrap/mod bounds (`suggestRingBound`) from backedges across all loop headers:
    let perHdrBackEdges : List (Nat × List (Nat × SymRegList × List Nat)) :=
      hdrs.map fun hdr =>
        let edges := (List.range n).flatMap fun b =>
          let inv := baseTbl.getD b {}
          if !inv.reachable || !termTargets (f.blockAt b).term hdr then []
          else
            let blk := f.blockAt b
            let (outRegs, _, _, _, _) := symExecStmts f inv.effCond inv.regs inv.phaseSpec blk.term blk.stmts
            match blk.term with
            | .goto dst args => if dst == hdr then [(b, outRegs, args)] else []
            | .condBr _ tBlk tArgs fBlk fArgs =>
                (if tBlk == hdr then [(b, outRegs, tArgs)] else []) ++
                (if fBlk == hdr then [(b, outRegs, fArgs)] else [])
            | _ => []
        (hdr, edges)
    let globalLims : List Nat :=
      perHdrBackEdges.foldl (fun acc (hdr, edges) =>
        let hdrBlk := f.blockAt hdr
        hdrBlk.args.foldl (fun acc2 fa =>
          edges.foldl (fun acc3 (b, outRegs, args) =>
            if b < hdr then acc3
            else
              let pr := bindSymBlockArgs outRegs outRegs hdrBlk.args args
              match pr.getVar fa.varId with
              | .int e =>
                  match suggestRingBound e with
                  | some lim => if acc3.contains lim then acc3 else lim :: acc3
                  | none => acc3
              | _ => acc3) acc2) acc) []
    -- 2. Seed optimistic candidate bounds (direct wrap bound OR outer loop arg initialized `< lim`):
    let seedCand : List (Nat × List (Nat × Nat × Nat)) :=
      perHdrBackEdges.filterMap fun (hdr, edges) =>
        let hdrBlk := f.blockAt hdr
        let lbs := hdrBlk.args.filterMap fun fa =>
          if fa.width ≤ 1 then none
          else
            let directLim := edges.findSome? fun (b, outRegs, args) =>
              if b < hdr then none
              else
                let pr := bindSymBlockArgs outRegs outRegs hdrBlk.args args
                match pr.getVar fa.varId with
                | .int e => suggestRingBound e
                | _ => none
            match directLim with
            | some lim => some (fa.varId, fa.width, lim)
            | none =>
                -- Check if forward entry edges pass a constant `< lim` for a `lim ∈ globalLims`:
                globalLims.findSome? fun lim =>
                  let fwdOk := edges.all fun (b, outRegs, args) =>
                    if b ≥ hdr then true
                    else
                      let pr := bindSymBlockArgs outRegs outRegs hdrBlk.args args
                      match pr.getVar fa.varId with
                      | .int (.const c) => decide (c < lim)
                      | _ => false
                  if fwdOk then some (fa.varId, fa.width, lim) else none
        if lbs.isEmpty then none else some (hdr, lbs)
    if seedCand.isEmpty then baseTbl
    else
      let rec houdini (fuel : Nat) (cand : List (Nat × List (Nat × Nat × Nat))) : List SymBlockInv :=
        let tbl := inferSymTablePass f fun hdr =>
          match cand.find? (fun p => p.1 == hdr) with
          | some (_, lbs) => lbs
          | none => []
        match fuel with
        | 0 => inferSymTablePass f (fun _ => [])
        | fuel' + 1 =>
            let pruned := pruneHdrBounds f tbl cand
            if pruned == cand then tbl
            else if pruned.isEmpty then baseTbl
            else houdini fuel' pruned
      houdini 5 seedCand

/-- Check pairwise distinctness of formal `BlockArg.varId`s in `args`. -/
def distinctBlockArgIds : List BlockArg → Bool
  | [] => true
  | fa :: rest => !(rest.map BlockArg.varId).contains fa.varId && distinctBlockArgIds rest

/-- Linear `O(vars)` register-list subsumption checker avoiding quadratic `getVar` indexing in `whnf`. -/
def checkRegsEquiv (edgeCond : SymBool) : List SymVal → List SymVal → Bool
  | [], _ => true
  | d :: ds, p :: ps =>
      (match d with | .unk => true | _ => symValEquivUnder edgeCond d p) &&
      checkRegsEquiv edgeCond ds ps
  | d :: ds, [] =>
      (match d with | .unk => true | _ => symValEquivUnder edgeCond d .unk) &&
      checkRegsEquiv edgeCond ds []

/-- Linear `O(vars)` loop-header outer-register subsumption checker (`idx` tracks current `varId`). -/
def checkLoopRegsEquiv (hdrArgIds : List Nat) (edgeCond : SymBool) :
    Nat → List SymVal → List SymVal → Bool
  | _, [], _ => true
  | idx, d :: ds, p :: ps =>
      let ok :=
        if hdrArgIds.contains idx then true
        else match d with
          | .unk => true
          | _ => d.disjointHavoc hdrArgIds && symValEquivUnder edgeCond d p
      ok && checkLoopRegsEquiv hdrArgIds edgeCond (idx + 1) ds ps
  | idx, d :: ds, [] =>
      let ok :=
        if hdrArgIds.contains idx then true
        else match d with
          | .unk => true
          | _ => d.disjointHavoc hdrArgIds && symValEquivUnder edgeCond d .unk
      ok && checkLoopRegsEquiv hdrArgIds edgeCond (idx + 1) ds []

/-- Soundness of `checkRegsEquiv`: implies `symValEquivUnder` at every `v : Nat`. -/
theorem checkRegsEquiv_getVar {edgeCond : SymBool} :
    ∀ (ds ps : List SymVal) (v : Nat),
      checkRegsEquiv edgeCond ds ps = true →
      symValEquivUnder edgeCond (ds.getD v .unk) (ps.getD v .unk) = true
  | [], _, _, _ => by simp [List.getD, symValEquivUnder]
  | d :: _, p :: _, 0, h => by
      simp only [checkRegsEquiv, Bool.and_eq_true] at h
      cases d <;> first | exact h.1 | simp [List.getD, symValEquivUnder]
  | _ :: ds, _ :: ps, v + 1, h => by
      simp only [checkRegsEquiv, Bool.and_eq_true] at h
      exact checkRegsEquiv_getVar ds ps v h.2
  | d :: _, [], 0, h => by
      simp only [checkRegsEquiv, Bool.and_eq_true] at h
      cases d <;> first | exact h.1 | simp [List.getD, symValEquivUnder]
  | _ :: ds, [], v + 1, h => by
      simp only [checkRegsEquiv, Bool.and_eq_true] at h
      simpa [List.getD] using checkRegsEquiv_getVar ds [] v h.2

/-- Soundness of `checkLoopRegsEquiv`: implies `disjointHavoc` and `symValEquivUnder` outside `hdrArgIds`. -/
theorem checkLoopRegsEquiv_getVar {hdrArgIds : List Nat} {edgeCond : SymBool} :
    ∀ (idx : Nat) (ds ps : List SymVal) (k : Nat),
      checkLoopRegsEquiv hdrArgIds edgeCond idx ds ps = true →
      hdrArgIds.contains (idx + k) = false →
      (ds.getD k .unk).disjointHavoc hdrArgIds = true ∧
      symValEquivUnder edgeCond (ds.getD k .unk) (ps.getD k .unk) = true
  | _, [], _, _, _, _ => by simp [List.getD, SymVal.disjointHavoc, symValEquivUnder]
  | idx, d :: _, p :: _, 0, h, hnc => by
      simp only [Nat.add_zero] at hnc
      simp only [checkLoopRegsEquiv, hnc, Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true] at h
      cases d <;> first | exact Bool.and_eq_true_iff.mp h.1 | simp [List.getD, SymVal.disjointHavoc, symValEquivUnder]
  | idx, _ :: ds, _ :: ps, k + 1, h, hnc => by
      simp only [checkLoopRegsEquiv, Bool.and_eq_true] at h
      have hnc' : hdrArgIds.contains ((idx + 1) + k) = false := by
        simpa [Nat.add_assoc, Nat.add_comm 1 k] using hnc
      exact checkLoopRegsEquiv_getVar (idx + 1) ds ps k h.2 hnc'
  | idx, d :: _, [], 0, h, hnc => by
      simp only [Nat.add_zero] at hnc
      simp only [checkLoopRegsEquiv, hnc, Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true] at h
      cases d <;> first | exact Bool.and_eq_true_iff.mp h.1 | simp [List.getD, SymVal.disjointHavoc, symValEquivUnder]
  | idx, _ :: ds, [], k + 1, h, hnc => by
      simp only [checkLoopRegsEquiv, Bool.and_eq_true] at h
      have hnc' : hdrArgIds.contains ((idx + 1) + k) = false := by
        simpa [Nat.add_assoc, Nat.add_comm 1 k] using hnc
      simpa [List.getD] using checkLoopRegsEquiv_getVar (idx + 1) ds [] k h.2 hnc'

/-- Validate a single CFG edge `(b -> dst)` passing `argVars` under `edgeCond` and `(outRegs, outPs)`. -/
def validSymEdge (f : MirFunc) (tbl : List SymBlockInv)
    (edgeCond : SymBool) (outRegs : SymRegList) (outPs : PhaseSpec)
    (dst : Nat) (argVars : List Nat) : Bool :=
  decide (dst < f.blocks.length) &&
  let dstBlk := f.blockAt dst
  let dstInv := tbl.getD dst {}
  decide (argVars.length == dstBlk.args.length) &&
  distinctBlockArgIds dstBlk.args &&
  dstInv.reachable &&
  dstInv.phaseSpec.subsumes outPs &&
  let passedRegs := bindSymBlockArgs outRegs outRegs dstBlk.args argVars
  let maxVars := max dstInv.regs.length passedRegs.length
  if !dstBlk.isLoopHeader then
    decide (dstInv.loopBounds == []) &&
    symImpliesVal edgeCond dstInv.pathCond true &&
    checkRegsEquiv edgeCond dstInv.regs passedRegs
  else
    let hdrArgIds := dstBlk.args.map BlockArg.varId
    (hdrArgIds.all fun vid => decide (vid < maxVars)) &&
    dstInv.pathCond.disjointHavoc hdrArgIds &&
    symImpliesVal edgeCond dstInv.pathCond true &&
    (dstInv.loopBounds.all fun (vid, w, lim) =>
      hdrArgIds.contains vid &&
      decide (0 < w) &&
      (dstInv.regs.getVar vid == .int (.havoc vid w)) &&
      match passedRegs.getVar vid with
      | .int e => symIntFitsWidth f w e && symIntBoundedUnder edgeCond e lim
      | _ => false) &&
    (dstBlk.args.all fun fa =>
      if fa.width == 0 then
        dstInv.regs.getVar fa.varId == .unk
      else
        dstInv.regs.getVar fa.varId == .int (.havoc fa.varId fa.width) &&
        match passedRegs.getVar fa.varId with
        | .int e => symIntFitsWidth f fa.width e
        | _ => false) &&
    checkLoopRegsEquiv hdrArgIds edgeCond 0 dstInv.regs passedRegs

/-- Computable certificate checker verifying that `tbl` is a sound inductive symbolic
    invariant for every block and CFG edge of `f` (`O(blocks * vars)`). -/
def validSymFunc (f : MirFunc) (tbl : List SymBlockInv) : Bool :=
  decide (0 < f.blocks.length) &&
  decide (tbl.length == f.blocks.length) &&
  let inv0 := tbl.getD 0 {}
  let initSR := initialSymRegs f
  inv0.reachable &&
  decide (inv0.pathCond == .lit true) &&
  decide (inv0.loopBounds == []) &&
  decide (inv0.phaseSpec == .exact 0) &&
  decide (inv0.regs == initSR) &&
  (List.range f.blocks.length).all fun b =>
    let inv := tbl.getD b {}
    if !inv.reachable then true
    else
      let blk := f.blockAt b
      let effC := inv.effCond
      let (outRegs, outPs, mevs, aevs, tevs) := symExecStmts f effC inv.regs inv.phaseSpec blk.term blk.stmts
      mevs.all (fun mev => mev.indexOpt.isSome) &&
      aevs.all (fun aev => aev.condOpt.isSome) &&
      tevs.all (fun tev => tev.relAddrOpt.isSome) &&
      match blk.term with
      | .ret | .unreachable => true
      | .goto dst args =>
          validSymEdge f tbl effC outRegs outPs dst args
      | .condBr cv tBlk tArgs fBlk fArgs =>
          match outRegs.getVar cv with
          | .bool cSym =>
              validSymEdge f tbl (SymBool.conj effC cSym) outRegs outPs tBlk tArgs &&
              validSymEdge f tbl (SymBool.conj effC (.not cSym)) outRegs outPs fBlk fArgs
          | .unk =>
              validSymEdge f tbl effC outRegs outPs tBlk tArgs &&
              validSymEdge f tbl effC outRegs outPs fBlk fArgs
          | _ => false

/-- Extract the `Nat` payload of a concrete `Val` (`0` otherwise). -/
def Val.asInt : Val → Nat
  | .int n => n
  | _ => 0


/-- Correspondence of `initialRegs` and `initialSymRegs` at kernel entry (`block 0`). -/
theorem initialSymRegs_correct (f : MirFunc) (env : InputEnv) (t : ThreadCtx) (havoc : Nat → Nat) :
    ∀ v, ((initialSymRegs f).getVar v).matchesVal env t havoc (initialRegs f env v) := by
  unfold initialRegs initialSymRegs
  dsimp only
  have hbase : ∀ (pairs : List (Nat × BlockArg)) (r : RegFile) (sr : SymRegList),
      (∀ v, (sr.getVar v).matchesVal env t havoc (r v)) →
      ∀ v, ((pairs.foldl (fun acc (idx, ba) =>
              match f.params[idx]? with
              | some (.scalar _) => acc.setVar ba.varId (.int (.paramScalar idx))
              | _ => acc.setVar ba.varId .unk) sr).getVar v).matchesVal env t havoc
            ((pairs.foldl (fun acc (idx, ba) =>
              match f.params[idx]? with
              | some (.scalar _) => acc.set ba.varId (.int (env.scalar idx))
              | _ => acc.set ba.varId .unk) r) v) := by
    intro pairs
    induction pairs with
    | nil => intro r sr hm v; exact hm v
    | cons hd tl ih =>
        intro r sr hm v
        rcases hd with ⟨idx, ba⟩
        dsimp only [List.foldl_cons]
        apply ih
        intro k
        cases f.params[idx]? with
        | none =>
            unfold RegFile.set; rw [getVar_setVar]
            by_cases hk : (k == ba.varId) = true
            · simp [hk, SymVal.matchesVal]
            · simp [hk]; exact hm k
        | some pk =>
            cases pk <;> (
              unfold RegFile.set; rw [getVar_setVar]
              by_cases hk : (k == ba.varId) = true
              · simp [hk, SymVal.matchesVal, SymInt.eval]
              · simp [hk]; exact hm k)
  apply hbase
  intro v; simp [SymRegList.getVar, List.getD, SymVal.matchesVal]

/-- Helper: prove `(loopBoundsCond lbs).eval env t havoc' = true` from per-triple `havoc'` bounds. -/
theorem loopBoundsCond_eval_true {env : InputEnv} {t : ThreadCtx} {havoc' : Nat → Nat} :
    ∀ lbs : List (Nat × Nat × Nat),
      (∀ p ∈ lbs, wordMod p.2.1 (havoc' p.1) < p.2.2) →
      (loopBoundsCond lbs).eval env t havoc' = true
  | [], _ => rfl
  | (vid, w, lim) :: rest, hall => by
      dsimp only [loopBoundsCond]
      apply symBool_conj_true
      · have hhd := hall (vid, w, lim) (List.Mem.head _)
        simp only [SymBool.eval, SymInt.eval, evalCmpOp]
        exact decide_eq_true hhd
      · exact loopBoundsCond_eval_true rest (fun p hp => hall p (List.Mem.tail _ hp))

/-- Edge preservation lemma: if `validSymEdge` holds on `(b -> dst)`, concrete `(outRegs, outPhase)`
    transitions to `dstInv` under an updated `havoc'`. -/
theorem validSymEdge_sound {f : MirFunc} {tbl : List SymBlockInv} {env : InputEnv} {t : ThreadCtx}
    (hvalidIn : f.validInput env)
    {edgeCond : SymBool} {outRegs : RegFile} {outSR : SymRegList}
    {outPhase : Nat} {outPs : PhaseSpec} {havoc : Nat → Nat}
    {dst : Nat} {argVars : List Nat}
    (hedge : validSymEdge f tbl edgeCond outSR outPs dst argVars = true)
    (hecond : edgeCond.eval env t havoc = true)
    (hmatch : ∀ v, (outSR.getVar v).matchesVal env t havoc (outRegs v))
    (hps : outPs.matchesPhase outPhase = true) :
    (tbl.getD dst {}).reachable = true ∧
    ∃ havoc' : Nat → Nat,
      (tbl.getD dst {}).effCond.eval env t havoc' = true ∧
      (tbl.getD dst {}).phaseSpec.matchesPhase outPhase = true ∧
      ∀ v, ((tbl.getD dst {}).regs.getVar v).matchesVal env t havoc'
             ((bindBlockArgs outRegs outRegs (f.blockAt dst).args argVars) v) := by
  unfold validSymEdge at hedge
  dsimp only at hedge
  have hpassed : ∀ v, ((bindSymBlockArgs outSR outSR (f.blockAt dst).args argVars).getVar v).matchesVal env t havoc
      ((bindBlockArgs outRegs outRegs (f.blockAt dst).args argVars) v) :=
    bindSymBlockArgs_correct hmatch (f.blockAt dst).args argVars outRegs outSR hmatch
  by_cases hhdr : (f.blockAt dst).isLoopHeader = false    -- Split: ordinary forward join vs. loop header
  · rw [hhdr] at hedge                                    -- Case 1: Acyclic destination (`havoc' := havoc` unchanged)
    simp only [Bool.not_false, ↓reduceIte, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hedge
    obtain ⟨_, ⟨⟨⟨_, _⟩, hreach⟩, hsub⟩, ⟨hlbsEmpty, hpcond⟩, hallV⟩ := hedge
    refine ⟨hreach, havoc, ?_, phaseSpec_subsumes_sound hsub hps, ?_⟩
    · unfold SymBlockInv.effCond                          -- Goal 1a: `edgeCond` implies `dstInv.pathCond`
      rw [hlbsEmpty]
      exact symBool_conj_true (symImpliesVal_sound hecond _ true hpcond) rfl
    · intro v                                             -- Goal 1b: Every variable in `dstInv.regs` subsumes `passedSR`
      let passedSR := bindSymBlockArgs outSR outSR (f.blockAt dst).args argVars
      let dstInv := tbl.getD dst {}
      have heqv := checkRegsEquiv_getVar dstInv.regs passedSR v hallV
      exact symValEquivUnder_sound hecond _ _ _ heqv (hpassed v)
  · have hhdrT : (f.blockAt dst).isLoopHeader = true := by -- Case 2: Loop header (`rebind havoc' on hdrArgIds`)
      cases hhb : (f.blockAt dst).isLoopHeader
      · exact False.elim (hhdr hhb)
      · rfl
    rw [hhdrT] at hedge
    simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true, decide_eq_true_eq] at hedge
    obtain ⟨_, ⟨⟨⟨_, _⟩, hreach⟩, hsub⟩, ⟨⟨⟨⟨_, hpcDisj⟩, hpcond⟩, hlbsOk⟩, hargsOk⟩, hallV⟩ := hedge
    let dstBlk := f.blockAt dst
    let dstInv := tbl.getD dst {}
    let passedSR := bindSymBlockArgs outSR outSR dstBlk.args argVars
    let hdrArgIds := dstBlk.args.map BlockArg.varId
    -- Update `havoc'` at `hdrArgIds` to the concrete runtime integers passed along this edge:
    let havoc' : Nat → Nat := fun k =>
      if hdrArgIds.contains k then
        Val.asInt ((bindBlockArgs outRegs outRegs dstBlk.args argVars) k)
      else havoc k
    have hagree : ∀ k, k ∉ hdrArgIds → havoc' k = havoc k := by -- Outer loop variables keep their previous `havoc` values
      intro k hk
      have hc : hdrArgIds.contains k = false := by
        cases hck : hdrArgIds.contains k
        · rfl
        · exact False.elim (hk (List.contains_iff_mem.mp hck))
      dsimp only [havoc']; rw [hc]; simp
    refine ⟨hreach, havoc', ?_, phaseSpec_subsumes_sound hsub hps, ?_⟩
    · apply symBool_conj_true                             -- Goal 2a: Both `pathCond` and inductive `loopBounds` hold under `havoc'`
      · rw [symBool_eval_congr hagree dstInv.pathCond hpcDisj]
        exact symImpliesVal_sound hecond dstInv.pathCond true hpcond
      · apply loopBoundsCond_eval_true                    -- Prove each ring counter `vid` stays `< lim` across this edge
        intro ⟨vid, w, lim⟩ hmemLb
        have hlb := List.all_eq_true.mp hlbsOk (vid, w, lim) hmemLb
        simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hlb
        rcases hlb with ⟨⟨⟨hcontains, _⟩, _⟩, hpassLb⟩
        cases hpsv : passedSR.getVar vid <;> rw [hpsv] at hpassLb
        all_goals try contradiction
        rename_i eSym
        dsimp only at hpassLb
        rcases Bool.and_eq_true_iff.mp hpassLb with ⟨hfitsW, hbndE⟩
        have hbVal := hpassed vid
        rw [hpsv, SymVal.matchesVal] at hbVal
        dsimp only [havoc', dstBlk]
        rw [hcontains, if_pos rfl, hbVal]
        dsimp only [Val.asInt]
        rw [symIntFitsWidth_sound hvalidIn hfitsW]
        exact symIntBoundedUnder_sound hecond eSym lim hbndE
    · intro v                                             -- Goal 2b: Every register in `dstInv.regs` matches under `havoc'`
      by_cases hmem : hdrArgIds.contains v = true         -- Split: loop-carried `BlockArg` vs. outer invariant var
      · have hvMem : v ∈ hdrArgIds := List.contains_iff_mem.mp hmem -- Subcase: loop-carried `BlockArg` (`SymInt.havoc vid w`)
        obtain ⟨fa, hfaMem, hfaId⟩ := List.mem_map.mp hvMem
        have hpOk := List.all_eq_true.mp hargsOk fa hfaMem
        subst hfaId
        by_cases hw0 : (fa.width == 0) = true
        · rw [hw0] at hpOk; simp only [↓reduceIte, beq_iff_eq] at hpOk
          rw [hpOk]; trivial
        · rw [if_neg hw0] at hpOk; simp only [Bool.and_eq_true, beq_iff_eq] at hpOk
          rcases hpOk with ⟨hdstH, hfitsM⟩
          rw [hdstH]
          cases hpsv : passedSR.getVar fa.varId <;> rw [hpsv] at hfitsM
          all_goals try contradiction
          rename_i eSym
          have hbVal := hpassed fa.varId
          rw [hpsv, SymVal.matchesVal] at hbVal
          dsimp only at hfitsM
          dsimp only [SymVal.matchesVal, SymInt.eval, havoc', dstBlk]
          rw [hmem, if_pos rfl, hbVal]
          dsimp only [Val.asInt]
          rw [symIntFitsWidth_sound hvalidIn hfitsM]
      · have hmf : hdrArgIds.contains (0 + v) = false := by -- Subcase: outer variable unchanged across loop backedge
          cases hc : hdrArgIds.contains v <;> simp_all
        have hvOk := checkLoopRegsEquiv_getVar 0 dstInv.regs passedSR v hallV hmf
        have hmBase := symValEquivUnder_sound hecond _ _ _ hvOk.2 (hpassed v)
        exact symVal_matches_congr hagree _ _ hvOk.1 hmBase

/-- Core inductive soundness theorem: every reachable concrete state `(b, regs, phase)`
    of `f` satisfies the validated symbolic invariant `tbl.getD b {}` for some `havoc`. -/
theorem symReachable_sound {f : MirFunc} {tbl : List SymBlockInv}
    (hvalid : validSymFunc f tbl = true)
    {env : InputEnv} {t : ThreadCtx} (hvalidIn : f.validInput env)
    {b phase : Nat} {regs : RegFile}
    (hreach : Reachable f env t b regs phase) :
    b < f.blocks.length ∧
    (tbl.getD b {}).reachable = true ∧
    ∃ havoc : Nat → Nat,
      (tbl.getD b {}).effCond.eval env t havoc = true ∧
      (tbl.getD b {}).phaseSpec.matchesPhase phase = true ∧
      ∀ v, ((tbl.getD b {}).regs.getVar v).matchesVal env t havoc (regs v) := by
  have hvf := hvalid                                      -- Unpack top-level `validSymFunc` entry & per-block checks
  unfold validSymFunc at hvf
  dsimp only at hvf
  simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hvf
  obtain ⟨⟨hpos, _⟩, ⟨⟨⟨⟨h0r, h0c⟩, h0lbs⟩, h0p⟩, h0regs⟩, hallB⟩ := hvf
  induction hreach with                                   -- Induction on concrete `Reachable` execution steps
  | entry _ =>                                            -- Base case: kernel entry at `block 0` (`phase = 0`)
      refine ⟨hpos, h0r, fun _ => 0, ?_, ?_, ?_⟩
      · unfold SymBlockInv.effCond; rw [h0c, h0lbs]; rfl
      · rw [h0p]; rfl
      · intro v; rw [h0regs]; exact initialSymRegs_correct f env t (fun _ => 0) v
  | @stepGoto b0 dst p0 outP r0 outR evs tevs args _ hexec hterm hdst ih => -- Inductive step: unconditional `goto dst`
      obtain ⟨hb0, hr0, havoc, hpc0, hps0, hm0⟩ := ih     --   Apply IH at source block `b0`
      have hbOk := List.all_eq_true.mp hallB b0 (List.mem_range.mpr hb0)
      rw [hr0] at hbOk; simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true] at hbOk
      obtain ⟨hmOut, hpsOut, _, _, _⟩ := symExecStmts_correct f env t havoc (tbl.getD b0 {}).effCond hpc0 (f.blockAt b0).term
        (f.blockAt b0).stmts r0 (tbl.getD b0 {}).regs p0 (tbl.getD b0 {}).phaseSpec hm0 hps0 hbOk.1.1.1 hbOk.1.2
      rw [hexec] at hmOut hpsOut                          --   Step through `b0.stmts` via `symExecStmts_correct`
      rw [hterm] at hbOk hmOut hpsOut
      obtain ⟨hrDst, havoc', hpcDst, hpsDst, hmDst⟩ :=    --   Cross `(b0 -> dst)` via `validSymEdge_sound`
        validSymEdge_sound hvalidIn hbOk.2 hpc0 hmOut hpsOut
      exact ⟨hdst, hrDst, havoc', hpcDst, hpsDst, hmDst⟩
  | @stepCondTrue b0 tBlk fBlk cond p0 outP r0 outR evs tevs tArgs fArgs _ hexec hterm hcond hdst ih => -- Inductive step: `condBr` true arm
      obtain ⟨hb0, hr0, havoc, hpc0, hps0, hm0⟩ := ih
      have hbOk := List.all_eq_true.mp hallB b0 (List.mem_range.mpr hb0)
      rw [hr0] at hbOk; simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true] at hbOk
      obtain ⟨hmOut, hpsOut, _, _, _⟩ := symExecStmts_correct f env t havoc (tbl.getD b0 {}).effCond hpc0 (f.blockAt b0).term
        (f.blockAt b0).stmts r0 (tbl.getD b0 {}).regs p0 (tbl.getD b0 {}).phaseSpec hm0 hps0 hbOk.1.1.1 hbOk.1.2
      rw [hexec] at hmOut hpsOut
      rw [hterm] at hbOk hmOut hpsOut
      rcases hbOk with ⟨_, hbrOk⟩
      dsimp only at hbrOk
      have hcv := hmOut cond; dsimp only at hcv
      cases hget : (symExecStmts f (tbl.getD b0 {}).effCond (tbl.getD b0 {}).regs (tbl.getD b0 {}).phaseSpec (.condBr cond tBlk tArgs fBlk fArgs) (f.blockAt b0).stmts).1.getVar cond with
      | bool cSym =>                                      --   Tracked boolean guard: strengthen `effCond` with `cSym = true`
          rw [hget] at hbrOk hcv                          --   Substitute `getVar cond = .bool cSym` into edge check & match
          rw [SymVal.matchesVal] at hcv                   --   Unpack `outR cond = .bool (cSym.eval env t havoc)`
          rw [hcv] at hcond                               --   Rewrite `outR cond ≠ .bool false` using `cSym.eval`
          have hcTrue : cSym.eval env t havoc = true := by --   A tracked boolean other than `false` evaluates to `true`
            simpa using hcond
          simp only [Bool.and_eq_true] at hbrOk           --   Split `validSymEdge` conjunction across both outgoing arms
          have hec := symBool_conj_true hpc0 hcTrue       --   Combine entry path condition `hpc0` with `cSym = true`
          obtain ⟨hrDst, havoc', hpcDst, hpsDst, hmDst⟩ :=
            validSymEdge_sound hvalidIn hbrOk.1 hec hmOut hpsOut
          exact ⟨hdst, hrDst, havoc', hpcDst, hpsDst, hmDst⟩
      | unk =>                                            --   Untracked float/runtime guard: preserve `effCond` conservatively
          rw [hget] at hbrOk
          simp only [Bool.and_eq_true] at hbrOk
          obtain ⟨hrDst, havoc', hpcDst, hpsDst, hmDst⟩ :=
            validSymEdge_sound hvalidIn hbrOk.1 hpc0 hmOut hpsOut
          exact ⟨hdst, hrDst, havoc', hpcDst, hpsDst, hmDst⟩
      | int _ | ptr _ _ _ | condPtr _ _ _ _ | smemOff _ _ | smemDesc _ _ | tmemPtr _ _ =>
          rw [hget] at hbrOk; contradiction
  | @stepCondFalse b0 tBlk fBlk cond p0 outP r0 outR evs tevs tArgs fArgs _ hexec hterm hcond hdst ih => -- Inductive step: `condBr` false arm
      obtain ⟨hb0, hr0, havoc, hpc0, hps0, hm0⟩ := ih
      have hbOk := List.all_eq_true.mp hallB b0 (List.mem_range.mpr hb0)
      rw [hr0] at hbOk; simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true] at hbOk
      obtain ⟨hmOut, hpsOut, _, _, _⟩ := symExecStmts_correct f env t havoc (tbl.getD b0 {}).effCond hpc0 (f.blockAt b0).term
        (f.blockAt b0).stmts r0 (tbl.getD b0 {}).regs p0 (tbl.getD b0 {}).phaseSpec hm0 hps0 hbOk.1.1.1 hbOk.1.2
      rw [hexec] at hmOut hpsOut
      rw [hterm] at hbOk hmOut hpsOut
      rcases hbOk with ⟨_, hbrOk⟩
      dsimp only at hbrOk
      have hcv := hmOut cond; dsimp only at hcv
      cases hget : (symExecStmts f (tbl.getD b0 {}).effCond (tbl.getD b0 {}).regs (tbl.getD b0 {}).phaseSpec (.condBr cond tBlk tArgs fBlk fArgs) (f.blockAt b0).stmts).1.getVar cond with
      | bool cSym =>                                      --   Tracked boolean guard: strengthen `effCond` with `!cSym = true`
          rw [hget] at hbrOk hcv                          --   Substitute `getVar cond = .bool cSym` into edge check & match
          rw [SymVal.matchesVal] at hcv                   --   Unpack `outR cond = .bool (cSym.eval env t havoc)`
          rw [hcv] at hcond                               --   Rewrite `outR cond ≠ .bool true` using `cSym.eval`
          have hcFalse : cSym.eval env t havoc = false := by -- A tracked boolean other than `true` evaluates to `false`
            simpa using hcond
          simp only [Bool.and_eq_true] at hbrOk           --   Split `validSymEdge` conjunction across both outgoing arms
          have hnotTrue : (SymBool.not cSym).eval env t havoc = true := by -- Negated guard `!cSym` evaluates to `true`
            simp [SymBool.eval, hcFalse]
          have hec := symBool_conj_true hpc0 hnotTrue     --   Combine entry path condition `hpc0` with `!cSym = true`
          obtain ⟨hrDst, havoc', hpcDst, hpsDst, hmDst⟩ :=
            validSymEdge_sound hvalidIn hbrOk.2 hec hmOut hpsOut
          exact ⟨hdst, hrDst, havoc', hpcDst, hpsDst, hmDst⟩
      | unk =>                                            --   Untracked float/runtime guard: preserve `effCond` conservatively
          rw [hget] at hbrOk
          simp only [Bool.and_eq_true] at hbrOk
          obtain ⟨hrDst, havoc', hpcDst, hpsDst, hmDst⟩ :=
            validSymEdge_sound hvalidIn hbrOk.2 hpc0 hmOut hpsOut
          exact ⟨hdst, hrDst, havoc', hpcDst, hpsDst, hmDst⟩
      | int _ | ptr _ _ _ | condPtr _ _ _ _ | smemOff _ _ | smemDesc _ _ | tmemPtr _ _ =>
          rw [hget] at hbrOk; contradiction

/-- Collect all `SymMemEvent`s emitted across reachable blocks in `tbl`. -/
def collectSymMemEvents (f : MirFunc) (tbl : List SymBlockInv) : List SymMemEvent :=
  (List.range f.blocks.length).flatMap fun b =>
    let inv := tbl.getD b {}
    if !inv.reachable then []
    else (symExecStmts f inv.effCond inv.regs inv.phaseSpec (f.blockAt b).term (f.blockAt b).stmts).2.2.1

/-- Collect all `SymAssertEvent`s emitted across reachable blocks in `tbl`. -/
def collectSymAssertEvents (f : MirFunc) (tbl : List SymBlockInv) : List SymAssertEvent :=
  (List.range f.blocks.length).flatMap fun b =>
    let inv := tbl.getD b {}
    if !inv.reachable then []
    else (symExecStmts f inv.effCond inv.regs inv.phaseSpec (f.blockAt b).term (f.blockAt b).stmts).2.2.2.1

/-- Collect all `SymTmemEvent`s emitted across reachable blocks in `tbl`. -/
def collectSymTmemEvents (f : MirFunc) (tbl : List SymBlockInv) : List SymTmemEvent :=
  (List.range f.blocks.length).flatMap fun b =>
    let inv := tbl.getD b {}
    if !inv.reachable then []
    else (symExecStmts f inv.effCond inv.regs inv.phaseSpec (f.blockAt b).term (f.blockAt b).stmts).2.2.2.2

/-- Lemma: every concrete `EmitsEvent` corresponds to a `SymMemEvent` in `collectSymMemEvents`. -/
theorem emitsEvent_mem_collect {f : MirFunc} {tbl : List SymBlockInv}
    (hvalid : validSymFunc f tbl = true)
    {env : InputEnv} {t : ThreadCtx} (hvalidIn : f.validInput env)
    {ev : MemEvent} (hemits : EmitsEvent f env t ev) :
    ∃ havoc : Nat → Nat, ∃ sev ∈ collectSymMemEvents f tbl, ∃ idxSym : SymInt, ∃ k : Nat,
      k < sev.span ∧ 0 < sev.elemDiv ∧
      sev.isWrite = ev.isWrite ∧ sev.space = ev.space ∧ sev.bufId = ev.bufId ∧
      evalSyncBar env t havoc sev.syncBar = ev.syncBar ∧ sev.indexOpt = some idxSym ∧
      (idxSym.eval env t havoc + k) / sev.elemDiv = ev.index ∧
      sev.pathCond.eval env t havoc = true ∧ sev.phaseSpec.matchesPhase ev.phase = true := by
  obtain ⟨b, phase, regs, hreach, hevMem⟩ := hemits
  obtain ⟨hb, hr, havoc, hpc, hps, hm⟩ := symReachable_sound hvalid hvalidIn hreach
  have hvf := hvalid
  unfold validSymFunc at hvf; dsimp only at hvf
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hvf
  obtain ⟨_, _, hallB⟩ := hvf
  have hbOk := List.all_eq_true.mp hallB b (List.mem_range.mpr hb)
  rw [hr] at hbOk; simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true] at hbOk
  obtain ⟨_, _, hmE, _, _⟩ := symExecStmts_correct f env t havoc (tbl.getD b {}).effCond hpc (f.blockAt b).term
    (f.blockAt b).stmts regs (tbl.getD b {}).regs phase (tbl.getD b {}).phaseSpec hm hps hbOk.1.1.1 hbOk.1.2
  obtain ⟨sev, hsev, idxS, k, hks, hdiv, hw, hsp, hbid, hab, hidxOpt, hidxEv, hpce, hpse⟩ := hmE ev hevMem
  refine ⟨havoc, sev, ?_, idxS, k, hks, hdiv, hw, hsp, hbid, hab, hidxOpt, hidxEv, hpce, hpse⟩
  unfold collectSymMemEvents
  rw [List.mem_flatMap]
  refine ⟨b, List.mem_range.mpr hb, ?_⟩
  dsimp only; rw [hr]; simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte]; exact hsev

/-- Lemma: every concrete `EmitsTmemEvent` corresponds to a `SymTmemEvent` in `collectSymTmemEvents`. -/
theorem emitsTmemEvent_mem_collect {f : MirFunc} {tbl : List SymBlockInv}
    (hvalid : validSymFunc f tbl = true)
    {env : InputEnv} {t : ThreadCtx} (hvalidIn : f.validInput env)
    {tev : TmemEvent} (hemits : EmitsTmemEvent f env t tev) :
    ∃ havoc : Nat → Nat, ∃ stev ∈ collectSymTmemEvents f tbl, ∃ relSym : SymInt,
      stev.allocCols = tev.allocCols ∧ stev.heightLanes = tev.heightLanes ∧
      stev.widthCols = tev.widthCols ∧ stev.warpRestricted = tev.warpRestricted ∧
      stev.relAddrOpt = some relSym ∧ relSym.eval env t havoc = tev.relAddr ∧
      stev.pathCond.eval env t havoc = true := by
  obtain ⟨b, phase, regs, hreach, htevMem⟩ := hemits
  obtain ⟨hb, hr, havoc, hpc, hps, hm⟩ := symReachable_sound hvalid hvalidIn hreach
  have hvf := hvalid
  unfold validSymFunc at hvf; dsimp only at hvf
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hvf
  obtain ⟨_, _, hallB⟩ := hvf
  have hbOk := List.all_eq_true.mp hallB b (List.mem_range.mpr hb)
  rw [hr] at hbOk; simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true] at hbOk
  obtain ⟨_, _, _, _, hmT⟩ := symExecStmts_correct f env t havoc (tbl.getD b {}).effCond hpc (f.blockAt b).term
    (f.blockAt b).stmts regs (tbl.getD b {}).regs phase (tbl.getD b {}).phaseSpec hm hps hbOk.1.1.1 hbOk.1.2
  obtain ⟨stev, hstev, relS, hac, hhl, hwc, hwr, hro, hre, hpce⟩ := hmT tev htevMem
  refine ⟨havoc, stev, ?_, relS, hac, hhl, hwc, hwr, hro, hre, ?_⟩
  · unfold collectSymTmemEvents
    rw [List.mem_flatMap]
    refine ⟨b, List.mem_range.mpr hb, ?_⟩
    dsimp only; rw [hr]; simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte]; exact hstev
  · rw [hpce]; exact hpc


end CudaOxide.Proofs.Program
