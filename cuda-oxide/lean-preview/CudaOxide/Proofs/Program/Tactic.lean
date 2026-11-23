/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide.Proofs.Program.Vc

/-!
# `mir_def_sym_table`, Witness Binders & `mir_vc` Tactic (`CudaOxide.Proofs.Program.Tactic`)
-/
namespace CudaOxide.Proofs.Program
open CudaOxide.Mir.Program
open CudaOxide.Contracts.Program

open Lean Elab Tactic Meta in
/-- Precomputes `symTable`, `symMemEvents`, `symAssertEvents`, `symTmemEvents`, and `symRankEvents`
    ONCE per module using a single native `evalExpr MirFunc` call. -/
elab "mir_def_sym_table " f:ident : command => do
  Command.liftTermElabM do                                -- Enter Lean's term-elaboration environment
    let ns ← getCurrNamespace                             -- Read current namespace (`CudaOxide.Generated`)
    let fExpr ← Term.elabTerm (← `($f:ident)) (some (mkConst ``MirFunc)) -- Resolve `mirFunc` identifier to an `Expr`
    let fVal ← unsafe evalExpr MirFunc (mkConst ``MirFunc) fExpr -- Evaluate `mirFunc` into a native compiled value
    let tblVal := inferSymTable fVal                      -- Run `inferSymTable` once in compiled native code
    let invTy := mkConst ``SymBlockInv
    let mut invConsts : Array Expr := #[]
    for h : i in [0 : tblVal.length] do                   -- Emit one small `symInv_i` constant per basic block
      let invName := ns ++ Name.mkSimple s!"symInv_{i}"   -- so `symTable.getD b` unfolds only block `b` in `whnf`
      addDecl <| Declaration.defnDecl {
        name := invName, levelParams := [], type := invTy, value := toExpr tblVal[i],
        hints := ReducibilityHints.abbrev, safety := DefinitionSafety.safe }
      invConsts := invConsts.push (mkConst invName)
    let tblTy ← Meta.mkAppM ``List #[invTy]
    let tblSpine ← Meta.mkListLit invTy invConsts.toList  -- Build `[symInv_0, symInv_1, ...]` list spine
    addDecl <| Declaration.defnDecl {                     -- Register `symTable` in the Lean environment
      name := ns ++ `symTable, levelParams := [], type := tblTy, value := tblSpine,
      hints := ReducibilityHints.abbrev, safety := DefinitionSafety.safe }
    let memTy ← Meta.mkAppM ``List #[mkConst ``SymMemEvent]
    let memVal := collectSymMemEvents fVal tblVal         -- Pre-collect global and shared memory access events
    addAndCompile <| Declaration.defnDecl {
      name := ns ++ `symMemEvents, levelParams := [], type := memTy, value := toExpr memVal,
      hints := ReducibilityHints.abbrev, safety := DefinitionSafety.safe }
    let assTy ← Meta.mkAppM ``List #[mkConst ``SymAssertEvent]
    let assVal := collectSymAssertEvents fVal tblVal      -- Pre-collect `mir.assert` candidate events
    addAndCompile <| Declaration.defnDecl {
      name := ns ++ `symAssertEvents, levelParams := [], type := assTy, value := toExpr assVal,
      hints := ReducibilityHints.abbrev, safety := DefinitionSafety.safe }
    let tmemTy ← Meta.mkAppM ``List #[mkConst ``SymTmemEvent]
    let tmemVal := collectSymTmemEvents fVal tblVal       -- Pre-collect Blackwell `TMEM` access events
    addAndCompile <| Declaration.defnDecl {
      name := ns ++ `symTmemEvents, levelParams := [], type := tmemTy, value := toExpr tmemVal,
      hints := ReducibilityHints.abbrev, safety := DefinitionSafety.safe }
    let rankTy ← Meta.mkAppM ``List #[mkConst ``SymRankEvent]
    let rankVal := collectSymRankEvents fVal tblVal       -- Pre-collect 2-CTA `map_shared_rank` peer-rank events
    addAndCompile <| Declaration.defnDecl {
      name := ns ++ `symRankEvents, levelParams := [], type := rankTy, value := toExpr rankVal,
      hints := ReducibilityHints.abbrev, safety := DefinitionSafety.safe }

open Lean Elab Tactic Meta in
/-- Generic `MetaM` witness binder: evaluates closed `lhs` to native value of type `α`,
    unifies `rhs` with `toExpr v`, and closes `lhs = rhs` in the kernel via `rfl`. -/
def mirBindGoal (α : Type) [ToExpr α] : TacticM Unit := withMainContext do
  let tgt ← instantiateMVars (← getMainTarget)            -- Read the current proof goal (`lhs = ?rhs`)
  let some (ty, lhs, rhs) := tgt.eq? | throwError "mir_bind: expected equality goal"
  let v ← unsafe evalExpr α ty lhs                        -- Compute the concrete list/number `v` in compiled code
  let _ ← isDefEq rhs (toExpr v)                          -- Fill the `?rhs` placeholder with `v`
  evalTactic (← `(tactic| rfl))                           -- Have Lean's trusted kernel verify `lhs = v` by `rfl`

elab "mir_bind_nat"          : tactic => mirBindGoal Nat
elab "mir_bind_events"       : tactic => mirBindGoal (List SymMemEvent)
elab "mir_bind_pairs"        : tactic => mirBindGoal (List (SymMemEvent × SymMemEvent))
elab "mir_bind_shared_pairs" : tactic => mirBindGoal (List (SymMemEvent × SymMemEvent × Option (SymInt × SymInt)))
elab "mir_bind_asserts"      : tactic => mirBindGoal (List SymAssertEvent)
elab "mir_bind_tmem"         : tactic => mirBindGoal (List SymTmemEvent)
elab "mir_bind_ranks"        : tactic => mirBindGoal (List SymRankEvent)

theorem pow2_8       : (2 : Nat) ^ 8        = 256 := rfl
theorem pow2_16      : (2 : Nat) ^ 16       = 65536 := rfl
theorem pow2_32      : (2 : Nat) ^ 32       = 4294967296 := rfl
theorem pow2_64      : (2 : Nat) ^ 64       = 18446744073709551616 := rfl
theorem pow2_succ_8  : (2 : Nat) ^ (7 + 1)  = 256 := rfl
theorem pow2_succ_16 : (2 : Nat) ^ (15 + 1) = 65536 := rfl
theorem pow2_succ_32 : (2 : Nat) ^ (31 + 1) = 4294967296 := rfl
theorem pow2_succ_64 : (2 : Nat) ^ (63 + 1) = 18446744073709551616 := rfl

/-- Power-of-two bitmask identities (`x &&& (2^k - 1) = x % 2^k`) unfolded in `O(1)` by `mir_vc`. -/
theorem land_mask_1          (a : Nat) : a &&& 1          = a % 2          := Nat.and_two_pow_sub_one_eq_mod a 1
theorem land_mask_3          (a : Nat) : a &&& 3          = a % 4          := Nat.and_two_pow_sub_one_eq_mod a 2
theorem land_mask_7          (a : Nat) : a &&& 7          = a % 8          := Nat.and_two_pow_sub_one_eq_mod a 3
theorem land_mask_15         (a : Nat) : a &&& 15         = a % 16         := Nat.and_two_pow_sub_one_eq_mod a 4
theorem land_mask_31         (a : Nat) : a &&& 31         = a % 32         := Nat.and_two_pow_sub_one_eq_mod a 5
theorem land_mask_63         (a : Nat) : a &&& 63         = a % 64         := Nat.and_two_pow_sub_one_eq_mod a 6
theorem land_mask_127        (a : Nat) : a &&& 127        = a % 128        := Nat.and_two_pow_sub_one_eq_mod a 7
theorem land_mask_255        (a : Nat) : a &&& 255        = a % 256        := Nat.and_two_pow_sub_one_eq_mod a 8
theorem land_mask_511        (a : Nat) : a &&& 511        = a % 512        := Nat.and_two_pow_sub_one_eq_mod a 9
theorem land_mask_1023       (a : Nat) : a &&& 1023       = a % 1024       := Nat.and_two_pow_sub_one_eq_mod a 10
theorem land_mask_4095       (a : Nat) : a &&& 4095       = a % 4096       := Nat.and_two_pow_sub_one_eq_mod a 12
theorem land_mask_16383      (a : Nat) : a &&& 16383      = a % 16384      := Nat.and_two_pow_sub_one_eq_mod a 14
theorem land_mask_65535      (a : Nat) : a &&& 65535      = a % 65536      := Nat.and_two_pow_sub_one_eq_mod a 16
theorem land_mask_1073741823 (a : Nat) : a &&& 1073741823 = a % 1073741824 := Nat.and_two_pow_sub_one_eq_mod a 30
theorem land_mask_4294967295 (a : Nat) : a &&& 4294967295 = a % 4294967296 := Nat.and_two_pow_sub_one_eq_mod a 32
theorem land_mask_1_rev      (a : Nat) : 1          &&& a = a % 2          := by rw [Nat.and_comm, land_mask_1]
theorem land_mask_3_rev      (a : Nat) : 3          &&& a = a % 4          := by rw [Nat.and_comm, land_mask_3]
theorem land_mask_7_rev      (a : Nat) : 7          &&& a = a % 8          := by rw [Nat.and_comm, land_mask_7]
theorem land_mask_15_rev     (a : Nat) : 15         &&& a = a % 16         := by rw [Nat.and_comm, land_mask_15]
theorem land_mask_31_rev     (a : Nat) : 31         &&& a = a % 32         := by rw [Nat.and_comm, land_mask_31]
theorem land_mask_63_rev     (a : Nat) : 63         &&& a = a % 64         := by rw [Nat.and_comm, land_mask_63]
theorem land_mask_127_rev    (a : Nat) : 127        &&& a = a % 128        := by rw [Nat.and_comm, land_mask_127]
theorem land_mask_255_rev    (a : Nat) : 255        &&& a = a % 256        := by rw [Nat.and_comm, land_mask_255]
theorem land_mask_65535_rev  (a : Nat) : 65535      &&& a = a % 65536      := by rw [Nat.and_comm, land_mask_65535]

/-- Disjoint bitwise-OR identity: when `a` is a multiple of `2^k` (low `k` bits zero) and
    `b < 2^k` (bits `≥ k` zero), `a ||| b = a + b`. Used by `mir_mul_mono` on bit-split indices. -/
theorem or_eq_add_of_mod_two_pow {a b : Nat} (k : Nat) (ha : a % (2 ^ k) = 0) (hb : b < 2 ^ k) :
    a ||| b = a + b := by
  let q := a / 2 ^ k                                      -- Factor `a` as `q * 2^k` using `a % 2^k = 0`
  have ha1 : a = 2 ^ k * q := by
    have hd := Nat.div_add_mod a (2 ^ k)                  -- Standard Euclidean identity `2^k * (a / 2^k) + a % 2^k = a`
    rw [ha, Nat.add_zero] at hd                           -- Remainder `a % 2^k` is `0`
    exact hd.symm
  have ha2 : a = q * 2 ^ k := by rw [ha1, Nat.mul_comm]
  apply Nat.eq_of_testBit_eq                              -- Prove `a ||| b = a + b` bit-by-bit across all bit indices `j`
  intro j
  have lhs_eq : (a ||| b).testBit j = ((q * 2 ^ k).testBit j || b.testBit j) := by
    rw [← ha2, Nat.testBit_or]                            -- Bit `j` of `a ||| b` is the boolean OR of bit `j` of each
  have rhs_eq : (a + b).testBit j = if j < k then b.testBit j else q.testBit (j - k) := by
    rw [ha1, Nat.testBit_two_pow_mul_add q hb j]          -- Adding `b < 2^k` to `2^k * q` never carries across bit `k`
  rw [lhs_eq, rhs_eq, Nat.testBit_mul_two_pow]
  by_cases hjk : j < k                                    -- Split: low bit `j < k` vs. high bit `k ≤ j`
  · have hnk : ¬(k ≤ j) := by omega                       -- Low bit `j < k`: `(q * 2^k).testBit j` is `false`
    simp [hjk, hnk]
  · have hkj : k ≤ j := by omega                          -- High bit `k ≤ j`: `b.testBit j` is `false` since `b < 2^k ≤ 2^j`
    have hbj : b.testBit j = false :=
      Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le hb (Nat.pow_le_pow_right (by omega) hkj))
    simp [hjk, hkj, hbj]

/-- Symmetric disjoint bitwise-OR identity when `a < 2^k` and `b % 2^k = 0`. -/
theorem or_eq_add_of_lt_two_pow {a b : Nat} (k : Nat) (ha : a < 2 ^ k) (hb : b % (2 ^ k) = 0) :
    a ||| b = a + b := by
  rw [Nat.or_comm, or_eq_add_of_mod_two_pow k hb ha, Nat.add_comm] -- Commute operands and reuse `or_eq_add_of_mod_two_pow`

/-- Tight upper bound `(2^32 - 1)^2` on the product of any two 32-bit quantities, leaving
    `≥ 2^32` headroom below `2^64` so `a * b + c < 2^64` holds for every 32-bit offset `c`. -/
theorem mul_lt_u64_of_u32 {a b : Nat} (ha : a < 4294967296) (hb : b < 4294967296) :
    a * b ≤ 18446744065119617025 := by
  have ha' : a ≤ 4294967295 := by omega                   -- Convert strict `< 2^32` to `≤ 2^32 - 1`
  have hb' : b ≤ 4294967295 := by omega                   -- Same for `b`
  exact Nat.mul_le_mul ha' hb'                            -- Multiply both upper bounds via `Nat.mul_le_mul`

open Lean Elab Tactic Meta in
/-- Injects `Nat.mul_le_mul_right` / `Nat.mul_le_mul_left` monotonicity facts, 32x32->64
    non-overflow bounds (`mul_lt_u64_of_u32`), `Nat.and_le_left`/`right` bounds on residual
    `&&&` terms, and disjoint `a ||| b = a + b` identities (`or_eq_add_of_mod_two_pow`) for `omega`. -/
elab "mir_mul_mono" : tactic => withMainContext do
  let mulRef ← IO.mkRef (#[] : Array (Expr × Expr))       -- Mutable buckets collecting nonlinear subterms (`*`, `&`, `|`, `^`)
  let andRef ← IO.mkRef (#[] : Array (Expr × Expr))       -- because `omega` only reasons over linear `+` and constant `*`
  let orRef  ← IO.mkRef (#[] : Array (Expr × Expr))
  let xorRef ← IO.mkRef (#[] : Array (Expr × Expr))
  let pushUniq (ref : IO.Ref (Array (Expr × Expr))) (a b : Expr) : IO Unit :=
    ref.modify fun acc =>
      if acc.any (fun (x, y) => x == a && y == b) then acc else acc.push (a, b)
  let collect (e : Expr) : MetaM Unit := do               -- Walk an expression tree recording unique `*`, `&`, `|`, `^` apps
    e.forEach fun sub => do
      if sub.isAppOfArity ``HMul.hMul 6 then
        let args := sub.getAppArgs
        pushUniq mulRef args[4]! args[5]!
      else if sub.isAppOfArity ``HAnd.hAnd 6 then
        let args := sub.getAppArgs
        pushUniq andRef args[4]! args[5]!
      else if sub.isAppOfArity ``HOr.hOr 6 then
        let args := sub.getAppArgs
        pushUniq orRef args[4]! args[5]!
      else if sub.isAppOfArity ``HXor.hXor 6 then
        let args := sub.getAppArgs
        pushUniq xorRef args[4]! args[5]!
  collect (← instantiateMVars (← getMainTarget))          -- Scan the target proposition
  for decl in (← getLCtx) do                              -- Scan all hypotheses in scope
    if !decl.isImplementationDetail then
      collect (← instantiateMVars decl.type)
  let mulTerms ← mulRef.get
  let andTerms ← andRef.get
  let orTerms  ← orRef.get
  let xorTerms ← xorRef.get
  let proveByOmega (prop : Expr) : TacticM (Option Expr) := do -- Helper: speculatively try proving `prop` with `omega`
    try
      let sg ← mkFreshExprMVar prop
      let saved ← getGoals
      setGoals [sg.mvarId!]
      evalTactic (← `(tactic| omega))
      setGoals saved
      pure (some sg)
    catch _ => pure none
  let mut k := 0
  for i in [:andTerms.size] do                            -- Pass 1 (`a &&& b`): add `a &&& b ≤ a` and `a &&& b ≤ b`
    let (a, b) := andTerms[i]!
    k := k + 1
    let pfL ← mkAppOptM ``Nat.and_le_left #[some a, some b]
    let g1 ← (← getMainGoal).assert (Name.mkSimple s!"handL_{k}") (← inferType pfL) pfL
    let (_, g1') ← g1.intro1P
    let pfR ← mkAppOptM ``Nat.and_le_right #[some a, some b]
    let g2 ← g1'.assert (Name.mkSimple s!"handR_{k}") (← inferType pfR) pfR
    let (_, g2') ← g2.intro1P
    replaceMainGoal [g2']
  let zeroNat := toExpr (0 : Nat)
  let proveReduced (pr : Expr) : TacticM (Option Expr) := do
    try
      let sg ← mkFreshExprMVar pr
      let saved ← getGoals
      setGoals [sg.mvarId!]
      evalTactic (← `(tactic| (simp only [Nat.reducePow]; omega)))
      setGoals saved
      pure (some sg)
    catch _ => pure none
  for i in [:orTerms.size] do                             -- Pass 2 (`a ||| b`): add `a ||| b = a + b` when bits are disjoint
    let (a, b) := orTerms[i]!
    let mut matched := false
    for shift in ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20, 24, 30, 32] : List Nat) do
      if !matched then
        let powVal : Nat := 2 ^ shift
        let powExpr := toExpr powVal
        let shiftExpr := toExpr shift
        let modA ← mkAppM ``HMod.hMod #[a, powExpr]
        if (← proveByOmega (← mkAppM ``Eq #[modA, zeroNat])).isSome then
          if (← proveByOmega (← mkAppM ``LT.lt #[b, powExpr])).isSome then
            let pModGoal ← mkAppM ``Eq #[← mkAppM ``HMod.hMod #[a, ← mkAppM ``HPow.hPow #[toExpr (2 : Nat), shiftExpr]], zeroNat]
            let pLtGoal  ← mkAppM ``LT.lt #[b, ← mkAppM ``HPow.hPow #[toExpr (2 : Nat), shiftExpr]]
            if let some h1 ← proveReduced pModGoal then
              if let some h2 ← proveReduced pLtGoal then
                k := k + 1
                let pf ← mkAppOptM ``or_eq_add_of_mod_two_pow #[some a, some b, some shiftExpr, some h1, some h2]
                let g' ← (← getMainGoal).assert (Name.mkSimple s!"hor_{k}") (← inferType pf) pf
                let (_, g'') ← g'.intro1P
                replaceMainGoal [g'']
                matched := true
        if !matched then
          let modB ← mkAppM ``HMod.hMod #[b, powExpr]
          if (← proveByOmega (← mkAppM ``LT.lt #[a, powExpr])).isSome then
            if (← proveByOmega (← mkAppM ``Eq #[modB, zeroNat])).isSome then
              let pLtGoal  ← mkAppM ``LT.lt #[a, ← mkAppM ``HPow.hPow #[toExpr (2 : Nat), shiftExpr]]
              let pModGoal ← mkAppM ``Eq #[← mkAppM ``HMod.hMod #[b, ← mkAppM ``HPow.hPow #[toExpr (2 : Nat), shiftExpr]], zeroNat]
              if let some h1 ← proveReduced pLtGoal then
                if let some h2 ← proveReduced pModGoal then
                  k := k + 1
                  let pf ← mkAppOptM ``or_eq_add_of_lt_two_pow #[some a, some b, some shiftExpr, some h1, some h2]
                  let g' ← (← getMainGoal).assert (Name.mkSimple s!"hor_{k}") (← inferType pf) pf
                  let (_, g'') ← g'.intro1P
                  replaceMainGoal [g'']
                  matched := true
  for i in [:xorTerms.size] do                            -- Pass 3 (`a ^^^ b`): add `a ^^^ b < 2^k` when `a, b < 2^k` (e.g. `rank ^ 1 < 2`)
    let (a, b) := xorTerms[i]!
    for shift in ([1, 2, 3, 4, 8, 16, 32] : List Nat) do
      let powVal : Nat := 2 ^ shift
      let powExpr := toExpr powVal
      let shiftExpr := toExpr shift
      if (← proveByOmega (← mkAppM ``LT.lt #[a, powExpr])).isSome then
        if (← proveByOmega (← mkAppM ``LT.lt #[b, powExpr])).isSome then
          let pA ← mkAppM ``LT.lt #[a, ← mkAppM ``HPow.hPow #[toExpr (2 : Nat), shiftExpr]]
          let pB ← mkAppM ``LT.lt #[b, ← mkAppM ``HPow.hPow #[toExpr (2 : Nat), shiftExpr]]
          if let some h1 ← proveReduced pA then
            if let some h2 ← proveReduced pB then
              k := k + 1
              let pf ← mkAppM ``Nat.xor_lt_two_pow #[h1, h2]
              let g' ← (← getMainGoal).assert (Name.mkSimple s!"hxor_{k}") (← inferType pf) pf
              let (_, g'') ← g'.intro1P
              replaceMainGoal [g'']
  let u32Bound := toExpr (4294967296 : Nat)
  for i in [:mulTerms.size] do                            -- Pass 4 (`a * b`): add 32x32->64 bounds & `a1 ≤ a2 -> a1 * b ≤ a2 * b`
    let (a1, b1) := mulTerms[i]!
    if let some pa ← proveByOmega (← mkAppM ``LT.lt #[a1, u32Bound]) then
      if let some pb ← proveByOmega (← mkAppM ``LT.lt #[b1, u32Bound]) then
        k := k + 1
        let pf ← mkAppM ``mul_lt_u64_of_u32 #[pa, pb]
        let pfTy ← inferType pf
        let g' ← (← getMainGoal).assert (Name.mkSimple s!"hmul64_{k}") pfTy pf
        let (_, g'') ← g'.intro1P
        replaceMainGoal [g'']
    for j in [:mulTerms.size] do
      if i != j then
        let (a2, b2) := mulTerms[j]!
        if (← isDefEq b1 b2) && !(← isDefEq a1 a2) then
          if let some ple ← proveByOmega (← mkAppM ``LE.le #[a1, a2]) then
            k := k + 1
            let pf ← mkAppM ``Nat.mul_le_mul_right #[b1, ple]
            let pfTy ← inferType pf
            let g' ← (← getMainGoal).assert (Name.mkSimple s!"hmul_{k}") pfTy pf
            let (_, g'') ← g'.intro1P
            replaceMainGoal [g'']
        if (← isDefEq a1 a2) && !(← isDefEq b1 b2) then
          if let some ple ← proveByOmega (← mkAppM ``LE.le #[b1, b2]) then
            k := k + 1
            let pf ← mkAppM ``Nat.mul_le_mul_left #[a1, ple]
            let pfTy ← inferType pf
            let g' ← (← getMainGoal).assert (Name.mkSimple s!"hmul_{k}") pfTy pf
            let (_, g'') ← g'.intro1P
            replaceMainGoal [g'']

/-- Classical double-negation elimination as an unconditional equality for `simp only` (`0` typeclass search). -/
theorem prop_not_not (p : Prop) : (¬¬p) = p := propext Classical.not_not

open Lean Elab Tactic Meta in
/-- Recursively discharges structured VC goals (`∧`, `→`, `ite` splits via native `MVarId.byCases`,
    and `omega` / `mir_mul_mono`) with bounded `iteFuel` (`≤ 16`) and zero `delab` fragility. -/
partial def solveVcGoals (fuel iteFuel : Nat) : TacticM Unit := do
  if fuel == 0 then return
  if (← getGoals).isEmpty then return
  if (← try evalTactic (← `(tactic| trivial)); pure true catch _ => pure false) then return
  let tgt ← withMainContext do whnfR (← instantiateMVars (← getMainTarget))
  if tgt.isAppOfArity ``And 2 then                        -- Step 1: Split `P ∧ Q` goals first so each event is solved in isolation
    evalTactic (← `(tactic| constructor))
    let gs ← getGoals
    for g in gs do
      setGoals [g]
      solveVcGoals (fuel - 1) iteFuel
    setGoals []
    return
  if tgt.isForall then                                    -- Step 2: Introduce `∀` variables and `→` premises into context
    let (_, g') ← (← getMainGoal).intro (Name.mkSimple s!"hvc_{fuel}")
    replaceMainGoal [g']
    solveVcGoals (fuel - 1) iteFuel
    return
  let mut gRen ← getMainGoal                              -- Step 3: Normalize hypothesis names so generated proofs never collide
  let mut rIdx := 0
  for d in (← gRen.getDecl).lctx do
    let nm := d.userName.toString
    if !d.isImplementationDetail && !(nm.startsWith "hvc_") && !(nm.startsWith "hacc_") && !(nm.startsWith "hite_") then
      rIdx := rIdx + 1
      gRen ← gRen.rename d.fvarId (Name.mkSimple s!"hacc_{fuel}_{rIdx}")
  replaceMainGoal [gRen]
  try evalTactic (← `(tactic|                             -- Step 4: Unfold `SymInt.eval` / `SymBool.holds` into plain `Nat` math
    simp only [CudaOxide.Mir.Program.paramBoundsProp, SymInt.eval, SymBool.holds,
               SymBool.normDisc, extractIteOneCond, SymBool.holdsRaw,
               SymBool.eval, symSpanInBounds, symRankInBounds, symPairRaceOblig, symSlotGuardOpt,
               evalCmpOp_eq_true, evalCmpOp_eq_false, evalCmpProp,
               CudaOxide.Mir.Program.evalBinOp, CudaOxide.Mir.Program.wordMod, Option.getD,
               ↓reduceIte, eq_self, ne_eq, Nat.not_lt_zero,
               Nat.zero_add, Nat.add_zero, Nat.reduceAdd, Nat.lt_irrefl,
               Nat.max_def, Nat.reduceLeDiff, Nat.reduceMod, Nat.reducePow,
               pow2_8, pow2_16, pow2_32, pow2_64,
               pow2_succ_8, pow2_succ_16, pow2_succ_32, pow2_succ_64,
               land_mask_1, land_mask_3, land_mask_7, land_mask_15, land_mask_31,
               land_mask_63, land_mask_127, land_mask_255, land_mask_511, land_mask_1023,
               land_mask_4095, land_mask_16383, land_mask_65535, land_mask_1073741823,
               land_mask_4294967295,
               land_mask_1_rev, land_mask_3_rev, land_mask_7_rev, land_mask_15_rev,
               land_mask_31_rev, land_mask_63_rev, land_mask_127_rev, land_mask_255_rev,
               land_mask_65535_rev,
               and_true, true_and, and_false, false_and,
               or_true, true_or, or_false, false_or, not_true_eq_false, not_false_eq_true,
               not_or, prop_not_not,
               List.mem_cons, List.not_mem_nil, forall_eq_or_imp, forall_eq,
               Bool.and_eq_true, Bool.or_eq_true, Bool.not_eq_true',
               Bool.and_eq_false_iff, Bool.or_eq_false_iff, Bool.not_eq_false',
               Bool.not_false, Bool.not_true, Bool.not_eq_false, Bool.not_eq_true,
               decide_false, decide_true, decide_eq_true_eq, decide_eq_true_iff,
               decide_eq_false_iff_not, beq_iff_eq, bne_iff_ne] at *))
  catch _ => pure ()
  if (← getGoals).isEmpty then return
  withMainContext do
    if (← try evalTactic (← `(tactic| omega)); pure true catch _ => pure false) then return -- Step 5: Fast-path linear `omega`
    for d in (← getLCtx) do                               -- Step 6: Split any `P ∧ Q` hypothesis into separate facts
      if !d.isImplementationDetail then
        let dTy ← whnfR (← instantiateMVars d.type)
        if dTy.isAppOfArity ``And 2 then
          let subs ← (← getMainGoal).cases d.fvarId
          for sub in subs do
            setGoals [sub.mvarId]
            solveVcGoals (fuel - 1) iteFuel
          setGoals []
          return
    if iteFuel > 0 then                                   -- Step 7: Case-split innermost `if cond then a else b` subterm
      let findInnerIte (root : Expr) : Option Expr :=
        let rec go (e : Expr) : Option Expr :=
          if e.isAppOfArity ``ite 5 then
            let args := e.getAppArgs
            (go args[1]!).orElse fun _ =>
            (go args[3]!).orElse fun _ =>
            (go args[4]!).orElse fun _ => some args[1]!
          else
            match e with
            | .app f a => (go f).orElse fun _ => go a
            | _ => none
        go root
      let mut iteCondOpt := findInnerIte (← instantiateMVars (← getMainTarget))
      if iteCondOpt.isNone then
        for decl in (← getLCtx) do
          if !decl.isImplementationDetail && iteCondOpt.isNone then
            iteCondOpt := findInnerIte (← instantiateMVars decl.type)
      if let some condExpr := iteCondOpt then
        let hName := Name.mkSimple s!"hite_{fuel}_{iteFuel}"
        let hIte := mkIdent hName
        let (sPos, sNeg) ← (← getMainGoal).byCases condExpr hName
        let simpExcept (thmStx : TSyntax `Lean.Parser.Tactic.simpLemma) : TacticM Unit := do
          if (← getGoals).isEmpty then return
          let decls ← withMainContext do pure (← getLCtx).decls.toList
          let mut locIds : Array Ident := #[]
          for dOpt in decls do
            if let some d := dOpt then
              if !d.isImplementationDetail && d.userName != hName then
                locIds := locIds.push (mkIdent d.userName)
          try
            if locIds.isEmpty then
              evalTactic (← `(tactic| simp only [$thmStx, ↓reduceIte, eq_self, Nat.not_lt_zero, and_false, false_and, or_false, false_or, and_true, true_and, or_true, true_or]))
            else
              evalTactic (← `(tactic| simp only [$thmStx, ↓reduceIte, eq_self, Nat.not_lt_zero, and_false, false_and, or_false, false_or, and_true, true_and, or_true, true_or] at $locIds* ⊢))
          catch _ => pure ()
        setGoals [sPos.mvarId]
        simpExcept (← `(Lean.Parser.Tactic.simpLemma| eq_true $hIte:ident))
        solveVcGoals (fuel - 1) (iteFuel - 1)
        setGoals [sNeg.mvarId]
        simpExcept (← `(Lean.Parser.Tactic.simpLemma| eq_false $hIte:ident))
        solveVcGoals (fuel - 1) (iteFuel - 1)
        setGoals []
        return
      for d in (← getLCtx) do                             -- Step 8: Case-split any `P ∨ Q` disjunction hypothesis
        if !d.isImplementationDetail then
          let dTy ← whnfR (← instantiateMVars d.type)
          if dTy.isAppOfArity ``Or 2 then
            let subs ← (← getMainGoal).cases d.fvarId
            for sub in subs do
              setGoals [sub.mvarId]
              solveVcGoals (fuel - 1) iteFuel
            setGoals []
            return
    if (← try evalTactic (← `(tactic| (mir_mul_mono; omega))); pure true catch _ => pure false) then return -- Step 9: Nonlinear bounds + `omega`
    evalTactic (← `(tactic| omega))                       -- Final step: run `omega` directly so any failure prints a clean counterexample

/-- Invoke `solveVcGoals` with structural fuel `512` and `ite` case-split fuel `16`. -/
elab "mir_solve_vc" : tactic => solveVcGoals 512 16

/-- Primary proof tactic for `MirFunc` arithmetic obligations (`no_oob_idx`, `disjoint_writes`,
    `no_data_race`, `MirAssertRedundant`). Unfolds `List.foldr` into per-event subgoals first
    (avoiding monolithic `simp` congruence proofs across large event-pair lists), then unfolds
    symbolic evaluators into `Nat` arithmetic and closes every leaf with `omega`. -/
macro "mir_vc" f:ident : tactic =>
  `(tactic| (
      unfold CudaOxide.Mir.Program.MirFunc.validInput CudaOxide.Mir.Program.MirFunc.validThread
      dsimp only [$f:ident, List.foldr, symSlotGuardOpt, symPairRaceOblig, Option.getD]
      mir_solve_vc))

end CudaOxide.Proofs.Program
