/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide.Proofs.Program.Expr
import CudaOxide.Proofs.Program.Exec
import CudaOxide.Proofs.Program.LoopBounds
import CudaOxide.Proofs.Program.Invariant
import CudaOxide.Proofs.Program.Vc
import CudaOxide.Proofs.Program.Tactic

/-!
# Verified Symbolic VC Generator & `mir_vc` Tactic (`CudaOxide.Proofs.Program`)

Umbrella module re-exporting the 6-file line-by-line `MirFunc` verification library:

```text
  CudaOxide/Proofs/Program/
  ├── Expr.lean        Symbolic ASTs (`SymInt`, `SymBool`, `SymVal`), `PhaseSpec` & implication
  ├── Exec.lean        Intra-block symbolic execution (`symEvalRhs`, `symExecStmts_correct`)
  ├── LoopBounds.lean  Inductive loop/ring counter bounds & word-width checks (`symIntBoundedUnder`)
  ├── Invariant.lean   CFG abstract interpreter (`inferSymTable`, `validSymFunc`, `symReachable_sound`)
  ├── Vc.lean          Verification Condition bridge theorems (`vcNoOob*_of_table`, `vcNoSharedRace_of_table`)
  └── Tactic.lean      `mir_def_sym_table`, generic `mirBindGoal`, `mir_mul_mono`, and `mir_vc`
```
-/
