# Lean 4 Kernel Verification (`cuda-oxide/oxide-proofs-preview`, preview)

> **Preview:** Compile-time Lean 4 verification (`#[verify(...)]` and `cuda-oxide/oxide-proofs-preview/`) is a preview feature, like `cute-rs-preview/`. It covers a validated subset of `PostMem2Reg` integer, shared-memory, barrier, cluster, and `tcgen05` / `TMEM` MIR; unsupported operations fail closed at compile time, and attribute syntax may change between releases.

**Rust proves memory validity and non-aliasing on the host; `#[launch_contract]` checks grid/length preconditions once before launch; Lean 4 proves GPU bounds, race freedom, barrier uniformity, and async hardware lifecycles at compile time.**

---

## 1. End-to-End Flow & Concrete Example

```rust
#[launch_contract(domain = 1, block = (256, 1, 1), requires = (a.len() >= c.len(), b.len() >= c.len()))]
#[verify(no_oob_idx(a, b, c), disjoint_writes(c), writes_all_elems(c))]
#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d().as_usize();
    if let Some(out) = c.get_mut(idx) {
        *out = a[idx] + b[idx];
    }
}
```

```text
  Rust `#[kernel]`  ──►  `PostMem2Reg` MIR (pre-unroll)
                                │
          ┌─────────────────────┴─────────────────────┐
          ▼                                           ▼
  `crates/lean-exporter`                      `mir-lower` -> LLVM -> PTX
  • Emits `def mirFunc` (1-to-1 blocks)               ▲
  • Emits `def controlFlowCfg` (barriers)             │
          │                                           │
          ▼                                           │
  `CudaOxide` in Lean 4 (`0 sorryAx`)                 │
  • `mir_def_sym_table` + `validSymFunc` (`rfl`)      │
  • Extracts Verification Conditions (`VC`)           │
    e.g. `idx < c.len ∧ c.len <= a.len -> idx < a.len`│
  • Discharges VCs via `mir_vc` (`omega`)             │
  • Erases proven `mir.assert` bounds checks ─────────┘
```

What is a **Verification Condition (`VC`)**? It is the plain `Nat` inequality extracted from `mirFunc` (`vcNoOobGlobal_of_table`, `vcNoSharedRace_of_table`, ...) so that proving the pure math formula via `omega` guarantees no reachable GPU execution step can go out of bounds or race.

---

## 2. Modular Package Layout (`CudaOxide/`)

```text
  CudaOxide/
  ├── Mir/                        1. Concrete IR & Hardware Semantics
  │   ├── Program.lean               `MirFunc`, `BasicBlock`, `Stmt`, `Rhs`, small-step `Reachable`
  │   ├── ControlFlow.lean           `ControlFlowCfg`, `CfgOp` (`mbarrier`, TMA, DSMEM, CLC, `TMEM`)
  │   ├── Scalar.lean                `Expr width arity` (`BitVec` word math)
  │   └── Launch.lean                `Thread1D` & `globalIndex` coordinate model
  │
  ├── Contracts/                  2. Declarative Safety Specifications (`Prop`)
  │   ├── Program.lean               `MirNoOobGlobal`, `MirNoOobShared`, `MirDisjointWrites`,
  │   │                              `MirNoSharedRace`, `MirNoOobTmem`, `MirNoOobClusterRank`,
  │   │                              `MirAssertRedundant`
  │   ├── ControlFlow.lean           `UniformBarriers`, `TmemLifecycle`, `WritesOnAllActivePaths`
  │   └── Access.lean                Standalone 1D coordinate contracts (`NoOobIdx`, `WritesAllElems`)
  │
  └── Proofs/                     3. Reusable Soundness Library (`0 sorryAx`)
      ├── Program/
      │   ├── Expr.lean              `SymInt`, `SymBool`, `SymVal`, `PhaseSpec` & implication
      │   ├── Exec.lean              Intra-block symbolic execution (`symExecStmts_correct`)
      │   ├── LoopBounds.lean        Inductive loop/ring counter bounds (`symIntBoundedUnder`)
      │   ├── Invariant.lean         CFG abstract interpreter (`inferSymTable`, `symReachable_sound`)
      │   ├── Vc.lean                Contract-to-VC bridge theorems (`vc*_of_table`)
      │   └── Tactic.lean            `mir_def_sym_table`, `mir_mul_mono`, and `mir_vc`
      ├── ControlFlow/
      │   ├── Lattice.lean           Abstract `CtaState` join/subsumption & `stepOp_mono`
      │   └── Soundness.lean         `ctaReachable_subsumed`, `uniform_barriers_of_stateMap`,
      │                              `tmem_lifecycle_of_stateMap`, `writes_on_active_paths_sound`
      ├── Bounds.lean                Standalone 1D bounds lemmas
      └── Launch.lean                Standalone 1D injectivity & grid-coverage lemmas
```

| Rust `#[verify(...)]` Clause | Lean Contract (`Contracts/`)                  | Reusable Bridge Theorem (`Proofs/`)            |
|:-----------------------------|:----------------------------------------------|:-----------------------------------------------|
| `no_oob_idx(slice)`          | `Program.MirNoOobGlobal`                      | `Program.vcNoOobGlobal_of_table` + `mir_vc`    |
| `no_oob_idx(SMEM)`           | `Program.MirNoOobShared`                      | `Program.vcNoOobShared_of_table` + `mir_vc`    |
| `disjoint_writes(slice)`     | `Program.MirDisjointWrites`                   | `Program.vcDisjointWrites_of_table` + `mir_vc` |
| `no_data_race(SMEM)`         | `Program.MirNoSharedRace`                     | `Program.vcNoSharedRace_of_table` + `mir_vc`   |
| `writes_all_elems(slice)`    | `ControlFlow.WritesOnAllActivePaths`          | `ControlFlow.writes_on_active_paths_sound`     |
| `uniform_barriers`           | `ControlFlow.UniformBarriers`                 | `ControlFlow.uniform_barriers_of_stateMap`     |
| `tmem_lifecycle`             | `ControlFlow.TmemLifecycle` + `MirNoOobTmem`  | `ControlFlow.tmem_lifecycle_of_stateMap`       |
|*(auto on `map_shared_rank`)* | `Program.MirNoOobClusterRank`                 | `Program.vcNoOobClusterRank_of_table` + `mir_vc`|
|*(auto on `mir.assert`)*      | `Program.MirAssertRedundant`                  | `Program.vcAssertRedundant_of_table` + `mir_vc`|

Extending to new GPU workloads (such as MoE routing, Attention, or block-scaled Tensor Cores) does not require redesigning the `MirFunc` or `ControlFlow` engines: new hardware instructions simply add small per-instruction rows to `Rhs`, `Stmt`, or `CfgOp`.

---

## 3. Commands

Run from `cuda-oxide/`:

```bash
# 1. Run all Lean library proofs, negative counterexamples, and exporter checks:
just check-lean

# 2. Build and run verified examples (Stages 1-10):
cargo oxide run safe_vecadd
cargo oxide run safe_warp_ring
cargo oxide run safe_cluster_clc
cargo oxide build safe_gemm_sol_final --arch sm_100a
```

## 4. Trust Boundary

`mir-importer`, `mem2reg`, `lean-exporter`, and downstream LLVM/PTX lowering are part of the trusted computing base (TCB). `lean-exporter` fails closed on unsupported MIR operations or attributes, and `runner.rs` rejects non-zero Lean exits, `sorryAx`, and unexpected axioms.
