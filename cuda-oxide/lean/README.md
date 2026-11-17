# Lean 4 Kernel Verification (`cuda-oxide/lean`)

**Rust establishes type and borrow guarantees; `#[launch_contract]` checks launch preconditions on the CPU; Lean 4 proves the remaining GPU-specific obligations at compile time.**

## End-to-End Architecture

```text
  WHAT THE USER WRITES (Rust)
  ┌─────────────────────────────────────────────────────────────────────┐
  │ #[launch_contract(domain = 1, block = (256, 1, 1),                  │
  │                   requires = (a.len() >= c.len(),                   │
  │                               b.len() >= c.len()))]                 │
  │ #[verify(no_oob_idx(a, b, c),                                       │
  │          disjoint_writes(c),                                        │
  │          writes_all_elems(c))]                                      │
  │ #[kernel]                                                           │
  │ pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>)      │
  └──────────────────────────────────┬──────────────────────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │   `dialect-mir` (`PostMem2Reg`) │
                    └────────┬───────────────┬────────┘
                             │               │
      Compile-time Lean gate │               │ Normal backend path
                             ▼               ▼
  ┌──────────────────────────────────┐   ┌──────────────────────────────┐
  │ `crates/lean-exporter`           │   │ `mir-lower` -> LLVM -> PTX   │
  │                                  │   └──────────────┬───────────────┘
  │ Emits `CudaOxide.Generated`:     │                  │
  │  1. `Mir.*` access summary       │                  │
  │  2. `Contracts.*` theorems       │                  │
  │  3. `Proofs.*` applications      │                  │
  └─────────────────┬────────────────┘                  │
                    │                                   │
                    ▼                                   │
  ┌──────────────────────────────────┐                  │
  │ `lake env lean` + Axiom Audit    │                  │
  │ (`propext`, `Quot.sound` only)   ├────── Pass ─────►┘
  └──────────────────────────────────┘
```

## Package Layout (`CudaOxide/`)

```text
  CudaOxide/
  ├── Mir/                   1. The MIR & GPU Representation in Lean
  │   ├── Scalar.lean           `Expr width arity` (`BitVec` wrapping word math)
  │   ├── Launch.lean           `Thread1D blocks blockSize` & `globalIndex`
  │   ├── Kernel.lean           `GuardedAccess1D`, `PhasedTileAccess` & `TmemAccess`
  │   └── ControlFlow.lean      `ControlFlowCfg`, `CfgOp` & `CtaState` transitions
  │
  ├── Contracts/             2. The `#[verify(...)]` Specifications (`Prop`)
  │   ├── Access.lean           `NoOobIdx`, `NoDataRace`, `NoOobTmemAccess`
  │   └── ControlFlow.lean      `UniformBarriers`, `TmemLifecycle`, `WritesOnAllActivePaths`
  │
  └── Proofs/                3. The Reusable Proof Library (`theorem`)
      ├── Bounds.lean           `no_oob_idx_*`, `no_oob_tile_*`, `no_oob_tmem_*`
      ├── Launch.lean           `disjoint_writes_1d`, `writes_all_elems_1d`, `no_data_race_*`
      └── ControlFlow.lean      `uniform_barriers_sound`, `tmem_lifecycle_sound` (`by decide`)
```

| Rust `#[verify(...)]` Clause | Lean Contract (`CudaOxide.Contracts`)             | Reusable Proof (`CudaOxide.Proofs`)                      |
|:-----------------------------|:--------------------------------------------------|:---------------------------------------------------------|
| `no_oob_idx(buf)`            | `NoOobIdx` / `NoOobTileAccess`                    | `Bounds.no_oob_idx_*` / `Bounds.no_oob_tile_*`           |
| `no_data_race(buf)`          | `NoDataRace`                                      | `Launch.no_data_race_*`                                  |
| `disjoint_writes(buf)`       | `DisjointWrites` / `DisjointGroupWrites`          | `Launch.disjoint_writes_1d`                              |
| `writes_all_elems(buf)`      | `WritesAllElems` + `ControlFlow.WritesOnAllActivePaths` | `Launch.writes_all_elems_1d` + `ControlFlow.writes_on_active_paths_sound` |
| `uniform_barriers`           | `ControlFlow.UniformBarriers`                     | `ControlFlow.uniform_barriers_sound` (`by decide`)       |
| `tmem_lifecycle`             | `ControlFlow.TmemLifecycle` + `NoOobTmemAccess`   | `ControlFlow.tmem_lifecycle_sound` + `Bounds.no_oob_tmem_*` |

## Commands

Run from the `cuda-oxide/` workspace directory:

```bash
# 1. Build and run the end-to-end verified vecadd example on the GPU:
cargo oxide run safe_vecadd

# 2. Inspect exported Lean for the scalar MIR fixture or vecadd MIR fixture:
cargo oxide lean-export
cargo oxide lean-export crates/lean-exporter/fixtures/vecadd.mir --function vecadd

# 3. Run the Lean proof & counterexample suite and exporter mutation suite:
python3 lean/check.py
python3 lean/check-exporter.py
```

## Trust Boundary

`lean-exporter`, `mir-importer`, `mem2reg`, and downstream LLVM/PTX lowering are
part of the trusted computing base (TCB). The exporter fails closed on
unsupported MIR operations, unguarded buffer accesses, and unknown attributes,
and the Lean runner rejects `sorryAx` and unexpected axioms.
