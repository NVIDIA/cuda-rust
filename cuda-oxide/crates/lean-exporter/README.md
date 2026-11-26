<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `dialect-mir` to Lean 4 Exporter (`lean-exporter`, preview)

> **Preview:** `#[verify(...)]` and `lean-exporter` are in preview alongside `cuda-oxide/oxide-proofs-preview/`. Expect attribute syntax and supported MIR coverage to evolve between releases.

Thin, fail-closed transcriber that converts `PostMem2Reg` `dialect-mir` into Lean 4 (`MirFunc` + `ControlFlowCfg`) so Lean 4 proves kernel safety before `cuda-oxide` lowers MIR to PTX.

---

## 1. One Concrete Example

Rust kernel code:

```rust
#[launch_contract(domain = 1, block = (128, 1, 1))]
#[verify(no_oob_idx(TILE))]
#[kernel]
pub fn stage_tile() {
    static mut TILE: SharedArray<f32, 256> = SharedArray::UNINIT;
    let tid = thread::threadIdx_x() as usize;
    if tid < 64 {
        TILE[tid * 4 + 3] = 1.0;
    }
}
```

What `lean-exporter` transcribes 1-to-1 at `PostMem2Reg` (before loop unrolling):

```text
  PostMem2Reg `dialect-mir`                     Generated `.lean` (`CudaOxide.Generated`)
  ------------------------------------------    -------------------------------------------------------
  tx   = nvvm.read_ptx_sreg_tid_x()        -->  (.assign 0 (.sreg .tidX))
  lt64 = mir.lt(tid, 64)                   -->  (.assign 2 (.cmpOp .lt 1 3))
  mir.cond_br(lt64) [^write, ^exit]        -->  (.condBr 2 1 [] 2 [])
  ok   = mir.lt(off, 256)                  -->  (.assign 7 (.cmpOp .lt 6 8))
  mir.assert(ok)                           -->  (.assertCond 0 7)
  ptr  = mir.ptr_offset(TILE, off)         -->  (.assign 9 (.ptrOffset 4 6))
  mir.store(ptr, 1.0)                      -->  (.store 9 10)
```

Once `lake env lean` proves `MirNoOobShared mirFunc 0` and `MirAssertRedundant mirFunc 0` with zero `sorryAx`, `runner.rs` erases the proven `mir.assert` op in memory and PTX lowering continues with zero bounds-check overhead.

---

## 2. Pipeline & Module Map

```text
  mir.func (`PostMem2Reg` SSA)
     │
     ├─► 1. `spec.rs`               Parse `#[launch_contract]` & `#[verify(...)]`
     ├─► 2. `translate/scalar.rs`   Pure integer `BitVec` AST (`arithmetic.mir`)
     ├─► 2. `translate/program.rs`  1-to-1 `MirFunc` blocks + `ControlFlowCfg`
     ├─► 3. `codegen.rs`            Render `.lean` definitions & safety `theorem`s
     └─► 4. `runner.rs`             Run `lake env lean`, audit axioms, erase `mir.assert`s
```

| File                        | Role                                              |
|:----------------------------|:--------------------------------------------------|
| `src/lib.rs`                | Entry API, MIR parser, and fail-closed attributes |
| `src/spec.rs`               | Parses `#[launch_contract]` and `#[verify(...)]`  |
| `src/ssa.rs`                | `ControlFlowCfg` graph types and scalar `SsaEnv`  |
| `src/translate.rs`          | Routes functions to `scalar.rs` or `program.rs`   |
| `src/translate/ir_utils.rs` | `BlockGraph` queries and Pliron operand helpers   |
| `src/translate/scalar.rs`   | Single-block scalar `BitVec` transcriber          |
| `src/translate/program.rs`  | 1-to-1 `MirFunc` + `ControlFlowCfg` transcriber   |
| `src/codegen.rs`            | Renders `.lean` modules and `theorem` obligations |
| `src/runner.rs`             | Invokes `lake env lean` and audits proof axioms   |

---

## 3. Try It Out

Run from `cuda-oxide/`:

```sh
# 1. Inspect scalar BitVec export:
cargo oxide lean-export

# 2. Inspect exported MirFunc + theorems for a verified GPU kernel:
cargo oxide lean-export crates/lean-exporter/fixtures/vecadd.mir --function vecadd

# 3. Run unit, negative, and Lean proof tests:
cargo test -p lean-exporter
python3 oxide-proofs-preview/check-exporter.py
```
