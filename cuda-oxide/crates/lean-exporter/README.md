<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `dialect-mir` to Lean 4 Exporter (`lean-exporter`)

Turns `cuda-oxide`'s `dialect-mir` into Lean 4 code so Lean can prove kernel safety properties (`no_oob_idx`, `no_data_race`, `disjoint_writes`, `writes_all_elems`, `uniform_barriers`) before the compiler emits PTX.

---

## How It Works: One Typed SSA Lookup Table (`SsaEnv`)

By the time `lean-exporter` runs (at `PostMem2Reg`), `dialect-mir` is already in **SSA form**: every instruction produces a numbered Pliron `Value`, and later instructions pass those `Value`s as inputs.

The exporter keeps **one typed lookup table (`SsaEnv`)** that maps each Pliron `Value` to a `TypedSsa { lean_name, kind }`:

```text
  Pliron `Value`       `TypedSsa { lean_name, kind }`
  ───────────────      ────────────────────────────────────────────────────
  `block` (arg 0)  ──► `{ lean_name: "arg0", kind: ScalarWord { width: 64 } }`
  `a`     (arg 0)  ──► `{ lean_name: "arg0", kind: BufferParam(0) }`
  `prod`  (op 0)   ──► `{ lean_name: "v0",   kind: ScalarWord { width: 64 } }`
  `idx`   (op 1)   ──► `{ lean_name: "v1",   kind: Index1D { offset: 0 } }`
  `c_len` (op 2)   ──► `{ lean_name: "v2",   kind: SliceLen(2) }`
  `cond`  (op 3)   ──► `{ lean_name: "v3",   kind: IndexLtLen { guard_param: 2 } }`
```

Every `mir.func` goes through the same 4 steps:

```text
  1. `spec.rs`       Read function signature & optional `#[verify(...)]` attributes
        │
        ▼
  2. `translate.rs`  Walk MIR instructions using `SsaEnv` (`ssa.rs`), checking
        │            operand `SsaKind`s and recording outputs -> `TranslatedFunction`
        ▼
  3. `codegen.rs`    Render `TranslatedFunction` into `.lean` code + `theorem`s
        │
        ▼
  4. `runner.rs`     Run `lake env lean` and verify zero `sorry`s / unexpected axioms
```

---

## Module Map

| File               | Job in plain English                                                 |
|:-------------------|:---------------------------------------------------------------------|
| `src/lib.rs`       | Parses `.mir` input and runs `translate` -> `codegen`                |
| `src/ssa.rs`       | Defines `TypedSsa { lean_name, kind }`, `SsaKind`, and `SsaEnv`      |
| `src/spec.rs`      | Reads `#[verify(...)]` and `#[launch_contract(...)]` on `mir.func`   |
| `src/translate.rs` | Single MIR translator walking ops with `SsaEnv`                      |
| `src/codegen.rs`   | Renders `TranslatedFunction` into a `.lean` file and `theorem`s      |
| `src/runner.rs`    | Invokes `lake env lean` and audits proof axioms                      |

---

## Try It Out

Run from the `cuda-oxide/` directory:

```sh
# 1. Inspect the scalar arithmetic MIR demo and its generated Lean:
cargo oxide lean-export

# 2. Inspect the verified vecadd kernel MIR and its generated Lean theorems:
cargo oxide lean-export crates/lean-exporter/fixtures/vecadd.mir --function vecadd

# 3. Run unit and mutation tests:
cargo test -p lean-exporter
python3 lean/check-exporter.py
```
