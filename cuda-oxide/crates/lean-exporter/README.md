<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Scalar MIR to Lean exporter

This experimental exporter reads a selected `mir.func` and translates its actual
SSA operations into `CudaOxide.Model.Scalar.Expr`. It accepts exactly one block,
one result, and arguments/results all of the same unsigned 8-, 16-, 32-, or 64-bit
integer type. Constants and `mir.add`, `mir.sub`, `mir.mul`, and `mir.return` are
supported. Integer arithmetic is modular, represented by Lean `BitVec`, including
overflow and subtraction underflow.

To inspect both the input MIR and generated Lean in one command, run from the
`cuda-oxide/` workspace directory:

```sh
cargo oxide lean-export
cargo oxide lean-export path/to/helper.mir --function helper
```

The first command selects the bundled demo below. The second selects a function
in your MIR file. Inspection does not invoke Lean or verify a kernel. For raw
Lean output only:

```sh
cargo run --quiet -p lean-exporter -- \
  crates/lean-exporter/fixtures/arithmetic.mir global_index > /tmp/Generated.lean
```

The input is either a standalone `mir.func` or a `builtin.module` containing direct
functions. Only the explicitly selected function is exported. The parser requires
end-of-file. Every operation in the function is checked, including dead operations.
Unsupported operations, semantic attributes, control flow, memory accesses, mixed
types, and malformed SSA are errors. Pliron structural and dominance verification
runs before output. Output uses fixed names independent of input identifiers.

The fixture is **hand-authored MIR**, not a Rust kernel extraction. Its exported
interface is `CudaOxide.Generated.program : Expr 64 3` and
`CudaOxide.Generated.evaluate : (Fin 3 → BitVec 64) → BitVec 64`. The three inputs
are block index, block size, and lane; the result is their modular multiply-add.

The exporter is trusted translation code. Lean can check the generated expression
and proofs about it, but that does not prove the exporter, Rust-to-MIR translation,
GPU launch behavior, floating-point arithmetic, memory safety, or emitted PTX.
There is no compiler build gate in this crate and it does not invoke Lean.

Run `cargo test -p lean-exporter` for accepted/rejected MIR and mutation tests.
