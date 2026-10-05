# dialect-nvvm

A [Pliron](https://github.com/vaivaswatha/pliron) dialect for NVIDIA GPU
operations. It is the compiler IR between imported CUDA intrinsics and their
LLVM or PTX implementation.

```text
cuda_device API
      │ rustc MIR
      ▼
mir-importer creates dialect-nvvm ops
      │
      ▼
mir-lower selects typed LLVM intrinsics or inline PTX
      │
      ▼
llvm-export emits LLVM IR → NVPTX backend → PTX
```

## Generated and handwritten operations

Leaf intrinsic operations are generated from the reviewed catalog under
`intrinsics/`. Generated Rust source lives in `src/ops/generated/` and should
not be edited by hand.

Top-level files in `src/ops/` contain only operations that do not fit a direct
generated leaf:

- compiler carriers such as inline PTX and general atomic IR;
- composite operations that need custom lowering;
- compatibility types retained for the public dialect API.

The `handwritten_ops_match_reviewed_allowlist` test requires every top-level
operation to stay explicitly reviewed.

## Verification

Generated operations use the contracts selected by their generator recipes.
Some handwritten compiler carriers use structural verification and rely on
their importer and lowering paths for additional checks.

## Registration

```rust
use pliron::context::Context;
use dialect_nvvm::register;

let mut ctx = Context::new();
register(&mut ctx);
```
