# pliron-mlir-interop

IR translation between pliron and MLIR.

Design Goal: The target IR is textual.

 - **Pliron -> MLIR**: should not require MLIR (or any C++ dependency)
 - **MLIR -> Pliron**: should not require Pliron (or any Rust dependency)

## Pliron -> MLIR

Translation is driven by the interfaces `ToMlirOp`, `ToMlirType` and `ToMlirAttr`.
`MlirPrinter` is the entry point, and implements `Display`.
See the [crate docs](https://pliron-org.github.io/pliron-mlir-interop/pliron_mlir_interop/)
for more information.

Translations for pliron's `builtin` and `LLVM` dialects are provided.
Other dialects must implement translations using the above interfaces.

A typical translation looks like this:

```rust
use pliron_mlir_interop::{MlirPrinter, llvm::lowering::lower_for_mlir};

lower_for_mlir(ctx, module.get_operation())?;
let mlir = String::try_from(MlirPrinter::new(ctx, &module.get_operation()))?;
```

Current translations target MLIR-23.

### Testing

`cargo test` checks the translated MLIR against `expect!` based snapshots.

Set `$MLIR_OPT` to an `mlir-opt` binary and the testsuite will additionally
verify the translated IR in the tests.

```sh
MLIR_OPT=/path/to/mlir-opt cargo test
```

