# llvm-export

A thin shim around [pliron-llvm](https://github.com/vaivaswatha/pliron) plus a
pure-Rust textual LLVM IR (`.ll`) exporter, targeting the NVPTX backend.

The LLVM dialect *modeling* (ops like `llvm.add`, types like `llvm.ptr`,
attributes, and op-interfaces) lives upstream in the `pliron-llvm` crate.
cuda-oxide consumes it as a git dependency and no longer models the LLVM
dialect locally. This crate re-exports `pliron-llvm` so existing
`llvm_export::{ops,types,attributes,op_interfaces}` paths keep resolving, adds
a few GPU-specific extensions pliron-llvm lacks, and keeps the textual `.ll`
exporter local (pliron-llvm only emits real `.ll` via an `llvm-sys` bridge,
which cuda-oxide avoids).

```text
dialect-mir ──► mir-lower ──► LLVM dialect ──► export ──► .ll
                              (pliron-llvm)    (this crate)
                                     │
                                     └── NVVM builds first run nvvm-transforms
```

## What this crate adds on top of pliron-llvm

| Extension              | Why it lives here                                          |
|------------------------|------------------------------------------------------------|
| Named address spaces   | pliron-llvm stores a raw `u32`; we name generic/global/... |
| `PointerTypeExt`       | `get_shared` / `get_global` / `is_tmem` convenience        |
| `LlvmSyncScope` enum   | upstream syncscope is `Option<String>`; we keep an enum    |
| fp16 bit helpers       | `fp16_attr_from_bits` / `fp16_attr_to_bits`                |
| `AsmKind` + `InlineAsmOpExt` | `AsmKind`-tagged inline-asm builder used across mir-lower |
| `GlobalOpExt`          | explicit alignment on a `GlobalOp`                         |

The named address spaces are generic=0, global=1, shared=3, constant=4,
local=5, tmem=6 (Blackwell tcgen05 operands).

## LLVM dialect modeling (upstream)

The types (`StructType`, `PointerType`, `ArrayType`, `VectorType`, `FuncType`,
`VoidType`), operations (arithmetic, atomic, comparison, cast, memory, control
flow, aggregate, constants, symbol, call, select, inline asm, variadic), and
attributes (overflow flags, fast-math flags, comparison predicates, GEP
indices, linkage, atomic ordering / scope / rmw-kind) are all defined upstream
in `pliron-llvm`. See that crate for the full op/type/attribute reference and
the `Verify` trait impls that catch lowering bugs before export.

## Export to LLVM IR

The `export` module serializes the LLVM dialect module to textual LLVM IR. Two
backend configurations control the output format:

| Configuration      | Use Case    | `@llvm.used` | `!nvvmir.version` | `!nvvm.annotations`    |
|--------------------|-------------|--------------|-------------------|------------------------|
| `PtxExportConfig`  | PTX via llc | No           | No                | Launch bounds only     |
| `NvvmExportConfig` | NVVM IR     | Yes          | Yes               | All kernels            |

```rust
use llvm_export::export::{
    export_module_to_string, export_module_to_string_with_config,
    NvvmExportConfig, NvvmIrDialect,
};

// Default (PTX path)
let ll = export_module_to_string(&ctx, &module)?;

// NVVM IR path (for libNVVM / LTOIR)
// The caller parses its CUDA target once and selects the matching dialect.
let config = NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7);
let nvvm_ir = export_module_to_string_with_config(&ctx, &module, &config)?;
```

`NvvmExportConfig` selects LLVM 7 typed-pointer NVVM IR for pre-Blackwell
targets and modern opaque-pointer NVVM IR for Blackwell and newer targets. In
LLVM 7 output, each internal pointer has one neutral byte-pointer type. The
exporter casts it to the exact pointer type needed by a load, store, GEP, call,
or other typed operation.

The export handles block-arg to PHI-node translation, grouped intrinsic
declarations, `convergent` attribute on synchronization ops, kernel metadata
(`!nvvm.annotations`), launch bounds, cluster config, and device extern FFI
declarations.

### Target Configuration

| Setting        | Value                                    |
|----------------|------------------------------------------|
| Target triple  | `nvptx64-nvidia-cuda`                    |
| Data layout    | 64-bit pointers, 128-bit i128 alignment  |
| PTX version    | 8.7+ (for sm_120)                        |

## Registration

Registration is automatic. Every dialect, op, type, and attribute linked into
the binary registers itself when a `Context` is created (`Context::default`
runs all link-time registrations), so there is no explicit `register()` entry
point.

```rust
use pliron::context::Context;

let ctx = Context::default();  // ops, types, and attributes are already registered
```

## Source Layout

```text
src/
├── lib.rs              # Shim: re-exports pliron-llvm + GPU-specific extensions
└── export/
    ├── mod.rs          # Export entry points + backend configs
    ├── config.rs       # PtxExportConfig / NvvmExportConfig
    ├── module.rs       # Top-level module emission
    ├── function.rs     # Function bodies, block-arg to PHI translation
    ├── ops.rs          # Per-op textual emission
    ├── types.rs        # Type printing
    ├── literals.rs     # Constant / literal printing
    ├── metadata.rs     # !nvvm.annotations, launch bounds, cluster config
    ├── externs.rs      # Intrinsic + device extern declarations
    ├── names.rs        # SSA value / symbol naming
    └── state.rs        # Export state tracking
```

## Further Reading

- [dialect-mir](../dialect-mir/) -- pliron dialect modelling Rust MIR (lowering source)
- [dialect-nvvm](../dialect-nvvm/) -- NVVM GPU intrinsics
- [mir-lower](../mir-lower/) -- lowers `dialect-mir` into the LLVM dialect
