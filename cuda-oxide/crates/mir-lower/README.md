# mir-lower

`dialect-mir` → LLVM dialect lowering pass for cuda-oxide.

Converts [`dialect-mir`](../dialect-mir/) operations into LLVM dialect
operations (the LLVM dialect is provided by `pliron-llvm`), with GPU-specific
operations lowered to NVVM intrinsics or inline PTX assembly. This is the
bridge between Rust semantics and LLVM's target-agnostic IR.

## Pipeline Position

```text
Rust Source Code
       │
       ▼
┌──────────────┐
│   rustc      │  (extracts Stable MIR)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ mir-importer │  (Stable MIR → dialect-mir, mem2reg, annotated unroll)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  mir-lower   │  ◄── THIS CRATE (dialect-mir → LLVM dialect)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ llvm-export  │  (exports to LLVM IR)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│     llc      │  (LLVM IR → PTX)
└──────────────┘
```

## How It Works

The crate uses pliron's `DialectConversion` framework. Each
`dialect-mir` / `dialect-nvvm` op declares its own lowering via the
`MirToLlvmConversion` op interface. The framework handles IR walking,
def-before-use ordering, type conversion, and block argument patching
automatically.

For each `MirFuncOp`, `convert_func` (in `lowering.rs`):

1. Creates an LLVM dialect function with a flattened type signature
2. Propagates GPU kernel attributes (`gpu_kernel`, `maxntid`, etc.)
3. Uses `inline_region` to move the `dialect-mir` blocks into the new function
4. Builds an entry prologue that reconstructs aggregates (slices, structs)
   from the flattened LLVM dialect arguments via `insertvalue`
5. Branches to the original entry block with the reconstructed values

## Module Structure

### Core Modules

| Module                    | Purpose                                                    |
|---------------------------|------------------------------------------------------------|
| `lowering`                | `convert_func` — per-function lowering via `inline_region` |
| `conversion_interface`    | `MirToLlvmConversion` op interface trait                   |
| `convert/interface_impls` | Op interface impls dispatching to converter functions      |
| `context`                 | CUDA-specific state maps (shared globals, dynamic smem)    |
| `helpers`                 | Constants, intrinsic declarations, utilities               |
| `type_conversion_interface` | Type interfaces for MIR → LLVM type conversion            |
| `convert/type_interface_impls` | `#[type_interface_impl]` registrations for MIR → LLVM type conversion |
| `scalarize_block_args`    | Scalarizes aggregate-typed block arguments after lowering  |
| `wgmma_deferred_accumulator` | Fuses sound BF16 WGMMA sequences before conversion      |
| `convert/enum_payload_storage` | Backing storage for enum payloads during conversion   |

### Operation Converters (`convert/ops/`)

| Module         | `dialect-mir` Operations Handled                                                                               |
|----------------|----------------------------------------------------------------------------------------------------------------|
| `arithmetic`   | `mir.add`, `mir.sub`, `mir.mul`, `mir.div`, `mir.rem`, checked variants, shifts, bitwise, `mir.neg`, `mir.not` |
| `memory`       | `mir.alloca`, `mir.load`, `mir.store`, `mir.ref`, `mir.assign`, `mir.ptr_offset`                               |
| `control_flow` | `mir.return`, `mir.goto`, `mir.cond_br`, `mir.assert`, `mir.unreachable`, `mir.storage_live`/`dead` (erased)   |
| `constants`    | `mir.constant`, `mir.float_constant`, `mir.undef`                                                              |
| `cast`         | `mir.cast` (widening, narrowing, int↔float, ptr)                                                               |
| `aggregate`    | Struct/tuple/array/enum extract, insert, construct, field/element addr                                         |
| `call`         | `mir.call` (function calls with arg flattening)                                                                |

### Type Converter (`convert/types/`)

| `dialect-mir` Type   | LLVM dialect Type                                   |
|----------------------|-----------------------------------------------------|
| `mir.tuple`          | `llvm.struct` (anonymous, ZST fields dropped)       |
| `mir.ptr`            | `llvm.ptr` with address space                       |
| `mir.array`          | `llvm.array`                                        |
| `mir.slice`          | `llvm.struct {ptr, i64}`                            |
| `mir.disjoint_slice` | `llvm.struct {ptr, i64}` (same as slice)            |
| `mir.struct`         | `llvm.struct` (padded if layout known, else flat)   |
| `mir.enum`           | `llvm.struct` matching rustc's byte layout          |

### GPU Intrinsic Converters (`convert/intrinsics/`)

Anything `intrinsics/catalog.json` describes is lowered by the generated
`convert/generated_intrinsics/` module (one file per intrinsic family) --
one level up, beside `intrinsics/`, not inside it. The modules below are
the hand-written converters that sit next to it, one row per file:

| Module                   | Purpose (from each module's own doc comment)                              |
|--------------------------|--------------------------------------------------------------------------|
| `asm`                    | User-authored inline PTX lowering                                        |
| `atomic`                 | Atomic operation conversion: NVVM atomic dialect → LLVM atomic instructions|
| `basic`                  | Basic NVVM intrinsic conversion for special registers                    |
| `clc`                    | Lower generated Cluster Launch Control operations through typed NVVM calls|
| `cluster`                | Compatibility lowering for derived cluster-grid values                   |
| `common`                 | Common helpers for GPU intrinsic conversion                              |
| `cp_async`               | Lower generated classic `cp.async` operations through the selected backend|
| `debug`                  | Debug and profiling intrinsic conversion                                 |
| `dotprod`                | Lower generated packed integer dot products through the selected backend |
| `execution_control`      | Lowering for counted barriers, programmatic dependent launch, and register control|
| `extended_minmax`        | Lowering helper for generated extended min/max operations                |
| `integer_minmax`         | Lowering helper for generated extended integer min/max operations        |
| `ldmatrix`               | Lower `ldmatrix` operations through the selected intrinsic backend       |
| `mbarrier`               | Mbarrier lowering for Ampere and newer GPUs                              |
| `memory`                 | Memory address-space conversion intrinsics                               |
| `packed`                 | Shared lowering helpers for generated packed arithmetic and conversions  |
| `prmt`                   | Lower generated byte permutations through the selected backend           |
| `scalar_arithmetic`      | Lowering helper for generated scalar floating-point arithmetic           |
| `scalar_conversion`      | Lowering helper for generated scalar conversions                         |
| `scalar_math`            | Lowering helper for generated unary scalar floating-point math           |
| `tma`                    | TMA conversion for Hopper and newer GPUs                                 |
| `warp`                   | Warp-level intrinsic conversion: shuffle and vote operations             |
| `wgmma`                  | WGMMA conversion for Hopper `sm_90a`                                     |
| `wmma`                   | Shared lowering helpers for generated matrix intrinsics                  |

Per-intrinsic PTX and minimum-SM requirements are recorded in the catalog and
rendered into `intrinsics/generated-reference.md`, which is regenerated with
the sources; this table deliberately keeps no second copy of them.

## DialectConversion Framework

The lowering uses pliron's `DialectConversion` + `DialectConversionRewriter`
rather than manual walk-and-replace. The framework manages:

- **Value mapping**: source (`dialect-mir`) → target (LLVM dialect) value tracking
- **Type conversion**: registered via `can_convert_type` / `convert_type`
- **Block argument patching**: automatic type conversion of block args
- **Def-before-use ordering**: operations are visited in correct order

Each converter function receives `(ctx, rewriter, op, operands_info)` and
uses `rewriter.insert_operation()` / `rewriter.replace_operation_with_values()`
to emit LLVM dialect operations.

## Lowering Strategies

### LLVM Intrinsic Calls

For operations with direct NVVM equivalents (thread IDs, barriers,
atomics, TMA):

```text
dialect-mir/dialect-nvvm: nvvm.read_ptx_sreg_tid_x
LLVM dialect:             call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
```

### Inline PTX Assembly

For complex operations or when LLVM intrinsics don't exist (WGMMA,
tcgen05, stmatrix). Uses `convergent` attribute to prevent LLVM from
moving warp-synchronous ops across control flow:

```text
dialect-nvvm:  nvvm.tcgen05_mma_ws_f16
LLVM dialect:  call void asm "tcgen05.mma.cta_group::1.kind::f16...", "..." #convergent
```

## Shared Memory Handling

- **Static** (`SharedArray<T, N>`): Lowered to `@__shared_*` globals
  in address space 3 with deduplication via `SharedGlobalsMap`.
- **Dynamic** (`DynamicSharedArray<T>`): Lowered to `@__dynamic_smem_*`
  extern globals. `DynamicSmemAlignmentMap` tracks max alignment per
  kernel for correct PTX metadata.

## Dependencies

- [pliron](https://github.com/vaivaswatha/pliron) — Pliron IR (MLIR-like) framework
- [dialect-mir](../dialect-mir/) — Source dialect (pliron dialect modelling Rust MIR)
- [llvm-export](../llvm-export/) — pliron-llvm shim + textual `.ll` exporter
- [dialect-nvvm](../dialect-nvvm/) — NVVM intrinsic ops

## Further Reading

- [mir-importer](../mir-importer/) — produces `dialect-mir` from rustc
- [llvm-export](../llvm-export/) — exports textual LLVM IR from an LLVM dialect module
