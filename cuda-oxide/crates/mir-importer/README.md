# mir-importer

Rust MIR to `dialect-mir` translator for cuda-oxide.

Translates rustc's Stable MIR into [`dialect-mir`](../dialect-mir/) (a pliron
dialect, MLIR-like) using the alloca + load/store model, then calls the shared
`cuda-oxide-codegen` backend for preparation, lowering, export, and PTX or NVVM
IR generation.

## Architecture

```text
┌────────────── mir-importer ──────────────┐
│ Stable MIR ──▶ dialect-mir translation  │
└────────────────────┬─────────────────────┘
                     │ translated module
                     ▼
┌────────── cuda-oxide-codegen ────────────┐
│ verify ─▶ mem2reg/unroll ─▶ lower       │
│        ─▶ LLVM export ─▶ PTX/NVVM IR    │
└──────────────────────────────────────────┘
```

## Pipeline Steps

```text
translate → verify → mem2reg → annotated unroll → lower/export → optimize → PTX
```

Full variable-debug builds skip `mem2reg` and annotated unrolling so source
variables remain in stable memory locations for cuda-gdb.

1. **Translate** — Convert Stable MIR into `dialect-mir` using the alloca +
   load/store model (one `mir.alloca` per non-ZST local).
2. **Verify** — Check type consistency and structural invariants on the
   `dialect-mir` module.
3. **mem2reg** — Promote scalar alloca slots back to SSA via
   `pliron::opts::mem2reg`, eliminating the load/store traffic the translator
   produced.
4. **Unroll** — Apply supported `#[unroll]` and `#[unroll(N)]` requests to the
   SSA form.
5. **Lower and export** — Convert `dialect-mir` → LLVM dialect (via `mir-lower`)
   and export LLVM IR. By default, ordinary float operations carry the
   `contract` fast-math flag so NVPTX can fuse `fmul+fadd` into `fma.rn.f32`
   (matching nvcc's `--fmad=true`). `--no-fmad` omits that permission.
6. **Optimize** — Run `opt -O2` (via `LlvmToolchain`) on the exported IR.
   Skipped for full-debug builds (`-G`) so locals stay inspectable under
   cuda-gdb. Override with `CUDA_OXIDE_NO_OPT=1`.
7. **Generate** — Invoke `llc -fp-contract=fast` for PTX (or emit NVVM IR).
   The `-fp-contract=fast` flag activates the NVPTX backend's FMA contract
   mode; pair with the IR `contract` flag from step 5. Disable both gates with
   `CUDA_OXIDE_NO_FMA=1` or `cargo oxide run --no-fmad`. Explicit fused
   operations such as `f32::mul_add` remain fused.

NVVM IR and LTOIR defer final code generation. Their versioned `.target` file
requires the sibling `.options` file, which tells cuda-host, libNVVM, and
nvJitLink whether to use `-fma=0` or `-fma=1`. Copy both sidecars with the
artifact; a missing required sidecar is an error rather than a silent fallback.

## Output Modes

| Mode            | Output               | Use Case                            |
|-----------------|----------------------|-------------------------------------|
| PTX (default)   | `.ptx` assembly      | Standard GPU compilation via `llc`  |
| NVVM IR         | `.ll` (NVVM format)  | For libNVVM with `-gen-lto`         |

## Module Structure

### `translator/` — MIR to `dialect-mir` Translation

| Module      | Purpose                                        |
|-------------|------------------------------------------------|
| `body`      | Function-level translation, alloca setup       |
| `block`     | Basic block translation coordinator            |
| `statement` | Statement translation (assignments, storage)   |
| `terminator`| Terminator translation (goto, call, return)    |
| `rvalue`    | Expression translation (binops, casts, etc.)   |
| `types`     | Rust type → `dialect-mir` type conversion      |
| `values`    | MIR local → alloca-slot mapping + load/store   |
| `layout`    | Shared readers over rustc's aggregate layout   |
| `location`  | Source-location helpers for MIR translation    |
| `payload_store` | Enum-payload stores whose storage type differs from its usage |

### `terminator/intrinsics/` — intrinsic handlers

Anything `intrinsics/catalog.json` describes is dispatched by `generated`;
the modules beside it hold the cases the catalog does not cover. One row per
file:

| Module        | Purpose (from each module's own doc comment)                                |
|---------------|-----------------------------------------------------------------------------|
| `asm`         | Inline PTX marker-call translation                                          |
| `atomic`      | Atomic operation intrinsic handlers                                         |
| `bigint`      | Rust compiler bigint helper intrinsics                                      |
| `bitops`      | Rust compiler bit-manipulation intrinsics                                   |
| `debug`       | Debug and profiling intrinsics                                              |
| `exact_div`   | Rust compiler `exact_div` intrinsic                                         |
| `float_math`  | Rust compiler floating-point math intrinsics                                |
| `generated`   | Generated raw/compatibility path dispatch for CUDA intrinsics               |
| `iket`        | Translation of `cuda_device::iket` compiler markers                         |
| `indexing`    | Thread and block indexing intrinsics                                        |
| `layout`      | Rust dynamic-layout intrinsics for slices, `str`, and slice-tailed structs  |
| `memory`      | Memory access and conversion intrinsics                                     |
| `saturating`  | Rust compiler saturating integer intrinsics                                 |
| `tma`         | Tensor Memory Accelerator (TMA) intrinsics                                  |
| `wgmma`       | Hopper WGMMA (Warpgroup Matrix Multiply-Accumulate) intrinsics              |

Per-intrinsic PTX and minimum-SM requirements live in the catalog and are
rendered into `intrinsics/generated-reference.md`, so they are not restated
here.

### `pipeline.rs` — Compilation Orchestration

Registers dialects and translates functions, then calls the single
`cuda-oxide-codegen` backend orchestrator. That backend owns verification,
`mem2reg`, unrolling, device extern insertion, lowering, LLVM IR export, and
optional `llc` PTX generation.

## Alloca + load/store model

MIR allows reading locals from any block. Rather than threading values
through block arguments via a liveness analysis, the translator emits one
`mir.alloca` per non-ZST local at the top of the entry block and mediates
every def/use through `mir.store` / `mir.load` on that slot. Pliron's
`mem2reg` pass promotes the allocas back to SSA before the `dialect-mir` →
LLVM dialect lowering runs.

```text
Rust MIR (not strict SSA):               dialect-mir (alloca + load/store):

bb0: {                                   ^bb0(%arg0: i32, ...):
    _1 = 42;                                 %s1 = mir.alloca : !mir.ptr<i32>
    goto -> bb1;                             %v1 = mir.const 42 : i32
}                                            mir.store %v1, %s1
bb1: {                                       mir.goto ^bb1
    _2 = _1;   // cross-block read!      ^bb1:
    return;                                  %r = mir.load %s1
}                                            mir.return %r : i32
```

## GPU Target Auto-Detection

The pipeline inspects which intrinsics the code uses and selects a target:

| Feature Used           | Target    | Architecture         |
|------------------------|-----------|----------------------|
| tcgen05 / TMEM         | sm_100a   | Blackwell datacenter |
| WGMMA                  | sm_90a    | Hopper only          |
| TMA / mbarrier         | sm_100    | Hopper+ compatible   |
| Basic CUDA             | sm_80     | Ampere+ (max compat) |

Override with `CUDA_OXIDE_TARGET=<target>`.

## Public API

### Types

| Type                 | Purpose                                           |
|----------------------|---------------------------------------------------|
| `CollectedFunction`  | MIR instance + kernel flag + export name          |
| `DeviceExternDecl`   | FFI-style device symbol declaration               |
| `DeviceExternAttrs`  | Convergent / pure / readonly markers              |
| `PipelineConfig`     | Output dir, verbosity, dump flags, emit modes     |
| `CompilationResult`  | Paths to `.ll` and `.ptx`, resolved target        |

### Entry Point

```rust
use mir_importer::{run_pipeline, CollectedFunction, PipelineConfig};

let result = run_pipeline(&functions, &device_externs, &config)?;
// result.ptx_path, result.ll_path, result.target
```

### Error Types

`PipelineError` is defined in `cuda-oxide-codegen` and re-exported here
(`pipeline.rs`), so this is the full set of variants `run_pipeline` can
return. The crate's own `TranslationErr` (`error.rs`) is the narrower
per-function error the translator raises: `Unsupported`, `TypeError` and
`InvalidOp`.


| Variant          | When                                             |
|------------------|--------------------------------------------------|
| `NoBody`         | Function has no MIR body                         |
| `Translation`    | MIR → `dialect-mir` conversion failed            |
| `Verification`   | IR invariant violated (includes op context)      |
| `Lowering`       | `dialect-mir` → LLVM dialect pass failed         |
| `LoweredVerification` | Lowered LLVM-dialect invariant failed       |
| `Export`         | LLVM IR export failed                            |
| `PtxGeneration`  | `llc` invocation failed                          |
| `Optimization`   | `opt` invocation failed                          |
| `TargetSelection` | No target satisfied the module's requirements   |
| `UnsupportedLinking` | Device symbols could not be linked           |
| `LibdeviceUnavailable` | libdevice was needed but not found         |
| `InvalidMirPassPipeline` | `CUDA_OXIDE_MIR_PASSES` named an unknown pass |

## Translation Flow

```text
run_pipeline()
  ├─▶ register_dialects()
  ├─▶ For each CollectedFunction:
  │     └─▶ body::translate_body()
  │           ├─▶ emit_entry_allocas()  // one mir.alloca per non-ZST local
  │           └─▶ For each reachable block:
  │                 └─▶ block::translate_block()
  │                       ├─▶ statement::translate_statement()
  │                       │     └─▶ rvalue::translate_rvalue()
  │                       └─▶ terminator::translate_terminator()
  └─▶ cuda-oxide-codegen shared backend
        └─▶ verify → prepare → externs → lower → export → PTX/NVVM IR
```

## Dependencies

- [cuda-oxide-codegen](../cuda-oxide-codegen/) — shared post-translation backend
- [pliron](https://github.com/vaivaswatha/pliron) — Pliron IR (MLIR-like) framework
- [dialect-mir](../dialect-mir/) — pliron dialect modelling Rust MIR
- [llvm-export](../llvm-export/) — pliron-llvm shim + textual `.ll` exporter
- [dialect-nvvm](../dialect-nvvm/) — NVVM intrinsic ops
- [mir-lower](../mir-lower/) — `dialect-mir` → LLVM dialect lowering pass

## Further Reading

- [rustc-codegen-cuda](../rustc-codegen-cuda/) — the codegen backend that drives this crate
