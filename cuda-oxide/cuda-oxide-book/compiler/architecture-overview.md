# Architecture Overview

You have written a `#[kernel]` function in Rust. It type-checks. It
borrow-checks. Now it needs to become PTX that runs on a GPU. This chapter
explains how cuda-oxide gets it there -- every stage, every crate, and the
reasoning behind each choice.

If you just want to write kernels, you never need to read this page. But if you
want to hack on the compiler, contribute a pass, or satisfy the "but how does it
*actually* work?" itch, welcome. Grab a coffee.

---

## Design Philosophy

The guiding principle is short enough to fit on a sticky note:

> **Use the best tool for each stage -- but own the full pipeline.**

Compilers are layer cakes. Each layer has a wildly different job, and different
tools excel at each one. cuda-oxide picks the strongest option per stage rather
than building everything from scratch:

- **Frontend: `rustc` + `rustc_public` (Stable MIR).**
  Why rewrite a type checker when one of the best ever built already exists?
  Rust's compiler handles parsing, name resolution, type inference, borrow
  checking, trait resolution, monomorphization, and MIR optimization. We take
  all of that for free.

- **Middle-end: `pliron` (Pliron IR, MLIR-like).**
  We need a place to transform Rust MIR into something LLVM-shaped. pliron is
  an extensible IR framework inspired by LLVM's MLIR, but written in pure Rust.
  No C++ dependency, no CMake, no tablegen -- just `cargo build`. We define
  two custom dialects here (one for MIR, one for NVIDIA GPU intrinsics) and
  consume the LLVM dialect from the upstream `pliron-llvm` crate.

- **Backend: LLVM NVPTX.**
  NVIDIA has poured years of work into the NVPTX backend in LLVM. It knows
  every register class, every instruction encoding, every scheduling quirk.
  We emit LLVM IR textually and hand it to `llc`. Standing on the shoulders
  of giants beats reinventing the PTX assembler.

The payoff: the entire compiler is written in Rust (except the final `llc`
invocation). There are no opaque handoffs to a C++ middle-end. You can set a
breakpoint in any transformation pass, `println!` your way through the IR, and
run the whole thing under Miri if you feel adventurous. Standard Rust tooling,
all the way down.

```{note}
`llc` is the one external binary. The pinned Rust toolchain's `llvm-tools`
component ships `llc` with the NVPTX backend enabled, so on a fresh clone
`rustup component add llvm-tools` (already listed in `rust-toolchain.toml`)
is the only step needed. A system LLVM 21+ install also works and serves as
the fallback. The CUDA Toolkit alone is not enough — it does not include
`llc`. All cuda-oxide stages up to LLVM IR emission are implemented in
Rust; after the backend writes the `.ll` file, it invokes external LLVM
`llc` to generate PTX.
```

---

## The Pipeline at a Glance

Here is the full journey of a `#[kernel]` function, from source to silicon:

```{figure} images/compiler-pipeline.svg
:align: center
:width: 100%

The full compilation pipeline. Rust source enters the rustc frontend, passes
through Stable MIR, is translated into `dialect-mir` (with `mem2reg` promoting
allocas back into SSA), applies annotated loop unrolling, lowers to the LLVM
dialect, exports textual LLVM IR, and finally compiles to PTX.
```

Stage by stage:

1. **Rust Source.**
   You write a function, slap `#[kernel]` on it, and go about your day. The
   proc macro renames it into the reserved `cuda_oxide_kernel_<hash>_<name>`
   namespace so the backend can spot it later. The exact prefix lives in the
   workspace-internal `reserved-oxide-symbols` crate; the `<hash>` makes
   accidental collisions impossible.

2. **rustc Frontend.**
   rustc parses, type-checks, borrow-checks, monomorphizes generics, and
   runs MIR optimization passes (inlining, constant propagation, dead code
   elimination). All the hard work happens here.

3. **Stable MIR.**
   The codegen backend receives rustc's internal MIR and bridges it to
   `rustc_public`'s stable types. This gives us a versioned, stable view of
   the MIR that won't break on the next nightly.

4. **`dialect-mir` (pliron).**
   `mir-importer` translates Stable MIR into `dialect-mir` -- a pliron
   dialect that models Rust MIR semantics (places, projections, `Rvalue`,
   `BinOp`, etc.). The initial form uses per-local `mir.alloca` slots with
   `mir.load`/`mir.store` for cross-block data flow; `pliron::opts::mem2reg`
   then promotes those slots back into SSA values.
   [Compiler optimizations](compiler-optimizations.md), beginning with
   annotated loop unrolling, run on that SSA form before lowering.

5. **LLVM dialect (pliron-llvm).**
   `mir-lower` transforms `dialect-mir` operations into LLVM dialect
   operations: `llvm.alloca`, `llvm.load`, `llvm.store`,
   `llvm.getelementptr`, `llvm.call`, and friends. This is where
   Rust-level concepts get flattened to machine-oriented IR. The LLVM
   dialect itself is provided by the upstream `pliron-llvm` crate.

6. **LLVM IR (.ll file).**
   NVVM builds first use `nvvm-transforms` to convert modern LLVM operations
   to the forms accepted by the selected libNVVM dialect. Ordinary PTX builds
   skip that transform. The `llvm-export` printer then writes the textual
   LLVM IR, which can be inspected or diffed between compiler versions.

7. **GPU artifact.**
   Ordinary builds use `llc` to compile the `.ll` file to PTX. NVVM builds use
   libNVVM and nvJitLink to produce LTOIR and then a cubin.

---

## Crate Map

cuda-oxide is split into focused crates. Here is every one and its role:

| Crate                | Role                                                                                   |
| :------------------- | :--------------------------------------------------------------------------------------|
| `rustc-codegen-cuda` | Custom rustc codegen backend -- intercepts `codegen_crate()`, splits host/device code  |
| `mir-importer`       | Translates Stable MIR into `dialect-mir`, orchestrates the full pipeline               |
| `dialect-mir`        | pliron dialect modeling Rust MIR semantics (places, rvalues, terminators)              |
| `mir-transforms`     | Analyzes and optimizes `dialect-mir` before lowering; currently provides loop unrolling |
| `nvvm-transforms`    | Converts lowered LLVM operations to the form accepted by the selected NVVM dialect     |
| `llvm-export`        | Re-exports `pliron-llvm`'s LLVM dialect + cuda-oxide's textual `.ll` exporter          |
| `dialect-nvvm`       | pliron dialect for NVIDIA GPU intrinsics (`tid`, `ntid`, barriers, TMA)                |
| `mir-lower`          | Lowers `dialect-mir` to the LLVM dialect -- the main transformation pass               |
| `cargo-oxide`        | CLI tool: `cargo oxide build`, `cargo oxide run`, `cargo oxide pipeline`               |
| `cuda-device`        | Device-side API: intrinsics, `DisjointSlice`, barriers, shared memory, warp ops        |
| `cuda-macros`        | Proc macros: `#[kernel]`, `#[device]`                                                  |
| `cuda-host`          | Host-side typed module loading and launch helpers                                      |
| `cuda-core`          | Shared with cutile-rs: safe Driver API bindings; SIMT surface under `cuda_core::simt`   |
| `cuda-async`         | Shared with cutile-rs: `DeviceOperation` model and combinators under `cuda_async::simt` |
| `cuda-bindings`      | Shared with cutile-rs: FFI to the CUDA driver; `libcuda` is loaded at run time         |
| `cuda-intrinsics`    | Generated low-level CUDA intrinsic declarations                                        |
| `cuda-intrinsics-gen`| Extractor and deterministic source generator for the intrinsics                        |
| `cuda-artifact-finalizer` | Driver-independent NVVM IR, LTOIR, and PTX finalization                           |
| `oxide-artifacts`    | Architecture-neutral embedded device artifact metadata                                 |
| `dialect-iket`       | pliron dialect modelling in-kernel event tracing                                       |
| `iket-lower`         | `dialect-iket` profiles + instrumentation lowering                                     |
| `dialect-ptx`        | pliron dialect modelling structured PTX                                                |
| `ptx-parse`          | Lossless structural views over PTX source text                                         |
| `ptx-schedule`       | PTX schedule-perturbation fuzzing (nanosleep injection campaigns)                      |
| `cuda-oxide-codegen` | Rustc-independent PTX backend                                                          |
| `libnvvm-sys`        | `dlopen` bindings to libNVVM (used by `cuda-host::ltoir`)                              |
| `cuda-target-spec`   | Shared CUDA target parsing and recorded LLVM PTX-floor policy                         |
| `nvjitlink-sys`      | `dlopen` bindings to nvJitLink (used by `cuda-host::ltoir`)                             |
| `reserved-oxide-symbols` | Workspace-private `cuda_oxide_*` symbol-name contract                              |
| `fuzzer`             | Differential codegen fuzzer support (rustlantis adapter)                               |

### Dependency flow

The compiler crates form a clear pipeline:

```{figure} images/crate-dependency-flow.svg
:align: center
:width: 100%

How the compiler crates connect. The pipeline flows left to right through the
codegen backend, importer, and lowering passes. Dialect crates sit underneath,
all built on pliron.
```

`pliron` sits underneath the dialect crates as the shared IR framework --
it provides the `Context`, `Module`, `Region`, `Block`, `Operation`, `Type`,
and `Attribute` infrastructure. The LLVM dialect comes from `pliron-llvm`
(also built on pliron); `llvm-export` re-exports it and adds the textual `.ll`
exporter. `rustc_public` provides the stable MIR types
that `mir-importer` reads from rustc. The user-facing crates (`cuda-device`,
`cuda-macros`, `cuda-host`, `cuda-core`, `cuda-async`) are independent of the
compiler internals and depend only on each other.

---

## The Two Key Dependencies

Two external projects make cuda-oxide possible. Neither is optional, and both
deserve a brief introduction before the deep-dive chapters that follow.

### pliron -- Pliron IR (MLIR-like)

[pliron](https://github.com/vaivaswatha/pliron) is an extensible compiler IR
framework inspired by LLVM's MLIR, but written entirely in Rust. It provides
the same core abstractions -- dialects, operations, types, attributes, regions,
and blocks -- without requiring a C++ toolchain, CMake, or tablegen.

cuda-oxide chose pliron over upstream MLIR for a pragmatic reason: we wanted
the entire compiler to build with `cargo`. Depending on MLIR means pulling in
the LLVM monorepo, a C++ build system, and Rust-C++ FFI glue -- all of which
add build complexity, slow down CI, and make contributor onboarding painful.
With pliron, dialects are defined using standard Rust traits and derive macros,
and the IR can be inspected with any Rust debugger.

cuda-oxide defines four dialects on top of pliron. Two carry the lowering
path: `dialect-mir` (models Rust MIR) and `dialect-nvvm` (NVIDIA GPU
intrinsics). Two sit off it: `dialect-iket` (in-kernel event tracing) and
`dialect-ptx` (structured terminal PTX). The LLVM dialect comes from the
upstream `pliron-llvm` crate; cuda-oxide's `llvm-export` crate re-exports it
and adds the textual `.ll` exporter.

:::{seealso}
For a deeper dive into pliron's architecture, see
[Pliron -- MLIR in Rust](pliron.md).
:::

### rustc_public -- Stable MIR

`rustc_public` (historically known as Stable MIR or `stable_mir`) is Rust's
official stable interface to the compiler's internals. MIR -- the Mid-level
Intermediate Representation -- is where borrow checking, lifetime validation,
and most optimizations happen. It is also a rich, high-level representation
that retains type information, making it an ideal starting point for a GPU
backend.

The problem: MIR is an *internal* representation. Its data structures change
between nightly versions with no stability guarantees. A backend that reads
internal MIR directly would break every time `rustc` refactors a field name or
reorders an enum variant -- which happens more often than you might hope.
`rustc_public` solves this by providing a versioned, stable API that bridges
internal types to a public surface. cuda-oxide hooks in at the
`CodegenBackend::codegen_crate()` entry point, bridges internal types to stable
MIR types, and hands the result to `mir-importer` for translation.

:::{seealso}
For a deeper dive into rustc_public, see
[rustc_public -- Stable MIR](rustc-public.md).
:::

---

## The Host/Device Split

cuda-oxide is a single-source compiler. Host code and device code live in the
same `.rs` files, and a single build command compiles both. Here is how that
works, step by step:

**1. cargo-oxide invokes rustc with the custom backend.**

```bash
cargo oxide run vecadd
```

Under the hood, this sets `-Z codegen-backend=librustc_codegen_cuda.so`, which
tells rustc to route code generation through cuda-oxide's backend instead of
the default LLVM one.

**2. rustc calls `codegen_crate()` for every crate in the dependency tree.**

This is not a cuda-oxide-specific step -- it is how rustc works. For every
crate being compiled (your binary, `cuda-device`, any other dependency), rustc
invokes the codegen backend.

**3. The backend scans for kernel entry points.**

It looks for monomorphized functions whose names contain the reserved
`cuda_oxide_kernel_<hash>_` prefix. These are the functions that `#[kernel]`
created.

**4. If kernels are found: build the device call graph and emit PTX.**

Starting from each kernel, the backend walks the call graph to collect every
device function the kernel transitively calls. This set of functions is handed
to `mir-importer`, which runs the full pipeline (`dialect-mir` ->
LLVM dialect -> `.ll` -> PTX). The result is a `.ptx` file written next to
the host binary.

**5. Always: delegate host code to the standard LLVM backend.**

Regardless of whether kernels were found, host code is compiled normally.
cuda-oxide's backend delegates to rustc's default LLVM codegen for everything
that is not device code. Your `main()` function, your CLI parsing, your async
runtime -- all compiled the usual way.

**6. Result: a host binary + a `.ptx` file, from one build.**

```text
target/debug/vecadd          ← host binary (loads PTX at runtime)
target/debug/vecadd.ptx      ← device code (loaded by CUDA driver)
```

```{note}
Device code from dependencies (like `cuda-device`) is compiled lazily.
Functions from external crates only get compiled to PTX when a kernel in your
crate transitively calls them. The MIR is available from `.rlib` metadata, so
there is no need to recompile dependencies from source -- the backend reads
their Stable MIR on demand.
```

### A simplified mental model

```{figure} images/host-device-split.svg
:align: center
:width: 100%

One build command, two compilation targets. Every function goes through rustc's
frontend. At the codegen boundary, kernels go to cuda-oxide; everything else
goes to LLVM.
```

Every function goes through rustc's frontend. At the codegen boundary, the
backend looks at each function and asks: "Are you a kernel or called by a
kernel?" If yes, you go right (cuda-oxide pipeline). If no, you go left
(standard LLVM). Some functions go both ways -- a generic helper used on both
host and device will be compiled twice, once per target.

---

## What rustc Gives Us For Free

One of the nicest things about building on `rustc` rather than inventing a new
language is the sheer volume of work we *do not* have to do. Here is what rustc
handles before cuda-oxide ever sees the code:

| What              | Value for GPU Code                                                                   |
| :---------------- | :------------------------------------------------------------------------------------|
| Type checking     | Catch errors before GPU compilation -- no cryptic PTX assembler failures             |
| Lifetime tracking | Safety guarantees that span the host/device boundary                                 |
| Borrow checking   | Enforce ordinary Rust aliasing; GPU-wide writes add `DisjointSlice` and launch proofs |
| Monomorphization  | Generics "just work" on the GPU -- `map<f32, _>` becomes a concrete PTX kernel       |
| MIR optimization  | Inlining, constant propagation, dead code elimination -- all applied before we begin |
| Trait resolution  | Trait objects are resolved, vtables are gone, everything is static dispatch          |
| Pattern matching  | `match` arms are lowered to optimized `SwitchInt` MIR terminators                    |

We do not reimplement any of this. rustc does the heavy lifting, and we pick
up the fully optimized, monomorphized, borrow-checked MIR at the end. Our job
is "just" the translation -- which, to be fair, is still plenty of work. But
it is a dramatically smaller problem than building a GPU language from scratch.

```{note}
This also means that Rust's error messages work normally. If you make a type
error in a kernel, you get the same helpful `rustc` diagnostic you would get
in any other Rust code -- complete with suggestions, span highlighting, and
"did you mean?" hints. No separate GPU compiler error format to learn.
```

---

## Where to Go Next

The rest of this chapter zooms into each piece of the architecture:

- **[Pliron -- Pliron IR (MLIR-like)](pliron.md)** -- the IR framework that holds
  the pipeline together.
- **[rustc_public -- Stable MIR](rustc-public.md)** -- how we read MIR without
  breaking on every nightly.
- **[The Code Generator: rustc-codegen-cuda](rustc-codegen-cuda.md)** -- the
  codegen backend that intercepts rustc.
- **[MIR Importer](mir-importer.md)** -- translating Stable MIR into pliron.
- **[Compiler Optimizations](compiler-optimizations.md)** -- how
  `mir-transforms` analyzes and rewrites `dialect-mir`.
- **[Pliron Dialects](mlir-dialects.md)** -- the three custom dialects and their
  operation sets.
- **[The Lowering Pipeline](lowering-pipeline.md)** -- `dialect-mir` to the
  LLVM dialect, pass by pass.
- **[Adding New Intrinsics](adding-new-intrinsics.md)** -- a contributor's
  guide to extending the compiler.
