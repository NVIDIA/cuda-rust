# `#[cuda_module]` in a Library Crate

Regression test for [issue #72]: a `#[cuda_module]` defined in a
*library* crate must still be loadable with `kernels::load(&ctx)` from
the application binary.

[issue #72]: https://github.com/NVlabs/cuda-oxide/issues/72

## Structure

```text
cuda_module_in_lib/
├── Cargo.toml           # Binary crate
├── src/main.rs          # Loads and launches the library's module
├── README.md            # This file
└── kernel-lib/          # Library crate "module-kernels"
    ├── Cargo.toml
    └── src/lib.rs       # The #[cuda_module] itself (nested concrete kernels)
```

## The Bug

The codegen backend embeds each crate's compiled PTX in an extra object
file whose only content is the `.oxart` data section. For binary crates
that object is handed straight to the linker, so it always survives.
For library crates it becomes a member of the crate's `.rlib` archive,
and linkers only extract archive members that define a symbol someone
references. The artifact object defined no symbols, so the member was
silently dropped:

```text
Before: rlib = [host code .o, artifact .o (no symbols)] -> linker keeps host code only
        load() -> ModuleNotFound { name: "module-kernels" }

After:  artifact .o defines  cuda_oxide_artifact_anchor_246e25db_module_kernels_0_1_0
        load_named() references that symbol -> linker extracts the member
        load() -> bundle found, kernels launch
```

## What This Tests

1. The library bundle (`module-kernels`) is present in the executable.
   This check parses the binary's own `.oxart` section before any CUDA
   call, so it catches the regression even on GPU-less machines.
2. The binary's own bundle (`cuda_module_in_lib`) coexists with the
   library bundle, and both load by name.
3. Kernels from both modules launch and produce correct results.
4. The library's root `LoadedModule` has no direct kernel. Its `load()` still
   pins the descendant artifact without relying on `ops::LoadedModule` being
   referenced first.

Contrast with [cross_crate_kernel](../cross_crate_kernel/), where the
library exports *generic* kernels: those monomorphize (and embed their
PTX) in the consuming binary, so they never hit the archive-member path
exercised here.

## Run

```bash
cargo oxide run cuda_module_in_lib
```

On a machine without a CUDA driver, the archive-link check can run by itself:

```bash
cargo oxide build cuda_module_in_lib
./cuda-oxide/crates/rustc-codegen-cuda/examples/cuda_module_in_lib/target/release/cuda_module_in_lib --verify-bundles
```

## Expected Output

```text
=== #[cuda_module] in Library Crate Test ===

Test 1: embedded bundles in the executable
  found bundles: ["cuda_module_in_lib", "module-kernels"]
  ✓ PASSED: library and binary bundles are both embedded

Test 2: module_kernels::kernels::load + scale_f32/add_f32
  ✓ PASSED: library kernels load and run by bundle name

Test 3: bin_kernels::load + iota_f32
  ✓ PASSED: binary kernels unaffected by the fix

SUCCESS: #[cuda_module] in a library crate loads and runs
```
