# cuda-tile-rs

Rust FFI bindings to the MLIR C-API and the cuda-tile C-API, built against the
bundled [cuda-tile](https://github.com/NVIDIA/cuda-tile) submodule. The crate
also ships `cuda-tile-translate` and can install it into a user-chosen
directory.

The submodule is pinned to the public
[CUDA Tile IR 13.5.0](https://github.com/NVIDIA/cuda-tile/releases/tag/v13.5.0) release. Its
translator reads and writes bytecode versions 13.1–13.5, including the
13.5 SIMT interoperability preview when explicitly enabled. This is a
separately built development tool; it does not require a CUDA toolkit or GPU.

No dependency on `mlir-sys` / `melior` — bindings are generated at build time
from the LLVM + MLIR sources that cuda-tile's cmake downloads.

## Build

`build.rs` invokes cmake on the submodule. cuda-tile's cmake uses
`FetchContent` to clone a pinned LLVM commit and builds it alongside cuda-tile.

<a id="first-build-cost-one-time-slow"></a>
### First build

The first build downloads and compiles LLVM from source:

- network access is required to clone `https://github.com/llvm/llvm-project`
  at a commit pinned by cuda-tile (several GB of sources and git history),
- host compilation of LLVM + MLIR + the cuda-tile libs consumes several GB
  of disk and most of your CPU cores for tens of minutes to an hour (depends
  on the machine),
- after that point, the compiled artifacts are cached (see below) and
  subsequent builds complete in seconds.

This crate is intentionally excluded from the workspace's `default-members`,
so top-level `cargo build` / `cargo test` do **not** trigger the LLVM build.
Build it explicitly:

```bash
# From the cuda-rust repository root:
git submodule update --init cutile-rs/cuda-tile-rs/cuda-tile
cd cutile-rs
cargo build --release -p cuda-tile-rs
```

The remaining Cargo commands assume the `cutile-rs/` workspace directory.

### Caching (opt-in)

By default, cmake builds into cargo's `$OUT_DIR`; `cargo clean` wipes it and
forces a full rebuild next time. To keep artifacts across `cargo clean`, opt
into a persistent, content-addressed cache keyed by
`(cuda-tile commit SHA, LLVM commit SHA)`.

To enable caching for the workspace, add this to `.cargo/config.toml`:

```toml
# at <workspace-root>/.cargo/config.toml
[env]
CUDA_TILE_RS_CACHE = "1"
```

The default cache location is `$XDG_CACHE_HOME/cuda-tile-rs/<key>/` (or
`$HOME/.cache/cuda-tile-rs/<key>/` if `XDG_CACHE_HOME` is unset).

**Alternative: per-invocation env var.**

```bash
CUDA_TILE_RS_CACHE=1 cargo build --release -p cuda-tile-rs

# Or a custom cache root:
CUDA_TILE_RS_CACHE_DIR=/path/to/cache cargo build --release -p cuda-tile-rs
```

Changing either pinned commit starts a fresh build in a new cache directory.
Old directories remain until you remove them.

### Force-clear the cache

Only meaningful when caching is enabled. Wipes the cache root before
building (useful if the cache got corrupted):

```bash
CUDA_TILE_RS_CACHE=1 CUDA_TILE_RS_CLEAR_CACHE=1 \
    cargo build --release -p cuda-tile-rs
```

## Use the Rust API

Bindings live under the `cuda_tile_rs::ffi` module (raw `unsafe extern "C"`
declarations for every `mlir-c/*` and `cuda_tile-c/*` symbol). Thin helpers:

```rust
use cuda_tile_rs::ffi::*;

unsafe {
    let registry = mlirDialectRegistryCreate();
    mlirCudaTileRegisterAllDialects(registry);
    let ctx = mlirContextCreateWithRegistry(registry, false);
    mlirContextLoadAllAvailableDialects(ctx);
    // ... use MLIR C-API from here ...
    mlirContextDestroy(ctx);
    mlirDialectRegistryDestroy(registry);
}
```

Raw bindings follow the pinned upstream C API. When upgrading from the 13.2
pin, note that `mlirCudaTileOptimizationHintsAttrGetEntryOpHint` now takes
`numWorkerWarps` between its `numCta` and `occupancy` arguments.

## Install `cuda-tile-translate`

Set `CUDA_TILE_TRANSLATE_INSTALL_DIR` to an existing directory and `build.rs`
will copy the built `cuda-tile-translate` binary there:

```bash
CUDA_TILE_TRANSLATE_INSTALL_DIR=$HOME/.cargo/bin \
    cargo build --release -p cuda-tile-rs
```

After this, `cuda-tile-translate` is on `PATH` (assuming `$HOME/.cargo/bin` is,
which is the rustup default). Usage:

```bash
cuda-tile-translate --cudatilebc-to-mlir kernel.tileirbc -o kernel.mlir
cuda-tile-translate --mlir-to-cudatilebc --bytecode-version=13.5 \
    kernel.mlir -o kernel.tileirbc
```

For the [tile-library SIMT preview](https://nvidia.github.io/cuda-rust/cutile/latest/guide/tile-libraries.html),
pass `--preview=simt` in both directions:

```bash
cuda-tile-translate --cudatilebc-to-mlir --preview=simt \
    library.tileirbc -o library.mlir
cuda-tile-translate --mlir-to-cudatilebc --preview=simt --bytecode-version=13.5 \
    library.mlir -o library.tileirbc
```

Raw C-API users must likewise call `mlirCudaTileContextSetPreviewFeatures`
with `"simt"` on the context before parsing or processing preview IR.

For tests that only inspect encoded bytecode, prefer the standalone
[`cutile_ir::decode_bytecode`](../cutile-ir/src/bytecode/decoder.rs) function:
it avoids building LLVM/MLIR. Its output is an inspection dump, not MLIR;
use the translator when a test needs MLIR parsing, verification, or conversion.

## Test

```bash
CUDA_TILE_RS_CACHE=1 cargo test --release -p cuda-tile-rs
```

This runs the C-API smoke tests and bytecode round trips with the translator
built by this crate, independently of `PATH`. Coverage includes versions
13.1–13.5, 13.5 pointer classifications, and SIMT preview declarations,
implementations and calls. These tests need no CUDA toolkit or GPU, but incur
the same initial LLVM/MLIR build cost as the crate itself.

After installing the translator as above, validate cuTile's standalone encoder
against it, including 13.5 SIMT preview calls:

```bash
CUTILE_REFERENCE_TRANSLATE=$HOME/.cargo/bin/cuda-tile-translate \
    cargo test -p cutile-ir --test bytecode_13_5
```

An explicitly configured translator must pass; an incompatible version is a
test failure rather than a skipped check.
