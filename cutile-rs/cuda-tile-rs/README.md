# cuda-tile-rs

Rust FFI bindings to the MLIR C-API and the cuda-tile C-API, built against the
bundled [cuda-tile](https://github.com/NVIDIA/cuda-tile) submodule. The crate
also ships `cuda-tile-translate` and can install it into a user-chosen
directory.

No dependency on `mlir-sys` / `melior` — bindings are generated at build time
from the LLVM + MLIR sources that cuda-tile's cmake downloads.

## Build

`build.rs` invokes cmake on the submodule. cuda-tile's cmake uses
`FetchContent` to clone a pinned LLVM commit and builds it as part of its
configure step.

<a id="first-build-cost-one-time-slow"></a>
### First build

The first build downloads and compiles LLVM from source:

- network access is required to clone `https://github.com/llvm/llvm-project`
  at a commit pinned by cuda-tile (~several hundred MB of git history),
- host compilation of LLVM + MLIR + the cuda-tile libs consumes several GB
  of disk and most of your CPU cores for tens of minutes to an hour (depends
  on the machine),
- after that point, the compiled artifacts are cached (see below) and
  subsequent builds complete in seconds.

This crate is intentionally excluded from the workspace's `default-members`,
so top-level `cargo build` / `cargo test` do **not** trigger the LLVM build.
Build it explicitly:

```bash
cargo build --release -p cuda-tile-rs
```

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
cuda-tile-translate --mlir-to-cudatilebc kernel.mlir -o kernel.tileirbc
```
