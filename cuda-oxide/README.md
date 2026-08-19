<p align="center">
  <a href="https://github.com/NVlabs/cuda-oxide/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/NVlabs/cuda-oxide/ci.yml?branch=main&style=flat-square&logo=github-actions&logoColor=white&label=CI"></a>
  <a href="https://github.com/NVlabs/cuda-oxide/actions/workflows/examples-compile.yml"><img alt="examples" src="https://img.shields.io/github/actions/workflow/status/NVlabs/cuda-oxide/examples-compile.yml?branch=main&style=flat-square&logo=github-actions&logoColor=white&label=examples"></a>
  <a href="https://discord.gg/ZUEr4AhH5C"><img alt="discord" src="https://img.shields.io/discord/1515530041767759993?style=flat-square&logo=discord&logoColor=white&label=discord&color=5865F2"></a>
  <br>
  <img src="assets/logo.png" alt="cuda-oxide logo" width="100%">
</p>

# cuda-oxide

cuda-oxide is a custom rustc backend for compiling GPU kernels in pure Rust.
The workspace combines:

- single-source compilation -- host and device code live in the same file, built with one `cargo oxide build`
- a rustc codegen backend that compiles `#[kernel]` functions to CUDA PTX
- device-side abstractions (type-safe indexing, shared memory, scoped atomics, barriers, TMA, warp/cluster ops)
- compile-time kernel policies for separate tuned specializations without runtime policy arguments
- a host-side runtime for memory management, pinned host transfers, and kernel launching (`cuda-core`, `cuda-async`)
- a rust-native compilation pipeline using [Pliron](https://github.com/vaivaswatha/pliron), an MLIR-like IR framework in Rust (Rust → Rust MIR → Pliron IR → LLVM IR → PTX)

## Project Status

cuda-oxide is an experimental compiler that demonstrates how CUDA SIMT kernels can be written natively in pure Rust -- no DSLs, no foreign language bindings -- and made available to the broader Rust community. The project is in an early stage (alpha) and under active development: you should expect bugs, incomplete features, and API breakage as we work to improve it. That said, we hope you'll try it in your own work and help shape its direction by sharing feedback on your experience.

Please see [CONTRIBUTING.md](CONTRIBUTING.md) if you're interested in contributing to the project.

## Quick Start

```rust
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

// Device: generic kernel that applies any function to each element.
// F can be a closure with captures — rustc monomorphizes it to a concrete type.
#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn map<T: Copy, F: Fn(T) -> T + Copy>(f: F, input: &[T], mut out: DisjointSlice<T>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = f(input[i]);
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let input = DeviceBuffer::from_host(&stream, &data).unwrap();
    let mut output = DeviceBuffer::<f32>::zeroed(&stream, 1024).unwrap();

    let module = kernels::load(&ctx).unwrap();

    // Launch with a closure — factor is captured and passed to the GPU automatically
    let factor = 2.5f32;
    // SAFETY: this raw configuration is fully 1-D, matches index_1d(), and
    // launches one thread per output element. A launch contract can move this
    // proof into the generated safe API.
    unsafe {
        module.map::<f32, _>(
            &stream,
            LaunchConfig::for_num_elems(1024),
            move |x: f32| x * factor,
            &input,
            &mut output,
        )
    }
    .unwrap();

    let result = output.to_host_vec(&stream).unwrap();
    assert!((result[1] - 2.5).abs() < 1e-5);
}
```

The above example defines a generic `#[kernel]` function `map` that accepts any
`Fn(T) -> T` closure. `#[cuda_module]` embeds the generated device artifact into
the host binary and generates a typed `module.map::<f32, _>(...)` launch method.
The closure `move |x| x * factor` is captured, scalarized, and passed as kernel
parameters automatically. `LaunchConfig` is intentionally raw data, so using
it to launch a kernel is unsafe: the caller must prove that its dimensions and
resources match the kernel. Kernels with `#[launch_contract(...)]` instead use
a checked `PreparedLaunch` through the safe generated method.

For composable async GPU work, `stream:` disappears, `{kernel}_async` returns a
lazy `DeviceOperation`, and execution happens when you call `.sync()` or
`.await`.

```rust
use cuda_async::device_operation::DeviceOperation;

// Assuming `module`, `input`, and `output` come from the cuda-async setup:
let factor = 2.5f32;
let launch = unsafe {
    // SAFETY: the raw launch is 1-D and matches this kernel's index space.
    module.map_async::<f32, _>(
        LaunchConfig::for_num_elems(1024),
        move |x: f32| x * factor,
        &input,
        &mut output,
    )?
};
launch.sync()?;
// or: .await?;
```

See the `async_mlp` example and `crates/cuda-async/README.md` for the full async setup.

```bash
# Build and run an example
cargo oxide run host_closure

# Build and print the generated PTX
cargo oxide inspect vecadd

# Show full compilation pipeline (Rust MIR → dialect-mir → mem2reg → LLVM dialect → LLVM IR → PTX)
cargo oxide pipeline vecadd

# Remove project-local build outputs and generated artifacts
cargo oxide clean

# Run CUDA correctness checks
cargo oxide sanitize vecadd --tool memcheck

# Debug with cuda-gdb
cargo oxide debug vecadd --tui

# Run Cargo tests through the cuda-oxide backend
cargo oxide test

# Compile a crate's device code to a binary LTOIR artifact in one step
cargo oxide emit-ltoir

# Refresh the cached codegen backend
cargo oxide update
```

`cargo oxide --help` lists every subcommand.

## Setup

### Requirements

- **cargo-oxide** — cargo subcommand that drives the build pipeline (`cargo oxide run`, `build`, `sanitize`, `debug`, etc.)
- **Rust nightly** with `rust-src` and `rustc-dev` and `llvm-tools` components (pinned in `rust-toolchain.toml`)
- **CUDA Toolkit** (12.x+)
- **Clang + libclang dev headers** (`clang-21` / `libclang-common-21-dev`) — needed by `bindgen` when building the host `cuda-bindings` crate
- **Linux** (tested on Ubuntu 24.04)

### Install

#### cargo-oxide

Inside the cuda-oxide repo, `cargo oxide` works out of the box via a workspace alias.

For use outside the repo (your own projects), install it with the pinned nightly toolchain:

```bash
cargo +nightly-2026-04-03 install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide
```

On first run, `cargo-oxide` will automatically fetch and build the codegen backend.

#### Nix (alternative)

If you have Nix with flakes enabled, `nix develop` in the repo gives you a reproducible shell with CUDA 13, LLVM 22, Clang, and the pinned Rust nightly — no manual apt installs. The shellHook auto-discovers host NVIDIA drivers on NixOS and non-NixOS systems.

```bash
nix develop                                       # full dev shell in this repo
nix run github:NVlabs/cuda-oxide#new my-project   # bootstrap a project
```

#### Rust

```bash
# Toolchain installed automatically via rust-toolchain.toml
# Manual install if needed:
rustup toolchain install nightly-2026-04-03
rustup component add rust-src rustc-dev rust-analyzer clippy rustfmt llvm-tools --toolchain nightly-2026-04-03
```

#### CUDA

```bash
export PATH="/usr/local/cuda/bin:$PATH"
nvcc --version
```

#### LLVM (optional)

```bash
# Ubuntu/Debian
sudo apt install llvm-21
```

If your distro packages do not provide `llvm-21`, use LLVM's apt helper:

```bash
sudo apt-get install -y lsb-release wget software-properties-common gnupg
wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh
sudo ./llvm.sh 21
```

```bash
# Verify NVPTX support
llc-21 --version | grep nvptx
```

The pipeline prefers `llc` in Rust toolchain, and auto-discovers `llc-22` and `llc-21` on `PATH` (in that order).
To pin a specific binary, set `CUDA_OXIDE_LLC=/usr/bin/llc-21`.

> We emit TMA / tcgen05 / WGMMA intrinsics that `llc` from LLVM 20 and earlier can't handle.
> Simple kernels might still work with an older `llc`, but anything Hopper / Blackwell needs 21+.

#### Clang (host `cuda-bindings`)

The host `cuda-bindings` crate runs `bindgen`, which loads libclang and needs
clang's own resource-dir `stddef.h` — a bare `libclang1-*` runtime is not
enough.

```bash
sudo apt install clang-21   # or libclang-common-21-dev
```

`cargo oxide doctor` catches this up front; the symptom otherwise is a cryptic
`'stddef.h' file not found` during the host build.

#### Dev Container

The repository includes a standard devcontainer setup in `.devcontainer/` for a
reproducible CUDA, LLVM, Clang, and Rust environment. See the
[installation chapter](cuda-oxide-book/getting-started/installation.md#dev-container)
for editor and CLI usage.

### Verifying Installation

```bash
# Check that all prerequisites are in place
cargo oxide doctor

# Build and run an example end-to-end
cargo oxide run vecadd

# Run the same example under NVIDIA Compute Sanitizer
cargo oxide sanitize vecadd
```

`cargo oxide doctor` validates your Rust toolchain, CUDA toolkit, LLVM, and
codegen backend. If everything is configured correctly, `cargo oxide run vecadd`
compiles a Rust kernel to PTX, launches it on the GPU, and prints
`✓ SUCCESS: All 1024 elements correct!`.

## Examples

**190+ examples** in `crates/rustc-codegen-cuda/examples/`. Highlights:

| Example              | Description                                                              |
|----------------------|--------------------------------------------------------------------------|
| `vecadd`             | Vector addition -- canonical first example                               |
| `host_closure`       | Generic kernels with closures passed from host                           |
| `generic`            | Generic kernels with monomorphization (`scale<T>`)                       |
| `ord_cmp`            | Device-side `Ord::cmp` lowering for signed and unsigned integers         |
| `gemm_sol_final`     | Canonical Blackwell GEMM SoL: size-specialized CLC + cg2 + vector stores |
| `gemm_sol`           | Historical GEMM kernel progression and comparison kernels                |
| `tcgen05`            | Blackwell tensor cores (sm_100a): TMEM, MMA, cta_group::2                |
| `atomics`            | GPU atomics: 6 types x 3 scopes x 5 orderings (20 tests)                 |
| `atomic_f16`         | Scalar f16 atomics: per-scope correctness checks + f32 vs f16 bench      |
| `cluster`            | Thread Block Clusters + DSMEM ring exchange (Hopper+)                    |
| `async_mlp`          | Async MLP pipeline: GEMM → MatVec → ReLU across concurrent streams       |
| `mathdx_ffi_test`    | cuFFTDx thread-level FFT + cuBLASDx block-level GEMM                     |
| `device_ffi_test`    | Device FFI: Rust kernels calling C++ CCCL warp-level reductions via LTOIR|
| `async_vecadd`       | Async GPU execution with `cuda-async` and `DeviceOperation`              |
| `cross_crate_kernel` | Library crates defining kernels, bundled into binaries                   |
| `cuda_module_in_lib` | `#[cuda_module]` in a library crate, loaded by embedded bundle name      |

```bash
cargo oxide run vecadd
cargo oxide run gemm_sol_final
```

## Crate Overview

### User-Facing Crates

| Crate               | Description                                                               |
|---------------------|---------------------------------------------------------------------------|
| `cuda-device`       | Device intrinsics (`thread::*`, `warp::*`, barriers)                      |
| `cuda-intrinsics`   | Generated low-level CUDA intrinsic declarations                           |
| `cuda-host`         | Typed module loading, launch helpers, LTOIR loader                        |
| `cuda-macros`       | Proc macros (`#[cuda_module]`, `#[kernel]`, `gpu_printf!`)                |
| `cuda-bindings`     | Raw `bindgen` FFI bindings to `cuda.h`                                    |
| `cuda-core`         | Safe RAII wrappers (`CudaContext`, `CudaStream`, `DeviceBuffer<T>`, ...)  |
| `cuda-async`        | Async execution layer (`DeviceOperation`, `DeviceFuture`, `DeviceBox<T>`) |
| `libnvvm-sys`       | `dlopen` bindings to libNVVM (used by `cuda-host::ltoir`)                 |
| `nvjitlink-sys`     | `dlopen` bindings to nvJitLink (used by `cuda-host::ltoir`)               |
| `ptx-parse`         | Lossless structural views over PTX source text                            |

### Compiler Crates

| Crate                | Description                                           |
|----------------------|-------------------------------------------------------|
| `rustc-codegen-cuda` | Custom rustc backend                                  |
| `mir-importer`       | Rust MIR -> `dialect-mir` translation + pipeline      |
| `mir-lower`          | `dialect-mir` -> LLVM dialect lowering                |
| `dialect-mir`        | pliron dialect modelling Rust MIR                     |
| `dialect-iket`       | pliron dialect modelling in-kernel event tracing      |
| `iket-lower`         | `dialect-iket` profiles + instrumentation lowering    |
| `llvm-export`        | pliron-llvm shim + textual `.ll` exporter             |
| `dialect-nvvm`       | pliron dialect modelling NVVM intrinsics              |
| `dialect-ptx`        | pliron dialect modelling structured PTX               |
| `mir-transforms`     | Optimization passes over the MIR dialect (loop unroll, ...) |
| `nvvm-transforms`    | Target-aware LLVM dialect legalization for NVVM      |
| `cuda-oxide-codegen` | Experimental rustc-independent PTX backend           |

### Build Tooling

| Crate                     | Description                                                    |
|---------------------------|----------------------------------------------------------------|
| `cargo-oxide`             | Cargo subcommand (`cargo oxide run`, etc.)                     |
| `cuda-intrinsics-gen`     | Extractor and deterministic source generator for the intrinsics |
| `cuda-artifact-finalizer` | Driver-independent NVVM IR, LTOIR, and PTX finalization         |
| `oxide-artifacts`         | Architecture-neutral embedded device artifact metadata         |
| `reserved-oxide-symbols`  | Workspace-private `cuda_oxide_*` symbol-name contract          |
| `fuzzer`                  | Differential codegen fuzzer support (rustlantis adapter)       |

### Documentation

| Directory           | Description                                                        |
|---------------------|--------------------------------------------------------------------|
| `cuda-oxide-book`   | Project book (Sphinx + MyST) — guides, compiler internals, API ref |

## Status

### Highlights:

- End-to-end Rust -> PTX compilation
- Unified single-source compilation (host + device in one file)
- Generic functions with monomorphization
- Closures with captures (move and non-move via HMM)
- User-defined structs, enums, pattern matching
- Full GPU intrinsic support (thread, warp, shared memory, barriers, TMA, clusters, atomics)
- Cross-crate kernels
- LTOIR generation for Blackwell+ (device-side LTO)
- Device FFI: Rust <-> C++/CCCL interop via LTOIR
- MathDx integration: cuFFTDx thread-level FFT, cuBLASDx block-level GEMM
- Tile interop (experimental): [`cutile_inter_kernel`](crates/rustc-codegen-cuda/examples/cutile_inter_kernel/README.md) chains a cutile-rs Tile kernel and a cuda-oxide SIMT PTX kernel on the same CUDA stream over shared device tensors. Intra-kernel Tile interop is work in progress and tracked in [#96](https://github.com/NVlabs/cuda-oxide/issues/96).
- Host runtime: `cuda-core` (explicit control, pinned host transfers) and `cuda-async` (composable async operations)
- Canonical Blackwell GEMM SoL example with size-specialized M256xN256/M512xN256 CLC + cta_group::2 kernels and vectorized epilogues (see `gemm_sol_final`)

## Documentation

**WIP:** 🚧 The **[cuda-oxide book](https://nvlabs.github.io/cuda-oxide/)** is the primary reference for the project. It covers SIMT kernel authoring in Rust, synchronous and asynchronous GPU programming, the compiler architecture, and more.

To build and serve the book locally, see [cuda-oxide-book/README.md](./cuda-oxide-book/README.md).

## Ecosystem

cuda-oxide is one of several Rust + GPU efforts under active development. Projects in this space address different parts of the problem — Vulkan/SPIR-V for graphics, implicit offload via LLVM, third-party CUDA backends, safe driver bindings — and we've been working with maintainers across the broader Rust GPU community on how to move GPU computing in Rust forward together. For where cuda-oxide fits relative to other projects, see the [Ecosystem appendix](https://nvlabs.github.io/cuda-oxide/appendix/ecosystem.html) of the book.

## License

cuda-oxide is licensed under the Apache License, Version 2.0: [LICENSE](LICENSE).
Third-party components retain the licenses stated in their files; see
[dependency-licenses.csv](dependency-licenses.csv) for the tracked license inventory.
