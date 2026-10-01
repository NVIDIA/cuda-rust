# Building from Source

This appendix walks through setting up a development environment for cuda-oxide
from a fresh checkout. If you just want to run an example, the
[Writing Your First Kernel](../getting-started/hello-gpu.md) chapter is faster.

---

## Requirements

| Dependency       | Version                       | Purpose                                                     |
|:-----------------|:----------------------------- |:------------------------------------------------------------|
| **Rust nightly** | `nightly-2026-08-28` (pinned) | Compiler toolchain with `rustc-dev` for the codegen backend |
| **CUDA Toolkit** | 13.0+ (with cuRAND headers)   | Driver API, `nvcc`, PTX assembler; `curand.h` for bindgen   |
| **Clang**        | 21+ (`clang-21` pkg)          | `bindgen` in host `cuda-bindings` needs clang's headers     |
| **Linux**        | Tested on Ubuntu 24.04        | Windows and macOS are not supported                         |
| **GPU**          | sm_80, sm_90, sm_100a         | Hardware target                                             |

## Clone the repository

```bash
git clone https://github.com/NVlabs/cuda-oxide.git
cd cuda-oxide
```

## Install the Rust toolchain

The repo ships a `rust-toolchain.toml` that pins the exact nightly version and
components. Rustup picks it up automatically:

```toml
# rust-toolchain.toml (already in the repo root)
[toolchain]
channel = "nightly-2026-08-28"
components = ["rust-src", "rustc-dev", "rust-analyzer", "clippy", "rustfmt", "llvm-tools"]
```

If you need to install manually:

```bash
rustup toolchain install nightly-2026-08-28
rustup component add rust-src rustc-dev rust-analyzer clippy rustfmt llvm-tools --toolchain nightly-2026-08-28
```

`rust-src` provides the standard library source for cross-compilation,
`rustc-dev` exposes compiler internals that the codegen backend links against,
and `llvm-tools` installs the toolchain-bundled `llc` used for PTX generation
(also required by `cargo oxide doctor`). The other three are not needed to
build: `rust-analyzer` powers IDE support, `clippy` is the lint gate CI runs,
and `rustfmt` backs `cargo oxide fmt`.

## Install CUDA

Make sure the CUDA toolkit is on your `PATH`:

```bash
export PATH="/usr/local/cuda/bin:$PATH"
nvcc --version   # should print 13.x or later
```

If you are building on a system without a GPU (e.g. CI), the toolkit is still
required for `ptxas` and header files, but you will not be able to run kernels.

## Install LLVM (usually optional)

The codegen pipeline emits LLVM IR and invokes `llc` to produce PTX. The
pinned Rust toolchain (`nightly-2026-08-28`) already ships LLVM 23 with the
NVPTX backend enabled via the `llvm-tools` component, so the recommended
path is:

```bash
rustup component add llvm-tools
```

The component is already listed in `rust-toolchain.toml`, so on a fresh
clone rustup installs it automatically; running the command above is the
one-shot fix for older clones. The pipeline auto-detects this `llc` at
`<sysroot>/lib/rustlib/<host>/bin/llc`.

If you would rather use a system LLVM (for a specific patch level, or
because you already have one installed), the pipeline falls back to
`llc-23` / `llc-22` / `llc-21` on `PATH`. LLVM 21 is the minimum — earlier releases
reject the TMA / tcgen05 / WGMMA intrinsic signatures that cuda-oxide
emits.

```bash
# Ubuntu / Debian
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

You should see `nvptx64 - NVIDIA PTX 64-bit` in the target list.

To pin a specific binary (rustup's, a distro's, or a custom build), set
`CUDA_OXIDE_LLC=/path/to/llc`. The pipeline's full resolution order is:

1. `$CUDA_OXIDE_LLC` (if set)
2. The Rust toolchain's `llvm-tools` llc
3. `llc-23`, then `llc-22`, then `llc-21`, then bare `llc` on `PATH`

```{note}
Older `llc` binaries (LLVM 20 and earlier) will compile simpler kernels when
pointed at via `CUDA_OXIDE_LLC=/path/to/llc-20`, but any example that uses
modern TMA / tcgen05 / WGMMA intrinsics (`tma_copy`, `gemm_sol`,
`tcgen05_matmul`, `wgmma`, …) will fail with
`Intrinsic has incorrect argument type!` until you upgrade to LLVM 21+.
```

## Install Clang (for host `cuda-bindings`)

The host `cuda-bindings` crate runs `bindgen`, which loads libclang and needs
clang's own resource-dir `stddef.h` — a bare `libclang1-*` runtime is not
enough.

```bash
sudo apt install clang-21   # or libclang-common-21-dev
```

`cargo oxide doctor` verifies this up front.

## Build the workspace

The main workspace contains the user-facing crates (`cuda-device`, `cuda-host`,
`cuda-macros`, etc.) and the build tooling (`cargo-oxide`). The host runtime
(`cuda-bindings`, `cuda-core`, `cuda-async`) is shared with cutile-rs and comes
from crates.io:

```bash
cargo build
```

```{note}
The codegen backend (`cuda-oxide/crates/rustc-codegen-cuda/`) is intentionally **not** a
workspace member because it requires special nightly features and a different
build process. `cargo-oxide` handles building it transparently.
```

## Install cargo-oxide

`cargo-oxide` is the cargo subcommand that drives the full compilation
pipeline. Inside the repo, it works via a workspace alias. For standalone use, install it with the pinned nightly toolchain:

```bash
cargo +nightly-2026-08-28 install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide
```

On first run, `cargo-oxide` automatically fetches and builds the codegen backend
dylib.

## Verify the installation

```bash
# Check all prerequisites
cargo oxide doctor

# Compile and run the canonical first example
cargo oxide run vecadd
```

`cargo oxide doctor` validates your Rust toolchain, CUDA toolkit (including
libNVVM / nvJitLink / libdevice for kernels that use math intrinsics), LLVM
installation, and codegen backend. If everything is configured correctly,
`cargo oxide run vecadd` compiles a Rust kernel to PTX, launches it on the GPU,
and prints a success message.

## Common commands

```bash
# Build and run an example
cargo oxide run <example>

# Print generated PTX only
cargo oxide inspect <example>

# Show the full compilation pipeline (MIR → LLVM IR → PTX)
cargo oxide pipeline <example>

# Remove local build outputs and generated artifacts
cargo oxide clean

# Run under NVIDIA Compute Sanitizer
cargo oxide sanitize <example> --tool memcheck

# Debug with cuda-gdb
cargo oxide debug <example> --tui

# Build NVVM IR for libNVVM/nvJitLink interop
cargo oxide build <example> --emit-nvvm-ir --arch sm_120
```

## Building the book

The documentation lives in `cuda-oxide-book/` and uses Sphinx with MyST
Markdown. To build and serve locally:

```bash
cd cuda-oxide/cuda-oxide-book
make setup      # creates venv, installs dependencies
source .venv/bin/activate
make livehtml   # starts dev server on http://localhost:8000
```

## Generating API documentation

Standard `cargo doc` works for the workspace crates:

```bash
cargo doc --no-deps --open
```

This generates rustdoc for `cuda-device`, `cuda-host`, `cuda-macros`, and all
other workspace members. The codegen backend is excluded since it is not a
workspace member, and the shared host runtime (`cuda-core`, `cuda-async`) is
documented from cutile-rs.

## Workspace structure

```text
repository/
├── Cargo.toml              # Workspace root
├── rust-toolchain.toml     # Pinned nightly + components
├── cutile-rs/              # Tile DSL (nested import)
└── cuda-oxide/
    ├── crates/
    │   ├── cuda-device/          # Device intrinsics (#![no_std])
    │   ├── cuda-host/            # Host launch APIs
    │   ├── cuda-macros/          # Proc macros (#[kernel], #[device], gpu_printf!)
    │   │                         # (cuda-bindings, cuda-core, cuda-async come from
    │   │                         #  NVlabs/cutile-rs; SIMT API under their simt modules)
    │   ├── cargo-oxide/          # Cargo subcommand
    │   ├── rustc-codegen-cuda/   # Codegen backend (not a workspace member)
    │   ├── mir-importer/         # MIR → Pliron IR translation
    │   ├── mir-lower/            # `dialect-mir` → LLVM dialect lowering
    │   ├── dialect-mir/          # pliron dialect modelling Rust MIR
    │   ├── llvm-export/          # shim re-exporting pliron-llvm + textual .ll export
    │   ├── dialect-nvvm/         # NVVM intrinsics dialect
    │   ├── libnvvm-sys/          # dlopen bindings to libNVVM
    │   ├── nvjitlink-sys/        # dlopen bindings to nvJitLink
    │   ├── reserved-oxide-symbols/ # Shared naming contract
    │   └── fuzzer/               # Differential testing support
    └── cuda-oxide-book/        # This book (Sphinx + MyST)
```

## Troubleshooting

`llc` not found or missing NVPTX
: The fastest fix is `rustup component add llvm-tools` — the pinned
  toolchain's `llc` is LLVM 23 with NVPTX enabled and is auto-picked up.
  Otherwise install a system LLVM 21+ (`sudo apt install llvm-21`); the
  pipeline probes the rustup `llc` first, then `llc-23` → `llc-22` →
  `llc-21` on `PATH`. To pin a specific binary set
  `CUDA_OXIDE_LLC=/path/to/llc`.

`Intrinsic has incorrect argument type!` (from `llc`)
: Your `llc` is older than LLVM 21 and cannot lower the modern TMA / tcgen05
  / WGMMA intrinsic signatures. Install `llvm-21` and re-run.

`error[E0463]: can't find crate for rustc_middle`
: You are missing the `rustc-dev` component. Run:
  `rustup component add rustc-dev --toolchain nightly-2026-08-28`.

CUDA driver version mismatch
: The toolkit version (compile-time) and driver version (runtime) must be
  compatible. Run `nvidia-smi` to check the driver version, and
  `nvcc --version` for the toolkit.

`cargo oxide doctor` fails on codegen backend
: The backend is built on first use. If the build fails, check that
  `rust-src` is installed and that the nightly version matches
  `rust-toolchain.toml`.
