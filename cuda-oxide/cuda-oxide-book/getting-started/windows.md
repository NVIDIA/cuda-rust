# Windows Setup

`cuda-oxide` supports `x86_64-pc-windows-msvc` on Windows 10 22H2 and Windows 11.
For Linux, see [Installation](installation.md).

## Requirements

- NVIDIA GPU and a driver compatible with the installed CUDA Toolkit.
- [CUDA Toolkit 13.0+](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/) (including `nvcc`, `libNVVM`, `nvJitLink`, and `cuRAND` headers).
- Visual Studio 2022 or newer with MSVC x64 Build Tools, Windows SDK, and C++ Clang tools (`libclang.dll` and `lld-link.exe`).
- Rust 1.99.1 (`x86_64-pc-windows-msvc`) with `rust-src`, `rustc-dev`, `rust-analyzer`, `rustfmt`, `clippy`, and `llvm-tools`.
- `libffi:x64-windows` installed via `vcpkg`.

Install the pinned Rust toolchain and components:

```powershell
rustup toolchain install 1.99.1
rustup component add rust-src rustc-dev rust-analyzer rustfmt clippy llvm-tools --toolchain 1.99.1
```

## CUDA and LLVM Environment

Point `CUDA_PATH` / `CUDA_TOOLKIT_PATH` at your installed CUDA Toolkit and `LIBCLANG_PATH` at your LLVM/Clang `bin` directory:

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
$env:CUDA_TOOLKIT_PATH = $env:CUDA_PATH
$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"
$env:PATH = "$env:CUDA_PATH\bin\x64;$env:CUDA_PATH\bin;$env:LIBCLANG_PATH;$env:PATH"
```

## Build from Source

Open a Developer PowerShell for VS 2022 (x64) and install `libffi:x64-windows` with `vcpkg`:

```powershell
$env:VCPKG_ROOT = "C:\vcpkg"
& "$env:VCPKG_ROOT\vcpkg.exe" install libffi:x64-windows
cd cuda-oxide
cargo install --locked --path crates/cargo-oxide
cargo oxide setup
cargo oxide doctor
cargo oxide run vecadd
```

For a build-only check without a GPU, run `scripts/smoketest.ps1 -BuildOnly` from `cuda-oxide/`:

```powershell
.\scripts\smoketest.ps1 -BuildOnly
```

## Compute Sanitizer on Windows (WDDM)

When running `cargo oxide sanitize` on a Windows GPU using the WDDM driver model, enable the WDDM debugger interface once from an elevated Administrator prompt:

```powershell
& "$env:CUDA_PATH\compute-sanitizer\EnableDebuggerInterface.bat"
```
