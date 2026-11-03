# libnvvm-sys

Runtime (`dlopen`) bindings to NVIDIA's libNVVM. libNVVM is the front-end of NVIDIA's PTX-targeting compiler; it accepts NVVM IR (an LLVM-IR dialect) and produces PTX or LTOIR.

## What this crate provides

- `LibNvvm` — RAII wrapper around the loaded library + resolved function pointers.
- `Program` — RAII wrapper around an `nvvmProgram` handle, with
  `add_module` / `verify` / `compile` methods.
- `CudaArch` — strict target parser that renders `compute_*` for libNVVM and
  `sm_*` for cubin-producing tools.
- `LibNvvm::ir_version` — `nvvmIRVersion` query for the accepted NVVM IR and
  debug-metadata versions.
- `LibNvvm::llvm_version` — optional `nvvmLLVMVersion` query for target-specific
  NVVM IR dialect selection on toolkits that expose it.
- `NvvmError` — typed errors with the libNVVM error log captured.

## Build requirements

None. The library is loaded at runtime, so the CUDA Toolkit only needs to be present when the program runs (not when it compiles).

## Library discovery

`LibNvvm::load()` tries (in order):

1. `LIBNVVM_PATH` env var, if set.
2. `<root>/nvvm/lib64/libnvvm.so` for `<root>` in `CUDA_TOOLKIT_PATH`, `CUDA_HOME`, `CUDA_PATH`, `/usr/local/cuda`, `/opt/cuda`.
3. The system loader (`libnvvm.so.4`, `libnvvm.so.3`, `libnvvm.so`).

libNVVM ships with the standard CUDA Toolkit at `<cuda>/nvvm/lib64/`. No separate download.

## Usage

This crate is low-level. Most users want the higher-level `cuda_host::ltoir::load_kernel_module` helper, which combines libNVVM + libdevice + nvJitLink behind one call. Use this crate directly only if you need explicit control over the libNVVM compile.

```rust
use libnvvm_sys::{LibNvvm, Program};

let nvvm = LibNvvm::load()?;
let mut program = Program::new(&nvvm)?;
program.add_module(&libdevice_bytes, "libdevice.10.bc")?;
program.add_module(&kernel_ll_bytes, "kernel.ll")?;
program.verify(&["-arch=compute_120"])?;
let ltoir = program.compile(&["-arch=compute_120", "-gen-lto"])?;
```

## Companion crate

[`nvjitlink-sys`](../nvjitlink-sys/) — same pattern, for nvJitLink. Together they cover the NVVM IR → LTOIR → cubin pipeline.
