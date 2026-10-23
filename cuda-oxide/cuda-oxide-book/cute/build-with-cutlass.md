# Build with the CUTLASS Compiler

CuTe kernels compile through NVIDIA's CUTLASS 4.7 compiler library, not through the
native LLVM path. `cargo oxide` installs the pinned release once, checks its digest on
every load, and embeds the resulting cubin in your binary.

```text
Rust kernel
  ──►  one Pliron module: dialect-cute, dialect-mir, dialect-nvvm
  ──►  whole-module verification and the target gate
  ──►  CUTLASS MLIR text
  ──►  libCutlassCompiler 4.7.0
  ──►  cubin embedded in the host binary
```

## Install once

```bash
cargo oxide toolchain install cutlass
```

- Downloads the CUTLASS 4.7.0 release archive from GitHub and verifies the archive and
  library SHA-256 digests. The bundle is under the NVIDIA EULA; the installer prints
  the notice.
- Linux x86_64 with CUDA 13 only. The library lands in Cargo's data directory under
  `cuda-oxide/toolchains/cutlass/4.7.0/x86_64-cu13`.
- `CUDA_OXIDE_CUTLASS_COMPILER=/abs/path/libCutlassCompiler.so` overrides discovery.
  The digest check still applies, so only the byte-identical 4.7.0 library loads.

## Build and run

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir cargo oxide run elementwise_cute --arch sm_120a
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir cargo oxide build fp16_gemm_256x352_cute --arch sm_100a
```

- Every CuTe kernel needs `CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir`. The native backend
  has no CuTe lowering and stops with a verification error.
- Two targets: `sm_100a` and `sm_120a`. Anything else is rejected before compilation
  and the error names the target. `tcgen05` operations need `sm_100a`; MXF4 register
  MMA needs `sm_120a`.
- `cargo oxide run` without `--arch` uses the detected GPU, so an RTX 5090 gets
  `sm_120a`. `cargo oxide build` needs `--arch`.
- Each example README states its device and shape requirements. The
  [support matrix](support-matrix.md) lists what the two targets cover.

## Options

| Variable | Purpose |
| --- | --- |
| `CUDA_OXIDE_DEVICE_BACKEND` | `native` (default) or `cutlass-mlir`; other values fail |
| `CUDA_OXIDE_MLIR_PROFILE` | `cutlass-full-cute-mlir22` (default) or `cutlass-primitives-mlir22` for kernels that use `ctm-rs` alone; part of the build fingerprint |
| `CUDA_OXIDE_CUTLASS_OPT_LEVEL` | `2` or `3` (default); the attention examples compare at `2`; part of the build fingerprint |
| `CUDA_OXIDE_MLIR_OUTPUT` | Path for the prepared MLIR text, for example `$PWD/module.mlir` |
| `CUDA_OXIDE_CUTLASS_COMPILER` | Absolute path to a `libCutlassCompiler.so` with the pinned digest |

`CUDA_OXIDE_BACKEND` selects the rustc backend library, not the device path. Keep MLIR
dumps and cubins out of version control.

## Failures

- Unsupported operations, rejected targets, verification errors and compiler errors
  stop the build. There is no fallback to another backend.
- The compiler checks layouts, operand roles, pipeline state and the launch ABI. It
  does not prove thread convergence, barrier phases, asynchronous completion or
  pointer lifetimes; see [Checks and Obligations](contracts.md).
