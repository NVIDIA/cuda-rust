# Build with the CUTLASS Compiler

CuTe kernels compile through NVIDIA's CUTLASS 4.8.1 compiler library rather than the
native LLVM path. `cargo oxide` installs the pinned release once, verifies its SHA-256
digest on every load, and embeds the resulting cubin straight into your Rust binary.

```text
Rust kernel (`cute-rs`)
  ──►  one Pliron module: dialect-cute, dialect-mir, dialect-nvvm
  ──►  whole-module verification and the target gate
  ──►  exporter profile (`cuda-oxide-mlir-export`)
  ──►  libCutlassCompiler 4.8.1 (`x86_64-cu13` or `aarch64-cu13`)
  ──►  cubin embedded in the host binary
```

## Install once

```bash
cargo oxide toolchain install cutlass
```

- Downloads the official CUTLASS 4.8.1 release archive from GitHub and verifies both the
  archive and `libCutlassCompiler.so` SHA-256 digests. The bundle is covered by the NVIDIA
  EULA, which the installer prints before downloading.
- Supports both **Linux `x86_64`** and **Linux `aarch64`** (including Grace-Blackwell and
  Vera-Rubin ARM nodes) with CUDA 13. The library lands in Cargo's data directory under
  `cuda-oxide/toolchains/cutlass/4.8.1/{x86_64-cu13,aarch64-cu13}`.
- `CUDA_OXIDE_CUTLASS_COMPILER=/abs/path/libCutlassCompiler.so` overrides discovery. The
  digest check still runs, so only the byte-identical 4.8.1 library loads.

## How CUTLASS upgrades stay compatible

CUTLASS's internal MLIR ops are not a stable public IR: minor CUTLASS releases routinely
rename op properties (such as `mmaKind` to `kind` or `scaleVecSize` to `blockScale`).

That is why `cargo oxide` pairs the SHA-pinned compiler with an **exporter profile** in
`cuda-oxide-mlir-export`:
- Your Rust kernel compiles into `dialect-cute` and `dialect-nvvm`, which are owned by
  `cuda-oxide` and stay stable across CUTLASS bumps.
- All CUTLASS-version-specific MLIR spelling lives inside the exporter profile, so moving
  from CUTLASS 4.7 to 4.8.1 requires **zero edits** to existing Rust kernels.
- Host crates (`cuda-core`, `cuda-bindings`, `cuda-async`) have zero dependency on
  `cute-rs` or the CUTLASS compiler and are versioned independently.

## Build and run

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir cargo oxide run elementwise_cute --arch sm_120a
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir cargo oxide build fp16_gemm_256x352_cute --arch sm_100a
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir cargo oxide run sm107_mxfp8_gemm_cute --arch sm_107a
```

- Every CuTe kernel needs `CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir`. The native backend
  has no CuTe lowering and stops with a verification error.
- Four validated targets: `sm_100a` (Blackwell B200), `sm_103a` (Blackwell Ultra B300),
  `sm_107a` (Vera Rubin), and `sm_120a` (RTX 50 series). Anything else is rejected before
  compilation with an error naming the target. Within that set, `tcgen05` operations need
  `sm_100a`, `sm_103a`, or `sm_107a`; SM107 `MXF8F6F4` block-scaled MMA needs `sm_107a`;
  and MXF4 register MMA needs `sm_120a`.
- `cargo oxide run` without `--arch` uses the detected GPU, so an RTX 5090 selects
  `sm_120a` and a Vera Rubin GPU selects `sm_107a`. `cargo oxide build` needs `--arch`.
- Each example README states its device and shape requirements. The
  [support matrix](support-matrix.md) lists what each target covers.

## Options

| Variable                        | Purpose                                                                          |
|:--------------------------------|:---------------------------------------------------------------------------------|
| `CUDA_OXIDE_DEVICE_BACKEND`     | `native` (default) or `cutlass-mlir`; other values fail                          |
| `CUDA_OXIDE_MLIR_PROFILE`       | `cutlass-full-cute-mlir22` (default) or `cutlass-primitives-mlir22` (`*_ctm`)    |
| `CUDA_OXIDE_CUTLASS_OPT_LEVEL`  | `2` or `3` (default); included in the build fingerprint                          |
| `CUDA_OXIDE_MLIR_OUTPUT`        | Path for the prepared MLIR text, for example `$PWD/module.mlir`                  |
| `CUDA_OXIDE_CUTLASS_COMPILER`   | Absolute path to a `libCutlassCompiler.so` matching the pinned 4.8.1 digest      |

`CUDA_OXIDE_BACKEND` selects the rustc backend library, not the device path. Keep MLIR
dumps and cubins out of version control.

## Failures

- Unsupported operations, rejected targets, verification errors, and compiler errors
  stop the build. There is no fallback to another backend.
- The compiler checks layouts, operand roles, pipeline state, and the launch ABI. It
  does not prove thread convergence, barrier phases, asynchronous completion, or
  pointer lifetimes; see [Checks and Obligations](contracts.md).
