# Official CUTLASS translation backend

The `cutlass-mlir` backend compiles Rust CuTe kernels with NVIDIA's pinned
CUTLASS 4.7 compiler. It preserves CuTe operations during preparation, maps
them to CUTLASS MLIR, and embeds the resulting CUDA cubin in the program.
Ordinary non-CuTe kernels retain the default MIR/NVVM/LLVM-to-PTX backend.

## Try it

Complete the normal cuda-oxide setup first. Running an example also needs a
compatible NVIDIA GPU and CUDA driver. From the `cuda-oxide/` directory:

```bash
cargo oxide toolchain install cutlass
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
  cargo oxide run elementwise_cute --arch sm_120a
```

The [elementwise example](../examples/elementwise_cute) adds two vectors and
checks every `f32` and `f16` output bit pattern against a host reference,
including partial tiles. Use `cargo oxide build` in place of `run` to compile
without executing the example.

Choose `--arch` for the example and the GPU that will run it:

- `sm_120a`: the NVFP4 GEMV and block-scaled GEMM examples.
- `sm_100a`: the SM100 FP16 GEMM and attention examples.
- Elementwise uses ordinary `f32`/`f16` operations; the command above targets
  an SM120 GPU. Use `sm_100a` when running it on an SM100 GPU.

SM100-specific operations are not available on consumer SM120 GPUs. Each
example's README gives its device and shape requirements.

## What gets translated

Mappings cover tensor tiling and copying, scaled GEMV, scheduled block-scaled
GEMM, and SM100 shared-memory/TMEM MMA pipelines. The compiler checks layouts,
operand roles, and supported profiles before calling CUTLASS.

See [the pliron-to-MLIR flow](pliron-mlir-flow.md) for the stages and diagram,
or [mapping and launch contracts](reference/translation-contracts.md) for the
supported examples and API details.

## Safety and failures

- Unsupported semantics or profiles, verification errors, and compiler
  failures stop the build. There is no fallback to another backend.
- Unsafe callers still own dynamic barrier phases, participating lanes,
  pointer lifetimes, and asynchronous buffer reuse. Verification does not
  prove thread convergence or completion across warps.
- SM100 collective TMA loads require every producer-warp lane to participate.
  See the [SM100 contracts](reference/translation-contracts.md#sm100-contracts)
  before changing a kernel's synchronization.

## Useful options

| Environment variable | Purpose |
| --- | --- |
| `CUDA_OXIDE_DEVICE_BACKEND` | `native` (default) or `cutlass-mlir`; other values fail. |
| `CUDA_OXIDE_CUTLASS_COMPILER` | Absolute path to the compiler library; its pinned digest is still checked. |
| `CUDA_OXIDE_CUTLASS_OPT_LEVEL` | `2` or `3` (default). Attention reference comparisons use `2`. |
| `CUDA_OXIDE_MLIR_OUTPUT` | Path for a prepared MLIR dump, for example `$PWD/module.mlir`. |

`CUDA_OXIDE_BACKEND` selects the rustc backend library, not the device path.
Keep MLIR dumps and generated cubins out of version control. The
[reference](reference/translation-contracts.md#compiler-selection-and-optimization)
includes compiler override commands and optimization fingerprint details.
