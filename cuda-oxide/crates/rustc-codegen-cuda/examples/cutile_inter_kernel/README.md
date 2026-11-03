# cutile-rs inter-kernel interop

This example demonstrates the inter-kernel Tile <> SIMT path:

1. A cutile-rs Tile kernel computes row-wise softmax.
2. A cuda-oxide Rust SIMT kernel thresholds and scales the softmax output.
3. Both kernels run in one host process, on one CUDA stream, over cutile-rs
   device tensors.

This is not the future intra-kernel interop path. It is the interop
that works today: separate kernels, shared stream, shared device memory.

## Layout

```text
cutile_inter_kernel/
  Cargo.toml          host runner plus cuda-oxide interop metadata
  src/main.rs         cutile-rs Tile kernel + same-stream SIMT launch
  simt/
    Cargo.toml        cuda-oxide device crate
    src/main.rs       #[kernel] threshold_scale_f32
```

## Pipeline

```text
input tensor
    |
    | cutile-rs Tile kernel
    | row_softmax(input -> softmax_out)
    v
softmax_out tensor
    |
    | cuda-oxide SIMT kernel loaded from generated PTX
    | threshold_scale_f32(softmax_out -> same allocation)
    v
processed tensor (same allocation)
```

The Tile stage is intentionally tile-shaped: it uses row reductions and
`exp`. The SIMT stage is intentionally per-thread and branchy: it applies a
custom threshold/scale post-process to each element.

## Requirements

- cuda-oxide from this repository.
- Network access the first time Cargo fetches the `cutile` crates from crates.io.
- CUDA Toolkit 13.1+ with `nvcc` and `tileiras` available. This example
  defaults `CUDA_TOOLKIT_PATH` to `/usr/local/cuda` through its local Cargo
  config for the cutile-rs CUDA bindings build; set `CUDA_TOOLKIT_PATH`
  yourself if your toolkit lives elsewhere.

`cargo oxide run` targets explicit `--arch` first, then `CUDA_OXIDE_TARGET`,
then auto-detects the local GPU. `cargo oxide build` uses explicit target
settings or the backend default.

## Run

From the cuda-oxide repo root:

```bash
cargo oxide run cutile_inter_kernel
```

`cargo oxide run` reads the interop metadata and does the SIMT compiler step for you:

```text
cargo oxide builds simt/ with rustc-codegen-cuda
    -> produces cutile_inter_kernel_simt.ptx
    -> writes simt/cutile_inter_kernel_simt.ptx
    -> host binary reads it from simt/ at runtime

cargo oxide run cutile_inter_kernel
    -> JITs the Tile kernel with tileiras
    -> loads the generated cuda-oxide PTX
    -> launches both kernels as one DeviceOp chain
```

Expected ending:

```text
PASS: cutile-rs Tile softmax -> cuda-oxide SIMT threshold/scale passed
```

## Same stream, shared arrays

Both stages use `cuda_async::device_operation::DeviceOp` and its shared execution context.
`then` submits them to the same stream in order. The example transfers the Tile
result with `Tensor::into_buffer`, passes it through the shared
`AsyncKernelLaunchBuilder` / `OwnedAsyncKernelLaunch` API, and returns it with
`Tensor::from_buffer`. It needs no custom `DeviceOp` implementation.

The SIMT kernel reads and overwrites each element in place. Assertions check
both allocation identity and device address before and after the transition,
then check every result against the CPU calculation. The only device-to-host
copy is the final verification readback. Launch descriptors retain the shared
owner and write access until GPU completion, including early future drop.
