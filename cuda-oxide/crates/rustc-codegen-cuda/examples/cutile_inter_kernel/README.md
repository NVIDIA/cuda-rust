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
    | threshold_scale_f32(softmax_out -> gated_out)
    v
gated_out tensor
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

The important cutile-rs mechanism is `DeviceOp::then`.

```text
tile_softmax::row_softmax(...)
    .then(|softmax_out| ThresholdScaleKernel { ... })
```

`then` executes both operations with the same cutile-rs `ExecutionContext`.
That context owns the CUDA stream, so the Tile launch and cuda-oxide PTX
launch are submitted to the same stream in order.

The shared arrays are just cutile-rs `Tensor` allocations. The custom SIMT
`DeviceOp` takes the Tile-produced tensor and passes its raw `CUdeviceptr` to
the cuda-oxide kernel:

```text
softmax_out: Tensor<f32>
    |
    | device_pointer().cu_deviceptr()
    v
threshold_scale_f32(..., input_ptr, output_ptr)
```

So the interop contract is deliberately small:

```text
same process
same CUDA context
same CUDA stream
same device allocations
PTX-loaded cuda-oxide kernel launched as a cutile-rs DeviceOp
```
