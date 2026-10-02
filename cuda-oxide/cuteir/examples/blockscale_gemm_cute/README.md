# Block-scaled MXFP4 GEMM

GEMM means “matrix times matrix”:

```text
C[M,N] = A[M,K] × B[K,N]

M = 2048    N = 2048    K = 1024
input values = packed E2M1 (FP4)
input scale  = one UE8M0 value per 32 K values
accumulator  = f32
output       = f16
```

The shared scale moves each group of small FP4 values into a useful numerical
range. The scale is applied as part of the matrix multiply.

The output is split into `128 × 128` tiles. A thread block, also called a CTA,
works on one output tile at a time.

The four hardware terms used below are:

- **TMA**, or Tensor Memory Accelerator, is a hardware engine that copies a
  whole multidimensional tile;
- **shared memory** is fast storage shared by the threads in one CTA;
- **MMA** is the tensor-core matrix multiply-and-accumulate operation;
- an **epilogue** converts and arranges completed accumulators for output.

## The Tensor flow

This example follows the same shape as a large CuTe kernel:

```text
global A Tensor ─┐
global B Tensor ─┼── TMA tile copies ──► shared-memory Tensors
A/B scale Tensors┘                              │
                                               ▼
                                      per-warp tiled views
                                               │
                                      load A/B/scale fragments
                                               │
                                               ▼
                                      block-scaled MMA
                                               │
                                               ▼
                                      f32 accumulator tile
                                               │
                                      tiled f16 epilogue
                                               │
                                               ▼
                                      shared output Tensor
                                               │
                                         TMA tile stores
                                               │
                                               ▼
                                        global C Tensor
```

## How work moves through the kernel

The kernel uses three shared-memory slots:

```text
slot 0       slot 1       slot 2
load here → compute here → waiting here → repeat
```

One producer warp issues the TMA loads. Eight compute warps consume completed
slots and update their accumulator tiles. While the compute warps use one slot,
TMA can fill another.

After a CTA finishes an output tile, a small scheduler may give it another
tile:

```text
physical CTA 0: logical tile 0, then tile 0 + number_of_CTAs, ...
physical CTA 1: logical tile 1, then tile 1 + number_of_CTAs, ...
```

This is called persistent scheduling. It changes which CTA owns a tile; it
does not change the matrix calculation.

The Rust types keep the important agreements together. For example, a
`TmaDesc<T, Layout>` built on the host must name the same shared-memory layout
as the device copy that consumes it. Matrix A and matrix B also carry different
operand-role types, so they cannot be silently swapped in a tiled MMA call.

## Device requirement

The packed MXFP4 tensor-core instruction used here requires an SM 12.x
Blackwell target (`sm_120` or `sm_121`). Build the example for the accelerated
target `sm_120a`.

On another GPU, the program skips device execution. Numerical GEMM validation
requires a compatible SM 12.x GPU.

## Build and run it

From the `cuda-oxide/` directory, install the pinned official CUTLASS 4.7 compiler
once:

```bash
cargo oxide toolchain install cutlass
```

Build or run this example through the translation backend:

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
  cargo oxide build blockscale_gemm_cute --arch sm_120a

CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
  cargo oxide run blockscale_gemm_cute --arch sm_120a
```

The program builds deterministic packed inputs, computes a host reference,
runs the GPU kernel, and compares every output bit pattern. The host reference
for the fixed `2048 × 2048 × 1024` problem is intentionally substantial, so a
checked run is not instant.

A successful device run confirms all 4,194,304 `f16` output bit patterns. It
then performs 100 warmup launches and measures 31 individual launches with
CUDA events. The report includes the median, p10, and p90 device time and the
median-derived TFLOP/s for `2 × M × N × K` floating-point operations.

## Single-kernel comparison

Measured on 2026-08-18 at commit `aafe6e47fb4e` on an RTX 5090 (`sm_120`),
Nsight Systems 2026.1.3 reported these kernel-active medians for `M=N=2048`,
`K=1024`, `L=1`:

| Implementation | Median | Median-derived rate |
| :--- | ---: | ---: |
| CUTLASS 4.7 translation | **11.264 µs** | 762.60 TFLOP/s |
| CuTeDSL 4.6.2 | **10.272 µs** | 836.25 TFLOP/s |

Translation is 0.992 µs, or 9.66%, longer. The measured translation cubin is
29,608 bytes with SHA-256
`69e1f1aad53a5889a40f6d7bfce0c6f66607eaeb1119640aa94f8ed3467dfbee`.
Translation matched all 4,194,304 expected `f16` bit patterns; the separate
CuTeDSL reference-checking run also passed.

This is a direct hot-reuse comparison with serial submission and no CUDA
graphs. Both implementations ran in one process with 100 warmups and 31
one-kernel samples; translation also performed one correctness launch. Nsight
measured only main-kernel active time, excluding setup and conversion kernels.

This is the full semantic acceptance example:

```text
Tensor layouts → tiled copies → pipeline → tiled MMA → output Tensor
```

The backend receives the same high-level module rather than a second kernel
description.
