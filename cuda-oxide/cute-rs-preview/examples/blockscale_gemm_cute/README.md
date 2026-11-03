# Block-scaled MXFP4 GEMM

This cuda-oxide example uses `cute-rs` to multiply matrices with packed FP4
values and a shared scale for each group of 32 values:

```text
C[M,N] = A[M,K] × B[K,N]
M = 2048, N = 2048, K = 1024
inputs: E2M1 (FP4) + UE8M0 scales
accumulation: f32; output: f16
```

The scale moves each group of small FP4 values into a useful numerical range.
It is applied as part of the matrix multiply. The output is split into
`128 × 128` tiles, each owned by one thread block (CTA).

## CuTe data flow

```text
global A/B + scale Tensors
          │ TMA tile copies
          ▼
shared-memory Tensors → per-warp tiled views
          │ load input and scale fragments
          ▼
block-scaled MMA → f32 accumulator tile
          │ tiled f16 epilogue
          ▼
shared output Tensor → TMA stores → global C Tensor
```

TMA (Tensor Memory Accelerator) copies whole tiles. MMA is the tensor-core
matrix multiply-and-accumulate operation. One producer warp fills three
shared-memory slots while eight compute warps consume completed slots.
A persistent scheduler lets each CTA process further output tiles.

Typed layouts connect the host's TMA descriptors to device copies. A and B
also carry distinct operand-role types through the tiled MMA operation.
The compiler lowers this CuTe computation directly.

## Build and run

This kernel needs an `sm_120a` GPU such as the RTX 5090 and builds for `sm_120a`
only. Other GPUs skip execution and cannot validate this example.
From `cuda-oxide/`, install the pinned CUTLASS 4.7 compiler once, then run:

```bash
cargo oxide toolchain install cutlass

CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 \
  cargo oxide build blockscale_gemm_cute --arch sm_120a

CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 \
  cargo oxide run blockscale_gemm_cute --arch sm_120a
```

The Rust program builds deterministic inputs and computes a substantial CPU
reference. Success means all **4,194,304 FP16 output bit patterns** match.
It then reports CUDA-event median, p10, p90 and TFLOP/s from **100 warmups and
31 single-launch samples**. This command runs Rust only; the
[CuTeDSL harness](reference/README.md) runs the separate comparison and
requires its Python dependencies.

## CuTeDSL comparison

RTX 5090 at `a0d5776b2` (2026-10-18), CUTLASS 4.7.0 at optimization level 2
on both sides. Both programs pass their respective correctness checks.
Each value is the median of three round medians, with 100 warmups and 31
single-launch samples per round, with reused buffers. Times are microseconds:

| M / N / K | CUDA events: Rust | CUDA events: CuTeDSL | Nsight kernels: Rust | Nsight kernels: CuTeDSL |
| --- | ---: | ---: | ---: | ---: |
| 2048 / 2048 / 1024 | 13.536 | 16.832 | 11.360 | 10.176 |

CUDA events include submission gaps; Nsight Systems reports only GEMM kernel
duration, including stalls within the kernel. Profiling runs separately, so
subtracting columns does not isolate host overhead. Rust has the shorter event
interval here; CuTeDSL has the shorter kernel duration.

Clocks were unlocked and Rust ran first. The programs use different input
fixtures; Rust starts a new process per round, while CuTeDSL keeps one process.
These results do not establish a stable speed ranking.

- [Reference sources, dependencies and provenance](reference/README.md)
- [Separate Rust CTM example](../blockscale_gemm_ctm)
- [CUDA C++ primitives counterpart](reference/cpp)
