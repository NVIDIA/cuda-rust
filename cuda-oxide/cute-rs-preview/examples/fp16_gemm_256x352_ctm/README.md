# FP16 GEMM with Rust CTM — SM100a (B200)

This example computes a matrix product with optional row bias:

```text
C[M, N] = A[M, K] @ B[N, K].T + bias[M, 1]
```

Inputs and output are FP16; accumulation and bias addition use FP32 before
rounding to FP16. M must be a positive multiple of **256**, and N/K of **8**.
The `256x352` name describes an output tile; `--mnk` sets the full matrix size.

## Tile flow

Two CTAs cooperate on each tile, splitting its 352 columns into 192 and 160:

```text
A/B global → TMA → shared [5 stages] → paired MMA → TMEM
C global   ← TMA ← shared [2 stages] ← FP16 + optional row bias
```

`ctm-rs` expresses this schedule through typed instructions and physical formats,
with explicit barriers and storage lifetimes. The Rust kernel uses no CuTe
operations or layout algebra. The [CuTe version](../fp16_gemm_256x352_cute/README.md)
expresses the same schedule with typed layouts, tensor operations and pipelines.

## Run it

You need an SM100 GPU and a compatible CUDA driver. The comparison uses the
checked-in [Python CTM reference](../fp16_gemm_256x352_cute/public_reference/README.md):
CUTLASS instruction primitives with CuTe compilation and epilogue helpers.
From `cuda-oxide/`, once per checkout:

```bash
cargo oxide toolchain install cutlass
python3 -m venv cute-rs-preview/.venv
cute-rs-preview/.venv/bin/pip install -r cute-rs-preview/requirements.txt
```

Then:

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir CUDA_OXIDE_MLIR_PROFILE=cutlass-primitives-mlir22 \
  cargo oxide build fp16_gemm_256x352_ctm --arch sm_100a

cute-rs-preview/.venv/bin/python \
  cute-rs-preview/examples/fp16_gemm_256x352_cute/compare.py \
  --rust-bin cute-rs-preview/examples/fp16_gemm_256x352_ctm/target/release/fp16_gemm_256x352_ctm \
  --mnk 8192,8096,8192 --warmup 2000 --iters 301 --json /tmp/ctm-gemm.json
```

Both outputs are checked against an independent PyTorch oracle on identical
inputs. The command ends with:

```text
CUTLASS (CuTe + CTM): <time> us, <rate> TFLOP/s
ctm-rs: <time> us, <rate> TFLOP/s
```

Add `--has-bias` for row bias. The standalone Rust executable uses CPU
verification and defaults to graph timing; large CPU checks are expensive.
Use the comparison command above for the throughput benchmark.

## B200 performance

Measured **2026-10-18**, CUTLASS 4.7.0, M8192/N8096/K8192, no bias, 74 clusters.
CUDA events report the median of three run medians, each with 2,000 warmups
and 301 single-kernel samples. Nsight Systems shows the median of 301 kernel
durations from a separate capture; TFLOP/s uses the CUDA-event median.

| Implementation | CUDA events (µs) | TFLOP/s | Nsight kernel (µs) |
| --- | ---: | ---: | ---: |
| Rust CTM (`ctm-rs`) | 780.320 | 1392.540 | 774.207 |
| Python CTM (CUTLASS primitives + CuTe helpers) | 770.848 | 1409.651 | 751.392 |

Event intervals can include launch gaps; Nsight durations exclude gaps between
kernels. Clocks were unlocked and Python ran first. These separate runs vary,
so subtracting columns does not isolate overhead or establish a stable ranking.

The Python reference has an unresolved group-two TMA memcheck diagnostic;
see [validation and known limitation](../fp16_gemm_256x352_cute/public_reference/README.md#validation-and-known-limitation).

- [CTM architecture and validation](public_reference/README.md)
- [Python setup, timing options and validation commands](../fp16_gemm_256x352_cute/public_reference/README.md)
