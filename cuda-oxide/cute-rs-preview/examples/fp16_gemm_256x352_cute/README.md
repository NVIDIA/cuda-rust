# FP16 GEMM with CuTe-RS — SM100a (B200)

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

`cute-rs` describes layouts, tensor operations and pipelines; `ctm-rs` supplies
explicit instruction primitives. The [CTM version](../fp16_gemm_256x352_ctm/README.md)
expresses the same schedule directly through instruction types and physical formats.

## Run it

You need an SM100 GPU and a compatible CUDA driver. From `cuda-oxide/`, once per checkout:

```bash
cargo oxide toolchain install cutlass
python3 -m venv cute-rs-preview/.venv
cute-rs-preview/.venv/bin/pip install -r cute-rs-preview/requirements.txt
```

Then:

```bash
export CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir

# Small correctness comparison
cargo oxide run fp16_gemm_256x352_cute --arch sm_100a

# Throughput comparison
cargo oxide run fp16_gemm_256x352_cute --arch sm_100a -- \
  --mnk 8192,8096,8192 --warmup 2000 --iters 301 --json /tmp/cute-gemm.json
```

Both outputs are checked against an independent PyTorch oracle on identical
inputs. The command ends with:

```text
CUTLASS (CuTe + CTM): <time> us, <rate> TFLOP/s
cute-rs: <time> us, <rate> TFLOP/s
```

The shared [Python CTM reference](public_reference/README.md) uses CUTLASS
instruction primitives, with CuTe compilation and epilogue helpers. Without a
prepared environment, the first comparison run creates `cute-rs-preview/.venv` itself.
Add `--has-bias` for row bias, `--rust-only` for standalone Rust with CPU
verification, or `--help` for options. Large standalone CPU checks are expensive.

## B200 performance

Measured **2026-10-18**, CUTLASS 4.7.0, M8192/N8096/K8192, no bias, 74 clusters.
CUDA events report the median of three run medians, each with 2,000 warmups
and 301 single-kernel samples. Nsight Systems shows the median of 301 kernel
durations from a separate capture; TFLOP/s uses the CUDA-event median.

| Implementation | CUDA events (µs) | TFLOP/s | Nsight kernel (µs) |
| --- | ---: | ---: | ---: |
| Rust CuTe (`cute-rs`) | 770.784 | 1409.768 | 766.752 |
| Python CTM (CUTLASS primitives + CuTe helpers) | 774.592 | 1402.838 | 732.416 |

Event intervals can include launch gaps; Nsight durations exclude gaps between
kernels. Clocks were unlocked and Python ran first. These separate runs vary,
so subtracting columns does not isolate overhead or establish a stable ranking.

<a id="regression-checks"></a>

The Python reference has an unresolved group-two TMA memcheck diagnostic;
see [validation and known limitation](public_reference/README.md#validation-and-known-limitation).

- [Layouts, pipeline, descriptor ABI and Rust API](public_reference/architecture.md)
- [Python setup, timing options and validation commands](public_reference/README.md)
- [Public reference provenance](public_reference/SOURCE.md)
