# A BF16 feed-forward block

This cuda-oxide example composes the bias-free feed-forward core:

```text
                 ┌─ gate projection → SiLU ─┐
input ───────────┤                           × → down projection → output
                 └─ up projection ──────────┘
```

All stored tensors are BF16. Projections accumulate in FP32 and round to BF16.
SiLU rounds before multiplication by the up projection, then the product rounds
again. RMSNorm and residual addition belong to the surrounding decoder layer.

## The CuTe flow

Each warp (32 threads) uses `Bf16Mma16x8x16` partitions to compute a 16×8
output tile, accumulate 16 input values at a time, and store its owned coordinates. Predicated loads supply zeros for
matrix tails. Weights are stored as `[output, input]` matrices.

The activation uses `Tensor` and `TensorMut` tiles with eight BF16 values per
thread. Typed weight roles distinguish gate, up, and down projections, while an
owned workspace holds the intermediate tensors. The runner waits for pending
input work and completes all four kernels before returning.

## Run it

From `cuda-oxide/`, with the pinned toolchain, CUDA toolkit, the CUTLASS compiler
(`cargo oxide toolchain install cutlass`) and an RTX 5090:

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
  cargo oxide run ffn_cute --arch sm_120a
```

A successful run prints `PASS FFN` for each shape and a final intermediate/output
count. It checks each stage against an FP64 reference, checks guards and tails,
and exercises the exact SiLU rounding boundary and pending input writes.

See [numerics and validation](reference/validation.md) for shape contracts,
tolerances, host tests, and sanitizer commands. This is synthetic FFN
correctness evidence; checkpoint inference and model performance remain
separate requirements.
