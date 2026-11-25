# RMSNorm with warp row partitions

This cuda-oxide example normalizes each matrix row and applies a weight vector:

```text
inverse_rms = 1 / sqrt(mean(row * row) + epsilon)
output[col] = row[col] * inverse_rms * weight[col]
```

It runs with FP32 and BF16 storage, accumulating the square sum in FP32.
For BF16, the normalized value rounds to BF16 before multiplication by the
weight; the final result rounds again.

## The row flow

```text
row → StridedTensor → local_partition::<32>(lane) → sum local squares
                                                             │
                                                  warp_reduction_sum_f32
                                                             │
weights → matching partition → normalize and multiply ────────┘
                                        │
output ← StridedTensorMut ← store owned columns
```

One warp (32 threads) owns one row; four warps share a block. Lanes own every
32nd column. Short rows and column tails contribute zero to the reduction,
so every lane still reaches the collective. No row alignment is required.

## Run it

From `cuda-oxide/`, with the pinned toolchain, CUDA toolkit, the CUTLASS compiler
(`cargo oxide toolchain install cutlass`) and an RTX 5090:

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
  cargo oxide run rmsnorm_cute --arch sm_120a
```

A successful run prints `PASS: FP32/BF16 RMSNorm`, reports numerical errors,
and confirms the exact BF16 rounding fixture and bit-preserving copies.
It exercises 45 shapes, including empty rows and partial blocks.

See [numerics and validation](reference/validation.md) for the checked host
boundary, tolerances, test commands, and sanitizer commands. These are
synthetic correctness checks, without checkpoint or performance claims.
