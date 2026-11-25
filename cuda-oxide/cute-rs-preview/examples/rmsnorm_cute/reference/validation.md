# Contracts and validation

[Back to the example](../README.md).

## Arithmetic and launch contract

Standalone cuda-oxide correctness example using one full warp per row, four rows
per block, and explicit BF16 storage. Inputs and weights retain their storage
dtype; both kernels accumulate squares and normalize in FP32. The BF16 kernel
rounds normalized values to BF16 **before** multiplying BF16 weights, then rounds
the result to BF16. This preserves the selected Qwen3-TTS reference's arithmetic
boundary without adding model-specific compiler operations.

The checked host boundary rejects zero widths, mismatched allocation lengths,
shape overflow, invalid epsilon and grid overflow. Zero rows skip allocation and
launch. All lanes in each participating warp reach the reusable
`cute_rs::arch::warp_reduction_sum_f32` collective; lanes beyond a short row contribute zero.
Scalar strided row accesses also support unaligned row starts.

## Host and GPU checks

From `cuda-oxide/`:

```bash
cargo test --offline --manifest-path cute-rs-preview/examples/rmsnorm_cute/Cargo.toml
CARGO_NET_OFFLINE=true CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
  cargo oxide run rmsnorm_cute --arch sm_120a
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
  cute-rs-preview/examples/rmsnorm_cute/target/release/rmsnorm_cute
/usr/local/cuda/bin/compute-sanitizer --tool synccheck --error-exitcode 1 \
  cute-rs-preview/examples/rmsnorm_cute/target/release/rmsnorm_cute
```

## Acceptance criteria

Validation covers widths 1, 7, 31, 32, 33, 128, 1024, 2048 and 2051; row counts
0, 1, 3, 4 and 5; and zero, constant, signed and tiny-valued inputs. An independent
FP64 reference checks stored input values. BF16 comparisons explicitly reproduce
both rounding boundaries using a separate integer rounding oracle. Tolerances
are `atol=1e-6, rtol=2e-5` for FP32 and `atol=1e-7, rtol=0.016` for BF16.
An additional exact BF16 fixture rejects the incorrect cast-once ordering,
independently of those numerical tolerances.

A separate BF16 copy kernel exercises typed 16-byte vector loads/stores and
scalar tails at lengths 0, 1, 7, 8, 9 and 1031. Copies must preserve every storage
bit, including signed zeros, infinities, subnormals and NaN payloads.

The fixtures keep magnitudes bounded: FP64 reference arithmetic must not hide
FP32 square overflow. This is synthetic operator validation, not checkpoint
validation, a performance benchmark or evidence of execution on SM100.
