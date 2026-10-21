# Contracts and validation

[Back to the example](../README.md).

## Operation and numerical contract

This cuda-oxide example computes `silu(gate) * up` in one kernel, with separate
FP32 and FP16 entry points. It is a synthetic validation of reusable tensor
and register-tile operations for the Qwen3-TTS implementation plan. It does
not establish BF16 support, checkpoint parity or model performance.

Each thread loads a complete 16-byte tile through a read-only `Tensor`,
combines the values in registers, and stores through a `TensorMut`. FP32 uses
four values per tile; FP16 uses eight. The final partial tile uses scalar
loads and stores. Runtime input/output lengths are checked before device
allocation or launch. Equal empty slices succeed without launching a kernel.

The FP16 path explicitly widens both tiles to FP32, computes the activation
and product in FP32, then narrows once at the output. It does not emulate an
implementation that rounds intermediate activations to FP16 or BF16.
Const-generic element counts enforce matching per-thread register-tile widths.

The device activation uses the existing approximate FP32 base-two exponential
and reciprocal intrinsics. A sign-dependent sigmoid formula keeps exponential
arguments nonpositive, avoiding overflow for large negative gates. The
reciprocal instruction flushes subnormals to zero; this example tests finite
inputs and finite outputs, and does not specify NaN or infinity behavior.

## Host and GPU checks

From `cuda-oxide/`, with the normal pinned toolchain, CUDA toolkit, CUTLASS
compiler and an RTX 5090 available:

```bash
cargo test --offline --manifest-path cute-rs-preview/examples/swiglu_cute/Cargo.toml

CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
  cargo oxide run swiglu_cute --arch sm_120a

CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
  cargo oxide sanitize swiglu_cute --arch sm_120a
```

## Acceptance criteria

The GPU run checks both dtypes at lengths **0, 1, 3, 4, 7, 8, 9, 3072, 6144
and 6147**, covering empty inputs, aligned tiles and scalar tails. Deterministic
inputs include signed zeros, small signed values and gates up to magnitude
65504. Every output is compared with an independent stable FP64 reference
computed from the actual stored input values:

| Output dtype | Absolute tolerance | Relative tolerance |
| --- | --- | --- |
| FP32 | `2e-6` | `2e-5` |
| FP16 | `2e-3` | `2e-3` |

Acceptance requires finite output and
`abs(actual - expected) <= atol + rtol * abs(expected)`. The program prints
the largest absolute error for each dtype. CPU tests cover the launch boundary,
empty-input policy, oracle values and numerical acceptance checks. They do
not replace real device compilation, GPU numerical validation or memcheck.

The example uses general Rust and CuTe compiler mappings; there is no SwiGLU-specific
compiler hook. See [Build with the CUTLASS compiler](../../../../cuda-oxide-book/cute/build-with-cutlass.md)
for compiler setup and MLIR inspection.
