# SwiGLU with register tiles

This cuda-oxide example combines two vectors in one GPU kernel:

```text
output[i] = silu(gate[i]) * up[i]
silu(x)   = x / (1 + exp(-x))
```

It runs with FP32 and FP16 storage. The FP16 path widens both inputs to FP32
and rounds once when storing the result.

## The Tensor flow

```text
gate, up → Tensor → slice(thread) → load register tiles
                                              │
                                       zip_map(swiglu)
                                              │
output  ← TensorMut ← store register tile ←────┘
```

`zipped_divide` gives each thread a 16-byte tile: four FP32 values or eight
FP16 values. `zip_map` combines corresponding registers. Complete tiles use
vector copies; the final partial tile touches only valid elements. Empty
inputs skip the launch.

## Run it

From `cuda-oxide/`, with the pinned toolchain, CUDA toolkit, CUTLASS compiler,
and an RTX 5090:

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
  cargo oxide run swiglu_cute --arch sm_120a
```

A successful run prints `PASS: FP32/FP16 SwiGLU` and the largest error for each
dtype. It compares ten lengths, including empty inputs, full tiles, and tails,
against an independent FP64 reference.

See [numerics and validation](reference/validation.md) for tolerances, host tests,
sanitizer commands, and compiler setup. This example validates synthetic FP32
and FP16 inputs; it does not establish BF16 or checkpoint correctness.
