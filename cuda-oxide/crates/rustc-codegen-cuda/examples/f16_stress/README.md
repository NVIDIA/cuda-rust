# f16_stress

Small stress test for Rust nightly's primitive `f16` support in cuda-oxide.

It checks that `f16` works as a real floating-point type through the device pipeline:

- MIR type import from Rust `f16`.
- Lowering to LLVM `half`.
- Constants and device memory traffic.
- Basic arithmetic and comparison.
- Casts between `f16` and `f32`.

Run it with:

```bash
cargo oxide run f16_stress
```

This example intentionally uses Rust's built-in nightly `f16`, not a CUDA library wrapper. Other low-precision formats such as `bf16`, `fp8`, `fp6`, and `fp4` should live as CUDA-facing library types instead.
