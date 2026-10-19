# fmaxmin_smoke

Smoke test for direct LLVM lowering of `max`, `min`, `abs`, and `copysign` on
`f32` and `f64`, followed by PTX generation.

The max/min methods lower to the `_nsz` flavor of the rustc maxNum / minNum
intrinsics in MIR:

| Public API | MIR intrinsic | Direct LLVM lowering |
| ---------- | ------------- | ---------------- |
| `f32::max` | `core::intrinsics::maximum_number_nsz_f32` | `fcmp` + `select` |
| `f64::max` | `core::intrinsics::maximum_number_nsz_f64` | `fcmp` + `select` |
| `f32::min` | `core::intrinsics::minimum_number_nsz_f32` | `fcmp` + `select` |
| `f64::min` | `core::intrinsics::minimum_number_nsz_f64` | `fcmp` + `select` |

The same kernel also covers the width-specific `llvm.fabs` and
`llvm.copysign` intrinsics.

This example exists as a regression test for that lowering chain. Before
the corresponding entries existed in `dialect-mir::rust_intrinsics`,
`mir-importer`, and `mir-lower`, `f32::max` / `f32::min` fell out of the
pipeline as unresolved intrinsic calls.

Run it with:

```bash
cargo oxide run fmaxmin_smoke
```

## How code reaches the GPU

The kernels in this example use LLVM comparisons/selects, `fabs`, and
`copysign`, so cuda-oxide emits ordinary PTX.
[`cuda_host::ltoir::load_kernel_module`] loads that PTX directly; it only
falls back to libNVVM + nvJitLink for modules that actually emit NVVM IR.

## What the smoke checks

For both `f32` and `f64`:

1. The finite case — `(1.5_f32).max(-2.5_f32) == 1.5_f32` and the matching
   `.min` — confirms the direct LLVM lowering is reached and the result is
   bit-exact.
2. Quiet and signaling NaNs in either operand position are ignored when paired
   with a number; two NaNs still produce a NaN.
3. Both signed-zero orders produce a valid zero result.
4. `abs` produces exact finite/zero results and clears the sign of a NaN; its
   NaN payload is not constrained.
5. `copysign` changes only the sign bit, including for NaN magnitudes, whose
   payload is checked exactly.
