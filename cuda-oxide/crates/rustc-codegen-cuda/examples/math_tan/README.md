# math_tan

Exercises `f32::tan()` / `f64::tan()` in a kernel and checks the GPU result
against host libm within 2 ULP.

## Why this exists

On current nightlies, `core_float_math` routes `sin`/`cos` through
`core::intrinsics`, but `tan` is not in that feature, so `f{32,64}::tan()`
lowers to `std::sys::cmath::tan{,f}`. The float-math dispatch did not
intercept that path, so a kernel calling `.tan()` failed with:

```text
CUDA-OXIDE: FORBIDDEN CRATE IN DEVICE CODE
Device code calls: std::sys::cmath::tan
```

The fix adds the `std::sys::cmath::{sin,cos,tan}{,f}` arms to the dispatch in
`cuda-oxide/crates/mir-importer/src/translator/terminator/intrinsics/float_math.rs`, so
`.tan()` now lowers to the `__nv_tan{,f}` libdevice call like every other
transcendental.

## Run

```bash
cargo oxide run math_tan
```

Exits 0 on `SUCCESS`, 1 on `FAILED`.
