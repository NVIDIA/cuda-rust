# math_hyperbolic

Exercises the hyperbolic + extended `f32`/`f64` math methods in a kernel and
checks the GPU results against host libm:

`sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `exp_m1`, `ln_1p`, `hypot`.

## Why this exists

Six of these (`sinh`, `cosh`, `tanh`, `exp_m1`, `ln_1p`, `hypot`) are not in
this toolchain's `core_float_math`, so they lower to a `std::sys::cmath::*`
shim. Without interception a kernel calling them fails with:

```text
CUDA-OXIDE: FORBIDDEN CRATE IN DEVICE CODE
Device code calls: std::sys::cmath::sinh
```

The float-math dispatch now maps each to the matching libdevice call
(`__nv_sinh`, `__nv_hypot`, ...). `hypot` is the one binary function.

The inverse hyperbolics (`asinh`, `acosh`, `atanh`) are **not** `cmath`
calls: `std` implements them as pure-Rust formulas over `ln`/`sqrt`/`ln_1p`,
so they need no new interception. `atanh` only worked once `ln_1p` was
intercepted (it is `0.5 * (...).ln_1p()`). They are included here as a
regression guard. `acosh` (needs arg >= 1) and `atanh` (needs |arg| < 1) are
fed in-domain transforms of the input.

## Run

```bash
cargo oxide run math_hyperbolic
```

Exits 0 on `SUCCESS`, 1 on `FAILED`.
