<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Special math and `libm` adapters

This example exercises the `f32` and `f64` math routes generated from
`intrinsics/overlay/libdevice_math.toml` and their handwritten semantic
adapters in `mir-lower`:

| Suite             | Operations                                                                                   | Checks |
|:------------------|:---------------------------------------------------------------------------------------------|:-------|
| Special scalar    | `tgamma`, `f32/f64::gamma()`, `ilogb`, `ldexp`, `scalbn`                                     | 264    |
| `frexp` tuple     | `libm::frexpf`, `libm::frexp` (helper boundary, projected tuple, loop)                       | 512    |
| Plain aliases     | `erf`, `erfc`, `exp10`, `j0`, `lgamma`, `y0`, `y1`, `fdim`, `fmod`, `nextafter`, `remainder` | 416    |
| Out-param pairs   | `libm::modff`, `libm::modf`, `libm::remquof`, `libm::remquo`                                 | 180    |
| Composed gamma    | `libm::lgammaf_r`, `libm::lgamma_r`                                                          | 208    |
| IEEE-754 min/max  | `fmaximum`, `fminimum`, `fmaximum_num`, `fminimum_num` (`f32`/`f64`)                         | 1,158  |
| Bessel `j1`       | `libm::j1f`, `libm::j1` (with `+-inf -> +0.0` normalization)                                 | 84     |

Each suite includes unrelated user functions with `libm`-like names (`libm_lookalike::*`)
to verify that interception is anchored to the `libm` crate root and never
rewrites user functions.

## Lowering notes

- **Out-parameter tuple adapters (`frexp`, `modf`, `remquo`)**: Libdevice
  returns the primary scalar and writes the second component through a pointer.
  The lowering allocates a static entry-block stack slot (reused across loop
  iterations), calls `__nv_frexp{f}` / `__nv_modf{f}` / `__nv_remquo{f}`, loads
  the out value, and packs the Rust tuple using `build_struct_slot_map`. For
  `frexp`, non-finite inputs select `(x, 0)` bitwise so NaN payloads are
  preserved. For `remquo`, invalid inputs normalize the quotient to `0` when
  the remainder is NaN.
- **Composed `lgamma_r`**: Composes validated scalar `__nv_lgamma{f}`,
  `__nv_floor{f}`, and `__nv_fmod{f}` calls to compute
  `(log(abs(gamma(x))), signgam)` with negative-interval parity, `+1` at
  integer poles, `-1` at `-0.0`, and `+inf` at `-inf`.
- **IEEE-754-2019 min/max (`fmaximum*`, `fminimum*`)**: Expands inline to
  ordered comparisons and selects without fast-math flags or libdevice calls,
  breaking `-0.0` vs `+0.0` ties via the sign bit and either propagating NaNs
  (`fmaximum`/`fminimum`) or preferring the numeric operand
  (`fmaximum_num`/`fminimum_num`).
- **Bessel `j1`**: Calls `__nv_j1{f}` and normalizes `+-inf` inputs to `+0.0`
  to match `libm` 0.2.16 without altering NaN results.

## Validation

Run from `cuda-oxide/` using the pinned toolchain:

```bash
cargo run -p cuda-intrinsics-gen --locked -- check
cargo test -p cuda-intrinsics-gen -p dialect-mir -p mir-importer -p mir-lower --all-targets
cargo clippy --locked --manifest-path crates/rustc-codegen-cuda/examples/math_special/Cargo.toml --all-targets -- -D warnings
cargo oxide run math_special
```

Expected output:

```text
SUCCESS: math_special passed 2822 checks across f32 and f64
```
