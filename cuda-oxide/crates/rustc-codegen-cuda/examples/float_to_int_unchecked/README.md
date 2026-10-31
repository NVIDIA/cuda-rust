# float_to_int_unchecked

Exercises `f32::to_int_unchecked` and `f64::to_int_unchecked`, which call
`core::intrinsics::float_to_int_unchecked`.

Before the intrinsic was implemented, a kernel using either method failed to
translate (#1595):

```
Translation failed: f64_to_i64: [".../core/src/convert/num.rs": line: 20, column: 30]
  Compilation error: invalid input program.
  Unsupported construct: rustc intrinsic `std::intrinsics::float_to_int_unchecked`
  is not yet supported on the device
```

That line is the `float_to_int_unchecked(self)` call inside `to_int_unchecked`.

## Run

```bash
cargo oxide run float_to_int_unchecked --arch sm_86
```

Expected output:

```
f32 -> i32: 256 / 256 correct
f32 -> u32: 256 / 256 correct
f64 -> i64: 256 / 256 correct
f64 -> u64: 256 / 256 correct

PASS
```

## What it checks

The intrinsic lowers to a plain `fptosi` or `fptoui`, picked from the
destination's signedness, with none of the clamping and NaN handling that the
saturating `as` cast needs. Each kernel feeds values that only the right
instruction converts correctly:

| Kernel       | Inputs                                       |
|--------------|----------------------------------------------|
| `f32_to_i32` | negative and positive, with fractional parts |
| `f32_to_u32` | up to about 4.08e9, above `i32::MAX`         |
| `f64_to_i64` | -8.96e18 to 8.89e18, outside the i32 range   |
| `f64_to_u64` | up to about 1.79e19, above `i64::MAX`        |

The host's `as` cast is the reference. For in-range inputs it gives the same
value as `to_int_unchecked`; they differ only where `to_int_unchecked` is
undefined behaviour. Every input is finite and in range for its destination,
as the method's contract requires.
