# float_rem_smoke

## Float `%` is exact

This example checks that the float remainder operator (`%`, `%=` and the
`rem_euclid` helper built on it) in device code matches the host for the
same operands, bit for bit.

## What This Example Does

- Four kernels (`%` and `rem_euclid`, one per float width) compute
  `xs[i] % ys[i]` per element.
- The host evaluates the same expressions with stdlib `f{32,64}` and
  compares bit patterns. Rust float remainder is C `fmod`, which is exact
  for every finite operand pair, so no tolerance is needed.
- The operand set includes quotients far beyond `2^53` (`1e20 % 7.0`,
  `3e17 % 0.7`), sign combinations, signed zero, subnormal divisors, and
  NaN/infinity edge cases. The large-quotient cases are the regression:
  a lowering that computes `a - trunc(a / b) * b` is off by thousands
  there.

Exits 0 on PASS, 1 on FAIL.

## Pipeline

The kernels lower `%` to libdevice `__nv_fmodf` / `__nv_fmod`, so the
pipeline resolves them the same way it resolves the other libdevice math
calls (IR-level link or libNVVM, depending on the toolkit).

## Run

```bash
cargo oxide run float_rem_smoke
```
