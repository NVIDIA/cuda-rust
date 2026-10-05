# cbrt_smoke

## `cbrt` via libdevice

This example demonstrates `f32::cbrt` and `f64::cbrt` in device code
lowering to NVIDIA libdevice (`__nv_cbrtf`, `__nv_cbrt`).

## What This Example Does

- Two small kernels — one per width — compute `x.cbrt()` per element using
  the native Rust method syntax.
- The host runs the kernels over 16 inputs spanning negative, zero, small
  and large magnitudes (plus perfect cubes whose root is exactly
  representable), then compares each result against the same expression
  evaluated with stdlib `f{32,64}::cbrt` on the host.
- Unlike `sqrt`, `cbrt` is defined for negative operands, so the
  sign-cross cases (`-8 -> -2`, `-64 -> -4`, `-0.0`) are the interesting
  part of the check.
- Tolerance: 2 ULP, matching the bound `math_atan` / `primitive_stress`
  use for the other libdevice transcendentals.

Exits 0 on PASS, 1 on FAIL.

## Pipeline

Because the kernels emit `__nv_*` calls, the cuda-oxide pipeline stops at
NVVM-IR (skipping `llc`). `ltoir::load_kernel_module` then drives libNVVM
(linking `libdevice.10.bc`) and nvJitLink to produce a cubin on first
launch.

## Run

```bash
cargo oxide run cbrt_smoke
```
