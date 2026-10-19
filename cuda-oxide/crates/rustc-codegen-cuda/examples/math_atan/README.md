# math_atan

## `atan` / `atan2` via libdevice

This example demonstrates the `f32::atan` / `f32::atan2` and
`f64::atan` / `f64::atan2` calls in device code lowering to NVIDIA
libdevice (`__nv_atanf`, `__nv_atan2f`, `__nv_atan`, `__nv_atan2`).

## What This Example Does

- Four small kernels — one per (width × {`atan`, `atan2`}) — compute their
  result per element using the native Rust method syntax (`y.atan2(x)`,
  `y.atan()`).
- The host runs the kernels over 16 inputs covering all four atan2
  quadrants and a mix of small/large/mixed-sign magnitudes, then compares
  each result against the same expression evaluated with stdlib
  `f{32,64}::atan{,2}` on the host.
- Tolerance: 2 ULP, matching the bound `primitive_stress` uses for the
  other libdevice transcendentals.

Exits 0 on PASS, 1 on FAIL.

## Pipeline

Because the kernels emit `__nv_*` calls, the cuda-oxide pipeline stops at
NVVM-IR (skipping `llc`). `ltoir::load_kernel_module` then drives libNVVM
(linking `libdevice.10.bc`) and nvJitLink to produce a cubin on first
launch.

## Run

```bash
cargo oxide run math_atan
```
