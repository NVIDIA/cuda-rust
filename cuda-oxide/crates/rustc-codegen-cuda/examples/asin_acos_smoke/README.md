# asin_acos_smoke

## `asin`/`acos` via libdevice

This example demonstrates `f32::asin`/`acos` and `f64::asin`/`acos` in
device code lowering to NVIDIA libdevice (`__nv_asinf`, `__nv_asin`,
`__nv_acosf`, `__nv_acos`).

## What This Example Does

- Four small kernels — `asin`/`acos` × `f32`/`f64` — compute `x.asin()` /
  `x.acos()` per element using the native Rust method syntax.
- The host runs the kernels over 14 inputs spanning the full `[-1, 1]`
  domain (including both endpoints and `-0.0`), then compares each result
  against the same expression evaluated with stdlib `f{32,64}::asin`/`acos`
  on the host.
- The domain endpoints are the interesting cases: `asin(±1) = ±pi/2`,
  `acos(-1) = pi`, `acos(1) = 0`.
- Tolerance: 2 ULP, matching the bound `math_atan` / `cbrt_smoke` /
  `primitive_stress` use for the other libdevice transcendentals.

Unlike `std::sys::cmath`, `asin`/`acos` have no `core` libm form (they are
std-only), so the inherent methods lower straight through the cmath shims.
The `libm`-crate spellings (`libm::asinf`, ...) glam emits on nvptx route to
the same libdevice path; those are covered by the importer unit tests.

Exits 0 on PASS, 1 on FAIL.

## Pipeline

Because the kernels emit `__nv_*` calls, the cuda-oxide pipeline stops at
NVVM-IR (skipping `llc`). `ltoir::load_kernel_module` then drives libNVVM
(linking `libdevice.10.bc`) and nvJitLink to produce a cubin on first
launch.

## Run

```bash
cargo oxide run asin_acos_smoke
```
