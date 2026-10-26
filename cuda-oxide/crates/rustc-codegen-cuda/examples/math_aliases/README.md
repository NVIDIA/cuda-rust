<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Plain libm math aliases

This example exercises both f32 and f64 forms of 11 ordinary libm aliases:
`erf`, `erfc`, `exp10`, `j0`, `lgamma`, `y0`, `y1`, `fdim`, `fmod`,
`nextafter`, and `remainder`. Previously, calls such as `libm::erff(x)` had no
device mapping. The generator now selects `__nv_erff` from a validated row.
An unrelated `libm_lookalike::erff` and an existing `libm::floorf` path serve
as controls. The 16 cases for each of 13 operations on two widths perform
416 comparisons with host libm, covering signed zeros, nonfinite values,
and tie-to-even remainder. The GPU test reports:

```text
SUCCESS: math_aliases passed 416 checks across f32 and f64
```

The shard now covers 104 of libm 0.2.16's 124 public f32/f64 exports. The
20 remaining exports need out-parameter adapters or separate semantics:
`modf{f}`, `remquo{f}`, `lgamma{f}_r`, eight C23 min/max variants, and
`j1{f}`, `jn{f}`, `yn{f}`. CUDA 13.0 libdevice returns NaN for negative
`jn`/`yn` orders, while libm supports them. `j1(-∞)` also differs in the
sign of zero. These Bessel variants require dedicated handling before mapping.
See the CUDA 13.0 [jn](https://docs.nvidia.com/cuda/archive/13.0.0/libdevice-users-guide/__nv_jn.html)
and [yn](https://docs.nvidia.com/cuda/archive/13.0.0/libdevice-users-guide/__nv_yn.html)
contracts.

Run from `cuda-oxide/` with the pinned toolchain and a CUDA-capable machine:

```bash
cargo oxide fmt --check
git add -N crates/cuda-intrinsics-gen/src/libdevice_math/libm-0.2.16-exports.json crates/rustc-codegen-cuda/examples/math_aliases
git diff --check
cargo test -p cuda-intrinsics-gen -p dialect-mir -p mir-importer -p mir-lower --all-targets --locked
cargo clippy -p cuda-intrinsics-gen -p dialect-mir -p mir-importer -p mir-lower --all-targets --locked -- -D warnings
cargo run -p cuda-intrinsics-gen --locked -- check
cargo oxide run math_aliases
cargo oxide run math_special
cargo oxide run libm_math
cargo oxide run math_hyperbolic
just check
```

`cargo oxide run` uses the release profile and does not accept `--profile`.
