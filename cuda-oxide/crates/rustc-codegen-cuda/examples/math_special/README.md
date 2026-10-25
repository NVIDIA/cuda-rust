<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Special scalar math

Exercises the first scalar additions for issue #1224:

| Rust call | f32 libdevice symbol | f64 libdevice symbol |
| --- | --- | --- |
| `libm::tgamma{f}` and `f32/f64::gamma` | `__nv_tgammaf` | `__nv_tgamma` |
| `libm::ilogb{f}` | `__nv_ilogbf` | `__nv_ilogb` |
| `libm::ldexp{f}`, `libm::scalbn{f}` | `__nv_ldexpf` | `__nv_ldexp` |

`ilogb` returns an `i32`; the exponent argument of `ldexp`/`scalbn` is
an `i32`. `float_gamma` is enabled to exercise the standard-library cmath
shim as well as the `libm` entry points. Use the repository's pinned toolchain
and its existing `RUSTC_BOOTSTRAP` configuration.

Before these mappings are added, the standard gamma path is rejected by the
collector and the libm paths have no dedicated libdevice lowering. Capture the
baseline diagnostic on the unmodified tree; no local baseline/GPU run is claimed
by this source package.

The example checks 264 results: 22 inputs, six checks per input, and two widths.
It compares gamma to host libm within a relative tolerance, compares integer
logarithms exactly, and compares scaled values bit for bit. The inputs include
negative non-integers, both exponent signs, subnormals, scaling overflow and
underflow. A user function with a libm-like module name must retain its own
body. `ilogb` inputs are finite and nonzero; its domain-error sentinels are
outside this example's comparison.

The lowering tests independently verify all six libdevice signatures on both
backends, reject malformed calls, and retain `powi` as a mixed-signature control.
Importer tests check public/internal libm spellings, aliases, cmath gamma,
unrelated names, and functions deferred to later PRs.

## Validation

Run from `cuda-oxide/` after copying the files into the existing checkout:

```bash
cargo oxide fmt
cargo oxide fmt --check
git diff --check
cargo test -p dialect-mir -p mir-importer -p mir-lower --all-targets
cargo clippy -p dialect-mir -p mir-importer -p mir-lower --all-targets -- -D warnings
(cd crates/rustc-codegen-cuda && cargo test --lib)
(cd crates/rustc-codegen-cuda && cargo clippy --all-targets -- -D warnings)
(cd crates/rustc-codegen-cuda/examples/math_special && cargo clippy --locked --all-targets -- -D warnings)
cargo oxide build math_special
cargo oxide run math_special
cargo oxide run libm_math
cargo oxide run math_hyperbolic
bash scripts/check-example-smoketest-contract.sh
bash scripts/sync-example-locks.sh --check
```

`math_special` must print `SUCCESS: math_special passed 264 checks across f32 and f64`.
GPU commands require a working CUDA toolkit, driver and device. The example is
a separate workspace; the parent workspace's tests do not build it.

Before marking the PR ready, run the repository gate from the repository root:

```bash
just -f cuda-oxide/Justfile check
```

This package has undergone static source and archive checks only. Rust, Cargo,
and CUDA were unavailable in the preparation environment; compile, clippy and
GPU validation are pending execution on a configured checkout.
