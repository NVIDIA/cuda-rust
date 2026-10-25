<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Special math

The scalar tests cover `tgamma`, standard-library `gamma()`, `ilogb`, and
`ldexp`/`scalbn` for f32/f64. The frexp tests additionally exercise
`libm::frexpf(f32) -> (f32, i32)` and `libm::frexp(f64) -> (f64, i32)`.

Libdevice returns a scalar fraction and writes an exponent through a pointer.
The lowering adapts this ABI to Rust's tuple using a function-local slot,
a libdevice call, a load and tuple construction. The slot lives in the function's
entry block so loop iterations reuse static storage. Tuple fields use the
existing layout map, including reordered fields and padding.

For infinities and NaNs, libm returns `(x, 0)`, whereas libdevice leaves the
exponent unspecified. The adapter initializes the exponent slot and explicitly
selects the original input and zero exponent for non-finite inputs. Integer
bit tests preserve NaN payloads and do not introduce floating-point comparisons.

References:

- [libm frexp implementation](https://github.com/rust-lang/libm/blob/master/libm/src/math/frexp.rs)
- [libm frexpf implementation](https://github.com/rust-lang/libm/blob/master/libm/src/math/frexpf.rs)
- [NVIDIA libdevice frexp contract](https://docs.nvidia.com/cuda/libdevice-users-guide/libdevice_users_guide/function_device__functions__decls_8h_1addcff9bac6691e5d398a2f8e684ae441.html)

## Coverage

The example retains the 264 scalar checks and adds 512 frexp checks, for an
expected total of 776. Each width has 32 frexp inputs: finite values, signed
zeros, normal limits, subnormal limits, infinities and positive/negative quiet
and signaling NaNs with payloads. Four routes check both tuple components:

- A device-function boundary returning the tuple.
- Assignment into a nested tuple field.
- A call inside a loop.
- An unrelated user function with a libm-like name, which must retain its body.

The host compares both frexp components exactly, including fraction bits.
Lowering tests cover both intrinsic backends, an out-of-entry-block call,
reordered tuple layouts, and rejected arities and types. Importer tests cover
public/internal libm spellings and unrelated names.

## Validation

Run from `cuda-oxide/` using the pinned toolchain:

```bash
cargo oxide fmt
cargo oxide fmt --check
git diff --check
cargo test -p dialect-mir -p mir-importer -p mir-lower --all-targets
cargo clippy -p dialect-mir -p mir-importer -p mir-lower --all-targets -- -D warnings
(cd crates/rustc-codegen-cuda && cargo test --lib)
(cd crates/rustc-codegen-cuda/examples/math_special && cargo clippy --locked --all-targets -- -D warnings)
cargo oxide run math_special
cargo oxide run libm_math
cargo oxide run math_hyperbolic
just check
```

Expected output from the extended example:

```text
SUCCESS: math_special passed 776 checks across f32 and f64
```

The tuple-returning helpers are marked `#[inline(never)]` to exercise the
device-function ABI in the optimized run, together with libdevice linking.
`cargo oxide run` uses the release profile and does not accept `--profile`.
