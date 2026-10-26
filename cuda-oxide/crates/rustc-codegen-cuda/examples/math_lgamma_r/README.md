<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gamma logarithm with sign

Before this PR, `libm::lgamma_r` and `libm::lgammaf_r` used their Rust
implementations on the device. The new generated placeholders compose the
existing, validated `__nv_lgamma{f}`, `__nv_floor{f}`, and `__nv_fmod{f}`
symbols into libm's `(log(abs(gamma(x))), signgam)` result. Negative interval
parity determines the sign without evaluating `sin(pi*x)` near underflow;
integer poles have sign +1 and negative zero has sign -1. CUDA documents
negative infinity for scalar `lgamma(-infinity)`, while some backends return
positive infinity. The tuple adapter always returns libm's positive infinity;
the unrelated scalar control accepts either infinity. Tuple fields use their
declared Rust layout.

The example compares both precisions against libm 0.2.16, including positive
values, negative intervals, poles, signed zero, subnormals, infinities, NaN,
and large negative arguments. Scalar `lgamma` and an unrelated same-named
function are controls. Compiler tests cover both backends, tuple layout and
malformed signatures.

From `cuda-oxide/` on a CUDA capable host:

```bash
cargo run -p cuda-intrinsics-gen --locked -- check
cargo test -p cuda-intrinsics-gen -p dialect-mir -p mir-importer -p mir-lower --all-targets --locked
cargo oxide run math_lgamma_r
cargo oxide run math_out_params
just check
```

The table now covers 110 of 124 libm exports. The remaining 14 are the eight
C23 min/max variants and six Bessel variants (`j1{f}`, `jn{f}`, `yn{f}`).
