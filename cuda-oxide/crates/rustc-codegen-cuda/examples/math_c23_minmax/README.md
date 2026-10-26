<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# IEEE 754 min/max semantics

Before this PR, `libm::fmaximum{f}`, `fminimum{f}`, `fmaximum_num{f}` and
`fminimum_num{f}` retained their Rust math bodies on the device. After this PR,
the generated dispatch recognizes their exact libm paths and lowers each pair
to ordered comparisons and selects. The first two operations propagate NaNs;
the `_num` forms return the numeric operand when only one input is NaN.
Maximum chooses +0 and minimum chooses -0 when the inputs have opposite
zero signs. The expansion has no fast-math relaxation or libdevice dependency.

The CUDA example compares 1,158 outcomes with libm 0.2.16, covering every
ordered pair of twelve inputs in both precisions. Non-NaN outputs are compared
bitwise, including signed zero. Existing `fmax`/`fmin` and same-named user
functions are controls. Compiler tests check both backends and reject malformed
signatures.

From `cuda-oxide/` on a CUDA capable host:

```bash
cargo run -p cuda-intrinsics-gen --locked -- check
cargo test -p cuda-intrinsics-gen -p dialect-mir -p mir-importer -p mir-lower --all-targets --locked
cargo oxide run math_c23_minmax
cargo oxide run math_lgamma_r
just check
```

The table now covers 118 of 124 libm exports. The remaining six Bessel
variants (`j1{f}`, `jn{f}`, `yn{f}`) need separate semantic adapters.
