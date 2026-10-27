<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Bessel j1 with libm semantics

`libm::j1{f}` maps to CUDA 13.0 `__nv_j1{f}` with a signed-zero correction at
negative infinity. Unlike CUDA's `j1(-inf) == -0`, libm 0.2.16 returns `+0`
for both infinities. The GPU example checks both widths against host libm,
including signed zero, finite positive and negative inputs, infinity, and NaN.
It also exercises existing `j0f` and unrelated `lookalike::j1` controls.

Run from `cuda-oxide/`:

```bash
cargo oxide run math_bessel_j1
```

The four `jn{f}`/`yn{f}` exports remain for a separate negative-order adapter.
