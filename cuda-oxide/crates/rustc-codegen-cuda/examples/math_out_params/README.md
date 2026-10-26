<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Out-parameter math functions

Before: `libm::modf{f}` and `libm::remquo{f}` retained host Rust math bodies
because the scalar libdevice ABI could not return their Rust tuples. After:
the generated placeholders call `__nv_modf{f}` and `__nv_remquo{f}` with
a pointer to a stack slot, then pack the scalar return and out value using
the Rust tuple layout. This uses the same call helper as `frexp`.

The example checks both precisions against libm 0.2.16 for signed zero,
finite values, ties to even, and nonfinite inputs. CUDA only promises the
low three quotient bits for `remquo`, so the example compares those bits and
the quotient sign. A same-named function outside `libm` is a control. Lowering
tests cover both backends, reordered tuple fields, and malformed signatures.

From `cuda-oxide/` on a CUDA capable host:

```bash
cargo run -p cuda-intrinsics-gen --locked -- check
cargo test -p cuda-intrinsics-gen -p dialect-mir -p mir-importer -p mir-lower --all-targets --locked
cargo oxide run math_out_params
cargo oxide run math_special
just check
```

This completes the out-parameter part of the series. The remaining 16
libm exports require separate semantic work: `lgamma{f}_r`, eight C23 min/max
variants, and `j1{f}`, `jn{f}`, `yn{f}`.
