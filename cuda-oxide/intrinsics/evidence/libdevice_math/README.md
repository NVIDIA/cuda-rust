<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Libdevice math ABI capture

`cuda-13.0.json` contains 102 verified libdevice signatures referenced by
the compiler math shard (120 of 124 libm exports plus core-only routes),
including tgamma, ilogb, ldexp, frexp, modf, remquo, plain libm aliases, the
j1 Bessel pair, and core powi.
They were extracted from NVIDIA's CUDA
13.0.88 `libdevice.10.bc` with LLVM 23.1.1 from the pinned Rust 1.99.1 toolchain.

Source: [NVIDIA nvidia-nvvm 13.0.88](https://pypi.org/project/nvidia-nvvm/13.0.88/),
Linux x86-64 wheel
`nvidia_nvvm-13.0.88-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl`.

- Wheel SHA-256: `c5f41ffeb6466944a026dfa5317d7d85355c119bbec279205d22f1869d1054e0`.
- Bitcode member: `nvidia/cu13/nvvm/libdevice/libdevice.10.bc`.
- Bitcode SHA-256: `91334d6e12748f6cb5bbf0a1cd965a56bcd93dc4f496d2e5c5f8c6e523094356`.

To reproduce the capture from matching local bitcode, run from `cuda-oxide/`:

```bash
libdevice_llvm_dis="$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's/^host: //p')/bin/llvm-dis"
cargo run -p cuda-intrinsics-gen --locked -- extract-libdevice \
  --libdevice /path/to/libdevice.10.bc --llvm-dis "$libdevice_llvm_dis"
cargo run -p cuda-intrinsics-gen --locked -- generate
cargo run -p cuda-intrinsics-gen --locked -- check
```

The extractor checks definitions, return types, operand types, and duplicate
symbols before writing. Only signatures required by the shard are retained.
Opaque LLVM pointers retain their address space, but do not encode a pointee
type; out-parameter semantics remain the handwritten adapter's responsibility.
Ordinary generation validates the capture's bitcode hash against the shard pin
and every referenced ABI against the captured signatures.
