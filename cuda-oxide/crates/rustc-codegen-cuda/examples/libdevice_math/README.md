# Libdevice math

This example uses Rust math methods that CUDA implements through libdevice. It
checks SwiGLU, `sin`, `exp`, `sqrt`, `atan`, `atan2`, `acos`, and `tan`.

These calls automatically select NVVM IR:

```text
Rust math method
      │
      ▼
CUDA libdevice call
      │
      ▼
NVVM IR for the selected GPU
```

Run the example normally:

```bash
cargo oxide run libdevice_math
```

The original bug appeared on pre-Blackwell GPUs because libNVVM expected LLVM
7 typed pointers. To compile that path on any machine, choose a legacy target:

```bash
cargo oxide emit-ltoir libdevice_math --arch sm_86 \
  --output /tmp/libdevice_math-sm86.ltoir
```

This command verifies and compiles the module with libNVVM. On Blackwell, the
same legacy target can also run through the forward PTX compatibility path:

```bash
cargo oxide run libdevice_math --arch sm_86
```

The selected CUDA toolkit must produce a PTX version supported by the installed
driver. If CUDA error 222 appears, upgrade the driver or select an older
compatible toolkit, for example:

```bash
CUDA_TOOLKIT_PATH=/usr/local/cuda-13.0 \
  cargo oxide run libdevice_math --arch sm_86
```

This example keeps regression coverage for
[issue #98](https://github.com/NVlabs/cuda-oxide/issues/98) and the related
community reports.
