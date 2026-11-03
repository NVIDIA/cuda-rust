# Small Type FFI Test

This example GPU-verifies small-scalar FFI round-trips: cuda-oxide kernels
calling `extern "C" __device__` functions (compiled to LTOIR by nvcc) that
pass and return i8/u8/i16/u16/bool/f16. Sub-32-bit scalars cross the boundary
as narrow LLVM types with signext/zeroext parameter and return attributes;
f16 is passed directly as `half`. cuda-oxide's emitted declarations must match
the nvcc-compiled definitions byte for byte for nvJitLink LTO to resolve them.

**This example REQUIRES the modern NVVM path (sm_100+, Blackwell).** The
legacy CUDA 12 LLVM 7 dialect used for pre-sm_100 targets rejects scalar f16
and sub-32-bit by-value externs by design:

```text
legacy NVVM IR cannot contain scalar f16 because cuda-oxide supports the
CUDA 12 LLVM 7 dialect; use f32/f64 or the modern NVVM/PTX path
```

---

## Quick Start

```bash
cargo oxide run small_type_ffi_test --emit-nvvm-ir --arch=<your_arch>  # sm_100 or newer, e.g. sm_120
```

If your default host compiler is newer than the CUDA Toolkit supports, choose
one explicitly for the `nvcc` steps:

```bash
NVCC_CCBIN=/usr/bin/g++-15 cargo oxide run small_type_ffi_test --emit-nvvm-ir --arch=sm_120
```

`CUDAHOSTCXX` is also honored as a fallback when `NVCC_CCBIN` is unset.

> **Note:** The `--emit-nvvm-ir` flag is **required** and `--arch` must be
> sm_100 or newer. Device FFI needs LTOIR linking (which needs NVVM IR
> output), and only the modern NVVM path supports small scalar externs.

This single command:
1. Builds the cuda-oxide compiler backend
2. Compiles the kernel to NVVM IR (`.ll` file)
3. Builds the external CUDA library to LTOIR
4. Compiles cuda-oxide IR to LTOIR
5. Links all LTOIR files to cubin
6. Runs the test on GPU

Expected output:

```text
=== Small Type FFI Test ===

=== Compiling cuda-oxide LLVM IR to LTOIR ===
  ✓ cuda-oxide LTOIR compiled

=== Linking LTOIR files ===
  ✓ LTOIR linked to cubin

=== Running GPU Tests ===
Device: NVIDIA GeForce RTX 5090

--- Test 1: test_small_type_ffi ---
    ✓ PASSED (all small-type round-trips correct)

✓ All tests PASSED!
```

---

## What Is Verified

| Slot | Call                            | Checks                                    |
|------|---------------------------------|-------------------------------------------|
| 0    | `small_widen_i8(-3)`            | signext of a negative i8 parameter        |
| 1    | `small_widen_u16(0xFF00)`       | zeroext of a high-bit u16 parameter       |
| 2    | `small_scale_i8(-5)`            | signext i8 return                         |
| 3    | `small_add_u8(200, 55)`         | zeroext u8 return                         |
| 4    | `small_scale_i16(-1000)`        | signext i16 return                        |
| 5    | `small_add_u16(65000, 500)`     | zeroext u16 return                        |
| 6    | `small_not_bool(false/true)`    | i1 zeroext in both directions             |
| 7    | `small_half_add(1.5, 2.0)`      | f16 passed and returned as `half` (3.5)   |

---

## Directory Structure

```text
small_type_ffi_test/
├── src/
│   └── main.rs              # Rust kernel + test harness
├── extern-libs/             # External CUDA library
│   ├── small_type_funcs.cu
│   └── build_ltoir.sh       # Build script
├── tools/                   # LTOIR compilation tools (C)
│   ├── compile_ltoir.c      # libNVVM wrapper
│   ├── link_ltoir.c         # nvJitLink wrapper
│   └── build_tools.sh
├── Cargo.toml
└── README.md
```

The LTOIR pipeline (NVVM IR → LTOIR → cubin via `tools/compile_ltoir` and
`tools/link_ltoir`) is identical to the `device_ffi_test` example; see its
README for the full architecture walkthrough and for why no NVVM attributes
are needed on the Rust extern declarations.
