# Device FFI Test

This example demonstrates cuda-oxide's ability to call external device functions
defined in CUDA C++ and compiled to LTOIR. The complete pipeline from Rust kernel
code through LTOIR linking to GPU execution is fully working.

**Status: ✅ Complete and verified on RTX 5090 (sm_120)**

---

## Quick Start

```bash
cargo oxide run device_ffi_test --emit-nvvm-ir --arch=<your_arch>  # e.g., sm_120
```

If your default host compiler is newer than the CUDA Toolkit supports, choose
one explicitly for the `nvcc` steps:

```bash
NVCC_CCBIN=/usr/bin/g++-15 cargo oxide run device_ffi_test --emit-nvvm-ir --arch=sm_120
```

`CUDAHOSTCXX` is also honored as a fallback when `NVCC_CCBIN` is unset.

> **Note:** The `--emit-nvvm-ir` and `--arch=<your_arch>` flags are **required** for this example.
> Running without these flags will result in a compilation error because device FFI
> requires LTOIR linking which needs NVVM IR output. A proper error message for this
> case is planned for a future update.

This single command:
1. Builds the cuda-oxide compiler backend
2. Compiles kernels to NVVM IR (`.ll` file)
3. Builds external CUDA libraries to LTOIR
4. Compiles cuda-oxide IR to LTOIR
5. Links all LTOIR files to cubin
6. Runs tests on GPU

Expected output:

```text
=== Device FFI Test ===

=== Compiling cuda-oxide LLVM IR to LTOIR ===
  ✓ cuda-oxide LTOIR compiled

=== Linking LTOIR files ===
  ✓ LTOIR linked to cubin

=== Running GPU Tests ===
Device: NVIDIA GeForce RTX 5090

--- Test 1: test_simple_device_funcs ---
    ✓ PASSED

--- Test 2: test_cub_warp_reduce ---
    ✓ PASSED (all warps sum to 496)

--- Test 3: test_mixed_attrs ---
    ✓ PASSED (outputs are finite)

✓ All tests PASSED!
```

---

## Directory Structure

```text
device_ffi_test/
├── src/
│   └── main.rs              # Rust kernels + test harness
├── extern-libs/             # External CUDA libraries
│   ├── external_device_funcs.cu
│   ├── cccl_wrappers.cu
│   └── build_ltoir.sh       # Build script
├── tools/                   # LTOIR compilation tools (C)
│   ├── compile_ltoir.c      # libNVVM wrapper
│   ├── link_ltoir.c         # nvJitLink wrapper
│   └── build_tools.sh
├── Cargo.toml
└── README.md
```

---

## Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DEVICE FFI PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  cuda-oxide (Rust)                      CUDA C++ (extern-libs/)             │
│  ─────────────────                      ───────────────────────             │
│  src/main.rs                            external_device_funcs.cu            │
│  • Kernel definitions                   cccl_wrappers.cu                    │
│  • #[device] extern "C" { ... }                                             │
│         │                                      │                            │
│         ▼                                      ▼                            │
│  cargo oxide run --emit-nvvm-ir              nvcc -dc -dlto                 │
│         │                                      │                            │
│         ▼                                      ▼                            │
│  device_ffi_test.ll                      extern-libs/*.ltoir                │
│         │                                      │                            │
│         ▼                                      │                            │
│  tools/compile_ltoir (libNVVM)                 │                            │
│         │                                      │                            │
│         ▼                                      │                            │
│  device_ffi_test.ltoir ────────────────────────┘                            │
│                        │                                                    │
│                        ▼                                                    │
│               tools/link_ltoir (nvJitLink)                                  │
│                        │                                                    │
│                        ▼                                                    │
│                 merged.cubin                                                │
│                        │                                                    │
│                        ▼                                                    │
│               cuda-core (Rust) → GPU                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## How It Works

### 1. Rust Kernel Code (`src/main.rs`)

```rust
use cuda_device::{device, kernel};

// Declare external device functions from CUDA C++
// Note: No NVVM attributes needed - see "Why No NVVM Attributes?" below
#[device]
unsafe extern "C" {
    fn magnitude_squared(x: f32, y: f32) -> f32;
    fn warp_reduce_sum(val: f32) -> f32;
}

// Kernel that calls external functions
#[kernel]
fn test_kernel(output: *mut f32) {
    let mag = unsafe { magnitude_squared(1.0, 2.0) };
    let sum = unsafe { warp_reduce_sum(mag) };
    // ...
}
```

### 2. External CUDA C++ (`extern-libs/external_device_funcs.cu`)

```cuda
extern "C" __device__ float magnitude_squared(float x, float y) {
    return x * x + y * y;
}

extern "C" __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
```

---

## External Device Functions

### From `extern-libs/external_device_funcs.cu`

| Function             | Description             |
|----------------------|-------------------------|
| `magnitude_squared`  | x² + y²                 |
| `fast_rsqrt`         | Fast inverse sqrt       |
| `dot_product`        | Vector dot product      |
| `warp_reduce_sum`    | Warp-level sum          |
| `warp_ballot`        | Warp ballot             |
| `simple_add`         | Simple a + b            |
| `char_to_upper`      | ASCII uppercase, `char` |
| `clamp_value`        | Clamp to range          |

### `char` in a `#[device]` extern

Rust `char` is a 32-bit Unicode scalar value. It therefore lowers to a plain
`i32` device-extern parameter, result, or pointer pointee with no `signext` or
`zeroext` attribute, matching CUDA C++ `unsigned int`:

```text
.ll:   declare i32 @char_to_upper(i32)      call i32 @char_to_upper(i32 113)
.ptx:  .extern .func (.param .b32 ...) char_to_upper (.param .b32 ...)
```

The CUDA function must return a valid Unicode scalar: values above `0x10FFFF`
and surrogates in `0xD800..=0xDFFF` violate Rust's `char` validity contract.
Rust's `improper_ctypes` lint also warns because C has no native equivalent;
callers that deliberately use this mapping may allow that lint on the extern.

### From `extern-libs/cccl_wrappers.cu` (CUB)

| Function                    | Description               |
|-----------------------------|---------------------------|
| `cub_warp_reduce_sum_f32`   | Warp-level sum reduction  |
| `cub_warp_reduce_max_f32`   | Warp-level max reduction  |
| `cub_block_reduce_sum_f32`  | Block-level sum (256 thr) |

---

## Why No NVVM Attributes on Extern Declarations?

When using LTOIR linking, **NVVM attributes (`#[convergent]`, `#[pure]`, `#[readonly]`)
are not needed on Rust extern function declarations**. Here's why:

### The LTOIR Already Has Proper Attributes

External CUDA C++ compiled with `nvcc -dc -dlto` produces LTOIR with complete
attribute specifications on the function **definitions**:

```llvm
; From cccl_wrappers_text.ltoir
attributes #0 = { convergent inlinehint mustprogress willreturn }
attributes #7 = { convergent nounwind }
```

### nvJitLink Uses Definition Attributes

During LTO linking:
1. cuda-oxide LTOIR has **declarations** (no function body)
2. External LTOIR has **definitions** with proper attributes  
3. nvJitLink resolves symbols and uses the **definition's attributes** for optimization

### Attributes on Declarations Are Redundant

Since the linker uses the definition's attributes:
- Attributes on declarations don't affect the final code
- The external library authors already set correct attributes
- User-specified attributes could be incorrect or misleading

### What About cuda-oxide Built-in Intrinsics?

For cuda-oxide's own intrinsics (like `sync_threads()`, `shuffle()`, etc.), the
compiler handles convergent semantics internally. Users don't need to annotate these.

### Historical Note

Earlier versions of this example used `#[convergent]`, `#[pure]`, and `#[readonly]`
attributes. These have been removed as they provided no benefit with LTOIR linking
and could mislead users into thinking they were necessary.

---

## Manual Build (if needed)

```bash
# 1. Build LTOIR tools
cd tools && ./build_tools.sh && cd ..

# 2. Build external LTOIR
cd extern-libs && ./build_ltoir.sh <your_arch> && cd ..  # e.g., sm_120

# 3. Generate cuda-oxide LLVM IR
cargo oxide run device_ffi_test --emit-nvvm-ir --arch=<your_arch>  # e.g., sm_120
# (This also runs the test harness which handles steps 4-6)
```

---

## Future Work

- [ ] Rust bindings for libNVVM (`libnvvm-sys`)
- [ ] Rust bindings for nvJitLink (`nvjitlink-sys`)
- [ ] Integrate LTOIR pipeline into `cuda-host` crate
