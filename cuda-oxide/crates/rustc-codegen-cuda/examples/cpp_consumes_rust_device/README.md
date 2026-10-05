# cpp_consumes_rust_device

C++ calling Rust `#[device]` functions via LTOIR — the full Phase 2 pipeline.

## What This Demonstrates

Rust device functions compiled to LTOIR by cuda-oxide, linked with a C++ kernel
via nvJitLink, and executed on the GPU. The C++ side defines the kernels, launches
them, and verifies results. The Rust side only provides the device function
implementations.

## Pipeline

> Arch placeholders below: replace `<your_arch>` with your GPU arch, e.g. `sm_120` (Blackwell).

```text
Rust #[device] fns                          C++ kernels
────────────────                            ──────────────────
fast_sqrt, clamp_f32,                       test_sqrt_clamp,
safe_sqrt, fma_f32, fma_i32                 test_safe_sqrt, test_fma
    │                                           │
    ▼                                           ▼
cargo oxide run                     nvcc -dc -gencode
  --emit-nvvm-ir --arch=<your_arch>            arch=compute_<X>,code=lto_<X>
    │                                           │
    ▼                                           ▼
cpp_consumes_rust_device.ll (NVVM IR)       caller_kernel.ltoir
    │
    ▼
libNVVM -gen-lto -arch=compute_<X>
    │
    ▼
cpp_consumes_rust_device.ltoir ─────────────────┘
                    │
                    ▼
              nvJitLink (LTO)
                    │
                    ▼
              merged.cubin ──► GPU tests (test_runner)
```

## GPU Tests

| # | Kernel            | What It Tests                                                    |
|---|-------------------|------------------------------------------------------------------|
| 1 | `test_sqrt_clamp` | `fast_sqrt` + `clamp_f32` — simple Rust device fn calls          |
| 2 | `test_safe_sqrt`  | `safe_sqrt` — transitive Rust-to-Rust device calls across LTOIR  |
| 3 | `test_fma`        | `fma_f32` + `fma_i32` — monomorphized generic device functions   |

## How to Run

### Step 1: Generate NVVM IR (Rust side)

```bash
# From workspace root
cargo oxide run cpp_consumes_rust_device --emit-nvvm-ir --arch=<your_arch>  # e.g., sm_120
```

This compiles the Rust `#[device]` functions and verifies the generated `.ll` file
has clean export names, `@llvm.used`, and `!nvvmir.version`.

### Step 2: Build LTOIR, link, and run GPU tests (C++ side)

```bash
cd cuda-oxide/crates/rustc-codegen-cuda/examples/cpp_consumes_rust_device/cuda-caller
./run_test.sh
```

The script handles:
1. Compile Rust NVVM IR → LTOIR (libNVVM)
2. Compile C++ caller kernel → LTOIR (nvcc)
3. Link both LTOIRs → cubin (nvJitLink)
4. Build and run the C++ test runner

Expected output:

```text
=== Phase 2 Test: C++ calling Rust device functions via LTOIR ===

Device: NVIDIA GeForce RTX 5090 (sm_120)

--- Test 1: test_sqrt_clamp (fast_sqrt + clamp_f32) ---
  PASS
--- Test 2: test_safe_sqrt (transitive Rust device calls) ---
  PASS
--- Test 3: test_fma (monomorphized generics: fma_f32, fma_i32) ---
  PASS

✓ All tests PASSED — C++ successfully called Rust device functions via LTOIR!
```

## Prerequisites

- CUDA Toolkit (nvcc, libNVVM, nvJitLink)
- LTOIR tools from `device_ffi_test/tools/` (built automatically by `run_test.sh`)
- Blackwell+ GPU (sm_100+) — LTOIR requires NVVM 20 dialect

If your default host compiler is newer than the CUDA Toolkit supports, set
`NVCC_CCBIN` or `CUDAHOSTCXX` before running the example:

```bash
NVCC_CCBIN=/usr/bin/g++-15 cargo oxide run cpp_consumes_rust_device --emit-nvvm-ir --arch=sm_120
```

## File Structure

```text
cpp_consumes_rust_device/
├── Cargo.toml
├── src/
│   └── main.rs              # Rust #[device] functions + .ll verification
├── cuda-caller/
│   ├── caller_kernel.cu      # C++ kernels that call Rust device fns
│   ├── test_runner.cu        # C++ test runner (CUDA Driver API)
│   └── run_test.sh           # End-to-end build + test script
└── README.md
```

## C++ Usage Pattern

On the C++ side, Rust device functions are declared with `extern "C" __device__`:

```cpp
// Declarations — symbols resolved at link time from Rust LTOIR
extern "C" __device__ float fast_sqrt(float x);
extern "C" __device__ float clamp_f32(float val, float min_val, float max_val);

// C++ kernel calling Rust device functions
extern "C" __global__ void my_kernel(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = clamp_f32(fast_sqrt((float)idx), 0.0f, 10.0f);
    }
}
```

## Related

- `standalone_device_fn/` — Foundation: verifies standalone `#[device]` fn compilation
- `device_ffi_test/` — Phase 3: the reverse direction (Rust calling C++ device fns)
