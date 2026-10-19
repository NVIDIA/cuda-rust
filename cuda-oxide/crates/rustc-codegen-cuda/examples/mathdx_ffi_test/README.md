# MathDx FFI Test

This example demonstrates calling NVIDIA MathDx device extension libraries
(cuFFTDx for FFT, cuBLASDx for GEMM) from cuda-oxide kernels via LTOIR linking.

## Overview

MathDx provides high-performance device-side mathematical operations:

- **cuFFTDx**: Device-side Fast Fourier Transform library
- **cuBLASDx**: Device-side BLAS (Basic Linear Algebra Subprograms) library

This example wraps MathDx C++ template functions as `extern "C"` device functions,
compiles them to LTOIR, and links them with cuda-oxide generated code.

## Prerequisites

1. **CUDA Toolkit 13.0+** with nvcc
2. **MathDx Library** - Download from: https://developer.nvidia.com/cublasdx-downloads
3. **cuda-oxide compiler** toolchain

If your default host compiler is newer than the CUDA Toolkit supports, set
`NVCC_CCBIN` or `CUDAHOSTCXX` before running the example:

```bash
NVCC_CCBIN=/usr/bin/g++-15 cargo oxide run mathdx_ffi_test --emit-nvvm-ir --arch=sm_120
```

## Directory Structure

```text
mathdx_ffi_test/
├── Cargo.toml
├── README.md
├── src/
│   └── main.rs                    # Rust kernels and test harness
├── extern-libs/
│   ├── build.sh                   # Build script for LTOIR
│   ├── cufftdx_wrappers.cu        # cuFFTDx C++ wrappers (main)
│   ├── cufftdx_wrappers_funcs.cu  # cuFFTDx additional functions
│   ├── cublasdx_wrappers.cu       # cuBLASDx C++ wrappers
│   ├── cuda_test_kernels.cu       # Debug/test kernels
│   ├── test_separate.cu           # Standalone CUDA validation test
│   └── README.md
└── tools -> ../device_ffi_test/tools  # Shared LTOIR tools
```

## Building

### Prerequisites

Set MathDx installation path (if not in default location):

```bash
export MATHDX_ROOT=/path/to/nvidia-mathdx-XX.YY.Z/nvidia/mathdx/XX.YY
```

### Step 1 (Optional): Manual LTOIR Build

The `cargo oxide run` command builds LTOIR automatically. For manual builds:

```bash
cd extern-libs
./build.sh sm_120       # Blackwell
./build.sh sm_90        # Hopper
./build.sh --test       # Also build CUDA C++ validation test
./build.sh --clean      # Clean and rebuild
```

This produces `*.ltoir` files for linking.

### Step 2: Build and Run

```bash
# From workspace root (builds LTOIR automatically if needed)
cargo oxide run mathdx_ffi_test --emit-nvvm-ir --arch=<your_arch>

# Examples:
cargo oxide run mathdx_ffi_test --emit-nvvm-ir --arch=sm_120  # Blackwell
cargo oxide run mathdx_ffi_test --emit-nvvm-ir --arch=sm_90   # Hopper
```

**Note:** `--emit-nvvm-ir` is required - this generates the `.ll` file that gets
compiled to LTOIR and linked with MathDx.

This single command:
1. Builds MathDx LTOIR (via `extern-libs/build.sh`)
2. Compiles Rust kernel to LLVM IR (`.ll`)
3. Compiles LLVM IR to LTOIR via libNVVM
4. Links all LTOIR with nvJitLink
5. Runs all tests

### Optional: libmathdx Feature (Option B)

This example also supports generating LTOIR at runtime using the `libmathdx` C API,
which eliminates the need for hand-written C++ wrapper code. This is optional and
requires additional setup.

**Prerequisites for libmathdx:**

1. Clone and build libmathdx:

   ```bash
   git clone https://gitlab-master.nvidia.com/cuda-hpc-libraries/device-libraries/libmathdx.git
   cd libmathdx
   mkdir build && cd build
   cmake .. -DCMAKE_INSTALL_PREFIX=../install
   make -j && make install
   ```

2. **Configure `LIBMATHDX_PATH`** (choose one method):

   **Option A: Edit `.cargo/config.toml` (recommended - persists across terminal sessions):**

   ```toml
   # In cuda-oxide/.cargo/config.toml, update this line:
   LIBMATHDX_PATH = "/your/path/to/libmathdx/install"
   ```

   **Option B: Export environment variable (temporary, current session only):**

   ```bash
   export LIBMATHDX_PATH=/path/to/libmathdx/install
   ```

3. Build with the feature enabled:

   ```bash
   cargo oxide run mathdx_ffi_test --emit-nvvm-ir --arch=<your_arch> --features libmathdx  # e.g., sm_120
   ```

This enables the `test_libmathdx_fft_16_roundtrip` test which uses FFT functions
generated entirely from Rust code via the libmathdx C API - no C++ wrappers needed!

## Wrapped Functions

### cuFFTDx (Thread-level FFT)

| Function                             | Description                         |
|--------------------------------------|-------------------------------------|
| `cufftdx_fft_8_c2c_f32_forward`      | 8-point forward FFT (C2C, float)    |
| `cufftdx_fft_8_c2c_f32_inverse`      | 8-point inverse FFT                 |
| `cufftdx_fft_8_storage_size`         | Query: storage size for 8-pt FFT    |
| `cufftdx_fft_8_elements_per_thread`  | Query: elements per thread (8-pt)   |
| `cufftdx_fft_16_c2c_f32_forward`     | 16-point forward FFT                |
| `cufftdx_fft_16_c2c_f32_inverse`     | 16-point inverse FFT                |
| `cufftdx_fft_16_storage_size`        | Query: storage size for 16-pt FFT   |
| `cufftdx_fft_16_elements_per_thread` | Query: elements per thread (16-pt)  |
| `cufftdx_fft_32_c2c_f32_forward`     | 32-point forward FFT                |
| `cufftdx_fft_32_storage_size`        | Query: storage size for 32-pt FFT   |

### Debug Functions

| Function                    | Description                              |
|-----------------------------|------------------------------------------|
| `debug_extern_double_array` | Debug helper: doubles array values       |

**Usage pattern (thread-level FFT):**

```rust
#[kernel]
fn my_fft_kernel(data: *mut f32) {
    // Each thread has its own local array
    let mut local: [f32; 16] = [0.0; 16];  // 8 complex = 16 floats
    
    // Load data into local array...
    
    // Execute FFT (each thread independently)
    unsafe { cufftdx_fft_8_c2c_f32_forward(local.as_mut_ptr()); }
    
    // Store results...
}
```

### cuBLASDx (Block-level GEMM)

| Function                                 | Description                          |
|------------------------------------------|--------------------------------------|
| `cublasdx_gemm_32x32x32_f32`             | 32x32x32 GEMM: C = A * B             |
| `cublasdx_gemm_32x32x32_f32_alphabeta`   | 32x32x32 GEMM: C = α*A*B + β*C       |
| `cublasdx_gemm_32x32x32_smem_size`       | Query: shared memory size (A,B,C)    |
| `cublasdx_gemm_32x32x32_smem_size_ab`    | Query: shared memory size (A,B only) |
| `cublasdx_gemm_32x32x32_block_dim_x/y/z` | Query: required block dimensions     |

**Usage pattern (block-level GEMM):**

```rust
#[kernel]
fn my_gemm_kernel(a: *const f32, b: *const f32, c: *mut f32) {
    // All 256 threads cooperate
    let smem: *mut i8 = DynamicSharedArray::<i8, 128>::get();
    
    // Execute GEMM (all threads must participate)
    unsafe { cublasdx_gemm_32x32x32_f32(a, b, c, smem); }
}

// Launch with: block_dim=(256,1,1), shared_mem=16KB
```

## Matrix Layouts

The cuBLASDx 32x32x32 GEMM uses:
- **A**: 32x32, row-major
- **B**: 32x32, col-major  
- **C**: 32x32, output

This matches the official cuBLASDx introduction example configuration.

## Tests

The example includes several tests:

| Test                              | Status  | Description                           |
|-----------------------------------|---------|---------------------------------------|
| `query_fft_config`                | ✅ Pass | Query cuFFTDx configuration values    |
| `query_gemm_config`               | ✅ Pass | Query cuBLASDx configuration values   |
| `debug_copy_through_local`        | ✅ Pass | Verify local array aliasing works     |
| `debug_extern_double_global`      | ✅ Pass | Verify extern writes to global memory |
| `debug_extern_double`             | ✅ Pass | Verify extern writes to local memory  |
| `test_fft_8_roundtrip`            | ✅ Pass | 8-point FFT forward + inverse         |
| `test_fft_16_roundtrip`           | ✅ Pass | 16-point FFT roundtrip                |
| `test_gemm_32x32x32`              | ✅ Pass | GEMM C = I*I                          |
| `test_gemm_32x32x32_alphabeta`    | ✅ Pass | GEMM C = αAB + βC                     |
| `test_libmathdx_fft_16_roundtrip` | ✅ Pass | libmathdx-generated FFT (Option B)*   |

\* Only runs with `--features libmathdx`

## Build Requirements

### Required nvcc Flags for cuBLASDx

cuBLASDx requires `--expt-relaxed-constexpr` to work correctly on Blackwell (sm_120).
Without this flag, the layout computation functions generate empty code.

```bash
nvcc --expt-relaxed-constexpr -arch=sm_120 ...
```

The `extern-libs/build.sh` script includes this flag automatically.

## How LTOIR Linking Works

### Option A: C++ Wrappers (Default)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MATHDX FFI PIPELINE (Option A)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  cuda-oxide (Rust)                      MathDx C++ (extern-libs/)           │
│  ─────────────────                      ─────────────────────────           │
│  src/main.rs                            cufftdx_wrappers.cu                 │
│  • Kernel definitions                   cublasdx_wrappers.cu                │
│  • #[device] extern "C" { ... }                                             │
│         │                                      │                            │
│         ▼                                      ▼                            │
│  cargo oxide run --emit-nvvm-ir              nvcc -dc -dlto                     │
│         │                                --expt-relaxed-constexpr           │
│         ▼                                      │                            │
│  mathdx_ffi_test.ll                            ▼                            │
│         │                               extern-libs/*.ltoir                 │
│         ▼                                      │                            │
│  tools/compile_ltoir (libNVVM)                 │                            │
│         │                                      │                            │
│         ▼                                      │                            │
│  mathdx_ffi_test.ltoir ────────────────────────┘                            │
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

### Option B: libmathdx C API (--features libmathdx)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MATHDX FFI PIPELINE (Option B)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  cuda-oxide (Rust)                      libmathdx C API                     │
│  ─────────────────                      ─────────────────                   │
│  src/main.rs                            libmathdx-sys (Rust bindings)       │
│  • Kernel definitions                   • cufftdxCreateDescriptor()         │
│  • #[device] extern "C" { ... }         • cufftdxSetOperator*()             │
│         │                               • commondxGetCodeLTOIR()            │
│         │                                      │                            │
│         │                                      ▼                            │
│         │                               generate_libmathdx_ltoir()          │
│         │                               (pure Rust, no C++ needed!)         │
│         │                                      │                            │
│         ▼                                      ▼                            │
│  cargo oxide run --emit-nvvm-ir              libmathdx_fft_*.ltoir          │
│         │                                      │                            │
│         ▼                                      │                            │
│  mathdx_ffi_test.ll                            │                            │
│         │                                      │                            │
│         ▼                                      │                            │
│  tools/compile_ltoir (libNVVM)                 │                            │
│         │                                      │                            │
│         ▼                                      │                            │
│  mathdx_ffi_test.ltoir ────────────────────────┘                            │
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

**Pipeline Steps:**

1. cuda-oxide compiles Rust kernels to LLVM IR (`.ll`)
2. `libNVVM` compiles `.ll` to LTOIR
3. `nvJitLink` links cuda-oxide LTOIR + MathDx wrapper LTOIR
4. Final `.cubin` contains all device code

The MathDx LTOIR includes heavily optimized template instantiations with
proper LLVM attributes for optimization.

## Troubleshooting

### "MathDx not found"

Set `MATHDX_ROOT` to your MathDx installation directory.

### Compilation errors in C++ wrappers

Ensure you have the correct CUDA Toolkit version (13.0+) and C++17 support.

### Shared memory errors at runtime

Increase `shared_mem_bytes` in the launch configuration. The 32x32x32 GEMM
typically needs ~8-16KB of shared memory.

## References

- [MathDx Documentation](https://docs.nvidia.com/cuda/mathdx/)
- [cuBLASDx Examples](https://docs.nvidia.com/cuda/cublasdx/)
- [cuFFTDx Examples](https://docs.nvidia.com/cuda/cufftdx/)
