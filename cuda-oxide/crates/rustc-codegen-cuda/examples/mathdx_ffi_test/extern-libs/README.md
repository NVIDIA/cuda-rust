# MathDx LTOIR Wrappers

This directory contains CUDA C++ wrapper files that expose MathDx template
functions as `extern "C"` device functions for linking with cuda-oxide.

## Files

| File                       | Description                          |
|----------------------------|--------------------------------------|
| `cufftdx_wrappers.cu`      | cuFFTDx thread-level FFT wrappers    |
| `cufftdx_wrappers_funcs.cu`| cuFFTDx debug/utility functions      |
| `cublasdx_wrappers.cu`     | cuBLASDx block-level GEMM wrappers   |
| `cuda_test_kernels.cu`     | CUDA C++ test kernels for validation |
| `test_separate.cu`         | Host-side test program               |
| `build.sh`                 | Unified build script                 |

## Building

```bash
# Set MathDx path if not in default location
export MATHDX_ROOT=/path/to/nvidia-mathdx-XX.YY.Z/nvidia/mathdx/XX.YY

# Build LTOIR only
./build.sh sm_120           # Blackwell
./build.sh sm_90            # Hopper

# Build LTOIR + CUDA C++ validation test
./build.sh --test sm_120

# Clean and rebuild
./build.sh --clean --test

# Show help
./build.sh --help
```

## Output Files

After building (not tracked in git, run `./build.sh` to generate):
- `*_wrappers.ltoir` - MathDx device code in LTOIR format
- `*_text.ltoir` - Human-readable LTOIR (for debugging)
- `test_separate.cubin` - Linked cubin (with `--test`)
- `test_separate` - Host test program (with `--test`)

## Build Requirements

### Required nvcc Flags

cuBLASDx requires specific nvcc flags to work correctly:

```bash
nvcc --expt-relaxed-constexpr -Wno-deprecated-declarations ...
```

- `--expt-relaxed-constexpr`: **REQUIRED** - enables constexpr functions in device code
- `-Wno-deprecated-declarations`: Suppresses MathDx internal deprecation warnings

Without `--expt-relaxed-constexpr`, cuBLASDx layout functions generate empty code
and GEMM operations silently fail. See `../bugs/BUG-004-*.md` for details.

## Architecture Notes

### SM Value Format

MathDx uses `SM<arch * 10>` internally:
- SM 90 (Hopper) → `SM<900>`
- SM 120 (Blackwell) → `SM<1200>`

The wrappers are configured for SM 120 (Blackwell). For other architectures,
modify the `using GEMM_32x32x32_F32 = GEMM_32x32x32_F32_T<...>` line in
`cublasdx_wrappers.cu`.

### cuFFTDx

Thread-level FFTs don't require SM-specific configuration in MathDx 25.12+.
The same LTOIR works across architectures (compiled with appropriate `-arch`).

### cuBLASDx

Block-level operations require SM-specific optimizations. The type definition
includes the target SM architecture.

## Adding New Configurations

### New FFT Size

Add to `cufftdx_wrappers.cu`:

```cpp
using FFT_NEW_SIZE = decltype(
    cufftdx::Thread() +
    cufftdx::Size<NEW_SIZE>() +
    cufftdx::Type<cufftdx::fft_type::c2c>() +
    cufftdx::Direction<cufftdx::fft_direction::forward>() +
    cufftdx::Precision<float>()
);

extern "C" __device__ void cufftdx_fft_NEW_c2c_f32_forward(float* data) {
    using FFT = FFT_NEW_SIZE;
    using complex_type = typename FFT::value_type;
    complex_type* cdata = reinterpret_cast<complex_type*>(data);
    FFT().execute(cdata);
}
```

### New GEMM Size

Add to `cublasdx_wrappers.cu`:

```cpp
template<unsigned int Arch>
using GEMM_MxNxK_F32_T = decltype(
    cublasdx::Size<M, N, K>() + 
    cublasdx::Precision<float>() + 
    cublasdx::Type<cublasdx::type::real>() +
    cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>() +
    cublasdx::Function<cublasdx::function::MM>() + 
    cublasdx::SM<Arch>() + 
    cublasdx::Block() +
    cublasdx::BlockDim<THREADS>()
);

using GEMM_MxNxK_F32 = GEMM_MxNxK_F32_T<1200>;

extern "C" __device__ void cublasdx_gemm_MxNxK_f32(...) {
    gemm_registers_impl<GEMM_MxNxK_F32>(...);
}
```

## LLVM Attributes

NVCC automatically emits appropriate LLVM attributes (`convergent`, `readnone`,
`readonly`, etc.) on the function definitions in LTOIR. When `nvJitLink`
performs LTO linking, it uses attributes from definitions, not declarations.

Therefore, the Rust `extern "C"` declarations don't need explicit attributes.
