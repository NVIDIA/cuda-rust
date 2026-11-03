# libmathdx-sys

Raw FFI bindings to NVIDIA libmathdx - a runtime code-generation library for MathDx (cuFFTDx, cuBLASDx, cuSolverDx).

## What is libmathdx?

libmathdx provides a C API to generate optimized LTOIR (Link-Time Optimized IR) for MathDx device functions at runtime. Instead of writing C++ wrapper code and compiling with nvcc, you can:

1. Create a descriptor (e.g., `cufftdxCreateDescriptor`)
2. Configure parameters (size, precision, execution mode, etc.)
3. Generate LTOIR (`commondxGetCodeLTOIR`)
4. Link with nvJitLink

This enables cuda-oxide to call MathDx device functions without maintaining manual C++ wrappers.

## Prerequisites

You need to build libmathdx from source. See the [libmathdx repository](https://gitlab-master.nvidia.com/cuda-hpc-libraries/device-libraries/libmathdx) for instructions.

## Building

Configure `LIBMATHDX_PATH` to point to your libmathdx installation:

**Option A: Edit `.cargo/config.toml` (recommended - persists across sessions):**

```toml
# In cuda-oxide/.cargo/config.toml
LIBMATHDX_PATH = "/path/to/libmathdx/install"
```

**Option B: Export environment variable (temporary):**

```bash
export LIBMATHDX_PATH=/path/to/libmathdx/install
cargo build
```

The installation directory should contain:
- `include/libmathdx.h` (and other headers)
- `lib/libmathdx.so` (or `libmathdx_static.a`)

## API Overview

### Common (libcommondx)

- `commondxCreateCode` / `commondxDestroyCode` - Code handle management
- `commondxGetCodeLTOIR` / `commondxGetCodeLTOIRSize` - Extract LTOIR
- `commondxSetCodeOptionInt64` - Set compilation options (target SM, etc.)

### cuFFTDx (libcufftdx)

- `cufftdxCreateDescriptor` / `cufftdxDestroyDescriptor` - FFT descriptor
- `cufftdxSetOperatorInt64` - Configure FFT (size, direction, precision, etc.)
- `cufftdxGetTraitInt64` - Query traits (shared memory size, block dim, etc.)
- `cufftdxFinalizeCode` - Generate LTOIR

### cuBLASDx (libcublasdx)

- `cublasdxCreateDescriptor` / `cublasdxDestroyDescriptor` - GEMM descriptor
- `cublasdxSetOperatorInt64` - Configure GEMM (M, N, K, precision, etc.)
- `cublasdxGetTraitInt64` - Query traits (block dim, leading dims, etc.)
- `cublasdxFinalizeCode` - Generate LTOIR

### cuSolverDx (libcusolverdx)

Similar pattern for Cholesky, LU, QR decompositions.

## Example

```rust,ignore
use libmathdx_sys::*;

unsafe {
    // Create descriptor for 32-point forward FFT
    let mut desc: cufftdxDescriptor = 0;
    cufftdxCreateDescriptor(&mut desc);
    
    cufftdxSetOperatorInt64(desc, cufftdxOperatorType_t_CUFFTDX_OPERATOR_SIZE, 32);
    cufftdxSetOperatorInt64(desc, cufftdxOperatorType_t_CUFFTDX_OPERATOR_DIRECTION, 
                           cufftdxDirection_t_CUFFTDX_DIRECTION_FORWARD as i64);
    cufftdxSetOperatorInt64(desc, cufftdxOperatorType_t_CUFFTDX_OPERATOR_PRECISION,
                           commondxPrecision_t_COMMONDX_PRECISION_F32 as i64);
    cufftdxSetOperatorInt64(desc, cufftdxOperatorType_t_CUFFTDX_OPERATOR_SM, 900); // Hopper
    cufftdxSetOperatorInt64(desc, cufftdxOperatorType_t_CUFFTDX_OPERATOR_EXECUTION,
                           commondxExecution_t_COMMONDX_EXECUTION_BLOCK as i64);
    cufftdxSetOperatorInt64(desc, cufftdxOperatorType_t_CUFFTDX_OPERATOR_API,
                           cufftdxApi_t_CUFFTDX_API_LMEM as i64);
    
    // Create code handle and set target architecture
    let mut code: commondxCode = 0;
    commondxCreateCode(&mut code);
    commondxSetCodeOptionInt64(code, commondxOption_t_COMMONDX_OPTION_TARGET_SM, 900);
    commondxSetCodeOptionInt64(code, commondxOption_t_COMMONDX_OPTION_CODE_CONTAINER,
                              commondxCodeContainer_t_COMMONDX_CODE_CONTAINER_LTOIR as i64);
    
    // Finalize descriptor into code handle
    cufftdxFinalizeCode(code, desc);
    
    // Extract LTOIR bytes
    let mut lto_size: usize = 0;
    commondxGetCodeLTOIRSize(code, &mut lto_size);
    let mut lto = vec![0u8; lto_size];
    commondxGetCodeLTOIR(code, lto_size, lto.as_mut_ptr() as *mut _);
    
    // lto now contains LTOIR that can be linked with nvJitLink
    
    cufftdxDestroyDescriptor(desc);
    commondxDestroyCode(code);
}
```
