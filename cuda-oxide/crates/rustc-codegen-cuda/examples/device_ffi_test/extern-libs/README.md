# External Device Libraries

This directory contains CUDA C++ device functions that are compiled to LTOIR
and linked with cuda-oxide kernels.

## Files

| File                       | Description                              |
|----------------------------|------------------------------------------|
| `external_device_funcs.cu` | Simple device functions for testing      |
| `cccl_wrappers.cu`         | CUB wrapper functions                    |
| `build_ltoir.sh`           | Build script (compiles to LTOIR)         |

## Building

```bash
./build_ltoir.sh sm_120
```

This generates:
- `*.ltoir` - Binary LTOIR (for nvJitLink)
- `*_text.ltoir` - Text LTOIR (for inspection)
- `*.o` - Object files

## Adding New External Libraries

1. Create a new `.cu` file with `extern "C" __device__` functions
2. Add compilation to `build_ltoir.sh`
3. Declare the functions in `src/main.rs` using `#[device] extern "C"`

Example:

```cuda
// my_lib.cu
extern "C" __device__ float my_func(float x) {
    return x * 2.0f;
}
```

```rust
// src/main.rs
#[device]
extern "C" {
    fn my_func(x: f32) -> f32;
}
```

## Note on LLVM Attributes

**No NVVM attributes are needed on Rust extern declarations.** NVCC automatically
emits proper LLVM attributes (convergent, nounwind, memory, etc.) in the LTOIR
based on the function's semantics:

- Functions using `__syncthreads()`, shuffles → `convergent`
- Pure functions (no side effects) → appropriate memory attributes
- Functions marked `__forceinline__` → `alwaysinline`

When nvJitLink links the LTOIR files, it uses these attributes from the function
**definitions**. Attributes on our declarations are ignored.
