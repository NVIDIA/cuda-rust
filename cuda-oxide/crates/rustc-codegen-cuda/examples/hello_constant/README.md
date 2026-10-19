# hello_constant

## Minimal Kernel - Raw Pointer Support

The absolute simplest CUDA kernel: writes the constant value `42` to a memory location. This tests raw pointer passing to kernels without any slice abstractions.

## What This Example Does

- Allocates a single i32 on the device
- Launches a kernel that writes `42` to that location
- Verifies the result

## Key Concepts Demonstrated

### Raw Pointer Parameters

```rust
#[kernel]
pub unsafe fn hello_constant(out: *mut i32) {
    *out = 42;
}
```

Unlike slice parameters, raw pointers are passed directly without (ptr, len) pairs.

### Unsafe Kernels

- Kernels with raw pointers must be marked `unsafe`
- No bounds checking is performed
- The caller is responsible for ensuring the pointer is valid

### Simple Launch

```rust
// SAFETY: one thread writes through a valid one-element device pointer.
unsafe {
    module.hello_constant(
        stream.as_ref(),
        LaunchConfig::for_num_elems(1),
        out_dev.cu_deviceptr() as *mut i32,
    )
}?;
```

## Build and Run

```bash
cargo oxide run hello_constant
```

## Expected Output

```text
=== Unified Hello Constant Example ===

Launching kernel...
Output: 42

✓ SUCCESS: Kernel wrote 42 correctly!
```

## Hardware Requirements

- **Minimum GPU**: Any CUDA-capable GPU
- **CUDA Driver**: 11.0+

## When to Use This Pattern

Raw pointers are useful when:
- You need maximum flexibility
- You're interfacing with existing CUDA C++ code
- The slice abstraction adds unwanted overhead
- You're implementing low-level primitives

For most kernels, prefer `&[T]` (read-only) or `DisjointSlice<T>` (write) for safety.

## Generated PTX

```ptx
.entry hello_constant (
    .param .u64 %out
) {
    ld.param.u64 %rd1, [%out];
    mov.u32 %r1, 42;
    st.global.u32 [%rd1], %r1;
    ret;
}
```
