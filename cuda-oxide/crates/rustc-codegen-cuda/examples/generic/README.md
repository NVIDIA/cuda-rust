# generic

## Generic Kernels - Type-Parameterized GPU Code

Tests whether the collector correctly handles monomorphized generic kernels. This is a key feature for writing reusable, type-safe GPU code.

## What This Example Does

- Defines generic kernels `scale<T>` and `add<T>` with trait bounds
- Host code calls `scale::<f32>` which triggers monomorphization
- Defines a captured closure inside a generic kernel body
- Verifies the monomorphized kernel executes correctly

## Key Concepts Demonstrated

### Generic Kernel Definition

```rust
#[kernel]
pub fn scale<T: Copy + Mul<Output = T>>(factor: T, input: &[T], mut out: DisjointSlice<T>) {
    let idx = thread::index_1d();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = input[idx.get()] * factor;
    }
}

#[kernel]
pub fn add<T: Copy + Add<Output = T>>(a: &[T], b: &[T], mut c: DisjointSlice<T>) {
    let idx = thread::index_1d();
    if let Some(c_elem) = c.get_mut(idx) {
        *c_elem = a[idx.get()] + b[idx.get()];
    }
}
```

### Monomorphization Trigger

```rust
// Type parameter on the typed module method forces monomorphization.
let module = kernels::load(&ctx)?;
// SAFETY: this is a 1D launch and both buffers contain N elements.
unsafe {
    module.scale::<f32>(
        &stream,
        LaunchConfig::for_num_elems(N as u32),
        factor,
        &input_dev,
        &mut output_dev,
    )
}?;
```

### How It Works

1. **Definition**: `scale<T>` is generic over any `Copy + Mul` type
2. **Instantiation**: `scale::<f32>` creates a concrete f32 version
3. **Collection**: rustc-codegen-cuda finds the monomorphized instance
4. **PTX Generation**: Backend generates PTX for the specific type

## Build and Run

```bash
cargo oxide run generic
```

## Expected Output

```text
=== Unified Generic Kernel Test ===

Launching scale::<f32> kernel...
  factor = 2.5
  N = 1024

✓ SUCCESS: All 1024 elements correct!
```

## Hardware Requirements

- **Minimum GPU**: Any CUDA-capable GPU
- **CUDA Driver**: 11.0+

## Supported Trait Bounds

Kernels can be generic over types that implement:
- `Copy` - Required for all kernel types (no heap allocation on GPU)
- `Add`, `Sub`, `Mul`, `Div` - Arithmetic operations
- `PartialOrd`, `PartialEq` - Comparisons
- Custom traits (if implemented for primitive types)

## Common Patterns

### Numeric Operations

```rust
#[kernel]
pub fn saxpy<T: Copy + Mul<Output = T> + Add<Output = T>>(
    a: T, x: &[T], y: &[T], mut out: DisjointSlice<T>
) {
    let idx = thread::index_1d();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = a * x[idx.get()] + y[idx.get()];
    }
}

// Use with different types (fields abbreviated for clarity)
// SAFETY: each raw config must be 1D and cover the supplied buffers.
// unsafe { module.saxpy::<f32>(&stream, config, a, &x, &y, &mut out) }?;
// unsafe { module.saxpy::<f64>(&stream, config, a, &x, &y, &mut out) }?;
// unsafe { module.saxpy::<i32>(&stream, config, a, &x, &y, &mut out) }?;
```

### With Closures

```rust
#[kernel]
pub fn map<T: Copy, F: Fn(T) -> T + Copy>(f: F, input: &[T], mut out: DisjointSlice<T>) {
    let idx = thread::index_1d();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = f(input[idx.get()]);
    }
}
```

## Generated PTX

For `scale::<f32>`:

```ptx
// Mangled name includes type info
.entry scale_f32 (
    .param .f32 %factor,
    .param .u64 %input_ptr, .param .u64 %input_len,
    .param .u64 %out_ptr, .param .u64 %out_len
) {
    // f32 multiplication
    mul.f32 %f_result, %f_input, %f_factor;
}
```

For `scale::<i32>`:

```ptx
.entry scale_i32 (...) {
    // i32 multiplication
    mul.lo.s32 %r_result, %r_input, %r_factor;
}
```

## Limitations

- Generic types must be `Copy` (no heap allocation on GPU)
- Some trait bounds may not have GPU implementations
- Very complex generics may increase compile time
