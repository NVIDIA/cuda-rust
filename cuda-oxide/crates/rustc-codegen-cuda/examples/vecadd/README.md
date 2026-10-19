# vecadd

## Vector Addition - The "Hello World" of CUDA

This example demonstrates the simplest possible CUDA kernel: element-wise vector addition (`c[i] = a[i] + b[i]`). It showcases the unified compilation model where both host and device code exist in a single file.

## What This Example Does

- Allocates three vectors (a, b, c) of 1024 floats
- Launches a kernel where each thread adds one pair of elements
- Verifies the results on the host

## Key Concepts Demonstrated

### Unified Compilation

```rust
// No #![cfg_attr(cuda_device, no_std)] needed!
// Single file compiles to both PTX and native host code

#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    if let Some((c_elem, idx)) = c.get_mut_indexed() {
        let i = idx.get();
        *c_elem = a[i] + b[i];
    }
}
```

In this example the kernel lives inside a `#[cuda_module]` module, which also
generates the typed host-side loader and launch method.

### Thread Indexing

- `c.get_mut_indexed()` is the one-call form: it mints the per-thread `ThreadIndex` and resolves it to a `&mut T` in a single shot, returning `None` for out-of-bounds threads.
- The explicit two-step form `let idx = thread::index_1d(); c.get_mut(idx)` is also available when you need the index to address other slices.
- `idx.get()` returns the raw `usize` index for use against regular slices like `a` and `b`.

### Memory Safety

- `DisjointSlice<T, IndexSpace>` provides mutable access with the guarantee that each thread writes to a unique location.
- The `ThreadIndex` witness is `!Send + !Sync + !Copy + !Clone` and `'kernel`-scoped, so it can't be smuggled across threads.
- Input slices `&[f32]` are read-only on the device.

### Typed Module Loading

```rust
let module = kernels::load(&ctx)?;
// SAFETY: this is a 1D launch and all three buffers contain N elements.
unsafe {
    module.vecadd(
        &stream,
        LaunchConfig::for_num_elems(N as u32),
        &a_dev,
        &b_dev,
        &mut c_dev,
    )
}?;
```

The `vecadd` method is generated from the kernel signature, so the host call
has autocomplete for the kernel name and typed `DeviceBuffer<T>` arguments.
The raw configuration is still an explicit unsafe boundary because its launch
dimensions are not tied to the kernel's 1D indexing model.

## Build and Run

```bash
cargo oxide run vecadd
```

## Expected Output

```text
=== Unified Compilation Vector Addition ===

Input vectors (first 5 elements):
  a = [0.0, 1.0, 2.0, 3.0, 4.0]
  b = [0.0, 2.0, 4.0, 6.0, 8.0]

Output vector (first 5 elements):
  c = [0.0, 3.0, 6.0, 9.0, 12.0]

✓ SUCCESS: All 1024 elements correct!
```

## Hardware Requirements

- **Minimum GPU**: Any CUDA-capable GPU (Kepler or newer recommended)
- **CUDA Driver**: 11.0+

## Potential Errors

| Error                                | Cause                      | Solution                                  |
|--------------------------------------|----------------------------|-------------------------------------------|
| `CUDA_ERROR_NO_DEVICE`               | No GPU found               | Ensure NVIDIA driver is installed         |
| `Failed to load embedded CUDA module`| Embedded PTX was not found | Build through `cargo oxide run vecadd`    |
| `Kernel launch failed`               | Invalid launch config      | Ensure grid/block dims don't exceed limits|

## How It Works Under the Hood

1. **rustc** parses the file, generates MIR for everything
2. **rustc-codegen-cuda** intercepts codegen:
   - Finds `cuda_oxide_kernel_<hash>_vecadd` (from `#[kernel]`)
   - Routes it to mir-importer → PTX generation
   - Routes `main` and other host code to standard LLVM
3. Final binary contains both native host code and embedded PTX

## Generated PTX

The kernel generates approximately:

```ptx
.entry vecadd (
    .param .u64 %a_ptr, .param .u64 %a_len,
    .param .u64 %b_ptr, .param .u64 %b_len,
    .param .u64 %c_ptr, .param .u64 %c_len
) {
    // Calculate global thread index
    // Bounds check
    // Load a[idx], b[idx]
    // Add and store to c[idx]
}
```
