# helper_fn

## Device Functions - Code Reuse in Kernels

Demonstrates the `#[device]` attribute for helper functions that are called from kernels. Device functions enable code reuse and modular kernel design.

## What This Example Does

- Defines a device function `vecadd_device` with `#[device]` attribute
- Kernel `vecadd_with_helper` delegates to the device function
- Host code launches the kernel and verifies results

## Key Concepts Demonstrated

### Device Function Declaration

```rust
#[device]
pub fn vecadd_device(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(c_elem) = c.get_mut(idx) {
        let i = idx.get();
        *c_elem = a[i] + b[i];
    }
}
```

### Calling Device Functions from Kernels

```rust
#[kernel]
pub fn vecadd_with_helper(a: &[f32], b: &[f32], c: DisjointSlice<f32>) {
    // Call the device function by its original name
    vecadd_device(a, b, c);
}
```

Note: `#[device]` generates a `cuda_oxide_device_<hash>_` prefixed internal symbol (the prefix is owned by `cuda-oxide/crates/reserved-oxide-symbols/`), but callers use the original name. The LLVM export layer strips the prefix in the final PTX.

### Function Inlining

- Device functions are typically inlined by the compiler
- No function call overhead in the generated PTX
- Enables clean code organization without performance cost

## Build and Run

```bash
cargo oxide run helper_fn
```

## Expected Output

```text
=== Unified Helper Function Example ===

Input vectors (first 5 elements):
  a = [0.0, 1.0, 2.0, 3.0, 4.0]
  b = [0.0, 2.0, 4.0, 6.0, 8.0]

Output vector (first 5 elements):
  c = [0.0, 3.0, 6.0, 9.0, 12.0]

✓ SUCCESS: All 1024 elements correct!
  (Kernel called device helper function successfully)
```

## Hardware Requirements

- **Minimum GPU**: Any CUDA-capable GPU
- **CUDA Driver**: 11.0+

## #[device] vs #[kernel]

| Attribute   | Entry Point     | Can Call       | Callable From  |
|-------------|-----------------|----------------|----------------|
| `#[kernel]` | Yes (GPU entry) | device, kernel | Host only      |
| `#[device]` | No              | device, kernel | Kernel/device  |

## Common Patterns

### Utility Functions

```rust
#[device]
fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    if x < lo { lo } else if x > hi { hi } else { x }
}

#[kernel]
pub fn apply_clamp(input: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(out_elem) = out.get_mut(idx) {
        // Call the device function by its original name. The macro
        // expands this to the reserved internal symbol behind the
        // scenes; users never type the mangled name themselves.
        *out_elem = clamp(input[idx.get()], 0.0, 1.0);
    }
}
```

### Modular Computation

```rust
#[device]
fn compute_partial(x: f32, y: f32) -> f32 { x * x + y * y }

#[device]
fn combine_partials(a: f32, b: f32) -> f32 { (a + b).sqrt() }

#[kernel]
pub fn complex_kernel(/* ... */) {
    let p1 = compute_partial(x1, y1);
    let p2 = compute_partial(x2, y2);
    let result = combine_partials(p1, p2);
}
```

## Generated PTX

The device function is typically inlined, so the PTX shows the operations directly:

```ptx
.entry vecadd_with_helper (...) {
    // Inlined code from vecadd_device
    mov.u32 %r1, %tid.x;
    // ... bounds check, load, add, store ...
}
```

If not inlined (for larger functions), a separate device function is generated:

```ptx
.func vecadd_device (...) {
    // Device function body
}
```
