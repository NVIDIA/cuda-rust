# error

## Error Test - Compiler Error Handling

This example is intentionally designed to **FAIL compilation**. It tests that the compiler produces helpful error messages for unsupported operations.

## What This Example Does

Contains two kernels:
1. **valid_f64_to_f32_kernel**: Valid f64→f32 cast (compiles correctly)
2. **unsupported_format_kernel**: Uses `format_args!` (should FAIL)

## Purpose

This is a **negative test** - it verifies that:
- The compiler detects unsupported GPU operations
- Error messages are clear and actionable
- Compilation fails gracefully

## Key Concept: Unsupported Operations

### Valid Code

```rust
#[kernel]
pub fn valid_f64_to_f32_kernel(a: &[f64], b: &[f64], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(c_elem) = c.get_mut(idx) {
        // Valid: explicit f64 to f32 cast
        *c_elem = (a[idx.get()] + b[idx.get()]) as f32;
    }
}
```

### Invalid Code (Should Fail)

```rust
#[kernel]
pub fn unsupported_format_kernel(a: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(c_elem) = c.get_mut(idx) {
        // INVALID: format_args! uses core::fmt which requires:
        // - Dynamic dispatch (dyn Trait)
        // - Heap allocation
        // - Complex runtime machinery
        let _formatted = core::format_args!("{}", a[idx.get()]);
        *c_elem = a[idx.get()];
    }
}
```

## Build and Run

```bash
cargo oxide run error
```

## Expected Output

**Compilation should FAIL** with an error message like:

```text
error: Unsupported function call in kernel: core::fmt::Arguments::new_v1
  --> error/src/main.rs:39:27
   |
39 |         let _formatted = core::format_args!("{}", a[idx.get()]);
   |                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   |
   = note: format_args! uses core::fmt which is not supported on GPU
```

If you see "This example is intentionally broken..." at runtime, something went wrong - the compilation should have failed.

## Why format_args! Doesn't Work on GPU

`format_args!` expands to code that requires:

1. **Dynamic dispatch**: `&dyn fmt::Display` trait objects
2. **Complex runtime**: `Arguments` struct with function pointers
3. **String handling**: May allocate or use static strings

None of these are available in GPU kernels which require:
- Static dispatch only
- No heap allocation
- No string operations

## Other Unsupported Operations

| Operation                      | Why It Fails              |
|--------------------------------|---------------------------|
| `println!`, `format!`          | Uses fmt machinery        |
| `panic!` (with message)        | Uses fmt machinery        |
| `Box::new()`                   | No device-side allocator  |
| `Vec::push()`                  | No device-side allocator  |
| `&dyn Trait`                   | Dynamic dispatch          |
| Recursive functions            | May exceed stack          |
| Floating-point rounding modes  | Not all supported         |

## Alternatives for Debugging

Instead of `println!`:

```rust
// Write to output array for inspection
if let Some(debug_elem) = debug_out.get_mut(idx) {
    *debug_elem = some_value;
}

// Use debug::trap() for fatal errors
if bad_condition {
    debug::trap();
}

// Use gpu_assert! for assertions
gpu_assert!(value >= 0);
```

## How to Fix "Unsupported Operation" Errors

1. **Identify the operation**: Check error message for function name
2. **Find the source**: Often it's a macro expansion (format!, println!, etc.)
3. **Replace with GPU-safe alternative**:
   - Debugging → write to output buffer
   - String formatting → compute values instead
   - Assertions → `gpu_assert!` or `debug::trap()`
   - Collections → fixed-size `SharedArray` or input slices

## Valid Type Conversions

The example includes a valid f64→f32 kernel to show what DOES work:

```ptx
// Generated PTX for valid cast
ld.global.f64 %fd1, [...];   // Load f64
ld.global.f64 %fd2, [...];   // Load f64
add.f64 %fd3, %fd1, %fd2;    // Add as f64
cvt.rn.f32.f64 %f1, %fd3;    // Convert to f32
st.global.f32 [...], %f1;    // Store f32
```
