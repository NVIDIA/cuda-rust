# host_closure

## Host Closures - Passing Closures from Host to Kernel

Tests passing closures from host code to generic GPU kernels. This enables functional programming patterns where behavior is parameterized at launch time.

## What This Example Does

- Defines a generic `map<T, F>` kernel that applies a function to each element
- Host code passes closures with 0-4 captured variables
- Backend extracts captures and passes them as kernel parameters

## Key Concepts Demonstrated

### Generic Kernel with Closure Parameter

```rust
#[kernel]
pub fn map<T: Copy, F: Fn(T) -> T + Copy>(f: F, input: &[T], mut out: DisjointSlice<T>) {
    let idx = thread::index_1d();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = f(input[idx.get()]);
    }
}
```

### Launching with Host Closure

```rust
let factor = 2.5f32;

// SAFETY: this is a 1D launch and both buffers contain N elements.
unsafe {
    module.map::<f32, _>(
        stream.as_ref(),
        LaunchConfig::for_num_elems(N as u32),
        move |x: f32| x * factor, // _ infers closure type
        &input_dev,
        &mut output_dev,
    )
}?;
```

### How Capture Extraction Works

1. **Macro parses** the closure `move |x: f32| x * factor`
2. **Identifies captures**: `factor` is captured by value
3. **Scalarizes captures**: `factor` becomes a kernel parameter
4. **Kernel receives**: `(factor: f32, input: &[f32], out: DisjointSlice<f32>)`

## Build and Run

```bash
cargo oxide run host_closure
```

## Expected Output

```text
=== Unified Closure Kernel Test ===

Test 1: Single capture (scale by factor)
  factor = 2.5
  N = 1024
  ✓ SUCCESS: All 1024 elements correct!

Test 2: Multiple captures (affine transform)
  scale = 2, offset = 10
  ✓ SUCCESS: All 1024 elements correct!

Test 3: Zero captures (double each element)
  ✓ SUCCESS: All 1024 elements correct!

Test 4: Three captures (polynomial: a*x^2 + b*x + c)
  a = 0.5, b = 2, c = 1
  ✓ SUCCESS: All 1024 elements correct!

Test 5: Four captures (weighted sum: w1*x + w2 + w3*w4)
  w1 = 3, w2 = 5, w3 = 2, w4 = 7
  ✓ SUCCESS: All 1024 elements correct!

=== All Tests Complete ===
```

## Hardware Requirements

- **Minimum GPU**: Any CUDA-capable GPU
- **CUDA Driver**: 11.0+

## Closure Tests

| Test | Closure                            | Captures | Formula            |
|------|------------------------------------|---------:|:-------------------|
| 1    | `move \|x\| x * factor`            |        1 | `x * 2.5`          |
| 2    | `move \|x\| x * scale + offset`    |        2 | `x * 2.0 + 10.0`   |
| 3    | `\|x\| x * 2.0`                    |        0 | `x * 2.0`          |
| 4    | `move \|x\| a*x*x + b*x + c`       |        3 | `0.5*x² + 2*x + 1` |
| 5    | `move \|x\| w1*x + w2 + w3*w4`     |        4 | `3*x + 5 + 14`     |

## The Closure Story

### CUDA C++ Approach

```cpp
float factor = 5.0f;
auto scale = [=](float x) { return x * factor; };
kernel<<<1, N>>>(scale, input, output);
// nvc++ handles closure serialization automatically
```

### cuda-oxide Approach

```rust
let factor = 5.0f32;
// SAFETY: this is a 1D launch and both buffers contain N elements.
unsafe {
    module.map::<f32, _>(
        stream.as_ref(),
        LaunchConfig::for_num_elems(N as u32),
        move |x: f32| x * factor,
        &input,
        &mut output,
    )
}?;
// The typed launch method passes the closure value as a kernel argument
```

## Supported Closure Types

| Type     | Captures   | Callable       |
|----------|------------|----------------|
| `Fn`     | By ref     | Multiple times |
| `FnMut`  | By mut ref | Multiple times |
| `FnOnce` | By value   | Once           |

For GPU kernels, `FnOnce` with `Copy` bound is most common (closures are copied to each thread).

## Generated PTX

For `map::<f32, {closure capturing factor}>`:

```ptx
.entry map_f32_closure_factor (
    .param .f32 %factor,        // Extracted capture
    .param .u64 %input_ptr,
    .param .u64 %input_len,
    .param .u64 %out_ptr,
    .param .u64 %out_len
) {
    // Load input
    ld.global.f32 %f_x, [%input_ptr + %offset];
    // Apply closure: x * factor
    mul.f32 %f_result, %f_x, %factor;
    // Store output
    st.global.f32 [%out_ptr + %offset], %f_result;
}
```

## Common Patterns

### Parameterized Transforms

```rust
let threshold = 0.5f32;
// SAFETY: cfg is 1D and covers both buffers.
unsafe {
    module.map::<f32, _>(
        stream.as_ref(),
        cfg,
        move |x: f32| if x > threshold { 1.0 } else { 0.0 },
        &input,
        &mut output,
    )
}?;
```

### Runtime Configuration

```rust
fn launch_with_config(scale: f32, offset: f32, ...) {
    // SAFETY: cfg is 1D and covers both buffers.
    unsafe {
        module.map::<f32, _>(
            stream.as_ref(),
            cfg,
            move |x: f32| x * scale + offset,
            &input,
            &mut output,
        )
    }?;
}
```

### Composition

```rust
let f = |x: f32| x.sin();
let g = |x: f32| x * 2.0;
// SAFETY: cfg is 1D and covers both buffers.
unsafe {
    module.map::<f32, _>(
        stream.as_ref(),
        cfg,
        move |x: f32| g(f(x)), // sin(x) * 2
        &input,
        &mut output,
    )
}?;
```
