# printf

Tests the `gpu_printf!` macro for formatted output from GPU kernels.

## Features Tested

- **Basic types**: integers (i32, u32, u64), floats (f32, f64), booleans
- **Default conversions**: `{}` prints its argument by type, `%llu` for a `u64`
  above `i64::MAX`, `%f` for an `f32` or `f64` binding or slice element, `%p`
  for a raw pointer; `{:.2}` on a float prints two decimals
- **Format specifiers**: hex (`{:x}`, `{:X}`), octal (`{:o}`), scientific (`{:e}`, `{:E}`), compact (`{:g}`, `{:G}`)
- **Flags**: left-justify (`{:-}`), sign (`{:+}`), space (`{: }`), alternate (`{:#}`), zero-pad (`{:0}`)
- **Width and precision**: `{:8}`, `{:.2}`, `{:8.2}`
- **Thread-indexed output**: Each thread prints its data
- **Return value**: `gpu_printf!` returns number of arguments on success, negative on error

## Running

```bash
cargo oxide run printf
```

## Expected Output

```text
=== GPU Printf Test (Unified) ===

--- Test 1: Integer formats ---
=== Integer Tests ===
Signed i32: -42
Unsigned u32: 255
Unsigned u64: 1311768467463790320
Hex (lower): ff
Hex (upper): FF
Hex with prefix: 0xff
Octal: 100
Octal with prefix: 0100

--- Test 2: Float formats ---
=== Float Tests ===
f32 default: 3.141593
f64 default: 2.718282
Precision .2: 3.14
Precision .6: 3.141593
Scientific (lower): 1.234568e+06
Scientific (upper): 1.234568E+06
Compact (g): 0.000123
Compact (G): 1.23457E+06

... (more tests)

--- Test 8: Default conversions ---
=== Default Conversion Tests ===
u64::MAX: 18446744073709551615
f32 binding: 3.141593
f64 binding: 2.718282
data[0]: 1.100000
Precision .2 on f32: 3.14
Zero-pad u64: 00000042
Pointer: 0x7f3a9cc00000
Mixed: -42 3.141593 18446744073709551615 2.72 ff

=== ALL PRINTF TESTS PASSED ===
```

The pointer value changes from run to run.

## How It Works

The `gpu_printf!` macro:

1. **Compile-time**: Translates the Rust format string (`{}`) to a C format
   string. A placeholder without a type character takes its conversion from the
   argument's type (`%d`, `%llu`, `%f`, `%p`); one with a type character
   (`{:x}`, `{:.2f}`) casts the argument to the matching C type
2. **Compile-time**: Generates argument packing code with C vararg promotion rules
3. **Runtime**: Calls CUDA's `vprintf(format, args)` on the GPU
4. **Synchronization**: Output appears after `cudaDeviceSynchronize()` or similar

## Notes

- Printf output is buffered on the GPU and flushed on host synchronization
- Default buffer size is 1MB; increase with `cudaDeviceSetLimit(cudaLimitPrintfFifoSize, size)`
- Output order across threads is non-deterministic
- Use sparingly in performance-critical code due to serialization overhead
