# printf

Tests the `gpu_printf!` macro for formatted output from GPU kernels.

## Features Tested

- **Basic types**: integers (i32, u32, u64), floats (f32, f64), booleans
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
Unsigned u64: 16045690984833335038
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

=== ALL PRINTF TESTS PASSED ===
```

## How It Works

The `gpu_printf!` macro:

1. **Compile-time**: Translates Rust format string (`{}`) to C format string (`%d`)
2. **Compile-time**: Generates argument packing code with C vararg promotion rules
3. **Runtime**: Calls CUDA's `vprintf(format, args)` on the GPU
4. **Synchronization**: Output appears after `cudaDeviceSynchronize()` or similar

## Notes

- Printf output is buffered on the GPU and flushed on host synchronization
- Default buffer size is 1MB; increase with `cudaDeviceSetLimit(cudaLimitPrintfFifoSize, size)`
- Output order across threads is non-deterministic
- Use sparingly in performance-critical code due to serialization overhead
