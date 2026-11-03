# Dynamic Shared Memory Example

Demonstrates `DynamicSharedArray<T, ALIGN>` for runtime-sized shared memory with user-specified alignment.

## What is Dynamic Shared Memory?

Unlike `SharedArray<T, N>` which requires compile-time known sizes, `DynamicSharedArray<T>`
allows the shared memory size to be specified at kernel launch time via
`LaunchConfig::shared_mem_bytes`.

This enables CUTLASS-style patterns where the same kernel PTX can be used with
different shared memory configurations without recompilation.

## Test Scenarios

This example tests four scenarios:

| Test           | Alignment    | Description                                    |
|----------------|--------------|------------------------------------------------|
| 1. Basic       | 16 (default) | Single partition, default alignment            |
| 2. Partitioned | 16 (default) | Two arrays via `offset()`                      |
| 3. Explicit    | 128          | TMA-compatible alignment                       |
| 4. Mixed       | 256 (max)    | Multiple calls with 16, 128, 256 → uses max    |

## API

```rust
use cuda_device::DynamicSharedArray;

// Default alignment (16 bytes, matches nvcc)
let smem: *mut f32 = DynamicSharedArray::<f32>::get();

// Explicit 128-byte alignment (required for TMA)
let tma_smem: *mut f32 = DynamicSharedArray::<f32, 128>::get();

// Get pointer at byte offset (for partitioning)
let smem_b: *mut f32 = DynamicSharedArray::<f32>::offset(1024);

// Mixed alignments in same kernel → compiler uses max
let a: *mut f32 = DynamicSharedArray::<f32>::get();           // 16
let b: *mut f32 = DynamicSharedArray::<f32, 128>::offset(x);  // 128
let c: *mut f32 = DynamicSharedArray::<f32, 256>::offset(y);  // 256
// Result: .extern .shared .align 256 ...
```

## Memory Partitioning

All calls to `DynamicSharedArray` in a kernel share the same underlying memory.
Use byte offsets to partition for multiple arrays:

```rust
// First array at offset 0
let array_a: *mut f32 = DynamicSharedArray::<f32>::get();

// Second array at offset 1024 bytes (after 256 f32s)
let array_b: *mut f32 = DynamicSharedArray::<f32>::offset(1024);
```

## Host-Side Launch

Specify the shared memory size in `LaunchConfig`:

```rust
let cfg = LaunchConfig {
    grid_dim: (blocks, 1, 1),
    block_dim: (256, 1, 1),
    shared_mem_bytes: 2048,  // 256 + 256 f32s
};
```

## Build and Run

```bash
cargo oxide run dynamic_smem
```

## Alignment

The `ALIGN` type parameter controls the base alignment:

| Alignment    | Use Case                           |
|--------------|------------------------------------|
| 16 (default) | General use, matches nvcc default  |
| 128          | TMA operations (required minimum)  |
| 256+         | Cache-line friendly patterns       |

When multiple `DynamicSharedArray` calls use different alignments in the same kernel,
the compiler uses the **maximum** alignment for the global.

When using `offset()`, ensure your byte offset maintains proper alignment for the
target type (e.g., 4-byte aligned for `f32`, 8-byte aligned for `f64`).

## PTX Output

Each kernel gets its own dynamic shared memory symbol. The hex component is
the compiling crate's stable id, which keeps one owner's symbol distinct when
several crates' PTX bundles are merged into one module at load time:

```ptx
; Different kernels with different alignments
.extern .shared .align 16 .b8 __dynamic_smem_00f223822842cf67_dynamic_smem_basic[];
.extern .shared .align 16 .b8 __dynamic_smem_00f223822842cf67_dynamic_smem_partition[];
.extern .shared .align 128 .b8 __dynamic_smem_00f223822842cf67_dynamic_smem_explicit_align[];
.extern .shared .align 256 .b8 __dynamic_smem_00f223822842cf67_dynamic_smem_mixed_align[];
```

This is different from static `SharedArray` which generates unique symbols per
allocation. The hex component is the compiling crate's stable id, which keeps
the symbols distinct when several crates' PTX bundles are merged into one
module at load time:

```ptx
.shared .align 4 .b8 __shared_mem_00f223822842cf67_0[1024];
.shared .align 4 .b8 __shared_mem_00f223822842cf67_1[512];
```
