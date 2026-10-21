# sharedmem

## Shared Memory - On-Chip Fast Memory

Demonstrates `SharedArray<T, N>` for block-level cooperation. Shared memory is ~100x faster than global memory and enables efficient data sharing between threads in the same block.

## What This Example Does

1. **shared_test**: Single SharedArray, each thread reads neighbor's value
2. **shared_dual**: Two SharedArrays, threads combine neighbor values from both
3. **shared_u64_align1**: A `SharedArray<u64, 2, 1>` asks for 1-byte alignment; its storage is still declared at 8 bytes
4. **shared_v4_natural**: A `SharedArray<V4, 4>` with a `#[repr(C, align(16))]` element and no `ALIGN` is declared at 16 bytes

The host checks every kernel's values, then reads `sharedmem.ptx` to confirm the two `.align` declarations.

## Key Concepts Demonstrated

### SharedArray Declaration

```rust
#[kernel]
pub fn shared_test(data: &[f32], mut out: DisjointSlice<f32>) {
    // Static shared memory - visible to all threads in the block
    static mut TILE: SharedArray<f32, 256> = SharedArray::UNINIT;

    let tid = thread::threadIdx_x() as usize;
    let gid = thread::index_1d().get();

    // Write to shared memory
    unsafe { TILE[tid] = data[gid]; }

    // Synchronize before reading
    thread::sync_threads();

    // Read neighbor's value
    unsafe {
        let neighbor_idx = (tid + 1) % 256;
        if let Some(out_elem) = out.get_mut(thread::index_1d()) {
            *out_elem = TILE[neighbor_idx];
        }
    }
}
```

### Thread Synchronization

- `thread::sync_threads()` is a **barrier** - all threads must reach it before any continue
- Critical for shared memory: ensures all writes are visible before reads
- Equivalent to `__syncthreads()` in CUDA C++

### Multiple Shared Arrays

```rust
static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
static mut TILE_B: SharedArray<f32, 256> = SharedArray::UNINIT;
```

Each `static mut` creates a separate shared memory allocation.

### Storage Alignment

Every shared array is declared at the larger of the requested `ALIGN` and the
element's ABI alignment. `ALIGN` must be 0 (natural) or a power of two; any
other value is a compile error that names the use site.

```rust
#[repr(C, align(16))]
#[derive(Copy, Clone)]
pub struct V4(pub [f32; 4]);

static mut DATA: SharedArray<u64, 2, 1> = SharedArray::UNINIT; // declared .align 8
static mut VS: SharedArray<V4, 4> = SharedArray::UNINIT;       // declared .align 16
```

Both alignment kernels touch the under-requested array before a smaller pad
array. Shared globals are inserted at the front of the module, so the first
array a kernel references lands last in the PTX, right behind the pad.
Declared at the requested 1 byte, or at the old blanket 8 for `V4`, ptxas
packed the array at an offset its own `st.shared.b64` and `st.shared.v4.b32`
could not use, and the launch failed with CUDA error 716, `misaligned
address`. With the ABI raise the arrays sit on 8- and 16-byte boundaries.

## Build and Run

```bash
cargo oxide run sharedmem
```

## Expected Output

```text
=== Unified Shared Memory Example ===

=== Test 1: Single SharedArray ===
Input data[0..5] = [0.0, 1.0, 2.0, 3.0, 4.0]
Output out[0..5] = [1.0, 2.0, 3.0, 4.0, 5.0]
✓ Single SharedArray: correct neighbor read

=== Test 2: Dual SharedArray ===
Input a[0..5] = [0.0, 1.0, 2.0, 3.0, 4.0]
Input b[0..5] = [100.0, 101.0, 102.0, 103.0, 104.0]
Output out[0..5] = [102.0, 104.0, 106.0, 108.0, 110.0]
✓ Dual SharedArray: correct neighbor sum from both tiles

=== Test 3: SharedArray<u64, 2, 1> behind a byte pad ===
Output out[0..4] = [1122334455667788, 112233445566778a, 112233445566778a, 112233445566778c]
✓ u64 storage requested at 1 byte reads back intact

=== Test 4: SharedArray<V4, 4> behind a u64 pad ===
Output out[0..4] = [3.5, 6.5, 9.5, 12.5]
✓ align(16) element at natural alignment reads back intact

=== Test 5: PTX storage declarations ===
✓ PTX declares the u64 array at .align 8 and the V4 array at .align 16

✓ SUCCESS: All shared memory tests passed!
```

## Hardware Requirements

- **Minimum GPU**: Any CUDA-capable GPU
- **Shared Memory**: 48KB+ (varies by GPU architecture)
- **CUDA Driver**: 11.0+

## Potential Errors

| Error                                  | Cause                    | Solution                              |
|----------------------------------------|--------------------------|---------------------------------------|
| `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`   | Too much shared memory   | Reduce SharedArray size               |
| Race condition / wrong values          | Missing `sync_threads()` | Add synchronization after writes      |
| Misaligned access                      | Accessing out-of-bounds  | Ensure indices stay within array size |

## Why Shared Memory Matters

| Memory Type  | Latency     | Bandwidth  | Scope         |
|--------------|-------------|------------|---------------|
| Registers    | 1 cycle     | N/A        | Single thread |
| **Shared**   | ~20 cycles  | ~1.5 TB/s  | Block         |
| L1 Cache     | ~30 cycles  | ~1.5 TB/s  | SM            |
| Global       | ~400 cycles | ~900 GB/s  | Device        |

Shared memory is programmer-managed cache - you control what data is kept close to the compute units.

## Common Patterns

### Tile Loading

```rust
// Each thread loads one element
TILE[tid] = global_data[gid];
sync_threads();
// Now all threads can read any element
```

### Neighbor Access

```rust
// After sync, safely read neighbor's data
let left = TILE[(tid + SIZE - 1) % SIZE];
let right = TILE[(tid + 1) % SIZE];
```

### Reduction

```rust
// Parallel reduction in shared memory
for stride in [128, 64, 32, 16, 8, 4, 2, 1] {
    if tid < stride {
        TILE[tid] += TILE[tid + stride];
    }
    sync_threads();
}
```

## Generated PTX

```ptx
// Shared memory declaration
.shared .align 4 .b8 TILE[1024];  // 256 * sizeof(f32)

// Storage declared at the element's ABI alignment, whatever ALIGN asked for
.shared .align 8 .b8 DATA[16];    // SharedArray<u64, 2, 1>
.shared .align 16 .b8 VS[64];     // SharedArray<V4, 4>

// Write
st.shared.f32 [%rd_tile + %r_offset], %f_val;

// Synchronization
bar.sync 0;

// Read
ld.shared.f32 %f_neighbor, [%rd_tile + %r_neighbor_off];
```
