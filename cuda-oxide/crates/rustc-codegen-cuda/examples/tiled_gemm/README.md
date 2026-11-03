# tiled_gemm

## Tiled GEMM - Shared Memory Optimization

High-performance matrix multiplication using shared memory tiling. Reduces global memory traffic by ~16x compared to naive GEMM.

## What This Example Does

- 1024×1024 matrix multiplication with 16×16 tiles
- Cooperative tile loading into shared memory
- Demonstrates ~10-20x speedup over naive approach
- Reports memory bandwidth improvement

## Key Concepts Demonstrated

### Shared Memory Tiles

```rust
use cuda_device::thread::Runtime2DIndex;

const TILE_SIZE: usize = 16;

#[kernel]
pub fn sgemm_tiled(
    m: u32, n: u32, k: u32,
    alpha: f32, a: &[f32], b: &[f32], beta: f32,
    mut c: DisjointSlice<f32, Runtime2DIndex>,  // C is M x N, runtime stride
) {
    // Shared memory tiles for A and B (16x16 = 256 elements each)
    static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut TILE_B: SharedArray<f32, 256> = SharedArray::UNINIT;

    // ... tiled computation ...
}
```

### Tiled Algorithm

```rust
let num_tiles = (k_size + TILE_SIZE - 1) / TILE_SIZE;

for tile in 0..num_tiles {
    // 1. Cooperatively load tiles into shared memory
    TILE_A[ty * TILE_SIZE + tx] = A[row][tile_start + tx];
    TILE_B[ty * TILE_SIZE + tx] = B[tile_start + ty][col];

    sync_threads();  // Wait for all loads

    // 2. Compute partial dot product from tile
    // (actual code uses flat indexing: TILE_A[ty * TILE_SIZE + i])
    for i in 0..TILE_SIZE {
        sum += TILE_A[ty][i] * TILE_B[i][tx];
    }

    sync_threads();  // Wait before loading next tile
}
```

### Cooperative Loading

```rust
// Thread (tx, ty) in block loads:
// - One element of A: A[row][tile_start + tx]
// - One element of B: B[tile_start + ty][col]

// After sync_threads(), ALL 256 elements are available to ALL threads
```

## Build and Run

```bash
cargo oxide run tiled_gemm
```

## Expected Output

```text
=== Unified Tiled GEMM with Shared Memory ===
Matrix dimensions: 1024x1024 * 1024x1024 = 1024x1024
alpha = 1, beta = 0

Initializing matrices...
Grid: (64, 64), Block: (16, 16)
Tile size: 16x16 = 256 elements per tile
Shared memory per block: 2048 bytes

Warmup...
Running 10 iterations...

=== Performance ===
Average time: ~2-10 ms
Throughput:   ~1000-5000 GFLOPS

Memory reduction: 16.0x fewer global reads than naive

Verifying (sampling 100 elements)...
Max error: <1e-6

✓ SUCCESS: Tiled GEMM computed correctly!
```

## Hardware Requirements

- **Minimum GPU**: Any CUDA-capable GPU
- **Shared Memory**: 48KB+ (we use 2KB per block)
- **CUDA Driver**: 11.0+

## Why Tiling Works

### Without Tiling (Naive)

Each thread computing C[row, col] reads:
- K elements from A row (each element read once per row thread)
- K elements from B column (each element read once per column thread)

Total global reads: `M * N * K * 2 = 2 billion` (for 1024³)

### With Tiling

Each block of 256 threads:
- Loads 256 elements of A (one per thread) → shared
- Loads 256 elements of B (one per thread) → shared
- Each thread reuses all 256 elements

Global reads per block per tile: 512 (instead of 256 * 32 = 8192)
Reduction: **16x** (= TILE_SIZE)

## Memory Access Pattern

```text
Global Memory                 Shared Memory
    A                            TILE_A
┌───────────────┐            ┌─────────┐
│     row       │──load──→   │ 16 × 16 │
│  (K elements) │            │ floats  │
└───────────────┘            └─────────┘
                                  ↓
                             256 threads
                             read from
                             shared memory
                             (100x faster!)
```

## Performance Comparison

| Implementation | Time (ms) | GFLOPS    | Global Reads |
|----------------|-----------|-----------|--------------|
| Naive          | 15-50     | 100-500   | 2B           |
| **Tiled**      | 2-10      | 1000-5000 | 125M         |
| cuBLAS         | 0.1-0.5   | 10000+    | Optimized    |

## Tile Size Tradeoffs

| Tile Size   | Shared Memory | Threads/Block | Reduction |
|-------------|---------------|---------------|-----------|
| 8×8         | 512 B         | 64            | 8x        |
| **16×16**   | 2048 B        | 256           | **16x**   |
| 32×32       | 8192 B        | 1024          | 32x       |

Larger tiles = better memory reuse, but:
- More shared memory per block
- Fewer concurrent blocks per SM
- Diminishing returns beyond ~32

## Generated PTX

```ptx
// Shared memory declarations
.shared .align 4 .b8 TILE_A[1024];  // 256 * 4 bytes
.shared .align 4 .b8 TILE_B[1024];

// Tile loop
loop:
    // Load A tile
    ld.global.f32 %f_a, [%rd_a + %offset];
    st.shared.f32 [%smem_a + %smem_off], %f_a;

    // Load B tile
    ld.global.f32 %f_b, [%rd_b + %offset];
    st.shared.f32 [%smem_b + %smem_off], %f_b;

    // Synchronize
    bar.sync 0;

    // Compute from shared memory (fast!)
    inner_loop:
        ld.shared.f32 %f_tile_a, [%smem_a + ...];
        ld.shared.f32 %f_tile_b, [%smem_b + ...];
        fma.rn.f32 %f_sum, %f_tile_a, %f_tile_b, %f_sum;

    bar.sync 0;  // Before next tile load
```

## Further Optimizations

This example is educational. Production GEMM adds:
- Bank conflict avoidance (padding)
- Vectorized loads (float4)
- Prefetching
- Register blocking
- Tensor cores (WMMA/tcgen05)

See `tcgen05_matmul` for tensor core implementation.
