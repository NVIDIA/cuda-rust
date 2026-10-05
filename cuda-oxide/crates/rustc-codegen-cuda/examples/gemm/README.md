# gemm

## Naive GEMM - Matrix Multiplication Baseline

Demonstrates matrix multiplication C = α·A·B + β·C using the simplest possible algorithm. Each thread computes one element of the output matrix.

## What This Example Does

- Multiplies 1024×1024 matrices
- Each thread computes one dot product (one element of C)
- Measures performance in GFLOPS
- Verifies correctness against host computation

## Key Concepts Demonstrated

### 2D Thread Indexing

```rust
use cuda_device::thread::Runtime2DIndex;

#[kernel]
pub fn sgemm_naive(
    m: u32, n: u32, k: u32,
    alpha: f32,
    a: &[f32],  // M x K
    b: &[f32],  // K x N
    beta: f32,
    mut c: DisjointSlice<f32, Runtime2DIndex>,  // M x N, runtime stride = N
) {
    let n_sz = n as usize;
    let row = thread::index_2d_row();    // blockIdx.y * blockDim.y + threadIdx.y
    let col = thread::index_2d_col();    // blockIdx.x * blockDim.x + threadIdx.x

    // The row width comes from `c`, bound on the host to this same `n`.
    if let Some(c_idx) = thread::index_2d_runtime(&c) {
        // col < n_sz guaranteed by `Some` -- no manual check needed
        if row < m as usize {
            let mut sum = 0.0f32;
            for i in 0..k as usize {
                sum += a[row * k as usize + i] * b[i * n_sz + col];
            }
            if let Some(c_elem) = c.get_mut(c_idx) {
                *c_elem = alpha * sum + beta * (*c_elem);
            }
        }
    }
}
```

For kernels where the row stride is known at compile time, prefer
`thread::index_2d::<STRIDE>()` with a `DisjointSlice<f32, Index2D<STRIDE>>` --
the const generic encodes the stride in the witness type, turning a
mismatched-stride bug into a compile error instead of a contract.

### 2D Launch Configuration

```rust
let block_size = 16u32;  // 16x16 = 256 threads per block
let grid_x = (N as u32 + block_size - 1) / block_size;  // Ceil division
let grid_y = (M as u32 + block_size - 1) / block_size;

let cfg = LaunchConfig {
    grid_dim: (grid_x, grid_y, 1),   // 64x64 blocks
    block_dim: (block_size, block_size, 1),  // 16x16 threads
    shared_mem_bytes: 0,
};
```

### Performance Measurement

```rust
let flops = 2.0 * M as f64 * N as f64 * K as f64;  // multiply-add = 2 ops
let gflops = flops / (avg_ms / 1000.0) / 1e9;
```

## Build and Run

```bash
cargo oxide run gemm
```

## Expected Output

```text
=== Unified GEMM Example (Naive Implementation) ===
Matrix dimensions: 1024x1024 * 1024x1024 = 1024x1024
alpha = 1, beta = 0

Initialized CUDA context
Initializing matrices...
Grid: (64, 64), Block: (16, 16)

Warmup...
Running 5 iterations...

Performance: ~15-50 ms, ~150-700 GFLOPS

Verifying (sampling 100 elements)...
Max error: <1e-6

✓ SUCCESS!
```

## Hardware Requirements

- **Minimum GPU**: Any CUDA-capable GPU
- **CUDA Driver**: 11.0+
- **Memory**: ~12 MB for 1024×1024 matrices

## Why This is "Naive"

| Issue              | Impact                        | Solution                             |
|--------------------|-------------------------------|--------------------------------------|
| No shared memory   | High global memory traffic    | Use tiling (see `tiled_gemm`).       |
| Poor memory access | Low bandwidth utilization     | Coalesced access patterns            |
| No vectorization   | Underutilized memory bus      | Load f32x4 vectors                   |
| No tensor cores    | Missing hardware acceleration | WMMA/tcgen05 instructions            |

Typical performance:
- **Naive**: 100-500 GFLOPS
- **Tiled**: 500-2000 GFLOPS
- **cuBLAS**: 10,000-20,000 GFLOPS (on modern GPUs)

## Algorithm

For each element C[row, col]:

```text
C[row, col] = alpha * sum(A[row, k] * B[k, col] for k in 0..K) + beta * C[row, col]
```

Memory access pattern (per thread):
- Read: K elements from row of A (strided)
- Read: K elements from column of B (very strided!)
- Read/Write: 1 element of C

Total global memory reads per thread: 2K
For 1024×1024 with K=1024: 2048 reads × 1M threads = **2 billion** global reads

## Generated PTX

```ptx
.entry sgemm_naive (
    .param .u32 %m, .param .u32 %n, .param .u32 %k,
    .param .f32 %alpha,
    .param .u64 %a_ptr, .param .u64 %a_len,
    .param .u64 %b_ptr, .param .u64 %b_len,
    .param .f32 %beta,
    .param .u64 %c_ptr, .param .u64 %c_len
) {
    // Calculate row = blockIdx.y * blockDim.y + threadIdx.y
    // Calculate col = blockIdx.x * blockDim.x + threadIdx.x
    // Loop over K dimension
    // Accumulate dot product
    // Write result
}
```

## Next Steps

For better performance, see:
- `tiled_gemm` - Shared memory tiling (10-20x faster)
- `tcgen05_matmul` - Tensor core acceleration (100x+ faster)
