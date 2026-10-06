# Useful Mental Models

cuTile Rust kernels operate on tiles. The compiler and runtime map those tiles
onto CUDA threads and memory.

## Tile-Based Programming

CUDA C++ kernels usually start from individual threads and explicit thread indices:

```cpp
__global__ void add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

cuTile Rust starts from tile-shaped regions of data:

```rust
#[cutile::entry()]
fn add<const S: [i32; 2]>(
    c: &mut Tensor<f32, S>,
    a: &Tensor<f32, { [-1, -1] }>,
    b: &Tensor<f32, { [-1, -1] }>,
) {
    let tile_a = a.load_like(c);
    let tile_b = b.load_like(c);
    c.store(tile_a + tile_b);
}
```

The kernel describes what happens to one tile-shaped region. The compiler and runtime map that work onto CUDA execution units.

## Tile Programs and Tile Blocks

A tile program is the logical unit of concurrent execution. Each tile program runs the entry function once and processes one region of the output. It can query its per-axis grid coordinates and extents:

```rust
let pid0 = program_id(0);
let n0 = num_programs(0);
```

The terms tile program, tile block, and tile thread refer to the same logical unit. The API uses `program_id(axis)` and `num_programs(axis)`; the tuple forms `get_tile_block_id()` and `get_num_tile_blocks()` return all three axes at once.

Tile blocks that fit on available Streaming Multiprocessors run at the same time. The full set of tile blocks is concurrent: their relative order is unspecified, and kernels must not depend on one tile block running before another unless an explicit synchronization mechanism exists outside the kernel.

## Grid Geometry

Partitioning a tensor creates a logical grid. A `[128, 256]` tensor partitioned with `[32, 64]` creates `(4, 4, 1)` tile blocks:

```text
Tensor shape:    [128, 256]
Partition shape: [ 32,  64]
Grid:            (ceil(128 / 32), ceil(256 / 64), 1)
```

Tensor dimension 0 maps to grid `x`, dimension 1 maps to grid `y`, and dimension 2 maps to grid `z`. For tensors of lower rank, trailing grid dimensions are 1.

Inside a kernel, the program id selects the logical region:

```rust
let pid_m = program_id(0);
let part_x = x.partition(shape![BM, BK]);
let tile_x = part_x.load([pid_m, k_tile]);
```

For mutable outputs, the selected sub-tensor is passed directly to the tile block.

## Memory Hierarchy

NVIDIA GPUs have several memory levels:

| Memory level | Role |
|---|---|
| Registers | Fastest storage; tiles live here during computation |
| Shared memory | Fast on-chip storage shared within a hardware block |
| L2 cache | Hardware-managed cache shared across SMs |
| HBM | Large global memory where tensors live |

In cuTile Rust, you load from tensors in HBM and compute on tiles in registers. The Tile IR compiler and runtime decide how to stage data through shared memory, caches, threads, warps, Tensor Cores, and Tensor Memory Accelerator (TMA) instructions when those mechanisms are useful.

```rust
let tile = x.load_like(z); // HBM -> tile.
let y = tile * 2.0f32;           // Register-resident computation.
z.store(y);                      // Tile -> HBM.
```

You still choose tile shapes and access patterns. The compiler handles the thread-level mapping.

## Coalescing and Strides

GPUs move memory most efficiently when nearby threads access nearby addresses. Tile loads are designed to produce coalesced access patterns for regular layouts, so a row-major contiguous tile is usually a good starting point.

Strided access can reduce effective bandwidth because memory requests become scattered:

```text
Good: memory[0], memory[1], memory[2], ...
Bad:  memory[0], memory[1024], memory[2048], ...
```

When performance is poor, check whether the algorithm's logical access pattern defeats contiguous tile loads.

## Concurrency and Parallelism

Parallelism means work is running at the same time on different hardware units. Concurrency means independent work is in progress and may be scheduled in any order. A kernel launch has both: some tile blocks run in parallel on available SMs, and all tile blocks are concurrent from the programmer's perspective.

Host-side `DeviceOp` composition uses the same idea. Independent operations can overlap on different streams. Dependent operations must be chained or placed on the same stream to force ordering.

## Coming from CUDA or Triton

Common CUDA and Triton concepts have these cuTile Rust counterparts:

| Low-level concern | cuTile Rust expression |
|---|---|
| Thread index math | Partitioned tensors and tile block ids |
| Shared memory staging | Tile loads and compiler-managed staging |
| Tensor Core placement | `mma` / `mmaf_scaled` |
| Kernel pipeline composition | `DeviceOp` combinators and CUDA graphs |
| External kernels | PTX/CUBIN interop through `AsyncKernelLaunch` |

Use [Interoperability](interoperability.md) when an algorithm needs a hand-written CUDA kernel or a PTX/CUBIN artifact from another compiler.
