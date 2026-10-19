# Glossary

This glossary defines key terms as they are used throughout the cuTile Rust book.

---

## Tile

A multi-dimensional array fragment that lives in GPU **registers** during kernel execution. Tiles are the fundamental unit of computation in cuTile Rust: you load data from tensors into tiles, compute on tiles, and store the results back. Tiles have compile-time static shapes and are represented by the type `Tile<E, S>`, where `E` is the element type and `S` is the shape (e.g., `Tile<f32, {[16, 16]}>`).

## Tensor

A multi-dimensional array stored in GPU **global memory** (HBM). Tensors are passed as kernel arguments — `&Tensor<E, S>` for read-only inputs and `&mut Tensor<E, S>` for writable outputs. Tensors do not support direct arithmetic; data must first be loaded into tiles.

## Partition

A logical division of a tensor into a grid of equally sized sub-regions, each of which is processed by one tile block. The term "partition" appears on both the host side and the device side, but refers to different things.

**Host-side partition (mutable tensors only).** Calling `.partition([M, N])` on a `Tensor<T>` produces a `Partition<Tensor<T>>`. This is a host-side wrapper that records the `partition_shape` (the tile dimensions) alongside the original tensor. A host-side `Partition<Tensor<T>>` is what you pass to a kernel launcher in the position of a `&mut Tensor<E, S>` parameter. The `partition_shape` stored in the host-side `Partition` determines the static shape `S` that the kernel sees — for example, passing a `Partition` with `partition_shape = [32, 64]` means the kernel receives a `&mut Tensor<T, {[32, 64]}>`.

Only mutable tensors must be partitioned on the host side. This is because each `&mut Tensor` sub-region is written to by exactly one tile block, satisfying Rust's exclusive access requirement for mutable memory: at most one writer may access a given region at a time. By partitioning before launch, the system guarantees that no two tile blocks write to overlapping memory.

**Shared tensor references (no host-side partition required).** Read-only inputs are passed as `&Tensor<T>`, `Tensor<T>`, `Arc<Tensor<T>>`, or `&TensorView<T>` on the host side, corresponding to `&Tensor<E, S>` in the kernel signature. These do *not* need to be partitioned on the host side — multiple tile blocks may safely read from the same tensor or overlapping regions simultaneously, so there is no exclusive-access constraint to enforce. Instead, shared tensors are partitioned on the device side for greater flexibility in how they are accessed.

**Device-side partition.** Inside a kernel, calling `.partition(shape![M, N])` on a `&Tensor` creates a read-only `Partition` view that can be indexed to load individual tiles (e.g., `part.load([i, j])`). This is how shared tensor references are divided into tiles for loading. Because the partitioning happens on the device side, the same `&Tensor` can be partitioned in different ways — or accessed with different indexing patterns — within the same kernel. For example, in GEMM the input matrices `x` and `y` are each partitioned with a different shape inside the kernel body (`shape![BM, BK]` and `shape![BK, BN]` respectively), even though both were passed unpartitioned from the host.

The generated launcher code accepts `Partition<Tensor<T>>`, `Partition<&mut Tensor<T>>`, or a mapped partition for every `&mut Tensor` parameter, and `&Tensor<T>`, `Tensor<T>`, `Arc<Tensor<T>>`, or `&TensorView<T>` for every `&Tensor` parameter.

**Grid dimensions.** A host-side partition's grid is computed by dividing the tensor's shape by the partition shape, rounding up: `grid[i] = ceil(tensor_shape[i] / partition_shape[i])`. The result is mapped to a 3D tuple `(x, y, z)`, with trailing dimensions set to 1 for tensors of rank less than 3. For example, a `[128, 256]` tensor partitioned with `[32, 64]` produces a grid of `(4, 4, 1)`.

**Launch grid inference.** At kernel launch time, the launcher calls `.grid()` on each mutable output argument's host-side wrapper and collects the resulting grids. For ordinary `&mut Tensor` parameters, this is the host-side `Partition` grid. For `MappedPartitionMut` parameters, this is the mapped physical tile-block grid. If no explicit grid is specified via `.grid()` or `.const_grid()`, the launch grid is **inferred** from these grids. When multiple mutable output parameters are present, all inferred grids must match or the launch will fail with an error. This is how partitioning a tensor on the host side determines how many tile blocks the kernel runs.

## Tile Program

The basic unit of concurrent execution on the GPU — one instance of the kernel function, running once as a single logical thread of execution and operating on one partition of the data. This is the entity you write: a tile program has no visible internal threads; it loads tiles, computes on them, and stores results. Each tile program is identified by its per-axis grid index, `program_id(axis)`, with grid extents from `num_programs(axis)` (following Triton's `tl.program_id` / `tl.num_programs`); the tuple forms `get_tile_block_id()` and `get_num_tile_blocks()` return all three axes at once. Also called a *tile block* (emphasizing its place in the launch grid and its mapping onto hardware) or a *tile thread* (emphasizing the single-threaded programming model).

## Tile Block

The same unit as a [Tile Program](#tile-program), viewed from the launch-grid and hardware side: the launch grid is a 3-dimensional arrangement of tile blocks, and the cuTile Rust compiler maps each one to one or more underlying CUDA execution units (thread blocks, clusters, or warps) depending on the target architecture. From the programmer's perspective, a tile block remains a single-threaded context that processes one tile of data.

## Tile Thread

An alias for [Tile Program](#tile-program), emphasizing the single-threaded programming model: each tile thread executes the kernel function once as a single logical thread of execution.

## Concurrent Execution

Multiple tile blocks making progress independently. The GPU scheduler assigns
blocks to Streaming Multiprocessors (SMs) as resources become available. Some
blocks run at the same time while others wait; their relative execution order
is unspecified.

On the host side, concurrency also arises through CUDA streams and async/await: multiple `DeviceOp`s submitted to different streams can overlap in time, and the async runtime schedules them without requiring the programmer to specify an exact execution order.

## Parallel Execution

Multiple tile blocks executing at the same time on different SMs. A kernel
launch has both parallel and concurrent execution: blocks that fit on available
SMs run in parallel, while the rest wait to be scheduled. See
[The Rust Programming Language, Ch. 17](https://doc.rust-lang.org/book/ch17-00-async-await.html#parallelism-and-concurrency)
for the same distinction in host programming.

## Streaming Multiprocessor (SM)

The primary processing unit on an NVIDIA GPU. Each SM has its own registers, shared memory, and execution pipelines including Tensor Cores. Tile blocks are scheduled onto SMs by the GPU's hardware scheduler. A single SM can run multiple tile blocks concurrently if it has sufficient resources (registers, shared memory, thread slots). For architecture-specific details on SM resources, see the [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).

## Tensor Cores

Specialized hardware units (available on Volta architecture and later) that perform small matrix multiply-accumulate operations in a single instruction. The `mma()` intrinsic in cuTile Rust maps to Tensor Core instructions. Tensor Cores impose alignment requirements on tile dimensions (e.g., dimensions must typically be multiples of 8 or 16 depending on the element type).

## Block Scaling

A low-precision matrix multiply layout where one scale value is shared by a
fixed-size group of input values, usually along the K dimension. In the NVFP4
E4M3-scale layout used in this book, that ratio is one FP8 scale value per 16
logical FP4 values. cuTile Rust's `mmaf_scaled()` operation consumes logical
low-precision operand tiles plus their scale tiles, then accumulates into `f32`.

## FP4 E2M1FN

The 4-bit floating-point encoding named `f4e2m1fn` in cuTile Rust and
`f4E2M1FN` in Tile IR. The four bits encode one sign bit, two exponent bits,
and one mantissa bit. This is the logical FP4 operand type used by
`mmaf_scaled()`.

## Packed FP4 Pair

The byte-addressable storage representation for two `f4e2m1fn` values. In
cuTile Rust this is `f4e2m1fnx2`; kernels unpack it to logical `f4e2m1fn`
tiles before calling `mmaf_scaled()`.

## Nibble

A 4-bit half-byte. Packed FP4 storage uses two nibbles per byte: cuTile Rust's
`f4e2m1fnx2` stores the first logical FP4 value in the low nibble and the second
logical FP4 value in the high nibble.

## NVFP4

An NVIDIA FP4 inference format based on 4-bit E2M1 values plus block scale
tensors. The FP4 payload is commonly exchanged as byte-addressable data with two
values per byte. In cuTile Rust, that storage is represented by `f4e2m1fnx2`;
kernels unpack it to logical `f4e2m1fn` tiles before using block-scaled MMA.

## Global Memory (HBM)

The GPU's main memory — High Bandwidth Memory. Global memory has the highest capacity but is slower than shared memory and registers. `Tensor` data resides in global memory.

## Registers (RMEM)

The fastest storage on the GPU, private to each thread within a tile block. `Tile` data lives in registers during computation. Each SM has a fixed register file, so larger tiles consume more registers, potentially reducing occupancy.

## Shared Memory (SMEM)

On-chip memory shared among threads within a tile block. It is slower than
registers but faster than global memory. The
[Tile IR](https://docs.nvidia.com/cuda/tile-ir/latest/) compiler manages shared
memory for tile kernels. For capacity and latency details, see the
[CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).

## Const Generics

Compile-time constant parameters on kernel functions, such as `const BM: i32`. Const generics enable the compiler to optimize register allocation, unroll loops, and generate architecture-specific code. Changing a const generic value creates a new compiled variant. See also [Const Generic Arrays](#const-generic-arrays).

## Const Generic Arrays

An **extension to the Rust programming language** that allows const generic parameters to have array types — for example, `const S: [i32; 2]`. Standard Rust only supports scalar const generics (integers, `bool`, `char`), so this syntax is not valid in ordinary Rust code. The cuTile Rust compiler recognizes array const generics and uses them to propagate tile shapes through the type system at compile time.

Const generic arrays are the idiomatic way to parameterize a kernel over its tile shape:

```rust
#[cutile::entry()]
fn add<const S: [i32; 2]>(
    z: &mut Tensor<f32, S>,
    x: &Tensor<f32, {[-1, -1]}>,
    y: &Tensor<f32, {[-1, -1]}>,
) { ... }
```

Here `S` is inferred from the host-side partition shape passed at launch time. Because `S` is a compile-time constant, the compiler can specialize the generated code for each distinct shape. A new value of `S` creates a new compiled variant, just like scalar const generics.

## Dynamic Dimensions

Tensor shape dimensions specified as `-1` in the kernel signature (e.g., `Tensor<f32, {[-1, -1]}>`). Dynamic dimensions can vary across kernel launches without creating a new compiled variant. They carry no compile-time optimization benefit but provide flexibility for problem sizes that change often.

## JIT Compilation

cuTile Rust normally compiles a kernel entry function at first launch through a multi-stage pipeline: Rust AST → Tile IR bytecode → cubin. The generated launcher caches compiled kernels in memory using the active device and resolved kernel specialization. The specialization includes the entry function identity, type and const generics, compile options, optional constant grid, and tensor or scalar specialization hints. See [Compilation](../guide/jit-compilation.md) for the full cache behavior and the compile-only API.

## DeviceOp

A lazy description of GPU work — allocation, kernel launch, or data transfer — that is not executed until either `.sync_on(&stream)`, `.sync()`, or `.await` is invoked. `DeviceOp`s can be composed with `zip!`, `.then()`, `.map()`, `.shared()`, and `.first()`/`.last()` to build dataflow graphs before submitting GPU work. Kernel launchers accept `Tensor<T>`, `Arc<Tensor<T>>`, `&Tensor<T>`, scalars, and `DeviceOp` arguments directly via `IntoDeviceOp` and `KernelInput`. The return type matches the input type — you get back what you put in.

## DeviceFuture

A `DeviceFuture` is a future that has been assigned resources — specifically, a device stream on which to execute — but has not yet started GPU work. A `DeviceFuture` is created when a `DeviceOp` is scheduled (e.g., via `into_future()`), at which point the scheduling policy selects a stream. The actual GPU work is not submitted until the `DeviceFuture` is polled for the first time, which happens when you `.await` it or `tokio::spawn` it.

## Broadcasting

Replicating a smaller tile (or scalar) to match the shape of a larger tile. Broadcasting is a compile-time transformation — no extra memory is allocated. For example, `a.broadcast(y.shape())` expands a scalar into a tile matching `y`'s partition shape.

## Kernel Fusion

Combining multiple logical operations into a single kernel so that intermediate results stay in registers rather than being written to and read back from global memory. Fused softmax is a canonical example: find-max, subtract, exponentiate, sum, and divide are all performed in one kernel launch.

## Arithmetic Intensity

The ratio of compute operations (FLOPs) to memory operations (bytes transferred). Higher arithmetic intensity means better GPU utilization. A kernel with low arithmetic intensity (e.g., element-wise addition) is **memory-bound**; a kernel with high arithmetic intensity (e.g., matrix multiplication) is **compute-bound**.

## CUDA Stream

An ordered queue of GPU operations. Operations on the same stream execute in submission order; operations on different streams may execute concurrently. cuTile Rust's default async scheduling policy distributes work across a pool of four streams in round-robin fashion.

## Occupancy

The ratio of active warps to the maximum number of warps an SM can support. Higher occupancy generally improves the GPU's ability to hide memory latency by switching between warps. Occupancy is affected by register usage, shared memory usage, and thread block size.

## Warp

A group of 32 GPU threads that execute instructions in lockstep. Warps are the smallest scheduling unit on an SM. Tile sizes that are multiples of 32 align well with warp-level execution.
