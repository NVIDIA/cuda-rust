# CUDA Rust

CUDA Rust is NVIDIA's CUDA platform for Rust. It provides host-side crates for
working with the CUDA driver and device-side crates and tools for writing GPU
kernels in idiomatic Rust.

* [cuda-oxide](https://github.com/NVIDIA/cuda-rust/tree/main/cuda-oxide)
  ([book](cuda-oxide/)): A rustc compiler plugin, cargo helper utility, and
  crates to help users target the traditional CUDA **SIMT** (Single Instruction,
  Multiple Threads) kernel programming model. Here users have access to every
  detail of device side CUDA programming.
* [cutile-rs](https://github.com/NVIDIA/cuda-rust/tree/main/cutile-rs)
  ([book](cutile/)): A proc-macro plugin and crates to help users write CUDA
  kernels using the CUDA **Tile** programming model. Tensors in memory are
  partitioned into sub-tensors; a kernel loads tiles from them, computes on
  whole tiles at a time, and stores tiles back, and the compiler decides how
  tiles map onto each architecture.
* Runtime: [`cuda-bindings`](https://github.com/NVIDIA/cuda-rust/tree/main/cuda-bindings),
  [`cuda-core`](https://github.com/NVIDIA/cuda-rust/tree/main/cuda-core)
  (with [`cuda-core-derive`](https://github.com/NVIDIA/cuda-rust/tree/main/cuda-core-derive)),
  and [`cuda-async`](https://github.com/NVIDIA/cuda-rust/tree/main/cuda-async)
  provide a shared runtime: driver bindings, contexts, streams, device buffers,
  events, and lazy, composable device operations.

Both kernel programming models extend Rust's ownership rules across the GPU
launch boundary: mutable buffers are partitioned into disjoint pieces before
launch, immutable buffers are shared, and launchers keep those borrows alive
while GPU work is in flight.

:::{note}
CUDA Rust is an early-stage research project. Expect bugs, incomplete features,
and API breakage. That said, we hope you'll try it in your own work and help
shape its direction by sharing feedback on your experience.
:::

## Picking a model

|  | cuda-oxide (SIMT) | cutile-rs (Tile) |
| --- | --- | --- |
| Feels familiar to | CUDA C++ programmers: threads, blocks, shared memory | NumPy or PyTorch programmers: ndarray / tensor operations on multi-dimensional tiles |
| Safety | Memory safety per thread via disjoint slices; you control how threads synchronize and use shared memory | Data-race freedom by construction within a kernel: no explicit thread indexing, shared memory, or synchronization |
| Indexing | Explicit thread/block indices | Implicit via partitions |
| Compiles | Ahead of time, Rust → PTX | JIT at first launch, Rust → Tile IR → cubin |
| Toolchain | Pinned nightly, managed by `cargo oxide` | Stable Rust 1.89 or newer |
| Portability | CUDA architecture-specific; you control tuning for each GPU | CUDA architecture-agnostic; the compiler tunes per GPU |
| Reach for it when | You need direct control over threads, warps, shared memory, TMA, or cluster ops | Your problem is naturally expressed as tensor computations and you want the compiler to handle scheduling details |

**cutile-rs — Tile.** You write a kernel as a *tile program*: a function that loads tiles from tensors, 
computes on whole tiles using operations similar to NumPy or PyTorch, and stores tiles back into tensors.

Tensors are views into in-memory data. On the host, you partition a mutable tensor into disjoint sub-tensors. 
Each tile program gets an exclusive `&mut Tensor` for its sub-tensor; immutable inputs use shared `&Tensor` references. 
The partition determines the launch grid: You don’t need to calculate grid indices yourself but can if needed. 
There is no thread indexing, shared memory, or synchronization in the program. 
Together with disjoint mutable inputs, execution within a kernel is data-race free by construction.

Kernels compile just in time on first launch, from Rust through CUDA Tile IR to a cubin. 
The compiler optimizes your tile kernel to your target architecture, leaving your source code architecture-agnostic. 
Use cutile-rs when your problem fits naturally into tensor computations and you want the compiler to handle scheduling.

**cuda-oxide — SIMT.** You write code for one thread and manage the grid yourself, including thread and block indices, shared memory, warp and cluster operations, and TMA. 
This is the same programming model as CUDA C++. Use cuda-oxide when porting existing CUDA C++ kernels or when you need direct hardware control.

Disjoint slices give each thread exclusive access to its own elements, providing memory safety within each thread. 
You control how threads synchronize and use shared memory.

Kernels compile ahead of time from Rust to PTX using a pinned nightly toolchain managed by `cargo oxide`. 
You have direct control over how your kernels are tuned for each GPU architecture.

If you are unsure which model to start with, reach for cutile-rs first: the compiler optimizes your kernel for your specific architecture. 
Drop down to cuda-oxide when you need control over threads, warps, shared memory, or architecture-specific features, with direct control over tuning for each GPU architectur.


Kernels from both models interoperate: a cutile-rs Tile kernel and a cuda-oxide
SIMT kernel can run on the same stream over shared device buffers.

## Getting started

cutile-rs and cuda-oxide each have a book with installation instructions,
guides, and reference material. The runtime crates (`cuda-bindings`,
`cuda-core`, `cuda-async`) are documented in the cutile-rs book.

- [cuda-oxide book](cuda-oxide/)
- [cutile-rs book](cutile/)

## Contributing

See [CONTRIBUTING.md](https://github.com/NVIDIA/cuda-rust/blob/main/CONTRIBUTING.md)
for commit sign-off requirements, pull requests, and CI. Toolchain requirements
differ between the two: cutile-rs builds on stable Rust, cuda-oxide needs its
pinned nightly via `cargo oxide`; each book covers its own.

```{toctree}
:maxdepth: 2
:caption: Contents

cuda-oxide <https://nvidia.github.io/cuda-rust/cuda-oxide/>
cutile-rs <https://nvidia.github.io/cuda-rust/cutile/>
```
