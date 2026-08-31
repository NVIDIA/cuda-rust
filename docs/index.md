# CUDA Rust

This repository is the Rust language entrypoint for the NVIDIA CUDA platform.
This includes "host-side" crates for CUDA driver control and also "device-side"
crates for authoring NVIDIA kernels with idiomatic Rust code.

* [cuda-oxide](./cuda-oxide): A rustc compiler plugin, cargo helper utility, and
  crates to help users target the traditional CUDA **SIMT** (Single Instruction,
  Multiple Threads) kernel programming model. Here users have access to every
  detail of device side CUDA programming.
* [cutile](./cutile): A proc-macro plugin and rates to help users write CUDA
  kernels using the CUDA **Tile** programming model. Kernels operate on
  conceptual tiles memory at a time.
* `cuda-core`, `cuda-async`, `cuda-bindings`: Host-side runtime crates —
  contexts, streams, device buffers, events, and low-level driver bindings for
  launching work and managing the GPU.

:::{note}
CUDA Rust is an early-stage research project. Expect bugs, incomplete features,
and API breakage. That said, we hope you'll try it in your own work and help
shape its direction by sharing feedback on your experience.
:::

## Picking a model

|  | cuda-oxide (SIMT) | cutile (Tile) |
| --- | --- | --- |
| You write | Per-thread code | Per-tile code |
| Indexing | Explicit thread/block indices | Implicit, from the partition |
| Compiles | Ahead of time, Rust → PTX | JIT, Rust AST → Tile IR → cubin |
| Reach for it when | You need direct control over threads, warps, shared memory, TMA, or cluster ops | Your problem is naturally shaped as tiles of tensors and you want the compiler to schedule them |


**cutile — Tile.** You write code for tiles of data and let the compiler handle the tile to thread mapping. You are still responsible for partitioning
tensors into tiles, but then the launch grid follows from the partition rather than from
indices you compute. Compilation is just-in-time, through CUDA Tile IR. This is the
shorter path when the problem is already shaped as tiles of tensors and you want
the scheduling handled for you.


**cuda-oxide — SIMT.** You write code for one thread and reason about the grid
yourself: thread and block indices, shared memory, warp and cluster operations,
TMA. Compilation is ahead of time, Rust straight to PTX. This is the model CUDA C++
uses, so it is the shorter path if you are porting existing kernels or need direct
control over the hardware.
If you are unsure of which model to get started with, start with
cutile and drop down to the flexibility and control of cuda-oxide when you need it.

If you are familiar with CUDA C++ already, then start with cuda-oxide since the
programming model will be most familiar.

Kernels from both models interoperate: a cutile Tile kernel and a cuda-oxide
SIMT kernel can run on the same stream over shared device tensors.

## Getting started

Each model has its own book, with installation instructions, guides, and reference
material:

- [cuda-oxide: getting started](https://nvidia.github.io/cuda-rust/cuda-oxide/latest/getting-started/)
- [cutile: guide](https://nvidia.github.io/cuda-rust/cutile/latest/guide/)

## Contributing

Repository-wide practice — commit sign-off, pull requests, CI — is documented in
[CONTRIBUTING.md](https://github.com/NVIDIA/cuda-rust/blob/main/CONTRIBUTING.md).
Toolchain requirements differ per model and are documented in each book.

```{toctree}
:maxdepth: 2
:caption: Contents

cuda-oxide <https://nvidia.github.io/cuda-rust/cuda-oxide/latest/>
cutile <https://nvidia.github.io/cuda-rust/cutile/latest/>
```
