# CUDA Rust

CUDA Rust is the home for accessing NVIDIA's CUDA platform from Rust. It hosts two
GPU programming models, both written in idiomatic Rust, plus the shared host-side
crates they build on.

- [cuda-oxide](https://nvidia.github.io/cuda-rust/cuda-oxide/latest/) — a rustc
  codegen backend that exposes the CUDA **SIMT** programming model. `#[kernel]`
  functions compile to PTX from ordinary Rust; host and device code live in one
  file and build with one `cargo oxide build`.
- [cutile](https://nvidia.github.io/cuda-rust/cutile/latest/) — a tile-based system
  that exposes the CUDA **Tile** programming model. Kernels operate on tiles of
  tensors; `#[cutile::module]` embeds a captured Rust AST that JIT-compiles through
  CUDA Tile IR to cubin at launch.
- `cuda-core`, `cuda-async`, `cuda-bindings` — host-side runtime crates: contexts,
  streams, device buffers, events, and low-level driver bindings.

Both models extend Rust's ownership rules across the GPU launch boundary. Mutable
buffers are partitioned into disjoint pieces before launch, immutable buffers are
shared, and the generated launchers keep those borrows alive while GPU work is in
flight. A kernel that would race is, in most cases, a kernel that does not compile.

:::{note}
CUDA Rust is an early-stage research project. Expect bugs, incomplete features, and
API breakage. Both books assume working knowledge of Rust — ownership, traits, and
generics.
:::

## Picking a model

The two models differ in what you write, not in what they can express.

**cuda-oxide — SIMT.** You write code for one thread and reason about the grid
yourself: thread and block indices, shared memory, warp and cluster operations,
TMA. Compilation is ahead of time, Rust straight to PTX. This is the model CUDA C++
uses, so it is the shorter path if you are porting existing kernels or need direct
control over the hardware.

**cutile — Tile.** You write code for one tile and let the compiler place it. You
partition a tensor, and the launch grid follows from the partition rather than from
indices you compute. Compilation is just-in-time, through CUDA Tile IR. This is the
shorter path when the problem is already shaped as tiles of tensors and you want
the scheduling handled for you.

If you are unsure, start with cuda-oxide.

The models interoperate. A cutile Tile kernel and a cuda-oxide SIMT kernel can run
on the same CUDA stream over shared device tensors, so a pipeline can use whichever
model fits each stage.

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
