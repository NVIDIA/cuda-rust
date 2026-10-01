# CUDA Rust

CUDA Rust is NVIDIA's CUDA platform for Rust. It provides
host-side crates for working with the CUDA driver and device-side crates and
tools for writing GPU kernels in idiomatic Rust.

* [cuda-oxide](./cuda-oxide) ([book](./cuda-oxide/cuda-oxide-book)): A rustc
  compiler plugin, cargo helper utility, and crates to help users target the
  traditional CUDA **SIMT** (Single Instruction, Multiple Threads) kernel
  programming model. Here users have access to every
  detail of device-side CUDA programming.
* [cutile-rs](./cutile-rs) ([book](./cutile-rs/cutile-book)): A proc-macro plugin
  and crates to help users write CUDA kernels using the CUDA **Tile**
  programming model. Tensors in memory are
  partitioned into sub-tensors; a kernel loads tiles from them, computes on
  whole tiles at a time, and stores tiles back, and the compiler decides how
  tiles map onto each architecture.
* Runtime: [`cuda-bindings`](./cuda-bindings), [`cuda-core`](./cuda-core)
  (with [`cuda-core-derive`](./cuda-core-derive)), and [`cuda-async`](./cuda-async)
  provide a shared runtime: driver bindings, contexts, streams, device buffers,
  events, and lazy, composable device operations.

Both kernel programming models extend Rust's ownership rules across the GPU
launch boundary: mutable buffers are partitioned into disjoint pieces before
launch, immutable buffers are shared, and launchers keep those borrows alive
while GPU work is in flight.

CUDA Rust is an early-stage research project. Expect bugs, incomplete features,
and API breakage. That said, we hope you'll try it in your own work and help
shape its direction by sharing feedback on your experience.

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

If you are unsure which model to start with, reach for cutile-rs first: the compiler optimizes your kernel for your specific architecture. Drop down to cuda-oxide when you need control over threads, warps, shared memory, architecture-specific features, and tuning per GPU architecture.

Kernels from both models interoperate: a cutile-rs Tile kernel and a cuda-oxide SIMT
kernel can run on the same stream over shared device buffers.

## Repository layout

```
cuda-rust/
├── cuda-bindings/      runtime — CUDA driver API bindings
├── cuda-core/          runtime — contexts, streams, memory, modules, events, launch
├── cuda-core-derive/   runtime — derive macros for cuda-core
├── cuda-async/         runtime — lazy ops, futures, streams, graphs
├── cuda-oxide/         SIMT model — compiler, device crates, cuda-oxide-book
└── cutile-rs/          Tile model — compiler, tile crates, cutile-book
```

## Documentation

Start with the [CUDA Rust docs](https://nvidia.github.io/cuda-rust/). cutile-rs
and cuda-oxide each have a book with installation instructions, guides, and
reference material. The runtime crates (cuda-bindings, cuda-core, cuda-async)
are documented in the cutile-rs book.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for commit sign-off requirements, pull requests, and CI. Toolchain requirements differ between the two: cutile-rs builds on stable Rust, cuda-oxide needs its pinned nightly via `cargo oxide`; each book covers its own.

## License

CUDA Rust is licensed under the [Apache License 2.0](./LICENSE). The root
`LICENSE` covers the whole repository. Each component also includes a copy with
its sources and built artifacts.

| Component | License | License file |
| --- | --- | --- |
| `cuda-oxide` | Apache-2.0 | root [`LICENSE`](./LICENSE) |
| `cutile-rs` | Apache-2.0 | [`cutile-rs/LICENSE`](./cutile-rs/LICENSE) |
| `cuda-bindings`, `cuda-core`, `cuda-core-derive`, `cuda-async` | Apache-2.0 | root [`LICENSE`](./LICENSE) |

Third-party attributions are listed in
[`THIRD_PARTY_NOTICES`](./THIRD_PARTY_NOTICES).
