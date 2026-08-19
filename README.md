# CUDA Rust

CUDA Rust is the home for accessing NVIDIA's CUDA platform from Rust. It hosts two
GPU programming models, both written in idiomatic Rust, plus the shared host-side
crates they build on.

* [cuda-oxide](https://nvidia.github.io/cuda-rust/cuda-oxide/latest/): A rustc
  codegen backend that exposes the CUDA **SIMT** programming model. `#[kernel]`
  functions compile to PTX from ordinary Rust — host and device code in one file,
  built with one `cargo oxide build`.
* [cutile](https://nvidia.github.io/cuda-rust/cutile/latest/): A tile-based system
  that exposes the CUDA **Tile** programming model. Kernels operate on tiles of
  tensors; `#[cutile::module]` embeds a captured Rust AST that JIT-compiles through
  CUDA Tile IR to cubin at launch.
* `cuda-core`, `cuda-async`, `cuda-bindings`: Host-side runtime crates — contexts,
  streams, device buffers, events, and low-level driver bindings.

Both models extend Rust's ownership rules across the GPU launch boundary: mutable
buffers are partitioned into disjoint pieces before launch, immutable buffers are
shared, and launchers keep those borrows alive while GPU work is in flight.

CUDA Rust is an early-stage research project. Expect bugs, incomplete features,
and API breakage.

## Picking a model

|  | cuda-oxide (SIMT) | cutile (Tile) |
| --- | --- | --- |
| You write | Per-thread code | Per-tile code |
| Indexing | Explicit thread/block indices | Implicit, from the partition |
| Compiles | Ahead of time, Rust → PTX | JIT, Rust AST → Tile IR → cubin |
| Reach for it when | You need direct control over threads, warps, shared memory, TMA, or cluster ops | Your problem is naturally shaped as tiles of tensors and you want the compiler to schedule them |

Coming from CUDA C++, SIMT is the familiar model. If you are unsure, start with
cuda-oxide.

The two interoperate: a cutile Tile kernel and a cuda-oxide SIMT kernel can run on
the same stream over shared device tensors.

## Repository layout

```
cuda-rust/
├── docs/            umbrella documentation
├── cuda-oxide/      SIMT model — compiler, device crates, book
├── cutile/          Tile model — compiler, tile crates, book
└── crates/          shared host-side crates
```

## Documentation

Start at [the CUDA Rust docs](https://nvidia.github.io/cuda-rust/). Each model has
its own book, with installation, guides, and reference.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for repository-wide practice — signing off
commits, pull requests, and CI. Each model has its own toolchain requirements;
those are documented in its book.

## License

CUDA Rust is licensed under the [Apache License 2.0](./LICENSE). Each component
ships a copy of the license alongside its sources so the license accompanies the
built artifact. The root `LICENSE` governs the repository as a whole.

| Component | License | License file |
| --- | --- | --- |
| `cuda-oxide` | Apache-2.0 | [`cuda-oxide/LICENSE`](./cuda-oxide/LICENSE) |
| `cutile` | Apache-2.0 | [`cutile/LICENSE`](./cutile/LICENSE) |

Third-party attributions are listed in
[`THIRD_PARTY_NOTICES`](./THIRD_PARTY_NOTICES).
