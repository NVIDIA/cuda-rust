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

Both kernel programming models extend Rust's ownership rules across the GPU
launch boundary: mutable buffers are partitioned into disjoint pieces before
launch, immutable buffers are shared, and launchers keep those borrows alive
while GPU work is in flight.

CUDA Rust is an early-stage research project. Expect bugs, incomplete features,
and API breakage. That said, we hope you'll try it in your own work and help
shape its direction by sharing feedback on your experience.

## Picking a model

|  | cuda-oxide (SIMT) | cutile (Tile) |
| --- | --- | --- |
| You write | Per-thread code | Per-tile code |
| Indexing | Explicit thread/block indices | Implicit, from the partition |
| Compiles | Ahead of time, Rust → PTX | JIT, Rust AST → Tile IR → cubin |
| Reach for it when | You need direct control over threads, warps, shared memory, TMA, or cluster ops | Your problem is naturally shaped as tiles of tensors and you want the compiler to schedule them |

If you are unsure of which model to get started with, start with
cutile and drop down to the flexibility and control of cuda-oxide when you need it.

Kernels from both models interoperate: a cutile Tile kernel and a cuda-oxide SIMT kernel can run on
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
