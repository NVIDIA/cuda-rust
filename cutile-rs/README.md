<!-- # cuTile Rust -->

<div align="center">

<img src="assets/logo.svg" alt="cuTile Rust" width="380">

[![Crates.io](https://badgen.net/crates/v/cutile)](https://crates.io/crates/cutile)
[![Build](https://img.shields.io/github/actions/workflow/status/NVlabs/cutile-rs/pr.yml?branch=main&event=push&label=build)](https://github.com/NVlabs/cutile-rs/actions/workflows/pr.yml)
[![Docs](https://img.shields.io/badge/docs-book-blue.svg)](https://nvlabs.github.io/cutile-rs/)

</div>

cuTile Rust (`cutile-rs`) is a tile-based system for writing memory-safe,
data-race-free GPU kernels in Rust. It extends Rust's ownership rules across host
and device: mutable outputs are split into disjoint pieces, while read-only inputs
can be shared. Kernels are JIT-compiled through CUDA Tile IR. The same operations
can run synchronously, with `async`/`await`, or as CUDA graph replay.

## Project Status
We are excited to release this research project as a demonstration of how GPU programming can be made available in the Rust ecosystem. The software is in an early stage and under active development: you should expect bugs, incomplete features, and API breakage as we work to improve it. That being said, we hope you'll be interested to try it in your work and help shape its direction by providing feedback on your experience.

Please check out [CONTRIBUTING.md](CONTRIBUTING.md) if you're interested in contributing.

## Quick Start

```rust
use cutile::prelude::*;

#[cutile::module]
mod kernel {
    use cutile::core::*;

    #[cutile::entry()]
    fn add<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tx = x.load_like(z);
        let ty = y.load_like(z);
        z.store(tx + ty);
    }
}

fn main() -> Result<(), Error> {
    let x = api::ones::<f32>(&[1024]);
    let y = api::ones::<f32>(&[1024]);
    let z = api::zeros::<f32>(&[1024]).partition([128]);

    let (z, _x, _y) = kernel::add(z, x, y).sync()?;
    let result = z.unpartition().to_host_vec().sync()?;
    assert_eq!(result, vec![2.0_f32; 1024]);
    Ok(())
}
```

The `#[cutile::module]` macro transforms `add` into a GPU kernel and generates a host-side launcher. The host code constructs lazy tensor operations, partitions the mutable output into 128-element chunks, and calls `.sync()` to JIT-compile and execute the kernel.

Launches return all runtime arguments in parameter order (`z, x, y` here), including inputs and scalars. `.unpartition().to_host_vec().sync()?` copies the partitioned tensor's contents into a vector on the host machine.

In the kernel signature, `z` is the exclusive mutable output; `x` and `y` are
shared read-only inputs. The kernel adds input tiles matching the output
partition and stores the result in `z`. The partition determines the launch
grid `(8, 1, 1)`: 1024÷128 = 8 tiles.

- Run a similar example via `cargo run -p cutile-examples --example saxpy`.
- [More kernel and host API examples](cutile-examples/examples).
- For NVIDIA Nsight Compute, Nsight Systems, and cuda-gdb workflows, see
  [Debugging and Profiling](cutile-book/guide/debugging-and-profiling.md).

## Setup

### Requirements

- **Rust:** stable 1.89+ (no nightly required).
- **Linux:** tested on Ubuntu 24.04.

GPU and emitted Tile IR requirements for cuTile Rust:

<!-- BEGIN TILE IR TARGETS -->
| GPU compute capability | Minimum Tile IR version |
|---|---|
| `sm_80`, `sm_86`, `sm_87`, `sm_88`, `sm_89` (Ampere / Ada) | 13.2 |
| `sm_90` (Hopper) | 13.3 |
| Blackwell `sm_100`, `sm_103`, `sm_110`, `sm_120`, `sm_121` | 13.2 |
| `sm_107` | 13.4 |
<!-- END TILE IR TARGETS -->

For DGX Spark / GB10 (`sm_121`), see the
[DGX Spark tutorial](https://nvlabs.github.io/cutile-rs/tutorials/12-dgx-spark-inference.html).

CUDA **13.3 is recommended**. FP4 packing and block-scaled MMA require 13.3.
GPUs below `sm_80` (such as `sm_70` and `sm_75`) are unsupported.

### Feature requirements

Raw features have additional requirements. Toolkit and architecture checks
are **both** required; an unsupported operation produces a source-located
JIT error before assembly.

<!-- BEGIN TILE IR REQUIREMENTS -->
| Raw feature | Requires |
|---|---|
| Allocation | Tile IR 13.3 |
| Gather/scatter views | Tile IR 13.3 |
| Strided views | Tile IR 13.3 |
| View atomic reduction | Tile IR 13.3 |
| FP4 packing | Tile IR 13.3 and `sm_100+` |
| Block-scaled MMA | Tile IR 13.3 and `sm_100+`; valid operand/scale configuration |
| `insert`, `fpowi`, GDC tokens, alias fence | Tile IR 13.4 |
| Saturating float-to-int, explicit pointer classification, view `inbounds` | Tile IR 13.4 |
| `f8e5m3fnu` | Tile IR 13.4 and `sm_107+` |
| Scaled MMA with `f8e5m3fnu` scales | Tile IR 13.4 and `sm_107` only |
| Programmatic dependent launch (unsafe, per launch) | Tile IR 13.4 and `sm_90+`; driver `cuLaunchKernelEx` support |
<!-- END TILE IR REQUIREMENTS -->

See the [version/feature matrix](cutile-book/reference/compatibility.md),
the [raw DSL reference](cutile-book/reference/dsl-api.md#raw-tile-ir-versioned-surface)
and the [launch contract](cutile-book/reference/host-api.md#programmatic-dependent-launch).

### Install

#### Rust

To install Rust:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable
```

#### CUDA

Install CUDA 13.3 for your OS by following the official instructions:
https://developer.nvidia.com/cuda-downloads

### Configure Environment

Set `CUDA_TOOLKIT_PATH` (or `CUDA_HOME`, consulted second) to your CUDA 13.3
install directory for a reproducible setup. If neither is set, cuTile
searches standard CUDA 13.4/13.3/13.2 install locations such as
`/usr/local/cuda-13.4`, `/usr/local/cuda-13.3`, `/usr/local/cuda-13.2`, `/usr/local/cuda-13`,
`/usr/local/cuda`, and `/opt/cuda`.

`CUTILE_TILEIRAS_PATH` is optional. If set, it overrides the `tileiras` binary
that would otherwise be taken from the selected CUDA toolkit. The selected
`tileiras` determines the emitted Tile IR bytecode version, and CUDA 13.2 and
13.3 keep their older wire layouts. Mixing a `tileiras` from one CUDA version
with a toolkit of another is not guaranteed to be compatible.

Example `.cargo/config.toml`:
```toml
[env]
CUDA_TOOLKIT_PATH = { value = "/usr/local/cuda-13", relative = false }
```

### Verifying Installation

Run the hello world example:

```bash
cargo run -p cutile-examples --example hello_world
```

If everything works, you should see: `Hello, I am tile <0, 0, 0> in a kernel with <1, 1, 1> tiles.`

## Via Nix

The repository includes a Nix flake. To enable flakes, add this to
`~/.config/nix/nix.conf`:
```
experimental-features = nix-command flakes
```

Run a command directly:
```bash
nix develop -c cargo run -p cutile-examples --example saxpy
```

Or open an interactive shell:
```bash
nix develop
# cutile-rs dev shell
#  ✓ CUDA  /nix/store/...-cuda-toolkit-13.3
#  ✓ Rust  1.90.0-nightly
```

The flake automatically locates host NVIDIA driver libraries on both NixOS and non-NixOS systems.

## Tests
- cuTile IR: `cargo test --package cutile-ir`
- cuTile Rust Compiler: `cargo test --package cutile-compiler`
- cuTile Rust Library: `cargo test --package cutile`
- Examples: run an individual example, for example `cargo run -p cutile-examples --example async_gemm`
- Benchmarks: `cargo bench`
- Everything: `./scripts/run_all.sh` (or pipe to a log file: `./scripts/run_all.sh 2>&1 | tee test_run.log`)

## Workspace Crates

```
cutile                 User-facing crate for authoring and executing tile kernels
├── cutile-macro
├── cutile-compiler
├── cuda-async
└── cuda-core

cutile-kernels         Reusable cuTile Rust kernels
└── cutile

cutile-macro           cuTile Rust proc-macro
└── cutile-compiler

cutile-compiler        Compiles cuTile Rust kernels to executables
├── cutile-ir
├── cuda-async
└── cuda-core

cutile-ir              Pure Rust Tile IR builder and bytecode writer

cuda-async             Async CUDA execution via async Rust
└── cuda-core

cuda-core              Idiomatic safe CUDA API
└── cuda-bindings

cuda-bindings          NVIDIA CUDA bindings
```

## Related Projects and References

- [Candle](https://github.com/huggingface/candle): Hugging Face's Rust ML framework, with [mixture-of-experts (MoE) kernels written in cuTile Rust](https://github.com/huggingface/candle/blob/main/candle-nn/src/moe/cutile.rs).
- [mistral.rs](https://github.com/EricLBuehler/mistral.rs): Rust LLM inference engine with [cuTile Rust kernels](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs-quant/src/cutile) for quantized linear layers and MoE, enabled by the optional `cutile` feature.
- [cuTile Python](https://github.com/nvidia/cutile-python): Python kernel programming with CUDA Tile.
- [TileGym](https://github.com/NVIDIA/TileGym): CUDA Tile kernel examples and tuning patterns, including a set of ready-to-use cuTile Rust kernels under [`ops/cutile_rs`](https://github.com/NVIDIA/TileGym/tree/main/src/tilegym/ops/cutile_rs).
- [cuda-oxide](https://github.com/NVlabs/cuda-oxide): NVlabs experimental Rust-to-CUDA compiler for writing SIMT-style GPU kernels in Rust.
- [CUDA Tile IR documentation](https://docs.nvidia.com/cuda/tile-ir/latest/index.html): CUDA Tile IR reference documentation.
- [CUDA documentation](https://docs.nvidia.com/cuda/): CUDA toolkit documentation.
- [Rust NVPTX backend](https://doc.rust-lang.org/rustc/platform-support/nvptx64-nvidia-cuda.html): rustc's target support for generating PTX for NVIDIA GPUs.

## Paper

[*Fearless Concurrency on the GPU*](https://arxiv.org/abs/2606.15991) evaluates the
runtime cost of safety and the performance of applications built with cuTile Rust.
On NVIDIA B200, element-wise operations reach 7 TB/s and `f16` GEMM reaches
2.1 PFlop/s (98% of cuBLAS). At `M=N=K=8192`, safe and unchecked Rust GEMM
perform within 0.1% of each other.

The paper also evaluates [Grout](https://github.com/huggingface/grout), a Qwen3
inference engine built with cuTile Rust in collaboration with Hugging Face.
Its batch-1 decode peaks at 171 tokens/s for Qwen3-4B on NVIDIA GeForce RTX 5090
and 82 tokens/s for Qwen3-32B on B200, on par with vLLM and SGLang.

The [reproducibility artifacts](cutile-benchmarks/paper/) use cuTile Rust 0.2.0.
The [Grout repository](https://github.com/huggingface/grout) contains the version
used for the paper.

## Citing

If you use cuTile Rust in research, please cite the paper:

```bibtex
@misc{elibol2026fearlessconcurrencygpu,
  title = {Fearless Concurrency on the GPU},
  author = {Elibol, Melih and Roesch, Jared and Gelado, Isaac and Buehler, Eric and Garland, Michael},
  year = {2026},
  eprint = {2606.15991},
  archivePrefix = {arXiv},
  primaryClass = {cs.PL},
  url = {https://arxiv.org/abs/2606.15991}
}
```

## License
All crates are licensed under the Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0
