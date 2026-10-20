# The cuda-oxide Book

```{image} _static/images/banner-light.png
:alt: cuda-oxide: write CUDA (SIMT) kernels in pure Rust
:align: center
:width: 640px
:class: only-light mb-4
```

```{image} _static/images/banner-dark.png
:alt: cuda-oxide: write CUDA (SIMT) kernels in pure Rust
:align: center
:width: 640px
:class: only-dark mb-4
```

**cuda-oxide** is a Rust-to-CUDA compiler that lets you write (SIMT) GPU kernels in safe(ish), idiomatic Rust. It compiles standard Rust code directly to PTX — no DSLs, no foreign language bindings, just Rust.

:::{note}
This book assumes familiarity with the Rust programming language, including ownership, traits, and generics. Later chapters on async GPU programming also assume working knowledge of `async`/`.await` and runtimes like tokio.

For a refresher, see [The Rust Programming Language](https://doc.rust-lang.org/book/), [Rust by Example](https://doc.rust-lang.org/rust-by-example/), or the [Async Book](https://rust-lang.github.io/async-book/).
:::

---

## Project Status

The v0.1.0 release is an early-stage alpha: **expect bugs, incomplete features, and API breakage** as we work to improve it. We hope you'll try it and help shape its direction by sharing feedback on your experience.

---

## 🚀 Quick start

```rust
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[i] + b[i];
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    // SAFETY: this package owns the embedded device bundle for `kernels`.
    let module = unsafe { kernels::load(&ctx).unwrap() };

    let a = DeviceBuffer::from_host(&stream, &[1.0f32; 1024]).unwrap();
    let b = DeviceBuffer::from_host(&stream, &[2.0f32; 1024]).unwrap();
    let mut c = DeviceBuffer::<f32>::zeroed(&stream, 1024).unwrap();

    let prepared = module
        .prepare_vecadd(LaunchConfig1D::new(1024u32.div_ceil(256), 256, 0))
        .unwrap();
    module
        .vecadd(&stream, &prepared, &a, &b, &mut c)
        .unwrap();

    let result = c.to_host_vec(&stream).unwrap();
    assert_eq!(result[0], 3.0);
}
```

Build and run with `cargo oxide run vecadd` upon installing the [prerequisites](getting-started/installation.md). The same launch-contract pattern is what `cargo oxide new` scaffolds; see [Writing Your First Kernel](getting-started/hello-gpu.md).

:::{note}
`#[cuda_module]` embeds the generated device artifact into the host binary and
generates a typed `kernels::load` function plus one launch method per kernel.
Kernel arguments are type-checked. A declared `#[launch_contract]` unlocks the
safe `PreparedLaunch` path (`prepare_*` + typed launch) described in
[Launching Kernels](gpu-programming/launching-kernels.md). A raw `LaunchConfig`
call remains available as an unsafe escape hatch when you need a one-off
geometry that the contract does not cover. The lower-level
`load_kernel_module` and unsafe `cuda_launch!` APIs remain available when you
need to load a specific sidecar artifact or build custom launch code.
:::

---

## Why cuda-oxide?

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 🦀  Rust on the GPU
Write GPU kernels with Rust's type system and ownership model.
Safety is a first-class goal, but GPUs have subtleties — read about
[the safety model](gpu-safety/the-safety-model.md).
:::

:::{grid-item-card} 💎  A SIMT Compiler
Not a DSL. A custom rustc codegen backend that compiles
pure Rust to PTX.
:::

:::{grid-item-card} ⚡  Async Execution
Compose GPU work as lazy `DeviceOperation` graphs.
Schedule across stream pools. Await results with `.await`.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2
:caption: Getting Started

getting-started/installation
getting-started/hello-gpu
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Writing GPU Programs

gpu-programming/execution-model
gpu-programming/kernels-and-device-functions
gpu-programming/memory-and-data-movement
gpu-programming/virtual-memory-and-peer-access
gpu-programming/launching-kernels
gpu-programming/kernel-families
gpu-programming/closures-and-generics
gpu-programming/error-handling-and-debugging
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Safety on the GPU

gpu-safety/the-safety-model
gpu-safety/bounds-checks
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Async GPU Programming

async-programming/the-device-operation-model
async-programming/combinators-and-composition
async-programming/scheduling-and-streams
async-programming/concurrent-execution
async-programming/overlapping-transfers-and-compute
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Building a Real Application

projects/async-mlp-pipeline
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Advanced GPU Features

advanced/shared-memory-and-synchronization
advanced/warp-level-programming
advanced/tensor-memory-accelerator
advanced/matrix-multiply-accelerators
advanced/cluster-programming
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Inside the Compiler

compiler/architecture-overview
compiler/pliron
compiler/rustc-public
compiler/rustc-codegen-cuda
compiler/mir-importer
compiler/compiler-optimizations
compiler/mlir-dialects
compiler/lowering-pipeline
compiler/adding-new-intrinsics
compiler/catalog-generated-intrinsics
compiler/fuzzing-and-differential-testing
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Appendix

appendix/building-from-source
appendix/api-quick-reference
appendix/supported-features
appendix/cuda-cpp-comparison
appendix/ecosystem
appendix/glossary
```
