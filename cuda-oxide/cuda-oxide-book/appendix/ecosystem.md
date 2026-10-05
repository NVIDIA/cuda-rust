# The Rust + GPU Ecosystem

cuda-oxide is one of several active efforts to make GPU computing accessible
from Rust. Each project tackles a different slice of the problem — graphics
shaders, implicit offload, CUDA programming, safe driver bindings — and the
landscape as a whole is moving forward quickly. This appendix is a brief map
of where cuda-oxide sits relative to its neighbors, and how the projects
relate to one another.

## Where cuda-oxide fits

| Project          | Approach                          | Target              | Scope                                          |
|:-----------------|:----------------------------------|:--------------------|:-----------------------------------------------|
| **cuda-oxide**   | `rustc` codegen backend           | NVIDIA PTX/SASS     | CUDA programming model in safe Rust            |
| **Rust-GPU**     | `rustc` → SPIR-V                  | Vulkan/Metal/DX     | Graphics shaders and compute via SPIR-V        |
| **rust-cuda**    | `rustc` → NVVM IR                 | NVIDIA PTX          | Rust language model on NVIDIA GPUs             |
| **CubeCL**       | Embedded DSL + JIT runtime        | CUDA/ROCm/WGPU      | Cross-vendor compute kernels from a Rust DSL   |
| **std::offload** | `rustc` + LLVM offload            | NVIDIA/AMD/Intel    | Implicit offload of CPU code                   |
| **cudarc**       | Safe CUDA driver bindings         | NVIDIA              | Host-side bindings to the CUDA driver          |
| **wgpu**         | WebGPU API + WGSL/Naga            | Cross-platform      | Portable compute via shader languages          |

The "Scope" column captures the **design center** each project optimizes for,
not a feature ceiling. Several of these projects overlap at the edges, and a
single application may pull in more than one of them.

## cuda-oxide and rust-cuda

The closest neighbor — and the one most often confused with cuda-oxide — is
`rust-cuda`. Both projects use a `rustc` codegen backend to target NVIDIA
GPUs, and from a distance they look interchangeable. Up close, as we read
the two projects, the design centers point in different directions:

- **rust-cuda**'s focus is on bringing **Rust to NVIDIA GPUs**: Rust
  ergonomics like `async`/`.await`, parts of the standard library running
  on-device, and a Rust-first programming model that abstracts over CUDA
  concepts.
- **cuda-oxide**'s focus is on bringing **CUDA into Rust**: kernel authoring,
  device intrinsics, the SIMT execution model, and the CUDA programming
  model expressed natively in safe Rust — closer in spirit to writing a
  `__global__` function in C++ than to writing a generic Rust function that
  happens to run on a GPU.

Both directions are valuable, and we believe the two projects are
complementary. We've been working with the rust-cuda maintainers as both
projects mature, and we expect to keep doing so.

## Other neighbors

- **Rust-GPU** targets graphics-oriented GPU programming via SPIR-V, which
  reaches Vulkan, Metal, and DirectX. Compute shaders are supported, but
  the design center is graphics; if you need cross-vendor portability or
  shader interop, Rust-GPU is the right tool.
- **CubeCL** is an embedded DSL: you annotate a Rust function with
  `#[cube]`, and a CubeCL runtime JIT-compiles it to CUDA, ROCm/HIP, or
  WGSL on demand. It is not a `rustc` backend — the `#[cube]` proc-macro
  rewrites the function body into a CubeCL IR that the runtime then
  lowers to whichever backend the host GPU exposes. The design center is
  *cross-vendor portability* (a single kernel runs on NVIDIA, AMD, and
  WGPU), not full Rust language coverage; in exchange CubeCL gives up
  the ability to use arbitrary Rust constructs inside a kernel and works
  with a deliberately restricted DSL surface. CubeCL is the compute
  backend behind the [Burn](https://github.com/tracel-ai/burn) ML
  framework, which is its main showcase application. cuda-oxide and
  CubeCL are largely complementary: CubeCL when you need one kernel to
  run across GPU vendors via a controlled DSL; cuda-oxide when you need
  to write idiomatic safe Rust against the full CUDA programming model
  on NVIDIA hardware.
- **std::offload** is a Rust language feature (currently nightly) that uses
  LLVM's offload runtime to move CPU loops to accelerators implicitly.
  Different programming model: the user writes CPU-style code and the
  compiler/runtime handles the offload. cuda-oxide is explicit; offload is
  implicit.
- **cudarc** provides safe Rust bindings to the CUDA driver API. It's a
  host-side library for launching kernels written elsewhere (typically PTX
  or CUDA C++). cuda-oxide's host-side layer (`cuda-core`, shared with
  cutile-rs) is tightly integrated with `cuda-host`'s launch macros, but
  the PTX cuda-oxide produces is portable — cudarc is a fine option if you
  want to launch cuda-oxide kernels from a project that doesn't depend on
  the rest of the cuda-oxide stack.
- **wgpu** is a Rust implementation of the WebGPU API — a runtime
  abstraction over Vulkan, Metal, and DirectX 12 (and WebGPU in the
  browser). Compute and graphics shaders are written in WGSL (the WebGPU
  shading language) and translated to backend-specific code (SPIR-V, MSL,
  HLSL) by Naga. Different level of the stack — wgpu is a runtime + shader
  DSL, cuda-oxide is a `rustc` codegen backend that compiles Rust itself.

## Engaging with the project

If you're building a Rust + GPU project — or evaluating which of the above
fits your needs — we're happy to compare notes. Join the community on
[Discord](https://discord.gg/ZUEr4AhH5C) for questions, design discussions,
and announcements, or reach the team via
[GitHub Discussions](https://github.com/NVlabs/cuda-oxide/discussions)
or by opening an issue on the repository.
