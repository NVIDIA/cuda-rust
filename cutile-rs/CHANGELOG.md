# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Removed

- Remove `cutile::tileir::load_view_tko_valid_check` and
  `cutile_ir::requirements::Feature::ValidityCheck`: validity checking is
  unsupported by the public CUDA 13.5 tooling. Low-level IR containing the
  `valid_check` attribute now reports an error instead of emitting unsupported
  bytecode. Callers that passed `false` can use `cutile::core::load_view_raw`
  with the final boolean argument removed; there is no supported replacement
  for `true`. The Tile workspace moves to 0.6.0 for this API removal; shared host
  crates remain at 0.5.0. Remove the corresponding documentation claims below.

### Added

- `cuda_core::memory::memcpy_batch_async` enqueues a batch of copies between
  device and host memory with one `cuMemcpyBatchAsync` call, with a
  `MemcpySrcAccessOrder` for the batch.
- Add opt-in B200 profiles to the fused QKV and NVFP4 MoE examples. Add public
  CTK 13.5 setup instructions, explicit large MoE fixture selection, focused
  correctness runs and raw timing samples; retain the original default profiles.
- Add optional cheap screening and budget-reserved paired confirmation to the
  preparation pipeline. Preparation stops before fresh incumbent comparisons;
  incomplete confirmation retains the incumbent and records raw samples and
  failures. The RMSNorm autotune example uses this path. Confirmation defaults
  are configurable repeatability heuristics, not statistical guarantees.
- Ordinary launches can join CPU preparation of the same specialization and
  load its binary on the launching thread. Retained prepared binaries are
  reusable without explicit installation. Waiting callers have independent
  cancellation, and interrupted or abandoned compilation remains retryable.
- Add an opt-in autotuning preparation pipeline under `experimental-tune`,
  separating CPU CUBIN compilation from foreground loading, correctness checks
  and measurement. It defaults to one candidate ahead, supports bounded workers
  and cooperative deadlines, and reports invalidity separately from timeouts
  and infrastructure failures. Results require independent finalist confirmation;
  the existing synchronous tuner and resume-log format are unchanged.
- Refresh `cutile-kernels` from Grout with split-K prefill/merge, persistent
  RMSNorm, and the remaining baseline and single-token variants. NVFP4 MoE
  routing, quantization, FC1/FC2, and finalization kernels now live in the
  reusable corpus and are shared by the example. Peak performance requires
  the CUDA Toolkit 13.5 Tile IR assembler (#1746).
- Add a reusable fused QKV projection, Q/K RMSNorm, FP8 conversion and paged
  KV-scatter kernel, with a correctness example and opt-in CUDA graph benchmark
  sweep. The initial schedule is not B200-tuned (#1746).

### Documentation

- Link the Tile library provider catalog and CUDA Oxide export workflow;
  refresh kernel/example discovery, workspace paths, CUDA 13.5 setup, and
  bytecode-version negotiation instructions (#1746).

### Fixed

- Keep masked QKV tail-row slot pointers in bounds, retaining the original
  M=67 fixture and numerical checks under compiler-instrumented memcheck.
- Update the optional `cuda-tile-rs` bindings and `cuda-tile-translate` build
  from CUDA Tile IR 13.2 to the public 13.5 release, including SIMT preview
  support and translator round-trip tests.

## [0.5.0] - 2026-11-04

CUDA 13.5 tile-library calls and GPU communication, a shared Tile/SIMT host
runtime, and stronger memory-safety contracts.

### Added

- **Tile libraries (public preview, CUDA 13.5):** typed
  `#[cutile::foreign] unsafe fn` declarations, JIT-resolved string arguments,
  implicit output values, and host-configured linking. Memory ordering survives
  branches, loops, and wrappers; cache keys include library contents and declared
  provider dependencies/settings. See the [tile-library guide](https://github.com/NVIDIA/cuda-rust/blob/9dc26c50a8c90026bcd0dfa7dc4b36f189126946/cutile-rs/cutile-book/guide/tile-libraries.md).
- **Unsafe communication:** unicast/multicast pointer classification, multicast
  load-reduce/reduce/store and atomic-view inbounds promises
  in `cutile::tileir`. System-scope signaling and alias fences support explicit
  synchronization. The [communication reference](https://github.com/NVIDIA/cuda-rust/blob/9dc26c50a8c90026bcd0dfa7dc4b36f189126946/cutile-rs/cutile-book/reference/unsafe-comms.md)
  includes a runnable unicast/multicast example.
- Minimum CTA residency controls and safe `condition.likely::<true/false>()`
  methods for scalar booleans and boolean tiles. **Likelihood hints are discarded
  in 13.5 bytecode and have no backend performance effect.**

### Changed / migration from 0.4

- Update Tile and shared host crates together to **0.5.0**. Replace
  `cuda_core::simt` imports with crate-root imports and use the unified
  `cuda_async` API. `Context` owns a context; `Device` identifies a GPU.
  Foreign allocations must supply their actual context. Buffers and tensors
  share storage through `Tensor::from_buffer`, `into_buffer`, and contiguous views.
- Scalar kernel parameters require `DeviceCopy` alongside `DType`.
  `KernelInputStored` / `KernelOutputStored` now require `unsafe impl` with
  documented ABI, extent, lifetime, and access obligations; raw launch packets
  require unsafe finalization. Safe buffer casts require `NoUninit` sources,
  tensor reinterpretation requires `AnyBitPattern` targets, and atomic buffer
  conversions use `DeviceRepr`. Arbitrary bytes cannot be reinterpreted as `bool`.
- `CompileOptions::codegen` replaces the separate debug/optimization fields.
  Replace `device_debug(true)` with `debug_info(DebugInfoLevel::Full)` and
  `lineinfo(true)` with `debug_info(DebugInfoLevel::Line)`. The last setter wins;
  full debugging uses optimization level zero. Runtime overrides are
  `CUTILE_JIT_DEBUG`, `CUTILE_JIT_OPT_LEVEL`, and `CUTILE_JIT_SANITIZE`.
- Compiler integrations use scoped `CUDATileModules::resolve_function` instead of
  removed flat function lookups. `TileirasOptions` gains `tilelibs` and loses
  `Copy`; `Opcode` and `Attribute` gain preview variants. Compiler structures and
  call/cache interfaces also change, including `ModuleItems`, `SiteResolution`,
  `TileKernel::jit_compile`, and `kernel_cache_slot`.

### Fixed

- Generic wrappers, nested trait calls, const-generic tags, inferred type
  arguments, and module-scoped function/constant resolution. Unsupported module
  references error instead of selecting unrelated same-named items.
- Correct 13.5 `if`/`select` bytecode and preservation of peer-pointer attributes.
  Added bytecode-reader, assembler, foreign-call ordering/preservation, and
  two-GPU signaling/buffer-reuse regressions.
- Allocation retention and access exclusion through asynchronous completion,
  graph replay, and early drops; tensor uniqueness; aligned pinned buffers;
  completed zero-initialization; sliced-view alignment; libclang 22 bindings;
  and generated/inlined debug locations.
- Optional CPU-reference dependencies updated to Candle 0.11.1 and patched ZIP,
  with all-feature dependency checks.

### Prerequisites and limits

Baseline support starts at emitted Tile IR 13.2 on supported targets; individual
operations and GPUs impose additional requirements. New 13.5 operations need a
compatible assembler. Tile libraries need matching toolkit-built libraries and
13.5 preview support; their declarations/calls are unsafe. Pointer operands and
shape/tuple constant parameters are not yet supported. `#[inout]` returns an
updated value while preserving the caller's original immutable tile.

Unicast/multicast target `sm_100`, `sm_103`, and `sm_107`: unicast needs NVLink;
multicast additionally needs NVLink SHARP-capable NVSwitch. Callers establish
mappings and synchronization. See the
[compatibility matrix](https://github.com/NVIDIA/cuda-rust/blob/9dc26c50a8c90026bcd0dfa7dc4b36f189126946/cutile-rs/cutile-book/reference/compatibility.md).
Rust 1.89+ is required; the library builds on 1.89, and runtime validation used 1.98.0.

Source lives in [NVIDIA/cuda-rust](https://github.com/NVIDIA/cuda-rust) under
`cutile-rs/`; the [book](https://nvidia.github.io/cuda-rust/cutile/) and release tags
(`cutile-rs/vX.Y.Z`) use the unified repository. Crate names are unchanged.

## [0.4.0] - 2026-09-25

Debug builds support cuda-gdb and Nsight. DGX Spark is now supported.
CUDA 13.4 Tile IR adds programmatic dependent launch. Runtime checks enforce
tensor lifetimes for asynchronous work and graph replay. Both launch-site
paths have lower overhead.

### Breaking changes

- `Global` requires a sealed device atomic type, such as
  `Global<AtomicI32, { [] }>` instead of `Global<i32, { [] }>`. Global
  accesses reject `Weak` ordering and `TileBlock` scope in Rust and the
  JIT; unsafe raw intrinsics are unchanged (#288).
- `Tensor::store` returns its completion `Token`. `Tensor::token` reads it;
  unsafe `Tensor::set_token` installs an external dependency. Installation
  rejects conditional/loop regions, block-local receivers, and subsequent
  shadowing until token updates use control-flow carries. The raw
  `set_tensor_token` and `make_partition_view` helpers are `unsafe`. Safe
  partition helpers preserve the tensor's own token.
- `Latency<const CYCLES: i32>` (was `u32`), so `Latency::<L>` works with an
  `i32` kernel const generic (#275, closes #246).
- `DeviceOp::execute` implementations must register owned resources with
  `ExecutionContext::retain` before enqueueing work. Returning or borrowing
  them in the output is insufficient. Conflicting cross-stream accesses
  return `DeviceError` at enqueue; same-stream work stays stream-ordered.
  Forgotten futures deliberately leak storage and access leases (#275).
- Device debug information follows Cargo's profile. Development builds use
  full device debugging at `O0`; release builds use none.
  `CUDA_RUST_DEBUG=none|line|full` overrides the profile at build time (`line`
  enables optimized profiling). Explicit `CompileOptions` take precedence
  (#285).

### Added

- Raw Tile IR 13.4 operations: `insert`, `fpowi`, `fpowf` (the existing
  `pow` spelling remains), GDC launch/wait tokens, and alias fencing (#298).
- `f8e5m3fnu`, explicit pointer classification, view `inbounds`, saturating
  float-to-int, `NearestAway`, and kernel returns inside `loop`/`while`
  on Tile IR 13.4. Returns in inlined helpers or beneath `for` remain
  restricted (#298).
- Raw controls for allocation, arbitrary-rank gather/scatter and
  strided view loads/stores, view atomic reduction, fast-accumulation MMA,
  exp/tanh rounding, full-width/general-rank assumptions, private/constant
  globals, and module producer metadata. Existing signatures are preserved
  (#298).
- Unsafe per-launch `programmatic_dependent_launch()` on generated builders
  and `AsyncKernelLaunchBuilder`, with runtime driver entry-point lookup
  (Tile IR 13.4, `sm_90+`) (#298).
- Tile IR 13.4 bytecode with assembler version negotiation. Version and
  architecture checks run before JIT assembly. The writer preserves
  13.2/13.3 layouts and rejects newer features when targeting them (#298).
- DGX Spark (GB10, `sm_121`, aarch64) support, verified with CUDA 13.3 and
  13.4 on the 580 driver. A new tutorial runs Qwen3 inference with Grout
  (#295, #303).
- The CUDA driver loader probes the WSL2 shim (`/usr/lib/wsl/lib`) and the
  Debian/Ubuntu multiarch path when `libcuda.so.1` is not on the library
  path (#277, fixes #276).
- Toolkit discovery accepts `/opt/cuda` (#190).
- Kernel-cache eviction APIs `clear_kernel_cache`, `evict_kernel`, and
  `retain_kernels` are available without `experimental-tune`. Serving engines
  can manage cached specializations independently of autotuning. Their
  unsafe quiesce-before-eviction contracts and return values are unchanged
  (#282, closes #268).

### Performance

- Launch-site cache hits no longer allocate in the launcher. Arguments use
  an inline arena, spilling to the heap beyond 32 parameter slots. Shape
  validation compares in place; the site probe borrows fixed-size arrays.
  A four-tensor launch dropped from 15 heap allocations to 2, both in the
  submission envelope (#307).
- Launch-site cache misses (changed scalar divisibility hints or new
  specializations) reuse the assembler fingerprint, resolved toolchain and
  `CUTILE_BYTECODE_VERSION` override for one second between environment and
  filesystem lookups. Mid-process toolchain switches take effect within
  that window (#307).
- Lifetime bookkeeping uses a refcount per argument instead of two heap
  allocations. Graph replay reacquires each distinct buffer once instead
  of visiting every recorded node. A 32-node graph enqueues in 1.67 µs
  (was 14.68 µs initially); a four-tensor launch takes 1.88 µs (#302).

### Changed

- Tile IR compiler, encoder and example/test prerequisites share a capability
  registry with checked documentation tables. Standalone bytecode validation
  uses the JIT's assembler-version negotiation, including the CUDA 13.2
  fallback for assemblers without `--list-versions`. CI now exercises 13.2
  alongside 13.3 and 13.4 (#298, #304).
- Cross-module user helpers and methods retain their source locations and
  inline scopes in device debug information. Core operations keep call-site
  attribution (#285).
- `cutile-examples` no longer depends on candle by default. CPU reference
  helpers and the `flash_attention` example require `reference-cpu`, enabled
  by `scripts/run_examples.sh`. The default workspace builds on generic
  aarch64, where candle's `gemm-f16` needs `fullfp16`. This failure was first
  seen on DGX Spark (#295).
- The workspace pins a stable Rust toolchain (`rust-toolchain.toml`); the
  reactor verification job alone uses nightly (#304).

### Fixed

- cuda-async prevents use after free from safe code by retaining all tensor
  storage used by a submission until completion, including after
  borrowed-argument recovery, discarded outputs, errors, panics, and
  forgotten futures. Graph replay records captured storage and reacquires
  its accesses on every launch. Device-to-host copies synchronize before
  exposing initialized `Vec` elements (#275, fixes #272 and #252).
- f8 constants round to nearest-even, including subnormal boundaries and
  E4M3FN finite saturation, instead of truncating the mantissa (#275,
  fixes #250).
- Read-only `Partition` loads no longer inherit each other's completion
  tokens. The `set_token` changes in 0.4.0 carried load completion tokens out
  of inlined `Partition::load` calls for immutable bindings. This serialized
  read-only partition loads and slowed load-bound kernels by up to 35% (a fused
  norm+RoPE kernel went from 20 to 27 µs). Only explicit `set_token` /
  `set_tensor_token` installs cross an inlining or block boundary for
  immutable bindings; mutable bindings are unchanged (#310).
- Free-function calls to trait-dispatch wrappers (`load_tile_like(x, out)`)
  retain their call-site spans. Previously, the inlined body pointed to the
  module's first line, so `break <call line>` found no code and stepping
  skipped the call (#311).
- Submission-lifetime GPU tests run in a separate single-threaded binary.
  In the aggregate binary, another test's module load could wait on a gated
  stream and block the gate owner's host callback (#295).

## [0.3.1] - 2026-09-02

A single `cargo add cutile` now suffices, kernels gain Triton-parity
program indexing and Rust-model raw pointers, and the experimental
autotuner grows the pieces its first engine-scale consumer asked for.

### Performance

- Cache-hit kernel launches are ~2.4x faster on the host (3.9 us -> 1.6 us
  per launch on a Ryzen 9 9950X, RTX 5090): generated launchers keep a
  per-site cache of the last resolved specialization, launch validation
  builds its messages only on failure, hoisting shape vectors are built
  only when a kernel has hoisted checks, and kernel arguments marshal
  through an inline slot arena instead of one heap allocation each
  (~108 -> single-digit allocations per launch). Evicting from the kernel
  cache invalidates every launch site, preserving the quiesce-then-clear
  contract.

### Breaking changes

0.3.1 ships the incompatible changes below. Each one closes a soundness or
correctness hole found by the September audit, the case the 0.1.0 notes
reserved for breaking changes. Projects that are not ready for them can stay
on 0.3.0.

- `Device::load_module_from_bytes` and
  `cuda_async::device_context::load_module_from_bytes` are `unsafe fn`: the
  driver reads header-declared offsets from the image, so a truncated cubin
  reads past the slice.
- The DSL ops `atomic_rmw_tko` and `atomic_cas_tko` are `unsafe fn`, like
  `load_ptr_tko` and `store_ptr_tko`.
- `DType` is an `unsafe trait`: implementors promise that every byte pattern
  the device can produce for their `DTYPE` is a valid value.
- `cuda_core::api::{malloc_async, malloc_from_pool_async, free_async,
  memcpy_htod_async, memcpy_dtoh_async, memcpy_dtod_async, api_version}` and
  `ExecutionContext::alloc_async` return `Result` instead of panicking, so a
  failed device allocation can be handled or retried.
- `Stream::launch_host_function` and `launch_host_function_with_sync_mode`
  require `F: 'static`.
- `api::memcpy` returns `Memcpy<'a>`, which borrows both tensors for the
  lifetime of the operation.
- `CudaGraph::update` accepts only `GraphNode + DeviceOp<Output = ()>` and
  returns `Result<(), DeviceError>`.
- Generated kernel launchers return a nameable `Launcher<…, KernelArgs<…>>`
  instead of `impl DeviceOp` and implement `GraphNode` only when every input
  does; `AsyncKernelLaunchBuilder::args` is no longer public.
- `TrialLog::open` takes a `LogProvenance` (`experimental-tune`).
- `BytecodeVersion::V13_1` is removed; the bytecode floor is CUDA 13.2.
- Kernel programs: a `return` below the function body and a `break` inside
  `for` are compile errors (they were silently miscompiled); integer `/`
  rounds toward zero, as in Rust, instead of toward negative infinity; `u8`
  and `u16` arithmetic and comparisons are unsigned. Programs that relied on
  the old lowering produce different results.

### Added

- `shape![..]`: the shape constructor macro, replacing `const_shape!` (now
  deprecated; both spellings work on both compile tracks). Runtime extents
  keep the explicit form `Shape::<{ [-1] }> { dims: &[n] }`.
- `x.load_like(z)`: method-form spelling of `load_tile_like(x, z)`, lowering
  to identical Tile IR with identical bounds-check placement. Both spellings
  remain supported; docs and examples now use the method form (#259).
- cuda-oxide host surface: `simt` modules in `cuda-core` and `cuda-async`
  (contexts, streams, events, device and pinned host buffers, module loading,
  kernel launch, VMM, reclaim) and a `cuda-core-derive` crate providing
  `#[derive(DeviceCopy)]` (#233).
- CUDA VMM wrappers (`PhysicalAllocation`, `VirtualReservation`, `Mapping`)
  and NVLink switch multicast (`MulticastObject`) in `cuda-core` (#202).
- Autotuner: trial logging and resume on the `Objective` path, `require`d
  configurations measured before the search, and kernel cache management
  (`clear_kernel_cache`, `evict_kernel`, `retain_kernels`), all behind
  `experimental-tune` (#239).
- Const raw pointers (`*const T`) in kernel signatures, with `Tensor::as_ptr`,
  `cast_const`/`cast_mut`, `cast_tile_const`/`cast_tile_mut`, and
  `get_tensor_base`; `program_id(axis)` and `num_programs(axis)` as aliases
  of the tile-block id and count (#240, restored by #256).
- `CompileOptions::{device_debug, lineinfo, sanitize_memcheck, opt_level}`
  pass `--device-debug`, `--lineinfo`, `--sanitize=memcheck`, and
  `--opt-level` to `tileiras` and join both cache keys (#236).
- Source locations in the bytecode debug section, with per-callee subprograms
  and inlined call-site chains, so `--lineinfo` and `--device-debug` builds
  attribute SASS to the kernel's `.rs` lines (#238).

### Changed

- The host-side crates (`cuda-bindings`, `cuda-core`, `cuda-async`) require
  CUDA 13.0 or newer; the Tile compiler floor stays at 13.2 (#233).
- Toolkit discovery honors `CUDA_HOME` after `CUDA_TOOLKIT_PATH`, probes
  `targets/<dir>/include` layouts, and accepts `CUDA_TOOLKIT_TARGET_DIR` to
  name one target tree; the host-crate tests run in the CPU and GPU test
  scripts (#251).
- `load_ptr_tko` and `store_ptr_tko` are `unsafe fn` (#240, restored by
  #256).

### Fixed

- Integer division and remainder serialize `rounding<zero>` and inferred
  signedness on `divi`/`remi` (#225, restored by #256).
- Persisted tuning records carry the L2 cache key schema, so key migrations
  are distinguishable from workspace drift (#241).
- Pointer alignment hints use the full 64-bit device address; an address
  whose low 32 bits were `0x8000_0000` previously lost its divisibility hint
  (#244).
- Generated launchers reach `cutile_compiler` through the `cutile` crate
  root, so consumer crates without a direct `cutile-compiler` dependency
  compile (#248).
- Three merges reverted by stale squash merges (#240, #225, #185) are
  re-applied (#256).

## [0.3.0] - 2026-08-20

This release introduces compiler optimizations that improve safe kernel
performance, async launch performance improvements, persistent kernel
tuning, and zero-copy interop with externally owned device memory. The
compiler now proves each partition access at compile time or verifies it
once at launch, and places the rare remaining check where it costs
nothing.

### Highlights

- Bounds-check placement: every partition access walks one proof ladder
  (axis provenance, static folding, declared preconditions, host-side launch
  checks) and only then pays an in-kernel assert, hoisted to the outermost
  provable loop preheader. Kernels indexed by loop iterands, mapped schedule
  components, or tile-block ids typically compile with zero in-kernel checks
  and no annotations. `PartitionMut::store` joins the checked-by-default
  surface (`store_unchecked` is the explicit escape) at identical register
  counts.
- `deny_in_kernel_checks = true` on an entry turns any check that would remain
  in the kernel into a compile error, so an assert-free kernel is a contract
  the build enforces rather than a property to audit.
- Partial-coverage launches: `partition_prefix` lets the launch grid cover a
  per-axis prefix of an output partition's block grid. Exceeding the grid on
  any axis remains a launch error, and the default `partition` keeps strict
  equality.
- Zero-copy interop: `Tensor::from_foreign` borrows device memory owned by an
  external framework (cudarc, torch, VMM ranges), holding the owner alive by
  refcount and verifying the addressable extent at construction;
  `Tensor::borrow_raw_parts`, `Device::borrow_with_owner`, and
  `Stream::borrow_with_owner` cover bare handles.
- Autotuner core: declared configuration spaces, pluggable searchers,
  resumable grid search, and provenance-checked tuning records.
- `cutile::bench` for device-event kernel benchmarking, an opt-in persistent
  on-disk cubin cache, a process-global single-flight kernel cache with a
  `.compile()` warmup API, and async launch latency at parity with sync.
- A new book chapter, Bounds-Check Placement, documents the placement rules
  and a verification workflow: per-kernel placement counts from
  `CUTILE_JIT_TIMING`, and diffing the emitted Tile IR against an unsafe twin
  before running anything.

### Performance results

- The fully checked flash-attention prefill kernel matches its unsafe twin
  on RTX 5090, within 2.5% of the checks-disabled floor.
- Async launch latency is at parity with sync, and a differential harness
  verifies that every check placement refines the outcome of checking at the
  access site.
- Grout, the Qwen3 inference engine from the 0.2.0 paper, now runs its
  25-kernel default engine path with zero unchecked kernels and six one-line
  unsafe view constructions in total, at safe-vs-unsafe parity on RTX 5090
  and B200.

### Changed

- `with_bounds` and `Dim::new` are deprecated: bounds inference and declared
  preconditions prove everything they proved, with the checks placed at
  launch instead of in the kernel. Migration notes are on the deprecations.
- Tile IR bytecode version selection reads the CUDA toolkit when
  discoverable and otherwise probes `tileiras` with a representative module
  (an entry with a `for` region and view chain), so an accepted version is
  one real kernels compile at.
- The `persistent_gemm` example is rewritten in the zero-annotation form and
  compiles under `deny_in_kernel_checks = true`.

### Fixed

- An entry declared as bare `#[cutile::entry]`, without an argument list, was
  silently treated as a plain function; both attribute forms are now
  recognized.
- Interval range facts no longer survive arithmetic that can wrap `i32`; a
  wrapped intermediate could previously discharge a check the machine value
  violated.
- Bounds analysis guards division and remainder against zero divisors,
  saturates interval arithmetic, and uses overlap analysis for `eq`/`ne`.
- Device-op `Send` bounds, CUDA driver flag types, and aarch64 flake support.
- Security bumps for crossbeam-epoch (RUSTSEC-2026-0204) and anyhow
  (RUSTSEC-2026-0190).

## [0.2.0] - 2026-06-16

This release collects the changes since `0.1.0` and focuses on low-precision
inference support while also publishing the reproducibility artifacts for the
cuTile Rust paper, *Fearless Concurrency on the GPU*.

### Highlights

- Added CUDA 13.3-oriented low-precision inference support, including NVFP4
  pack/unpack support, block-scaled matrix multiply support, and runnable NVFP4
  and MXFP8 linear-tile examples.
- Added `cutile-kernels`, a reusable kernel crate organized by function for
  inference workloads. It includes attention, normalization, positional
  encoding, KV-cache update, embedding, argmax, and pointwise kernels, with
  experimental low-level and benchmark-oriented kernels for fused transformer
  paths, KVBM layout conversion, and grouped GEMM/MoE work.
- Added compile-only coverage and smoke tests for reusable kernels, and moved
  test-only examples into the test suite so `cutile-examples` stays focused on
  user-facing examples.
- Added paper reproducibility artifacts under `cutile-benchmarks/paper/`,
  including benchmark harnesses, committed result files, machine notes, and
  plotting scripts.
- Updated the root README with the paper link, citation information, paper
  artifacts, related projects, and the current cuTile Rust execution/lowering
  model.

### Paper Results

- On NVIDIA B200, cuTile Rust reaches 7 TB/s for element-wise operations and
  2 PFlop/s for GEMM, about 91% of peak memory bandwidth and 92% of dense `f16`
  peak, respectively.
- Safe Rust persistent GEMM reaches 2.07 PFlop/s at `M=N=K=8192`, within 0.3%
  of the corresponding low-level Tile IR variant, showing safety without
  measurable runtime overhead.
- Grout, a Qwen3 inference engine built with cuTile Rust in collaboration with
  Hugging Face, reaches 171 tokens/s for Qwen3-4B on NVIDIA GeForce RTX 5090
  and 82 tokens/s for Qwen3-32B on B200 in batch-1 decode, showing competitive
  state-of-the-art performance on memory-bound inference tasks as measured by
  the HBM roofline analysis.

### Changed

- Split CPU and GPU test entry points so compile-only tests do not require a
  GPU, while GPU tests actually execute GPU work.
- Updated compile-only testing to use `KernelCompiler` and default to `sm_120`
  where a local GPU architecture is not required.
- Reorganized and refreshed the book and examples around the current
  host/device API, CUDA graphs, JIT compilation, performance guidance, and
  low-precision inference, with support for versioned book builds.

## [0.1.1] - 2026-06-01

This patch release added the first CUDA 13.3 low-precision Tile IR support and
refreshed the book publishing flow.

### Added

- Added NVFP4 support in CUDA dtype handling, Tile IR formatting, bytecode
  encoding/decoding, compiler intrinsic lowering, and the public device DSL.
- Added CUDA 13.3 bytecode, per-op round-trip, and tensor/matrix operation
  tests covering the new low-precision Tile IR surface.
- Added runnable NVFP4 and MXFP8 examples, plus a new book tutorial for NVFP4
  inference.
- Added scripts and documentation for building and publishing versioned book
  releases.

### Changed

- Updated compiler lowering, specialization handling, and Tile IR type support
  for CUDA 13.3 low-precision operations.
- Updated examples, book references, and architecture notes to describe the
  current lowering path and low-precision inference APIs.
- Bumped the workspace package versions to `0.1.1`.

## [0.1.0] - 2026-05-16

This is the first cuTile Rust beta release with stable host-side and
device-side APIs. We do not plan further breaking changes to the kernel
authoring model, tensor launch API, `DeviceOp` execution model, or core device
operation surface; future work is expected to extend these APIs compatibly
unless a correctness issue requires otherwise.

### Highlights

- Stabilized the public host API around lazy `DeviceOp`s, borrowed/shared
  tensors, mutable partitions, async execution, CUDA graph capture, and CUDA
  interop.
- Stabilized the public device DSL around Tile IR-aligned operations,
  rank-polymorphic helpers such as `load_tile_like`, tensor views, partition
  views, memory ordering, atomics, tokens, shape operations, and tile math.
- Added type-check-driven JIT lowering with stable node IDs, richer expression
  type inference, static dispatch lowering, type aliases, global constants,
  `Global`, `else if`, and source-location preserving diagnostics.
- Added mapped partition support for safe persistent scheduling, including
  proof-carrying partition indexes and examples for persistent GEMM-style
  output traversal.
- Improved dynamic-shape performance by propagating `num_tiles` bounds, fixing
  nested partition overhangs, supporting static-shaped `load_tile_like`, and
  using zero-padded read-only tile-like loads where they generate better code.
- Added CUDA runtime ergonomics: dynamically loaded CUDA bindings, configurable
  `tileiras` binary override, custom memory pools, memory accounting, and JIT
  timing support.
- Updated the book, README, and examples to describe the stable host/device
  APIs and current interop story.

### Fixed

- Restored scalar divisibility hint lowering for kernel arguments.
- Fixed compile-only kernel compiler hooks and several JIT/type inference
  failures exposed by examples and downstream kernels.
- Fixed CUDA 13.0-13.2 `CUmemLocation` layout compatibility.
- Fixed custom memory pool resolution outside the default device policy
  closure.

## [0.0.2] - 2026-04-26

This release is a broad API and compiler update focused on making kernel
launching composable, removing the JIT's dependency on external MLIR tooling,
and aligning the Rust DSL with the Tile IR operation model.

### Added

- `DeviceOp` combinators, shared/boxed operations, heterogeneous operation
  collections, and a unified launcher API for kernels.
- CUDA graph capture APIs, including scoped graph capture and graph launches
  that compose as `DeviceOp`s.
- Safe tensor views and slicing, plus host helpers such as `linspace`, `eye`,
  and generic random tensor creation.
- `cutile-ir`, a pure Rust Tile IR representation, formatter, bytecode writer,
  decoder, validation tests, and round-trip coverage.
- JIT compiler infrastructure for name resolution, stable node IDs, typed
  dispatch lowering, type inference groundwork, specialization hints, and
  linker-based module discovery.
- Type-safe Tile IR op modifiers for rounding, overflow, memory ordering,
  scope, padding, TMA, FTZ, NaN propagation, comparison predicates, and related
  static attributes.
- `cuda-tile-rs` as an opt-in wrapper around the bundled cuda-tile C++ library
  and `cuda-tile-translate`.
- New examples, benchmarks, and book/reference material for DeviceOps, CUDA
  graphs, interop, tensor slicing, and the updated DSL.

### Changed

- Renamed `DeviceOperation` to `DeviceOp` and simplified scheduling around a
  smaller `SchedulingPolicy` API.
- Renamed CUDA wrapper types from `CudaContext`/`CudaStream`/`CudaModule`/
  `CudaFunction` to `Device`/`Stream`/`Module`/`Function`, with borrowed raw
  handle constructors for interop.
- Consolidated tensor copy, reshape, view, random, and creation APIs around
  dynamic shapes and clearer ownership/borrowing behavior.
- Updated kernel parameter handling so tensors, borrowed tensors, mutable
  outputs, partitions, scalars, and `DeviceOp` inputs can be mixed more
  naturally.
- Reworked rank-polymorphic macro expansion through shadow dispatch and rank
  instantiation instead of the old variadic registry machinery.
- Aligned `_core.rs` with the Tile IR operation groups and expanded named DSL
  coverage for numeric, conversion, comparison, memory, atomic, view, token,
  shape, matrix, and misc operations.
- Collapsed `load_tile_like_*` helpers into a single `load_tile_like`, and
  reduced partition view construction to `make_partition_view` and
  `make_partition_view_mut`.

### Fixed

- Corrected `arange` behavior across multiple tile blocks.
- Propagated stream synchronization errors instead of panicking.
- Fixed concurrent CUDA graph capture failures caused by unnecessary context
  synchronization.
- Fixed bytecode defaults and silent-drop cases in the JIT/compiler path.
- Restored nested marker type path resolution for static op modifiers.
- Improved compiler, macro, and DSL error messages and source locations.

### Removed

- The external LLVM/MLIR dependency from the default JIT compiler path.
- The generated `_op()` launcher variant; the unified launcher is now the
  public entry point.
- Legacy `DeviceOperation*` aliases, old copy/reshape/view helper traits, and
  unused cudarc event-tracking infrastructure.

## [0.0.1] - 2026-04-07

Initial tagged release. Pre-DeviceOp redesign baseline.

### Features
- Tile-based GPU programming model with `#[cutile::entry()]` kernels.
- `DeviceOperation` trait with `.apply()`, `.and_then()`, `zip!`, `.unzip()`.
- JIT compilation pipeline: Rust AST → MLIR → CUDA PTX.
- Async execution via tokio with `DeviceFuture`.
- `Arc<Tensor<T>>` for shared inputs, `Partition<Tensor<T>>` for mutable outputs.
- Flash attention, GEMM, RMSNorm, softmax examples.
- cuTile Rust Book with tutorials 1-9.
