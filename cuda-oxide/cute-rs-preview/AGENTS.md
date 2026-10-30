# cute-rs agent guide

cute-rs exists to show that Rust kernels written as CuTe operations compile
through the CUTLASS compiler into CuTe kernels. Everything in this guide
follows from that: the kernels must be CuTe, the names must be CuTeDSL's, and
the target is parity with CuTeDSL rather than a second design.

## Kernel architecture

Kernels intended to demonstrate or exercise cute-rs (including the Qwen
acceptance workloads and every `*_cute` example) must express tensor
computation through cute-rs tensor views, layouts, partitions, register tiles,
copies, reductions and MMA operations. This is a hard requirement, alongside
numerical correctness and competitive measured performance.

- A `*_cute` kernel imports `cute_rs` and, for instruction-level steps,
  `ctm_rs`. It does not import `cuda_device`. Kernel entry attributes, thread
  identifiers and shared-storage declarations come through the `ctm_rs`
  re-exports. If a kernel needs something neither crate offers, add it to
  cute-rs (or to ctm-rs when it is one instruction) as a typed operation; do
  not reach past them.
- The deliverable is the exported MLIR, not a kernel that happens to run. The
  tensor computation must reach the CUTLASS compiler as `cute.*` operations;
  check with `CUDA_OXIDE_MLIR_OUTPUT=<file>` that the loads, copies and MMA
  steps appear as CuTe operations rather than scalar code. A kernel rewritten
  around a compiler gap or a missing operation is a workaround, and the
  repository rule applies: fix the compiler or add the operation.
- Keep instruction-level CTM operations underneath reusable cute-rs
  operations. CTM kernel declarations, launch contracts, thread identifiers and
  ABI storage types may remain at kernel entry points. Explicit `*_ctm`
  examples continue to demonstrate the lower-level instruction layer.
- Do not hand-code hardware fragment packing, lane-to-value maps, or
  shared-memory swizzle formulas in cute-rs workload kernels. Describe those
  mappings using typed layouts and derive accesses from them.
- When an abstraction is missing, extend cute-rs with a reusable, typed
  operation and test its invariants. Moving a workload-specific implementation
  behind a new name, or wrapping a raw instruction while leaving its layout
  logic in the kernel, does not satisfy this requirement.
- Keep operand roles, element formats, dimensions, storage bounds and mutable
  ownership explicit. State collective-participation and aliasing obligations
  at unsafe boundaries; do not imply the type system proves convergence.
- Preserve arithmetic order and observable rounding unless the task explicitly
  authorizes changing them. Test layout coverage and tails independently of
  device code, then validate GPU numerics, sanitizers and generated
  instructions.
- Benchmark representative end-to-end workloads against the preserved
  baseline. Better abstractions must not silently sacrifice performance.
  Record remaining performance gaps honestly; abstraction migration alone is
  not a speedup claim.
- Existing legacy violations are migration work, not a precedent for new code.
  Report any remaining violations explicitly when delivering a migration.

## Encoding CuTe in Rust's type system (mandatory pattern)

CUTLASS CuTe’s layout algebra, partitions, TMA transfers, pipelines, and
tensor-core MMA operations **must be encoded into Rust’s type system**. No agent
may write a regular `cuda-oxide` SIMT kernel (raw pointer offsets, manual bit
shifts, untyped `u64` descriptors) and pass it off as a `cute-rs` kernel.

Every `cute-rs` GEMM / MoE / attention kernel must encode its compile-time plan
through types so that mismatched layouts, swapped `OperandA` / `OperandB` tiles,
swapped `Mkl` / `Nkl` scale tiles, or mismatched TMA transaction byte counts fail
at compile time:

```text
MANDATORY TYPE-ENCODED `cute-rs` ARCHITECTURE
┌──────────────────────────────────────────────────────────────────────────────┐
│  1. Compile-Time Layouts & Typed TMA Descriptors (`#[grid_constant]`)        │
│     type WeightTileSmem = RowMajor<128, 256>;   // 32,768 B (K=512 FP4)      │
│     type ActTileSmem    = RowMajor<64, 256>;    // 16,384 B (K=512 FP4)      │
│     type ScaleTmaSmem   = RowMajor<8, 256>;     //  4,096 B (8 scale atoms)  │
│     type EpilogueSmem   = RowMajor<64, 128>;    // 64x128 FP32 epilogue      │
│     #[grid_constant] w_tma:   &TensorMap<u8,  WeightTileSmem>                │
│     #[grid_constant] sfa_tma: &TensorMap<u16, ScaleTmaSmem>                  │
├──────────────────────────────────────────────────────────────────────────────┤
│  2. Warp-Specialized Producer/Consumer Pipeline (`PipelineTmaAsync`)         │
│     const TMA_BURST_TX_BYTES: u32 =                                          │
│         size_in_bytes::<u8, WeightTileSmem>()                                │
│         + size_in_bytes::<u16, ScaleTmaSmem>(); // Verified by dialect-cute! │
│     type MainloopPipeline = PipelineTmaAsync<1, 16, TMA_BURST_TX_BYTES>;     │
│     • Producer Warp (`PipelineState<Producer, 1>`):                          │
│         acquire ──► expect_tx ──► copy_tma_2d(w_tma, sfa_tma) ──► tail       │
│     • Compute Warps (`PipelineState<Consumer, 1>`):                          │
│         consumer_wait ──► TiledMma ──► consumer_release ──► advance          │
├──────────────────────────────────────────────────────────────────────────────┤
│  3. Role-Checked Shared Tiles, Block-Scaled MMA & Staged TMEM Epilogue       │
│     type SharedWeightTile   = Sm100SharedTile<WeightTileSmem, OperandA>;     │
│     type SharedActTile      = Sm100SharedTile<ActTileSmem, OperandB>;        │
│     type SharedWeightScales = Sm100ScaleTile<Mkl>;                           │
│     type SharedActScales    = Sm100ScaleTile<Nkl>;                           │
│     type TiledMma = Sm100BlockScaledTiledMma<WeightTileSmem, ActTileSmem>;   │
│     type StagedEpilogue = Sm100TmemStagedEpilogue<EpilogueSmem, 128>;        │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Names follow CuTeDSL

A CuTe concept carries the name CuTeDSL 4.7 gives it, so that a CuTeDSL author
reads a cute-rs kernel without a glossary.

- Look the name up before writing it: the CuTeDSL sources
  (`python/CuTeDSL/cutlass/cute`, `cutlass/pipeline`, `cutlass/utils` and
  `cutlass/experimental/primitives` in the CUTLASS repository) and the
  [comparison page](../cuda-oxide-book/cute/cutedsl-comparison.md). Functions
  stay snake_case, types CamelCase; where rustc's `non_snake_case` lint forbids
  a CuTeDSL method name such as `partition_A`, the Rust form is `partition_a`.
- Rust mechanics keep Rust idioms: constructors are `new` or `from_*`, `len`
  pairs with `is_empty`, `fold` takes an initial value, conversions are
  `from_*` and `to_*`, primitive types are `i32`, `f32` and `f16`. A
  compile-time plan is a type or a const generic (`RowMajor<128, 64>`,
  `Swizzle<3, 3, 3>`, `zipped_divide::<8>()`), because the compiler reads it
  from the generic arguments; that is the one place the `make_*` vocabulary
  does not carry over.
- ctm-rs uses PTX instruction names (`mbarrier_arrive_expect_tx`,
  `barrier_cluster_arrive`, `cp_async_bulk_tensor_shared_cta_global`). The
  cute.arch concept names (`thread_idx`, `warp_idx`, `elect_one_sync`,
  `cluster_arrive`, `mbarrier_init_fence`) live in `cute_rs::arch` as
  forwarders, mirroring cute.arch over primitives in CuTeDSL.
- One name per concept across cute-rs, ctm-rs and the compiler crates. No
  abbreviations the DSL does not use (`Fragment`, not `Frag`; `Mxf4`, the
  form of the PTX kind and the DSL op), no second name for something that
  already has one, and no CUDA C++ or PTX name where cute.arch has a concept
  name.
- The compiler recognises cute-rs definitions by path (the importer's
  registry and static-configuration constants), some author-facing methods by
  name (the importer's classify arms) and types by item name (the codegen
  collector). An alias or a forwarding function is free; a rename moves those
  constants and their tests in the same change. Callable intrinsics are
  forwarded as functions, never re-exported with `pub use`.

## Parity with CuTeDSL, not reinvention

The fundamentals have landed. The work now is bringing cute-rs to feature
parity with CuTeDSL, and every addition is measured against it.

- Before adding an operation, type, attribute, pipeline, scheduler or helper,
  find the CuTeDSL counterpart: what it is called, what it accepts, what it
  lowers to, and which CuTeDSL example uses it. Implement that concept under
  that name.
- The implementation may differ where Rust does better: compile-time plans as
  types, typestate for collective and asynchronous operations, sealed roles
  for operands and CTA groups, `Result` instead of exceptions. The surface
  vocabulary and the semantics still follow CuTeDSL, and a deliberate
  deviation is written down in the [comparison page](../cuda-oxide-book/cute/cutedsl-comparison.md).
- The [support matrix](../cuda-oxide-book/cute/support-matrix.md) is the
  parity ledger. When a gap closes, update its row and the figure it is drawn
  from; when a new operation is only partly covered, say which forms.
- Do not design a cute-rs-only feature where CuTeDSL has one that solves the
  same problem; port theirs. Where cute-rs genuinely adds something CuTeDSL
  lacks (ahead-of-time compilation, compile-time layout and protocol checks),
  document it as an addition, not as the CuTe surface.

## Example and model documentation

- **Keep documentation simple, short, jargon-free, and visual.** Avoid dense
  walls of text. Use visual ASCII diagrams to show the kernel pipeline, memory
  hierarchy (`Global ──► Shared ──► TMEM / Registers`), warp roles, and what is
  captured inside a CUDA Graph:

  ```text
  THE 4 KERNELS OF THE MOE LAYER:
  [K1: Quantize BF16->FP4] ──► [K2: Route] ──► [K3: FC1 + ReLU²] ──► [K4: FC2 + Finalize]

  CUDA GRAPH CAPTURE & 1,000× REPLAY TIMER
  ────────────────────────────────────────
     BeginCapture                                                      EndCapture
     ┌─────────────────────────────────────────────────────────────────────────┐
     │  [K1: Quantize] ──► [K2: Route] ──► [K3: FC1+ReLU²] ──► [K4: FC2+Final] │
     └─────────────────────────────────────────────────────────────────────────┘
     Start Event ──► Replay Graph #1 .. #1000 ──► Stop Event (mean = total / 1000)
  ```

- Keep each entry README approachable, following `examples/elementwise_cute` and
  `examples/blockscale_gemm_cute`: explain the computation, show the CuTe data
  flow visually, give a runnable command, and describe the expected result.
- **Document every single public AND private item with at least 2–3 lines of
  documentation.** This includes every private helper, `#[kernel]`, `struct`,
  struct field, `enum`, enum variant, `type` alias, `const`, `static` shared
  array, and `mod`. Explain shapes, units, ownership, readiness, rounding and
  error behavior in plain, jargon-free language. Never suppress `missing_docs`
  or `clippy::missing_docs_in_private_items` with `#[allow(...)]`. Unsafe APIs
  must state caller obligations; comments must not claim that types prove thread
  convergence or asynchronous completion.
- Put numerical contracts, comparison harnesses and reproduction commands in
  the workload's `reference/` folder, linked from its README. Do not commit
  measurement records: result JSON, traces, profiles, logs and SASS dumps
  belong in a run's output directory outside the tree. Keep essential
  prerequisites and known user-visible limitations in the entry README.
