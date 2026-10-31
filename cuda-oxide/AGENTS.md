# cuda-oxide agent guide

`cuda-oxide` compiles Rust GPU kernels to PTX, SASS/cubins, and CUTLASS MLIR.
Use this guide alongside the root [`AGENTS.md`](../AGENTS.md) and, when working
under `cute-rs-preview/`, [`cute-rs-preview/AGENTS.md`](cute-rs-preview/AGENTS.md).

---

## 1. Two kernel models: Native SIMT vs. `cute-rs`

`cuda-oxide` supports two distinct ways of writing GPU kernels. Never mix them
up or pass one off as the other:

```text
┌─────────────────────────────────────┬──────────────────────────────────────────────┐
│  1. Native SIMT (`cuda_device`)     │  2. CuTe in Rust (`cute-rs` + `ctm-rs`)      │
├─────────────────────────────────────┼──────────────────────────────────────────────┤
│  • Explicit per-thread / per-warp   │  • CUTLASS CuTe's layout algebra, partitions,│
│    control (`thread`, `warp`,       │    TMA transfers, pipelines, and Tensor Core │
│    `SharedArray`, `tcgen05_*`).     │    MMA ops encoded in Rust's type system.    │
│  • Compiles via `dialect-mir` ──►   │  • Compiles via `dialect-cute` ──►           │
│    LLVM / libNVVM ──► PTX.          │    `cutlass-mlir` (CUTLASS 4.7 compiler).    │
│  • Lives in `models/*` & `examples` │  • Lives in `cute-rs-preview/`               │
└─────────────────────────────────────┴──────────────────────────────────────────────┘
```

* **Hard rule for `cute-rs` kernels (`cute-rs-preview/`):** CuTe’s layout
  algebra (`RowMajor`, `ComposedLayout`, `Swizzle`, `Sm1xxBlockScaleKMajor`),
  partitions (`StridedTensor`, `RegisterTile`), TMA descriptors and copies
  (`TensorMap<T, Layout>`, `copy_tma_2d`), producer/consumer pipelines
  (`PipelineTmaAsync`, `PipelineTmaUmma`, `PipelineUmmaAsync`,
  `PipelineTmaStore`), shared-memory operand tiles (`Sm100SharedTile<Layout, Role>`,
  `Sm100ScaleTile<Role>`), and Tensor Core MMAs (`Sm100TiledMma`,
  `Sm100BlockScaledTiledMma`, `Mxf4TiledMma`) **must be encoded in Rust's type
  system**, and **all `cute-rs` names must match CuTeDSL 4.7** (`NamedBarrier`,
  `PipelineTmaAsync`, `PipelineTmaUmma`, `PipelineUmmaAsync`, `PipelineTmaStore`,
  `Producer`, `Consumer`, `PipelineState`, `prefetch_descriptor`, etc.). Do not
  write a regular SIMT kernel with raw pointer arithmetic or custom names and
  call it a `cute-rs` kernel. See [`cute-rs-preview/AGENTS.md`](cute-rs-preview/AGENTS.md).

---

## 2. Compiler optimization rules (what we accept vs. reject)

### Architectural boundary (hard rule)

**Completely avoid putting analysis or transforms in `rustc-codegen-cuda` (`rustc_private`) — use `dialect-mir` + `mir-transforms` instead.**

We intentionally keep `rustc-codegen-cuda` as thin as possible because it is the
only crate tied to unstable internal `rustc` APIs (everything needed at that
boundary comes through `rustc_public` / `stable_mir`). All analysis and
transformations happen further down the pipeline on `dialect-mir`, which is
completely `rustc`-free and built on `pliron`'s `PassManager` and
`AnalysisManager`:

```text
rustc-codegen-cuda  ──►  mir-importer  ──►  cuda-oxide-codegen + mir-transforms
(Thin rustc_public       (To Pliron)        (rustc-free, Pliron PassManager)
 bridge only)                                 ├─► mem2reg (already in SSA!)
                                              ├─► AnalysisManager (cached CFG/loops)
                                              ├─► crates/mir-transforms (passes)
                                              └─► mir_pass_registry.rs (opt-in passes)
```

When adding or modifying compiler passes (`mir-importer`, `mir-transforms`,
`mir-lower`, `llvm-export`, `cuda-oxide-mlir-export`) or intrinsics:

1. **General-purpose, proof-backed transforms only — no benchmark-specific hacks.**
   * Every transform in `mir-transforms` or `mir-lower` (such as loop unrolling,
     stage-index folding, `forward_compiler_result_bundles`, and
     `scalarize_borrowed_aggregate_reads`) must state and verify its exact IR
     preconditions over the complete use/CFG graph.
   * **Fail closed:** if any precondition is not proved (unknown pointer
     provenance, potential aliasing, non-dominating store, runtime base offset
     whose alignment/divisibility is unknown), leave the IR untouched rather
     than guessing.
   * Never pattern-match a specific benchmark's variable names, magic tile
     constants, or kernel signatures inside the compiler.

2. **Fix the compiler or admit the intrinsic properly — never paper over gaps.**
   * Never hand-edit `@generated` files or use raw `ptx_asm!` in first-party
     crates to work around a missing intrinsic or lowering bug.
   * Admit new PTX instructions through the standard `cuda-intrinsics-gen`
     pipeline:
     ```text
     intrinsics/evidence/* + intrinsics/overlay/*.toml + intrinsics/abi-v1.toml
       ──► cargo run -p cuda-intrinsics-gen
       ──► just -f cuda-oxide/Justfile check-intrinsics
     ```
   * Ensure target floors (`minimum_sm`, `minimum_ptx` in `intrinsics/catalog.json`)
     match the actual PTX ISA specification (for example, `sm_75` for Turing
     scalar math and `m16n8k8` FP16 MMA, `sm_100a` for `tcgen05`), rather than
     accidentally gating older instructions to a newer architecture.

3. **Preserve loop-unroll and shared-memory address-space contracts end-to-end.**
   * **Loop unrolling (`#[unroll]`, `#[unroll(N)]`, `#[unroll(never)]` / `#[nounroll]`):**
     ```text
     #[unroll]        ──► full unroll in `mir-transforms` (compile-time trip count, `while` & range `for`)
     #[unroll(N)]     ──► partial unroll by N in `mir-transforms` + emits `!llvm.loop.unroll.disable`
     #[unroll(never)] ──► keeps 1x loop in `mir-transforms`       + emits `!llvm.loop.unroll.disable` (`.pragma "nounroll"`)
     ```
     Always propagate `!llvm.loop.unroll.disable` onto the lowered backedge when
     partial unrolling or `#[unroll(never)]` / `#[nounroll]` is requested, so
     LLVM `opt -O2` and `ptxas` do not secretly fully unroll the loop and blow up
     register pressure.
   * **32-bit shared-memory pointers (`addrspace(3)`):** Keep `SharedArray` and
     `DynamicSharedArray` pointer arithmetic (`.add()`, `.cast()`) in
     `addrspace(3)` (`p3:32:32:32` with `--nvptx-short-ptr` on `llc`). This
     allows LLVM and `ptxas` to emit 32-bit `%r` registers and fold constant byte
     offsets directly into `ld.shared` / `st.shared` / `ldmatrix` immediates.
   * **Zero stack spills:** Runtime-indexed local arrays (`arr[i]` where `i` is
     not a compile-time constant) cannot be promoted to registers by `mem2reg` or
     LLVM SROA and spill to `.local` memory. Use `#[unroll] for` with constant
     bounds and check `cargo oxide build <target> --verbose` + `ptxas -v` to
     confirm `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`.

---

## 3. Kernel engineering & benchmarking lessons (from `speakrs` and SOTA workloads)

Real-world deployments (such as [speakrs](https://avencera.com/blog/writing-gpu-kernels-in-rust-speakrs-60x-faster-than-pyannote))
and Blackwell SOTA benchmarks highlight four practical rules for agents:

1. **Prefer typed `cuda_device` / `ctm_rs` / `cute_rs` operations over opaque `ptx_asm!`.**
   * Opaque `ptx_asm!` hides address arithmetic and memory effects from LLVM
     (causing Loop Strength Reduction to hoist base pointers and lose immediate
     offsets, or forcing overly conservative memory clobbers across
     `thread::sync_threads()`). Use typed shared-memory pointers, `warp::*`
     collectives, `mbarrier`, and generated intrinsics wherever available.

2. **Profile per GPU architecture — never assume one kernel wins everywhere.**
   * Different GPUs have different execution units and bottlenecks (Turing `sm_75`
     relies on FP16 `mma.sync`; Ampere/Ada `sm_80`/`sm_89` add `cp.async` and
     TF32/FP16 `mma.sync`; Blackwell `sm_100a`/`sm_120a` adds 5th-gen `tcgen05.mma`,
     Tensor Memory `TMEM`, `tcgen05.cp`, and TMA). Always check which hardware unit
     the SOTA baseline uses on the target GPU.

3. **Measure end-to-end payoff first, and make CUDA Graph comparisons apples-to-apples.**
   * When benchmarking multi-kernel pipelines with CUDA Graphs, ensure every
     implementation captures the **exact same DAG of kernels** between
     `BeginCapture` and `EndCapture`, and measure steady-state execution by
     replaying the captured graph `1,000×` between CUDA events:

     ```text
     THE 4 KERNELS OF THE MOE LAYER:
     [K1: Quantize BF16->FP4] ──► [K2: Route] ──► [K3: FC1 + ReLU²] ──► [K4: FC2 + Finalize]

     APPLES-TO-APPLES CUDA GRAPH CAPTURE & 1,000× REPLAY TIMER
     ─────────────────────────────────────────────────────────
        BeginCapture                                                      EndCapture
        ┌─────────────────────────────────────────────────────────────────────────┐
        │  [K1: Quantize] ──► [K2: Route] ──► [K3: FC1+ReLU²] ──► [K4: FC2+Final] │
        └─────────────────────────────────────────────────────────────────────────┘
        Start Event ──► Replay Graph #1 .. #1000 ──► Stop Event (mean = total / 1000)
     ```

   * Never compare a graph that leaves `K1` outside `BeginCapture` against a
     graph that captures `K1 + K2 + K3 + K4` inside `BeginCapture`.

4. **Keep validation fast, numerical, and self-contained.**
   * Verify every kernel change against a CPU or reference oracle (`normalized_max_error`),
     check register/spill counts with `ptxas -v`, and run an A/B timing comparison.

---

## 4. Documentation style: simple, short, jargon-free, visual, and 100% complete (public + private)

All documentation in `cuda-oxide` (crate-level `//!` docs, kernel docstrings,
`README.md` files, and compiler pass headers) **must be easy to read at a glance
and cover every single item — both public and private**:

* **Mandatory coverage of all public AND private items (2–3+ lines per item):**
  Every single item in a file — public or private `fn`, `#[kernel]`, `struct`,
  struct field, `enum`, enum variant, `type` alias, `const`, `static` shared
  array, and `mod` — **must have at least 2 to 3 lines of documentation**
  explaining what it is, its shape/units/invariants, and how it fits into the
  pipeline. Never silence `missing_docs` or `clippy::missing_docs_in_private_items`
  with `#[allow(...)]`.
* **Avoid walls of dense jargon.** Explain *what* the code does and *why* in
  plain, direct sentences.
* **Use visual ASCII diagrams** to show:
  1. The end-to-end kernel pipeline (`[K1] ──► [K2] ──► [K3] ──► [K4]`).
  2. How data moves through Global Memory $\rightarrow$ Shared Memory $\rightarrow$
     Tensor Memory (`TMEM`) / Registers.
  3. What each warp or thread group does when warps are specialized (e.g.,
     Producer Warp vs. Compute Warps).
* State safety, alignment, and collective-synchronization obligations on all
  `unsafe` functions and kernels without claiming that types prove thread
  convergence.
