# Support Matrix (preview)

cute-rs is a **preview**. It compiles Rust CuTe kernels ahead of time through the pinned CUTLASS 4.7 compiler for two targets, `sm_100a` and `sm_120a`, and covers a validated subset of what CuTeDSL 4.7 offers on them. This page lists what the subset holds, what it does not, and what cute-rs adds. Status at commit `5eaa6cbda`, 2026-10-22.

```{figure} images/support-matrix.svg
:align: center
:width: 100%

One glance: CuTeDSL 4.7 across its architectures against the cute-rs preview on its two targets.
```

## Targets

```{figure} images/support-targets.svg
:align: center
:width: 100%

The two targets the preview compiles for, and the ones it rejects.
```

:::{dropdown} Text version

| Target | GPU | CuTeDSL 4.7 | cute-rs preview |
| :--- | :--- | :--- | :--- |
| `sm_80`, `sm_86`, `sm_87` | Ampere | yes | **rejected** |
| `sm_89` | Ada | yes | **rejected** |
| `sm_90a` | Hopper | yes | **rejected** |
| `sm_100a` | Blackwell datacenter, B200 class | yes | **yes** |
| `sm_100`, `sm_100f`, `sm_103a`, `sm_110a` | Blackwell datacenter, other variants | yes | **rejected** |
| `sm_120a` | Blackwell GeForce, RTX 50 series | yes | **yes** |
| `sm_120`, `sm_120f`, `sm_121a` | Blackwell GeForce, other variants | yes | **rejected** |

:::

The target check runs before any MLIR is built, and the error names the target. Within the pair, `tcgen05` operations need `sm_100a` and MXF4 register MMA needs `sm_120a`. `cargo oxide run` uses the detected GPU, so an RTX 5090 gets `sm_120a`; `cargo oxide build` needs `--arch`. Every CuTe kernel needs `CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir`; the native backend has no CuTe lowering.

## Feature areas

The figure at the top of the page is the feature table, with the cute-rs column split by target. Where each row is proven:

- Layout algebra, static: cute-layout tests, elementwise_cute, blockscale_gemm_cute
- Dynamic layouts, runtime shapes: fmha_prefill_sm120a_cute, fmha_decode_sm120a_cute
- Tensors, partitions, tiled atoms: elementwise_cute, nvfp4_gemv_cute, blockscale_gemm_cute
- TMA copies: blockscale_gemm_cute, fp16_gemm_256x352_cute, sm100_fmha_prefill_cute
- cp.async, bulk, DSMEM copies: fmha_prefill_sm120a_cute
- ldmatrix, stmatrix, atomics: blockscale_gemm_cute, ffn_cute, sm100_mla_decode_fp16_cute
- tcgen05 MMA kinds: fp16_gemm_256x352_cute, sm100_fmha_prefill_cute, sm100_mla_decode_fp16_cute
- Warp MMA kinds (mma.sync): blockscale_gemm_cute, ffn_cute, fmha_prefill_sm120a_cute, qwen3_8b_cute
- NVFP4 tensor cores (E4M3 scales): qwen3_8b_cute (SM120 m16n8k64 E2M1 + E4M3 SF_VEC=16 prefill)
- wgmma (Hopper): none
- TMEM: fp16_gemm_256x352_cute, sm100_fmha_prefill_cute
- Pipelines and barriers: blockscale_gemm_cute, fp16_gemm_256x352_cute, sm100_mla_decode_fp16_cute
- Schedulers, persistence, PDL: blockscale_gemm_cute, sm100_fmha_prefill_cute
- Clusters and multicast: fp16_gemm_256x352_cute, sm100_mla_decode_fp16_cute
- Epilogue fusion (EFC): blockscale_gemm_cute, fp16_gemm_256x352_cute, swiglu_cute
- Numerics and intrinsics: rmsnorm_cute, swiglu_cute, qwen_decoder_cute
- Control flow, printf, debugging: every example
- Host: JIT, torch, JAX, AOT export: every example; fp16_gemm_256x352_cute for CUDA graphs
- Autotuning, Task Scheduling, IKET: none
- Verification before codegen: dialect-cute verify tests, 48 cute-rs and 30 ctm-rs compile-fail cases
- Grouped, MoE, mixed-input, FP8: nemotron3_moe_cute (NVFP4 ReLU² grouped MoE), sm100_qkv_epilogue_cute (fused QKV + RMSNorm + FP8 quant + paged KV scatter)
- Shipped kernel examples: the 15 crates under cute-rs-preview/examples
- End-to-end models: qwen_decoder_cute, qwen3_8b_cute, nemotron3_moe_cute

:::{dropdown} Text version of the feature table

"partial" means a fixed or narrower form of the feature exists; "n/a" means the feature does not apply to those targets.

| Feature area | CuTeDSL 4.7 | cute-rs preview | What the preview has |
| :--- | :--- | :--- | :--- |
| Layout algebra, static | yes | partial | layouts are Rust types checked at compile time; no layout ops in kernels |
| Dynamic layouts, runtime shapes | yes | partial | runtime lengths and one runtime row pitch; strides and tiles are types |
| Tensors, partitions, tiled atoms | yes | partial | 1D tiles (zipped_divide, slice, 1D local_tile), StridedTensor, and TiledCopy / ThrCopy (copy_g2s, copy_s2g, copy_r2s) |
| TMA copies | yes | partial | 2D tile load and store, two-CTA load; no im2col, reduce-store, multicast |
| cp.async, bulk, DSMEM copies | yes | partial | cooperative cp.async with a coverage proof; no bulk or DSMEM data copies |
| ldmatrix, stmatrix, atomics | yes | partial | ldmatrix x4 and x2.trans (f16 and BFloat16), stmatrix x2, no-allocate stores; no atomics |
| tcgen05 MMA kinds | yes | partial, sm_100a only | 1 of 11 kinds: F16 in, F32 accumulate, one or two CTAs |
| Warp MMA kinds (mma.sync) | yes | partial | 4 of 7 kinds: MXF4 with UE8M0 scales, NVFP4 with E4M3 scales (sm_120a), F16, BF16 |
| NVFP4 tensor cores (E4M3 scales) | yes | partial, sm_120a only | SM120 m16n8k64 E2M1 with E4M3 scales (SF_VEC=16) in Nvfp4TiledMma and qwen3_8b_cute |
| wgmma (Hopper) | yes | n/a | no sm_90a target |
| TMEM | yes | partial, sm_100a only | `Sm100Tmem`, `TmemAllocator`, 32x32b load and store (`TmemFragment`), FP32 to FP16 epilogues; no 16xNb, LdRed or S2T |
| Pipelines and barriers | yes | partial | typed 1-CTA (`PipelineTmaAsync`, `PipelineTmaStore`, `PipelineTmaUmma`, `PipelineUmmaAsync`, `PipelineAsyncUmma`, `PipelineAsync`) and 2-CTA (`TwoCtaRing`) pipelines; no CLC |
| Schedulers, persistence, PDL | yes | partial | one static 1D persistent scheduler; no grouped, CLC or stream-K |
| Clusters and multicast | yes | partial, sm_100a only | two-CTA clusters validated; no N-CTA multicast plan |
| Epilogue fusion (EFC) | yes | partial | two fixed FP16 epilogues, bias on sm_100a; activations fused by hand |
| Numerics and intrinsics | yes | partial | f16, f32 native; BF16 and FP4 carriers; no inline PTX or libdevice |
| Control flow, printf, debugging | yes | partial | full Rust control flow and traps, MLIR dumps; no device printf |
| Host: JIT, torch, JAX, AOT export | yes | partial | AOT cubin inside the Rust binary; no JIT, no Python interop |
| Autotuning, Task Scheduling, IKET | yes | no | no autotuner, schedule checker or in-kernel tracing |
| Verification before codegen | yes | yes | whole-module layout, pipeline and role checks; CuTeDSL checks at trace |
| Grouped, MoE, mixed-input, FP8 | yes | partial, sm_100a | NVFP4 grouped MoE (nemotron3_moe_cute) and FP8 QKV epilogue (sm100_qkv_epilogue_cute) |
| Shipped kernel examples | yes | partial | 5 sm_100a and 10 sm_120a example crates versus 86 CuTeDSL kernel files |
| End-to-end models | no | sm_100a + sm_120a | Nemotron-3 MoE (sm_100a), Qwen3-TTS 1.7B and Qwen3-8B (sm_120a); CuTeDSL ships none |

:::

## Feature additions in CuTeDSL 4.4 to 4.7

```{figure} images/support-additions.svg
:align: center
:width: 100%

What a CuTeDSL user relies on beyond the per-architecture atoms, and what the preview has instead. Epilogue fusion is the row most often asked about.
```

:::{dropdown} Text version, with the full comparison

| Feature | CuTeDSL 4.7 | cute-rs preview | Status |
| :--- | :--- | :--- | :--- |
| Epilogue fusion (EFC) | A Python epilogue function over the accumulator and extra tensors; the GEMM stages its inputs through TMA, runs it per sub-tile with built-in activations, alpha, beta and bias, and stores every output. | Two fixed epilogues: FP32 to FP16 with a row bias on sm_100a, FP32 to FP16 through stmatrix on sm_120a. Activations are fused by hand in register code (swiglu_cute, ffn_cute). No user hook. | partial |
| Dynamic layouts | Fully dynamic layouts; one compiled kernel serves any shape. | Runtime lengths and one runtime row pitch; tile sizes, strides and swizzles are types. One binary covers any M, N, K the static tile divides. | partial |
| JIT and caching | Trace and compile at call time with an on-disk cache. | Ahead of time through rustc and Cargo; nothing to JIT. | n/a |
| AOT export | Export a kernel as a .so plus C header; aarch64 cross build. | The cubin is embedded in the Rust binary. No C export; write an extern C wrapper by hand. | partial |
| TVM FFI, PyTorch, JAX, DLPack | Framework tensors flow in through DLPack; TVM FFI modules for C++, PyTorch and JAX. | None. Buffers live in cuda-core DeviceBuffer; Python appears only in reference harnesses. | no |
| Autotuning | autotune_jit and benchmark helpers search a configuration space and cache the winner. | None. Tile shapes are const generics; tune by instantiating variants in a host harness. | no |
| Task Scheduling | Static analysis of warp-specialized schedules; compilation stops on known concurrency issues (4.7, Primitives-only kernels). | None. The whole-module verifier checks pipeline and epilogue protocol order, not schedules or races. | no |
| Inline PTX and cute.ffi | llvm.inline_asm in kernels; external LLVM bitcode functions. | None on the CuTe path; hardware access goes through the ctm-rs and cuda-device intrinsics. | no |
| printf, print_tensor, IR dumps | Device printf and tensor printing, LaTeX layouts, IR, PTX and SASS dumps. | MLIR, MIR and LLVM dumps and host-side layout Display. No device printf in CuTe kernels. | partial |
| IKET profiling | In-kernel event markers collected into timelines. | None. Use Nsight on the embedded cubin. | no |
| CUDA graphs, PDL, cooperative launch | Graph capture, programmatic dependent launch, cooperative launch, launch events. | Graphs, the PDL launch attribute and cooperative launch from the Rust host. No device-side PDL helpers in kernels. | partial |
| Typed shared memory | SmemAllocator bump allocator, cute.struct, auto-deduced smem size. | Typed static and dynamic shared arrays and layout-typed tiles; carve dynamic shared memory by hand. | partial |
| TensorSSA register vectors | Immutable register vectors with arithmetic, where, conversions and reductions. | RegisterTile with elementwise `+`, `-`, `*`, `/` and unary `-`, map, zip_map, fold and gather; conversions as closures. | partial |
| block_copy | One call issues TMA (multicast hidden) or shared-to-TMEM copies. | A typed two-CTA TMA copy. No shared-to-TMEM copy. | partial |
| Grouped GEMM, MoE, ptr-array | Grouped and MoE persistent schedulers, device-only problem shapes. | None. One launch per group. | no |
| Mixed-input GEMM | Narrow int4 and int8 weights with bf16 or fp8 activations, dequant in the mainloop. | None. Operands share one type. | no |
| Distributed kernels | NVSHMEM all-reduce and GEMM plus collective kernels. | None. Single GPU. | no |
| Grouped block-scaled and blockwise GEMM | Grouped MXFP4 and NVFP4 GEMM, 128x128 blockwise scaling. | Dense block-scaled GEMM and GEMV only. | no |
| FMHA backward | dQ, dK, dV in one SM100 kernel. | Forward only. | no |
| Convolution | TMA im2col copy atom; the 4.7 example tree ships no convolution kernel. | Implicit im2col on the BF16 mma.sync register path in the Qwen codec kernels; no TMA im2col. | partial |
| Sub-byte types | Int4, FP4, two FP6 formats, packed carriers, MXF8F6F4 mixes. | FP4 and scale carriers for block-scaled GEMM and GEMV. No Int4, no FP6. | partial |
| Primitives API (4.7, experimental) | A thin NVVM wrapper beneath CuTe for Tensor Core programming through SIMT. | ctm-rs: typed tcgen05, TMEM, TMA, mbarrier, named barrier, fence, cluster, cp.async and warp primitives for the shipped kernels. No atomics, reductions, wgmma or libdevice. | partial |

:::

## What cute-rs adds

- Ahead-of-time compilation with no Python runtime: one Rust binary carries the cubin.
- Layouts, swizzles, operand roles and pipeline states are Rust types, decoded by the compiler and checked at compile time.
- A whole-module verifier runs before any MLIR is built: tensor provenance, pipeline lifecycle, operand roles, TMA encodability and swizzle alignment.
- Two end-to-end models, Qwen3-TTS 1.7B and Qwen3-8B on sm_120a, with reference comparisons. CuTeDSL 4.7 ships component kernels only.

## What the compiler does not prove

The verifier checks what it can see in types and in the module. These stay with the author at every `unsafe` boundary:

- Thread convergence and lane participation at collective operations.
- Barrier phases and asynchronous completion before a buffer is reused.
- Pointer lifetimes behind raw views and descriptors.
- Disjoint writes across threads.

## Sources

- CuTeDSL 4.7 at tag `v4.7.0` of the public CUTLASS repository: the `python/CuTeDSL/cutlass` package, the `examples/python/CuTeDSL` tree and the Python DSL documentation. CuTeDSL calls itself a public beta.
- The target gate: `crates/cuda-oxide-mlir-export/src/profile.rs`. The pinned compiler: `cargo oxide toolchain install cutlass` installs CUTLASS 4.7.0 and checks its digest.
- The figure is generated from the same row data as this table; regenerate both when the compiler surface changes.
