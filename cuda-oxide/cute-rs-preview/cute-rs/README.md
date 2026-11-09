# cute-rs

CuTe-style tensor computation in Rust, organized 1-to-1 with Python **CuTeDSL 4.7** (`cutlass/cute`, `cutlass/pipeline`, `cutlass/utils`) and compiled through `dialect-cute` and the pinned CUTLASS 4.7 MLIR compiler (`0` `cutlass-primitives` dependencies).

```text
CuTeDSL (Python)                          cute-rs/src/ (1-to-1 Module Tree)
────────────────────────────────────────  ────────────────────────────────────────
cutlass/cute/                             cute-rs/src/
├── core.py         (Layout, Composed)    ├── layout.rs       (re-exports cute-layout)
├── tensor.py       (Tensor, TensorSSA)   ├── tensor/         (Tensor, RegisterTile)
├── atom.py         (CopyAtom, MmaAtom)   ├── atom/           (TiledCopy, copy_g2s)
├── typing.py       (f16, bf16, fp4, fp8) ├── numeric.rs      (BFloat16, Float4E2M1FN)
├── arch/           (barriers, tmem, ...) ├── arch.rs         (thread_idx, NamedBarrier)
└── nvgpu/          (GPU hardware atoms)  └── nvgpu/
    ├── common.py                             ├── common.rs   (OperandA, OperandB, Mkl, Nkl)
    ├── cpasync/    (G2S cp.async & TMA)      ├── cpasync/    (TensorMap, copy_tma_2d, TmaCopy)
    ├── warp/       (SM80/SM120 mma/ldm)      ├── warp/       (FragmentA/B/C, Bf16Mma16x8x16, Mxf4TiledMma)
    └── tcgen05/    (SM100 UMMA & TMEM)       └── tcgen05/    (Sm100TiledMma, Sm100BlockScaledTiledMma, MmaSs, MmaTs, TmemFragment)

cutlass/pipeline/                         ├── pipeline/       (PipelineTmaAsync, PipelineTmaUmma,
                                          │                    PipelineUmmaAsync, PipelineTmaStore, TwoCtaRing)
cutlass/utils/                            └── utils/
                                              ├── tmem_allocator.rs (TmemAllocator, Sm100Tmem)
                                              ├── scheduler.rs      (StaticPersistentTileScheduler)
                                              └── blockscaled/      (Sm1xxBlockScaleKMajor, BlockScaledTensor)
```

| Module | CuTeDSL Counterpart | Contents |
| --- | --- | --- |
| `layout` | `cutlass.cute.core` | Static layout markers (`RowMajor`, `ColMajor`, `Swizzle`, `ComposedLayout`), `CopyG2S`, `CopyUniversal`, `LeadingDim` |
| `tensor` | `cutlass.cute.tensor` | `Tensor`, `TensorMut`, `RegisterTile` (`TensorSSA`), `ReductionOp`, `zipped_divide`, `local_tile`, `partition::{StridedTensor, StridedTensorMut}` (lowered to MLIR `vector<NxT>` & `cute.*` ops) |
| `atom` | `cutlass.cute.atom` | `TiledCopy`, `ThrCopy`, `SmemTile`, `copy_g2s`, `copy_s2g` |
| `numeric` | `cutlass.cute.typing` | `f16`, `BFloat16`, `Float8E4M3FN`, `Float8E8M0FNU`, `Float4E2M1FN`, `Float4E2M1FNx8`, `Acquire` |
| `arch` | `cutlass.cute.arch` | Thread/warp/cluster coordinates, `NamedBarrier`, `mbarrier_*`, TMEM and warp reduction forwarders over `cuda-device` |
| `nvgpu` | `cutlass.cute.nvgpu` | Hardware copy & MMA atoms (`cpasync`, `warp`, `tcgen05`), all lowered via `__compiler` $\rightarrow$ `dialect-cute` |
| `pipeline` | `cutlass.pipeline` | `PipelineTmaAsync`, `PipelineTmaUmma`, `PipelineUmmaAsync`, `PipelineTmaStore`, `PipelineState`, `SingleCtaRing`, `TwoCtaRing` |
| `utils` | `cutlass.utils` | `TmemAllocator`, `StaticPersistentTileScheduler`, `blockscaled` (`Sm1xxBlockScaleKMajor`, `BlockScaledTensor`) |

The crate is `no_std`. The opt-in `host` feature exposes host TMA descriptor encoding (`create_tensor_map_tiled`, `create_tensor_map_3d`). Compile-time checking of a view does not establish thread convergence, allocation readiness, or safe asynchronous reuse.

## Import paths

Every module has one defining path matching CuTeDSL. Root item re-exports such as `cute_rs::TensorMap` name the exact same items as their defining modules (`cute_rs::nvgpu::cpasync::TensorMap`) without wrapping or duplicating implementations.

## Checks

Run from the `cuda-oxide/` workspace with its pinned toolchain:

```sh
cargo test -p cute-rs --no-default-features
cargo test -p cute-rs --features host
cargo clippy -p cute-rs --all-targets --features host -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc -p cute-rs --features host --no-deps
```
