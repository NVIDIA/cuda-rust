# From Python CuTeDSL to cute-rs

If you have used NVIDIA's Python **CuTeDSL**, the concepts in cute-rs will feel immediately familiar: layouts, composed swizzles, `zipped_divide`, TMA copy atoms, and tiled MMA. Both compile down to the **exact same CUTLASS `cute` and `cute_nvgpu` MLIR operations**.

The key difference is **where the static configuration lives** and **when the IR is constructed**:

```{figure} images/cute-rs-vs-cutedsl.svg
:align: center
:width: 100%

Python CuTeDSL traces Python objects wrapping live MLIR values at runtime; Rust cute-rs encodes layouts, swizzles, and MMA plans in Rust types and translates them ahead of time.
```

---

## Why the Kernel Signatures Look Different

In Python **CuTeDSL**, `@cute.jit` traces a Python function at runtime to build an MLIR module. Because Python objects like `cute.Tensor`, `cute.CopyAtom`, `cute.TiledMma`, and `cute.ComposedLayout` directly wrap MLIR values, the host script constructs all of them and passes them as parameters into `@cute.kernel`:

```python
# Python CuTeDSL: 18 parameters wrapping MLIR types during @cute.jit tracing
@cute.kernel
def kernel(
    self,
    tma_atom_a: cute.CopyAtom,              mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,              mB_nkl: cute.Tensor,
    tma_atom_sfa: cute.CopyAtom,            mSFA_mkl: cute.Tensor,
    tma_atom_sfb: cute.CopyAtom,            mSFB_nkl: cute.Tensor,
    tma_atom_c: cute.CopyAtom,              mC_mnl: cute.Tensor,
    tiled_mma: cute.TiledMma,               cta_layout_mnk: cute.Layout,
    a_smem_layout: cute.ComposedLayout,     b_smem_layout: cute.ComposedLayout,
    sfa_smem_layout: cute.Layout,           sfb_smem_layout: cute.Layout,
    epi_smem_layout: cute.ComposedLayout,   tile_sched_params: PersistentTileSchedulerParams,
): ...
```

In Rust **cute-rs**, there is no runtime tracer: rustc compiles the kernel ahead of time. Nothing is hidden:
1. **Kernel parameters carry only the hardware launch ABI** (for example, 128-byte hardware TMA descriptors passed by value via `#[grid_constant]`, device pointers, and scalar dimensions).
2. **Layouts, swizzles, operand roles, and MMA configurations live in Rust types** (`ASmem`, `Sm100TiledMma<...>`, `MainloopPipeline`) with zero runtime byte overhead.

```rust
// Rust cute-rs: static configuration lives in types; arguments carry only runtime TMA descriptors
type ASmem = ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<128, 64>>;
type B0Smem = ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<96, 64>>;
type B1Smem = ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<80, 64>>;
type CSmem = ComposedLayout<Swizzle<2, 3, 3>, 0, RowMajor<128, 32>>;
type TiledMma = Sm100TiledMma<ASmem, B0Smem, B1Smem, 192, 160>;
type MainloopPipeline = PipelineTmaUmma<5, 77_824>;

#[kernel]
pub fn fp16_gemm_256x352(
    #[grid_constant] a: &TensorMap<f16, ASmem>,
    #[grid_constant] b0: &TensorMap<f16, B0Smem>,
    #[grid_constant] b1: &TensorMap<f16, B1Smem>,
    #[grid_constant] c: &TensorMap<f16, CSmem>,
    bias: *const f16, m: u64, n: u64, k: u64, has_bias: u64,
) { ... }
```

---

## How mir-importer Translates Rust Types to CuTe IR

When mir-importer translates a cute-rs call, it matches the callee's canonical rustc `DefId`, structurally walks its monomorphized generic type arguments (`static_config.rs`), runs cute-layout static validation proofs, and emits a dialect-cute operation combining those static attributes with the call's runtime SSA operands:

```{figure} images/rust-types-to-cute-ir.svg
:align: center
:width: 100%

Compile-time Rust types supply the static IR attributes (shape, stride, swizzle, operand role, MMA partitions), while kernel values supply the runtime SSA operands.
```

---

## Side-by-Side Concept Map

| Concept | Python CuTeDSL | Rust cute-rs |
| :------ | :------------- | :----------- |
| **Layout definition** | `cute.make_layout((128, 64), stride=(64, 1))` | `RowMajor<128, 64>` or `Layout<(C<128>, C<64>), (C<64>, C<1>)>` |
| **Swizzled SMEM layout** | `cute.make_composed_layout(Swizzle(3, 3, 3), 0, layout)` | `ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<128, 64>>` |
| **Tensor view** | `cute.make_tensor(ptr, layout)` | `Tensor::from_slice(a)`, `TensorMut::from_disjoint_slice(&mut c)`, `Tensor::from_storage(storage, layout)`: constructors are `from_*`, the Rust form of `make_*` |
| **Tiling** | `cute.local_tile(mA, tiler, coord)` | `Tensor::local_tile::<TILE>(tile)`, equal to `zipped_divide::<TILE>().slice(tile)`: a one-dimensional tiler and one tile coordinate, no projection |
| **Register fragment** | `cute.make_fragment_like(tensor, dtype)` | no function: the tile or MMA atom type fixes the fragment. `RegisterTile<T, TILE>` comes from `Tensor::load()` or `RegisterTile::from_array(..)`; `FragmentA`, `FragmentB` and `FragmentC::ZERO` serve the MMA atoms |
| **TMA descriptor** | `(tma_atom, tma_tensor)` pair passed into `@cute.kernel` | `#[grid_constant] a: &TensorMap<f16, ASmem>` (128B by value) |
| **TMA encoding (host)** | `cpasync.make_tiled_tma_atom(op, mA, smem_layout, cta_tiler)` | `create_tensor_map_tiled::<T, SmemL>(ptr, rows, cols, ld)` in `cute_rs::copy::tma`, the rank-three SM100 form in `cute_rs::attention::sm100`; `size_in_bytes::<T, SmemL>()` gives the expected bytes |
| **TMA copy** | `cute.copy(tma_atom, src, dst, tma_bar_ptr=bar)` in both directions | `copy_tma_2d` and `copy_tma_s2g_2d`; `Sm100ClusterTmaCopy::copy`; `TmaCopy::load(desc, &mut dst, row, col, barrier, rank)` and `TmaCopy::store(desc, &src, row, col)` in `cute_rs::attention::sm100` |
| **Shared tensor view** | `storage.sA.get_tensor(a_smem_layout)` | `SharedTensor<Mxf4E2M1, f16, SmemP, Mkl>` or `Sm100SharedTile<ASmem, OperandA>` |
| **TMA pipeline** | `pipeline.PipelineTmaAsync.create(...)`, `pipeline.PipelineTmaUmma.create(...)`, `pipeline.PipelineUmmaAsync.create(...)`, `pipeline.PipelineAsyncUmma.create(...)`, `pipeline.PipelineAsync.create(...)` | `PipelineTmaAsync<STAGES, CONSUMER_WARPS, TX_BYTES>`, `PipelineTmaUmma<5, 77_824>` (in `cute_rs::sm100`), or `PipelineTmaUmma<STAGES>`, `PipelineUmmaAsync<STAGES>`, `PipelineAsyncUmma<STAGES>`, `PipelineAsync<STAGES>` and `TwoCtaRing` (in `cute_rs::attention::sm100::pipeline`) |
| **TMEM allocator** | `utils.TmemAllocator` with `allocate(cols)`, `relinquish_alloc_permit()`, `retrieve_ptr()`, `free(ptr, cols)` | `Sm100Tmem<512>` and `Sm100DeallocBarrier` (2-CTA GEMM `dialect-cute`), or `TmemAllocator<COLUMNS, GROUP>` with `allocate`, `relinquish_alloc_permit`, `retrieve_ptr`, `free` and `tmem_row_address` (SM100 attention) |
| **Pipeline state** | `make_pipeline_state(PipelineUserType.Producer, stages)`, `state.index`, `state.phase`, `state.advance()` | `PipelineState::<Producer, STAGES>::new()`, `index() -> usize`, `phase()`, `advance()`; the SM100 attention ring's `RingCursor::index() -> u32` |
| **Pipeline producer step** | `producer_acquire(state)` waits on the empty barrier and arrives with the expected bytes, under `is_leader_cta` in `PipelineTmaUmma` | `producer_acquire` waits only; the elected lane then calls `producer_expect_tx` on `PipelineTmaAsync`, `PipelineTmaUmma` and `TwoCtaRing` (`producer_expect_tx_cursor` with a cursor) |
| **Persistent scheduler** | `StaticPersistentTileScheduler.create(params, block_idx, grid_dim)`, `get_grid_shape(params, max_active_clusters)`, `work = get_current_work()`, `work.is_valid_tile`, `work.tile_idx`, `advance_to_next_work(advance_count=1)` | `StaticPersistentTileScheduler::<M_TILES, N_TILES, BATCHES>::new_1d()`, `get_grid_shape(sm_count, active_ctas_per_sm) -> Option<u32>`, `has_work()`, `get_current_work()`, `tile_idx() -> (m, n, l)` with `linear_idx()`, `advance_to_next_work()` |
| **Tiled MMA** | `cute.make_tiled_mma(mma_op, atom_layout, ...)` | `Mxf4TiledMma::<SmemP>::get_slice(lane)` or `Sm100TiledMma<ASmem, B0Smem, B1Smem, 192, 160>` |
| **Tiled copy** | `thr_copy = tiled_copy.get_slice(tidx)`; `cute.copy(tiled_copy, thr_copy.partition_S(gA), thr_copy.partition_D(sA))` | `tiled_copy.get_slice(thread_idx).copy(&g_a, tile_coord, &mut s_a)`: both partitions come from the plan's layouts inside `copy_g2s`. No kernel can reach this surface yet; see the next section |
| **Register vectors** | `TensorSSA` with `a + b`, `a - b`, `a * b`, `a / b`, `a // b`, `-a`, `where`, conversions and reductions | `RegisterTile<T, N>` with `+`, `-`, `*`, `/` and unary `-`; `map`, `zip_map`, `fold` and `gather` for the rest |
| **Architecture primitives** | `cute.arch.thread_idx()`, `warp_idx()`, `elect_one()`, `cluster_arrive()`, `cluster_dim()`, `fence_view_async_shared()`, `fence_view_async_tmem_load()`, `fence_view_async_tmem_store()`, `{add,mul,fma}_packed_f32x2()`, `cvt_f16x2_f32()`, `warp_reduction_{max,sum}()`, `fence_sync_restrict(MemOrder.ACQUIRE)`, ... | `cute_rs::arch::thread_idx()`, `warp_idx()`, `elect_one_sync()`, `cluster_arrive()`, `cluster_dim()`, `fence_view_async_shared()`, `fence_view_async_tmem_load()`, `fence_view_async_tmem_store()`, `{add,mul,fma}_packed_f32x2()`, `cvt_f16x2_f32()`, `warp_reduction_{max,sum}_f32()`, `fence_sync_restrict::<Acquire>()`, ...: forwarders over the PTX-named ctm-rs layer |

---

## Where cute-rs Deviates, and Why

The names follow CuTeDSL 4.7. Where the Rust form differs on purpose, the difference and its reason are listed here, so a CuTeDSL author is not surprised.

| Concept | CuTeDSL 4.7 | Rust cute-rs | Why |
| :------ | :---------- | :----------- | :-- |
| **Constructors and plans** | `make_layout`, `make_tensor`, `make_tiled_mma` and `make_tiled_copy` build objects while tracing | `new` and `from_*` constructors; a compile-time plan is a type or a const generic (`RowMajor<128, 64>`, `TiledCopy<Atom, ThreadLayout, ValueLayout, TileLayout>`, `zipped_divide::<8>()`) | the compiler reads the plan from the generic arguments, so there is nothing for a `make_*` call to build |
| **Pipeline producer step** | `producer_acquire` folds the arrive-expect into the wait, under `is_leader_cta` in `PipelineTmaUmma` | `producer_acquire` waits; the elected lane calls `producer_expect_tx`, or `producer_expect_tx_cursor` on `TwoCtaRing` | electing the lane is a convergence obligation the compiler does not prove, so it stays visible in kernel code; the pipeline value stores pointers only |
| **Scheduler** | `WorkTileInfo.is_valid_tile`; `get_grid_shape(params, max_active_clusters)` returns `(m, n, persistent_clusters)`; `advance_to_next_work(advance_count=1)`; `tile_idx` is a `Coord` | `has_work()` on the scheduler; `get_grid_shape(sm_count, active_ctas_per_sm) -> Option<u32>`; `advance_to_next_work()`; `tile_idx() -> (m, n, l)` with `linear_idx()` beside it | the grid is one dimensional and has no params object; the bounds check stays outside the coordinate arithmetic; the step is always one grid of blocks; the coordinate is a Rust tuple |
| **Pipeline state** | `PipelineState.index` is an `Int32` | `PipelineState::index() -> usize`; `RingCursor::index() -> u32` | a `usize` scales a shared-buffer offset directly; the SM100 attention ring keeps the `u32` its barriers take |
| **TMA names** | `make_tiled_tma_atom` returns the atom and a TMA tensor; one `cute.copy` loads and stores | `TensorMap` is the one descriptor type; `create_tensor_map_tiled` is the one host encoder, with the rank-three SM100 form in `cute_rs::attention::sm100`; `TmaCopy::load` and `TmaCopy::store` stay two methods; `size_in_bytes::<T, SmemL>()` is `cute.size_in_bytes`, a cosize, so a padded row stride counts and a swizzle does not | the descriptor is CUDA's `CUtensorMap` typed by element and shared layout; load takes a barrier and a CTA rank and store takes neither, so one signature cannot serve both |
| **Clusters** | `cute.arch.cluster_dim()` is the number of clusters in the grid; `cute.arch.block_in_cluster_dim()` is CTAs per cluster | the same two names with the same meanings | the compiler's own `cluster_dim` vocabulary (the `#[cluster_launch]` attribute, the `cluster_dim` field of `cuda_launch!` and the `cluster_dim_x/y/z` NVVM annotations) means CTAs per cluster; in a kernel, `cluster_dim()` reads as the DSL does and `block_in_cluster_dim()` is the launch value |
| **Named barriers** | `pipeline.NamedBarrier(barrier_id, num_threads)` with `arrive_and_wait()`, its alias `sync()`, `arrive()`, `wait()` and the unaligned forms | `NamedBarrier::<ID, THREADS>::sync()` only; `cute_rs::arch::barrier(id, n)` and `barrier_arrive(id, n)` are the `cute.arch` free functions | both values are const generics, so the id and the thread count (a multiple of 32, at most 1024) are checked at compile time; the split arrive and wait methods are not bound on the type |
| **Sync-restricted fence** | `fence_sync_restrict(order)` in the experimental primitives layer, `order` a `MemOrder` string enum | `cute_rs::arch::fence_sync_restrict::<Acquire>()`, the order a sealed marker type; acquire only | the instruction is chosen when the kernel compiles; the release form waits for its intrinsic to be bound |
| **Tiled copy** | `make_tiled_copy`, `get_slice`, `partition_S`, `partition_D`, `cute.copy` | `TiledCopy`, `get_slice` and `ThrCopy::copy` exist, but no kernel can reach them yet | `GlobalCopyTensor` has no public constructor: `from_carrier` is `pub(crate)` and the only public producers are the block-scaled tensors' `values_for_copy` and `scale_words_for_copy`; the importer's `shared_tensor_overlay` hook accepts only the block-scaled roles `Mkl` and `Nkl`; the verifier rejects a role-typed shared carrier as a `copy_g2s` destination. The kernels that need a block-wide copy call `copy_g2s` directly |
| **Register vectors** | `/` on integer vectors converts both sides to f32 and divides; `//` floors; `-a` is `0 - a` | `/` on integer lanes truncates toward zero; unary `-` negates each lane, so a `+0.0` lane becomes `-0.0` | the operators are Rust's own: floating-point division matches the DSL, integer division and the sign of a negated zero differ |

ctm-rs keeps PTX instruction names (`cp_async_bulk_tensor_shared_cta_global`, `mbarrier_arrive_expect_tx`, `fence_sync_restrict_acquire`); the `cute.arch` concept names live in `cute_rs::arch` as forwarders over them. Dialect op, type and attribute names are compiler-internal and are not part of this surface.
