# From Python `CuTeDSL` to `cute-rs`

If you have used NVIDIA's Python **`CuTeDSL`**, the concepts in `cute-rs`—layouts, composed swizzles, `zipped_divide`, TMA copy atoms, and tiled MMA—will feel immediately familiar. Both compile down to the **exact same CUTLASS `cute` and `cute_nvgpu` MLIR operations**.

The key difference is **where the static configuration lives** and **when the IR is constructed**:

```{figure} images/cute-rs-vs-cutedsl.svg
:align: center
:width: 100%

Python `CuTeDSL` traces Python objects wrapping live MLIR values at runtime; Rust `cute-rs` encodes layouts, swizzles, and MMA plans in Rust types and translates them ahead of time.
```

---

## Why the Kernel Signatures Look Different

In Python **`CuTeDSL`**, `@cute.jit` traces a Python function at runtime to build an MLIR module. Because Python objects like `cute.Tensor`, `cute.CopyAtom`, `cute.TiledMma`, and `cute.ComposedLayout` directly wrap MLIR values, the host script constructs all of them and passes them as parameters into `@cute.kernel`:

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

In Rust **`cute-rs`**, there is no runtime tracer—`rustc` compiles the kernel ahead of time. Nothing is hidden:
1. **Kernel parameters carry only the hardware launch ABI** (for example, 128-byte hardware TMA descriptors passed by value via `#[grid_constant]`, device pointers, and scalar dimensions).
2. **Layouts, swizzles, operand roles, and MMA configurations live in Rust types** (`ASmem`, `Sm100TiledMma<...>`, `MainloopPipeline`) with zero runtime byte overhead.

```rust
// Rust cute-rs: static configuration lives in types; arguments carry only runtime TMA descriptors
type ASmem = Composed<Swizzle<3, 3, 3>, 0, RowMajor<128, 64>>;
type B0Smem = Composed<Swizzle<3, 3, 3>, 0, RowMajor<96, 64>>;
type B1Smem = Composed<Swizzle<3, 3, 3>, 0, RowMajor<80, 64>>;
type CSmem = Composed<Swizzle<2, 3, 3>, 0, RowMajor<128, 32>>;
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

## How `mir-importer` Translates Rust Types to CuTe IR

When `mir-importer` translates a `cute-rs` call, it matches the callee's canonical `rustc` `DefId`, structurally walks its monomorphized generic type arguments (`static_config.rs`), runs `cute-layout` static validation proofs, and emits a `dialect-cute` operation combining those static attributes with the call's runtime SSA operands:

```{figure} images/rust-types-to-cute-ir.svg
:align: center
:width: 100%

Compile-time Rust types supply the static IR attributes (shape, stride, swizzle, operand role, MMA partitions), while kernel values supply the runtime SSA operands.
```

---

## Side-by-Side Concept Map

| Concept | Python `CuTeDSL` | Rust `cute-rs` |
| :------ | :--------------- | :------------- |
| **Layout definition** | `cute.make_layout((128, 64), stride=(64, 1))` | `RowMajor<128, 64>` or `L<T2<C<128>, C<64>>, T2<C<64>, C<1>>>` |
| **Swizzled SMEM layout** | `cute.make_composed_layout(Swizzle(3, 3, 3), 0, layout)` | `Composed<Swizzle<3, 3, 3>, 0, RowMajor<128, 64>>` |
| **TMA descriptor** | `(tma_atom, tma_tensor)` pair passed into `@cute.kernel` | `#[grid_constant] a: &TensorMap<f16, ASmem>` (128B by value) |
| **Shared tensor view** | `storage.sA.get_tensor(a_smem_layout)` | `SharedTensor<Mxf4E2M1, f16, SmemP, Mkl>` or `Sm100SharedTile<ASmem, OperandA>` |
| **TMA pipeline** | `pipeline.PipelineTmaAsync.create(...)` or `pipeline.PipelineTmaUmma.create(...)` | `TmaLoadPipeline<STAGES>` or `PipelineTmaUmma<5, 77_824>` |
| **Tiled MMA** | `cute.make_tiled_mma(mma_op, atom_layout, ...)` | `Mxfp4TiledMma::<SmemP>::new()` or `Sm100TiledMma<ASmem, B0Smem, B1Smem, 192, 160>` |
| **Architecture primitives** | `cute.arch.thread_idx()`, `cute.arch.warp_idx()`, `cute.arch.elect_one()`, `cute.arch.cluster_arrive()`, ... | `cute_rs::arch::thread_idx()`, `warp_idx()`, `elect_one_sync()`, `cluster_arrive()`, ...: forwarders over the PTX-named `ctm_rs` layer, the mask spelled out |
