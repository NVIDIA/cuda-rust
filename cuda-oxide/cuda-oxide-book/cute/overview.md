# Overview: cute-rs and ctm-rs (preview)

:::{admonition} Preview
:class: warning
cute-rs compiles for two targets, `sm_100a` and `sm_120a`, through the pinned CUTLASS 4.7 compiler, and covers a validated subset of CuTeDSL 4.7 on them. The [support matrix](support-matrix.md) lists what is supported, what is partial and what is missing. Expect API changes between releases.
:::

**cute-rs** brings CUTLASS CuTe's layout algebra, partitions, TMA transfers, pipelines, and tensor-core MMA operations into Rust's type system for two Blackwell targets, without a Python interpreter or runtime JIT tracer.

```{figure} images/cute-rs-preview-stack-and-flow.svg
:align: center
:width: 100%

The cute-rs-preview stack: kernels author tensor computation in cute-rs over ctm-rs instruction primitives, import into one Pliron module (dialect-cute, dialect-mir, and dialect-nvvm), pass whole-module verification, and compile through CUTLASS 4.7 MLIR into an embedded cubin.
```

---

## The Two-Layer Kernel Stack

Every CuTe kernel in cuda-oxide is built on two complementary device crates plus a shared layout algebra crate:

| Crate | Layer | What it owns |
| :---- | :---- | :----------- |
| **cute-rs** | Semantic CuTe layer | `Tensor` / `TensorMut`, `zipped_divide`, `slice`, `local_tile`, cooperative/tiled/TMA copies, producer/consumer pipelines, reductions, and tiled MMA (`mma::bf16`, `mma::block_scaled`, `sm100`, `attention::sm100`). |
| **ctm-rs** | Instruction & ABI layer | `#[cuda_module]`, `#[kernel]`, launch contracts, thread/warp/cluster coordinates, `mbarrier` state machines, `tcgen05::tmem::TmemGuard`, and packed register math over cuda-device. |
| **cute-layout** | Shared layout algebra | `IntTuple`, `Layout`, `ComposedLayout`, `Swizzle`, and static compile-time proofs for cooperative copies, `ldmatrix`, and TMA swizzle phase alignment. |

```{figure} images/three-tiers-comparison.svg
:align: center
:width: 100%

When to use cute-rs, ctm-rs, and cuda-device.
```

---

## Your First cute-rs Kernel

In cute-rs, a `Tensor` pairs storage with a logical coordinate layout. Splitting a vector into 4-element (16-byte) tiles and slicing the tile owned by `tid` takes three lines:

```rust
use ctm_rs::memory::DisjointSlice;
use ctm_rs::{cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn add_f32(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let tid = thread::index_1d().get();
        let t_a = cute_rs::Tensor::from_slice(a).zipped_divide::<4>().slice(tid);
        let t_b = cute_rs::Tensor::from_slice(b).zipped_divide::<4>().slice(tid);
        let mut t_c = cute_rs::TensorMut::from_disjoint_slice(&mut c).zipped_divide::<4>().slice(tid);

        if t_a.is_full() {
            // Complete 16-byte tile -> lowers to a vectorized cute.copy
            unsafe { t_c.store(t_a.load() + t_b.load()) };
        } else {
            // Ragged boundary tile -> safe scalar tail loop
            let mut k = t_a.base();
            while k < a.len() {
                unsafe { t_c.store_at(k, a[k] + b[k]) };
                k += 1;
            }
        }
    }
}
```

---

## Run and Inspect the Emitted CuTe MLIR

Install the pinned CUTLASS 4.7 compiler once, then run any `*_cute` example from `cuda-oxide/`:

```bash
# Install the managed CUTLASS 4.7 compiler
cargo oxide toolchain install cutlass

# Run the f32 + f16 elementwise example through the CUTLASS MLIR backend
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
  cargo oxide run elementwise_cute --arch sm_120a

# Dump the prepared CUTLASS MLIR module to inspect the generated cute ops
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
CUDA_OXIDE_MLIR_OUTPUT="$PWD/module.mlir" \
  cargo oxide build elementwise_cute --arch sm_120a
```

:::{note}
dialect-cute, dialect-mir, and dialect-nvvm coexist inside a **single Pliron module** rather than acting as separate sequential passes. Before exporting MLIR, dialect-cute's whole-module verifier checks tensor origins, pipeline state transitions, and TMA/MMA layout and collector contracts.
:::
