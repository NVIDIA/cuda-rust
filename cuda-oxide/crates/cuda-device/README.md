# cuda-device

`#![no_std]` device-side intrinsics and abstractions for writing CUDA kernels in Rust. This crate provides everything that runs on the GPU: thread identification, memory abstractions, synchronization, warp primitives, tensor cores, TMA, atomics, and debug facilities.

```text
  User kernel code
       │
       │  uses
       ▼
  cuda-device  ─────────────────────────────────┐
  │                                             │
  │  thread     warp       barrier     cusimd   │  Universal
  │  disjoint   shared     atomic      debug    │
  │  fence      grid       coop_grps            │
  │                                             │
  │  tma        wgmma      stmatrix    cluster  │  Hopper+
  │  tcgen05    clc                             │  Blackwell+
  └─────────────────────────────────────────────┘
       │
       │  compiled by
       ▼
  rustc-codegen-cuda  →  MIR → LLVM IR → PTX
```

## Modules

| Module               | Description                                                                  | GPU     |
|----------------------|------------------------------------------------------------------------------|---------|
| `thread`             | Thread/block IDs, `index_1d`/`index_2d::<S>`, `sync_threads`                 | All     |
| `config`             | Metadata-only policy shapes, layouts, memory spaces, tiles, and atoms        | All     |
| `disjoint`           | `DisjointSlice<T, IndexSpace>` -- safe parallel writes via `ThreadIndex`     | All     |
| `shared`             | `SharedArray<T, N>`, `DynamicSharedArray<T>` -- block-scoped shared memory   | All     |
| `warp`               | Shuffle (xor/up/down/idx for i32 and f32), lane_id, vote (all/any/ballot)    | All     |
| `atomic`             | Scoped GPU atomics; `core::sync::atomic` types also supported on device      | sm_70+  |
| `debug`              | `clock`/`clock64`, `trap`, `breakpoint`, `gpu_printf!`, `gpu_assert!`        | All     |
| `fence`              | `threadfence_block` / `threadfence` / `threadfence_system` memory fences     | All     |
| `grid`               | Grid-scoped queries and `sync` (cooperative kernel launches only)            | sm_70+  |
| `cooperative_groups` | Typed group handles (`Grid`/`ThreadBlock`/`WarpTile<N>`/`CoalescedThreads`)  | All     |
| `barrier`            | `Barrier`, `ManagedBarrier<State, Kind>` -- async mbarrier for TMA           | sm_90+  |
| `cluster`            | Thread block clusters, DSMEM (`map_shared_rank`), `cluster_sync`             | sm_90+  |
| `tma`                | `TmaDescriptor`, bulk tensor copies (1D-5D global↔shared, multicast)         | sm_90+  |
| `wgmma`              | Warpgroup MMA fence/commit/wait, smem descriptors, bf16/f16/tf32 MMA         | sm_90   |
| `tcgen05`            | 5th-gen tensor cores: TMEM alloc/dealloc, MMA, stmatrix, CG2 variants        | sm_100+ |
| `cusimd`             | `CuSimd<T, N>` vector register type, `Float2`/`Float4`/`TmemRegs*` aliases   | All     |
| `clc`                | Cluster Launch Control: cancel, query first ctaid                            | sm_100+ |

- *cooperative_groups* also features support for warp/block reductions and scans

### The remaining public modules

The table above covers the most-used modules, but it is not the whole crate.
The remaining public modules are below. There is no GPU column here because
the minimum architecture varies per function, not per module. `int`, for
example, has plain scalar min/max that runs anywhere, plus `min.relu.s32`
which needs `sm_90+`. Each function's rustdoc states its own requirement.

| Module       | Description                                                                     |
|--------------|---------------------------------------------------------------------------------|
| `access`     | Choosing a tile shape from a target memory transaction width.                   |
| `async_copy` | Classic global-to-shared asynchronous copy intrinsics                           |
| `bf16`       | Scalar `bf16` min/max; each value is the `u16` bit pattern of one bfloat16       |
| `bf16x2`     | Packed `bf16x2` arithmetic; two values per `u32`, the first in the low 16 bits   |
| `constant`   | `ConstantMemory<T>` -- module-scope statics in PTX `.const` (address space 4)    |
| `convert`    | PTX type-conversion instructions                                                |
| `dotprod`    | Integer dot products: `dp4a` (packed bytes) and `dp2a` (packed halves)           |
| `f16`        | Scalar `f16` min/max; each value is the `u16` bit pattern of one IEEE half       |
| `f16x2`      | Packed `f16x2` arithmetic; two values per `u32`, the first in the low 16 bits    |
| `f32x2`      | Native SM100 two-lane `f32` arithmetic, each pair carried in a `u64`            |
| `float`      | Scalar floating-point intrinsics                                                |
| `i16x2`      | Packed 16-bit integer min/max; two values per `u32`, the first in the low bits   |
| `iket`       | Semantic annotations for In-Kernel Event Tracing; compiler markers behind macros |
| `int`        | Scalar integer intrinsics, including the `sm_90+` `min.relu.s32`/`max.relu.s32`  |
| `mma_frag`   | Fragment index algebra for the `m16n8k16` tensor-core accumulator                |
| `prmt`       | Byte permutation intrinsics                                                     |
| `ptx`        | Inline PTX support; user code goes through `ptx_asm!`, not this module directly  |
| `swizzle`    | XOR swizzles for shared memory, and the bank-conflict arithmetic to pick one     |
| `uniform`    | Launch-uniform scalar witnesses for APIs needing a value identical in every thread|
| `vector`     | Over-aligned element types, and a checked way to view a flat slice as them       |
| `view`       | Checked-once views for 32-bit kernel indexing, backed by a launch contract       |
| `wmma`       | Warp-level matrix ops: `movmatrix`, `mma.sync`, and cooperative `ldmatrix` loads |

## Key Types

### `ThreadIndex<'kernel, IndexSpace>` and `DisjointSlice<T, IndexSpace>`

`ThreadIndex` is an opaque witness with no public constructor. The trusted index functions are:

- `thread::index_1d()` -- unique only for a fully 1-D launch: grid and block
  Y/Z dimensions must all be `1`.
- `thread::index_2d::<S>()` -- const-stride 2D index. The witness type carries `S`, so a `DisjointSlice<T, Index2D<S>>` rejects mismatched strides at compile time.
- `thread::index_2d_runtime(&slice)` -- runtime row strides, taken from the slice the index will address, where the host bound the row width once for the launch.

The witness is `!Send + !Sync + !Copy + !Clone` and `'kernel`-scoped, so threads can't launder it through shared memory and it can't outlive the kernel body. `DisjointSlice<T, IndexSpace>` accepts only a `ThreadIndex` whose `IndexSpace` matches its own.

```rust
#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    if let Some((c_elem, idx)) = c.get_mut_indexed() {   // mints + resolves in one call
        let i = idx.get();
        *c_elem = a[i] + b[i];
    }
}
```

The explicit two-step form `let idx = thread::index_1d(); c.get_mut(idx)` is
also available when you need the index for arithmetic against multiple slices.
A `#[launch_contract(domain = 1, ...)]` moves the 1-D geometry proof into the
safe prepared host launch; raw launch geometry is unsafe. For non-trivial
patterns (reductions, histograms), `get_unchecked_mut(usize)` is the unsafe
element-access escape hatch.

### Compile-time kernel policies

The `config` module provides names for shapes, layouts, memory spaces, tiles,
and operations. These are zero-sized descriptions: they do not allocate or
access memory. A kernel library can place them and tuning constants on a policy
trait, then specialize one generic kernel for each policy type. The policy is
not passed to the GPU at runtime.

```rust
#[kernel]
#[launch_bounds(P::MAX_THREADS, P::MIN_BLOCKS)]
#[launch_contract(domain = 1)]
fn transform<P: TransformPolicy>(out: DisjointSlice<u32>) { /* ... */ }
```

`MAX_THREADS` is the largest accepted block, not an exact block size. A
prepared launch checks the concrete value for `P` before calling CUDA.

### `SharedArray<T, N, ALIGN>` and `DynamicSharedArray<T, ALIGN>`

Block-scoped shared memory. `SharedArray` is compile-time sized; `DynamicSharedArray` is runtime-sized (set via `LaunchConfig::shared_mem_bytes`). Both are `!Sync` -- concurrent access requires explicit GPU barriers.

```rust
#[kernel]
pub fn tiled(data: &[f32], mut out: DisjointSlice<f32>) {
    static mut TILE: SharedArray<f32, 256> = SharedArray::UNINIT;

    let tid = thread::threadIdx_x() as usize;
    unsafe { TILE[tid] = data[thread::index_1d().get()]; }
    thread::sync_threads();
    // ... read from TILE ...
}
```

Use `ALIGN = 128` for TMA destinations.

### `ManagedBarrier<State, Kind, ID>`

Typestate-based async barrier for TMA and MMA synchronization (Hopper+). Tracks initialization state (`Uninit` → `Ready` → `Invalidated`) and barrier kind (`TmaBarrier`, `MmaBarrier`, `GeneralBarrier`) at compile time.

### Atomics

Two kinds of atomics work on device:

- **`cuda_device::atomic::*`** -- 21 scoped GPU atomic types across three scopes (`Device`/`.gpu`, `Block`/`.cta`, `System`/`.sys`) and seven value types (u32, i32, u64, i64, f16, f32, f64). These give explicit control over scope and ordering.
- **`core::sync::atomic::*`** -- standard library atomics (`AtomicU32`, `AtomicBool`, etc.) also compile to GPU code. These always lower with **system scope** (`.sys`), not device scope, for safe host-device coherence -- matching CUDA C++ `cuda::atomic<T>` defaults. Reach for the `cuda_device::atomic` types above when you want `.gpu` or `.cta` instead.

Both paths emit the same NVVM atomic ops and share the full lowering pipeline to PTX.

```rust
use cuda_device::atomic::{DeviceAtomicU32, AtomicOrdering};

static COUNTER: DeviceAtomicU32 = DeviceAtomicU32::new(0);
// In kernel:
COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
```

### Tensor Cores and TMA

**TMA** (`tma` module): Hardware DMA via `TmaDescriptor`. Async bulk tensor copies in 1D-5D between global and shared memory, with multicast and CTA-group-2 variants.

**WGMMA** (`wgmma` module): Warpgroup MMA for Hopper -- fence/commit/wait pipeline, shared memory descriptors, bf16/f16/tf32 accumulate operations.

**tcgen05** (`tcgen05` module): Blackwell 5th-gen tensor cores with Tensor Memory (TMEM). `TmemGuard<State, N_COLS>` manages TMEM lifetime with typestate. Includes MMA operations, SMEM↔TMEM copies, stmatrix stores, descriptor builders, bf16 packing helpers, and CTA-pair (cg2) variants.

## Debug Facilities

```rust
use cuda_device::{gpu_printf, gpu_assert};

#[kernel]
pub fn debug_kernel(data: &[f32]) {
    let idx = thread::index_1d();
    let value = data[idx.get()];

    gpu_printf!(
        "thread %d: val = %f\n",
        idx.get() as i32,
        value as f64
    );
    gpu_assert!(value >= 0.0, "expected non-negative value");
}
```

`gpu_printf!` compiles to device-side `vprintf` with C vararg promotion.

`gpu_assert!(condition)` uses `trap()` when the condition is false. The
string-literal message form, `gpu_assert!(condition, "message")`, calls CUDA's
device-side `__assertfail` system call. The CUDA driver reports the assertion
message and call-site metadata, and synchronization returns
`CUDA_ERROR_ASSERT`.

The `debug` module also exposes GPU timing register reads such as `clock64()`
and `globaltimer()`.

## Proc-Macro Re-exports

These are defined in `cuda-macros` and re-exported from `cuda-device` for
convenience. The table is the whole `pub use cuda_macros::{...}` list in
`src/lib.rs`:

| Attribute                | Purpose                                                                                |
|--------------------------|----------------------------------------------------------------------------------------|
| `#[kernel]`              | Mark a function as a GPU kernel entry point                                            |
| `#[device]`              | Mark a helper function or extern block                                                 |
| `#[cuda_module]`         | Collect a module's kernels into a typed host module with `load` and launchers          |
| `#[launch_bounds]`       | Set max threads / min blocks per SM                                                    |
| `#[launch_contract]`     | Declare a kernel's launch requirements, unlocking a safe launch                        |
| `#[cluster_launch]`      | Set compile-time cluster dimensions                                                    |
| `#[cooperative_launch]`  | Launch as cooperative (for `grid::sync()`)                                             |
| `#[convergent]`          | Mark as convergent (barrier semantics)                                                 |
| `#[pure]`                | Mark as pure (no side effects)                                                         |
| `#[readonly]`            | Mark as read-only                                                                      |
| `#[constant]`            | Place a `ConstantMemory<T>` static in constant memory, plus a host `set_<name>` setter |
| `gpu_printf!`            | Device-side printf                                                                     |
| `ptx_asm!`               | Unsafe CUDA inline PTX                                                                 |

## Safety Model

1. **`ThreadIndex`** -- unconstructible except via trusted functions; guarantees unique indices
2. **`DisjointSlice::get_mut()`** -- bounds-checked `Option<&mut T>`; `get_unchecked_mut()` is the explicit `unsafe` escape
3. **`SharedArray` / `DynamicSharedArray`** -- `!Sync`; all access via `static mut` requires `unsafe`
4. **Barriers, TMA, WGMMA, tcgen05** -- all `unsafe` functions; caller ensures synchronization semantics
5. **Atomics** -- `unsafe impl Sync`; ordering semantics match CUDA scoped atomics

## Further Reading

- [cuda-host](../cuda-host/) -- host-side launch infrastructure
- [cuda-macros](../cuda-macros/) -- proc-macro implementations
- [cuda-core](../../../cuda-core/) -- CUDA driver API bindings
