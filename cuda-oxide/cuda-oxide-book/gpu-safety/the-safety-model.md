# Safe Kernels Using Rust's Type System

On a CPU, `rustc` prevents data races through single-thread ownership: one mutable reference, zero aliases. On a GPU, thousands of threads execute the same kernel function at once against the same output buffer, so single-thread borrow checking alone is not enough.

`cuda-oxide` closes that gap in **three layers**, so most kernels need zero `unsafe`:

| Layer       | Mechanism                                          | What It Covers                                               | `unsafe` Needed? |
|:------------|:---------------------------------------------------|:-------------------------------------------------------------|:-----------------|
| **Layer 1** | Rust types (`DisjointSlice` + `ThreadIndex`)       | One-thread-per-slot and tiled global writes (`PreparedLaunch`)| No               |
| **Layer 2** | `#[verify(...)]` (Lean 4 proof, preview)           | `SharedArray`, `&mut [T]`, barriers, TMA, clusters, `TMEM`   | No               |
| **Layer 3** | Scoped `unsafe` + `// SAFETY:` contract            | Raw pointer casts, custom warp masks, unverified intrinsics  | Yes, scoped      |

- **Start in Layer 1** for elementwise, stencil, and view-based kernels (requires only `rustc`).
- **Reach for Layer 2 (`#[verify]`, preview)** when threads cooperate through shared memory, plain `&mut [T]`, `mbarrier` rings, 2-CTA clusters, or Blackwell `tcgen05` / `TMEM`; see [Provably Safe Kernels (preview)](verified-kernels.md).
- **Drop to Layer 3 (`unsafe`)** only for operations outside both layers.

---

## Layer 1: Safe by Construction (`DisjointSlice` + `ThreadIndex`)

### How `DisjointSlice` prevents cross-thread write races

Layer 1 pairs two types so distinct threads never obtain `&mut T` to the same element:

- **`ThreadIndex<'kernel, IndexSpace>`**: An unforgeable witness around a `usize` with no public constructor. It is minted only from hardware registers (`threadIdx`, `blockIdx`, `blockDim`) and is `!Send + !Sync + !Copy + !Clone` with a `'kernel` lifetime, so threads cannot copy it or pass it through shared memory.
- **`DisjointSlice<T, IndexSpace>`**: A mutable output view whose `get_mut()` accepts only a matching `ThreadIndex` and returns `Option<&mut T>` (`None` when out of bounds).

```rust
use cuda_device::{DisjointSlice, kernel};

#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    if let Some((c_elem, idx)) = c.get_mut_indexed() {
        let i = idx.get();
        *c_elem = a[i] + b[i];
    }
}
```

When you need the index before calling `get_mut`, mint it explicitly:

```rust
#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(c_elem) = c.get_mut(idx) {
        *c_elem = a[i] + b[i];
    }
}
```

Why this is race-free end to end:

1. **Unique coordinates**: `index_1d()` is unique when grid and block Y/Z dimensions are `1` (enforced on the host by a `domain = 1` `PreparedLaunch`; raw launches with active Y/Z fail closed on device and return `None`).
2. **Bounds-checked**: Out-of-range threads get `None` from `get_mut()`.
3. **Layout-branded**: `DisjointSlice<T, Index2D<128>>` rejects `ThreadIndex<'_, Index2D<256>>` at compile time.
4. **Non-transferable**: `!Send + !Sync + !Copy + !Clone` keeps each witness private to the thread that minted it.

```text
1D PreparedLaunch (Y = Z = 1)  ->  unique index_1d per thread  ->  Some(&mut c[i])
Raw 2D launch (active Y or Z)  ->  invalid witness             ->  None (safe fallback)
```

### Trusted index functions

| Function                   | Formula                                 | Return Type                                    | Role                            |
|:---------------------------|:----------------------------------------|:-----------------------------------------------|:--------------------------------|
| `index_1d()`               | `blockIdx.x * blockDim.x + threadIdx.x` | `ThreadIndex<'kernel, Index1D>`                | 1D witness (`Y = Z = 1`)        |
| `index_2d::<S>()`          | `row * S + col`                         | `Option<ThreadIndex<'kernel, Index2D<S>>>`     | Compile-time 2D stride witness  |
| `index_2d_runtime(&slice)` | `row * width + col`                     | `Option<ThreadIndex<'kernel, Runtime2DIndex>>` | Host-bound runtime stride       |
| `index_2d_row()`           | `blockIdx.y * blockDim.y + threadIdx.y` | `usize`                                        | Row coordinate for math         |
| `index_2d_col()`           | `blockIdx.x * blockDim.x + threadIdx.x` | `usize`                                        | Column coordinate for math      |

`index_2d_row()` and `index_2d_col()` return plain `usize` values for arithmetic; only the linearized constructors mint a `ThreadIndex`.

(check-a-tile-once)=
### Check a tile once

When each thread owns a small fixed tile, `LinearTiles<N>` checks the entire tile in one comparison and serves constant offsets (`at_const::<0>()`) with zero extra checks:

```rust
use cuda_device::{DisjointSlice, LinearTiles, kernel, launch_contract, thread};

#[kernel(launch_context = launch_context)]
#[launch_contract(
    domain = 1,
    coordinates = u32,
    block = (64, 1, 1),
)]
pub fn add_one(mut values: DisjointSlice<u32, LinearTiles<4>>) {
    let thread_index = thread::index_1d_u32(launch_context);

    if let Some(mut tile) = values.tile_thread32(thread_index) {
        let first = tile.at_const::<0>().read();
        tile.at_const::<0>().write(first + 1);

        let second = tile.at_const::<1>().read();
        tile.at_const::<1>().write(second + 1);
    }
}
```

How the two attributes cooperate:

- **`launch_context`**: Zero-sized token injected into the kernel body carrying the launch contract's static facts.
- **`domain = 1, coordinates = u32`**: Unlocks `thread::index_1d_u32(launch_context)` without 64-bit index math.
- **`block = (64, 1, 1)`**: Checked on the host by `prepare_*` *and* emitted into PTX as `.reqntid 64, 1, 1` so the CUDA driver rejects any mismatched raw launch too.
- **2D tiles**: Pair `thread::coord_2d_u32(launch_context)` with `RowMajorTiles<ROWS, COLS, ROW_STRIDE>`; see [`proof_carrying_views`](https://github.com/NVIDIA/cuda-rust/tree/main/cuda-oxide/crates/rustc-codegen-cuda/examples/proof_carrying_views).

### 2D strides and launch-uniform scalars

- **Compile-time stride (`Index2D<S>`)**: Prefer this whenever row width `S` is a constant. Mixing strides fails to type-check:

```rust
#[kernel]
pub fn rejected(mut out: DisjointSlice<u32, Index2D<128>>) {
    if let Some(idx) = thread::index_2d::<256>() { // compile error: 256 != 128
        if let Some(slot) = out.get_mut(idx) { *slot = 1; }
    }
}
```

- **Runtime stride (`Runtime2DIndex` and `RowWidth`)**: When matrix width `n` is only known at launch, two threads must never disagree on the stride (thread `(1, 0)` at stride `5` and thread `(0, 5)` at stride `100` both resolve flat slot `5`). Bind the width once on the host so every thread reads the same value from the slice itself:

```rust
// Host: bind row width `n` into `c` once at launch
module.write_c(&stream, &prepared, cuda_host::RowWidth::new(&mut c, n))?;
```

```rust
#[kernel(launch_context = lc)]
#[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1))]
pub fn write_c(mut c: DisjointSlice<f32, RuntimeRowMajorTiles<1, 1>>) {
    let coord = thread::coord_2d_u32(lc);
    if let Some(mut cell) = c.tile_2d32_rt(coord) {
        cell.at_const::<0, 0>().write(1.0);
    }
}
```

- **Launch-uniform scalars (`Uniform<T>`)**: Wrap scalar parameters as `Uniform<u32>` when every thread must observe the same launch-constant value (`rows.get()`). Host callers still pass a plain `u32`, and device code cannot forge a `Uniform<T>` from thread-varying data.

For hoisting *input* slice bounds checks on `a[...]` and `b[...]` out of inner loops, see [Bounds Checks: Pay Once or Opt Out](bounds-checks.md).

---

## Layer 2: Cooperative & Async Kernels (`#[verify]`, preview)

Layer 1 handles independent per-thread writes, while cooperative GPU kernels share `static mut SharedArray` tiles, stage DMA transfers with `Barrier`, or pass plain `&mut [T]` buffers.

Adding `#[verify(...)]` next to `#[launch_contract(...)]` hands the kernel's `PostMem2Reg` MIR to **Lean 4** at compile time:

```rust
#[kernel]
#[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1))]
#[verify(no_oob_idx(TILE_A, TILE_B), no_data_race(TILE_A, TILE_B), uniform_barriers)]
pub fn sgemm_tiled(a: &[f32], b: &[f32], mut c: DisjointSlice<f32, RuntimeRowMajorTiles<1, 1>>) {
    static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut TILE_B: SharedArray<f32, 256> = SharedArray::UNINIT;
    // Direct TILE_A[...] and TILE_B[...] reads/writes with 0 `unsafe` blocks!
}
```

What `#[verify(...)]` unlocks in safe Rust (zero `unsafe`, zero runtime check overhead):

- **Shared-memory tiling & reductions**: Proves `SharedArray` bounds (`no_oob_idx`), cross-thread race freedom (`no_data_race`), and non-divergent `sync_threads()` (`uniform_barriers`).
- **Plain `&mut [T]` parameters**: Unverified `&mut [T]` kernel parameters are rejected at compile time; `#[verify(no_oob_idx(c), disjoint_writes(c))]` proves cross-thread exclusivity and unlocks `&mut [T]`.
- **Async TMA, warp rings, 2-CTA clusters, and Blackwell `TMEM`**: Proves `mbarrier` token ordering, `SharedArray<Barrier, N>` ring back-pressure, 2-CTA `cluster_sync()` + `map_shared_rank` rank bounds, `CLC` responses, and `tcgen05` + `TMEM` lifecycles (`tmem_lifecycle`).

See [Provably Safe Kernels (preview)](verified-kernels.md) for the full guide and nine runnable examples from `safe_vecadd` to `2.118 PFLOP/s` Blackwell GEMM.

---

## Layer 3: Scoped `unsafe` (When You Need an Escape Hatch)

When you write a kernel *without* `#[verify]` (or use a low-level feature outside `#[verify]`'s scope), `unsafe` blocks stay small, explicit, and local:

| Pattern                         | Why It Needs `unsafe` (Without `#[verify]`)          | Safe Alternative / Contract                              |
|:--------------------------------|:-----------------------------------------------------|:---------------------------------------------------------|
| Unverified `SharedArray<T, N>`  | `rustc` cannot check 256 threads on `static mut`     | Add `#[verify(no_oob_idx, no_data_race, uniform_barriers)]` |
| `DynamicSharedArray<T>`         | Size comes from runtime `shared_mem_bytes`           | Match `LaunchConfig` byte size and synchronize phases    |
| Custom-mask warp shuffles       | Hardware requires lanes in `mask` to be converged    | Prefer full-warp helpers (`u32::MAX`); document masks    |
| Atomics from raw pointers       | `DeviceAtomicU32::from_ptr` trusts pointer validity  | Ensure pointer is live, aligned, and non-aliasing        |
| Non-uniform global slot writes  | Leader-only `out.get_unchecked_mut(warp_idx)`        | Document uniqueness or prove with `#[verify]`            |
| Unverified Hopper/Blackwell ops | Manual proxy fences, `wgmma`, or raw global stores   | Follow PTX ISA ordering or verify TMA/CLC/`tcgen05` in Layer 2 |

### Atomics from raw pointers

Calling `.fetch_add(...)` on a `DeviceAtomicU32` is safe; only constructing one from a raw pointer is `unsafe`:

```rust
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};

// SAFETY: `ptr` points to a valid, 4-byte aligned `u32` in global memory.
let atom = unsafe { DeviceAtomicU32::from_ptr(ptr) };
atom.fetch_add(1, AtomicOrdering::Relaxed);
```

### Writing a good `// SAFETY:` comment

Every `unsafe` block should state **why the invariant holds**, not repeat what the code does:

```rust
// SAFETY: Only lane 0 of each warp enters this branch, and `gid / 32` is
// unique per warp in a 1D launch, so no two threads alias `warp_idx`.
if warp::lane_id() == 0 {
    let warp_idx = gid.get() / 32;
    unsafe { *partial_sums.get_unchecked_mut(warp_idx) = warp_sum; }
}
```

---

## Summary: What Is Enforced vs. Manual Today

| Safety Property                                        | Status in `cuda-oxide`                                         |
|:-------------------------------------------------------|:---------------------------------------------------------------|
| Ownership, lifetimes, and host non-aliasing            | Enforced (`rustc` borrow checker)                              |
| 1D / 2D disjoint global writes (`DisjointSlice`)       | Enforced (`ThreadIndex` + `PreparedLaunch` / `RowWidth`)       |
| Plain `&mut [T]` kernel parameters                     | Enforced: rejected unless proved by `#[verify(disjoint_writes)]`|
| `SharedArray` bounds, phase races, and `sync_threads`  | Enforced when annotated with `#[verify(...)]`                  |
| Async TMA, `mbarrier` rings, 2-CTA DSMEM, `CLC`, `TMEM`| Enforced when annotated with `#[verify(...)]`                  |
| Optimizer barrier safety (`convergent`)                | Enforced automatically on barriers, fences, and shuffles       |
| Arbitrary partial warp-shuffle `mask` convergence      | Manual `unsafe` obligation                                     |
| Raw pointer arithmetic and `DynamicSharedArray` bounds | Manual `unsafe` obligation                                     |
| Floating-point rounding and value-level math formulas  | Outside safety scope (test against a reference oracle)         |
