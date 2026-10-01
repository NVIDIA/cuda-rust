# API Quick Reference

This appendix is a condensed reference for the cuda-oxide device and host APIs.
For full documentation, run `cargo doc --no-deps --open` from the workspace
root.

---

## Attributes and Macros

### Kernel and Device Attributes

```rust
use cuda_device::{cluster_launch, cooperative_launch, device, kernel, launch_bounds};

#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) { /* ... */ }

#[kernel]
pub fn unrolled(mut data: DisjointSlice<f32>) {
    let mut i = 0;
    #[unroll(4)]
    while i < 16 { /* ... */ i += 1; }
}

#[kernel]
#[launch_bounds(256, 2)]
pub fn tuned_kernel(data: &mut [f32]) { /* ... */ }

#[kernel]
#[cluster_launch(4, 1, 1)]
pub fn cluster_kernel(data: &mut [f32]) { /* ... */ }

#[kernel]
#[cooperative_launch]
pub fn grid_sync_kernel(data: &mut [f32]) { /* ... */ }

#[device]
fn helper(x: f32) -> f32 { x * x }
```

| Attribute                                   | Purpose                                                             |
|:--------------------------------------------|:--------------------------------------------------------------------|
| `#[cuda_module]`                            | Collect a module's kernels into a typed host module with a `load` and launchers |
| `#[kernel]`                                 | Mark a function as a GPU kernel entry point (`.entry` in PTX)       |
| `#[device]`                                 | Mark a helper function or `extern "C"` block for device compilation |
| `#[unroll]` / `#[unroll(N)]`               | Request full unrolling, or unrolling by a factor `N >= 2`            |
| `#[launch_bounds(max_threads, min_blocks)]` | Occupancy hints for register allocation                             |
| `#[constant]`                               | Place a `ConstantMemory<T>` static in constant memory, with a host `set_<name>` |
| `#[launch_contract(...)]`                   | Declare the launch shape a kernel requires, unlocking a safe (non-`unsafe`) launch |
| `#[cluster_launch(x, y, z)]`                | Set compile-time cluster dimensions (Hopper+)                       |
| `#[cooperative_launch]`                     | Launch cooperatively via `#[cuda_module]` (enables `grid::sync()`)  |
| `#[convergent]`                             | Mark as convergent (barrier semantics)                              |
| `#[pure]`                                   | Mark as side-effect free                                            |
| `#[readonly]`                               | Mark as read-only                                                   |

Use these annotations only on an explicit counted `while` loop inside a
`#[kernel]` or `#[device]` function. Range-based `for` loops are not yet
recognized by the unroll pass. Nested loops and multiple `continue` paths are
supported. Full `#[unroll]` preserves `break` paths and multiple exit targets.

Partial `#[unroll(N)]` requires a positive step, a `<` or `<=` test, an
unchanging limit, and no exit besides the normal header test. Other requests
warn and are not unrolled.

One annotation may create at most 1,024 body copies, 8,192 cloned basic blocks,
and 65,536 cloned operations. Larger requests warn and are not unrolled.

### Debug and PTX Macros

```rust
use cuda_device::{gpu_assert, gpu_printf, ptx_asm};

gpu_printf!("thread %d: val = %f\n", idx as i32, val as f64);
gpu_assert!(val.is_finite());
gpu_assert!(val >= 0.0, "expected non-negative value");

let y: u32;
unsafe {
    ptx_asm!("add.u32 %0, %1, %1;", out("=r") y, in("r") x, options(register_only));
}
```

| Macro                               | Purpose                                                           |
|:------------------------------------|:------------------------------------------------------------------|
| `gpu_printf!(fmt, args...)`         | Device-side formatted output (lowers to `vprintf`)                |
| `gpu_assert!(condition)`            | Runtime assertion; calls `trap()` on failure                      |
| `gpu_assert!(condition, "message")` | Runtime assertion with CUDA diagnostic; message must be a literal |
| `ptx_asm!(...)`                     | Unsafe CUDA inline PTX                                            |

The message form lowers to CUDA's device-side `__assertfail` system call.
The driver reports the message and call-site metadata, and synchronization
returns `CUDA_ERROR_ASSERT`.

---

## Compile-time policy configuration

```rust
use cuda_device::config::{
    Atom, AtomKind, AtomSpec, Block, Cluster, ColumnMajor, Global, Layout, MemorySpace, Policy,
    PolicyId, Register, RowMajor, Scope, Shape, Shape1, Shape2, Shape3, Shared, TensorMemory,
    Thread, Tile, TileSpec, Warp, WarpGroup,
};
```

Policies describe compile-time kernel configurations using zero-sized Rust
types. A generic kernel is monomorphized once for every concrete policy type;
the policy is not passed as a runtime kernel argument.

```rust
trait VectorPolicy: Policy {
    type BlockTile: TileSpec;
    type ElementAtom: AtomSpec;

    const MAX_THREADS: u32;
    const MIN_BLOCKS: u32;
    const UNROLL: u32;
}

enum SmallTilePolicy {}

impl Policy for SmallTilePolicy {
    const ID: PolicyId =
        PolicyId::new(0x706f_6c69_6379_5f63, 1);
}
```

| API                  | Description                                                                             |
| :------------------- |:----------------------------------------------------------------------------------------|
| `Policy`             | Minimal base trait for a named compile-time kernel policy                               |
| `PolicyId`           | Explicit stable identity containing a project-specific namespace and policy-local value |
| `Shape`              | Trait exposing a static shape's rank, extents, and checked element count                |
| `Shape1<D0>`         | One-dimensional compile-time shape                                                      |
| `Shape2<D0, D1>`     | Two-dimensional compile-time shape                                                      |
| `Shape3<D0, D1, D2>` | Three-dimensional compile-time shape                                                    |
| `Tile<S, L, M, Q>`   | Metadata-only description combining shape, layout, memory space, and execution scope    |
| `TileSpec`           | Type-level access to the components of a tile description                               |
| `Atom<K, S, Q>`      | Metadata-only description of an operation, logical footprint, and participating threads |
| `AtomKind`           | Open marker trait identifying a domain-specific operation                               |
| `AtomSpec`           | Type-level access to an atom's operation kind, shape, and scope                         |
| `Layout`             | Open trait for memory-order metadata                                                    |
| `RowMajor`           | Layout whose rightmost coordinate is contiguous                                         |
| `ColumnMajor`        | Layout whose leftmost coordinate is contiguous                                          |
| `MemorySpace`        | Open trait describing a CUDA storage location                                           |
| `Global`             | Device global-memory marker                                                             |
| `Shared`             | Per-block shared-memory marker                                                          |
| `Register`           | Thread-local register-storage marker                                                    |
| `TensorMemory`       | Hardware tensor-memory marker                                                           |
| `Scope`              | Open trait describing the threads cooperating on an operation                           |
| `Thread`             | Single-thread execution scope                                                           |
| `Warp`               | Single-warp execution scope                                                             |
| `WarpGroup`          | Hardware warpgroup execution scope                                                      |
| `Block`              | Thread-block execution scope                                                            |
| `Cluster`            | Thread-block-cluster execution scope                                                    |

`Tile` and `Atom` are descriptions only. They do not allocate storage, provide
pointer access, emit GPU instructions, synchronize threads, or establish a
safety property. A domain-specific policy trait gives those descriptors
meaning and validates supported combinations.

`PolicyId` values are supplied explicitly by the policy library. They are not
derived from Rust `TypeId`, type names, compiler mangling, or hashes. Keep an ID
stable while the policy's generated behavior remains unchanged and allocate a
new value when that behavior changes.

Policy-associated constants can currently be used in compile-time
`launch_bounds` and partial-loop `unroll` expressions:

```rust
use cuda_device::{kernel, launch_bounds};

#[kernel]
#[launch_bounds(P::MAX_THREADS, P::MIN_BLOCKS)]
pub unsafe fn transform<P: VectorPolicy>(
    input: *const u32,
    output: *mut u32,
    count: u32,
) {
    let mut lane = 0;

    #[unroll(P::UNROLL)]
    while lane < count {
        // ...
        lane += 1;
    }
}
```

Generic policy expressions require `#![feature(generic_const_exprs)]`.
`launch_contract` fields, cluster dimensions, and dynamic shared-memory sizes
currently remain literal.

See the
[`policy_config`](https://github.com/NVlabs/cuda-oxide/tree/main/crates/rustc-codegen-cuda/examples/policy_config)
example for two concrete policies that generate independent PTX
specializations and policy-specific prepared launches.

---

## Thread Identification

```rust
use cuda_device::thread;

let idx     = thread::index_1d();                            // ThreadIndex<'_, Index1D>
let idx2d   = thread::index_2d::<128>();                     // Option<ThreadIndex<'_, Index2D<128>>>
let idx2d_r = thread::index_2d_runtime(&out);                // Option<ThreadIndex<'_, Runtime2DIndex>>
//            `out` is the DisjointSlice<T, Runtime2DIndex> the index will address;
//            its row width was bound once by the host via cuda_host::RowWidth.
let idx32   = thread::index_1d_u32(launch_context);          // ThreadIndex32<'_>
let pos32   = thread::coord_2d_u32(launch_context);          // ThreadCoord2D32<'_>

let tid_x  = thread::threadIdx_x();    // u32
let bid_x  = thread::blockIdx_x();     // u32
let bdim_x = thread::blockDim_x();     // u32
```

| Function                                    | Returns                                          | Description                                                |
|:--------------------------------------------|:-------------------------------------------------|:-----------------------------------------------------------|
| `thread::index_1d()`                        | `ThreadIndex<'_, Index1D>`                       | Unique linear index (1D grids)                             |
| `thread::index_2d::<S>()`                   | `Option<ThreadIndex<'_, Index2D<S>>>`            | Const-stride 2D index; mismatched strides are a type error |
| `thread::index_2d_runtime(&slice)`          | `Option<ThreadIndex<'_, Runtime2DIndex>>`        | Runtime-width 2D index; row width read from the slice      |
| `thread::index_1d_u32(launch_context)`      | `ThreadIndex32<'_>`                              | 1-D index as `u32`; requires checked `u32` coordinates     |
| `thread::coord_2d_u32(launch_context)`      | `ThreadCoord2D32<'_>`                            | 2-D row/column as `u32`; requires checked `u32` coordinates|
| `thread::index_2d_row()`                    | `usize`                                          | 2D row index                                               |
| `thread::index_2d_col()`                    | `usize`                                          | 2D column index                                            |
| `thread::threadIdx_{x,y,z}()`               | `u32`                                            | Thread index within block                                  |
| `thread::blockIdx_{x,y,z}()`                | `u32`                                            | Block index within grid                                    |
| `thread::blockDim_{x,y,z}()`                | `u32`                                            | Block dimensions                                           |

`thread::index_2d::<S>()` and `thread::index_2d_runtime(&slice)` return
`None` when the computed column exceeds the stride, which skips the
right-edge tail in non-aligned 2D kernels.

`index_2d::<S>` is the safe const-stride form; the const generic encodes the
stride in the witness type so threads cannot use different strides.
`index_1d` requires inactive Y/Z dimensions, and 2D indices require inactive
Z. A matching `PreparedLaunch<K>` proves this without device checks. Otherwise,
the device rejects the wrong rank: `index_1d` creates an invalid witness and
2D helpers return `None`. A raw launch remains unsafe because its other memory
and launch obligations are unchecked. `index_2d_runtime` covers launches whose
stride is only known at runtime: the row width travels inside the slice, written
once by the host into the launch packet (`cuda_host::RowWidth`), so there is no
`unsafe` and no per-call stride for threads to disagree about. The witness
stores the thread's `(row, col)` coordinates and the addressed slice resolves
them against its own row width. Full
discussion in [The Safety Model](../gpu-safety/the-safety-model.md).

---

## Safe Parallel Writes — DisjointSlice

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

| Method                  | Signature                                        | Description                                                          |
|:------------------------|:-------------------------------------------------|:---------------------------------------------------------------------|
| `get_mut_indexed`       | `() -> Option<(&mut T, ThreadIndex<'_, IS>)>`    | One-call form: mints the witness and resolves it. Index1D / Index2D. |
| `get_mut`               | `(ThreadIndex<'_, IS>) -> Option<&mut T>`        | Bounds-checked mutable access from an explicit witness               |
| `get_unchecked_mut`     | `(usize) -> &mut T`                              | Unsafe, unchecked access                                             |
| `len`                   | `() -> usize`                                    | Number of elements                                                   |

`get_mut_indexed` is gated on `IndexSpace: IndexFormula` (impl'd by
`Index1D` and `Index2D<S>`). For `Runtime2DIndex` slices, use the
explicit `thread::index_2d_runtime(&slice)` + `get_mut(idx)` pair; the
slice resolves the witness's coordinates against its own host-bound
row width (`slice.row_width()`).

For fixed-size tiles, use `DisjointSlice<T, LinearTiles<N>>::tile_thread32`
or `DisjointSlice<T, RowMajorTiles<R, C, S>>::tile_2d32`. Each method checks a
complete tile once, then `at_const` accesses known positions without another
runtime bounds check. `S` is the caller-declared logical row width and must
match the buffer layout. See {ref}`Check a tile once <check-a-tile-once>`.

---

## Shared Memory

```rust
use cuda_device::{DynamicSharedArray, SharedArray, thread};

#[kernel]
pub fn tiled(data: &[f32], mut out: DisjointSlice<f32>) {
    static mut TILE: SharedArray<f32, 256> = SharedArray::UNINIT;
    let tid = thread::threadIdx_x() as usize;
    unsafe { TILE[tid] = data[thread::index_1d().get()]; }
    thread::sync_threads();
    // ... read from TILE ...
}

#[kernel]
pub fn dynamic(data: &[f32]) {
    static mut BUF: DynamicSharedArray<f32> = DynamicSharedArray::UNINIT;
    // Size set at launch via LaunchConfig::shared_mem_bytes
}
```

| Type                      | Description                                               |
|:--------------------------|:----------------------------------------------------------|
| `SharedArray<T, N>`       | Compile-time sized, block-scoped shared memory            |
| `SharedArray<T, N, 128>`  | With 128-byte alignment (required for TMA destinations)   |
| `DynamicSharedArray<T>`   | Runtime-sized shared memory (set via `LaunchConfig`)      |
| `SharedPtr<T>`            | Pointer into this block's shared memory, `addrspace(3)` by type |

Both are `!Sync` — concurrent access requires explicit barriers.

---

## Synchronization

### Block-Level

```rust
thread::sync_threads();   // __syncthreads() equivalent
```

### Managed Barriers (Hopper+)

```rust
use cuda_device::{ManagedBarrier, Ready, TmaBarrierHandle, Uninit};

// Typestate lifecycle: Uninit → Ready → Invalidated
let bar: TmaBarrierHandle<Uninit> = TmaBarrierHandle::from_static(ptr);
let bar: TmaBarrierHandle<Ready> = unsafe { bar.init(thread_count) };
let token = bar.arrive();
bar.wait(token);
unsafe { bar.inval() };
```

| Operation                   | Description                                      |
|:----------------------------|:-------------------------------------------------|
| `.init(count)`              | Initialize barrier with expected arrival count   |
| `.arrive()`                 | Signal arrival, returns `BarrierToken`           |
| `.arrive_expect_tx(bytes)`  | Arrive and set expected TX byte count (for TMA)  |
| `.wait(token)`              | Block until all arrivals + TX complete           |
| `.inval()`                  | Invalidate barrier (cleanup)                     |

---

## Warp Primitives

```rust
use cuda_device::warp;

let lane = warp::lane_id();      // 0–31
let wid  = warp::warp_id();

// Shuffle
let partner = warp::shuffle_xor_f32(val, mask);
let from_above = warp::shuffle_down_f32(val, delta);
let from_below = warp::shuffle_up_f32(val, delta);
let from_lane  = warp::shuffle_f32(val, src_lane);

// u32 is the unsuffixed form
let partner_u = warp::shuffle_xor(val_u32, mask);

// Vote
let all_true = warp::all(predicate);
let any_true = warp::any(predicate);
let mask     = warp::ballot(predicate);
let count    = warp::popc(predicate); // == ballot(predicate).count_ones()
```

### Shuffle Operations

The unsuffixed name takes `u32`; `_f32`, `_f64` and `_u64` are the other
widths. Each also has a `_sync` form taking an explicit member mask. There is
no `_i32` variant — reinterpret an `i32` and use the `u32` form, since a
shuffle moves bits and does not interpret them.

| Function                                    | Description                       |
|:--------------------------------------------|:----------------------------------|
| `shuffle_xor(val, mask)` + `_f32/_f64/_u64` | Exchange with lane `id ^ mask`    |
| `shuffle_down(val, delta)` + `_f32/_f64/_u64` | Read from lane `id + delta`     |
| `shuffle_up(val, delta)` + `_f32/_f64/_u64` | Read from lane `id - delta`       |
| `shuffle(val, src)` + `_f32/_f64/_u64`      | Read from specific lane           |

### Vote Operations

| Function       | Returns  | Description                                  |
|:---------------|:---------|:---------------------------------------------|
| `all(pred)`    | `bool`   | True if predicate holds for all lanes        |
| `any(pred)`    | `bool`   | True if predicate holds for any lane         |
| `ballot(pred)` | `u32`    | Bitmask of lanes where predicate is true     |
| `popc(pred)`   | `u32`    | Count of lanes where predicate is true       |

---

## Atomics

### Scoped GPU Atomics

```rust
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};

static COUNTER: DeviceAtomicU32 = DeviceAtomicU32::new(0);

// In kernel:
COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
let old = COUNTER.load(AtomicOrdering::Acquire);
```

| Scope                                        | Types                         |
|:---------------------------------------------|:------------------------------|
| `DeviceAtomic{U32,I32,U64,I64,F16,F32,F64}`  | `.gpu` scope                  |
| `BlockAtomic{U32,I32,U64,I64,F16,F32,F64}`   | `.cta` scope                  |
| `SystemAtomic{U32,I32,U64,I64,F16,F32,F64}`  | `.sys` scope (CPU-GPU shared) |

Twenty-one types: seven value widths in each of the three scopes. The `F16`
variants take the same surface as `F32`/`F64` -- `load`, `store`, `fetch_add`,
`fetch_sub`, `swap` -- with `fetch_add`/`fetch_sub` lowering to hardware
`atom.add.noftz.f16`.

`core::sync::atomic` types (`AtomicU32`, `AtomicBool`, etc.) also compile to
GPU code, defaulting to system scope.

On nvptx64, `core::sync::atomic::AtomicPtr<T>` supports `load`, `store`,
`swap`, and `compare_exchange` for generic-address-space pointer values while
preserving pointer types through lowering.
Local atomic storage uses ordinary thread-private accesses. Global and shared
storage retain atomic ordering and scope, including when a helper receives a
generic pointer. Legacy NVVM uses scoped PTX for pointer exchange and
compare-exchange while retaining pointer-typed operands and results.

---

## TMA — Tensor Memory Accelerator (Hopper+)

```rust
use cuda_device::tma::TmaDescriptor;
use cuda_device::tma::{cp_async_bulk_commit_group, cp_async_bulk_tensor_2d_g2s};

// Host: build descriptor (128 bytes, opaque)
// Device: issue async bulk copy
cp_async_bulk_tensor_2d_g2s(smem_ptr, &desc, coord_x, coord_y, barrier_ptr);
cp_async_bulk_commit_group();
```

| Function                                      | Description                          |
|:----------------------------------------------|:-------------------------------------|
| `cp_async_bulk_tensor_{1..5}d_g2s(...)`       | Global → shared async bulk copy      |
| `cp_async_bulk_tensor_{1..5}d_s2g(...)`       | Shared → global async bulk copy      |
| `cp_async_bulk_tensor_2d_g2s_multicast(...)`  | Multicast to all CTAs in cluster     |
| `cp_async_bulk_commit_group()`                | Commit outstanding copies            |
| `cp_async_bulk_wait_group(n)`                 | Wait until ≤ n groups remain         |

---

## Cluster Programming (Hopper+)

```rust
use cuda_device::cluster;

let rank = cluster::block_rank();        // This block's rank in the cluster
let size = cluster::cluster_size();      // Number of blocks in cluster
cluster::cluster_sync();                 // Barrier across all cluster blocks

// Distributed Shared Memory. Both are `unsafe fn` and both take the *local*
// pointer plus the target rank; they do the rank mapping themselves, so
// never pass a pointer already returned by `map_shared_rank` back in.

// `map_shared_rank` returns a real pointer into the target block's shared
// memory (cluster-shared address space). Plain reads and writes through it
// compile to ld.shared::cluster / st.shared::cluster and just work on sm_90+.
let remote_ptr = unsafe { cluster::map_shared_rank(local_ptr, target_rank) };
let val = unsafe { *remote_ptr };  // ld.shared::cluster

// Fixed-width alternative: map and read one u32 in a single call.
let val = unsafe { cluster::dsmem_read_u32(local_u32_ptr, target_rank) };
```

---

## Tensor Cores — WGMMA (Hopper, SM 90)

```rust
use cuda_device::wgmma;

wgmma::wgmma_fence();
wgmma::wgmma_commit_group();
wgmma::wgmma_wait_group::<0>();
```

Warpgroup MMA: 4 warps (128 threads) issue matrix multiply-accumulate from
shared memory. Operands described by SMEM descriptors; accumulator in registers.

---

## Tensor Cores — tcgen05 (Blackwell, SM 100+)

```rust
use cuda_device::SharedArray;
use cuda_device::tcgen05::{TmemGuard, TmemReady, TmemUninit};

static mut TMEM_SLOT: SharedArray<u32, 1, 4> = SharedArray::UNINIT;

let guard = TmemGuard::<TmemUninit, 512>::from_static(&raw mut TMEM_SLOT as *mut u32);
let guard = unsafe { guard.alloc() };   // TmemUninit → TmemReady
// ... issue MMA, read results via guard.address() ...
let _guard = unsafe { guard.dealloc() }; // TmemReady → TmemDeallocated
```

Single-thread MMA issue into dedicated Tensor Memory (TMEM). `TmemGuard`
manages TMEM lifetime with typestate: `TmemUninit → TmemReady → TmemDeallocated`.
N_COLS must be a power of 2 in the range [32, 512].

---

## Host-Side: Kernel Launch

### Typed Synchronous

```rust
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};

let ctx = CudaContext::new(0).unwrap();
let stream = ctx.default_stream();
let module = kernels::load(&ctx).unwrap();

let a = DeviceBuffer::from_host(&stream, &a_host).unwrap();
let b = DeviceBuffer::from_host(&stream, &b_host).unwrap();
let mut output = DeviceBuffer::<f32>::zeroed(&stream, n).unwrap();

// SAFETY: this is a 1D launch and all buffers contain n elements.
unsafe {
    module.vecadd(&stream, LaunchConfig::for_num_elems(n), &a, &b, &mut output)
}
.unwrap();
```

### Typed Async

```rust
use cuda_async::simt::device_operation::DeviceOperation;

let module = kernels::load_async(0)?;
// SAFETY: this is 1D, buffers contain n elements, and module/scheduler share a context.
let op = unsafe {
    module.vecadd_async(LaunchConfig::for_num_elems(n), &a, &b, &mut output)
}?;

op.sync()?;       // blocking
// or: op.await?;  // async with tokio
```

Raw generated calls are unsafe because `LaunchConfig` is not tied to a kernel.
A kernel with `#[launch_contract]` instead uses `LaunchConfig1D/2D/3D` to create
a checked `PreparedLaunch<K>`, then launches safely. `cuda_launch!` and
`cuda_launch_async!` remain unsafe lower-level APIs for explicit module loading
and custom launch code.

### LaunchConfig

| Method                                                   | Description                                  |
|:---------------------------------------------------------|:---------------------------------------------|
| `LaunchConfig::for_num_elems(n)`                         | Auto-configure grid/block for `n` elements   |
| `LaunchConfig { grid_dim, block_dim, shared_mem_bytes }` | Direct struct construction                   |

---

## Host-Side: Virtual Memory Management

### VMM lifecycle

| API                                         | Purpose                                     |
|:--------------------------------------------|:--------------------------------------------|
| `vmm::allocation_granularity(device)`       | Query the required allocation granularity   |
| `vmm::align_size(size, granularity)`        | Round a size to the required granularity    |
| `PhysicalAllocation::new(device, size)`     | Allocate physical memory                    |
| `VirtualReservation::new(size, alignment)`  | Reserve a virtual address range             |
| `Mapping::new(va, size, &physical, offset)` | Map physical memory into a VA range         |
| `vmm::set_access(va, size, devices)`        | Grant read/write access to selected devices |

Mappings must be dropped before their virtual reservations and physical
allocations.

---

## Host-Side: Peer Access

| API                                   | Purpose                                           |
|:--------------------------------------|:--------------------------------------------------|
| `peer::can_access_peer(from, to)`     | Query whether the topology supports direct access |
| `peer::enable_peer_access(from, to)`  | Enable one-directional peer access                |
| `peer::disable_peer_access(from, to)` | Disable one-directional peer access               |

Peer access is directional. Enable both directions when both devices must
initiate accesses.

---

## Host-Side: Kernel Families

| Type | Purpose |
|:-----|:--------|
| `KernelFamily<Id, Entry, Meta, N>` | Fixed set of ahead-of-time compiled variants |
| `KernelVariant<Id, Entry, Meta>` | Stable ID, callable entry, and policy metadata |
| `KernelProblem<Variant>` | Validates whether a variant is eligible |
| `KernelSelector<Problem, Variant, Id>` | Chooses among already eligible variants |
| `KernelSelectionCache<Problem, Id>` | Stores stable selection IDs |
| `NoKernelSelectionCache` | Disables caching |
| `SelectionMode::Auto` | Uses validated cache results or invokes the selector |
| `SelectionMode::Force(id)` | Bypasses cache and selector but still validates eligibility |
| `SelectedVariant` | Returns the selected variant and its provenance |
| `SelectionSource` | Reports override, cache, or selector provenance |

`KernelFamily::try_new` rejects empty families, blank family names, and
duplicate variant IDs. The family name and revision form the cache namespace.
Increment the revision whenever variant semantics, membership, ordering, or
selection policy changes.

See [Kernel Families](../gpu-programming/kernel-families.md) for the complete
selection model and example.

---

## Debug Facilities

```rust
use cuda_device::debug;

let t = debug::clock64();       // Cycle counter
debug::trap();                  // Abort kernel
debug::breakpoint();            // cuda-gdb breakpoint
unsafe { cuda_device::barrier::nanosleep(1000) }; // Sleep ~1μs
debug::prof_trigger::<7>();     // Nsight profiler trigger
```

---

## Quick Reference Tables

### cuda-device Modules

| Module               | Description                                                      | Min SM   |
|:---------------------|:-----------------------------------------------------------------|:---------|
| `thread`             | Thread/block IDs, `index_1d`, `sync_threads`                     | All      |
| `config`             | Compile-time policies, shapes, tiles, atoms, layouts, memory spaces, and scopes | All |
| `disjoint`           | `DisjointSlice<T>` — typed writes completed by a launch proof    | All      |
| `shared`             | `SharedArray<T, N>`, `DynamicSharedArray<T>`, `SharedPtr<T>`     | All      |
| `warp`               | Shuffle, vote, match, lane/warp ID                               | All      |
| `atomic`             | Scoped atomics (device/block/system)                             | sm_70+   |
| `debug`              | `clock64`, `trap`, `breakpoint`, `gpu_printf!`                   | All      |
| `fence`              | `threadfence_block` / `threadfence` / `threadfence_system`       | All      |
| `grid`               | Grid-scoped `sync` (cooperative kernel launches)                 | sm_70+   |
| `cooperative_groups` | Typed handles, warp/block reductions and scans                   | All      |
| `barrier`            | `ManagedBarrier` — async mbarrier for TMA/MMA                    | sm_90+   |
| `cluster`            | Thread block clusters, DSMEM                                     | sm_90+   |
| `tma`                | `TmaDescriptor`, bulk tensor copies (1D–5D)                      | sm_90+   |
| `wgmma`              | Warpgroup MMA (fence/commit/wait)                                | sm_90    |
| `tcgen05`            | 5th-gen tensor cores, TMEM, `TmemGuard`                          | sm_100+  |
| `cusimd`             | `CuSimd<T, N>`, `Float2`/`Float4`                                | All      |
| `clc`                | Cluster Launch Control                                           | sm_100+  |

### Crate Map

| Crate             | Role                                                                   |
|:------------------|:-----------------------------------------------------------------------|
| `cuda-device`     | Device intrinsics and types (`#![no_std]`)                             |
| `cuda-macros`     | Proc macros (`#[kernel]`, `#[device]`, `gpu_printf!`, `ptx_asm!`)      |
| `cuda-host`       | Typed module loading plus low-level launch helpers                     |
| `cuda-core`       | Safe RAII wrappers for contexts, streams, buffers, VMM, and P2P        |
| `cuda-async`      | `DeviceOperation`, `DeviceFuture`, `DeviceBox<T>`                      |
| `cuda-bindings`   | Raw `bindgen` FFI to `cuda.h`                                          |
| `cargo-oxide`     | Cargo subcommand (`cargo oxide run`, `build`, `inspect`, `clean`, `sanitize`, `debug`) |
