# The Safety Model

A GPU kernel runs thousands of threads that all see the same memory at the
same time. On a CPU, Rust prevents data races through ownership and borrowing
-- one mutable reference, no aliases, enforced at compile time. On a GPU,
you have 2048 threads per SM, all launched from the same function, all
pointing at the same output buffer. The borrow checker was not designed for
this.

cuda-oxide solves the problem in layers. The common case -- one thread writes
one element under a checked launch contract -- is safe by construction.
Uncommon cases -- shared memory, warp shuffles, hardware intrinsics -- require
`unsafe` with documented contracts. And the frontier cases -- TMA, tensor
cores, cluster-level communication -- are fully manual, matching the
complexity of the hardware they control.

This chapter explains the model, walks through each layer, and tells you
exactly when you need `unsafe` and why.

---

## Three tiers

cuda-oxide organizes kernel safety into three tiers based on how much the
compiler can verify:

| Tier       | Description                                             | `unsafe` Required? |
|:-----------|:--------------------------------------------------------|:-------------------|
| **Tier 1** | Safe kernel body plus a checked `PreparedLaunch`        | No                 |
| **Tier 2** | Explicit `unsafe` with clear safety contracts           | Yes, scoped        |
| **Tier 3** | Raw hardware intrinsics -- full user responsibility     | Yes, pervasive     |

Most application kernels live entirely in Tier 1 or straddle Tier 1 and 2.
Tier 3 is for performance engineers building at the level of CUTLASS or
Triton IR. If you are writing a vecadd, a GEMM, or a reduction, you will
rarely leave Tier 2.

---

## Tier 1: safe by default

### The core idea: `DisjointSlice<T>` + `ThreadIndex`

The primary device-side safety abstraction is a pair of types that together
support race-free parallel writes:

- **`ThreadIndex<'kernel, IndexSpace>`** -- an opaque witness around a
  `usize`, with no public constructor. The only way to obtain one is
  through trusted functions (`index_1d`, `index_2d::<S>`) that derive
  the value from **hardware built-in variables** (`threadIdx`, `blockIdx`,
  `blockDim`) -- read-only special registers the runtime fills in at
  launch. `ThreadIndex` is `!Send + !Sync + !Copy + !Clone`, and the
  `'kernel` lifetime ties it to a stack-local scope inside the kernel
  body, so it cannot be smuggled across threads via shared memory or
  outlive the kernel.
- **`DisjointSlice<T, IndexSpace>`** -- a slice-like type whose
  `get_mut()` method only accepts a `ThreadIndex` whose `IndexSpace`
  matches its own. Returns `Option<&mut T>` -- `None` for out-of-bounds
  indices.

Put them together and you get a kernel body with zero `unsafe`:

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

`get_mut_indexed` is the one-call form: it mints the per-thread witness
and resolves it to a `&mut T` in a single shot. The explicit two-step
form is also available when you need the index for parallel arithmetic
on multiple slices:

```rust
#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(c_elem) = c.get_mut(idx) {
        let i = idx.get();
        *c_elem = a[i] + b[i];
    }
}
```

Safety follows from five facts:

1. `index_1d()` uses only X coordinates. Its value is unique when block and
   grid Y/Z dimensions are all 1. A `domain = 1` prepared launch proves that;
   without that typed proof, the device checks Y/Z and marks the witness
   invalid when either axis is active. Safe slice access then returns `None`.
2. `get_mut()` is bounds-checked -- out-of-range threads get `None`.
3. The `IndexSpace` parameter ties each witness to its layout: a
   `DisjointSlice<T, Index2D<128>>` will not accept a
   `ThreadIndex<'_, Index2D<256>>` -- mixing strides is a compile error.
4. The witness is `!Send + !Sync + !Copy + !Clone` and `'kernel`-scoped,
   so threads cannot launder each other's indices through shared memory.
5. A checked `PreparedLaunch<K>` ties the host geometry to the exact kernel.

The borrow checker sees a single `&mut T` per thread. The hardware
provides the coordinates; the launch proof makes their linearization disjoint.
The type system ties the device and host proofs together.

```text
1D prepared launch: Y=Z=1 -> index_1d values are unique -> safe get_mut
raw 2D launch:       active Y/Z -> invalid witness -> safe get_mut returns None
```

A raw launch is still `unsafe`: the host must uphold its other memory and
launch obligations even though this rank check fails closed on the device.

### Trusted index functions

`ThreadIndex` is only as trustworthy as the functions that create it. Here
are the constructors cuda-oxide provides:

| Function                      | Formula                                 | Return Type                                   | Notes                                            |
|:------------------------------|:----------------------------------------|:----------------------------------------------|:-------------------------------------------------|
| `index_1d()`                  | `blockIdx.x * blockDim.x + threadIdx.x` | `ThreadIndex<'kernel, Index1D>`               | Invalid when an untyped launch has active Y/Z    |
| `index_2d::<S>()`             | `row * S + col`                         | `Option<ThreadIndex<'kernel, Index2D<S>>>`    | Const stride; mixing strides is a compile error  |
| `index_2d_runtime(&slice)`    | `row * width + col`                     | `Option<ThreadIndex<'kernel, Runtime2DIndex>>`| Row width read from the slice, bound by the host |
| `index_2d_row()`              | `blockIdx.y * blockDim.y + threadIdx.y` | `usize`                                       | Component accessor, not a witness constructor    |
| `index_2d_col()`              | `blockIdx.x * blockDim.x + threadIdx.x` | `usize`                                       | Component accessor, not a witness constructor    |

`index_2d_row()` and `index_2d_col()` return plain `usize` -- they give
you the components for arithmetic, but cannot be used to index into a
`DisjointSlice`. Only the linearized result earns a `ThreadIndex`.
Likewise, 2D uniqueness assumes the launch has no active Z dimension. A
`domain = 2` prepared launch proves that. Without a typed proof, the device
checks Z and returns `None` when it is active.

(check-a-tile-once)=
### Check a tile once

For a small fixed tile, one bounds check can cover every element in that tile:

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

`LinearTiles<4>` gives thread 0 elements 0--3, thread 1 elements 4--7,
and so on. `tile_thread32` checks the complete four-element tile once. If only
part of a tile fits, it returns `None`. Constant positions such as
`at_const::<0>()` need no further runtime bounds check; an invalid constant
fails compilation.

The two annotations provide the checked facts:

- `#[kernel(launch_context = launch_context)]` creates a local device value
  carrying facts about this launch. It is not a kernel argument.
- `domain = 1` says only the X axis is active.
- `coordinates = u32` allows `thread::index_1d_u32(launch_context)`, which
  returns this thread's unique index across the full 1-D launch as a `u32`.
- `block = (64, 1, 1)` requires exactly 64 threads in each block. The launch
  may still contain many blocks.

An exact `block` is checked twice, in two places that fail independently.
Preparation rejects a mismatched launch configuration on the host, before any
driver call. The shape also reaches the device compiler, which emits
`.reqntid 64, 1, 1`, so the CUDA driver refuses any launch whose block differs
on any axis. The second check covers the paths the first one cannot see: an
`unsafe` raw launch, and a module whose loaded PTX is not the artifact the
generated Rust markers describe.

`.reqntid` replaces `.maxntid` for a contracted kernel. ptxas rejects an entry
declaring both, and an exact shape already bounds the thread count. A kernel
carrying only `#[launch_bounds]` keeps `.maxntid`, which bounds the product
`x * y * z` and admits every shape reaching it. That difference is the reason a
thread bound cannot stand in for a block shape: under `.maxntid 64, 1, 1` the
driver accepts `(8, 8, 1)` and `(4, 4, 4)` alongside `(64, 1, 1)`.

For 2-D tiles, use `thread::coord_2d_u32(launch_context)` with
`RowMajorTiles<ROWS, COLS, ROW_STRIDE>`. `ROW_STRIDE` is the logical row width
in elements and must match the buffer's layout. The buffer length does not
have to be a multiple of the row width; only complete tiles are returned.

See the runnable
[`proof_carrying_views`](https://github.com/NVlabs/cuda-oxide/tree/main/cuda-oxide/crates/rustc-codegen-cuda/examples/proof_carrying_views)
example for 1-D and 2-D versions.

### How `index_2d` is type-safe

`index_2d::<S>()` is const-generic over the row stride. The witness
comes back as `ThreadIndex<'kernel, Index2D<S>>`, and a
`DisjointSlice<T, Index2D<S>>` only accepts that exact `S`. Two
threads cannot mint witnesses with different strides and feed them
into the same slice -- the type system rejects it:

```rust
#[kernel]
pub fn ok(mut out: DisjointSlice<u32, Index2D<128>>) {
    if let Some(idx) = thread::index_2d::<128>() {  // matches
        if let Some(slot) = out.get_mut(idx) { *slot = 1; }
    }
}

#[kernel]
pub fn rejected(mut out: DisjointSlice<u32, Index2D<128>>) {
    if let Some(idx) = thread::index_2d::<256>() { // ⛔ Index2D<256> != Index2D<128>
        if let Some(slot) = out.get_mut(idx) { *slot = 1; }
    }
}
```

The witness is also `!Send + !Sync + !Copy + !Clone`, and its `'kernel`
lifetime is borrowed from a stack-local scope the macros inject -- so a
thread can't park its `ThreadIndex` in shared memory and have a neighbour
pick it up later, and the witness can't outlive the kernel body.

#### Truly runtime strides: `index_2d_runtime`

Some kernels really do receive their stride at launch time
(e.g. matrix dimensions known only on the host). For those cases there is a
corresponding witness `Runtime2DIndex`, minted from the slice the index will
address:

```rust
let idx = thread::index_2d_runtime(&c)?;
```

No `unsafe`, because there is no per-call stride to disagree about. The row
width travels inside the slice, written once by the host into the launch packet, so
every thread that indexes `c` uses the same row width by construction. Passing
the stride separately could not give that: two `ThreadIndex<'_, Runtime2DIndex>`
values produced under different runtime strides have the same type, so the
type system cannot tell them apart. If you can pin the stride at compile time,
prefer `index_2d::<S>()`.

One consequence shapes the witness itself. Two `Runtime2DIndex` slices in one
kernel may carry different row widths, and their witnesses share a type, so
safe code can mint a witness from each and hand either one to either slice. The
witness therefore stores the thread's raw `(row, col)` coordinates rather than
a flat index, and `get_mut` resolves them against the **addressed** slice's
own row width (`row * width + col`, requiring `col < width`). Distinct threads
hold distinct coordinates and every thread reads the same host-bound width
from the slice it addresses, so the mapping stays injective per slice no
matter which slice minted the witness. A flat index baked in at minting time
would instead smuggle the minting slice's grid across: `(1, 0)` flattened at
width 5 and `(0, 5)` flattened at width 100 both name element 5.

#### Where a runtime row width lives

The *launch* establishes "every thread saw the same value": the host marshals
one word into the launch packet, and every thread of every block reads those
same bytes. A kernel body has no comparable way to establish it about one of
its own locals.

So a slice whose index space carries a runtime row width receives that width
the same way. On the host, `RowWidth` binds it:

```rust
module.write_c(&stream, &prepared, cuda_host::RowWidth::new(&mut c, n))?;
```

and on the device the accessor reads it back with no argument of its own:

```rust
#[kernel(launch_context = lc)]
#[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1))]
pub fn write_c(mut c: DisjointSlice<f32, RuntimeRowMajorTiles<1, 1>>) {
    let coord = thread::coord_2d_u32(lc);
    // No `unsafe`: the row width came from the host, bound to `c`.
    if let Some(mut cell) = c.tile_2d32_rt(coord) {
        cell.at_const::<0, 0>().write(1.0);
    }
}
```

Why this is enough: under one row width per slice the `last_col < stride`
check keeps each tile inside one logical row, which makes the
coordinate-to-offset mapping injective, so distinct threads own disjoint
elements. Two threads disagreeing about the width is exactly the case that
breaks it. With 1x1 tiles,
thread `(1, 0)` at stride 5 resolves flat index 5, and thread `(0, 5)` at
stride 100 also resolves 5: one element, two `&mut`.

A width supplied per call cannot rule that out even when each candidate value
is itself launch-uniform, because safe code can select between two of them
under a thread-varying condition, or pass different ones at two call sites on
one slice. Binding at the host boundary removes the choice.

The bound value still has to be the row width the buffer actually uses for
the kernel to compute the intended result. That is a correctness requirement
rather than a safety one; a width that misdescribes the layout gives wrong
answers inside the slice, never aliasing or an out-of-bounds access.

#### Launch-uniform scalars: `Uniform<T>`

The same reasoning applies to any scalar a kernel needs to be the same in
every thread, not only to a row width. `Uniform<T>` is that fact as a type.
Declare the parameter and the obligation is discharged by the kernel ABI:

```rust
#[kernel(launch_context = lc)]
#[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1))]
pub fn write_c(rows: Uniform<u32>, mut c: DisjointSlice<f32, RuntimeRowMajorTiles<1, 1>>) {
    let coord = thread::coord_2d_u32(lc);
    if coord.row() >= rows.get() {
        return;
    }
    if let Some(mut cell) = c.tile_2d32_rt(coord) {
        cell.at_const::<0, 0>().write(1.0);
    }
}
```

The host method still takes a plain `u32`; `#[cuda_module]` wraps it, and
`#[repr(transparent)]` keeps the launch packet byte-identical. There is no
constructor, so device code cannot promote a per-thread value into a witness.
`Uniform::new_unchecked` remains for a value proven uniform some other way, and
`block_dim_x()` and its siblings return witnesses directly, since a launch has
one block shape.

Uniformity is closed under arithmetic whose operands are all uniform, so a
derived value keeps the witness. It is not closed under selection: choosing
between two uniform values on a thread-varying condition gives a value that
differs between threads, which is why a witness of this kind cannot carry a
proof that has to hold across a whole slice.

### The GEMM pattern

For const-stride 2D kernels (the common case -- tiled GEMM, stencil
kernels, image kernels with a fixed channel count), the const-generic
form is the safe default:

```rust
const STRIDE: usize = 1024;     // C is M x STRIDE

#[kernel]
pub fn gemm(a: &[f32], b: &[f32], mut c: DisjointSlice<f32, Index2D<STRIDE>>, m: u32) {
    let row = thread::index_2d_row();
    if let Some((c_elem, _)) = c.get_mut_indexed() {
        // col < STRIDE is guaranteed by `Some` -- no manual check needed
        if row < m as usize {
            // ... compute dot product into a local accumulator `sum` ...
            *c_elem = alpha * sum + beta * (*c_elem);
        }
    }
}
```

For runtime strides, the same shape works and stays safe; the row width comes
from the output slice rather than from a const generic:

```rust
#[kernel]
pub fn gemm_runtime(a: &[f32], b: &[f32], mut c: DisjointSlice<f32, Runtime2DIndex>, m: u32, n: u32) {
    let row = thread::index_2d_row();

    // The row width comes from `c`, bound once by the host. No `unsafe`.
    if let Some(c_idx) = thread::index_2d_runtime(&c) {
        if row < m as usize {
            // ... compute dot product ...
            if let Some(c_elem) = c.get_mut(c_idx) {
                *c_elem = alpha * sum + beta * (*c_elem);
            }
        }
    }
}
```

The `if let Some` from `index_2d_runtime` replaces the manual `col < n`
guard you'd write in CUDA C++. The `row < m` check remains because it
guards against reading garbage from the input matrices.

That covers writing the output. The *input* side of GEMM -- and the cost
of the bounds checks on every `a[...]` read -- has its own chapter:
[Bounds Checks: Pay Once or Opt Out](bounds-checks.md).

### What makes a kernel Tier 1

A kernel and its launch are fully safe -- Tier 1 -- when:

1. All mutable output access goes through `DisjointSlice::get_mut(ThreadIndex)`
2. All inputs are shared immutable references (`&[T]`)
3. No shared memory, no raw pointers, no intrinsics beyond thread indexing
4. The host uses a matching checked `PreparedLaunch<K>`

Examples in this tier include `vecadd`, `helper_fn`, `generic`, `host_closure`,
and the naive GEMM kernels in the `gemm` and `async_mlp` examples when launched
through matching contracts. Their raw `LaunchConfig` paths remain unsafe.

---

## Tier 2: scoped `unsafe`

Not every kernel fits the "one thread, one output element" pattern. When
threads need to cooperate -- sharing data through fast on-chip memory,
communicating across lanes in a warp, or performing atomic updates -- you
need `unsafe`. The key property of Tier 2 is that the `unsafe` is *scoped*
and *auditable*: each block has a documented safety contract, and the rest
of the kernel remains safe.

### Shared memory

Shared memory is fast, on-chip, and visible to every thread in a block.
That last property is exactly why it requires `unsafe` -- the borrow checker
cannot reason about 256 threads writing to the same `static mut` array:

```rust
static mut TILE: SharedArray<f32, 256> = SharedArray::UNINIT;

unsafe { TILE[ty * TILE_SIZE + tx] = value; }

thread::sync_threads();

let neighbor = unsafe { TILE[other_idx] };
```

The contract: ensure no conflicting writes from concurrent threads without
synchronization. The `sync_threads()` barrier is the tool that makes this
work -- it guarantees all threads have finished writing before any thread
reads.

| API                       | Safety Obligation                                                                                  |
|:--------------------------|:---------------------------------------------------------------------------------------------------|
| `SharedArray<T, N>`       | Accessed via `static mut`. No conflicting writes without synchronization.                          |
| `DynamicSharedArray<T>`   | Same rules, but size is set at launch time via `LaunchConfig::shared_mem_bytes`.                   |

:::{seealso}
[Shared Memory and Synchronization](../advanced/shared-memory-and-synchronization.md)
for the full treatment: tiling, bank conflicts, dynamic allocation, and
double-buffered pipelines.
:::

### Warp intrinsics

Warp-level primitives let threads within a warp exchange data without
touching memory at all -- register-to-register transfers, coordinated in
hardware. They are `unsafe` because the hardware does not check thread
convergence: if you pass a mask that includes a diverged thread, you get
undefined behavior (typically a silent hang, which is worse than a crash).

| API                                                                | Safety Obligation                                            |
|:-------------------------------------------------------------------|:-------------------------------------------------------------|
| `shfl_sync`, `shfl_up_sync`, `shfl_down_sync`, `shfl_xor_sync`     | Source lane must be active; mask must include calling thread |
| `ballot_sync`, `any_sync`, `all_sync`                              | All threads in mask must be converged                        |
| `activemask`                                                       | Result is only meaningful at the point of call               |

:::{seealso}
[Warp-Level Programming](../advanced/warp-level-programming.md) for shuffle
patterns, reductions, and prefix sums using warp intrinsics.
:::

### Barriers and lifecycle

The `ManagedBarrier` typestate API encodes the barrier lifecycle
(`Uninit` -> `Ready` -> `Invalidated`) in the type system, so you cannot
wait on a barrier that was never initialized or use one that has been
invalidated. The `init()` and `inval()` transitions still require `unsafe`
because they interact with the hardware, but the type states prevent the
most common mistakes at compile time.

| API                                    | Safety Obligation                                                               |
|:---------------------------------------|:--------------------------------------------------------------------------------|
| `mbarrier_init`                        | Must be called by exactly one thread; barrier must be in shared memory          |
| `mbarrier_arrive` / `mbarrier_wait`    | Barrier must be initialized; token must match                                   |
| `ManagedBarrier` (typestate)           | `init()` and `inval()` require `unsafe`; state machine enforced at compile time |

### Atomics

Atomic operations are safe to *call* once you have a valid atomic reference.
The `unsafe` surface is at construction -- creating a `DeviceAtomicU32`
from a raw pointer requires the caller to guarantee that the pointer is
valid and properly aligned:

```rust
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};

// `from_ptr` is the unsafe constructor: it reinterprets existing memory.
// `DeviceAtomicU32::new(value)` is the safe one, for an owned static.
let atom = unsafe { DeviceAtomicU32::from_ptr(ptr) };
atom.fetch_add(1, AtomicOrdering::Relaxed);  // safe call
```

### Unchecked slice access

When the "one thread, one element" model does not fit -- for instance, in a
warp-level reduction where only lane 0 writes the result --
`DisjointSlice::get_unchecked_mut(usize)` provides an escape hatch:

```rust
if warp::lane_id() == 0 {
    let warp_idx = gid.get() / 32;
    // SAFETY: Only lane 0 of each warp writes; warp indices are unique
    unsafe { *out.get_unchecked_mut(warp_idx) = sum; }
}
```

The safety obligation is the same as the `ThreadIndex` system enforces
automatically: index in bounds, no two threads share the same index. The
difference is that you prove it yourself instead of letting the type system
do it for you.

This is per-access. To remove the checks behind ordinary `a[i]` indexing
for a whole kernel at once, see the `unchecked_indexing` flag in
[Bounds Checks: Pay Once or Opt Out](bounds-checks.md).

---

## Tier 3: raw hardware

At the bottom of the stack are the raw hardware intrinsics -- the APIs
that talk directly to specific functional units on specific GPU
architectures. Every call is `unsafe`, the safety contracts are complex
and architecture-dependent, and the documentation lives in the PTX ISA
manual more than in Rust doc comments.

| Feature                              | Key APIs                                                      | Architectures          |
|:-------------------------------------|:--------------------------------------------------------------|:-----------------------|
| **TMA** (Tensor Memory Accelerator)  | `tma_load_2d`, `tma_store_2d`, `TmaDescriptor`                | sm_90+ (Hopper)        |
| **tcgen05** (Tensor Core Gen 5)      | `tcgen05_mma`, `tcgen05_commit`, `TensorMemoryHandle`         | sm_100a (Blackwell DC) |
| **WGMMA** (Warpgroup MMA)            | `wgmma_mma_async`, `wgmma_commit_group`, `wgmma_wait_group`   | sm_90+ (Hopper)        |
| **Cluster**                          | `cluster_rank`, `map_shared_rank`, `cluster_barrier_arrive`   | sm_90+ (Hopper)        |
| **CLC** (Cluster Launch Control)     | `clc_prefetch`, `clc_query_channel`                           | sm_100+ (Blackwell)    |
| **TMEM** (Tensor Memory)             | `TmemGuard` (typestate), `tmem_alloc`, `tmem_dealloc`         | sm_100a (Blackwell DC) |

If you are writing application-level kernels, you should not need Tier 3
APIs. They exist for the people building the next CUTLASS -- and for those
people, cuda-oxide provides the same hardware access as inline PTX in
CUDA C++, with Rust's type system available (but not enforced) as a
guardrail.

:::{seealso}
[Tensor Memory Accelerator](../advanced/tensor-memory-accelerator.md),
[Matrix Multiply Accelerators](../advanced/matrix-multiply-accelerators.md),
and [Cluster Programming](../advanced/cluster-programming.md) for detailed
coverage of Tier 3 features.
:::

---

## What the borrow checker gives you

cuda-oxide is not a DSL or a macro system -- it runs the real `rustc`
frontend on your kernel code. That means every safety guarantee Rust
provides on the CPU is also enforced on the GPU:

| Guarantee                            | How It Works                                                                                                                                             |
|:-------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Ownership and borrowing**          | Lifetime errors, use-after-free, and aliasing violations caught at compile time                                                                          |
| **Safe parallel writes**             | `DisjointSlice<T>` + `ThreadIndex` + matching prepared geometry prove that writes do not race                                                            |
| **Explicit `unsafe` scoping**        | Raw pointer access requires `unsafe`, making obligations visible and auditable                                                                           |
| **Convergent attribute enforcement** | Sync primitives (barriers, fences, shuffles) marked `convergent` in the IR, preventing the optimizer from moving or duplicating them across control flow |

The first three are standard Rust. The fourth is GPU-specific: CUDA's
`bar.sync`, fence, and warp shuffle instructions must not be duplicated or
reordered by the compiler. cuda-oxide marks them `convergent` in the IR so
that LLVM's optimization passes leave them alone.

---

## The hard problems

Rust's borrow checker was designed for single-threaded ownership with
`Send`/`Sync` for CPU concurrency. SIMT execution introduces patterns that
the borrow checker was never taught to reason about. Here is an honest
accounting of what cuda-oxide does *not* enforce today -- and why these
problems are solvable.

### Thread-divergent control flow

Rustc's JumpThreading MIR optimization duplicates function calls into both
branches of an if-statement -- a perfectly sound optimization on CPUs, but
it breaks GPU barrier semantics where all threads in a block must converge
at the same `bar.sync` instruction. cuda-oxide currently disables
JumpThreading for device code (`-Z mir-enable-passes=-JumpThreading`). A
proper solution would teach the compiler about convergence requirements so
it can optimize around them instead of disabling the pass entirely.

### Shared memory access patterns

The borrow checker cannot reason about whether thread 0 writing `smem[0]`
and thread 1 writing `smem[1]` is safe -- it sees `&mut smem` and rejects
it. `DisjointSlice` solves the unique-index-write pattern, but not
cooperative patterns like reductions, scans, or producer/consumer pipelines
where multiple threads intentionally access overlapping regions with
synchronization between phases.

### Warp-level convergence

Operations like `shfl_sync` and `ballot_sync` require that all threads
named in the participation mask are actually converged at the call site.
The type system cannot enforce this today. If threads have diverged and you
pass a full mask, you get a silent hang -- the worst kind of bug, because
there is no crash and no error message, just a kernel that never finishes.

### Memory space awareness

GPU memory has distinct address spaces -- global, shared, local, TMEM.
A `&mut` to shared memory is visible to every thread in the block; a
`&mut` to local memory is private to one thread. The borrow checker treats
them identically. This is conservative (it rejects some safe programs) but
never unsound (it does not accept unsafe ones). Still, a memory-space-aware
borrow checker could accept more programs without `unsafe`.

### Why these are solvable

The building blocks already exist in Rust's type system. They need to be
extended, not reinvented:

| Idea                                  | What It Solves                                                                                                                                                    |
|:--------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Execution-resource-aware types**    | Functions annotated with their execution level (grid / block / warp / thread). A barrier call inside a divergent branch becomes a compile-time error.             |
| **Memory views**                      | Generalized parallel access patterns -- like `DisjointSlice` but covering blocked, striped, transposed, and composed layouts. Type-checked race-freedom at scale. |
| **Extended borrow checking for sync** | Statically enforce that barriers cannot be forgotten, placed at divergent control flow, or duplicated by the optimizer. Convergence in the type system.           |

The first pieces of the memory-views idea have landed: row and column
read views over runtime-sized matrices, and runtime-width output tiles.
[Bounds Checks: Pay Once or Opt Out](bounds-checks.md) covers them.

All of this is compile-time analysis. The generated PTX is identical to what
you would write by hand -- the safety net disappears at code generation.
Zero runtime cost.

cuda-oxide is well-positioned to deliver this incrementally. The real `rustc`
borrow checker already runs on device code. The IR infrastructure (pliron
dialects) supports GPU-aware analysis passes. The full compilation pipeline
from MIR to PTX is under our control. And each new safety check is additive
-- existing kernels keep compiling while new ones get stronger guarantees.

---

## Writing safe kernels: a cheat sheet

### The default path

For most kernels, start here:

```rust
#[kernel]
pub fn my_kernel(input: &[f32], mut output: DisjointSlice<f32>) {
    if let Some((out, idx)) = output.get_mut_indexed() {
        *out = transform(input[idx.get()]);
    }
}
```

For const-stride 2D, parameterise the slice and ask for the const index:

```rust
#[kernel]
pub fn tile_kernel(mut output: DisjointSlice<f32, Index2D<1024>>) {
    if let Some((out, _idx)) = output.get_mut_indexed() {
        *out = ...;
    }
}
```

The rules:

- Use `DisjointSlice` for all mutable outputs.
- Use `&[T]` for all read-only inputs.
- For 1D grids, declare `launch_contract(domain = 1, ...)` and use a prepared
  launch. Inside the kernel, default to `get_mut_indexed()`. If you need the index for
  arithmetic against multiple slices, fall back to the explicit pair:
  `let idx = thread::index_1d(); slice.get_mut(idx)`.
- For const-stride 2D grids, parameterise the slice as
  `DisjointSlice<T, Index2D<S>>` and use `get_mut_indexed()` or
  `thread::index_2d::<S>()`. Mismatched strides are a compile error.
- For runtime strides, reach for `thread::index_2d_runtime(&slice)` with a
  `Runtime2DIndex`-tagged slice. The row width travels inside the slice,
  bound once by the host, so no `unsafe` and no per-call stride to disagree
  about.
- Always bounds-check via `get_mut()` / `get_mut_indexed()` (both return
  `Option`).

If the kernel body compiles without `unsafe`, it satisfies the device-side
part of the proof. Race freedom also requires a matching prepared launch, or
an explicit unsafe proof for a raw configuration.

### When you need `unsafe`

| Pattern              | Why                                           | Mitigation                                                          |
|:---------------------|:----------------------------------------------|:--------------------------------------------------------------------|
| Shared memory        | Multiple threads access the same `static mut` | Synchronize with `sync_threads()` before cross-thread reads         |
| Warp shuffles        | Thread convergence is not compiler-checked    | Use `FULL_MASK` for full-warp operations; document partial masks    |
| Atomics              | Construction from a raw pointer               | Wrap in a helper; the atomic operations themselves are safe         |
| Non-uniform writes   | Not every thread writes to its own index      | Use `get_unchecked_mut` with a documented uniqueness argument       |
| Hardware intrinsics  | Complex, architecture-specific contracts      | Follow the PTX ISA documentation; test on target hardware           |

### The `SAFETY` comment

For every `unsafe` block, document *why* the invariants hold. Not what the
code does -- the code already says that -- but why this particular usage is
safe:

```rust
// SAFETY: Only lane 0 of each warp executes this branch.
// Warp indices (gid / 32) are unique across warps, so no two
// threads write to the same output element.
if warp::lane_id() == 0 {
    let warp_idx = gid.get() / 32;
    unsafe { *partial_sums.get_unchecked_mut(warp_idx) = warp_sum; }
}
```

This is not ceremony. When a kernel data-races at 2 AM and you are staring
at a `compute-sanitizer` log, past-you's safety comments are the fastest
path to the bug.

:::{tip}
If you cannot write a convincing `SAFETY` comment for an `unsafe` block,
that is a signal that the invariant is not actually maintained. Restructure
the code until the argument is obvious, or use a safe API instead.
:::

---

## Summary

| Property                                                        | Status                                               |
|:----------------------------------------------------------------|:-----------------------------------------------------|
| Borrow checker on device code                                   | Enforced (real `rustc` frontend)                     |
| Safe 1D parallel writes (`DisjointSlice + index_1d`)            | Enforced with a `domain = 1` prepared launch         |
| Safe 2D parallel writes -- const stride                         | Enforced (`Index2D<S>` mismatch is a compile error)  |
| Safe 2D parallel writes -- runtime stride                       | Enforced (host-bound row width, per-slice resolution)|
| `ThreadIndex` non-transferable across threads (smem laundering) | Enforced (`!Send + !Sync + !Copy + !Clone + 'kernel`)|
| `&mut [T]` kernel parameter                                     | NOT enforced -- treat any `&mut` arg as `unsafe`     |
| Explicit `unsafe` for shared memory, intrinsics                 | Enforced (Rust language rules)                       |
| Convergent attribute on sync primitives                         | Enforced (IR-level `convergent` marking)             |
| Thread convergence for warp ops                                 | NOT enforced (runtime obligation)                    |
| Memory space awareness (shared vs global)                       | NOT enforced (future work)                           |

`&mut [T]` as a kernel parameter is the next outstanding gap: the macro
accepts the type today but the runtime layout (every thread sees the same
backing pointer) means a write through `&mut data[i]` from two different
threads is the same kind of aliasing that `DisjointSlice` exists to
prevent. Until the macro rejects `&mut [T]` outright (or rewrites it to
`DisjointSlice`), treat any kernel that takes one as if every line in it
were `unsafe`.

The safety model is designed to make the common case safe through a kernel
body plus a prepared launch, while providing explicit escape hatches for
everything else. Use `unsafe` only when you are supplying a proof that the
compiler and launch contract do not carry.
