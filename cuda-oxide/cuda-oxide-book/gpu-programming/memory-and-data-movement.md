# Memory and Data Movement

GPUs have their own memory, separate from the host. Getting data to and from the
device -- and choosing the right kind of memory once it's there -- is
fundamental to every CUDA program. This chapter covers cuda-oxide's memory
abstractions, from host/device transfers to shared memory and the kernel ABI.

:::{seealso}
[CUDA Programming Guide -- Device Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/#device-memory-accesses)
for the authoritative reference on the CUDA memory hierarchy and access patterns.
:::

## The CUDA memory hierarchy

NVIDIA GPUs expose several tiers of memory, each with different capacity,
latency, and scope:

| Memory                     | Scope       | Typical size                 | Latency          | cuda-oxide API                       |
|:---------------------------|:------------|:-----------------------------|:-----------------|:-------------------------------------|
| **Registers**              | Per thread  | ~255 × 32-bit                | 0 cycles         | Local variables                      |
| **Shared memory**          | Per block   | 48--228 KB (arch-dependent)  | ~5 cycles        | `SharedArray`, `DynamicSharedArray`  |
| **L1 cache**               | Per SM      | Combined with shared         | Hardware-managed | Automatic                            |
| **Constant memory**        | All threads | 64 KB, read-only on device   | Broadcast-cached | `ConstantMemory` via `#[constant]`   |
| **L2 cache**               | Chip-wide   | Up to 50 MB (Hopper)         | ~30 cycles       | Automatic                            |
| **Global memory (DRAM)**   | All threads | 16--80 GB (HBM)              | ~400 cycles      | `DeviceBuffer`, `DeviceBox`          |

The guiding principle: move frequently accessed data into faster, closer memory.
Registers are fastest but per-thread; shared memory is fast and visible to the
whole block; global memory is large but slow.

```{figure} images/memory-hierarchy.svg
:align: center
:width: 100%

The CUDA memory hierarchy from fastest (registers, per-thread) to largest
(global DRAM, all threads). Each tier trades capacity for latency. The right
panel shows scope and the cuda-oxide API for each level.
```

## Contexts and streams

Before diving into memory APIs, two host-side concepts that appear in every
code example need introduction: **contexts** and **streams**.

A **CUDA context** (`CudaContext`) binds the host thread to a specific GPU. It
owns all resources on that device -- modules, streams, allocations. You
typically create one at the start of your program:

```rust
use cuda_core::CudaContext;

let ctx = CudaContext::new(0).unwrap();   // bind to GPU 0
```

A **CUDA stream** (`CudaStream`) is an ordered queue of GPU operations.
Operations enqueued on the **same** stream execute in FIFO order -- each one
sees all side effects of the preceding operations. Operations on **different**
streams may overlap and run concurrently with no ordering guarantees between
them.

```rust
let stream = ctx.default_stream();     // the implicit, always-available stream
let work_stream = stream.fork()?;      // a new stream, joined to the parent's current position
```

Every memory transfer and kernel launch requires a stream. For single-stream
programs (which covers most examples in this book), the **default stream** is
all you need -- everything is sequential and correct by construction.
Multi-stream pipelines unlock overlap between compute and data movement, but
require explicit synchronization via events or `join`:

```text
Same stream:       [kernel_A] → [memcpy_B] → [kernel_C]     (automatic ordering)
Different streams: [kernel_A on stream 1] | [memcpy_B on stream 2]  (concurrent, needs events)
```

```{figure} images/cuda-streams.svg
:align: center
:width: 100%

Top: single-stream execution where operations are automatically ordered in FIFO.
Bottom: multi-stream execution where streams A and B run concurrently, with an
event establishing the data dependency between kernel_write and kernel_read.
```

:::{seealso}
The [Launching Kernels](launching-kernels.md) chapter covers stream usage in
launch macros, and the [Async GPU Programming](../async-programming/the-device-operation-model.md)
section covers `DeviceOperation` which manages streams automatically.
:::

## `DeviceBuffer` -- host/device transfers

`DeviceBuffer<T>` in `cuda_core` is the primary way to allocate device memory
and move data between host and GPU:

```rust
use cuda_core::{CudaContext, DeviceBuffer};

let ctx = CudaContext::new(0).unwrap();
let stream = ctx.default_stream();

// Host → device: copy a host slice to GPU memory
let a_dev = DeviceBuffer::from_host(&stream, &host_data).unwrap();

// Allocate zeroed device memory
let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, 1024).unwrap();

// Device → host: read results back
let results = c_dev.to_host_vec(&stream).unwrap();
```

### Key methods

| Method                            | Direction     | Description                    |
|:--------------------------------- |:--------------|:-------------------------------|
| `from_host(&stream, &[T])`        | Host → Device | Allocate + blocking copy       |
| `zeroed(&stream, len)`            | --            | Allocate + zero-fill           |
| `to_host_vec(&stream)`            | Device → Host | Async copy + return `Vec<T>`   |
| `copy_to_host(&stream, &mut [T])` | Device → Host | Copy into existing slice       |
| `cu_deviceptr()`                  | --            | Raw `CUdeviceptr` for FFI      |

### Ownership and drop

`DeviceBuffer` frees its allocation **synchronously** on drop via `cuMemFree`.
This is a blocking driver call -- it internally synchronizes the entire device
to ensure no in-flight kernel is still touching the memory. In practice, this
means:

- **Dropping while a kernel is running** will stall the host thread until the
  GPU is completely idle, then free the memory.
- **Dropping after synchronization** (e.g., after `to_host_vec` or
  `stream.synchronize()`) has no additional cost because the device is already
  idle.

For single-stream workloads this is fine -- everything executes in FIFO order,
so by the time you read results back, all kernels are done and the free is
instant. The cost becomes visible in multi-stream scenarios where you want to
overlap compute with memory operations; a synchronous free on one stream can
stall work on all other streams.

## `DeviceBox` -- async-friendly device memory

`DeviceBox<T>` in `cuda_async` solves the synchronous-free problem. On drop, it
frees memory via `cuMemFreeAsync` on a **dedicated deallocator stream**. This
is a stream-ordered operation -- the free is enqueued on the deallocator stream
and executes only after all preceding work on that stream completes. Critically,
it does **not** synchronize the device:

```rust
use cuda_async::simt::device_box::DeviceBox;
use cuda_async::simt::device_context::init_device_contexts;

init_device_contexts(0, 1)?;  // Initialize device context map (default device 0)

// DeviceBox wraps a device pointer; freed asynchronously on drop
let dev_ptr: DeviceBox<f32> = /* allocated by DeviceOperation chain */;
// When dev_ptr is dropped, cuMemFreeAsync is called on the deallocator stream.
// Other streams continue running without stalling.
```

### Choosing between them

|                     | `DeviceBuffer`                      | `DeviceBox`                            |
|:--------------------|:------------------------------------|:---------------------------------------|
| **Crate**           | `cuda_core`                         | `cuda_async`                           |
| **Free on drop**    | `cuMemFree` (sync -- stalls device) | `cuMemFreeAsync` (async -- no stall)   |
| **Use with**        | typed sync launches                 | typed async launches                   |
| **Host readback**   | `to_host_vec()`                     | Via explicit memcpy operation          |
| **Best for**        | Single-stream, blocking workloads   | Multi-stream, pipelined workloads      |

:::{tip}
For latency-sensitive teardown in multi-stream pipelines, prefer `DeviceBox`.
For straightforward single-stream examples, `DeviceBuffer` is simpler and
the synchronous free is effectively zero-cost.
:::

(memory-argument-scalarization)=

## Argument scalarization

When you write a kernel that takes `&[f32]`, the host and device don't agree on
how to represent a Rust slice in memory -- the struct layout can differ between
the host's x86 ABI and the NVPTX ABI. cuda-oxide solves this by
**scalarizing** aggregate types at the kernel boundary: decomposing them into
primitive values that both sides interpret identically.

| Kernel parameter type          | What the host actually passes  |
|:-------------------------------|:-------------------------------|
| `&[T]`                         | `ptr: *const T` + `len: u64`   |
| `DisjointSlice<T>`             | `ptr: *mut T` + `len: u64`     |
| `T` (scalar)                   | `T` directly                   |
| `#[grid_constant] value: &T`    | One by-value `T`; device borrows launch storage; host launch is unsafe |
| Struct `{ a: u32, b: f32 }`    | One byval value (whole struct) |
| Closure (with N captures)      | One byval value (whole struct) |
| Zero-sized types               | Stripped entirely              |

This is why typed `#[cuda_module]` methods accept `&DeviceBuffer<T>` for `&[T]`
and `&mut DeviceBuffer<T>` for writable slice-like parameters. The generated
method extracts the pointer and length for you. Inside the kernel, the compiler
reconstitutes the slice struct from the scalar parameters, so your kernel code
sees normal `&[T]` types.

```{figure} images/scalarization-abi.svg
:align: center
:width: 100%

Argument scalarization: the host passes Rust slices as (ptr, len) pairs through
the ABI boundary. The device kernel receives flat scalar parameters and the
compiler reconstitutes the original Rust types inside the kernel.
```

:::{tip}
Scalarization is completely invisible in normal kernel code. You write `&[f32]`
in the signature and use it as a regular slice. The generated launch method and
the compiler handle everything else.
:::

## `DisjointSlice` -- safe parallel writes

In CUDA C++, the standard pattern for parallel output is a raw `__global__`
pointer that every thread indexes into. This is inherently unsafe -- nothing
prevents two threads from writing to the same location.

cuda-oxide provides `DisjointSlice<T, IndexSpace>` as a safe alternative. It
wraps a mutable slice and only allows writes through a `ThreadIndex` whose
`IndexSpace` matches its own. Combined with matching prepared launch geometry,
this ensures each thread accesses a unique element:

```rust
use cuda_device::{DisjointSlice, kernel};

#[kernel]
pub fn double(input: &[f32], mut out: DisjointSlice<f32>) {
    if let Some((out_elem, idx)) = out.get_mut_indexed() {
        *out_elem = input[idx.get()] * 2.0;
    }
}
```

- `get_mut_indexed()` is the one-call form: it mints the per-thread witness
  and resolves it to a `&mut T` in a single shot. `None` covers both
  out-of-grid threads (e.g. `col >= ROW_STRIDE` for 2D) and out-of-slice
  indices.
- The explicit two-step form `let idx = thread::index_1d(); out.get_mut(idx)`
  is also available when you need the index for parallel arithmetic across
  multiple slices.
- For patterns like reductions where multiple threads intentionally write
  to the same location, `get_unchecked_mut` (unsafe) provides an escape
  hatch.

### Why `ThreadIndex` makes this safe

The key to `DisjointSlice`'s safety is `ThreadIndex<'kernel, IndexSpace>` --
an opaque witness with no public constructor. The only way to obtain one
is through trusted index functions that derive the value from hardware
built-in variables (`threadIdx`, `blockIdx`, `blockDim`):

```rust
let idx = thread::index_1d();          // ThreadIndex<'_, Index1D> -- ok
let bad = ThreadIndex::new(42);        // does not exist -- private constructor
```

This works because CUDA's thread indices are **uniform values provided by the
hardware**: every thread in a block receives a unique `threadIdx` from the GPU's
warp scheduler. For 1D grid launches (where only the x dimension is used), the
global index derived from `blockIdx.x * blockDim.x + threadIdx.x` is unique
across the entire grid.

That last statement is conditional: `index_1d()` ignores Y and Z. A 2D grid
would repeat the same X-derived index for every Y coordinate.

```text
PreparedLaunch(domain=1) -> Y/Z inactive -> safe DisjointSlice writes
raw LaunchConfig         -> caller proves Y/Z inactive -> unsafe launch
```

The witness proves the device-side index space; the prepared launch proves the
host-side geometry. Both are needed for a fully safe launch.

The witness is also `!Send + !Sync + !Copy + !Clone`, and its `'kernel`
lifetime is borrowed from a stack-local scope the macros inject -- so a
thread can't park its `ThreadIndex` in shared memory for a neighbour to
pick up later, and the witness can't outlive the kernel body. Combine
that with the `IndexSpace` parameter (`Index1D`, `Index2D<S>`,
`Runtime2DIndex`) and the type system rejects mismatched 2D strides at
compile time too -- a data-race hazard becomes a type error.

## Shared memory

**Shared memory** is fast on-chip memory visible to all threads within a block.
It is the primary tool for inter-thread communication and data reuse within a
block, sitting between registers (per-thread) and global memory (all threads)
in both speed and scope.

### Static shared memory -- `SharedArray`

When the size is known at compile time, use `SharedArray<T, N>`:

```rust
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};

static mut TILE: SharedArray<f32, 256> = SharedArray::UNINIT;

#[kernel]
pub fn smem_example(input: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let tid = thread::threadIdx_x() as usize;

    // Load from global to shared
    unsafe { TILE[tid] = input[idx.get()]; }
    thread::sync_threads();

    // Read neighbor from shared (much faster than global)
    let neighbor = if tid > 0 {
        unsafe { TILE[tid - 1] }
    } else {
        0.0
    };
    thread::sync_threads();

    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = unsafe { TILE[tid] } + neighbor;
    }
}
```

Each `static mut SharedArray` maps to a separate `.shared` allocation in PTX.
The `sync_threads()` barrier ensures all threads have finished writing before
any thread reads.

### Dynamic shared memory -- `DynamicSharedArray`

When the size depends on runtime parameters, use `DynamicSharedArray<T>` and
specify the allocation size via `LaunchConfig::shared_mem_bytes`:

```rust
use cuda_device::{DisjointSlice, DynamicSharedArray, kernel, thread};

#[kernel]
pub fn dynamic_smem_example(input: &[f32], mut out: DisjointSlice<f32>) {
    let smem = DynamicSharedArray::<f32>::get();
    let tid = thread::threadIdx_x() as usize;

    unsafe { *smem.add(tid) = input[thread::index_1d().get()]; }
    thread::sync_threads();
    // ... use smem ...
}
```

On the host, set the size at launch:

```rust
let config = LaunchConfig {
    grid_dim: (num_blocks, 1, 1),
    block_dim: (256, 1, 1),
    shared_mem_bytes: 256 * std::mem::size_of::<f32>() as u32,
};
```

Multiple dynamic arrays can share the same allocation by using
`DynamicSharedArray::offset(byte_offset)` to partition it.

### Alignment

| Type                          | Default alignment  | Notes                        |
|:------------------------------|:-------------------|:-----------------------------|
| `SharedArray<T, N>`           | `align_of::<T>()`  | Standard alignment           |
| `SharedArray<T, N, 128>`      | 128 bytes          | Required for TMA operations  |
| `DynamicSharedArray<T>`       | 16 bytes           | nvcc-compatible default      |
| `DynamicSharedArray<T, 128>`  | 128 bytes          | Required for TMA             |

### Common pitfalls

- **Missing `sync_threads()`:** Without a barrier between shared memory writes
  and reads, threads may read stale or uninitialized data.
- **Exceeding SM limits:** Requesting too much shared memory causes
  `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`. Check your architecture's limit.

:::{seealso}
[CUDA Programming Guide -- Shared Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/#shared-memory)
for details on bank conflicts, broadcasting, and optimal access patterns.
:::

## Constant memory -- `#[constant]`

Constant memory is a small, read-only window the host writes between launches
and every thread in the grid reads. It is served by a cache built for
**broadcast**: when all lanes of a warp want the same address it is one
transaction, and when they want different addresses those are served in
sequence.

Declare it as a `static` inside a `#[cuda_module]`:

```rust
#[cuda_module]
mod kernels {
    use cuda_device::{ConstantMemory, DisjointSlice, constant, kernel, thread};

    #[constant]
    static SCALE: ConstantMemory<f32> = ConstantMemory::UNINIT;

    #[kernel]
    pub fn multiply(mut out: DisjointSlice<f32>) {
        let s = SCALE.get();
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(e) = out.get_mut(idx) {
            *e = (i as f32) * s;
        }
    }
}
```

The macro generates two host setters per constant, named `set_` plus the
static's name lowercased, so `SCALE` becomes `set_scale` and `TABLE_CONST`
becomes `set_table_const`. The plain form is stream-ordered; a
`set_<name>_blocking` sibling does a synchronous one-shot copy instead:

```rust
let module = kernels::load(&ctx)?;
module.set_scale(&stream, &3.0)?;
```

`ConstantMemory<T>` accepts `T: ConstantMemoryValue`, an unsafe marker
implemented for the primitive integers and floats, `*const T` / `*mut T`,
tuples up to eight elements, and arrays of those. The requirement it encodes is
that an all-zero bit pattern is a valid `T`, since `ConstantMemory::UNINIT`
starts out zeroed -- which is why references and `NonZero*` types are excluded,
and why a custom struct needs an explicit `#[repr(C)]` layout before opting in.

### `get()` versus `get_ref()`

This is the one thing worth knowing before using it for anything but a scalar.

`get()` returns a by-value copy. A value has no address, so indexing a
`ConstantMemory<[f32; N]>` copy at a runtime index forces the whole array to be
spilled to the thread's local depot first -- one store per element, in *every*
thread -- and the lookup then reads thread-private memory. `get_ref()` borrows
the storage instead, so the index stays in constant space and the read is a
single `ld.const`.

The `constant_memory_table` example measures both against the third option, a
plain `const T: [f32; N]` materialized as an immutable device global. Measured
on an A10G (sm_86), 256-entry `f32` table, 64 dependent lookups per thread
over 8388608 threads, all variants bit-identical:

| index        | `get()` | `get_ref()` | `const [f32; N]` (global) |
|:-------------|--------:|------------:|--------------------------:|
| divergent    | 16048us |      7616us |                 **322us** |
| warp-uniform | 16065us |   **258us** |                     322us |

Two conclusions. `get_ref` is worth 2.1x over `get` on a divergent index and
62x on a warp-uniform one, so prefer it for anything larger than a scalar. And
constant memory is not simply "faster memory": on a divergent index it is 23.7x
*behind* an ordinary array constant, because a warp's divergent reads of a small
table coalesce and stay resident in L1 while the constant cache serves the
distinct addresses in sequence. Reach for `#[constant]` when the index is
warp-uniform; reach for a plain `const` array when it is not.

Reads are not folded away across launches: the `UnsafeCell` interior keeps the
storage mutable as far as the compiler is concerned, so a `set_<name>` between
launches stays visible. Repeated reads of one entry *within* a launch may still
be merged, which is sound, because the host only writes between launches.

:::{seealso}
[CUDA Programming Guide -- `__constant__`](https://docs.nvidia.com/cuda/cuda-programming-guide/#constant)
for the hardware limits and caching rules this maps onto.
:::

## Unified memory and HMM

By default, GPUs operate in a **separate address space** from the CPU. A GPU
cannot dereference an ordinary host pointer -- the address simply doesn't map to
anything in the GPU's page tables. The traditional CUDA workflow therefore
requires explicit allocation in device memory followed by explicit copies:

```text
┌──────────────────┐              ┌──────────────────┐
│   CPU Memory     │    PCIe /    │   GPU Memory     │
│   (host DRAM)    │◄────────────►│   (device HBM)   │
│                  │   NVLink     │                  │
└──────────────────┘    copy      └──────────────────┘
   Separate address spaces -- GPU cannot dereference host pointers
```

CUDA provides mechanisms that relax this restriction by letting the GPU access
host memory transparently, at the cost of page-fault latency on first access.

### Memory access modes at a glance

| Mode                 | What GPU can access   | Allocation required    | First-access cost            | Hardware requirement      |
|:---------------------|:----------------------|:-----------------------|:-----------------------------|:--------------------------|
| **Explicit copy**    | Device memory only    | `DeviceBuffer`         | None (data copied upfront)   | Any CUDA GPU              |
| **Pinned (mapped)**  | Specific host buffers | `cudaHostAlloc`        | High (~10--20 µs per access) | Any CUDA GPU              |
| **Unified Memory**   | Managed allocations   | `cudaMallocManaged`    | Medium (page migration)      | Kepler+ (sm_30+)          |
| **HMM**              | *Any* host memory     | None                   | Medium (page fault + fetch)  | Turing+ on Linux          |

cuda-oxide primarily uses **explicit copies** (`DeviceBuffer`, `DeviceBox`) for
bulk data and **HMM** for non-move closure captures and small configuration
data.

### Unified Memory

Unified Memory is CUDA's managed-memory allocator (`cudaMallocManaged`). The
resulting pointer is valid on both the CPU and GPU -- the CUDA runtime tracks
which processor "owns" each page and migrates it on demand. When the GPU
accesses a page that currently resides in host DRAM, the runtime transparently
copies it to device memory before the kernel reads the data. This migration is
invisible to your code but not free: the first access from the "wrong" side
incurs a page fault and a DMA transfer over the interconnect. Subsequent
accesses to the same page hit the GPU's local cache.

cuda-oxide does not currently wrap `cudaMallocManaged` directly. For managed-
memory workflows you would use the CUDA driver API through raw bindings.
In practice, `DeviceBuffer::from_host` (explicit copy) covers most use cases
and gives predictable performance.

### HMM (Heterogeneous Memory Management)

HMM is a Linux kernel feature that extends Unified Memory's demand-paging model
to **all system memory** -- heap allocations, `mmap` regions, and even stack
variables. With HMM enabled, the GPU can dereference any valid host pointer
without a special CUDA allocator:

```rust
let factor = 5i32;                           // ordinary stack variable
let scale = |x: i32| x * factor;            // captures &factor (non-move)
// SAFETY: args match the kernel's signature; &factor stays alive via HMM.
unsafe { cuda_launch! { kernel: scale, args: [...] } }  // GPU reads &factor via HMM
```

Unlike Unified Memory, HMM requires no special allocation API -- the pointer is
a plain host address. When ATS (Address Translation Services) is available on
hardware-coherent platforms like Grace Hopper, it supersedes HMM and provides
hardware coherence at cache-line granularity; HMM is automatically disabled.

### What happens on a page fault

When a kernel loads from an address whose page is not resident in device memory,
the hardware and driver cooperate to fetch it:

1. The SM executes a global load (`ld.global`) for a virtual address.
2. The **GPU MMU** looks up the address in the TLB. On a miss, it walks the
   device page table.
3. If the page table has no mapping, the GPU raises a **page fault**. The
   faulting warp stalls; other warps on the same SM can continue.
4. The **CUDA driver fault handler** determines the source of the page:
   - *Unified Memory* -- the CUDA runtime identifies the managed allocation and
     initiates migration.
   - *HMM* -- the Linux kernel's HMM layer resolves the host virtual address,
     pins the host page, and either migrates it or creates a remote mapping.
5. A **DMA transfer** over PCIe or NVLink copies the page from host DRAM to
   device HBM. The GPU memory controller writes the data; the host memory
   controller services the read.
6. The GPU page table is updated, the TLB is refilled, and the **warp
   resumes**. The page is now local and cached in L2; subsequent accesses cost
   only hundreds of cycles.

The latency of step 5 depends on the interconnect:

| Interconnect           | Bandwidth  | Fault latency | Notes                                  |
|:-----------------------|:-----------|:--------------|:---------------------------------------|
| PCIe 4.0 x16           | ~25 GB/s   | ~10--20 µs    | Most desktop / workstation GPUs        |
| PCIe 5.0 x16           | ~50 GB/s   | ~5--15 µs     | Ada Lovelace + newer platforms         |
| NVLink 4.0             | ~900 GB/s  | ~1--5 µs      | Data-center GPUs (H100, B100)          |
| Grace Hopper C2C       | ~900 GB/s  | <1 µs         | Hardware coherent -- uses ATS, not HMM |

Because faults operate at page granularity (4 KB or 2 MB), a single fault can
satisfy many threads. Warp-level coalescing also helps: 32 threads reading
consecutive 4-byte elements touch at most one or two pages, not 32. On PCIe
systems, a single fault costs roughly the same as a small `cudaMemcpy` -- the
advantage of demand paging is that you only pay for the pages you actually
touch.

### How cuda-oxide uses HMM

cuda-oxide leverages HMM in two ways:

1. **Non-move closure captures.** When a non-`move` closure is passed to a
   kernel, captured variables remain on the host stack and the GPU accesses
   them through HMM pointers. This avoids copying data that the kernel only
   reads once or infrequently.

2. **Struct ABI with dynamic layout.** cuda-oxide matches Rust's actual struct
   layout (including `#[repr(Rust)]` field reordering) on the device side, so
   HMM-accessed host structs are read correctly without `#[repr(C)]` or manual
   layout specification. The compiler queries `rustc` for field offsets and
   builds matching LLVM struct types with explicit padding.

### HMM system requirements

| Requirement      | Minimum                              |
|:-----------------|:-------------------------------------|
| GPU architecture | Turing (compute capability 7.5+)     |
| Linux kernel     | 6.1.24+, 6.2.11+, or 6.3+            |
| CUDA driver      | 535+ with Open Kernel Modules        |

Check whether HMM is active on your system:

```bash
nvidia-smi -q | grep Addressing
# Addressing Mode : HMM  ← HMM is enabled
```

### When to use HMM vs explicit copies

| Scenario                                    | Recommended approach                       |
|:--------------------------------------------|:-------------------------------------------|
| Large arrays processed by many threads      | `DeviceBuffer::from_host` (explicit copy)  |
| Small read-only configuration data          | HMM (pass pointer, let GPU page-fault)     |
| Data shared between CPU and GPU iteratively | Explicit copies with double-buffering      |
| Prototyping / quick experiments             | HMM (simplest -- no copies needed)         |

:::{tip}
HMM is a convenience, not a performance strategy. For bandwidth-sensitive
kernels, explicit copies to device memory will always be faster because they
avoid page-fault overhead and use the full memory bus width.
:::

:::{seealso}
[CUDA Programming Guide -- Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html)
and [NVIDIA Blog -- Simplifying GPU Development with HMM](https://developer.nvidia.com/blog/simplifying-gpu-application-development-with-heterogeneous-memory-management/)
for the full details on page migration, prefetching, and system requirements.
:::
