# Host API

The host API covers tensors, CUDA contexts and streams, kernel launches, and
`DeviceOp` execution. For an introduction, see
[Host vs. Device Code](../guide/host-vs-device.md),
[Tensors and Tiles](../guide/tensors-and-tiles.md), and
[Device Operations](../guide/device-operations.md).

---

## Tensor Creation and Views

### `api::*` constructors

All creation functions return a `DeviceOp` — allocation and initialization happen when the operation runs, not when it is constructed.

| Function | Output | Description |
|---|---|---|
| `api::zeros::<T>(shape: &[usize])` | `DeviceOp<Output = Tensor<T>>` | All zeros |
| `api::ones::<T>(shape: &[usize])` | `DeviceOp<Output = Tensor<T>>` | All ones |
| `api::full::<T>(val, shape: &[usize])` | `DeviceOp<Output = Tensor<T>>` | Fill with scalar value |
| `api::fill::<T>(tensor, val)` | `DeviceOp<Output = Tensor<T>>` | Fill an existing tensor and return it |
| `api::arange::<T>(len: usize)` | `DeviceOp<Output = Tensor<T>>` | `[0, 1, 2, ..., len-1]` (1D) |
| `api::linspace(start: f32, stop: f32, n: usize)` | `DeviceOp<Output = Tensor<f32>>` | `n` values evenly spaced from `start` to `stop` |
| `api::eye(n: usize)` | `DeviceOp<Output = Tensor<f32>>` | `n × n` identity matrix |
| `api::eye_rect(rows: usize, cols: usize)` | `DeviceOp<Output = Tensor<f32>>` | `rows × cols`, ones on main diagonal |
| `api::convert::<From, To>(src: Arc<Tensor<From>>)` | `DeviceOp<Output = Tensor<To>>` | Convert tensor element type |
| `api::rand::<T, RANK>(shape: [usize; RANK], seed: Option<u64>)` | `DeviceOp<Output = Tensor<T>>` | Uniform `[0, 1)` from cuRAND (`T: RandUniform`) |
| `api::randn::<T, RANK>(mean: T, std: T, shape: [usize; RANK], seed: Option<u64>)` | `DeviceOp<Output = Tensor<T>>` | Normal `N(mean, std²)` from cuRAND (`T: RandNormal`) |
| `api::randn_f16(mean: f16, std: f16, shape: [usize; RANK], seed: Option<u64>)` | `DeviceOp<Output = Tensor<f16>>` | Normal for `f16` (generates `f32` and converts; cuRAND has no native `f16`) |

Shape conventions vary across the module: `zeros`/`ones`/`full` take `&[usize]` slices of arbitrary length; `rand`/`randn` take `[usize; RANK]` arrays (rank is a const generic). The `RANK` parameter is usually inferred from the array literal.

```rust
use cutile::api;

let z = api::zeros::<f32>(&[1024]).sync_on(&stream)?;
let m = api::ones::<f32>(&[256, 256]).sync_on(&stream)?;
let r = api::randn(0.0f32, 1.0, [32, 64, 128], None).sync_on(&stream)?;   // 3D N(0, 1)
let u = api::rand::<f32, 1>([1024], Some(42)).sync_on(&stream)?;          // Uniform with fixed seed
let idx = api::arange::<i32>(1024).sync_on(&stream)?;
let I = api::eye(64).sync_on(&stream)?;
```

### Tensor and `DeviceOp` shape helpers

Host-side reshapes are zero-copy metadata changes. They require the new shape
to preserve the element count and, for borrowed views, to be contiguous.

| API | Description |
|---|---|
| `tensor.reshape(&shape)` | Consume and return `Result<Tensor<T>, Error>` with a new shape (`Reshape` trait). |
| `(&arc_tensor).reshape(&shape)` | Return `Result<Arc<Tensor<T>>, Error>`: a new `Arc<Tensor<T>>` sharing the same allocation with new shape metadata. |
| `device_op.reshape(&shape)` | Reshape the `Tensor<T>` or `Arc<Tensor<T>>` produced by a `DeviceOp`. |
| `tensor.partition(shape)` | Consume a tensor and create a mutable output partition for kernel launch. Panics if the tensor's storage is shared with another tensor or view (for example a `(&arc_tensor).reshape(..)` alias). |
| `arc_tensor.try_partition(shape)` | Consume an `Arc<Tensor<T>>` only if it has a single owner (returns `Err` otherwise), then `partition` it, with the same shared-storage panic. |
| `partition.unpartition()` | Recover the owned tensor from a partition returned by a kernel. |

```rust
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{Reshape, Tensor, TryPartition};
use cutile::tile_kernel::PartitionOp;
use std::sync::Arc;

let x = api::arange::<f32>(32).reshape(&[4, 8]).sync_on(&stream)?;
let z = api::zeros::<f32>(&[32]).partition([4]);

let weights: Arc<Tensor<f32>> = api::ones::<f32>(&[4, 8]).sync_on(&stream)?.into();
let weights_2d = (&weights).reshape(&[8, 4])?;
drop(weights); // weights_2d is now the storage's only owner; keeping `weights` alive would make `partition` panic
let partitioned = weights_2d.try_partition([2, 4])?;
```

### Tensor metadata and reinterpretation

`Tensor<T>` stores shape and layout metadata alongside the device allocation.
These accessors do not synchronize with the GPU:

| API | Description |
|---|---|
| `tensor.shape()` | Runtime dimensions as `&[i32]` |
| `tensor.strides()` | Runtime strides as `&[i32]` |
| `tensor.size()` | Number of elements |
| `tensor.num_bytes()` | Number of bytes in the tensor view |
| `tensor.is_contiguous()` | Whether the view is contiguous |
| `tensor.device_id()` | CUDA device ordinal for the allocation |
| `tensor.device_pointer()` | Typed non-owning `DevicePointer<T>` for interop |
| `arc_tensor.reinterpret::<U>(&shape)` | Zero-copy reinterpretation as `Arc<Tensor<U>>` |

`reinterpret` requires an `Arc<Tensor<T>>`, contiguous storage, matching total
byte size, and compatible pointer alignment:

```rust
use cutile::tensor::Tensor;
use std::sync::Arc;

let raw: Arc<Tensor<u32>> = api::arange::<u32>(4).sync_on(&stream)?.into();
let floats: Arc<Tensor<f32>> = raw.reinterpret::<f32>(&[4])?;
assert_eq!(floats.shape(), &[4]);
```

Packed FP4 model bytes can use the same zero-copy metadata path. If an
interoperability layer gives you byte storage, reinterpret it as
`f4e2m1fnx2` before launching kernels that expect typed packed FP4 tensors:

```rust
use cutile::api::{self, DeviceOpReshape};
use cutile::cuda_core::f4e2m1fnx2;
use cutile::tensor::Tensor;
use std::sync::Arc;

let packed_bytes: Arc<Vec<u8>> = Arc::new(model_bytes);
let raw: Arc<Tensor<u8>> = api::copy_host_vec_to_device(&packed_bytes)
    .reshape(&[m, k / 2])
    .sync_on(&stream)?
    .into();
let fp4: Arc<Tensor<f4e2m1fnx2>> = raw.reinterpret::<f4e2m1fnx2>(&[m, k / 2])?;
```

`reinterpret` does not repack or validate the FP4 payload. It preserves the
bytes and changes the tensor element type metadata, so the producer of
`model_bytes` is responsible for the low-nibble, high-nibble packing order.

### `TensorView`: zero-copy views and slices

`TensorView` provides zero-copy borrowed views of a tensor with a different shape or offset. Views borrow the underlying tensor — the tensor cannot be mutated while a view exists. The offset is applied host-side, so passing a view to a kernel hands the kernel a pointer to the correct starting address without any data movement.

| Method | Description |
|---|---|
| `tensor.view(&shape)` | Reshape to the given shape without copying. Total element count must match. |
| `tensor.slice(&ranges)` | Borrow a rectangular sub-region (one numpy-style range per dimension). |
| `view.slice(&ranges)` | Chain-slice further; offsets accumulate. |

```rust
let tensor = api::arange::<f32>(1024).sync_on(&stream)?;

// Reshape without copying.
let matrix = tensor.view(&[32, 32])?;

// Slice: borrow a subregion (numpy-style ranges).
let first_half = tensor.slice(&[0..512])?;       // elements 0-511
let row_slice = matrix.slice(&[1..3])?;          // rows 1-2, all columns
let block = matrix.slice(&[1..3, 2..6])?;        // rows 1-2, cols 2-5

// Chained slices accumulate offsets.
let inner = tensor.slice(&[100..200])?.slice(&[10..20])?;  // = tensor[110..120]
```

Pass views and slices to kernels as `&Tensor` parameters to process part of an
existing tensor without allocation or copying.

---

## Host-Device and Device-Device Transfers

Moving data between CPU and GPU, or between two device tensors, uses APIs that
return `DeviceOp`s — the copy is scheduled when the op runs, not constructed:

| API | Returns | Description |
|---|---|---|
| `api::copy_host_vec_to_device::<T>(vec: &Arc<Vec<T>>)` | `DeviceOp<Output = Tensor<T>>` | Copy host `Vec<T>` to a new device `Tensor<T>` |
| `api::copy_device_to_host_vec::<T>(tensor: &Arc<Tensor<T>>)` | `DeviceOp<Output = Vec<T>>` | Copy a device `Tensor<T>` to a host `Vec<T>` |
| `tensor.to_host_vec()` | `DeviceOp<Output = Vec<T>>` | Method form of `copy_device_to_host_vec` (preferred) |
| `device_op.to_host_vec()` | `DeviceOp<Output = Vec<T>>` | Copy the `Tensor<T>` produced by a `DeviceOp` to host |
| `api::dup(&tensor)` / `tensor.dup()` | `DeviceOp<Output = Tensor<T>>` | Allocate a new tensor and copy device-to-device |
| `api::memcpy(&mut dst, &src)` | `DeviceOp<Output = ()>` | Copy device-to-device into an existing tensor, used especially for CUDA graph updates |

```rust
// Host -> device
let data: Arc<Vec<f32>> = Arc::new(vec![1.0; 1024]);
let tensor: Tensor<f32> = api::copy_host_vec_to_device(&data).sync_on(&stream)?;

// Device -> host
let result: Vec<f32> = tensor.to_host_vec().sync_on(&stream)?;

// Device -> device
let copy = tensor.dup().sync_on(&stream)?;
```

The host-side `Vec` must remain alive until the op completes — the async copy
reads from it until the stream synchronizes. `Arc<Vec<T>>` makes this
straightforward for shared access. `to_host_vec` is available on `Tensor<T>`,
`Arc<Tensor<T>>`, and `&Arc<Tensor<T>>`; each returns the same
`DeviceOp<Output = Vec<T>>`. It is also available on a
`DeviceOp<Output = Tensor<T>>`, which is the common form after a kernel chain:

```rust
let host: Vec<f32> = kernel(out.partition([128]), &input)
    .first()
    .unpartition()
    .to_host_vec()
    .sync_on(&stream)?;
```

`api::memcpy` copies between already allocated tensors and panics if the source
and destination element counts differ. It is the usual way to update graph
input buffers before replay:

```rust
graph.update(api::memcpy(&mut input_buffer, &new_input))?;
graph.launch().sync_on(&stream)?;
```

---

## Devices and Streams

Every host program starts with a `Device`, plus one or more `Stream`s for scheduling GPU work:

```rust
use cuda_core::Device;

let device = Device::new(0)?;              // Device ordinal 0
let stream = device.new_stream()?;         // A new stream owned by this device
```

| Method | Returns | Description |
|---|---|---|
| `Device::new(ordinal: usize)` | `Result<Arc<Device>, DriverError>` | Create a device handle bound to a GPU ordinal |
| `Device::device_count()` | `Result<i32, DriverError>` | Number of CUDA-capable devices |
| `device.ordinal()` | `usize` | GPU ordinal this handle represents |
| `device.name()` | `Result<String, DriverError>` | Device name |
| `device.new_stream()` | `Result<Arc<Stream>, DriverError>` | Create a new stream on this device |
| `unsafe Device::borrow_raw(cu_ctx, cu_device, ordinal)` | `Arc<Device>` | Borrow an externally owned CUDA context/device for interop |
| `unsafe Device::borrow_with_owner(cu_ctx, cu_device, ordinal, owner)` | `Arc<Device>` | Same, holding an `Arc<dyn ForeignOwner>` so the foreign owner outlives the handle |
| `unsafe Stream::borrow_raw(cu_stream, &device)` | `Arc<Stream>` | Borrow an externally owned CUDA stream for interop |
| `unsafe Stream::borrow_with_owner(cu_stream, &device, owner)` | `Arc<Stream>` | Same, holding an `Arc<dyn ForeignOwner>` |
| `unsafe Module::borrow_raw(...)` / `unsafe Function::borrow_raw(...)` | CUDA module/function wrappers | Borrow externally owned CUDA handles |

Devices are `Arc`-wrapped for sharing across threads; streams are also `Arc`-wrapped and can be passed to `.sync_on(&stream)` for explicit stream scheduling.

The default round-robin scheduling policy handles stream assignment automatically for most workloads — these APIs are for when you need explicit stream control (debugging, deterministic ordering, paired with `AsyncKernelLaunch`, or overlapping compute with transfers on dedicated streams).

The `borrow_raw` constructors do not take ownership of the underlying CUDA
handles and therefore do not destroy them on drop. They are `unsafe fn`s: the
caller guarantees the handles stay valid for as long as the borrow is used.
The `borrow_with_owner` variants keep an `Arc<dyn ForeignOwner>` alive so the
owning runtime's object cannot be dropped first. Use them when integrating
with another runtime that owns the context, stream, module, or function.

---

## Kernel Launch Configuration

Several types configure how kernels compile and launch.

**`CompileOptions`** — runtime overrides for entry-level `optimization_hints`, typically used for autotuning:

```rust
use cutile::tile_kernel::CompileOptions;

let opts = CompileOptions::default()
    .occupancy(4)
    .num_cta_in_cga(2)
    .max_divisibility(16)
    .num_worker_warps_per_cta(4); // Bytecode 13.3+; use 4 or 8.

let result = my_kernel(args).compile_options(opts).grid(grid).await?;
```

Tile IR 13.4 restricts `num_worker_warps_per_cta` to 4 or 8. Tile IR 13.3
accepts powers of two in `[1, 32]` (the assembler clamps to its supported
worker counts). The JIT checks the selected version's range.

Different `CompileOptions` values trigger separate JIT compilations and are part of the kernel cache key.

Generated `#[cutile::entry]` launchers also expose launch-time configuration
methods:

| Method | Description |
|---|---|
| `.grid((x, y, z))` | Set an explicit runtime launch grid instead of inferring it from partitioned tensor inputs. |
| `.const_grid((x, y, z))` | Set a compile-time constant grid, enabling grid-dependent optimizations. |
| `.compile_options(opts)` | Override occupancy, cluster/CTA, worker-warp, and divisibility hints for this compilation. |
| `.generics(values)` | Bind type and const generic arguments manually when they cannot be inferred. |

The JIT compiler resolves `tileiras` in this order: `CUTILE_TILEIRAS_PATH`
when set, then `$CUDA_TOOLKIT_PATH/bin/tileiras`, then the default CUDA 13.2+
install directories (`/usr/local/cuda-13.4`, `/usr/local/cuda-13.3`, `/usr/local/cuda-13.2`,
`/usr/local/cuda-13`, `/usr/local/cuda`), and finally `tileiras` through
normal `PATH` lookup. Set `CUTILE_TILEIRAS_PATH` to force a specific binary:

```bash
CUTILE_TILEIRAS_PATH=/opt/cuda-tile/bin/tileiras cargo test -p cutile
```

The selected binary's supported bytecode versions determine emission, not
the CUDA headers used to build the Rust crates. The default is the highest
mutually supported version (13.2, 13.3, or 13.4). `CUTILE_BYTECODE_VERSION`
can pin one of those versions, but cannot force a version that the selected
assembler does not support. Configure these variables before launching;
already-resolved launch sites retain their toolchain snapshot.

**`LaunchConfig`** — grid/block/shared-memory specification for `AsyncKernelLaunch` (raw CUDA kernels launched outside the `#[cutile::entry]` path):

```rust
use cuda_core::LaunchConfig;

LaunchConfig {
    grid_dim: ((n + 255) / 256, 1, 1),    // 3D grid of thread blocks
    block_dim: (256, 1, 1),                // 3D block of threads
    shared_mem_bytes: 0,                   // Dynamic shared memory per block
}
```

**`AsyncKernelLaunch`** — wraps a CUDA driver kernel launch as a `DeviceOp`. Build the argument list with `push_arg` (safe, for `DType` scalars) or `push_device_ptr` (`unsafe`, for raw device pointers), set the launch config, then `.await` or `.sync_on()`:

```rust
use cuda_async::launch::AsyncKernelLaunch;

let mut launcher = AsyncKernelLaunch::new(function.clone());
launcher.push_arg(num_elements as u32);
launcher.push_arg(scale);
let input_ptr = input.device_pointer();
let output_ptr = output.device_pointer();
unsafe {
    launcher
        .push_device_ptr(input_ptr.cu_deviceptr())
        .push_device_ptr(output_ptr.cu_deviceptr());
}
launcher.set_launch_config(LaunchConfig {
    grid_dim: ((num_elements as u32 + 255) / 256, 1, 1),
    block_dim: (256, 1, 1),
    shared_mem_bytes: 0,
});
launcher.await?;  // Executes as a DeviceOp
```

See [Interoperability](../guide/interoperability.md) for the full walkthrough and the wrapper pattern that hides `unsafe` at the call site.

**`.generics(Vec<String>)`** — `#[cutile::entry]`-generated launchers accept this method to bind const generics and type parameters at runtime:

```rust
let generics = vec![
    "f32".to_string(),  // E
    "16".to_string(),   // BM
    "16".to_string(),   // BN
    "8".to_string(),    // BK
    "128".to_string(),  // K
];
gemm(z, x, y).generics(generics).sync_on(&stream)?;
```

Generic values are part of the kernel cache key: each unique combination triggers its own JIT compilation.

---

### Programmatic dependent launch

The generated kernel builder has an unsafe, consuming
`.programmatic_dependent_launch()` method. It is disabled by default and is
separate from `CompileOptions`. It requires Tile IR 13.4, sm_90 or newer,
and a driver providing `cuLaunchKernelEx`; unsupported requests fail rather
than silently using ordinary serialization. `cuda_async::AsyncKernelLaunch`
also exposes an unsafe mutable-builder method for lower-level callers.

Enqueue both kernels on the same stream, enabling PDL on the **consumer**:

```rust
unsafe {
    kernels::producer(input.device_pointer()).grid((1, 1, 1))
        .then(|_| kernels::consumer(input.device_pointer(), output.device_pointer())
            .programmatic_dependent_launch().grid((1, 1, 1)))
}.sync_on(&stream)?;
```

In the producer, chain its stores into
`gdc_launch_dependents_tko(Some(stored))` when dependent kernels may begin;
`Tensor::store` returns the token to chain from. Every producer tile block
must signal or finish. In the consumer, call `gdc_wait_tko(None)` and make
every predecessor-dependent memory operation consume its result token: for
a tensor input, `input.set_token(ready)` (unsafe) before creating any view
of it, so every later load through the tensor is chained after the wait;
for pointer operations, pass the token directly. Source order alone does
not establish that dependency. The launch signal permits scheduling; the
wait establishes dependency completion before consuming data.

Install the token on a function-level binding, outside conditionals and
loops. An `unsafe { input.set_token(ready) }` block is fine when `input` is
declared at function level. Block-local receivers and rebinding the tensor
after installation are rejected by the JIT; explicit token updates do not
yet participate in control-flow carries. Existing views keep their old
token, so create views only after installation. The raw
`make_partition_view` constructor is unsafe because its caller-supplied
token can otherwise discard the ordering required by safe view loads.

Work before the wait must not race unfinished producer work. Neither kernel
may rely on actual overlap for progress, and all resources must remain alive
until both kernels have finished using them. Raw pointers do not retain the
owning tensors. PDL is an unsafe opt-in because these obligations exceed the
ordinary stream-ordering guarantees of the safe tensor API.

For a complete executable producer/consumer pair, see
`cutile-examples/examples/pdl.rs` (with a stream-order vs PDL timing
comparison) or the test
`cutile/tests/gpu/tile_ir_13_4.rs::dependent_launch_waits_for_producer_data`.

## The Futures Analogy

`DeviceOp` is to GPU work what `Future` is to async I/O. Both are lazy
descriptions of work that don't execute until driven:

| Concept | `std::future::Future` | `DeviceOp` |
|---|---|---|
| What it represents | Async computation | GPU computation |
| When it runs | On `.await` or `poll()` | On `.sync()`, `.sync_on()`, or `.await` |
| Chaining | `.then()`, `.map()` via `FutureExt` | `.then()`, `.map()` on `DeviceOp` |
| Fan-in | `join!` | `zip!` |
| Fan-out | N/A (single consumer) | `.unzip()` |
| Shared access | `FutureExt::shared()` | `.shared()` |
| Type erasure | `BoxFuture` | `.boxed()` → `BoxedDeviceOp` |
| Output wrapper | `Poll<T>` | `Result<T, DeviceError>` |

An async runtime drives a `Future` through `poll()`. A `DeviceOp` submits
GPU work through `execute()`. Converting it to a future with `.await` or
`.into_future()` produces a `DeviceFuture`, which the runtime polls for GPU
completion.

---

## Combinator Reference

All combinators follow established Rust conventions. The "Precedent" column
shows which standard library or `futures` crate method inspired the design.

### Composition

| Combinator | Signature | Precedent | What it does |
|---|---|---|---|
| `zip!(a, b, …)` | `(impl DeviceOp, …) → impl DeviceOp<Output=(A, B, …)>` | `Iterator::zip` | Combine N operations into a single tuple-producing operation |
| `.unzip()` | `impl DeviceOp<Output=(A, B, …)> → (impl DeviceOp<Output=A>, …)` | `Iterator::unzip` | Split a tuple operation into independent per-element operations |
| `.then(f)` | `self → f(Self::Output) → impl DeviceOp<Output=O>` | `FutureExt::then` | Chain follow-up GPU work **on the same stream** |
| `.map(f)` | `self → f(Self::Output) → O` (no GPU work) | `FutureExt::map` | Transform output without issuing GPU work |
| `.inspect(f)` | `self → f(&Self::Output)` (passthrough) | `FutureExt::inspect` | Peek at output for debugging; returns it unchanged |

### Selection

| Combinator | Signature | Precedent | What it does |
|---|---|---|---|
| `.first()` | `impl DeviceOp<Output=(A, B, …)> → impl DeviceOp<Output=A>` | `slice::first` | Extract the first element of a tuple output |
| `.last()` | `impl DeviceOp<Output=(A, B, …)> → impl DeviceOp<Output=Z>` | `slice::last` | Extract the last element of a tuple output |

### Sharing and Erasure

| Combinator | Signature | Precedent | What it does |
|---|---|---|---|
| `.shared()` | `self → SharedDeviceOp<Self::Output>` (requires `Self: 'static`, `Output: Sync`) | `FutureExt::shared` | Cloneable, execute-once; output is `Arc<T>` |
| `shared(arc)` | `Arc<T> → SharedDeviceOp<T>` | — | Wrap an existing `Arc` as a pre-computed `SharedDeviceOp` |
| `.boxed()` | `self → BoxedDeviceOp<Self::Output>` | `FutureExt::boxed` | Type-erase for heterogeneous collections |

### Execution

| Method | Stream chosen by | Blocks? | Use case |
|---|---|---|---|
| `.sync()` | Default policy (round-robin) | Yes | Quick scripts |
| `.sync_on(&stream)` | The explicit stream | Yes | Deterministic ordering, debugging |
| `.await` | Default policy (round-robin) | No (suspends task) | Async production code |
| `.into_future()` | Default policy | No (returns `DeviceFuture`) | Manual future handling |
| `.schedule(&policy)` | The `Arc<dyn SchedulingPolicy>` you provide | No (returns `Result<DeviceFuture, DeviceError>`) | Multi-device dispatch |
| `.graph()` | Default policy (round-robin) | Yes (captures + syncs) | CUDA graph capture |
| `.graph_on(stream)` | The explicit stream | Yes (captures + syncs) | CUDA graph capture on specific stream |

:::{note}
If any kernel input is `&Tensor<T>` (borrowed), the operation is not
`'static` and cannot be used with `tokio::spawn`. Use `.sync_on()` or
`.await` in the same scope, or switch to `Arc<Tensor<T>>` for spawned tasks.
:::

---

## Supported Kernel Parameter Types

| Kernel param | Host type | Return type |
|---|---|---|
| `&Tensor<T, S>` | `Tensor<T>`, `Arc<Tensor<T>>`, `&Tensor<T>`, or `&TensorView<T>` | Same as input |
| `&mut Tensor<T, S>` | `Partition<Tensor<T>>`, `Partition<&mut Tensor<T>>`, or their `MappedLaunchPartition<…>` forms from `.map(..)` | Same as input |
| Scalar (`f32`, `i32`, etc.) | Same scalar | Same scalar |
| `*mut T`, `*const T` (unsafe only) | `DevicePointer<T>` | `DevicePointer<T>` |

The borrowed partition form (`Partition<&mut Tensor<T>>`) writes in place — no
`unpartition()` needed. Create it with `(&mut tensor).partition(shape)`.

Raw pointer entry points are `unsafe fn`s. Obtain a typed device pointer from a
tensor with `tensor.device_pointer()`, and make sure the pointer remains valid
for the duration of the kernel launch:

```rust
let backing = api::zeros::<f32>(&[1024]).sync_on(&stream)?;
let ptr = backing.device_pointer();
unsafe { raw_ptr_kernel(ptr, 1024) }.sync_on(&stream)?;
```

---

## Ownership Model

Launchers return arguments in the same ownership form they received.

### Read-only inputs (`&Tensor` params)

| Input | Returned | `tokio::spawn`? |
|---|---|---|
| `Tensor<T>` | `Tensor<T>` | Yes |
| `Arc<Tensor<T>>` | `Arc<Tensor<T>>` | Yes |
| `&'a Tensor<T>` | `&'a Tensor<T>` | No (not `'static`) |

### Mutable outputs (`&mut Tensor` params)

| Input | Returned | `unpartition()` needed? |
|---|---|---|
| `Partition<Tensor<T>>` (owned) | `Partition<Tensor<T>>` | Yes |
| `Partition<&'a mut Tensor<T>>` (borrowed) | `Partition<&'a mut Tensor<T>>` | No — tensor is written in place |

The borrowed form is created with `(&mut tensor).partition(shape)`:

### Owned: `Tensor<T>`

Pass a tensor directly — the launcher wraps it in `Arc` internally for the
kernel, then unwraps it back afterward (safe because refcount is 1):

```rust
let output = my_kernel(
    api::zeros(&[1024]).partition([128]),
    api::ones::<f32>(&[1024]),  // DeviceOp<Output=Tensor<f32>>
)
.first()
.unpartition()
.sync_on(&stream)?;
```

Use this for single-use tensors where you don't need shared access.

### Shared: `Arc<Tensor<T>>`

Wrap in `Arc` when the same tensor is passed to multiple kernels:

```rust
let x: Arc<Tensor<f32>> = api::ones(&[1024]).sync_on(&stream)?.into();

let a = kernel_a(out_a, x.clone()).sync_on(&stream)?;
let b = kernel_b(out_b, x.clone()).sync_on(&stream)?;
```

This is the most common pattern in existing code.

### Borrowed: `&Tensor<T>`

Pass a reference when you want to retain ownership and avoid `Arc` overhead.
The borrow checker ensures the tensor outlives the kernel:

```rust
let weights: Tensor<f32> = api::ones(&[1024]).sync_on(&stream)?;

// Borrow — no Arc allocation, no refcount.
let result = my_kernel(out_partition, &weights).sync_on(&stream)?;

// weights is still available here.
```

Because `&Tensor<T>` is not `'static`,
`tokio::spawn` rejects operations that borrow tensors:

```rust
let op = my_kernel(out, &weights);  // borrows weights
tokio::spawn(op.into_future());      // ← compile error: not 'static
```

### `.shared()`: Clone + Execute-Once

`.shared()` converts a `DeviceOp` into a `SharedDeviceOp<T>` that is
`Clone`. It requires the operation to be `'static` and its output to be
`Sync`. The underlying operation runs **at most once**; every clone
receives `Arc::clone()` of the cached result:

```rust
let x = api::ones::<f32>(&[32, 32]).shared();

let a = kernel_a(x.clone()).sync()?;  // x executes here (first clone to run)
let b = kernel_b(x.clone()).sync()?;  // uses cached Arc<Tensor<f32>>
```

Output type changes: `DeviceOp<Output=T>` becomes
`SharedDeviceOp` with `Output=Arc<T>`.

For pre-computed values (e.g., weight tensors), use the
`shared()` free function to wrap an `Arc<T>` directly:

```rust
use cuda_async::device_operation::shared;

let w: Arc<Tensor<f32>> = /* loaded weights */;
let w_op: SharedDeviceOp<Tensor<f32>> = shared(w);
```

### `.unwrap_arc()`

`.shared()` produces `Arc<T>` outputs (`unzip` keeps the tuple's element
types). When you need owned `T` back (e.g., to partition a tensor), use
`.unwrap_arc()`:

```rust
let x: Arc<Tensor<f32>> = api::ones(&[1024]).shared().sync()?;

let owned: Tensor<f32> = value(x).unwrap_arc().sync()?;
let partitioned = owned.partition([128]);
```

Panics if the Arc has multiple owners.

### IntoDeviceOp: Automatic Wrapping

The `IntoDeviceOp` trait lets kernel launchers accept both `DeviceOp`s and
plain values:

| Type | Wraps as |
|---|---|
| Any `impl DeviceOp<Output=T>` | Pass-through |
| `Tensor<T>` | `Value<Tensor<T>>` |
| `Arc<T>` | `Value<Arc<T>>` |
| `&'a Tensor<T>` | `Value<&'a Tensor<T>>` |
| `&Arc<T>` | `Value<Arc<T>>` (clones the Arc) |
| `&'a TensorView<'a, T>` | `Value<&'a TensorView<'a, T>>` |
| `f32`, `f64`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `usize`, `bool`, `f16`, `bf16` | `Value<T>` |
| `Partition<Tensor<T>>`, `Partition<&'a mut Tensor<T>>` | `Value<Partition<…>>` |
| `MappedLaunchPartition<Partition<…>>` | `Value<MappedLaunchPartition<…>>` |
| `DevicePointer<T>` | `Value<DevicePointer<T>>` |

```rust
// All of these work as inputs to a &Tensor kernel param:
my_kernel(out, tensor);              // Tensor<T>
my_kernel(out, arc_tensor);          // Arc<Tensor<T>>
my_kernel(out, &tensor);             // &Tensor<T>
my_kernel(out, api::ones(&[1024]));  // DeviceOp<Output=Tensor<T>>
```

---

## Scheduling Model

### Stream assignment

When you call `.sync()` or `.await`, the operation asks the **default
device's scheduling policy** for a stream. The default policy is
`StreamPoolRoundRobin` with 4 streams:

```text
op_a.sync()  →  Stream 0
op_b.sync()  →  Stream 1
op_c.sync()  →  Stream 2
op_d.sync()  →  Stream 3
op_e.sync()  →  Stream 0  (wraps around)
```

Consecutive independent operations land on different streams, enabling GPU
overlap. Operations chained with `.then()` share the parent's stream,
preserving data-dependency ordering.

### Explicit Stream: `.sync_on()`

Bypasses the policy entirely. All operations given the same stream execute
in call order:

```rust
let stream = device.new_stream()?;
let a = op_a.sync_on(&stream)?;  // Stream X
let b = op_b.sync_on(&stream)?;  // Stream X — guaranteed after op_a
```

### Available Policies

| Policy | Behavior |
|---|---|
| `StreamPoolRoundRobin` (default) | Rotates through N streams (default 4) |
| `SingleStream` | All operations on one stream — strict ordering |
| Custom `impl SchedulingPolicy` | Implement `fn next_stream()` for your own strategy |

### `.then()` Guarantees

`.then()` is the recommended way to express data dependencies. Both
operations share a single stream, so the second is guaranteed to see the
first's output fully written — no manual synchronization needed:

```rust
let result = allocate_buffer()
    .then(|buf| fill_kernel(buf))      // same stream
    .then(|buf| process_kernel(buf))   // same stream
    .sync()?;
```

**Non-reentrancy:** On any given thread, only one DeviceOp may be
executing at a time. Calling `sync_on`, `sync`, or `.await` inside a
`then` closure will return a runtime error. This prevents CUDA data
races from cross-stream access to in-flight tensors. If you need
nested execution and have verified there are no cross-stream data
races, use `unsafe then_unchecked`.

---

## Error Propagation

All execution methods return `Result<T, DeviceError>`. Errors propagate
through combinators: if any operation in a `.then()` chain fails, the
error short-circuits to the caller.

### DeviceError Variants

| Variant | When it occurs |
|---|---|
| `Driver(DriverError)` | CUDA driver call failed (OOM, invalid argument, etc.) |
| `Context { device_id, message }` | Device context assertion failed |
| `KernelCache(String)` | Declared, but no current code path constructs it |
| `Scheduling(String)` | No stream available or policy misconfigured |
| `Launch(String)` | A launcher-generated precondition failed (`kernel_launch_assert`: argument shapes, declared `preconditions`) |
| `Internal(String)` | Bug in cuda-async internals, or a nested execution attempt rejected by the non-reentrancy guard |
| `Anyhow(String)` | Converted from `anyhow::Error` or `cutile::error::Error`; JIT compilation, grid inference, and tensor errors arrive here |

### Error Handling Patterns

```rust
// Pattern 1: Propagate with ?
let x = api::zeros(&[1024]).sync_on(&stream)?;

// Pattern 2: Match specific errors
match my_kernel(args).sync_on(&stream) {
    Ok(result) => { /* use result */ }
    Err(DeviceError::Launch(msg)) => {
        eprintln!("launch precondition failed: {msg}");
    }
    Err(DeviceError::Anyhow(msg)) => {
        eprintln!("compile, grid, or tensor error: {msg}");
    }
    Err(e) => return Err(e.into()),
}
```

### cutile::error::Error vs DeviceError

`cutile::error::Error` is the top-level error type that wraps
`DeviceError` alongside other error categories (I/O, shape mismatches,
etc.). Functions that only do GPU work return `DeviceError`; functions
that mix host and device work (like the examples) return
`cutile::error::Error`.

---

## CUDA Graph Integration

### Combinator approach: `.graph_on(stream)`

Any `DeviceOp` can be captured into a replayable CUDA graph:

```rust
let forward_op = build_forward(&cfg, &weights, input, buffers);
let mut graph = forward_op.graph_on(stream.clone())?;
let output = graph.take_output().unwrap();

// Replay loop — no graph rebuilding, no kernel re-compilation.
for token in tokens {
    graph.update(api::memcpy(&mut input_buf, &token))?;
    graph.launch().sync_on(&stream)?;
}
```

This requires `Arc<Tensor<T>>` + `try_partition` for shared buffers.

### Scope approach: `CudaGraph::scope`

`CudaGraph::scope` provides an imperative alternative using `&mut` borrows
instead of `Arc`. Each `s.record(op)` records a graph node and releases
borrows immediately. A buffer written by one `record` call can be read
by the next:

```rust
let mut output = api::zeros::<f32>(&[d]).sync_on(&stream)?;
let weights = api::ones::<f32>(&[d]).sync_on(&stream)?;

let graph = CudaGraph::scope(&stream, |s| {
    s.record(kernel1((&mut output).partition([128]), &weights))?;
    s.record(kernel2((&mut output).partition([64]), &weights))?;
    Ok(())
})?;

graph.launch().sync_on(&stream)?;
```

`record` only accepts operations that implement `GraphNode` — kernel
launches and `memcpy`. Allocation ops (`zeros`, `ones`, `dup`) are
rejected at compile time because their addresses may change on replay.

### `GraphNode` trait

`GraphNode` is a marker trait for operations safe to record in a CUDA
graph. Only operations that do not allocate or free device memory
implement it:

| Implements `GraphNode` | Why safe |
|---|---|
| Macro-generated kernel launchers | Kernel launch only — no alloc/free |
| `Memcpy` (`api::memcpy`) | Copy between pre-allocated buffers |
| `Value<T>` (`value(x)`) | No GPU work |

### CudaGraph methods

| Method | What it does |
|---|---|
| `.graph()` / `.graph_on(stream)` | Capture a `DeviceOp` into a `CudaGraph<T>` |
| `CudaGraph::scope(&stream, \|s\| { … })` | Scoped capture with `&mut` borrows |
| `s.record(op: impl GraphNode)` | Record a graph node inside a scope |
| `graph.take_output()` | Retrieve the output from the capture execution |
| `graph.update(op)` | Run a `DeviceOp` on the graph's stream (e.g., copy new input) |
| `graph.launch()` | Returns a `DeviceOp` that replays the captured graph |

All device pointers are baked in at capture time. To vary inputs, pre-allocate
a buffer, pass it into the operation, and `memcpy` new data before each
launch. See [Tutorial 10](../tutorials/10-cuda-graphs.md) for a
complete walkthrough.

---

## Additional Public API

Public host-side API not covered in the sections above:

| API | Description |
|---|---|
| `unsafe Tensor::from_foreign(owner: Arc<dyn DeviceAllocation>, shape, strides)` | Borrow device memory owned by an external framework (cudarc, torch, VMM ranges); the owner is held alive by refcount and the addressable extent is verified at construction |
| `unsafe Tensor::borrow_raw_parts(dptr, device_id, shape, strides)` | Borrow a bare device pointer as a `Tensor<T>` |
| `tensor.partition_prefix(shape)` | Output partition whose launch grid may cover a per-axis prefix of the block grid (`partition` keeps strict equality) |
| `partition.map(map_shape, num_tile_blocks)` | `MappedLaunchPartition` for persistent kernels that iterate `MappedPartitionMut` indices; the launch grid is `(num_tile_blocks, 1, 1)` |
| `cutile::jit_cache::{enable, enable_default, disable, is_enabled, stats}` | Opt-in persistent on-disk cubin cache; off by default |
| launcher `.compile()` / `.compile_on(&stream)` | JIT-compile and cache the specialization without launching |
| launcher `.specialize()`, `.l1_cache_key()`, `.l2_cache_key()` | Resolve the specialization identity and the in-memory / on-disk cache keys without compiling or launching |
| `api::meta::<T>(&shape)` | Placeholder tensor carrying shape and dtype without allocating, for `.compile()` warmups |
| `CompileOptions::{opt_level, device_debug, lineinfo, sanitize_memcheck}` | Device compiler flags; each is part of the cache keys (see [Debugging and Profiling](../guide/debugging-and-profiling.md)) |
| `cutile::bench::{do_bench, do_bench_paired, BenchOptions, Measurement}` | Device-event kernel timing (see [Performance](../guide/performance.md)) |
| `cuda_async::device_context::{set_default_device, global_policy, with_device_policy}` | Set the thread's default device; read or borrow a device's scheduling policy |

---

## See Also

- [Device Operations](../guide/device-operations.md) — tutorial-style guide to streams, scheduling, and composition patterns
- [Tutorial 10](../tutorials/10-cuda-graphs.md) — end-to-end CUDA graph example
- [Interoperability](../guide/interoperability.md) — integrating custom CUDA C++ kernels into the DeviceOp model
