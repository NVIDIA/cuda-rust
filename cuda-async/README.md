# CUDA Async

CUDA Async lets programmers asynchronously compose DAGs of CUDA operations
and execute them on multiple devices using any async Rust runtime (such as tokio).

The API separates composition, scheduling, and execution:
- **Device operations** — composed using the `DeviceOp` trait and combinators.
- **Scheduling** — an implementation of `SchedulingPolicy` maps `DeviceOp`s to streams.
- **Execution** — `.sync_on(&stream)`, `.sync()`, or `.await`.

## Device Operations

`DeviceOp<Output=T>` is a lazy, composable GPU operation. Nothing executes
until `.sync()`, `.sync_on()`, or `.await` is called.

```rust
use cutile::prelude::*;

fn main() -> Result<(), DeviceError> {
    let device = cuda_core::Context::new(0)?;
    let stream = device.new_stream()?;

    let mut z = api::zeros::<f32>(&[16, 16]).sync_on(&stream)?;
    let x = api::ones::<f32>(&[16, 16]).sync_on(&stream)?;
    let y = api::ones::<f32>(&[16, 16]).sync_on(&stream)?;

    // Borrow-based: &mut z for output, &x and &y for inputs.
    let _ = saxpy((&mut z).partition([4, 4]), 2.0, &x).sync_on(&stream)?;
    // z already has the result.
    Ok(())
}
```

### Kernel Input Modes

Kernel `&Tensor` params accept three input forms. You get back the same
type you put in:

| Input | Returned | `tokio::spawn`? |
|---|---|---|
| `Tensor<T>` | `Tensor<T>` | Yes |
| `Arc<Tensor<T>>` | `Arc<Tensor<T>>` | Yes |
| `&Tensor<T>` | `&Tensor<T>` | No (not `'static`) |

Kernel `&mut Tensor` params accept two partition forms:

| Input | Returned |
|---|---|
| `Partition<Tensor<T>>` (owned) | `Partition<Tensor<T>>` |
| `Partition<&mut Tensor<T>>` (borrowed) | `Partition<&mut Tensor<T>>` |

The borrowed form writes in place — no `unpartition()` needed.

### Combinators

Operations compose via combinators that follow `futures` crate conventions:

```rust
// Chain dependent work on the same stream.
let result = allocate_buffer()
    .then(|buf| fill_kernel(buf))
    .then(|buf| process_kernel(buf))
    .sync()?;

// Combine independent operations.
let (a, b) = zip!(op_a, op_b).sync()?;

// Transform output without GPU work.
let doubled = op.map(|x| x * 2);

// Cloneable, execute-once. Clones may be executed concurrently (even from
// several threads): the first executor runs the op, the rest wait for it,
// and a clone consumed on another stream is ordered after the producing
// work through a CUDA event.
let shared = op.shared();
```

A `.then()` closure runs under a per-thread execution lock and may not
execute other operations (`.sync()`, `.sync_on()`, a nested `.await`); doing
so returns a `DeviceError` rather than racing streams against each other.
`unsafe { op.then_unchecked(|x| ...) }` releases the lock for the closure.
The caller must ensure it does not access memory on another stream while
the chain's stream is still using that memory.

## Scheduling

The `SchedulingPolicy` trait decides which CUDA stream each operation
runs on. The default `StreamPoolRoundRobin` rotates through 4 streams,
enabling overlap of independent operations.

```rust
// Implicit: .sync() and .await use the default round-robin policy.
let result = my_kernel(out, input).sync()?;

// Explicit: pin to a specific stream.
let result = my_kernel(out, input).sync_on(&stream)?;

// Multi-device: schedule on a specific device's policy.
let future = my_kernel(out, input).schedule(&policy)?;
```

Operations chained with `.then()` share a single stream and always
execute in order. Operations on different streams may overlap.

## Execution and Completion

`.sync()` / `.sync_on()` submit the work and block on `cuStreamSynchronize`.
`.await` submits the work on the first poll and then resolves in one of
three ways, in order:

1. **Inline spin.** The stream is polled with `cuStreamQuery` for a short
   budget, so microsecond-scale pipelines resolve at `sync`-like latency
   without a waker round trip. `CUDA_ASYNC_SPIN_BUDGET_US` sets the budget
   in microseconds (default `20`); `CUDA_ASYNC_SPIN_BUDGET_US=0` disables the
   spin and forces every pipeline through the notification path below (the
   correctness tests use this so the reactor is actually exercised).
2. **Flag-write reactor** (default notification path). Completion is a
   `cuStreamWriteValue32` into a slot of pinned host memory enqueued after
   the work; one process-wide `cuda-async-reactor` thread scans the armed
   slots and wakes the futures, parking when nothing is in flight. The
   wake-up cost is amortized across all in-flight pipelines instead of a
   driver-thread hop per pipeline.
3. **Host callback** (`cuLaunchHostFunc`) when the reactor is unavailable
   (slot pool exhausted, stream mem-ops unsupported), or always when
   selected explicitly: `CUDA_ASYNC_HOST_SYNC=spin` (or `spinwait`) uses
   `CU_HOST_TASK_SPINWAIT`, `CUDA_ASYNC_HOST_SYNC=block` (or `blocking`)
   uses `CU_HOST_TASK_BLOCKING`. These modes bypass the reactor so all
   three paths can be compared from one build.

Both environment variables are read once, on first use.

### Device faults

A fault on the device (illegal address, `trap`, an assert) is a sticky
error that kills the process's CUDA context. On the default paths an
awaiting future resolves with `Err(DeviceError::Driver(..))` instead of
hanging: the inline spin surfaces the error it sees, and the reactor
probes the streams of slots that stop making progress, retires the ones
the driver reports as faulted, and wakes their futures, which observe the
error on re-poll. The explicit `CUDA_ASYNC_HOST_SYNC` modes rely on the
driver running the host function, which it does not do for a faulted
context; a future awaiting through them may not resolve after a fault.

### Cancellation and Forgotten Futures

Dropping a `DeviceFuture` does not cancel submitted GPU work and does not wait
for it. Its output drops normally. GPU allocations, DMA owners and access leases
are registered independently with the submission before work is enqueued; they
retire after a CUDA event proves completion. If the driver cannot establish
completion, those owners and leases are deliberately leaked. An unpolled future
has submitted no work.

Tile and SIMT launches use this same runtime. Borrowed arguments, tensor views,
copies and values discarded by `.then()` or tuple projection remain protected
through GPU completion, including partial submission errors and panics. Successful
completion releases submission owners before returning the result.

`mem::forget(future)` deliberately leaks those owners and their device access
leases. It does not cancel work or release the leases when the GPU finishes.
Same-stream use remains ordered, and concurrent reads are allowed; conflicting
access on another stream returns an error. Await or drop a future normally to
release its leases. Mutable partitions and `memcpy` destinations must also have
unique user-visible storage; internal submission owners do not count as aliases.

Custom `KernelInputStored` and `KernelOutputStored` implementations
must implement `retain`. Custom `DeviceOp`s must register owned resources with
`ExecutionContext::retain` before enqueueing work. The unsafe `async_on` escape
hatch leaves resource lifetime and access ordering to its caller. On return,
including an error, it releases registered owners and access leases without
synchronizing or arranging background retention. The caller must keep every
submitted allocation, module, graph and DMA buffer alive until completion,
including resources discarded by combinators. Device-to-host `Vec` copies
synchronize before exposing initialized elements to host combinators.

## CUDA Graphs

`CudaGraph<T>` captures a `DeviceOp` into a replayable CUDA graph using
[stream capture](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#creating-a-graph-using-stream-capture):

Recorded tensor storage is owned by the graph, independently of `take_output`.
Each replay acquires its tensor access leases on the actual launch stream and
retains the graph through completion. Recording alone does not claim that its
device accesses have executed.

```rust
// Capture: records all GPU work into a graph. Nothing runs yet; the
// output's buffers are computed by the first launch.
let forward_op = build_forward(&cfg, &weights, input.clone(), buffers);
let mut graph = forward_op.graph_on(stream.clone())?;
let buffers = graph.take_output().unwrap();

// Replay loop — single driver call per iteration.
for token in tokens {
    graph.update(api::memcpy(&mut input_buf, &token))?;
    graph.launch().sync_on(graph.stream())?;
}
```

All device pointers are baked in at capture time. To vary inputs, copy
new data into pre-allocated buffers via `graph.update(op)` before each
`graph.launch()`; `update` accepts unit-output `GraphNode`s (kernel
launches, `memcpy`) and issues them on the graph's stream, so they are
ordered before a launch that runs on that same stream — not before one
scheduled elsewhere. `launch()` returns a [`DeviceOp`] that shares
ownership of the instantiated graph — use `.sync_on()`, `.sync()`, or
`.await` to control when and where the graph executes. Capture holds the
execution lock (no nested `.sync()` inside the captured op) and always
ends the capture, even if the op fails or panics.

## API Argument Conventions

| Layer | Arguments | Return |
|---|---|---|
| **API functions** (`zeros`, `dup`, etc.) | Concrete values | `impl DeviceOp` |
| **Extension traits** (`.reshape()`, `.to_host_vec()`, etc.) | Concrete values | `impl DeviceOp` |
| **Kernel functions** (`rms_norm`, etc.) | `IntoDeviceOp` / `KernelInput` / `KernelOutput` args | `impl DeviceOp` |

Kernel launchers accept `Tensor<T>`, `Arc<Tensor<T>>`, `&Tensor<T>`,
`Partition<Tensor<T>>`, `Partition<&mut Tensor<T>>`, scalars, and lazy
`DeviceOp`s interchangeably via trait-based dispatch.

# Testing

Run the crate tests with:

```bash
cargo test -p cuda-async
```

The unit tests, doctests, and `tests/error_handling.rs` are host-only. The
other integration binaries need a GPU; `tests/device_fault.rs` deliberately
faults the device (a sticky error that kills the process's context), which
is why it holds a single test and runs as its own process. To exercise the
reactor rather than the inline spin, run the GPU tests with
`CUDA_ASYNC_SPIN_BUDGET_US=0`; to exercise the host-callback fallback, add
`CUDA_ASYNC_HOST_SYNC=spin` (or `block`).

## Shared host API migration

`cuda_async::simt` has been removed. Import `device_operation`, `device_future`,
`device_context`, `scheduling_policies`, `device_box` and `launch` directly from
`cuda_async`. `DeviceOperation` aliases `DeviceOp`; `and_then`, `apply` and `arc`
delegate to the shared combinators. `SingleStream::from_stream(stream)` schedules
both programming models on an existing shared `cuda_core::Stream`.

Raw launches use `AsyncKernelLaunchBuilder`, followed by an explicit unsafe
`finalize_unchecked(config)` to produce an immutable `AsyncKernelLaunch`.
Tile's generated launchers perform this step internally. Kernel scalar arguments
share one `KernelArgument` blanket implementation for `DeviceCopy`. A generic
Tile scalar therefore needs both `DType` and `DeviceCopy`; tensor-only generic
parameters keep their `DType` bound. Over-aligned arguments are supported.
Non-`DeviceCopy` boxed arguments require `push_boxed_arg_unchecked` and an
explicit ABI proof.

Custom `DeviceOp::execute` implementations must return valid Rust values that
can be dropped immediately: host combinators may discard them before completion.
Retain GPU/DMA resources independently before enqueueing them. Bare borrowed
host slices cannot provide ownership for a forgettable asynchronous transfer;
use an owned retained source or a synchronous transfer. Device-to-host Vec
transfers complete before exposing the Vec to host code.
