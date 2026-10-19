# async_mlp

## Async Multi-Kernel Pipeline — MLP Forward Pass

Demonstrates the full `cuda-async` execution model by composing three kernels
(GEMM, MatVec, ReLU) into a lazy `DeviceOperation` pipeline and running
multiple batches concurrently across a pool of CUDA streams.

This is the most complete async example in the repository. Where `async_vecadd`
shows a single kernel with `.sync()`, this example chains multiple dependent
kernels with `.await`, shares device memory across tasks via `Arc`, and launches
concurrent work through `tokio::spawn`.

## What This Example Does

- Allocates shared model weights (W0: 64×64, W1: 64) on device via `zip!` + `.arc()`
- Builds a 4-stage pipeline per batch: GEMM → MatVec → ReLU → D2H
- Spawns 4 batches as concurrent Tokio tasks over a round-robin stream pool
- Awaits all results and verifies ReLU correctness (all outputs ≥ 0)

## Pipeline

```text
For each batch:
  input [64×64] ──► GEMM(input, W0) ──► hidden [64×64]
                                            │
                           MatVec(hidden, W1) ──► output [64]
                                                     │
                                               ReLU(output) ──► result [64]
```

## Async Patterns Showcased

### `zip!` — Concurrent Independent Operations

`zip!` composes independent `DeviceOperation`s that can execute concurrently.
Here it allocates three device buffers in parallel:

```rust
zip!(h2d(batch_data), zeros(DIM * DIM), zeros(DIM))
```

The tuple `(input, hidden, output)` is only available once all three complete.

### `and_then` — Sequencing Dependent Stages

Each kernel stage depends on the output of the previous one. `and_then` chains
them so that Stage 2 only launches after Stage 1 completes on the same stream:

```rust
zip!(...)
    .and_then(|(...)|  /* GEMM launch  */ )
    .and_then(|(...)|  /* MatVec launch */ )
    .and_then(|(...)|  /* ReLU launch   */ )
    .and_then(d2h)
```

### `.arc()` — Sharing Device Memory Across Tasks

Model weights are allocated once and wrapped in `Arc` so every batch pipeline
can read them without copies:

```rust
let (w0, w1): (Arc<DeviceBox<[f32]>>, Arc<DeviceBox<[f32]>>) =
    zip!(h2d(w0_host).arc(), h2d(w1_host).arc()).await?;

// Each batch clones the Arc (cheap refcount bump, no device copy)
let w0 = w0.clone();
```

### `value()` — Threading Data Between Stages

`DeviceOperation` closures consume their inputs and must return a new
`DeviceOperation`. `value()` lifts host-side data (device handles, module
references) into the graph so the next stage can receive them:

```rust
launch.and_then(move |()| value((hidden, output, w1, module)))
```

### `tokio::spawn` + `.into_future()` — Concurrent Execution

Each pipeline is a lazy `DeviceOperation` — no GPU work happens until it is
polled. Converting to a `Future` via `.into_future()` and spawning it onto
Tokio's executor submits all the staged GPU work onto a CUDA stream:

```rust
handles.push(tokio::spawn(pipeline.into_future()));
```

`tokio::spawn` hands the task to the executor immediately. The executor polls
it on a worker thread, which submits the entire operation chain onto a CUDA
stream and registers a host callback. The future returns `Poll::Pending` and
is woken when the GPU signals completion.

By the time the loop reaches `handle.await`, the GPU has typically already
finished (especially for small workloads). The `.await` just picks up the
already-resolved result. For larger workloads you would genuinely block here,
but all streams are running concurrently on the GPU regardless.

### `with_context` — Stream-Aware Memory Operations

The helper functions `h2d`, `zeros`, and `d2h` use `device_operation::with_context`
to access the currently assigned CUDA stream and perform async memory operations:

```rust
fn h2d(host_data: Vec<f32>) -> impl DeviceOperation<Output = DeviceBox<[f32]>> {
    device_operation::with_context(move |ctx| {
        let stream = ctx.get_cuda_stream();
        unsafe {
            let dptr = malloc_async(stream.cu_stream(), num_bytes).unwrap();
            memcpy_htod_async(dptr, host_data.as_ptr(), num_bytes, stream.cu_stream()).unwrap();
            value(DeviceBox::from_raw_parts(dptr, n, ctx.get_device_id()))
        }
    })
}
```

## Build and Run

```bash
cargo oxide run async_mlp
```

## Expected Output

```text
=== Async MLP Pipeline ===

Allocating model weights...
  W0: 64x64 on device (Arc refcount=1)
  W1: 64 on device (Arc refcount=1)

Launched 4 batches concurrently, awaiting results...

Batch 0: 64 elements, first 8 = [0.0020799995, 0.0, 0.0, 0.0, 0.0, 0.00108, 0.00244, 0.0025] [ReLU OK]
Batch 1: 64 elements, first 8 = [0.0, 0.0, 0.0, 0.00108, 0.00244, 0.0025, 0.00035000034, 0.0] [ReLU OK]
Batch 2: 64 elements, first 8 = [0.0, 0.00108, 0.00244, 0.0025, 0.00035000034, 0.0, 0.0, 0.0] [ReLU OK]
Batch 3: 64 elements, first 8 = [0.00244, 0.0025, 0.00035000034, 0.0, 0.0, 0.0, 0.0014999998, 0.0020799995] [ReLU OK]

SUCCESS: All batches completed.
```

## Hardware Requirements

- **Minimum GPU**: Any CUDA-capable GPU
- **CUDA Driver**: 11.0+
- **Memory**: < 1 MB (64×64 matrices)

## Type Annotation Notes

The deeply nested generic types produced by `zip!` + `and_then` chains exceed
what Rust's type inference can resolve automatically. Explicit type annotations
are required on `and_then` closure parameters:

```rust
.and_then(move |(hidden, output, w1, module): (
    DeviceBox<[f32]>,
    DeviceBox<[f32]>,
    Arc<DeviceBox<[f32]>>,
    Arc<CudaModule>,
)| { ... })
```

The `Zippable` trait must also be explicitly imported for `zip!` to work:

```rust
use cuda_async::simt::device_operation::Zippable;
```
