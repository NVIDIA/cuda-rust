# 10. CUDA Graphs

CUDA graphs let you capture an entire GPU workload once and replay it
many times, eliminating per-launch overhead. This tutorial builds a
multi-layer forward pass using DeviceOp combinators, captures it as
a CUDA graph, and replays it in a token loop.

---

## Motivation

Every kernel launch involves CPU-side work: selecting a stream, setting up
arguments, invoking the driver. For workloads that repeat the same graph of
operations (e.g., the forward pass of a transformer), this per-launch
overhead can dominate — especially at small batch sizes where kernels are
fast relative to their launch cost.

A CUDA graph records the entire sequence of operations once, then replays
it with a single driver call. The GPU sees the full graph up front and can
schedule internal work more aggressively.

```text
Without graphs:                     With graphs:

  CPU: launch → wait → launch →      CPU: launch_graph → wait
       wait → launch → wait               (single call)
  GPU: ████   ████   ████            GPU: ████████████████
       gaps between kernels                no gaps
```

---

## The Model

We'll build a minimal transformer-style layer stack: each layer performs
RMSNorm → Q projection (matvec) → O projection (matvec) → residual add.
The hidden state flows through all layers sequentially.

```text
input
  │
  ├─ Layer 0: RMSNorm → Q MatVec → O MatVec → Add(residual, hidden)
  │
  ├─ Layer 1: RMSNorm → Q MatVec → O MatVec → Add(residual, hidden)
  │
  └─ … (n_layers)
```

### Kernels

Three cutile kernels handle the compute. Each follows the output-first
convention (`&mut Tensor` as the first parameter):

```rust
#[cutile::module]
mod kernels {
    use cutile::core::*;

    /// RMS normalization: out = rms_norm(x) * w
    #[cutile::entry()]
    pub fn rms_norm<const D: i32, const BS: i32>(
        out: &mut Tensor<f32, { [1, D] }>,
        x: &Tensor<f32, { [-1, D] }>,
        w: &Tensor<f32, { [D] }>,
        eps: f32,
    ) { /* tile-level implementation */ }

    /// Matrix-vector multiply: out = x @ w^T
    #[cutile::entry()]
    pub fn matvec<const BN: i32, const BK: i32, const K: i32>(
        out: &mut Tensor<f32, { [BN] }>,
        x: &Tensor<f32, { [-1, K] }>,
        w: &Tensor<f32, { [-1, K] }>,
    ) { /* tile-level implementation */ }

    /// Element-wise add: out = a + b
    #[cutile::entry()]
    pub fn add<const B: i32>(
        out: &mut Tensor<f32, { [B] }>,
        a: &Tensor<f32, { [-1] }>,
        b: &Tensor<f32, { [-1] }>,
    ) { /* tile-level implementation */ }
}
```

### Model State

Weights are shared across all forward calls. Buffers are pre-allocated once
and reused every token — the graph replays into the same memory:

```rust
struct LayerWeights {
    norm_w: Arc<Tensor<f32>>,
    wq: Arc<Tensor<f32>>,
    wo: Arc<Tensor<f32>>,
}

struct LayerBuffers {
    norm: Arc<Tensor<f32>>,
    q: Arc<Tensor<f32>>,
    o: Arc<Tensor<f32>>,
    residual: Arc<Tensor<f32>>,
}
```

---

## Building the Lazy Graph

Build the forward pass as a lazy `DeviceOp` for graph capture:

```rust
fn build_forward(
    cfg: &Config,
    weights: &[LayerWeights],
    input: Arc<Tensor<f32>>,
    buffers: Vec<LayerBuffers>,
) -> (impl DeviceOp<Output = ()>, SharedDeviceOp<Tensor<f32>>) {
    let mut ops: Vec<BoxedDeviceOp<()>> = Vec::with_capacity(buffers.len());
    let mut hidden: SharedDeviceOp<Tensor<f32>> =
        cuda_async::device_operation::shared(input);

    for (w, bufs) in weights.iter().zip(buffers) {
        // RMSNorm: hidden(1,d) × norm_w → norm(1,d)
        let norm = rms_norm(
            bufs.norm.try_partition([1, cfg.d]).expect("sole buffer owner"),
            hidden.clone().reshape(&[1, cfg.d]),
            w.norm_w.clone(),
            cfg.eps,
        )
        .generics(cfg.rms_generics())
        .first()
        .unpartition()
        .shared();

        // Q projection: norm @ wq^T → q
        let q = matvec(
            bufs.q.try_partition([cfg.bn]).expect("sole buffer owner"),
            norm.clone(),
            w.wq.clone(),
        )
        .generics(cfg.mv_generics())
        .first()
        .unpartition()
        .shared();

        // O projection: q @ wo^T → o
        let o = matvec(
            bufs.o.try_partition([cfg.bn]).expect("sole buffer owner"),
            q.clone().reshape(&[1, cfg.d]),
            w.wo.clone(),
        )
        .generics(cfg.mv_generics())
        .first()
        .unpartition()
        .shared();

        // Residual add: hidden + o → residual
        let residual = add(
            bufs.residual.try_partition([cfg.block]).expect("sole buffer owner"),
            hidden.clone().reshape(&[cfg.d]),
            o.clone(),
        )
        .first()
        .unpartition()
        .shared();

        hidden = residual.clone();

        // Keep the layer's work in the graph; the fixed buffers hold the data.
        ops.push(
            zip!(norm, q, o, residual)
                .map(|_| ())
                .boxed(),
        );
    }

    // DeviceOpVec produces Vec<()>; discard it so the capture yields CudaGraph<()>.
    (DeviceOpVec::new(ops).map(|_| ()), hidden)
}
```

The combinators in `build_forward` have these roles:

- **`.shared()`** — Each intermediate result is shared so it can feed into
  both the next kernel and the final buffer collection. The underlying
  computation runs once; downstream consumers get `Arc::clone()`.
- **`.first()`** — Kernel launches return a tuple of all arguments.
  `.first()` extracts just the output (the `&mut Tensor` parameter).
- **`try_partition`** — Converts `Arc<Tensor<T>>` into a `Partition` by
  proving sole ownership (Arc refcount == 1). It returns `Err` when the Arc
  has other owners; the `.expect(...)` above turns that into a panic.
  (`shared(...)` is `cuda_async::device_operation::shared`; it is not in the
  prelude.)
- **`DeviceOpVec`** — Collects boxed ops for each layer's graph work. Its
  output is `Vec<()>`, so the trailing `.map(|_| ())` is what makes the
  captured graph a `CudaGraph<()>`.

---

## Capturing the Graph

`.graph_on(stream)` executes the operation once in CUDA's stream capture mode,
recording all GPU work into a replayable graph:

```rust
let input: Tensor<f32> = api::rand([cfg.d], None).sync_on(&stream)?;
let input_arc: Arc<Tensor<f32>> = unsafe { input.into_shared_alias() };
let buffers: Vec<_> = (0..cfg.n_layers)
    .map(|_| LayerBuffers::allocate(cfg.d, &stream))
    .collect::<Result<_, _>>()?;
unsafe { stream.synchronize() }?;

// Build lazy graph (no GPU work).
let (forward_op, output_shared) = build_forward(&cfg, &weights, input_arc, buffers);

// Capture: executes once, records everything, returns CudaGraph.
let mut graph = forward_op.graph_on(stream.clone())?;

// Retrieve the output from the capture execution.
let output = output_shared.sync_on(&stream)?;
```

After capture:
- `graph` holds the recorded CUDA graph, ready for replay.
- `input` is the fixed input buffer whose pointer is baked into the graph.
- `output` points to the final layer's residual buffer, which the graph rewrites on replay.

---

## The Module Pattern

Implement `Module` to call the graph through `forward`:

```rust
trait Module {
    type Input: Send + ?Sized;
    type Output: Send;
    fn forward(&mut self, input: &Self::Input) -> Result<Self::Output, DeviceError>;
}

struct GraphModel {
    graph: CudaGraph<()>,
    input: Tensor<f32>,
    output: Arc<Tensor<f32>>,
}

impl Module for GraphModel {
    type Input = Tensor<f32>;
    type Output = Arc<Tensor<f32>>;

    fn forward(&mut self, input: &Self::Input) -> Result<Self::Output, DeviceError> {
        // Copy new embedding into the baked-in input buffer.
        self.graph.update(api::memcpy(&mut self.input, input))?;
        // Replay the entire forward pass with a single driver call.
        self.graph.launch().sync_on(self.graph.stream())?;
        Ok(self.output.clone())
    }
}
```

Each `forward` call:

1. **`graph.update(memcpy(…))`** — Copies new input data into the
   pre-allocated input buffer. This runs on the graph's stream, so it
   completes before the graph launches.
2. **`graph.launch().sync_on(…)`** — Replays all captured kernels. The GPU sees the
   full operation sequence and can schedule aggressively.
3. **Returns `output.clone()`** — The output `Arc` points to the same
   device memory the graph wrote into. No copy needed.

---

<a id="putting-it-together"></a>
## Inference loop

```rust
fn main() -> Result<(), Error> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;

    let cfg = Config { d: 2048, n_layers: 22, block: 128, bn: 16, bk: 16, eps: 1e-5 };

    // Allocate weights (random for this example).
    let weights: Vec<LayerWeights> = (0..cfg.n_layers)
        .map(|_| Ok(LayerWeights {
            norm_w: api::rand([cfg.d], None).sync_on(&stream)?.into(),
            wq: api::rand([cfg.d, cfg.d], None).sync_on(&stream)?.into(),
            wo: api::rand([cfg.d, cfg.d], None).sync_on(&stream)?.into(),
        }))
        .collect::<Result<_, Error>>()?;

    // Build and capture.
    let mut model = GraphModel::new(&cfg, &weights, &stream)?;

    // Inference loop — each call is a single graph launch.
    let n_tokens = 512;
    for _ in 0..n_tokens {
        let embedding: Tensor<f32> = api::rand([cfg.d], None).sync_on(&stream)?;
        let _output = model.forward(&embedding)?;
    }

    Ok(())
}
```

---

## Alternative: `CudaGraph::scope`

The combinator approach above requires `Arc<Tensor<T>>` + `try_partition`
for pre-allocated buffers. `CudaGraph::scope` provides an imperative
alternative using `&mut` borrows:

```rust
let mut input: Tensor<f32> = api::ones::<f32>(&[cfg.d]).sync_on(&stream)?;
let mut buffers: Vec<LayerBuffers> = /* pre-allocate */;

let graph = CudaGraph::scope(&stream, |s| {
    for (w, bufs) in weights.iter().zip(buffers.iter_mut()) {
        let hidden_2d = input.view(&[1, cfg.d])?;

        s.record(rms_norm(
            (&mut bufs.norm).partition([1, cfg.d]),
            &hidden_2d,
            &w.norm_w,
            cfg.eps,
        ).generics(cfg.rms_generics()))?;

        s.record(matvec(
            (&mut bufs.q).partition([cfg.bn]),
            &bufs.norm,
            &w.wq,
        ).generics(cfg.mv_generics()))?;

        let q_2d = bufs.q.view(&[1, cfg.d])?;
        s.record(matvec(
            (&mut bufs.o).partition([cfg.bn]),
            &q_2d,
            &w.wo,
        ).generics(cfg.mv_generics()))?;

        s.record(add(
            (&mut bufs.residual).partition([cfg.block]),
            &input,
            &bufs.o,
        ))?;

        s.record(api::memcpy(&mut input, &bufs.residual))?;
    }
    Ok(())
})?;

graph.launch().sync_on(&stream)?;
```

| | Combinator (`.graph()`) | Scope (`CudaGraph::scope`) |
|---|---|---|
| Buffer ownership | `Arc<Tensor<T>>` + `try_partition` | `&mut Tensor<T>` borrows |
| Write-then-read | Via `.shared()` + cloning | Via `record()` releasing borrows |
| Failure mode | Runtime `Err` from `try_partition` (refcount != 1) | Compile error (borrow conflict) |
| Composability | Chains with `.then()`, `.map()` | Imperative sequential code |

`s.record(op)` only accepts operations that implement `GraphNode` — kernel
launches and `memcpy`. Allocation ops (`api::zeros`, `dup`, etc.) are rejected
at compile time because their addresses may change on graph replay.

---

## Use cases

| Scenario | Use CUDA graphs? | Why |
|---|---|---|
| Repeat the same operation graph many times | **Yes** | Amortizes capture cost; eliminates per-launch overhead |
| Dynamic shapes per iteration | No | Captured graphs bake in tensor dimensions |
| Dynamic control flow per iteration | No | Captured graphs bake in the branch structure |
| Small number of iterations | Maybe | Capture cost (~1 execution) must be amortized |
| Profiling individual kernels | No | Graph replay shows as a single event |

---

## Key Takeaways

| Concept | What it means |
|---|---|
| **`DeviceOp` graph** | Lazy composition of GPU work — no execution until driven |
| **`.graph_on(stream)`** | Capture the entire operation into a replayable `CudaGraph` |
| **`CudaGraph::scope`** | Imperative graph capture with `&mut` borrows |
| **`graph.launch()`** | Returns a `DeviceOp` that replays all captured work — execute with `.sync_on()`, `.sync()`, or `.await` |
| **`graph.update(op)`** | Run a DeviceOp on the graph's stream before replay (e.g., memcpy new input) |
| **Pre-allocated buffers** | Graph writes into fixed memory; vary inputs via `memcpy` |
| **`.shared()` in graphs** | Each intermediate executes once during capture; clones share the result |

:::{note}
Weight tensors in `build_forward` are passed as `Arc<Tensor<T>>` because
they're shared across layers via `.clone()`. In a sync context where weights
are only read (not shared across spawned tasks), you could pass `&Tensor<T>`
instead — the borrow checker ensures the weights outlive the graph capture.
See [Tutorial 7](07-intro-to-async.md#borrowed-inputs-and-spawn-safety) for
details on borrowed inputs.
:::

---

### Exercise 1: Add a Second Graph

Capture a second graph that computes the backward pass (or a simplified
version). How would you sequence the forward and backward graphs?

### Exercise 2: Measure the Speedup

Add timing around `graph.launch()` vs a non-graph path that rebuilds and
executes the `DeviceOp` each iteration. How does the speedup scale with
`n_layers`?

### Exercise 3: Dynamic Input Shapes

The current approach bakes in the tensor dimensions at capture time. What
would need to change to support variable sequence lengths? (Hint: consider
capturing multiple graphs for common sizes.)

---

## Full Reference Examples

The scope-based reference example with benchmarks:

```bash
cargo run -p cutile-examples --example cuda_graphs
```

The DeviceOp-combinator variant:

```bash
cargo run -p cutile-examples --example cuda_graphs_deviceop
```

---

## See also

- [Device Operations](../guide/device-operations.md) — where CUDA graphs fit alongside sync and async execution
- [Host API: CUDA Graph Integration](../reference/host-api.md#cuda-graph-integration) — `.graph_on(stream)` and `CudaGraph::scope` signatures
