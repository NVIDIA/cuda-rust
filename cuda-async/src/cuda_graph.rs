/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::device_context::with_default_device_policy;
use crate::device_future::DeviceFuture;
use crate::device_operation::{DeviceOp, ExecutionContext, GraphNode, ReplayResource};
use crate::error::DeviceError;
use cuda_core::{sys, Device, IntoResult, Stream};
use std::collections::HashMap;
use std::future::IntoFuture;
use std::mem::MaybeUninit;
use std::sync::{Arc, Mutex, OnceLock};

const CU_STREAM_CAPTURE_MODE_RELAXED: sys::CUstreamCaptureMode = 2;

/// A captured and instantiated CUDA graph, ready for replay.
///
/// Created via [`CudaGraph::capture`], which runs a [`DeviceOp`] once on a
/// capture stream, recording all GPU work into a graph. The graph can then
/// be replayed any number of times via [`launch`](CudaGraph::launch).
///
/// All device pointers used by the operation are baked into the graph at capture
/// time. To vary inputs between replays, pre-allocate an input buffer, pass it
/// into the operation, and memcpy new data into that buffer before each launch.
///
/// # Replay safety
///
/// A replay dereferences the device pointers recorded at capture time long
/// after the capturing code has returned. Two runtime mechanisms keep that
/// sound without tying the graph to the buffers by borrow:
///
/// - **Recorded storage stays alive.** Every device access made while
///   recording registers the buffer's storage with the graph, so a captured
///   buffer whose last host handle is dropped is not freed while the graph
///   can still replay. Only the handle is gone; the data is unreachable until
///   the graph itself is dropped.
/// - **Each replay re-leases its accesses.** A launch registers the graph and
///   one access lease per distinct captured storage on its own submission. A
///   replay that conflicts with an in-flight access from another stream is
///   rejected before it is enqueued, and an operation on another stream that
///   conflicts with an in-flight replay is rejected the same way. Work on the
///   same stream stays stream-ordered. A launch future forgotten mid-flight
///   leaks the graph and those leases rather than releasing them early.
///
/// What the runtime does not do is order host-side reads for you. The output
/// `T` from [`take_output`](CudaGraph::take_output) refers to buffers whose
/// contents exist only once a replay has completed; read them through a
/// device operation ordered after the launch, on the same stream or after
/// synchronizing the launch stream.
///
/// # Examples
///
/// ```rust,ignore
/// use cuda_async::prelude::*;
///
/// // Build a lazy operation (no GPU work yet).
/// let forward_op = build_forward_pass(&model, &bufs);
///
/// // Capture: records the op's GPU work into a graph. Nothing has run yet.
/// let mut graph = CudaGraph::capture(stream.clone(), forward_op)?;
/// let bufs = graph.take_output().unwrap();
///
/// // Replay loop.
/// for _ in 0..n_tokens {
///     // Optionally: copy new input into a pre-allocated buffer here.
///     graph.launch().sync_on(&stream)?;
/// }
/// ```
pub struct CudaGraph<T> {
    stream: Arc<Stream>,
    exec: Arc<GraphExecHandle>,
    output: Option<T>,
}

/// Owns an instantiated CUDA graph: the `CUgraph` it was instantiated from
/// and the `CUgraphExec` that replays it.
///
/// Shared (via `Arc`) between the [`CudaGraph`] and every [`GraphLaunch`] it
/// hands out, so a launch can never replay an exec that has already been
/// destroyed: the handles live until the last owner — graph or pending
/// launch — is dropped. `GraphLaunch` used to copy the raw `CUgraphExec`, so
/// `let l = graph.launch(); drop(graph); l.sync()` launched a destroyed exec.
struct GraphExecHandle {
    /// The capture context: keeps host-side owners the recorded work needs
    /// (pinned host buffers, ...) alive for as long as the graph can replay.
    capture_ctx: OnceLock<ExecutionContext>,
    /// The accesses the recorded work makes, one per distinct storage (a
    /// write subsumes reads), reacquired on every replay. Frozen at capture.
    captured: OnceLock<Vec<Arc<dyn ReplayResource>>>,
    /// Contexts of `update` ops not yet consumed by a replay.
    pending: Mutex<Vec<ExecutionContext>>,
    /// Bound before destruction: the driver needs the owning context current.
    device: Arc<Device>,
    cu_graph: sys::CUgraph,
    cu_graph_exec: sys::CUgraphExec,
}

// SAFETY: both fields are opaque driver handles that the CUDA driver
// synchronizes internally; `cuGraphLaunch` may be issued from any thread. The
// only mutation is the destroy in `Drop`, which runs exactly once, after the
// last `Arc` owner is gone.
unsafe impl Send for GraphExecHandle {}
unsafe impl Sync for GraphExecHandle {}

impl GraphExecHandle {
    /// Instantiates `cu_graph` and pre-uploads the exec on `stream`. Takes
    /// ownership of `cu_graph` on every path: it is destroyed if
    /// instantiation fails, and owned by the returned handle otherwise
    /// (whose `Drop` also covers an upload failure).
    fn instantiate(
        device: Arc<Device>,
        cu_graph: sys::CUgraph,
        stream: &Stream,
    ) -> Result<Self, DeviceError> {
        let cu_graph_exec = unsafe {
            let mut cu_graph_exec = MaybeUninit::<sys::CUgraphExec>::uninit();
            match sys::cuGraphInstantiateWithFlags(cu_graph_exec.as_mut_ptr(), cu_graph, 0).result()
            {
                Ok(()) => cu_graph_exec.assume_init(),
                Err(e) => {
                    let _ = sys::cuGraphDestroy(cu_graph).result();
                    return Err(DeviceError::Driver(e));
                }
            }
        };
        let handle = Self {
            capture_ctx: OnceLock::new(),
            captured: OnceLock::new(),
            pending: Mutex::new(Vec::new()),
            device,
            cu_graph,
            cu_graph_exec,
        };
        // Upload (pre-stages graph resources on the device). On failure
        // `handle` drops here and destroys both objects.
        unsafe { sys::cuGraphUpload(handle.cu_graph_exec, stream.cu_stream()).result()? };
        Ok(handle)
    }

    /// Takes ownership of the capture context and its recorded resources.
    /// Resources are deduplicated by storage so a replay reacquires each
    /// buffer once, however many recorded nodes touch it.
    fn adopt_capture(&self, ctx: ExecutionContext) {
        let recorded = ctx.take_recorded();
        let mut index: HashMap<usize, usize> = HashMap::with_capacity(recorded.len());
        let mut captured: Vec<Arc<dyn ReplayResource>> = Vec::with_capacity(recorded.len());
        for resource in recorded {
            let (storage, write) = resource.replay_identity();
            match index.get(&storage) {
                Some(&i) => {
                    if write && !captured[i].replay_identity().1 {
                        captured[i] = resource;
                    }
                }
                None => {
                    index.insert(storage, captured.len());
                    captured.push(resource);
                }
            }
        }
        let _ = self.captured.set(captured);
        let _ = self.capture_ctx.set(ctx);
    }
}

impl Drop for GraphExecHandle {
    fn drop(&mut self) {
        let _ = self.device.bind_to_thread();
        if !self.cu_graph_exec.is_null() {
            let _ = unsafe { sys::cuGraphExecDestroy(self.cu_graph_exec).result() };
        }
        if !self.cu_graph.is_null() {
            let _ = unsafe { sys::cuGraphDestroy(self.cu_graph).result() };
        }
    }
}

/// Runs `record` with `stream` in (relaxed) capture mode and turns the
/// recorded work into an instantiated, uploaded graph.
///
/// `cuStreamEndCapture` runs on every path — success, a `record` error, and
/// (before the panic resumes) a panic inside `record` — so a failure can
/// never leave the stream stuck in capture mode, and a graph handle produced
/// on a failing path is destroyed rather than leaked.
///
/// The caller must hold the execution lock: recording executes a `DeviceOp`.
fn capture_on<R>(
    stream: &Arc<Stream>,
    record: impl FnOnce() -> Result<R, DeviceError>,
) -> Result<(R, Arc<GraphExecHandle>), DeviceError> {
    let device = stream.device().clone();
    device.bind_to_thread()?;

    unsafe {
        stream.begin_capture(CU_STREAM_CAPTURE_MODE_RELAXED)?;
    }

    let recorded = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(record)) {
        Ok(result) => result,
        Err(payload) => {
            if let Ok(cu_graph) = unsafe { stream.end_capture() } {
                destroy_graph(cu_graph);
            }
            std::panic::resume_unwind(payload);
        }
    };

    // End capture — must happen regardless of the record result.
    let end_result = unsafe { stream.end_capture() };

    // Handle the (recorded, end_result) matrix, cleaning up on failure.
    let (output, cu_graph) = match (recorded, end_result) {
        (Err(err), Ok(cu_graph)) => {
            destroy_graph(cu_graph);
            return Err(err);
        }
        (Err(err), Err(_)) => return Err(err),
        (Ok(_), Err(capture_err)) => return Err(DeviceError::Driver(capture_err)),
        (Ok(_), Ok(cu_graph)) if cu_graph.is_null() => {
            return Err(DeviceError::Internal(
                "cuStreamEndCapture returned null graph".into(),
            ));
        }
        (Ok(output), Ok(cu_graph)) => (output, cu_graph),
    };

    let exec = GraphExecHandle::instantiate(device, cu_graph, stream)?;

    // Drain the upload. The recorded work itself has *not* run.
    unsafe { stream.synchronize() }?;

    Ok((output, Arc::new(exec)))
}

fn destroy_graph(cu_graph: sys::CUgraph) {
    if !cu_graph.is_null() {
        let _ = unsafe { sys::cuGraphDestroy(cu_graph).result() };
    }
}

impl<T: Send> CudaGraph<T> {
    /// Capture a [`DeviceOp`] into a replayable CUDA graph.
    ///
    /// Runs `op` once on `stream` in capture mode. All GPU work (kernel
    /// launches, memcpys, etc.) issued by the operation is *recorded* into a
    /// graph, not executed. The graph is then instantiated and uploaded.
    ///
    /// The output `T` is available immediately via
    /// [`take_output`](CudaGraph::take_output), but only its host-side
    /// metadata (shapes, device pointers, handles) is meaningful: the GPU
    /// data behind it is first computed when the graph is launched. Read it
    /// after `graph.launch().sync_on(graph.stream())`.
    ///
    /// Capture holds the thread-local execution lock for the duration of
    /// `op`, so nested `.sync()` / `.sync_on()` / `.await` inside it return
    /// the non-reentrant error (see [`Scope`]). If `op` fails or panics, the
    /// capture is ended and the stream left usable before the error or panic
    /// propagates.
    pub fn capture(
        stream: Arc<Stream>,
        op: impl DeviceOp<Output = T>,
    ) -> Result<Self, DeviceError> {
        let _execution_lock = crate::device_operation::acquire_execution_lock()?;
        let exec_ctx = ExecutionContext::for_capture(stream.clone());
        let (output, exec) = capture_on(&stream, || unsafe { op.execute(&exec_ctx) })?;
        exec.adopt_capture(exec_ctx);
        Ok(Self {
            stream,
            exec,
            output: Some(output),
        })
    }

    /// Take the output produced during the capture execution.
    ///
    /// Returns `Some(T)` on the first call, `None` thereafter. Use this to
    /// recover intermediate buffers or inspect the initial result.
    ///
    /// The handle is valid immediately (shapes, strides, device pointers),
    /// but the buffers behind it hold meaningful data only after a replay has
    /// completed. The graph keeps their storage alive independently of this
    /// handle; see [Replay safety](CudaGraph#replay-safety).
    pub fn take_output(&mut self) -> Option<T> {
        self.output.take()
    }

    /// Enqueue a [`GraphNode`] on the graph's stream without synchronizing.
    ///
    /// Use this to refresh graph inputs — typically a memcpy into a
    /// pre-allocated buffer the graph reads — before [`launch`](CudaGraph::launch).
    ///
    /// # Ordering
    ///
    /// The operation is issued on [`stream`](CudaGraph::stream), so it
    /// completes before any *later work on that same stream*. That includes
    /// a launch executed on the graph's stream:
    /// `graph.launch().sync_on(graph.stream())`. A launch scheduled anywhere
    /// else — `.sync()`, `.await`, or `.sync_on(&other_stream)` — is **not**
    /// ordered after the update.
    ///
    /// # Why the bounds
    ///
    /// `update` runs a `DeviceOp` without waiting for it, which is only
    /// sound when nothing can observe the operation early: the op must be a
    /// [`GraphNode`] (it neither allocates nor frees device memory) and must
    /// produce no output. Ops with an output, or ops that are not
    /// `GraphNode`, are rejected at compile time:
    ///
    /// ```compile_fail,E0271
    /// use cuda_async::cuda_graph::CudaGraph;
    /// use cuda_async::device_operation::value;
    ///
    /// fn reject_output(graph: &CudaGraph<()>) {
    ///     let _ = graph.update(value(5));
    /// }
    /// ```
    ///
    /// ```compile_fail,E0277
    /// use cuda_async::cuda_graph::CudaGraph;
    /// use cuda_async::device_operation::{value, DeviceOp};
    ///
    /// fn reject_non_graph_node(graph: &CudaGraph<()>) {
    ///     let _ = graph.update(value(()).boxed());
    /// }
    /// ```
    ///
    /// Like `sync_on`, this takes the thread-local execution lock.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // Copy a new embedding into the graph's pre-allocated input buffer.
    /// graph.update(api::memcpy(&mut h_input, &new_embedding))?;
    /// graph.launch().sync_on(graph.stream())?;
    /// ```
    pub fn update<N>(&self, op: N) -> Result<(), DeviceError>
    where
        N: GraphNode + DeviceOp<Output = ()>,
    {
        let _execution_lock = crate::device_operation::acquire_execution_lock()?;
        let ctx = ExecutionContext::new(self.stream.clone());
        // The context retains submitted resources and access leases even
        // after the op's Rust borrows end. A replay takes ownership of pending
        // update contexts; otherwise the graph keeps them until destruction.
        let result = unsafe { op.execute(&ctx) };
        self.exec
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(ctx);
        result
    }

    /// Return a [`DeviceOp`] that replays the captured graph.
    ///
    /// The graph launches on whichever stream the returned op is executed
    /// on. Use the standard [`DeviceOp`] methods to control execution:
    ///
    /// ```rust,ignore
    /// graph.launch().sync_on(&stream)?;          // explicit stream, blocking
    /// graph.launch().sync()?;                    // default policy, blocking
    /// graph.launch().then(next_op).sync()?;      // compose with other ops
    /// ```
    ///
    /// The launch shares ownership of the instantiated graph, so it stays
    /// valid even if the `CudaGraph` is dropped first. When executed, it
    /// registers the graph and a lease on every captured buffer with its
    /// submission, so a conflicting in-flight access on another stream makes
    /// the launch fail instead of racing; see
    /// [Replay safety](CudaGraph#replay-safety).
    ///
    /// Operations issued via [`update`](CudaGraph::update) are guaranteed to
    /// complete before the graph runs **only when the launch is executed on
    /// [`stream`](CudaGraph::stream)** (same-stream ordering); on any other
    /// stream there is no ordering between them.
    pub fn launch(&self) -> GraphLaunch {
        GraphLaunch {
            exec: Arc::clone(&self.exec),
        }
    }

    /// Returns a reference to the stream this graph was captured on.
    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }
}

/// A [`DeviceOp`] that replays a captured CUDA graph.
///
/// Created by [`CudaGraph::launch`]. The graph executes on whichever stream
/// the op is scheduled on (via `.sync_on(&stream)`, `.sync()`, or `.await`).
/// Holds a shared reference to the instantiated graph, so it may outlive the
/// [`CudaGraph`] that created it.
pub struct GraphLaunch {
    exec: Arc<GraphExecHandle>,
}

impl DeviceOp for GraphLaunch {
    type Output = ();

    unsafe fn execute(self, context: &ExecutionContext) -> Result<(), DeviceError> {
        context.retain(self.exec.clone())?;
        // Reacquire every captured buffer for this replay: one lease per
        // distinct storage, registered on the launch submission so other
        // streams see the replay's in-flight accesses.
        if let Some(captured) = self.exec.captured.get() {
            for resource in captured {
                resource.retain_for_launch(context)?;
            }
        }
        // Pending updates need not accumulate over the graph's lifetime.
        // The launch submission now owns them; their contexts wait on the
        // update stream if it differs from the launch stream.
        let pending =
            std::mem::take(&mut *self.exec.pending.lock().unwrap_or_else(|e| e.into_inner()));
        for ctx in pending {
            context.retain(ctx)?;
        }
        sys::cuGraphLaunch(
            self.exec.cu_graph_exec,
            context.get_cuda_stream().cu_stream(),
        )
        .result()?;
        Ok(())
    }
}

impl IntoFuture for GraphLaunch {
    type Output = Result<(), DeviceError>;
    type IntoFuture = DeviceFuture<(), GraphLaunch>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| {
            let stream = policy.next_stream()?;
            let mut f = DeviceFuture::new();
            f.device_operation = Some(self);
            f.execution_context = Some(ExecutionContext::new(stream));
            Ok(f)
        }) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) => DeviceFuture::failed(e),
            Err(e) => DeviceFuture::failed(e),
        }
    }
}

/// A scope for recording GPU operations into a CUDA graph.
///
/// Created by [`CudaGraph::scope`]. Each call to [`record`](Scope::record)
/// records a [`GraphNode`] as a graph node. The op is consumed immediately,
/// releasing any borrows it holds. This means a buffer written by one
/// kernel can be read by the next — `record` releases the `&mut` borrow,
/// allowing a subsequent `record` to take `&` on the same buffer.
///
/// ```rust,ignore
/// let graph = CudaGraph::scope(&stream, |s| {
///     s.record(rms_norm((&mut bufs.norm).partition([1, d]), &input, &w))?;
///     // bufs.norm borrow released — can now read it:
///     s.record(matvec((&mut bufs.q).partition([bn]), &bufs.norm, &wq))?;
///     Ok(())
/// })?;
///
/// graph.launch().sync_on(&stream)?;
/// ```
///
/// # Safety proof: why `record` is safe
///
/// A CUDA data race occurs when two accesses to the same device memory
/// are unordered and at least one is a write. This is UB per both CUDA
/// and Rust.
///
/// `record` is safe because of two complementary mechanisms:
///
/// ## Capture mode prevents concurrent GPU execution
///
/// The scope's stream is in **capture mode** during the closure (via
/// `cuStreamBeginCapture`). In capture mode:
///
/// 1. **No GPU work executes.** `record` records operations as graph
///    nodes — kernels are not launched, memcpys are not issued. There
///    is no in-flight GPU work that could race with anything.
///
/// 2. **Same-stream ordering is preserved.** All `record` calls go to
///    the same capture stream. When the graph is later launched via
///    [`CudaGraph::launch`], the nodes execute in recorded order on a
///    single stream. Sequential same-stream execution is ordered — no
///    data races.
///
/// 3. **Other executions inside the closure are rejected.** The scope
///    holds the thread-local execution lock for the duration of the
///    closure, so `op.sync_on(&other_stream)`, `op.sync()`, and
///    `op.sync_on(&capture_stream)` all return the non-reentrant
///    `DeviceError` before reaching the driver. Nothing executes eagerly
///    alongside the recording. (Independently, CUDA itself rejects
///    synchronizing or querying a capturing stream with
///    `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`.) The one way around the
///    lock is a [`then_unchecked`](DeviceOp::then_unchecked) chain recorded
///    via `record`, whose closure may execute eagerly — that is exactly the
///    caller's `unsafe` assertion.
///
/// 4. **Borrow checker enforces `&mut` exclusivity.** `record` consumes
///    the op, releasing `&mut`. The next `record` can then borrow the
///    same buffer as `&` for reading.
///
/// ## `GraphNode` prevents allocation during capture
///
/// `record` accepts [`GraphNode`] (not [`DeviceOp`]). `GraphNode` is only
/// implemented by operations that do not allocate or free device memory
/// (kernel launches, `memcpy`, `value`). This prevents:
///
/// - **Address instability:** `cuMemAllocAsync` during capture allocates
///   memory, but on replay the allocation node may return a different
///   address. Subsequent nodes bake in the capture-time pointer — UB.
///
/// - **Uninitialized reads:** An allocation during capture gives the user
///   a tensor handle. The initialization (e.g., memset from `zeros`) was
///   recorded, not executed. Passing the tensor to `sync_on(&other_stream)`
///   reads uninitialized memory.
///
/// - **Invalid frees:** If a tensor allocated inside the scope is dropped,
///   `cuMemFreeAsync` is recorded. On replay, it frees the capture-time
///   address, which may no longer be valid.
///
/// Since no tensors can be allocated inside the scope, all buffers are
/// pre-allocated and passed in via borrows. No tensor created inside
/// the scope means no tensor dropped inside the scope.
///
/// ## Replay safety after capture
///
/// The two mechanisms above make *recording* safe. Replays happen later,
/// after `record`'s borrows have ended, and are covered by the runtime
/// contract described under [`CudaGraph`](CudaGraph#replay-safety): every
/// storage recorded here stays alive while the graph can replay, and each
/// launch re-leases those accesses, so a conflicting in-flight access on
/// another stream is rejected rather than raced. Dropping a buffer's host
/// handle after the scope therefore frees nothing the graph still uses; it
/// only makes the data unreachable, so keep a handle to whatever you need to
/// read back or refresh with [`update`](CudaGraph::update).
///
/// # What happens if you call other operations inside the closure
///
/// While `s.record(op)` is the intended API, other operations inside
/// the closure have well-defined behavior:
///
/// | Operation | What happens |
/// |---|---|
/// | `op.sync_on(&capture_stream)` | Non-reentrant execution-lock error; nothing executes |
/// | `op.sync_on(&other_stream)` | Non-reentrant execution-lock error; nothing executes |
/// | `op.sync()` / `op.await` | Non-reentrant execution-lock error; nothing executes |
///
/// These are all defined behavior but serve no purpose inside a graph
/// capture scope — use `s.record(op)` instead.
///
/// # Thread safety
///
/// `Scope` is `!Send` — it cannot escape to another thread.
pub struct Scope {
    ctx: ExecutionContext,
    _not_send: std::marker::PhantomData<*const ()>,
}

impl Scope {
    /// Record a [`GraphNode`] into the graph being captured.
    ///
    /// The op is consumed, recording its GPU work (kernel launch, memcpy)
    /// as a graph node. Any borrows held by the op are released when this
    /// call returns. The return value contains valid metadata (tensor
    /// shapes, device pointers) but GPU data is not yet computed — the
    /// actual computation happens when the graph is replayed via
    /// [`CudaGraph::launch`].
    ///
    /// Only operations that implement [`GraphNode`] can be recorded.
    /// This excludes allocation ops (`zeros`, `ones`, `dup`, etc.)
    /// whose addresses may change on replay.
    pub fn record<T, N>(&self, op: N) -> Result<T, DeviceError>
    where
        T: Send,
        N: GraphNode + DeviceOp<Output = T>,
    {
        // SAFETY: The scope's stream is in capture mode. No GPU work
        // executes — ops are recorded as graph nodes. The GraphNode bound
        // ensures no alloc/free ops are recorded. See Scope docs for
        // the full safety proof.
        unsafe { op.execute(&self.ctx) }
    }
}

impl CudaGraph<()> {
    /// Capture a CUDA graph using a scoped closure.
    ///
    /// The closure receives a [`Scope`] for recording operations. Each
    /// `s.record(op)` records a graph node and consumes the op, releasing
    /// borrows. A buffer written by one `record` call can be read by the
    /// next.
    ///
    /// Pre-allocate all buffers before calling this method — the graph
    /// replays into the same device pointers. The graph retains the storage
    /// of every buffer it records, so a buffer you will not read or refresh
    /// again may be dropped after capture without affecting later replays.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut output = api::zeros::<f32>(&[d]).sync_on(&stream)?;
    /// let weights = api::ones::<f32>(&[d]).sync_on(&stream)?;
    ///
    /// let graph = CudaGraph::scope(&stream, |s| {
    ///     s.record(kernel1((&mut output).partition([128]), &weights))?;
    ///     s.record(kernel2((&mut output).partition([64]), &weights))?;
    ///     Ok(())
    /// })?;
    ///
    /// graph.launch().sync_on(&stream)?;
    /// ```
    ///
    /// See [`Scope`] for the safety proof and edge-case behavior.
    pub fn scope<F>(stream: &Arc<Stream>, f: F) -> Result<Self, DeviceError>
    where
        F: FnOnce(&Scope) -> Result<(), DeviceError>,
    {
        // The guard releases the lock on return and on unwind; `capture_on`
        // ends the capture itself before a panic propagates.
        let _execution_lock = crate::device_operation::acquire_execution_lock()?;
        let scope = Scope {
            ctx: ExecutionContext::for_capture(stream.clone()),
            _not_send: std::marker::PhantomData,
        };
        let ((), exec) = capture_on(stream, || f(&scope))?;
        exec.adopt_capture(scope.ctx);
        Ok(CudaGraph {
            stream: stream.clone(),
            exec,
            output: Some(()),
        })
    }
}

/// A graph-backed inference module.
///
/// Implementations own a [`CudaGraph`] captured at construction time.
/// Each call to [`forward`](Module::forward) updates the input buffer and
/// replays the graph, returning the result synchronously.
///
/// # Construction
///
/// Graph capture is model-specific and happens in the implementation's
/// constructor — not in the trait. A typical pattern:
///
/// ```rust,ignore
/// use cuda_async::prelude::*;
///
/// struct MyModel {
///     graph: CudaGraph<Arc<Tensor<f32>>>,
///     h_input: Tensor<f32>,
///     output: Arc<Tensor<f32>>,
/// }
///
/// impl MyModel {
///     fn new(stream: Arc<Stream>) -> Result<Self, DeviceError> {
///         let h_input = api::zeros(&[d]).sync_on(&stream)?;
///         let forward_op = build_forward(h_input.clone().into());
///         let mut graph = forward_op.graph_on(stream)?;
///         let output = graph.take_output().unwrap();
///         Ok(Self { graph, h_input, output })
///     }
/// }
///
/// impl Module for MyModel {
///     type Input = Arc<Tensor<f32>>;
///     type Output = Arc<Tensor<f32>>;
///
///     fn forward(&mut self, input: Self::Input)
///         -> Result<Self::Output, DeviceError>
///     {
///         self.graph.update(
///             api::memcpy(&mut self.h_input, &input)
///         )?;
///         self.graph.launch().sync_on(self.graph.stream())?;
///         Ok(self.output.clone())
///     }
/// }
/// ```
///
/// # Future extensions
///
/// This trait covers the forward pass. Planned companion traits:
/// - `Backward` — gradient computation for autodiff
/// - `Parameterized` — access to learnable parameters for optimizers
pub trait Module {
    /// The input to the module (e.g., an embedding tensor).
    type Input: Send;
    /// The output of the module (e.g., logits or a hidden state).
    type Output: Send;

    /// Run the forward pass: update the input, launch the graph, return
    /// the result.
    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, DeviceError>;
}
