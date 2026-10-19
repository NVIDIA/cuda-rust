/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lazy, composable GPU operations and combinator types.

use crate::device_context::{pool_for_stream, with_default_device_policy};
use crate::device_future::DeviceFuture;
use crate::error::DeviceError;
use crate::scheduling_policies::SchedulingPolicy;
use cuda_core::{Device, Event, MemPool, Stream};
use std::cell::Cell;
use std::fmt::Debug;
use std::future::IntoFuture;
use std::marker::PhantomData;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::ThreadId;

// ── Thread-local execution guard ───────────────────────────────────────────
//
// Invariant: On any given thread, only one DeviceOp may be executing at a time.
//
// This prevents CUDA data races from nested execution (e.g., calling
// sync_on(&other_stream) inside a `then` closure with in-flight tensors).

thread_local! {
    static DEVICE_OP_EXECUTING: Cell<bool> = const { Cell::new(false) };
}

/// Ownership of the thread-local execution lock.
///
/// Released on drop, so every exit from an executing region — a normal
/// return, a `?` early return, or a panic unwinding out of a user closure —
/// gives the lock back. Before this guard existed, a panic inside `execute`
/// left `DEVICE_OP_EXECUTING` set and every later operation on the thread
/// failed with the non-reentrant error.
#[must_use = "dropping the guard immediately releases the execution lock"]
pub(crate) struct ExecutionLockGuard(());

impl Drop for ExecutionLockGuard {
    fn drop(&mut self) {
        DEVICE_OP_EXECUTING.with(|flag| flag.set(false));
    }
}

/// Acquire the thread-local execution lock. Returns an error if another
/// DeviceOp is already executing on this thread.
pub(crate) fn acquire_execution_lock() -> Result<ExecutionLockGuard, DeviceError> {
    DEVICE_OP_EXECUTING.with(|flag| {
        if flag.get() {
            Err(DeviceError::Internal(
                "DeviceOp execution is non-reentrant: another DeviceOp is already \
                 executing on this thread. If this is intentional and you have \
                 verified there are no cross-stream data races, use \
                 `then_unchecked`."
                    .into(),
            ))
        } else {
            flag.set(true);
            Ok(ExecutionLockGuard(()))
        }
    })
}

/// Temporarily gives up the execution lock for the scope of a
/// [`then_unchecked`](DeviceOp::then_unchecked) closure.
///
/// On construction the thread-local flag is cleared so the closure can run
/// `.sync()`, `.sync_on()`, or a nested `.await`; on drop (normal exit or
/// unwind) the flag is restored to whatever it was, so the enclosing chain
/// keeps holding the lock afterwards. Nested regions inside the closure use
/// their own [`ExecutionLockGuard`]s and never observe this one.
struct ExecutionLockRelease {
    was_held: bool,
}

impl ExecutionLockRelease {
    fn new() -> Self {
        Self {
            was_held: DEVICE_OP_EXECUTING.with(|flag| flag.replace(false)),
        }
    }
}

impl Drop for ExecutionLockRelease {
    fn drop(&mut self) {
        DEVICE_OP_EXECUTING.with(|flag| flag.set(self.was_held));
    }
}

pub type DeviceOrdinal = usize;

pub use crate::submission::{AccessLease, AccessTracked};

/// A graph-owned resource whose device accesses must be acquired on each replay.
/// Implementations must register the access with the supplied execution context.
#[doc(hidden)]
pub trait ReplayResource: Send + Sync {
    fn retain_for_launch(&self, ctx: &ExecutionContext) -> Result<(), DeviceError>;
    /// `(storage identity, writes)`: resources with the same identity are
    /// reacquired once per replay, a write subsuming reads.
    fn replay_identity(&self) -> (usize, bool);
}

#[derive(Debug, Clone)]
pub struct ExecutionContext {
    ordinal: DeviceOrdinal,
    cuda_stream: Arc<Stream>,
    device: Arc<Device>,
    pool: Option<Arc<MemPool>>,
    submission: Arc<crate::submission::Submission>,
    recording: bool,
}

impl ExecutionContext {
    pub fn new(cuda_stream: Arc<Stream>) -> Self {
        let device = cuda_stream.device().clone();
        let ordinal = device.ordinal();
        let pool = pool_for_stream(&cuda_stream);
        Self {
            recording: false,
            submission: Arc::new(crate::submission::Submission::new(cuda_stream.clone())),
            cuda_stream,
            device,
            ordinal,
            pool,
        }
    }
    pub fn get_cuda_stream(&self) -> &Arc<Stream> {
        &self.cuda_stream
    }
    /// Retain an owned resource until this submission completes. Register
    /// resources before enqueueing work, including resources absent from the
    /// operation's output. A forgotten future deliberately leaks these owners.
    ///
    /// Clones share the submission; a completed context cannot be reused.
    pub fn retain(&self, owner: impl Send + 'static) -> Result<(), DeviceError> {
        self.submission.retain(owner)
    }

    /// Retain an in-flight access lease until this submission completes.
    /// The allocation-free counterpart of [`retain`](Self::retain) for the
    /// per-argument hot path.
    #[doc(hidden)]
    pub fn retain_lease(&self, lease: AccessLease) -> Result<(), DeviceError> {
        self.submission.retain_lease(lease)
    }

    #[doc(hidden)]
    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// Keep graph storage alive without claiming that its recorded accesses
    /// have executed. The graph reacquires those accesses on each launch.
    #[doc(hidden)]
    pub fn record_resource(&self, resource: Arc<dyn ReplayResource>) {
        assert!(self.recording);
        self.submission.record(resource);
    }

    pub(crate) fn for_capture(stream: Arc<Stream>) -> Self {
        Self {
            recording: true,
            ..Self::new(stream)
        }
    }

    pub(crate) fn fresh_submission(&self) -> Self {
        Self {
            submission: Arc::new(crate::submission::Submission::new(self.cuda_stream.clone())),
            recording: false,
            ..self.clone()
        }
    }

    /// Drains the resources recorded during capture; the graph owns them
    /// from here on and reacquires them on each replay.
    pub(crate) fn take_recorded(&self) -> Vec<Arc<dyn ReplayResource>> {
        self.submission.take_recorded()
    }

    /// The caller has proved completion, or explicitly assumes responsibility
    /// for all submitted resources (the unsafe `async_on` escape hatch).
    pub(crate) unsafe fn complete(&self) {
        self.submission.complete();
    }
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }
    pub fn get_device_id(&self) -> DeviceOrdinal {
        self.ordinal
    }
    pub fn get_pool(&self) -> Option<&Arc<MemPool>> {
        self.pool.as_ref()
    }
    /// Allocates device memory on this context's stream, using the custom pool if set.
    ///
    /// # Safety
    /// The stream must be valid and not destroyed.
    ///
    /// Fails with the driver's own diagnosis, typically
    /// `CUDA_ERROR_OUT_OF_MEMORY`, which is not sticky: the context stays
    /// usable, so a caller may free memory and retry (with back-off, a bounded
    /// number of times, or by falling back to a smaller request).
    pub unsafe fn alloc_async(
        &self,
        num_bytes: usize,
    ) -> Result<cuda_core::sys::CUdeviceptr, DeviceError> {
        let allocated = match &self.pool {
            Some(pool) => cuda_core::malloc_from_pool_async(num_bytes, pool, &self.cuda_stream),
            None => cuda_core::malloc_async(num_bytes, &self.cuda_stream),
        };
        Ok(allocated?)
    }
    #[expect(
        dead_code,
        reason = "kept for direct synchronous execution in tests and future blocking APIs"
    )]
    fn execute<T: Send>(&self, op: impl DeviceOp<Output = T>) -> Result<T, DeviceError> {
        unsafe {
            // Safety: ExecutionContext is only available within a DeviceOp closure.
            // DeviceOp closures can only be converted into DeviceFuture
            // which synchronizes device operations with the host thread via a host callback.
            op.execute(self)
        }
    }
}

/// A lazy, composable GPU operation that may be executed synchronously or asynchronously on a CUDA device.
///
/// `DeviceOp` represents a resource-agnostic computation that will be scheduled and executed.
/// The actual execution resource (stream, device, host machine, cluster, etc.) is determined when the
/// operation is either executed or converted into a future.
/// Device operations are lazy - they don't execute until synchronously executed, or a corresponding
/// future is awaited upon. Multiple operations can be composed together before execution,
/// enabling efficient streaming of GPU work.
///
/// # Scheduling and Stream Assignment
///
/// How an operation reaches the GPU depends on which method you use:
///
/// | Method              | Stream chosen by                      | Blocks thread?      |
/// |---------------------|---------------------------------------|---------------------|
/// | `.await`            | Default device's [`SchedulingPolicy`] | No (suspends task)  |
/// | `.sync()`           | Default device's [`SchedulingPolicy`] | Yes                 |
/// | `.sync_on(&stream)` | The explicit `stream` you provide     | Yes                 |
/// | `.into_future()`    | Default device's [`SchedulingPolicy`] | No (returns future) |
/// | `.schedule(policy)` | The `policy` you provide              | No (returns future) |
///
/// With the default [`StreamPoolRoundRobin`](crate::scheduling_policies::StreamPoolRoundRobin) policy (4 streams), consecutive `.await` or
/// `.sync()` calls rotate through streams, so independent operations can overlap on the GPU.
/// Operations chained with [`.then()`](DeviceOp::then) share a single stream
/// and always execute in order.
///
/// See [`SchedulingPolicy`] for a full explanation of ordering guarantees.
///
/// # Safety
///
/// The `execute` method is unsafe because it's asynchronous - the GPU may still be writing to
/// memory allocated by the output after `execute` returns. Converting a `DeviceOp` into
/// a `DeviceFuture` ensures memory operations complete before the output can be accessed.
///
/// ## Examples
///
/// ```rust,ignore
/// use cuda_async::device_operation::{DeviceOp, value};
///
/// // Create a simple value operation
/// let op1 = value(42);
///
/// // Chain operations together
/// let op2 = op1.then(|x| value(x * 2));
///
/// // Execute synchronously (blocks until GPU completes)
/// let result = op2.sync().expect("Device operation failed."); // returns 84
/// ```
///
/// ```rust,ignore
/// use cuda_async::device_operation::{DeviceOp, zip};
/// use cutile::api;
///
/// // Compose multiple tensor operations
/// let x = api::zeros(&[64, 64]);
/// let y = api::ones(&[64, 64]);
/// let combined = zip!(x, y).then(|(x, y)| {
///     // Both tensors are ready here
///     value((x, y))
/// });
/// ```
///
/// ## Async Usage
///
/// Operations automatically implement `IntoFuture`, enabling use with `.await`:
///
/// ```rust,ignore
/// let x: Arc<Tensor<f32>> = api::randn(0.0, 1.0, &[100, 100]).await?.into();
/// let y = some_kernel(x.clone()).await?;
/// ```
pub trait DeviceOp:
    Send + Sized + IntoFuture<Output = Result<<Self as DeviceOp>::Output, DeviceError>>
{
    type Output: Send;

    // Consumes DeviceOp and executes the implementing operation.
    // This is unsafe because it is asynchronous: A device may be writing to memory allocated
    // by the output.
    // Converting DeviceOp into a DeviceFuture ensures any memory operations are complete
    // before the output can be accessed by the async runtime.
    /// Enqueue this operation on `context`.
    ///
    /// Implementations must register owned resources with [`ExecutionContext::retain`]
    /// before submitting work. Returning them or borrowing them in the output
    /// is insufficient: safe callers may discard outputs or forget futures.
    ///
    /// # Safety
    /// The caller must establish completion before host access to the output,
    /// and must retain the context until completion or deliberately leak it.
    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError>;
    /// Schedule this operation on a specific policy and return a [`DeviceFuture`].
    fn schedule(
        self,
        policy: &Arc<dyn SchedulingPolicy>,
    ) -> Result<DeviceFuture<<Self as DeviceOp>::Output, Self>, DeviceError> {
        let stream = policy.next_stream()?;
        let mut future = DeviceFuture::new();
        future.device_operation = Some(self);
        future.execution_context = Some(ExecutionContext::new(stream));
        Ok(future)
    }
    /// Chain a follow-up operation that runs **on the same stream** as `self`.
    ///
    /// Because both operations share a stream, `f` is guaranteed to see `self`'s output
    /// fully written. This is the recommended way to express data dependencies without
    /// manual synchronization.
    ///
    /// The closure must not execute other DeviceOps (e.g., via `sync_on` or `sync`).
    /// This is enforced at runtime by the thread-local execution lock — attempting
    /// nested execution will return a `DeviceError`. See
    /// [`then_unchecked`](DeviceOp::then_unchecked) to opt out of that check.
    fn then<O: Send, DO, F>(self, f: F) -> AndThen<<Self as DeviceOp>::Output, Self, O, DO, F>
    where
        DO: DeviceOp<Output = O>,
        F: FnOnce(<Self as DeviceOp>::Output) -> DO,
    {
        AndThen {
            op: self,
            closure: f,
        }
    }
    /// Like [`then`](DeviceOp::then), but the closure runs with the
    /// thread-local execution lock **released**, so it may execute other
    /// operations: `.sync()`, `.sync_on(&stream)`, or a nested `.await`
    /// (for example through `futures::executor::block_on`). The lock is
    /// re-acquired when the closure returns (or unwinds), so the rest of the
    /// chain — and the operation the closure returns — executes under the
    /// lock as usual.
    ///
    /// # Safety
    ///
    /// The lock exists to rule out cross-stream data races, and this method
    /// removes that protection for the closure. The caller asserts that
    /// nothing the closure executes touches, on a stream other than the
    /// chain's, memory that is reachable from `self`'s output or otherwise
    /// still in flight on the chain's stream. Host-only work, work on
    /// unrelated data, and work explicitly issued on the chain's own stream
    /// are fine. Violating this races the GPU against itself, which is
    /// undefined behavior under CUDA.
    unsafe fn then_unchecked<O: Send, DO, F>(
        self,
        f: F,
    ) -> AndThenUnchecked<<Self as DeviceOp>::Output, Self, O, DO, F>
    where
        DO: DeviceOp<Output = O>,
        F: FnOnce(<Self as DeviceOp>::Output) -> DO,
    {
        AndThenUnchecked {
            op: self,
            closure: f,
        }
    }
    /// Transform the output of this operation without issuing new GPU work.
    fn map<O: Send, F>(
        self,
        f: F,
    ) -> AndThen<
        <Self as DeviceOp>::Output,
        Self,
        O,
        Value<O>,
        impl FnOnce(<Self as DeviceOp>::Output) -> Value<O> + Send,
    >
    where
        F: FnOnce(<Self as DeviceOp>::Output) -> O + Send,
    {
        self.then(move |x| value(f(x)))
    }
    /// Peek at the output for debugging without consuming or transforming it.
    fn inspect<F>(
        self,
        f: F,
    ) -> AndThen<
        <Self as DeviceOp>::Output,
        Self,
        <Self as DeviceOp>::Output,
        Value<<Self as DeviceOp>::Output>,
        impl FnOnce(<Self as DeviceOp>::Output) -> Value<<Self as DeviceOp>::Output> + Send,
    >
    where
        F: FnOnce(&<Self as DeviceOp>::Output) + Send,
    {
        self.map(move |x| {
            f(&x);
            x
        })
    }
    fn and_then_with_context<O: Send, DO, F>(
        self,
        f: F,
    ) -> AndThenWithContext<<Self as DeviceOp>::Output, Self, O, DO, F>
    where
        DO: DeviceOp<Output = O>,
        F: FnOnce(&ExecutionContext, <Self as DeviceOp>::Output) -> DO,
    {
        AndThenWithContext {
            op: self,
            closure: f,
        }
    }
    /// Type-erase this operation into a [`BoxedDeviceOp`].
    ///
    /// This allows heterogeneous collections of operations that share the same
    /// output type but differ in their concrete type (e.g. mixing `Value`,
    /// `SelectLeft`, etc. in a single `Vec`).
    fn boxed(self) -> BoxedDeviceOp<<Self as DeviceOp>::Output>
    where
        Self: 'static,
    {
        BoxedDeviceOp {
            inner: Box::new(move |ctx| unsafe { self.execute(ctx) }),
        }
    }
    /// Convert into a cloneable, execute-once operation.
    ///
    /// The underlying op executes at most once; every clone gets `Arc::clone()`
    /// of the cached result. Follows the `FutureExt::shared()` convention.
    fn shared(self) -> SharedDeviceOp<<Self as DeviceOp>::Output>
    where
        Self: 'static,
        <Self as DeviceOp>::Output: Sync,
    {
        SharedDeviceOp {
            inner: Arc::new(ExecuteOnce::pending(Box::new(
                move |ctx: &ExecutionContext| unsafe { self.execute(ctx) },
            ))),
        }
    }
    /// Capture this operation into a replayable [`CudaGraph`](crate::cuda_graph::CudaGraph)
    /// using the default device's scheduling policy to pick a stream.
    fn graph(
        self,
    ) -> Result<crate::cuda_graph::CudaGraph<<Self as DeviceOp>::Output>, DeviceError> {
        let stream = with_default_device_policy(|policy| policy.next_stream())??;
        self.graph_on(stream)
    }
    /// Capture this operation into a replayable [`CudaGraph`](crate::cuda_graph::CudaGraph)
    /// on an **explicit stream**.
    ///
    /// Executes the operation once on `stream` in capture mode, recording
    /// all GPU work. Returns a `CudaGraph<Self::Output>` containing the
    /// replayable graph and the initial output.
    fn graph_on(
        self,
        stream: Arc<Stream>,
    ) -> Result<crate::cuda_graph::CudaGraph<<Self as DeviceOp>::Output>, DeviceError> {
        crate::cuda_graph::CudaGraph::capture(stream, self)
    }
    /// Execute synchronously using the default device's scheduling policy.
    ///
    /// The policy picks a stream (round-robin by default), submits the work, and blocks
    /// until the GPU finishes. Equivalent to `.await` but blocking.
    fn sync(self) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        let stream = with_default_device_policy(|policy| policy.next_stream())??;
        self.sync_on(&stream)
    }
    /// Execute on a stream without synchronizing. The GPU may still be
    /// writing to the output when this returns.
    ///
    /// # Safety
    ///
    /// The caller must ensure the stream is synchronized before accessing
    /// GPU data from the output, and must keep every submitted resource alive
    /// and prevent conflicting accesses until completion, including resources
    /// discarded by combinators. Unlike safe execution, this method does not
    /// retain resources after it returns.
    unsafe fn async_on(
        self,
        stream: &Arc<Stream>,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        let ctx = ExecutionContext::new(stream.clone());
        let result = unsafe { self.execute(&ctx) };
        unsafe { ctx.complete() };
        result
    }
    /// Execute on an **explicit stream** and block until the GPU finishes.
    ///
    /// This bypasses the scheduling policy entirely. All operations `sync_on` the same
    /// stream are guaranteed to execute in call order. Use this when you need deterministic
    /// ordering or are debugging concurrency issues.
    fn sync_on(self, stream: &Arc<Stream>) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        // Held until this function returns; released even if `execute` panics.
        let _execution_lock = acquire_execution_lock()?;
        let ctx = ExecutionContext::new(stream.clone());
        let res = unsafe { self.execute(&ctx) };
        let sync_res = unsafe { stream.synchronize() };
        sync_res?;
        unsafe { ctx.complete() };
        res
    }
}

// ── GraphNode ────────────────────────────────────────────────────────────────

/// Marker trait for [`DeviceOp`]s that are safe to record in a CUDA graph.
///
/// Only operations that do **not** allocate or free device memory should
/// implement this trait. During CUDA graph capture, allocation nodes may
/// return different addresses on replay, breaking baked-in pointers.
///
/// Implementors:
/// - Macro-generated kernel launchers (kernel launch only)
/// - Memcpy operations between pre-allocated buffers
/// - [`Value<T>`] (no GPU work)
///
/// Non-implementors (allocate device memory):
/// - `api::zeros`, `api::ones`, `api::full`, `api::arange`
/// - `api::randn`, `api::rand`
/// - `dup`, `copy_host_vec_to_device`
///
/// See [`Scope`](crate::cuda_graph::Scope) for the full safety proof.
pub trait GraphNode: DeviceOp {}

// Arc

// Boxed (type-erased) DeviceOp

/// A type-erased [`DeviceOp`] that boxes the execution closure.
///
/// Created via [`DeviceOp::boxed()`].
/// Useful when you need to store operations with different concrete types but
/// the same `Output` in a homogeneous collection (e.g.
/// `Vec<BoxedDeviceOp<'_, T>>`).
pub struct BoxedDeviceOp<T: Send> {
    inner: Box<dyn FnOnce(&ExecutionContext) -> Result<T, DeviceError> + Send>,
}

impl<T: Send> DeviceOp for BoxedDeviceOp<T> {
    type Output = T;

    unsafe fn execute(self, context: &ExecutionContext) -> Result<T, DeviceError> {
        (self.inner)(context)
    }
}

impl<T: Send> IntoFuture for BoxedDeviceOp<T> {
    type Output = Result<T, DeviceError>;
    type IntoFuture = DeviceFuture<T, Self>;
    fn into_future(self) -> Self::IntoFuture {
        let stream = match with_default_device_policy(|policy| policy.next_stream()) {
            Ok(Ok(stream)) => stream,
            Ok(Err(e)) | Err(e) => return DeviceFuture::failed(e),
        };
        let mut f = DeviceFuture::new();
        f.device_operation = Some(self);
        f.execution_context = Some(ExecutionContext::new(stream));
        f
    }
}

// ── Execute-once memoization (shared by `SharedDeviceOp` and `unzip`) ───────

/// Where a memoized result was produced: the stream the operation ran on and
/// an event recorded on that stream immediately after its work.
///
/// A consumer that picks up the cached value on a *different* stream has no
/// stream-order relationship to the producing work, so before the value is
/// handed out its stream is made to wait on the event (`cuStreamWaitEvent`).
/// Consumers on the producing stream are already ordered and skip the wait.
struct Producer {
    stream: Arc<Stream>,
    event: Event,
}

impl Producer {
    /// Records the completion event for the work just enqueued on the
    /// context's stream.
    fn record(ctx: &ExecutionContext) -> Result<Self, DeviceError> {
        let stream = Arc::clone(ctx.get_cuda_stream());
        let event = stream.device().new_event()?;
        event.record(&stream)?;
        Ok(Self { stream, event })
    }

    /// Orders all future work on `consumer` after the producing work.
    fn join(&self, consumer: &Arc<Stream>) -> Result<(), DeviceError> {
        if consumer.cu_stream() == self.stream.cu_stream() {
            return Ok(());
        }
        consumer.wait_event(&self.event)?;
        Ok(())
    }
}

enum OnceState<Op, Out> {
    /// Not yet executed; holds the operation.
    Pending(Op),
    /// Being executed by `thread`. Others wait on the condvar; the executing
    /// thread itself re-entering is a bug reported as an error, not a
    /// deadlock.
    Running { thread: ThreadId },
    /// Executed; `producer` orders consumers on other streams after the
    /// work. `None` for values that never had GPU work behind them.
    Done {
        value: Out,
        producer: Option<Producer>,
    },
    /// Execution failed (or the executor panicked); every consumer gets the
    /// error. The operation was consumed and cannot be retried.
    Failed(DeviceError),
}

/// Runs an operation exactly once across any number of concurrent callers
/// and memoizes the outcome.
///
/// This replaces the former check-then-act on `UnsafeCell`s behind
/// hand-written `Send`/`Sync` impls: two executors that both observed
/// "not computed" both took the operation (the second got "already taken")
/// and raced on the result cell. Here the state lives under a `Mutex`; the
/// first caller to see `Pending` takes the operation and runs it *outside*
/// the lock, later callers block on the condvar until it is `Done` or
/// `Failed`. `Send`/`Sync` are derived from the field types.
struct ExecuteOnce<Op, Out> {
    state: Mutex<OnceState<Op, Out>>,
    settled: Condvar,
}

impl<Op, Out> ExecuteOnce<Op, Out> {
    fn pending(op: Op) -> Self {
        Self {
            state: Mutex::new(OnceState::Pending(op)),
            settled: Condvar::new(),
        }
    }

    /// Already settled with a value that has no GPU work behind it.
    fn done(value: Out) -> Self {
        Self {
            state: Mutex::new(OnceState::Done {
                value,
                producer: None,
            }),
            settled: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, OnceState<Op, Out>> {
        // A poisoned lock only means a holder panicked; the state machine is
        // never left mid-transition while the lock is held.
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Stores a pre-computed value produced on `ctx`'s stream.
    fn settle_done(&self, value: Out, ctx: &ExecutionContext) {
        let outcome = match Producer::record(ctx) {
            Ok(producer) => OnceState::Done {
                value,
                producer: Some(producer),
            },
            Err(error) => {
                // Without an event, consumers on other streams could not be
                // ordered after the producing work. The value may own memory
                // that work still writes; releasing it now would be the
                // in-flight-free hazard, so leak it loudly instead.
                crate::leak::report_leak(format_args!(
                    "cuda-async: leaking a memoized result after its completion \
                     event could not be recorded: {error}"
                ));
                std::mem::forget(value);
                OnceState::Failed(error)
            }
        };
        *self.lock() = outcome;
        self.settled.notify_all();
    }

    /// Executes `run(op, ctx)` if this is the first caller (waiting for it
    /// if another caller is mid-execution), then hands `take` the cached
    /// value after ordering `ctx`'s stream behind the producing work.
    fn execute<R>(
        &self,
        ctx: &ExecutionContext,
        run: impl FnOnce(Op, &ExecutionContext) -> Result<Out, DeviceError>,
        take: impl FnOnce(&mut Out) -> Result<R, DeviceError>,
    ) -> Result<R, DeviceError> {
        // `Pending` is observed at most once per `ExecuteOnce`, so `run` is
        // called at most once; the `Option` makes that visible to the borrow
        // checker across the loop.
        let mut run = Some(run);
        let mut state = self.lock();
        loop {
            match &*state {
                OnceState::Pending(_) => {
                    let running = OnceState::Running {
                        thread: std::thread::current().id(),
                    };
                    let OnceState::Pending(op) = std::mem::replace(&mut *state, running) else {
                        unreachable!("matched Pending above");
                    };
                    drop(state);
                    let run = run.take().expect("Pending is observed at most once");
                    // If `run` unwinds, settle as Failed so waiters are
                    // released instead of blocking on `Running` forever.
                    let mut settle_on_unwind = SettleOnUnwind { once: Some(self) };
                    let outcome = run(op, ctx);
                    settle_on_unwind.once = None;
                    match outcome {
                        Ok(value) => self.settle_done(value, ctx),
                        Err(error) => {
                            *self.lock() = OnceState::Failed(error);
                            self.settled.notify_all();
                        }
                    }
                    state = self.lock();
                }
                OnceState::Running { thread } => {
                    if *thread == std::thread::current().id() {
                        return Err(DeviceError::Internal(
                            "execute-once operation re-entered from inside its own \
                             execution (a shared or unzipped op executing itself)"
                                .into(),
                        ));
                    }
                    state = self
                        .settled
                        .wait(state)
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                }
                OnceState::Done { .. } => break,
                OnceState::Failed(error) => return Err(error.clone()),
            }
        }
        let OnceState::Done { value, producer } = &mut *state else {
            unreachable!("loop exits only on Done");
        };
        if let Some(producer) = producer {
            producer.join(ctx.get_cuda_stream())?;
        }
        take(value)
    }
}

/// Marks an `ExecuteOnce` as failed if the executing closure unwinds.
struct SettleOnUnwind<'a, Op, Out> {
    once: Option<&'a ExecuteOnce<Op, Out>>,
}

impl<Op, Out> Drop for SettleOnUnwind<'_, Op, Out> {
    fn drop(&mut self) {
        if let Some(once) = self.once {
            *once.lock() = OnceState::Failed(DeviceError::Internal(
                "execute-once operation panicked while executing".into(),
            ));
            once.settled.notify_all();
        }
    }
}

// Shared (cloneable, execute-once) DeviceOp

type SharedOp<T> = Box<dyn FnOnce(&ExecutionContext) -> Result<T, DeviceError> + Send>;

/// A cloneable, execute-once [`DeviceOp`].
///
/// Created via [`DeviceOp::shared()`]. The underlying operation executes at most
/// once; every clone gets `Arc::clone()` of the cached result. Follows the
/// `FutureExt::shared()` convention from the `futures` crate.
///
/// Output is always `Arc<T>` — the result is wrapped on first execution and
/// shared via refcount thereafter.
///
/// Clones may be executed concurrently from several threads: the first
/// executor runs the operation, the others wait for it. A clone executed on
/// a different stream than the one that produced the value has its stream
/// wait on a completion event recorded after the producing work, so the
/// cached value is safe to consume there. If the operation fails, every
/// clone receives the same error.
pub struct SharedDeviceOp<T: Send + Sync> {
    inner: Arc<ExecuteOnce<SharedOp<T>, Arc<T>>>,
}

impl<T: Send + Sync> Clone for SharedDeviceOp<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T: Send + Sync> DeviceOp for SharedDeviceOp<T> {
    type Output = Arc<T>;

    unsafe fn execute(self, context: &ExecutionContext) -> Result<Arc<T>, DeviceError> {
        self.inner.execute(
            context,
            |op, ctx| op(ctx).map(Arc::new),
            |value| Ok(Arc::clone(value)),
        )
    }
}

impl<T: Send + Sync> IntoFuture for SharedDeviceOp<T> {
    type Output = Result<Arc<T>, DeviceError>;
    type IntoFuture = DeviceFuture<Arc<T>, SharedDeviceOp<T>>;
    fn into_future(self) -> Self::IntoFuture {
        let stream = match with_default_device_policy(|policy| policy.next_stream()) {
            Ok(Ok(stream)) => stream,
            Ok(Err(e)) | Err(e) => return DeviceFuture::failed(e),
        };
        let mut f = DeviceFuture::new();
        f.device_operation = Some(self);
        f.execution_context = Some(ExecutionContext::new(stream));
        f
    }
}

/// Create a pre-computed [`SharedDeviceOp`] from an existing `Arc<T>`.
///
/// The returned op is already "executed" — cloning it just bumps the refcount.
/// No GPU work is associated with the value, so executing a clone on any
/// stream hands it out without a cross-stream wait.
pub fn shared<T: Send + Sync>(val: Arc<T>) -> SharedDeviceOp<T> {
    SharedDeviceOp {
        inner: Arc::new(ExecuteOnce::done(val)),
    }
}

// IntoDeviceOp — convert plain values or existing DeviceOps into DeviceOp

/// Conversion trait that accepts both plain values and existing [`DeviceOp`]s.
///
/// The blanket impl covers all `DeviceOp` types (pass-through). Specific impls
/// cover plain data types (`f32`, `Arc<Tensor<T>>`, etc.) that wrap via [`Value`].
pub trait IntoDeviceOp<T: Send> {
    type Op: DeviceOp<Output = T>;
    fn into_op(self) -> Self::Op;
}

impl<T: Send, DO: DeviceOp<Output = T>> IntoDeviceOp<T> for DO {
    type Op = DO;
    fn into_op(self) -> DO {
        self
    }
}

// IntoDeviceOp impls for Arc<T> and &Arc<T> — wraps in Value.
impl<T: Send + Sync + 'static> IntoDeviceOp<Arc<T>> for Arc<T> {
    type Op = Value<Arc<T>>;
    fn into_op(self) -> Value<Arc<T>> {
        value(self)
    }
}

impl<T: Send + Sync + 'static> IntoDeviceOp<Arc<T>> for &Arc<T> {
    type Op = Value<Arc<T>>;
    fn into_op(self) -> Value<Arc<T>> {
        value(self.clone())
    }
}

// Scalar IntoDeviceOp impls — wraps the value in Value<T>.
macro_rules! impl_into_device_op_scalar {
    ($($ty:ty),*) => {
        $(
            impl IntoDeviceOp<$ty> for $ty {
                type Op = Value<$ty>;
                fn into_op(self) -> Value<$ty> { value(self) }
            }
        )*
    };
}
impl_into_device_op_scalar!(
    f32,
    f64,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
    usize,
    bool,
    half::f16,
    half::bf16
);

// DevicePointer impl — for unsafe kernel pointer arguments.
impl<T: cuda_core::DType + Send> IntoDeviceOp<crate::device_buffer::DevicePointer<T>>
    for crate::device_buffer::DevicePointer<T>
{
    type Op = Value<crate::device_buffer::DevicePointer<T>>;
    fn into_op(self) -> Value<crate::device_buffer::DevicePointer<T>> {
        value(self)
    }
}

// Unwrap Arc
/// Extension trait: `.unwrap_arc()` on `DeviceOp<Output = Arc<T>>`.
///
/// Unwraps the Arc at execution time. Fails if the Arc has multiple owners.
pub trait DeviceOpUnwrapArc<T: Send + Sync>: DeviceOp<Output = Arc<T>> + Sized {
    fn unwrap_arc(
        self,
    ) -> AndThen<Arc<T>, Self, T, Value<T>, impl FnOnce(Arc<T>) -> Value<T> + Send> {
        self.then(|arc| {
            value(
                Arc::try_unwrap(arc)
                    .unwrap_or_else(|_| panic!("unwrap_arc: Arc has multiple owners")),
            )
        })
    }
}

impl<T: Send + Sync, DI: DeviceOp<Output = Arc<T>>> DeviceOpUnwrapArc<T> for DI {}

// AndThen

pub struct AndThen<I: Send, DI, O: Send, DO, F>
where
    DI: DeviceOp<Output = I>,
    DO: DeviceOp<Output = O>,
    F: FnOnce(I) -> DO,
{
    op: DI,
    closure: F,
}

unsafe impl<I: Send, DI, O: Send, DO, F> Send for AndThen<I, DI, O, DO, F>
where
    DI: DeviceOp<Output = I>,
    DO: DeviceOp<Output = O>,
    F: FnOnce(I) -> DO + Send,
{
}

impl<I: Send, DI, O: Send, DO, F> DeviceOp for AndThen<I, DI, O, DO, F>
where
    DI: DeviceOp<Output = I>,
    DO: DeviceOp<Output = O>,
    F: FnOnce(I) -> DO + Send,
{
    type Output = O;

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        let input: I = self.op.execute(context)?;
        let output_device_op: DO = (self.closure)(input);
        output_device_op.execute(context)
    }
}

impl<I: Send, DI, O: Send, DO, F> IntoFuture for AndThen<I, DI, O, DO, F>
where
    DI: DeviceOp<Output = I>,
    DO: DeviceOp<Output = O>,
    F: FnOnce(I) -> DO + Send,
{
    type Output = Result<O, DeviceError>;
    type IntoFuture = DeviceFuture<O, AndThen<I, DI, O, DO, F>>;
    fn into_future(self) -> Self::IntoFuture {
        let stream = match with_default_device_policy(|policy| policy.next_stream()) {
            Ok(Ok(stream)) => stream,
            Ok(Err(e)) | Err(e) => return DeviceFuture::failed(e),
        };
        let mut f = DeviceFuture::new();
        f.device_operation = Some(self);
        f.execution_context = Some(ExecutionContext::new(stream));
        f
    }
}

// AndThenUnchecked

/// The combinator behind [`DeviceOp::then_unchecked`]: [`AndThen`] whose
/// closure runs with the thread-local execution lock released.
///
/// `Send` is derived: the only fields are the upstream op (`DeviceOp: Send`)
/// and the closure, which the `DeviceOp` impl requires to be `Send`.
pub struct AndThenUnchecked<I: Send, DI, O: Send, DO, F>
where
    DI: DeviceOp<Output = I>,
    DO: DeviceOp<Output = O>,
    F: FnOnce(I) -> DO,
{
    op: DI,
    closure: F,
}

impl<I: Send, DI, O: Send, DO, F> DeviceOp for AndThenUnchecked<I, DI, O, DO, F>
where
    DI: DeviceOp<Output = I>,
    DO: DeviceOp<Output = O>,
    F: FnOnce(I) -> DO + Send,
{
    type Output = O;

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        let input: I = self.op.execute(context)?;
        let output_device_op: DO = {
            // Released for the closure only. The guard restores the lock on
            // every exit — including a panic unwinding out of the closure —
            // before the returned operation executes under it below.
            let _released = ExecutionLockRelease::new();
            (self.closure)(input)
        };
        output_device_op.execute(context)
    }
}

impl<I: Send, DI, O: Send, DO, F> IntoFuture for AndThenUnchecked<I, DI, O, DO, F>
where
    DI: DeviceOp<Output = I>,
    DO: DeviceOp<Output = O>,
    F: FnOnce(I) -> DO + Send,
{
    type Output = Result<O, DeviceError>;
    type IntoFuture = DeviceFuture<O, AndThenUnchecked<I, DI, O, DO, F>>;
    fn into_future(self) -> Self::IntoFuture {
        let stream = match with_default_device_policy(|policy| policy.next_stream()) {
            Ok(Ok(stream)) => stream,
            Ok(Err(e)) | Err(e) => return DeviceFuture::failed(e),
        };
        let mut f = DeviceFuture::new();
        f.device_operation = Some(self);
        f.execution_context = Some(ExecutionContext::new(stream));
        f
    }
}

// Value

/// Wraps an immediate value as a completed device op.
///
/// `Value<T>` is `Send` exactly when `T` is — the compiler derives it, and
/// nothing here may override that with a manual `unsafe impl Send` (one did
/// exist, and made `Value<Rc<_>>` sendable from safe code). The doctest below
/// fails to compile only while that stays true:
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<cuda_async::device_operation::Value<std::rc::Rc<u8>>>();
/// ```
pub struct Value<T>(T);

impl<T> Value<T> {
    pub fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T: Send> DeviceOp for Value<T> {
    type Output = T;

    unsafe fn execute(
        self,
        _context: &ExecutionContext,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        Ok(self.0)
    }
}

impl<T: Send> GraphNode for Value<T> {}

impl<T: Send> IntoFuture for Value<T> {
    type Output = Result<T, DeviceError>;
    type IntoFuture = DeviceFuture<T, Value<T>>;
    fn into_future(self) -> Self::IntoFuture {
        let stream = match with_default_device_policy(|policy| policy.next_stream()) {
            Ok(Ok(stream)) => stream,
            Ok(Err(e)) | Err(e) => return DeviceFuture::failed(e),
        };
        let mut f = DeviceFuture::new();
        f.device_operation = Some(self);
        f.execution_context = Some(ExecutionContext::new(stream));
        f
    }
}

pub fn value<T: Send>(x: T) -> Value<T> {
    Value::new(x)
}

impl From<f32> for Value<f32> {
    fn from(val: f32) -> Self {
        Value::new(val)
    }
}

// Empty (closure)

pub struct Empty<O: Send, DO: DeviceOp<Output = O>, F: FnOnce() -> DO + Send> {
    closure: F,
}

pub fn empty<O: Send, DO: DeviceOp<Output = O>, F: FnOnce() -> DO + Send>(
    closure: F,
) -> Empty<O, DO, F> {
    Empty { closure }
}

impl<O: Send, DO, F> DeviceOp for Empty<O, DO, F>
where
    DO: DeviceOp<Output = O>,
    F: FnOnce() -> DO + Send,
{
    type Output = O;

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        let out_device_op = (self.closure)();
        out_device_op.execute(context)
    }
}

impl<O: Send, DO: DeviceOp<Output = O>, F: FnOnce() -> DO + Send> IntoFuture for Empty<O, DO, F> {
    type Output = Result<O, DeviceError>;
    type IntoFuture = DeviceFuture<O, Empty<O, DO, F>>;
    fn into_future(self) -> Self::IntoFuture {
        let stream = match with_default_device_policy(|policy| policy.next_stream()) {
            Ok(Ok(stream)) => stream,
            Ok(Err(e)) | Err(e) => return DeviceFuture::failed(e),
        };
        let mut f = DeviceFuture::new();
        f.device_operation = Some(self);
        f.execution_context = Some(ExecutionContext::new(stream));
        f
    }
}

// Zip

pub struct Zip<T1: Send, T2: Send, A: DeviceOp<Output = T1>, B: DeviceOp<Output = T2>> {
    phantom: PhantomData<(T1, T2)>,
    a: A,
    b: B,
}

unsafe impl<T1: Send, T2: Send, A: DeviceOp<Output = T1>, B: DeviceOp<Output = T2>> Send
    for Zip<T1, T2, A, B>
{
}

fn _zip<T1: Send, T2: Send, A: DeviceOp<Output = T1>, B: DeviceOp<Output = T2>>(
    a: A,
    b: B,
) -> Zip<T1, T2, A, B> {
    Zip {
        phantom: PhantomData,
        a,
        b,
    }
}

impl<T1: Send, T2: Send, A: DeviceOp<Output = T1>, B: DeviceOp<Output = T2>> DeviceOp
    for Zip<T1, T2, A, B>
{
    type Output = (T1, T2);

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        let a: T1 = self.a.execute(context)?;
        let b: T2 = self.b.execute(context)?;
        Ok((a, b))
    }
}

impl<T1: Send, T2: Send, A: DeviceOp<Output = T1>, B: DeviceOp<Output = T2>> IntoFuture
    for Zip<T1, T2, A, B>
{
    type Output = Result<(T1, T2), DeviceError>;
    type IntoFuture = DeviceFuture<(T1, T2), Zip<T1, T2, A, B>>;
    fn into_future(self) -> Self::IntoFuture {
        let stream = match with_default_device_policy(|policy| policy.next_stream()) {
            Ok(Ok(stream)) => stream,
            Ok(Err(e)) | Err(e) => return DeviceFuture::failed(e),
        };
        let mut f = DeviceFuture::new();
        f.device_operation = Some(self);
        f.execution_context = Some(ExecutionContext::new(stream));
        f
    }
}

pub trait Zippable<I, O: Send> {
    fn zip(self) -> impl DeviceOp<Output = O>;
}

impl<T0: Send, T1: Send, DI0: DeviceOp<Output = T0>, DI1: DeviceOp<Output = T1>>
    Zippable<(DI0, DI1), (T0, T1)> for (DI0, DI1)
{
    fn zip(self) -> impl DeviceOp<Output = (T0, T1)> {
        _zip(self.0, self.1)
    }
}

impl<
        T0: Send,
        T1: Send,
        T2: Send,
        DI0: DeviceOp<Output = T0>,
        DI1: DeviceOp<Output = T1>,
        DI2: DeviceOp<Output = T2>,
    > Zippable<(DI0, DI1, DI2), (T0, T1, T2)> for (DI0, DI1, DI2)
{
    fn zip(self) -> impl DeviceOp<Output = (T0, T1, T2)> {
        let cons = _zip(self.1, self.2);
        let cons = _zip(self.0, cons);
        cons.then(|(arg0, (arg1, arg2))| value((arg0, arg1, arg2)))
    }
}

impl<
        T0: Send,
        T1: Send,
        T2: Send,
        T3: Send,
        DI0: DeviceOp<Output = T0>,
        DI1: DeviceOp<Output = T1>,
        DI2: DeviceOp<Output = T2>,
        DI3: DeviceOp<Output = T3>,
    > Zippable<(DI0, DI1, DI2, DI3), (T0, T1, T2, T3)> for (DI0, DI1, DI2, DI3)
{
    fn zip(self) -> impl DeviceOp<Output = (T0, T1, T2, T3)> {
        let cons = _zip(self.2, self.3);
        let cons = _zip(self.1, cons);
        let cons = _zip(self.0, cons);
        cons.then(|(arg0, (arg1, (arg2, arg3)))| value((arg0, arg1, arg2, arg3)))
    }
}

impl<
        T0: Send,
        T1: Send,
        T2: Send,
        T3: Send,
        T4: Send,
        DI0: DeviceOp<Output = T0>,
        DI1: DeviceOp<Output = T1>,
        DI2: DeviceOp<Output = T2>,
        DI3: DeviceOp<Output = T3>,
        DI4: DeviceOp<Output = T4>,
    > Zippable<(DI0, DI1, DI2, DI3, DI4), (T0, T1, T2, T3, T4)> for (DI0, DI1, DI2, DI3, DI4)
{
    fn zip(self) -> impl DeviceOp<Output = (T0, T1, T2, T3, T4)> {
        let cons = _zip(self.3, self.4);
        let cons = _zip(self.2, cons);
        let cons = _zip(self.1, cons);
        let cons = _zip(self.0, cons);
        cons.then(|(arg0, (arg1, (arg2, (arg3, arg4))))| value((arg0, arg1, arg2, arg3, arg4)))
    }
}

impl<
        T0: Send,
        T1: Send,
        T2: Send,
        T3: Send,
        T4: Send,
        T5: Send,
        DI0: DeviceOp<Output = T0>,
        DI1: DeviceOp<Output = T1>,
        DI2: DeviceOp<Output = T2>,
        DI3: DeviceOp<Output = T3>,
        DI4: DeviceOp<Output = T4>,
        DI5: DeviceOp<Output = T5>,
    > Zippable<(DI0, DI1, DI2, DI3, DI4, DI5), (T0, T1, T2, T3, T4, T5)>
    for (DI0, DI1, DI2, DI3, DI4, DI5)
{
    fn zip(self) -> impl DeviceOp<Output = (T0, T1, T2, T3, T4, T5)> {
        let cons = _zip(self.4, self.5);
        let cons = _zip(self.3, cons);
        let cons = _zip(self.2, cons);
        let cons = _zip(self.1, cons);
        let cons = _zip(self.0, cons);
        cons.then(|(arg0, (arg1, (arg2, (arg3, (arg4, arg5)))))| {
            value((arg0, arg1, arg2, arg3, arg4, arg5))
        })
    }
}

#[macro_export]
macro_rules! zip {
    ($arg0:expr) => {
        $arg0
    };
    ($arg0:expr, $arg1:expr) => {
        ($arg0, $arg1).zip()
    };
    ($arg0:expr, $arg1:expr, $arg2:expr) => {
        ($arg0, $arg1, $arg2).zip()
    };
    ($arg0:expr, $arg1:expr, $arg2:expr, $arg3:expr) => {
        ($arg0, $arg1, $arg2, $arg3).zip()
    };
    ($arg0:expr, $arg1:expr, $arg2:expr, $arg3:expr, $arg4:expr) => {
        ($arg0, $arg1, $arg2, $arg3, $arg4).zip()
    };
    ($arg0:expr, $arg1:expr, $arg2:expr, $arg3:expr, $arg4:expr, $arg5:expr) => {
        ($arg0, $arg1, $arg2, $arg3, $arg4, $arg5).zip()
    };
}
pub use zip;

// Unzip

fn _unzip<T1: Send, T2: Send, DI>(input: DI) -> (SelectLeft<T1, T2, DI>, SelectRight<T1, T2, DI>)
where
    DI: DeviceOp<Output = (T1, T2)>,
{
    let select_arc = Arc::new(Select {
        once: ExecuteOnce::pending(input),
    });
    let out1 = SelectLeft {
        select: select_arc.clone(),
    };
    let out2 = SelectRight { select: select_arc };
    (out1, out2)
}

// Select: Execute a device operation at most once.

/// Execute-once state shared by the two halves of an [`unzip`](Unzippable2::unzip).
///
/// Whichever half executes first runs the input; the other half waits for
/// it if it is mid-execution (on another thread), then takes its side of the
/// result. A half executed on a different stream than the producing one has
/// its stream wait on the producer's completion event first. `Send`/`Sync`
/// are derived: the input is `DeviceOp: Send` and both halves are `Send`.
pub struct Select<T1: Send, T2: Send, DI>
where
    DI: DeviceOp<Output = (T1, T2)>,
{
    once: ExecuteOnce<DI, (Option<T1>, Option<T2>)>,
}

impl<T1: Send, T2: Send, DI> Select<T1, T2, DI>
where
    DI: DeviceOp<Output = (T1, T2)>,
{
    /// Runs the input if needed and hands `take` the stored halves.
    unsafe fn execute<R>(
        &self,
        context: &ExecutionContext,
        take: impl FnOnce(&mut (Option<T1>, Option<T2>)) -> Option<R>,
        side: &'static str,
    ) -> Result<R, DeviceError> {
        self.once.execute(
            context,
            |input, ctx| {
                let (left, right) = unsafe { input.execute(ctx) }?;
                Ok((Some(left), Some(right)))
            },
            |halves| {
                take(halves).ok_or_else(|| {
                    DeviceError::Internal(format!("unzip: the {side} result was already taken"))
                })
            },
        )
    }
}

// Select Left: Execute Select and take the left result.

pub struct SelectLeft<T1: Send, T2: Send, DI>
where
    DI: DeviceOp<Output = (T1, T2)>,
{
    select: Arc<Select<T1, T2, DI>>,
}

impl<T1: Send, T2: Send, DI> IntoFuture for SelectLeft<T1, T2, DI>
where
    DI: DeviceOp<Output = (T1, T2)>,
{
    type Output = Result<T1, DeviceError>;
    type IntoFuture = DeviceFuture<T1, SelectLeft<T1, T2, DI>>;
    fn into_future(self) -> Self::IntoFuture {
        let stream = match with_default_device_policy(|policy| policy.next_stream()) {
            Ok(Ok(stream)) => stream,
            Ok(Err(e)) | Err(e) => return DeviceFuture::failed(e),
        };
        let mut f = DeviceFuture::new();
        f.device_operation = Some(self);
        f.execution_context = Some(ExecutionContext::new(stream));
        f
    }
}

impl<T1: Send, T2: Send, DI> DeviceOp for SelectLeft<T1, T2, DI>
where
    DI: DeviceOp<Output = (T1, T2)>,
{
    type Output = T1;

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        self.select
            .execute(context, |(left, _)| left.take(), "left")
    }
}

// Select Right: Execute Select and take the right result.

pub struct SelectRight<T1: Send, T2: Send, DI>
where
    DI: DeviceOp<Output = (T1, T2)>,
{
    select: Arc<Select<T1, T2, DI>>,
}

impl<T1: Send, T2: Send, DI> IntoFuture for SelectRight<T1, T2, DI>
where
    DI: DeviceOp<Output = (T1, T2)>,
{
    type Output = Result<T2, DeviceError>;
    type IntoFuture = DeviceFuture<T2, SelectRight<T1, T2, DI>>;
    fn into_future(self) -> Self::IntoFuture {
        let stream = match with_default_device_policy(|policy| policy.next_stream()) {
            Ok(Ok(stream)) => stream,
            Ok(Err(e)) | Err(e) => return DeviceFuture::failed(e),
        };
        let mut f = DeviceFuture::new();
        f.device_operation = Some(self);
        f.execution_context = Some(ExecutionContext::new(stream));
        f
    }
}

impl<T1: Send, T2: Send, DI> DeviceOp for SelectRight<T1, T2, DI>
where
    DI: DeviceOp<Output = (T1, T2)>,
{
    type Output = T2;

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        self.select
            .execute(context, |(_, right)| right.take(), "right")
    }
}

pub trait Unzippable1<T0: Send>
where
    Self: DeviceOp<Output = (T0,)>,
{
    fn unzip(self) -> (impl DeviceOp<Output = T0>,) {
        (self.then(|(r,)| value(r)),)
    }
}
impl<T0: Send, DI: DeviceOp<Output = (T0,)>> Unzippable1<T0> for DI {}

pub trait Unzippable2<T0: Send, T1: Send>
where
    Self: DeviceOp<Output = (T0, T1)>,
{
    fn unzip(self) -> (impl DeviceOp<Output = T0>, impl DeviceOp<Output = T1>) {
        _unzip(self)
    }
    fn first(self) -> impl DeviceOp<Output = T0>
    where
        Self: Sized,
    {
        self.then(|(first, _)| value(first))
    }
    fn last(self) -> impl DeviceOp<Output = T1>
    where
        Self: Sized,
    {
        self.then(|(_, last)| value(last))
    }
}
impl<T0: Send, T1: Send, DI: DeviceOp<Output = (T0, T1)>> Unzippable2<T0, T1> for DI {}

pub trait Unzippable3<T0: Send, T1: Send, T2: Send>
where
    Self: DeviceOp<Output = (T0, T1, T2)>,
{
    fn unzip(
        self,
    ) -> (
        impl DeviceOp<Output = T0>,
        impl DeviceOp<Output = T1>,
        impl DeviceOp<Output = T2>,
    ) {
        let cons = self.then(|(arg0, arg1, arg2)| value((arg0, (arg1, arg2))));
        let (car, cdr) = _unzip(cons);
        let (cdr_car, cdr_cdr) = _unzip(cdr);
        (car, cdr_car, cdr_cdr)
    }
    fn first(self) -> impl DeviceOp<Output = T0>
    where
        Self: Sized,
    {
        self.then(|(first, _, _)| value(first))
    }
    fn last(self) -> impl DeviceOp<Output = T2>
    where
        Self: Sized,
    {
        self.then(|(_, _, last)| value(last))
    }
}
impl<T0: Send, T1: Send, T2: Send, DI: DeviceOp<Output = (T0, T1, T2)>> Unzippable3<T0, T1, T2>
    for DI
{
}

pub trait Unzippable4<T0: Send, T1: Send, T2: Send, T3: Send>
where
    Self: DeviceOp<Output = (T0, T1, T2, T3)>,
{
    fn unzip(
        self,
    ) -> (
        impl DeviceOp<Output = T0>,
        impl DeviceOp<Output = T1>,
        impl DeviceOp<Output = T2>,
        impl DeviceOp<Output = T3>,
    ) {
        let cons = self.then(|(a0, a1, a2, a3)| value((a0, (a1, (a2, a3)))));
        let (car, cdr) = _unzip(cons);
        let (cdr0, cdr1) = _unzip(cdr);
        let (cdr1_0, cdr1_1) = _unzip(cdr1);
        (car, cdr0, cdr1_0, cdr1_1)
    }
    fn first(self) -> impl DeviceOp<Output = T0>
    where
        Self: Sized,
    {
        self.then(|(first, _, _, _)| value(first))
    }
    fn last(self) -> impl DeviceOp<Output = T3>
    where
        Self: Sized,
    {
        self.then(|(_, _, _, last)| value(last))
    }
}
impl<T0: Send, T1: Send, T2: Send, T3: Send, DI: DeviceOp<Output = (T0, T1, T2, T3)>>
    Unzippable4<T0, T1, T2, T3> for DI
{
}

pub trait Unzippable5<T0: Send, T1: Send, T2: Send, T3: Send, T4: Send>
where
    Self: DeviceOp<Output = (T0, T1, T2, T3, T4)>,
{
    fn unzip(
        self,
    ) -> (
        impl DeviceOp<Output = T0>,
        impl DeviceOp<Output = T1>,
        impl DeviceOp<Output = T2>,
        impl DeviceOp<Output = T3>,
        impl DeviceOp<Output = T4>,
    ) {
        let cons = self.then(|(a0, a1, a2, a3, a4)| value((a0, (a1, (a2, (a3, a4))))));
        let (car, cdr) = _unzip(cons);
        let (cdr0, cdr1) = _unzip(cdr);
        let (cdr1_0, cdr1_1) = _unzip(cdr1);
        let (cdr2_0, cdr2_1) = _unzip(cdr1_1);
        (car, cdr0, cdr1_0, cdr2_0, cdr2_1)
    }
    fn first(self) -> impl DeviceOp<Output = T0>
    where
        Self: Sized,
    {
        self.then(|(first, _, _, _, _)| value(first))
    }
    fn last(self) -> impl DeviceOp<Output = T4>
    where
        Self: Sized,
    {
        self.then(|(_, _, _, _, last)| value(last))
    }
}
impl<
        T0: Send,
        T1: Send,
        T2: Send,
        T3: Send,
        T4: Send,
        DI: DeviceOp<Output = (T0, T1, T2, T3, T4)>,
    > Unzippable5<T0, T1, T2, T3, T4> for DI
{
}

pub trait Unzippable6<T0: Send, T1: Send, T2: Send, T3: Send, T4: Send, T5: Send>
where
    Self: DeviceOp<Output = (T0, T1, T2, T3, T4, T5)>,
{
    fn unzip(
        self,
    ) -> (
        impl DeviceOp<Output = T0>,
        impl DeviceOp<Output = T1>,
        impl DeviceOp<Output = T2>,
        impl DeviceOp<Output = T3>,
        impl DeviceOp<Output = T4>,
        impl DeviceOp<Output = T5>,
    ) {
        let cons = self.then(|(a0, a1, a2, a3, a4, a5)| value((a0, (a1, (a2, (a3, (a4, a5)))))));
        let (car, cdr) = _unzip(cons);
        let (cdr0, cdr1) = _unzip(cdr);
        let (cdr1_0, cdr1_1) = _unzip(cdr1);
        let (cdr2_0, cdr2_1) = _unzip(cdr1_1);
        let (cdr3_0, cdr3_1) = _unzip(cdr2_1);
        (car, cdr0, cdr1_0, cdr2_0, cdr3_0, cdr3_1)
    }
    fn first(self) -> impl DeviceOp<Output = T0>
    where
        Self: Sized,
    {
        self.then(|(first, _, _, _, _, _)| value(first))
    }
    fn last(self) -> impl DeviceOp<Output = T5>
    where
        Self: Sized,
    {
        self.then(|(_, _, _, _, _, last)| value(last))
    }
}
impl<
        T0: Send,
        T1: Send,
        T2: Send,
        T3: Send,
        T4: Send,
        T5: Send,
        DI: DeviceOp<Output = (T0, T1, T2, T3, T4, T5)>,
    > Unzippable6<T0, T1, T2, T3, T4, T5> for DI
{
}

#[macro_export]
macro_rules! unzip {
    ($arg0:expr) => {
        $arg0.unzip()
    };
}
pub use unzip;

// StreamOperation

pub struct StreamOperation<
    O: Send,
    DO: DeviceOp<Output = O>,
    F: FnOnce(&ExecutionContext) -> DO + Send,
> {
    f: F,
}

impl<O: Send, DO: DeviceOp<Output = O>, F: FnOnce(&ExecutionContext) -> DO + Send> DeviceOp
    for StreamOperation<O, DO, F>
{
    type Output = O;

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        let dop_out: DO = (self.f)(context);
        dop_out.execute(context)
    }
}

pub fn with_context<
    O: Send,
    DO: DeviceOp<Output = O>,
    F: FnOnce(&ExecutionContext) -> DO + Send,
>(
    f: F,
) -> impl DeviceOp<Output = O> {
    StreamOperation { f }
}

impl<O: Send, DO: DeviceOp<Output = O>, F: FnOnce(&ExecutionContext) -> DO + Send> IntoFuture
    for StreamOperation<O, DO, F>
{
    type Output = Result<O, DeviceError>;
    type IntoFuture = DeviceFuture<O, StreamOperation<O, DO, F>>;
    fn into_future(self) -> Self::IntoFuture {
        let stream = match with_default_device_policy(|policy| policy.next_stream()) {
            Ok(Ok(stream)) => stream,
            Ok(Err(e)) | Err(e) => return DeviceFuture::failed(e),
        };
        let mut f = DeviceFuture::new();
        f.device_operation = Some(self);
        f.execution_context = Some(ExecutionContext::new(stream));
        f
    }
}

// AndThenWithContext

pub struct AndThenWithContext<I: Send, DI, O: Send, DO, F>
where
    DI: DeviceOp<Output = I>,
    DO: DeviceOp<Output = O>,
    F: FnOnce(&ExecutionContext, I) -> DO,
{
    op: DI,
    closure: F,
}

unsafe impl<I: Send, DI, O: Send, DO, F> Send for AndThenWithContext<I, DI, O, DO, F>
where
    DI: DeviceOp<Output = I>,
    DO: DeviceOp<Output = O>,
    F: FnOnce(&ExecutionContext, I) -> DO + Send,
{
}

impl<I: Send, DI, O: Send, DO, F> DeviceOp for AndThenWithContext<I, DI, O, DO, F>
where
    DI: DeviceOp<Output = I>,
    DO: DeviceOp<Output = O>,
    F: FnOnce(&ExecutionContext, I) -> DO + Send,
{
    type Output = O;

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        let input: I = self.op.execute(context)?;
        let output_device_op: DO = (self.closure)(context, input);
        output_device_op.execute(context)
    }
}

impl<I: Send, DI, O: Send, DO, F> IntoFuture for AndThenWithContext<I, DI, O, DO, F>
where
    DI: DeviceOp<Output = I>,
    DO: DeviceOp<Output = O>,
    F: FnOnce(&ExecutionContext, I) -> DO + Send,
{
    type Output = Result<O, DeviceError>;
    type IntoFuture = DeviceFuture<O, AndThenWithContext<I, DI, O, DO, F>>;
    fn into_future(self) -> Self::IntoFuture {
        let stream = match with_default_device_policy(|policy| policy.next_stream()) {
            Ok(Ok(stream)) => stream,
            Ok(Err(e)) | Err(e) => return DeviceFuture::failed(e),
        };
        let mut f = DeviceFuture::new();
        f.device_operation = Some(self);
        f.execution_context = Some(ExecutionContext::new(stream));
        f
    }
}

// DeviceOpVec — execute a Vec of boxed operations, sync once.

/// A [`DeviceOp`] that executes a vector of [`BoxedDeviceOp`]s
/// sequentially on the same stream and collects their outputs into a `Vec<T>`.
///
/// This avoids per-element stream synchronization: the caller can issue a
/// single `.sync_on(&stream)` after all operations have been submitted.
pub struct DeviceOpVec<T: Send> {
    ops: Vec<BoxedDeviceOp<T>>,
}

impl<T: Send + 'static> DeviceOpVec<T> {
    pub fn empty() -> Self {
        Self { ops: Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            ops: Vec::with_capacity(capacity),
        }
    }

    pub fn new(ops: Vec<BoxedDeviceOp<T>>) -> Self {
        Self { ops }
    }

    pub fn push<DO: DeviceOp<Output = T> + 'static>(&mut self, op: DO) {
        self.ops.push(op.boxed());
    }

    pub fn remove(&mut self, index: usize) -> BoxedDeviceOp<T> {
        self.ops.remove(index)
    }

    pub fn last(&self) -> Option<&BoxedDeviceOp<T>> {
        self.ops.last()
    }
}

impl<T: Send> DeviceOp for DeviceOpVec<T> {
    type Output = Vec<T>;

    unsafe fn execute(self, context: &ExecutionContext) -> Result<Vec<T>, DeviceError> {
        let mut results = Vec::with_capacity(self.ops.len());
        for op in self.ops {
            results.push(op.execute(context)?);
        }
        Ok(results)
    }
}

impl<T: Send> IntoFuture for DeviceOpVec<T> {
    type Output = Result<Vec<T>, DeviceError>;
    type IntoFuture = DeviceFuture<Vec<T>, Self>;
    fn into_future(self) -> Self::IntoFuture {
        let stream = match with_default_device_policy(|policy| policy.next_stream()) {
            Ok(Ok(stream)) => stream,
            Ok(Err(e)) | Err(e) => return DeviceFuture::failed(e),
        };
        let mut f = DeviceFuture::new();
        f.device_operation = Some(self);
        f.execution_context = Some(ExecutionContext::new(stream));
        f
    }
}

impl<T: Send + 'static> From<Vec<BoxedDeviceOp<T>>> for DeviceOpVec<T> {
    fn from(ops: Vec<BoxedDeviceOp<T>>) -> Self {
        Self::new(ops)
    }
}

// New names — old names kept as re-exports for backwards compatibility.

#[cfg(test)]
mod send_bounds {
    use super::*;

    fn assert_send<T: Send>() {}

    /// `DeviceOp: Send` and `DeviceFuture: Send` are public API promises —
    /// they are what let a launch be awaited on a multi-threaded executor.
    /// These are compile-time checks; the function body running is incidental.
    #[test]
    fn ops_and_futures_are_send() {
        assert_send::<Value<std::sync::Arc<u8>>>();
        assert_send::<crate::device_future::DeviceFuture<i32, Value<i32>>>();
    }
}
