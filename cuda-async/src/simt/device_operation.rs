/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lazy, composable GPU operations and combinator types.
//!
//! The [`DeviceOperation`] trait is the core abstraction. Each operation
//! describes GPU work without binding to a stream. Combinators (`and_then`,
//! `zip`, `apply`, `with_context`) compose operations into dataflow graphs
//! that remain stream-agnostic until scheduling time.
//!
//! # Scheduling model
//!
//! | Method       | What it does                                                      |
//! |--------------|-------------------------------------------------------------------|
//! | [`schedule`] | Pairs the operation with a stream, returns a [`DeviceFuture`].    |
//! | [`sync`]     | Shorthand: schedule + execute + synchronize on the default device.|
//! | [`sync_on`]  | Execute and synchronize on a specific stream.                     |
//! | [`async_on`] | `unsafe`: execute on a specific stream **without** synchronizing.  |
//!
//! [`schedule`]: DeviceOperation::schedule
//! [`sync`]: DeviceOperation::sync
//! [`sync_on`]: DeviceOperation::sync_on
//! [`async_on`]: DeviceOperation::async_on
//! [`DeviceFuture`]: crate::simt::device_future::DeviceFuture

use crate::simt::device_context::with_default_device_policy;
use crate::simt::device_future::DeviceFuture;
use crate::simt::error::DeviceError;
use crate::simt::scheduling_policies::SchedulingPolicy;
use cuda_core::{CudaContext, CudaStream};
use std::cell::UnsafeCell;
use std::future::IntoFuture;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// CUDA device ordinal. Type alias for readability.
pub type Device = usize;

/// Binds a [`DeviceOperation`] to a concrete CUDA stream and context for
/// execution.
///
/// Created by the scheduling policy when an operation is scheduled. Passed to
/// [`DeviceOperation::execute`] to provide the stream and context.
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Device ordinal derived from the CUDA context.
    device: Device,
    /// Stream on which GPU work will be enqueued.
    cuda_stream: Arc<CudaStream>,
    /// CUDA context that owns the stream.
    cuda_context: Arc<CudaContext>,
}

impl ExecutionContext {
    /// Constructs a context from a stream, deriving the device and CUDA context
    /// from the stream's owning context.
    pub fn new(cuda_stream: Arc<CudaStream>) -> Self {
        let cuda_context = Arc::clone(cuda_stream.context());
        let device = cuda_context.ordinal();
        Self {
            cuda_stream,
            cuda_context,
            device,
        }
    }

    /// Returns the CUDA stream.
    pub fn get_cuda_stream(&self) -> &Arc<CudaStream> {
        &self.cuda_stream
    }

    /// Returns the CUDA context.
    pub fn get_cuda_context(&self) -> &Arc<CudaContext> {
        &self.cuda_context
    }

    /// Returns the device ordinal.
    pub fn get_device_id(&self) -> Device {
        self.device
    }
}

/// A lazy, composable GPU operation that may be executed synchronously or
/// asynchronously.
///
/// `DeviceOperation` is the core trait of the `cuda-async` crate. It
/// represents a unit of GPU work that is **stream-agnostic**: the concrete
/// CUDA stream is chosen only at scheduling time, not at construction time.
///
/// # Composing operations
///
/// Combinators build complex dataflow graphs without touching streams:
///
/// | Combinator                | Effect                                          |
/// |---------------------------|-------------------------------------------------|
/// | [`and_then`]              | Sequence: `A` then `f(result_a)`.               |
/// | [`and_then_with_context`] | Like `and_then` but the closure sees the stream.|
/// | [`apply`]                 | Alias for `and_then`.                           |
/// | [`arc`]                   | Wraps the output in `Arc<T>`.                   |
/// | `zip!`                  | Runs two or three operations, returns a tuple.  |
/// | `unzip!`                | Splits a tuple-producing operation.             |
///
/// # Executing operations
///
/// | Method       | Picks stream via         | Blocks? | Async? | `unsafe`? |
/// |--------------|--------------------------|---------|--------|-----------|
/// | [`schedule`] | `SchedulingPolicy`       | No      | Yes    | No        |
/// | `.await`     | Default policy           | No      | Yes    | No        |
/// | [`sync`]     | Default policy           | Yes     | No     | No        |
/// | [`sync_on`]  | Caller-provided stream   | Yes     | No     | No        |
/// | [`async_on`] | Caller-provided stream   | No      | No     | **`unsafe`** |
///
/// [`async_on`] is the only one that is `unsafe`, and it is the only one that
/// returns while GPU work may still be in flight: the caller must synchronize
/// the stream before consuming device-side outputs. The other four either block
/// or hand back a future that does the waiting.
///
/// # Implementors
///
/// Implement [`execute`] to describe the GPU work. The blanket [`IntoFuture`]
/// impl must also be provided (typically via the same boilerplate that
/// delegates to `with_default_device_policy`).
///
/// [`and_then`]: DeviceOperation::and_then
/// [`and_then_with_context`]: DeviceOperation::and_then_with_context
/// [`apply`]: DeviceOperation::apply
/// [`arc`]: DeviceOperation::arc
/// [`schedule`]: DeviceOperation::schedule
/// [`sync`]: DeviceOperation::sync
/// [`sync_on`]: DeviceOperation::sync_on
/// [`async_on`]: DeviceOperation::async_on
/// [`execute`]: DeviceOperation::execute
pub trait DeviceOperation:
    Send + Sized + IntoFuture<Output = Result<<Self as DeviceOperation>::Output, DeviceError>>
{
    /// The value produced when the operation completes successfully.
    ///
    /// The `'static` bound exists because a cancelled in-flight result is
    /// parked in the [`reclaim`](crate::simt::reclaim) limbo and dropped only
    /// after the GPU work completes, which can be after any non-`'static`
    /// borrow it carries would have expired.
    type Output: Send + 'static;

    /// Submits GPU work to the stream in `context` and returns the result.
    ///
    /// # Safety
    ///
    /// GPU work may still be in flight when this returns. The caller must
    /// synchronize the stream before reading device-side outputs.
    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOperation>::Output, DeviceError>;

    /// Pairs this operation with a stream chosen by `policy` and returns a
    /// [`DeviceFuture`] that can be `.await`-ed.
    fn schedule<P: SchedulingPolicy>(
        self,
        policy: &P,
    ) -> Result<DeviceFuture<<Self as DeviceOperation>::Output, Self>, DeviceError> {
        policy.schedule(self)
    }

    /// Chains a dependent operation: executes `self`, then passes its output
    /// to `f` to produce the next operation.
    fn and_then<O: Send, DO, F>(
        self,
        f: F,
    ) -> AndThen<<Self as DeviceOperation>::Output, Self, O, DO, F>
    where
        DO: DeviceOperation<Output = O>,
        F: FnOnce(<Self as DeviceOperation>::Output) -> DO,
    {
        AndThen {
            op: self,
            closure: f,
        }
    }

    /// Like [`and_then`](Self::and_then), but the closure also receives the
    /// [`ExecutionContext`] so it can inspect the stream or device.
    fn and_then_with_context<O: Send, DO, F>(
        self,
        f: F,
    ) -> AndThenWithContext<<Self as DeviceOperation>::Output, Self, O, DO, F>
    where
        DO: DeviceOperation<Output = O>,
        F: FnOnce(&ExecutionContext, <Self as DeviceOperation>::Output) -> DO,
    {
        AndThenWithContext {
            op: self,
            closure: f,
        }
    }

    /// Wraps the output in an [`Arc`], useful when the result must be shared
    /// across multiple consumers.
    fn arc(self) -> DeviceOperationArc<<Self as DeviceOperation>::Output, Self>
    where
        <Self as DeviceOperation>::Output: Sync,
    {
        DeviceOperationArc { op: self }
    }

    /// Alias for [`and_then`](Self::and_then).
    fn apply<O: Send, DO, F>(
        self,
        f: F,
    ) -> AndThen<<Self as DeviceOperation>::Output, Self, O, DO, F>
    where
        DO: DeviceOperation<Output = O>,
        F: FnOnce(<Self as DeviceOperation>::Output) -> DO,
    {
        self.and_then(f)
    }

    /// Executes the operation synchronously on the default device using the
    /// thread-local scheduling policy. Blocks until the stream is idle.
    fn sync(self) -> Result<<Self as DeviceOperation>::Output, DeviceError> {
        with_default_device_policy(|policy| policy.sync(self))?
    }

    /// Executes the operation on `stream` **without** synchronizing.
    ///
    /// # Safety
    ///
    /// GPU work may still be in flight when this returns. The caller must
    /// synchronize `stream` before consuming device-side outputs.
    unsafe fn async_on(
        self,
        stream: &Arc<CudaStream>,
    ) -> Result<<Self as DeviceOperation>::Output, DeviceError> {
        let ctx = ExecutionContext::new(Arc::clone(stream));
        unsafe { self.execute(&ctx) }
    }

    /// Executes the operation on `stream` and synchronizes before returning.
    fn sync_on(
        self,
        stream: &Arc<CudaStream>,
    ) -> Result<<Self as DeviceOperation>::Output, DeviceError> {
        let ctx = ExecutionContext::new(Arc::clone(stream));
        let res = unsafe { self.execute(&ctx) };
        finish_sync(res, stream.synchronize())
    }
}

fn finish_sync<T>(
    operation_result: Result<T, DeviceError>,
    synchronize_result: Result<(), cuda_core::DriverError>,
) -> Result<T, DeviceError> {
    let output = operation_result?;
    synchronize_result.map_err(DeviceError::Driver)?;
    Ok(output)
}

// --- Combinators ---

/// Wraps a [`DeviceOperation`] whose output is `Arc`-wrapped.
///
/// Produced by [`DeviceOperation::arc`].
pub struct DeviceOperationArc<I: Send + Sync, DI: DeviceOperation<Output = I>> {
    /// The inner operation whose result will be wrapped in [`Arc`].
    op: DI,
}

/// # Safety
///
/// `DI` is `Send` (required by `DeviceOperation`), and `I: Send + Sync`
/// ensures the `Arc<I>` output is safe to transfer.
unsafe impl<I: Send + Sync, DI: DeviceOperation<Output = I>> Send for DeviceOperationArc<I, DI> {}

/// Executes the inner operation and wraps the result in [`Arc`].
impl<I: Send + Sync + 'static, DI: DeviceOperation<Output = I>> DeviceOperation
    for DeviceOperationArc<I, DI>
{
    type Output = Arc<I>;

    unsafe fn execute(self, context: &ExecutionContext) -> Result<Arc<I>, DeviceError> {
        unsafe {
            let val = self.op.execute(context)?;
            Ok(Arc::new(val))
        }
    }
}

/// Schedules via the thread-local default policy.
impl<I: Send + Sync + 'static, DI: DeviceOperation<Output = I>> IntoFuture
    for DeviceOperationArc<I, DI>
{
    type Output = Result<Arc<I>, DeviceError>;
    type IntoFuture = DeviceFuture<Arc<I>, DeviceOperationArc<I, DI>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
        }
    }
}

/// Sequential composition: execute `DI`, then pass its output through `F` to
/// produce a second operation `DO`, and execute that.
///
/// Produced by [`DeviceOperation::and_then`].
pub struct AndThen<I: Send, DI, O: Send, DO, F>
where
    DI: DeviceOperation<Output = I>,
    DO: DeviceOperation<Output = O>,
    F: FnOnce(I) -> DO,
{
    /// First operation.
    op: DI,
    /// Closure mapping the first operation's output to the second operation.
    closure: F,
}

/// # Safety
///
/// Both `DI` and `F` are `Send`. The struct owns them exclusively, so
/// transferring across threads is safe.
unsafe impl<I: Send, DI, O: Send, DO, F> Send for AndThen<I, DI, O, DO, F>
where
    DI: DeviceOperation<Output = I>,
    DO: DeviceOperation<Output = O>,
    F: FnOnce(I) -> DO + Send,
{
}

/// Executes the first operation, feeds its result to the closure, then
/// executes the resulting second operation on the same stream.
impl<I: Send, DI, O: Send + 'static, DO, F> DeviceOperation for AndThen<I, DI, O, DO, F>
where
    DI: DeviceOperation<Output = I>,
    DO: DeviceOperation<Output = O>,
    F: FnOnce(I) -> DO + Send,
{
    type Output = O;

    unsafe fn execute(self, context: &ExecutionContext) -> Result<O, DeviceError> {
        unsafe {
            let input = self.op.execute(context)?;
            let output_op = (self.closure)(input);
            output_op.execute(context)
        }
    }
}

/// Schedules via the thread-local default policy.
impl<I: Send, DI, O: Send + 'static, DO, F> IntoFuture for AndThen<I, DI, O, DO, F>
where
    DI: DeviceOperation<Output = I>,
    DO: DeviceOperation<Output = O>,
    F: FnOnce(I) -> DO + Send,
{
    type Output = Result<O, DeviceError>;
    type IntoFuture = DeviceFuture<O, AndThen<I, DI, O, DO, F>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
        }
    }
}

/// Like [`AndThen`] but the closure additionally receives the
/// [`ExecutionContext`], giving access to the stream and device.
///
/// Produced by [`DeviceOperation::and_then_with_context`].
pub struct AndThenWithContext<I: Send, DI, O: Send, DO, F>
where
    DI: DeviceOperation<Output = I>,
    DO: DeviceOperation<Output = O>,
    F: FnOnce(&ExecutionContext, I) -> DO,
{
    /// First operation.
    op: DI,
    /// Closure mapping `(context, result)` to the second operation.
    closure: F,
}

/// # Safety
///
/// Both `DI` and `F` are `Send`. The struct owns them exclusively.
unsafe impl<I: Send, DI, O: Send, DO, F> Send for AndThenWithContext<I, DI, O, DO, F>
where
    DI: DeviceOperation<Output = I>,
    DO: DeviceOperation<Output = O>,
    F: FnOnce(&ExecutionContext, I) -> DO + Send,
{
}

/// Executes the first operation, then passes `(context, result)` to the
/// closure and executes the resulting second operation.
impl<I: Send, DI, O: Send + 'static, DO, F> DeviceOperation for AndThenWithContext<I, DI, O, DO, F>
where
    DI: DeviceOperation<Output = I>,
    DO: DeviceOperation<Output = O>,
    F: FnOnce(&ExecutionContext, I) -> DO + Send,
{
    type Output = O;

    unsafe fn execute(self, context: &ExecutionContext) -> Result<O, DeviceError> {
        unsafe {
            let input = self.op.execute(context)?;
            let output_op = (self.closure)(context, input);
            output_op.execute(context)
        }
    }
}

/// Schedules via the thread-local default policy.
impl<I: Send, DI, O: Send + 'static, DO, F> IntoFuture for AndThenWithContext<I, DI, O, DO, F>
where
    DI: DeviceOperation<Output = I>,
    DO: DeviceOperation<Output = O>,
    F: FnOnce(&ExecutionContext, I) -> DO + Send,
{
    type Output = Result<O, DeviceError>;
    type IntoFuture = DeviceFuture<O, AndThenWithContext<I, DI, O, DO, F>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
        }
    }
}

/// A [`DeviceOperation`] that immediately returns a pre-computed value without
/// touching the GPU.
pub struct Value<T>(T);

/// # Safety
///
/// `Value` holds only the inner `T`. If `T` is `Send` the wrapper is too;
/// the bound is enforced by the `DeviceOperation` impl.
unsafe impl<T> Send for Value<T> {}

/// Returns the wrapped value directly -- no GPU work is performed.
impl<T: Send + 'static> DeviceOperation for Value<T> {
    type Output = T;

    unsafe fn execute(self, _context: &ExecutionContext) -> Result<T, DeviceError> {
        Ok(self.0)
    }
}

/// Schedules via the thread-local default policy.
impl<T: Send + 'static> IntoFuture for Value<T> {
    type Output = Result<T, DeviceError>;
    type IntoFuture = DeviceFuture<T, Value<T>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
        }
    }
}

/// Wraps `x` in a [`Value`] operation that returns it immediately.
pub fn value<T: Send>(x: T) -> Value<T> {
    Value(x)
}

/// Converts any `Send` value into a no-op [`DeviceOperation`] via [`value`].
pub trait IntoDeviceOperation<T: Send> {
    /// Wraps `self` into a [`Value`] device operation.
    fn device_operation(self) -> Value<T>;
}

impl<T: Send> IntoDeviceOperation<T> for T {
    fn device_operation(self) -> Value<T> {
        value(self)
    }
}

/// Deferred-closure operation: the closure produces the real operation at
/// execution time rather than at construction time.
///
/// Useful when building the inner operation requires state only available
/// after scheduling (though it does not receive the [`ExecutionContext`] --
/// see [`StreamOperation`] for that).
pub struct Empty<O: Send, DO: DeviceOperation<Output = O>, F: FnOnce() -> DO> {
    /// Closure that produces the inner operation.
    closure: F,
}

/// Wraps a closure in an [`Empty`] deferred operation.
pub fn empty<O: Send, DO: DeviceOperation<Output = O>, F: FnOnce() -> DO>(
    closure: F,
) -> Empty<O, DO, F> {
    Empty { closure }
}

/// # Safety
///
/// The closure `F` and its produced operation `DO` are both `Send`. The
/// struct owns the closure exclusively.
unsafe impl<O: Send, DO: DeviceOperation<Output = O>, F: FnOnce() -> DO> Send for Empty<O, DO, F> {}

/// Invokes the closure to produce the inner operation, then executes it.
impl<O: Send + 'static, DO: DeviceOperation<Output = O>, F: FnOnce() -> DO> DeviceOperation
    for Empty<O, DO, F>
{
    type Output = O;

    unsafe fn execute(self, context: &ExecutionContext) -> Result<O, DeviceError> {
        unsafe {
            let op = (self.closure)();
            op.execute(context)
        }
    }
}

/// Schedules via the thread-local default policy.
impl<O: Send + 'static, DO: DeviceOperation<Output = O>, F: FnOnce() -> DO> IntoFuture
    for Empty<O, DO, F>
{
    type Output = Result<O, DeviceError>;
    type IntoFuture = DeviceFuture<O, Empty<O, DO, F>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
        }
    }
}

/// Pair combinator: executes two operations sequentially on the same stream
/// and returns both results as a tuple.
///
/// Constructed via `_zip` or the `zip!` macro.
pub struct Zip<T1: Send, T2: Send, A: DeviceOperation<Output = T1>, B: DeviceOperation<Output = T2>>
{
    phantom: PhantomData<(T1, T2)>,
    /// First operation.
    a: A,
    /// Second operation.
    b: B,
}

/// # Safety
///
/// Both `A` and `B` are `Send` (required by `DeviceOperation`).
unsafe impl<T1: Send, T2: Send, A: DeviceOperation<Output = T1>, B: DeviceOperation<Output = T2>>
    Send for Zip<T1, T2, A, B>
{
}

/// Constructs a [`Zip`] from two operations.
fn _zip<T1: Send, T2: Send, A: DeviceOperation<Output = T1>, B: DeviceOperation<Output = T2>>(
    a: A,
    b: B,
) -> Zip<T1, T2, A, B> {
    Zip {
        phantom: PhantomData,
        a,
        b,
    }
}

/// Executes `a` then `b` on the same stream, returning `(T1, T2)`.
impl<
        T1: Send + 'static,
        T2: Send + 'static,
        A: DeviceOperation<Output = T1>,
        B: DeviceOperation<Output = T2>,
    > DeviceOperation for Zip<T1, T2, A, B>
{
    type Output = (T1, T2);

    unsafe fn execute(self, context: &ExecutionContext) -> Result<(T1, T2), DeviceError> {
        unsafe {
            let a = self.a.execute(context)?;
            let b = self.b.execute(context)?;
            Ok((a, b))
        }
    }
}

/// Schedules via the thread-local default policy.
impl<
        T1: Send + 'static,
        T2: Send + 'static,
        A: DeviceOperation<Output = T1>,
        B: DeviceOperation<Output = T2>,
    > IntoFuture for Zip<T1, T2, A, B>
{
    type Output = Result<(T1, T2), DeviceError>;
    type IntoFuture = DeviceFuture<(T1, T2), Zip<T1, T2, A, B>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
        }
    }
}

/// Trait enabling `.zip()` on tuples of [`DeviceOperation`]s.
///
/// Implemented for 2-tuples and 3-tuples.
pub trait Zippable<I, O: Send> {
    /// Combines the operations into a single operation returning a tuple of
    /// results.
    fn zip(self) -> impl DeviceOperation<Output = O>;
}

/// Zips two operations into a pair.
impl<
        T0: Send + 'static,
        T1: Send + 'static,
        DI0: DeviceOperation<Output = T0>,
        DI1: DeviceOperation<Output = T1>,
    > Zippable<(DI0, DI1), (T0, T1)> for (DI0, DI1)
{
    fn zip(self) -> impl DeviceOperation<Output = (T0, T1)> {
        _zip(self.0, self.1)
    }
}

/// Zips three operations into a triple by nesting two binary zips.
impl<
        T0: Send + 'static,
        T1: Send + 'static,
        T2: Send + 'static,
        DI0: DeviceOperation<Output = T0>,
        DI1: DeviceOperation<Output = T1>,
        DI2: DeviceOperation<Output = T2>,
    > Zippable<(DI0, DI1, DI2), (T0, T1, T2)> for (DI0, DI1, DI2)
{
    fn zip(self) -> impl DeviceOperation<Output = (T0, T1, T2)> {
        let cons = _zip(self.1, self.2);
        let cons = _zip(self.0, cons);
        cons.and_then(|(arg0, (arg1, arg2))| value((arg0, arg1, arg2)))
    }
}

/// Zips one, two, or three [`DeviceOperation`]s into a single operation
/// returning a tuple of results.
///
/// ```ignore
/// let (a, b) = zip!(op_a, op_b).sync()?;
/// let (x, y, z) = zip!(op_x, op_y, op_z).sync()?;
/// ```
#[allow(unused_macros)] // collides with this crate's exported zip!; reconcile at merge
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
}
#[allow(unused_imports)] // kept for parity with oxide's exported form
pub(crate) use zip;

/// Deferred operation that receives the [`ExecutionContext`] before producing
/// the inner operation.
///
/// Unlike [`Empty`], the closure has access to the stream and device, making
/// it possible to build context-dependent operations at execution time.
pub struct StreamOperation<
    O: Send,
    DO: DeviceOperation<Output = O>,
    F: FnOnce(&ExecutionContext) -> DO + Send,
> {
    /// Closure that receives the execution context and produces the inner op.
    f: F,
}

/// Calls the closure with the context, then executes the resulting operation.
impl<
        O: Send + 'static,
        DO: DeviceOperation<Output = O>,
        F: FnOnce(&ExecutionContext) -> DO + Send,
    > DeviceOperation for StreamOperation<O, DO, F>
{
    type Output = O;

    unsafe fn execute(self, context: &ExecutionContext) -> Result<O, DeviceError> {
        unsafe {
            let op = (self.f)(context);
            op.execute(context)
        }
    }
}

/// Wraps a closure that needs the [`ExecutionContext`] into a
/// [`DeviceOperation`].
///
/// The closure is invoked at execution time with the stream and context,
/// and must return a `DeviceOperation` that will be immediately executed.
pub fn with_context<
    O: Send + 'static,
    DO: DeviceOperation<Output = O>,
    F: FnOnce(&ExecutionContext) -> DO + Send,
>(
    f: F,
) -> impl DeviceOperation<Output = O> {
    StreamOperation { f }
}

/// Schedules via the thread-local default policy.
impl<
        O: Send + 'static,
        DO: DeviceOperation<Output = O>,
        F: FnOnce(&ExecutionContext) -> DO + Send,
    > IntoFuture for StreamOperation<O, DO, F>
{
    type Output = Result<O, DeviceError>;
    type IntoFuture = DeviceFuture<O, StreamOperation<O, DO, F>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
        }
    }
}

/// Shared state backing the [`SelectLeft`] / [`SelectRight`] split.
///
/// Holds the source operation and its memoized results. The first selector
/// to execute triggers the source; subsequent selectors read the cached
/// values. Interior mutability is used via [`UnsafeCell`] because execution
/// is always sequential within a single stream.
pub struct Select<T1: Send, T2: Send, DI: DeviceOperation<Output = (T1, T2)>> {
    /// Guards one-shot execution of the source operation.
    computed: AtomicBool,
    /// Source operation. Consumed on the first call to [`execute`](Self::execute).
    input: UnsafeCell<Option<DI>>,
    /// Cached left result.
    left: UnsafeCell<Option<T1>>,
    /// Cached right result.
    right: UnsafeCell<Option<T2>>,
}

impl<T1: Send, T2: Send, DI: DeviceOperation<Output = (T1, T2)>> Select<T1, T2, DI> {
    /// Executes the source operation if it has not been executed yet, caching
    /// both halves of the tuple.
    ///
    /// # Safety
    ///
    /// Must only be called from within a single-stream execution context.
    /// Concurrent calls from different threads would race on the `UnsafeCell`s.
    unsafe fn execute(self: &Arc<Self>, context: &ExecutionContext) -> Result<(), DeviceError> {
        unsafe {
            if !self.computed.load(Ordering::Acquire) {
                let input = self.input.get();
                let input = input.as_mut();
                let input = input.unwrap().take().ok_or_else(|| {
                    crate::simt::error::device_error(
                        context.get_device_id(),
                        "Select operation failed.",
                    )
                })?;
                let (left, right) = input.execute(context)?;
                *self.left.get() = Some(left);
                *self.right.get() = Some(right);
                self.computed.store(true, Ordering::Release);
            }
            Ok(())
        }
    }

    /// Takes the cached left value. Must be called after [`execute`](Self::execute).
    ///
    /// # Safety
    ///
    /// Caller must ensure `execute` has completed and this is called at most once.
    unsafe fn left(&self) -> T1 {
        let cell = self.left.get();
        let cell = unsafe { cell.as_mut() };
        cell.unwrap().take().unwrap()
    }

    /// Takes the cached right value. Must be called after [`execute`](Self::execute).
    ///
    /// # Safety
    ///
    /// Caller must ensure `execute` has completed and this is called at most once.
    unsafe fn right(&self) -> T2 {
        let cell = self.right.get();
        let cell = unsafe { cell.as_mut() };
        cell.unwrap().take().unwrap()
    }
}

/// Operation that extracts the **left** element of an unzipped pair.
///
/// Shares a [`Select`] with its corresponding [`SelectRight`] so the source
/// operation is executed at most once.
pub struct SelectLeft<T1: Send, T2: Send, DI: DeviceOperation<Output = (T1, T2)>> {
    /// Shared memoization state.
    select: Arc<Select<T1, T2, DI>>,
}

/// # Safety
///
/// The `Arc<Select<..>>` is safe to send because `Select` is only mutated
/// through `UnsafeCell` during single-stream execution, never concurrently.
unsafe impl<T1: Send, T2: Send, DI: DeviceOperation<Output = (T1, T2)>> Send
    for SelectLeft<T1, T2, DI>
{
}

/// Triggers the shared source operation (if not yet done) and returns the
/// left element.
impl<T1: Send + 'static, T2: Send + 'static, DI: DeviceOperation<Output = (T1, T2)>> DeviceOperation
    for SelectLeft<T1, T2, DI>
{
    type Output = T1;

    unsafe fn execute(self, context: &ExecutionContext) -> Result<T1, DeviceError> {
        unsafe {
            self.select.execute(context)?;
            Ok(self.select.left())
        }
    }
}

/// Schedules via the thread-local default policy.
impl<T1: Send + 'static, T2: Send + 'static, DI: DeviceOperation<Output = (T1, T2)>> IntoFuture
    for SelectLeft<T1, T2, DI>
{
    type Output = Result<T1, DeviceError>;
    type IntoFuture = DeviceFuture<T1, SelectLeft<T1, T2, DI>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
        }
    }
}

/// Operation that extracts the **right** element of an unzipped pair.
///
/// Shares a [`Select`] with its corresponding [`SelectLeft`] so the source
/// operation is executed at most once.
pub struct SelectRight<T1: Send, T2: Send, DI: DeviceOperation<Output = (T1, T2)>> {
    /// Shared memoization state.
    select: Arc<Select<T1, T2, DI>>,
}

/// # Safety
///
/// See [`SelectLeft`]'s `Send` impl.
unsafe impl<T1: Send, T2: Send, DI: DeviceOperation<Output = (T1, T2)>> Send
    for SelectRight<T1, T2, DI>
{
}

/// Triggers the shared source operation (if not yet done) and returns the
/// right element.
impl<T1: Send + 'static, T2: Send + 'static, DI: DeviceOperation<Output = (T1, T2)>> DeviceOperation
    for SelectRight<T1, T2, DI>
{
    type Output = T2;

    unsafe fn execute(self, context: &ExecutionContext) -> Result<T2, DeviceError> {
        unsafe {
            self.select.execute(context)?;
            Ok(self.select.right())
        }
    }
}

/// Schedules via the thread-local default policy.
impl<T1: Send + 'static, T2: Send + 'static, DI: DeviceOperation<Output = (T1, T2)>> IntoFuture
    for SelectRight<T1, T2, DI>
{
    type Output = Result<T2, DeviceError>;
    type IntoFuture = DeviceFuture<T2, SelectRight<T1, T2, DI>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
        }
    }
}

/// Splits a tuple-producing operation into two independent operations that
/// share execution: the source runs at most once, and each selector extracts
/// one element.
fn _unzip<T1: Send, T2: Send, DI: DeviceOperation<Output = (T1, T2)>>(
    input: DI,
) -> (SelectLeft<T1, T2, DI>, SelectRight<T1, T2, DI>) {
    let select = Select {
        computed: AtomicBool::new(false),
        input: UnsafeCell::new(Some(input)),
        left: UnsafeCell::new(None),
        right: UnsafeCell::new(None),
    };
    let select = Arc::new(select);
    let out1 = SelectLeft {
        select: Arc::clone(&select),
    };
    let out2 = SelectRight { select };
    (out1, out2)
}

/// Trait enabling `.unzip()` on any [`DeviceOperation`] that produces a
/// 2-tuple.
pub trait Unzippable2<T0: Send + 'static, T1: Send + 'static>
where
    Self: DeviceOperation<Output = (T0, T1)>,
{
    /// Splits this operation into two independent operations, one for each
    /// tuple element. The source executes at most once.
    fn unzip(
        self,
    ) -> (
        impl DeviceOperation<Output = T0>,
        impl DeviceOperation<Output = T1>,
    ) {
        _unzip(self)
    }
}

/// Blanket impl: any operation producing `(T0, T1)` is unzippable.
impl<T0: Send + 'static, T1: Send + 'static, DI: DeviceOperation<Output = (T0, T1)>>
    Unzippable2<T0, T1> for DI
{
}

/// Splits a tuple-producing [`DeviceOperation`] into per-element operations.
///
/// ```ignore
/// let (left, right) = unzip!(pair_op);
/// ```
#[allow(unused_macros)] // collides with this crate's exported unzip!; reconcile at merge
macro_rules! unzip {
    ($arg0:expr) => {
        $arg0.unzip()
    };
}
#[allow(unused_imports)] // kept for parity with oxide's exported form
pub(crate) use unzip;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finish_sync_returns_operation_result_after_successful_synchronize() {
        let result = finish_sync::<u32>(Ok(7), Ok(()));

        assert_eq!(result, Ok(7));
    }

    #[test]
    fn finish_sync_preserves_operation_error_after_successful_synchronize() {
        let operation_error = DeviceError::Launch("launch failed".to_string());
        let result = finish_sync::<u32>(Err(operation_error.clone()), Ok(()));

        assert_eq!(result, Err(operation_error));
    }

    #[test]
    fn finish_sync_propagates_synchronize_error_instead_of_panicking() {
        let driver_error =
            cuda_core::DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE);
        let result = finish_sync::<u32>(Ok(7), Err(driver_error));

        assert_eq!(result, Err(DeviceError::Driver(driver_error)));
    }

    #[test]
    fn finish_sync_preserves_operation_error_when_synchronize_also_fails() {
        let operation_error = DeviceError::Launch("launch failed".to_string());
        let driver_error =
            cuda_core::DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE);
        let result = finish_sync::<u32>(Err(operation_error.clone()), Err(driver_error));

        assert_eq!(result, Err(operation_error));
    }
}
