/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Future type that bridges CUDA stream callbacks with Rust's async executor.
//!
//! [`DeviceFuture`] drives a [`DeviceOp`] through a small state machine:
//!
//! ```text
//!   Idle ──first poll──> Executing ──completion signal──> Complete
//!            (execute on the stream,          (hand out the result)
//!             then inline spin, then
//!             register for completion)
//! ```
//!
//! # Cancellation
//!
//! Dropping a future does not cancel submitted GPU work or wait for it.
//! The output drops normally; independently registered resource owners and
//! access leases retire after completion. If the driver cannot establish
//! completion, those owners and leases are deliberately leaked. Forgetting a
//! future also leaks its submission, preserving resources and access exclusion.

use crate::device_operation::{DeviceOp, ExecutionContext};
use crate::error::DeviceError;
use cuda_core::{DriverError, Stream};
use futures::task::AtomicWaker;
use std::future::Future;
use std::mem::MaybeUninit;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

/// State machine for tracking the lifecycle of a device future.
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum DeviceFutureState {
    // The future was created with an error and will resolve immediately on first poll.
    /// The future was created with an error and will resolve immediately.
    Failed,
    // The stream operation has not yet been scheduled. No callback has been added.
    /// The stream operation has not yet been scheduled.
    Idle,
    // The stream operation has been scheduled and a callback has been added to the stream.
    // The callback should be added such that it immediately succeeds the scheduled operation.
    /// The stream operation is in-flight and a completion callback is registered.
    Executing,
    // The callback has been fired, indicating the completion of the stream operation.
    /// The stream callback has fired, indicating the operation is done.
    Complete,
}

/// Shared state between a CUDA stream callback and the async waker.
#[derive(Debug, Default)]
pub struct StreamCallbackState {
    pub(crate) waker: AtomicWaker,
    pub(crate) complete: AtomicBool,
}

impl StreamCallbackState {
    /// Creates a new callback state with the completion flag unset.
    pub fn new() -> Self {
        Self::default()
    }
    /// Marks the operation as complete and wakes the associated task.
    ///
    /// `Release` pairs with the `Acquire` loads in [`Future::poll`]: the poll
    /// that observes `complete == true` hands out the result, so everything
    /// the signalling side did before the store (the reactor's flag read, the
    /// host callback's ordering after the stream work) must be visible to it.
    pub fn signal(&self) {
        self.complete.store(true, Ordering::Release);
        self.waker.wake();
    }

    /// Wakes the task **without** marking the operation complete.
    ///
    /// The re-poll re-examines the stream itself. The reactor uses this for
    /// slots whose stream has faulted: the completion flag can never land,
    /// so the future has to observe the driver error on its own.
    pub fn wake(&self) {
        self.waker.wake();
    }
}

/// Non-blocking health of a stream, as seen by the completion paths.
pub(crate) enum StreamHealth {
    /// The stream is recording a graph. Querying or synchronizing it would
    /// invalidate the capture, so callers must not touch it.
    Capturing,
    /// Work is still in flight.
    Busy,
    /// All enqueued work has completed.
    Idle,
    /// The driver reports an error for the stream — typically a sticky
    /// fault (illegal address, `trap`, ...) that has killed the context.
    Faulted(DriverError),
}

/// Probes `stream` without blocking and without disturbing a capture.
///
/// Requires the stream's context to be current, or at least a context of
/// the same process on the calling thread; callers on threads that never
/// touched CUDA bind the device first.
pub(crate) fn probe_stream(stream: &Stream) -> StreamHealth {
    let mut status = MaybeUninit::uninit();
    // SAFETY: the stream handle is valid (RAII wrapper) and `status` is a
    // valid out-pointer. `cuStreamIsCapturing` is legal on a capturing stream.
    let code =
        unsafe { cuda_bindings::cuStreamIsCapturing(stream.cu_stream(), status.as_mut_ptr()) };
    if code != cuda_bindings::cudaError_enum_CUDA_SUCCESS {
        return StreamHealth::Faulted(DriverError(code));
    }
    // SAFETY: the driver initialized `status` on success.
    let status = unsafe { status.assume_init() };
    if status != cuda_bindings::CUstreamCaptureStatus_enum_CU_STREAM_CAPTURE_STATUS_NONE {
        return StreamHealth::Capturing;
    }
    // SAFETY: see above; not capturing, so a query is permitted.
    match stream.query() {
        Ok(true) => StreamHealth::Idle,
        Ok(false) => StreamHealth::Busy,
        Err(e) => StreamHealth::Faulted(e),
    }
}

/// A future that executes a [`DeviceOp`] on a CUDA stream and resolves upon completion.
///
/// # Cancellation and retained resources
///
/// Dropping a future never waits for GPU work. Its possibly borrowed output is
/// dropped normally, just as an output projection can discard it during enqueue.
/// Every operation must independently register GPU/DMA owners and access leases
/// before submitting work. The submission retires those owners on a CUDA event;
/// error paths retain resources if completion cannot be established. Forgetting
/// a future leaks its submission and therefore preserves ownership and access
/// exclusion. Successful completion releases registered owners before delivery.
#[derive(Debug)]
pub struct DeviceFuture<T: Send, DO: DeviceOp<Output = T>> {
    pub(crate) device_operation: Option<DO>,
    pub(crate) execution_context: Option<ExecutionContext>,
    pub(crate) result: Option<T>,
    pub(crate) error: Option<DeviceError>,
    pub(crate) state: DeviceFutureState,
    pub(crate) callback_state: Option<Arc<StreamCallbackState>>,
}

impl<T: Send, DO: DeviceOp<Output = T>> DeviceFuture<T, DO> {
    /// Creates an idle device future with no operation or execution context set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a device future with the context's stream, device, and pool.
    /// The submission is independent of any clones of the supplied context:
    /// one future's completion must not release another future's resources.
    pub fn scheduled(op: DO, ctx: ExecutionContext) -> Self {
        // Spelled out rather than `..Default::default()`: struct update
        // syntax moves out of the default value, which `Drop` forbids.
        Self {
            device_operation: Some(op),
            execution_context: Some(ctx.fresh_submission()),
            result: None,
            error: None,
            state: DeviceFutureState::Idle,
            callback_state: None,
        }
    }

    /// Create a future that is pre-loaded with an error.
    ///
    /// On first poll it immediately returns `Poll::Ready(Err(error))`.
    /// This is used by `IntoFuture` implementations to surface scheduling
    /// failures without panicking.
    pub fn failed(error: DeviceError) -> Self {
        Self {
            execution_context: None,
            device_operation: None,
            state: DeviceFutureState::Failed,
            callback_state: None,
            result: None,
            error: Some(error),
        }
    }

    /// Registers a host callback on the CUDA stream to signal completion.
    ///
    /// # Safety
    /// The execution context's stream must be valid for the lifetime of the callback.
    unsafe fn register_callback(
        &self,
        waker_state: Arc<StreamCallbackState>,
    ) -> Result<(), DeviceError> {
        let ctx = self
            .execution_context
            .as_ref()
            .ok_or(DeviceError::Internal(
                "Cannot execute future without setting stream on which to execute.".to_string(),
            ))?;
        // Completion strategy, runtime-selectable via `CUDA_ASYNC_HOST_SYNC`:
        //   unset -> flag-write reactor when compiled in, else the host hop
        //   spin  -> cuLaunchHostFunc_v2 + CU_HOST_TASK_SPINWAIT (the driver
        //            spin-waits its host-task thread: lower callback latency at
        //            the cost of a busy core)
        //   block -> cuLaunchHostFunc_v2 + CU_HOST_TASK_BLOCKING
        // The explicit modes bypass the reactor so all three completion paths
        // can be A/B'd from a single build.
        fn host_task_sync_mode() -> Option<::core::ffi::c_uint> {
            static MODE: std::sync::OnceLock<Option<::core::ffi::c_uint>> =
                std::sync::OnceLock::new();
            // CUDA driver flag bindings have platform-dependent integer types, so FFI calls cast them as `_`.
            *MODE.get_or_init(
                || match std::env::var("CUDA_ASYNC_HOST_SYNC").ok()?.as_str() {
                    "spin" | "spinwait" => Some(cuda_bindings::CU_HOST_TASK_SPINWAIT),
                    "block" | "blocking" => Some(cuda_bindings::CU_HOST_TASK_BLOCKING),
                    _ => None,
                },
            )
        }
        if let Some(mode) = host_task_sync_mode() {
            ctx.get_cuda_stream()
                .launch_host_function_with_sync_mode(move || waker_state.signal(), mode)?;
            return Ok(());
        }
        #[cfg(not(loom))]
        {
            // Flag-write reactor path; fall back to the host-function hop if
            // the slot pool is exhausted or stream mem-ops are unavailable.
            if crate::reactor::register(ctx.get_cuda_stream(), waker_state.clone()).is_ok() {
                return Ok(());
            }
        }
        ctx.get_cuda_stream()
            .launch_host_function(move || waker_state.signal())?;
        Ok(())
    }
    /// Executes the stored device operation on the associated stream.
    fn execute(&mut self) -> Result<(), DeviceError> {
        let ctx = self
            .execution_context
            .as_ref()
            .ok_or(DeviceError::Internal(
                "Cannot execute future without setting stream on which to execute.".to_string(),
            ))?;
        let operation = self.device_operation.take().ok_or(DeviceError::Internal(
            "Unable to execute future: No operation has been set.".to_string(),
        ))?;
        let out = unsafe { operation.execute(ctx) }?;
        self.result = Some(out);
        Ok(())
    }

    fn take_completed_result(&mut self) -> T {
        if let Some(ctx) = &self.execution_context {
            // Called only after the stream or its completion marker finished.
            unsafe { ctx.complete() };
        }
        self.result
            .take()
            .expect("Expected future result to be Some.")
    }

    /// Release the host output immediately; its GPU owners belong to the
    /// submission, which retires independently when the context is dropped.
    fn release_in_flight_result(&mut self) {
        drop(self.result.take());
    }
}

impl<T: Send, DO: DeviceOp<Output = T>> Drop for DeviceFuture<T, DO> {
    fn drop(&mut self) {
        self.release_in_flight_result();
    }
}

impl<T: Send, DO: DeviceOp<Output = T>> Default for DeviceFuture<T, DO> {
    fn default() -> Self {
        Self {
            device_operation: None,
            execution_context: None,
            result: None,
            error: None,
            state: DeviceFutureState::Idle,
            callback_state: None,
        }
    }
}

impl<T: Send, DO: DeviceOp<Output = T>> Unpin for DeviceFuture<T, DO> {}

impl<T: Send, DO: DeviceOp<Output = T>> Future for DeviceFuture<T, DO> {
    type Output = Result<T, DeviceError>;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.state == DeviceFutureState::Failed {
            self.state = DeviceFutureState::Complete;
            let error = self
                .error
                .take()
                .expect("Failed state must carry an error.");
            return Poll::Ready(Err(error));
        }

        // If this is being polled, it needs a waker.
        if self.callback_state.is_none() {
            self.callback_state = Some(Arc::new(StreamCallbackState::new()));
        }
        let waker_state = self.callback_state.as_ref().cloned().expect("Impossible.");
        match self.state {
            DeviceFutureState::Idle => {
                // Acquire the thread-local execution lock. The guard lives to
                // the end of this arm and releases the lock on every exit path,
                // including a panic inside `execute`. The GPU work is submitted
                // by then; completion is signalled asynchronously.
                let _execution_lock = match crate::device_operation::acquire_execution_lock() {
                    Ok(guard) => guard,
                    Err(e) => {
                        self.state = DeviceFutureState::Complete;
                        return Poll::Ready(Err(e));
                    }
                };
                // Initialize the waker.
                waker_state.waker.register(cx.waker());
                // Execute this future's operation.
                if let Err(e) = self.execute() {
                    self.state = DeviceFutureState::Complete;
                    return Poll::Ready(Err(e));
                }
                // Inline fast path: bounded spin on cuStreamQuery before any
                // completion registration. Microsecond-scale waits are too
                // short for a waker round trip (reactor/host-fn -> waker ->
                // scheduler -> re-poll); spinning at the wait site resolves
                // short pipelines at sync-like latency. Budget-bounded so
                // long pipelines fall through to the reactor/callback path.
                // A query error is the stream's (sticky) fault: no completion
                // signal will ever arrive, so it is the future's error.
                // Default 20 us ≈ Q3 of measured decode-step kernel durations.
                // `CUDA_ASYNC_SPIN_BUDGET_US=0` forces every pipeline through
                // the completion-notification path (used by correctness tests
                // so the reactor is actually exercised).
                fn inline_spin_budget_us() -> u64 {
                    static BUDGET: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
                    *BUDGET.get_or_init(|| {
                        std::env::var("CUDA_ASYNC_SPIN_BUDGET_US")
                            .ok()
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(20)
                    })
                }
                let spin_outcome: Result<bool, DriverError> = 'spin: {
                    if inline_spin_budget_us() == 0 {
                        break 'spin Ok(false);
                    }
                    let Some(ctx) = self.execution_context.as_ref() else {
                        break 'spin Ok(false);
                    };
                    let deadline = std::time::Instant::now()
                        + std::time::Duration::from_micros(inline_spin_budget_us());
                    loop {
                        match ctx.get_cuda_stream().query() {
                            Ok(true) => break 'spin Ok(true),
                            Ok(false) => {}
                            Err(e) => break 'spin Err(e),
                        }
                        if std::time::Instant::now() >= deadline {
                            break 'spin Ok(false);
                        }
                        std::hint::spin_loop();
                    }
                };
                match spin_outcome {
                    Ok(true) => {
                        self.state = DeviceFutureState::Complete;
                        return Poll::Ready(Ok(self.take_completed_result()));
                    }
                    Ok(false) => {}
                    Err(e) => {
                        // The output can drop normally. The submission preserves
                        // owners and access leases when completion is unknown.
                        self.state = DeviceFutureState::Complete;
                        return Poll::Ready(Err(DeviceError::Driver(e)));
                    }
                }
                // Add the callback. We only want to do this once.
                if let Err(e) = unsafe { self.register_callback(waker_state.clone()) } {
                    self.state = DeviceFutureState::Complete;
                    return Poll::Ready(Err(e));
                }
                // Transition the future's state to "Executing." The lock guard
                // drops on return.
                self.state = DeviceFutureState::Executing;
                Poll::Pending
            }
            DeviceFutureState::Executing => {
                // The future may have been polled by the waker firing or by some other mechanism.
                // Check if the complete flag has been set by the callback.
                if waker_state.complete.load(Ordering::Acquire) {
                    self.state = DeviceFutureState::Complete;
                    // If the future was polled by some mechanism other than the waker,
                    // then the old waker still may fire, but the future will not be polled
                    // again if we return Poll::Ready.
                    return Poll::Ready(Ok(self.take_completed_result()));
                }
                // Not complete. This is a spurious wake, or the reactor woke
                // us because the stream faulted (its flag write can never
                // land, so the flag path alone would wait forever). Probe
                // the stream: a driver error is the future's error; an idle
                // stream means the work finished and only the signal is late.
                // A capturing stream must not be queried; wait for the flag.
                let health = match self.execution_context.as_ref() {
                    Some(ctx) => {
                        // The poll may run on an executor thread that never
                        // touched CUDA; the probe needs a current context.
                        let _ = ctx.context().bind_to_thread();
                        probe_stream(ctx.get_cuda_stream())
                    }
                    None => StreamHealth::Busy,
                };
                match health {
                    StreamHealth::Faulted(e) => {
                        // Submission owners and leases remain retained after
                        // the fault; the output itself can drop normally.
                        self.state = DeviceFutureState::Complete;
                        return Poll::Ready(Err(DeviceError::Driver(e)));
                    }
                    StreamHealth::Idle => {
                        self.state = DeviceFutureState::Complete;
                        return Poll::Ready(Ok(self.take_completed_result()));
                    }
                    StreamHealth::Busy | StreamHealth::Capturing => {}
                }
                // The future is still incomplete. Update the waker to the latest context.
                waker_state.waker.register(cx.waker());
                // Check if the callback has fired after updating the waker.
                // If the callback triggers the old waker before the new waker is registered,
                // the newly registered waker will never be called.
                if waker_state.complete.load(Ordering::Acquire) {
                    self.state = DeviceFutureState::Complete;
                    Poll::Ready(Ok(self.take_completed_result()))
                } else {
                    Poll::Pending
                }
            }
            DeviceFutureState::Complete => {
                // We set the future's state to complete before returning Poll::Ready.
                // The executor *should* never poll this task again.
                panic!("Poll called after completion.");
            }
            DeviceFutureState::Failed => {
                // Already handled above; this arm is unreachable.
                unreachable!();
            }
        }
    }
}

#[cfg(test)]
mod release_tests {
    //! Host-only tests for the in-flight release path: the future is built
    //! directly in each state and the wait is a closure, so no driver is
    //! needed. The GPU-backed variant lives in `tests/drop_in_flight.rs`.

    use super::*;
    use crate::device_operation::Value;
    use std::sync::atomic::AtomicUsize;

    fn future_in_state<T: Send>(
        state: DeviceFutureState,
        result: Option<T>,
    ) -> DeviceFuture<T, Value<T>> {
        DeviceFuture {
            device_operation: None,
            execution_context: None,
            result,
            error: None,
            state,
            callback_state: None,
        }
    }

    #[test]
    fn cancellation_drops_borrowed_outputs_without_waiting() {
        struct Borrowed<'a>(&'a AtomicUsize);
        impl Drop for Borrowed<'_> {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
        let drops = AtomicUsize::new(0);
        let future = future_in_state(DeviceFutureState::Executing, Some(Borrowed(&drops)));
        drop(future);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn dropping_failed_future_is_a_noop() {
        let future: DeviceFuture<u32, Value<u32>> =
            DeviceFuture::failed(DeviceError::Internal("never scheduled".into()));
        drop(future);
    }
}

#[cfg(test)]
mod callback_state_tests {
    //! Host-only contract tests for [`StreamCallbackState`], the shared state
    //! a completion signal (host callback or reactor flag) fires against.
    //! Patterned on tokio's `sync/tests/atomic_waker.rs`; no GPU required.

    use super::StreamCallbackState;
    use futures::task::{waker, ArcWake};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct CountingWaker(AtomicUsize);
    impl ArcWake for CountingWaker {
        fn wake_by_ref(arc_self: &Arc<Self>) {
            arc_self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// tokio `wake_without_register`: signaling with no registered waker is a
    /// no-op, not a panic. This is the exact property our cancellation story
    /// leans on — a future dropped mid-flight leaves the reactor to fire
    /// `signal()` against state with no live waker, and that must be benign.
    #[test]
    fn signal_without_registered_waker_is_a_noop() {
        let state = StreamCallbackState::new();
        state.signal(); // must not panic
        assert!(state.complete.load(Ordering::Relaxed));
        state.signal(); // idempotent: second signal is also benign
        assert!(state.complete.load(Ordering::Relaxed));
    }

    /// A registered waker is woken exactly once by a single signal, and the
    /// completion flag is observable afterward (the `Executing` poll arm
    /// reads it).
    #[test]
    fn signal_wakes_registered_waker_and_sets_complete() {
        let state = StreamCallbackState::new();
        let counter = Arc::new(CountingWaker(AtomicUsize::new(0)));
        state.waker.register(&waker(counter.clone()));
        assert_eq!(counter.0.load(Ordering::SeqCst), 0);
        state.signal();
        assert_eq!(counter.0.load(Ordering::SeqCst), 1);
        assert!(state.complete.load(Ordering::Relaxed));
    }

    /// Re-registering a second waker before completion means only the latest
    /// is woken — the AtomicWaker contract the `Executing` arm depends on when
    /// a task is re-polled with a fresh context.
    #[test]
    fn signal_wakes_only_the_latest_registered_waker() {
        let state = StreamCallbackState::new();
        let first = Arc::new(CountingWaker(AtomicUsize::new(0)));
        let second = Arc::new(CountingWaker(AtomicUsize::new(0)));
        state.waker.register(&waker(first.clone()));
        state.waker.register(&waker(second.clone()));
        state.signal();
        assert_eq!(first.0.load(Ordering::SeqCst), 0, "stale waker was woken");
        assert_eq!(second.0.load(Ordering::SeqCst), 1);
    }
}
