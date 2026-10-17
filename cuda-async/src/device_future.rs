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
//! Dropping the future never cancels submitted GPU work; kernels run to
//! completion regardless. What the drop decides is *when the host releases
//! the resources that work still uses*. Those resources live in the
//! execution context's submission (storage leases and other owners the
//! operation retained, see [`ExecutionContext::retain`]); the result itself
//! is a handle by the [`DeviceOp::execute`] contract and is released on the
//! spot. The context is parked with the completion reactor
//! ([`crate::reaper`]) and released on a dedicated thread once a flag write
//! enqueued behind the work lands, so the drop returns immediately and never
//! stalls an executor thread. If the context cannot be parked it is released
//! inline, which waits for the stream; if the stream cannot be waited on (a
//! faulted context, a stream mid-capture) the owners are leaked: releasing
//! memory the device may still write to is the worse failure. The leak is
//! reported on stderr unless the future already resolved with the stream's
//! fault — then the caller has the error and the leak is its documented
//! consequence.

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
    match unsafe { stream.query() } {
        Ok(true) => StreamHealth::Idle,
        Ok(false) => StreamHealth::Busy,
        Err(e) => StreamHealth::Faulted(e),
    }
}

/// A future that executes a [`DeviceOp`] on a CUDA stream and resolves upon completion.
///
/// # Cancellation and the in-flight result
///
/// Dropping a `DeviceFuture` after its first poll — after `execute` has
/// enqueued GPU work — but before it resolved leaves an *undelivered
/// submission*. The operation's output is released immediately: by the
/// [`DeviceOp::execute`] contract it holds no resource the device still
/// uses, those being retained by the execution context's submission. The
/// context is handed to the reaper (see [`crate::reaper`]), which drops it
/// once the stream has passed the abandoned work, releasing the storage
/// leases and other owners. Dropping is therefore non-blocking; only a
/// context the reactor cannot take falls back to waiting inline.
///
/// Outputs may borrow the caller's buffers (`&'a Tensor<T>`,
/// `Partition<&'a mut Tensor<T>>`). That is sound because the borrow is not
/// what keeps the device memory alive: the submission's lease on the
/// storage is, and it is released only after the stream has drained. What
/// the early end of the borrow does allow is a *host* access to the buffer
/// through an unchecked path (a raw device pointer handed to another
/// library) while the device may still be writing; stream-ordered accesses
/// through this crate remain ordered or are rejected by the cross-stream
/// conflict check.
#[derive(Debug)]
pub struct DeviceFuture<T: Send, DO: DeviceOp<Output = T>> {
    pub(crate) device_operation: Option<DO>,
    pub(crate) execution_context: Option<ExecutionContext>,
    pub(crate) result: Option<T>,
    pub(crate) error: Option<DeviceError>,
    pub(crate) state: DeviceFutureState,
    pub(crate) callback_state: Option<Arc<StreamCallbackState>>,
    /// Set when `poll` resolved with the stream's fault: the caller has the
    /// error, so the drop-time leak of the stored result is the documented
    /// consequence of that fault, not news worth reporting again.
    pub(crate) fault_delivered: bool,
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
            fault_delivered: false,
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
            fault_delivered: false,
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

    /// Returns `true` when GPU work was submitted but the stored result has
    /// not been handed to the caller.
    ///
    /// `result` is only populated by a successful [`execute`], so a stored
    /// result implies submitted GPU work. The state check excludes futures
    /// that never submitted anything (`Idle`, `Failed`); a `Complete` future
    /// can still hold a result when completion registration failed after a
    /// successful launch, or when the stream faulted before completion.
    ///
    /// [`execute`]: Self::execute
    fn has_undelivered_submission(&self) -> bool {
        matches!(
            self.state,
            DeviceFutureState::Executing | DeviceFutureState::Complete
        ) && self.result.is_some()
    }

    /// Releases an undelivered submission without waiting for the device.
    ///
    /// The result is a handle (see the type docs) and drops here. The
    /// execution context, whose submission owns everything the device may
    /// still touch, is parked with the reaper and released once the stream
    /// has passed the abandoned work; when it cannot be parked it is dropped
    /// inline, which waits for the stream, or leaks the owners if the stream
    /// cannot be waited on.
    fn release_in_flight_result(&mut self) {
        if !self.has_undelivered_submission() {
            return;
        }
        drop(self.result.take());
        let Some(ctx) = self.execution_context.take() else {
            return;
        };
        if self.fault_delivered {
            // The caller already received this stream's fault from `poll`;
            // the leak on release is its documented consequence, not news.
            ctx.mark_fault_delivered();
        }
        #[cfg(not(loom))]
        crate::reaper::park_or_wait(ctx);
        #[cfg(loom)]
        drop(ctx);
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
            fault_delivered: false,
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
                        match unsafe { ctx.get_cuda_stream().query() } {
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
                        // The result stays stored; `Drop` decides how to
                        // release it (it will leak, quietly, on a dead
                        // context — the caller receives the fault here).
                        self.state = DeviceFutureState::Complete;
                        self.fault_delivered = true;
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
                        let _ = ctx.device().bind_to_thread();
                        probe_stream(ctx.get_cuda_stream())
                    }
                    None => StreamHealth::Busy,
                };
                match health {
                    StreamHealth::Faulted(e) => {
                        // The result stays stored for `Drop` (which leaks it,
                        // quietly — the caller receives the fault here).
                        self.state = DeviceFutureState::Complete;
                        self.fault_delivered = true;
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
    use std::sync::Mutex;

    #[derive(Clone)]
    struct DropTracker {
        events: Arc<Mutex<Vec<&'static str>>>,
    }

    impl Drop for DropTracker {
        fn drop(&mut self) {
            self.events.lock().unwrap().push("drop");
        }
    }

    struct CountDrop(Arc<AtomicUsize>);

    impl Drop for CountDrop {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

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
            fault_delivered: false,
        }
    }

    /// Releasing an `Executing` future drops the result handle on the spot:
    /// nothing the device uses lives in the result.
    #[test]
    fn release_drops_the_result_handle_immediately() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut future = future_in_state(
            DeviceFutureState::Executing,
            Some(CountDrop(Arc::clone(&drops))),
        );

        future.release_in_flight_result();

        assert_eq!(drops.load(Ordering::Relaxed), 1);
        assert!(future.result.is_none());
        assert!(!future.has_undelivered_submission());
    }

    /// The registration-failure shape: `execute` succeeded (work submitted,
    /// result stored) but completion registration failed, so poll flipped
    /// the state to `Complete` with the result still stored. Release must
    /// treat it exactly like a cancelled `Executing` future.
    #[test]
    fn release_covers_complete_future_left_by_registration_failure() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let tracker = DropTracker {
            events: Arc::clone(&events),
        };
        let mut future = future_in_state(DeviceFutureState::Complete, Some(tracker));

        assert!(future.has_undelivered_submission());
        future.release_in_flight_result();

        assert_eq!(events.lock().unwrap().as_slice(), ["drop"]);
        assert!(!future.has_undelivered_submission());
    }

    /// A delivered result leaves nothing to release.
    #[test]
    fn release_is_noop_after_result_delivery() {
        let mut future: DeviceFuture<u32, Value<u32>> =
            future_in_state(DeviceFutureState::Complete, None);

        assert!(!future.has_undelivered_submission());
        future.release_in_flight_result();
        assert!(!future.has_undelivered_submission());
    }

    /// An `Idle` future never submitted work: no wait, and dropping it drops
    /// its contents normally.
    #[test]
    fn release_is_noop_for_idle_future() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut future =
            future_in_state(DeviceFutureState::Idle, Some(CountDrop(Arc::clone(&drops))));

        future.release_in_flight_result();
        assert_eq!(drops.load(Ordering::Relaxed), 0);
        assert!(future.result.is_some());

        drop(future);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
    }

    /// `Drop` on a `Failed` future (a scheduling error) is a plain drop.
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
