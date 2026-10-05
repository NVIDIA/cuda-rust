/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Future type that bridges CUDA stream callbacks with Rust's async executor.
//!
//! [`DeviceFuture`] wraps a [`DeviceOperation`] and drives it through a
//! three-state machine:
//!
//! ```text
//!   Idle ──poll()──> Executing ──callback fires──> Complete
//!                       │                              │
//!                  (enqueue work                  (return result)
//!                   + host callback)
//! ```
//!
//! On the first poll the operation is executed on its assigned stream and a
//! host callback is registered via `cuLaunchHostFunc`. When the GPU reaches
//! the callback, it wakes the future through an [`AtomicWaker`], avoiding
//! busy-waits.
//!
//! [`DeviceOperation`]: crate::simt::device_operation::DeviceOperation
//! [`AtomicWaker`]: futures::task::AtomicWaker

use crate::simt::device_operation::{DeviceOperation, ExecutionContext};
use crate::simt::error::DeviceError;
use crate::simt::reclaim;
use futures::task::AtomicWaker;
use std::future::Future;
use std::mem;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

/// Lifecycle state of a [`DeviceFuture`].
#[derive(Debug, Default, Eq, PartialEq, Copy, Clone)]
pub enum DeviceFutureState {
    /// The future was constructed in a failed state (e.g. scheduling error).
    Failed,
    /// Initial state: the operation has not yet been submitted to the GPU.
    #[default]
    Idle,
    /// The operation has been submitted; waiting for the stream callback.
    Executing,
    /// The result has been produced. Polling again will panic.
    Complete,
}

/// Shared state between a [`DeviceFuture`] and its `cuLaunchHostFunc` callback.
///
/// The callback sets `complete` and wakes the stored waker, allowing the
/// executor to re-poll the future without busy-waiting.
#[derive(Debug, Default)]
pub struct StreamCallbackState {
    /// Waker registered by the executor during [`Future::poll`].
    pub(crate) waker: AtomicWaker,
    /// Set to `true` by the host callback when the GPU reaches the callback
    /// point in the stream.
    pub(crate) complete: AtomicBool,
}

impl StreamCallbackState {
    /// Creates a new callback state with no registered waker and
    /// `complete = false`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Marks the stream callback as complete and wakes the associated future.
    ///
    /// Called from the `cuLaunchHostFunc` host-side callback.
    pub fn signal(&self) {
        self.complete.store(true, Ordering::Relaxed);
        self.waker.wake();
    }
}

/// A [`Future`] that executes a [`DeviceOperation`] on a CUDA stream and
/// resolves when the GPU signals completion via a host callback.
///
/// Constructed by [`SchedulingPolicy::schedule`] or by the [`IntoFuture`] impl
/// on any `DeviceOperation`.
///
/// # Cancellation
///
/// Dropping the future never cancels submitted GPU work; kernels always run
/// to completion. Dropping an in-flight future records a CUDA event on the
/// assigned stream and parks the stored result in the [`reclaim`] limbo,
/// deferring host-side reclamation until the device timeline passes that
/// event. The drop itself never blocks on GPU progress.
///
/// [`reclaim`]: crate::simt::reclaim
///
/// [`SchedulingPolicy::schedule`]: crate::simt::scheduling_policies::SchedulingPolicy::schedule
/// [`IntoFuture`]: std::future::IntoFuture
#[derive(Debug)]
pub struct DeviceFuture<T: Send + 'static, DO: DeviceOperation<Output = T>> {
    /// The operation to execute. Consumed on first poll.
    pub(crate) device_operation: Option<DO>,
    /// Stream and context for execution. Set by the scheduling policy.
    pub(crate) execution_context: Option<ExecutionContext>,
    /// Holds the result between execution and final poll resolution.
    pub(crate) result: Option<T>,
    /// Holds an error when the future is in the `Failed` state.
    pub(crate) error: Option<DeviceError>,
    /// Current lifecycle state.
    pub(crate) state: DeviceFutureState,
    /// Shared state with the `cuLaunchHostFunc` callback.
    pub(crate) callback_state: Option<Arc<StreamCallbackState>>,
}

impl<T: Send + 'static, DO: DeviceOperation<Output = T>> DeviceFuture<T, DO> {
    /// Creates an idle future with no operation or context attached.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a future that will immediately yield `Err(error)` on first poll.
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

    /// Registers a `cuLaunchHostFunc` callback that will signal `waker_state`
    /// when the GPU reaches this point in the stream.
    ///
    /// # Safety
    ///
    /// The execution context must hold a valid, non-destroyed CUDA stream.
    unsafe fn register_callback(
        &self,
        waker_state: Arc<StreamCallbackState>,
    ) -> Result<(), DeviceError> {
        let ctx = self.execution_context.as_ref().ok_or_else(|| {
            DeviceError::Internal("Cannot execute future without an execution context.".to_string())
        })?;
        ctx.get_cuda_stream().launch_host_function(move || {
            waker_state.signal();
        })?;
        Ok(())
    }

    /// Takes the stored operation, executes it on the bound stream, and stashes
    /// the result. Called exactly once during the `Idle -> Executing` transition.
    fn execute(&mut self) -> Result<(), DeviceError> {
        let ctx = self.execution_context.as_ref().ok_or_else(|| {
            DeviceError::Internal("Cannot execute future without an execution context.".to_string())
        })?;
        let operation = self
            .device_operation
            .take()
            .ok_or_else(|| DeviceError::Internal("No operation has been set.".to_string()))?;
        let out = unsafe { operation.execute(ctx) }?;
        self.result = Some(out);
        Ok(())
    }

    /// Returns `true` when GPU work was submitted but the stored result has
    /// not been handed to the caller.
    ///
    /// `result` is only populated by a successful [`execute`], so a stored
    /// result implies submitted GPU work. The state check excludes futures
    /// that never submitted anything (`Idle`, `Failed`); a `Complete` future
    /// can still hold a result when callback registration failed after a
    /// successful launch.
    ///
    /// [`execute`]: Self::execute
    fn has_undelivered_submission(&self) -> bool {
        matches!(
            self.state,
            DeviceFutureState::Executing | DeviceFutureState::Complete
        ) && self.result.is_some()
    }

    /// Hands a submitted-but-undelivered result to deferred reclamation.
    ///
    /// Records a completion event on the assigned stream and parks the
    /// result in the [`reclaim`] limbo; the result is dropped by a later
    /// sweep once the device timeline passes the event. Never blocks in
    /// this path. Only when the event cannot be recorded does it fall back
    /// to [`cleanup_executing_result_with`], which synchronizes the stream
    /// and, if even that fails, leaks the result loudly.
    ///
    /// [`cleanup_executing_result_with`]: Self::cleanup_executing_result_with
    fn reclaim_in_flight_result(&mut self) {
        if !self.has_undelivered_submission() {
            return;
        }
        let stream = self
            .execution_context
            .as_ref()
            .map(|ctx| Arc::clone(ctx.get_cuda_stream()));
        if let Some(stream) = &stream {
            if let Ok(event) = stream.record_event(None) {
                let result = self.result.take().expect("Checked above.");
                reclaim::park(event, Box::new(result));
                return;
            }
        }
        // No execution context, or the driver rejected the event record.
        // Fall back to the blocking path before resorting to a loud leak.
        self.cleanup_executing_result_with(move || {
            let stream = stream.ok_or_else(|| {
                DeviceError::Internal(
                    "Cannot clean up an in-flight future without an execution context.".to_string(),
                )
            })?;
            stream.synchronize().map_err(DeviceError::Driver)
        });
    }

    /// Blocking cleanup fallback: runs `synchronize` and drops the stored
    /// result on success; on failure the result is leaked loudly, because
    /// dropping resources the device may still use is worse than leaking.
    ///
    /// Deferred (non-blocking) reclamation in
    /// [`reclaim_in_flight_result`](Self::reclaim_in_flight_result) is
    /// always preferred; this path only runs when no completion event could
    /// be recorded.
    fn cleanup_executing_result_with<F>(&mut self, synchronize: F)
    where
        F: FnOnce() -> Result<(), DeviceError>,
    {
        if !self.has_undelivered_submission() {
            return;
        }

        let Some(result) = self.result.take() else {
            return;
        };

        if let Err(error) = synchronize() {
            crate::leak::report_leak(format_args!(
                "cuda-async: leaking in-flight future result after cleanup failure: {error}"
            ));
            // If cleanup cannot prove the stream is idle, leaking the owned
            // result is safer than dropping buffers that device work may still
            // be using.
            mem::forget(result);
            return;
        }

        drop(result);
    }
}

impl<T: Send + 'static, DO: DeviceOperation<Output = T>> Default for DeviceFuture<T, DO> {
    fn default() -> Self {
        Self {
            device_operation: Default::default(),
            execution_context: Default::default(),
            result: Default::default(),
            error: Default::default(),
            state: Default::default(),
            callback_state: Default::default(),
        }
    }
}

/// `DeviceFuture` does not contain self-referential pointers, so it is safe
/// to move.
impl<T: Send + 'static, DO: DeviceOperation<Output = T>> Unpin for DeviceFuture<T, DO> {}

impl<T: Send + 'static, DO: DeviceOperation<Output = T>> Drop for DeviceFuture<T, DO> {
    fn drop(&mut self) {
        // Reclaim earlier cancellations whose GPU work has since finished,
        // then defer this future's own in-flight result (if any).
        reclaim::sweep();
        self.reclaim_in_flight_result();
    }
}

/// State-machine implementation of [`Future`] for CUDA device work.
///
/// | State       | Action on poll                                          |
/// |-------------|---------------------------------------------------------|
/// | `Failed`    | Immediately returns `Err`.                              |
/// | `Idle`      | Executes the operation, registers callback, -> Pending. |
/// | `Executing` | Checks callback flag; returns result if done.           |
/// | `Complete`  | Panics -- must not poll after completion.               |
impl<T: Send + 'static, DO: DeviceOperation<Output = T>> Future for DeviceFuture<T, DO> {
    type Output = Result<T, DeviceError>;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Opportunistically reclaim cancelled results whose GPU work has
        // completed. Cheap when nothing is parked (one atomic load).
        reclaim::sweep();

        if self.state == DeviceFutureState::Failed {
            self.state = DeviceFutureState::Complete;
            let error = self
                .error
                .take()
                .expect("Failed state must carry an error.");
            return Poll::Ready(Err(error));
        }

        if self.callback_state.is_none() {
            self.callback_state = Some(Arc::new(StreamCallbackState::new()));
        }
        let waker_state = self
            .callback_state
            .as_ref()
            .map(Arc::clone)
            .expect("Impossible.");

        match self.state {
            DeviceFutureState::Idle => {
                waker_state.waker.register(cx.waker());
                if let Err(e) = self.execute() {
                    self.state = DeviceFutureState::Complete;
                    return Poll::Ready(Err(e));
                }
                if let Err(e) = unsafe { self.register_callback(Arc::clone(&waker_state)) } {
                    self.state = DeviceFutureState::Complete;
                    // The launch already succeeded, so GPU work is in flight
                    // and `result` holds the owned resources. Route them
                    // through deferred reclamation instead of leaving them
                    // for an immediate drop while the kernel may still run
                    // (issue #99, callback-registration-failure path).
                    self.reclaim_in_flight_result();
                    return Poll::Ready(Err(e));
                }
                self.state = DeviceFutureState::Executing;
                Poll::Pending
            }
            DeviceFutureState::Executing => {
                if waker_state.complete.load(Ordering::Relaxed) {
                    self.state = DeviceFutureState::Complete;
                    return Poll::Ready(Ok(self.result.take().expect("Expected result.")));
                }
                waker_state.waker.register(cx.waker());
                if waker_state.complete.load(Ordering::Relaxed) {
                    self.state = DeviceFutureState::Complete;
                    Poll::Ready(Ok(self.result.take().expect("Expected result.")))
                } else {
                    Poll::Pending
                }
            }
            DeviceFutureState::Complete => panic!("Poll called after completion."),
            DeviceFutureState::Failed => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simt::device_operation::Value;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
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

    #[test]
    fn cleanup_executing_result_synchronizes_before_drop() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let tracker = DropTracker {
            events: Arc::clone(&events),
        };
        let mut future: DeviceFuture<DropTracker, Value<DropTracker>> = DeviceFuture {
            device_operation: None,
            execution_context: None,
            result: Some(tracker),
            error: None,
            state: DeviceFutureState::Executing,
            callback_state: None,
        };

        future.cleanup_executing_result_with(|| {
            events.lock().unwrap().push("sync");
            Ok(())
        });

        assert_eq!(events.lock().unwrap().as_slice(), ["sync", "drop"]);
        assert!(future.result.is_none());
    }

    #[test]
    fn cleanup_executing_result_leaks_when_synchronize_fails() {
        let drops = Arc::new(AtomicUsize::new(0));

        struct CountDrop(Arc<AtomicUsize>);

        impl Drop for CountDrop {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        let mut future: DeviceFuture<CountDrop, Value<CountDrop>> = DeviceFuture {
            device_operation: None,
            execution_context: None,
            result: Some(CountDrop(Arc::clone(&drops))),
            error: None,
            state: DeviceFutureState::Executing,
            callback_state: None,
        };

        let mut capture = crate::leak::capture::start();
        future.cleanup_executing_result_with(|| Err(DeviceError::Internal("boom".to_string())));
        let reports = capture.take();

        assert_eq!(drops.load(Ordering::Relaxed), 0);
        assert!(future.result.is_none());
        assert_eq!(reports.len(), 1, "the leak must be reported: {reports:?}");
    }

    /// The shape left behind by issue #99's second path: `execute()`
    /// succeeded (GPU work submitted, `result` populated) but
    /// `register_callback()` failed, so poll flipped the state to
    /// `Complete` with the result still stored. Cleanup must treat that
    /// exactly like a cancelled `Executing` future.
    #[test]
    fn cleanup_covers_complete_future_left_by_callback_registration_failure() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let tracker = DropTracker {
            events: Arc::clone(&events),
        };
        let mut future: DeviceFuture<DropTracker, Value<DropTracker>> = DeviceFuture {
            device_operation: None,
            execution_context: None,
            result: Some(tracker),
            error: None,
            state: DeviceFutureState::Complete,
            callback_state: None,
        };

        assert!(future.has_undelivered_submission());
        future.cleanup_executing_result_with(|| {
            events.lock().unwrap().push("sync");
            Ok(())
        });

        assert_eq!(events.lock().unwrap().as_slice(), ["sync", "drop"]);
        assert!(future.result.is_none());
        assert!(!future.has_undelivered_submission());
    }

    /// A `Complete` future whose result was already delivered to the caller
    /// has nothing to reclaim.
    #[test]
    fn cleanup_is_noop_after_result_delivery() {
        let mut future: DeviceFuture<u32, Value<u32>> = DeviceFuture {
            device_operation: None,
            execution_context: None,
            result: None,
            error: None,
            state: DeviceFutureState::Complete,
            callback_state: None,
        };

        assert!(!future.has_undelivered_submission());
        future.cleanup_executing_result_with(|| {
            panic!("delivered futures must not synchronize during cleanup")
        });
        future.reclaim_in_flight_result();
    }

    #[test]
    fn cleanup_is_noop_for_idle_future() {
        let drops = Arc::new(AtomicUsize::new(0));

        struct CountDrop(Arc<AtomicUsize>);

        impl Drop for CountDrop {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        let mut future: DeviceFuture<CountDrop, Value<CountDrop>> = DeviceFuture {
            device_operation: None,
            execution_context: None,
            result: Some(CountDrop(Arc::clone(&drops))),
            error: None,
            state: DeviceFutureState::Idle,
            callback_state: None,
        };

        future.cleanup_executing_result_with(|| {
            panic!("idle futures should not synchronize during cleanup")
        });

        assert_eq!(drops.load(Ordering::Relaxed), 0);
        assert!(future.result.is_some());

        drop(future);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
    }
}
