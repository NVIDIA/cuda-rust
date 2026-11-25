/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Dropping a `DeviceFuture` while its GPU work is in flight must not release
//! independently retained owners before the device has finished with them.
//!
//! The op enqueues a chain of memsets and then records a CUDA event on the
//! same stream, handing the event back inside its output. The output's
//! `Drop` queries that event: if the output were released early, the query
//! would report the work still in flight.
//!
//! The core test keeps a mapped-memory gate closed through the first poll and
//! drop, checks that the retained output is still alive, then explicitly opens
//! the gate. A watchdog releases stuck GPU work on failure; expiration fails
//! the test rather than silently satisfying its ordering checks.

use cuda_async::device_context::{global_policy, init_device_contexts, with_device};
use cuda_async::device_operation::{DeviceOp, ExecutionContext};
use cuda_async::error::DeviceError;
use cuda_core::Event;
use std::future::{Future, IntoFuture};
use std::mem::MaybeUninit;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
use std::time::{Duration, Instant};

fn on_fresh_thread<F: FnOnce() + Send + 'static>(f: F) {
    std::thread::spawn(f).join().expect("test thread panicked");
}

fn noop_waker() -> Waker {
    fn noop(_: *const ()) {}
    fn clone(p: *const ()) -> RawWaker {
        RawWaker::new(p, &VTABLE)
    }
    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);
    unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
}

fn bind_device() {
    with_device(0, |device| device.bind_to_thread())
        .expect("device context")
        .expect("bind_to_thread failed");
}

fn alloc_device(bytes: usize) -> u64 {
    bind_device();
    let mut dptr = MaybeUninit::uninit();
    let code = unsafe { cuda_bindings::cuMemAlloc_v2(dptr.as_mut_ptr(), bytes) };
    assert_eq!(code, 0, "cuMemAlloc failed: {code}");
    unsafe { dptr.assume_init() }
}

#[path = "support/gate.rs"]
mod gate;
use gate::Gate;

/// What the output's `Drop` observed: whether the device had passed the
/// event recorded after the op's work when the output was released.
type DropLog = Arc<Mutex<Vec<bool>>>;

/// The op's output: owns the completion event and reports, on drop, whether
/// the work had completed by then.
struct Tracked {
    event: Event,
    log: DropLog,
}

impl Drop for Tracked {
    fn drop(&mut self) {
        let done = self.event.query().unwrap_or(false);
        self.log.lock().unwrap().push(done);
    }
}

/// `passes` memsets of a `bytes` buffer, then an event recorded after them.
/// With a `gate`, the whole chain waits behind it.
struct SlowOp {
    dptr: u64,
    bytes: usize,
    passes: usize,
    gate: Option<Arc<Gate>>,
    log: DropLog,
    executions: Arc<AtomicUsize>,
}

impl DeviceOp for SlowOp {
    type Output = Arc<Tracked>;
    unsafe fn execute(self, context: &ExecutionContext) -> Result<Arc<Tracked>, DeviceError> {
        self.executions.fetch_add(1, Ordering::SeqCst);
        let stream = context.get_cuda_stream();
        if let Some(gate) = &self.gate {
            context.retain(gate.clone())?;
            gate.arm(stream.cu_stream())?;
        }
        for _ in 0..self.passes {
            let code =
                cuda_bindings::cuMemsetD8Async(self.dptr, 0x5A, self.bytes, stream.cu_stream());
            if code != cuda_bindings::cudaError_enum_CUDA_SUCCESS {
                return Err(DeviceError::Internal(format!(
                    "cuMemsetD8Async failed: {code}"
                )));
            }
        }
        let event = stream.context().new_event()?;
        event.record(stream)?;
        let tracked = Arc::new(Tracked {
            event,
            log: self.log,
        });
        context.retain(tracked.clone())?;
        Ok(tracked)
    }
}

impl IntoFuture for SlowOp {
    type Output = Result<Arc<Tracked>, DeviceError>;
    type IntoFuture = cuda_async::device_future::DeviceFuture<Arc<Tracked>, SlowOp>;
    fn into_future(self) -> Self::IntoFuture {
        let policy = global_policy(0).expect("global policy");
        match self.schedule(&policy) {
            Ok(future) => future,
            Err(error) => cuda_async::device_future::DeviceFuture::failed(error),
        }
    }
}

const BUF: usize = 64 << 20;
/// ~64 MiB x 32 of memset: real device work behind each op's event. The
/// ungated tests lean on it being far past the 20 us inline-spin budget; the
/// core test does not, its gate holds the work back for as long as it needs.
const PASSES: usize = 32;
fn slow_op(dptr: u64, log: &DropLog) -> SlowOp {
    SlowOp {
        dptr,
        bytes: BUF,
        passes: PASSES,
        gate: None,
        log: Arc::clone(log),
        executions: Arc::new(AtomicUsize::new(0)),
    }
}

fn gated_op(dptr: u64, log: &DropLog, gate: &Arc<Gate>) -> SlowOp {
    SlowOp {
        gate: Some(Arc::clone(gate)),
        ..slow_op(dptr, log)
    }
}

fn block_on_with_deadline<F: Future + Unpin>(mut future: F, deadline: Duration) -> F::Output {
    let start = Instant::now();
    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);
    loop {
        match Pin::new(&mut future).poll(&mut cx) {
            Poll::Ready(out) => return out,
            Poll::Pending => {
                assert!(
                    start.elapsed() < deadline,
                    "future did not complete within {deadline:?}"
                );
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }
}

/// The core regression: poll once with the work provably in flight (held
/// behind a closed gate), drop the future, and check that the output was
/// released only after the device passed the event recorded behind the work.
///
/// Drop must return while the gate remains closed and the retained output
/// stays alive. Only then does this test allow GPU completion.
#[test]
fn dropping_in_flight_future_returns_promptly_and_retires_owners() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");
        let dptr = alloc_device(BUF);
        let log: DropLog = Arc::new(Mutex::new(Vec::new()));

        // Bring up the completion path (reactor slab, scanner thread) on an
        // ungated op, so the crate's one-time driver setup does not happen
        // while a gate is closed.
        let scratch: DropLog = Default::default();
        block_on_with_deadline(
            slow_op(dptr, &scratch).into_future(),
            Duration::from_secs(30),
        )
        .expect("warm-up op failed");

        for round in 0..4 {
            let gate = Gate::closed();
            let mut future = gated_op(dptr, &log, &gate).into_future();
            let waker = noop_waker();
            let mut cx = Context::from_waker(&waker);
            let hold = gate.hold();
            assert!(
                Pin::new(&mut future).poll(&mut cx).is_pending(),
                "gated operation must be left in flight"
            );
            drop(future);
            assert_eq!(
                log.lock().unwrap().len(),
                round,
                "drop released a retained owner before GPU completion"
            );
            drop(hold);
            let deadline = Instant::now() + Duration::from_secs(30);
            while log.lock().unwrap().len() != round + 1 {
                assert!(Instant::now() < deadline, "completed owner was not retired");
                std::thread::yield_now();
            }
        }

        let deadline = Instant::now() + Duration::from_secs(5);
        while log.lock().unwrap().len() != 4 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        let log = log.lock().unwrap();
        assert_eq!(log.len(), 4, "every dropped future must release its output");
        assert!(
            log.iter().all(|&done| done),
            "an output was released while its GPU work was still in flight: {log:?}"
        );
    });
}

/// A future dropped before its first poll never executed: nothing was
/// submitted, nothing is waited on, and no output exists.
#[test]
fn dropping_unpolled_future_submits_nothing() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");
        let dptr = alloc_device(BUF);
        let log: DropLog = Arc::new(Mutex::new(Vec::new()));

        let op = slow_op(dptr, &log);
        let executions = op.executions.clone();
        drop(op.into_future());
        assert_eq!(executions.load(Ordering::SeqCst), 0, "unpolled op executed");
        assert!(log.lock().unwrap().is_empty());
    });
}

/// After the result has been delivered, dropping the future is a no-op; the
/// delivered output is the caller's and reports completion when dropped.
#[test]
fn delivered_result_is_owned_by_the_caller() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");
        let dptr = alloc_device(BUF);
        let log: DropLog = Arc::new(Mutex::new(Vec::new()));

        let tracked =
            block_on_with_deadline(slow_op(dptr, &log).into_future(), Duration::from_secs(30))
                .expect("op failed");
        assert!(log.lock().unwrap().is_empty(), "nothing dropped yet");
        drop(tracked);
        assert_eq!(log.lock().unwrap().as_slice(), [true]);
    });
}

/// Cancelling a future does not disturb later work on the same streams.
#[test]
fn later_pipelines_complete_after_cancellations() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");
        let dptr = alloc_device(BUF);
        let log: DropLog = Arc::new(Mutex::new(Vec::new()));

        for _ in 0..8 {
            let mut future = slow_op(dptr, &log).into_future();
            let waker = noop_waker();
            let mut cx = Context::from_waker(&waker);
            let _ = Pin::new(&mut future).poll(&mut cx);
            drop(future);
        }
        for _ in 0..4 {
            block_on_with_deadline(slow_op(dptr, &log).into_future(), Duration::from_secs(30))
                .expect("op after cancellations failed");
        }
        assert!(log.lock().unwrap().iter().all(|&done| done));
    });
}
