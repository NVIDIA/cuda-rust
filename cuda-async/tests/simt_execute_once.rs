/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The two halves of a simt `unzip` execute their input exactly once, no
//! matter how the halves are driven: from two threads at the same time, as
//! two futures joined on one thread, or one from inside the other.
//!
//! Requires a GPU.

use cuda_async::simt::device_context::init_device_contexts;
use cuda_async::simt::device_future::DeviceFuture;
use cuda_async::simt::device_operation::{value, DeviceOperation, ExecutionContext, Unzippable2};
use cuda_async::simt::error::DeviceError;
use std::future::IntoFuture;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::time::Duration;

/// Runs `f` on a fresh thread so the thread-local device map starts clean.
fn on_fresh_thread<F: FnOnce() + Send + 'static>(f: F) {
    std::thread::spawn(f).join().expect("test thread panicked");
}

/// An operation that bumps `counter` every time it executes and yields `val`.
fn counted_op<T: Send + 'static>(
    counter: &Arc<AtomicUsize>,
    val: T,
) -> impl DeviceOperation<Output = T> {
    let counter = Arc::clone(counter);
    value(()).and_then(move |()| {
        counter.fetch_add(1, Ordering::SeqCst);
        value(val)
    })
}

/// Like [`counted_op`], but slow enough that a second executor on another
/// thread is guaranteed to arrive while the first is still running.
fn slow_counted_op<T: Send + 'static>(
    counter: &Arc<AtomicUsize>,
    val: T,
) -> impl DeviceOperation<Output = T> {
    let counter = Arc::clone(counter);
    value(()).and_then(move |()| {
        counter.fetch_add(1, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(150));
        value(val)
    })
}

/// An operation that fails at execute time.
struct FailingOp;

impl DeviceOperation for FailingOp {
    type Output = (u64, u64);
    unsafe fn execute(self, _context: &ExecutionContext) -> Result<(u64, u64), DeviceError> {
        Err(DeviceError::Internal("failing op".into()))
    }
}

impl IntoFuture for FailingOp {
    type Output = Result<(u64, u64), DeviceError>;
    type IntoFuture = DeviceFuture<(u64, u64), FailingOp>;
    fn into_future(self) -> Self::IntoFuture {
        DeviceFuture::failed(DeviceError::Internal("not used".into()))
    }
}

/// The two halves are executed at the same time on two threads. Each thread
/// initializes its own device contexts and therefore its own stream pool, so
/// the half that arrives second consumes the value on a different stream and
/// exercises the wait on the producer's completion event. The input runs
/// once and each half receives its side.
#[test]
fn unzip_concurrent_halves_run_input_once() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let counter = Arc::new(AtomicUsize::new(0));
        let (left, right) = slow_counted_op(&counter, (1_u64, 2_u64)).unzip();
        let barrier = Arc::new(Barrier::new(2));

        let left_thread = {
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                init_device_contexts(0, 1).expect("per-thread init failed");
                barrier.wait();
                left.sync().expect("left failed")
            })
        };
        let right_thread = {
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                init_device_contexts(0, 1).expect("per-thread init failed");
                barrier.wait();
                right.sync().expect("right failed")
            })
        };

        assert_eq!(left_thread.join().expect("left panicked"), 1);
        assert_eq!(right_thread.join().expect("right panicked"), 2);
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "the input must run exactly once"
        );
    });
}

/// Both halves driven concurrently as futures on one thread.
#[test]
fn unzip_join_on_one_thread() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let counter = Arc::new(AtomicUsize::new(0));
        let (left, right) = counted_op(&counter, (3_u64, 4_u64)).unzip();

        let (a, b) = futures::executor::block_on(async {
            futures::join!(left.into_future(), right.into_future())
        });
        assert_eq!(a.expect("left failed"), 3);
        assert_eq!(b.expect("right failed"), 4);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    });
}

/// A failing input reports its own error to both halves; the second half
/// does not see an "already taken" error and the input is not retried.
#[test]
fn unzip_failure_is_reported_to_both_halves() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let (left, right) = FailingOp.unzip();
        for result in [left.sync(), right.sync()] {
            assert!(
                matches!(&result, Err(DeviceError::Internal(m)) if m == "failing op"),
                "expected the input's own error, got {result:?}"
            );
        }
    });
}

/// An input that executes the other half from inside its own execution must
/// get an error, not deadlock on the shared state.
#[test]
fn unzip_reentrant_execution_is_an_error_not_a_deadlock() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        // The right half is handed to the input through a type-erased slot:
        // naming its type inside the closure that produces it would be
        // circular.
        type RunRight = Box<dyn FnOnce() -> Result<u64, DeviceError> + Send>;
        let slot: Arc<Mutex<Option<RunRight>>> = Arc::new(Mutex::new(None));
        let slot_in_op = Arc::clone(&slot);
        let (left, right) = value(())
            .and_then(move |()| {
                let run_right = slot_in_op
                    .lock()
                    .unwrap()
                    .take()
                    .expect("slot is filled before execution");
                let nested = run_right();
                assert!(
                    matches!(&nested, Err(DeviceError::Internal(m)) if m.contains("re-entered")),
                    "re-entrant execution must be reported, got {nested:?}"
                );
                value((1_u64, 2_u64))
            })
            .unzip();
        *slot.lock().unwrap() = Some(Box::new(move || right.sync()));

        assert_eq!(left.sync().expect("outer execution failed"), 1);
    });
}
