/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tests verifying that `unzip`, `zip`, and `shared` execute ancestor
//! operations exactly once, regardless of how many downstream branches
//! consume the results.

use cuda_async::device_context::init_device_contexts;
use cuda_async::device_future::DeviceFuture;
use cuda_async::device_operation::{
    value, BoxedDeviceOp, DeviceOp, ExecutionContext, SharedDeviceOp, Unzippable2, Unzippable3,
    Zippable,
};
use cuda_async::error::DeviceError;
use cuda_async::zip;
use std::future::IntoFuture;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::time::Duration;

/// Run `f` on a fresh thread so thread-local `DEVICE_CONTEXTS` starts clean.
fn on_fresh_thread<F: FnOnce() + Send + 'static>(f: F) {
    std::thread::spawn(f).join().expect("test thread panicked");
}

/// Helper: create a `BoxedDeviceOp` that increments `counter` on each execution.
fn counted_op<T: Send + 'static>(counter: &Arc<AtomicUsize>, val: T) -> BoxedDeviceOp<T> {
    let c = counter.clone();
    value(())
        .then(move |()| {
            c.fetch_add(1, Ordering::SeqCst);
            value(val)
        })
        .boxed()
}

/// Like [`counted_op`], but the execution takes a while, so a second executor
/// on another thread is guaranteed to arrive while the first is still running.
fn slow_counted_op<T: Send + 'static>(counter: &Arc<AtomicUsize>, val: T) -> BoxedDeviceOp<T> {
    let c = counter.clone();
    value(())
        .then(move |()| {
            c.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(150));
            value(val)
        })
        .boxed()
}

/// An op that fails at execute time.
struct FailingOp;

impl DeviceOp for FailingOp {
    type Output = u64;
    unsafe fn execute(self, _context: &ExecutionContext) -> Result<u64, DeviceError> {
        Err(DeviceError::Internal("failing op".into()))
    }
}

impl IntoFuture for FailingOp {
    type Output = Result<u64, DeviceError>;
    type IntoFuture = DeviceFuture<u64, FailingOp>;
    fn into_future(self) -> Self::IntoFuture {
        DeviceFuture::failed(DeviceError::Internal("not used".into()))
    }
}

// ---------------------------------------------------------------------------
// .shared() — cloneable, execute-once operations
// ---------------------------------------------------------------------------

#[test]
fn shared_executes_ancestor_exactly_once() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let counter = Arc::new(AtomicUsize::new(0));
        let op = counted_op(&counter, 42u64);

        let shared = op.shared();
        let a = shared.clone().sync().expect("first failed");
        let b = shared.sync().expect("second failed");

        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert_eq!(*a, 42);
        assert_eq!(*b, 42);
        assert!(Arc::ptr_eq(&a, &b));
    });
}

#[test]
fn shared_n_way_clone() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let counter = Arc::new(AtomicUsize::new(0));
        let shared = counted_op(&counter, 99u64).shared();

        let results: Vec<Arc<u64>> = (0..5)
            .map(|_| shared.clone().sync().expect("clone failed"))
            .collect();

        assert_eq!(counter.load(Ordering::SeqCst), 1);
        for r in &results {
            assert_eq!(**r, 99);
            assert!(Arc::ptr_eq(r, &results[0]));
        }
    });
}

#[test]
fn shared_into_zip() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let counter = Arc::new(AtomicUsize::new(0));
        let shared = counted_op(&counter, 42u64).shared();

        let (a, b) = zip!(shared.clone(), shared).sync().expect("sync failed");

        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert_eq!(*a, 42);
        assert_eq!(*b, 42);
        assert!(Arc::ptr_eq(&a, &b));
    });
}

#[test]
fn shared_pre_computed() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let val = Arc::new(7u64);
        let shared: SharedDeviceOp<u64> = cuda_async::device_operation::shared(val.clone());

        let a = shared.clone().sync().expect("first failed");
        let b = shared.sync().expect("second failed");

        assert_eq!(*a, 7);
        assert!(Arc::ptr_eq(&a, &b));
        assert!(Arc::ptr_eq(&a, &val));
    });
}

/// Two threads execute clones at the same time. Each thread has its own
/// device context and therefore its own streams, so the second consumer also
/// exercises the cross-stream wait on the producer's completion event. The
/// op runs exactly once and both threads get the same `Arc`.
#[test]
fn shared_concurrent_executors_run_once_and_share_the_value() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let counter = Arc::new(AtomicUsize::new(0));
        let shared = slow_counted_op(&counter, 7u64).shared();
        let barrier = Arc::new(Barrier::new(2));

        let workers: Vec<_> = (0..2)
            .map(|_| {
                let shared = shared.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    init_device_contexts(0, 1).expect("per-thread init failed");
                    barrier.wait();
                    shared.sync().expect("concurrent execute failed")
                })
            })
            .collect();
        let results: Vec<Arc<u64>> = workers
            .into_iter()
            .map(|w| w.join().expect("worker panicked"))
            .collect();

        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "op must run exactly once"
        );
        assert_eq!(*results[0], 7);
        assert!(Arc::ptr_eq(&results[0], &results[1]));
    });
}

/// Two clones driven concurrently as futures on one thread.
#[test]
fn shared_join_on_one_thread() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let counter = Arc::new(AtomicUsize::new(0));
        let shared = counted_op(&counter, 11u64).shared();

        let (a, b) = futures::executor::block_on(async {
            futures::join!(shared.clone().into_future(), shared.into_future())
        });
        let (a, b) = (a.expect("first failed"), b.expect("second failed"));

        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert_eq!(*a, 11);
        assert!(Arc::ptr_eq(&a, &b));
    });
}

/// A failed operation is not silently retried and does not turn into an
/// "already taken" error for the second clone: both see the original error.
#[test]
fn shared_failure_is_reported_to_every_clone() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let shared = FailingOp.shared();
        let first = shared.clone().sync();
        let second = shared.sync();
        for result in [first, second] {
            assert!(
                matches!(&result, Err(DeviceError::Internal(m)) if m == "failing op"),
                "expected the op's own error, got {result:?}"
            );
        }
    });
}

/// A shared op that (through `then_unchecked`) executes itself from inside
/// its own execution must get an error, not deadlock on the mutex.
#[test]
fn shared_reentrant_execution_is_an_error_not_a_deadlock() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let slot: Arc<Mutex<Option<SharedDeviceOp<u64>>>> = Arc::new(Mutex::new(None));
        let slot_in_closure = Arc::clone(&slot);
        let shared = unsafe {
            value(()).then_unchecked(move |()| {
                let me = slot_in_closure
                    .lock()
                    .unwrap()
                    .clone()
                    .expect("slot is filled before execution");
                let nested = me.sync();
                assert!(
                    matches!(&nested, Err(DeviceError::Internal(m)) if m.contains("re-entered")),
                    "re-entrant execution must be reported, got {nested:?}"
                );
                value(1u64)
            })
        }
        .shared();
        *slot.lock().unwrap() = Some(shared.clone());

        assert_eq!(*shared.sync().expect("outer execution failed"), 1);
    });
}

// ---------------------------------------------------------------------------
// unzip (2-tuple)
// ---------------------------------------------------------------------------

/// The two halves are executed at the same time on two threads: the input
/// runs once and each half receives its side.
#[test]
fn unzip2_concurrent_halves_run_input_once() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let counter = Arc::new(AtomicUsize::new(0));
        let (left, right) = slow_counted_op(&counter, (1u64, 2u64)).unzip();
        let barrier = Arc::new(Barrier::new(2));

        let left_thread = {
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                init_device_contexts(0, 1).expect("per-thread init failed");
                barrier.wait();
                left.sync().expect("left failed")
            })
        };
        let right_thread = {
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                init_device_contexts(0, 1).expect("per-thread init failed");
                barrier.wait();
                right.sync().expect("right failed")
            })
        };

        assert_eq!(left_thread.join().expect("left panicked"), 1);
        assert_eq!(right_thread.join().expect("right panicked"), 2);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn unzip2_join_on_one_thread() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let counter = Arc::new(AtomicUsize::new(0));
        let (left, right) = counted_op(&counter, (3u64, 4u64)).unzip();

        let (a, b) = futures::executor::block_on(async {
            futures::join!(left.into_future(), right.into_future())
        });
        assert_eq!(a.expect("left failed"), 3);
        assert_eq!(b.expect("right failed"), 4);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    });
}

/// A failing input reports its error to both halves.
#[test]
fn unzip2_failure_is_reported_to_both_halves() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let (left, right) = FailingOp.then(|v| value((v, v))).unzip();
        for result in [left.sync(), right.sync()] {
            assert!(
                matches!(&result, Err(DeviceError::Internal(m)) if m == "failing op"),
                "expected the input's own error, got {result:?}"
            );
        }
    });
}

#[test]
fn unzip2_executes_ancestor_exactly_once() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let op = value(()).then(move |()| {
            c.fetch_add(1, Ordering::SeqCst);
            value((1u64, 2u64))
        });

        let (left, right) = op.unzip();
        let a = left.sync().expect("left failed");
        let b = right.sync().expect("right failed");

        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert_eq!(a, 1);
        assert_eq!(b, 2);
    });
}

#[test]
fn unzip2_right_before_left() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let op = value(()).then(move |()| {
            c.fetch_add(1, Ordering::SeqCst);
            value((1u64, 2u64))
        });

        let (left, right) = op.unzip();
        let b = right.sync().expect("right failed");
        let a = left.sync().expect("left failed");

        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert_eq!(a, 1);
        assert_eq!(b, 2);
    });
}

// ---------------------------------------------------------------------------
// unzip (3-tuple — exercises nested _unzip chain)
// ---------------------------------------------------------------------------

#[test]
fn unzip3_executes_ancestor_exactly_once() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let op = value(()).then(move |()| {
            c.fetch_add(1, Ordering::SeqCst);
            value((1u64, 2u64, 3u64))
        });

        let (a, b, c_op) = op.unzip();
        let a = a.sync().expect("a failed");
        let b = b.sync().expect("b failed");
        let c_val = c_op.sync().expect("c failed");

        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert_eq!(a, 1);
        assert_eq!(b, 2);
        assert_eq!(c_val, 3);
    });
}

#[test]
fn unzip3_reversed_execution_order() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let op = value(()).then(move |()| {
            c.fetch_add(1, Ordering::SeqCst);
            value((1u64, 2u64, 3u64))
        });

        let (a, b, c_op) = op.unzip();
        let c_val = c_op.sync().expect("c failed");
        let b = b.sync().expect("b failed");
        let a = a.sync().expect("a failed");

        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert_eq!(a, 1);
        assert_eq!(b, 2);
        assert_eq!(c_val, 3);
    });
}

// ---------------------------------------------------------------------------
// zip then unzip (round-trip)
// ---------------------------------------------------------------------------

#[test]
fn zip2_then_unzip2_executes_each_input_once() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let counter_a = Arc::new(AtomicUsize::new(0));
        let counter_b = Arc::new(AtomicUsize::new(0));

        let op_a = counted_op(&counter_a, 10u64);
        let op_b = counted_op(&counter_b, 20u64);

        let (a, b) = zip!(op_a, op_b).unzip();
        let a = a.sync().expect("a failed");
        let b = b.sync().expect("b failed");

        assert_eq!(counter_a.load(Ordering::SeqCst), 1);
        assert_eq!(counter_b.load(Ordering::SeqCst), 1);
        assert_eq!(a, 10);
        assert_eq!(b, 20);
    });
}

#[test]
fn zip3_then_unzip3_executes_each_input_once() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let counter_a = Arc::new(AtomicUsize::new(0));
        let counter_b = Arc::new(AtomicUsize::new(0));
        let counter_c = Arc::new(AtomicUsize::new(0));

        let op_a = counted_op(&counter_a, 1u64);
        let op_b = counted_op(&counter_b, 2u64);
        let op_c = counted_op(&counter_c, 3u64);

        let (a, b, c_op) = zip!(op_a, op_b, op_c).unzip();
        let a = a.sync().expect("a failed");
        let b = b.sync().expect("b failed");
        let c_val = c_op.sync().expect("c failed");

        assert_eq!(counter_a.load(Ordering::SeqCst), 1);
        assert_eq!(counter_b.load(Ordering::SeqCst), 1);
        assert_eq!(counter_c.load(Ordering::SeqCst), 1);
        assert_eq!(a, 1);
        assert_eq!(b, 2);
        assert_eq!(c_val, 3);
    });
}

// ---------------------------------------------------------------------------
// Fan-out then fan-in (unzip → re-zip)
// ---------------------------------------------------------------------------

#[test]
fn unzip_then_rezip_executes_ancestor_once() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let op = value(()).then(move |()| {
            c.fetch_add(1, Ordering::SeqCst);
            value((1u64, 2u64))
        });

        let (left, right) = op.unzip();
        let rezipped = zip!(left, right);
        let (a, b) = rezipped.sync().expect("sync failed");

        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert_eq!(a, 1);
        assert_eq!(b, 2);
    });
}

// ---------------------------------------------------------------------------
// Diamond: zip two inputs, unzip, transform each branch, re-zip
// ---------------------------------------------------------------------------

#[test]
fn diamond_graph_executes_each_leaf_once() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let counter_a = Arc::new(AtomicUsize::new(0));
        let counter_b = Arc::new(AtomicUsize::new(0));

        let op_a = counted_op(&counter_a, 10u64);
        let op_b = counted_op(&counter_b, 20u64);

        // zip → unzip (fan-out) → transform each branch → re-zip (fan-in)
        let (a, b) = zip!(op_a, op_b).unzip();
        let a = a.then(|v| value(v + 1));
        let b = b.then(|v| value(v + 2));
        let result = zip!(a, b).sync().expect("sync failed");

        assert_eq!(counter_a.load(Ordering::SeqCst), 1);
        assert_eq!(counter_b.load(Ordering::SeqCst), 1);
        assert_eq!(result, (11, 22));
    });
}
