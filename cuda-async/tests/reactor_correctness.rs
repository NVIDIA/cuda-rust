/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Correctness tests for the completion paths of `DeviceFuture`, patterned
//! on tokio's `AtomicWaker` and runtime waker-contract tests (spurious wake,
//! waker replacement, cancellation mid-flight).
//!
//! Every test sets `CUDA_ASYNC_SPIN_BUDGET_US=0`, which disables the inline
//! spin fast path so every pipeline goes through the completion-notification
//! path (the flag-write reactor by default; `CUDA_ASYNC_HOST_SYNC=block`/`spin`
//! routes through the host-callback fallback instead). Run the default path,
//! and optionally the fallback:
//!
//! ```text
//! cargo test -p cuda-async --test reactor_correctness
//! CUDA_ASYNC_HOST_SYNC=spin cargo test -p cuda-async --test reactor_correctness
//! ```
//!
//! A single future is always polled from the thread that scheduled it
//! (execution contexts are thread-local). But many such threads submit into
//! one process-wide reactor concurrently, which `concurrent_submitters_share_one_reactor`
//! stresses directly.

use cuda_async::device_context::{global_policy, init_device_contexts};
use cuda_async::device_operation::{DeviceOp, ExecutionContext};
use cuda_async::error::DeviceError;
use futures::task::ArcWake;
use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

fn on_fresh_thread<F: FnOnce() + Send + 'static>(f: F) {
    std::thread::spawn(f).join().expect("test thread panicked");
}

fn force_notification_path() {
    // OnceLock-cached on first poll; every test in this binary wants 0.
    #[allow(unused_unsafe)]
    unsafe {
        std::env::set_var("CUDA_ASYNC_SPIN_BUDGET_US", "0")
    };
}

// ---------------------------------------------------------------------------
// A device op with real, controllable GPU-side duration: `passes` async
// memsets of a `bytes`-sized buffer, so the work is still in flight when the
// first poll happens and the notification path must carry completion.
// ---------------------------------------------------------------------------

struct MemsetOp {
    dptr: u64,
    bytes: usize,
    passes: usize,
    value: u8,
}

impl DeviceOp for MemsetOp {
    type Output = ();
    unsafe fn execute(self, context: &ExecutionContext) -> Result<(), DeviceError> {
        let stream = context.get_cuda_stream().cu_stream();
        for _ in 0..self.passes {
            let code = cuda_bindings::cuMemsetD8Async(self.dptr, self.value, self.bytes, stream);
            if code != cuda_bindings::cudaError_enum_CUDA_SUCCESS {
                return Err(DeviceError::Internal(format!(
                    "cuMemsetD8Async failed: {code}"
                )));
            }
        }
        Ok(())
    }
}

impl IntoFuture for MemsetOp {
    type Output = Result<(), DeviceError>;
    type IntoFuture = cuda_async::device_future::DeviceFuture<(), MemsetOp>;
    fn into_future(self) -> Self::IntoFuture {
        let policy = global_policy(0).expect("global policy");
        match self.schedule(&policy) {
            Ok(future) => future,
            Err(error) => cuda_async::device_future::DeviceFuture::failed(error),
        }
    }
}

fn alloc_device(bytes: usize) -> u64 {
    cuda_async::device_context::with_device(0, |device| device.bind_to_thread())
        .expect("device context")
        .expect("bind_to_thread failed");
    let mut dptr = std::mem::MaybeUninit::uninit();
    let code = unsafe { cuda_bindings::cuMemAlloc_v2(dptr.as_mut_ptr(), bytes) };
    assert_eq!(code, 0, "cuMemAlloc failed: {code}");
    unsafe { dptr.assume_init() }
}

fn read_device(dptr: u64, bytes: usize) -> Vec<u8> {
    let mut host = vec![0u8; bytes];
    let code = unsafe { cuda_bindings::cuMemcpyDtoH_v2(host.as_mut_ptr() as *mut _, dptr, bytes) };
    assert_eq!(code, 0, "cuMemcpyDtoH failed: {code}");
    host
}

/// ~64 MiB x passes of memset: long enough to be in flight at first poll.
fn slow_op(dptr: u64, bytes: usize, value: u8) -> MemsetOp {
    MemsetOp {
        dptr,
        bytes,
        passes: 16,
        value,
    }
}

// ---------------------------------------------------------------------------
// Minimal polling harness: a parker waker plus an unpark-counting wrapper,
// so tests can observe exactly who gets woken (tokio waker-test idiom).
// ---------------------------------------------------------------------------

struct FlagWaker {
    woken: AtomicBool,
    thread: std::thread::Thread,
}

impl ArcWake for FlagWaker {
    fn wake_by_ref(arc_self: &Arc<Self>) {
        arc_self.woken.store(true, Ordering::SeqCst);
        arc_self.thread.unpark();
    }
}

fn flag_waker() -> (Arc<FlagWaker>, std::task::Waker) {
    let state = Arc::new(FlagWaker {
        woken: AtomicBool::new(false),
        thread: std::thread::current(),
    });
    let waker = futures::task::waker(state.clone());
    (state, waker)
}

/// block_on with a deadline; panics if the future does not resolve in time.
fn block_on_with_deadline<F: Future + Unpin>(mut future: F, deadline: Duration) -> F::Output {
    let start = Instant::now();
    let (_state, waker) = flag_waker();
    let mut cx = Context::from_waker(&waker);
    loop {
        match Pin::new(&mut future).poll(&mut cx) {
            Poll::Ready(out) => return out,
            Poll::Pending => {
                assert!(
                    start.elapsed() < deadline,
                    "future did not complete within {deadline:?} (missed wake?)"
                );
                std::thread::park_timeout(Duration::from_millis(1));
            }
        }
    }
}

const DEADLINE: Duration = Duration::from_secs(20);
const BUF: usize = 64 << 20;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Completion carries through the notification path and the awaited data is
/// fully visible to the host afterwards (would catch a flag-write ordered
/// before the data).
#[test]
fn notification_path_completes_and_data_is_visible() {
    on_fresh_thread(|| {
        force_notification_path();
        init_device_contexts(0, 1).expect("init failed (requires GPU)");
        let dptr = alloc_device(BUF);
        for round in 0..8u8 {
            let value = 0xA0 + round;
            block_on_with_deadline(slow_op(dptr, BUF, value).into_future(), DEADLINE)
                .expect("op failed");
            let host = read_device(dptr, 4096);
            assert!(
                host.iter().all(|&b| b == value),
                "round {round}: data not visible after await"
            );
        }
    });
}

/// Spurious wake: waking the task without completion must re-poll without
/// panicking, staying Pending while the work is in flight, and the future
/// must still complete afterwards. A re-poll probes the stream, so once the
/// memsets have drained it may return Ready; a completed future must not be
/// polled again (the `Future` contract), so the loop stops there.
#[test]
fn spurious_wake_repolls_to_pending_then_completes() {
    on_fresh_thread(|| {
        force_notification_path();
        init_device_contexts(0, 1).expect("init failed (requires GPU)");
        let dptr = alloc_device(BUF);
        let mut future = slow_op(dptr, BUF, 0x11).into_future();
        let (_state, waker) = flag_waker();
        let mut cx = Context::from_waker(&waker);
        let first = Pin::new(&mut future).poll(&mut cx);
        assert!(first.is_pending(), "slow op resolved on first poll");
        // Spurious wake + immediate re-poll, several times.
        let mut completed = None;
        for _ in 0..4 {
            waker.wake_by_ref();
            if let Poll::Ready(result) = Pin::new(&mut future).poll(&mut cx) {
                completed = Some(result);
                break;
            }
        }
        match completed {
            Some(result) => result.expect("op failed on a re-poll after the work drained"),
            None => {
                block_on_with_deadline(future, DEADLINE).expect("op failed after spurious wakes")
            }
        }
    });
}

/// Waker replacement: when a task is re-polled with a new waker before
/// completion, the completion must wake the NEW waker (last-registered
/// wins — the AtomicWaker contract tokio's tests pin down).
#[test]
fn completion_wakes_the_latest_registered_waker() {
    on_fresh_thread(|| {
        force_notification_path();
        init_device_contexts(0, 1).expect("init failed (requires GPU)");
        let dptr = alloc_device(BUF);
        let mut future = slow_op(dptr, BUF, 0x22).into_future();

        let (state_a, waker_a) = flag_waker();
        let mut cx_a = Context::from_waker(&waker_a);
        assert!(Pin::new(&mut future).poll(&mut cx_a).is_pending());

        let (state_b, waker_b) = flag_waker();
        let mut cx_b = Context::from_waker(&waker_b);
        if Pin::new(&mut future).poll(&mut cx_b).is_pending() {
            // Wait for the completion signal to land.
            let start = Instant::now();
            while !state_b.woken.load(Ordering::SeqCst) {
                assert!(start.elapsed() < DEADLINE, "waker B was never woken");
                std::thread::park_timeout(Duration::from_millis(1));
            }
            assert!(
                !state_a.woken.load(Ordering::SeqCst),
                "stale waker A was woken after replacement"
            );
            match Pin::new(&mut future).poll(&mut cx_b) {
                Poll::Ready(result) => result.expect("op failed"),
                Poll::Pending => panic!("woken but still pending"),
            }
        }
    });
}

/// Cancellation: dropping a future mid-flight (after registration) must not
/// panic when the GPU later completes, must not corrupt later pipelines, and
/// must recycle completion slots (exercised by outnumbering the pool).
#[test]
fn drop_mid_flight_recycles_and_later_pipelines_work() {
    on_fresh_thread(|| {
        force_notification_path();
        init_device_contexts(0, 1).expect("init failed (requires GPU)");
        let dptr = alloc_device(BUF);

        for _ in 0..32 {
            let mut future = slow_op(dptr, BUF, 0x33).into_future();
            let (_state, waker) = flag_waker();
            let mut cx = Context::from_waker(&waker);
            // Register with the completion path, then cancel.
            assert!(Pin::new(&mut future).poll(&mut cx).is_pending());
            drop(future);
        }

        // Give the dropped completions time to land and slots to recycle.
        std::thread::sleep(Duration::from_millis(200));

        // Later pipelines complete with correct data.
        for round in 0..8u8 {
            let value = 0x40 + round;
            block_on_with_deadline(slow_op(dptr, BUF, value).into_future(), DEADLINE)
                .expect("op after cancellations failed");
            let host = read_device(dptr, 4096);
            assert!(host.iter().all(|&b| b == value));
        }
    });
}

/// Idle/active ping-pong: sequential awaits with idle gaps cycle the
/// scanner's park/unpark edge (targets the missed-unpark window; the park
/// token must make an unpark-before-park return immediately).
#[test]
fn sequential_pingpong_park_unpark() {
    on_fresh_thread(|| {
        force_notification_path();
        init_device_contexts(0, 1).expect("init failed (requires GPU)");
        let small = 8 << 20;
        let dptr = alloc_device(small);
        for i in 0..200u32 {
            let value = (i % 251) as u8;
            let op = MemsetOp {
                dptr,
                bytes: small,
                passes: 2,
                value,
            };
            block_on_with_deadline(op.into_future(), DEADLINE).expect("ping-pong op failed");
            if i % 10 == 0 {
                // Force a genuinely idle window so the scanner parks.
                std::thread::sleep(Duration::from_millis(2));
            }
        }
        let host = read_device(dptr, 4096);
        assert!(host.iter().all(|&b| b == 199_u8));
    });
}

/// Concurrent submitters share one reactor: many control threads, each with
/// its own thread-local device context and buffer, drive awaited ops whose
/// completions all register into the single process-wide reactor at once.
/// Stresses concurrent `register()` (free-list pop + bitmap publish) against
/// the scanner's concurrent harvest — the multi-thread path the single-thread
/// tests never hit. Patterned on tokio's `loom_mpsc`-style real-runtime
/// stress (many producers, one shared consumer), run as an actual GPU load
/// rather than a model.
#[test]
fn concurrent_submitters_share_one_reactor() {
    on_fresh_thread(|| {
        force_notification_path();
        // Establish the device on the coordinating thread first so workers
        // retain an already-live primary context.
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        const THREADS: usize = 8;
        const OPS_PER_THREAD: usize = 40;
        let buf = 8 << 20;

        let workers: Vec<_> = (0..THREADS)
            .map(|t| {
                std::thread::spawn(move || {
                    // Device contexts are thread-local: each worker sets up
                    // its own view of the (refcounted) primary context.
                    init_device_contexts(0, 1).expect("per-thread init failed");
                    let dptr = alloc_device(buf);
                    let value = (t as u8).wrapping_add(1);
                    for _ in 0..OPS_PER_THREAD {
                        let op = MemsetOp {
                            dptr,
                            bytes: buf,
                            passes: 2,
                            value,
                        };
                        block_on_with_deadline(op.into_future(), DEADLINE)
                            .expect("concurrent op failed");
                    }
                    // Each thread owns its buffer, so its value must survive
                    // intact — a cross-thread slot mix-up would corrupt it.
                    let host = read_device(dptr, 4096);
                    assert!(
                        host.iter().all(|&b| b == value),
                        "thread {t}: data corrupted (slot mix-up?)"
                    );
                })
            })
            .collect();

        for (t, w) in workers.into_iter().enumerate() {
            w.join()
                .unwrap_or_else(|_| panic!("worker thread {t} panicked"));
        }
    });
}
