/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Dropping a `DeviceFuture` while its GPU work is in flight must not release
//! the operation's output before the device has finished with it.
//!
//! The op enqueues a chain of memsets and then records a CUDA event on the
//! same stream, handing the event back inside its output. The output's
//! `Drop` queries that event: if the output were released early, the query
//! would report the work still in flight.
//!
//! The core test does not rely on GPU speed to keep that work in flight. Its
//! op starts with a [`Gate`]: a `cuStreamWaitValue32` on a word of pinned
//! host memory that a helper thread opens after a fixed delay. Until then
//! nothing behind the gate can run, whatever the device's memset throughput
//! or the inline-spin budget (`CUDA_ASYNC_SPIN_BUDGET_US`), so the first poll
//! is guaranteed to find the work in flight and the drop is guaranteed to
//! have to wait for it. Requires a GPU.

use cuda_async::device_context::{global_policy, init_device_contexts, with_device};
use cuda_async::device_operation::{DeviceOp, ExecutionContext};
use cuda_async::error::DeviceError;
use cuda_core::Event;
use std::future::{Future, IntoFuture};
use std::mem::MaybeUninit;
use std::pin::Pin;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
use std::thread::JoinHandle;
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

/// A host-controlled barrier at the head of a stream.
///
/// [`arm`](Gate::arm) enqueues a `cuStreamWaitValue32` on the gate's word, a
/// page of pinned, device-mapped host memory; work submitted to the stream
/// after that stays queued until [`open`](Gate::open) stores the release
/// value. This is the host-mapped-flag mechanism the crate's completion
/// reactor uses, in the other direction.
struct Gate {
    host: *mut u32,
    dptr: cuda_bindings::CUdeviceptr,
}

// SAFETY: `host` is pinned, device-mapped memory owned by the gate for its
// whole lifetime; the host side only touches it through atomics.
unsafe impl Send for Gate {}
unsafe impl Sync for Gate {}

impl Gate {
    /// Allocates a closed gate. Requires a current CUDA context on the
    /// calling thread.
    fn closed() -> Arc<Self> {
        let mut host = MaybeUninit::uninit();
        let flags =
            cuda_bindings::CU_MEMHOSTALLOC_PORTABLE | cuda_bindings::CU_MEMHOSTALLOC_DEVICEMAP;
        let code = unsafe {
            cuda_bindings::cuMemHostAlloc(host.as_mut_ptr(), std::mem::size_of::<u32>(), flags)
        };
        assert_eq!(code, 0, "cuMemHostAlloc failed: {code}");
        let host = unsafe { host.assume_init() } as *mut u32;
        unsafe { AtomicU32::from_ptr(host) }.store(0, Ordering::SeqCst);
        let mut dptr = MaybeUninit::uninit();
        let code =
            unsafe { cuda_bindings::cuMemHostGetDevicePointer_v2(dptr.as_mut_ptr(), host as _, 0) };
        assert_eq!(code, 0, "cuMemHostGetDevicePointer failed: {code}");
        Arc::new(Self {
            host,
            dptr: unsafe { dptr.assume_init() },
        })
    }

    /// Enqueues the wait: everything submitted to `stream` after this call
    /// stays queued until the gate is opened.
    fn arm(&self, stream: cuda_bindings::CUstream) -> Result<(), DeviceError> {
        // CUDA driver flag bindings have platform-dependent integer types, so
        // the FFI call casts the flag as `_`.
        let code = unsafe {
            cuda_bindings::cuStreamWaitValue32_v2(
                stream,
                self.dptr,
                1,
                cuda_bindings::CUstreamWaitValue_flags_enum_CU_STREAM_WAIT_VALUE_GEQ as _,
            )
        };
        if code != cuda_bindings::cudaError_enum_CUDA_SUCCESS {
            return Err(DeviceError::Internal(format!(
                "cuStreamWaitValue32 failed: {code}"
            )));
        }
        Ok(())
    }

    /// Releases every stream waiting on the gate.
    fn open(&self) {
        unsafe { AtomicU32::from_ptr(self.host) }.store(1, Ordering::SeqCst);
    }

    /// Opens the gate from a helper thread once `delay` has elapsed.
    fn open_after(self: &Arc<Self>, delay: Duration) -> JoinHandle<()> {
        let gate = Arc::clone(self);
        std::thread::spawn(move || {
            std::thread::sleep(delay);
            gate.open();
        })
    }
}

impl Drop for Gate {
    fn drop(&mut self) {
        // Every wait on the gate has passed by now: the test keeps its own
        // handle until the gated stream has been synchronized. The last owner
        // may be the opener thread, which never bound a context.
        let _ = with_device(0, |device| device.bind_to_thread());
        let code = unsafe { cuda_bindings::cuMemFreeHost(self.host as _) };
        if !std::thread::panicking() {
            assert_eq!(code, 0, "cuMemFreeHost failed: {code}");
        }
    }
}

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
}

impl DeviceOp for SlowOp {
    type Output = Tracked;
    unsafe fn execute(self, context: &ExecutionContext) -> Result<Tracked, DeviceError> {
        let stream = context.get_cuda_stream();
        if let Some(gate) = &self.gate {
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
        let event = stream.device().new_event()?;
        event.record(stream)?;
        Ok(Tracked {
            event,
            log: self.log,
        })
    }
}

impl IntoFuture for SlowOp {
    type Output = Result<Tracked, DeviceError>;
    type IntoFuture = cuda_async::device_future::DeviceFuture<Tracked, SlowOp>;
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
/// How long a gate stays closed. Generous, so the first poll happens while
/// the gate is provably closed even on a loaded machine, and so a drop that
/// waits measurably cannot finish before the gate opens.
const GATE_DELAY: Duration = Duration::from_millis(200);

fn slow_op(dptr: u64, log: &DropLog) -> SlowOp {
    SlowOp {
        dptr,
        bytes: BUF,
        passes: PASSES,
        gate: None,
        log: Arc::clone(log),
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
/// The gate opens from a helper thread after `GATE_DELAY`, so a drop that
/// waits cannot return before then, and one that does not is caught twice:
/// it returns early, and its output's event reports the work still in flight.
#[test]
fn dropping_in_flight_future_releases_output_after_the_device_finished() {
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

        for _ in 0..4 {
            let gate = Gate::closed();
            let mut future = gated_op(dptr, &log, &gate).into_future();
            let waker = noop_waker();
            let mut cx = Context::from_waker(&waker);
            let started = Instant::now();
            let opener = gate.open_after(GATE_DELAY);
            match Pin::new(&mut future).poll(&mut cx) {
                Poll::Pending => {}
                Poll::Ready(Ok(_)) => {
                    let elapsed = started.elapsed();
                    assert!(
                        elapsed >= GATE_DELAY,
                        "the first poll found the stream idle after {elapsed:?} with the gate \
                         still closed: the gate did not hold the work back"
                    );
                    panic!(
                        "the first poll resolved only once the gate had opened ({elapsed:?}): \
                         the op was never left in flight (a huge CUDA_ASYNC_SPIN_BUDGET_US, or \
                         CUDA_LAUNCH_BLOCKING, in the environment?)"
                    );
                }
                Poll::Ready(Err(error)) => {
                    panic!("the first poll failed instead of leaving the op in flight: {error}")
                }
            }
            drop(future);
            let elapsed = started.elapsed();
            assert!(
                elapsed >= GATE_DELAY,
                "the drop returned after {elapsed:?}, before the gate opened at {GATE_DELAY:?}: \
                 it did not wait for the in-flight work"
            );
            opener.join().expect("gate opener thread panicked");
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

        let started = Instant::now();
        drop(slow_op(dptr, &log).into_future());
        assert!(started.elapsed() < Duration::from_millis(50));
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
