/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DriverError, PinnedHostBuffer};
use std::sync::{mpsc, Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;

const BLOCKED_TIMEOUT: Duration = Duration::from_millis(100);
const COMPLETION_TIMEOUT: Duration = Duration::from_secs(1);
/// The error the sticky-state tests record; any code that is not `Ok` works.
const INJECTED: DriverError = DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE);
static TEST_LOCK: Mutex<()> = Mutex::new(());

/// Serializes every test in this file.
///
/// The gated tests park a `cuLaunchHostFunc` callback on a channel until the
/// test thread releases it. While that callback blocks, a synchronizing driver
/// call made by *any other* test thread in this process (`cuMemFree` from a
/// `DeviceBuffer` drop, `cuStreamDestroy`, `cuCtxSynchronize`) waits for the
/// gated stream while holding the driver's context lock, and the gate owner
/// then blocks on that lock inside an unrelated call such as `cuEventCreate`
/// before it can release the gate. Observed as a hard deadlock under the
/// default parallel test runner; running the file serially removes it.
fn serialize_test() -> MutexGuard<'static, ()> {
    TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

enum CompletionObservation<T> {
    Blocked,
    Completed(T),
    Disconnected,
}

fn observe_completion<T>(rx: &mpsc::Receiver<T>) -> CompletionObservation<T> {
    match rx.recv_timeout(BLOCKED_TIMEOUT) {
        Ok(value) => CompletionObservation::Completed(value),
        Err(mpsc::RecvTimeoutError::Timeout) => CompletionObservation::Blocked,
        Err(mpsc::RecvTimeoutError::Disconnected) => CompletionObservation::Disconnected,
    }
}

fn finish_gated_worker<T>(
    label: &str,
    release_gate: mpsc::Sender<()>,
    started_rx: mpsc::Receiver<()>,
    completion_rx: mpsc::Receiver<T>,
    worker: JoinHandle<()>,
) -> T {
    let started_result = started_rx.recv_timeout(COMPLETION_TIMEOUT);
    let observation = observe_completion(&completion_rx);

    // Release the CUDA callback and collect the worker before asserting the
    // observation. A failing regression must not leave driver work blocked.
    let release_result = release_gate.send(());
    let (was_blocked, disconnected, completion_result) = match observation {
        CompletionObservation::Blocked => {
            (true, false, completion_rx.recv_timeout(COMPLETION_TIMEOUT))
        }
        CompletionObservation::Completed(value) => (false, false, Ok(value)),
        CompletionObservation::Disconnected => {
            (false, true, Err(mpsc::RecvTimeoutError::Disconnected))
        }
    };
    let worker_result = worker.join();

    started_result.unwrap_or_else(|error| panic!("{label} worker did not start: {error}"));
    release_result.unwrap_or_else(|error| panic!("failed to release {label} gate: {error}"));
    worker_result.unwrap_or_else(|_| panic!("{label} worker panicked"));
    assert!(!disconnected, "{label} worker disconnected");
    assert!(
        was_blocked,
        "{label} completed before the gated stream was released"
    );
    completion_result
        .unwrap_or_else(|error| panic!("{label} did not complete after releasing gate: {error}"))
}

fn gate_stream(stream: &CudaStream) -> mpsc::Sender<()> {
    let (tx, rx) = mpsc::channel();
    stream
        .launch_host_function(move || {
            let _ = rx.recv();
        })
        .expect("failed to enqueue stream gate");
    tx
}

#[test]
fn device_buffer_from_host_roundtrip() {
    let _guard = serialize_test();
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let data = [1_u32, 2, 3, 4, 5];
    let dev_buf =
        DeviceBuffer::from_host(&stream, &data).expect("failed to allocate DeviceBuffer from host");

    assert_eq!(dev_buf.len(), 5);
    assert_eq!(dev_buf.num_bytes(), 20);
    assert!(!dev_buf.is_empty());

    let host_vec = dev_buf
        .to_host_vec(&stream)
        .expect("failed to copy back to host");
    assert_eq!(host_vec, data);
}

/// Every `DeviceBuffer` entry point binds `stream.context()` itself. A
/// freshly spawned thread has no current CUDA context, so a constructor that
/// reached `cuMemAlloc` without binding would fail with
/// `CUDA_ERROR_INVALID_CONTEXT` there (and, on a multi-GPU host, a thread
/// bound to another device would allocate in the wrong context).
#[test]
fn device_buffer_binds_the_stream_context_on_an_unbound_thread() {
    let _guard = serialize_test();
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");
    let pinned =
        PinnedHostBuffer::from_slice(&ctx, &[5_u32, 8, 13, 21]).expect("failed to allocate pinned");

    std::thread::spawn(move || {
        let mut current: cuda_bindings::CUcontext = std::ptr::null_mut();
        assert_eq!(
            unsafe { cuda_bindings::cuCtxGetCurrent(&mut current) },
            cuda_bindings::cudaError_enum_CUDA_SUCCESS
        );
        assert!(
            current.is_null(),
            "a new thread must start without a context"
        );

        let zeroed =
            DeviceBuffer::<u32>::zeroed(&stream, 4).expect("zeroed must bind the stream context");
        let from_host = DeviceBuffer::from_host(&stream, &[1_u32, 2, 3, 4])
            .expect("from_host must bind the stream context");
        // SAFETY: `pinned` outlives the `to_host_vec` synchronization below.
        let from_pinned = unsafe { DeviceBuffer::from_pinned_host(&stream, &pinned) }
            .expect("from_pinned_host must bind the stream context");
        // SAFETY: the buffer is fully written by `copy_from_host` before it is read.
        let mut uninit = unsafe { DeviceBuffer::<u32>::uninitialized_async(&stream, 4) }
            .expect("uninitialized_async must bind the stream context");
        uninit
            .copy_from_host(&stream, &[9_u32, 9, 9, 9])
            .expect("copy_from_host must bind the stream context");

        assert_eq!(zeroed.to_host_vec(&stream).expect("readback"), [0, 0, 0, 0]);
        assert_eq!(
            from_host.to_host_vec(&stream).expect("readback"),
            [1, 2, 3, 4]
        );
        assert_eq!(
            from_pinned.to_host_vec(&stream).expect("readback"),
            pinned.as_slice()
        );
        assert_eq!(uninit.to_host_vec(&stream).expect("readback"), [9, 9, 9, 9]);
    })
    .join()
    .expect("worker thread panicked");
}

#[test]
fn device_buffer_zeroed_initializes_with_zeros() {
    let _guard = serialize_test();
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let dev_buf =
        DeviceBuffer::<f32>::zeroed(&stream, 4).expect("failed to allocate zeroed DeviceBuffer");

    assert_eq!(dev_buf.len(), 4);
    assert_eq!(dev_buf.num_bytes(), 16);

    let host_vec = dev_buf
        .to_host_vec(&stream)
        .expect("failed to copy back to host");
    assert_eq!(host_vec, &[0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn device_buffer_supports_empty_allocations() {
    let _guard = serialize_test();
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let dev_buf =
        DeviceBuffer::<u8>::zeroed(&stream, 0).expect("failed to allocate empty device buffer");
    assert_eq!(dev_buf.len(), 0);
    assert_eq!(dev_buf.num_bytes(), 0);
    assert!(dev_buf.is_empty());

    let dev_buf_host = DeviceBuffer::<u8>::from_host(&stream, &[])
        .expect("failed to allocate empty device buffer from empty slice");
    assert_eq!(dev_buf_host.len(), 0);
    assert_eq!(dev_buf_host.num_bytes(), 0);
    assert!(dev_buf_host.is_empty());
}

#[test]
fn device_buffer_rejects_allocation_size_overflow() {
    let _guard = serialize_test();
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");
    let overflowing_len = usize::MAX / std::mem::size_of::<u64>() + 1;

    assert!(DeviceBuffer::<u64>::zeroed(&stream, overflowing_len).is_err());
    // SAFETY: the constructor returns an error before allocation, and this
    // test never reads from the uninitialized buffer.
    assert!(unsafe { DeviceBuffer::<u64>::uninitialized_async(&stream, overflowing_len) }.is_err());
}

#[test]
fn device_buffer_async_compat_methods_roundtrip() {
    let _guard = serialize_test();
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let data = [7_u32, 11, 13, 17];
    let mut dev = unsafe { DeviceBuffer::<u32>::uninitialized_async(&stream, data.len()) }
        .expect("failed to allocate uninitialized device buffer");
    // SAFETY: `data` remains alive and unmodified until the later
    // `to_host_vec` call synchronizes the stream.
    unsafe { dev.copy_from_host_async_unchecked(&stream, &data) }
        .expect("failed to copy host data into device buffer");

    let mut clone = unsafe { DeviceBuffer::<u32>::uninitialized_async(&stream, data.len()) }
        .expect("failed to allocate clone device buffer");
    clone
        .copy_from_device_async(&dev, &stream)
        .expect("failed to copy device buffer");
    assert_eq!(
        clone
            .to_host_vec(&stream)
            .expect("failed to copy clone back to host"),
        data
    );

    clone
        .zero_async(&stream)
        .expect("failed to zero device buffer");
    assert_eq!(
        clone
            .to_host_vec(&stream)
            .expect("failed to copy zeroed buffer back to host"),
        [0, 0, 0, 0]
    );

    // SAFETY: all prior work on these buffers ran on `stream` itself.
    unsafe { clone.drop_async(&stream) }.expect("failed to async free clone");
    unsafe { dev.drop_async(&stream) }.expect("failed to async free source");

    let empty = unsafe { DeviceBuffer::<u8>::uninitialized_async(&stream, 0) }
        .expect("failed to allocate empty uninitialized device buffer");
    // SAFETY: the empty buffer has no pending work on any stream.
    unsafe { empty.drop_async(&stream) }.expect("failed to async free empty buffer");
    stream.synchronize().expect("stream sync failed");
}

#[test]
fn async_allocation_ordinary_drop_waits_for_cross_stream_work() {
    let _guard = serialize_test();
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let allocation_stream = ctx
        .new_stream()
        .expect("failed to create allocation stream");
    let use_stream = ctx.new_stream().expect("failed to create use stream");

    let mut dev = unsafe { DeviceBuffer::<u32>::uninitialized_async(&allocation_stream, 4) }
        .expect("failed to allocate async device buffer");
    use_stream
        .join(&allocation_stream)
        .expect("failed to order use stream after allocation stream");
    let release_gate = gate_stream(&use_stream);
    dev.zero_async(&use_stream)
        .expect("failed to enqueue cross-stream use");

    let (started_tx, started_rx) = mpsc::channel();
    let (completion_tx, completion_rx) = mpsc::channel();
    let drop_thread = std::thread::spawn(move || {
        started_tx
            .send(())
            .expect("failed to send drop worker start");
        drop(dev);
        completion_tx
            .send(())
            .expect("failed to send drop completion");
    });
    finish_gated_worker(
        "ordinary async buffer drop",
        release_gate,
        started_rx,
        completion_rx,
        drop_thread,
    );
    ctx.synchronize().expect("context cleanup failed");
}

#[test]
fn async_allocation_drop_async_orders_free_after_allocation_stream() {
    let _guard = serialize_test();
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let allocation_stream = ctx
        .new_stream()
        .expect("failed to create allocation stream");
    let free_stream = ctx.new_stream().expect("failed to create free stream");

    let mut dev = unsafe { DeviceBuffer::<u32>::uninitialized_async(&allocation_stream, 4) }
        .expect("failed to allocate async device buffer");
    let release_gate = gate_stream(&allocation_stream);
    dev.zero_async(&allocation_stream)
        .expect("failed to enqueue allocation-stream work");

    // SAFETY: the only pending work runs on the allocation stream, which
    // drop_async orders `free_stream` after; that ordering is what this
    // test observes.
    unsafe { dev.drop_async(&free_stream) }
        .expect("drop_async should order free after allocation stream");
    let (started_tx, started_rx) = mpsc::channel();
    let (completion_tx, completion_rx) = mpsc::channel();
    let free_stream_for_thread = free_stream.clone();
    let sync_thread = std::thread::spawn(move || {
        started_tx
            .send(())
            .expect("failed to send free-stream sync worker start");
        completion_tx
            .send(free_stream_for_thread.synchronize())
            .expect("failed to send free-stream sync result");
    });
    finish_gated_worker(
        "cross-stream async free",
        release_gate,
        started_rx,
        completion_rx,
        sync_thread,
    )
    .expect("cross-stream async free failed");
}

#[test]
fn sync_allocation_allows_async_drop_after_queued_work() {
    let _guard = serialize_test();
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let mut dev = DeviceBuffer::<u32>::zeroed(&stream, 4)
        .expect("failed to allocate synchronous device buffer");
    let release_gate = gate_stream(&stream);
    dev.zero_async(&stream)
        .expect("failed to enqueue work before async free");
    // SAFETY: all prior work on this buffer was enqueued on `stream` itself.
    unsafe { dev.drop_async(&stream) }
        .expect("synchronous allocation should support stream-ordered free");

    let (started_tx, started_rx) = mpsc::channel();
    let (completion_tx, completion_rx) = mpsc::channel();
    let stream_for_thread = stream.clone();
    let sync_thread = std::thread::spawn(move || {
        started_tx
            .send(())
            .expect("failed to send stream sync worker start");
        completion_tx
            .send(stream_for_thread.synchronize())
            .expect("failed to send stream sync result");
    });
    finish_gated_worker(
        "synchronous allocation async free",
        release_gate,
        started_rx,
        completion_rx,
        sync_thread,
    )
    .expect("synchronous allocation async free failed");
}

/// Gates `stream`, enqueues a device-to-host copy into pinned memory behind
/// the gate, records a sticky error, and runs the wait built by `make_wait`
/// on a worker thread. The wait must block until the gate is released, then
/// report the recorded error, and the copy must be complete by the time it
/// returns: a wait that returned the recorded error *instead of* waiting
/// would hand the caller host memory the DMA is still writing.
fn assert_wait_completes_under_sticky_error<W>(
    label: &str,
    make_wait: impl FnOnce(&Arc<CudaContext>, &Arc<CudaStream>) -> W,
) where
    W: FnOnce() -> Result<(), DriverError> + Send + 'static,
{
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");
    let expected = [1_u32, 2, 3, 4];
    let src = DeviceBuffer::from_host(&stream, &expected).expect("failed to allocate source");
    let mut dst = PinnedHostBuffer::<u32>::zeroed(&ctx, expected.len())
        .expect("failed to allocate pinned destination");

    let release_gate = gate_stream(&stream);
    // SAFETY: `dst` stays alive, unread, and unaliased until the worker's
    // wait has returned and been checked below.
    unsafe { src.copy_to_pinned_host_async(&stream, &mut dst) }
        .expect("failed to enqueue the gated copy");
    let wait = make_wait(&ctx, &stream);
    ctx.record_err::<()>(Err(INJECTED));

    let (started_tx, started_rx) = mpsc::channel();
    let (completion_tx, completion_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        started_tx
            .send(())
            .expect("failed to send wait worker start");
        completion_tx
            .send(wait())
            .expect("failed to send wait result");
    });
    let result = finish_gated_worker(label, release_gate, started_rx, completion_rx, worker);

    assert_eq!(
        result,
        Err(INJECTED),
        "{label}: the recorded error must be reported after the wait, not swallowed"
    );
    assert_eq!(
        ctx.check_err(),
        Ok(()),
        "{label}: the wait must have drained the recorded error"
    );
    assert_eq!(
        stream.query(),
        Ok(true),
        "{label}: the wait must have covered the in-flight copy"
    );
    assert_eq!(
        dst.as_slice(),
        &expected,
        "{label}: the copy must have landed before the wait returned"
    );
}

#[test]
fn stream_synchronize_waits_for_in_flight_work_under_a_sticky_error() {
    let _guard = serialize_test();
    assert_wait_completes_under_sticky_error("CudaStream::synchronize", |_, stream| {
        let stream = Arc::clone(stream);
        move || stream.synchronize()
    });
}

#[test]
fn context_synchronize_waits_for_in_flight_work_under_a_sticky_error() {
    let _guard = serialize_test();
    assert_wait_completes_under_sticky_error("CudaContext::synchronize", |ctx, _| {
        let ctx = Arc::clone(ctx);
        move || ctx.synchronize()
    });
}

#[test]
fn event_synchronize_waits_for_in_flight_work_under_a_sticky_error() {
    let _guard = serialize_test();
    assert_wait_completes_under_sticky_error("CudaEvent::synchronize", |_, stream| {
        let event = stream
            .record_event(None)
            .expect("failed to record the completion event");
        move || event.synchronize()
    });
}

/// Runs a safe enqueue-then-wait `DeviceBuffer` wrapper while a sticky error
/// lands *between* its enqueue and its wait. The gate callback records the
/// error when released, which the stream orders after the wrapper's own
/// pre-enqueue drain and before the copy it gates, so the wrapper cannot
/// return early: it must finish the copy and then report the error.
///
/// `op` performs the copy from the source buffer into host memory it owns,
/// returning the wrapper's result and the host bytes it can still observe
/// (`None` when the wrapper's error path consumed them).
///
/// Only a pinned destination discriminates: the driver stages a
/// device-to-host copy into pageable memory synchronously, so `copy_to_host`
/// and `to_host_vec` complete inside the enqueue call regardless of what the
/// wait does afterwards. They are exercised for coverage of the reporting
/// path; `copy_to_pinned_host` is the case that fails without the fix.
fn assert_copy_wrapper_finishes_before_reporting_sticky_error<F>(label: &str, op: F)
where
    F: FnOnce(&DeviceBuffer<u32>, &Arc<CudaStream>) -> (Result<(), DriverError>, Option<Vec<u32>>)
        + Send
        + 'static,
{
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");
    let expected = vec![1_u32, 2, 3, 4];
    let src = DeviceBuffer::from_host(&stream, &expected).expect("failed to allocate source");

    let (release_gate, release_rx) = mpsc::channel();
    let ctx_for_gate = Arc::clone(&ctx);
    stream
        .launch_host_function(move || {
            let _ = release_rx.recv();
            ctx_for_gate.record_err::<()>(Err(INJECTED));
        })
        .expect("failed to enqueue the recording gate");

    let (started_tx, started_rx) = mpsc::channel();
    let (completion_tx, completion_rx) = mpsc::channel();
    let stream_for_thread = Arc::clone(&stream);
    let worker = std::thread::spawn(move || {
        started_tx
            .send(())
            .expect("failed to send copy worker start");
        completion_tx
            .send(op(&src, &stream_for_thread))
            .expect("failed to send copy result");
    });
    let (result, observed) =
        finish_gated_worker(label, release_gate, started_rx, completion_rx, worker);

    assert_eq!(
        result,
        Err(INJECTED),
        "{label}: the error recorded during the copy must be reported"
    );
    assert_eq!(
        stream.query(),
        Ok(true),
        "{label}: the wrapper must not return while its copy is in flight"
    );
    if let Some(observed) = observed {
        assert_eq!(
            observed, expected,
            "{label}: the copy must have completed before the wrapper returned"
        );
    }
}

#[test]
fn copy_to_pinned_host_finishes_the_copy_before_reporting_a_sticky_error() {
    let _guard = serialize_test();
    assert_copy_wrapper_finishes_before_reporting_sticky_error(
        "copy_to_pinned_host",
        |src, stream| {
            let mut dst = PinnedHostBuffer::<u32>::zeroed(stream.context(), src.len())
                .expect("failed to allocate pinned destination");
            let result = src.copy_to_pinned_host(stream, &mut dst);
            (result, Some(dst.to_vec()))
        },
    );
}

#[test]
fn copy_to_host_finishes_the_copy_before_reporting_a_sticky_error() {
    let _guard = serialize_test();
    assert_copy_wrapper_finishes_before_reporting_sticky_error("copy_to_host", |src, stream| {
        let mut dst = vec![0_u32; src.len()];
        let result = src.copy_to_host(stream, &mut dst);
        (result, Some(dst))
    });
}

#[test]
fn to_host_vec_finishes_the_copy_before_reporting_a_sticky_error() {
    let _guard = serialize_test();
    assert_copy_wrapper_finishes_before_reporting_sticky_error("to_host_vec", |src, stream| {
        (src.to_host_vec(stream).map(drop), None)
    });
}

#[test]
fn drop_async_bind_error_preserves_ordinary_cleanup() {
    let _guard = serialize_test();
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");
    let dev = DeviceBuffer::<u8>::zeroed(&stream, 4096).expect("failed to allocate device buffer");
    let ptr = dev.cu_deviceptr();

    let mut base = 0;
    let mut size = 0;
    assert_eq!(
        unsafe { cuda_bindings::cuMemGetAddressRange_v2(&mut base, &mut size, ptr) },
        cuda_bindings::cudaError_enum_CUDA_SUCCESS,
        "allocation must be live before drop_async"
    );

    let injected = DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE);
    ctx.record_err::<()>(Err(injected));
    assert_eq!(
        // SAFETY: no pending work on any stream touches this buffer.
        unsafe { dev.drop_async(&stream) },
        Err(injected),
        "drop_async must propagate the pre-disarm bind error"
    );

    ctx.bind_to_thread()
        .expect("ordinary drop should leave the context usable");
    assert_eq!(
        unsafe { cuda_bindings::cuMemGetAddressRange_v2(&mut base, &mut size, ptr) },
        cuda_bindings::cudaError_enum_CUDA_ERROR_NOT_FOUND,
        "ordinary drop must reclaim the still-armed allocation"
    );
}

#[test]
fn from_host_with_pinned_source_allows_source_drop_after_return() {
    let _guard = serialize_test();
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let expected = vec![21_u32, 34, 55, 89];
    let input =
        PinnedHostBuffer::from_slice(&ctx, &expected).expect("failed to allocate pinned input");
    let dev = DeviceBuffer::from_host(&stream, input.as_slice())
        .expect("failed to copy pinned input to device");
    drop(input);

    assert_eq!(
        dev.to_host_vec(&stream)
            .expect("failed to copy device buffer back to host"),
        expected
    );
}

#[test]
fn copy_from_host_with_pinned_source_allows_source_reuse_after_return() {
    let _guard = serialize_test();
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let expected = vec![3_u32, 5, 8, 13];
    let mut input =
        PinnedHostBuffer::from_slice(&ctx, &expected).expect("failed to allocate pinned input");
    let mut dev =
        DeviceBuffer::<u32>::zeroed(&stream, input.len()).expect("failed to allocate device");

    dev.copy_from_host(&stream, input.as_slice())
        .expect("failed to copy pinned input to device");
    input.as_mut_slice().fill(0);

    assert_eq!(
        dev.to_host_vec(&stream)
            .expect("failed to copy device buffer back to host"),
        expected
    );
}

// Dangerous-path regression: a stream-ordered (`cuMemAllocAsync`) buffer that
// is dropped *implicitly* while a copy is still in flight, with no explicit
// synchronization by the caller. Ordinary `Drop` synchronizes the context
// before enqueueing the free, so the in-flight copy cannot race deallocation.
// Run under `compute-sanitizer --tool memcheck` to catch a regression.
#[test]
fn uninitialized_async_implicit_drop_waits_for_pending_work() {
    let _guard = serialize_test();
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let n = 1 << 20; // 4 MiB of u32
    let src = DeviceBuffer::<u32>::zeroed(&stream, n).expect("failed to allocate source buffer");
    for _ in 0..64 {
        let mut dst = unsafe { DeviceBuffer::<u32>::uninitialized_async(&stream, n) }
            .expect("failed to allocate uninitialized device buffer");
        // Enqueue a large async device-to-device copy, then let `dst` drop
        // immediately with no `stream.synchronize()` in between.
        dst.copy_from_device_async(&src, &stream)
            .expect("failed to enqueue device-to-device copy");
        drop(dst);
    }
    stream.synchronize().expect("stream sync failed");
}

#[test]
fn uninitialized_async_cast_elem_implicit_drop_is_stream_ordered() {
    let _guard = serialize_test();
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let n = 1 << 20; // 4 MiB of u32
    let src = DeviceBuffer::<u32>::zeroed(&stream, n).expect("failed to allocate source buffer");
    for _ in 0..64 {
        let mut dst = unsafe { DeviceBuffer::<u32>::uninitialized_async(&stream, n) }
            .expect("failed to allocate uninitialized device buffer");
        dst.copy_from_device_async(&src, &stream)
            .expect("failed to enqueue device-to-device copy");
        let dst = dst.cast_elem::<std::num::Wrapping<u32>>();
        drop(dst);
    }
    stream.synchronize().expect("stream sync failed");
}
