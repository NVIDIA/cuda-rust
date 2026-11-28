/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Background completion polling and resource cleanup.
//!
//! Stream destructors may still call the driver to snapshot completion before
//! returning; the workers here wait for that completion and release resources.

use crate::{Context, DriverError, Event, IntoResult, Stream};
use std::borrow::Borrow;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::{mpsc, Arc, OnceLock};
use std::time::Duration;

type Cleanup = Box<dyn FnOnce() + Send>;

/// Temporarily permits operations on retirement resources outside every capture.
/// The saved mode belongs to this host thread, so the guard must stay on it.
struct RetirementCaptureMode<'a> {
    context: &'a Context,
    previous: cuda_bindings::CUstreamCaptureMode,
    restored: bool,
    _thread_affinity: PhantomData<Rc<()>>,
}

impl RetirementCaptureMode<'_> {
    fn restore(&mut self) -> Result<(), DriverError> {
        let mut previous = self.previous;
        // SAFETY: previous was returned by a successful exchange on this thread.
        unsafe { cuda_bindings::cuThreadExchangeStreamCaptureMode(&mut previous).result()? };
        self.restored = true;
        Ok(())
    }
}

impl Drop for RetirementCaptureMode<'_> {
    fn drop(&mut self) {
        if !self.restored {
            let result = self.restore();
            self.context.record_err(result);
        }
    }
}

/// Runs only CUDA operations on resources known not to belong to a capture.
/// Restore the interaction mode before invoking arbitrary cleanup or owners.
pub(crate) fn with_retirement_capture_mode<T>(
    context: &Context,
    operation: impl FnOnce() -> Result<T, DriverError>,
) -> Result<T, DriverError> {
    let mut previous = cuda_bindings::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_RELAXED;
    // SAFETY: previous is a valid mode and a writable output pointer.
    unsafe { cuda_bindings::cuThreadExchangeStreamCaptureMode(&mut previous).result() }
        .inspect_err(|error| context.record_err(Err::<(), _>(*error)))?;
    let mut mode = RetirementCaptureMode {
        context,
        previous,
        restored: false,
        _thread_affinity: PhantomData,
    };
    let result = operation();
    if let Err(error) = &result {
        context.record_err(Err::<(), _>(*error));
    }
    let restored = mode.restore();
    context.record_err(restored);
    restored?;
    result
}

/// Scheduling for event polling and ordinary cleanup jobs.
trait Retirement: Send + 'static {
    fn is_ready(&self) -> Result<bool, DriverError> {
        Ok(true)
    }
    fn finish(self);
}

impl Retirement for Cleanup {
    fn finish(self) {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(self));
    }
}

struct PendingCompletion {
    event: Arc<Event>,
    cleanup: Cleanup,
}

impl Retirement for PendingCompletion {
    fn is_ready(&self) -> Result<bool, DriverError> {
        with_retirement_capture_mode(&self.event.context, || self.event.query())
    }

    fn finish(self) {
        // Owner destructors can block; keep them off the poller.
        defer(move || {
            let _ = cleanup_after_event(self.event, self.cleanup);
        });
    }
}

pub(crate) fn cleanup_after_event(
    event: impl Borrow<Event>,
    cleanup: impl FnOnce(),
) -> Result<(), DriverError> {
    // Restore the worker's capture mode before event destruction or owner cleanup.
    let complete =
        with_retirement_capture_mode(&event.borrow().context, || event.borrow().synchronize());
    if complete.is_err() {
        std::mem::forget((event, cleanup));
        return complete;
    }
    // An external owner may destroy a borrowed context. Destroy the event first
    // and keep owned context metadata alive through cleanup, including unwinding.
    let _context = event.borrow().context.clone();
    drop(event);
    cleanup();
    Ok(())
}

fn enqueue<T: Retirement>(queue: &OnceLock<Option<mpsc::Sender<T>>>, name: &str, job: T) {
    let queue = queue.get_or_init(|| {
        let (sender, receiver) = mpsc::channel::<T>();
        std::thread::Builder::new()
            .name(name.into())
            .spawn(move || {
                let mut pending = Vec::<T>::new();
                loop {
                    match receiver.recv_timeout(Duration::from_millis(1)) {
                        Ok(item) => pending.push(item),
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                    }
                    pending.extend(receiver.try_iter());
                    let mut i = 0;
                    while i < pending.len() {
                        match pending[i].is_ready() {
                            Ok(false) => i += 1,
                            Ok(true) => pending.swap_remove(i).finish(),
                            Err(_) => {
                                // Unknown completion/restoration: retain the event,
                                // owner and access lease instead of risking early reuse.
                                std::mem::forget(pending.swap_remove(i));
                            }
                        }
                    }
                }
                for item in pending {
                    std::mem::forget(item);
                }
            })
            .ok()
            .map(|_| sender)
    });
    // A failed worker cannot safely release the resources in its queued jobs.
    if let Some(queue) = queue {
        if let Err(error) = queue.send(job) {
            std::mem::forget(error.0);
        }
    } else {
        std::mem::forget(job);
    }
}

pub(crate) fn defer(cleanup: impl FnOnce() + Send + 'static) {
    static QUEUE: OnceLock<Option<mpsc::Sender<Cleanup>>> = OnceLock::new();
    enqueue(&QUEUE, "cuda-owner-cleanup", Box::new(cleanup));
}

pub(crate) fn defer_stream(cleanup: impl FnOnce() + Send + 'static) {
    static QUEUE: OnceLock<Option<mpsc::Sender<Cleanup>>> = OnceLock::new();
    enqueue(&QUEUE, "cuda-stream-retirement", Box::new(cleanup));
}

/// Reclaims synchronous allocations on a private per-thread stream. A batch
/// holds every context until its final stream synchronization has completed.
pub(crate) fn defer_sync_free(ptr: cuda_bindings::CUdeviceptr, context: Arc<Context>) {
    struct Free {
        ptr: cuda_bindings::CUdeviceptr,
        context: Arc<Context>,
    }
    static QUEUE: OnceLock<Option<mpsc::Sender<Free>>> = OnceLock::new();
    let queue = QUEUE.get_or_init(|| {
        let (sender, receiver) = mpsc::channel::<Free>();
        std::thread::Builder::new()
            .name("cuda-sync-free".into())
            .spawn(move || {
                while let Ok(first) = receiver.recv() {
                    // Allow a burst of drops to share one device fence.
                    std::thread::sleep(Duration::from_micros(75));
                    let stream = crate::runtime::PER_THREAD_STREAM_HANDLE as cuda_bindings::CUstream;
                    let batch = std::iter::once(first).chain(receiver.try_iter().take(4095));
                    let mut pending = std::collections::HashMap::<usize, Vec<Arc<Context>>>::new();
                    for Free { ptr, context } in batch {
                        if let Err(error) = context.bind_to_thread() {
                            context.record_err(Err::<(), _>(error));
                            std::mem::forget(context);
                            continue;
                        }
                        let pools = unsafe {
                            crate::cudarc_shim::device::get_attribute(
                                context.cu_device(),
                                cuda_bindings::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
                            )
                        } == Ok(1);
                        let result = if pools {
                            // This worker's per-thread stream is private in the
                            // bound context; all access leases have drained.
                            unsafe { crate::memory::free_async(ptr, stream) }
                        } else {
                            unsafe { crate::memory::free_sync(ptr) }
                        };
                        if let Err(error) = result {
                            context.record_err(Err::<(), _>(error));
                            std::mem::forget(context);
                        } else if pools {
                            pending.entry(context.cu_ctx() as usize).or_default().push(context);
                        }
                    }
                    for (_, contexts) in pending {
                        let result = contexts[0].bind_to_thread().and_then(|()| unsafe {
                            crate::cudarc_shim::stream::synchronize(stream)
                        });
                        for context in &contexts {
                            context.record_err(result);
                        }
                        if result.is_err() {
                            std::mem::forget(contexts);
                        }
                    }
                }
            })
            .ok()
            .map(|_| sender)
    });
    let free = Free { ptr, context };
    if let Some(queue) = queue {
        if let Err(error) = queue.send(free) {
            std::mem::forget(error.0);
        }
    } else {
        std::mem::forget(free);
    }
}

// Context workers have a separate registry: a blocked owner in one context
// must not hold up another context's retirement.
pub(crate) fn defer_context(key: usize, cleanup: impl FnOnce() + Send + 'static) {
    use std::collections::HashMap;
    use std::sync::Mutex;
    static WORKERS: OnceLock<Mutex<HashMap<usize, mpsc::Sender<Cleanup>>>> = OnceLock::new();
    let workers = WORKERS.get_or_init(Default::default);
    let mut queues = workers.lock().unwrap_or_else(|e| e.into_inner());
    let queue = queues.entry(key).or_insert_with(|| {
        let (sender, receiver) = mpsc::channel::<Cleanup>();
        let _ = std::thread::Builder::new()
            .name("cuda-context-retirement".into())
            .spawn(move || loop {
                let job = match receiver.recv_timeout(Duration::from_millis(100)) {
                    Ok(job) => job,
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        let mut queues = workers.lock().unwrap_or_else(|e| e.into_inner());
                        match receiver.try_recv() {
                            Ok(job) => job,
                            Err(_) => {
                                queues.remove(&key);
                                break;
                            }
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                };
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
            });
        sender
    });
    if let Err(error) = queue.send(Box::new(cleanup)) {
        std::mem::forget(error.0);
    }
}

/// Keeps an owner or access lease alive until work already queued on `stream`
/// completes. Dropping the caller's future cannot shorten this lifetime.
///
/// Failure to record or observe completion intentionally leaks the owner;
/// releasing it could invalidate work that the driver has already accepted.
pub fn retain_until_complete(
    owner: impl Send + 'static,
    stream: &Stream,
) -> Result<(), DriverError> {
    let event = match stream.record_event(None) {
        Ok(event) => event,
        Err(error) => {
            std::mem::forget(owner);
            return Err(error);
        }
    };
    retain_until_event(owner, Arc::new(event));
    Ok(())
}

pub(crate) fn retain_until_event(owner: impl Send + 'static, event: Arc<Event>) {
    static QUEUE: OnceLock<Option<mpsc::Sender<PendingCompletion>>> = OnceLock::new();
    enqueue(
        &QUEUE,
        "cuda-completion-polling",
        PendingCompletion {
            event,
            cleanup: Box::new(move || drop(owner)),
        },
    );
}
