/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Completion-bound retirement without waiting in destructors.

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

struct Pending {
    event: Option<Arc<Event>>,
    cleanup: Cleanup,
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

fn enqueue(pending: Pending) {
    static QUEUE: OnceLock<Option<mpsc::Sender<Pending>>> = OnceLock::new();
    let queue = QUEUE.get_or_init(|| {
        let (sender, receiver) = mpsc::channel::<Pending>();
        std::thread::Builder::new()
            .name("cuda-resource-retirement".into())
            .spawn(move || {
                let mut pending = Vec::<Pending>::new();
                loop {
                    match receiver.recv_timeout(Duration::from_millis(1)) {
                        Ok(item) => pending.push(item),
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                    }
                    pending.extend(receiver.try_iter());
                    let mut i = 0;
                    while i < pending.len() {
                        let ready = pending[i].event.as_ref().map_or(Ok(true), |event| {
                            with_retirement_capture_mode(&event.context, || event.query())
                        });
                        match ready {
                            Ok(false) => i += 1,
                            Ok(true) => {
                                let item = pending.swap_remove(i);
                                let _ =
                                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                        match item.event {
                                            Some(event) => {
                                                let _ = cleanup_after_event(event, item.cleanup);
                                            }
                                            None => (item.cleanup)(),
                                        }
                                    }));
                            }
                            Err(_) => {
                                // Completion is unknown: leaking the owner and its access
                                // lease is safer than freeing memory or allowing conflicting work.
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
    if let Some(queue) = queue {
        if let Err(error) = queue.send(pending) {
            std::mem::forget(error.0);
        }
    } else {
        std::mem::forget(pending);
    }
}

pub(crate) fn defer(cleanup: impl FnOnce() + Send + 'static) {
    enqueue(Pending {
        event: None,
        cleanup: Box::new(cleanup),
    });
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
    enqueue(Pending {
        event: Some(Arc::new(event)),
        cleanup: Box::new(move || drop(owner)),
    });
    Ok(())
}

pub(crate) fn retain_until_event(owner: impl Send + 'static, event: Arc<Event>) {
    enqueue(Pending {
        event: Some(event),
        cleanup: Box::new(move || drop(owner)),
    });
}
