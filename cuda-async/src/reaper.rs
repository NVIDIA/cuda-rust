/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Non-blocking release of what an abandoned in-flight future still owns.
//!
//! Dropping a [`DeviceFuture`] cannot cancel GPU work that has already been
//! submitted. What the drop decides is *when the host releases the resources
//! that work still uses*: the tensor storage and other owners retained by the
//! submission (see [`ExecutionContext::retain`]). Releasing them early would
//! free memory the device may still write; waiting for the stream inside
//! `Drop` stalls whichever thread runs the drop, which under `select!` or a
//! timeout is an executor thread.
//!
//! The reaper removes the wait without weakening the guarantee. On drop, the
//! future's [`ExecutionContext`] is handed to the completion reactor together
//! with a flag write enqueued *behind* the abandoned work on the same stream.
//! When the flag lands, the reactor passes the context to a dedicated release
//! thread, whose drop of the context releases the submission's owners. The
//! dropping thread returns immediately.
//!
//! ```text
//!   drop(future)                    reactor (flag lands)     reaper thread
//!     release result handle           hand off context   ->    drop context:
//!     arm flag write on the stream                             owners released
//!     park context, return
//! ```
//!
//! Only the crate's own drops run on the reactor and reaper threads: the
//! parked payload is an [`ExecutionContext`], whose owners are storage
//! leases, stream/pool handles, and whatever an operation registered with
//! [`ExecutionContext::retain`]. The future's *result* is not parked. By the
//! [`DeviceOp::execute`] contract every device-visible resource is retained
//! in the submission, so a result is a handle that may be released at any
//! time, on the dropping thread, with the caller's own `Drop` semantics.
//!
//! # Fallbacks
//!
//! Parking is best effort and never trades safety for latency. If the stream
//! is already idle the context is simply dropped. If the reactor cannot take
//! the payload (slot pool exhausted, stream mem-ops unavailable, reactor
//! failed to start), the context is dropped inline, which is the previous
//! blocking wait. A stream mid-capture must not receive a flag write, and a
//! faulted stream can never land one; both drop inline too, where the
//! submission's own release reports and leaks the owners instead of freeing
//! them.
//!
//! [`DeviceFuture`]: crate::device_future::DeviceFuture
//! [`DeviceOp::execute`]: crate::device_operation::DeviceOp::execute
//! [`ExecutionContext::retain`]: crate::device_operation::ExecutionContext::retain

use crate::device_future::{probe_stream, StreamHealth};
use crate::device_operation::ExecutionContext;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, OnceLock};
use std::thread;

/// An abandoned submission: dropping it releases the submission's owners.
pub(crate) struct Parked {
    /// Held only for its `Drop`: releasing the submission's owners.
    _ctx: ExecutionContext,
}

impl Drop for Parked {
    fn drop(&mut self) {
        // `_ctx` drops after this body: the submission waits (or, on a stream
        // it cannot wait on, leaks) and then releases its owners.
        PARKED.fetch_sub(1, Ordering::AcqRel);
        REAPED.fetch_add(1, Ordering::Relaxed);
    }
}

/// Contexts currently parked: handed to the reactor, flag not yet landed
/// or release not yet run.
static PARKED: AtomicUsize = AtomicUsize::new(0);
/// Contexts released by the reaper since process start.
static REAPED: AtomicU64 = AtomicU64::new(0);

/// Number of abandoned submissions whose resources are still held by the
/// reaper. Memory an abandoned future owned now outlives its dropping scope
/// by however long its GPU work takes; this is how much is outstanding.
pub fn parked() -> usize {
    PARKED.load(Ordering::Acquire)
}

/// Number of abandoned submissions the reaper has released so far.
pub fn reaped_total() -> u64 {
    REAPED.load(Ordering::Relaxed)
}

/// Releases `ctx` without blocking the caller when possible.
///
/// Called from [`DeviceFuture`]'s drop after the result handle has been
/// released. See the module docs for the exact policy.
///
/// [`DeviceFuture`]: crate::device_future::DeviceFuture
pub(crate) fn park_or_wait(ctx: ExecutionContext) {
    let stream = Arc::clone(ctx.get_cuda_stream());
    // Probing and arming need a current context; a thread that cannot bind
    // the device falls back to the inline release, which reports the error.
    if stream.device().bind_to_thread().is_err() {
        drop(ctx);
        return;
    }
    match probe_stream(&stream) {
        // Nothing in flight: the inline release costs one more query.
        StreamHealth::Idle => drop(ctx),
        StreamHealth::Busy => {
            PARKED.fetch_add(1, Ordering::AcqRel);
            let parked = Parked { _ctx: ctx };
            // SAFETY: the stream is valid and its context is current on this
            // thread (bound above); the flag write is ordered behind the
            // abandoned work because it is enqueued on the same stream.
            if let Err(parked) = unsafe { crate::reactor::park(&stream, parked) } {
                // The reactor could not take it: release inline, which is the
                // blocking wait. `Parked::drop` keeps the counters honest.
                drop(parked);
            }
        }
        // A capturing stream must not receive a flag write, and a faulted one
        // can never land it. The submission's release handles both: it
        // cannot prove completion, so it reports and leaks the owners.
        StreamHealth::Capturing | StreamHealth::Faulted(_) => drop(ctx),
    }
}

/// Passes a landed payload to the release thread, so the reactor's scan loop
/// never runs a release itself. If the thread is gone the payload is
/// released on the caller's thread: still sound, just not isolated.
pub(crate) fn release(parked: Parked) {
    if let Err(mpsc::SendError(parked)) = sender().send(parked) {
        drop(parked);
    }
}

fn sender() -> &'static Sender<Parked> {
    static SENDER: OnceLock<Sender<Parked>> = OnceLock::new();
    SENDER.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Parked>();
        // If the thread cannot be spawned the receiver drops here, every
        // later `send` fails, and `release` falls back to the inline drop.
        let _ = thread::Builder::new()
            .name("cuda-async-reaper".into())
            .spawn(move || {
                for parked in rx {
                    drop(parked);
                }
            });
        tx
    })
}
