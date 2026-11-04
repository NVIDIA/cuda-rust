/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `CudaStream::query` reports completion without blocking.
//!
//! Work is enqueued with `launch_host_function` rather than a kernel: the
//! point under test is the driver's completion status for a stream, a host
//! callback holds the stream busy until the test releases it, and
//! `cuda-core`'s own tests do not compile device code.

use cuda_core::CudaContext;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::Arc;
use std::time::Duration;

/// Stops a hang only. Expiring is recorded, and the test fails on it.
const WATCHDOG: Duration = Duration::from_secs(30);

#[test]
fn an_idle_stream_reports_complete() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create stream");

    assert!(
        stream.query().expect("query() failed on an idle stream"),
        "a stream with nothing enqueued must report complete"
    );
}

#[test]
fn query_reports_in_flight_work_and_never_blocks() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create stream");
    let (started_tx, started) = mpsc::channel();
    let (release, gate) = mpsc::channel::<()>();
    let finished = Arc::new(AtomicBool::new(false));
    let watchdog_expired = Arc::new(AtomicBool::new(false));

    let callback_finished = finished.clone();
    let callback_expired = watchdog_expired.clone();
    stream
        .launch_host_function(move || {
            let _ = started_tx.send(());
            if let Err(RecvTimeoutError::Timeout) = gate.recv_timeout(WATCHDOG) {
                callback_expired.store(true, Ordering::SeqCst);
            }
            callback_finished.store(true, Ordering::SeqCst);
        })
        .expect("failed to enqueue the host callback");
    started
        .recv_timeout(WATCHDOG)
        .expect("the host callback never started");

    // The gate opens only after query() answers, so a query() that waited for
    // the work (for example by falling back to synchronize()) could return
    // only once the watchdog gave up, and would then report complete.
    let in_flight = stream.query().expect("query() failed on a busy stream");
    assert!(
        !watchdog_expired.load(Ordering::SeqCst),
        "query() waited for the gated work until the watchdog expired"
    );
    assert!(
        !in_flight,
        "a stream whose host callback is still gated must report incomplete"
    );
    assert!(
        !finished.load(Ordering::SeqCst),
        "the callback must still have been running when query() answered"
    );

    release
        .send(())
        .expect("the host callback dropped its gate");
    stream.synchronize().expect("synchronize() failed");
    assert!(
        stream.query().expect("query() failed after synchronize()"),
        "a stream must report complete once its work has finished"
    );
    assert!(
        finished.load(Ordering::SeqCst),
        "synchronize() must have waited for the callback"
    );
}
