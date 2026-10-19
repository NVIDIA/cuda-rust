/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `CudaStream::query` reports completion without blocking.
//!
//! Work is enqueued with `launch_host_function` rather than a kernel: the
//! point under test is the driver's completion status for a stream, a host
//! callback holds the stream busy for a known interval, and `cuda-core`'s own
//! tests do not compile device code.

use cuda_core::CudaContext;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Long enough that the first query lands inside it on a loaded machine,
/// short enough to keep the suite quick.
const BUSY: Duration = Duration::from_millis(500);

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
    let finished = Arc::new(AtomicBool::new(false));

    let callback_finished = finished.clone();
    stream
        .launch_host_function(move || {
            std::thread::sleep(BUSY);
            callback_finished.store(true, Ordering::SeqCst);
        })
        .expect("failed to enqueue the host callback");

    // The call itself must return while the callback is still running. A
    // query() that fell back to synchronize() would pass an "is it done"
    // assertion later on, so the check is on both the answer and the time
    // taken to give it.
    let before = Instant::now();
    let in_flight = stream.query().expect("query() failed on a busy stream");
    let elapsed = before.elapsed();

    assert!(
        !in_flight,
        "a stream whose host callback is still sleeping must report incomplete"
    );
    assert!(
        elapsed < BUSY / 2,
        "query() must return without waiting for the work, took {elapsed:?}"
    );
    assert!(
        !finished.load(Ordering::SeqCst),
        "the callback must still have been running when query() answered"
    );

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
