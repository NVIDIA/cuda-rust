/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `CudaStream::query` reports completion without blocking.
//!
//! Work is enqueued with `launch_host_function` rather than a kernel: the
//! point under test is the driver's completion status for a stream, a host
//! callback holds the stream busy until the test explicitly releases it, and `cuda-core`'s own
//! tests do not compile device code.

use cuda_core::CudaContext;
#[path = "support/stream_gate.rs"]
mod stream_gate;
use stream_gate::StreamGate;

// Gated work must not overlap another test's context-synchronizing setup.
static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn an_idle_stream_reports_complete() {
    let _guard = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create stream");
    assert!(stream.query().expect("idle query failed"));
}

#[test]
fn query_reports_in_flight_work_and_never_blocks() {
    let _guard = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create stream");
    let gate = StreamGate::arm(&stream);
    assert!(!stream.query().expect("busy query failed"));
    // A synchronizing query cannot reach here until the watchdog releases
    // the gate, which fails this assertion even if it returns false.
    gate.assert_held();
    drop(gate);
    stream.synchronize().expect("synchronize failed");
    assert!(stream.query().expect("completed query failed"));
}
