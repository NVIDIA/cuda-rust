/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tests for concurrent CUDA stream capture behavior.

use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::value;
use cuda_async::error::DeviceError;
use std::sync::mpsc;
use std::time::Duration;

const WATCHDOG: Duration = Duration::from_secs(60);

fn has_gpu() -> bool {
    cuda_core::Context::device_count()
        .map(|n| n > 0)
        .unwrap_or(false)
}

/// Two threads capture simultaneously on different streams from
/// the same CUDA context.
#[test]
fn concurrent_capture_same_context() {
    if !has_gpu() {
        return;
    }

    let device = cuda_core::Context::new(0).unwrap();
    let stream_a = device.new_stream().unwrap();
    let stream_b = device.new_stream().unwrap();

    let (a_entered, wait_a) = mpsc::channel();
    let (b_entered, wait_b) = mpsc::channel();
    let handle_a = std::thread::spawn(move || -> Result<(), DeviceError> {
        CudaGraph::scope(&stream_a, |s| {
            s.record(value(1))?;
            a_entered.send(()).unwrap();
            wait_b
                .recv_timeout(WATCHDOG)
                .expect("capture B did not enter");
            s.record(value(2))?;
            Ok(())
        })?;
        Ok(())
    });

    let handle_b = std::thread::spawn(move || -> Result<(), DeviceError> {
        CudaGraph::scope(&stream_b, |s| {
            s.record(value(3))?;
            b_entered.send(()).unwrap();
            wait_a
                .recv_timeout(WATCHDOG)
                .expect("capture A did not enter");
            s.record(value(4))?;
            Ok(())
        })?;
        Ok(())
    });

    let result_a = handle_a.join().expect("thread A panicked");
    let result_b = handle_b.join().expect("thread B panicked");

    assert!(
        result_a.is_ok() && result_b.is_ok(),
        "Concurrent capture failed: A={result_a:?}, B={result_b:?}"
    );
}

/// Thread A captures while thread B creates a new stream from a
/// fresh Device::new(0). Previously this failed because
/// new_stream called cuCtxSynchronize (for event tracking init),
/// which conflicted with A's active capture.
#[test]
fn new_stream_during_capture_on_another_thread() {
    if !has_gpu() {
        return;
    }

    let device_a = cuda_core::Context::new(0).unwrap();
    let stream_a = device_a.new_stream().unwrap();

    let (entered, capture_started) = mpsc::channel();
    let (finished, stream_created) = mpsc::channel();

    let handle_a = std::thread::spawn(move || -> Result<(), DeviceError> {
        CudaGraph::scope(&stream_a, |s| {
            s.record(value(1))?;
            entered.send(()).unwrap();
            stream_created
                .recv_timeout(WATCHDOG)
                .expect("stream creation did not finish during capture");
            s.record(value(2))?;
            Ok(())
        })?;
        Ok(())
    });

    let handle_b = std::thread::spawn(move || -> Result<(), DeviceError> {
        capture_started
            .recv_timeout(WATCHDOG)
            .expect("capture did not start");
        let device_b = cuda_core::Context::new(0).unwrap();
        let result = device_b.new_stream();
        finished.send(()).unwrap();
        let _stream = result?;
        Ok(())
    });

    let result_a = handle_a.join().expect("thread A panicked");
    let result_b = handle_b.join().expect("thread B panicked");

    assert!(result_a.is_ok(), "Capture should succeed: {result_a:?}");
    assert!(
        result_b.is_ok(),
        "new_stream should succeed during concurrent capture: {result_b:?}"
    );
}
