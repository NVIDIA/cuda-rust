/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tests for concurrent CUDA stream capture behavior.

use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::value;
use cuda_async::error::DeviceError;

fn has_gpu() -> bool {
    cuda_core::Device::device_count()
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

    let device = cuda_core::Device::new(0).unwrap();
    let stream_a = device.new_stream().unwrap();
    let stream_b = device.new_stream().unwrap();

    let handle_a = std::thread::spawn(move || -> Result<(), DeviceError> {
        CudaGraph::scope(&stream_a, |s| {
            s.record(value(1))?;
            std::thread::sleep(std::time::Duration::from_millis(50));
            s.record(value(2))?;
            Ok(())
        })?;
        Ok(())
    });

    let handle_b = std::thread::spawn(move || -> Result<(), DeviceError> {
        CudaGraph::scope(&stream_b, |s| {
            s.record(value(3))?;
            std::thread::sleep(std::time::Duration::from_millis(50));
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

    let device_a = cuda_core::Device::new(0).unwrap();
    let stream_a = device_a.new_stream().unwrap();

    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    let barrier_a = barrier.clone();
    let barrier_b = barrier.clone();

    let handle_a = std::thread::spawn(move || -> Result<(), DeviceError> {
        CudaGraph::scope(&stream_a, |s| {
            s.record(value(1))?;
            barrier_a.wait();
            std::thread::sleep(std::time::Duration::from_millis(100));
            s.record(value(2))?;
            Ok(())
        })?;
        Ok(())
    });

    let handle_b = std::thread::spawn(move || -> Result<(), DeviceError> {
        barrier_b.wait();
        let device_b = cuda_core::Device::new(0).unwrap();
        let _stream = device_b.new_stream()?;
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
