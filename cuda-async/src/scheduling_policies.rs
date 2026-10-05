/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Stream scheduling policies that control how operations are assigned to CUDA streams.

use crate::error::DeviceError;
use cuda_core::{Device, Stream};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

/// A strategy for selecting which CUDA stream to run the next operation on.
///
/// This is the only decision a policy makes — which stream. The `DeviceOp` trait
/// handles the rest (constructing futures, synchronizing, etc.).
///
/// Object-safe: no generic methods. Stored as `Arc<dyn SchedulingPolicy>` in the
/// thread-local device context.
///
/// # Stream Ordering Guarantees
///
/// CUDA guarantees that work items on the **same stream** execute in submission order.
/// Work on **different streams** has no ordering guarantee.
pub trait SchedulingPolicy: Send + Sync {
    /// Select the next CUDA stream for an operation.
    fn next_stream(&self) -> Result<Arc<Stream>, DeviceError>;
}

/// Distributes operations across a fixed-size pool of CUDA streams using round-robin assignment.
///
/// This is the **default scheduling policy**. Each call to `next_stream()` picks the
/// next stream in the pool (wrapping around), so consecutive operations typically land
/// on **different streams** and may run concurrently on the GPU.
pub struct StreamPoolRoundRobin {
    next_stream_idx: AtomicUsize,
    stream_pool: Vec<Arc<Stream>>,
}

impl StreamPoolRoundRobin {
    /// Creates a round-robin pool with `num_streams` streams on the given device.
    pub fn new(device: &Arc<Device>, num_streams: usize) -> Result<Self, DeviceError> {
        let mut stream_pool = Vec::with_capacity(num_streams);
        for _ in 0..num_streams {
            stream_pool.push(device.new_stream()?);
        }
        Ok(Self {
            stream_pool,
            next_stream_idx: AtomicUsize::new(0),
        })
    }
}

impl SchedulingPolicy for StreamPoolRoundRobin {
    fn next_stream(&self) -> Result<Arc<Stream>, DeviceError> {
        let idx = self
            .next_stream_idx
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            % self.stream_pool.len();
        Ok(Arc::clone(&self.stream_pool[idx]))
    }
}

/// Routes every operation to a single CUDA stream, guaranteeing strict sequential execution.
pub struct SingleStream {
    stream: Arc<Stream>,
}

impl SingleStream {
    /// Creates a single-stream policy on the given device.
    pub fn new(device: &Arc<Device>) -> Result<Self, DeviceError> {
        Ok(Self {
            stream: device.new_stream()?,
        })
    }

    /// Returns a reference to the underlying stream.
    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }
}

impl SchedulingPolicy for SingleStream {
    fn next_stream(&self) -> Result<Arc<Stream>, DeviceError> {
        Ok(Arc::clone(&self.stream))
    }
}
