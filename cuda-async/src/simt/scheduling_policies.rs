/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Stream scheduling policies that control how [`DeviceOperation`]s are
//! assigned to CUDA streams.
//!
//! A [`SchedulingPolicy`] decouples operation construction from stream
//! selection. The policy chooses a stream when an operation is scheduled or
//! executed synchronously, enabling transparent overlap of independent work.
//!
//! | Policy                   | Behaviour                                    |
//! |--------------------------|----------------------------------------------|
//! | [`StreamPoolRoundRobin`] | Rotates across *N* streams for HW overlap    |
//! | [`SingleStream`]         | Serialises all work onto one stream          |
//!
//! [`DeviceOperation`]: crate::simt::device_operation::DeviceOperation

use crate::simt::device_future::DeviceFuture;
use crate::simt::device_operation::{DeviceOperation, ExecutionContext};
use crate::simt::error::{device_error, DeviceError};
use cuda_core::{CudaContext, CudaStream};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

/// Top-level enum wrapping all supported scheduling policy implementations.
///
/// Used as the concrete policy type stored in
/// [`AsyncDeviceContext`](crate::simt::device_context::AsyncDeviceContext).
pub enum GlobalSchedulingPolicy {
    /// Round-robin across a pool of CUDA streams.
    RoundRobin(StreamPoolRoundRobin),
}

impl GlobalSchedulingPolicy {
    /// Returns a reference to the inner policy as a trait object.
    pub fn as_scheduling_policy(&self) -> Result<&impl SchedulingPolicy, DeviceError> {
        match self {
            GlobalSchedulingPolicy::RoundRobin(rr) => Ok(rr),
        }
    }
}

/// Delegates to the inner policy variant.
impl SchedulingPolicy for GlobalSchedulingPolicy {
    fn init(&mut self, ctx: &Arc<CudaContext>) -> Result<(), DeviceError> {
        match self {
            GlobalSchedulingPolicy::RoundRobin(rr) => rr.init(ctx),
        }
    }
    fn schedule<T: Send, O: DeviceOperation<Output = T>>(
        &self,
        op: O,
    ) -> Result<DeviceFuture<T, O>, DeviceError> {
        match self {
            GlobalSchedulingPolicy::RoundRobin(rr) => rr.schedule(op),
        }
    }
    fn sync<T: Send, O: DeviceOperation<Output = T>>(&self, op: O) -> Result<T, DeviceError> {
        match self {
            GlobalSchedulingPolicy::RoundRobin(rr) => rr.sync(op),
        }
    }
}

/// `Arc`-wrapped policy delegates scheduling but rejects re-initialization,
/// since the inner data cannot be mutated through an `Arc`.
impl SchedulingPolicy for Arc<GlobalSchedulingPolicy> {
    fn init(&mut self, _ctx: &Arc<CudaContext>) -> Result<(), DeviceError> {
        Err(DeviceError::Scheduling(
            "Cannot initialize scheduling policy inside an Arc.".to_string(),
        ))
    }
    fn schedule<T: Send, O: DeviceOperation<Output = T>>(
        &self,
        op: O,
    ) -> Result<DeviceFuture<T, O>, DeviceError> {
        match self.as_ref() {
            GlobalSchedulingPolicy::RoundRobin(rr) => rr.schedule(op),
        }
    }
    fn sync<T: Send, O: DeviceOperation<Output = T>>(&self, op: O) -> Result<T, DeviceError> {
        match self.as_ref() {
            GlobalSchedulingPolicy::RoundRobin(rr) => rr.sync(op),
        }
    }
}

/// Strategy for assigning [`DeviceOperation`]s to CUDA streams.
///
/// Implementations must be `Sync` because the policy is shared across all
/// operations on a device context.
///
/// [`DeviceOperation`]: crate::simt::device_operation::DeviceOperation
pub trait SchedulingPolicy: Sync {
    /// One-time initialization with the CUDA context (creates streams, etc.).
    fn init(&mut self, ctx: &Arc<CudaContext>) -> Result<(), DeviceError>;

    /// Assigns a stream to `op` and returns a [`DeviceFuture`] ready to be
    /// polled.
    fn schedule<T: Send, O: DeviceOperation<Output = T>>(
        &self,
        op: O,
    ) -> Result<DeviceFuture<T, O>, DeviceError>;

    /// Executes `op` synchronously on a policy-chosen stream and blocks until
    /// the stream is idle.
    fn sync<T: Send, O: DeviceOperation<Output = T>>(&self, op: O) -> Result<T, DeviceError>;
}

/// Round-robin scheduler over a fixed-size pool of CUDA streams.
///
/// Each call to [`schedule`](SchedulingPolicy::schedule) or
/// [`sync`](SchedulingPolicy::sync) atomically advances the stream index,
/// distributing work across streams to enable hardware-level overlap of
/// independent kernels and memory transfers.
#[derive(Debug)]
pub struct StreamPoolRoundRobin {
    /// Device ordinal this pool belongs to.
    device_id: usize,
    /// Monotonically increasing counter; wrapped with `% num_streams`.
    next_stream_idx: AtomicUsize,
    /// Number of streams in the pool.
    pub(crate) num_streams: usize,
    /// `None` until [`init`](SchedulingPolicy::init) is called.
    pub(crate) stream_pool: Option<Vec<Arc<CudaStream>>>,
}

impl StreamPoolRoundRobin {
    /// Creates an un-initialized round-robin policy for `device_id` with
    /// `num_streams` streams.
    ///
    /// # Safety
    ///
    /// The caller must call [`SchedulingPolicy::init`] before scheduling any
    /// operations. Using the policy before initialization produces an error.
    pub unsafe fn new(device_id: usize, num_streams: usize) -> Self {
        Self {
            device_id,
            num_streams,
            stream_pool: None,
            next_stream_idx: AtomicUsize::new(0),
        }
    }
}

impl SchedulingPolicy for StreamPoolRoundRobin {
    /// Allocates `num_streams` CUDA streams on the provided context.
    fn init(&mut self, ctx: &Arc<CudaContext>) -> Result<(), DeviceError> {
        let mut pool = vec![];
        for _ in 0..self.num_streams {
            pool.push(ctx.new_stream()?);
        }
        self.stream_pool = Some(pool);
        Ok(())
    }

    /// Picks the next stream (round-robin), executes `op`, and synchronizes.
    fn sync<T: Send, O: DeviceOperation<Output = T>>(&self, op: O) -> Result<T, DeviceError> {
        let idx = self
            .next_stream_idx
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            % self.num_streams;
        let pool = self
            .stream_pool
            .as_ref()
            .ok_or_else(|| device_error(self.device_id, "Stream pool not initialized."))?;
        op.sync_on(&pool[idx])
    }

    /// Picks the next stream (round-robin) and wraps `op` in a
    /// [`DeviceFuture`].
    fn schedule<T: Send, O: DeviceOperation<Output = T>>(
        &self,
        op: O,
    ) -> Result<DeviceFuture<T, O>, DeviceError> {
        let idx = self
            .next_stream_idx
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            % self.num_streams;
        let pool = self
            .stream_pool
            .as_ref()
            .ok_or_else(|| device_error(self.device_id, "Stream pool not initialized."))?;
        // `DeviceFuture` implements `Drop`, so functional-update syntax
        // (`..Default::default()`) would move fields out of a droppable
        // value (E0509). Spell every field out instead.
        Ok(DeviceFuture {
            device_operation: Some(op),
            execution_context: Some(ExecutionContext::new(Arc::clone(&pool[idx]))),
            result: None,
            error: None,
            state: Default::default(),
            callback_state: None,
        })
    }
}

/// Single-stream scheduler that serialises all work onto one CUDA stream.
///
/// Useful for debugging or when ordering guarantees are required between
/// all operations on a device.
#[derive(Debug)]
pub struct SingleStream {
    /// The single stream. `None` until [`init`](SchedulingPolicy::init).
    pub stream: Option<Arc<CudaStream>>,
}

impl SingleStream {
    /// Creates an un-initialized single-stream policy.
    ///
    /// # Safety
    ///
    /// The caller must call [`SchedulingPolicy::init`] before use.
    pub unsafe fn new() -> Self {
        Self { stream: None }
    }
}

impl SchedulingPolicy for SingleStream {
    /// Allocates one CUDA stream on `ctx`.
    fn init(&mut self, ctx: &Arc<CudaContext>) -> Result<(), DeviceError> {
        self.stream = Some(ctx.new_stream()?);
        Ok(())
    }

    /// Wraps `op` in a [`DeviceFuture`] bound to the single stream.
    fn schedule<T: Send, O: DeviceOperation<Output = T>>(
        &self,
        op: O,
    ) -> Result<DeviceFuture<T, O>, DeviceError> {
        let stream = self.stream.as_ref().unwrap();
        // See `StreamPoolRoundRobin::schedule` for why the fields are
        // spelled out instead of using `..Default::default()`.
        Ok(DeviceFuture {
            device_operation: Some(op),
            execution_context: Some(ExecutionContext::new(Arc::clone(stream))),
            result: None,
            error: None,
            state: Default::default(),
            callback_state: None,
        })
    }

    /// Executes `op` synchronously on the single stream.
    fn sync<T: Send, O: DeviceOperation<Output = T>>(&self, op: O) -> Result<T, DeviceError> {
        let stream = self.stream.as_ref().unwrap();
        op.sync_on(stream)
    }
}
