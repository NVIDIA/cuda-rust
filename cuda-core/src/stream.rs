/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Stream ordering and priority helpers.

pub use crate::runtime::{Stream, Stream as CudaStream};
use crate::{DriverError, Event, IntoResult};
use std::{mem::MaybeUninit, sync::Arc};

impl Stream {
    /// Returns the priority this stream actually runs at.
    ///
    /// Wraps `cuStreamGetPriority`. This is the value to read after
    /// [`Context::new_stream_with_priority`](crate::context::Context::new_stream_with_priority),
    /// which clamps an out-of-range request to the device's range without
    /// reporting it. Lower numbers are higher priorities. The default stream
    /// and any stream from [`Context::new_stream`](crate::context::Context::new_stream)
    /// report `0`.
    pub fn priority(&self) -> Result<i32, DriverError> {
        self.context.bind_to_thread()?;
        let mut priority = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuStreamGetPriority(self.cu_stream, priority.as_mut_ptr()).result()?;
            Ok(priority.assume_init())
        }
    }
    /// Creates a new non-blocking stream that waits on all prior work in
    /// `self` before executing its own.
    ///
    /// Semantically equivalent to creating a new stream and calling
    /// [`join`](Self::join) on it with `self`, establishing a fork point
    /// in the stream DAG.
    pub fn fork(&self) -> Result<Arc<Self>, DriverError> {
        let stream = self.context.new_stream()?;
        stream.join(self)?;
        Ok(stream)
    }
    /// Makes `self` wait on all prior work in `other`.
    ///
    /// Records an event on `other` and enqueues a wait on `self`. This is the
    /// join side of the fork/join pattern: after this call, work enqueued on
    /// `self` is guaranteed to observe all side effects of prior work on
    /// `other`.
    pub fn join(&self, other: &Stream) -> Result<(), DriverError> {
        self.wait(&other.record_event(None)?)
    }
    /// Records an event on this stream and returns it.
    ///
    /// `flags` defaults to `CU_EVENT_DISABLE_TIMING` when `None`, which is
    /// cheaper than a timing-enabled event. Pass
    /// `Some(CU_EVENT_DEFAULT)` if you need [`Event::elapsed_ms`].
    pub fn record_event(
        &self,
        flags: Option<cuda_bindings::CUevent_flags>,
    ) -> Result<Event, DriverError> {
        let event = self.context.new_event_with_flags(flags)?;
        event.record(self)?;
        Ok(event)
    }
    /// Enqueues a wait on `event` into this stream.
    ///
    /// All work enqueued on `self` after this call will not begin until
    /// `event` has been recorded (i.e., all prior work on the stream that
    /// recorded `event` has completed).
    pub fn wait(&self, event: &Event) -> Result<(), DriverError> {
        self.wait_event(event)
    }
}
