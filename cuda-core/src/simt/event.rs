/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA event management (RAII, timing, synchronization).
//!
//! A [`CudaEvent`] wraps a `CUevent` handle and ties its lifetime to its
//! parent [`CudaContext`]. Events are the fundamental synchronization
//! primitive between CUDA streams: record an event on one stream, then wait
//! on it from another to establish ordering.
//!
//! # Timing
//!
//! By default, events are created with `CU_EVENT_DISABLE_TIMING` for lower
//! overhead. Pass `Some(CU_EVENT_DEFAULT)` to
//! [`CudaContext::new_event`] or [`CudaStream::record_event`] if you
//! need [`elapsed_ms`](CudaEvent::elapsed_ms).

use crate::error::{DriverError, IntoResult};
use crate::simt::context::CudaContext;
use crate::simt::stream::CudaStream;
use std::mem::MaybeUninit;
use std::sync::Arc;

/// An RAII wrapper around a `CUevent` handle.
///
/// Holds an `Arc<CudaContext>` to ensure the context outlives the event.
/// Destroyed automatically via `cuEventDestroy` on [`Drop`].
#[derive(Debug)]
pub struct CudaEvent {
    /// Raw CUDA event handle.
    pub(crate) cu_event: cuda_bindings::CUevent,
    /// Owning context. Kept alive for the lifetime of this event.
    pub(crate) ctx: Arc<CudaContext>,
}

/// # Safety
///
/// `CUevent` handles are not thread-local. The CUDA driver permits recording
/// and waiting on events from any thread, provided the owning context is bound.
unsafe impl Send for CudaEvent {}
/// See [`Send`] impl.
unsafe impl Sync for CudaEvent {}

/// Destroys the underlying `CUevent` on drop.
///
/// Binds the context to the current thread first (required by
/// `cuEventDestroy`). Errors are recorded on the context rather than
/// panicking.
impl Drop for CudaEvent {
    fn drop(&mut self) {
        self.ctx.record_err(self.ctx.bind_to_thread());
        self.ctx
            .record_err(unsafe { cuda_bindings::cuEventDestroy_v2(self.cu_event).result() });
    }
}

impl CudaContext {
    /// Creates a new CUDA event in this context.
    ///
    /// `flags` defaults to `CU_EVENT_DISABLE_TIMING` when `None`. Use
    /// `Some(CU_EVENT_DEFAULT)` to enable timing queries via
    /// [`CudaEvent::elapsed_ms`].
    pub fn new_event(
        self: &Arc<Self>,
        flags: Option<cuda_bindings::CUevent_flags>,
    ) -> Result<CudaEvent, DriverError> {
        let flags = flags.unwrap_or(cuda_bindings::CUevent_flags_enum_CU_EVENT_DISABLE_TIMING);
        self.bind_to_thread()?;
        let mut cu_event = MaybeUninit::uninit();
        let cu_event = unsafe {
            cuda_bindings::cuEventCreate(cu_event.as_mut_ptr(), flags).result()?;
            cu_event.assume_init()
        };
        Ok(CudaEvent {
            cu_event,
            ctx: self.clone(),
        })
    }
}

impl CudaEvent {
    /// Returns the raw `CUevent` handle.
    pub fn cu_event(&self) -> cuda_bindings::CUevent {
        self.cu_event
    }

    /// Returns the parent [`CudaContext`].
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Records this event on `stream`.
    ///
    /// The event captures the point in the stream's work queue at the time of
    /// the call. A subsequent [`CudaStream::wait`] on this event will block
    /// the waiting stream until all work prior to the record point completes.
    pub fn record(&self, stream: &CudaStream) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { cuda_bindings::cuEventRecord(self.cu_event, stream.cu_stream()).result() }
    }

    /// Blocks the calling thread until this event has been recorded and all
    /// preceding stream work has completed.
    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { cuda_bindings::cuEventSynchronize(self.cu_event).result() }
    }

    /// Returns `true` when all work captured by this event has completed,
    /// `false` when it is still in flight. Never blocks.
    ///
    /// Wraps `cuEventQuery`, mapping `CUDA_SUCCESS` to `Ok(true)` and
    /// `CUDA_ERROR_NOT_READY` to `Ok(false)`. Any other code is a real
    /// driver error.
    pub fn query(&self) -> Result<bool, DriverError> {
        self.ctx.bind_to_thread()?;
        match unsafe { cuda_bindings::cuEventQuery(self.cu_event) } {
            cuda_bindings::cudaError_enum_CUDA_SUCCESS => Ok(true),
            cuda_bindings::cudaError_enum_CUDA_ERROR_NOT_READY => Ok(false),
            err => Err(DriverError(err)),
        }
    }

    /// Returns the elapsed time in milliseconds between `self` (start) and
    /// `end`.
    ///
    /// Both events are synchronized before querying. Both events must have
    /// been created **without** `CU_EVENT_DISABLE_TIMING`; otherwise the
    /// driver returns `CUDA_ERROR_INVALID_HANDLE`.
    ///
    /// `self` must have been recorded before `end` in wall-clock time.
    pub fn elapsed_ms(&self, end: &Self) -> Result<f32, DriverError> {
        self.synchronize()?;
        end.synchronize()?;
        let mut ms: f32 = 0.0;
        unsafe {
            cuda_bindings::cu_event_elapsed_time(&mut ms as *mut _, self.cu_event, end.cu_event)
                .result()?;
        }
        Ok(ms)
    }
}
