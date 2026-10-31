/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Event timing and configuration helpers.

pub use crate::runtime::{Event, Event as CudaEvent};
use crate::{Context, DriverError, IntoResult};
use std::{mem::MaybeUninit, sync::Arc};

impl Context {
    /// Creates a new CUDA event in this context.
    ///
    /// `flags` defaults to `CU_EVENT_DISABLE_TIMING` when `None`. Use
    /// `Some(CU_EVENT_DEFAULT)` to enable timing queries via
    /// [`Event::elapsed_ms`].
    pub fn new_event_with_flags(
        self: &Arc<Self>,
        flags: Option<cuda_bindings::CUevent_flags>,
    ) -> Result<Event, DriverError> {
        let flags = flags.unwrap_or(cuda_bindings::CUevent_flags_enum_CU_EVENT_DISABLE_TIMING);
        self.bind_to_thread()?;
        let mut cu_event = MaybeUninit::uninit();
        let cu_event = unsafe {
            cuda_bindings::cuEventCreate(cu_event.as_mut_ptr(), flags).result()?;
            cu_event.assume_init()
        };
        Ok(Event {
            cu_event,
            context: self.clone(),
        })
    }
}
impl Event {
    /// Returns the raw `CUevent` handle.
    pub fn cu_event(&self) -> cuda_bindings::CUevent {
        self.cu_event
    }
    /// Returns the parent [`Context`].
    pub fn context(&self) -> &Arc<Context> {
        &self.context
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
