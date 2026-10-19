/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Error types and assertion helpers for CUDA device operations.
//!
//! [`DeviceError`] is the unified error type for the `cuda-async` crate. It
//! captures failures from the CUDA driver, device context management, kernel
//! cache lookups, scheduling, and kernel launches. Assertion helpers provide a
//! concise way to produce contextual errors in device-facing code.

use cuda_core::DriverError;

/// Unified error type for all `cuda-async` operations.
///
/// Each variant identifies a distinct failure domain so callers can
/// pattern-match on the error source without parsing messages.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum DeviceError {
    /// Propagated from the CUDA driver API (see [`DriverError`]).
    #[error("CUDA driver error: {0}")]
    Driver(#[from] DriverError),

    /// Device context initialization or access failure for a specific device.
    #[error("device context error (device_id={device_id}): {message}")]
    Context {
        /// Ordinal of the device that reported the error.
        device_id: usize,
        /// Human-readable description of the failure.
        message: String,
    },

    /// Failure during kernel module loading or function cache lookup.
    #[error("kernel cache error: {0}")]
    KernelCache(String),

    /// Stream scheduling or policy configuration error.
    #[error("scheduling error: {0}")]
    Scheduling(String),

    /// Kernel launch parameter validation or driver-level launch failure.
    #[error("kernel launch error: {0}")]
    Launch(String),

    /// Logic error internal to the async runtime.
    #[error("internal error: {0}")]
    Internal(String),

    /// Catch-all for errors converted from [`anyhow::Error`].
    #[error("{0}")]
    Anyhow(String),
}

/// Converts an [`anyhow::Error`] by capturing its debug representation.
impl From<anyhow::Error> for DeviceError {
    fn from(error: anyhow::Error) -> Self {
        DeviceError::Anyhow(format!("{:?}", error))
    }
}

/// Returns `Err(DeviceError::Launch)` when `pred` is `false`.
///
/// Use for precondition checks on kernel launch parameters (grid/block dims,
/// shared memory size, argument counts, etc.).
pub fn kernel_launch_assert(pred: bool, message: &str) -> Result<(), DeviceError> {
    if !pred {
        Err(DeviceError::Launch(message.to_string()))
    } else {
        Ok(())
    }
}

/// Returns `Err(DeviceError::Context)` when `pred` is `false`.
///
/// Attaches `device_id` to the error for per-device diagnostics.
pub fn device_assert(device_id: usize, pred: bool, message: &str) -> Result<(), DeviceError> {
    if !pred {
        Err(DeviceError::Context {
            device_id,
            message: message.to_string(),
        })
    } else {
        Ok(())
    }
}

/// Constructs a [`DeviceError::Context`] for the given device and message.
pub fn device_error(device_id: usize, message: &str) -> DeviceError {
    DeviceError::Context {
        device_id,
        message: message.to_string(),
    }
}
