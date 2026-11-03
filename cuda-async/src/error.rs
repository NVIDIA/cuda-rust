/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Error types and assertion helpers for CUDA device operations.

use cuda_core::DriverError;
use thiserror;
/// Errors that can occur during CUDA device operations.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum DeviceError {
    #[error("CUDA driver error: {0}")]
    Driver(#[from] DriverError),

    #[error("device context error (device_id={device_id}): {message}")]
    Context { device_id: usize, message: String },

    #[error("kernel cache error: {0}")]
    KernelCache(String),

    #[error("scheduling error: {0}")]
    Scheduling(String),

    #[error("kernel launch error: {0}")]
    Launch(String),

    #[error("internal error: {0}")]
    Internal(String),

    #[error("{0}")]
    Anyhow(String),
}

impl From<anyhow::Error> for DeviceError {
    fn from(error: anyhow::Error) -> Self {
        DeviceError::Anyhow(format!("{:?}", error))
    }
}

/// Returns `Err(DeviceError::Launch)` if `pred` is false; the message is
/// built only on failure, so passing checks cost no allocation.
#[inline]
pub fn kernel_launch_assert_with(
    pred: bool,
    message: impl FnOnce() -> String,
) -> Result<(), DeviceError> {
    if !pred {
        Err(DeviceError::Launch(message()))
    } else {
        Ok(())
    }
}

/// Returns `Err(DeviceError::Launch)` if `pred` is false.
pub fn kernel_launch_assert(pred: bool, message: &str) -> Result<(), DeviceError> {
    if !pred {
        Err(DeviceError::Launch(message.to_string()))
    } else {
        Ok(())
    }
}

/// Returns `Err(DeviceError::Context)` if `pred` is false.
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

/// Constructs a `DeviceError::Context` for the given device and message.
pub fn device_error(device_id: usize, message: &str) -> DeviceError {
    DeviceError::Context {
        device_id,
        message: message.to_string(),
    }
}
