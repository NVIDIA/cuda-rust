/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Error types for the cutile crate, covering tensor, kernel launch, JIT, and device errors.

use cuda_async::error::DeviceError;
use cuda_core::DriverError;
use cutile_compiler::error::JITError;
use std::{
    error,
    fmt::{self, Display, Formatter},
};

/// An error originating from an invalid tensor operation (e.g. shape mismatch).
#[derive(Debug)]
pub struct TensorError(pub String);

impl Display for TensorError {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let message = &self.0;
        write!(formatter, "{message}")
    }
}

impl error::Error for TensorError {}

/// An error raised when a kernel launch fails (e.g. mismatched grid dimensions).
#[derive(Debug)]
pub struct KernelLaunchError(pub String);

impl Display for KernelLaunchError {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let message = &self.0;
        write!(formatter, "{message}")
    }
}

impl error::Error for KernelLaunchError {}

/// Unified error type aggregating all error sources in the cutile crate.
pub enum Error {
    /// A tensor-related error (shape, rank, size, etc.).
    Tensor(TensorError),
    /// A kernel launch configuration error (grid, block, shared memory, etc.).
    KernelLaunch(KernelLaunchError),
    /// A JIT compilation error from the cutile compiler.
    JIT(JITError),
    /// An asynchronous device execution error.
    Device(DeviceError),
    /// A low-level CUDA driver error.
    Driver(DriverError),
    /// A catch-all error from `anyhow`.
    Anyhow(anyhow::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl fmt::Debug for Error {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        // Delegate to Display so that `.unwrap()` produces the same
        // user-facing output (including source locations from JITError).
        Display::fmt(self, formatter)
    }
}

impl Display for Error {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        match self {
            Self::Tensor(error) => write!(formatter, "error: {error}"),
            Self::KernelLaunch(error) => write!(formatter, "error: {error}"),
            Self::JIT(error) => write!(formatter, "error: {error}"),
            Self::Device(error) => write!(formatter, "error: {error}"),
            Self::Driver(error) => write!(formatter, "error: {error}"),
            Self::Anyhow(error) => write!(formatter, "error: {error}"),
        }
    }
}

impl error::Error for Error {}

// Tensor

/// Creates a `Error::Tensor` from the given message string.
pub fn tensor_error(err_str: &str) -> Error {
    Error::Tensor(TensorError(err_str.to_string()))
}

/// Returns `Err(Error::Tensor(...))` with the given message string.
pub fn tensor_error_result<R>(err_str: &str) -> Result<R, Error> {
    Err(tensor_error(err_str))
}

// Kernel Launch

/// Creates a `Error::KernelLaunch` from the given message string.
pub fn kernel_launch_error(err_str: &str) -> Error {
    Error::KernelLaunch(KernelLaunchError(err_str.to_string()))
}

/// Returns `Err(Error::KernelLaunch(...))` with the given message string.
pub fn kernel_launch_error_result<R>(err_str: &str) -> Result<R, Error> {
    Err(kernel_launch_error(err_str))
}

// anyhow

/// Converts an `anyhow::Error` into `Error::Anyhow`.
impl From<anyhow::Error> for Error {
    fn from(error: anyhow::Error) -> Self {
        Self::Anyhow(error)
    }
}

// Device

/// Converts an `Error` into a `DeviceError` by formatting via `Debug`.
impl From<Error> for DeviceError {
    fn from(error: Error) -> Self {
        DeviceError::Anyhow(format!("{:?}", error))
    }
}

/// Converts a `DeviceError` into `Error::Device`.
impl From<DeviceError> for Error {
    fn from(error: DeviceError) -> Self {
        Self::Device(error)
    }
}

// DriverError

/// Converts a `DriverError` into `Error::Driver`.
impl From<DriverError> for Error {
    fn from(error: DriverError) -> Self {
        Self::Driver(error)
    }
}

// Syn

/// Converts a `JITError` into `Error::JIT`.
impl From<JITError> for Error {
    fn from(error: JITError) -> Self {
        Self::JIT(error)
    }
}
