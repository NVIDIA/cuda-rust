/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA driver error types and result conversion utilities.

use std::ffi::CStr;
use std::mem::MaybeUninit;
use std::{
    error,
    fmt::{self, Display, Formatter},
};

/// Wrapper around a CUDA driver API error code.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct DriverError(pub cuda_bindings::CUresult);

impl DriverError {
    /// Returns true when the driver cannot JIT the PTX version in a module.
    ///
    /// This usually means the selected CUDA toolkit is newer than the
    /// installed driver. PTX requires direct driver support even when other
    /// parts of the toolkit can use CUDA minor-version compatibility.
    pub fn is_unsupported_ptx_version(&self) -> bool {
        self.0 == cuda_bindings::cudaError_enum_CUDA_ERROR_UNSUPPORTED_PTX_VERSION
    }

    fn _fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        self.fmt_with_loader_error(formatter, cuda_bindings::cuda_driver_load_error())
    }

    /// Reports a driver library that could not be loaded when the code says
    /// so. Takes the loader's cached failure as a parameter so the keying can
    /// be unit-tested on a machine where the driver loads fine.
    ///
    /// Every driver entry point goes through a loader shim that returns
    /// `CUDA_ERROR_NOT_INITIALIZED` when libcuda could not be loaded at all;
    /// `CUDA_ERROR_SHARED_OBJECT_INIT_FAILED` is what the driver itself (and
    /// an earlier shim) reports for the same condition. Both codes are
    /// otherwise opaque, while the loader's message names the library
    /// candidates tried and why they failed, which is the actionable part.
    fn fmt_with_loader_error(
        &self,
        formatter: &mut Formatter,
        load_error: Option<&cuda_bindings::DynLoadError>,
    ) -> fmt::Result {
        if let Some(load_error) = load_error {
            if self.0 == cuda_bindings::cudaError_enum_CUDA_ERROR_NOT_INITIALIZED
                || self.0 == cuda_bindings::cudaError_enum_CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
            {
                return formatter
                    .debug_tuple("DriverError")
                    .field(&self.0)
                    .field(&format!("CUDA driver library unavailable: {load_error}"))
                    .finish();
            }
        }

        let help = "the CUDA driver cannot JIT PTX from the selected toolkit; upgrade the driver \
                    or select a compatible toolkit with CUDA_TOOLKIT_PATH or CUDA_HOME";

        let mut output = formatter.debug_tuple("DriverError");
        output.field(&self.0);
        match self.error_string() {
            Ok(err_str) => {
                output.field(&err_str);
            }
            Err(_) => {
                output.field(&"<Failure when calling cuGetErrorString()>");
            }
        }
        if self.is_unsupported_ptx_version() {
            output.field(&help);
        }
        output.finish()
    }
}

impl Display for DriverError {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        self._fmt(formatter)
    }
}

impl std::fmt::Debug for DriverError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> fmt::Result {
        self._fmt(formatter)
    }
}

impl error::Error for DriverError {}

/// Converts a CUDA driver call return value into a `Result`.
pub trait IntoResult<T> {
    /// Returns `Ok` on `CUDA_SUCCESS`, or `Err(DriverError)` otherwise.
    fn result(self) -> Result<T, DriverError>
    where
        Self: Sized;
}

impl IntoResult<()> for cuda_bindings::CUresult {
    fn result(self) -> Result<(), DriverError> {
        match self {
            cuda_bindings::cudaError_enum_CUDA_SUCCESS => Ok(()),
            _ => Err(DriverError(self)),
        }
    }
}

impl<T> IntoResult<T> for (cuda_bindings::CUresult, T) {
    fn result(self) -> Result<T, DriverError> {
        match self.0 {
            cuda_bindings::cudaError_enum_CUDA_SUCCESS => Ok(self.1),
            _ => Err(DriverError(self.0)),
        }
    }
}

impl<T> IntoResult<T> for (cuda_bindings::CUresult, MaybeUninit<T>) {
    fn result(self) -> Result<T, DriverError> {
        match self.0 {
            cuda_bindings::cudaError_enum_CUDA_SUCCESS => Ok(unsafe { self.1.assume_init() }),
            _ => Err(DriverError(self.0)),
        }
    }
}

impl DriverError {
    /// Returns the short error name string for this CUDA error code.
    pub fn error_name(&self) -> Result<&CStr, DriverError> {
        let mut err_str = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuGetErrorName(self.0, err_str.as_mut_ptr()).result()?;
            Ok(CStr::from_ptr(err_str.assume_init()))
        }
    }

    /// Returns the human-readable description string for this CUDA error code.
    pub fn error_string(&self) -> Result<&CStr, DriverError> {
        let mut err_str = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuGetErrorString(self.0, err_str.as_mut_ptr()).result()?;
            Ok(CStr::from_ptr(err_str.assume_init()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DriverError;

    #[test]
    fn identifies_unsupported_ptx_version() {
        let unsupported =
            DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_UNSUPPORTED_PTX_VERSION);
        let unrelated = DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE);

        assert!(unsupported.is_unsupported_ptx_version());
        assert!(!unrelated.is_unsupported_ptx_version());
    }

    const NOT_INITIALIZED: cuda_bindings::CUresult =
        cuda_bindings::cudaError_enum_CUDA_ERROR_NOT_INITIALIZED;
    const SHARED_OBJECT_INIT_FAILED: cuda_bindings::CUresult =
        cuda_bindings::cudaError_enum_CUDA_ERROR_SHARED_OBJECT_INIT_FAILED;

    /// Renders `error` as `Display` would, but with an explicit loader state.
    fn format_with(error: DriverError, load_error: Option<&cuda_bindings::DynLoadError>) -> String {
        struct WithLoader<'a>(DriverError, Option<&'a cuda_bindings::DynLoadError>);
        impl std::fmt::Display for WithLoader<'_> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.0.fmt_with_loader_error(f, self.1)
            }
        }
        WithLoader(error, load_error).to_string()
    }

    /// The loader shims report an unloadable libcuda as `NOT_INITIALIZED`
    /// (3), so the hint must key on that code as well as on the driver's own
    /// `SHARED_OBJECT_INIT_FAILED` (303), and on nothing else. Runs the real
    /// formatting path with a synthetic loader failure, so it holds on a
    /// machine where the driver loads.
    #[test]
    fn loader_hint_is_attached_to_both_unavailable_driver_codes() {
        let load_error = cuda_bindings::DynLoadError::RuntimeTooOld {
            compile_version: 13030,
            runtime_version: 12040,
        };

        for code in [NOT_INITIALIZED, SHARED_OBJECT_INIT_FAILED] {
            let formatted = format_with(DriverError(code), Some(&load_error));
            assert!(
                formatted.contains("CUDA driver library unavailable"),
                "code {code}: expected the loader hint, got: {formatted}"
            );
            assert!(
                formatted.contains("CUDA driver too old"),
                "code {code}: expected the loader's own message, got: {formatted}"
            );
        }

        // Any other code keeps the plain rendering even while the loader has failed.
        let unrelated = format_with(
            DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE),
            Some(&load_error),
        );
        assert!(
            !unrelated.contains("CUDA driver library unavailable"),
            "an unrelated code must not carry the loader hint, got: {unrelated}"
        );

        // And with a loaded driver the two codes render plainly too.
        for code in [NOT_INITIALIZED, SHARED_OBJECT_INIT_FAILED] {
            let formatted = format_with(DriverError(code), None);
            assert!(
                !formatted.contains("CUDA driver library unavailable"),
                "code {code}: no loader failure, no hint; got: {formatted}"
            );
        }
    }

    /// The end-to-end path, exercised only where the driver really is
    /// unavailable: `Display` picks up the loader's cached failure itself.
    #[test]
    fn display_surfaces_the_real_loader_error_when_driver_is_missing() {
        let Some(load_error) = cuda_bindings::cuda_driver_load_error() else {
            return;
        };
        let expected_detail = match load_error {
            cuda_bindings::DynLoadError::LoadFailed { .. } => "failed to load any of",
            cuda_bindings::DynLoadError::RuntimeTooOld { .. } => "CUDA driver too old",
        };

        for code in [NOT_INITIALIZED, SHARED_OBJECT_INIT_FAILED] {
            let formatted = DriverError(code).to_string();
            assert!(
                formatted.contains("CUDA driver library unavailable"),
                "code {code}: expected a human-readable loader hint, got: {formatted}"
            );
            assert!(
                formatted.contains(expected_detail),
                "code {code}: expected the cached loader failure context, got: {formatted}"
            );
        }
    }
}
