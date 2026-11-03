// SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime loading policy for CUDA driver and cuRAND.
//!
//! Bindgen-generated wrappers own the ABI surface and function signatures.
//! This module only chooses library names, caches the generated wrappers, and
//! exposes the stable flat free-function API used throughout the workspace.

use std::sync::OnceLock;

#[allow(unused_imports)]
use crate::*;

type GeneratedCudaDriverApi = crate::generated_cuda::CudaDriverApi;
type GeneratedCurandApi = crate::generated_curand::CurandApi;

#[derive(Debug)]
pub enum DynLoadError {
    LoadFailed {
        names: &'static [&'static str],
        source: libloading::Error,
    },
    RuntimeTooOld {
        compile_version: u32,
        runtime_version: u32,
    },
}

impl std::fmt::Display for DynLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DynLoadError::LoadFailed { names, source } => {
                write!(f, "failed to load any of {names:?}: {source}")
            }
            DynLoadError::RuntimeTooOld {
                compile_version,
                runtime_version,
            } => {
                write!(
                    f,
                    "CUDA driver too old: built against {}.{} but runtime is {}.{}",
                    compile_version / 1000,
                    (compile_version % 1000) / 10,
                    runtime_version / 1000,
                    (runtime_version % 1000) / 10,
                )
            }
        }
    }
}

impl std::error::Error for DynLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DynLoadError::LoadFailed { source, .. } => Some(source),
            DynLoadError::RuntimeTooOld { .. } => None,
        }
    }
}

#[cfg(target_os = "linux")]
const CUDA_LIB_NAMES: &[&str] = &[
    "libcuda.so.1",
    "/usr/lib/wsl/lib/libcuda.so.1",
    "libcuda.so",
    "/usr/lib/wsl/lib/libcuda.so",
    "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
];
#[cfg(target_os = "macos")]
const CUDA_LIB_NAMES: &[&str] = &["libcuda.dylib"];
#[cfg(target_os = "windows")]
const CUDA_LIB_NAMES: &[&str] = &["nvcuda.dll"];
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
const CUDA_LIB_NAMES: &[&str] = &["libcuda.so"];

#[cfg(target_os = "linux")]
const CURAND_LIB_NAMES: &[&str] = &["libcurand.so.10", "libcurand.so"];
#[cfg(target_os = "macos")]
const CURAND_LIB_NAMES: &[&str] = &["libcurand.dylib"];
#[cfg(target_os = "windows")]
const CURAND_LIB_NAMES: &[&str] = &["curand64_10.dll"];
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
const CURAND_LIB_NAMES: &[&str] = &["libcurand.so"];

// The generated shims are `extern "C"`, so nothing reachable from them may
// panic (unwinding out of extern "C" aborts). `load_api`'s expect below is
// reachable from every shim; these make its precondition a compile-time
// fact rather than a runtime hope.
const _: () = assert!(!CUDA_LIB_NAMES.is_empty());
const _: () = assert!(!CURAND_LIB_NAMES.is_empty());

trait GeneratedApi: Sized {
    unsafe fn open(path: &str) -> Result<Self, libloading::Error>;
}

impl GeneratedApi for GeneratedCudaDriverApi {
    unsafe fn open(path: &str) -> Result<Self, libloading::Error> {
        unsafe { Self::new(path) }
    }
}

impl GeneratedApi for GeneratedCurandApi {
    unsafe fn open(path: &str) -> Result<Self, libloading::Error> {
        unsafe { Self::new(path) }
    }
}

fn load_api<T: GeneratedApi>(names: &'static [&'static str]) -> Result<T, DynLoadError> {
    let mut last_error = None;
    for &name in names {
        match unsafe { T::open(name) } {
            Ok(api) => return Ok(api),
            Err(error) => last_error = Some(error),
        }
    }

    Err(DynLoadError::LoadFailed {
        names,
        source: last_error.expect("library candidate lists must be non-empty"),
    })
}

fn cached_api<T: GeneratedApi>(
    slot: &'static OnceLock<Result<T, DynLoadError>>,
    names: &'static [&'static str],
) -> Result<&'static T, &'static DynLoadError> {
    slot.get_or_init(|| load_api::<T>(names)).as_ref()
}

static CUDA_DRIVER: OnceLock<Result<GeneratedCudaDriverApi, DynLoadError>> = OnceLock::new();
static CURAND: OnceLock<Result<GeneratedCurandApi, DynLoadError>> = OnceLock::new();

fn load_and_verify_cuda_driver() -> Result<GeneratedCudaDriverApi, DynLoadError> {
    let api = load_api::<GeneratedCudaDriverApi>(CUDA_LIB_NAMES)?;

    // Since CUDA 11, minor-version compatibility allows newer toolkits on
    // older same-major drivers, so only reject on a major-version mismatch.
    // https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html
    if let Ok(get_version) = &api.cuDriverGetVersion {
        let mut runtime_version: std::ffi::c_int = 0;
        if unsafe { get_version(&mut runtime_version) } == 0 {
            let compile_major = crate::CUDA_VERSION / 1000;
            let runtime_major = (runtime_version as u32) / 1000;
            if runtime_major < compile_major {
                return Err(DynLoadError::RuntimeTooOld {
                    compile_version: crate::CUDA_VERSION,
                    runtime_version: runtime_version as u32,
                });
            }
        }
    }

    Ok(api)
}

fn cuda_driver() -> Result<&'static GeneratedCudaDriverApi, &'static DynLoadError> {
    CUDA_DRIVER
        .get_or_init(load_and_verify_cuda_driver)
        .as_ref()
}

fn curand_api() -> Result<&'static GeneratedCurandApi, &'static DynLoadError> {
    cached_api(&CURAND, CURAND_LIB_NAMES)
}

pub fn is_cuda_driver_available() -> bool {
    cuda_driver().is_ok()
}

pub fn cuda_driver_load_error() -> Option<&'static DynLoadError> {
    cuda_driver().err()
}

pub fn is_curand_available() -> bool {
    curand_api().is_ok()
}

pub fn curand_load_error() -> Option<&'static DynLoadError> {
    curand_api().err()
}

include!(concat!(env!("OUT_DIR"), "/cuda_driver_shims.rs"));
include!(concat!(env!("OUT_DIR"), "/curand_shims.rs"));
