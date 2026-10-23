/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared allocation ownership and typed device addresses.

use cuda_core::sys::CUdeviceptr;
pub use cuda_core::{DeviceAllocation, RawDeviceBuffer as DeviceBuffer};
use std::marker::PhantomData;

/// A non-owning, copyable handle to a typed device memory address.
#[derive(Debug)]
#[repr(transparent)]
pub struct DevicePointer<T> {
    dtype: PhantomData<fn() -> T>,
    pub dptr: CUdeviceptr,
}

impl<T> Copy for DevicePointer<T> {}
impl<T> Clone for DevicePointer<T> {
    fn clone(&self) -> Self {
        *self
    }
}
// SAFETY: this transparent u64 address owns no memory; all address bit patterns,
// including zero, are valid wrapper values. Dereferencing remains unsafe.
unsafe impl<T> cuda_core::DeviceCopy for DevicePointer<T> {}

impl<T> DevicePointer<T> {
    /// Returns the raw CUDA device pointer.
    pub fn cu_deviceptr(&self) -> CUdeviceptr {
        self.dptr
    }

    /// Constructs a typed device pointer from a raw CUDA device pointer.
    ///
    /// # Safety
    /// The caller must ensure `dptr` is valid for values of type `T`.
    pub unsafe fn from_cu_deviceptr(dptr: CUdeviceptr) -> Self {
        Self {
            dtype: PhantomData,
            dptr,
        }
    }
}
