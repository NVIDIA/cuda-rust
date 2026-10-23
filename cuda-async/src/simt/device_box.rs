/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Owned and borrowed wrappers around CUDA device pointers.
//!
//! [`DevicePointer`] is a thin, `Copy` handle to device memory suitable for
//! passing as a kernel argument. [`DeviceBox`] owns device memory and frees it
//! asynchronously through a dedicated deallocator stream on drop.
//!
//! # Async deallocation
//!
//! [`DeviceBox`] enqueues `cuMemFreeAsync` on a per-device deallocator stream
//! rather than blocking with `cuMemFree`. Callers **must** synchronize all
//! streams that reference the allocation before dropping the box.

use crate::simt::launch::{AsyncKernelLaunchBuilder, KernelArgument};
use cuda_bindings::CUdeviceptr;
use std::{marker::PhantomData, sync::Arc};

/// Non-owning, `Copy` handle to a typed device allocation.
///
/// Does not manage lifetime -- the underlying memory must outlive all uses.
#[derive(Debug, Copy, Clone)]
pub struct DevicePointer<T> {
    /// Ties the pointer to element type `T` without owning one.
    _marker: PhantomData<T>,
    /// Raw CUDA device pointer.
    pub dptr: CUdeviceptr,
}

/// # Safety
///
/// `DevicePointer` is a plain integer handle with no thread-affine state.
/// Sending it across threads does not violate CUDA's driver API contract.
unsafe impl<T> Send for DevicePointer<T> {}

impl<T> DevicePointer<T> {
    /// Returns the raw [`CUdeviceptr`].
    pub fn cu_deviceptr(&self) -> CUdeviceptr {
        self.dptr
    }
}

/// Pushes the underlying device address as a kernel argument.
impl<T: Send + Sized> KernelArgument for DevicePointer<T> {
    fn push_arg(self, launcher: &mut AsyncKernelLaunchBuilder<'_>) {
        launcher.push_arg(self.cu_deviceptr());
    }
}

/// Owning wrapper backed by the shared CUDA allocation owner and access tracker.
/// Generated launch adapters retain storage until GPU completion. Raw-pointer
/// callers must independently retain storage and exclude conflicting accesses.
#[derive(Debug)]
pub struct DeviceBox<T: Send + ?Sized> {
    storage: cuda_core::RawDeviceBuffer,
    /// Ordinal of the device that owns this allocation.
    device_id: usize,
    /// Raw CUDA device pointer to the start of the allocation.
    cudptr: CUdeviceptr,
    /// Number of elements (not bytes) in the allocation.
    len: usize,
    /// Ties the box to the element type.
    _marker: PhantomData<T>,
}

/// # Safety
///
/// Device pointers are plain integers. The CUDA driver API is thread-safe for
/// distinct streams, and `DeviceBox` holds no thread-affine state.
unsafe impl<T: Send + ?Sized> Send for DeviceBox<T> {}

/// # Safety
///
/// See the [`Send`] impl. Shared references only expose the device pointer
/// value, which is safe to read from any thread.
unsafe impl<T: Send + ?Sized> Sync for DeviceBox<T> {}

impl<DType: Send + Sized> DeviceBox<[DType]> {
    /// Constructs a `DeviceBox<[DType]>` from a raw device pointer, element
    /// count, and allocating stream.
    ///
    /// # Safety
    ///
    /// * `dptr` must point to a valid device allocation of at least
    ///   `len * size_of::<DType>()` bytes, allocated asynchronously on `stream`.
    /// * Every element must be valid for its device-side type before use.
    /// * The caller must order uses after allocation and track all raw accesses.
    /// * Ownership of the allocation transfers to the returned `DeviceBox`.
    pub unsafe fn from_raw_parts(
        dptr: CUdeviceptr,
        len: usize,
        stream: Arc<cuda_core::Stream>,
    ) -> Self {
        let bytes = len
            .checked_mul(std::mem::size_of::<DType>())
            .expect("device allocation size overflow");
        let device_id = stream.context().ordinal();
        let storage =
            unsafe { cuda_core::RawDeviceBuffer::from_raw_parts_async(dptr, bytes, stream) };
        Self {
            storage,
            _marker: PhantomData,
            cudptr: dptr,
            len,
            device_id,
        }
    }

    pub fn storage(&self) -> &cuda_core::RawDeviceBuffer {
        &self.storage
    }

    /// Returns `true` if the buffer contains zero elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of elements in the buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the raw [`CUdeviceptr`].
    pub fn cu_deviceptr(&self) -> CUdeviceptr {
        self.cudptr
    }

    /// Returns the ordinal of the device that owns this allocation.
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// Creates a non-owning [`DevicePointer`] into this allocation.
    pub fn device_pointer(&self) -> DevicePointer<DType> {
        DevicePointer {
            _marker: PhantomData,
            dptr: self.cudptr,
        }
    }
}
