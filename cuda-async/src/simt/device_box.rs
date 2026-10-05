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

use crate::simt::device_context::with_deallocator_stream;
use crate::simt::error::DeviceError;
use crate::simt::launch::{AsyncKernelLaunchBuilder, KernelArgument};
use cuda_bindings::CUdeviceptr;
use cuda_core::simt::memory::free_async;
use std::io::{self, Write};
use std::marker::PhantomData;

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

/// Owning wrapper around a device allocation that frees memory asynchronously
/// on drop.
///
/// `DeviceBox<[DType]>` represents a contiguous device buffer of `len`
/// elements. The allocation is freed via `cuMemFreeAsync` on the device's
/// deallocator stream when the box is dropped.
///
/// # Safety contract
///
/// All streams that reference this allocation **must** be synchronized before
/// the box is dropped. Violating this causes use-after-free on the device.
#[derive(Debug)]
pub struct DeviceBox<T: Send + ?Sized> {
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

/// Best-effort enqueue of an asynchronous free on the device's deallocator
/// stream.
///
/// Drop cannot return errors, so failures to access the deallocator stream or
/// enqueue the free are reported to stderr instead of panicking.
impl<T: Send + ?Sized> Drop for DeviceBox<T> {
    fn drop(&mut self) {
        let result = unsafe {
            with_deallocator_stream(self.device_id, |stream| {
                free_async(self.cudptr, stream.cu_stream()).map_err(DeviceError::Driver)
            })
        };

        match result {
            Ok(Ok(())) => {}
            Ok(Err(err)) | Err(err) => {
                let mut stderr = io::stderr().lock();
                let _ = writeln!(
                    stderr,
                    "cuda-async: failed to enqueue async free for device pointer on device_id={}: {}",
                    self.device_id, err
                );
            }
        }
    }
}

impl<DType: Send + Sized> DeviceBox<[DType]> {
    /// Constructs a `DeviceBox<[DType]>` from a raw device pointer, element
    /// count, and device ordinal.
    ///
    /// # Safety
    ///
    /// * `dptr` must point to a valid device allocation of at least
    ///   `len * size_of::<DType>()` bytes on device `device_id`.
    /// * Ownership of the allocation transfers to the returned `DeviceBox`.
    pub unsafe fn from_raw_parts(dptr: CUdeviceptr, len: usize, device_id: usize) -> Self {
        Self {
            _marker: PhantomData,
            cudptr: dptr,
            len,
            device_id,
        }
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
