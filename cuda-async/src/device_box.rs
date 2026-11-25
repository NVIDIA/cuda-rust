/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Owned and borrowed wrappers around CUDA device pointers.
//!
//! [`DevicePointer`] is a thin, `Copy` handle to device memory suitable for
//! passing as a kernel argument. [`DeviceBox`] owns device memory and frees it
//! through the shared allocation owner without waiting on drop. Generated
//! launches retain that owner and its access leases until GPU completion.
//! Raw-pointer callers must register storage with `ExecutionContext::retain_buffer`
//! before enqueueing work, or establish completion themselves.

pub use crate::device_buffer::DevicePointer;
use cuda_bindings::CUdeviceptr;
use std::{marker::PhantomData, sync::Arc};

/// Owning wrapper backed by the shared CUDA allocation owner and access tracker.
/// Generated launch adapters retain storage until GPU completion. Raw-pointer
/// callers must independently retain storage and exclude conflicting accesses.
#[derive(Debug)]
pub struct DeviceBox<T: Send + ?Sized> {
    storage: cuda_core::RawDeviceBuffer,
    shared_access_writes: bool,
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
    ) -> Self
    where
        DType: Copy,
    {
        let bytes = len
            .checked_mul(std::mem::size_of::<DType>())
            .expect("device allocation size overflow");
        let device_id = stream.context().ordinal();
        let storage =
            unsafe { cuda_core::RawDeviceBuffer::from_raw_parts_async(dptr, bytes, stream) };
        Self {
            storage,
            shared_access_writes: false,
            _marker: PhantomData,
            cudptr: dptr,
            len,
            device_id,
        }
    }

    /// Whether a shared reference may mutate through interior mutability.
    #[doc(hidden)]
    pub fn shared_access_writes(&self) -> bool {
        self.shared_access_writes
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
        // SAFETY: this box retains the allocation described by its pointer.
        unsafe { DevicePointer::from_cu_deviceptr(self.cudptr) }
    }
}

impl<T: cuda_core::DeviceCopy + Send> DeviceBox<[T]> {
    /// Transfers a typed buffer without copying device memory. Storage identity,
    /// context, allocation extent and access tracking remain unchanged.
    pub fn from_buffer(buffer: cuda_core::DeviceBuffer<T>) -> Self {
        Self {
            cudptr: buffer.cu_deviceptr(),
            len: buffer.len(),
            device_id: buffer.context().ordinal(),
            shared_access_writes: buffer.shared_access_writes(),
            storage: buffer.into_storage(),
            _marker: PhantomData,
        }
    }

    /// Transfers ownership back to the shared typed buffer without a GPU copy.
    pub fn into_buffer(self) -> cuda_core::DeviceBuffer<T> {
        // SAFETY: this box owns valid T elements; consuming it transfers that
        // ownership and the same access tracker without creating a typed alias.
        unsafe { cuda_core::DeviceBuffer::from_storage_copy(self.storage, self.len) }
            .expect("DeviceBox storage must cover its elements")
    }
}
