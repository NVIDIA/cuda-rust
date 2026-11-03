/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA device memory allocation and transfer operations.
//!
//! Provides both **stream-ordered** (`*_async`) and **synchronous** (`*_sync`)
//! variants. The async functions enqueue the operation on a stream and return
//! immediately; the transfer completes in stream order. The sync variants block
//! the calling thread until the operation finishes.
//!
//! All functions in this module are `unsafe` because they operate on raw device
//! pointers whose validity cannot be checked at compile time.

use crate::error::{DriverError, IntoResult};
use cuda_bindings::CUdeviceptr;
use std::ffi::c_void;
use std::mem::MaybeUninit;

/// Allocates `num_bytes` of device memory on `stream` using the stream-ordered
/// allocator (`cuMemAllocAsync`).
///
/// The returned pointer is usable by any kernel or memcpy enqueued on `stream`
/// after this call. Pair with [`free_async`] on a stream ordered after this
/// allocation and every use of the pointer.
///
/// # Safety
///
/// - A CUDA context must be bound to the calling thread.
/// - `stream` must be a valid `CUstream` from the current context.
/// - `num_bytes` must not exceed the device memory pool limits.
pub unsafe fn malloc_async(
    stream: cuda_bindings::CUstream,
    num_bytes: usize,
) -> Result<CUdeviceptr, DriverError> {
    let mut dev_ptr = MaybeUninit::uninit();
    unsafe {
        cuda_bindings::cuMemAllocAsync(dev_ptr.as_mut_ptr(), num_bytes, stream).result()?;
        Ok(dev_ptr.assume_init())
    }
}

/// Frees device memory previously allocated with [`malloc_async`] or
/// [`malloc_sync`].
///
/// The free is enqueued on `stream` and completes in stream order. The pointer
/// must not be accessed by any work enqueued after this call. If the pointer
/// came from [`malloc_async`], `stream` must be ordered after its allocation
/// stream. Every other stream using the pointer must also be ordered before
/// `stream`.
///
/// # Safety
///
/// - `dptr` must have been returned by [`malloc_async`] or [`malloc_sync`] and
///   not yet freed.
/// - `stream` must be a valid `CUstream` from the same context as the
///   allocation.
/// - All allocation and use work on other streams must happen before the free
///   on `stream`.
pub unsafe fn free_async(
    dptr: CUdeviceptr,
    stream: cuda_bindings::CUstream,
) -> Result<(), DriverError> {
    unsafe { cuda_bindings::cuMemFreeAsync(dptr, stream) }.result()
}

/// Allocates `num_bytes` of device memory synchronously (`cuMemAlloc`).
///
/// Blocks the calling thread until the allocation completes. Pair with
/// [`free_sync`].
///
/// # Safety
///
/// - A CUDA context must be bound to the calling thread.
/// - `num_bytes` must not exceed available device memory.
pub unsafe fn malloc_sync(num_bytes: usize) -> Result<CUdeviceptr, DriverError> {
    let mut dev_ptr = MaybeUninit::uninit();
    unsafe {
        cuda_bindings::cuMemAlloc_v2(dev_ptr.as_mut_ptr(), num_bytes).result()?;
        Ok(dev_ptr.assume_init())
    }
}

/// Frees device memory previously allocated with [`malloc_sync`].
///
/// Blocks the calling thread. All pending GPU work referencing `dptr` must have
/// completed before this call.
///
/// # Safety
///
/// - `dptr` must have been returned by [`malloc_sync`] and not yet freed.
/// - No in-flight GPU operations may reference `dptr`.
pub unsafe fn free_sync(dptr: CUdeviceptr) -> Result<(), DriverError> {
    unsafe { cuda_bindings::cuMemFree_v2(dptr) }.result()
}

/// Copies `num_bytes` from host memory at `src` to device memory at `dst`,
/// enqueued on `stream` (host-to-device, async).
///
/// The host buffer at `src` must remain valid and unmodified until the copy
/// completes (i.e., until a synchronization point on `stream`). For
/// guaranteed asynchronous behavior, `src` should point to page-locked
/// (pinned) host memory.
///
/// # Safety
///
/// - `dst` must be a valid device pointer with at least `num_bytes` allocated.
/// - `src` must point to at least `num_bytes` of readable host memory.
/// - `stream` must be a valid `CUstream` from the current context.
pub unsafe fn memcpy_htod_async<T>(
    dst: CUdeviceptr,
    src: *const T,
    num_bytes: usize,
    stream: cuda_bindings::CUstream,
) -> Result<(), DriverError> {
    unsafe { cuda_bindings::cuMemcpyHtoDAsync_v2(dst, src as *const _, num_bytes, stream) }.result()
}

/// Copies `num_bytes` from host memory at `src` to device memory at `dst`,
/// synchronously (host-to-device).
///
/// Blocks the calling thread until the copy is complete.
///
/// # Safety
///
/// - `dst` must be a valid device pointer with at least `num_bytes` allocated.
/// - `src` must point to at least `num_bytes` of readable host memory.
/// - The active CUDA context must own `dst` (caller binds the context).
pub unsafe fn memcpy_htod_sync<T>(
    dst: CUdeviceptr,
    src: *const T,
    num_bytes: usize,
) -> Result<(), DriverError> {
    unsafe { cuda_bindings::cuMemcpyHtoD_v2(dst, src as *const _, num_bytes) }.result()
}

/// Copies `num_bytes` from device memory at `src` to host memory at `dst`,
/// enqueued on `stream` (device-to-host, async).
///
/// The host buffer at `dst` must not be read until the copy completes.
/// For guaranteed asynchronous behavior, `dst` should point to page-locked
/// (pinned) host memory.
///
/// # Safety
///
/// - `src` must be a valid device pointer with at least `num_bytes` accessible.
/// - `dst` must point to at least `num_bytes` of writable host memory.
/// - `stream` must be a valid `CUstream` from the current context.
pub unsafe fn memcpy_dtoh_async<T>(
    dst: *mut T,
    src: CUdeviceptr,
    num_bytes: usize,
    stream: cuda_bindings::CUstream,
) -> Result<(), DriverError> {
    unsafe { cuda_bindings::cuMemcpyDtoHAsync_v2(dst as *mut _, src, num_bytes, stream) }.result()
}

/// Copies `num_bytes` from device memory at `src` to device memory at `dst`,
/// enqueued on `stream` (device-to-device, async).
///
/// `src` and `dst` may reside on different devices if peer access is enabled.
///
/// # Safety
///
/// - Both `dst` and `src` must be valid device pointers with at least
///   `num_bytes` accessible.
/// - `stream` must be a valid `CUstream` from the current context.
/// - `dst` and `src` must not overlap unless they are identical.
pub unsafe fn memcpy_dtod_async(
    dst: CUdeviceptr,
    src: CUdeviceptr,
    num_bytes: usize,
    stream: cuda_bindings::CUstream,
) -> Result<(), DriverError> {
    unsafe { cuda_bindings::cuMemcpyDtoDAsync_v2(dst, src, num_bytes, stream) }.result()
}

/// Sets `num_bytes` of device memory at `dptr` to `value`, enqueued on
/// `stream`.
///
/// Each byte in the range `[dptr, dptr + num_bytes)` is set to `value`.
///
/// # Safety
///
/// - `dptr` must be a valid device pointer with at least `num_bytes` allocated.
/// - `stream` must be a valid `CUstream` from the current context.
pub unsafe fn memset_d8_async(
    dptr: CUdeviceptr,
    value: u8,
    num_bytes: usize,
    stream: cuda_bindings::CUstream,
) -> Result<(), DriverError> {
    unsafe { cuda_bindings::cuMemsetD8Async(dptr, value, num_bytes, stream) }.result()
}

/// Allocates `num_bytes` of page-locked host memory.
///
/// Pinned host memory can be used as a staging area for CUDA transfers that
/// need higher bandwidth, and is required for host-device copies that are
/// intended to overlap with GPU work. Pair with [`free_host`].
///
/// # Safety
///
/// - A CUDA context must be bound to the calling thread.
/// - `num_bytes` must not exceed the host memory available for page-locked
///   allocations. Passing zero bytes is not useful and the CUDA driver reports
///   it as an error.
pub unsafe fn malloc_host(num_bytes: usize) -> Result<*mut c_void, DriverError> {
    let mut host_ptr = MaybeUninit::uninit();
    unsafe {
        cuda_bindings::cuMemAllocHost_v2(host_ptr.as_mut_ptr(), num_bytes).result()?;
        Ok(host_ptr.assume_init())
    }
}

/// Frees page-locked host memory previously allocated with [`malloc_host`].
///
/// # Safety
///
/// - `ptr` must have been returned by [`malloc_host`] and not yet freed.
/// - No in-flight CUDA transfer or kernel may reference `ptr`.
pub unsafe fn free_host(ptr: *mut c_void) -> Result<(), DriverError> {
    unsafe { cuda_bindings::cuMemFreeHost(ptr) }.result()
}
