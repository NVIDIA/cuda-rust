/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Page-locked host memory for CUDA transfers.
//!
//! [`PinnedHostBuffer<T>`] owns CUDA page-locked host memory allocated with
//! `cuMemAllocHost`. Pinned memory is useful as a staging area for host-device
//! copies that need higher transfer bandwidth. Copies only overlap with GPU
//! work when they are enqueued with non-blocking transfer APIs and synchronized
//! later by the caller.

use std::fmt;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::slice;
use std::sync::Arc;

use crate::error::DriverError;
use crate::simt::context::CudaContext;
use crate::simt::device_buffer::DeviceCopy;

/// Owned page-locked host buffer.
///
/// The buffer is initialized and exposes normal Rust slices. The backing
/// allocation is freed with `cuMemFreeHost` on drop.
///
/// Pinned transfer buffers require `T: DeviceCopy`, so host-owned values such
/// as [`String`] are rejected.
///
/// ```compile_fail
/// # use cuda_core::{CudaContext, PinnedHostBuffer};
/// # fn rejects_non_device_copy(ctx: &std::sync::Arc<CudaContext>) {
/// let _ = PinnedHostBuffer::<String>::zeroed(ctx, 1);
/// # }
/// ```
pub struct PinnedHostBuffer<T: DeviceCopy> {
    ptr: NonNull<T>,
    len: usize,
    num_bytes: usize,
    ctx: Arc<CudaContext>,
    _marker: PhantomData<T>,
}

// SAFETY: the allocation is host memory. Moving ownership to another thread is
// safe when `T` is safe to send.
unsafe impl<T: DeviceCopy + Send> Send for PinnedHostBuffer<T> {}
// SAFETY: shared access only exposes `&[T]`.
unsafe impl<T: DeviceCopy + Sync> Sync for PinnedHostBuffer<T> {}

impl<T: DeviceCopy> PinnedHostBuffer<T> {
    /// Allocates a pinned host buffer and fills it with zero bytes.
    pub fn zeroed(ctx: &Arc<CudaContext>, len: usize) -> Result<Self, DriverError> {
        let mut buffer = Self::allocate(ctx, len)?;
        if buffer.num_bytes != 0 {
            unsafe {
                std::ptr::write_bytes(buffer.as_mut_ptr().cast::<u8>(), 0, buffer.num_bytes);
            }
        }

        Ok(buffer)
    }

    /// Allocates a pinned host buffer and copies `data` into it.
    pub fn from_slice(ctx: &Arc<CudaContext>, data: &[T]) -> Result<Self, DriverError> {
        let buffer = Self::allocate(ctx, data.len())?;
        if !data.is_empty() {
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), buffer.ptr.as_ptr(), data.len());
            }
        }
        Ok(buffer)
    }

    /// Number of elements in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the buffer contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total size in bytes (`len * size_of::<T>()`).
    #[inline]
    pub fn num_bytes(&self) -> usize {
        self.num_bytes
    }

    /// Returns the CUDA context used to allocate this buffer.
    #[inline]
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Returns the host pointer.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Returns the mutable host pointer.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Returns the buffer as a host slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Returns the buffer as a mutable host slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    fn allocate(ctx: &Arc<CudaContext>, len: usize) -> Result<Self, DriverError> {
        let num_bytes = allocation_size::<T>(len)?;
        let ptr = if num_bytes == 0 {
            NonNull::dangling()
        } else {
            ctx.bind_to_thread()?;
            let ptr = unsafe { crate::simt::memory::malloc_host(num_bytes)? };
            // `cuMemAllocHost` returns page-aligned memory in practice, but
            // the slices handed out below assume `align_of::<T>()`, so check
            // rather than trust: a misaligned `&[T]` is undefined behavior.
            if ptr.is_null() || !(ptr as usize).is_multiple_of(std::mem::align_of::<T>()) {
                if !ptr.is_null() {
                    // SAFETY: `ptr` came from `malloc_host` above and has not
                    // been handed out.
                    ctx.record_err(unsafe { crate::simt::memory::free_host(ptr) });
                }
                return Err(DriverError(
                    cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE,
                ));
            }
            // SAFETY: checked non-null just above.
            unsafe { NonNull::new_unchecked(ptr.cast::<T>()) }
        };

        Ok(Self {
            ptr,
            len,
            num_bytes,
            ctx: ctx.clone(),
            _marker: PhantomData,
        })
    }
}

impl<T: DeviceCopy> Drop for PinnedHostBuffer<T> {
    fn drop(&mut self) {
        if self.num_bytes != 0 {
            // Not `bind_to_thread`: a recorded error must not skip the free.
            self.ctx.record_err(self.ctx.make_current());
            self.ctx
                .record_err(unsafe { crate::simt::memory::free_host(self.ptr.as_ptr().cast()) });
        }
    }
}

impl<T: DeviceCopy> fmt::Debug for PinnedHostBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PinnedHostBuffer")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .field("num_bytes", &self.num_bytes)
            .field("ctx", &self.ctx)
            .finish()
    }
}

impl<T: DeviceCopy> AsRef<[T]> for PinnedHostBuffer<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: DeviceCopy> AsMut<[T]> for PinnedHostBuffer<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: DeviceCopy> Deref for PinnedHostBuffer<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: DeviceCopy> DerefMut for PinnedHostBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

fn allocation_size<T>(len: usize) -> Result<usize, DriverError> {
    len.checked_mul(std::mem::size_of::<T>()).ok_or(DriverError(
        cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE,
    ))
}
