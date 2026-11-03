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

use crate::context::Context;
use crate::device_buffer::DeviceCopy;
use crate::error::DriverError;

/// Owned page-locked host buffer.
///
/// The buffer is initialized and exposes normal Rust slices. The backing
/// allocation is freed with `cuMemFreeHost` on drop. Extra space is reserved
/// when needed to align the elements, even for types whose alignment exceeds
/// CUDA's host-allocation alignment.
///
/// Pinned transfer buffers require `T: DeviceCopy`, so host-owned values such
/// as [`String`] are rejected.
///
/// ```compile_fail
/// # use cuda_core::{Context, PinnedHostBuffer};
/// # fn rejects_non_device_copy(ctx: &std::sync::Arc<Context>) {
/// let _ = PinnedHostBuffer::<String>::zeroed(ctx, 1);
/// # }
/// ```
pub struct PinnedHostBuffer<T: DeviceCopy> {
    ptr: NonNull<T>,
    // CUDA must free the original allocation, not the aligned interior pointer.
    allocation: Option<NonNull<u8>>,
    len: usize,
    num_bytes: usize,
    ctx: Arc<Context>,
    _marker: PhantomData<T>,
}

// SAFETY: the allocation is host memory. Moving ownership to another thread is
// safe when `T` is safe to send.
unsafe impl<T: DeviceCopy + Send> Send for PinnedHostBuffer<T> {}
// SAFETY: shared access only exposes `&[T]`.
unsafe impl<T: DeviceCopy + Sync> Sync for PinnedHostBuffer<T> {}

impl<T: DeviceCopy> PinnedHostBuffer<T> {
    /// Allocates a pinned host buffer and fills it with zero bytes.
    pub fn zeroed(ctx: &Arc<Context>, len: usize) -> Result<Self, DriverError> {
        let mut buffer = Self::allocate(ctx, len)?;
        if buffer.num_bytes != 0 {
            unsafe {
                std::ptr::write_bytes(buffer.as_mut_ptr().cast::<u8>(), 0, buffer.num_bytes);
            }
        }

        Ok(buffer)
    }

    /// Allocates a pinned host buffer and copies `data` into it.
    pub fn from_slice(ctx: &Arc<Context>, data: &[T]) -> Result<Self, DriverError> {
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
    pub fn context(&self) -> &Arc<Context> {
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

    fn allocate(ctx: &Arc<Context>, len: usize) -> Result<Self, DriverError> {
        let (num_bytes, allocation_bytes) = allocation_sizes::<T>(len)?;
        let (ptr, allocation) = if num_bytes == 0 {
            (NonNull::dangling(), None)
        } else {
            ctx.bind_to_thread()?;
            let raw = unsafe { crate::memory::malloc_host(allocation_bytes)? };
            let allocation = NonNull::new(raw.cast::<u8>()).ok_or(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE,
            ))?;
            let offset = allocation.as_ptr().align_offset(std::mem::align_of::<T>());
            // SAFETY: for a byte pointer, align_offset is at most align_of::<T>()
            // - 1. allocation_sizes reserves that much slack, leaving num_bytes
            // accessible after this aligned, non-null interior pointer. The
            // entire allocation fits in isize::MAX for pointer arithmetic.
            let ptr =
                unsafe { NonNull::new_unchecked(allocation.as_ptr().add(offset).cast::<T>()) };
            (ptr, Some(allocation))
        };

        Ok(Self {
            ptr,
            allocation,
            len,
            num_bytes,
            ctx: ctx.clone(),
            _marker: PhantomData,
        })
    }
}

impl<T: DeviceCopy> Drop for PinnedHostBuffer<T> {
    fn drop(&mut self) {
        if let Some(allocation) = self.allocation {
            self.ctx.record_err(self.ctx.bind_to_thread());
            self.ctx
                .record_err(unsafe { crate::memory::free_host(allocation.as_ptr().cast()) });
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

fn allocation_sizes<T>(len: usize) -> Result<(usize, usize), DriverError> {
    let invalid_size = DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE);
    // Rust slices and pointer arithmetic require sizes bounded by isize::MAX,
    // even when len * size_of::<T>() would fit in usize.
    let layout = std::alloc::Layout::array::<T>(len).map_err(|_| invalid_size)?;
    let num_bytes = layout.size();
    if num_bytes == 0 {
        return Ok((0, 0));
    }
    let allocation_bytes = num_bytes
        .checked_add(layout.align() - 1)
        .filter(|&size| size <= isize::MAX as usize)
        .ok_or(invalid_size)?;
    Ok((num_bytes, allocation_bytes))
}

#[cfg(test)]
mod tests {
    use super::allocation_sizes;

    #[test]
    fn rejects_sizes_that_cannot_form_a_rust_slice() {
        assert!(allocation_sizes::<u8>(isize::MAX as usize + 1).is_err());
        assert!(allocation_sizes::<u32>(usize::MAX).is_err());
        assert_eq!(
            allocation_sizes::<u8>(isize::MAX as usize).unwrap(),
            (isize::MAX as usize, isize::MAX as usize)
        );
    }

    #[test]
    fn empty_and_zero_sized_buffers_need_no_allocation() {
        assert_eq!(allocation_sizes::<u64>(0).unwrap(), (0, 0));
        assert_eq!(allocation_sizes::<()>(usize::MAX).unwrap(), (0, 0));
    }
}
