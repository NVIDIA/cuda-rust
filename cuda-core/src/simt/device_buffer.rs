/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Owning device memory buffer with ergonomic host-device transfer methods.
//!
//! [`DeviceBuffer<T>`] is analogous to `Vec<T>` on the host: it owns a
//! contiguous allocation of `len` elements on the device and frees it on
//! drop. The stream is an explicit parameter on every transfer operation,
//! making data-flow and synchronization transparent. Buffers allocated with
//! [`DeviceBuffer::uninitialized_async`] retain their allocation stream for
//! deallocation.
//!
//! Ordinary drop synchronizes the context before freeing an asynchronous
//! allocation, because safe operations may have submitted work using the
//! buffer on any stream in that context. The unsafe
//! [`DeviceBuffer::drop_async`] avoids that host-side synchronization by
//! freeing on a chosen stream; its caller takes over the obligation to
//! order every other stream that uses the buffer before that stream.
//!
//! # Quick start
//!
//! ```ignore
//! let a_dev = DeviceBuffer::from_host(&stream, &a_host)?;
//! let c_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;
//! // ... kernel launch ...
//! let c_host = c_dev.to_host_vec(&stream)?;
//! ```

use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::num::Wrapping;
use std::sync::Arc;

use cuda_bindings::CUdeviceptr;

use crate::error::DriverError;
use crate::simt::context::CudaContext;
use crate::simt::pinned_host_buffer::PinnedHostBuffer;
use crate::simt::stream::CudaStream;

/// Marker trait for values that can be safely copied between host and device
/// memory as raw bytes.
///
/// Types implementing `DeviceCopy` must not contain Rust-owned allocations,
/// references, or other values whose validity depends on host-side ownership or
/// drop semantics. This is the device-memory equivalent of a plain-old-data
/// contract.
///
/// # Safety
///
/// Implementors must be safe to duplicate with a byte-for-byte copy. Values
/// copied back from device memory must have a bit pattern that is valid for
/// `Self`, and the all-zero bit pattern must also be valid because
/// [`DeviceBuffer::zeroed`] initializes memory with zero bytes.
///
/// `Copy` alone is not enough: types such as `bool`, `char`, and
/// `NonZeroU32` are `Copy`, but not every byte pattern is a valid value of
/// those types. `DeviceCopy` is the stronger promise required when
/// `DeviceBuffer` turns raw device bytes back into initialized Rust values.
pub unsafe trait DeviceCopy: Copy {}

macro_rules! impl_device_copy {
    ($($ty:ty),+ $(,)?) => {
        $(
            unsafe impl DeviceCopy for $ty {}
        )+
    };
}

impl_device_copy!(
    (),
    i8,
    i16,
    i32,
    i64,
    i128,
    isize,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    f32,
    f64
);

unsafe impl<T: DeviceCopy, const N: usize> DeviceCopy for [T; N] {}
unsafe impl<T: ?Sized> DeviceCopy for *const T {}
unsafe impl<T: ?Sized> DeviceCopy for *mut T {}

// Wrapper types that don't change the byte representation: a value of the
// wrapper has the same layout and validity invariants as the inner `T`.
// `PhantomData<T>` is a zero-sized marker -- always trivially copyable
// regardless of `T`. `MaybeUninit<T>` accepts any bit pattern by design.
// `Wrapping<T>` is a `#[repr(transparent)]` newtype.
unsafe impl<T: ?Sized> DeviceCopy for PhantomData<T> {}
unsafe impl<T: DeviceCopy> DeviceCopy for MaybeUninit<T> {}
unsafe impl<T: DeviceCopy> DeviceCopy for Wrapping<T> {}

macro_rules! impl_device_copy_tuple {
    ($($name:ident),+ $(,)?) => {
        unsafe impl<$($name: DeviceCopy),+> DeviceCopy for ($($name,)+) {}
    };
}

impl_device_copy_tuple!(A);
impl_device_copy_tuple!(A, B);
impl_device_copy_tuple!(A, B, C);
impl_device_copy_tuple!(A, B, C, D);
impl_device_copy_tuple!(A, B, C, D, E);
impl_device_copy_tuple!(A, B, C, D, E, F);
impl_device_copy_tuple!(A, B, C, D, E, F, G);
impl_device_copy_tuple!(A, B, C, D, E, F, G, H);

#[cfg(feature = "f16")]
unsafe impl DeviceCopy for f16 {}
unsafe impl DeviceCopy for half::bf16 {}
unsafe impl DeviceCopy for half::f16 {}

/// Owning handle to a contiguous device allocation of `T` elements.
///
/// Holds a raw device pointer, element count, and a reference-counted
/// context that keeps the CUDA context alive. Synchronous allocations are
/// freed with `cuMemFree`. Dropping a stream-ordered allocation synchronizes
/// its context before enqueueing `cuMemFreeAsync` on its retained allocation
/// stream. Use the unsafe [`DeviceBuffer::drop_async`] when the caller can
/// provide explicit stream ordering and must avoid that context-wide
/// synchronization.
///
/// Device buffers may only transfer plain device-copyable values. Owning host
/// types such as [`String`] are rejected because copying their bytes to and
/// from device memory would not preserve Rust ownership invariants.
///
/// ```compile_fail
/// # use cuda_core::{CudaStream, DeviceBuffer};
/// # fn rejects_non_device_copy(stream: &CudaStream) {
/// let _ = DeviceBuffer::<String>::zeroed(stream, 1);
/// # }
/// ```
pub struct DeviceBuffer<T> {
    ptr: CUdeviceptr,
    len: usize,
    num_bytes: usize,
    ctx: Arc<CudaContext>,
    /// Retains the allocation stream for a stream-ordered (`cuMemAllocAsync`)
    /// allocation. Ordinary `Drop` first synchronizes the context so work
    /// submitted on any stream has completed, then frees on this stream.
    /// `None` identifies a synchronous (`cuMemAlloc`) allocation.
    dealloc_stream: Option<Arc<CudaStream>>,
    _marker: PhantomData<T>,
}

// SAFETY: CUdeviceptr is a u64 handle valid across threads when the owning
// context is bound. The PhantomData<T> is Send if T is Send.
unsafe impl<T: Send> Send for DeviceBuffer<T> {}
// SAFETY: &DeviceBuffer only exposes cu_deviceptr() and len(), both of which
// return Copy values. No interior mutability.
unsafe impl<T: Send + Sync> Sync for DeviceBuffer<T> {}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if self.ptr != 0 {
            self.ctx.record_err(self.ctx.bind_to_thread());
            // Safe buffer operations can enqueue work on any stream in this
            // context. Synchronize all of them before implicitly freeing a
            // stream-ordered allocation.
            let result = match &self.dealloc_stream {
                Some(stream) => match self.ctx.synchronize() {
                    Ok(()) => unsafe {
                        crate::simt::memory::free_async(self.ptr, stream.cu_stream())
                    },
                    Err(error) => Err(error),
                },
                None => unsafe { crate::simt::memory::free_sync(self.ptr) },
            };
            self.ctx.record_err(result);
        }
    }
}

impl<T> DeviceBuffer<T> {
    /// Returns the raw `CUdeviceptr` for use in kernel argument lists.
    #[inline]
    pub fn cu_deviceptr(&self) -> CUdeviceptr {
        self.ptr
    }

    /// Number of `T` elements in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the buffer has zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total size in bytes (`len * size_of::<T>()`).
    #[inline]
    pub fn num_bytes(&self) -> usize {
        self.num_bytes
    }

    /// Returns a reference to the owning context.
    #[inline]
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Constructs a `DeviceBuffer` from pre-existing raw parts.
    ///
    /// # Safety
    ///
    /// - `ptr` must have been allocated via `cuMemAlloc*` with at least
    ///   `len * size_of::<T>()` bytes.
    /// - `ptr` must belong to the same CUDA context as `ctx`.
    /// - The caller transfers ownership -- `ptr` will be freed on drop.
    /// - `ptr` is assumed to be a synchronous (`cuMemAlloc`) allocation and is
    ///   freed with the synchronous `cuMemFree` on drop. Do not pass a
    ///   stream-ordered (`cuMemAllocAsync`) pointer here.
    ///
    /// # Panics
    ///
    /// Panics if `len * size_of::<T>()` overflows `usize`.
    pub unsafe fn from_raw_parts(ptr: CUdeviceptr, len: usize, ctx: Arc<CudaContext>) -> Self {
        // SAFETY: `from_raw_parts` has the same raw-allocation safety contract,
        // with no stream-ordered deallocation metadata attached.
        unsafe { Self::from_raw_parts_with_dealloc_stream(ptr, len, ctx, None) }
    }

    unsafe fn from_raw_parts_with_dealloc_stream(
        ptr: CUdeviceptr,
        len: usize,
        ctx: Arc<CudaContext>,
        dealloc_stream: Option<Arc<CudaStream>>,
    ) -> Self {
        let num_bytes =
            allocation_size::<T>(len).expect("DeviceBuffer::from_raw_parts byte size overflow");
        Self {
            ptr,
            len,
            num_bytes,
            ctx,
            dealloc_stream,
            _marker: PhantomData,
        }
    }

    /// Consumes the buffer and returns the raw parts without freeing.
    ///
    /// The caller is responsible for eventually freeing `ptr` with the
    /// allocator that matches how it was created. For stream-ordered
    /// allocations, this does not return the stored deallocation stream; the
    /// caller must already know which stream to use for `cuMemFreeAsync`.
    pub fn into_raw_parts(self) -> (CUdeviceptr, usize, Arc<CudaContext>) {
        let (ptr, len, ctx, _dealloc_stream) = self.into_all_raw_parts();
        (ptr, len, ctx)
    }

    fn into_all_raw_parts(
        self,
    ) -> (
        CUdeviceptr,
        usize,
        Arc<CudaContext>,
        Option<Arc<CudaStream>>,
    ) {
        // Suppress the buffer's `Drop` (which would free `ptr`) while still
        // moving out the heap-owned fields. Callers that do not need
        // `dealloc_stream` can drop it after this helper returns.
        let this = std::mem::ManuallyDrop::new(self);
        let ptr = this.ptr;
        let len = this.len;
        // SAFETY: `this` is `ManuallyDrop` and is never used again, so reading
        // out its non-`Copy` fields takes ownership without a double drop.
        let ctx = unsafe { std::ptr::read(&this.ctx) };
        let dealloc_stream = unsafe { std::ptr::read(&this.dealloc_stream) };
        (ptr, len, ctx, dealloc_stream)
    }

    /// Reinterpret the element type of this buffer as `A`.
    ///
    /// `A` must have the same size and alignment as `T` (e.g. `A` is
    /// `#[repr(transparent)]` over `T`). This is the "atomic-slice launch
    /// mapping" for issue #151: allocate and initialize a plain
    /// `DeviceBuffer<u64>`, then hand it to a kernel that takes
    /// `&[DeviceAtomicU64]`. The pointer, length, and bytes are unchanged;
    /// only the element type the kernel sees changes to one whose pointee
    /// permits shared mutation (so rustc does not mark it `readonly`/`noalias`).
    ///
    /// Element counts are preserved because `size_of::<A>() == size_of::<T>()`.
    pub fn cast_elem<A>(self) -> DeviceBuffer<A> {
        assert_eq!(
            std::mem::size_of::<A>(),
            std::mem::size_of::<T>(),
            "cast_elem requires the same element size"
        );
        assert_eq!(
            std::mem::align_of::<A>(),
            std::mem::align_of::<T>(),
            "cast_elem requires the same element alignment"
        );
        let (ptr, len, ctx, dealloc_stream) = self.into_all_raw_parts();
        // SAFETY: `ptr` came from a valid `DeviceBuffer<T>` allocation of `len`
        // elements; `A` has identical size and alignment, so the same allocation
        // is a valid `DeviceBuffer<A>` of the same length and the same byte
        // extent. Ownership transfers; the original buffer's `Drop` is
        // suppressed by `into_all_raw_parts`, and the allocation metadata is
        // preserved for the new element type.
        unsafe {
            DeviceBuffer::<A>::from_raw_parts_with_dealloc_stream(ptr, len, ctx, dealloc_stream)
        }
    }

    /// Reinterpret this buffer as `A`, adjusting the element count.
    ///
    /// Where [`Self::cast_elem`] requires `A` to be layout-identical to `T`,
    /// this allows a *different* size and alignment and recomputes the length
    /// from the byte extent. That is what makes it usable for grouping scalars
    /// into an over-aligned vector element, which is by construction a
    /// different size and alignment and so cannot go through `cast_elem`.
    ///
    /// ```rust,ignore
    /// // 4N floats become N 16-byte-aligned quads, same allocation.
    /// let quads: DeviceBuffer<F32x4> = floats.cast_chunks()?;
    /// ```
    ///
    /// # Why this exists
    ///
    /// Wide memory transactions require an over-aligned element type. Without a
    /// length-adjusting cast, adopting one means every producer and consumer of
    /// a buffer has to agree on the element type simultaneously: in practice
    /// that meant six kernel signature changes and a dummy buffer to migrate a
    /// single buffer pair. This lets the element type be chosen at the boundary
    /// instead, so a kernel that wants wide accesses can take `&[F32x4]` while
    /// the buffer is still allocated and filled as `f32`.
    ///
    /// # Errors
    ///
    /// Returns the buffer unchanged if the reinterpretation would not be exact:
    ///
    /// - the byte extent is not a whole number of `A`
    /// - the device pointer is not aligned for `A`
    /// - `A` is zero-sized
    ///
    /// The alignment check is expected to pass: `cuMemAlloc` returns at least
    /// 256-byte-aligned memory, which satisfies any vector type. It is checked
    /// rather than assumed because a buffer can also be built from
    /// [`Self::from_raw_parts`] with a hand-computed pointer, and that is
    /// exactly the case where being wrong is silent.
    ///
    /// Returning the original on failure rather than panicking keeps the
    /// fallback path available, since a caller that cannot widen usually has a
    /// scalar version to fall back to.
    pub fn cast_chunks<A>(self) -> Result<DeviceBuffer<A>, Self> {
        let bytes = self.len.saturating_mul(std::mem::size_of::<T>());
        let Some(new_len) = chunk_cast_len(
            bytes,
            self.ptr as usize,
            std::mem::size_of::<A>(),
            std::mem::align_of::<A>(),
        ) else {
            return Err(self);
        };
        let (ptr, _len, ctx, dealloc_stream) = self.into_all_raw_parts();
        // SAFETY: `ptr` came from a valid allocation of `bytes` bytes, checked
        // above to be exactly `new_len` elements of `A` and to be aligned for
        // `A`. The allocation and its ownership are unchanged; only the element
        // type and the count describing the same bytes change. `Drop` on the
        // original is suppressed by `into_all_raw_parts`.
        Ok(unsafe {
            DeviceBuffer::<A>::from_raw_parts_with_dealloc_stream(ptr, new_len, ctx, dealloc_stream)
        })
    }

    /// Whether [`Self::cast_chunks`] to `A` would succeed.
    ///
    /// For choosing between a wide and a scalar path without consuming the
    /// buffer to find out.
    #[must_use]
    pub fn can_cast_chunks<A>(&self) -> bool {
        let bytes = self.len.saturating_mul(std::mem::size_of::<T>());
        chunk_cast_len(
            bytes,
            self.ptr as usize,
            std::mem::size_of::<A>(),
            std::mem::align_of::<A>(),
        )
        .is_some()
    }
}

impl<T: DeviceCopy> DeviceBuffer<T> {
    /// Allocates device memory, copies `data` from the host on `stream`, and
    /// synchronizes `stream` before returning.
    ///
    /// The synchronization keeps this safe for borrowed host slices: `data`
    /// may be dropped, reused, or mutated immediately after this function
    /// returns. For true host-device overlap with caller-managed source
    /// lifetimes, use [`Self::from_host_async_unchecked`].
    ///
    /// An empty `data` slice yields an empty buffer without touching the
    /// driver allocator.
    ///
    /// # Allocation safety on error
    ///
    /// The buffer takes ownership of the device allocation immediately after
    /// `malloc_sync`, before the fallible `memcpy_htod_async` enqueue and
    /// stream synchronization run. If either step fails, the early return
    /// drops the buffer and its `Drop` impl frees the allocation, so no
    /// device memory is leaked.
    pub fn from_host(stream: &CudaStream, data: &[T]) -> Result<Self, DriverError> {
        let ctx = stream.context().clone();
        let len = data.len();
        let num_bytes = allocation_size::<T>(len)?;

        // cuMemAlloc rejects zero-byte requests with CUDA_ERROR_INVALID_VALUE,
        // so represent an empty buffer as a null pointer (Drop skips it).
        if num_bytes == 0 {
            // SAFETY: a null pointer with zero bytes is never dereferenced
            // and Drop ignores it.
            return Ok(unsafe { Self::from_raw_parts(0, len, ctx) });
        }

        let ptr = unsafe { crate::simt::memory::malloc_sync(num_bytes)? };
        // SAFETY: `ptr` was just allocated with `num_bytes` bytes in the
        // stream's context; ownership transfers to `buf` here so any early
        // return below frees it through the buffer's own `Drop`.
        let buf = unsafe { Self::from_raw_parts(ptr, len, ctx) };
        let enqueue_result = unsafe {
            crate::simt::memory::memcpy_htod_async(
                buf.ptr,
                data.as_ptr(),
                num_bytes,
                stream.cu_stream(),
            )
        };
        let sync_result = stream.synchronize();
        enqueue_result?;
        sync_result?;
        Ok(buf)
    }

    /// Allocates device memory and enqueues a host-to-device copy from `data`
    /// on `stream`, returning without synchronizing.
    ///
    /// # Safety
    ///
    /// This call only enqueues the host-to-device copy and returns; CUDA may
    /// still be reading from `data` after the borrow is released. The caller
    /// must ensure `data` is not dropped, freed, mutated, or aliased until the
    /// enqueued copy has completed, typically after the next
    /// [`CudaStream::synchronize`] call or a stream-ordered event wait.
    pub unsafe fn from_host_async_unchecked(
        stream: &CudaStream,
        data: &[T],
    ) -> Result<Self, DriverError> {
        let ctx = stream.context().clone();
        let len = data.len();
        let num_bytes = std::mem::size_of_val(data);

        // cuMemAlloc rejects zero-byte requests with CUDA_ERROR_INVALID_VALUE,
        // so represent an empty buffer as a null pointer (Drop skips it).
        if num_bytes == 0 {
            // SAFETY: a null pointer with zero bytes is never dereferenced
            // and Drop ignores it.
            return Ok(unsafe { Self::from_raw_parts(0, len, ctx) });
        }

        let ptr = unsafe { crate::simt::memory::malloc_sync(num_bytes)? };
        // SAFETY: `ptr` was just allocated with `num_bytes` bytes in the
        // stream's context; ownership transfers to `buf` here so any early
        // return below frees it through the buffer's own `Drop`.
        let buf = unsafe { Self::from_raw_parts(ptr, len, ctx) };
        unsafe {
            crate::simt::memory::memcpy_htod_async(
                buf.ptr,
                data.as_ptr(),
                num_bytes,
                stream.cu_stream(),
            )?;
        }
        Ok(buf)
    }

    /// Allocates device memory and enqueues a host-to-device copy from a
    /// pinned host buffer on `stream`, returning without synchronizing.
    ///
    /// Pinned host memory allows CUDA to avoid the pageable-memory staging
    /// path and is required when host-device copies need true asynchronous
    /// overlap with other stream work.
    ///
    /// `PinnedHostBuffer` currently uses `cuMemAllocHost` without the
    /// `PORTABLE` flag, so the allocation is only pinned in the context that
    /// created it. In debug builds this asserts that `data` and `stream`
    /// share the same [`CudaContext`].
    ///
    /// The device-to-host counterparts are [`Self::copy_to_pinned_host`]
    /// (blocking) and [`Self::copy_to_pinned_host_async`] (non-blocking). To
    /// refill an existing device buffer instead of allocating a new one, use
    /// [`Self::copy_from_pinned_host_async`].
    ///
    /// # Safety
    ///
    /// This call only enqueues the host-to-device copy on `stream` and
    /// returns; CUDA may still be reading from `data`'s pinned pointer long
    /// after this function returns. The caller is responsible for ensuring
    /// `data` is not dropped, freed, mutated, or aliased until the enqueued
    /// copy has completed, typically after the next
    /// [`CudaStream::synchronize`] call or a stream-ordered event wait.
    /// Dropping `data` before that synchronization point calls
    /// `cuMemFreeHost` while the in-flight transfer is still reading the
    /// buffer, which is undefined behavior.
    pub unsafe fn from_pinned_host(
        stream: &CudaStream,
        data: &PinnedHostBuffer<T>,
    ) -> Result<Self, DriverError> {
        debug_assert!(
            Arc::ptr_eq(data.context(), stream.context()),
            "pinned host buffer and stream must belong to the same CUDA context"
        );
        // SAFETY: this method's safety contract requires the caller to keep
        // the pinned source valid until the enqueued copy completes.
        unsafe { Self::from_host_async_unchecked(stream, data.as_slice()) }
    }

    /// Allocates zero-initialized device memory of `len` elements, enqueued
    /// on `stream`.
    ///
    /// A `len` of zero (or a zero-sized `T`) yields an empty buffer without
    /// touching the driver allocator.
    ///
    /// # Allocation safety on error
    ///
    /// The returned buffer takes ownership of the device allocation
    /// immediately after `malloc_sync`, before the fallible
    /// `memset_d8_async` enqueue runs. If the enqueue fails, the early
    /// return drops the buffer and its `Drop` impl frees the allocation, so
    /// no device memory is leaked.
    pub fn zeroed(stream: &CudaStream, len: usize) -> Result<Self, DriverError> {
        let ctx = stream.context().clone();
        let num_bytes = allocation_size::<T>(len)?;

        // cuMemAlloc rejects zero-byte requests with CUDA_ERROR_INVALID_VALUE,
        // so represent an empty buffer as a null pointer (Drop skips it).
        if num_bytes == 0 {
            // SAFETY: a null pointer with zero bytes is never dereferenced
            // and Drop ignores it.
            return Ok(unsafe { Self::from_raw_parts(0, len, ctx) });
        }

        let ptr = unsafe { crate::simt::memory::malloc_sync(num_bytes)? };
        // SAFETY: `ptr` was just allocated with `num_bytes` bytes in the
        // stream's context; ownership transfers to `buf` here so any early
        // return below frees it through the buffer's own `Drop`.
        let buf = unsafe { Self::from_raw_parts(ptr, len, ctx) };
        unsafe {
            crate::simt::memory::memset_d8_async(buf.ptr, 0, num_bytes, stream.cu_stream())?;
        }
        Ok(buf)
    }

    /// Copies the entire buffer back to the host, returning a `Vec<T>`.
    ///
    /// Synchronizes on `stream` before returning so the host vector is safe
    /// to read immediately.
    pub fn to_host_vec(&self, stream: &CudaStream) -> Result<Vec<T>, DriverError> {
        let mut host = Vec::with_capacity(self.len);
        unsafe {
            crate::simt::memory::memcpy_dtoh_async(
                host.as_mut_ptr(),
                self.ptr,
                self.num_bytes(),
                stream.cu_stream(),
            )?;
        }
        stream.synchronize()?;
        unsafe { host.set_len(self.len) };
        Ok(host)
    }

    /// Copies the buffer contents into an existing host slice.
    ///
    /// Synchronizes on `stream` before returning. Panics if
    /// `dst.len() < self.len()`.
    pub fn copy_to_host(&self, stream: &CudaStream, dst: &mut [T]) -> Result<(), DriverError> {
        assert!(
            dst.len() >= self.len,
            "destination slice too small: {} < {}",
            dst.len(),
            self.len
        );
        unsafe {
            crate::simt::memory::memcpy_dtoh_async(
                dst.as_mut_ptr(),
                self.ptr,
                self.num_bytes(),
                stream.cu_stream(),
            )?;
        }
        stream.synchronize()
    }

    /// Copies the buffer contents into an existing pinned host buffer and
    /// synchronizes `stream` before returning.
    ///
    /// Panics if `dst.len() < self.len()`. Use pinned destinations when you
    /// need the transfer to avoid pageable-memory staging; this helper still
    /// waits for completion before returning, matching [`Self::copy_to_host`].
    ///
    /// For true DtoH overlap, use [`Self::copy_to_pinned_host_async`] and
    /// synchronize the stream later.
    pub fn copy_to_pinned_host(
        &self,
        stream: &CudaStream,
        dst: &mut PinnedHostBuffer<T>,
    ) -> Result<(), DriverError> {
        // SAFETY: we synchronize the stream below before returning, so the
        // pinned destination is no longer being written to by CUDA when the
        // mutable borrow on `dst` is released to the caller.
        unsafe { self.copy_to_pinned_host_async(stream, dst)? };
        stream.synchronize()
    }

    /// Enqueues a device-to-host copy into an existing pinned host buffer and
    /// returns without synchronizing.
    ///
    /// Panics if `dst.len() < self.len()`.
    ///
    /// `PinnedHostBuffer` currently uses `cuMemAllocHost` without the
    /// `PORTABLE` flag, so the allocation is only pinned in the context that
    /// created it. In debug builds this asserts that `dst` and `stream`
    /// share the same [`CudaContext`].
    ///
    /// # Safety
    ///
    /// This call only enqueues the device-to-host copy on `stream` and
    /// returns; CUDA may still be writing into `dst`'s pinned pointer long
    /// after this function returns. The caller is responsible for ensuring
    /// `dst` is not dropped, freed, read, or aliased until the enqueued copy
    /// has completed, typically after the next [`CudaStream::synchronize`]
    /// call or a stream-ordered event wait. Dropping `dst` before that
    /// synchronization point calls `cuMemFreeHost` while the in-flight
    /// transfer is still writing the buffer, which is undefined behavior.
    pub unsafe fn copy_to_pinned_host_async(
        &self,
        stream: &CudaStream,
        dst: &mut PinnedHostBuffer<T>,
    ) -> Result<(), DriverError> {
        debug_assert!(
            Arc::ptr_eq(dst.context(), stream.context()),
            "pinned host buffer and stream must belong to the same CUDA context"
        );
        assert!(
            dst.len() >= self.len,
            "destination pinned host buffer too small: {} < {}",
            dst.len(),
            self.len
        );
        unsafe {
            crate::simt::memory::memcpy_dtoh_async(
                dst.as_mut_ptr(),
                self.ptr,
                self.num_bytes(),
                stream.cu_stream(),
            )
        }
    }

    /// Enqueues a host-to-device copy from a pinned host buffer into this
    /// device buffer and returns without synchronizing.
    ///
    /// This is the symmetric counterpart of
    /// [`Self::copy_to_pinned_host_async`]: it refills an existing device
    /// allocation from rotating pinned host stagers instead of allocating a
    /// fresh device buffer per refresh, which is the typical shape for
    /// asynchronous overlap pipelines.
    ///
    /// Panics if `src.len() > self.len()`.
    ///
    /// `PinnedHostBuffer` currently uses `cuMemAllocHost` without the
    /// `PORTABLE` flag, so the allocation is only pinned in the context that
    /// created it. In debug builds this asserts that `src` and `stream`
    /// share the same [`CudaContext`].
    ///
    /// # Safety
    ///
    /// This call only enqueues the host-to-device copy on `stream` and
    /// returns; CUDA may still be reading from `src`'s pinned pointer long
    /// after this function returns. The caller is responsible for ensuring
    /// `src` is not dropped, freed, mutated, or aliased until the enqueued
    /// copy has completed, typically after the next
    /// [`CudaStream::synchronize`] call or a stream-ordered event wait.
    /// Dropping `src` before that synchronization point calls
    /// `cuMemFreeHost` while the in-flight transfer is still reading the
    /// buffer, which is undefined behavior.
    pub unsafe fn copy_from_pinned_host_async(
        &mut self,
        stream: &CudaStream,
        src: &PinnedHostBuffer<T>,
    ) -> Result<(), DriverError> {
        debug_assert!(
            Arc::ptr_eq(src.context(), stream.context()),
            "pinned host buffer and stream must belong to the same CUDA context"
        );
        assert!(
            src.len() <= self.len,
            "source pinned host buffer too large: {} > {}",
            src.len(),
            self.len
        );
        let num_bytes = src.num_bytes();
        unsafe {
            crate::simt::memory::memcpy_htod_async(
                self.ptr,
                src.as_ptr(),
                num_bytes,
                stream.cu_stream(),
            )
        }
    }

    /// Allocates `len` elements of uninitialized device memory, enqueued on
    /// `stream`.
    ///
    /// Unlike [`Self::zeroed`], no `cuMemsetD8` is enqueued. The contents of
    /// the returned buffer are undefined until the caller writes them.
    ///
    /// The buffer co-owns `stream` (via the `Arc`) so its implicit `Drop` can
    /// release the stream-ordered allocation after synchronizing the context.
    /// Call the unsafe [`Self::drop_async`] to free explicitly on a chosen
    /// stream without a context-wide synchronization.
    ///
    /// # Safety
    ///
    /// Reading from the returned buffer before any kernel or memcpy has
    /// written it is undefined behavior.
    pub unsafe fn uninitialized_async(
        stream: &Arc<CudaStream>,
        len: usize,
    ) -> Result<Self, DriverError> {
        let ctx = stream.context().clone();
        let num_bytes = allocation_size::<T>(len)?;
        if num_bytes == 0 {
            // SAFETY: a null pointer with zero bytes is never dereferenced
            // and Drop/drop_async ignore it.
            return Ok(unsafe { Self::from_raw_parts(0, len, ctx) });
        }

        let ptr = unsafe { crate::simt::memory::malloc_async(stream.cu_stream(), num_bytes)? };
        Ok(Self {
            ptr,
            len,
            num_bytes,
            ctx,
            dealloc_stream: Some(stream.clone()),
            _marker: PhantomData,
        })
    }

    /// Copies `other` into `self` device-to-device, enqueued on `stream`.
    ///
    /// Panics if `other.len() != self.len()`.
    pub fn copy_from_device_async(
        &mut self,
        other: &DeviceBuffer<T>,
        stream: &CudaStream,
    ) -> Result<(), DriverError> {
        assert_eq!(
            self.len, other.len,
            "device-to-device copy length mismatch: dst {} != src {}",
            self.len, other.len
        );
        if self.num_bytes() == 0 {
            return Ok(());
        }
        unsafe {
            crate::simt::memory::memcpy_dtod_async(
                self.ptr,
                other.ptr,
                self.num_bytes(),
                stream.cu_stream(),
            )
        }
    }

    /// Copies `src` into `self` host-to-device on `stream` and synchronizes
    /// `stream` before returning.
    ///
    /// The synchronization keeps this safe for borrowed host slices: `src`
    /// may be dropped, reused, or mutated immediately after this function
    /// returns. Panics if `src.len() != self.len()`.
    pub fn copy_from_host(&mut self, stream: &CudaStream, src: &[T]) -> Result<(), DriverError> {
        // SAFETY: this safe wrapper synchronizes `stream` before returning,
        // so the borrowed host slice cannot be used by CUDA after this call.
        let enqueue_result = unsafe { self.copy_from_host_async_unchecked(stream, src) };
        let sync_result = if self.num_bytes() == 0 {
            Ok(())
        } else {
            stream.synchronize()
        };
        enqueue_result?;
        sync_result
    }

    /// Copies `src` into `self` host-to-device, enqueued on `stream`, and
    /// returns without synchronizing.
    ///
    /// # Safety
    ///
    /// This call only enqueues the host-to-device copy and returns; CUDA may
    /// still be reading from `src` after the borrow is released. The caller
    /// must ensure `src` is not dropped, freed, mutated, or aliased until the
    /// enqueued copy has completed, typically after the next
    /// [`CudaStream::synchronize`] call or a stream-ordered event wait.
    /// Panics if `src.len() != self.len()`.
    pub unsafe fn copy_from_host_async_unchecked(
        &mut self,
        stream: &CudaStream,
        src: &[T],
    ) -> Result<(), DriverError> {
        assert_eq!(
            self.len,
            src.len(),
            "host-to-device copy length mismatch: dst {} != src {}",
            self.len,
            src.len()
        );
        if self.num_bytes() == 0 {
            return Ok(());
        }
        unsafe {
            crate::simt::memory::memcpy_htod_async(
                self.ptr,
                src.as_ptr(),
                self.num_bytes(),
                stream.cu_stream(),
            )
        }
    }

    /// Consumes the buffer and frees it asynchronously on `stream`.
    ///
    /// For a stream-ordered allocation, this method makes `stream` wait for
    /// work already submitted on the allocation stream before enqueueing the
    /// free.
    ///
    /// Returns [`CUDA_ERROR_INVALID_CONTEXT`](cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_CONTEXT)
    /// if `stream` belongs to a different context. Validation and allocation
    /// stream ordering happen before the buffer is disarmed, so an error in
    /// either step leaves ordinary [`Drop`] responsible for cleanup. Once
    /// disarmed immediately before `cuMemFreeAsync`, an enqueue error leaks
    /// the allocation instead of attempting an unordered fallback free.
    ///
    /// # Safety
    ///
    /// The caller must ensure that every stream other than the allocation
    /// stream that has pending work touching this buffer is ordered before
    /// `stream` (for example via [`CudaStream::join`]). Only the allocation
    /// stream is joined automatically; an unordered third stream still
    /// racing the free is a driver-level use-after-free. This is the same
    /// deferred-use contract as [`Self::copy_from_host_async_unchecked`]:
    /// ordinary [`Drop`] is the safe alternative and synchronizes the whole
    /// context first.
    pub unsafe fn drop_async(mut self, stream: &CudaStream) -> Result<(), DriverError> {
        if self.ctx.as_ref() != stream.context().as_ref() {
            return Err(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_CONTEXT,
            ));
        }
        if self.ptr == 0 {
            return Ok(());
        }

        self.ctx.bind_to_thread()?;
        if let Some(allocation_stream) = &self.dealloc_stream {
            if allocation_stream.as_ref() != stream {
                stream.join(allocation_stream)?;
            }
        }

        let ptr = self.ptr;
        self.ptr = 0;
        unsafe { crate::simt::memory::free_async(ptr, stream.cu_stream()) }
    }

    /// Zeroes every byte in the buffer asynchronously on `stream`.
    pub fn zero_async(&mut self, stream: &CudaStream) -> Result<(), DriverError> {
        if self.num_bytes() == 0 {
            return Ok(());
        }
        unsafe {
            crate::simt::memory::memset_d8_async(self.ptr, 0, self.num_bytes(), stream.cu_stream())
        }
    }
}

fn allocation_size<T>(len: usize) -> Result<usize, DriverError> {
    len.checked_mul(std::mem::size_of::<T>()).ok_or(DriverError(
        cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE,
    ))
}

/// Element count for reinterpreting `bytes` at `addr` as elements of size
/// `elem_size` and alignment `align`, or `None` if it would not be exact.
///
/// Split out so [`DeviceBuffer::cast_chunks`] and
/// [`DeviceBuffer::can_cast_chunks`] cannot disagree, and so the decision is
/// testable without a device.
fn chunk_cast_len(bytes: usize, addr: usize, elem_size: usize, align: usize) -> Option<usize> {
    if elem_size == 0 || align == 0 {
        return None;
    }
    if !bytes.is_multiple_of(elem_size) {
        return None;
    }
    if !addr.is_multiple_of(align) {
        return None;
    }
    Some(bytes / elem_size)
}

#[cfg(test)]
mod chunk_cast_tests {
    use super::chunk_cast_len;

    /// A device allocation is at least 256-byte aligned, so the alignment check
    /// is expected to pass; these pin that it does, and that a hand-computed
    /// pointer is still rejected.
    #[test]
    fn accepts_an_aligned_allocation_that_divides() {
        // 1024 f32 viewed as 256 quads of 16 bytes.
        assert_eq!(chunk_cast_len(4096, 0x1000, 16, 16), Some(256));
        // 8-byte pairs out of the same buffer.
        assert_eq!(chunk_cast_len(4096, 0x1000, 8, 8), Some(512));
        // Identity cast.
        assert_eq!(chunk_cast_len(4096, 0x1000, 4, 4), Some(1024));
    }

    /// A length that is not a whole number of elements is refused rather than
    /// truncated, so a dropped tail cannot go unnoticed.
    #[test]
    fn refuses_a_byte_extent_that_does_not_divide() {
        assert_eq!(
            chunk_cast_len(12, 0x1000, 16, 16),
            None,
            "3 f32 into a quad"
        );
        assert_eq!(chunk_cast_len(4100, 0x1000, 16, 16), None);
        assert_eq!(chunk_cast_len(4088, 0x1000, 16, 16), None);
    }

    /// The case the check exists for: a pointer that did not come from
    /// `cuMemAlloc`, such as one offset by hand into a larger allocation.
    #[test]
    fn refuses_a_misaligned_base() {
        assert_eq!(chunk_cast_len(4096, 0x1004, 16, 16), None, "4-byte offset");
        assert_eq!(chunk_cast_len(4096, 0x1008, 16, 16), None, "8-byte offset");
        // Still fine for a narrower element.
        assert_eq!(chunk_cast_len(4096, 0x1008, 8, 8), Some(512));
        assert_eq!(chunk_cast_len(4096, 0x1004, 4, 4), Some(1024));
    }

    /// Every 256-byte-aligned base satisfies every vector alignment, which is
    /// why the check is expected to pass for a real allocation.
    #[test]
    fn a_cuda_allocation_alignment_satisfies_every_vector_type() {
        for base in [0usize, 256, 512, 4096, 1 << 20] {
            for align in [4usize, 8, 16] {
                assert!(
                    chunk_cast_len(4096, base, align, align).is_some(),
                    "base {base:#x} should satisfy align {align}"
                );
            }
        }
    }

    #[test]
    fn rejects_degenerate_parameters() {
        assert_eq!(chunk_cast_len(4096, 0x1000, 0, 16), None, "zero-sized");
        assert_eq!(chunk_cast_len(4096, 0x1000, 16, 0), None);
        // An empty buffer casts to an empty buffer.
        assert_eq!(chunk_cast_len(0, 0x1000, 16, 16), Some(0));
    }
}
