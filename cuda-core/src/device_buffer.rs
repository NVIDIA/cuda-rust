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

use bytemuck::{AnyBitPattern, NoUninit};
use cuda_bindings::CUdeviceptr;
use cuda_repr::DeviceRepr;

use crate::context::Context;
use crate::error::DriverError;
use crate::pinned_host_buffer::PinnedHostBuffer;
use crate::stream::Stream;

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
/// Implementors must be safe to duplicate with a byte-for-byte copy, and the
/// all-zero bit pattern must be a valid value because [`DeviceBuffer::zeroed`]
/// initializes memory with zero bytes. Types whose zero pattern is invalid,
/// such as `NonZeroU32`, cannot implement `DeviceCopy`.
///
/// `DeviceCopy` does not promise that every bit pattern is valid. `bool`
/// implements it although only `0` and `1` are `bool` values. That is sound
/// because a typed `DeviceBuffer<T>` holds valid `T` values by construction:
/// the safe constructors copy `T` values from the host or write zero bytes, a
/// kernel that writes through `&mut T` writes a `T`, and the raw constructors
/// are `unsafe`, so their caller vouches for the contents. The one place
/// arbitrary bytes could become typed values is the safe reinterpreting casts,
/// [`DeviceBuffer::cast_elem`] and [`DeviceBuffer::cast_chunks`]. They require
/// the source to implement [`bytemuck::NoUninit`] and the target to implement
/// [`bytemuck::AnyBitPattern`]. This excludes uninitialized source bytes and
/// padding, and rejects `bool` as a target, so a `DeviceBuffer<bool>` can only
/// come from memory written as `bool`.
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

// Zero is `false`, and the trait documentation explains why the `0`/`1`
// validity requirement does not stop `bool` from being `DeviceCopy`. It is not
// `bytemuck::AnyBitPattern`, so the reinterpreting casts cannot produce it.
unsafe impl DeviceCopy for bool {}

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

// SAFETY: these transparent numeric storage wrappers own no resources, have
// valid all-zero representations, and can be duplicated byte-for-byte. Packed
// FP4 remains one byte holding two logical values; copying does not unpack it.
impl_device_copy!(
    crate::tf32,
    crate::f8e4m3fn,
    crate::f8e5m2,
    crate::f8e8m0fnu,
    crate::f8e5m3fnu,
    crate::f4e2m1fnx2
);

/// Owning handle to a contiguous device allocation of `T` elements.
///
/// Shares an allocation owner, its CUDA context, and an access tracker with
/// tensor and byte views. Safe submissions retain independent access leases
/// until completion, including when a launch future is dropped or forgotten.
/// Drop retires deallocation on a worker without waiting on the caller's
/// thread. Unsafe raw-pointer users must retain storage through their own work.
///
/// Device buffers may only transfer plain device-copyable values. Owning host
/// types such as [`String`] are rejected because copying their bytes to and
/// from device memory would not preserve Rust ownership invariants.
///
/// ```compile_fail
/// # use cuda_core::{Stream, DeviceBuffer};
/// # fn rejects_non_device_copy(stream: &Stream) {
/// let _ = DeviceBuffer::<String>::zeroed(stream, 1);
/// # }
/// ```
pub struct DeviceBuffer<T> {
    storage: crate::RawDeviceBuffer,
    shared_access_writes: bool,
    ptr: CUdeviceptr,
    len: usize,
    num_bytes: usize,
    _marker: PhantomData<T>,
}

// SAFETY: CUdeviceptr is a u64 handle valid across threads when the owning
// context is bound. The PhantomData<T> is Send if T is Send.
unsafe impl<T: Send> Send for DeviceBuffer<T> {}
// SAFETY: &DeviceBuffer only exposes cu_deviceptr() and len(), both of which
// return Copy values. No interior mutability.
unsafe impl<T: Send + Sync> Sync for DeviceBuffer<T> {}

impl<T> DeviceBuffer<T> {
    fn enqueue_access(
        &self,
        stream: &Stream,
        write: bool,
        enqueue: impl FnOnce() -> Result<(), DriverError>,
    ) -> Result<(), DriverError> {
        let lease = self.storage.acquire(stream, write, false)?;
        self.storage.context().bind_to_thread()?;
        let submitted = enqueue();
        let retained = lease.retain_until_complete(stream);
        submitted.and(retained)
    }

    /// The shared byte storage and allocation identity underlying this view.
    pub fn storage(&self) -> &crate::RawDeviceBuffer {
        &self.storage
    }
    /// Whether a shared kernel reference may mutate elements through interior
    /// mutability. Representation wrappers and unclassified raw types are
    /// conservative; ordinary DeviceCopy values can share read accesses.
    #[doc(hidden)]
    pub fn shared_access_writes(&self) -> bool {
        self.shared_access_writes
    }

    /// Erases the element type without copying or releasing storage.
    pub fn into_storage(self) -> crate::RawDeviceBuffer {
        self.storage
    }

    /// Constructs a typed view over shared byte storage.
    ///
    /// # Safety
    /// Every initialized element must be a valid `T`; the caller must preserve
    /// element validity across all aliases (including future writes through
    /// aliases with another element type). The storage must be initialized before
    /// any read, and raw foreign accesses must obey its access-exclusion contract.
    pub unsafe fn from_storage(
        storage: crate::RawDeviceBuffer,
        len: usize,
    ) -> Result<Self, DriverError> {
        let num_bytes = allocation_size::<T>(len)?;
        let ptr = storage.cu_deviceptr();
        if num_bytes != storage.len_bytes()
            || (num_bytes != 0 && !(ptr as usize).is_multiple_of(std::mem::align_of::<T>()))
        {
            return Err(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE,
            ));
        }
        Ok(Self {
            ptr,
            len,
            num_bytes,
            storage,
            shared_access_writes: true,
            _marker: PhantomData,
        })
    }

    /// Borrows a contiguous range of elements while retaining the same storage
    /// owner and access tracker. No device memory is copied.
    pub fn slice(
        &self,
        range: std::ops::Range<usize>,
    ) -> Result<DeviceBufferView<'_, T>, DriverError> {
        if range.start > range.end || range.end > self.len {
            return Err(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE,
            ));
        }
        let storage = self.storage.slice_bytes(
            allocation_size::<T>(range.start)?,
            allocation_size::<T>(range.end - range.start)?,
        )?;
        Ok(DeviceBufferView {
            storage,
            shared_access_writes: self.shared_access_writes,
            len: range.end - range.start,
            _borrow: PhantomData,
        })
    }

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
    pub fn context(&self) -> &Arc<Context> {
        self.storage.context()
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
    pub unsafe fn from_raw_parts(ptr: CUdeviceptr, len: usize, ctx: Arc<Context>) -> Self {
        // SAFETY: `from_raw_parts` has the same raw-allocation safety contract,
        // with no stream-ordered deallocation metadata attached.
        unsafe { Self::from_raw_parts_with_dealloc_stream(ptr, len, ctx, None) }
    }

    unsafe fn from_raw_parts_with_dealloc_stream(
        ptr: CUdeviceptr,
        len: usize,
        ctx: Arc<Context>,
        dealloc_stream: Option<Arc<Stream>>,
    ) -> Self {
        let num_bytes =
            allocation_size::<T>(len).expect("DeviceBuffer::from_raw_parts byte size overflow");
        let storage = if let Some(stream) = &dealloc_stream {
            unsafe { crate::RawDeviceBuffer::from_raw_parts_async(ptr, num_bytes, stream.clone()) }
        } else {
            unsafe { crate::RawDeviceBuffer::from_raw_parts(ptr, num_bytes, ctx.clone()) }
        };
        Self {
            storage,
            shared_access_writes: true,
            ptr,
            len,
            num_bytes,
            _marker: PhantomData,
        }
    }

    /// Consumes the buffer and returns the raw parts without freeing.
    ///
    /// The caller is responsible for eventually freeing `ptr` with the
    /// allocator that matches how it was created. For stream-ordered
    /// allocations, this does not return the stored deallocation stream; the
    /// caller must already know which stream to use for `cuMemFreeAsync`.
    pub fn into_raw_parts(self) -> (CUdeviceptr, usize, Arc<Context>) {
        let (ptr, len, ctx, _dealloc_stream) = self.into_all_raw_parts();
        (ptr, len, ctx)
    }

    fn into_all_raw_parts(self) -> (CUdeviceptr, usize, Arc<Context>, Option<Arc<Stream>>) {
        let Self { storage, len, .. } = self;
        let (ptr, _, ctx, stream) = storage.into_owned_parts();
        (ptr, len, ctx, stream)
    }

    /// Reinterpret the element type of this buffer as `A`.
    ///
    /// `A` must have the same size and alignment as `T`. The pointer, length,
    /// and bytes are unchanged; only the element type changes. Element counts
    /// are preserved because `size_of::<A>() == size_of::<T>()`.
    ///
    /// `T: NoUninit` ensures that every source byte is initialized, including
    /// bytes that would otherwise be padding. `A: AnyBitPattern` accepts every
    /// initialized bit pattern, so those bytes are valid `A` values. Destination
    /// types with invalid bit patterns, such as `bool`, are rejected, so a
    /// `DeviceBuffer<bool>` can only come from memory written as `bool`.
    ///
    /// Atomic types and other `#[repr(transparent)]` wrappers that are not
    /// `Copy` cannot be `AnyBitPattern`; use [`Self::wrap_elem`] and
    /// [`Self::peel_elem`] for those.
    pub fn cast_elem<A: AnyBitPattern>(self) -> DeviceBuffer<A>
    where
        T: NoUninit,
    {
        // SAFETY: NoUninit guarantees initialized source bytes, and every
        // initialized bit pattern is valid as A; relabel checks the layout.
        let mut buffer = unsafe { self.relabel() };
        buffer.shared_access_writes = false;
        buffer
    }

    /// View this buffer through a `#[repr(transparent)]` wrapper of `T`.
    ///
    /// This is the atomic-slice launch mapping for issue #151: allocate and
    /// initialize a plain `DeviceBuffer<u64>`, then hand it to a kernel that
    /// takes `&[DeviceAtomicU64]`. The pointer, length, and bytes are
    /// unchanged; only the element type the kernel sees changes to one whose
    /// pointee permits shared mutation, so rustc does not mark the slice
    /// `readonly`/`noalias`.
    ///
    /// `W: DeviceRepr<T>` guarantees the same layout and valid initialized
    /// representations in both directions, without element destruction or
    /// additional ownership obligations. It permits no shared-reference conversions.
    /// The wrapper need not be `Copy` or `DeviceCopy`; the
    /// atomic types are neither. [`Self::peel_elem`] goes back to `T`, for
    /// example before [`Self::to_host_vec`].
    ///
    /// This consumes the buffer and retains its allocation, context, and
    /// deallocation stream. It does not synchronize pending kernel accesses;
    /// those must obey the launch API's borrowing and ordering requirements.
    pub fn wrap_elem<W: DeviceRepr<T>>(self) -> DeviceBuffer<W> {
        // SAFETY: DeviceRepr guarantees every initialized T representation
        // is valid as W; relabel preserves ownership and checks the layout.
        unsafe { self.relabel() }
    }

    /// Remove a `#[repr(transparent)]` wrapper from the element type.
    ///
    /// The inverse of [`Self::wrap_elem`]: `T` must implement `DeviceRepr<I>`,
    /// so a `DeviceBuffer<DeviceAtomicU64>` peels back to the
    /// `DeviceBuffer<u64>` it was made from, which can be read on the host.
    pub fn peel_elem<I>(self) -> DeviceBuffer<I>
    where
        T: DeviceRepr<I>,
    {
        // SAFETY: DeviceRepr guarantees every initialized T representation
        // is valid as I; relabel preserves ownership and checks the layout.
        let mut buffer = unsafe { self.relabel() };
        buffer.shared_access_writes = false;
        buffer
    }

    /// Hand this allocation to a `DeviceBuffer<A>` of the same length.
    ///
    /// # Safety
    ///
    /// Every value of `T` in the allocation must also be a valid `A`. The
    /// three public casts above each establish that through a trait bound.
    unsafe fn relabel<A>(self) -> DeviceBuffer<A> {
        assert_eq!(
            std::mem::size_of::<A>(),
            std::mem::size_of::<T>(),
            "element casts require the same element size"
        );
        assert_eq!(
            std::mem::align_of::<A>(),
            std::mem::align_of::<T>(),
            "element casts require the same element alignment"
        );
        DeviceBuffer {
            storage: self.storage,
            shared_access_writes: true,
            ptr: self.ptr,
            len: self.len,
            num_bytes: self.num_bytes,
            _marker: PhantomData,
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
    ///
    /// As with [`Self::cast_elem`], `T: NoUninit` guarantees initialized source
    /// bytes and `A: AnyBitPattern` accepts every initialized bit pattern. A
    /// host-defined vector element gets there with `unsafe impl bytemuck::Pod`
    /// next to its `DeviceCopy` impl.
    pub fn cast_chunks<A: AnyBitPattern>(self) -> Result<DeviceBuffer<A>, Self>
    where
        T: NoUninit,
    {
        let bytes = self.len.saturating_mul(std::mem::size_of::<T>());
        let Some(new_len) = chunk_cast_len(
            bytes,
            self.ptr as usize,
            std::mem::size_of::<A>(),
            std::mem::align_of::<A>(),
        ) else {
            return Err(self);
        };
        Ok(DeviceBuffer {
            storage: self.storage,
            shared_access_writes: false,
            ptr: self.ptr,
            len: new_len,
            num_bytes: self.num_bytes,
            _marker: PhantomData,
        })
    }

    /// Whether the layout permits [`Self::cast_chunks`] to `A`.
    ///
    /// For choosing between a wide and a scalar path without consuming the
    /// buffer to find out. This only checks layout; the actual cast also
    /// requires `T: NoUninit`.
    #[must_use]
    pub fn can_cast_chunks<A: AnyBitPattern>(&self) -> bool {
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
    fn empty_copy(len: usize, ctx: Arc<Context>) -> Self {
        // SAFETY: a null pointer with zero bytes is never dereferenced
        // and Drop/drop_async ignore it. DeviceCopy excludes interior mutability.
        let mut buf = unsafe { Self::from_raw_parts(0, len, ctx) };
        buf.shared_access_writes = false;
        buf
    }

    /// Constructs a plain-value typed view of shared byte storage.
    ///
    /// # Safety
    /// The initialization, validity, and aliasing requirements of
    /// `from_storage` apply. DeviceCopy additionally excludes interior mutability.
    pub unsafe fn from_storage_copy(
        storage: crate::RawDeviceBuffer,
        len: usize,
    ) -> Result<Self, DriverError> {
        let mut buffer = unsafe { Self::from_storage(storage, len)? };
        buffer.shared_access_writes = false;
        Ok(buffer)
    }

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
    pub fn from_host(stream: &Stream, data: &[T]) -> Result<Self, DriverError> {
        let ctx = stream.context().clone();
        let len = data.len();
        let num_bytes = allocation_size::<T>(len)?;

        // cuMemAlloc rejects zero-byte requests with CUDA_ERROR_INVALID_VALUE,
        // so represent an empty buffer as a null pointer (Drop skips it).
        if num_bytes == 0 {
            return Ok(Self::empty_copy(len, ctx));
        }

        ctx.bind_to_thread()?;
        let ptr = unsafe { crate::memory::malloc_sync(num_bytes)? };
        // SAFETY: `ptr` was just allocated with `num_bytes` bytes in the
        // stream's context; ownership transfers to `buf` here so any early
        // return below frees it through the buffer's own `Drop`.
        let mut buf = unsafe { Self::from_raw_parts(ptr, len, ctx) };
        let enqueue_result = unsafe {
            crate::memory::memcpy_htod_async(buf.ptr, data.as_ptr(), num_bytes, stream.cu_stream())
        };
        let sync_result = stream.synchronize();
        enqueue_result?;
        sync_result?;
        buf.shared_access_writes = false;
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
    /// [`Stream::synchronize`] call or a stream-ordered event wait.
    pub unsafe fn from_host_async_unchecked(
        stream: &Stream,
        data: &[T],
    ) -> Result<Self, DriverError> {
        let ctx = stream.context().clone();
        let len = data.len();
        let num_bytes = std::mem::size_of_val(data);

        // cuMemAlloc rejects zero-byte requests with CUDA_ERROR_INVALID_VALUE,
        // so represent an empty buffer as a null pointer (Drop skips it).
        if num_bytes == 0 {
            return Ok(Self::empty_copy(len, ctx));
        }

        ctx.bind_to_thread()?;
        let ptr = unsafe { crate::memory::malloc_sync(num_bytes)? };
        // SAFETY: `ptr` was just allocated with `num_bytes` bytes in the
        // stream's context; ownership transfers to `buf` here so any early
        // return below frees it through the buffer's own `Drop`.
        let mut buf = unsafe { Self::from_raw_parts(ptr, len, ctx) };
        buf.enqueue_access(stream, true, || unsafe {
            crate::memory::memcpy_htod_async(buf.ptr, data.as_ptr(), num_bytes, stream.cu_stream())
        })?;
        buf.shared_access_writes = false;
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
    /// created it. This checks that `data` and `stream` share the same
    /// CUDA context.
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
    /// [`Stream::synchronize`] call or a stream-ordered event wait.
    /// Dropping `data` before that synchronization point calls
    /// `cuMemFreeHost` while the in-flight transfer is still reading the
    /// buffer, which is undefined behavior.
    pub unsafe fn from_pinned_host(
        stream: &Stream,
        data: &PinnedHostBuffer<T>,
    ) -> Result<Self, DriverError> {
        if data.context().as_ref() != stream.context().as_ref() {
            return Err(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_CONTEXT,
            ));
        }
        // SAFETY: this method's safety contract requires the caller to keep
        // the pinned source valid until the enqueued copy completes.
        unsafe { Self::from_host_async_unchecked(stream, data.as_slice()) }
    }

    /// Allocates zero-initialized device memory of `len` elements.
    ///
    /// The zero-fill is enqueued on `stream`, then this method synchronizes
    /// that stream before returning. The initialized buffer is therefore safe
    /// to use on another stream, including for types such as `bool` whose
    /// validity depends on initialization having completed.
    ///
    /// A `len` of zero (or a zero-sized `T`) yields an empty buffer without
    /// touching the driver allocator.
    ///
    /// # Allocation safety on error
    ///
    /// The returned buffer takes ownership of the device allocation
    /// immediately after `malloc_sync`, before the fallible
    /// `memset_d8_async` enqueue and stream synchronization run. If either
    /// fails, the early return drops the buffer and its `Drop` impl frees the
    /// allocation, so no device memory is leaked.
    pub fn zeroed(stream: &Stream, len: usize) -> Result<Self, DriverError> {
        let ctx = stream.context().clone();
        let num_bytes = allocation_size::<T>(len)?;

        // cuMemAlloc rejects zero-byte requests with CUDA_ERROR_INVALID_VALUE,
        // so represent an empty buffer as a null pointer (Drop skips it).
        if num_bytes == 0 {
            return Ok(Self::empty_copy(len, ctx));
        }

        ctx.bind_to_thread()?;
        let ptr = unsafe { crate::memory::malloc_sync(num_bytes)? };
        // SAFETY: `ptr` was just allocated with `num_bytes` bytes in the
        // stream's context; ownership transfers to `buf` here so any early
        // return below frees it through the buffer's own `Drop`.
        let mut buf = unsafe { Self::from_raw_parts(ptr, len, ctx) };
        unsafe {
            crate::memory::memset_d8_async(buf.ptr, 0, num_bytes, stream.cu_stream())?;
        }
        stream.synchronize()?;
        buf.shared_access_writes = false;
        Ok(buf)
    }

    /// Copies the entire buffer back to the host, returning a `Vec<T>`.
    ///
    /// Synchronizes on `stream` before returning so the host vector is safe
    /// to read immediately.
    pub fn to_host_vec(&self, stream: &Stream) -> Result<Vec<T>, DriverError> {
        let mut host = Vec::<T>::with_capacity(self.len);
        if self.num_bytes() == 0 {
            // SAFETY: empty buffers or zero-sized DeviceCopy values need no bytes.
            unsafe {
                host.set_len(self.len);
            }
            return Ok(host);
        }
        let lease = self.storage.acquire(stream, false, false)?;
        self.storage.context().bind_to_thread()?;
        let submitted = unsafe {
            crate::memory::memcpy_dtoh_async(
                host.as_mut_ptr(),
                self.ptr,
                self.num_bytes(),
                stream.cu_stream(),
            )
        };
        let completed = stream.synchronize();
        if let Err(error) = completed {
            // The driver has not established that it stopped writing host memory.
            std::mem::forget((host, lease));
            return Err(error);
        }
        submitted?;
        unsafe {
            host.set_len(self.len);
        }
        Ok(host)
    }

    /// Copies the buffer contents into an existing host slice.
    ///
    /// Synchronizes on `stream` before returning. Panics if
    /// `dst.len() < self.len()`.
    pub fn copy_to_host(&self, stream: &Stream, dst: &mut [T]) -> Result<(), DriverError> {
        assert!(dst.len() >= self.len(), "destination host slice too small");
        if self.num_bytes() == 0 {
            return Ok(());
        }
        let lease = self.storage.acquire(stream, false, false)?;
        self.storage.context().bind_to_thread()?;
        let submitted = unsafe {
            crate::memory::memcpy_dtoh_async(
                dst.as_mut_ptr(),
                self.ptr,
                self.num_bytes(),
                stream.cu_stream(),
            )
        };
        let completed = stream.synchronize();
        if completed.is_err() {
            std::mem::forget(lease);
        }
        submitted.and(completed)
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
        stream: &Stream,
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
    /// share the same [`Context`].
    ///
    /// # Safety
    ///
    /// This call only enqueues the device-to-host copy on `stream` and
    /// returns; CUDA may still be writing into `dst`'s pinned pointer long
    /// after this function returns. The caller is responsible for ensuring
    /// `dst` is not dropped, freed, read, or aliased until the enqueued copy
    /// has completed, typically after the next [`Stream::synchronize`]
    /// call or a stream-ordered event wait. Dropping `dst` before that
    /// synchronization point calls `cuMemFreeHost` while the in-flight
    /// transfer is still writing the buffer, which is undefined behavior.
    pub unsafe fn copy_to_pinned_host_async(
        &self,
        stream: &Stream,
        dst: &mut PinnedHostBuffer<T>,
    ) -> Result<(), DriverError> {
        assert!(
            dst.len() >= self.len(),
            "destination pinned host buffer too small"
        );
        if dst.context().as_ref() != self.storage.context().as_ref() {
            return Err(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_CONTEXT,
            ));
        }
        if self.num_bytes() == 0 {
            return Ok(());
        }
        self.enqueue_access(stream, false, || unsafe {
            crate::memory::memcpy_dtoh_async(
                dst.as_mut_ptr(),
                self.ptr,
                self.num_bytes(),
                stream.cu_stream(),
            )
        })
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
    /// created it. This checks that `src` and `stream` share the same
    /// CUDA context.
    ///
    /// # Safety
    ///
    /// This call only enqueues the host-to-device copy on `stream` and
    /// returns; CUDA may still be reading from `src`'s pinned pointer long
    /// after this function returns. The caller is responsible for ensuring
    /// `src` is not dropped, freed, mutated, or aliased until the enqueued
    /// copy has completed, typically after the next
    /// [`Stream::synchronize`] call or a stream-ordered event wait.
    /// Dropping `src` before that synchronization point calls
    /// `cuMemFreeHost` while the in-flight transfer is still reading the
    /// buffer, which is undefined behavior.
    pub unsafe fn copy_from_pinned_host_async(
        &mut self,
        stream: &Stream,
        src: &PinnedHostBuffer<T>,
    ) -> Result<(), DriverError> {
        assert!(
            src.len() <= self.len(),
            "source pinned host buffer too large"
        );
        if src.context().as_ref() != self.storage.context().as_ref() {
            return Err(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_CONTEXT,
            ));
        }
        if src.num_bytes() == 0 {
            return Ok(());
        }
        self.enqueue_access(stream, true, || unsafe {
            crate::memory::memcpy_htod_async(
                self.ptr,
                src.as_ptr(),
                src.num_bytes(),
                stream.cu_stream(),
            )
        })
    }

    /// Allocates `len` elements of uninitialized device memory, enqueued on
    /// `stream`.
    ///
    /// Unlike [`Self::zeroed`], no `cuMemsetD8` is enqueued. The contents of
    /// the returned buffer are undefined until the caller writes them.
    ///
    /// The buffer co-owns `stream` (via the `Arc`) so its implicit `Drop` can
    /// retire the stream-ordered allocation after tracked accesses complete.
    /// Unsafe callers must establish allocation and use ordering.
    ///
    /// # Safety
    ///
    /// Reading from the returned buffer before any kernel or memcpy has
    /// written it is undefined behavior.
    pub unsafe fn uninitialized_async(
        stream: &Arc<Stream>,
        len: usize,
    ) -> Result<Self, DriverError> {
        let ctx = stream.context().clone();
        let num_bytes = allocation_size::<T>(len)?;
        if num_bytes == 0 {
            return Ok(Self::empty_copy(len, ctx));
        }

        ctx.bind_to_thread()?;
        let ptr = unsafe { crate::memory::malloc_async(stream.cu_stream(), num_bytes)? };
        let mut buffer = unsafe {
            Self::from_raw_parts_with_dealloc_stream(ptr, len, ctx, Some(stream.clone()))
        };
        buffer.shared_access_writes = false;
        Ok(buffer)
    }

    /// Copies `other` into `self` device-to-device, enqueued on `stream`.
    ///
    /// Panics if `other.len() != self.len()`.
    pub fn copy_from_device_async(
        &mut self,
        other: &DeviceBuffer<T>,
        stream: &Stream,
    ) -> Result<(), DriverError> {
        assert_eq!(self.len, other.len, "device-to-device copy length mismatch");
        if self.num_bytes() == 0 {
            return Ok(());
        }
        if self.storage.context().as_ref() != stream.context().as_ref()
            || other.storage.context().as_ref() != self.storage.context().as_ref()
        {
            return Err(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_CONTEXT,
            ));
        }
        if self.ptr == other.ptr {
            return Ok(());
        }
        if self.ptr < other.ptr + other.num_bytes as u64
            && other.ptr < self.ptr + self.num_bytes as u64
        {
            return Err(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE,
            ));
        }
        self.enqueue_access(stream, true, || {
            other.enqueue_access(stream, false, || unsafe {
                crate::memory::memcpy_dtod_async(
                    self.ptr,
                    other.ptr,
                    self.num_bytes(),
                    stream.cu_stream(),
                )
            })
        })
    }

    /// Copies `src` into `self` host-to-device on `stream` and synchronizes
    /// `stream` before returning.
    ///
    /// The synchronization keeps this safe for borrowed host slices: `src`
    /// may be dropped, reused, or mutated immediately after this function
    /// returns. Panics if `src.len() != self.len()`.
    pub fn copy_from_host(&mut self, stream: &Stream, src: &[T]) -> Result<(), DriverError> {
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
    /// [`Stream::synchronize`] call or a stream-ordered event wait.
    /// Panics if `src.len() != self.len()`.
    pub unsafe fn copy_from_host_async_unchecked(
        &mut self,
        stream: &Stream,
        src: &[T],
    ) -> Result<(), DriverError> {
        assert_eq!(self.len, src.len(), "host-to-device copy length mismatch");
        if self.num_bytes() == 0 {
            return Ok(());
        }
        self.enqueue_access(stream, true, || unsafe {
            crate::memory::memcpy_htod_async(
                self.ptr,
                src.as_ptr(),
                self.num_bytes(),
                stream.cu_stream(),
            )
        })
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
    /// `stream` (for example via [`Stream::join`]). Only the allocation
    /// stream is joined automatically; an unordered third stream still
    /// racing the free is a driver-level use-after-free. This is the same
    /// deferred-use contract as [`Self::copy_from_host_async_unchecked`]:
    /// ordinary [`Drop`] relies on completion leases acquired by safe APIs.
    /// Unsafe raw-pointer launches must retain their own completion leases.
    pub unsafe fn drop_async(self, stream: &Stream) -> Result<(), DriverError> {
        if self.context().as_ref() != stream.context().as_ref() {
            return Err(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_CONTEXT,
            ));
        }
        stream.context().bind_to_thread()?;
        if let Some(allocation_stream) = self.storage.allocation_stream() {
            stream.join(allocation_stream)?;
        }
        crate::completion::retain_until_complete(self.storage, stream)
    }

    /// Zeroes every byte in the buffer asynchronously on `stream`.
    pub fn zero_async(&mut self, stream: &Stream) -> Result<(), DriverError> {
        if self.num_bytes() == 0 {
            return Ok(());
        }
        self.enqueue_access(stream, true, || unsafe {
            crate::memory::memset_d8_async(self.ptr, 0, self.num_bytes(), stream.cu_stream())
        })
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

/// A borrowed contiguous typed view sharing its parent's allocation and access tracker.
pub struct DeviceBufferView<'a, T> {
    storage: crate::RawDeviceBuffer,
    shared_access_writes: bool,
    len: usize,
    _borrow: PhantomData<&'a DeviceBuffer<T>>,
}
impl<T> DeviceBufferView<'_, T> {
    pub fn cu_deviceptr(&self) -> CUdeviceptr {
        self.storage.cu_deviceptr()
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn num_bytes(&self) -> usize {
        self.storage.len_bytes()
    }
    pub fn context(&self) -> &Arc<Context> {
        self.storage.context()
    }
    pub fn storage(&self) -> &crate::RawDeviceBuffer {
        &self.storage
    }
    /// Whether a shared kernel reference may mutate elements through interior
    /// mutability. Representation wrappers and unclassified raw types are
    /// conservative; ordinary DeviceCopy values can share read accesses.
    #[doc(hidden)]
    pub fn shared_access_writes(&self) -> bool {
        self.shared_access_writes
    }
}

impl<'a, T: DeviceCopy> DeviceBufferView<'a, T> {
    /// Borrows initialized elements from shared byte storage without a copy.
    ///
    /// # Safety
    /// The range must contain valid `T` values. All aliases must preserve that
    /// validity and use the same access tracker for every device access.
    pub unsafe fn borrow_storage(
        storage: &'a crate::RawDeviceBuffer,
        offset: usize,
        len: usize,
    ) -> Result<Self, DriverError> {
        let storage = storage.slice_bytes(offset, allocation_size::<T>(len)?)?;
        if !storage.is_empty()
            && !(storage.cu_deviceptr() as usize).is_multiple_of(std::mem::align_of::<T>())
        {
            return Err(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE,
            ));
        }
        Ok(Self {
            storage,
            len,
            shared_access_writes: false,
            _borrow: PhantomData,
        })
    }
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
