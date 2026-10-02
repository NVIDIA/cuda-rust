/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Wrappers around CUDA device pointers: owned (freed on drop), foreign (kept
//! alive by an external owner via a liveness token), and raw-borrowed (an
//! `unsafe` escape that does nothing on drop).

use crate::device_context::with_deallocator_stream;
use cuda_core::free_async;
use cuda_core::sys::CUdeviceptr;
use std::marker::PhantomData;
use std::sync::Arc;

/// A CUDA device allocation owned by an external framework (cudarc, torch, a VMM
/// range, ...).
///
/// A borrowed [`DeviceBuffer`] can hold an `Arc<dyn DeviceAllocation>` as a
/// *liveness token*: while the buffer — and anything derived from it — is alive,
/// the owner's refcount is nonzero, so the memory cannot be freed out from under
/// cutile. This turns liveness from an ongoing caller promise into a fact
/// enforced at construction.
///
/// # Safety
/// Implementors must guarantee that, for as long as the implementor is alive:
/// - `device_ptr()` returns a valid CUDA device pointer on `device_id()`,
/// - the allocation there is at least `len_bytes()` bytes, and
/// - all three getters are **stable**: they return the same values on every
///   call for the implementor's entire lifetime. Consumers read them once at
///   construction and cache the results (`DeviceBuffer::foreign`), so an
///   implementor whose pointer moves would silently diverge from the cached
///   copy. This is the same stability `std`'s allocator traits demand of a
///   currently-allocated block.
///
/// This positional obligation is discharged **once, in the `impl`** — not at
/// each borrow. The temporal obligation (nobody else touches the bytes while
/// cutile work is in flight) cannot live here: it is per-borrow, and is why
/// `Tensor::from_foreign` remains an `unsafe` call.
pub unsafe trait DeviceAllocation: Send + Sync + 'static {
    /// The device pointer of the allocation.
    fn device_ptr(&self) -> CUdeviceptr;
    /// The size of the allocation in bytes.
    fn len_bytes(&self) -> usize;
    /// The device ordinal the allocation lives on.
    fn device_id(&self) -> usize;
}

/// How a [`DeviceBuffer`]'s memory is owned.
enum Owner {
    /// cutile allocated the memory; free it asynchronously on drop.
    Owned,
    /// A raw, unowned borrow (via `borrowed_from_raw_parts`); do nothing on drop.
    /// Liveness is the caller's unverified obligation.
    Borrowed,
    /// A foreign allocation kept alive by its owner; do nothing on drop, and
    /// hold the owner so the memory provably outlives this buffer. The `Arc` is
    /// held purely for that RAII liveness effect — never read directly.
    Foreign(#[allow(dead_code)] Arc<dyn DeviceAllocation>),
}

/// A non-owning, copyable handle to a typed device memory address.
#[derive(Debug, Copy, Clone)]
pub struct DevicePointer<T> {
    dtype: PhantomData<T>,
    pub dptr: CUdeviceptr,
}

unsafe impl<T> Send for DevicePointer<T> {}

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

/// A type-erased handle to a CUDA device memory allocation.
///
/// `DeviceBuffer` manages a raw byte buffer on the GPU. It stores only the device pointer,
/// byte length, and device id — all type information lives in higher-level wrappers
/// such as `Tensor<T>`.
///
/// An *owned* buffer (the default) is freed asynchronously on drop. A *foreign*
/// buffer (see [`foreign`](Self::foreign)) wraps memory owned by an external
/// allocator (cudarc, torch, a VMM range, ...), holds a liveness token, and is a
/// no-op on drop. A raw *borrowed* buffer
/// (see [`borrowed_from_raw_parts`](Self::borrowed_from_raw_parts)) is the same
/// no-op-on-drop but without a token — an `unsafe` escape whose liveness is the
/// caller's responsibility.
pub struct DeviceBuffer {
    device_id: usize,
    cudptr: CUdeviceptr,
    len: usize,
    owner: Owner,
}

impl std::fmt::Debug for DeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match self.owner {
            Owner::Owned => "owned",
            Owner::Borrowed => "borrowed",
            Owner::Foreign(_) => "foreign",
        };
        f.debug_struct("DeviceBuffer")
            .field("device_id", &self.device_id)
            .field("cudptr", &self.cudptr)
            .field("len", &self.len)
            .field("owner", &kind)
            .finish()
    }
}

unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        // Only owned buffers are freed here. Borrowed/foreign buffers wrap memory
        // owned elsewhere; freeing it would be a double free (or invalid, e.g.
        // for a VMM-mapped range). A `Foreign` owner's `Arc` drops with `self`.
        if !matches!(self.owner, Owner::Owned) {
            return;
        }
        // A destructor has no way to report failure, and a panic here would
        // abort the process if this drop runs during an unwind. Both a missing
        // device context and a refused `cuMemFreeAsync` are therefore ignored:
        // the allocation leaks until the primary context is released, which
        // is the least bad outcome for memory the driver would not free.
        // SAFETY: the pointer was allocated stream-ordered by this crate, and
        // the driver completes queued work on the deallocator stream before
        // the memory is handed out again.
        let _ = unsafe {
            with_deallocator_stream(self.device_id, |stream| free_async(self.cudptr, stream))
        };
    }
}

impl DeviceBuffer {
    /// Constructs an owned `DeviceBuffer` from a raw device pointer, byte length, and device id.
    ///
    /// The buffer takes ownership of the allocation and frees it asynchronously on drop.
    ///
    /// # Safety
    /// The caller must ensure `dptr` points to a valid device allocation of at least
    /// `len_bytes` bytes that is safe to free via the deallocator stream.
    pub unsafe fn from_raw_parts(dptr: CUdeviceptr, len_bytes: usize, device_id: usize) -> Self {
        Self {
            cudptr: dptr,
            len: len_bytes,
            device_id,
            owner: Owner::Owned,
        }
    }

    /// Wraps `len_bytes` of a foreign allocation, holding `owner` alive so the
    /// memory outlives this buffer. Dropping it does **not** free the memory.
    ///
    /// `len_bytes` is the borrowing view's **logical** size (what the tensor
    /// invariant checks), not the owner's allocation size — the owner may be
    /// larger, and the addressable-extent-vs-owner check happens in
    /// `Tensor::from_foreign` before this is called. Liveness is enforced by
    /// `owner`'s refcount; pointer validity was asserted once when `owner`'s
    /// type implemented [`DeviceAllocation`].
    pub fn foreign(owner: Arc<dyn DeviceAllocation>, len_bytes: usize) -> Self {
        Self {
            cudptr: owner.device_ptr(),
            len: len_bytes,
            device_id: owner.device_id(),
            owner: Owner::Foreign(owner),
        }
    }

    /// Constructs a borrowed `DeviceBuffer` that wraps memory owned by an external
    /// allocator. Dropping it does **not** free the allocation.
    ///
    /// # Safety
    /// This borrows, rather than takes ownership of, `dptr`; dropping the buffer
    /// never frees it. The caller must ensure `dptr` points to a valid device
    /// allocation of at least `len_bytes` bytes, and must keep that allocation
    /// mapped — never freed, reallocated, or resized — for the entire lifetime of
    /// this buffer and every value derived from it. When wrapped in a tensor used
    /// as a mutable output, the owner must additionally not access the same memory
    /// concurrently; see `Tensor::borrow_raw_parts` in the `cutile` crate for the
    /// full obligations.
    pub unsafe fn borrowed_from_raw_parts(
        dptr: CUdeviceptr,
        len_bytes: usize,
        device_id: usize,
    ) -> Self {
        Self {
            cudptr: dptr,
            len: len_bytes,
            device_id,
            owner: Owner::Borrowed,
        }
    }
    /// Returns whether the allocation is empty (zero bytes).
    pub fn is_empty(&self) -> bool {
        self.len_bytes() == 0
    }
    /// Returns the byte length of the allocation.
    pub fn len_bytes(&self) -> usize {
        self.len
    }
    /// Returns the raw CUDA device pointer.
    pub fn cu_deviceptr(&self) -> CUdeviceptr {
        self.cudptr
    }
    /// Returns the device id this allocation belongs to.
    pub fn device_id(&self) -> usize {
        self.device_id
    }
}
