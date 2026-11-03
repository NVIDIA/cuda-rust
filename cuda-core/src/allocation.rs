/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared allocation ownership, byte views, and in-flight access tracking.

use crate::{Context, DriverError, Stream};
use cuda_bindings::CUdeviceptr;
use std::sync::{Arc, Mutex};

/// A foreign allocation whose owner keeps both its memory and CUDA context alive.
///
/// # Safety
/// The pointer, byte extent, and context must remain stable and valid for the
/// owner's lifetime. The pointer must address at least `len_bytes` bytes in that
/// context. Per-operation exclusion from foreign accesses remains the obligation
/// of callers constructing typed views over this memory.
pub unsafe trait DeviceAllocation: Send + Sync + 'static {
    fn device_ptr(&self) -> CUdeviceptr;
    fn len_bytes(&self) -> usize;
    fn context(&self) -> Arc<Context>;
    fn device_id(&self) -> usize {
        self.context().ordinal()
    }
}

struct Access {
    id: u64,
    stream: usize,
    write: bool,
    recording: bool,
    completion: Option<Arc<crate::Event>>,
}

#[derive(Default)]
struct Accesses {
    next_id: u64,
    active: Vec<Access>,
}

impl Accesses {
    fn register(
        &mut self,
        stream: usize,
        write: bool,
        recording: bool,
    ) -> Result<u64, DriverError> {
        // A synchronous wait on another thread may have completed a retired
        // operation before the worker got to its lease. Do not report a stale
        // conflict in that case. Errors leave the exclusion intact.
        self.active
            .retain(|a| !matches!(a.completion.as_ref().map(|e| e.query()), Some(Ok(true))));
        if !recording
            && self
                .active
                .iter()
                .any(|a| !a.recording && a.stream != stream && (write || a.write))
        {
            return Err(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_NOT_READY,
            ));
        }
        let id = self.next_id.checked_add(1).ok_or(DriverError(
            cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE,
        ))?;
        self.next_id = id;
        self.active.push(Access {
            id,
            stream,
            write,
            recording,
            completion: None,
        });
        Ok(id)
    }
}

enum Owner {
    Synchronous,
    StreamOrdered(Arc<Stream>),
    Borrowed,
    Foreign(#[allow(dead_code)] Arc<dyn DeviceAllocation>),
}

struct Allocation {
    ptr: CUdeviceptr,
    bytes: usize,
    context: Arc<Context>,
    owner: Owner,
    accesses: Mutex<Accesses>,
}

impl Drop for Allocation {
    fn drop(&mut self) {
        if self.ptr == 0 || matches!(self.owner, Owner::Borrowed | Owner::Foreign(_)) {
            return;
        }
        let ptr = self.ptr;
        let context = self.context.clone();
        let allocation_stream = match &self.owner {
            Owner::StreamOrdered(s) => Some(s.clone()),
            _ => None,
        };
        // All safe users hold an independent access lease until completion.
        // Driver teardown is retired on a worker, never on the dropping thread.
        crate::completion::defer(move || {
            if let Err(error) = context.bind_to_thread() {
                context.record_err(Err::<(), _>(error));
                std::mem::forget((context, allocation_stream));
                return;
            }
            let result = if let Some(stream) = &allocation_stream {
                // SAFETY: the allocation is no longer accessed and the retained
                // stream is the one on which its allocation became valid.
                unsafe { crate::memory::free_async(ptr, stream.cu_stream()) }
            } else {
                // SAFETY: ownership of this synchronous allocation is unique.
                unsafe { crate::memory::free_sync(ptr) }
            };
            if let Err(error) = result {
                context.record_err(Err::<(), _>(error));
                std::mem::forget((context, allocation_stream));
            }
        });
    }
}

/// A byte view sharing an allocation owner and access tracker with every derived
/// typed buffer, tensor, and sliced view. Cloning does not copy device memory.
#[derive(Clone)]
pub struct RawDeviceBuffer {
    allocation: Arc<Allocation>,
    // Only user-visible views share this token. Completion leases retain the
    // allocation directly, so creating/dropping them cannot affect uniqueness.
    views: Arc<()>,
    offset: usize,
    len: usize,
}

impl std::fmt::Debug for RawDeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawDeviceBuffer")
            .field("ptr", &self.cu_deviceptr())
            .field("len", &self.len)
            .field("context", &self.allocation.context)
            .finish_non_exhaustive()
    }
}

impl RawDeviceBuffer {
    fn new(ptr: CUdeviceptr, bytes: usize, context: Arc<Context>, owner: Owner) -> Self {
        assert!(
            ptr.checked_add(bytes as u64).is_some(),
            "device allocation extent overflows"
        );
        Self {
            allocation: Arc::new(Allocation {
                ptr,
                bytes,
                context,
                owner,
                accesses: Mutex::default(),
            }),
            views: Arc::new(()),
            offset: 0,
            len: bytes,
        }
    }

    /// Adopts a synchronous CUDA allocation.
    ///
    /// # Safety
    /// `ptr` must uniquely own `bytes` bytes allocated by `cuMemAlloc` in
    /// `context`, with no outstanding untracked accesses.
    pub unsafe fn from_raw_parts(ptr: CUdeviceptr, bytes: usize, context: Arc<Context>) -> Self {
        Self::new(ptr, bytes, context, Owner::Synchronous)
    }

    /// Adopts a stream-ordered CUDA allocation.
    ///
    /// # Safety
    /// `ptr` must uniquely own `bytes` bytes from `cuMemAllocAsync` or its pool
    /// variant on `stream`. The caller must establish allocation ordering for
    /// every use and retain the allocation through outstanding work.
    pub unsafe fn from_raw_parts_async(
        ptr: CUdeviceptr,
        bytes: usize,
        stream: Arc<Stream>,
    ) -> Self {
        Self::new(
            ptr,
            bytes,
            stream.context().clone(),
            Owner::StreamOrdered(stream),
        )
    }

    /// Borrows externally managed memory without taking ownership.
    ///
    /// # Safety
    /// Memory and context must outlive every derived view and queued use. No
    /// external access may conflict with accesses through this allocation.
    pub unsafe fn borrowed_from_raw_parts(
        ptr: CUdeviceptr,
        bytes: usize,
        context: Arc<Context>,
    ) -> Self {
        Self::new(ptr, bytes, context, Owner::Borrowed)
    }

    /// Retains an external owner and limits the logical view to `bytes` bytes.
    /// No CUDA calls are made, permitting metadata validation before submission.
    pub fn foreign(owner: Arc<dyn DeviceAllocation>, bytes: usize) -> Self {
        assert!(
            bytes <= owner.len_bytes(),
            "view exceeds foreign allocation"
        );
        Self::new(
            owner.device_ptr(),
            bytes,
            owner.context(),
            Owner::Foreign(owner),
        )
    }

    pub fn cu_deviceptr(&self) -> CUdeviceptr {
        self.allocation.ptr + self.offset as u64
    }
    pub fn len_bytes(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn context(&self) -> &Arc<Context> {
        &self.allocation.context
    }
    pub(crate) fn allocation_stream(&self) -> Option<&Arc<Stream>> {
        match &self.allocation.owner {
            Owner::StreamOrdered(stream) => Some(stream),
            _ => None,
        }
    }
    pub fn device_id(&self) -> usize {
        self.context().ordinal()
    }
    /// Identity of the owner/access tracker, unchanged across byte views.
    pub fn allocation_id(&self) -> usize {
        Arc::as_ptr(&self.allocation) as usize
    }

    /// Creates a bounds-checked view without changing allocation identity.
    pub fn slice_bytes(&self, offset: usize, bytes: usize) -> Result<Self, DriverError> {
        if offset.checked_add(bytes).is_none_or(|end| end > self.len) {
            return Err(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE,
            ));
        }
        Ok(Self {
            allocation: self.allocation.clone(),
            views: self.views.clone(),
            offset: self.offset + offset,
            len: bytes,
        })
    }

    /// Whether only this user-visible view owns the allocation. Completion leases
    /// do not count as user aliases; they still enforce in-flight exclusion.
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.views) == 1
    }

    /// Acquires an allocation-wide access, rejecting cross-stream conflicts.
    /// The returned lease must be retained until the submitted work completes.
    pub fn acquire(
        &self,
        stream: &Stream,
        write: bool,
        recording: bool,
    ) -> Result<AllocationLease, DriverError> {
        if self.context().as_ref() != stream.context().as_ref() {
            return Err(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_CONTEXT,
            ));
        }
        self.acquire_on_stream(stream.cu_stream() as usize, write, recording)
    }

    /// Registers bookkeeping for a known stream handle. This does not submit GPU
    /// work or grant permission to dereference pointers. Runtime adapters must
    /// separately validate the stream's context before submitting work.
    #[doc(hidden)]
    pub fn acquire_on_stream(
        &self,
        stream: usize,
        write: bool,
        recording: bool,
    ) -> Result<AllocationLease, DriverError> {
        let mut accesses = self
            .allocation
            .accesses
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let id = accesses.register(stream, write, recording)?;
        Ok(AllocationLease {
            allocation: self.allocation.clone(),
            id,
        })
    }

    pub(crate) fn into_owned_parts(
        self,
    ) -> (CUdeviceptr, usize, Arc<Context>, Option<Arc<Stream>>) {
        assert_eq!(self.offset, 0, "cannot detach a sliced allocation");
        assert_eq!(
            self.len, self.allocation.bytes,
            "cannot detach a partial allocation"
        );
        let mut allocation = Arc::try_unwrap(self.allocation)
            .unwrap_or_else(|_| panic!("cannot detach a shared or in-flight allocation"));
        let stream = match &allocation.owner {
            Owner::Synchronous => None,
            Owner::StreamOrdered(stream) => Some(stream.clone()),
            _ => panic!("cannot detach an allocation owned externally"),
        };
        let ptr = std::mem::replace(&mut allocation.ptr, 0);
        (ptr, allocation.bytes, allocation.context.clone(), stream)
    }
}

/// Keeps an allocation alive and excludes conflicting accesses until dropped.
/// A runtime must hold this beyond enqueue, until completion has been proven.
pub struct AllocationLease {
    allocation: Arc<Allocation>,
    id: u64,
}

impl Drop for AllocationLease {
    fn drop(&mut self) {
        let mut accesses = self
            .allocation
            .accesses
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(pos) = accesses.active.iter().position(|a| a.id == self.id) {
            accesses.active.swap_remove(pos);
        }
    }
}

impl AllocationLease {
    /// Transfers this lease to completion-bound retirement after submission.
    pub fn retain_until_complete(self, stream: &Stream) -> Result<(), DriverError> {
        let event = match stream.record_event(None) {
            Ok(event) => Arc::new(event),
            Err(error) => {
                std::mem::forget(self);
                return Err(error);
            }
        };
        {
            let mut accesses = self
                .allocation
                .accesses
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if let Some(access) = accesses.active.iter_mut().find(|a| a.id == self.id) {
                access.completion = Some(event.clone());
            }
        }
        crate::completion::retain_until_event(self, event);
        Ok(())
    }
}

impl std::fmt::Debug for AllocationLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AllocationLease")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn metadata() -> RawDeviceBuffer {
        // SAFETY: bookkeeping only. These borrowed handles are never passed to
        // CUDA, dereferenced, or freed by their destructors.
        unsafe {
            RawDeviceBuffer::borrowed_from_raw_parts(
                0x1000,
                128,
                Context::borrow_raw(0x2000 as *mut _, 0, 0),
            )
        }
    }

    #[test]
    fn uniqueness_counts_byte_views_but_not_completion_leases() {
        let raw = metadata();
        let view = raw.slice_bytes(16, 32).unwrap();
        let clone = raw.clone();
        let lease = raw.acquire_on_stream(1, true, false).unwrap();
        assert!(!raw.is_unique());
        drop(view);
        assert!(!raw.is_unique());
        drop(clone);
        assert!(raw.is_unique());
        assert!(raw.acquire_on_stream(2, false, false).is_err());
        drop(lease);
        assert!(raw.acquire_on_stream(2, false, false).is_ok());
    }

    #[test]
    fn concurrent_lease_retirement_cannot_change_user_uniqueness() {
        let raw = metadata();
        let stop = AtomicBool::new(false);
        let mut false_negatives = 0;
        std::thread::scope(|scope| {
            scope.spawn(|| {
                while !stop.load(Ordering::Relaxed) {
                    drop(raw.acquire_on_stream(1, false, false).unwrap());
                }
            });
            for _ in 0..100_000 {
                false_negatives += usize::from(!raw.is_unique());
            }
            stop.store(true, Ordering::Relaxed);
        });
        assert_eq!(false_negatives, 0);
    }
}
