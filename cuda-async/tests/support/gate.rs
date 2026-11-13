/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! A mapped-memory GPU gate for completion and cancellation tests.
use cuda_async::device_context::with_device;
use cuda_async::error::DeviceError;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// A host-controlled barrier at the head of a stream.
///
/// [`arm`](Gate::arm) enqueues a `cuStreamWaitValue32` on the gate's word, a
/// page of pinned, device-mapped host memory; work submitted to the stream
/// after that stays queued until [`open`](Gate::open) stores the release
/// value. This is the host-mapped-flag mechanism the crate's completion
/// reactor uses, in the other direction.
pub struct Gate {
    host: *mut u32,
    dptr: cuda_bindings::CUdeviceptr,
}

// SAFETY: `host` is pinned, device-mapped memory owned by the gate for its
// whole lifetime; the host side only touches it through atomics.
unsafe impl Send for Gate {}
unsafe impl Sync for Gate {}

impl Gate {
    /// Allocates a closed gate. Requires a current CUDA context on the
    /// calling thread.
    pub fn closed() -> Arc<Self> {
        let mut host = MaybeUninit::uninit();
        let flags =
            cuda_bindings::CU_MEMHOSTALLOC_PORTABLE | cuda_bindings::CU_MEMHOSTALLOC_DEVICEMAP;
        let code = unsafe {
            cuda_bindings::cuMemHostAlloc(host.as_mut_ptr(), std::mem::size_of::<u32>(), flags)
        };
        assert_eq!(code, 0, "cuMemHostAlloc failed: {code}");
        let host = unsafe { host.assume_init() } as *mut u32;
        unsafe { AtomicU32::from_ptr(host) }.store(0, Ordering::SeqCst);
        let mut dptr = MaybeUninit::uninit();
        let code =
            unsafe { cuda_bindings::cuMemHostGetDevicePointer_v2(dptr.as_mut_ptr(), host as _, 0) };
        assert_eq!(code, 0, "cuMemHostGetDevicePointer failed: {code}");
        Arc::new(Self {
            host,
            dptr: unsafe { dptr.assume_init() },
        })
    }

    /// Enqueues the wait: everything submitted to `stream` after this call
    /// stays queued until the gate is opened.
    pub fn arm(&self, stream: cuda_bindings::CUstream) -> Result<(), DeviceError> {
        // CUDA driver flag bindings have platform-dependent integer types, so
        // the FFI call casts the flag as `_`.
        let code = unsafe {
            cuda_bindings::cuStreamWaitValue32_v2(
                stream,
                self.dptr,
                1,
                cuda_bindings::CUstreamWaitValue_flags_enum_CU_STREAM_WAIT_VALUE_GEQ as _,
            )
        };
        if code != cuda_bindings::cudaError_enum_CUDA_SUCCESS {
            return Err(DeviceError::Internal(format!(
                "cuStreamWaitValue32 failed: {code}"
            )));
        }
        Ok(())
    }

    /// Releases every stream waiting on the gate.
    fn open(&self) {
        unsafe { AtomicU32::from_ptr(self.host) }.store(1, Ordering::SeqCst);
    }

    /// Keep the gate closed until this guard is dropped. The deadline only
    /// rescues a stuck test; a watchdog release is always a test failure.
    pub fn hold(self: &Arc<Self>) -> Hold {
        let (release, receive) = std::sync::mpsc::channel();
        let gate = self.clone();
        let worker = std::thread::spawn(move || {
            let explicit = receive.recv_timeout(Duration::from_secs(60)).is_ok();
            gate.open();
            explicit
        });
        Hold {
            release,
            worker: Some(worker),
        }
    }
}

impl Drop for Gate {
    fn drop(&mut self) {
        // Every wait on the gate has passed by now: the test keeps its own
        // handle until the gated stream has been synchronized. The last owner
        // may be the opener thread, which never bound a context.
        let _ = with_device(0, |device| device.bind_to_thread());
        let code = unsafe { cuda_bindings::cuMemFreeHost(self.host as _) };
        if !std::thread::panicking() {
            assert_eq!(code, 0, "cuMemFreeHost failed: {code}");
        }
    }
}

/// Releases the GPU gate on normal completion and unwinding.
/// An automatic watchdog release must never count as successful ordering.
pub struct Hold {
    release: std::sync::mpsc::Sender<()>,
    worker: Option<std::thread::JoinHandle<bool>>,
}

impl Drop for Hold {
    fn drop(&mut self) {
        let _ = self.release.send(());
        let explicit = self.worker.take().unwrap().join().unwrap_or(false);
        if !std::thread::panicking() {
            assert!(
                explicit,
                "GPU gate watchdog expired: operation did not return before release"
            );
        }
    }
}
