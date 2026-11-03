/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::{Context, DeviceBuffer, RawDeviceBuffer};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

fn metadata(context: usize) -> RawDeviceBuffer {
    // SAFETY: these fake handles are used only for bookkeeping; no CUDA API
    // sees them and borrowed allocation/context destruction performs no calls.
    unsafe {
        RawDeviceBuffer::borrowed_from_raw_parts(
            0x1000,
            128,
            Context::borrow_raw(context as *mut _, 0, 0),
        )
    }
}

#[test]
fn byte_views_share_extent_identity_and_access_exclusion() {
    let raw = metadata(0x2000);
    let view = raw.slice_bytes(16, 32).unwrap();
    assert_eq!(view.cu_deviceptr(), raw.cu_deviceptr() + 16);
    assert_eq!(view.allocation_id(), raw.allocation_id());
    assert!(raw.slice_bytes(120, 9).is_err());
    assert!(raw.slice_bytes(usize::MAX, 1).is_err());
    assert!(!raw.is_unique());
    let read = raw.acquire_on_stream(1, false, false).unwrap();
    let second_read = view.acquire_on_stream(2, false, false).unwrap();
    assert!(view.acquire_on_stream(2, true, false).is_err());
    drop((read, second_read));
    let write = view.acquire_on_stream(2, true, false).unwrap();
    assert!(raw.acquire_on_stream(1, false, false).is_err());
    assert!(raw.acquire_on_stream(2, true, false).is_ok());
    drop(view);
    assert!(raw.is_unique(), "completion leases are not user aliases");
    drop(write);
}

#[test]
fn same_ordinal_does_not_mean_same_context() {
    let raw = metadata(0x2000);
    let other = metadata(0x3000);
    assert_eq!(raw.device_id(), other.device_id());
    assert_ne!(raw.context(), other.context());
    // SAFETY: this fake borrowed stream is never submitted or destroyed: forget
    // it after testing the context check, before Drop could query the driver.
    let stream = unsafe { cuda_core::Stream::borrow_raw(0x4000 as *mut _, other.context()) };
    let result = raw.acquire(&stream, false, false);
    std::mem::forget(stream);
    assert!(result.is_err());
}

struct Gate(mpsc::Sender<()>);
impl Gate {
    fn arm(stream: &cuda_core::Stream) -> Self {
        let (tx, rx) = mpsc::channel();
        stream
            .launch_host_function(move || {
                let _ = rx.recv_timeout(Duration::from_secs(10));
            })
            .unwrap();
        Self(tx)
    }
}
impl Drop for Gate {
    fn drop(&mut self) {
        let _ = self.0.send(());
    }
}

#[test]
fn pending_copy_excludes_other_stream_and_outlives_dropped_buffer() {
    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let other = context.new_stream().unwrap();
    let mut buffer = DeviceBuffer::from_host(&stream, &[7u32; 32]).unwrap();
    let storage = buffer.storage().clone();
    let gate = Gate::arm(&stream);
    buffer.zero_async(&stream).unwrap();
    assert!(storage.acquire(&other, false, false).is_err());
    let before = Instant::now();
    drop(buffer);
    assert!(
        before.elapsed() < Duration::from_secs(2),
        "buffer drop waited for GPU work"
    );
    assert!(storage.acquire(&other, false, false).is_err());
    drop(gate);
    stream.synchronize().unwrap();
    // SAFETY: the zero fill completed, u32 accepts every representation, and
    // this test is the only remaining user of the allocation.
    let buffer = unsafe { DeviceBuffer::<u32>::from_storage_copy(storage, 32).unwrap() };
    assert_eq!(buffer.to_host_vec(&other).unwrap(), vec![0; 32]);
}

#[test]
fn stream_drop_is_nonblocking_and_holds_context_until_completion() {
    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let gate = Gate::arm(&stream);
    let weak = Arc::downgrade(&context);
    drop(context);
    let before = Instant::now();
    drop(stream);
    assert!(
        before.elapsed() < Duration::from_secs(2),
        "stream drop waited for GPU work"
    );
    assert!(weak.upgrade().is_some());
    drop(gate);
    let deadline = Instant::now() + Duration::from_secs(5);
    while weak.upgrade().is_some() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(
        weak.upgrade().is_none(),
        "retirement leaked an idle stream/context"
    );
}

#[test]
fn fresh_thread_allocation_binds_its_context() {
    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    std::thread::spawn(move || {
        let buffer = DeviceBuffer::from_host(&stream, &[11u32, 29]).unwrap();
        assert_eq!(buffer.to_host_vec(&stream).unwrap(), [11, 29]);
    })
    .join()
    .unwrap();
}

#[test]
fn empty_and_peeled_buffers_preserve_shared_read_access() {
    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let empty = DeviceBuffer::<u32>::from_host(&stream, &[]).unwrap();
    assert!(!empty.shared_access_writes());
    let zeroed_empty = DeviceBuffer::<u32>::zeroed(&stream, 0).unwrap();
    assert!(!zeroed_empty.shared_access_writes());

    let words = DeviceBuffer::from_host(&stream, &[1u32, 2, 3]).unwrap();
    let wrapped = words.wrap_elem::<std::num::Wrapping<u32>>();
    assert!(wrapped.shared_access_writes());
    let peeled = wrapped.peel_elem::<u32>();
    assert!(!peeled.shared_access_writes());
}
