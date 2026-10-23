/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#![cfg(feature = "async")]

use cuda_async::{
    device_operation::DeviceOperation, launch::AsyncKernelLaunchBuilder,
    scheduling_policies::SingleStream,
};
use cuda_core::{Context, DeviceBuffer, LaunchConfig};
use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, mpsc},
    task::{Context as TaskContext, Waker},
    time::Duration,
};

const PTX: &str = ".version 7.0\n.target sm_52\n.address_size 64\n.visible .entry use_slice(.param .u64 ptr, .param .u64 len) { ret; }";

struct Gate(mpsc::Sender<()>);
impl Drop for Gate {
    fn drop(&mut self) {
        let _ = self.0.send(());
    }
}

fn pending_launch_keeps_access(forget: bool, atomic: bool) {
    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let other = context.new_stream().unwrap();
    let function = Arc::new(
        context
            .load_module_from_ptx_src(PTX)
            .unwrap()
            .load_function("use_slice")
            .unwrap(),
    );
    let buffer = DeviceBuffer::from_host(&stream, &[1u32; 32]).unwrap();
    let storage = buffer.storage().clone();
    let mut builder = AsyncKernelLaunchBuilder::new(function);
    if atomic {
        let buffer = buffer.wrap_elem::<cuda_device::atomic::DeviceAtomicU32>();
        cuda_host::push_async_read_only_device_slice(&mut builder, &buffer);
        drop(buffer);
    } else {
        let mut buffer = buffer;
        cuda_host::push_async_writable_device_slice(&mut builder, &mut buffer);
        drop(buffer);
    }
    let (tx, rx) = mpsc::channel();
    stream
        .launch_host_function(move || {
            let _ = rx.recv_timeout(Duration::from_secs(10));
        })
        .unwrap();
    let gate = Gate(tx);
    // SAFETY: PTX declares exactly the pointer/length packet above and no
    // memory accesses. The descriptors still model a writable kernel argument,
    // so this test can observe exclusion without deliberately racing GPU memory.
    let launch = unsafe {
        builder.finalize_unchecked(LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        })
    };
    let mut future = launch
        .schedule(&SingleStream::from_stream(stream.clone()))
        .unwrap();
    assert!(
        Pin::new(&mut future)
            .poll(&mut TaskContext::from_waker(Waker::noop()))
            .is_pending()
    );
    if forget {
        std::mem::forget(future);
    } else {
        drop(future);
    }
    assert!(
        storage.acquire(&other, false, false).is_err(),
        "early future release lost a writable argument's access lease"
    );
    drop(gate);
    stream.synchronize().unwrap();
    if forget {
        assert!(
            storage.acquire(&other, false, false).is_err(),
            "a forgotten submission conservatively retains its access leases"
        );
    } else {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while storage.acquire(&other, false, false).is_err() {
            assert!(
                std::time::Instant::now() < deadline,
                "completed work failed to release exclusion"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

#[test]
#[ignore = "requires a live CUDA GPU"]
fn dropped_borrowed_launch_retains_buffers_and_accesses() {
    pending_launch_keeps_access(false, false);
}
#[test]
#[ignore = "requires a live CUDA GPU"]
fn forgotten_borrowed_launch_retains_buffers_and_accesses() {
    pending_launch_keeps_access(true, false);
}
#[test]
#[ignore = "requires a live CUDA GPU"]
fn shared_atomic_arguments_exclude_ordinary_reads_after_drop() {
    pending_launch_keeps_access(false, true);
}

#[test]
#[ignore = "requires a live CUDA GPU"]
fn device_box_preserves_concurrent_read_permissions() {
    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let other = context.new_stream().unwrap();
    let function = Arc::new(
        context
            .load_module_from_ptx_src(PTX)
            .unwrap()
            .load_function("use_slice")
            .unwrap(),
    );
    let buffer = cuda_async::device_box::DeviceBox::from_buffer(
        DeviceBuffer::from_host(&stream, &[1u32; 32]).unwrap(),
    );
    let raw_ptr = unsafe { cuda_core::memory::malloc_async(stream.cu_stream(), 32 * 4).unwrap() };
    stream.synchronize().unwrap();
    let raw_box = unsafe {
        cuda_async::device_box::DeviceBox::<[u32]>::from_raw_parts(raw_ptr, 32, stream.clone())
    };
    assert!(!raw_box.shared_access_writes());
    let make_launch = || {
        let mut builder = AsyncKernelLaunchBuilder::new(function.clone());
        cuda_host::push_async_read_only_device_slice(&mut builder, &buffer);
        cuda_host::push_async_read_only_device_slice(&mut builder, &raw_box);
        // SAFETY: the PTX's pointer/length signature matches the first descriptor;
        // the no-op kernel never dereferences the modeled shared slices.
        unsafe { builder.finalize_unchecked(LaunchConfig::for_num_elems(32)) }
    };
    let (tx, rx) = mpsc::channel();
    stream
        .launch_host_function(move || {
            let _ = rx.recv_timeout(Duration::from_secs(10));
        })
        .unwrap();
    let gate = Gate(tx);
    let mut first = make_launch()
        .schedule(&SingleStream::from_stream(stream.clone()))
        .unwrap();
    assert!(
        Pin::new(&mut first)
            .poll(&mut TaskContext::from_waker(Waker::noop()))
            .is_pending()
    );
    // A read on another stream must succeed while the first read is gated.
    // Treating DeviceBox's ordinary u32 elements as interior-mutable would fail.
    make_launch().sync_on(&other).unwrap();
    drop(gate);
    cuda_async::futures::executor::block_on(first).unwrap();
}
