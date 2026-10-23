/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#![cfg(feature = "async")]

use cuda_async::simt::{
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
        .schedule(&SingleStream {
            stream: Some(stream.clone()),
        })
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
    assert!(
        storage.acquire(&other, false, false).is_ok(),
        "completed work must release exclusion"
    );
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
