/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression test for `Tensor::borrow_raw_parts` (issue #69): a borrowed
//! tensor must NOT free the underlying allocation on drop — the external owner
//! keeps it. If drop freed (as it does for an owned tensor), the second borrow
//! of the same pointer below would be a use-after-free.

use cuda_async::device_buffer::DeviceAllocation;
use cuda_core::sys::CUdeviceptr;
use cuda_core::{free_async, malloc_async, memcpy_dtoh_async, memcpy_htod_async, Stream};
use cutile::prelude::*;
use cutile::tile_kernel::global_policy;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::common;

#[test]
fn borrow_raw_parts_does_not_free_foreign_memory() {
    common::with_test_stack(|| {
        let _policy = global_policy(0);
        let device = cuda_core::Device::new(0).expect("device");
        let stream = device.new_stream().expect("stream");
        let ordinal = device.ordinal();

        const N: usize = 256;
        let bytes = N * std::mem::size_of::<f32>();
        let host_in = [3.0f32; N];

        // Foreign framework owns this allocation for the whole test.
        let mem = unsafe {
            let m = malloc_async(bytes, &stream).expect("malloc");
            memcpy_htod_async(m, host_in.as_ptr(), N, &stream).expect("h2d enqueue");
            stream.synchronize().expect("h2d");
            m
        };

        // Borrow → use → drop, three times over the SAME pointer. An owned
        // tensor would free `mem` on the first drop, making borrows 2 and 3
        // use-after-free; a borrowed tensor leaves `mem` untouched.
        for _ in 0..3 {
            let t =
                unsafe { Tensor::<f32>::borrow_raw_parts(mem, ordinal, vec![N as i32], vec![1]) };
            let out = t
                .to_host_vec()
                .sync_on(&stream)
                .expect("read borrowed tensor");
            assert!(out.iter().all(|&v| v == 3.0), "borrowed data changed");
            // `t` drops here — must be a no-op on `mem`.
        }

        // The foreign memory is still valid and unchanged after all the drops.
        let mut host_out = [0.0f32; N];
        unsafe {
            memcpy_dtoh_async(host_out.as_mut_ptr(), mem, N, &stream).expect("d2h enqueue");
            stream.synchronize().expect("d2h");
        }
        assert!(host_out.iter().all(|&v| v == 3.0));

        // The owner frees its own memory exactly once.
        unsafe {
            free_async(mem, &stream).expect("free enqueue");
            stream.synchronize().expect("free");
        }
    });
}

/// A foreign owner: owns the allocation, frees it and records its drop.
struct Owner {
    dptr: CUdeviceptr,
    len_bytes: usize,
    device_id: usize,
    stream: Arc<Stream>,
    dropped: Arc<AtomicBool>,
}

impl Drop for Owner {
    fn drop(&mut self) {
        // A destructor cannot report a refused free; the flag is what the
        // test observes.
        let _ = unsafe { free_async(self.dptr, &self.stream) };
        self.dropped.store(true, Ordering::SeqCst);
    }
}

// SAFETY: `dptr` is a live `len_bytes` allocation on `device_id` until `Owner` drops.
unsafe impl DeviceAllocation for Owner {
    fn device_ptr(&self) -> CUdeviceptr {
        self.dptr
    }
    fn len_bytes(&self) -> usize {
        self.len_bytes
    }
    fn device_id(&self) -> usize {
        self.device_id
    }
}

#[test]
fn from_foreign_keep_alive_holds_owner() {
    common::with_test_stack(|| {
        let _policy = global_policy(0);
        let device = cuda_core::Device::new(0).expect("device");
        let stream = device.new_stream().expect("stream");

        const N: usize = 256;
        let bytes = N * std::mem::size_of::<f32>();
        let host_in = [7.0f32; N];

        let dropped = Arc::new(AtomicBool::new(false));
        let owner: Arc<dyn DeviceAllocation> = Arc::new(Owner {
            dptr: unsafe {
                let m = malloc_async(bytes, &stream).expect("malloc");
                memcpy_htod_async(m, host_in.as_ptr(), N, &stream).expect("h2d enqueue");
                stream.synchronize().expect("h2d");
                m
            },
            len_bytes: bytes,
            device_id: device.ordinal(),
            stream: stream.clone(),
            dropped: dropped.clone(),
        });

        // The tensor holds a clone of the owner Arc as its liveness token.
        // SAFETY: nothing else touches the bytes after the h2d fill above; the
        // test only reads through cutile from here on.
        let t = Arc::new(unsafe {
            Tensor::<f32>::from_foreign(owner.clone(), vec![N as i32], vec![1])
        });

        // The framework drops its own handle; the owner is NOT dropped, because
        // the tensor still holds it.
        drop(owner);
        assert!(
            !dropped.load(Ordering::SeqCst),
            "owner dropped while a borrowed tensor still holds it"
        );

        // The memory is alive and correct (read by ref so `t` survives).
        let out = (&t).to_host_vec().sync_on(&stream).expect("read");
        assert!(out.iter().all(|&v| v == 7.0));

        // Dropping the tensor releases the last reference → the owner drops (and
        // frees) exactly once.
        drop(t);
        assert!(
            dropped.load(Ordering::SeqCst),
            "owner not dropped after the last borrowing tensor was dropped"
        );
        unsafe { stream.synchronize().expect("drain free") };
    });
}
