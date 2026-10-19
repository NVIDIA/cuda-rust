/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tests for the thread-local execution lock: `then` rejects nested
//! execution, `then_unchecked` releases the lock for its closure and takes it
//! back afterwards, and a panic inside an executing region releases the lock
//! instead of poisoning the thread. Requires a GPU (stream creation and
//! synchronization go through the driver).

use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_context::init_device_contexts;
use cuda_async::device_operation::{value, DeviceOp, Value};
use cuda_async::error::DeviceError;
use std::future::IntoFuture;

/// Run `f` on a fresh thread so the thread-local execution lock and
/// `DEVICE_CONTEXTS` start clean.
fn on_fresh_thread<F: FnOnce() + Send + 'static>(f: F) {
    std::thread::spawn(f).join().expect("test thread panicked");
}

fn is_non_reentrant_error<T: std::fmt::Debug>(result: &Result<T, DeviceError>) -> bool {
    matches!(result, Err(DeviceError::Internal(msg)) if msg.contains("non-reentrant"))
}

#[test]
fn then_closure_cannot_nest_execution() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");
        let out = value(1)
            .then(|x| {
                let nested = value(2).sync();
                assert!(
                    is_non_reentrant_error(&nested),
                    "nested sync inside `then` must hit the lock, got {nested:?}"
                );
                value(x)
            })
            .sync()
            .expect("outer chain failed");
        assert_eq!(out, 1);
    });
}

#[test]
fn then_unchecked_closure_can_nest_sync_sync_on_and_await() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");
        let device = cuda_core::Device::new(0).unwrap();
        let other_stream = device.new_stream().unwrap();

        let chain = unsafe {
            value(1).then_unchecked(move |x| {
                let a = value(2).sync().expect("nested sync must succeed");
                let b = value(3)
                    .sync_on(&other_stream)
                    .expect("nested sync_on must succeed");
                let c = futures::executor::block_on(value(4).into_future())
                    .expect("nested await must succeed");
                value(x + a + b + c)
            })
        };
        assert_eq!(chain.sync().expect("chain failed"), 10);
    });
}

/// The lock is re-taken after the unchecked closure: a plain `then` later in
/// the same chain is still protected.
#[test]
fn lock_is_restored_after_then_unchecked_closure() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");
        let chain = unsafe { value(1).then_unchecked(|x| value(x + 1)) }.then(|x| {
            let nested = value(0).sync();
            assert!(
                is_non_reentrant_error(&nested),
                "lock must be held again after then_unchecked, got {nested:?}"
            );
            value(x)
        });
        assert_eq!(chain.sync().expect("chain failed"), 2);
    });
}

/// The unchecked closure also works when the chain is driven as a future.
#[test]
fn then_unchecked_works_under_await() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");
        let chain = unsafe {
            value(5).then_unchecked(|x| {
                let y = value(6).sync().expect("nested sync must succeed");
                value(x + y)
            })
        };
        let out = futures::executor::block_on(chain.into_future()).expect("await failed");
        assert_eq!(out, 11);
    });
}

/// A panic inside `execute` (here: inside a `then` closure) must release the
/// lock on unwind; before the RAII guard it stayed set and every later
/// operation on the thread failed with the non-reentrant error.
#[test]
fn panic_inside_sync_releases_lock() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");
        let panicked = std::panic::catch_unwind(|| {
            value(())
                .then(|()| -> Value<()> { panic!("intentional panic inside execute") })
                .sync()
        });
        assert!(panicked.is_err(), "the panic must propagate");
        assert_eq!(
            value(7).sync().expect("lock must be free after the panic"),
            7
        );
    });
}

#[test]
fn panic_inside_future_poll_releases_lock() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");
        let panicked = std::panic::catch_unwind(|| {
            let fut = value(())
                .then(|()| -> Value<()> { panic!("intentional panic inside poll") })
                .into_future();
            futures::executor::block_on(fut)
        });
        assert!(panicked.is_err(), "the panic must propagate");
        let out = futures::executor::block_on(value(8).into_future())
            .expect("lock must be free after the panic");
        assert_eq!(out, 8);
    });
}

#[test]
fn panic_inside_scope_releases_lock() {
    on_fresh_thread(|| {
        let device = cuda_core::Device::new(0).unwrap();
        let stream = device.new_stream().unwrap();
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            CudaGraph::scope(&stream, |_s| panic!("intentional panic in scope"))
        }));
        assert!(panicked.is_err(), "the panic must propagate");
        assert_eq!(
            value(9)
                .sync_on(&stream)
                .expect("lock must be free after the panic"),
            9
        );
    });
}
