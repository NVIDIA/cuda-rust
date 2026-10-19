/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Integration tests for custom memory pool allocation.
//!
//! Tests that exercise the `MemPool` lifecycle, device-level pool
//! configuration, and pool-aware allocation through `ExecutionContext`.
//!
//! Each test runs on a fresh thread so that thread-local `DEVICE_CONTEXTS`
//! starts clean.

use cuda_async::device_context::{
    clear_device_pool, get_device_pool, global_policy, init_device_contexts, set_device_pool,
    with_device,
};
use cuda_async::device_operation::{value, DeviceOp};
use cuda_async::prelude::*;
use std::future::IntoFuture;

fn on_fresh_thread<F: FnOnce() + Send + 'static>(f: F) {
    std::thread::spawn(f).join().expect("test thread panicked");
}

// ---------------------------------------------------------------------------
// MemPool RAII lifecycle
// ---------------------------------------------------------------------------

#[test]
fn create_and_drop_mem_pool() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool creation failed");

        assert!(!pool.cu_pool().is_null());
    });
}

#[test]
fn default_mem_pool_is_not_owned() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool = with_device(0, |device| device.default_mem_pool())
            .expect("get context failed")
            .expect("default pool failed");

        assert!(!pool.cu_pool().is_null());
    });
}

#[test]
fn set_release_threshold() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool creation failed");

        pool.set_release_threshold(u64::MAX)
            .expect("set threshold failed");

        pool.set_release_threshold(1024 * 1024)
            .expect("set finite threshold failed");
    });
}

// ---------------------------------------------------------------------------
// Device-level pool configuration
// ---------------------------------------------------------------------------

#[test]
fn set_and_get_device_pool() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool_opt = get_device_pool(0).expect("get pool failed");
        assert!(pool_opt.is_none());

        let pool = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool creation failed");
        let pool_ptr = pool.cu_pool();
        set_device_pool(0, pool).expect("set pool failed");

        let retrieved = get_device_pool(0)
            .expect("get pool failed")
            .expect("pool should be set");
        assert_eq!(retrieved.cu_pool(), pool_ptr);
    });
}

#[test]
fn clear_device_pool_reverts_to_none() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool creation failed");
        set_device_pool(0, pool).expect("set pool failed");

        assert!(get_device_pool(0).expect("get failed").is_some());

        clear_device_pool(0).expect("clear pool failed");
        assert!(get_device_pool(0).expect("get failed").is_none());
    });
}

#[test]
fn set_device_pool_rejects_cross_device_pool() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool creation failed");

        let err = set_device_pool(99, pool).expect_err("expected cross-device pool to be rejected");
        match err {
            cuda_async::error::DeviceError::Context { device_id, message } => {
                assert_eq!(device_id, 99, "error should point to target device");
                assert!(
                    message.contains("pool belongs to device 0")
                        && message.contains("expected device 99"),
                    "message should name both devices, got: {message}"
                );
            }
            other => panic!("expected DeviceError::Context, got {other:?}"),
        }

        assert!(get_device_pool(0).expect("get pool failed").is_none());
    });
}

// ---------------------------------------------------------------------------
// Pool-aware allocation through DeviceOp
// ---------------------------------------------------------------------------

#[test]
fn alloc_with_custom_pool_via_device_op() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool creation failed");
        pool.set_release_threshold(u64::MAX)
            .expect("set threshold failed");
        let pool_ptr = pool.cu_pool() as usize;
        set_device_pool(0, pool).expect("set pool failed");

        let op = with_context(move |ctx| {
            let p = ctx
                .get_pool()
                .expect("pool must be present in ExecutionContext via sync()");
            assert_eq!(
                p.cu_pool() as usize,
                pool_ptr,
                "sync() must route allocation through the registered pool"
            );
            let num_bytes = 1024;
            let dptr = unsafe { ctx.alloc_async(num_bytes) }.expect("device allocation failed");
            assert!(dptr != 0, "allocation returned null pointer");
            value(dptr)
        });
        let dptr = op.sync().expect("device op failed");
        assert!(dptr != 0);
    });
}

#[test]
fn alloc_with_custom_pool_via_into_future() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool creation failed");
        pool.set_release_threshold(u64::MAX)
            .expect("set threshold failed");
        let pool_ptr = pool.cu_pool() as usize;
        set_device_pool(0, pool).expect("set pool failed");

        let op = with_context(move |ctx| {
            let p = ctx
                .get_pool()
                .expect("pool must be present in ExecutionContext via into_future()");
            assert_eq!(
                p.cu_pool() as usize,
                pool_ptr,
                "into_future() must route allocation through the registered pool"
            );
            value(())
        });
        futures::executor::block_on(op.into_future()).expect("future failed");
    });
}

#[test]
fn alloc_without_pool_uses_default() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let op = with_context(|ctx| {
            assert!(ctx.get_pool().is_none());
            let num_bytes = 1024;
            let dptr = unsafe { ctx.alloc_async(num_bytes) }.expect("device allocation failed");
            assert!(dptr != 0, "allocation returned null pointer");
            value(dptr)
        });
        let dptr = op.sync().expect("device op failed");
        assert!(dptr != 0);
    });
}

#[test]
fn pool_is_frozen_at_scheduling_time() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool_a = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool A creation failed");
        let pool_b = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool B creation failed");

        let pool_a_ptr = pool_a.cu_pool() as usize;

        set_device_pool(0, pool_a).expect("set pool_a failed");

        // Schedule while pool_a is active — freezes pool_a into ExecutionContext.
        let policy = global_policy(0).expect("get policy failed");
        let future = with_context(move |ctx| {
            let p = ctx.get_pool().expect("pool should be present");
            assert_eq!(
                p.cu_pool() as usize,
                pool_a_ptr,
                "should use frozen pool_a, not pool_b"
            );
            value(())
        })
        .schedule(&policy)
        .expect("schedule failed");

        // Change global pool AFTER scheduling — must not affect the already-frozen ExecutionContext.
        set_device_pool(0, pool_b).expect("set pool_b failed");

        // Execute: DeviceFuture carries pool_a in its ExecutionContext, pool_b is ignored.
        futures::executor::block_on(future).expect("future failed");
    });
}

// ---------------------------------------------------------------------------
// Explicit .schedule() path
// ---------------------------------------------------------------------------

#[test]
fn schedule_applies_device_pool() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool creation failed");
        pool.set_release_threshold(u64::MAX)
            .expect("set threshold failed");
        let pool_ptr = pool.cu_pool() as usize;
        set_device_pool(0, pool).expect("set pool failed");

        let policy = global_policy(0).expect("get policy failed");
        let future = with_context(move |ctx| {
            let p = ctx.get_pool().expect("pool should be present via schedule");
            assert_eq!(
                p.cu_pool() as usize,
                pool_ptr,
                "schedule must pick up device pool"
            );
            let dptr = unsafe { ctx.alloc_async(512) }.expect("device allocation failed");
            assert!(dptr != 0, "allocation returned null pointer");
            value(dptr)
        })
        .schedule(&policy)
        .expect("schedule failed");

        let dptr = futures::executor::block_on(future).expect("future failed");
        assert!(dptr != 0);
    });
}

#[test]
fn sync_on_applies_device_pool() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool creation failed");
        pool.set_release_threshold(u64::MAX)
            .expect("set threshold failed");
        let pool_ptr = pool.cu_pool() as usize;
        set_device_pool(0, pool).expect("set pool failed");

        let stream = global_policy(0)
            .expect("get policy failed")
            .next_stream()
            .expect("get stream failed");

        let dptr = with_context(move |ctx| {
            let p = ctx.get_pool().expect("pool should be present via sync_on");
            assert_eq!(
                p.cu_pool() as usize,
                pool_ptr,
                "sync_on must pick up device pool"
            );
            let dptr = unsafe { ctx.alloc_async(512) }.expect("device allocation failed");
            assert!(dptr != 0, "allocation returned null pointer");
            value(dptr)
        })
        .sync_on(&stream)
        .expect("sync_on failed");
        assert!(dptr != 0);
    });
}

// ---------------------------------------------------------------------------
// Multiple pools
// ---------------------------------------------------------------------------

#[test]
fn switch_between_pools() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool_a = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool A creation failed");
        let pool_b = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool B creation failed");

        let pool_a_ptr = pool_a.cu_pool();
        let pool_b_ptr = pool_b.cu_pool();
        assert_ne!(pool_a_ptr, pool_b_ptr, "pools should be distinct");

        set_device_pool(0, pool_a).expect("set A failed");
        let op_a = with_context(|ctx| {
            let dptr = unsafe { ctx.alloc_async(512) }.expect("device allocation failed");
            assert!(dptr != 0);
            value(())
        });
        op_a.sync().expect("op A failed");

        set_device_pool(0, pool_b).expect("set B failed");
        let op_b = with_context(|ctx| {
            let dptr = unsafe { ctx.alloc_async(512) }.expect("device allocation failed");
            assert!(dptr != 0);
            value(())
        });
        op_b.sync().expect("op B failed");

        clear_device_pool(0).expect("clear failed");
        let op_default = with_context(|ctx| {
            assert!(ctx.get_pool().is_none());
            let dptr = unsafe { ctx.alloc_async(512) }.expect("device allocation failed");
            assert!(dptr != 0);
            value(())
        });
        op_default.sync().expect("default op failed");
    });
}

// ---------------------------------------------------------------------------
// Memory accounting
// ---------------------------------------------------------------------------

#[test]
fn mem_stats_initially_zero() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool creation failed");

        let stats = pool.mem_stats().expect("read stats failed");
        assert_eq!(stats.used_current, 0, "no allocations yet");
        assert_eq!(stats.used_high, 0, "no allocations yet");
    });
}

#[test]
fn mem_stats_track_alloc_and_free() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool creation failed");
        pool.set_release_threshold(u64::MAX)
            .expect("set threshold failed");

        let stream = global_policy(0)
            .expect("get policy failed")
            .next_stream()
            .expect("get stream failed");

        const N: usize = 1 << 20;

        let dptr = unsafe { cuda_core::malloc_from_pool_async(N, &pool, &stream) }
            .expect("pool allocation failed");
        assert!(dptr != 0);
        unsafe { stream.synchronize() }.expect("stream sync failed");

        let stats = pool.mem_stats().expect("read stats failed");
        assert!(
            stats.used_current >= N as u64,
            "used_current should reflect alloc, got {}",
            stats.used_current
        );
        assert!(
            stats.used_high >= N as u64,
            "used_high should track peak, got {}",
            stats.used_high
        );

        unsafe { cuda_core::free_async(dptr, &stream) }.expect("free failed");
        unsafe { stream.synchronize() }.expect("stream sync failed");

        let stats = pool.mem_stats().expect("read stats failed");
        assert_eq!(
            stats.used_current, 0,
            "used_current should return to 0 after free + sync"
        );
        assert!(
            stats.used_high >= N as u64,
            "used_high is a watermark and must not decrease, got {}",
            stats.used_high
        );
    });
}

#[test]
fn reset_used_high_collapses_watermark_to_current() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool creation failed");
        pool.set_release_threshold(u64::MAX)
            .expect("set threshold failed");

        let stream = global_policy(0)
            .expect("get policy failed")
            .next_stream()
            .expect("get stream failed");

        const N: usize = 1 << 20;

        let dptr = unsafe { cuda_core::malloc_from_pool_async(N, &pool, &stream) }
            .expect("pool allocation failed");
        unsafe { cuda_core::free_async(dptr, &stream) }.expect("free failed");
        unsafe { stream.synchronize() }.expect("stream sync failed");

        let before = pool.mem_stats().expect("read stats failed");
        assert_eq!(before.used_current, 0);
        assert!(
            before.used_high >= N as u64,
            "precondition: high watermark should be > 0"
        );

        pool.reset_used_high().expect("reset used_high failed");

        let after = pool.mem_stats().expect("read stats failed");
        assert_eq!(
            after.used_high, after.used_current,
            "after reset, used_high should collapse to used_current"
        );
    });
}

// ---------------------------------------------------------------------------
// Thread-local context map survives failed lookups and rejected re-inits
// ---------------------------------------------------------------------------

/// Regression: `with_global_device_context` used to return through `?` with
/// the thread's context map taken out of its `Cell`, dropping every live
/// context on the thread. A lookup for a device that cannot be initialized
/// must fail *without* losing the pool registered for device 0.
#[test]
fn failed_lookup_keeps_other_device_contexts_alive() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool creation failed");
        let pool_ptr = pool.cu_pool();
        set_device_pool(0, pool).expect("set pool failed");

        // Device 9999 does not exist: the default-policy init fails after the
        // map has been leased out of the thread-local.
        let lookup = cuda_async::device_context::with_global_device_context(9999, |_| ());
        assert!(lookup.is_err(), "lookup of a bogus device must fail");

        let retrieved = get_device_pool(0)
            .expect("get pool failed")
            .expect("pool must survive a failed lookup on another device");
        assert_eq!(retrieved.cu_pool(), pool_ptr);
    });
}

/// Regression: a rejected second `init_device_contexts` used to discard the
/// live map while reporting "already initialized".
#[test]
fn rejected_reinit_keeps_device_contexts_alive() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool creation failed");
        let pool_ptr = pool.cu_pool();
        set_device_pool(0, pool).expect("set pool failed");

        assert!(
            init_device_contexts(0, 1).is_err(),
            "second init must be rejected"
        );

        let retrieved = get_device_pool(0)
            .expect("get pool failed")
            .expect("pool must survive a rejected re-initialization");
        assert_eq!(retrieved.cu_pool(), pool_ptr);
    });
}
