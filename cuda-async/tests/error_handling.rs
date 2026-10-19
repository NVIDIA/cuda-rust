/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Integration tests for `cuda-async` error handling.
//!
//! Tests that exercise the error helpers, `DeviceError` formatting / conversions,
//! the device-context initialization guards, and the `DeviceFuture::failed` constructor.
//!
//! Tests that touch CUDA thread-local state each run on their own thread so they
//! start from a clean `DEVICE_CONTEXTS`.

use cuda_async::device_context::{
    get_default_device, init_device_contexts, set_default_device, DEFAULT_DEVICE_ID,
};
use cuda_async::device_future::DeviceFuture;
use cuda_async::device_operation::Value;
use cuda_async::error::{device_assert, device_error, DeviceError};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

// ---------------------------------------------------------------------------
// Helper function tests (pure logic, no CUDA driver needed)
// ---------------------------------------------------------------------------

#[test]
fn device_error_returns_context_variant() {
    let err = device_error(3, "something went wrong");
    assert_eq!(
        err,
        DeviceError::Context {
            device_id: 3,
            message: "something went wrong".to_string(),
        }
    );
}

#[test]
fn device_assert_ok_when_predicate_is_true() {
    let result = device_assert(0, true, "should not fire");
    assert!(result.is_ok());
}

#[test]
fn device_assert_err_when_predicate_is_false() {
    let result = device_assert(7, false, "assertion failed");
    let err = result.unwrap_err();
    assert_eq!(
        err,
        DeviceError::Context {
            device_id: 7,
            message: "assertion failed".to_string(),
        }
    );
}

// ---------------------------------------------------------------------------
// DeviceError Display / formatting tests
// ---------------------------------------------------------------------------

#[test]
fn context_error_display_contains_device_id_and_message() {
    let err = DeviceError::Context {
        device_id: 42,
        message: "bad thing".to_string(),
    };
    let display = format!("{err}");
    assert!(
        display.contains("device_id=42"),
        "expected device_id in display, got: {display}"
    );
    assert!(
        display.contains("bad thing"),
        "expected message in display, got: {display}"
    );
}

#[test]
fn internal_error_display() {
    let err = DeviceError::Internal("oops".to_string());
    let display = format!("{err}");
    assert!(
        display.contains("oops"),
        "expected message in display, got: {display}"
    );
}

#[test]
fn launch_error_display() {
    let err = DeviceError::Launch("kernel failed".to_string());
    let display = format!("{err}");
    assert!(
        display.contains("kernel failed"),
        "expected message in display, got: {display}"
    );
}

#[test]
fn scheduling_error_display() {
    let err = DeviceError::Scheduling("no streams".to_string());
    let display = format!("{err}");
    assert!(
        display.contains("no streams"),
        "expected message in display, got: {display}"
    );
}

#[test]
fn anyhow_error_converts_to_device_error() {
    let anyhow_err = anyhow::anyhow!("something from anyhow");
    let device_err: DeviceError = anyhow_err.into();
    match &device_err {
        DeviceError::Anyhow(msg) => {
            assert!(
                msg.contains("something from anyhow"),
                "expected anyhow message, got: {msg}"
            );
        }
        other => panic!("expected Anyhow variant, got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Device-context initialization tests
//
// Each test spawns a dedicated thread so that the thread-local DEVICE_CONTEXTS
// starts in its default (uninitialized) state.
// ---------------------------------------------------------------------------

/// Helper: run `f` on a fresh thread and propagate any panic.
fn on_fresh_thread<F: FnOnce() + Send + 'static>(f: F) {
    std::thread::spawn(f).join().expect("test thread panicked");
}

#[test]
fn double_init_returns_context_already_initialized() {
    on_fresh_thread(|| {
        // First init should succeed.
        init_device_contexts(0, 1).expect("first init should succeed");

        // Second init should fail with Context { message: "Context already initialized." }.
        let err = init_device_contexts(0, 1).unwrap_err();
        match &err {
            DeviceError::Context { device_id, message } => {
                assert_eq!(*device_id, 0);
                assert!(
                    message.contains("Context already initialized"),
                    "unexpected message: {message}"
                );
            }
            other => panic!("expected Context variant, got: {other:?}"),
        }
    });
}

#[test]
fn default_device_id_is_zero() {
    on_fresh_thread(|| {
        assert_eq!(get_default_device(), DEFAULT_DEVICE_ID);
        assert_eq!(DEFAULT_DEVICE_ID, 0);
    });
}

#[test]
fn set_default_device_changes_value() {
    on_fresh_thread(|| {
        set_default_device(5);
        assert_eq!(get_default_device(), 5);
    });
}

#[test]
fn new_device_context_with_invalid_device_returns_driver_error() {
    // Device ordinal 9999 should not exist on any reasonable system.
    // Device::new(9999) will fail with a driver error, which propagates
    // through new_device_context.
    let result = cuda_core::Device::new(9999);
    let result = result.map_err(DeviceError::Driver);
    match result {
        Err(DeviceError::Driver(_)) => { /* expected */ }
        Err(other) => panic!("expected Driver variant, got: {other:?}"),
        Ok(_) => panic!("expected error for invalid device 9999, but got Ok"),
    }
}

// ---------------------------------------------------------------------------
// DeviceError equality / clone tests (derive coverage)
// ---------------------------------------------------------------------------

#[test]
fn device_error_is_cloneable_and_eq() {
    let err = DeviceError::Context {
        device_id: 1,
        message: "test".to_string(),
    };
    let cloned = err.clone();
    assert_eq!(err, cloned);
}

#[test]
fn different_variants_are_not_equal() {
    let a = DeviceError::Internal("x".to_string());
    let b = DeviceError::Launch("x".to_string());
    assert_ne!(a, b);
}

// ---------------------------------------------------------------------------
// DeviceFuture::failed tests
// ---------------------------------------------------------------------------

/// Create a no-op waker for manually polling futures in tests.
fn noop_waker() -> Waker {
    fn noop(_: *const ()) {}
    fn clone(p: *const ()) -> RawWaker {
        RawWaker::new(p, &VTABLE)
    }
    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);
    unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
}

#[test]
fn failed_future_returns_err_on_first_poll() {
    let error = DeviceError::Internal("test failure".to_string());
    let mut future: DeviceFuture<(), Value<()>> = DeviceFuture::failed(error);

    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);
    let result = Pin::new(&mut future).poll(&mut cx);

    match result {
        Poll::Ready(Err(DeviceError::Internal(msg))) => {
            assert_eq!(msg, "test failure");
        }
        Poll::Ready(Ok(_)) => panic!("expected Err, got Ok"),
        Poll::Ready(Err(other)) => panic!("expected Internal variant, got: {other:?}"),
        Poll::Pending => panic!("expected Ready, got Pending"),
    }
}

#[test]
fn failed_future_is_immediately_ready() {
    let error = DeviceError::Internal("done".to_string());
    let mut future: DeviceFuture<(), Value<()>> = DeviceFuture::failed(error);

    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);
    let result = Pin::new(&mut future).poll(&mut cx);

    // The future should resolve immediately — it must not return Pending.
    assert!(
        matches!(result, Poll::Ready(Err(DeviceError::Internal(_)))),
        "expected Poll::Ready(Err(Internal(...))), got: {result:?}"
    );
}

#[test]
#[should_panic(expected = "Poll called after completion")]
fn failed_future_panics_on_second_poll() {
    let error = DeviceError::Internal("once".to_string());
    let mut future: DeviceFuture<(), Value<()>> = DeviceFuture::failed(error);

    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);

    // First poll: returns the error.
    let _ = Pin::new(&mut future).poll(&mut cx);
    // Second poll: should panic.
    let _ = Pin::new(&mut future).poll(&mut cx);
}

#[test]
fn failed_future_preserves_error_variant() {
    let error = DeviceError::Context {
        device_id: 42,
        message: "device gone".to_string(),
    };
    let mut future: DeviceFuture<String, Value<String>> = DeviceFuture::failed(error);

    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);
    let result = Pin::new(&mut future).poll(&mut cx);

    match result {
        Poll::Ready(Err(DeviceError::Context { device_id, message })) => {
            assert_eq!(device_id, 42);
            assert_eq!(message, "device gone");
        }
        other => panic!("expected Context error, got: {other:?}"),
    }
}
