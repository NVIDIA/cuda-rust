/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tests for `CudaGraph` — scoped and op-based capture, launch lifetime,
//! and `update`.

use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_future::DeviceFuture;
use cuda_async::device_operation::{value, DeviceOp, ExecutionContext, Value};
use cuda_async::error::DeviceError;
use std::future::IntoFuture;

fn has_gpu() -> bool {
    cuda_core::Device::device_count()
        .map(|n| n > 0)
        .unwrap_or(false)
}

fn on_fresh_thread<F: FnOnce() + Send + 'static>(f: F) {
    std::thread::spawn(f).join().expect("test thread panicked");
}

#[test]
fn scope_empty_closure() {
    if !has_gpu() {
        return;
    }
    on_fresh_thread(|| {
        let device = cuda_core::Device::new(0).unwrap();
        let stream = device.new_stream().unwrap();

        let graph = CudaGraph::scope(&stream, |_s| Ok(())).unwrap();
        graph.launch().sync_on(&stream).unwrap();
    });
}

#[test]
fn scope_records_value_ops() {
    if !has_gpu() {
        return;
    }
    on_fresh_thread(|| {
        let device = cuda_core::Device::new(0).unwrap();
        let stream = device.new_stream().unwrap();

        let mut recorded = Vec::new();
        let graph = CudaGraph::scope(&stream, |s| {
            let a = s.record(value(42))?;
            let b = s.record(value("hello"))?;
            recorded.push(a);
            recorded.push(b.len() as i32);
            Ok(())
        })
        .unwrap();

        assert_eq!(recorded, vec![42, 5]);
        graph.launch().sync_on(&stream).unwrap();
    });
}

#[test]
fn scope_error_propagation() {
    if !has_gpu() {
        return;
    }
    on_fresh_thread(|| {
        let device = cuda_core::Device::new(0).unwrap();
        let stream = device.new_stream().unwrap();

        let result = CudaGraph::scope(&stream, |_s| {
            Err(DeviceError::Internal("test error".into()))
        });

        assert!(result.is_err());
        match result {
            Err(DeviceError::Internal(msg)) => {
                assert!(
                    msg.contains("test error"),
                    "Expected test error, got: {msg}"
                );
            }
            Err(e) => panic!("Expected Internal error, got: {e}"),
            Ok(_) => panic!("Expected error, got Ok"),
        }
    });
}

#[test]
fn scope_panic_safety() {
    if !has_gpu() {
        return;
    }
    let result = std::thread::spawn(|| {
        let device = cuda_core::Device::new(0).unwrap();
        let stream = device.new_stream().unwrap();

        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            CudaGraph::scope(&stream, |_s| {
                panic!("intentional panic in scope");
            })
        }));

        // Stream should still be usable after the panic.
        unsafe { stream.synchronize() }.unwrap();
    })
    .join();

    assert!(
        result.is_ok(),
        "Thread should not panic after scope cleanup"
    );
}

#[test]
fn scope_multiple_launches() {
    if !has_gpu() {
        return;
    }
    on_fresh_thread(|| {
        let device = cuda_core::Device::new(0).unwrap();
        let stream = device.new_stream().unwrap();

        let graph = CudaGraph::scope(&stream, |_s| Ok(())).unwrap();

        for _ in 0..10 {
            graph.launch().sync_on(&stream).unwrap();
        }
    });
}

#[test]
fn scope_nested_execution_rejected() {
    // Any attempt to execute a DeviceOp inside the scope closure
    // (via sync_on, sync, etc.) is rejected by the thread-local
    // execution lock — enforcing the invariant that only one
    // DeviceOp may be executing at a time per thread.
    if !has_gpu() {
        return;
    }
    on_fresh_thread(|| {
        let device = cuda_core::Device::new(0).unwrap();
        let stream = device.new_stream().unwrap();
        let other_stream = device.new_stream().unwrap();

        // sync_on capture stream — rejected by execution lock.
        let result = CudaGraph::scope(&stream, |_s| {
            let _ = value(42).sync_on(&stream)?;
            Ok(())
        });
        assert!(result.is_err(), "nested sync_on should fail");

        // sync_on other stream — also rejected by execution lock.
        let result = CudaGraph::scope(&stream, |_s| {
            let _ = value(42).sync_on(&other_stream)?;
            Ok(())
        });
        assert!(result.is_err(), "nested sync_on (other stream) should fail");

        // sync — also rejected.
        let result = CudaGraph::scope(&stream, |_s| {
            value(42).sync()?;
            Ok(())
        });
        assert!(result.is_err(), "nested sync should fail");
    });
}

// ---------------------------------------------------------------------------
// Launch lifetime: a GraphLaunch shares ownership of the instantiated graph
// ---------------------------------------------------------------------------

/// `GraphLaunch` used to copy the raw `CUgraphExec`, so dropping the graph
/// before replaying the launch ran a destroyed exec.
#[test]
fn launch_outlives_its_graph() {
    if !has_gpu() {
        return;
    }
    on_fresh_thread(|| {
        let device = cuda_core::Device::new(0).unwrap();
        let stream = device.new_stream().unwrap();

        let graph = CudaGraph::scope(&stream, |s| {
            s.record(value(1))?;
            Ok(())
        })
        .unwrap();
        let launch = graph.launch();
        drop(graph);
        launch
            .sync_on(&stream)
            .expect("launch must keep the instantiated graph alive");

        // Several pending launches, replayed on different streams, after
        // the graph is gone.
        let graph = CudaGraph::capture(stream.clone(), value(3)).unwrap();
        let a = graph.launch();
        let b = graph.launch();
        drop(graph);
        a.sync_on(&stream).unwrap();
        b.sync().unwrap();
    });
}

// ---------------------------------------------------------------------------
// `capture` holds the execution lock and always ends the capture
// ---------------------------------------------------------------------------

#[test]
fn capture_rejects_nested_execution() {
    if !has_gpu() {
        return;
    }
    on_fresh_thread(|| {
        let device = cuda_core::Device::new(0).unwrap();
        let stream = device.new_stream().unwrap();

        let mut graph = CudaGraph::capture(
            stream.clone(),
            value(1).then(|x| {
                let nested = value(0).sync();
                assert!(
                    matches!(&nested, Err(DeviceError::Internal(m)) if m.contains("non-reentrant")),
                    "nested execution inside capture must hit the lock, got {nested:?}"
                );
                value(x)
            }),
        )
        .expect("capture failed");
        assert_eq!(graph.take_output(), Some(1));
    });
}

/// After a panic inside the captured op the stream must not be stuck in
/// capture mode, the execution lock must be free, and the stream must be
/// capturable again.
#[test]
fn capture_panic_ends_capture_and_releases_lock() {
    if !has_gpu() {
        return;
    }
    on_fresh_thread(|| {
        let device = cuda_core::Device::new(0).unwrap();
        let stream = device.new_stream().unwrap();

        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            CudaGraph::capture(
                stream.clone(),
                value(()).then(|()| -> Value<()> { panic!("intentional panic in capture") }),
            )
        }));
        assert!(panicked.is_err(), "the panic must propagate");

        unsafe { stream.synchronize() }.expect("stream must not be left in capture mode");
        assert_eq!(value(2).sync_on(&stream).expect("lock must be free"), 2);
        let mut graph = CudaGraph::capture(stream.clone(), value(4)).expect("recapture");
        assert_eq!(graph.take_output(), Some(4));
    });
}

/// An op that fails at execute time, for the error path of `capture`.
struct FailingOp;

impl DeviceOp for FailingOp {
    type Output = ();
    unsafe fn execute(self, _context: &ExecutionContext) -> Result<(), DeviceError> {
        Err(DeviceError::Internal("failing op".into()))
    }
}

impl IntoFuture for FailingOp {
    type Output = Result<(), DeviceError>;
    type IntoFuture = DeviceFuture<(), FailingOp>;
    fn into_future(self) -> Self::IntoFuture {
        DeviceFuture::failed(DeviceError::Internal("not used".into()))
    }
}

#[test]
fn capture_error_ends_capture_and_releases_lock() {
    if !has_gpu() {
        return;
    }
    on_fresh_thread(|| {
        let device = cuda_core::Device::new(0).unwrap();
        let stream = device.new_stream().unwrap();

        let err = match CudaGraph::capture(stream.clone(), FailingOp) {
            Err(err) => err,
            Ok(_) => panic!("capture of a failing op must fail"),
        };
        assert!(
            matches!(&err, DeviceError::Internal(m) if m == "failing op"),
            "op error must propagate unchanged, got {err:?}"
        );

        unsafe { stream.synchronize() }.expect("stream must not be left in capture mode");
        assert_eq!(value(2).sync_on(&stream).expect("lock must be free"), 2);
        CudaGraph::capture(stream.clone(), value(4)).expect("recapture");
    });
}

/// The `graph()` / `graph_on()` combinators route through `capture` and so
/// inherit the lock and the panic safety.
#[test]
fn graph_combinators_capture_and_replay() {
    if !has_gpu() {
        return;
    }
    on_fresh_thread(|| {
        let device = cuda_core::Device::new(0).unwrap();
        let stream = device.new_stream().unwrap();

        let mut graph = value(5).graph_on(stream.clone()).expect("graph_on");
        assert_eq!(graph.take_output(), Some(5));
        graph.launch().sync_on(&stream).unwrap();

        let mut graph = value(6).graph().expect("graph");
        assert_eq!(graph.take_output(), Some(6));
        graph.launch().sync().unwrap();

        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            value(())
                .then(|()| -> Value<()> { panic!("intentional panic in graph_on") })
                .graph_on(stream.clone())
        }));
        assert!(panicked.is_err());
        unsafe { stream.synchronize() }.expect("stream must not be left in capture mode");
        assert_eq!(value(7).sync_on(&stream).expect("lock must be free"), 7);
    });
}

// ---------------------------------------------------------------------------
// `update`
// ---------------------------------------------------------------------------

/// `update` accepts unit-output `GraphNode`s only (the rejections are
/// compile-fail doctests on `CudaGraph::update`), runs them on the graph's
/// stream, and takes the execution lock like `sync_on`.
#[test]
fn update_runs_unit_graph_nodes() {
    if !has_gpu() {
        return;
    }
    on_fresh_thread(|| {
        let device = cuda_core::Device::new(0).unwrap();
        let stream = device.new_stream().unwrap();

        let graph = CudaGraph::scope(&stream, |_s| Ok(())).unwrap();
        graph.update(value(())).expect("unit GraphNode");
        graph.launch().sync_on(graph.stream()).unwrap();

        // `update` executes under the lock: from inside an executing region
        // it is rejected like any other nested execution.
        let nested = value(())
            .then(|()| {
                let r = graph.update(value(()));
                assert!(
                    matches!(&r, Err(DeviceError::Internal(m)) if m.contains("non-reentrant")),
                    "update inside an executing op must hit the lock, got {r:?}"
                );
                value(())
            })
            .sync_on(&stream);
        assert!(nested.is_ok());
    });
}
