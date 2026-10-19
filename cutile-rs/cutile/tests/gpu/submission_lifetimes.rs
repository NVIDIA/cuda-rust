/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cutile::cuda_async::device_buffer::DeviceAllocation;
use cutile::cuda_async::device_future::DeviceFuture;
use cutile::cuda_async::device_operation::{value, ExecutionContext, Value};
use cutile::cuda_async::error::DeviceError;
use cutile::cuda_async::futures::{executor::block_on, task::noop_waker};
use cutile::prelude::*;
use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::sync::{mpsc, Arc};
use std::task::{Context, Poll};
use std::time::Duration;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry]
    fn copy<const S: [i32; 1]>(out: &mut Tensor<f32, S>, input: &Tensor<f32, { [-1] }>) {
        out.store(load_tile_like(input, out));
    }
}

struct Allocation(Tensor<f32>);

// SAFETY: the owned tensor has a stable, live allocation. The test's rescue
// handle is used only to observe refcounts, never to access its GPU bytes.
unsafe impl DeviceAllocation for Allocation {
    fn device_ptr(&self) -> u64 {
        self.0.device_pointer().cu_deviceptr()
    }
    fn len_bytes(&self) -> usize {
        self.0.num_bytes()
    }
    fn device_id(&self) -> usize {
        0
    }
}

fn tracked(stream: &Arc<cuda_core::Stream>) -> (Tensor<f32>, Arc<Allocation>) {
    let allocation = Arc::new(Allocation(api::ones(&[32]).sync_on(stream).unwrap()));
    // SAFETY: no accesses are issued through `allocation`; only this foreign
    // tensor accesses the bytes. The rescue owner also makes a failing lifetime
    // assertion harmless, rather than causing an actual GPU use-after-free.
    let tensor = unsafe { Tensor::from_foreign(allocation.clone(), vec![32], vec![1]) };
    (tensor, allocation)
}

/// Diagnostics for the first-poll assertion: how many Gates were armed in
/// this process and how many of their callbacks have started running.
static GATES_ARMED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static GATES_STARTED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static GATES_FINISHED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static GATE_WAIT_OUTCOMES: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
static LAST_GATE_ARMED_AT: std::sync::Mutex<Option<std::time::Instant>> =
    std::sync::Mutex::new(None);

/// Upper bound on how long a Gate may hold its stream if a test never drops
/// it. Only a safety net: a gated test's first poll can legitimately sit for
/// many seconds behind other tests' context-wide synchronizes when the suite
/// runs 20-wide on one GPU, so this must dwarf any such stall.
const GATE_BOUND: Duration = Duration::from_secs(60);

struct Gate {
    release: mpsc::Sender<()>,
    /// Set by the callback when it hit `GATE_BOUND`: this gate's stream drained
    /// on its own, so a `Ready` first poll is a harness stall, not an ordering
    /// bug. Per gate, because the process-wide counters below also see other
    /// tests' gates when the suite runs in parallel.
    expired: Arc<std::sync::atomic::AtomicBool>,
}

impl Gate {
    fn arm(stream: &Arc<cuda_core::Stream>) -> Self {
        let (sender, receiver) = mpsc::channel();
        let expired = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let expired_flag = expired.clone();
        GATES_ARMED.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        *LAST_GATE_ARMED_AT.lock().unwrap() = Some(std::time::Instant::now());
        unsafe {
            stream
                .launch_host_function(move || {
                    GATES_STARTED.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    // Bounded even if an unexpected synchronous path is introduced.
                    let outcome = receiver.recv_timeout(GATE_BOUND);
                    if outcome.is_err() {
                        expired_flag.store(true, std::sync::atomic::Ordering::SeqCst);
                    }
                    GATE_WAIT_OUTCOMES
                        .lock()
                        .unwrap()
                        .push(format!("{outcome:?}"));
                    GATES_FINISHED.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                })
                .unwrap();
        }
        Self {
            release: sender,
            expired,
        }
    }

    fn expired(&self) -> bool {
        self.expired.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl Drop for Gate {
    fn drop(&mut self) {
        let _ = self.release.send(());
    }
}

/// A poll issued while a Gate blocks `stream` must be `Pending` on either
/// completion path. Anything else is reported with the process-wide Gate
/// counters and a fresh stream query, so a failure names its mechanism.
fn assert_gated_pending<T>(
    outcome: Poll<Result<T, cuda_async::error::DeviceError>>,
    stream: &Arc<cuda_core::Stream>,
    gate: &Gate,
) {
    match outcome {
        Poll::Pending => {}
        Poll::Ready(Ok(_)) => {
            let armed = GATES_ARMED.load(std::sync::atomic::Ordering::SeqCst);
            let finished = GATES_FINISHED.load(std::sync::atomic::Ordering::SeqCst);
            let age = LAST_GATE_ARMED_AT.lock().unwrap().map(|t| t.elapsed());
            let outcomes = GATE_WAIT_OUTCOMES.lock().unwrap().clone();
            if gate.expired() {
                panic!(
                    "harness stall: this test's Gate expired ({GATE_BOUND:?} bound) before the first poll; newest gate armed {age:?} ago. Not an ordering failure — the stream drained legitimately (gates armed: {armed}, finished: {finished}, outcomes: {outcomes:?})"
                );
            }
            panic!(
                "gated submission completed on the first poll while its Gate should still be blocking the stream (newest gate armed {age:?} ago; gates armed: {armed}, callbacks started: {}, finished: {finished}, wait outcomes: {outcomes:?}, stream query now: {:?})",
                GATES_STARTED.load(std::sync::atomic::Ordering::SeqCst),
                unsafe { stream.query() },
            )
        }
        Poll::Ready(Err(e)) => panic!("gated submission failed on the first poll: {e}"),
    }
}

fn forget_pending<O: DeviceOp>(op: O, stream: &Arc<cuda_core::Stream>, gate: &Gate) {
    let mut future = DeviceFuture::scheduled(op, ExecutionContext::new(stream.clone()));
    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);
    assert_gated_pending(Pin::new(&mut future).poll(&mut cx), stream, gate);
    std::mem::forget(future);
}

fn on_gpu(f: impl FnOnce(Arc<cuda_core::Stream>, Arc<cuda_core::Stream>) + Send + 'static) {
    crate::common::with_test_stack(move || {
        let device = cuda_core::Device::new(0).unwrap();
        let stream = device.new_stream().unwrap();
        let other = device.new_stream().unwrap();
        let src = api::ones::<f32>(&[32]).sync_on(&stream).unwrap();
        let mut dst = api::zeros::<f32>(&[32]).sync_on(&stream).unwrap();
        kernels::copy((&mut dst).partition([4]), &src)
            .sync_on(&stream)
            .unwrap();
        // Warm every kernel variant and the allocator the gated tests use.
        // Loading a freshly compiled module and growing the stream-ordered
        // pool both synchronize the context; done for the first time under a
        // Gate they wait on the test's own gated stream, and only the Gate's
        // safety bound breaks the cycle (seen as 10 s stalls and spurious
        // "completed early" failures on DGX Spark under 20-way parallelism).
        let view = src.view(&[32]).unwrap();
        kernels::copy((&mut dst).partition([4]).map([1], 8), &view)
            .sync_on(&stream)
            .unwrap();
        let _ = api::dup(&src).sync_on(&stream).unwrap();
        api::memcpy(&mut dst, &src).sync_on(&stream).unwrap();
        // Initialize the completion backend before placing any work behind a gate.
        unsafe {
            stream
                .launch_host_function(|| std::thread::sleep(Duration::from_millis(10)))
                .unwrap()
        };
        block_on(DeviceFuture::scheduled(
            value(()),
            ExecutionContext::new(stream.clone()),
        ))
        .unwrap();
        f(stream, other);
    });
}

#[test]
fn forgotten_borrowed_kernel_retains_inputs_outputs_and_excludes_conflicts() {
    on_gpu(|stream, other| {
        let (mut src, src_owner) = tracked(&stream);
        let (mut dst, dst_owner) = tracked(&stream);
        let mut scratch = api::zeros::<f32>(&[32]).sync_on(&other).unwrap();
        // A pre-recorded graph must perform the same access checks on replay.
        let graph = api::memcpy(&mut scratch, &dst)
            .graph_on(other.clone())
            .unwrap();
        let gate = Gate::arm(&stream);
        forget_pending(
            kernels::copy((&mut dst).partition([4]), &src),
            &stream,
            &gate,
        );
        assert!(api::memcpy(&mut scratch, &dst).sync_on(&other).is_err());
        assert!(api::memcpy(&mut src, &scratch).sync_on(&other).is_err());
        assert!(graph.launch().sync_on(&other).is_err());
        drop(graph);
        drop((src, dst));
        assert!(
            Arc::strong_count(&src_owner) > 1,
            "borrowed input was released after recover"
        );
        assert!(
            Arc::strong_count(&dst_owner) > 1,
            "borrowed output was released after recover"
        );
        drop(gate);
        unsafe { stream.synchronize().unwrap() };
    });
}

#[test]
fn forgotten_view_and_mapped_output_keep_their_storage() {
    on_gpu(|stream, _| {
        let (src, src_owner) = tracked(&stream);
        let (mut dst, dst_owner) = tracked(&stream);
        let view = src.view(&[32]).unwrap();
        let gate = Gate::arm(&stream);
        forget_pending(
            kernels::copy((&mut dst).partition([4]).map([1], 8), &view).then(|_| value(())),
            &stream,
            &gate,
        );
        drop(view);
        drop((src, dst));
        assert!(Arc::strong_count(&src_owner) > 1);
        assert!(Arc::strong_count(&dst_owner) > 1);
        drop(gate);
        unsafe { stream.synchronize().unwrap() };
    });
}

#[test]
fn forgotten_memcpy_keeps_both_allocations() {
    on_gpu(|stream, _| {
        let (src, src_owner) = tracked(&stream);
        let (mut dst, dst_owner) = tracked(&stream);
        let gate = Gate::arm(&stream);
        forget_pending(api::memcpy(&mut dst, &src), &stream, &gate);
        drop((src, dst));
        assert!(Arc::strong_count(&src_owner) > 1);
        assert!(Arc::strong_count(&dst_owner) > 1);
        drop(gate);
        unsafe { stream.synchronize().unwrap() };
    });
}

#[test]
fn projected_owned_inputs_live_until_ready_and_then_release() {
    on_gpu(|stream, other| {
        let (src, src_owner) = tracked(&stream);
        let (dst, dst_owner) = tracked(&stream);
        let mut future = DeviceFuture::scheduled(
            kernels::copy(dst.partition([4]), src).then(|_| value(())),
            ExecutionContext::new(stream.clone()),
        );
        let gate = Gate::arm(&stream);
        let waker = noop_waker();
        assert_gated_pending(
            Pin::new(&mut future).poll(&mut Context::from_waker(&waker)),
            &stream,
            &gate,
        );
        assert!(Arc::strong_count(&src_owner) > 1);
        assert!(Arc::strong_count(&dst_owner) > 1);
        drop(gate);
        block_on(&mut future).unwrap();
        assert_eq!(Arc::strong_count(&src_owner), 1);
        assert_eq!(Arc::strong_count(&dst_owner), 1);

        let src = api::ones::<f32>(&[32]).sync_on(&stream).unwrap();
        let mut dst = api::zeros::<f32>(&[32]).sync_on(&stream).unwrap();
        kernels::copy((&mut dst).partition([4]), &src)
            .sync_on(&stream)
            .unwrap();
        api::memcpy(&mut dst, &src).sync_on(&other).unwrap();
        let host = dst.to_host_vec().sync_on(&other).unwrap();
        assert_eq!(host, vec![1.0; 32]);
    });
}

#[test]
fn projected_dup_keeps_source_until_submission_completes() {
    on_gpu(|stream, _| {
        let (src, owner) = tracked(&stream);
        let op = api::dup(&src).then(|_| value(()));
        drop(src);
        let gate = Gate::arm(&stream);
        forget_pending(op, &stream, &gate);
        assert!(Arc::strong_count(&owner) > 1);
        drop(gate);
        unsafe { stream.synchronize().unwrap() };
    });
}

struct Fail;

impl DeviceOp for Fail {
    type Output = ();
    unsafe fn execute(self, _: &ExecutionContext) -> Result<(), DeviceError> {
        Err(DeviceError::Internal("injected submission error".into()))
    }
}

impl IntoFuture for Fail {
    type Output = Result<(), DeviceError>;
    type IntoFuture = DeviceFuture<(), Self>;
    fn into_future(self) -> Self::IntoFuture {
        DeviceFuture::failed(DeviceError::Internal("injected submission error".into()))
    }
}

#[test]
fn partial_submission_error_retains_discarded_resources() {
    on_gpu(|stream, _| {
        let (src, owner) = tracked(&stream);
        let dst = api::zeros::<f32>(&[32]).sync_on(&stream).unwrap();
        let mut future = DeviceFuture::scheduled(
            kernels::copy(dst.partition([4]), src).then(|_| Fail),
            ExecutionContext::new(stream.clone()),
        );
        let gate = Gate::arm(&stream);
        let waker = noop_waker();
        let result = Pin::new(&mut future).poll(&mut Context::from_waker(&waker));
        assert!(matches!(result, std::task::Poll::Ready(Err(_))));
        assert!(Arc::strong_count(&owner) > 1);
        drop(gate);
        drop(future);
        assert_eq!(Arc::strong_count(&owner), 1);
    });
}

#[test]
fn partial_submission_panic_retains_discarded_resources() {
    on_gpu(|stream, _| {
        let (src, owner) = tracked(&stream);
        let dst = api::zeros::<f32>(&[32]).sync_on(&stream).unwrap();
        let mut future = DeviceFuture::scheduled(
            kernels::copy(dst.partition([4]), src)
                .then(|_| -> Value<()> { panic!("injected panic") }),
            ExecutionContext::new(stream.clone()),
        );
        let gate = Gate::arm(&stream);
        let waker = noop_waker();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = Pin::new(&mut future).poll(&mut Context::from_waker(&waker));
        }));
        assert!(panic.is_err());
        assert!(Arc::strong_count(&owner) > 1);
        drop(gate);
        drop(future);
        assert_eq!(Arc::strong_count(&owner), 1);
    });
}

#[test]
fn graph_replay_retains_projected_storage_and_releases_launch_accesses() {
    on_gpu(|stream, other| {
        let (src, src_owner) = tracked(&stream);
        let (mut dst, dst_owner) = tracked(&stream);
        let graph = kernels::copy((&mut dst).partition([4]), &src)
            .then(|_| value(()))
            .graph_on(stream.clone())
            .unwrap();
        drop(src);
        assert!(Arc::strong_count(&src_owner) > 1);
        graph.launch().sync_on(&other).unwrap();
        let replacement = api::ones::<f32>(&[32]).sync_on(&stream).unwrap();
        graph.update(api::memcpy(&mut dst, &replacement)).unwrap();
        graph.launch().sync_on(&stream).unwrap();
        // A completed replay releases its leases even while the graph lives.
        let host = dst.dup().to_host_vec().sync_on(&other).unwrap();
        assert_eq!(host, vec![1.0; 32]);
        drop(dst);
        let launch = graph.launch();
        drop(graph);
        let gate = Gate::arm(&other);
        forget_pending(launch, &other, &gate);
        assert!(Arc::strong_count(&src_owner) > 1);
        assert!(Arc::strong_count(&dst_owner) > 1);
        drop(gate);
        unsafe { other.synchronize().unwrap() };
    });
}

#[test]
fn cloned_execution_contexts_have_independent_submission_owners() {
    on_gpu(|stream, _| {
        let (src, src_owner) = tracked(&stream);
        let (dst, dst_owner) = tracked(&stream);
        let context = ExecutionContext::new(stream.clone());
        let mut first = DeviceFuture::scheduled(value(()), context.clone());
        let mut second = DeviceFuture::scheduled(
            kernels::copy(dst.partition([4]), src).then(|_| value(())),
            context,
        );
        let first_gate = Gate::arm(&stream);
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        assert_gated_pending(Pin::new(&mut first).poll(&mut cx), &stream, &first_gate);
        let second_gate = Gate::arm(&stream);
        assert_gated_pending(Pin::new(&mut second).poll(&mut cx), &stream, &second_gate);
        drop(first_gate);
        block_on(&mut first).unwrap();
        assert!(Arc::strong_count(&src_owner) > 1);
        assert!(Arc::strong_count(&dst_owner) > 1);
        drop(second_gate);
        block_on(&mut second).unwrap();
        assert_eq!(Arc::strong_count(&src_owner), 1);
        assert_eq!(Arc::strong_count(&dst_owner), 1);
    });
}
