/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Leak regression probes for `DeviceBuffer` construction (require a GPU).
//!
//! `DeviceBuffer::from_host` and `DeviceBuffer::zeroed` allocate device
//! memory synchronously and then enqueue a fallible async operation. Two
//! resources must be accounted for across that window and the buffer's
//! whole lifecycle:
//!
//! - the device allocation itself (a leaked `CUdeviceptr` bleeds VRAM), and
//! - the `Arc<CudaContext>` strong count (a leaked count means
//!   `CudaContext::drop` / `cuCtxDestroy` never runs).
//!
//! These tests pin both to their baselines after construct/drop cycles, and
//! pin the zero-length construction path, which must not call the driver
//! allocator at all (`cuMemAlloc` rejects zero-byte requests).

use std::sync::Arc;

use cuda_core::{CudaContext, DeviceBuffer};

/// A live buffer must hold exactly one extra context strong count, and the
/// count must return to baseline once the buffer drops. A construction
/// helper that `mem::forget`s a context clone (as an arm/disarm pointer
/// guard would) fails the live-count assertion by one per construction.
#[test]
fn ctx_strong_count_returns_to_baseline_after_buffer_lifecycle() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let baseline = Arc::strong_count(&ctx);

    {
        let buf = DeviceBuffer::from_host(&stream, &[1u32, 2, 3, 4])
            .expect("from_host failed on happy path");
        assert_eq!(
            Arc::strong_count(&ctx),
            baseline + 1,
            "live DeviceBuffer must add exactly one ctx strong count"
        );
        drop(buf);
    }

    assert_eq!(
        Arc::strong_count(&ctx),
        baseline,
        "ctx strong count must return to baseline after from_host buffer drop"
    );

    {
        let buf = DeviceBuffer::<f32>::zeroed(&stream, 16).expect("zeroed failed on happy path");
        drop(buf);
    }

    assert_eq!(
        Arc::strong_count(&ctx),
        baseline,
        "ctx strong count must return to baseline after zeroed buffer drop"
    );
}

/// Repeated construct/drop cycles must not bleed device memory.
///
/// This cannot deterministically force the async-enqueue failure that
/// triggered the original leak (no public API constructs an invalid
/// `CudaStream`), but it pins the happy-path accounting with
/// `cuMemGetInfo` before and after the cycles.
#[test]
fn vram_returns_to_baseline_after_buffer_cycles() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    fn free_mem() -> usize {
        let mut free = 0usize;
        let mut total = 0usize;
        let rc = unsafe { cuda_bindings::cuMemGetInfo_v2(&mut free, &mut total) };
        assert_eq!(rc, 0, "cuMemGetInfo failed: {rc}");
        free
    }

    // Warm up driver allocator caches so the measured window is stable.
    for _ in 0..4 {
        let b = DeviceBuffer::<u8>::zeroed(&stream, 1 << 20).expect("warmup alloc failed");
        drop(b);
    }
    ctx.synchronize().expect("sync failed");

    let before = free_mem();
    for _ in 0..64 {
        let b = DeviceBuffer::<u8>::zeroed(&stream, 1 << 20).expect("cycle alloc failed");
        drop(b);
    }
    ctx.synchronize().expect("sync failed");
    let after = free_mem();

    // Allow small driver-side jitter; 64 leaked 1 MiB buffers would show
    // up loudly.
    assert!(
        before.abs_diff(after) < (8 << 20),
        "device free memory drifted by {} bytes across 64 construct/drop cycles",
        before.abs_diff(after)
    );
}

/// Zero-length construction must succeed for both constructors and must
/// not leak anything on drop. `cuMemAlloc` rejects zero-byte requests with
/// CUDA_ERROR_INVALID_VALUE, so empty buffers are represented without a
/// driver allocation.
#[test]
fn zero_length_construction_succeeds_for_both_constructors() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let baseline = Arc::strong_count(&ctx);

    let zeroed = DeviceBuffer::<u8>::zeroed(&stream, 0).expect("zeroed(len=0) must succeed");
    assert_eq!(zeroed.len(), 0);
    assert_eq!(zeroed.num_bytes(), 0);
    assert!(zeroed.is_empty());

    let from_host =
        DeviceBuffer::<u32>::from_host(&stream, &[]).expect("from_host(empty) must succeed");
    assert_eq!(from_host.len(), 0);
    assert_eq!(from_host.num_bytes(), 0);
    assert!(from_host.is_empty());

    drop(zeroed);
    drop(from_host);
    assert_eq!(
        Arc::strong_count(&ctx),
        baseline,
        "empty buffers must not leak a ctx strong count"
    );
}
