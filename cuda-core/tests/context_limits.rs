/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `CudaContext::{limit, set_limit}` over every [`ContextLimit`] variant.
//!
//! Its own test binary, for the reason `context_sync_policy.rs` has one: a
//! limit is state on the device's primary context, shared by every handle in
//! the process, so a test that wrote one while another test in the same binary
//! read it would race. One test function walking the variants in sequence
//! keeps the whole check on one thread.
//!
//! `MallocHeapSize` and `PrintfFifoSize` are writable here only because this
//! binary launches no kernel: the driver refuses both once a kernel using
//! `malloc` or `printf` has run in the process.

use cuda_core::{ContextLimit, CudaContext};

/// Every variant, in declaration order.
const ALL: [ContextLimit; 7] = [
    ContextLimit::StackSize,
    ContextLimit::PrintfFifoSize,
    ContextLimit::MallocHeapSize,
    ContextLimit::DevRuntimeSyncDepth,
    ContextLimit::DevRuntimePendingLaunchCount,
    ContextLimit::MaxL2FetchGranularity,
    ContextLimit::PersistingL2CacheSize,
];

/// `CUDA_ERROR_UNSUPPORTED_LIMIT`, the documented answer for a limit this
/// device does not implement (`DevRuntimeSyncDepth` above compute 9.0).
const UNSUPPORTED: cuda_core::sys::CUresult =
    cuda_core::sys::cudaError_enum_CUDA_ERROR_UNSUPPORTED_LIMIT;

#[test]
fn every_limit_reads_and_the_writable_ones_round_trip() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");

    // Reading must either succeed or give the one documented refusal. Any
    // other status means to_raw() handed the driver a value it did not
    // recognise.
    for limit in ALL {
        match ctx.limit(limit) {
            Ok(_) => {}
            Err(e) if e.0 == UNSUPPORTED => {}
            Err(e) => panic!("limit({limit:?}) failed with an unexpected status: {e:?}"),
        }
    }

    // The stack size is the one limit with an exact, driver-honoured
    // round trip, so it is what proves set_limit reaches the right CUlimit
    // rather than silently writing a neighbouring one.
    let original = ctx
        .stack_size()
        .expect("stack_size() must be readable on any device");
    let raised = original + 4096;
    ctx.set_limit(ContextLimit::StackSize, raised)
        .expect("set_limit(StackSize) failed");
    assert_eq!(
        ctx.limit(ContextLimit::StackSize).unwrap(),
        raised,
        "set_limit(StackSize, {raised}) must be observable through limit(StackSize)"
    );

    // The inherent accessors are now thin wrappers; this pins them to the
    // same value rather than to a second, drifting implementation.
    assert_eq!(
        ctx.stack_size().unwrap(),
        ctx.limit(ContextLimit::StackSize).unwrap(),
        "stack_size() must agree with limit(StackSize)"
    );
    ctx.set_stack_size(original)
        .expect("set_stack_size failed to restore the original");
    assert_eq!(
        ctx.stack_size().unwrap(),
        original,
        "set_stack_size() must agree with set_limit(StackSize, ..)"
    );

    // Writing one limit must not disturb its neighbours: a to_raw() that
    // mapped two variants onto one CUlimit would pass every check above.
    let heap_before = ctx.limit(ContextLimit::MallocHeapSize).unwrap();
    let fifo_before = ctx.limit(ContextLimit::PrintfFifoSize).unwrap();
    let heap_target = heap_before + (1 << 20);
    ctx.set_limit(ContextLimit::MallocHeapSize, heap_target)
        .expect("set_limit(MallocHeapSize) failed before any kernel launch");
    assert_eq!(
        ctx.limit(ContextLimit::MallocHeapSize).unwrap(),
        heap_target,
        "the malloc heap must hold the size just written"
    );
    assert_eq!(
        ctx.limit(ContextLimit::PrintfFifoSize).unwrap(),
        fifo_before,
        "writing MallocHeapSize must leave PrintfFifoSize untouched"
    );
    ctx.set_limit(ContextLimit::MallocHeapSize, heap_before)
        .expect("failed to restore the malloc heap size");
}
