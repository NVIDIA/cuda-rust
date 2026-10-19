/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `CudaContext::{stream_priority_range, new_stream_with_priority}` and
//! `CudaStream::priority`.
//!
//! Stream priority is per-stream state rather than the process-wide context
//! state `context_sync_policy.rs` has to serialise, so these run in one
//! binary without a race. What they do share is the device's range, which is
//! read-only.

use cuda_core::CudaContext;

#[test]
fn range_is_ordered_the_way_cuda_orders_it() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let range = ctx
        .stream_priority_range()
        .expect("stream_priority_range() failed");

    assert!(
        range.greatest() <= range.least(),
        "the greatest priority is the numerically smaller value, got greatest={} least={}",
        range.greatest(),
        range.least()
    );
    assert!(
        range.contains(range.least()) && range.contains(range.greatest()),
        "both ends must lie inside the range they define"
    );
    assert_eq!(
        range.is_supported(),
        range.least() != range.greatest(),
        "is_supported() must read the both-ends-zero answer the driver gives \
         on a device without priority support"
    );
}

#[test]
fn a_stream_reports_the_priority_it_was_given() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let range = ctx
        .stream_priority_range()
        .expect("stream_priority_range() failed");

    for requested in [range.least(), range.greatest()] {
        let stream = ctx
            .new_stream_with_priority(requested)
            .unwrap_or_else(|e| panic!("new_stream_with_priority({requested}) failed: {e:?}"));
        assert_eq!(
            stream.priority().unwrap(),
            requested,
            "a priority inside the device's range must survive stream creation"
        );
    }
}

#[test]
fn an_out_of_range_priority_is_clamped_rather_than_refused() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let range = ctx
        .stream_priority_range()
        .expect("stream_priority_range() failed");

    // The driver clamps and reports nothing, so the check is that creation
    // succeeds AND that the stream lands on the end of the range clamp()
    // predicts. Asserting only that creation succeeds would pass against a
    // clamp() that returned any value at all.
    for requested in [i32::MIN, i32::MAX] {
        let expected = range.clamp(requested);
        assert!(
            range.contains(expected),
            "clamp({requested}) must land inside the range"
        );
        let stream = ctx
            .new_stream_with_priority(requested)
            .unwrap_or_else(|e| panic!("new_stream_with_priority({requested}) failed: {e:?}"));
        assert_eq!(
            stream.priority().unwrap(),
            expected,
            "an out-of-range request must land where clamp({requested}) says it will"
        );
    }
}

#[test]
fn an_ordinary_stream_reports_the_default_priority() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");

    assert_eq!(
        ctx.new_stream().unwrap().priority().unwrap(),
        0,
        "new_stream() must keep the driver's default priority"
    );
    assert_eq!(
        ctx.default_stream().priority().unwrap(),
        0,
        "the default stream must report the default priority"
    );
}
