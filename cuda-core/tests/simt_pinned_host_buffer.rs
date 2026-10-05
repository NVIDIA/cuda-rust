/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::{CudaContext, DeviceBuffer, PinnedHostBuffer};

#[test]
fn pinned_host_buffer_exposes_initialized_slice() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let mut host =
        PinnedHostBuffer::<u32>::zeroed(&ctx, 4).expect("failed to allocate pinned host");

    assert_eq!(host.len(), 4);
    assert_eq!(host.num_bytes(), 16);
    assert!(!host.is_empty());
    assert_eq!(host.as_slice(), &[0, 0, 0, 0]);
    assert!(format!("{host:?}").contains("PinnedHostBuffer"));

    host.as_mut_slice().copy_from_slice(&[1, 2, 3, 4]);
    assert_eq!(&host[..], &[1, 2, 3, 4]);
}

#[test]
fn pinned_host_buffer_supports_empty_allocations() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let host =
        PinnedHostBuffer::<u32>::zeroed(&ctx, 0).expect("failed to create empty pinned host");

    assert_eq!(host.len(), 0);
    assert_eq!(host.num_bytes(), 0);
    assert!(host.is_empty());
    assert_eq!(host.as_slice(), &[]);
}

#[test]
fn pinned_host_buffer_from_slice_supports_empty_input() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let host = PinnedHostBuffer::<u32>::from_slice(&ctx, &[])
        .expect("failed to create empty pinned host from slice");

    assert_eq!(host.len(), 0);
    assert_eq!(host.num_bytes(), 0);
    assert_eq!(host.as_slice(), &[]);
}

#[test]
fn pinned_host_buffer_supports_zero_sized_types() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let host = PinnedHostBuffer::<()>::zeroed(&ctx, 8)
        .expect("failed to create zero-sized pinned host buffer");

    assert_eq!(host.len(), 8);
    assert_eq!(host.num_bytes(), 0);
    assert_eq!(host.as_slice(), &[(); 8]);
}

#[test]
fn pinned_host_buffer_roundtrips_through_device_buffer() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let input = PinnedHostBuffer::from_slice(&ctx, &[1_u32, 2, 3, 4])
        .expect("failed to allocate pinned input");
    // SAFETY: `input` is kept alive for the entire roundtrip, and
    // `copy_to_pinned_host` synchronizes before returning, so both pinned
    // buffers outlive their in-flight copies.
    let device = unsafe { DeviceBuffer::from_pinned_host(&stream, &input) }
        .expect("failed to copy input to device");
    let mut output = PinnedHostBuffer::<u32>::zeroed(&ctx, input.len())
        .expect("failed to allocate pinned output");

    device
        .copy_to_pinned_host(&stream, &mut output)
        .expect("failed to copy output to host");

    assert_eq!(output.as_slice(), input.as_slice());
}

#[test]
fn pinned_host_buffer_async_copy_can_be_synchronized_later() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let input = PinnedHostBuffer::from_slice(&ctx, &[5_u32, 6, 7, 8])
        .expect("failed to allocate pinned input");
    // SAFETY: both pinned buffers are kept alive past `stream.synchronize()`
    // below, satisfying the contract of the async pinned helpers.
    let device = unsafe { DeviceBuffer::from_pinned_host(&stream, &input) }
        .expect("failed to copy input to device");
    let mut output = PinnedHostBuffer::<u32>::zeroed(&ctx, input.len())
        .expect("failed to allocate pinned output");

    unsafe { device.copy_to_pinned_host_async(&stream, &mut output) }
        .expect("failed to enqueue output copy");
    stream.synchronize().expect("failed to synchronize stream");

    assert_eq!(output.as_slice(), input.as_slice());
}

#[test]
fn device_buffer_can_be_refilled_from_pinned_host_async() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let mut device =
        DeviceBuffer::<u32>::zeroed(&stream, 4).expect("failed to allocate device buffer");
    let mut output =
        PinnedHostBuffer::<u32>::zeroed(&ctx, 4).expect("failed to allocate pinned output buffer");

    for round in 0..3 {
        let payload = [round * 10, round * 10 + 1, round * 10 + 2, round * 10 + 3];
        let input =
            PinnedHostBuffer::from_slice(&ctx, &payload).expect("failed to allocate pinned input");

        // SAFETY: `input` lives until the end of this scope and the
        // `stream.synchronize()` below completes the in-flight refill before
        // `input` is dropped. `output` outlives the synchronize call too.
        unsafe { device.copy_from_pinned_host_async(&stream, &input) }
            .expect("failed to enqueue refill");
        unsafe { device.copy_to_pinned_host_async(&stream, &mut output) }
            .expect("failed to enqueue readback");
        stream.synchronize().expect("failed to synchronize stream");

        assert_eq!(output.as_slice(), &payload);
    }
}
