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

// CUDA's page alignment alone does not satisfy every valid Rust element layout.
#[repr(C, align(1048576))]
#[derive(Clone, Copy, cuda_core::DeviceCopy)]
struct Overaligned([u8; 1048576]);

#[test]
fn pinned_host_buffer_aligns_elements() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    // Keep a small allocation alive to exercise non-MiB-aligned CUDA allocations.
    let _prefix = PinnedHostBuffer::<u8>::zeroed(&ctx, 4096).unwrap();
    let mut buffers = Vec::new();
    for _ in 0..8 {
        let mut host = PinnedHostBuffer::<Overaligned>::zeroed(&ctx, 1).unwrap();
        // Check before forming any references, so the regression fails without UB.
        assert!(host.as_ptr().is_aligned());
        assert_eq!(host.num_bytes(), std::mem::size_of::<Overaligned>());
        for element in host.as_slice() {
            assert!(element.0.iter().all(|&byte| byte == 0));
        }
        host.as_mut_slice()[0].0.fill(42);
        assert!(host.as_slice()[0].0.iter().all(|&byte| byte == 42));
        buffers.push(host);
    }
    drop(buffers);
    ctx.check_err().expect("failed to free pinned allocations");
}

#[test]
fn pinned_host_buffer_copies_overaligned_elements() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    // Initialize on the heap to avoid putting MiB-sized values on the stack.
    let mut source = PinnedHostBuffer::<Overaligned>::zeroed(&ctx, 2).unwrap();
    source.as_mut_slice()[0].0.fill(17);
    source.as_mut_slice()[1].0.fill(42);
    let copied = PinnedHostBuffer::from_slice(&ctx, source.as_slice()).unwrap();
    assert!(copied.as_ptr().is_aligned());
    assert_eq!(copied.len(), 2);
    assert_eq!(copied.num_bytes(), 2 * std::mem::size_of::<Overaligned>());
    assert!(copied.as_slice()[0].0.iter().all(|&byte| byte == 17));
    assert!(copied.as_slice()[1].0.iter().all(|&byte| byte == 42));
    drop((copied, source));
    ctx.check_err().expect("failed to free pinned allocations");
}

#[repr(align(1048576))]
#[derive(Clone, Copy, cuda_core::DeviceCopy)]
struct AlignedZst;

#[test]
fn empty_and_zero_sized_pinned_buffers_preserve_alignment() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let empty = PinnedHostBuffer::<Overaligned>::zeroed(&ctx, 0).unwrap();
    assert!(empty.as_ptr().is_aligned());
    assert!(empty.as_slice().is_empty());
    let mut zst = PinnedHostBuffer::<AlignedZst>::zeroed(&ctx, 8).unwrap();
    assert!(zst.as_ptr().is_aligned());
    assert_eq!(zst.num_bytes(), 0);
    assert_eq!(zst.as_mut_slice().len(), 8);
    let copied = PinnedHostBuffer::from_slice(&ctx, zst.as_slice()).unwrap();
    assert!(copied.as_ptr().is_aligned());
    assert_eq!(copied.as_slice().len(), 8);
}
