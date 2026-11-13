/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `memory::memcpy_batch_async` copies between device and pinned host memory in
//! one driver call, and rejects mismatched slices before calling the driver.

use cuda_core::memory::{self, MemcpySrcAccessOrder};
use cuda_core::{CudaContext, DriverError, PinnedHostBuffer};

const PIECES: usize = 64;
const PIECE: usize = 4096;

#[test]
fn a_batch_scatters_and_gathers_pages() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create stream");
    let bytes = PIECES * PIECE;
    let host_in = PinnedHostBuffer::from_slice(
        &ctx,
        &(0..bytes).map(|i| (i % 251) as u8).collect::<Vec<u8>>(),
    )
    .expect("pinned source");
    let mut host_out = PinnedHostBuffer::<u8>::zeroed(&ctx, bytes).expect("pinned destination");
    let dev = unsafe { memory::malloc_sync(bytes) }.expect("device buffer");

    // Host -> device with the pieces reversed, then device -> host in order:
    // host_out holds host_in's pieces in reverse order.
    let host_at = |base: *const u8, i: usize| unsafe { base.add(i * PIECE) } as u64;
    let dev_at = |i: usize| dev + (i * PIECE) as u64;
    let sizes = [PIECE; PIECES];
    let up_dst: Vec<u64> = (0..PIECES).map(|i| dev_at(PIECES - 1 - i)).collect();
    let up_src: Vec<u64> = (0..PIECES).map(|i| host_at(host_in.as_ptr(), i)).collect();
    let down_dst: Vec<u64> = (0..PIECES)
        .map(|i| host_at(host_out.as_mut_ptr(), i))
        .collect();
    let down_src: Vec<u64> = (0..PIECES).map(dev_at).collect();
    unsafe {
        memory::memcpy_batch_async(
            &up_dst,
            &up_src,
            &sizes,
            MemcpySrcAccessOrder::Stream,
            stream.cu_stream(),
        )
        .expect("host -> device batch");
        memory::memcpy_batch_async(
            &down_dst,
            &down_src,
            &sizes,
            MemcpySrcAccessOrder::Stream,
            stream.cu_stream(),
        )
        .expect("device -> host batch");
    }
    stream.synchronize().expect("synchronize");
    for i in 0..PIECES {
        let j = PIECES - 1 - i;
        assert_eq!(
            host_out[i * PIECE..(i + 1) * PIECE],
            host_in[j * PIECE..(j + 1) * PIECE],
            "piece {i}"
        );
    }
    unsafe { memory::free_sync(dev) }.expect("free");
}

#[test]
fn an_empty_batch_enqueues_nothing() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create stream");
    unsafe {
        memory::memcpy_batch_async(&[], &[], &[], MemcpySrcAccessOrder::Any, stream.cu_stream())
    }
    .expect("an empty batch succeeds");
}

#[test]
fn mismatched_slices_are_rejected_before_the_driver() {
    // No context and no stream: the length check comes first.
    let r = unsafe {
        memory::memcpy_batch_async(
            &[0, 0],
            &[0],
            &[1, 1],
            MemcpySrcAccessOrder::Stream,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(
        r,
        Err(DriverError(
            cuda_core::sys::cudaError_enum_CUDA_ERROR_INVALID_VALUE
        ))
    );
}
