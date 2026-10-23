/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile-pass probe: type-check producer and consumer cursors with matching ring policies and stage counts.
use cute_rs::attention::sm100::pipeline::{
    AsyncCompletion, Consumer, Producer, RingCursor, SharedReuse, SharedToAsync, TensorMemory,
    TwoCtaRing,
};

/// Type-check producer and consumer cursors with matching ring policies and stage counts. This device-only probe is never executed.

unsafe fn valid_cursors(
    copies: TwoCtaRing<15, AsyncCompletion>,
    tensor: TwoCtaRing<2, TensorMemory>,
    shared: TwoCtaRing<2, SharedToAsync, SharedReuse>,
) {
    let producer = RingCursor::<15, Producer>::new();
    let consumer = RingCursor::<15, Consumer>::new();
    let tensor_producer = RingCursor::<2, Producer>::new();
    let tensor_consumer = RingCursor::<2, Consumer>::new();
    unsafe {
        copies.acquire_cursor(producer);
        copies.producer_expect_tx_cursor(producer, 16);
        copies.mma_full_cursor(producer);
        copies.wait_cursor(consumer);
        copies.mma_empty_cursor(consumer);
        tensor.thread_full_cursor(tensor_producer);
        tensor.thread_empty_cursor(tensor_consumer);
        shared.thread_full_cursor(tensor_producer);
        shared.mma_empty_cursor(tensor_consumer);
    }
}
/// Unused host entry point; the device or rejection probe is only type-checked.
fn main() {}
