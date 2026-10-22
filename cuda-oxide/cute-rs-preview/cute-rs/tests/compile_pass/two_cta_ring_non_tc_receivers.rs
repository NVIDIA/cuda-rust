/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile-pass probe: type-check raw and cursor operations supported by async-copy and local-shared receivers.
use cute_rs::attention::sm100::pipeline::{
    AsyncCompletion, AsyncCopyReuse, Consumer, LocalShared, Producer, RingCursor, TwoCtaRing,
};

/// Type-check raw and cursor operations supported by async-copy and local-shared receivers. This device-only probe is never executed.

unsafe fn non_tc_receivers(
    copies: TwoCtaRing<15, AsyncCompletion, AsyncCopyReuse>,
    pages: TwoCtaRing<4, LocalShared>,
) {
    let producer = RingCursor::<15, Producer>::new();
    let consumer = RingCursor::<15, Consumer>::new();
    let page_producer = RingCursor::<4, Producer>::new();
    let page_consumer = RingCursor::<4, Consumer>::new();
    unsafe {
        copies.acquire(0);
        copies.producer_expect_tx(0, 16);
        copies.wait(0);
        copies.mma_empty(0);
        copies.acquire_cursor(producer);
        copies.producer_expect_tx_cursor(producer, 16);
        copies.wait_cursor(consumer);
        copies.mma_empty_cursor(consumer);
        pages.acquire(0);
        pages.wait(0);
        pages.acquire_cursor(page_producer);
        pages.wait_cursor(page_consumer);
        let _ = pages.full_at_cursor(page_producer);
        let _ = pages.empty_at_cursor(page_consumer);
    }
}

/// Unused host entry point; the device or rejection probe is only type-checked.

fn main() {}
