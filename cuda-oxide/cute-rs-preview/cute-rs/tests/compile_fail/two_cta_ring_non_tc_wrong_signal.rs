/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile-fail probe: reject completion signals unavailable for async-copy reuse and local-shared policies.
use cute_rs::attention::sm100::pipeline::{
    AsyncCompletion, AsyncCopyReuse, Consumer, LocalShared, RingCursor, TwoCtaRing,
};

/// Reject completion signals unavailable for async-copy reuse and local-shared policies.

unsafe fn wrong_non_tc_signals(
    copies: TwoCtaRing<15, AsyncCompletion, AsyncCopyReuse>,
    reuse: TwoCtaRing<2, AsyncCopyReuse>,
    pages: TwoCtaRing<4, LocalShared>,
) {
    unsafe {
        copies.thread_empty(0);
        reuse.producer_expect_tx(0, 16);
        pages.producer_expect_tx(0, 16);
        pages.mma_full(0);
        pages.mma_empty_cursor(RingCursor::<4, Consumer>::new());
        pages.thread_full(0);
    }
}

/// Unused host entry point; the device or rejection probe is only type-checked.

fn main() {}
