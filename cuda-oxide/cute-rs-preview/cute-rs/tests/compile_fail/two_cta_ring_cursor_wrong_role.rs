/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile-fail probe: reject consumer cursors on producer operations and producer cursors on consumer operations.
use cute_rs::sm100_attention::pipeline::{
    AsyncCompletion, Consumer, Producer, RingCursor, TwoCtaRing,
};

/// Reject consumer cursors on producer operations and producer cursors on consumer operations.

unsafe fn wrong_roles(ring: TwoCtaRing<15, AsyncCompletion>) {
    unsafe {
        ring.acquire_cursor(RingCursor::<15, Consumer>::new());
        ring.wait_cursor(RingCursor::<15, Producer>::new());
        ring.mma_full_cursor(RingCursor::<15, Consumer>::new());
        ring.mma_empty_cursor(RingCursor::<15, Producer>::new());
    }
}
/// Unused host entry point; the device or rejection probe is only type-checked.
fn main() {}
