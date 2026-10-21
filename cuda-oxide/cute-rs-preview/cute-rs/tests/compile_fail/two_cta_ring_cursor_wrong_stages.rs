/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile-fail probe: reject cursors whose stage count differs from the ring.
use cute_rs::sm100_attention::pipeline::{
    AsyncCompletion, Consumer, Producer, RingCursor, TwoCtaRing,
};

/// Reject cursors whose stage count differs from the ring.

unsafe fn wrong_stages(ring: TwoCtaRing<15, AsyncCompletion>) {
    unsafe {
        ring.acquire_cursor(RingCursor::<2, Producer>::new());
        ring.wait_cursor(RingCursor::<2, Consumer>::new());
    }
}
/// Unused host entry point; the device or rejection probe is only type-checked.
fn main() {}
