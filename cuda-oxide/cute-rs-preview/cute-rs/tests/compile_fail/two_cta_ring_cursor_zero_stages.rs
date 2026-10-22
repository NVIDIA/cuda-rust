/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile-fail probe: reject a zero-stage cursor during const evaluation.
use cute_rs::attention::sm100::pipeline::{Producer, RingCursor};

/// Reject a zero-stage cursor during const evaluation.

const _: RingCursor<0, Producer> = RingCursor::new();
/// Unused host entry point; the device or rejection probe is only type-checked.
fn main() {}
