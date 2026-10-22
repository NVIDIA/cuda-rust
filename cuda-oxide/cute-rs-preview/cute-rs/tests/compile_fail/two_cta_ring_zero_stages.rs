/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile-fail probe: reject a zero-stage ring during const evaluation.
use cute_rs::attention::sm100::pipeline::{AsyncCompletion, TwoCtaRing};

/// Reject a zero-stage ring during const evaluation.

const _: TwoCtaRing<0, AsyncCompletion> = unsafe { TwoCtaRing::from_raw(core::ptr::null_mut(), 0) };

/// Unused host entry point; the device or rejection probe is only type-checked.

fn main() {}
