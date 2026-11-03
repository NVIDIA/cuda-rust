/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Compile-fail probe: require the cp.async wait count to be a compile-time immediate.
use ctm_rs::async_copy::cp_async_wait_group;

/// Require the cp.async wait count to be a compile-time immediate.

fn wrong(pending: u32) {
    unsafe { cp_async_wait_group::<{ pending }>() };
}

/// Unused host entry point; the device or rejection probe is only type-checked.

fn main() {}
