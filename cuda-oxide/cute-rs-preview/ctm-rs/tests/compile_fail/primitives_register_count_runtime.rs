/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Compile-fail probe: a register count is an instruction immediate.
use ctm_rs::thread::setmaxnreg_inc;
/// Reject a runtime register count.
fn wrong(count: u32) {
    unsafe { setmaxnreg_inc::<{ count }>() };
}
/// Unused host entry point.
fn main() {}
