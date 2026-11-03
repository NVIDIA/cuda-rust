/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Compile-fail probe: register counts must be multiples of eight.
use ctm_rs::thread::setmaxnreg_dec;
/// Trigger validation of a count that is in range but not aligned.
fn main() {
    unsafe { setmaxnreg_dec::<25>() };
}
