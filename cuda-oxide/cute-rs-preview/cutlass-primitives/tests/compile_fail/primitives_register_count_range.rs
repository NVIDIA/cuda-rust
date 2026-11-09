/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Compile-fail probe: register counts may not exceed 256.
use cutlass_primitives::thread::setmaxnreg_inc;
/// Trigger validation of an aligned count beyond the instruction range.
fn main() {
    unsafe { setmaxnreg_inc::<264>() };
}
