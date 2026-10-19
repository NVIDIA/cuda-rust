/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: raw intrinsics cannot be invoked through Fn/FnMut/FnOnce.

use cuda_device::{device, kernel};
use cuda_intrinsics::sreg::thread_idx_x;

#[device]
fn invoke<F: Fn() -> u32>(function: F) -> u32 {
    function()
}

#[kernel]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn intrinsic_callable_trait(output: *mut u32) {
    // SAFETY: this example is rejected when the intrinsic is used as a
    // callable value; the pointer keeps that result live in device MIR.
    unsafe {
        output.write(invoke(thread_idx_x));
    }
}

fn main() {
    println!("This example must fail: generated intrinsics require direct calls.");
}
