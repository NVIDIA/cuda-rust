/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: raw intrinsics cannot be converted to function pointers.

use cuda_device::kernel;
use cuda_intrinsics::sreg::thread_idx_x;

#[kernel]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn intrinsic_function_pointer(output: *mut u32) {
    let read_index: fn() -> u32 = thread_idx_x;
    // SAFETY: this example is rejected when the function item is reified;
    // the pointer merely keeps the indirect result live in device MIR.
    unsafe {
        output.write(read_index());
    }
}

fn main() {
    println!("This example must fail: generated intrinsics require direct calls.");
}
