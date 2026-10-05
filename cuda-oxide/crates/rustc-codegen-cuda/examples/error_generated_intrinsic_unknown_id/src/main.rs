/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: unknown IDs in the supported ABI must fail closed.

use cuda_device::kernel;
use cuda_intrinsics::__cuda_oxide_intrinsic_abi_v1::i9999;

#[kernel]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn unknown_intrinsic_id(output: *mut u32) {
    // SAFETY: this example is rejected by intrinsic-ID validation before it
    // can run; the pointer keeps the invalid result live in device MIR.
    unsafe {
        output.write(i9999());
    }
}

fn main() {
    println!("This example must fail: i9999 is not assigned in intrinsic ABI v1.");
}
