/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: a raw crate from a newer intrinsic ABI must fail closed.

use cuda_device::kernel;
use cuda_intrinsics::__cuda_oxide_intrinsic_abi_v2::i0001;

#[kernel]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn unsupported_intrinsic_abi(output: *mut u32) {
    // SAFETY: this example is rejected by intrinsic-ABI validation before it
    // can run; the pointer keeps the invalid result live in device MIR.
    unsafe {
        output.write(i0001());
    }
}

fn main() {
    println!("This example must fail: cuda-oxide supports intrinsic ABI v1, not v2.");
}
