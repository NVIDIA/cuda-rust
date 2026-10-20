// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![no_std]

fn registration() -> &'static str {
    cuda_device::library::registration!("FIXTURE_REGISTRATION")
}

/// Load a complete block tile through a native provider in a dependency crate.
///
/// # Safety
///
/// Launch 128-thread one-dimensional blocks. Input/output must cover every
/// block's complete tile; input stays immutable and output tiles are disjoint.
#[inline(never)]
pub unsafe fn load_add<const ITEMS: usize>(input: *const i32, output: *mut i32) {
    let lane = cuda_device::thread::threadIdx_x() as usize;
    let origin = cuda_device::thread::blockIdx_x() as usize * 128 * ITEMS;
    let mut values = [0_i32; ITEMS];
    // SAFETY: each thread owns its initialized local array, and the caller
    // supplies the full input tile and required geometry. This recipe uses no scratch.
    unsafe {
        cuda_device::library::call_v1::<i32, _, _, 128, ITEMS>(
            registration,
            (input.add(origin), values.as_mut_ptr(), 7_i32),
        );
        for (item, value) in values.iter().enumerate() {
            output.add(origin + lane * ITEMS + item).write(*value);
        }
    }
}
