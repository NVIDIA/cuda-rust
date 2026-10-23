// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![no_std]

trait Recipe<const ITEMS: usize> {
    const REGISTRATION: &'static str;
}

struct LoadAddRecipe;

impl<const ITEMS: usize> Recipe<ITEMS> for LoadAddRecipe {
    const REGISTRATION: &'static str = cuda_device::library::registration!("FIXTURE_REGISTRATION");
}

fn registration<R: Recipe<ITEMS>, const ITEMS: usize>() -> &'static str {
    R::REGISTRATION
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
            registration::<LoadAddRecipe, ITEMS>,
            (input.add(origin), values.as_mut_ptr(), 7_i32),
        );
        for (item, value) in values.iter().enumerate() {
            output.add(origin + lane * ITEMS + item).write(*value);
        }
    }
}

fn payload_copy() -> &'static str {
    cuda_device::library::registration!("PAYLOAD_COPY_REGISTRATION")
}

fn payload_add() -> &'static str {
    cuda_device::library::registration!("PAYLOAD_ADD_REGISTRATION")
}

/// Produce fresh local results through checked Block and Warp native calls.
///
/// # Safety
///
/// Launch exactly three 128-thread one-dimensional blocks. Output must cover
/// their complete tiles, with no conflicting access during the launch.
#[inline(never)]
pub unsafe fn payload_roundtrip(output: *mut i32) {
    let lane = cuda_device::thread::threadIdx_x() as usize;
    let index = cuda_device::thread::blockIdx_x() as usize * 128 + lane;
    let input = index as i32;
    let mut block_result = 0_i32;
    let mut warp_result = 0_i32;
    // SAFETY: each pointer refers to a separate live per-thread scalar; both
    // recipes initialize their output, preserve the input and use no scratch.
    unsafe {
        cuda_device::library::checked_block_payload_call_v1::<i32, _, _, _, _, 1>(
            payload_copy,
            (),
            (),
            (&input as *const i32, &mut block_result as *mut i32),
        );
        let seed = cuda_device::library::warp_group_index_v1::<8>() as i32;
        cuda_device::library::checked_warp_payload_call_v1::<i32, _, _, _, 8, 1>(
            payload_add,
            (seed,),
            (&block_result as *const i32, &mut warp_result as *mut i32),
        );
        output.add(index).write(warp_result);
    }
}
