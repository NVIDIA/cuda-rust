// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_device::{kernel, launch_contract, thread};

#[kernel(launch_context = launch_context)]
#[launch_contract(domain = 1, coordinates = u32, block = (64, 1, 1))]
pub fn fast_helper<const N: u32>() {
    let _ = thread::index_1d_u32(launch_context);
    let _ = N;
}

#[kernel]
#[launch_contract(domain = 2, coordinates = u32, block = (8, 8, 1))]
pub fn wrong_entry() {
    // A generic kernel implementation cannot mint a fresh launch context.
    // Its generated entry owns the hidden proof parameter.
    fast_helper::<4>();
}

fn main() {}
