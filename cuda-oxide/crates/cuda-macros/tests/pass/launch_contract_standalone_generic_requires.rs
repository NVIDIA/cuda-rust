// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Well-formed `requires` relations on a standalone GENERIC kernel must keep
// compiling: #[kernel] validates them against the source signature before
// routing #[launch_contract] onto the generated entry wrapper.

#[cuda_macros::kernel]
#[cuda_macros::launch_contract(
    domain = 1,
    block = (64, 1, 1),
    requires = (input.len() >= n, output.len() >= n),
)]
pub fn scaled<T: Copy>(n: u32, input: &[T], mut output: cuda_device::DisjointSlice<T>) {
    let _ = (n, input, &mut output);
}

fn main() {}
