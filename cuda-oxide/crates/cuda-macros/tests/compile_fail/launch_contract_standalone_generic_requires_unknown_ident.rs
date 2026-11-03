// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A standalone GENERIC kernel routes #[launch_contract] onto its generated
// entry wrapper (synthetic parameter names), so #[kernel] itself must
// validate `requires` against the source signature: `n` is not a parameter
// of this kernel.

#[cuda_macros::kernel]
#[cuda_macros::launch_contract(
    domain = 1,
    block = (64, 1, 1),
    requires = (input.len() >= n),
)]
pub fn scaled<T: Copy>(input: &[T], mut output: cuda_device::DisjointSlice<T>) {
    let _ = (input, &mut output);
}

fn main() {}
