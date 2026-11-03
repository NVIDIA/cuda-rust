/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! A launch-uniform witness has no safe constructor. Without this, a kernel
//! could wrap a per-thread value and present it to any API whose argument has
//! to be the same in every thread of the launch.

use cuda_device::Uniform;

fn main() {
    let _from_struct_literal = Uniform::<u32> { value: 7 };
}
